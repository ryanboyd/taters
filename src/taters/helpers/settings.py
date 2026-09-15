"""
Persistent user settings, and the one that matters today: where downloaded
models are kept.

Taters has a home folder (``$TATERS_HOME`` or ``~/.taters``) for the
library; this module puts a small ``settings.json`` beside it for choices
that should outlive a session and are not about any one pipeline. The first
such choice is the model cache: every transformer, sentence-transformers
and Whisper model is downloaded by the Hugging Face hub library into its
cache, which lives under the user's home by default. On a shared server
that is the wrong place -- one copy per user of a 500 MB encoder, on a home
partition that is small on purpose -- and the fix has been an environment
variable that has to be set in every shell. Here it is a setting, chosen
once in Settings and applied every time Taters starts, before any of those
libraries is imported.

Resolution order for the model cache, most explicit first:

1. ``TATERS_MODEL_CACHE`` in the environment (an administrator's or a
   test's word, and the one thing that beats a saved setting);
2. ``model_cache`` in ``settings.json`` (the user's choice in Settings);
3. ``HF_HUB_CACHE`` or ``HF_HOME`` from the environment (the hub library's
   own conventions, honored as they always were);
4. the hub library's default, ``~/.cache/huggingface/hub``.

Whichever wins is exported as ``HF_HUB_CACHE`` at import (see
:func:`apply_model_cache`), so the hub library, transformers,
sentence-transformers and faster-whisper all download to and read from the
same place -- and it is passed explicitly to the transformer steps too, in
case those libraries were imported before Taters was.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

__all__ = ["settings_path", "load_settings", "save_setting", "clear_setting",
           "model_cache_dir", "model_cache_source", "apply_model_cache",
           "MODEL_CACHE_ENV", "MODEL_CACHE_KEY"]

PathLike = Union[str, Path]

MODEL_CACHE_ENV = "TATERS_MODEL_CACHE"
MODEL_CACHE_KEY = "model_cache"


def _home() -> Path:
    base = os.environ.get("TATERS_HOME")
    return Path(base) if base else Path.home() / ".taters"


def settings_path() -> Path:
    """``$TATERS_HOME/settings.json`` (or ``~/.taters/settings.json``)."""
    return _home() / "settings.json"


def load_settings() -> Dict[str, Any]:
    """The saved settings, or ``{}``; a damaged file reads as empty rather
    than stopping Taters from starting."""
    path = settings_path()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def save_setting(key: str, value: Any) -> Path:
    """Write one setting, keeping the others, atomically."""
    from .atomic import atomic_write

    doc = load_settings()
    doc[str(key)] = value
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return path


def clear_setting(key: str) -> None:
    """Forget one setting; nothing happens when it was not set."""
    from .atomic import atomic_write

    doc = load_settings()
    if str(key) not in doc:
        return
    del doc[str(key)]
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


def _hub_default() -> Path:
    """The hub library's own idea of its cache, from the environment it
    reads (``HF_HUB_CACHE``, else ``HF_HOME/hub``), else the usual default.
    Computed here rather than imported, so a setting can be applied before
    the hub library is ever imported."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.path.expandvars(os.environ["HF_HUB_CACHE"])).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.path.expandvars(os.environ["HF_HOME"])).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_source() -> Tuple[Path, str]:
    """
    The model cache and which rule chose it: ``"environment"``
    (``TATERS_MODEL_CACHE``), ``"setting"`` (chosen in Settings),
    ``"hub environment"`` (``HF_HUB_CACHE``/``HF_HOME``) or ``"default"``.
    """
    override = os.environ.get(MODEL_CACHE_ENV)
    if override:
        return Path(os.path.expandvars(override)).expanduser(), "environment"
    saved = load_settings().get(MODEL_CACHE_KEY)
    if saved:
        return Path(os.path.expandvars(str(saved))).expanduser(), "setting"
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return _hub_default(), "hub environment"
    return _hub_default(), "default"


def model_cache_dir() -> Path:
    """Where downloaded models live; see the module docstring for the order."""
    return model_cache_source()[0]


def apply_model_cache() -> Optional[Path]:
    """
    Export the chosen cache as ``HF_HUB_CACHE`` so every downloading library
    agrees with it. Called when Taters is imported. Returns the path when a
    Taters-level choice (environment or setting) was applied, else None --
    the hub library's own environment is left exactly as found.
    """
    path, source = model_cache_source()
    if source not in ("environment", "setting"):
        return None
    os.environ["HF_HUB_CACHE"] = str(path)
    # older transformers releases read this name; it's harmless on newer ones.
    os.environ.setdefault("TRANSFORMERS_CACHE", str(path))
    return path


def describe_model_cache() -> str:
    """One line for the setup check: the folder, how it was chosen, and
    how much is in it."""
    path, source = model_cache_source()
    how = {"environment": f"from {MODEL_CACHE_ENV}",
           "setting": "chosen in Settings",
           "hub environment": "from HF_HUB_CACHE / HF_HOME",
           "default": "the default"}[source]
    if not path.is_dir():
        return f"{path}  ·  {how}  ·  nothing downloaded yet"
    models = [p for p in path.glob("models--*") if p.is_dir()]
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    from .library import _human_size

    return f"{path}  ·  {how}  ·  {len(models)} model(s), {_human_size(size)}"
