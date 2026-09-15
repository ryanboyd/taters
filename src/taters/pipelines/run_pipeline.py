#!/usr/bin/env python3
"""
Taters Pipeline Runner (robust templating + flexible call resolution)

- ITEM steps run once per input (fan-out concurrently).
- GLOBAL steps run once (barrier before/after).
- Templating preserves native types when the *entire* value is a single template
  (e.g., {{var:text_cols}} → list, not "['text']").
- Calls:
    * "potato.*"      → call via a Taters() instance (e.g., potato.text.analyze_with_dictionaries)
    * dotted path → import and call any function (e.g., taters.helpers.feature_gather.aggregate_features)

Usage example:
  python -m taters.pipelines.run_pipeline \
    --root_dir videos \
    --file_type video \
    --preset conversation_video \
    --workers 4 \
    --var device=cuda --var overwrite_existing=true
"""

from __future__ import annotations

import functools
import argparse
import concurrent.futures as cf
import importlib
import inspect
import json
import os
import re
import sys
import textwrap
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# cheap import: the facade only pulls in the heavy stuff when a method gets called.
from taters import Taters
from ..helpers.gpu import worker_cap

_BUILTIN_PRESETS_DIR = Path(__file__).parent / "presets"

def _get_preset_dirs(root: Optional[Path] = None) -> List[Path]:
    """
    Where to look for presets, in order.

    The third source is the wizard's doing: it gives each pipeline a folder of
    its own -- ``./readability_test/readability_test.yaml`` alongside that run's
    ``features/`` and manifest -- so a project folder holds one tidy directory
    per pipeline instead of everything's output piled together.

    The rule for recognizing one is deliberately narrow: a directory containing
    a YAML file of the *same name*. Anything looser would sweep up every stray
    ``config.yaml`` in the working folder and offer it as a pipeline.
    """
    # `root` is where the interactive app was pointed (`taters --dir X`); the
    # CLI's default stays the working directory. the two used to disagree: the
    # wizard *saved* under `--dir` while this scanned `Path.cwd()`, so a
    # just-saved pipeline showed up in Manage but Run could never find it.
    root = Path(root) if root is not None else Path.cwd()
    dirs = [_BUILTIN_PRESETS_DIR, root / "pipelines"]
    try:
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / f"{child.name}.yaml").is_file() or (child / f"{child.name}.yml").is_file():
                dirs.append(child)
    except OSError:
        pass
    return dirs

# ============ List/Describe Presets ============

def _iter_presets(root: Optional[Path] = None):
    seen: set[Path] = set()
    for base in _get_preset_dirs(root):
        if not base.is_dir():
            continue
        # a pipeline folder only contributes its own file. if we globbed it
        # we'd also pick up whatever YAML the run happens to write inside.
        own = base / f"{base.name}.yaml"
        patterns = ("**/*.yaml", "**/*.yml") if not own.is_file() else (own.name,)
        for pattern in patterns:
            for p in sorted(base.glob(pattern)):
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield p

def available_presets(root: Optional[Path] = None) -> List[Tuple[Path, dict]]:
    """
    Every preset the runner can see, with its metadata.

    Both search directories are covered -- the ones that ship with Taters and
    the ones in ``./pipelines/`` -- which is what lets a UI offer the built-in
    pipelines alongside anything the user has built. Sorted by title so the
    list is stable between runs.

    Returns
    -------
    list[tuple[Path, dict]]
        ``(path, meta)`` pairs. ``meta`` always has ``id``, ``title``,
        ``summary`` and ``tags``, defaulted from the filename when the file
        does not declare them.
    """
    out = [(path, _load_preset_meta(path)) for path in _iter_presets(root)]
    return sorted(out, key=lambda pair: str(pair[1].get("title", "")).lower())


@functools.lru_cache(maxsize=1)
def _builtin_presets_root() -> Path:
    """The shipped-presets directory, resolved once.

    `resolve()` is a filesystem call, and this one was being made on every
    single `is_builtin_preset` check -- which the management screens do once
    per preset, per repaint. The directory ships inside the installed package
    and cannot move while the process runs, so once is enough."""
    return _BUILTIN_PRESETS_DIR.resolve()


def is_builtin_preset(path: Path) -> bool:
    """Whether a preset ships with Taters, and so must not be edited or deleted."""
    try:
        Path(path).resolve().relative_to(_builtin_presets_root())
        return True
    except ValueError:
        return False


@functools.lru_cache(maxsize=512)
def _preset_meta_uncached(path_str: str, _mtime_ns: int, _size: int) -> dict:
    """Parse one preset's `meta:` block. Keyed on the file's mtime and size as
    well as its path, so an edited preset is re-read and a cache hit can only
    describe the bytes that are actually there."""
    path = Path(path_str)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    meta = data.get("meta", {}) or {}
    # sensible fallbacks
    meta.setdefault("id", path.stem)
    meta.setdefault("title", path.stem.replace("_"," "))
    meta.setdefault("summary", "")
    meta.setdefault("tags", [])
    return meta


def _load_preset_meta(path: Path) -> dict:
    """
    One preset's metadata, parsed at most once per version of the file.

    The listing screens ask for this repeatedly for the same files -- and
    every ask was a full read and a YAML parse. That is cheap on a local disk
    and not cheap on a Windows drive mounted into WSL, where the repo often
    lives; the same listing is also built several times during one wizard
    session.

    A copy is returned rather than the cached dict itself, because callers
    treat the result as theirs and would otherwise be able to poison every
    later reader.
    """
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return {"id": path.stem, "title": path.stem.replace("_", " "),
                "summary": "", "tags": []}
    return dict(_preset_meta_uncached(str(path), st.st_mtime_ns, st.st_size))

def _cmd_list_presets():
    rows = []
    for p in _iter_presets():
        m = _load_preset_meta(p)
        rows.append((m["id"], m["title"], ", ".join(m.get("tags", [])), str(p)))
    if not rows:
        print("No presets found.")
        return
    w_id = max(len(r[0]) for r in rows)
    w_title = max(len(r[1]) for r in rows)
    print(f"{'ID'.ljust(w_id)}  {'Title'.ljust(w_title)}  Tags")
    print(f"{'-'*w_id}  {'-'*w_title}  {'-'*20}")
    for r in rows:
        print(f"{r[0].ljust(w_id)}  {r[1].ljust(w_title)}  {r[2]}")

def _cmd_describe_preset(preset_name: str):
    # search by id or file stem match
    for p in _iter_presets():
        m = _load_preset_meta(p)
        if preset_name in {m["id"], p.stem}:
            print(f"{m['title']}  [{m['id']}]")
            if m.get("summary"):
                print("\nSummary:\n" + textwrap.fill(m["summary"], 100))
            if m.get("use_cases"):
                print("\nUse cases: " + ", ".join(m["use_cases"]))
            if m.get("inputs"):
                print("\nInputs:", m["inputs"])
            if m.get("outputs"):
                print("\nOutputs:")
                for o in m["outputs"]:
                    print(f"  - {o.get('path','')} — {o.get('desc','')}")
            if m.get("variables"):
                print("\nVariables:")
                for k, spec in m["variables"].items():
                    default = spec.get("default", "")
                    desc = spec.get("desc", "")
                    print(f"  - {k} (default: {default}) — {desc}")
            if m.get("cli_example"):
                print("\nExample:\n" + m["cli_example"])
            print(f"\nFile: {p}")
            return
    print(f"Preset '{preset_name}' not found.")

# ============== JSON-safe casting ==============

def _json_safe(obj: Any) -> Any:
    """
    Convert arbitrary Python objects into JSON-serializable structures.

    This helper is used when writing the run manifest and step results
    to disk. It handles a few common non-JSON types and normalizes them:

    - `pathlib.Path` → `str`
    - `dataclasses.dataclass` → nested `dict` via `asdict(...)`
    - Arbitrary objects with `__dict__` → `vars(obj)` (best-effort)
    - `list` / `tuple` / `set` and `dict` → deep-converted recursively

    Parameters
    ----------
    obj : Any
        The object to normalize.

    Returns
    -------
    Any
        A structure that `json.dumps(...)` can serialize (strings, numbers,
        booleans, `None`, lists, and dicts). Any unknown objects fall
        through unchanged (letting `json` raise if it still cannot serialize).
    """
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return obj


# ============== Discovery ==============

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", ".3gp"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".aifc"}

def discover_inputs(root_dir: Path, kind: str) -> List[Path]:
    """
    Recursively discover input files under a root folder.

    The preset's ITEM-scoped steps operate over a list of inputs. This
    function builds that list by scanning `root_dir` and selecting files by
    type:

    - kind == "video": only common video extensions (e.g., .mp4, .mov, .mkv)
    - kind == "audio": only common audio extensions (e.g., .wav, .mp3, .flac)
    - kind == "any":   all files

    Parameters
    ----------
    root_dir : Path
        Directory to scan (will be resolved to an absolute path).
    kind : {"audio","video","any"}
        Filter that determines which file extensions are included.

    Returns
    -------
    List[Path]
        Sorted list of absolute file paths.

    Raises
    ------
    FileNotFoundError
        If `root_dir` does not exist.
    """
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    out: List[Path] = []
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if kind == "video" and ext in _VIDEO_EXTS:
            out.append(p)
        elif kind == "audio" and ext in _AUDIO_EXTS:
            out.append(p)
        elif kind == "any":
            out.append(p)
    return sorted(out)


# ============== Preset loading / vars ==============

def resolve_preset_path(name: str) -> Path:
    """
    Find a preset file by its `meta.id` or filename stem.

    Searches every directory returned by `_get_preset_dirs()` — the built-in
    `taters/pipelines/presets/` folder first, then `./pipelines` relative to the
    current working directory — so anything shown by `--list-presets` can also
    be loaded with `--preset`.

    Parameters
    ----------
    name : str
        Preset `meta.id` or filename stem (with or without a `.yaml`/`.yml`
        extension).

    Returns
    -------
    Path
        Path to the matching preset file.

    Raises
    ------
    FileNotFoundError
        If no preset matches, listing the presets that *are* available.
    """
    # drop a .yaml/.yml suffix if the caller included one (".yml" is four
    # characters, ".yaml" is five — hence Path.stem rather than slicing).
    wanted = Path(name).stem if Path(name).suffix.lower() in {".yaml", ".yml"} else name
    available: List[str] = []
    for p in _iter_presets():
        meta = _load_preset_meta(p)
        available.append(meta["id"])
        if wanted in {meta["id"], p.stem}:
            return p
    searched = ", ".join(str(d) for d in _get_preset_dirs())
    known = ", ".join(sorted(set(available))) or "(none)"
    raise FileNotFoundError(
        f"Preset not found: {name}\nSearched: {searched}\nAvailable presets: {known}"
    )


def load_preset_by_name(name: str) -> dict:
    """
    Load a named pipeline preset by `meta.id` or filename stem.

    Presets are resolved with :func:`resolve_preset_path`, which searches the
    built-in `presets/` folder as well as a project-local `./pipelines` folder.

    Parameters
    ----------
    name : str
        Preset `meta.id` or filename stem.

    Returns
    -------
    dict
        Parsed YAML as a Python dictionary. Returns `{}` for an empty file.

    Raises
    ------
    FileNotFoundError
        If no preset with that name exists in any search directory.
    """
    path = resolve_preset_path(name)
    print(f"[pipeline] Using preset: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_yaml_file(path: Path) -> dict:
    """
    Load a YAML file into a Python dictionary.

    Parameters
    ----------
    path : Path
        Full path to a YAML file.

    Returns
    -------
    dict
        Parsed YAML contents. Empty files yield `{}`.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merge_vars(base: dict, overlay: dict) -> dict:
    """
    Shallow-merge two variable dictionaries.

    Later sources of variables (e.g., `--vars-file`, then repeated `--var`
    overrides) should replace keys from earlier sources. This helper
    applies a simple `dict.update(...)` and returns a new dictionary.

    Parameters
    ----------
    base : dict
        The starting dictionary of variables.
    overlay : dict
        The dictionary whose keys override entries in `base`.

    Returns
    -------
    dict
        A new dictionary with merged keys/values.
    """
    out = dict(base or {})
    out.update(overlay or {})
    return out

def parse_var_overrides(pairs: List[str]) -> dict:
    """
    Parse `--var key=value` CLI overrides into typed Python values.

    Typing rules:
      - "true"/"false" (case-insensitive) → bool
      - "null"/"none" (case-insensitive)  → None
      - integer or float strings → numeric
      - all else → raw string

    Parameters
    ----------
    pairs : List[str]
        CLI arguments of the form `["k1=v1", "k2=v2", ...]`.

    Returns
    -------
    dict
        Mapping from variable name to parsed value.

    Raises
    ------
    ValueError
        If any entry does not contain an '=' separator.
    """
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"--var expects key=value, got: {s}")
        k, v = s.split("=", 1)
        vs = v.strip()
        if vs.lower() in {"true", "false"}:
            out[k] = (vs.lower() == "true")
        elif vs.lower() in {"null", "none"}:
            out[k] = None
        else:
            try:
                out[k] = float(vs) if "." in vs else int(vs)
            except Exception:
                out[k] = v
    return out


# ============== Templating (native-type preserving) ==============

_VAR_RE = re.compile(r"\{\{([^}]+)\}\}")

def _deep_get(d: Any, dotted: str) -> Any:
    """
    Resolve a dotted attribute/key path within nested dicts/objects.

    This is used by templating expressions like `{{global.some.nested.value}}`
    or `{{pick:artifact.path.to.field}}`.

    Resolution order per path segment:
      1) If `d` is a dict and has the key → descend by key
      2) Else if `d` has an attribute with that name → use `getattr`
      3) Otherwise → KeyError

    Parameters
    ----------
    d : Any
        Root object/dict to traverse.
    dotted : str
        Dotted path, e.g. `"a.b.c"`.

    Returns
    -------
    Any
        The resolved value.

    Raises
    ------
    KeyError
        If any path segment cannot be resolved.
    """
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            raise KeyError(f"Could not resolve '{part}' in {_json_safe(cur)}")
    return cur

def _eval_expr(
    expr: str,
    *,
    item_ctx: dict,
    globals_ctx: dict,
    vars_ctx: dict,
    input_path: Path
) -> Any:
    """
    Evaluate a single templating expression and return a native Python value.

    Supported expressions
    ---------------------
      - input              → absolute path to the current item's input
      - cwd                → current working directory
      - var:<key>          → look up vars[<key>]
      - global.<path>      → deep lookup on globals (explicit)
      - pick:<name>.<path> → deep lookup inside an artifact; searches item, then globals
      - <name>             → *bare* artifact name; resolves from item_ctx, else globals_ctx

    Notes
    -----
    - If the expression cannot be resolved by any rule, the raw {{...}} string
      is returned unchanged. This preserves progressive templating behavior
      and yields clearer downstream errors if something stays unresolved.
    """
    expr = expr.strip()

    # special literals
    if expr == "input":
        return str(input_path)
    if expr == "cwd":
        return str(Path.cwd())

    # variables
    if expr.startswith("var:"):
        key = expr.split(":", 1)[1]
        if key not in vars_ctx:
            raise KeyError(f"Variable '{key}' not found")
        return vars_ctx[key]

    # explicit global lookup: {{global.foo.bar}}
    if expr.startswith("global."):
        keypath = expr.split(".", 1)[1]
        return _deep_get(globals_ctx, keypath) if keypath else globals_ctx

    # nested artifact lookup with pick:
    # we prefer item artifacts, and fall back to globals if the item hasn't got it.
    if expr.startswith("pick:"):
        path = expr.split(":", 1)[1]
        if "." not in path:
            raise KeyError("pick: requires 'artifact.nested.path'")
        art, nested = path.split(".", 1)
        base = item_ctx.get(art, None)
        if base is None:
            base = globals_ctx.get(art, None)
        if base is None:
            raise KeyError(f"Artifact '{art}' not found in item or globals context")
        return _deep_get(base, nested)

    # bare artifact name: resolve from the item first, then globals
    if expr in item_ctx:
        return item_ctx[expr]
    if expr in globals_ctx:
        return globals_ctx[expr]

    # fallback: leave the template as-is (string)
    return "{{" + expr + "}}"


def render_value(
    val: Any,
    *,
    item_ctx: dict,
    globals_ctx: dict,
    vars_ctx: dict,
    input_path: Path
) -> Any:
    """
    Render templating expressions within a value (str, list, or dict).

    Behavior
    --------
      - Dicts/lists/tuples: render recursively.
      - If a string is exactly one template token (e.g., "{{var:text_cols}}"),
        return the *native* value of that expression (list, int, bool, ...).
      - Otherwise, perform string substitution for every {{...}} occurrence and
        return the resulting string.

    Resolution rules (summary)
    --------------------------
      - {{input}} / {{cwd}}
      - {{var:key}}
      - {{global.path}} (explicit globals)
      - {{pick:name.path}}  → search item, then globals
      - {{name}}            → bare name; search item, then globals
    """
    if isinstance(val, dict):
        return {
            k: render_value(
                v,
                item_ctx=item_ctx,
                globals_ctx=globals_ctx,
                vars_ctx=vars_ctx,
                input_path=input_path,
            )
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple)):
        return [
            render_value(
                v,
                item_ctx=item_ctx,
                globals_ctx=globals_ctx,
                vars_ctx=vars_ctx,
                input_path=input_path,
            )
            for v in val
        ]
    if not isinstance(val, str):
        return val

    # the entire string is one template → hand back the native type
    m = _VAR_RE.fullmatch(val.strip())
    if m:
        return _eval_expr(
            m.group(1),
            item_ctx=item_ctx,
            globals_ctx=globals_ctx,
            vars_ctx=vars_ctx,
            input_path=input_path,
        )

    # otherwise we substitute each token as a string
    def _subst(match: re.Match) -> str:
        expr = match.group(1)
        v = _eval_expr(
            expr,
            item_ctx=item_ctx,
            globals_ctx=globals_ctx,
            vars_ctx=vars_ctx,
            input_path=input_path,
        )
        return str(v)

    return _VAR_RE.sub(_subst, val)



# ============== Call resolver (no hard-coded mapping) ==============

def resolve_call(call_name: str, potato: Taters):
    """
    Resolve a call target from a preset step into an actual callable.

    Supported forms
    ---------------
    1) Taters instance methods (recommended):
       - `"potato.audio.convert_to_wav"`
       - `"potato.text.analyze_with_dictionaries"`
       The function is resolved via attribute chaining on a single
       `Taters()` instance created for the whole run.

    2) Dotted import paths:
       - `"package.module:function"`
       - `"package.module.func"`
       - `"package.module.Class.method"`
       The target is imported and attributes are resolved. The final target
       must be callable.

    Parameters
    ----------
    call_name : str
        Call string from the preset step's `call:` field.
    potato : Taters
        The shared `Taters` instance for resolving `"potato.*"` calls.

    Returns
    -------
    Callable
        The function/object that will be invoked for the step.

    Raises
    ------
    AttributeError, KeyError, TypeError
        If the target cannot be resolved or is not callable.
    """
    if call_name.startswith("potato."):
        obj: Any = potato
        for part in call_name.split(".")[1:]:
            if not hasattr(obj, part):
                raise AttributeError(f"{call_name}: '{part}' not found on {obj}")
            obj = getattr(obj, part)
        if not callable(obj):
            raise TypeError(f"{call_name} is not callable")
        return obj

    # allow dotted import paths
    # we support both "pkg.mod:func" and "pkg.mod.func"
    mod_path, sep, tail = call_name.partition(":")
    if not sep:
        # split at last dot for function
        parts = call_name.rsplit(".", 1)
        if len(parts) == 2:
            mod_path, tail = parts
        else:
            raise KeyError(f"Cannot resolve call target: {call_name}")
    module = importlib.import_module(mod_path)
    target = module
    for attr in tail.split("."):
        if not hasattr(target, attr):
            raise AttributeError(f"{call_name}: '{attr}' not found in {target}")
        target = getattr(target, attr)
    if not callable(target):
        raise TypeError(f"{call_name} resolved to non-callable: {target}")
    return target


# ============== Step runners ==============

def run_item_step_for_one_input(
    *, step: dict, input_path: Path, potato: Taters, item_artifacts: Dict[str, Any],
    globals_ctx: Dict[str, Any], vars_ctx: Dict[str, Any],
    on_progress: Optional[Callable[..., None]] = None, quiet: bool = False
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Execute a single ITEM-scoped step for one input path.

    Lifecycle
    ---------
    1) Template the step's `with:` parameters using `render_value(...)`.
    2) Validate any `require:` keys after templating (fail fast if missing).
    3) Resolve the callable (Taters method or import path).
    4) Invoke with keyword arguments.
    5) If the step specified `save_as: <name>`, store the return value under
       that name in the item's `artifacts` dict.

    Parameters
    ----------
    step : dict
        The step definition block from the preset.
    input_path : Path
        The current input file for ITEM scope.
    potato : Taters
        Shared Taters instance used to call `potato.*` targets.
    item_artifacts : Dict[str, Any]
        The current item's artifact dictionary (mutated across steps).
    globals_ctx : Dict[str, Any]
        Global artifacts (from GLOBAL steps).
    vars_ctx : Dict[str, Any]
        Merged variables.
    on_progress : callable, optional
        Progress sink for *this file*, passed on to the step function when it
        declares one. An ITEM step is counted from outside -- files finished out
        of files found -- but that says nothing while a single file is running,
        and transcription is routinely minutes per file. Without this a stalled
        step and a working one look identical until the first file lands.
    quiet : bool, default False
        Suppress the step function's own printing. See
        :func:`_inject_runner_kwargs`.

    Returns
    -------
    Tuple[str, Dict[str, Any], Dict[str, Any]]
        A tuple `(status, new_artifacts, err)` where:
          - `status` is `"ok"` or `"error"`.
          - `new_artifacts` is a (possibly empty) dict of artifacts to merge.
          - `err` contains an `"error"` message on failure.
    """
    call = step["call"]
    params = step.get("with", {})

    # --- templating can fail (e.g. pick:<artifact>.* when a prior step failed)
    try:
        rendered = render_value(
            params,
            item_ctx=item_artifacts,
            globals_ctx=globals_ctx,
            vars_ctx=vars_ctx,
            input_path=input_path
        )
    except KeyError as e:
        # common case: a previous step failed for this item, so the artifact's missing
        msg = f"Templating failed (likely missing artifact): {e}"
        return ("error", {}, {"error": f"{call} failed: {msg}"})
    except Exception as e:
        return ("error", {}, {"error": f"{call} failed during templating: {e}"})


    # required keys check (post-templating)
    for key in step.get("require", []):
        if key not in rendered or rendered[key] in (None, "", []):
            return ("error", {}, {"error": f"Missing required parameter '{key}' after templating"})

    # --- invoke target
    func = resolve_call(call, potato)
    # workers=1: an ITEM step's parallelism is the fan-out itself; a pool
    # inside each fanned-out call would multiply into cores-squared.
    rendered = _inject_runner_kwargs(func, rendered, on_progress=on_progress,
                                     quiet=quiet, workers=1)
    try:
        result = func(**rendered)
    except Exception as e:
        # the exception's *type* is part of the message: a bare KeyError
        # renders as just its key ("'1467-'"), which told a user nothing about
        # what happened, let alone where.
        return ("error", {}, {"error": f"{call} failed: {type(e).__name__}: {e}"})

    out: Dict[str, Any] = {}
    if "save_as" in step:
        out[step["save_as"]] = result
    return ("ok", out, {})


def _accepts_on_progress(func: Any) -> bool:
    """
    Whether a step function can report its own progress.

    A signature check rather than a registry: a new analyzer opts in by adding
    the parameter, and there is nothing here to remember to update.

    The `**kwargs` case is the Taters facade, whose real signature is invisible
    from here; `Taters._forward` drops the argument again for targets that
    cannot take it, which makes offering it safe rather than a guess.
    """
    return _accepts(func, "on_progress")


def _guessed_ceiling(step: dict) -> Optional[int]:
    """
    A ceiling for a step that never declared one.

    Presets written by hand -- or by a version of the wizard older than
    `max_workers` -- carry no ceiling at all, and the step most likely to need
    one is the step most likely to predate it. So a step that hands `device` to
    a model is capped at one file at a time unless it says otherwise.

    That is a guess, and it is the cautious direction: wrong-but-slow is
    recoverable, while four copies of `large-v3` on an 8 GB card is an
    out-of-memory error halfway through a batch. A preset that wants more says
    so with `max_workers:` or `workers:`, both of which win over this.
    """
    params = step.get("with") or {}
    reads_device = any(
        isinstance(v, str) and "{{var:device}}" in v for v in params.values()
    )
    # `worker_cap(None)` is the shared rules' cautious answer, so the guess here
    # and the wizard's default can't drift apart.
    return worker_cap(None) if reads_device else None


def _run_wide_workers(explicit: Optional[int], preset_vars: Dict[str, Any]) -> int:
    """
    The run's parallelism budget, resolved from one dial in three places.

    Priority: an explicit ask (the ``--workers`` flag, or a caller argument)
    wins; otherwise the preset's ``workers`` variable -- the same shared
    setting the text steps read for their own process pools, so one number
    steers the whole run. ``0``, ``None`` or absent all mean automatic,
    whichever layer they came from, and every answer goes through
    ``resolve_workers``: automatic is three-quarters of this machine's
    logical cores, an explicit ask is honored up to the core count.
    """
    from ..helpers.parallel_map import resolve_workers

    if explicit:
        return resolve_workers(max(1, int(explicit)))
    try:
        from_vars = int(preset_vars.get("workers", 0) or 0)
    except (TypeError, ValueError):
        from_vars = 0
    return resolve_workers(max(0, from_vars))


def _workers_for(step: dict, workers: int,
                 say: Optional[Callable[[str], None]] = None) -> int:
    """
    How many files this step may work on at once.

    Three inputs, in priority order:

    * ``workers:`` on the step -- an explicit instruction, obeyed as written.
      Someone who pins a number has decided, and this is the escape hatch for a
      machine with more GPU memory than the default assumes.
    * ``max_workers:`` on the step -- a ceiling, not a setting. The wizard writes
      it for steps that put a model on the GPU, where more workers means more
      copies of it. `--workers 8` still means 8 for the ffmpeg and merge steps
      in the same run; it just stops applying where more is worse.
    * ``workers`` -- the run-wide default, from `--workers`.

    The ceiling is deliberately not conditional on the device turning out to be
    a GPU. Resolving that here would mean asking a question the runner cannot
    answer honestly -- CTranslate2 reports a usable CUDA device and then fails
    at the first encode -- and the capped steps are no faster with extra workers
    on a CPU either.
    """
    if "workers" in step:
        return max(1, int(step["workers"]))

    wanted = max(1, int(workers))
    ceiling = step.get("max_workers", _guessed_ceiling(step))
    if ceiling is None:
        return wanted

    allowed = max(1, min(wanted, int(ceiling)))
    if allowed < wanted and say is not None:
        # worth a line: someone who passed `--workers 8` and sees a step running
        # two files at a time should be told why, or they'll figure the setting
        # got ignored.
        say(f"[pipeline] Running {allowed} at a time (this step loads a model on "
            f"the GPU; --workers {wanted} still applies elsewhere)")
    return allowed


def _accepts(func: Any, name: str) -> bool:
    """Whether `func` takes a parameter called `name` (or absorbs **kwargs)."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _materialize_assets(func: Any, rendered: Dict[str, Any],
                        spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write the word lists and fitted files a model carries, and bind them.

    A private extraction made for a saved model (``for_model`` in the step)
    has to read the model's *own* lists -- the dictionaries it was fitted
    with, the vocabulary its matrix was scanned against, the theme model it
    applies -- and the model file is where those live. The step says which
    model file and which entries (``{"from": path, "take": {param:
    [names]}}``); this writes each beside the step's output, verifies the
    digest, and hands the parameter the paths (or the one path, for a
    parameter that takes a single file).

    Written every time the digest disagrees or the file is missing, and
    left alone otherwise, so a re-run finds them in place.
    """
    from ..helpers.atomic import atomic_write
    from ..helpers.provenance import text_digest

    model_path = Path(str(spec.get("from") or ""))
    try:
        with model_path.open("r", encoding="utf-8") as fh:
            carried = (json.load(fh) or {}).get("assets") or {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"the model file {model_path.name} could not be read for the "
            f"word lists and fitted files this step needs: {e}") from None
    out = rendered.get("out_features_csv")
    root = (Path(str(out)).parent if out else Path.cwd()) / "assets"
    bound = dict(rendered)
    for param, names in (spec.get("take") or {}).items():
        have = {e.get("name"): e for e in carried.get(param) or []}
        paths: List[str] = []
        for name in names:
            entry = have.get(name)
            if not entry or entry.get("text") is None:
                raise ValueError(
                    f"the model file {model_path.name} no longer carries "
                    f"{name!r}, which this step needs for {param}.")
            target = root / param / Path(str(name)).name
            wanted = entry.get("sha256")
            if not target.is_file() or text_digest(
                    target.read_text(encoding="utf-8-sig",
                                     errors="replace")) != wanted:
                target.parent.mkdir(parents=True, exist_ok=True)
                with atomic_write(target, mode="w", encoding="utf-8") as fh:
                    fh.write(entry["text"])
            paths.append(str(target))
        bound[param] = paths if _takes_many(func, param) else \
            (paths[0] if paths else None)
    return bound


def _takes_many(func: Any, param: str) -> bool:
    """Whether a parameter is typed to take several paths rather than one."""
    try:
        annotation = inspect.signature(func).parameters[param].annotation
    except (KeyError, TypeError, ValueError):
        return False
    text = annotation if isinstance(annotation, str) else repr(annotation)
    return any(word in text for word in ("Sequence", "List", "list",
                                         "Iterable", "Tuple", "tuple"))


def _inject_runner_kwargs(
    func: Any,
    rendered: Dict[str, Any],
    *,
    on_progress: Optional[Callable[..., None]],
    quiet: bool,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Add the arguments the runner supplies rather than the preset.

    `workers` goes to any step that declares it -- the same opt-in-by-signature
    contract as `on_progress`, and it is what makes a new module's authoring
    story one line: *take a `workers` kwarg and the pipeline's parallelism
    dial reaches you*, no recipe wiring required. GLOBAL steps receive the
    run's resolved budget; ITEM steps receive 1, because their parallelism is
    the file fan-out itself and an internal pool per fanned-out call would
    multiply into cores-squared processes.

    The other two are about who owns the screen:

    `on_progress` goes to any step that declares it. A step that can count
    itself is the only thing that can, so this is opt-in by signature rather
    than by a registry there would be no reason to keep up to date.

    `verbose=False` goes to any step that takes it -- but *only* to turn output
    off, never on. Every step function defaults to printing, which is right at a
    shell prompt and wrong underneath a live display, and passing the runner's
    `verbose=True` down would change what a CLI run prints for any step whose
    own default is quieter. Silence is the only direction that is safe to force.

    A value the preset set explicitly always wins: someone who wrote
    `verbose: true` into a step meant it.
    """
    extra: Dict[str, Any] = {}
    if on_progress is not None and "on_progress" not in rendered and _accepts(func, "on_progress"):
        extra["on_progress"] = on_progress
    if quiet and "verbose" not in rendered and _accepts(func, "verbose"):
        extra["verbose"] = False
    if workers is not None and "workers" not in rendered and _accepts(func, "workers"):
        extra["workers"] = workers
    if not extra:
        return rendered
    return {**rendered, **extra}


def run_global_step(
    *, step: dict, potato: Taters, globals_ctx: Dict[str, Any], vars_ctx: Dict[str, Any],
    manifest_path: Path, on_progress: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False, workers: Optional[int] = None
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Execute a single GLOBAL-scoped step (runs once per pipeline).

    Differences from ITEM steps
    ---------------------------
    - The templating `item_ctx` is empty.
    - The run manifest path is injected into `vars` as `run_manifest`,
      so presets can reference it in GLOBAL stages.
    - On success, any values from `save_as:` are merged into the `globals`
      artifact map.

    Parameters
    ----------
    step : dict
        The step definition block from the preset.
    potato : Taters
        Shared Taters instance used to call `potato.*` targets.
    globals_ctx : Dict[str, Any]
        Accumulated global artifacts (readable by later steps).
    vars_ctx : Dict[str, Any]
        Merged variables.
    manifest_path : Path
        Path where the JSON run manifest is (or will be) saved.

    Returns
    -------
    Tuple[str, Dict[str, Any], Dict[str, Any]]
        A tuple `(status, new_globals, err)` mirroring the ITEM step shape.
    """
    call = step["call"]
    params = step.get("with", {})

    # expose the manifest path via vars
    vars_aug = dict(vars_ctx)
    vars_aug["run_manifest"] = str(manifest_path)

    # --- templating can fail too (e.g., referencing a global that wasn't saved yet)
    try:
        rendered = render_value(
            params,
            item_ctx={},  # no item context in GLOBAL
            globals_ctx=globals_ctx,
            vars_ctx=vars_aug,
            input_path=manifest_path
        )
    except KeyError as e:
        msg = f"Templating failed (likely missing global artifact): {e}"
        return ("error", {}, {"error": f"{call} failed: {msg}"})
    except Exception as e:
        return ("error", {}, {"error": f"{call} failed during templating: {e}"})


    func = resolve_call(call, potato)

    if step.get("assets"):
        try:
            rendered = _materialize_assets(func, rendered, step["assets"])
        except Exception as e:
            return ("error", {}, {"error": f"{call} failed: {type(e).__name__}: {e}"})

    # a GLOBAL step is a single call, so from out here we can't count its
    # progress: it's either not started or finished. functions that *can*
    # count themselves say so by declaring an `on_progress` parameter and get
    # one injected here -- and the same signature trick hands `workers` (the
    # run's resolved parallelism budget) to any step that can spend it.
    rendered = _inject_runner_kwargs(func, rendered, on_progress=on_progress,
                                     quiet=quiet, workers=workers)

    try:
        result = func(**rendered)
    except Exception as e:
        # the exception's *type* is part of the message: a bare KeyError
        # renders as just its key ("'1467-'"), which told a user nothing about
        # what happened, let alone where.
        return ("error", {}, {"error": f"{call} failed: {type(e).__name__}: {e}"})

    out: Dict[str, Any] = {}
    if "save_as" in step:
        out[step["save_as"]] = result
    return ("ok", out, {})


# ====== Per-item worker (module-level so ProcessPoolExecutor can pickle it) ======

def _run_item_step(
    i: int,
    p: Path,
    step: dict,
    potato: Any,
    item_artifacts: Dict[str, Any],
    globals_ctx: Dict[str, Any],
    vars_ctx: Dict[str, Any],
    on_start: Optional[Callable[[int, str], None]] = None,
    on_progress: Optional[Callable[..., None]] = None,
    quiet: bool = False,
) -> Tuple[int, str, Dict[str, Any], Dict[str, Any]]:
    # called from inside the worker, so a UI can show which file each slot is
    # actually working on -- and, through `on_progress`, how far into it the
    # worker has got. we only supply both for the thread engine: a
    # ProcessPoolExecutor has to pickle its arguments, and a live callback into
    # the parent process can't cross that boundary.
    if on_start is not None:
        on_start(i, str(p))

    status, new_artifacts, err = run_item_step_for_one_input(
        step=step,
        input_path=p,
        potato=potato,
        item_artifacts=item_artifacts,
        globals_ctx=globals_ctx,
        vars_ctx=vars_ctx,
        on_progress=on_progress,
        quiet=quiet,
    )
    return i, status, new_artifacts, err


# ============== Main ==============

def run_preset(
    preset: dict,
    *,
    root_dir: Path | str | None = None,
    file_type: str = "any",
    vars_ctx: Dict[str, Any] | None = None,
    workers: Optional[int] = None,
    out_manifest: Path | str | None = None,
    preset_name: str | None = None,
    on_event=None,
    verbose: bool = True,
    work_dir: Optional[Path] = None,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a loaded preset and return its manifest.

    This is the engine `main()` wraps. It is separate so that callers other
    than the command line -- the setup wizard in :mod:`taters.ui.wizard`, and
    anything else that already holds a preset dict -- can run a pipeline
    without building an argv and without the process exiting underneath them.

    Parameters
    ----------
    preset : dict
        A loaded preset: `steps`, and optionally `vars` and `meta`.
    root_dir : Path | str | None, optional
        Folder to scan for inputs. Required when the preset has any ITEM-scoped
        steps; ignored when it does not.
    file_type : {"audio","video","any"}, default "any"
        Extension filter for discovery.
    vars_ctx : dict, optional
        The variable context, already merged. When omitted, the preset's own
        `vars` block is used as-is.
    workers : int, default 4
        Default concurrency for ITEM steps. A step's own `workers:` wins.
    out_manifest : Path | str | None, optional
        Where to write the run manifest. Defaults to `./run_manifest.json`.
    preset_name : str, optional
        Recorded in the manifest for provenance.
    on_event : callable, optional
        Called as ``on_event(name, **payload)`` as the run progresses, with
        `name` one of `run_start`, `step_start`, `item_done`, `step_done`,
        `run_done`. Exceptions raised by the callback are swallowed: a UI bug
        must not take down a run that may be hours in.
    verbose : bool, default True
        Print progress to stdout. Set False when `on_event` is doing the
        reporting instead -- this silences the runner's own lines *and* is
        passed down to every step function that accepts a `verbose` argument.
        Steps print by default, which is right at a shell prompt and ruinous
        underneath a live display: text written into the region a display owns
        corrupts it. Note the asymmetry -- False is pushed down, True is not,
        so a step that is quiet by default stays quiet.

    Returns
    -------
    dict
        The run manifest. Check `manifest["errors"]` and each item's
        `"status"` to tell a clean run from a partial one -- unlike `main()`,
        this function never calls `sys.exit`.
    """
    # relative output paths in a preset ("features/readability.csv") resolve
    # against the process's working directory. `work_dir` moves that for the
    # duration of the run, and that's what lets a pipeline own a folder without
    # every recipe having to spell out an absolute path.
    #
    # we restore it in a finally: leaving the interpreter somewhere else
    # afterwards would silently relocate everything the caller does next.
    previous_cwd = None
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        # resolved BEFORE the chdir. these paths were given relative to where
        # the caller stood; resolving them after moving re-anchored them under
        # work_dir, so `taters --dir data` wrote its manifest to the phantom
        # data/mypipe/data/mypipe/run_manifest.json while the finish screen
        # pointed at the real path -- which didn't exist.
        if out_manifest is not None:
            out_manifest = Path(out_manifest).resolve()
        if root_dir is not None:
            root_dir = Path(root_dir).resolve()
        previous_cwd = Path.cwd()
        os.chdir(work_dir)
    try:
        return _run_preset(
            preset, root_dir=root_dir, file_type=file_type, vars_ctx=vars_ctx,
            workers=workers, out_manifest=out_manifest, preset_name=preset_name,
            on_event=on_event, verbose=verbose, command=command,
        )
    finally:
        if previous_cwd is not None:
            os.chdir(previous_cwd)


def _run_preset(
    preset: Dict[str, Any],
    *,
    root_dir: Optional[Path] = None,
    file_type: str = "any",
    vars_ctx: Optional[Dict[str, Any]] = None,
    workers: Optional[int] = None,
    out_manifest: Optional[Path] = None,
    preset_name: Optional[str] = None,
    on_event: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    """The body of :func:`run_preset`, with the working directory already set."""
    def emit(name: str, **payload) -> None:
        if on_event is None:
            return
        try:
            on_event(name, **payload)
        except Exception:
            pass

    def say(message: str) -> None:
        if verbose:
            print(message)

    steps: List[dict] = preset.get("steps", []) or []
    if not steps:
        raise ValueError("Preset has no steps")

    if vars_ctx is None:
        vars_ctx = dict(preset.get("vars", {}) or {})

    # one dial: an explicit --workers wins, else the preset's `workers`
    # variable (the same shared setting the text steps read for their own
    # pools), else automatic. we resolve it once, then write it back into the
    # vars so the `{{var:workers}}` templates inside text steps and the
    # per-file fan-out below both see the same number -- whichever door it
    # came in by.
    workers = _run_wide_workers(workers, vars_ctx)
    vars_ctx["workers"] = workers

    has_item_steps = any((step.get("scope", "item") == "item") for step in steps)

    inputs: List[Path] = []
    resolved_root: Path | None = None
    if has_item_steps:
        if not root_dir:
            raise ValueError("root_dir is required because this preset contains ITEM-scoped steps.")
        resolved_root = Path(root_dir).resolve()
        inputs = discover_inputs(resolved_root, file_type)
        say(f"[pipeline] Found {len(inputs)} '{file_type}' input(s) under {resolved_root}")
        if not inputs:
            say("[pipeline] No inputs found; ITEM steps will be skipped.")
    else:
        say("[pipeline] Preset has only GLOBAL steps; skipping input discovery.")

    manifest: Dict[str, Any] = {
        "preset": preset_name or (preset.get("meta", {}) or {}).get("id") or "<inline>",
        # the exact shell command that reproduces this run, parameters and all,
        # when the caller can tell us (the TUI can; see ui.wizard.repro_command).
        # we record it so a run built by clicking through menus can be re-run
        # from a terminal without reverse-engineering the choices behind it.
        "command": command,
        "root_dir": str(resolved_root) if resolved_root else None,
        "file_type": file_type if has_item_steps else None,
        "vars": _json_safe(vars_ctx),
        "items": [{"input": str(p), "artifacts": {}, "status": "pending", "errors": []} for p in inputs],
        "globals": {},
        "errors": [],
    }
    out_manifest_path = Path(out_manifest or (Path.cwd() / "run_manifest.json"))

    potato = Taters()
    globals_ctx: Dict[str, Any] = {}

    def _persist_manifest() -> None:
        """Write the run manifest to disk (called after every step, and on failure)."""
        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)

    emit("run_start", inputs=[str(p) for p in inputs], steps=len(steps),
         manifest_path=str(out_manifest_path))

    for idx, step in enumerate(steps, 1):
        scope = step.get("scope", "item")
        call_name = step.get("call")
        say(f"[pipeline] Step {idx}/{len(steps)}: {call_name}  (scope={scope})")
        emit("step_start", index=idx, total=len(steps), call=call_name, scope=scope,
             items=len(inputs) if scope == "item" else 1,
             # a run can contain the same step twice: once as the user asked
             # for it, once measured to a saved model's own settings. without
             # this the progress display shows two identical lines and the
             # second looks like a bug.
             for_model=step.get("for_model"))

        if scope == "item":
            if not inputs:
                say(f"[pipeline] No inputs; skipping ITEM step: {call_name}")
                emit("step_done", index=idx, call=call_name, status="skipped")
                continue

            step_engine = step.get("engine", "thread")  # "thread" (default) or "process"
            step_workers = _workers_for(step, workers, say)

            def _item_started(item_index: int, path_str: str,
                              _idx: int = idx, _call: str = call_name) -> None:
                """
                Announce that a worker has picked up a file.

                Fires from inside the worker thread, which is what makes a bar
                per concurrent slot possible: without it a UI knows only how
                many items are outstanding, not which ones are in flight.
                """
                emit("item_start", index=_idx, call=_call,
                     item=item_index, input=path_str)

            def _item_progress_for(item_index: int, path_str: str,
                                   _idx: int = idx, _call: str = call_name):
                """
                Build the progress sink for one file.

                Bound per item rather than shared, because the four workers all
                report into the same event stream and the item index is the only
                thing that tells their numbers apart.
                """
                def _report(done: int, total: Optional[int] = None,
                            message: Optional[str] = None,
                            unit: Optional[str] = None) -> None:
                    emit("item_progress", index=_idx, call=_call, item=item_index,
                         input=path_str, done=done, total=total, message=message,
                         unit=unit)
                return _report

            # a ProcessPoolExecutor pickles everything it's handed, and a live
            # callback into the parent process can't survive that. threads --
            # the default everywhere -- share memory, so they can.
            if step_engine != "thread":
                _item_started = None          # noqa: F811 - deliberate, see above
                _item_progress_for = None     # noqa: F811 - deliberate, see above

            # with nobody listening, offering a step a progress sink isn't free:
            # counting rows to get a denominator is a whole extra pass over the
            # input, which a plain `--preset` run on the command line has never
            # paid for and shouldn't start paying for now.
            if on_event is None:
                _item_progress_for = None     # noqa: F811 - deliberate, see above

            Executor = cf.ProcessPoolExecutor if step_engine == "process" else cf.ThreadPoolExecutor
            with Executor(max_workers=step_workers) as pool:
                future_to_i = {}
                for i, p in enumerate(inputs):
                    f = pool.submit(
                        _run_item_step, i, p, step, potato,
                        manifest["items"][i]["artifacts"],
                        globals_ctx, vars_ctx, _item_started,
                        _item_progress_for(i, str(p)) if _item_progress_for else None,
                        not verbose,
                    )
                    future_to_i[f] = i

                # recorded as each file lands, not after the pool has drained.
                # collecting the results first and replaying them afterwards
                # left the step's counter reading 0/N for the entire step --
                # with every file still shown as in flight -- and then jumping
                # straight to N/N. on a transcription that takes minutes per
                # file that looks exactly like a hang, which is the one thing
                # this display exists to rule out.
                for fut in cf.as_completed(future_to_i):
                    i = future_to_i[fut]
                    itm = manifest["items"][i]
                    try:
                        _i, status, new_artifacts, err = fut.result()
                    except Exception as e:
                        itm["status"] = "error"
                        itm["errors"].append(f"Worker crashed in step '{call_name}': {e}")
                        _persist_manifest()
                        emit("item_done", index=idx, item=i, input=str(inputs[i]),
                             status="error", error=str(e))
                        continue

                    if status == "ok":
                        for k, v in (new_artifacts or {}).items():
                            itm["artifacts"][k] = _json_safe(v)
                        if itm["status"] != "error":
                            itm["status"] = "ok"
                    else:
                        itm["status"] = "error"
                        itm["errors"].append(err.get("error", "unknown error"))
                        # written the moment a file fails, rather than waiting
                        # for the step to end. if the rest of the step then
                        # stalls -- and a step where one file has already failed
                        # is a step where that's worth considering -- the
                        # manifest is the only place the reason survives.
                        # errors are rare by definition, so this costs nothing
                        # on a healthy run of any size.
                        _persist_manifest()
                    emit("item_done", index=idx, item=i, input=str(inputs[i]),
                         status=status, error=err.get("error") if err else None)

        elif scope == "global":
            def _progress(done: int, total: Optional[int],
                          message: Optional[str] = None,
                          unit: Optional[str] = None,
                          _idx: int = idx, _call: str = call_name,
                          **extra) -> None:
                """
                The progress sink handed to a step that can count itself.

                ``total=None`` means "still working out how much there is" --
                ``done`` is then a running tally rather than a position.
                ``**extra`` carries newer, optional detail -- ``inflight``,
                the documents a parallel phase is working on right now -- and
                is forwarded whole so the display can grow without this
                function knowing.
                """
                emit("step_progress", index=_idx, call=_call,
                     done=done, total=total, message=message, unit=unit,
                     **extra)

            status, new_globals, err = run_global_step(
                on_progress=_progress,
                workers=workers,
                step=step,
                potato=potato,
                globals_ctx=globals_ctx,
                vars_ctx=vars_ctx,
                manifest_path=out_manifest_path,
                quiet=not verbose,
            )
            if status != "ok":
                say(f"[pipeline] GLOBAL step failed: {err.get('error')}")
                manifest["errors"].append(err.get("error", "unknown error"))
                # persist before bailing out: a failed run is exactly when the
                # manifest is worth reading.
                _persist_manifest()
                emit("step_done", index=idx, call=call_name, status="error",
                     error=err.get("error"))
                break
            for k, v in (new_globals or {}).items():
                globals_ctx[k] = v
                manifest["globals"][k] = _json_safe(v)

        else:
            raise ValueError(f"Invalid scope: {scope}")

        _persist_manifest()
        emit("step_done", index=idx, call=call_name, status="ok")

    _persist_manifest()
    say(f"[pipeline] Manifest written to: {out_manifest_path}")
    emit("run_done", manifest=manifest, manifest_path=str(out_manifest_path))
    return manifest


def summarize_manifest(manifest: Dict[str, Any], *, verbose: bool = True) -> bool:
    """
    Report on a finished run and say whether it was clean.

    Parameters
    ----------
    manifest : dict
        As returned by :func:`run_preset`.
    verbose : bool, default True
        Print the summary. When False, only the return value is produced.

    Returns
    -------
    bool
        True when every item succeeded and no global step failed.
    """
    failed_items = [itm for itm in manifest["items"] if itm.get("status") == "error"]
    ok_items = [itm for itm in manifest["items"] if itm.get("status") == "ok"]
    global_errors = manifest["errors"]

    if verbose:
        if manifest["items"]:
            print(f"[pipeline] Items: {len(ok_items)} ok, {len(failed_items)} failed")
        if failed_items:
            for itm in failed_items[:10]:
                first = itm["errors"][0] if itm["errors"] else "unknown error"
                print(f"[pipeline]   FAILED {itm['input']}: {first}")
            if len(failed_items) > 10:
                print(f"[pipeline]   ...and {len(failed_items) - 10} more (see the manifest)")
        for err in global_errors:
            print(f"[pipeline] GLOBAL ERROR: {err}")

    return not (global_errors or failed_items)


def main():
    """
    Entry point for the Taters Pipeline Runner.

    Responsibilities
    ----------------
    - Parse CLI arguments (`--preset` or `--preset-file`, optional `--vars-file`
      and repeated `--var key=value` overrides, `--workers`, `--quiet`, etc.).
    - Load the preset YAML and merge variables from three sources in order:
        1) preset `vars` block
        2) `--vars-file` (YAML)
        3) repeated `--var` CLI flags
    - Decide whether input discovery is required:
        * If the preset has any ITEM-scoped steps, `--root_dir` is required and
          files are discovered with `discover_inputs(...)`.
        * If there are only GLOBAL steps, discovery is skipped entirely.
    - Build a run manifest skeleton (preset name, inputs, vars, globals).
    - Create a single `Taters()` instance (shared across all steps in the run).
    - Execute each step in order:
        * ITEM steps: fan out across discovered inputs using a thread or process
          pool (configurable per step). A given step reuses one pool for all
          items to amortize worker startup.
        * GLOBAL steps: run once, in order, with a barrier between steps.
    - After each step, update and persist the JSON manifest so long-running runs
      are observable and resumable.
    - Print the final manifest path on completion.

    Concurrency Notes
    -----------------
    - The default executor for ITEM steps is a `ThreadPoolExecutor` (good for
      I/O-bound steps and for GPU inference that releases the GIL).
    - For heavy Python/CPU work, presets may set `engine: process` on a step to
      use a `ProcessPoolExecutor`. In that case, be mindful that a new Python
      process is spawned for each worker (model weights may be reloaded once per
      worker).

    Error Handling
    --------------
    - Individual ITEM step failures do not crash the pipeline; they mark that
      item as `"error"` in the manifest and continue.
    - GLOBAL step failures are terminal for the run (the loop breaks), and the
      manifest is written before bailing out.
    - The process exits with status 1 if anything failed — a global error or any
      individual item — and 0 only when every step succeeded.

    Returns
    -------
    None
        The function exits the process after writing the manifest.
    """
    # ---------------------------
    # CLI
    # ---------------------------
    ap = argparse.ArgumentParser(
        description="Taters Pipeline Runner (robust templating + flexible calls)"
    )
    ap.add_argument("--root_dir", default=None,
                    help="Folder to scan for inputs (required only if preset has ITEM steps)")
    ap.add_argument("--file_type", default="any", choices=["audio", "video", "any"],
                    help="Input type filter for discovery")

    # NOTE: not required here — we enforce after handling list/describe.
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--preset", help="Preset name (taters/pipelines/presets/<name>.yaml)")
    group.add_argument("--preset-file", dest="preset_file", help="Path to preset YAML")

    ap.add_argument("--vars-file", dest="vars_file", help="YAML file with 'vars' overrides")
    ap.add_argument("--var", action="append", default=[], help="Single override key=value (repeatable)")
    ap.add_argument("--workers", type=int, default=None,
                    help="Parallelism budget for the run (files at a time for per-file steps, processes inside text steps). Default: the preset\'s `workers` variable, else one per CPU core.")
    ap.add_argument("--out-manifest", dest="out_manifest", default=None,
                    help="Run manifest (JSON). Default: ./run_manifest.json")
    ap.add_argument("--quiet", dest="verbose", action="store_false", default=True,
                    help="Suppress the step-by-step chatter, including each step's "
                         "own output (transcription prints a line per segment). The "
                         "final summary and the exit code are unaffected")

    # discovery / docs helpers
    ap.add_argument("--list-presets", action="store_true",
                    help="List all discovered presets and exit")
    ap.add_argument("--describe-preset", metavar="NAME",
                    help="Show metadata for a preset (by id or filename) and exit")

    args = ap.parse_args()

    # early-exit helpers (no preset required)
    if args.list_presets:
        _cmd_list_presets()
        sys.exit(0)

    if args.describe_preset:
        _cmd_describe_preset(args.describe_preset)
        sys.exit(0)

    # now we enforce that one of --preset/--preset-file is present
    if not (args.preset or args.preset_file):
        ap.error("one of --preset or --preset-file is required "
                 "unless using --list-presets or --describe-preset")

    # ---------------------------
    # Load preset and vars first
    # ---------------------------
    preset = load_preset_by_name(args.preset) if args.preset else load_yaml_file(Path(args.preset_file))
    steps: List[dict] = preset.get("steps", []) or []
    if not steps:
        raise ValueError("Preset has no steps")

    vars_ctx: Dict[str, Any] = dict(preset.get("vars", {}) or {})
    if args.vars_file:
        vars_ctx = merge_vars(vars_ctx, load_yaml_file(Path(args.vars_file)))
    vars_ctx = merge_vars(vars_ctx, parse_var_overrides(args.var))

    # checked here, not in run_preset(), so the message names the CLI flag the
    # user actually typed rather than the library parameter behind it.
    if not args.root_dir and any(st.get("scope", "item") == "item" for st in steps):
        raise ValueError("--root_dir is required because this preset contains ITEM-scoped steps.")

    manifest = run_preset(
        preset,
        root_dir=args.root_dir,
        file_type=args.file_type,
        vars_ctx=vars_ctx,
        workers=args.workers,
        out_manifest=args.out_manifest,
        preset_name=args.preset or str(args.preset_file),
        verbose=args.verbose,
    )

    # not gated on --quiet, on purpose. this is the outcome, not the chatter:
    # a quiet run still has to say whether it worked, and which files didn't.
    if not summarize_manifest(manifest):
        # exit non-zero so scripts, schedulers and CI can tell a partial run
        # from a clean one. the manifest has the detail.
        sys.exit(1)


if __name__ == "__main__":
    main()
