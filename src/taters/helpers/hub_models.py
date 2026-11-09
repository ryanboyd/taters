"""
A Hugging Face model on this machine, or on its way here.

What the wizard needs to offer a download at the moment a model is chosen,
instead of the first run failing on it minutes later: whether the model is
already here in full, what fetching it would cost, whether this machine is
allowed to fetch it (a gated model needs an approved account and a token),
and the download itself, with a byte count for a progress line.

No questions are asked here; the wizard asks them. Every function is safe
offline: a hub that cannot be reached is an answer (``"offline"``), never
an exception.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["is_downloaded", "DownloadPlan", "plan_download", "access", "save_token",
           "download", "model_page", "CachedRepo", "cached_repos", "pipelines_naming",
           "delete_repos", "old_versions", "delete_old_versions", "export_repo",
           "export_folder_name", "INTERNAL_PARTS", "settle_links"]

#: the files a model is run from; everything else in a repository (the
#: original-format weights, GGUF and ONNX exports, other frameworks' copies)
#: is left on the hub. `.bin` weights are fetched only when no safetensors are
_RUN_FROM = (".json", ".safetensors", ".model", ".txt", ".jinja", ".tiktoken", ".py")


_SETTLED: set = set()
_SETTLE_LOCK = threading.Lock()


def settle_links(cache: Optional[Path] = None) -> None:
    """
    Have the hub library decide, once and on one thread, whether it may make
    links in the model cache, before anything downloads into it.

    It decides the first time a file lands in a cache folder, and it records
    "yes" before it has tried: a model download fetches several files at
    once, a thread that asks in that moment is told yes, and on Windows
    without the privilege to make links its file fails with WinError 1314
    while the others are copied. In a real run that one file failed a 3 GB
    Whisper download, which was then taken for a broken GPU. Settled here
    first, every later thread reads the true answer. Cheap after the first
    call; never raises.
    """
    try:
        from huggingface_hub import constants
        from huggingface_hub import file_download
    except Exception:  # noqa: BLE001 - no hub library, nothing to settle
        return
    are_symlinks_supported = _guard_link_check(file_download)
    folders = {str(Path(constants.HF_HUB_CACHE).expanduser()), str(Path(cache) if cache else _cache())}
    with _SETTLE_LOCK:
        for folder in folders:
            if folder in _SETTLED:
                continue
            try:
                are_symlinks_supported(folder)
            except Exception:  # noqa: BLE001 - an unwritable folder says so when used
                pass
            _SETTLED.add(folder)


def _guard_link_check(file_download):
    """
    The hub library's link check, made to answer one thread at a time.

    Settling the cache folder up front wasn't enough: the library asks again
    about each model's *own* folder (models--org--name) the first time a
    file lands in it, which is in the middle of a download with several
    threads going -- a real chat on Windows died on exactly that, with the
    cache root already settled. So we close the race itself: behind a lock,
    a second thread waits out the first one's test and then reads its true
    answer instead of the "yes" it records before trying. Installed once;
    the hub library's download code calls the module's name, so it gets
    this one. Returns the guarded check.
    """
    current = file_download.are_symlinks_supported
    if getattr(current, "_taters_guard", False):
        return current
    lock = threading.Lock()

    def are_symlinks_supported(cache_dir=None):
        with lock:
            return current(cache_dir)

    are_symlinks_supported._taters_guard = True
    are_symlinks_supported.__wrapped__ = current
    file_download.are_symlinks_supported = are_symlinks_supported
    return are_symlinks_supported


def model_page(hub_id: str) -> str:
    return f"https://huggingface.co/{hub_id}"


def _cache() -> Path:
    from .settings import model_cache_dir

    return Path(model_cache_dir())


def _snapshot(hub_id: str, cache: Path) -> Optional[Path]:
    """The folder of the newest snapshot the cache holds, or None."""
    folder = cache / ("models--" + hub_id.replace("/", "--"))
    ref = folder / "refs" / "main"
    if ref.is_file():
        snap = folder / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return snap
    snaps = sorted((folder / "snapshots").glob("*")) if (folder / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


def is_downloaded(hub_id: str, cache: Optional[Path] = None) -> bool:
    """
    Whether the model can be run from this machine's cache alone: its
    config, its tokenizer, and every weight file its index names. A config
    alone is not enough -- a download stopped halfway leaves one, and that
    is exactly the model the first run would still have to fetch.
    """
    snap = _snapshot(hub_id, Path(cache) if cache else _cache())
    if snap is None or not (snap / "config.json").is_file():
        return False
    if not any((snap / name).is_file() for name in
               ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json")):
        return False
    for single, index in (("model.safetensors", "model.safetensors.index.json"),
                          ("pytorch_model.bin", "pytorch_model.bin.index.json")):
        if (snap / single).is_file():
            return True
        if (snap / index).is_file():
            try:
                shards = set(json.loads((snap / index).read_text(encoding="utf-8"))
                             .get("weight_map", {}).values())
            except (OSError, ValueError):
                return False
            return bool(shards) and all((snap / s).is_file() for s in shards)
    return False


@dataclass
class DownloadPlan:
    """What fetching a model means: the files, their total size, and why
    it cannot be looked up when it cannot (``state`` is ``"ok"``,
    ``"missing"``, ``"offline"``, or ``"unusable"`` for a repository that
    exists but holds nothing a step can run, such as a GGUF export)."""

    hub_id: str
    state: str = "ok"
    files: List[str] = field(default_factory=list)
    size: int = 0
    why: str = ""

    @property
    def gigabytes(self) -> str:
        gb = self.size / 1e9
        return f"{gb:.1f} GB" if gb >= 0.1 else f"{self.size / 1e6:.0f} MB"


def plan_download(hub_id: str, *, token: Optional[str] = None) -> DownloadPlan:
    """The files a run needs and their size, from the hub's listing (which
    a gated model shows to anyone; only the files themselves are locked).
    ``token`` is a sign-in for this call; without one the hub library's
    saved sign-in is used, when there is one."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        info = HfApi(token=token).model_info(hub_id, files_metadata=True)
    except Exception as e:  # noqa: BLE001 - every kind is an answer
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
        return DownloadPlan(hub_id, state="missing" if isinstance(e, RepositoryNotFoundError)
                            else "offline", why=first)
    siblings = [(s.rfilename, int(getattr(s, "size", 0) or 0))
                for s in (getattr(info, "siblings", None) or [])]
    top = [(name, size) for name, size in siblings if "/" not in name]
    has_safetensors = any(name.endswith(".safetensors") for name, _ in top)
    files = [(name, size) for name, size in top
             if name.endswith(_RUN_FROM) or (not has_safetensors and name.endswith(".bin"))]
    weights = [n for n, _ in files if n.endswith((".safetensors", ".bin"))]
    if not weights:
        # a GGUF or ONNX export, or weights in a subfolder: nothing here loads
        # in transformers, and an empty file list would have fetched the
        # whole repository (every multi-GB export in it)
        kinds = sorted({Path(n).suffix for n, _ in siblings if Path(n).suffix})
        return DownloadPlan(hub_id, state="unusable", why=(
            f"{hub_id} holds no weights a step can run (it has {', '.join(kinds[:6]) or 'no files'}"
            f"). A GGUF or ONNX export runs on a server such as Ollama or LM Studio; pick "
            f"the model's transformers repository, or a local server."))
    return DownloadPlan(hub_id, files=[n for n, _ in files], size=sum(s for _, s in files))


def access(hub_id: str) -> str:
    """
    Whether this machine may download the model, with whatever token the
    hub library finds (``HF_TOKEN``, or a saved sign-in): ``"ok"``;
    ``"gated"`` (needs an approved account, and no token is set);
    ``"bad_token"`` (a token is set and the hub does not accept it);
    ``"pending"`` (the token is good, the account is not approved yet);
    ``"missing"``; or ``"offline"``.
    """
    try:
        from huggingface_hub import auth_check, get_token
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover - transformers brings it
        return "offline"
    token = get_token()
    try:
        auth_check(hub_id)
        return "ok"
    except GatedRepoError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if not token:
            return "gated"
        return "bad_token" if status == 401 else "pending"
    except RepositoryNotFoundError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        # with a bad token the hub calls every repository not found
        return "bad_token" if token and status == 401 else "missing"
    except Exception:  # noqa: BLE001
        return "offline"


def save_token(token: str) -> None:
    """Sign this machine in to the hub with ``token``, kept where the hub
    library keeps it, so every later run finds it. Raises ``ValueError``
    when the hub does not accept the token."""
    from huggingface_hub import login

    try:
        login(token=token.strip(), add_to_git_credential=False)
    except Exception as e:  # noqa: BLE001 - a bad token is a sentence
        raise ValueError(str(e).splitlines()[0] if str(e) else type(e).__name__) from None


def _bytes_in(folder: Path) -> int:
    total = 0
    blobs = folder / "blobs"
    if blobs.is_dir():
        for f in blobs.iterdir():
            try:
                total += f.stat().st_size
            except OSError:
                continue
    return total


def download(plan: DownloadPlan, *, on_bytes: Optional[Callable[[int], None]] = None,
             every: float = 0.5, token: Optional[str] = None) -> Path:
    """
    Fetch the plan's files into the model cache, calling ``on_bytes`` with
    the bytes on disk so far every ``every`` seconds. A stopped download
    keeps what arrived and resumes from it. Returns the snapshot folder;
    raises what the hub raised.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    if plan.state != "ok" or not plan.files:
        # never `allow_patterns=None`: that is the whole repository
        raise ValueError(plan.why or f"nothing to download for {plan.hub_id}")
    cache = _cache()
    settle_links(cache)
    folder = cache / ("models--" + plan.hub_id.replace("/", "--"))
    start = _bytes_in(folder)
    result: dict = {}

    def fetch() -> None:
        try:
            result["path"] = snapshot_download(plan.hub_id, allow_patterns=plan.files or None,
                                               cache_dir=str(cache), token=token)
        except BaseException as e:  # noqa: BLE001 - handed back to the caller's thread
            result["error"] = e

    # the hub's own bars write over whatever screen is up; the caller has one
    disable_progress_bars()
    try:
        worker = threading.Thread(target=fetch, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(every)
            if on_bytes is not None:
                on_bytes(max(0, _bytes_in(folder) - start))
    finally:
        enable_progress_bars()
    if "error" in result:
        raise result["error"]
    if on_bytes is not None:
        on_bytes(max(0, _bytes_in(folder) - start))
    return Path(result["path"])



# ---------------------------------------------------------------------------
# What the cache holds
# ---------------------------------------------------------------------------

#: Repositories Taters downloads for its own use inside a step, which nobody
#: picks: what each one does, in words. Matched before anything else, so the
#: punctuation model is not offered as a text encoder for being one.
INTERNAL_PARTS = {
    "kredor/punctuate-all": "puts punctuation back for speaker diarization",
    "MahmoudAshraf/mms-300m-1130-forced-aligner": "times each word for speaker diarization",
    "opencv/face_detection_yunet": "finds faces in video",
    "opencv/face_recognition_sface": "tells faces apart in video",
    "stanfordnlp/stanza-en": "Stanza's English models (tagging and lemmas)",
}
#: Whisper's audio front end, read by the Whisper-embeddings step for the
#: model it runs (``openai/whisper-<size>``): a config file, not a model.
_WHISPER_FRONT = "openai/whisper-"
#: Vision encoders the video-embeddings step can run, by their config's type.
_VISION_TYPES = {"clip", "siglip", "siglip2", "vit", "dinov2", "beit", "convnext"}


@dataclass
class CachedRepo:
    """
    One repository in the model cache, and what it is to Taters.

    ``role`` is ``"step"`` (a model you can pick in the steps named in
    ``uses``), ``"part"`` (downloaded by a step for its own use; ``uses``
    says what for) or ``"other"`` (not used by Taters: another program's
    model, a dataset). ``complete`` is False for a model whose weights are
    not here, which a run would still have to download. ``old_size`` is
    what its leftover versions take: ones no branch points to any more,
    left behind when the model was updated. A version a branch still
    points to is kept even when it is not ``main`` -- transformers fetches
    a model's safetensors from the hub's conversion branch, so that
    version is the one in use (a real cache: CLIP's).
    """

    hub_id: str
    kind: str                       # "model" | "dataset" | "space"
    path: Path
    size: int
    last_used: float
    revisions: List[str]
    current: Optional[str]
    old_size: int
    leftovers: List[str] = field(default_factory=list)
    role: str = "other"
    uses: str = ""
    complete: bool = True

    @property
    def snapshot(self) -> Optional[Path]:
        return self.path / "snapshots" / self.current if self.current else None


def _step_labels(*ids: str) -> str:
    from ..ui import recipes as _recipes

    out = []
    for rid in ids:
        try:
            out.append(_recipes.by_id(rid).label)
        except Exception:  # noqa: BLE001 - a recipe renamed: say nothing wrong
            continue
    return "; ".join(out)


def _whisper_names() -> Dict[str, str]:
    """faster-whisper's own repositories, by the short name a pipeline
    gives (``base.en`` -> ``Systran/faster-whisper-base.en``)."""
    try:
        from faster_whisper.utils import _MODELS  # type: ignore

        return {repo: name for name, repo in dict(_MODELS).items()}
    except Exception:  # noqa: BLE001 - not installed: match by the repo name
        return {}


def _role(hub_id: str, kind: str, config: Optional[dict], picks: Dict[str, str],
          whisper: Dict[str, str]) -> Tuple[str, str]:
    if kind != "model":
        return "other", f"a {kind}, not a model"
    if hub_id in INTERNAL_PARTS:
        return "part", INTERNAL_PARTS[hub_id]
    if hub_id.startswith(_WHISPER_FRONT):
        return "part", "Whisper's audio settings, for Whisper embeddings"
    if hub_id in whisper or "faster-whisper" in hub_id or "faster-distil-whisper" in hub_id:
        return "step", _step_labels("transcribe")
    if hub_id in picks:
        return "step", picks[hub_id]
    if config and str(config.get("model_type", "")) in _VISION_TYPES:
        return "step", _step_labels("video_embeddings")
    return "other", "not used by Taters"


def _pickable() -> Dict[str, str]:
    """Every cached model a step's picker offers, with the steps: the
    pickers' own rules, so this list and theirs cannot disagree."""
    from ..text._generation import cached_hub_generative_models
    from ..text._transformer_common import cached_hub_encoders, cached_hub_sentence_models
    from ..text.hf_classifier import cached_hub_classifiers

    uses: Dict[str, List[str]] = {}
    for finder, steps in (
            (cached_hub_generative_models, ("generate_with_llm",)),
            (cached_hub_sentence_models, ("sentence_embeddings", "archetypes")),
            (cached_hub_encoders, ("transformer_embeddings", "adapt_encoder",
                                   "finetune_text_predictor")),
            (cached_hub_classifiers, ("hf_classifier_apply",))):
        try:
            found = [h for h, _note in finder()]
        except Exception:  # noqa: BLE001 - one broken finder costs only its rows
            found = []
        for hub_id in found:
            uses.setdefault(hub_id, [])
            uses[hub_id] += [s for s in steps if s not in uses[hub_id]]
    return {h: _step_labels(*steps) for h, steps in uses.items()}


def cached_repos(cache: Optional[Path] = None) -> List[CachedRepo]:
    """
    Everything in the model cache, each with its size, when it was last
    used, its versions, and what it is to Taters. The steps' models first,
    then the parts steps download for themselves, then the rest; the
    biggest first within each.
    """
    from huggingface_hub import scan_cache_dir

    root = Path(cache) if cache else _cache()
    if not root.is_dir():
        return []
    info = scan_cache_dir(root)
    picks = _pickable()
    whisper = _whisper_names()
    out: List[CachedRepo] = []
    for repo in info.repos:
        revs = sorted(repo.revisions, key=lambda r: r.last_modified)
        current = next((r for r in revs if "main" in r.refs), revs[-1] if revs else None)
        config = None
        if current is not None and (current.snapshot_path / "config.json").is_file():
            try:
                config = json.loads((current.snapshot_path / "config.json")
                                    .read_text(encoding="utf-8"))
            except (OSError, ValueError):
                config = None
        role, uses = _role(repo.repo_id, repo.repo_type, config, picks, whisper)
        leftovers = [r for r in revs if not r.refs and r is not current]
        entry = CachedRepo(
            hub_id=repo.repo_id, kind=repo.repo_type, path=Path(repo.repo_path),
            size=int(repo.size_on_disk), last_used=float(repo.last_accessed or 0),
            revisions=[r.commit_hash for r in revs],
            current=current.commit_hash if current else None,
            old_size=sum(int(r.size_on_disk) for r in leftovers),
            leftovers=[r.commit_hash for r in leftovers],
            role=role, uses=uses)
        if role == "step" and config is not None and config.get("model_type"):
            # a transformers model: its weights, in any version, or a run
            # downloads them (a Whisper or ONNX model has no such config)
            entry.complete = _has_weights([r.snapshot_path for r in revs])
        out.append(entry)
    order = {"step": 0, "part": 1, "other": 2}
    return sorted(out, key=lambda e: (order[e.role], -e.size, e.hub_id.lower()))


def _has_weights(snapshots: Sequence[Path]) -> bool:
    """Whether any of a repository's versions holds its weights in full."""
    for snap in snapshots:
        snap = Path(snap)
        for single, index in (("model.safetensors", "model.safetensors.index.json"),
                              ("pytorch_model.bin", "pytorch_model.bin.index.json")):
            if (snap / single).is_file():
                return True
            if (snap / index).is_file():
                try:
                    shards = set(json.loads((snap / index).read_text(encoding="utf-8"))
                                 .get("weight_map", {}).values())
                except (OSError, ValueError):
                    continue
                if shards and all((snap / x).is_file() for x in shards):
                    return True
    return False


def pipelines_naming(repo: CachedRepo, root: Optional[Path] = None) -> List[str]:
    """
    The saved pipelines that name this model anywhere in their settings,
    by title, the shipped ones marked: the ones that would download it
    again after a delete. A Whisper model is also named by its short name
    (``base.en``). ``root`` is where the person's own pipelines are looked
    for, as the run menu does.
    """
    from ..pipelines.run_pipeline import available_presets, is_builtin_preset

    names = {repo.hub_id}
    short = _whisper_names().get(repo.hub_id)
    if short:
        names.add(short)

    def mentions(node: Any) -> bool:
        if isinstance(node, str):
            text = node.strip()
            return text in names or any(text.startswith(n + "#") for n in names)
        if isinstance(node, dict):
            return any(mentions(v) for v in node.values())
        if isinstance(node, list):
            return any(mentions(v) for v in node)
        return False

    import yaml

    found = []
    for path, meta in available_presets(root):
        try:
            doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if mentions(doc):
            title = str(meta.get("title") or Path(path).stem)
            found.append(f"{title} (built in)" if is_builtin_preset(Path(path)) else title)
    return found


def delete_repos(repos: Sequence[CachedRepo], cache: Optional[Path] = None) -> int:
    """Remove these repositories from the cache, every version; returns the
    bytes freed. The hub library does the deleting, the way its own
    ``hf cache delete`` does."""
    from huggingface_hub import scan_cache_dir

    hashes = [h for r in repos for h in r.revisions]
    if not hashes:
        return 0
    plan = scan_cache_dir(Path(cache) if cache else _cache()).delete_revisions(*hashes)
    freed = int(plan.expected_freed_size)
    plan.execute()
    return freed


def old_versions(repos: Sequence[CachedRepo]) -> List[Tuple[CachedRepo, str]]:
    """Every leftover version (see :class:`CachedRepo`), with its repository."""
    return [(r, h) for r in repos for h in r.leftovers]


def delete_old_versions(repos: Sequence[CachedRepo], cache: Optional[Path] = None) -> int:
    """Remove the older versions beside each repository's current one;
    returns the bytes freed."""
    from huggingface_hub import scan_cache_dir

    hashes = [h for _r, h in old_versions(repos)]
    if not hashes:
        return 0
    plan = scan_cache_dir(Path(cache) if cache else _cache()).delete_revisions(*hashes)
    freed = int(plan.expected_freed_size)
    plan.execute()
    return freed


def export_folder_name(repo: CachedRepo, dest: Path) -> Path:
    """Where an export of this repository goes in ``dest``: its own name,
    with the owner added when that name is taken, then a number."""
    name = repo.hub_id.split("/")[-1]
    candidates = [name, repo.hub_id.replace("/", "--")]
    for c in candidates:
        if not (dest / c).exists():
            return dest / c
    n = 2
    while (dest / f"{candidates[-1]}-{n}").exists():
        n += 1
    return dest / f"{candidates[-1]}-{n}"


def export_repo(repo: CachedRepo, target: Path, *,
                on_bytes: Optional[Callable[[int], None]] = None) -> Path:
    """
    Copy the current version of a repository into ``target`` as a plain
    model folder: real files, no links into the cache, so it loads from
    its path anywhere (``from_pretrained(target)``, or the path typed into
    a Taters step) and can be carried to a machine with no internet. A
    copy that fails leaves nothing half-written behind.
    """
    import shutil

    snap = repo.snapshot
    if snap is None or not snap.is_dir():
        raise FileNotFoundError(f"{repo.hub_id} has no files in the cache")
    staging = target.with_name(target.name + ".copying")
    if staging.exists():
        shutil.rmtree(staging)
    done = 0
    try:
        for src in sorted(snap.rglob("*")):
            rel = src.relative_to(snap)
            out = staging / rel
            if src.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.resolve(), out)      # follow the cache's links
            done += out.stat().st_size
            if on_bytes is not None:
                on_bytes(done)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
