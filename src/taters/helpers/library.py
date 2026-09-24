"""
The library: user-imported assets that outlive any one project.

A "dictionary" here is the general term -- a content-coding dictionary and an
archetype dictionary are different *kinds*, with different formats, consumed by
different modules. The library stores each kind in its own folder under the
user's home, so importing a dictionary once makes it available to every
pipeline on the machine, wherever it is run from. That replaces the old
convention of a ``dictionaries/`` folder relative to the working directory,
which silently failed for anyone who did not happen to have one.

No UI imports (the :mod:`taters.helpers.gpu` pattern): the wizard is one
consumer, but the runner or a future GUI can read the same library. All the
operations are deliberately file-level and boring -- an entry *is* its file,
its name *is* the filename stem -- so a user can also just manage the folder by
hand and nothing here will be surprised.

Adding a new kind -- pretrained classifier models are the expected next one --
is a :data:`KINDS` entry plus a ``library=`` line on the recipe that consumes
it; the manager, the picker, and the empty-library contingency in the wizard
all key off those two declarations.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Set, Tuple

from .atomic import SCRATCH_SUFFIX, atomic_write

__all__ = [
    "KINDS", "LibraryKind", "LibraryCollision", "kind_by_id",
    "library_home", "kind_dir", "entries", "display_name", "expand",
    "import_into", "export_to", "rename", "delete", "asset_problem",
]


@dataclass(frozen=True)
class LibraryKind:
    """
    One category of importable asset.

    Attributes
    ----------
    id : str
        Stable identifier; also the folder name under the library.
    label, help : str
        What menus call it, and one sentence including the accepted formats.
    suffixes : tuple of str
        File extensions this kind accepts, lowercase with the dot.
    """

    id: str
    label: str
    help: str
    suffixes: Tuple[str, ...]
    #: The kind's real reader, asked at import time: "why can you not load this
    #: file?", answered with "" when it loads. Optional because not every kind
    #: has an affordable one -- loading an archetype dictionary drags in a
    #: sentence-transformer model, which is not a price an import screen pays.
    deep_check: Optional[Callable[[Path], str]] = None
    #: How to label one entry, when the filename is not enough. Saved models
    #: are all `.json` and are not interchangeable -- a ridge cannot score a
    #: MEM model's themes -- so their rows say which kind each one is, and
    #: what columns it will add. Returns ``(label, note)``; the label
    #: replaces the file stem and the note replaces the size.
    describe_entry: Optional[Callable[[Path], Tuple[str, str]]] = None


def _contentcoder_problem(path: Path) -> str:
    """
    Ask the parser that will actually read this file at run time.

    This is what makes the import gate exactly as strict as reality: the
    cheap shape heuristics in :func:`asset_problem` pass files that
    contentcoder still rejects (an empty ``%%`` header block, a word line
    referencing a category number the header never defined), and the only
    authority on what loads is the loader. The construction here is
    byte-for-byte the one ``multi_dict_analyzer._load_coders`` performs.

    Returns "" when the file loads -- and also when contentcoder is not
    installed in this environment, because a gate that refuses everything the
    moment an optional dependency is missing is worse than the heuristics
    alone. In that case the heuristic verdict simply stands.
    """
    try:
        from contentcoder.ContentCoder import ContentCoder
    except Exception:
        return ""
    import contextlib
    import io

    try:
        # contentcoder prints "Dictionary loaded." to stdout on success, which
        # would land smack in the middle of whatever screen is asking. so we
        # muzzle it.
        with contextlib.redirect_stdout(io.StringIO()):
            ContentCoder(dicFilename=str(path), fileEncoding="utf-8-sig")
    except Exception as e:
        return (
            f"'{path.name}' looks like a dictionary but the dictionary parser "
            f"could not load it ({type(e).__name__}: {e}). Fix the file "
            "first -- imported as-is, it would only be skipped at run time."
        )
    return ""


#: The row BUTTER's weighted-dictionary plugin put at the top of every norm
#: file: a per-category constant it added to the score. Taters does not do
#: that -- it reports the mean rating and nothing else -- so a file still
#: carrying one was built for different arithmetic than the one that will be
#: applied to it.
INTERCEPT_TERM = "_intercept"


def _norms_problem(path: Path) -> str:
    """
    A norm set has to load like a dictionary *and* carry no intercept row.

    The loader check is the same one content-coding dictionaries get, because
    the file format is the same. The extra rule is about arithmetic: an
    ``_intercept`` row means the weights were fitted as part of a linear model
    with a constant term, and scoring it as a plain mean silently drops that
    term. Better to refuse the file than to publish the wrong number.
    """
    problem = _contentcoder_problem(path)
    if problem:
        return problem

    import csv as _csv

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as fh:
            for row in _csv.reader(fh):
                if row and row[0].strip().lower() == INTERCEPT_TERM:
                    return (
                        f"'{path.name}' still has an '{INTERCEPT_TERM}' row. "
                        "That is a constant from a fitted linear model, and "
                        "norm scoring reports the mean rating instead, so the "
                        "constant would be quietly dropped. Delete the row if "
                        "the weights stand on their own, or keep the file as a "
                        "saved model rather than a norm set."
                    )
    except Exception:
        # unreadable for some other reason is the loader check's business,
        # and it already passed, so we do not invent a second complaint here
        return ""
    return ""


#: The module whose `_load_model` vets each kind of model at import time --
#: the only authority on what can score. A test pins this to
#: `model_spec.MODEL_TYPES`, because an unregistered id used to pass the
#: gate silently.
MODEL_LOADERS: Dict[str, Tuple[str, str]] = {
    "ridge": ("..stats.ridge", "taters.stats.ridge"),
    "classifier": ("..stats.classify", "taters.stats.classify"),
    "mem": ("..text.topic_model_mem", "taters.text.topic_model_mem"),
    "lda": ("..text.topic_model_lda", "taters.text.topic_model_lda"),
    "nmf": ("..text.topic_model_nmf", "taters.text.topic_model_nmf"),
    "word_vectors": ("..text.word_vectors", "taters.text.word_vectors"),
    "text_predictor": ("..text.finetune_predictor", "taters.text.finetune_predictor"),
    "hf_classifier": ("..text.hf_classifier", "taters.text.hf_classifier"),
}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _encoder_problem(path: Path) -> str:
    """Why this file cannot be used as a text encoder, in words."""
    from .model_spec import encoder_problem

    return encoder_problem(path)


def _describe_encoder(path: Path) -> Tuple[str, str]:
    from .model_spec import describe_encoder

    label, note = describe_encoder(path)
    return label, f"{note} · {_human_size(payload_size(path))}"


def _model_problem(path: Path) -> str:
    """
    Why this file cannot be used as a model, in its own loader's words.

    Two gates, in order. First: is it a model at all, and of a kind this
    build can score? Second: delegate to the loader that will actually read
    it, because the only authority on what can score is the thing that does
    the scoring -- the same posture as the dictionary gate, and the reason
    the import screen is exactly as strict as run time rather than
    approximately.
    """
    from .model_spec import UnknownModel, describe

    try:
        info = describe(path)
    except UnknownModel as e:
        return str(e)
    except Exception as e:
        return f"{path.name} could not be read as a model ({e})."
    module = MODEL_LOADERS.get(info.type_id)
    if module is None:              # pragma: no cover - registry mismatch
        return ""
    import importlib

    try:
        loader = importlib.import_module(module[1])._load_model
    except Exception:               # pragma: no cover - import failure
        return ""
    try:
        loader(path)
    except Exception as e:
        return str(e)
    return ""


def _describe_model(path: Path) -> Tuple[str, str]:
    """
    One model's row: what it is called, what kind it is, what it will add.

    Every model file is a `.json`, so a listing of file stems and sizes says
    nothing a person can choose from -- and choosing wrong means a ridge
    being handed a topic model. The kind goes in brackets after the name,
    which is how the user asked for it, and the note names the columns the
    model will write, because that is the thing they are about to be stuck
    with in their results.
    """
    from .model_spec import describe

    try:
        info = describe(path)
    except Exception:
        return path.stem, f"{path.suffix} · not a readable model"
    if info.bulk_outputs and info.n_outputs > 3:
        adds = f"{info.n_outputs} columns, {info.outputs[0]}…"
    else:
        adds = ", ".join(info.outputs[:4]) + \
            ("…" if info.n_outputs > 4 else "")
    note = f"adds {adds}"
    if info.needs_tables:
        # we show this before the model gets chosen, not after the run fails:
        # the whole point is that picking it commits you to extracting these.
        note += f" · needs {', '.join(info.needs_tables)}"

    # can this model measure its own features? we want to say so right when we
    # offer it, not have the user find out an hour into a run. a model that can
    # brings its own settings along, so picking it is enough; one that can't is
    # going to refuse later, and the reason belongs here, where the user can
    # still go pick a different model.
    from .model_spec import feature_plan

    try:
        plan = feature_plan(info)
    except Exception:
        return info.display(), note
    if plan.replayable:
        note += " · measures its own features"
    elif plan.problems and info.needs == "features":
        note += " · cannot reproduce its features"
    if payload_of(path):
        note += f" · {_human_size(payload_size(path))}"
    return info.display(), note


KINDS: Dict[str, LibraryKind] = {
    "dictionaries": LibraryKind(
        id="dictionaries",
        label="Content-coding dictionaries",
        help="LIWC-format word-counting dictionaries: .dic, .dicx, or .csv.",
        suffixes=(".dic", ".dicx", ".csv"),
        deep_check=_contentcoder_problem,
    ),
    "norms": LibraryKind(
        id="norms",
        label="Word norms",
        help="Word rating tables -- concreteness, valence, age of acquisition "
             "and the like (.csv, one term per row with a rating per column). "
             "Scored as the average rating of the words that had one, which is "
             "not the same thing a content-coding dictionary does, which is "
             "why they live apart.",
        suffixes=(".csv",),
        deep_check=_norms_problem,
    ),
    "archetypes": LibraryKind(
        id="archetypes",
        label="Archetype dictionaries",
        help="Archetype definition files for similarity scoring: .csv.",
        suffixes=(".csv",),
    ),
    "models": LibraryKind(
        id="models", label="Saved models",
        help="Models written by an earlier run or trained from the main "
             "menu: topic models, prediction models and classifiers, word "
             "vectors, fine-tuned text predictors (*.json, with any weights "
             "beside it), and classifiers from the Hugging Face hub. Import "
             "one to score a new dataset with the same fitted weights. They "
             "live together because 'score this dataset with a model I "
             "already have' is one thing to want, and each file says which "
             "kind it is.",
        suffixes=(".json",),
        deep_check=_model_problem,
        describe_entry=_describe_model,
    ),
    "encoders": LibraryKind(
        id="encoders", label="Text encoders",
        help="Language models adapted to your own texts, or trained from "
             "scratch on them, under 'Wrangle Language Models' (*.json plus "
             "its weights folder). Pick one as the base for transformer "
             "embeddings or for fine-tuning a predictor.",
        suffixes=(".json",),
        deep_check=_encoder_problem,
        describe_entry=_describe_encoder,
    ),
    "connectives": LibraryKind(
        id="connectives",
        label="Connectives lists",
        help="Cohesion connective categories (.txt, one word or phrase per "
             "line, optional TAB + POS-tag constraint). Each file becomes "
             "one cohesion column named by its filename.",
        suffixes=(".txt",),
    ),
    "stoplists": LibraryKind(
        id="stoplists",
        label="Stop word lists",
        help="Plain .txt, one word or character per line. Built-in lists "
             "cover 22 languages plus punctuation characters.",
        suffixes=(".txt",),
    ),
}


class LibraryCollision(Exception):
    """An import would overwrite an entry that already exists.

    Raised instead of overwriting so the UI can ask replace-or-rename; the
    existing path rides along as ``.existing``.
    """

    def __init__(self, existing: Path):
        self.existing = existing
        super().__init__(f"'{existing.stem}' is already in the library")


def kind_by_id(kind_id: str) -> LibraryKind:
    """One kind, or a KeyError that names the valid ids (it is always a typo)."""
    try:
        return KINDS[kind_id]
    except KeyError:
        raise KeyError(
            f"unknown library kind {kind_id!r}. Known: {', '.join(sorted(KINDS))}"
        ) from None


def library_home() -> Path:
    """
    Where the library lives: ``$TATERS_HOME/library``, or ``~/.taters/library``.

    The environment override exists so tests -- and anyone who wants their
    library on a different drive -- can move the whole thing without patching.
    """
    base = os.environ.get("TATERS_HOME")
    return (Path(base) if base else Path.home() / ".taters") / "library"


def kind_dir(kind: LibraryKind) -> Path:
    """
    This kind's folder, created and kept level with what the package ships.

    Seeding used to happen only when the folder did not exist, which meant a
    release that added a dictionary reached new users and nobody else. So it
    now reconciles file by file against a ledger of what we have installed
    before (see :func:`_seed`), and an upgrade brings its new and corrected
    built-ins to a library that has been in use for years.

    What it will never do is undo a decision: a built-in the user deleted
    stays deleted, and a copy they edited stays edited. An empty library is a
    valid state the UI explains, not one this quietly "repairs."
    """
    target = library_home() / kind.id
    target.mkdir(parents=True, exist_ok=True)
    _seed(kind, target)
    return target


def _shipped_dir(kind: LibraryKind) -> Path:
    """Where the package's built-in assets for this kind live, if any ship."""
    return Path(__file__).resolve().parent.parent / "data" / "library" / kind.id


#: Where we record what we have installed, beside the kind folders rather than
#: inside one: the `models` kind's own suffix is ``.json``, so a ledger living
#: in that folder would list itself as a saved model.
_LEDGER_NAME = ".seeded.json"

#: Recorded instead of a digest when the name was already taken by something
#: we did not put there. It marks the name as the user's for good: we neither
#: overwrite it now nor "update" it in a later release.
_USER_OWNED = "user"


def _install(src: Path, dest: Path) -> None:
    """
    Copy a built-in into the library so that an interruption cannot leave a
    half-written one behind.

    ``shutil.copy2`` straight onto ``dest`` writes in place: stop it midway --
    Ctrl-C on a slow first run, a full disk, a laptop lid -- and what is left
    is a truncated dictionary sitting under a name that says it is fine. It
    would parse as far as the cut and then fail in the middle of somebody's
    analysis. Writing beside it and renaming means ``dest`` is either the old
    file or the whole new one, never half of either.
    """
    scratch = dest.with_name(dest.name + SCRATCH_SUFFIX)
    try:
        shutil.copy2(src, scratch)
        os.replace(scratch, dest)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _shipped_fingerprint(shipped: Path, kind: LibraryKind) -> str:
    """
    A cheap "has anything shipped changed?" stamp: every built-in's name and
    size, one ``stat`` apiece and not a byte read.

    This is what keeps the reconcile off the hot path. ``kind_dir`` is called
    on every library access, and hashing 13 MB of dictionaries each time would
    be absurd; hashing happens only when this stamp moves, which is to say
    once per release that touches the built-ins.
    """
    parts = [f"{f.name}:{f.stat().st_size}"
             for f in sorted(shipped.iterdir())
             if f.suffix.lower() in kind.suffixes]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _read_ledger() -> Dict[str, dict]:
    try:
        return json.loads((library_home() / _LEDGER_NAME).read_text("utf-8"))
    except Exception:
        # unreadable or not there yet. a missing ledger is the normal state
        # for a library built before this existed, and a corrupt one is not
        # worth refusing to start over -- both just mean "work it out again"
        return {}


def _write_ledger(ledger: Dict[str, dict]) -> None:
    
    try:
        with atomic_write(library_home() / _LEDGER_NAME, encoding="utf-8") as fh:
            json.dump(ledger, fh, indent=1, sort_keys=True)
    except Exception:
        # a read-only or full home must not stop the app from running. the
        # cost of failing here is that we reconcile again next time
        pass


def _seed(kind: LibraryKind, target: Path) -> None:
    """
    Bring this kind's folder level with what the package ships.

    Four cases, one per file, decided against the ledger:

    * **never installed** -- copy it in. If the name is already taken, that
      file is the user's and we say so in the ledger rather than clobber it.
    * **installed, now missing** -- they deleted it. Leave it deleted.
    * **installed, edited since** -- their copy now. Leave it alone.
    * **installed, untouched, and we ship a different version** -- update it.

    The adoption in the first case matters more than it looks: every library
    that predates this ledger has built-in stoplists and connectives sitting
    in it with nothing recorded, and a file whose bytes are exactly what we
    ship is plainly one of ours, not something the user wrote.

    It only reaches so far, and deliberately. With no ledger AND a file we
    have *also* changed in the same release, "their edit" and "our old
    version" are the same evidence, and we call it theirs -- an update nobody
    receives beats an edit nobody can get back.
    """
    shipped = _shipped_dir(kind)
    if not shipped.is_dir():
        return
    ledger = _read_ledger()
    mine = ledger.setdefault(kind.id, {})
    fingerprint = _shipped_fingerprint(shipped, kind)
    if mine.get("fingerprint") == fingerprint:
        return
    installed: Dict[str, str] = mine.setdefault("files", {})

    for f in sorted(shipped.iterdir()):
        if f.suffix.lower() not in kind.suffixes:
            continue
        here = target / f.name
        was = installed.get(f.name)
        if was is None:
            if not here.exists():
                _install(f, here)
                installed[f.name] = _digest(f)
            else:
                ours = _digest(f)
                installed[f.name] = ours if _digest(here) == ours else _USER_OWNED
            continue
        if was == _USER_OWNED or not here.exists():
            continue
        if _digest(here) != was:
            continue
        fresh = _digest(f)
        if fresh != was:
            _install(f, here)
            installed[f.name] = fresh

    mine["fingerprint"] = fingerprint
    _write_ledger(ledger)


def entries(kind: LibraryKind) -> List[Path]:
    """Every entry of this kind, sorted by name. Only the kind's own formats
    count -- a stray .txt dropped into the folder by hand is ignored, not an
    error."""
    folder = kind_dir(kind)
    carried = payload_dirs(folder)
    return sorted(
        (f for f in folder.iterdir()
         if f.is_file() and f.suffix.lower() in kind.suffixes
         and f not in carried),
        key=lambda f: f.name.lower(),
    )


def display_name(path: Path) -> str:
    """How an entry is named in menus: the filename stem."""
    return path.stem


def expand(kind: LibraryKind, values) -> List[Path]:
    """
    Resolve a mixed list of files and folders to this kind's files.

    A folder means "everything of this kind inside it, recursively" -- the
    reading the analyzers give a folder path, and what the wizard's default
    writes into a preset (the kind's whole library folder). Any screen seeded
    through this shows exactly what a run would use; seeding from the raw
    values made the picker intersect a folder path against entry file paths
    and open with everything unticked while the settings row said "all 10".
    """
    out: List[Path] = []
    for v in values:
        p = Path(v)
        if p.is_dir():
            out.extend(f for f in _files_under(p)
                       if f.suffix.lower() in kind.suffixes)
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Payloads: the weights a model carries beside its manifest
# ---------------------------------------------------------------------------
#
# a trained model is a small JSON that says what it is, plus weights that
# have no business living in a JSON: a word-vector matrix (`.npy`), a
# transformer checkpoint (a folder). the manifest names them under
# ``"payload"`` as sibling names, and we treat the whole set as ONE library
# entry -- imported, exported, renamed and deleted together, never listed
# apart. `ModelType` used to declare a `payload` glob that nothing ever
# actually read; now the manifest's own word is the one and only mechanism,
# because encoders aren't `ModelType`s and need the same treatment.

def payload_of(entry: Path) -> List[Path]:
    """
    The sibling files and folders a ``.json`` entry declares as its payload.

    Empty for anything that is not a JSON manifest with a ``payload`` list.
    Names are taken as siblings only: a name with a path separator, or a
    dot entry, is ignored rather than allowed to point outside the folder.
    """
    entry = Path(entry)
    if entry.suffix.lower() != ".json" or not entry.is_file():
        return []
    try:
        doc = json.loads(entry.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    names = doc.get("payload") if isinstance(doc, dict) else None
    if not isinstance(names, list):
        return []
    out: List[Path] = []
    for name in names:
        text = str(name)
        if not text or "/" in text or "\\" in text or text in (".", ".."):
            continue
        out.append(entry.with_name(text))
    return out


def payload_dirs(folder: Path) -> Set[Path]:
    """Every payload path declared by the manifests directly in ``folder``."""
    folder = Path(folder)
    if not folder.is_dir():
        return set()
    carried: Set[Path] = set()
    for manifest in folder.glob("*.json"):
        carried.update(payload_of(manifest))
    return carried


def _files_under(folder: Path) -> List[Path]:
    """Every file under ``folder``, recursively, skipping the inside of any
    payload -- a checkpoint folder holds a ``config.json`` and a
    ``tokenizer.json`` that are not library entries."""
    folder = Path(folder)
    carried = payload_dirs(folder)
    out: List[Path] = []
    for f in sorted(folder.rglob("*")):
        if not f.is_file():
            continue
        if f in carried or any(c in f.parents for c in carried):
            continue
        # a payload declared by a manifest deeper down the tree.
        if any(f in payload_dirs(parent) or
               any(c in f.parents for c in payload_dirs(parent))
               for parent in f.parents if parent != folder
               and folder in parent.parents):
            continue
        out.append(f)
    return out


def model_files(folder: Path) -> List[Path]:
    """
    The ``.json`` manifests under a folder, without the ones inside a payload.

    What a step handed "a folder" resolves to, and what the finish screen
    scans a whole run folder with. Only ``.json`` files are walked -- a run
    over thousands of recordings holds tens of thousands of WAVs and CSVs,
    and listing every one of them to find a handful of manifests made the
    finish screen wait on a network drive. A payload declared by any
    manifest found is pruned, wherever in the tree it sits.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    candidates = sorted(f for f in folder.rglob("*.json") if f.is_file())
    carried: Set[Path] = set()
    for f in candidates:
        carried.update(payload_of(f))
    return [f for f in candidates
            if f not in carried and not any(c in f.parents for c in carried)]


def _payload_destinations(src: Path, dest: Path) -> List[Tuple[Path, Path]]:
    """Where each payload of ``src`` goes when the manifest lands at ``dest``:
    a payload named after the source stem follows the new stem."""
    pairs = []
    for payload in payload_of(src):
        name = payload.name
        if name.startswith(src.stem):
            name = dest.stem + name[len(src.stem):]
        pairs.append((payload, dest.with_name(name)))
    return pairs


def _rewrite_payload_names(manifest: Path, renamed: Mapping[str, str]) -> None:
    """Point a copied or renamed manifest at its payloads' new names."""
    if not renamed:
        return
    from .atomic import atomic_write

    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["payload"] = [renamed.get(str(n), str(n)) for n in doc.get("payload") or []]
    with atomic_write(manifest, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


def _copy_payload(src: Path, dest: Path) -> None:
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def _remove_payload(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def payload_size(entry: Path) -> int:
    """Bytes of an entry and everything it carries."""
    total = 0
    for p in [Path(entry)] + payload_of(entry):
        try:
            if p.is_dir():
                total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            else:
                total += p.stat().st_size
        except OSError:
            continue
    return total


def find_payload(kind: LibraryKind, name: str, digests: Mapping[str, str]) -> Optional[Path]:
    """
    A payload in the library whose files match the digests a manifest records.

    A model carried into another run travels as its manifest text only (see
    ``pipelines.run_pipeline._materialize_assets``); the weights stay in the
    library. The manifest records ``payload_digests`` -- sha256 per file --
    so the loader can find them here and prove they are the same bytes. The
    match is by content, not by name: a renamed library entry still counts.
    ``name`` is the payload's suffix-bearing name, used to skip payloads of
    the wrong shape quickly.
    """
    wanted = {str(k): str(v) for k, v in (digests or {}).items()}
    if not wanted:
        return None
    suffix = Path(name).suffix
    for manifest in entries(kind):
        for payload in payload_of(manifest):
            if suffix and payload.suffix != suffix:
                continue
            if not payload.exists():
                continue
            if _payload_matches(payload, wanted):
                return payload
    return None


def _payload_matches(payload: Path, wanted: Mapping[str, str]) -> bool:
    from .provenance import file_digest

    try:
        if payload.is_dir():
            return all((payload / rel).is_file()
                       and file_digest(payload / rel) == sha
                       for rel, sha in wanted.items())
        # a single-file payload records itself under its own name, or under
        # the lone key "" if the writer didn't know its name yet.
        sha = next(iter(wanted.values()))
        return file_digest(payload) == sha
    except OSError:
        return False


def asset_problem(path: Path, kind: Optional[LibraryKind] = None) -> str:
    """
    Why this file cannot work as a library asset, or "" when it can.

    The canonical check, shared by import (refuse the file while the user is
    holding it and can act) and by the analyzers (a file can still arrive by
    CLI path without ever passing through the library). Two layers:

    * cheap shape heuristics, always: an empty file is not a dictionary, and a
      ``.dic`` with no ``%...%`` category header is a bare word list -- the
      mistake that made contentcoder die with a bare "list index out of range"
      mid-run, an hour after it was made;
    * with a ``kind``, that kind's :attr:`~LibraryKind.deep_check` -- the real
      parser. Import passes the kind; the analyzers do not, because they
      construct the real parser on the very next line and asking it twice
      buys nothing.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError as e:
        return f"'{path.name}' could not be read: {e}"
    if not text.strip():
        return f"'{path.name}' is empty -- not a dictionary."
    if path.suffix.lower() == ".dic" and text.count("%") < 2:
        return (
            f"'{path.name}' has no %...% category header, so it is a plain "
            "word list rather than a LIWC .dic. Add the header block "
            "(%, one 'number<TAB>name' per category, %) first."
        )
    if kind is not None and kind.deep_check is not None:
        return kind.deep_check(path)
    return ""


def import_into(kind: LibraryKind, src: Path, *, replace: bool = False) -> Path:
    """
    Copy a file into the library.

    Raises
    ------
    ValueError
        For a format the kind does not accept -- with the formats it does,
        because "wrong extension" without the right ones is a dead end -- or
        for a file the kind's own parser cannot load (see
        :func:`asset_problem`), with the parser's reason.
    LibraryCollision
        When an entry with this name exists and ``replace`` is False. The UI
        turns this into a replace-or-keep question rather than deciding here.
    """
    src = Path(src)
    if src.suffix.lower() not in kind.suffixes:
        raise ValueError(
            f"{src.name} is not a {kind.label.lower()} file "
            f"(accepted: {', '.join(kind.suffixes)})"
        )
    problem = asset_problem(src, kind)
    if problem:
        # we refuse right here, while the user still has the file in hand and
        # can fix it -- if we let it in, it blows up a run an hour later instead.
        raise ValueError(problem)
    missing = [p.name for p in payload_of(src) if not p.exists()]
    if missing:
        raise ValueError(
            f"{src.name} names weights that are not beside it "
            f"({', '.join(missing)}). Copy the model file and its weights "
            f"together, or re-run the step that made them.")
    dest = kind_dir(kind) / src.name
    pairs = _payload_destinations(src, dest)
    if not replace:
        for taken in [dest] + [d for _s, d in pairs]:
            if taken.exists():
                raise LibraryCollision(dest)
    shutil.copy2(src, dest)
    try:
        for payload, target in pairs:
            _copy_payload(payload, target)
        _rewrite_payload_names(dest, {s.name: d.name for s, d in pairs
                                      if s.name != d.name})
    except BaseException:
        # half an entry is worse than none: a manifest with no weights would get
        # listed, picked, and then refused at run time.
        dest.unlink(missing_ok=True)
        for _payload, target in pairs:
            _remove_payload(target)
        raise
    return dest


def export_to(kind: LibraryKind, name: str, dest_dir: Path,
              *, replace: bool = False) -> Path:
    """
    Copy an entry out, keeping its filename. Returns the new path.

    Raises
    ------
    LibraryCollision
        When the destination file already exists and ``replace`` is False --
        the same contract as :func:`import_into`, for the same reason: a copy
        that silently overwrites is a delete nobody asked for.
    """
    src = _entry(kind, name)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    pairs = _payload_destinations(src, dest)
    if not replace:
        for taken in [dest] + [d for _s, d in pairs]:
            if taken.exists():
                raise LibraryCollision(dest)
    shutil.copy2(src, dest)
    for payload, target in pairs:
        _copy_payload(payload, target)
    return dest


def rename(kind: LibraryKind, old: str, new: str) -> Path:
    """
    Rename an entry, keeping its suffix.

    The suffix carries the *format*, which renaming must not be able to lie
    about -- so a new name arriving with an extension has it stripped rather
    than honored.
    """
    src = _entry(kind, old)
    if "/" in new or "\\" in new:
        # Path(new).stem quietly keeps only what follows the last separator:
        # "a/b" became "b", and a name ending "[/]" became "]". we found this
        # one by fuzzing; better to refuse than to mangle.
        raise ValueError(
            f"a name cannot contain '/' or '\\' (got {new!r})"
        )
    stem = Path(new).stem.strip()
    if not stem:
        raise ValueError("the new name is empty")
    dest = src.with_name(stem + src.suffix)
    pairs = _payload_destinations(src, dest)
    if dest != src:
        for taken in [dest] + [d for _s, d in pairs if d != _s]:
            if taken.exists():
                raise LibraryCollision(dest)
    src.rename(dest)
    for payload, target in pairs:
        if target != payload and payload.exists():
            payload.rename(target)
    _rewrite_payload_names(dest, {s.name: d.name for s, d in pairs
                                  if s.name != d.name})
    return dest


def delete(kind: LibraryKind, name: str) -> None:
    """Remove an entry and the weights it carries. Permanent, so the UI
    confirms before calling this."""
    entry = _entry(kind, name)
    payloads = payload_of(entry)
    entry.unlink()
    for payload in payloads:
        _remove_payload(payload)


def _singular(label: str) -> str:
    """"dictionaries" -> "dictionary"; chopping the last letter gave
    "content-coding dictionarie"."""
    return label[:-3] + "y" if label.endswith("ies") else label.rstrip("s")


def _entry(kind: LibraryKind, name: str) -> Path:
    """
    The entry called `name`, or a KeyError naming what does exist.

    Matched by full filename first. A bare stem is accepted only while it is
    unambiguous: the dictionaries kind takes three formats, so `foo.csv` and
    `foo.dic` can both exist, both called "foo" -- and resolving a stem to the
    first sorted match deleted, renamed and exported the file the user did
    *not* pick. Ambiguity is an error here, never a guess.
    """
    listed = entries(kind)
    for f in listed:
        if f.name == name:
            return f
    stem_matches = [f for f in listed if f.stem == name]
    if len(stem_matches) == 1:
        return stem_matches[0]
    if stem_matches:
        twins = ", ".join(f.name for f in stem_matches)
        raise KeyError(
            f"{name!r} is ambiguous -- the library has {twins}. "
            "Say which, with its extension."
        )
    have = ", ".join(f.name for f in listed) or "(none)"
    raise KeyError(f"no {_singular(kind.label.lower())} called {name!r}. Have: {have}")
