"""
A record, beside every feature table, of how that table was measured.

Why this exists
---------------
A saved model records its predictors **by name**. That is enough to refuse a
table which lacks a column, and it is no defense at all against a table whose
columns have the right names and the wrong numbers. Those are easy to produce
by accident: ``lemmatize`` on instead of off, a MATTR window of 100 instead of
50, ``nltk`` instead of ``stanza``, ``rounding=6`` instead of 4, an edited
connective list. None of them changes a single column heading.

In a real run, scoring a ridge model against cohesion features whose values had
been produced under different settings moved the mean predicted age from 36.5
to 44.0 years -- and reported 904 predictions without a word of complaint. That
is the failure this module exists to make impossible, and it needs a record
that does not exist anywhere else: nothing in Taters wrote down *how* a feature
table was made. The run manifest keeps the resolved pipeline variables, not the
settings each function actually ran with, and the settings are mostly not in
the pipeline anyway -- ``analyze_lexical_richness`` owns eight of them and
states none, because they are function defaults.

The shape of the answer
-----------------------
Every setting a function ran with is captured **at the analyzer**, where the
real signature is visible. It cannot be captured at the call site: a
``potato.*`` call resolves to a facade method whose signature is literally
``(**kwargs)``, so binding defaults there yields nothing.

Settings are then split three ways, and the split matters:

* **instrument** -- changes the numbers. Everything not named as something else
  lands here, which is the safe default: a setting nobody has classified is
  treated as load-bearing, so forgetting to classify one costs a redundant
  extraction rather than a wrong answer.
* **binding** -- says which text was read (``csv_path``, ``text_cols``,
  ``encoding``). Recorded for diagnosis, never compared: a model exists in
  order to meet a *different* corpus, so comparing input identity would mean
  nothing ever matched.
* **grain** -- what one row represents. Recorded, reported, and deliberately
  **never compared**. Training a model on user-level aggregates and applying it
  to user-months is an intended use, not a mistake; the grain is a property of
  the run being set up now, and a model does not get a veto over it.

The record is bound to its table's **content**, not its timestamp, so a
hand-edited or externally-regenerated CSV reads as having no record at all
rather than as matching.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from pathlib import Path
from typing import (Any, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

__all__ = ["PROVENANCE_VERSION", "PLUMBING", "TEXT_INPUT",
           "TEXT_GRAIN", "records_settings", "sidecar_path", "read",
           "differences", "explain", "canonical", "digest", "asset_manifest",
           "for_columns", "compare_tables", "embedded_assets", "replays"]

#: Bumped whenever the recorded shape or the canonicalizer changes. It is the
#: first key inside every hashed blob, so bumping it retires every stored
#: digest into the safe direction (nothing matches) rather than leaving old
#: records to be compared against new ones under the same hash.
PROVENANCE_VERSION = 2

#: The pseudo-setting a record-format mismatch is reported under.
RECORD_FORMAT = "(record format)"

#: Read in 1 MB blocks: a 300 MB feature table hashes in about 0.17s on an
#: ordinary machine (~1.7 GB/s), which is nothing beside the minutes it took
#: to produce, so every table is hashed and there is no size cap. A cap would
#: have to mark large tables unverifiable, and a warning that always fires on
#: sentence embeddings is a warning nobody reads.
_BLOCK = 1024 * 1024

#: Where a float stops being a setting and starts being noise.
_FLOAT_PLACES = 12

#: The input family every text step shares: which text was read, and how it
#: was found. Recorded for diagnosis, never compared -- a saved model exists
#: in order to meet a *different* corpus, so comparing these would mean
#: nothing ever matched.
#:
#: Declared once and reused, so that `@records_settings(binding=TEXT_INPUT,
#: grain=TEXT_GRAIN, outputs=("out_features_csv",))` reads at a glance as
#: "an ordinary text analyzer" and anything unusual about a step stands out
#: as the part that differs.
#:
#: `joiner` is deliberately NOT here. It joins several text columns before
#: measurement, so " " against "\n" changes sentence segmentation and
#: therefore every readability, cohesion and part-of-speech number -- under
#: identical column names. It is an instrument setting.
TEXT_INPUT = ("csv_path", "txt_dir", "analysis_csv", "gathered_csv",
              "text_cols", "id_cols", "delimiter", "recursive", "pattern",
              "id_from", "include_source_path", "encoding",
              "pass_through_cols")

#: What one row represents. Recorded and reported, never compared: training a
#: model on user-level aggregates and applying it to user-months is an
#: intended use, and the grain belongs to the run being set up now.
TEXT_GRAIN = ("mode", "group_by")

#: Parameters that exist in every analyzer and measure nothing: callbacks,
#: chattiness, parallelism, device placement, the overwrite contract, and the
#: spill mechanics of grouping a large spreadsheet. One short list rather than
#: a per-function annotation, because these mean the same thing everywhere --
#: and `test_declared_non_measuring_settings_really_do_not_measure` varies
#: every one of them and asserts the output bytes do not move, so a wrong
#: entry here fails at review time instead of silently blessing a mismatch.
PLUMBING = ("on_progress", "verbose", "workers", "device",
            "overwrite_existing", "tmp_root", "num_buckets",
            "max_open_bucket_files")


def _strict() -> bool:
    """Whether a provenance failure should be raised rather than swallowed.

    Off for users, on for this project's own test suite -- see the comment on
    rule 4 in `records_settings`.
    """
    import os

    return os.environ.get("TATERS_PROVENANCE_STRICT", "") not in ("", "0")


# ---------------------------------------------------------------------------
# Canonicalization and digests
# ---------------------------------------------------------------------------

def canonical(obj: Any) -> Any:
    """
    One JSON-safe shape for a settings value, whatever it arrived as.

    ``Path`` and ``PosixPath`` stringify, tuples and sets become lists, floats
    round. Without this the same settings hash differently depending on
    whether a caller passed ``["a.txt"]`` or ``("a.txt",)`` or a ``Path`` --
    all of which happen, because these functions are called from a pipeline,
    from the facade and directly from tests.

    List order is preserved. It matters at least once: ``dict_paths`` order
    decides which dictionary supplies the unprefixed global columns.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, set):
        return sorted(canonical(v) for v in obj)
    if isinstance(obj, float):
        return round(obj, _FLOAT_PLACES)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    # anything else -- a module, a callable, an open handle -- isn't a
    # setting anyone can compare. so we just name it by type: the record
    # stays readable, and it can never accidentally equal a real value.
    return f"<{type(obj).__name__}>"


def digest(obj: Any) -> str:
    """A short, stable hash of a canonicalized object."""
    blob = json.dumps({"v": PROVENANCE_VERSION, "d": canonical(obj)},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def file_digest(path: Path) -> str:
    """sha256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Library assets
# ---------------------------------------------------------------------------

#: What an asset entry says when the file is not on this machine. It can never
#: equal a real digest, so a missing asset can never read as a match.
ABSENT = "absent"


def asset_manifest(kind: Optional[str], values: Any, *,
                   embed: bool = True) -> List[dict]:
    """
    Every file behind one asset setting, by **content**.

    ``kind`` is a library kind when the asset lives in the library (a folder
    then expands to that kind's files) and ``None`` for an asset that is a
    plain file the run produced -- the vocabulary a document-term matrix was
    scanned against, a fitted theme model. Those are still assets in every
    sense that matters here: their *content* decided the numbers, so it is
    hashed, embedded and compared, and their path is nobody's business.

    Paths are never compared. The wizard writes absolute resolved paths into a
    preset (``/home/<user>/.taters/library/...``), so a path comparison refuses
    every asset on a colleague's machine while happily missing a file that was
    edited under an unchanged name -- exactly backwards.

    Folders are expanded, because the wizard's default writes a whole library
    *folder* into the preset rather than a file list.

    ``embed`` carries each file's text inside the record, which is what makes a
    model portable to a machine that has never seen the asset. It also means a
    shared model file contains that content, so whatever writes a model is
    responsible for saying so at the moment of sharing.
    """
    from . import library as lib

    out: List[dict] = []
    try:
        paths = lib.expand(lib.kind_by_id(kind), _as_list(values)) \
            if kind else [Path(v) for v in _as_list(values)]
    except Exception:
        paths = [Path(v) for v in _as_list(values)]
    for path in sorted(paths, key=lambda p: p.name.lower()):
        entry: dict = {"name": path.name}
        try:
            # we digest the text as *read*, not the bytes on disk. what a
            # model carries is the text, and a replay writes that text back
            # out: when we hashed bytes, the copy differed from the original
            # over a byte-order mark or Windows line endings, and the gate
            # then refused a model its own vocabulary. hashing what we
            # actually embed makes the copy match by construction.
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            entry["sha256"] = text_digest(text)
            if embed:
                entry["text"] = text
        except OSError:
            entry["sha256"] = ABSENT
        out.append(entry)
    return out


def text_digest(text: str) -> str:
    """The digest an asset's embedded text carries; see :func:`asset_manifest`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        return [values]
    if isinstance(values, (list, tuple, set)):
        return list(values)
    return [values]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

def sidecar_path(csv_path: Any) -> Path:
    """``features/cohesion.csv`` -> ``features/cohesion_settings.json``."""
    p = Path(csv_path)
    return p.with_name(f"{p.stem}_settings.json")


def read(csv_path: Any) -> Optional[dict]:
    """
    The record for a feature table, or ``None``.

    ``None`` for four different reasons, all of which have to mean the same
    thing to a caller -- *this table's provenance is unknown* -- because the
    alternative is a caller that treats one of them as a match:

    * no sidecar was ever written;
    * the sidecar is unreadable or was written by a newer Taters;
    * the sidecar says ``incomplete``;
    * **the table's bytes are not the bytes the sidecar describes.** This is
      the load-bearing one. It is what makes the record durable rather than
      advisory: it survives the coarse mtimes of a Windows drive mounted into
      WSL, and it catches a CSV that was hand-edited or regenerated by
      something outside Taters.
    """
    path = sidecar_path(csv_path)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return None
    if not isinstance(rec, dict):
        return None
    if int(rec.get("version", 0)) != PROVENANCE_VERSION:
        return None
    if rec.get("state") != "complete":
        return None
    stated = (rec.get("output") or {}).get("sha256")
    try:
        if not stated or stated != file_digest(Path(csv_path)):
            return None
    except OSError:
        return None
    return rec


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def differences(expected: Mapping, actual: Mapping) -> List[tuple]:
    """
    Every way two records disagree about *the numbers*, and nothing else.

    Compares ``instrument`` and ``assets``. Does **not** compare ``binding``
    (a model exists to meet a different corpus) and does **not** compare
    ``grain`` (see the module docstring: applying a user-level model to
    user-months is an intended use, so the grain is reported and never
    policed).

    Pure, so the refusal message and the tests are built from the same list.
    """
    diffs: List[tuple] = []
    exp_i = dict((expected.get("instrument") or {}))
    act_i = dict((actual.get("instrument") or {}))
    for key in sorted(set(exp_i) | set(act_i)):
        was, now = exp_i.get(key, "<not recorded>"), act_i.get(key, "<not recorded>")
        if canonical(was) != canonical(now):
            diffs.append((key, was, now, "changes the values, not the names"))

    exp_a = (expected.get("assets") or {})
    act_a = (actual.get("assets") or {})
    for key in sorted(set(exp_a) | set(act_a)):
        for name, was, now in _asset_diffs(exp_a.get(key) or [],
                                           act_a.get(key) or []):
            diffs.append((f"{key}:{name}", was, now,
                          "the word list itself is different"))
    return diffs


def _asset_diffs(expected: Sequence[dict], actual: Sequence[dict]):
    """Per-file asset differences, by name then content."""
    was = {e.get("name"): e.get("sha256") for e in expected}
    now = {a.get("name"): a.get("sha256") for a in actual}
    for name in sorted(set(was) | set(now)):
        a, b = was.get(name, "not used"), now.get(name, "not present")
        if a != b:
            yield name, a, b


def explain(what: str, diffs: Sequence[tuple], *,
            table: Optional[str] = None) -> str:
    """
    The house-style sentence for a settings mismatch.

    Names every differing setting with both values, then says why it matters,
    because "provenance mismatch" tells a researcher nothing they can act on
    and "engine was 'stanza', this run 'nltk'" tells them everything.
    """
    if not diffs:
        return ""
    width = max(len(str(k)) for k, _w, _n, _y in diffs)
    was_width = max(len(_short(w)) for _k, w, _n, _y in diffs)
    rows = "\n".join(
        f"    {str(key).ljust(width)}  was {_short(was).ljust(was_width)}"
        f"   this run: {_short(now)}"
        for key, was, now, _why in diffs)
    where = f"this run's {table!r} features" if table else "this run's features"
    head = f"{what} cannot be scored on {where}."
    upstream = any(k == "(the text it measured)" for k, _w, _n, _y in diffs)
    tail = ("Those settings change the numbers, not the column names, so "
            "scoring would report a prediction computed from features this "
            "model has never seen. Re-run with those settings, or score with "
            "a model fitted on the features you have.")
    if any(k == RECORD_FORMAT for k, _w, _n, _y in diffs):
        tail = ("The model and this run's features were recorded under "
                "different versions of the settings record, so their "
                "settings cannot be compared. Re-fit the model with this "
                "version of Taters, or extract the features with the version "
                "that fitted it.")
    if upstream:
        tail = ("The measuring settings agree, but the text itself was put "
                "together differently before it was measured -- how the text "
                "columns were joined, or which ones were read. A model "
                "records that its text was prepared differently without "
                "recording how, so this one cannot be fixed automatically: "
                "gather the text the way the model was fitted, or score with "
                "a model fitted on the text you have.")
    return (f"{head}\n\nIt was fitted with settings this run did not use:\n\n"
            f"{rows}\n\n{tail}")


def _short(value: Any, limit: int = 28) -> str:
    text = repr(canonical(value))
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------

def records_settings(*, binding: Sequence[str] = (), grain: Sequence[str] = (),
                     outputs: Sequence[str] = (),
                     assets: Optional[Mapping[str, str]] = None,
                     advisory: Sequence[str] = (),
                     defines_text: bool = False,
                     replay: Optional[Tuple[str, Mapping[str, str]]] = None,
                     absent_means_zero: bool = False,
                     redo_without_record: bool = False,
                     bookkeeping: Sequence[str] = (),
                     reshapes: str = ""):
    """
    Record how this function measured, beside whatever it wrote.

    Declare only what is *not* an instrument setting. Everything unnamed is
    treated as load-bearing, so the failure mode of forgetting to classify a
    new parameter is a redundant extraction, never a wrong answer.

    Parameters
    ----------
    binding
        Which text was read. Recorded for diagnosis, never compared.
    grain
        What one row represents. Recorded and reported, never compared.
    outputs
        Where the result went. Excluded from everything -- a path is not a
        setting.
    assets
        ``{parameter: library kind}`` for parameters holding word lists, whose
        *content* is hashed and embedded rather than their paths. The kind is
        ``None`` for an asset that is not a library file -- the frequency
        list a document-term matrix takes its vocabulary from. A model
        fitted on that matrix then carries the vocabulary, and a later run
        scans new text against *it*: re-deriving the vocabulary from the new
        corpus produced a matrix with different columns, and the model could
        not be scored at all (a real refusal, four columns short).
    replay
        ``(call, {parameter: output})`` when this table cannot be measured
        the same way twice, because the step *fits* something to the corpus
        -- a topic model chooses its themes from the texts in front of it --
        and a new corpus must instead have that fitted thing **applied**.
        ``call`` is the applying function, ``"module:function"``, and the
        mapping says which of this step's outputs it takes under which
        parameter. The output's content is recorded as an asset of the
        replay, so a model fitted on the themes carries the theme model and
        can score a corpus the themes were never fitted on -- which is the
        only honest way to do it, since refitting gave a different number of
        themes and the model refused.
    bookkeeping
        Columns this table writes beside its measures that are *not*
        measures: the token count a matrix or topic model reports, the word
        count beside a dictionary's categories, a readability step's raw
        sentence and syllable counts. Numeric, so the statistics would
        otherwise treat them as features -- and did, in a real run where a
        ridge on a document-term matrix learned from ``token_count``. The
        assemble step keeps them in the analysis table (there to filter on
        and to look at) and out of the feature sets, and says so; one shared
        setting puts them back in for anyone who wants length as a predictor.
        The analyzer declares them because the analyzer is what knows.
    redo_without_record
        True for a step cheap enough that an existing output with no record
        of how it was made is redone rather than trusted -- the two text
        gathers, which take seconds. A stale metadata table from an earlier
        pipeline in the same folder, its record lost, was reused by a real
        run and the analyses then could not find their outcome columns.
        Left False for anything expensive: a transcript from before records
        existed may have taken hours, and unknown is not grounds to redo it.
    reshapes
        The parameter naming a feature table this step *reshapes* rather than
        measures -- averaging one up to a coarser row, say. The output's
        record then inherits that table's measuring identity: its ``call``,
        ``instrument``, ``assets`` and bookkeeping columns, and its place in
        the chain. Only the grain differs, and the grain is never compared.

        Without this the record would describe the reshaping, and two things
        would go wrong at once. A model fitted on averaged readability could
        never be scored on un-averaged readability, because the gate would
        compare an averaging step's settings against an analyzer's and find
        every one of them different -- the opposite of the portability
        ``defines_text`` exists to protect. And the measure's own bookkeeping
        columns would be forgotten, so a word count that has always been kept
        out of the feature sets would quietly arrive in one as a predictor.

        The reshaping step is still in the pipeline and its manifest; what
        this says is that reshaping is not measuring. ``reshaped_by`` on the
        record names the step that did it.
    absent_means_zero
        True when the table's *columns* depend on the corpus and a missing
        column means "this never occurred", not "this was not measured": a
        part-of-speech tag no text in the new corpus happens to use. A model
        fitted on such a table scores the absent column as zero instead of
        refusing, because zero is exactly what the count would have been.
    advisory
        Instrument settings to record and compare but never refuse on. For
        settings whose difference is worth reporting and is not by itself
        grounds to stop.
    defines_text
        True for a step that decides what text *exists* rather than how
        existing text is measured -- a gather, a transcription. Its settings
        are recorded and are kept out of the chain that downstream steps
        fold in, because they describe the **dataset** and not the
        instrument.

        This is the line that makes a saved model portable. A model exists in
        order to meet new text, and new text is legitimately assembled
        differently: pre-aggregated, differently joined, from another study
        entirely. Someone applying a model to a new corpus to see whether it
        reproduces a nomological network there is doing the thing models are
        for, and nothing about how that corpus was put together is grounds to
        stop them.

        The chain still has a job, and it is the reason this is a flag rather
        than the removal of chaining: three feature tables are written by a
        *gather* over per-item measurements, so the settings that decided
        their numbers -- the Whisper model behind an embedding, the acoustics
        thresholds -- live upstream and must still be compared. The same tool
        can fall on either side: `transcribe_with_whisper` defines text, and
        `extract_whisper_embeddings` measures it.

    Notes
    -----
    Four rules govern the write, and each of them closes a hole that was
    verified to exist:

    1. **After the call, never before.** Writing first would let a
       short-circuited analyzer's old numbers be certified as the new
       settings, which is worse than no record at all.
    2. **Never write when the analyzer may not have done the work.** Every
       analyzer in this project returns an existing file when
       ``overwrite_existing`` is false, and the return value cannot
       distinguish that from a fresh computation. So the output's bytes are
       compared across the call, and on any ambiguity any existing sidecar is
       **deleted** -- unknown provenance must degrade to "cannot verify",
       never to a blessing.
    3. **Bind the record to the output's content**, so it cannot outlive the
       bytes it describes.
    4. **Provenance may never break an extraction.** The whole write is
       wrapped: a bug in fingerprinting must not surface to a user as
       "analyze_cohesion failed".
    """
    asset_keys = dict(assets or {})
    skip = set(binding) | set(grain) | set(outputs) | set(asset_keys)

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs = _redo_if_made_differently(
                fn, signature, args, kwargs, outputs=outputs, skip=skip,
                binding=binding, grain=grain, asset_keys=asset_keys,
                redo_without_record=redo_without_record)
            before = _peek_output(signature, args, kwargs, outputs)
            result = fn(*args, **kwargs)
            try:
                _write_record(fn, signature, args, kwargs, result,
                              before=before, skip=skip, binding=binding,
                              grain=grain, asset_keys=asset_keys,
                              advisory=advisory, defines_text=defines_text,
                              replay=replay,
                              absent_means_zero=absent_means_zero,
                              bookkeeping=bookkeeping, reshapes=reshapes)
            except Exception:
                # rule 4. a record is a nicety; the measurement isn't, so a
                # bug in here must never show up to a researcher as
                # "analyze_lexical_richness failed".
                #
                # but swallowing errors hides them from *us* too, and that's
                # not hypothetical: `_upstream` once raised on `recursive=True`
                # (Path(True) is a TypeError), so every record quietly lost
                # its upstream chain and the tests still passed. so, under
                # TATERS_PROVENANCE_STRICT we re-raise, and the test suite
                # sets it -- users get resilience, we get the traceback.
                if _strict():
                    raise
                pass
            return result

        wrapper.__provenance__ = {
            "binding": tuple(binding), "grain": tuple(grain),
            "outputs": tuple(outputs), "assets": dict(asset_keys),
            "advisory": tuple(advisory), "defines_text": bool(defines_text),
            "replay": (str(replay[0]), dict(replay[1])) if replay else None,
            "absent_means_zero": bool(absent_means_zero),
            "bookkeeping": tuple(bookkeeping),
            "reshapes": str(reshapes),
        }
        return wrapper

    return decorate


def _peek_output(signature, args, kwargs, outputs) -> Optional[dict]:
    """
    The output file's identity before the call, when it can be known.

    Rule 2 needs to tell "the analyzer wrote this" from "the analyzer found it
    already there". Timestamps cannot: on a Windows drive mounted into WSL --
    where this project is developed -- mtime granularity is coarse enough that
    a fast write can look older than the call that made it, which would report
    every fresh table as unverifiable.

    Content can. When the caller named the output path (the pipeline always
    does), its bytes are hashed before and after; unchanged bytes mean the work
    cannot be shown to have happened. When the path is computed inside the
    function and cannot be known in advance, this returns ``None`` and the
    caller falls back to "the file did not exist before" as the only safe
    evidence.
    """
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    for key in outputs:
        value = bound.arguments.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            return {"path": str(path), "existed": False}
        try:
            return {"path": str(path), "existed": True,
                    "sha256": file_digest(path)}
        except OSError:
            return None
    return None


def _write_record(fn, signature, args, kwargs, result, *, before, skip,
                  binding, grain, asset_keys, advisory,
                  defines_text: bool = False, replay=None,
                  absent_means_zero: bool = False,
                  bookkeeping: Sequence[str] = (),
                  reshapes: str = "") -> None:
    from .. import __version__ as taters_version

    out = Path(result) if isinstance(result, (str, Path)) else None
    if out is None or not out.is_file():
        return

    # rule 2: if there's any doubt the work actually happened, we remove the
    # record rather than write one. an output handed back unchanged keeps the
    # record it already had -- that record described exactly these bytes, and
    # the settings that made them are still the truth about the file. we used
    # to delete it here, which meant every resumed run stripped every
    # short-circuited table of its record, so a saved model couldn't be
    # checked against features that had merely been re-used, and a gather
    # made under other settings could no longer be told apart from one made
    # under these. so only a record that doesn't match the bytes gets tossed.
    if before is not None and before.get("existed"):
        if before.get("sha256") == file_digest(out):
            old = read(out)
            if not old or (old.get("output") or {}).get("sha256") != before["sha256"]:
                sidecar_path(out).unlink(missing_ok=True)
            return
    elif before is None:
        # we had no way to know the path ahead of time, so nothing can tell
        # us whether the work happened or an existing file got handed back.
        # unknown degrades to "can't verify": no record, and any older one
        # removed. this branch used to be a comment with `pass` under it, and
        # a second call with new settings and overwrite_existing=False then
        # restamped the old numbers as the new settings -- the exact thing
        # certification rule 2 is there to prevent. the pipeline and the
        # wizard always name the output path, so their records are fine; a
        # direct call that lets the function choose its own path gets no
        # record, and can pass the path if it wants one.
        sidecar_path(out).unlink(missing_ok=True)
        return

    every = _effective_arguments(signature, args, kwargs)
    told = _classify(every, skip=skip, binding=binding, grain=grain,
                     asset_keys=asset_keys)
    instrument = told["instrument"]
    record = {
        "version": PROVENANCE_VERSION,
        "taters": taters_version,
        "call": f"{fn.__module__}:{fn.__qualname__}",
        "table": out.stem,
        "state": "complete",
        "defines_text": bool(defines_text),
        "absent_means_zero": bool(absent_means_zero),
        "bookkeeping": [str(c) for c in bookkeeping],
        "instrument": instrument,
        "advisory": {k: canonical(every.get(k)) for k in advisory},
        "assets": told["assets"],
        "grain": told["grain"],
        "binding": told["binding"],
        "upstream": _upstream(every, binding),
        "output": {"stem": out.stem, "bytes": out.stat().st_size,
                   "sha256": file_digest(out)},
    }
    replayed = _replay_record(replay, every)
    if replayed:
        record["replay"] = replayed
    if reshapes:
        _inherit_measurement(record, every.get(reshapes))
    # read the instrument back off the record rather than from the local: a
    # reshaping step amends the record just above, and digesting the local
    # meant the digests described settings the record no longer claimed.
    record["digests"] = {
        "instrument": digest([record["call"], record["instrument"],
                              record["assets"]]),
        "chain": digest([record["call"], record["instrument"],
                         record["assets"], record["upstream"]]),
    }
    from .atomic import atomic_write

    with atomic_write(sidecar_path(out), mode="w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)


def _inherit_measurement(record: dict, source: Any) -> None:
    """
    Take the measuring identity of the table this step reshaped.

    Reshaping is not measuring. A table of readability averaged per speaker
    holds readability's numbers at a coarser grain, so what a later check
    wants to know about it is how readability was measured -- and the grain,
    which is recorded and never compared. Written into the record before the
    digests are taken, so the chain digest comes out equal to the measured
    table's and the two are comparable rather than merely similar.

    Silent when the source table has no readable record: unknown provenance
    stays unknown, and the gate has a separate, waivable answer for that.
    """
    parent = read(source) if source else None
    if not parent:
        return
    record["reshaped_by"] = record["call"]
    record["call"] = parent.get("call") or record["call"]
    record["instrument"] = parent.get("instrument") or {}
    record["assets"] = parent.get("assets") or {}
    record["advisory"] = {**(parent.get("advisory") or {}),
                          **(record.get("advisory") or {})}
    record["absent_means_zero"] = bool(parent.get("absent_means_zero"))
    # `or` is wrong here and cost an afternoon: a measure whose only upstream
    # step is a gather has an EMPTY upstream, because a gather is
    # `defines_text` and stays out of the chain. falling back on empty put
    # the reshaping step's own upstream in instead, and the chain digest then
    # differed from the measured table's by exactly the thing we were trying
    # to make equal.
    if "upstream" in parent:
        record["upstream"] = parent["upstream"]
    if parent.get("replay"):
        record["replay"] = parent["replay"]
    # both: the measure's own (a word count nobody wants as a predictor) and
    # this step's (how many rows went into each average).
    record["bookkeeping"] = sorted(set(record.get("bookkeeping") or [])
                                   | set(parent.get("bookkeeping") or []))


def _effective_arguments(signature, args, kwargs) -> dict:
    """Every argument the call will run with, defaults applied, plumbing
    removed -- callbacks, handles and workers are not settings."""
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    every = dict(bound.arguments)
    every.pop("self", None)
    for noise in PLUMBING:
        every.pop(noise, None)
    return every


def _classify(every: Mapping, *, skip, binding, grain, asset_keys) -> dict:
    """The four recorded views of one call's arguments, canonicalized."""
    return {
        "instrument": {k: canonical(v) for k, v in every.items() if k not in skip},
        "assets": {k: asset_manifest(kind, every.get(k))
                   for k, kind in asset_keys.items()},
        "grain": {k: canonical(every.get(k)) for k in grain},
        "binding": {k: canonical(every.get(k)) for k in binding},
    }


def _redo_if_made_differently(fn, signature, args, kwargs, *, outputs, skip,
                              binding, grain, asset_keys,
                              redo_without_record: bool = False) -> dict:
    """
    Reuse an existing output only if it was made the way this call would make it.

    Every analyzer returns an existing output untouched when
    ``overwrite_existing`` is False -- the resume contract. But "existing"
    said nothing about *how* it was made: a second pipeline saved into the
    same folder found the first one's gathered text table and used it, ids
    and all, and a real run then died at the join because the table's
    ``row_<n>`` ids matched nothing (938 rows, 0 joined). The record beside
    the file knows how it was made; when it disagrees with this call on
    anything that decides the numbers -- binding, grain, instrument, assets
    -- the call is turned into an overwrite, and says so.

    A file with no record is left alone: it may be a transcript that took
    hours, made by a version before records existed, and "unknown" is not
    grounds to throw it away.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return kwargs
    if "overwrite_existing" not in signature.parameters \
            or bound.arguments.get("overwrite_existing"):
        return kwargs
    out = None
    for key in outputs:
        value = bound.arguments.get(key)
        if value:
            out = Path(value)
            break
    if out is None or not out.is_file():
        return kwargs
    old = read(out)
    if not old:
        if not redo_without_record:
            return kwargs
        try:
            verbose = _effective_arguments(signature, args, kwargs).get("verbose", True)
        except Exception:
            verbose = True
        if verbose:
            print(f"[taters] {out.name} exists with no record of how it was "
                  f"made; redoing it.")
        return {**kwargs, "overwrite_existing": True}
    try:
        every = _effective_arguments(signature, args, kwargs)
        mine = _classify(every, skip=skip, binding=binding, grain=grain,
                         asset_keys=asset_keys)
    except Exception:
        return kwargs
    changed = []
    for view in ("binding", "grain", "instrument"):
        was, now = old.get(view) or {}, mine[view]
        changed += [k for k in sorted(set(was) | set(now))
                    if canonical(was.get(k)) != canonical(now.get(k))]
    for key, entries in mine["assets"].items():
        if any(_asset_diffs((old.get("assets") or {}).get(key) or [], entries)):
            changed.append(key)
    if not changed:
        return kwargs
    if every.get("verbose", True):
        print(f"[taters] {out.name} exists but was made with different "
              f"settings ({', '.join(changed)}); redoing it.")
    return {**kwargs, "overwrite_existing": True}


def _replay_record(replay, every: Mapping) -> Optional[dict]:
    """
    How to measure this table again on other text, with what it fitted.

    Recorded only when every output the applying function needs was named
    by the caller: a fitted model whose path the analyzer chose for itself
    is not knowable here, and a replay that names the wrong file is worse
    than none. The pipeline and the wizard always name outputs, so their
    records carry the replay; a direct call can pass the path to get one.
    """
    if not replay:
        return None
    call, takes = replay
    assets = {}
    for param, output in dict(takes).items():
        value = every.get(output)
        if not value:
            return None
        manifest = asset_manifest(None, value)
        if any(e.get("sha256") == ABSENT for e in manifest):
            return None
        assets[str(param)] = manifest
    return {"call": str(call), "assets": assets}


def _upstream(every: Mapping, binding: Sequence[str]) -> List[str]:
    """
    The chain digests of the records beside whatever this step read.

    Three of the joinable feature tables are written by a *gather*, not by a
    measurement: the acoustics summary, the aggregated Whisper embeddings and
    the aggregated sentence embeddings. A gather's own settings are plumbing;
    the settings that decided the numbers -- ``acoustics_target_dbfs``, the
    Whisper model, the embedding model -- live upstream. Without this walk the
    record for exactly the tables most likely to be model predictors would be
    a lie by omission.

    Deliberately records no *data* identity: which files were read, and how
    many, are excluded, because a saved model exists to meet a different
    corpus and a chain covering the inputs could never match anything.
    """
    found: List[str] = []
    for key in binding:
        value = every.get(key)
        for candidate in _as_list(value):
            # only path-like values name an upstream file. `binding` also
            # carries `recursive=True`, `delimiter=","` and column-name lists,
            # and `Path(True)` raises -- rule 4 would then swallow that, and
            # every record would quietly lose its upstream chain.
            if not isinstance(candidate, (str, Path)):
                continue
            path = Path(candidate)
            try:
                if path.is_dir():
                    sources: Iterable[Path] = sorted(path.glob("*.csv"))
                elif path.is_file():
                    sources = [path]
                else:
                    continue
            except OSError:
                continue
            for source in sources:
                rec = read(source)
                if not rec:
                    continue
                if rec.get("defines_text"):
                    # a step that decides what text exists is describing the
                    # dataset, not the instrument, so we keep it out of the
                    # chain -- otherwise we could never apply a model to a
                    # corpus that was assembled differently, and that's the
                    # whole point of having a model. see `defines_text`.
                    continue
                chain = (rec.get("digests") or {}).get("chain")
                if chain and chain not in found:
                    found.append(chain)
    return sorted(found)


# ---------------------------------------------------------------------------
# What a model needs to record about the features it was fitted on
# ---------------------------------------------------------------------------

def for_columns(table_csv: Any, columns: Sequence[str]) -> dict:
    """
    The records for the feature tables that supplied ``columns``.

    Returns ``{stem: comparable record}``, ready to store inside a model. A
    table with no readable record appears with ``{"state": "unavailable"}``
    rather than being omitted, because "there was no record" and "this table
    was not involved" have to be different things to whatever checks later --
    omitting it would let an unverifiable table pass as one that was never
    used.

    The stored record is trimmed to what a comparison actually reads. The
    embedded asset *text* is dropped here and kept only in the model's own
    asset copy, so a model does not carry every word list twice.
    """
    import json

    wanted = {str(c) for c in columns}
    table = Path(table_csv)
    sidecar = table.with_name(table.stem + "_sets.json")
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}

    sources = doc.get("sources") or {}
    out: dict = {}
    for stem, owned in (doc.get("sets") or {}).items():
        if not wanted.intersection(str(c) for c in (owned or ())):
            continue
        path = sources.get(stem)
        rec = read(path) if path else None
        out[str(stem)] = comparable(rec)
    return out


def comparable(rec: Optional[dict]) -> dict:
    """
    The part of a record that :func:`compare_tables` reads, in one shape.

    Built here for both sides of the comparison -- the fit-time side stored
    in a model (:func:`for_columns`) and the run-time side read beside the
    table about to be scored -- because the two were assembled by hand in
    two modules and drifted: the run-time copy had no ``digests``, so the
    chain comparison never fired at the gate and a table prepared
    differently upstream (a different gather ``joiner`` moving every number
    under identical column names) scored without a word.

    The word lists' text is dropped: they are embedded in the model to be
    *used*, and comparing them by content digest is what ``assets`` is for.
    """
    if rec is None:
        return {"state": "unavailable"}
    return {
        "state": "recorded",
        "version": rec.get("version"),
        "call": rec.get("call"),
        "instrument": rec.get("instrument"),
        "assets": {k: [{kk: vv for kk, vv in entry.items() if kk != "text"}
                       for entry in v]
                   for k, v in (rec.get("assets") or {}).items()},
        "advisory": rec.get("advisory") or {},
        "grain": rec.get("grain") or {},
        "digests": rec.get("digests"),
        "absent_means_zero": bool(rec.get("absent_means_zero")),
        "bookkeeping": [str(c) for c in (rec.get("bookkeeping") or [])],
        "replay": _without_text(rec.get("replay")),
    }


def _without_text(replay: Optional[Mapping]) -> Optional[dict]:
    """A replay's asset manifest by digest only; the text lives in the
    model's own asset copy, once, beside the word lists."""
    if not replay:
        return None
    return {
        "call": replay.get("call"),
        "assets": {k: [{kk: vv for kk, vv in entry.items() if kk != "text"}
                       for entry in v]
                   for k, v in (replay.get("assets") or {}).items()},
    }


def replays(expected: Mapping, actual: Mapping) -> bool:
    """
    Whether ``actual`` is ``expected`` measured again by its own replay.

    A topic model's themes were fitted on one corpus; the table on a second
    corpus is written by *applying* that fitted model, so its record names
    a different function and carries the model file as an asset. The two
    agree when the applying function is the one the fit-time record named
    and every asset it applied is, byte for byte, the file the fit wrote.
    Nothing else is compared: the fitted file pins the vocabulary and the
    loadings, which is everything that decided the numbers.
    """
    want = (expected or {}).get("replay") or {}
    if not want or (actual or {}).get("call") != want.get("call"):
        return False
    have = (actual or {}).get("assets") or {}
    for key, entries in (want.get("assets") or {}).items():
        if any(_asset_diffs(entries, have.get(key) or [])):
            return False
    return True


def embedded_assets(table_csv: Any, columns: Sequence[str]) -> dict:
    """
    Every word list behind these columns, with its text, for a model to carry.

    Kept separately from the per-table records, which hold digests only, so a
    model that was fitted on three tables sharing one stop list carries that
    list once.

    Carrying the text is what makes a model portable: a colleague can score
    with it on a machine that has never seen your connective lists. It also
    means the model file *contains* those lists, which is a fact whatever
    shares a model has to state plainly -- a published model may carry
    licensed or confidential material its author should check first.
    """
    import json

    wanted = {str(c) for c in columns}
    table = Path(table_csv)
    sidecar = table.with_name(table.stem + "_sets.json")
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}

    sources = doc.get("sources") or {}
    out: dict = {}
    for stem, owned in (doc.get("sets") or {}).items():
        if not wanted.intersection(str(c) for c in (owned or ())):
            continue
        rec = read(sources.get(stem)) if sources.get(stem) else None
        # the replay's assets ride along too: a model fitted on themes has to
        # carry the theme model with it, or it can't meet a second corpus at all.
        carried = dict((rec or {}).get("assets") or {})
        carried.update(((rec or {}).get("replay") or {}).get("assets") or {})
        for kind, entries in carried.items():
            bucket = out.setdefault(kind, {})
            for entry in entries:
                name = entry.get("name")
                if name and name not in bucket:
                    bucket[name] = entry
    return {kind: [bucket[n] for n in sorted(bucket)]
            for kind, bucket in out.items()}


def compare_tables(expected: Mapping[str, Mapping],
                   actual: Mapping[str, Mapping]) -> Tuple[list, list]:
    """
    Compare a model's recorded feature tables against this run's.

    Returns ``(mismatched, unverifiable)``:

    * ``mismatched`` -- ``(stem, [differences...])`` where both sides are
      known and the measuring settings differ. Never overridable: both
      answers are in hand and they disagree.
    * ``unverifiable`` -- stems where one side has no record. A different
      thing entirely, and the only one a researcher can reasonably waive,
      per table.

    Keeping them apart is the whole ethic of the check: *refuse when you know
    what you are missing, and say so when you do not.*
    """
    mismatched: list = []
    unverifiable: list = []
    for stem, want in expected.items():
        have = actual.get(stem)
        if want.get("state") != "recorded" or not have \
                or have.get("state") != "recorded":
            unverifiable.append(stem)
            continue
        if replays(want, have):
            continue
        # two records written under different record formats hash their
        # assets differently, so we can't compare them setting by setting;
        # we once told a user "the word list itself is different" about a list
        # nobody had touched, and sent them hunting for a change that never
        # happened.
        if want.get("version") and have.get("version") \
                and want.get("version") != have.get("version"):
            mismatched.append((stem, [(
                RECORD_FORMAT, want.get("version"), have.get("version"),
                "the two records cannot be compared")]))
            continue
        diffs = list(differences(want, have))
        # the step's own settings can match while the text it measured was
        # prepared differently upstream -- a different `joiner` in the
        # gather changes sentence segmentation, so every number moves under
        # identical column names and `differences` above sees nothing at
        # all. the chain digest is what catches this; when we only compared
        # the instrument, we left exactly the hole the chain was built to close.
        want_chain = (want.get("digests") or {}).get("chain")
        have_chain = (have.get("digests") or {}).get("chain")
        if want_chain and have_chain and want_chain != have_chain \
                and canonical(want.get("instrument")) == \
                canonical(have.get("instrument")):
            diffs.append((
                "(the text it measured)", want_chain[:8], have_chain[:8],
                "prepared differently before measuring"))
        if diffs:
            mismatched.append((stem, diffs))
    return mismatched, unverifiable
