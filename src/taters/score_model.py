"""
Score this dataset with a model somebody already fitted.

A saved model is the one artifact in Taters meant to leave the run that made
it: fit a ridge on a corpus where you *have* the outcome, then apply it to a
corpus where you do not. That is one job as far as the researcher is
concerned, and it was three -- a MEM topic model, a ridge and a classifier
each had their own menu entry, their own vocabulary and their own place in
the pipeline, and the ridge's place was at the very end, among the
statistics, where it is least useful: what comes back is a *feature*, a
column per text, and the next thing anyone wants to do with it is analyze it
like any other.

So there is one entry, it lives with feature extraction, and it works out
what to do from the model file. What a model needs is a fact about the
model, not a question for the user:

* a **MEM topic model** needs text, and re-derives its own vocabulary from
  it;
* a **ridge** or a **classifier** needs the feature columns it was fitted
  on, by name, which means the steps that produce them have to have run.

The second case is the interesting one, and the honest thing to do when
those columns are absent is to name them and stop -- not to impute, and not
to score a model on a feature it has never seen and call the result a
prediction.

What this does not yet do
-------------------------
A model fitted on features that were themselves *transformed* -- sentence
embeddings reduced to fifty dimensions by PCA, say -- needs that whole
recipe replayed before its weights mean anything, and a model file records
its predictor names but not how they were made. So the columns have to
already exist under the names the model knows. Getting from "here is a
folder of text" to "here are the exact fifty numbers this model wants" is a
pipeline the model would have to carry with it, and that is not built yet;
until it is, this step tells you precisely which columns it wanted.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from .helpers.model_spec import (MODEL_WORK_DIR, ModelInfo, describe,
                                 one_model_path, slug)
from .helpers.cliargs import CliSpec

PathLike = Union[str, Path]

__all__ = ["score_with_model", "main"]


def score_with_model(
    *,
    model_json: Union[PathLike, Sequence[PathLike]],

    # ----- what a feature-reading model scores -----
    feature_csvs: Sequence[PathLike] = (),
    metadata_csv: Optional[PathLike] = None,
    key_cols: Sequence[str] = ("text_id",),
    keep_inputs: bool = False,
    unverified_ok: Sequence[str] = (),
    allow_unrecorded: bool = False,

    # ----- what a text-reading model scores. pick one of these three; the
    # ----- arguments belonging to the other two get ignored. this is the
    # ----- same input contract every text step in Taters has, so that the
    # ----- composer can wire this one up exactly like the rest.
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,

    # ----- output -----
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    workers: Optional[int] = None,
    device: str = "auto",
    encoding: str = "utf-8-sig",
    rounding: int = 6,

    # ====== CSV GATHER OPTIONS (used when csv_path is provided) ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: str = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,

    # ====== TXT FOLDER GATHER OPTIONS (used when txt_dir is provided) ======
    recursive: bool = True,
    pattern: Optional[str] = None,
    id_from: str = "stem",
    include_source_path: bool = True,
) -> Path:
    """
    Score this dataset with a saved model, whatever kind of model it is.

    Parameters
    ----------
    model_json
        One or more models saved by earlier runs -- topic models, ridges,
        classifiers, word vectors, fine-tuned predictors -- as a file, a
        list of files, or a folder holding them (every model in it). Each
        model says what kind it is and what it needs; nothing here has to
        be told. Several models are scored one after another, each against
        the feature tables *it* was fitted on, into one file per model and
        one merged table; two models whose names read the same are refused
        (rename one under Settings → Manage saved models).
    feature_csvs
        The feature tables this run produced, joined on ``key_cols`` to find
        the columns a ridge or classifier was fitted on. Ignored by a model
        that works from text.
    metadata_csv
        A table keyed like the feature tables that carries the spreadsheet's
        own columns -- needed only by a model fitted with controls, whose
        ``age`` or ``gender`` no feature step produces. In a pipeline the
        metadata gather writes it; the composer adds that step whenever a
        chosen model needs controls.
    key_cols
        The columns identifying a row, used to join the feature tables.
    keep_inputs
        Keep the joined feature table the scoring read. Off by default: the
        row accounting that explains an unscored row is reported either way,
        and the table itself is a copy of columns that already exist.
    unverified_ok
        Feature tables to score against even though their measuring settings
        cannot be checked -- named one at a time, never a blanket flag,
        because a single boolean would let you waive one unverifiable table
        and silently waive the acoustics table sitting next to it.

        This waives *not knowing*. It cannot waive a known mismatch: when
        both the model and the table state their settings and the settings
        differ, scoring is refused outright.
    allow_unrecorded
        Score a model that predates settings-recording at all. Such a model
        cannot be checked in either direction, so this is refused by default:
        it is exactly the case that reported a prediction seven years wrong
        without complaint.
    csv_path, txt_dir, analysis_csv, gathered_csv
        Where the text comes from, for a model that reads text: a
        spreadsheet, a folder of ``.txt`` files, or an already-gathered
        analysis-ready table. Exactly one applies; a model that reads
        features ignores all four.
    out_csv
        Where the scores go. With one model, its scores; default
        ``features/model_scores/<model name>.csv``. With several, the
        **merged** table -- an outer join of every model's scores on
        ``key_cols``, every score column prefixed ``<model name>__`` so
        two models predicting the same outcome cannot collide -- default
        ``features/model_scores.csv``, with each model's own file beside it
        under ``<out_csv stem>/<model name>.csv`` and its own unscored-row
        accounting; the merged ``_unscored.csv`` names the model per row.
    workers : int, optional
        Parallelism for a text model, which has real work to do per row.
        Ignored by the others, whose arithmetic is one matrix multiply.
    device : {"auto", "cuda", "cpu"}
        Where a text model's tagger runs, if it was built with the stanza
        engine -- a runtime choice, deliberately not stored in the model.
    overwrite_existing
        When False (default) and the scores exist, they are returned
        untouched.
    rounding
        Decimal places in the scores.
    text_cols, id_cols, mode, group_by, delimiter, joiner
        How to read a spreadsheet of text: which columns hold it, which
        identify a row, and whether several text columns are joined
        (``"concat"``) or scored one at a time (``"separate"``). Only used
        with ``csv_path``.
    num_buckets, max_open_bucket_files, tmp_root
        Spill settings for grouping a large spreadsheet, passed straight
        through to the gatherer.
    recursive, pattern, id_from, include_source_path
        How to read a folder of ``.txt`` files. Only used with ``txt_dir``.

    Returns
    -------
    pathlib.Path
        The scores: one row per text, with the model's own output columns.
        A row the model could not score is blank rather than absent, so the
        table still joins back to everything else.

    Raises
    ------
    ValueError
        When the model needs feature columns this run does not have. The
        message names them, because the fix is to add the steps that produce
        them and nothing else in the message helps with that.
    """
    from .helpers.progress import announce
    from .stats._common import reusable

    infos = _models(model_json)
    several = len(infos) > 1
    if not several:
        info = infos[0]
        out_path = Path(out_csv) if out_csv else (
            Path("features") / "model_scores" / f"{slug(info.name)}.csv")
        tables = _tables_for(info, feature_csvs)
        # gate first, *then* check whether the scores already exist. if we
        # did it the other way around, a stale scores file sitting next to a
        # feature table that somebody re-measured with different settings
        # would sail right through, and that's exactly how this bites people
        # in real life (re-running a results folder after tweaking something)
        _check_provenance(info, tables, unverified_ok=unverified_ok,
                          allow_unrecorded=allow_unrecorded, verbose=verbose)
        return _score_one(
            info, tables, out_path, metadata_csv=metadata_csv, key_cols=key_cols,
            keep_inputs=keep_inputs, overwrite_existing=overwrite_existing,
            on_progress=on_progress, verbose=verbose, encoding=encoding,
            rounding=rounding, workers=workers, device=device,
            text_input=dict(
                csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
                gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
                mode=mode, group_by=group_by, delimiter=delimiter, joiner=joiner,
                num_buckets=num_buckets, max_open_bucket_files=max_open_bucket_files,
                tmp_root=tmp_root, recursive=recursive, pattern=pattern,
                id_from=id_from, include_source_path=include_source_path))

    # several models. the order here matters: we run every model's gate
    # before we score anything, because nobody wants a five-model run to
    # die on model #4 after chewing through the first three. so we pick each
    # model's tables, check them all, and complain about everything at once
    merged_path = Path(out_csv) if out_csv else Path("features") / "model_scores.csv"
    per_dir = merged_path.parent / merged_path.stem
    chosen_tables = []
    problems = []
    for info in infos:
        try:
            tables = _tables_for(info, feature_csvs)
            _check_provenance(info, tables, unverified_ok=unverified_ok,
                              allow_unrecorded=allow_unrecorded, verbose=verbose)
        except ValueError as e:
            problems.append(str(e))
            tables = []
        chosen_tables.append(tables)
    if problems:
        if len(problems) == 1:
            raise ValueError(problems[0])
        raise ValueError(
            f"{len(problems)} of the {len(infos)} models cannot be scored on this "
            f"run's features:\n- " + "\n- ".join(problems))

    text_input = dict(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, joiner=joiner,
        num_buckets=num_buckets, max_open_bucket_files=max_open_bucket_files,
        tmp_root=tmp_root, recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path)
    per_paths = []
    for info, tables in zip(infos, chosen_tables):
        per_path = per_dir / f"{slug(info.name)}.csv"
        per_paths.append(_score_one(
            info, tables, per_path, metadata_csv=metadata_csv, key_cols=key_cols,
            keep_inputs=keep_inputs, overwrite_existing=overwrite_existing,
            on_progress=on_progress, verbose=verbose, encoding=encoding,
            rounding=rounding, workers=workers, device=device,
            text_input=text_input))

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if reusable(merged_path, *per_paths, *(i.path for i in infos),
                overwrite_existing=overwrite_existing, verbose=verbose,
                what="the merged model scores"):
        return merged_path
    announce(on_progress, f"merging the scores of {len(infos)} models")
    _merge_scores(infos, per_paths, merged_path, key_cols=key_cols, encoding=encoding)
    _merge_unscored(infos, per_paths, merged_path, encoding=encoding)
    if verbose:
        print(f"[score] {len(infos)} models scored; each in {per_dir}, all in "
              f"{merged_path} with columns prefixed by the model's name")
    return merged_path


def _models(model_json) -> List[ModelInfo]:
    """
    Every model named -- a file, files, or folders of them -- described.

    Two models whose names slug to the same word are refused: the slug is
    the per-model file name, the column prefix and the private-table
    folder, so the two could not be told apart anywhere they land.
    """
    from .helpers.library import model_files

    given = list(model_json) if isinstance(model_json, (list, tuple)) else [model_json]
    paths: List[Path] = []
    for item in given:
        path = Path(str(item))
        if path.is_dir():
            paths.extend(model_files(path))
        else:
            paths.append(path)
    if not paths:
        # nothing found -- we let the single-model resolver do the
        # complaining, since it already has the right words for it
        one_model_path(model_json)
    infos = [describe(p) for p in paths]
    seen: Dict[str, str] = {}
    for info in infos:
        key = slug(info.name)
        if key in seen and seen[key] != str(info.path):
            raise ValueError(
                f"two models are both called {info.name!r} (or spell it so it "
                f"reads the same: {seen[key]} and {info.path}); their scores "
                f"could not be told apart. Rename one under Settings → Manage "
                f"saved models.")
        seen[key] = str(info.path)
    # the same file handed to us twice is still just the one model
    unique: Dict[str, ModelInfo] = {}
    for info in infos:
        unique.setdefault(str(info.path.resolve()), info)
    return list(unique.values())


def _owner_of(path: Path) -> Optional[str]:
    """The slug of the model a private feature table was measured for, from
    its place under ``model_inputs/<slug>/``; None for a shared table."""
    parts = list(Path(path).parts)
    if MODEL_WORK_DIR in parts:
        at = parts.index(MODEL_WORK_DIR)
        if at + 1 < len(parts) - 1:
            return parts[at + 1]
    return None


def _tables_for(info: ModelInfo, feature_csvs) -> List[Path]:
    """
    The feature tables *this* model reads, out of everything the run made.

    A run scoring several models carries every model's tables, private
    ones included: model A's parts of speech at its own settings under
    ``model_inputs/a/…/pos.csv`` and model B's under ``model_inputs/b/…/
    pos.csv``. The gate keys tables by stem, so handed the union it saw
    one ``pos`` -- whichever came last -- and refused the other model as a
    known mismatch. So each model gets the tables it recorded, its own
    private table preferred over the shared one, never another model's,
    and a stem still doubled after that is refused rather than guessed. A
    model with no record of its tables (scored under ``allow_unrecorded``)
    gets the shared tables alone.
    """
    paths = [Path(p) for p in (feature_csvs or ()) if str(p).strip()]
    mine = slug(info.name)
    expected = set(info.provenance or {}) | set(info.needs_tables or ())
    if not expected:
        return [p for p in paths if _owner_of(p) is None]
    chosen: Dict[str, Path] = {}
    for stem in expected:
        own = [p for p in paths if p.stem == stem and _owner_of(p) == mine]
        shared = [p for p in paths if p.stem == stem and _owner_of(p) is None]
        candidates = own or shared
        if len(candidates) > 1:
            raise ValueError(
                f"{info.display()} reads the {stem!r} feature table, and this run "
                f"has {len(candidates)} of them ({', '.join(str(c) for c in candidates)}); "
                f"it cannot tell which was measured for it.")
        if candidates:
            chosen[stem] = candidates[0]
    # anything the model didn't record (but that isn't some *other* model's
    # private table) still comes along for the ride. the gate can decide
    # what to make of it
    extras = [p for p in paths if p.stem not in expected and _owner_of(p) is None]
    return list(chosen.values()) + extras


def _score_one(info: ModelInfo, tables, out_path: Path, *, metadata_csv, key_cols,
               keep_inputs, overwrite_existing, on_progress, verbose, encoding,
               rounding, workers, device, text_input) -> Path:
    """One model to one file: the resume check, then the dispatch by what
    the model reads. The gate has already run."""
    from .helpers.progress import announce
    from .stats._common import reusable

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reusable(out_path, *list(tables or ()), metadata_csv, info.path,
                overwrite_existing=overwrite_existing, verbose=verbose,
                what="the model scores"):
        if verbose:
            print(f"Model scores already exist; returning existing file: "
                  f"{out_path}")
        return out_path
    if verbose:
        print(f"[score] {info.display()} -> {len(info.outputs)} output(s): "
              f"{_few(info.outputs)}")
    announce(on_progress, f"scoring with {info.name}")
    if info.needs == "text":
        return _score_text(
            info, out_path=out_path, overwrite_existing=overwrite_existing,
            on_progress=on_progress, verbose=verbose, encoding=encoding,
            rounding=rounding, workers=workers, device=device,
            text_input=text_input)
    return _score_features(info, feature_csvs=tables, metadata_csv=metadata_csv,
                           out_path=out_path, key_cols=key_cols,
                           keep_inputs=keep_inputs,
                           overwrite_existing=overwrite_existing,
                           on_progress=on_progress, verbose=verbose,
                           encoding=encoding, rounding=rounding)


def _merge_scores(infos: Sequence[ModelInfo], per_paths: Sequence[Path],
                  merged: Path, *, key_cols, encoding: str) -> None:
    """
    One table from every model's: an outer join on the key columns, rows
    in the first model's order (then any the others alone have), every
    score column prefixed ``<model name>__``.

    Written with ``csv`` directly, not through the statistics' assembler:
    that one drops columns that are not numbers, and a classifier's
    predicted class is a word. The dictionary analyzer's convention for
    many-in-one (``<dictionary>__<column>``), so a table's columns say
    where each came from without anyone remembering.
    """
    import csv

    from .helpers.atomic import atomic_write

    keys = [str(k) for k in key_cols]
    header: List[str] = list(keys)
    order: List[tuple] = []
    rows: Dict[tuple, Dict[str, str]] = {}
    for info, path in zip(infos, per_paths):
        prefix = f"{slug(info.name)}__"
        with Path(path).open("r", newline="", encoding=encoding) as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            present_keys = [k for k in keys if k in fields]
            for col in fields:
                if col not in keys:
                    header.append(prefix + col)
            for row in reader:
                key = tuple(row.get(k, "") for k in present_keys) + tuple(
                    "" for k in keys if k not in present_keys)
                if key not in rows:
                    rows[key] = dict(zip(keys, key))
                    order.append(key)
                for col in fields:
                    if col not in keys:
                        rows[key][prefix + col] = row.get(col, "")
    with atomic_write(merged, mode="w", newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(header)
        for key in order:
            writer.writerow([rows[key].get(col, "") for col in header])


def _merge_unscored(infos: Sequence[ModelInfo], per_paths: Sequence[Path],
                    merged: Path, *, encoding: str) -> None:
    """The per-model unscored-row accountings as one file with a leading
    ``model`` column -- written only when something went unscored, removed
    otherwise, as the single-model file is."""
    import csv

    from .helpers.atomic import atomic_write
    from .stats._fit_common import read_unscored, unscored_path

    entries = []
    for info, path in zip(infos, per_paths):
        for reason, detail, n in read_unscored(path, encoding=encoding):
            entries.append((slug(info.name), reason, detail, n))
    target = unscored_path(merged)
    if not entries:
        target.unlink(missing_ok=True)
        return
    with atomic_write(target, mode="w", newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(["model", "reason", "detail", "rows"])
        for model, reason, detail, n in entries:
            writer.writerow([model, reason, detail, str(n)])


def _score_text(info: ModelInfo, *, out_path, overwrite_existing,
                on_progress, verbose, encoding, rounding, workers, device,
                text_input) -> Path:
    """
    A model that reads text: hand it the text and let it do its own reading.

    The whole text-input family is forwarded rather than resolved here,
    because the model's own scoring function already accepts it and already
    knows how to gather a spreadsheet or a folder. Resolving it here would
    be a second implementation of the gather, agreeing with the first until
    one of them changed.
    """
    given = [k for k in ("csv_path", "txt_dir", "analysis_csv")
             if text_input.get(k)]
    if not given:
        raise ValueError(
            f"{info.display()} scores text, and no text was given. In a "
            f"pipeline this comes from the gather step; called directly, "
            f"pass one of csv_path, txt_dir or analysis_csv.")
    from .helpers.model_spec import scorer

    kwargs = {k: v for k, v in text_input.items()
              if v is not None and k != "pattern"}
    if text_input.get("pattern"):
        kwargs["pattern"] = text_input["pattern"]
    return scorer(info.type_id)(
        model_json=str(info.path), out_features_csv=out_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers or 0, device=device, encoding=encoding,
        rounding=rounding, **kwargs)


def _score_features(info: ModelInfo, *, feature_csvs, metadata_csv,
                    out_path, key_cols, keep_inputs, overwrite_existing,
                    on_progress, verbose, encoding, rounding) -> Path:
    """
    A model that reads features: join what this run made, then score.

    The join is the same one the statistics use, so the row accounting and
    the collision rules are the ones already documented rather than a second
    set that agrees with them until it does not.
    """
    from .stats.assemble import assemble_analysis_table

    paths = [Path(p) for p in feature_csvs if str(p).strip()]
    if not paths:
        raise ValueError(
            f"{info.display()} scores feature columns, and this run has no "
            f"feature tables. " + _needs_sentence(info)
            + f" It needs {len(info.inputs)} column(s): "
              f"{_few(info.inputs)}.")

    work = out_path.parent / f"{out_path.stem}_inputs"
    # a categorical control shows up as the words the researcher typed, so
    # we have to tell the join to keep them. otherwise they'd get dropped as
    # "non-numeric", and a model fitted with `gender` could never be scored
    # at all.
    carry = sorted({str(entry["column"]) for entry in _control_spec(info)
                    if "level" in entry})
    # the controls themselves come from the spreadsheet, not from any
    # feature table. we refuse here, in words that say what to do about it,
    # rather than letting the design matrix choke after the join has quietly
    # failed to find them.
    controls = sorted({str(entry["column"]) for entry in _control_spec(info)})
    have = _columns_of(paths, encoding=encoding)
    from_features = [c for c in controls if c in have]
    if controls and not metadata_csv and from_features != controls:
        missing = [c for c in controls if c not in have]
        raise ValueError(
            f"{info.display()} was fitted with control column(s) "
            f"{', '.join(missing)}, which come from the spreadsheet rather "
            f"than from a feature table. Pass metadata_csv -- a table keyed "
            f"like the features that carries them; in a pipeline the "
            f"metadata gather writes it.")
    try:
        table = assemble_analysis_table(
            feature_csvs=[str(p) for p in paths], key_cols=key_cols,
            metadata_csv=str(metadata_csv) if metadata_csv else None,
            metadata_cols=[c for c in controls if c not in from_features]
            if metadata_csv else (),
            text_cols=carry, out_dir=work,
            out_csv=work / "scoring_table.csv",
            overwrite_existing=True, on_progress=on_progress,
            verbose=False, encoding=encoding)
        lost = _report_join(work, verbose=verbose)
        _check_inputs(info, table, encoding=encoding)
        # we pull the scorer from the registry, not an if/else. this used to
        # be `if classifier ... else ridge`, so a fourth kind of
        # feature-reading model would've been scored *as a ridge* --
        # silently, reading coefficients that weren't there.
        from .helpers.model_spec import scorer

        apply_fn = scorer(info.type_id)
        # only the join keys ride along next to the predictions. the table
        # was built from the model's own feature tables, so anything else in
        # it is either a measure the fit set aside or a raw control column
        # -- we once had a scores file carrying `vocd_50` and `gender` as if
        # they were identifiers.
        scored = apply_fn(
            model_json=str(info.path), input_csv=table, out_csv=out_path,
            overwrite_existing=True, on_progress=on_progress,
            verbose=verbose, encoding=encoding, rounding=rounding,
            id_cols=list(key_cols))
        _account_for_join(out_path, lost, encoding=encoding)
    finally:
        if not keep_inputs:
            shutil.rmtree(work, ignore_errors=True)
    return scored


def _columns_of(paths, *, encoding: str) -> set:
    """Every column name across these tables, by their headers alone."""
    import csv

    have: set = set()
    for path in paths:
        try:
            with Path(path).open("r", encoding=encoding, newline="") as fh:
                have.update(next(csv.reader(fh), []))
        except OSError:
            continue
    return have


def _check_inputs(info: ModelInfo, table: Path, *, encoding: str) -> None:
    """
    Name the missing columns before the scoring does, and say what to do.

    The apply functions refuse a missing predictor too, but from inside a
    step whose own name is the model's -- and the fix is never in the model,
    it is a feature step that was not selected. Controls are excluded from
    the check because the scoring rebuilds those from the source columns.
    """
    import csv

    with Path(table).open("r", encoding=encoding, newline="") as fh:
        have = set(next(csv.reader(fh), []))
    controls = {str(e.get("column")) for e in _control_spec(info)}
    from_controls = {name for name in info.inputs
                     if name.split("=")[0] in controls}
    # a column the model marks as a count that might just never occur is the
    # scorer's job to fill with zeros, not this check's to refuse.
    zeroable = set(info.zero_when_absent)
    absent = [name for name in info.inputs
              if name not in have and name not in from_controls
              and name not in zeroable]
    if absent:
        raise ValueError(
            f"{info.display()} was fitted on {len(info.inputs)} feature "
            f"column(s) and {len(absent)} of them are not in this run: "
            f"{_few(absent)}. " + _needs_sentence(info)
            + " Otherwise, score with a model fitted on the features you "
              "have.")


def _check_provenance(info: ModelInfo, feature_csvs, *, unverified_ok,
                      allow_unrecorded, verbose: bool) -> None:
    """
    Refuse to score a model on features it was not fitted on.

    Three states, and keeping them apart is the whole ethic of the check:

    * **both sides known and different** -- refuse, naming every setting and
      both values. Not overridable: there is nothing to interpret, the model
      was fitted on other numbers.
    * **one side unknown** -- refuse, but waivable per table through
      ``unverified_ok``. This is "I cannot check", which is a different claim
      from "these disagree".
    * **the model predates recording** -- refuse unless
      ``allow_unrecorded=True``. No ridge or classifier model has ever
      shipped, so this costs nobody anything and closes the case that
      reported a prediction seven years wrong in silence.

    A model that reads text (a topic model) re-derives its own instrument
    from its own file and has nothing to check.
    """
    from .helpers import provenance as prov

    if info.needs != "features":
        return
    expected = dict(info.provenance or {})
    if not expected:
        if allow_unrecorded:
            if verbose:
                print(f"[score] {info.display()} predates settings-recording; "
                      f"scoring without checking how its features were "
                      f"measured.")
            return
        raise ValueError(
            f"{info.display()} does not record how its features were "
            f"measured, so there is no way to tell whether this run's "
            f"features match the ones it was fitted on. Re-fit it, or pass "
            f"allow_unrecorded=True to score it anyway and accept that the "
            f"numbers may not be comparable.")

    actual = {}
    for path in feature_csvs:
        path = Path(path)
        # the same shape the model stored at fit time, from the same
        # function. we used to hand-build a copy here, and it left out the
        # chain digest -- so the one comparison that notices differently
        # prepared text never ran.
        actual[path.stem] = prov.comparable(prov.read(path))

    mismatched, unverifiable = prov.compare_tables(expected, actual)
    if mismatched:
        stem, diffs = mismatched[0]
        raise ValueError(prov.explain(info.display(), diffs, table=stem))

    waived = {str(s) for s in unverified_ok}
    blocked = [s for s in unverifiable if s not in waived]
    if blocked:
        raise ValueError(
            f"{info.display()} was fitted on the "
            f"{', '.join(repr(s) for s in blocked)} feature "
            f"{'table' if len(blocked) == 1 else 'tables'}, and this run "
            f"has no record of how {'it was' if len(blocked) == 1 else 'they were'} "
            f"measured -- so the settings cannot be compared. Re-run the "
            f"extraction so a record is written, or pass "
            f"unverified_ok={tuple(blocked)!r} to score without checking "
            f"{'that table' if len(blocked) == 1 else 'those tables'}.")
    if verbose and waived & set(unverifiable):
        print(f"[score] scoring without checking: "
              f"{', '.join(sorted(waived & set(unverifiable)))}")


def _needs_sentence(info: ModelInfo) -> str:
    """
    Name the feature tables to add, when the model recorded them.

    "165 columns are missing" tells a user what is wrong and not what to do.
    The model knows which steps produced its predictors, so the message can
    name those instead -- which is the actual next action.
    """
    if not info.needs_tables:
        return ("Add the step(s) that produce those columns (this model did "
                "not record which ones they were).")
    listed = ", ".join(f"'{name}'" for name in info.needs_tables)
    return (f"It was fitted on the {listed} feature "
            f"{'table' if len(info.needs_tables) == 1 else 'tables'}, so "
            f"select that step too.")


def _control_spec(info: ModelInfo):
    """The model's control recipe, or nothing. Read once by `describe`; this
    used to re-open the model file for it, a second read of a file that had
    just been parsed."""
    return list(info.controls)


def _report_join(work: Path, *, verbose: bool) -> list:
    """
    The join's row accounting, which is what explains a row that is not
    in the scores file at all.

    Returns ``[(table, rows_lost), ...]`` for every feature table the join
    lost rows at, and echoes the total when verbose. The join's own
    manifest lives in a work folder that is deleted after scoring, so what
    it says has to be carried out of there before it goes.
    """
    import json

    manifest = work / "assemble_manifest.json"
    if not manifest.is_file():
        return []
    try:
        with manifest.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception:
        return []
    kept = doc.get("rows_final")
    started = doc.get("rows_start")
    if verbose and kept is not None and started and kept < started:
        print(f"[score] {kept:,} of {started:,} row(s) appear in every "
              f"feature table and can be scored; the rest are missing at "
              f"least one column the model needs")
    lost = []
    for join in doc.get("joins") or []:
        before, after = join.get("rows_before"), join.get("rows_after")
        if before is not None and after is not None and after < before:
            lost.append(("missing from feature table", str(join.get("table")),
                         int(before) - int(after)))
        if join.get("rows_unmatched"):
            lost.append(("not in every earlier feature table",
                         str(join.get("table")), int(join["rows_unmatched"])))
    return lost


def _account_for_join(out_path: Path, lost, *, encoding: str) -> None:
    """
    Add the rows the join lost to the scorer's unscored-row accounting.

    The scorer writes ``<scores>_unscored.csv`` naming the predictor each
    blank row lacked; a row missing from a feature table altogether never
    reached the scorer and has to be added here, under its own reason, so
    one file answers "why is this text not scored" whatever the cause.
    """
    from .stats._fit_common import read_unscored, write_unscored

    if not lost:
        return
    entries = list(lost)
    entries += read_unscored(out_path, encoding=encoding)
    write_unscored(out_path, entries, encoding=encoding)


def _one_model(model_json: PathLike) -> ModelInfo:
    """One model, described -- or a refusal naming what was found instead."""
    
    return describe(one_model_path(model_json))


def _few(names: Sequence[str], limit: int = 8) -> str:
    from .stats._common import name_a_few

    return name_a_few(list(names), limit)


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings the
# old hand-written parser used, and we keep them so that every documented
# invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    score_with_model,
    description='Score a dataset with a saved model of any kind.',
    aliases={},
    legacy={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
