"""
Build the analysis table: every chosen feature table, one row per text,
joined with the metadata columns the statistics need.

The feature extractors each write their own CSV keyed by ``text_id`` (or by a
multi-column key for media runs). Statistics want ONE wide table -- features
side by side, with the grouping and outcome columns from the user's
spreadsheet -- and nothing in Taters joined columns before this module: the
gathers stack rows, they never widen.

The join is INNER, with accounting. A left join would seed missing values
into every downstream analysis for rows some extractor skipped (a zero-token
document missing from a term matrix, say); keeping only rows every table has
is the honest denominator, and honesty about the cost lives in the manifest:
rows in, rows surviving each join, rows each filter removed. Those counts are
also written into the report, because "we analyzed 99,881 of your 100,000
rows, and here is where the rest went" is the first thing a reviewer asks.

Filters are the "select * where" of the stage: ``[column, op, value]``
triples a row must ALL satisfy to stay. A blank (missing) cell fails every
filter -- a row whose word count is unknown does not sneak past
``word_count >= 25``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Literal, Mapping, Optional, Sequence, Union

from ..helpers import row_filter as _row_filter
from ..helpers.atomic import atomic_write
from ..helpers.feature_columns import SEPARATOR
from ..helpers.progress import announce
from ._common import (looks_numeric, name_a_few, taters_version,
                      write_section, reusable)
from ..helpers.cliargs import CliSpec
from ._common import read_str_csv

PathLike = Union[str, Path]

#: The comparison operators a filter may use. Ordering operators demand
#: numbers on both sides; the rest compare numerically when both sides parse
#: and as stripped strings otherwise.
#: Re-exported: the comparison now lives beside the text gather's copy of the
#: same question, because a filter has to mean the same thing before the text
#: is joined and after it has been measured. See `helpers.row_filter`.
FILTER_OPS = _row_filter.FILTER_OPS


def _refuse_duplicate_keys(df, key_cols: Sequence[str], label: str) -> None:
    """A duplicated join key silently multiplies rows in a join -- refuse,
    naming the file and the first offending key, while the user can act."""
    dupes = df.duplicated(subset=list(key_cols), keep=False)
    if bool(dupes.any()):
        first = df.loc[dupes, list(key_cols)].iloc[0].tolist()
        raise ValueError(
            f"{label} has more than one row for the key "
            f"{tuple(first)!r}. Every input to the analysis table must have "
            f"exactly one row per key, or the join would multiply rows."
        )


_matches = _row_filter.matches


def _validate_filters(filters, columns: Sequence[str], sample) -> List[list]:
    """
    The filter spec, checked while the user can still fix it.

    Shape, operator, column existence, and -- for ordering operators -- that
    the column and the value actually hold numbers, judged over the whole
    column rather than discovered as a crash halfway through a run.
    """
    available = set(columns)
    out: List[list] = []
    for spec in filters or []:
        spec = list(spec)
        if len(spec) != 3:
            raise ValueError(
                f"a filter is [column, operator, value]; got {spec!r}")
        column, op, value = str(spec[0]), str(spec[1]), spec[2]
        if op not in FILTER_OPS:
            raise ValueError(
                f"filter operator {op!r} is not one of {list(FILTER_OPS)}")
        if column not in available:
            raise ValueError(
                f"filter column {column!r} is not in the analysis table -- "
                f"nothing this run measured writes a column by that name. "
                f"It has {len(available)}: {name_a_few(sorted(available))}. "
                f"The full list is the header of analysis_table.csv.")
        if op in ("in", "not_in") and not isinstance(value, (list, tuple, set)):
            raise ValueError(
                f"filter [{column}, {op}, …] needs a list of values, "
                f"got {value!r}")
        if op in ("<", "<=", ">", ">="):
            try:
                float(str(value).strip())
            except ValueError:
                raise ValueError(
                    f"filter [{column}, {op}, {value!r}]: the value is not "
                    f"a number, and {op!r} orders numbers") from None
            if not looks_numeric(sample[column].tolist()):
                raise ValueError(
                    f"filter [{column}, {op}, …]: column {column!r} holds "
                    f"non-numeric values, and {op!r} orders numbers")
        out.append([column, op, value])
    return out


def assemble_analysis_table(
    *,
    feature_csvs: Sequence[PathLike],
    filter_csvs: Sequence[PathLike] = (),
    metadata_csv: Optional[PathLike] = None,
    metadata_cols: Sequence[str] = (),
    key_cols: Sequence[str] = ("text_id",),
    text_cols: Sequence[str] = (),
    split_col: Optional[str] = None,
    filters: Optional[Sequence[Sequence]] = None,
    bookkeeping: Literal["aside", "features"] = "aside",
    bookkeeping_cols: Sequence[str] = (),
    out_dir: PathLike = "stats_results",
    out_csv: Optional[PathLike] = None,
    keep_table: bool = True,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
) -> Path:
    """
    Join feature tables and metadata into one wide analysis table.

    Parameters
    ----------
    feature_csvs
        The feature tables to include, each with exactly one row per key.
        Non-numeric columns other than the key are dropped (with a note):
        the analyses consume numbers, and a stray ``text`` column riding
        along in a feature file is weight, not information.
    text_cols
        Columns to carry through even though they are not numbers. Needed
        because a saved model fitted with a categorical control knows a
        predictor called ``gender=male`` and the table being scored still
        holds ``gender`` as the words the researcher typed -- dropped as
        "non-numeric", the control could never be rebuilt. Named
        explicitly, never guessed: a stray ``text`` column riding along in
        a feature file really is weight rather than information.
    filter_csvs
        Tables joined in so their columns can be *filtered on*, but never
        offered to the analyses as features. A word count computed because
        someone asked to drop texts under 25 words is a gate, not a
        predictor -- and it turned up as one in a ridge regression, which is
        the kind of result that looks fine and is not (a real report). Their
        columns land in the analysis table and in the sidecar's ``filters``
        list; only ``feature_csvs`` columns reach ``sets``, which is what
        every analysis reads to decide what a feature is.
    metadata_csv
        Optional table holding the grouping/outcome columns (typically the
        run's ``gathered/metadata.csv``). When omitted, the table is features
        only -- fine for a PCA, useless for an ANOVA.
    metadata_cols
        Which metadata columns to carry. Empty means all of them (minus any
        column literally named ``text``).
    key_cols
        The join key. ``("text_id",)`` for text runs; media runs can key on
        several columns.
    split_col
        A second dimension the *feature* tables are keyed on but the
        metadata is not -- ``"source_col"``, when a spreadsheet's text
        columns were measured one at a time. Each text then has one feature
        row per column, while the participant has one row of metadata, so
        the features join to each other on the key plus this column and the
        metadata broadcasts across them. The analyses split on it in turn,
        so one person's several answers are never treated as several
        independent observations.
    filters
        Row filters, ``[column, op, value]`` triples that a row must ALL
        satisfy to stay. Operators: ``== != < <= > >= in not_in``. A blank
        cell fails every filter. Each filter's removed-row count is recorded.
    bookkeeping
        What to do with the count columns the steps write beside their
        measures -- a matrix's ``token_count``, a dictionary's ``WC``, a
        readability step's raw sentence and syllable counts -- which each
        analyzer declares in the record beside its table. ``"aside"`` (the
        default) joins them into the table, where they can be filtered on
        and looked at, but keeps them out of the feature sets every analysis
        reads; ``"features"`` analyses them like any other measure. A ridge
        once learned from ``token_count`` because nothing told it apart.
    bookkeeping_cols
        Column names to keep aside as well, for a table with no record to
        declare them (one made by hand, or by an older build).
    out_dir, out_csv
        Where the table goes; ``out_csv`` defaults to
        ``<out_dir>/analysis_table.csv``. Two sidecars land next to it:
        ``analysis_table_sets.json`` (which columns came from which feature
        table -- the handle for per-feature-set analyses) and
        ``assemble_manifest.json`` (the row accounting).
    keep_table
        Whether the merged table survives the run. It is written either way
        -- every analysis reads it from disk -- but with False the report
        step deletes it at the end. Keep it when you want the dataset the
        statistics were computed on, which is usually: it is the file to
        hand to a colleague, or to open in R. Turn it off when it is
        enormous and derivable, since a wide feature set makes it the
        largest thing the run produces.
    overwrite_existing
        When False (default) and the table exists, it is returned untouched.

    Returns
    -------
    Path
        The analysis table.
    """
    import json


    feature_paths = [Path(p) for p in (feature_csvs or [])]
    filter_paths = [Path(p) for p in (filter_csvs or [])]
    if not feature_paths:
        raise ValueError(
            "feature_csvs is empty: nothing to assemble"
            + (" -- every table given was a filter-only one, and an analysis "
               "needs something to analyze" if filter_paths else ""))
    key_cols = [str(k) for k in key_cols]
    if not key_cols:
        raise ValueError("key_cols is empty: there is nothing to join on")
    split_col = str(split_col) if split_col else None
    # the feature tables carry the extra (split) dimension; the metadata doesn't.
    feature_key = key_cols + ([split_col] if split_col else [])

    out_dir = Path(out_dir)
    out_path = Path(out_csv) if out_csv else out_dir / "analysis_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sets_path = out_path.with_name(out_path.stem + "_sets.json")
    manifest_path = out_path.parent / "assemble_manifest.json"

    if reusable(out_path, feature_csvs, filter_csvs, metadata_csv,
                overwrite_existing=overwrite_existing, verbose=verbose,
                what="the analysis table"):
        if verbose:
            print(f"Analysis table already exists; returning existing file: "
                  f"{out_path}")
        return out_path

    if bookkeeping not in ("aside", "features"):
        raise ValueError(
            f"bookkeeping must be 'aside' or 'features', got {bookkeeping!r}")
    manifest: dict = {"key": feature_key, "joins": [], "filters": [],
                      "filter_only": [], "bookkeeping": {},
                      "bookkeeping_mode": bookkeeping,
                      "dropped_columns": {}, "renamed_columns": {},
                      # we write this down rather than just acting on it,
                      # so that a folder with no analysis_table.csv in it
                      # can say whether that was a choice or a failure.
                      "kept": bool(keep_table)}

    # ------------------------------------------------------------------ read
    announce(on_progress, "reading the feature tables")
    frames: List[tuple] = []      # (stem, DataFrame w/ key + kept cols, is_feature)
    seen_stems: dict = {}
    # per table, the numeric columns that aren't actually measures: whatever
    # the analyzer declared in the record beside the table (see
    # `records_settings(bookkeeping=...)`), plus anything the caller names.
    # we join these in but keep them out of the feature sets unless asked --
    # we once had a ridge on a document-term matrix happily learning from
    # `token_count`. oops.
    aside: dict = {}
    named = {str(c) for c in bookkeeping_cols}
    for path, is_feature in ([(p, True) for p in feature_paths]
                             + [(p, False) for p in filter_paths]):
        if not path.is_file():
            raise FileNotFoundError(f"feature table not found: {path}")
        stem = path.stem
        if stem in seen_stems:
            raise ValueError(
                f"two feature tables share the name {stem!r} "
                f"({seen_stems[stem]} and {path}); rename one -- the name "
                f"labels its feature set in every result table.")
        seen_stems[stem] = path
        df = read_str_csv(path, encoding=encoding)
        missing = [k for k in feature_key if k not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing key column(s) {missing}; it cannot "
                f"be joined into the analysis table.")
        _refuse_duplicate_keys(df, feature_key, path.name)
        kept, dropped = [], []
        wanted = {str(c) for c in text_cols}
        for col in df.columns:
            if col in feature_key:
                continue
            if col in wanted or looks_numeric(df[col].tolist()):
                kept.append(col)
            else:
                dropped.append(col)
        if dropped:
            manifest["dropped_columns"][stem] = dropped
            if verbose:
                print(f"[assemble] {path.name}: dropping non-numeric "
                      f"column(s) {dropped}")
        if not kept:
            raise ValueError(
                f"{path.name} has no numeric feature columns beyond the key; "
                f"there is nothing in it to analyze.")
        frames.append((stem, df[feature_key + kept], is_feature))
        if not is_feature:
            manifest["filter_only"].append(stem)
        elif bookkeeping == "aside":
            declared = set(_declared_bookkeeping(path)) | named
            found = [c for c in kept if c in declared]
            if found:
                aside[stem] = found
                manifest["bookkeeping"][stem] = found

    # ----------------------------------------------------------- metadata
    meta = None
    meta_kept: List[str] = []
    if metadata_csv is not None:
        announce(on_progress, "reading the metadata")
        meta = read_str_csv(Path(metadata_csv), encoding=encoding)
        missing = [k for k in key_cols if k not in meta.columns]
        if missing:
            raise ValueError(
                f"{Path(metadata_csv).name} is missing key column(s) "
                f"{missing}; it cannot anchor the analysis table.")
        _refuse_duplicate_keys(meta, key_cols, Path(metadata_csv).name)
        if metadata_cols:
            absent = [c for c in metadata_cols if c not in meta.columns]
            if absent:
                spare = [c for c in meta.columns if c not in key_cols]
                raise ValueError(
                    f"metadata_cols not in {Path(metadata_csv).name}: "
                    f"{name_a_few(absent)}. It has {len(spare)}: "
                    f"{name_a_few(spare)}.")
            meta_kept = [str(c) for c in metadata_cols]
        else:
            meta_kept = [c for c in meta.columns
                         if c not in key_cols and c != "text"]
        meta = meta[key_cols + meta_kept]

    # ---------------------------------------------------------- collisions
    # here, we deal with name clashes. any feature column used by two feature
    # tables (or clashing with a key/metadata name) gets renamed
    # "<stem>__<col>" in EVERY feature table that has it -- all the colliders,
    # not just the later ones, so that the outcome doesn't depend on file
    # order. keys and metadata keep their names.
    #
    # this is the safety net, not the plan. the measures Taters ships declare
    # their column names up front and a build test refuses any two that could
    # agree (see helpers/feature_columns.py), because a rename that only
    # happens when something else is in the run makes a column's name depend on
    # the rest of the pipeline -- the same model, two studies, two names. what
    # is left for this to catch is what we cannot declare: the user's own
    # spreadsheet columns, their dictionary categories, two runs of one
    # instrument.
    #
    # `__` rather than `.`, both because the rest of the codebase already uses
    # it (score_model, _concept_dicts, feature_gather) and because a dot in a
    # column name has to be quoted in an R or patsy formula, which is exactly
    # where these tables end up.
    reserved = set(feature_key) | set(meta_kept)
    counts: dict = {}
    for _, df, _is_feature in frames:
        for col in df.columns:
            if col not in feature_key:
                counts[col] = counts.get(col, 0) + 1
    colliding = {c for c, n in counts.items() if n > 1} | \
                {c for c in counts if c in reserved}
    sets: dict = {}
    filter_columns: dict = {}
    bookkeeping_columns: dict = {}
    # we also keep track of where each table came from. without this, a model
    # knows the *names* of its predictors but has no way of getting back to
    # the record of how they were measured -- and that's the whole point of
    # recording it.
    sources: dict = {}
    renamed_frames = []
    for stem, df, is_feature in frames:
        renames = {c: f"{stem}{SEPARATOR}{c}" for c in df.columns
                   if c not in feature_key and c in colliding}
        if renames:
            df = df.rename(columns=renames)
            manifest["renamed_columns"].update(renames)
        columns_here = [c for c in df.columns if c not in feature_key]
        # only a feature table's columns become a *set*. every analysis reads
        # the sidecar's `sets` to decide what counts as a feature, so if a
        # filter-only table landed there we'd end up predicting from it. a
        # table's bookkeeping columns (renamed along with the rest if they
        # collided) go beside the sets, not in them.
        kept_aside = {renames.get(c, c) for c in aside.get(stem, ())}
        if is_feature and kept_aside:
            bookkeeping_columns[stem] = [c for c in columns_here
                                         if c in kept_aside]
            columns_here = [c for c in columns_here if c not in kept_aside]
        (sets if is_feature else filter_columns)[stem] = columns_here
        if is_feature and stem in seen_stems:
            sources[stem] = str(Path(seen_stems[stem]).resolve())
        renamed_frames.append((stem, df))

    # --------------------------------------------------------------- join
    announce(on_progress, "joining the tables")
    if meta is not None:
        table = meta
        start_label = Path(metadata_csv).name
    else:
        stem, table = renamed_frames[0]
        start_label = stem
        renamed_frames = renamed_frames[1:]
    manifest["rows_start"] = int(len(table))
    manifest["start"] = start_label
    for stem, df in renamed_frames:
        before = int(len(table))
        # the metadata is one row per text; a feature table under a split is
        # several. joining on what they share broadcasts the metadata across
        # all of a text's rows -- that's what "this participant's answers"
        # means, so it's what we want.
        on = [c for c in feature_key if c in table.columns and c in df.columns]
        # rows of this table with no partner so far get lost too, and the
        # before/after count can't see them: a text that's in every table but
        # the first would vanish from the join without showing up in any
        # number. what it's really missing from is an *earlier* table, so
        # that's how we count it.
        unmatched = int((df[on].drop_duplicates()
                         .merge(table[on].drop_duplicates(), on=on,
                                how="left", indicator=True)["_merge"]
                         == "left_only").sum())
        table = table.merge(df, how="inner", on=on)
        manifest["joins"].append({"table": stem, "rows_before": before,
                                  "rows_after": int(len(table)),
                                  "rows_unmatched": unmatched})
        if on_progress is not None:
            on_progress(len(manifest["joins"]), len(frames),
                        f"joined {stem}")

    if len(table) == 0:
        raise ValueError(
            "the join produced no rows: no key value appears in every input. "
            "Check that the metadata and feature tables describe the same "
            "texts and share the same key columns "
            f"({key_cols}). Row counts: {manifest['joins']}")

    # ------------------------------------------------------------- filters
    checked = _validate_filters(filters, table.columns, table)
    for column, op, value in checked:
        before = int(len(table))
        mask = table[column].map(lambda cell: _matches(cell, op, value))
        table = table[mask]
        manifest["filters"].append(
            {"filter": [column, op, value], "removed": before - int(len(table))})
    if checked and len(table) == 0:
        raise ValueError(
            "the filters removed every row. Removed per filter: "
            + "; ".join(f"{f['filter']} removed {f['removed']}"
                        for f in manifest["filters"]))
    manifest["rows_final"] = int(len(table))

    # -------------------------------------------------------------- write
    announce(on_progress, "writing the analysis table")
    with atomic_write(out_path, mode="w", encoding=encoding,
                      newline="") as fh:
        table.to_csv(fh, index=False)
    with atomic_write(sets_path, mode="w", encoding="utf-8") as fh:
        json.dump({"key": feature_key, "split": split_col or "",
                   "metadata": meta_kept, "sets": sets,
                   "sources": sources,
                   "filters": filter_columns,
                   "bookkeeping": bookkeeping_columns}, fh, indent=1)
    manifest["taters"] = taters_version()
    with atomic_write(manifest_path, mode="w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    write_section(out_path.parent, "assemble",
                  _section_md(manifest, sets))
    if verbose:
        print(f"[assemble] {manifest['rows_final']:,} rows × "
              f"{len(table.columns):,} columns -> {out_path}")
    return out_path


def _declared_bookkeeping(path: Path) -> List[str]:
    """
    The bookkeeping columns the analyzer declares for this table.

    From the record beside the table when it carries the declaration; from
    the analyzer itself -- looked up by the record's ``call`` -- when the
    table was made by a build from before the declaration existed. The
    declaration describes the analyzer's output, which the code knows now
    whatever the file's age; a real run kept `token_count` as a predictor
    because its matrix had been extracted the day before.
    """
    import importlib

    from ..helpers import provenance

    rec = provenance.read(path)
    if not rec:
        return []
    if "bookkeeping" in rec:
        return [str(c) for c in (rec.get("bookkeeping") or [])]
    call = str(rec.get("call") or "")
    if ":" not in call:
        return []
    module, _, name = call.partition(":")
    try:
        fn = importlib.import_module(module)
        for part in name.split("."):
            fn = getattr(fn, part)
    except Exception:
        return []
    declared = getattr(fn, "__provenance__", {}) or {}
    return [str(c) for c in (declared.get("bookkeeping") or ())]


def _section_md(manifest: dict, sets: Mapping[str, Sequence[str]]) -> str:
    lines = ["## The analysis table", ""]
    joins = manifest["joins"]
    total_features = sum(len(v) for v in sets.values())
    lines.append(
        f"Assembled **{manifest['rows_final']:,} rows** and "
        f"{total_features:,} feature columns from {len(sets)} feature "
        f"table(s), starting from {manifest['start']} "
        f"({manifest['rows_start']:,} rows).")
    for j in joins:
        lost = j["rows_before"] - j["rows_after"]
        note = f" (-{lost:,})" if lost else ""
        lines.append(f"- joined `{j['table']}`: {j['rows_after']:,} rows{note}")
    if manifest["filters"]:
        lines.append("")
        lines.append("Filters:")
        for f in manifest["filters"]:
            col, op, val = f["filter"]
            lines.append(f"- `{col} {op} {val!r}` removed {f['removed']:,} "
                         f"row(s)")
    for stem, cols in (manifest.get("dropped_columns") or {}).items():
        lines.append(f"- note: dropped non-numeric column(s) from "
                     f"`{stem}`: {', '.join(cols)}")
    aside = manifest.get("bookkeeping") or {}
    if aside:
        listed = "; ".join(f"`{stem}`: {', '.join(cols)}"
                           for stem, cols in aside.items())
        lines.append(
            f"- kept beside the features, not analyzed (counts the steps "
            f"report rather than measures): {listed}. They are in the table "
            f"to filter on; the shared setting *bookkeeping* = features "
            f"puts them into the analyses.")
    elif manifest.get("bookkeeping_mode") == "features":
        lines.append("- count columns the steps report (token and word "
                     "counts) are included as features, as asked.")
    lines.append("")
    if manifest.get("kept", True):
        lines.append("Details: `analysis_table.csv`, "
                     "`analysis_table_sets.json`, `assemble_manifest.json`.")
    else:
        lines.append("The merged table itself was not kept (you asked for "
                     "that); re-run with the 'keep table' setting on to save "
                     "it. Details: `assemble_manifest.json`.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    assemble_analysis_table,
    description='Join feature tables and metadata into one analysis table, with optional row filters.',
    aliases={
        'out_csv': ['--out'],
    },
    parsers={"filters": __import__("json").loads},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
