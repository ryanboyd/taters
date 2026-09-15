from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
import re
import pandas as pd
import numpy as np
from .atomic import atomic_write
from .progress import Ticker, announce
from .provenance import records_settings
from .cliargs import CliSpec
from .csvio import widen_csv_field_limit

PathLike = Union[str, Path]


# ---------------------------
# File discovery (single root)
# ---------------------------

def _iter_csv_files(
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
):
    """
    Yield CSV files under a root directory, optionally searching recursively.

    Parameters
    ----------
    root_dir : PathLike
        Root directory to search, or a single CSV file. If a file is passed
        and it matches `pattern`, it is yielded.
    pattern : str, default="*.csv"
        Glob pattern for file discovery.
    recursive : bool, default=True
        If True, search with ``Path.rglob``; otherwise use ``Path.glob`` in the
        top-level folder only.

    Yields
    ------
    pathlib.Path
        Absolute paths to files that match the pattern.

    Notes
    -----
    If `root_dir` does not exist, nothing is yielded. If `root_dir` is a file
    and matches `pattern`, it is yielded even though the typical use is a folder.
    """

    root = Path(root_dir)
    if not root.exists():
        return
    if root.is_file():
        # not the intended mode, but let's be permissive: if it's a file and
        # matches, yield it.
        if root.match(pattern):
            yield root.resolve()
        return
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


# ---------------------------
# Loading / concatenation
# ---------------------------

def _read_csv_add_source(
    path: Path,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
) -> pd.DataFrame:
    """
    Read a CSV and insert origin metadata columns.

    The function reads a CSV into a DataFrame (all columns as object dtype),
    then inserts a leading ``"source"`` column with the file stem. If
    ``add_source_path`` is True, it also inserts a ``"source_path"`` column
    immediately after ``"source"`` with the absolute path.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.
    delimiter : str, default=","
        Field delimiter for the CSV.
    encoding : str, default="utf-8-sig"
        Text encoding for the CSV file.
    add_source_path : bool, default=False
        If True, include a ``"source_path"`` column with the absolute path.
    source_col_stem : str, default="source_stem"
        Deprecated/compatibility parameter; not used.
    source_col_path : str, default="source_path"
        Deprecated/compatibility parameter; not used.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame with ``"source"`` (and optionally ``"source_path"``)
        inserted at the beginning.

    Notes
    -----
    The CSV field size limit is raised to handle very large text fields. Non-fatal
    read errors should be handled by the caller.
    """

    # let very large cells through (long transcripts blow past the default)
    widen_csv_field_limit()

    df = pd.read_csv(path, dtype="object", sep=delimiter, encoding=encoding)

    def _next_available(base: str, existing: set[str]) -> str:
        """Return base.N with the smallest N>=1 not in existing."""
        n = 1
        candidate = f"{base}.{n}"
        while candidate in existing:
            n += 1
            candidate = f"{base}.{n}"
        return candidate

    def _make_room(name: str) -> None:
        """Rename an input column that collides with one we are about to add
        (`source`, `source_path`) to `name.N`, so nothing is overwritten."""
        cols = list(df.columns)
        if name not in cols:
            return
        taken = set(cols)
        renamed = []
        for col in cols:
            if col == name:
                col = _next_available(name, taken)
                taken.add(col)
            renamed.append(col)
        df.columns = renamed

    # the pipeline's own `source` (and `source_path`) lead the table; if an
    # input column has the same name, we keep it (renamed) rather than clobber it.
    _make_room("source")
    df.insert(0, "source", path.stem)

    if add_source_path:
        _make_room("source_path")
        # place our 'source_path' right after 'source' if possible
        insert_at = 1 if "source" in df.columns else 0
        df.insert(insert_at, "source_path", str(path.resolve()))

    return df


# ---------------------------------
# Figure out what we're going to do
# ---------------------------------

def make_plan(
    *,
    group_by: Sequence[str],
    per_file: bool = True,
    stats: Sequence[str] = ("mean", "std"),
    exclude_cols: Sequence[str] = (),
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    dropna: bool = False,
) -> AggregationPlan:
    """
    Create an :class:`AggregationPlan` from simple arguments.

    Parameters
    ----------
    group_by : Sequence[str]
        Grouping key(s) to use (e.g., ``["speaker"]``).
    per_file : bool, default=True
        If True, group within files by including ``"source"`` in group keys.
    stats : Sequence[str], default=("mean", "std")
        Statistical reductions to compute per numeric column.
    exclude_cols : Sequence[str], default=()
        Columns to drop prior to feature selection.
    include_regex : str or None, default=None
        Regex to include feature columns by name.
    exclude_regex : str or None, default=None
        Regex to exclude feature columns by name.
    dropna : bool, default=False
        Drop rows with NA in any group key. Off by default so rows with a
        missing key are still reported rather than silently discarded.

    Returns
    -------
    AggregationPlan
        A configured plan instance for :func:`aggregate_features`.
    """

    return AggregationPlan(
        group_by=tuple(group_by),
        per_file=per_file,
        stats=tuple(stats),
        exclude_cols=tuple(exclude_cols),
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        dropna=dropna,
    )


def _promote_inner_keys(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """
    When both a file-level key (e.g., 'source') and an inner CSV key
    that was renamed to 'source.N' exist, promote the inner key to the
    base name and demote/rename the file-level key to 'file_<key>'.

    This is intended for per_file=False: we want to aggregate across files
    using the *inner* keys rather than the injected file stem.
    """
    def _unique(col: str, existing: set[str]) -> str:
        if col not in existing:
            return col
        i = 1
        while f"{col}.{i}" in existing:
            i += 1
        return f"{col}.{i}"

    cols = list(df.columns)
    existing = set(cols)

    for base in keys:
        # go looking for numbered variants like base.1, base.2, ...
        numbered = [c for c in cols if c == f"{base}.1" or c.startswith(f"{base}.")]
        has_base = base in existing
        if numbered:
            numbered.sort()
            inner_col = numbered[0]              # first after sorting, so it's stable
            if has_base:
                # demote the file-level base to 'file_<base>' (uniquified if needed)
                demoted = _unique(f"file_{base}", existing)
                df.rename(columns={base: demoted}, inplace=True)
                existing.discard(base)
                existing.add(demoted)
            # now we promote the inner column up to the base name
            df.rename(columns={inner_col: base}, inplace=True)
            existing.discard(inner_col)
            existing.add(base)
            # refresh these for the next time around
            cols = list(df.columns)
            existing = set(cols)

    return df


# -----------------------------------------------------
# Gather features sub-functions, abstracted and unified
# -----------------------------------------------------

def feature_gather(
    *,
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
    # toggle aggregation; when True you must pass a plan (or plan_args below)
    aggregate: bool = False,
    plan: Optional[AggregationPlan] = None,
    # optional “quick plan” args (only used if plan=None and aggregate=True)
    group_by: Optional[Sequence[str]] = None,
    per_file: bool = True,
    stats: Sequence[str] = ("mean", "std"),
    exclude_cols: Sequence[str] = (),
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    dropna: bool = False,
    # output
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Single entry point to concatenate or aggregate feature CSVs from one folder.

    If ``aggregate=False``, CSVs are concatenated with origin metadata
    (see :func:`gather_csvs_to_one`). If ``aggregate=True``, numeric feature
    columns are aggregated per the provided or constructed plan
    (see :func:`aggregate_features`).

    Parameters
    ----------
    root_dir : PathLike
        Folder containing per-item CSVs (or a single CSV file).
    pattern : str, default="*.csv"
        Glob pattern for selecting CSV files.
    recursive : bool, default=True
        Recurse into subdirectories when True.
    delimiter : str, default=","
        CSV delimiter.
    encoding : str, default="utf-8-sig"
        CSV encoding.
    add_source_path : bool, default=False
        If True, include a ``"source_path"`` column in outputs.
    aggregate : bool, default=False
        Toggle aggregation mode. If False, files are concatenated.
    plan : AggregationPlan or None, default=None
        Explicit plan for aggregation. Required if ``aggregate=True`` and
        ``group_by`` is not given.
    group_by : Sequence[str] or None, default=None
        Quick-plan keys. Used only when ``aggregate=True`` and ``plan`` is None.
    per_file : bool, default=True
        Quick-plan flag; include ``"source"`` in grouping keys to aggregate per file.
    stats : Sequence[str], default=("mean", "std")
        Quick-plan statistics to compute per numeric column.
    exclude_cols : Sequence[str], default=()
        Quick-plan columns to drop before numeric selection.
    include_regex : str or None, default=None
        Quick-plan regex to include feature columns by name.
    exclude_regex : str or None, default=None
        Quick-plan regex to exclude feature columns by name.
    dropna : bool, default=False
        Quick-plan NA handling for group keys. When True, rows whose group key
        is missing are dropped before aggregating.
    out_csv : PathLike or None, default=None
        Output CSV path. If None, defaults to
        ``<root_dir_parent>/<root_dir_name>.csv``.
    overwrite_existing : bool, default=False
        If False and `out_csv` exists, the existing path is returned without
        recomputation.

    Returns
    -------
    pathlib.Path
        Path to the resulting CSV.

    Raises
    ------
    ValueError
        If ``aggregate=True`` and neither ``plan`` nor ``group_by`` is provided.

    See Also
    --------
    gather_csvs_to_one : Concatenate CSVs with origin metadata.
    aggregate_features : Aggregate numeric columns according to a plan.
    """

    if not aggregate:
        return gather_csvs_to_one(
            on_progress=on_progress,
            root_dir=root_dir,
            pattern=pattern,
            recursive=recursive,
            delimiter=delimiter,
            encoding=encoding,
            add_source_path=add_source_path,
            out_csv=out_csv,
            overwrite_existing=overwrite_existing,
            verbose=verbose,
        )

    # aggregate=True
    if plan is None:
        if not group_by:
            raise ValueError("When aggregate=True, you must provide 'plan' or 'group_by'.")
        plan = make_plan(
            group_by=group_by,
            per_file=per_file,
            stats=stats,
            exclude_cols=exclude_cols,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            dropna=dropna,
        )

    return aggregate_features(
        on_progress=on_progress,
        root_dir=root_dir,
        pattern=pattern,
        recursive=recursive,
        delimiter=delimiter,
        encoding=encoding,
        add_source_path=add_source_path,
        plan=plan,
        out_csv=out_csv,
        overwrite_existing=overwrite_existing,
        verbose=verbose,
    )


#: The gather steps turn per-item acoustics and per-utterance embeddings into
#: one joinable table, and their own settings are plumbing. The ones that
#: decided the numbers -- the Whisper model, the embedding model, the
#: acoustics thresholds -- live upstream, which is why `root_dir` is binding:
#: the chain walk follows it and folds in the records beside the files it
#: read. One tuple for both writers: only the aggregating one carried it, so
#: the plain concatenation (the acoustics summary) wrote no record and a
#: model fitted on it could never be checked.
GATHER_BINDING = ("root_dir", "pattern", "recursive", "delimiter",
                  "encoding", "add_source_path")


@records_settings(binding=GATHER_BINDING, outputs=("out_csv",))
def gather_csvs_to_one(
    *,
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Concatenate many CSVs into a single CSV with origin metadata.

    Each input CSV is loaded (all columns as object dtype), a leading
    ``"source"`` column is inserted (and optionally ``"source_path"``), and
    rows are appended. The final CSV ensures ``"source"`` (and, if present,
    ``"source_path"``) lead the column order.

    Parameters
    ----------
    root_dir : PathLike
        Folder containing CSVs, or a single CSV file.
    pattern : str, default="*.csv"
        Glob pattern for selecting files.
    recursive : bool, default=True
        Recurse into subdirectories when True.
    delimiter : str, default=","
        CSV delimiter.
    encoding : str, default="utf-8-sig"
        CSV encoding for read/write.
    add_source_path : bool, default=False
        If True, include absolute path in ``"source_path"``.
    out_csv : PathLike or None, default=None
        Output path. If None, defaults to
        ``<root_dir_parent>/<root_dir_name>.csv``.
    overwrite_existing : bool, default=False
        If False and `out_csv` exists, return it without recomputation.

    Returns
    -------
    pathlib.Path
        Path to the written CSV.

    Raises
    ------
    FileNotFoundError
        If no files match the pattern under `root_dir`.
    RuntimeError
        If files were found but none could be read successfully.

    Notes
    -----
    Input rows are not type-coerced beyond object dtype. Column order from
    inputs is preserved after the leading origin columns.
    """

    root = Path(root_dir)
    if out_csv is None:
        out_csv = root.parent / f"{root.name}.csv"

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite_existing:
        if verbose:
            print(f"Aggregated feature output file already exists; returning existing file: {out_csv}")
        return out_csv

    files = list(_iter_csv_files(root, pattern=pattern, recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {root}")

    # we tick once per table read: a big embeddings folder took a minute under
    # a bare spinner, which looks an awful lot like a hang.
    announce(on_progress, "reading feature tables")
    ticker = Ticker(on_progress, len(files))
    frames = []
    for fp in files:
        try:
            frames.append(
                _read_csv_add_source(
                    fp,
                    delimiter=delimiter,
                    encoding=encoding,
                    add_source_path=add_source_path,
                )
            )
        except Exception as e:
            if verbose:
                print(f"[gather] WARNING: failed to read {fp}: {e}")
        ticker.tick(message="reading feature tables")

    if not frames:
        raise RuntimeError("No CSVs could be read successfully.")

    merged = pd.concat(frames, axis=0, ignore_index=True)

    # make sure 'source' is first (and 'source_path' next, if we have it)
    cols = list(merged.columns)
    if "source" in cols:
        lead = ["source"] + (["source_path"] if "source_path" in cols else [])
        rest = [c for c in cols if c not in lead]
        merged = merged[lead + rest]

    # atomic write, like every other skip-if-exists table: the next run will
    # treat this one as finished, so a Ctrl-C mid-write used to leave a sticky
    # truncated table behind.
    with atomic_write(out_csv, mode="w", newline="", encoding=encoding) as fh:
        merged.to_csv(fh, index=False)
    return out_csv


# ---------------------------
# Aggregation
# ---------------------------

@dataclass
class AggregationPlan:
    """
    Plan describing how numeric feature columns should be aggregated.

    Parameters
    ----------
    group_by : Sequence[str]
        One or more column names used as grouping keys (e.g., ``["speaker"]``).
    per_file : bool, default=True
        If True, include ``"source"`` in the grouping keys to aggregate within
        each input file; if False, aggregate across all files globally.
    stats : Sequence[str], default=("mean", "std")
        Statistical reductions to compute for each numeric feature column.
        Values are passed to ``pandas.DataFrame.agg`` (e.g., ``"mean"``, ``"std"``,
        ``"median"``, etc.).
    exclude_cols : Sequence[str], default=()
        Columns to drop before filtering/selecting numeric features (e.g.,
        timestamps or free text).
    include_regex : str or None, default=None
        Optional regex; if provided, only columns matching this pattern are kept
        (after excluding `exclude_cols`).
    exclude_regex : str or None, default=None
        Optional regex; if provided, columns matching this pattern are removed
        (after applying `include_regex`, if any).
    dropna : bool, default=False
        Whether to drop rows with NA in any of the group-by keys before grouping.
        The default keeps them, so rows with a missing key land in their own
        clearly-labeled group instead of vanishing from the output.

    Notes
    -----
    This plan is consumed by :func:`aggregate_features`. Column filtering happens
    before numeric selection; only columns that remain and can be coerced to numeric
    will be aggregated.
    """

    group_by: Sequence[str]
    per_file: bool = True
    stats: Sequence[str] = ("mean", "std")
    exclude_cols: Sequence[str] = ()
    include_regex: Optional[str] = None
    exclude_regex: Optional[str] = None
    dropna: bool = False


def _filter_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: Sequence[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    must_keep: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Filter columns prior to numeric feature selection.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame containing group keys and feature columns.
    exclude_cols : Sequence[str]
        Columns to drop unconditionally.
    include_regex : str or None
        If provided, keep only columns whose names match this regex.
    exclude_regex : str or None
        If provided, drop columns whose names match this regex (applied after
        `include_regex`).
    must_keep : Sequence[str] or None

    Returns
    -------
    pandas.DataFrame
        A view of `df` with columns filtered according to the rules.
    """

    cols = list(df.columns)
    keep = [c for c in cols if c not in set(exclude_cols)]
    if include_regex:
        rx = re.compile(include_regex)
        keep = [c for c in keep if rx.search(c)]
    if exclude_regex:
        rx = re.compile(exclude_regex)
        keep = [c for c in keep if not rx.search(c)]
    # always hang onto the group keys (dropping duplicates, but keeping order).
    # we only re-add keys that actually exist: selecting a missing column here
    # would raise a bare pandas KeyError and beat the caller's much clearer
    # "Missing group-by columns in data: [...]" error to the punch.
    present = set(df.columns)
    keep = list(dict.fromkeys([k for k in must_keep if k in present] + keep))
    return df[keep]


def _numeric_subframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns to numeric and return only numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame containing candidate feature columns.

    Returns
    -------
    pandas.DataFrame
        Frame consisting only of columns that could be coerced to numeric
        (non-numeric values become NaN and non-numeric columns are dropped).

    Notes
    -----
    Uses ``pandas.to_numeric(errors="coerce")`` followed by
    ``select_dtypes(include=[numpy.number])``.
    """

    num_df = df.apply(pd.to_numeric, errors="coerce")
    num_df = num_df.select_dtypes(include=[np.number])
    # a column of pure text coerces to all-NaN, which is still a float column --
    # if we kept it we'd spit out meaningless "<textcol>__mean" columns full of
    # NaN. any column where at least one value parsed as a number, we keep.
    usable = [c for c in num_df.columns if num_df[c].notna().any()]
    return num_df[usable]


@records_settings(binding=GATHER_BINDING, outputs=("out_csv",))
def aggregate_features(
    *,
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
    plan: AggregationPlan,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Discover files, read, concatenate, and aggregate numeric columns per plan.

    This function consolidates CSVs from a single folder, filters columns,
    coerces candidate features to numeric, groups by the specified keys,
    and computes the requested statistics. Output columns for aggregated
    features are flattened with the pattern ``"{column}__{stat}"``.

    Parameters
    ----------
    root_dir : PathLike
        Folder containing per-item CSVs, or a single CSV file.
    pattern : str, default="*.csv"
        Glob pattern for selecting files.
    recursive : bool, default=True
        Recurse into subdirectories when True.
    delimiter : str, default=","
        CSV delimiter.
    encoding : str, default="utf-8-sig"
        CSV encoding for read/write.
    add_source_path : bool, default=False
        If True, include absolute path in ``"source_path"`` prior to filtering.
    plan : AggregationPlan
        Aggregation configuration (group keys, stats, filters, NA handling).
    out_csv : PathLike or None, default=None
        Output path. If None, defaults to
        ``<root_dir_parent>/<root_dir_name>.csv``.
    overwrite_existing : bool, default=False
        If False and `out_csv` exists, return it without recomputation.

    Returns
    -------
    pathlib.Path
        Path to the written CSV of aggregated features.

    Raises
    ------
    FileNotFoundError
        If no files match the pattern under `root_dir`.
    RuntimeError
        If files were found but none could be read successfully.
    ValueError
        If required group-by columns are missing,
        or if no numeric columns remain after filtering,
        or if per-file grouping is requested but the ``"source"`` column is absent.

    Notes
    -----
    Group keys are preserved as leading columns in the output. The output places
    ``"source"`` (and optionally ``"source_path"``) first when present.
    """

    root = Path(root_dir)
    if out_csv is None:
        out_csv = root.parent / f"{root.name}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite_existing:
        if verbose:
            print(f"Aggregated feature output file already exists; returning existing file: {out_csv}")
        return out_csv

    files = list(_iter_csv_files(root, pattern=pattern, recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {root}")

    announce(on_progress, "reading feature tables")
    ticker = Ticker(on_progress, len(files))
    frames = []
    for fp in files:
        try:
            frames.append(
                _read_csv_add_source(
                    fp,
                    delimiter=delimiter,
                    encoding=encoding,
                    add_source_path=add_source_path,
                )
            )
        except Exception as e:
            if verbose:
                print(f"[aggregate] WARNING: failed to read {fp}: {e}")
        ticker.tick(message="reading feature tables")

    if not frames:
        raise RuntimeError("No CSVs could be read successfully.")

    df = pd.concat(frames, axis=0, ignore_index=True)

    # if we're aggregating across files (per_file=False), we promote the inner
    # keys (e.g., 'source.1' -> 'source') and demote the file-level keys.
    if not plan.per_file:
        df = _promote_inner_keys(df, plan.group_by)


    def _resolve_keys(base_keys, columns):
        cols = set(columns)
        resolved = []
        for k in base_keys:
            if k in cols:
                resolved.append(k)
                continue
            # go looking for numbered variants like 'k.1', 'k.2', ...
            prefix = f"{k}."
            candidates = [c for c in columns if c == f"{k}.1" or c.startswith(prefix)]
            if candidates:
                # sorted first, so we pick the same one every time
                resolved.append(sorted(candidates)[0])
            else:
                # leave it unresolved; we'll error out below with a helpful message
                resolved.append(k)
        return resolved

    # build the (base) group keys from the plan
    group_keys = list(plan.group_by)
    if plan.per_file:
        if "source" not in df.columns:
            raise ValueError("source column is missing; cannot group per_file.")
        group_keys = ["source"] + group_keys

    # resolve any collisions against the columns we actually have
    group_keys = _resolve_keys(group_keys, df.columns)

    # now we filter, but we ALWAYS keep the group keys
    df_f = _filter_columns(
        df,
        exclude_cols=tuple(plan.exclude_cols) + ("source_path",),
        include_regex=plan.include_regex,
        exclude_regex=plan.exclude_regex,
        must_keep=group_keys,
    )

    missing = [k for k in group_keys if k not in df_f.columns]
    if missing:
        raise ValueError(f"Missing group-by columns in data: {missing}")

    # everything that isn't a key is a candidate feature; keep the numeric ones
    feature_cols = [c for c in df_f.columns if c not in set(group_keys)]
    numeric_df = _numeric_subframe(df_f[feature_cols])
    if numeric_df.empty:
        raise ValueError("No numeric columns available for aggregation after filtering.")

    # stick the group keys back on so we can group by them
    gdf = pd.concat([df_f[group_keys].reset_index(drop=True),
                     numeric_df.reset_index(drop=True)], axis=1)

    agg_ops = {c: list(plan.stats) for c in numeric_df.columns}
    grouped = gdf.groupby(group_keys, dropna=plan.dropna).agg(agg_ops)

    # flatten the MultiIndex columns down to '<col>__<stat>'
    grouped.columns = [f"{c}__{stat}" for (c, stat) in grouped.columns]
    grouped = grouped.reset_index()

    # make sure 'source' (and 'source_path' if present) lead the output
    cols = list(grouped.columns)
    lead = [c for c in ("source", "source_path") if c in cols]
    rest = [c for c in cols if c not in lead]
    grouped = grouped[lead + rest]

    with atomic_write(out_csv, mode="w", newline="", encoding=encoding) as fh:
        grouped.to_csv(fh, index=False)
    return out_csv


# ---------------------------
# Minimal CLI
# ---------------------------


# ---------------------------------------------------------------------------
# Command line -- derived from the functions above; see helpers.cliargs.CliSpec.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"gather": gather_csvs_to_one, "aggregate": aggregate_features,
     "run": feature_gather},
    description="Concatenate or aggregate the feature tables under a folder.",
    legacy={"--no-recursive": ["--recursive", "false"]},
    # we build the plan from the other arguments (`group_by`, `stats`, ...);
    # there's no sensible way to type one out on a command line.
    skip=("plan",),
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
