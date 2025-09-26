from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union
import re
import csv
import sys
import pandas as pd
import numpy as np

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
        # Not the intended mode, but be permissive: if it's a file and matches, yield it.
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
    # kept for backward compat but unused in layout now
    source_col_stem: str = "source_stem",
    source_col_path: str = "source_path",
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

    # Robust CSV field size (giant text cells)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    df = pd.read_csv(path, dtype="object", sep=delimiter, encoding=encoding)

    # New canonical name & position
    df.insert(0, "source", path.stem)

    # Optional full path just after source
    if add_source_path:
        df.insert(1, "source_path", str(path.resolve()))

    # (Optional backwards compatibility columns—uncomment if other code relies on them)
    # df["source_stem"] = path.stem

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
    dropna: bool = True,
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
    dropna : bool, default=True
        Drop rows with NA in any group key.

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
    dropna: bool = True,
    # output
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
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
    dropna : bool, default=True
        Quick-plan NA handling for group keys.
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
            root_dir=root_dir,
            pattern=pattern,
            recursive=recursive,
            delimiter=delimiter,
            encoding=encoding,
            add_source_path=add_source_path,
            out_csv=out_csv,
            overwrite_existing=overwrite_existing,
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
        root_dir=root_dir,
        pattern=pattern,
        recursive=recursive,
        delimiter=delimiter,
        encoding=encoding,
        add_source_path=add_source_path,
        plan=plan,
        out_csv=out_csv,
        overwrite_existing=overwrite_existing,
    )


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
        print(f"Aggregated feature output file already exists; returning existing file: {out_csv}")
        return out_csv

    files = list(_iter_csv_files(root, pattern=pattern, recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {root}")

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
            print(f"[gather] WARNING: failed to read {fp}: {e}")

    if not frames:
        raise RuntimeError("No CSVs could be read successfully.")

    merged = pd.concat(frames, axis=0, ignore_index=True)

    # Ensure 'source' is first (and 'source_path' next if present)
    cols = list(merged.columns)
    if "source" in cols:
        lead = ["source"] + (["source_path"] if "source_path" in cols else [])
        rest = [c for c in cols if c not in lead]
        merged = merged[lead + rest]

    merged.to_csv(out_csv, index=False, encoding=encoding)
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
    dropna : bool, default=True
        Whether to drop rows with NA in any of the group-by keys before grouping.

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
    dropna: bool = True


def _filter_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: Sequence[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
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
    return num_df.select_dtypes(include=[np.number])


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
        print(f"Aggregated feature output file already exists; returning existing file: {out_csv}")
        return out_csv

    files = list(_iter_csv_files(root, pattern=pattern, recursive=recursive))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {root}")

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
            print(f"[aggregate] WARNING: failed to read {fp}: {e}")

    if not frames:
        raise RuntimeError("No CSVs could be read successfully.")

    df = pd.concat(frames, axis=0, ignore_index=True)

    # Filter columns (remove known non-feature columns, optional regex filters)
    df_f = _filter_columns(
        df,
        exclude_cols=tuple(plan.exclude_cols) + ("source_path",),
        include_regex=plan.include_regex,
        exclude_regex=plan.exclude_regex,
    )

    # Build group keys
    group_keys = list(plan.group_by)
    if plan.per_file:
        if "source" not in df_f.columns:
            raise ValueError("source column is missing; cannot group per_file.")
        group_keys = ["source"] + group_keys

    if plan.dropna:
        df_f = df_f.dropna(subset=[k for k in group_keys if k in df_f.columns], how="any")

    missing = [k for k in group_keys if k not in df_f.columns]
    if missing:
        raise ValueError(f"Missing group-by columns in data: {missing}")

    # Candidate numeric features
    feature_cols = [c for c in df_f.columns if c not in set(group_keys)]
    numeric_df = _numeric_subframe(df_f[feature_cols])
    if numeric_df.empty:
        raise ValueError("No numeric columns available for aggregation after filtering.")

    # Reattach group keys for grouping
    gdf = pd.concat([df_f[group_keys].reset_index(drop=True),
                     numeric_df.reset_index(drop=True)], axis=1)

    agg_ops = {c: list(plan.stats) for c in numeric_df.columns}
    grouped = gdf.groupby(group_keys, dropna=False).agg(agg_ops)

    # Flatten MultiIndex columns and order 'source' first
    grouped.columns = [f"{c}__{stat}" for (c, stat) in grouped.columns]
    grouped = grouped.reset_index()

    # Ensure 'source' (and 'source_path' if present) lead the output
    cols = list(grouped.columns)
    lead = [c for c in ("source", "source_path") if c in cols]
    rest = [c for c in cols if c not in lead]
    grouped = grouped[lead + rest]

    grouped.to_csv(out_csv, index=False, encoding=encoding)
    return out_csv


# ---------------------------
# Minimal CLI
# ---------------------------

def _build_parser():
    """
    Create an ``argparse.ArgumentParser`` for the CLI.

    The parser defines three subcommands:

    - ``gather``: Concatenate CSVs with origin metadata.
    - ``aggregate``: Aggregate numeric columns by group keys.
    - ``run``: Single entry point; toggles aggregation via ``--aggregate``.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with subcommands and options.
    """

    import argparse
    p = argparse.ArgumentParser(
        description="Gather and optionally aggregate feature CSVs across a single folder."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # existing: gather
    g = sub.add_parser("gather", help="Concatenate CSVs with source metadata.")
    g.add_argument("--root_dir", required=True, help="Root folder (or a single CSV file)")
    g.add_argument("--pattern", default="*.csv")
    g.add_argument("--no-recursive", action="store_true")
    g.add_argument("--delimiter", default=",")
    g.add_argument("--encoding", default="utf-8-sig")
    g.add_argument("--add-source-path", action="store_true")
    g.add_argument("--out-csv", default=None)
    g.add_argument("--overwrite-existing", action="store_true", default=False)

    # existing: aggregate
    a = sub.add_parser("aggregate", help="Aggregate numeric columns by keys.")
    a.add_argument("--root_dir", required=True)
    a.add_argument("--pattern", default="*.csv")
    a.add_argument("--no-recursive", action="store_true")
    a.add_argument("--delimiter", default=",")
    a.add_argument("--encoding", default="utf-8-sig")
    a.add_argument("--add-source-path", action="store_true")
    a.add_argument("--group-by", nargs="+", required=True, help="e.g., speaker")
    a.add_argument("--per-file", action="store_true", help="Group per file via source_stem")
    a.add_argument("--stats", nargs="+", default=["mean", "std"])
    a.add_argument("--exclude-cols", nargs="*", default=[], help="Columns to drop before aggregation")
    a.add_argument("--include-regex", default=None)
    a.add_argument("--exclude-regex", default=None)
    a.add_argument("--out-csv", default=None)
    a.add_argument("--overwrite-existing", action="store_true", default=False)

    # new: run (single entry; toggle aggregation with a flag)
    r = sub.add_parser("run", help="Single entry: concat or aggregate depending on --aggregate.")
    r.add_argument("--root_dir", required=True)
    r.add_argument("--pattern", default="*.csv")
    r.add_argument("--no-recursive", action="store_true")
    r.add_argument("--delimiter", default=",")
    r.add_argument("--encoding", default="utf-8-sig")
    r.add_argument("--add-source-path", action="store_true")
    r.add_argument("--aggregate", action="store_true", help="Enable aggregation mode")
    r.add_argument("--group-by", nargs="+", help="Keys for aggregation (required if --aggregate)")
    r.add_argument("--per-file", action="store_true", help="Group per file via source_stem")
    r.add_argument("--stats", nargs="+", default=["mean", "std"])
    r.add_argument("--exclude-cols", nargs="*", default=[], help="Columns to drop before aggregation")
    r.add_argument("--include-regex", default=None)
    r.add_argument("--exclude-regex", default=None)
    r.add_argument("--out-csv", default=None)
    r.add_argument("--overwrite-existing", action="store_true", default=False)

    return p


def main():
    """
    Entry point for the command-line interface.

    Parses arguments, dispatches to :func:`gather_csvs_to_one`,
    :func:`aggregate_features`, or :func:`feature_gather` depending on the
    selected subcommand, and prints the resulting output path.

    Notes
    -----
    This function is invoked when the module is executed as a script::

        python -m taters.helpers.feature_gather <subcommand> [options]
    """

    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "gather":
        out = gather_csvs_to_one(
            root_dir=args.root_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            delimiter=args.delimiter,
            encoding=args.encoding,
            add_source_path=args.add_source_path,
            out_csv=args.out_csv,
            overwrite_existing=args.overwrite_existing,
        )
        print(str(out))
        return

    if args.cmd == "aggregate":
        plan = AggregationPlan(
            group_by=args.group_by,
            per_file=args.per_file,
            stats=tuple(args.stats),
            exclude_cols=tuple(args.exclude_cols or []),
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
        )
        out = aggregate_features(
            root_dir=args.root_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            delimiter=args.delimiter,
            encoding=args.encoding,
            add_source_path=args.add_source_path,
            plan=plan,
            out_csv=args.out_csv,
            overwrite_existing=args.overwrite_existing,
        )
        print(str(out))
        return

    if args.cmd == "run":
        out = feature_gather(
            root_dir=args.root_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            delimiter=args.delimiter,
            encoding=args.encoding,
            add_source_path=args.add_source_path,
            aggregate=args.aggregate,
            group_by=args.group_by,          # may be None if aggregate=False
            per_file=args.per_file,
            stats=tuple(args.stats),
            exclude_cols=tuple(args.exclude_cols or []),
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            out_csv=args.out_csv,
            overwrite_existing=args.overwrite_existing,
        )
        print(str(out))
        return



if __name__ == "__main__":
    main()
