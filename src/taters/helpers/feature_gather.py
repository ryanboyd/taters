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

    # Allow very large cells (long transcripts)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    df = pd.read_csv(path, dtype="object", sep=delimiter, encoding=encoding)

    def _next_available(base: str, existing: set[str]) -> str:
        """Return base.N with the smallest N>=1 not in existing."""
        n = 1
        candidate = f"{base}.{n}"
        while candidate in existing:
            n += 1
            candidate = f"{base}.{n}"
        return candidate

    # ---- De-duplicate any pre-existing columns named exactly 'source'
    cols = list(df.columns)
    seen = set(cols)

    # Rename exact 'source' occurrences
    if "source" in seen:
        new_cols = []
        # We'll track taken names as we rename
        taken = set(cols)
        for name in cols:
            if name == "source":
                new_name = _next_available("source", taken)
                new_cols.append(new_name)
                taken.add(new_name)
                taken.discard("source")  # doesn't matter, just for clarity
            else:
                new_cols.append(name)
        df.columns = new_cols
        cols = new_cols
        seen = set(cols)

    # Now we can safely insert our pipeline 'source'
    df.insert(0, "source", path.stem)

    # ---- Handle source_path similarly if requested
    if add_source_path:
        cols = list(df.columns)
        seen = set(cols)
        if "source_path" in seen:
            new_cols = []
            taken = set(cols)
            for name in cols:
                if name == "source_path":
                    new_name = _next_available("source_path", taken)
                    new_cols.append(new_name)
                    taken.add(new_name)
                    taken.discard("source_path")
                else:
                    new_cols.append(name)
            df.columns = new_cols

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
        # Find numbered variants like base.1, base.2, ...
        numbered = [c for c in cols if c == f"{base}.1" or c.startswith(f"{base}.")]
        has_base = base in existing
        if numbered:
            numbered.sort()
            inner_col = numbered[0]              # choose the first stable candidate
            if has_base:
                # Demote file-level base to 'file_<base>' (unique if needed)
                demoted = _unique(f"file_{base}", existing)
                df.rename(columns={base: demoted}, inplace=True)
                existing.discard(base)
                existing.add(demoted)
            # Promote inner to base
            df.rename(columns={inner_col: base}, inplace=True)
            existing.discard(inner_col)
            existing.add(base)
            # refresh for next iteration
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
    # Always preserve group keys (and drop duplicates while preserving order)
    keep = list(dict.fromkeys(list(must_keep) + keep))
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

    # If we are aggregating across files (per_file=False),
    # promote inner keys (e.g., 'source.1' -> 'source') and demote file-level keys.
    if not plan.per_file:
        df = _promote_inner_keys(df, plan.group_by)


    def _resolve_keys(base_keys, columns):
        cols = set(columns)
        resolved = []
        for k in base_keys:
            if k in cols:
                resolved.append(k)
                continue
            # Look for numbered variants like 'k.1', 'k.2', ...
            prefix = f"{k}."
            candidates = [c for c in columns if c == f"{k}.1" or c.startswith(prefix)]
            if candidates:
                # pick the first stable candidate
                resolved.append(sorted(candidates)[0])
            else:
                # leave unresolved; we'll error below with a helpful message
                resolved.append(k)
        return resolved

    # Build (base) group keys from the plan
    group_keys = list(plan.group_by)
    if plan.per_file:
        if "source" not in df.columns:
            raise ValueError("source column is missing; cannot group per_file.")
        group_keys = ["source"] + group_keys

    # Resolve collisions against actual columns
    group_keys = _resolve_keys(group_keys, df.columns)

    # Now filter but ALWAYS keep group keys
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
