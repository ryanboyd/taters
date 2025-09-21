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
    """Yield CSV files under one folder (optionally recursive)."""
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
    source_col_stem: str = "source_stem",
    source_col_path: str = "source_path",
) -> pd.DataFrame:
    """Read a CSV and add columns identifying its origin."""
    # Robust CSV field size (giant text cells)
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    df = pd.read_csv(path, dtype="object", sep=delimiter, encoding=encoding)
    df[source_col_stem] = path.stem
    if add_source_path:
        df[source_col_path] = str(path.resolve())
    return df


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
    Concatenate many CSVs from one folder into a single CSV with 'source_stem'
    (and optional 'source_path'). Returns the output path.

    Default out path (if not provided):
        <root_dir_parent>/<root_dir_name>.csv
        e.g., /features/whisper-embeddings → /features/whisper-embeddings.csv
    """
    root = Path(root_dir)
    if out_csv is None:
        out_csv = root.parent / f"{root.name}.csv"

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite_existing:
        print("Aggregated feature output file already exists; returning existing file.")
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
    merged.to_csv(out_csv, index=False, encoding=encoding)
    return out_csv


# ---------------------------
# Aggregation
# ---------------------------

@dataclass
class AggregationPlan:
    """
    Plan for aggregating numeric columns.
    - group_by: keys to group on (e.g., ["speaker"])
    - per_file: if True, include 'source_stem' in grouping (stats per input file);
                if False, aggregate across files globally.
    - stats: stat names understood by pandas ('mean', 'std', 'median', etc.)
    - exclude_cols: columns to drop before aggregation (e.g., 'start_time', 'end_time', 'text')
    - include_regex / exclude_regex: optional regex filters for feature column names
    - dropna: whether to drop rows with NA in group keys
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
    """Apply include/exclude filters to columns before numeric selection."""
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
    """Coerce to numeric where possible; keep only numeric columns."""
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
    Discover → read → concat → aggregate numeric columns per plan.
    Default out path (if not provided):
        <root_dir_parent>/<root_dir_name>.csv
    """
    root = Path(root_dir)
    if out_csv is None:
        out_csv = root.parent / f"{root.name}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not overwrite_existing:
        print("Aggregated feature output file already exists; returning existing file.")
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
        if "source_stem" not in df_f.columns:
            raise ValueError("source_stem column is missing; cannot group per_file.")
        group_keys = ["source_stem"] + group_keys

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

    # Flatten MultiIndex columns: "<col>__<stat>"
    grouped.columns = [f"{c}__{stat}" for (c, stat) in grouped.columns]
    grouped = grouped.reset_index()

    grouped.to_csv(out_csv, index=False, encoding=encoding)
    return out_csv


# ---------------------------
# Minimal CLI
# ---------------------------

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Gather and optionally aggregate feature CSVs across a single folder."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # gather
    g = sub.add_parser("gather", help="Concatenate CSVs with source metadata.")
    g.add_argument("--root_dir", required=True, help="Root folder (or a single CSV file)")
    g.add_argument("--pattern", default="*.csv")
    g.add_argument("--no-recursive", action="store_true")
    g.add_argument("--delimiter", default=",")
    g.add_argument("--encoding", default="utf-8-sig")
    g.add_argument("--add-source-path", action="store_true")
    g.add_argument("--out-csv", default=None)
    g.add_argument("--overwrite-existing", action="store_true", default=False)

    # aggregate
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

    return p


def main():
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


if __name__ == "__main__":
    main()
