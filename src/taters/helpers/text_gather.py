from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Optional

PathLike = Union[str, Path]

# Allow very large CSV fields (handles huge text columns safely).
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # macOS / some platforms can’t take sys.maxsize; fall back to 2^31-1
    csv.field_size_limit(2**31 - 1)


# -----------------------------
# Config / constants
# -----------------------------

DEFAULT_ENCODING = "utf-8-sig"  # plays nicely with Excel-exported CSVs
DEFAULT_JOINER = " "
DEFAULT_DELIM = ","


# -----------------------------
# Utilities
# -----------------------------

def _ensure_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)

def _detect_delimiter(sample: bytes, default: str = DEFAULT_DELIM) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"))
        return dialect.delimiter
    except Exception:
        return default

def _compose_id(values: Sequence[object], sep: str = " | ") -> str:
    vals = [str(v) for v in values if v is not None and str(v) != ""]
    return sep.join(vals) if vals else ""

def _bucket_of_key(key_tuple: Tuple[str, ...], num_buckets: int) -> int:
    # Stable across processes and platforms (unlike Python's salted hash()).
    h = hashlib.blake2b("|".join(key_tuple).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % max(1, num_buckets)

# -----------------------------------------
# Other Helpers, primarily around filenames
# -----------------------------------------

def _sanitize_for_filename(s: str) -> str:
    s2 = re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_")
    return s2 or "x"

def _default_csv_out_path(in_csv: Path, mode: str, text_cols: Sequence[str], group_by: Optional[Sequence[str]]) -> Path:
    stem = in_csv.stem
    if group_by:
        suffix = "grouped_" + "_".join(_sanitize_for_filename(g) for g in group_by)
    else:
        suffix = _sanitize_for_filename(mode) + "_" + "_".join(_sanitize_for_filename(c) for c in text_cols)
    return in_csv.parent / f"{stem}_{suffix}.csv"

def _default_txt_out_path(root_dir: Path, *, id_from: str, recursive: bool, pattern: str) -> Path:
    stem = root_dir.name
    parts = ["txt"]
    if id_from != "stem":
        parts.append(f"id{id_from}")
    if recursive:
        parts.append("recursive")
    if pattern and pattern != "*.txt":
        parts.append(_sanitize_for_filename(pattern))
    return root_dir / f"{stem}_{'_'.join(parts)}.csv"


@dataclass(frozen=True)
class _LRUHandle:
    path: Path
    fh: io.TextIOBase
    writer: csv.writer


class _LRUFileCache:
    """
    Keeps a bounded number of CSV writers open at once to avoid 'too many open files'.
    """
    def __init__(
        self,
        max_open: int,
        newline: str = "",
        encoding: str = DEFAULT_ENCODING,
        delimiter: str = DEFAULT_DELIM,          # ← NEW
    ):
        self.max_open = max_open
        self.newline = newline
        self.encoding = encoding
        self.delimiter = delimiter               # ← NEW
        self._cache: OrderedDict[int, _LRUHandle] = OrderedDict()

    def get(self, bucket_idx: int, path: Path, header: List[str]) -> csv.writer:
        # Return existing writer if cached
        if bucket_idx in self._cache:
            h = self._cache.pop(bucket_idx)
            self._cache[bucket_idx] = h
            return h.writer

        # Evict one if at capacity
        if len(self._cache) >= self.max_open:
            _, old = self._cache.popitem(last=False)
            try:
                old.fh.close()
            except Exception:
                pass

        # OPEN IN APPEND MODE so we never truncate prior rows
        first_write = not path.exists() or path.stat().st_size == 0
        fh = path.open("a", newline=self.newline, encoding=self.encoding)  # ← "a", not "w"
        w = csv.writer(fh, delimiter=self.delimiter)                        # ← use same delimiter
        if first_write:
            w.writerow(header)
        self._cache[bucket_idx] = _LRUHandle(path=path, fh=fh, writer=w)
        return w

    def close_all(self) -> None:
        while self._cache:
            _, h = self._cache.popitem(last=True)
            try:
                h.fh.close()
            except Exception:
                pass


# -----------------------------
# Core writer: analysis-ready CSV
#   Header is: text_id,text,(source_col?),(source_path?)
# -----------------------------

def _open_out_csv(
    path: Path,
    include_source_col: bool,
    include_source_path: bool,
    include_group_count: bool = False,
    id_col_names: Optional[List[str]] = None,      # NEW
    group_by_names: Optional[List[str]] = None,    # NEW
) -> Tuple[csv.writer, io.TextIOBase, List[str]]:
    """
    Open an output CSV and write the header.

    Column order:
      - Non-grouped: text_id, <id_cols...>, text[, source_col][, source_path]
      - Grouped:     text_id, <group_by...>, text, group_count[, source_col][, source_path]
    """
    id_col_names = list(id_col_names or [])
    group_by_names = list(group_by_names or [])

    cols = ["text_id"]
    if group_by_names:
        cols += group_by_names
    elif id_col_names:
        cols += id_col_names

    cols.append("text")
    if include_group_count:
        cols.append("group_count")
    if include_source_col:
        cols.append("source_col")
    if include_source_path:
        cols.append("source_path")

    fh = path.open("w", newline="", encoding="utf-8-sig")
    w = csv.writer(fh)
    w.writerow(cols)
    return w, fh, cols





# -----------------------------
# Public API
# -----------------------------

def csv_to_analysis_ready_csv(
    *,
    csv_path: PathLike,
    out_csv: PathLike | None = None,
    overwrite_existing: bool = False,
    text_cols: Sequence[str],
    id_cols: Sequence[str] | None = None,
    mode: str = "concat",
    group_by: Sequence[str] | None = None,
    delimiter: str | None = None,
    encoding: str = DEFAULT_ENCODING,
    joiner: str = DEFAULT_JOINER,
    num_buckets: int = 1024,
    max_open_bucket_files: int = 64,
    tmp_root: PathLike | None = None,
    include_id_cols: bool = True,             # NEW (default on)
) -> Path:
    """
    Stream a (possibly huge) CSV into a compact **analysis-ready** CSV with a
    stable schema and optional external grouping.

    Output schema
    -------------
    Always writes a header and enforces a consistent column order:

    • No grouping:
        `text_id,text`                            (plus `source_col` if `mode="separate"`)
    • With grouping:
        `text_id,text,group_count`                (plus `source_col` if `mode="separate"`)

    Where:
      - `text_id` is either the composed ID from `id_cols` or `row_<n>` when
        `id_cols=None`.
      - `mode="concat"` joins all `text_cols` using `joiner` per row or group.
      - `mode="separate"` emits one row per (`row_or_group`, `text_col`) and
        fills `source_col` with the contributing column name.

    Grouping at scale
    -----------------
    If `group_by` is provided, the function performs a **two-pass external
    grouping** that does not require presorting:
      1) Hash-partition rows to on-disk “bucket” CSVs (bounded writers with LRU).
      2) Aggregate each bucket into final rows (concat or separate mode), writing
         `group_count` to record how many pieces contributed. :contentReference[oaicite:1]{index=1}

    Parameters
    ----------
    csv_path
        Source CSV with at least the columns in `text_cols` (and `group_by` if
        grouping).
    out_csv
        Destination CSV. If `None`, a name is derived from the input and options
        (e.g., `<stem>_grouped_<group_by>.csv` or `<stem>_concat_<cols>.csv`).
    overwrite_existing
        If `False` (default) and `out_csv` exists, the function returns early.
    text_cols
        One or more text fields to concatenate or emit separately.
    id_cols
        Optional columns to compose `text_id` when not grouping. When omitted, a
        synthetic `row_<n>` is used.
    mode
        `"concat"` (default) or `"separate"`. See schema above.
    group_by
        Optional list of columns to aggregate by; works on unsorted CSVs.
    delimiter, encoding, joiner
        Parsing/formatting options. If `delimiter=None`, sniffs from a sample.
    num_buckets, max_open_bucket_files, tmp_root
        External grouping controls (partition count, LRU limit, temp root).
    include_id_cols
        When aggregating/concatenating, retains the identifiers in the output file.

    Returns
    -------
    Path
        Path to the analysis-ready CSV.

    Raises
    ------
    ValueError
        If required columns are missing or `mode` is invalid.

    Examples
    --------
    Concatenate two text fields per row:

    >>> csv_to_analysis_ready_csv(
    ...     csv_path="transcripts.csv",
    ...     text_cols=["prompt","response"],
    ...     id_cols=["speaker"],
    ... )

    Group by speaker and join rows:

    >>> csv_to_analysis_ready_csv(
    ...     csv_path="transcripts.csv",
    ...     text_cols=["text"],
    ...     group_by=["speaker"],
    ... )
    """
    in_path = _ensure_path(csv_path)

    # Detect delimiter if not provided
    if delimiter is None:
        with in_path.open("rb") as fb:
            sample = fb.read(8192)
        delimiter = _detect_delimiter(sample, default=DEFAULT_DELIM)

    text_cols = list(text_cols)
    if not text_cols:
        raise ValueError("text_cols must be non-empty")
    mode = mode.strip().lower()
    if mode not in ("concat", "separate"):
        raise ValueError("mode must be 'concat' or 'separate'")

    include_source_col = (mode == "separate")
    include_source_path = False  # this function deals with CSV; folder variant uses this flag

    # Decide output path (default next to input if not specified)
    out_path = _ensure_path(out_csv) if out_csv is not None else _default_csv_out_path(
        in_csv=in_path, mode=mode, text_cols=text_cols, group_by=group_by)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_path).is_file():
        print("File with gathered text already exists; returning existing file.")
        return out_path

    # If no grouping, we can stream straight to the output
    if not group_by:
        writer, fh, _ = _open_out_csv(
            out_path,
            include_source_col,
            include_source_path,
            include_group_count=False,
            id_col_names=(list(id_cols) if include_id_cols and id_cols else None),
            group_by_names=None,
        )
        try:
            with in_path.open("r", newline="", encoding=encoding) as f:
                rdr = csv.DictReader(f, delimiter=delimiter)
                headers = rdr.fieldnames or []
                missing = [c for c in (id_cols or []) + text_cols if c not in headers]
                if missing:
                    raise ValueError(f"Missing columns: {missing}. Make sure that you try specifying a delimiter manually if you see this error message.")

                for idx, row in enumerate(rdr, start=1):
                    text_id = _compose_id([row.get(c, "") for c in (id_cols or [])]) if id_cols else f"row_{idx}"
                    if mode == "concat":
                        parts = [row.get(c, "") for c in text_cols if row.get(c, "")]
                        if not parts:
                            continue
                        row_prefix = [text_id]
                        if include_id_cols and id_cols:
                            row_prefix += [row.get(c, "") for c in id_cols]
                        writer.writerow(row_prefix + [joiner.join(parts)])
                    else:
                        for col in text_cols:
                            val = row.get(col, "")
                            if not val:
                                continue
                            row_prefix = [text_id]
                            if include_id_cols and id_cols:
                                row_prefix += [row.get(c, "") for c in id_cols]
                            writer.writerow(row_prefix + [val, col])

        finally:
            fh.close()
        return out_path

    # Otherwise, do external grouping (two-pass)
    group_by = list(group_by)

    # Phase 1: partition into hash buckets
    tmp_base = Path(tempfile.mkdtemp(prefix="gather_partitions_", dir=str(tmp_root) if tmp_root else None))
    part_dir = tmp_base / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    # Bucket writer cache
    header_small = group_by + text_cols
    cache = _LRUFileCache(
        max_open=max_open_bucket_files,
        newline="",
        encoding=encoding,
        delimiter=delimiter,             # ← NEW
    )


    try:
        with in_path.open("r", newline="", encoding=encoding) as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            headers = rdr.fieldnames or []
            missing = [c for c in group_by + text_cols if c not in headers]
            if missing:
                raise ValueError(f"Missing columns: {missing}. Make sure that you try specifying a delimiter manually if you see this error message.")

            for row in rdr:
                key_tuple = tuple(row[g] for g in group_by)
                bucket = _bucket_of_key(key_tuple, num_buckets)
                bpath = part_dir / f"bucket_{bucket:05d}.csv"
                w = cache.get(bucket, bpath, header_small)
                # write only needed fields to keep partitions lean
                w.writerow([row.get(c, "") for c in header_small])
    finally:
        cache.close_all()

    # Phase 2: per-bucket aggregation → final writer
    writer, out_fh, _ = _open_out_csv(
        out_path,
        include_source_col,
        include_source_path,
        include_group_count=True,
        id_col_names=None,
        group_by_names=group_by,   # NEW
    )



    try:
        for bfile in sorted(part_dir.glob("bucket_*.csv")):
            # Aggregate this bucket in memory
            if mode == "concat":
                # key -> list[text]
                agg: Dict[Tuple[str, ...], List[str]] = {}
                with bfile.open("r", newline="", encoding=encoding) as bf:
                    br = csv.DictReader(bf, delimiter=delimiter)
                    for row in br:
                        key = tuple(row[g] for g in group_by)
                        parts = [row.get(c, "") for c in text_cols if row.get(c, "")]
                        if not parts:
                            continue
                        agg.setdefault(key, []).append(joiner.join(parts))
                # Emit
                for key, pieces in agg.items():
                    text_id = _compose_id(key) or "group"
                    writer.writerow([text_id, *key, joiner.join(pieces), len(pieces)])
            else:
                # key -> col -> list[text]
                agg: Dict[Tuple[str, ...], Dict[str, List[str]]] = {}
                with bfile.open("r", newline="", encoding=encoding) as bf:
                    br = csv.DictReader(bf, delimiter=delimiter)
                    for row in br:
                        key = tuple(row[g] for g in group_by)
                        box = agg.setdefault(key, {})
                        for col in text_cols:
                            val = row.get(col, "")
                            if val:
                                box.setdefault(col, []).append(val)
                # Emit
                for key, per_col in agg.items():
                    text_id = _compose_id(key) or "group"
                    for col in text_cols:
                        vals = per_col.get(col, [])
                        if not vals:
                            continue
                        writer.writerow([text_id, *key, joiner.join(vals), len(vals), col])

    finally:
        out_fh.close()
        # Clean up partitions
        try:
            for p in part_dir.glob("bucket_*.csv"):
                p.unlink(missing_ok=True)
            part_dir.rmdir()
            tmp_base.rmdir()
        except Exception:
            pass

    return out_path


def txt_folder_to_analysis_ready_csv(
    *,
    root_dir: PathLike,
    out_csv: PathLike | None = None,
    recursive: bool = False,
    pattern: str = "*.txt",
    encoding: str = "utf-8",
    id_from: str = "stem",            # "stem" | "name" | "path"
    include_source_path: bool = True, # writes 'source_path' column
    overwrite_existing: bool = False  # if the file already exists, let's not overwrite by default

) -> Path:
    """
    Stream a folder of `.txt` files into an analysis-ready CSV with predictable,
    reproducible IDs.

    For each file matching `pattern`, the emitted row contains:
      - `text_id`: the basename (stem), full filename, or relative path (see
        `id_from`), and
      - `text`: the file contents.
      - `source_path`: optional column with path relative to `root_dir`.

    Parameters
    ----------
    root_dir
        Folder containing `.txt` files.
    out_csv
        Destination CSV. If `None`, a descriptive default is created next to
        `root_dir` (e.g., `<folder>_txt_recursive_*.csv`).
    recursive
        Recurse into subfolders. Default: `False`.
    pattern
        Glob for matching text files. Default: `"*.txt"`.
    encoding
        File decoding. Default: `"utf-8"`.
    id_from
        How to derive `text_id`: `"stem"` (basename without extension),
        `"name"` (filename), or `"path"` (relative path).
    include_source_path
        If `True` (default), add a `source_path` column showing the relative path.
    overwrite_existing
        If `False` (default) and `out_csv` exists, returns the existing file.

    Returns
    -------
    Path
        Path to the analysis-ready CSV.

    Examples
    --------
    >>> txt_folder_to_analysis_ready_csv(root_dir="notes", recursive=True, id_from="path")
    """
    root = _ensure_path(root_dir)
    out_path = _ensure_path(out_csv) if out_csv is not None else _default_txt_out_path(
        root, id_from=id_from, recursive=recursive, pattern=pattern)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_path).is_file():
        print("File with gathered text already exists; returning existing file.")
        return out_path

    writer, fh, _ = _open_out_csv(out_path, include_source_col=False, include_source_path=include_source_path)
    try:
        files = root.rglob(pattern) if recursive else root.glob(pattern)
        for p in files:
            if not p.is_file():
                continue
            if id_from == "stem":
                text_id = p.stem
            elif id_from == "name":
                text_id = p.name
            elif id_from == "path":
                text_id = str(p.relative_to(root))
            else:
                raise ValueError("id_from must be 'stem', 'name', or 'path'")
            text = p.read_text(encoding=encoding, errors="ignore")
            if include_source_path:
                writer.writerow([text_id, text, str(p.relative_to(root))])
            else:
                writer.writerow([text_id, text])
    finally:
        fh.close()

    return out_path


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    def _flatten_str_list(values):
        # Accept multiple flags like --text-col A --text-col B
        # and return ["A","B"]. If user passed none, return [].
        return [v for v in (values or []) if v is not None and str(v) != ""]

    parser = argparse.ArgumentParser(
        prog="text_gather",
        description="Stream input (CSV or folder of .txt files) into an analysis-ready CSV: text_id,text[,source_col|source_path].",
    )

    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument("--csv", type=Path, help="Path to input CSV file.")
    io_group.add_argument("--txt-dir", type=Path, help="Path to a folder containing .txt files.")

    parser.add_argument("--out", type=Path, required=False,
                        help="Output CSV path. If omitted, a default is created next to the input (e.g., "
                             "<inputstem>_grouped_<cols>.csv or <inputstem>_<mode>_<textcols>.csv for CSV mode; "
                             "<foldername>_txt[...].csv for TXT mode).")

    # Common parsing options
    parser.add_argument("--encoding", default="utf-8-sig", help="Input text/CSV encoding. Default: utf-8-sig")

    parser.add_argument("--overwrite_existing", type=bool, default=False,
                        help="Do you want to overwrite the output file if it already exists?")

    # CSV mode options
    parser.add_argument("--text-col", action="append", dest="text_cols",
                        help="Text column to use (repeat for multiple). REQUIRED for --csv.")
    parser.add_argument("--id-col", action="append", dest="id_cols",
                        help="ID column(s) to compose text_id (optional). Repeatable.")
    parser.add_argument("--mode", choices=["concat", "separate"], default="concat",
                        help="For CSV: concat all text cols per item, or emit separate rows per col. Default: concat")
    parser.add_argument("--group-by", action="append", dest="group_by",
                        help="Group by these column(s) (unsorted CSV ok; uses external partitioning). Repeatable.")
    parser.add_argument("--delimiter", default=None,
                        help="CSV delimiter (auto-detected if omitted).")
    parser.add_argument("--joiner", default=" ",
                        help="String used to join multiple text columns/rows. Default: single space")

    # External grouping tuning
    parser.add_argument("--num-buckets", type=int, default=1024,
                        help="Number of hash partitions for external grouping. Default: 1024")
    parser.add_argument("--max-open-bucket-files", type=int, default=64,
                        help="Max open files during partitioning (LRU). Default: 64")
    parser.add_argument("--tmp-root", type=Path, default=None,
                        help="Directory to store temporary partition files (defaults to system temp).")

    # TXT folder options
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories when using --txt-dir.")
    parser.add_argument("--pattern", default="*.txt",
                        help="Glob pattern for --txt-dir. Default: *.txt")
    parser.add_argument("--id-from", choices=["stem", "name", "path"], default="stem",
                        help="How to form text_id for .txt files. Default: stem")
    parser.add_argument("--no-source-path", action="store_true",
                        help="Do NOT include source_path column for .txt folder mode.")
    

    args = parser.parse_args()

    try:
        if args.csv:
            # Validate CSV-specific requirements
            text_cols = _flatten_str_list(args.text_cols)
            if not text_cols:
                parser.error("--text-col is required when using --csv (repeat flag for multiple columns).")

            id_cols = _flatten_str_list(args.id_cols) or None
            group_by = _flatten_str_list(args.group_by) or None

            out_path = csv_to_analysis_ready_csv(
                csv_path=args.csv,
                out_csv=args.out,
                overwrite_existing=args.overwrite_existing,
                text_cols=text_cols,
                id_cols=id_cols,
                mode=args.mode,
                group_by=group_by,
                delimiter=args.delimiter,
                encoding=args.encoding,
                joiner=args.joiner,
                num_buckets=args.num_buckets,
                max_open_bucket_files=args.max_open_bucket_files,
                tmp_root=args.tmp_root,
            )
            print(f"Wrote analysis-ready CSV: {out_path}")

        else:
            # TXT folder mode
            out_path = txt_folder_to_analysis_ready_csv(
                root_dir=args.txt_dir,
                out_csv=args.out,
                overwrite_existing=args.overwrite_existing,
                recursive=args.recursive,
                pattern=args.pattern,
                encoding=args.encoding,
                id_from=args.id_from,
                include_source_path=not args.no_source_path,
            )
            print(f"Wrote analysis-ready CSV: {out_path}")

    except Exception as e:
        print(f"[text_gather] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
