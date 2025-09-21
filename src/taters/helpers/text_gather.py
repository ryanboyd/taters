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
    include_group_count: bool = False,  # ← NEW
) -> Tuple[csv.writer, io.TextIOBase, List[str]]:
    cols = ["text_id", "text"]
    if include_group_count:
        cols.append("group_count")      # ← placed right after 'text'
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
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default
    text_cols: Sequence[str],
    id_cols: Sequence[str] | None = None,
    mode: str = "concat",                 # "concat" or "separate"
    group_by: Sequence[str] | None = None,
    delimiter: str | None = None,
    encoding: str = DEFAULT_ENCODING,
    joiner: str = DEFAULT_JOINER,
    # external grouping params
    num_buckets: int = 1024,               # tune up if many groups / very large file
    max_open_bucket_files: int = 64,      # file descriptor cap (LRU)
    tmp_root: PathLike | None = None,     # where to place partitions (default: system tmp)
) -> Path:
    """
    Convert an arbitrary (possibly huge, unsorted) CSV into an "analysis-ready" CSV with a stable schema:
        text_id,text[,source_col]

    - No grouping: single streaming pass (constant memory).
    - With grouping: two-pass external aggregation that does NOT require presorting.

    Args:
        csv_path: input CSV
        out_csv: output CSV to write (will be created/overwritten)
        text_cols: columns containing the text
        id_cols: optional columns to compose text_id (if grouping is None). If omitted, uses row index.
        mode:
            "concat"   -> join all text_cols per row (or per group) into one text
            "separate" -> emit one row per (row, col) (or per (group, col)) with source_col metadata
        group_by: optional list of columns to aggregate by (unsorted OK).
        delimiter, encoding, joiner: parsing options
        num_buckets: number of hash partitions for external grouping
        max_open_bucket_files: limit for open file handles during partitioning
        tmp_root: optional temp directory root

    Returns:
        Path to the written CSV.
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
        writer, fh, _ = _open_out_csv(out_path, include_source_col, include_source_path)
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
                        writer.writerow([text_id, joiner.join(parts)])
                    else:
                        for col in text_cols:
                            val = row.get(col, "")
                            if not val:
                                continue
                            writer.writerow([text_id, val, col])
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
                    writer.writerow([text_id, joiner.join(pieces), len(pieces)])  # <- add count
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
                        writer.writerow([text_id, joiner.join(vals), len(vals), col])  # <- count before source_col

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
    Stream a folder of .txt files into an analysis-ready CSV: text_id,text[,source_path]
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
