from __future__ import annotations

import csv
import warnings
import hashlib
import io
import re
import tempfile

from .doc_text import (DOCUMENT_PATTERN, DocumentReadError,
                       read_document_text)
from .parallel_map import ordered_parallel_map, pool_workers
from .progress import Ticker, announce, count_rows
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Union, Optional
from .provenance import TEXT_GRAIN, records_settings
from .row_filter import keeps_row
from .cliargs import CliSpec
from .csvio import widen_csv_field_limit
from .feature_columns import ColumnSpec

FEATURE_COLUMNS = ColumnSpec(
    label="Spreadsheet columns",
    dynamic="the user's own columns, carried through under their own names",
)

widen_csv_field_limit()

PathLike = Union[str, Path]



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
    # stable across processes and platforms (unlike Python's salted hash())
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
    if pattern and pattern not in ("*.txt", DOCUMENT_PATTERN):
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
        delimiter: str = DEFAULT_DELIM,
    ):
        self.max_open = max_open
        self.newline = newline
        self.encoding = encoding
        self.delimiter = delimiter
        self._cache: OrderedDict[int, _LRUHandle] = OrderedDict()

    def get(self, bucket_idx: int, path: Path, header: List[str]) -> csv.writer:
        # already have one open? hand it back (and bump it to most-recent)
        if bucket_idx in self._cache:
            h = self._cache.pop(bucket_idx)
            self._cache[bucket_idx] = h
            return h.writer

        # if we're full up, kick out the oldest one
        if len(self._cache) >= self.max_open:
            _, old = self._cache.popitem(last=False)
            try:
                old.fh.close()
            except Exception:
                pass

        # APPEND MODE ("a", not "w"), so we never truncate rows we already wrote;
        # and the same delimiter the reader used, or pass 2 can't parse it back
        first_write = not path.exists() or path.stat().st_size == 0
        fh = path.open("a", newline=self.newline, encoding=self.encoding)  # append!
        w = csv.writer(fh, delimiter=self.delimiter)  # same delimiter
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

def _emit_names(names: Optional[Sequence[str]]) -> List[str]:
    """
    The id/group columns to write *alongside* the composed ``text_id``.

    ``text_id`` is built from these columns, so a column literally called
    ``text_id`` is already in the output under that name -- writing it again
    produces a file with two columns of the same name and identical contents,
    which pandas reads back as ``text_id`` and ``text_id.1``.

    Used for both the header and the row values so the two cannot drift. A
    column dropped from one and not the other would shift every field in the
    file, which is far worse than a duplicate name.
    """
    return [n for n in (names or []) if n != "text_id"]


def _carried_names(carry: Optional[Sequence[str]], already: Sequence[str]) -> List[str]:
    """
    The carried columns that still need a place in the header.

    A column can be asked for twice without it being a mistake -- ``speaker``
    is a sensible thing to group by *and* a sensible thing to carry -- so a
    name already emitted as ``text_id`` or as an id/group column is dropped
    here rather than written twice under the same heading.
    """
    seen = {"text_id", "text", *already}
    out: List[str] = []
    for name in carry or []:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


_AGG_STATS = {"mean", "sum", "min", "max"}


class _Agg:
    """
    One summarized column of one group: the running sum, count, min and
    max, which is everything `_summaries` ever reported. Held instead of
    the rows, so a group the size of the corpus costs four numbers. The sum
    is compensated (Neumaier), so it equals what ``sum()`` gives on the
    same values in the same order.
    """

    __slots__ = ("n", "total", "comp", "lo", "hi")

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.comp = 0.0
        self.lo: Optional[float] = None
        self.hi: Optional[float] = None

    def see(self, raw: object) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        try:
            v = float(text)
        except ValueError:
            return
        self.n += 1
        t = self.total + v
        if abs(self.total) >= abs(v):
            self.comp += (self.total - t) + v
        else:
            self.comp += (v - t) + self.total
        self.total = t
        self.lo = v if self.lo is None or v < self.lo else self.lo
        self.hi = v if self.hi is None or v > self.hi else self.hi

    @property
    def sum(self) -> float:
        return self.total + self.comp

    def cells(self, stat: str) -> List[str]:
        if not self.n:
            return ["", "0"]
        value = {"mean": self.sum / self.n, "sum": self.sum,
                 "min": self.lo, "max": self.hi}[stat]
        return [format(value, ".12g"), str(self.n)]


_MIXED = object()


class _Group:
    """
    One group being combined: its text pieces, what its carried columns
    agree on (or that they don't), and its summaries. The pieces have to be
    held until the group is complete -- they are the output -- but nothing
    else about its rows is.
    """

    __slots__ = ("pieces", "per_col", "agreed", "aggs", "count")

    def __init__(self, carried: Sequence[str], agg: Mapping[str, str]):
        self.pieces: List[str] = []
        self.per_col: Dict[str, List[str]] = {}
        self.agreed: Dict[str, object] = {}
        self.aggs = {c: _Agg() for c in agg}
        self.count = 0

    def see_agg(self, row: Mapping[str, str]) -> None:
        for c, a in self.aggs.items():
            a.see(row.get(c, ""))

    def see_carried(self, carried: Sequence[str], row: Mapping[str, str]) -> None:
        for name in carried:
            v = row.get(name, "")
            had = self.agreed.get(name, v)
            self.agreed[name] = v if had == v else _MIXED

    def agreed_cells(self, carried: Sequence[str]) -> List[str]:
        return ["" if self.agreed.get(c, "") is _MIXED else str(self.agreed.get(c, ""))
                for c in carried]

    def stat_cells(self, agg: Mapping[str, str]) -> List[str]:
        out: List[str] = []
        for c, stat in agg.items():
            out.extend(self.aggs[c].cells(stat))
        return out


def _summaries(agg: Mapping[str, str], rows: List[Dict[str, str]]) -> List[str]:
    """
    The aggregate columns for one group: ``<col>_<stat>`` and ``<col>_n``
    for each requested column, in the order the mapping names them.

    A blank or non-numeric cell is left out of the statistic and out of
    ``n``, so a column of ``"n/a"`` gives an empty statistic and ``n = 0``
    rather than a crash. Built on :class:`_Agg`, so the file and database
    gathers summarize identically whichever holds the rows.
    """
    g = _Group((), agg)
    for row in rows:
        g.see_agg(row)
    return g.stat_cells(agg)


class _OutTable:
    """
    The gathered table being written, wherever it goes.

    ``writerow`` takes the list a ``csv.writer`` took; ``close(ok)`` finishes
    the table or throws the partial one away. A file is written under a
    scratch name and renamed when complete, a database table loaded and
    swapped in, both by :func:`taters.helpers.tables.write_table` -- so a
    half-written gather left by an interrupted run is never mistaken for a
    finished one, which matters because the gathered table is skip-if-exists
    like everything else here.
    """

    def __init__(self, ref, cols: List[str], hints: Dict[str, str]):
        from . import tables

        self.ref = ref
        self.cols = cols
        self._cm = tables.write_table(ref, cols, hints=hints)
        self._w = self._cm.__enter__()

    def writerow(self, row: Sequence[object]) -> None:
        self._w.writerow(list(row))

    def close(self, ok: bool) -> None:
        if ok:
            self._cm.__exit__(None, None, None)
        else:
            try:
                self._cm.__exit__(RuntimeError, RuntimeError("abandoned"), None)
            except Exception:
                pass


def _open_out_csv(
    path,
    include_source_col: bool,
    include_source_path: bool,
    include_group_count: bool = False,
    id_col_names: Optional[List[str]] = None,
    group_by_names: Optional[List[str]] = None,
    carry_col_names: Optional[List[str]] = None,
    agg_col_names: Optional[List[str]] = None,     # the "<col>_<stat>" summaries
    include_text: bool = True,                     # False: a no-text wrangle
) -> Tuple[_OutTable, _OutTable, List[str]]:
    """
    Open the output table and write the header.

    Column order:
      - Non-grouped: text_id, <id_cols...>, <carry_cols...>, text[, source_col][, source_path]
      - Grouped:     text_id, <group_by...>, <carry_cols...>, text, group_count[, <agg>...][, source_col][, source_path]

    Returns the writer twice (the second slot used to be the file handle;
    ``close(ok)`` on either is the one call that finishes or discards it).
    """
    id_col_names = list(id_col_names or [])
    group_by_names = list(group_by_names or [])

    cols = ["text_id"]
    if group_by_names:
        cols += _emit_names(group_by_names)
    elif id_col_names:
        cols += _emit_names(id_col_names)

    cols += _carried_names(carry_col_names, cols)
    if include_text:
        cols.append("text")
    if include_group_count:
        cols.append("group_count")
    cols += list(agg_col_names or [])
    if include_source_col:
        cols.append("source_col")
    if include_source_path:
        cols.append("source_path")

    # everything here is text except the counts and the summaries, and a
    # database that is told so skips a typing pass over the whole table
    hints = {c: "text" for c in cols}
    if include_group_count:
        hints["group_count"] = "int"
    for name in agg_col_names or []:
        hints[name] = "int" if name.endswith("_n") else "float"
    out = _OutTable(path, cols, hints)
    return out, out, cols


# -----------------------------
# Public API
# -----------------------------

@records_settings(
    # which rows and columns we read, and which metadata rode along.
    # `joiner` is NOT in here on purpose: joining two text columns with
    # " " rather than "\n" changes sentence segmentation, so it decides
    # every number every downstream analyzer produces.
    binding=("csv_path", "text_cols", "id_cols", "delimiter",
             "encoding", "include_id_cols", "carry_cols", "agg_cols"),
    # this takes seconds to redo, and it's the one table every join depends
    # on: an existing copy with no record gets rebuilt rather than trusted.
    redo_without_record=True,
    grain=TEXT_GRAIN,
    outputs=("out_csv",),
    # this step decides what text EXISTS. its settings describe the
    # dataset, so we record them but keep them out of the chain that
    # downstream steps compare -- a model has to be applicable to a corpus
    # assembled differently, which is the whole point of saving one.
    defines_text=True)
def csv_to_analysis_ready_csv(
    *,
    csv_path: PathLike,
    out_csv: PathLike | None = None,
    overwrite_existing: bool = False,
    text_cols: Sequence[str] | None = None,
    id_cols: Sequence[str] | None = None,
    mode: str = "concat",
    group_by: Sequence[str] | None = None,
    delimiter: str | None = None,
    encoding: str = DEFAULT_ENCODING,
    joiner: str = DEFAULT_JOINER,
    num_buckets: int = 1024,
    max_open_bucket_files: int = 64,
    tmp_root: PathLike | None = None,
    include_id_cols: bool = True,
    carry_cols: Sequence[str] | None = None,
    agg_cols: Mapping[str, str] | Sequence[str] | None = None,
    row_filters: Sequence[Sequence[object]] | None = None,
    verbose: bool = True,
    on_progress=None,   # on_progress(done, total, message) per row / bucket
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

    `carry_cols` inserts the named source columns between the identifiers and
    `text` in both shapes.

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
         `group_count` to record how many pieces contributed.

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
        Text fields to concatenate or emit separately. May be empty or None:
        the spreadsheet is then wrangled without any text -- rows are still
        combined, counted, summarized and carried -- and the output has no
        ``text`` column.
    id_cols
        Optional columns to compose `text_id` when not grouping. When omitted, a
        synthetic `row_<n>` is used.
    carry_cols
        Source columns to copy through to the output untouched. Distinct from
        `id_cols`, which *compose* `text_id` and so cannot be used to carry a
        column whose values repeat -- naming `speaker` there would give every
        utterance by one person the same `text_id`. Downstream analyzers offer a
        `pass_through_cols` argument that is meaningless unless the gather that
        feeds them preserved the columns, which is what this is for.

        When grouping, a carried column is written only where its value is the
        same for every row in the group; where rows disagree the cell is blank,
        because there is no honest single answer and picking the first row's is
        the kind of quiet wrong answer that survives into a published table.

        Names in `carry_cols` that the source does not have are dropped with a
        warning rather than raising: unlike `text_cols`, a carried column is a
        convenience, and half the presets that ask for `speaker` run over inputs
        that never had one.
    agg_cols
        Number columns to summarize per group, e.g. a per-post score becoming
        the average score of everything a user wrote. A sequence of names
        means "the mean of each"; a mapping picks the statistic per column
        from ``mean``, ``sum``, ``min`` and ``max``. Each summary lands in
        its own output column named ``<col>_<stat>``, placed after
        ``group_count``.

        Values that are blank or not parseable as numbers are skipped rather
        than treated as zero -- a missing score is missing, not 0 -- and a
        group with no numeric values at all gets a blank cell, for the same
        NA-is-not-zero reason a disagreeing carried column does.
    row_filters
        Which rows to keep, as ``[[column, operator, value], ...]`` over the
        source spreadsheet's own columns -- ``[["age", ">=", 18]]``,
        ``[["condition", "in", ["A", "B"]]]``. A row has to clear all of
        them. The operators and the comparison are
        :mod:`taters.helpers.row_filter`'s, the same ones the statistics use,
        so a filter means one thing wherever it is applied.

        Only the spreadsheet's own columns, because nothing has been measured
        yet: a word count is not available here and does not belong here
        either. What somebody knows at this point is what they collected --
        a screener somebody failed, a condition they are not analyzing, an
        age below the one they meant to study.

        It matters most when rows are being combined, since a row inside
        somebody else's joined text cannot be taken out again afterwards;
        that is why it is applied before the rows are bucketed rather than
        filtered later.

        Summaries run over **every** row of the group, including rows whose
        text cells are empty -- `group_count` counts only the rows whose
        text was joined, so the two can disagree. To keep that visible, each
        summary is followed by a ``<col>_n`` column giving the number of
        numeric values it actually used.

        Requires `group_by`: without groups there is nothing to summarize
        over, and silently ignoring the ask would hide a real mistake.
        Unlike `carry_cols`, a name the source lacks raises: a summary is an
        explicit computation, not a convenience.
    mode
        `"concat"` (default) or `"separate"`. See schema above.
    group_by
        Optional list of columns to aggregate by; works on unsorted CSVs.
    delimiter, encoding, joiner
        Parsing/formatting options. If `delimiter=None`, sniffs from a sample.
    num_buckets, max_open_bucket_files, tmp_root
        External grouping controls (partition count, LRU limit, temp root).
    include_id_cols
        When not grouping, write the id columns beside ``text_id``. Grouped
        output carries the grouping columns instead, whatever this says.
    on_progress
        Optional ``(done, total, message)`` callback. The gather names each
        phase as it runs -- counting the rows, then copying them (or, when
        grouping, sorting them into groups and then combining each group) --
        because a silent spinner over a large file reads as a hang.

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
    keep = [list(f) for f in (row_filters or [])]
    spec = _db_source(csv_path)
    if spec is not None:
        return _gather_from_db(
            spec, out_csv=out_csv, overwrite_existing=overwrite_existing,
            text_cols=list(text_cols or []), id_cols=list(id_cols or []),
            mode=mode, group_by=list(group_by or []), joiner=joiner,
            include_id_cols=include_id_cols, carry_cols=list(carry_cols or []),
            agg_cols=agg_cols, keep=keep, verbose=verbose, on_progress=on_progress)
    in_path = _ensure_path(csv_path)

    # sniff out the delimiter if we weren't given one
    if delimiter is None:
        with in_path.open("rb") as fb:
            sample = fb.read(8192)
        delimiter = _detect_delimiter(sample, default=DEFAULT_DELIM)

    # empty text_cols is something people actually ask for, not a mistake: a
    # spreadsheet without text can still be combined and summarized (one row
    # per person, with average scores). the output just has no `text` column.
    text_cols = list(text_cols or [])
    mode = mode.strip().lower()
    if mode not in ("concat", "separate"):
        raise ValueError("mode must be 'concat' or 'separate'")
    if mode == "separate" and not text_cols:
        raise ValueError("mode='separate' splits rows by text column, which needs text_cols")

    # normalize agg_cols to {column: stat}. a bare sequence means "the mean of
    # each", because the average per group is almost always what people want.
    if agg_cols and not group_by:
        raise ValueError("agg_cols requires group_by: there are no groups to summarize over")
    agg: Dict[str, str] = (dict(agg_cols) if isinstance(agg_cols, Mapping)
                           else {c: "mean" for c in (agg_cols or [])})
    bad_stats = {s for s in agg.values() if s not in _AGG_STATS}
    if bad_stats:
        raise ValueError(f"agg_cols statistics must be one of {sorted(_AGG_STATS)}, got {sorted(bad_stats)}")

    include_source_col = (mode == "separate")
    include_source_path = False  # we're doing CSV here; the folder variant uses this

    # we resolve these against the real header before either path runs, so the
    # two paths can't disagree about which columns exist.
    carry: List[str] = list(carry_cols or [])
    if carry:
        with in_path.open("r", newline="", encoding=encoding) as f:
            available = csv.DictReader(f, delimiter=delimiter).fieldnames or []
        absent = [c for c in carry if c not in available]
        if absent:
            if verbose:
                print(f"[text-gather] WARNING: cannot carry columns not in the input: {absent}")
            carry = [c for c in carry if c not in absent]

    # figure out where the output goes (next to the input, unless told otherwise).
    # a table in a database is as good a place as a file: the run decides
    from . import tables

    out_path = tables.parse_ref(out_csv) if out_csv is not None else tables.parse_ref(
        _default_csv_out_path(in_csv=in_path, mode=mode, text_cols=text_cols,
                              group_by=group_by))
    if not out_path.is_db:
        out_path.path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and tables.exists(out_path):
        if verbose:
            print("File with gathered text already exists; returning existing file.")
        return tables.handle(out_path)

    # no grouping? then we can just stream straight through to the output
    if not group_by:
        writer, fh, _ = _open_out_csv(
            out_path,
            include_source_col,
            include_source_path,
            include_group_count=False,
            id_col_names=(list(id_cols) if include_id_cols and id_cols else None),
            group_by_names=None,
            carry_col_names=carry,
            include_text=bool(text_cols),
        )
        carried = _carried_names(carry, ["text_id"] + (
            _emit_names(id_cols) if include_id_cols and id_cols else []))
        announce(on_progress, "counting rows")
        ticker = Ticker(on_progress,
                        count_rows(in_path, on_progress=on_progress,
                                   encoding=encoding))
        try:
            with in_path.open("r", newline="", encoding=encoding) as f:
                rdr = csv.DictReader(f, delimiter=delimiter)
                headers = rdr.fieldnames or []
                missing = [c for c in (id_cols or []) + text_cols if c not in headers]
                if missing:
                    raise ValueError(f"Missing columns: {missing}. Make sure that you try specifying a delimiter manually if you see this error message.")

                for idx, row in enumerate(rdr, start=1):
                    ticker.tick(message="copying rows")
                    # before `text_id` is used but after it is numbered, so a
                    # row left out here does not renumber the rows after it
                    # -- every other table in the run is keyed on that
                    # number, and shifting it would join the wrong rows to
                    # the wrong groups while looking perfectly reasonable.
                    if keep and not keeps_row(row, keep):
                        continue
                    text_id = _compose_id([row.get(c, "") for c in (id_cols or [])]) if id_cols else f"row_{idx}"
                    if mode == "concat":
                        parts = [row.get(c, "") for c in text_cols if row.get(c, "")]
                        # if we have text columns, a row with nothing in any of
                        # them has nothing to give us; if we have none, every
                        # row counts on its own.
                        if text_cols and not parts:
                            continue
                        row_prefix = [text_id]
                        if include_id_cols and id_cols:
                            row_prefix += [row.get(c, "") for c in _emit_names(id_cols)]
                        row_prefix += [row.get(c, "") for c in carried]
                        writer.writerow(row_prefix + ([joiner.join(parts)] if text_cols else []))
                    else:
                        for col in text_cols:
                            val = row.get(col, "")
                            if not val:
                                continue
                            row_prefix = [text_id]
                            if include_id_cols and id_cols:
                                row_prefix += [row.get(c, "") for c in _emit_names(id_cols)]
                            row_prefix += [row.get(c, "") for c in carried]
                            writer.writerow(row_prefix + [val, col])

        except BaseException:
            fh.close(ok=False)
            raise
        else:
            fh.close(ok=True)
        return tables.handle(out_path)

    # otherwise we do the grouping on disk, in two passes
    group_by = list(group_by)

    # pass 1: partition the rows into hash buckets
    tmp_base = Path(tempfile.mkdtemp(prefix="gather_partitions_", dir=str(tmp_root) if tmp_root else None))
    part_dir = tmp_base / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    # the bucket writer cache
    carried = _carried_names(carry, ["text_id"] + _emit_names(group_by))
    # the columns we're aggregating ride along in the partitions too (deduped:
    # it's fine to summarize a column that's also grouped or carried, and the
    # DictReader in pass 2 needs each name to show up exactly once).
    agg_extra = [c for c in agg if c not in set(group_by) | set(carried) | set(text_cols)]
    header_small = group_by + carried + agg_extra + text_cols
    cache = _LRUFileCache(
        max_open=max_open_bucket_files,
        newline="",
        encoding=encoding,
        delimiter=delimiter,
    )


    announce(on_progress, "counting rows")
    ticker = Ticker(on_progress,
                    count_rows(in_path, on_progress=on_progress,
                               encoding=encoding))
    try:
        with in_path.open("r", newline="", encoding=encoding) as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            headers = rdr.fieldnames or []
            missing = [c for c in group_by + list(agg) + text_cols if c not in headers]
            if missing:
                raise ValueError(f"Missing columns: {missing}. Make sure that you try specifying a delimiter manually if you see this error message.")

            for row in rdr:
                ticker.tick(message="sorting rows into groups (pass 1 of 2)")
                if keep and not keeps_row(row, keep):
                    # dropped before it is bucketed, on purpose: once a row
                    # is inside somebody else's joined text there is no way
                    # to take it back out again.
                    continue
                key_tuple = tuple(row[g] for g in group_by)
                bucket = _bucket_of_key(key_tuple, num_buckets)
                bpath = part_dir / f"bucket_{bucket:05d}.csv"
                w = cache.get(bucket, bpath, header_small)
                # only the fields we need, so the partitions stay lean
                w.writerow([row.get(c, "") for c in header_small])
    finally:
        cache.close_all()

    # pass 2: aggregate each bucket and hand it to the final writer
    writer, out_fh, _ = _open_out_csv(
        out_path,
        include_source_col,
        include_source_path,
        include_group_count=True,
        id_col_names=None,
        group_by_names=group_by,
        carry_col_names=carried,
        agg_col_names=[n for c, s in agg.items() for n in (f"{c}_{s}", f"{c}_n")],
        include_text=bool(text_cols),
    )

    buckets = sorted(part_dir.glob("bucket_*.csv"))
    ticker = Ticker(on_progress, len(buckets))
    try:
        for bfile in buckets:
            ticker.tick(message="combining the groups (pass 2 of 2)")
            # a bucket is every row whose key hashed here, which with a
            # two-level key is half the file. so the rows are not kept: each
            # group remembers its text pieces, whether its carried columns
            # agree, and running summaries, and the rows go by once.
            # summaries see every row in the group, textless ones included:
            # a person's average score is the average over what they posted,
            # not over what happened to have text. group_count still counts
            # only the rows whose text got joined, and the per-column n makes
            # the difference visible instead of leaving it for someone to
            # discover in an audit. the carried columns have to agree across
            # the rows that contributed text, same as before.
            groups: Dict[Tuple[str, ...], _Group] = {}
            with bfile.open("r", newline="", encoding=encoding) as bf:
                br = csv.DictReader(bf, delimiter=delimiter)
                for row in br:
                    key = tuple(row[g] for g in group_by)
                    group = groups.get(key)
                    if group is None:
                        group = groups[key] = _Group(carried, agg)
                    if agg:
                        group.see_agg(row)
                    if mode == "concat":
                        parts = [row.get(c, "") for c in text_cols if row.get(c, "")]
                        if text_cols and not parts:
                            continue
                        group.pieces.append(joiner.join(parts))
                        if carried:
                            group.see_carried(carried, row)
                    else:
                        wrote = False
                        for col in text_cols:
                            val = row.get(col, "")
                            if val:
                                group.per_col.setdefault(col, []).append(val)
                                wrote = True
                        if carried and wrote:
                            group.see_carried(carried, row)
            # now we write out one row per key (per column, in separate mode)
            for key, group in groups.items():
                text_id = _compose_id(key) or "group"
                kept = [v for n, v in zip(group_by, key, strict=True) if n != "text_id"]
                extra = group.agreed_cells(carried) if carried else []
                stats = group.stat_cells(agg) if agg else []
                if mode == "concat":
                    if not group.pieces:
                        continue
                    body = [joiner.join(group.pieces)] if text_cols else []
                    writer.writerow([text_id, *kept, *extra, *body, len(group.pieces), *stats])
                else:
                    for col in text_cols:
                        vals = group.per_col.get(col, [])
                        if not vals:
                            continue
                        writer.writerow([text_id, *kept, *extra, joiner.join(vals),
                                         len(vals), *stats, col])
            groups.clear()

    except BaseException:
        out_fh.close(ok=False)
        raise
    else:
        out_fh.close(ok=True)
    finally:
        # lastly, clean up after ourselves
        try:
            for p in part_dir.glob("bucket_*.csv"):
                p.unlink(missing_ok=True)
            part_dir.rmdir()
            tmp_base.rmdir()
        except Exception:
            pass

    return tables.handle(out_path)


def _db_source(csv_path):
    """The database source ``csv_path`` names, or None for a file."""
    from .tables import DB_SCHEME

    if csv_path is None:
        return None
    if isinstance(csv_path, Mapping):
        from . import sources

        return sources.coerce(csv_path)
    text = str(csv_path)
    if text.startswith(DB_SCHEME) or text.startswith("db:/"):
        from . import sources

        return sources.coerce(text)
    return None


def _agreed_values(carried: Sequence[str], rows: List[Dict[str, str]]) -> List[str]:
    """One value per carried column, or blank where the group disagrees --
    the rule the CSV gather's pass 2 applies, as a function."""
    out: List[str] = []
    for name in carried:
        distinct = {v.get(name, "") for v in rows}
        out.append(distinct.pop() if len(distinct) == 1 else "")
    return out


def _gather_from_db(spec, *, out_csv, overwrite_existing: bool,
                    text_cols: List[str], id_cols: List[str], mode: str,
                    group_by: List[str], joiner: str, include_id_cols: bool,
                    carry_cols: List[str], agg_cols, keep: List[list],
                    verbose: bool, on_progress):
    """
    The gather, reading a database table (or a join) instead of a file.

    Same output, same rules, no text on local disk: the rows arrive from
    the server already sorted by the group keys, and the same Python that
    combines a bucket of the CSV gather combines each run of equal keys
    here. Rows cannot be numbered by position -- a table has no order to
    number -- so an id is required, and the base table's key is the default.
    """
    import itertools

    from . import db, sources, tables

    mode = mode.strip().lower()
    if mode not in ("concat", "separate"):
        raise ValueError("mode must be 'concat' or 'separate'")
    if mode == "separate" and not text_cols:
        raise ValueError("mode='separate' splits rows by text column, which needs text_cols")
    if agg_cols and not group_by:
        raise ValueError("agg_cols requires group_by: there are no groups to summarize over")
    agg: Dict[str, str] = (dict(agg_cols) if isinstance(agg_cols, Mapping)
                           else {c: "mean" for c in (agg_cols or [])})
    bad_stats = {s for s in agg.values() if s not in _AGG_STATS}
    if bad_stats:
        raise ValueError(f"agg_cols statistics must be one of {sorted(_AGG_STATS)}, "
                         f"got {sorted(bad_stats)}")

    anchor = tables.parse_ref(f"db://{spec.connection}/{spec.table or 'query'}")
    session = tables._session(anchor)
    resolved = sources.resolve_columns(spec, session)
    names = [n for n, _a, _c in resolved]
    missing = [c for c in [*id_cols, *group_by, *agg, *text_cols] if c not in names]
    if missing:
        raise ValueError(f"Missing columns: {missing}. The source has: {', '.join(names)}")
    carry = [c for c in carry_cols if c in names]
    absent = [c for c in carry_cols if c not in names]
    if absent and verbose:
        print(f"[text-gather] WARNING: cannot carry columns not in the input: {absent}")

    # rows cannot be numbered by position: the key is the id, and the base
    # table's own key is what a person means when they do not say
    primary = ()
    if spec.table:
        try:
            info = session.d.describe(session, spec.table, schema=spec.schema)
            primary = tuple(c for c in info.primary_key if c in names)
        except Exception:
            primary = ()
    if not group_by and not id_cols:
        if primary:
            id_cols = list(primary)
        else:
            raise ValueError(
                f"{spec.describe()} has no key column, so its rows cannot be told "
                f"apart: name id_cols. A row numbered by its position would join "
                f"the wrong rows the next time the table is read.")

    out = (tables.parse_ref(out_csv) if out_csv is not None else tables.parse_ref(
        f"db://{spec.connection}/taters/{(spec.table or 'query')}_"
        + ("grouped_" + "_".join(_sanitize_for_filename(g) for g in group_by) if group_by
           else _sanitize_for_filename(mode) + "_"
           + "_".join(_sanitize_for_filename(c) for c in text_cols))))
    if not overwrite_existing and tables.exists(out):
        if verbose:
            print("Table with gathered text already exists; returning existing table.")
        return out

    # the joins get checked again here rather than trusted from whenever
    # the pipeline was composed, since the tables may have changed since.
    # counts only, never example keys: the run log is a local file
    for j, report in zip(spec.joins, sources.join_report(spec, session)):
        left, right = report.left, report.right
        if report.relationship == "many-to-many" and not j.allow_many_to_many:
            raise ValueError(
                f"Keys repeat on both sides of the join between {left.file} and "
                f"{right.file} ({report.many_to_many_keys:,} such "
                f"key{'s' if report.many_to_many_keys != 1 else ''}), so every "
                f"pair of rows would be combined. That usually means one more "
                f"column is needed to tell rows apart. Set allow_many_to_many on "
                f"the join to combine every pair anyway.")
        grown = report.rows_out.get("left", left.rows) - left.rows
        if grown > 0 and verbose:
            print(f"[text-gather] WARNING: some rows in {left.file} match several "
                  f"rows in {right.file}; {left.rows:,} rows become "
                  f"{left.rows + grown:,}, and each copy is measured on its own.")

    include_text = bool(text_cols)
    include_source_col = (mode == "separate")
    if group_by:
        carried = _carried_names(carry, ["text_id"] + _emit_names(group_by))
        header = ["text_id", *_emit_names(group_by), *carried]
        if include_text:
            header.append("text")
        header.append("group_count")
        agg_names = [n for c, s in agg.items() for n in (f"{c}_{s}", f"{c}_n")]
        header += agg_names
    else:
        carried = _carried_names(carry, ["text_id"] + (
            _emit_names(id_cols) if include_id_cols and id_cols else []))
        header = ["text_id", *(_emit_names(id_cols) if include_id_cols and id_cols else []),
                  *carried]
        if include_text:
            header.append("text")
        agg_names = []
    if include_source_col:
        header.append("source_col")
    hints = {c: "text" for c in header}
    hints["group_count"] = "int"
    for c, s in agg.items():
        hints[f"{c}_{s}"] = "float"
        hints[f"{c}_n"] = "int"

    # the server sorts, by the group keys first (equal values adjacent,
    # blanks and NULLs together) and then by the within-group order that
    # decides how texts are joined
    order_exprs: List[str] = []
    if group_by:
        order_exprs = [session.d.sort_key(sources.expression_of(spec, session, g))
                       for g in group_by]
    inner = list(spec.order_by) or ([f"{spec.alias}.{c}" for c in primary] if primary else [])
    if group_by and not inner:
        raise ValueError(
            f"{spec.describe()} has no key to order rows by, so the texts in a group "
            f"would be joined in whatever order the server returns them -- a "
            f"different order each time. Give the source an order_by (a timestamp, "
            f"a sequence number).")
    for ref in inner:
        a, c = sources._split(ref, spec.alias)
        order_exprs.append(f"{session.d.quote(a)}.{session.d.quote(c)}")
    # only the columns the gather uses come over the wire: a row dict of
    # every source column, for a wide join, is most of a group's weight
    need = (set(id_cols) | set(group_by) | set(carry) | set(agg) | set(text_cols)
            | {str(f[0]) for f in keep})
    wanted = [n for n in names if n in need]
    sql, params = sources.render(spec, session, columns=wanted,
                                 order_by_exprs=order_exprs or None)

    announce(on_progress, "counting rows")
    total = int(session.one(f"SELECT COUNT(*) FROM ({sql}) AS q", params) or 0)
    ticker = Ticker(on_progress, total)
    seen = [0]

    def on_rows(n: int) -> None:
        ticker.tick(n - seen[0], message=("combining the groups" if group_by
                                          else "copying rows"))
        seen[0] = n
    _names, rows = db.stream_query(session, sql, params, on_rows=on_rows)
    if keep:
        rows = (r for r in rows if keeps_row(r, keep))

    with sources.guarded(spec, session), tables.write_table(out, header, hints=hints) as writer:
        if not group_by:
            for row in rows:
                text_id = _compose_id([row.get(c, "") for c in id_cols])
                prefix = [text_id]
                if include_id_cols and id_cols:
                    prefix += [row.get(c, "") for c in _emit_names(id_cols)]
                prefix += [row.get(c, "") for c in carried]
                if mode == "concat":
                    parts = [row.get(c, "") for c in text_cols if row.get(c, "")]
                    if text_cols and not parts:
                        continue
                    writer.writerow(prefix + ([joiner.join(parts)] if include_text else []))
                else:
                    for col in text_cols:
                        val = row.get(col, "")
                        if not val:
                            continue
                        writer.writerow(prefix + [val, col])
        else:
            # the sort is checked as we go rather than with a set of every
            # key seen: the rows arrive in byte order, so a key that falls
            # below the last one is the only way a group could come back
            # around. blanks are left out of the comparison, since where a
            # NULL sorts is the one thing the engines do not agree on
            last_key = None
            for key, group in itertools.groupby(rows, key=lambda r: tuple(
                    r.get(g, "") for g in group_by)):
                if last_key is not None and "" not in key and "" not in last_key \
                        and key < last_key:
                    raise RuntimeError(
                        f"the database returned rows for {' | '.join(key)!r} after "
                        f"{' | '.join(last_key)!r}; the sort by {', '.join(group_by)} did "
                        f"not hold")
                last_key = key
                members = list(group)
                text_id = _compose_id(key) or "group"
                kept = [v for n, v in zip(group_by, key) if n != "text_id"]
                stats = _summaries(agg, members) if agg else []
                if mode == "concat":
                    pieces = []
                    with_text = []
                    for row in members:
                        parts = [row.get(c, "") for c in text_cols if row.get(c, "")]
                        if text_cols and not parts:
                            continue
                        pieces.append(joiner.join(parts))
                        with_text.append(row)
                    if text_cols and not pieces:
                        continue
                    extra = _agreed_values(carried, with_text if (carried and text_cols)
                                           else members) if carried else []
                    body = [joiner.join(pieces)] if include_text else []
                    writer.writerow([text_id, *kept, *extra, *body, len(pieces), *stats])
                else:
                    per_col: Dict[str, List[str]] = {}
                    with_text = []
                    for row in members:
                        wrote = False
                        for col in text_cols:
                            val = row.get(col, "")
                            if val:
                                per_col.setdefault(col, []).append(val)
                                wrote = True
                        if wrote:
                            with_text.append(row)
                    extra = _agreed_values(carried, with_text) if carried else []
                    for col in text_cols:
                        vals = per_col.get(col, [])
                        if not vals:
                            continue
                        writer.writerow([text_id, *kept, *extra, joiner.join(vals),
                                         len(vals), *stats, col])
    if verbose:
        print(f"[text-gather] {spec.describe()} -> {out} ({writer.rows:,} rows)")
    return out


def _read_one(args: Tuple[str, str]) -> Tuple[str, str, list]:
    """
    Read one document; the unit of work the reader pool runs.

    Module-level and argument-packed because spawn pickles it by name. Returns
    ``(status, payload, warnings)`` -- warnings raised inside a worker process
    would otherwise vanish, so they ride back as text and are re-issued by the
    parent, keeping skip-and-warn identical under any worker count.
    """
    path, encoding = args
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            text = read_document_text(Path(path), encoding=encoding)
        except (DocumentReadError, OSError) as e:
            return ("error", str(e), [str(w.message) for w in caught])
    return ("ok", text, [str(w.message) for w in caught])


def _pair_paths_with_results(files, workers: int, encoding: str,
                             reporter=None):
    """Each path with its read result, in sorted-walk order whatever the
    worker count -- the property that keeps the gathered CSV deterministic.

    The reporter (a :class:`~taters.helpers.progress.FlightReporter`) hears
    when each file's read starts and finishes, so the display can show one
    sub-bar per document in flight, the way item steps do."""
    results = ordered_parallel_map(
        _read_one,
        ((str(p), encoding) for p in files),
        # a handful of files reads faster serially than the pool even takes to
        # spawn; if the user asked for workers explicitly, we still honor it.
        workers=pool_workers(workers, len(files)),
        on_start=None if reporter is None
        else (lambda args: reporter.start(Path(args[0]).name)),
        on_finish=None if reporter is None
        else (lambda args: reporter.finish(Path(args[0]).name)),
    )
    return zip(files, results)


@records_settings(
    # one text per file, so there's no grain to record here
    binding=("root_dir", "recursive", "pattern", "encoding",
             "id_from", "include_source_path"),
    outputs=("out_csv",),
    redo_without_record=True,
    # this step decides what text EXISTS. its settings describe the
    # dataset, so we record them but keep them out of the chain that
    # downstream steps compare -- a model has to be applicable to a corpus
    # assembled differently, which is the whole point of saving one.
    defines_text=True)
def txt_folder_to_analysis_ready_csv(
    *,
    root_dir: PathLike,
    out_csv: PathLike | None = None,
    recursive: bool = False,
    pattern: str = DOCUMENT_PATTERN,
    encoding: str = "utf-8",
    id_from: str = "stem",            # "stem" | "name" | "path"
    include_source_path: bool = True, # writes 'source_path' column
    overwrite_existing: bool = False, # if the file already exists, let's not overwrite by default
    verbose: bool = True,
    on_progress=None,                 # on_progress(done, total, message) per file
    workers: int = 0,                 # parallel readers; 0 = auto, 1 = off
) -> Path:
    """
    Stream a folder of documents into an analysis-ready CSV with predictable,
    reproducible IDs.

    Documents are ``.txt``, ``.docx``, ``.doc`` and ``.pdf`` -- text is
    extracted per type by :func:`taters.helpers.doc_text.read_document_text`
    (no OCR: a PDF without a machine-readable text layer has zero text). A
    document that cannot be read -- corrupt, password-protected, a legacy
    ``.doc`` with no way to convert it -- is skipped with a warning naming it,
    never allowed to take the whole gather down.

    For each readable file matching `pattern`, the emitted row contains:
      - `text_id`: the basename (stem), full filename, or relative path (see
        `id_from`), and
      - `text`: the extracted text.
      - `source_path`: optional column with path relative to `root_dir`.

    Parameters
    ----------
    root_dir
        Folder containing documents.
    out_csv
        Destination CSV. If `None`, a descriptive default is created next to
        `root_dir` (e.g., `<folder>_txt_recursive_*.csv`).
    recursive
        Recurse into subfolders. Default: `False`.
    pattern
        Glob(s) for matching files; several can be joined with ``;``.
        Default: every document type (``"*.txt;*.docx;*.doc;*.pdf"``).
    encoding
        Decoding for plain-text files. Default: `"utf-8"`.
    workers
        Parallel reader processes -- PDF and Word parsing is CPU-bound, and a
        big folder reads several times faster in parallel. ``0`` (default)
        means automatic: three-quarters of the logical cores; ``1`` turns parallelism off.
        The output file is **identical whatever the worker count**: files are
        walked in sorted order and results are written in that same order.
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
    from . import tables

    root = _ensure_path(root_dir)
    out_path = tables.parse_ref(out_csv) if out_csv is not None else tables.parse_ref(
        _default_txt_out_path(root, id_from=id_from, recursive=recursive, pattern=pattern))
    if not out_path.is_db:
        out_path.path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and tables.exists(out_path):
        if verbose:
            print("File with gathered text already exists; returning existing file.")
        return tables.handle(out_path)

    writer, fh, _ = _open_out_csv(out_path, include_source_col=False, include_source_path=include_source_path)
    try:
        # several globs, but one deduplicated, *sorted* walk: with more than one
        # pattern the concatenation order would otherwise be up to the
        # filesystem, and two runs over the same folder need to write the same
        # file.
        found: set = set()
        for one_pattern in pattern.split(";"):
            one_pattern = one_pattern.strip()
            if one_pattern:
                found.update(root.rglob(one_pattern) if recursive
                             else root.glob(one_pattern))
        files = [p for p in sorted(found) if p.is_file()]
        if id_from not in ("stem", "name", "path"):
            raise ValueError("id_from must be 'stem', 'name', or 'path'")

        # the gather gets to be its own visible phase: the step row reads
        # "gathering N documents" and (on displays that can show it) each
        # document being read gets a sub-bar of its own -- same language as
        # ffmpeg's or Whisper's per-file rows, rather than a step label that
        # looks like it's doing somebody else's work.
        from .progress import FlightReporter

        reporter = FlightReporter(on_progress, len(files),
                                  "gathering documents") \
            if on_progress is not None else None
        for p, (status, payload, relayed) in _pair_paths_with_results(
                files, workers, encoding, reporter):
            if reporter is not None:
                reporter.consumed()
            for message in relayed:
                # warnings raised inside a worker process would otherwise just
                # vanish; they ride back with the result and we re-issue them here.
                warnings.warn(message, stacklevel=2)
            if status == "error":
                # a broken document -- an image wearing a .docx name, a
                # truncated PDF, an unreadable file -- only costs itself: we
                # treat it as having no text, name it in a warning, and never
                # let it take the whole run down.
                warnings.warn(f"{payload} Skipped.", stacklevel=2)
                continue
            text = payload
            if id_from == "stem":
                text_id = p.stem
            elif id_from == "name":
                text_id = p.name
            else:
                text_id = str(p.relative_to(root))
            if len(text) > 500_000:
                # a whole proceedings volume or book as one "document".
                # perfectly legal, but dictionary-style scoring grows
                # super-linearly with length, so one of these can take minutes
                # while its neighbors take milliseconds -- better to say so
                # now, and name it, than to look hung later.
                warnings.warn(
                    f"'{p.name}' extracted {len(text):,} characters -- a very "
                    "large document; some analyses can be slow on it.",
                    stacklevel=2)
            if not text.strip() and p.suffix.lower() != ".txt":
                # a scanned PDF, an image-only Word file: zero text is the
                # documented answer, but if we stayed quiet it'd look like we
                # lost their data.
                warnings.warn(
                    f"'{p.name}' contains no machine-readable text (no OCR "
                    "is attempted); it contributes nothing.", stacklevel=2)
                continue
            if include_source_path:
                writer.writerow([text_id, text, str(p.relative_to(root))])
            else:
                writer.writerow([text_id, text])
    except BaseException:
        fh.close(ok=False)
        raise
    else:
        fh.close(ok=True)

    return tables.handle(out_path)


def resolve_analysis_ready(*, csv_path=None, txt_dir=None, analysis_csv=None,
                           gathered_csv=None, text_cols=("text",), id_cols=None,
                           mode="concat", group_by=None, delimiter=",",
                           encoding="utf-8-sig", joiner=" ", num_buckets=64,
                           max_open_bucket_files=64, tmp_root=None,
                           recursive=True, pattern="*.txt", id_from="stem",
                           include_source_path=False, overwrite_existing=False,
                           on_progress=None, workers=0, carry_cols=None,
                           verbose=None) -> Path:
    """
    Accept an analysis-ready table, or gather one -- the analyzers' front door.

    Every text analyzer takes its input three ways: a spreadsheet to gather
    from, a folder of documents to gather from, or a table already gathered.
    The forty lines that told those apart, announced the gather and called
    one of the two gatherers with a dozen forwarded settings were copied into
    eleven modules, and had begun to drift (one forwarded ``verbose`` and
    ``carry_cols``, the others did not). This is that block, once. Settings
    recording is unaffected: the provenance decorator reads the analyzer's
    own bound arguments, not this function's.

    ``carry_cols`` and ``verbose`` are forwarded to the gatherers only when
    given, so the analyzers that never passed them behave exactly as before.
    """
    # a file comes back as the Path it always was -- every analyzer that
    # still opens it by hand keeps working -- and only a table in a
    # database comes back as a ref
    from .tables import exists, handle, parse_ref

    if analysis_csv is not None:
        analysis_ready = parse_ref(analysis_csv)
        if not exists(analysis_ready):
            raise FileNotFoundError(f"analysis_csv not found: {analysis_ready}")
        return handle(analysis_ready)
    if (csv_path is None) == (txt_dir is None):
        raise ValueError(
            "Provide exactly one of csv_path or txt_dir (or pass analysis_csv).")

    from .progress import announce

    # gathering reads the whole source and writes the analysis-ready table.
    # on a big spreadsheet that's the longest silent stretch in the run, so
    # we say what's happening before it starts.
    announce(on_progress, "reading the input")
    extra = {}
    if verbose is not None:
        extra["verbose"] = verbose
    if csv_path is not None:
        if carry_cols:
            extra["carry_cols"] = list(carry_cols)
        return handle(parse_ref(csv_to_analysis_ready_csv(
            csv_path=csv_path, out_csv=gathered_csv,
            text_cols=list(text_cols),
            id_cols=list(id_cols) if id_cols else None,
            mode=mode, group_by=list(group_by) if group_by else None,
            delimiter=delimiter, encoding=encoding, joiner=joiner,
            num_buckets=num_buckets,
            max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
            overwrite_existing=overwrite_existing, **extra)))
    return handle(parse_ref(txt_folder_to_analysis_ready_csv(
        root_dir=txt_dir, out_csv=gathered_csv, recursive=recursive,
        pattern=pattern, encoding=encoding, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers, **extra)))


# ---------------------------------------------------------------------------
# Command line -- derived from the function(s) above; see helpers.cliargs.CliSpec.
# The aliases and legacy flags are the spellings the hand-written parser used,
# kept so every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"csv": csv_to_analysis_ready_csv, "txt": txt_folder_to_analysis_ready_csv},
    description='Stream a spreadsheet (csv) or a folder of documents (txt) into an analysis-ready table.',
    aliases={
        'csv_path': ['--csv'],
        'out_csv': ['--out'],
        'root_dir': ['--txt-dir'],
    },
    legacy={
        '--no-source-path': ['--include-source-path', 'false'],
    },
    # the old command line had no subcommand: --csv or --txt-dir told us which.
    rewrite=lambda argv: (["csv"] if "--csv" in argv or "--csv-path" in argv
                          else ["txt"] if "--txt-dir" in argv or "--root-dir" in argv
                          else []) + list(argv)
    if not argv or argv[0] not in ("csv", "txt") else list(argv),
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
