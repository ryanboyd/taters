"""
Merge, stack and reshape spreadsheets, without holding any of them in memory.

Four things a person does to a spreadsheet before it is ready to analyze,
and that until now meant leaving Taters for R or pandas:

* :func:`merge_spreadsheets` -- add the columns of one file onto the rows of
  another, matching rows by an ID they share (a participant's age and
  condition onto every one of their trials).
* :func:`stack_spreadsheets` -- several files one under another (one per
  session, per wave, per participant), with a column saying which.
* :func:`wide_to_long` -- ``score_t1, score_t2, score_t3`` into one ``score``
  column plus a column saying which time.
* :func:`long_to_wide` -- the other way round.

Every one of them has a planner (:func:`plan_merge` and friends) that reads
the files, writes nothing, and comes back with a report: how many rows match,
which don't, what will collide. The wizard reads that report back in words
before anything is written, and the writers run the same plan again and
refuse the same things, so the API and the app cannot disagree.

Two rules shape everything here. Nothing holds a file's rows: the stacks and
the unpivot stream, and the merge and the pivot go through an on-disk sqlite
database in a temp folder, the way the n-gram counter already spills. And
keys are compared as text, never as floats: a 17-digit ID must never equal
its neighbor because they round the same.

Unlike the text gathers, ``encoding`` and ``delimiter`` default to *detect*
rather than to utf-8-sig and comma. The files this handles are the messy
ones -- a cp1252 export from Windows, a semicolon file from a European
Excel, a tab-separated "Unicode Text" dump -- and a default that only works
on the tidy ones is the wrong default. The output is always utf-8-sig and
comma-separated, which every other Taters step reads.
"""
from __future__ import annotations

import codecs
import csv
import json
import os
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple, Union)

from .atomic import atomic_write
from .cliargs import CliSpec
from .csvio import widen_csv_field_limit
from .feature_columns import SEPARATOR
from .progress import Ticker, announce
from .provenance import records_settings
from .text_gather import _detect_delimiter

__all__ = [
    "merge_spreadsheets", "stack_spreadsheets", "wide_to_long", "long_to_wide",
    "plan_merge", "plan_stack", "plan_wide_to_long", "plan_long_to_wide",
    "detect_format", "normalize_key",
    "TableFormat", "MergeReport", "StackReport", "UnpivotReport", "PivotReport",
    "SideSummary", "ColumnClash",
    "KEY_MATCH", "HOW", "ON_DUPLICATE", "OUTPUT_ENCODING",
]

widen_csv_field_limit()

PathLike = Union[str, Path]
Progress = Optional[Callable[..., None]]

#: The separators a spreadsheet can actually have; the same whitelist the
#: wizard's scan uses. The Sniffer, handed prose, happily returns "e".
DELIMITERS = (",", "\t", ";", "|")

#: How two keys are compared. ``exact`` is byte-for-byte; ``trim`` ignores
#: surrounding whitespace; ``loose`` also ignores case, invisible characters,
#: inner whitespace runs and the way a number is written (``007`` is ``7``).
KEY_MATCH = ("exact", "trim", "loose")

#: Which rows a merge keeps: every left row, only the matched ones, or both
#: files' rows.
HOW = ("left", "inner", "outer")

#: What a pivot does when one key has the same name twice.
ON_DUPLICATE = ("refuse", "first", "last", "mean")

#: What every file written here is encoded as, whatever came in.
OUTPUT_ENCODING = "utf-8-sig"

#: Joins the parts of a composite key. A control character, so no real cell
#: can contain it and ``a|b`` + ``c`` can never equal ``a`` + ``b|c``.
_KEY_SEP = "\x1f"

#: How many example keys a report carries per complaint. Enough to recognize
#: the problem by, few enough to read.
_EXAMPLES = 5

#: Rows per sqlite ``executemany``.
_BATCH = 10_000

#: How often a reader reports progress, in rows.
_TICK_EVERY = 2_000

#: A plain decimal number: an optional sign, digits, an optional fraction.
#: Deliberately *not* ``float()``'s grammar: no exponent, no ``nan``, no
#: ``inf``, no underscores, so an ID that merely looks numeric is never
#: rounded on its way to being compared.
_DECIMAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

#: What Excel writes when it decides a long ID is a number: ``1.23457E+11``.
_EXCEL_MANGLED = re.compile(r"^\d(?:\.\d+)?E[+-]?\d+$", re.IGNORECASE)

#: Characters that are invisible on screen and routinely pasted into IDs.
_INVISIBLE = re.compile("[​‌‍﻿­]")
_INNER_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Reading a file: encoding, delimiter, header
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableFormat:
    """
    How a spreadsheet has to be read, worked out from the file itself.

    ``columns`` are the names as this module will call them: stripped of
    surrounding spaces and any stray BOM, a blank header filled in as
    ``column_<n>``, a duplicate made unique as ``name_2``. ``renamed`` says
    which ones changed, so a screen can mention it.
    """

    path: Path
    encoding: str
    delimiter: str
    columns: Tuple[str, ...]
    renamed: Tuple[Tuple[str, str], ...] = ()

    @property
    def name(self) -> str:
        return self.path.name

    def notes(self) -> List[str]:
        """What a person should know about how the file was read, or []."""
        out = []
        if self.encoding == "utf-16":
            out.append(f"{self.name} is a UTF-16 file (Excel's \"Unicode Text\"), "
                       f"and was read as one.")
        elif self.encoding in ("cp1252", "latin-1"):
            out.append(f"{self.name} isn't UTF-8, so it was read as Windows "
                       f"({self.encoding}) text. Accented letters should come "
                       f"through fine.")
        if self.delimiter != ",":
            word = {"\t": "tab", ";": "semicolon", "|": "pipe"}[self.delimiter]
            out.append(f"{self.name} is {word}-separated.")
        for was, now in self.renamed:
            if was:
                out.append(f"{self.name} has more than one column called "
                           f"{was}; the next one is {now}.")
            else:
                out.append(f"A column in {self.name} had no name; it's called "
                           f"{now}.")
        return out


def _decodes(path: Path, encoding: str) -> bool:
    """Whether the whole file decodes as ``encoding`` -- the whole file,
    because the one accented name is on row 900,000, not in a sample."""
    decoder = codecs.getincrementaldecoder(encoding)()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    decoder.decode(b"", final=True)
                    return True
                decoder.decode(chunk)
    except UnicodeDecodeError:
        return False


def _detect_encoding(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(4)
    if head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if _decodes(path, "utf-8"):
        return "utf-8-sig"       # the BOM-tolerant spelling costs nothing
    if _decodes(path, "cp1252"):
        return "cp1252"
    # latin-1 maps every byte to something, so this is the floor
    return "latin-1"


def _sniff(path: Path, encoding: str) -> str:
    with path.open("rb") as fh:
        sample = fh.read(64 * 1024)
    text = sample.decode(encoding, errors="ignore")
    found = _detect_delimiter(text.encode("utf-8"), default=",")
    return found if found in DELIMITERS else ","


def _unique_names(raw: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Tidy header names, made unique; and which ones changed."""
    names: List[str] = []
    renamed: List[Tuple[str, str]] = []
    taken = set()
    for i, cell in enumerate(raw, 1):
        name = str(cell).lstrip("﻿").strip()
        if not name:
            name = f"column_{i}"
            renamed.append(("", name))
        if name in taken:
            base, n = name, 2
            while f"{base}_{n}" in taken or f"{base}_{n}" in raw:
                n += 1
            renamed.append((name, f"{base}_{n}"))
            name = f"{base}_{n}"
        taken.add(name)
        names.append(name)
    return names, renamed


def detect_format(path: PathLike, *, encoding: Optional[str] = None,
                  delimiter: Optional[str] = None) -> TableFormat:
    """
    Work out how to read a spreadsheet: its encoding, separator and columns.

    Parameters
    ----------
    path : str or Path
        The spreadsheet.
    encoding : str, optional
        Use this instead of detecting. Detection checks for a BOM, then
        whether the whole file is UTF-8, then falls back to cp1252 and
        finally latin-1 -- which reads anything, so a file is never refused
        for its encoding.
    delimiter : str, optional
        Use this instead of sniffing. Sniffing is done on decoded text, so a
        UTF-16 file sniffs correctly.

    Returns
    -------
    TableFormat

    Raises
    ------
    FileNotFoundError
        No such file.
    ValueError
        The file is empty, has no header row, or does not look like a CSV
        at all (an .xlsx or .docx is a zip, and its first bytes decode as
        text full of control characters).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if path.stat().st_size == 0:
        raise ValueError(f"{path.name} is empty")
    encoding = encoding or _detect_encoding(path)
    delimiter = delimiter or _sniff(path, encoding)
    try:
        with path.open("r", newline="", encoding=encoding) as fh:
            header = next(csv.reader(fh, delimiter=delimiter), None)
    except UnicodeDecodeError:
        raise ValueError(
            f"{path.name} isn't readable as {encoding}; leave the encoding "
            f"unset and Taters will work it out.") from None
    if not header or not any(str(c).strip() for c in header):
        raise ValueError(f"{path.name} has no header row to read columns from")
    if any(ch < " " and ch not in "\t\r\n" for cell in header for ch in str(cell)):
        raise ValueError(
            f"{path.name} looks like an Excel or Word file, not a CSV. In "
            f"Excel: File, Save As, CSV UTF-8.")
    names, renamed = _unique_names(header)
    return TableFormat(path=path, encoding=encoding, delimiter=delimiter,
                       columns=tuple(names), renamed=tuple(renamed))


class _Reader:
    """
    The data rows of a table, one list at a time, every row the header's
    width: short rows padded with blanks, long rows cut, both counted.

    Iterating opens the file and never keeps a row. ``rows``, ``short`` and
    ``long`` are the counts after a full pass.
    """

    def __init__(self, fmt: TableFormat, *,
                 on_rows: Optional[Callable[[int], None]] = None):
        self.fmt = fmt
        self.on_rows = on_rows
        self.rows = self.short = self.long = 0

    def __iter__(self) -> Iterator[List[str]]:
        width = len(self.fmt.columns)
        self.rows = self.short = self.long = 0
        with self.fmt.path.open("r", newline="",
                                encoding=self.fmt.encoding) as fh:
            reader = csv.reader(fh, delimiter=self.fmt.delimiter)
            next(reader, None)
            for raw in reader:
                if not raw:
                    continue        # a blank line is not a row
                if len(raw) < width:
                    self.short += 1
                    raw = raw + [""] * (width - len(raw))
                elif len(raw) > width:
                    self.long += 1
                    raw = raw[:width]
                self.rows += 1
                yield raw
                if self.on_rows is not None and self.rows % _TICK_EVERY == 0:
                    self.on_rows(self.rows)
        if self.on_rows is not None:
            self.on_rows(self.rows)


def _positions(fmt: TableFormat, wanted: Sequence[str], *, what: str,
               allow_missing: bool = False) -> List[int]:
    """Column positions by name, with a near miss named when one is missing."""
    index = {c: i for i, c in enumerate(fmt.columns)}
    out = []
    for name in wanted:
        name = str(name).strip()
        if name in index:
            out.append(index[name])
            continue
        if allow_missing:
            out.append(-1)
            continue
        folded = {c.casefold(): c for c in fmt.columns}
        hint = folded.get(name.casefold())
        raise ValueError(
            f"{fmt.name} has no column {name!r} for {what}"
            + (f" -- did you mean {hint!r}?" if hint else "")
            + f". Its columns are: {', '.join(fmt.columns)}")
    return out


def _as_names(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value]


def _check_output(out: Path, inputs: Sequence[Path]) -> None:
    """Writing over one of the files being read is refused, always."""
    for src in inputs:
        same = False
        try:
            same = out.exists() and os.path.samefile(out, src)
        except OSError:
            same = False
        if same or out.resolve() == Path(src).resolve():
            raise ValueError(
                f"{out.name} is one of the files being read; pick another "
                f"name so the original stays as it was")


def _display(parts: Sequence[str]) -> str:
    """A composite key as a person would write it."""
    return " | ".join(parts)


def _counting(ticker: Ticker) -> Callable[[int], None]:
    """A reader reports its running total; a Ticker wants increments."""
    seen = [0]

    def on_rows(n: int) -> None:
        ticker.tick(n - seen[0])
        seen[0] = n
    return on_rows


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def _canonical_number(text: str) -> Optional[str]:
    """``007``, ``7.0``, ``+7`` and ``-0`` as ``7``; None if not a plain number."""
    if not _DECIMAL.match(text):
        return None
    sign = ""
    if text[0] in "+-":
        sign, text = text[0], text[1:]
    whole, _, frac = text.partition(".")
    whole = whole.lstrip("0") or "0"
    frac = frac.rstrip("0")
    if whole == "0" and not frac:
        return "0"
    return f"{'-' if sign == '-' else ''}{whole}{'.' + frac if frac else ''}"


def normalize_key(value: Any, mode: str = "trim") -> str:
    """
    One key cell as it is compared.

    Parameters
    ----------
    value : any
        The cell. ``None`` reads as blank.
    mode : {"exact", "trim", "loose"}
        ``exact`` compares the text as it stands. ``trim`` drops surrounding
        whitespace, including the non-breaking kind, because invisible
        padding is never a meaningful difference. ``loose`` also drops
        zero-width characters, collapses inner whitespace to one space,
        normalizes Unicode to NFC, ignores case, and writes a plain number
        one way -- so ``007``, ``7``, ``7.0`` and ``+7`` are one key. Never
        a float: ``1e3`` is not a number here, and a 17-digit ID cannot
        collide with its neighbor.

    Returns
    -------
    str
    """
    if mode not in KEY_MATCH:
        raise ValueError(f"key_match must be one of {KEY_MATCH}, not {mode!r}")
    text = "" if value is None else str(value)
    if mode == "exact":
        return text
    text = text.strip()
    if mode == "trim":
        return text
    text = _INVISIBLE.sub("", text)
    text = _INNER_SPACE.sub(" ", text).strip()
    text = unicodedata.normalize("NFC", text).casefold()
    number = _canonical_number(text)
    return number if number is not None else text


def _key(parts: Sequence[str], mode: str) -> Optional[str]:
    """A composite key under ``mode``, or None when any part is blank: a
    blank never matches anything, the way SQL treats NULL."""
    if any(not str(p).strip() for p in parts):
        return None
    return _KEY_SEP.join(normalize_key(p, mode) for p in parts)


def _same_cell(a: str, b: str) -> bool:
    """Whether two cells say the same thing: trimmed, and ``25`` is ``25.0``."""
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    na, nb = _canonical_number(a), _canonical_number(b)
    return na is not None and na == nb


def _sqlite(tmp_root: Optional[PathLike]) -> Tuple[sqlite3.Connection, Path]:
    """A throwaway database in a folder of its own; the caller removes it."""
    folder = Path(tempfile.mkdtemp(prefix="taters_reshape_",
                                   dir=str(tmp_root) if tmp_root else None))
    con = sqlite3.connect(str(folder / "work.sqlite3"))
    # same settings the n-gram spill uses: nothing here survives the call,
    # so durability would be paying for nothing
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA cache_size=-65536")
    return con, folder


def _discard(con: Optional[sqlite3.Connection], folder: Optional[Path]) -> None:
    if con is not None:
        try:
            con.close()
        except Exception:
            pass
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SideSummary:
    """One file's half of a merge report."""

    file: str
    encoding: str
    delimiter: str
    rows: int
    keys: Tuple[str, ...]
    renamed_headers: Tuple[Tuple[str, str], ...]
    ragged_short: int
    ragged_long: int
    #: Rows whose key is blank in any part; they match nothing.
    blank_keys: int
    distinct_keys: int
    #: Keys that appear on more than one row.
    duplicated_keys: int
    rows_matched: int
    rows_unmatched: int
    unmatched_keys: int
    #: The first few unmatched keys, in file order, as written in the file.
    unmatched_examples: Tuple[str, ...]
    #: Keys this file keeps apart that ``loose`` matching would fold together.
    loose_collisions: int
    loose_collision_examples: Tuple[Tuple[str, str], ...]
    #: Keys that look like ``1.23457E+11``: Excel rounded them.
    mangled_ids: int
    key_samples: Tuple[str, ...]


@dataclass(frozen=True)
class ColumnClash:
    """A non-key column present in both files."""

    name: str
    agrees: bool
    disagreements: int
    example: Optional[Tuple[str, str, str]]
    #: What the column(s) are called in the output: one name when the files
    #: agree, two when they do not.
    output_names: Tuple[str, ...]


@dataclass(frozen=True)
class MergeReport:
    """What a merge would do, before it does it."""

    left: SideSummary
    right: SideSummary
    on: Tuple[str, ...]
    right_on: Tuple[str, ...]
    key_match: str
    #: one-to-one, one-to-many (a left row matches several right rows),
    #: many-to-one, or many-to-many.
    relationship: str
    many_to_many_keys: int
    many_to_many_examples: Tuple[Tuple[str, int, int], ...]
    #: Rows the output would have, for each ``how``.
    rows_out: Dict[str, int]
    #: Left rows that ``loose`` matching would match and the chosen mode does not.
    loose_extra_matches: int
    loose_examples: Tuple[Tuple[str, str], ...]
    clashes: Tuple[ColumnClash, ...]
    columns_out: Tuple[str, ...]

    @property
    def matched(self) -> bool:
        return self.left.rows_matched > 0

    def refusal(self, *, allow_many_to_many: bool = False) -> Optional[str]:
        """Why the merge must not be written as asked, or None."""
        if self.relationship == "many-to-many" and not allow_many_to_many:
            shown = "; ".join(f"{k}: {nl} rows in {self.left.file} × {nr} in "
                              f"{self.right.file} = {nl * nr} combined rows"
                              for k, nl, nr in self.many_to_many_examples)
            return (f"Some {_display(self.on)} values appear more than once in "
                    f"both files ({shown}; {self.many_to_many_keys:,} like "
                    f"this). That usually means one more column is needed to "
                    f"tell rows apart -- a session or a time point. Pass "
                    f"allow_many_to_many=True to combine every pair anyway.")
        return None

    def sentences(self, how: str = "left") -> List[str]:
        """The report in words, as the wizard reads it back."""
        left, right, keys = self.left, self.right, _display(self.on)
        out: List[str] = []
        if not left.rows_matched and left.rows > left.blank_keys:
            # nothing matched at all: the formats side by side are the
            # fastest way to see "P001 on one side, 1 on the other"
            out.append(f"No row in {left.file} matches a row in {right.file}. "
                       f"{left.file} keys look like: "
                       f"{', '.join(left.key_samples) or '(none)'}. "
                       f"{right.file} keys look like: "
                       f"{', '.join(right.key_samples) or '(none)'}.")
        elif left.rows_unmatched:
            eg = ", ".join(left.unmatched_examples)
            out.append(f"{left.rows_matched:,} of {left.rows:,} rows in "
                       f"{left.file} match a row in {right.file}; "
                       f"{left.rows_unmatched:,} do not (e.g. {eg}).")
        elif left.rows_matched:
            out.append(f"Every row in {left.file} with a {keys} matches a row "
                       f"in {right.file}.")
        if right.rows_unmatched:
            eg = ", ".join(right.unmatched_examples)
            out.append(f"{right.rows_unmatched:,} of {right.rows:,} rows in "
                       f"{right.file} have no match in {left.file} (e.g. {eg}).")
        for side in (left, right):
            if side.blank_keys:
                out.append(f"{side.blank_keys:,} rows in {side.file} have no "
                           f"{keys}; they can't match anything.")
        if self.relationship == "many-to-many":
            out.append(self.refusal() or "")
        elif self.rows_out.get("inner", 0) > left.rows_matched:
            out.append(f"Some rows in {left.file} match several rows in "
                       f"{right.file} and will repeat once per match: "
                       f"{self.rows_out[how]:,} rows out of {left.rows:,}.")
        for clash in self.clashes:
            if clash.agrees:
                out.append(f"{clash.name} is in both files and agrees on every "
                           f"matched row, so it's kept once.")
            else:
                k, a, b = clash.example or ("", "", "")
                out.append(f"{clash.name} is in both files and differs on "
                           f"{clash.disagreements:,} matched rows (e.g. {k}: "
                           f"{a!r} vs {b!r}); both are kept, as "
                           f"{' and '.join(clash.output_names)}.")
        if self.loose_extra_matches:
            l, r = self.loose_examples[0] if self.loose_examples else ("", "")
            out.append(f"Some keys only differ in how they're written ({l} in "
                       f"{left.file} is {r} in {right.file}); matching those "
                       f"too would add {self.loose_extra_matches:,} rows.")
        for side in (left, right):
            if side.loose_collisions:
                a, b = side.loose_collision_examples[0]
                out.append(f"{side.file} has keys that differ only in case or "
                           f"spacing ({a}, {b}), so those differences matter "
                           f"there.")
            if side.mangled_ids:
                out.append(f"{side.mangled_ids:,} keys in {side.file} look like "
                           f"1.23457E+11, which Excel writes after rounding a "
                           f"long number; those can't be matched reliably. "
                           f"Re-export that column as text.")
        return [s for s in out if s]

    def to_dict(self) -> dict:
        return asdict(self)


class _MergeEngine:
    """
    The merge, in three passes and a sqlite file: the right table loaded,
    the left table scanned against it, and the output written. The plan is
    passes one and two; the writer is all three on the same store.
    """

    def __init__(self, left: TableFormat, right: TableFormat, *,
                 on: Sequence[str], right_on: Sequence[str],
                 right_cols: Optional[Sequence[str]], key_match: str,
                 tmp_root: Optional[PathLike]):
        self.left, self.right = left, right
        self.on, self.right_on = list(on), list(right_on)
        self.key_match = key_match
        self.left_key = _positions(left, self.on, what="the matching columns")
        self.right_key = _positions(right, self.right_on,
                                    what="the matching columns")
        # which right columns ride along: all the non-key ones unless told
        if right_cols is None:
            keep = [c for i, c in enumerate(right.columns) if i not in self.right_key]
        else:
            keep = []
            for c in _as_names(right_cols):
                pos = _positions(right, [c], what="the columns to bring over")[0]
                if pos in self.right_key:
                    continue        # a key is matched, not brought over
                if c not in keep:
                    keep.append(c)
        self.keep = keep
        self.keep_pos = _positions(right, keep, what="the columns to bring over")
        left_non_key = [c for i, c in enumerate(left.columns) if i not in self.left_key]
        self.shared = [c for c in keep if c in left_non_key]
        self.left_pos = {c: i for i, c in enumerate(left.columns)}
        self.con: Optional[sqlite3.Connection] = None
        self.folder: Optional[Path] = None
        self.tmp_root = tmp_root
        self._right_reader: Optional[_Reader] = None
        self._left_reader: Optional[_Reader] = None
        self.mangled = {"left": 0, "right": 0}
        self.samples: Dict[str, List[str]] = {"left": [], "right": []}
        # shared column -> (disagreements, example)
        self.disagree: Dict[str, Tuple[int, Optional[Tuple[str, str, str]]]] = \
            {c: (0, None) for c in self.shared}
        self._report: Optional[MergeReport] = None

    def __enter__(self) -> "_MergeEngine":
        self.con, self.folder = _sqlite(self.tmp_root)
        self.con.execute("CREATE TABLE R (ord INTEGER PRIMARY KEY, k TEXT, "
                         "kl TEXT, raw TEXT, row TEXT)")
        self.con.execute("CREATE TABLE L (ord INTEGER PRIMARY KEY, k TEXT, "
                         "kl TEXT, raw TEXT)")
        return self

    def __exit__(self, *_exc) -> None:
        _discard(self.con, self.folder)
        self.con = self.folder = None

    # ---- the passes --------------------------------------------------

    def _keys(self, row: Sequence[str], positions: Sequence[int], side: str
              ) -> Tuple[Optional[str], Optional[str], List[str]]:
        raw = [row[i] for i in positions]
        if any(_EXCEL_MANGLED.match(v.strip()) for v in raw):
            self.mangled[side] += 1
        if len(self.samples[side]) < 3 and all(v.strip() for v in raw):
            self.samples[side].append(_display(raw))
        return _key(raw, self.key_match), _key(raw, "loose"), raw

    def load_right(self, on_rows=None) -> None:
        assert self.con is not None
        reader = _Reader(self.right, on_rows=on_rows)
        self._right_reader = reader
        batch = []
        with self.con:
            for row in reader:
                k, kl, raw = self._keys(row, self.right_key, "right")
                batch.append((k, kl, json.dumps(raw),
                              json.dumps([row[i] for i in self.keep_pos])))
                if len(batch) >= _BATCH:
                    self.con.executemany("INSERT INTO R (k, kl, raw, row) "
                                         "VALUES (?, ?, ?, ?)", batch)
                    batch.clear()
            if batch:
                self.con.executemany("INSERT INTO R (k, kl, raw, row) "
                                     "VALUES (?, ?, ?, ?)", batch)
            self.con.execute("CREATE INDEX R_k ON R (k)")
            self.con.execute("CREATE INDEX R_kl ON R (kl)")

    def _matches(self, k: str) -> List[List[str]]:
        assert self.con is not None
        return [json.loads(r) for (r,) in self.con.execute(
            "SELECT row FROM R WHERE k = ? ORDER BY ord", (k,))]

    def scan_left(self, on_rows=None) -> None:
        assert self.con is not None
        reader = _Reader(self.left, on_rows=on_rows)
        self._left_reader = reader
        batch = []
        shared = [(c, self.left_pos[c], self.keep.index(c)) for c in self.shared]
        last_key: Optional[str] = None
        last_rows: List[List[str]] = []
        with self.con:
            for row in reader:
                k, kl, raw = self._keys(row, self.left_key, "left")
                batch.append((k, kl, json.dumps(raw)))
                if len(batch) >= _BATCH:
                    self.con.executemany("INSERT INTO L (k, kl, raw) VALUES (?, ?, ?)",
                                         batch)
                    batch.clear()
                if not shared or k is None:
                    continue
                # does the right file say the same thing in the columns both
                # files have? one lookup per key, cached for the usual case of
                # a sorted left file
                if k != last_key:
                    last_key, last_rows = k, self._matches(k)
                for name, lpos, rpos in shared:
                    mine = row[lpos]
                    for other in last_rows:
                        theirs = other[rpos]
                        if not mine.strip() or not theirs.strip():
                            continue
                        if not _same_cell(mine, theirs):
                            n, eg = self.disagree[name]
                            self.disagree[name] = (
                                n + 1, eg or (_display(raw), mine, theirs))
                            break
            if batch:
                self.con.executemany("INSERT INTO L (k, kl, raw) VALUES (?, ?, ?)",
                                     batch)
            self.con.execute("CREATE INDEX L_k ON L (k)")
            self.con.execute("CREATE INDEX L_kl ON L (kl)")

    # ---- the report ----------------------------------------------------

    def _one(self, sql: str, *args) -> Any:
        assert self.con is not None
        return self.con.execute(sql, args).fetchone()[0]

    def _side(self, table: str, other: str, fmt: TableFormat, reader: _Reader,
              keys: Sequence[str], side: str) -> SideSummary:
        con = self.con
        assert con is not None
        one = self._one
        matched = one(f"SELECT COUNT(*) FROM {table} WHERE k IN "
                      f"(SELECT k FROM {other} WHERE k IS NOT NULL)")
        unmatched_sql = (f"FROM {table} WHERE k IS NOT NULL AND k NOT IN "
                         f"(SELECT k FROM {other} WHERE k IS NOT NULL)")
        unmatched = one(f"SELECT COUNT(*) {unmatched_sql}")
        unmatched_keys = one(f"SELECT COUNT(DISTINCT k) {unmatched_sql}")
        examples = tuple(_display(json.loads(raw)) for raw, _o in con.execute(
            f"SELECT raw, MIN(ord) {unmatched_sql} GROUP BY k ORDER BY MIN(ord) "
            f"LIMIT {_EXAMPLES}"))
        collisions = [kl for (kl,) in con.execute(
            f"SELECT kl FROM {table} WHERE kl IS NOT NULL GROUP BY kl "
            f"HAVING COUNT(DISTINCT k) > 1 ORDER BY MIN(ord)")]
        collision_examples = []
        for kl in collisions[:3]:
            pair = [_display(json.loads(raw)) for raw, _o in con.execute(
                f"SELECT raw, MIN(ord) FROM {table} WHERE kl = ? GROUP BY k "
                f"ORDER BY MIN(ord) LIMIT 2", (kl,))]
            if len(pair) == 2:
                collision_examples.append((pair[0], pair[1]))
        return SideSummary(
            file=fmt.name, encoding=fmt.encoding, delimiter=fmt.delimiter,
            rows=reader.rows, keys=tuple(keys), renamed_headers=fmt.renamed,
            ragged_short=reader.short, ragged_long=reader.long,
            blank_keys=one(f"SELECT COUNT(*) FROM {table} WHERE k IS NULL"),
            distinct_keys=one(f"SELECT COUNT(DISTINCT k) FROM {table}"),
            duplicated_keys=one(f"SELECT COUNT(*) FROM (SELECT k FROM {table} "
                                f"WHERE k IS NOT NULL GROUP BY k HAVING COUNT(*) > 1)"),
            rows_matched=matched, rows_unmatched=unmatched,
            unmatched_keys=unmatched_keys, unmatched_examples=examples,
            loose_collisions=len(collisions),
            loose_collision_examples=tuple(collision_examples),
            mangled_ids=self.mangled[side],
            key_samples=tuple(self.samples[side]))

    def report(self) -> MergeReport:
        if self._report is not None:
            return self._report
        con = self.con
        assert con is not None and self._left_reader and self._right_reader
        left = self._side("L", "R", self.left, self._left_reader, self.on, "left")
        right = self._side("R", "L", self.right, self._right_reader,
                           self.right_on, "right")
        inner = self._one("SELECT COUNT(*) FROM L JOIN R ON L.k = R.k")
        rows_out = {"inner": inner,
                    "left": inner + (left.rows - left.rows_matched)}
        rows_out["outer"] = rows_out["left"] + (right.rows - right.rows_matched)
        m2m = list(con.execute(
            "SELECT a.k, a.raw, a.n, b.n FROM "
            "(SELECT k, raw, COUNT(*) AS n, MIN(ord) AS o FROM L WHERE k IS NOT NULL "
            " GROUP BY k HAVING n > 1) AS a JOIN "
            "(SELECT k, COUNT(*) AS n FROM R WHERE k IS NOT NULL GROUP BY k "
            " HAVING n > 1) AS b ON a.k = b.k ORDER BY a.n * b.n DESC, a.o"))
        left_multi = self._one(
            "SELECT COUNT(*) FROM (SELECT k FROM L WHERE k IN (SELECT k FROM R "
            "WHERE k IS NOT NULL) GROUP BY k HAVING COUNT(*) > 1)")
        right_multi = self._one(
            "SELECT COUNT(*) FROM (SELECT k FROM R WHERE k IN (SELECT k FROM L "
            "WHERE k IS NOT NULL) GROUP BY k HAVING COUNT(*) > 1)")
        if m2m:
            relationship = "many-to-many"
        elif right_multi:
            relationship = "one-to-many"
        elif left_multi:
            relationship = "many-to-one"
        else:
            relationship = "one-to-one"
        loose_extra = 0
        loose_examples: List[Tuple[str, str]] = []
        if self.key_match != "loose":
            loose_matched = self._one(
                "SELECT COUNT(*) FROM L WHERE kl IN (SELECT kl FROM R WHERE kl "
                "IS NOT NULL)")
            loose_extra = max(0, loose_matched - left.rows_matched)
            if loose_extra:
                loose_examples = [
                    (_display(json.loads(a)), _display(json.loads(b)))
                    for a, b in con.execute(
                        "SELECT L.raw, R.raw FROM L JOIN R ON L.kl = R.kl WHERE "
                        "L.k IS NOT NULL AND L.k NOT IN (SELECT k FROM R WHERE k "
                        "IS NOT NULL) GROUP BY L.kl ORDER BY MIN(L.ord) LIMIT 3")]
        clashes, columns_out = self._columns()
        self._report = MergeReport(
            left=left, right=right, on=tuple(self.on), right_on=tuple(self.right_on),
            key_match=self.key_match, relationship=relationship,
            many_to_many_keys=len(m2m),
            many_to_many_examples=tuple(
                (_display(json.loads(raw)), nl, nr) for _k, raw, nl, nr in m2m[:3]),
            rows_out=rows_out, loose_extra_matches=loose_extra,
            loose_examples=tuple(loose_examples), clashes=clashes,
            columns_out=columns_out)
        return self._report

    def _columns(self) -> Tuple[Tuple[ColumnClash, ...], Tuple[str, ...]]:
        """The output header, and what happened to each shared column."""
        taken = set(self.left.columns)
        clashes: List[ColumnClash] = []
        renamed: Dict[str, str] = {}
        for name in self.shared:
            n, eg = self.disagree[name]
            if n == 0:
                clashes.append(ColumnClash(name, True, 0, None, (name,)))
                continue
            new = f"{self.right.path.stem}{SEPARATOR}{name}"
            k = 2
            while new in taken:
                new = f"{self.right.path.stem}{SEPARATOR}{name}_{k}"
                k += 1
            taken.add(new)
            renamed[name] = new
            clashes.append(ColumnClash(name, False, n, eg, (name, new)))
        self.right_out = [(i, renamed.get(c, c)) for i, c in enumerate(self.keep)
                          if c not in self.shared or c in renamed]
        self.fill = [(self.left_pos[c], self.keep.index(c)) for c in self.shared
                     if c not in renamed]
        header = tuple(self.left.columns) + tuple(name for _i, name in self.right_out)
        return tuple(clashes), header

    # ---- the output -----------------------------------------------------

    def write(self, out: Path, how: str, on_rows=None) -> int:
        report = self.report()
        con = self.con
        assert con is not None
        blanks = [""] * len(self.right_out)
        written = 0
        with atomic_write(out, "w", newline="", encoding=OUTPUT_ENCODING) as fh:
            w = csv.writer(fh)
            w.writerow(report.columns_out)
            last_key: Optional[str] = None
            last_rows: List[List[str]] = []
            for row in _Reader(self.left, on_rows=on_rows):
                k = _key([row[i] for i in self.left_key], self.key_match)
                if k is None:
                    matches: List[List[str]] = []
                else:
                    if k != last_key:
                        last_key, last_rows = k, self._matches(k)
                    matches = last_rows
                if not matches:
                    if how == "inner":
                        continue
                    w.writerow(list(row) + blanks)
                    written += 1
                    continue
                for other in matches:
                    line = list(row)
                    for lpos, rpos in self.fill:
                        if not line[lpos].strip():
                            line[lpos] = other[rpos]
                    w.writerow(line + [other[i] for i, _n in self.right_out])
                    written += 1
            if how == "outer":
                width = len(self.left.columns)
                for raw, packed in con.execute(
                        "SELECT raw, row FROM R WHERE k IS NULL OR k NOT IN "
                        "(SELECT k FROM L WHERE k IS NOT NULL) ORDER BY ord"):
                    line = [""] * width
                    for pos, value in zip(self.left_key, json.loads(raw)):
                        line[pos] = value
                    other = json.loads(packed)
                    for lpos, rpos in self.fill:
                        line[lpos] = other[rpos]
                    w.writerow(line + [other[i] for i, _n in self.right_out])
                    written += 1
        return written


def _merge_engine(left_csv, right_csv, *, on, right_on, right_cols, key_match,
                  left_delimiter, right_delimiter, left_encoding, right_encoding,
                  tmp_root) -> _MergeEngine:
    if key_match not in KEY_MATCH:
        raise ValueError(f"key_match must be one of {KEY_MATCH}, not {key_match!r}")
    on = _as_names(on)
    if not on:
        raise ValueError("on: name at least one column to match rows by")
    right_on = _as_names(right_on) or list(on)
    if len(right_on) != len(on):
        raise ValueError(f"on names {len(on)} column(s) but right_on names "
                         f"{len(right_on)}; they have to pair up")
    left = detect_format(left_csv, encoding=left_encoding, delimiter=left_delimiter)
    right = detect_format(right_csv, encoding=right_encoding,
                          delimiter=right_delimiter)
    if left.path.resolve() == right.path.resolve():
        raise ValueError("left_csv and right_csv are the same file")
    return _MergeEngine(left, right, on=on, right_on=right_on,
                        right_cols=right_cols, key_match=key_match,
                        tmp_root=tmp_root)


def plan_merge(*, left_csv: PathLike, right_csv: PathLike,
               on: Union[str, Sequence[str]],
               right_on: Union[str, Sequence[str], None] = None,
               key_match: str = "trim",
               right_cols: Optional[Sequence[str]] = None,
               left_delimiter: Optional[str] = None,
               right_delimiter: Optional[str] = None,
               left_encoding: Optional[str] = None,
               right_encoding: Optional[str] = None,
               tmp_root: Optional[PathLike] = None,
               on_progress: Progress = None) -> MergeReport:
    """
    What :func:`merge_spreadsheets` would do, without writing anything.

    Same parameters; see there. Reads both files once each.
    """
    engine = _merge_engine(
        left_csv, right_csv, on=on, right_on=right_on, right_cols=right_cols,
        key_match=key_match, left_delimiter=left_delimiter,
        right_delimiter=right_delimiter, left_encoding=left_encoding,
        right_encoding=right_encoding, tmp_root=tmp_root)
    with engine:
        announce(on_progress, f"reading {engine.right.name}")
        engine.load_right(on_rows=_counting(Ticker(on_progress)))
        announce(on_progress, f"matching {engine.left.name}")
        engine.scan_left(on_rows=_counting(Ticker(on_progress)))
        return engine.report()


@records_settings(
    # which files were read and how their rows were paired. recorded for
    # diagnosis, never compared -- and because the inputs are bound, a
    # feature table merged with its metadata keeps the chain of whatever
    # measured it.
    binding=("left_csv", "right_csv", "on", "right_on", "right_cols",
             "key_match", "allow_many_to_many", "left_delimiter",
             "right_delimiter", "left_encoding", "right_encoding"),
    # which rows survive is what a row *is*, not how anything was measured
    grain=("how",),
    outputs=("out_csv",),
    redo_without_record=True)
def merge_spreadsheets(
    *,
    left_csv: PathLike,
    right_csv: PathLike,
    on: Union[str, Sequence[str]],
    right_on: Union[str, Sequence[str], None] = None,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    how: str = "left",
    key_match: str = "trim",
    right_cols: Optional[Sequence[str]] = None,
    allow_many_to_many: bool = False,
    left_delimiter: Optional[str] = None,
    right_delimiter: Optional[str] = None,
    left_encoding: Optional[str] = None,
    right_encoding: Optional[str] = None,
    tmp_root: Optional[PathLike] = None,
    verbose: bool = True,
    on_progress: Progress = None,
) -> Path:
    """
    Add the columns of one spreadsheet onto the rows of another, matched by key.

    Parameters
    ----------
    left_csv : str or Path
        The file whose rows you are keeping -- the trials, the posts, the
        utterances. Its rows come out in their own order.
    right_csv : str or Path
        The file with the columns to add -- one row per participant, say.
    on : str or sequence of str
        The column(s) that identify the same thing in both files. Several
        names make a composite key (participant and session together).
    right_on : str or sequence of str, optional
        The same key under different names in ``right_csv``, paired with
        ``on`` in order. Omit when the names match.
    out_csv : str or Path, optional
        Where to write. Defaults to ``<left>_with_<right>.csv`` beside the
        left file. Never one of the inputs.
    overwrite_existing : bool, default False
        Rebuild the output if it is already there; otherwise it is returned
        as it is.
    how : {"left", "inner", "outer"}
        ``left`` keeps every left row, with blank added columns where nothing
        matched. ``inner`` keeps only the matched rows. ``outer`` also
        appends the right rows nothing matched, at the end.
    key_match : {"exact", "trim", "loose"}
        How keys are compared; see :func:`normalize_key`. ``trim`` by default.
        ``loose`` is what makes ``007`` match ``7`` and ``P01`` match ``p01``,
        and is worth turning on after :func:`plan_merge` has shown what it
        would add.
    right_cols : sequence of str, optional
        Only these right columns are brought over. All of them by default.
    allow_many_to_many : bool, default False
        A key that repeats in *both* files multiplies rows, and is almost
        always a sign that one more key column is needed. Refused unless
        this is set.
    left_delimiter, right_delimiter : str, optional
        The separators. Sniffed per file when omitted.
    left_encoding, right_encoding : str, optional
        The encodings. Detected per file when omitted: UTF-8 (with or
        without a BOM), UTF-16, then cp1252, then latin-1.
    tmp_root : str or Path, optional
        Where the working database goes while the merge runs. It is about
        the size of the right file and is removed afterwards.
    verbose : bool, default True
        Print the plan's sentences and the row count.
    on_progress : callable, optional
        Progress callback.

    Returns
    -------
    pathlib.Path
        ``out_csv``.

    Notes
    -----
    A non-key column present in both files is kept once when the two agree
    on every matched row (blank on either side counts as agreement, and the
    left's blanks are filled from the right); otherwise the right copy is
    renamed ``<right stem>__<column>``. A blank key never matches anything.
    Rows with no match are counted and the first few are named, so a key
    typed as ``P001`` on one side and ``1`` on the other is found before the
    analysis, not after.
    """
    if how not in HOW:
        raise ValueError(f"how must be one of {HOW}, not {how!r}")
    engine = _merge_engine(
        left_csv, right_csv, on=on, right_on=right_on, right_cols=right_cols,
        key_match=key_match, left_delimiter=left_delimiter,
        right_delimiter=right_delimiter, left_encoding=left_encoding,
        right_encoding=right_encoding, tmp_root=tmp_root)
    left, right = engine.left, engine.right
    out = Path(out_csv) if out_csv is not None else \
        left.path.with_name(f"{left.path.stem}_with_{right.path.stem}.csv")
    _check_output(out, [left.path, right.path])
    if out.is_file() and not overwrite_existing:
        if verbose:
            print(f"[reshape] {out} exists; leaving it (overwrite_existing=False)")
        return out
    with engine:
        announce(on_progress, f"reading {right.name}")
        engine.load_right(on_rows=_counting(Ticker(on_progress)))
        announce(on_progress, f"matching {left.name}")
        engine.scan_left(on_rows=_counting(Ticker(on_progress)))
        report = engine.report()
        why = report.refusal(allow_many_to_many=allow_many_to_many)
        if why:
            raise ValueError(why)
        if verbose:
            for line in report.sentences(how):
                print(f"[reshape] {line}")
            if not report.matched:
                print("[reshape] WARNING: nothing matched; check that the "
                      "keys are written the same way in both files")
        announce(on_progress, f"writing {out.name}")
        ticker = Ticker(on_progress, total=report.left.rows)
        written = engine.write(out, how, on_rows=_counting(ticker))
    expected = report.rows_out[how]
    if written != expected:          # the plan and the write share one store
        raise RuntimeError(f"merge wrote {written} rows but planned {expected}")
    if verbose:
        print(f"[reshape] {left.name} + {right.name} -> {out} ({written:,} rows, "
              f"{how})")
    return out


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackFile:
    file: str
    encoding: str
    delimiter: str
    rows: int
    columns: Tuple[str, ...]
    #: Columns no other file in the stack has.
    only_here: Tuple[str, ...]
    ragged_short: int = 0
    ragged_long: int = 0


@dataclass(frozen=True)
class StackReport:
    files: Tuple[StackFile, ...]
    columns_out: Tuple[str, ...]
    #: Groups of column names that differ only by case or spacing. Merged
    #: into one column under ``match_columns="loose"``; kept apart otherwise.
    near_duplicates: Tuple[Tuple[str, ...], ...]
    #: Files sharing no column with the rest: stacked, they are a checkerboard.
    disjoint_files: Tuple[str, ...]
    #: What the column naming each row's file is called, or None.
    source_col: Optional[str]
    duplicates_dropped: int
    match_columns: str

    @property
    def rows_out(self) -> int:
        return sum(f.rows for f in self.files)

    def sentences(self) -> List[str]:
        out = [f"{len(self.files)} files, {self.rows_out:,} rows, "
               f"{len(self.columns_out)} columns"
               + (f" plus {self.source_col}." if self.source_col else ".")]
        if self.near_duplicates and self.match_columns != "loose":
            shown = "; ".join(" / ".join(g) for g in self.near_duplicates[:3])
            out.append(f"Some column names differ only in capitals or spacing "
                       f"({shown}); they are kept as separate columns.")
        for f in self.files:
            if f.only_here:
                out.append(f"Only {f.file} has {', '.join(f.only_here[:5])}"
                           + (f" and {len(f.only_here) - 5} more" if len(f.only_here) > 5 else "")
                           + "; the other files get blanks there.")
        for name in self.disjoint_files:
            out.append(f"{name} shares no column with the other files, so "
                       f"stacking it gives a checkerboard of blanks. Did you "
                       f"mean to match it up by an ID (merge) instead?")
        if self.duplicates_dropped:
            out.append(f"{self.duplicates_dropped} file(s) were listed twice "
                       f"and are stacked once.")
        return out

    def to_dict(self) -> dict:
        return asdict(self)


def _loose_name(name: str) -> str:
    return _INNER_SPACE.sub(" ", name).strip().casefold()


class _Stack:
    """The formats, counts and column plan of a stack; shared by plan and write."""

    def __init__(self, csv_paths, *, source_col, match_columns, delimiter,
                 encoding, on_rows=None):
        if match_columns not in ("exact", "loose"):
            raise ValueError("match_columns must be 'exact' or 'loose', not "
                             f"{match_columns!r}")
        paths: List[Path] = []
        seen = set()
        dropped = 0
        listed = [csv_paths] if isinstance(csv_paths, (str, Path)) else list(csv_paths)
        for p in listed:
            r = Path(p).resolve()
            if r in seen:
                dropped += 1
                continue
            seen.add(r)
            paths.append(Path(p))
        if len(paths) < 2:
            raise ValueError("stacking needs at least two different files")
        self.formats = [detect_format(p, encoding=encoding, delimiter=delimiter)
                        for p in paths]
        self.readers = [_Reader(f, on_rows=on_rows) for f in self.formats]
        self.match = match_columns
        self.dropped = dropped
        # the output columns: first-seen spelling, in first-seen order
        canon = (lambda n: n) if match_columns == "exact" else _loose_name
        self.columns: List[str] = []
        self.place: Dict[str, int] = {}
        for fmt in self.formats:
            for name in fmt.columns:
                key = canon(name)
                if key not in self.place:
                    self.place[key] = len(self.columns)
                    self.columns.append(name)
        self.canon = canon
        # names that only the loose rule would fold together, reported either way
        groups: Dict[str, List[str]] = {}
        for fmt in self.formats:
            for name in fmt.columns:
                lst = groups.setdefault(_loose_name(name), [])
                if name not in lst:
                    lst.append(name)
        self.near = tuple(tuple(g) for g in groups.values() if len(g) > 1)
        self.source_col: Optional[str] = None
        if source_col:
            name = source_col
            if name in self.columns:
                name = "source_file"
                k = 2
                while name in self.columns:
                    name = f"source_file_{k}"
                    k += 1
            self.source_col = name
        # what to call each file in that column: the stem, unless two files
        # share one, and then the path under their common folder
        stems = [f.path.stem for f in self.formats]
        if len(set(stems)) == len(stems):
            self.labels = stems
        else:
            common = Path(os.path.commonpath([str(f.path.resolve().parent)
                                              for f in self.formats]))
            self.labels = [str(f.path.resolve().relative_to(common).with_suffix("")
                               ).replace(os.sep, "/") for f in self.formats]

    def count(self) -> None:
        for reader in self.readers:
            for _row in reader:
                pass

    def report(self) -> StackReport:
        files = []
        for fmt, reader in zip(self.formats, self.readers):
            others = set()
            for other in self.formats:
                if other is not fmt:
                    others |= {self.canon(c) for c in other.columns}
            only = tuple(c for c in fmt.columns if self.canon(c) not in others)
            files.append(StackFile(
                file=fmt.name, encoding=fmt.encoding, delimiter=fmt.delimiter,
                rows=reader.rows, columns=fmt.columns, only_here=only,
                ragged_short=reader.short, ragged_long=reader.long))
        disjoint = tuple(f.file for f, fmt in zip(files, self.formats)
                         if not ({self.canon(c) for c in fmt.columns}
                                 - {self.canon(c) for c in f.only_here}))
        return StackReport(files=tuple(files), columns_out=tuple(self.columns),
                           near_duplicates=self.near, disjoint_files=disjoint,
                           source_col=self.source_col,
                           duplicates_dropped=self.dropped,
                           match_columns=self.match)


def plan_stack(*, csv_paths: Sequence[PathLike], source_col: str = "source",
               match_columns: str = "exact", delimiter: Optional[str] = None,
               encoding: Optional[str] = None, count_rows: bool = True,
               on_progress: Progress = None) -> StackReport:
    """
    What :func:`stack_spreadsheets` would do. Reads each file once.

    ``count_rows=False`` skips the pass and reports zero rows per file: the
    column plan alone, from the headers, for a screen that wants to settle
    how columns match before it reads anything.
    """
    stack = _Stack(csv_paths, source_col=source_col, match_columns=match_columns,
                   delimiter=delimiter, encoding=encoding)
    if count_rows:
        announce(on_progress, "reading the files")
        ticker = Ticker(on_progress)
        for reader in stack.readers:
            reader.on_rows = _counting(ticker)
        stack.count()
    return stack.report()


@records_settings(
    binding=("csv_paths", "match_columns", "delimiter", "encoding"),
    grain=("source_col",),
    outputs=("out_csv",),
    redo_without_record=True)
def stack_spreadsheets(
    *,
    csv_paths: Sequence[PathLike],
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    source_col: str = "source",
    match_columns: str = "exact",
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    verbose: bool = True,
    on_progress: Progress = None,
) -> Path:
    """
    Put the rows of several spreadsheets one under another.

    Parameters
    ----------
    csv_paths : sequence of str or Path
        The files, in the order their rows should come out. At least two.
        A file listed twice is stacked once.
    out_csv : str or Path, optional
        Where to write. Defaults to ``<first stem>_stacked.csv`` beside the
        first file.
    overwrite_existing : bool, default False
        Rebuild the output if it is already there.
    source_col : str, default "source"
        A first column naming the file each row came from (the file's stem,
        or its path under the files' common folder when stems repeat). An
        empty string leaves it out. If the name is taken, ``source_file``.
    match_columns : {"exact", "loose"}
        Whether ``Age`` and ``age`` are the same column. Surrounding spaces
        never count either way.
    delimiter, encoding : str, optional
        Applied to every file; detected per file when omitted.
    verbose : bool, default True
        Print what was stacked.
    on_progress : callable, optional
        Progress callback.

    Returns
    -------
    pathlib.Path
        ``out_csv``.

    Notes
    -----
    Columns are the union, in order of first appearance, and a cell a file
    does not have is blank. A file sharing no column with the others is
    stacked anyway, and the plan says so: that is usually a merge wanted
    by mistake.
    """
    stack = _Stack(csv_paths, source_col=source_col, match_columns=match_columns,
                   delimiter=delimiter, encoding=encoding)
    first = stack.formats[0].path
    out = Path(out_csv) if out_csv is not None else \
        first.with_name(f"{first.stem}_stacked.csv")
    _check_output(out, [f.path for f in stack.formats])
    if out.is_file() and not overwrite_existing:
        if verbose:
            print(f"[reshape] {out} exists; leaving it (overwrite_existing=False)")
        return out
    header = ([stack.source_col] if stack.source_col else []) + stack.columns
    width = len(stack.columns)
    ticker = Ticker(on_progress, total=len(stack.formats))
    written = 0
    with atomic_write(out, "w", newline="", encoding=OUTPUT_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for fmt, reader, label in zip(stack.formats, stack.readers, stack.labels):
            places = [stack.place[stack.canon(c)] for c in fmt.columns]
            for row in reader:
                line = [""] * width
                for value, at in zip(row, places):
                    line[at] = value
                w.writerow(([label] if stack.source_col else []) + line)
                written += 1
            ticker.tick(1, message=fmt.name)
    if verbose:
        for line in stack.report().sentences():
            print(f"[reshape] {line}")
        print(f"[reshape] stacked {len(stack.formats)} files -> {out} "
              f"({written:,} rows)")
    return out


# ---------------------------------------------------------------------------
# Wide to long
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnpivotReport:
    file: str
    encoding: str
    delimiter: str
    rows_in: int
    rows_out: int
    value_cols: Tuple[str, ...]
    #: What each value column is called in the names column (after any prefix strip).
    names: Tuple[str, ...]
    id_cols: Tuple[str, ...]
    columns_out: Tuple[str, ...]
    #: Value columns holding text when others hold numbers.
    mixed_kinds: Tuple[str, ...]
    blanks: int
    dropped: int
    names_to: str
    values_to: str

    def sentences(self) -> List[str]:
        out = [f"{self.rows_in:,} rows × {len(self.value_cols)} columns become "
               f"{self.rows_out:,} rows, with {self.names_to} saying which "
               f"column a value came from and {self.values_to} holding it."]
        if self.mixed_kinds:
            out.append(f"{', '.join(self.mixed_kinds)} hold text while the "
                       f"other columns hold numbers; they share one column, "
                       f"which R and Excel will read as text.")
        if self.blanks:
            out.append(f"{self.blanks:,} of those cells are empty"
                       + ("; those rows are left out." if self.dropped
                          else "; they come out as rows with an empty value."))
        return out

    def to_dict(self) -> dict:
        return asdict(self)


class _Unpivot:
    def __init__(self, csv_path, *, value_cols, id_cols, names_to, values_to,
                 strip_prefix, drop_blank, delimiter, encoding, on_rows=None):
        self.fmt = detect_format(csv_path, encoding=encoding, delimiter=delimiter)
        fmt = self.fmt
        self.value_cols = _as_names(value_cols)
        if len(self.value_cols) < 2:
            raise ValueError("value_cols: name at least two columns to turn "
                             "into rows")
        self.value_pos = _positions(fmt, self.value_cols, what="value_cols")
        if id_cols is None:
            # everything that is not being unpivoted rides along, the way
            # tidyr does it; an explicit list narrows that
            self.id_cols = [c for c in fmt.columns if c not in self.value_cols]
        else:
            self.id_cols = [c for c in _as_names(id_cols) if c not in self.value_cols]
        self.id_pos = _positions(fmt, self.id_cols, what="id_cols")
        self.row_col: Optional[str] = None
        if not self.id_cols:
            name = "row"
            k = 2
            while name in fmt.columns or name in (names_to, values_to):
                name = f"row_{k}"
                k += 1
            self.row_col = name
        names_to, values_to = str(names_to).strip(), str(values_to).strip()
        if not names_to or not values_to:
            raise ValueError("names_to and values_to both need a name")
        if names_to == values_to:
            raise ValueError(f"names_to and values_to are both {names_to!r}; "
                             f"they have to differ")
        for new in (names_to, values_to):
            if new in self.id_cols or new == self.row_col:
                raise ValueError(f"there is already a column called {new!r}; "
                                 f"pick another name")
        self.names_to, self.values_to = names_to, values_to
        self.names = list(self.value_cols)
        if strip_prefix:
            bad = [c for c in self.value_cols if not c.startswith(strip_prefix)]
            if bad:
                raise ValueError(f"strip_prefix {strip_prefix!r}: "
                                 f"{', '.join(bad)} do not start with it")
            self.names = [c[len(strip_prefix):] or c for c in self.value_cols]
        self.drop_blank = bool(drop_blank)
        self.reader = _Reader(fmt, on_rows=on_rows)
        self.blanks = 0
        self.numeric = [True] * len(self.value_cols)
        self.filled = [0] * len(self.value_cols)

    def scan(self) -> None:
        self.blanks = 0
        self.numeric = [True] * len(self.value_cols)
        self.filled = [0] * len(self.value_cols)
        for row in self.reader:
            for j, pos in enumerate(self.value_pos):
                v = row[pos].strip()
                if not v:
                    self.blanks += 1
                    continue
                self.filled[j] += 1
                if self.numeric[j] and _canonical_number(v) is None:
                    try:
                        float(v)
                    except ValueError:
                        self.numeric[j] = False

    def report(self) -> UnpivotReport:
        rows_out = self.reader.rows * len(self.value_cols)
        dropped = self.blanks if self.drop_blank else 0
        numbers = [c for c, n, f in zip(self.value_cols, self.numeric, self.filled)
                   if n and f]
        words = [c for c, n, f in zip(self.value_cols, self.numeric, self.filled)
                 if not n and f]
        columns = ([self.row_col] if self.row_col else list(self.id_cols)) \
            + [self.names_to, self.values_to]
        return UnpivotReport(
            file=self.fmt.name, encoding=self.fmt.encoding,
            delimiter=self.fmt.delimiter, rows_in=self.reader.rows,
            rows_out=rows_out - dropped, value_cols=tuple(self.value_cols),
            names=tuple(self.names), id_cols=tuple(self.id_cols),
            columns_out=tuple(columns),
            mixed_kinds=tuple(words) if numbers and words else (),
            blanks=self.blanks, dropped=dropped,
            names_to=self.names_to, values_to=self.values_to)


def plan_wide_to_long(*, csv_path: PathLike, value_cols: Sequence[str],
                      id_cols: Optional[Sequence[str]] = None,
                      names_to: str = "variable", values_to: str = "value",
                      strip_prefix: str = "", drop_blank: bool = False,
                      delimiter: Optional[str] = None,
                      encoding: Optional[str] = None,
                      on_progress: Progress = None) -> UnpivotReport:
    """What :func:`wide_to_long` would do. Reads the file once."""
    job = _Unpivot(csv_path, value_cols=value_cols, id_cols=id_cols,
                   names_to=names_to, values_to=values_to,
                   strip_prefix=strip_prefix, drop_blank=drop_blank,
                   delimiter=delimiter, encoding=encoding)
    announce(on_progress, f"reading {job.fmt.name}")
    job.reader.on_rows = _counting(Ticker(on_progress))
    job.scan()
    return job.report()


@records_settings(
    binding=("csv_path", "value_cols", "id_cols", "strip_prefix", "drop_blank",
             "delimiter", "encoding"),
    grain=("names_to", "values_to"),
    outputs=("out_csv",),
    redo_without_record=True)
def wide_to_long(
    *,
    csv_path: PathLike,
    value_cols: Sequence[str],
    id_cols: Optional[Sequence[str]] = None,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    names_to: str = "variable",
    values_to: str = "value",
    strip_prefix: str = "",
    drop_blank: bool = False,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    verbose: bool = True,
    on_progress: Progress = None,
) -> Path:
    """
    Turn several columns into rows: one row per (original row, column).

    Parameters
    ----------
    csv_path : str or Path
        The wide spreadsheet.
    value_cols : sequence of str
        The columns that hold the same kind of measurement at different
        times or in different conditions -- ``score_t1, score_t2,
        score_t3``. Each becomes its own row. At least two.
    id_cols : sequence of str, optional
        The columns that ride along unchanged on every one of those rows.
        By default every column not in ``value_cols``. If none are left, a
        ``row`` column numbers the original rows so they can be traced.
    out_csv : str or Path, optional
        Where to write. Defaults to ``<stem>_long.csv`` beside the input.
    overwrite_existing : bool, default False
        Rebuild the output if it is already there.
    names_to : str, default "variable"
        The new column saying which of ``value_cols`` a row came from.
    values_to : str, default "value"
        The new column holding the value.
    strip_prefix : str, default ""
        Removed from the front of every value column's name in ``names_to``
        (``score_t1`` becomes ``1``). Every value column has to start with
        it.
    drop_blank : bool, default False
        Leave out the rows whose value is empty. Kept by default, so a
        balanced design stays visibly balanced.
    delimiter, encoding : str, optional
        Detected when omitted.
    verbose : bool, default True
        Print what was done.
    on_progress : callable, optional
        Progress callback.

    Returns
    -------
    pathlib.Path
        ``out_csv``.

    Notes
    -----
    Values are copied as they are. When some value columns hold numbers and
    others hold text, the plan says so, because the one value column will
    then read as text everywhere.
    """
    job = _Unpivot(csv_path, value_cols=value_cols, id_cols=id_cols,
                   names_to=names_to, values_to=values_to,
                   strip_prefix=strip_prefix, drop_blank=drop_blank,
                   delimiter=delimiter, encoding=encoding)
    src = job.fmt.path
    out = Path(out_csv) if out_csv is not None else \
        src.with_name(f"{src.stem}_long.csv")
    _check_output(out, [src])
    if out.is_file() and not overwrite_existing:
        if verbose:
            print(f"[reshape] {out} exists; leaving it (overwrite_existing=False)")
        return out
    announce(on_progress, f"reading {src.name}")
    job.scan()
    report = job.report()
    announce(on_progress, f"writing {out.name}")
    ticker = Ticker(on_progress, total=job.reader.rows)
    written = 0
    with atomic_write(out, "w", newline="", encoding=OUTPUT_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(report.columns_out)
        for n, row in enumerate(_Reader(job.fmt), 1):
            head = [str(n)] if job.row_col else [row[i] for i in job.id_pos]
            for name, pos in zip(job.names, job.value_pos):
                value = row[pos]
                if job.drop_blank and not value.strip():
                    continue
                w.writerow(head + [name, value])
                written += 1
            if n % _TICK_EVERY == 0:
                ticker.tick(_TICK_EVERY)
    if written != report.rows_out:
        raise RuntimeError(f"wide_to_long wrote {written} rows but planned "
                           f"{report.rows_out}")
    if verbose:
        for line in report.sentences():
            print(f"[reshape] {line}")
        print(f"[reshape] {src.name} -> {out} ({written:,} rows)")
    return out


# ---------------------------------------------------------------------------
# Long to wide
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PivotReport:
    file: str
    encoding: str
    delimiter: str
    rows_in: int
    rows_out: int
    id_cols: Tuple[str, ...]
    names_from: str
    values_from: Tuple[str, ...]
    #: The distinct names, in the order their columns come out.
    new_columns: Tuple[str, ...]
    columns_out: Tuple[str, ...]
    blank_names: int
    blank_keys: int
    #: (key, name) pairs that occur more than once.
    collisions: int
    collision_examples: Tuple[Tuple[str, str, Tuple[str, ...]], ...]
    #: Whether every colliding value is a number, so averaging is possible.
    collisions_numeric: bool
    #: Carry columns that vary within a key: (column, key, one value, another).
    varying_carry: Tuple[Tuple[str, str, str, str], ...]
    too_many_columns: bool
    max_columns: int
    on_duplicate: str

    def refusal(self) -> Optional[str]:
        if self.too_many_columns:
            return (f"{self.names_from} has {len(self.new_columns):,} different "
                    f"values, which would make {len(self.new_columns):,} "
                    f"columns (the limit is {self.max_columns:,}). Usually that "
                    f"means an ID or a timestamp was picked by mistake.")
        if self.varying_carry:
            col, key, a, b = self.varying_carry[0]
            return (f"{col} has more than one value for {key} ({a!r} and "
                    f"{b!r}), so it cannot be brought along unchanged. Either "
                    f"leave it out or add the column that tells those rows "
                    f"apart to the key.")
        if self.collisions and self.on_duplicate == "refuse":
            k, name, values = self.collision_examples[0]
            return (f"{k} has {len(values)} rows for {self.names_from} = {name} "
                    f"(values {', '.join(repr(v) for v in values)}); "
                    f"{self.collisions:,} such pairs. Add a column that tells "
                    f"them apart to the key, or say what to do with them: "
                    f"on_duplicate='first', 'last'"
                    + (" or 'mean'" if self.collisions_numeric else "") + ".")
        if self.collisions and self.on_duplicate == "mean" \
                and not self.collisions_numeric:
            return ("on_duplicate='mean' needs numbers, and some of the "
                    "repeated values are not; use 'first' or 'last'.")
        return None

    def sentences(self) -> List[str]:
        shown = ", ".join(self.new_columns[:6]) + \
            (f" and {len(self.new_columns) - 6} more" if len(self.new_columns) > 6 else "")
        out = [f"{self.rows_in:,} rows become {self.rows_out:,}, one per "
               f"{_display(self.id_cols)}, with {len(self.new_columns):,} new "
               f"column(s): {shown}."]
        if self.blank_names:
            out.append(f"{self.blank_names:,} rows have no {self.names_from} "
                       f"and are left out.")
        if self.blank_keys:
            out.append(f"{self.blank_keys:,} rows have no {_display(self.id_cols)} "
                       f"and are left out.")
        if self.collisions:
            k, name, values = self.collision_examples[0]
            what = {"refuse": "that has to be resolved first",
                    "first": "the first one is kept",
                    "last": "the last one is kept",
                    "mean": "they are averaged"}[self.on_duplicate]
            out.append(f"{k} has {len(values)} rows for {self.names_from} = "
                       f"{name} (values {', '.join(repr(v) for v in values)}); "
                       f"{self.collisions:,} such pairs, and {what}.")
        return out

    def to_dict(self) -> dict:
        return asdict(self)


class _Pivot:
    """The pivot's working store; shared by the plan and the writer."""

    def __init__(self, csv_path, *, id_cols, names_from, values_from, carry_cols,
                 on_duplicate, max_columns, key_match, delimiter, encoding,
                 tmp_root):
        if on_duplicate not in ON_DUPLICATE:
            raise ValueError(f"on_duplicate must be one of {ON_DUPLICATE}, "
                             f"not {on_duplicate!r}")
        if key_match not in KEY_MATCH:
            raise ValueError(f"key_match must be one of {KEY_MATCH}, not "
                             f"{key_match!r}")
        self.fmt = detect_format(csv_path, encoding=encoding, delimiter=delimiter)
        fmt = self.fmt
        self.id_cols = _as_names(id_cols)
        if not self.id_cols:
            raise ValueError("id_cols: name at least one column that says "
                             "which rows belong together")
        names_from = str(names_from).strip()
        self.values_from = _as_names(values_from)
        if not self.values_from:
            raise ValueError("values_from: name the column holding the values")
        self.carry = [c for c in _as_names(carry_cols) if c not in self.id_cols]
        for name, what in ((names_from, "names_from"), *((v, "values_from") for v in self.values_from)):
            if name in self.id_cols:
                raise ValueError(f"{name!r} is in id_cols and {what}; a column "
                                 f"cannot be both")
        if names_from in self.values_from:
            raise ValueError(f"{names_from!r} is names_from and values_from; "
                             f"a column cannot be both")
        for c in self.carry:
            if c == names_from or c in self.values_from:
                raise ValueError(f"{c!r} is being pivoted and cannot also be "
                                 f"brought along")
        self.names_from = names_from
        self.id_pos = _positions(fmt, self.id_cols, what="id_cols")
        self.name_pos = _positions(fmt, [names_from], what="names_from")[0]
        self.value_pos = _positions(fmt, self.values_from, what="values_from")
        self.carry_pos = _positions(fmt, self.carry, what="carry_cols")
        self.on_duplicate = on_duplicate
        self.max_columns = int(max_columns)
        self.key_match = key_match
        self.tmp_root = tmp_root
        self.con: Optional[sqlite3.Connection] = None
        self.folder: Optional[Path] = None
        self.reader = _Reader(fmt)
        self.blank_names = self.blank_keys = 0
        self.varying: Dict[str, Tuple[str, str, str]] = {}
        self._report: Optional[PivotReport] = None

    def __enter__(self) -> "_Pivot":
        self.con, self.folder = _sqlite(self.tmp_root)
        self.con.execute("CREATE TABLE cells (k TEXT, name TEXT, ord INTEGER, "
                         "vals TEXT)")
        self.con.execute("CREATE TABLE keys (k TEXT PRIMARY KEY, first INTEGER, "
                         "raw TEXT, carry TEXT)")
        return self

    def __exit__(self, *_exc) -> None:
        _discard(self.con, self.folder)
        self.con = self.folder = None

    def load(self, on_rows=None) -> None:
        con = self.con
        assert con is not None
        self.reader.on_rows = on_rows
        batch = []
        last_key: Optional[str] = None
        last_carry: Optional[List[str]] = None
        with con:
            for ord_, row in enumerate(self.reader, 1):
                k = _key([row[i] for i in self.id_pos], self.key_match)
                if k is None:
                    self.blank_keys += 1
                    continue
                name = row[self.name_pos].strip()
                if not name:
                    self.blank_names += 1
                    continue
                carry = [row[i] for i in self.carry_pos]
                if k != last_key:
                    found = con.execute("SELECT carry FROM keys WHERE k = ?",
                                        (k,)).fetchone()
                    if found is None:
                        con.execute("INSERT INTO keys VALUES (?, ?, ?, ?)",
                                    (k, ord_, json.dumps([row[i] for i in self.id_pos]),
                                     json.dumps(carry)))
                        last_carry = carry
                    else:
                        last_carry = json.loads(found[0])
                    last_key = k
                # a brought-along column has to say one thing per key; a
                # blank defers to whatever another row says
                if self.carry and last_carry is not None:
                    changed = False
                    for j, (col, mine) in enumerate(zip(self.carry, carry)):
                        theirs = last_carry[j]
                        if not mine.strip():
                            continue
                        if not theirs.strip():
                            last_carry[j] = mine
                            changed = True
                        elif not _same_cell(mine, theirs) and col not in self.varying:
                            self.varying[col] = (_display([row[i] for i in self.id_pos]),
                                                 theirs, mine)
                    if changed:
                        con.execute("UPDATE keys SET carry = ? WHERE k = ?",
                                    (json.dumps(last_carry), k))
                batch.append((k, name, ord_, json.dumps([row[i] for i in self.value_pos])))
                if len(batch) >= _BATCH:
                    con.executemany("INSERT INTO cells VALUES (?, ?, ?, ?)", batch)
                    batch.clear()
            if batch:
                con.executemany("INSERT INTO cells VALUES (?, ?, ?, ?)", batch)
            con.execute("CREATE INDEX cells_k ON cells (k, ord)")
            con.execute("CREATE INDEX cells_kn ON cells (k, name)")

    def _names(self) -> List[str]:
        con = self.con
        assert con is not None
        names = [n for (n,) in con.execute(
            "SELECT name FROM cells GROUP BY name ORDER BY MIN(ord)")]
        if names and all(_DECIMAL.match(n) for n in names):
            names.sort(key=lambda n: Decimal(n))
        return names

    def report(self) -> PivotReport:
        if self._report is not None:
            return self._report
        con = self.con
        assert con is not None
        names = self._names()
        rows_out = con.execute("SELECT COUNT(*) FROM keys").fetchone()[0]
        collisions = list(con.execute(
            "SELECT k, name, COUNT(*) FROM cells GROUP BY k, name HAVING COUNT(*) > 1 "
            "ORDER BY MIN(ord)"))
        examples = []
        numeric = True
        for k, name, _n in collisions[:_EXAMPLES]:
            raw = con.execute("SELECT raw FROM keys WHERE k = ?", (k,)).fetchone()[0]
            vals = [json.loads(v)[0] for (v,) in con.execute(
                "SELECT vals FROM cells WHERE k = ? AND name = ? ORDER BY ord",
                (k, name))]
            examples.append((_display(json.loads(raw)), name, tuple(vals)))
        if collisions:
            for (v,) in con.execute(
                    "SELECT vals FROM cells WHERE (k, name) IN (SELECT k, name FROM "
                    "cells GROUP BY k, name HAVING COUNT(*) > 1)"):
                for cell in json.loads(v):
                    if _number(cell) is None:
                        numeric = False
                        break
                if not numeric:
                    break
        self.generated = self._generated(names)
        columns = list(self.id_cols) + list(self.carry) + \
            [g for _name, g in self.generated]
        self._report = PivotReport(
            file=self.fmt.name, encoding=self.fmt.encoding,
            delimiter=self.fmt.delimiter, rows_in=self.reader.rows,
            rows_out=rows_out, id_cols=tuple(self.id_cols),
            names_from=self.names_from, values_from=tuple(self.values_from),
            new_columns=tuple(names), columns_out=tuple(columns),
            blank_names=self.blank_names, blank_keys=self.blank_keys,
            collisions=len(collisions), collision_examples=tuple(examples),
            collisions_numeric=numeric,
            varying_carry=tuple((c, *v) for c, v in self.varying.items()),
            too_many_columns=len(names) > self.max_columns,
            max_columns=self.max_columns, on_duplicate=self.on_duplicate)
        return self._report

    def _generated(self, names: Sequence[str]) -> List[Tuple[Tuple[str, int], str]]:
        """``((name, value index), output column)`` for every generated column."""
        taken = set(self.id_cols) | set(self.carry)
        if len(self.values_from) == 1:
            bare = list(names)
            if len(set(bare)) == len(bare) and not (set(bare) & taken):
                return [((n, 0), n) for n in names]
        # several value columns, or a bare name would collide: prefix all of
        # them the same way rather than some
        out = []
        for j, vc in enumerate(self.values_from):
            for n in names:
                out.append(((n, j), f"{vc}_{n}"))
        return out

    def write(self, out: Path, on_rows=None) -> int:
        report = self.report()
        con = self.con
        assert con is not None
        slots = {key: i for i, (key, _g) in enumerate(self.generated)}
        width = len(self.generated)
        written = 0
        with atomic_write(out, "w", newline="", encoding=OUTPUT_ENCODING) as fh:
            w = csv.writer(fh)
            w.writerow(report.columns_out)
            for k, raw, carry in con.execute(
                    "SELECT k, raw, carry FROM keys ORDER BY first"):
                line = [""] * width
                seen: Dict[Tuple[str, int], List[str]] = {}
                for name, vals in con.execute(
                        "SELECT name, vals FROM cells WHERE k = ? ORDER BY ord", (k,)):
                    for j, v in enumerate(json.loads(vals)):
                        seen.setdefault((name, j), []).append(v)
                for key, values in seen.items():
                    at = slots[key]
                    if len(values) == 1 or self.on_duplicate == "first":
                        line[at] = values[0]
                    elif self.on_duplicate == "last":
                        line[at] = values[-1]
                    else:                      # mean, numeric by the plan's check
                        nums = [_number(v) for v in values]
                        line[at] = f"{sum(nums) / len(nums):.10g}"
                w.writerow(json.loads(raw) + json.loads(carry) + line)
                written += 1
                if on_rows is not None and written % _TICK_EVERY == 0:
                    on_rows(written)
        return written


def _number(text: str) -> Optional[float]:
    """A cell as a number for averaging, or None. Finite only."""
    try:
        x = float(str(text).strip())
    except ValueError:
        return None
    return x if x == x and x not in (float("inf"), float("-inf")) else None


def plan_long_to_wide(*, csv_path: PathLike, id_cols: Sequence[str],
                      names_from: str, values_from: Union[str, Sequence[str]],
                      carry_cols: Optional[Sequence[str]] = None,
                      on_duplicate: str = "refuse", max_columns: int = 1000,
                      key_match: str = "trim", delimiter: Optional[str] = None,
                      encoding: Optional[str] = None,
                      tmp_root: Optional[PathLike] = None,
                      on_progress: Progress = None) -> PivotReport:
    """What :func:`long_to_wide` would do. Reads the file once."""
    job = _Pivot(csv_path, id_cols=id_cols, names_from=names_from,
                 values_from=values_from, carry_cols=carry_cols,
                 on_duplicate=on_duplicate, max_columns=max_columns,
                 key_match=key_match, delimiter=delimiter, encoding=encoding,
                 tmp_root=tmp_root)
    with job:
        announce(on_progress, f"reading {job.fmt.name}")
        job.load(on_rows=_counting(Ticker(on_progress)))
        return job.report()


@records_settings(
    binding=("csv_path", "names_from", "values_from", "carry_cols",
             "on_duplicate", "max_columns", "key_match", "delimiter", "encoding"),
    grain=("id_cols",),
    outputs=("out_csv",),
    redo_without_record=True)
def long_to_wide(
    *,
    csv_path: PathLike,
    id_cols: Sequence[str],
    names_from: str,
    values_from: Union[str, Sequence[str]],
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    carry_cols: Optional[Sequence[str]] = None,
    on_duplicate: str = "refuse",
    max_columns: int = 1000,
    key_match: str = "trim",
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    tmp_root: Optional[PathLike] = None,
    verbose: bool = True,
    on_progress: Progress = None,
) -> Path:
    """
    Turn rows into columns: one row per key, one column per name.

    Parameters
    ----------
    csv_path : str or Path
        The long spreadsheet.
    id_cols : sequence of str
        The column(s) that say which rows belong together -- one output row
        per distinct combination, in order of first appearance.
    names_from : str
        The column whose values become column names: a condition, a time
        point, a question number.
    values_from : str or sequence of str
        The column holding the values that go into them. Several columns
        give ``<column>_<name>`` for each.
    out_csv : str or Path, optional
        Where to write. Defaults to ``<stem>_wide.csv`` beside the input.
    overwrite_existing : bool, default False
        Rebuild the output if it is already there.
    carry_cols : sequence of str, optional
        Other columns to bring along. Each has to hold one value per key (a
        blank defers to the other rows); one that varies is refused, because
        in a pivot that means the key is missing a column.
    on_duplicate : {"refuse", "first", "last", "mean"}
        What to do when one key has the same name on several rows. Refused
        by default, with examples, since that nearly always means a missing
        key column. ``mean`` needs every repeated value to be a number.
    max_columns : int, default 1000
        Refuse to make more generated columns than this. An ID or timestamp
        picked as ``names_from`` is the usual way past it.
    key_match : {"exact", "trim", "loose"}
        How the key values are compared; see :func:`normalize_key`.
    delimiter, encoding : str, optional
        Detected when omitted.
    tmp_root : str or Path, optional
        Where the working database goes meanwhile; removed afterwards.
    verbose : bool, default True
        Print what was done.
    on_progress : callable, optional
        Progress callback.

    Returns
    -------
    pathlib.Path
        ``out_csv``.

    Notes
    -----
    Generated columns are bare names (``A``, ``B``) when there is one value
    column and none would collide with an existing column; otherwise every
    one is ``<values_from>_<name>``, so the naming is consistent rather than
    piecemeal. Names that are all numbers come out in numeric order. A row
    with a blank key or a blank name is left out and counted. Missing
    combinations are blank.
    """
    job = _Pivot(csv_path, id_cols=id_cols, names_from=names_from,
                 values_from=values_from, carry_cols=carry_cols,
                 on_duplicate=on_duplicate, max_columns=max_columns,
                 key_match=key_match, delimiter=delimiter, encoding=encoding,
                 tmp_root=tmp_root)
    src = job.fmt.path
    out = Path(out_csv) if out_csv is not None else \
        src.with_name(f"{src.stem}_wide.csv")
    _check_output(out, [src])
    if out.is_file() and not overwrite_existing:
        if verbose:
            print(f"[reshape] {out} exists; leaving it (overwrite_existing=False)")
        return out
    with job:
        announce(on_progress, f"reading {src.name}")
        job.load()
        report = job.report()
        why = report.refusal()
        if why:
            raise ValueError(why)
        announce(on_progress, f"writing {out.name}")
        ticker = Ticker(on_progress, total=report.rows_out)
        written = job.write(out, on_rows=_counting(ticker))
    if written != report.rows_out:
        raise RuntimeError(f"long_to_wide wrote {written} rows but planned "
                           f"{report.rows_out}")
    if verbose:
        for line in report.sentences():
            print(f"[reshape] {line}")
        print(f"[reshape] {src.name} -> {out} ({written:,} rows)")
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"merge": merge_spreadsheets, "stack": stack_spreadsheets,
     "longer": wide_to_long, "wider": long_to_wide},
    description="Merge, stack, or reshape spreadsheets.",
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
