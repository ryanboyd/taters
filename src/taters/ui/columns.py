"""
Reading a spreadsheet's shape, for screens that have to ask about its columns.

Two screens need this and cannot share it any other way: the Wrangle task asks
which columns hold text and which numbers to average, and the wizard's
analysis stage asks which column names the groups and which hold the outcomes.
The wizard cannot import the tasks (the tasks import the wizard), so the
sniffing lives here, below both.

Nothing here decides anything -- it reads a header, walks the rows, and says
whether a column looks like numbers. What to do with that is the screen's
business.

Nothing here holds the rows, either. The scan keeps a small record per column
and throws each row away as it goes, because a 700 MB file held as Python
strings is about three gigabytes, and the wizard was holding exactly that. The
questions that need more than one column at a time -- how many groups, whether
a column is constant within them, how many rows a filter keeps -- read the
file again when asked, with a progress line, and remember the answer.
"""

from __future__ import annotations

import csv
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Callable, Dict, FrozenSet, List, Mapping, Optional,
                    Sequence, Tuple, Union)

PathLike = Union[str, Path]

#: How many rows :func:`peek_csv` keeps when a caller does not say. Only the
#: screens that want a cheap look at a header use that default now; the wizard
#: inspects the whole file (see :func:`inspect_csv`).
_SAMPLE_ROWS = 200

#: How often a pass over the file reports progress, in rows. Often enough that
#: the count visibly moves on a big file, rarely enough that the reporting is
#: not the expensive part.
_TICK_EVERY = 2000

#: At most this many distinct values for a column to be offered as labels.
#: Defined up here because the Inspection methods use it as a default.
MAX_LABELS = 20

#: Distinct values are counted exactly up to here and reported as "over" past
#: it. The count exists to tell a grouping column from an id or a measurement,
#: and 10,000+ settles that as well as 393,725 would -- while holding 393,725
#: strings per free-text column, for thirty columns, would not fit in memory.
UNIQUE_CEILING = 10_000


class _Row(Mapping):
    """
    One kept row, as a mapping, without a dict per row.

    Only :func:`peek_csv` keeps rows now, and only a few, for screens that
    want to show a preview. The header's index is built once and shared, and
    a row is the tuple of its values.

    Read-only and only ever asked for ``.get``, which is all the screens do.
    """

    __slots__ = ("_index", "_values")

    def __init__(self, index: Dict[str, int], values: Tuple[str, ...]):
        self._index = index
        self._values = values

    def __getitem__(self, key: str) -> str:
        return self._values[self._index[key]]

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return repr(dict(self))


@dataclass
class ColumnStats:
    """
    What the scan keeps about one column, which is everything the pickers ask.

    ``column_kind``, ``plausible_labels``, ``looks_numeric``, ``kind_options``
    and the value tick-lists all used to walk every row to answer -- per
    screen, since nothing cached them, and on a 500,000-row file that was
    twenty-odd silent seconds a screen. Every one of those answers comes out
    of this record, so it is gathered once, inside the scan that already
    shows a progress bar, and the screens become lookups.

    ``counts`` is the exact tally of every value while the column has at most
    ``MAX_LABELS + 1`` distinct ones -- enough to list a labels column with
    how many rows hold each -- and ``None`` past that, when a tally would be
    a copy of the column. ``distinct`` is the first ``MAX_LABELS + 1`` values
    seen either way, in order, so a screen can show a few back.
    """

    filled: int = 0
    all_numeric: bool = True
    distinct: List[str] = field(default_factory=list)
    counts: Optional[Dict[str, int]] = None
    #: Exact distinct count up to UNIQUE_CEILING; past it, the ceiling plus one.
    unique: int = 0
    unique_over: bool = False

    @property
    def distinct_count(self) -> int:
        return len(self.distinct)

    @property
    def over_limit(self) -> bool:
        return len(self.distinct) > MAX_LABELS


@dataclass(frozen=True)
class Grain:
    """
    The file seen at one grain: rows grouped by a set of key columns.

    Everything the wizard says about a grouping before committing to it --
    "then one per Education (7 rows)", which columns still have one value
    per group and can be compared or controlled for, whether the groups
    differ in size -- comes from one pass over the file with these keys.
    """

    keys: Tuple[str, ...]
    #: Distinct key combinations, i.e. rows the grouped table would have.
    groups: int
    rows: int
    smallest: int
    largest: int
    #: The columns the pass watched for a single value per group ...
    tracked: FrozenSet[str]
    #: ... and the ones that turned out to hold more than one somewhere.
    varies: FrozenSet[str]

    @property
    def sizes_differ(self) -> bool:
        return self.smallest != self.largest

    def constant(self, column: str) -> bool:
        """Whether ``column`` has one value per group (blanks ignored)."""
        return column in self.keys or (column in self.tracked
                                        and column not in self.varies)


@dataclass
class Inspection:
    """
    What one pass over a spreadsheet found.

    Held rather than recomputed: the source stage, the level question and the
    analysis stage all ask about the same file, and they used to scan it once
    each. The per-column facts live in ``stats``; the cross-column ones are
    read on demand by :meth:`grain` and :meth:`survivors`, which go back to
    the file at ``path``.
    """

    columns: List[str]
    #: Only what :func:`peek_csv` was asked to keep. The wizard keeps none.
    rows: List[Mapping] = field(default_factory=list)
    scanned: int = 0
    #: A row limit stopped the scan before the end of the file.
    truncated: bool = False
    #: Rows with more fields than the header has columns, and fewer. Either
    #: means a quoting or delimiter problem, and the point of saying so at
    #: the moment the file is chosen is that the alternative is finding out
    #: after an hour of extraction.
    over: int = 0
    under: int = 0
    #: Per-column facts gathered during the scan. Absent (empty) for an
    #: Inspection built by hand from a header, in which case the methods
    #: below fall back to walking ``rows``, which is correct and merely slow.
    stats: Dict[str, "ColumnStats"] = field(default_factory=dict)
    #: Where the file is, so the grain and survivor passes can read it again.
    path: Optional[Path] = None
    delimiter: Optional[str] = None
    _grains: Dict[Tuple[str, ...], Grain] = field(default_factory=dict,
                                                  repr=False, compare=False)

    @property
    def ragged(self) -> int:
        return self.over + self.under

    # ---- the questions the pickers ask, answered from the summary --------

    def kind(self, column: str, *, max_labels: int = MAX_LABELS) -> str:
        s = self.stats.get(column)
        if s is None:
            return column_kind([r.get(column) for r in self.rows],
                               max_labels=max_labels)
        if not s.filled:
            return "empty"
        if s.all_numeric:
            return "numbers"
        if 2 <= s.distinct_count <= max_labels and s.distinct_count < s.filled:
            return "labels"
        return "text"

    def plausible_labels(self, column: str, *,
                         max_labels: int = MAX_LABELS) -> bool:
        s = self.stats.get(column)
        if s is None:
            return plausible_labels([r.get(column) for r in self.rows],
                                    max_labels=max_labels)
        return 2 <= s.distinct_count <= max_labels and s.distinct_count < s.filled

    def looks_numeric(self, column: str) -> bool:
        s = self.stats.get(column)
        if s is None:
            return looks_numeric(self.rows, column)
        return bool(s.filled) and s.all_numeric

    def kind_options(self, column: str, kind: str, *,
                     max_labels: int = MAX_LABELS) -> List[str]:
        s = self.stats.get(column)
        if s is None:
            return kind_options(kind, [r.get(column) for r in self.rows],
                                max_labels=max_labels)
        if kind == "numbers":
            return ["numbers", "labels"] if s.filled else ["numbers"]
        if kind == "labels":
            return ["labels", "numbers"] if s.filled and s.all_numeric \
                else ["labels"]
        return [kind]

    def unique_counts(self, columns: Sequence[str]) -> Dict[str, int]:
        """
        How many different values each column holds, blanks excluded.

        Exact up to ``UNIQUE_CEILING``; above it the ceiling plus one, which
        the screen renders as "10,000+". This used to be a full pass per
        column per screen -- six silent seconds on a 500k-row file, three
        screens running.
        """
        out: Dict[str, int] = {}
        slow: List[str] = []
        for c in columns:
            s = self.stats.get(c)
            if s is None:
                slow.append(c)
            else:
                out[c] = s.unique
        for c in slow:
            out[c] = len({str(r.get(c) or "").strip() for r in self.rows} - {""})
        return out

    def distinct(self, column: str, limit: int = 12) -> List[str]:
        """First-seen distinct values, up to ``limit + 1`` of them."""
        s = self.stats.get(column)
        if s is None or limit > MAX_LABELS:
            return distinct_values(self.rows, column, limit=limit)
        return s.distinct[:limit + 1]

    def value_counts(self, column: str) -> Optional[Dict[str, int]]:
        """
        How many rows hold each value of a column, commonest first -- or
        None when there are too many values for that to be a list anyone
        would tick through (more than ``MAX_LABELS + 1``).
        """
        s = self.stats.get(column)
        if s is None:
            tally: Dict[str, int] = {}
            for row in self.rows:
                v = str(row.get(column) or "").strip()
                if v:
                    tally[v] = tally.get(v, 0) + 1
            if len(tally) > MAX_LABELS + 1:
                return None
        elif s.counts is None:
            return None
        else:
            tally = s.counts
        return dict(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])))

    # ---- the questions that need more than one column ---------------------

    def grain(self, keys: Sequence[str], *, track: Sequence[str] = (),
              on_rows: Optional[Callable[[int], None]] = None) -> Grain:
        """
        The file grouped by ``keys``, watching ``track`` for constancy.

        One pass, remembered per key set, so the same grouping asked about
        by three screens is read once. A remembered pass that did not watch
        every column now asked for is redone with the union.
        """
        keys = tuple(keys)
        wanted = frozenset(track) - set(keys)
        have = self._grains.get(keys)
        if have is not None and wanted <= have.tracked:
            return have
        if have is not None:
            wanted = wanted | have.tracked
        found = _scan_grain(self, keys, wanted, on_rows=on_rows)
        self._grains[keys] = found
        return found

    def survivors(self, filters: Sequence[Sequence[object]], *,
                  on_rows: Optional[Callable[[int], None]] = None) -> int:
        """
        How many rows every filter keeps.

        Exact from the tally when it is one filter on a column short enough
        to have one, since that is what the tick-list was built from; a pass
        over the file otherwise.
        """
        from ..helpers.row_filter import keeps_row

        if not filters:
            return self.scanned
        if len(filters) == 1 and str(filters[0][1]) == "in":
            counts = self.value_counts(str(filters[0][0]))
            if counts is not None:
                keep = {str(v) for v in (filters[0][2] or ())}
                return sum(n for v, n in counts.items() if v in keep)
        kept = 0
        for n, row in _each_row(self, on_rows=on_rows):
            if keeps_row(row, filters):
                kept += 1
        return kept

    def trouble(self) -> str:
        """One line about the file's formatting, or ''."""
        if not self.ragged:
            return ""
        parts = []
        if self.over:
            parts.append(f"{self.over:,} with extra fields")
        if self.under:
            parts.append(f"{self.under:,} short of the header")
        return (f"{self.ragged:,} of {self.scanned:,} rows do not match the "
                f"header ({', '.join(parts)}). Usually an unescaped quote or "
                f"the wrong separator.")


def inspect_csv(path: PathLike, *, delimiter: Optional[str] = None,
                limit: int = 0, keep_rows: int = 0,
                on_rows: Optional[Callable[[int], None]] = None
                ) -> Inspection:
    """
    Read a spreadsheet and report its columns, its problems, and what each
    column holds.

    The whole file by default. The questions built on this are offers rather
    than contracts -- the analyses read every row regardless and refuse
    honestly -- but an offer made from the first two hundred rows is wrong in
    a way nobody can see: on a real file with 143 columns, ten columns looked
    constant within a group in the sample and were not in the other 738 rows,
    so they were offered as controls that would have arrived empty.

    Parameters
    ----------
    path : str or Path
        The spreadsheet.
    delimiter : str, optional
        The separator already settled for this file. Sniffed when absent.
    limit : int, default 0
        Stop after this many rows. ``0`` reads all of them.
    keep_rows : int, default 0
        Hold on to this many rows from the top, for a screen that wants to
        show a preview. Nothing else needs them, and the default keeps none:
        the record per column is what the questions are answered from.
    on_rows : callable, optional
        Called with the running row count every few thousand rows, for a
        progress line. A scan of a large file is the first slow thing the
        wizard does and the only one that used to happen in silence.

    Returns
    -------
    Inspection
    """
    path = Path(path)
    delimiter = delimiter or sniff_delimiter(path)
    columns: List[str] = []
    rows: List[Mapping] = []
    over = under = scanned = 0
    truncated = False
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        for raw in reader:
            columns = [str(c) for c in raw]
            break
        index = {name: i for i, name in enumerate(columns)}
        width = len(columns)

        # per-column accumulators as parallel lists indexed by position: this
        # loop runs width x rows times, and attribute lookups on an object per
        # cell were measurably slower than list indexing. A column stops
        # being examined once there is nothing left to learn about it -- text
        # with more distinct values than the labels limit -- except whether it
        # is empty, which is one flag.
        filled = [0] * width
        numeric = [True] * width
        # value -> how many rows hold it, kept while the column has few
        # enough values to list; frozen into `first_seen` and dropped past
        # the cap, when the tally would be most of the column over again
        tally: List[Optional[Dict[str, int]]] = [{} for _ in range(width)]
        first_seen: List[List[str]] = [[] for _ in range(width)]
        settled = [False] * width
        cap = MAX_LABELS + 1
        # one set per column for the exact unique count, dropped -- and its
        # memory with it -- the moment it passes the ceiling
        uniq: List[Optional[set]] = [set() for _ in range(width)]

        for raw in reader:
            if not raw:
                continue            # a blank line is not a row
            if len(raw) > width:
                over += 1
            elif len(raw) < width:
                under += 1
            values = tuple(raw[:width]) if len(raw) >= width \
                else tuple(raw) + ("",) * (width - len(raw))
            if scanned < keep_rows:
                rows.append(_Row(index, values))
            scanned += 1

            for i, cell in enumerate(values):
                v = cell.strip()
                if not v:
                    continue
                filled[i] += 1
                us = uniq[i]
                if us is not None:
                    us.add(v)
                    if len(us) > UNIQUE_CEILING:
                        uniq[i] = None
                if settled[i]:
                    continue
                if numeric[i]:
                    try:
                        float(v)
                    except ValueError:
                        numeric[i] = False
                t = tally[i]
                if t is not None:
                    n = t.get(v)
                    if n is not None:
                        t[v] = n + 1
                    elif len(t) < cap:
                        t[v] = 1
                    else:
                        # too many values to list: keep the first few
                        # seen and stop counting
                        first_seen[i] = list(t)
                        tally[i] = None
                        # non-numeric and past the labels limit: it is free
                        # text, and nothing more can change that verdict
                        if not numeric[i]:
                            settled[i] = True
                elif not numeric[i]:
                    settled[i] = True

            if on_rows is not None and scanned % _TICK_EVERY == 0:
                on_rows(scanned)
            if limit and scanned >= limit:
                truncated = True
                break
    if on_rows is not None:
        on_rows(scanned)
    stats = {}
    for i, name in enumerate(columns):
        t = tally[i]
        stats[name] = ColumnStats(
            filled=filled[i], all_numeric=numeric[i],
            distinct=list(t) if t is not None else first_seen[i],
            counts=t,
            unique=(len(uniq[i]) if uniq[i] is not None else UNIQUE_CEILING + 1),
            unique_over=uniq[i] is None)
    return Inspection(columns=columns, rows=rows, scanned=scanned,
                      truncated=truncated, over=over, under=under,
                      stats=stats, path=path, delimiter=delimiter)


def _each_row(insp: Inspection, *,
              on_rows: Optional[Callable[[int], None]] = None):
    """
    The file's rows again, one at a time, as ``(count, mapping)``.

    For the passes that need whole rows -- a grouping, a filter -- and are
    made after the scan, on demand. Reads from ``insp.path``; an Inspection
    built by hand walks whatever rows it was given instead.
    """
    if insp.path is None:
        for n, row in enumerate(insp.rows, 1):
            yield n, row
        return
    delimiter = insp.delimiter or sniff_delimiter(insp.path)
    with Path(insp.path).open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            return
        columns = [str(c) for c in header]
        index = {name: i for i, name in enumerate(columns)}
        width = len(columns)
        n = 0
        for raw in reader:
            if not raw:
                continue
            values = tuple(raw[:width]) if len(raw) >= width \
                else tuple(raw) + ("",) * (width - len(raw))
            n += 1
            yield n, _Row(index, values)
            if on_rows is not None and n % _TICK_EVERY == 0:
                on_rows(n)
        if on_rows is not None:
            on_rows(n)


def _scan_grain(insp: Inspection, keys: Tuple[str, ...],
                track: FrozenSet[str], *,
                on_rows: Optional[Callable[[int], None]] = None) -> Grain:
    """
    One pass over the file grouped by ``keys``.

    Group sizes are an int per group. A tracked column holds, per group, the
    hash of the value seen there -- eight bytes, whatever the value is -- and
    the moment two rows of one group disagree the column is marked as varying
    and its array is dropped. So the pass costs eight bytes per group per
    column that really is constant within the groups, and never holds a
    row or a value. (A hash collision could call a varying column constant;
    at 64 bits that is not a thing that happens, and the gatherer would
    leave the column blank and the analysis say so if it did.)
    """
    ids: Dict[Tuple[str, ...], int] = {}
    sizes = array("i")
    header = {name: i for i, name in enumerate(insp.columns)}
    live = [c for c in track if c not in keys and c in header]
    at = {c: header[c] for c in live}
    key_at = [header.get(k, -1) for k in keys]
    per_group: Dict[str, array] = {c: array("q") for c in live}
    varies: set = set()
    rows = 0
    for rows, row in _each_row(insp, on_rows=on_rows):
        vals = getattr(row, "_values", None)
        if vals is None:            # a hand-built Inspection's dict rows
            vals = tuple(str(row.get(c) or "") for c in insp.columns)
        key = tuple(vals[i].strip() if i >= 0 else "" for i in key_at)
        gid = ids.get(key)
        if gid is None:
            gid = ids[key] = len(ids)
            sizes.append(1)
            for c in live:
                per_group[c].append(-1)
        else:
            sizes[gid] += 1
        dropped = False
        for c in live:
            v = vals[at[c]].strip()
            if not v:
                continue            # a blank does not disagree with anything
            code = hash(v)          # never -1: Python reserves it
            seen = per_group[c][gid]
            if seen == -1:
                per_group[c][gid] = code
            elif seen != code:
                varies.add(c)
                dropped = True
        if dropped:
            # stop watching what has already told us
            live = [c for c in live if c not in varies]
            for c in varies:
                per_group.pop(c, None)
    return Grain(keys=keys, groups=len(ids), rows=rows,
                 smallest=min(sizes) if sizes else 0,
                 largest=max(sizes) if sizes else 0,
                 tracked=frozenset(c for c in track if c not in keys),
                 varies=frozenset(varies))


#: The separators a spreadsheet can actually have. The Sniffer, handed prose,
#: happily returns "e" or a space; anything outside this set is less
#: trustworthy than the boring default.
DELIMITERS = (",", "\t", ";", "|")


def sniff_delimiter(path: PathLike) -> str:
    """
    Comma, tab, semicolon or pipe, decided by the file rather than by its name.

    The one sniff for the whole wizard. There were two: this module used
    ``csv.Sniffer`` bare while the source stage routed through the gatherer's
    sniffer and then whitelisted the answer, so on a file where the two
    disagreed the analysis screen offered columns the composed pipeline --
    which carries the source stage's answer -- would never see.
    """
    from ..helpers.text_gather import _detect_delimiter

    try:
        sample = Path(path).open("rb").read(64 * 1024)
    except OSError:
        return ","
    found = _detect_delimiter(sample, default=",")
    return found if found in DELIMITERS else ","


def peek_csv(path: PathLike, n: int = _SAMPLE_ROWS,
             delimiter: Optional[str] = None) -> Tuple[List[str], List[dict]]:
    """
    A spreadsheet's column names and a sample of its rows.

    The cheap look, for a screen that wants a header and a few rows.
    :func:`inspect_csv` is the one the wizard uses: it reads the whole file,
    so what it says about a column is true of the column rather than of its
    first two hundred values.

    ``delimiter`` is the answer already settled for this file (the source
    stage's); when a screen has one it must pass it, so what it offers is
    what the run will actually see. Without one the file is sniffed the way
    the gatherer sniffs it -- a tab-separated file named ``.csv`` reads as one
    column under a comma reader, and every column question built on that
    would be wrong.
    """
    n = max(0, int(n))
    found = inspect_csv(path, delimiter=delimiter, limit=n, keep_rows=n)
    return found.columns, found.rows


def looks_numeric(rows: Sequence[dict], column: str) -> bool:
    """
    Whether every non-blank sampled value in a column parses as a number.

    The test for offering a column as something to average, correlate or
    predict. Blank cells are ignored (missing is not "not a number"), but at
    least one real value has to be there: an entirely empty column is not a
    measurement.
    """
    seen = False
    for row in rows:
        raw = (row.get(column) or "").strip()
        if not raw:
            continue
        try:
            float(raw)
        except ValueError:
            return False
        seen = True
    return seen


def distinct_values(rows: Sequence[dict], column: str,
                    limit: int = 12) -> List[str]:
    """
    The distinct non-blank values seen in a column, in first-seen order.

    Used to show a group column's levels back to the user ("A, B, C") before
    they commit to it, and to spot the column that is really a measurement:
    a "group" with 200 distinct values in a 200-row sample is an id or a
    score, whatever it is called.
    """
    seen: Dict[str, None] = {}
    for row in rows:
        value = (row.get(column) or "").strip()
        if value and value not in seen:
            seen[value] = None
            if len(seen) > limit:
                break
    return list(seen)


def constant_within(rows: Sequence[dict], keys: Sequence[str],
                    column: str) -> bool:
    """
    Whether ``column`` holds one value per group defined by ``keys``.

    This is what separates the two senses of "group" that a spreadsheet
    invites you to confuse. Combining texts on ``subreddit + author`` is a
    decision about what one row *is*; comparing moderators against regular
    users is a decision about what you want to *test*. They are independent,
    and the only thing that ties them is this question -- after the combining,
    does the column still have a single value to speak for each row?

    Three cases, on a run combined by ``subreddit + author``:

    * ``subreddit`` -- a key, so one value per group by construction.
    * ``is_moderator`` -- not a key, but a fact about the author, so still
      one value per group. Usable, and previously refused: the wizard offered
      only the keys themselves, which made "compare moderators with regular
      users on their combined comments" impossible to express.
    * ``score`` -- differs from comment to comment, so there is no single
      value for the group. The gatherer already leaves such a column blank
      rather than quietly reporting the first row's value, so picking one
      would produce an analysis with every row dropped.

    Sampled, like everything else here, so it is an offer rather than a
    contract: a column that looks constant in 200 rows and varies later ends
    up blank in the gathered table, and the analysis says so instead of
    inventing a value.
    """
    if not keys:
        return False
    seen: Dict[tuple, str] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")).strip() for k in keys)
        value = str(row.get(column, "")).strip()
        if value == "":
            continue
        if seen.setdefault(key, value) != value:
            return False
    return bool(seen)


def read_columns(path: PathLike, columns: Sequence[str], *,
                 delimiter: Optional[str] = None,
                 encoding: str = "utf-8-sig",
                 on_rows: Optional[Callable[[int], None]] = None
                 ) -> Dict[str, List[str]]:
    """
    Every value of a few columns, in row order -- the whole file this time.

    :func:`peek_csv` samples 200 rows, which is right for *offering* a column
    and wrong for *vetting* one: a class with three members, an id that
    repeats on row 500, an outcome that says "n/a" on row 812 are all
    invisible in a sample and all stop the run at its last step. Reading a
    handful of named columns end to end costs a second on a spreadsheet of a
    hundred thousand rows, and the checks that read this are the ones that
    decide whether the run can succeed at all.
    """
    wanted = [str(c) for c in columns]
    out: Dict[str, List[str]] = {c: [] for c in wanted}
    if not wanted:
        return out
    delimiter = delimiter or sniff_delimiter(path)
    with Path(path).open("r", newline="", encoding=encoding) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = [str(c) for c in next(reader, [])]
        # position per wanted column; a name the file lacks reads blank
        at = [(c, header.index(c)) if c in header else (c, -1) for c in wanted]
        n = 0
        for raw in reader:
            if not raw:
                continue
            n += 1
            for c, i in at:
                out[c].append(raw[i] if 0 <= i < len(raw) else "")
            if on_rows is not None and n % _TICK_EVERY == 0:
                on_rows(n)
    if on_rows is not None:
        on_rows(n)
    return out


#: What a column is treated as. Shown beside every column name the wizard
#: offers, so the treatment is visible before it matters -- "numbers" go into
#: correlations and predictions as measurements, "labels" name groups,
#: classes and categorical controls, "text" is free writing and fits nowhere
#: but the text question.
KINDS = ("numbers", "labels", "text", "empty")

#: More distinct labels than this and a column is an identifier or a
#: measurement, not a set of categories -- the classifier's own limit.


def _is_number(raw: str) -> bool:
    try:
        float(raw)
        return True
    except ValueError:
        return False


def _repeating(filled: Sequence[str], max_labels: int) -> bool:
    distinct = set(filled)
    return 2 <= len(distinct) <= max_labels and len(distinct) < len(filled)


def plausible_labels(values: Sequence[object], *,
                     max_labels: int = MAX_LABELS) -> bool:
    """
    Whether a column's values could serve as labels *as they stand*: at
    least two distinct, no more than ``max_labels``, and some repeating.

    The arrows let a person call any column of numbers labels; this is the
    narrower question the wizard asks before *offering* one -- a column with
    a value per row is nothing to compare or classify, whatever it is called.
    """
    filled = [s for s in (str(v or "").strip() for v in values) if s]
    return _repeating(filled, max_labels)


def column_kind(values: Sequence[object], *, max_labels: int = MAX_LABELS) -> str:
    """
    ``"numbers"``, ``"labels"``, ``"text"`` or ``"empty"``, from the values.

    Numbers win: a column that parses is a measurement until someone says
    otherwise (a 1/2 gender code is the case for saying otherwise -- see
    :func:`kind_options`). Labels are strings that repeat, and not too many
    of them; anything else is free text, or an identifier, which for the
    statistics is the same thing.
    """
    filled = [s for s in (str(v or "").strip() for v in values) if s]
    if not filled:
        return "empty"
    if all(_is_number(v) for v in filled):
        return "numbers"
    if _repeating(filled, max_labels):
        return "labels"
    return "text"


def kind_options(kind: str, values: Sequence[object], *,
                 max_labels: int = MAX_LABELS) -> List[str]:
    """
    The kinds a column may be *treated* as, current one first.

    A column of numbers may always be treated as labels: the 1/2 gender
    code and the 1-3 condition code are the usual case, but a column of many
    distinct numbers is the person's to call too -- an item number, a site
    code -- and whether the analysis can *use* it that way is checked when
    it is chosen, in words, not refused here in silence. Labels may go back
    to numbers only if every value parses. Free text is free text. The
    wizard turns this into the left/right arrows on a column's row.
    """
    filled = [s for s in (str(v or "").strip() for v in values) if s]
    if kind == "numbers":
        return ["numbers", "labels"] if filled else ["numbers"]
    if kind == "labels":
        return ["labels", "numbers"] if filled and all(
            _is_number(v) for v in filled) else ["labels"]
    return [kind]
