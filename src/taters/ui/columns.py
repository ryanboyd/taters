"""
Reading a spreadsheet's shape, for screens that have to ask about its columns.

Two screens need this and cannot share it any other way: the Wrangle task asks
which columns hold text and which numbers to average, and the wizard's
analysis stage asks which column names the groups and which hold the outcomes.
The wizard cannot import the tasks (the tasks import the wizard), so the
sniffing lives here, below both.

Nothing here decides anything -- it reads a header, samples some rows, and
says whether a column looks like numbers. What to do with that is the
screen's business.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

#: How much of the file to read for the sample. Big enough that a column of
#: mostly-blank scores still shows its type, small enough that pointing the
#: wizard at a two-gigabyte export does not stall the screen.
_SAMPLE_BYTES = 256 * 1024

#: How many rows the sample keeps. ~200 is plenty to tell a score column from
#: a username column, and the questions built on it are offers, not contracts:
#: the analyses themselves read every row and refuse honestly if a column
#: turns out to hold something else.
_SAMPLE_ROWS = 200


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

    ``delimiter`` is the answer already settled for this file (the source
    stage's); when a screen has one it must pass it, so what it offers is
    what the run will actually see. Without one the file is sniffed the way
    the gatherer sniffs it -- a tab-separated file named ``.csv`` reads as one
    column under a comma reader, and every column question built on that
    would be wrong.
    """
    path = Path(path)
    delimiter = delimiter or sniff_delimiter(path)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(_SAMPLE_BYTES)
    # The last line of a truncated read may be half a row; drop it unless the
    # sample held the whole file.
    lines = sample.splitlines()
    if len(sample) == _SAMPLE_BYTES and len(lines) > 1:
        lines = lines[:-1]
    reader = csv.DictReader(lines, delimiter=delimiter)
    rows: List[dict] = []
    for row in reader:
        rows.append(row)
        if len(rows) >= n:
            break
    return list(reader.fieldnames or []), rows


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
                 encoding: str = "utf-8-sig") -> Dict[str, List[str]]:
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
        for row in csv.DictReader(fh, delimiter=delimiter):
            for c in wanted:
                out[c].append(row.get(c) or "")
    return out


#: What a column is treated as. Shown beside every column name the wizard
#: offers, so the treatment is visible before it matters -- "numbers" go into
#: correlations and predictions as measurements, "labels" name groups,
#: classes and categorical controls, "text" is free writing and fits nowhere
#: but the text question.
KINDS = ("numbers", "labels", "text", "empty")

#: More distinct labels than this and a column is an identifier or a
#: measurement, not a set of categories -- the classifier's own limit.
MAX_LABELS = 20


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
    filled = [str(v).strip() for v in values if str(v or "").strip()]
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
    filled = [str(v).strip() for v in values if str(v or "").strip()]
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
    filled = [str(v).strip() for v in values if str(v or "").strip()]
    if kind == "numbers":
        return ["numbers", "labels"] if filled else ["numbers"]
    if kind == "labels":
        return ["labels", "numbers"] if filled and all(
            _is_number(v) for v in filled) else ["labels"]
    return [kind]
