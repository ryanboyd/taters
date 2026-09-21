"""
Leaving rows out: one comparison, used at both ends of a run.

Rows get left out at two moments. Before the text is joined, on what the
spreadsheet already says -- a condition you are not analyzing, a screener
question somebody failed, an age below the one you meant to study. And before
the statistics, on what the run has since measured -- a word count, a
readability score.

The two questions are different but the comparison is the same, and it has to
mean the same thing in both places: somebody who leaves out rows where
``score >= 3`` before joining and somebody who does it afterwards must get the
same arithmetic. So the comparison lives here, once, and both callers use it.

Everything is **keep** semantics: a filter names the rows that stay. That is
how the statistics have always stored one (``["condition", "in", ["A", "B"]]``
keeps A and B), and it is what a saved pipeline carries, so a question phrased
as "leave out" is translated into a keep list by whoever asks it rather than
by introducing a second convention here.

A blank cell fails every operator, so a row with nothing in the filtered
column is left out. An unknown value is not evidence of anything, and the
alternative -- keeping it -- means a column that is half empty quietly stops
filtering.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = ["FILTER_OPS", "matches", "keeps_row"]

#: The comparisons a filter may use. ``in`` and ``not_in`` take a list.
FILTER_OPS = ("==", "!=", "<", "<=", ">", ">=", "in", "not_in")


def _numeric_member(text: str, values: Sequence[Any]) -> bool:
    """Whether ``text`` equals any of ``values`` read as numbers, so that
    "25" matches 25.0 in a list typed by hand."""
    try:
        number = float(text)
    except ValueError:
        return False
    for v in values:
        try:
            if float(str(v).strip()) == number:
                return True
        except ValueError:
            continue
    return False


def _same_value(text: str, value: Any) -> bool:
    """Numeric equality when both sides are numbers ("25" == "25.0"),
    stripped-string equality otherwise."""
    other = str(value).strip()
    try:
        return float(text) == float(other)
    except ValueError:
        return text == other


def matches(cell: Any, op: str, value: Any) -> bool:
    """One filter comparison. Blank cells fail every operator."""
    text = (str(cell) if cell is not None else "").strip()
    if not text:
        return False
    if op in ("<", "<=", ">", ">="):
        left = float(text)
        right = float(value)
        return {"<": left < right, "<=": left <= right,
                ">": left > right, ">=": left >= right}[op]
    if op in ("in", "not_in"):
        wanted = {str(v).strip() for v in value}
        hit = text in wanted or _numeric_member(text, value)
        return hit if op == "in" else not hit
    same = _same_value(text, value)
    return same if op == "==" else not same


def keeps_row(row: Mapping[str, Any],
              filters: Sequence[Sequence[Any]]) -> bool:
    """
    Whether a row survives every filter.

    All of them, not any: two filters are two things you asked to leave out,
    and a row has to clear both. A column a filter names but the row does not
    have reads as blank, which fails -- the same answer as an empty cell,
    because in both cases there is nothing to compare.
    """
    for spec in filters or ():
        column, op, value = spec[0], spec[1], spec[2]
        if not matches(row.get(str(column), ""), str(op), value):
            return False
    return True
