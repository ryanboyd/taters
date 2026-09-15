"""
Catch an analysis that cannot work before a single feature is extracted.

A statistics run fails in a handful of predictable ways, and every one of
them is visible in the spreadsheet on the first screen: an id column that
does not identify rows, a class with three members when the classifier needs
five in every fold, a group of one, an outcome column with "n/a" in it, a
gender column whose fourth and fifth levels have two rows each. Left to the
engines, each is discovered at the *last* step, after ten minutes of
extraction, as a refusal naming a column the user chose forty screens ago.

These checks read the whole column -- not the 200-row sample the wizard
offers columns from -- and return a :class:`Finding` the wizard turns into
one of two things: a re-asked question when nothing can fix it, or a
checkbox of values to keep when dropping the thin ones would. The thresholds
mirror the engines' own refusals; a test pins them together.

Pure functions over lists of strings, so they are cheap to test and the
wizard stays the only place that talks to a person.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

__all__ = ["Finding", "MIN_PER_CLASS", "MIN_PER_GROUP", "THIN", "MAX_CLASSES",
           "level_counts", "repeated_ids", "thin_classes", "small_groups",
           "non_numeric_outcome", "rare_levels", "few_rows"]

#: The classifier refuses a class with fewer rows than this: it cannot appear
#: in every fold, so its recall would be measured on nothing. Kept equal to
#: ``stats.classify.MIN_PER_CLASS`` by a test rather than by import, so the
#: wizard does not load numpy to ask a question.
MIN_PER_CLASS = 5
#: The group-differences step excludes a group with fewer rows than this.
MIN_PER_GROUP = 2
#: A level with fewer rows than this is worth a word even where the engine
#: would accept it: a mean of three is not a mean anyone should test.
THIN = 5
#: More distinct labels than this is an identifier, not a category.
MAX_CLASSES = 20


@dataclass
class Finding:
    """
    One thing that would stop the run, in words, with the fix when there is one.

    ``keep`` names the values worth keeping when a row filter would rescue
    the analysis; ``drop`` says which values are too thin and by how much.
    ``fatal`` means no filter helps -- the column itself is wrong for the
    question -- and the wizard asks the question again.
    """

    column: str
    message: str
    keep: List[str] = field(default_factory=list)
    drop: Dict[str, int] = field(default_factory=dict)
    fatal: bool = False


def level_counts(values: Sequence[object]) -> Counter:
    """Rows per distinct non-blank value, whitespace trimmed."""
    return Counter(str(v).strip() for v in values if str(v or "").strip())


def _few(items: Sequence[str], limit: int = 4) -> str:
    items = [f"'{i}'" for i in items]
    return ", ".join(items[:limit]) + (f" and {len(items) - limit} more"
                                       if len(items) > limit else "")


def _rows(n: int) -> str:
    return f"{n} row{'' if n == 1 else 's'}"


def repeated_ids(values_by_col: Mapping[str, Sequence[object]]) -> Optional[Finding]:
    """
    Whether the chosen id columns identify each row, composed as the gather
    composes ``text_id``.

    A repeating id gives several rows one ``text_id`` without combining
    them, and the join then refuses duplicate keys after every feature has
    been measured. The remedy is another column, or combining rows on the
    level question -- never a filter.
    """
    if not values_by_col:
        return None
    cols = list(values_by_col)
    n = min(len(v) for v in values_by_col.values())
    ids = Counter(" | ".join(str(values_by_col[c][i] or "").strip()
                             for c in cols) for i in range(n))
    dupes = {k: v for k, v in ids.items() if v > 1}
    if not dupes:
        return None
    worst = sorted(dupes.items(), key=lambda kv: (-kv[1], kv[0]))
    example = ", ".join(f"'{k}' × {v}" for k, v in worst[:3])
    joined = " + ".join(cols)
    return Finding(
        column=joined, fatal=True,
        message=(f"{joined} does not identify each row: {len(dupes)} value(s) "
                 f"repeat ({example}), so several rows would share one id "
                 f"and the run would stop when their features are joined. "
                 f"Pick column(s) that together are unique to each row, or "
                 f"say there is no id column and combine the rows that share "
                 f"one at the next stage."))


def thin_classes(values: Sequence[object], column: str) -> Optional[Finding]:
    """
    Whether a category column can be classified, and which classes can.

    Mirrors the classifier's refusals: at least two classes, no more than
    :data:`MAX_CLASSES`, and :data:`MIN_PER_CLASS` rows in every class it is
    asked to predict. A class too thin to predict is dropped by filter, not
    argued with -- three "prefer not to say" rows cannot be learned from.
    """
    counts = level_counts(values)
    if len(counts) < 2:
        return Finding(column, fatal=True, message=(
            f"{column} holds {'one value' if counts else 'nothing'}"
            f"{' (' + _few(list(counts)) + ')' if counts else ''}; "
            f"classifying needs at least two classes."))
    if len(counts) > MAX_CLASSES:
        return Finding(column, fatal=True, message=(
            f"{column} has {len(counts)} distinct values, which is more like "
            f"an identifier or a measurement than a set of categories "
            f"({MAX_CLASSES} is the limit). A measurement is predicted by "
            f"the ridge step instead."))
    thin = {v: n for v, n in counts.items() if n < MIN_PER_CLASS}
    if not thin:
        return None
    keep = [v for v, n in counts.most_common() if n >= MIN_PER_CLASS]
    listed = ", ".join(f"'{v}' ({_rows(n)})"
                       for v, n in sorted(thin.items(), key=lambda kv: kv[1]))
    if len(keep) < 2:
        return Finding(column, fatal=True, drop=thin, message=(
            f"{column} cannot be classified: a class needs at least "
            f"{MIN_PER_CLASS} rows to appear in every fold, and "
            f"{'only ' + repr(keep[0]) + ' has that many' if keep else 'none do'}"
            f" -- {listed}."))
    return Finding(column, keep=keep, drop=thin, message=(
        f"{column} has {len(thin)} class(es) with fewer than {MIN_PER_CLASS} "
        f"rows -- {listed} -- and a class that cannot appear in every fold "
        f"has its recall measured on nothing. The classifier would stop; "
        f"leaving those rows out lets it run on {', '.join(map(repr, keep))}."))


def small_groups(values: Sequence[object], column: str) -> Optional[Finding]:
    """
    Whether the groups can be compared, and which ones are worth comparing.

    The step itself excludes a group with fewer than :data:`MIN_PER_GROUP`
    rows; a group under :data:`THIN` rows is accepted and still not worth
    testing, so both are offered for dropping, with the thin ones unticked.
    """
    counts = level_counts(values)
    if len(counts) < 2:
        return Finding(column, fatal=True, message=(
            f"{column} holds {'one value' if counts else 'nothing'}"
            f"{' (' + _few(list(counts)) + ')' if counts else ''}; comparing "
            f"groups needs at least two."))
    small = {v: n for v, n in counts.items() if n < THIN}
    if not small:
        return None
    keep = [v for v, n in counts.most_common() if n >= THIN]
    if len(keep) < 2:
        keep = [v for v, n in counts.most_common() if n >= MIN_PER_GROUP]
    listed = ", ".join(f"'{v}' ({_rows(n)})"
                       for v, n in sorted(small.items(), key=lambda kv: kv[1]))
    if len(keep) < 2:
        return Finding(column, fatal=True, drop=small, message=(
            f"{column} cannot be compared across: fewer than two of its "
            f"groups have {MIN_PER_GROUP} or more rows ({listed})."))
    if len(keep) == len(counts):
        # Every group is small and none can be spared: a word, not a
        # question, since there is nothing to choose.
        return Finding(column, message=(
            f"{column} has small groups -- {listed}. Each is compared, and "
            f"a difference between groups this size should not be trusted."))
    return Finding(column, keep=keep, drop={v: counts[v] for v in counts
                                            if v not in keep}, message=(
        f"{column} has {len(small)} group(s) with fewer than {THIN} rows -- "
        f"{listed}. A group of one is excluded from every test; a group of "
        f"three or four is compared, and should not be trusted. Leaving "
        f"those rows out compares {', '.join(map(repr, keep))} cleanly."))


def non_numeric_outcome(values: Sequence[object], column: str) -> Optional[Finding]:
    """
    Whether an outcome column really holds numbers, all the way down.

    The wizard offers columns whose first 200 rows parse; row 812 saying
    "n/a" is what stops the ridge at the last step. Nothing to keep here:
    the words are the problem and the question is re-asked, with the cells
    named so they can be fixed in the spreadsheet.
    """
    bad = Counter()
    seen = 0
    for v in values:
        raw = str(v or "").strip()
        if not raw:
            continue
        try:
            float(raw)
            seen += 1
        except ValueError:
            bad[raw] += 1
    if not seen and not bad:
        return Finding(column, fatal=True,
                       message=f"{column} is empty; an outcome needs numbers.")
    if not bad:
        return None
    listed = ", ".join(f"'{v}' ({_rows(n)})" for v, n in bad.most_common(4))
    return Finding(column, fatal=True, drop=dict(bad), message=(
        f"{column} holds words as well as numbers -- {listed}"
        f"{' and more' if len(bad) > 4 else ''} -- so it cannot be predicted "
        f"or correlated as it is. Blank the cells that are not measurements "
        f"(a blank is missing, which every analysis handles), or pick "
        f"another column."))


def rare_levels(values: Sequence[object], column: str) -> Optional[Finding]:
    """
    Whether a categorical control has levels too thin to hold anything constant.

    A level with two rows is dummy coded like any other and the model fits a
    coefficient to two people. Legal, and worth offering to drop -- the
    3 "other" and 1 "prefer not to say" of a real gender column are exactly
    what a researcher wants to see named before deciding.
    """
    counts = level_counts(values)
    rare = {v: n for v, n in counts.items() if n < THIN}
    if not rare or len(counts) < 2:
        return None
    keep = [v for v, n in counts.most_common() if n >= THIN]
    listed = ", ".join(f"'{v}' ({_rows(n)})"
                       for v, n in sorted(rare.items(), key=lambda kv: kv[1]))
    if len(keep) < 2:
        return Finding(column, message=(
            f"{column} has level(s) with fewer than {THIN} rows -- {listed} "
            f"-- and dropping them would leave fewer than two levels to hold "
            f"constant. It will be fitted as it is."))
    return Finding(column, keep=keep, drop=rare, message=(
        f"{column} has {len(rare)} level(s) with fewer than {THIN} rows -- "
        f"{listed}. Each gets its own coefficient, fitted to that handful of "
        f"people. Leaving those rows out holds {', '.join(map(repr, keep))} "
        f"constant on rows there are enough of."))


def few_rows(n_rows: int, *, minimum: int = 10) -> Optional[Finding]:
    """Whether there are enough rows for any cross-validated model at all."""
    if n_rows >= minimum:
        return None
    return Finding("(rows)", fatal=True, message=(
        f"only {n_rows} row(s) have text; a cross-validated model needs at "
        f"least {minimum}, and group comparisons and correlations on so few "
        f"say little."))
