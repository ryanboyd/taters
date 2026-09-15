"""
Control variables, shared by every analysis that can hold something constant.

"What is the relationship between language and depression, *after* age and
gender?" is not a different question from the unadjusted one -- it is the
question, most of the time, and an analysis that cannot answer it is an
analysis a reviewer sends back. So controls are not a feature of one step
here: correlations become partial correlations, group comparisons become
covariance-adjusted comparisons, and a prediction model can be fitted with
the controls alone, the language alone, or both, so the language's own
contribution can be read off.

Two kinds of control column, and the difference matters:

**Continuous** (age, word count) enters as itself.

**Categorical** (gender, condition, site) cannot enter as itself -- there is
no sense in which "female" is one more than "male" -- so it is expanded
into indicator columns, one per level but the first, which becomes the
reference the others are measured against. Which level that is, is recorded
in the output rather than left to be inferred, because every adjusted mean
in the table is relative to it.

Whether a column is one or the other is decided by looking, not by asking:
a column whose every value parses as a number is continuous, and anything
else is categorical. A numeric column that is *really* categorical (a site
coded 1/2/3) can be named explicitly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ._common import looks_numeric, name_a_few

#: How many levels a categorical control may have before it is refused. Each
#: level costs a column and a degree of freedom, so a "control" with fifty
#: of them is not a control -- it is an id, and adjusting for it would
#: explain the outcome away entirely.
MAX_LEVELS = 20


def split_kinds(table, control_cols: Sequence[str],
                categorical: Optional[Sequence[str]] = None
                ) -> Tuple[List[str], List[str]]:
    """Which controls are continuous and which are categorical."""
    named = {str(c) for c in (categorical or ())}
    missing = [c for c in control_cols if c not in table.columns]
    if missing:
        raise ValueError(
            f"control column(s) not in the analysis table: "
            f"{name_a_few(missing)}. It has {len(table.columns)}: "
            f"{name_a_few(list(table.columns))}.")
    unknown = [c for c in named if c not in control_cols]
    if unknown:
        raise ValueError(
            f"named as categorical but not listed as a control: "
            f"{name_a_few(unknown)}")
    continuous, discrete = [], []
    for column in control_cols:
        column = str(column)
        if column in named or not looks_numeric(table[column].tolist()):
            discrete.append(column)
        else:
            continuous.append(column)
    return continuous, discrete


def build(table, control_cols: Sequence[str],
          categorical: Optional[Sequence[str]] = None):
    """
    The control design matrix, its column names, and what was done to build it.

    Returns ``(matrix, names, notes, spec)``. ``matrix`` has one row per
    table row
    and one column per continuous control plus one per non-reference level of
    each categorical one; a row with any control missing is NaN throughout,
    so the analyses drop it the same way they drop a missing outcome.

    ``notes`` records each categorical control's reference level, because
    every coefficient and every adjusted mean in the output is relative to
    it and a reader cannot work that out from the numbers.

    ``spec`` says how each design column was built, in order, so a saved
    model can rebuild the same columns from a fresh spreadsheet -- which is
    the only way a model fitted with a control like ``gender`` can score a
    second dataset, where that column is still the original strings and
    nothing has yet dummy-coded them. See :func:`rebuild`.
    """
    import numpy as np

    from ._common import numeric_column

    control_cols = [str(c) for c in control_cols]
    if not control_cols:
        return np.zeros((len(table), 0)), [], [], []

    continuous, discrete = split_kinds(table, control_cols, categorical)
    columns, names, notes, spec = [], [], [], []

    for column in continuous:
        columns.append(numeric_column(table[column].tolist(), column=column))
        names.append(column)
        spec.append({"column": column})

    for column in discrete:
        values = [str(v).strip() for v in table[column].tolist()]
        levels = sorted({v for v in values if v})
        if len(levels) < 2:
            raise ValueError(
                f"control column {column!r} has "
                f"{'one value' if levels else 'no values'} "
                f"({levels or 'all blank'}); a control has to vary.")
        if len(levels) > MAX_LEVELS:
            raise ValueError(
                f"control column {column!r} has {len(levels)} distinct "
                f"values, which is more like an identifier than a control "
                f"({MAX_LEVELS} is the limit). Adjusting for it would "
                f"explain the outcome away rather than account for it.")
        reference = levels[0]
        notes.append(f"{column}: levels compared against {reference!r}")
        for level in levels[1:]:
            # a blank stays missing here -- we don't want it turning into "not
            # this level" (a row with no gender recorded isn't male because
            # of it).
            columns.append(np.array(
                [np.nan if not v else (1.0 if v == level else 0.0)
                 for v in values]))
            names.append(f"{column}={level}")
            spec.append({"column": column, "level": level,
                         "reference": reference})

    return np.column_stack(columns), names, notes, spec


def rebuild(table, spec):
    """
    The same control columns again, from a table that has not seen them.

    A model fitted with ``age`` and ``gender`` stores this recipe and
    replays it here, because the spreadsheet being scored holds ``gender``
    as the words the researcher typed -- nothing has dummy-coded it, and the
    reference level that every coefficient is relative to lives only in the
    model.

    Returns ``(matrix, names, unknown)``, where ``unknown`` counts rows whose
    categorical value the model never saw. Those rows are left missing, not
    coded as the reference level: a respondent whose gender is recorded as
    something the training data lacked is not thereby the reference, and
    zeroing every indicator would say exactly that while scoring them
    anyway.
    """
    import numpy as np

    from ._common import numeric_column

    needed = sorted({entry["column"] for entry in spec})
    absent = [c for c in needed if c not in table.columns]
    if absent:
        raise ValueError(
            f"the table is missing control column(s) this model was fitted "
            f"with: {', '.join(absent)}. Score a table that carries the same "
            f"controls, or use a model fitted without them.")
    if not spec:
        return np.zeros((len(table), 0)), [], {}

    columns, names = [], []
    unknown: dict = {}
    known: dict = {}
    for entry in spec:
        if "level" in entry:
            known.setdefault(entry["column"], {entry["reference"]})
            known[entry["column"]].add(entry["level"])
    for entry in spec:
        column = entry["column"]
        if "level" not in entry:
            columns.append(numeric_column(table[column].tolist(),
                                          column=column))
            names.append(column)
            continue
        values = [str(v).strip() for v in table[column].tolist()]
        seen = known[column]
        strange = {v for v in values if v and v not in seen}
        if strange:
            unknown[column] = sorted(strange)
        columns.append(np.array(
            [np.nan if (not v or v not in seen)
             else (1.0 if v == entry["level"] else 0.0) for v in values]))
        names.append(f"{column}={entry['level']}")
    return np.column_stack(columns), names, unknown


def residualize(values, controls, *, ridge: float = 0.0):
    """
    What is left of ``values`` once the controls have had their say.

    Ordinary least squares against an intercept and the controls, returning
    the residuals -- which is what "adjusted for age and gender" means, and
    what makes a partial correlation partial.

    Rows with a missing value or a missing control come back NaN: they can
    take no part in the adjustment, and filling them in would invent the
    very thing being controlled for. ``ridge`` adds a tiny penalty for the
    case where two controls are perfectly collinear (a dummy set that
    duplicates another column), which would otherwise make the solve
    singular.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    controls = np.asarray(controls, dtype=float)
    out = np.full(values.shape, np.nan)
    if controls.shape[1] == 0:
        return values.copy()

    usable = ~np.isnan(values) & ~np.isnan(controls).any(axis=1)
    if usable.sum() < controls.shape[1] + 2:
        return out
    x = np.column_stack([np.ones(int(usable.sum())), controls[usable]])
    y = values[usable]
    gram = x.T @ x
    if ridge:
        penalty = np.eye(gram.shape[0]) * ridge
        penalty[0, 0] = 0.0          # we leave the intercept alone; no penalty
        gram = gram + penalty
    try:
        beta = np.linalg.solve(gram, x.T @ y)
    except np.linalg.LinAlgError:
        # perfectly collinear controls. least squares still has a minimum, it
        # just isn't unique -- but the residuals are, and those are all we
        # hand back anyway.
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    out[usable] = y - x @ beta
    return out


def control_subsets(controls, combos: str):
    """
    Which sets of control variables to fit, given how thorough to be.

    Shared by every fitter; it lived in ridge.py and the classifier reached
    in for it, which put a control-variable rule in the one module with no
    other business knowing it.

    ``"none_and_all"`` (the default) is the pair that answers the question
    people actually ask: the controls on their own, and the controls plus
    language. Enumerating every subset -- which is what a thorough
    exploration does -- is offered, but it is 2^k models and the table grows
    with it, so it is not what you get without asking.
    """
    from itertools import combinations

    controls = [str(c) for c in controls]
    if not controls:
        return [()]
    combos = str(combos or "none_and_all").strip().lower()
    if combos == "none_and_all":
        return [(), tuple(controls)]
    if combos == "each":
        # one control at a time, so that we can see which of them is actually
        # doing the work.
        return [()] + [(c,) for c in controls] + [tuple(controls)]
    if combos == "subsets":
        return [tuple(s) for r in range(len(controls) + 1)
                for s in combinations(controls, r)]
    raise ValueError(
        f"control_combos must be 'none_and_all', 'each' or 'subsets', "
        f"got {combos!r}")

