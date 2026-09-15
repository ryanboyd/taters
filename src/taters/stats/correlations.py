"""
Correlations between features and outcomes, laid out for perusal.

Features as rows, outcomes as column blocks -- for each outcome its r, its
p, its FDR-adjusted p, and the pairwise N -- because that is how a researcher
actually reads this table: scan down an outcome's column for the big
coefficients, then check that the N behind each one is respectable.

Missing data is handled by **pairwise deletion**: each (feature, outcome)
cell uses exactly the rows where both values exist, and the ``_n`` column
says how many that was. NA is never zero, and a cell whose pairwise-complete
subset is too small (n < 3) or constant is left honestly blank -- with its
``_n`` still filled, so the blank explains itself.

Spearman coefficients rank each pair's complete subset (average ranks for
ties), matching ``scipy.stats.spearmanr`` on that subset; a global ranking
would let rows missing from one pair distort another's coefficients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from ._common import (MAX_MISSING, P_ADJUST_METHODS, adjust_pvalues, components_appendix, default_feature_cols, fmt, name_a_few, numeric_column, reduce_sets, resolve_feature_sets, split_rows, write_section, reusable)
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]


def _pair_r(x, y, *, spearman: bool, controls=None):
    """
    ``(r, p, n)`` for one feature-outcome pair, pairwise-complete.

    With ``controls``, this is the **partial** correlation: each variable is
    regressed on an intercept and the controls, and the correlation is taken
    between what is left over. That is what "the relationship after age and
    gender" means, and it is the same arithmetic as the unadjusted case with
    an empty control matrix -- so there is one code path and no chance of
    the two drifting.

    The degrees of freedom fall by one per control column, which is the
    part people forget: a partial correlation of .20 over 100 rows and six
    control columns is not tested against n - 2.

    Rows missing a control take no part, so ``n`` is the number of rows with
    the feature, the outcome AND every control present.
    """
    import numpy as np
    from scipy.stats import rankdata
    from scipy.stats import t as t_dist

    valid = ~np.isnan(x) & ~np.isnan(y)
    k = 0
    if controls is not None and getattr(controls, "shape", (0, 0))[1]:
        valid = valid & ~np.isnan(controls).any(axis=1)
        k = int(controls.shape[1])
    n = int(valid.sum())
    # make sure we've got enough rows: two points always correlate perfectly,
    # and a partial correlation needs room for the controls on top of that.
    if n < 3 + k:
        return float("nan"), float("nan"), n
    a, b = x[valid], y[valid]
    if spearman:
        a, b = rankdata(a), rankdata(b)
    if k:
        c = controls[valid]
        if spearman:
            # we rank the controls too, so that the whole computation lives
            # in ranks. for an indicator column, ranking is an affine map and
            # changes nothing; for a continuous control, it's what makes the
            # result robust to the same skew that ranking x and y handles.
            c = np.column_stack([rankdata(c[:, j]) for j in range(k)])
        design = np.column_stack([np.ones(n), c])
        beta_a = np.linalg.lstsq(design, a, rcond=None)[0]
        beta_b = np.linalg.lstsq(design, b, rcond=None)[0]
        a = a - design @ beta_a
        b = b - design @ beta_b
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return float("nan"), float("nan"), n
    r = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    r = max(-1.0, min(1.0, r))
    df = n - 2 - k
    if abs(r) == 1.0 or df < 1:
        return r, 0.0 if abs(r) == 1.0 else float("nan"), n
    t_value = r * (df / (1 - r * r)) ** 0.5
    p = float(2 * t_dist.sf(abs(t_value), df))
    return r, p, n


def analyze_correlations(
    *,
    table_csv: PathLike,
    outcome_cols: Sequence[str],
    feature_sets=None,
    control_cols: Sequence[str] = (),
    categorical_controls: Sequence[str] = (),
    pca: object = "off",
    pca_components: int = 0,
    pca_retain: Literal["parallel", "kaiser"] = "parallel",
    pca_rotation: bool = True,
    pca_max_missing: float = MAX_MISSING,
    split_col: Optional[str] = None,
    method: str = "pearson",
    p_adjust: str = "fdr_bh",
    out_dir: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
) -> Path:
    """
    Correlate every feature with every outcome column.

    Parameters
    ----------
    table_csv
        The assembled analysis table.
    outcome_cols
        The outcome columns (numeric). Each becomes a block of four columns
        in the output: ``<outcome>_r``, ``_p``, ``_p_adj``, ``_n``.
    feature_sets
        ``None`` for all features as one family, ``"per_table"`` to repeat
        per source feature table, or ``{name: [columns]}``.
    control_cols
        Columns to hold constant, turning every coefficient into a **partial**
        correlation: the relationship that is left once these have had their
        say. Continuous controls (age, word count) enter as themselves;
        categorical ones (gender, site) are expanded into indicators against
        a reference level, which the report names. The degrees of freedom
        fall by one per control column, and rows missing any control drop
        out -- so the ``_n`` columns can be smaller than they were, which is
        the honest cost of adjusting.
    categorical_controls
        Which of ``control_cols`` to treat as categories even though they
        look numeric -- a site coded 1/2/3, a condition coded 0/1/2. Left to
        itself, a column whose every value parses as a number is treated as
        continuous.
    pca : str or list of str, default="off"
        Analyze components instead of the raw measures. ``"off"`` uses the
        features as they are; ``"all"`` reduces every feature set; a list of
        set names reduces those and leaves the rest alone -- which is the
        case worth having, since a hundred dictionary categories are worth
        reducing and eight readability indices are not.

        Set per analysis, deliberately: raw variables read better in a
        correlation table, where each row is a measure you can name, while a
        ridge over four hundred collinear measures is what components are
        for. The loadings land beside these results and named after them,
        because a component means nothing without the table saying what
        loads on it.
    pca_components : int, default=0
        How many components to keep, when reducing. ``0`` decides by the
        Kaiser criterion -- a starting point, not an answer.
    pca_retain : {"parallel", "kaiser"}, default="parallel"
        How the component count is chosen when ``pca_components`` is 0:
        parallel analysis keeps a component while its eigenvalue beats what
        random data of the same size produce at that rank; the Kaiser rule
        keeps every eigenvalue above 1, which on a wide table is most of them.
    pca_rotation : bool, default=True
        Rotate the components (varimax) so each loads on a small cluster of
        features and is therefore nameable.
    pca_max_missing : float, default=0.5
        When reducing, set aside any feature missing for more than this
        fraction of the rows rather than letting it delete them -- the same
        rule, and the same default, the prediction steps apply to predictors.
        Of the rows that remain, the components are fitted on those with
        every kept feature present and a row still missing one is left
        unscored rather than guessed at; both counts are reported.
    split_col
        Run the whole analysis once per value of this column, labeling each
        result with it. Set to ``"source_col"`` when a spreadsheet's text
        columns were measured separately: a participant then has one row per
        column, and analyzing those together would count one person's
        several answers as several independent observations -- inflating the
        degrees of freedom, and every p-value with them.
    method : {"pearson", "spearman", "both"}, default="pearson"
        ``"both"`` writes two files and returns the Pearson one. The type is
        spelled out here because it is what the options screen reads: a
        closed set becomes a picker, and without one the user is left typing
        into a box with no way to learn that "spearman" is a word it knows.
    p_adjust : {"none", "fdr_bh", "fdr_by", "holm", "bonferroni"}, default="fdr_bh"
        How p-values are adjusted for the number of correlations tested:
        ``"fdr_by"`` is valid under any dependence between features, and
        stricter for it; ``"holm"`` and ``"bonferroni"`` control the
        family-wise error rate instead; ``"none"`` corrects nothing. The family is one feature set's
        whole feature-by-outcome grid, per method. With ``"none"`` no
        ``_p_adj`` columns are written -- an unadjusted number under an
        "adjusted" heading is worse than no column.
    out_dir
        Where the CSV(s) go; defaults to the analysis table's folder.
    overwrite_existing
        When False (default) and the results already exist, they are
        returned untouched instead of recomputed.
    rounding
        Decimal places in the output tables. Values too small for it (a p of
        3e-12) switch to significant digits rather than collapsing to 0.

    Returns
    -------
    Path
        ``correlations_<method>.csv`` (the Pearson file under ``"both"``).

    Notes
    -----
    A p-value printed as ``0`` means the tail underflowed double precision
    (roughly p < 1e-300), not that the probability is zero. The columns stay
    numeric so R, Excel and pandas read them as numbers; report such a value
    the way every statistics package does, as "p < .001".
    """
    import csv

    method = str(method).strip().lower()
    if method not in ("pearson", "spearman", "both"):
        raise ValueError(
            f"method must be 'pearson', 'spearman' or 'both', got {method!r}")
    outcome_cols = [str(c) for c in outcome_cols]
    if not outcome_cols:
        raise ValueError("outcome_cols is empty: nothing to correlate with")
    # a preset that says `p_adjust:` with nothing after it means "I didn't
    # set this", so we fall back to the default -- it does NOT mean "correct
    # nothing". only the explicit string "none" turns correction off.
    p_adjust = "fdr_bh" if p_adjust in (None, "") else str(p_adjust)
    if p_adjust not in P_ADJUST_METHODS:
        raise ValueError(
            f"unknown p-value adjustment {p_adjust!r}; choose one of "
            f"{sorted(P_ADJUST_METHODS)}")
    adjusting = p_adjust != "none"

    table_csv = Path(table_csv)
    folder = Path(out_dir) if out_dir else table_csv.parent
    folder.mkdir(parents=True, exist_ok=True)
    methods = ("pearson", "spearman") if method == "both" else (method,)
    primary = folder / f"correlations_{methods[0]}.csv"
    if reusable(primary, table_csv, overwrite_existing=overwrite_existing,
                verbose=verbose, what="the correlations"):
        if verbose:
            print(f"Correlations already exist; returning existing file: "
                  f"{primary}")
        return primary

    announce(on_progress, "reading the analysis table")
    from ._common import read_str_csv

    table = read_str_csv(table_csv, encoding=encoding)
    missing = [c for c in outcome_cols if c not in table.columns]
    if missing:
        raise ValueError(
            f"outcome column(s) not in the analysis table: "
            f"{name_a_few(missing)}. It has {len(table.columns)}: "
            f"{name_a_few(list(table.columns))}.")

    # a control isn't a feature. if we left it in, `age` would get correlated
    # with the outcome while age was being partialled out of it -- we'd get a
    # row of zeros dressed up as a finding.
    feature_cols = default_feature_cols(
        table, table_csv,
        exclude=tuple(outcome_cols) + tuple(str(c) for c in control_cols))
    sets = resolve_feature_sets(feature_sets, table_csv=table_csv,
                                feature_cols=feature_cols)

    from . import _controls

    control_matrix, control_names, control_notes, control_spec = \
        _controls.build(table, control_cols, categorical_controls)

    outcomes = {c: numeric_column(table[c].tolist(), column=c)
                for c in outcome_cols}
    features = {c: numeric_column(table[c].tolist(), column=c)
                for cols in sets.values() for c in cols}
    # swap in components for raw measures, if this analysis asked for them --
    # we do this per analysis and per feature set, since the right answer
    # differs between them.
    sets, features, pca_notes, _reductions = reduce_sets(
        sets, features, pca=pca, n_components=pca_components,
        rotation=pca_rotation, out_stem=folder / "correlations",
        encoding=encoding, rounding=rounding, verbose=verbose,
        max_missing=pca_max_missing, retain=pca_retain)


    lead = [split_col] if split_col else []
    subsets = list(split_rows(table, split_col))
    header = lead + ["feature_set", "feature"]
    for outcome in outcome_cols:
        header += [f"{outcome}_r", f"{outcome}_p"]
        if adjusting:
            header.append(f"{outcome}_p_adj")
        header.append(f"{outcome}_n")

    total = (sum(len(cols) for cols in sets.values()) * len(methods)
             * len(subsets))
    done = 0
    strongest = {}                 # method -> [(abs r, r, set, feature, outcome, n)]
    written = []
    for m in methods:
        rows = []
        for subset, mask in subsets:
          prefix = [subset] if split_col else []
          for set_name, cols in sets.items():
            cells = {}             # (feature, outcome) -> (r, p, n)
            for feature in cols:
                done += 1
                if on_progress is not None:
                    on_progress(done, total, f"correlating {feature}")
                for outcome in outcome_cols:
                    x = features[feature]
                    y = outcomes[outcome]
                    c = control_matrix
                    if mask is not None:
                        x, y = x[mask], y[mask]
                        c = c[mask] if c.shape[1] else c
                    cells[(feature, outcome)] = _pair_r(
                        x, y, spearman=(m == "spearman"), controls=c)
            # one correction family per (set, method). every feature x
            # outcome test goes in it, since those are the tests we're
            # scanning across.
            keys = [(f, o) for f in cols for o in outcome_cols]
            adjusted = dict(zip(
                keys, adjust_pvalues([cells[k][1] for k in keys], p_adjust),
                strict=True))
            for feature in cols:
                row = prefix + [set_name, feature]
                for outcome in outcome_cols:
                    r, p, n = cells[(feature, outcome)]
                    q = adjusted[(feature, outcome)]
                    row += [fmt(r, rounding), fmt(p, rounding)]
                    if adjusting:
                        row.append(fmt(q, rounding))
                    row.append(str(n))
                    if r == r:     # i.e. not NaN
                        strongest.setdefault(m, []).append(
                            (abs(r), r,
                             f"{set_name} @ {subset}" if subset else set_name,
                             feature, outcome, n, q))
                rows.append(row)
        path = folder / f"correlations_{m}.csv"
        with atomic_write(path, mode="w", encoding=encoding, newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        written.append(path)

    section = _section_md(
        methods=methods, outcome_cols=outcome_cols, sets=sets,
        strongest=strongest, files=[p.name for p in written],
        p_adjust=p_adjust, subsets=[s for s, _ in subsets if s],
        split_col=split_col, control_names=control_names,
        control_notes=control_notes)
    section += components_appendix(pca_notes)
    write_section(folder, "correlations", section)
    if verbose:
        print(f"[correlations] {len(features)} feature(s) × "
              f"{len(outcome_cols)} outcome(s), {', '.join(methods)} "
              f"-> {primary}")
    return primary


_ADJUST_WORDS = {
    "fdr_bh": "Benjamini-Hochberg correction per feature set",
    "fdr_by": "Benjamini-Yekutieli correction per feature set (valid under "
              "any dependence between features)",
    "holm": "Holm-Bonferroni correction per feature set",
    "bonferroni": "Bonferroni correction per feature set",
    "none": "**no correction** for the number of tests, which was asked for "
            "-- read these accordingly",
}


def _section_md(*, methods, outcome_cols, sets, strongest, files,
                p_adjust="fdr_bh", subsets=(), split_col=None,
                control_names=(), control_notes=()) -> str:
    n_features = sum(len(v) for v in sets.values())
    lines = ["## Correlations" + (" (partial)" if control_names else ""), "",
             f"Correlated {n_features} feature(s) with "
             f"{len(outcome_cols)} outcome(s) ({', '.join(outcome_cols)}), "
             f"{' and '.join(methods)}, pairwise deletion, "
             f"{_ADJUST_WORDS[p_adjust]}."]
    for m in methods:
        top = sorted(strongest.get(m, []), reverse=True)[:8]
        if not top:
            continue
        lines += ["", f"Strongest {m} correlations:", ""]
        for _, r, set_name, feature, outcome, n, q in top:
            flag = ("" if (q == q and q < 0.05)
                    else " (does not survive the correction)"
                    if p_adjust != "none" else "")
            lines.append(f"- `{feature}` × `{outcome}` ({set_name}): "
                         f"r = {r:.3f}, n = {n}{flag}")
    if control_names:
        lines += ["", f"These are **partial** correlations, holding "
                  f"{len(control_names)} column(s) constant: "
                  f"{', '.join('`' + c + '`' for c in control_names)}. Each "
                  f"one costs a degree of freedom, so a coefficient here is "
                  f"tested against n - 2 - {len(control_names)}, and rows "
                  f"missing any control take no part."]
        for note in control_notes:
            lines.append(f"- {note}")
    if subsets:
        lines += ["", f"Run separately for each `{split_col}` "
                  f"({', '.join(subsets)}), so one text's several answers are "
                  f"never counted as independent observations."]
    lines += ["", "Details: " + ", ".join(f"`{f}`" for f in files) + "."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_correlations,
    description='Correlate every feature with every outcome column of an analysis table.',
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
