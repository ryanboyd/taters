"""
Group differences: one-way ANOVA per feature, with post-hoc tests that agree
with the omnibus test's assumptions.

The default is the classic decomposition -- pooled-variance F with
Tukey-Kramer post-hocs (Tukey's HSD generalized to unequal group sizes).
``welch=True`` switches BOTH tests: Welch's F (variances not pooled,
Satterthwaite degrees of freedom) with Games-Howell post-hocs. The pairing is
deliberate: Tukey's q statistic divides by a pooled error term, which is
incoherent if you chose Welch precisely because you distrust pooling;
Games-Howell is the standard heteroscedastic companion, built from per-pair
Welch degrees of freedom and the same studentized-range distribution.

Multiple comparisons are handled twice, at two levels: post-hoc p-values are
already family-adjusted within a feature (that is what the studentized range
does), and the omnibus p-values are corrected *across features* within each
feature set by the method you choose (``p_adjust``; Benjamini-Hochberg by
default) -- testing 160 cohesion measures at alpha=.05 without correction
would hand you eight "findings" by chance alone.

Missing data: listwise per feature (a row with a blank feature value or a
blank group cell sits out that feature's test); the per-group ``n_*`` columns
make the cost visible. NA is never zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from ._common import (MAX_MISSING, P_ADJUST_METHODS, adjust_pvalues, components_appendix, default_feature_cols, fmt, name_a_few, numeric_column, reduce_sets, resolve_feature_sets, split_rows, write_section, reusable)
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]


def _descriptives(values, labels, levels):
    """(n, mean, sd) per level, over the listwise-complete rows."""
    import numpy as np

    out = {}
    for level in levels:
        sub = values[labels == level]
        n = int(sub.size)
        mean = float(np.mean(sub)) if n else float("nan")
        sd = float(np.std(sub, ddof=1)) if n >= 2 else float("nan")
        out[level] = (n, mean, sd)
    return out


def _classic_anova(stats):
    """Pooled one-way ANOVA from per-group (n, mean, var). Returns
    (F, df1, df2, p, ms_within); the caller owns eta-squared."""
    import numpy as np
    from scipy.stats import f as f_dist

    ns = np.array([n for n, _, _ in stats], dtype=float)
    means = np.array([m for _, m, _ in stats], dtype=float)
    variances = np.array([v for _, _, v in stats], dtype=float)
    k = len(ns)
    total_n = ns.sum()
    grand = float((ns * means).sum() / total_n)
    ss_between = float((ns * (means - grand) ** 2).sum())
    ss_within = float(((ns - 1) * variances).sum())
    df1, df2 = k - 1, total_n - k
    ms_within = ss_within / df2
    if ms_within == 0.0:
        # every group is constant on the inside. if the means differ, an
        # infinite F is the honest statistic; identical means would've been
        # caught earlier on as a constant feature.
        return float("inf"), df1, df2, 0.0, 0.0
    f_value = (ss_between / df1) / ms_within
    return f_value, df1, df2, float(f_dist.sf(f_value, df1, df2)), ms_within


def _welch_anova(stats):
    """Welch's heteroscedastic F with Satterthwaite df2."""
    import numpy as np
    from scipy.stats import f as f_dist

    ns = np.array([n for n, _, _ in stats], dtype=float)
    means = np.array([m for _, m, _ in stats], dtype=float)
    variances = np.array([v for _, _, v in stats], dtype=float)
    if np.any(variances == 0.0):
        return float("inf"), len(ns) - 1, float("nan"), 0.0
    k = len(ns)
    w = ns / variances
    w_total = w.sum()
    grand = float((w * means).sum() / w_total)
    a = float((w * (means - grand) ** 2).sum()) / (k - 1)
    tail = float((((1 - w / w_total) ** 2) / (ns - 1)).sum())
    b = 1 + (2 * (k - 2) / (k * k - 1)) * tail
    df2 = (k * k - 1) / (3 * tail)
    f_value = a / b
    return f_value, k - 1, df2, float(f_dist.sf(f_value, k - 1, df2))


def _eta_squared(stats):
    """SS_between / SS_total from descriptives -- reported for both methods,
    because reviewers expect one effect size whatever the F flavor."""
    import numpy as np

    ns = np.array([n for n, _, _ in stats], dtype=float)
    means = np.array([m for _, m, _ in stats], dtype=float)
    variances = np.array([v for _, _, v in stats], dtype=float)
    grand = float((ns * means).sum() / ns.sum())
    ss_between = float((ns * (means - grand) ** 2).sum())
    ss_within = float(((ns - 1) * variances).sum())
    total = ss_between + ss_within
    return ss_between / total if total > 0 else float("nan")


def _ancova(values, labels, levels, controls):
    """
    A group comparison that holds the controls constant.

    Two nested least-squares fits and the F between them: one model with the
    controls alone, one with the controls plus the group indicators. The
    difference in error is what the groups explain *over and above* age and
    gender, which is the question -- not what they explain in a world where
    nobody has an age.

    This is a nested-model F test rather than an ANOVA on residuals, because
    the two only agree when the controls are orthogonal to the grouping, and
    in real data they never are: if the older participants are mostly in one
    condition, residualizing first quietly hands that difference to age.

    Returns a dict carrying the omnibus test, each group's adjusted mean --
    its predicted value at the mean of the controls, which is the number an
    ANCOVA table reports and the only one the F is actually about -- and the
    pieces a pairwise contrast needs afterwards.
    """
    import numpy as np
    from scipy.stats import f as f_dist

    n = values.size
    k_groups = len(levels)
    # indicators for every level but the first. with an intercept in the
    # design, all k of them together would be collinear.
    dummies = np.column_stack([(labels == lv).astype(float)
                               for lv in levels[1:]]) if k_groups > 1 else \
        np.zeros((n, 0))
    ones = np.ones((n, 1))
    reduced = np.hstack([ones, controls])
    full = np.hstack([ones, controls, dummies])

    def sse(design):
        beta, *_ = np.linalg.lstsq(design, values, rcond=None)
        resid = values - design @ beta
        return float(resid @ resid), beta

    sse_reduced, _ = sse(reduced)
    sse_full, beta_full = sse(full)
    df1 = dummies.shape[1]
    df2 = n - full.shape[1]
    if df1 < 1 or df2 < 1 or sse_full <= 0:
        return None
    f_value = ((sse_reduced - sse_full) / df1) / (sse_full / df2)
    p = float(f_dist.sf(f_value, df1, df2))
    # partial eta-squared: of the variance left over after the controls, how
    # much do the groups account for? note that this is NOT plain eta-squared
    # (a share of the total) -- that isn't what an adjusted model is telling us.
    partial_eta2 = ((sse_reduced - sse_full) / sse_reduced
                    if sse_reduced > 0 else float("nan"))

    # now, each group's predicted value with the controls held at their means.
    control_means = controls.mean(axis=0) if controls.shape[1] else np.zeros(0)
    adjusted = {}
    for i, level in enumerate(levels):
        row = np.hstack([[1.0], control_means,
                         np.array([1.0 if j == i - 1 else 0.0
                                   for j in range(df1)])])
        adjusted[level] = float(row @ beta_full)
    return {
        "f": f_value, "df1": df1, "df2": float(df2), "p": p,
        "eta2": partial_eta2, "adjusted": adjusted,
        "se_resid": float(np.sqrt(sse_full / df2)),
        # (X'X)^-1 and where the group indicators sit in it, so that later we
        # can give a contrast between two adjusted means a standard error.
        "xtx_inv": np.linalg.pinv(full.T @ full),
        "dummy_at": full.shape[1] - df1,
        "n_terms": full.shape[1],
    }


def _ancova_pair(fit, levels, a, b, *, n_pairs: int, posthoc: str,
                 alpha: float):
    """
    One pairwise contrast between adjusted means.

    The difference of two adjusted means, with the standard error the model
    implies -- ``se_resid * sqrt(c' (X'X)^-1 c)`` for the contrast vector
    ``c`` -- rather than a standard error computed from the raw group data,
    which would ignore the adjustment the difference is being made under.
    """
    import math

    import numpy as np
    from scipy.stats import studentized_range
    from scipy.stats import t as t_dist

    c = np.zeros(fit["n_terms"])
    for level, sign in ((a, 1.0), (b, -1.0)):
        index = levels.index(level)
        if index > 0:                    # the reference level has no column of its own
            c[fit["dummy_at"] + index - 1] = sign
    diff = fit["adjusted"][a] - fit["adjusted"][b]
    se = fit["se_resid"] * math.sqrt(float(c @ fit["xtx_inv"] @ c))
    if se <= 0:
        return diff, diff, diff, 0.0
    df = fit["df2"]
    if posthoc == "tukey":
        q = abs(diff) / (se / math.sqrt(2))
        p = float(studentized_range.sf(q, len(levels), df))
        half = float(studentized_range.ppf(1 - alpha, len(levels), df)) \
            * se / math.sqrt(2)
        return diff, diff - half, diff + half, p
    p = float(2 * t_dist.sf(abs(diff) / se, df))
    if posthoc == "bonferroni":
        p = min(p * n_pairs, 1.0)
        crit = float(t_dist.ppf(1 - alpha / (2 * n_pairs), df))
    else:
        crit = float(t_dist.ppf(1 - alpha / 2, df))
    return diff, diff - crit * se, diff + crit * se, p


def _pooled_ms(stats) -> float:
    """The pooled within-group mean square, from per-group (n, mean, var)."""
    total_n = sum(n for n, _, _ in stats)
    ss_within = sum((n - 1) * v for n, _, v in stats)
    return ss_within / (total_n - len(stats))


def _pairs(levels):
    return [(a, b) for i, a in enumerate(levels) for b in levels[i + 1:]]


def _tukey_pair(desc, a, b, *, k, ms_within, df_within, alpha):
    """Tukey-Kramer: pooled error, unequal-n standard error."""
    import math

    from scipy.stats import studentized_range

    (n1, m1, _), (n2, m2, _) = desc[a], desc[b]
    diff = m1 - m2
    se = math.sqrt(ms_within / 2 * (1 / n1 + 1 / n2))
    if se == 0.0:
        return diff, diff, diff, 0.0
    q = abs(diff) / se
    p = float(studentized_range.sf(q, k, df_within))
    half = float(studentized_range.ppf(1 - alpha, k, df_within)) * se
    return diff, diff - half, diff + half, p


def _games_howell_pair(desc, a, b, *, k, alpha):
    """Games-Howell: per-pair variances and Welch-Satterthwaite df."""
    import math

    from scipy.stats import studentized_range

    (n1, m1, s1), (n2, m2, s2) = desc[a], desc[b]
    v1, v2 = s1 * s1, s2 * s2
    diff = m1 - m2
    se_sq = v1 / n1 + v2 / n2
    if se_sq == 0.0:
        return diff, diff, diff, 0.0
    df = se_sq ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    q = abs(diff) / math.sqrt(se_sq / 2)
    p = float(studentized_range.sf(q, k, df))
    half = float(studentized_range.ppf(1 - alpha, k, df)) * math.sqrt(se_sq / 2)
    return diff, diff - half, diff + half, p


def _pairwise_t(desc, a, b, *, welch: bool, ms_within, df_within,
                n_pairs: int, bonferroni: bool, alpha: float):
    """
    A plain pairwise t-test, corrected by Bonferroni or not at all.

    Offered because "which adjustment" is the user's decision, and the
    studentized-range tests make it for them. Uncorrected pairwise tests
    over many groups inflate the error rate badly -- six pairs at .05 is a
    ~26% chance of at least one false positive -- so the choice is theirs to
    make deliberately, and the output names which one was used.
    """
    import math

    from scipy.stats import t as t_dist

    (n1, m1, s1), (n2, m2, s2) = desc[a], desc[b]
    diff = m1 - m2
    if welch:
        v1, v2 = s1 * s1, s2 * s2
        se_sq = v1 / n1 + v2 / n2
        df = (se_sq ** 2 / ((v1 / n1) ** 2 / (n1 - 1) +
                            (v2 / n2) ** 2 / (n2 - 1))) if se_sq else float("nan")
    else:
        se_sq = ms_within * (1 / n1 + 1 / n2)
        df = df_within
    if not se_sq:
        return diff, diff, diff, 0.0
    se = math.sqrt(se_sq)
    p = float(2 * t_dist.sf(abs(diff) / se, df))
    if bonferroni:
        p = min(p * n_pairs, 1.0)
        # the interval has to be corrected along with the test. otherwise, a
        # "significant" difference could sit inside a 95% CI that spans zero.
        crit = float(t_dist.ppf(1 - alpha / (2 * n_pairs), df))
    else:
        crit = float(t_dist.ppf(1 - alpha / 2, df))
    return diff, diff - crit * se, diff + crit * se, p


def _cohens_d(desc, a, b):
    import math

    (n1, m1, s1), (n2, m2, s2) = desc[a], desc[b]
    pooled = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2)
    if pooled == 0.0:
        return float("nan")
    return (m1 - m2) / math.sqrt(pooled)


def analyze_group_differences(
    *,
    table_csv: PathLike,
    group_col: str,
    feature_sets=None,
    control_cols: Sequence[str] = (),
    categorical_controls: Sequence[str] = (),
    pca: object = "off",
    pca_components: int = 0,
    pca_retain: Literal["parallel", "kaiser"] = "parallel",
    pca_rotation: bool = True,
    pca_max_missing: float = MAX_MISSING,
    split_col: Optional[str] = None,
    welch: bool = False,
    posthoc: str = "auto",
    p_adjust: str = "fdr_bh",
    alpha: float = 0.05,
    out_dir: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
) -> Path:
    """
    Test every feature for differences between the groups in one column.

    Parameters
    ----------
    table_csv
        The assembled analysis table (see the assemble step).
    group_col
        The column naming each row's group. Rows with a blank group sit out.
    feature_sets
        Which features to test: ``None`` for all of them as one family,
        ``"per_table"`` to repeat per source feature table (using the
        sidecar the assemble step wrote), or ``{name: [columns]}``.
    control_cols
        Columns to hold constant, which turns each comparison into an
        **analysis of covariance**: the groups are compared on what is left
        once these have had their say, and the table reports each group's
        adjusted mean -- its predicted value at the average age, the average
        word count -- rather than its raw one. Continuous controls enter as
        themselves; categorical ones become indicators against a reference
        level the report names. ``eta2`` then holds *partial* eta-squared,
        the share of what the controls left over, and ``method`` says
        ``ancova`` so the two are never confused.
    categorical_controls
        Which of ``control_cols`` to treat as categories despite looking
        numeric -- a site coded 1/2/3.
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
    welch
        False (default): classic pooled ANOVA. True: Welch's F, for when the
        groups' variances should not be pooled.
    posthoc : {"auto", "tukey", "games_howell", "bonferroni", "none"}, default="auto"
        Which pairwise comparison to run, and therefore how the pairwise
        p-values are corrected:

        - ``"auto"`` (default) -- Tukey-Kramer under a pooled ANOVA,
          Games-Howell under Welch. The pairing matters: Tukey's statistic
          divides by a pooled error term, which is incoherent once you have
          chosen Welch precisely because you distrust pooling.
        - ``"tukey"`` / ``"games_howell"`` -- force one of them.
        - ``"bonferroni"`` -- pairwise t-tests, Bonferroni-corrected.
        - ``"none"`` -- pairwise t-tests, uncorrected. Six pairs at .05 give
          a ~26% chance of at least one false positive, so this is a
          deliberate choice, not a shortcut.
    p_adjust : {"none", "fdr_bh", "fdr_by", "holm", "bonferroni"}, default="fdr_bh"
        How the *omnibus* p-values are adjusted across features, which is a
        separate family from the pairwise one. ``"fdr_by"`` is valid under
        any dependence between features, and stricter for it; ``"holm"`` and
        ``"bonferroni"`` control the family-wise error rate instead. With
        ``"none"`` no ``p_adj`` column is written -- an unadjusted number
        under an "adjusted" heading is worse than no column.
    alpha
        The threshold used for the ``pairwise_sig`` summary and the post-hoc
        confidence intervals. It does not gate what is written -- every test
        is in the tables.
    out_dir
        Where the two CSVs go; defaults to the analysis table's folder.
    overwrite_existing
        When False (default) and the results already exist, they are
        returned untouched instead of recomputed.
    rounding
        Decimal places in the output tables. Values too small for it (a p of
        3e-12) switch to significant digits rather than collapsing to 0.

    Returns
    -------
    Path
        ``group_differences.csv`` (one row per feature; the pairwise detail
        is in ``group_differences_pairwise.csv`` beside it).

    Notes
    -----
    A p-value printed as ``0`` means the tail underflowed double precision
    (roughly p < 1e-300), not that the probability is zero. The columns stay
    numeric so R, Excel and pandas read them as numbers; report such a value
    the way every statistics package does, as "p < .001".
    """
    import csv

    import numpy as np

    table_csv = Path(table_csv)
    folder = Path(out_dir) if out_dir else table_csv.parent
    folder.mkdir(parents=True, exist_ok=True)
    main_path = folder / "group_differences.csv"
    pair_path = folder / "group_differences_pairwise.csv"
    if reusable(main_path, table_csv, overwrite_existing=overwrite_existing,
                verbose=verbose, what="the group differences"):
        if verbose:
            print(f"Group differences already exist; returning existing "
                  f"file: {main_path}")
        return main_path

    announce(on_progress, "reading the analysis table")
    from ._common import read_str_csv

    table = read_str_csv(table_csv, encoding=encoding)
    if group_col not in table.columns:
        raise ValueError(
            f"group column {group_col!r} is not in the analysis table, "
            f"which has {len(table.columns)}: "
            f"{name_a_few(list(table.columns))}.")

    labels_all = np.array([str(v).strip() for v in table[group_col].tolist()])
    levels = sorted({v for v in labels_all if v})
    if len(levels) < 2:
        raise ValueError(
            f"column {group_col!r} holds "
            f"{'one group' if levels else 'no group labels'} "
            f"({levels or 'all blank'}); comparing groups needs at least two.")

    # a control isn't a feature. comparing groups on age while adjusting for
    # age would just hand us a row of zeros dressed up as a finding.
    feature_cols = default_feature_cols(
        table, table_csv,
        exclude=(group_col,) + tuple(str(c) for c in control_cols))
    sets = resolve_feature_sets(feature_sets, table_csv=table_csv,
                                feature_cols=feature_cols)

    from . import _controls

    control_matrix, control_names, control_notes, control_spec = \
        _controls.build(table, control_cols, categorical_controls)
    has_controls = bool(control_names)
    if has_controls and welch:
        raise ValueError(
            "welch=True cannot be combined with control columns: Welch's F "
            "exists precisely to avoid pooling the error term, and an "
            "analysis of covariance pools it by construction. Drop the "
            "controls, or drop welch.")

    method = "ancova" if has_controls else ("welch" if welch else "anova")
    # unset reads as the documented default, for the same reason as p_adjust
    # below: a null in a preset means "I didn't choose", and not choosing
    # ought to get you the coherent pairing (Tukey with a pooled F,
    # Games-Howell with Welch).
    posthoc = str(posthoc or "auto").strip().lower()
    if posthoc not in ("auto", "tukey", "games_howell", "bonferroni", "none"):
        raise ValueError(
            f"posthoc must be 'auto', 'tukey', 'games_howell', 'bonferroni' "
            f"or 'none', got {posthoc!r}")
    if posthoc == "auto":
        posthoc = "games_howell" if welch else "tukey"
    if has_controls and posthoc == "games_howell":
        raise ValueError(
            "posthoc='games_howell' cannot be combined with control "
            "columns, for the same reason as welch: it is built for "
            "unpooled variances and an analysis of covariance pools them.")
    posthoc_name = {"tukey": "tukey_hsd", "games_howell": "games_howell",
                    "bonferroni": "t_bonferroni", "none": "t_uncorrected"}[posthoc]
    # a preset that says `p_adjust:` with nothing after it means "I didn't
    # set this", so we fall back to the default -- it does NOT mean "correct
    # nothing". only the explicit string "none" turns correction off.
    p_adjust = "fdr_bh" if p_adjust in (None, "") else str(p_adjust)
    if p_adjust not in P_ADJUST_METHODS:
        raise ValueError(
            f"unknown p-value adjustment {p_adjust!r}; choose one of "
            f"{sorted(P_ADJUST_METHODS)}")
    adjusting = p_adjust != "none"
    desc_header = [f"{stat}_{level}" for level in levels
                   for stat in ("n", "mean", "sd")]
    if has_controls:
        # the adjusted means go next to the raw ones rather than replacing
        # them. the difference between the two columns is what the controls
        # did, and hiding the raw mean would hide that too.
        desc_header += [f"adj_mean_{level}" for level in levels]
    lead = [split_col] if split_col else []
    main_header = (lead + ["feature_set", "feature"] + desc_header +
                   ["F", "df1", "df2", "p"] + (["p_adj"] if adjusting else []) +
                   ["eta2", "method", "pairwise_sig", "note"])
    # we name this column for what's actually in it: under posthoc="none" the
    # pairwise p is an uncorrected t-test p, and calling it `p_adj` (like we
    # used to) said otherwise.
    pair_header = lead + ["feature_set", "feature", "group_1", "group_2",
                          "n_1", "n_2", "mean_diff", "ci_low", "ci_high",
                          "d", "p" if posthoc == "none" else "p_adj", "method"]

    main_rows, pair_rows = [], []
    total = sum(len(cols) for cols in sets.values())
    done = 0
    n_tested = 0
    top_effects = []           # (eta2, set, feature, sig_summary)

    # we parse the numeric columns once and mask them per subset rather than
    # re-parsing for each one. parsing is the expensive part, and the subsets
    # are all just views of the same rows anyway.
    columns = {c: numeric_column(table[c].tolist(), column=c)
               for cols in sets.values() for c in cols}
    # swap in components for raw measures, if this analysis asked for them --
    # we do this per analysis and per feature set, since the right answer
    # differs between them.
    sets, columns, pca_notes, _reductions = reduce_sets(
        sets, columns, pca=pca, n_components=pca_components,
        rotation=pca_rotation, out_stem=folder / "group_differences",
        encoding=encoding, rounding=rounding, verbose=verbose,
        max_missing=pca_max_missing, retain=pca_retain)

    subsets = list(split_rows(table, split_col))
    total *= len(subsets)

    for subset, subset_mask in subsets:
        prefix = [subset] if split_col else []
        for set_name, cols in sets.items():
            omnibus_ps = []
            set_rows = []
            for feature in cols:
                done += 1
                if on_progress is not None:
                    on_progress(done, total, f"testing {feature}")
                values_all = columns[feature]
                valid = ~np.isnan(values_all) & (labels_all != "")
                if has_controls:
                    valid = valid & ~np.isnan(control_matrix).any(axis=1)
                if subset_mask is not None:
                    valid = valid & subset_mask
                values, labels = values_all[valid], labels_all[valid]

                desc = _descriptives(values, labels, levels)
                desc_cells = []
                for level in levels:
                    n, mean, sd = desc[level]
                    desc_cells += [str(n), fmt(mean, rounding), fmt(sd, rounding)]

                blank_adjusted = [""] * len(levels) if has_controls else []
                usable = [lv for lv in levels if desc[lv][0] >= 2]
                note = ""
                excluded = [lv for lv in levels if lv not in usable]
                if excluded:
                    note = f"excluded (n<2): {', '.join(excluded)}"

                if len(usable) < 2:
                    omnibus_ps.append(float("nan"))
                    why = (note + "; " if note else "") + \
                        "fewer than 2 groups with data"
                    set_rows.append(prefix + [set_name, feature] + desc_cells
                                    + blank_adjusted
                                    + [""] * (6 if adjusting else 5)
                                    + [method, "", why])
                    continue

                stats = [(desc[lv][0], desc[lv][1], desc[lv][2] ** 2)
                         for lv in usable]
                if np.all(values[np.isin(labels, usable)] ==
                          values[np.isin(labels, usable)][0]):
                    omnibus_ps.append(float("nan"))
                    row = (prefix + [set_name, feature] + desc_cells
                           + blank_adjusted
                           + [""] * (6 if adjusting else 5)
                           + [method, "",
                              (note + "; " if note else "") + "constant feature"])
                    set_rows.append(row)
                    continue

                fit = None
                if has_controls:
                    keep = np.isin(labels, usable)
                    fit = _ancova(values[keep], labels[keep], usable,
                                  control_matrix[valid][keep])
                    if fit is None:
                        omnibus_ps.append(float("nan"))
                        set_rows.append(
                            prefix + [set_name, feature] + desc_cells
                            + blank_adjusted + [""] * (6 if adjusting else 5)
                            + [method, "",
                               (note + "; " if note else "")
                               + "too few rows for the controls"])
                        continue
                    f_value, df1, df2 = fit["f"], fit["df1"], fit["df2"]
                    p, eta2 = fit["p"], fit["eta2"]
                    ms_within, df_within = None, None
                    # now we fill in the adjusted-mean cells where the blanks were.
                    adjusted_cells = [
                        fmt(fit["adjusted"].get(lv, float("nan")), rounding)
                        for lv in levels]
                    desc_cells = desc_cells + adjusted_cells
                elif welch:
                    f_value, df1, df2, p = _welch_anova(stats)
                    ms_within, df_within = None, None
                    eta2 = _eta_squared(stats)
                else:
                    f_value, df1, df2, p, ms_within = _classic_anova(stats)
                    df_within = df2
                    eta2 = _eta_squared(stats)
                omnibus_ps.append(p)
                n_tested += 1

                k = len(usable)
                sig_parts = []
                all_pairs = _pairs(usable)
                for a, b in all_pairs:
                    if fit is not None:
                        diff, lo, hi, p_adj = _ancova_pair(
                            fit, usable, a, b, n_pairs=len(all_pairs),
                            posthoc=posthoc, alpha=alpha)
                    elif posthoc == "games_howell":
                        diff, lo, hi, p_adj = _games_howell_pair(
                            desc, a, b, k=k, alpha=alpha)
                    elif posthoc == "tukey":
                        # Tukey needs a pooled error term, and Welch's omnibus
                        # never computed one -- so we compute it here, so that
                        # the choice is there whichever F was asked for.
                        pooled = ms_within if ms_within is not None else \
                            _pooled_ms(stats)
                        pooled_df = df_within if df_within is not None else \
                            sum(n for n, _, _ in stats) - len(stats)
                        diff, lo, hi, p_adj = _tukey_pair(
                            desc, a, b, k=k, ms_within=pooled,
                            df_within=pooled_df, alpha=alpha)
                    else:
                        diff, lo, hi, p_adj = _pairwise_t(
                            desc, a, b, welch=welch,
                            ms_within=(ms_within if ms_within is not None
                                       else _pooled_ms(stats)),
                            df_within=(df_within if df_within is not None
                                       else sum(n for n, _, _ in stats) - len(stats)),
                            n_pairs=len(all_pairs),
                            bonferroni=(posthoc == "bonferroni"), alpha=alpha)
                    d = _cohens_d(desc, a, b)
                    pair_rows.append(prefix + [
                        set_name, feature, a, b, str(desc[a][0]),
                        str(desc[b][0]), fmt(diff, rounding), fmt(lo, rounding),
                        fmt(hi, rounding), fmt(d, rounding), fmt(p_adj, rounding),
                        posthoc_name])
                    if p_adj < alpha:
                        sig_parts.append(f"{a}>{b}" if diff > 0 else f"{b}>{a}")

                set_rows.append(
                    prefix + [set_name, feature] + desc_cells
                    + [fmt(f_value, rounding), fmt(df1, rounding),
                       fmt(df2, rounding), fmt(p, rounding)]
                    + ([""] if adjusting else [])   # p_adj; we fill this in below
                    + [fmt(eta2, rounding), method, "; ".join(sig_parts), note])
                top_effects.append(
                    (eta2, f"{set_name} @ {subset}" if subset else set_name,
                     feature, "; ".join(sig_parts) or "no pair survived"))

            # the adjustment family is the set. each feature set is one batch
            # of hypotheses, and correcting across sets would punish someone
            # for having asked a second question.
            if adjusting:
                adjusted = adjust_pvalues(omnibus_ps, p_adjust)
                adj_col = main_header.index("p_adj")
                for row, value in zip(set_rows, adjusted, strict=True):
                    row[adj_col] = fmt(value, rounding)
            main_rows.extend(set_rows)

    announce(on_progress, "writing the results")
    with atomic_write(main_path, mode="w", encoding=encoding,
                      newline="") as fh:
        w = csv.writer(fh)
        w.writerow(main_header)
        w.writerows(main_rows)
    with atomic_write(pair_path, mode="w", encoding=encoding,
                      newline="") as fh:
        w = csv.writer(fh)
        w.writerow(pair_header)
        w.writerows(pair_rows)

    verdict_col = main_header.index("p_adj" if adjusting else "p")
    survivors = sum(1 for row in main_rows
                    if row[verdict_col] != ""
                    and float(row[verdict_col]) < alpha)
    section = _section_md(
        group_col=group_col, levels=levels, method=method,
        posthoc=posthoc_name, n_tested=n_tested, survivors=survivors,
        alpha=alpha, top_effects=top_effects, sets=sets,
        p_adjust=p_adjust, subsets=[s for s, _ in subsets if s],
        split_col=split_col, control_names=control_names,
        control_notes=control_notes)
    section += components_appendix(pca_notes)
    write_section(folder, "group-differences", section)
    if verbose:
        print(f"[group-differences] {n_tested} feature(s) tested across "
              f"{len(levels)} group(s); {survivors} survive FDR "
              f"-> {main_path}")
    return main_path


_POSTHOC_WORDS = {
    "tukey_hsd": "Tukey-Kramer post-hocs",
    "games_howell": "Games-Howell post-hocs",
    "t_bonferroni": "Bonferroni-corrected pairwise t-tests",
    "t_uncorrected": "**uncorrected** pairwise t-tests",
}

_ADJUST_WORDS = {
    "fdr_bh": "Benjamini-Hochberg correction",
    "fdr_by": "Benjamini-Yekutieli correction (valid under any dependence)",
    "holm": "Holm-Bonferroni correction",
    "bonferroni": "Bonferroni correction",
    "none": "no correction across features",
}


def _section_md(*, group_col, levels, method, posthoc, n_tested, survivors,
                alpha, top_effects, sets, p_adjust="fdr_bh", subsets=(),
                split_col=None, control_names=(), control_notes=()) -> str:
    method_line = ("an analysis of covariance with " if method == "ancova"
                   else "Welch's ANOVA with " if method == "welch"
                   else "one-way ANOVA with ") + _POSTHOC_WORDS[posthoc]
    lines = ["## Group differences", "",
             f"Compared **{len(levels)} groups** ({', '.join(levels)}) from "
             f"`{group_col}` on {n_tested} feature(s) "
             f"across {len(sets)} feature set(s), using {method_line}.",
             "",
             f"**{survivors} feature(s)** differ between groups with "
             f"{_ADJUST_WORDS[p_adjust]} "
             f"(p < {alpha}"
             + ("" if p_adjust == "none" else ", per feature set") + ")."
             + ("" if p_adjust != "none" else
                " Nothing was corrected for testing many features at once, "
                "which was asked for -- read these accordingly.")]
    strongest = sorted(top_effects, reverse=True)[:10]
    if strongest:
        lines += ["", "Largest effects (by eta-squared):", ""]
        for eta2, set_name, feature, sig in strongest:
            lines.append(f"- `{feature}` ({set_name}): eta2 = {eta2:.3f} — {sig}")
    if control_names:
        lines += ["", f"Every comparison holds {len(control_names)} "
                  f"column(s) constant: "
                  f"{', '.join('`' + c + '`' for c in control_names)}. The "
                  f"groups are compared on what those left over, so the "
                  f"table reports each group's **adjusted mean** -- its "
                  f"predicted value at the average of the controls -- "
                  f"beside its raw one, and `eta2` is *partial* "
                  f"eta-squared: a share of what the controls left, not of "
                  f"the total."]
        for note in control_notes:
            lines.append(f"- {note}")
    if subsets:
        lines += ["", f"Run separately for each `{split_col}` "
                  f"({', '.join(subsets)}), so one text's several answers are "
                  f"never counted as independent observations. Each is its "
                  f"own correction family."]
    lines += ["", "Details: `group_differences.csv` (one row per feature) "
              "and `group_differences_pairwise.csv` (every pair)."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_group_differences,
    description='One-way group differences (ANOVA or Welch) with post-hoc tests over an analysis table.',
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
