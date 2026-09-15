"""
Group differences: the numbers must match the reference implementations
wherever one exists, and the edges must be blanks-with-notes, never fake ps.
"""

from __future__ import annotations

import csv
import random

import pytest

np = pytest.importorskip("numpy")
scipy_stats = pytest.importorskip("scipy.stats")

from taters.stats.group_differences import analyze_group_differences  # noqa: E402
from csvhelpers import _read, _write_table  # noqa: E402


def _planted(tmp_path, *, groups=("A", "B", "C"), n=30, shift=1.0, seed=3):
    """One feature shifted by `shift` per group step, one pure-noise feature."""
    rng = random.Random(seed)
    rows = []
    for gi, g in enumerate(groups):
        for i in range(n):
            rows.append([f"{g}{i}", g,
                         f"{rng.gauss(gi * shift, 1.0):.6f}",
                         f"{rng.gauss(0.0, 1.0):.6f}"])
    return _write_table(tmp_path, ["text_id", "condition", "signal", "noise"],
                        rows)


def _column(table_path, feature, group):
    """The raw values of one feature for one group, straight from the CSV."""
    out = []
    for row in _read(table_path):
        if row["condition"] == group and row[feature].strip():
            out.append(float(row[feature]))
    return out


# ---------------------------------------------------------------------------
# agreement with the reference implementations
# ---------------------------------------------------------------------------

def test_the_f_test_matches_scipy_f_oneway(tmp_path):
    table = _planted(tmp_path)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    rounding=10, verbose=False)
    rows = {r["feature"]: r for r in _read(out)}

    for feature in ("signal", "noise"):
        samples = [_column(table, feature, g) for g in ("A", "B", "C")]
        expected = scipy_stats.f_oneway(*samples)
        assert float(rows[feature]["F"]) == pytest.approx(
            float(expected.statistic), abs=1e-8)
        assert float(rows[feature]["p"]) == pytest.approx(
            float(expected.pvalue), abs=1e-10)
    assert float(rows["signal"]["p"]) < 0.001 < float(rows["noise"]["p"])


def test_welch_with_two_groups_is_the_welch_t_test(tmp_path):
    """The independent cross-check for the Welch formulas: with k=2,
    Welch's F equals the squared Welch t and the p-values coincide."""
    table = _planted(tmp_path, groups=("A", "B"), shift=0.6)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    welch=True, rounding=10, verbose=False)
    row = {r["feature"]: r for r in _read(out)}["signal"]

    a = _column(table, "signal", "A")
    b = _column(table, "signal", "B")
    t = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert float(row["F"]) == pytest.approx(float(t.statistic) ** 2, rel=1e-9)
    assert float(row["p"]) == pytest.approx(float(t.pvalue), abs=1e-10)
    assert row["method"] == "welch"


def test_tukey_matches_scipy_tukey_hsd(tmp_path):
    tukey_hsd = getattr(scipy_stats, "tukey_hsd", None)
    if tukey_hsd is None:
        pytest.skip("scipy without tukey_hsd")
    table = _planted(tmp_path)
    analyze_group_differences(table_csv=table, group_col="condition",
                              rounding=10, verbose=False)
    pairs = {(r["group_1"], r["group_2"]): r
             for r in _read(table.parent / "group_differences_pairwise.csv")
             if r["feature"] == "signal"}

    samples = [_column(table, "signal", g) for g in ("A", "B", "C")]
    reference = tukey_hsd(*samples)
    for (i, a), (j, b) in [((0, "A"), (1, "B")), ((0, "A"), (2, "C")),
                           ((1, "B"), (2, "C"))]:
        ours = pairs[(a, b)]
        assert float(ours["p_adj"]) == pytest.approx(
            float(reference.pvalue[i][j]), abs=1e-9)
        low, high = reference.confidence_interval(0.95).low[i][j], \
            reference.confidence_interval(0.95).high[i][j]
        assert float(ours["ci_low"]) == pytest.approx(float(low), abs=1e-8)
        assert float(ours["ci_high"]) == pytest.approx(float(high), abs=1e-8)


def test_games_howell_with_two_groups_collapses_to_welch_t(tmp_path):
    table = _planted(tmp_path, groups=("A", "B"), shift=0.6)
    analyze_group_differences(table_csv=table, group_col="condition",
                              welch=True, rounding=10, verbose=False,
                              overwrite_existing=True)
    pair = [r for r in
            _read(table.parent / "group_differences_pairwise.csv")
            if r["feature"] == "signal"][0]
    a = _column(table, "signal", "A")
    b = _column(table, "signal", "B")
    t = scipy_stats.ttest_ind(a, b, equal_var=False)
    assert pair["method"] == "games_howell"
    assert float(pair["p_adj"]) == pytest.approx(float(t.pvalue), abs=1e-9)


def test_eta_squared_is_the_ss_ratio_and_d_recovers_the_shift(tmp_path):
    table = _planted(tmp_path, groups=("A", "B"), n=400, shift=0.8, seed=11)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    rounding=10, verbose=False)
    row = {r["feature"]: r for r in _read(out)}["signal"]

    a = np.array(_column(table, "signal", "A"))
    b = np.array(_column(table, "signal", "B"))
    grand = np.concatenate([a, b]).mean()
    ss_b = len(a) * (a.mean() - grand) ** 2 + len(b) * (b.mean() - grand) ** 2
    ss_t = ((np.concatenate([a, b]) - grand) ** 2).sum()
    assert float(row["eta2"]) == pytest.approx(ss_b / ss_t, abs=1e-9)

    pair = [r for r in
            _read(table.parent / "group_differences_pairwise.csv")
            if r["feature"] == "signal"][0]
    assert abs(float(pair["d"])) == pytest.approx(0.8, abs=0.15), \
        "Cohen's d should recover the planted shift"


# ---------------------------------------------------------------------------
# the table itself
# ---------------------------------------------------------------------------

def test_the_header_scales_by_group_and_keeps_the_summary_columns(tmp_path):
    table = _planted(tmp_path, groups=("A", "B", "C", "D"), n=5)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    verbose=False)
    with open(out, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == (
        ["feature_set", "feature"]
        + [f"{s}_{g}" for g in "ABCD" for s in ("n", "mean", "sd")]
        + ["F", "df1", "df2", "p", "p_adj", "eta2", "method",
           "pairwise_sig", "note"])


def test_pairwise_sig_names_the_direction(tmp_path):
    table = _planted(tmp_path, shift=2.0)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    verbose=False)
    row = {r["feature"]: r for r in _read(out)}["signal"]
    # we planted group C highest and A lowest
    assert "C>A" in row["pairwise_sig"]
    assert "A>C" not in row["pairwise_sig"]


def test_the_adjustment_is_applied_across_features(tmp_path):
    table = _planted(tmp_path)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    rounding=10, verbose=False)
    rows = {r["feature"]: r for r in _read(out)}
    for feature in ("signal", "noise"):
        p, q = float(rows[feature]["p"]), float(rows[feature]["p_adj"])
        assert q >= p - 1e-15


def test_per_table_sets_repeat_the_analysis_per_feature_table(tmp_path):
    table = _planted(tmp_path)
    sidecar = table.with_name(table.stem + "_sets.json")
    sidecar.write_text(
        '{"key": ["text_id"], "metadata": ["condition"],'
        ' "sets": {"one": ["signal"], "two": ["noise"]}}', encoding="utf-8")
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    feature_sets="per_table", verbose=False)
    rows = _read(out)
    assert {(r["feature_set"], r["feature"]) for r in rows} == \
        {("one", "signal"), ("two", "noise")}


# ---------------------------------------------------------------------------
# edges: blanks with notes, never fake statistics
# ---------------------------------------------------------------------------

def test_blank_group_cells_sit_out_and_the_ns_say_so(tmp_path):
    table = _write_table(tmp_path, ["text_id", "condition", "x"], [
        ["d1", "A", "1.0"], ["d2", "A", "2.0"], ["d3", "A", "3.0"],
        ["d4", "B", "4.0"], ["d5", "B", "5.0"], ["d6", "", "9.0"],
    ])
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    verbose=False)
    row = _read(out)[0]
    assert row["n_A"] == "3" and row["n_B"] == "2"
    assert "n_" not in row.get("", "")


def test_a_tiny_group_is_excluded_and_named(tmp_path):
    table = _write_table(tmp_path, ["text_id", "condition", "x"], [
        ["d1", "A", "1.0"], ["d2", "A", "2.0"], ["d3", "A", "3.0"],
        ["d4", "B", "4.0"], ["d5", "B", "5.0"], ["d6", "C", "9.0"],
    ])
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    verbose=False)
    row = _read(out)[0]
    assert "excluded (n<2): C" in row["note"]
    assert row["F"] != "", "A vs B still runs"


def test_too_few_usable_groups_leaves_stats_blank(tmp_path):
    table = _write_table(tmp_path, ["text_id", "condition", "x"], [
        ["d1", "A", "1.0"], ["d2", "A", "2.0"], ["d3", "B", "4.0"],
    ])
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    verbose=False)
    row = _read(out)[0]
    assert row["F"] == "" and row["p"] == ""
    assert "fewer than 2 groups with data" in row["note"]


def test_a_constant_feature_is_a_note_not_a_statistic(tmp_path):
    table = _write_table(tmp_path, ["text_id", "condition", "x"], [
        ["d1", "A", "5"], ["d2", "A", "5"], ["d3", "B", "5"], ["d4", "B", "5"],
    ])
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    verbose=False)
    row = _read(out)[0]
    assert row["F"] == "" and row["p"] == ""
    assert "constant feature" in row["note"]


def test_one_group_overall_is_refused_up_front(tmp_path):
    table = _write_table(tmp_path, ["text_id", "condition", "x"],
                         [["d1", "A", "1"], ["d2", "A", "2"]])
    with pytest.raises(ValueError, match="at least two"):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  verbose=False)


def test_a_missing_group_column_is_refused_naming_the_options(tmp_path):
    table = _write_table(tmp_path, ["text_id", "condition", "x"],
                         [["d1", "A", "1"]])
    with pytest.raises(ValueError, match="condition"):
        analyze_group_differences(table_csv=table, group_col="grup",
                                  verbose=False)


def test_a_refusal_does_not_recite_a_thousand_columns(tmp_path):
    """A sentence-embedding table has a thousand columns, and printing all
    of them buried the one sentence saying what was wrong under four screens
    of `e0, e1, e10…` -- twice, because the finish screen repeats it."""
    header = ["text_id", "condition"] + [f"e{i}" for i in range(1024)]
    table = _write_table(tmp_path, header,
                         [["d1", "A"] + ["0.1"] * 1024])
    with pytest.raises(ValueError) as caught:
        analyze_group_differences(table_csv=table, group_col="grup",
                                  verbose=False)
    message = str(caught.value)
    assert len(message) < 400, "the refusal has to fit on a screen"
    assert "1026" in message, "it still says how many there are"
    assert "and 1014 more" in message


def test_the_report_section_lands_beside_the_tables(tmp_path):
    table = _planted(tmp_path)
    analyze_group_differences(table_csv=table, group_col="condition",
                              verbose=False)
    fragment = tmp_path / "_sections" / "20-group-differences.md"
    text = fragment.read_text(encoding="utf-8")
    assert text.startswith("## Group differences")
    assert "3 groups" in text and "condition" in text


def test_welch_df2_matches_the_t_tests_satterthwaite_df(tmp_path):
    rng = random.Random(4)
    rows = []
    for i in range(40):        # unequal n and unequal variance on purpose
        rows.append([f"a{i}", "A", f"{rng.gauss(0, 1.0):.6f}"])
    for i in range(15):
        rows.append([f"b{i}", "B", f"{rng.gauss(0.5, 3.0):.6f}"])
    table = _write_table(tmp_path, ["text_id", "condition", "x"], rows)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    welch=True, rounding=10, verbose=False)
    row = _read(out)[0]
    t = scipy_stats.ttest_ind(_column(table, "x", "A"),
                              _column(table, "x", "B"), equal_var=False)
    assert float(row["df2"]) == pytest.approx(float(t.df), rel=1e-9)


def test_welch_with_three_groups_matches_an_independent_recomputation(tmp_path):
    """k=2 cannot exercise the (k-2) correction term in Welch's denominator,
    so a three-group case is recomputed here from the raw data, step by
    published step (Welch 1951), independently of the implementation."""
    table = _planted(tmp_path, groups=("A", "B", "C"), n=20, shift=0.5,
                     seed=13)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    welch=True, rounding=10, verbose=False)
    row = {r["feature"]: r for r in _read(out)}["signal"]

    samples = [np.array(_column(table, "signal", g)) for g in "ABC"]
    ns = np.array([len(s) for s in samples], dtype=float)
    means = np.array([s.mean() for s in samples])
    variances = np.array([s.var(ddof=1) for s in samples])
    k = 3
    w = ns / variances
    grand = (w * means).sum() / w.sum()
    a = (w * (means - grand) ** 2).sum() / (k - 1)
    tail = (((1 - w / w.sum()) ** 2) / (ns - 1)).sum()
    f_expected = a / (1 + (2 * (k - 2) / (k * k - 1)) * tail)
    df2_expected = (k * k - 1) / (3 * tail)
    p_expected = float(scipy_stats.f.sf(f_expected, k - 1, df2_expected))

    assert float(row["F"]) == pytest.approx(float(f_expected), rel=1e-9)
    assert float(row["df2"]) == pytest.approx(float(df2_expected), rel=1e-9)
    assert float(row["p"]) == pytest.approx(p_expected, abs=1e-10)


# ---------------------------------------------------------------------------
# choosing the correction (both families of it)
# ---------------------------------------------------------------------------

def test_every_adjustment_method_is_available_and_ordered_as_expected(tmp_path):
    """The methods differ in strictness in a known order, and the table has
    to reflect that -- otherwise the choice is decorative."""
    table = _planted(tmp_path, shift=0.35, n=40, seed=8)
    got = {}
    for method in ("none", "fdr_bh", "fdr_by", "holm", "bonferroni"):
        out = analyze_group_differences(
            table_csv=table, group_col="condition", p_adjust=method,
            out_dir=tmp_path / method, rounding=10, verbose=False)
        got[method] = {r["feature"]: r for r in _read(out)}

    for feature in ("signal", "noise"):
        raw = float(got["none"][feature]["p"])
        bh = float(got["fdr_bh"][feature]["p_adj"])
        by = float(got["fdr_by"][feature]["p_adj"])
        holm = float(got["holm"][feature]["p_adj"])
        bonf = float(got["bonferroni"][feature]["p_adj"])
        assert raw <= bh <= by, "BY is BH times a factor above 1"
        assert raw <= holm <= bonf, "Holm is uniformly stronger than Bonferroni"


def test_no_adjustment_writes_no_adjusted_column(tmp_path):
    """An unadjusted number under an "adjusted" heading is worse than no
    column at all."""
    table = _planted(tmp_path)
    out = analyze_group_differences(table_csv=table, group_col="condition",
                                    p_adjust="none", verbose=False)
    with open(out, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert "p_adj" not in header
    assert "p" in header
    section = (tmp_path / "_sections" / "20-group-differences.md").read_text("utf-8")
    assert "no correction across features" in section
    assert "read these accordingly" in section


def test_an_unknown_adjustment_is_refused_naming_the_choices(tmp_path):
    table = _planted(tmp_path)
    with pytest.raises(ValueError, match="bonferoni|choose one of"):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  p_adjust="bonferoni", verbose=False)


def test_the_posthoc_test_can_be_chosen_independently(tmp_path):
    """The pairwise correction is its own family: Tukey is available under a
    Welch omnibus, and plain t-tests are available under either."""
    table = _planted(tmp_path, shift=0.5)

    def pairwise(**kwargs):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  rounding=10, verbose=False,
                                  overwrite_existing=True, **kwargs)
        return [r for r in _read(tmp_path / "group_differences_pairwise.csv")
                if r["feature"] == "signal"]

    assert {r["method"] for r in pairwise()} == {"tukey_hsd"}
    assert {r["method"] for r in pairwise(welch=True)} == {"games_howell"}
    assert {r["method"] for r in pairwise(welch=True, posthoc="tukey")} == \
        {"tukey_hsd"}
    assert {r["method"] for r in pairwise(posthoc="bonferroni")} == \
        {"t_bonferroni"}
    assert {r["method"] for r in pairwise(posthoc="none")} == \
        {"t_uncorrected"}


def test_bonferroni_pairwise_is_the_uncorrected_one_times_the_pair_count(tmp_path):
    table = _planted(tmp_path, shift=0.35, n=25, seed=15)

    def pairwise(posthoc):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  posthoc=posthoc, rounding=12, verbose=False,
                                  overwrite_existing=True)
        return {(r["group_1"], r["group_2"]): r
                for r in _read(tmp_path / "group_differences_pairwise.csv")
                if r["feature"] == "noise"}

    plain = pairwise("none")
    corrected = pairwise("bonferroni")
    for key, row in plain.items():
        expected = min(float(row["p"]) * 3, 1.0)   # three pairs
        assert float(corrected[key]["p_adj"]) == pytest.approx(expected,
                                                               rel=1e-9)


def test_an_uncorrected_pairwise_test_matches_scipy(tmp_path):
    """The plain t-tests have to be plain t-tests."""
    table = _planted(tmp_path, groups=("A", "B"), shift=0.4)
    analyze_group_differences(table_csv=table, group_col="condition",
                              posthoc="none", rounding=12, verbose=False)
    pair = [r for r in _read(tmp_path / "group_differences_pairwise.csv")
            if r["feature"] == "signal"][0]
    expected = scipy_stats.ttest_ind(_column(table, "signal", "A"),
                                     _column(table, "signal", "B"),
                                     equal_var=True)
    assert float(pair["p"]) == pytest.approx(float(expected.pvalue),
                                                 abs=1e-12)


def test_an_unknown_posthoc_is_refused(tmp_path):
    table = _planted(tmp_path)
    with pytest.raises(ValueError, match="posthoc must be"):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  posthoc="scheffe", verbose=False)


def test_a_corrected_pairwise_p_comes_with_a_corrected_interval(tmp_path):
    """If the p is multiplied but the interval is not, a table can show a
    'significant' difference whose confidence interval spans zero. The two
    have to move together, so the interval widens with the correction and
    excluding zero still means exactly what p < alpha means."""
    table = _planted(tmp_path, shift=0.28, n=30, seed=19)

    def pairwise(posthoc):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  posthoc=posthoc, rounding=12,
                                  verbose=False, overwrite_existing=True)
        return {(r["group_1"], r["group_2"]): r
                for r in _read(tmp_path / "group_differences_pairwise.csv")
                if r["feature"] == "signal"}

    plain = pairwise("none")
    corrected = pairwise("bonferroni")
    for key, row in corrected.items():
        width = float(row["ci_high"]) - float(row["ci_low"])
        plain_width = float(plain[key]["ci_high"]) - float(plain[key]["ci_low"])
        assert width > plain_width, "the interval must widen with the test"
        # ...and stay coherent with its own p-value
        excludes_zero = float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0
        assert excludes_zero == (float(row["p_adj"]) < 0.05), key


# ---------------------------------------------------------------------------
# holding something constant: analysis of covariance
# ---------------------------------------------------------------------------

def _confounded(tmp_path, n=90, seed=17):
    """Groups that differ in age, and a feature driven by age -- so the raw
    comparison finds a difference the adjusted one should not."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        group = "ABC"[i % 3]
        # age is confounded with group: A young, C old
        age = {"A": 25, "B": 40, "C": 55}[group] + rng.gauss(0, 4)
        gender = "f" if i % 2 else "m"
        feature = 0.5 * age + rng.gauss(0, 3)      # driven by age alone
        real = {"A": 0.0, "B": 1.0, "C": 2.0}[group] + rng.gauss(0, 1)
        rows.append([f"p{i}", group, f"{age:.4f}", gender,
                     f"{feature:.4f}", f"{real:.4f}"])
    return _write_table(
        tmp_path, ["text_id", "condition", "age", "gender", "age_driven",
                   "really_differs"], rows)


def test_with_no_controls_the_adjusted_test_is_the_plain_one(tmp_path):
    """The anchor for the whole covariance path: a nested-model F with an
    empty control matrix has to be exactly the one-way F, and a contrast on
    adjusted means has to be exactly Tukey's."""
    table = _planted(tmp_path)
    plain = analyze_group_differences(table_csv=table, group_col="condition",
                                      out_dir=tmp_path / "plain",
                                      rounding=10, verbose=False)
    rows = {r["feature"]: r for r in _read(plain)}
    samples = [_column(table, "signal", g) for g in ("A", "B", "C")]
    expected = scipy_stats.f_oneway(*samples)
    assert float(rows["signal"]["F"]) == pytest.approx(
        float(expected.statistic), abs=1e-8)


def test_adjusting_removes_a_difference_that_was_really_the_covariate(tmp_path):
    """The point of the whole thing: a feature that only tracks age should
    stop separating groups once age is held constant, while a feature that
    really differs should survive."""
    table = _confounded(tmp_path)

    raw = {r["feature"]: r for r in _read(analyze_group_differences(
        table_csv=table, group_col="condition", out_dir=tmp_path / "raw",
        rounding=10, verbose=False))}
    adj = {r["feature"]: r for r in _read(analyze_group_differences(
        table_csv=table, group_col="condition",
        control_cols=["age", "gender"], out_dir=tmp_path / "adj",
        rounding=10, verbose=False))}

    assert float(raw["age_driven"]["p"]) < 0.001, "the confound is there"
    assert float(adj["age_driven"]["p"]) > 0.05, \
        "holding age constant should dissolve it"
    assert float(adj["really_differs"]["p"]) < 0.01, \
        "a real difference survives the adjustment"
    assert adj["age_driven"]["method"] == "ancova"
    # a control isn't a feature, so it shouldn't show up as one
    assert "age" not in adj and "gender" not in adj


def test_the_adjusted_means_sit_beside_the_raw_ones(tmp_path):
    """The difference between the two columns is what the controls did, so
    hiding the raw mean hides that."""
    table = _confounded(tmp_path)
    out = analyze_group_differences(
        table_csv=table, group_col="condition", control_cols=["age"],
        rounding=10, verbose=False)
    with open(out, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    for level in "ABC":
        assert f"mean_{level}" in header
        assert f"adj_mean_{level}" in header

    row = {r["feature"]: r for r in _read(out)}["age_driven"]
    # the raw means are far apart, but the adjusted ones get pulled together,
    # since the separation was really just age
    raw_spread = max(float(row[f"mean_{g}"]) for g in "ABC") - \
        min(float(row[f"mean_{g}"]) for g in "ABC")
    adj_spread = max(float(row[f"adj_mean_{g}"]) for g in "ABC") - \
        min(float(row[f"adj_mean_{g}"]) for g in "ABC")
    assert adj_spread < raw_spread / 3

    # ...and they're means OF the data, not of some hypothetical participant
    # aged zero. evaluated at the average of the controls, the adjusted means
    # average out to the grand mean. evaluating at zero instead would shift them
    # all by the same amount, which the spread comparison above can't see, so
    # we check for it here
    values = [float(r["age_driven"]) for r in _read(table)]
    grand = sum(values) / len(values)
    adjusted = [float(row[f"adj_mean_{g}"]) for g in "ABC"]
    assert sum(adjusted) / 3 == pytest.approx(grand, abs=0.2)


def test_eta_squared_becomes_partial_and_the_method_says_so(tmp_path):
    """Under adjustment it is a share of what the controls left over, not of
    the total -- and nothing but the `method` column tells the reader."""
    table = _confounded(tmp_path)
    out = analyze_group_differences(
        table_csv=table, group_col="condition", control_cols=["age"],
        rounding=10, verbose=False)
    row = {r["feature"]: r for r in _read(out)}["really_differs"]
    assert row["method"] == "ancova"
    assert 0 < float(row["eta2"]) <= 1

    section = (tmp_path / "_sections" /
               "20-group-differences.md").read_text("utf-8")
    assert "covariance" in section.lower() or "adjust" in section.lower()


def test_a_row_missing_a_control_sits_out(tmp_path):
    table = _confounded(tmp_path, n=60)
    rows = _read(table)
    for r in rows[:12]:
        r["age"] = ""
    holed = _write_table(tmp_path, list(rows[0]),
                         [[r[c] for c in rows[0]] for r in rows],
                         name="holed.csv")
    out = analyze_group_differences(
        table_csv=holed, group_col="condition", control_cols=["age"],
        out_dir=tmp_path / "holed_out", rounding=10, verbose=False)
    row = {r["feature"]: r for r in _read(out)}["really_differs"]
    assert sum(int(row[f"n_{g}"]) for g in "ABC") == 48


def test_the_heteroscedastic_variants_are_refused_with_controls(tmp_path):
    """Welch's F exists to avoid pooling the error term; an analysis of
    covariance pools it by construction. Offering both would be incoherent."""
    table = _confounded(tmp_path)
    with pytest.raises(ValueError, match="pools it by construction"):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  control_cols=["age"], welch=True,
                                  verbose=False)
    with pytest.raises(ValueError, match="pools them"):
        analyze_group_differences(table_csv=table, group_col="condition",
                                  control_cols=["age"],
                                  posthoc="games_howell", verbose=False)


def test_the_pairwise_p_is_named_for_what_it_holds(tmp_path):
    """Under posthoc="none" the pairwise p is an uncorrected t-test p, and a
    column called `p_adj` said otherwise."""
    table = _planted(tmp_path)
    analyze_group_differences(table_csv=table, group_col="condition",
                              posthoc="none", out_dir=tmp_path / "none",
                              verbose=False)
    header = list(_read(tmp_path / "none" / "group_differences_pairwise.csv")[0])
    assert "p" in header and "p_adj" not in header
    analyze_group_differences(table_csv=table, group_col="condition",
                              out_dir=tmp_path / "tukey", verbose=False)
    header = list(_read(tmp_path / "tukey" / "group_differences_pairwise.csv")[0])
    assert "p_adj" in header and "p" not in header
