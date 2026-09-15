"""
Correlations: per-cell agreement with scipy under pairwise deletion, the
outcomes-as-columns layout, and blanks that explain themselves.
"""

from __future__ import annotations

import csv
import random

import pytest

np = pytest.importorskip("numpy")
scipy_stats = pytest.importorskip("scipy.stats")

from taters.stats.correlations import analyze_correlations  # noqa: E402
from csvhelpers import _read, _write_table  # noqa: E402


def _fixture(tmp_path, *, n=60, seed=5):
    """Two outcomes, two features; feat_a tracks openness by construction,
    and both features carry planted missing cells."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        openness = rng.gauss(3.0, 1.0)
        neuro = rng.gauss(2.5, 1.0)
        feat_a = 0.8 * openness + rng.gauss(0, 0.4)
        feat_b = rng.gauss(0, 1.0)
        rows.append([
            f"d{i}",
            "" if i % 9 == 0 else f"{openness:.6f}",
            f"{neuro:.6f}",
            "" if i % 7 == 0 else f"{feat_a:.6f}",
            f"{feat_b:.6f}",
        ])
    return _write_table(
        tmp_path, ["text_id", "openness", "neuroticism", "feat_a", "feat_b"],
        rows)


def _pairwise(table_path, feature, outcome):
    xs, ys = [], []
    for row in _read(table_path):
        if row[feature].strip() and row[outcome].strip():
            xs.append(float(row[feature]))
            ys.append(float(row[outcome]))
    return np.array(xs), np.array(ys)


def _run(tmp_path, table, **kwargs):
    return analyze_correlations(
        table_csv=table, outcome_cols=["openness", "neuroticism"],
        rounding=10, verbose=False, **kwargs)


# ---------------------------------------------------------------------------
# agreement with scipy, under pairwise deletion
# ---------------------------------------------------------------------------

def test_pearson_cells_match_scipy_on_the_pairwise_subset(tmp_path):
    table = _fixture(tmp_path)
    out = _run(tmp_path, table)
    rows = {r["feature"]: r for r in _read(out)}

    for feature in ("feat_a", "feat_b"):
        for outcome in ("openness", "neuroticism"):
            xs, ys = _pairwise(table, feature, outcome)
            expected = scipy_stats.pearsonr(xs, ys)
            row = rows[feature]
            assert float(row[f"{outcome}_r"]) == pytest.approx(
                float(expected.statistic), abs=1e-10)
            assert float(row[f"{outcome}_p"]) == pytest.approx(
                float(expected.pvalue), abs=1e-10)
            assert int(row[f"{outcome}_n"]) == len(xs), \
                "the pairwise N is that cell's own complete count"
    # the way we built the data should show through: feat_a tracks openness,
    # feat_b doesn't
    assert float(rows["feat_a"]["openness_r"]) > 0.6
    assert abs(float(rows["feat_b"]["openness_r"])) < 0.3


def test_the_planted_missingness_gives_cells_different_ns(tmp_path):
    table = _fixture(tmp_path)
    out = _run(tmp_path, table)
    row = {r["feature"]: r for r in _read(out)}["feat_a"]
    assert int(row["openness_n"]) < int(row["neuroticism_n"]), \
        "openness and feat_a both have holes; neuroticism is complete"


def test_spearman_matches_scipy_with_ties(tmp_path):
    rng = random.Random(2)
    rows = []
    for i in range(50):
        x = rng.choice([1, 2, 2, 3, 4])          # we want ties in here
        y = x + rng.gauss(0, 1.0)
        rows.append([f"d{i}", f"{y:.6f}", str(x)])
    table = _write_table(tmp_path, ["text_id", "outcome", "feat"], rows)

    out = analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                               method="spearman", rounding=10, verbose=False)
    row = _read(out)[0]
    xs, ys = _pairwise(table, "feat", "outcome")
    expected = scipy_stats.spearmanr(xs, ys)
    assert float(row["outcome_r"]) == pytest.approx(
        float(expected.statistic), abs=1e-10)
    assert float(row["outcome_p"]) == pytest.approx(
        float(expected.pvalue), abs=1e-10)


# ---------------------------------------------------------------------------
# layout and files
# ---------------------------------------------------------------------------

def test_outcomes_are_column_blocks_in_order(tmp_path):
    table = _fixture(tmp_path)
    out = _run(tmp_path, table)
    with open(out, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert header == ["feature_set", "feature",
                      "openness_r", "openness_p", "openness_p_adj",
                      "openness_n",
                      "neuroticism_r", "neuroticism_p", "neuroticism_p_adj",
                      "neuroticism_n"]


def test_both_writes_two_files_and_returns_pearson(tmp_path):
    table = _fixture(tmp_path)
    out = _run(tmp_path, table, method="both")
    assert out.name == "correlations_pearson.csv"
    assert (tmp_path / "correlations_spearman.csv").is_file()


def test_the_correction_family_is_the_whole_grid_of_one_set(tmp_path):
    table = _fixture(tmp_path)
    out = _run(tmp_path, table)
    for row in _read(out):
        for outcome in ("openness", "neuroticism"):
            p = float(row[f"{outcome}_p"])
            q = float(row[f"{outcome}_p_adj"])
            assert q >= p - 1e-15


def test_the_correction_can_be_chosen_or_declined(tmp_path):
    table = _fixture(tmp_path)
    strict = {r["feature"]: r for r in
              _read(_run(tmp_path, table, p_adjust="bonferroni",
                         out_dir=tmp_path / "bonf"))}
    lenient = {r["feature"]: r for r in
               _read(_run(tmp_path, table, p_adjust="fdr_bh",
                          out_dir=tmp_path / "bh"))}
    for feature in strict:
        assert float(strict[feature]["openness_p_adj"]) >= \
            float(lenient[feature]["openness_p_adj"])

    plain = _run(tmp_path, table, p_adjust="none", out_dir=tmp_path / "raw")
    with open(plain, encoding="utf-8-sig", newline="") as fh:
        header = next(csv.reader(fh))
    assert not any(h.endswith("_p_adj") for h in header)
    section = (tmp_path / "raw" / "_sections" /
               "30-correlations.md").read_text("utf-8")
    assert "no correction" in section


def test_an_unknown_correlation_adjustment_is_refused(tmp_path):
    table = _fixture(tmp_path)
    with pytest.raises(ValueError, match="choose one of"):
        analyze_correlations(table_csv=table, outcome_cols=["openness"],
                             p_adjust="fdr", verbose=False)


def test_per_table_sets_show_up_as_the_first_column(tmp_path):
    table = _fixture(tmp_path)
    sidecar = table.with_name(table.stem + "_sets.json")
    sidecar.write_text(
        '{"key": ["text_id"], "metadata": ["openness", "neuroticism"],'
        ' "sets": {"alpha": ["feat_a"], "beta": ["feat_b"]}}',
        encoding="utf-8")
    out = _run(tmp_path, table, feature_sets="per_table")
    assert {(r["feature_set"], r["feature"]) for r in _read(out)} == \
        {("alpha", "feat_a"), ("beta", "feat_b")}


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------

def test_a_starved_cell_is_blank_but_its_n_is_not(tmp_path):
    table = _write_table(tmp_path, ["text_id", "outcome", "feat"], [
        ["d1", "1.0", "2.0"], ["d2", "2.0", ""], ["d3", "", "3.0"],
        ["d4", "3.0", "4.0"],
    ])
    out = analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                               verbose=False)
    row = _read(out)[0]
    assert row["outcome_r"] == "" and row["outcome_p"] == ""
    assert row["outcome_n"] == "2", "the blank explains itself"


def test_a_constant_pairwise_subset_is_blank(tmp_path):
    table = _write_table(tmp_path, ["text_id", "outcome", "feat"], [
        ["d1", "1.0", "5"], ["d2", "2.0", "5"], ["d3", "3.0", "5"],
        ["d4", "4.0", "5"],
    ])
    out = analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                               verbose=False)
    row = _read(out)[0]
    assert row["outcome_r"] == ""
    assert row["outcome_n"] == "4"


def test_missing_outcomes_and_empty_outcomes_are_refused(tmp_path):
    table = _fixture(tmp_path)
    with pytest.raises(ValueError, match="not in the analysis table"):
        analyze_correlations(table_csv=table, outcome_cols=["openess"],
                             verbose=False)
    with pytest.raises(ValueError, match="empty"):
        analyze_correlations(table_csv=table, outcome_cols=[], verbose=False)


def test_an_existing_output_is_returned_untouched(tmp_path):
    table = _fixture(tmp_path)
    out = _run(tmp_path, table)
    before = out.read_bytes()
    again = _run(tmp_path, table)
    assert again.read_bytes() == before


def test_the_report_section_names_the_strongest_correlation(tmp_path):
    table = _fixture(tmp_path)
    _run(tmp_path, table)
    text = (tmp_path / "_sections" / "30-correlations.md").read_text("utf-8")
    assert text.startswith("## Correlations")
    assert "`feat_a` × `openness`" in text


# ---------------------------------------------------------------------------
# partial correlations: what's left after age and gender
# ---------------------------------------------------------------------------

def test_one_continuous_control_matches_the_closed_form(tmp_path):
    """A partial correlation with a single control has a textbook formula --
    (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)) -- so the residualizing
    implementation has something exact to agree with."""
    rng = np.random.default_rng(5)
    n = 120
    age = rng.normal(40, 12, n)
    feat = 0.5 * age + rng.normal(0, 5, n)
    out = 0.4 * age + rng.normal(0, 5, n)
    table = _write_table(
        tmp_path, ["text_id", "age", "outcome", "feat"],
        [[f"d{i}", f"{age[i]:.6f}", f"{out[i]:.6f}", f"{feat[i]:.6f}"]
         for i in range(n)])

    rows = {r["feature"]: r for r in _read(analyze_correlations(
        table_csv=table, outcome_cols=["outcome"], control_cols=["age"],
        rounding=10, verbose=False))}
    assert "age" not in rows, "a control must not also be a feature"
    got = rows["feat"]

    r_xy = scipy_stats.pearsonr(feat, out).statistic
    r_xz = scipy_stats.pearsonr(feat, age).statistic
    r_yz = scipy_stats.pearsonr(out, age).statistic
    expected = ((r_xy - r_xz * r_yz)
                / ((1 - r_xz ** 2) ** 0.5 * (1 - r_yz ** 2) ** 0.5))
    # abs=1e-8 because residualizing by least squares and the closed form only
    # agree to floating-point noise, not to the last bit
    assert float(got["outcome_r"]) == pytest.approx(expected, abs=1e-8)

    # ...and controlling for the common cause ought to shrink it
    plain = {r["feature"]: r for r in _read(analyze_correlations(
        table_csv=table, outcome_cols=["outcome"], out_dir=tmp_path / "plain",
        rounding=10, verbose=False))}["feat"]
    assert abs(float(got["outcome_r"])) < abs(float(plain["outcome_r"]))


def test_the_partial_p_value_loses_a_degree_of_freedom_per_control(tmp_path):
    """The part people forget: a partial correlation is not tested against
    n - 2."""
    from scipy.stats import t as t_dist

    rng = np.random.default_rng(11)
    n = 60
    age = rng.normal(0, 1, n)
    site = rng.integers(0, 3, n)          # three levels, so two indicators
    feat = rng.normal(0, 1, n)
    out = 0.3 * feat + rng.normal(0, 1, n)
    table = _write_table(
        tmp_path, ["text_id", "age", "site", "outcome", "feat"],
        [[f"d{i}", f"{age[i]:.6f}", f"s{site[i]}", f"{out[i]:.6f}",
          f"{feat[i]:.6f}"] for i in range(n)])

    got = {r["feature"]: r for r in _read(analyze_correlations(
        table_csv=table, outcome_cols=["outcome"],
        control_cols=["age", "site"], rounding=10, verbose=False))}["feat"]

    r = float(got["outcome_r"])
    k = 3                                  # age + two site indicators
    df = n - 2 - k
    expected = float(2 * t_dist.sf(abs(r) * (df / (1 - r * r)) ** 0.5, df))
    assert float(got["outcome_p"]) == pytest.approx(expected, abs=1e-9)
    assert int(got["outcome_n"]) == n


def test_a_categorical_control_is_expanded_against_a_reference(tmp_path):
    """"female" is not one more than "male", so gender enters as indicators
    -- and the report has to name the level everything is measured against,
    because no reader can infer it from the numbers."""
    rng = np.random.default_rng(3)
    n = 80
    gender = ["f" if i % 2 else "m" for i in range(n)]
    shift = np.array([1.5 if g == "f" else 0.0 for g in gender])
    feat = shift + rng.normal(0, 1, n)
    out = shift + rng.normal(0, 1, n)
    table = _write_table(
        tmp_path, ["text_id", "gender", "outcome", "feat"],
        [[f"d{i}", gender[i], f"{out[i]:.6f}", f"{feat[i]:.6f}"]
         for i in range(n)])

    out_path = analyze_correlations(
        table_csv=table, outcome_cols=["outcome"], control_cols=["gender"],
        rounding=10, verbose=False)
    partial = float({r["feature"]: r
                     for r in _read(out_path)}["feat"]["outcome_r"])
    plain = float({r["feature"]: r for r in _read(analyze_correlations(
        table_csv=table, outcome_cols=["outcome"],
        out_dir=tmp_path / "plain", rounding=10,
        verbose=False))}["feat"]["outcome_r"])
    assert abs(partial) < abs(plain), "gender was the whole relationship"

    section = (tmp_path / "_sections" / "30-correlations.md").read_text("utf-8")
    assert "partial" in section.lower()
    # 'f' sorts first, so that's our reference and 'm' is the indicator
    assert "gender=m" in section
    assert "compared against 'f'" in section


def test_a_row_missing_a_control_takes_no_part(tmp_path):
    """Filling it in would invent the very thing being controlled for."""
    rng = np.random.default_rng(8)
    n = 50
    rows = []
    for i in range(n):
        age = "" if i < 10 else f"{rng.normal(40, 10):.4f}"
        rows.append([f"d{i}", age, f"{rng.normal():.6f}", f"{rng.normal():.6f}"])
    table = _write_table(tmp_path, ["text_id", "age", "outcome", "feat"], rows)

    got = {r["feature"]: r for r in _read(analyze_correlations(
        table_csv=table, outcome_cols=["outcome"], control_cols=["age"],
        rounding=10, verbose=False))}["feat"]
    assert int(got["outcome_n"]) == 40, "the ten blank ages are out"


def test_an_identifier_masquerading_as_a_control_is_refused(tmp_path):
    """Adjusting for a column with one level per row explains the outcome
    away rather than accounting for anything."""
    rows = [[f"d{i}", f"p{i}", f"{i}", f"{i * 2}"] for i in range(40)]
    table = _write_table(tmp_path, ["text_id", "pid", "outcome", "feat"], rows)
    with pytest.raises(ValueError, match="identifier than a control"):
        analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                             control_cols=["pid"], verbose=False)


def test_no_controls_is_exactly_the_unadjusted_path(tmp_path):
    """One code path for both, so the two cannot drift: an empty control
    matrix has to reproduce the plain correlation bit for bit."""
    table = _fixture(tmp_path)
    a = _run(tmp_path, table, out_dir=tmp_path / "a")
    b = _run(tmp_path, table, control_cols=(), out_dir=tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()


def test_a_blank_categorical_control_is_missing_not_a_category(tmp_path):
    """A row whose gender is unrecorded is not thereby male. Coding the
    blank as a zero in every indicator would quietly assign it to the
    reference level and keep it in the analysis."""
    rng = np.random.default_rng(6)
    n = 60
    rows = []
    for i in range(n):
        gender = "" if i < 15 else ("f" if i % 2 else "m")
        rows.append([f"d{i}", gender, f"{rng.normal():.6f}",
                     f"{rng.normal():.6f}"])
    table = _write_table(tmp_path, ["text_id", "gender", "outcome", "feat"],
                         rows)

    got = {r["feature"]: r for r in _read(analyze_correlations(
        table_csv=table, outcome_cols=["outcome"], control_cols=["gender"],
        rounding=10, verbose=False))}["feat"]
    assert int(got["outcome_n"]) == 45, "the fifteen blank genders are out"
