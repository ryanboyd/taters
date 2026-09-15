"""
The assemble step, the shared stats internals, and the report writer.

Everything here runs on tiny synthetic CSVs built in-test (the
test_stats_pca.py idiom); the statistical analyses have their own files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from taters.stats._common import bh_fdr, fmt, resolve_feature_sets
from taters.stats.assemble import assemble_analysis_table
from taters.stats.report import write_stats_report
from csvhelpers import _write


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _inputs(tmp_path):
    """Metadata + two feature tables; d3 is missing from cohesion.csv."""
    meta = _write(tmp_path / "metadata.csv",
                  ["text_id", "condition", "openness"],
                  [["d1", "A", "3.5"], ["d2", "B", "4.0"],
                   ["d3", "A", "2.5"], ["d4", "B", ""]])
    read = _write(tmp_path / "readability.csv",
                  ["text_id", "word_count", "flesch"],
                  [["d1", "30", "70.1"], ["d2", "10", "65.0"],
                   ["d3", "40", "50.5"], ["d4", "25", "60.0"]])
    coh = _write(tmp_path / "cohesion.csv",
                 ["text_id", "word_count", "overlap"],
                 [["d1", "30", "0.5"], ["d2", "10", "0.4"],
                  ["d4", "25", "0.6"]])
    return meta, read, coh


def _read_table(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# BH-FDR and cell formatting
# ---------------------------------------------------------------------------

def test_bh_fdr_matches_the_hand_computation():
    """The worked example every stats textbook uses: adjusted = p * m / rank,
    monotone from the top down."""
    adjusted = bh_fdr([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bh_fdr_leaves_nans_out_of_the_family():
    """A feature whose test could not run is not a test; counting it would
    make every real p harder to keep than it should be."""
    import math

    adjusted = bh_fdr([0.01, float("nan"), 0.04])
    assert math.isnan(adjusted[1])
    assert adjusted[0] == pytest.approx(0.02)   # m == 2, not 3
    assert adjusted[2] == pytest.approx(0.04)


def test_bh_fdr_is_capped_and_single_test_is_identity():
    assert bh_fdr([0.9, 0.95]) == pytest.approx([0.95, 0.95])
    assert bh_fdr([0.123]) == pytest.approx([0.123])


def test_small_p_values_do_not_collapse_to_zero():
    """"p = 0" is a claim no test can make."""
    assert fmt(3.2e-12, 4) != "0"
    assert "e-" in fmt(3.2e-12, 4)
    assert fmt(0.05, 4) == "0.05"
    assert fmt(float("nan"), 4) == ""
    assert fmt(3.0, 4) == "3"


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------

def test_the_join_is_inner_and_every_loss_is_accounted(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    out = assemble_analysis_table(
        feature_csvs=[read, coh], metadata_csv=meta,
        out_dir=tmp_path / "stats_results", verbose=False)

    rows = {r["text_id"] for r in _read_table(out)}
    assert rows == {"d1", "d2", "d4"}, "d3 is missing from cohesion.csv"

    manifest = json.loads((tmp_path / "stats_results" /
                           "assemble_manifest.json").read_text("utf-8"))
    assert manifest["rows_start"] == 4
    joined = {j["table"]: j for j in manifest["joins"]}
    assert joined["cohesion"]["rows_before"] == 4
    assert joined["cohesion"]["rows_after"] == 3
    assert manifest["rows_final"] == 3


def test_a_multi_column_key_joins_correctly(tmp_path):
    meta = _write(tmp_path / "m.csv", ["source", "speaker", "age"],
                  [["s1", "alice", "30"], ["s1", "bob", "40"]])
    feat = _write(tmp_path / "f.csv", ["source", "speaker", "pitch"],
                  [["s1", "bob", "120"], ["s1", "alice", "220"]])
    out = assemble_analysis_table(
        feature_csvs=[feat], metadata_csv=meta,
        key_cols=("source", "speaker"),
        out_dir=tmp_path / "out", verbose=False)
    rows = {(r["source"], r["speaker"]): r for r in _read_table(out)}
    assert rows[("s1", "alice")]["pitch"] == "220"
    assert rows[("s1", "bob")]["age"] == "40"


def test_duplicate_keys_are_refused_naming_file_and_key(tmp_path):
    meta, read, _ = _inputs(tmp_path)
    bad = _write(tmp_path / "dupes.csv", ["text_id", "x"],
                 [["d1", "1"], ["d1", "2"]])
    with pytest.raises(ValueError, match=r"dupes\.csv.*d1"):
        assemble_analysis_table(feature_csvs=[read, bad], metadata_csv=meta,
                                out_dir=tmp_path / "out", verbose=False)


def test_colliding_columns_are_renamed_in_every_file_that_has_them(tmp_path):
    """word_count appears in both feature files: BOTH copies get the stem
    prefix, so the outcome cannot depend on file order. Unique names stay
    bare."""
    meta, read, coh = _inputs(tmp_path)
    out = assemble_analysis_table(
        feature_csvs=[read, coh], metadata_csv=meta,
        out_dir=tmp_path / "stats_results", verbose=False)
    header = _read_table(out)[0].keys()
    assert "readability.word_count" in header
    assert "cohesion.word_count" in header
    assert "word_count" not in header
    assert "flesch" in header and "overlap" in header


def test_a_feature_column_clashing_with_metadata_is_renamed(tmp_path):
    meta = _write(tmp_path / "m.csv", ["text_id", "score"],
                  [["d1", "1"], ["d2", "2"]])
    feat = _write(tmp_path / "f.csv", ["text_id", "score"],
                  [["d1", "9"], ["d2", "8"]])
    out = assemble_analysis_table(
        feature_csvs=[feat], metadata_csv=meta,
        out_dir=tmp_path / "out", verbose=False)
    row = {r["text_id"]: r for r in _read_table(out)}["d1"]
    assert row["score"] == "1", "the metadata column keeps its name"
    assert row["f.score"] == "9"


def test_non_numeric_feature_columns_are_dropped_and_noted(tmp_path):
    meta, read, _ = _inputs(tmp_path)
    feat = _write(tmp_path / "emb.csv", ["text_id", "text", "dim_1"],
                  [["d1", "hello there", "0.1"], ["d2", "again", "0.2"],
                   ["d3", "x", "0.3"], ["d4", "y", "0.4"]])
    out = assemble_analysis_table(
        feature_csvs=[read, feat], metadata_csv=meta,
        out_dir=tmp_path / "stats_results", verbose=False)
    assert "text" not in _read_table(out)[0]
    manifest = json.loads((tmp_path / "stats_results" /
                           "assemble_manifest.json").read_text("utf-8"))
    assert manifest["dropped_columns"]["emb"] == ["text"]


def test_metadata_is_optional_and_features_only_works(tmp_path):
    _, read, coh = _inputs(tmp_path)
    out = assemble_analysis_table(
        feature_csvs=[read, coh], out_dir=tmp_path / "out", verbose=False)
    assert {r["text_id"] for r in _read_table(out)} == {"d1", "d2", "d4"}


def test_two_feature_files_with_the_same_stem_are_refused(tmp_path):
    meta, read, _ = _inputs(tmp_path)
    other = tmp_path / "elsewhere" / "readability.csv"
    _write(other, ["text_id", "z"], [["d1", "1"]])
    with pytest.raises(ValueError, match="readability"):
        assemble_analysis_table(feature_csvs=[read, other], metadata_csv=meta,
                                out_dir=tmp_path / "out", verbose=False)


def test_a_join_with_no_common_keys_errors_usefully(tmp_path):
    meta = _write(tmp_path / "m.csv", ["text_id", "g"], [["a", "1"]])
    feat = _write(tmp_path / "f.csv", ["text_id", "x"], [["b", "2"]])
    with pytest.raises(ValueError, match="no rows"):
        assemble_analysis_table(feature_csvs=[feat], metadata_csv=meta,
                                out_dir=tmp_path / "out", verbose=False)


def test_an_existing_table_is_returned_untouched(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    out_dir = tmp_path / "stats_results"
    out = assemble_analysis_table(feature_csvs=[read, coh], metadata_csv=meta,
                                  out_dir=out_dir, verbose=False)
    before = Path(out).read_bytes()
    again = assemble_analysis_table(feature_csvs=[read], metadata_csv=meta,
                                    out_dir=out_dir, verbose=False)
    assert Path(again).read_bytes() == before, "skip-if-exists"


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def _filtered(tmp_path, filters):
    meta, read, coh = _inputs(tmp_path)
    out = assemble_analysis_table(
        feature_csvs=[read, coh], metadata_csv=meta, filters=filters,
        out_dir=tmp_path / "stats_results", overwrite_existing=True,
        verbose=False)
    manifest = json.loads((tmp_path / "stats_results" /
                           "assemble_manifest.json").read_text("utf-8"))
    return {r["text_id"] for r in _read_table(out)}, manifest


def test_ordering_filters_keep_only_matching_rows(tmp_path):
    kept, manifest = _filtered(tmp_path, [["readability.word_count", ">=", 25]])
    assert kept == {"d1", "d4"}, "d2 has word_count 10"
    assert manifest["filters"][0]["removed"] == 1


def test_equality_membership_and_na_rules(tmp_path):
    kept, _ = _filtered(tmp_path, [["condition", "==", "A"]])
    assert kept == {"d1"}
    kept, _ = _filtered(tmp_path, [["condition", "in", ["A", "B"]]])
    assert kept == {"d1", "d2", "d4"}
    kept, _ = _filtered(tmp_path, [["condition", "not_in", ["A"]]])
    assert kept == {"d2", "d4"}
    # a blank cell fails EVERY filter, != and not_in included. not knowing the
    # openness isn't evidence that the row passes
    kept, manifest = _filtered(tmp_path, [["openness", "!=", "99"]])
    assert kept == {"d1", "d2"}, "d4's blank openness fails the filter"
    assert manifest["filters"][0]["removed"] == 1


def test_numbers_compare_numerically_not_textually(tmp_path):
    kept, _ = _filtered(tmp_path, [["openness", "==", 3.5]])
    assert kept == {"d1"}, '"3.5" equals 3.5 numerically'
    # "10" < "9" is true as text but false as numbers, so d2 shouldn't pass and
    # ALL three rows go. that trips the empty-table guard, and its accounting
    # is what proves the point
    with pytest.raises(ValueError, match="removed 3"):
        _filtered(tmp_path, [["readability.word_count", "<", 9]])


def test_bad_filters_are_refused_before_any_work(tmp_path):
    meta, read, coh = _inputs(tmp_path)

    def run(filters):
        return assemble_analysis_table(
            feature_csvs=[read, coh], metadata_csv=meta, filters=filters,
            out_dir=tmp_path / "out", overwrite_existing=True, verbose=False)

    with pytest.raises(ValueError, match="not in the analysis table"):
        run([["no_such_column", ">=", 1]])
    with pytest.raises(ValueError, match="orders numbers"):
        run([["condition", ">=", 1]])          # non-numeric column
    with pytest.raises(ValueError, match="orders numbers"):
        run([["flesch", ">=", "high"]])        # non-numeric value
    with pytest.raises(ValueError, match="not one of"):
        run([["flesch", "~", 1]])
    with pytest.raises(ValueError, match="list of values"):
        run([["condition", "in", "A"]])
    with pytest.raises(ValueError, match=r"\[column, operator, value\]"):
        run([["flesch", ">="]])


def test_filters_that_remove_everything_error_with_the_accounting(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    with pytest.raises(ValueError, match="removed every row"):
        assemble_analysis_table(
            feature_csvs=[read, coh], metadata_csv=meta,
            filters=[["flesch", ">", 1000]],
            out_dir=tmp_path / "out", verbose=False)


# ---------------------------------------------------------------------------
# the feature-set sidecar and its resolution
# ---------------------------------------------------------------------------

def test_the_sidecar_maps_sets_to_post_rename_columns(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    out = assemble_analysis_table(
        feature_csvs=[read, coh], metadata_csv=meta,
        out_dir=tmp_path / "stats_results", verbose=False)
    sets = json.loads(Path(out).with_name("analysis_table_sets.json")
                      .read_text("utf-8"))
    assert sets["key"] == ["text_id"]
    assert sets["metadata"] == ["condition", "openness"]
    assert sets["sets"]["readability"] == ["readability.word_count", "flesch"]
    assert sets["sets"]["cohesion"] == ["cohesion.word_count", "overlap"]


def test_feature_sets_resolution_covers_every_accepted_form(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    table = assemble_analysis_table(
        feature_csvs=[read, coh], metadata_csv=meta,
        out_dir=tmp_path / "stats_results", verbose=False)
    feature_cols = ["readability.word_count", "flesch",
                    "cohesion.word_count", "overlap"]

    combined = resolve_feature_sets(None, table_csv=table,
                                    feature_cols=feature_cols)
    assert combined == {"all": feature_cols}

    per_table = resolve_feature_sets("per_table", table_csv=table,
                                     feature_cols=feature_cols)
    assert set(per_table) == {"readability", "cohesion"}

    explicit = resolve_feature_sets({"mine": ["flesch"]}, table_csv=table,
                                    feature_cols=feature_cols)
    assert explicit == {"mine": ["flesch"]}

    with pytest.raises(ValueError, match="does not have"):
        resolve_feature_sets({"mine": ["ghost"]}, table_csv=table,
                             feature_cols=feature_cols)
    with pytest.raises(ValueError, match="empty"):
        resolve_feature_sets({"mine": []}, table_csv=table,
                             feature_cols=feature_cols)
    with pytest.raises(FileNotFoundError, match="re-run the assemble"):
        resolve_feature_sets("per_table", table_csv=tmp_path / "nowhere.csv",
                             feature_cols=feature_cols)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_the_report_concatenates_sections_in_analysis_order(tmp_path):
    from taters.stats._common import write_section

    stats_dir = tmp_path / "stats_results"
    write_section(stats_dir, "correlations", "## Correlations\ncorr body")
    write_section(stats_dir, "assemble", "## The analysis table\nrows in/out")

    out = write_stats_report(stats_dir=stats_dir, verbose=False)
    text = Path(out).read_text("utf-8")
    assert text.startswith("# Statistical results")
    assert text.index("analysis table") < text.index("Correlations"), \
        "assemble (slot 10) comes before correlations (slot 30)"


def test_the_report_refuses_when_nothing_ran(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run at least one analysis"):
        write_stats_report(stats_dir=tmp_path / "empty", verbose=False)


def test_the_assemble_section_carries_the_row_accounting(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    stats_dir = tmp_path / "stats_results"
    assemble_analysis_table(
        feature_csvs=[read, coh], metadata_csv=meta,
        filters=[["readability.word_count", ">=", 25]],
        out_dir=stats_dir, verbose=False)
    out = write_stats_report(stats_dir=stats_dir, verbose=False)
    text = Path(out).read_text("utf-8")
    assert "2 rows" in text
    assert "removed 1 row" in text


# ---------------------------------------------------------------------------
# choosing the multiple-comparison adjustment
# ---------------------------------------------------------------------------

def test_the_fdr_methods_match_scipy(tmp_path):
    """BH and BY have a reference implementation; agreeing with it to
    machine precision is cheaper than trusting my own arithmetic."""
    scipy_stats = pytest.importorskip("scipy.stats")
    fdc = getattr(scipy_stats, "false_discovery_control", None)
    if fdc is None:
        pytest.skip("scipy without false_discovery_control")
    from taters.stats._common import adjust_pvalues

    ps = [0.01, 0.04, 0.03, 0.005, 0.2, 0.9, 0.0001]
    for ours, theirs in (("fdr_bh", "bh"), ("fdr_by", "by")):
        assert adjust_pvalues(ps, ours) == pytest.approx(
            list(fdc(ps, method=theirs)), abs=1e-12)


def test_bonferroni_and_holm_are_the_textbook_ones():
    from taters.stats._common import adjust_pvalues

    ps = [0.01, 0.04, 0.03, 0.005]
    assert adjust_pvalues(ps, "bonferroni") == pytest.approx(
        [0.04, 0.16, 0.12, 0.02])
    # Holm: sorted .005 .01 .03 .04 times 4 3 2 1, then made non-decreasing
    assert adjust_pvalues(ps, "holm") == pytest.approx(
        [0.03, 0.06, 0.06, 0.02])


def test_no_adjustment_returns_the_raw_values():
    from taters.stats._common import adjust_pvalues

    ps = [0.01, 0.9, 0.04]
    assert adjust_pvalues(ps, "none") == pytest.approx(ps)


def test_the_methods_are_ordered_from_lenient_to_strict():
    """The property that makes the choice meaningful: for the same p-values,
    raw <= BH <= BY and raw <= Holm <= Bonferroni, always."""
    import random

    from taters.stats._common import adjust_pvalues

    rng = random.Random(4)
    for _ in range(20):
        ps = [rng.random() for _ in range(12)]
        raw = adjust_pvalues(ps, "none")
        bh = adjust_pvalues(ps, "fdr_bh")
        by = adjust_pvalues(ps, "fdr_by")
        holm = adjust_pvalues(ps, "holm")
        bonf = adjust_pvalues(ps, "bonferroni")
        for i in range(len(ps)):
            assert raw[i] <= bh[i] + 1e-12 <= by[i] + 1e-12
            assert raw[i] <= holm[i] + 1e-12 <= bonf[i] + 1e-12


def test_every_method_leaves_nans_out_of_the_family():
    import math

    from taters.stats._common import P_ADJUST_METHODS, adjust_pvalues

    for method in P_ADJUST_METHODS:
        adjusted = adjust_pvalues([0.01, float("nan"), 0.04], method)
        assert math.isnan(adjusted[1]), method
        if method == "bonferroni":
            # m == 2, not 3, since the untestable feature isn't a test
            assert adjusted[0] == pytest.approx(0.02)


def test_an_unknown_method_is_refused_naming_the_options():
    from taters.stats._common import adjust_pvalues

    with pytest.raises(ValueError, match="choose one of"):
        adjust_pvalues([0.1], "benjamini")


# ---------------------------------------------------------------------------
# keeping (or not keeping) the merged table
# ---------------------------------------------------------------------------

def test_the_merged_table_is_kept_by_default(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    stats_dir = tmp_path / "stats_results"
    assemble_analysis_table(feature_csvs=[read, coh], metadata_csv=meta,
                            out_dir=stats_dir, verbose=False)
    write_stats_report(stats_dir=stats_dir, verbose=False)
    assert (stats_dir / "analysis_table.csv").is_file()
    assert (stats_dir / "analysis_table_sets.json").is_file()


def test_declining_to_keep_it_removes_it_after_the_report(tmp_path):
    """Every analysis reads the table, so the only safe moment to remove it
    is after the last one -- which is why the report step does it."""
    meta, read, coh = _inputs(tmp_path)
    stats_dir = tmp_path / "stats_results"
    assemble_analysis_table(feature_csvs=[read, coh], metadata_csv=meta,
                            out_dir=stats_dir, keep_table=False, verbose=False)
    # it's there while the analyses would be running...
    assert (stats_dir / "analysis_table.csv").is_file()

    write_stats_report(stats_dir=stats_dir, keep_table=False, verbose=False)
    assert not (stats_dir / "analysis_table.csv").exists()
    assert not (stats_dir / "analysis_table_sets.json").exists()
    # ...but the row accounting stays: how many rows got analyzed is part of
    # the result, not part of the file we threw away
    assert (stats_dir / "assemble_manifest.json").is_file()
    assert (stats_dir / "report.md").is_file()


def test_a_missing_table_says_whether_that_was_a_choice(tmp_path):
    meta, read, coh = _inputs(tmp_path)
    stats_dir = tmp_path / "stats_results"
    assemble_analysis_table(feature_csvs=[read, coh], metadata_csv=meta,
                            out_dir=stats_dir, keep_table=False, verbose=False)
    manifest = json.loads((stats_dir / "assemble_manifest.json")
                          .read_text("utf-8"))
    assert manifest["kept"] is False
    report = write_stats_report(stats_dir=stats_dir, keep_table=False,
                                verbose=False)
    assert "was not kept (you asked for that)" in \
        Path(report).read_text("utf-8")


def test_an_unset_adjustment_is_never_read_as_no_adjustment():
    """`str(None).lower()` is the string "none", which is a legal method --
    so a null from a hand-edited preset would have switched correction OFF
    and reported raw p-values under an "adjusted" heading."""
    from taters.stats._common import adjust_pvalues

    with pytest.raises(ValueError, match="is None"):
        adjust_pvalues([0.01, 0.9], None)
    # correcting nothing is still allowed, but you have to ask for it by name
    assert adjust_pvalues([0.01, 0.9], "none") == pytest.approx([0.01, 0.9])


def test_the_analyses_read_an_unset_adjustment_as_their_default(tmp_path):
    """A preset that says `p_adjust:` with nothing after it means "I did not
    set this", which is the default -- not "correct nothing"."""
    import csv as _csv

    from taters.stats.correlations import analyze_correlations

    path = tmp_path / "analysis_table.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = _csv.writer(fh)
        w.writerow(["text_id", "outcome", "a", "b"])
        for i in range(20):
            w.writerow([f"d{i}", i, i * 2 + (i % 3), 20 - i])

    for unset in (None, ""):
        out = analyze_correlations(
            table_csv=path, outcome_cols=["outcome"], p_adjust=unset,
            out_dir=tmp_path / f"o{unset!r}", rounding=10, verbose=False)
        with open(out, encoding="utf-8-sig", newline="") as fh:
            header = next(_csv.reader(fh))
        assert "outcome_p_adj" in header, \
            "an unset adjustment must not silently become 'none'"


# ---------------------------------------------------------------------------
# count columns beside the measures: in the table, out of the features
# ---------------------------------------------------------------------------

from taters.helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings  # noqa: E402


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",), bookkeeping=("token_count",))
def toy_counted(*, analysis_csv, out_features_csv, overwrite_existing=False):
    """A stand-in analyzer that writes a count beside two measures. At module
    level, so a record's `call` can find it again the way the real ones are
    found."""
    return _write(out_features_csv, ["text_id", "token_count", "potato", "gravy"],
                  [["d1", "30", "0.1", "0.9"], ["d2", "10", "0.3", "0.7"],
                   ["d3", "40", "0.2", "0.8"], ["d4", "25", "0.4", "0.6"]])


def _counted_table(tmp_path, name="matrix.csv"):
    """A feature table whose analyzer declares a bookkeeping column, written
    through the recording decorator so the record beside it says so."""
    ready = _write(tmp_path / "ready.csv", ["text_id", "text"], [["d1", "x"]])
    return toy_counted(analysis_csv=ready, out_features_csv=tmp_path / name)


def test_a_declared_count_column_is_kept_beside_the_features_not_among_them(tmp_path):
    """The whole point: a ridge on a document-term matrix once learned from
    token_count. The analyzer says which columns are counts; the table keeps
    them (filterable, visible) and the feature sets do not."""
    meta, read, _coh = _inputs(tmp_path)
    matrix = _counted_table(tmp_path)
    out = assemble_analysis_table(
        feature_csvs=[str(matrix)], metadata_csv=str(meta),
        metadata_cols=["openness"], out_dir=tmp_path / "stats",
        overwrite_existing=True, verbose=False)
    sets = json.loads((tmp_path / "stats" / "analysis_table_sets.json").read_text())
    assert sets["sets"]["matrix"] == ["potato", "gravy"]
    assert sets["bookkeeping"] == {"matrix": ["token_count"]}
    assert "token_count" in _read_table(out)[0], "still in the table, to filter on"
    section = (tmp_path / "stats" / "_sections" / "10-assemble.md").read_text()
    assert "kept beside the features, not analyzed" in section
    assert "token_count" in section
    # and every analysis reads the sets, so none of them ever sees it
    from taters.stats._common import default_feature_cols
    assert "token_count" not in default_feature_cols(_read_table(out), out)


def test_the_count_columns_can_be_made_features_with_one_setting(tmp_path):
    meta, _read, _coh = _inputs(tmp_path)
    matrix = _counted_table(tmp_path)
    assemble_analysis_table(
        feature_csvs=[str(matrix)], metadata_csv=str(meta),
        metadata_cols=["openness"], out_dir=tmp_path / "stats",
        bookkeeping="features", overwrite_existing=True, verbose=False)
    sets = json.loads((tmp_path / "stats" / "analysis_table_sets.json").read_text())
    assert sets["sets"]["matrix"] == ["token_count", "potato", "gravy"]
    assert sets["bookkeeping"] == {}
    section = (tmp_path / "stats" / "_sections" / "10-assemble.md").read_text()
    assert "included as features, as asked" in section
    with pytest.raises(ValueError, match="aside"):
        assemble_analysis_table(feature_csvs=[str(matrix)], out_dir=tmp_path / "s2",
                                bookkeeping="maybe", verbose=False)


def test_a_recordless_table_can_still_name_its_count_columns(tmp_path):
    """A hand-made table has no record; the caller names the columns."""
    matrix = _write(tmp_path / "hand.csv", ["text_id", "n", "score"],
                    [["d1", "3", "0.5"], ["d2", "5", "0.1"]])
    assemble_analysis_table(feature_csvs=[str(matrix)], out_dir=tmp_path / "stats",
                            bookkeeping_cols=["n"], overwrite_existing=True,
                            verbose=False)
    sets = json.loads((tmp_path / "stats" / "analysis_table_sets.json").read_text())
    assert sets["sets"]["hand"] == ["score"] and sets["bookkeeping"] == {"hand": ["n"]}


def test_every_analyzer_declares_its_count_columns():
    """Pinned, so a new analyzer that writes a token_count beside its
    measures is caught here rather than by a model learning from it."""
    from taters.text.analyze_cohesion import analyze_cohesion
    from taters.text.analyze_parts_of_speech import analyze_parts_of_speech
    from taters.text.analyze_readability import analyze_readability
    from taters.text.analyze_with_archetypes import analyze_with_archetypes
    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries
    from taters.text.analyze_word_count import analyze_word_count
    from taters.text.build_doc_term_matrix import build_doc_term_matrix
    from taters.text.topic_model_mem import apply_mem_model, topic_model_mem

    declared = {fn.__name__: set(fn.__provenance__["bookkeeping"]) for fn in (
        build_doc_term_matrix, topic_model_mem, apply_mem_model,
        analyze_parts_of_speech, analyze_cohesion, analyze_with_dictionaries,
        analyze_with_archetypes, analyze_readability, analyze_word_count)}
    assert declared == {
        "build_doc_term_matrix": {"token_count"},
        "topic_model_mem": {"token_count"}, "apply_mem_model": {"token_count"},
        "analyze_parts_of_speech": {"token_count"},
        "analyze_cohesion": {"nwords"},
        "analyze_with_dictionaries": {"WC"}, "analyze_with_archetypes": {"WC"},
        "analyze_readability": {"lexicon_count", "sentence_count", "char_count",
                                "syllable_count", "difficult_words"},
        "analyze_word_count": {"word_count"},
    }


def test_a_table_from_before_the_declaration_is_read_through_its_analyzer(tmp_path):
    """The record names the function that wrote the table; the function's
    current declaration applies whatever the file's age. A real matrix
    extracted the day before kept token_count as a predictor."""
    from taters.helpers import provenance as pv

    matrix = _counted_table(tmp_path)
    rec = json.loads(pv.sidecar_path(matrix).read_text(encoding="utf-8"))
    del rec["bookkeeping"]
    pv.sidecar_path(matrix).write_text(json.dumps(rec), encoding="utf-8")
    assemble_analysis_table(feature_csvs=[str(matrix)], out_dir=tmp_path / "stats",
                            overwrite_existing=True, verbose=False)
    sets = json.loads((tmp_path / "stats" / "analysis_table_sets.json").read_text())
    assert sets["bookkeeping"] == {"matrix": ["token_count"]}
    assert sets["sets"]["matrix"] == ["potato", "gravy"]
