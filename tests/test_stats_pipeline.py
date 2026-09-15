"""
The stats stage as a composed, running pipeline.

These tests build presets through the real composer and run them through the
real runner over tiny spreadsheets -- the whole path a wizard user takes,
minus the questions.
"""

from __future__ import annotations

import csv
import json
import random

import pytest

pytest.importorskip("scipy")

from taters.pipelines.run_pipeline import run_preset  # noqa: E402
from taters.ui.compose import ComposeError, compose  # noqa: E402
from csvhelpers import _read


def _spreadsheet(tmp_path, n=24, seed=9):
    """Three groups whose texts differ in verbosity, plus a numeric outcome."""
    rng = random.Random(seed)
    path = tmp_path / "study.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "condition", "openness", "text"])
        for i in range(n):
            group = "ABC"[i % 3]
            words = 8 + 6 * "ABC".index(group) + rng.randrange(3)
            text = " ".join(rng.choice(["potato", "gravy", "butter",
                                        "delightful", "fluffy"])
                            for _ in range(words))
            w.writerow([f"p{i}", group, f"{rng.gauss(3, 1):.4f}", text])
    return path


def _stats_preset(tmp_path, src, extra_vars=None, selected=None):
    return compose(
        selected or ["readability", "stats_group_differences",
                     "stats_correlations"],
        source="csv", input_path=str(src), text_cols=["text"],
        id_cols=["participant"], name="study stats",
        var_values={
            "stats_group_col": "condition",
            "stats_outcome_cols": ["openness"],
            "stats_meta_carry": ["condition", "openness"],
            **(extra_vars or {}),
        })


def test_a_composed_stats_preset_runs_end_to_end(tmp_path):
    src = _spreadsheet(tmp_path)
    preset = _stats_preset(tmp_path, src)
    work = tmp_path / "run"
    work.mkdir()

    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]

    stats = work / "stats_results"
    table = _read(stats / "analysis_table.csv")
    assert len(table) == 24
    assert {"text_id", "condition", "openness"} <= set(table[0])
    assert any(c.startswith("flesch") or "reading" in c for c in table[0]), \
        "readability features joined in"

    diffs = {r["feature"]: r for r in _read(stats / "group_differences.csv")}
    assert diffs, "the ANOVA table has rows"
    for row in diffs.values():
        assert row["method"] == "anova"

    corr = _read(stats / "correlations_pearson.csv")
    assert corr and "openness_r" in corr[0]

    report = (stats / "report.md").read_text(encoding="utf-8")
    assert report.startswith("# Statistical results")
    assert "## The analysis table" in report
    assert "## Group differences" in report
    assert "## Correlations" in report


def test_row_filters_flow_from_var_to_table(tmp_path):
    src = _spreadsheet(tmp_path)
    preset = _stats_preset(
        tmp_path, src,
        extra_vars={"stats_filters": [["condition", "!=", "C"]]})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]
    table = _read(work / "stats_results" / "analysis_table.csv")
    assert len(table) == 16 and {r["condition"] for r in table} == {"A", "B"}


def test_without_id_columns_the_two_gathers_still_agree(tmp_path):
    """Both gathers fall back to synthetic row_<n> ids over the same file in
    the same order -- the join must still find every row."""
    src = _spreadsheet(tmp_path)
    preset = compose(
        ["readability", "stats_correlations"],
        source="csv", input_path=str(src), text_cols=["text"],
        name="no ids",
        var_values={"stats_outcome_cols": ["openness"],
                    "stats_meta_carry": ["openness"]})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]
    table = _read(work / "stats_results" / "analysis_table.csv")
    assert len(table) == 24, "row_<n> ids matched across both gathers"


def test_grouped_runs_join_on_the_group_key_and_rename_outcomes(tmp_path):
    """Combining rows by condition: the metadata gather groups identically,
    outcomes become <col>_mean, and the wizard-side naming contract holds."""
    src = _spreadsheet(tmp_path)
    preset = compose(
        ["readability", "stats_correlations"],
        source="csv", input_path=str(src), text_cols=["text"],
        group_by=["condition"], level="group", name="grouped",
        var_values={"stats_outcome_cols": ["openness_mean"],
                    "stats_meta_agg": {"openness": "mean"}})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]
    table = _read(work / "stats_results" / "analysis_table.csv")
    assert len(table) == 3, "one row per condition"
    assert "openness_mean" in table[0]
    corr = _read(work / "stats_results" / "correlations_pearson.csv")
    assert "openness_mean_r" in corr[0]


# ---------------------------------------------------------------------------
# compose-level wiring
# ---------------------------------------------------------------------------

def test_the_assemble_step_receives_the_selected_feature_tables(tmp_path):
    preset = compose(
        ["readability", "lexical_richness", "stats_group_differences"],
        source="csv", input_path="x.csv", text_cols=["text"],
        var_values={"stats_group_col": "g"})
    assemble = [s for s in preset["steps"]
                if s["call"] == "potato.stats.assemble_analysis_table"][0]
    assert assemble["with"]["feature_csvs"] == [
        "{{readability_features}}", "{{lexrich_features}}"]


def test_a_wizard_narrowing_of_tables_beats_the_default(tmp_path):
    preset = compose(
        ["readability", "lexical_richness", "stats_group_differences"],
        source="csv", input_path="x.csv", text_cols=["text"],
        overrides={"stats_assemble":
                   {"feature_csvs": ["{{readability_features}}"]}},
        var_values={"stats_group_col": "g"})
    assemble = [s for s in preset["steps"]
                if s["call"] == "potato.stats.assemble_analysis_table"][0]
    assert assemble["with"]["feature_csvs"] == ["{{readability_features}}"]


def test_stats_with_no_feature_steps_is_refused(tmp_path):
    with pytest.raises(ComposeError, match="at least one feature table"):
        compose(["stats_group_differences"], source="csv",
                input_path="x.csv", text_cols=["text"])


def test_stats_on_a_non_csv_source_is_refused(tmp_path):
    with pytest.raises(ComposeError):
        compose(["readability", "stats_group_differences"], source="txt_dir",
                input_path="docs")


def test_the_metadata_gather_mirrors_the_text_gathers_identity(tmp_path):
    ungrouped = compose(
        ["readability", "stats_correlations"], source="csv",
        input_path="x.csv", text_cols=["text"], id_cols=["participant"],
        var_values={"stats_outcome_cols": ["openness"]})
    meta = [s for s in ungrouped["steps"]
            if s.get("save_as") == "stats_metadata"][0]
    assert meta["with"]["id_cols"] == ["participant"]
    assert "group_by" not in meta["with"]
    assert meta["with"]["text_cols"] == []

    grouped = compose(
        ["readability", "stats_correlations"], source="csv",
        input_path="x.csv", text_cols=["text"], group_by=["condition"],
        level="group",
        var_values={"stats_outcome_cols": ["openness_mean"]})
    meta = [s for s in grouped["steps"]
            if s.get("save_as") == "stats_metadata"][0]
    assert meta["with"]["group_by"] == ["condition"]


def test_the_report_runs_last(tmp_path):
    preset = _stats_preset(tmp_path, tmp_path / "x.csv")
    calls = [s["call"] for s in preset["steps"]]
    assert calls[-1] == "potato.stats.write_stats_report"
    assert calls.index("potato.stats.assemble_analysis_table") \
        > calls.index("potato.text.analyze_readability")


def _scored_spreadsheet(tmp_path, name, n=80, seed=4):
    """Texts whose length genuinely predicts the outcome."""
    rng = random.Random(seed)
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "condition", "openness", "text"])
        for i in range(n):
            group = "ABC"[i % 3]
            words = 12 + 5 * "ABC".index(group) + rng.randrange(4)
            text = " ".join(rng.choice(["potato", "gravy", "butter",
                                        "delightful", "considerable"])
                            for _ in range(words))
            openness = 2.0 + 0.12 * words + rng.gauss(0, 0.4)
            w.writerow([f"p{i}", group, f"{openness:.4f}", text])
    return path


def test_a_ridge_model_fitted_in_one_run_scores_another(tmp_path):
    """The whole point of saving a model: fit on study A, score study B --
    standardized against A's training sample, so the scores are comparable."""
    src_a = _scored_spreadsheet(tmp_path, "study_a.csv", seed=4)
    preset = compose(
        ["readability", "stats_ridge_fit"],
        source="csv", input_path=str(src_a), text_cols=["text"],
        id_cols=["participant"], name="fit run",
        var_values={"stats_outcome_cols": ["openness"],
                    "stats_meta_carry": ["openness"]})
    work_a = tmp_path / "run_a"
    work_a.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work_a,
                          out_manifest=work_a / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]

    metrics = _read(work_a / "stats_results" / "ridge_cv_metrics.csv")[0]
    assert float(metrics["cv_r2"]) > 0.5, "length really does predict here"
    model = work_a / "stats_results" / "models" / "ridge__all__openness.json"
    assert model.is_file()

    # now a second study, scored with the first study's model. this goes through
    # the feature-extraction stage, which is where scoring belongs: what comes
    # back is a column per text, and that's a feature, not a result
    src_b = _scored_spreadsheet(tmp_path, "study_b.csv", n=40, seed=77)
    preset_b = compose(
        ["readability", "score_with_model"],
        source="csv", input_path=str(src_b), text_cols=["text"],
        id_cols=["participant"], name="apply run",
        var_values={"model_path": str(model)})
    work_b = tmp_path / "run_b"
    work_b.mkdir()
    manifest_b = run_preset(preset_b, workers=1, work_dir=work_b,
                            out_manifest=work_b / "run_manifest.json",
                            verbose=False)
    assert not manifest_b["errors"], manifest_b["errors"]

    scored = _read(work_b / "features" / "model_scores.csv")
    assert len(scored) == 40
    assert all(r["pred_openness"] for r in scored)
    # the predictions have to track the real thing, otherwise the model didn't
    # travel (this is the check that catches a standardization mistake).
    # we read the true value from the source spreadsheet rather than carrying it
    # through the run: scoring a new dataset is exactly the case where the
    # outcome isn't there, so the pipeline had better not need it
    import csv as _csv
    with src_b.open("r", encoding="utf-8-sig", newline="") as fh:
        truth = {r["participant"]: float(r["openness"])
                 for r in _csv.DictReader(fh)}
    import statistics
    predicted = [float(r["pred_openness"]) for r in scored]
    actual = [truth[r["text_id"]] for r in scored]
    assert statistics.correlation(predicted, actual) > 0.5


def test_the_ridge_model_is_a_valid_library_asset(tmp_path):
    """It has to survive the library's import gate, or a user could never
    move it between machines."""
    from taters.helpers.library import KINDS, asset_problem

    src = _scored_spreadsheet(tmp_path, "study.csv")
    preset = compose(
        ["readability", "stats_ridge_fit"],
        source="csv", input_path=str(src), text_cols=["text"],
        id_cols=["participant"], name="fit run",
        var_values={"stats_outcome_cols": ["openness"],
                    "stats_meta_carry": ["openness"]})
    work = tmp_path / "run"
    work.mkdir()
    run_preset(preset, workers=1, work_dir=work,
               out_manifest=work / "run_manifest.json", verbose=False)

    model = work / "stats_results" / "models" / "ridge__all__openness.json"
    assert asset_problem(model, KINDS["models"]) == ""

    # ...and a model of some other kind gets turned away, with an explanation
    other = tmp_path / "impostor.json"
    other.write_text('{"kind": "taters-pca-model", "format": 1}',
                     encoding="utf-8")
    problem = asset_problem(other, KINDS["models"])
    assert "taters-pca-model" in problem


def test_a_word_count_filter_runs_end_to_end(tmp_path):
    """The user's own example: analyze everything, but leave out anything
    under 25 words -- with no step selected that would otherwise count."""
    rng = random.Random(11)
    src = tmp_path / "study.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant", "condition", "text"])
        for i in range(24):
            # every third response is a curt one, and the filter needs to drop it
            words = 4 if i % 3 == 0 else 30 + rng.randrange(6)
            w.writerow([f"p{i}", "ABC"[i % 3],
                        " ".join(rng.choice(["potato", "gravy", "butter"])
                                 for _ in range(words))])

    preset = compose(
        ["lexical_richness", "word_count", "stats_group_differences"],
        source="csv", input_path=str(src), text_cols=["text"],
        id_cols=["participant"], name="wc filter",
        var_values={"stats_group_col": "condition",
                    "stats_meta_carry": ["condition"],
                    "stats_filters": [["word_count", ">=", 25]]})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]

    counts = {r["text_id"]: int(r["word_count"])
              for r in _read(work / "features" / "word_count.csv")}
    assert min(counts.values()) == 4 and max(counts.values()) >= 30

    table = _read(work / "stats_results" / "analysis_table.csv")
    assert len(table) == 16, "the eight four-word rows are gone"
    assert all(int(r["word_count"]) >= 25 for r in table)

    report = (work / "stats_results" / "report.md").read_text(encoding="utf-8")
    assert "word_count >= 25" in report and "removed 8 row(s)" in report


def test_separate_text_columns_are_analyzed_one_column_at_a_time(tmp_path):
    """A survey with two open-ended questions, measured separately. Each
    participant contributes one row per question, so each question gets its
    own analysis -- rows stay independent, and the results say which
    question they describe."""
    rng = random.Random(8)
    src = tmp_path / "survey.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "TIPI_Open", "answer", "followup"])
        for i in range(30):
            w.writerow([
                f"p{i}", f"{rng.gauss(3, 1):.3f}",
                " ".join(rng.choice(["potato", "gravy", "butter"])
                         for _ in range(12 + rng.randrange(6))),
                " ".join(rng.choice(["delightful", "considerable", "fluffy"])
                         for _ in range(20 + rng.randrange(8)))])

    preset = compose(
        ["readability", "stats_correlations"],
        source="csv", input_path=str(src), text_cols=["answer", "followup"],
        text_mode="separate", id_cols=["pid"], name="separate run",
        var_values={"stats_outcome_cols": ["TIPI_Open"],
                    "stats_meta_carry": ["TIPI_Open"],
                    "stats_split_col": "source_col"})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]

    table = _read(work / "stats_results" / "analysis_table.csv")
    assert len(table) == 60, "one row per participant per question"
    assert {r["source_col"] for r in table} == {"answer", "followup"}
    # the participant's outcome repeats across their rows, which is exactly
    # why the analyses can't go pooling them
    by_pid = {}
    for r in table:
        by_pid.setdefault(r["text_id"], set()).add(r["TIPI_Open"])
    assert all(len(v) == 1 for v in by_pid.values())

    corr = _read(work / "stats_results" / "correlations_pearson.csv")
    assert "source_col" in corr[0], "results say which question they describe"
    assert {r["source_col"] for r in corr} == {"answer", "followup"}
    # ...and each analysis only used its own question's rows
    for row in corr:
        assert int(row["TIPI_Open_n"]) <= 30

    report = (work / "stats_results" / "report.md").read_text(encoding="utf-8")
    assert "Run separately for each `source_col`" in report


def test_a_column_computed_only_to_filter_on_is_never_a_predictor(tmp_path):
    """The report this fixes: a ridge regression listed word_count among its
    predictors. It existed only because the user asked to drop texts under
    25 words -- a gate, not a variable of interest -- and a model built on it
    looks perfectly fine while answering a question nobody asked."""
    rng = random.Random(15)
    src = tmp_path / "study.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "openness", "text"])
        for i in range(40):
            words = 6 if i % 5 == 0 else 30 + rng.randrange(10)
            w.writerow([f"p{i}", f"{rng.gauss(3, 1):.3f}",
                        " ".join(rng.choice(["potato", "gravy", "butter"])
                                 for _ in range(words))])

    preset = compose(
        ["readability", "word_count", "stats_ridge_fit"],
        source="csv", input_path=str(src), text_cols=["text"],
        id_cols=["pid"], name="filter only",
        overrides={"stats_assemble": {"filter_csvs": ["{{word_counts}}"]}},
        var_values={"stats_outcome_cols": ["openness"],
                    "stats_meta_carry": ["openness"],
                    "stats_filters": [["word_count", ">=", 25]]})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]

    stats = work / "stats_results"
    # it's in the table (the filter needs it there), and the filter worked
    table = _read(stats / "analysis_table.csv")
    assert "word_count" in table[0]
    assert all(int(r["word_count"]) >= 25 for r in table)
    assert len(table) == 32, "the eight short texts were dropped"

    # ...but it's nowhere among the predictors
    predictors = {r["predictor"] for r in _read(stats / "ridge_coefficients.csv")}
    assert "word_count" not in predictors
    assert predictors, "the model still has real predictors"
    assert any("flesch" in p or "lexicon" in p for p in predictors)

    sets = json.loads((stats / "analysis_table_sets.json").read_text("utf-8"))
    assert "word_count" not in sets["sets"]
    assert sets["filters"]["word_count"] == ["word_count"]


def test_a_run_of_nothing_but_filter_tables_is_refused(tmp_path):
    from taters.ui.compose import ComposeError

    with pytest.raises(ComposeError, match="at least one feature table"):
        compose(["word_count", "stats_correlations"], source="csv",
                input_path="x.csv", text_cols=["text"],
                overrides={"stats_assemble":
                           {"filter_csvs": ["{{word_counts}}"]}},
                var_values={"stats_outcome_cols": ["o"]})


def test_a_classifier_fitted_in_one_run_scores_another(tmp_path):
    """
    The classifier's half of the round trip. It was reachable from the API
    and from the model library, but nothing composed it into a run, so the
    one path a user actually takes -- pick it in the wizard, fit it, score a
    second study with it -- had never been executed end to end.
    """
    src_a = _scored_spreadsheet(tmp_path, "study_a.csv", n=90, seed=4)
    preset = compose(
        ["readability", "stats_classify_fit"],
        source="csv", input_path=str(src_a), text_cols=["text"],
        id_cols=["participant"], name="fit run",
        var_values={"stats_class_cols": ["condition"],
                    "stats_meta_carry": ["condition"]})
    work_a = tmp_path / "run_a"
    work_a.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work_a,
                          out_manifest=work_a / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]

    metrics = _read(work_a / "stats_results" / "classifier_cv_metrics.csv")[0]
    # length separates the cohorts in this fixture, so a classifier that
    # works ought to beat just guessing the commonest class
    assert float(metrics["accuracy"]) > float(metrics["baseline_accuracy"])
    model = work_a / "stats_results" / "models" / "classifier__all__condition.json"
    assert model.is_file()

    src_b = _scored_spreadsheet(tmp_path, "study_b.csv", n=40, seed=77)
    preset_b = compose(
        ["readability", "score_with_model"],
        source="csv", input_path=str(src_b), text_cols=["text"],
        id_cols=["participant"], name="apply run",
        var_values={"model_path": str(model)})
    work_b = tmp_path / "run_b"
    work_b.mkdir()
    manifest_b = run_preset(preset_b, workers=1, work_dir=work_b,
                            out_manifest=work_b / "run_manifest.json",
                            verbose=False)
    assert not manifest_b["errors"], manifest_b["errors"]

    scored = _read(work_b / "features" / "model_scores.csv")
    assert len(scored) == 40
    predicted = [r["pred_condition"] for r in scored]
    assert all(p in {"A", "B", "C"} for p in predicted), predicted[:5]
    # a classifier that answers the same class for all forty rows would pass
    # every check above and still be worthless, so let's make sure it doesn't
    assert len(set(predicted)) > 1


def test_a_result_older_than_its_inputs_is_redone_not_reused(tmp_path):
    """
    A real run rebuilt its metadata table with the right columns, and the
    assemble step then handed back the analysis table joined from the old
    one, so the ridge could not find its outcomes. An existing result is
    reused only when it is at least as new as every input it was made from;
    the whole statistics stage follows the rule, so a rebuilt table flows
    through to fresh results and a fresh report.
    """
    import os
    import time

    src = _spreadsheet(tmp_path)
    preset = _stats_preset(tmp_path, src)
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    stats = work / "stats_results"
    table_before = (stats / "analysis_table.csv").read_bytes()
    report_before = (stats / "report.md").stat().st_mtime

    # second run, nothing changed, so everything gets reused as it stands
    run_preset(preset, workers=1, work_dir=work,
               out_manifest=work / "run_manifest.json", verbose=False)
    assert (stats / "analysis_table.csv").read_bytes() == table_before
    assert (stats / "report.md").stat().st_mtime == report_before

    # now an input changes: we touch the metadata table so it's newer than the
    # table (by a moment, and in the past, so the rebuilt table is newer still)
    table_path = stats / "analysis_table.csv"
    table_time = table_path.stat().st_mtime
    meta = work / "gathered" / "metadata.csv"
    os.utime(meta, (table_time + 0.5, table_time + 0.5))
    time.sleep(0.6)
    run_preset(preset, workers=1, work_dir=work,
               out_manifest=work / "run_manifest.json", verbose=False)
    assert table_path.stat().st_mtime > table_time, \
        "the analysis table should have been remade from the newer metadata"
    assert (stats / "report.md").stat().st_mtime > report_before, \
        "and the report after it"


# ---------------------------------------------------------------------------
# word clouds
# ---------------------------------------------------------------------------

def test_word_clouds_are_drawn_after_the_analyses_and_shown_in_the_report(tmp_path):
    """
    Every analysis brings the cloud step with it, once, placed after the
    last analysis and before the report -- so the pictures exist when the
    report is written and the report shows them.
    """
    pytest.importorskip("PIL")
    src = _spreadsheet(tmp_path, n=30)
    preset = _stats_preset(tmp_path, src, selected=[
        "readability", "stats_group_differences", "stats_correlations",
        "stats_ridge_fit"])
    calls = [s["call"] for s in preset["steps"]]
    assert calls.count("potato.figures.stats_wordclouds") == 1
    clouds = calls.index("potato.figures.stats_wordclouds")
    assert clouds > max(calls.index(c) for c in calls if c.startswith("potato.stats.")
                        and "report" not in c)
    assert clouds < calls.index("potato.stats.write_stats_report")

    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    figures = work / "stats_results" / "figures" / "wordclouds"
    pngs = sorted(q.relative_to(figures).as_posix() for q in figures.rglob("*.png"))
    assert any(q.startswith("ridge-regression/") for q in pngs), pngs
    report = (work / "stats_results" / "report.md").read_text(encoding="utf-8")
    assert "## Word clouds" in report
    assert report.index("## Word clouds") > report.index("## Correlations")
    assert "figures/wordclouds/ridge-regression/" in report


def test_word_clouds_can_be_turned_off_with_one_variable(tmp_path):
    src = _spreadsheet(tmp_path, n=30)
    preset = _stats_preset(tmp_path, src, extra_vars={"wordclouds": False})
    assert preset["vars"]["wordclouds"] is False
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    assert not (work / "stats_results" / "figures").exists()
    report = (work / "stats_results" / "report.md").read_text(encoding="utf-8")
    assert "## Word clouds" not in report


def test_a_run_writes_descriptives_for_its_feature_tables(tmp_path):
    """Whether or not statistics were asked for: one table of descriptive
    statistics per feature table, under its own folder, with a README."""
    from taters.stats.describe import COLUMNS

    src = _spreadsheet(tmp_path)
    preset = compose(["readability"], source="csv", input_path=str(src),
                     text_cols=["text"], id_cols=["participant"], name="feats")
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    folder = work / "stats_descriptives"
    rows = _read(folder / "readability.csv")
    assert list(rows[0].keys()) == COLUMNS
    described = {r["variable"] for r in rows}
    assert "flesch_reading_ease" in described and "text_id" not in described
    assert all(r["n"] == "24" for r in rows)
    assert (folder / "README.md").is_file()
    assert not (work / "stats_results").exists(), "no statistics were asked for"
