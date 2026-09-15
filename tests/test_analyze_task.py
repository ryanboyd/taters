"""
"Run analyses": the statistics over a spreadsheet somebody already has.

Every other road to the statistics runs through feature extraction. This one
does not: the user's own columns are the measures. What these check is that
the flow asks the right questions, that a column can be a predictor or an
outcome but never both, and that the run really produces the statistics --
including the grouped case, where several rows per participant are averaged
before anything is fitted.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from taters.ui import recipes as _recipes
from taters.ui import wizard as wiz
from taters.ui.prompts import QuitRequested, ScriptedPrompter
from taters.ui.tasks import TaskContext, all_tasks
from taters.ui.tasks.analyze import TASK
from wizard_helpers import browse_to


def _study(path: Path, *, per_person: int = 1, n: int = 60, seed: int = 4) -> Path:
    """A planted study: `openness` is twice `f1`, `f2` is noise. With
    ``per_person`` above 1 each participant contributes that many rows."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        # `subject_no` is a NUMBER that identifies a person. it reads as
        # numeric, so nothing but the id binding keeps it out of the models
        w.writerow(["pid", "subject_no", "wave", "f1", "f2", "openness",
                    "condition"])
        for i in range(n):
            base = rng.gauss(0, 1)
            for k in range(per_person):
                w.writerow([f"p{i}", i, k, f"{base + rng.gauss(0, .2):.4f}",
                            f"{rng.gauss(0, 1):.4f}",
                            f"{2 * base + rng.gauss(0, .3):.4f}",
                            "A" if i % 2 else "B"])
    return path


def _rows(path: Path):
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Where it sits, and what it asks
# ---------------------------------------------------------------------------

def test_analyzing_a_spreadsheet_sits_beside_wrangling_one():
    """
    It shares a row with wrangling rather than standing on the front page:
    both start from a file that already exists, neither extracts a feature,
    and the same person often does one straight after the other -- combine
    the rows per participant, then predict something from them.
    """
    from taters.ui.tasks.data import entries

    ids = [t.id for t in all_tasks()]
    assert "analyze_spreadsheet" not in ids, "it is a submenu row now"
    assert ids[0] == "data" and ids.index("data") < ids.index("extract")

    beneath = [t.id for t in entries()]
    assert beneath == ["wrangle", "analyze_spreadsheet"], \
        "tidying produces the file the other one reads; it goes first"
    assert TASK.label == "Analyze data"


def test_the_source_stage_asks_for_predictors_rather_than_text(tmp_path):
    """
    There is no text to measure -- the numbers are already in the file. So the
    question that would ask which columns hold the text asks which columns
    *are* the measures, and the how-should-several-text-columns-be-read
    question never appears.
    """
    study = _study(tmp_path / "s.csv")
    p = ScriptedPrompter([
        "csv", *browse_to(study), ["f1", "f2"], True, ["pid"], "row",
        ["stats_correlations"], ["openness"], False, "fdr_bh", False,
        ":done", "Run", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path, banner=False, analyses=True,
                   preselected=["spreadsheet_columns"],
                   var_defaults={"wordclouds": False})
    asked = [q for _k, q in p.asked]
    assert "Which columns are the predictors?" in asked
    assert not any("hold the text" in q for q in asked)
    assert not any("How should they be measured" in q for q in asked)
    # only columns that read as numbers are offered as predictors
    offered = {c.value for c in p.offered_choices("Which columns are the predictors?")}
    assert offered == {"f1", "f2", "openness", "wave", "subject_no"}, \
        "pid and condition are not numbers"


def test_a_column_is_a_predictor_or_an_outcome_but_never_both(tmp_path):
    """
    A column on both sides of a regression predicts itself perfectly and means
    nothing. The columns ticked as predictors are spoken for, so the outcome,
    group, control and identifier questions never offer them again.
    """
    study = _study(tmp_path / "s.csv")
    p = ScriptedPrompter([
        "csv", *browse_to(study), ["f1", "f2"], True, ["pid"], "row",
        ["stats_correlations", "stats_group_differences"],
        "condition",                    # the group column
        ["openness"], False, "fdr_bh", False,
        ":done", "Run", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path, banner=False, analyses=True,
                   preselected=["spreadsheet_columns"],
                   var_defaults={"wordclouds": False})
    outcomes = {c.value for c in p.offered_choices("Which column(s) hold the outcomes?")}
    assert outcomes == {"openness", "wave", "subject_no"}
    assert not ({"f1", "f2"} & outcomes), "a predictor is not also an outcome"
    ids = {c.value for c in p.offered_choices("Which one(s)?")}
    assert not ({"f1", "f2"} & ids), "a predictor is not also the row's identifier"
    groups = {c.value for c in p.offered_choices("Which column separates the groups")}
    assert not ({"f1", "f2"} & groups), "a predictor is not also the grouping column"


def test_the_predictors_and_the_silenced_word_clouds_reach_the_preset(tmp_path):
    """The columns travel as a pipeline variable, so the saved YAML re-runs on
    a new file; and a cloud of column names is not what anyone came for."""
    study = _study(tmp_path / "s.csv")
    p = ScriptedPrompter([
        "csv", *browse_to(study), ["f1", "f2"], True, ["pid"], "row",
        ["stats_correlations"], ["openness"], False, "fdr_bh", False,
        ":done", "Run", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, banner=False, analyses=True,
                            preselected=["spreadsheet_columns"],
                            var_defaults={"wordclouds": False})
    variables = result.preset["meta"]["variables"]
    assert variables["analysis_predictor_cols"]["default"] == ["f1", "f2"]
    assert variables["wordclouds"]["default"] is False
    step = next(s for s in result.preset["steps"]
                if s["with"].get("out_csv", "").endswith("spreadsheet_columns.csv"))
    assert step["with"]["carry_cols"] == "{{var:analysis_predictor_cols}}"
    assert step["with"]["include_id_cols"] is False, \
        "an id column in a feature table becomes a predictor"
    # the preset is a real one, by the same rules every shipped preset follows
    from preset_checks import check_preset

    assert check_preset(result.preset) == []


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def test_the_statistics_run_on_the_spreadsheets_own_columns(tmp_path):
    """End to end, for real: the planted signal comes back out of the ridge,
    the report is written, and nothing draws a word cloud."""
    study = _study(tmp_path / "s.csv")
    p = ScriptedPrompter([
        # the identifier is a *number*, which is the case that matters: left
        # to ride into the feature table it reads as a perfectly good
        # predictor, and a model that learns from a participant number has
        # learned the order people were recruited in
        "csv", *browse_to(study), ["f1", "f2"], True, ["subject_no"], "row",
        ["stats_correlations", "stats_ridge_fit"], ["openness"], False,
        "fdr_bh", False, ":done", "Run", "run", "quit",
    ])
    # the finish screen's "Finish" is how a run ends; anything else escaping
    # here would be a real failure, so we name it rather than catching Exception
    with pytest.raises(QuitRequested):
        TASK.run(TaskContext(prompter=p, cwd=tmp_path))

    stats = tmp_path / "run" / "stats_results"
    sets = json.loads((stats / "analysis_table_sets.json").read_text(encoding="utf-8"))
    assert sets["sets"] == {"spreadsheet_columns": ["f1", "f2"]}, \
        "the identifier is not a predictor, however numeric it looks"
    assert "openness" in sets["metadata"]

    metrics = _rows(stats / "ridge_cv_metrics.csv")
    assert float(metrics[0]["cv_r2"]) > 0.8, "openness is twice f1; it should be found"
    assert (stats / "correlations_pearson.csv").is_file()
    assert (stats / "report.md").is_file()
    assert (stats / "models" / "ridge__all__openness.json").is_file()
    assert not list(stats.glob("figures/wordclouds/**/*.png")), "no text, no clouds"


def test_several_rows_per_person_are_averaged_before_anything_is_fitted(tmp_path):
    """
    Somebody with three waves per participant says so here rather than being
    sent somewhere else and back. The predictors are averaged per participant,
    arriving as `<column>_mean` -- and the counts that averaging writes beside
    them are set aside, because "how many rows this person had" is not a
    predictor anybody wants in their model.
    """
    study = _study(tmp_path / "waves.csv", per_person=3, n=40, seed=7)
    p = ScriptedPrompter([
        "csv", *browse_to(study), ["f1", "f2"],
        False,                          # no single id column; pid repeats
        "group", ["pid"],               # combine the rows of each participant
        ["stats_ridge_fit"], ["openness"],
        False,                          # no controls
        False,                          # no row filters
        ":done", "Waves", "run", "quit",
    ])
    with pytest.raises(QuitRequested):
        TASK.run(TaskContext(prompter=p, cwd=tmp_path))

    stats = tmp_path / "waves" / "stats_results"
    table = _rows(stats / "analysis_table.csv")
    assert len(table) == 40, "one row per participant, not per wave"

    sets = json.loads((stats / "analysis_table_sets.json").read_text(encoding="utf-8"))
    features = next(iter(sets["sets"].values()))
    assert sorted(features) == ["f1_mean", "f2_mean"]
    counted = [c for c in features if c.endswith("_n") or "group_count" in c]
    assert not counted, f"row counts became predictors: {counted}"
    assert float(_rows(stats / "ridge_cv_metrics.csv")[0]["cv_r2"]) > 0.8


def test_the_step_reads_the_spreadsheet_the_same_way_the_metadata_gather_does():
    """
    The two tables join on `text_id`, and they only join at all because both
    are built by the same gather, from the same file, with the same delimiter
    and the same key. A run where one sniffed the delimiter and the other was
    told it would silently join nothing.
    """
    features = _recipes.by_id("spreadsheet_columns")
    metadata = _recipes.by_id("stats_gather_metadata")
    assert features.target == metadata.target
    for key in ("csv_path", "delimiter", "text_cols"):
        assert features.with_[key] == metadata.with_[key], key
    assert features.keys_like_metadata and metadata.keys_like_metadata
    # and the level question has to be asked, or there is no way to say that
    # several rows belong to one person
    assert _recipes.level_aware(features)
