"""
Tests for the setup wizard, driven end to end without a terminal (see
wizard_helpers for how). If a test fails with 'ran out of answers', the
message names the question that went unanswered -- insert an answer at
that position.

This file: the analysis stage, the two extraction flows, and Esc meaning the
previous question everywhere.
"""

from __future__ import annotations


import pytest
from preset_checks import assert_valid_preset

from taters.ui import wizard as wiz
from taters.ui.prompts import (QuitRequested,
                                ScriptedPrompter)
from wizard_helpers import (clean_machine,  # noqa: F401  (autouse fixture)
                            browse_to, run, EscapingPrompter, _tuning,
                            _escaping_tuning, _engine_tuning, _asked)


# ---------------------------------------------------------------------------
# the analysis stage
# ---------------------------------------------------------------------------


def test_a_folder_of_documents_is_never_asked_about_statistics(tmp_path, essays):
    """No metadata columns, no statistics -- and the stage says so rather
    than vanishing, so its absence is not read as a bug."""
    p = ScriptedPrompter([
        "txt_dir", *browse_to(essays), ["readability"], False, "Docs run", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert not any("statistics on the results" in q for _k, q in p.asked)
    assert "grouping or outcome columns" in p.text_output


def test_a_spreadsheet_is_offered_statistics_and_may_decline(tmp_path, study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        [],                       # tick nothing: just the feature table
        False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("statistics on the results" in q for _k, q in p.asked)
    calls = [s["call"] for s in result.preset["steps"]]
    # (the descriptives step comes with every feature table, so it doesn't count
    # as the statistics stage)
    assert not any("stats" in c and "describe" not in c for c in calls), \
        "declining must leave the pipeline exactly as it was"


def test_picking_group_differences_wires_the_whole_stats_tail(tmp_path, study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False,
        "condition",
        False,  # no control variables
        # the grouping column
        "fdr_bh",                 # multiple-comparison correction
        False,                    # no row filters
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.stats.assemble_analysis_table" in calls
    assert "potato.stats.analyze_group_differences" in calls
    assert "potato.stats.write_stats_report" in calls
    assert result.preset["vars"]["stats_group_col"] == "condition"
    # the grouping column has to reach the metadata table, otherwise the analysis
    # would go looking for a column nothing wrote
    assert result.preset["vars"]["stats_meta_carry"] == ["condition"]
    assert_valid_preset(result.preset)


def test_controls_can_be_chosen_and_reach_every_analysis(tmp_path):
    """
    Controlling for age and gender is a thing researchers do to every
    analysis, and the engines supported it for a while before any screen
    asked -- so it was unreachable unless you hand-edited the pipeline (a
    real report: "I don't recall seeing at any point in the ux flow the
    ability to add control variables").

    The answer has to reach all three analyses from one question, or
    "controlling for age" would mean a different sample in the comparison
    than in the correlations.
    """
    path = tmp_path / "survey.csv"
    rows = ["pid,condition,gender,age,openness,text"]
    for i in range(12):
        rows.append(f"p{i},{'AB'[i % 2]},{1 + i % 2},{20 + i},"
                    f"{3 + i * 0.1:.1f},\"words about potatoes {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        True,                       # yes, control for something
        "\x00right:gender",         # gender is coded 1/2: → makes it labels
        ["age", "gender"],          # which columns
        "fdr_bh", False,
        ":done", "Controlled", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    v = result.preset["vars"]
    assert v["stats_control_cols"] == ["age", "gender"]
    # `gender` is coded 1/2, so it looks numeric but isn't. read as a measurement
    # it would say each step from one gender to the other is worth 0.4 of
    # something
    assert v["stats_categorical_controls"] == ["gender"]
    # a control has to reach the analysis table like any other metadata column,
    # otherwise the analysis gets handed a column name nothing wrote
    assert set(v["stats_meta_carry"]) >= {"condition", "age", "gender"}
    assert_valid_preset(result.preset)


def test_a_numeric_control_is_not_assumed_to_be_a_measurement(tmp_path):
    """The follow-up question is only put when it can matter -- when a
    chosen control holds numbers. Columns holding words are categories
    beyond argument, so asking would be asking the user to confirm
    arithmetic they never suggested."""
    path = tmp_path / "survey.csv"
    rows = ["pid,condition,site,openness,text"]
    for i in range(12):
        rows.append(f"p{i},{'AB'[i % 2]},{'north south'.split()[i % 2]},"
                    f"{3 + i * 0.1:.1f},\"words {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        True,
        ["site"],                   # a column of words: no follow-up asked
        "fdr_bh", False,
        ":done", "Words", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    v = result.preset["vars"]
    assert v["stats_control_cols"] == ["site"]
    # recognized as a category without anybody being asked
    assert v["stats_categorical_controls"] == ["site"]
    asked = [q for _, q in p.asked]
    assert not any("categories rather than measurements" in q for q in asked)


def test_an_identifier_column_is_never_offered_as_a_control(tmp_path):
    """Holding a per-row id constant either explains the outcome away
    entirely or means nothing. `pid` was offered because the run had not
    declared its id columns, which is the common case."""
    path = tmp_path / "survey.csv"
    rows = ["pid,condition,age,text"]
    for i in range(12):
        rows.append(f"p{i},{'AB'[i % 2]},{20 + i},\"words {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        True, ["age"],              # age is a measurement, shown as numbers
        "fdr_bh", False,
        ":done", "No ids", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    offered = [c.value for c in
               p.offered_choices("Which column(s) should be held constant?")]
    assert "pid" not in offered, f"the identifier was offered: {offered}"
    assert "age" in offered
    # and not the column we're already comparing on, either
    assert "condition" not in offered


def test_only_numeric_columns_are_offered_as_outcomes(tmp_path, study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["openness"],
        False,  # no control variables
        "fdr_bh",                 # correction
        False,                    # no filters
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    offered = [c.value for c in
               p.offered_choices("Which column(s) hold the outcomes?")]
    assert offered == ["openness"], "condition is a label, not a measurement"
    assert result.preset["vars"]["stats_outcome_cols"] == ["openness"]


def test_a_grouped_run_renames_outcomes_to_their_group_averages(tmp_path,
                                                                study_csv):
    """The subtle one: combining rows means an outcome becomes the group's
    average, which the gatherer names <column>_mean. The analysis has to be
    told the new name, or it correlates nothing."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "group", ["condition"], False,
        ["stats_correlations"],
        ["openness"],
        False,  # no control variables
        "fdr_bh", False, ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["vars"]["stats_outcome_cols"] == ["openness_mean"]
    assert result.preset["vars"]["stats_meta_agg"] == {"openness": "mean"}


def test_a_grouped_run_grays_out_columns_that_vary_inside_a_group(
        tmp_path, study_csv):
    """A column that differs from row to row inside a combined row has no
    single value to speak for that row -- the gatherer leaves it blank
    rather than reporting the first row's value, so choosing it would
    produce an analysis with every row dropped. Grayed with the reason
    rather than hidden: a column the user can see in their spreadsheet and
    cannot find on this screen reads as a bug in the screen."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "group", ["condition"], False,
        ["stats_group_differences"],
        "condition",
        "fdr_bh", False, ":done", "Study", "save"
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    rows = {c.value: c for c in p.offered_choices(
        "Which column separates the groups you want to compare?")}
    assert not rows["condition"].disabled
    # `openness` differs between the rows sharing a condition, and `pid` is
    # unique to each of them
    for column in ("openness", "pid"):
        assert rows[column].disabled, f"{column} should be grayed out"
        assert "differs within" in rows[column].disabled


def test_a_column_constant_within_each_group_can_still_be_compared(tmp_path):
    """
    The two senses of "group" are independent, and conflating them made a
    real analysis impossible to express.

    Combining on ``subreddit + author`` decides what one row *is*. Comparing
    moderators against regular users decides what you want to *test*.
    ``is_moderator`` is not one of the columns combined on, but it is a fact
    about the author, so it still has exactly one value per combined row --
    and the offer used to be restricted to the combining columns themselves,
    which meant "combine each author's comments, then compare moderators
    with regular users" could not be asked for at all.
    """
    path = tmp_path / "reddit.csv"
    rows = ["comment_id,subreddit,author,is_moderator,score,text"]
    for i in range(24):
        rows.append(f"c{i},{'r_science' if i % 2 else 'r_cooking'},u{i % 6},"
                    f"{'moderator' if i % 6 < 2 else 'regular'},{i},"
                    f"\"a comment about potatoes {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], False,
        ["readability"], "group", ["subreddit", "author"], False,
        ["stats_group_differences"], False,
        "is_moderator",             # not a combining column, and valid
        False,                      # no controls
        "fdr_bh", False,
        ":done", "Reddit", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    rows_offered = {c.value: c for c in p.offered_choices(
        "Which column separates the groups you want to compare?")}
    assert not rows_offered["is_moderator"].disabled
    assert not rows_offered["subreddit"].disabled
    # `score` differs from comment to comment within one author
    assert rows_offered["score"].disabled

    assert result.preset["vars"]["stats_group_col"] == "is_moderator"
    # it's not one of the combining columns, so it has to be carried into the
    # metadata table explicitly, otherwise the analysis goes looking for a
    # column nothing wrote
    assert "is_moderator" in result.preset["vars"]["stats_meta_carry"]
    assert_valid_preset(result.preset)


def test_row_filters_are_collected_into_the_preset(tmp_path, study_csv):
    """Pick the places, tick the variables, then set each one in turn."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh",
        True,                       # yes, filter rows
        ["original"],               # ...on spreadsheet columns
        ["openness"],               # ...this one
        ">=", "3.2",                # operator, value
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["vars"]["stats_filters"] == [["openness", ">=", 3.2]]


def test_several_filters_are_ticked_at_once_then_set_one_by_one(tmp_path,
                                                                study_csv):
    """The whole point of the restructure: choosing the variables and
    setting their conditions are different decisions, and interleaving them
    meant answering "and another?" between every one."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh",
        True, ["original"],
        ["openness", "condition"],  # two at once
        ">=", "3.2",                # openness
        ["A", "B"],                 # condition: a column of labels is a
                                    # checkbox of the values to keep
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["vars"]["stats_filters"] == [
        ["openness", ">=", 3.2], ["condition", "in", ["A", "B"]]]
    assert not any("and another" in q.lower() for _k, q in p.asked)


def test_word_count_is_the_first_thing_offered_and_brings_its_own_step(
        tmp_path, study_csv):
    """A length filter is what nearly everyone means, and it used to be the
    one thing you could not simply pick: you had to know readability calls
    its count `lexicon_count`, and to have selected readability."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["lexical_richness"], "row",       # nothing here counts words
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh",
        True, ["generated"],
        [wiz._WORD_COUNT],
        ">=", "25",
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    offered = p.offered_choices("Which variable(s) do you want to filter on?")
    assert offered[0].value == wiz._WORD_COUNT, "it goes first"
    assert offered[0].label == "Word count"
    assert [c.value for c in offered].index(wiz._WORD_COUNT) < \
        [c.value for c in offered].index(wiz._MEASURED_COLUMN)

    assert result.preset["vars"]["stats_filters"] == [["word_count", ">=", 25]]
    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.text.analyze_word_count" in calls, \
        "the step that produces the column has to be added"

    # joined so the filter can name its column, but as a FILTER table, not a
    # feature one. wanting to drop short texts isn't wanting text length as a
    # predictor, and a ridge regression happily took it as one
    assemble = next(s for s in result.preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    assert assemble["with"]["filter_csvs"] == ["{{word_counts}}"]
    assert "{{word_counts}}" not in assemble["with"]["feature_csvs"]
    assert_valid_preset(result.preset)


def test_the_filter_places_narrow_the_variable_list(tmp_path, study_csv):
    """A 150-column spreadsheet should not have to be scrolled past to reach
    the measures."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh",
        True, ["generated"],           # only the run's own measures
        [wiz._WORD_COUNT], ">=", "25",
        ":done", "Study", "save"
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = [c.value for c in
               p.offered_choices("Which variable(s) do you want to filter on?")]
    assert "condition" not in offered and "openness" not in offered


def test_a_filter_value_that_is_not_a_number_is_re_asked(tmp_path, study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh",
        True, ["original"], ["openness"], ">=", "loads", "3.2",
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert "is not a number" in p.text_output
    assert result.preset["vars"]["stats_filters"] == [["openness", ">=", 3.2]]


def test_choosing_a_subset_of_feature_tables_narrows_the_assemble(tmp_path,
                                                                  study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability", "lexical_richness"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        ["readability"],          # only this table feeds the statistics
        "fdr_bh",                 # correction
        False,                    # no filters
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assemble = next(s for s in result.preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    assert assemble["with"]["feature_csvs"] == ["{{readability_features}}"]


def test_separate_feature_sets_are_recorded_as_such(tmp_path, study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability", "lexical_richness"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        ["readability", "lexical_richness"],
        "separate",               # one analysis per table
        "fdr_bh", False, ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["vars"]["stats_feature_sets"] == "per_table"


def test_backing_out_of_the_options_screen_re_asks_the_analysis(tmp_path,
                                                               study_csv):
    """Esc is one stage back, everywhere. With statistics on the menu that
    stage is the analysis one -- and answering it differently must not leave
    the first answer's steps behind."""
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh", False,
        "__esc__",                # Esc at the options gate -> analysis
        [],                       # this time, no statistics at all
        False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    asked = [q for _k, q in p.asked if "statistics on the results" in q]
    assert len(asked) == 2, "Esc did not land back on the analysis stage"
    calls = [s["call"] for s in result.preset["steps"]]
    # (the descriptives step comes with every feature table, so it doesn't count
    # as the statistics stage)
    assert not any("stats" in c and "describe" not in c for c in calls), \
        "the abandoned answer's steps survived"
    assert not any(k.startswith("stats_") and v
                   for k, v in result.preset["vars"].items()), \
        "the abandoned answer's variables survived"


def test_a_prediction_model_asks_for_outcomes_too(tmp_path, study_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_ridge_fit"], False,
        ["openness"],
        False,  # no control variables
        # what to predict
        False,                    # no filters
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("hold the outcomes" in q for _k, q in p.asked)
    assert result.preset["vars"]["stats_outcome_cols"] == ["openness"]
    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.stats.fit_ridge_csv" in calls
    assert_valid_preset(result.preset)


def test_the_correction_is_asked_and_recorded(tmp_path, study_csv):
    """Which correction to use changes what counts as a finding, so it is a
    question rather than a hidden default -- and it lands in the preset."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "bonferroni",             # the strictest
        False, ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("corrected for multiple comparisons" in q for _k, q in p.asked)
    assert result.preset["vars"]["stats_p_adjust"] == "bonferroni"
    offered = [c.value for c in p.offered_choices(
        "How should p-values be corrected for multiple comparisons?")]
    assert "none" in offered, "declining to correct has to be on the menu"
    assert offered[1] == "fdr_bh", "the default sits where the default goes"


def test_the_correction_is_shared_by_every_analysis(tmp_path, study_csv):
    """Two analyses in one report correcting differently would be a
    methods section nobody could write."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences", "stats_correlations"], False,
        "condition", ["openness"], "holm",
        False, ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    using = [s["with"]["p_adjust"] for s in result.preset["steps"]
             if "p_adjust" in s.get("with", {})]
    assert using == ["{{var:stats_p_adjust}}"] * 2
    assert result.preset["vars"]["stats_p_adjust"] == "holm"


def test_a_prediction_only_run_is_not_asked_about_corrections(tmp_path,
                                                              study_csv):
    """Ridge reports cross-validated performance, not p-values; there is
    nothing there to correct."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_ridge_fit"], False, ["openness"],
        False,  # no control variables
        False, ":done", "Study", "save"
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert not any("corrected for multiple comparisons" in q
                   for _k, q in p.asked)


def test_a_wide_spreadsheet_is_summarized_not_recited(tmp_path):
    """Naming all 150 columns above a list of those same 150 columns says
    nothing new and shrinks the list to a porthole (a real report)."""
    path = tmp_path / "wide.csv"
    names = ["Gender", "Country"] + [f"Img{i}_Valence_mean" for i in range(80)]
    path.write_text(",".join(names) + "\n" + ",".join("x" for _ in names) + "\n",
                    encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["Gender"], False,
        ["readability"], "row", [], False, "Wide", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    summary = next(line for line in p.output if "column(s)" in line)
    assert "82 column(s)" in summary, "the count is the useful part"
    assert "Gender" in summary and "and 74 more" in summary
    assert "Img70_Valence_mean" not in summary, "it stopped reciting"
    # ...and every column is still offered on the question itself
    offered = [c.value for c in p.offered_choices(
        "Which column(s) hold the text you want analyzed?")]
    assert len(offered) == 82


def test_the_measure_filter_names_the_tables_it_will_search(tmp_path,
                                                            study_csv):
    """The report this fixes: the prompt defaulted to `lexicon_count`, which
    only the readability step writes. Offered to someone who had selected
    lexical richness and sentence embeddings, it cost a four-minute run and
    produced a wall of red at the last step."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["lexical_richness"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh",
        True, ["generated"], [wiz._MEASURED_COLUMN], "ttr", ">=", "0.4",
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    picker = p.offered_choices("Which variable(s) do you want to filter on?")
    measured = next(c for c in picker if c.value == wiz._MEASURED_COLUMN)
    assert "Lexical richness" in measured.help, \
        "the prompt has to name the table the column must come from"
    # no pre-filled guess, since the obvious one is wrong more often than not
    typed = [q for kind, q in p.asked if "Name of the measure" in q]
    assert typed, "the typed-name question was asked"
    assert result.preset["vars"]["stats_filters"] == [["ttr", ">=", 0.4]]


def test_a_long_failure_is_trimmed_on_the_finish_screen(tmp_path):
    """A step that refuses because a column is missing can name a thousand
    columns; the screen shows a sentence and points at the manifest."""
    p = ScriptedPrompter(["quit"])
    huge = ("assemble failed: ValueError: filter column 'lexicon_count' is "
            "not in the analysis table. It has 1030: "
            + ", ".join(f"e{i}" for i in range(1024)))
    with pytest.raises(QuitRequested):
        wiz.finish_screen(p, ok=False, manifest={"errors": [huge]},
                          folder=tmp_path, manifest_path=tmp_path / "m.json",
                          failures=[huge])

    shown = [line for line in p.output if "filter column" in line]
    assert shown, "the failure still reaches the screen"
    assert all(len(line) < 400 for line in shown), "trimmed to a screenful"
    assert any("more characters" in line for line in shown)
    assert len(shown) == 1, "the same failure must not be printed twice"


def test_measuring_columns_separately_now_analyzes_them_separately(tmp_path):
    """Asking for each text column to be measured on its own used to rule
    statistics out entirely. It does not have to: a participant has one row
    per column, so each column becomes its own analysis -- which is what
    keeps one person's several answers from counting as several independent
    observations, and is the consistent continuation of what they asked for
    at the source stage."""
    path = tmp_path / "survey.csv"
    rows = ["pid,TIPI_Open,answer,followup"]
    for i in range(10):
        rows.append(f"p{i},{3 + i * 0.1:.1f},"
                    f"\"first thoughts number {i}\","
                    f"\"on reflection number {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer", "followup"], "separate",
        True, ["pid"],
        ["readability"], "row",
        ["stats_correlations"], ["TIPI_Open"],
        "fdr_bh",
        False,                     # no filters
        ":done", "Separate", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("statistics on the results" in q for _k, q in p.asked), \
        "the stage has to be offered, not silently skipped"
    assert result.preset["vars"]["stats_split_col"] == "source_col"
    steps = {s.get("save_as"): s for s in result.preset["steps"]}
    assert steps["analysis_table"]["with"]["split_col"] == \
        "{{var:stats_split_col}}"
    assert steps["correlations"]["with"]["split_col"] == \
        "{{var:stats_split_col}}"
    assert_valid_preset(result.preset)


def test_joining_the_text_columns_leaves_the_split_empty(tmp_path):
    path = tmp_path / "survey.csv"
    rows = ["pid,TIPI_Open,answer,followup"]
    for i in range(10):
        rows.append(f"p{i},{3 + i * 0.1:.1f},\"words {i}\",\"more words {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer", "followup"], "concat",
        True, ["pid"],
        ["readability"], "row",
        ["stats_correlations"], ["TIPI_Open"],
        "fdr_bh", False,
        ":done", "Joined", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["vars"]["stats_split_col"] == ""


def test_a_measure_and_its_merge_are_one_row_on_the_table_picker(tmp_path,
                                                                 study_csv):
    """The report this fixes: the picker offered "Sentence embeddings" and
    "Merge sentence embeddings" as though they were two feature sets. They
    are one measure and its follow-up, and only one of the two tables is
    ever joined -- so the picker shows one row, named for the measure."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability", "sentence_embeddings"], "group", ["condition"], False,
        "sentence-transformers/all-roberta-large-v1",   # the model question
        ["stats_correlations"], ["openness"],
        False,  # no control variables
        # ticked by id: at this level the surviving table is the merge, and
        # what the row *says* is the measure's name
        ["readability", "gather_sentence_embeddings"], "together",
        "fdr_bh", False, ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    offered = p.offered_choices(
        "Which feature tables should feed the statistics?")
    labels = [c.label for c in offered]
    assert labels.count("Sentence embeddings (meaning-tuned model)") == 1
    assert "Merge sentence embeddings" not in labels
    assert sorted(labels) == ["Readability scores", "Sentence embeddings (meaning-tuned model)"]

    # ...and what the picker offered is what the run actually joins
    assemble = next(s for s in result.preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    assert assemble["with"]["feature_csvs"] == [
        "{{readability_features}}", "{{sent_embeds_agg}}"]


def test_the_picker_and_the_composer_never_disagree(tmp_path, study_csv):
    """Whatever the level, the tables offered are the tables wired: they are
    computed by one function now, because two computations drifted."""
    from taters.ui.compose import feature_tables, resolve_selection

    for level, group_by in (("row", []), ("group", ["condition"])):
        steps = resolve_selection(
            ["readability", "sentence_embeddings"], source="csv")
        offered = feature_tables(steps, "csv", level, group_by)
        preset = compose_for(level, group_by)
        assemble = next(s for s in preset["steps"]
                        if s["call"] == "potato.stats.assemble_analysis_table")
        assert assemble["with"]["feature_csvs"] == \
            ["{{" + r.save_as + "}}" for r, _ in offered]


def compose_for(level, group_by):
    from taters.ui.compose import compose

    return compose(["readability", "sentence_embeddings",
                    "stats_correlations"],
                   source="csv", input_path="x.csv", text_cols=["text"],
                   level=level, group_by=group_by,
                   var_values={"stats_outcome_cols": ["o"]})


# ---------------------------------------------------------------------------
# the two extraction flows
# ---------------------------------------------------------------------------

def test_the_feature_only_flow_never_mentions_statistics(tmp_path, study_csv):
    """The whole point of splitting the menu: someone who came to turn text
    into numbers should not be asked about analyses they have no columns
    for. It was optional, which meant it was asked of everybody."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        False, "Features only", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=False)

    assert not any("statistic" in q.lower() for _k, q in p.asked)
    assert not any("stats" in s["call"] and "describe" not in s["call"]
                   for s in result.preset["steps"])
    # ...and the rail doesn't carry a stage this run was never going to visit
    assert "analysis" not in {s.key for s in p.stages}


def test_the_analysis_flow_insists_on_an_answer(tmp_path, study_csv):
    """In the "+ run analyses" flow an empty checklist is not a decision but
    a dead end, so it is re-asked rather than accepted."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        [],                          # nothing ticked -- not allowed here
        ["stats_correlations"], False, ["openness"],
        False,  # no control variables
        "fdr_bh", False,
        ":done", "With stats", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    assert "pick at least one analysis" in p.text_output
    assert any(s["call"] == "potato.stats.analyze_correlations"
               for s in result.preset["steps"])


def test_the_analysis_flow_only_offers_a_spreadsheet(tmp_path):
    """Saying so at the first question costs one screen. Saying it after
    they have browsed to a folder, picked features and answered the level
    question costs all of that."""
    p = ScriptedPrompter(["csv"])
    try:
        wiz.ask_source(p, analyses=True)
    except Exception:
        pass
    offered = [c.value for c in p.offered_choices(
        "What kind of data do you have?")]
    assert offered == ["csv"]
    assert "Statistics need a spreadsheet" in p.text_output

    # ...and the ordinary flow still offers everything
    p2 = ScriptedPrompter(["csv"])
    try:
        wiz.ask_source(p2)
    except Exception:
        pass
    assert len(p2.offered_choices("What kind of data do you have?")) == 5


def test_the_optional_middle_mode_still_exists_for_callers(tmp_path,
                                                           study_csv):
    """A programmatic caller with no opinion gets the old behavior: asked,
    and allowed to decline."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        [], False, "Neither", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert any("statistics on the results" in q for _k, q in p.asked)
    assert not any("stats" in s["call"] and "describe" not in s["call"]
                   for s in result.preset["steps"])


def test_an_analysis_this_data_cannot_run_is_grayed_out_not_accepted(tmp_path):
    """A spreadsheet of nothing but labels cannot support a correlation.
    Ticking one used to be *accepted* and then quietly dropped, which left
    the stage with nothing chosen and the wizard bouncing back to the
    feature checklist -- a screen that could not fix it -- forever."""
    path = tmp_path / "labels.csv"
    # labels that actually repeat: two groups of two. a column of unique labels
    # is an id, and sixty groups of one isn't a comparison (the next test
    # covers that one)
    path.write_text("condition,answer\n"
                    "A,\"some words here\"\nA,\"other words here\"\n"
                    "B,\"more words here\"\nB,\"further words here\"\n",
                    encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        "fdr_bh", False,
        ":done", "Labels", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    offered = {c.value: c for c in
               p.offered_choices("Which statistics do you want?")}
    assert offered["stats_correlations"].disabled, \
        "an analysis that cannot run has to say so, not be accepted"
    assert "column of numbers" in offered["stats_correlations"].disabled
    assert not offered["stats_group_differences"].disabled
    assert any(s["call"] == "potato.stats.analyze_group_differences"
               for s in result.preset["steps"])


def test_a_spreadsheet_with_nothing_but_text_is_refused_where_it_is_fixable(
        tmp_path):
    """The analysis stage's own check fires four screens later, and the
    "+ run analyses" flow then sent the user back to the FEATURE checklist,
    where no answer can conjure a metadata column. The two screens bounced
    off each other until Esc twice."""
    path = tmp_path / "only_text.csv"
    path.write_text("answer\n\"some words here about potatoes\"\n"
                    "\"more words over here\"\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer"],     # every column is the text
        "csv", *browse_to(path), ["answer"],     # ...asked again, not looped
    ])
    with pytest.raises(Exception):
        wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    kinds = [q for _k, q in p.asked if q == "What kind of data do you have?"]
    assert len(kinds) == 2, "it re-asked the source, not the feature list"
    assert "nothing left to group by or predict" in p.text_output
    assert not any(q == "Which features do you want to extract?" for _k, q in p.asked), \
        "it never reached the feature checklist, which could not fix this"


def test_the_reason_survives_a_rejected_spreadsheet(tmp_path):
    """The renderer clears a reason once its question is answered, so
    setting it once outside the loop left a rejected file coming back to a
    one-option menu with nothing on screen saying why."""
    path = tmp_path / "only_text.csv"
    path.write_text("answer\n\"words here\"\n", encoding="utf-8")
    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer"],
        "csv", *browse_to(path), ["answer"],
    ])
    with pytest.raises(Exception):
        wiz.ask_source(p, analyses=True)
    assert p.text_output.count("Statistics need a spreadsheet") >= 2


def test_a_column_of_unique_labels_is_not_offered_as_groups(tmp_path):
    """A participant id is a column of labels too. Comparing sixty groups of
    one is not a comparison, so the group-differences row says so rather
    than letting someone pick an id and get a table of empty tests."""
    path = tmp_path / "ids.csv"
    rows = ["pid,answer"] + [f"p{i},\"words number {i} here\"" for i in range(8)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer"], False,
        ["readability"], "row",
        ["stats_group_differences"],
        False,  # no control variables
    ])
    with pytest.raises(Exception):
        wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    offered = {c.value: c for c in
               p.offered_choices("Which statistics do you want?")}
    assert "column of labels" in offered["stats_group_differences"].disabled


def test_a_pick_the_statistics_cannot_use_is_turned_back_at_the_checklist(
        tmp_path, study_csv):
    """
    An n-gram frequency list is a table of the corpus, not of each text, so
    a pick made only of it gives the statistics nothing to join. That used
    to be discovered two screens later -- after the level question -- and
    announced with a two-line note the layout then cut down to "Pick
    something else to extract, whose measures the statistics" (a real
    report). The checklist now says so itself, once, before asking anything
    else, and the second answer may keep the frequency list.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["ngram_frequencies"],                  # corpus-level: nothing joinable
        ["ngram_frequencies", "readability"],   # fixed, keeping the list
        "row",
        ["stats_correlations"], False, ["openness"],
        False,  # no control variables
        "fdr_bh", False,
        ":done", "Fixed", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    asked = [q for _k, q in p.asked]
    features = [i for i, q in enumerate(asked)
                if q == "Which features do you want to extract?"]
    level = asked.index("Measure every row on its own, or join rows together first?")
    assert len(features) == 2 and features[1] < level, \
        "the level was asked before the pick could be used"
    assert p.text_output.count("None of those produces") == 1
    assert "Nothing selected produces" not in p.text_output, \
        "the analysis stage was reached with an unusable pick"
    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.stats.analyze_correlations" in calls
    assert "potato.text.analyze_ngram_frequencies" in calls


def test_rows_the_statistics_cannot_use_are_marked_only_in_that_flow(
        tmp_path, study_csv):
    """The mark is a warning about a decision the user is about to make, so
    it appears where the decision matters and nowhere else."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path, analyses=True)
    marks = {c.value: c.annotation for c in
             p.offered_choices("Which features do you want to extract?")}
    assert marks["ngram_frequencies"] == wiz.NOT_FOR_STATISTICS
    assert marks["doc_term_matrix"] == "", "a per-text table was marked"
    assert marks["readability"] == ""

    plain = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row", [], False, "Plain", "save",
    ])
    wiz.run_wizard(plain, cwd=tmp_path)
    assert not any(c.annotation for c in
                   plain.offered_choices("Which features do you want to extract?"))


def test_a_document_term_matrix_can_carry_the_statistics(tmp_path, study_csv):
    """
    One row per text, keyed by text_id: a document-term matrix is a feature
    table, and predicting an outcome from n-gram frequencies is the classic
    open-vocabulary design. It was flagged as "a matrix, not features", so
    picking it with statistics was refused outright (a real report).
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["doc_term_matrix"], "row",
        ["stats_ridge_fit"], False, ["openness"],
        False,  # no control variables
        False,  # no filters
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    assert "None of those produces" not in p.text_output
    assemble = next(s for s in result.preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    assert assemble["with"]["feature_csvs"] == ["{{doc_term_matrix}}"]
    assert_valid_preset(result.preset)


def test_the_topic_models_matrix_never_becomes_a_step_of_its_own(tmp_path, study_csv):
    """
    This used to read "a table pulled in only as an input does not join": the
    topic model was wired to a shared document-term matrix step, and the test
    checked that the matrix ran but stayed out of the analysis table -- nobody
    asking for topic scores wants five thousand term columns correlated with
    their outcome as well.

    The topic model builds its own matrix now, so the question does not arise:
    there is no matrix *step* to keep out of anything. Every topic model wants
    a different matrix (LDA needs counts, NMF wants tf-idf), and the shared one
    meant two of them rebuilding over each other's file.

    What still has to hold is the conclusion: one table reaches the statistics,
    the topic scores, and picking a single table is not a question worth asking.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["topic_model_mem"], "row",
        ["stats_correlations"], False, ["openness"],
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.text.topic_model_mem" in calls
    assert "potato.text.build_doc_term_matrix" not in calls, (
        "the topic model builds its own matrix; a shared step would be the "
        "clobbering this change removed")
    assemble = next(s for s in result.preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    assert assemble["with"]["feature_csvs"] == ["{{mem_topics}}"]
    assert not any(q == "Which feature tables should feed the statistics?"
                   for _k, q in p.asked), "one table is not a choice"


def test_a_matrix_and_the_topics_built_from_it_are_two_tables(tmp_path,
                                                             study_csv):
    """
    The rule that folds "Sentence embeddings" and its merge into one table is
    keyed on "one step consumes what the other produces". The topic model used
    to consume the document-term matrix, and folding them lost the matrix from
    the picker and offered the topic scores under its name.

    They are independent now -- the topic model builds its own matrix -- which
    makes the answer more obviously right rather than less: someone who asks
    for both a matrix and topic scores wants two tables, because they are two
    measures and not one re-shaped.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["doc_term_matrix", "topic_model_mem"], "row",
        ["stats_correlations"], False, ["openness"],
        False,  # no control variables
        ["doc_term_matrix", "topic_model_mem"], "together",
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    offered = {c.value: c.label for c in p.offered_choices(
        "Which feature tables should feed the statistics?")}
    assert offered == {"doc_term_matrix": "Document-term matrix",
                       "topic_model_mem": "Topic model: meaning extraction method"}
    assemble = next(s for s in result.preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    assert sorted(assemble["with"]["feature_csvs"]) == \
        ["{{doc_term_matrix}}", "{{mem_topics}}"]


def test_every_grayed_reason_fits_beside_its_label_at_eighty_columns(tmp_path):
    """
    questionary draws a disabled row's reason after its label on the same
    line, so the two have to share a row. "needs a model you saved from an
    earlier run; import one under Settings first" beside a 33-character
    label ran off the right edge of a real terminal (a real report). Every
    reason the wizard writes is held to the room a standard terminal has.
    """
    from taters.ui.wizard import NOT_FOR_STATISTICS

    path = tmp_path / "unique.csv"          # no numbers, no repeating labels
    rows = ["pid,note,text"]
    for i in range(8):
        rows.append(f"p{i},note{i},\"words number {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    p = EscapingPrompter([
        "csv", *browse_to(path), ["text"], False,
        ["readability"], "row", [], False, "Run", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    seen = 0
    for _q, choices in p.offered:
        for c in choices:
            if not c.disabled:
                continue
            seen += 1
            # 6 for questionary's glyphs and margin, 3 for " (" and ")"
            row = len(c.label) + 3 + len(c.disabled)
            if c.annotation:
                row += 1 + len(c.annotation)
            assert 6 + row <= 80, (c.label, c.disabled)
    # the library is empty under test, so the saved-model row is grayed, and
    # the spreadsheet above grays every analysis: so at least those
    assert seen >= 4, "the screens with grayed rows were not reached"
    assert len(NOT_FOR_STATISTICS) <= 22


def test_the_analysis_stages_answers_are_not_re_asked_on_the_options_screen():
    """
    Which column, which tables, together or apart, what to hold constant:
    the analysis stage asks these with the spreadsheet's columns on screen.
    Offered again on the options screen as free text ("feature sets: ›"
    with the docstring's "{name: [columns]}" for help) they read as a
    second, unrelated question nobody could answer (a real report).
    """
    from taters.ui import recipes as _r
    from taters.ui.wizard import is_wired, shared_variables

    owned = {"outcome_cols", "group_col", "feature_sets", "split_col",
             "control_cols", "categorical_controls"}
    analyses = [_r.by_id(i) for i in ("stats_group_differences",
                                      "stats_correlations", "stats_ridge_fit",
                                      "stats_classify_fit")]
    for recipe in analyses:
        for name in owned & set(recipe.with_):
            assert is_wired(recipe, name), (recipe.id, name)
        # whatever the stage does NOT answer stays tunable per step
        assert not is_wired(recipe, "pca")
    shared = shared_variables(analyses)
    assert "stats_feature_sets" not in shared
    assert "stats_outcome_cols" not in shared
    assert "stats_pca" in shared, "the premise no longer holds"


def test_bracketed_docstring_text_survives_the_options_screen():
    """
    ``{name: [columns]}`` and ``[col, op, value]`` are how the docstrings
    write a shape. rich reads a bracketed word as a style tag and swallows
    it, so the screen said "{name: }" (a real report).
    """
    import io

    from rich.console import Console

    from taters.ui.introspect import ParamSpec
    from taters.ui.prompts import QuestionaryPrompter

    prompter = QuestionaryPrompter()
    buf = io.StringIO()
    prompter._console = Console(file=buf, force_terminal=False, width=100)
    wiz._explain(prompter, ParamSpec(
        name="feature_sets",
        desc="None for all features as one set, or {name: [columns]}.",
        choices=["[a]", "b"]), None)
    out = buf.getvalue()
    assert "{name: [columns]}" in out
    assert "One of: [a], b" in out

def test_scoring_with_a_saved_model_is_grayed_out_until_there_is_one(
        tmp_path, study_csv):
    """Scoring with a model you fitted earlier needs that model to exist.
    The preflight screen offers to import one, but by then the user has
    already picked it from a list that gave no hint it was unavailable, so
    the row is grayed out on the checklist itself -- where it is offered."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh", False,
        ":done", "No model yet", "save"
    ])
    wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    offered = {c.value: c for c in
               p.offered_choices("Which features do you want to extract?")}
    assert "import one in Settings" in offered["score_with_model"].disabled
    # ...and the ones that need nothing extra can be picked
    assert not offered["readability"].disabled


def test_a_step_whose_extra_has_no_release_for_this_python_is_grayed_out(
        tmp_path, study_csv, monkeypatch):
    """
    On Python 3.14 there is no gensim, so "pip install taters[vectors]"
    installs nothing and word vectors can never run here. Offering the row
    plainly and refusing at the preflight (or, with the keep-anyway answer,
    at run time) sent somebody round in circles; the row is grayed out with
    the reason, where it is offered. Diarization on 3.14 is the same story.
    """
    from taters.ui.tasks import gpu

    # the wizard fixtures pretend everything is installed; here gensim isn't
    monkeypatch.setattr(wiz, "missing_extras",
                        lambda r: ["vectors"] if "vectors" in r.extras else [])
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "gensim")
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row", [], False, "Vec", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value: c for c in
               p.offered_choices("Which features do you want to extract?")}
    assert offered["word_vectors_train"].disabled.startswith("not available for Python")
    assert "gensim" in offered["word_vectors_train"].disabled
    assert not offered["readability"].disabled


def test_dropping_the_only_feature_at_the_preflight_returns_to_the_checklist(
        tmp_path, study_csv, monkeypatch):
    """
    Word vectors as the only feature, statistics asked for, then "No, leave
    it out" when the preflight found gensim missing: the statistics had
    nothing to assemble and the compose error escaped as a traceback (a
    laptop trial). Now the note says so and the checklist comes back.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["word_vectors_train"], "row",
        ["stats_group_differences"], False, "condition", False, "fdr_bh", False,
        "drop",                     # leave word vectors out -> nothing to extract
        ["readability"], "row",     # back at the checklist: pick something else
        ["stats_group_differences"], False, "condition", False, "fdr_bh", False,
        ":done", "Second try", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)
    calls = [s["call"] for s in result.preset["steps"]]
    assert any("readability" in c for c in calls)
    assert not any("word_vectors" in c for c in calls)
    assert any("needs at least one feature table" in n for n in p.output)


def test_a_merely_uninstalled_extra_leaves_the_row_tickable(tmp_path, study_csv, monkeypatch):
    """The other case has to stay as it was: an extra that *can* be installed
    is one pip command away, and the preflight offers to keep the step for
    exactly that reason."""
    from taters.ui.tasks import gpu

    # the wizard fixtures pretend everything is installed; here gensim isn't
    monkeypatch.setattr(wiz, "missing_extras",
                        lambda r: ["vectors"] if "vectors" in r.extras else [])
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: False)
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row", [], False, "Vec", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value: c for c in
               p.offered_choices("Which features do you want to extract?")}
    assert not offered["word_vectors_train"].disabled


def test_an_empty_library_never_grays_out_the_rows_you_supply_files_for(
        tmp_path, study_csv):
    """
    Only a *saved model* waits on an earlier run. Everything else the library
    holds is something you bring or that ships with Taters.

    The graying-out gate was applied to every library-backed row, which
    disabled "Dictionary counts (LIWC-style)" and "Archetype similarity" on a
    fresh install -- neither of which ships any files -- and told the user
    they needed "a model you saved from an earlier run" (a real report).
    Neither is a model. An archetype run embeds the text with
    sentence-transformers but scores it against definitions the user
    supplies, exactly like LIWC-style counting; an empty library there is an
    ordinary starting state, and the preflight screen already offers to
    import into it.

    Dictionary counts is the most-used feature in the app, so this grayed out
    the main reason a lot of people open Taters at all.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh", False,
        ":done", "Empty library", "save"
    ])
    wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    offered = {c.value: c for c in
               p.offered_choices("Which features do you want to extract?")}
    # nothing ships for these two, so an empty library is exactly the state a
    # brand-new install is in
    for row in ("dictionaries", "archetypes"):
        assert not offered[row].disabled, (
            f"{row} was grayed out on an empty library: "
            f"{offered[row].disabled!r}")
    # these two ship built-in lists, so they were never at risk. we assert it
    # anyway so a future change to what ships can't quietly disable them either
    for row in ("cohesion", "ngram_frequencies"):
        assert not offered[row].disabled
    # and the one row that really does need a previous run still says so
    assert "import one in Settings" in offered["score_with_model"].disabled


def test_an_imported_model_makes_that_row_available(tmp_path, study_csv,
                                                    monkeypatch):
    """The mirror image: with a model in the library the row is offered."""
    from taters.helpers import library

    home = tmp_path / "taters_home"
    monkeypatch.setenv("TATERS_HOME", str(home))
    kind = library.KINDS["models"]
    model = tmp_path / "ridge__all.json"
    model.write_text('{"kind": "taters-ridge-model", "format": 1,'
                     ' "predictors": ["a"], "outcomes": {"o": {"kept": [0],'
                     ' "mu": [0.0], "sigma": [1.0], "coef": [1.0],'
                     ' "intercept": 0.0}}}', encoding="utf-8")
    library.import_into(kind, model)

    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        False,  # no control variables
        "fdr_bh", False,
        ":done", "Has a model", "save"
    ])
    wiz.run_wizard(p, cwd=tmp_path, analyses=True)
    offered = {c.value: c for c in
               p.offered_choices("Which features do you want to extract?")}
    assert not offered["score_with_model"].disabled


def test_a_classification_model_asks_for_the_category_not_a_number(tmp_path,
                                                                   study_csv):
    """
    Classification was reachable from the API and from the model library --
    you could score a spreadsheet with a classifier somebody else fitted --
    but there was no way to *fit* one from the wizard, so the guides
    described a sibling step the screens did not offer. It asks for a label
    column, not the numeric one the ridge asks for.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_classify_fit"], False,
        ["condition"],            # the category to predict
        False,                    # no control variables
        False,                    # no filters
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("category to predict" in q for _k, q in p.asked)
    assert result.preset["vars"]["stats_class_cols"] == ["condition"]
    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.stats.fit_classifier_csv" in calls
    # it has to reach the analysis table like any other metadata column, or
    # the step gets handed a column name that isn't there
    assert "condition" in result.preset["vars"]["stats_meta_carry"]
    assert_valid_preset(result.preset)


def test_predicting_a_number_and_a_category_are_kept_apart(tmp_path,
                                                           study_csv):
    """
    One run can sensibly do both, and each step refuses the other's column
    by name. Sharing one variable would hand `openness` to the classifier
    and `condition` to the ridge, and the run would die at the last step
    having already extracted every feature.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_ridge_fit", "stats_classify_fit"], False,
        ["openness"],             # the number
        ["condition"],            # the category
        # no controls question: every other column is now the outcome, the
        # category, or the identifier
        False,                    # no filters
        ":done", "Study", "save"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["vars"]["stats_outcome_cols"] == ["openness"]
    assert result.preset["vars"]["stats_class_cols"] == ["condition"]
    outcomes = {s["call"]: s["with"]["outcome_cols"]
                for s in result.preset["steps"]
                if "outcome_cols" in s.get("with", {})}
    assert outcomes["potato.stats.fit_ridge_csv"] == "{{var:stats_outcome_cols}}"
    assert outcomes["potato.stats.fit_classifier_csv"] == \
        "{{var:stats_class_cols}}"


def test_classification_is_grayed_out_without_a_column_of_labels(tmp_path):
    """Same rule as the group comparison, for the same reason: a column of
    one-off values would be one class per row, which is nothing to learn."""
    path = tmp_path / "unique.csv"
    rows = ["pid,note,openness,text"]
    for i in range(8):
        rows.append(f"p{i},note{i},{3 + i * 0.1:.1f},\"words number {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    p = EscapingPrompter([
        "csv", *browse_to(path), ["text"], False,
        ["readability"], "row",
        [], False, "Run", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value: c.disabled for c in p.offered_choices(
        "Run any statistics on the results? (optional)")}
    assert offered["stats_classify_fit"], "offered with nothing to predict"
    assert not offered["stats_ridge_fit"], "a numeric outcome is available"


# ---------------------------------------------------------------------------
# Esc means the previous question, everywhere
# ---------------------------------------------------------------------------


def test_escape_at_the_id_columns_returns_to_the_id_question(tmp_path, study_csv):
    """
    Esc used to be caught only at stage boundaries. From the fifth
    spreadsheet question it left the wizard entirely; the previous question
    is "is there an id column?", and that is where it lands now -- with the
    earlier answer under the pointer, and the file and text columns kept.
    """
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"],
        True,                     # is there an id column?
        "__esc__",                # ...at "Which one(s)?"
        False,                    # changed my mind: no id column
        ["readability"], "row", [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert len(_asked(p, "Is there a column that identifies each row (an ID or key)?")) == 2
    assert len(_asked(p, "Where is your spreadsheet?")) == 1, "the file was re-asked"
    assert len(_asked(p, "Which column(s) hold the text you want analyzed?")) == 1
    # the second time around, the earlier "yes" is the default
    confirms = [(q, d) for (k, q), d in zip(p.asked, p.presented)
                if q.startswith("Is there a column that identifies")]
    assert [d for _q, d in confirms] == [False, True]
    assert result.preset["steps"], "the run did not complete"


def test_escape_at_the_text_columns_returns_to_the_file(tmp_path, study_csv):
    p = EscapingPrompter([
        "csv", *browse_to(study_csv),
        "__esc__",                # at the text-columns question
        *browse_to(study_csv),    # the file browser again, not the kind
        ["text"], False,
        ["readability"], "row", [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert len(_asked(p, "What kind of data do you have?")) == 1
    assert len(_asked(p, "Where is your spreadsheet?")) == 2


def test_escape_in_the_file_browser_returns_to_the_kind_of_data(tmp_path,
                                                                study_csv):
    """It used to climb out of the wizard altogether."""
    p = EscapingPrompter([
        "csv", "__esc__",         # Esc in the browser
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row", [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert len(_asked(p, "What kind of data do you have?")) == 2
    assert result.preset["steps"]


def test_escape_at_the_outcomes_returns_to_the_statistics_checklist(tmp_path,
                                                                    study_csv):
    """
    From the fourth analysis question Esc rewound to the feature checklist,
    throwing away the level and every analysis answer so far (a real report:
    "sometimes the exact previous screen, sometimes five questions back").
    """
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        "__esc__",                # at "Which column(s) hold the outcomes?"
        "__esc__",                # at "Average the measures...?" -- one back each time
        ["stats_correlations"], False,   # the checklist again, previous pick ticked
        ["openness"],
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert len(_asked(p, "Which features do you want to extract?")) == 1
    assert len(_asked(p, "Measure every row on its own, or join rows together first?")) == 1
    checklists = [choices for q, choices in p.offered
                  if q == "Run any statistics on the results? (optional)"]
    assert len(checklists) == 2
    ticked = {c.value for c in checklists[1] if c.checked}
    assert ticked == {"stats_correlations"}, "the earlier pick was not kept"
    assert result.preset["vars"]["stats_outcome_cols"] == ["openness"]


def test_escape_at_the_statistics_checklist_returns_to_the_level(tmp_path,
                                                                 study_csv):
    """The screen before the first analysis question is the level question,
    not the feature checklist -- and it comes back with the earlier answer
    under the pointer."""
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        "__esc__",                # at the statistics checklist
        "row",                    # the level again
        [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert len(_asked(p, "Which features do you want to extract?")) == 1
    assert len(_asked(p, "Measure every row on its own, or join rows together first?")) == 2
    levels = [d for (_k, q), d in zip(p.asked, p.presented)
              if q == "Measure every row on its own, or join rows together first?"]
    assert levels[1] == "row", "the earlier level was not under the pointer"


def test_escape_at_the_grouping_columns_returns_to_the_level(tmp_path,
                                                             study_csv):
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "group",
        "__esc__",                # at "Join the text from rows that share which column(s)?"
        "row",
        [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert len(_asked(p, "Which features do you want to extract?")) == 1
    assert len(_asked(p, "Measure every row on its own, or join rows together first?")) == 2


def test_going_back_and_changing_the_picks_leaves_no_stale_answer(tmp_path,
                                                                  study_csv):
    """An outcome column chosen for a correlation that was then unticked
    must not be wired into the pipeline as if it were still wanted."""
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"],
        "__esc__",                # at the controls question: back to outcomes
        "__esc__",                # at outcomes: back to averaging
        "__esc__",                # at averaging: back to the checklist
        ["stats_group_differences"], False, "condition",
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["vars"]["stats_group_col"] == "condition"
    assert not result.preset["vars"].get("stats_outcome_cols")
    assert not any("correlations" in s["call"] for s in result.preset["steps"])


def test_each_column_question_says_which_analysis_it_is_for(tmp_path, study_csv):
    """
    With a correlation and a classifier both ticked, the outcome question
    read as *the* outcome question: nothing said it served the analyses
    that want numbers, or that the classifier's category was a separate
    question one screen on -- so the category column was hunted for here.
    Each question now opens with the analyses it is for and, where it
    matters, what comes next.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations", "stats_classify_fit"], False,
        ["openness"],             # outcomes, for the correlations
        ["condition"],            # the category, for the classifier
        # (no control question: pid is an id, the other two are spoken for)
        "fdr_bh",                 # correction
        False,                    # no filters
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    outcomes = next(r for r in p.reasons if "*numbers*" in r)
    assert outcomes.startswith("For Correlations with outcomes, which relates")
    assert "categories are a separate question, asked next" in outcomes
    classes = next(r for r in p.reasons if "*category*" in r)
    assert classes.startswith("For Classification model")
    assert "different question from the outcomes you just chose" in classes


def test_the_outcome_question_does_not_promise_a_next_question_that_will_not_come(
        tmp_path, study_csv):
    """Without a classifier there is no category question, so the outcome
    question must not say one is coming."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["openness"],
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    outcomes = next(r for r in p.reasons if "*numbers*" in r)
    assert "asked next" not in outcomes
    assert not any("*category*" in r for r in p.reasons)


# ---------------------------------------------------------------------------
# Measure each row, then average the numbers
# ---------------------------------------------------------------------------

AVERAGE_Q = "Average each measure within a group before the statistics?"
AGAIN_Q = "Average again, by something coarser?"


@pytest.fixture
def turns_csv(tmp_path):
    """Conversation turns: the shape the two orders actually differ on."""
    path = tmp_path / "turns.csv"
    rows = ["conv,speaker,text,satisfaction"]
    for c in range(12):
        for speaker in ("alice", "bob", "cara"):
            for t in range(3):
                rows.append(f"c{c:02d},{speaker},"
                            f"\"words about potatoes number {c}{t}\","
                            f"{3 + (c % 5) * 0.5:.1f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _averaging_steps(preset) -> list:
    return [s for s in preset["steps"]
            if s["call"] == "potato.helpers.average_feature_table"]


def test_declining_to_average_leaves_the_run_exactly_as_it_was(tmp_path, turns_csv):
    """The default has to be the old behavior, answer for answer."""
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert _averaging_steps(result.preset) == []
    assert result.preset["vars"]["stats_outcome_cols"] == ["satisfaction"], \
        "an un-averaged run should read the outcome column as it stands"


def test_averaging_once_measures_every_row_and_then_collapses(tmp_path, turns_csv):
    """
    The plain case: measure each turn, average per conversation, correlate at
    conversation level. The text steps still see one row per turn.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    steps = _averaging_steps(result.preset)

    assert len(steps) == 1
    assert steps[0]["with"]["group_by"] == ["conv"]
    reader = next(s for s in result.preset["steps"]
                  if s["call"] == "potato.text.analyze_readability")
    assert not reader["with"].get("group_by"), \
        "the text was joined after all, which is the other order"


def test_averaging_can_be_repeated_to_weight_the_middle_level_equally(
        tmp_path, turns_csv):
    """
    Turns to speakers to conversations is not the same arithmetic as turns
    straight to conversations: the first weights the three speakers equally,
    the second weights whoever talked most. The loop is what lets somebody
    say which they meant.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"],
        True, ["conv", "speaker"],      # first: one row per speaker per conv
        True, ["conv"],                 # then: one row per conv
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    steps = _averaging_steps(result.preset)

    assert [s["with"]["group_by"] for s in steps] == [["conv", "speaker"],
                                                      ["conv"]]
    assert steps[1]["with"]["in_csv"] == "{{" + steps[0]["save_as"] + "}}", \
        "the second average read the raw table instead of the first average"


def test_averaging_again_can_only_go_coarser(tmp_path, turns_csv):
    """
    Once the rows are averaged together, every column but the ones they were
    grouped on is gone -- so a later round can only offer a subset. Offering
    the rest would be offering a grouping the table cannot do.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"],
        True, ["conv", "speaker"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    rounds = [cs for q, cs in p.offered if q.startswith("Average by")
              or q.startswith("Average again, by which")]

    assert len(rounds) == 2
    assert {c.value for c in rounds[1]} == {"conv", "speaker"}, \
        [c.value for c in rounds[1]]


def test_one_column_is_as_coarse_as_it_gets_so_it_stops_asking(tmp_path,
                                                                turns_csv):
    """A subset of one column is either itself or nothing, so there is no
    coarser grouping to offer and the loop ends without a question."""
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert not _asked(p, AGAIN_Q), "it offered a coarser cut of one column"


def test_the_outcome_is_averaged_to_match_even_with_no_text_joining(
        tmp_path, turns_csv):
    """
    The join and the averaging are different questions, and the outcome
    follows the *analyzed* grain rather than the joining one. Twelve
    conversations have thirty-six satisfaction scores between them; the
    analysis sees one per conversation, so the column arrives averaged and
    the analyses are pointed at the averaged name.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    v = result.preset["vars"]

    assert v["stats_outcome_cols"] == ["satisfaction_mean"]
    assert v["stats_meta_agg"] == {"satisfaction": "mean"}
    meta = next(s for s in result.preset["steps"]
                if s.get("save_as") == "stats_metadata")
    assert meta["with"]["group_by"] == ["conv"]


def test_joining_the_text_and_averaging_the_measures_can_both_happen(
        tmp_path, turns_csv):
    """
    The user's own example: join each speaker's turns within a conversation,
    measure that once, then average the speakers to conversation level. Two
    different operations at two different grains, in one run.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "group", ["conv", "speaker"], False,
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    reader = next(s for s in result.preset["steps"]
                  if s["call"] == "potato.text.analyze_readability")

    assert reader["with"]["group_by"] == ["conv", "speaker"], "text not joined"
    assert [s["with"]["group_by"] for s in _averaging_steps(result.preset)] \
        == [["conv"]]


def test_after_joining_there_is_nothing_to_average_by_but_the_joining_columns(
        tmp_path, turns_csv):
    """
    A row of results is already one joining key, so the only coarser cut
    left is a subset of those keys. Offering the rest would be offering a
    column the table no longer has.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "group", ["conv", "speaker"], False,
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered if q.startswith("Average by"))

    assert {c.value for c in offered} == {"conv", "speaker"}


def test_joining_on_one_column_leaves_nothing_to_average_so_it_is_not_asked(
        tmp_path, turns_csv):
    """One joining column has no coarser subset, so the question would have
    no answer worth giving."""
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "group", ["conv"], False,
        ["stats_correlations"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert not _asked(p, AVERAGE_Q)


def test_a_column_with_a_different_value_in_every_row_is_not_offered(
        tmp_path, study_csv):
    """
    Averaging by a column that groups nothing is a no-op wearing an
    average's name -- the same narrower question the group-comparison picker
    asks before offering a column.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["condition"],
        ["openness"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered if q.startswith("Average by"))

    assert "pid" not in {c.value for c in offered}, \
        "pid is unique per row and groups nothing"


def test_no_statistics_means_the_question_is_never_asked(tmp_path, turns_csv):
    """Averaging exists to serve the analyses. Without them there is nothing
    for it to serve, and the feature tables are the deliverable."""
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        [],                       # no statistics
        False, "Study", "save",   # no settings to change
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert not _asked(p, AVERAGE_Q)
    assert _averaging_steps(result.preset) == []


def test_the_shape_is_read_back_after_every_answer_that_moves_it(tmp_path,
                                                                  turns_csv):
    """
    The shape of the thing being analyzed is the easiest part of a pipeline
    to lose track of and the most expensive to get wrong, so it is said in
    plain words rather than left to be inferred from a level name.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"],
        True, ["conv", "speaker"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    said = "\n".join(p.output)

    # 12 conversations x 3 speakers x 3 turns. the counts are the part that
    # makes this actionable: "then one per conv" sounds reasonable and
    # "then one per conv (12 rows)" does not.
    assert "one row per spreadsheet row (108 rows)" in said
    assert "then one per conv + speaker (36 rows)" in said
    assert "then one per conv (12 rows)." in said


# ---------------------------------------------------------------------------
# Leaving rows out before anything is measured
# ---------------------------------------------------------------------------

OUT_Q = "Leave any rows out before joining?"
WHICH_COL_Q = "Leave rows out based on which column?"
ANOTHER_Q = "Leave rows out based on another column?"


@pytest.fixture
def screener_csv(tmp_path):
    """A spreadsheet with the shapes the filter question has to handle: a
    short list of labels, a wide range of numbers, and a free-text id."""
    path = tmp_path / "screener.csv"
    rows = ["pid,arm,age,note,text"]
    for i in range(60):
        rows.append(f"p{i},{'ABC'[i % 3]},{18 + i % 45},note-{i},"
                    f"\"words about potatoes number {i}\"")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_rows_are_left_out_on_what_the_spreadsheet_already_says(tmp_path,
                                                                screener_csv):
    """
    This used to ask for a minimum word count, which is neither available
    before anything has been measured nor a thing anybody knows in advance.
    What somebody does know at this point is what they collected.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "arm", ["A", "B"], False,     # leave arm C out
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    gather = next(s for s in result.preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert gather["with"]["row_filters"] == [["arm", "in", ["A", "B"]]]
    reader = next(s for s in result.preset["steps"]
                  if s["call"] == "potato.text.analyze_readability")
    assert reader["with"]["analysis_csv"] == "{{gathered_texts}}", \
        "the analyzer would gather again, without the filter"


def test_a_short_list_of_values_is_a_tick_list_with_its_row_counts(
        tmp_path, screener_csv):
    """
    Two questions, not four: which column, then which of its values stay.
    A column with a handful of values needs no operator, and the counts say
    what each tick costs.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "arm", ["A", "B"], False,
        [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered
                   if q == "Which values of arm should stay?")

    assert {c.value for c in offered} == {"A", "B", "C"}
    assert all(c.checked for c in offered), "they start as they are"
    assert "20 rows" in next(c.help for c in offered if c.value == "A")


def test_a_column_with_too_many_values_to_list_becomes_a_threshold(
        tmp_path, screener_csv):
    """45 different ages is not a tick list."""
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "age", ">=", "21", False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    gather = next(s for s in result.preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert gather["with"]["row_filters"] == [["age", ">=", 21.0]]


def test_a_value_that_is_not_a_number_is_asked_for_again(tmp_path,
                                                          screener_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "age", ">=", "twenty-one", "21", False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    gather = next(s for s in result.preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert gather["with"]["row_filters"] == [["age", ">=", 21.0]]
    assert any("not a number" in line for line in p.output)


def test_a_free_text_column_with_many_values_is_typed_rather_than_listed(
        tmp_path, screener_csv):
    """Sixty different notes have no short list worth showing, and no
    threshold either, so the values to leave out are typed."""
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "note", "note-3, note-4", False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    gather = next(s for s in result.preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert gather["with"]["row_filters"] == [
        ["note", "not_in", ["note-3", "note-4"]]]


def test_keeping_every_value_is_not_a_filter(tmp_path, screener_csv):
    """Ticking everything is the same as not filtering, and recording it as
    a filter would put a step in the pipeline that does nothing."""
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "arm", ["A", "B", "C"], False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert not [s for s in result.preset["steps"]
                if s.get("save_as") == "gathered_texts"]


def test_how_many_rows_survive_is_said_after_every_filter(tmp_path,
                                                           screener_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "arm", ["A", "B"], False,
        [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert any("Keeping 40 of 60 rows" in line for line in p.output)


def test_a_filter_that_would_leave_nothing_is_refused(tmp_path, screener_csv):
    """Untick everything and there is no corpus. Said, and not recorded."""
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "arm", [], False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("nothing would be left" in line for line in p.output)
    assert not [s for s in result.preset["steps"]
                if s.get("save_as") == "gathered_texts"]


def test_two_filters_both_have_to_be_cleared(tmp_path, screener_csv):
    """Two filters are two things you asked to leave out, so a row has to
    clear both."""
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"],
        True, "arm", ["A", "B"], True, "age", ">=", "21", False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    gather = next(s for s in result.preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert gather["with"]["row_filters"] == [["arm", "in", ["A", "B"]],
                                             ["age", ">=", 21.0]]


def test_the_question_is_only_asked_where_it_cannot_be_done_afterwards(
        tmp_path, screener_csv):
    """
    Without joining there is no hurry: the filter question in the statistics
    stage does the same job and can use anything the run has measured by
    then, not just what came in the file.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "row",
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert not _asked(p, OUT_Q)
    assert not [s for s in result.preset["steps"]
                if s.get("save_as") == "gathered_texts"]


def test_declining_leaves_the_gather_where_it_has_always_been(tmp_path,
                                                               screener_csv):
    """No extra step, and the analyzers gather for themselves as before."""
    p = ScriptedPrompter([
        "csv", *browse_to(screener_csv), ["text"], False,
        ["readability"], "group", ["arm"], False,
        [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    reader = next(s for s in result.preset["steps"]
                  if s["call"] == "potato.text.analyze_readability")

    assert not [s for s in result.preset["steps"]
                if s.get("save_as") == "gathered_texts"]
    assert reader["with"]["csv_path"] == "{{var:input_csv}}"


def test_the_columns_the_averaging_needs_survive_the_early_gather(tmp_path,
                                                                   turns_csv):
    """
    The gather writes the analysis-ready table the analyzers read, so any
    column a later average groups by has to be carried through it. Left out,
    the averaging step would have nothing to group on and the run would die
    after every feature had been measured.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "group", ["conv", "speaker"],
        True, "speaker", ["alice", "bob"], False,
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    gather = next(s for s in result.preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert set(gather["with"]["carry_cols"]) >= {"conv", "speaker"}



# ---------------------------------------------------------------------------
# Saying why a control list is short
# ---------------------------------------------------------------------------

def test_the_controls_question_says_why_it_is_not_offering_your_columns(
        tmp_path, turns_csv):
    """
    The report this came from: a file of 143 columns, averaged to one row per
    education level, offered 14 columns as controls and withheld age and
    gender. Withholding them is right -- there is no one age for everybody
    with a bachelor's degree -- but saying nothing made it read as a bug, and
    the help text named age and gender as its two examples while refusing
    them.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"],
        True, ["conv"],           # controls: only conv survives the grain
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    said = "\n".join(p.reasons)

    assert "One row of results is one conv" in said
    assert "single value per conv" in said
    assert "speaker" in said, "the withheld column is not named"
    assert "age, gender" not in said, \
        "the help still offers as examples the columns it withholds"
    # both questions carry it: on the gate, so the answer is informed, and on
    # the picker, so a short list is explained where the short list is
    gate = next(r for r in p.reasons if r.startswith("A control is"))
    picker = next(r for r in p.reasons if r.startswith("Each column shows"))
    assert "single value per conv" in gate
    assert "single value per conv" in picker


def test_nothing_is_said_about_the_grain_when_nothing_is_withheld(tmp_path,
                                                                   study_csv):
    """An un-averaged run withholds nothing, so the caveat would be noise."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["openness"],
        True, ["condition"],
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert not any("single value per" in r for r in p.reasons)


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------

def test_the_spreadsheet_is_read_once_and_the_answer_carried(tmp_path,
                                                              turns_csv,
                                                              monkeypatch):
    """
    Three screens ask about the same file -- the source stage, the level
    question, the analysis stage -- and each used to read it for itself.
    Three passes over two hundred rows was cheap and wrong; three passes over
    a whole file is neither.
    """
    real = wiz.inspect_csv
    calls = []

    def counted(*a, **kw):
        calls.append(a[0] if a else kw.get("path"))
        return real(*a, **kw)

    monkeypatch.setattr(wiz, "inspect_csv", counted)
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], True, ["conv"], "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert len(calls) == 1, f"the file was read {len(calls)} times"


def test_choosing_a_file_says_it_is_being_read(tmp_path, turns_csv):
    """An infinite bar with a climbing count, because the size is not known
    until it has been read."""
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row", [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert any("Reading turns.csv" in line for line in p.output)
    assert getattr(p, "scanned", []), "no row count was ever reported"


def test_a_file_whose_rows_do_not_match_its_header_says_so_at_the_start(
        tmp_path):
    """
    Worth saying when the file is chosen. Said any later and it is said
    after an hour of extraction, which is when a quoting problem used to
    surface.
    """
    broken = tmp_path / "broken.csv"
    broken.write_text('g,n,text\na,1,"hi"\nb,2,"oops",extra\nc,3,"ok"\n',
                      encoding="utf-8")
    p = ScriptedPrompter([
        "csv", *browse_to(broken), ["text"], False,
        ["readability"], "row", [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert any("do not match the header" in line for line in p.output)


def test_a_column_that_never_varies_is_not_offered_as_a_control(tmp_path):
    """
    You cannot hold constant something that is already constant, and a model
    fitted on it spends a degree of freedom to learn nothing. Twelve of the
    fourteen columns a real run offered here were like this: a survey's
    `Finished` column reading 1 for every row, and ten items answered by
    half the sample and blank for the rest.
    """
    sheet = tmp_path / "survey.csv"
    rows = ["pid,condition,openness,Finished,Item_01,text"]
    for i in range(30):
        rows.append(f"p{i},{'ABC'[i % 3]},{3 + (i % 10) * 0.1:.1f},1,"
                    f"{'30' if i % 2 else ''},"
                    f"\"some words about potatoes number {i}\"")
    sheet.write_text("\n".join(rows) + "\n", encoding="utf-8")
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["openness"],
        True, ["condition"],
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value for c in next(
        cs for q, cs in p.offered if q == "Which column(s) should be held constant?")}

    assert "Finished" not in offered, "one value for every row"
    assert "Item_01" not in offered, "one value where answered, blank elsewhere"
    assert "condition" in offered, "a column that does vary is still offered"


# ---------------------------------------------------------------------------
# How many of each -- shown where the grouping is chosen
# ---------------------------------------------------------------------------

def test_a_grouping_picker_says_how_many_values_each_column_has(tmp_path,
                                                                 turns_csv):
    """
    The number *is* the decision. Grouping 938 rows by a column with seven
    values leaves seven rows, and a run with seven rows is not a run -- and
    it was possible to answer this question without ever being told.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"], True, ["conv"], "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value: c.annotation for c in next(
        cs for q, cs in p.offered if q.startswith("Average by"))}

    assert "12 unique" in offered["conv"], offered["conv"]
    assert "3 unique" in offered["speaker"], offered["speaker"]


def test_the_join_picker_says_it_too(tmp_path, turns_csv):
    """Both moments where rows get combined, since the question is the same
    one: how many rows will this leave?"""
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "group", ["conv"], False,
        [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value: c.annotation for c in next(
        cs for q, cs in p.offered
        if q == "Join the text from rows that share which column(s)?")}

    assert "12 unique" in offered["conv"]


def test_the_count_survives_pressing_an_arrow_on_the_row():
    """
    The arrows change how a column is read, and the renderer rewrites that
    row's annotation from the cycler's return value -- so a cycler that did
    not know about the counts would quietly wipe one off the row somebody
    was pointing at.
    """
    from taters.ui.wizard import _cycler, _kind_rows

    rows = [{"g": "a", "n": str(i % 4)} for i in range(20)]
    kinds = {"g": "labels", "n": "numbers"}
    counts = {"g": 1, "n": 4}
    before = {c.value: c.annotation
              for c in _kind_rows(["g", "n"], kinds, counts=counts)}
    after = _cycler(kinds, rows, ["g", "n"], counts=counts)("n", 1)

    assert "4 unique" in before["n"]
    assert after is not None and "4 unique" in after
    assert kinds["n"] == "labels", "the arrow did not change the kind"


# ---------------------------------------------------------------------------
# Controlling for how much went into each group
# ---------------------------------------------------------------------------

@pytest.fixture
def uneven_csv(tmp_path):
    """Conversations of different lengths -- the shape that makes the group
    size worth holding constant."""
    path = tmp_path / "uneven.csv"
    rows = ["conv,speaker,text,satisfaction"]
    for c in range(14):
        for t in range(2 + c):          # 2 turns in the first, 15 in the last
            rows.append(f"c{c:02d},{'abc'[t % 3]},"
                        f"\"words about potatoes number {c}{t}\","
                        f"{3 + (c % 5) * 0.5:.1f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_the_group_size_can_be_held_constant_on_a_combined_run(tmp_path,
                                                                uneven_csv):
    """
    The confound a real run walked into and could not ask about. Joining 938
    responses by age gave documents from 75 to 11,000 words; document length
    correlated with group size at .99, and the three largest coefficients in
    the winning model were its three most length-sensitive measures. The
    number was sitting in the analysis table and this question -- built from
    the spreadsheet's own columns -- could not see it, because the gather
    manufactures it rather than reading it.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(uneven_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"],
        True, ["group_count"],
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered
                   if q == "Which column(s) should be held constant?")
    row = next(c for c in offered if c.value == "group_count")

    assert "how many rows each group was built from" in row.help
    assert result.preset["vars"]["stats_control_cols"] == ["group_count"]


def test_the_group_size_is_not_asked_of_the_spreadsheet_it_never_came_from(
        tmp_path, uneven_csv):
    """
    The gather writes it, so asking the metadata step to *carry* it would
    name a column the file does not have -- a warning, and no column.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(uneven_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], True, ["conv"],
        ["satisfaction"],
        True, ["group_count"],
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert "group_count" not in result.preset["vars"].get("stats_meta_carry", [])
    assert "group_count" not in result.preset["vars"].get("stats_categorical_controls", [])


def test_an_uncombined_run_has_no_group_size_to_offer(tmp_path, study_csv):
    """There are no groups, so there is no group size -- and `group_count`
    is not a column of the file."""
    p = ScriptedPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["openness"],
        True, ["condition"],
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered
                   if q == "Which column(s) should be held constant?")

    assert "group_count" not in {c.value for c in offered}


def test_groups_that_are_all_the_same_size_are_nothing_to_hold_constant(
        tmp_path, turns_csv):
    """
    Every conversation has exactly nine turns in this fixture, so the group
    size never varies and offering it would spend a degree of freedom to
    learn nothing -- the same rule the other columns get.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(turns_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"],
        True, ["conv", "speaker"], False,   # three turns each, every group
        ["satisfaction"],
        True, ["conv"],
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered
                   if q == "Which column(s) should be held constant?")

    assert {c.value for c in offered} == {"conv", "speaker"}
    assert "group_count" not in {c.value for c in offered}
