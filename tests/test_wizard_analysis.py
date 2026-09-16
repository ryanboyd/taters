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
        ["stats_group_differences"],
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_correlations"],
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
        ["readability"], "group", ["condition"],
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
        ["readability"], "group", ["condition"],
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
        ["readability"], "group", ["subreddit", "author"],
        ["stats_group_differences"],
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_ridge_fit"],
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences", "stats_correlations"],
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
        ["stats_ridge_fit"], ["openness"],
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
        ["stats_group_differences"], "condition",
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
        ["readability", "sentence_embeddings"], "group", ["condition"],
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
        ["stats_correlations"], ["openness"],
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
        ["stats_group_differences"], "condition",
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
        ["stats_correlations"], ["openness"],
        False,  # no control variables
        "fdr_bh", False,
        ":done", "Fixed", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=True)

    asked = [q for _k, q in p.asked]
    features = [i for i, q in enumerate(asked)
                if q == "Which features do you want to extract?"]
    level = asked.index("What should one row of results describe?")
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
        ["stats_correlations"], ["openness"], False, "fdr_bh", False,
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
        ["stats_ridge_fit"], ["openness"],
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
        ["stats_correlations"], ["openness"],
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
        ["stats_correlations"], ["openness"],
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition", False, "fdr_bh", False,
        "drop",                     # leave word vectors out -> nothing to extract
        ["readability"], "row",     # back at the checklist: pick something else
        ["stats_group_differences"], "condition", False, "fdr_bh", False,
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
        ["stats_group_differences"], "condition",
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
        ["stats_group_differences"], "condition",
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
        ["stats_classify_fit"],
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
        ["stats_ridge_fit", "stats_classify_fit"],
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
        ["stats_correlations"],
        "__esc__",                # at "Which column(s) hold the outcomes?"
        ["stats_correlations"],   # the checklist again, previous pick ticked
        ["openness"],
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert len(_asked(p, "Which features do you want to extract?")) == 1
    assert len(_asked(p, "What should one row of results describe?")) == 1
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
    assert len(_asked(p, "What should one row of results describe?")) == 2
    levels = [d for (_k, q), d in zip(p.asked, p.presented)
              if q == "What should one row of results describe?"]
    assert levels[1] == "row", "the earlier level was not under the pointer"


def test_escape_at_the_grouping_columns_returns_to_the_level(tmp_path,
                                                             study_csv):
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "group",
        "__esc__",                # at "Combine rows that share which column(s)?"
        "row",
        [], False, "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert len(_asked(p, "Which features do you want to extract?")) == 1
    assert len(_asked(p, "What should one row of results describe?")) == 2


def test_going_back_and_changing_the_picks_leaves_no_stale_answer(tmp_path,
                                                                  study_csv):
    """An outcome column chosen for a correlation that was then unticked
    must not be wired into the pipeline as if it were still wanted."""
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], ["openness"],
        "__esc__",                # at the controls question: back to outcomes
        "__esc__",                # at outcomes: back to the checklist
        ["stats_group_differences"], "condition",
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
        ["stats_correlations", "stats_classify_fit"],
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
        ["stats_correlations"],
        ["openness"],
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    outcomes = next(r for r in p.reasons if "*numbers*" in r)
    assert "asked next" not in outcomes
    assert not any("*category*" in r for r in p.reasons)
