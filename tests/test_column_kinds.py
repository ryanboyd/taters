"""
Every column picker is a two-column table: the name, and what the column is
treated as. The left and right arrows change the treatment on the row.

A 1/2 gender code reads as numbers and is a category; a column of words that
repeat is a set of labels; a column of free writing fits nowhere but the text
question. Showing the treatment where the choice is made, and letting it be
changed there, replaced a follow-up question ("which of these are categories
rather than measurements?") that only made sense to people who already knew
why it was being asked.
"""
from __future__ import annotations

import csv

import pytest

from taters.ui import wizard as wiz
from taters.ui.columns import column_kind, kind_options
from taters.ui.prompts import Choice, ScriptedPrompter, with_annotation

from wizard_helpers import browse_to, clean_machine  # noqa: F401


# ---------------------------------------------------------------------------
# Detection and what a column may become
# ---------------------------------------------------------------------------

def test_a_column_is_numbers_labels_text_or_empty():
    assert column_kind(["1", "2", "1", "2"]) == "numbers"
    assert column_kind(["3.5", "", "4"]) == "numbers"
    assert column_kind(["a", "b", "a", "b"]) == "labels"
    assert column_kind([f"p{i}" for i in range(30)]) == "text", \
        "one value per row is an identifier, not a set of labels"
    assert column_kind(["some words", "other words", "more"]) == "text"
    assert column_kind(["", "", None]) == "empty"


def test_numbers_may_always_become_labels():
    """Whether an analysis can use fifty labels is checked when the column
    is chosen, in words; the arrow itself never refuses (a real request)."""
    assert kind_options("numbers", ["1", "2", "1", "2", "3"]) == ["numbers", "labels"]
    assert kind_options("numbers", [str(i) for i in range(50)]) == ["numbers", "labels"]
    assert kind_options("numbers", []) == ["numbers"]
    # labels only go back to numbers if every value parses; text stays text.
    assert kind_options("labels", ["1", "2", "1"]) == ["labels", "numbers"]
    assert kind_options("labels", ["a", "b", "a"]) == ["labels"]
    assert kind_options("text", ["hello there"]) == ["text"]


def test_a_rendered_row_takes_a_new_annotation_in_place():
    plain = with_annotation("gender", "labels")
    assert plain == [("class:text", "gender"), ("class:annotation", " labels")]
    again = with_annotation(plain, "numbers")
    assert again[-1] == ("class:annotation", " numbers") and again[0][1] == "gender"
    tagged = with_annotation([("class:text", "[x] gender")], "labels")
    assert tagged == [("class:text", "[x] gender"), ("class:annotation", " labels")]


# ---------------------------------------------------------------------------
# The wizard's pickers
# ---------------------------------------------------------------------------

def _sheet(tmp_path, header, rows):
    path = tmp_path / "study.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _coded(tmp_path, n=30):
    """pid, a 1/2/3 condition code, a 1/2 gender code, an age, an outcome, text."""
    return _sheet(tmp_path, ["pid", "cond", "gender", "age", "openness", "text"],
                  [[f"p{i}", 1 + i % 3, 1 + i % 2, 20 + i, f"{3 + (i % 10) * .1:.1f}",
                    f"words about potatoes {i}"] for i in range(n)])


def _annotations(p, question):
    return {c.value: c.annotation.strip()
            for c in p.offered_choices(question)}


def test_every_column_picker_shows_what_each_column_is_treated_as(tmp_path):
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True, ["pid"],
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"],
        True, ["age"],
        "fdr_bh", False,
        ":done", "Kinds", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    text_rows = _annotations(p, "Which column(s) hold the text you want analyzed?")
    assert text_rows == {"pid": "text", "cond": "numbers", "gender": "numbers",
                         "age": "numbers", "openness": "numbers", "text": "text"}
    assert _annotations(p, "Which one(s)?")["pid"] == "text"
    assert _annotations(p, "Which column(s) hold the outcomes?")["openness"] == "numbers"
    assert _annotations(p, "Which column(s) should be held constant?")["age"] == "numbers"
    # under one heading we pad every annotation out to the widest name, so
    # they all line up.
    rows = p.offered_choices("Which column(s) hold the text you want analyzed?")
    starts = {len(c.label) + len(c.annotation) - len(c.annotation.strip())
              for c in rows}
    assert len(starts) == 1, "every kind starts in the same column"


def test_the_right_arrow_turns_a_coded_column_into_labels_and_it_stays_so(tmp_path):
    """The gender example: coded 1/2, read as numbers, and a category. One
    arrow press on its row makes it labels -- and the statistics then hold it
    constant as a category, with no separate question about it."""
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True, ["pid"],
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"],
        True,
        "\x00right:gender",             # → on gender: numbers -> labels
        ["age", "gender"],
        "fdr_bh", False,
        ":done", "Kinds", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert [(c, (a or "").strip()) for c, a in p.cycled] == [("gender", "labels")]
    v = result.preset["vars"]
    assert v["stats_control_cols"] == ["age", "gender"]
    assert v["stats_categorical_controls"] == ["gender"]
    assert not any("categories rather than measurements" in q for _k, q in p.asked)


def test_the_arrows_cycle_a_column_of_many_numbers_both_ways(tmp_path):
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True, ["pid"],
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"],
        True,
        "\x00right:age",                # thirty distinct ages: not labels
        "\x00left:age",
        ["age"],
        "fdr_bh", False,
        ":done", "Kinds", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    # right makes thirty distinct ages into labels (the person's call); left
    # turns them back. net result: numbers, and a numeric control.
    assert [(c, (a or "").strip()) for c, a in p.cycled] == [
        ("age", "labels"), ("age", "numbers")]
    assert result.preset["vars"]["stats_categorical_controls"] == []


def test_a_coded_column_picked_as_the_category_is_treated_as_labels(tmp_path):
    """Picking a 1/2/3 code as the class to predict says what it is; the
    wizard treats it as labels and says so, rather than refusing a column
    the person plainly meant."""
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True, ["pid"],
        ["readability"], "row",
        ["stats_classify_fit"], False, ["cond"],
        False, False,
        ":done", "Kinds", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert any("Treating cond as labels" in line for line in p.output)
    assert result.preset["vars"]["stats_class_cols"] == ["cond"]


def test_a_measurement_is_grayed_out_as_a_group_and_text_is_not_offered_as_an_outcome(tmp_path):
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True, ["pid"],
        ["readability"], "row",
        ["stats_group_differences", "stats_correlations"], False,
        "cond", ["openness"],
        False, "fdr_bh", False,
        ":done", "Kinds", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    groups = {c.value: c for c in p.offered_choices(
        "Which column separates the groups you want to compare?")}
    # a few repeating numbers get offered as groups; thirty distinct ages are
    # a measurement (though the arrows on an earlier picker can still make them
    # labels, and then we offer them); free text never gets offered.
    for name in ("cond", "gender", "openness"):
        assert not groups[name].disabled, name
    assert groups["age"].disabled == "a measurement, not labels"
    assert "text" not in groups
    assert groups["pid"].disabled == "free text"
    outcomes = [c.value for c in p.offered_choices("Which column(s) hold the outcomes?")]
    assert "text" not in outcomes and "pid" not in outcomes
    assert set(outcomes) == {"cond", "gender", "age", "openness"}


def test_a_treatment_changed_on_one_screen_holds_on_the_next(tmp_path):
    """Kinds live on the source spec, not on the screen: gender made labels
    while choosing a group is still labels when it comes up as a control."""
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True, ["pid"],
        ["readability"], "row",
        ["stats_group_differences"], False,
        "\x00right:gender", "cond",
        True, ["gender"],
        "fdr_bh", False,
        ":done", "Kinds", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert _annotations(p, "Which column(s) should be held constant?")["gender"] == "labels"
    assert result.preset["vars"]["stats_categorical_controls"] == ["gender"]


@pytest.mark.parametrize("key", ["\x1b[C", "\x1b[D"])
def test_the_live_arrows_repaint_the_row_with_its_new_treatment(key):
    """On a real terminal: the arrow calls the cycle and the row shows the
    new annotation without the list being rebuilt."""
    from test_live import render, staged as _staged  # noqa: F401
    from taters.ui.live import LivePrompter

    prompter = LivePrompter()
    seen = []

    def cycle(value, direction):
        seen.append((value, direction))
        return "labels"

    rows = [Choice("gender", "gender", annotation="numbers"),
            Choice("age", "age", annotation="numbers")]
    answer, drawn = render(prompter, key + "\r",
                           lambda p: p.select("Pick:", rows, cycle=cycle))
    assert answer == "gender"
    assert seen == [("gender", 1 if key == "\x1b[C" else -1)]
    assert "labels" in drawn
    assert "[←→] change type" in drawn


def test_the_arrows_work_on_the_text_and_id_pickers_too(tmp_path):
    """From a real screen: the first picker already shows every kind, so the
    treatment can be corrected there, before the statistics ever come up,
    and it holds all the way through."""
    sheet = _coded(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet),
        "\x00right:gender", ["text"],       # on the text picker: gender -> labels
        True, ["pid"],
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"],
        True, ["gender"],
        "fdr_bh", False,
        ":done", "Kinds", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert [(c, (a or "").strip()) for c, a in p.cycled] == [("gender", "labels")]
    assert _annotations(p, "Which one(s)?")["gender"] == "labels"
    assert _annotations(p, "Which column(s) should be held constant?")["gender"] == "labels"
    assert result.preset["vars"]["stats_categorical_controls"] == ["gender"]
