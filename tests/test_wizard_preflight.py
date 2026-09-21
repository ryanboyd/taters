"""
The wizard vets the columns the statistics will run on before anything runs.

Every one of these is a run that used to die at the last step, after every
feature had been measured, with a refusal naming a column chosen forty
screens earlier. Now the column is read whole when it is chosen, and the
user is either asked again (nothing can fix it) or offered the values worth
keeping (a filter can).
"""
from __future__ import annotations

import csv

import pytest

from taters.ui import wizard as wiz
from taters.ui.prompts import ScriptedPrompter

from wizard_helpers import browse_to, clean_machine  # noqa: F401


def _sheet(tmp_path, rows, header=("pid", "condition", "openness", "gender", "text")):
    path = tmp_path / "study.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _rows(n, *, condition, gender=lambda i: "f" if i % 2 else "m",
          openness=lambda i: f"{3 + (i % 10) * 0.1:.1f}", pid=lambda i: f"p{i}"):
    return [[pid(i), condition(i), openness(i), gender(i),
             f"some words about potatoes number {i}"] for i in range(n)]


def test_a_thin_class_is_offered_for_dropping_and_becomes_a_filter(tmp_path):
    """The gender example: 400 males, 300 females, three "other", one
    "prefer not to say". A classifier cannot learn a class of three; the
    wizard says so when the column is picked and offers to keep the two it
    can, which becomes a row filter."""
    sheet = _sheet(tmp_path, _rows(
        40, condition=lambda i: ("Other" if i in (0, 1, 2) else
                                 "Prefer not to say" if i == 3 else
                                 "Female" if i % 2 else "Male")))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_classify_fit"], False,
        ["condition"],                  # the class column, thin classes and all
        ["Female", "Male"],             # ...and we keep just these two
        False,                          # no controls
        False,                          # no filters
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert any("Which values of condition" in q for _k, q in p.asked)
    assert any("fewer than 5 rows" in line for line in p.output)
    assert result.preset["vars"]["stats_filters"] == [
        ["condition", "in", ["Female", "Male"]]]
    assert result.preset["vars"]["stats_class_cols"] == ["condition"]


def test_a_class_column_nothing_can_rescue_is_asked_again(tmp_path):
    """Three classes of three: no filter helps, so the question comes back
    with the reason above it, and another column can be chosen."""
    sheet = _sheet(tmp_path, _rows(
        30, condition=lambda i: "ABC"[i % 3] if i < 9 else "D",
        gender=lambda i: "f" if i % 2 else "m"))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_classify_fit"], False,
        ["condition"],                  # A, B, C have 3 rows, D 21: only one usable
        ["gender"],                     # so we're asked again, and pick one that works
        False, False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    asked = [q for _k, q in p.asked if "category to predict" in q]
    assert len(asked) == 2, "the fatal finding must re-ask the question"
    assert any("cannot be classified" in line for line in p.output)
    assert result.preset["vars"]["stats_class_cols"] == ["gender"]
    assert "stats_filters" not in result.preset["vars"] \
        or result.preset["vars"]["stats_filters"] == []


def test_an_id_column_that_repeats_is_asked_again(tmp_path):
    """The failure behind a real run: an id that does not identify rows
    gives several rows one text_id, and the join finds nothing to join.
    Read whole, said at the question, re-asked."""
    sheet = _sheet(tmp_path, _rows(
        12, condition=lambda i: "AB"[i % 2], pid=lambda i: f"p{i // 3}")
        , header=("pid", "condition", "openness", "gender", "text"))
    rows = list(csv.reader(sheet.open(encoding="utf-8")))
    rows[0].append("rid")
    for i, r in enumerate(rows[1:]):
        r.append(f"r{i}")
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], True,
        ["pid"],                        # repeats: p0 × 3, p1 × 3, ...
        ["rid"],                        # asked again, and this one's unique
        ["readability"], "row", [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert [q for _k, q in p.asked].count("Which one(s)?") == 2
    assert any("does not identify each row" in line for line in p.output)
    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_readability"))
    assert step["with"]["id_cols"] == ["rid"]


def test_a_group_of_one_is_offered_for_dropping(tmp_path):
    sheet = _sheet(tmp_path, _rows(
        31, condition=lambda i: "Z" if i == 30 else "AB"[i % 2]))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False,
        "condition",                    # A 15, B 15, Z 1
        ["A", "B"],                     # keep the two we can actually compare
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert any("Which values of condition" in q for _k, q in p.asked)
    assert result.preset["vars"]["stats_filters"] == [["condition", "in", ["A", "B"]]]


def _sheet_with_a_bad_cell_far_down(tmp_path):
    """240 rows where row 230 of `openness` says "n/a", plus a clean `age`."""
    rows = _rows(240, condition=lambda i: "AB"[i % 2])
    rows[230][2] = "n/a"
    header = ("pid", "condition", "openness", "gender", "text")
    sheet = _sheet(tmp_path, rows, header=header)
    table = list(csv.reader(sheet.open(encoding="utf-8")))
    table[0].append("age")
    for i, r in enumerate(table[1:]):
        r.append(str(20 + i % 40))
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(table)
    return sheet


def test_a_column_with_a_bad_cell_far_down_is_never_offered_at_all(tmp_path):
    """
    Reading the whole file means the offer is right the first time. This
    used to be a sample of 200 rows, so `openness` was offered as a number,
    taken, and then rejected by a whole-column check one question later --
    correct in the end, and a question nobody should have had to answer.
    """
    sheet = _sheet_with_a_bad_cell_far_down(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["age"],
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    offered = next(cs for q, cs in p.offered
                   if q == "Which column(s) hold the outcomes?")

    assert [q for _k, q in p.asked].count(
        "Which column(s) hold the outcomes?") == 1, "it had to ask twice"
    assert "openness" not in {c.value for c in offered}
    assert result.preset["vars"]["stats_outcome_cols"] == ["age"]


def test_the_whole_column_check_still_catches_it_when_the_scan_is_capped(
        tmp_path, monkeypatch):
    """
    Someone with files big enough to lower the limit in Settings gets the
    old behavior, and the old safety net has to still be there for them: the
    column is offered, the whole column is then read, and the question comes
    back naming the cell that spoiled it.
    """
    monkeypatch.setenv("TATERS_INSPECT_ROWS", "200")
    sheet = _sheet_with_a_bad_cell_far_down(tmp_path)
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False,
        ["openness"],                   # the sample said this was numbers
        ["age"],                        # so we get asked again
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert [q for _k, q in p.asked].count("Which column(s) hold the outcomes?") == 2
    assert any("'n/a' (1 row)" in line for line in p.output)
    assert result.preset["vars"]["stats_outcome_cols"] == ["age"]


def test_a_rare_control_level_is_offered_for_dropping(tmp_path):
    sheet = _sheet(tmp_path, _rows(
        42, condition=lambda i: "AB"[i % 2],
        gender=lambda i: "Other" if i < 2 else ("Female" if i % 2 else "Male")))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"],
        True, ["gender"],               # control for gender
        ["Female", "Male"],             # ...but only the levels with enough rows
        "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert any("Which values of gender" in q for _k, q in p.asked)
    assert result.preset["vars"]["stats_filters"] == [["gender", "in", ["Female", "Male"]]]
    assert result.preset["vars"]["stats_control_cols"] == ["gender"]


def test_keeping_every_value_adds_no_filter(tmp_path):
    sheet = _sheet(tmp_path, _rows(
        31, condition=lambda i: "Z" if i == 30 else "AB"[i % 2]))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_group_differences"], False, "condition",
        ["A", "B", "Z"],                # never mind, keep them all
        False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert "stats_filters" not in result.preset["vars"] \
        or result.preset["vars"]["stats_filters"] == []


def test_a_label_column_is_filtered_by_ticking_the_values_to_keep(tmp_path):
    """The filter conversation: "only males and females" is one checkbox,
    not two typed conditions."""
    sheet = _sheet(tmp_path, _rows(
        40, condition=lambda i: "AB"[i % 2],
        gender=lambda i: ("Other" if i == 0 else "Prefer not to say" if i == 1
                          else "Female" if i % 2 else "Male")))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "row",
        ["stats_correlations"], False, ["openness"], False, "fdr_bh",
        True, ["original"], ["gender"],
        ["Female", "Male"],
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert any(q.startswith("Keep rows where gender is") for _k, q in p.asked)
    assert result.preset["vars"]["stats_filters"] == [["gender", "in", ["Female", "Male"]]]


@pytest.mark.parametrize("combined", [True])
def test_nothing_is_vetted_when_rows_are_combined(tmp_path, combined):
    """Combined rows are not the raw column: a class of three respondents
    may be thirty rows, and the raw count would refuse a fine analysis."""
    sheet = _sheet(tmp_path, _rows(
        12, condition=lambda i: "ABC"[i % 3], pid=lambda i: f"p{i // 2}"))
    p = ScriptedPrompter([
        "csv", *browse_to(sheet), ["text"], False,
        ["readability"], "group", ["pid"], False,
        ["stats_classify_fit"], ["condition"],
        False, False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["vars"]["stats_class_cols"] == ["condition"]
    assert not any("Which values of" in q for _k, q in p.asked)
