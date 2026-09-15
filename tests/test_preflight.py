"""
The checks that stop an analysis before a feature is extracted.

Each mirrors an engine's refusal (a test pins the thresholds together) and
says, in words, what would have gone wrong forty screens later -- plus which
values to keep when a row filter is the fix.
"""
from __future__ import annotations

from taters.ui import preflight as pf


def test_the_thresholds_are_the_engines_own():
    """A wizard that vets against a different number than the classifier
    refuses on would pass a class the run then rejects, or refuse one the
    run would have taken."""
    from taters.stats import classify

    assert pf.MIN_PER_CLASS == classify.MIN_PER_CLASS
    assert pf.MAX_CLASSES == classify.MAX_CLASSES


def test_a_repeating_id_is_named_with_its_worst_offenders():
    finding = pf.repeated_ids({"pid": ["a", "b", "a", "c", "a", "b"]})
    assert finding is not None and finding.fatal
    assert "2 value(s) repeat" in finding.message
    assert "'a' × 3" in finding.message and "'b' × 2" in finding.message
    assert pf.repeated_ids({"pid": ["a", "b", "c"]}) is None
    # when there are several id columns, it's the combination that has to be
    # unique (that's how the gather puts them together)
    assert pf.repeated_ids({"study": ["s1", "s1"], "pid": ["a", "b"]}) is None
    assert pf.repeated_ids({"study": ["s1", "s1"], "pid": ["a", "a"]}) is not None


def test_thin_classes_are_offered_for_dropping_when_two_remain():
    values = ["Female"] * 40 + ["Male"] * 30 + ["Other"] * 3 + ["Prefer not to say"]
    finding = pf.thin_classes(values, "gender")
    assert finding is not None and not finding.fatal
    assert finding.keep == ["Female", "Male"]
    assert finding.drop == {"Other": 3, "Prefer not to say": 1}
    assert "'Prefer not to say' (1 row)" in finding.message
    assert "'Other' (3 rows)" in finding.message


def test_a_class_column_nothing_can_rescue_is_fatal():
    """Three classes of three: no filter makes a classifier out of that, so
    the question is asked again rather than a checkbox offered."""
    finding = pf.thin_classes(["A", "B", "C"] * 3, "condition")
    assert finding is not None and finding.fatal and not finding.keep
    one = pf.thin_classes(["A"] * 20, "condition")
    assert one is not None and one.fatal and "at least two classes" in one.message
    many = pf.thin_classes([f"p{i}" for i in range(30)], "pid")
    assert many is not None and many.fatal and "identifier" in many.message
    assert pf.thin_classes(["yes"] * 10 + ["no"] * 8, "outcome") is None


def test_small_groups_are_offered_for_dropping_or_merely_noted():
    values = ["A"] * 20 + ["B"] * 15 + ["C"] * 1 + ["D"] * 3
    finding = pf.small_groups(values, "condition")
    assert finding is not None and not finding.fatal
    assert finding.keep == ["A", "B"]
    assert set(finding.drop) == {"C", "D"}
    # every group is small and there's nothing to drop, so we just get a warning
    # rather than a "drop these?" question
    noted = pf.small_groups(["A", "B", "C"] * 3, "condition")
    assert noted is not None and not noted.fatal and not noted.keep
    assert "should not be trusted" in noted.message
    # a group of one next to a single real group leaves nothing to compare, so
    # this one's fatal
    assert pf.small_groups(["A"] * 20 + ["B"], "condition").fatal
    assert pf.small_groups(["A"] * 20 + ["B"] * 20, "condition") is None


def test_an_outcome_with_words_in_it_is_fatal_and_names_the_cells():
    values = ["1.5", "2", "", "n/a", "3", "n/a", "missing"]
    finding = pf.non_numeric_outcome(values, "score")
    assert finding is not None and finding.fatal
    assert "'n/a' (2 rows)" in finding.message and "'missing' (1 row)" in finding.message
    assert pf.non_numeric_outcome(["1", "", "2.5"], "score") is None
    assert pf.non_numeric_outcome(["", ""], "score").fatal


def test_rare_control_levels_are_offered_for_dropping():
    values = ["Female"] * 40 + ["Male"] * 30 + ["Other"] * 2
    finding = pf.rare_levels(values, "gender")
    assert finding is not None and not finding.fatal
    assert finding.keep == ["Female", "Male"] and finding.drop == {"Other": 2}
    # dropping would leave just one level, so we fit it as-is and say so
    lone = pf.rare_levels(["Female"] * 40 + ["Other"] * 2, "gender")
    assert lone is not None and not lone.keep and "fitted as it is" in lone.message
    assert pf.rare_levels(["a"] * 10 + ["b"] * 10, "site") is None


def test_too_few_rows_is_said_before_anything_runs():
    assert pf.few_rows(9).fatal
    assert pf.few_rows(10) is None
