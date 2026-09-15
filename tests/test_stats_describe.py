"""
Tests for `taters.stats.describe`: one table of descriptive statistics per
feature table, whether or not any statistics were asked for.

`describe_values` is a pure function over one column, so the numbers are
checked against hand-computed values; the file-level tests check what is
described, what is skipped, and that a second run writes nothing.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from taters.stats.describe import (COLUMNS, describe_features,  # noqa: E402
                                   describe_values)
from csvhelpers import _read, write_rows  # noqa: E402


def test_the_textbook_numbers_come_out():
    """Mean, n-1 standard deviation, quartiles and range of a small column,
    with a blank counted as missing and not as a value."""
    s = describe_values([1.0, 2.0, 3.0, 4.0, 10.0, float("nan")])
    assert s["n"] == 5 and s["missing"] == 1
    assert s["mean"] == 4.0
    assert abs(s["sd"] - 3.5355) < 1e-3
    assert (s["min"], s["median"], s["max"]) == (1.0, 3.0, 10.0)
    assert s["q1"] == 2.0 and s["q3"] == 4.0
    assert s["zeros"] == 0 and s["unique"] == 5


def test_skew_and_kurtosis_are_the_bias_adjusted_sample_statistics():
    """Zero skew for a symmetric column, positive for a long right tail; the
    kurtosis of a spread-out symmetric set is negative (flatter than normal)
    -- the G1/G2 pair SPSS and R's psych::describe(type=2) print."""
    symmetric = describe_values([1, 2, 3, 4, 5, 6, 7])
    assert abs(symmetric["skew"]) < 1e-9
    assert symmetric["kurtosis"] < 0
    skewed = describe_values([1, 1, 1, 2, 2, 3, 12])
    assert skewed["skew"] > 1.5
    # now let's check it against the formula by hand: G1 = n / ((n-1)(n-2)) * sum(z^3)
    x = np.array([1, 1, 1, 2, 2, 3, 12], dtype=float)
    z = (x - x.mean()) / x.std(ddof=1)
    n = len(x)
    assert abs(skewed["skew"] - n / ((n - 1) * (n - 2)) * (z ** 3).sum()) < 1e-9


def test_too_few_values_give_blanks_not_divisions_by_zero():
    one = describe_values([3.0])
    assert one["n"] == 1 and one["mean"] == 3.0 and one["sd"] is None
    assert one["skew"] is None and one["kurtosis"] is None
    two = describe_values([3.0, 5.0])
    assert two["sd"] is not None and two["skew"] is None
    constant = describe_values([2.0, 2.0, 2.0, 2.0])
    assert constant["sd"] == 0.0 and constant["skew"] == 0.0
    assert constant["kurtosis"] is None and constant["unique"] == 1
    empty = describe_values([float("nan")] * 3)
    assert empty["n"] == 0 and empty["missing"] == 3 and empty["mean"] is None


def test_every_numeric_measure_is_described_and_nothing_else(tmp_path):
    """Identifiers and word columns are skipped; count columns are not --
    they are the first thing to look at when a measure looks strange."""
    table = write_rows(tmp_path / "features" / "readability.csv",
                       ["text_id", "source", "label", "score", "token_count", "gap"],
                       [["1", "a.txt", "yes", "0.5", "12", ""],
                        ["2", "b.txt", "no", "1.5", "30", "4"],
                        ["3", "c.txt", "yes", "", "7", "6"]])
    out = describe_features([table], tmp_path / "stats_descriptives", verbose=False)
    rows = {r["variable"]: r for r in _read(out / "readability.csv")}
    assert list(rows) == ["score", "token_count", "gap"]
    assert list(_read(out / "readability.csv")[0].keys()) == COLUMNS
    assert rows["score"]["n"] == "2" and rows["score"]["missing"] == "1"
    assert rows["gap"]["mean"] == "5.0"
    assert rows["token_count"]["max"] == "30.0"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "`readability.csv`" in readme and "skewness" in readme


def test_a_second_run_rewrites_nothing_and_a_changed_table_is_redone(tmp_path):
    """The resume contract: newer descriptives than their table stand; a
    table edited afterwards gets fresh ones."""
    import os
    import time

    table = write_rows(tmp_path / "f.csv", ["text_id", "x"], [["1", "1"], ["2", "3"]])
    out = describe_features([table], tmp_path / "d", verbose=False)
    first = (out / "f.csv").stat().st_mtime_ns
    describe_features([table], tmp_path / "d", verbose=False)
    assert (out / "f.csv").stat().st_mtime_ns == first
    write_rows(table, ["text_id", "x"], [["1", "1"], ["2", "9"]])
    later = time.time() + 5
    os.utime(table, (later, later))
    describe_features([table], tmp_path / "d", verbose=False)
    assert _read(out / "f.csv")[0]["mean"] == "5.0"


def test_a_missing_table_is_refused_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.csv"):
        describe_features([tmp_path / "nowhere.csv"], tmp_path / "d", verbose=False)


def test_nothing_to_describe_writes_nothing(tmp_path):
    out = describe_features([], tmp_path / "d", verbose=False)
    assert not out.exists()
