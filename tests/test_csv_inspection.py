"""
Reading a spreadsheet over before asking anything about it.

Every column question the wizard asks -- which hold numbers, which hold
labels that repeat, which could group rows, which could be held constant
within a group -- is built from what this read. So what it reads decides
whether those questions are right.

It used to read the first two hundred rows. The bug that changed it came from
a real trial: a file of 938 responses and 143 columns, analyzed at one row
per education level. Ten `Math_*` columns looked constant within education
across the sample and were not across the other 738 rows, so they were
offered as controls that would have arrived empty -- while age and gender,
correctly withheld because an education level has no single age, looked
simply missing. Reading everything fixes the first half; saying why fixes the
second, and that is tested with the controls question.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from taters.helpers import settings as user_settings
from taters.ui.columns import Inspection, inspect_csv, peek_csv


def _sheet(tmp_path, rows, header=("g", "n", "text")) -> Path:
    path = tmp_path / "study.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


@pytest.fixture
def wide(tmp_path) -> Path:
    """300 rows where the last one is the only thing that spoils a column."""
    rows = [["a" if i % 2 else "b", str(i), f"words number {i}"]
            for i in range(300)]
    rows[299][1] = "n/a"
    return _sheet(tmp_path, rows)


# ---------------------------------------------------------------------------
# What it reads
# ---------------------------------------------------------------------------

def test_the_whole_file_is_read_by_default(wide):
    found = inspect_csv(wide)

    assert found.scanned == 300
    assert found.truncated is False
    assert found.columns == ["g", "n", "text"]


def test_a_value_only_the_last_row_has_is_still_seen(wide):
    """
    The point of the whole exercise. A sample of 200 says `n` is a column of
    numbers; it is not, and every question built on that answer is wrong in
    a way nobody can see from the screen.
    """
    from taters.ui.columns import column_kind

    whole = inspect_csv(wide).rows
    sampled = inspect_csv(wide, limit=200).rows

    assert column_kind([r.get("n") for r in sampled]) == "numbers"
    assert column_kind([r.get("n") for r in whole]) != "numbers"


def test_a_limit_stops_the_scan_and_says_it_stopped(wide):
    found = inspect_csv(wide, limit=50)

    assert found.scanned == 50
    assert found.truncated is True


def test_a_limit_bigger_than_the_file_is_not_a_truncation(wide):
    assert inspect_csv(wide, limit=5000).truncated is False


def test_blank_lines_are_not_rows(tmp_path):
    path = tmp_path / "gappy.csv"
    path.write_text("g,n\na,1\n\nb,2\n\n", encoding="utf-8")

    assert inspect_csv(path).scanned == 2


# ---------------------------------------------------------------------------
# Saying so while it happens
# ---------------------------------------------------------------------------

def test_the_running_count_is_reported_and_ends_on_the_total(tmp_path):
    """
    A scan of a large file is the first slow thing the wizard does, and it
    used to happen in silence. The last call is the real total, so the line
    does not stop short of where it got to.
    """
    rows = [["a", str(i), "x"] for i in range(5000)]
    path = _sheet(tmp_path, rows)
    seen = []
    inspect_csv(path, on_rows=seen.append)

    assert len(seen) > 1, "a big file reported nothing on the way"
    assert seen == sorted(seen)
    assert seen[-1] == 5000


def test_a_small_file_still_reports_its_total_once(wide):
    seen = []
    inspect_csv(wide, limit=10, on_rows=seen.append)

    assert seen[-1] == 10


# ---------------------------------------------------------------------------
# Rows that do not match the header
# ---------------------------------------------------------------------------

def test_rows_wider_than_the_header_are_counted_and_described(tmp_path):
    """
    Usually an unescaped quote or the wrong separator. Worth saying when the
    file is chosen, because the alternative is finding out after an hour of
    extraction.
    """
    path = tmp_path / "broken.csv"
    path.write_text("g,n\na,1\nb,2,3,4\nc,3\n", encoding="utf-8")
    found = inspect_csv(path)

    assert found.over == 1 and found.under == 0
    assert found.ragged == 1
    assert "extra fields" in found.trouble()
    assert "3 rows" in found.trouble()


def test_rows_shorter_than_the_header_are_padded_not_dropped(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("g,n,text\na,1,hi\nb\n", encoding="utf-8")
    found = inspect_csv(path)

    assert found.under == 1
    assert found.scanned == 2, "the short row was thrown away"
    assert found.rows[1].get("text") == "", "a missing cell should read blank"
    assert "short of the header" in found.trouble()


def test_a_tidy_file_has_nothing_to_report(wide):
    assert inspect_csv(wide).ragged == 0
    assert inspect_csv(wide).trouble() == ""


# ---------------------------------------------------------------------------
# The rows themselves
# ---------------------------------------------------------------------------

def test_a_row_behaves_like_the_mapping_every_screen_expects(wide):
    row = inspect_csv(wide, limit=1).rows[0]

    assert row.get("g") == "b"
    assert row.get("nope", "fallback") == "fallback"
    assert row["n"] == "0"
    assert set(row) == {"g", "n", "text"}
    assert dict(row)["text"] == "words number 0"


def test_the_rows_do_not_each_carry_their_own_copy_of_the_header(tmp_path):
    """
    143 columns and a million rows is 143 million key references if every
    row is its own dict, and the wizard holds the whole scan to answer
    questions about it. One shared index, one tuple per row.
    """
    rows = [["a", str(i), "x"] for i in range(200)]
    scanned = inspect_csv(_sheet(tmp_path, rows)).rows

    assert scanned[0]._index is scanned[-1]._index


def test_the_cheap_peek_still_returns_a_header_and_a_few_rows(wide):
    columns, rows = peek_csv(wide, 3)

    assert columns == ["g", "n", "text"]
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# The setting
# ---------------------------------------------------------------------------

def test_everything_is_read_unless_somebody_says_otherwise(monkeypatch):
    monkeypatch.delenv(user_settings.INSPECT_ROWS_ENV, raising=False)

    assert user_settings.inspect_row_limit() == 0


def test_a_saved_limit_is_honored(monkeypatch, tmp_path):
    monkeypatch.delenv(user_settings.INSPECT_ROWS_ENV, raising=False)
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    user_settings.save_setting(user_settings.INSPECT_ROWS_KEY, 1000)

    assert user_settings.inspect_row_limit() == 1000
    user_settings.clear_setting(user_settings.INSPECT_ROWS_KEY)
    assert user_settings.inspect_row_limit() == 0


def test_the_environment_beats_the_saved_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    user_settings.save_setting(user_settings.INSPECT_ROWS_KEY, 1000)
    monkeypatch.setenv(user_settings.INSPECT_ROWS_ENV, "25")

    assert user_settings.inspect_row_limit() == 25


def test_a_hand_edited_setting_that_makes_no_sense_reads_as_everything(
        monkeypatch):
    """Not a reason to refuse to start, and reading it all is the safe
    answer."""
    monkeypatch.setenv(user_settings.INSPECT_ROWS_ENV, "lots")

    assert user_settings.inspect_row_limit() == 0


def test_the_setting_is_on_the_settings_menu():
    from taters.ui.tasks import settings

    assert "inspect_rows" in [t.id for t in settings.entries()]


def test_an_unreadable_file_does_not_take_the_screen_down_with_it(tmp_path):
    """`Inspection` has to be constructible as an empty answer, because the
    source stage falls back to one when the read raises."""
    empty = Inspection(columns=["a"], rows=[])

    assert empty.scanned == 0 and empty.trouble() == ""
