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
    whole = inspect_csv(wide)
    sampled = inspect_csv(wide, limit=200)

    assert sampled.kind("n") == "numbers"
    assert whole.kind("n") != "numbers"


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
    found = inspect_csv(path, keep_rows=5)

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
    row = inspect_csv(wide, limit=1, keep_rows=1).rows[0]

    assert row.get("g") == "b"
    assert row.get("nope", "fallback") == "fallback"
    assert row["n"] == "0"
    assert set(row) == {"g", "n", "text"}
    assert dict(row)["text"] == "words number 0"


def test_the_rows_do_not_each_carry_their_own_copy_of_the_header(tmp_path):
    """
    The few rows a peek keeps share one header index rather than each
    carrying a dict of their own. One shared index, one tuple per row.
    """
    rows = [["a", str(i), "x"] for i in range(200)]
    scanned = inspect_csv(_sheet(tmp_path, rows), keep_rows=200).rows

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


# ---------------------------------------------------------------------------
# The per-column stats must be a drop-in for walking the rows
# ---------------------------------------------------------------------------

def test_the_scan_stats_answer_every_picker_question_exactly(tmp_path):
    """
    The pickers used to walk all rows per column per screen -- five seconds
    for kinds, nine for labels, six for unique counts, on a 500k-row file,
    all silent. The scan now gathers the facts once, and every screen is a
    lookup. That is only allowed if the lookup gives the *same answer* as the
    walk did, on every branch, so this builds a column for each branch and
    checks the fast path against the original helpers.
    """
    import csv

    from taters.ui.columns import (column_kind, distinct_values, inspect_csv,
                                   kind_options, looks_numeric, plausible_labels)

    N = 40
    cols = {
        "num":       [f"{i * 1.5}" for i in range(N)],
        "num_blank": [f"{i}" if i % 3 else "" for i in range(N)],
        "lab":       [["a", "b", "c"][i % 3] for i in range(N)],
        "lab_num":   [["1", "2"][i % 2] for i in range(N)],       # a coded gender
        "text":      [f"word{i}" for i in range(N)],              # > MAX_LABELS
        "mixed":     [["7", "x", "7", "y"][i % 4] for i in range(N)],
        "empty":     [""] * N,
        "single":    ["same"] * N,
        "ids":       [f"id{i}" for i in range(20)] + [""] * (N - 20),
        "edge_20":   [f"v{i % 20}" for i in range(N)],            # exactly MAX_LABELS
        "edge_21":   [f"v{i % 21}" for i in range(N)],            # one over
    }
    path = tmp_path / "branches.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(zip(*cols.values()))

    insp = inspect_csv(path, keep_rows=N)
    rows = insp.rows
    assert len(rows) == N, "kept for the comparison only"
    assert set(insp.stats) == set(cols), "a stat per column"

    for c in cols:
        vals = [r.get(c) for r in rows]
        kind = column_kind(vals)
        assert insp.kind(c) == kind, c
        assert insp.plausible_labels(c) == plausible_labels(vals), c
        assert insp.looks_numeric(c) == looks_numeric(rows, c), c
        assert insp.kind_options(c, kind) == kind_options(kind, vals), c
        for limit in (4, 6, 12):
            assert insp.distinct(c, limit) == distinct_values(rows, c, limit), (c, limit)
        # exact unique count, the old way, for every column here (all are
        # far below the ceiling)
        slow_unique = len({str(r.get(c) or "").strip() for r in rows} - {""})
        assert insp.unique_counts([c])[c] == slow_unique, c


def test_unique_counts_are_exact_up_to_the_ceiling_and_capped_past_it(tmp_path):
    """
    Holding every distinct string of a free-text column is a memory hazard --
    393,725 of them for one column of a real file -- so the count is exact up
    to UNIQUE_CEILING and a sentinel past it. The boundary has to be right on
    the nose: exactly the ceiling is still exact, one more is capped.
    """
    import csv

    from taters.ui.columns import UNIQUE_CEILING, inspect_csv

    path = tmp_path / "ceiling.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["at", "over", "few"])
        for i in range(UNIQUE_CEILING + 1):
            w.writerow([f"a{min(i, UNIQUE_CEILING - 1)}",  # exactly CEILING distinct
                        f"o{i}",                             # CEILING + 1 distinct
                        f"f{i % 5}"])

    counts = inspect_csv(path).unique_counts(["at", "over", "few"])
    stats = inspect_csv(path).stats

    assert counts["at"] == UNIQUE_CEILING and not stats["at"].unique_over
    assert counts["over"] == UNIQUE_CEILING + 1 and stats["over"].unique_over
    assert counts["few"] == 5


# ---------------------------------------------------------------------------
# Nothing is held: the scan keeps a record per column and no rows
# ---------------------------------------------------------------------------

def test_the_scan_holds_no_rows(wide):
    """
    A 700 MB file held as Python strings is about three gigabytes, and the
    wizard was holding exactly that to answer questions a record per column
    answers. The default keeps nothing; a peek asks for what it wants.
    """
    assert inspect_csv(wide).rows == []
    assert len(inspect_csv(wide, keep_rows=3).rows) == 3
    assert inspect_csv(wide, keep_rows=3).scanned == 300, \
        "keeping a few rows does not stop the scan"


def test_value_counts_are_exact_while_a_column_is_short_enough_to_list(tmp_path):
    """The tick-list filter shows each value with how many rows hold it."""
    rows = [["a" if i % 3 else "b", str(i), "x"] for i in range(30)]
    insp = inspect_csv(_sheet(tmp_path, rows))

    assert insp.value_counts("g") == {"a": 20, "b": 10}, "commonest first"
    assert insp.value_counts("n") is None, "30 values is not a list to tick"
    assert insp.value_counts("text") == {"x": 30}


def _grouped(tmp_path):
    """
    Six rows in three groups. `steady` has one value per group, `shaky` does
    not, `gappy` is constant where it is filled and blank elsewhere, and
    `empty` never has anything.
    """
    header = ["g", "steady", "shaky", "gappy", "empty", "text"]
    rows = [
        ["a", "1", "x", "p", "", "t1"],
        ["a", "1", "y", "",  "", "t2"],
        ["b", "2", "x", "q", "", "t3"],
        ["b", "2", "x", "q", "", "t4"],
        ["b", "2", "x", "",  "", "t5"],
        ["c", "3", "z", "r", "", "t6"],
    ]
    return _sheet(tmp_path, rows, header=header)


def test_a_grain_pass_answers_the_grouping_questions_from_the_file(tmp_path):
    """
    The count, the sizes and which columns still speak for a whole group
    all used to come from walking the held rows. They come from one pass
    over the file now, and the answers have to be the ones the rows gave.
    """
    from taters.ui.columns import constant_within

    insp = inspect_csv(_grouped(tmp_path))
    kept = inspect_csv(_grouped(tmp_path), keep_rows=6).rows
    watch = ["steady", "shaky", "gappy", "empty", "text"]

    g = insp.grain(["g"], track=watch)

    assert g.groups == 3 and g.rows == 6
    assert (g.smallest, g.largest) == (1, 3) and g.sizes_differ
    assert g.constant("g"), "a key is constant within itself"
    assert g.constant("steady") and g.constant("gappy"), \
        "a blank does not disagree with anything"
    assert not g.constant("shaky") and not g.constant("text")
    assert g.constant("empty"), "nothing to disagree about"
    for c in ("steady", "shaky", "gappy", "text"):
        assert g.constant(c) == constant_within(kept, ["g"], c), c


def test_a_grain_is_read_once_and_widened_when_more_columns_are_asked_about(
        tmp_path):
    insp = inspect_csv(_grouped(tmp_path))

    first = insp.grain(["g"], track=["steady"])
    again = insp.grain(["g"], track=["steady"])
    wider = insp.grain(["g"], track=["shaky"])

    assert again is first, "the same question is not a second pass"
    assert wider is not first and wider.tracked >= {"steady", "shaky"}, \
        "a new column means one pass that covers both"
    assert insp.grain(["g"], track=["steady"]) is wider


def test_a_grain_on_several_keys_counts_their_combinations(tmp_path):
    insp = inspect_csv(_grouped(tmp_path))

    assert insp.grain(["g", "shaky"]).groups == 4     # a/x a/y b/x c/z


def test_survivors_come_from_the_tally_when_they_can_and_the_file_when_not(
        tmp_path):
    rows = [["a" if i % 3 else "b", str(i), "x"] for i in range(30)]
    insp = inspect_csv(_sheet(tmp_path, rows))

    assert insp.survivors([]) == 30
    assert insp.survivors([["g", "in", ["b"]]]) == 10
    assert insp.survivors([["g", "in", ["b"]], ["n", ">=", 15]]) == 5
    assert insp.survivors([["n", "<=", 4]]) == 5, "no tally for n; a pass"


def test_a_hand_built_inspection_answers_from_the_rows_it_was_given():
    """The wizard falls back to one when a file cannot be read, and the
    tests build them by hand; both must still get answers, just slowly."""
    rows = [{"g": "a", "v": "1"}, {"g": "a", "v": "1"}, {"g": "b", "v": "2"}]
    insp = Inspection(columns=["g", "v"], rows=rows)

    assert insp.grain(["g"], track=["v"]).groups == 2
    assert insp.grain(["g"], track=["v"]).constant("v")
    assert insp.survivors([["g", "in", ["a"]]]) == 2
    assert insp.value_counts("g") == {"a": 2, "b": 1}
