"""
"Wrangle data": the gather helpers behind a front-door conversation.

The flows run through the ScriptedPrompter against real files, so what is
asserted is the actual spreadsheet a user would get.
"""

from __future__ import annotations

import csv
import pytest

from taters.ui.browse import _CHOOSE
from taters.ui.prompts import ScriptedPrompter
from taters.ui.tasks import TaskContext, all_tasks
from taters.ui.tasks.wrangle import TASK
from csvhelpers import _read


def test_wrangle_is_the_first_thing_on_the_menu():
    """
    Getting data into shape is where every project starts, so it opens the
    list -- now one level down, under the row it shares with analyzing, since
    both start from a file that already exists. Still the first thing seen
    on the way in.
    """
    from taters.ui.tasks.data import entries

    tasks = all_tasks()
    assert tasks[0].id == "data"
    assert tasks[0].label == "Wrangle/Analyze data"
    assert [t.id for t in entries()] == ["wrangle", "analyze_spreadsheet"]
    assert entries()[0].label == "Wrangle data"


def test_a_folder_of_documents_becomes_a_spreadsheet(tmp_path):
    docs = tmp_path / "essays"
    docs.mkdir()
    (docs / "a.txt").write_text("The cat sat.", encoding="utf-8")
    (docs / "b.txt").write_text("The dog ran.", encoding="utf-8")
    out = tmp_path / "essays_out.csv"

    p = ScriptedPrompter([
        "folder",               # what are we wrangling?
        str(docs), _CHOOSE,     # browse into the folder, choose it
        True,                   # include subfolders
        str(out),               # output path
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = {r["text_id"]: r["text"] for r in _read(out)}
    assert rows == {"a": "The cat sat.", "b": "The dog ran."}
    assert any("Wrangled 2 row(s)" in line for line in p.output)


def test_a_csv_aggregates_by_the_intersection_of_columns(tmp_path):
    """The headline use: one row per subreddit-by-username, texts joined."""
    src = tmp_path / "posts.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subreddit", "username", "text"])
        w.writerows([
            ["cats", "alice", "meow one"],
            ["cats", "alice", "meow two"],
            ["cats", "bob", "purr"],
            ["dogs", "alice", "woof"],
        ])
    out = tmp_path / "by_user.csv"

    p = ScriptedPrompter([
        "csv",                          # what are we wrangling?
        str(src),                       # spreadsheet path
        ["text"],                       # text column(s)
        "combine",                      # shape: combine rows
        ["subreddit", "username"],      # the intersection
        str(out),                       # output path
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = {r["text_id"]: r for r in _read(out)}
    assert len(rows) == 3, "three unique subreddit-by-username combinations"
    alice_cats = next(r for k, r in rows.items() if "cats" in k and "alice" in k)
    assert alice_cats["text"] == "meow one meow two"
    assert alice_cats["group_count"] == "2"

    # we find the spreadsheet with the file picker, not a bare text field, so
    # the browse screen had better have been shown, with the file on it
    picker = p.offered_choices("Where is the spreadsheet?")
    assert any("posts.csv" in (c.label or "") for c in picker)


def test_a_csv_without_grouping_keeps_identifiers(tmp_path):
    src = tmp_path / "posts.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subreddit", "text"])
        w.writerows([["cats", "meow"], ["dogs", "woof"]])
    out = tmp_path / "flat.csv"

    p = ScriptedPrompter([
        "csv", str(src),
        ["text"],           # text column
        "as_is",            # shape: one row out per row in
        "subreddit",        # that column names each row
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    rows = _read(out)
    assert len(rows) == 2 and rows[0]["subreddit"] == "cats"


def test_an_existing_output_needs_an_explicit_yes(tmp_path):
    docs = tmp_path / "d"
    docs.mkdir()
    (docs / "a.txt").write_text("hello there", encoding="utf-8")
    out = tmp_path / "out.csv"
    out.write_text("old contents", encoding="utf-8")
    fresh = tmp_path / "fresh.csv"

    p = ScriptedPrompter([
        "folder", str(docs), _CHOOSE, True,
        str(out),           # first try: exists
        False,              # overwrite? no
        str(fresh),         # pick another name instead
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert out.read_text(encoding="utf-8") == "old contents"
    assert fresh.exists()


def _scored_posts(tmp_path):
    src = tmp_path / "posts.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["username", "score", "flair", "text"])
        w.writerows([
            ["alice", "3", "old-timer", "meow one"],
            ["alice", "4", "old-timer", "meow two"],
            ["bob", "10", "lurker", "purr"],
        ])
    return src


def test_number_columns_can_ride_along_as_averages(tmp_path):
    """Combining rows can also summarize numbers: a post's score becomes the
    person's average score, in its own column."""
    src = _scored_posts(tmp_path)
    out = tmp_path / "by_user.csv"

    p = ScriptedPrompter([
        "csv", str(src),
        ["text"],           # text column
        "combine",          # shape: combine rows
        ["username"],       # who belongs together
        True,               # average the number columns?
        ["score"],          # which
        False,              # nothing else brought along
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = {r["text_id"]: r for r in _read(out)}
    assert rows["alice"]["score_mean"] == "3.5"
    assert rows["alice"]["score_n"] == "2"
    assert rows["bob"]["score_mean"] == "10"

    # only columns that actually look numeric get offered for averaging, so
    # the question got asked at all here, and it never is for a text column
    offered = p.offered_choices("Average which column(s)?")
    assert [c.value for c in offered] == ["score"]


def test_the_live_display_does_not_break_the_csv_gather(tmp_path):
    """
    Regression: with a real console the display wrapper passed
    ``on_progress`` to ``csv_to_analysis_ready_csv``, which has no such
    parameter, and the whole task died with a TypeError mid-run. The wrapper
    now checks the gather's signature first.
    """
    import io

    from rich.console import Console

    src = _scored_posts(tmp_path)
    out = tmp_path / "by_user.csv"

    p = ScriptedPrompter([
        "csv", str(src),
        ["text"], "combine", ["username"],
        False,              # no averages
        False,              # nothing brought along
        str(out),
    ])
    p._console = Console(file=io.StringIO(), force_terminal=True, width=100)
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert len(_read(out)) == 2


def test_a_spreadsheet_without_text_can_still_be_wrangled(tmp_path):
    """Ticking "None — there's no text to wrangle" lets a numbers-only
    dataset be combined and summarized; the output has no text column."""
    from taters.ui.tasks.wrangle import _NO_TEXT

    src = _scored_posts(tmp_path)
    out = tmp_path / "by_user.csv"

    p = ScriptedPrompter([
        "csv", str(src),
        [_NO_TEXT],         # no text columns
        "combine",          # shape: combine rows
        ["username"],       # who belongs together
        True,               # average the number columns?
        ["score"],          # which
        False,              # nothing else brought along
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = {r["text_id"]: r for r in _read(out)}
    assert "text" not in rows["alice"]
    assert rows["alice"]["score_mean"] == "3.5"
    assert rows["alice"]["group_count"] == "2"
    # the closing note stays honest: no text column, so no "text step" claim
    assert not any("text step" in line for line in p.output)


def test_ticking_none_alongside_real_columns_keeps_the_columns(tmp_path):
    from taters.ui.tasks.wrangle import _NO_TEXT

    src = _scored_posts(tmp_path)
    out = tmp_path / "o.csv"

    p = ScriptedPrompter([
        "csv", str(src),
        ["text", _NO_TEXT], # a contradiction: the real column wins
        "combine", ["username"],
        False, False,       # no averages, nothing brought along
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    rows = {r["text_id"]: r for r in _read(out)}
    assert rows["alice"]["text"] == "meow one meow two"


# ---------------------------------------------------------------------------
# Merging, stacking and reshaping
# ---------------------------------------------------------------------------

from csvhelpers import write_rows as _write_rows  # noqa: E402


def _trials(tmp_path):
    return _write_rows(tmp_path / "trials.csv", ["pid", "trial", "score", "age"], [
        ["p1", "1", "10", "30"], ["p1", "2", "12", "30"], ["p1", "3", "11", "30"],
        ["p2", "1", "7", ""], ["p2", "2", "8", ""], ["p9", "1", "1", "22"]])


def _participants(tmp_path):
    return _write_rows(tmp_path / "participants.csv", ["pid", "age", "group"], [
        ["p1", "30", "A"], ["p2", "25", "B"], ["p3", "50", "C"]])


def _asked_questions(p, kind=None):
    return [q for k, q in p.asked if kind is None or k == kind]


def test_merge_flow_participants_onto_trials_reads_back_the_match(tmp_path):
    """
    The headline flow. The read-back has to carry real numbers -- that is
    what lets a person see a bad key before the file is written -- and the
    "which rows" question has to be built from the same numbers.
    """
    trials, people = _trials(tmp_path), _participants(tmp_path)
    out = tmp_path / "merged.csv"
    p = ScriptedPrompter([
        "merge", str(trials), str(people),
        ["pid"],                    # the matching column (pre-ticked for us)
        True,                       # bring over every other column
        "left",                     # which rows
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = _read(out)
    assert len(rows) == 6 and rows[0]["group"] == "A" and rows[-1]["group"] == ""
    assert ("5 of 6 rows in trials.csv match a row in participants.csv; 1 do not "
            "(e.g. p9).") in [o.strip() for o in p.output]
    assert any("Matching trials.csv against participants.csv" in o for o in p.output)
    key_rows = {c.value: c for c in p.offered_choices(
        "Which column(s) match rows between the two files?")}
    assert key_rows["pid"].checked, "the shared, right-unique column is suggested"
    assert "also in participants.csv" in key_rows["pid"].help
    assert "not in participants.csv" in key_rows["trial"].help
    how = {c.value: c for c in p.offered_choices("Which rows should the merged")}
    assert how["left"].label == "All 6 rows of trials.csv"
    assert how["inner"].label == "Only the 5 rows that matched"
    assert how["outer"].label == "Everything -- 7 rows"
    assert any("Wrangled 6 row(s)" in o for o in p.output)


def test_merge_asks_for_the_right_key_only_when_the_names_differ(tmp_path):
    trials = _trials(tmp_path)
    people = _write_rows(tmp_path / "people.csv", ["participant", "group"],
                         [["p1", "A"], ["p2", "B"]])
    out = tmp_path / "merged.csv"
    p = ScriptedPrompter([
        "merge", str(trials), str(people),
        ["pid"],
        "participant",              # which column in people.csv holds pid
        # one other column in people.csv: brought over without asking
        "left",                     # p9 has no match, so the question is asked
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert "In people.csv, which column holds pid?" in _asked_questions(p, "select")
    assert not any(q.startswith("Bring over") for q in _asked_questions(p))
    header = list(_read(out)[0])
    assert header == ["pid", "trial", "score", "age", "group"]


def test_merge_skips_the_which_rows_question_when_everything_matches(tmp_path):
    left = _write_rows(tmp_path / "a.csv", ["k", "x"], [["1", "a"], ["2", "b"]])
    right = _write_rows(tmp_path / "b.csv", ["k", "y"], [["1", "A"], ["2", "B"]])
    out = tmp_path / "m.csv"
    p = ScriptedPrompter(["merge", str(left), str(right), ["k"], str(out)])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert not any(q.startswith("Which rows should") for q in _asked_questions(p))
    assert any("Every row in a.csv with a k matches" in o for o in p.output)


def test_merge_offers_loose_matching_when_it_would_add_rows(tmp_path):
    left = _write_rows(tmp_path / "a.csv", ["k", "x"],
                       [["007", "a"], ["P02", "b"], ["3", "c"]])
    right = _write_rows(tmp_path / "b.csv", ["k", "y"],
                        [["7", "A"], ["p02", "B"], ["3", "C"]])
    out = tmp_path / "m.csv"
    p = ScriptedPrompter([
        "merge", str(left), str(right), ["k"],
        True,                       # match those too
        str(out),                   # everything matched now: no "which rows"
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert "Match those too? That's 2 more rows." in _asked_questions(p, "confirm")
    assert [r["y"] for r in _read(out)] == ["A", "B", "C"]
    assert any("007 in a.csv is 7 in b.csv" in r for r in p.reasons)


def test_merge_many_to_many_goes_back_to_the_key_question(tmp_path):
    left = _write_rows(tmp_path / "a.csv", ["k", "s", "x"],
                       [["1", "a", "x1"], ["1", "b", "x2"], ["2", "a", "x3"]])
    right = _write_rows(tmp_path / "b.csv", ["k", "s", "y"],
                        [["1", "a", "y1"], ["1", "b", "y2"], ["1", "c", "y3"]])
    out = tmp_path / "m.csv"
    p = ScriptedPrompter([
        "merge", str(left), str(right),
        ["k"], True,                # k alone: 1 is on two rows of each file
        "back",                     # the block screen
        ["k", "s"],                 # k and s together; only y is left to bring
        "left",                     # (2, a) and (1, c) are unmatched
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    assert any("one more column is needed" in o for o in p.output)
    keys = [q for q in _asked_questions(p, "checkbox")
            if q.startswith("Which column(s) match")]
    assert len(keys) == 2, "asked, blocked, asked again"
    second = p.offered[[q for q, _c in p.offered].index(
        "Which column(s) match rows between the two files?") + 1:]
    again = next(cs for q, cs in second if q.startswith("Which column(s) match"))
    assert {c.value for c in again if c.checked} == {"k"}, "the earlier answer is kept"
    assert [r["y"] for r in _read(out)] == ["y1", "y2", ""]


def test_merge_esc_from_which_rows_returns_to_the_keys(tmp_path):
    from wizard_helpers import EscapingPrompter

    trials, people = _trials(tmp_path), _participants(tmp_path)
    out = tmp_path / "merged.csv"
    p = EscapingPrompter([
        "merge", str(trials), str(people), ["pid"], True,
        "__esc__",                  # on "which rows": back to the keys
        ["pid"], True, "inner", str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert sum(q.startswith("Which column(s) match") for q in _asked_questions(p)) == 2
    assert len(_read(out)) == 5


def test_merge_refuses_to_write_over_an_input(tmp_path):
    trials, people = _trials(tmp_path), _participants(tmp_path)
    out = tmp_path / "merged.csv"
    p = ScriptedPrompter([
        "merge", str(trials), str(people), ["pid"], True, "left",
        str(trials),                # no
        str(out),                   # fine
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert any("one of the files we're reading" in o for o in p.output)
    assert _read(trials)[0]["pid"] == "p1"


def test_merge_announces_every_pass_over_the_files(tmp_path):
    trials, people = _trials(tmp_path), _participants(tmp_path)
    p = ScriptedPrompter(["merge", str(trials), str(people), ["pid"], True,
                          "left", str(tmp_path / "m.csv")])
    TASK.run(TaskContext(prompter=p, cwd=tmp_path))
    labels = [o for o in p.output if o.startswith(("Reading ", "Matching "))]
    assert labels == ["Reading trials.csv", "Reading participants.csv",
                      "Matching trials.csv against participants.csv"]
    assert p.reasons, "every non-obvious question has a line above it"


def test_a_cp1252_spreadsheet_goes_through_the_plain_csv_flow(tmp_path):
    """The old file pick decoded utf-8 only, and a Windows export died there."""
    src = tmp_path / "posts.csv"
    src.write_bytes("username,text\nJosé,meow\n".encode("cp1252"))
    out = tmp_path / "out.csv"
    p = ScriptedPrompter([
        "csv", str(src), ["text"], "as_is", "username", str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert _read(out)[0]["text_id"] == "José"
    assert any("Windows (cp1252)" in o for o in p.output)


def test_stack_flow_lines_up_columns_and_names_the_source(tmp_path):
    a = _write_rows(tmp_path / "wave1.csv", ["pid", "x"], [["1", "a"]])
    b = _write_rows(tmp_path / "wave2.csv", ["x", "pid", "y"], [["b", "2", "B"]])
    out = tmp_path / "all.csv"
    p = ScriptedPrompter([
        "stack",
        "\x00space:" + str(a), str(b),    # tick one, enter on the other
        True,                             # source column
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    assert list(_read(out)[0]) == ["source", "pid", "x", "y"]
    assert [r["source"] for r in _read(out)] == ["wave1", "wave2"]
    title, rows, headers = p.tables[0]
    assert title == "What's in each file" and headers[0] == "file"
    assert [r[0] for r in rows] == ["wave1.csv", "wave2.csv"]
    assert rows[1][3] == "y", "the column only wave2 has is named"


def test_stack_flow_merges_case_variants_on_a_yes_and_flags_a_stranger(tmp_path):
    a = _write_rows(tmp_path / "a.csv", ["Age", "pid"], [["1", "p"]])
    b = _write_rows(tmp_path / "b.csv", ["age", "pid"], [["2", "q"]])
    c = _write_rows(tmp_path / "c.csv", ["q", "r"], [["1", "2"]])
    out = tmp_path / "all.csv"
    p = ScriptedPrompter([
        "stack", "\x00space:" + str(a), str(c),
        False,                            # a stranger: don't stack it anyway
        "\x00space:" + str(a), str(b),
        True,                             # Age and age are one column
        True,                             # source
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert any("checkerboard" in r for r in p.reasons)
    assert "Treat each of those as one column?" in _asked_questions(p, "confirm")
    assert [r["Age"] for r in _read(out)] == ["1", "2"]


def _wide(tmp_path):
    return _write_rows(tmp_path / "wide.csv",
                       ["pid", "group", "score_t1", "score_t2", "score_t3"],
                       [["p1", "A", "1", "2", "3"], ["p2", "B", "4", "", "6"]])


def test_reshape_longer_flow_strips_the_prefix_and_names_the_columns(tmp_path):
    wide = _wide(tmp_path)
    out = tmp_path / "long.csv"
    p = ScriptedPrompter([
        "reshape", str(wide), "longer",
        ["score_t1", "score_t2", "score_t3"],
        ["pid", "group"],                 # keep the others
        True,                             # drop "score_t"
        "group",                          # taken: re-asked
        "time", "score",
        True,                             # keep rows for empty cells
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = _read(out)
    assert list(rows[0]) == ["pid", "group", "time", "score"]
    assert [r["time"] for r in rows[:3]] == ["1", "2", "3"]
    assert len(rows) == 6 and rows[4]["score"] == ""
    assert any("already a column called group" in o for o in p.output)
    assert "Drop the shared start 'score_t' from the names (score_t1 becomes 1)?" \
        in _asked_questions(p, "confirm")
    assert any("2 rows × 3 columns become 6 rows" in o for o in p.output)


def test_reshape_wider_flow_carries_a_steady_column_and_averages_a_clash(tmp_path):
    long = _write_rows(tmp_path / "long.csv", ["pid", "group", "cond", "score"], [
        ["p1", "A", "B", "2"], ["p1", "A", "A", "1"], ["p1", "A", "A", "5"],
        ["p2", "B", "A", "3"], ["p2", "B", "C", "4"]])
    out = tmp_path / "wide.csv"
    p = ScriptedPrompter([
        "reshape", str(long), "wider",
        "cond", "score",
        ["pid"],                          # pre-ticked: it looks like an id
        True, ["group"],                  # bring group along (steady per pid)
        "mean",                           # p1 has two A rows
        str(out),
    ])
    assert TASK.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = _read(out)
    assert list(rows[0]) == ["pid", "group", "B", "A", "C"]
    assert rows[0] == {"pid": "p1", "group": "A", "B": "2", "A": "3", "C": ""}
    keys = {c.value: c for c in p.offered_choices("Which column(s) say which rows")}
    assert keys["pid"].checked
    assert "Counting rows per pid" in p.output
    assert any("p1 has 2 rows for cond = A" in r for r in p.reasons)


def test_reshape_wider_grays_out_a_column_with_too_many_values(tmp_path):
    long = _write_rows(tmp_path / "long.csv", ["pid", "t", "v"],
                       [["p", f"t{i}", "1"] for i in range(1001)])
    p = ScriptedPrompter(["reshape", str(long), "wider", "pid"])
    with pytest.raises(AssertionError):
        TASK.run(TaskContext(prompter=p, cwd=tmp_path))   # the script ends here
    rows = {c.value: c for c in p.offered_choices("Which column's values become")}
    assert "too many to become columns" in rows["t"].disabled
    assert "1,001" in rows["t"].disabled
