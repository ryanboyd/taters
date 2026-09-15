"""
"Wrangle data": the gather helpers behind a front-door conversation.

The flows run through the ScriptedPrompter against real files, so what is
asserted is the actual spreadsheet a user would get.
"""

from __future__ import annotations

import csv

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
