"""
Tests for the file browser and the presentation helpers around it.

Typing a path was the most error-prone thing the wizard asked for, and its
failure mode was the worst kind: a typo produces "no files found", which reads
as "there is nothing here" rather than "you are looking in the wrong place".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taters.ui.browse import (
    _CHOOSE,
    _TYPE,
    _UP,
    browse_for_file,
    browse_for_folder,
)
from taters.ui.prompts import Choice, ScriptedPrompter, fit


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "essays").mkdir()
    for i in range(3):
        (tmp_path / "essays" / f"e{i}.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "a.wav").write_bytes(b"RIFF")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path


def labels(prompter, question_startswith):
    return [c.label for c in prompter.offered_choices(question_startswith)]


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def test_choosing_the_current_folder(tree):
    p = ScriptedPrompter([_CHOOSE])
    assert browse_for_folder(p, start=tree) == tree.resolve()


def test_descending_then_choosing(tree):
    p = ScriptedPrompter([str(tree / "essays"), _CHOOSE])
    assert browse_for_folder(p, start=tree) == (tree / "essays").resolve()


def test_going_up_a_level(tree):
    p = ScriptedPrompter([_UP, _CHOOSE])
    assert browse_for_folder(p, start=tree / "essays") == tree.resolve()


def test_folders_show_how_many_usable_files_they_hold(tree):
    """
    The annotation is what turns "which of these eleven folders was it?" into
    a question someone can answer by looking.
    """
    p = ScriptedPrompter([_CHOOSE])
    browse_for_folder(p, start=tree, want_files=[".txt"])

    # the count rides on Choice.annotation rather than being baked into the
    # label: the renderer gives annotations their own color, and that's what
    # makes "3 files" easy to spot while scanning a column of folder names.
    rows = {c.label.strip(): c.annotation for c in p.offered[0][1]}
    assert rows.get("essays/") == "3 files"
    assert rows.get("audio/") == ""


def test_hidden_folders_are_offered_last_and_say_so(tree):
    """
    The Hugging Face cache lives under ~/.cache, and importing a model from
    it was impossible while the browser skipped every dot-folder: the folder
    simply was not there to walk into. So they are listed -- after the
    ordinary folders, marked "hidden" -- rather than dropped.
    """
    p = ScriptedPrompter([_CHOOSE])
    browse_for_folder(p, start=tree)
    rows = p.offered_choices("Which folder?")
    names = [c.label.strip() for c in rows]
    hidden_at = next(i for i, n in enumerate(names) if n.startswith(".hidden/"))
    essays_at = next(i for i, n in enumerate(names) if n.startswith("essays/"))
    assert hidden_at > essays_at, "hidden folders go after the ordinary ones"
    assert rows[hidden_at].annotation == "hidden"


def test_a_typed_path_is_still_accepted(tree):
    """Browsing to a path you already know is the slow way round."""
    p = ScriptedPrompter([_TYPE, str(tree / "essays")])
    assert browse_for_folder(p, start=tree) == (tree / "essays").resolve()


def test_a_typed_path_that_does_not_exist_keeps_the_browser_open(tree):
    """
    The old behavior re-asked a bare prompt, which left someone who did not
    know the path with no way forward but to guess again.
    """
    p = ScriptedPrompter([_TYPE, str(tree / "nope"), _CHOOSE])
    assert browse_for_folder(p, start=tree) == tree.resolve()
    assert "not something I can see" in p.text_output


def test_there_is_no_home_row(tree):
    """
    Dropped on request: launch location covers "work from home", and every
    extra action row pushes the actual folders further down the screen.
    """
    p = ScriptedPrompter([_CHOOSE])
    browse_for_folder(p, start=tree)
    labels_shown = [c.label for c in p.offered[0][1]]
    assert not any("Home folder" in label for label in labels_shown)


def test_up_leads_and_use_this_folder_is_the_green_second(tree):
    """
    Browsing is mostly walking, so "up" comes first; the affirmative act is
    next and carries the green tone, so it reads as "yes, do this" rather than
    as one more folder in the pile.
    """
    p = ScriptedPrompter([_CHOOSE])
    browse_for_folder(p, start=tree, want_files=[".txt"])

    offered = p.offered[0][1]
    assert offered[0].value == _UP
    assert offered[0].tone == "nav", "the way out carries the amber nav tone"
    assert offered[1].value == _CHOOSE
    assert offered[1].tone == "good"
    assert offered[-1].value == _TYPE, "typing a path stays last"


def test_a_start_that_is_not_a_folder_falls_back_to_the_working_directory(tmp_path):
    p = ScriptedPrompter([_CHOOSE])
    result = browse_for_folder(p, start=tmp_path / "does-not-exist")
    assert result == Path.cwd().resolve()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def test_only_matching_files_are_offered(tree):
    p = ScriptedPrompter([str(tree / "notes.csv")])
    assert browse_for_file(p, start=tree, suffixes=(".csv",)) == tree / "notes.csv"

    shown = [row.strip() for row in labels(p, "Which file?")]
    assert any(row.startswith("notes.csv") for row in shown)
    assert not any(row.startswith("a.wav") for row in shown)
    # folders have to stay, otherwise there'd be no way to reach a file elsewhere.
    assert any(row.startswith("essays/") for row in shown)


def test_choosing_a_folder_is_not_offered_when_picking_a_file(tree):
    p = ScriptedPrompter([str(tree / "notes.csv")])
    browse_for_file(p, start=tree, suffixes=(".csv",))
    assert not any(c.value == _CHOOSE for c in p.offered[0][1])


def test_an_empty_folder_says_so_rather_than_looking_broken(tree):
    """On the breadcrumb, not as a note: notes forced a repaint under the
    transient prompt, and the message belongs to the listing it describes."""
    p = ScriptedPrompter([_UP, str(tree / "notes.csv")])
    browse_for_file(p, start=tree / "audio", suffixes=(".csv",))
    assert any("nothing matching in this folder" in b for b in p.breadcrumbs)
    # and it goes away once we move somewhere with content.
    assert "nothing matching" not in p.breadcrumbs[-1]


def test_an_unreadable_folder_does_not_end_the_wizard(tree, monkeypatch):
    """A permissions error should narrow the choices, not raise."""
    def deny(self):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "iterdir", deny)
    p = ScriptedPrompter([_CHOOSE])
    assert browse_for_folder(p, start=tree) == tree.resolve()


# ---------------------------------------------------------------------------
# Fitting text to the terminal
# ---------------------------------------------------------------------------

def test_long_lines_are_elided_rather_than_wrapped():
    """
    questionary draws a choice on one line and does no wrapping, so anything
    too long is hard-wrapped by the terminal -- mid-word, with the remainder
    dangling under no pointer.
    """
    long = "Extract features from my data  —  " + ("word " * 40)
    out = fit(long, reserve=6, width=80)

    assert len(out) <= 74
    assert "\n" not in out
    assert out.endswith("…")


def test_short_lines_are_left_alone():
    assert fit("Quit", reserve=6, width=80) == "Quit"


def test_line_breaks_are_flattened_but_column_padding_is_kept():
    """
    Only line breaks are collapsed. Callers pad names into columns, and
    flattening runs of spaces would undo the alignment this exists to protect.
    """
    assert fit("a\nb", reserve=0, width=80) == "a b"
    assert fit("name      value", reserve=0, width=80) == "name      value"


def test_a_hopelessly_narrow_terminal_is_left_alone():
    """Truncating to nothing is worse than overflowing."""
    assert fit("some text", reserve=0, width=4) == "some text"


def test_the_rows_line_up_in_columns(tree):
    """
    A file listing is scanned, not read. Names in one column and annotations in
    another is the difference between a list and a wall of text.
    """
    p = ScriptedPrompter([_CHOOSE])
    browse_for_folder(p, start=tree, want_files=[".txt"])

    # labels with an annotation get padded to one shared column, so every
    # annotation starts in the same place no matter how long the name before it.
    annotated = [c.label for c in p.offered[0][1] if c.annotation]
    assert annotated, "nothing was annotated at all"
    assert len({len(label) for label in annotated}) == 1, (
        f"annotations do not align: {annotated!r}"
    )


def test_file_sizes_are_human_readable(tree):
    p = ScriptedPrompter([str(tree / "notes.csv")])
    browse_for_file(p, start=tree, suffixes=(".csv",))
    row = next(c for _q, cs in p.offered for c in cs
               if c.label.strip().startswith("notes.csv"))
    assert "B" in row.annotation


# ---------------------------------------------------------------------------
# Navigating without flicker
# ---------------------------------------------------------------------------

def test_navigation_asks_transiently(tree):
    """
    Walking into a folder is one continuous act. Wiping the terminal on every
    keypress makes it flicker as though something restarted each time, so each
    step redraws over the one before it instead.
    """
    seen = []

    class Recorder(ScriptedPrompter):
        def select(self, question, choices, *, default=None, transient=False,
                   **kw):
            seen.append(transient)
            return super().select(question, choices, default=default, **kw)

    p = Recorder([str(tree / "essays"), _UP, _CHOOSE])
    browse_for_folder(p, start=tree)

    assert seen and all(seen), "every browse step should be transient"


def test_the_screen_furniture_is_drawn_once_for_a_whole_browse(tree):
    p = ScriptedPrompter([str(tree / "essays"), _UP, _CHOOSE])
    browse_for_folder(p, start=tree)

    assert p.output.count("<repaint>") == 1


def test_a_message_gets_the_furniture_redrawn_under_it(tree):
    """
    A transient prompt paints over what is already there, so a note shown
    mid-browse needs the screen rebuilt beneath it or it lands on top of the
    previous listing.
    """
    p = ScriptedPrompter([_TYPE, str(tree / "nope"), _CHOOSE])
    browse_for_folder(p, start=tree)

    assert p.output.count("<repaint>") == 2
    assert "not something I can see" in p.text_output


def test_a_crowded_folder_still_shows_a_fair_share_of_files(tmp_path):
    """
    From the code review (issue 18): the 200-entry cap gave folders absolute
    priority, so a directory with 250 subfolders and 300 matching files showed
    200 folders and exactly one file -- which reads as "my files are missing".
    """
    from taters.ui.browse import _MAX_ENTRIES, _entries

    for i in range(250):
        (tmp_path / f"sub{i:03d}").mkdir()
    for i in range(300):
        (tmp_path / f"clip{i:03d}.wav").touch()

    folders, files, truncated = _entries(tmp_path, want_files=(".wav",))

    assert truncated
    assert len(folders) + len(files) <= _MAX_ENTRIES
    assert len(files) >= _MAX_ENTRIES // 2, f"only {len(files)} files shown"
    assert len(folders) >= _MAX_ENTRIES // 4, "folders squeezed out instead"


def test_a_folder_light_directory_gives_files_the_leftover_room(tmp_path):
    from taters.ui.browse import _MAX_ENTRIES, _entries

    for i in range(10):
        (tmp_path / f"sub{i}").mkdir()
    for i in range(400):
        (tmp_path / f"clip{i:03d}.wav").touch()

    folders, files, truncated = _entries(tmp_path, want_files=(".wav",))

    assert len(folders) == 10
    assert len(files) == _MAX_ENTRIES - 10


def test_the_count_hint_stops_counting_in_enormous_folders(tmp_path):
    """Round-2 cut list: the hint stat'd every entry of every listed folder on
    every render -- a fifty-thousand-entry share stalled the whole browser."""
    from taters.ui.browse import _count_hint

    big = tmp_path / "big"
    big.mkdir()
    for i in range(30):
        (big / f"f{i:02d}.wav").touch()

    assert _count_hint(big, (".wav",)) == "30 files"
    assert _count_hint(big, (".wav",), cap=10) == "10+ files"
    assert _count_hint(big, (".flac",), cap=10) == ""


def test_the_pointer_starts_on_use_this_folder(tree):
    """Requested directly: most visits end in "use this folder", so that is
    where the pointer starts -- one Enter, not a step down first."""
    p = ScriptedPrompter([_CHOOSE])
    browse_for_folder(p, start=tree, want_files=[".txt"])

    question = next(q for q, _c in p.offered)
    assert p.select_defaults[question] == _CHOOSE


# ---------------------------------------------------------------------------
# Navigation without teardown
# ---------------------------------------------------------------------------

def test_the_scripted_prompter_swaps_rows_like_the_live_renderer():
    """
    The contract behind the no-flash browser: `navigate` inspects the value
    enter landed on and either swaps the rows in place (walking) or lets the
    value out (answering). The scripted prompter must walk the same way the
    live one does, or tests exercise a flow production never runs.
    """
    def navigate(value):
        if value == "go":
            return {"choices": [Choice("deep", "Deep")]}
        return None

    p = ScriptedPrompter(["go", "deep"])
    out = p.select("Walk", [Choice("go", "Go/")], navigate=navigate)

    assert out == "deep", "the swap was skipped and 'go' leaked out as an answer"
    assert [q for q, _ in p.offered] == ["Walk", "Walk"], (
        "each swapped listing is recorded under the same, single select")


def test_the_breadcrumb_walks_with_the_navigation(tree):
    """The path moved out of the question (frozen at build time) onto a live
    breadcrumb, so it can follow the walk without rebuilding the prompt."""
    p = ScriptedPrompter([str(tree / "essays"), _UP, _CHOOSE])
    browse_for_folder(p, start=tree)

    assert len(p.breadcrumbs) == 3
    assert p.breadcrumbs[0].endswith(tree.name)
    assert "essays" in p.breadcrumbs[1]
    assert p.breadcrumbs[2].endswith(tree.name)


def test_the_tick_browser_walks_without_teardown(tree):
    """browse_and_tick passes the same navigate/breadcrumb pair the folder
    browser does -- ticks and walk share one prompt, no rebuild per folder."""
    from taters.ui.browse import browse_and_tick

    e0 = tree / "essays" / "e0.txt"
    e1 = tree / "essays" / "e1.txt"
    p = ScriptedPrompter([
        str(tree / "essays"),           # walk in: this is a swap, not a re-ask
        "\x00space:" + str(e0),
        str(e1),                        # enter: we get the ticked one plus this one
    ])
    picked = browse_and_tick(p, suffixes=(".txt",), start=tree)

    assert picked == [e0, e1]
    assert len(p.breadcrumbs) == 2, "the walk left the prompt (or had no breadcrumb)"
    assert "essays" in p.breadcrumbs[1]
