"""
Tests for the live-region renderer.

The interesting property here is not "does it draw the rail" but "does it draw
the rail *only while the question is open*". prompt_toolkit leaves an
application's final frame in the terminal scrollback -- that is how
questionary's "? Question  answer" record survives -- so a rail that is still
present in that last frame would be left behind once per question. Eight
questions, eight rails. `test_the_rail_is_gone_from_the_final_frame` is the one
that pins that down.

These drive the real questionary widgets through a prompt_toolkit pipe with a
VT100 output captured into a string, so what is asserted is the actual bytes a
terminal would receive.
"""

from __future__ import annotations

import io
import re

import pytest

pytest.importorskip("prompt_toolkit")
pytest.importorskip("questionary")

from prompt_toolkit.application import create_app_session  # noqa: E402
from prompt_toolkit.data_structures import Size  # noqa: E402
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output.vt100 import Vt100_Output  # noqa: E402

from taters.ui.live import LivePrompter  # noqa: E402
from taters.ui.prompts import Choice  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


def render(prompter, keys, ask):
    """Run one question against a captured VT100 terminal; return what it drew."""
    buf = io.StringIO()
    out = Vt100_Output(buf, lambda: Size(rows=24, columns=90), term="xterm-256color")
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=out):
            pipe.send_text(keys)
            answer = ask(prompter)
    return answer, ANSI.sub("", buf.getvalue())


@pytest.fixture()
def staged():
    p = LivePrompter()
    p.stage("source", "Source", status="done", detail="./essays  (3 files)")
    p.stage("features", "Features", status="active")
    p.stage("options", "Options", status="todo")
    return p


# ---------------------------------------------------------------------------
# The rail itself
# ---------------------------------------------------------------------------

def test_a_stage_is_updated_in_place_not_appended():
    """
    The wizard marks a stage done without knowing whether it announced it
    first. Appending instead of updating would grow the rail on every call.
    """
    p = LivePrompter()
    p.stage("source", "Source", status="active")
    p.stage("source", "Source", status="done", detail="./essays")

    assert len(p._stages) == 1
    assert p._stages[0].status == "done"
    assert p._stages[0].detail == "./essays"


def test_the_markers_differ_by_status_not_only_by_color():
    """
    Color is the first thing lost -- piped output, a monochrome terminal, a
    color-blind reader. The glyphs have to carry the meaning on their own.
    """
    p = LivePrompter()
    p.stage("a", "Done", status="done")
    p.stage("b", "Active", status="active")
    p.stage("c", "Todo", status="todo")

    text = "".join(chunk for _, chunk in p._rail_lines())
    assert "✓ Done" in text
    assert "▸ Active" in text
    assert "○ Todo" in text
    # three distinct glyphs, so nothing hinges on the color.
    assert len({"✓", "▸", "○"} & set(text)) == 3
    assert "✓ Todo" not in text and "▸ Todo" not in text


def test_an_unknown_status_falls_back_rather_than_raising():
    p = LivePrompter()
    p.stage("a", "Mystery", status="banana")
    assert "Mystery" in "".join(c for _, c in p._rail_lines())


def test_no_stages_means_no_rail():
    """A prompter nobody told about stages must render exactly as before."""
    assert LivePrompter()._rail_lines() == []


# ---------------------------------------------------------------------------
# Rendering, through the real widgets
# ---------------------------------------------------------------------------

def test_the_rail_is_drawn_above_the_question(staged):
    _, drawn = render(staged, "\r", lambda p: p.select(
        "Which features do you want to extract?",
        [Choice("a", "Dictionaries"), Choice("b", "Readability")],
    ))
    for probe in ("✓ Source", "▸ Features", "○ Options"):
        assert probe in drawn, f"{probe!r} missing from the rendered frame"
    # one line, so it carries position and nothing else. the detail it used to
    # print alongside each stage now lives only on the screen that owns it.
    assert "./essays" not in drawn
    rail = [ln for ln in drawn.split("\n") if "✓ Source" in ln]
    assert len(rail) == 1 and "○ Options" in rail[0], rail


def test_the_key_hints_match_the_question_type(staged):
    _, from_select = render(staged, "\r", lambda p: p.select(
        "Pick", [Choice("a", "A"), Choice("b", "B")]))
    _, from_checkbox = render(staged, "\x1b[B \x1b[A\r", lambda p: p.checkbox(
        "Tick", [Choice("a", "A"), Choice("b", "B")]))

    # keys get brackets so "a all" can't read as a typo to anyone, native
    # English speaker or not.
    assert "[enter] picks the highlighted row" in from_select
    # the checkbox hint teaches our one tick dialect: boxes tick, the Done row
    # proceeds (see LivePrompter.checkbox).
    assert "ticks a box" in from_checkbox
    assert "Done" in from_checkbox


def test_the_rail_is_gone_from_the_final_frame(staged):
    """
    The regression that matters. prompt_toolkit leaves the last frame in
    scrollback, so anything still rendered there is permanent. Only
    questionary's own one-line record should survive.
    """
    _, drawn = render(staged, "\r", lambda p: p.select(
        "Which features do you want to extract?",
        [Choice("a", "Dictionaries"), Choice("b", "Readability")],
    ))
    final = "? Which features do you want to extract?" + drawn.rsplit("? Which features do you want to extract?", 1)[-1]

    assert "Dictionaries" in final
    assert "✓ Source" not in final
    assert "[enter]" not in final


# ---------------------------------------------------------------------------
# The widgets still work
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keys,ask,expected", [
    ("\r", lambda p: p.select("Q", [Choice("a", "A"), Choice("b", "B")]), "a"),
    ("\x1b[B \x1b[A\r",
     lambda p: p.checkbox("Q", [Choice("a", "A"), Choice("b", "B")]), ["a"]),
    ("\r", lambda p: p.confirm("Q", default=True), True),
    ("hi\r", lambda p: p.text("Q"), "hi"),
])
def test_wrapping_the_layout_does_not_break_the_widget(staged, keys, ask, expected):
    """
    Replacing an Application's layout can lose the focused window, which
    manifests as a prompt that accepts no keys at all. Every question type is
    driven here for exactly that reason.
    """
    answer, _ = render(staged, keys, ask)
    assert answer == expected


def test_a_layout_it_cannot_wrap_still_asks_the_question(staged, monkeypatch):
    """
    The rail is decoration. If a future questionary reshapes its layout in a
    way this does not expect, the wizard must still be usable.
    """
    def explode(self, application, hint):
        raise RuntimeError("layout changed shape")

    monkeypatch.setattr(LivePrompter, "_wrap", explode)
    answer, _ = render(staged, "\r", lambda p: p.select(
        "Q", [Choice("a", "A"), Choice("b", "B")]))
    assert answer == "a"


# ---------------------------------------------------------------------------
# One screen per question
# ---------------------------------------------------------------------------

def test_each_question_wipes_the_screen_first(staged, monkeypatch):
    """
    The wizard is a sequence of screens, not a growing log. Without the wipe,
    ten questions leave ten answered prompts stacked above the current one.
    """
    painted = []
    monkeypatch.setattr(type(staged._console), "clear",
                        lambda self, *a, **k: painted.append(True))
    render(staged, "\r", lambda p: p.select("Q", [Choice("a", "A"), Choice("b", "B")]))
    assert painted, "the screen was never cleared"


def test_a_note_survives_until_its_question_is_answered(staged):
    """
    Each question wipes the screen, so "Found 412 .txt files" printed just
    before one would vanish before it could be read. It has to be redrawn on
    the screen it is about -- and dropped once that question is done.
    """
    staged.note("  Found 412 .txt file(s).", style="green")
    assert staged._screen_notes

    render(staged, "\r", lambda p: p.select("Q", [Choice("a", "A")]))
    assert staged._screen_notes == []


def test_the_header_is_redrawn_on_every_screen(staged, capsys):
    """
    The header is drawn by rich, straight to stdout, while the question is
    drawn by prompt_toolkit into its own output. Two streams, one screen --
    which is why this looks at stdout rather than at the rendered frame.
    """
    staged.set_header("BANNER-MARKER")
    render(staged, "\r", lambda p: p.select("Q", [Choice("a", "A")]))
    assert "BANNER-MARKER" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Esc
# ---------------------------------------------------------------------------

def test_escape_asks_to_go_back_rather_than_cancelling(staged):
    """
    GoBack and Cancelled mean opposite things: "I answered wrong" versus
    "stop". Collapsing them would make a typo cost the whole session.
    """
    from taters.ui.prompts import GoBack

    with pytest.raises(GoBack):
        render(staged, "\x1b", lambda p: p.select("Q", [Choice("a", "A"), Choice("b", "B")]))


@pytest.mark.parametrize("ask", [
    lambda p: p.select("Q", [Choice("a", "A"), Choice("b", "B")]),
    lambda p: p.checkbox("Q", [Choice("a", "A")]),
    lambda p: p.confirm("Q"),
    lambda p: p.text("Q"),
])
def test_escape_works_on_every_question_type(staged, ask):
    from taters.ui.prompts import GoBack

    with pytest.raises(GoBack):
        render(staged, "\x1b", ask)


def test_the_key_hint_mentions_escape(staged):
    _, drawn = render(staged, "\r", lambda p: p.select("Q", [Choice("a", "A")]))
    assert "[esc] back" in drawn


# ---------------------------------------------------------------------------
# Yes/no without typing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keys,expected", [
    ("y", True), ("Y", True), ("1", True),
    ("n", False), ("N", False), ("0", False),
])
def test_a_single_keypress_answers_a_confirm(staged, keys, expected):
    """
    questionary's own confirm renders `(Y/n)` and waits on a text buffer. A
    single `y` does answer it, but nothing on screen says so -- it reads as
    "type a word and press enter", the one interaction that worked differently
    from every other question in the wizard.
    """
    answer, _ = render(staged, keys, lambda p: p.confirm("Sure?", default=True))
    assert answer is expected


def test_enter_takes_the_default(staged):
    assert render(staged, "\r", lambda p: p.confirm("Sure?", default=True))[0] is True
    assert render(staged, "\r", lambda p: p.confirm("Sure?", default=False))[0] is False


def test_a_confirm_can_also_be_navigated_like_any_other_list(staged):
    """Arrow keys and enter behave as they do everywhere else."""
    answer, _ = render(staged, "\x1b[B\r", lambda p: p.confirm("Sure?", default=True))
    assert answer is False


def test_the_default_is_listed_first(staged):
    """So that enter and 'the top item' agree with each other."""
    _, drawn = render(staged, "\r", lambda p: p.confirm("Sure?", default=False))
    body = drawn.split("? Sure?")[1]
    assert body.index("No") < body.index("Yes")


def test_the_hint_names_the_shortcuts(staged):
    _, drawn = render(staged, "\r", lambda p: p.confirm("Sure?", default=True))
    assert "[y]/[n] answer" in drawn


# ---------------------------------------------------------------------------
# Tables have to survive a screen wipe too
# ---------------------------------------------------------------------------

def test_a_table_is_redrawn_on_the_next_screen(staged, capsys):
    """
    The bug this catches was reported from a real run: the whole setup report
    was drawn, then wiped by the very next question, leaving only a one-line
    conclusion under a header that suggested nothing had been printed.

    `note` was screen-aware and `table` was not.
    """
    staged.table("Hardware", [["Graphics card", "yes", "an RTX"]], ["", "", ""])
    capsys.readouterr()                      # toss the first, pre-wipe print

    staged._paint()
    redrawn = capsys.readouterr().out
    assert "Hardware" in redrawn
    assert "Graphics card" in redrawn


def test_a_table_stops_being_redrawn_once_its_question_is_answered(staged):
    staged.table("Hardware", [["a", "yes", "b"]], ["", "", ""])
    assert staged._screen_notes

    render(staged, "\r", lambda p: p.select("Q", [Choice("a", "A")]))
    assert staged._screen_notes == []


def test_notes_and_tables_keep_their_order(staged, capsys):
    """A caption printed before a table must not end up after it."""
    staged.note("  Checking what this machine can do…")
    staged.table("Hardware", [["a", "yes", "b"]], ["", "", ""])
    capsys.readouterr()

    staged._paint()
    out = capsys.readouterr().out
    assert out.index("Checking what") < out.index("Hardware")


# ---------------------------------------------------------------------------
# Answering a menu with a number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [("1", "a"), ("2", "b"), ("3", "c")])
def test_a_digit_picks_and_confirms_in_one_keypress(staged, key, expected):
    """
    questionary's own shortcuts only move the cursor and still want enter,
    which is two keys for what should be one.
    """
    answer, _ = render(staged, key, lambda p: p.select(
        "Main menu",
        [Choice("a", "Alpha"), Choice("b", "Beta"), Choice("c", "Gamma")],
    ))
    assert answer == expected


def test_the_numbers_are_visible_not_just_bound(staged):
    """A shortcut nobody can see is a shortcut nobody uses."""
    _, drawn = render(staged, "\r", lambda p: p.select(
        "Main menu", [Choice("a", "Alpha"), Choice("b", "Beta")]))
    assert "1. Alpha" in drawn
    assert "2. Beta" in drawn


def test_arrow_keys_still_work_alongside_the_numbers(staged):
    answer, _ = render(staged, "\x1b[B\r", lambda p: p.select(
        "Main menu", [Choice("a", "Alpha"), Choice("b", "Beta")]))
    assert answer == "b"


def test_the_default_still_applies_once_labels_are_renumbered(staged):
    """
    Numbering rewrites the label, and questionary matches a default against the
    *value* -- so this would break silently if the two were confused.
    """
    answer, _ = render(staged, "\r", lambda p: p.select(
        "Main menu", [Choice("a", "Alpha"), Choice("b", "Beta")], default="b"))
    assert answer == "b"


def test_a_list_too_long_to_number_is_not_numbered_at_all(staged):
    """
    From a real report: numbering that stopped at nine "creates a weird
    asymmetry". A digit can only address nine things, so either every row
    carries a number or none does -- half a list of numbers reads as a list
    that lost them part way down, and "10." advertises a key that does nothing.
    """
    choices = [Choice(f"v{i}", f"Item {i}") for i in range(1, 13)]
    _, drawn = render(staged, "\r", lambda p: p.select("Pick", choices))

    assert "Item 1" in drawn
    assert "1. Item 1" not in drawn
    assert "10. Item 10" not in drawn
    # ...and the hint has to stop offering a key that isn't bound any more.
    assert "number" not in drawn


def test_a_list_that_fits_is_still_numbered(staged):
    """The shortcut is worth keeping everywhere a digit can actually reach."""
    choices = [Choice(f"v{i}", f"Item {i}") for i in range(1, 8)]
    _, drawn = render(staged, "\r", lambda p: p.select("Pick", choices))

    assert "1. Item 1" in drawn
    assert "7. Item 7" in drawn
    assert "[1-9]" in drawn


def test_a_digit_still_answers_a_list_that_fits(staged):
    """Numbered means answerable by that number, in one keypress."""
    choices = [Choice(f"v{i}", f"Item {i}") for i in range(1, 8)]
    answer, _ = render(staged, "3", lambda p: p.select("Pick", choices))

    assert answer == "v3"


def test_a_confirm_is_not_numbered(staged):
    """
    Yes/No already answers to y/n/1/0. Showing "2. No" beside a working "0"
    would be two different keys for one answer.
    """
    _, drawn = render(staged, "\r", lambda p: p.confirm("Sure?", default=True))
    assert "1. Yes" not in drawn
    assert "Yes" in drawn


def test_the_hint_mentions_numbers(staged):
    _, drawn = render(staged, "\r", lambda p: p.select("Pick", [Choice("a", "A")]))
    assert "[1-9] jump" in drawn
    assert "[enter] picks the highlighted row" in drawn


# ---------------------------------------------------------------------------
# Every screen is its own screen
# ---------------------------------------------------------------------------

def test_a_pause_does_not_leave_its_notes_for_the_next_screen():
    """
    From a real report: after "Check my setup" the main menu appeared near the
    bottom of the terminal. `pause` painted the report and waited, but never
    dropped the notes it had been holding -- so the *next* screen redrew the
    whole report again and put the menu underneath it.

    `_ask` has always cleared them once a question is answered. A pause is an
    answer too.
    """
    from taters.ui.live import LivePrompter

    prompter = LivePrompter.__new__(LivePrompter)
    prompter._screen_notes = [("note", ("a whole setup report", "", True))]
    prompter._header = ""
    prompter._paint = lambda: None
    prompter._wait_for_key = lambda message: None

    prompter.pause()

    assert prompter._screen_notes == [], "the next screen would redraw the report"


def test_a_pause_still_paints_before_waiting():
    """The other half: waiting at a screen the last `clear()` wiped is worse."""
    from taters.ui.live import LivePrompter

    order = []
    prompter = LivePrompter.__new__(LivePrompter)
    prompter._screen_notes = []
    prompter._paint = lambda: order.append("paint")
    prompter._wait_for_key = lambda message: order.append("wait")

    prompter.pause()
    assert order == ["paint", "wait"]


# ---------------------------------------------------------------------------
# Help text you can actually read
# ---------------------------------------------------------------------------

def test_help_text_is_not_glued_onto_the_label():
    """
    It used to be `f"{label}  —  {help}"`, which had to be elided to keep the
    row on one line -- so the end of every explanation was replaced with an
    ellipsis. The text existed to be read and was the part guaranteed not to be.
    """
    from taters.ui.prompts import Choice, QuestionaryPrompter

    prompter = QuestionaryPrompter()
    long_help = ("Score each speaker's text against archetype dictionaries "
                 "using sentence embeddings. Downloads a model on first run.")
    q = prompter._to_q(Choice("archetypes", "Archetype similarity", long_help))

    assert "—" not in str(q.title), "the help is still being appended to the label"
    assert "…" not in q.description, "the help was truncated"

    # we compare word by word, not as one string: the description is wrapped
    # now, so it carries newlines and indentation the source text doesn't. what
    # has to survive is every word.
    assert q.description.split() == long_help.split()

    # and the wrapping has to happen *here*, on the way to questionary --
    # `wrap_description` being correct is no use if nothing calls it.
    assert "\n" in q.description, "the description reached questionary unwrapped"


def test_a_choice_without_help_has_no_description():
    from taters.ui.prompts import Choice, QuestionaryPrompter

    q = QuestionaryPrompter()._to_q(Choice("quit", "Quit"))
    assert q.description is None


def test_the_label_is_still_kept_to_one_line():
    """
    questionary draws a row on a single line whatever is in it, so a long label
    would be hard-wrapped by the terminal mid-word, with the remainder dangling
    under no pointer. Labels are short by design; this is the backstop.
    """
    from taters.ui.prompts import Choice, QuestionaryPrompter

    q = QuestionaryPrompter()._to_q(Choice("x", "y" * 400))
    assert len(str(q.title)) < 400


def test_a_disabled_rows_reason_is_fitted_beside_its_label(monkeypatch):
    """
    questionary appends " (reason)" to a disabled row itself, after the
    label -- outside anything fitted here. A 75-character reason ran the row
    off the right edge: "...import one under Settings firs" (a real report).
    The label keeps its width and the reason gets what is left.
    """
    from taters.ui import prompts
    from taters.ui.prompts import Choice, QuestionaryPrompter

    monkeypatch.setattr(prompts, "terminal_width", lambda default=80: 80)
    long_reason = ("needs a model you saved from an earlier run; import one "
                   "under Settings first")
    q = QuestionaryPrompter()._to_q(Choice(
        "score", "Score with a model I already have", disabled=long_reason))
    assert 6 + len(str(q.title)) + 3 + len(q.disabled) <= 80
    assert q.disabled.endswith("…")
    assert str(q.title) == "Score with a model I already have", \
        "the label paid for the reason"

    short = QuestionaryPrompter()._to_q(Choice(
        "score", "Score with a model I already have",
        disabled="none saved yet; import one in Settings"))
    assert short.disabled == "none saved yet; import one in Settings", \
        "a reason that fits was changed"
    assert QuestionaryPrompter()._to_q(Choice("x", "y")).disabled is None


def test_a_version_number_is_not_recoloured_halfway_through():
    """
    From a real report: the banner showed `v0.` in one color and `2.1` in
    another, as though the version had been cut in half.

    rich's automatic highlighter colors anything resembling a number, a path
    or a URL. In prose that is noise; in `[dim]v0.2.1[/dim]` it is a visible
    bug, because the highlighter wins inside the dim span. Everything the app
    prints is styled deliberately or not at all.
    """
    import io
    import re

    from rich.console import Console

    from taters.ui.prompts import QuestionaryPrompter

    prompter = QuestionaryPrompter()
    assert prompter._console._highlight is False, "the highlighter is still on"

    # and now what that setting actually does, on a console that emits escapes
    # -- the prompter's own console resolves `color_system` against a non-tty
    # stdout under pytest and renders everything plain, so it can't show this
    # by itself.
    def render(highlight: bool) -> str:
        buf = io.StringIO()
        Console(file=buf, force_terminal=True, width=40,
                highlight=highlight).print("[dim]v0.2.1[/dim]")
        return buf.getvalue()

    assert not re.search(r"v0\.2\.1", render(True)), (
        "the premise no longer holds: rich left the version alone"
    )
    assert re.search(r"v0\.2\.1", render(prompter._console._highlight)), (
        "the version is still being broken into two colors"
    )


def test_a_long_description_is_wrapped_rather_than_cut():
    """
    From a real menu, cut mid-word at the terminal edge:

        Description: Transcribe and work out who spoke when. Needs the
        diarization extra, which is a large install (NeMo) plu

    questionary renders a description as one run of text and does not wrap it,
    so the sentence explaining an option -- the sentence someone is reading
    precisely because they cannot decide -- was the thing being truncated.
    """
    from taters.ui.prompts import _DESCRIPTION_PREFIX, wrap_description

    text = ("Transcribe and work out who spoke when. Needs the diarization "
            "extra, which is a large install (NeMo) plus three packages from "
            "GitHub.")
    wrapped = wrap_description(text, width=80)
    lines = wrapped.split("\n")

    assert len(lines) > 1, "a description far wider than the terminal did not wrap"
    assert wrapped.split() == text.split(), "wrapping lost or altered words"

    # the first line sits after the prefix questionary writes; the rest line up
    # under it, so the block reads as one paragraph rather than loose text
    # under the list.
    assert len(_DESCRIPTION_PREFIX) + len(lines[0]) <= 80
    for line in lines[1:]:
        assert line.startswith(" " * len(_DESCRIPTION_PREFIX)), (
            f"a continuation line is not indented: {line!r}"
        )


def test_a_narrow_terminal_is_left_alone():
    """Below about twenty usable columns there is no wrapping worth doing."""
    from taters.ui.prompts import wrap_description

    text = "a fairly long description that cannot usefully be folded"
    assert wrap_description(text, width=20) == text


def test_the_description_gets_its_own_style_class():
    """
    questionary tags it `class:text`, the same class as every unselected option
    title -- so coloring the description would recolour the whole list. It
    needs a class of its own before it can look like a note rather than another
    option.
    """
    from questionary.prompts.common import Choice as QChoice
    from questionary.prompts.common import InquirerControl

    from taters.ui.prompts import QuestionaryPrompter

    QuestionaryPrompter()          # building one applies the interception
    control = InquirerControl(
        [QChoice("One", "1", description="why you might pick this"),
         QChoice("Two", "2")],
        show_description=True,
    )
    classes = {token[0] for token in control._get_choice_tokens()}

    assert "class:description" in classes
    assert "class:text" in classes, "the option titles lost their own class"


def test_the_description_style_is_defined():
    from taters.ui.prompts import QuestionaryPrompter

    rules = dict(QuestionaryPrompter()._style.style_rules)
    assert "description" in rules
    assert rules["description"] != rules["text"], (
        "the description is styled the same as the options it explains"
    )


def test_backing_out_of_a_question_retires_its_notes_too():
    """
    From a real report: Esc while setting options "doesn't clear the screen".

    The notes a question puts up -- what a setting does, what it is currently
    -- were dropped only on the way to returning an answer. Backing out took
    the other exit, so they stayed, and a few Esc presses in a row stacked
    several explanations above the next question and pushed it down the screen.
    """
    from taters.ui.live import LivePrompter
    from taters.ui.prompts import GoBack

    prompter = LivePrompter.__new__(LivePrompter)
    prompter._screen_notes = [("note", ("what this setting does", "dim", True))]
    prompter._paint = lambda: None
    prompter._wrap = lambda application, hint: None
    prompter._bind_escape = lambda application: None

    class _Escaped:
        application = None

        @staticmethod
        def ask():
            return LivePrompter._BACK

    with pytest.raises(GoBack):
        prompter._ask(_Escaped())

    assert prompter._screen_notes == [], "an escaped question left its notes up"


def test_cancelling_a_question_retires_its_notes_too():
    """Ctrl-C takes the same exit and leaves the same debris behind."""
    from taters.ui.live import LivePrompter
    from taters.ui.prompts import Cancelled

    prompter = LivePrompter.__new__(LivePrompter)
    prompter._screen_notes = [("note", ("what this setting does", "dim", True))]
    prompter._paint = lambda: None
    prompter._wrap = lambda application, hint: None
    prompter._bind_escape = lambda application: None

    class _Cancelled:
        application = None

        @staticmethod
        def ask():
            return None

    with pytest.raises(Cancelled):
        prompter._ask(_Cancelled())

    assert prompter._screen_notes == []


# ---------------------------------------------------------------------------
# Lists longer than the terminal
#
# prompt_toolkit already scrolled to keep the pointer visible, so a long list
# was always navigable -- but nothing said so. rows simply weren't there, with
# no hint that arrowing further would reveal them, which looks like a list
# that's missing options rather than one that continues.
# ---------------------------------------------------------------------------

def _rendered(choices, pointed_at=0, description=False, room=6):
    """
    The visible text of a choice list, as questionary would draw it.

    `room` is pinned rather than derived from a terminal size: how many rows fit
    is its own question, tested separately, and letting it vary with whatever
    terminal the suite happens to run in would make these assertions weather
    reports.
    """
    import taters.ui.prompts as pr
    from questionary.prompts.common import Choice as QChoice
    from questionary.prompts.common import InquirerControl

    from taters.ui.prompts import QuestionaryPrompter

    QuestionaryPrompter()                      # building one applies the interceptions
    before = pr.visible_rows
    pr.visible_rows = lambda lines=None: room
    try:
        control = InquirerControl(
            [QChoice(f"Opt {i}", str(i),
                     description="why this one" if description else None)
             for i in range(choices)],
            show_description=description,
        )
        control.pointed_at = pointed_at
        return "".join("\n" if tuple(t[:2]) == ("", "\n") else t[1]
                       for t in control._get_choice_tokens())
    finally:
        pr.visible_rows = before


def test_a_list_that_fits_is_left_exactly_alone():
    text = _rendered(3)
    assert "▲" not in text and "▼" not in text
    assert text.count("Opt ") == 3


def test_a_long_list_says_how_much_is_below():
    text = _rendered(14, pointed_at=0)
    assert "▼ 8 more below" in text
    assert "▲" not in text, "nothing is above the first option"


def test_a_long_list_says_how_much_is_above():
    text = _rendered(14, pointed_at=13)
    assert "▲ 8 more above" in text
    assert "▼" not in text, "nothing is below the last option"


def test_the_window_follows_the_pointer():
    text = _rendered(14, pointed_at=7)
    assert "▲ 4 more above" in text and "▼ 4 more below" in text
    assert "Opt 7" in text, "the option being pointed at scrolled out of view"
    assert "Opt 0" not in text


def test_the_counts_add_up_to_the_whole_list():
    """
    A marker that lies about how much is off-screen is worse than no marker:
    it invites someone to stop arrowing before they have seen everything.
    """
    import re

    text = _rendered(20, pointed_at=10)
    above = int(re.search(r"▲ (\d+) more above", text).group(1))
    below = int(re.search(r"▼ (\d+) more below", text).group(1))
    shown = text.count("Opt ")
    assert above + shown + below == 20


def test_a_windowed_list_is_the_same_height_wherever_the_pointer_is():
    """
    The ▲ marker used to appear only once something scrolled off the top and
    the ▼ one vanished at the end, so the list grew a line as the pointer
    passed the middle and shrank one at the bottom. Every change of height
    scrolled the terminal, and the explanation above the question walked up
    the screen a line or two per keypress (the feature checklist did this on
    a laptop). Both marker rows are drawn once a list scrolls, blank when
    there is nothing on that side.
    """
    heights = {at: _rendered(14, pointed_at=at).count("\n") for at in (0, 3, 7, 12, 13)}
    assert len(set(heights.values())) == 1, heights
    # and the blank marker really is blank: no glyph claims there is more
    assert "▲" not in _rendered(14, pointed_at=0)
    assert "▼" not in _rendered(14, pointed_at=13)


def test_a_description_still_follows_a_windowed_list():
    text = _rendered(12, pointed_at=6, description=True)
    assert "▲" in text and "▼" in text
    assert text.rstrip().endswith("Description: why this one"), (
        "the description was lost or reordered by the windowing"
    )


def test_a_tiny_terminal_still_shows_something():
    from taters.ui.prompts import visible_rows

    assert visible_rows(lines=10) >= 5
    assert visible_rows(lines=1) >= 5


# ---------------------------------------------------------------------------
# Note layout
#
# callers indent by writing spaces into the string. that only ever indented the
# first line -- rich wrapped the rest back to column 0 -- so paragraphs had a
# ragged left edge and blocks of notes ran into one another instead of
# stacking. these pin the column, the measure, and the spacing between blocks.
# ---------------------------------------------------------------------------


def painted(text, *, style="", wrap=True, width=100):
    """What one note actually puts on a terminal that wide, sans color."""
    from rich.console import Console
    from taters.ui.prompts import QuestionaryPrompter

    p = QuestionaryPrompter()
    buf = io.StringIO()
    p._console = Console(file=buf, force_terminal=True, width=width,
                         highlight=False)
    p.note(text, style=style, wrap=wrap)
    return ANSI.sub("", buf.getvalue()).split("\n")


def test_every_line_of_a_wrapped_note_keeps_the_margin():
    """
    The bug the user reported as blocks of text running together: only the
    first line was indented, so the second snapped back to the screen edge.
    """
    lines = [ln for ln in painted("    " + "word " * 40) if ln.strip()]

    assert len(lines) > 1, "needs to actually wrap for this to prove anything"
    assert all(ln.startswith("    ") for ln in lines), lines
    assert not any(ln.startswith("     ") for ln in lines), "margin drifted"


def test_a_note_leaves_no_trailing_whitespace():
    """Invisible on screen, and there in whatever the user copies out."""
    lines = [ln for ln in painted("  " + "word " * 40) if ln.strip()]

    assert lines and all(ln == ln.rstrip() for ln in lines), lines


def test_prose_wraps_to_the_measure_not_to_the_whole_terminal():
    """
    Text run to the width of a maximized terminal is hard to read and left the
    screen with two right edges: one at the banner, one far off to its right.
    """
    from taters.ui.prompts import MEASURE

    lines = [ln for ln in painted("  " + "word " * 60, width=200) if ln.strip()]

    assert len(lines) > 1
    assert max(len(ln) for ln in lines) <= MEASURE


def test_a_narrow_terminal_is_not_padded_out_to_the_measure():
    """The measure is a ceiling, not a fixed width."""
    lines = [ln for ln in painted("  " + "word " * 20, width=30) if ln.strip()]

    assert max(len(ln) for ln in lines) <= 30


def test_an_unwrappable_line_is_left_alone():
    """
    `wrap=False` is for things that must survive copy-paste -- a command with a
    long path in it. Folding one in half is worse than letting it run off.
    """
    path = r"  C:\Users\someone\a very long path\that goes on\forever.csv"
    lines = [ln for ln in painted(path, wrap=False, width=40) if ln.strip()]

    assert lines[0].rstrip() == path


def test_runs_of_blank_notes_collapse_to_one():
    """
    Spacing is written by hand at ~40 call sites, so doubled gaps are
    unavoidable there. Collapsing here is what keeps the rhythm even.
    """
    from rich.console import Console
    from taters.ui.prompts import QuestionaryPrompter

    p = QuestionaryPrompter()
    buf = io.StringIO()
    p._console = Console(file=buf, force_terminal=True, width=80)
    p.note("  Something.")
    p.note("")
    p.note("")
    p.note("")
    p.note("  Something else.")

    assert "\n\n\n" not in ANSI.sub("", buf.getvalue())


def test_a_blank_note_at_the_top_of_a_screen_is_dropped():
    """It has nothing to separate; it only pushes the first line down."""
    from rich.console import Console
    from taters.ui.prompts import QuestionaryPrompter

    p = QuestionaryPrompter()
    buf = io.StringIO()
    p._console = Console(file=buf, force_terminal=True, width=80)
    # we put something on screen first, so that we're testing the reset rather
    # than a fresh prompter's starting state.
    p.note("  Whatever the last screen said.")
    p.clear()
    buf.truncate(0), buf.seek(0)
    p.note("")
    p.note("  The first thing.")

    # no lstrip here: the leading gap is the whole thing we're testing.
    assert ANSI.sub("", buf.getvalue()).startswith("  The first")


def test_a_repaint_starts_a_fresh_screen_for_spacing():
    """
    The live renderer wipes and replays its notes. The flag has to reset with
    the screen, or the first note after a repaint loses its leading gap.
    """
    p = LivePrompter()
    p._blank_last = False
    p.repaint()

    assert p._blank_last is True


# ---------------------------------------------------------------------------
# Sizing the window against the real chrome
#
# somebody reported this one: "when a module's options are too long to list,
# the banner runs up off screen". we were sizing the window against a fixed
# guess of 16 rows of chrome, and a screen carrying a banner, a rail and three
# paragraphs of explanation has well over twice that -- so the list grew taller
# than the terminal and pushed everything above it out of view.
# ---------------------------------------------------------------------------


def _screenful(prompter, rows, choices, keys="\r"):
    """
    Render one question into a terminal exactly `rows` tall.

    Both halves are captured. The banner and the notes are printed by *rich*,
    straight to the console, while the rail and the list are drawn by
    prompt_toolkit into its own output -- and it is the two of them together
    that have to fit, which is exactly what the first version of this missed.
    """
    from rich.console import Console

    paint = io.StringIO()
    prompter._console = Console(file=paint, force_terminal=True, width=90,
                                highlight=False)
    buf = io.StringIO()
    out = Vt100_Output(buf, lambda: Size(rows=rows, columns=90),
                       term="xterm-256color")
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=out):
            pipe.send_text(keys)
            prompter.select("Change a setting:", choices)
    return ANSI.sub("", paint.getvalue()), ANSI.sub("", buf.getvalue())


def _rows(text):
    return len(text.replace("\r\n", "\n").rstrip("\n").split("\n")) if text.strip() else 0


def test_a_long_list_does_not_push_the_banner_off_the_screen(staged):
    """The reported symptom, measured: what is drawn has to fit the terminal."""
    staged.set_header("BANNERLINE\n" * 6)
    for i in range(4):
        staged.note(f"    Paragraph {i} of explanation for this setting.")
    # with help text, because every actual module option has some -- and it's
    # the description line under the pointer, plus the scroll markers, that our
    # first count of the rows below the list forgot to reserve.
    choices = [Choice(f"v{i}", f"Setting {i}", f"what setting {i} does")
               for i in range(1, 41)]

    painted, drawn = _screenful(staged, 30, choices)

    # the banner gets drawn first and scrolls off the top the moment the two
    # halves together overflow. that's the whole of the reported bug.
    assert _rows(painted) + _rows(drawn) <= 30, (
        f"{_rows(painted)} painted + {_rows(drawn)} drawn overflows a 30-row screen"
    )
    assert "BANNERLINE" in painted, "the banner has to survive"
    assert "▼" in drawn, "a windowed list has to say what is below it"


def test_more_explanation_above_leaves_room_for_fewer_rows_below(staged):
    """
    The point of measuring rather than guessing: the same list on the same
    terminal gets a smaller window when there is more above it.
    """
    choices = [Choice(f"v{i}", f"Setting {i}") for i in range(1, 41)]

    staged.set_header("BANNER\n")
    _, bare = _screenful(staged, 30, choices)

    staged.set_header("BANNER\n")
    for i in range(8):
        staged.note(f"    Paragraph {i} of explanation.")
    _, padded = _screenful(staged, 30, choices)

    shown = lambda d: len([l for l in d.split("\n") if "Setting " in l])
    assert shown(padded) < shown(bare), "the window ignored what was above it"


def test_a_repainted_note_keeps_its_margin_too():
    """
    Every question repaints, so the *replay* is how a note is usually seen.
    Painting it straight at the console skipped the margin handling entirely,
    which quietly undid the wrapping fix everywhere it actually mattered.
    """
    from rich.console import Console
    from taters.ui.live import LivePrompter

    p = LivePrompter()
    p.note("    " + "word " * 40)          # this gets remembered for the next screen

    buf = io.StringIO()
    p._console = Console(file=buf, force_terminal=True, width=100,
                         highlight=False)
    p.repaint()

    lines = [ln for ln in ANSI.sub("", buf.getvalue()).split("\n") if ln.strip()]
    assert len(lines) > 1, "needs to wrap for this to prove anything"
    assert all(ln.startswith("    ") for ln in lines), lines


# ---------------------------------------------------------------------------
# The rail belongs to the task that raised it
#
# another one somebody reported: after finishing at the wizard's closing
# question and going back to the main menu, "the pipeline still shows up" --
# the rail was still there, describing a pipeline nobody was building any more.
# ---------------------------------------------------------------------------


def test_the_rail_can_be_forgotten():
    p = LivePrompter()
    p.stage("source", "Source", status="done")
    p.stage("features", "Features", status="active")

    p.reset_stages()

    assert p._rail_lines() == []
    assert p._rail_height() == 0


def test_the_rail_costs_two_rows_however_many_stages_there_are():
    """
    The point of one line: five stages used to cost seven rows of a screen
    that also has to hold a banner, an explanation and a list.
    """
    p = LivePrompter()
    for i in range(5):
        p.stage(f"s{i}", f"Stage {i}", status="todo")

    assert p._rail_height() == 2
    text = "".join(chunk for _, chunk in p._rail_lines())
    assert text.rstrip("\n").count("\n") == 0, "the rail wrapped onto a second line"


def test_a_stage_detail_no_longer_takes_room_on_the_rail():
    """
    A detail as long as five chosen features could not fit, so the rail
    truncated the one line that most needed the room. The details are still on
    the screens that own them.
    """
    p = LivePrompter()
    p.stage("features", "Features", status="done",
            detail="Acoustic measures, Whisper audio embeddings, Readability "
                   "scores, Lexical richness, Dictionaries")

    text = "".join(chunk for _, chunk in p._rail_lines())
    assert "✓ Features" in text
    assert "Readability" not in text


# ---------------------------------------------------------------------------
# Why a question is on screen
#
# as reported: "the message informing us why we're seeing this page is
# placed/colored in a way that makes it easy to miss". it was a plain note, so
# it went to the console *above the rail* -- separated from the question it
# explains by the rail and by every other note on screen -- and dim to boot.
# ---------------------------------------------------------------------------


def test_the_reason_sits_between_the_rail_and_the_question(staged):
    staged.reason("What you picked needs a transcript.")
    _, drawn = render(staged, "\r", lambda p: p.select(
        "How should Taters produce a transcript?",
        [Choice("t", "One speaker"), Choice("d", "Several speakers")]))

    lines = [ln for ln in drawn.split("\n")]
    rail = next(i for i, ln in enumerate(lines) if "✓ Source" in ln)
    reason = next(i for i, ln in enumerate(lines) if "needs a transcript" in ln)
    question = next(i for i, ln in enumerate(lines)
                    if "How should Taters produce" in ln)
    assert rail < reason < question, (rail, reason, question)


def test_the_reason_is_not_dressed_like_a_note(staged):
    """Color and a mark, because reading like commentary is how it got lost."""
    from taters.ui.prompts import _REASON_MARK

    staged.reason("What you picked needs a transcript.")
    _, drawn = render(staged, "\r", lambda p: p.select(
        "Pick", [Choice("a", "A"), Choice("b", "B")]))

    assert f"{_REASON_MARK} What you picked needs a transcript." in drawn


def test_the_reason_retires_with_the_question_it_explains(staged):
    """It explains one question. Left standing it would explain the next one."""
    staged.reason("What you picked needs a transcript.")
    render(staged, "\r", lambda p: p.select(
        "Pick", [Choice("a", "A"), Choice("b", "B")]))

    _, second = render(staged, "\r", lambda p: p.select(
        "Something else", [Choice("a", "A"), Choice("b", "B")]))
    assert "needs a transcript" not in second


def test_the_reason_takes_room_from_the_list_not_from_the_screen(staged):
    """It is drawn in the layout, so it has to be counted like everything else."""
    choices = [Choice(f"v{i}", f"Setting {i}", f"help {i}") for i in range(1, 41)]

    staged.set_header("BANNER\n")
    _, bare = _screenful(staged, 30, choices)

    staged.set_header("BANNER\n")
    staged.reason("A sentence explaining why this question is being asked.")
    _, with_reason = _screenful(staged, 30, choices)

    count = lambda d: len([ln for ln in d.split("\n") if "Setting " in ln])
    assert count(with_reason) < count(bare)


# ---------------------------------------------------------------------------
# One "you are here"
# ---------------------------------------------------------------------------


def test_going_back_a_stage_demotes_the_ones_after_it():
    """
    From a real report: the rail showed "▸ Features  ▸ Options" at once. Backing
    up re-activated the earlier stage and left the later ones active from the
    previous pass, so the rail claimed you were in two places.
    """
    p = LivePrompter()
    p.stage("source", "Source", status="done")
    p.stage("features", "Features", status="done", detail="3 chosen")
    p.stage("options", "Options", status="active")

    p.stage("features", "Features", status="active")     # we backed up a step

    text = "".join(chunk for _, chunk in p._rail_lines())
    assert text.count("▸") == 1, text
    assert "▸ Features" in text and "○ Options" in text


def test_a_later_stage_loses_its_detail_when_you_back_up_past_it():
    """A detail from a pass that is being redone describes work now undone."""
    p = LivePrompter()
    p.stage("options", "Options", status="done", detail="4 changed")
    # give it a detail it actually carries, otherwise clearing it proves nothing.
    p.stage("review", "Review", status="active", detail="named 'My pipeline'")

    p.stage("options", "Options", status="active")

    assert [st.detail for st in p._stages if st.key == "review"] == [""]
    assert [st.detail for st in p._stages if st.key == "options"] == ["4 changed"]


def test_a_digit_on_a_disabled_row_does_nothing(staged):
    """
    From the code review (issue 8): the digit bindings exit the prompt directly
    with the row's value, sidestepping questionary's disabled gate -- one
    keypress selected a grayed-out action, and the handler behind it crashed on
    the very empty state the graying-out was protecting.
    """
    choices = [Choice("run", "Run"),
               Choice("rename", "Rename", disabled="you have no pipelines yet"),
               Choice("back", "Back")]

    answer, _ = render(staged, "2\r", lambda p: p.select("Pick", choices))

    assert answer != "rename"
    assert answer == "run", "the digit should be inert; enter picks the pointer"


def test_digits_still_answer_enabled_rows_around_a_disabled_one(staged):
    choices = [Choice("run", "Run"),
               Choice("rename", "Rename", disabled="not yet"),
               Choice("back", "Back")]

    answer, _ = render(staged, "3", lambda p: p.select("Pick", choices))

    assert answer == "back", "numbering must not shift around the disabled row"


def test_a_callable_header_is_rendered_fresh_on_every_paint():
    """
    From the code review (issue 22): the banner's border-color drift was
    computed once, baked into a static header string, and never visibly moved.
    A callable header is re-rendered per paint, so it can.
    """
    from rich.console import Console
    from taters.ui.live import LivePrompter

    frames = iter(["FRAME-ONE", "FRAME-TWO"])
    p = LivePrompter()
    p.set_header(lambda: next(frames))
    buf = io.StringIO()
    p._console = Console(file=buf, force_terminal=True, width=80,
                         highlight=False)

    p.repaint()
    p.repaint()

    drawn = ANSI.sub("", buf.getvalue())
    assert "FRAME-ONE" in drawn and "FRAME-TWO" in drawn


def test_the_hub_hands_the_live_renderer_the_banner_function():
    """String headers still work; the hub must pass the callable to get the
    drift. Pinned so a refactor back to banner() cannot quietly re-freeze it."""
    import inspect

    from taters.ui import hub as hub_mod

    src = inspect.getsource(hub_mod.run_hub)
    assert "set_header(banner)" in src
    assert "set_header(banner())" not in src


def test_annotations_and_tones_reach_the_terminal_in_color(staged):
    """
    The folder counts and the "yes, do this" row carry their color through
    questionary's formatted titles. Asserted against the raw VT stream because
    the whole point is what the terminal shows, not what the dataclass holds.
    """
    buf = io.StringIO()
    out = Vt100_Output(buf, lambda: Size(rows=24, columns=90),
                       term="xterm-256color")
    choices = [Choice("use", "✓ Use this folder", tone="good"),
               Choice("f", "interviews/", annotation="4 files"),
               Choice("g", "notes/")]
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=out):
            # we answer a row that is NOT the toned one: the answer line gets
            # rendered in this same green, so answering "Use this folder"
            # would satisfy the assertion even with the tone broken.
            pipe.send_text("3")
            staged.select("Which folder?", choices)

    raw = buf.getvalue()
    green = "38;5;35"
    # we go chunk by chunk: after splitting on ESC[, each chunk starts with the
    # SGR that styles the text inside it -- so the color has to sit in the
    # *same* chunk as the text it claims to color, not just somewhere on the
    # screen (the answer line is this green too, which fooled a looser check).
    chunks = raw.split("\x1b[")
    assert any(green in c and "4 files" in c for c in chunks), (
        "the annotation lost its color")
    # the toned row is also the POINTED row here, and the pointed row now
    # turns the highlight color whole -- that's the feature, not a loss.
    # so the tone has to show on the row when it is NOT pointed:
    highlight = "38;5;37"
    assert any((green in c or highlight in c) and "Use this folder" in c
               for c in chunks), "the tone/highlight lost its color"


def test_a_reason_renders_even_without_a_stage_rail():
    """Round-2 cut list: `_wrap` bailed out entirely when there were no stages,
    so a reason() on a stage-less screen (the hub's menus) never rendered --
    an explanation the user was owed, silently dropped."""
    p = LivePrompter()          # no stages
    p.reason("This is why you are seeing this question.")

    _, drawn = render(p, "\r", lambda pr: pr.select(
        "Pick", [Choice("a", "A"), Choice("b", "B")]))

    assert "This is why you are seeing this question." in drawn


# ---------------------------------------------------------------------------
# Space ticks; the pointer stays put
# ---------------------------------------------------------------------------


def test_space_ticks_in_place_without_leaving_the_prompt(staged):
    """
    Direct feedback, round two: space used to exit the prompt so the caller
    could rebuild the menu -- the whole screen wiped on every tick, and space
    became indistinguishable from enter. Now space flips the tick inside the
    running prompt (the app repaints only itself) and enter is free to mean
    "choose". One select call ticks two rows and then chooses Back.
    """
    ticked: set = set()
    choices = [Choice(":back", "↩ Back", tone="nav"),
               Choice("a.dicx", "[ ] a"),
               Choice("b.dicx", "[ ] b")]

    answer, drawn = render(
        staged, "\x1b[B \x1b[B \x1b[A\x1b[A\r",   # ↓ space ↓ space ↑↑ enter
        lambda p: p.select("Lib:", choices,
                           toggle_values={"a.dicx", "b.dicx"}, ticked=ticked))

    assert answer == ":back", "the prompt exited on space instead of enter"
    assert ticked == {"a.dicx", "b.dicx"}
    # we pin the on-screen mark flip separately (test_flip_tick_mark_*): here
    # the harness pre-feeds every key, so intermediate repaints coalesce and
    # the captured frames aren't a reliable witness.
    assert drawn  # at least the prompt rendered


def test_flip_tick_mark_handles_both_title_shapes():
    from taters.ui.live import flip_tick_mark

    assert flip_tick_mark("2. [ ] affect") == "2. [x] affect"
    assert flip_tick_mark("2. [x] affect") == "2. [ ] affect"
    flipped = flip_tick_mark([("class:text", "[ ] a  "),
                              ("class:annotation", " 1.2 KB")])
    assert flipped == [("class:text", "[x] a  "),
                       ("class:annotation", " 1.2 KB")]


def test_space_is_inert_on_action_rows(staged):
    """Idle space on "Delete the ticked files" must never fire it or tick."""
    ticked: set = set()
    choices = [Choice(":delete", "✕ Delete the ticked files", tone="danger"),
               Choice("a.dicx", "[ ] a")]

    answer, _ = render(staged, " \x1b[B \r",  # space on action, ↓ space, enter
                       lambda p: p.select("Lib:", choices,
                                          toggle_values={"a.dicx"},
                                          ticked=ticked))

    assert ticked == {"a.dicx"}
    assert answer == "a.dicx", "enter chose the pointed row"


def test_the_pointer_returns_to_the_toggled_row(dicts_lib=None):
    """
    The other half of the feedback: "kicked back to the top each time is very
    painful." The tick screens pass the toggled value back as `default=`, so
    working down a list is ↓ space ↓ space -- not a climb from the top.
    """
    from taters.helpers import library as lib
    from taters.ui.library import manage_library

    kind = lib.KINDS["dictionaries"]
    for name in ("a.dicx", "b.dicx", "c.dicx"):
        f = lib.kind_dir(kind) / name
        f.write_text("x", encoding="utf-8")

    from taters.ui.prompts import ScriptedPrompter

    p = ScriptedPrompter(["b.dicx", ":back"])
    manage_library(p, kind)

    # the loop after the toggle has to point back at what we just toggled.
    assert p.select_defaults[f"{kind.label}:"] == "b.dicx"


# ---------------------------------------------------------------------------
# Navigation without teardown
# ---------------------------------------------------------------------------

def test_enter_navigates_in_place_instead_of_rebuilding(staged):
    """
    The browser used to exit the application and build a new one per folder;
    the teardown-then-relist gap read as the screen flashing. Two enters here,
    ONE select: the first swaps the rows in place, the second answers from the
    swapped listing -- which is only possible if the prompt never went down.
    """
    def navigate(value):
        if value == "folder":
            return {"choices": [Choice("inner", "Inner file")]}
        return None

    answer, _ = render(staged, "\r\r", lambda p: p.select(
        "Walk", [Choice("folder", "A folder/")],
        transient=True, navigate=navigate))

    assert answer == "inner"


def test_ticks_follow_the_walk_into_a_new_folder(staged):
    """The space binding reads a shared toggle set that navigation refreshes,
    so a folder entered mid-walk brings its files' tickability along."""
    ticked = set()

    def navigate(value):
        if value == "folder":
            return {"choices": [Choice("inner", "[ ] inner")],
                    "toggle_values": {"inner"}}
        return None

    answer, _ = render(staged, "\r \r", lambda p: p.select(
        "Walk", [Choice("folder", "A folder/")],
        transient=True, toggle_values=(), ticked=ticked, navigate=navigate))

    assert answer == "inner"
    assert ticked == {"inner"}, "space did not tick a row born of the swap"


def test_the_breadcrumb_is_drawn_beside_the_question(staged):
    _, drawn = render(staged, "\r", lambda p: p.select(
        "Walk", [Choice("a", "A")], breadcrumb=lambda: "…/potatoes/deep"))
    assert "…/potatoes/deep" in drawn


def test_a_text_prompt_marks_where_typing_lands(staged):
    """Reported from the delete screen: the question ended in a colon and then
    nothing. The '› ' sits exactly where the typed answer will land."""
    _, drawn = render(staged, "hi\r", lambda p: p.text("Type the name to confirm:"))
    assert "› " in drawn


def test_the_description_gets_a_breath_of_air_above_it(staged):
    """Butted straight against the last option, the pointed row's description
    read as one more row of the list -- reported as claustrophobic."""
    _, drawn = render(staged, "\r", lambda p: p.select(
        "Pick", [Choice("a", "Alpha", "The first letter."),
                 Choice("b", "Beta", "The second letter.")]))
    lines = drawn.splitlines()
    i = next(n for n, ln in enumerate(lines) if "Description:" in ln)
    assert lines[i - 1].strip() == "", (
        f"no blank line above the description: {lines[i-1]!r}")
    assert "Beta" in lines[i - 2] or "Alpha" in lines[i - 2]


# ---------------------------------------------------------------------------
# Checkbox: one screen; space ticks; enter only works on Done, only with ticks
# ---------------------------------------------------------------------------

def test_enter_is_inert_everywhere_but_a_ticked_done():
    """The user's rule, third iteration: enter on an entry row does NOTHING
    (no toggle, no proceed, no screen rebuild), and enter on Done waits until
    something is ticked. The proof: an inert enter on 'a' leaves it out of
    the answer, and an early enter on Done did not end the prompt."""
    p = LivePrompter()
    answer, _screen = render(
        p, "\r\x1b[B\r\x1b[B \x1b[A\x1b[A\r",
        # Done+enter (inert: nothing ticked) · down to a · enter (inert) ·
        # down to b · space (tick) · up, up to Done · enter (confirms)
        lambda pr: pr.checkbox("pick", [Choice("a", "a"), Choice("b", "b")]))
    assert answer == ["b"]


def test_ticks_confirm_wherever_the_pointer_was_before_done():
    p = LivePrompter()
    answer, _screen = render(
        p, "\x1b[B \x1b[A\r",             # tick a, back to Done, confirm
        lambda pr: pr.checkbox("pick", [Choice("a", "a"), Choice("b", "b"),
                                        Choice("c", "c")]))
    assert answer == ["a"]


def test_prechecked_rows_arrive_ticked_and_confirm_at_once():
    p = LivePrompter()
    answer, screen = render(
        p, "\r",                            # Done immediately: pre-tick counts
        lambda pr: pr.checkbox("pick", [Choice("a", "a", checked=True),
                                        Choice("b", "b")]))
    assert answer == ["a"]
    assert "[x] a" in screen and "[ ] b" in screen, (
        "the boxes are the visual language that says this is a tick screen"
    )
    assert "✓ Done" in screen


def test_the_confirm_default_says_so_in_words():
    """The default used to be signaled only by row order and the starting
    highlight -- a convention, not information. Now the row says (default)."""
    p = LivePrompter()
    answer, screen = render(p, "\r",
                            lambda pr: pr.confirm("Overwrite?", default=False))
    assert answer is False
    assert "(default)" in screen


# ---------------------------------------------------------------------------
# Pinned rows: the way out of a gated tick screen never scrolls away
# ---------------------------------------------------------------------------

def _rendered_sticky(choices, pointed_at, room=6):
    """Like _rendered, with the first row marked always-visible."""
    import taters.ui.prompts as pr
    from questionary.prompts.common import Choice as QChoice
    from questionary.prompts.common import InquirerControl

    from taters.ui.prompts import QuestionaryPrompter

    QuestionaryPrompter()
    before = pr.visible_rows
    pr.visible_rows = lambda lines=None: room
    try:
        control = InquirerControl(
            [QChoice("Done row", "done")] +
            [QChoice(f"Opt {i}", str(i)) for i in range(choices - 1)])
        control.pointed_at = pointed_at
        control._taters_sticky_rows = 1
        return "".join("\n" if tuple(t[:2]) == ("", "\n") else t[1]
                       for t in control._get_choice_tokens())
    finally:
        pr.visible_rows = before


def test_the_pinned_row_survives_scrolling_to_the_bottom():
    text = _rendered_sticky(20, pointed_at=19)
    assert "Done row" in text, "the way out scrolled off the top"
    assert "▲" in text and "▼" not in text
    # the marker sits between the pinned row and the window, not above it.
    assert text.index("Done row") < text.index("▲")


def test_the_pinned_counts_still_add_up():
    import re

    text = _rendered_sticky(20, pointed_at=10)
    above = int(re.search(r"▲ (\d+) more above", text).group(1))
    below = int(re.search(r"▼ (\d+) more below", text).group(1))
    shown = text.count("Opt ") + text.count("Done row")
    assert above + shown + below == 20


def test_an_unpinned_list_is_left_exactly_alone():
    """The pin is opt-in: an ordinary long select still windows from its top
    row, with nothing held back."""
    text = _rendered(14, pointed_at=13)
    assert "▲ 8 more above" in text and "Opt 0" not in text


def test_a_long_checkbox_still_ticks_and_confirms():
    """End to end: a checkbox tall enough to scroll, the pointer driven all
    the way down and back -- ticking and the Done gate must survive the
    scrolling. (The pinned rendering itself is asserted in the
    _rendered_sticky tests: prompt_toolkit coalesces piped keys into one
    repaint, so intermediate scrolled frames are never actually drawn here.)"""
    p = LivePrompter()
    downs = "\x1b[B" * 25
    keys = downs + " " + "\x1b[A" * 25 + "\r"    # tick the last, return, enter
    choices = [Choice(f"c{i}", f"Column {i}") for i in range(25)]
    answer, _ = render(p, keys, lambda pr: pr.checkbox("Which?", choices))
    assert answer == ["c24"]


def test_the_gate_pins_its_done_row_for_the_scroller():
    """The checkbox must actually ask for the pin: the sticky mark has to be
    on the list control before the question runs, or a long list scrolls the
    ✓ Done row -- the only way out of the screen -- off the top."""
    from questionary.prompts.common import InquirerControl

    from taters.ui.prompts import Cancelled

    seen = {}
    orig = LivePrompter._ask

    def spy(self, question_obj, *args, **kwargs):
        for window in question_obj.application.layout.find_all_windows():
            if isinstance(window.content, InquirerControl):
                seen["sticky"] = getattr(window.content,
                                         "_taters_sticky_rows", 0)
        raise Cancelled()

    LivePrompter._ask = spy
    try:
        p = LivePrompter()
        import pytest as _pytest
        with _pytest.raises(Cancelled):
            p.checkbox("Which?", [Choice("a", "A"), Choice("b", "B")])
    finally:
        LivePrompter._ask = orig
    assert seen.get("sticky") == 1


# ---------------------------------------------------------------------------
# The screen budget: notes must never crowd out the list
# ---------------------------------------------------------------------------

def _sized(rows=30, width=100):
    """A prompter drawing into a terminal of a known height."""
    import io

    from rich.console import Console

    p = LivePrompter()
    p._console = Console(file=io.StringIO(), width=width, height=rows)
    p.set_header("  T A T E R S")
    p.stage("source", "Source", status="active")
    p.stage("features", "Features", status="todo")
    return p


def _painted(p):
    import io

    p._console.file = io.StringIO()
    p._paint()
    return p._console.file.getvalue().rstrip("\n").split("\n")


def test_a_huge_note_is_trimmed_so_the_list_keeps_its_room():
    """The report this fixes: a spreadsheet of 150 columns printed all 150
    above a list of those same 150, leaving five rows to pick from."""
    from taters.ui.prompts import _MIN_VISIBLE_ROWS

    p = _sized(rows=30)
    p.note("  study.csv: 156 column(s) — " +
           ", ".join(f"Img{i}_Valence_mean" for i in range(60)))
    p._reserve_rows = _MIN_VISIBLE_ROWS

    lines = _painted(p)
    assert len(lines) < 12, "the note still swallowed the screen"
    assert "study.csv: 156 column(s)" in lines[1], "its head survived"
    assert any("more line(s)" in ln for ln in lines), \
        "a trimmed note has to say it was trimmed"


def test_a_screen_with_no_list_is_never_trimmed():
    """The setup report *is* its screen; trimming it to leave room for a list
    that never comes would be vandalism."""
    p = _sized(rows=24)
    p.note("  " + " ".join(f"line{i}" for i in range(200)))
    p._reserve_rows = 0            # this is a pause screen, not a question

    lines = _painted(p)
    assert len(lines) > 15, "a report screen keeps everything it printed"
    assert not any("more line(s)" in ln for ln in lines)


def test_the_oldest_notes_are_the_ones_that_go():
    """The newest note is the one about the question on screen."""
    from taters.ui.prompts import _MIN_VISIBLE_ROWS

    p = _sized(rows=26)
    for i in range(8):
        p.note(f"  note number {i}")
    p._reserve_rows = _MIN_VISIBLE_ROWS

    text = "\n".join(_painted(p))
    assert "note number 7" in text, "the most recent note is the relevant one"
    assert "note number 0" not in text
    assert "earlier note(s) hidden" in text


def test_a_short_list_does_not_reserve_a_long_one_s_room():
    """A three-option menu has no use for twelve rows, and claiming them
    would trim notes for nothing."""
    p = _sized(rows=26)
    p.note("  " + " ".join(f"word{i}" for i in range(120)))
    lines_before = len(_painted(p))

    render(p, "\r", lambda pr: pr.select(
        "Q", [Choice("a", "A"), Choice("b", "B")]))
    assert p._reserve_rows == 0, "the reservation is released after the ask"
    p.note("  " + " ".join(f"word{i}" for i in range(120)))
    assert len(_painted(p)) >= lines_before


def test_note_lines_agrees_with_what_note_prints():
    """The painter measures a note's height with the same function that
    wraps it; if the two ever disagreed the budget would be fiction."""
    from taters.ui.prompts import note_lines

    p = _sized(rows=40)
    text = ("  a margin-carrying note with quite a lot of words in it, "
            "enough to wrap more than once at this width, and a second "
            "sentence to be sure of it")
    predicted = note_lines(text, p.note_width())

    import io
    p._console.file = io.StringIO()
    p._rows_painted = 0
    p._blank_last = True
    p.note(text)
    assert p._rows_painted == len(predicted)


# ---------------------------------------------------------------------------
# A list's height does not change as the pointer moves
# ---------------------------------------------------------------------------

def test_the_description_block_is_held_at_the_tallest_descriptions_height():
    """
    From a real report: "when I scroll through the checkbox items the
    terminal window scrolls and the yellow text moves up a couple of lines".
    questionary draws the pointed row's description under the list and
    nothing when that row has none, so the screen's height changed with
    every arrow press. The block is now reserved at the tallest description
    and padded, and the list's room shrinks by the same amount.
    """
    from taters.ui import prompts as pr

    short = Choice("a", "A", "one line")
    tall = Choice("b", "B", "x " * 200)          # wraps to several lines
    none = Choice("c", "C")
    rows = pr.description_rows([short, tall, none], width=80)
    assert rows >= 3
    assert pr.description_rows([none]) == 0
    assert pr.description_rows([Choice("d", "D", "y " * 2000)], width=80) == 6, \
        "capped, so one essay cannot eat the list"

    # the renderer pads what questionary emitted up to the reservation...
    assert len(pr._padded_tail([], rows)) == rows
    assert len(pr._padded_tail([[("class:text", "  Description: one")]], rows)) == rows
    # ...and leaves a taller block whole rather than cutting it.
    tall_block = [[("class:text", "l")] for _ in range(rows + 2)]
    assert pr._padded_tail(tall_block, rows) == tall_block
    assert pr._padded_tail([], 0) == []

    before, chrome = pr._reserved_description_rows, pr._measured_chrome
    try:
        pr.set_chrome_rows(16)
        pr.set_description_rows(0)
        room_plain = pr.visible_rows(lines=80)      # well above the floor
        pr.set_description_rows(rows)
        assert pr.visible_rows(lines=80) == room_plain - rows, \
            "the list gives up exactly the rows the description keeps"
    finally:
        pr.set_description_rows(before)
        pr.set_chrome_rows(chrome)


# ---------------------------------------------------------------------------
# Enter on a ticked row glides back to Done
# ---------------------------------------------------------------------------

def test_a_glide_has_a_fixed_duration_and_lands_on_the_target():
    """Five hundred rows glide in the same two seconds as forty, never more
    frames than rows, and the last frame is the target itself."""
    from taters.ui.live import GLIDE_FPS, GLIDE_SECONDS, glide_positions

    long = glide_positions(480, 0)
    assert long[-1] == 0 and long[0] < 480
    assert len(long) == round(GLIDE_SECONDS * GLIDE_FPS)
    assert all(b <= a for a, b in zip(long, long[1:])), "monotone toward the top"
    short = glide_positions(3, 0)
    assert short == [2, 1, 0]
    assert glide_positions(5, 5) == []
    down = glide_positions(0, 7, seconds=0.1, fps=20)
    assert down[-1] == 7 and len(down) == 2


def test_enter_on_an_item_glides_to_done_and_enter_there_confirms():
    """
    From a real request: after ticking row 480 of 500, Done was 480 presses
    of the up arrow away. Enter anywhere else now glides the pointer back
    to Done -- visibly, in a quarter of a second, without redrawing the screen --
    and enter on Done confirms as it always did.
    """
    import threading
    import time

    from taters.ui.live import LivePrompter

    prompter = LivePrompter()
    choices = [Choice(f"v{i}", f"Item {i}") for i in range(1, 41)]
    buf = io.StringIO()
    out = Vt100_Output(buf, lambda: Size(rows=30, columns=90), term="xterm-256color")
    started = time.monotonic()
    with create_pipe_input() as pipe:
        with create_app_session(input=pipe, output=out):
            # down four rows, tick, enter on the item (a glide, not an
            # answer); then, once the glide is over, enter on Done.
            pipe.send_text("\x1b[B\x1b[B\x1b[B\x1b[B \r")
            threading.Timer(0.8, pipe.send_text, ["\r"]).start()
            # if the glide didn't happen, enter on the item does nothing and
            # the screen would sit there forever: Esc ends it as a failure.
            bail = threading.Timer(6.0, pipe.send_text, ["\x1b"])
            bail.start()
            try:
                answer = prompter.checkbox("Pick items:", choices)
            finally:
                bail.cancel()
    elapsed = time.monotonic() - started
    assert answer == ["v4"]
    assert elapsed >= 0.25, "the glide should have taken a quarter second"
    drawn = ANSI.sub("", buf.getvalue())
    assert "Done" in drawn


def test_the_review_table_is_shown_not_hidden_behind_a_blank_line():
    """
    The review screen prints a blank line, the table of steps, and a blank
    line before "What next?". The blank line counted as the newest note and
    was kept; the table then did not fit beside it and went into "2 earlier
    note(s) hidden" -- the one screen whose whole point was the table showed
    an empty line instead (a real report).
    """
    from taters.ui.prompts import _MIN_VISIBLE_ROWS

    p = _sized(rows=30)
    p.note("")
    p.table("My pipeline — 11 steps over 1 file",
            [[str(i), f"Step number {i}", "once", "", "you picked"]
             for i in range(1, 12)],
            ["#", "Step", "Runs", "Measures", "Why"])
    p.note("")
    p._reserve_rows = _MIN_VISIBLE_ROWS

    text = "\n".join(_painted(p))
    assert "My pipeline — 11 steps" in text, "the table's title is on screen"
    assert "Step number 1" in text, "and at least its first rows"
    assert "earlier note(s) hidden" not in text, \
        "a blank spacer that fell off the top is not a lost note"


def test_a_table_taller_than_the_screen_shows_its_head_and_says_so():
    """Like a long note: the head, and how much is missing -- never the
    whole table gone."""
    from taters.ui.prompts import _MIN_VISIBLE_ROWS

    p = _sized(rows=20)
    p.table("Big", [[str(i), f"row {i}"] for i in range(40)], ["#", "What"])
    p._reserve_rows = _MIN_VISIBLE_ROWS

    lines = _painted(p)
    text = "\n".join(lines)
    assert "row 0" in text and "row 39" not in text
    assert "more row(s)" in text
    assert len(lines) <= 20
