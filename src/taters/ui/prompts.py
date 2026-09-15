"""
The thin layer between the wizard's questions and whatever is asking them.

:class:`Prompter` is the whole contract: five ways to ask something, one way to
wait, and two ways to say something. :mod:`taters.ui.wizard` is written against it and never
imports ``questionary`` or ``rich`` itself.

That indirection earns its keep twice. It lets the test suite drive the entire
wizard with a scripted answer list and no terminal at all -- see
:class:`ScriptedPrompter` -- and it means a GUI would replace this one file
rather than the wizard's logic.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, List, Optional, Protocol, Sequence,
                    Tuple)

__all__ = ["Choice", "Stage", "Prompter", "QuestionaryPrompter", "ScriptedPrompter",
           "Cancelled", "GoBack", "QuitRequested", "PAUSE_MESSAGE",
           "fit", "wrap_description", "style_descriptions_separately",
           "note_lines", "scroll_long_lists", "visible_rows",
           "terminal_width"]


class QuitRequested(Exception):
    """
    The user chose to finish from inside a task.

    Distinct from :class:`Cancelled`, which means "not this, take me back". A
    finished run offers "Quit" as a deliberate ending, and reporting that as
    having been backed out of would be a lie about work that succeeded.

    Carries the run's verdict, because the exception skips the task's normal
    ``return False`` path: quitting from a *failed* run's finish screen used to
    exit 0, and anything scripted around the TUI read that as success.
    """

    def __init__(self, ok: bool = True):
        self.ok = ok
        super().__init__()


class GoBack(Exception):
    """
    The user pressed Esc: undo the last question rather than the whole task.

    Distinct from :class:`Cancelled` because the two mean opposite things.
    Cancelled is "stop, I did not want this"; GoBack is "keep going, I just
    answered something wrong". Collapsing them would make a typo cost the whole
    session, which is exactly the sharp edge Esc exists to file off.
    """


class Cancelled(Exception):
    """The user backed out -- Ctrl-C, or Esc on a questionary prompt."""


@dataclass
class Stage:
    """One entry in the wizard's progress rail."""

    key: str
    label: str
    status: str = "todo"     # "done" | "active" | "todo"
    detail: str = ""


@dataclass
class Choice:
    """One option in a select or checkbox list."""

    value: str
    label: str
    help: str = ""
    checked: bool = False
    disabled: str = ""      # non-empty = "show it, but grayed out, with this reason"
    #: A short trailing fact -- "4 files", "1.2 MB" -- rendered in its own
    #: color so it can be *scanned*. Baked into the label it was the same gray
    #: as everything else, and the one thing a folder list exists to answer
    #: ("which of these has my files?") was the hardest thing on it to spot.
    annotation: str = ""
    #: Color the whole label carries: "" (ordinary) or "good" (green). For
    #: the row that means "yes, do the thing" -- ✓ Use this folder -- which
    #: should read as the affirmative act it is, not as one more entry in the
    #: pile of folders around it.
    tone: str = ""


def with_annotation(title, annotation: str):
    """
    A rendered row title with its annotation replaced (or added).

    Rows are built by ``_to_q`` as ``[(label_class, label),
    ("class:annotation", " " + annotation)]``; a plain string is a row that
    never had one. The left/right binding rewrites the pointed row's
    annotation in place so the list repaints without being rebuilt.
    """
    if isinstance(title, str):
        return [("class:text", title), ("class:annotation", " " + annotation)]
    rows = [tuple(t) for t in title]
    if rows and rows[-1][0] == "class:annotation":
        rows[-1] = ("class:annotation", " " + annotation)
    else:
        rows.append(("class:annotation", " " + annotation))
    return rows


def flip_tick_mark(title):
    """
    "[ ]" <-> "[x]" in a row's rendered title, whichever shape it has.

    questionary titles are either a plain string or a list of (style, text)
    tuples (ours carry the annotation as a second tuple); the mark always
    lives in the first text segment. Used by the in-place space toggle, where
    the row must change on screen without the prompt being torn down.
    """
    def swap(text: str) -> str:
        if "[ ]" in text:
            return text.replace("[ ]", "[x]", 1)
        return text.replace("[x]", "[ ]", 1)

    if isinstance(title, str):
        return swap(title)
    if isinstance(title, list) and title:
        style, text = title[0]
        return [(style, swap(text))] + list(title[1:])
    return title


def ask_at_least_one(prompter: "Prompter", question: str,
                     choices: Sequence[Choice], *, thing: str = "one",
                     cycle: Optional[Callable[[str, int], Optional[str]]] = None
                     ) -> List[str]:
    """
    A checkbox that will not take "nothing" for an answer.

    Three copies of this loop had grown -- the text columns, the grouping
    columns, the feature checklist -- each with its own wording for the same
    complaint. An empty answer is never meaningful at any of them: it would
    either crash later or quietly analyze nothing.

    The "space to tick, enter when done" gloss the questions used to carry is
    gone: the key hint bar under every checkbox already says `[space] tick ·
    [enter] confirm`, so it was the same instruction twice, and it pushed the
    questions past the width prose wraps at.
    """
    while True:
        picked = list(prompter.checkbox(question, choices, cycle=cycle)
                      if cycle is not None else prompter.checkbox(question, choices))
        if picked:
            return picked
        prompter.note(f"  Nothing ticked — pick at least {thing}.", style="yellow")


#: What a report screen says when it has finished and is waiting to be left.
PAUSE_MESSAGE = "Press any key to go back"

# how wide we let a column of text get, in cells, margin included.
#
# prose set to the full width of a maximized terminal is genuinely hard to read
# (the eye loses its place on the way back to the start of the next line), and
# it left every screen as text running out to column 200 underneath a banner
# that stopped at 64, which looks like clutter rather than a layout. so notes
# wrap to the banner's width and the screen gets one right edge instead of two.
# `hub.banner` takes its own width from this for the same reason.
MEASURE = 64

# how we draw a "why you're seeing this" line. bright, and marked, because the
# whole problem with the old treatment was that it read like more commentary.
_REASON_STYLE = "bold #ffd479"
_REASON_MARK = "›"


class Prompter(Protocol):
    """What the wizard needs from a user interface."""

    #: Whether `select(..., ticked=...)` can flip a row's mark in place (space
    #: ticks, enter acts). A renderer that cannot re-asks the list instead,
    #: so enter on a tickable row means "tick it" there -- and a screen whose
    #: only way to approve is enter-on-a-row has to offer an approve row.
    ticks_in_place: bool

    def clear(self) -> None: ...
    def repaint(self) -> None: ...
    def note(self, text: str, *, style: str = "", wrap: bool = True) -> None: ...
    def table(self, title: str, rows: Sequence[Sequence[str]],
              headers: Sequence[str]) -> None: ...
    def text(self, question: str, *, default: str = "") -> str: ...
    def path(self, question: str, *, default: str = "") -> str: ...
    def confirm(self, question: str, *, default: bool = True) -> bool: ...
    # `navigate` and `breadcrumb` are here for the file browsers. on enter, we
    # show `navigate` the pointed value and it either hands back a replacement
    # screen ({"choices": [...], "toggle_values": ..., "default": ...}) -- we
    # swap the rows in place, no exit, no flash -- or None, meaning this is a
    # real answer. a renderer that can't swap in place ignores `navigate` and
    # just returns the value; the caller's loop re-asks with the new listing.
    # `breadcrumb` says where the walk currently is, live.
    # `cycle(value, direction)` is what the left/right arrows do on a row: it
    # returns the row's new annotation, or None when that row has nothing to
    # cycle. the wizard uses it to change how a column is treated ("numbers"
    # <-> "labels") without leaving the list.
    def select(self, question: str, choices: Sequence[Choice],
               *, default: Optional[str] = None, transient: bool = False,
               toggle_values: Sequence[str] = (),
               ticked: Optional[set] = None,
               navigate: Optional[Callable[[str], Optional[dict]]] = None,
               breadcrumb: Optional[Callable[[], str]] = None,
               cycle: Optional[Callable[[str, int], Optional[str]]] = None
               ) -> str: ...
    def checkbox(self, question: str, choices: Sequence[Choice], *,
                 cycle: Optional[Callable[[str, int], Optional[str]]] = None
                 ) -> List[str]: ...

    # an acknowledgment, not a question. a screen that only reports something
    # ("Check my setup") has nothing to ask, and asking anyway is worse than
    # noise: a "Done? yes/no" whose two answers both go back to the same menu
    # invites people to hunt for the difference between them.
    def pause(self, message: str = ...) -> None: ...

    # "something slow is starting" -- a line that has to hit the screen *now*,
    # before the blocking call, not whenever the next question paints.
    # otherwise importing a heavy dependency or parsing a big dictionary leaves
    # a finished-looking screen sitting there silent, which looks like a hang.
    def working(self, text: str) -> None: ...

    # progress reporting. renderers with nowhere to put it (the plain
    # questionary one, the scripted one) just ignore it, so the wizard can call
    # it without checking.
    def stage(self, key: str, label: str, *, status: str = "active",
              detail: str = "") -> None: ...

    def reset_stages(self) -> None: ...

    def reason(self, text: str) -> None: ...


# ---------------------------------------------------------------------------
# the real one
# ---------------------------------------------------------------------------

class QuestionaryPrompter:
    ticks_in_place = False
    """
    A :class:`Prompter` backed by ``questionary`` for input and ``rich`` for output.

    Both are imported in ``__init__`` rather than at module scope, so that
    importing :mod:`taters.ui.wizard` -- which the tests do -- never requires a
    terminal library to be present.
    """

    # a class attribute, not just an instance one, because the tests routinely
    # build subclasses with their own console that never call
    # ``super().__init__`` -- and a missing flag would take `note` down rather
    # than just lose a blank line.
    _blank_last: bool = True

    #: Lines printed since the screen was last wiped. Counted rather than
    #: estimated so a list can be sized against the real chrome above it.
    _rows_painted: int = 0

    def __init__(self) -> None:
        try:
            import questionary
            # imported for the check, not for use -- `make_console` does the
            # building. it stays inside the try so that a missing rich gets the
            # message below rather than a traceback from three frames deeper.
            import rich.console  # noqa: F401
        except ImportError as e:  # pragma: no cover - both are base dependencies
            raise ImportError(
                "The setup wizard needs `questionary` and `rich`, which ship "
                "with the base install. Reinstall with `pip install taters`."
            ) from e
        self._q = questionary
        style_descriptions_separately()
        scroll_long_lists()
        # `highlight=False` because rich's automatic highlighter colors
        # anything that looks like a number, a path or a URL. in prose that's
        # noise, and in the banner it was a visible bug: `v0.2.1` came out with
        # `v0.` dim and `2.1` in cyan, like the version got cut in half. if
        # something's styled here, it's because we styled it.
        # through `make_console` rather than straight to rich: over SSH the
        # client announces a bare `xterm`, rich reads no color suffix off it
        # and drops to 16 colors, and the banner comes out gray. see
        # `ui/console.py` for why 256 is the right floor and which two
        # terminals are exempt from it.
        from .console import make_console
        self._console = make_console(highlight=False)
        # questionary renders its own colors; keeping rich to plain output for
        # notes stops the two from fighting over the same line.
        # one accent color (teal) for whatever the user is acting on, one for
        # what they've already settled (green), and gray for context. any more
        # than that and the terminal starts to look like a fruit machine.
        self._style = questionary.Style([
            ("qmark", "fg:#00afaf bold"),
            ("question", "fg:#d7d7d7 bold"),
            ("answer", "fg:#00af5f bold"),
            ("pointer", "fg:#00afaf bold"),
            ("highlighted", "fg:#00afaf bold"),
            ("selected", "fg:#00af5f"),
            ("separator", "fg:#585858"),
            ("instruction", "fg:#808080 italic"),
            ("text", "fg:#d7d7d7"),
            # trailing facts on a row (file counts, sizes): green, the same
            # green as "done" and "answer", so that a folder with usable files
            # in it reads as a hit while you're scanning.
            ("annotation", "fg:#00af5f"),
            ("tone-good", "fg:#00af5f bold"),
            # wayfinding rows (↑ Up to …): warm, and distinct from both the
            # affirmative green and the cyan pointer, so the way back is easy
            # to find at a glance.
            ("tone-nav", "fg:#d7af5f"),
            # destructive acts: red, so "Delete the 6 ticked files" can't be
            # mistaken for one more forward-moving green row.
            ("tone-danger", "fg:#d75f5f"),
            # the help for whichever option is highlighted. dimmer than the
            # options and italic, so it reads as a note attached to the list
            # rather than another thing in it.
            ("description", "fg:#9a9a9a italic"),
            ("disabled", "fg:#585858 italic"),
        ])

    # -- output ------------------------------------------------------------
    def repaint(self) -> None:
        """
        Redraw the screen furniture. A no-op here: this renderer has none.
        """

    def working(self, text: str) -> None:
        """
        Announce slow work *before* it starts, on screen immediately.

        A note plus a repaint, which is the pair that guarantees the line is
        visible before the blocking call rather than after it: the live
        renderer's screen is wiped per question, and a note printed onto a
        just-finished screen without the repaint could be cleared before it
        was ever seen. Dim, because it is narration, not an answer.
        """
        self.note(f"  {text}", style="dim")
        self.repaint()

    def clear(self) -> None:
        """
        Start on a clean screen.

        The scrollback is untouched -- this scrolls the screen rather than
        erasing history, so whatever the user had before is still there to page
        back to.
        """
        self._console.clear()
        # nothing on screen yet, so a leading blank note has nothing to
        # separate; we drop it rather than push the first line down.
        self._blank_last = True
        self._rows_painted = 0

    def note(self, text: str, *, style: str = "", wrap: bool = True) -> None:
        """
        Print a note, keeping its left margin on every line.

        Callers indent by writing spaces into the string, which reads naturally
        at the call site but only ever indented the *first* line: rich wrapped
        the rest back to column 0. Paragraphs therefore had a ragged left edge
        that alternated between the margin and the screen edge, which is what
        made blocks of notes run into each other instead of stacking.

        So the leading spaces are read off as a **margin** and re-applied as a
        hanging indent, per line, and the text is wrapped inside what is left.
        Every line of a block now starts in the same column, which is what lets
        the eye see where one block stops and the next begins.
        """
        # `wrap=False` is for lines that have to survive copy-paste intact -- a
        # command with a long path in it, say. better to let those run off the
        # edge than fold them in half, so we print them exactly as given.
        if not wrap:
            self._console.print(text, style=style or None, soft_wrap=True)
            self._blank_last = False
            self._rows_painted += str(text).count("\n") + 1
            return

        for wrapped in note_lines(text, self.note_width()):
            if not wrapped.strip():
                # consecutive blanks (and a blank landing at the top of a fresh
                # screen) collapse into one. spacing between blocks is written
                # by hand at ~40 call sites, so doubled and leading gaps were
                # bound to happen; swallowing them here keeps the rhythm of the
                # screen even without us auditing every caller.
                if not self._blank_last:
                    self._console.print()
                    self._blank_last = True
                    self._rows_painted += 1
                continue
            self._console.print(wrapped, style=style or None, soft_wrap=True)
            self._rows_painted += 1
            self._blank_last = False

    def note_width(self) -> int:
        """How wide a note's text may be, in cells."""
        return max(min(self._console.width, MEASURE), 20)

    def table(self, title: str, rows: Sequence[Sequence[str]],
              headers: Sequence[str]) -> None:
        from rich.table import Table

        # if the columns don't need names, we don't draw an empty band across
        # the top of every table.
        named = any(str(h).strip() for h in headers)
        table = Table(title=title, title_justify="left", header_style="bold",
                      title_style="bold cyan", show_header=named,
                      border_style="grey42")
        for h in headers:
            table.add_column(str(h))
        for row in rows:
            table.add_row(*[str(c) for c in row])
        # measured rather than guessed, for the same reason as the notes: a
        # table of unknown height sits between the banner and the question.
        self._rows_painted += len(self._console.render_lines(table, pad=False))
        self._console.print(table)
        self._blank_last = False

    def stage(self, key: str, label: str, *, status: str = "active",
              detail: str = "") -> None:
        """No-op: this renderer prints line by line and has no rail to update."""

    def reset_stages(self) -> None:
        """No-op, for the same reason: there is no rail to forget."""

    def reason(self, text: str) -> None:
        """
        Say why the next question is being asked.

        Distinct from :meth:`note` in where it lands and how loudly. A note is
        commentary printed at the top of the screen; a reason is the sentence
        that makes the question underneath it make sense, and it was getting
        lost -- dim, and separated from its question by the whole rail.
        """
        self.note("")
        self.note(f"  {_REASON_MARK} {text}", style=_REASON_STYLE)

    # -- input -------------------------------------------------------------
    def _ask(self, question_obj) -> Any:
        answer = question_obj.ask()
        if answer is None:
            raise Cancelled()
        return answer

    def text(self, question: str, *, default: str = "") -> str:
        # the "› " sits exactly where typing will land. without it a typed
        # answer had no visual target at all -- this bit us on the delete
        # screen, where the question ends in a colon and then... nothing.
        return self._ask(self._q.text(question, default=default,
                                      style=self._style, instruction="› "))

    def path(self, question: str, *, default: str = "") -> str:
        # questionary's path prompt gives us filesystem tab-completion, which
        # is the single biggest quality-of-life win in the whole wizard.
        return self._ask(self._q.path(question, default=default, style=self._style))

    def confirm(self, question: str, *, default: bool = True) -> bool:
        return self._ask(self._q.confirm(question, default=default, style=self._style))

    def pause(self, message: str = PAUSE_MESSAGE) -> None:
        """
        Wait for the reader, and take no answer from them.

        Deliberately not routed through :meth:`_ask`, which turns Esc into
        `Cancelled`. There is nothing here to cancel -- the screen has already
        been shown -- and a reader who presses Esc means the same thing as one
        who presses Enter.
        """
        self._wait_for_key(message)

    def _wait_for_key(self, message: str) -> None:
        """
        Block until a key is pressed. Split out so :class:`LivePrompter` can
        repaint around it without reimplementing the read.
        """
        try:
            self._q.press_any_key_to_continue(message, style=self._style).ask()
        except Exception:
            # the any-keypress reader is a courtesy. a questionary version that
            # moved it, or a terminal that can't host it, shouldn't take down a
            # screen whose work is already finished and up on display.
            pass

    def select(self, question: str, choices: Sequence[Choice],
               *, default: Optional[str] = None, transient: bool = False,
               toggle_values: Sequence[str] = (),
               ticked: Optional[set] = None,
               navigate: Optional[Callable[[str], Optional[dict]]] = None,
               breadcrumb: Optional[Callable[[], str]] = None,
               cycle: Optional[Callable[[str, int], Optional[str]]] = None
               ) -> str:
        # `cycle` needs a key binding this renderer doesn't have; the
        # annotation still shows what each column is being treated as.
        # this renderer rebuilds the prompt for every screen, so there's
        # nothing to swap in place: we ignore `navigate`, hand the folder value
        # back, and the caller's fallback loop re-asks with the new listing.
        # the breadcrumb gets folded into the question, re-read on each ask.
        if breadcrumb is not None:
            question = f"{question}   {breadcrumb()}"
        rows = list(choices)
        toggles = set(toggle_values)
        while True:
            picked = self._ask(self._q.select(
                question,
                choices=[self._to_q(c) for c in rows],
                default=default,
                style=self._style,
                use_shortcuts=False,
                # draws the highlighted row's `description` under the list
                show_description=True,
            ))
            if ticked is None or picked not in toggles:
                return picked
            # no space binding here, so enter on a tickable row flips it and we
            # re-ask the list with the pointer left on that row. before this,
            # the ticks were silently ignored: under --plain a step could use
            # all of its dictionaries, or exactly one, and nothing in between.
            # (a genuine hole, and the old comment up here said nothing about
            # it.)
            ticked.symmetric_difference_update({picked})
            rows = [replace(c, label=flip_tick_mark(c.label)) if c.value == picked
                    else c for c in rows]
            default = picked

    def checkbox(self, question: str, choices: Sequence[Choice], *,
                 cycle: Optional[Callable[[str, int], Optional[str]]] = None
                 ) -> List[str]:
        return self._ask(self._q.checkbox(
            question,
            choices=[self._to_q(c) for c in choices],
            style=self._style,
            show_description=True,
        ))

    def _to_q(self, choice: Choice):
        """
        One of our choices as one of questionary's.

        The help goes in `description`, which questionary renders in full under
        the list for whichever row is highlighted -- not appended to the label.
        Appended, it had to be elided to keep each choice on one line, so the
        end of every explanation was replaced by an ellipsis: the text existed
        to be read and was the one part guaranteed not to be.
        """
        from rich.cells import cell_len

        # the label still can't wrap (questionary draws a row on one line), but
        # labels are short by design, and we leave 6 columns for the pointer
        # glyphs and a right-hand margin (plus the annotation's own width, when
        # there is one, so that the colored part is never what gets truncated
        # away).
        width = terminal_width()
        label_class = f"class:tone-{choice.tone}" if choice.tone else "class:text"
        if choice.annotation:
            label = fit(choice.label, width=width,
                        reserve=6 + cell_len(choice.annotation) + 1)
            title = [(label_class, label),
                     ("class:annotation", " " + choice.annotation)]
            row = cell_len(label) + 1 + cell_len(choice.annotation)
        elif choice.tone:
            label = fit(choice.label, width=width, reserve=6)
            title = [(label_class, label)]
            row = cell_len(label)
        else:
            title = fit(choice.label, width=width, reserve=6)
            row = cell_len(title)

        disabled = None
        if choice.disabled:
            # questionary draws a disabled row's reason itself, after the label
            # on the same line, as " (reason)" -- so it sat outside everything
            # we fitted above, and a 75-character reason ran the row off the
            # right edge: "...import one under Settings firs". someone actually
            # saw that. so the label keeps its width and the reason gets
            # what's left, elided as a last resort: the shape of the list
            # matters more than the tail of the reason. (this is also why the
            # wizard keeps its reasons short enough to fit beside their labels
            # at 80 columns.)
            disabled = fit(choice.disabled, width=max(width - 6 - row - 3, 12))
        return self._q.Choice(
            title=title,
            value=choice.value,
            checked=choice.checked,
            disabled=disabled,
            description=wrap_description(choice.help) if choice.help else None,
        )


# ---------------------------------------------------------------------------
# the test one
# ---------------------------------------------------------------------------

@dataclass
class ScriptedPrompter:
    ticks_in_place = True   # "\x00space:<value>" scripts the key
    """
    A :class:`Prompter` that reads its answers from a list instead of a person.

    This is what makes the wizard testable end to end. Give it the answers a
    user would have given, in order, and every question is answered without a
    terminal:

        p = ScriptedPrompter(["./media", "audio", ["transcribe"], False, "run", True, False])

    Answers are consumed in the order the wizard asks. Everything printed is
    kept in :attr:`output`, every question in :attr:`asked`, every
    pre-filled default in :attr:`presented`, and every list of options in
    :attr:`offered` -- so a test can assert on what the
    user would have seen, not just on what came back. `presented` matters more
    than it looks: a prompt's default *is* how the wizard shows you the current
    value of a setting, and showing the wrong one is a real bug even though the
    returned answer is unaffected.
    """

    answers: List[Any] = field(default_factory=list)
    asked: List[Tuple[str, str]] = field(default_factory=list)
    presented: List[Any] = field(default_factory=list)
    offered: List[Tuple[str, List[Choice]]] = field(default_factory=list)
    #: (row, new annotation) for every scripted arrow press, for assertions.
    cycled: List[tuple] = field(default_factory=list)
    tables: List[Tuple[str, List[List[str]], List[str]]] = field(default_factory=list)
    stages: List[Stage] = field(default_factory=list)
    #: Why each question was asked, in order, for tests that check a
    #: question explained itself before it was put.
    reasons: List[str] = field(default_factory=list)
    #: The `default=` each select was given, by question -- where the pointer
    #: starts, which is a separate fact from the order the options are listed
    #: in, and worth asserting separately.
    select_defaults: Dict[str, Optional[str]] = field(default_factory=dict)
    #: What the breadcrumb said on each screen that had one, in order -- the
    #: browser's "you are here", one entry per listing shown.
    breadcrumbs: List[str] = field(default_factory=list)
    output: List[str] = field(default_factory=list)
    _cursor: int = 0

    def _next(self, kind: str, question: str, default: Any = None) -> Any:
        self.asked.append((kind, question))
        self.presented.append(default)
        if self._cursor >= len(self.answers):
            raise AssertionError(
                f"ScriptedPrompter ran out of answers at question "
                f"{len(self.asked)}: [{kind}] {question!r}"
            )
        value = self.answers[self._cursor]
        self._cursor += 1
        return value

    # -- output ------------------------------------------------------------
    def clear(self) -> None:
        self.output.append("<clear>")

    def repaint(self) -> None:
        self.output.append("<repaint>")

    def working(self, text: str) -> None:
        # we record this in `output` in sequence, so a test can check that the
        # "hold on, this may take a moment" line landed *before* the slow
        # work's results.
        self.output.append(text)

    def note(self, text: str, *, style: str = "", wrap: bool = True) -> None:
        self.output.append(text)

    def table(self, title: str, rows: Sequence[Sequence[str]],
              headers: Sequence[str]) -> None:
        self.output.append(title)
        # we keep the whole table, so a test can check what it actually
        # contained rather than just that one got drawn.
        self.tables.append((title, [list(r) for r in rows], list(headers)))
        for row in rows:
            self.output.append(" | ".join(str(c) for c in row))

    # -- input -------------------------------------------------------------
    def text(self, question: str, *, default: str = "") -> str:
        return self._next("text", question, default)

    def path(self, question: str, *, default: str = "") -> str:
        return self._next("path", question, default)

    def confirm(self, question: str, *, default: bool = True) -> bool:
        return self._next("confirm", question, default)

    def select(self, question: str, choices: Sequence[Choice],
               *, default: Optional[str] = None, transient: bool = False,
               toggle_values: Sequence[str] = (),
               ticked: Optional[set] = None,
               navigate: Optional[Callable[[str], Optional[dict]]] = None,
               breadcrumb: Optional[Callable[[], str]] = None,
               cycle: Optional[Callable[[str, int], Optional[str]]] = None
               ) -> str:
        self.offered.append((question, list(choices)))
        self.select_defaults[question] = default
        if breadcrumb is not None:
            self.breadcrumbs.append(breadcrumb())
        valid = {c.value for c in choices}
        toggles = set(toggle_values)
        while True:
            value = self._next("select", question, default)
            if self._scripted_cycle(value, cycle, valid):
                continue
            # "\x00space:<value>" scripts the space key: it ticks in place and
            # the prompt keeps running, so it consumes an answer without
            # returning one -- exactly like the real gesture.
            if isinstance(value, str) and value.startswith("\x00space:"):
                target = value[len("\x00space:"):]
                assert ticked is not None, "space scripted on a screen without ticks"
                assert target in toggles, (
                    f"space on {target!r}, which is not tickable here")
                ticked.symmetric_difference_update({target})
                continue
            assert value in valid, (
                f"scripted answer {value!r} not among {sorted(valid)}")
            if navigate is not None:
                swap = navigate(str(value))
                if swap is not None:
                    # the live renderer swaps the rows in place, so the script
                    # does the same: walking folders happens inside ONE select,
                    # eating one answer per step and recording each listing.
                    choices = list(swap["choices"])
                    self.offered.append((question, choices))
                    valid = {c.value for c in choices}
                    toggles = set(swap.get("toggle_values", ()))
                    if breadcrumb is not None:
                        self.breadcrumbs.append(breadcrumb())
                    continue
            return value

    def pause(self, message: str = PAUSE_MESSAGE) -> None:
        """
        Recorded, but consumes no scripted answer.

        That is the point of it being separate from `confirm`: a pause is not a
        decision, so a test driving a screen that ends in one does not have to
        supply an answer for it -- and cannot accidentally feed the pause an
        answer meant for the next real question.
        """
        self.asked.append(("pause", message))
        self.output.append(message)

    def checkbox(self, question: str, choices: Sequence[Choice], *,
                 cycle: Optional[Callable[[str, int], Optional[str]]] = None
                 ) -> List[str]:
        self.offered.append((question, list(choices)))
        valid = {c.value for c in choices}
        while True:
            value = self._next("checkbox", question,
                               [c.value for c in choices if c.checked])
            if self._scripted_cycle(value, cycle, valid):
                continue
            unknown = set(value) - valid
            assert not unknown, f"scripted answers {sorted(unknown)} not among {sorted(valid)}"
            return list(value)

    def _scripted_cycle(self, value, cycle, valid) -> bool:
        """``"\x00right:<value>"`` / ``"\x00left:<value>"`` script the arrow
        keys on a row: the cycle runs and the question is still open."""
        if not (isinstance(value, str) and value.startswith(("\x00right:",
                                                             "\x00left:"))):
            return False
        key, target = value[1:].split(":", 1)
        assert cycle is not None, f"{key} scripted on a screen with nothing to cycle"
        assert target in valid, f"{key} on {target!r}, which is not on this screen"
        self.cycled.append((target, cycle(target, 1 if key == "right" else -1)))
        return True

    @property
    def text_output(self) -> str:
        """Everything printed, joined -- convenient for `in` assertions."""
        return "\n".join(self.output)

    def stage(self, key: str, label: str, *, status: str = "active",
              detail: str = "") -> None:
        self.stages.append(Stage(key=key, label=label, status=status, detail=detail))

    def reset_stages(self) -> None:
        # we record this rather than drop it, so a test can check the rail
        # got cleared without losing what was on it beforehand.
        self.output.append("<reset stages>")
        self.stages = []

    def reason(self, text: str) -> None:
        self.output.append(text)
        self.reasons.append(text)

    def offered_choices(self, question_startswith: str) -> List[Choice]:
        """
        The options shown for the first matching question.

        What a user was *offered* is as much a part of the interface as what
        they answered: a checklist that quietly includes vocal acoustics for a
        folder of essays is a bug no assertion on the return value would catch.
        """
        for question, choices in self.offered:
            if question.startswith(question_startswith):
                return choices
        raise AssertionError(
            f"no question started with {question_startswith!r}. Asked: "
            + "; ".join(q for q, _ in self.offered)
        )


def terminal_width(default: int = 80) -> int:
    """Usable width, for wrapping help text."""
    return shutil.get_terminal_size((default, 24)).columns


#: What questionary prints in front of a choice's description. Its length is
#: what the continuation lines have to be indented by to line up under it.
_DESCRIPTION_PREFIX = "  Description: "


#: Lines the surroundings take before a list gets any: the banner, the progress
#: rail, the question, and a wrapped description under it. Deliberately generous
#: -- guessing high shows a few rows fewer than would fit, while guessing low
#: puts rows off the bottom with a marker claiming they are visible.
_CHROME_LINES = 16

#: What the renderer measured above and below the list on *this* screen, or
#: None when nothing has measured. A fixed guess cannot work: the chrome is a
#: banner plus a progress rail plus however many lines of explanation the
#: current setting happens to carry, which on a well-documented parameter is
#: three times the guess. Sizing the window off the guess is what let a long
#: option list push the banner off the top of the screen.
_measured_chrome: Optional[int] = None


def set_chrome_rows(rows: Optional[int]) -> None:
    """Tell :func:`visible_rows` how much of the screen is already spoken for."""
    global _measured_chrome
    _measured_chrome = rows


#: How many lines the description under a list is held at, whichever row is
#: pointed at. Zero when nothing has asked for a reservation.
_reserved_description_rows = 0


def set_description_rows(rows: int) -> None:
    """
    Hold the description block under a list at a fixed height.

    questionary draws the pointed row's description under the list and
    nothing at all when that row has none, so a list's height changed with
    every arrow press -- and each change scrolled the terminal, walking the
    explanation printed above the question up the screen a line or two at a
    time (a real report: "the yellow text moves up"). Reserving the tallest
    description's height and padding shorter ones keeps the whole screen
    still.
    """
    global _reserved_description_rows
    _reserved_description_rows = max(0, int(rows))


def description_rows(choices: Sequence["Choice"], *,
                     width: Optional[int] = None, cap: int = 6) -> int:
    """The lines the tallest description among ``choices`` will take, capped
    so one essay of a help text cannot eat the list's room."""
    tallest = 0
    for choice in choices:
        if choice.help:
            lines = wrap_description(choice.help, width=width).count("\n") + 1
            tallest = max(tallest, lines)
    return min(tallest, cap)

#: Never show fewer than this, however small the terminal claims to be, and
#: however much wants the space above. Five was enough to prove a list
#: scrolls and nothing like enough to *use* one: picking two text columns out
#: of 150 through a five-row porthole is not a thing anyone should be asked
#: to do (a real report). The painter above trims what it prints to keep
#: this many rows free, so this floor is the backstop rather than the plan.
_MIN_VISIBLE_ROWS = 12


def visible_rows(lines: Optional[int] = None) -> int:
    """
    How many choices a list can show before it has to scroll.

    Measured from the terminal prompt_toolkit is actually drawing into, not from
    `shutil`. The two agree in production and can differ anywhere the output is
    redirected -- a captured session, a test harness -- and a window sized to a
    different terminal than the one being drawn puts rows off the bottom while
    claiming they are visible.
    """
    if lines is None:
        try:
            from prompt_toolkit.application.current import get_app

            lines = get_app().output.get_size().rows
        except Exception:
            import shutil

            lines = shutil.get_terminal_size((80, 24)).lines
    chrome = _CHROME_LINES if _measured_chrome is None else _measured_chrome
    return max(_MIN_VISIBLE_ROWS, lines - chrome - _reserved_description_rows)


def note_lines(text: object, width: int) -> List[str]:
    """
    Exactly the lines :meth:`Prompter.note` will print for this text.

    Wrapped here rather than by rich so the indent is real text instead of
    padding. Padding fills each line out to the full width, which leaves
    trailing whitespace on every line of every note -- invisible on screen,
    and there in anything the user copies out of the terminal.

    Shared with the screen painter, which has to know how tall a note is
    *before* printing it so a long one cannot crowd the question underneath.
    Measuring by re-implementing the wrapping is how the two drift apart, so
    there is one function and the printer calls it too. (Runs of blank lines
    collapse when printed, which depends on what came before; this returns
    them uncollapsed, so a height taken from it is an upper bound.)
    """
    import textwrap

    out: List[str] = []
    for line in str(text).split("\n"):
        body = line.strip()
        if not body:
            out.append("")
            continue
        margin = " " * (len(line) - len(line.lstrip(" ")))
        out += textwrap.wrap(
            body, width=width, initial_indent=margin, subsequent_indent=margin,
            # both off, because both would mangle what they touch here: paths
            # get broken across lines at their separators, and a sentence
            # ending in a version number picks up a stray second space.
            break_on_hyphens=False, break_long_words=False,
        ) or [margin + body]
    return out


def _window_rows(rows: List[list], pointed_at: int, room: int) -> Tuple[List[list], int, int]:
    """
    The slice of `rows` to show, and how many fall outside it either side.

    Centered on the pointer where there is room to center, and clamped at both
    ends so moving into the first or last few choices does not leave half the
    window empty.
    """
    if len(rows) <= room:
        return rows, 0, 0
    start = max(0, min(pointed_at - room // 2, len(rows) - room))
    return rows[start:start + room], start, len(rows) - (start + room)


def _padded_tail(tail: List[list], reserved: int) -> List[list]:
    """
    The description rows under a list, held at ``reserved`` lines.

    questionary emits the pointed row's description as its own lines and
    emits nothing when that row has none, so the block is as tall as the
    current description -- see :func:`set_description_rows` for why that
    has to stop. Blank rows make up the difference; a description taller
    than the reservation is left whole rather than cut.
    """
    if reserved <= 0:
        return tail
    rows = list(tail)
    while len(rows) < reserved:
        rows.append([("", " ")])
    return rows


def scroll_long_lists() -> bool:
    """
    Show a window onto a long list, with markers for what is off each end.

    prompt_toolkit already scrolls to keep the pointer visible, so a list longer
    than the terminal has always been navigable -- but nothing said so. Rows
    simply were not there, with no hint that arrowing further would reveal them,
    which reads as a list that is missing options rather than one that continues.

    Applied to every list at once -- module options, the file browser, the step
    menus -- because the fix belongs to the renderer rather than to any one
    question.
    """
    try:
        from questionary.prompts.common import InquirerControl
    except Exception:
        return False

    if getattr(InquirerControl, "_taters_scrolls", False):
        return True

    original = InquirerControl._get_choice_tokens

    def windowed(self):
        tokens = original(self)
        rows: List[list] = [[]]
        for token in tokens:
            if tuple(token[:2]) == ("", "\n"):
                rows.append([])
            else:
                rows[-1].append(token)

        count = len(getattr(self, "filtered_choices", []) or [])
        choices, tail = rows[:count], rows[count:]
        tail = _padded_tail(tail, _reserved_description_rows)
        pointed = int(getattr(self, "pointed_at", 0) or 0)

        # the pointed row turns the highlight color *whole*. questionary only
        # applies `class:highlighted` to plain-string titles; a row carrying
        # its own styles (the green "✓ Use this folder", an annotated folder)
        # kept them when pointed, so the only sign of where you were was the
        # pointer glyph. we append the class here so the highlight wins, since
        # later classes override earlier ones.
        if 0 <= pointed < len(choices):
            choices[pointed] = [
                (f"{style} class:highlighted" if "pointer" not in style else style,
                 text)
                for style, text in choices[pointed]
            ]

        room = visible_rows()
        if not choices or len(choices) <= room:
            out = []
            for chunk in choices:
                out += chunk
                out.append(("", "\n"))
            for chunk in tail:
                out += chunk
                out.append(("", "\n"))
            if out and tuple(out[-1][:2]) == ("", "\n"):
                out.pop()
            return out

        # rows pinned above the scroll. a gated tick screen sets one: its
        # "✓ Done" row is the way out, and a way out that scrolls off the top
        # of a forty-option list is a door someone has to go looking for. the
        # pinned rows stay put; the "▲ N more above" marker sits between them
        # and the window.
        sticky_n = min(int(getattr(self, "_taters_sticky_rows", 0) or 0),
                       len(choices))
        sticky, rest = choices[:sticky_n], choices[sticky_n:]
        shown, above, below = _window_rows(
            rest, max(0, int(getattr(self, "pointed_at", 0) or 0) - sticky_n),
            max(1, room - sticky_n))

        out: List[tuple] = []
        for row in sticky:
            out += row
            out.append(("", "\n"))
        # both marker rows get drawn once a list scrolls, blank when there's
        # nothing on that side. we used to draw them only when they had
        # something to say, so the list grew a line the moment the pointer
        # passed the middle (▲ showed up) and lost one at the very end (▼ went
        # away) -- and every change of height scrolled the terminal, walking
        # the explanation above the question up the screen. now a windowed
        # list is the same height on every keypress, and it's the height we
        # budgeted for anyway (_BELOW_LIST counts both markers)
        out += [("class:instruction", f"   ▲ {above} more above" if above else " "),
                ("", "\n")]
        for row in shown:
            out += row
            out.append(("", "\n"))
        out += [("class:instruction", f"   ▼ {below} more below" if below else " ")]
        for row in tail:
            if out and tuple(out[-1][:2]) != ("", "\n"):
                out.append(("", "\n"))
            out += row
        return out

    try:
        InquirerControl._get_choice_tokens = windowed
        InquirerControl._taters_scrolls = True
    except Exception:      # pragma: no cover - a hardened questionary
        return False
    return True


def style_descriptions_separately() -> bool:
    """
    Give a choice's description its own style class, so it can be colored.

    questionary tags the description with ``class:text`` -- the same class it
    uses for every unselected option title. Restyling that class would recolour
    the whole list, so the description cannot be told apart from the options it
    is explaining without this.

    The interception is deliberately narrow: it rewrites the class of exactly
    one token, identified by the prefix questionary itself writes, and leaves
    every other token as it found it. Applied once, and reported rather than
    assumed -- if a future questionary builds its tokens differently the styling
    is simply not applied, which is the state this started in.

    Returns
    -------
    bool
        Whether the interception is in place.
    """
    try:
        from questionary.prompts.common import InquirerControl
    except Exception:
        return False

    if getattr(InquirerControl, "_taters_description_class", False):
        return True

    original = InquirerControl._get_choice_tokens

    def with_description_class(self):
        out = []
        for token in original(self):
            if (len(token) == 2 and isinstance(token[1], str)
                    and token[1].startswith(_DESCRIPTION_PREFIX)):
                # a breath between the menu and the sentence explaining the
                # pointed row. butted straight up against the last option, the
                # description read as one more row of the list.
                out.append(("", "\n"))
                out.append(("class:description", token[1]))
            else:
                out.append(token)
        return out

    try:
        InquirerControl._get_choice_tokens = with_description_class
        InquirerControl._taters_description_class = True
    except Exception:      # pragma: no cover - a hardened questionary
        return False
    return True


def wrap_description(text: str, *, width: Optional[int] = None) -> str:
    """
    Fold a choice's description so all of it is visible.

    questionary renders a description as a single run of text and does not wrap
    it, so anything longer than the terminal is simply cut -- and the sentence
    that explains an option is exactly the sentence someone is reading when they
    cannot decide. One real menu ended mid-word: "...which is a large install
    (NeMo) plu".

    Continuation lines are indented to sit under the first rather than returning
    to column zero, so the block reads as one paragraph attached to the option
    instead of as unrelated text under the list.
    """
    import textwrap

    columns = (width or terminal_width()) - len(_DESCRIPTION_PREFIX) - 2
    if columns < 20:                     # a terminal this narrow is beyond help
        return text
    lines = textwrap.wrap(" ".join(str(text).split()), width=columns)
    return ("\n" + " " * len(_DESCRIPTION_PREFIX)).join(lines)


def fit(text: str, *, reserve: int = 0, width: Optional[int] = None) -> str:
    """
    Shorten one line so the terminal never has to break it.

    questionary draws a choice on a single line and does no wrapping of its
    own, so anything too long is hard-wrapped by the terminal -- mid-word, with
    the remainder dangling on the next line under no pointer. In a menu that
    turns a tidy list into a wall. Better to lose the tail of a description
    than the shape of the list, so this elides instead.

    Parameters
    ----------
    reserve : int
        Columns already spoken for by whatever draws the line -- questionary's
        pointer and checkbox glyphs, an indent.
    """
    limit = (width or terminal_width()) - reserve
    if limit < 12:                       # a terminal this narrow is beyond help
        return text
    # we only collapse line breaks, not runs of spaces: callers pad names
    # into columns, and flattening that would undo the very alignment this is
    # supposed to protect.
    single = str(text).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    if len(single) <= limit:
        return single
    return single[: limit - 1].rstrip() + "…"
