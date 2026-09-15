"""
The application-style renderer: a live region above each question.

:class:`~taters.ui.prompts.QuestionaryPrompter` asks one question after another
and lets them scroll past, which reads like a shell script rather than a
program. This renderer keeps a **progress rail** pinned above whatever question
is currently open -- what you have answered, what you are answering, what is
still to come -- so the wizard feels like one application rather than a
sequence of prompts.

Why it is not a full-screen app
-------------------------------
Taking over the terminal (the alt screen, like ``vim`` or ``htop``) would make
the framing easier, and it would also throw the entire session away the moment
the wizard exits: no scrollback, nothing to copy, nothing to paste into a bug
report. So this renders **inline**, in the normal buffer. The rail is drawn as
part of the prompt's own layout, which means prompt_toolkit erases it when the
question is answered, leaving only questionary's one-line record of the answer
behind. Scroll up after a run and you see the questions and your answers, in
order, exactly as if they had been printed.

How it works
------------
Every questionary question is a ``prompt_toolkit`` ``Application`` whose layout
is reachable as ``question.application.layout``. :meth:`LivePrompter._ask`
wraps that layout in an ``HSplit`` with the rail on top and a key hint below,
then hands it back. questionary's own widgets -- and all of their editing,
filtering and validation behavior -- are untouched, which is the whole reason
this is a hundred lines instead of a thousand.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Optional, Sequence

from .prompts import (MEASURE, PAUSE_MESSAGE, _REASON_MARK, _REASON_STYLE,
                      Cancelled, Choice, GoBack, QuestionaryPrompter,
                      Stage, _MIN_VISIBLE_ROWS, description_rows,
                      flip_tick_mark, note_lines, set_chrome_rows,
                      set_description_rows, with_annotation)

__all__ = ["LivePrompter"]


# marker and color for each stage status. we picked these to survive a
# monochrome terminal: the glyphs differ, so the rail still reads with the
# color stripped out.
_MARKERS = {
    "done": ("✓ ", "fg:#00af5f"),
    "active": ("▸ ", "fg:#00afaf bold"),
    "todo": ("○ ", "fg:#585858"),
}

# every key gets brackets, and every action is a word that works for readers
# whose first language isn't English. the old bare style spelled a key and its
# action as two naked words -- "a all", "i invert" -- which looks like a typo,
# not an instruction, unless you already know it's a key binding.
#: Every hint names what [enter] acts on -- "the highlighted row" -- because
#: the highlight itself was read as a state ("already selected? the default?
#: the best one?") rather than as the cursor it is.
_KEY_HINTS = {
    "select": "[↑↓] move · [enter] picks the highlighted row · [1-9] jump · [esc] back",
    # too many rows to number, so the hint had better not offer a digit
    "select_long": "[↑↓] move · [enter] picks the highlighted row · [esc] back",
    # tick screens: entries toggle, actions choose
    "tick": "[↑↓] move · [space] tick · [enter] picks the highlighted row · [esc] back",
    # the one multi-select dialect (see `checkbox`): boxes tick, and the Done
    # row is the only way forward
    "checkbox": "[space] ticks a box · [enter] on ✓ Done continues · [esc] back",
    "confirm": "[y]/[n] answer · [enter] picks the highlighted row · [esc] back",
}
_DEFAULT_HINT = "[enter] accept · [esc] back · [ctrl-c] quit"

# the rail's one line, plus the blank that separates it from the question
_RAIL_ROWS = 2

# rows a question takes up besides the choices themselves: the question line,
# the two "N more above/below" markers, the blank line before the description,
# the key hint, and one spare so the shell prompt doesn't land tight against
# the last row. we count these out rather than guess, because our first version
# forgot every one of them and the screen overshot by six. the description
# block gets reserved separately, at the height of the tallest description on
# the screen (`set_description_rows`).
_BELOW_LIST = 1 + 2 + 1 + 1 + 1

#: Rows the option list is entitled to, whatever else wants the screen. The
#: list is the *point* of the screen; the notes above it are context. A
#: spreadsheet of 150 columns produced a note naming all 150, which pushed the
#: list of those same columns down to five visible rows -- a porthole to pick
#: two columns through, with the answer scrolled off in both directions (a
#: real report). So the notes are trimmed to fit around the list rather than
#: the list being squeezed into whatever the notes leave.
#:
#: The same number the renderer refuses to go below, taken from there rather
#: than written twice: a painter that reserves less than the list will insist
#: on is a painter that trims notes for nothing.
_LIST_FLOOR = _MIN_VISIBLE_ROWS

#: The fewest rows of notes worth keeping on a screen too small for both.
#: Below this the trimming is doing more harm than the crowding was.
_NOTES_FLOOR = 2

#: Rows a rich table spends on itself around its body: the title, the top
#: border, the header row, the rule under it and the bottom border.
_TABLE_CHROME = 5


def _is_blank(entry) -> bool:
    """Whether a remembered note is only spacing."""
    kind, payload = entry
    return kind == "note" and not str(payload[0]).strip()


#: How long the glide back to the Done row takes, whatever the distance --
#: long enough to read as motion, short enough not to be waited for. Two
#: seconds was asked for and found slow in use; a quarter of a second is a
#: flick the eye still registers as movement rather than a jump.
GLIDE_SECONDS = 0.25
#: Frames per second for the glide; prompt_toolkit repaints only the list.
GLIDE_FPS = 40


def glide_positions(start: int, target: int, *, seconds: float = GLIDE_SECONDS,
                    fps: int = GLIDE_FPS) -> List[int]:
    """
    The pointer positions of a glide from ``start`` to ``target``, one per
    frame, ending exactly on the target.

    Fixed duration, not fixed speed: a list of five hundred rows glides in
    the same two seconds as a list of forty, so the wait never grows with
    the list. Never more frames than rows -- a three-row glide is three
    frames -- and never fewer than one.
    """
    distance = target - start
    if distance == 0:
        return []
    frames = max(1, min(abs(distance), int(round(seconds * fps))))
    return [start + round(distance * (i / frames)) for i in range(1, frames + 1)]


def _glide(app, control, target: int, *, seconds: float = GLIDE_SECONDS,
           fps: int = GLIDE_FPS) -> None:
    """Move the pointer to ``target`` over ``seconds``, repainting each frame.
    Stops the moment the person moves the pointer themselves."""
    import asyncio

    positions = glide_positions(int(control.pointed_at), target,
                                seconds=seconds, fps=fps)
    if not positions:
        return
    interval = seconds / len(positions)
    expected = [int(control.pointed_at)]
    loop = asyncio.get_event_loop()

    def step(i: int) -> None:
        if int(control.pointed_at) != expected[0]:
            return                      # the user grabbed the wheel; leave it
        control.pointed_at = positions[i]
        expected[0] = positions[i]
        app.invalidate()
        if i + 1 < len(positions):
            loop.call_later(interval, step, i + 1)

    loop.call_later(interval, step, 0)


def _inquirer_control(application):
    """The list widget inside a running questionary prompt, or None. Three key
    bindings need it and each carried its own copy of this walk."""
    from questionary.prompts.common import InquirerControl

    for window in application.layout.find_all_windows():
        if isinstance(window.content, InquirerControl):
            return window.content
    return None


class LivePrompter(QuestionaryPrompter):
    ticks_in_place = True
    """
    A :class:`~taters.ui.prompts.Prompter` that keeps a live progress rail.

    Everything :class:`~taters.ui.prompts.QuestionaryPrompter` does for output
    and input is inherited unchanged; this only adds the framing.
    """

    # class-level defaults, for the same reason `_blank_last` has one: the tests
    # build a bare renderer with ``__new__`` to poke at one method, and a
    # missing attribute here would take the whole prompt down instead of just
    # mis-sizing a window.
    _stages: List[Stage] = []
    _header: str = ""
    _reason: str = ""
    #: Rows the question about to be asked needs for its own list. Zero on a
    #: screen that only reports something -- the "Check my setup" table is
    #: the whole point of its screen, and trimming it to leave room for a
    #: list that never comes would be vandalism.
    _reserve_rows: int = 0

    def __init__(self, title: str = "Taters") -> None:
        super().__init__()
        self._title = title
        self._stages: List[Stage] = []
        self._header: str = ""
        self._screen_notes: List[tuple] = []
        self._reason: str = ""

    # -- the screen --------------------------------------------------------
    def set_header(self, header) -> None:
        """
        The banner to redraw at the top of every screen.

        Accepts a string, or a zero-argument callable rendered at each paint.
        The callable is what makes the banner's slow border-color drift real:
        rendered once into a string, the drift was recomputed exactly once per
        session and the "animation" never visibly moved.
        """
        self._header = header

    def note(self, text: str, *, style: str = "", wrap: bool = True) -> None:
        """
        Print, and remember it for the next screen.

        Each question wipes the screen, so a note printed between two questions
        would vanish before it had been read. Holding onto it until the next
        question has been *answered* is what lets "Found 412 .txt files" or
        "ffmpeg was not found" stay visible for exactly as long as it is about
        the thing on screen.
        """
        super().note(text, style=style, wrap=wrap)
        self._screen_notes.append(("note", (text, style, wrap)))

    def table(self, title: str, rows, headers) -> None:
        """
        Print a table, and remember it for the next screen.

        Same reason as :meth:`note`, and the omission was worse here: a whole
        setup report would be drawn, then wiped by the very next question,
        leaving only the one-line advice underneath a header that suggested
        nothing had been printed at all.
        """
        super().table(title, rows, headers)
        self._screen_notes.append((
            "table",
            (title, [list(r) for r in rows], list(headers)),
        ))

    def repaint(self) -> None:
        """Redraw the screen furniture without asking anything."""
        self._paint()

    def _entry_height(self, entry) -> int:
        """How many rows a remembered note or table will take."""
        kind, payload = entry
        if kind == "note":
            text, _style, wrap = payload
            if not wrap:
                return str(text).count("\n") + 1
            return len(note_lines(text, self.note_width()))
        title, rows, headers = payload
        from rich.table import Table

        table = Table(title=title, show_header=any(str(h).strip()
                                                   for h in headers))
        for h in headers:
            table.add_column(str(h))
        for row in rows:
            table.add_row(*[str(c) for c in row])
        return len(self._console.render_lines(table, pad=False))

    def _notes_budget(self) -> int:
        """
        How many rows the notes may take on this screen.

        Everything else on it is either fixed (the banner, the rail, the
        question and its key hints) or owed a minimum (the list). What is
        left over is the notes' share -- which is the right way round: a note
        explains the question, and an explanation that hides the answers has
        stopped explaining anything.
        """
        if self._reserve_rows <= 0:
            return 10 ** 6          # nothing's competing for space; print it all
        rows = getattr(self._console.size, "height", 24) or 24
        spoken_for = (self._rows_painted + self._rail_height()
                      + len(self._reason_lines()) + _BELOW_LIST
                      + self._reserve_rows)
        return max(_NOTES_FLOOR, rows - spoken_for)

    def _paint(self) -> None:
        """Wipe and redraw everything above the question."""
        self._console.clear()
        self._blank_last = True
        self._rows_painted = 0
        if self._header:
            header = self._header() if callable(self._header) else self._header
            self._console.print(header, soft_wrap=True)
            self._rows_painted += header.count("\n") + 1

        # we keep the most recent notes that fit: the newest one is about the
        # question on screen, and an older one that's already been read is the
        # right thing to lose when something has to go.
        budget = self._notes_budget()
        kept, used = [], 0
        said_something = False
        for entry in reversed(self._screen_notes):
            height = self._entry_height(entry)
            if said_something and used + height > budget:
                break
            kept.append(entry)
            used += height
            # a blank line is spacing, not a note. when we counted it as "the
            # newest note, kept", the empty line printed before "What next?"
            # satisfied the rule and pushed the review table (the whole point
            # of that screen) into "2 earlier note(s) hidden".
            said_something = said_something or not _is_blank(entry)
        kept.reverse()
        # spacing that fell off the top isn't a note the reader lost
        dropped = sum(1 for e in self._screen_notes[:len(self._screen_notes)
                                                     - len(kept)]
                      if not _is_blank(e))
        if dropped:
            super().note(f"  … {dropped} earlier note(s) hidden to make room",
                         style="dim")

        for kind, payload in kept:
            if kind == "note":
                text, style, wrap = payload
                # we go through `note` rather than straight at the console.
                # painting it raw skipped the margin handling and the row
                # counting -- and since every question repaints, the wrapped
                # indent was only ever right the *first* time a line showed up.
                lines = (note_lines(text, self.note_width()) if wrap
                         else str(text).split("\n"))
                room = budget - (self._rows_painted - self._header_rows())
                if len(lines) > max(1, room):
                    # one note taller than the whole budget: we show its head
                    # (the informative end of "file.csv: 151 columns — a, b, …")
                    # and say what got cut, rather than silently stopping
                    # mid-list.
                    keep = max(1, room - 1)
                    for line in lines[:keep]:
                        self._console.print(line, style=style or None,
                                            soft_wrap=True)
                        self._rows_painted += 1
                    self._blank_last = False
                    super().note(f"  … and {len(lines) - keep} more line(s)",
                                 style="dim")
                    continue
                super().note(text, style=style, wrap=wrap)
            else:
                title, rows, headers = payload
                room = budget - (self._rows_painted - self._header_rows())
                height = self._entry_height((kind, payload))
                if height > max(1, room):
                    # a table taller than the screen allows shows its head and
                    # says how much is missing, same as a long note -- rather
                    # than vanishing whole. that's what happened to the
                    # eleven-step review table on a short window once.
                    fits = max(1, room - _TABLE_CHROME - 1)
                    super().table(title, rows[:fits], headers)
                    super().note(f"  … and {len(rows) - fits} more row(s) -- "
                                 f"a taller window shows them all",
                                 style="dim")
                    continue
                super().table(title, rows, headers)

    def _header_rows(self) -> int:
        """Rows the banner took, so the notes' own usage can be measured."""
        if not self._header:
            return 0
        header = self._header() if callable(self._header) else self._header
        return header.count("\n") + 1

    # -- progress ----------------------------------------------------------
    def stage(self, key: str, label: str, *, status: str = "active",
              detail: str = "") -> None:
        """
        Add or update one entry in the rail.

        Updating in place (rather than appending) is what lets the wizard mark
        a stage done without knowing whether it had announced it before.
        """
        for i, existing in enumerate(self._stages):
            if existing.key == key:
                existing.label = label
                existing.status = status
                if detail:
                    existing.detail = detail
                if status == "active":
                    # a rail has exactly one "you are here". backing up to an
                    # earlier stage used to leave the later ones marked active
                    # from the previous pass, so the rail showed two arrows and
                    # claimed you were in two places at once. anything after
                    # the active stage is, by definition, not done yet.
                    for later in self._stages[i + 1:]:
                        later.status = "todo"
                        later.detail = ""
                return
        self._stages.append(Stage(key=key, label=label, status=status, detail=detail))

    # -- rendering ---------------------------------------------------------
    def _rail_height(self) -> int:
        """Rows the rail occupies: the one line, and a blank under it."""
        return _RAIL_ROWS if self._stages else 0

    def _rail_lines(self) -> List[Any]:
        """
        The rail as prompt_toolkit formatted text: a list of (style, text).

        One line, marker and color per stage, because that is all the rail is
        for -- saying where you are. It used to be a stacked list carrying each
        stage's detail alongside it, which cost seven rows out of the screen
        and still could not fit a detail as long as five chosen features, so
        the one line that most needed the room was the one that got truncated.
        The details are all still on the screens that own them.
        """
        if not self._stages:
            return []

        out: List[Any] = [("", "  ")]
        for i, stage in enumerate(self._stages):
            if i:
                out.append(("", "  "))
            marker, style = _MARKERS.get(stage.status, _MARKERS["todo"])
            out.append((style, f"{marker}{stage.label}"))
        out.append(("", "\n\n"))
        return out

    def reason(self, text: str) -> None:
        """
        Say why the next question is being asked -- next to the question.

        Held rather than printed. A note goes to the console above the rail, so
        the sentence explaining a question ended up separated from it by the
        whole rail and every other note on the screen, in a color that read as
        more commentary. This one is drawn inside the question's own layout,
        immediately above it, and in a color that is meant to be caught.
        """
        self._reason = text

    def _reason_lines(self) -> List[str]:
        """The reason, wrapped, or nothing."""
        if not self._reason:
            return []
        import textwrap

        width = max(min(self._console.width, MEASURE), 20)
        return textwrap.wrap(f"{_REASON_MARK} {self._reason}", width=width - 2,
                             initial_indent="  ", subsequent_indent="    ",
                             # same as in `note`: a URL or a shell command in
                             # here can't be broken across lines.
                             break_on_hyphens=False, break_long_words=False)

    def reset_stages(self) -> None:
        """
        Forget the rail.

        A rail belongs to the task that raised it. Nothing cleared it, so after
        the wizard finished -- or was backed out of -- its stages stayed on
        screen above the main menu, describing a pipeline that was no longer
        being built.
        """
        self._stages = []

    def _wrap(self, application, hint: str, breadcrumb=None) -> None:
        """Put the rail above the question's own layout, and a hint below it."""
        from prompt_toolkit.filters import is_done
        from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        rail = self._rail_lines()

        body = application.layout.container
        try:
            focused = application.layout.current_window
        except Exception:
            # no focusable window yet; wrapping would break focus resolution,
            # and a missing rail is a much smaller problem than a dead prompt.
            return

        # ~is_done is what makes this a *live region* instead of more output.
        # prompt_toolkit leaves its final frame in the scrollback (that's how
        # questionary's "? Question  answer" line survives), so without this
        # filter every answered question would leave a copy of the rail behind,
        # and eight questions would leave eight rails.
        # the rail is optional (the hub's menus have no stages); the reason
        # line and the key hint aren't. we used to bail out entirely when there
        # was no rail, which meant a `reason()` on a stage-less screen just
        # never rendered -- an explanation the user was owed, silently dropped.
        header = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: self._rail_lines()),
                height=_RAIL_ROWS if rail else 0,
                always_hide_cursor=True,
            ),
            filter=~is_done,
        )
        footer = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: [("fg:#808080 italic", f"  {hint}")]),
                height=1,
                always_hide_cursor=True,
            ),
            filter=~is_done,
        )

        # this goes between the rail and the question, so that the sentence
        # explaining why a question is on screen sits right against the
        # question and not above everything else that happens to be up there.
        reason_rows = len(self._reason_lines())
        middle = [header]
        if reason_rows:
            middle.append(ConditionalContainer(
                Window(
                    FormattedTextControl(
                        lambda: [(_REASON_STYLE, "\n".join(self._reason_lines())),
                                 ("", "\n")]),
                    height=reason_rows + 1,
                    always_hide_cursor=True,
                ),
                filter=~is_done,
            ))
        if breadcrumb is not None:
            # "you are here", re-read on every paint -- that's what lets
            # in-app navigation move the path without rebuilding the prompt.
            middle.append(ConditionalContainer(
                Window(
                    FormattedTextControl(
                        lambda: [("fg:#808080", f"  {breadcrumb()}")]),
                    height=1,
                    always_hide_cursor=True,
                ),
                filter=~is_done,
            ))
        application.layout = Layout(
            HSplit([*middle, body, footer]),
            focused_element=focused,
        )

    # -- input -------------------------------------------------------------
    _BACK = object()          # the sentinel our Esc binding exits the app with

    def _bind_escape(self, application) -> None:
        """
        Make Esc mean "undo that last answer".

        prompt_toolkit reserves Escape as the prefix of every escape sequence,
        so a bare binding waits to see whether an arrow key is arriving. `eager`
        says do not wait -- without it, Esc appears to do nothing until the next
        keypress, which reads as a broken key rather than a deliberate one.
        """
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()

        @bindings.add("escape", eager=True)
        def _(event):
            event.app.exit(result=self._BACK)

        existing = application.key_bindings
        if existing is None:
            application.key_bindings = bindings
        else:
            from prompt_toolkit.key_binding.key_bindings import merge_key_bindings
            application.key_bindings = merge_key_bindings([existing, bindings])

    def _ask(self, question_obj, hint: str = _DEFAULT_HINT,
             transient: bool = False, breadcrumb=None) -> Any:
        # a transient question redraws over the one before it instead of wiping
        # the screen and starting over. stepping through folders is one act of
        # navigation, and clearing the terminal on every keypress makes it
        # flicker like something restarted each time.
        if not transient:
            self._paint()
        # a list gets whatever the screen has left. everything above it was
        # just painted and counted; below it sit the question itself and the
        # key hint. before this, we sized the list against a fixed guess of 16
        # rows, so a step with long option descriptions grew a window taller
        # than the terminal and scrolled the banner off the top.
        set_chrome_rows(self._rows_painted + self._rail_height()
                        + len(self._reason_lines())
                        + (1 if breadcrumb is not None else 0) + _BELOW_LIST)
        # held at the tallest description on this screen, so that moving
        # between rows with long and short (or no) help never changes the
        # list's height -- that's what used to scroll the explanation above
        # the question up the screen a line or two per keypress.
        set_description_rows(getattr(self, "_desc_rows", 0))
        try:
            self._wrap(question_obj.application, hint, breadcrumb)
            self._bind_escape(question_obj.application)
            if transient:
                # prompt_toolkit keeps its last frame on screen so questionary
                # can leave a "? Question  answer" record behind. for a step of
                # navigation there's no answer worth recording, and keeping
                # them would rebuild the very log we're trying to avoid.
                question_obj.application.erase_when_done = True
        except Exception:
            # the rail and the Esc binding are both niceties. if some
            # questionary version reshapes its layout in a way we don't expect,
            # we ask the question anyway rather than taking the whole wizard
            # down over cosmetics.
            pass

        answer = question_obj.ask()

        # we clear however the question ended, not only when it was answered.
        # the notes were about *this* question; backing out of it retires them
        # just like answering does. when we only cleared on the way to a return
        # value, an Esc left its explanation on screen, and a few of those in a
        # row pushed the next question off the bottom.
        self._screen_notes.clear()
        self._reason = ""

        if answer is self._BACK:
            raise GoBack()
        if answer is None:
            raise Cancelled()
        return answer

    def select(self, question: str, choices: Sequence[Choice],
               *, default: Optional[str] = None, transient: bool = False,
               numbered: bool = True,
               toggle_values: Sequence[str] = (),
               ticked: Optional[set] = None,
               navigate=None, breadcrumb=None,
               hint_override: Optional[str] = None,
               enter_gate: Optional[str] = None,
               cycle=None) -> str:
        """
        Pick one option.

        Numbered so a menu can be answered with a single digit as well as with
        the arrow keys -- and the digit both selects *and* confirms, where
        questionary's own shortcuts only move the cursor and still want enter.

        A digit can only ever address nine things, so a list longer than that
        is not numbered at all. Numbering the first nine and leaving the rest
        bare -- what this used to do -- reads as a list that lost its numbers
        half way down, and a label reading "10." would advertise a key that
        does nothing. Either every row carries a number or none does.
        """
        shown = list(choices)
        # a navigating select swaps its rows out from under any digit
        # bindings, leaving keys wired to values that aren't on screen anymore
        # -- so we never number one.
        numbered = numbered and len(shown) <= 9 and navigate is None
        if numbered:
            shown = [replace(c, label=f"{i}. {c.label}")
                     for i, c in enumerate(shown, 1)]

        self._desc_rows = description_rows(shown)
        question_obj = self._q.select(
            question,
            choices=[self._to_q(c) for c in shown],
            default=default,
            style=self._style,
            use_shortcuts=False,
            # the hint bar below already names the keys; questionary's own
            # "(Use arrow keys)" just pushes the question past the right edge.
            # it tests the instruction for truthiness, so "" falls back to the
            # default -- a single space is what actually shuts it up.
            instruction=" ",
        )
        if numbered:
            self._bind_number_keys(question_obj.application, choices)
        # one shared, mutable set: navigation swaps its contents when the rows
        # change, and the space binding reads it live -- so a folder we enter
        # mid-walk brings its files' tickability along with it.
        toggles = set(toggle_values)
        if ticked is not None or toggles:
            self._bind_space_toggle(question_obj.application, toggles, ticked)
        if navigate is not None:
            self._bind_navigation(question_obj.application, navigate, toggles)
        if enter_gate is not None:
            self._bind_enter_gate(question_obj.application, enter_gate,
                                  ticked if ticked is not None else set())
            # the gate row is the way out of the screen, and it lives at the
            # top -- we pin it there so a long list scrolling underneath can
            # never hide it (someone asked for this after fighting a
            # forty-column spreadsheet).
            self._pin_top_rows(question_obj.application, 1)
        if cycle is not None:
            self._bind_cycle(question_obj.application, cycle)
        if hint_override is not None:
            hint = hint_override
        elif ticked is not None or toggles:
            hint = _KEY_HINTS["tick"]
        else:
            hint = _KEY_HINTS["select"] if numbered else _KEY_HINTS["select_long"]
        if cycle is not None:
            hint = hint.replace(" · [esc] back", " · [←→] change type · [esc] back")
        # a list is coming, so the painter above owes it room. only as much as
        # there is to show, though: a three-option menu has no use for twelve
        # rows, and claiming them would trim notes for nothing.
        self._reserve_rows = min(_LIST_FLOOR, len(shown))
        try:
            return self._ask(question_obj, hint, transient=transient,
                             breadcrumb=breadcrumb)
        finally:
            self._reserve_rows = 0

    def _bind_cycle(self, application, cycle) -> None:
        """
        Left/right change how the pointed row is treated, in place.

        The wizard shows what kind of thing each column holds beside its
        name -- numbers, labels, text -- because that decides whether it
        can be an outcome, a group, a categorical control. A 1/2 gender
        code reads as numbers and is not; this is where the person says
        so, on the row itself, without leaving the list. `cycle` owns the
        rule (what a column may become) and returns the new annotation;
        the row is repainted with it and nothing else moves.
        """
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_bindings import merge_key_bindings

        bindings = KeyBindings()

        def turn(event, direction: int):
            control = _inquirer_control(application)
            if control is None:
                return
            pointed = control.get_pointed_at()
            if pointed is None:
                return
            new = cycle(str(pointed.value), direction)
            if new is None:
                return
            pointed.title = with_annotation(pointed.title, new)
            event.app.invalidate()

        @bindings.add("right", eager=True)
        def _(event):
            turn(event, 1)

        @bindings.add("left", eager=True)
        def _(event):
            turn(event, -1)

        existing = application.key_bindings
        application.key_bindings = (
            bindings if existing is None
            else merge_key_bindings([existing, bindings]))

    def _bind_space_toggle(self, application, toggle_values: set,
                           ticked: Optional[set]) -> None:
        """
        Space ticks the pointed row, in place -- no exit, no redraw of the
        world, no flash.

        The first version exited the prompt with the row's value and let the
        caller rebuild the menu, which wiped the screen on every single tick
        and made space indistinguishable from enter. Now the binding flips the
        caller's `ticked` set directly, rewrites the pointed row's own title
        ("[ ]" <-> "[x]"), and invalidates the app so prompt_toolkit repaints
        just its region. Enter is thereby freed to mean what fingers expect:
        choose the action, open the folder, proceed.

        Space on anything that is not tickable does nothing at all, so idly
        pressing it on "Delete the ticked files" cannot fire the action.
        """
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_bindings import merge_key_bindings

        def _control():
            return _inquirer_control(application)

        bindings = KeyBindings()

        @bindings.add(" ", eager=True)
        def _(event):
            control = _control()
            if control is None or ticked is None:
                return
            pointed = control.get_pointed_at()
            if pointed is None or pointed.value not in toggle_values:
                return
            ticked.symmetric_difference_update({pointed.value})
            pointed.title = flip_tick_mark(pointed.title)
            event.app.invalidate()

        existing = application.key_bindings
        application.key_bindings = (
            bindings if existing is None
            else merge_key_bindings([existing, bindings])
        )

    def _bind_navigation(self, application, navigate, toggles: set) -> None:
        """
        Walk folders inside the running prompt -- no exit, no rebuild, no flash.

        Navigation used to end the application and build a new one per folder,
        and the teardown-then-relist gap (worst on a slow mount, where counting
        a folder's files takes real time) read as the whole screen flashing.
        Now enter is intercepted: the caller's `navigate` looks at the pointed
        value and either returns a replacement screen -- rows swapped in place,
        the same move space uses to flip one tick mark -- or None, meaning this
        is a real answer, and the app exits with it exactly as questionary's
        own handler would. While `navigate` computes the new listing, the old
        one stays on screen, so a slow folder shows as a pause, not a blank.
        """
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_bindings import merge_key_bindings

        def _control():
            return _inquirer_control(application)

        bindings = KeyBindings()

        @bindings.add("enter", eager=True)
        def _(event):
            control = _control()
            pointed = control.get_pointed_at() if control is not None else None
            if pointed is None:
                return
            swap = navigate(str(pointed.value))
            if swap is None:
                event.app.exit(result=pointed.value)
                return
            new = [self._to_q(c) for c in swap["choices"]]
            pointed_at = None
            wanted = swap.get("default")
            if wanted is not None:
                pointed_at = next(
                    (i for i, c in enumerate(swap["choices"])
                     if c.value == wanted and not c.disabled), None)
            # `default` steered questionary's initial selection; if we left it
            # in place it'd re-tick a row of the old screen by value coincidence.
            control.default = None
            control.selected_options = []
            control._init_choices(new, pointed_at)
            toggles.clear()
            toggles.update(swap.get("toggle_values", ()))
            event.app.invalidate()

        existing = application.key_bindings
        application.key_bindings = (
            bindings if existing is None
            else merge_key_bindings([existing, bindings])
        )

    def _bind_number_keys(self, application,
                          choices: Sequence[Choice]) -> None:
        """
        Answer with a digit: pick that option and finish, in one keypress.

        A disabled row keeps its number for alignment but gets no binding.
        These bindings exit the prompt directly with the choice's value,
        sidestepping questionary's own disabled gate -- so binding every row
        made one keypress select a grayed-out action, and the handler behind
        it then crashed the whole session on the empty state the graying-out
        existed to protect.
        """
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_bindings import merge_key_bindings

        bindings = KeyBindings()

        def _choose(value: str):
            def handler(event):
                event.app.exit(result=value)
            return handler

        for index, choice in enumerate(choices[:9], 1):
            if not choice.disabled:
                bindings.add(str(index))(_choose(choice.value))

        existing = application.key_bindings
        application.key_bindings = (
            bindings if existing is None else merge_key_bindings([existing, bindings])
        )

    #: The tick screens' explicit way forward; \x00 so it can never collide
    #: with a caller's value.
    _TICKS_DONE = "\x00__ticks_done__"

    def checkbox(self, question: str, choices: Sequence[Choice], *,
                 cycle=None) -> List[str]:
        """
        Multi-selection on ONE screen, in the one dialect every tick screen
        uses: rows wear ``[x]`` boxes, [space] toggles a box in place, and
        [enter] does exactly one thing -- confirm, and only while pointing at
        the "✓ Done" row with at least one box ticked. Everywhere else,
        enter is a no-op: the screen does not move, does not flash, and can
        never carry a half-made selection forward.

        This is the third iteration of this screen, each driven by a real
        user report. Stock questionary's enter proceeded with the empty set
        (dropping the row someone was pointing at); a reloop-per-toggle
        design fixed that but tore the prompt down on every enter, and the
        redraw read as "the screen changed" when it had not. The fix is a
        key-binding gate inside a SINGLE prompt: nothing is rebuilt, space
        flips marks in place, and enter is inert except on Done.
        """
        ticked = {c.value for c in choices if c.checked}
        order = [c.value for c in choices]
        rows = [Choice(self._TICKS_DONE, "✓ Done — use the ticked items",
                       help="Tick boxes with [space]; this row's [enter] "
                            "confirms them. It waits until something is "
                            "ticked.",
                       tone="good")]
        for c in choices:
            mark = "x" if c.value in ticked else " "
            rows.append(replace(c, label=f"[{mark}] {c.label}"))
        self.select(question, rows, numbered=False,
                    toggle_values=set(order), ticked=ticked,
                    enter_gate=self._TICKS_DONE,
                    hint_override=(_KEY_HINTS["checkbox"] if cycle is None else
                                   _KEY_HINTS["checkbox"].replace(
                                       " · [esc] back",
                                       " · [←→] change type · [esc] back")),
                    cycle=cycle)
        return [v for v in order if v in ticked]

    @staticmethod
    def _pin_top_rows(application, count: int) -> None:
        """Mark the list's first `count` rows as always-visible; the scrolling
        renderer (:func:`~taters.ui.prompts.scroll_long_lists`) reads the mark
        and windows only the rows beneath them."""
        from questionary.prompts.common import InquirerControl

        try:
            for window in application.layout.find_all_windows():
                if isinstance(window.content, InquirerControl):
                    window.content._taters_sticky_rows = count
        except Exception:      # pragma: no cover - a reshaped questionary
            pass

    def _bind_enter_gate(self, application, gate_value: str,
                         ticked: set) -> None:
        """
        Enter exits the prompt ONLY on the gate row, only with ticks.

        Anywhere else -- an entry row, or Done with nothing ticked -- enter
        does nothing at all. A visible non-event is the honest signal here:
        the alternatives either smuggled an answer forward or rebuilt the
        screen, and both were misread by real users.
        """
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_bindings import merge_key_bindings

        def _control():
            return _inquirer_control(application)

        bindings = KeyBindings()

        @bindings.add("enter", eager=True)
        def _(event):
            control = _control()
            pointed = control.get_pointed_at() if control is not None else None
            if pointed is not None and pointed.value == gate_value and ticked:
                event.app.exit(result=gate_value)
                return
            # anywhere else, enter takes you to the Done row -- visibly, as a
            # quick glide rather than a jump -- so that a tick made at row 480
            # of 500 doesn't end with 480 presses of the up arrow (someone
            # actually hit this). not an answer, not a redraw of the world: the
            # pointer moves, the window follows it, the screen stays put.
            if control is not None and pointed is not None:
                target = next((i for i, c in enumerate(control.choices)
                               if c.value == gate_value), None)
                if target is not None:
                    _glide(event.app, control, target)

        existing = application.key_bindings
        application.key_bindings = (
            bindings if existing is None
            else merge_key_bindings([existing, bindings])
        )

    def pause(self, message: str = PAUSE_MESSAGE) -> None:
        """
        Repaint the screen, then wait to be dismissed.

        The paint is the whole reason for the override. Every other prompt gets
        it from `_ask`, which `pause` deliberately skips -- so without this the
        reader is left waiting at a screen the last `clear()` wiped.

        Clearing the held notes afterwards matters just as much, and for the
        same reason `_ask` does it: they have been read now. Without it a whole
        setup report was redrawn on the *next* screen too, and the menu that
        followed appeared underneath it, near the bottom of the terminal.
        """
        self._paint()
        self._wait_for_key(message)
        self._screen_notes.clear()

    def confirm(self, question: str, *, default: bool = True) -> bool:
        """
        Yes or no, as a list you move through rather than a word you type.

        questionary's own confirm renders as `(Y/n)` and waits on a text
        buffer. A single `y` does answer it, but nothing on screen says so, so
        it reads as "type a word and press enter" -- which is the one
        interaction in the whole wizard that works differently from the rest.

        This is an ordinary two-item selection, so arrow keys and enter behave
        exactly as they do everywhere else, with `y`/`n` and `1`/`0` bound as
        shortcuts for anyone who would rather not move at all.
        """
        # the default row SAYS it's the default. the old signal (it came first
        # and started highlighted) was a design convention nobody was told
        # about, and users read the highlight as anything from "already
        # chosen" to "the best answer".
        yes = Choice("yes", "Yes", annotation="(default)" if default else "")
        no = Choice("no", "No", annotation="" if default else "(default)")
        ordered = [yes, no] if default else [no, yes]

        self._desc_rows = description_rows(ordered)
        question_obj = self._q.select(
            question,
            choices=[self._to_q(c) for c in ordered],
            default="yes" if default else "no",
            style=self._style,
            use_shortcuts=False,
            instruction=" ",
        )
        self._bind_confirm_keys(question_obj.application)
        return self._ask(question_obj, _KEY_HINTS["confirm"]) == "yes"

    def _bind_confirm_keys(self, application) -> None:
        """Answer a yes/no on a single keypress, without moving or pressing enter."""
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.key_bindings import merge_key_bindings

        bindings = KeyBindings()

        def _answer(value: str):
            def handler(event):
                event.app.exit(result=value)
            return handler

        # "1"/"0" as well as "y"/"n": for the numeric keypad, and for anyone
        # whose keyboard layout doesn't put y and n where English expects them.
        for key in ("y", "Y", "1"):
            bindings.add(key)(_answer("yes"))
        for key in ("n", "N", "0"):
            bindings.add(key)(_answer("no"))

        existing = application.key_bindings
        application.key_bindings = (
            bindings if existing is None else merge_key_bindings([existing, bindings])
        )
