"""
Live progress for a running pipeline.

The wizard used to print a line per step and then go quiet. For a single global
step -- a whole spreadsheet scored in one call -- that meant a cursor sitting on
an unchanging line for minutes, which is indistinguishable from a hang. The
first thing a user does then is press Ctrl-C, which is the one thing guaranteed
to waste the work.

So there are two bars, stacked:

* **Overall** -- steps finished out of steps planned. Always meaningful.
* **Current step** -- files finished out of files found, when the step fans out
  over inputs. A GLOBAL step is one call and cannot report its own internal
  progress, so it gets an elapsed-time spinner instead: no false precision, but
  visible proof of life.

`run_preset` already emits everything needed through its ``on_event`` callback,
so nothing in the runner changes to support this.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.progress import ProgressColumn
from rich.markup import escape

__all__ = ["RunDisplay", "reporter_for"]

# how many files we name at once. the number of rows is really bounded by the
# worker count, not the file count (four workers = four rows, however many
# files), but somebody running 32 workers would still get pushed down the
# screen. four is enough to see work turning over, and we keep the *oldest*
# ones still running, since that's where a stuck file shows up.
_MAX_SLOTS = 4

# roughly how much of a row everything *but* the filename eats: the `  ↳ `
# prefix, spinner, bar, count, elapsed clock, and the spaces between. we
# subtract this from the console width to figure out how much room a filename
# gets. otherwise a row of long names shoves the bar off the edge and we're
# left with four identical `━━━━ 0:0…` stubs and no way to tell files apart.
_ROW_OVERHEAD = 58


def _take(text: str, budget: int, *, from_end: bool = False) -> str:
    """As much of `text` as fits in `budget` terminal cells, from one end."""
    from rich.cells import cell_len

    used, kept = 0, []
    for ch in (reversed(text) if from_end else text):
        width = cell_len(ch)
        if used + width > budget:
            break
        used += width
        kept.append(ch)
    return "".join(reversed(kept)) if from_end else "".join(kept)


def _fit(text: str, budget: int) -> str:
    """
    Shorten a filename from the middle, keeping both ends.

    The ends are what tell two files apart: real corpora are full of names that
    share a long prefix -- a series, a date, a session -- and differ only in a
    trailing id, and just as many that differ only at the start. Cutting from
    the middle keeps whichever it is.

    Measured in terminal cells rather than characters, because the names that
    make this necessary are the ones most likely to need it. A title downloaded
    from the web arrives with `：` and other fullwidth punctuation standing in
    for characters a filesystem will not take, and each of those occupies two
    columns while counting as one character. Sizing by `len` therefore
    overshoots exactly on the rows that were already too long.
    """
    from rich.cells import cell_len

    if cell_len(text) <= budget:
        return text
    if budget < 4:
        return _take(text, budget)
    head = (budget - 1) // 2
    return f"{_take(text, head)}…{_take(text, budget - 1 - head, from_end=True)}"


def _drop_unbalanced_parens(text: str) -> str:
    """
    Remove parentheses whose partner was eaten by a middle cut.

    ``_fit`` keeps both ends of a string, which is right for filenames but
    leaves prose like "sorting rows into groups (pass 1 of 2)" reading
    "sorting row…pass 1 of 2)" -- a dangling parenthesis that looks like a
    rendering bug (a real report). Dropping the orphans reads cleanly either
    way; the words carry the meaning.
    """
    orphans: set = set()
    stack: list = []
    for i, ch in enumerate(text):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                stack.pop()
            else:
                orphans.add(i)
    orphans.update(stack)
    if not orphans:
        return text
    return "".join(ch for i, ch in enumerate(text) if i not in orphans)


class RunDisplay:
    """
    A ``rich`` progress display driven by ``run_preset`` events.

    Used as a context manager. Failures are collected rather than printed as
    they happen: writing into the area a live display owns corrupts it, and a
    per-file error is better read at the end anyway, next to the count.
    """

    def __init__(self, console: Any = None) -> None:
        from rich.console import Console

        self._console = console or Console()
        self._progress = None
        self._overall = None
        self._step = None
        self._label = ""
        self._slots: Dict[int, Any] = {}      # item -> visible task
        self._inflight: Dict[int, str] = {}   # item -> filename, visible or not
        # item -> (done, total, message, unit), for files that report their own
        # position. we keep this even for files with no bar on screen, because
        # a hidden file can get promoted into a slot at any moment, and a bar
        # that shows up at zero and then jumps is worse than one that shows up
        # where the work actually is.
        self._itemprog: Dict[int, tuple] = {}
        self._overflow = None
        self.failures: List[str] = []
        # warnings raised while the display is up, deduped. pandas and
        # sentence-transformers both like to warn during a perfectly normal
        # run, and warnings go to stderr rather than through `verbose`, so
        # quieting the steps doesn't quiet these. we do want to show them --
        # just not on top of a live progress bar, and not three times.
        self.notices: List[str] = []
        self._showwarning = None
        # `item_start` fires from inside the worker thread (that's the whole
        # point of it) while every other event comes in on the main thread. so
        # the bookkeeping below runs concurrently, and its check-then-act
        # ("not in _slots", then "add_task") is exactly the kind of thing that
        # loses a race: two workers both see a file unslotted, both add a bar,
        # and one of them gets orphaned. an orphaned bar isn't in any of our
        # dicts, so nothing ever removes it and it survives every step
        # boundary. that's how a four-file run ended up showing the same
        # filename three times under a step that finished minutes earlier.
        # hence the lock.
        self._lock = threading.RLock()

    def __enter__(self) -> "RunDisplay":
        import os

        from rich.progress import (
            BarColumn,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        # while the live display owns the terminal, nobody else gets to draw on
        # it. model downloads (stanza, sentence-transformers, whisper) go
        # through huggingface-hub, whose tqdm bars write raw to stderr and
        # sliced right through our bars the first time we tried this. this env
        # var is the hub's own off switch; we turn it back on when we exit.
        self._hf_bars_before = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        self._progress = _RuledProgress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            _CountColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=False,
        )
        self._progress.start()
        self._capture_warnings()
        return self

    def __exit__(self, *exc) -> None:
        import os

        if getattr(self, "_hf_bars_before", None) is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = self._hf_bars_before
        self._release_warnings()
        if self._progress is not None:
            self._progress.stop()
        # we hold these back until the display has stopped and we've got the
        # cursor back.
        for notice in self.notices:
            # escape it: this is captured warning text, and anything in square
            # brackets looks like markup to rich -- silently swallowed at best,
            # a MarkupError inside __exit__ at worst, which would crash the
            # display while unwinding after a possibly hours-long run. no
            # thanks.
            self._console.print(f"[dim]note: {escape(str(notice))}[/dim]")
        return None

    def _capture_warnings(self) -> None:
        """
        Divert `warnings` output into :attr:`notices` for the duration.

        `warnings.showwarning` rather than a filter, because a filter would
        decide *whether* the warning happens and this only decides where it is
        written. Worker threads are covered for free: the hook is global.
        """
        import warnings

        self._showwarning = warnings.showwarning

        def _collect(message, category, filename, lineno, file=None, line=None):
            text = f"{getattr(category, '__name__', category)}: {message}"
            with self._lock:
                if text not in self.notices:
                    self.notices.append(text)

        warnings.showwarning = _collect

    def _release_warnings(self) -> None:
        import warnings

        if self._showwarning is not None:
            warnings.showwarning = self._showwarning
            self._showwarning = None

    def _fill_slots(self) -> None:
        """
        Show the oldest in-flight files, and count the rest.

        Oldest rather than newest on purpose: a file that has been running far
        longer than its neighbors is the one worth seeing, and it is the one a
        newest-first display would hide.
        """
        for item in list(self._inflight):
            if len(self._slots) >= _MAX_SLOTS:
                break
            if item not in self._slots:
                self._slots[item] = self._progress.add_task(
                    self._row_label(item), total=None
                )
                if item in self._itemprog:
                    self._apply_item_progress(item, *self._itemprog[item])

        hidden = len(self._inflight) - len(self._slots)
        if hidden > 0:
            label = f"  ↳ …and {hidden} more file{'' if hidden == 1 else 's'}"
            if self._overflow is None:
                self._overflow = self._progress.add_task(label, total=None)
            else:
                self._progress.update(self._overflow, description=label)
        elif self._overflow is not None:
            self._progress.remove_task(self._overflow)
            self._overflow = None

    def _row_label(self, item: int, message: Any = None) -> str:
        """
        The text on one file's row: which file, and what is happening to it.

        Sized against the terminal, because these names are long. A row that
        overflows costs the bar and the clock -- the two things that show the
        work is moving -- to keep characters of a filename that was already
        legible at half the length.
        """
        name = self._inflight.get(item, "?")
        budget = max(16, self._console.width - _ROW_OVERHEAD)
        if message:
            from rich.cells import cell_len

            suffix = f" · {message}"
            return f"  ↳ {_fit(name, max(8, budget - cell_len(suffix)))}{suffix}"
        return f"  ↳ {_fit(name, budget)}"

    #: How many in-flight sub-bars to draw before folding the rest into one
    #: "…and N more" row. The set is bounded by 2x the worker count, but on a
    #: big machine that is still sixteen rows of terminal.
    _FLIGHT_ROWS = 8

    def _reconcile_flight(self, names: list) -> None:
        """
        Make the sub-bars match a parallel phase's in-flight set, exactly.

        Called from the runner's progress sink -- which a worker-pool's
        done-callbacks may invoke from executor threads -- so it takes the
        same lock the per-item bars do.
        """
        with self._lock:
            shown = names[: self._FLIGHT_ROWS]
            extra = len(names) - len(shown)
            want = {f"f:{name}" for name in shown}
            if extra > 0:
                want.add("f:+")
            stale = [key for key in self._slots
                     if isinstance(key, str) and key.startswith("f:")
                     and key not in want]
            for key in stale:
                try:
                    self._progress.remove_task(self._slots.pop(key))
                except KeyError:        # pragma: no cover - already gone
                    pass
            budget = max(16, self._console.width - _ROW_OVERHEAD)
            for name in shown:
                key = f"f:{name}"
                if key not in self._slots:
                    self._slots[key] = self._progress.add_task(
                        f"  ↳ {_fit(name, budget)}", total=None)
            if extra > 0:
                label = f"  ↳ …and {extra} more"
                if "f:+" not in self._slots:
                    self._slots["f:+"] = self._progress.add_task(label, total=None)
                else:
                    self._progress.update(self._slots["f:+"], description=label)

    def _clear_slots(self) -> None:
        """
        Drop the per-file bars. A step boundary makes all of them stale.

        Everything except the two bars that outlive a step goes, rather than
        only the bars this object still has a handle on. Removing by hand from
        `_slots` would leave anything already dropped from it on screen for the
        rest of the run, and a stale bar is worse than a missing one: it names
        a file that is not being worked on.
        """
        keep = {self._overall, self._step}
        for task in list(self._progress.task_ids):
            if task in keep:
                continue
            try:
                self._progress.remove_task(task)
            except KeyError:
                pass
        self._slots.clear()
        self._inflight.clear()
        self._itemprog.clear()
        self._overflow = None

    # -- events ------------------------------------------------------------
    def handle(self, name: str, **payload) -> None:
        if self._progress is None:
            return
        with self._lock:
            self._handle(name, **payload)

    def _handle(self, name: str, **payload) -> None:
        """The body of :meth:`handle`, with the lock already held."""
        if name == "run_start":
            self._overall = self._progress.add_task(
                "Overall", total=max(1, int(payload.get("steps", 1)))
            )

        elif name == "step_start":
            label = _short_call(payload.get("call", "step"))
            if payload.get("for_model"):
                # same call, different settings: say whose.
                label = f"{label} (for {payload['for_model']})"
            self._label = label
            items = payload.get("items")
            fans_out = payload.get("scope") == "item" and items

            self._clear_slots()
            if self._step is not None:
                self._progress.remove_task(self._step)
            self._step = self._progress.add_task(
                label,
                # a GLOBAL step is one call and has no way to tell us how far
                # along it is. `total=None` gets rich to show a pulsing bar
                # instead of one frozen at 0%, which just looks stuck.
                total=int(items) if fans_out else None,
            )

        elif name == "step_progress":
            # a GLOBAL step that counts itself. we only find out the total once
            # the step has read its input, so the task starts out indeterminate
            # and we upgrade it here instead of creating it with a total.
            if self._step is not None:
                total = payload.get("total")
                fields = {
                    "completed": int(payload.get("done", 0)),
                    "total": int(total) if total else None,
                    "unit": payload.get("unit"),
                }
                # say which phase we're in. until we know the total there's
                # nothing else on the line to explain why a big file is just
                # sitting there apparently doing nothing.
                message = payload.get("message")
                if message:
                    # fit it like the per-file rows. the message now carries
                    # document names ("reading <file>.pdf"), and a long one
                    # widened this column past the terminal, pushing the bar
                    # and the clock (our proof the run is moving) clean off the
                    # row. the floor of 40 keeps short phase messages ("reading
                    # the input") intact even on a narrow terminal, where the
                    # row overflowed before this existed anyway.
                    budget = max(40, self._console.width - _ROW_OVERHEAD)
                    from rich.cells import cell_len

                    prefix = f"{self._label} · "
                    fitted = _fit(str(message), max(18, budget - cell_len(prefix)))
                    if fitted != str(message):
                        fitted = _drop_unbalanced_parens(fitted)
                    fields["description"] = prefix + fitted
                elif total:
                    fields["description"] = self._label
                self._progress.update(self._step, **fields)
                # a step can run in phases over the same bar (count the rows,
                # then score them), and each phase ends at N/N. rich stamps a
                # finished_time the first time completed reaches total, and the
                # elapsed clock freezes there for good -- so we had the step's
                # timer standing still through whole phases while the sub-bars
                # ticked away. if we're still getting step_progress the step
                # isn't finished, whatever the counter says, so we unfreeze it.
                for task in self._progress.tasks:
                    if task.id == self._step and task.finished_time is not None:
                        task.finished_time = None
                        task.finished_speed = None

                inflight = payload.get("inflight")
                if inflight is not None:
                    # one sub-bar per document a parallel phase is chewing on
                    # right now -- same look as the per-file rows an ITEM step
                    # gets for ffmpeg or Whisper.
                    self._reconcile_flight(list(inflight))

        elif name == "item_start":
            # one bar per file actually in flight. the runner fans an ITEM step
            # out across a pool, so without this all you'd see is "3 of 40
            # done" and no sign of which files are moving. on a long
            # transcription that's the difference between "working" and "hung".
            item = payload.get("item")
            if item is not None and item not in self._inflight:
                # the stem, not the filename. the event carries the file the
                # *pipeline* was handed, which isn't the file the step in front
                # of you is actually reading: by the time Whisper runs, the item
                # is a `.wav` that `convert_to_wav` made, so a row saying
                # "transcribe with whisper ↳ lecture.mkv" is just untrue.
                # dropping the extension keeps the part that answers the only
                # question the row is for (which of my files is this?), and
                # that part stays true for every step in the run.
                self._inflight[item] = Path(str(payload.get("input", "?"))).stem
                self._fill_slots()

        elif name == "item_progress":
            # how far into *one file* a worker has gotten. `item_start` tells us
            # which files are in flight; without this the row for a 20-minute
            # transcription just sits there naming a file and nothing else,
            # which is exactly what it does when the step has died.
            item = payload.get("item")
            if item is None:
                return
            state = (payload.get("done"), payload.get("total"),
                     payload.get("message"), payload.get("unit"))
            self._itemprog[item] = state
            if item in self._slots:
                self._apply_item_progress(item, *state)

        elif name == "item_done":
            item = payload.get("item")
            self._inflight.pop(item, None)
            self._itemprog.pop(item, None)
            task = self._slots.pop(item, None)
            if task is not None:
                try:
                    self._progress.remove_task(task)
                except KeyError:
                    # losing a bar can't be allowed to cost us the count. `emit`
                    # swallows whatever this method raises, so an exception
                    # here would skip the `advance` below and freeze the step's
                    # counter -- the exact symptom this display exists to rule
                    # out.
                    pass
            self._fill_slots()
            if self._step is not None:
                self._progress.advance(self._step)
            if payload.get("status") != "ok":
                failure = (f"{Path(str(payload.get('input', '?'))).name} — "
                           f"{payload.get('error')}")
                self.failures.append(failure)
                # say it now, not just in the summary at the end. a run that
                # stalls after one file fails never gets to its summary, so the
                # one message explaining what went wrong is the one message you
                # never see -- and that's exactly where a user got stranded
                # once. `rich` renders this above the live region rather than
                # into it, so printing it doesn't mess anything up.
                self._console.print(f"  [red]x[/red] {escape(str(failure))}")

        elif name == "step_done":
            if self._overall is not None:
                self._progress.advance(self._overall)
            if payload.get("status") == "error":
                self.failures.append(
                    f"{_short_call(payload.get('call', 'step'))} — {payload.get('error')}"
                )

        elif name == "run_done":
            self._clear_slots()
            if self._step is not None:
                self._progress.remove_task(self._step)
                self._step = None


    def _apply_item_progress(self, item: int, done: Any, total: Any,
                             message: Any, unit: Any) -> None:
        """Push one file's reported position onto its bar."""
        task = self._slots.get(item)
        if task is None:
            return
        fields: Dict[str, Any] = {
            "completed": int(done or 0),
            # a phase with no size yet (loading a model, scanning for speech)
            # keeps the bar indeterminate. zero out of a known total looks like
            # stalled work; a pulsing bar looks like work with no denominator
            # yet, and that's exactly what's going on.
            "total": int(total) if total else None,
            "unit": str(unit) if unit else None,
        }
        # the phase name only earns its spot on the row while it's the only
        # thing we have to say. once the numbers are flowing they say it better.
        fields["description"] = self._row_label(item, message)
        self._progress.update(task, **fields)


class _RuledProgress:
    """
    A ``rich`` progress display with a line drawn above it.

    Without one, everything the run prints -- a step's own output, a warning, a
    subprocess traceback relayed by the parent -- ends up flush against the top
    bar, and the two read as one block of text. That is at its worst exactly
    when it matters: a stack trace whose last line abuts a spinner is hard to
    even find the end of.

    The rule makes the boundary explicit. Above it is what has happened; below
    it is what is happening now.

    Implemented by subclassing at runtime rather than at import, because
    ``rich`` is imported lazily throughout this module -- a plain
    ``class X(Progress)`` here would pull it in at module import and cost the
    CLI its startup time.
    """

    def __new__(cls, *columns, **kwargs):
        from rich.progress import Progress
        from rich.rule import Rule

        class Ruled(Progress):
            def get_renderables(self):
                # dim and unlabeled: it's just furniture, and a titled rule
                # would fight the bars for attention.
                yield Rule(style="grey30")
                yield from super().get_renderables()

        return Ruled(*columns, **kwargs)


class _CountColumn(ProgressColumn):
    """
    ``12/40`` when the total is known, and nothing when it is not.

    rich's own MofNCompleteColumn renders ``0/?`` for an indeterminate task,
    which looks like a bug rather than like "this step cannot count itself".
    The spinner and the elapsed clock already carry the liveness; a question
    mark only adds doubt.
    """

    def render(self, task):
        from rich.text import Text

        # some work gets counted in things ("row 4,000 of 90,000") and some in
        # time ("4:12 of 12:51"). a transcription shown as `252/771` invites
        # exactly the wrong reading (segments, not seconds), so a step that
        # counts something other than items says so.
        fmt = _clock if task.fields.get("unit") == "seconds" else _thousands

        if task.total is None:
            # still working out how much there is. we show the running tally
            # anyway because on a big file the number climbing is the only
            # proof that a long silent pass is doing anything at all.
            if task.completed:
                return Text(f"{fmt(task.completed)}…", style="progress.data.speed")
            return Text("", style="progress.data.speed")
        return Text(f"{fmt(task.completed)}/{fmt(task.total)}",
                    style="progress.download")


def _thousands(value: Any) -> str:
    return f"{int(value):,}"


def _clock(value: Any) -> str:
    """Seconds as ``4:12`` -- or ``1:04:12`` once there is an hour to show."""
    total = max(0, int(value))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _short_call(call: str) -> str:
    """`potato.text.analyze_readability` -> `analyze readability`."""
    leaf = str(call).rsplit(".", 1)[-1]
    return leaf.replace("_", " ")


def reporter_for(display: Optional[RunDisplay]) -> Callable[..., None]:
    """Adapt a display to the ``on_event`` signature ``run_preset`` expects."""
    def on_event(name: str, **payload: Dict[str, Any]) -> None:
        if display is not None:
            display.handle(name, **payload)

    return on_event
