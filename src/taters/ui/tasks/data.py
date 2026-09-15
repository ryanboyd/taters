"""
"Wrangle/Analyze data" -- the two things you do to a spreadsheet you have.

Wrangling and analyzing were separate front-page rows, which read as two
unrelated errands. They are not: both start from a file already on disk and
neither extracts a single feature. One tidies the file, the other runs the
statistics over it, and a lot of the time the same person does both in the
same sitting -- combine the rows per participant, then predict something
from them.

So they share one row, and the choice between them is the first question
rather than a decision made on the front page. The front page is shorter for
it, which matters more than it sounds: the list is what a new user reads to
work out what the program is even for.
"""

from __future__ import annotations

from typing import Optional

from .. import glyphs
from ..prompts import Cancelled, Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK", "entries"]


def entries():
    """The submenu, in order. Lazy for the same reason the registry is."""
    from . import analyze, wrangle

    # tidying comes first: it is the step that produces the file the other one
    # reads, and doing them in the other order is usually a mistake
    return [wrangle.TASK, analyze.TASK]


def _run(ctx: TaskContext) -> Optional[bool]:
    prompter = ctx.prompter
    ran = False

    while True:
        items = entries()
        choices = [
            Choice(task.id, task.label, task.help, disabled=task.blocked(ctx))
            for task in items
        ]
        choices.append(Choice("back", f"{glyphs.BACK} Back", "Return to the main menu",
                              tone="nav"))

        try:
            picked = str(prompter.select("Wrangle or analyze data", choices))
        except GoBack:
            return ran or None

        if picked == "back":
            return ran or None

        task = next(t for t in items if t.id == picked)
        try:
            if task.run(ctx):
                ran = True
        except (Cancelled, GoBack):
            # backing out of one returns here rather than to the front page:
            # the likeliest reason to back out is picking the wrong one of two
            prompter.note("")


TASK = Task(
    id="data",
    label="Wrangle/Analyze data",
    help="Tidy a spreadsheet you already have, or run the statistics over it",
    run=_run,
)
