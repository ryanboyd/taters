"""
"How much of a spreadsheet to read" -- the one setting behind every column question.

When you point Taters at a spreadsheet it looks the file over before asking
anything about it: which columns hold numbers, which hold labels that repeat,
which could group rows together, which could be held constant within a group.
Every one of those answers is only as true as what it read.

It used to read the first two hundred rows. On a real file of 938 responses
and 143 columns that was wrong in a way nobody could see: ten columns looked
constant within a group across the sample and were not across the rest, so
they were offered as control variables that would have arrived empty, while
the columns somebody actually wanted were correctly withheld and looked
missing. Reading everything also says up front whether a file's rows match
its header, which is far cheaper to learn when you choose the file than after
an hour of extraction.

So it reads all of it, and this is where somebody whose files are big enough
to mind can say otherwise.
"""
from __future__ import annotations

from .. import glyphs
from ..prompts import Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK"]

_BACK = "back"

#: The caps on offer. `0` is every row.
_CHOICES = (
    (0, "Every row", "The whole file. Slower on a very large export, and the "
                     "only answer that is true of the file."),
    (1000, "First 1,000 rows", "A compromise for files too big to read twice."),
    (200, "First 200 rows", "What Taters used to do. Fast, and wrong about any "
                            "column whose values change further down."),
)


def _run(ctx: TaskContext):
    import os

    from ...helpers.settings import (INSPECT_ROWS_ENV, INSPECT_ROWS_KEY,
                                     clear_setting, inspect_row_limit,
                                     save_setting)

    prompter = ctx.prompter
    while True:
        limit = inspect_row_limit()
        now = next((label for value, label, _h in _CHOICES if value == limit),
                   f"First {limit:,} rows")
        prompter.note(f"\n  Reading: {now}", style="cyan", wrap=False)
        if os.environ.get(INSPECT_ROWS_ENV, "").strip():
            prompter.reason(f"{INSPECT_ROWS_ENV} is set in this environment, "
                            f"and it wins over anything chosen here.")
        prompter.reason(
            "Taters looks a spreadsheet over before asking about its columns. "
            "What it reads is what those questions can be right about.")

        rows = [Choice(str(value), label, help_text)
                for value, label, help_text in _CHOICES]
        rows.append(Choice(_BACK, f"{glyphs.BACK} Back", tone="nav"))

        try:
            picked = str(prompter.select("How much of a spreadsheet to read:",
                                         rows))
        except GoBack:
            return None
        if picked == _BACK:
            return None

        # stored only when it differs from the default, so a settings file
        # from an older Taters -- which has no key here -- keeps reading
        # everything rather than being read as an explicit choice.
        if int(picked) == 0:
            clear_setting(INSPECT_ROWS_KEY)
        else:
            save_setting(INSPECT_ROWS_KEY, int(picked))
        prompter.note("  Saved. It applies the next time you choose a file.",
                      style="green")


TASK = Task(
    id="inspect_rows",
    label="How much of a spreadsheet to read",
    help="All of it by default; lower it only if your files are very large",
    run=_run,
)
