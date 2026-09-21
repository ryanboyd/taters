"""
"Settings and tools" -- the things that are not a run.

Managing pipeline files and checking whether the GPU works have nothing to do
with each other except this: neither is the thing you came here to do. Putting
them on the front page alongside "Extract features from my data" gave equal
weight to a housekeeping task and the point of the program.

So the front page keeps the two verbs that matter -- extract, run -- and
everything else lives one level down.
"""

from __future__ import annotations

from .. import glyphs
from ..prompts import Cancelled, Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK", "entries"]


def entries():
    """The submenu, in order. Lazy for the same reason the registry is."""
    from . import (gpu, hashbrowns, inspect_rows, manage_data, terminal,
                   update_check)

    # the question people actually arrive with is "is my GPU working?", so it
    # opens the list. everything Taters keeps on disk shares the row beneath
    # it rather than spreading five near-identical "Manage ..." rows across
    # this screen. and the hashbrowns sit last, where nobody trips over them
    # on the way to work.
    return [gpu.TASK, manage_data.TASK, inspect_rows.TASK, terminal.TASK,
            update_check.TASK, hashbrowns.TASK]


def _run(ctx: TaskContext) -> None:
    prompter = ctx.prompter

    while True:
        items = entries()
        choices = [
            Choice(task.id, task.label, task.help, disabled=task.blocked(ctx))
            for task in items
        ]
        choices.append(Choice("back", f"{glyphs.BACK} Back", "Return to the main menu",
                              tone="nav"))

        try:
            picked = str(prompter.select("Settings and tools", choices))
        except GoBack:
            return

        if picked == "back":
            return

        task = next(t for t in items if t.id == picked)
        try:
            task.run(ctx)
        except (Cancelled, GoBack):
            # backing out of one tool returns to this menu, not out of settings
            # altogether -- otherwise a mistyped path costs the whole detour.
            prompter.note("")


TASK = Task(
    id="settings",
    label="Settings and tools",
    help="Check your setup, or manage saved pipelines",
    run=_run,
)
