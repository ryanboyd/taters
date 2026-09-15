"""
"Manage Taters data" -- everything Taters keeps between runs.

Settings had grown into a flat list where "Check my setup" sat beside five
near-identical "Manage ..." rows, and the one question people actually arrive
with -- is my GPU working? -- was buried among them. The five are all the same
kind of errand, though: something Taters has stored on your disk, which you
want to rename, export or throw away.

So they share a row. What lives here is anything with a lifetime longer than
one run: the pipelines you saved, the dictionaries and archetypes and stop
lists you imported, the models and encoders your runs produced, and the
downloaded model cache that quietly grows until somebody looks at it.
"""

from __future__ import annotations

from ..prompts import Cancelled, Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK", "entries"]


def entries():
    """The submenu, in order. Lazy for the same reason the registry is."""
    from . import library, manage, model_cache

    # pipelines first (the thing a run leaves behind that you re-run), then
    # the things you import, then the things runs produce, and last the cache
    # -- which is not yours so much as downloaded on your behalf
    return [manage.TASK, library.TASK, library.STOPLISTS_TASK,
            library.MODELS_TASK, library.ENCODERS_TASK, model_cache.TASK]


def _run(ctx: TaskContext) -> None:
    prompter = ctx.prompter

    while True:
        items = entries()
        choices = [
            Choice(task.id, task.label, task.help, disabled=task.blocked(ctx))
            for task in items
        ]
        choices.append(Choice("back", "↩ Back", "Return to settings",
                              tone="nav"))

        try:
            picked = str(prompter.select("Manage Taters data", choices))
        except GoBack:
            return

        if picked == "back":
            return

        task = next(t for t in items if t.id == picked)
        try:
            task.run(ctx)
        except (Cancelled, GoBack):
            # backing out of one shelf returns to this menu, not out of
            # settings altogether -- a mistyped path costs the whole detour
            prompter.note("")


TASK = Task(
    id="manage_data",
    label="Manage Taters data",
    help="Saved pipelines, dictionaries, stop lists, models, encoders, and "
         "the downloaded model cache",
    run=_run,
)
