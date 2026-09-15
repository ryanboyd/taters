"""
"Hashbrowns" -- the demo, as a menu entry.

The production itself lives in :mod:`taters.ui.hashbrowns` and is left
exactly as written. This is only the door: a row on the opening menu, sitting
just above Quit, that hands the terminal to the demo and takes it back when
the user presses a key. It runs nothing that can succeed or fail, so it
returns ``None`` and the menu carries on as if nothing happened.
"""
from __future__ import annotations

from typing import Optional

from . import Task, TaskContext


def _run(ctx: TaskContext) -> Optional[bool]:
    # imported here, not at the top: the demo pulls in numpy and builds its
    # meshes on first use, and the main menu shouldn't pay for that unless
    # somebody actually picks the row
    from ..hashbrowns import run

    run()
    return None


TASK = Task(
    id="hashbrowns",
    label="Hashbrowns",
    help="When you need something to sizzle your mind... Press [esc] when you need to rejoin reality...",
    run=_run,
)
