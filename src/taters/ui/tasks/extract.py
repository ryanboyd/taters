"""
The two doing-verbs on the front page: extract features, or extract features
and analyze them.

They are one flow with one difference, and the difference is an *intention*
rather than a setting. "Turn my recordings into numbers" and "find out
whether these groups differ" are different reasons to be here, and the second
one changes what the wizard should ask: it knows there are outcomes, so it
can insist on a spreadsheet at the first question instead of discovering the
problem after the user has browsed to a folder and picked features for it.

Offering them separately also stops the first flow asking a question most of
its users cannot answer. The analysis stage was optional, which meant every
feature-extraction run was asked about statistics it had no columns for.

The flow itself still lives in :mod:`taters.ui.wizard`, which is where its
tests point and where it has been driven end to end against real media. This
module is the thin adapter that lets the hub offer it.
"""

from __future__ import annotations

from . import Task, TaskContext

__all__ = ["TASK", "ANALYZE_TASK"]


def _run_features_only(ctx: TaskContext):
    from ..wizard import run_wizard

    return run_wizard(ctx.prompter, cwd=ctx.cwd, banner=False,
                      analyses=False).ok


def _run_with_analyses(ctx: TaskContext):
    from ..wizard import run_wizard

    return run_wizard(ctx.prompter, cwd=ctx.cwd, banner=False,
                      analyses=True).ok


TASK = Task(
    id="extract",
    label="Extract features from my data",
    help="Turn audio, video or text into measures you can analyze",
    run=_run_features_only,
)

ANALYZE_TASK = Task(
    id="extract_analyze",
    label="Extract features and run analyses",
    help="The same, then compare groups, correlate with outcomes, or build a "
         "prediction model — needs a spreadsheet with those columns in it",
    run=_run_with_analyses,
)
