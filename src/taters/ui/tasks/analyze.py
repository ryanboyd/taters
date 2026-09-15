"""
"Analyze data" -- the statistics, on a spreadsheet you already have.

Every other road to the statistics runs through feature extraction, which is
no help to somebody whose numbers already exist: LIWC output from elsewhere,
questionnaire scales, measures computed in R. The analyses themselves never
cared where the columns came from, so this task hands them the user's own.

It is the ordinary wizard with one step chosen in advance -- the step that
reads the spreadsheet's own columns as the measures -- so the source
question, the level question, the whole statistics stage, the options
screen, naming, saving and running are all the ones every other flow uses.
"""
from __future__ import annotations

from typing import Optional

from . import Task, TaskContext

__all__ = ["TASK"]


def _run(ctx: TaskContext) -> Optional[bool]:
    from ..wizard import run_wizard

    result = run_wizard(
        ctx.prompter, cwd=ctx.cwd, banner=False,
        # statistics are the whole point here, so the stage insists on an
        # answer rather than offering to skip
        analyses=True,
        preselected=["spreadsheet_columns"],
        # a word cloud of column names is not what anyone came here for. off
        # by default, still on the options screen for anyone who wants them
        var_defaults={"wordclouds": False},
    )
    return result.ok if result.ran else None


TASK = Task(
    id="analyze_spreadsheet",
    label="Analyze data",
    help="Statistics on a spreadsheet you already have: group differences, "
         "correlations, prediction models.",
    run=_run,
)
