"""
"Train a model" on the front page.

Training is its own intention -- the run exists for the model it leaves
behind, not for the feature table -- so it sits beside the extraction verbs
rather than inside them. The flow lives in :mod:`taters.ui.train`; this is
the adapter the hub offers.
"""
from __future__ import annotations

from . import Task, TaskContext

__all__ = ["TASK"]


def _run(ctx: TaskContext):
    from ..train import run_train

    return run_train(ctx.prompter, cwd=ctx.cwd)


TASK = Task(
    id="train",
    label="Train a model (beta testing)",
    help="Word vectors, an adapted language model, or a transformer fine-tuned "
         "to predict your outcomes -- saved with a report, kept in your "
         "library, applied to new data",
    run=_run,
)
