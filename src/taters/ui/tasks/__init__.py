"""
The things a user can ask Taters to do, as a registry rather than a flow.

The wizard began as one linear script: where is your data, what do you want,
here it is. That shape did not survive: "Run a saved pipeline" and "Manage
pipelines" do not involve data at all, and "Extract features and run analyses"
-- what words relate to an outcome, whether groups differ on a measure --
starts from a spreadsheet's columns rather than from files. A single
hard-coded sequence would have to grow branches at the top for each of those.

So the front door is a **registry**. Each :class:`Task` is a self-contained
flow with a label, a one-line explanation, and a ``run`` function. Adding a
task later is adding a module and one entry here; it is not a refactor of
anything that already works.

What deliberately does *not* live here
--------------------------------------
The pipeline machinery, because it already generalizes. An analysis step is a
GLOBAL step that reads a features CSV, and
:func:`taters.ui.compose.resolve_selection` already chains backwards from a
goal through the capability graph -- ask for something that needs n-grams and
it will pull in the transcript, the transcription, and the WAV conversion on
its own. Future analysis tasks contribute *recipes*, not a second execution
model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from ..prompts import Prompter

__all__ = ["Task", "TaskContext", "all_tasks"]


@dataclass
class TaskContext:
    """Everything a task needs from the outside world."""

    prompter: Prompter
    cwd: Path

    @property
    def pipelines_dir(self) -> Path:
        """Where the user's own pipelines live. Not created until something is saved."""
        return self.cwd / "pipelines"

    def own_pipelines(self) -> List[Path]:
        """
        The user's saved pipelines. Built-ins are never included.

        Discovery is delegated to the runner's own rules
        (:func:`run_pipeline.available_presets`) rather than re-implemented:
        a private copy lived here and had already drifted -- the runner
        searches ``pipelines/`` recursively, the copy did not, so a nested
        preset appeared under Run but not under Manage. One set of rules, or
        the two screens disagree about what exists.
        """
        from ...pipelines.run_pipeline import available_presets, is_builtin_preset

        return sorted(path for path, _meta in available_presets(self.cwd)
                      if not is_builtin_preset(path))

    @staticmethod
    def owns_folder(path: Path) -> bool:
        """Whether this preset has a folder of its own, named for it."""
        path = Path(path)
        return path.parent.name == path.stem


@dataclass(frozen=True)
class Task:
    """
    One thing the user can choose from the opening menu.

    Attributes
    ----------
    id, label, help : str
        Identity, the menu line, and the explanation under it.
    run : callable
        ``run(ctx) -> bool | None``. ``False`` means the work ran and
        finished with problems, which is what lets ``taters`` still exit
        non-zero for a script that is watching. ``None`` means nothing ran
        that could succeed or fail. Raising
        :class:`~taters.ui.prompts.Cancelled` means "I backed out", and returns
        the user to the menu rather than ending the session.
    unavailable_because : callable, optional
        ``(ctx) -> str``. A non-empty string means the task is shown grayed out
        with that reason. Showing *why* something cannot be chosen is the point:
        hiding "Run a saved pipeline" until a pipeline exists leaves a new user
        unable to discover that saved pipelines are a thing at all.
    """

    id: str
    label: str
    help: str
    run: Callable[[TaskContext], Optional[bool]] = field(repr=False)
    unavailable_because: Optional[Callable[[TaskContext], str]] = field(
        default=None, repr=False
    )

    def blocked(self, ctx: TaskContext) -> str:
        return self.unavailable_because(ctx) if self.unavailable_because else ""


def all_tasks() -> List[Task]:
    """
    The menu, in order.

    Imported lazily so that a task module can import the wizard without the
    wizard's own import of this package becoming a cycle.
    """
    from . import data, extract, run_saved, settings, train

    # wrangling comes first: getting the data into shape is where every
    # project actually starts, and the user asked for it at the top. then the
    # extraction verbs -- features alone, and features plus the statistics
    # that answer a question about them. those are separate entries because
    # they're separate intentions, and the second one lets the wizard assume
    # there's something to analyze. housekeeping (managing pipeline files,
    # checking whether the GPU works) lives under settings, so that the first
    # screen isn't an equal-weight list of "do the thing" and "tidy up".
    # wrangling and running the statistics share one row: both start from a
    # file that already exists and neither extracts anything, so the choice
    # between them is the first question rather than two front-page rows.
    # training a model comes after those: the run exists for the model it
    # leaves behind, which the checklist then applies. the hashbrowns used to
    # sit last, right above Quit; they are a treat rather than a task, so
    # they moved under settings where nobody meets them on the way to work
    return [data.TASK, extract.TASK, extract.ANALYZE_TASK,
            train.TASK, run_saved.TASK, settings.TASK]

