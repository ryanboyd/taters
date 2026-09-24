"""
Settings entries for the library: the dictionaries, and the stop lists.

The management flow itself lives in :mod:`taters.ui.library` and is shared with
the wizard's picker, so "Manage dictionaries…" mid-pipeline and these menus are
one and the same screen. This module only chooses which *kind* to manage --
and skips even that question where there is nothing to choose: stop lists are
one kind, so their entry opens the manager directly.
"""

from __future__ import annotations

from .. import glyphs
from ..prompts import Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK", "STOPLISTS_TASK", "MODELS_TASK", "ENCODERS_TASK"]

#: The kinds the dictionaries entry offers. Stop lists are deliberately not
#: here -- they have their own Settings row, so someone hunting for "where do
#: my stopwords live" is not sent through a dictionary-shaped door.
#:
#: Word norms *are* here. They are a third shelf rather than a flavor of
#: dictionary, because the two are scored differently and the file formats are
#: indistinguishable -- putting them in the same drawer is how somebody ends up
#: with a concreteness table scored as a word count.
_DICTIONARY_KINDS = ("dictionaries", "norms", "archetypes")


def _run_dictionaries(ctx: TaskContext) -> None:
    from ...helpers.library import KINDS, entries
    from ..library import manage_library

    prompter = ctx.prompter
    while True:
        kinds = [KINDS[kid] for kid in _DICTIONARY_KINDS]
        choices = [
            Choice(kind.id, kind.label,
                   f"{len(entries(kind))} imported · {kind.help}")
            for kind in kinds
        ]
        choices.append(Choice(":back", f"{glyphs.BACK} Back", tone="nav"))

        try:
            picked = str(prompter.select("Which library?", choices))
        except GoBack:
            return
        if picked == ":back":
            return
        manage_library(prompter, next(k for k in kinds if k.id == picked))


def _run_models(ctx: TaskContext) -> None:
    from ...helpers.library import KINDS
    from ..library import manage_library

    manage_library(ctx.prompter, KINDS["models"])


def _run_encoders(ctx: TaskContext) -> None:
    from ...helpers.library import KINDS
    from ..library import manage_library

    manage_library(ctx.prompter, KINDS["encoders"])


def _run_stoplists(ctx: TaskContext) -> None:
    from ...helpers.library import KINDS
    from ..library import manage_library

    manage_library(ctx.prompter, KINDS["stoplists"])

TASK = Task(
    id="library",
    label="Manage dictionaries",
    help="Import, rename, export or delete the dictionaries your analyses use",
    run=_run_dictionaries,
)

MODELS_TASK = Task(
    id="models",
    label="Manage saved models",
    help="Import, rename, export or delete the saved models any pipeline can "
         "score with -- topic models, prediction models, word vectors, "
         "fine-tuned text predictors",
    run=_run_models,
)

ENCODERS_TASK = Task(
    id="encoders",
    label="Manage text encoders",
    help="Language models adapted to your texts, the base for embeddings "
         "and fine-tuning",
    run=_run_encoders,
)

STOPLISTS_TASK = Task(
    id="stoplists",
    label="Manage stop lists",
    help="Import, rename, export or delete the stop word lists the n-gram "
         "tools can apply",
    run=_run_stoplists,
)
