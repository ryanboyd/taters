"""
The "Train a model" flow: what to train, then the wizard with that step
already chosen, then the offer to keep the model.

Training is not feature extraction -- the point of the run is the model,
not the table -- so it gets its own front-page verb. What it *shares* with
extraction is everything after the first question: the source, the level,
the options screen, the saved and re-runnable pipeline, the live progress,
the finish screen. So this module asks its one question and hands the
answer to :func:`taters.ui.wizard.run_wizard` as a preselected step; a
second copy of that flow would drift from the first on the first change to
either.

Importing pre-trained vectors is here too, though it trains nothing: it is
where someone looks for "get a word-vector model into Taters". It is a
one-off, not a pipeline -- there is no corpus to re-run it over -- so it
asks for the file and writes the model, then offers the library.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional

from .prompts import Cancelled, Choice, GoBack, Prompter

__all__ = ["run_train", "TRAINABLE", "encoder_stem"]

#: The kinds of model that can be trained, in menu order: (value, label,
#: help, the recipe that trains it). A kind whose recipe needs an extra
#: that is not installed is offered grayed out, with the install command.
TRAINABLE = [
    ("word_vectors", "Word vectors from my texts",
     "word2vec or fastText trained on your corpus: a vector per word, the mean "
     "per text as features, similarity to concept dictionaries. Saves a model.",
     "word_vectors_train"),
    ("adapt_encoder", "Adapt a language model to my texts",
     "Continue an encoder's own pretraining on your texts, no labels needed, "
     "so it speaks their dialect. The base for raw transformer embeddings or "
     "fine-tuning. Slow on a CPU.", "adapt_encoder"),
    ("finetune_predictor", "Fine-tune a transformer to predict my outcomes",
     "Fine-tune a transformer to predict spreadsheet columns from text -- "
     "numbers or categories, cross-validated. Saves a scoring model with a "
     "full report. Slow on a CPU.",
     "finetune_text_predictor"),
]

_IMPORT = "import_vectors"
_FORMATS = [
    Choice("auto", "Work it out from the file", "Recommended."),
    Choice("glove", "GloVe text", "One word and its numbers per line, no header."),
    Choice("word2vec_text", "word2vec text", "A 'count dimensions' header line, then words."),
    Choice("word2vec_bin", "word2vec binary", "GoogleNews-style .bin (needs gensim)."),
    Choice("fasttext_bin", "fastText binary", "cc.en.300.bin and the like (needs gensim)."),
]


def _grayed(recipe_id: str) -> str:
    """Why a kind cannot be trained on this machine, or ''."""
    from . import recipes as _recipes
    from .wizard import missing_extras, unavailable_reason

    recipe = _recipes.by_id(recipe_id)
    # no release for this Python beats "go pip install it": the install
    # would succeed and change nothing (gensim on 3.14)
    stuck = unavailable_reason(recipe)
    if stuck:
        return stuck
    absent = missing_extras(recipe)
    if absent:
        return "needs " + " and ".join(f'pip install "taters[{x}]"' for x in absent)
    if recipe.encoder_param is not None:
        # every step that takes an encoder runs on torch and transformers.
        from ..text._transformer_common import torch_missing_reason

        if torch_missing_reason():
            return "needs torch and transformers (see the install guide)"
    return ""


#: Which variable names the model each transformer kind produces, and the
#: suffix its default name takes after the encoder it started from.
_NAME_VARS = {
    "adapt_encoder": ("encoder_name", "adapted"),
    "finetune_text_predictor": ("predictor_name", "finetuned"),
}


def encoder_stem(encoder: str) -> str:
    """
    The short name of an encoder, for naming what is trained from it.

    ``sentence-transformers/all-MiniLM-L6-v2`` is ``all-MiniLM-L6-v2``, a
    library encoder's ``.json`` path is its file stem, ``distilroberta-base``
    is itself -- spelled through :func:`~taters.helpers.model_spec.slug`, so
    the result is a file name.
    """
    from ..helpers.model_spec import slug

    text = str(encoder).strip().rstrip("/\\")
    tail = re.split(r"[\\/]", text)[-1] if text else ""
    if tail.lower().endswith(".json"):
        tail = tail[:-5]
    return slug(tail, fallback="encoder")


def _seed_encoder(recipe_id: str, encoder: str) -> Callable[..., None]:
    """
    The ``before_options`` hook for a transformer kind: the encoder the user
    just picked and a name built from it land in the pipeline's variables,
    and the predictor's outcome columns are asked in the same slot.
    """
    def hook(prompter: Prompter, src, var_values: dict, overrides: dict) -> None:
        # setdefault, so that coming back through here (Esc to the source and
        # forward again) doesn't clobber anything the user already changed
        var_values.setdefault("encoder", encoder)
        name_var, suffix = _NAME_VARS[recipe_id]
        var_values.setdefault(name_var, f"{encoder_stem(encoder)}-{suffix}")
        if recipe_id == "finetune_text_predictor":
            ask_outcomes(prompter, src, var_values, overrides)
    return hook


def run_train(prompter: Prompter, *, cwd: Path) -> Optional[bool]:
    """
    Ask what to train and run the flow for it.

    Returns what the wizard returns for a run (``ok``), ``None`` when
    nothing ran, and raises :class:`Cancelled` when the user backs out of
    the first question -- the menu that launched this is the screen before.
    """
    rows: List[Choice] = []
    for value, label, help_text, recipe_id in TRAINABLE:
        rows.append(Choice(value, label, help_text, disabled=_grayed(recipe_id)))
    rows.append(Choice(_IMPORT, "Bring in pre-trained word vectors",
                       "A GloVe, word2vec or fastText file from elsewhere, saved "
                       "as a model that scores your texts like a trained one."))
    prompter.reason(
        "A model trained here is saved with a report of how it was trained, "
        "can be added to your library, and then scores any dataset from the "
        "feature checklist.")
    from . import recipes as _recipes
    from .library import pick_encoder
    from .wizard import run_wizard

    while True:
        what = str(prompter.select("What would you like to train?", rows))
        if what == _IMPORT:
            return _import_vectors(prompter, cwd=cwd)
        recipe_id = next(r for v, _l, _h, r in TRAINABLE if v == what)
        recipe = _recipes.by_id(recipe_id)

        hook = None
        if recipe.encoder_param is not None:
            # first thing's first: which model are we starting from? this used
            # to live on the options screen, where nobody found it -- and
            # nobody noticed that a default had quietly been picked for them.
            # it's the one choice that shapes everything after it, so we ask
            # it before anything else, and name the result after it
            # (distilroberta-base-finetuned). Esc here is back to the list
            # above, since that's the previous screen
            encoder = pick_encoder(prompter, str(recipe.vars["encoder"]["default"]))
            if encoder is None:
                continue
            hook = _seed_encoder(recipe_id, encoder)

        result = run_wizard(prompter, cwd=cwd, banner=False, analyses=False,
                            preselected=[recipe_id], text_only=True,
                            before_options=hook)
        return result.ok if result.ran else None


def ask_outcomes(prompter: Prompter, src, var_values: dict, overrides: dict) -> None:
    """
    Which columns the predictor should predict, from the spreadsheet's own.

    Offered are the columns that are neither the text nor an identifier
    and hold something (numbers or repeating labels), each shown with what
    it is treated as. A column of numbers becomes a regression, a column
    of labels a classification; the split follows the detected kind, and a
    numeric code that is really a category is named as categorical here
    so the model classifies it rather than regressing on the code.
    """
    from .prompts import Choice
    from .wizard import ask_at_least_one

    taken = set(src.text_cols) | set(src.id_cols) | set(src.group_by)
    kinds = dict(src.kinds or {})
    rows = []
    for col in src.columns:
        if col in taken:
            continue
        kind = kinds.get(col, "")
        if kind in ("text", "empty"):
            continue
        rows.append(Choice(col, col, {"numbers": "a measurement: the model will predict "
                                                 "how much",
                                      "labels": "a category: the model will predict which"
                                      }.get(kind, "")))
    if not rows:
        prompter.note("  This spreadsheet has no column besides the text and the "
                      "identifiers that a model could predict.", style="yellow")
        raise GoBack()
    prompter.reason(
        "The model learns to predict these from the text. A column of numbers "
        "is predicted as a quantity (regression); a column of labels as a "
        "category (classification). Several columns train one model with a "
        "head for each.")
    picked = ask_at_least_one(prompter, "Which column(s) should the model predict?",
                              rows, thing="one column")
    categorical = [c for c in picked if kinds.get(c) == "labels"]
    var_values["predictor_outcomes"] = list(picked)
    var_values["predictor_categorical"] = categorical


def _ask_concept_dicts(prompter: Prompter) -> List[str]:
    """Which of the library's dictionaries the model should measure texts
    against -- nothing ticked means none. Not asked when the library holds
    no dictionaries: there is nothing to pick, and importing one belongs
    to Settings, not to the middle of this flow."""
    from ..helpers import library as _lib
    from .library import pick_from_library

    kind = _lib.KINDS["dictionaries"]
    if not _lib.entries(kind):
        return []
    prompter.reason(
        "Optionally, tick dictionaries whose categories the model should "
        "measure texts against: each category becomes a sim_ column. Esc or "
        "nothing ticked means none.")
    picked = pick_from_library(prompter, kind, start_unticked=True)
    return list(picked or [])


def _import_vectors(prompter: Prompter, *, cwd: Path) -> Optional[bool]:
    """Ask for the file and the few settings that matter, write the model
    under ``<cwd>/models/``, and offer the library."""
    from ..helpers.model_spec import slug
    from .browse import browse_for_file
    from .library import offer_library_import

    try:
        path = browse_for_file(
            prompter, question="Where is the vectors file?",
            suffixes=(".txt", ".vec", ".bin"))
        fmt = str(prompter.select("What format is it?", _FORMATS, default="auto"))
        name = str(prompter.text("What do you want to call this model?",
                                 default=slug(path.stem))).strip() or slug(path.stem)
        concept_dicts = _ask_concept_dicts(prompter)
    except GoBack:
        raise Cancelled()
    out = cwd / "models" / f"{slug(name)}.json"
    if out.exists():
        if not prompter.confirm(f"{out.name} already exists here. Replace it?",
                                default=False):
            return None
    from ..text.word_vectors import import_word_vectors

    prompter.note(f"  Reading {path.name}…", style="dim")
    try:
        model = import_word_vectors(path, out, format=fmt, name=name,
                                    concept_dicts=concept_dicts,
                                    overwrite_existing=True, verbose=False)
    except (ValueError, ImportError, OSError) as e:
        from rich.markup import escape

        prompter.note(f"  Not imported: {escape(str(e))}", style="red")
        return False
    from ..helpers.model_spec import describe

    info = describe(model)
    prompter.note(f"  ✓ Saved {info.display()} ({info.n_outputs} dimensions) to "
                  f"{model}", style="green")
    prompter.note("    Its report and nearest neighbors are beside it.", style="dim")
    offer_library_import(prompter, [(model, info.display())])
    return True
