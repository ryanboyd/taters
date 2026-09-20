"""
"Wrangle Language Models": what to do, then with what, then the wizard with
that step already chosen.

Training is not feature extraction -- the point of the run is the model,
not the table -- so it gets its own front-page verb. What it *shares* with
extraction is everything after its own questions: the source, the level,
the options screen, the saved and re-runnable pipeline, the live progress,
the finish screen. So this module asks its questions and hands the answer
to :func:`taters.ui.wizard.run_wizard` as a preselected step; a second copy
of that flow would drift from the first on the first change to either.

The questions are intention first, then material. Four things a person
comes here to do -- train a model from scratch, adapt an existing one,
fine-tune one to predict, or move models in and out -- and then, for the
ones that train, *with what*: word embeddings or a transformer. The second
question is asked even when it has one answer. That is deliberate: a menu of
one row still tells you exactly what you are about to do, and it is the
scaffolding the next option slots into.

Importing pre-trained vectors is here too, under the fourth verb, though it
trains nothing: it is where someone looks for "get a word-vector model into
Taters". It is a one-off, not a pipeline -- there is no corpus to re-run it
over -- so it asks for the file and writes the model, then offers the library.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import glyphs
from .prompts import Cancelled, Choice, GoBack, Prompter

__all__ = ["run_train", "VERBS", "TRAINABLE", "WRANGLE", "encoder_stem",
           "ask_outcomes", "seed_scratch"]

#: The first question: what to do. ``(value, label, help)``.
VERBS: List[Tuple[str, str, str]] = [
    ("scratch", "Train a model from scratch",
     "Word vectors, or a whole transformer, from nothing but your texts."),
    ("adapt", "Adapt an existing model to my texts",
     "Continue a language model's own pretraining on your corpus, so it speaks "
     "your texts' dialect before it embeds or predicts anything."),
    ("predict", "Fine-tune a transformer to predict my outcomes",
     "Train an encoder end to end to predict spreadsheet columns from text -- "
     "numbers or categories, cross-validated."),
    ("wrangle", "Import or export language models",
     "Bring a model in, or copy one out to share."),
]

#: The second question, per verb that trains: ``(value, label, help, recipe)``.
#: A kind whose recipe needs an extra that is not installed is offered
#: grayed out, with the install command; a verb all of whose kinds are grayed
#: is grayed itself, with the first kind's reason.
TRAINABLE: Dict[str, List[Tuple[str, str, str, str]]] = {
    "scratch": [
        ("word_vectors", "Word embeddings",
         "word2vec or fastText trained on your corpus: a vector per word, the "
         "mean per text as features, similarity to concept dictionaries. Saves "
         "a model; minutes on a CPU.",
         "word_vectors_train"),
        ("scratch_transformer", "A transformer, from random weights",
         "A whole encoder, and a tokenizer learned from your texts. Heavy-duty: "
         "hours on one GPU, days on a CPU -- and below a few million words the "
         "result will be worse than any pretrained model you could adapt.",
         "pretrain_encoder"),
    ],
    "adapt": [
        ("adapt_encoder", "A transformer",
         "Continue an encoder's masked-word pretraining on your texts, no labels "
         "needed. The base for raw transformer embeddings or fine-tuning. Slow "
         "on a CPU.",
         "adapt_encoder"),
    ],
    "predict": [
        ("finetune_predictor", "A transformer",
         "Fine-tune a transformer to predict spreadsheet columns from text -- "
         "numbers or categories, cross-validated. Saves a scoring model with a "
         "full report. Slow on a CPU.",
         "finetune_text_predictor"),
    ],
}

#: The second question under "import or export": which shelf, or the
#: one importer that is not a library shelf.
_IMPORT = "import_vectors"
WRANGLE: List[Tuple[str, str, str]] = [
    (":encoders", "Text encoders",
     "Language models adapted to your texts or trained from scratch here. "
     "Import, export, rename or delete them."),
    (":models", "Saved models",
     "Predictors, classifiers, word vectors, topic models -- anything that "
     "scores a dataset. Import, export, rename or delete them."),
    (_IMPORT, "Bring in pre-trained word vectors",
     "A GloVe, word2vec or fastText file from elsewhere, saved as a model that "
     "scores your texts like a trained one."),
]

_FORMATS = [
    Choice("auto", "Work it out from the file", "Recommended."),
    Choice("glove", "GloVe text", "One word and its numbers per line, no header."),
    Choice("word2vec_text", "word2vec text", "A 'count dimensions' header line, then words."),
    Choice("word2vec_bin", "word2vec binary", "GoogleNews-style .bin (needs gensim)."),
    Choice("fasttext_bin", "fastText binary", "cc.en.300.bin and the like (needs gensim)."),
]

#: The size presets offered when training a transformer from scratch, with
#: what each costs -- said before the run, because "heavy-duty" means little
#: until it is a number of hours.
_PRESET_ROWS = [
    Choice("small", "Small: 4 layers, 256 wide (about 10M parameters)",
           "An overnight run on one GPU. The right first try."),
    Choice("base", "Base: 12 layers, 768 wide (about 110M parameters)",
           "The BERT/RoBERTa shape. Days on one GPU; wants millions of words."),
    Choice("custom", "Custom: set the layers, width and heads myself",
           "On the options screen that follows."),
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
    # every step that takes an encoder runs on torch and transformers -- and
    # so does one that builds an encoder from nothing, which names no encoder
    # to start from and says so with `needs_torch` instead
    if recipe.encoder_param is not None or recipe.needs_torch:
        from ..text._transformer_common import torch_missing_reason

        if torch_missing_reason():
            return "needs torch and transformers (see the install guide)"
    return ""


def _kind_rows(verb: str) -> List[Choice]:
    return [Choice(value, label, help_text, disabled=_grayed(recipe_id))
            for value, label, help_text, recipe_id in TRAINABLE[verb]]


def _verb_rows() -> List[Choice]:
    """The first screen. A verb is grayed only when nothing under it can run."""
    rows = []
    for value, label, help_text in VERBS:
        disabled = ""
        if value in TRAINABLE:
            kinds = _kind_rows(value)
            if kinds and all(c.disabled for c in kinds):
                disabled = kinds[0].disabled
        rows.append(Choice(value, label, help_text, disabled=disabled))
    return rows


#: Which variable names the model each transformer kind produces, and the
#: suffix its default name takes after the encoder it started from (or, for
#: one started from nothing, after its size preset).
_NAME_VARS = {
    "adapt_encoder": ("encoder_name", "adapted"),
    "finetune_text_predictor": ("predictor_name", "finetuned"),
    "pretrain_encoder": ("encoder_name", "scratch"),
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


def _count_texts(src) -> int:
    """Roughly how many texts the source holds, cheaply, for the estimate."""
    try:
        if src.source == "csv":
            with Path(src.path).open("rb") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        return len(src.inputs or [])
    except OSError:
        return 0


def _parameters_m(preset: str) -> float:
    """A preset's parameter count in millions, for the time estimate."""
    from ..text._transformer_common import PRESETS

    shape = PRESETS[preset]
    if preset == "custom":
        return 30.0                      # a middle guess; the options screen decides
    h, layers, vocab = shape["hidden_size"], shape["layers"], shape["vocab_size"]
    return (12 * h * h * layers + vocab * h) / 1e6


def seed_scratch() -> Callable[..., None]:
    """
    The ``before_options`` hook for a transformer trained from nothing: which
    size, then what it will cost -- in hours, on this machine, for this many
    texts -- and a chance to think again before the run is composed.
    """
    def hook(prompter: Prompter, src, var_values: dict, overrides: dict) -> None:
        from ..text._transformer_common import estimate_minutes, torch_missing_reason

        prompter.reason(
            "Nothing here is pretrained: the model starts from random weights and "
            "its tokenizer is learned from your texts. A language model learns "
            "language from quantity -- below a few million words, the result will "
            "be worse at everything than any pretrained model you could adapt.")
        preset = str(prompter.select("Which size?", _PRESET_ROWS,
                                     default=var_values.get("pretrain_preset", "small")))
        var_values["pretrain_preset"] = preset
        var_values.setdefault("encoder_name", f"{preset}-scratch")

        n_texts = _count_texts(src)
        device_name = "cpu"
        devices = 1
        if not torch_missing_reason():
            import torch

            if torch.cuda.is_available():
                device_name, devices = "cuda", int(torch.cuda.device_count())
        minutes = estimate_minutes(n_texts, 200.0, epochs=40, passes=1,
                                   params_m=_parameters_m(preset),
                                   device_name=device_name)
        where = (f"{devices} GPUs, from one process (not as efficient as a "
                 f"distributed launch)" if devices > 1
                 else "one GPU" if device_name == "cuda" else "the CPU")
        hours = minutes / 60
        cost = (f"about {hours:.0f} hours" if hours >= 2
                else f"about {max(1, round(minutes)):.0f} minutes")
        prompter.reason(
            f"A rough estimate for {n_texts:,} texts at the {preset} size, if all "
            f"40 epochs run (training stops earlier once the held-out loss stops "
            f"falling): {cost} on {where}. Within a factor of two either way.")
        if not prompter.confirm("Go ahead with training from scratch?", default=True):
            raise GoBack()
    return hook


def run_train(prompter: Prompter, *, cwd: Path) -> Optional[bool]:
    """
    Ask what to do and with what, then run the flow for it.

    Returns what the wizard returns for a run (``ok``), ``None`` when
    nothing ran, and raises :class:`GoBack` when the user backs out of the
    first question -- the menu that launched this is the screen before.

    Esc walks back one screen at a time: from the encoder picker to the
    kind, from the kind to the verb, from the verb to the main menu.
    """
    from . import recipes as _recipes
    from .library import pick_encoder
    from .wizard import run_wizard

    prompter.reason(
        "A model trained here is saved with a report of how it was trained, "
        "can be added to your library, and then scores any dataset from the "
        "feature checklist.")

    stage = "verb"
    verb = ""
    recipe = None
    hook: Optional[Callable[..., None]] = None
    while True:
        if stage == "verb":
            verb = str(prompter.select("What would you like to do?", _verb_rows()))
            stage = "kind"

        elif stage == "kind":
            if verb == "wrangle":
                try:
                    return _wrangle(prompter, cwd=cwd)
                except GoBack:
                    stage = "verb"
                    continue
            try:
                kind = str(prompter.select("With what?", _kind_rows(verb)))
            except GoBack:
                stage = "verb"
                continue
            recipe_id = next(r for v, _l, _h, r in TRAINABLE[verb] if v == kind)
            recipe = _recipes.by_id(recipe_id)
            hook = None
            if recipe.encoder_param is not None:
                stage = "encoder"
            else:
                if recipe.id == "pretrain_encoder":
                    hook = seed_scratch()
                stage = "run"

        elif stage == "encoder":
            # first thing's first: which model are we starting from? this used
            # to live on the options screen, where nobody found it -- and
            # nobody noticed that a default had quietly been picked for them.
            # it's the one choice that shapes everything after it, so we ask
            # it before anything else, and name the result after it
            # (distilroberta-base-finetuned). Esc here is back to the kind,
            # since that's the previous screen
            encoder = pick_encoder(prompter, str(recipe.vars["encoder"]["default"]),
                                   for_step=recipe.label)
            if encoder is None:
                stage = "kind"
                continue
            hook = _seed_encoder(recipe.id, encoder)
            stage = "run"

        else:
            result = run_wizard(prompter, cwd=cwd, banner=False, analyses=False,
                                preselected=[recipe.id], text_only=True,
                                before_options=hook)
            return result.ok if result.ran else None


def _wrangle(prompter: Prompter, *, cwd: Path) -> Optional[bool]:
    """
    The fourth verb: which models, then the library screen that already
    imports, exports, renames and deletes them -- or the one importer that
    is not a library shelf.
    """
    from ..helpers import library as _lib
    from .library import manage_library

    rows = [Choice(value, label, help_text) for value, label, help_text in WRANGLE]
    prompter.reason(
        "Your library keeps models between sessions. Export copies one out to "
        "share; import brings one in from a colleague, or from a run's folder.")
    picked = str(prompter.select("Which models?", rows))     # GoBack -> the verb
    if picked == _IMPORT:
        return _import_vectors(prompter, cwd=cwd)
    kind = _lib.KINDS["encoders" if picked == ":encoders" else "models"]
    manage_library(prompter, kind)
    return None


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
        # no concept dictionaries here: bringing a model in is not applying
        # it. they are set per model, later, under Settings -> saved models
        model = import_word_vectors(path, out, format=fmt, name=name,
                                    overwrite_existing=True, verbose=False)
    except (ValueError, ImportError, OSError) as e:
        from rich.markup import escape

        prompter.note(f"  Not imported: {escape(str(e))}", style="red")
        return False
    from ..helpers.model_spec import describe

    info = describe(model)
    prompter.note(f"  {glyphs.TICK} Saved {info.display()} ({info.n_outputs} dimensions) to "
                  f"{model}", style="green")
    prompter.note("    Its report and nearest neighbors are beside it.", style="dim")
    offer_library_import(prompter, [(model, info.display())])
    return True
