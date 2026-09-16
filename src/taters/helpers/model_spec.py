"""
One description of a saved model, whatever kind it is.

Taters writes several kinds of reusable instrument -- a MEM topic model, a
ridge regression, a classifier -- and they were each self-describing to
their own scoring function and to nothing else. That was fine while each
kind had its own menu entry, and stops being fine the moment a screen wants
to offer *models* as a category: to list them it has to say what each one
is, to score with one it has to know whether the thing needs text or
features, and to write the results it has to know what columns will appear.

So each kind registers those three answers here, and every screen asks this
module instead of sniffing JSON keys for itself. The alternative -- a
per-kind branch at each call site -- is the shape where adding a fourth kind
means finding all of them.

Adding a kind of model
----------------------
Everything a screen or a scoring run needs to know about a model comes from
this registry, so a new kind -- a fine-tuned transformer, a gradient-boosted
tree, whatever comes next -- is added in one place:

1. Give it a ``kind`` tag in the file it writes (``"taters-<x>-model"``) and a
   ``format`` integer.
2. Add a :class:`ModelType` to :data:`MODEL_TYPES` stating:

   * ``needs`` -- ``"text"`` if it reads text and re-derives its own
     instrument (the MEM pattern), ``"features"`` if it reads feature columns
     someone else measured. That one word decides whether the
     settings-provenance gate applies to it, and nothing else has to know.
   * ``read`` -- doc -> ``(outputs, columns, inputs)``, so listings can say
     what it is and what it will add.
   * ``score`` -- ``"module:function"`` for applying it. Omit this and
     scoring refuses by name instead of guessing.
   * a ``"payload"`` list in the model file -- the sibling weights the
     library moves with it (see ``helpers.library.payload_of``)
   * ``write_outputs`` -- if its output columns can be renamed at import.
   * ``classes`` -- doc -> ``{outcome: [class, ...]}`` when it predicts
     categories, so a class can be relabeled per model (``"0"`` ->
     ``"control"``) and every applier writes the label through
     :func:`class_label`.
   * ``apply_settings`` -- the defaults that govern how it is applied to new
     data (batch size, weighting, whether to emit probabilities). They live
     in the model file's ``apply`` block, editable per model in Settings, and
     an applier reads them through :func:`apply_defaults` so a pipeline gets
     the model's own settings unless a call overrides them.

3. Add a ``_load_model``-style gate in the module that owns it, and register
   it in :func:`taters.helpers.library._model_problem` so the import screen is
   exactly as strict as run time.

Nothing else needs editing. In particular the scoring entry point, the
library folder, the naming flow, the wizard row and the provenance gate are
all written against this registry rather than against the three kinds that
happen to exist today.

Naming
------
A model file is named by whoever exported it, which is nearly always the
step's own output name (``ridge__all__age.json``), and that says nothing about
what it predicts. So a model carries a **name** the researcher chose, and
**output names** for the columns it will write: a ridge fitted to predict
age on a blog corpus should be able to land in a new dataset as
``pred_age_blogs``, not as ``pred_age`` colliding with the age already
there. Both are stored in the model file, so they travel with it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (Callable, Dict, List, Mapping, Optional,
                    Sequence, Tuple, Union)

from .atomic import atomic_write

__all__ = ["ModelInfo", "ModelType", "ApplySetting", "MODEL_TYPES",
           "LIBRARY_KIND", "FeaturePlan", "TablePlan", "feature_plan",
           "scorer",
           "describe", "describe_all", "edit_model", "rename_model",
           "output_label", "class_label", "apply_defaults",
           "UnknownModel"]

PathLike = Union[str, Path]

#: The one library folder every kind of saved model imports into.
LIBRARY_KIND = "models"

#: Where a run keeps the private feature tables it measures for a saved
#: model at that model's own settings: ``features/model_inputs/<model
#: slug>/<digest>/<stem>.csv``. Named here because both the composer (which
#: writes there) and the scorer (which tells one model's tables from
#: another's by the slug in the path) need the same word.
MODEL_WORK_DIR = "model_inputs"


class UnknownModel(Exception):
    """The file is not a saved Taters model, or is a kind this build cannot
    describe. Carries the ``kind`` tag it did have, when it had one."""

    def __init__(self, message: str, kind: Optional[str] = None):
        self.kind = kind
        super().__init__(message)


@dataclass(frozen=True)
class ModelInfo:
    """
    What one saved model is, in the terms a screen needs.

    Attributes
    ----------
    path : pathlib.Path
        The file itself.
    type_id : str
        Short kind key: ``"ridge"``, ``"classifier"``, ``"mem"``.
    type_label : str
        What to show a person: ``"ridge"``, ``"MEM topic model"``. Menus put
        it in brackets after the name, so a library listing reads
        ``age_blogs [ridge]`` and the two are told apart at a glance.
    name : str
        The researcher's name for the model, defaulting to the file stem.
    outputs : tuple of str
        The output *labels*, after any renaming -- one per predicted
        outcome, or one per theme.
    columns : tuple of str
        The column names those outputs will actually produce, which is not
        the same list: a classifier writes a predicted class *and* a
        probability per class for one outcome.
    inputs : tuple of str
        Feature columns the model needs by name. Empty when the model works
        from text.
    needs : str
        ``"features"`` or ``"text"`` -- what has to exist upstream before
        this model can score anything.
    needs_tables : tuple of str
        *Which* feature tables, named by the step that writes them. A model
        records its predictors by name, which is enough to score a table
        that already has them and no help at all in getting there: someone
        who fitted on cohesion features and came back a week later had 165
        column names and nothing saying a cohesion step was needed.
    library_kind : str
        The :data:`~taters.helpers.library.KINDS` id this model imports as.
    bulk_outputs : bool
        True when the outputs are a numbered family (MEM themes) rather than
        a handful of named quantities, so renaming them means choosing one
        prefix rather than editing a hundred labels.
    n_outputs : int
        How many outputs there are, for a listing that does not want to
        print a hundred theme names.
    """

    path: Path
    type_id: str
    type_label: str
    name: str
    outputs: Tuple[str, ...]
    columns: Tuple[str, ...]
    inputs: Tuple[str, ...]
    needs: str
    #: The feature tables that supplied this model's predictors, by the name
    #: of the step that writes each one. Empty when the model did not record
    #: them (anything fitted before this was tracked, or a text model, which
    #: needs no features at all).
    needs_tables: Tuple[str, ...] = ()
    #: Per source table, the record of how its columns were measured, as the
    #: model stored it. Empty for a model fitted before this was recorded, and
    #: for a model that needs no features at all. Read through
    #: :func:`requirements` rather than directly, so a new kind of model can
    #: describe what it needs without every consumer learning about it.
    provenance: Mapping[str, Mapping] = field(default_factory=dict)
    library_kind: str = LIBRARY_KIND
    bulk_outputs: bool = False
    #: The control-variable recipe a ridge or classifier was fitted with
    #: (see `stats._controls.rebuild`), read once with everything else. The
    #: scorer used to re-open the model file for it.
    controls: Tuple[Mapping, ...] = ()
    #: Predictors whose absence from a table means "never occurred", so the
    #: scorer fills zero rather than refusing: a part-of-speech tag no text
    #: in the new corpus uses. Everything else absent is a real mismatch.
    zero_when_absent: Tuple[str, ...] = ()
    #: Per categorical outcome (by its fitted name), the classes it predicts
    #: *as they will be written* -- the data's own labels unless relabeled.
    #: Empty for a model that predicts no categories.
    classes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Per categorical outcome, ``{class as fitted: label to write}`` for
    #: the classes that were relabeled. What the Settings screen edits.
    class_names: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: How this model is applied to new data: the registry's defaults with
    #: the model file's ``apply`` block laid over them. Empty for a kind
    #: with no such settings.
    apply: Mapping[str, object] = field(default_factory=dict)
    #: What kind of data the model reads -- ``"text"`` today; audio, image
    #: and video later. From the registry, so a screen can offer a folder of
    #: recordings the audio models and not the text ones.
    modality: str = "text"

    @property
    def n_outputs(self) -> int:
        return len(self.outputs)

    def display(self) -> str:
        """``age_blogs [ridge]`` -- the one-line form every menu uses."""
        return f"{self.name} [{self.type_label}]"


@dataclass(frozen=True)
class ApplySetting:
    """
    One default that governs how a kind of model is applied to new data.

    Stored per model in the file's ``apply`` block and read back through
    :func:`apply_defaults`, so a model carries its own settings wherever it
    goes -- a word-vector model that should weight by types, a predictor
    that fits a small card at batch 8 -- rather than every pipeline having
    to know. ``kind`` says how a typed-in value is read: ``"int"``,
    ``"float"``, ``"bool"``, ``"str"``, or ``"text"`` for free text that is
    not one of a fixed set. ``choices`` restricts a ``str`` to a list.
    """

    default: object
    help: str
    kind: str = "str"
    choices: Optional[Tuple[str, ...]] = None
    #: A check run on the coerced value, raising ValueError with the reason
    #: -- for free text that has a grammar (a concept list).
    validate: Optional[Callable[[object], None]] = None

    def coerce(self, value, name: str):
        """The value as this setting stores it, or a refusal naming both."""
        out = self._coerce(value, name)
        if self.validate is not None:
            self.validate(out)
        return out

    def _coerce(self, value, name: str):
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            word = str(value).strip().lower()
            if word in ("true", "yes", "y", "1", "on"):
                return True
            if word in ("false", "no", "n", "0", "off"):
                return False
            raise ValueError(f"{name} must be yes or no, not {value!r}")
        if self.kind in ("int", "float"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a number, not {value!r}") from None
            if self.kind == "int":
                if number != int(number):
                    raise ValueError(f"{name} must be a whole number, not {value!r}")
                return int(number)
            return number
        if self.kind.startswith("library:"):
            # a set of library entries, kept as what the model needs from
            # them (for dictionaries: the terms and weights themselves)
            if self.kind == "library:dictionaries":
                return _concept_dicts_setting(value)
            return list(value) if isinstance(value, (list, tuple)) else [value]
        text = str(value).strip()
        if self.choices is not None and text not in self.choices:
            raise ValueError(
                f"{name} must be one of {', '.join(self.choices)}, not {text!r}")
        return text


@dataclass(frozen=True)
class ModelType:
    """One kind of model, and how to read and rewrite its names."""

    id: str
    label: str
    kind_tag: str
    needs: str
    #: doc -> (outputs, columns, inputs)
    read: Callable[[dict], Tuple[Sequence[str], Sequence[str], Sequence[str]]]
    #: ``"module:function"`` that scores a new table with this kind of model,
    #: imported only when used. In the registry rather than at the call site
    #: on purpose: the dispatch used to be ``if classifier ... else ridge``,
    #: so a fourth kind of model would have been scored *as a ridge* --
    #: reading coefficients that were not there, silently. Adding a kind now
    #: means adding one entry, and forgetting to is a named refusal.
    score: Optional[str] = None
    #: (doc, new labels) -> None, mutating the doc in place
    write_outputs: Optional[Callable[[dict, Sequence[str]], None]] = None
    bulk_outputs: bool = False
    #: doc -> ``{outcome: [class, ...]}`` as fitted, for a kind that predicts
    #: categories. None for one that does not.
    classes: Optional[Callable[[dict], Mapping[str, Sequence[str]]]] = None
    #: The per-model defaults for applying it; see :class:`ApplySetting`.
    apply_settings: Mapping[str, ApplySetting] = field(default_factory=dict)
    #: What the model reads: ``"text"`` for everything today. Audio, image
    #: and video models are coming, and a screen offering "models" for a
    #: folder of recordings has to be able to tell them from the text ones
    #: -- so the word is here from the start, on every kind, rather than
    #: bolted on to the first non-text one.
    modality: str = "text"


def output_label(doc: dict, outcome: str) -> str:
    """
    The column-name stem for one outcome: its rename, or the outcome itself.

    Read by the scoring functions rather than by the wizard, so a renamed
    model produces renamed columns however it is invoked -- through the app,
    through the API, or from the command line.
    """
    names = doc.get("output_names") or {}
    label = names.get(outcome)
    return str(label) if label else str(outcome)


def class_label(doc: dict, outcome: str, cls) -> str:
    """
    The label written for one predicted class: its relabel, or itself.

    A model fitted on a ``condition`` column coded 0/1 predicts ``"0"`` and
    ``"1"`` -- correct, and unreadable in a results table a month later.
    The relabel lives in the model file (``class_names``), keyed by the
    class as fitted, and every applier writes predictions and the
    ``p_<outcome>_<class>`` columns through this one function, so a model
    relabeled in Settings is relabeled however it is invoked.
    """
    names = (doc.get("class_names") or {}).get(outcome) or {}
    label = names.get(str(cls))
    return str(label) if label else str(cls)


def apply_defaults(doc: dict) -> Dict[str, object]:
    """
    How a model wants to be applied: registry defaults under its own.

    Only the settings its kind registers are read, each coerced to its
    type; a key the file carries from a newer Taters is ignored rather than
    passed on. An applier takes ``apply_defaults(doc)[key]`` wherever its
    caller left the argument as None, so the model's own settings win over
    the function's signature but never over an explicit call.
    """
    spec = MODEL_TYPES.get(str(doc.get("kind")))
    if spec is None or not spec.apply_settings:
        return {}
    stored = doc.get("apply") or {}
    out: Dict[str, object] = {}
    for key, setting in spec.apply_settings.items():
        value = setting.default
        if key in stored:
            try:
                value = setting.coerce(stored[key], key)
            except ValueError:
                value = setting.default    # a damaged value shouldn't block scoring
        out[key] = value
    return out


def _inputs_of(doc: dict) -> list:
    """The columns a run must supply: the raw features behind a reduction,
    or the predictors themselves."""
    return list(doc.get("input_columns") or doc.get("predictors") or [])


def _read_ridge(doc: dict):
    outcomes = list(doc.get("outcomes") or {})
    labels = [output_label(doc, o) for o in outcomes]
    return labels, [f"pred_{label}" for label in labels], _inputs_of(doc)


def _read_classifier(doc: dict):
    outcomes = list(doc.get("outcomes") or {})
    labels = [output_label(doc, o) for o in outcomes]
    columns: List[str] = []
    for outcome, label in zip(outcomes, labels):
        columns += [f"pred_{label}", f"prob_{label}"]
        for c in (doc["outcomes"][outcome].get("classes") or []):
            columns.append(f"p_{label}_{class_label(doc, outcome, c)}")
    return labels, columns, _inputs_of(doc)


def _classifier_classes(doc: dict) -> Dict[str, List[str]]:
    return {o: [str(c) for c in (block.get("classes") or [])]
            for o, block in (doc.get("outcomes") or {}).items()}


def _write_named_outputs(doc: dict, labels: Sequence[str]) -> None:
    outcomes = list(doc.get("outcomes") or {})
    if len(labels) != len(outcomes):
        raise ValueError(
            f"this model has {len(outcomes)} outcome(s) and "
            f"{len(labels)} new name(s) were given")
    # we key by the *fitted* outcome, never by the previous label: otherwise
    # renaming twice would write a second entry and leave the first one
    # deciding the column name.
    doc["output_names"] = {o: str(label)
                           for o, label in zip(outcomes, labels)}


def _read_mem(doc: dict):
    themes = list((doc.get("model") or {}).get("themes") or [])
    return themes, themes, []


def _write_mem(doc: dict, labels: Sequence[str]) -> None:
    themes = list((doc.get("model") or {}).get("themes") or [])
    if len(labels) != len(themes):
        raise ValueError(
            f"this model has {len(themes)} theme(s) and {len(labels)} new "
            f"name(s) were given")
    doc["model"]["themes"] = [str(label) for label in labels]


def _read_topics(doc: dict):
    """LDA and NMF both name their components in ``model.topics``, so one
    pair of functions serves both: the names are the outputs *and* the
    columns, exactly as MEM's themes are."""
    topics = list((doc.get("model") or {}).get("topics") or [])
    return topics, topics, []


def _write_topics(doc: dict, labels: Sequence[str]) -> None:
    topics = list((doc.get("model") or {}).get("topics") or [])
    if len(labels) != len(topics):
        raise ValueError(
            f"this model has {len(topics)} topic(s) and {len(labels)} new "
            f"name(s) were given")
    doc["model"]["topics"] = [str(label) for label in labels]


def _read_word_vectors(doc: dict):
    """Outputs are the dimensions (renamable by prefix); the columns add one
    ``sim_<dictionary>__<category>`` per category of every concept
    dictionary stored in the model's apply block."""
    from ..text._concept_dicts import concept_column

    labels = [str(x) for x in (doc.get("dim_labels") or [])]
    columns = list(labels)
    for d in ((doc.get("apply") or {}).get("concept_dicts") or []):
        for cat in (d.get("categories") or {}):
            columns.append(concept_column(str(d.get("name") or "dict"), str(cat)))
    return labels, columns, []


def _concept_dicts_setting(value):
    """
    The stored form of the concept dictionaries: their content.

    The Settings screen and a call hand over dictionary *files* (paths); the
    model keeps terms and weights, so it measures the same concepts wherever
    it goes. Content already in the stored shape passes through.
    """
    from ..text._concept_dicts import (concept_dicts_from_json,
                                       concept_dicts_to_json,
                                       load_concept_dicts)

    items = list(value) if isinstance(value, (list, tuple)) else (
        [value] if value not in (None, "") else [])
    if not items:
        return []
    if all(isinstance(i, dict) for i in items):
        return concept_dicts_to_json(concept_dicts_from_json(items))
    if all(isinstance(i, (str, Path)) for i in items):
        return concept_dicts_to_json(load_concept_dicts(items))
    raise ValueError("concept_dicts must be dictionary files (paths) or the "
                     "dictionaries as a model stores them")


def _write_word_vectors(doc: dict, labels: Sequence[str]) -> None:
    dims = list(doc.get("dim_labels") or [])
    if len(labels) != len(dims):
        raise ValueError(
            f"this model has {len(dims)} dimension(s) and {len(labels)} new "
            f"name(s) were given")
    doc["dim_labels"] = [str(label) for label in labels]


def _read_text_predictor(doc: dict):
    """Outputs are the outcomes; a category adds a probability column and
    one per class, named through the relabels. A multi-label outcome (an
    imported Hugging Face head where several labels can apply) has no
    winning-class probability; it gets a 1/0 column per label saying whether
    that label cleared the threshold, and a probability column per label."""
    outcomes = dict(doc.get("outcomes") or {})
    labels = [output_label(doc, o) for o in outcomes]
    columns: List[str] = []
    for outcome, label in zip(outcomes, labels):
        columns.append(f"pred_{label}")
        spec = outcomes[outcome] or {}
        classes = [class_label(doc, outcome, c) for c in (spec.get("classes") or [])]
        if spec.get("task") == "classification":
            columns.append(f"prob_{label}")
            columns += [f"p_{label}_{c}" for c in classes]
        elif spec.get("task") == "multi_label":
            columns += [f"pred_{label}_{c}" for c in classes]
            columns += [f"p_{label}_{c}" for c in classes]
    return labels, columns, []


def _text_predictor_classes(doc: dict) -> Dict[str, List[str]]:
    return {o: [str(c) for c in (spec.get("classes") or [])]
            for o, spec in (doc.get("outcomes") or {}).items()
            if (spec or {}).get("task") in ("classification", "multi_label")}


def _unit_interval(value) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"a threshold is a probability, between 0 and 1, not {value!r}")


def _at_least_one(noun: str):
    """
    A validator for a count that has to be a real count.

    ``kind="int"`` only promises the value is a number, so 0 and -4 went
    through and failed much later, in somebody else's words: a batch size of
    zero reaches ``range(0, n, 0)`` and dies with "range() arg 3 must not be
    zero", which names nothing the user typed.
    """
    def check(value) -> None:
        if int(value) < 1:
            raise ValueError(f"{noun} must be at least 1, not {value!r}")
    return check


#: The apply settings a transformer scorer takes, shared by the fine-tuned
#: predictor and an imported Hugging Face classifier: they read text the
#: same way (windows of max_length, batched, averaged per text).
_TRANSFORMER_APPLY = {
    "batch_size": ApplySetting(
        32, "Texts per forward pass when scoring; smaller for a small "
        "card.", kind="int", validate=_at_least_one("a batch size")),
    "max_length": ApplySetting(
        256, "Tokens per window when scoring; a longer text is scored "
        "in windows and given the mean of their predictions. Capped at "
        "what the model itself accepts.",
        kind="int", validate=_at_least_one("a window length")),
    "precision": ApplySetting(
        "auto", "Half precision on a GPU (auto), always full (fp32) or "
        "always half (fp16).", choices=("auto", "fp32", "fp16")),
    "emit_probabilities": ApplySetting(
        True, "Write the probabilities beside each prediction: "
        "prob_<outcome> and one p_<outcome>_<class> per class, or -- for a "
        "multi-label head, which has no winning class -- one "
        "p_<outcome>_<label> per label.", kind="bool"),
}

MODEL_TYPES: Dict[str, ModelType] = {
    "taters-ridge-model": ModelType(
        id="ridge", label="ridge", kind_tag="taters-ridge-model",
        score="taters.stats.ridge:apply_ridge_csv",
        needs="features", read=_read_ridge,
        write_outputs=_write_named_outputs),
    "taters-classifier-model": ModelType(
        id="classifier", label="classifier",
        kind_tag="taters-classifier-model",
        score="taters.stats.classify:apply_classifier_csv",
        needs="features", read=_read_classifier,
        write_outputs=_write_named_outputs, classes=_classifier_classes),
    "taters-mem-model": ModelType(
        id="mem", label="MEM topic model", kind_tag="taters-mem-model",
        score="taters.text.topic_model_mem:apply_mem_model",
        needs="text", read=_read_mem,
        write_outputs=_write_mem, bulk_outputs=True),
    "taters-lda-model": ModelType(
        id="lda", label="LDA topic model", kind_tag="taters-lda-model",
        score="taters.text.topic_model_lda:apply_lda_model",
        needs="text", read=_read_topics,
        write_outputs=_write_topics, bulk_outputs=True),
    "taters-nmf-model": ModelType(
        id="nmf", label="NMF topic model", kind_tag="taters-nmf-model",
        score="taters.text.topic_model_nmf:apply_nmf_model",
        needs="text", read=_read_topics,
        write_outputs=_write_topics, bulk_outputs=True),
    "taters-word-vectors-model": ModelType(
        id="word_vectors", label="word vectors",
        kind_tag="taters-word-vectors-model",
        score="taters.text.word_vectors:apply_word_vectors",
        needs="text", read=_read_word_vectors,
        write_outputs=_write_word_vectors, bulk_outputs=True,
        apply_settings={
            "weighting": ApplySetting(
                "tokens", "How words are averaged into a text's vector: "
                "every occurrence (tokens), each distinct word once (types), "
                "or smooth inverse frequency (sif; the commonest words count "
                "least -- needs the counts a trained model has).",
                choices=("tokens", "types", "sif")),
            "normalize_words": ApplySetting(
                False, "Scale each word vector to unit length before "
                "averaging, so a frequent word's long vector does not "
                "dominate.", kind="bool"),
            "concept_dicts": ApplySetting(
                [], "LIWC-22 dictionaries (.dic, .dicx, .csv) whose categories "
                "become sim_<dictionary>__<category> columns: the cosine "
                "between a text's vector and the weighted mean of the "
                "category's terms. Their terms and weights are stored in the "
                "model.", kind="library:dictionaries"),
        }),
    "taters-text-predictor-model": ModelType(
        id="text_predictor", label="fine-tuned text predictor",
        kind_tag="taters-text-predictor-model",
        score="taters.text.finetune_predictor:apply_text_predictor",
        needs="text", read=_read_text_predictor,
        write_outputs=_write_named_outputs, classes=_text_predictor_classes,
        apply_settings=dict(_TRANSFORMER_APPLY)),
    # a classifier or regressor somebody else trained and published on the
    # Hugging Face hub (a sentiment model, say), imported from the hub cache
    # or a checkpoint folder. its outcomes block has the fine-tuned
    # predictor's exact shape, so the same readers serve both
    "taters-hf-classifier-model": ModelType(
        id="hf_classifier", label="Hugging Face classifier",
        kind_tag="taters-hf-classifier-model",
        score="taters.text.hf_classifier:apply_hf_classifier",
        needs="text", read=_read_text_predictor,
        write_outputs=_write_named_outputs, classes=_text_predictor_classes,
        apply_settings={**_TRANSFORMER_APPLY,
                        "max_length": ApplySetting(
                            512, "Tokens per window when scoring; a longer "
                            "text is scored in windows and given the mean of "
                            "their predictions. Published classifiers were "
                            "trained at up to 512, and this is capped at what "
                            "the model itself accepts.", kind="int",
                            validate=_at_least_one("a window length")),
                        "threshold": ApplySetting(
                            0.5, "For a multi-label head only: the "
                            "probability at which a label counts as present. "
                            "Sets its pred_<outcome>_<label> column to 1 and "
                            "adds it to the joined pred_<outcome>.",
                            kind="float", validate=_unit_interval)}),
}

#: Every model type, keyed by its short id rather than its file tag.
BY_ID: Dict[str, ModelType] = {t.id: t for t in MODEL_TYPES.values()}


def scorer(type_id: str):
    """
    The function that scores a new table with this kind of model.

    Resolved from the registry rather than chosen at the call site, so that
    adding a kind of model is adding a registry entry -- and forgetting the
    entry is this refusal, rather than the model being quietly scored by
    whichever applier happened to be the `else` branch.
    """
    import importlib

    spec = BY_ID.get(type_id)
    if spec is None or not spec.score:
        raise ValueError(
            f"no scoring function is registered for a {type_id!r} model. Add "
            f"`score=` to its entry in MODEL_TYPES -- see 'Adding a kind of "
            f"model' in this module's docstring.")
    module, _, name = spec.score.partition(":")
    return getattr(importlib.import_module(module), name)


def _read_doc(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"no model file at {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise UnknownModel(
            f"{path.name} is not readable as a model file: {e}") from None
    if not isinstance(doc, dict):
        raise UnknownModel(f"{path.name} is not a Taters model file at all.")
    return doc


def describe(model_json: PathLike) -> ModelInfo:
    """
    Describe one saved model, or say why it cannot be described.

    Deliberately cheap and structural: it reads the file's own declaration
    of what it is and does not vet the matrices. Vetting is the scoring
    loader's job (and the library import gate delegates to it), because a
    listing that had to fully validate every model would be slow and would
    hide a damaged model behind a blank row instead of naming it.
    """
    path = Path(model_json)
    doc = _read_doc(path)
    tag = doc.get("kind")
    spec = MODEL_TYPES.get(tag) if isinstance(tag, str) else None
    if spec is None:
        raise UnknownModel(
            f"{path.name} is "
            + (f"a {tag!r} file, which is not a model this build can score"
               if tag else "not a Taters model file at all")
            + ". Models are written by the topic-model, prediction and "
              "classification steps.",
            kind=tag if isinstance(tag, str) else None)
    outputs, columns, inputs = spec.read(doc)
    fitted = dict(spec.classes(doc)) if spec.classes is not None else {}
    relabels = doc.get("class_names") or {}
    return ModelInfo(
        classes={o: tuple(class_label(doc, o, c) for c in cs)
                 for o, cs in fitted.items()},
        class_names={o: {str(k): str(v) for k, v in (m or {}).items()}
                     for o, m in relabels.items() if o in fitted},
        apply=apply_defaults(doc),
        path=path, type_id=spec.id, type_label=spec.label,
        name=str(doc.get("name") or path.stem),
        outputs=tuple(outputs), columns=tuple(columns), inputs=tuple(inputs),
        needs=spec.needs, bulk_outputs=spec.bulk_outputs, modality=spec.modality,
        provenance=((doc.get("needs") or {}).get("feature_provenance") or {}),
        controls=tuple(doc.get("controls") or ()),
        zero_when_absent=tuple(str(c) for c in
                               (doc.get("zero_when_absent") or ())),
        needs_tables=tuple(
            str(s) for s in ((doc.get("needs") or {}).get("feature_tables")
                             or ())))


def describe_all(paths: Sequence[PathLike]) -> List[ModelInfo]:
    """
    Describe every model that can be described, skipping the rest silently.

    Used by listings, where one unreadable file in a library folder must not
    take the whole menu down -- the import gate already refused anything
    broken, so a file that fails here arrived some other way.
    """
    out = []
    for path in paths:
        try:
            out.append(describe(path))
        except (UnknownModel, FileNotFoundError, OSError):
            continue
    return out


def edit_model(model_json: PathLike, *, name: Optional[str] = None,
               outputs: Optional[Sequence[str]] = None,
               prefix: Optional[str] = None,
               class_names: Optional[Mapping[str, Mapping[str, str]]] = None,
               apply: Optional[Mapping[str, object]] = None) -> ModelInfo:
    """
    Change what a model is called, what it writes, and how it is applied.

    Parameters
    ----------
    model_json
        The model file, rewritten in place (atomically).
    name
        The researcher's name for the model. Shown in every menu.
    outputs
        One new label per output, in the order :func:`describe` lists them.
    prefix
        For a model whose outputs are a numbered family (MEM themes), the
        stem to number from: ``"fb_topics"`` gives ``fb_topics_1`` upward.
        Cannot be combined with ``outputs``.
    class_names
        ``{outcome: {class as fitted: label to write}}`` for a model that
        predicts categories. Merged with what the file already has, one
        outcome at a time: a class left out keeps its current label, and
        giving a class its own name (or a blank) removes its relabel. Two
        classes of one outcome cannot end up with the same label.
    apply
        Values for the settings its kind registers (:attr:`ModelType.
        apply_settings`), coerced and checked; an unknown key is refused
        naming the ones there are. Merged with the file's ``apply`` block.

    Returns
    -------
    ModelInfo
        The model as it now reads.

    Notes
    -----
    Every edit rewrites the file rather than a sidecar, so a model that is
    copied, zipped or emailed keeps the names and settings it was given --
    a sidecar would be left behind by every one of those, and the columns
    would quietly revert.
    """
    path = Path(model_json)
    doc = _read_doc(path)
    info = describe(path)
    spec = BY_ID[info.type_id]

    if class_names is not None:
        fitted = dict(spec.classes(doc)) if spec.classes is not None else {}
        if not fitted:
            raise ValueError(
                f"a {info.type_label} model predicts no categories, so it "
                f"has no classes to relabel")
        current = {o: dict(m or {}) for o, m in
                   (doc.get("class_names") or {}).items()}
        for outcome, mapping in class_names.items():
            if outcome not in fitted:
                raise ValueError(
                    f"this model has no categorical outcome {outcome!r} "
                    f"(it has {', '.join(fitted) or 'none'})")
            table = current.setdefault(outcome, {})
            for cls, label in (mapping or {}).items():
                cls = str(cls)
                if cls not in [str(c) for c in fitted[outcome]]:
                    raise ValueError(
                        f"outcome {outcome!r} has no class {cls!r} (its "
                        f"classes are {', '.join(map(str, fitted[outcome]))})")
                text = str(label if label is not None else "").strip()
                if not text or text == cls:
                    table.pop(cls, None)
                else:
                    table[cls] = _valid_label(text, what="class label")
            written = [table.get(str(c), str(c)) for c in fitted[outcome]]
            clashes = sorted({w for w in written if written.count(w) > 1})
            if clashes:
                raise ValueError(
                    f"two classes of {outcome!r} cannot share the label "
                    f"{clashes[0]!r} -- the results could not tell them apart")
        doc["class_names"] = {o: m for o, m in current.items() if m}
        if not doc["class_names"]:
            doc.pop("class_names", None)

    if apply is not None:
        if not spec.apply_settings:
            raise ValueError(
                f"a {info.type_label} model has no settings for how it is "
                f"applied")
        block = dict(doc.get("apply") or {})
        for key, value in apply.items():
            setting = spec.apply_settings.get(str(key))
            if setting is None:
                raise ValueError(
                    f"{key!r} is not a setting of a {info.type_label} model "
                    f"(they are: {', '.join(spec.apply_settings)})")
            block[str(key)] = setting.coerce(value, str(key))
        doc["apply"] = block

    if outputs is not None and prefix is not None:
        raise ValueError(
            "give either one label per output or a single prefix, not both")
    if prefix is not None:
        if not info.bulk_outputs:
            raise ValueError(
                f"a {info.type_label} model has {info.n_outputs} named "
                f"output(s), so name them individually rather than by "
                f"prefix")
        outputs = [f"{prefix}_{i + 1}" for i in range(info.n_outputs)]
    if outputs is not None:
        labels = [_valid_label(label) for label in outputs]
        duplicates = sorted({label for label in labels
                             if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                f"two outputs cannot share a name ({', '.join(duplicates)}) "
                f"-- one column would overwrite the other")
        if spec.write_outputs is None:
            raise ValueError(
                f"a {info.type_label} model's output names cannot be changed")
        spec.write_outputs(doc, labels)
    if name is not None:
        doc["name"] = _valid_label(name, what="model name")

    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return replace(describe(path), path=path)


def rename_model(model_json: PathLike, *, name: Optional[str] = None,
                 outputs: Optional[Sequence[str]] = None,
                 prefix: Optional[str] = None) -> ModelInfo:
    """Name a model and its output columns -- :func:`edit_model` without
    the class labels and apply settings. Kept for the callers that only
    ever name things."""
    return edit_model(model_json, name=name, outputs=outputs, prefix=prefix)


def _valid_label(label: str, what: str = "output name") -> str:
    """
    A label that can be a CSV column and survive a round trip.

    Rejected rather than silently mangled: a name containing a comma or a
    newline would split into two columns when the results are read back,
    and the researcher would find the mistake in their analysis rather than
    here.
    """
    text = str(label).strip()
    if not text:
        raise ValueError(f"the {what} cannot be blank")
    bad = [c for c in ",\t\r\n\"" if c in text]
    if bad:
        shown = ", ".join(repr(c) for c in bad)
        raise ValueError(
            f"the {what} cannot contain {shown} -- it becomes a column "
            f"heading, and that would split the row when the results are "
            f"read back")
    return text


# ---------------------------------------------------------------------------
# What a run has to do to reproduce a model's features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TablePlan:
    """
    One feature table a model needs, and exactly how to measure it.

    Attributes
    ----------
    stem : str
        The table's name **at fit time**. Load-bearing: the stem is what fixed
        the model's predictor names, so a replayed table has to be written
        under it or the names stop matching.
    target : str
        ``"module:function"`` of the analyzer, as recorded. Byte-identical to
        the producing recipe's ``target``, which is how a composer joins the
        two without knowing anything about models.
    instrument : dict
        The measuring settings, as literal values. Complete, because they were
        recorded *after* defaults were applied -- which is the whole reason a
        replay is constructible rather than a guess.
    assets : dict
        ``{parameter: [{name, sha256, text}]}`` for the word lists it used,
        carried inside the model so the replay works on a machine that never
        had them.
    digest : str
        The instrument digest. Belongs in the private output *directory*: two
        models needing identical settings then share one extraction for free,
        and a re-fit under the same model name cannot short-circuit onto the
        previous private table.
    grain : dict
        What one row was at fit time. Informational only -- never compared.
    replay : dict or None
        ``{"target": "module:function", "assets": {parameter: [...]}}`` when
        the table cannot be measured the same way twice and must instead be
        produced by *applying* what the fit-time step fitted -- a topic
        model's saved themes. The assets carry the fitted file's text, the
        same way ``assets`` carries word lists. ``None`` for a table whose
        settings alone reproduce it.
    """

    stem: str
    target: str
    instrument: Mapping
    assets: Mapping
    digest: str
    grain: Mapping
    replay: Optional[Mapping] = None


@dataclass(frozen=True)
class FeaturePlan:
    """
    Everything a run needs in order to score with one model, as plain data.

    Deliberately free of paths, recipes and pipeline templates: it is built
    from the model file alone, so the composer stays pure and a test can hand
    a synthetic plan straight in.

    ``problems`` is the honest half. A plan with problems cannot be replayed,
    and each entry is a sentence a researcher can act on -- which is better
    than a partial replay, because a partial replay is a wrong answer.
    """

    model: str
    slug: str
    tables: Tuple[TablePlan, ...] = ()
    problems: Tuple[str, ...] = ()
    #: The model file itself. A private extraction that needs the model's
    #: word lists or fitted files reads them from here at run time, so the
    #: preset stays a page of settings rather than a copy of every list.
    path: str = ""
    #: The spreadsheet columns the model was fitted with as controls. They
    #: are not features -- no step produces them -- so a run has to carry
    #: them from the spreadsheet itself, and the composer needs to know.
    controls: Tuple[str, ...] = ()

    @property
    def replayable(self) -> bool:
        return bool(self.tables) and not self.problems


def feature_plan(info: ModelInfo) -> FeaturePlan:
    """
    Read a model's own account of the features it was fitted on.

    Returns a :class:`FeaturePlan`. A model that reads text needs no features
    and gets an empty plan with no problems -- it re-derives its own
    instrument, so there is nothing for a run to arrange.
    """
    import json

    name_slug = slug(info.name)
    controls = tuple(sorted({str(e.get("column")) for e in info.controls
                             if e.get("column")}))
    if info.needs != "features":
        return FeaturePlan(model=info.display(), slug=name_slug,
                           path=str(info.path))

    recorded = dict(info.provenance or {})
    if not recorded:
        return FeaturePlan(
            model=info.display(), slug=name_slug, path=str(info.path),
            controls=controls,
            problems=(f"{info.display()} does not record how its features "
                      f"were measured, so they cannot be reproduced.",))

    try:
        with Path(info.path).open("r", encoding="utf-8") as fh:
            carried = json.load(fh).get("assets") or {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        carried = {}

    tables, problems = [], []
    for stem, rec in sorted(recorded.items()):
        if rec.get("state") != "recorded":
            problems.append(
                f"{info.display()} was fitted on the {stem!r} feature table "
                f"and has no record of how it was measured, so that table "
                f"cannot be reproduced.")
            continue
        assets, missing = _carried(rec.get("assets"), carried)
        replay = None
        if rec.get("replay"):
            fitted, gone = _carried((rec["replay"] or {}).get("assets"),
                                    carried)
            missing += gone
            replay = {"target": str((rec["replay"] or {}).get("call") or ""),
                      "assets": fitted}
        if missing:
            problems.append(
                f"{info.display()} used word list(s) it does not carry "
                f"({', '.join(sorted(set(missing)))}), so the {stem!r} table "
                f"cannot be reproduced. Fit the model again with a build of "
                f"Taters that embeds them.")
            continue
        tables.append(TablePlan(
            stem=str(stem), target=str(rec.get("call") or ""),
            instrument=dict(rec.get("instrument") or {}),
            assets=assets,
            digest=str((rec.get("digests") or {}).get("instrument") or ""),
            grain=dict(rec.get("grain") or {}), replay=replay))
    return FeaturePlan(model=info.display(), slug=name_slug,
                       tables=tuple(tables), problems=tuple(problems),
                       path=str(info.path), controls=controls)


def _carried(wanted: Optional[Mapping], carried: Mapping):
    """
    The model's own copy of each asset a record names, with its text.

    Returns ``({parameter: [entries]}, [names not carried])``. Matched by
    name inside the model's asset copy, which is keyed the same way the
    record is; an entry without text is as good as absent, since a replay
    has to *write* the file.
    """
    assets, missing = {}, []
    for key, entries in (wanted or {}).items():
        names = [e.get("name") for e in entries]
        have = {e.get("name"): e for e in carried.get(key, [])}
        resolved = [have[n] for n in names if n in have and have[n].get("text")]
        if len(resolved) != len(names):
            missing += [n for n in names if n not in have
                        or not have[n].get("text")]
        assets[key] = resolved
    return assets, missing


def slug(name: str, fallback: str = "model") -> str:
    """
    Filesystem- and template-safe: artifact references split on ``.`` and
    ``:``, and a ``save_as`` built from this must not contain either.

    One spelling for every model-shaped file name. Three private copies had
    grown -- here, in ridge (fallback "set") and in score_model -- and two of
    them kept the dot the third refused, so the same set name could produce
    two different file names depending on which module wrote it.
    """
    import re

    return re.sub(r"[^0-9A-Za-z_-]+", "-", str(name)).strip("-") or fallback


#: A language model adapted to a corpus by "Train a model": not a scoring
#: model (it predicts nothing) and so not a `ModelType`, but a library entry
#: with the same manifest-plus-payload shape, kept in the `encoders` kind.
ENCODER_KIND = "taters-encoder"
ENCODER_FORMAT = 1
#: The files a saved encoder's payload folder has to hold to be loadable.
_ENCODER_FILES = ("config.json",)
_ENCODER_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
_ENCODER_TOKENIZER = ("tokenizer.json", "vocab.txt", "vocab.json",
                      "sentencepiece.bpe.model", "spm.model", "tokenizer.model")


def encoder_problem(model_json: PathLike) -> str:
    """
    Why this file is not a usable text encoder, or "" when it is.

    Structural and torch-free: the manifest's kind and format, and a payload
    folder beside it holding a config, weights and a tokenizer. Loading the
    weights is the extractor's job, minutes later, on the device it chose.
    """
    path = Path(model_json)
    try:
        doc = _read_doc(path)
    except UnknownModel as e:
        return str(e)
    except FileNotFoundError as e:
        return str(e)
    if doc.get("kind") != ENCODER_KIND:
        return (f"{path.name} is not a text encoder "
                f"(a {doc.get('kind')!r} file).")
    try:
        fmt = int(doc.get("format", 0))
    except (TypeError, ValueError):
        return f"{path.name} has no readable format number."
    if fmt > ENCODER_FORMAT:
        return (f"{path.name} was written by a newer Taters (format {fmt}); "
                f"this build reads format {ENCODER_FORMAT}. Update Taters.")
    from .library import payload_of

    declared = payload_of(path)
    if not declared:
        return (f"{path.name} names no weights folder beside it. The "
                f"encoder's weights folder has to travel with the file.")
    folder = declared[0]
    if not folder.is_dir():
        return (f"{path.name}'s weights folder {folder.name} is not beside "
                f"it. Copy the model file and its weights together.")
    names = {f.name for f in folder.iterdir()}
    for needed in _ENCODER_FILES:
        if needed not in names:
            return f"{folder.name} has no {needed}; the encoder is incomplete."
    if not names & set(_ENCODER_WEIGHTS):
        return f"{folder.name} has no model weights; the encoder is incomplete."
    if not names & set(_ENCODER_TOKENIZER):
        return f"{folder.name} has no tokenizer; the encoder is incomplete."
    return ""


def describe_encoder(model_json: PathLike) -> Tuple[str, str]:
    """One encoder's row: its name and base model, then what adapting did."""
    path = Path(model_json)
    doc = _read_doc(path)
    name = str(doc.get("name") or path.stem)
    base = str(doc.get("base_model") or "?")
    bits = []
    layers = doc.get("num_hidden_layers")
    if layers:
        bits.append(f"{layers} layers")
    ev = doc.get("evaluation") or {}
    before, after = ev.get("perplexity_before"), ev.get("perplexity_after")
    if before is not None and after is not None:
        bits.append(f"perplexity {float(before):.1f} → {float(after):.1f}")
    return f"{name} [{base}]", " · ".join(bits)


def models_produced(folder: PathLike) -> List[Tuple[Path, str]]:
    """
    The model files a run left behind, each with a label for a menu.

    Scoring models of every registered kind and adapted encoders alike,
    found by their manifests (never inside a payload folder), so the finish
    screen and the training task can offer to add them to the library.
    """
    from .library import model_files

    out: List[Tuple[Path, str]] = []
    for path in model_files(Path(folder)):
        try:
            doc = _read_doc(path)
        except (UnknownModel, FileNotFoundError, OSError):
            continue
        if doc.get("kind") == ENCODER_KIND:
            label, _note = describe_encoder(path)
            out.append((path, f"{label} (text encoder)"))
            continue
        try:
            info = describe(path)
        except (UnknownModel, FileNotFoundError, OSError):
            continue
        out.append((path, info.display()))
    return out


def library_kind_for(model_json: PathLike) -> str:
    """Which library kind a model file belongs in: ``encoders`` for an
    adapted encoder, ``models`` for anything that scores."""
    try:
        doc = _read_doc(Path(model_json))
    except (UnknownModel, FileNotFoundError, OSError):
        return "models"
    return "encoders" if doc.get("kind") == ENCODER_KIND else "models"


def one_model_path(model_json) -> Path:
    """
    Exactly one model file, or an actionable refusal.

    The library hands steps a folder (or a list) meaning "everything of this
    kind", which is right for dictionaries and wrong for a model: scoring
    with an unspecified one of three is not a thing anyone means. A single
    file passes through, a folder or a list resolves to its ``.json`` files,
    and anything other than exactly one refuses with the fix named.

    Owned here because it is about model files, not about any one kind of
    model: ridge and the topic model each carried a copy, with different
    refusal wording and one of them letting a folder through unresolved.
    """
    from .library import model_files

    values = list(model_json) if isinstance(model_json, (list, tuple)) \
        else [model_json]
    found: List[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            # not `rglob`: a checkpoint folder inside a model's payload holds
            # a config.json and a tokenizer.json that aren't model files.
            found += model_files(path)
        else:
            found.append(path)
    if len(found) == 1:
        return found[0]
    if not found:
        raise ValueError(
            f"no model file found at {model_json!r}. Import one into your "
            f"library, or point the step at one.")
    names = ", ".join(p.name for p in found[:6])
    raise ValueError(
        f"{len(found)} model files found ({names}) and a step can only use "
        f"one. Pick it in the step's options.")
