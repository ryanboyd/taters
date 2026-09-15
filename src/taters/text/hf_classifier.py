"""
Score texts with a classifier or regressor from the Hugging Face hub.

Thousands of finished text models are published on the hub -- sentiment,
emotion, stance, toxicity, formality, a regression head trained on some
rating -- and many researchers already have a few in their Hugging Face
cache from other work. This module lets one of those be *imported* into
the library as a saved model and then applied exactly like a model Taters
trained itself: it appears on "Score with models I already have", alone or
alongside a ridge and a fine-tuned predictor, and writes the same shape of
columns (``pred_<outcome>``, ``prob_<outcome>``, ``p_<outcome>_<class>``;
one ``pred_<outcome>`` for a regression head; a ``pred_<outcome>_<class>``
per label, 1 or 0, for a multi-label head).

What is accepted
----------------
A checkpoint whose architecture is sequence classification or regression
(``*ForSequenceClassification`` with ``num_labels`` > 1, or
``num_labels == 1`` / ``problem_type == "regression"``) on a *text* model.
A multi-label head (several labels can be true at once: an emotion model
that tags a text as both *joy* and *surprise*) is accepted too, and scored
with an independent probability per label and a threshold. Anything else
is refused by name with what it is: a bare encoder or masked language
model (adapt or fine-tune it with "Train a model" instead), a token
classifier, or a classifier whose backbone reads audio, images or video.
Those last are refused *because of their modality*, and the manifest
records ``modality`` for
the same reason: audio and image models are coming, and the day they
arrive a screen offering "models" for a folder of recordings must be able
to tell them from the text ones.

Where the model lives
---------------------
Imported from a folder, the checkpoint is copied into the library as the
manifest's payload (``<stem>.hfmodel/``), so the model travels with the
library like every other saved model. Imported by hub name, the manifest
records the name and the weights are read from the model cache
(downloaded on first use if they are not there yet).

How it scores
-------------
The fine-tuned predictor's reading of text: a text longer than
``max_length`` tokens is cut into windows, every text's windows are
batched together, and the text's prediction is the mean over its windows
-- of softmax probabilities for a classifier, of sigmoid probabilities per
label for a multi-label head (which writes one ``pred_<label>`` column per
label, 1 or 0 against the model's ``threshold``, plus a ``pred_`` column
joining the labels that cleared it with ``|``), of raw outputs for a
regressor.
Labels come from the checkpoint's own ``id2label`` and can be renamed per
model in Settings like any other class labels.
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, List, Literal, Optional, Sequence, Tuple,
                    Union)

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.model_spec import class_label, one_model_path, output_label, slug
from ..helpers.progress import announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready

__all__ = ["HF_CLASSIFIER_KIND", "CheckpointInfo", "inspect_checkpoint",
           "checkpoint_problem", "cached_hub_classifiers", "import_hf_classifier",
           "apply_hf_classifier"]

PathLike = Union[str, Path]

HF_CLASSIFIER_KIND = "taters-hf-classifier-model"
HF_CLASSIFIER_FORMAT = 1
#: The folder an imported checkpoint is copied into, beside its manifest.
PAYLOAD_SUFFIX = ".hfmodel"

#: Backbone model types that read something other than text. A
#: ``ForSequenceClassification`` head sits on some of these too (wav2vec2 has
#: one), so the *head* alone cannot say what a model reads; the backbone can.
AUDIO_MODEL_TYPES = frozenset({
    "wav2vec2", "wav2vec2-bert", "wav2vec2-conformer", "hubert", "wavlm",
    "whisper", "audio-spectrogram-transformer", "data2vec-audio", "unispeech",
    "unispeech-sat", "sew", "sew-d", "clap", "encodec", "seamless_m4t",
    "speech_to_text", "speecht5"})
IMAGE_MODEL_TYPES = frozenset({
    "vit", "vit_mae", "vit_msn", "convnext", "convnextv2", "resnet", "beit",
    "deit", "swin", "swinv2", "dinov2", "clip", "siglip", "efficientnet",
    "mobilenet_v1", "mobilenet_v2", "mobilevit", "regnet", "segformer", "detr",
    "yolos", "data2vec-vision", "levit", "poolformer", "pvt", "bit", "cvt",
    "focalnet", "dpt"})
VIDEO_MODEL_TYPES = frozenset({"videomae", "vivit", "timesformer", "xclip"})


@dataclass(frozen=True)
class CheckpointInfo:
    """What a checkpoint is, read from its config alone."""

    #: The hub id or folder path, as given.
    source: str
    #: ``"hub"`` or ``"folder"``.
    kind: str
    model_type: str
    architectures: Tuple[str, ...]
    #: ``"text"``, ``"audio"``, ``"image"``, ``"video"`` or ``"unknown"``.
    modality: str
    #: ``"classification"``, ``"regression"``, ``"multi_label"`` or ``"other"``
    #: (no sequence head at all).
    task: str
    #: The class labels in head order, for a classifier (single- or multi-label).
    labels: Tuple[str, ...]
    #: The longest input the backbone accepts.
    max_length: int

    @property
    def short_name(self) -> str:
        """``cardiffnlp/twitter-roberta-base-sentiment`` -> the part after the
        slash; a folder -> its name."""
        text = self.source.rstrip("/\\")
        return text.replace("\\", "/").rsplit("/", 1)[-1] or "classifier"


def _modality_of(model_type: str, architectures: Sequence[str]) -> str:
    kind = str(model_type or "").lower()
    if kind in AUDIO_MODEL_TYPES or any("Audio" in a for a in architectures):
        return "audio"
    if kind in VIDEO_MODEL_TYPES or any("Video" in a for a in architectures):
        return "video"
    if kind in IMAGE_MODEL_TYPES or any("Image" in a for a in architectures):
        return "image"
    return "text" if kind else "unknown"


def _task_of(cfg: dict) -> Tuple[str, Tuple[str, ...]]:
    archs = [str(a) for a in (cfg.get("architectures") or [])]
    if not any(a.endswith("ForSequenceClassification") for a in archs):
        return "other", ()
    id2label = cfg.get("id2label") or {}
    n = int(cfg.get("num_labels") or len(id2label) or 0)
    problem = str(cfg.get("problem_type") or "")
    if problem == "regression" or n == 1:
        return "regression", ()
    # id2label's keys arrive as strings from JSON, as ints from a live config.
    # we want the labels in head order, whatever the keys look like
    labels = []
    for i in range(n):
        label = id2label.get(str(i), id2label.get(i, f"LABEL_{i}"))
        labels.append(str(label))
    if problem == "multi_label_classification":
        return "multi_label", tuple(labels)
    return "classification", tuple(labels)


def _config_of(source: str) -> Tuple[dict, str]:
    """A checkpoint's config and whether it came from a folder or the hub."""
    path = Path(source)
    if path.is_dir():
        cfg_path = path / "config.json"
        if not cfg_path.is_file():
            raise ValueError(f"{path} holds no config.json, so it is not a "
                             f"transformers checkpoint folder.")
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8")), "folder"
        except ValueError as e:
            raise ValueError(f"{cfg_path} is not readable JSON ({e}).") from None
    if path.suffix.lower() == ".json":
        raise ValueError(f"{source} is a file, not a checkpoint folder or a "
                         f"Hugging Face model name.")
    # the cache first (offline, instant), the hub only when it isn't there yet
    from ._transformer_common import cached_hub_configs

    for hub_id, cfg in cached_hub_configs():
        if hub_id == source:
            return cfg, "hub"
    try:
        from transformers import AutoConfig

        from ..helpers.settings import model_cache_dir

        cfg = AutoConfig.from_pretrained(source, cache_dir=str(model_cache_dir()))
    except Exception as e:
        raise ValueError(f"{source!r} could not be read as a Hugging Face model "
                         f"({type(e).__name__}: {e}). Check the name, or that this "
                         f"machine can reach the hub.") from None
    return cfg.to_dict(), "hub"


def inspect_checkpoint(source: PathLike) -> CheckpointInfo:
    """
    What a checkpoint folder or hub name is, from its config.

    Reads the config only -- no weights -- so it is cheap enough for a
    picker to call on every cached model. A hub name not in the cache
    fetches just its config.
    """
    text = str(source).strip()
    if not text:
        raise ValueError("no model was named: give a Hugging Face model name or "
                         "a checkpoint folder")
    cfg, kind = _config_of(text)
    archs = tuple(str(a) for a in (cfg.get("architectures") or []))
    model_type = str(cfg.get("model_type") or "")
    task, labels = _task_of(cfg)
    max_length = int(cfg.get("max_position_embeddings") or 512)
    return CheckpointInfo(source=text, kind=kind, model_type=model_type,
                          architectures=archs,
                          modality=_modality_of(model_type, archs), task=task,
                          labels=labels, max_length=max_length)


def checkpoint_problem(info: CheckpointInfo) -> str:
    """Why this checkpoint cannot be imported as a text classifier, or ``""``."""
    what = info.short_name
    if info.modality in ("audio", "image", "video"):
        return (f"{what} is an {info.modality} model ({info.model_type}). Taters "
                f"scores text models today; {info.modality} models are on the way, "
                f"and this one will be importable then.")
    if info.task == "other":
        heads = ", ".join(info.architectures) or "no architecture named"
        return (f"{what} has no classification or regression head ({heads}). A "
                f"base encoder or masked language model is something to adapt or "
                f"fine-tune under 'Train a model', not something to score with.")
    return ""


def cached_hub_classifiers(cache_dir: Optional[PathLike] = None
                           ) -> List[Tuple[str, str]]:
    """
    The text classifiers and regressors already downloaded to this machine:
    ``(hub id, note)``, the note naming the classes (or "regression").
    Offline: it reads the cache's configs and nothing else.
    """
    from ._transformer_common import cached_hub_configs

    out: List[Tuple[str, str]] = []
    for hub_id, cfg in cached_hub_configs(cache_dir):
        archs = tuple(str(a) for a in (cfg.get("architectures") or []))
        task, labels = _task_of(cfg)
        if task not in ("classification", "regression", "multi_label"):
            continue
        if _modality_of(str(cfg.get("model_type") or ""), archs) != "text":
            continue
        shown = ", ".join(labels[:6]) + ("…" if len(labels) > 6 else "")
        if task == "regression":
            note = "regression head · one number per text · downloaded"
        elif task == "multi_label":
            note = (f"multi-label classifier · {len(labels)} labels, several can "
                    f"apply: {shown} · downloaded")
        else:
            note = f"text classifier · {len(labels)} classes: {shown} · downloaded"
        out.append((hub_id, note))
    return out


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_hf_classifier(*, source: PathLike, out_dir: PathLike,
                         name: Optional[str] = None,
                         outcome: Optional[str] = None,
                         copy: bool = True) -> Path:
    """
    Write a Taters model manifest for a Hugging Face classifier.

    Parameters
    ----------
    source
        A checkpoint folder, or a Hugging Face model name.
    out_dir
        Where the manifest (``<name>.json``) goes. A folder checkpoint is
        copied beside it as ``<name>.hfmodel/`` (the payload the library
        moves with the manifest) unless ``copy`` is False, in which case the
        folder is referenced where it is.
    name
        The model's name in menus and its file stem; default the
        checkpoint's short name.
    outcome
        What the model predicts, as the stem of its columns
        (``pred_<outcome>``); default the short name. Renamable later.

    Returns
    -------
    Path
        The manifest. Refuses, by name, anything that is not a single-label
        text classifier or a regression head -- see :func:`checkpoint_problem`.
    """
    info = inspect_checkpoint(source)
    problem = checkpoint_problem(info)
    if problem:
        raise ValueError(problem)
    stem = slug(name or info.short_name, fallback="classifier")
    outcome = str(outcome or "").strip() or slug(info.short_name, fallback="label")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"{stem}.json"

    if info.task == "regression":
        spec: dict = {"task": "regression"}
    else:
        # "classification" or "multi_label", each with its labels in head order
        spec = {"task": info.task, "classes": list(info.labels)}
    doc = {
        "kind": HF_CLASSIFIER_KIND, "format": HF_CLASSIFIER_FORMAT,
        "name": stem,
        "modality": info.modality,
        "source": {"kind": info.kind,
                   "hub_id": info.source if info.kind == "hub" else None,
                   "folder": None},
        "outcomes": {outcome: spec},
        "encoder": {"model_type": info.model_type,
                    "architectures": list(info.architectures),
                    "max_length": info.max_length},
        "apply": {"batch_size": 32, "max_length": min(info.max_length, 512),
                  "precision": "auto", "emit_probabilities": True,
                  # only a multi-label head reads this: a label counts as
                  # present when its probability clears it
                  "threshold": 0.5},
    }
    if info.kind == "folder":
        src = Path(info.source)
        if copy:
            payload = out_dir / f"{stem}{PAYLOAD_SUFFIX}"
            if payload.exists():
                shutil.rmtree(payload)
            shutil.copytree(src, payload)
            doc["payload"] = [payload.name]
        else:
            doc["source"]["folder"] = str(src.resolve())
    with atomic_write(manifest, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# Load and apply
# ---------------------------------------------------------------------------

def _load_model(model_json: PathLike) -> dict:
    """
    Load and *vet* a classifier manifest: the import gate's authority too.

    Returns ``{"doc", "source"}``, the source being the folder to load from
    or the hub name. Structural and offline: it does not touch the weights,
    so a hub model not yet downloaded still passes (the download happens
    when it is first applied).
    """
    from ._transformer_common import _payload_folder, torch_missing_reason

    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    path = Path(model_json)
    if not path.exists():
        raise FileNotFoundError(f"model_json not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    kind = doc.get("kind")
    if kind != HF_CLASSIFIER_KIND:
        what = f"a {kind!r} file" if kind else "not a Taters model file at all"
        raise ValueError(f"{path.name} is not an imported Hugging Face classifier ({what}).")
    if int(doc.get("format", 0)) > HF_CLASSIFIER_FORMAT:
        raise ValueError(f"{path.name} was written by a newer Taters (format "
                         f"{doc.get('format')}; this build reads up to "
                         f"{HF_CLASSIFIER_FORMAT}). Update Taters to use it.")

    def _broken(why: str) -> ValueError:
        return ValueError(f"{path.name} is damaged or incomplete: {why}. Import the "
                          f"classifier again to write a fresh one.")

    outcomes = doc.get("outcomes")
    if not isinstance(outcomes, dict) or len(outcomes) != 1:
        raise _broken("it should name exactly one outcome")
    (spec,) = outcomes.values()
    if (spec or {}).get("task") not in ("classification", "regression", "multi_label"):
        raise _broken("its outcome is neither a classification nor a regression")
    if spec.get("task") == "classification" and len(spec.get("classes") or []) < 2:
        raise _broken("its classifier names fewer than two classes")
    if spec.get("task") == "multi_label" and not spec.get("classes"):
        raise _broken("its multi-label head names no labels")
    if doc.get("modality", "text") != "text":
        raise ValueError(f"{path.name} is a {doc.get('modality')} model; this build "
                         f"scores text models only.")

    source_doc = doc.get("source") or {}
    if doc.get("payload"):
        folder = _payload_folder(path, doc, "models")
        if not (folder / "config.json").is_file():
            raise _broken(f"{folder.name} holds no config.json")
        source = str(folder)
    elif source_doc.get("folder"):
        folder = Path(str(source_doc["folder"]))
        if not (folder / "config.json").is_file():
            raise ValueError(f"{path.name} points at {folder}, which holds no "
                             f"checkpoint any more. Import the classifier again.")
        source = str(folder)
    elif source_doc.get("hub_id"):
        source = str(source_doc["hub_id"])
    else:
        raise _broken("it names neither a payload folder nor a Hugging Face model")
    return {"doc": doc, "source": source}


def _load_classifier(source: str, device: str, verbose: bool):
    """The tokenizer and the classification model on the resolved device."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ..helpers.gpu import resolve_device
    from ..helpers.settings import model_cache_dir

    def probe():
        torch.zeros(1, device="cuda") + 1

    device_name, reason = resolve_device(device, backend="torch", probe=probe)
    if verbose:
        print(f"[transformers] loading {source} on {device_name}")
        if reason:
            print(f"[transformers] {reason}")
    cache = {} if Path(source).is_dir() else {"cache_dir": str(model_cache_dir())}
    tokenizer = AutoTokenizer.from_pretrained(source, **cache)
    model = AutoModelForSequenceClassification.from_pretrained(source, **cache)
    model.to(device_name)
    model.eval()
    return model, tokenizer, device_name


def _predict(model, tokenizer, texts: Sequence[str], task: str, *, max_length: int,
             batch_size: int, device_name: str, precision: str, verbose: bool
             ) -> Tuple[list, List[int], List[int]]:
    """Every text's prediction as the mean over its windows (a probability
    vector for a classifier -- softmax for one label, a sigmoid per label
    for a multi-label head -- a number for a regressor), plus token counts
    and window counts."""
    import numpy as np
    import torch

    from ._transformer_common import autocast_for, run_with_oom_fallback
    from .finetune_predictor import _encode_windows, _units, _windows_of

    windows, counts = _windows_of(tokenizer, texts, max_length)
    units = _units(windows, range(len(texts)))
    order = sorted(range(len(units)),
                   key=lambda u: len(windows[units[u][0]][units[u][1]]))
    sums: list = []

    def run(size: int):
        sums[:] = [None] * len(texts)
        for start in range(0, len(units), size):
            idx = [units[u] for u in order[start:start + size]]
            batch = _encode_windows(tokenizer, [windows[t][w] for t, w in idx], device_name)
            # a bare head wants just ids and mask; token_type_ids only when
            # the model has them, which the tokenizer's pad already decided
            with torch.inference_mode(), autocast_for(device_name, precision):
                logits = model(**batch).logits.float().detach().cpu().numpy()
            if task == "classification":
                logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                logits /= logits.sum(axis=1, keepdims=True)
            elif task == "multi_label":
                # each label on its own: a sigmoid per output, no competition
                # between them, so the probabilities needn't sum to one
                logits = 1.0 / (1.0 + np.exp(-logits))
            for j, (t, _w) in enumerate(idx):
                part = float(logits[j, 0]) if task == "regression" else logits[j]
                sums[t] = part if sums[t] is None else sums[t] + part
        return None

    run_with_oom_fallback(run, batch_size, verbose=verbose, what="a batch of windows")
    preds = []
    for t in range(len(texts)):
        mean = sums[t] / len(windows[t])
        preds.append(float(mean) if task == "regression" else [float(v) for v in mean])
    return preds, counts, [len(ws) for ws in windows]


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN, assets={"model_json": None},
                  outputs=("out_features_csv",), bookkeeping=("token_count", "n_windows"))
def apply_hf_classifier(
    *,
    model_json: PathLike,
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    workers: int = 0,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    rounding: int = 4,
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
    precision: Optional[Literal["auto", "fp32", "fp16"]] = None,
    emit_probabilities: Optional[bool] = None,
    threshold: Optional[float] = None,
) -> Path:
    """
    Score new texts with an imported Hugging Face classifier or regressor.

    The model's own ``apply`` settings (batch size, max_length, precision,
    whether to write the per-class probabilities) are used unless the call
    gives its own. Predicted classes are written as the checkpoint's labels
    -- or whatever they were renamed to in Settings.

    Parameters
    ----------
    model_json
        A manifest written by :func:`import_hf_classifier`; its checkpoint
        is read from the payload beside it, or from the model cache for a
        hub name.
    csv_path, txt_dir, analysis_csv, gathered_csv, ...
        The same input contract as the other text analyzers.
    out_features_csv : str or Path, optional
        Default ``./features/hf_classifier_applied.csv``.
    overwrite_existing : bool, default False
        If False and the output exists, return it untouched.
    workers : int, default 0
        Parallel processes for the gather and CPU threads for torch.
    text_cols : sequence of str, default ("text",)
        When gathering from a CSV, the column(s) holding the text.
    id_cols : sequence of str, optional
        Columns that identify each row when gathering from a CSV.
    mode : {"concat", "separate"}, default "concat"
        With several text columns: join or treat separately.
    group_by : sequence of str, optional
        Columns to combine rows by before scoring.
    pattern : str, default every document type
        Which files to read from a folder of documents.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where the model runs.
    rounding : int, default 4
        Decimal places written.
    batch_size, max_length, precision, emit_probabilities, threshold
        Overrides for the model's own apply settings. ``threshold`` is read
        by a multi-label head only: a label is written into ``pred_`` when
        its probability reaches it.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id, token_count, n_windows,`` then
        ``pred_<label>`` and, for a classifier, ``prob_<label>`` and
        ``p_<label>_<class>`` columns. A multi-label head writes the labels
        above the threshold joined with ``|`` in ``pred_`` (empty when none
        clears it), a ``pred_<label>_<class>`` per label holding 1 when that
        label cleared the threshold and 0 when it did not, and -- with
        probabilities on -- one ``p_<label>_<class>`` per label. No
        ``prob_``: no single class won. A
        text longer than ``max_length`` is scored in windows and gets the
        mean of their predictions.
    """
    from ..helpers.gpu import device_note
    from ..helpers.model_spec import apply_defaults
    from ._transformer_common import set_threads, window_length
    from .finetune_predictor import _fmt

    loaded = _load_model(one_model_path(model_json))
    doc, source = loaded["doc"], loaded["source"]
    defaults = apply_defaults(doc)
    batch_size = int(batch_size if batch_size is not None else defaults.get("batch_size", 32))
    max_length = int(max_length if max_length is not None
                     else defaults.get("max_length", doc["encoder"]["max_length"]))
    precision = str(precision if precision is not None else defaults.get("precision", "auto"))
    emit = bool(emit_probabilities if emit_probabilities is not None
                else defaults.get("emit_probabilities", True))
    threshold = float(threshold if threshold is not None else defaults.get("threshold", 0.5))
    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers)
    out_features_csv = Path(out_features_csv) if out_features_csv else \
        Path.cwd() / "features" / "hf_classifier_applied.csv"
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        if verbose:
            print("Classifier output already exists; returning existing file.")
        return out_features_csv

    set_threads(workers)
    announce(on_progress, f"loading {doc.get('name')}")
    model, tokenizer, device_name = _load_classifier(source, device, verbose)
    (outcome, spec), = doc["outcomes"].items()
    task = spec["task"]
    classes = [str(c) for c in (spec.get("classes") or [])]
    if task != "regression" and int(getattr(model.config, "num_labels", len(classes))) != len(classes):
        raise ValueError(f"{Path(model_json).name} names {len(classes)} classes but the "
                         f"checkpoint's head has {model.config.num_labels} outputs. "
                         f"Import the classifier again.")

    import numpy as np

    with analysis_ready.open("r", newline="", encoding=encoding) as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))
    texts = [r.get("text") or "" for r in rows]
    # the device goes in the message that stays up for the whole scoring run,
    # the same as the embedding steps: see helpers.gpu.device_note
    announce(on_progress, device_note(f"scoring {len(rows)} text(s)", device_name))
    # the model's own ceiling wins over the setting, and says so rather than
    # quietly reading less than was asked for
    max_length, capped = window_length(tokenizer, max_length)
    if capped:
        announce(on_progress, capped)
    preds, counts, n_windows = _predict(
        model, tokenizer, texts, task, max_length=max_length, batch_size=batch_size,
        device_name=device_name, precision=precision, verbose=verbose)

    label = output_label(doc, outcome)
    header = ["text_id", "token_count", "n_windows", f"pred_{label}"]
    if task == "classification" and emit:
        header += [f"prob_{label}"] + [f"p_{label}_{class_label(doc, outcome, c)}"
                                       for c in classes]
    elif task == "multi_label":
        # a label per column, 1 when it cleared the threshold. this is the
        # *prediction*, not a probability, so it is written whether or not
        # probabilities were asked for -- the joined pred_ column is a
        # convenience for reading, not something you can group or model on.
        header += [f"pred_{label}_{class_label(doc, outcome, c)}" for c in classes]
        if emit:
            # no prob_ column: there's no single winning class whose
            # probability it would be. one column per label is the whole story
            header += [f"p_{label}_{class_label(doc, outcome, c)}"
                       for c in classes]
    with atomic_write(out_features_csv, mode="w", newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(header)
        for i, r in enumerate(rows):
            blank = not texts[i].strip()
            cells: list = [r.get("text_id", ""), counts[i], n_windows[i]]
            p = preds[i]
            if task == "regression":
                cells.append("" if blank else _fmt(p, rounding))
            elif task == "multi_label":
                present = [class_label(doc, outcome, c) for c, v in zip(classes, p)
                           if v >= threshold]
                cells.append("" if blank else "|".join(present))
                cells += ["" if blank else int(v >= threshold) for v in p]
                if emit:
                    cells += ["" if blank else _fmt(v, rounding) for v in p]
            else:
                best = int(np.argmax(p))
                cells.append("" if blank else class_label(doc, outcome, classes[best]))
                if emit:
                    cells.append("" if blank else _fmt(p[best], rounding))
                    cells += ["" if blank else _fmt(v, rounding) for v in p]
            writer.writerow(cells)
    return out_features_csv


# ---------------------------------------------------------------------------
# command line -- derived from the two functions above
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"import": import_hf_classifier, "apply": apply_hf_classifier},
    description="Import a classifier or regressor from the Hugging Face hub as a "
                "Taters model, or score texts with one already imported.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
