"""
Fine-tune a transformer to predict outcomes from text -- one outcome or
several at once -- with the same cross-validated honesty as the ridge.

A ridge over features asks "which measures predict this outcome?"; a
fine-tuned encoder asks "how well can the text itself predict it?", with
the encoder's own weights adjusted to the task. Every outcome gets a head
of its own on one shared encoder: a regression head (mean squared error on
the standardized outcome) for a numeric column, a classification head
(cross-entropy) for a categorical one, and a row missing one outcome still
trains the others -- so a sheet with a personality score and a diagnosis
column trains one model for both (multi-task learning, Caruana 1997), which
is the usual way to get more out of a small corpus.

* Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional
  transformers for language understanding. *NAACL 2019*.
* Caruana, R. (1997). Multitask learning. *Machine Learning, 28*, 41–75.

The discipline is the ridge's: every headline number is **out of fold**.
The rows are dealt into folds (balanced on the first outcome, stratified
when it is categorical); each fold trains a fresh model on the rest, with
a slice of the training rows held out for early stopping, and predicts the
rows it never saw; the metrics are computed over those predictions, per
fold for the standard error and pooled for the headline. A final model is
then trained on every row for the median best epoch and saved -- encoder
and heads -- as a model any pipeline can apply to new text, whose class
predictions carry the data's own labels.

Predicted classes are written as **labels**, never indices, and every
label can be renamed per model in Settings (``class_names``); so can the
settings that govern how the model is applied (``apply``).
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import (Callable, Dict, List, Literal, Optional, Sequence, Tuple,
                    Union)

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.csvio import widen_csv_field_limit
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.model_spec import class_label, one_model_path, output_label, slug
from ..helpers.progress import announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready
from ._report import machine, md_table, versions, write_markdown
from ._transformer_common import (CURATED_ENCODERS, PREDICTOR_KIND, window_length,
                                  autocast_for, freeze_below, human_count,
                                  load_encoder, parse_layers, payload_digests,
                                  pool_hidden, resolve_encoder,
                                  run_with_oom_fallback, set_threads,
                                  torch_missing_reason)

widen_csv_field_limit()

__all__ = ["finetune_text_predictor", "apply_text_predictor", "PREDICTOR_FORMAT",
           "PREDICTOR_KIND", "SECTION_SLUG", "parse_task_weights"]

PathLike = Union[str, Path]

PREDICTOR_FORMAT = 1
SECTION_SLUG = "text-predictor"
#: Dropout before every head, the value BERT's classifier uses.
_HEAD_DROPOUT = 0.1
#: The fewest rows a categorical outcome's class needs to appear in every
#: fold; the classifier's rule, kept here for the same reason.
_MIN_CLASS_ROWS = 5


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def parse_task_weights(spec: Union[str, Dict[str, float], None],
                       outcomes: Sequence[str]) -> Dict[str, float]:
    """``"age: 1, condition: 2"`` -> weights per outcome, 1.0 where unsaid.
    An unknown outcome is refused by name."""
    weights = {o: 1.0 for o in outcomes}
    if not spec:
        return weights
    items = spec.items() if isinstance(spec, dict) else (
        p.partition(":")[::2] for p in str(spec).split(",") if p.strip())
    for name, value in items:
        name = str(name).strip()
        if name not in weights:
            raise ValueError(f"task_weights names {name!r}, which is not one of the "
                             f"outcomes ({', '.join(outcomes)})")
        try:
            weights[name] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"the weight for {name!r} must be a number, not "
                             f"{value!r}") from None
        if weights[name] <= 0:
            raise ValueError(f"the weight for {name!r} must be positive")
    return weights


def _split_validation(train_rows: Sequence[int], fraction: float, seed: int
                      ) -> Tuple[List[int], List[int]]:
    """A slice of the training rows kept for early stopping, by row."""
    rows = list(train_rows)
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    k = int(round(len(rows) * float(fraction)))
    if len(rows) >= 4:
        k = min(max(1, k), len(rows) - 2)
    else:
        k = 0
    return sorted(rows[k:]), sorted(rows[:k])


def _outcome_tasks(table: List[dict], outcome_cols: Sequence[str],
                   categorical: Sequence[str]) -> Dict[str, dict]:
    """
    What each outcome is and how to encode it: a regression task with the
    mean and standard deviation the head is trained against, or a
    classification task with the sorted class labels. A numeric column is a
    regression unless named categorical; a column of words is always
    classification, and is refused if it was named as numeric implicitly
    by having too many classes.
    """
    from ..stats._common import looks_numeric

    tasks: Dict[str, dict] = {}
    for col in outcome_cols:
        values = [(r.get(col) or "").strip() for r in table]
        present = [v for v in values if v]
        if not present:
            raise ValueError(f"outcome column {col!r} is blank in every row")
        if col in categorical or not looks_numeric(present):
            classes = sorted(set(present))
            if len(classes) < 2:
                raise ValueError(f"outcome {col!r} has one class ({classes[0]!r}); "
                                 f"classifying needs at least two.")
            if len(classes) > 20:
                raise ValueError(
                    f"outcome {col!r} has {len(classes)} distinct values, which is "
                    f"too many for a category. If it is a measurement, leave it "
                    f"out of categorical_outcomes.")
            counts = {c: present.count(c) for c in classes}
            thin = [c for c, n in counts.items() if n < _MIN_CLASS_ROWS]
            if thin:
                raise ValueError(
                    f"outcome {col!r}: class {thin[0]!r} has {counts[thin[0]]} "
                    f"row(s); each class needs at least {_MIN_CLASS_ROWS} to "
                    f"appear in every fold.")
            tasks[col] = {"task": "classification", "classes": classes,
                          "counts": counts}
        else:
            nums = [float(v) for v in present]
            mean = statistics.fmean(nums)
            sd = statistics.pstdev(nums) if len(nums) > 1 else 0.0
            if sd <= 0:
                raise ValueError(f"outcome {col!r} is constant; nothing to predict.")
            tasks[col] = {"task": "regression", "mean": mean, "std": sd,
                          "n": len(nums)}
    return tasks


def _fold_of(table: List[dict], tasks: Dict[str, dict], first: str,
             n_folds: int, seed: int, stratify: bool):
    """Fold per row, dealt on the first outcome: balanced on a measurement,
    stratified on a category -- the ridge's and the classifier's rules."""
    import numpy as np

    from ..stats._fit_common import balanced_folds, random_folds

    n = len(table)
    if not stratify:
        return np.asarray(random_folds(n, n_folds, seed), dtype=int)
    values = [(r.get(first) or "").strip() for r in table]
    if tasks[first]["task"] == "regression":
        y = np.array([float(v) if v else np.nan for v in values])
        filled = np.where(np.isnan(y), np.nanmean(y), y)
        return np.asarray(balanced_folds(filled, n_folds, seed), dtype=int)
    from ..stats.classify import _stratified_folds

    labels = np.array([v if v else tasks[first]["classes"][0] for v in values])
    return np.asarray(_stratified_folds(labels, tasks[first]["classes"], n_folds, seed),
                      dtype=int)


# ---------------------------------------------------------------------------
# The model: one encoder, a head per outcome
# ---------------------------------------------------------------------------

def _build_heads(hidden: int, tasks: Dict[str, dict]):
    import torch

    heads = torch.nn.ModuleDict()
    for name, spec in tasks.items():
        width = len(spec["classes"]) if spec["task"] == "classification" else 1
        heads[_key(name)] = torch.nn.Sequential(torch.nn.Dropout(_HEAD_DROPOUT),
                                                torch.nn.Linear(int(hidden), width))
    return heads


def _key(name: str) -> str:
    """ModuleDict keys cannot hold dots; outcomes are stored by slug."""
    return slug(name, fallback="outcome")


def _windows_of(tokenizer, texts: Sequence[str], max_length: int
                ) -> Tuple[List[List[List[int]]], List[int]]:
    """
    Every text's token ids in windows of at most ``max_length`` (special
    tokens included), plus each text's token count.

    A text used to be one sequence cut at ``max_length``, so a 600-token
    essay trained and was scored on its first 254 tokens. Now it is all of
    its windows: each trains with the text's label, and the prediction is
    the mean over them. A tweet is one window either way.
    """
    from ._transformer_common import chunk_ids

    body = max(8, int(max_length) - 2)
    # verbose=False: the tokenizer would warn that a text is longer than the
    # model's limit, but that's exactly what we're about to window away
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False,
                        verbose=False)["input_ids"]
    windows = []
    for ids in encoded:
        # an empty text is one empty window ([CLS] [SEP]), so that every text
        # gets a row of predictions and nothing downstream has to special-case
        chunks = chunk_ids(ids, body) or [[]]
        windows.append([tokenizer.build_inputs_with_special_tokens(c) for c in chunks])
    return windows, [len(ids) for ids in encoded]


def _units(windows: Sequence[Sequence[Sequence[int]]], text_rows) -> List[Tuple[int, int]]:
    """``(text, window)`` for every window of the given texts: the training
    and scoring units. Folds are dealt by text, so a text's windows are always
    on one side of a split."""
    return [(t, w) for t in text_rows for w in range(len(windows[t]))]


def _unit_weights(windows, units: Sequence[Tuple[int, int]]) -> List[float]:
    """1 / windows-of-its-text per unit: a long text does not outvote a
    short one just by taking more windows to say its piece."""
    return [1.0 / len(windows[t]) for t, _w in units]


def _encode_windows(tokenizer, ids_list: Sequence[Sequence[int]], device_name: str):
    batch = tokenizer.pad({"input_ids": [list(w) for w in ids_list]}, return_tensors="pt")
    return {k: v.to(device_name) for k, v in batch.items()}


def _forward(model, heads, batch, *, layer_idx: List[int], combine: str,
             pooling: str):
    out = model(**batch, output_hidden_states=True)
    states = out.hidden_states
    pooled = [pool_hidden(states[i].float(), batch["attention_mask"], pooling)
              for i in layer_idx]
    import torch

    features = torch.cat(pooled, dim=1) if combine == "concat" \
        else torch.stack(pooled, dim=0).mean(dim=0)
    return {name: head(features) for name, head in heads.items()}


def _targets(rows: Sequence[dict], tasks: Dict[str, dict], device_name: str):
    """Per outcome: the target tensor and a mask of rows that have it."""
    import torch

    out = {}
    for name, spec in tasks.items():
        raw = [(r.get(name) or "").strip() for r in rows]
        mask = torch.tensor([bool(v) for v in raw], dtype=torch.bool, device=device_name)
        if spec["task"] == "regression":
            vals = [((float(v) - spec["mean"]) / spec["std"]) if v else 0.0 for v in raw]
            target = torch.tensor(vals, dtype=torch.float32, device=device_name)
        else:
            index = {c: i for i, c in enumerate(spec["classes"])}
            target = torch.tensor([index.get(v, 0) for v in raw], dtype=torch.long,
                                  device=device_name)
        out[name] = (target, mask)
    return out


def _loss(outputs, targets, tasks: Dict[str, dict], weights: Dict[str, float],
          sample_weight=None):
    import torch

    total = None
    for name, spec in tasks.items():
        logits = outputs[_key(name)]
        target, mask = targets[name]
        if not bool(mask.any()):
            continue
        if spec["task"] == "regression":
            per = (logits.squeeze(-1)[mask] - target[mask]) ** 2
        else:
            per = torch.nn.functional.cross_entropy(logits[mask], target[mask],
                                                    reduction="none")
        if sample_weight is None:
            part = per.mean()
        else:
            # a weighted mean over the units in the batch. the weights are
            # 1/windows-of-the-text, so a text that took five windows adds
            # up to one text's worth of loss, same as a one-window tweet
            sw = sample_weight[mask]
            part = (per * sw).sum() / sw.sum()
        part = part * float(weights[name])
        total = part if total is None else total + part
    if total is None:
        total = torch.zeros((), device=next(iter(outputs.values())).device,
                            requires_grad=True)
    return total


def _predict(model, heads, tokenizer, texts: Sequence[str], tasks: Dict[str, dict], *,
             layer_idx, combine, pooling, max_length, batch_size, device_name,
             precision, windows=None, verbose=False
             ) -> Tuple[Dict[str, list], List[int], List[int]]:
    """Predictions for every text -- a value per regression outcome, a
    probability vector per classification outcome -- as the mean over the
    text's windows; plus each text's token count and how many windows it
    took. ``windows`` may be handed in ready-made (the trainer builds them
    once for every fold); otherwise they are cut here."""
    import numpy as np
    import torch

    model.eval()
    heads.eval()
    n = len(texts)
    if windows is None:
        windows, counts = _windows_of(tokenizer, texts, max_length)
    else:
        counts = [sum(max(0, len(w) - 2) for w in ws) for ws in windows]
    units = _units(windows, range(n))
    # windows of a similar length together, so a batch pads as little as
    # possible. a very long text is just many units here, batched with
    # everyone else's -- never one giant sequence
    order = sorted(range(len(units)),
                   key=lambda u: len(windows[units[u][0]][units[u][1]]))
    sums: Dict[str, list] = {}

    def run(size: int):
        # start over on every attempt: the OOM fallback re-runs us with a
        # smaller batch, and half-summed windows would count twice
        for name in tasks:
            sums[name] = [None] * n
        for start in range(0, len(units), size):
            idx = [units[u] for u in order[start:start + size]]
            batch = _encode_windows(tokenizer, [windows[t][w] for t, w in idx], device_name)
            with torch.inference_mode(), autocast_for(device_name, precision):
                outputs = _forward(model, heads, batch, layer_idx=layer_idx,
                                   combine=combine, pooling=pooling)
            for name, spec in tasks.items():
                values = outputs[_key(name)].float().detach().cpu().numpy()
                if spec["task"] == "classification":
                    # probabilities, not logits, get averaged across windows:
                    # the mean of softmaxes is still a distribution
                    values = np.exp(values - values.max(axis=1, keepdims=True))
                    values /= values.sum(axis=1, keepdims=True)
                for j, (t, _w) in enumerate(idx):
                    part = values[j] if spec["task"] == "classification" else values[j, 0]
                    sums[name][t] = part if sums[name][t] is None else sums[name][t] + part
        return None

    run_with_oom_fallback(run, batch_size, verbose=verbose, what="a batch of windows")
    preds: Dict[str, list] = {}
    for name, spec in tasks.items():
        out = []
        for t in range(n):
            mean = sums[name][t] / len(windows[t])
            if spec["task"] == "regression":
                # averaged in standardized units, then put back on the
                # outcome's scale (same thing, since that's linear)
                out.append(float(mean) * spec["std"] + spec["mean"])
            else:
                out.append([float(v) for v in mean])
        preds[name] = out
    return preds, counts, [len(ws) for ws in windows]


def _train_one(base, tokenizer, table: List[dict], tasks, weights, train_rows, val_rows,
               *, windows, layer_idx, combine, pooling, max_length, batch_size, grad_accum,
               epochs, learning_rate, weight_decay, warmup_fraction, train_layers,
               gradient_checkpointing, device_name, precision, seed, early_stopping,
               warm_heads, on_progress, label: str, verbose: bool):
    """
    Train one model (encoder + heads) on ``train_rows``; when ``val_rows`` is
    given and early stopping is on, keep the epoch with the lowest
    validation loss. Returns ``(model, heads, history, best_epoch)``.

    ``train_rows`` and ``val_rows`` are *texts*; what actually goes through
    the model are their ``windows``, each carrying its text's labels and a
    weight of one over the text's window count.
    """
    import torch
    from transformers import AutoModel, get_linear_schedule_with_warmup

    torch.manual_seed(int(seed))
    model = AutoModel.from_pretrained(base)
    model.to(device_name)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    hidden = int(model.config.hidden_size) * (len(layer_idx) if combine == "concat" else 1)
    heads = _build_heads(hidden, tasks).to(device_name)
    if warm_heads:
        _copy_heads(heads, warm_heads, tasks)
    freeze_below(model, int(train_layers))
    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    train_units = _units(windows, train_rows)
    val_units = _units(windows, val_rows)
    steps_per_epoch = math.ceil(math.ceil(len(train_units) / int(batch_size)) / int(grad_accum))
    total_steps = max(1, steps_per_epoch * int(epochs))
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate),
                                  weight_decay=float(weight_decay))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(round(total_steps * float(warmup_fraction))), total_steps)
    scaler = torch.amp.GradScaler("cuda") if (
        device_name.startswith("cuda") and precision in ("auto", "fp16")) else None
    rng = random.Random(int(seed))
    history: List[dict] = []
    best_state, best_loss, best_epoch = None, float("inf"), int(epochs)
    state = {"batch": int(batch_size), "accum": int(grad_accum)}

    def _batch(idx):
        """The tensors for a list of (text, window) units, their targets and
        their per-unit weights."""
        batch = _encode_windows(tokenizer, [windows[t][w] for t, w in idx], device_name)
        targets = _targets([table[t] for t, _w in idx], tasks, device_name)
        sw = torch.tensor(_unit_weights(windows, idx), dtype=torch.float32,
                          device=device_name)
        return batch, targets, sw

    def val_loss() -> Optional[float]:
        if not val_units:
            return None
        model.eval()
        heads.eval()
        total, n = 0.0, 0
        with torch.inference_mode(), autocast_for(device_name, precision):
            for start in range(0, len(val_units), state["batch"]):
                idx = val_units[start:start + state["batch"]]
                batch, targets, sw = _batch(idx)
                outputs = _forward(model, heads, batch, layer_idx=layer_idx,
                                   combine=combine, pooling=pooling)
                loss = _loss(outputs, targets, tasks, weights, sample_weight=sw)
                total += float(loss.item()) * len(idx)
                n += len(idx)
        return total / n if n else None

    def run_epoch(size: int):
        if size != state["batch"]:
            state["accum"] = max(1, int(round(state["accum"] * state["batch"] / size)))
            state["batch"] = size
        model.train()
        heads.train()
        # shuffled at the window level: a long text's windows spread through
        # the epoch rather than arriving as one block
        order = list(train_units)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        running, seen, accumulated = 0.0, 0, 0
        n_batches = math.ceil(len(order) / size)
        for b_i in range(n_batches):
            idx = order[b_i * size:(b_i + 1) * size]
            batch, targets, sw = _batch(idx)
            with autocast_for(device_name, precision):
                outputs = _forward(model, heads, batch, layer_idx=layer_idx,
                                   combine=combine, pooling=pooling)
                loss = _loss(outputs, targets, tasks, weights, sample_weight=sw)
                scaled = loss / state["accum"]
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            running += float(loss.item()) * len(idx)
            seen += len(idx)
            accumulated += 1
            if accumulated == state["accum"] or b_i == n_batches - 1:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if scaler is not None:
                    # if the scaler skipped this step (inf gradients while
                    # it's still finding its scale), we mustn't advance the
                    # schedule either -- otherwise torch warns and we lose the
                    # first learning-rate value
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= before:
                        scheduler.step()
                else:
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        return running / max(1, seen)

    for epoch in range(int(epochs)):
        train_loss = run_with_oom_fallback(run_epoch, state["batch"], verbose=verbose,
                                           what=f"{label}, epoch {epoch + 1}")
        v = val_loss()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": v})
        if on_progress is not None:
            on_progress(epoch + 1, int(epochs),
                        f"{label}: epoch {epoch + 1}/{epochs}, loss {train_loss:.3f}"
                        + (f", validation {v:.3f}" if v is not None else ""))
        if early_stopping and v is not None and v < best_loss - 1e-9:
            best_loss, best_epoch = v, epoch + 1
            best_state = ({k: t.detach().cpu().clone() for k, t in model.state_dict().items()},
                          {k: t.detach().cpu().clone() for k, t in heads.state_dict().items()})
    if best_state is not None and best_epoch != int(epochs):
        model.load_state_dict(best_state[0])
        heads.load_state_dict(best_state[1])
    if not early_stopping or best_state is None:
        best_epoch = int(epochs)
    return model, heads, history, best_epoch


def _copy_heads(heads, warm: dict, tasks: Dict[str, dict]) -> List[str]:
    """Reuse a saved predictor's heads for outcomes of the same name, task
    and (for categories) class list; new outcomes keep fresh heads."""
    import torch

    reused = []
    for name, spec in tasks.items():
        saved = warm.get("outcomes", {}).get(name)
        tensors = warm.get("tensors", {})
        if not saved or saved.get("task") != spec["task"]:
            continue
        if spec["task"] == "classification" and list(saved.get("classes") or []) != spec["classes"]:
            continue
        key = _key(name)
        w, b = tensors.get(f"{key}.weight"), tensors.get(f"{key}.bias")
        head = heads[key][1]
        if w is None or b is None or tuple(w.shape) != tuple(head.weight.shape):
            continue
        with torch.no_grad():
            head.weight.copy_(w.to(head.weight.device))
            head.bias.copy_(b.to(head.bias.device))
        reused.append(name)
    return reused


def _load_warm(base_spec: str) -> Optional[dict]:
    """A predictor manifest named as the base: its outcomes and head weights."""
    path = Path(str(base_spec))
    if path.suffix.lower() != ".json" or not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("kind") != PREDICTOR_KIND:
        return None
    loaded = _load_model(path)
    return {"outcomes": loaded["doc"]["outcomes"], "tensors": loaded["heads"],
            "name": loaded["doc"].get("name")}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _regression_metrics(actual, predicted) -> dict:
    import numpy as np

    from ..stats.ridge import _metrics

    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(p)
    if ok.sum() < 2:
        return {"r2": float("nan"), "r": float("nan"), "rho": float("nan"),
                "rmse": float("nan"), "mae": float("nan"), "mse": float("nan"),
                "r_p": float("nan"), "rho_p": float("nan"), "n": int(ok.sum())}
    out = _metrics(a[ok], p[ok])
    out["n"] = int(ok.sum())
    out["baseline_mae"] = float(np.abs(a[ok] - a[ok].mean()).mean())
    return out


def _classification_metrics(labels, probabilities, classes) -> dict:
    import numpy as np

    from ..stats.classify import _log_loss, _score

    labels = np.asarray(labels)
    probs = np.asarray(probabilities, dtype=float)
    out = _score(labels, list(classes), probs)
    indicator = np.column_stack([(labels == c).astype(float) for c in classes])
    out["log_loss"] = _log_loss(indicator, probs)
    out["n"] = int(labels.shape[0])
    return out


def _fmt(v, rounding: int) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return ""
    return str(round(f, rounding))


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]],
               encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", newline="", encoding=encoding) as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        for r in rows:
            w.writerow(["" if c is None else c for c in r])


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def finetune_text_predictor(
    *,
    csv_path: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_dir: PathLike = "stats_results",
    out_models_dir: Optional[PathLike] = None,
    name: Optional[str] = None,
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

    # ----- what to predict -----
    outcome_cols: Sequence[str] = (),
    categorical_outcomes: Sequence[str] = (),
    task_weights: str = "",

    # ----- the model -----
    base_model: str = CURATED_ENCODERS[0][0],
    layers: str = "last",
    pooling: Literal["mean", "cls", "max"] = "mean",
    max_length: int = 256,
    train_layers: int = 0,
    gradient_checkpointing: bool = False,

    # ----- training -----
    n_folds: int = 5,
    stratify: bool = True,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    grad_accum: int = 1,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.06,
    early_stopping: bool = True,
    val_fraction: float = 0.1,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    precision: Literal["auto", "fp32", "fp16"] = "auto",
    seed: int = 42,
    rounding: int = 4,
) -> Path:
    """
    Fine-tune an encoder to predict one or more outcome columns from text.

    Parameters
    ----------
    csv_path, analysis_csv, gathered_csv
        The spreadsheet: the text in ``text_cols`` and the outcomes in
        ``outcome_cols``, one row per text (or a prebuilt analysis-ready
        table carrying the outcome columns). A folder of documents has no
        outcome columns and is not accepted.
    out_dir : str or Path, default "stats_results"
        Where the metrics, fold, epoch and prediction tables, the report
        section and the figures go.
    out_models_dir : str or Path, optional
        Where the model lands, default ``<out_dir>/models``: a manifest
        ``<name>.json`` (``text_predictor__<outcomes>.json`` when no name is
        given) and its weights folder beside it.
    name : str, optional
        The model's name in menus and its file stem; default
        ``text_predictor__<outcomes>``.
    overwrite_existing : bool, default False
        If False and the metrics table exists, return it untouched.
    workers : int, default 0
        Parallel processes for the gather, and the CPU threads torch may
        use. 0 means automatic.
    text_cols : sequence of str, default ("text",)
        The column(s) holding the text.
    id_cols : sequence of str, optional
        Columns that identify each row.
    mode : {"concat", "separate"}, default "concat"
        With several text columns: join them into one text per row, or
        treat each as its own text.
    group_by : sequence of str, optional
        Columns to combine rows by before training (one text per group);
        an outcome then has to be constant within a group.
    outcome_cols : sequence of str
        The columns to predict. A numeric column is a regression (mean
        squared error on the standardized value); a column of labels, or
        one named in ``categorical_outcomes``, is a classification. Several
        columns train one model with a head per outcome (multi-task).
    categorical_outcomes : sequence of str
        Which of ``outcome_cols`` are categories even though they look
        numeric -- a 0/1 condition code.
    task_weights : str
        ``"age: 1, condition: 2"``: how much each outcome's loss counts;
        unsaid means 1.
    base_model : str
        The encoder to start from: a Hugging Face name, a checkpoint
        folder, a Taters text encoder file, or a Taters fine-tuned
        predictor file -- in which case its encoder is the starting point
        and its heads are reused for outcomes with the same name and type
        (a warm start on new data), with fresh heads for new outcomes.
    layers : str, default "last"
        Which hidden layers feed the heads (``last`` is right when the
        encoder is being trained; see the embeddings step for the others).
    pooling : {"mean", "cls", "max"}, default "mean"
        How a text's token vectors become one.
    max_length : int, default 256
        The most tokens read per text; longer texts are cut (the share cut
        is reported).
    train_layers : int, default 0
        Train only the top this-many encoder layers and the heads; 0 trains
        everything. Two is a good CPU compromise.
    gradient_checkpointing : bool, default False
        Trade compute for memory on a small card.
    n_folds : int, default 5
        Cross-validation folds. Every headline number is out of fold.
    stratify : bool, default True
        Deal the folds balanced on the first outcome (stratified when it
        is a category) rather than at random.
    epochs : int, default 3
        The most passes over the training rows per fold; with early
        stopping the epoch with the lowest validation loss is kept.
    learning_rate : float, default 2e-5
        The peak learning rate of AdamW, after warm-up.
    batch_size : int, default 16
        Texts per forward pass; halved automatically if the GPU runs out of
        memory (accumulation doubled to compensate).
    grad_accum : int, default 1
        Batches accumulated per optimizer step.
    weight_decay, warmup_fraction
        AdamW's weight decay; the share of steps spent warming up.
    early_stopping : bool, default True
        Keep, per fold, the epoch with the lowest loss on a validation
        slice of the training rows (``val_fraction``); the final model
        trains for the median best epoch.
    val_fraction : float, default 0.1
        The share of each fold's training rows held out for early stopping.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where training runs.
    precision : {"auto", "fp32", "fp16"}, default "auto"
        Half precision on a GPU (auto), always full, or always half.
    seed : int, default 42
        Seeds the folds, the shuffles, the validation slices and the heads.
    rounding : int, default 4
        Decimal places written.

    Returns
    -------
    Path
        ``<out_dir>/text_predictor_cv_metrics.csv``: one row per outcome
        with the ridge's and the classifier's columns, so the two kinds of
        model can be compared in one table.
    """
    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    outcome_cols = [str(c) for c in outcome_cols if str(c).strip()]
    if not outcome_cols:
        raise ValueError("outcome_cols names no column to predict")
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2: every headline number is out of fold")
    if int(epochs) < 1 or int(batch_size) < 1:
        raise ValueError("epochs and batch_size must be at least 1")
    if pooling not in ("mean", "cls", "max"):
        raise ValueError(f"pooling must be mean, cls or max, not {pooling!r}")
    started = time.time()
    # we grab this now, before the per-outcome loops below rebind `name`. we
    # used to read it afterwards, and the model got named after whichever
    # outcome happened to come last
    model_name = str(name).strip() if name else ""
    out_dir = Path(out_dir)
    models_dir = Path(out_models_dir) if out_models_dir else out_dir / "models"
    metrics_path = out_dir / "text_predictor_cv_metrics.csv"
    if metrics_path.is_file() and not overwrite_existing:
        if verbose:
            print("Text predictor results already exist; returning existing file.")
        return metrics_path
    weights = parse_task_weights(task_weights, outcome_cols)

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=None, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers, carry_cols=list(outcome_cols), verbose=verbose)
    with analysis_ready.open("r", newline="", encoding=encoding) as fh:
        table = [r for r in csv.DictReader(fh, delimiter=delimiter)
                 if (r.get("text") or "").strip()]
    missing_cols = [c for c in outcome_cols if table and c not in table[0]]
    if missing_cols:
        raise ValueError(
            f"outcome column(s) {', '.join(missing_cols)} are not in the table "
            f"(it has {', '.join(table[0].keys())})")
    tasks = _outcome_tasks(table, outcome_cols, [str(c) for c in categorical_outcomes])
    labeled = [i for i, r in enumerate(table)
                if any((r.get(c) or "").strip() for c in outcome_cols)]
    if len(labeled) < 2 * int(n_folds):
        raise ValueError(
            f"{len(labeled)} row(s) have an outcome; {n_folds}-fold cross-validation "
            f"needs at least {2 * int(n_folds)}.")
    table = [table[i] for i in labeled]
    texts = [r.get("text") or "" for r in table]
    ids = [r.get("text_id", "") for r in table]

    import numpy as np
    import torch

    threads = set_threads(workers)
    announce(on_progress, f"loading {base_model}")
    warm = _load_warm(str(base_model))
    resolved = resolve_encoder(base_model)
    _model0, tokenizer, _res, device_name, fallback = load_encoder(
        base_model, device=device, verbose=verbose)
    n_layers = int(getattr(_model0.config, "num_hidden_layers", 0) or
                   getattr(_model0.config, "n_layers", 0))
    hidden_size = int(_model0.config.hidden_size)
    n_params = int(sum(p.numel() for p in _model0.parameters()))
    del _model0
    layer_idx, combine = parse_layers(layers, n_layers)
    common = dict(layer_idx=layer_idx, combine=combine, pooling=pooling,
                  max_length=int(max_length), batch_size=int(batch_size),
                  grad_accum=int(grad_accum), epochs=int(epochs),
                  learning_rate=float(learning_rate), weight_decay=float(weight_decay),
                  warmup_fraction=float(warmup_fraction), train_layers=int(train_layers),
                  gradient_checkpointing=bool(gradient_checkpointing),
                  device_name=device_name, precision=precision,
                  early_stopping=bool(early_stopping), on_progress=on_progress,
                  verbose=verbose)
    predict_common = dict(layer_idx=layer_idx, combine=combine, pooling=pooling,
                          max_length=int(max_length), batch_size=int(batch_size),
                          device_name=device_name, precision=precision, verbose=verbose)
    # every text's windows, cut once and shared by every fold and the final
    # model (tokenizing is cheap, but there's no reason to do it six times)
    announce(on_progress, "tokenizing the texts")
    windows, _token_counts = _windows_of(tokenizer, texts, int(max_length))
    common["windows"] = windows

    # first, cross-validation
    folds = _fold_of(table, tasks, outcome_cols[0], int(n_folds), int(seed), bool(stratify))
    oof: Dict[str, list] = {name: [None] * len(table) for name in tasks}
    fold_rows: List[list] = []
    epoch_rows: List[list] = []
    best_epochs: List[int] = []
    reused_heads: List[str] = []
    for f in range(int(n_folds)):
        test_rows = [i for i in range(len(table)) if folds[i] == f]
        rest = [i for i in range(len(table)) if folds[i] != f]
        train_rows, val_rows = _split_validation(rest, val_fraction if early_stopping else 0.0,
                                                 int(seed) + f)
        announce(on_progress, f"fold {f + 1}/{n_folds}: training on {len(train_rows)} texts")
        model, heads, history, best = _train_one(
            resolved.source, tokenizer, table, tasks, weights, train_rows, val_rows,
            seed=int(seed) + f, warm_heads=warm, label=f"fold {f + 1}", **common)
        if warm and f == 0:
            fresh = _build_heads(hidden_size * (len(layer_idx) if combine == "concat" else 1), tasks)
            reused_heads = _copy_heads(fresh, warm, tasks)
        best_epochs.append(best)
        for h in history:
            epoch_rows.append([f + 1, h["epoch"], _fmt(h["train_loss"], 6),
                               _fmt(h["val_loss"], 6), "yes" if h["epoch"] == best else ""])
        preds, _counts, _nwin = _predict(model, heads, tokenizer,
                                         [texts[i] for i in test_rows], tasks,
                                         windows=[windows[i] for i in test_rows],
                                         **predict_common)
        for name in tasks:
            for j, i in enumerate(test_rows):
                oof[name][i] = preds[name][j]
        for name, spec in tasks.items():
            actual = [(table[i].get(name) or "").strip() for i in test_rows]
            have = [j for j, v in enumerate(actual) if v]
            if spec["task"] == "regression":
                m = _regression_metrics([float(actual[j]) for j in have],
                                        [preds[name][j] for j in have])
                fold_rows.append([name, f + 1, m["n"], _fmt(m["r2"], rounding),
                                  _fmt(m["r"], rounding), _fmt(m["rho"], rounding),
                                  _fmt(m["mae"], rounding), "", "", ""])
            else:
                m = _classification_metrics([actual[j] for j in have],
                                            [preds[name][j] for j in have], spec["classes"])
                fold_rows.append([name, f + 1, m["n"], "", "", "", "",
                                  _fmt(m["accuracy"], rounding), _fmt(m["auc"], rounding),
                                  _fmt(m["macro"]["f1"], rounding)])
        del model, heads
        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()

    # now we pool the out-of-fold predictions and get metrics per outcome
    results: Dict[str, dict] = {}
    for name, spec in tasks.items():
        actual = [(r.get(name) or "").strip() for r in table]
        have = [i for i, v in enumerate(actual) if v and oof[name][i] is not None]
        if spec["task"] == "regression":
            m = _regression_metrics([float(actual[i]) for i in have], [oof[name][i] for i in have])
            per_fold = [r for r in fold_rows if r[0] == name]
            r2s = [float(r[3]) for r in per_fold if r[3] != ""]
            m["r2_folds"] = float(np.mean(r2s)) if r2s else float("nan")
            m["r2_folds_se"] = float(np.std(r2s, ddof=1) / np.sqrt(len(r2s))) if len(r2s) > 1 else float("nan")
        else:
            m = _classification_metrics([actual[i] for i in have], [oof[name][i] for i in have],
                                        spec["classes"])
            per_fold = [r for r in fold_rows if r[0] == name]
            accs = [float(r[7]) for r in per_fold if r[7] != ""]
            aucs = [float(r[8]) for r in per_fold if r[8] != ""]
            m["accuracy_folds"] = float(np.mean(accs)) if accs else float("nan")
            m["accuracy_folds_se"] = float(np.std(accs, ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else float("nan")
            m["auc_folds"] = float(np.mean(aucs)) if aucs else float("nan")
            m["auc_folds_se"] = float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))) if len(aucs) > 1 else float("nan")
        results[name] = m

    # the final model: every row, the median best epoch from the folds, and no
    # early stopping
    final_epochs = int(round(statistics.median(best_epochs))) if best_epochs else int(epochs)
    final_epochs = max(1, final_epochs)
    announce(on_progress, f"training the final model on all {len(table)} texts "
                          f"for {final_epochs} epoch(s)")
    final_common = dict(common)
    final_common.update(epochs=final_epochs, early_stopping=False)
    model, heads, final_history, _best = _train_one(
        resolved.source, tokenizer, table, tasks, weights, list(range(len(table))), [],
        seed=int(seed), warm_heads=warm, label="final model", **final_common)
    train_preds, counts, n_windows = _predict(model, heads, tokenizer, texts, tasks,
                                              windows=windows, **predict_common)

    # Save the model. a named model is a file of that name (the Train task
    # names one after its encoder: distilroberta-base-finetuned); an unnamed
    # one is named after what it predicts, as it always was
    stem = (slug(model_name, fallback="text_predictor") if model_name
            else f"text_predictor__{slug('_'.join(outcome_cols), fallback='outcomes')}")
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest = models_dir / f"{stem}.json"
    payload = models_dir / f"{stem}.predictor"
    if payload.exists():
        import shutil

        shutil.rmtree(payload)
    model.eval()
    model.save_pretrained(payload / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(payload / "encoder")
    from safetensors.torch import save_file

    head_tensors = {f"{_key(n)}.weight": heads[_key(n)][1].weight.detach().cpu().contiguous()
                    for n in tasks}
    head_tensors.update({f"{_key(n)}.bias": heads[_key(n)][1].bias.detach().cpu().contiguous()
                         for n in tasks})
    save_file(head_tensors, str(payload / "heads.safetensors"))
    wall = round(time.time() - started, 1)
    outcomes_doc = {}
    for name, spec in tasks.items():
        entry = {"task": spec["task"], "head": _key(name)}
        if spec["task"] == "regression":
            entry.update(mean=spec["mean"], std=spec["std"],
                         cv={k: (None if isinstance(v, float) and math.isnan(v) else v)
                             for k, v in results[name].items()
                             if k in ("r2", "r", "r_p", "rho", "rho_p", "rmse", "mae",
                                      "baseline_mae", "n", "r2_folds", "r2_folds_se")})
        else:
            entry.update(classes=spec["classes"], counts=spec["counts"],
                         cv={k: (None if isinstance(v, float) and math.isnan(v) else v)
                             for k, v in results[name].items()
                             if k in ("accuracy", "baseline_accuracy", "auc", "log_loss",
                                      "n", "accuracy_folds", "accuracy_folds_se",
                                      "auc_folds", "auc_folds_se")}
                         | {"f1_macro": results[name]["macro"]["f1"]})
        outcomes_doc[name] = entry
    doc = {
        "kind": PREDICTOR_KIND, "format": PREDICTOR_FORMAT,
        "name": model_name or stem,
        "payload": [payload.name],
        "payload_digests": payload_digests(payload),
        "base_model": resolved.base_model if resolved.kind != "hub" else resolved.source,
        "started_from": resolved.label,
        "reused_heads": reused_heads,
        "encoder": {"layers": layers, "pooling": pooling, "max_length": int(max_length),
                    "num_hidden_layers": n_layers, "hidden_size": hidden_size,
                    "parameters": n_params},
        "outcomes": outcomes_doc,
        "task_weights": weights,
        "training": {
            "n_texts": len(table), "n_folds": int(n_folds), "stratify": bool(stratify),
            "epochs": int(epochs), "best_epoch_per_fold": best_epochs,
            "final_epochs": final_epochs, "early_stopping": bool(early_stopping),
            "val_fraction": float(val_fraction), "learning_rate": float(learning_rate),
            "batch_size": int(batch_size), "grad_accum": int(grad_accum),
            "weight_decay": float(weight_decay), "warmup_fraction": float(warmup_fraction),
            "train_layers": int(train_layers),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "seed": int(seed), "device": device_name, "precision": precision,
            "threads": threads,
            "windowed_share": sum(1 for k in n_windows if k > 1) / max(1, len(n_windows)),
            "final_history": final_history, "wall_seconds": wall,
        },
        "apply": {"batch_size": int(batch_size), "max_length": int(max_length),
                  "precision": precision, "emit_probabilities": True},
    }
    with atomic_write(manifest, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)

    # the tables
    announce(on_progress, "writing the results")
    metrics_header = ["feature_set", "n_feature_sets", "n_features", "outcome", "model",
                      "task", "n_used", "n_classes", "n_folds", "epochs_used", "base_model",
                      "cv_r2", "cv_r2_folds", "cv_r2_folds_se", "cv_r", "cv_r_p", "cv_rho",
                      "cv_rho_p", "cv_rmse", "cv_mae", "baseline_mae",
                      "accuracy", "baseline_accuracy", "accuracy_folds", "accuracy_folds_se",
                      "auc", "auc_folds", "auc_folds_se", "f1_macro", "f1_weighted",
                      "precision_macro", "recall_macro", "log_loss"]
    metric_rows = []
    for name, spec in tasks.items():
        m = results[name]
        row = ["text", 1, hidden_size, name, "fine-tuned transformer", spec["task"],
               m.get("n"), len(spec["classes"]) if spec["task"] == "classification" else "",
               int(n_folds), final_epochs, doc["base_model"]]
        if spec["task"] == "regression":
            row += [_fmt(m.get(k), rounding) for k in
                    ("r2", "r2_folds", "r2_folds_se", "r", "r_p", "rho", "rho_p", "rmse",
                     "mae", "baseline_mae")] + [""] * 12
        else:
            row += [""] * 10
            row += [_fmt(m.get("accuracy"), rounding), _fmt(m.get("baseline_accuracy"), rounding),
                    _fmt(m.get("accuracy_folds"), rounding), _fmt(m.get("accuracy_folds_se"), rounding),
                    _fmt(m.get("auc"), rounding), _fmt(m.get("auc_folds"), rounding),
                    _fmt(m.get("auc_folds_se"), rounding), _fmt(m["macro"]["f1"], rounding),
                    _fmt(m["weighted"]["f1"], rounding), _fmt(m["macro"]["precision"], rounding),
                    _fmt(m["macro"]["recall"], rounding), _fmt(m.get("log_loss"), rounding)]
        metric_rows.append(row)
    _write_csv(metrics_path, metrics_header, metric_rows, encoding)
    _write_csv(out_dir / "text_predictor_folds.csv",
               ["outcome", "fold", "n", "r2", "r", "rho", "mae", "accuracy", "auc", "f1_macro"],
               fold_rows, encoding)
    _write_csv(out_dir / "text_predictor_epochs.csv",
               ["fold", "epoch", "train_loss", "val_loss", "kept"], epoch_rows, encoding)
    class_rows, conf_rows = [], []
    for name, spec in tasks.items():
        if spec["task"] != "classification":
            continue
        m = results[name]
        for c in spec["classes"]:
            pc = m["per_class"][c]
            class_rows.append([name, class_label(doc, name, c), pc["support"],
                               _fmt(pc["precision"], rounding), _fmt(pc["recall"], rounding),
                               _fmt(pc["f1"], rounding), _fmt(pc.get("auc_vs_rest"), rounding)])
        for (t, p), n_ in m["confusion"].items():
            conf_rows.append([name, class_label(doc, name, t), class_label(doc, name, p), n_])
    if class_rows:
        _write_csv(out_dir / "text_predictor_per_class.csv",
                   ["outcome", "class", "support", "precision", "recall", "f1", "auc_vs_rest"],
                   class_rows, encoding)
        _write_csv(out_dir / "text_predictor_confusion.csv",
                   ["outcome", "true_class", "predicted_class", "n"], conf_rows, encoding)
    # the predictions: observed, out-of-fold, and the final model's own
    pred_header = ["text_id"]
    for name, spec in tasks.items():
        label = output_label(doc, name)
        pred_header += [name, f"oof_{label}", f"fold_{label}", f"pred_{label}"]
        if spec["task"] == "classification":
            pred_header += [f"prob_{label}"] + [f"p_{label}_{class_label(doc, name, c)}"
                                                 for c in spec["classes"]]
    pred_rows = []
    for i in range(len(table)):
        row: list = [ids[i]]
        for name, spec in tasks.items():
            observed = (table[i].get(name) or "").strip()
            o, p = oof[name][i], train_preds[name][i]
            if spec["task"] == "regression":
                row += [observed, _fmt(o, rounding), int(folds[i]) + 1, _fmt(p, rounding)]
            else:
                cls = spec["classes"]
                o_lab = class_label(doc, name, cls[int(np.argmax(o))]) if o is not None else ""
                p_lab = class_label(doc, name, cls[int(np.argmax(p))])
                row += [class_label(doc, name, observed) if observed else "", o_lab,
                        int(folds[i]) + 1, p_lab, _fmt(max(p), rounding)]
                row += [_fmt(v, rounding) for v in p]
        pred_rows.append(row)
    _write_csv(out_dir / f"text_predictor_predictions__{slug('_'.join(outcome_cols), fallback='outcomes')}.csv",
               pred_header, pred_rows, encoding)

    # lastly, the figures, the report section, and the report beside the model
    figures = _figures(out_dir, doc, tasks, results, table, oof, epoch_rows, final_history)
    from ..stats._common import write_section

    write_section(out_dir, SECTION_SLUG,
                  _section_md(doc, tasks, results, figures, out_dir, fallback))
    _write_report(manifest.with_name(f"{stem}_report.md"), doc, manifest, payload, tasks,
                  results, figures, out_dir, fallback)
    if verbose:
        for name, spec in tasks.items():
            m = results[name]
            head = (f"R² {m['r2']:.3f}, r {m['r']:.3f}" if spec["task"] == "regression"
                    else f"accuracy {m['accuracy']:.3f} (baseline {m['baseline_accuracy']:.3f}), "
                         f"AUC {m['auc']:.3f}")
            print(f"[text_predictor] {name}: out-of-fold {head}")
        print(f"[text_predictor] model -> {manifest}; results -> {out_dir}")
    return metrics_path


# ---------------------------------------------------------------------------
# Figures and reports
# ---------------------------------------------------------------------------

def _figures(out_dir: Path, doc: dict, tasks, results, table, oof, epoch_rows,
             final_history) -> Dict[str, Path]:
    """Loss curves per fold, a scatter per numeric outcome, a confusion
    heat table per category; ``{}`` without Pillow."""
    import numpy as np

    try:
        from ..figures.charts import heat_table, line_chart, pillow_missing_reason, scatter_chart
    except ImportError:
        return {}
    if pillow_missing_reason():
        return {}
    folder = out_dir / "figures" / "text_predictor"
    folder.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    series: Dict[str, list] = {}
    for fold, epoch, tr, va, _kept in epoch_rows:
        series.setdefault(f"fold {fold} train", []).append((int(epoch), float(tr)))
        if va != "":
            series.setdefault(f"fold {fold} validation", []).append((int(epoch), float(va)))
    if series:
        out["loss"] = line_chart(series, folder / "loss_per_fold.png",
                                 title="Training and validation loss per epoch, per fold",
                                 x_label="epoch", y_label="loss")
    if final_history:
        out["final_loss"] = line_chart(
            {"final model": [(h["epoch"], float(h["train_loss"])) for h in final_history]},
            folder / "final_model_loss.png", title="Final model: training loss per epoch",
            x_label="epoch", y_label="loss")
    for name, spec in tasks.items():
        m = results[name]
        stem = slug(name, fallback="outcome")
        if spec["task"] == "regression":
            pts = [(float(r.get(name)), float(oof[name][i])) for i, r in enumerate(table)
                   if (r.get(name) or "").strip() and oof[name][i] is not None]
            out[f"scatter_{name}"] = scatter_chart(
                pts, folder / f"predicted_vs_observed_{stem}.png",
                title=f"{name}: out-of-fold predicted vs observed",
                x_label="observed", y_label="predicted (out of fold)",
                note=f"r = {m['r']:.3f}, R² = {m['r2']:.3f}, n = {m['n']}")
        else:
            classes = spec["classes"]
            labels = [class_label(doc, name, c) for c in classes]
            matrix = [[m["confusion"][(t, p)] for p in classes] for t in classes]
            out[f"confusion_{name}"] = heat_table(
                matrix, folder / f"confusion_{stem}.png", row_labels=labels,
                col_labels=labels, title=f"{name}: out-of-fold confusion matrix",
                row_title="observed", col_title="predicted")
    _ = np
    return out


def _rel(path: Path, root: Path) -> str:
    """A link from a Markdown file in ``root`` to ``path`` -- relative, so
    the report survives a move of the whole results folder."""
    import os

    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()
    except ValueError:          # a different drive on Windows; no relative path
        return Path(path).resolve().as_posix()


def _outcome_lines(doc: dict, tasks, results, figures: Dict[str, Path], root: Path,
                   rounding: int = 3) -> List[str]:
    lines: List[str] = []
    for name, spec in tasks.items():
        m = results[name]
        lines += [f"### {name}", ""]
        if spec["task"] == "regression":
            lines += [f"Regression on {m['n']} texts: out-of-fold R² = {m['r2']:.{rounding}f} "
                      f"(per-fold mean {m['r2_folds']:.{rounding}f}, SE {m['r2_folds_se']:.{rounding}f}), "
                      f"r = {m['r']:.{rounding}f} (p = {m['r_p']:.3g}), ρ = {m['rho']:.{rounding}f}, "
                      f"RMSE = {m['rmse']:.{rounding}f}, MAE = {m['mae']:.{rounding}f} against a "
                      f"predict-the-mean MAE of {m['baseline_mae']:.{rounding}f}.", ""]
            fig = figures.get(f"scatter_{name}")
            if fig is not None:
                lines += [f"![{name} predicted vs observed]({_rel(fig, root)})", ""]
        else:
            lines += [f"Classification of {m['n']} texts into {len(spec['classes'])} classes: "
                      f"out-of-fold accuracy = {m['accuracy']:.{rounding}f} (always-the-commonest "
                      f"baseline {m['baseline_accuracy']:.{rounding}f}; per-fold mean "
                      f"{m['accuracy_folds']:.{rounding}f}, SE {m['accuracy_folds_se']:.{rounding}f}), "
                      f"AUC = {m['auc']:.{rounding}f}, macro F1 = {m['macro']['f1']:.{rounding}f}, "
                      f"log loss = {m['log_loss']:.{rounding}f}.", ""]
            rows = [(class_label(doc, name, c), m["per_class"][c]["support"],
                     f"{m['per_class'][c]['precision']:.{rounding}f}",
                     f"{m['per_class'][c]['recall']:.{rounding}f}",
                     f"{m['per_class'][c]['f1']:.{rounding}f}") for c in spec["classes"]]
            lines += md_table(["class", "support", "precision", "recall", "F1"], rows) + [""]
            fig = figures.get(f"confusion_{name}")
            if fig is not None:
                lines += [f"![{name} confusion matrix]({_rel(fig, root)})", ""]
    return lines


def _methods_paragraph(doc: dict, tasks) -> str:
    t, e = doc["training"], doc["encoder"]
    # built in two steps rather than one nested f-string: reusing the same
    # quote inside a replacement field only became legal in 3.12 (PEP 701),
    # and this file has to import on 3.11. CI caught it as a SyntaxError at
    # collection, which takes the whole run down rather than one test.
    def _kind(spec) -> str:
        if spec["task"] == "regression":
            return "regression"
        return f"{len(spec['classes'])}-class classification"

    kinds = [f"{n} ({_kind(s)})"
             for n, s in tasks.items()]
    multi = len(tasks) > 1
    return (
        f"A text predictor was fine-tuned from {doc['base_model']} "
        f"({e['num_hidden_layers']} layers, {human_count(e['parameters'])} parameters"
        + (f", started from the Taters model '{doc['started_from']}'"
           f"{', reusing its heads for ' + ', '.join(doc['reused_heads']) if doc.get('reused_heads') else ''}"
           if doc.get("started_from") != doc["base_model"] else "")
        + f") to predict {', '.join(kinds)} from the text"
        + (" with one shared encoder and a head per outcome (multi-task learning; rows "
           "missing an outcome still trained the others)" if multi else "")
        + f". Texts were read in windows of {e['max_length']} tokens"
        + (f" ({100 * t['windowed_share']:.1f}% needed more than one; every window "
           f"trained with its text's labels, weighted so each text counted once, and "
           f"a text's prediction is the mean over its windows)"
           if t["windowed_share"] else "")
        + f", token vectors from the {e['layers']} layer were {e['pooling']}-pooled, and "
        f"the model trained with AdamW (peak learning rate {t['learning_rate']:g}, weight "
        f"decay {t['weight_decay']:g}, linear warm-up over {int(round(100 * t['warmup_fraction']))}% "
        f"of steps then decay), batch {t['batch_size']} × {t['grad_accum']}, gradient "
        f"clipping at 1.0, {t['precision']} precision on {t['device']}"
        + (f", training only the top {t['train_layers']} layer(s) and the heads"
           if t["train_layers"] else "")
        + f". Performance was estimated by {t['n_folds']}-fold cross-validation "
        + ("(folds balanced on the first outcome)" if t["stratify"] else "(random folds)")
        + f": each fold trained a fresh model for up to {t['epochs']} epochs"
        + (f", keeping the epoch with the lowest loss on a {int(round(100 * t['val_fraction']))}% "
           f"validation slice of its training rows (best epochs: "
           f"{', '.join(map(str, t['best_epoch_per_fold']))})" if t["early_stopping"] else "")
        + f", and every reported number is computed on the rows a model never saw. The final "
        f"model was trained on all {t['n_texts']} texts for {t['final_epochs']} epoch(s) "
        f"(seed {t['seed']}); fine-tuning took {t['wall_seconds']} s.")


def _section_md(doc, tasks, results, figures, out_dir: Path, fallback) -> str:
    lines = ["## Prediction (fine-tuned transformer)", "",
             _methods_paragraph(doc, tasks), ""]
    if fallback:
        lines += [f"*{fallback}*", ""]
    lines += _outcome_lines(doc, tasks, results, figures, out_dir)
    if "loss" in figures:
        lines += ["Training and validation loss per epoch, per fold:", "",
                  f"![loss per fold]({_rel(figures['loss'], out_dir)})", ""]
    lines += [f"Tables: `text_predictor_cv_metrics.csv` (the ridge's and classifier's "
              f"columns), `text_predictor_folds.csv`, `text_predictor_epochs.csv`, "
              f"`text_predictor_predictions__*.csv` (observed, out-of-fold and final "
              f"predictions). The saved model is `models/{doc['payload'][0].replace('.predictor', '.json')}`; "
              f"add it to your library to score new texts, and rename its classes in "
              f"Settings if the data's labels are codes.", ""]
    return "\n".join(lines)


def _write_report(path: Path, doc: dict, manifest: Path, payload: Path, tasks, results,
                  figures, out_dir: Path, fallback) -> None:
    t, e = doc["training"], doc["encoder"]
    lines = [f"# Fine-tuned text predictor: {doc['name']}", "",
             "## Methods paragraph", "", _methods_paragraph(doc, tasks), ""]
    if fallback:
        lines += [f"*{fallback}*", ""]
    lines += ["## Settings", ""]
    settings = [("model file", manifest.name), ("weights folder", payload.name),
                ("base model", doc["base_model"]), ("started from", doc["started_from"]),
                ("outcomes", ", ".join(f"{n} ({s['task']})" for n, s in tasks.items())),
                ("task weights", ", ".join(f"{k}: {v:g}" for k, v in doc["task_weights"].items())),
                ("layers read", e["layers"]), ("pooling", e["pooling"]),
                ("max_length", e["max_length"]), ("encoder layers", e["num_hidden_layers"]),
                ("hidden size", e["hidden_size"]), ("parameters", human_count(e["parameters"]))]
    settings += [(k, t[k]) for k in ("n_texts", "n_folds", "stratify", "epochs",
                                     "early_stopping", "val_fraction", "final_epochs",
                                     "learning_rate", "batch_size", "grad_accum",
                                     "weight_decay", "warmup_fraction", "train_layers",
                                     "gradient_checkpointing", "seed", "device",
                                     "precision", "threads")]
    settings.append(("texts longer than one window", f"{100 * t['windowed_share']:.1f}%"))
    lines += md_table(["setting", "value"], settings) + [""]
    lines += ["## Out-of-fold results", ""] + _outcome_lines(doc, tasks, results, figures, path.parent)
    lines += ["## Training", "",
              f"Best epoch per fold: {', '.join(map(str, t['best_epoch_per_fold']))}; the final "
              f"model trained for {t['final_epochs']} epoch(s).", ""]
    if "loss" in figures:
        lines += [f"![loss per fold]({_rel(figures['loss'], path.parent)})", ""]
    if t.get("final_history"):
        lines += md_table(["epoch", "training loss"],
                          [(h["epoch"], f"{h['train_loss']:.4f}") for h in t["final_history"]]) + [""]
        if "final_loss" in figures:
            lines += [f"![final model loss]({_rel(figures['final_loss'], path.parent)})", ""]
    lines += ["## Outputs", "",
              f"- `{manifest.name}` — the model; add it to your library (the finish screen "
              f"offers to) and it scores new texts from the feature checklist. Predicted "
              f"classes are written with the data's own labels; rename them per model in "
              f"Settings.",
              f"- `{payload.name}/` — the encoder and the heads.",
              f"- `{_rel(out_dir, path.parent)}/text_predictor_*.csv` — the metrics, folds, "
              f"epochs, predictions, per-class and confusion tables.", ""]
    lines += ["## Provenance", ""]
    prov = [("wall time (s)", t["wall_seconds"]), ("machine", machine(t["device"]))]
    prov += list(versions("torch", "transformers", "numpy", "safetensors").items())
    lines += md_table(["item", "value"], prov) + [""]
    write_markdown(path, lines)


# ---------------------------------------------------------------------------
# Load and apply
# ---------------------------------------------------------------------------

def _load_model(model_json: PathLike) -> dict:
    """
    Load and *vet* a predictor manifest: the import gate's authority too.

    Returns ``{"doc", "payload", "heads"}`` -- the heads as CPU tensors.
    Every refusal names what is wrong and what to do.
    """
    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    from ._transformer_common import _payload_folder

    path = Path(model_json)
    if not path.exists():
        raise FileNotFoundError(f"model_json not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    kind = doc.get("kind")
    if kind != PREDICTOR_KIND:
        what = f"a {kind!r} file" if kind else "not a Taters model file at all"
        raise ValueError(f"{path.name} is not a fine-tuned text predictor ({what}).")
    if int(doc.get("format", 0)) > PREDICTOR_FORMAT:
        raise ValueError(f"{path.name} was written by a newer Taters (format "
                         f"{doc.get('format')}; this build reads up to {PREDICTOR_FORMAT}). "
                         f"Update Taters to use it.")

    def _broken(why: str) -> ValueError:
        return ValueError(f"{path.name} is damaged or incomplete: {why}. Re-run the "
                          f"fine-tuning step to write a fresh one.")

    for key in ("outcomes", "encoder", "payload"):
        if key not in doc:
            raise _broken(f"it lacks {key!r}")
    payload = _payload_folder(path, doc, "models")
    if not (payload / "encoder" / "config.json").is_file():
        raise _broken(f"{payload.name} holds no encoder")
    heads_file = payload / "heads.safetensors"
    if not heads_file.is_file():
        raise _broken(f"{payload.name} holds no heads.safetensors")
    from safetensors.torch import load_file

    heads = load_file(str(heads_file))
    for name, spec in doc["outcomes"].items():
        key = spec.get("head") or _key(name)
        if f"{key}.weight" not in heads:
            raise _broken(f"no head weights for outcome {name!r}")
        if spec.get("task") == "classification" and \
                heads[f"{key}.weight"].shape[0] != len(spec.get("classes") or []):
            raise _broken(f"the head for {name!r} has {heads[f'{key}.weight'].shape[0]} "
                          f"outputs but {len(spec.get('classes') or [])} classes")
    return {"doc": doc, "payload": payload, "heads": heads}


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN, assets={"model_json": None},
                  outputs=("out_features_csv",), bookkeeping=("token_count", "n_windows"))
def apply_text_predictor(
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
) -> Path:
    """
    Score new texts with a fine-tuned text predictor.

    The model's own ``apply`` settings (batch size, max_length, precision,
    whether to write the per-class probabilities) are used unless the call
    gives its own. Predicted classes are written as the model's labels --
    the data's own, or whatever they were renamed to in Settings.

    Parameters
    ----------
    model_json
        A predictor manifest written by :func:`finetune_text_predictor`;
        its weights are read from beside it, or from the library when the
        manifest traveled alone.
    csv_path, txt_dir, analysis_csv, gathered_csv, ...
        The same input contract as the other text analyzers.
    out_features_csv : str or Path, optional
        Default ``./features/text_predictor_applied.csv``.
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
    batch_size, max_length, precision, emit_probabilities
        Overrides for the model's own apply settings.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id, token_count, n_windows,`` then per
        outcome ``pred_<label>`` and, for a category, ``prob_<label>`` and
        ``p_<label>_<class>`` columns. A text longer than ``max_length`` is
        scored in windows and gets the mean of their predictions.
    """
    from ..helpers.model_spec import apply_defaults

    loaded = _load_model(one_model_path(model_json))
    doc, payload = loaded["doc"], loaded["payload"]
    defaults = apply_defaults(doc)
    batch_size = int(batch_size if batch_size is not None else defaults.get("batch_size", 32))
    max_length = int(max_length if max_length is not None
                     else defaults.get("max_length", doc["encoder"]["max_length"]))
    precision = str(precision if precision is not None else defaults.get("precision", "auto"))
    emit = bool(emit_probabilities if emit_probabilities is not None
                else defaults.get("emit_probabilities", True))
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
        Path.cwd() / "features" / "text_predictor_applied.csv"
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        if verbose:
            print("Text predictor output already exists; returning existing file.")
        return out_features_csv

    import numpy as np

    set_threads(workers)
    announce(on_progress, f"loading {doc.get('name')}")
    model, tokenizer, _res, device_name, _reason = load_encoder(
        payload / "encoder", device=device, verbose=verbose)
    tasks = {name: dict(spec) for name, spec in doc["outcomes"].items()}
    n_layers = int(getattr(model.config, "num_hidden_layers", 0) or
                   getattr(model.config, "n_layers", 0))
    layer_idx, combine = parse_layers(doc["encoder"]["layers"], n_layers)
    hidden = int(model.config.hidden_size) * (len(layer_idx) if combine == "concat" else 1)
    heads = _build_heads(hidden, tasks)
    import torch

    with torch.no_grad():
        for name in tasks:
            key = tasks[name].get("head") or _key(name)
            heads[_key(name)][1].weight.copy_(loaded["heads"][f"{key}.weight"])
            heads[_key(name)][1].bias.copy_(loaded["heads"][f"{key}.bias"])
    heads.to(device_name)

    with analysis_ready.open("r", newline="", encoding=encoding) as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))
    texts = [r.get("text") or "" for r in rows]
    announce(on_progress, f"scoring {len(rows)} text(s)")
    # the encoder's own ceiling wins over the setting, and says so rather than
    # quietly reading less than was asked for
    max_length, capped = window_length(tokenizer, max_length)
    if capped:
        announce(on_progress, capped)
    preds, counts, n_windows = _predict(
        model, heads, tokenizer, texts, tasks, layer_idx=layer_idx, combine=combine,
        pooling=doc["encoder"]["pooling"], max_length=max_length, batch_size=batch_size,
        device_name=device_name, precision=precision, verbose=verbose)
    header = ["text_id", "token_count", "n_windows"]
    for name, spec in tasks.items():
        label = output_label(doc, name)
        header.append(f"pred_{label}")
        if spec["task"] == "classification" and emit:
            header += [f"prob_{label}"] + [f"p_{label}_{class_label(doc, name, c)}"
                                           for c in spec["classes"]]
    with atomic_write(out_features_csv, mode="w", newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(header)
        for i, r in enumerate(rows):
            blank = not texts[i].strip()
            cells: list = [r.get("text_id", ""), counts[i], n_windows[i]]
            for name, spec in tasks.items():
                p = preds[name][i]
                if spec["task"] == "regression":
                    cells.append("" if blank else _fmt(p, rounding))
                else:
                    cls = spec["classes"]
                    cells.append("" if blank else class_label(doc, name, cls[int(np.argmax(p))]))
                    if emit:
                        cells.append("" if blank else _fmt(max(p), rounding))
                        cells += ["" if blank else _fmt(v, rounding) for v in p]
            writer.writerow(cells)
    if verbose:
        print(f"[text_predictor] {len(rows)} text(s) scored with {doc.get('name')} "
              f"on {device_name} -> {out_features_csv}")
    return out_features_csv


# we keep this importable for the registry's reader, without dragging torch in
_ = copy


# ---------------------------------------------------------------------------
# command line -- we derive this from the functions above; see
# helpers.cliargs.CliSpec
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"fit": finetune_text_predictor, "apply": apply_text_predictor},
    description="Fine-tune a transformer to predict outcomes from text (one or "
                "several at once, cross-validated), or score new texts with a saved one.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
