"""
Domain-adaptive pretraining: continue a language model's own training on
your texts, so it speaks their dialect before it is asked to embed or
predict anything.

A pre-trained encoder learned from web text and books. A corpus of
bereavement-forum posts, clinical notes or adolescents' diaries uses words
the encoder rarely saw and uses familiar words differently; adaptation
continues the masked-language-model objective -- hide fifteen percent of
the tokens, predict them -- on the corpus itself, with no labels, and
leaves an encoder whose held-out perplexity on that corpus has fallen. That
encoder is then the base for embeddings (:mod:`transformer_embeddings`) or
for fine-tuning a predictor (:mod:`finetune_predictor`).

* Gururangan, S., et al. (2020). Don't stop pretraining: Adapt language
  models to domains and tasks. *ACL 2020*.

The loop is plain torch: AdamW with linear warm-up and decay, gradient
clipping, half precision on a GPU, dynamic masking from transformers'
collator, a held-out split *by text* whose loss and perplexity are measured
before and after with the same masking seed, so the two numbers are
comparable. A GPU that runs out of memory halves the batch and doubles the
accumulation, keeping the effective batch the same.

What it writes: ``<name>.json`` (a ``taters-encoder`` manifest), the
checkpoint folder ``<name>.encoder`` beside it, and ``<name>_report.md``
with everything a methods section needs.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.progress import announce
from ..helpers.text_gather import resolve_analysis_ready
from ._mlm_train import split_heldout, train_mlm
from ._report import line_plot, machine, md_table, versions, write_markdown
from ._transformer_common import (ENCODER_KIND, CURATED_ENCODERS, freeze_below,
                                  human_count, load_encoder, payload_digests,
                                  set_threads, torch_missing_reason)

__all__ = ["adapt_encoder", "ENCODER_FORMAT", "split_heldout"]

PathLike = Union[str, Path]

#: The manifest format; bumped on any incompatible change.
ENCODER_FORMAT = 1

#: How many training-loss points the report table shows (the plot shows all).
_LOSS_TABLE_ROWS = 12


def _tokenizer_mismatch(texts: Sequence[str], tokenizer, top: int = 15
                        ) -> List[Tuple[str, int, int]]:
    """The corpus's most frequent words the tokenizer breaks into pieces:
    ``(word, count, pieces)``. The vocabulary gap adaptation is closing."""
    counts: Counter = Counter()
    for t in texts:
        for w in str(t).lower().split():
            w = w.strip(".,;:!?\"'()[]{}")
            if w.isalpha():
                counts[w] += 1
    out = []
    for word, c in counts.most_common(2000):
        pieces = len(tokenizer(word, add_special_tokens=False)["input_ids"])
        if pieces >= 2:
            out.append((word, c, pieces))
        if len(out) >= top:
            break
    return out


def adapt_encoder(
    *,
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_model_json: Optional[PathLike] = None,
    out_report_md: Optional[PathLike] = None,
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

    # ----- what to adapt, and how -----
    base_model: str = CURATED_ENCODERS[0][0],
    name: Optional[str] = None,
    epochs: int = 3,
    max_length: int = 256,
    batch_size: int = 16,
    grad_accum: int = 2,
    learning_rate: float = 5e-5,
    warmup_fraction: float = 0.06,
    weight_decay: float = 0.01,
    mlm_probability: float = 0.15,
    heldout_fraction: float = 0.1,
    train_layers: int = 0,
    gradient_checkpointing: bool = False,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    precision: Literal["auto", "fp32", "fp16"] = "auto",
    seed: int = 42,
) -> Path:
    """
    Continue an encoder's masked-language-model training on these texts.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as every other text step: a spreadsheet of
        texts, a folder of documents, or a prebuilt analysis-ready CSV.
    out_model_json : str or Path, optional
        The encoder manifest; the checkpoint lands beside it as
        ``<stem>.encoder``. Default ``./models/adapted_encoder.json``.
    out_report_md : str or Path, optional
        The training report, default ``<stem>_report.md`` beside the model.
    overwrite_existing : bool, default False
        If False and the manifest exists, return it untouched.
    workers : int, default 0
        Parallel processes for the gather, and the CPU threads torch may
        use. 0 means automatic.
    text_cols : sequence of str, default ("text",)
        When gathering from a CSV, the column(s) holding the text.
    id_cols : sequence of str, optional
        Columns that identify each row when gathering from a CSV.
    mode : {"concat", "separate"}, default "concat"
        With several text columns: join them into one text per row, or
        treat each as its own text.
    group_by : sequence of str, optional
        Columns to combine rows by before training (one text per group).
    pattern : str, default every document type
        Which files to read when gathering from a folder of documents.
    base_model : str
        The encoder to start from: a Hugging Face name, a checkpoint
        folder, or a Taters text encoder file (adapting twice is allowed).
    name : str, optional
        The encoder's name in menus; default the manifest's file stem.
    epochs : int, default 3
        Passes over the corpus. One to three is usual for adaptation; the
        report's held-out loss per epoch shows when more stopped helping.
    max_length : int, default 256
        Tokens per training window. Longer texts are cut into windows so
        every token is trained on; 256 is a good trade of context for speed.
    batch_size : int, default 16
        Windows per forward pass; halved automatically if the GPU runs out
        of memory, with ``grad_accum`` doubled to compensate.
    grad_accum : int, default 2
        Batches accumulated per optimizer step. The effective batch is
        ``batch_size × grad_accum``.
    learning_rate : float, default 5e-5
        The peak learning rate of AdamW, after warm-up.
    warmup_fraction : float, default 0.06
        The share of optimizer steps spent warming the learning rate up
        linearly from zero; it then decays linearly to zero.
    weight_decay : float, default 0.01
        AdamW's weight decay.
    mlm_probability : float, default 0.15
        The share of tokens masked in each window, freshly drawn each pass.
    heldout_fraction : float, default 0.1
        The share of *texts* set aside, never trained on, to measure the
        loss and perplexity before and after.
    train_layers : int, default 0
        Train only the top this-many layers (and the prediction head); 0
        trains everything. Two layers is a good CPU compromise.
    gradient_checkpointing : bool, default False
        Trade compute for memory on a small card.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where training runs.
    precision : {"auto", "fp32", "fp16"}, default "auto"
        Half precision on a GPU (auto), always full, or always half.
    seed : int, default 42
        Seeds the held-out split, the shuffles, the masking and the
        initialization of anything new.

    Returns
    -------
    Path
        ``out_model_json``.
    """
    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    if int(epochs) < 1 or int(batch_size) < 1 or int(grad_accum) < 1:
        raise ValueError("epochs, batch_size and grad_accum must be at least 1")
    if not (0.0 < float(mlm_probability) < 1.0):
        raise ValueError("mlm_probability must be between 0 and 1")
    if not (0.0 <= float(heldout_fraction) < 1.0):
        raise ValueError("heldout_fraction must be at least 0 and below 1")
    started = time.time()

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
    out_model_json = Path(out_model_json) if out_model_json else \
        Path.cwd() / "models" / "adapted_encoder.json"
    if out_model_json.is_file() and not overwrite_existing:
        if verbose:
            print("Adapted encoder already exists; returning existing file.")
        return out_model_json
    out_report_md = Path(out_report_md) if out_report_md else \
        out_model_json.with_name(out_model_json.stem + "_report.md")
    out_model_json.parent.mkdir(parents=True, exist_ok=True)

    import csv

    from ..helpers.csvio import widen_csv_field_limit

    widen_csv_field_limit()
    ids: List[str] = []
    texts: List[str] = []
    with analysis_ready.open("r", newline="", encoding=encoding) as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            text = (row.get("text") or "").strip()
            if text:
                ids.append(str(row.get("text_id", "")))
                texts.append(text)
    if len(texts) < 2:
        raise ValueError(
            f"only {len(texts)} text(s) with words in them; adapting an encoder "
            f"needs a corpus (at least two texts, ideally thousands).")

    import torch

    torch.manual_seed(int(seed))
    threads = set_threads(workers)
    announce(on_progress, f"loading {base_model}")
    model, tokenizer, resolved, device_name, reason = load_encoder(
        base_model, device=device, for_mlm=True, verbose=verbose)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    trainable, frozen = freeze_below(model, int(train_layers))

    run = train_mlm(
        model=model, tokenizer=tokenizer, texts=texts, ids=ids,
        device_name=device_name, precision=precision, seed=int(seed),
        epochs=int(epochs), max_length=int(max_length),
        batch_size=int(batch_size), grad_accum=int(grad_accum),
        learning_rate=float(learning_rate), warmup_fraction=float(warmup_fraction),
        weight_decay=float(weight_decay), mlm_probability=float(mlm_probability),
        heldout_fraction=float(heldout_fraction),
        # adaptation is the fall from before to after, so before is measured;
        # and it runs the epochs it was asked for, the report showing when
        # more stopped helping
        measure_before=True, early_stopping=False,
        verbose=verbose, on_progress=on_progress,
        announce=lambda phrase: announce(on_progress, phrase))
    loss_before, loss_after = run.loss_before, run.loss_after
    train_idx, held_idx, token_counts = run.train_idx, run.held_idx, run.token_counts

    # lastly, we save: the checkpoint folder goes beside the manifest, and
    # then the manifest itself
    announce(on_progress, "saving the adapted encoder")
    folder = out_model_json.with_suffix(".encoder")
    if folder.exists():
        import shutil

        shutil.rmtree(folder)
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    cfg = model.config
    model_name = str(name or out_model_json.stem)
    wall = round(time.time() - started, 1)
    doc = {
        "kind": ENCODER_KIND, "format": ENCODER_FORMAT, "name": model_name,
        "payload": [folder.name],
        "payload_digests": payload_digests(folder),
        "base_model": resolved.base_model if resolved.kind != "hub" else resolved.source,
        "adapted_from": resolved.label,
        "architecture": str(getattr(cfg, "model_type", "")),
        "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layers", 0)),
        "hidden_size": int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "dim", 0)),
        "vocab_size": int(getattr(cfg, "vocab_size", 0)),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "text": {"max_length": int(max_length)},
        "training": {
            "epochs": int(epochs), "max_length": int(max_length),
            "batch_size": int(batch_size), "grad_accum": int(grad_accum),
            "final_batch_size": run.final_batch_size, "final_grad_accum": run.final_grad_accum,
            "oom_restarts": run.oom_restarts,
            "learning_rate": float(learning_rate),
            "warmup_fraction": float(warmup_fraction),
            "weight_decay": float(weight_decay),
            "mlm_probability": float(mlm_probability),
            "heldout_fraction": float(heldout_fraction),
            "train_layers": int(train_layers), "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "seed": int(seed), "device": device_name, "precision": precision,
            "threads": threads, "optimizer_steps": run.optimizer_steps,
            "n_texts": len(texts), "n_train_texts": len(train_idx),
            "n_heldout_texts": len(held_idx), "n_tokens": int(sum(token_counts)),
            "n_train_windows": run.n_train_windows, "n_heldout_windows": run.n_heldout_windows,
            "windowed_share": run.windowed_share,
            "step_loss": [round(x, 5) for x in run.step_losses],
            "learning_rate_trace": [float(f"{x:.3g}") for x in run.lr_trace],
            "heldout_loss_per_epoch": [round(x, 5) for x in run.heldout_per_epoch],
            "wall_seconds": wall,
        },
        "evaluation": {
            "loss_before": round(loss_before, 5), "loss_after": round(loss_after, 5),
            "perplexity_before": round(math.exp(loss_before), 3),
            "perplexity_after": round(math.exp(loss_after), 3),
            "heldout_texts": len(held_idx),
        },
        # how this encoder gets read when it embeds text; people can edit this
        # per model
        "apply": {"layers": "second_to_last", "pooling": "mean"},
    }
    with atomic_write(out_model_json, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    _write_report(out_report_md, doc, out_model_json, folder,
                  mismatch=_tokenizer_mismatch(texts, tokenizer),
                  device_name=device_name, fallback=reason)
    if verbose:
        ev = doc["evaluation"]
        print(f"[adapt_encoder] {model_name}: held-out perplexity "
              f"{ev['perplexity_before']:.2f} -> {ev['perplexity_after']:.2f} "
              f"after {epochs} epoch(s) on {len(train_idx)} text(s) -> {out_model_json}")
    return out_model_json


def _methods_paragraph(doc: dict) -> str:
    t, ev = doc["training"], doc["evaluation"]
    frozen = (f" Only the top {t['train_layers']} layer(s) and the prediction head "
              f"were trained ({human_count(t['trainable_parameters'])} of "
              f"{human_count(doc['parameters'])} parameters)." if t["train_layers"]
              else "")
    return (
        f"The encoder {doc['base_model']} ({doc['num_hidden_layers']} layers, "
        f"{human_count(doc['parameters'])} parameters) was adapted to the corpus "
        f"by continued masked-language-model pretraining (Gururangan et al., "
        f"2020): {t['n_train_texts']} texts ({t['n_tokens']} tokens in all, "
        f"cut into {t['n_train_windows']} windows of at most {t['max_length']} "
        f"tokens) for {t['epochs']} epoch(s), masking {int(round(100 * t['mlm_probability']))}% "
        f"of tokens per pass, with AdamW (peak learning rate {t['learning_rate']:g}, "
        f"weight decay {t['weight_decay']:g}, linear warm-up over "
        f"{int(round(100 * t['warmup_fraction']))}% of {t['optimizer_steps']} "
        f"optimizer steps then linear decay), an effective batch of "
        f"{t['final_batch_size'] * t['final_grad_accum']} windows, gradient "
        f"clipping at 1.0, seed {t['seed']}, in {t['precision']} precision on "
        f"{t['device']}.{frozen} A held-out {int(round(100 * t['heldout_fraction']))}% "
        f"of texts ({t['n_heldout_texts']}) was never trained on; its "
        f"masked-token perplexity fell from {ev['perplexity_before']:.2f} to "
        f"{ev['perplexity_after']:.2f} (loss {ev['loss_before']:.3f} → "
        f"{ev['loss_after']:.3f}), measured under the same masks. Training took "
        f"{t['wall_seconds']} s.")


def _write_report(path: Path, doc: dict, manifest: Path, folder: Path, *,
                  mismatch: Sequence[Tuple[str, int, int]], device_name: str,
                  fallback: Optional[str]) -> None:
    t, ev = doc["training"], doc["evaluation"]
    lines = [f"# Adapted encoder: {doc['name']}", "",
             "## Methods paragraph", "", _methods_paragraph(doc), ""]
    lines += ["## Settings", ""]
    settings = [("model file", manifest.name), ("weights folder", folder.name),
                ("base model", doc["base_model"]), ("architecture", doc["architecture"]),
                ("layers", doc["num_hidden_layers"]), ("hidden size", doc["hidden_size"]),
                ("parameters", human_count(doc["parameters"]))]
    settings += [(k, t[k]) for k in ("epochs", "max_length", "batch_size", "grad_accum",
                                     "learning_rate", "warmup_fraction", "weight_decay",
                                     "mlm_probability", "heldout_fraction",
                                     "train_layers", "gradient_checkpointing", "seed",
                                     "device", "precision", "threads")]
    if t["oom_restarts"]:
        settings.append(("out-of-memory restarts",
                         f"{t['oom_restarts']} (final batch {t['final_batch_size']} × "
                         f"{t['final_grad_accum']} accumulation)"))
    lines += md_table(["setting", "value"], settings) + [""]
    if fallback:
        lines += [f"*{fallback}*", ""]

    lines += ["## Corpus", "",
              f"{t['n_texts']} texts, {t['n_tokens']} tokens; {t['n_train_texts']} "
              f"trained on ({t['n_train_windows']} windows), {t['n_heldout_texts']} "
              f"held out ({t['n_heldout_windows']} windows). "
              f"{100 * t['windowed_share']:.1f}% of texts were longer than one "
              f"window and were cut into several.", ""]
    if mismatch:
        lines += ["The corpus's most frequent words that the base tokenizer splits "
                  "into pieces -- the vocabulary gap adaptation is closing:", ""]
        lines += md_table(["word", "count", "pieces"], mismatch) + [""]

    lines += ["## Held-out loss and perplexity", "",
              f"Before adapting: loss {ev['loss_before']:.4f}, perplexity "
              f"{ev['perplexity_before']:.2f}. After: loss {ev['loss_after']:.4f}, "
              f"perplexity {ev['perplexity_after']:.2f}. Lower is better; the two are "
              f"measured on the same held-out texts under the same masks.", ""]
    per_epoch = [(0, f"{ev['loss_before']:.4f}", f"{ev['perplexity_before']:.2f}")]
    per_epoch += [(i + 1, f"{v:.4f}", f"{math.exp(v):.2f}")
                  for i, v in enumerate(t["heldout_loss_per_epoch"])]
    lines += md_table(["epoch", "held-out loss", "perplexity"], per_epoch) + [""]
    plot = line_plot(path.with_name(manifest.stem + "_heldout.png"),
                     {"held-out loss": [(i, float(v)) for i, v, _p in
                                        [(r[0], r[1], r[2]) for r in per_epoch]]},
                     title="Held-out loss per epoch (0 = before adapting)",
                     x_label="epoch", y_label="loss")
    if plot is not None:
        lines += [f"![held-out loss]({plot.name})", ""]

    lines += ["## Training loss", ""]
    steps = t["step_loss"]
    if steps:
        n = len(steps)
        picks = sorted({0, n - 1, *[int(round(i * (n - 1) / (_LOSS_TABLE_ROWS - 1)))
                                    for i in range(_LOSS_TABLE_ROWS)]})
        lines += [f"Training loss per optimizer step ({n} steps; every step is in "
                  f"the plot, a sample in the table):", ""]
        lines += md_table(["step", "loss", "learning rate"],
                          [(i + 1, f"{steps[i]:.4f}", f"{t['learning_rate_trace'][i]:.2e}")
                           for i in picks]) + [""]
        plot = line_plot(path.with_name(manifest.stem + "_loss.png"),
                         {"training loss": [(i + 1, float(v)) for i, v in enumerate(steps)]},
                         title="Training loss per optimizer step", x_label="step",
                         y_label="loss", x_scale="linear")
        if plot is not None:
            lines += [f"![training loss]({plot.name})", ""]
        lr = line_plot(path.with_name(manifest.stem + "_lr.png"),
                       {"learning rate": [(i + 1, float(v)) for i, v in
                                          enumerate(t["learning_rate_trace"])]},
                       title="Learning-rate schedule", x_label="step",
                       y_label="learning rate", x_scale="linear")
        if lr is not None:
            lines += [f"![learning rate]({lr.name})", ""]
    lines += ["## Outputs", "",
              f"- `{manifest.name}` — the encoder manifest; add it to your library "
              f"(Settings → Manage text encoders) to embed or fine-tune with it.",
              f"- `{folder.name}/` — the checkpoint (weights, config, tokenizer).", ""]
    lines += ["## Provenance", ""]
    prov = [("wall time (s)", t["wall_seconds"]), ("machine", machine(device_name))]
    prov += list(versions("torch", "transformers", "numpy").items())
    lines += md_table(["item", "value"], prov) + [""]
    write_markdown(path, lines)


# ---------------------------------------------------------------------------
# command line -- we derive this from the function above; see
# helpers.cliargs.CliSpec
# ---------------------------------------------------------------------------

CLI = CliSpec(
    adapt_encoder,
    description="Domain-adaptive pretraining: continue a transformer encoder's "
                "masked-language-model training on your texts and save the result.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
