"""
Pretrain a transformer encoder from scratch on your own texts.

Adapting (:mod:`adapt_encoder`) starts from a model somebody else pretrained
and continues its training on your corpus. This starts from nothing: random
weights, and a tokenizer built from your texts rather than borrowed from web
text -- so a corpus of clinical notes, forum posts or eighteenth-century
letters gets a vocabulary of its own words, whole, instead of one that
shatters them into pieces.

It is heavy-duty, and the step says so before it runs. A language model
learns language from *quantity*: below a few million words of text the
result will be worse at everything than any pretrained model you could have
adapted, and even a modest model takes hours on one GPU and days on a CPU.
The corpus size is left to your judgment -- testing the machinery on a small
one is a legitimate thing to do -- but the report says what it was trained
on, so nobody mistakes a toy for a tool.

The objective is masked-language modeling (Devlin et al., 2019) in the
RoBERTa style (Liu et al., 2019): byte-level BPE, dynamic masking, no
next-sentence task. The loop is the one adaptation uses
(:mod:`_mlm_train`), with two things pretraining needs and adaptation does
not: early stopping on the held-out loss, keeping the best epoch, and a
learning-rate schedule sized for a model that knows nothing yet.

* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training
  of deep bidirectional transformers for language understanding. *NAACL 2019*.
* Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining
  approach. *arXiv:1907.11692*.

What it writes: ``<name>.json`` (a ``taters-encoder`` manifest, the same
kind adaptation writes, so the library, the embeddings step and fine-tuning
treat it like any other encoder), the checkpoint folder ``<name>.encoder``
beside it, and ``<name>_report.md``.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.progress import announce
from ..helpers.text_gather import resolve_analysis_ready
from ._mlm_train import split_heldout, train_mlm
from ._report import line_plot, machine, md_table, versions, write_markdown
from ._transformer_common import (ENCODER_KIND, PRESETS, fresh_encoder, human_count,
                                  payload_digests, preset_config, set_threads,
                                  torch_missing_reason, unwrap)

__all__ = ["pretrain_encoder", "train_tokenizer", "SPECIAL_TOKENS"]

PathLike = Union[str, Path]

#: The manifest format -- the same as adaptation's, because the shape is the
#: same. A from-scratch manifest adds keys (``trained_from``, ``tokenizer``);
#: it takes none away.
ENCODER_FORMAT = 1

#: The special tokens a RoBERTa-style tokenizer needs, in the order that gives
#: them the ids the model's config expects (pad is 1, mask is last).
SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>", "<mask>")

#: Below this a byte-level BPE has barely more than its 256 byte symbols and
#: nothing is gained over characters.
MIN_VOCAB = 300

#: How many training-loss points the report table shows (the plot shows all).
_LOSS_TABLE_ROWS = 12


def train_tokenizer(texts: Sequence[str], *, vocab_size: int, min_frequency: int = 2):
    """
    A byte-level BPE tokenizer learned from ``texts`` alone.

    Byte-level, so no text can contain a character it cannot represent --
    every byte is a symbol before any merge is learned -- and RoBERTa-style,
    so the model's config and the collator's masking agree with it without
    special handling.

    Parameters
    ----------
    texts : sequence of str
        The texts to learn merges from. Pass the *training* texts only: a
        tokenizer that has seen the held-out texts has leaked a little of
        them into every measurement made on them.
    vocab_size : int
        How many symbols, byte symbols and special tokens included.
    min_frequency : int, default 2
        A merge has to occur this often to be kept.

    Returns
    -------
    transformers.RobertaTokenizerFast
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaTokenizerFast

    if int(vocab_size) < MIN_VOCAB:
        raise ValueError(f"vocab_size must be at least {MIN_VOCAB}: a byte-level "
                         f"tokenizer starts with 256 byte symbols, and below "
                         f"{MIN_VOCAB} it has learned almost nothing beyond them.")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(list(texts), vocab_size=int(vocab_size),
                            min_frequency=int(min_frequency),
                            special_tokens=list(SPECIAL_TOKENS))
    bos, pad, eos, unk, mask = SPECIAL_TOKENS
    return RobertaTokenizerFast(tokenizer_object=bpe, bos_token=bos, eos_token=eos,
                                sep_token=eos, cls_token=bos, unk_token=unk,
                                pad_token=pad, mask_token=mask)


def pretrain_encoder(
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

    # ----- what to build -----
    name: Optional[str] = None,
    preset: str = "small",
    vocab_size: Optional[int] = None,
    layers: Optional[int] = None,
    hidden_size: Optional[int] = None,
    attention_heads: Optional[int] = None,
    max_length: int = 256,

    # ----- how to train it -----
    epochs: int = 40,
    patience: int = 3,
    batch_size: int = 32,
    grad_accum: int = 2,
    learning_rate: float = 5e-4,
    warmup_fraction: float = 0.1,
    weight_decay: float = 0.01,
    mlm_probability: float = 0.15,
    heldout_fraction: float = 0.1,
    gradient_checkpointing: bool = False,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    precision: Literal["auto", "fp32", "fp16"] = "auto",
    seed: int = 42,
) -> Path:
    """
    Train a transformer encoder, and its tokenizer, from nothing but these texts.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as every other text step: a spreadsheet of
        texts, a folder of documents, or a prebuilt analysis-ready CSV.
    out_model_json : str or Path, optional
        The encoder manifest; the checkpoint lands beside it as
        ``<stem>.encoder``. Default ``./models/scratch_encoder.json``.
    out_report_md : str or Path, optional
        The training report, default ``<stem>_report.md`` beside the model.
    overwrite_existing : bool, default False
        If False and the manifest exists, return it untouched.
    workers : int, default 0
        Parallel processes for the gather, and the CPU threads torch may
        use. 0 means automatic.
    text_cols, id_cols, mode, group_by, pattern
        Gather options, as in every text step.
    name : str, optional
        The encoder's name in menus; default the manifest's file stem.
    preset : {"small", "base", "custom"}, default "small"
        The architecture. ``small`` is 4 layers, 256 wide (about 10M
        parameters, an overnight run on one GPU); ``base`` is the
        BERT/RoBERTa shape, 12 layers, 768 wide (about 110M, days).
        ``custom`` takes ``layers``, ``hidden_size`` and ``attention_heads``.
    vocab_size : int, optional
        Symbols in the tokenizer. Default follows the preset (8,000 for
        small, 30,000 for base). At least 300.
    layers, hidden_size, attention_heads : int, optional
        The numbers behind a ``custom`` preset; ignored otherwise. The
        feed-forward width is four times ``hidden_size``, as it is in BERT.
    max_length : int, default 256
        Tokens per training window, and the longest input the finished
        model will read. Longer texts are cut into windows for training.
    epochs : int, default 40
        The *most* passes over the corpus. Training stops earlier when the
        held-out loss has not fallen for ``patience`` epochs, and keeps the
        weights from the best epoch.
    patience : int, default 3
        How many epochs without improvement end the run.
    batch_size : int, default 32
        Windows per forward pass; halved automatically if the GPU runs out
        of memory, with ``grad_accum`` scaled to compensate.
    grad_accum : int, default 2
        Batches accumulated per optimizer step.
    learning_rate : float, default 5e-4
        The peak learning rate of AdamW -- ten times adaptation's, because
        there is nothing here worth preserving yet.
    warmup_fraction : float, default 0.1
        The share of optimizer steps warming the learning rate up from zero.
    weight_decay : float, default 0.01
        AdamW's weight decay.
    mlm_probability : float, default 0.15
        The share of tokens masked in each window, freshly drawn each pass.
    heldout_fraction : float, default 0.1
        The share of *texts* never trained on -- and never shown to the
        tokenizer -- whose loss decides when to stop.
    gradient_checkpointing : bool, default False
        Trade compute for memory on a small card.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where training runs. With more than one GPU visible, all of them
        are used from this one process (``torch.nn.DataParallel``).
    precision : {"auto", "fp32", "fp16"}, default "auto"
        Half precision on a GPU (auto), always full, or always half.
    seed : int, default 42
        Seeds the weights, the held-out split, the shuffles and the masks.

    Returns
    -------
    Path
        ``out_model_json``.
    """
    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, not {preset!r}")
    if int(epochs) < 1 or int(batch_size) < 1 or int(grad_accum) < 1 or int(patience) < 1:
        raise ValueError("epochs, patience, batch_size and grad_accum must be at least 1")
    if not (0.0 < float(mlm_probability) < 1.0):
        raise ValueError("mlm_probability must be between 0 and 1")
    if not (0.0 <= float(heldout_fraction) < 1.0):
        raise ValueError("heldout_fraction must be at least 0 and below 1")
    if int(max_length) < 16:
        raise ValueError("max_length must be at least 16 tokens")
    shape = dict(PRESETS[preset])
    if preset == "custom":
        for key, value in (("layers", layers), ("hidden_size", hidden_size),
                           ("attention_heads", attention_heads)):
            if value is None:
                raise ValueError(f"preset 'custom' needs {key}")
            shape[key] = int(value)
        if shape["hidden_size"] % shape["attention_heads"]:
            raise ValueError("hidden_size must be a multiple of attention_heads")
    vocab = int(vocab_size) if vocab_size is not None else int(shape["vocab_size"])
    if vocab < MIN_VOCAB:
        raise ValueError(f"vocab_size must be at least {MIN_VOCAB}")
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
        Path.cwd() / "models" / "scratch_encoder.json"
    if out_model_json.is_file() and not overwrite_existing:
        if verbose:
            print("Encoder already exists; returning existing file.")
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
            f"only {len(texts)} text(s) with words in them; pretraining an encoder "
            f"needs a corpus (at least two texts -- realistically, millions of words).")

    import torch

    torch.manual_seed(int(seed))
    threads = set_threads(workers)

    # the tokenizer learns from the training texts only. the split here is the
    # same seeded split the loop makes again below, so the two agree exactly.
    train_idx, held_idx = split_heldout(ids, heldout_fraction, seed)
    announce(on_progress, f"learning a {vocab:,}-symbol vocabulary from "
                          f"{len(train_idx)} texts")
    tokenizer = train_tokenizer([texts[i] for i in train_idx], vocab_size=vocab)
    # the finished model's real limit, so that anything reading it later
    # (the embeddings step, fine-tuning) windows at the right length
    tokenizer.model_max_length = int(max_length)

    announce(on_progress, f"building a {preset} encoder from random weights")
    config = preset_config(shape, vocab_size=tokenizer.vocab_size,
                           max_length=int(max_length), tokenizer=tokenizer)
    model, device_name, reason, devices = fresh_encoder(
        config, device=device, seed=int(seed), verbose=verbose)
    if gradient_checkpointing and hasattr(unwrap(model), "gradient_checkpointing_enable"):
        unwrap(model).gradient_checkpointing_enable()

    run = train_mlm(
        model=model, tokenizer=tokenizer, texts=texts, ids=ids,
        device_name=device_name, precision=precision, seed=int(seed),
        epochs=int(epochs), max_length=int(max_length),
        batch_size=int(batch_size), grad_accum=int(grad_accum),
        learning_rate=float(learning_rate), warmup_fraction=float(warmup_fraction),
        weight_decay=float(weight_decay), mlm_probability=float(mlm_probability),
        heldout_fraction=float(heldout_fraction),
        # a random model's "before" is a guess at the vocabulary size, not a
        # measurement worth reporting; and epochs is a ceiling, not a plan
        measure_before=False, early_stopping=True, patience=int(patience),
        verbose=verbose, on_progress=on_progress,
        announce=lambda phrase: announce(on_progress, phrase))
    assert run.train_idx == train_idx and run.held_idx == held_idx, \
        "the tokenizer and the loop split the corpus differently"

    announce(on_progress, "saving the encoder")
    folder = out_model_json.with_suffix(".encoder")
    if folder.exists():
        import shutil

        shutil.rmtree(folder)
    bare = unwrap(model)
    bare.eval()
    bare.save_pretrained(folder, safe_serialization=True)
    tokenizer.save_pretrained(folder)
    cfg = bare.config
    model_name = str(name or out_model_json.stem)
    wall = round(time.time() - started, 1)
    loss_final = run.loss_after
    doc = {
        "kind": ENCODER_KIND, "format": ENCODER_FORMAT, "name": model_name,
        "payload": [folder.name],
        "payload_digests": payload_digests(folder),
        # no base model: this is what tells a reader, and the library row,
        # that nothing here was pretrained by anyone else
        "trained_from": "scratch",
        "preset": preset,
        "architecture": str(getattr(cfg, "model_type", "")),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "attention_heads": int(cfg.num_attention_heads),
        "vocab_size": int(cfg.vocab_size),
        "parameters": int(sum(p.numel() for p in bare.parameters())),
        "tokenizer": {"kind": "byte_bpe", "vocab_size": int(tokenizer.vocab_size),
                      "trained_on": "training texts", "min_frequency": 2},
        "text": {"max_length": int(max_length)},
        "training": {
            "epochs": int(epochs), "epochs_run": run.epochs_run,
            "best_epoch": run.best_epoch, "stopped_early": run.stopped_early,
            "patience": int(patience),
            "max_length": int(max_length),
            "batch_size": int(batch_size), "grad_accum": int(grad_accum),
            "final_batch_size": run.final_batch_size, "final_grad_accum": run.final_grad_accum,
            "oom_restarts": run.oom_restarts,
            "learning_rate": float(learning_rate),
            "warmup_fraction": float(warmup_fraction),
            "weight_decay": float(weight_decay),
            "mlm_probability": float(mlm_probability),
            "heldout_fraction": float(heldout_fraction),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "seed": int(seed), "device": device_name, "devices": devices,
            "precision": precision, "threads": threads,
            "optimizer_steps": run.optimizer_steps,
            "n_texts": len(texts), "n_train_texts": len(run.train_idx),
            "n_heldout_texts": len(run.held_idx),
            "n_tokens": int(sum(run.token_counts)),
            "n_train_windows": run.n_train_windows, "n_heldout_windows": run.n_heldout_windows,
            "windowed_share": run.windowed_share,
            "step_loss": [round(x, 5) for x in run.step_losses],
            "learning_rate_trace": [float(f"{x:.3g}") for x in run.lr_trace],
            "heldout_loss_per_epoch": [round(x, 5) for x in run.heldout_per_epoch],
            "wall_seconds": wall,
        },
        "evaluation": {
            "loss_final": round(loss_final, 5),
            "perplexity_final": round(math.exp(loss_final), 3),
            "heldout_texts": len(run.held_idx),
        },
        # how this encoder gets read when it embeds text; people can edit this
        # per model
        "apply": {"layers": "second_to_last", "pooling": "mean"},
    }
    with atomic_write(out_model_json, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    _write_report(out_report_md, doc, out_model_json, folder,
                  device_name=device_name, fallback=reason)
    if verbose:
        ev = doc["evaluation"]
        print(f"[pretrain_encoder] {model_name}: held-out perplexity "
              f"{ev['perplexity_final']:.2f} after {run.epochs_run} epoch(s) "
              f"(best: {run.best_epoch}) on {len(run.train_idx)} text(s) -> {out_model_json}")
    return out_model_json


def _methods_paragraph(doc: dict) -> str:
    t, ev, tok = doc["training"], doc["evaluation"], doc["tokenizer"]
    where = t["device"] + (f" ({t['devices']} devices)" if t["devices"] > 1 else "")
    stopped = (f" Training stopped after epoch {t['epochs_run']} of at most "
               f"{t['epochs']}, the held-out loss having not improved for "
               f"{t['patience']} epochs; the weights are those of epoch "
               f"{t['best_epoch']}, the best." if t["stopped_early"]
               else f" All {t['epochs']} epochs were run; the weights are those of "
                    f"epoch {t['best_epoch']}, the best by held-out loss.")
    return (
        f"A {doc['architecture']}-style encoder ({doc['num_hidden_layers']} layers, "
        f"{doc['hidden_size']} wide, {doc['attention_heads']} attention heads, "
        f"{human_count(doc['parameters'])} parameters; the \"{doc['preset']}\" "
        f"preset) was trained from random initialization by masked-language "
        f"modeling (Devlin et al., 2019; Liu et al., 2019). Its tokenizer, a "
        f"byte-level BPE with {tok['vocab_size']:,} symbols, was learned from the "
        f"training texts alone. The corpus was {t['n_train_texts']} texts "
        f"({t['n_tokens']:,} tokens in all, cut into {t['n_train_windows']} windows "
        f"of at most {t['max_length']} tokens), masking "
        f"{int(round(100 * t['mlm_probability']))}% of tokens per pass, with AdamW "
        f"(peak learning rate {t['learning_rate']:g}, weight decay "
        f"{t['weight_decay']:g}, linear warm-up over "
        f"{int(round(100 * t['warmup_fraction']))}% of the scheduled optimizer "
        f"steps then linear decay), an effective batch of "
        f"{t['final_batch_size'] * t['final_grad_accum']} windows, gradient clipping "
        f"at 1.0, seed {t['seed']}, in {t['precision']} precision on {where}."
        f"{stopped} A held-out {int(round(100 * t['heldout_fraction']))}% of texts "
        f"({t['n_heldout_texts']}) was never trained on and never shown to the "
        f"tokenizer; its final masked-token perplexity was {ev['perplexity_final']:.2f} "
        f"(loss {ev['loss_final']:.3f}). Training took {t['wall_seconds']} s.")


def _write_report(path: Path, doc: dict, manifest: Path, folder: Path, *,
                  device_name: str, fallback: Optional[str]) -> None:
    t, ev = doc["training"], doc["evaluation"]
    lines = [f"# Encoder trained from scratch: {doc['name']}", "",
             "## Methods paragraph", "", _methods_paragraph(doc), ""]
    lines += ["## Settings", ""]
    settings: List[Tuple[str, object]] = [
        ("model file", manifest.name), ("weights folder", folder.name),
        ("trained from", "scratch (random initialization)"),
        ("preset", doc["preset"]), ("architecture", doc["architecture"]),
        ("layers", doc["num_hidden_layers"]), ("hidden size", doc["hidden_size"]),
        ("attention heads", doc["attention_heads"]),
        ("parameters", human_count(doc["parameters"])),
        ("tokenizer", f"byte-level BPE, {doc['tokenizer']['vocab_size']:,} symbols, "
                      f"learned from the training texts")]
    settings += [(k, t[k]) for k in ("epochs", "epochs_run", "best_epoch", "patience",
                                     "max_length", "batch_size", "grad_accum",
                                     "learning_rate", "warmup_fraction", "weight_decay",
                                     "mlm_probability", "heldout_fraction",
                                     "gradient_checkpointing", "seed", "device",
                                     "devices", "precision", "threads")]
    if t["oom_restarts"]:
        settings.append(("out-of-memory restarts",
                         f"{t['oom_restarts']} (final batch {t['final_batch_size']} × "
                         f"{t['final_grad_accum']} accumulation)"))
    lines += md_table(["setting", "value"], settings) + [""]
    if fallback:
        lines += [f"*{fallback}*", ""]

    lines += ["## Corpus", "",
              f"{t['n_texts']} texts, {t['n_tokens']:,} tokens; {t['n_train_texts']} "
              f"trained on ({t['n_train_windows']} windows), {t['n_heldout_texts']} "
              f"held out ({t['n_heldout_windows']} windows). "
              f"{100 * t['windowed_share']:.1f}% of texts were longer than one "
              f"window and were cut into several.", ""]
    if t["n_tokens"] < 1_000_000:
        lines += ["**A note on size.** A language model learns language from "
                  "quantity, and this corpus is under a million tokens. The "
                  "encoder will have learned this corpus's habits, not the "
                  "language; for most purposes an adapted pretrained encoder "
                  "would be the stronger choice. Judge it by what it does on your "
                  "task, not by the perplexity below.", ""]

    lines += ["## Held-out loss and perplexity", "",
              f"Final: loss {ev['loss_final']:.4f}, perplexity "
              f"{ev['perplexity_final']:.2f} on {t['n_heldout_texts']} held-out texts "
              f"the model and its tokenizer never saw. Lower is better; every epoch "
              f"is measured under the same masks.", ""]
    per_epoch = [(i + 1, f"{v:.4f}", f"{math.exp(v):.2f}",
                  "best" if i + 1 == t["best_epoch"] else "")
                 for i, v in enumerate(t["heldout_loss_per_epoch"])]
    lines += md_table(["epoch", "held-out loss", "perplexity", ""], per_epoch) + [""]
    plot = line_plot(path.with_name(manifest.stem + "_heldout.png"),
                     {"held-out loss": [(i + 1, float(v)) for i, v in
                                        enumerate(t["heldout_loss_per_epoch"])]},
                     title="Held-out loss per epoch", x_label="epoch", y_label="loss")
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
              f"(Wrangle Language Models → import/export) to embed or fine-tune with it.",
              f"- `{folder.name}/` — the checkpoint (weights, config, tokenizer).", ""]
    lines += ["## Provenance", ""]
    prov = [("wall time (s)", t["wall_seconds"]), ("machine", machine(device_name))]
    prov += list(versions("torch", "transformers", "tokenizers", "numpy").items())
    lines += md_table(["item", "value"], prov) + [""]
    write_markdown(path, lines)


# ---------------------------------------------------------------------------
# command line -- we derive this from the function above; see
# helpers.cliargs.CliSpec
# ---------------------------------------------------------------------------

CLI = CliSpec(
    pretrain_encoder,
    description="Pretrain a transformer encoder, and its tokenizer, from scratch "
                "on your texts by masked-language modeling, and save the result.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
