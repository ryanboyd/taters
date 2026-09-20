"""
The masked-language-model training loop, shared by adapting and pretraining.

Two steps train an encoder on nothing but its own objective -- hide some of
the tokens, predict them: :mod:`adapt_encoder` continues a pretrained model's
training on a corpus, and :mod:`pretrain_encoder` starts a fresh one from
random weights. The loop is the same in both, and it lived inside the adapt
step until the second step needed it. It is lifted out here rather than
copied so that a fix to one cannot silently miss the other.

What is here: windowing the corpus so every token is trained on once per
epoch, the held-out split *by text*, dynamic masking from transformers'
collator, AdamW with linear warm-up and decay, gradient clipping, half
precision on a GPU, a GPU that runs out of memory halving the batch and
doubling the accumulation so the effective batch stays put, and a held-out
loss measured under a fixed masking seed so that two measurements are
comparable. Early stopping is optional: adaptation runs the epochs it was
asked for, pretraining stops when the held-out loss stops falling and keeps
the best checkpoint.

Nothing about the model's origin is known in here. That is the point.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from ._transformer_common import autocast_for, run_with_oom_fallback, truncation_share

__all__ = ["MlmResult", "split_heldout", "train_mlm"]


def split_heldout(text_ids: Sequence[str], fraction: float, seed: int
                  ) -> Tuple[List[int], List[int]]:
    """
    Indices of the training and held-out texts: a seeded shuffle, split by
    *text*, never by chunk, so no held-out sentence has a neighbor from the
    same document in the training set inflating the after-score. At least
    one text is held out whenever there are two or more.
    """
    n = len(text_ids)
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    k = int(round(n * float(fraction)))
    if n >= 2:
        k = min(max(1, k), n - 1)
    else:
        k = 0
    return sorted(order[k:]), sorted(order[:k])


@dataclass
class MlmResult:
    """Everything a manifest and a report need to know about one training run."""
    train_idx: List[int]
    held_idx: List[int]
    token_counts: List[int]
    n_train_windows: int
    n_heldout_windows: int
    windowed_share: float
    #: The held-out loss before any training, or ``None`` when it was not
    #: measured -- a model with random weights has nothing worth measuring.
    loss_before: Optional[float]
    heldout_per_epoch: List[float] = field(default_factory=list)
    step_losses: List[float] = field(default_factory=list)
    lr_trace: List[float] = field(default_factory=list)
    final_batch_size: int = 0
    final_grad_accum: int = 0
    oom_restarts: int = 0
    epochs_run: int = 0
    #: The epoch whose weights the model now carries: the last one, or under
    #: early stopping the one with the lowest held-out loss.
    best_epoch: int = 0
    stopped_early: bool = False

    @property
    def loss_after(self) -> float:
        """The held-out loss of the weights the model ended up with."""
        return self.heldout_per_epoch[self.best_epoch - 1]

    @property
    def optimizer_steps(self) -> int:
        return len(self.step_losses)


def train_mlm(
    *,
    model,
    tokenizer,
    texts: Sequence[str],
    ids: Sequence[str],
    device_name: str,
    precision: str,
    seed: int,
    epochs: int,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    warmup_fraction: float,
    weight_decay: float,
    mlm_probability: float,
    heldout_fraction: float,
    measure_before: bool = True,
    early_stopping: bool = False,
    patience: int = 3,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
    announce: Optional[Callable[[str], None]] = None,
) -> MlmResult:
    """
    Train ``model`` on ``texts`` with the masked-language-model objective.

    The model arrives loaded, on its device, with whatever layers the caller
    wanted frozen already frozen; it leaves trained, in eval mode, carrying
    the weights of ``result.best_epoch``. The caller saves it.

    Parameters
    ----------
    model, tokenizer
        A ``*ForMaskedLM`` model on ``device_name`` and its tokenizer.
    texts, ids
        The corpus, one entry per text, and an identifier per text (the
        held-out split is by text).
    device_name, precision
        As resolved by the caller; ``precision`` is ``"auto"``, ``"fp32"``
        or ``"fp16"``.
    seed
        Seeds the held-out split, the shuffles and the masks. The caller is
        expected to have seeded torch already for anything it built.
    epochs
        Passes over the corpus -- exactly this many, or at most this many
        under ``early_stopping``.
    max_length
        Tokens per training window; texts longer than that are cut into
        several so every token is trained on.
    batch_size, grad_accum
        Windows per forward pass, and passes per optimizer step. A GPU that
        runs out of memory halves the first and scales the second so the
        effective batch stays the same.
    learning_rate, warmup_fraction, weight_decay
        AdamW, with a linear warm-up over that share of the optimizer steps
        and a linear decay to zero after.
    mlm_probability
        The share of tokens masked in each window, freshly drawn each pass.
    heldout_fraction
        The share of texts never trained on, whose loss is measured.
    measure_before
        Measure the held-out loss before training. True for adaptation,
        where the fall from before to after is the result; false for a
        fresh model, whose "before" is a guess at the vocabulary size.
    early_stopping, patience
        Stop when the held-out loss has not improved for ``patience`` epochs,
        and keep the weights from the best one.
    announce
        Called with a short phrase at each stage, for a progress display.

    Returns
    -------
    MlmResult
    """
    import torch
    from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

    from ._transformer_common import chunk_ids

    def say(phrase: str) -> None:
        if announce is not None:
            announce(phrase)

    # chop everything into windows, then split train/held-out by text
    say("tokenizing the corpus")
    body = max(8, int(max_length) - 2)
    # verbose=False: we're about to window these ourselves, so the tokenizer's
    # "longer than the specified maximum sequence length" warning is noise here
    # (and it scared somebody into asking whether their run was broken)
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False,
                        verbose=False)["input_ids"]
    token_counts = [len(e) for e in encoded]
    train_idx, held_idx = split_heldout(ids, heldout_fraction, seed)
    if not held_idx:
        raise ValueError("heldout_fraction left no text held out; the held-out "
                         "measurement needs at least one (two texts or more).")

    def windows_of(indices):
        out = []
        for i in indices:
            for chunk in chunk_ids(encoded[i], body):
                out.append(tokenizer.build_inputs_with_special_tokens(chunk))
        return out

    train_windows = windows_of(train_idx)
    held_windows = windows_of(held_idx)
    if not train_windows:
        raise ValueError("no text was left to train on after the held-out split")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True,
                                               mlm_probability=float(mlm_probability))

    def batches(windows, size: int, order: Optional[List[int]] = None):
        idx = order if order is not None else list(range(len(windows)))
        for start in range(0, len(idx), size):
            chunk = [windows[i] for i in idx[start:start + size]]
            batch = collator([{"input_ids": torch.tensor(w, dtype=torch.long)} for w in chunk])
            yield {k: v.to(device_name) for k, v in batch.items()}

    def evaluate(size: int) -> float:
        """Mean held-out MLM loss under a fixed masking seed, so before and
        after are compared on the same masks."""
        model.eval()
        state = torch.random.get_rng_state()
        torch.manual_seed(int(seed) + 1)
        total, n = 0.0, 0
        with torch.inference_mode(), autocast_for(device_name, precision):
            for batch in batches(held_windows, size):
                out = model(**batch)
                k = int((batch["labels"] != -100).sum().item())
                if k:
                    total += float(out.loss.item()) * k
                    n += k
        torch.random.set_rng_state(state)
        model.train()
        return total / n if n else float("nan")

    loss_before: Optional[float] = None
    if measure_before:
        say("measuring the held-out loss before training")
        loss_before = run_with_oom_fallback(evaluate, batch_size, verbose=verbose,
                                            what="a held-out batch")

    # now the optimizer and the learning-rate schedule
    params = [p for p in model.parameters() if p.requires_grad]
    steps_per_epoch = math.ceil(math.ceil(len(train_windows) / int(batch_size)) / int(grad_accum))
    total_steps = max(1, steps_per_epoch * int(epochs))
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate),
                                  weight_decay=float(weight_decay))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(round(total_steps * float(warmup_fraction))), total_steps)
    scaler = torch.amp.GradScaler("cuda") if (
        device_name.startswith("cuda") and precision in ("auto", "fp16")) else None

    result = MlmResult(train_idx=train_idx, held_idx=held_idx,
                       token_counts=token_counts,
                       n_train_windows=len(train_windows),
                       n_heldout_windows=len(held_windows),
                       windowed_share=truncation_share(token_counts, max_length),
                       loss_before=loss_before)
    state = {"batch": int(batch_size), "accum": int(grad_accum), "restarts": 0,
             "epoch": 0}
    rng = random.Random(int(seed))
    model.train()

    def run_epoch(size: int):
        # if we had to shrink the batch (OOM), we accumulate over more steps
        # so that the effective batch size stays the same
        if size != state["batch"]:
            state["accum"] = max(1, int(round(state["accum"] * state["batch"] / size)))
            state["batch"] = size
            state["restarts"] += 1
        order = list(range(len(train_windows)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        running = 0.0
        for b_i, batch in enumerate(batches(train_windows, size, order)):
            with autocast_for(device_name, precision):
                out = model(**batch)
                loss = out.loss / state["accum"]
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(out.loss.item())
            accumulated += 1
            last = (b_i + 1) * size >= len(order)
            if accumulated == state["accum"] or last:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                if scaler is not None:
                    # if the scaler skipped this step (inf gradients while it's
                    # still finding its scale), we mustn't advance the schedule
                    # either -- otherwise torch warns and we lose the first
                    # learning-rate value
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= before:
                        scheduler.step()
                else:
                    optimizer.step()
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                result.step_losses.append(running / accumulated)
                result.lr_trace.append(float(scheduler.get_last_lr()[0]))
                running, accumulated = 0.0, 0
                if on_progress is not None:
                    on_progress(len(result.step_losses), total_steps,
                                f"epoch {state['epoch'] + 1}/{epochs}: "
                                f"loss {result.step_losses[-1]:.3f}")
        return None

    best_loss = float("inf")
    best_state = None
    since_best = 0
    for epoch in range(int(epochs)):
        state["epoch"] = epoch
        run_with_oom_fallback(lambda size: run_epoch(size), state["batch"],
                              verbose=verbose, what=f"epoch {epoch + 1}")
        held = run_with_oom_fallback(evaluate, state["batch"], verbose=verbose,
                                     what="a held-out batch")
        result.heldout_per_epoch.append(held)
        result.epochs_run = epoch + 1
        if not early_stopping:
            continue
        if held < best_loss - 1e-9:
            best_loss, since_best = held, 0
            result.best_epoch = epoch + 1
            # a copy on the CPU, so that keeping it costs no GPU memory and a
            # later epoch that turns out worse cannot touch it
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= int(patience):
                result.stopped_early = True
                say(f"stopping: the held-out loss has not improved for {patience} epoch(s)")
                break

    if early_stopping and best_state is not None:
        if result.best_epoch != result.epochs_run:
            model.load_state_dict(best_state)
    else:
        result.best_epoch = result.epochs_run
    result.final_batch_size = state["batch"]
    result.final_grad_accum = state["accum"]
    result.oom_restarts = state["restarts"]
    model.eval()
    return result
