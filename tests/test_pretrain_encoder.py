"""
Pretraining an encoder from scratch.

Why this file exists
--------------------
Every model in Taters used to arrive by ``from_pretrained``; this is the first
step that builds one from random weights, and the first that trains a
tokenizer. What is pinned here is what makes that honest: the tokenizer has
seen only the training texts, the weights really do start from the seed and
nothing else, early stopping keeps the *best* epoch rather than the last, and
the result is an ordinary text encoder to everything downstream -- the
library gate, the resolver, the embeddings step -- none of which knows or
cares that nobody pretrained it.

Tiny shapes throughout (two layers, 32 wide, a few hundred BPE symbols) so it
all runs offline in seconds. Random weights on a toy corpus learn nothing
worth having; the tests are about the plumbing.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from taters.helpers.model_spec import describe_encoder, encoder_problem  # noqa: E402
from taters.text import _transformer_common as tc  # noqa: E402
from taters.text.pretrain_encoder import MIN_VOCAB, pretrain_encoder, train_tokenizer  # noqa: E402

FOOD = "potato gravy butter salt dinner supper kitchen recipe pasta soup".split()
WORK = "meeting deadline email boss office budget invoice memo client report".split()


def _corpus(path: Path, n=60, seed=1, *, rare_in_heldout: str = "") -> Path:
    """A planted corpus; ``rare_in_heldout`` is a word placed in exactly one
    text, so a seeded split can be checked for which side it landed on."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            pool = FOOD if i % 2 else WORK
            words = [rng.choice(pool + ["the", "and", "a", "was", "good", "."])
                     for _ in range(rng.randint(12, 30))]
            w.writerow([f"d{i}", " ".join(words)])
    return path


def _pretrain(tmp_path, **kw):
    args = dict(text_cols=["text"],
                id_cols=["text_id"], out_model_json=tmp_path / "models" / "scratch.json",
                preset="custom", layers=2, hidden_size=32, attention_heads=2,
                vocab_size=MIN_VOCAB, max_length=32, epochs=3, patience=2,
                batch_size=8, grad_accum=1, learning_rate=5e-4, device="cpu",
                precision="fp32", verbose=False)
    args.update(kw)
    # only when the caller gave none -- and with an explicit `if`, because
    # `setdefault(..., _corpus(...))` evaluates its argument eagerly and
    # rewrote a caller's corpus at the same path anyway, so a test that
    # planted a word in it was silently testing a corpus without the word
    if "csv_path" not in args:
        args["csv_path"] = _corpus(tmp_path / "c.csv")
    return pretrain_encoder(**args)


# ---------------------------------------------------------------------------
# The tokenizer
# ---------------------------------------------------------------------------

def test_the_tokenizer_is_learned_from_the_training_texts_only(tmp_path):
    """
    A word that occurs in a held-out text and nowhere else must not be a
    whole symbol in the vocabulary. A tokenizer that had seen the held-out
    texts would have leaked a little of them into every measurement made on
    them -- the same reason the split is by text in the first place.

    The first version of this test was vacuous: at the smallest vocabulary
    the marker could not become one symbol even when it *was* trained on, so
    the assertion held either way. Now the test first proves the marker is
    learnable -- a tokenizer trained on every text does make it one symbol --
    and only then checks that the step's tokenizer did not.
    """
    from taters.text._mlm_train import split_heldout

    path = _corpus(tmp_path / "c.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ids = [r["text_id"] for r in rows]
    _train, held = split_heldout(ids, 0.1, 42)
    marker = "qx"                                              # short: few merges to learn
    rows[held[0]]["text"] += " " + " ".join([marker] * 300)    # very frequent, but held out
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["text_id", "text"])
        w.writeheader()
        w.writerows(rows)

    # the premise: given every text, the marker becomes a single symbol
    leaky = train_tokenizer([r["text"] for r in rows], vocab_size=600)
    assert len(leaky(marker, add_special_tokens=False)["input_ids"]) == 1, \
        "the marker is not learnable at this size; the test cannot tell anything"

    out = _pretrain(tmp_path, csv_path=path, epochs=1, vocab_size=600)
    tokenizer = tc.load_encoder(str(out), device="cpu")[1]
    pieces = tokenizer(marker, add_special_tokens=False)["input_ids"]
    assert len(pieces) > 1, (
        "a word seen only in a held-out text became one symbol: the tokenizer "
        "was trained on texts it should never have seen")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["tokenizer"]["trained_on"] == "training texts"


def test_a_vocabulary_too_small_to_learn_anything_is_refused_by_name():
    with pytest.raises(ValueError, match="vocab_size"):
        train_tokenizer(["some words here"], vocab_size=MIN_VOCAB - 1)


# ---------------------------------------------------------------------------
# Random weights, and the seed
# ---------------------------------------------------------------------------

def test_the_weights_start_from_the_seed_and_nothing_else(tmp_path):
    """
    With the learning rate at zero the saved weights *are* the initialization,
    so two seeds must give two models and one seed must give one -- and none
    of it may be anybody's pretrained checkpoint.
    """
    from safetensors.torch import load_file

    a = _pretrain(tmp_path / "a", learning_rate=0.0, epochs=1, seed=1)
    b = _pretrain(tmp_path / "b", learning_rate=0.0, epochs=1, seed=2)
    c = _pretrain(tmp_path / "c", learning_rate=0.0, epochs=1, seed=1)
    wa = load_file(a.with_suffix(".encoder") / "model.safetensors")
    wb = load_file(b.with_suffix(".encoder") / "model.safetensors")
    wc = load_file(c.with_suffix(".encoder") / "model.safetensors")
    embed = next(k for k in wa if "word_embeddings" in k)
    assert not torch.equal(wa[embed], wb[embed]), "two seeds gave the same weights"
    assert torch.equal(wa[embed], wc[embed]), "one seed gave two different models"
    doc = json.loads(a.read_text(encoding="utf-8"))
    assert doc["trained_from"] == "scratch"
    assert "base_model" not in doc


def test_a_bad_preset_and_a_corpus_of_one_text_are_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="preset"):
        _pretrain(tmp_path, preset="enormous")
    with pytest.raises(ValueError, match="custom.*needs"):
        _pretrain(tmp_path, preset="custom", hidden_size=None)
    with pytest.raises(ValueError, match="multiple of attention_heads"):
        _pretrain(tmp_path, hidden_size=30, attention_heads=4)
    one = tmp_path / "one.csv"
    one.write_text("text_id,text\nd0,just one text here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a corpus"):
        _pretrain(tmp_path, csv_path=one)


# ---------------------------------------------------------------------------
# Early stopping keeps the best epoch
# ---------------------------------------------------------------------------

def test_early_stopping_keeps_the_best_epoch_not_the_last(tmp_path):
    """
    ``epochs`` is a ceiling. The run may stop before it, and whichever epoch
    it stops at, the weights saved are the ones with the lowest held-out
    loss -- checked by reloading them and measuring again under the same
    masks, which must reproduce ``loss_final`` exactly.

    On a real corpus the loss usually just keeps falling, so this checks the
    bookkeeping a real run *can* show -- the manifest is consistent and the
    saved weights reproduce the loss it reports. The case where the best
    epoch is not the last is forced deterministically in
    :func:`test_the_best_epoch_is_kept_when_a_later_one_is_worse`.
    """
    out = _pretrain(tmp_path, epochs=6, patience=2, learning_rate=2e-3)
    doc = json.loads(out.read_text(encoding="utf-8"))
    t, ev = doc["training"], doc["evaluation"]
    per_epoch = t["heldout_loss_per_epoch"]
    assert 1 <= t["best_epoch"] <= t["epochs_run"] <= 6
    assert per_epoch[t["best_epoch"] - 1] == min(per_epoch)
    assert ev["loss_final"] == pytest.approx(min(per_epoch), abs=1e-5)

    # the saved weights really are the best epoch's: measured again under the
    # same masks they give the same held-out loss
    from taters.text._mlm_train import train_mlm

    model, tokenizer, _res, dev, _r = tc.load_encoder(str(out), device="cpu", for_mlm=True)
    rows = list(csv.DictReader((tmp_path / "c.csv").open(encoding="utf-8")))
    again = train_mlm(model=model, tokenizer=tokenizer,
                      texts=[r["text"] for r in rows], ids=[r["text_id"] for r in rows],
                      device_name=dev, precision="fp32", seed=42, epochs=1,
                      max_length=32, batch_size=8, grad_accum=1, learning_rate=0.0,
                      warmup_fraction=0.0, weight_decay=0.0, mlm_probability=0.15,
                      heldout_fraction=0.1, measure_before=True, early_stopping=False,
                      verbose=False)
    assert again.loss_before == pytest.approx(ev["loss_final"], abs=1e-4), \
        "the saved weights are not the epoch the manifest says they are"


# ---------------------------------------------------------------------------
# An ordinary encoder to everything downstream
# ---------------------------------------------------------------------------

def test_the_result_is_an_ordinary_text_encoder_to_everything_downstream(tmp_path):
    out = _pretrain(tmp_path)
    assert encoder_problem(out) == ""
    resolved = tc.resolve_encoder(str(out))
    assert resolved.kind == "encoder"
    model, tokenizer, _res, dev, _r = tc.load_encoder(str(out), device="cpu")
    vectors, counts = tc.encode_sentences(model, tokenizer, ["potato gravy for dinner"],
                                          device_name=dev, layers="second_to_last",
                                          pooling="mean", max_length=32, batch_size=4)
    assert vectors.shape == (1, 32) and len(counts) == 1
    assert tokenizer.model_max_length == 32, "the finished model's real limit is not recorded"

    label, note = describe_encoder(out)
    assert label == "scratch [trained from scratch]"
    assert "[?]" not in label
    assert "2 layers" in note and "perplexity" in note and "→" not in note


def test_the_report_says_it_was_trained_from_nothing_and_from_how_little(tmp_path):
    out = _pretrain(tmp_path)
    report = out.with_name("scratch_report.md").read_text(encoding="utf-8")
    assert report.startswith("# Encoder trained from scratch: scratch")
    assert "## Methods paragraph" in report
    assert "random initialization" in report and "Liu" in report and "Devlin" in report
    assert "learned from the training texts alone" in report
    assert "Before adapting" not in report and "| 0 |" not in report, \
        "a random model's 'before' perplexity is not a measurement worth reporting"
    assert "A note on size" in report, "a toy corpus was not called one"
    assert "![held-out loss](scratch_heldout.png)" in report
    assert (out.parent / "scratch_heldout.png").exists()
    assert "| best |" in report or "best" in report
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["training"]["devices"] == 1 or doc["training"]["device"] == "cpu"


def test_an_existing_encoder_is_returned_untouched_unless_asked(tmp_path):
    first = _pretrain(tmp_path)
    before = first.read_bytes()
    again = _pretrain(tmp_path, seed=7)
    assert again == first and again.read_bytes() == before
    redone = _pretrain(tmp_path, seed=7, overwrite_existing=True)
    assert json.loads(redone.read_text(encoding="utf-8"))["training"]["seed"] == 7


class _Drifter(torch.nn.Module):
    """
    A one-parameter stand-in for an encoder whose training loss falls while
    its held-out loss rises: in training mode the loss is ``-p``, a drift with
    no floor, so every optimizer step pushes ``p`` up; in eval mode the loss
    is ``p``², so every epoch of training makes the held-out measurement
    worse. (A loss that pulled ``p`` toward a target overshot it and came
    back, and the held-out loss stopped being monotone.) Deterministic, and
    exactly the situation "keep the best epoch" exists for. Real models on
    toy corpora refuse to produce it -- however absurd the learning rate,
    they recover.
    """

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_):
        import types

        loss = -self.p if self.training else self.p ** 2
        return types.SimpleNamespace(loss=loss)


def test_the_best_epoch_is_kept_when_a_later_one_is_worse(tmp_path):
    """
    The contract of early stopping, pinned on a model built to break it: the
    held-out loss is lowest after epoch one and rises thereafter, so the run
    must stop after ``patience`` more epochs and the weights it leaves behind
    must be epoch one's -- not the last epoch's, which are the ones a loop
    without the restore would keep.
    """
    from taters.text._mlm_train import train_mlm

    texts = [" ".join(FOOD)] * 20
    tokenizer = train_tokenizer(texts, vocab_size=MIN_VOCAB)
    model = _Drifter()
    run = train_mlm(model=model, tokenizer=tokenizer, texts=texts,
                    ids=[f"d{i}" for i in range(20)], device_name="cpu",
                    precision="fp32", seed=0, epochs=10, max_length=16,
                    batch_size=4, grad_accum=1, learning_rate=0.1,
                    warmup_fraction=0.0, weight_decay=0.0, mlm_probability=0.15,
                    heldout_fraction=0.1, measure_before=False,
                    early_stopping=True, patience=2, verbose=False)

    losses = run.heldout_per_epoch
    assert losses == sorted(losses) and losses[0] < losses[-1], \
        "the premise failed: the held-out loss did not rise every epoch"
    assert run.best_epoch == 1
    assert run.stopped_early and run.epochs_run == 3, "patience of two after the best"
    assert run.loss_after == losses[0]
    # the weights are epoch one's: p² must equal the first held-out loss (to
    # float32 precision, which is what the loss was measured in) and be
    # nowhere near the last epoch's
    kept = float(model.p) ** 2
    assert kept == pytest.approx(losses[0], rel=1e-5), (
        "the model carries the last epoch's weights, not the best epoch's")
    assert kept < losses[-1] * 0.9, "the kept weights are the last epoch's"
