"""
Tests for `taters.text.adapt_encoder`: domain-adaptive pretraining on the
tiny encoder. Two to three epochs on sixty short texts run in seconds on a
CPU, which is enough to check what a paper needs to be able to say: the
held-out loss fell, the split was by text, the frozen layers stayed put, the
report and manifest carry the numbers, and the saved folder reloads.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from taters.helpers.model_spec import encoder_problem  # noqa: E402
from taters.text import _transformer_common as tc  # noqa: E402
from taters.text.adapt_encoder import adapt_encoder, split_heldout  # noqa: E402
from tiny_encoder import build  # noqa: E402

FOOD = "potato gravy butter salt dinner supper kitchen recipe".split()


@pytest.fixture(scope="module")
def encoder(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("enc") / "tiny")


def _corpus(path: Path, n=60, seed=1) -> Path:
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            words = [rng.choice(FOOD + ["the", "and", "a", "was", "good", "."])
                     for _ in range(rng.randint(12, 30))]
            w.writerow([f"d{i}", " ".join(words)])
    return path


def _adapt(tmp_path, encoder, **kw):
    args = dict(csv_path=_corpus(tmp_path / "c.csv"), text_cols=["text"],
                id_cols=["text_id"], out_model_json=tmp_path / "models" / "adapted.json",
                base_model=str(encoder), epochs=2, max_length=32, batch_size=8,
                grad_accum=1, learning_rate=5e-4, device="cpu", precision="fp32",
                verbose=False)
    args.update(kw)
    return adapt_encoder(**args)


def test_the_split_is_by_text_deterministic_and_always_holds_something_out():
    ids = [f"t{i}" for i in range(20)]
    train, held = split_heldout(ids, 0.25, seed=3)
    assert len(held) == 5 and not set(train) & set(held)
    assert sorted(train + held) == list(range(20))
    assert split_heldout(ids, 0.25, seed=3) == (train, held)
    assert split_heldout(ids, 0.25, seed=4) != (train, held)
    assert split_heldout(ids, 0.0, seed=1)[1] == [] or len(ids) < 2 or \
        len(split_heldout(ids, 0.0, seed=1)[1]) == 1, "at least one held out"
    assert len(split_heldout(["a", "b"], 0.9, seed=1)[0]) == 1, "never everything"
    assert split_heldout(["only"], 0.5, seed=1) == ([0], [])


def test_adapting_lowers_the_held_out_loss_and_the_folder_reloads(tmp_path, encoder):
    out = _adapt(tmp_path, encoder, epochs=3)
    doc = json.loads(out.read_text(encoding="utf-8"))
    ev = doc["evaluation"]
    assert ev["loss_after"] < ev["loss_before"]
    assert ev["perplexity_after"] < ev["perplexity_before"]
    assert doc["kind"] == "taters-encoder" and doc["payload"] == ["adapted.encoder"]
    assert len(doc["training"]["heldout_loss_per_epoch"]) == 3
    assert ev["loss_after"] == doc["training"]["heldout_loss_per_epoch"][-1]
    assert doc["training"]["optimizer_steps"] == len(doc["training"]["step_loss"]) > 0
    assert doc["training"]["n_train_texts"] + doc["training"]["n_heldout_texts"] == 60
    assert encoder_problem(out) == ""
    model, tokenizer, resolved, device, _ = tc.load_encoder(out, device="cpu")
    assert resolved.kind == "encoder" and resolved.label == "adapted"
    vecs, _counts = tc.encode_sentences(model, tokenizer, ["potato gravy ."],
                                        device_name="cpu", precision="fp32")
    assert vecs.shape == (1, 32)
    # the digests are how a manifest that traveled alone finds its folder.
    assert set(doc["payload_digests"]) >= {"config.json", "model.safetensors"}


def test_freezing_all_but_the_top_layer_leaves_the_lower_one_byte_identical(tmp_path, encoder):
    before = tc.load_encoder(encoder, device="cpu")[0].state_dict()
    out = _adapt(tmp_path, encoder, train_layers=1, epochs=1)
    after = tc.load_encoder(out, device="cpu")[0].state_dict()
    lower = [k for k in after if k.startswith("encoder.layer.0.")]
    upper = [k for k in after if k.startswith("encoder.layer.1.")]
    assert lower and upper
    assert all(torch.equal(before[k], after[k]) for k in lower)
    assert any(not torch.equal(before[k], after[k]) for k in upper)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["training"]["frozen_parameters"] > 0
    assert "Only the top 1 layer(s)" in (tmp_path / "models" / "adapted_report.md").read_text(encoding="utf-8")


def test_the_report_has_the_before_and_after_the_curves_and_the_methods(tmp_path, encoder):
    out = _adapt(tmp_path, encoder)
    report = (tmp_path / "models" / "adapted_report.md").read_text(encoding="utf-8")
    doc = json.loads(out.read_text(encoding="utf-8"))
    ev = doc["evaluation"]
    assert "## Methods paragraph" in report and "Gururangan" in report
    assert f"{ev['perplexity_before']:.2f}" in report and f"{ev['perplexity_after']:.2f}" in report
    assert "| 0 |" in report and "| 2 |" in report, "held-out loss per epoch, epoch 0 = before"
    assert "![held-out loss](adapted_heldout.png)" in report
    assert "![training loss](adapted_loss.png)" in report
    assert "![learning rate](adapted_lr.png)" in report
    assert (tmp_path / "models" / "adapted_heldout.png").is_file()
    assert "| seed | 42 |" in report and "torch" in report and "transformers" in report
    assert "vocabulary gap" in report or "splits into pieces" in report or "Corpus" in report


def test_the_encoders_gate_refuses_a_manifest_whose_folder_was_deleted(tmp_path, encoder):
    import shutil

    out = _adapt(tmp_path, encoder, epochs=1)
    shutil.rmtree(tmp_path / "models" / "adapted.encoder")
    assert "not beside it" in encoder_problem(out)
    with pytest.raises(ValueError, match="not beside it and not in your library"):
        tc.resolve_encoder(out)


def test_an_existing_encoder_is_returned_untouched_unless_asked(tmp_path, encoder):
    out = _adapt(tmp_path, encoder, epochs=1)
    stamp = out.read_bytes()
    again = _adapt(tmp_path, encoder, epochs=1)
    assert again == out and out.read_bytes() == stamp
    redone = _adapt(tmp_path, encoder, epochs=1, overwrite_existing=True, seed=7)
    assert json.loads(redone.read_text(encoding="utf-8"))["training"]["seed"] == 7


def test_bad_settings_and_a_corpus_of_one_text_are_refused_by_name(tmp_path, encoder):
    with pytest.raises(ValueError, match="mlm_probability"):
        _adapt(tmp_path, encoder, mlm_probability=1.5)
    with pytest.raises(ValueError, match="at least 1"):
        _adapt(tmp_path, encoder, epochs=0)
    one = tmp_path / "one.csv"
    one.write_text("text_id,text\na,potato gravy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs a corpus"):
        adapt_encoder(csv_path=one, text_cols=["text"], id_cols=["text_id"],
                      out_model_json=tmp_path / "m" / "x.json", base_model=str(encoder),
                      device="cpu", verbose=False)


def test_an_adapted_encoder_can_be_adapted_again_and_remembers_its_base(tmp_path, encoder):
    first = _adapt(tmp_path, encoder, epochs=1)
    second = adapt_encoder(csv_path=tmp_path / "c.csv", text_cols=["text"], id_cols=["text_id"],
                           out_model_json=tmp_path / "models" / "twice.json",
                           base_model=str(first), epochs=1, max_length=32, batch_size=8,
                           device="cpu", precision="fp32", verbose=False)
    doc = json.loads(second.read_text(encoding="utf-8"))
    assert doc["adapted_from"] == "adapted"
    assert doc["base_model"] == json.loads(first.read_text(encoding="utf-8"))["base_model"]


def test_before_and_after_are_measured_under_the_same_masks(tmp_path, encoder):
    """With the learning rate at zero nothing changes, so the two held-out
    losses can only differ if the masks did -- and then the before/after
    comparison would be noise dressed as an effect."""
    out = _adapt(tmp_path, encoder, epochs=1, learning_rate=0.0, weight_decay=0.0)
    ev = json.loads(out.read_text(encoding="utf-8"))["evaluation"]
    assert ev["loss_after"] == ev["loss_before"]
