"""
Tests for `taters.text.transformer_embeddings` and the shared
`_transformer_common` helpers, on a tiny randomly initialized BERT written
into tmp -- no download, no GPU. What is checked is the plumbing a paper has
to be able to describe: which layer was read, how tokens were pooled, that
padding never leaks into a mean, that a document's vector is the mean of its
sentences, that a long sentence is windowed rather than cut, and that
batching across rows changes nothing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from taters.text import _transformer_common as tc  # noqa: E402
from taters.text.transformer_embeddings import extract_transformer_embeddings  # noqa: E402
from tiny_encoder import build  # noqa: E402


@pytest.fixture(scope="module")
def encoder(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("enc") / "tiny")


@pytest.fixture(scope="module")
def loaded(encoder):
    model, tokenizer, resolved, device, _reason = tc.load_encoder(encoder, device="cpu")
    return model, tokenizer


def _encode(loaded, sentences, **kw):
    model, tokenizer = loaded
    kw.setdefault("device_name", "cpu")
    kw.setdefault("precision", "fp32")
    return tc.encode_sentences(model, tokenizer, sentences, **kw)


def _corpus(path: Path, texts) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i, t in enumerate(texts):
            w.writerow([f"t{i}", t])
    return path


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# layers and pooling
# ---------------------------------------------------------------------------

def test_layer_choices_map_onto_hidden_states_and_zero_is_refused():
    """hidden_states has n+1 entries; index 0 is the embedding lookup."""
    assert tc.parse_layers("second_to_last", 6) == ([5], "mean")
    assert tc.parse_layers("last", 6) == ([6], "mean")
    assert tc.parse_layers("last4_mean", 6) == ([3, 4, 5, 6], "mean")
    assert tc.parse_layers("last4_concat", 6) == ([3, 4, 5, 6], "concat")
    assert tc.parse_layers("-1,-2", 6) == ([5, 6], "mean")
    assert tc.parse_layers("2, 3", 6) == ([2, 3], "mean")
    assert tc.parse_layers("second_to_last", 1) == ([1], "mean"), "one layer: use it"
    with pytest.raises(ValueError, match="layer 0 is the embedding lookup"):
        tc.parse_layers("0", 6)
    with pytest.raises(ValueError, match="layers 1..6"):
        tc.parse_layers("7", 6)
    with pytest.raises(ValueError, match="one of second_to_last"):
        tc.parse_layers("middle", 6)


def test_the_layer_read_changes_the_vector_and_minus_two_is_second_to_last(loaded):
    a, _ = _encode(loaded, ["the cat sat on the mat ."], layers="second_to_last")
    b, _ = _encode(loaded, ["the cat sat on the mat ."], layers="last")
    c, _ = _encode(loaded, ["the cat sat on the mat ."], layers="-2")
    assert not np.allclose(a, b)
    assert np.allclose(a, c)
    wide, _ = _encode(loaded, ["the cat sat ."], layers="last4_concat")
    assert wide.shape[1] == 2 * 32, "two layers exist, so concat is twice the width"


def test_padding_never_leaks_into_the_mean():
    """The same sentence alone and beside a much longer one must pool to the
    same vector: padding tokens are masked out of the average."""
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    mean = tc.pool_hidden(hidden, mask, "mean")
    assert torch.allclose(mean[0], hidden[0, :2].mean(dim=0))
    assert torch.allclose(mean[1], hidden[1].mean(dim=0))
    assert torch.allclose(tc.pool_hidden(hidden, mask, "cls")[0], hidden[0, 0])
    mx = tc.pool_hidden(hidden, mask, "max")
    assert torch.allclose(mx[0], hidden[0, :2].max(dim=0).values)
    with pytest.raises(ValueError, match="pooling must be one of"):
        tc.pool_hidden(hidden, mask, "sum")


def test_a_sentence_pools_the_same_alone_and_in_a_padded_batch(loaded):
    short = "the cat sat ."
    alone, _ = _encode(loaded, [short])
    together, _ = _encode(loaded, [short, "the dog ran on the rug and the bird flew in the sky today ."])
    assert np.allclose(alone[0], together[0], atol=1e-5)


# ---------------------------------------------------------------------------
# windows and batches
# ---------------------------------------------------------------------------

def test_a_sentence_longer_than_the_window_is_windowed_and_averaged(loaded):
    """Never cut: the vector is the mean over overlapping windows, and the
    token count is the sentence's whole length."""
    long = " ".join(["the cat sat on the mat and the dog ran"] * 6)
    vecs, counts = _encode(loaded, [long], max_length=16, stride=4)
    assert counts[0] > 16
    ids = tc._windows(list(range(counts[0])), 16, 4)
    assert len(ids) > 1 and all(len(w) <= 14 for w in ids)
    assert ids[0][:14] == list(range(14)) and ids[1][0] == 10, "overlap of four"
    assert ids[-1][-1] == counts[0] - 1, "the tail is kept"
    # now let's encode the windows one at a time and take the mean ourselves
    model, tokenizer = loaded
    parts = []
    all_ids = tokenizer([long], add_special_tokens=False)["input_ids"][0]
    for w in tc._windows(all_ids, 16, 4):
        text = tokenizer.decode(w)
        v, _ = _encode(loaded, [text], max_length=16, stride=4)
        parts.append(v[0])
    assert np.allclose(vecs[0], np.mean(parts, axis=0), atol=1e-4)


def test_batching_across_rows_equals_row_by_row(loaded):
    sentences = ["the cat sat .", "a dog ran on the rug !", "the sea was good today .",
                 "the boss sent an email about the budget .", "potato gravy butter salt ."]
    batched, _ = _encode(loaded, sentences, batch_size=2)
    single = np.vstack([_encode(loaded, [s])[0] for s in sentences])
    assert np.allclose(batched, single, atol=1e-5)


def test_the_oom_fallback_halves_the_batch_and_gives_up_with_advice(capsys):
    seen = []

    def fn(size):
        seen.append(size)
        if size > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return size

    assert tc.run_with_oom_fallback(fn, 8, verbose=True) == 2
    assert seen == [8, 4, 2]
    assert "retrying" in capsys.readouterr().out

    def never(size):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="even one item at a time"):
        tc.run_with_oom_fallback(never, 2)

    def other(size):
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        tc.run_with_oom_fallback(other, 2)


def test_precision_is_half_only_on_cuda():
    import contextlib

    assert isinstance(tc.autocast_for("cpu", "auto"), contextlib.nullcontext)
    assert isinstance(tc.autocast_for("cpu", "fp16"), contextlib.nullcontext)
    with pytest.raises(ValueError, match="precision must be one of"):
        tc.autocast_for("cpu", "bf16")


# ---------------------------------------------------------------------------
# finding an encoder
# ---------------------------------------------------------------------------

def test_an_encoder_is_found_by_name_folder_or_taters_manifest(tmp_path, encoder):
    hub = tc.resolve_encoder("distilroberta-base")
    assert (hub.kind, hub.source, hub.base_model) == ("hub", "distilroberta-base", "distilroberta-base")
    folder = tc.resolve_encoder(encoder)
    assert folder.kind == "folder" and folder.source == str(encoder)
    with pytest.raises(ValueError, match="no config.json"):
        tc.resolve_encoder(tmp_path)
    with pytest.raises(FileNotFoundError):
        tc.resolve_encoder(tmp_path / "nowhere.json")
    with pytest.raises(ValueError, match="no encoder was named"):
        tc.resolve_encoder("  ")
    import json
    import shutil

    payload = tmp_path / "clinical.encoder"
    shutil.copytree(encoder, payload)
    manifest = tmp_path / "clinical.json"
    manifest.write_text(json.dumps({"kind": tc.ENCODER_KIND, "format": 1,
                                    "name": "clinical", "payload": ["clinical.encoder"],
                                    "base_model": "distilroberta-base"}), encoding="utf-8")
    enc = tc.resolve_encoder(manifest)
    assert (enc.kind, enc.source, enc.label, enc.base_model) == \
        ("encoder", str(payload), "clinical", "distilroberta-base")
    (tmp_path / "other.json").write_text(json.dumps({"kind": "taters-mem-model"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a text encoder"):
        tc.resolve_encoder(tmp_path / "other.json")


def test_freezing_leaves_the_lower_layers_out_of_training(loaded):
    model, _ = loaded
    trainable, frozen = tc.freeze_below(model, 1)
    lower = list(model.base_model.encoder.layer)[0]
    upper = list(model.base_model.encoder.layer)[-1]
    assert not any(p.requires_grad for p in lower.parameters())
    assert all(p.requires_grad for p in upper.parameters())
    assert frozen > 0 and trainable > 0
    all_on, none_off = tc.freeze_below(model, 0)
    assert none_off == 0 or all(p.requires_grad for p in upper.parameters())


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

def test_a_documents_vector_is_the_mean_of_its_sentences(tmp_path, encoder, loaded):
    """The sentence-transformers contract: each sentence encoded, the text
    their plain mean (equal) or their token-weighted mean (tokens)."""
    two = "The cat sat on the mat. The dog ran on the rug and the bird flew in the sky today!"
    out = extract_transformer_embeddings(
        csv_path=_corpus(tmp_path / "c.csv", [two, "Potato gravy."]), text_cols=["text"],
        id_cols=["text_id"], model_name_or_path=str(encoder), device="cpu",
        precision="fp32", out_features_csv=tmp_path / "e.csv", verbose=False, rounding=6)
    rows = _rows(out)
    assert list(rows[0])[:4] == ["text_id", "token_count", "sentence_count", "e_1"]
    assert rows[0]["sentence_count"] == "2" and rows[1]["sentence_count"] == "1"
    sents, counts = _encode(loaded, ["The cat sat on the mat.",
                                     "The dog ran on the rug and the bird flew in the sky today!"])
    assert rows[0]["token_count"] == str(sum(counts))
    got = np.array([float(rows[0][f"e_{i + 1}"]) for i in range(32)])
    assert np.allclose(got, sents.mean(axis=0), atol=1e-5)
    weighted = extract_transformer_embeddings(
        csv_path=tmp_path / "c.csv", text_cols=["text"], id_cols=["text_id"],
        model_name_or_path=str(encoder), device="cpu", precision="fp32",
        sentence_weighting="tokens", out_features_csv=tmp_path / "w.csv",
        verbose=False, rounding=6)
    w = np.asarray(counts, dtype=float)
    expect = (sents * w[:, None]).sum(axis=0) / w.sum()
    got_w = np.array([float(_rows(weighted)[0][f"e_{i + 1}"]) for i in range(32)])
    assert np.allclose(got_w, expect, atol=1e-5)
    assert not np.allclose(got_w, got)


def test_the_step_says_which_device_it_is_encoding_on(tmp_path, encoder, monkeypatch):
    """
    The run display shows the last message it was given, and the per-row ticks
    carry none -- so this is the label somebody watches for the whole encode.
    It used to be a print behind `verbose`, which the app never sets: an
    hour-long run on the CPU was indistinguishable from a fast one on the card.
    """
    from taters.helpers import gpu

    seen = []

    def on_progress(done, total, message=None, **kw):
        if message:
            seen.append(message)

    extract_transformer_embeddings(
        csv_path=_corpus(tmp_path / "c.csv", ["Potato gravy."]), text_cols=["text"],
        id_cols=["text_id"], model_name_or_path=str(encoder), device="cpu",
        precision="fp32", out_features_csv=tmp_path / "e.csv", verbose=False,
        on_progress=on_progress)
    assert "encoding on the CPU" in seen
    assert seen[-1] == "encoding on the CPU", "the ticks leave it up, so it is what stays"

    # and when `auto` wanted the GPU but could not have it, the note says why
    monkeypatch.setattr(gpu, "_cuda_visible", lambda backend: (False, "no driver"))
    seen.clear()
    extract_transformer_embeddings(
        csv_path=tmp_path / "c.csv", text_cols=["text"], id_cols=["text_id"],
        model_name_or_path=str(encoder), device="auto", precision="fp32",
        out_features_csv=tmp_path / "e2.csv", verbose=False, on_progress=on_progress)
    fell_back = [m for m in seen if m.startswith("encoding on the CPU")]
    assert fell_back and "no driver" in fell_back[0]


def test_batching_across_rows_in_the_step_equals_one_row_at_a_time(tmp_path, encoder, monkeypatch):
    texts = [f"The cat sat on the mat {i}. The dog ran." for i in range(7)] + ["Salt."]
    a = _rows(extract_transformer_embeddings(
        csv_path=_corpus(tmp_path / "c.csv", texts), text_cols=["text"], id_cols=["text_id"],
        model_name_or_path=str(encoder), device="cpu", precision="fp32",
        out_features_csv=tmp_path / "a.csv", verbose=False))
    from taters.text import transformer_embeddings as te

    monkeypatch.setattr(te, "_SENTENCES_PER_CALL", 1)
    b = _rows(extract_transformer_embeddings(
        csv_path=tmp_path / "c.csv", text_cols=["text"], id_cols=["text_id"],
        model_name_or_path=str(encoder), device="cpu", precision="fp32",
        out_features_csv=tmp_path / "b.csv", verbose=False))
    assert a == b


def test_an_empty_text_gets_blanks_and_a_header_is_still_written(tmp_path, encoder):
    out = extract_transformer_embeddings(
        csv_path=_corpus(tmp_path / "c.csv", ["...", "The cat sat."]), text_cols=["text"],
        id_cols=["text_id"], model_name_or_path=str(encoder), device="cpu",
        precision="fp32", out_features_csv=tmp_path / "e.csv", verbose=False)
    rows = _rows(out)
    assert rows[0]["sentence_count"] == "1", "'...' is one sentence to the splitter"
    empty = _corpus(tmp_path / "d.csv", ["   "])
    out = extract_transformer_embeddings(
        csv_path=empty, text_cols=["text"], id_cols=["text_id"],
        model_name_or_path=str(encoder), device="cpu", precision="fp32",
        out_features_csv=tmp_path / "f.csv", verbose=False)
    with Path(out).open(encoding="utf-8-sig") as fh:
        header = fh.readline().strip().split(",")
    assert header[:3] == ["text_id", "token_count", "sentence_count"] and len(header) == 3 + 32
    # if we hand over an analysis-ready table directly (so no gather to drop the
    # row), a text with no sentences gets blank cells. never a zero vector: that
    # would sit right in the middle of every comparison looking like a real
    # measurement
    ready = tmp_path / "ready.csv"
    ready.write_text("text_id,text\nx,\ny,The cat sat.\n", encoding="utf-8")
    rows = _rows(extract_transformer_embeddings(
        analysis_csv=ready, model_name_or_path=str(encoder), device="cpu",
        precision="fp32", out_features_csv=tmp_path / "g.csv", verbose=False))
    assert rows[0]["sentence_count"] == "0" and rows[0]["e_1"] == "" and rows[0]["e_32"] == ""
    assert rows[1]["e_1"] != ""


def test_the_record_carries_the_layer_pooling_and_encoder(tmp_path, encoder):
    from taters.helpers.provenance import read as read_record

    out = extract_transformer_embeddings(
        csv_path=_corpus(tmp_path / "c.csv", ["The cat sat."]), text_cols=["text"],
        id_cols=["text_id"], model_name_or_path=str(encoder), device="cpu",
        precision="fp32", layers="last", pooling="cls",
        out_features_csv=tmp_path / "e.csv", verbose=False)
    rec = read_record(out)
    settings = rec["settings"] if "settings" in rec else rec
    flat = str(settings)
    assert "'layers': 'last'" in flat and "'pooling': 'cls'" in flat
    assert str(encoder) in flat


def test_normalizing_gives_unit_vectors_and_the_resume_contract_holds(tmp_path, encoder):
    out = extract_transformer_embeddings(
        csv_path=_corpus(tmp_path / "c.csv", ["The cat sat.", "Salt!"]), text_cols=["text"],
        id_cols=["text_id"], model_name_or_path=str(encoder), device="cpu",
        precision="fp32", normalize_l2=True, out_features_csv=tmp_path / "e.csv",
        verbose=False, rounding=6)
    for r in _rows(out):
        v = np.array([float(r[f"e_{i + 1}"]) for i in range(32)])
        assert abs(np.linalg.norm(v) - 1.0) < 1e-4
    stamp = Path(out).read_bytes()
    # same settings again, so nothing should be redone. (a changed setting, say a
    # different precision, is a different measurement and does get redone)
    extract_transformer_embeddings(
        csv_path=tmp_path / "c.csv", text_cols=["text"], id_cols=["text_id"],
        model_name_or_path=str(encoder), device="cpu", precision="fp32",
        normalize_l2=True, out_features_csv=tmp_path / "e.csv", verbose=False,
        rounding=6)
    assert Path(out).read_bytes() == stamp


def test_a_bad_weighting_or_missing_torch_is_refused_by_name(tmp_path, encoder, monkeypatch):
    with pytest.raises(ValueError, match="sentence_weighting must be"):
        extract_transformer_embeddings(
            csv_path=_corpus(tmp_path / "c.csv", ["x"]), text_cols=["text"],
            model_name_or_path=str(encoder), sentence_weighting="median",
            out_features_csv=tmp_path / "e.csv", verbose=False)
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "transformers" else real(name, *a, **k))
    assert "transformers" in tc.torch_missing_reason()
