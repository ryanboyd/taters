"""
Importing a classifier from the Hugging Face hub and scoring with it.

Somebody who already has a sentiment model in their Hugging Face cache from
other work should be able to point Taters at it and score a dataset, the same
way they would with a model Taters trained. The checkpoints here are tiny
untrained BERTs with a classification (or regression) head, so everything
runs offline in seconds; what is under test is the reading of the config, the
labels, the manifest, the library gate and the columns -- not the meaning of
any prediction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from csvhelpers import _read  # noqa: E402

from taters.helpers import library as lib  # noqa: E402
from taters.helpers.model_spec import describe, edit_model  # noqa: E402
from taters.text import hf_classifier as hf  # noqa: E402
from taters.text.hf_classifier import (apply_hf_classifier,  # noqa: E402
                                       cached_hub_classifiers, checkpoint_problem,
                                       import_hf_classifier, inspect_checkpoint)
from tiny_encoder import build, build_classifier  # noqa: E402

FOOD = "potato gravy butter salt dinner supper kitchen recipe".split()
WORK = "meeting deadline email boss office budget".split()


@pytest.fixture(scope="module")
def classifier(tmp_path_factory) -> Path:
    return build_classifier(tmp_path_factory.mktemp("clf") / "tiny_clf",
                            labels=("food", "work"))


@pytest.fixture(scope="module")
def regressor(tmp_path_factory) -> Path:
    return build_classifier(tmp_path_factory.mktemp("reg") / "tiny_reg", regression=True)


def _fake_cache(root: Path, hub_id: str, cfg: dict) -> None:
    """One checkpoint's config where the Hugging Face cache would keep it."""
    folder = root / f"models--{hub_id.replace('/', '--')}" / "snapshots" / "abc123"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _texts_csv(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text"])
        w.writerow(["short", " ".join(FOOD[:5])])
        w.writerow(["long", " ".join(FOOD * 12)])            # several windows at 16
        w.writerow(["work", " ".join(WORK)])
    return path


# ---------------------------------------------------------------------------
# Reading a checkpoint
# ---------------------------------------------------------------------------

def test_a_classifier_checkpoint_is_read_with_its_labels_in_head_order(classifier, regressor):
    info = inspect_checkpoint(classifier)
    assert (info.kind, info.task, info.modality) == ("folder", "classification", "text")
    assert info.labels == ("food", "work"), "id2label, in the order of the head's outputs"
    assert info.short_name == "tiny_clf"
    assert checkpoint_problem(info) == ""

    reg = inspect_checkpoint(regressor)
    assert reg.task == "regression" and reg.labels == ()
    assert checkpoint_problem(reg) == ""


def test_a_bare_encoder_is_refused_as_something_to_train_not_score(tmp_path):
    """The tiny masked-LM encoder has no head. Importing it as a classifier
    would score every text with random numbers; the refusal points at the
    place it belongs."""
    info = inspect_checkpoint(build(tmp_path / "enc"))
    assert info.task == "other"
    problem = checkpoint_problem(info)
    assert "Train a model" in problem and "no classification or regression head" in problem


def test_an_audio_classifier_is_refused_by_its_modality_not_its_head(tmp_path):
    """
    wav2vec2 has a ForSequenceClassification head too, so the head alone
    would let an audio model through as a text classifier. The backbone
    decides, and the refusal says audio models are coming rather than
    pretending the file is broken.
    """
    folder = tmp_path / "aud"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "model_type": "wav2vec2",
        "architectures": ["Wav2Vec2ForSequenceClassification"],
        "num_labels": 3, "id2label": {"0": "angry", "1": "calm", "2": "happy"},
    }), encoding="utf-8")
    info = inspect_checkpoint(folder)
    assert info.modality == "audio" and info.task == "classification"
    problem = checkpoint_problem(info)
    assert "audio model" in problem and "on the way" in problem
    with pytest.raises(ValueError, match="audio"):
        import_hf_classifier(source=folder, out_dir=tmp_path / "out")


def test_a_multi_label_head_is_read_with_its_labels_and_accepted(tmp_path):
    """An emotion model that can tag a text as both joy and surprise: the
    labels are read in head order, and it imports like any classifier."""
    folder = tmp_path / "multi"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({
        "model_type": "roberta", "architectures": ["RobertaForSequenceClassification"],
        "num_labels": 4, "problem_type": "multi_label_classification",
        "id2label": {str(i): f"e{i}" for i in range(4)},
    }), encoding="utf-8")
    info = inspect_checkpoint(folder)
    assert info.task == "multi_label" and info.labels == ("e0", "e1", "e2", "e3")
    assert checkpoint_problem(info) == ""


def test_the_cache_lists_text_classifiers_and_the_encoder_picker_leaves_them_out(tmp_path, monkeypatch):
    """
    The same cache feeds two pickers. A finished classifier belongs on the
    import screen with its labels, not on the encoder picker as a base to
    build on; a bare encoder belongs on the encoder picker and not here.
    """
    from taters.text._transformer_common import cached_hub_encoders

    cache = tmp_path / "hub"
    _fake_cache(cache, "org/sentiment", {
        "model_type": "roberta", "architectures": ["RobertaForSequenceClassification"],
        "num_labels": 3, "id2label": {"0": "negative", "1": "neutral", "2": "positive"},
        "num_hidden_layers": 12, "hidden_size": 768})
    _fake_cache(cache, "org/base", {
        "model_type": "roberta", "architectures": ["RobertaModel"],
        "num_hidden_layers": 12, "hidden_size": 768})
    _fake_cache(cache, "org/speech", {
        "model_type": "wav2vec2", "architectures": ["Wav2Vec2ForSequenceClassification"],
        "num_labels": 2, "id2label": {"0": "a", "1": "b"}})
    _fake_cache(cache, "org/rating", {
        "model_type": "bert", "architectures": ["BertForSequenceClassification"],
        "num_labels": 1, "problem_type": "regression"})
    _fake_cache(cache, "org/emotions", {
        "model_type": "roberta", "architectures": ["RobertaForSequenceClassification"],
        "num_labels": 3, "problem_type": "multi_label_classification",
        "id2label": {"0": "joy", "1": "anger", "2": "surprise"}})

    listed = dict(cached_hub_classifiers(cache))
    assert set(listed) == {"org/sentiment", "org/rating", "org/emotions"}
    # the kind sits beside every name, as on every other list of models
    assert listed["org/sentiment"].startswith("text classifier · 3 classes: negative, neutral, positive")
    assert listed["org/rating"].startswith("regression head")
    assert listed["org/emotions"].startswith("multi-label classifier · 3 labels")

    encoders = dict(cached_hub_encoders(cache))
    assert "org/base" in encoders
    assert "org/sentiment" not in encoders, "a classifier is not a base encoder"


# ---------------------------------------------------------------------------
# Importing
# ---------------------------------------------------------------------------

def test_importing_a_folder_copies_it_beside_the_manifest_and_the_library_takes_both(
        classifier, tmp_path):
    manifest = import_hf_classifier(source=classifier, out_dir=tmp_path / "out",
                                    name="food clf", outcome="topic")
    assert manifest.name == "food-clf.json"
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert doc["kind"] == hf.HF_CLASSIFIER_KIND and doc["modality"] == "text"
    assert doc["payload"] == ["food-clf.hfmodel"]
    assert (manifest.parent / "food-clf.hfmodel" / "config.json").is_file()
    assert doc["outcomes"] == {"topic": {"task": "classification", "classes": ["food", "work"]}}

    info = describe(manifest)
    assert (info.type_id, info.needs, info.modality) == ("hf_classifier", "text", "text")
    assert info.columns == ("pred_topic", "prob_topic", "p_topic_food", "p_topic_work")
    assert info.classes == {"topic": ("food", "work")}

    # and the library's own gate accepts it, weights and all
    landed = lib.import_into(lib.KINDS["models"], manifest)
    assert (landed.parent / "food-clf.hfmodel" / "config.json").is_file()
    assert lib._model_problem(landed) == ""


def test_a_manifest_whose_checkpoint_is_gutted_is_refused_by_the_gate(classifier, tmp_path):
    """A payload folder that exists but holds no config.json is a half-copied
    checkpoint; the gate names the missing file rather than letting the
    scorer discover it an hour into a run."""
    manifest = import_hf_classifier(source=classifier, out_dir=tmp_path / "out", name="clf")
    (manifest.parent / "clf.hfmodel" / "config.json").unlink()
    problem = lib._model_problem(manifest)
    assert "config.json" in problem, problem


def test_a_hub_name_is_recorded_as_the_source_without_a_payload(tmp_path, monkeypatch):
    """A model in the cache is not copied -- the cache is where it lives, and
    the model cache setting says where that is. The manifest names it, and
    loading reads the name."""
    cache = tmp_path / "hub"
    _fake_cache(cache, "org/sentiment", {
        "model_type": "roberta", "architectures": ["RobertaForSequenceClassification"],
        "num_labels": 3, "id2label": {"0": "negative", "1": "neutral", "2": "positive"},
        "max_position_embeddings": 514})
    monkeypatch.setattr("taters.helpers.settings.model_cache_dir", lambda: cache)

    manifest = import_hf_classifier(source="org/sentiment", out_dir=tmp_path / "out",
                                    outcome="sentiment")
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.name == "sentiment.json"
    assert doc["source"]["hub_id"] == "org/sentiment" and "payload" not in doc
    assert doc["apply"]["max_length"] == 512, "capped at what published classifiers train at"
    assert hf._load_model(manifest)["source"] == "org/sentiment"
    assert describe(manifest).columns[0] == "pred_sentiment"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_scoring_writes_labels_probabilities_and_windows(classifier, tmp_path):
    manifest = import_hf_classifier(source=classifier, out_dir=tmp_path / "m",
                                    name="clf", outcome="topic")
    rows = _read(apply_hf_classifier(model_json=manifest, csv_path=_texts_csv(tmp_path / "t.csv"),
                                     text_cols=["text"], id_cols=["pid"],
                                     out_features_csv=tmp_path / "scored.csv",
                                     device="cpu", verbose=False, max_length=16))
    assert list(rows[0]) == ["text_id", "token_count", "n_windows", "pred_topic",
                             "prob_topic", "p_topic_food", "p_topic_work"]
    by = {r["text_id"]: r for r in rows}
    assert by["short"]["n_windows"] == "1" and int(by["long"]["n_windows"]) > 1
    for key in ("short", "long", "work"):
        assert by[key]["pred_topic"] in ("food", "work")
        probs = float(by[key]["p_topic_food"]) + float(by[key]["p_topic_work"])
        assert abs(probs - 1.0) < 1e-3, "the mean of softmaxes is still a distribution"
        assert float(by[key]["prob_topic"]) == max(float(by[key]["p_topic_food"]),
                                                    float(by[key]["p_topic_work"]))


def test_relabeled_classes_reach_every_column_and_probabilities_can_be_left_out(
        classifier, tmp_path):
    manifest = import_hf_classifier(source=classifier, out_dir=tmp_path / "m",
                                    name="clf", outcome="topic")
    edit_model(manifest, class_names={"topic": {"food": "Food talk"}},
               apply={"emit_probabilities": False})
    rows = _read(apply_hf_classifier(model_json=manifest, csv_path=_texts_csv(tmp_path / "t.csv"),
                                     text_cols=["text"], id_cols=["pid"],
                                     out_features_csv=tmp_path / "scored.csv",
                                     device="cpu", verbose=False))
    assert list(rows[0]) == ["text_id", "token_count", "n_windows", "pred_topic"]
    assert {r["pred_topic"] for r in rows} <= {"Food talk", "work"}
    assert describe(manifest).columns == ("pred_topic", "prob_topic", "p_topic_Food talk",
                                          "p_topic_work")


def test_a_regression_head_writes_one_number_per_text(regressor, tmp_path):
    manifest = import_hf_classifier(source=regressor, out_dir=tmp_path / "m",
                                    name="rating", outcome="rating")
    assert describe(manifest).columns == ("pred_rating",)
    rows = _read(apply_hf_classifier(model_json=manifest, csv_path=_texts_csv(tmp_path / "t.csv"),
                                     text_cols=["text"], id_cols=["pid"],
                                     out_features_csv=tmp_path / "scored.csv",
                                     device="cpu", verbose=False))
    assert list(rows[0]) == ["text_id", "token_count", "n_windows", "pred_rating"]
    for r in rows:
        float(r["pred_rating"])


def test_a_multi_label_head_writes_a_probability_per_label_and_the_labels_above_threshold(
        tmp_path):
    """
    Several labels can be true at once, so there is no winning class and no
    prob_ column. Each label gets two columns: `pred_<outcome>_<label>`, 1 or
    0 against the model's threshold, and `p_<outcome>_<label>`, its sigmoid
    probability. `pred_<outcome>` joins the labels that cleared the threshold
    with | -- readable, but a string nobody can group or model on, which is
    why the 1/0 columns exist beside it.

    The threshold is the model's own apply setting, so raising it in
    Settings changes what is flagged without touching the probabilities.
    """
    from tiny_encoder import build_classifier as _build

    folder = _build(tmp_path / "emo", labels=("joy", "anger", "surprise"), multi_label=True)
    manifest = import_hf_classifier(source=folder, out_dir=tmp_path / "m",
                                    name="emotions", outcome="emotion")
    info = describe(manifest)
    assert info.columns == ("pred_emotion",
                            "pred_emotion_joy", "pred_emotion_anger",
                            "pred_emotion_surprise",
                            "p_emotion_joy", "p_emotion_anger",
                            "p_emotion_surprise")
    assert info.apply["threshold"] == 0.5

    rows = _read(apply_hf_classifier(model_json=manifest, csv_path=_texts_csv(tmp_path / "t.csv"),
                                     text_cols=["text"], id_cols=["pid"],
                                     out_features_csv=tmp_path / "scored.csv",
                                     device="cpu", verbose=False))
    labels = ("joy", "anger", "surprise")
    assert list(rows[0]) == ["text_id", "token_count", "n_windows", "pred_emotion",
                             "pred_emotion_joy", "pred_emotion_anger",
                             "pred_emotion_surprise",
                             "p_emotion_joy", "p_emotion_anger", "p_emotion_surprise"]
    for r in rows:
        probs = {lab: float(r[f"p_emotion_{lab}"]) for lab in labels}
        assert all(0.0 <= v <= 1.0 for v in probs.values())
        expected = [lab for lab in labels if probs[lab] >= 0.5]
        assert r["pred_emotion"] == "|".join(expected), (r["pred_emotion"], probs)
        # the flag column and the joined string have to tell the same story
        for lab in labels:
            flag = r[f"pred_emotion_{lab}"]
            assert flag in ("0", "1"), flag
            assert (flag == "1") == (probs[lab] >= 0.5), (lab, flag, probs[lab])

    # an untrained head sits near 0.5 everywhere; a threshold nothing can reach
    # empties pred_ and leaves the probabilities exactly as they were
    edit_model(manifest, apply={"threshold": 0.99})
    strict = _read(apply_hf_classifier(model_json=manifest, csv_path=tmp_path / "t.csv",
                                       text_cols=["text"], id_cols=["pid"],
                                       out_features_csv=tmp_path / "strict.csv",
                                       device="cpu", verbose=False))
    assert all(r["pred_emotion"] == "" for r in strict)
    assert [r["p_emotion_joy"] for r in strict] == [r["p_emotion_joy"] for r in rows]
    with pytest.raises(ValueError, match="between 0 and 1"):
        edit_model(manifest, apply={"threshold": 1.5})


def test_score_with_model_dispatches_to_the_classifier_through_the_registry(classifier, tmp_path):
    """The scoring entry point looks the kind up rather than guessing, so an
    imported classifier scores through 'Score with models I already have'
    with no special case anywhere."""
    from taters.score_model import score_with_model

    manifest = import_hf_classifier(source=classifier, out_dir=tmp_path / "m",
                                    name="clf", outcome="topic")
    out = score_with_model(model_json=manifest, csv_path=_texts_csv(tmp_path / "t.csv"),
                           text_cols=["text"], id_cols=["pid"],
                           out_csv=tmp_path / "scores.csv", device="cpu", verbose=False)
    rows = _read(out)
    assert "pred_topic" in rows[0] and "p_topic_food" in rows[0]
    assert len(rows) == 3


def test_renaming_a_multi_label_heads_labels_reaches_both_of_its_columns(tmp_path):
    """
    A hub checkpoint that was never given label names arrives as LABEL_0,
    LABEL_1, ... -- which is most of them, and useless in a results table.
    Renaming is per class, in Settings, and both columns a label owns have to
    follow it: the 1/0 flag and the probability. One following and not the
    other would put `pred_x_anxiety` next to `p_x_LABEL_1` and leave the
    reader to guess they are the same thing.
    """
    from tiny_encoder import build_classifier as _build

    folder = _build(tmp_path / "raw",
                    labels=("LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"),
                    multi_label=True)
    manifest = import_hf_classifier(source=folder, out_dir=tmp_path / "m",
                                    name="constructs", outcome="risk")
    named = {"LABEL_0": "anxiety", "LABEL_1": "anger",
             "LABEL_2": "sadness", "LABEL_3": "hope"}
    edit_model(manifest, class_names={"risk": named})

    assert describe(manifest).columns == (
        "pred_risk",
        "pred_risk_anxiety", "pred_risk_anger", "pred_risk_sadness", "pred_risk_hope",
        "p_risk_anxiety", "p_risk_anger", "p_risk_sadness", "p_risk_hope")

    rows = _read(apply_hf_classifier(
        model_json=manifest, csv_path=_texts_csv(tmp_path / "t.csv"),
        text_cols=["text"], id_cols=["pid"],
        out_features_csv=tmp_path / "scored.csv", device="cpu", verbose=False))
    assert list(rows[0]) == ["text_id", "token_count", "n_windows", "pred_risk",
                             "pred_risk_anxiety", "pred_risk_anger",
                             "pred_risk_sadness", "pred_risk_hope",
                             "p_risk_anxiety", "p_risk_anger",
                             "p_risk_sadness", "p_risk_hope"]
    assert not any("LABEL_" in c for c in rows[0]), "a raw head name survived"
    for r in rows:
        for new in named.values():
            assert (r[f"pred_risk_{new}"] == "1") == (float(r[f"p_risk_{new}"]) >= 0.5)
        flagged = [n for n in named.values() if r[f"pred_risk_{n}"] == "1"]
        assert sorted(r["pred_risk"].split("|") if r["pred_risk"] else []) == sorted(flagged)


def test_a_batch_size_or_window_length_below_one_is_refused_where_it_is_typed(tmp_path):
    """
    `kind="int"` only promises a number, so 0 and -4 used to be stored and
    then fail deep inside scoring: a batch size of zero reaches
    `range(0, n, 0)` and raises "range() arg 3 must not be zero", which names
    nothing the user did. The refusal belongs at the screen they typed it on.
    """
    from tiny_encoder import build_classifier as _build

    manifest = import_hf_classifier(source=_build(tmp_path / "c", labels=("a", "b")),
                                    out_dir=tmp_path / "m", name="c", outcome="o")
    for field in ("batch_size", "max_length"):
        for bad in (0, -4):
            with pytest.raises(ValueError, match="at least 1"):
                edit_model(manifest, apply={field: bad})
    # and the good values still land
    assert edit_model(manifest, apply={"batch_size": 8}).apply["batch_size"] == 8


def test_a_window_longer_than_the_model_accepts_is_capped_and_said_out_loud(tmp_path):
    """
    RoBERTa advertises `max_position_embeddings` of 514 and accepts 512 --
    its positions start at an offset -- so a max_length nobody sanity-checked
    goes out of range mid-run. The tokenizer knows the real ceiling, so it
    decides; and because scoring at a length other than the one asked for
    would look like the setting was ignored, it is announced.
    """
    from taters.text._transformer_common import window_length

    class Tok:
        def __init__(self, limit):
            self.model_max_length = limit

    assert window_length(Tok(512), 256) == (256, "")
    length, note = window_length(Tok(512), 514)
    assert length == 512
    assert "512" in note and "514" in note
    # a tokenizer that does not know writes a sentinel in the quintillions;
    # that is "no opinion", not permission to cap at it
    assert window_length(Tok(10 ** 30), 4096) == (4096, "")
    assert window_length(Tok(0), 4096) == (4096, "")
