"""
Tests for `taters.text.finetune_predictor` on the tiny encoder: a planted
signal is recovered out of fold, one head per outcome, rows missing an
outcome still train the others, classes are written as labels and can be
relabeled, the saved model reproduces its own predictions and honors its
apply settings, a warm start reuses heads, and the report carries the
methods, tables and plots.

Two folds on sixty short texts keep every test to a few seconds on a CPU.
The tiny encoder has random weights, so the signal is carried by the words
themselves: two vocabularies, two outcomes derived from which was used.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
np = pytest.importorskip("numpy")

from csvhelpers import _read  # noqa: E402

from taters.helpers.model_spec import describe, edit_model  # noqa: E402
from taters.text import finetune_predictor as fp  # noqa: E402
from taters.text.finetune_predictor import (apply_text_predictor,  # noqa: E402
                                            finetune_text_predictor,
                                            parse_task_weights)
from tiny_encoder import build  # noqa: E402

FOOD = "potato gravy butter salt dinner supper kitchen recipe".split()
WORK = "meeting deadline email boss office budget".split()


@pytest.fixture(scope="module")
def encoder(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("enc") / "tiny")


def _study(path: Path, n=60, seed=1, blank_every=0) -> Path:
    """Two vocabularies; `score` tracks the vocabulary, `topic` names it.
    With ``blank_every`` set, every k-th row has no score."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text", "score", "topic", "code"])
        for i in range(n):
            food = i % 2 == 0
            words = [rng.choice(FOOD if food else WORK) for _ in range(rng.randint(10, 20))]
            score = (3.0 if food else 1.0) + rng.gauss(0, 0.3)
            blank = blank_every and i % blank_every == 0
            w.writerow([f"d{i}", " ".join(words), "" if blank else f"{score:.2f}",
                        "food" if food else "work", "1" if food else "0"])
    return path


def _fit(tmp_path, encoder, **kw):
    args = dict(csv_path=_study(tmp_path / "s.csv"), text_cols=["text"], id_cols=["pid"],
                outcome_cols=["score", "topic"], out_dir=tmp_path / "stats",
                base_model=str(encoder), n_folds=2, epochs=3, max_length=32,
                batch_size=8, learning_rate=1e-3, device="cpu", precision="fp32",
                verbose=False)
    args.update(kw)
    return finetune_text_predictor(**args)


def _model(tmp_path, stem="text_predictor__score_topic") -> Path:
    return tmp_path / "stats" / "models" / f"{stem}.json"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_task_weights_are_parsed_and_checked():
    assert parse_task_weights("", ["a", "b"]) == {"a": 1.0, "b": 1.0}
    assert parse_task_weights("b: 2", ["a", "b"]) == {"a": 1.0, "b": 2.0}
    assert parse_task_weights({"a": 0.5}, ["a"]) == {"a": 0.5}
    with pytest.raises(ValueError, match="not one of the outcomes"):
        parse_task_weights("c: 1", ["a", "b"])
    with pytest.raises(ValueError, match="must be positive"):
        parse_task_weights("a: 0", ["a"])
    with pytest.raises(ValueError, match="must be a number"):
        parse_task_weights("a: lots", ["a"])


def test_outcomes_are_typed_from_their_values_unless_told_otherwise(tmp_path):
    table = [{"score": f"{i}.5", "topic": "food" if i % 2 else "work", "code": str(i % 2)}
             for i in range(20)]
    tasks = fp._outcome_tasks(table, ["score", "topic", "code"], ["code"])
    assert tasks["score"]["task"] == "regression" and tasks["score"]["std"] > 0
    assert tasks["topic"] == {"task": "classification", "classes": ["food", "work"],
                              "counts": {"food": 10, "work": 10}}
    assert tasks["code"]["classes"] == ["0", "1"], "named categorical: a code, not a number"
    with pytest.raises(ValueError, match="one class"):
        fp._outcome_tasks([{"t": "x"}] * 10, ["t"], [])
    with pytest.raises(ValueError, match="at least 5"):
        fp._outcome_tasks([{"t": "x"}] * 10 + [{"t": "y"}] * 2, ["t"], [])
    with pytest.raises(ValueError, match="is constant"):
        fp._outcome_tasks([{"s": "2"}] * 10, ["s"], [])
    with pytest.raises(ValueError, match="blank in every row"):
        fp._outcome_tasks([{"s": ""}] * 10, ["s"], [])


def test_the_validation_slice_is_a_seeded_disjoint_part_of_the_training_rows():
    train, val = fp._split_validation(list(range(30)), 0.2, seed=1)
    assert len(val) == 6 and not set(train) & set(val)
    assert sorted(train + val) == list(range(30))
    assert fp._split_validation(list(range(30)), 0.2, seed=1) == (train, val)
    assert fp._split_validation([1, 2, 3], 0.5, seed=1) == ([1, 2, 3], []), "too few to slice"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def test_a_planted_signal_is_recovered_out_of_fold_with_one_head_per_outcome(tmp_path, encoder):
    out = _fit(tmp_path, encoder)
    rows = {r["outcome"]: r for r in _read(out)}
    assert list(rows) == ["score", "topic"], "one metrics row per outcome"
    assert float(rows["score"]["cv_r2"]) > 0.3
    assert rows["score"]["task"] == "regression" and rows["score"]["accuracy"] == ""
    assert float(rows["topic"]["accuracy"]) > 0.6 and rows["topic"]["cv_r2"] == ""
    assert rows["topic"]["n_classes"] == "2" and rows["topic"]["baseline_accuracy"] == "0.5"
    assert rows["score"]["feature_set"] == "text" and rows["score"]["n_folds"] == "2"
    folds = _read(tmp_path / "stats" / "text_predictor_folds.csv")
    assert {(r["outcome"], r["fold"]) for r in folds} == {("score", "1"), ("score", "2"),
                                                          ("topic", "1"), ("topic", "2")}
    epochs = _read(tmp_path / "stats" / "text_predictor_epochs.csv")
    assert len(epochs) == 6 and sum(r["kept"] == "yes" for r in epochs) == 2
    doc = json.loads(_model(tmp_path).read_text(encoding="utf-8"))
    assert set(doc["outcomes"]) == {"score", "topic"}
    assert doc["outcomes"]["topic"]["classes"] == ["food", "work"]
    assert doc["training"]["best_epoch_per_fold"] and doc["training"]["final_epochs"] >= 1
    from safetensors.torch import load_file

    heads = load_file(str(tmp_path / "stats" / "models" / "text_predictor__score_topic.predictor"
                          / "heads.safetensors"))
    assert heads["score.weight"].shape == (1, 32) and heads["topic.weight"].shape == (2, 32)


def test_the_folds_are_the_ridges_balanced_folds(tmp_path, encoder):
    from taters.stats._fit_common import balanced_folds

    _fit(tmp_path, encoder, epochs=1)
    preds = _read(tmp_path / "stats" / "text_predictor_predictions__score_topic.csv")
    y = np.array([float(r["score"]) for r in preds])
    expected = balanced_folds(y, 2, 42)
    assert [int(r["fold_score"]) - 1 for r in preds] == list(expected)
    assert [r["fold_score"] for r in preds] == [r["fold_topic"] for r in preds], "one deal"


def test_a_row_missing_one_outcome_still_trains_the_other(tmp_path, encoder):
    out = _fit(tmp_path, encoder, csv_path=_study(tmp_path / "gaps.csv", blank_every=5))
    rows = {r["outcome"]: r for r in _read(out)}
    assert rows["score"]["n_used"] == "48" and rows["topic"]["n_used"] == "60"
    preds = _read(tmp_path / "stats" / "text_predictor_predictions__score_topic.csv")
    blank = [r for r in preds if r["score"] == ""]
    assert len(blank) == 12
    assert all(r["oof_score"] != "" and r["pred_topic"] in ("food", "work") for r in blank), \
        "a row without the outcome is still predicted for both"


def test_predictions_carry_class_labels_and_relabeling_reaches_every_column(tmp_path, encoder):
    """The complaint was "outcome 0, outcome 1". Classes are the data's own
    labels, and a relabel in the model file changes the predicted class and
    the per-class probability headings alike."""
    _fit(tmp_path, encoder, outcome_cols=["code"], categorical_outcomes=["code"])
    model = _model(tmp_path, "text_predictor__code")
    scored = _read(apply_text_predictor(model_json=model, csv_path=tmp_path / "s.csv",
                                        text_cols=["text"], id_cols=["pid"],
                                        out_features_csv=tmp_path / "a.csv",
                                        device="cpu", verbose=False))
    assert {r["pred_code"] for r in scored} <= {"0", "1"}
    assert {"p_code_0", "p_code_1", "prob_code"} <= set(scored[0])
    info = edit_model(model, class_names={"code": {"0": "work", "1": "food"}})
    assert info.classes == {"code": ("work", "food")}
    assert set(info.columns) >= {"p_code_work", "p_code_food"}
    again = _read(apply_text_predictor(model_json=model, csv_path=tmp_path / "s.csv",
                                       text_cols=["text"], id_cols=["pid"],
                                       out_features_csv=tmp_path / "b.csv",
                                       device="cpu", verbose=False))
    assert {r["pred_code"] for r in again} <= {"work", "food"}
    assert not {"p_code_0", "p_code_1"} & set(again[0])
    for a, b in zip(scored, again):
        assert b["pred_code"] == {"0": "work", "1": "food"}[a["pred_code"]]
        assert b["p_code_work"] == a["p_code_0"]


def test_apply_reproduces_the_final_models_own_predictions_and_honors_its_settings(tmp_path, encoder):
    _fit(tmp_path, encoder)
    model = _model(tmp_path)
    training = _read(tmp_path / "stats" / "text_predictor_predictions__score_topic.csv")
    scored = _read(apply_text_predictor(model_json=model, csv_path=tmp_path / "s.csv",
                                        text_cols=["text"], id_cols=["pid"],
                                        out_features_csv=tmp_path / "a.csv",
                                        device="cpu", verbose=False))
    assert list(scored[0]) == ["text_id", "token_count", "n_windows", "pred_score",
                               "pred_topic", "prob_topic", "p_topic_food", "p_topic_work"]
    for t, s in zip(training, scored):
        assert abs(float(t["pred_score"]) - float(s["pred_score"])) < 1e-3
        assert t["pred_topic"] == s["pred_topic"]
    # now the model's own apply block: once we edit it, no more probabilities.
    edit_model(model, apply={"emit_probabilities": False, "max_length": 8})
    info = describe(model)
    assert info.apply["emit_probabilities"] is False and info.apply["max_length"] == 8
    lean = _read(apply_text_predictor(model_json=model, csv_path=tmp_path / "s.csv",
                                      text_cols=["text"], id_cols=["pid"],
                                      out_features_csv=tmp_path / "c.csv",
                                      device="cpu", verbose=False))
    assert list(lean[0]) == ["text_id", "token_count", "n_windows", "pred_score", "pred_topic"]
    assert all(int(r["n_windows"]) > 1 for r in lean), "max_length 8 windows every text"
    # ...and an explicit call still wins over the file.
    full = _read(apply_text_predictor(model_json=model, csv_path=tmp_path / "s.csv",
                                      text_cols=["text"], id_cols=["pid"],
                                      out_features_csv=tmp_path / "d.csv",
                                      device="cpu", verbose=False, emit_probabilities=True,
                                      max_length=32))
    assert "prob_topic" in full[0] and all(r["n_windows"] == "1" for r in full)


# ---------------------------------------------------------------------------
# Long texts: windows, not a cut
#
# most of the user's corpora run far past 256 tokens (some are tweets). a text
# used to be one sequence cut at max_length, so an essay trained and was scored
# on its first 254 tokens. now every window trains, and a text is predicted as
# the mean over its windows.
# ---------------------------------------------------------------------------

def _tok(encoder):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(encoder))


def test_a_long_text_becomes_windows_that_together_hold_every_token(encoder):
    from taters.text.finetune_predictor import _windows_of

    tok = _tok(encoder)
    long = " ".join(FOOD * 12)                      # ~96 tokens
    windows, counts = _windows_of(tok, [long, "potato", ""], max_length=16)
    assert counts[0] > 14 and len(windows[0]) >= 2, "a long text takes several windows"
    assert all(len(w) <= 16 for w in windows[0])
    cls, sep = tok.cls_token_id, tok.sep_token_id
    assert all(w[0] == cls and w[-1] == sep for w in windows[0])
    bodies = [t for w in windows[0] for t in w[1:-1]]
    assert bodies == tok(long, add_special_tokens=False)["input_ids"], \
        "the windows' bodies are the whole text, in order, nothing dropped"
    assert len(windows[1]) == 1, "a short text is one window, as before"
    assert windows[2] == [[cls, sep]], "an empty text is one empty window, so it still gets a row"


def test_every_text_counts_once_however_many_windows_it_takes():
    """
    Units are windows, but the label belongs to the text: a text of five
    windows carries weight 1/5 on each, so it adds up to one text's worth of
    loss in the weighted mean, same as a one-window tweet.
    """
    import torch

    from taters.text.finetune_predictor import _loss, _unit_weights, _units

    windows = [[[1], [2], [3], [4], [5]], [[6]]]
    units = _units(windows, [0, 1])
    assert units == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0)]
    assert _unit_weights(windows, units) == [0.2] * 5 + [1.0]

    # a regression head's squared errors of 1,1,1,1,1 (the long text) and 9
    # (the tweet): plain mean 2.33, weighted mean (0.2*5*1 + 1*9) / 2 = 5
    outputs = {"score": torch.zeros(6, 1)}
    target = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 3.0])
    mask = torch.ones(6, dtype=torch.bool)
    tasks = {"score": {"task": "regression"}}
    sw = torch.tensor(_unit_weights(windows, units))
    weighted = float(_loss(outputs, {"score": (target, mask)}, tasks, {"score": 1.0},
                           sample_weight=sw))
    plain = float(_loss(outputs, {"score": (target, mask)}, tasks, {"score": 1.0}))
    assert abs(weighted - 5.0) < 1e-6
    assert abs(plain - 14 / 6) < 1e-6


def test_a_texts_windows_stay_on_one_side_of_a_split():
    """Folds are dealt by text; the units of a fold are all of its texts'
    windows and none of any other text's."""
    from taters.text.finetune_predictor import _units

    windows = [[[1], [2]], [[3]], [[4], [5], [6]]]
    assert _units(windows, [0, 2]) == [(0, 0), (0, 1), (2, 0), (2, 1), (2, 2)]
    assert _units(windows, [1]) == [(1, 0)]


def test_a_long_text_is_scored_as_the_mean_over_its_windows(tmp_path, encoder):
    """
    Text B is text A followed by text C, with max_length set so that A fills
    exactly one window: B's windows are then A's and C's, and its prediction
    has to be their mean -- for the regression value and for every class
    probability. Before, B was scored as A (its first window) and the rest
    was thrown away.
    """
    _fit(tmp_path, encoder)
    tok = _tok(encoder)
    a = " ".join(FOOD + FOOD[:2])                   # 10 words
    c = " ".join(WORK)                              # 6 words, fewer tokens than a
    n_a = len(tok(a, add_special_tokens=False)["input_ids"])
    assert n_a >= 8 and len(tok(c, add_special_tokens=False)["input_ids"]) <= n_a
    assert tok(a + " " + c, add_special_tokens=False)["input_ids"] == \
        tok(a, add_special_tokens=False)["input_ids"] + tok(c, add_special_tokens=False)["input_ids"]
    path = tmp_path / "abc.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text"])
        w.writerows([["a", a], ["b", a + " " + c], ["c", c]])
    rows = _read(apply_text_predictor(model_json=_model(tmp_path), csv_path=path,
                                      text_cols=["text"], id_cols=["pid"],
                                      out_features_csv=tmp_path / "abc_scored.csv",
                                      device="cpu", verbose=False, max_length=n_a + 2))
    by = {r["text_id"]: r for r in rows}
    assert [by[k]["n_windows"] for k in "abc"] == ["1", "2", "1"]
    for col in ("pred_score", "p_topic_food", "p_topic_work"):
        mean = (float(by["a"][col]) + float(by["c"][col])) / 2
        assert abs(float(by["b"][col]) - mean) < 1e-4, (col, by["a"][col], by["b"][col], by["c"][col])
    assert abs(float(by["a"]["pred_score"]) - float(by["c"]["pred_score"])) > 0.05, \
        "the two halves have to disagree for the test to mean anything"


def test_a_named_model_takes_its_name_as_its_file_stem(tmp_path, encoder):
    """
    The Train task names a model after its encoder (distilroberta-base-
    finetuned). A name that only lived inside the manifest while the file kept
    calling itself text_predictor__score_topic would put two names on one
    model, and the library shows the file.
    """
    _fit(tmp_path, encoder, name="tiny-finetuned")
    manifest = _model(tmp_path, "tiny-finetuned")
    assert manifest.is_file()
    assert (manifest.parent / "tiny-finetuned.predictor").is_dir()
    assert json.loads(manifest.read_text(encoding="utf-8"))["name"] == "tiny-finetuned"
    assert not _model(tmp_path).exists(), "the outcomes stem is for unnamed models"


def test_a_warm_start_from_a_saved_predictor_reuses_its_heads(tmp_path, encoder):
    _fit(tmp_path, encoder, epochs=1)
    first = _model(tmp_path)
    second = finetune_text_predictor(
        csv_path=_study(tmp_path / "s2.csv", seed=9), text_cols=["text"], id_cols=["pid"],
        outcome_cols=["score", "code"], categorical_outcomes=["code"],
        out_dir=tmp_path / "again", base_model=str(first), n_folds=2, epochs=1,
        max_length=32, batch_size=8, device="cpu", precision="fp32", verbose=False)
    doc = json.loads((tmp_path / "again" / "models" / "text_predictor__score_code.json")
                     .read_text(encoding="utf-8"))
    assert doc["reused_heads"] == ["score"], "same name and task: reused; code is new"
    assert doc["started_from"] == "text_predictor__score_topic"
    assert doc["base_model"] == json.loads(first.read_text(encoding="utf-8"))["base_model"]
    assert "reusing its heads for score" in \
        (tmp_path / "again" / "models" / "text_predictor__score_code_report.md").read_text(encoding="utf-8")
    assert second.is_file()


def test_score_with_model_dispatches_to_the_predictor(tmp_path, encoder):
    from taters.score_model import score_with_model

    _fit(tmp_path, encoder, epochs=1)
    out = score_with_model(model_json=_model(tmp_path), csv_path=tmp_path / "s.csv",
                           out_csv=tmp_path / "scored.csv", verbose=False)
    rows = _read(out)
    assert {"pred_score", "pred_topic"} <= set(rows[0]) and len(rows) == 60


def test_the_registry_describes_the_predictor(tmp_path, encoder):
    _fit(tmp_path, encoder, epochs=1)
    info = describe(_model(tmp_path))
    assert (info.type_id, info.needs) == ("text_predictor", "text")
    assert info.outputs == ("score", "topic")
    assert info.columns == ("pred_score", "pred_topic", "prob_topic", "p_topic_food",
                            "p_topic_work")
    assert info.classes == {"topic": ("food", "work")}
    assert info.apply["batch_size"] == 8 and info.apply["emit_probabilities"] is True
    from taters.helpers import library as lib

    assert lib.asset_problem(_model(tmp_path), lib.KINDS["models"]) == ""


def test_the_report_section_and_figures_are_written(tmp_path, encoder):
    _fit(tmp_path, encoder)
    stats = tmp_path / "stats"
    section = (stats / "_sections" / "45-text-predictor.md").read_text(encoding="utf-8")
    assert section.startswith("## Prediction (fine-tuned transformer)")
    assert "### score" in section and "### topic" in section
    assert "out-of-fold R²" in section and "always-the-commonest baseline" in section
    figs = stats / "figures" / "text_predictor"
    assert (figs / "loss_per_fold.png").is_file()
    assert (figs / "predicted_vs_observed_score.png").is_file()
    assert (figs / "confusion_topic.png").is_file()
    assert (figs / "final_model_loss.png").is_file()
    report = (stats / "models" / "text_predictor__score_topic_report.md").read_text(encoding="utf-8")
    assert "## Methods paragraph" in report and "cross-validation" in report
    assert "| class | support | precision | recall | F1 |" in report
    assert "![loss per fold]" in report and "safetensors" in report
    per_class = _read(stats / "text_predictor_per_class.csv")
    assert [r["class"] for r in per_class] == ["food", "work"]
    confusion = _read(stats / "text_predictor_confusion.csv")
    assert sum(int(r["n"]) for r in confusion) == 60


def test_a_damaged_model_and_bad_settings_are_refused_by_name(tmp_path, encoder):
    _fit(tmp_path, encoder, epochs=1)
    model = _model(tmp_path)
    (tmp_path / "stats" / "models" / "text_predictor__score_topic.predictor" /
     "heads.safetensors").unlink()
    with pytest.raises(ValueError, match="no heads.safetensors"):
        fp._load_model(model)
    with pytest.raises(ValueError, match="names no column"):
        _fit(tmp_path, encoder, outcome_cols=[], out_dir=tmp_path / "x")
    with pytest.raises(ValueError, match="at least 2"):
        _fit(tmp_path, encoder, n_folds=1, out_dir=tmp_path / "y")
    with pytest.raises(ValueError, match="not in the table"):
        _fit(tmp_path, encoder, outcome_cols=["nowhere"], out_dir=tmp_path / "z")


def test_an_existing_result_is_returned_untouched(tmp_path, encoder):
    out = _fit(tmp_path, encoder, epochs=1)
    stamp = out.read_bytes()
    # different settings, same output path: the resume contract says we hand
    # back what's already there rather than silently refitting over it.
    assert _fit(tmp_path, encoder, epochs=2) == out and out.read_bytes() == stamp


def test_the_loss_ignores_rows_that_lack_an_outcome():
    """A row missing one outcome must not pull that head towards the
    stand-in target; only the rows that have the outcome count."""
    tasks = {"score": {"task": "regression", "mean": 0.0, "std": 1.0},
             "topic": {"task": "classification", "classes": ["a", "b"]}}
    outputs = {"score": torch.tensor([[1.0], [100.0]]),
               "topic": torch.tensor([[2.0, 0.0], [0.0, 50.0]])}
    rows = [{"score": "1.0", "topic": "a"}, {"score": "", "topic": ""}]
    targets = fp._targets(rows, tasks, "cpu")
    assert targets["score"][1].tolist() == [True, False]
    loss = fp._loss(outputs, targets, tasks, {"score": 1.0, "topic": 1.0})
    expected = torch.nn.functional.cross_entropy(outputs["topic"][:1], torch.tensor([0]))
    assert abs(float(loss) - float(expected)) < 1e-6, "row two contributed nothing"
    weighted = fp._loss(outputs, targets, tasks, {"score": 1.0, "topic": 3.0})
    assert abs(float(weighted) - 3 * float(expected)) < 1e-6


def test_early_stopping_hands_back_the_best_epochs_weights(tmp_path, encoder):
    """A learning rate high enough to wreck later epochs makes the best
    epoch an early one; the model returned must be that epoch's, so its
    validation loss equals the lowest in the history."""
    from taters.text._transformer_common import load_encoder, parse_layers

    study = _study(tmp_path / "s.csv")
    with study.open(encoding="utf-8", newline="") as fh:
        table = list(csv.DictReader(fh))
    tasks = fp._outcome_tasks(table, ["score"], [])
    _m, tokenizer, _r, _d, _f = load_encoder(encoder, device="cpu")
    idx, combine = parse_layers("last", 2)
    train_rows, val_rows = fp._split_validation(list(range(len(table))), 0.25, seed=1)
    common = dict(layer_idx=idx, combine=combine, pooling="mean", max_length=32,
                  batch_size=8, grad_accum=1, epochs=4, learning_rate=0.05,
                  weight_decay=0.0, warmup_fraction=0.0, train_layers=0,
                  gradient_checkpointing=False, device_name="cpu", precision="fp32",
                  seed=4, early_stopping=True, warm_heads=None, on_progress=None,
                  label="t", verbose=False)
    windows, _counts = fp._windows_of(tokenizer, [r["text"] for r in table], 32)
    model, heads, history, best = fp._train_one(
        str(encoder), tokenizer, table, tasks, {"score": 1.0}, train_rows, val_rows,
        windows=windows, **common)
    losses = [h["val_loss"] for h in history]
    assert best == losses.index(min(losses)) + 1
    if best == 4:
        # this seed overfits after epoch 2 on every torch build we've tried; on a
        # build where it doesn't, there's nothing to restore, and we'd rather say
        # so than have a test pass without ever exercising the path.
        pytest.skip("no early best epoch on this build; the restore path was not exercised")
    model.eval()
    heads.eval()
    units = fp._units(windows, val_rows)
    with torch.inference_mode():
        batch = fp._encode_windows(tokenizer, [windows[t][w] for t, w in units], "cpu")
        outputs = fp._forward(model, heads, batch, layer_idx=idx, combine=combine, pooling="mean")
        now = float(fp._loss(outputs, fp._targets([table[t] for t, _w in units], tasks, "cpu"),
                             tasks, {"score": 1.0},
                             sample_weight=torch.tensor(fp._unit_weights(windows, units))))
    assert abs(now - min(losses)) < 1e-4
