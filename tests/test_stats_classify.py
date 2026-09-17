"""
Classification: the fit must match a reference implementation, the reported
numbers must be honest about an imbalanced outcome, and a saved classifier
must label a new table exactly as its own fit did.

The recurring failure this file guards is a number that looks good and means
nothing: 90% accuracy on an outcome where 90% of people are in one class.
Several tests exist only to pin the *baseline* beside the headline.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from taters.stats.classify import (CLASSIFIER_MODEL_FORMAT,  # noqa: E402
                                   MIN_PER_CLASS, _auc, _cv_logistic,
                                   _fit_logistic, _load_model, _score,
                                   _stratified_folds, apply_classifier_csv,
                                   fit_classifier_csv)
from taters.stats.ridge import default_alphas  # noqa: E402
from csvhelpers import _read, _write  # noqa: E402


def _table(tmp_path, *, n=240, seed=5, name="analysis_table.csv",
           imbalance=0.5, with_controls=False):
    """A planted study: `sig` predicts the class, `noise` does not."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    noise = rng.normal(size=n)
    age = rng.normal(40, 10, n)
    gender = rng.choice(["female", "male"], n)
    lift = np.log(imbalance / (1 - imbalance))
    lin = 1.6 * sig + lift
    if with_controls:
        lin = lin + 0.07 * (age - 40) + 0.6 * (gender == "male")
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-lin)), "yes", "no")
    header = ["text_id", "cond", "age", "gender", "sig", "noise"]
    rows = [[f"d{i}", y[i], f"{age[i]:.6f}", gender[i],
             f"{sig[i]:.10f}", f"{noise[i]:.10f}"] for i in range(n)]
    return _write(tmp_path / name, header, rows)


def _two_class(n=240, p=3, seed=0):
    """A clean, well-separated problem -- the case that breaks anything
    dividing by p(1-p), so the one worth testing the selection on."""
    import numpy as np

    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    x = np.column_stack([signal] + [rng.normal(size=n) for _ in range(p - 1)])
    labels = np.array(["yes" if v > 0 else "no"
                       for v in signal + rng.normal(scale=0.3, size=n)])
    return x, labels, ["no", "yes"]


def test_no_row_can_influence_the_penalty_that_predicts_it():
    """
    What makes the classifier's log loss -- and the AUC and accuracy read off
    the same probabilities -- out of fold in the full sense.

    Each row is held out by exactly one fold, and that fold's penalty is
    chosen by an inner k-fold over its *training* rows only. So this row's
    label cannot reach the model that predicts it. Flip it and its own
    held-out probabilities must not move at all.

    The old shape could not promise this: it fitted the grid in every fold,
    pooled the out-of-fold probabilities, and chose the penalty with the
    lowest pooled loss -- a loss that included this very row.
    """
    import numpy as np

    x, labels, classes = _two_class()
    grid = default_alphas()
    first, _ = _cv_logistic(x, labels, classes, grid=grid, n_folds=5, seed=5,
                            zscore=True, stratify=False)

    # every label in one fold, flipped. That fold's training rows are all the
    # *other* folds, so none of this touches them -- while a selection made
    # over the pooled predictions would see fifty changed labels and move.
    # (One flipped label is too small to shift an argmin over 33 penalties,
    # which made an earlier version of this test pass for the wrong reason.)
    held = first["folds"] == 0
    flipped = labels.copy()
    flipped[held] = np.where(labels[held] == "no", "yes", "no")
    second, _ = _cv_logistic(x, flipped, classes, grid=grid, n_folds=5, seed=5,
                             zscore=True, stratify=False)

    assert np.array_equal(first["folds"], second["folds"]), \
        "the deal moved, so this compares two different splits"
    assert first["fold_alphas"][0] == second["fold_alphas"][0], \
        "labels inside fold 0 changed the penalty chosen for fold 0"
    assert first["oof"][held] == pytest.approx(second["oof"][held]), \
        "labels inside fold 0 changed fold 0's own held-out probabilities"


def test_the_penalty_is_chosen_well_on_a_separable_problem():
    """
    A regression test for a fix that was tried and rejected.

    The one-step leave-one-out approximation (Pregibon 1981) would have made
    this free, the way the ridge's exact identity does -- but it divides by
    `p(1-p)`, which collapses on confident predictions, so on a clean problem
    it drove the choice to maximum shrinkage and the classifier to a single
    class. Brute-force leave-one-out wanted alpha near 0.01 here; the
    approximation asked for 1e5 and a coin flip.

    So: a well-separated problem has to come back with a *small* penalty, two
    classes predicted, and a log loss nowhere near log(2).
    """
    import numpy as np

    x, labels, classes = _two_class()
    grid = default_alphas()
    fit, _ = _cv_logistic(x, labels, classes, grid=grid, n_folds=5, seed=1,
                          zscore=True, stratify=True)

    assert fit["alpha"] < 100.0, (
        f"chose alpha={fit['alpha']:g} on a separable problem -- that is the "
        "shrink-to-nothing failure the approximation had")
    assert fit["log_loss"] < 0.45, fit["log_loss"]
    predicted = {classes[i] for i in np.argmax(fit["oof"], axis=1)}
    assert predicted == set(classes), \
        f"collapsed to {predicted} instead of predicting both classes"


def test_each_fold_chooses_its_own_penalty_and_says_so():
    x, labels, classes = _two_class()
    grid = default_alphas()
    fit, _ = _cv_logistic(x, labels, classes, grid=grid, n_folds=5, seed=2,
                          zscore=True, stratify=True)
    assert len(fit["fold_alphas"]) == 5
    assert all(a in grid for a in fit["fold_alphas"])
    assert fit["alpha"] in grid



def test_every_outcome_gets_a_classifier_file_of_its_own(tmp_path):
    """The ridge's rule, and for the same reason: a model is the thing that
    predicts one outcome, so two outcomes are two models."""
    from taters.helpers.model_spec import describe

    table = _table(tmp_path)
    rows = _read(table)
    header = list(rows[0]) + ["mood"]
    out = tmp_path / "two.csv"
    _write(out, header,
           [[r[c] for c in rows[0]] + ["up" if r["cond"] == "yes" else "down"]
            for r in rows])

    fit_classifier_csv(table_csv=out, outcome_cols=["cond", "mood"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "fit", verbose=False)
    models = sorted(p.name for p in (tmp_path / "fit" / "models").glob("*.json"))
    assert models == ["classifier__lang__cond.json", "classifier__lang__mood.json"]
    for outcome in ("cond", "mood"):
        path = tmp_path / "fit" / "models" / f"classifier__lang__{outcome}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert list(doc["outcomes"]) == [outcome]
        assert describe(path).outputs == (outcome,)


# --------------------------------------------------------------- the math

def test_the_logistic_fit_matches_scikit_learn(tmp_path):
    """Hand-rolled IRLS, so it is checked against the implementation everyone
    else uses. A penalized logistic fit that is subtly wrong -- a penalized
    intercept, a missing weight -- still converges to plausible-looking
    coefficients, which is exactly why this needs an external referee."""
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 3))
    y = (rng.random(400) < 1 / (1 + np.exp(-(x @ [1.5, -0.8, 0.0] + 0.5)))
         ).astype(float)

    alpha = 2.0
    intercept, coef = _fit_logistic(x, y, alpha)
    # sklearn parameterizes the penalty as C = 1/alpha, and it doesn't
    # penalize its intercept either
    ref = sk.LogisticRegression(C=1.0 / alpha, fit_intercept=True,
                                tol=1e-10, max_iter=5000).fit(x, y)
    assert coef == pytest.approx(ref.coef_[0], abs=1e-3)
    assert intercept == pytest.approx(float(ref.intercept_[0]), abs=1e-3)


def test_the_area_under_the_curve_matches_scikit_learn():
    """Computed from ranks rather than by sweeping thresholds, which is exact
    and handles ties by average rank -- but only if the Mann-Whitney identity
    is applied correctly, and an off-by-one in it produces a number that is
    still between 0 and 1."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    for _ in range(5):
        truth = (rng.random(200) < 0.4).astype(int)
        score = rng.random(200)
        assert _auc(truth, score) == pytest.approx(
            sk.roc_auc_score(truth, score), abs=1e-12)
    # now ties, where a threshold sweep and the rank identity can disagree
    truth = np.array([0, 0, 1, 1])
    tied = np.array([0.5, 0.5, 0.5, 0.9])
    assert _auc(truth, tied) == pytest.approx(
        sk.roc_auc_score(truth, tied), abs=1e-12)


def test_the_per_class_report_matches_scikit_learn():
    """Precision, recall and F1 per class, plus both averages. The macro and
    weighted averages disagree exactly when the imbalance matters, so getting
    one of them subtly wrong is invisible on a balanced fixture."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    classes = ["a", "b", "c"]
    labels = np.array(rng.choice(classes, 300, p=[0.6, 0.3, 0.1]))
    probs = rng.random((300, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)
    got = _score(labels, classes, probs)
    predicted = got["predicted"]

    ref = sk.classification_report(labels, predicted, labels=classes,
                                   output_dict=True, zero_division=0)
    for c in classes:
        for key in ("precision", "recall", "f1-score"):
            ours = got["per_class"][c][key.replace("f1-score", "f1")]
            assert ours == pytest.approx(ref[c][key], abs=1e-12)
        assert got["per_class"][c]["support"] == ref[c]["support"]
    assert got["macro"]["f1"] == pytest.approx(ref["macro avg"]["f1-score"],
                                               abs=1e-12)
    assert got["weighted"]["f1"] == pytest.approx(
        ref["weighted avg"]["f1-score"], abs=1e-12)
    assert got["accuracy"] == pytest.approx(ref["accuracy"], abs=1e-12)


def test_every_fold_contains_every_class():
    """Dealt within class, not over the whole sample. A plain shuffle can hand
    a fold none of a small class -- and a fold with no positive cases has no
    recall and no area under any curve, so one unlucky split silently turns
    five-fold validation into four-fold."""
    labels = np.array(["rare"] * 12 + ["common"] * 200)
    classes = ["common", "rare"]
    folds = _stratified_folds(labels, classes, 5, seed=0)
    for f in range(5):
        present = set(labels[folds == f])
        assert present == {"common", "rare"}, f"fold {f} lost a class"
    # and the same seed had better deal the same hand, or nothing's reproducible
    assert (folds == _stratified_folds(labels, classes, 5, seed=0)).all()


# ------------------------------------------------------- honest reporting

def test_accuracy_is_always_reported_beside_the_do_nothing_baseline(tmp_path):
    """The number everyone quotes, and close to useless alone: on an outcome
    where 88% of rows are in one class, a model that always answers "that
    one" is 88% accurate. Without the baseline in the same row, that model
    reads as a success."""
    table = _table(tmp_path, n=300, imbalance=0.88, seed=9)
    out = fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                             feature_sets={"lang": ["sig", "noise"]},
                             out_dir=tmp_path / "stats", verbose=False)
    row = _read(out)[0]
    assert 0.8 < float(row["baseline_accuracy"]) < 0.95
    assert float(row["accuracy"]) >= float(row["baseline_accuracy"]) - 0.05
    # the number we lead with doesn't care about the imbalance, and the signal
    # we planted is real, so it should be well clear of chance
    assert float(row["auc"]) > 0.7


def test_the_per_class_table_shows_where_the_model_is_useless(tmp_path):
    """A model can be excellent on the majority class and useless on the one
    the researcher cares about, and every average hides that. The per-class
    file is the only place that shows it, so it must carry a row per class
    with the support behind each number."""
    table = _table(tmp_path, n=300, imbalance=0.85, seed=4)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "stats", verbose=False)
    rows = _read(tmp_path / "stats" / "classifier_per_class.csv")
    by_class = {r["class"]: r for r in rows}
    assert set(by_class) == {"no", "yes"}
    assert sum(int(r["support"]) for r in rows) == 300
    for r in rows:
        assert r["precision"] and r["recall"] and r["f1"]

    # and the confusion matrix is the only output that tells you what the
    # mistakes actually were
    conf = _read(tmp_path / "stats" / "classifier_confusion.csv")
    assert len(conf) == 4
    assert sum(int(r["n"]) for r in conf) == 300


def test_every_reported_number_is_out_of_fold(tmp_path):
    """Each row is classified by a model that never saw it. Scored in-sample
    the numbers would be higher and would mean nothing, so the training
    accuracy rides along in its own column -- named as such -- and the gap
    between the two is the thing a reader needs."""
    table = _table(tmp_path, n=200, seed=2)
    out = fit_classifier_csv(table_csv=table, rounding=6, outcome_cols=["cond"],
                             feature_sets={"lang": ["sig", "noise"]},
                             out_dir=tmp_path / "stats", verbose=False)
    row = _read(out)[0]
    assert float(row["train_accuracy"]) >= float(row["accuracy"]) - 0.02
    assert int(row["n_folds"]) == 5

    folds = _read(tmp_path / "stats" / "classifier_folds.csv")
    assert len(folds) == 5
    assert sum(int(r["n"]) for r in folds) == int(row["n_used"])
    # the per-fold spread that the standard error comes from has to be there
    # to read, otherwise nobody can check the claim in the metrics table
    per_fold = np.array([float(r["accuracy"]) for r in folds])
    assert float(row["accuracy_folds"]) == pytest.approx(per_fold.mean(),
                                                         abs=1e-6)
    assert float(row["accuracy_folds_se"]) == pytest.approx(
        per_fold.std(ddof=1) / np.sqrt(per_fold.size), abs=1e-6)


def test_the_predictions_carry_both_the_model_and_the_held_out_verdict(
        tmp_path):
    """Two different predictions, and conflating them is the mistake: the
    saved model's own answer (which reproduces exactly through apply) and the
    answer from the fold that had not seen the row (which is the one to use
    downstream). The fold is beside them so a suspicious result can be traced
    to one split."""
    table = _table(tmp_path, n=120, seed=6)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "stats", verbose=False)
    rows = _read(tmp_path / "stats" / "classifier_predictions__lang.csv")
    assert len(rows) == 120
    for r in rows:
        assert r["cond"] in ("no", "yes")
        assert r["pred_cond"] in ("no", "yes")
        assert r["oof_pred_cond"] in ("no", "yes")
        assert 0.0 <= float(r["oof_prob_cond"]) <= 1.0
        assert 1 <= int(r["fold_cond"]) <= 5
    # the two should disagree somewhere, or one of them isn't what it claims
    assert any(r["pred_cond"] != r["oof_pred_cond"] for r in rows)


# ------------------------------------------------------------- refusals

def test_a_measurement_handed_to_the_classifier_is_sent_to_the_ridge(
        tmp_path):
    """The two modules are each other's near miss. Told to classify age, the
    old numeric reader said "'40.2' is not a category, fix the value", which
    is confident, unhelpful and wrong -- nothing about the cell needs fixing.
    Each module now names the other."""
    table = _table(tmp_path, n=60, seed=1)
    with pytest.raises(ValueError) as e:
        fit_classifier_csv(table_csv=table, outcome_cols=["age"],
                           feature_sets={"lang": ["sig"]},
                           out_dir=tmp_path / "s1", verbose=False)
    assert "ridge" in str(e.value)

    from taters.stats.ridge import fit_ridge_csv
    with pytest.raises(ValueError) as e:
        fit_ridge_csv(table_csv=table, outcome_cols=["gender"],
                      feature_sets={"lang": ["sig"]},
                      out_dir=tmp_path / "s2", verbose=False)
    assert "classification step" in str(e.value)
    assert "words rather than numbers" in str(e.value)


def test_a_class_too_thin_to_appear_in_every_fold_is_refused(tmp_path):
    """A class with three rows cannot be in all five folds, so its recall is
    measured on nothing and its per-class row reads as a real result. Refused
    while the user can act, naming the class and its count."""
    rows = [[f"d{i}", "yes" if i < 4 else "no", "40", "female",
             f"{i * 0.01}", "0.5"] for i in range(60)]
    table = _write(tmp_path / "thin.csv",
                   ["text_id", "cond", "age", "gender", "sig", "noise"], rows)
    with pytest.raises(ValueError) as e:
        fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                           feature_sets={"lang": ["sig", "noise"]},
                           out_dir=tmp_path / "stats", verbose=False)
    assert str(MIN_PER_CLASS) in str(e.value)
    assert "'yes': 4" in str(e.value) or "yes" in str(e.value)


def test_an_outcome_with_one_class_or_too_many_is_refused(tmp_path):
    """One class is not a comparison; a hundred is an identifier. Both would
    otherwise produce a table of rows that look like results."""
    same = [[f"d{i}", "yes", "40", "female", f"{i * 0.01}", "0.5"]
            for i in range(40)]
    header = ["text_id", "cond", "age", "gender", "sig", "noise"]
    with pytest.raises(ValueError, match="at least two"):
        fit_classifier_csv(table_csv=_write(tmp_path / "one.csv", header,
                                            same),
                           outcome_cols=["cond"],
                           feature_sets={"lang": ["sig"]},
                           out_dir=tmp_path / "a", verbose=False)
    unique = [[f"d{i}", f"label_{i}", "40", "female", f"{i * 0.01}", "0.5"]
              for i in range(40)]
    with pytest.raises(ValueError, match="identifier"):
        fit_classifier_csv(table_csv=_write(tmp_path / "many.csv", header,
                                            unique),
                           outcome_cols=["cond"],
                           feature_sets={"lang": ["sig"]},
                           out_dir=tmp_path / "b", verbose=False)


# --------------------------------------------------------------- controls

def test_the_language_is_credited_only_for_what_it_adds_over_controls(
        tmp_path):
    """Three models on ONE common sample: controls alone, language alone,
    both. Listwise deletion over different column sets would give the
    controls-only model more rows than the combined one, and the difference
    between their scores -- which is the whole point of running both -- would
    then be part sample and part language."""
    table = _table(tmp_path, n=300, seed=8, with_controls=True)
    out = fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        feature_sets={"lang": ["sig", "noise"]},
        control_cols=["age", "gender"], categorical_controls=["gender"],
        out_dir=tmp_path / "stats", verbose=False)
    rows = {r["model"]: r for r in _read(out)}
    assert set(rows) == {"controls", "language", "controls+language"}
    assert len({r["n_used"] for r in rows.values()}) == 1, \
        "the three models were not fitted on one common sample"

    both = rows["controls+language"]
    assert float(both["auc"]) > float(rows["controls"]["auc"])
    assert float(both["delta_auc_over_controls"]) == pytest.approx(
        float(both["auc"]) - float(rows["controls"]["auc"]), abs=1e-6)
    # the controls are never also predictors: if `age` were both, it'd get
    # credited to the language it was supposed to be adjusted away from
    coefs = _read(tmp_path / "stats" / "classifier_coefficients.csv")
    language_only = [r for r in coefs if r["model"] == "language"]
    assert {r["predictor"] for r in language_only} == {"sig", "noise"}


def test_a_string_valued_outcome_and_control_need_no_recoding(tmp_path):
    """The researcher's spreadsheet says `male` and `female`, and asking them
    to dummy-code it by hand before Taters will look at it is asking them to
    do the part that is easy to get wrong."""
    table = _table(tmp_path, n=200, seed=12, with_controls=True)
    out = fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        feature_sets={"lang": ["sig"]},
        control_cols=["gender"], categorical_controls=["gender"],
        out_dir=tmp_path / "stats", verbose=False)
    assert any(r["controls"] == "gender" for r in _read(out))
    coefs = _read(tmp_path / "stats" / "classifier_coefficients.csv")
    # coded against the alphabetically first level, and the report says so
    # (a coefficient on `gender=male` means nothing without that)
    assert any(r["predictor"] == "gender=male" for r in coefs)
    section = next((tmp_path / "stats" / "_sections").glob("*.md")).read_text()
    assert "compared against 'female'" in section


def test_a_blank_control_leaves_the_row_out_rather_than_guessing(tmp_path):
    """A row whose gender is unrecorded is not thereby female. Coded as the
    reference level it would be scored anyway, and silently."""
    rng = np.random.default_rng(15)
    rows = []
    for i in range(200):
        gender = "" if i < 20 else ("male" if i % 2 else "female")
        rows.append([f"d{i}", "yes" if rng.random() < 0.5 else "no",
                     "40", gender, f"{rng.normal():.6f}", "0.5"])
    table = _write(tmp_path / "gaps.csv",
                   ["text_id", "cond", "age", "gender", "sig", "noise"], rows)
    out = fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        feature_sets={"lang": ["sig"]},
        control_cols=["gender"], categorical_controls=["gender"],
        control_combos="none_and_all",
        out_dir=tmp_path / "stats", verbose=False)
    used = {int(r["n_used"]) for r in _read(out)}
    assert used == {180}, f"the 20 blank rows were not dropped: {used}"


# ------------------------------------------------------- the saved model

def test_a_classifier_fitted_here_labels_a_new_table_identically(tmp_path):
    """One prediction path for fit and apply. Two nearly-identical ones drift
    apart on the first change to either, and the symptom -- a model that
    scores differently through the two doors -- is the kind nobody notices."""
    table = _table(tmp_path, n=200, seed=21)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "stats", verbose=False)
    model = tmp_path / "stats" / "models" / "classifier__lang__cond.json"
    scored = apply_classifier_csv(model_json=model, input_csv=table,
                                  out_csv=tmp_path / "again.csv",
                                  verbose=False)
    theirs = _read(tmp_path / "stats" / "classifier_predictions__lang.csv")
    ours = _read(scored)
    assert len(ours) == len(theirs) == 200
    for a, b in zip(theirs, ours):
        assert a["pred_cond"] == b["pred_cond"]
        assert a["prob_cond"] == b["prob_cond"]
    # and a probability per class, since "which class" hides how close the
    # call was: a 0.51 verdict and a 0.99 one shouldn't read the same
    assert {"p_cond_no", "p_cond_yes"} <= set(ours[0])
    for r in ours:
        assert float(r["p_cond_no"]) + float(r["p_cond_yes"]) == \
            pytest.approx(1.0, abs=1e-4)


def test_a_control_level_the_model_never_saw_goes_unscored(tmp_path):
    """Zeroing every indicator would code the row as the reference level and
    score it anyway -- so a respondent recorded as something the training
    data lacked would be silently treated as female."""
    table = _table(tmp_path, n=200, seed=22, with_controls=True)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig"]},
                       control_cols=["gender"],
                       categorical_controls=["gender"],
                       out_dir=tmp_path / "stats", verbose=False)
    model = tmp_path / "stats" / "models" / "classifier__lang__cond.json"
    # the model was fitted with `gender` and keeps the recipe, so it can
    # score a raw spreadsheet whose gender column is still words
    rows = _read(table)
    rows[0]["gender"] = "nonbinary"
    fresh = _write(tmp_path / "b.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])
    scored = _read(apply_classifier_csv(model_json=model, input_csv=fresh,
                                        out_csv=tmp_path / "s.csv",
                                        verbose=False))
    assert scored[0]["pred_cond"] == ""
    assert all(r["pred_cond"] for r in scored[1:])


def test_the_model_gate_turns_away_everything_that_is_not_one(tmp_path):
    """A model is an instrument someone publishes from, so a file that merely
    parses is not enough. Every refusal happens while the user is holding the
    file and names what is wrong."""
    table = _table(tmp_path, n=120, seed=31)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "stats", verbose=False)
    good = tmp_path / "stats" / "models" / "classifier__lang__cond.json"
    assert _load_model(good)["kind"] == "taters-classifier-model"

    doc = json.loads(good.read_text(encoding="utf-8"))

    def write(**changes):
        path = tmp_path / "candidate.json"
        merged = {**doc, **changes}
        path.write_text(json.dumps(merged), encoding="utf-8")
        return path

    # a ridge model is the near miss, so we point it back at its own step
    with pytest.raises(ValueError) as e:
        _load_model(write(kind="taters-ridge-model"))
    assert "ridge step" in str(e.value)
    with pytest.raises(ValueError, match="newer Taters"):
        _load_model(write(format=CLASSIFIER_MODEL_FORMAT + 1))
    with pytest.raises(ValueError, match="names no predictors"):
        _load_model(write(predictors=[]))
    with pytest.raises(ValueError, match="no fitted outcomes"):
        _load_model(write(outcomes={}))

    def broken_outcome(**changes):
        block = {**doc["outcomes"]["cond"], **changes}
        return write(outcomes={"cond": block})

    with pytest.raises(ValueError, match="fewer than two classes"):
        _load_model(broken_outcome(classes=["only"]))
    with pytest.raises(ValueError, match="keeps no predictors"):
        _load_model(broken_outcome(kept=[]))
    with pytest.raises(ValueError, match="do not exist"):
        _load_model(broken_outcome(kept=[99]))
    with pytest.raises(ValueError, match="mismatched sizes"):
        _load_model(broken_outcome(mu=[0.0]))
    with pytest.raises(ValueError, match="fitted class model"):
        _load_model(broken_outcome(models=[]))
    with pytest.raises(ValueError, match="no fitted coefficients"):
        _load_model(broken_outcome(models=[None, None]))
    with pytest.raises(ValueError, match="coefficient"):
        _load_model(broken_outcome(
            models=[{"intercept": 0.0, "coef": [1.0, 2.0, 3.0]}, None]))
    with pytest.raises(ValueError, match="no intercept"):
        _load_model(broken_outcome(models=[{"coef": [1.0]}, None]))


def test_the_same_seed_fits_the_same_classifier_twice(tmp_path):
    """A model someone publishes from has to be reproducible, and the fold
    shuffle is the only randomness in it."""
    table = _table(tmp_path, n=180, seed=41)
    first = _read(fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        feature_sets={"lang": ["sig", "noise"]},
        out_dir=tmp_path / "a", seed=3, verbose=False))
    second = _read(fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        feature_sets={"lang": ["sig", "noise"]},
        out_dir=tmp_path / "b", seed=3, verbose=False))
    assert first == second
    other = _read(fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        feature_sets={"lang": ["sig", "noise"]},
        out_dir=tmp_path / "c", seed=4, verbose=False))
    assert other[0]["accuracy"] != first[0]["accuracy"] or \
        other[0]["auc"] != first[0]["auc"], \
        "the seed changed nothing, so the folds are not being shuffled"


def test_results_that_already_exist_are_not_recomputed(tmp_path):
    """The overwrite contract every step in Taters keeps: a re-run after
    adding one measure must not redo the classification."""
    table = _table(tmp_path, n=120, seed=51)
    out = fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                             feature_sets={"lang": ["sig", "noise"]},
                             out_dir=tmp_path / "stats", verbose=False)
    out.write_text("sentinel\n", encoding="utf-8-sig")
    again = fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                               feature_sets={"lang": ["sig", "noise"]},
                               out_dir=tmp_path / "stats", verbose=False)
    assert again == out
    assert out.read_text(encoding="utf-8-sig") == "sentinel\n"
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "stats", overwrite_existing=True,
                       verbose=False)
    assert out.read_text(encoding="utf-8-sig") != "sentinel\n"


def test_the_saved_model_reproduces_the_training_accuracy_it_reported(
        tmp_path):
    """A cross-path check, and the only one that can catch a standardization
    mistake. `pred_` comes from the saved model through the shared prediction
    helper; `train_accuracy` comes from the fitting code standardizing its
    own design. Comparing the model's predictions against *apply* proves
    only that one shared path agrees with itself -- drop the divide by sigma
    and both doors are wrong together, identically. These two numbers are
    computed by different code and have to agree.

    The predictors are deliberately on wildly different scales. On tidy
    standard-normal features every sigma is about 1, dividing by it changes
    almost nothing, and this test passed while the standardization was
    missing -- which is what the mutation checker said."""
    rng = np.random.default_rng(61)
    n = 200
    big = rng.normal(0, 400, n)            # sigma nowhere near 1
    small = rng.normal(0, 0.002, n)
    lin = 0.004 * big + 800.0 * small
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-lin)), "yes", "no")
    table = _write(tmp_path / "scaled.csv",
                   ["text_id", "cond", "sig", "noise"],
                   [[f"d{i}", y[i], f"{big[i]:.10f}", f"{small[i]:.10f}"]
                    for i in range(n)])
    out = fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                             feature_sets={"lang": ["sig", "noise"]},
                             out_dir=tmp_path / "stats", verbose=False)
    reported = float(_read(out)[0]["train_accuracy"])
    rows = _read(tmp_path / "stats" / "classifier_predictions__lang.csv")
    hits = sum(1 for r in rows if r["pred_cond"] == r["cond"])
    assert hits / len(rows) == pytest.approx(reported, abs=1e-6)


def test_a_control_is_never_also_a_predictor(tmp_path):
    """With no feature set named, the features default to every numeric
    column that is not an outcome -- and `age` used as both a control and a
    predictor would be credited to the language it was supposed to be
    adjusted away from. The delta over controls would then be measuring the
    same variable on both sides."""
    table = _table(tmp_path, n=200, seed=62, with_controls=True)
    fit_classifier_csv(
        table_csv=table, outcome_cols=["cond"],
        control_cols=["age", "gender"], categorical_controls=["gender"],
        out_dir=tmp_path / "stats", verbose=False)
    coefs = _read(tmp_path / "stats" / "classifier_coefficients.csv")
    language = {r["predictor"] for r in coefs if r["model"] == "language"}
    assert language == {"sig", "noise"}, \
        f"a control leaked into the predictors: {language}"


def test_the_classifier_says_how_much_of_the_sample_it_used(tmp_path):
    """
    The ridge table carries n_available and n_dropped_sparse beside n_used,
    and the report warns when a model saw under 80% of the rows that had the
    outcome -- the defense the whole stats guide describes against a
    silent 52-of-938 fit. The classifier wrote neither column and never
    warned, so a classifier fitted on a fifth of the corpus read like any
    other.
    """
    rng = np.random.default_rng(3)
    n = 200
    sig = rng.normal(size=n)
    y = np.where(rng.random(n) < 1 / (1 + np.exp(-1.6 * sig)), "yes", "no")
    rows = []
    for i in range(n):
        rows.append([f"d{i}", y[i], f"{sig[i]:.6f}",
                     # present for 60% of rows: kept, and that costs us 40% of
                     # the sample by listwise deletion
                     f"{rng.normal():.6f}" if i % 5 < 3 else "",
                     # present for 10%: set aside as sparse
                     f"{rng.normal():.6f}" if i % 10 == 0 else ""])
    table = _write(tmp_path / "analysis_table.csv",
                   ["text_id", "cond", "sig", "holey", "rare"], rows)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"], n_folds=3,
                       out_dir=tmp_path / "o", verbose=False)
    metrics = _read(tmp_path / "o" / "classifier_cv_metrics.csv")[0]
    assert metrics["n_available"] == "200"
    assert metrics["n_dropped_sparse"] == "1"
    assert int(metrics["n_used"]) == 120
    section = next((tmp_path / "o" / "_sections").glob("*classifier.md")
                   ).read_text(encoding="utf-8")
    assert "About the sample" in section
    assert "used 120 of 200 rows" in section


def test_the_classifier_report_names_predictors_with_gaps_too(tmp_path):
    """Same rule, same section, as the ridge -- the classifier used to be
    the fitter that forgot to say things."""
    table = _table(tmp_path)
    rows = _read(table)
    for i, r in enumerate(rows):
        if i % 4 == 0:
            r["noise"] = ""
    gappy = _write(tmp_path / "gappy.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])
    fit_classifier_csv(table_csv=gappy, outcome_cols=["cond"],
                       verbose=False, out_dir=tmp_path / "fit")
    section = next((tmp_path / "fit" / "_sections").glob("*classifier*")
                   ).read_text(encoding="utf-8")
    assert "### Predictors with gaps" in section
    assert "`noise` (25%)" in section


def test_a_classifier_fitted_on_components_is_applied_to_raw_features(tmp_path):
    """Same contract as the ridge: the reduction travels in the model."""
    table = _table(tmp_path)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"], pca="all",
                       pca_components=2,
                       feature_sets={"lang": ["sig", "noise"]}, verbose=False,
                       out_dir=tmp_path / "fit")
    model = tmp_path / "fit" / "models" / "classifier__lang__cond.json"
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert doc["input_columns"] == ["sig", "noise"]
    # prefixed by the set it was reduced from, always -- so a predictor name
    # still says where it came from once it is out of this run
    assert all(p.startswith("lang_Component_") for p in doc["predictors"])
    scored = _read(apply_classifier_csv(model_json=model, input_csv=table,
                                        out_csv=tmp_path / "s.csv",
                                        verbose=False))
    rows = _read(table)
    agree = sum(r["pred_cond"] == t["cond"] for r, t in zip(scored, rows))
    assert agree / len(rows) > 0.7, "the components lost the signal"


def test_a_warm_start_reaches_the_same_optimum_in_fewer_steps():
    """The penalized log-likelihood is strictly concave, so Newton lands on
    the same coefficients from any start; beginning from the neighboring
    penalty's fit only saves the steps. On 933 texts by 1024 embedding
    columns the cold version took an hour."""
    from taters.stats.classify import _fit_logistic

    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 40))
    y = (x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.5, size=200) > 0).astype(float)
    cold = _fit_logistic(x, y, 1.0)
    neighbor = _fit_logistic(x, y, 1.5)
    warm = _fit_logistic(x, y, 1.0, start=neighbor)
    assert abs(cold[0] - warm[0]) < 1e-6
    assert np.abs(cold[1] - warm[1]).max() < 1e-6

    # and the start really is where it begins: given one Newton step, a fit
    # that starts at the optimum is already there, one that starts at zero isn't
    from taters.stats import classify

    saved = classify._MAX_IRLS
    try:
        classify._MAX_IRLS = 1
        from_optimum = classify._fit_logistic(x, y, 1.0, start=cold)
        from_zero = classify._fit_logistic(x, y, 1.0)
    finally:
        classify._MAX_IRLS = saved
    assert np.abs(from_optimum[1] - cold[1]).max() < 1e-6
    assert np.abs(from_zero[1] - cold[1]).max() > 1e-3


def test_the_grid_is_walked_from_the_largest_penalty_with_warm_starts(monkeypatch):
    """Each penalty after the first begins from the previous fit."""
    from taters.stats import classify

    seen = []
    real = classify._fit_logistic

    def spy(x, y, alpha, start=None):
        seen.append((alpha, start is not None))
        return real(x, y, alpha, start=start)

    monkeypatch.setattr(classify, "_fit_logistic", spy)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 4))
    labels = np.where(x[:, 0] > 0, "yes", "no")
    classify._cv_logistic(x, labels, ["no", "yes"], grid=[0.1, 1.0, 10.0],
                          n_folds=3, seed=0, zscore=True)
    per_fold = [seen[i:i + 3] for i in range(0, 9, 3)]
    for fold in per_fold:
        assert [a for a, _ in fold] == [10.0, 1.0, 0.1], "largest penalty first"
        assert [w for _, w in fold] == [False, True, True], \
            "every penalty after the first begins from the last fit"


def test_the_classifier_can_use_random_folds_and_says_so(tmp_path):
    """Stratified by class is the default and the reason is in the report;
    turned off, it still runs and the report names the risk."""
    table = _table(tmp_path)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"], verbose=False,
                       stratify=False, out_dir=tmp_path / "plain")
    doc = json.loads((tmp_path / "plain" / "models" / "classifier__all__cond.json")
                     .read_text(encoding="utf-8"))
    assert doc["cv"]["stratified"] is False
    section = next((tmp_path / "plain" / "_sections").glob("*classifier*")).read_text()
    assert "random folds" in section and "not stratified" in section

    fit_classifier_csv(table_csv=table, outcome_cols=["cond"], verbose=False,
                       out_dir=tmp_path / "strat")
    doc = json.loads((tmp_path / "strat" / "models" / "classifier__all__cond.json")
                     .read_text(encoding="utf-8"))
    assert doc["cv"]["stratified"] is True


def test_the_classifier_combines_the_tables_the_same_way(tmp_path):
    """The same preamble, the same answer: each table, every pair, all."""
    from pathlib import Path

    table = _table(tmp_path, n=200, imbalance=0.5, seed=3)
    rows = _read(table)
    header = list(rows[0].keys())
    assert {"sig", "noise"} <= set(header)
    (Path(table).with_name(Path(table).stem + "_sets.json")).write_text(json.dumps({
        "key": ["text_id"], "metadata": ["cond"],
        "sets": {"signal": ["sig"], "junk": ["noise"]},
    }), encoding="utf-8")
    out = fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                             out_dir=tmp_path / "stats", verbose=False)
    rows = {r["feature_set"]: r for r in _read(out)}
    assert set(rows) == {"signal", "junk", "all"}
    assert (rows["all"]["n_feature_sets"], rows["all"]["n_features"]) == ("2", "2")
    assert (rows["signal"]["n_feature_sets"], rows["signal"]["n_features"]) == ("1", "1")
    section = (tmp_path / "stats" / "_sections" / "50-classifier.md").read_text(encoding="utf-8")
    assert "How the feature sets compare" in section and "| AUC |" in section


def test_a_relabeled_class_is_written_as_its_label_in_every_column(tmp_path):
    """The complaint was a results table reading "outcome 0 / outcome 1".
    The relabel in the model file reaches the predicted class and the
    per-class probability headings alike, and the probabilities are the
    same numbers under the new names."""
    from taters.helpers.model_spec import edit_model

    table = _table(tmp_path, n=200, seed=21)
    fit_classifier_csv(table_csv=table, outcome_cols=["cond"],
                       feature_sets={"lang": ["sig", "noise"]},
                       out_dir=tmp_path / "stats", verbose=False)
    model = tmp_path / "stats" / "models" / "classifier__lang__cond.json"
    before = _read(apply_classifier_csv(model_json=model, input_csv=table,
                                        out_csv=tmp_path / "before.csv",
                                        verbose=False))
    edit_model(model, class_names={"cond": {"no": "control", "yes": "patient"}})
    after = _read(apply_classifier_csv(model_json=model, input_csv=table,
                                       out_csv=tmp_path / "after.csv",
                                       verbose=False))
    assert {"p_cond_control", "p_cond_patient"} <= set(after[0])
    assert not {"p_cond_no", "p_cond_yes"} & set(after[0])
    assert {r["pred_cond"] for r in after} == {"control", "patient"}
    for a, b in zip(before, after):
        assert b["pred_cond"] == {"no": "control", "yes": "patient"}[a["pred_cond"]]
        assert b["p_cond_control"] == a["p_cond_no"]
        assert b["prob_cond"] == a["prob_cond"]
