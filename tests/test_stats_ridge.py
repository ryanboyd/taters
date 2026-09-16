"""
Ridge regression: the coefficients must match a reference implementation, the
cross-validation must be honest and reproducible, and a saved model must
score a new table exactly as its own fit did.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from taters.stats.ridge import (RIDGE_MODEL_FORMAT, _load_model,  # noqa: E402
                                _svd_coefficients, apply_ridge_csv,
                                fit_ridge_csv)
from csvhelpers import _read, _write  # noqa: E402


def _table(tmp_path, *, n=120, p=5, seed=7, noise=0.3, name="analysis_table.csv"):
    """y = X w + noise, with known w -- so the fit has something to recover."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    w = np.array([2.0, -1.5, 0.8, 0.0, 0.0][:p])
    y = x @ w + rng.normal(scale=noise, size=n)
    header = ["text_id"] + [f"f{i}" for i in range(p)] + ["outcome"]
    rows = [[f"d{i}"] + [f"{v:.10f}" for v in x[i]] + [f"{y[i]:.10f}"]
            for i in range(n)]
    return _write(tmp_path / name, header, rows), w


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

def test_the_svd_shortcut_equals_solving_each_alpha_directly(tmp_path):
    """One decomposition serves the whole grid; it had better agree with the
    textbook solve it replaces."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 4))
    x -= x.mean(axis=0)
    y = rng.normal(size=60)
    y -= y.mean()
    grid = [0.01, 1.0, 25.0, 1000.0]

    ours = _svd_coefficients(x, y, grid)
    for i, alpha in enumerate(grid):
        direct = np.linalg.solve(x.T @ x + alpha * np.eye(4), x.T @ y)
        assert np.allclose(ours[i], direct, atol=1e-10)


def test_coefficients_match_sklearn(tmp_path):
    sklearn = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 5))
    y = rng.normal(size=80)
    xc, yc = x - x.mean(axis=0), y - y.mean()

    for alpha in (0.1, 10.0, 500.0):
        ours = _svd_coefficients(xc, yc, [alpha])[0]
        ref = sklearn.Ridge(alpha=alpha, fit_intercept=False,
                            solver="svd").fit(xc, yc)
        assert np.allclose(ours, ref.coef_, atol=1e-8)


def test_the_fit_recovers_the_planted_weights(tmp_path):
    table, w = _table(tmp_path, n=300, noise=0.2)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        zscore=False, alphas=[0.01, 0.1, 1.0],
                        verbose=False)
    # wide format: one row per predictor, one column per outcome
    coefs = {r["predictor"]: float(r["outcome"])
             for r in _read(tmp_path / "ridge_coefficients.csv")}
    for i, expected in enumerate(w):
        assert coefs[f"f{i}"] == pytest.approx(expected, abs=0.15)

    metrics = _read(out)[0]
    assert float(metrics["cv_r2"]) > 0.9, "a nearly noiseless fit should predict"
    assert metrics["n_used"] == "300"


def test_a_pure_noise_outcome_is_reported_as_predicting_nothing(tmp_path):
    rng = np.random.default_rng(5)
    header = ["text_id", "a", "b", "outcome"]
    rows = [[f"d{i}", f"{rng.normal():.6f}", f"{rng.normal():.6f}",
             f"{rng.normal():.6f}"] for i in range(80)]
    table = _write(tmp_path / "analysis_table.csv", header, rows)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        verbose=False)
    assert float(_read(out)[0]["cv_r2"]) < 0.2, \
        "out-of-fold R2 must not reward memorizing noise"


# ---------------------------------------------------------------------------
# cross-validation behavior
# ---------------------------------------------------------------------------

def test_the_same_seed_gives_byte_identical_results(tmp_path):
    table, _ = _table(tmp_path)
    first = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                          out_dir=tmp_path / "a", verbose=False).read_bytes()
    second = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                           out_dir=tmp_path / "b", verbose=False).read_bytes()
    assert first == second


def test_a_different_seed_shuffles_the_folds(tmp_path):
    table, _ = _table(tmp_path, noise=1.5)
    a = _read(fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                            seed=0, out_dir=tmp_path / "a", verbose=False))[0]
    b = _read(fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                            seed=99, out_dir=tmp_path / "b", verbose=False))[0]
    assert a["cv_rmse"] != b["cv_rmse"], \
        "the fold split is part of the estimate; a new seed must move it"


def test_out_of_fold_beats_in_sample_on_a_hard_problem(tmp_path):
    """The honesty check: with more predictors than signal, in-sample R2
    flatters and the cross-validated one does not."""
    table, _ = _table(tmp_path, n=40, p=5, noise=2.0, seed=21)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        verbose=False)
    row = _read(out)[0]
    assert float(row["train_r2"]) > float(row["cv_r2"])


def test_the_alpha_path_records_every_penalty_searched(tmp_path):
    table, _ = _table(tmp_path)
    grid = [0.1, 1.0, 10.0, 100.0]
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], alphas=grid,
                  verbose=False)
    path = _read(tmp_path / "ridge_alpha_path.csv")
    assert [float(r["alpha"]) for r in path] == grid
    chosen = float(_read(tmp_path / "ridge_cv_metrics.csv")[0]["alpha"])
    best = min(path, key=lambda r: float(r["cv_rmse"]))
    assert chosen == pytest.approx(float(best["alpha"]))


def test_per_table_sets_produce_a_comparison_table(tmp_path):
    """The headline use: one row per (feature set, outcome), so the sets can
    be ranked. The set that carries the signal has to win."""
    rng = np.random.default_rng(13)
    n = 200
    signal = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 2))
    y = signal @ np.array([1.5, -1.0]) + rng.normal(scale=0.3, size=n)
    header = ["text_id", "good.a", "good.b", "junk.a", "junk.b", "outcome"]
    rows = [[f"d{i}", f"{signal[i,0]:.8f}", f"{signal[i,1]:.8f}",
             f"{noise[i,0]:.8f}", f"{noise[i,1]:.8f}", f"{y[i]:.8f}"]
            for i in range(n)]
    table = _write(tmp_path / "analysis_table.csv", header, rows)
    (tmp_path / "analysis_table_sets.json").write_text(json.dumps({
        "key": ["text_id"], "metadata": ["outcome"],
        "sets": {"good": ["good.a", "good.b"], "junk": ["junk.a", "junk.b"]},
    }), encoding="utf-8")

    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        feature_sets="per_table", verbose=False)
    by_set = {r["feature_set"]: r for r in _read(out)}
    assert set(by_set) == {"good", "junk"}
    assert float(by_set["good"]["cv_r2"]) > 0.8
    assert float(by_set["junk"]["cv_r2"]) < 0.2
    assert (tmp_path / "models" / "ridge__good__outcome.json").is_file()
    assert (tmp_path / "models" / "ridge__junk__outcome.json").is_file()


# ---------------------------------------------------------------------------
# the model file, and applying it
# ---------------------------------------------------------------------------

def test_the_model_records_the_instrument_and_nothing_runtime(tmp_path):
    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], n_folds=4,
                  seed=3, verbose=False)
    model = json.loads((tmp_path / "models" / "ridge__all__outcome.json")
                       .read_text(encoding="utf-8"))
    assert model["kind"] == "taters-ridge-model"
    assert model["format"] == RIDGE_MODEL_FORMAT
    assert model["predictors"] == [f"f{i}" for i in range(5)]
    assert model["cv"]["n_folds"] == 4 and model["cv"]["seed"] == 3
    block = model["outcomes"]["outcome"]
    assert len(block["mu"]) == len(block["sigma"]) == len(block["coef"])
    assert "device" not in model and "workers" not in model


def test_every_outcome_gets_a_model_file_of_its_own(tmp_path):
    """
    Predict five things and you have five models, not one. A run that fitted
    five personality traits handed back a single `ridge__all.json` and the
    researcher reasonably asked where the other four were -- a model is "the
    thing that predicts X", so each X gets a file, a library entry and a name.
    """
    import csv

    from taters.helpers.model_spec import describe

    table, _ = _table(tmp_path)
    rows = _read(table)
    header = list(rows[0]) + ["second"]
    with (tmp_path / "two.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            # a second outcome that is the first one mirrored, so both fit
            w.writerow([r[c] for c in rows[0]] + [f"{-float(r['outcome']):.10f}"])

    fit_ridge_csv(table_csv=tmp_path / "two.csv", outcome_cols=["outcome", "second"],
                  out_dir=tmp_path / "fit", verbose=False)
    models = sorted(p.name for p in (tmp_path / "fit" / "models").glob("*.json"))
    assert models == ["ridge__all__outcome.json", "ridge__all__second.json"]

    for name, outcome in (("outcome", "outcome"), ("second", "second")):
        doc = json.loads((tmp_path / "fit" / "models" / f"ridge__all__{name}.json")
                         .read_text(encoding="utf-8"))
        assert list(doc["outcomes"]) == [outcome], "a file holds its own outcome only"
        assert doc["predictors"] == [f"f{i}" for i in range(5)]
        assert describe(tmp_path / "fit" / "models" / f"ridge__all__{name}.json").outputs \
            == (outcome,)

    # and each one scores on its own, writing only its own column
    scored = _read(apply_ridge_csv(
        model_json=tmp_path / "fit" / "models" / "ridge__all__second.json",
        input_csv=tmp_path / "two.csv", out_csv=tmp_path / "s.csv", verbose=False))
    assert "pred_second" in scored[0] and "pred_outcome" not in scored[0]


def test_applying_to_the_training_table_reproduces_the_fit_byte_for_byte(tmp_path):
    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    from_fit = _read(tmp_path / "ridge_predictions__all.csv")

    scored = apply_ridge_csv(model_json=tmp_path / "models" / "ridge__all__outcome.json",
                             input_csv=table, verbose=False)
    from_apply = {r["text_id"]: r for r in _read(scored)}
    for row in from_fit:
        assert from_apply[row["text_id"]]["pred_outcome"] == row["pred_outcome"]


def test_apply_matches_predictors_by_name_not_position(tmp_path):
    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)

    # the same data, but with the columns shuffled and an extra one riding along
    rows = _read(table)
    header = ["f3", "note", "f0", "text_id", "f4", "f1", "f2"]
    shuffled = _write(tmp_path / "other.csv", header,
                      [[r["f3"], "hello", r["f0"], r["text_id"], r["f4"],
                        r["f1"], r["f2"]] for r in rows])
    scored = apply_ridge_csv(
        model_json=tmp_path / "models" / "ridge__all__outcome.json",
        input_csv=shuffled, verbose=False)
    original = {r["text_id"]: r["pred_outcome"]
                for r in _read(tmp_path / "ridge_predictions__all.csv")}
    for row in _read(scored):
        assert row["pred_outcome"] == original[row["text_id"]]
        assert row["note"] == "hello", "unknown columns pass through"


def test_a_missing_predictor_is_refused_by_name(tmp_path):
    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    rows = _read(table)
    short = _write(tmp_path / "short.csv", ["text_id", "f0", "f1"],
                   [[r["text_id"], r["f0"], r["f1"]] for r in rows])
    with pytest.raises(ValueError, match="f2"):
        apply_ridge_csv(model_json=tmp_path / "models" / "ridge__all__outcome.json",
                        input_csv=short, verbose=False)


def test_a_row_missing_a_predictor_is_left_blank_not_guessed(tmp_path):
    table, _ = _table(tmp_path, n=60)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    rows = _read(table)
    rows[0]["f2"] = ""
    holed = _write(tmp_path / "holed.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])
    scored = _read(apply_ridge_csv(
        model_json=tmp_path / "models" / "ridge__all__outcome.json",
        input_csv=holed, verbose=False))
    assert scored[0]["pred_outcome"] == ""
    assert scored[1]["pred_outcome"] != "", "one hole does not stop the rest"


# ---------------------------------------------------------------------------
# the loader's refusals
# ---------------------------------------------------------------------------

def _model(tmp_path):
    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    return tmp_path / "models" / "ridge__all__outcome.json"


def test_a_model_from_the_future_is_refused(tmp_path):
    path = _model(tmp_path)
    model = json.loads(path.read_text(encoding="utf-8"))
    model["format"] = RIDGE_MODEL_FORMAT + 1
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="newer Taters"):
        _load_model(path)


def test_another_kind_of_model_is_named_not_guessed_at(tmp_path):
    path = _model(tmp_path)
    model = json.loads(path.read_text(encoding="utf-8"))
    model["kind"] = "taters-pca-model"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="taters-pca-model"):
        _load_model(path)

    path.write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a Taters model file"):
        _load_model(path)


def test_a_damaged_model_says_what_is_wrong_and_what_to_do(tmp_path):
    path = _model(tmp_path)
    good = json.loads(path.read_text(encoding="utf-8"))

    def write(mutate):
        model = json.loads(json.dumps(good))
        mutate(model)
        path.write_text(json.dumps(model), encoding="utf-8")

    write(lambda m: m["outcomes"]["outcome"].__setitem__(
        "coef", m["outcomes"]["outcome"]["coef"][:-1]))
    with pytest.raises(ValueError, match="mismatched sizes"):
        _load_model(path)

    write(lambda m: m["outcomes"]["outcome"].__setitem__("kept", [99]))
    with pytest.raises(ValueError, match="do not exist"):
        _load_model(path)

    write(lambda m: m.__setitem__("predictors", []))
    with pytest.raises(ValueError, match="names no predictors"):
        _load_model(path)

    write(lambda m: m["outcomes"]["outcome"].__setitem__(
        "sigma", [0.0] * len(good["outcomes"]["outcome"]["sigma"])))
    with pytest.raises(ValueError, match="zero scale"):
        _load_model(path)

    # every refusal should tell you what to do next
    write(lambda m: m.__setitem__("outcomes", {}))
    with pytest.raises(ValueError, match="Re-run the prediction step"):
        _load_model(path)


def test_a_folder_of_several_models_refuses_to_pick_one(tmp_path):
    path = _model(tmp_path)
    (path.parent / "ridge__other.json").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8")
    table, _ = _table(tmp_path, name="other_table.csv")
    with pytest.raises(ValueError, match="can only use one"):
        apply_ridge_csv(model_json=path.parent, input_csv=table,
                        verbose=False)


# ---------------------------------------------------------------------------
# refusals at fit time
# ---------------------------------------------------------------------------

def test_too_few_rows_for_the_folds_is_refused_with_the_numbers(tmp_path):
    table, _ = _table(tmp_path, n=8)
    with pytest.raises(ValueError, match="needs at least"):
        fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], n_folds=5,
                      verbose=False)


def test_a_zero_or_negative_alpha_is_refused(tmp_path):
    table, _ = _table(tmp_path)
    with pytest.raises(ValueError, match="greater than zero"):
        fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                      alphas=[0.0, 1.0], verbose=False)


def test_constant_predictors_are_dropped_and_counted(tmp_path):
    table, _ = _table(tmp_path, n=60)
    rows = _read(table)
    header = list(rows[0]) + ["flat"]
    padded = _write(tmp_path / "padded.csv", header,
                    [[r[c] for c in rows[0]] + ["1"] for r in rows])
    out = fit_ridge_csv(table_csv=padded, outcome_cols=["outcome"],
                        out_dir=tmp_path / "out", verbose=False)
    row = _read(out)[0]
    assert row["n_dropped_constant"] == "1"
    assert row["n_predictors"] == "5"


def test_missing_outcomes_are_refused_naming_the_options(tmp_path):
    table, _ = _table(tmp_path)
    with pytest.raises(ValueError, match="It has 7"):
        fit_ridge_csv(table_csv=table, outcome_cols=["nope"], verbose=False)


def test_the_report_section_names_the_best_set(tmp_path):
    table, _ = _table(tmp_path, n=200, noise=0.2)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    text = (tmp_path / "_sections" / "40-ridge.md").read_text(encoding="utf-8")
    assert text.startswith("## Prediction (cross-validated ridge)")
    assert "out-of-fold" in text
    assert "`outcome`" in text


def test_a_row_missing_only_a_dropped_predictor_still_scores(tmp_path):
    """A predictor dropped as constant is not read at scoring time, so a row
    with a hole in *that* column is still perfectly scorable -- blanking it
    would throw away a usable row for a column the model ignores."""
    table, _ = _table(tmp_path, n=60)
    rows = _read(table)
    header = list(rows[0]) + ["flat"]
    padded = _write(tmp_path / "padded.csv", header,
                    [[r[c] for c in rows[0]] + ["1"] for r in rows])
    fit_ridge_csv(table_csv=padded, outcome_cols=["outcome"], verbose=False)

    holed_rows = [[r[c] for c in rows[0]] + ["1"] for r in rows]
    holed_rows[0][-1] = ""          # the dropped column, missing
    holed = _write(tmp_path / "holed.csv", header, holed_rows)
    scored = _read(apply_ridge_csv(
        model_json=tmp_path / "models" / "ridge__all__outcome.json",
        input_csv=holed, verbose=False))
    assert scored[0]["pred_outcome"] not in ("", None)


def test_feature_sets_fitted_on_different_rows_are_flagged(tmp_path):
    """Real-run finding: a windowed measure undefined for short texts takes
    every row it is missing from out of its own model, so one set can be
    fitted on a fifth of the data as another. Comparing those R2s without
    knowing that is exactly the mistake the report has to prevent."""
    rng = np.random.default_rng(31)
    n = 120
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = a + rng.normal(scale=0.5, size=n)
    header = ["text_id", "full.a", "full.b", "holey.a", "holey.b", "outcome"]
    rows = []
    for i in range(n):
        # missing for 40% of rows: enough that the two sets get fitted on
        # different samples, but under `max_missing`, so the columns stay and
        # it's the ROWS that go. above that threshold the sparse columns get
        # dropped instead, which the next test covers
        holey = (["", ""] if i % 5 < 2
                 else [f"{b[i]:.6f}", f"{b[i]*2:.6f}"])
        rows.append([f"d{i}", f"{a[i]:.6f}", f"{b[i]:.6f}", *holey,
                     f"{y[i]:.6f}"])
    table = _write(tmp_path / "analysis_table.csv", header, rows)
    (tmp_path / "analysis_table_sets.json").write_text(json.dumps({
        "key": ["text_id"], "metadata": ["outcome"],
        "sets": {"full": ["full.a", "full.b"],
                 "holey": ["holey.a", "holey.b"]},
    }), encoding="utf-8")

    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        feature_sets="per_table", verbose=False)
    used = {r["feature_set"]: int(r["n_used"]) for r in _read(out)}
    # "one set on a fifth of the data" isn't reachable by default any more, and
    # that's the whole point of `max_missing`: keeping a column at all now means
    # keeping at least half the rows. what's still reachable (and still has to
    # get flagged) is two sets fitted on samples that differ enough that
    # comparing their R2 means comparing two studies
    assert used["holey"] < used["full"], "the fixture lost its point"
    assert used["full"] - used["holey"] >= 0.3 * used["full"]

    section = (tmp_path / "_sections" / "40-ridge.md").read_text("utf-8")
    assert "different numbers of rows" in section
    assert "not directly comparable" in section


def test_equal_sized_sets_are_not_flagged(tmp_path):
    table, _ = _table(tmp_path, n=120)
    (tmp_path / "analysis_table_sets.json").write_text(json.dumps({
        "key": ["text_id"], "metadata": ["outcome"],
        "sets": {"one": ["f0", "f1"], "two": ["f2", "f3"]},
    }), encoding="utf-8")
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                  feature_sets="per_table", verbose=False)
    section = (tmp_path / "_sections" / "40-ridge.md").read_text("utf-8")
    assert "different numbers of rows" not in section


# ---------------------------------------------------------------------------
# the metric set: several of them, because they disagree in useful ways
# ---------------------------------------------------------------------------

def test_the_metrics_table_reports_every_statistic_worth_reading(tmp_path):
    table, _ = _table(tmp_path, n=200, noise=1.2, seed=7)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        rounding=10, verbose=False)
    row = _read(out)[0]

    for column in ("cv_r2", "cv_r2_folds", "cv_r2_folds_se",
                   "cv_r", "cv_r_p", "cv_r_folds", "cv_rho", "cv_rho_p",
                   "cv_mse", "cv_rmse", "cv_mae", "baseline_mae"):
        assert row[column] != "", f"{column} is missing"

    # internal consistency that you can check by eye
    assert float(row["cv_mse"]) == pytest.approx(float(row["cv_rmse"]) ** 2,
                                                 rel=1e-6)
    # the p-value has to be the p-value OF the r sitting beside it: the standard
    # t transform on n - 2 degrees of freedom
    from scipy.stats import t as t_dist
    r, n = float(row["cv_r"]), int(row["n_used"])
    expected = float(2 * t_dist.sf(abs(r) * ((n - 2) / (1 - r * r)) ** 0.5,
                                   n - 2))
    assert float(row["cv_r_p"]) == pytest.approx(expected, rel=1e-6)


def test_the_baseline_is_what_predicting_the_mean_would_have_cost(tmp_path):
    """Every other number should be read against this floor, so it has to be
    exactly that floor: the mean absolute error of predicting one value for
    everybody."""
    table, _ = _table(tmp_path, n=120, seed=9)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        rounding=10, verbose=False)
    row = _read(out)[0]

    y = np.array([float(r["outcome"]) for r in _read(table)])
    assert float(row["baseline_mae"]) == pytest.approx(
        float(np.abs(y - y.mean()).mean()), rel=1e-9)
    assert float(row["cv_mae"]) < float(row["baseline_mae"]), \
        "this fixture is predictable, so the model should beat the mean"


def test_variance_explained_can_go_negative_and_says_so(tmp_path):
    """R-squared here is 1 - SSE/SST, not a squared correlation. The
    difference matters exactly when it is worst: a model that does worse
    than predicting the mean has a NEGATIVE R-squared, and squaring r would
    report that disaster as a mild positive."""
    rng = np.random.default_rng(4)
    n = 40
    # predictors that carry nothing, and not many rows, so cross-validation is
    # going to score this below the mean
    header = ["text_id"] + [f"f{i}" for i in range(12)] + ["outcome"]
    rows = [[f"d{i}"] + [f"{v:.8f}" for v in rng.normal(size=12)]
            + [f"{rng.normal():.8f}"] for i in range(n)]
    table = _write(tmp_path / "analysis_table.csv", header, rows)

    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        alphas=[1e-3, 1e-2], rounding=10, verbose=False)
    row = _read(out)[0]
    assert float(row["cv_r2"]) < 0, \
        "a useless model must be allowed to report as useless"
    # ...and the correlation tells a different story, which is why we report
    # both
    assert row["cv_r"] != ""


def test_per_fold_and_pooled_are_reported_separately(tmp_path):
    """They answer different questions -- a fold's R-squared is measured
    against that fold's own mean -- so they are different numbers, and the
    spread across folds is what says whether either can be trusted."""
    table, _ = _table(tmp_path, n=150, noise=1.5, seed=21)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        n_folds=5, rounding=10, verbose=False)
    row = _read(out)[0]

    assert float(row["cv_r2"]) != float(row["cv_r2_folds"])
    assert float(row["cv_r2_folds_se"]) > 0
    # the standard error of a mean, not the spread of the values: with five
    # folds it's smaller than their standard deviation by sqrt(5)
    assert float(row["cv_r2_folds_se"]) < 0.5


def test_the_saved_model_carries_the_same_metrics(tmp_path):
    table, _ = _table(tmp_path, n=120)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    model = json.loads((tmp_path / "models" / "ridge__all__outcome.json")
                       .read_text(encoding="utf-8"))
    cv = model["outcomes"]["outcome"]["cv"]
    for key in ("r2", "r2_folds", "r2_folds_se", "r", "r_p", "rho", "rho_p",
                "mse", "rmse", "mae", "baseline_mae"):
        assert key in cv, f"the model does not record {key}"


def test_every_fold_reports_its_own_score(tmp_path):
    """"R-squared .20 +- .13" is a summary a reader has to trust. The five
    numbers behind it are what lets them see that one fold carried the
    result -- and they make the summary checkable."""
    table, _ = _table(tmp_path, n=150, noise=1.5, seed=21)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        n_folds=5, rounding=10, verbose=False)
    folds = _read(tmp_path / "ridge_folds.csv")
    assert len(folds) == 5
    assert [r["fold"] for r in folds] == ["1", "2", "3", "4", "5"]
    assert sum(int(r["n"]) for r in folds) == 150, \
        "every row is held out exactly once"

    # the summary in the metrics table has to be the mean of these, and its
    # "+-" has to be the standard error of that mean rather than their spread.
    # the two differ by sqrt(k), and that's the whole distinction
    row = _read(out)[0]
    values = np.array([float(r["r2"]) for r in folds])
    # abs, not rel: the table gets written to `rounding` decimal places, so the
    # tolerance has to sit above the written precision
    assert float(row["cv_r2_folds"]) == pytest.approx(values.mean(), abs=1e-9)
    assert float(row["cv_r2_folds_se"]) == pytest.approx(
        values.std(ddof=1) / np.sqrt(values.size), abs=1e-9)
    assert float(row["cv_r_folds"]) == pytest.approx(
        np.array([float(r["r"]) for r in folds]).mean(), abs=1e-9)


# ---------------------------------------------------------------------------
# controls: what the language adds over age and gender
# ---------------------------------------------------------------------------

def _with_controls(tmp_path, n=200, seed=13, language_matters=True):
    """Age predicts the outcome; the language features add a little more
    (or nothing, when asked)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n)
    gender = np.array(["f" if i % 2 else "m" for i in range(n)])
    feats = rng.normal(size=(n, 4))
    y = (0.08 * age + 0.7 * (gender == "f")
         + (1.2 * feats[:, 0] if language_matters else 0.0)
         + rng.normal(0, 1.0, n))
    header = (["text_id", "age", "gender"]
              + [f"f{i}" for i in range(4)] + ["outcome"])
    rows = [[f"d{i}", f"{age[i]:.6f}", gender[i]]
            + [f"{v:.6f}" for v in feats[i]] + [f"{y[i]:.6f}"]
            for i in range(n)]
    return _write(tmp_path / "analysis_table.csv", header, rows)


def test_controls_produce_three_comparable_models(tmp_path):
    """The controls alone, the language alone, and both -- and the reason
    all three are worth having is the difference between the first and the
    last."""
    table = _with_controls(tmp_path)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        control_cols=["age", "gender"], rounding=10,
                        verbose=False)
    rows = {r["model"]: r for r in _read(out)}
    assert set(rows) == {"controls", "language", "controls+language"}

    # fitted on ONE sample, so the difference between them can't be part
    # sample and part language
    assert len({r["n_used"] for r in rows.values()}) == 1

    controls_only = float(rows["controls"]["cv_r2"])
    combined = float(rows["controls+language"]["cv_r2"])
    assert combined > controls_only, "the language really does add here"
    assert float(rows["controls+language"]["delta_r2_over_controls"]) == \
        pytest.approx(combined - controls_only, abs=1e-9)
    # the other two rows have nothing to be incremental over
    assert rows["controls"]["delta_r2_over_controls"] == ""
    assert rows["language"]["delta_r2_over_controls"] == ""
    assert rows["controls"]["controls"] == "age|gender"
    assert rows["language"]["controls"] == ""


def test_language_that_adds_nothing_reports_adding_nothing(tmp_path):
    """The honest negative: features unrelated to the outcome must not be
    able to show a gain over the controls."""
    table = _with_controls(tmp_path, language_matters=False, seed=27)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        control_cols=["age", "gender"], rounding=10,
                        verbose=False)
    rows = {r["model"]: r for r in _read(out)}
    delta = float(rows["controls+language"]["delta_r2_over_controls"])
    assert delta < 0.02, f"noise appeared to add {delta:.3f}"


def test_a_control_is_never_also_a_predictor(tmp_path):
    table = _with_controls(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                  control_cols=["age", "gender"], rounding=10, verbose=False)
    coefs = _read(tmp_path / "ridge_coefficients.csv")
    language = {r["predictor"] for r in coefs if r["model"] == "language"}
    assert language <= {f"f{i}" for i in range(4)}
    assert "age" not in language and "gender=m" not in language
    # ...but the controls-only model is made of exactly the controls
    only = {r["predictor"] for r in coefs if r["model"] == "controls"}
    assert only == {"age", "gender=m"}


def test_the_report_states_what_the_language_added(tmp_path):
    table = _with_controls(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                  control_cols=["age", "gender"], verbose=False)
    section = (tmp_path / "_sections" / "40-ridge.md").read_text("utf-8")
    assert "What the language added" in section
    assert "gender: levels compared against 'f'" in section


def test_asking_for_each_control_alone_fits_each_control_alone(tmp_path):
    table = _with_controls(tmp_path)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        control_cols=["age", "gender"],
                        control_combos="each", rounding=10, verbose=False)
    combos = {r["controls"] for r in _read(out)}
    assert combos == {"", "age", "gender", "age|gender"}


def test_without_controls_nothing_changes_but_the_label(tmp_path):
    """The three-model machinery must not alter a run that has no controls:
    one row, labeled `language`."""
    table, _ = _table(tmp_path, n=120)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        rounding=10, verbose=False)
    rows = _read(out)
    assert len(rows) == 1
    assert rows[0]["model"] == "language" and rows[0]["controls"] == ""


def test_every_model_is_fitted_on_the_same_rows(tmp_path):
    """The comparison is the point, so the sample has to be held equal.
    Fitting each model on whatever rows it happened to have would make the
    difference between them part sample and part language, with no way to
    tell which."""
    table = _with_controls(tmp_path, n=200)
    rows = _read(table)
    for r in rows[:40]:
        r["age"] = ""                      # the controls model loses these
    holed = _write(tmp_path / "holed.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])

    out = fit_ridge_csv(table_csv=holed, outcome_cols=["outcome"],
                        control_cols=["age", "gender"],
                        out_dir=tmp_path / "out", rounding=10, verbose=False)
    used = {r["model"]: int(r["n_used"]) for r in _read(out)}
    assert set(used.values()) == {160}, (
        "all three models share the complete-case sample, and it is the "
        "one the controls could be measured on")


def test_a_predictor_missing_for_most_rows_is_dropped_not_the_rows(tmp_path):
    """
    The failure this guards looked exactly like success.

    A cohesion run writes ``adjacent_overlap_2_*_para`` columns, which
    compare paragraphs two apart and so are blank for any text with fewer
    than three. In a real run 60 of 166 predictors were blank for 885 of 938
    rows; listwise deletion then took every one of those rows out, a ridge
    was fitted on **52 rows with 165 predictors**, and an R-squared was
    reported with nothing anywhere saying the sample had lost 94% of the
    data. Dropping the column instead keeps 904 of the rows.
    """
    rng = np.random.default_rng(77)
    n = 400
    good = rng.normal(size=n)
    y = good + rng.normal(scale=0.4, size=n)
    header = ["text_id", "good", "para_only", "outcome"]
    rows = []
    for i in range(n):
        # present for 5% of rows, the way a paragraph-pair measure is
        para = f"{rng.normal():.6f}" if i % 20 == 0 else ""
        rows.append([f"d{i}", f"{good[i]:.6f}", para, f"{y[i]:.6f}"])
    table = _write(tmp_path / "analysis_table.csv", header, rows)

    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        out_dir=tmp_path / "kept", verbose=False)
    row = _read(out)[0]
    assert row["n_used"] == "400", "the rows were dropped instead of the column"
    assert row["n_available"] == "400"
    assert row["n_dropped_sparse"] == "1"
    assert row["n_predictors"] == "1"

    # you can opt back in to the old behavior, and then the report says what
    # it cost you, since the scores now describe 20 texts and not 400
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        max_missing=1.0, out_dir=tmp_path / "all",
                        verbose=False)
    row = _read(out)[0]
    assert int(row["n_used"]) == 20
    assert row["n_available"] == "400"
    section = (tmp_path / "all" / "_sections" / "40-ridge.md").read_text(
        encoding="utf-8")
    assert "used 20 of 400 rows" in section


def test_the_coefficients_are_one_row_per_predictor_across_outcomes(tmp_path):
    """
    Long format put every (predictor, outcome) pair on its own line, so
    comparing what predicted one outcome against what predicted another
    meant reading 165 rows, then another 165, and holding them side by side
    in your head. Nobody does that.
    """
    rng = np.random.default_rng(4)
    n = 200
    a, b = rng.normal(size=n), rng.normal(size=n)
    header = ["text_id", "a", "b", "first", "second"]
    rows = [[f"d{i}", f"{a[i]:.6f}", f"{b[i]:.6f}",
             f"{a[i] * 2 + rng.normal(scale=0.3):.6f}",
             f"{b[i] * -3 + rng.normal(scale=0.3):.6f}"] for i in range(n)]
    table = _write(tmp_path / "analysis_table.csv", header, rows)
    fit_ridge_csv(table_csv=table, outcome_cols=["first", "second"],
                  out_dir=tmp_path / "s", verbose=False)

    rows_out = _read(tmp_path / "s" / "ridge_coefficients.csv")
    # ordered by the largest absolute coefficient across the outcomes, so the
    # predictors that did something sit at the top of the file instead of
    # wherever the alphabet put them. `b` weighs -3 on `second`, `a` weighs 2
    # on `first`, so `b` leads
    assert [r["predictor"] for r in rows_out] == ["b", "a"]
    # one column per outcome, on the same row as its predictor
    by_name = {r["predictor"]: r for r in rows_out}
    assert float(by_name["a"]["first"]) > 1.0
    assert float(by_name["b"]["second"]) < -1.0
    # feature set and model stay columns, so fitting several sets adds rows
    # rather than another block of columns
    assert {"feature_set", "model", "controls"} <= set(rows_out[0])


# ---------------------------------------------------------------------------
# what went unscored, and why (beside the scores, not just on the screen)
# ---------------------------------------------------------------------------

def test_unscored_rows_are_accounted_for_beside_the_predictions(tmp_path):
    """Half a real study went unscored because one measure is undefined
    for short texts, and only a verbose print said so. The accounting
    names the predictor and how often it was blank, in a file that exists
    exactly when something went unscored."""
    from taters.stats._fit_common import unscored_path

    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False)
    model = tmp_path / "models" / "ridge__all__outcome.json"

    rows = _read(table)
    for i, r in enumerate(rows):
        if i % 4 == 0:
            r["f1"] = ""
        if i % 10 == 0:
            r["f2"] = ""
    gappy = _write(tmp_path / "gappy.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])
    out = apply_ridge_csv(model_json=model, input_csv=gappy,
                          out_csv=tmp_path / "scores.csv", verbose=False)
    account = {(r["reason"], r["detail"]): int(r["rows"])
               for r in _read(unscored_path(out))}
    assert account[("unscored", "rows with a blank prediction")] == 36
    assert account[("blank predictor", "f1")] == 30
    assert account[("blank predictor", "f2")] == 12
    assert ("blank predictor", "f0") not in account

    # a complete table leaves no accounting behind, and clears out a stale one
    apply_ridge_csv(model_json=model, input_csv=table,
                    out_csv=tmp_path / "scores.csv", overwrite_existing=True,
                    verbose=False)
    assert not unscored_path(out).exists()


def test_predictors_with_gaps_are_named_in_the_report_at_fit_time(tmp_path):
    """A predictor blank in some rows is kept (fewer than half are missing)
    and every row lacking it is unscorable -- here and in any later study.
    The report says which, and how often, while the researcher can still
    change the feature set."""
    table, _ = _table(tmp_path)
    rows = _read(table)
    for i, r in enumerate(rows):
        if i % 5 == 0:
            r["f3"] = ""
    gappy = _write(tmp_path / "gappy.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])
    fit_ridge_csv(table_csv=gappy, outcome_cols=["outcome"], verbose=False,
                  out_dir=tmp_path / "gappy_fit")
    section = next((tmp_path / "gappy_fit" / "_sections").glob("*ridge*")
                   ).read_text(encoding="utf-8")
    assert "### Predictors with gaps" in section
    assert "`f3` (20%)" in section
    assert "f0" not in section.split("### Predictors with gaps")[1].split("\n\n")[1]

    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False,
                  out_dir=tmp_path / "clean_fit")
    clean = next((tmp_path / "clean_fit" / "_sections").glob("*ridge*")
                 ).read_text(encoding="utf-8")
    assert "Predictors with gaps" not in clean


# ---------------------------------------------------------------------------
# a model fitted on components carries its reduction with it
# ---------------------------------------------------------------------------

def test_a_model_fitted_on_components_is_applied_to_raw_features(tmp_path):
    """Before: the model named `Component_1` as a predictor, no table
    produces a column by that name, and the model could never be applied.
    Now it carries the axes, names the raw features as its inputs, and
    rebuilds the components from them -- standardized against the fitting
    sample, which is what makes the scores comparable."""
    from taters.helpers.model_spec import describe
    from taters.stats._fit_common import unscored_path

    table, _ = _table(tmp_path)
    # full rank, so the components span the raw features exactly and the fit
    # is just a change of basis: whatever a raw fit predicts, this should too
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  pca_components=5, verbose=False)
    model = tmp_path / "models" / "ridge__all__outcome.json"
    doc = json.loads(model.read_text(encoding="utf-8"))
    # `all_` because a reduced column now always carries the name of the set
    # it was built from, even when that set is the only one in the run -- the
    # name is the only thing that travels with a column out of the run.
    assert doc["predictors"] == [f"all_Component_{i}" for i in range(1, 6)]
    assert doc["input_columns"] == [f"f{i}" for i in range(5)]
    assert doc["reduction"]["features"] == doc["input_columns"]
    assert set(doc["reduction"]["axes"]) == {"kept", "mu", "sigma",
                                             "projection", "n_components"}
    assert describe(model).inputs == tuple(doc["input_columns"])

    scored = _read(apply_ridge_csv(model_json=model, input_csv=table,
                                   out_csv=tmp_path / "s.csv", verbose=False))
    rows = _read(table)
    pred = np.array([float(r["pred_outcome"]) for r in scored])
    truth = np.array([float(r["outcome"]) for r in rows])
    assert len(pred) == len(truth) == 120
    assert np.corrcoef(pred, truth)[0, 1] > 0.95, "the components lost the signal"
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False,
                  out_dir=tmp_path / "raw")
    raw = _read(apply_ridge_csv(model_json=tmp_path / "raw" / "models" /
                                "ridge__all__outcome.json", input_csv=table,
                                out_csv=tmp_path / "raw.csv", verbose=False))
    same = np.array([float(r["pred_outcome"]) for r in raw])
    assert np.corrcoef(pred, same)[0, 1] > 0.99, \
        "a full-rank reduction must predict what the raw features predict"
    # the raw features are inputs, not identifiers, but the outcome column
    # (which the model neither reads nor predicts from) rides along as one
    assert list(scored[0]) == ["text_id", "outcome", "pred_outcome"]

    # a row missing a raw feature gets blank components and goes unscored, and
    # the accounting blames the raw feature, not the component
    rows[0]["f2"] = ""
    gappy = _write(tmp_path / "gappy.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows])
    out = apply_ridge_csv(model_json=model, input_csv=gappy,
                          out_csv=tmp_path / "g.csv", verbose=False)
    assert _read(out)[0]["pred_outcome"] == ""
    account = {(r["reason"], r["detail"]): int(r["rows"])
               for r in _read(unscored_path(out))}
    assert {d for r, d in account if r == "blank predictor"} == {"f2"}, \
        "only the feature that was blank is to blame, by its own name"


def test_the_scoring_step_supplies_raw_features_to_a_component_model(tmp_path):
    """`score_with_model` checks the model's inputs against the joined
    feature tables; for a component model those are the raw features."""
    from taters.score_model import score_with_model

    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  verbose=False)
    model = tmp_path / "models" / "ridge__all__outcome.json"
    rows = _read(table)
    features = _write(tmp_path / "features" / "f.csv",
                      ["text_id"] + [f"f{i}" for i in range(5)],
                      [[r["text_id"]] + [r[f"f{i}"] for i in range(5)]
                       for r in rows])
    scored = _read(score_with_model(model_json=model, feature_csvs=[features],
                                    out_csv=tmp_path / "scores.csv",
                                    allow_unrecorded=True, verbose=False))
    assert len(scored) == 120 and all(r["pred_outcome"] for r in scored)


# ---------------------------------------------------------------------------
# folds alike by default
# ---------------------------------------------------------------------------

def test_balanced_folds_share_the_outcome_and_the_rows_evenly():
    """Rows are ordered by the outcome and dealt round the folds, so every
    fold gets its share of low, middle and high values. A plain shuffle can
    put most of the high scores in one fold, and the fold-to-fold R-squared
    then says more about the deal than about the model."""
    from taters.stats._fit_common import balanced_folds, random_folds

    rng = np.random.default_rng(0)
    y = rng.normal(size=203)
    folds = balanced_folds(y, 5, seed=1)
    sizes = np.bincount(folds, minlength=5)
    assert sizes.max() - sizes.min() <= 1 and set(folds) == set(range(5))
    # dealt by rank: every fold's mean rank is within a couple of places of
    # every other's, whereas a shuffle's folds drift by tens of places
    ranks = np.argsort(np.argsort(y))
    balanced_ranks = np.ptp([ranks[folds == k].mean() for k in range(5)])
    plain = random_folds(len(y), 5, seed=1)
    plain_ranks = np.ptp([ranks[plain == k].mean() for k in range(5)])
    assert balanced_ranks <= 5, balanced_ranks
    assert balanced_ranks < plain_ranks, (balanced_ranks, plain_ranks)
    assert np.ptp([y[folds == k].mean() for k in range(5)]) \
        < np.ptp([y[plain == k].mean() for k in range(5)])
    # deterministic by seed, and different seeds deal differently
    assert np.array_equal(folds, balanced_folds(y, 5, seed=1))
    assert not np.array_equal(folds, balanced_folds(y, 5, seed=2))
    # too few rows to deal a full run, so we fall back to the plain shuffle
    assert np.array_equal(balanced_folds(y[:7], 5, seed=3), random_folds(7, 5, 3))


def test_stratify_off_is_the_plain_seeded_shuffle_as_before():
    from taters.stats._fit_common import random_folds
    from taters.stats.ridge import _fold_assignments

    y = np.arange(40, dtype=float)
    plain = _fold_assignments(40, 5, seed=4, y=y, stratify=False)
    rng = np.random.default_rng(4)
    order = rng.permutation(40)
    expected = np.empty(40, dtype=int)
    expected[order] = np.arange(40) % 5
    assert np.array_equal(plain, expected)
    assert np.array_equal(plain, random_folds(40, 5, 4))
    assert not np.array_equal(plain, _fold_assignments(40, 5, seed=4, y=y))


def test_the_fold_choice_is_recorded_and_reported(tmp_path):
    table, _ = _table(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False,
                  out_dir=tmp_path / "balanced")
    doc = json.loads((tmp_path / "balanced" / "models" / "ridge__all__outcome.json")
                     .read_text(encoding="utf-8"))
    assert doc["cv"]["stratify"] is True
    section = next((tmp_path / "balanced" / "_sections").glob("*ridge*")).read_text()
    assert "folds balanced on the outcome" in section

    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], verbose=False,
                  stratify=False, out_dir=tmp_path / "plain")
    doc = json.loads((tmp_path / "plain" / "models" / "ridge__all__outcome.json")
                     .read_text(encoding="utf-8"))
    assert doc["cv"]["stratify"] is False
    section = next((tmp_path / "plain" / "_sections").glob("*ridge*")).read_text()
    assert "random folds" in section


# ---------------------------------------------------------------------------
# feature tables together: what does each one add?
# ---------------------------------------------------------------------------

def _three_tables(tmp_path, n=160, seed=21):
    """Three feature tables: two carry the signal (one strongly), one is
    noise -- so the combinations have a story to tell."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 2))
    b = rng.normal(size=(n, 2))
    junk = rng.normal(size=(n, 2))
    y = a @ np.array([1.5, -1.0]) + 0.6 * b[:, 0] + rng.normal(scale=0.3, size=n)
    # `junk.c` is blank for most texts: it counts as a feature the table
    # brought, but the sparse rule sets it aside as a predictor
    header = ["text_id", "dict.a", "dict.b", "mem.a", "mem.b", "junk.a", "junk.b",
              "junk.c", "outcome"]
    rows = [[f"d{i}"] + [f"{v:.8f}" for v in (*a[i], *b[i], *junk[i])]
            + [f"{rng.normal():.4f}" if i % 5 == 0 else ""] + [f"{y[i]:.8f}"]
            for i in range(n)]
    table = _write(tmp_path / "analysis_table.csv", header, rows)
    (tmp_path / "analysis_table_sets.json").write_text(json.dumps({
        "key": ["text_id"], "metadata": ["outcome"],
        "sets": {"dict": ["dict.a", "dict.b"], "mem": ["mem.a", "mem.b"],
                 "junk": ["junk.a", "junk.b", "junk.c"]},
    }), encoding="utf-8")
    return table


def test_together_fits_every_combination_of_the_tables_up_to_all(tmp_path):
    """
    "Is the matrix worth having once the dictionary is in?" is a comparison
    of two rows. Analyzed together, the tables are fitted each alone, in
    every combination, and all together -- named by their members -- with a
    model file for each, and the report ranks them.
    """
    table = _three_tables(tmp_path)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        out_dir=tmp_path / "s", verbose=False)
    by_set = {r["feature_set"]: r for r in _read(out)}
    assert set(by_set) == {"dict", "mem", "junk", "dict+mem", "dict+junk",
                           "mem+junk", "all"}
    assert float(by_set["dict+mem"]["cv_r2"]) > float(by_set["dict"]["cv_r2"]) \
        > float(by_set["junk"]["cv_r2"])
    assert (tmp_path / "s" / "models" / "ridge__dict-mem__outcome.json").is_file()
    assert (tmp_path / "s" / "models" / "ridge__all__outcome.json").is_file()
    # how much went into each row, at a glance: tables and their columns...
    assert [(by_set[s]["n_feature_sets"], by_set[s]["n_features"])
            for s in ("dict", "dict+mem", "all")] == [("1", "2"), ("2", "4"), ("3", "7")]
    # ...as brought. what the model actually used comes after the sparse rule
    assert by_set["all"]["n_predictors"] == "6" and by_set["all"]["n_dropped_sparse"] == "1"
    header = list(_read(out)[0].keys())
    assert header[:4] == ["feature_set", "n_feature_sets", "n_features", "outcome"]
    section = (tmp_path / "s" / "_sections" / "40-ridge.md").read_text(encoding="utf-8")
    assert "How the feature sets compare" in section
    assert "| `outcome` | dict+mem |" in section


def test_set_combos_can_be_each_and_all_or_off(tmp_path):
    table = _three_tables(tmp_path)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        set_combos="each_and_all", out_dir=tmp_path / "e",
                        verbose=False)
    assert {r["feature_set"] for r in _read(out)} == {"dict", "mem", "junk", "all"}
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        set_combos="none", out_dir=tmp_path / "n", verbose=False)
    assert {r["feature_set"] for r in _read(out)} == {"all"}


def test_every_combination_is_bounded_and_says_so_when_it_is_not_fitted():
    """Six tables would be sixty-three models per outcome; above five, each
    and all are fitted and the report says why."""
    from taters.stats._fit_common import combine_sets

    sets = {f"t{i}": [f"t{i}.x"] for i in range(6)}
    combined, _reds, notes = combine_sets(sets, {}, "subsets")
    assert set(combined) == {*sets, "all"}
    assert notes and "63 combinations" in notes[0]
    combined, _reds, notes = combine_sets({"a": ["a.x"], "b": ["b.x"]}, {}, "subsets")
    assert list(combined) == ["a", "b", "all"] and not notes
    combined, _reds, _n = combine_sets({"a": ["a.x"], "b": ["b.x"]}, {}, "none")
    assert combined == {"all": ["a.x", "b.x"]}
    with pytest.raises(ValueError, match="set_combos must be one of"):
        combine_sets({"a": ["a.x"]}, {}, "everything")


def test_a_combination_of_reduced_tables_carries_every_reduction_and_scores(tmp_path):
    """
    With a reduction on, each table is reduced on its own and a combination
    is fitted on both tables' components. Its model has to carry both
    reductions, and scoring the fitting table with it must reproduce the
    fit's own predictions -- otherwise the model could never be applied.
    """
    table = _three_tables(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  pca_components=1, set_combos="each_and_all",
                  out_dir=tmp_path / "p", verbose=False)
    model = json.loads((tmp_path / "p" / "models" / "ridge__all__outcome.json")
                       .read_text(encoding="utf-8"))
    assert len(model["reductions"]) == 3 and "reduction" not in model
    assert set(model["input_columns"]) == {"dict.a", "dict.b", "mem.a", "mem.b",
                                           "junk.a", "junk.b"}
    # (junk.c, blank for most texts, got set aside before the reduction)
    scored = apply_ridge_csv(model_json=tmp_path / "p" / "models" / "ridge__all__outcome.json",
                             input_csv=table, out_csv=tmp_path / "scored.csv",
                             verbose=False)
    fitted = {r["text_id"]: float(r["pred_outcome"])
              for r in _read(tmp_path / "p" / "ridge_predictions__all.csv")}
    again = {r["text_id"]: float(r["pred_outcome"]) for r in _read(scored)}
    assert fitted.keys() == again.keys()
    assert all(abs(fitted[k] - again[k]) < 1e-6 for k in fitted)


def test_n_feature_sets_counts_the_tables_even_when_they_are_not_combined(tmp_path):
    """`all` fitted as one joint set still came from three tables; a table
    on its own is one; a run with no sidecar has one set of its own."""
    from taters.stats._fit_common import members_of

    assert members_of(["all"], ["dict", "mem", "junk"])["all"] == ["dict", "mem", "junk"]
    assert members_of(["dict+mem", "junk"], ["dict", "mem", "junk"]) == {
        "dict+mem": ["dict", "mem"], "junk": ["junk"]}
    assert members_of(["all"], [])["all"] == ["all"]
    table = _three_tables(tmp_path)
    out = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                        set_combos="none", out_dir=tmp_path / "n", verbose=False)
    (row,) = _read(out)
    assert row["feature_set"] == "all" and row["n_feature_sets"] == "3"
    assert row["n_features"] == "7"
