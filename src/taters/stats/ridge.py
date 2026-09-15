"""
Ridge regression with cross-validation: predict an outcome from the features,
honestly, and keep the model.

Language features are many and collinear -- three readability indices are
three views of sentence length -- which is exactly the situation ordinary
least squares handles worst and ridge handles well: the penalty trades a
little bias for a large drop in variance, and it never has to invert a
singular matrix. What ridge does not do is choose its own penalty, so this
searches a grid of alphas by k-fold cross-validation and reports the
out-of-fold performance at the chosen one. In-sample R-squared is reported
too, next to it, because the gap between them is the whole story of whether
a model learned anything or memorized.

The arithmetic is plain numpy on purpose. One economy SVD per fold serves the
*entire* alpha grid -- coefficients for any alpha are
``V diag(s/(s^2+a)) U' y`` -- so a 33-point grid costs one decomposition
rather than 33 solves, and the numbers are exactly reproducible from the
stored model rather than depending on a solver's iteration count.

A fit is an instrument, not a result: the model file carries the predictor
names, the training centering and scaling, the chosen alpha and the grid it
came from, and the coefficients, so the same model can score a new dataset
later -- standardized against the *training* sample, which is what makes the
scores comparable. That is the same fit-once/apply-many discipline the MEM
topic model and PCA use here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.model_spec import slug
from ._fit_common import (balanced_folds, random_folds, comparison_lines,
                          load_model_doc, model_header, output_folders,
                          prepare_apply, prepare_fit, score_design, set_design,
                          write_predictions)
from ..helpers.model_spec import output_label
from ..helpers.progress import announce
from ._common import (MAX_MISSING, _write_csv, components_appendix, fmt,
                      gap_lines, gaps_section, name_a_few, numeric_column,
                      sample_warnings, write_section, reusable)
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

#: Bumped on any change that would make an older Taters misread a
#: model. Format 2 added `output_names`: a build that ignored it
#: would write `pred_age` where the researcher asked for
#: `pred_age_blogs`, so refusing the file outright is the only safe
#: way to be wrong about it.
#: A model from the future is refused rather than guessed at.
RIDGE_MODEL_FORMAT = 2

#: The default penalties searched: 33 points from 1e-3 to 1e5, evenly spaced
#: in log space. Wide because the right alpha depends on how many predictors
#: there are and how collinear they are, neither of which the caller should
#: have to reason about; cheap because the whole grid rides on one SVD.
_DEFAULT_ALPHA_COUNT = 33
_DEFAULT_ALPHA_LOG_LOW = -3.0
_DEFAULT_ALPHA_LOG_HIGH = 5.0

#: Fewest complete rows a fit will accept, relative to the fold count. Below
#: this a "cross-validated" number is theater: with 8 rows and 5 folds some
#: fold holds one row, and its R-squared is undefined.
_MIN_ROWS_PER_FOLD = 2
_MIN_ROWS = 10


def default_alphas() -> List[float]:
    """The default penalty grid, as a plain list (JSON-storable)."""
    import numpy as np

    return [float(a) for a in np.logspace(_DEFAULT_ALPHA_LOG_LOW,
                                          _DEFAULT_ALPHA_LOG_HIGH,
                                          _DEFAULT_ALPHA_COUNT)]


def _fold_assignments(n: int, n_folds: int, seed: int, y=None,
                      stratify: bool = True):
    """Which fold each row belongs to: balanced on the outcome when asked
    and the outcome is given (see `_fit_common.balanced_folds`), otherwise
    a seeded shuffle."""
    if stratify and y is not None:
        return balanced_folds(y, n_folds, seed)
    return random_folds(n, n_folds, seed)


def _svd_coefficients(x, y, alphas):
    """
    Ridge coefficients for every alpha at once, from one SVD.

    ``x`` and ``y`` must already be centered (and scaled, if wanted): the
    intercept is never penalized, and recovering it from the means afterwards
    is both cheaper and exact.
    """
    import numpy as np

    u, s, vt = np.linalg.svd(x, full_matrices=False)
    uty = u.T @ y
    out = np.empty((len(alphas), x.shape[1]), dtype=float)
    for i, alpha in enumerate(alphas):
        out[i] = vt.T @ (s / (s * s + alpha) * uty)
    return out


def _metrics(actual, predicted) -> dict:
    """
    How well these predictions did, by every measure worth reporting.

    Deliberately several, because they disagree in informative ways and a
    reader who is handed only one cannot tell which situation they are in:

    ``r2`` is the variance-explained sense -- ``1 - SSE/SST`` -- which can
    go negative when the model does worse than predicting the mean. That is
    a real and useful signal, and it is why this is not reported as a
    squared correlation: squaring r hides the sign, so a model that is
    *worse than useless* comes back looking mildly positive.

    ``r`` is the correlation between predicted and observed, with its
    p-value. It answers a different question than R-squared: a model whose
    predictions track the outcome perfectly but on the wrong scale has
    r = 1 and a dismal R-squared.

    ``rho`` is the same in ranks -- robust to the skew that outcome
    measures so often have, and the statistic to trust when a handful of
    extreme values are doing the work.

    ``mae`` is in the outcome's own units, which is the only one of these a
    non-statistician can interpret directly.
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    resid = actual - predicted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    out = {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mse": float((resid ** 2).mean()),
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(np.abs(resid).mean()),
        "r": float("nan"), "r_p": float("nan"),
        "rho": float("nan"), "rho_p": float("nan"),
    }
    # a constant column has no correlation to report, and if we ask scipy for
    # one anyway it raises instead of handing back NaN.
    if actual.size >= 3 and actual.std() > 0 and predicted.std() > 0:
        r, r_p = pearsonr(actual, predicted)
        rho, rho_p = spearmanr(actual, predicted)
        out.update({"r": float(r), "r_p": float(r_p),
                    "rho": float(rho), "rho_p": float(rho_p)})
    return out


def predict_rows(cells, kept, mu, sigma, coef, intercept, rounding: int):
    """
    Predictions for a block of rows -- the one scorer fit and apply share.

    ``cells`` is (n_rows, n_predictors) in the model's predictor order. A row
    with a missing value among the predictors it needs cannot be scored: the
    missing value carries through the arithmetic as NaN -- deliberately, not
    incidentally -- and the row comes back as ``None`` for the caller to
    write as a blank cell. A prediction built from a filled-in guess would
    look exactly like a real one in the output.

    Only the predictors the model actually kept are read, so a row missing a
    predictor that was dropped as constant still scores.
    """
    import numpy as np

    x = np.asarray(cells, dtype=float)[:, kept]
    predicted = ((x - mu) / sigma) @ coef + intercept
    return [None if np.isnan(v) else round(float(v), rounding)
            for v in predicted]


def _cv_ridge(x, y, *, names, grid, n_folds: int, seed: int, zscore: bool,
              stratify: bool = True):
    """
    Cross-validate a ridge over one design matrix, and refit the winner.

    Everything that learns from the data -- the centering, the scaling, the
    coefficients -- is fitted inside each fold and applied to the held-out
    rows, never the other way round. Standardizing the whole sample first
    would let the test rows influence their own predictions, which inflates
    every number below by an amount nobody can estimate afterwards.

"""
    import numpy as np

    n = int(x.shape[0])
    spread = x.std(axis=0)
    kept = [i for i, s in enumerate(spread) if s > 0]
    dropped = [names[i] for i, s in enumerate(spread) if s <= 0]
    if not kept:
        return None, dropped
    xk = x[:, kept]

    folds = _fold_assignments(n, n_folds, seed, y=y, stratify=stratify)
    oof = np.full((len(grid), n), np.nan)
    for fold in range(n_folds):
        test = folds == fold
        train = ~test
        x_tr, y_tr = xk[train], y[train]
        mu = x_tr.mean(axis=0)
        sigma = x_tr.std(axis=0) if zscore else np.ones(x_tr.shape[1])
        # a predictor that's constant *within a fold* would have us dividing
        # by zero. leaving it unscaled keeps the fold usable; the penalty
        # takes care of the rest.
        sigma = np.where(sigma > 0, sigma, 1.0)
        y_mean = y_tr.mean()
        coefs = _svd_coefficients((x_tr - mu) / sigma, y_tr - y_mean, grid)
        # (rows x alphas) comes out of the product, (alphas x rows) goes into
        # the store -- one row per penalty, so that afterwards we can score
        # the whole grid against y in one vectorized pass.
        oof[:, test] = (((xk[test] - mu) / sigma) @ coefs.T + y_mean).T

    rmse_by_alpha = np.sqrt(((oof - y) ** 2).mean(axis=1))
    # ties go to the larger penalty. between two models that predict equally
    # well out of fold, the more regularized one is the one more likely to
    # keep it up.
    best_i = int(np.max(np.flatnonzero(rmse_by_alpha == rmse_by_alpha.min())))
    alpha = grid[best_i]

    cv = _metrics(y, oof[best_i])
    per_fold = [_metrics(y[folds == f], oof[best_i][folds == f])
                for f in range(n_folds)]
    fold_r2 = np.array([m["r2"] for m in per_fold], dtype=float)
    fold_r = np.array([m["r"] for m in per_fold], dtype=float)
    usable_r2 = fold_r2[~np.isnan(fold_r2)]
    usable_r = fold_r[~np.isnan(fold_r)]
    cv["r2_folds"] = float(usable_r2.mean()) if usable_r2.size else float("nan")
    # the standard error of the mean across folds -- that's what "+-" means
    # in a results table, NOT the standard deviation of the folds.
    cv["r2_folds_se"] = (float(usable_r2.std(ddof=1) / np.sqrt(usable_r2.size))
                         if usable_r2.size > 1 else float("nan"))
    cv["r_folds"] = float(usable_r.mean()) if usable_r.size else float("nan")
    # what we'd have paid for just predicting the training mean for everybody.
    # this is the floor that every number above should be read against, in
    # the outcome's own units.
    cv["baseline_mae"] = float(np.abs(y - y.mean()).mean())

    # now for the model we actually keep: we refit on everything, at the
    # chosen penalty.
    design = xk
    mu = design.mean(axis=0)
    sigma = design.std(axis=0) if zscore else np.ones(design.shape[1])
    sigma = np.where(sigma > 0, sigma, 1.0)
    intercept = float(y.mean())
    coef = _svd_coefficients((design - mu) / sigma, y - intercept, [alpha])[0]
    train_pred = ((design - mu) / sigma) @ coef + intercept

    return {
        "n": n, "kept": kept, "alpha": alpha, "cv": cv, "per_fold": per_fold,
        "folds": folds, "rmse_by_alpha": [float(v) for v in rmse_by_alpha],
        "mu": [float(v) for v in mu], "sigma": [float(v) for v in sigma],
        "coef": [float(v) for v in coef], "intercept": intercept,
        "train_r2": _metrics(y, train_pred)["r2"],
    }, dropped


def _load_model(model_json: PathLike) -> dict:
    """
    Read a ridge model, refusing anything that is not one -- in words.

    The refusals are the point. A file that is the wrong kind of model, or
    was written by a newer Taters, or is structurally incoherent, must be
    turned away while the user is holding it and can act, not halfway through
    scoring a corpus. These messages are also what the library shows when
    someone imports a model file, so they are written for a person.
    """
    path, model, broken = load_model_doc(
        model_json, kind="taters-ridge-model", fmt=RIDGE_MODEL_FORMAT,
        noun="ridge prediction model", step="prediction")
    predictors = model["predictors"]
    outcomes = model["outcomes"]
    for name, block in outcomes.items():
        if not isinstance(block, dict):
            raise broken(f"the entry for {name!r} is not a model")
        kept = block.get("kept")
        if not isinstance(kept, list) or not kept:
            raise broken(f"{name!r} keeps no predictors")
        if max(kept) >= len(predictors) or min(kept) < 0:
            raise broken(f"{name!r} points at predictors that do not exist")
        sizes = {len(kept), len(block.get("mu") or []),
                 len(block.get("sigma") or []), len(block.get("coef") or [])}
        if len(sizes) != 1:
            raise broken(
                f"{name!r} has mismatched sizes for its predictors, means, "
                f"scales and coefficients")
        if block.get("intercept") is None:
            raise broken(f"{name!r} has no intercept")
        if any(float(s) == 0.0 for s in block.get("sigma") or []):
            raise broken(f"{name!r} records a zero scale, which cannot divide")
    return model


def fit_ridge_csv(
    *,
    table_csv: PathLike,
    outcome_cols: Sequence[str],
    feature_sets=None,
    control_cols: Sequence[str] = (),
    categorical_controls: Sequence[str] = (),
    control_combos: str = "none_and_all",
    set_combos: Literal["none", "each_and_all", "subsets"] = "subsets",
    pca: object = "off",
    pca_components: int = 0,
    pca_retain: Literal["parallel", "kaiser"] = "parallel",
    pca_rotation: bool = True,
    split_col: Optional[str] = None,
    alphas: Optional[Sequence[float]] = None,
    n_folds: int = 5,
    stratify: bool = True,
    zscore: bool = True,
    max_missing: float = MAX_MISSING,
    seed: int = 0,
    out_dir: Optional[PathLike] = None,
    out_models_dir: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
) -> Path:
    """
    Fit a cross-validated ridge model per outcome, and keep it.

    Parameters
    ----------
    table_csv
        The assembled analysis table.
    outcome_cols
        The numeric column(s) to predict. Each gets its own model, its own
        chosen penalty and its own row of performance.
    feature_sets
        ``None`` to use every feature as one predictor set, ``"per_table"``
        to fit a separate model per source feature table -- which is how you
        answer "which feature set actually predicts this?", since the
        resulting rows sit side by side in one metrics table -- or an
        explicit ``{name: [columns]}``.
    control_cols
        Columns to fit alongside -- and instead of -- the language, so the
        language's own contribution can be read off. With controls the run
        fits three models per outcome on **one common sample**: the controls
        alone, the language alone, and both; the combined model's row
        carries ``delta_r2_over_controls``, which is what "language adds
        this much over age and gender" means. Continuous controls enter as
        themselves, categorical ones as indicators against a reference
        level.

        The shared sample is the part that matters: fitting each model on
        whatever rows it happened to have would make the difference between
        them part sample and part language, with no way to tell which.
    categorical_controls
        Which of ``control_cols`` to treat as categories despite looking
        numeric.
    set_combos : {"none", "each_and_all", "subsets"}, default="subsets"
        When the feature tables are analyzed together (``feature_sets`` not
        given), how to combine them: ``subsets`` fits every combination of
        the tables -- each alone, every pair, and so on up to all of them --
        so the report can say what each table adds to the others;
        ``each_and_all`` fits each alone and all together; ``none`` fits
        only all together. Every combination is fitted for up to five
        tables; above that, each and all. A combination is named by its
        members (``dictionary+readability``) in the tables and the models.
    control_combos : {"none_and_all", "each", "subsets"}, default="none_and_all"
        How many control sets to try. The default pair answers the usual
        question. ``"each"`` adds one model per single control, which is how
        you see which of them is doing the work; ``"subsets"`` runs all
        2^k of them, which is thorough and grows the table accordingly.
    pca : str or list of str, default="off"
        Analyze components instead of the raw measures. ``"off"`` uses the
        features as they are; ``"all"`` reduces every feature set; a list of
        set names reduces those and leaves the rest alone -- which is the
        case worth having, since a hundred dictionary categories are worth
        reducing and eight readability indices are not.

        Set per analysis, deliberately: raw variables read better in a
        correlation table, where each row is a measure you can name, while a
        ridge over four hundred collinear measures is what components are
        for. The loadings land beside these results and named after them,
        because a component means nothing without the table saying what
        loads on it.
    pca_components : int, default=0
        How many components to keep, when reducing. ``0`` decides by the
        Kaiser criterion -- a starting point, not an answer.
    pca_retain : {"parallel", "kaiser"}, default="parallel"
        How the component count is chosen when ``pca_components`` is 0:
        parallel analysis keeps a component while its eigenvalue beats what
        random data of the same size produce at that rank; the Kaiser rule
        keeps every eigenvalue above 1, which on a wide table is most of them.
    pca_rotation : bool, default=True
        Rotate the components (varimax) so each loads on a small cluster of
        features and is therefore nameable.
    split_col
        Fit separately for each value of this column, labeling each result
        with it. Set to ``"source_col"`` when a spreadsheet's text columns
        were measured separately: a participant then has one row per column,
        and cross-validating over them together leaks -- the same person
        appears in the training and the held-out fold, and the reported
        performance is not out-of-sample at all.
    alphas
        Penalties to search. Default: 33 points from 1e-3 to 1e5. Every value
        must be greater than zero; alpha = 0 is ordinary least squares, which
        is exactly what ridge exists to avoid on collinear features.
    n_folds
        Cross-validation folds. The reported performance is out-of-fold:
        each row is predicted by a model that never saw it.
    stratify
        Make the folds alike: each gets the same spread of the outcome and
        sizes within one of each other (rows are ordered by the outcome and
        dealt round the folds). Off, the folds are a plain seeded shuffle,
        which can put most of the high scores in one fold.
    zscore
        Standardize predictors on the training statistics. Coefficients are
        then per standard deviation and comparable to each other; with False
        they stay in the features' own units and are not.
    max_missing
        Set aside any predictor missing for more than this fraction of the
        rows, rather than letting it delete them. Some cohesion measures
        compare paragraphs two apart and so are blank for any text with
        fewer than three; left in, 60 such columns once deleted 885 of 938
        rows and the model was fitted on 52. Set it to 1.0 to keep every
        column and accept the row loss; what was dropped is always reported.
    seed
        Fixes the fold shuffle. Stored in the model, because a
        cross-validated number nobody can reproduce is a number nobody can
        check.
    out_dir, out_models_dir
        Where the tables and the model files go; both default beside the
        analysis table (models under ``models/``).
    overwrite_existing
        When False (default) and the metrics table exists, it is returned
        untouched.
    rounding
        Decimal places in the output tables.

    Returns
    -------
    Path
        ``ridge_cv_metrics.csv`` -- one row per (feature set, outcome), which
        is also the table to sort when comparing feature sets.

    Notes
    -----
    Rows missing any predictor or the outcome sit out that model's fit
    entirely (listwise), and ``n_used`` in the metrics table says how many
    were left. A predictor that never varies in the training rows is dropped
    by name rather than silently contributing nothing.
    """
    import numpy as np

    table_csv, folder, models_dir = output_folders(table_csv, out_dir,
                                                   out_models_dir)
    metrics_path = folder / "ridge_cv_metrics.csv"
    coef_path = folder / "ridge_coefficients.csv"
    path_path = folder / "ridge_alpha_path.csv"
    if reusable(metrics_path, table_csv, overwrite_existing=overwrite_existing,
                verbose=verbose, what="the ridge results"):
        if verbose:
            print(f"Ridge results already exist; returning existing file: "
                  f"{metrics_path}")
        return metrics_path

    outcome_cols = [str(c) for c in outcome_cols]
    inputs = prepare_fit(
        table_csv=table_csv, folder=folder, stem="ridge",
        outcome_cols=outcome_cols, feature_sets=feature_sets,
        control_cols=control_cols, categorical_controls=categorical_controls,
        control_combos=control_combos, pca=pca, pca_components=pca_components,
        pca_rotation=pca_rotation, pca_retain=pca_retain, split_col=split_col,
        set_combos=set_combos, alphas=alphas,
        n_folds=n_folds, max_missing=max_missing, encoding=encoding,
        rounding=rounding, verbose=verbose, on_progress=on_progress,
        zero_alpha_reason=("alpha = 0 is ordinary least squares, which is "
                           "what ridge exists to avoid here"))
    n_folds = int(n_folds)
    table, grid, sets, columns = (inputs.table, inputs.grid, inputs.sets,
                                  inputs.columns)
    pca_notes = inputs.pca_notes
    control_matrix, control_names, control_notes, control_spec = (
        inputs.control_matrix, inputs.control_names, inputs.control_notes,
        inputs.control_spec)
    control_sets, subsets, lead, total = (inputs.control_sets, inputs.subsets,
                                          inputs.lead, inputs.total)
    for c in outcome_cols:
        _refuse_categorical_outcome(c, table[c].tolist())
    outcomes = {c: numeric_column(table[c].tolist(), column=c)
                for c in outcome_cols}
    # (outcome, control-set, with-language) -> pooled R2, so that we can say
    # how much the combined model added over the controls alone.
    model_scores: dict = {}

    metrics_rows, coef_rows, path_rows, fold_rows = [], [], [], []
    best: dict = {}
    gaps: list = []
    done = 0

    for subset, subset_mask in subsets:
      prefix = [subset] if split_col else []
      for set_name, all_cols in sets.items():
        cols, sparse, design_all, design = set_design(
            set_name, all_cols, columns, subset_mask, max_missing,
            verbose=verbose, tag="ridge")
        gaps += gap_lines(set_name, cols, design)
        model_doc = model_header(
            kind="taters-ridge-model", fmt=RIDGE_MODEL_FORMAT,
            set_name=set_name, cols=cols, table_csv=table_csv,
            reduction=inputs.reductions.get(set_name),
            cv={"n_folds": n_folds, "seed": seed, "stratify": bool(stratify),
                "selection": "min_mean_out_of_fold_rmse_prefer_larger_alpha",
                "alphas": grid},
            zscore=zscore)
        predictions = {}
        # what each outcome's own model file needs beyond the shared header:
        # its coefficient block, and the predictors and controls it was fitted
        # with (they can differ between outcomes when a constant column gets
        # dropped for one and not another)
        per_outcome: dict = {}

        for outcome in outcome_cols:
            y_all = (outcomes[outcome] if subset_mask is None
                     else outcomes[outcome][subset_mask])
            ctrl = (control_matrix if subset_mask is None
                    else control_matrix[subset_mask])

            # ONE complete-case sample for every model of this outcome. if we
            # did listwise deletion over different column sets, the
            # controls-only model would get more rows than the combined one,
            # and the difference between their scores (the whole point of
            # running both) would be part sample and part language. no good.
            complete = ~np.isnan(design).any(axis=1) & ~np.isnan(y_all)
            if ctrl.shape[1]:
                complete = complete & ~np.isnan(ctrl).any(axis=1)
            y = y_all[complete]
            n_used = int(complete.sum())
            # how many rows COULD have been used, i.e. the outcome is present.
            # we print this next to n_used because the gap between the two is
            # a question anyone reading these results needs to be able to ask.
            n_available = int((~np.isnan(y_all)).sum())
            floor = max(_MIN_ROWS_PER_FOLD * n_folds, _MIN_ROWS)
            if n_used < floor:
                raise ValueError(
                    f"{outcome} on feature set {set_name!r}"
                    + (f" ({split_col} {subset!r})" if subset else "")
                    + f": only {n_used} row(s) have every predictor"
                    + (", every control" if ctrl.shape[1] else "")
                    + f" and the outcome, and {n_folds}-fold "
                    f"cross-validation needs at least {floor}. Use fewer "
                    f"features, fill the gaps, or fewer folds.")

            keep_best = None
            for control_set in control_sets:
                idx = [control_names.index(c) for c in control_names
                       if c.split("=")[0] in control_set]
                for with_language in (False, True):
                    if not with_language and not idx:
                        # "neither the controls nor the language" isn't a
                        # model, it's just the mean. skip it.
                        continue
                    done += 1
                    if on_progress is not None:
                        on_progress(done, total,
                                    f"fitting {outcome} on {set_name}")
                    blocks, names = [], []
                    if idx:
                        blocks.append(ctrl[complete][:, idx])
                        names += [control_names[i] for i in idx]
                    if with_language:
                        blocks.append(design[complete])
                        names += list(cols)
                    x = np.hstack(blocks)
                    label = ("controls+language" if idx and with_language
                             else "controls" if idx else "language")

                    fit, dropped = _cv_ridge(
                        x, y, names=names, grid=grid, n_folds=n_folds,
                        seed=seed, zscore=zscore, stratify=stratify)
                    if fit is None:
                        raise ValueError(
                            f"{outcome} on feature set {set_name!r}: every "
                            f"predictor is constant across the usable rows, "
                            f"so there is nothing to predict from.")
                    if dropped and verbose:
                        print(f"[ridge] {set_name}/{outcome} ({label}): "
                              f"dropping constant predictor(s) {dropped}")

                    cv = fit["cv"]
                    combo = "|".join(control_set)
                    # language's own contribution: same rows, same folds,
                    # same penalty grid -- we hold everything equal except
                    # whether the words were in the model.
                    delta = ""
                    if idx and with_language:
                        base = model_scores.get((outcome, combo, False))
                        if base is not None and base == base:
                            delta = fmt(cv["r2"] - base, rounding)
                    model_scores[(outcome, combo, with_language)] = cv["r2"]

                    metrics_rows.append(prefix + [
                        set_name, str(len(inputs.set_members.get(set_name, [set_name]))),
                        str(len(all_cols)), outcome, label, combo,
                        str(n_used), str(n_available),
                        str(len(sparse)), str(len(fit["kept"])),
                        str(len(dropped)), str(n_folds), str(bool(zscore)),
                        fmt(fit["alpha"], 8),
                        fmt(cv["r2"], rounding), delta,
                        fmt(cv["r2_folds"], rounding),
                        fmt(cv["r2_folds_se"], rounding),
                        fmt(cv["r"], rounding), fmt(cv["r_p"], rounding),
                        fmt(cv["r_folds"], rounding),
                        fmt(cv["rho"], rounding), fmt(cv["rho_p"], rounding),
                        fmt(cv["mse"], rounding), fmt(cv["rmse"], rounding),
                        fmt(cv["mae"], rounding),
                        fmt(cv["baseline_mae"], rounding),
                        fmt(fit["train_r2"], rounding),
                        fmt(fit["intercept"], rounding)])
                    for a, rmse in zip(grid, fit["rmse_by_alpha"],
                                       strict=True):
                        path_rows.append(prefix + [set_name, outcome, label,
                                                   fmt(a, 8),
                                                   fmt(float(rmse), rounding)])
                    for f, m in enumerate(fit["per_fold"]):
                        fold_rows.append(prefix + [
                            set_name, outcome, label, str(f + 1),
                            str(int((fit["folds"] == f).sum())),
                            fmt(m["r2"], rounding), fmt(m["r"], rounding),
                            fmt(m["rho"], rounding), fmt(m["mae"], rounding)])
                    coef = np.asarray(fit["coef"])
                    for i in np.argsort(-np.abs(coef)):
                        coef_rows.append(
                            (tuple(prefix) + (set_name, label, combo,
                                              names[fit["kept"][i]]),
                             outcome, float(coef[i])))

                    if cv["r2"] == cv["r2"] and cv["r2"] > best.get(
                            outcome, (-1e9,))[0]:
                        best[outcome] = (
                            cv["r2"],
                            f"{set_name} @ {subset}" if subset else set_name,
                            fit["alpha"])
                    # the model we keep (and apply later on) is the richest
                    # one: everything the run had to predict with.
                    if with_language:
                        keep_best = (fit, names, idx)

            fit, names, control_idx = keep_best
            # the recipe for the control columns, not just their names. the
            # next table this model scores will have `gender` as whatever
            # words the researcher typed, and which level is the reference
            # lives only here.
            per_outcome[outcome] = {
                "controls": [control_spec[i] for i in control_idx],
                "predictors": names,
                "block": {
                    "n_rows": n_used,
                    "alpha": fit["alpha"],
                    "kept": fit["kept"],
                    "mu": fit["mu"],
                    "sigma": fit["sigma"],
                    "coef": fit["coef"],
                    "intercept": fit["intercept"],
                    "cv": fit["cv"],
                },
            }
            # we score through the same helper that apply uses, over every
            # row of the table -- so that a fit and an apply on the same data
            # agree to the byte. that's what keeps the two from drifting
            # apart.
            full = np.hstack([ctrl[:, [control_names.index(c) for c in names
                                       if c in control_names]],
                              design]) if control_names else design
            predictions[outcome] = predict_rows(
                full, fit["kept"], np.asarray(fit["mu"]),
                np.asarray(fit["sigma"]), np.asarray(fit["coef"]),
                fit["intercept"], rounding)


        model_doc["subset"] = subset
        models_dir.mkdir(parents=True, exist_ok=True)
        stem = slug(set_name) + (f"__{slug(subset)}" if subset else "")
        # one model file per outcome. we used to bundle every outcome of a
        # feature set into one file, and somebody who'd predicted five
        # personality traits got one model back and asked where the other
        # four were. a model is "the thing that predicts X", so each X gets
        # its own file, its own name in the library, and its own import
        for outcome, parts in per_outcome.items():
            doc = dict(model_doc)
            doc["outcomes"] = {outcome: parts["block"]}
            doc["controls"] = parts["controls"]
            doc["predictors"] = parts["predictors"]
            model_path = models_dir / f"ridge__{stem}__{slug(outcome)}.json"
            with atomic_write(model_path, mode="w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1)
        _write_predictions(folder / f"ridge_predictions__{stem}.csv",
                           table if subset_mask is None else table[subset_mask],
                           outcome_cols, predictions, encoding)

    announce(on_progress, "writing the results")
    metrics_header = (
           lead + ["feature_set",
                   # how many tables the set is made of, and how many
                   # feature columns they brought along (before the sparse
                   # ones were set aside; `n_predictors` is what the model
                   # actually used, controls included) -- so that a row's
                   # score can be read against how much went into it.
                   "n_feature_sets", "n_features", "outcome",
                   # which model this row is: the controls alone, the
                   # language alone, or both -- and which controls, exactly.
                   "model", "controls",
                   "n_used", "n_available", "n_dropped_sparse",
                   "n_predictors",
                   "n_dropped_constant", "n_folds", "zscore", "alpha",
                   # variance explained, pooled and per-fold (the mean and
                   # the standard error of that mean), plus what the
                   # language added over the controls alone.
                   "cv_r2", "delta_r2_over_controls",
                   "cv_r2_folds", "cv_r2_folds_se",
                   # predicted-vs-observed association, three ways.
                   "cv_r", "cv_r_p", "cv_r_folds", "cv_rho", "cv_rho_p",
                   # error, and the floor we should read it against.
                   "cv_mse", "cv_rmse", "cv_mae", "baseline_mae",
                   "train_r2", "intercept"])
    _write_csv(metrics_path, metrics_header, metrics_rows, encoding)
    _write_coefficients(coef_path, coef_rows, outcome_cols, lead, rounding,
                        encoding)
    _write_csv(path_path,
           lead + ["feature_set", "outcome", "model", "alpha", "cv_rmse"],
           path_rows, encoding)
    _write_csv(folder / "ridge_folds.csv",
           lead + ["feature_set", "outcome", "model", "fold", "n", "r2", "r",
                   "rho", "mae"], fold_rows, encoding)

    section = _section_md(
        sets=sets, outcome_cols=outcome_cols, best=best, n_folds=n_folds,
        stratify=stratify,
        zscore=zscore, metrics_rows=metrics_rows,
        metrics_header=metrics_header, control_names=control_names,
        control_notes=control_notes, model_scores=model_scores,
        gap_notes=gaps_section(gaps, max_missing=max_missing),
        set_notes=inputs.set_notes)
    section += components_appendix(pca_notes)
    write_section(folder, "ridge", section)
    if verbose:
        print(f"[ridge] {len(metrics_rows)} model(s) -> {metrics_path}")
    return metrics_path


def apply_ridge_csv(
    *,
    model_json: PathLike,
    input_csv: PathLike,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
    id_cols: Optional[Sequence[str]] = None,
) -> Path:
    """
    Score a new table with a saved ridge model.

    Predictors are matched **by name**, in any column order; every other
    column passes through as an identifier. A predictor the new table lacks
    is refused by name rather than imputed -- a model scored on a feature it
    never sees is not the model that was validated.

    Parameters
    ----------
    model_json
        A model written by :func:`fit_ridge_csv` (or a folder holding exactly
        one).
    input_csv
        The table to score: an analysis table, or any feature table carrying
        the model's predictors.
    out_csv
        Defaults to ``<input stem>_ridge_predictions.csv`` beside the input.
    overwrite_existing
        When False (default) and the predictions already exist, they are
        returned untouched instead of recomputed.
    rounding
        Decimal places in the predicted values.

    Returns
    -------
    Path
        The predictions table: the input's identifier columns plus one
        ``pred_<outcome>`` column per outcome the model holds.
    """
    import numpy as np

    got = prepare_apply(
        model_json=model_json, load=_load_model, input_csv=input_csv,
        out_csv=out_csv, default_suffix="_ridge_predictions.csv",
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        verbose=verbose, encoding=encoding, id_cols=id_cols)
    if isinstance(got, Path):
        return got
    model, design = got.model, got.design
    columns = {}
    skipped = {}
    for outcome, block in model["outcomes"].items():
        preds = predict_rows(design, block["kept"],
                             np.asarray(block["mu"]),
                             np.asarray(block["sigma"]),
                             np.asarray(block["coef"]),
                             float(block["intercept"]), rounding)
        label = output_label(model, outcome)
        columns[f"pred_{label}"] = preds
        skipped[label] = sum(1 for p in preds if p is None)

    return write_predictions(got, columns, skipped, encoding=encoding,
                             on_progress=on_progress, verbose=verbose,
                             tag="ridge")


def _refuse_categorical_outcome(column: str, values) -> None:
    """
    Send a class-valued outcome next door instead of calling it a typo.

    Without this the numeric reader raises "'female' is not a number, fix
    the value", which is confident, unhelpful and wrong: nothing about the
    cell needs fixing, the researcher simply wants a classifier. The two
    modules are each other's near miss, so each names the other.
    """
    from ._common import looks_numeric

    present = [str(v).strip() for v in values if str(v).strip() != ""]
    if not present or looks_numeric(present):
        return
    levels = sorted(set(present))
    raise ValueError(
        f"outcome {column!r} holds words rather than numbers "
        f"({name_a_few(levels)}), so there is no variance in it to explain. "
        f"Predicting which class a text belongs to is the classification "
        f"step's job -- run that instead. If this column really is a "
        f"measurement, the non-numeric values are the thing to fix.")


#: Kept under its old name for the classifier, which imports it.
_score_design = score_design


def _write_coefficients(path: Path, entries, outcome_cols, lead, rounding,
                        encoding: str) -> None:
    """
    One row per predictor, one column per outcome.

    Long format put every (predictor, outcome) pair on its own line, so
    comparing what predicted age against what predicted extraversion meant
    reading 165 rows, then another 165, and holding them side by side in
    your head. Nobody does that. Wide is the shape this table is actually
    read in -- and the shape it goes into a paper in.

    Feature set and model stay as columns rather than becoming more column
    *blocks*: fitting five feature sets separately then adds five times the
    rows, which sorts and filters, where five times the columns would not.

    Predictors are ordered by their largest absolute coefficient across the
    outcomes, so the ones that did something are at the top of the file
    instead of wherever the alphabet put them.
    """
    if not entries:
        _write_csv(path, lead + ["feature_set", "model", "controls", "predictor"],
               [], encoding)
        return
    cells: dict = {}
    for key, outcome, value in entries:
        cells.setdefault(key, {})[outcome] = value
    order = sorted(
        cells,
        key=lambda k: (k[:-1], -max(abs(v) for v in cells[k].values())))
    header = (lead + ["feature_set", "model", "controls", "predictor"]
              + list(outcome_cols))
    rows = []
    for key in order:
        row = list(key)
        # a blank means this predictor wasn't in that outcome's model at all
        # (dropped as constant, or set aside as too sparse) -- that's a
        # different thing from a coefficient of zero.
        row += [fmt(cells[key][o], rounding) if o in cells[key] else ""
                for o in outcome_cols]
        rows.append(row)
    _write_csv(path, header, rows, encoding)


def _write_predictions(path: Path, table, outcome_cols, predictions,
                       encoding: str) -> None:
    """The training table with its observed and predicted values side by
    side -- the file a byte-identity check against apply is run on."""
    key = [c for c in ("text_id",) if c in table.columns] or \
        [table.columns[0]]
    header = key + [c for outcome in outcome_cols
                    for c in (outcome, f"pred_{outcome}")]
    rows = []
    for i in range(len(table)):
        row = [table[c].iloc[i] for c in key]
        for outcome in outcome_cols:
            value = predictions[outcome][i]
            row += [table[outcome].iloc[i], "" if value is None else value]
        rows.append(row)
    _write_csv(path, header, rows, encoding)


def _section_md(*, sets, outcome_cols, best, n_folds, zscore,
                metrics_rows, metrics_header=(), control_names=(),
                control_notes=(), model_scores=None, gap_notes=(),
                stratify: bool = True, set_notes=()) -> str:
    lines = ["## Prediction (cross-validated ridge)", "",
             f"Fitted ridge models for {len(outcome_cols)} outcome(s) "
             f"({', '.join(outcome_cols)}) over {len(sets)} feature set(s), "
             f"with {n_folds}-fold cross-validation"
             + (" (folds balanced on the outcome)" if stratify
                else " (random folds)")
             + (" on standardized predictors." if zscore
                else ", predictors left in their own units."),
             "",
             "Every number below is **out-of-fold**: each row was predicted "
             "by a model that never saw it."]
    lines += sample_warnings(metrics_rows, metrics_header)
    lines += list(gap_notes)
    if control_names:
        lines += ["", f"Fitted with {len(control_names)} control column(s) "
                  f"held alongside -- and instead of -- the language: "
                  f"{', '.join('`' + c + '`' for c in control_names)}. Each "
                  f"outcome therefore has up to three rows: the controls "
                  f"alone, the language alone, and both. All three are "
                  f"fitted on the same rows and the same folds, so "
                  f"`delta_r2_over_controls` on the combined row is what "
                  f"the language added over the controls -- the number a "
                  f"paper reports."]
        for note in control_notes:
            lines.append(f"- {note}")
        gains = []
        for (outcome, combo, with_lang), score in (model_scores or {}).items():
            if not with_lang or not combo:
                continue
            base = (model_scores or {}).get((outcome, combo, False))
            if base is not None and base == base and score == score:
                gains.append((score - base, outcome, base, score))
        if gains:
            lines += ["", "What the language added, over the controls "
                      "alone:", ""]
            for delta, outcome, base, score in sorted(gains, reverse=True):
                lines.append(
                    f"- `{outcome}`: R² {base:.3f} → {score:.3f} "
                    f"(**{delta:+.3f}**)")
    if best:
        lines += ["", "Best feature set per outcome:", ""]
        for outcome, (r2, set_name, alpha) in best.items():
            verdict = ("predicts nothing better than the mean"
                       if r2 <= 0 else f"cross-validated R² = {r2:.3f}")
            lines.append(f"- `{outcome}`: **{set_name}** — {verdict} "
                         f"(alpha = {alpha:g})")
    for note in set_notes:
        lines += ["", note]
    if len(sets) > 1:
        lines += comparison_lines(metrics_rows, metrics_header, score="cv_r2",
                                  label="CV R²")
        lines += ["", "Sort `ridge_cv_metrics.csv` by `cv_r2` within an "
                  "outcome to compare the sets directly."]
        # feature sets fit on different numbers of rows aren't being compared
        # on equal terms, and the reason is usually invisible: a measure
        # that's undefined for short texts (a windowed one, say) takes every
        # row it's missing from down with it under listwise deletion. we
        # spell it out here because someone comparing two R2s has no other
        # way of knowing that one of them was computed on a fifth of the data.
        at = {name: i for i, name in enumerate(metrics_header)}
        by_outcome: dict = {}
        for row in metrics_rows:
            by_outcome.setdefault(row[at["outcome"]], []).append(
                (row[at["feature_set"]], int(row[at["n_used"]])))
        for outcome, pairs in by_outcome.items():
            counts = {n for _, n in pairs}
            if len(counts) > 1 and min(counts) < 0.9 * max(counts):
                detail = ", ".join(f"{name} {n:,}" for name, n in pairs)
                lines += ["", f"⚠ For `{outcome}` the sets were fitted on "
                          f"different numbers of rows ({detail}). A feature "
                          f"that is undefined for some texts takes those "
                          f"rows out of its own model, so these scores are "
                          f"not directly comparable."]
    lines += ["", "Details: `ridge_cv_metrics.csv`, "
              "`ridge_coefficients.csv`, `ridge_folds.csv` (every fold's "
              "own score, so you can see whether one of them carried the "
              "result), `ridge_alpha_path.csv`, and one reusable model per "
              "feature set under `models/`."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"fit": fit_ridge_csv, "apply": apply_ridge_csv},
    description='Cross-validated ridge regression over an analysis table, and scoring with a saved model.',
    aliases={
        'out_csv': ['--out'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
