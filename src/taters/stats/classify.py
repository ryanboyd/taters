"""
Classification: predict which category a text belongs to, and say honestly
how well.

The regression module answers "how much?" -- this one answers "which?". A
gender coded ``male``/``female``, a diagnosis, a condition label: these are
outcomes you predict *classes* for, and handing one to a regression is not a
near-miss but a category error, so each module refuses the other's outcomes
by name and points at its sibling.

The model is L2-penalized logistic regression, fitted by iteratively
reweighted least squares -- the classification analog of the ridge next
door, and for the same reason: language features are many and collinear, and
the penalty is what keeps the fit stable when two predictors say almost the
same thing. Multi-class outcomes are fitted one-class-against-the-rest and
the probabilities normalized, which is transparent about what it is doing in
a way a single softmax fit is not.

What gets reported, and why it is more than one number
-----------------------------------------------------
**Accuracy** alone is close to useless and is the number everyone quotes: on
an outcome where 90% of people are in one class, a model that always answers
"that one" is 90% accurate and worthless. So accuracy is never reported
without the accuracy of exactly that model -- always guess the commonest
class -- beside it.

**Area under the ROC curve** asks whether the model *ranks* cases correctly,
independent of where you put the threshold, and is unmoved by class
imbalance. It is the number to lead with.

**Precision, recall and F1, per class**, because a model can be excellent on
the majority class and useless on the one you care about, and every average
hides that. The macro average weights classes equally, the weighted average
weights them by how common they are, and they disagree exactly when the
imbalance matters.

**The confusion matrix**, which is the only output that says *what* the
mistakes were.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.model_spec import class_label, output_label, slug
from ._fit_common import (load_model_doc, model_header, output_folders, comparison_lines,
                          prepare_apply, prepare_fit, set_design,
                          write_predictions, random_folds)
from ..helpers.progress import announce
from ._common import (MAX_MISSING, _write_csv, components_appendix, fmt, sample_warnings, write_section,
                      gap_lines, gaps_section, reusable)
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

#: Bumped on any change that would make an older Taters misread a
#: model. Format 2 added `output_names`: a build that ignored it
#: would write `pred_age` where the researcher asked for
#: `pred_age_blogs`, so refusing the file outright is the only safe
#: way to be wrong about it.
CLASSIFIER_MODEL_FORMAT = 2

#: Fewest rows a class may have and still be modeled. Below this the class
#: cannot appear in every fold, so its recall is measured on nothing.
MIN_PER_CLASS = 5

#: How many distinct values an outcome may have before it stops being a set
#: of categories and starts being an identifier.
MAX_CLASSES = 20

#: IRLS converges in a handful of steps on a penalized problem; the cap is a
#: backstop against a pathological design, not a schedule.
_MAX_IRLS = 50
_IRLS_TOL = 1e-9


def categorical_column(values: Sequence[object], *, column: str) -> List[str]:
    """One column as class labels, blanks meaning missing."""
    return [("" if v is None else str(v)).strip() for v in values]


def _fit_logistic(x, y, alpha: float, start=None):
    """
    L2-penalized logistic regression by iteratively reweighted least squares.

    ``y`` is 0/1. Returns ``(intercept, coefficients)``. The intercept is
    never penalized -- shrinking it would pull every predicted probability
    toward one half regardless of how common the class actually is.

    ``start`` is a previous ``(intercept, coefficients)`` to begin from.
    Newton converges to the same optimum from any start (the penalized
    log-likelihood is strictly concave), so this changes the cost and not
    the answer: begun from the neighboring penalty's solution the fit takes
    two or three steps instead of twenty, and on a real run -- 933 texts,
    1024 embedding columns, 33 penalties, 5 folds, three model variants --
    that was the difference between an hour and a few minutes.

    Newton's method on the penalized log-likelihood, which for this problem
    is the textbook IRLS update and converges in a handful of steps. The
    penalty also removes the one way this fit can fail to converge at all:
    with perfectly separable classes the unpenalized coefficients run off to
    infinity, and any alpha above zero keeps them finite.
    """
    import numpy as np

    n, p = x.shape
    design = np.hstack([np.ones((n, 1)), x])
    beta = np.zeros(p + 1)
    if start is not None:
        beta[0], beta[1:] = float(start[0]), np.asarray(start[1], dtype=float)
    penalty = np.eye(p + 1) * alpha
    penalty[0, 0] = 0.0
    for _ in range(_MAX_IRLS):
        eta = design @ beta
        # we clip before exponentiating. a confident fit overflows exp() long
        # before it changes any probability enough to round differently.
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        weight = np.maximum(prob * (1 - prob), 1e-10)
        gradient = design.T @ (y - prob) - penalty @ beta
        hessian = (design * weight[:, None]).T @ design + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
        if float(np.abs(step).max()) < _IRLS_TOL:
            break
    return float(beta[0]), beta[1:]


def _probabilities(x, models):
    """Class probabilities from one-against-the-rest fits, normalized.

    Each class has its own binary fit, so the probabilities do not sum to
    one on their own; dividing by their total is what makes them a
    distribution. For a two-class outcome there is one fit and the second
    probability is its complement, so nothing is normalized and the numbers
    are exactly the logistic ones.
    """
    import numpy as np

    if len(models) == 2 and models[1] is None:
        intercept, coef = models[0]
        prob = 1.0 / (1.0 + np.exp(-np.clip(x @ coef + intercept, -30, 30)))
        return np.column_stack([1.0 - prob, prob])
    raw = []
    for entry in models:
        intercept, coef = entry
        raw.append(1.0 / (1.0 + np.exp(-np.clip(x @ coef + intercept,
                                                -30, 30))))
    out = np.column_stack(raw)
    totals = out.sum(axis=1, keepdims=True)
    return np.divide(out, totals, out=np.full_like(out, np.nan),
                     where=totals > 0)


def _auc(truth, score) -> float:
    """
    Area under the ROC curve, from ranks.

    The rank formulation (the Mann-Whitney U identity) is exact, handles
    ties by average rank, and needs no threshold sweep: AUC is the
    probability that a randomly chosen positive case is ranked above a
    randomly chosen negative one.
    """
    import numpy as np
    from scipy.stats import rankdata

    truth = np.asarray(truth)
    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[truth == 1].sum()
                  - positives * (positives + 1) / 2) / (positives * negatives))


def _log_loss(indicator, probability) -> float:
    """Mean negative log-likelihood -- the proper score alpha is chosen by.

    Accuracy is a step function of the probabilities, so choosing a penalty
    by it is choosing between ties; log loss reads the confidence as well as
    the verdict, which is what makes it able to tell two penalties apart.
    """
    import numpy as np

    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.sum(indicator * np.log(clipped), axis=1)))


def _class_report(truth, predicted, classes):
    """Precision, recall, F1 and support per class, plus the two averages."""
    import numpy as np

    rows = {}
    for label in classes:
        tp = int(np.sum((predicted == label) & (truth == label)))
        fp = int(np.sum((predicted == label) & (truth != label)))
        fn = int(np.sum((predicted != label) & (truth == label)))
        support = int(np.sum(truth == label))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        if precision == precision and recall == recall and precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support else float("nan")
        rows[label] = {"precision": precision, "recall": recall, "f1": f1,
                       "support": support}
    total = sum(r["support"] for r in rows.values()) or 1
    macro = {k: float(np.nanmean([r[k] for r in rows.values()]))
             for k in ("precision", "recall", "f1")}
    weighted = {k: float(sum(r[k] * r["support"] for r in rows.values()
                             if r[k] == r[k]) / total)
                for k in ("precision", "recall", "f1")}
    return rows, macro, weighted


def _stratified_folds(labels, classes, n_folds: int, seed: int):
    """
    Folds that each contain every class, in roughly the original proportions.

    Dealt within class rather than over the whole sample, because a plain
    shuffle can hand a fold none of a small class -- and a fold with no
    positive cases has no recall and no area under any curve, so one unlucky
    split silently turns k-fold validation into (k-1)-fold.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for label in classes:
        where = np.flatnonzero(labels == label)
        order = rng.permutation(where.size)
        folds[where[order]] = np.arange(where.size) % n_folds
    return folds


def _select_penalty(z, labels, classes, *, grid, n_folds: int, seed: int,
                    stratify: bool) -> int:
    """
    Pick a penalty from these rows and nothing else, by inner k-fold.

    This is what scikit-learn's ``LogisticRegressionCV`` does: split the rows,
    fit the whole grid on each split's training part, score it on the held-out
    part, average each penalty's score across the splits, and take the best.
    Nothing exotic -- and unlike the ridge next door there is no closed form
    to shortcut it with, because a penalized logistic fit has to iterate. So
    the grid costs real fits here where the ridge gets it from one
    decomposition, which is the whole of the difference between them.

    Largest penalty first with each fit warm-started from the last, the same
    trick the outer loop uses: neighboring penalties have neighboring
    solutions, so Newton takes a step or two rather than starting from zero.

    An inner split that loses a class has no log loss worth averaging, so it
    is skipped; if every one of them does, the answer falls back to scoring
    the grid in-sample, which is weak but is not a crash.
    """
    import numpy as np

    n = len(labels)
    inner = (_stratified_folds(labels, classes, n_folds, seed + 1) if stratify
             else random_folds(n, n_folds, seed + 1))
    totals = np.zeros(len(grid))
    scored = 0
    for j in range(n_folds):
        held, rest = inner == j, inner != j
        if not held.any() or not rest.any():
            continue
        if any((labels[rest] == c).sum() == 0 for c in classes):
            continue
        indicators = np.column_stack([(labels[held] == c).astype(float)
                                      for c in classes])
        fits = None
        for a_i in sorted(range(len(grid)), key=lambda i: -grid[i]):
            fits = _fit_one_vs_rest(z[rest], labels[rest], classes, grid[a_i],
                                    starts=fits)
            totals[a_i] += _log_loss(indicators, _probabilities(z[held], fits))
        scored += 1
    if not scored:                             # pragma: no cover - tiny sample
        indicators = np.column_stack([(labels == c).astype(float)
                                      for c in classes])
        fits = None
        for a_i in sorted(range(len(grid)), key=lambda i: -grid[i]):
            fits = _fit_one_vs_rest(z, labels, classes, grid[a_i], starts=fits)
            totals[a_i] = _log_loss(indicators, _probabilities(z, fits))
        scored = 1
    mean = totals / scored
    # ties go to the larger penalty, same as we do in the ridge next door.
    return int(np.max(np.flatnonzero(mean == np.nanmin(mean))))


def _cv_logistic(x, labels, classes, *, grid, n_folds: int, seed: int,
                 zscore: bool, stratify: bool = True):
    """
    Cross-validate a penalized logistic fit, then refit the winner.

    The penalty is chosen **inside** each fold, from that fold's training rows
    alone, by the inner k-fold that scikit-learn's ``LogisticRegressionCV``
    uses (:func:`_select_penalty`) -- so nothing about a held-out row, its
    label included, reaches the model that predicts it, and ``log_loss`` and
    everything read off the same probabilities are out of fold in the full
    sense.

    This used to fit the whole grid in every fold, pool the out-of-fold
    probabilities, and pick the penalty whose pooled log loss was lowest --
    then report that same pooled loss. Every prediction was honestly out of
    fold; the *choice of which penalty's predictions to report* looked at all
    of them, which is a choice made on the outcome and left the result
    flattering.

    It costs more than the ridge's version of the same fix, and the reason is
    the model rather than the design: ridge reads its whole penalty grid off
    one decomposition, so nesting was free, while a penalized logistic fit has
    to iterate and every penalty is a real fit. Warm starts down the grid keep
    the factor small.

    What comes back estimates the **procedure** rather than one fixed penalty.
    ``fold_alphas`` says what each fold picked; the model that gets saved runs
    the same rule over every row, so its ``alpha`` need not match any single
    fold's.
    """
    import numpy as np

    n = x.shape[0]
    spread = x.std(axis=0)
    kept = [i for i, s in enumerate(spread) if s > 0]
    if not kept:
        return None, list(range(x.shape[1]))
    dropped = [i for i, s in enumerate(spread) if s <= 0]
    xk = x[:, kept]
    indicator = np.column_stack([(labels == c).astype(float) for c in classes])

    folds = (_stratified_folds(labels, classes, n_folds, seed) if stratify
             else random_folds(len(labels), n_folds, seed))
    oof = np.full((n, len(classes)), np.nan)
    fold_alphas: list = []
    for fold in range(n_folds):
        test = folds == fold
        train = ~test
        x_tr = xk[train]
        mu = x_tr.mean(axis=0)
        sigma = x_tr.std(axis=0) if zscore else np.ones(x_tr.shape[1])
        sigma = np.where(sigma > 0, sigma, 1.0)
        z_tr, z_te = (x_tr - mu) / sigma, (xk[test] - mu) / sigma
        chosen = _select_penalty(z_tr, labels[train], classes, grid=grid,
                                 n_folds=n_folds, seed=seed, stratify=stratify)
        fold_alphas.append(float(grid[chosen]))
        fits = _fit_one_vs_rest(z_tr, labels[train], classes, grid[chosen])
        oof[test] = _probabilities(z_te, fits)

    # the number we report: held-out probabilities from models whose penalties
    # those rows had no part in choosing
    best = _log_loss(indicator, oof)

    mu = xk.mean(axis=0)
    sigma = xk.std(axis=0) if zscore else np.ones(xk.shape[1])
    sigma = np.where(sigma > 0, sigma, 1.0)
    # the model we keep: same rule, every row. It may see all the data
    # because it is not what is being scored above.
    best_i = _select_penalty((xk - mu) / sigma, labels, classes, grid=grid,
                             n_folds=n_folds, seed=seed, stratify=stratify)
    alpha = grid[best_i]
    final = _fit_one_vs_rest((xk - mu) / sigma, labels, classes, alpha)
    return {
        "n": n, "kept": kept, "alpha": alpha, "folds": folds,
        "fold_alphas": fold_alphas,
        "oof": oof, "log_loss": best,
        "mu": [float(v) for v in mu], "sigma": [float(v) for v in sigma],
        "models": final,
        "train_probabilities": _probabilities((xk - mu) / sigma, final),
    }, dropped


def _fit_one_vs_rest(x, labels, classes, alpha: float, starts=None):
    """One binary fit per class -- or exactly one for a two-class outcome,
    where the second is its complement and fitting it twice would only
    invite the two to disagree. ``starts`` are the fits for a neighboring
    penalty, to begin from."""
    starts = starts or [None] * len(classes)
    if len(classes) == 2:
        return [_fit_logistic(x, (labels == classes[1]).astype(float), alpha,
                              start=starts[0]),
                None]
    return [_fit_logistic(x, (labels == c).astype(float), alpha,
                          start=starts[i])
            for i, c in enumerate(classes)]


def _score(labels, classes, probabilities) -> dict:
    """Every headline number for one set of out-of-fold probabilities."""
    import numpy as np

    predicted = np.array([classes[i] for i in np.argmax(probabilities,
                                                        axis=1)])
    accuracy = float(np.mean(predicted == labels))
    # the "always answer the commonest class" model. every accuracy figure
    # has to be read against this one -- accuracy by itself means nothing.
    counts = {c: int(np.sum(labels == c)) for c in classes}
    baseline = max(counts.values()) / len(labels)
    if len(classes) == 2:
        auc = _auc((labels == classes[1]).astype(int), probabilities[:, 1])
        per_class_auc = {classes[1]: auc, classes[0]: auc}
    else:
        # one class against the rest, averaged with every class counting
        # equally -- otherwise the majority class would get to decide the
        # number.
        per_class_auc = {c: _auc((labels == c).astype(int),
                                 probabilities[:, i])
                         for i, c in enumerate(classes)}
        auc = float(np.nanmean(list(per_class_auc.values())))
    rows, macro, weighted = _class_report(labels, predicted, classes)
    for c in classes:
        rows[c]["auc_vs_rest"] = per_class_auc[c]
    confusion = {(t, p): int(np.sum((labels == t) & (predicted == p)))
                 for t in classes for p in classes}
    return {"accuracy": accuracy, "baseline_accuracy": baseline, "auc": auc,
            "per_class": rows, "macro": macro, "weighted": weighted,
            "confusion": confusion, "predicted": predicted}


def classify_rows(cells, kept, mu, sigma, models, classes, rounding: int):
    """
    Predicted class and its probability for every row, or ``None`` per row.

    The one code path used both when fitting and when scoring a new table,
    so that applying a saved model to the data it was fitted on reproduces
    the training predictions exactly. Two nearly-identical prediction
    functions would drift apart on the first change to either, and the
    symptom -- a model that scores differently through the two doors -- is
    the kind nobody notices.

    A row missing any predictor the model kept is left unscored rather than
    filled in: imputing a zero on a standardized scale silently asserts the
    row is average on that feature.
    """
    import numpy as np

    kept = list(kept)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fits = [None if e is None
            else (float(e["intercept"]), np.asarray(e["coef"], dtype=float))
            for e in models]
    block = np.asarray(cells, dtype=float)[:, kept]
    usable = ~np.isnan(block).any(axis=1)
    labels: List[Optional[str]] = [None] * block.shape[0]
    probs: List[Optional[float]] = [None] * block.shape[0]
    if not usable.any():
        return labels, probs
    probabilities = _probabilities((block[usable] - mu) / sigma, fits)
    picked = np.argmax(probabilities, axis=1)
    for out_i, row_i in enumerate(np.flatnonzero(usable)):
        labels[row_i] = classes[int(picked[out_i])]
        probs[row_i] = round(float(probabilities[out_i, picked[out_i]]),
                             rounding)
    return labels, probs


def fit_classifier_csv(
    *,
    table_csv: PathLike,
    outcome_cols: Sequence[str],
    feature_sets=None,
    control_cols: Sequence[str] = (),
    categorical_controls: Sequence[str] = (),
    control_combos: Literal["none_and_all", "each", "subsets"] = "none_and_all",
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
    Predict a categorical outcome from the features, and report honestly.

    Parameters
    ----------
    table_csv
        The assembled analysis table.
    outcome_cols
        The categorical column(s) to predict -- ``male``/``female``, a
        diagnosis, a condition. Class labels are taken as they are written;
        blanks mean missing.
    feature_sets
        ``None`` for all features as one set, ``"per_table"`` to fit one
        model per source feature table and compare them, or
        ``{name: [columns]}``.
        Naming the sets yourself also turns **off** the combining that
        ``set_combos`` describes: you get exactly the sets you named, each
        fitted on its own, and no "all together" model. Combinations are
        built only when this is left as ``None``, where the groups come from
        the sidecar the assemble step writes beside the table
        (``<name>_sets.json``) -- which is how the app runs it, and why the
        app gets an ``__all__`` model without asking. The two read alike and
        do not behave alike, so it is worth saying out loud.
    control_cols, categorical_controls, control_combos, set_combos : see the ridge step
        The same design: with controls, each outcome is fitted three ways on
        one common sample -- controls alone, language alone, both -- so the
        language's own contribution can be read off, here as a gain in area
        under the curve rather than in R-squared.
    pca : str or list of str, default="off"
        Analyze components instead of the raw measures. ``"off"`` uses the
        features as they are; ``"all"`` reduces every feature set; a list of
        set names reduces those and leaves the rest alone. Set per analysis,
        deliberately -- a classifier over four hundred collinear measures is
        what components are for, and the loadings land beside these results
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
        accuracy is not out-of-sample at all.
    alphas
        L2 penalties to search. Chosen by out-of-fold **log loss** rather
        than accuracy: accuracy is a step function of the probabilities, so
        selecting on it means choosing between ties.
    n_folds
        Cross-validation folds. Stratified by class, so every fold contains
        every class -- a fold with none of a small class has no recall and
        no curve to measure, which silently turns k-fold into (k-1)-fold.
    stratify
        Make the folds alike: the same mix of classes in every fold (the
        default, for the reason above). Off, the folds are a plain seeded
        shuffle and a fold may lack a class; the report says so.
    zscore
        Standardize predictors on the training statistics, so coefficients
        are per standard deviation and comparable to each other.
    max_missing
        Set aside any predictor missing for more than this fraction of the
        rows, rather than letting it delete them. See the ridge step: left
        in, 60 structurally-blank cohesion columns once cost 885 of 938
        rows. Set it to 1.0 to keep every column and accept the row loss.
    seed
        Fixes the fold shuffle, and is stored in the model.
    out_dir, out_models_dir
        Where the tables and the model files go; both default beside the
        analysis table.
    overwrite_existing
        When False (default) and the metrics table exists, it is returned
        untouched.
    rounding
        Decimal places in the output tables.

    Returns
    -------
    Path
        ``classifier_cv_metrics.csv`` -- one row per (feature set, outcome,
        model). Four companions land beside it: ``classifier_per_class.csv``
        (where a model is good and where it is useless),
        ``classifier_confusion.csv`` (what the mistakes actually were),
        ``classifier_folds.csv`` and ``classifier_coefficients.csv``.

    Notes
    -----
    Every number is out-of-fold: each row is classified by a model that
    never saw it. Accuracy is always reported beside the accuracy of always
    guessing the commonest class, because on an imbalanced outcome the
    second number is most of the first.
    """
    import numpy as np

    table_csv, folder, models_dir = output_folders(table_csv, out_dir,
                                                   out_models_dir)
    metrics_path = folder / "classifier_cv_metrics.csv"
    if reusable(metrics_path, table_csv, overwrite_existing=overwrite_existing,
                verbose=verbose, what="the classification results"):
        if verbose:
            print(f"Classification results already exist; returning existing "
                  f"file: {metrics_path}")
        return metrics_path

    outcome_cols = [str(c) for c in outcome_cols]
    inputs = prepare_fit(
        table_csv=table_csv, folder=folder, stem="classifier",
        outcome_cols=outcome_cols, feature_sets=feature_sets,
        control_cols=control_cols, categorical_controls=categorical_controls,
        control_combos=control_combos, pca=pca, pca_components=pca_components,
        pca_rotation=pca_rotation, pca_retain=pca_retain, split_col=split_col,
        set_combos=set_combos, alphas=alphas,
        n_folds=n_folds, max_missing=max_missing, encoding=encoding,
        rounding=rounding, verbose=verbose, on_progress=on_progress,
        zero_alpha_reason=("an unpenalized logistic fit does not converge "
                           "at all when the classes are separable"))
    n_folds = int(n_folds)
    table, grid, sets, columns = (inputs.table, inputs.grid, inputs.sets,
                                  inputs.columns)
    pca_notes = inputs.pca_notes
    control_matrix, control_names, control_notes, control_spec = (
        inputs.control_matrix, inputs.control_names, inputs.control_notes,
        inputs.control_spec)
    control_sets, subsets, lead, total = (inputs.control_sets, inputs.subsets,
                                          inputs.lead, inputs.total)
    labels_by_outcome = {c: np.array(categorical_column(table[c].tolist(),
                                                        column=c))
                         for c in outcome_cols}
    metric_rows, class_rows, conf_rows, fold_rows, coef_rows = [], [], [], [], []
    scores: dict = {}
    best: dict = {}
    gaps: list = []
    done = 0

    for subset, subset_mask in subsets:
        prefix = [subset] if split_col else []
        for set_name, all_cols in sets.items():
            cols, sparse, design_all, design = set_design(
                set_name, all_cols, columns, subset_mask, max_missing,
                verbose=verbose, tag="classifier")
            gaps += gap_lines(set_name, cols, design)
            model_doc = model_header(
                kind="taters-classifier-model", fmt=CLASSIFIER_MODEL_FORMAT,
                set_name=set_name, cols=cols, table_csv=table_csv,
                reduction=inputs.reductions.get(set_name),
                cv={"n_folds": n_folds, "seed": seed,
                    "selection": "min_out_of_fold_log_loss_prefer_larger_alpha",
                    "alphas": grid, "stratified": bool(stratify)},
                zscore=zscore, subset=subset)
            predictions = {}
            per_outcome: dict = {}

            for outcome in outcome_cols:
                labels_all = (labels_by_outcome[outcome] if subset_mask is None
                              else labels_by_outcome[outcome][subset_mask])
                ctrl = (control_matrix if subset_mask is None
                        else control_matrix[subset_mask])
                # how many rows *had* the outcome, so that the table can say
                # what fraction of them the model actually got to look at.
                n_available = int((labels_all != "").sum())
                complete = (~np.isnan(design).any(axis=1)
                            & (labels_all != ""))
                if ctrl.shape[1]:
                    complete = complete & ~np.isnan(ctrl).any(axis=1)
                labels = labels_all[complete]
                classes = sorted(set(labels))
                _check_classes(outcome, classes, labels, n_folds)

                keep_best = None
                for control_set in control_sets:
                    idx = [i for i, name in enumerate(control_names)
                           if name.split("=")[0] in control_set]
                    for with_language in (False, True):
                        if not with_language and not idx:
                            continue
                        done += 1
                        if on_progress is not None:
                            on_progress(done, total,
                                        f"classifying {outcome} on {set_name}")
                        blocks, names = [], []
                        if idx:
                            blocks.append(ctrl[complete][:, idx])
                            names += [control_names[i] for i in idx]
                        if with_language:
                            blocks.append(design[complete])
                            names += list(cols)
                        label = ("controls+language" if idx and with_language
                                 else "controls" if idx else "language")

                        fit, dropped = _cv_logistic(
                            np.hstack(blocks), labels, classes, grid=grid,
                            n_folds=n_folds, seed=seed, zscore=zscore, stratify=stratify)
                        if fit is None:
                            raise ValueError(
                                f"{outcome} on feature set {set_name!r}: "
                                f"every predictor is constant across the "
                                f"usable rows.")
                        s = _score(labels, classes, fit["oof"])
                        train = _score(labels, classes,
                                       fit["train_probabilities"])
                        per_fold = [
                            _score(labels[fit["folds"] == f], classes,
                                   fit["oof"][fit["folds"] == f])
                            for f in range(n_folds)]
                        acc_folds = np.array([m["accuracy"] for m in per_fold])
                        auc_folds = np.array([m["auc"] for m in per_fold],
                                             dtype=float)
                        usable_auc = auc_folds[~np.isnan(auc_folds)]
                        combo = "|".join(control_set)

                        delta_acc = delta_auc = ""
                        if idx and with_language:
                            base = scores.get((outcome, combo, False))
                            if base:
                                delta_acc = fmt(s["accuracy"]
                                                - base["accuracy"], rounding)
                                delta_auc = fmt(s["auc"] - base["auc"],
                                                rounding)
                        scores[(outcome, combo, with_language)] = s

                        metric_rows.append(prefix + [
                            set_name, str(len(inputs.set_members.get(set_name, [set_name]))),
                            str(len(all_cols)), outcome, label, combo,
                            str(len(labels)), str(n_available),
                            str(len(sparse)), str(len(classes)),
                            str(len(fit["kept"])), str(len(dropped)),
                            str(n_folds), str(bool(zscore)),
                            fmt(fit["alpha"], 8),
                            fmt(s["accuracy"], rounding), delta_acc,
                            fmt(s["baseline_accuracy"], rounding),
                            fmt(s["auc"], rounding), delta_auc,
                            fmt(s["macro"]["f1"], rounding),
                            fmt(s["weighted"]["f1"], rounding),
                            fmt(s["macro"]["precision"], rounding),
                            fmt(s["macro"]["recall"], rounding),
                            fmt(float(acc_folds.mean()), rounding),
                            fmt(float(acc_folds.std(ddof=1)
                                      / np.sqrt(acc_folds.size))
                                if acc_folds.size > 1 else float("nan"),
                                rounding),
                            fmt(float(usable_auc.mean()) if usable_auc.size
                                else float("nan"), rounding),
                            fmt(float(usable_auc.std(ddof=1)
                                      / np.sqrt(usable_auc.size))
                                if usable_auc.size > 1 else float("nan"),
                                rounding),
                            fmt(fit["log_loss"], rounding),
                            fmt(train["accuracy"], rounding)])
                        for c in classes:
                            r = s["per_class"][c]
                            class_rows.append(prefix + [
                                set_name, outcome, label, c,
                                str(r["support"]),
                                fmt(r["precision"], rounding),
                                fmt(r["recall"], rounding),
                                fmt(r["f1"], rounding),
                                fmt(r["auc_vs_rest"], rounding)])
                        for (t_c, p_c), count in s["confusion"].items():
                            conf_rows.append(prefix + [
                                set_name, outcome, label, t_c, p_c,
                                str(count)])
                        for f, m in enumerate(per_fold):
                            fold_rows.append(prefix + [
                                set_name, outcome, label, str(f + 1),
                                str(int((fit["folds"] == f).sum())),
                                # each fold chose this from its own training
                                # rows, by the inner k-fold
                                fmt(fit["fold_alphas"][f], 8),
                                fmt(m["accuracy"], rounding),
                                fmt(m["auc"], rounding),
                                fmt(m["macro"]["f1"], rounding)])
                        for c_i, entry in enumerate(fit["models"]):
                            if entry is None:
                                continue
                            intercept, coef = entry
                            against = (classes[1] if len(classes) == 2
                                       else classes[c_i])
                            for i in np.argsort(-np.abs(coef)):
                                coef_rows.append(prefix + [
                                    set_name, outcome, label, against,
                                    names[fit["kept"][i]],
                                    fmt(float(coef[i]), rounding)])
                        if s["auc"] == s["auc"] and s["auc"] > best.get(
                                outcome, (-1.0,))[0]:
                            best[outcome] = (
                                s["auc"],
                                f"{set_name} @ {subset}" if subset
                                else set_name, label)
                        if with_language:
                            # we hang onto the scored design over EVERY
                            # row of the subset, in the column order that
                            # `names` records, so that the saved model can
                            # be replayed over rows the fit itself had to
                            # drop.
                            full = ([ctrl[:, idx]] if idx else []) + [design]
                            keep_best = (fit, names, classes,
                                         np.hstack(full), idx)

                fit, names, classes, full_design, control_idx = keep_best
                entries = [None if e is None
                           else {"intercept": e[0],
                                 "coef": [float(v) for v in e[1]]}
                           for e in fit["models"]]
                # this outcome's own model file, written below: its block,
                # and the predictors and controls (the recipe, not just the
                # names -- see the ridge module for why) it was fitted with
                per_outcome[outcome] = {
                    "controls": [control_spec[i] for i in control_idx],
                    "predictors": names,
                    "block": {
                        "n_rows": len(labels), "classes": classes,
                        "alpha": fit["alpha"], "kept": fit["kept"],
                        "mu": fit["mu"], "sigma": fit["sigma"],
                        "models": entries,
                    },
                }

                final_pred, final_prob = classify_rows(
                    full_design, fit["kept"], fit["mu"], fit["sigma"],
                    entries, classes, rounding)
                oof_pred: list = [None] * len(labels_all)
                oof_prob: list = [None] * len(labels_all)
                fold_of: list = [None] * len(labels_all)
                picked = np.argmax(fit["oof"], axis=1)
                for j, row_i in enumerate(np.flatnonzero(complete)):
                    oof_pred[row_i] = classes[int(picked[j])]
                    oof_prob[row_i] = round(
                        float(fit["oof"][j, picked[j]]), rounding)
                    fold_of[row_i] = int(fit["folds"][j]) + 1
                predictions[outcome] = {
                    "observed": list(labels_all), "pred": final_pred,
                    "prob": final_prob, "oof_pred": oof_pred,
                    "oof_prob": oof_prob, "fold": fold_of}

            models_dir.mkdir(parents=True, exist_ok=True)
            stem = slug(set_name, "set") + (f"__{slug(subset)}" if subset else "")
            # one model file per outcome, same as the ridge (and for the same
            # reason: a model is the thing that predicts one outcome)
            for outcome, parts in per_outcome.items():
                doc = dict(model_doc)
                doc["outcomes"] = {outcome: parts["block"]}
                doc["controls"] = parts["controls"]
                doc["predictors"] = parts["predictors"]
                with atomic_write(models_dir / f"classifier__{stem}__{slug(outcome)}.json",
                                  mode="w", encoding="utf-8") as fh:
                    json.dump(doc, fh, indent=1)
            _write_predictions(
                folder / f"classifier_predictions__{stem}.csv",
                table if subset_mask is None else table[subset_mask],
                outcome_cols, predictions, encoding)

    announce(on_progress, "writing the results")
    metrics_header = (
           lead + ["feature_set", "n_feature_sets", "n_features", "outcome",
                   "model", "controls",
                   # the same three the ridge table carries, in the same
                   # order, so that "how much of my data went into this?"
                   # gets the same answer in the same place for both kinds
                   # of model.
                   "n_used", "n_available", "n_dropped_sparse",
                   "n_classes", "n_predictors", "n_dropped_constant",
                   "n_folds", "zscore", "alpha",
                   # accuracy never goes out alone. the baseline next to it
                   # is the model that always answers the commonest class.
                   "accuracy", "delta_accuracy_over_controls",
                   "baseline_accuracy",
                   "auc", "delta_auc_over_controls",
                   "f1_macro", "f1_weighted", "precision_macro",
                   "recall_macro",
                   "accuracy_folds", "accuracy_folds_se",
                   "auc_folds", "auc_folds_se",
                   "log_loss", "train_accuracy"])
    _write_csv(metrics_path, metrics_header, metric_rows, encoding)
    _write_csv(folder / "classifier_per_class.csv",
           lead + ["feature_set", "outcome", "model", "class", "support",
                   "precision", "recall", "f1", "auc_vs_rest"],
           class_rows, encoding)
    _write_csv(folder / "classifier_confusion.csv",
           lead + ["feature_set", "outcome", "model", "true_class",
                   "predicted_class", "n"], conf_rows, encoding)
    _write_csv(folder / "classifier_folds.csv",
           lead + ["feature_set", "outcome", "model", "fold", "n", "alpha",
                   "accuracy", "auc", "f1_macro"], fold_rows, encoding)
    _write_csv(folder / "classifier_coefficients.csv",
           lead + ["feature_set", "outcome", "model", "class", "predictor",
                   "coef"], coef_rows, encoding)

    section = _section_md(
        best=best, outcome_cols=outcome_cols, sets=sets, n_folds=n_folds,
        stratify=stratify,
        scores=scores, control_names=control_names,
        control_notes=control_notes, metrics_rows=metric_rows,
        metrics_header=metrics_header,
        gap_notes=gaps_section(gaps, max_missing=max_missing),
        set_notes=inputs.set_notes)
    section += components_appendix(pca_notes)
    write_section(folder, "classifier", section)
    if verbose:
        print(f"[classifier] {len(metric_rows)} model(s) -> {metrics_path}")
    return metrics_path


def _check_classes(outcome: str, classes, labels, n_folds: int) -> None:
    """Refuse an outcome that is not a set of categories, in words."""
    import numpy as np

    from ._common import looks_numeric

    if len(classes) < 2:
        raise ValueError(
            f"outcome {outcome!r} has "
            f"{'one class' if classes else 'no values'} "
            f"({classes or 'all blank'}); classifying needs at least two.")
    if len(classes) > MAX_CLASSES:
        raise ValueError(
            f"outcome {outcome!r} has {len(classes)} distinct values, which "
            f"is more like an identifier than a set of categories "
            f"({MAX_CLASSES} is the limit). If it is a measurement, predict "
            f"it with the ridge step instead.")
    if looks_numeric(labels) and len(classes) > n_folds:
        raise ValueError(
            f"outcome {outcome!r} holds {len(classes)} distinct numbers, so "
            f"it looks like a measurement rather than a set of categories. "
            f"Predict a measurement with the ridge step, which reports "
            f"variance explained; classification would treat "
            f"{classes[0]!r} and {classes[1]!r} as unrelated labels.")
    thin = {c: int(np.sum(labels == c)) for c in classes}
    too_thin = {c: n for c, n in thin.items() if n < MIN_PER_CLASS}
    if too_thin:
        raise ValueError(
            f"outcome {outcome!r} has class(es) with fewer than "
            f"{MIN_PER_CLASS} rows ({too_thin}); a class that cannot appear "
            f"in every fold has its recall measured on nothing.")


def _section_md(*, best, outcome_cols, sets, n_folds, scores,
                control_names=(), control_notes=(), metrics_rows=(),
                metrics_header=(), gap_notes=(), stratify: bool = True,
                set_notes=()) -> str:
    lines = ["## Classification", "",
             f"Predicted {len(outcome_cols)} categorical outcome(s) "
             f"({', '.join(outcome_cols)}) from {len(sets)} feature set(s), "
             f"with {n_folds}-fold cross-validation "
             + ("stratified by class." if stratify else
                "on random folds -- not stratified, as asked, so a fold may "
                "hold none of a small class and its recall is then measured "
                "on nothing."),
             "",
             "Every number is **out-of-fold**: each row was classified by a "
             "model that never saw it. Accuracy sits beside the accuracy of "
             "always guessing the commonest class, because on an "
             "imbalanced outcome that is most of it."]
    lines += sample_warnings(metrics_rows, metrics_header, model="classifier")
    lines += list(gap_notes)
    if best:
        lines += ["", "Best model per outcome, by area under the curve:", ""]
        for outcome, (auc, set_name, label) in best.items():
            verdict = ("no better than chance" if auc <= 0.55
                       else f"AUC = {auc:.3f}")
            lines.append(f"- `{outcome}`: **{set_name}** ({label}) — "
                         f"{verdict}")
    for note in set_notes:
        lines += ["", note]
    if len(sets) > 1:
        lines += comparison_lines(metrics_rows, metrics_header, score="auc",
                                  label="AUC", se="auc_folds_se")
    if control_names:
        lines += ["", f"Fitted with {len(control_names)} control column(s): "
                  f"{', '.join('`' + c + '`' for c in control_names)}, so "
                  f"`delta_auc_over_controls` is what the language added "
                  f"over them alone."]
        for note in control_notes:
            lines.append(f"- {note}")
    lines += ["", "Details: `classifier_cv_metrics.csv`, "
              "`classifier_per_class.csv` (where the model is good and where "
              "it is useless), `classifier_confusion.csv` (what the mistakes "
              "were), `classifier_folds.csv`, "
              "`classifier_coefficients.csv`, and one reusable model per "
              "feature set under `models/`."]
    return "\n".join(lines)


def _write_predictions(path: Path, table, outcome_cols, predictions,
                       encoding: str) -> None:
    """
    Every row's observed class beside two predictions of it.

    ``pred_`` comes from the saved model, so scoring this same table through
    :func:`apply_classifier_csv` reproduces those columns exactly. ``oof_``
    comes from cross-validation -- the prediction made by a model that had
    not seen the row -- which is the column to use for anything downstream,
    and the fold it was held out in is beside it so a suspicious result can
    be traced to one split.
    """
    key = [c for c in ("text_id",) if c in table.columns] or \
        [table.columns[0]]
    header = list(key)
    for outcome in outcome_cols:
        header += [outcome, f"pred_{outcome}", f"prob_{outcome}",
                   f"oof_pred_{outcome}", f"oof_prob_{outcome}",
                   f"fold_{outcome}"]
    rows = []
    for i in range(len(table)):
        row = [table[c].iloc[i] for c in key]
        for outcome in outcome_cols:
            block = predictions[outcome]
            row.append(block["observed"][i])
            for field in ("pred", "prob", "oof_pred", "oof_prob", "fold"):
                value = block[field][i]
                row.append("" if value is None else value)
        rows.append(row)
    _write_csv(path, header, rows, encoding)


def _load_model(model_json: PathLike) -> dict:
    """
    Read a classifier, refusing anything that is not one -- in words.

    The refusals happen while the user is holding the file and can act, not
    halfway through scoring a corpus, and they are what the library shows
    when someone imports a model, so they are written for a person rather
    than for a stack trace.
    """
    path, model, broken = load_model_doc(
        model_json, kind="taters-classifier-model",
        fmt=CLASSIFIER_MODEL_FORMAT, noun="classification model",
        step="classification",
        # a ridge model names itself, has coefficients, and would happily
        # score here while meaning something else entirely.
        wrong_kind_hint=lambda found: (
            " That is a prediction model for measurements -- score it with "
            "the ridge step." if found == "taters-ridge-model" else ""))
    predictors = model["predictors"]
    outcomes = model["outcomes"]
    for name, block in outcomes.items():
        if not isinstance(block, dict):
            raise broken(f"the entry for {name!r} is not a model")
        classes = block.get("classes")
        if not isinstance(classes, list) or len(classes) < 2:
            raise broken(f"{name!r} names fewer than two classes")
        kept = block.get("kept")
        if not isinstance(kept, list) or not kept:
            raise broken(f"{name!r} keeps no predictors")
        if max(kept) >= len(predictors) or min(kept) < 0:
            raise broken(f"{name!r} points at predictors that do not exist")
        sizes = {len(kept), len(block.get("mu") or []),
                 len(block.get("sigma") or [])}
        if len(sizes) != 1:
            raise broken(
                f"{name!r} has mismatched sizes for its predictors, means "
                f"and scales")
        fits = block.get("models")
        # one fit per class, except for a two-class outcome -- there, the
        # second class is just the complement of the first, so we store a
        # null for it.
        if not isinstance(fits, list) or len(fits) != len(classes):
            raise broken(
                f"{name!r} has {len(fits) if isinstance(fits, list) else 'no'}"
                f" fitted class model(s) for {len(classes)} class(es)")
        if all(f is None for f in fits):
            raise broken(f"{name!r} holds no fitted coefficients")
        for f in fits:
            if f is None:
                continue
            if not isinstance(f, dict) or f.get("intercept") is None:
                raise broken(f"a class model for {name!r} has no intercept")
            if len(f.get("coef") or []) != len(kept):
                raise broken(
                    f"a class model for {name!r} has "
                    f"{len(f.get('coef') or [])} coefficient(s) for "
                    f"{len(kept)} predictor(s)")
    return model


def apply_classifier_csv(
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
    Classify a new table with a saved model.

    Predictors are matched **by name**, in any column order; every other
    column passes through as an identifier. A predictor the new table lacks
    is refused by name rather than filled in -- a model scored on a feature
    it never sees is not the model that was validated.

    Parameters
    ----------
    model_json
        A model written by :func:`fit_classifier_csv` (or a folder holding
        exactly one).
    input_csv
        The table to score: an analysis table, or any feature table carrying
        the model's predictors.
    out_csv
        Defaults to ``<input stem>_classifier_predictions.csv`` beside the input.
    overwrite_existing
        When False (default) and the predictions already exist, they are
        returned untouched instead of recomputed.
    rounding
        Decimal places in the probabilities.

    Returns
    -------
    Path
        The predictions: the input's identifier columns plus, per outcome,
        the predicted class, the probability behind it, and one probability
        column per class -- because "which class" hides how close the call
        was, and a 0.51 verdict and a 0.99 one should not read the same.
    """

    got = prepare_apply(
        model_json=model_json, load=_load_model, input_csv=input_csv,
        out_csv=out_csv, default_suffix="_classifier_predictions.csv",
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        verbose=verbose, encoding=encoding, id_cols=id_cols)
    if isinstance(got, Path):
        return got
    model, design = got.model, got.design
    columns: dict = {}
    skipped = {}
    for outcome, block in model["outcomes"].items():
        classes = block["classes"]
        labels, probs = classify_rows(design, block["kept"], block["mu"],
                                      block["sigma"], block["models"],
                                      classes, rounding)
        name = output_label(model, outcome)
        # the class as fitted ("1") or as the model was told to call it
        # ("patient"). there's one relabel in the model file and we apply it
        # here and in the column headings alike, so that the two can't
        # disagree.
        columns[f"pred_{name}"] = [None if v is None
                                   else class_label(model, outcome, v)
                                   for v in labels]
        columns[f"prob_{name}"] = probs
        full = _all_probabilities(design, block, classes, rounding)
        for c in classes:
            columns[f"p_{name}_{class_label(model, outcome, c)}"] = full[c]
        skipped[name] = sum(1 for v in labels if v is None)

    return write_predictions(got, columns, skipped, encoding=encoding,
                             on_progress=on_progress, verbose=verbose,
                             tag="classifier")


def _all_probabilities(cells, block, classes, rounding: int):
    """One probability column per class, ``None`` where a row was unscored."""
    import numpy as np

    kept = list(block["kept"])
    fits = [None if e is None
            else (float(e["intercept"]), np.asarray(e["coef"], dtype=float))
            for e in block["models"]]
    design = np.asarray(cells, dtype=float)[:, kept]
    usable = ~np.isnan(design).any(axis=1)
    out = {c: [None] * design.shape[0] for c in classes}
    if usable.any():
        z = (design[usable] - np.asarray(block["mu"], dtype=float)) \
            / np.asarray(block["sigma"], dtype=float)
        probabilities = _probabilities(z, fits)
        for out_i, row_i in enumerate(np.flatnonzero(usable)):
            for c_i, c in enumerate(classes):
                out[c][row_i] = round(float(probabilities[out_i, c_i]),
                                      rounding)
    return out


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"fit": fit_classifier_csv, "apply": apply_classifier_csv},
    description='Cross-validated classification over an analysis table, and scoring with a saved classifier.',
    aliases={},
    legacy={"--no-zscore": ["--zscore", "false"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
