"""
The generic PCA engine: exact, streaming, and a model that travels.

The MEM topic model rides on this module, so its heavier tests (planted
themes, rotation separation, Kaiser) live in test_topic_model_mem.py and
exercise this engine through the import. Here: the generic fit/apply over an
arbitrary feature table, the by-name feature matching, and the streaming
accumulation's chunk arithmetic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from taters.stats import pca
from csvhelpers import _read


def _table(tmp_path, n=40, name="features.csv", ids=("row_id",), shuffle_cols=None):
    import random
    rng = random.Random(5)
    cols = ["alpha", "beta", "gamma", "delta"]
    rows = []
    for i in range(n):
        a, b = rng.uniform(0, 10), rng.uniform(0, 10)
        rows.append({"row_id": f"r{i}", "batch": f"b{i % 3}",
                     "alpha": round(a, 3), "beta": round(a * 2 + rng.uniform(0, 1), 3),
                     "gamma": round(b, 3), "delta": round(b * 3 + rng.uniform(0, 1), 3)})
    header = [*ids, *(shuffle_cols or cols)]
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def test_fit_then_apply_on_the_training_table_is_byte_identical(tmp_path):
    src = _table(tmp_path)
    scores = pca.fit_pca_csv(src, n_components=2, overwrite_existing=True)
    out = pca.apply_pca_csv(tmp_path / "features_pca_model.json", src,
                            out_scores_csv=tmp_path / "reapplied.csv",
                            overwrite_existing=True)
    assert Path(out).read_bytes() == Path(scores).read_bytes()
    header = list(_read(scores)[0])
    assert header == ["row_id", "Component_1", "Component_2"]


def test_apply_matches_features_by_name_not_by_position(tmp_path):
    """A new table with the feature columns shuffled and an extra identifier
    must score identically -- position-based matching would silently feed
    delta's values into alpha's slot."""
    src = _table(tmp_path)
    scores = pca.fit_pca_csv(src, n_components=2, overwrite_existing=True)

    moved = _table(tmp_path, name="shuffled.csv", ids=("row_id", "batch"),
                   shuffle_cols=["delta", "alpha", "gamma", "beta"])
    out = pca.apply_pca_csv(tmp_path / "features_pca_model.json", moved,
                            overwrite_existing=True)
    want = {r["row_id"]: (r["Component_1"], r["Component_2"]) for r in _read(scores)}
    got = _read(out)
    assert list(got[0]) == ["row_id", "batch", "Component_1", "Component_2"], (
        "non-feature columns must pass through as identifiers"
    )
    for row in got:
        assert (row["Component_1"], row["Component_2"]) == want[row["row_id"]]


def test_a_table_missing_a_model_feature_is_refused(tmp_path):
    src = _table(tmp_path)
    pca.fit_pca_csv(src, n_components=2, overwrite_existing=True)
    partial = _table(tmp_path, name="partial.csv",
                     shuffle_cols=["alpha", "beta", "gamma"])
    with pytest.raises(ValueError, match="missing"):
        pca.apply_pca_csv(tmp_path / "features_pca_model.json", partial,
                          overwrite_existing=True)


def test_chunked_accumulation_matches_direct_arithmetic(tmp_path, monkeypatch):
    """The streaming pass folds rows in BLAS-sized chunks; a chunk size that
    does not divide the row count leaves a partial tail whose loss would be
    silent. Force many small chunks and compare to numpy computed directly."""
    import numpy as np

    monkeypatch.setattr(pca, "_CHUNK_ROWS", 7)
    src = _table(tmp_path, n=23)
    n, sums, cross = pca.stream_moments(src, encoding="utf-8-sig", skip_cols=1)

    X = np.asarray([[float(r[c]) for c in ("alpha", "beta", "gamma", "delta")]
                    for r in _read(src)])
    assert n == 23
    assert np.allclose(sums, X.sum(axis=0))
    assert np.allclose(cross, X.T @ X)


def test_loadings_match_sklearn_to_machine_precision():
    """The one cross-check against an independent implementation: unrotated
    loadings equal scikit-learn's full-SVD PCA on standardized data, up to
    sklearn's n-1 variance convention and per-column sign."""
    np = pytest.importorskip("numpy")
    sk = pytest.importorskip("sklearn.decomposition")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 30)) \
        + rng.normal(size=(500, 30)) * 0.5
    n = len(X)
    kept, mu, sigma, loadings, _p, _e, _pc, _r = pca.fit_axes(
        n, X.sum(axis=0), X.T @ X, n_components=5, rotation=False)

    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    ours = loadings
    ref = sk.PCA(n_components=5, svd_solver="full").fit(Z)
    theirs = ref.components_.T * np.sqrt(ref.explained_variance_)
    theirs = theirs * np.sqrt((n - 1) / n)          # their n-1, our n
    for j in range(5):
        i = int(np.abs(ours[:, j]).argmax())
        if np.sign(theirs[i, j]) != np.sign(ours[i, j]):
            theirs[:, j] *= -1
    assert float(np.abs(ours - theirs).max()) < 1e-10


def test_start_col_carries_leading_identifiers_through(tmp_path):
    src = _table(tmp_path, ids=("row_id", "batch"))
    scores = pca.fit_pca_csv(src, start_col=3, n_components=2,
                             overwrite_existing=True)
    rows = _read(scores)
    assert list(rows[0]) == ["row_id", "batch", "Component_1", "Component_2"]
    assert rows[0]["batch"] == "b0"


def test_the_width_warning_fires_only_when_memory_would_hurt():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pca.warn_if_wide(250)                        # the DTM default: silent
    with pytest.warns(UserWarning, match="GB"):
        pca.warn_if_wide(20_000)


def test_a_newer_model_format_is_refused(tmp_path):
    src = _table(tmp_path)
    pca.fit_pca_csv(src, n_components=2, overwrite_existing=True)
    model_path = tmp_path / "features_pca_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["format"] = pca.PCA_MODEL_FORMAT + 1
    model_path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        pca.apply_pca_csv(model_path, src, overwrite_existing=True,
                          out_scores_csv=tmp_path / "x.csv")


def _varimax_criterion(L):
    """The (gamma=1) varimax objective on KAISER-NORMALIZED loadings: the
    summed variance of squared loadings within each component."""
    import numpy as np

    h = np.sqrt((L ** 2).sum(axis=1))
    h[h == 0] = 1.0
    V = (L / h[:, None]) ** 2
    return float((V.var(axis=0)).sum())


def _heterogeneous_loadings(seed=9, p=120, k=6):
    """Unrotated loadings with wildly unequal communalities -- the regime
    where normalized and raw varimax genuinely disagree."""
    import numpy as np

    rng = np.random.default_rng(seed)
    A = rng.normal(size=(p, k))
    A *= rng.uniform(0.05, 2.0, size=p)[:, None]
    return A


def test_varimax_is_kaiser_normalized_like_r_and_spss():
    """psych::principal, stats::varimax and SPSS all rotate Kaiser-normalized
    loadings; rotating raw loadings lands on a measurably different solution
    (a real MEM run diverged from its R twin until this matched). The
    default rotation must therefore beat the unnormalized one on the
    normalized criterion -- and still be an exact orthogonal transform of
    the input."""
    import numpy as np

    A = _heterogeneous_loadings()
    L_norm, R_norm = pca.varimax(A)
    L_raw, R_raw = pca.varimax(A, normalize=False)

    assert _varimax_criterion(L_norm) > _varimax_criterion(L_raw) * 1.0001, (
        "the default rotation does not optimize the normalized criterion, "
        "so it cannot be Kaiser-normalized"
    )
    assert np.allclose(L_norm, A @ R_norm), (
        "normalization may only steer the CHOICE of rotation; the returned "
        "loadings must still be loadings @ R exactly"
    )


def test_varimax_iterates_to_convergence_not_to_a_sweep_cap():
    """The predecessor script stopped after 20 sweeps, which at MEM sizes is
    far from converged (worst theme correlated 0.77 vs 0.996 against R). The
    default budget must land on the same solution as an absurdly large one."""
    import numpy as np

    A = _heterogeneous_loadings(seed=3, p=300, k=12)
    L_default, _ = pca.varimax(A)
    L_exhaustive, _ = pca.varimax(A, q=20000, tol=1e-14)
    assert np.abs(L_default - L_exhaustive).max() < 1e-4

    # and 20 sweeps really isn't enough here, otherwise this test proves nothing
    L_capped, _ = pca.varimax(A, q=20)
    assert np.abs(L_capped - L_exhaustive).max() > 1e-3


def test_varimax_says_so_when_the_budget_runs_out():
    """Exhausting the sweep budget means the rotation is not the converged
    optimum; returning it silently is the plausible-numbers failure. A
    starved budget must warn -- and a normal fit must not cry wolf."""
    import warnings

    A = _heterogeneous_loadings()
    with pytest.warns(UserWarning, match="did not converge"):
        pca.varimax(A, q=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pca.varimax(A)                       # plenty of budget: silent


# ---------------------------------------------------------------------------
# components as a per-analysis, per-feature-set choice
# ---------------------------------------------------------------------------

def _two_sets(tmp_path, n=140):
    """A table with two feature sets that differ in exactly the way that makes
    one answer wrong for both: many collinear measures, and a few distinct
    ones."""
    import json

    import numpy as np

    rng = np.random.default_rng(4)
    base = rng.normal(size=(n, 2))
    # `dictionary`: 10 columns that are really just 2 things, so worth reducing
    dic = np.hstack([base[:, [k]] + rng.normal(scale=.25, size=(n, 5))
                     for k in range(2)])
    # `readability`: 3 columns that really are distinct, so not worth it
    read = rng.normal(size=(n, 3))
    dic_cols = [f"dic_{i}" for i in range(10)]
    read_cols = [f"read_{i}" for i in range(3)]
    path = tmp_path / "analysis_table.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "outcome"] + dic_cols + read_cols)
        for i in range(n):
            w.writerow([f"d{i}", round(2 * base[i, 0] + rng.normal(0, .5), 6)]
                       + [round(v, 6) for v in dic[i]]
                       + [round(v, 6) for v in read[i]])
    (tmp_path / "analysis_table_sets.json").write_text(json.dumps({
        "key": ["text_id"], "metadata": ["outcome"], "sources": {},
        "sets": {"dictionary": dic_cols, "readability": read_cols},
    }), encoding="utf-8")
    return path, dic_cols, read_cols


def test_one_feature_set_can_be_reduced_and_another_left_alone(tmp_path):
    """
    The case a boolean cannot express, and the reason this is not one switch
    for the run.

    A hundred dictionary categories are worth reducing; eight readability
    indices are not. They differ so much in number and in kind that one
    answer for both is the wrong answer for one of them.
    """
    from taters.stats.correlations import analyze_correlations

    table, dic_cols, read_cols = _two_sets(tmp_path)
    out = analyze_correlations(
        table_csv=table, outcome_cols=["outcome"], feature_sets="per_table",
        pca=["dictionary"], out_dir=tmp_path / "mixed", verbose=False)
    features = {r["feature"] for r in _read(out)}

    # the dictionary set came back as components...
    assert any(f.startswith("dictionary_Component_") for f in features)
    assert not (set(dic_cols) & features), "raw dictionary columns survived"
    # ...and the readability set came back as itself
    assert set(read_cols) <= features, "readability was reduced too"


def test_the_same_run_can_analyze_raw_here_and_components_there(tmp_path):
    """
    Per analysis, not per run. Raw variables read better in a correlation
    table, where every row is a measure you can name; a ridge over hundreds
    of collinear measures is exactly what components are for. Those are
    different questions about the same features.
    """
    from taters.stats.correlations import analyze_correlations
    from taters.stats.ridge import fit_ridge_csv

    table, dic_cols, read_cols = _two_sets(tmp_path)
    corr = analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                                pca="off", out_dir=tmp_path / "c",
                                verbose=False)
    ridge = fit_ridge_csv(table_csv=table, outcome_cols=["outcome"],
                          pca="all", out_dir=tmp_path / "r", verbose=False)

    assert set(dic_cols) <= {r["feature"] for r in _read(corr)}
    predictors = int(_read(ridge)[0]["n_predictors"])
    assert predictors < len(dic_cols) + len(read_cols)


def test_the_loadings_land_beside_the_analysis_that_made_them(tmp_path):
    """A component means nothing without the table saying what loads on it --
    and two analyses reducing different sets have genuinely different
    components under the same names, so the loadings cannot be shared."""
    from taters.stats.ridge import fit_ridge_csv

    table, _dic, _read_cols = _two_sets(tmp_path)
    # when we're not combining the tables, we get one joint reduction over
    # everything: the single "all" set, one loadings file
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  set_combos="none", out_dir=tmp_path / "r", verbose=False)
    loadings = tmp_path / "r" / "ridge_pca_loadings.csv"
    eigen = tmp_path / "r" / "ridge_pca_eigenvalues.csv"
    assert loadings.is_file() and eigen.is_file()
    rows = _read(loadings)
    assert {r["feature_set"] for r in rows} == {"all"}
    assert len(rows) > 0
    # combining the tables (the default) reduces each table on its own. this is
    # because a component mixing dictionary categories with readability indices
    # couldn't be given a sensible name, so the loadings come out per table
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  out_dir=tmp_path / "c", verbose=False)
    assert (tmp_path / "c" / "ridge_pca_loadings__dictionary.csv").is_file()
    assert (tmp_path / "c" / "ridge_pca_loadings__readability.csv").is_file()


def test_leaving_it_off_writes_no_loadings_and_changes_nothing(tmp_path):
    """Off is the default and has to mean off: same numbers as before this
    existed, and no stray files implying a reduction happened."""
    from taters.stats.correlations import analyze_correlations

    table, dic_cols, _read_cols = _two_sets(tmp_path)
    out = analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                               out_dir=tmp_path / "plain", verbose=False)
    assert set(dic_cols) <= {r["feature"] for r in _read(out)}
    assert not (tmp_path / "plain" / "correlations_pca_loadings.csv").exists()


def test_naming_a_feature_set_that_does_not_exist_is_refused(tmp_path):
    """A typo would otherwise reduce nothing and say nothing, and the results
    would look exactly like a deliberate choice to leave it raw."""
    from taters.stats.correlations import analyze_correlations

    table, _dic, _read_cols = _two_sets(tmp_path)
    with pytest.raises(ValueError) as e:
        analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                             feature_sets="per_table", pca=["dictionry"],
                             out_dir=tmp_path / "typo", verbose=False)
    assert "dictionry" in str(e.value)
    assert "dictionary" in str(e.value)


def test_a_feature_missing_for_most_rows_is_set_aside_not_filled(tmp_path):
    """
    Nothing is invented. A measure defined for a tenth of the corpus cannot
    contribute a direction the other nine tenths lie along, so it is dropped
    -- by the same rule and the same default the prediction steps use, which
    is what stops "how missing data is handled" from meaning three different
    things in one report.

    Filling with the column mean, which this used to do, is worse than it
    looks: it moves every row toward the center on that feature, which is
    exactly the direction a PCA is trying to measure.
    """
    import numpy as np
    from taters.stats.correlations import analyze_correlations

    rng = np.random.default_rng(9)
    n = 200
    base = rng.normal(size=(n, 2))
    good = np.hstack([base[:, [k]] + rng.normal(scale=.3, size=(n, 4))
                      for k in range(2)])
    y = 2 * base[:, 0] + rng.normal(scale=.5, size=n)
    path = tmp_path / "analysis_table.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "outcome"] + [f"f{i}" for i in range(8)]
                   + ["para_only"])
        for i in range(n):
            vals = [f"{v:.6f}" for v in good[i]]
            if i < 12:                       # a few scattered blanks
                vals[0] = ""
            w.writerow([f"d{i}", round(y[i], 6)] + vals
                       # present for a tenth of the rows, as a paragraph-pair
                       # measure is
                       + [f"{rng.normal():.4f}" if i % 10 == 0 else ""])

    analyze_correlations(table_csv=path, outcome_cols=["outcome"], pca="all",
                         out_dir=tmp_path / "o", verbose=False)
    section = (tmp_path / "o" / "_sections" / "30-correlations.md").read_text(
        encoding="utf-8")
    assert "para_only" in section and "set aside" in section
    assert "8 features" in section, "the sparse column was counted as usable"
    # the 12 rows with a scattered blank go unscored rather than guessed at,
    # and the report says how many
    assert "12 row(s) were missing at least one feature" in section
    loadings = _read(tmp_path / "o" / "correlations_pca_loadings.csv")
    assert "para_only" not in {r["feature"] for r in loadings}


def test_a_row_missing_one_feature_gets_no_component_score(tmp_path):
    """Left unscored, not imputed -- so it drops out of the analysis exactly
    as a row missing any other feature does, and the analysis's own row count
    describes the rows it actually used."""
    import numpy as np
    from taters.stats._common import reduce_sets

    rng = np.random.default_rng(2)
    n = 60
    base = rng.normal(size=(n, 2))
    cols = [f"f{i}" for i in range(6)]
    data = {c: np.hstack([base[:, [0]], base[:, [1]]])[:, i % 2]
               + rng.normal(scale=.2, size=n) for i, c in enumerate(cols)}
    data["f0"][:5] = np.nan          # five rows short of one feature

    sets, columns, notes, _reductions = reduce_sets(
        {"all": cols}, data, pca="all", n_components=0, rotation=True,
        out_stem=tmp_path / "x", encoding="utf-8-sig", rounding=6,
        verbose=False)
    scores = columns[sets["all"][0]]
    assert np.isnan(scores[:5]).all(), "missing rows were given scores anyway"
    assert not np.isnan(scores[5:]).any()
    assert any("5 row(s)" in line for line in notes)


def test_a_set_with_too_few_usable_features_is_refused(tmp_path):
    """After the sparse columns go, there may be nothing left to reduce. A
    refusal naming the emptiest beats a component built from one feature,
    which is a rename wearing a hat."""
    import numpy as np
    from taters.stats._common import reduce_sets

    n = 40
    data = {"good": np.arange(n, dtype=float),
            "holey": np.full(n, np.nan),
            "holey2": np.full(n, np.nan)}
    data["holey"][:2] = 1.0
    with pytest.raises(ValueError) as e:
        reduce_sets({"all": ["good", "holey", "holey2"]}, data, pca="all",
                    n_components=0, rotation=True, out_stem=tmp_path / "x",
                    encoding="utf-8-sig", rounding=6, verbose=False)
    assert "at least two" in str(e.value)
    assert "holey" in str(e.value)


def test_one_missingness_answer_governs_both_predictors_and_components(tmp_path):
    """
    A ridge that refuses to fit on a three-quarters-empty column should not
    then reduce that same column into a component and fit on *that*. The
    threshold the user answered once applies to both, so "how much missing is
    too much" means one thing per analysis rather than two.
    """
    import numpy as np
    from taters.stats.ridge import fit_ridge_csv

    rng = np.random.default_rng(4)
    n = 120
    base = rng.normal(size=(n, 2))
    good = np.hstack([base[:, [k % 2]] + rng.normal(scale=.3, size=(n, 3))
                      for k in range(2)])
    y = base[:, 0] * 3 + rng.normal(scale=.5, size=n)
    path = tmp_path / "analysis_table.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "age"] + [f"f{i}" for i in range(6)] + ["thin"])
        for i in range(n):
            # present for 70% of rows: kept at the 50% default, set aside
            # once the answer is 20%
            thin = f"{rng.normal():.4f}" if i % 10 < 7 else ""
            w.writerow([f"d{i}", round(y[i], 6)]
                       + [f"{v:.6f}" for v in good[i]] + [thin])

    def _loadings(out, **kw):
        fit_ridge_csv(table_csv=path, outcome_cols=["age"], pca="all",
                      n_folds=3, out_dir=out, out_models_dir=out / "m",
                      verbose=False, **kw)
        return {r["feature"] for r in _read(out / "ridge_pca_loadings.csv")}

    assert "thin" in _loadings(tmp_path / "lenient")
    assert "thin" not in _loadings(tmp_path / "strict", max_missing=0.2)


@pytest.mark.parametrize("analysis", ["correlations", "group_differences"])
def test_the_missingness_threshold_is_answerable_on_every_analysis(
        tmp_path, analysis):
    """
    The rule that sets a mostly-empty feature aside has a threshold, and it
    is reachable on the analyses that only reduce -- not just on the two that
    also fit models. Otherwise "set max_missing" would silently mean nothing
    on a correlation table, and the same corpus would reduce to different
    components depending on which analysis asked.
    """
    import numpy as np

    rng = np.random.default_rng(11)
    n = 120
    base = rng.normal(size=(n, 2))
    good = np.hstack([base[:, [k % 2]] + rng.normal(scale=.3, size=(n, 3))
                      for k in range(2)])
    path = tmp_path / "analysis_table.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "age", "cohort"]
                   + [f"f{i}" for i in range(6)] + ["thin"])
        for i in range(n):
            # present for 70% of rows: kept at the 0.5 default, set aside at 0.2
            thin = f"{rng.normal():.4f}" if i % 10 < 7 else ""
            w.writerow([f"d{i}", round(float(base[i, 0]), 6), f"c{i % 3}"]
                       + [f"{v:.6f}" for v in good[i]] + [thin])

    if analysis == "correlations":
        from taters.stats.correlations import analyze_correlations as run
        fixed = {"outcome_cols": ["age"]}
    else:
        from taters.stats.group_differences import (
            analyze_group_differences as run)
        fixed = {"group_col": "cohort"}

    def _loadings(out, **kw):
        run(table_csv=path, pca="all", out_dir=out, verbose=False,
            **fixed, **kw)
        return {r["feature"]
                for r in _read(out / f"{analysis}_pca_loadings.csv")}

    assert "thin" in _loadings(tmp_path / "lenient")
    assert "thin" not in _loadings(tmp_path / "strict", pca_max_missing=0.2)


def test_the_options_screen_offers_only_pca_values_the_analyses_accept():
    """
    The numpydoc type string is what the options screen turns into a picker.
    It read {"off", "all", "per_set"}: a closed set with one value nothing
    accepted (`per_set` failed at run time as a set name that does not
    exist) and without the list-of-set-names form, which is the useful one.
    Free text, and the parser takes every documented form.
    """
    from taters.stats._common import wanted_sets
    from taters.stats.classify import fit_classifier_csv
    from taters.stats.correlations import analyze_correlations
    from taters.stats.group_differences import analyze_group_differences
    from taters.stats.ridge import fit_ridge_csv
    from taters.ui.introspect import describe

    for fn in (analyze_correlations, analyze_group_differences,
               fit_ridge_csv, fit_classifier_csv):
        param = describe(fn).get("pca")
        assert param is not None and param.choices is None, fn.__name__
    sets = {"dictionaries": ["a", "b"], "readability": ["c", "d"]}
    assert wanted_sets("off", sets) == []
    assert set(wanted_sets("all", sets)) == set(sets)
    assert wanted_sets(["dictionaries"], sets) == ["dictionaries"]
    assert wanted_sets("dictionaries, readability", sets) == list(sets)
    with pytest.raises(ValueError):
        wanted_sets("per_set", sets)


def test_the_loadings_honor_the_analysis_rounding(tmp_path):
    """`reduce_sets` took `rounding` and wrote four places regardless, so an
    analysis asked for two decimals shipped loadings and eigenvalues at four
    beside results at two."""
    import numpy as np
    from taters.stats.correlations import analyze_correlations

    rng = np.random.default_rng(1)
    n = 60
    base = rng.normal(size=(n, 2))
    feats = np.hstack([base[:, [k % 2]] + rng.normal(scale=.3, size=(n, 2))
                       for k in range(2)])
    path = tmp_path / "analysis_table.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "y"] + [f"f{i}" for i in range(4)])
        for i in range(n):
            w.writerow([f"d{i}", f"{base[i, 0]:.6f}"]
                       + [f"{v:.6f}" for v in feats[i]])
    analyze_correlations(table_csv=path, outcome_cols=["y"], pca="all",
                         rounding=2, out_dir=tmp_path / "o", verbose=False)
    for name in ("correlations_pca_loadings.csv",
                 "correlations_pca_eigenvalues.csv"):
        for row in _read(tmp_path / "o" / name):
            for key, value in row.items():
                if key in ("feature_set", "feature", "component"):
                    continue
                assert len(value.split(".")[-1]) <= 2 if "." in value else True, \
                    (name, key, value)


def test_each_reduced_set_gets_its_own_loadings_table_named_as_the_results_are(tmp_path):
    """
    Two sets reduced separately used to share one wide loadings file under a
    common `Component_1..N` header, and it read as one PCA over both (a real
    report: "the PCA with cohesion ALSO included the dictionary elements?").
    One table per set, with the component names the results use, says what
    happened; the report says it in words too.
    """
    from taters.stats.ridge import fit_ridge_csv

    table, dic_cols, read_cols = _two_sets(tmp_path)
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  feature_sets="per_table", out_dir=tmp_path / "r",
                  verbose=False)
    out = tmp_path / "r"
    assert not (out / "ridge_pca_loadings.csv").exists()
    eigen = _read(out / "ridge_pca_eigenvalues.csv")
    for name, cols in (("dictionary", dic_cols), ("readability", read_cols)):
        rows = _read(out / f"ridge_pca_loadings__{name}.csv")
        assert {r["feature_set"] for r in rows} == {name}
        assert {r["feature"] for r in rows} == set(cols)
        components = [r["component"] for r in eigen if r["feature_set"] == name]
        assert components and components[0].startswith(f"{name}_")
        assert [k for k in rows[0] if k not in ("feature_set", "feature")] \
            == components
    section = (out / "_sections" / "40-ridge.md").read_text(encoding="utf-8")
    assert "reduced **on its own**" in section
    assert "ridge_pca_loadings__dictionary.csv" in section


# ---------------------------------------------------------------------------
# choosing the number of components
# ---------------------------------------------------------------------------

def _structure_plus_noise(n=150, noise_cols=20, seed=5):
    """Two planted factors of five features each, plus columns of pure noise.
    Chance alone lifts the leading noise eigenvalues past 1, which is exactly
    what the Kaiser rule cannot tell from structure."""
    import numpy as np

    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 2))
    blocks = np.hstack([f[:, [k]] + rng.normal(scale=0.4, size=(n, 5))
                        for k in range(2)])
    return np.hstack([blocks, rng.normal(size=(n, noise_cols))])


def test_parallel_analysis_keeps_the_structure_and_drops_the_chance():
    """
    Twenty noise columns among thirty: the Kaiser rule keeps several of them
    (their eigenvalues sit above 1 by chance), parallel analysis keeps the
    two planted factors and nothing else -- which is why it is the default.
    """
    x = _structure_plus_noise()
    parallel = pca.fit_in_memory(x, retain="parallel")
    kaiser = pca.fit_in_memory(x, retain="kaiser")
    assert parallel["n_components"] == 2
    assert kaiser["n_components"] > 2
    assert parallel["retention"]["rule"] == "parallel"
    # and we can read the decision off: the third eigenvalue fell short of chance
    eig, thr = parallel["retention"]["unrotated_eigenvalues"], parallel["retention"]["thresholds"]
    assert len(eig) == 3 and eig[2] <= thr[2] and eig[1] > thr[1]


def test_parallel_thresholds_fall_with_rank_and_repeat_under_a_seed():
    """The chance eigenvalues are largest for the first component and shrink
    by rank; the first is well above 1, which is the whole case against the
    Kaiser rule. Seeded, so a re-run decides the same way."""
    import numpy as np

    a = pca.parallel_thresholds(150, 30, draws=20, seed=1)
    b = pca.parallel_thresholds(150, 30, draws=20, seed=1)
    assert a.shape == (30,)
    assert a[0] > 1.3
    assert all(a[i] >= a[i + 1] for i in range(29))
    assert np.array_equal(a, b)


def test_parallel_thresholds_never_hold_a_random_table_whole(monkeypatch):
    """Memory-safe by construction: the random rows are generated in chunks
    and folded into a p-by-p cross-product, so no draw ever asks for more
    than ``chunk_rows`` rows at once -- there is never an n-by-p array."""
    import numpy as np

    asked = []
    real = np.random.default_rng

    class Spy:
        """A generator that notes how many rows each draw asked for."""

        def __init__(self, seed):
            self._rng = real(seed)

        def standard_normal(self, shape):
            asked.append(shape[0])
            return self._rng.standard_normal(shape)

    monkeypatch.setattr(np.random, "default_rng", Spy)
    small = pca.parallel_thresholds(1000, 8, draws=3, seed=2, chunk_rows=64)
    assert small.shape == (8,) and 1.0 < small[0] < 1.5
    assert asked and max(asked) <= 64, f"a chunk of {max(asked)} rows was drawn"
    assert sum(asked) == 3 * 1000


def test_an_unknown_retention_rule_is_refused():
    with pytest.raises(ValueError, match="retain must be one of"):
        pca.fit_in_memory(_structure_plus_noise(), retain="elbow")


def test_the_retention_decision_travels_with_the_model(tmp_path):
    """The model file says how many components and *why*, so a reader can
    argue with the count instead of trusting it."""
    import json

    x = _structure_plus_noise()
    path = tmp_path / "feats.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id"] + [f"f{i}" for i in range(x.shape[1])])
        for i, row in enumerate(x):
            w.writerow([f"d{i}"] + [f"{v:.5f}" for v in row])
    pca.fit_pca_csv(path, out_scores_csv=tmp_path / "s.csv",
                    out_model_json=tmp_path / "m.json")
    model = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert model["model"]["retention"]["rule"] == "parallel"
    assert model["model"]["retention"]["draws"] == pca.PARALLEL_DRAWS
    pca.fit_pca_csv(path, retain="kaiser", overwrite_existing=True,
                    out_scores_csv=tmp_path / "s.csv",
                    out_model_json=tmp_path / "m.json")
    model = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert model["model"]["retention"]["rule"] == "kaiser"


def test_the_report_says_how_the_component_count_was_chosen(tmp_path):
    from taters.stats.correlations import analyze_correlations

    table, _d, _r = _two_sets(tmp_path)
    analyze_correlations(table_csv=table, outcome_cols=["outcome"], pca="all",
                         out_dir=tmp_path / "p", verbose=False)
    section = (tmp_path / "p" / "_sections" / "30-correlations.md").read_text(encoding="utf-8")
    assert "chosen by parallel analysis" in section
    analyze_correlations(table_csv=table, outcome_cols=["outcome"], pca="all",
                         pca_retain="kaiser", out_dir=tmp_path / "k", verbose=False)
    section = (tmp_path / "k" / "_sections" / "30-correlations.md").read_text(encoding="utf-8")
    assert "chosen by the Kaiser rule" in section

    # the fitters take the same setting through their shared preamble
    from taters.stats.ridge import fit_ridge_csv

    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], pca="all",
                  pca_retain="kaiser", out_dir=tmp_path / "r", verbose=False)
    section = (tmp_path / "r" / "_sections" / "40-ridge.md").read_text(encoding="utf-8")
    assert "chosen by the Kaiser rule" in section
