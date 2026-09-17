"""
The MEM topic model: exact PCA themes, and a model that travels.

The contract that carries the feature: a saved model applied to its own
training texts reproduces the training scores byte for byte -- which is only
true when apply rebuilds the matrix with the same scan, weights it the same
way, and standardizes with the TRAINING means and deviations rather than the
new data's. Most tests here lean on that roundtrip.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from taters.stats.pca import varimax as _varimax
from taters.text.topic_model_mem import (
    MODEL_FORMAT,
    apply_mem_model,
    topic_model_mem,
)
from csvhelpers import _read

def matrix_of(out) -> Path:
    """Where a step put the matrix it built for itself.

    Derived from the output's *stem*, not a fixed `matrix/` beside it: two
    steps writing into one folder would otherwise share one matrix folder,
    which is how three topic models came to build over each other's and how an
    apply left a frozen vocabulary for a later fit to pick up.
    """
    out = Path(out)
    return out.with_name(f"{out.stem}_matrix")


FOOD = "potato gravy butter salt dinner".split()
WORK = "meeting deadline email boss office".split()


def _corpus_csv(tmp_path, docs, name="corpus.csv"):
    src = tmp_path / name
    src.write_text("text_id,text\n" +
                   "".join(f"d{i},{t}\n" for i, t in enumerate(docs)),
                   encoding="utf-8")
    return src


def _two_theme_docs(n=80, cross=0.0, seed=11):
    """Documents with two blocks of words that rise and fall together --
    every FOOD word tracks intensity ``a``, every WORK word intensity ``b``.
    ``cross`` couples the intensities, manufacturing the general factor that
    only varimax rotation untangles. "and"/"the" appear exactly once per
    document: constant columns, which the fit must drop and the model must
    still carry for matching."""
    rng = random.Random(seed)
    docs = []
    for _ in range(n):
        a = rng.randint(1, 9)
        b = max(1, int(round(a * cross)) + rng.randint(1, 9))
        words = []
        for w in FOOD:
            words += [w] * max(0, a + rng.randint(-1, 1))
        for w in WORK:
            words += [w] * max(0, b + rng.randint(-1, 1))
        words += ["and", "the"]
        rng.shuffle(words)
        docs.append(" ".join(words))
    return docs


def _fitted(tmp_path, docs, **mem_kwargs):
    """corpus -> fitted MEM; returns paths dict.

    The frequency list and the matrix used to be built here and handed in.
    MEM builds its own now -- every topic model wants a different one, and a
    shared matrix meant two of them clobbering each other's file -- so this
    passes the corpus and then looks in the `matrix/` folder beside the
    results for what MEM made.
    """
    src = _corpus_csv(tmp_path, docs)
    scores = Path(topic_model_mem(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "mem.csv", overwrite_existing=True,
        min_freq=1, min_obs_pct=0, min_token_count=1,
        **mem_kwargs))
    matrix = matrix_of(scores)
    return {"src": src, "gathered": tmp_path / "gathered.csv",
            "freq": matrix / "freq_list.csv", "dtm": matrix / "dtm.csv",
            "scores": scores,
            "model": tmp_path / "mem_model.json",
            "loadings": tmp_path / "mem_loadings.csv",
            "eigen": tmp_path / "mem_eigenvalues.csv",
            "variance": tmp_path / "mem_theme_variance.csv"}


# ---------------------------------------------------------------------------
# the roundtrip contract
# ---------------------------------------------------------------------------

def test_applying_a_model_to_its_training_texts_reproduces_the_scores(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs())
    out = apply_mem_model(model_json=paths["model"],
                          analysis_csv=paths["gathered"],
                          out_features_csv=tmp_path / "reapplied.csv",
                          overwrite_existing=True)
    assert Path(out).read_bytes() == paths["scores"].read_bytes()


def test_apply_standardizes_with_the_training_statistics_not_the_new_data(tmp_path):
    """Score a SUBSET of the training rows: the subset's own means and
    deviations differ from the training ones, so an apply that re-derived
    them from the new data would drift -- silently, plausibly, wrongly."""
    paths = _fitted(tmp_path, _two_theme_docs())
    rows = paths["gathered"].read_text(encoding="utf-8-sig").splitlines()
    subset = tmp_path / "subset.csv"
    subset.write_text("\n".join(rows[:1] + rows[1:12]) + "\n", encoding="utf-8")

    out = apply_mem_model(model_json=paths["model"], analysis_csv=subset,
                          out_features_csv=tmp_path / "subset_scores.csv",
                          overwrite_existing=True)
    full = {r["text_id"]: r for r in _read(paths["scores"])}
    for row in _read(out):
        assert row == full[row["text_id"]], (
            "a training row scored differently through the saved model"
        )


def test_new_texts_with_unseen_words_score_on_the_saved_themes(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs())
    new = _corpus_csv(tmp_path, ["gravy dinner potato butter salt salt",
                                 "zebra quantum flugelhorn"], name="new.csv")
    out = apply_mem_model(model_json=paths["model"], csv_path=new,
                          text_cols=["text"], id_cols=["text_id"],
                          gathered_csv=tmp_path / "new_gathered.csv",
                          out_features_csv=tmp_path / "new_scores.csv",
                          overwrite_existing=True)
    rows = _read(out)
    assert len(rows) == 2
    assert all(r["text_id"] and r["Theme_1"] != "" for r in rows)
    # the all-unseen document matched nothing at all, so every cell is 0 but
    # still a finite number
    float(rows[1]["Theme_1"])


# ---------------------------------------------------------------------------
# the themes themselves
# ---------------------------------------------------------------------------

def test_two_planted_themes_are_recovered_and_ordered(tmp_path):
    """A shared 'verbosity' factor couples the two families, which smears the
    unrotated first component across both; varimax must pull them apart. The
    eigenvalue table must come out in descending order."""
    paths = _fitted(tmp_path, _two_theme_docs(cross=0.6), n_components=2)
    loads = _read(paths["loadings"])
    # the themes' own variance, which is what "ordered" is about here
    eig = [float(r["variance"]) for r in _read(paths["variance"])]
    assert eig == sorted(eig, reverse=True) and len(eig) == 2
    # and the spectrum is its own table, descending too and strictly larger at
    # the top: rotation moves variance down off the leading axis
    spectrum = _read(paths["eigen"])
    raw = [float(r["eigenvalue"]) for r in spectrum]
    assert raw == sorted(raw, reverse=True)
    assert raw[0] > eig[0], "rotation should have spread the first axis out"
    # the spectrum runs past the cut, so the curve can be seen crossing
    assert len(raw) > 2, "only the kept ranks were written; the curve is cut off"
    assert [r["kept"] for r in spectrum[:2]] == ["yes", "yes"]
    assert spectrum[2]["kept"] == ""

    def family_of_theme(col):
        top = sorted(loads, key=lambda r: -abs(float(r[col])))[:4]
        fams = {("food" if r["term"] in FOOD else
                 "work" if r["term"] in WORK else "other") for r in top}
        return fams

    assert family_of_theme("Theme_1") in ({"food"}, {"work"})
    assert family_of_theme("Theme_2") in ({"food"}, {"work"})
    assert family_of_theme("Theme_1") != family_of_theme("Theme_2"), (
        "rotation failed to separate the planted families"
    )
    # our sign convention: the defining terms of each theme load positive
    for col in ("Theme_1", "Theme_2"):
        top = max(loads, key=lambda r: abs(float(r[col])))
        assert float(top[col]) > 0


def test_kaiser_picks_the_component_count_automatically(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs(), k_selection="kaiser",
                    kaiser_cutoff=1.0)  # two planted blocks
    eig = _read(paths["variance"])
    assert len(eig) == 2, "two planted blocks; Kaiser must find two themes"
    header = _read(paths["scores"])[0]
    assert "token_count" in header and "Theme_1" in header


def test_the_spectrum_and_the_theme_variances_are_separate_tables(tmp_path):
    """
    They were one table with a shared `rank` column, and that column made a
    claim that is false: theme 3 is not built from eigenvector 3. Varimax
    rotates the kept axes within the space they span, so every rotated theme
    is a remix of all of them. Both lists come out sorted descending, which is
    the only thing they share -- and side by side that coincidence reads as a
    correspondence.

    So: two files, and neither carries a column that would let them be lined
    up by row.
    """
    paths = _fitted(tmp_path, _two_theme_docs(), n_components=2)

    spectrum = _read(paths["eigen"])
    variance = _read(paths["variance"])
    assert set(spectrum[0]) == {"rank", "eigenvalue", "chance_threshold", "kept"}
    assert set(variance[0]) == {"theme", "variance", "pct_variance"}
    # nothing in either table invites a join with the other
    assert "theme" not in spectrum[0]
    assert "rank" not in variance[0] and "eigenvalue" not in variance[0]
    # and they are not even the same length, which is the honest shape: the
    # spectrum has a rank for every term, the themes only for what was kept
    assert len(spectrum) > len(variance)


def test_an_explicit_component_count_is_honored(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs(), n_components=3)
    assert [r["theme"] for r in _read(paths["variance"])] == \
        ["Theme_1", "Theme_2", "Theme_3"]


def test_varimax_rotation_is_orthogonal_and_variance_preserving():
    import numpy as np

    rng = np.random.default_rng(3)
    loadings = rng.normal(size=(20, 4))
    rotated, R = _varimax(loadings)
    assert np.allclose(R @ R.T, np.eye(4), atol=1e-8)
    assert np.isclose((rotated ** 2).sum(), (loadings ** 2).sum())


# ---------------------------------------------------------------------------
# the instrument file
# ---------------------------------------------------------------------------

def test_the_model_records_the_instrument_but_not_the_runtime(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs())
    model = json.loads(paths["model"].read_text(encoding="utf-8"))
    assert model["kind"] == "taters-mem-model"
    assert model["format"] == MODEL_FORMAT
    assert model["text"] == {"lemmatize": False, "pos_tagged": False,
                             "engine": "nltk", "tokenizer": "potts",
                             "stanza_lang": "en", "keep_punctuation": False}
    assert "device" not in json.dumps(model), (
        "where a model RUNS is a per-machine choice, never part of the model"
    )
    assert len(model["matrix"]["terms"]) == len(model["matrix"]["idf"])
    assert model["model"]["n_documents"] == 80


def test_a_pos_tagged_model_roundtrips(tmp_path):
    """Tagged terms carry the \\x1f separator through the JSON and back."""
    paths = _fitted(tmp_path, _two_theme_docs(), pos_tagged=True)
    out = apply_mem_model(model_json=paths["model"],
                          analysis_csv=paths["gathered"],
                          out_features_csv=tmp_path / "reapplied.csv",
                          overwrite_existing=True)
    assert Path(out).read_bytes() == paths["scores"].read_bytes()
    loads = _read(paths["loadings"])
    assert "pos" in loads[0], "tagged loadings must name each term's tag"


def test_the_matrix_cannot_disagree_with_the_vocabulary_any_more(tmp_path):
    """
    This used to be `test_a_vocabulary_mismatch_is_refused_not_modeled`: MEM was
    handed a matrix and a frequency list built by other steps, re-derived the
    vocabulary, and refused when the two did not line up -- because nothing
    guaranteed they had been made with the same settings.

    MEM builds both itself now, so there is no mismatch left to refuse. What
    replaces the refusal is this: ask for a different vocabulary and you get a
    different matrix, built to match, rather than an error or -- far worse --
    the wrong terms scored quietly.
    """
    src = _corpus_csv(tmp_path, _two_theme_docs())
    widths = {}
    for name, top_n in (("narrow", 3), ("wide", 12)):
        out = Path(topic_model_mem(
            csv_path=src, text_cols=["text"], id_cols=["text_id"],
            out_features_csv=tmp_path / name / "mem.csv",
            min_freq=1, min_obs_pct=0, min_token_count=1,
            vocab_top_n=top_n, overwrite_existing=True))
        header = (matrix_of(out) / "dtm.csv").read_text(
            encoding="utf-8-sig").splitlines()[0]
        widths[name] = len(header.split(",")) - 2      # less text_id, token_count

    # not an exact count: `vocab_top_n` is a cut on a ranked list and ties at
    # the boundary come along with it. what has to hold is that the setting
    # reached the matrix at all, and that the two runs did not share a file.
    assert widths["narrow"] < widths["wide"], widths


def test_a_model_from_a_newer_taters_is_refused(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs())
    model = json.loads(paths["model"].read_text(encoding="utf-8"))
    model["format"] = MODEL_FORMAT + 1
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="newer"):
        apply_mem_model(model_json=newer, analysis_csv=paths["gathered"],
                        out_features_csv=tmp_path / "x.csv",
                        overwrite_existing=True)


def test_scores_are_the_standardized_matrix_through_the_saved_projection(tmp_path):
    """Recompute one document by hand from the model file: fit and apply
    share the projection code, so the roundtrip alone cannot catch that code
    going wrong -- this pins the arithmetic to the published instrument."""
    import numpy as np

    paths = _fitted(tmp_path, _two_theme_docs(), n_components=2)
    model = json.loads(paths["model"].read_text(encoding="utf-8"))
    kept = np.asarray(model["model"]["kept"])
    mu = np.asarray(model["model"]["mu"])
    sigma = np.asarray(model["model"]["sigma"])
    P = np.asarray(model["model"]["projection"])

    dtm_rows = _read(paths["dtm"])
    score_rows = _read(paths["scores"])
    columns = [c for c in dtm_rows[0] if c not in ("text_id", "token_count")]
    for i in (0, 7, 33):
        x = np.asarray([float(dtm_rows[i][c]) for c in columns])
        want = ((x[kept] - mu) / sigma) @ P
        got = [float(score_rows[i][f"Theme_{j + 1}"]) for j in range(P.shape[1])]
        assert got == pytest.approx(want, abs=10 ** -3), f"row {i}"


# ---------------------------------------------------------------------------
# the model as a library asset
# ---------------------------------------------------------------------------

def test_the_import_gate_admits_real_models_and_names_impostors(tmp_path):
    """A model is an instrument someone publishes from, so the library import
    must be exactly as strict as the loader: a fitted MEM model passes, a
    stats.pca model file is refused BY NAME as the wrong kind, arbitrary JSON
    is refused, and a truncated model is refused as damaged.

    Every kind of model shares one library folder, so the first two refusals
    come from the shared gate (which names the kind the file claims to be)
    and the damage refusal comes from the MEM loader itself -- the gate
    delegates once it knows which loader to ask."""
    from taters.helpers.library import asset_problem, kind_by_id

    kind = kind_by_id("models")
    paths = _fitted(tmp_path, _two_theme_docs())
    assert asset_problem(paths["model"], kind) == ""

    pca_ish = tmp_path / "other_procedure.json"
    pca_ish.write_text(json.dumps({"kind": "taters-pca-model", "format": 1,
                                   "features": ["a"], "model": {}}),
                       encoding="utf-8")
    problem = asset_problem(pca_ish, kind)
    assert "taters-pca-model" in problem
    assert "not a model this build can score" in problem

    not_ours = tmp_path / "settings.json"
    not_ours.write_text('{"theme": "dark", "font": 12}', encoding="utf-8")
    assert "not a Taters model file at all" in asset_problem(not_ours, kind)

    truncated = json.loads(paths["model"].read_text(encoding="utf-8"))
    del truncated["model"]["projection"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(truncated), encoding="utf-8")
    assert "damaged or incomplete" in asset_problem(broken, kind)


def test_a_model_with_disagreeing_matrices_is_refused(tmp_path):
    paths = _fitted(tmp_path, _two_theme_docs())
    model = json.loads(paths["model"].read_text(encoding="utf-8"))
    model["model"]["mu"] = model["model"]["mu"][:-1]        # lop one off the end
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree in size"):
        apply_mem_model(model_json=bad, analysis_csv=paths["gathered"],
                        out_features_csv=tmp_path / "x.csv",
                        overwrite_existing=True)


def test_the_library_hands_a_folder_and_one_model_resolves(tmp_path):
    """The picker machinery passes lists, and its untouched default is the
    whole library folder. With exactly one model imported that scores; with
    two it refuses and says to pick; with none it says to import."""
    paths = _fitted(tmp_path, _two_theme_docs())
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    one = shelf / "study1_model.json"
    one.write_bytes(paths["model"].read_bytes())

    out = apply_mem_model(model_json=[str(shelf)],
                          analysis_csv=paths["gathered"],
                          out_features_csv=tmp_path / "via_folder.csv",
                          overwrite_existing=True)
    assert Path(out).read_bytes() == paths["scores"].read_bytes()

    (shelf / "study2_model.json").write_bytes(paths["model"].read_bytes())
    with pytest.raises(ValueError, match="can only use one"):
        apply_mem_model(model_json=[str(shelf)],
                        analysis_csv=paths["gathered"],
                        out_features_csv=tmp_path / "y.csv",
                        overwrite_existing=True)

    # ValueError here, same as every other step that takes a model: an empty
    # choice is a wrong value, not a missing file (one helper handles all of them
    # now)
    with pytest.raises(ValueError, match="Import one"):
        apply_mem_model(model_json=[], analysis_csv=paths["gathered"],
                        out_features_csv=tmp_path / "z.csv",
                        overwrite_existing=True)


def test_the_theme_table_carries_a_settings_record(tmp_path):
    """
    A feature table by declaration, so its Theme_k columns are legitimate
    predictors -- but nothing wrote a record beside it, so every model fitted
    on themes was told to "re-run the extraction so a record is written",
    advice that could not be followed.
    """
    from taters.helpers import provenance as pv

    paths = _fitted(tmp_path, _two_theme_docs())
    rec = pv.read(paths["scores"])
    assert rec is not None, "the topic model wrote no record"
    assert rec["instrument"], "the fit's own settings are the instrument"
    # the corpus is the binding now. it used to be the matrix and the frequency
    # list, because other steps built those and the chain had to walk back
    # through their records to reach the gather; MEM makes them itself, so the
    # gather is one hop away and this is the same binding every text step uses.
    assert "csv_path" in rec["binding"], "the corpus is what the chain walks to"


def test_the_model_carries_the_punctuation_rule_and_old_models_keep_theirs(
        tmp_path):
    """A saved model scans new text with the rule its vocabulary was built
    under. A model from before the rule existed was built with punctuation
    in, so it keeps it -- silently dropping it would move every count."""
    from taters.text.topic_model_mem import _load_model

    paths = _fitted(tmp_path, _two_theme_docs(), n_components=2)
    model = json.loads(paths["model"].read_text(encoding="utf-8"))
    assert model["text"]["keep_punctuation"] is False

    del model["text"]["keep_punctuation"]
    old = tmp_path / "old_model.json"
    old.write_text(json.dumps(model), encoding="utf-8")
    assert "keep_punctuation" not in _load_model(old)["text"]
    docs = ["food food . . . work"]
    src = _corpus_csv(tmp_path, docs, "new.csv")
    from taters.text.topic_model_mem import apply_mem_model
    scored = _read(apply_mem_model(model_json=old, csv_path=src,
                                   text_cols=["text"], id_cols=["text_id"],
                                   out_features_csv=tmp_path / "old_scores.csv",
                                   overwrite_existing=True))
    fresh = _read(apply_mem_model(model_json=paths["model"], csv_path=src,
                                  text_cols=["text"], id_cols=["text_id"],
                                  out_features_csv=tmp_path / "new_scores.csv",
                                  gathered_csv=tmp_path / "g2.csv",
                                  overwrite_existing=True))
    # the old model counts the three periods as tokens, the new one doesn't
    assert int(scored[0]["token_count"]) == 6
    assert int(fresh[0]["token_count"]) == 3


def test_the_kaiser_rule_records_the_cutoff_it_used(tmp_path):
    """
    "Chosen by Kaiser" without the number is not a decision anyone can argue
    with -- especially since the right number depends on the shape of the
    matrix, and the textbook 1.0 is far below the level chance reaches on a
    wide one.
    """
    import json

    paths = _fitted(tmp_path, _two_theme_docs(), k_selection="kaiser",
                    kaiser_cutoff=1.5)
    model = json.loads(Path(paths["model"]).read_text(encoding="utf-8"))
    retention = model["model"]["retention"]
    assert retention["rule"] == "kaiser"
    assert retention["cutoff"] == 1.5


def test_parallel_analysis_is_the_default_and_records_its_thresholds(tmp_path):
    """
    The default because it is the only rule that adapts to the shape of the
    matrix. A document-term matrix is wide, and chance alone makes large
    eigenvalues on a wide one -- at 938 documents by 515 terms the noise
    ceiling is about 3.0, so a fixed cutoff of 1.5 keeps a hundred themes
    that are not there.
    """
    import json

    paths = _fitted(tmp_path, _two_theme_docs())
    model = json.loads(Path(paths["model"]).read_text(encoding="utf-8"))
    retention = model["model"]["retention"]
    assert retention["rule"] == "parallel"
    assert len(retention["thresholds"]) == len(retention["unrotated_eigenvalues"])
    assert len(_read(paths["variance"])) == 2, "two planted blocks, two themes"
