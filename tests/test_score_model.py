"""
One entry for scoring with any saved model.

What these tests guard is the dispatch and the refusals, not the arithmetic
-- each model kind's own tests cover that. The dispatch matters because it
replaced three menu entries with one, and the refusals matter because the
thing that goes wrong in practice is a model needing feature columns the run
does not have, where the useful message names them.
"""

from __future__ import annotations

import csv
import json

import pytest

np = pytest.importorskip("numpy")

from taters.helpers.model_spec import rename_model  # noqa: E402
from taters.score_model import score_with_model  # noqa: E402
from taters.stats.classify import fit_classifier_csv  # noqa: E402
from taters.stats.ridge import fit_ridge_csv  # noqa: E402
from csvhelpers import _read, _write  # noqa: E402


def _training_table(tmp_path, n=120, seed=3):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    age = 40 + 8 * f1 + rng.normal(scale=1.0, size=n)
    cond = np.where(f1 + rng.normal(scale=0.4, size=n) > 0, "yes", "no")
    return _write(tmp_path / "train.csv",
                  ["text_id", "age", "cond", "f1", "f2"],
                  [[f"d{i}", f"{age[i]:.6f}", cond[i],
                    f"{f1[i]:.10f}", f"{f2[i]:.10f}"] for i in range(n)])


def _feature_tables(tmp_path, n=120, seed=9):
    """Two feature tables, as a run's extraction stage would leave them."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    a = _write(tmp_path / "features" / "one.csv", ["text_id", "f1"],
               [[f"d{i}", f"{f1[i]:.10f}"] for i in range(n)])
    b = _write(tmp_path / "features" / "two.csv", ["text_id", "f2"],
               [[f"d{i}", f"{f2[i]:.10f}"] for i in range(n)])
    return [a, b]


def _ridge_model(tmp_path, **kwargs):
    fit_ridge_csv(table_csv=_training_table(tmp_path), outcome_cols=["age"],
                  feature_sets={"lang": ["f1", "f2"]},
                  out_dir=tmp_path / "fit", verbose=False, **kwargs)
    return tmp_path / "fit" / "models" / "ridge__lang__age.json"


def _classifier_model(tmp_path):
    fit_classifier_csv(table_csv=_training_table(tmp_path),
                       outcome_cols=["cond"],
                       feature_sets={"lang": ["f1", "f2"]},
                       out_dir=tmp_path / "fitc", verbose=False)
    return tmp_path / "fitc" / "models" / "classifier__lang__cond.json"


def _mem_stub(tmp_path, k=2):
    """Enough of a MEM model to be *described*. The text branch is dispatched
    on the description, and refuses before any loading happens, so a stub is
    the right fixture for the dispatch itself."""
    path = tmp_path / "themes_model.json"
    path.write_text(json.dumps({
        "kind": "taters-mem-model", "format": 1,
        "model": {"themes": [f"Theme_{i + 1}" for i in range(k)]},
    }), encoding="utf-8")
    return path


# the tests below build their feature tables by hand, so those tables carry no
# settings record and the models fitted from them don't either. they pass
# `allow_unrecorded=True` on purpose: each one is about dispatch, output naming,
# controls or the join, and spelling the waiver out in the call is how you can
# tell we're stepping around the gate knowingly rather than by accident. the
# gate itself gets covered at the bottom of this file

def test_each_kind_of_model_is_scored_the_way_that_kind_needs(tmp_path):
    """The dispatch is the whole point of the one entry: a ridge and a
    classifier read feature columns, a topic model reads text, and which it
    is is a fact about the model rather than a question for the user. Before
    this there were three menu rows and the user had to know."""
    features = _feature_tables(tmp_path)

    scored = _read(score_with_model(
        model_json=_ridge_model(tmp_path), feature_csvs=features,
        out_csv=tmp_path / "r.csv", allow_unrecorded=True, verbose=False))
    assert len(scored) == 120
    assert all(r["pred_age"] for r in scored)

    scored = _read(score_with_model(
        model_json=_classifier_model(tmp_path), feature_csvs=features,
        out_csv=tmp_path / "c.csv", allow_unrecorded=True, verbose=False))
    assert all(r["pred_cond"] in ("no", "yes") for r in scored)
    assert all(0.0 <= float(r["prob_cond"]) <= 1.0 for r in scored)

    # a text model gets routed to the text branch. we can tell by which input
    # it asks for, since that's the only decision of the branch we can see
    with pytest.raises(ValueError) as e:
        score_with_model(model_json=_mem_stub(tmp_path),
                         feature_csvs=features, out_csv=tmp_path / "m.csv",
                         allow_unrecorded=True, verbose=False)
    assert "scores text" in str(e.value)
    assert "csv_path" in str(e.value)


def test_a_model_needing_features_this_run_lacks_names_them(tmp_path):
    """The failure that actually happens, and the fix is never in the model:
    it is a feature step that was not selected. So the message names the
    columns rather than saying the model could not be applied."""
    model = _ridge_model(tmp_path)
    only_one = [_feature_tables(tmp_path)[0]]
    with pytest.raises(ValueError) as e:
        score_with_model(model_json=model, feature_csvs=only_one,
                         out_csv=tmp_path / "x.csv", allow_unrecorded=True, verbose=False)
    assert "f2" in str(e.value)
    # no sidecar beside this fixture's table, so the model couldn't record
    # where its predictors came from, and the message says so rather than
    # naming a step it doesn't know
    assert "did not record which ones they were" in str(e.value)

    # and with no feature tables at all, it says how many columns it wanted
    # rather than reporting an empty join
    with pytest.raises(ValueError) as e:
        score_with_model(model_json=model, out_csv=tmp_path / "y.csv",
                         allow_unrecorded=True, verbose=False)
    assert "no feature tables" in str(e.value)
    assert "f1" in str(e.value) and "f2" in str(e.value)


def test_the_columns_are_named_the_way_the_user_named_the_model(tmp_path):
    """A ridge predicting age from a blog corpus arrives writing `pred_age`,
    which collides with the `age` column a new dataset already has -- and
    with the second such model scored in the same run. The name given at
    import has to reach the output."""
    model = _ridge_model(tmp_path)
    rename_model(model, name="age_blogs", outputs=["age_blogs"])
    out = score_with_model(model_json=model,
                           feature_csvs=_feature_tables(tmp_path),
                           out_csv=tmp_path / "r.csv", allow_unrecorded=True, verbose=False)
    assert "pred_age_blogs" in _read(out)[0]
    assert "pred_age" not in _read(out)[0]

    # the default output path is named after the model too, so two models
    # scored in one run can't overwrite each other
    import os
    here = os.getcwd()
    os.chdir(tmp_path)
    try:
        default = score_with_model(
            model_json=model, feature_csvs=_feature_tables(tmp_path),
            allow_unrecorded=True, verbose=False)
    finally:
        os.chdir(here)
    assert default.name == "age_blogs.csv"


def test_a_model_fitted_with_controls_scores_a_raw_spreadsheet(tmp_path):
    """The reason the control *recipe* is stored and not just the names. A
    model fitted with `gender` knows a predictor called `gender=male`, and
    the spreadsheet being scored has a column called `gender` holding words
    -- nothing has dummy-coded it, and the reference level that every
    coefficient is relative to lives only in the model."""
    rng = np.random.default_rng(17)
    n = 160
    f1 = rng.normal(size=n)
    gender = rng.choice(["female", "male"], n)
    age = 40 + 6 * f1 + 3 * (gender == "male") + rng.normal(size=n)
    train = _write(tmp_path / "t.csv",
                   ["text_id", "age", "gender", "f1"],
                   [[f"d{i}", f"{age[i]:.6f}", gender[i], f"{f1[i]:.10f}"]
                    for i in range(n)])
    fit_ridge_csv(table_csv=train, outcome_cols=["age"],
                  feature_sets={"lang": ["f1"]},
                  control_cols=["gender"], categorical_controls=["gender"],
                  out_dir=tmp_path / "fit", verbose=False)
    model = tmp_path / "fit" / "models" / "ridge__lang__age.json"
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert "gender=male" in doc["predictors"]
    assert doc["controls"] == [{"column": "gender", "level": "male",
                               "reference": "female"}]

    # a feature table carrying `f1`, with the raw `gender` column beside it
    fresh = _write(tmp_path / "features" / "f.csv",
                   ["text_id", "f1", "gender"],
                   [[f"d{i}", f"{f1[i]:.10f}", gender[i]]
                    for i in range(n)])
    scored = _read(score_with_model(model_json=model, feature_csvs=[fresh],
                                    out_csv=tmp_path / "s.csv",
                                    allow_unrecorded=True, verbose=False))
    assert all(r["pred_age"] for r in scored)


def test_the_joined_table_it_read_is_not_left_behind(tmp_path):
    """Scoring joins the run's feature tables to find the model's columns,
    and that intermediate is a copy of columns that already exist. Left in
    `features/`, every scored run would leave a folder nobody asked for --
    but it is kept on request, because "which rows survived the join" is the
    question a blank prediction raises."""
    model = _ridge_model(tmp_path)
    out = score_with_model(model_json=model,
                           feature_csvs=_feature_tables(tmp_path),
                           out_csv=tmp_path / "out" / "r.csv", allow_unrecorded=True, verbose=False)
    assert not (out.parent / "r_inputs").exists()

    kept = score_with_model(model_json=model,
                            feature_csvs=_feature_tables(tmp_path),
                            out_csv=tmp_path / "out2" / "r.csv",
                            keep_inputs=True, allow_unrecorded=True, verbose=False)
    work = kept.parent / "r_inputs"
    assert (work / "scoring_table.csv").is_file()
    assert (work / "assemble_manifest.json").is_file()


def test_a_row_missing_from_one_feature_table_is_accounted_for(tmp_path,
                                                               capsys):
    """The inner join is the same one the statistics use, so a text present
    in one table and not another drops out. Silently, that is a shorter
    results file than the user expected and no explanation; the row
    accounting is what answers it."""
    a, b = _feature_tables(tmp_path)
    short = _read(b)[:100]
    _write(b, ["text_id", "f2"], [[r["text_id"], r["f2"]] for r in short])
    out = score_with_model(model_json=_ridge_model(tmp_path),
                           feature_csvs=[a, b],
                           out_csv=tmp_path / "r.csv", allow_unrecorded=True, verbose=True)
    assert len(_read(out)) == 100
    said = capsys.readouterr().out
    assert "100" in said and "120" in said


def test_scores_that_already_exist_are_not_recomputed(tmp_path):
    """The overwrite contract every step in Taters keeps."""
    model = _ridge_model(tmp_path)
    features = _feature_tables(tmp_path)
    out = score_with_model(model_json=model, feature_csvs=features,
                           out_csv=tmp_path / "r.csv", allow_unrecorded=True, verbose=False)
    out.write_text("sentinel\n", encoding="utf-8-sig")
    again = score_with_model(model_json=model, feature_csvs=features,
                             out_csv=out, allow_unrecorded=True, verbose=False)
    assert again == out
    assert out.read_text(encoding="utf-8-sig") == "sentinel\n"
    score_with_model(model_json=model, feature_csvs=features, out_csv=out,
                     overwrite_existing=True, allow_unrecorded=True, verbose=False)
    assert out.read_text(encoding="utf-8-sig") != "sentinel\n"


def test_an_empty_folder_is_refused_and_a_full_one_scores_every_model(tmp_path):
    """The library hands steps a folder meaning "everything of this kind".
    For dictionaries that was always right; for models it used to be a
    refusal ("a step can only use one"), and now it means every model in
    it, each to its own file plus one merged table."""
    _ridge_model(tmp_path)
    models = tmp_path / "fit" / "models"
    second = json.loads((models / "ridge__lang__age.json").read_text(encoding="utf-8"))
    second["name"] = "second"
    (models / "second.json").write_text(json.dumps(second), encoding="utf-8")
    out = score_with_model(model_json=models, feature_csvs=_feature_tables(tmp_path),
                           out_csv=tmp_path / "scores.csv", allow_unrecorded=True,
                           verbose=False)
    rows = _read(out)
    assert list(rows[0]) == ["text_id", "ridge__lang__age__pred_age", "second__pred_age"]
    assert len(rows) == 120
    assert (tmp_path / "scores" / "ridge__lang__age.csv").is_file()
    assert (tmp_path / "scores" / "second.csv").is_file()
    empty = tmp_path / "empty_library"
    empty.mkdir()
    with pytest.raises(ValueError, match="no model file found"):
        score_with_model(model_json=empty,
                         feature_csvs=_feature_tables(tmp_path),
                         out_csv=tmp_path / "r2.csv", allow_unrecorded=True, verbose=False)


# ---------------------------------------------------------------------------
# several models at once
# ---------------------------------------------------------------------------

def test_one_model_writes_the_table_it_always_did(tmp_path):
    """No prefix, no subfolder, no merged accounting: a single model's
    output is byte for byte what it was before several were allowed."""
    model = _ridge_model(tmp_path)
    out = score_with_model(model_json=[model], feature_csvs=_feature_tables(tmp_path),
                           out_csv=tmp_path / "one.csv", allow_unrecorded=True,
                           verbose=False)
    assert list(_read(out)[0]) == ["text_id", "pred_age"]
    assert not (tmp_path / "one").exists()


def test_several_models_merge_with_their_names_on_every_column(tmp_path):
    """A ridge and a classifier, both fitted on the same columns: the merged
    table is an outer join keyed on text_id, with each model's columns
    prefixed by its name so two models predicting one outcome never
    collide; each model also keeps its own plain-named file."""
    ridge = _ridge_model(tmp_path)
    clf = _classifier_model(tmp_path)
    rename_model(ridge, name="age_blogs")
    features = _feature_tables(tmp_path)
    out = score_with_model(model_json=[ridge, clf], feature_csvs=features,
                           out_csv=tmp_path / "features" / "model_scores.csv",
                           allow_unrecorded=True, verbose=False)
    merged = _read(out)
    assert list(merged[0]) == ["text_id", "age_blogs__pred_age", "classifier__lang__cond__pred_cond",
                               "classifier__lang__cond__prob_cond", "classifier__lang__cond__p_cond_no",
                               "classifier__lang__cond__p_cond_yes"]
    own_ridge = _read(tmp_path / "features" / "model_scores" / "age_blogs.csv")
    own_clf = _read(tmp_path / "features" / "model_scores" / "classifier__lang__cond.csv")
    assert list(own_ridge[0]) == ["text_id", "pred_age"]
    for m, r, c in zip(merged, own_ridge, own_clf):
        assert m["text_id"] == r["text_id"] == c["text_id"]
        assert m["age_blogs__pred_age"] == r["pred_age"]
        assert m["classifier__lang__cond__pred_cond"] == c["pred_cond"]
    assert {m["classifier__lang__cond__pred_cond"] for m in merged} <= {"yes", "no"}, \
        "labels survive: the merge never goes through the numeric assembler"


def test_two_models_that_read_the_same_are_refused_before_any_scoring(tmp_path):
    ridge = _ridge_model(tmp_path)
    twin = tmp_path / "fit" / "models" / "twin.json"
    doc = json.loads(ridge.read_text(encoding="utf-8"))
    doc["name"] = "ridge__lang__age"    # exactly the other model's slug
    twin.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="both called 'ridge__lang__age'"):
        score_with_model(model_json=[ridge, twin], feature_csvs=_feature_tables(tmp_path),
                         out_csv=tmp_path / "m.csv", allow_unrecorded=True, verbose=False)
    assert not (tmp_path / "m").exists() and not (tmp_path / "m.csv").exists()


def test_a_row_lost_by_one_model_is_accounted_for_under_its_name(tmp_path):
    ridge = _ridge_model(tmp_path)
    clf = _classifier_model(tmp_path)
    features = _feature_tables(tmp_path)
    rows = _read(features[0])
    _write(features[0], list(rows[0]), [[r["text_id"], r["f1"]] for r in rows[:-3]])
    score_with_model(model_json=[ridge, clf], feature_csvs=features,
                     out_csv=tmp_path / "s.csv", allow_unrecorded=True, verbose=False)
    account = _read(tmp_path / "s_unscored.csv")
    assert list(account[0]) == ["model", "reason", "detail", "rows"]
    assert {a["model"] for a in account} == {"ridge__lang__age", "classifier__lang__cond"}
    assert all(a["rows"] == "3" for a in account if a["reason"].startswith("missing"))
    # both models read f1, so both lost the same three rows to the join
    assert len(_read(tmp_path / "s.csv")) == 117


def test_the_merge_is_an_outer_join_in_the_first_models_order(tmp_path):
    """Pinned on hand-written per-model files: a text_id only the second
    model saw is kept (blank for the first), the first model's order wins,
    and every non-key column carries its model's name."""
    from taters.helpers.model_spec import describe
    from taters.score_model import _merge_scores

    a = _write(tmp_path / "a.csv", ["text_id", "pred_age"], [["d2", "1"], ["d1", "2"]])
    b = _write(tmp_path / "b.csv", ["text_id", "pred_cond", "prob_cond"],
               [["d1", "yes", "0.9"], ["d9", "no", "0.6"]])
    infos = [describe(_ridge_model(tmp_path)), describe(_classifier_model(tmp_path))]
    _merge_scores(infos, [a, b], tmp_path / "m.csv", key_cols=["text_id"], encoding="utf-8-sig")
    rows = _read(tmp_path / "m.csv")
    assert list(rows[0]) == ["text_id", "ridge__lang__age__pred_age", "classifier__lang__cond__pred_cond",
                             "classifier__lang__cond__prob_cond"]
    assert [r["text_id"] for r in rows] == ["d2", "d1", "d9"]
    assert rows[0]["classifier__lang__cond__pred_cond"] == "" and rows[2]["ridge__lang__age__pred_age"] == ""
    assert rows[1] == {"text_id": "d1", "ridge__lang__age__pred_age": "2",
                       "classifier__lang__cond__pred_cond": "yes", "classifier__lang__cond__prob_cond": "0.9"}


def test_adding_a_model_rescored_only_that_one_and_remerges(tmp_path):
    ridge = _ridge_model(tmp_path)
    clf = _classifier_model(tmp_path)
    features = _feature_tables(tmp_path)
    out = tmp_path / "s.csv"
    score_with_model(model_json=[ridge], feature_csvs=features, out_csv=out,
                     allow_unrecorded=True, verbose=False)
    first = out.read_bytes()
    score_with_model(model_json=[ridge, clf], feature_csvs=features, out_csv=out,
                     allow_unrecorded=True, verbose=False)
    assert out.read_bytes() != first, "two models now: the merged table was rebuilt"
    import os
    import time

    # push the merged file's clock back a day, so a rewrite would show up on
    # any filesystem, however coarse its timestamps are
    yesterday = time.time() - 86400
    for path in (out, tmp_path / "s" / "ridge__lang__age.csv", tmp_path / "s" / "classifier__lang__cond.csv"):
        os.utime(path, (yesterday, yesterday))
    for path in (ridge, clf, *features):
        os.utime(path, (yesterday - 10, yesterday - 10))
    score_with_model(model_json=[ridge, clf], feature_csvs=features, out_csv=out,
                     allow_unrecorded=True, verbose=False)
    assert abs((tmp_path / "s" / "ridge__lang__age.csv").stat().st_mtime - yesterday) < 5
    assert abs(out.stat().st_mtime - yesterday) < 5, "nothing newer: nothing redone"


def test_every_models_gate_problem_is_named_at_once(tmp_path):
    """Two models that predate recording, both refused, both named in one
    message -- not the first one only, with the second waiting to fail
    after a re-run."""
    ridge = _ridge_model(tmp_path)
    clf = _classifier_model(tmp_path)
    with pytest.raises(ValueError) as e:
        score_with_model(model_json=[ridge, clf], feature_csvs=_feature_tables(tmp_path),
                         out_csv=tmp_path / "s.csv", verbose=False)
    text = str(e.value)
    assert text.startswith("2 of the 2 models cannot be scored")
    assert "ridge__lang__age [ridge]" in text and "classifier__lang__cond [classifier]" in text


def test_each_model_reads_its_own_private_table_and_never_anothers(tmp_path):
    """Two models fitted on parts of speech at different roundings; the run
    measured a private `pos` for each under model_inputs/<slug>/. Handed
    the union, the gate used to see one `pos` (whichever came last) and
    refuse the other model as a known mismatch."""
    from taters.helpers.model_spec import MODEL_WORK_DIR, describe, rename_model
    from taters.score_model import _tables_for

    six = _measured(tmp_path, "pos.csv", rounding=6)
    a, _doc = _fit_on(tmp_path, six, "fit_a")
    rename_model(a, name="six")
    (tmp_path / "two").mkdir()
    two = _measured(tmp_path, "two/pos.csv", rounding=2)     # same stem, other settings
    b, _doc = _fit_on(tmp_path, two, "fit_b")
    rename_model(b, name="two")
    # the run's tables, laid out the way the composer lays them out
    import shutil

    priv_a = tmp_path / "features" / MODEL_WORK_DIR / "six" / "abcd1234" / "pos.csv"
    priv_b = tmp_path / "features" / MODEL_WORK_DIR / "two" / "ef567890" / "pos.csv"
    for src, dst in ((six, priv_a), (two, priv_b)):
        dst.parent.mkdir(parents=True)
        shutil.copy(src, dst)
        shutil.copy(src.with_name(src.stem + "_settings.json"),
                    dst.with_name("pos_settings.json"))
    shared = tmp_path / "features" / "other.csv"
    shutil.copy(six, shared)
    # and a private table of model two with a stem model six never recorded:
    # it shouldn't ride along as an "extra" for six
    stray = tmp_path / "features" / MODEL_WORK_DIR / "two" / "ef567890" / "extra.csv"
    shutil.copy(six, stray)
    union = [priv_a, priv_b, shared, stray]
    info_a, info_b = describe(a), describe(b)
    assert _tables_for(info_a, union) == [priv_a, shared]
    assert _tables_for(info_b, union) == [priv_b, shared]
    out = score_with_model(model_json=[a, b], feature_csvs=union,
                           out_csv=tmp_path / "s.csv", verbose=False)
    assert {"six__pred_outcome", "two__pred_outcome"} <= set(_read(out)[0])


def test_two_tables_with_one_stem_that_cannot_be_told_apart_are_refused(tmp_path):
    from taters.helpers.model_spec import describe
    from taters.score_model import _tables_for

    table = _measured(tmp_path, "pos.csv")
    model, _doc = _fit_on(tmp_path, table, "fit")
    other = tmp_path / "elsewhere" / "pos.csv"
    other.parent.mkdir()
    other.write_text("text_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has 2 of them"):
        _tables_for(describe(model), [table, other])


# ---------------------------------------------------------------------------
# the gate: features measured with other settings
# ---------------------------------------------------------------------------

def _measured(tmp_path, name, *, rounding=6, **kwargs):
    """
    A real feature table with a real settings record beside it.

    Parts of speech with `rounding`, deliberately. The planted difference has
    to be one that changes the *values* and not the column names, because a
    name change already fails safely as a missing predictor and would prove
    nothing about the gate. `rounding` is exactly that: 6 against 2 moves
    eight columns of numbers under identical headings.

    The obvious first choice, `analyze_lexical_richness`'s `mattr_window`,
    turns out to be the wrong kind -- that analyzer puts its settings in its
    column names (`mattr_100`, `mtld_0_72`), which is good design and makes
    its mismatches loud already. The canary test below is what caught that.
    """
    from taters.text.analyze_parts_of_speech import analyze_parts_of_speech

    ready = tmp_path / "ready.csv"
    if not ready.exists():
        with ready.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["text_id", "text"])
            for i in range(30):
                w.writerow([f"d{i}", f"The quick brown fox number {i} jumps "
                                     f"over the lazy dogs and runs fast."])
    return analyze_parts_of_speech(
        analysis_csv=ready, out_features_csv=tmp_path / name,
        engine="nltk", rounding=rounding, overwrite_existing=True, **kwargs)


def _fit_on(tmp_path, table, out_dir):
    """Assemble a one-table analysis table, then fit a ridge on it."""
    import json as _json

    from taters.stats.assemble import assemble_analysis_table
    from taters.stats.ridge import fit_ridge_csv

    rows = _read(table)
    meta = tmp_path / f"{out_dir}_meta.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "outcome"])
        for i, r in enumerate(rows):
            w.writerow([r["text_id"], i % 11])
    assembled = assemble_analysis_table(
        feature_csvs=[str(table)], metadata_csv=str(meta),
        metadata_cols=["outcome"], out_dir=tmp_path / out_dir,
        overwrite_existing=True, verbose=False)
    fit_ridge_csv(table_csv=assembled, outcome_cols=["outcome"],
                  out_dir=tmp_path / out_dir, overwrite_existing=True,
                  verbose=False)
    model = tmp_path / out_dir / "models" / "ridge__all__outcome.json"
    assert model.is_file()
    doc = _json.loads(model.read_text(encoding="utf-8"))
    return model, doc


def test_a_model_records_how_its_features_were_measured(tmp_path):
    """Without this the gate has nothing to compare against, and a predictor
    name is all a model knows -- which is no defense at all against a column
    of the right name holding numbers from different settings."""
    table = _measured(tmp_path, "pos.csv", rounding=6)
    _model, doc = _fit_on(tmp_path, table, "fit")
    recorded = doc["needs"]["feature_provenance"]
    assert "pos" in recorded
    assert recorded["pos"]["state"] == "recorded"
    assert recorded["pos"]["instrument"]["rounding"] == 6


def test_refuses_features_measured_with_other_settings(tmp_path):
    """
    **The test whose failure mode is "the model was silently scored on the
    wrong numbers."**

    This is the bug, reproduced: fit on a table measured one way, rebuild the
    same table the other way, and score. The column names are identical, so
    nothing else in the system can tell. In a real run this moved a mean
    predicted age from 36.5 to 44.0 years and reported 904 predictions
    without a word.
    """
    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, _doc = _fit_on(tmp_path, table, "fit")

    # same stem, same path, same column names, but a different measuring setting
    _measured(tmp_path, "pos.csv", rounding=2)

    with pytest.raises(ValueError) as e:
        score_with_model(model_json=model, feature_csvs=[table],
                         out_csv=tmp_path / "scores.csv", verbose=False)
    message = str(e.value)
    assert "rounding" in message
    assert "6" in message and "2" in message
    assert "change the numbers, not the column names" in message
    assert not (tmp_path / "scores.csv").exists(), \
        "a refusal must not leave scores behind"


def test_the_two_settings_really_do_produce_different_numbers(tmp_path):
    """
    The canary that stops the test above from being tautological.

    If `mattr_window` ever stopped affecting the output, the mismatch test
    would still pass -- it would just be asserting that a check fires, not
    that the check is protecting anything. So: assert the two tables actually
    differ. If this goes green-by-equality, the mismatch above is proving
    nothing and needs a different planted difference.
    """
    a = _read(_measured(tmp_path, "a.csv", rounding=6))
    b = _read(_measured(tmp_path, "b.csv", rounding=2))
    # the names have to match, otherwise the mismatch above would get caught by
    # the ordinary missing-predictor check and we'd have proven nothing new
    assert list(a[0]) == list(b[0]), \
        "the planted difference changes column names, so it fails safely " \
        "already and is the wrong kind of difference to plant"
    moved = [c for c in a[0] if a[0][c] != b[0][c]]
    assert moved, \
        "rounding no longer changes the numbers; re-plant the mismatch"


def test_matching_settings_score_without_complaint(tmp_path):
    """The other half: the gate must not cry wolf. Same settings, same
    numbers, no refusal."""
    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, _doc = _fit_on(tmp_path, table, "fit")
    out = score_with_model(model_json=model, feature_csvs=[table],
                           out_csv=tmp_path / "scores.csv", verbose=False)
    assert out.is_file()
    assert _read(out), "no predictions were written"


def test_a_stale_scores_file_does_not_bypass_the_gate(tmp_path):
    """
    The likeliest way to meet this bug in practice: a results folder re-run
    after somebody changed a setting. `score_with_model` returns an existing
    scores file before doing anything else when `overwrite_existing` is
    false, so a check placed after that short-circuit would never see the
    mismatch at all.
    """
    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, _doc = _fit_on(tmp_path, table, "fit")
    scores = tmp_path / "scores.csv"
    scores.write_text("text_id,pred_outcome\nd0,1.0\n", encoding="utf-8-sig")

    _measured(tmp_path, "pos.csv", rounding=2)
    with pytest.raises(ValueError, match="rounding"):
        score_with_model(model_json=model, feature_csvs=[table],
                         out_csv=scores, overwrite_existing=False,
                         verbose=False)


def test_an_unverifiable_table_is_refused_until_named(tmp_path):
    """"I cannot check this" is a different claim from "these disagree", and
    only the first is a researcher's to waive -- per table, never with a
    blanket flag that would also waive the table sitting next to it."""
    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, _doc = _fit_on(tmp_path, table, "fit")
    # now we remove the record but leave the numbers untouched
    from taters.helpers import provenance as prov
    prov.sidecar_path(table).unlink()

    with pytest.raises(ValueError) as e:
        score_with_model(model_json=model, feature_csvs=[table],
                         out_csv=tmp_path / "s.csv", verbose=False)
    assert "no record of how" in str(e.value)
    assert "unverified_ok" in str(e.value)

    out = score_with_model(model_json=model, feature_csvs=[table],
                           out_csv=tmp_path / "s.csv",
                           unverified_ok=("pos",), verbose=False)
    assert out.is_file()


def test_a_waiver_cannot_wave_through_a_known_mismatch(tmp_path):
    """`unverified_ok` waives *not knowing*. When both sides state their
    settings and the settings differ, there is nothing to interpret and no
    override -- the model was fitted on other numbers."""
    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, _doc = _fit_on(tmp_path, table, "fit")
    _measured(tmp_path, "pos.csv", rounding=2)

    with pytest.raises(ValueError, match="rounding"):
        score_with_model(model_json=model, feature_csvs=[table],
                         out_csv=tmp_path / "s.csv",
                         unverified_ok=("pos",), allow_unrecorded=True,
                         verbose=False)


def test_a_model_from_before_recording_is_refused_by_default(tmp_path):
    """No ridge or classifier model has ever shipped, so refusing costs
    nobody anything -- and it closes exactly the case that reported a
    prediction seven years wrong without complaint."""
    import json as _json

    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, doc = _fit_on(tmp_path, table, "fit")
    doc["needs"].pop("feature_provenance")
    model.write_text(_json.dumps(doc), encoding="utf-8")

    with pytest.raises(ValueError) as e:
        score_with_model(model_json=model, feature_csvs=[table],
                         out_csv=tmp_path / "s.csv", verbose=False)
    assert "does not record how its features were measured" in str(e.value)
    assert "allow_unrecorded" in str(e.value)

    out = score_with_model(model_json=model, feature_csvs=[table],
                           out_csv=tmp_path / "s.csv",
                           allow_unrecorded=True, verbose=False)
    assert out.is_file()


def test_a_text_model_has_nothing_to_check(tmp_path):
    """A topic model re-derives its own instrument from its own file, so the
    gate has no business asking it about feature settings."""
    stub = _mem_stub(tmp_path)
    with pytest.raises(ValueError, match="scores text"):
        score_with_model(model_json=stub, feature_csvs=[],
                         out_csv=tmp_path / "s.csv", verbose=False)


def test_text_prepared_differently_upstream_is_refused_at_the_gate(tmp_path):
    """
    Identical instrument, different chain: the case the chain digest exists
    for, and the one the gate could not see while it assembled the run-time
    record without `digests`.
    """
    import json as _json

    table = _measured(tmp_path, "pos.csv", rounding=6)
    model, doc = _fit_on(tmp_path, table, "fit")
    doc["needs"]["feature_provenance"]["pos"]["digests"]["chain"] = "0" * 16
    model.write_text(_json.dumps(doc), encoding="utf-8")

    with pytest.raises(ValueError) as e:
        score_with_model(model_json=model, feature_csvs=[table],
                         out_csv=tmp_path / "scores.csv", verbose=False)
    assert "prepared differently" in str(e.value)


def test_rows_lost_in_the_join_are_in_the_unscored_accounting_too(tmp_path):
    """A row missing from one feature table never reaches the scorer, so
    the scorer's own accounting cannot name it. The scoring step adds it
    under its own reason, and one file answers "why is this text not
    scored" whatever the cause."""
    from taters.stats._fit_common import unscored_path

    tables = _feature_tables(tmp_path)
    rows = _read(tables[0])
    short = _write(tmp_path / "features" / "short.csv", list(rows[0]),
                   [[r[c] for c in rows[0]] for r in rows[:-3]])
    model = _ridge_model(tmp_path)
    out = score_with_model(model_json=model, feature_csvs=[short, *tables[1:]],
                           out_csv=tmp_path / "s.csv", allow_unrecorded=True,
                           verbose=False)
    account = {(r["reason"], r["detail"]): int(r["rows"])
               for r in _read(unscored_path(out))}
    # `short` was the join's base, so the three rows the other table has and it
    # doesn't show up as unmatched rows of that other table
    assert account[("not in every earlier feature table", "two")] == 3
