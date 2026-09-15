"""
A saved model carries everything needed to measure its features again.

Scoring a second study with a ridge fitted on the first failed in a real
end-to-end run over four missing columns: a term the new texts never used, a
part-of-speech tag none of them contained, and two topic-model themes --
because the document-term matrix was re-derived from the new corpus, the
topic model was refitted to it, and the tag simply did not occur. These tests
pin the three answers: the matrix is scanned against the vocabulary the
model carries, the themes are *applied* from the fitted theme model the
model carries, and an absent tag column counts as zero.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from taters.helpers import provenance as pv
from taters.helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings

from csvhelpers import _read


# ---------------------------------------------------------------------------
# a toy analyzer family: a fit step whose table can only be replayed by
# applying what it fitted, plus the apply step that does exactly that
# ---------------------------------------------------------------------------

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv", "out_model_json"),
                  replay=(f"{__name__}:apply_toy",
                          {"model_json": "out_model_json"}))
def fit_toy(*, analysis_csv, out_features_csv, out_model_json=None,
            scale=1.0, overwrite_existing=False):
    rows = _read(analysis_csv)
    out = Path(out_features_csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "toy"])
        for r in rows:
            w.writerow([r["text_id"], round(len(r["text"]) * scale, 3)])
    if out_model_json:
        Path(out_model_json).write_text(json.dumps({"scale": scale}),
                                        encoding="utf-8")
    return out


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  assets={"model_json": None}, outputs=("out_features_csv",))
def apply_toy(*, model_json, analysis_csv, out_features_csv,
              overwrite_existing=False):
    scale = json.loads(Path(model_json).read_text(encoding="utf-8"))["scale"]
    return fit_toy.__wrapped__(analysis_csv=analysis_csv,
                               out_features_csv=out_features_csv, scale=scale)


def _texts(tmp_path, name="ready.csv", n=6):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            w.writerow([f"d{i}", "potato " * (i + 1)])
    return path


def _fit(tmp_path, stem="toy", n=6, **kw):
    return fit_toy(analysis_csv=_texts(tmp_path, n=n),
                   out_features_csv=tmp_path / f"{stem}.csv",
                   out_model_json=tmp_path / f"{stem}_model.json", **kw)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_a_fitted_table_records_how_to_apply_what_it_fitted(tmp_path):
    """The replay names the applying function and carries the fitted file
    by content, so a model fitted on this table can take it along."""
    table = _fit(tmp_path)
    rec = pv.read(table)
    assert rec["replay"]["call"] == f"{__name__}:apply_toy"
    [entry] = rec["replay"]["assets"]["model_json"]
    assert entry["name"] == "toy_model.json"
    assert json.loads(entry["text"]) == {"scale": 1.0}
    assert entry["sha256"] == pv.text_digest(entry["text"])


def test_no_replay_is_recorded_when_the_fitted_file_was_not_named(tmp_path):
    """A model path the analyzer chose for itself is not knowable to the
    record, and a replay naming the wrong file is worse than none."""
    table = fit_toy(analysis_csv=_texts(tmp_path),
                    out_features_csv=tmp_path / "toy.csv")
    assert "replay" not in pv.read(table)


def test_an_applied_table_passes_for_the_fitted_one_it_replays(tmp_path):
    """The gate's question is "were these numbers produced by the instrument
    the model was fitted beside?" -- and for a fitted instrument the answer
    is yes exactly when the applied file is byte-for-byte the fitted one."""
    fitted = pv.read(_fit(tmp_path))
    applied = pv.read(apply_toy(
        model_json=tmp_path / "toy_model.json",
        analysis_csv=_texts(tmp_path, "other.csv", n=3),
        out_features_csv=tmp_path / "applied.csv"))
    assert pv.replays(pv.comparable(fitted), pv.comparable(applied))

    mismatched, unverifiable = pv.compare_tables(
        {"toy": pv.comparable(fitted)}, {"toy": pv.comparable(applied)})
    assert not mismatched and not unverifiable


def test_applying_some_other_fitted_file_does_not_pass(tmp_path):
    """Same applying function, a theme model fitted elsewhere: refused, with
    the difference named, exactly as a wrong word list would be."""
    fitted = pv.read(_fit(tmp_path))
    other = tmp_path / "toy_model.json"
    other.write_text(json.dumps({"scale": 2.0}), encoding="utf-8")
    applied = pv.read(apply_toy(
        model_json=other, analysis_csv=_texts(tmp_path, "other.csv", n=3),
        out_features_csv=tmp_path / "applied.csv"))
    assert not pv.replays(pv.comparable(fitted), pv.comparable(applied))
    mismatched, _ = pv.compare_tables(
        {"toy": pv.comparable(fitted)}, {"toy": pv.comparable(applied)})
    assert mismatched


def test_an_asset_digests_by_its_text_so_a_copy_from_the_model_matches(
        tmp_path):
    """The model carries the *text*; a replay writes that text back out. A
    byte digest made the copy differ from the original over a byte-order
    mark or Windows line endings, and the gate refused a model its own
    vocabulary."""
    original = tmp_path / "list.txt"
    original.write_bytes("﻿potato\r\ngravy\r\n".encode("utf-8"))
    [entry] = pv.asset_manifest(None, original)
    copy = tmp_path / "copy.txt"
    copy.write_text(entry["text"], encoding="utf-8")
    [again] = pv.asset_manifest(None, copy)
    assert again["sha256"] == entry["sha256"]


def test_the_analyzers_declare_their_side_of_the_bargain():
    """The matrix's vocabulary is an asset, the topic model says how it is
    applied, the applier records the model it applied, and a tag table says
    its absent columns are zeros. Pinned so a refactor cannot quietly undo
    the end-to-end fix."""
    from taters.text.analyze_parts_of_speech import analyze_parts_of_speech
    from taters.text.build_doc_term_matrix import build_doc_term_matrix
    from taters.text.topic_model_mem import apply_mem_model, topic_model_mem

    assert "freq_list_csv" in build_doc_term_matrix.__provenance__["assets"]
    assert "freq_list_csv" not in build_doc_term_matrix.__provenance__["binding"]
    call, takes = topic_model_mem.__provenance__["replay"]
    assert call.endswith(":apply_mem_model")
    assert takes == {"model_json": "out_model_json"}
    assert "model_json" in apply_mem_model.__provenance__["assets"]
    assert analyze_parts_of_speech.__provenance__["absent_means_zero"]


# ---------------------------------------------------------------------------
# zero for a count that never occurred
# ---------------------------------------------------------------------------

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",), absent_means_zero=True)
def count_toy(*, analysis_csv, out_features_csv, columns=("a", "b", "c"),
              overwrite_existing=False):
    rows = _read(analysis_csv)
    out = Path(out_features_csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", *columns])
        for i, r in enumerate(rows):
            w.writerow([r["text_id"], *[(i * 7 + k) % 5 for k in
                                         range(len(columns))]])
    return out


def _fit_ridge_on(tmp_path, table, out_dir):
    from taters.stats.assemble import assemble_analysis_table
    from taters.stats.ridge import fit_ridge_csv

    tables = list(table) if isinstance(table, list) else [table]
    rows = _read(tables[0])
    meta = tmp_path / f"{out_dir}_meta.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "outcome"])
        for i, r in enumerate(rows):
            w.writerow([r["text_id"], (i * 3) % 11])
    assembled = assemble_analysis_table(
        feature_csvs=[str(t) for t in tables], metadata_csv=str(meta),
        metadata_cols=["outcome"], out_dir=tmp_path / out_dir,
        overwrite_existing=True, verbose=False)
    fit_ridge_csv(table_csv=assembled, outcome_cols=["outcome"],
                  out_dir=tmp_path / out_dir, overwrite_existing=True,
                  verbose=False)
    return tmp_path / out_dir / "models" / "ridge__all__outcome.json"


def test_a_model_fitted_on_counts_names_the_columns_that_may_read_zero(
        tmp_path):
    """Per predictor and inside the model file, so the scorer never has to
    guess from a column's name which absences are measurements."""
    from taters.stats._common import zero_when_absent

    counts = count_toy(analysis_csv=_texts(tmp_path, n=30),
                       out_features_csv=tmp_path / "counts.csv")
    # a second table that makes no such claim, so its column can still be refused
    plain = _fit(tmp_path, n=30)
    model = _fit_ridge_on(tmp_path, [counts, plain], "fit")
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert "toy" in doc["predictors"]
    assert doc["zero_when_absent"] == ["a", "b", "c"]
    assert zero_when_absent(tmp_path / "fit" / "analysis_table.csv",
                            ["a", "b", "toy", "outcome"]) == ["a", "b"]


def test_a_count_column_the_new_corpus_never_has_is_scored_as_zero(tmp_path):
    """The prediction equals the one from a table holding explicit zeros,
    which is what the count was. Any other absent column is still refused."""
    from taters.stats.ridge import apply_ridge_csv

    table = count_toy(analysis_csv=_texts(tmp_path, n=30),
                      out_features_csv=tmp_path / "counts.csv")
    model = _fit_ridge_on(tmp_path, table, "fit")

    rows = _read(table)[:5]
    with_zeros = tmp_path / "zeros.csv"
    without = tmp_path / "without.csv"
    with with_zeros.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "a", "b", "c"])
        for r in rows:
            w.writerow([r["text_id"], r["a"], r["b"], 0])
    with without.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "a", "b"])
        for r in rows:
            w.writerow([r["text_id"], r["a"], r["b"]])

    a = _read(apply_ridge_csv(model_json=model, input_csv=with_zeros,
                              out_csv=tmp_path / "a.csv", verbose=False))
    b = _read(apply_ridge_csv(model_json=model, input_csv=without,
                              out_csv=tmp_path / "b.csv", verbose=False))
    assert [r["pred_outcome"] for r in a] == [r["pred_outcome"] for r in b]

    doc = json.loads(model.read_text(encoding="utf-8"))
    doc["zero_when_absent"] = []
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 1 predictor"):
        apply_ridge_csv(model_json=strict, input_csv=without,
                        out_csv=tmp_path / "c.csv", verbose=False)


def test_the_scoring_step_agrees_with_the_scorer_about_zeros(tmp_path):
    """`score_with_model` names missing columns before scoring, in its own
    words; it must not refuse a column the scorer would have filled."""
    from taters.helpers.model_spec import describe
    from taters.score_model import _check_inputs

    table = count_toy(analysis_csv=_texts(tmp_path, n=30),
                      out_features_csv=tmp_path / "counts.csv")
    model = _fit_ridge_on(tmp_path, table, "fit")
    info = describe(model)
    assert info.zero_when_absent == ("a", "b", "c")

    partial = tmp_path / "partial.csv"
    partial.write_text("text_id,a,b\nd0,1,2\n", encoding="utf-8")
    _check_inputs(info, partial, encoding="utf-8")       # no refusal

    doc = json.loads(model.read_text(encoding="utf-8"))
    doc["zero_when_absent"] = []
    strict = tmp_path / "strict.json"
    strict.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="not in this run: c"):
        _check_inputs(describe(strict), partial, encoding="utf-8")


# ---------------------------------------------------------------------------
# the plan a model yields, and the steps the composer builds from it
# ---------------------------------------------------------------------------

def _model_with(tmp_path, *, replay=True):
    """A ridge fitted on the toy fitted table, so its plan carries the
    replay and the fitted file."""
    table = _fit(tmp_path, n=30)
    return _fit_ridge_on(tmp_path, table, "fit")


def test_the_plan_carries_the_fitted_file_and_where_it_came_from(tmp_path):
    from taters.helpers.model_spec import describe, feature_plan

    model = _model_with(tmp_path)
    plan = feature_plan(describe(model))
    assert plan.replayable, plan.problems
    assert plan.path == str(model)
    [table] = plan.tables
    assert table.replay["target"] == f"{__name__}:apply_toy"
    [entry] = table.replay["assets"]["model_json"]
    assert json.loads(entry["text"]) == {"scale": 1.0}


def test_a_model_that_lost_its_fitted_file_cannot_be_replayed(tmp_path):
    """Honest half: without the fitted file's text the replay is not
    constructible, and the plan says so instead of writing an empty one."""
    from taters.helpers.model_spec import describe, feature_plan

    model = _model_with(tmp_path)
    doc = json.loads(model.read_text(encoding="utf-8"))
    for entry in doc["assets"]["model_json"]:
        entry.pop("text")
    model.write_text(json.dumps(doc), encoding="utf-8")
    plan = feature_plan(describe(model))
    assert not plan.replayable
    assert any("toy_model.json" in p for p in plan.problems)


def _csv_kwargs(**extra):
    kwargs = dict(source="csv", input_path="study.csv", text_cols=["text"],
                  id_cols=["pid"], name="t")
    kwargs.update(extra)
    return kwargs


def test_a_topic_model_table_is_replayed_by_applying_the_saved_themes():
    """The user's own topic-model step, whatever its settings, fits new
    themes to the new corpus and measures something else. The model's
    table is produced by a private step applying the fitted theme model,
    and the scoring step reads that and nothing else."""
    from taters.helpers.model_spec import FeaturePlan, TablePlan
    from taters.ui.compose import compose
    from taters.ui.recipes import by_id

    fitted = [{"name": "topic_model_mem_model.json", "sha256": "ab" * 32,
               "text": "{}"}]
    plan = FeaturePlan(
        model="themes [ridge]", slug="themes", path="/lib/models/themes.json",
        tables=(TablePlan(
            stem="topic_model_mem", target=by_id("topic_model_mem").target,
            instrument={}, assets={}, digest="feedface", grain={},
            replay={"target": by_id("topic_model_mem_apply").target,
                    "assets": {"model_json": fitted}}),))
    preset = compose(["topic_model_mem", "score_with_model"],
                     model_plans=[plan], **_csv_kwargs())
    steps = preset["steps"]
    private = [s for s in steps if s.get("for_model")]
    assert len(private) == 1
    [applied] = private
    assert applied["call"] == "potato.text.apply_mem_model"
    assert applied["assets"] == {"from": "/lib/models/themes.json",
                                 "take": {"model_json":
                                          ["topic_model_mem_model.json"]}}
    assert "model_json" not in applied["with"], \
        "the parameter is bound at run time from the model's copy"
    assert applied["with"]["out_features_csv"].endswith(
        "/model_inputs/themes/feedface/topic_model_mem.csv")
    scoring = next(s for s in steps if s["call"].endswith("score_with_model"))
    assert scoring["with"]["feature_csvs"] == ["{{" + applied["save_as"] + "}}"]
    # the user's own fit is still there, untouched, and it's not what gets scored
    assert any(s["call"] == "potato.text.topic_model_mem"
               and not s.get("for_model") for s in steps)


def test_a_matrix_table_is_scanned_against_the_vocabulary_the_model_carries():
    """A document-term matrix reads a frequency list -- another step's
    output -- which used to make it impossible to measure twice. The model
    carries the list it was fitted against, so the private matrix scans the
    new text against that and needs no second frequency step."""
    from taters.helpers.model_spec import FeaturePlan, TablePlan
    from taters.ui.compose import _instrument_defaults, compose
    from taters.ui.recipes import by_id

    dtm = by_id("doc_term_matrix")
    vocab = [{"name": "ngram_frequencies.csv", "sha256": "cd" * 32,
              "text": "ngram,frequency\npotato,9\n"}]
    plan = FeaturePlan(
        model="vocab [ridge]", slug="vocab", path="/lib/models/vocab.json",
        tables=(TablePlan(
            stem="doc_term_matrix_count", target=dtm.target,
            instrument={**_instrument_defaults(dtm), "weighting": "tfidf"},
            assets={"freq_list_csv": vocab}, digest="0badf00d", grain={}),))
    preset = compose(["doc_term_matrix", "score_with_model"],
                     model_plans=[plan], **_csv_kwargs())
    private = [s for s in preset["steps"] if s.get("for_model")]
    assert len(private) == 1, "the matrix was not measured for the model"
    [clone] = private
    assert clone["with"]["weighting"] == "tfidf"
    assert "freq_list_csv" not in clone["with"]
    assert clone["assets"]["take"] == {"freq_list_csv":
                                       ["ngram_frequencies.csv"]}


def test_reuse_compares_the_word_lists_by_content_not_only_the_settings(
        tmp_path):
    """Identical settings and different dictionaries produce different
    columns. Reusing the user's table for a model fitted on other lists
    was caught at run time -- as a dead end, since the user's only way past
    was to change their own dictionaries."""
    from taters.helpers.model_spec import FeaturePlan, TablePlan
    from taters.ui.compose import _instrument_defaults, compose
    from taters.ui.recipes import by_id

    recipe = by_id("dictionaries")
    folder = tmp_path / "dicts"
    folder.mkdir()
    (folder / "mine.dic").write_text("%\n1\tcalm\n%\ncalm\t1\n",
                                     encoding="utf-8")
    [mine] = pv.asset_manifest("dictionaries", folder)

    def plan(entries):
        return FeaturePlan(
            model="m [ridge]", slug="m", path="/lib/models/m.json",
            tables=(TablePlan(
                stem="dictionaries", target=recipe.target,
                instrument=dict(_instrument_defaults(recipe)),
                assets={"dict_paths": entries}, digest="1234abcd",
                grain={}),))

    same = compose(["dictionaries", "score_with_model"],
                   model_plans=[plan([mine])],
                   var_values={"dictionaries_path": str(folder)},
                   **_csv_kwargs())
    assert not any(s.get("for_model") for s in same["steps"]), \
        "the same list, byte for byte, should share one extraction"

    other = dict(mine, sha256="ff" * 32)
    differs = compose(["dictionaries", "score_with_model"],
                      model_plans=[plan([other])],
                      var_values={"dictionaries_path": str(folder)},
                      **_csv_kwargs())
    private = [s for s in differs["steps"] if s.get("for_model")]
    assert len(private) == 1, "a different list must be measured separately"
    assert private[0]["assets"]["take"] == {"dict_paths": ["mine.dic"]}


# ---------------------------------------------------------------------------
# the runner writes the carried files and binds them
# ---------------------------------------------------------------------------

def test_the_runner_writes_a_carried_file_beside_the_output_and_binds_it(
        tmp_path):
    from taters.pipelines.run_pipeline import _materialize_assets

    text = "ngram,frequency\npotato,9\n"
    model = tmp_path / "m.json"
    entry = {"name": "ngram_frequencies.csv", "sha256": pv.text_digest(text),
             "text": text}
    model.write_text(json.dumps({"assets": {"freq_list_csv": [entry],
                                            "dict_paths": [entry]}}),
                     encoding="utf-8")

    def one(*, freq_list_csv: str, out_features_csv): ...

    def many(*, dict_paths, out_features_csv): ...
    many.__annotations__["dict_paths"] = "Sequence[str]"

    out = tmp_path / "features" / "model_inputs" / "m" / "1234" / "dtm.csv"
    rendered = _materialize_assets(
        one, {"out_features_csv": str(out)},
        {"from": str(model), "take": {"freq_list_csv":
                                      ["ngram_frequencies.csv"]}})
    written = Path(rendered["freq_list_csv"])
    assert written == out.parent / "assets" / "freq_list_csv" / \
        "ngram_frequencies.csv"
    assert written.read_text(encoding="utf-8") == text

    rendered = _materialize_assets(
        many, {"out_features_csv": str(out)},
        {"from": str(model), "take": {"dict_paths": ["ngram_frequencies.csv"]}})
    assert rendered["dict_paths"] == [str(out.parent / "assets" / "dict_paths"
                                          / "ngram_frequencies.csv")]


def test_a_carried_file_the_model_no_longer_has_stops_the_step(tmp_path):
    from taters.pipelines.run_pipeline import _materialize_assets

    model = tmp_path / "m.json"
    model.write_text(json.dumps({"assets": {}}), encoding="utf-8")

    def one(*, freq_list_csv: str, out_features_csv): ...

    with pytest.raises(ValueError, match="no longer carries"):
        _materialize_assets(one, {"out_features_csv": str(tmp_path / "x.csv")},
                            {"from": str(model),
                             "take": {"freq_list_csv": ["gone.csv"]}})


# ---------------------------------------------------------------------------
# controls: the spreadsheet's own columns, carried along to the scoring step
# ---------------------------------------------------------------------------

def _controlled_plan(**extra):
    from taters.helpers.model_spec import FeaturePlan

    return FeaturePlan(model="ctl [ridge]", slug="ctl", path="/lib/ctl.json",
                       controls=("age", "gender"), **extra)


def test_a_model_fitted_with_controls_gets_the_spreadsheet_columns_carried():
    """No feature step produces `age` or `gender`; the metadata gather does,
    keyed like the features. It is added ahead of the scoring step, told to
    carry exactly those columns, and its table handed to the scorer -- a
    real run refused at the last step without this."""
    from taters.ui.compose import compose

    preset = compose(["readability", "score_with_model"],
                     model_plans=[_controlled_plan()], **_csv_kwargs())
    calls = [s["call"] for s in preset["steps"]]
    gather = calls.index("potato.helpers.csv_to_analysis_ready_csv")
    scoring = calls.index("potato.score_with_model")
    assert gather < scoring, calls
    step = preset["steps"][scoring]
    assert step["with"]["metadata_csv"] == "{{stats_metadata}}"
    assert preset["vars"]["stats_meta_carry"] == ["age", "gender"]
    meta = preset["steps"][gather]
    assert meta["with"]["carry_cols"] == "{{var:stats_meta_carry}}"


def test_the_carried_columns_join_the_ones_the_analyses_already_carry():
    """A run that also does statistics carries its group and outcome
    columns; the controls are added to that list, never in place of it."""
    from taters.ui.compose import compose

    preset = compose(["readability", "stats_correlations", "score_with_model"],
                     model_plans=[_controlled_plan()],
                     var_values={"stats_outcome_cols": ["openness"],
                                 "stats_meta_carry": ["openness", "age"]},
                     **_csv_kwargs())
    assert preset["vars"]["stats_meta_carry"] == ["openness", "age", "gender"]
    calls = [s["call"] for s in preset["steps"]]
    assert calls.count("potato.helpers.csv_to_analysis_ready_csv") == 1
    assert calls.index("potato.helpers.csv_to_analysis_ready_csv") \
        < calls.index("potato.score_with_model")


def test_a_controlled_model_cannot_score_a_folder_of_documents(tmp_path):
    """There is no spreadsheet to take the columns from. Refused by name,
    with the remedy, at compose time rather than after the extraction."""
    from taters.ui.compose import ComposeError, compose

    with pytest.raises(ComposeError, match="age, gender"):
        compose(["readability", "score_with_model"],
                model_plans=[_controlled_plan()], source="txt_dir",
                input_path=str(tmp_path), name="t")


def test_the_wizard_refuses_a_spreadsheet_lacking_a_control_column(tmp_path):
    """The header is known before anything runs, so the refusal comes while
    another model or spreadsheet can still be chosen."""
    from types import SimpleNamespace

    from taters.ui.compose import ComposeError
    from taters.ui.wizard import _check_controls

    sheet = tmp_path / "s.csv"
    sheet.write_text("pid,age,text\np1,30,hello\n", encoding="utf-8")
    src = SimpleNamespace(source="csv", path=sheet, delimiter=",")
    with pytest.raises(ComposeError, match="has no gender column"):
        _check_controls([_controlled_plan()], src)

    sheet.write_text("pid,age,gender,text\np1,30,f,hello\n", encoding="utf-8")
    _check_controls([_controlled_plan()], src)          # nothing to say
    _check_controls([_controlled_plan()],
                    SimpleNamespace(source="txt_dir", path=tmp_path,
                                    delimiter=","))    # not its call


def test_the_scorer_takes_the_controls_from_the_metadata_table(tmp_path):
    """Fitted with `gender` held constant, scored on features that do not
    carry it: the metadata table supplies it, and without one the refusal
    names `metadata_csv` instead of failing inside the design matrix."""
    np = pytest.importorskip("numpy")
    from taters.score_model import score_with_model
    from taters.stats.ridge import fit_ridge_csv

    rng = np.random.default_rng(3)
    n = 120
    f1 = rng.normal(size=n)
    gender = rng.choice(["female", "male"], n)
    age = 40 + 6 * f1 + 3 * (gender == "male") + rng.normal(size=n)

    def write(path, header, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return path

    train = write(tmp_path / "t.csv", ["text_id", "age", "gender", "f1"],
                  [[f"d{i}", f"{age[i]:.6f}", gender[i], f"{f1[i]:.10f}"]
                   for i in range(n)])
    fit_ridge_csv(table_csv=train, outcome_cols=["age"],
                  feature_sets={"lang": ["f1"]}, control_cols=["gender"],
                  categorical_controls=["gender"], out_dir=tmp_path / "fit",
                  verbose=False)
    model = tmp_path / "fit" / "models" / "ridge__lang__age.json"
    features = write(tmp_path / "features" / "f.csv", ["text_id", "f1"],
                     [[f"d{i}", f"{f1[i]:.10f}"] for i in range(n)])
    metadata = write(tmp_path / "gathered" / "metadata.csv",
                     ["text_id", "gender", "extra"],
                     [[f"d{i}", gender[i], "x"] for i in range(n)])

    with pytest.raises(ValueError, match="metadata_csv"):
        score_with_model(model_json=model, feature_csvs=[features],
                         out_csv=tmp_path / "no.csv", allow_unrecorded=True,
                         verbose=False)
    scored = _read(score_with_model(
        model_json=model, feature_csvs=[features], metadata_csv=metadata,
        out_csv=tmp_path / "s.csv", allow_unrecorded=True, verbose=False))
    assert len(scored) == n and all(r["pred_age"] for r in scored)
    # just the key and the prediction. not the raw control, not the metadata
    # table's other columns (we've shipped a scores file that carried both)
    assert list(scored[0]) == ["text_id", "pred_age"]


# ---------------------------------------------------------------------------
# the whole thing, on real analyzers: fit on one study, score another
# ---------------------------------------------------------------------------

def _study(path, *, n, words, seed):
    """Texts drawn from `words`, an outcome tied to their length, and an
    `age` column to hold constant."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "openness", "age", "text"])
        for i in range(n):
            k = 18 + rng.randrange(14)
            text = " ".join(rng.choice(words) for _ in range(k))
            w.writerow([f"p{i}", f"{2 + 0.05 * k + rng.gauss(0, .4):.3f}",
                        rng.randrange(18, 70), text.capitalize() + "."])
    return path


@pytest.mark.slow
def test_a_ridge_fitted_on_matrix_themes_and_tags_scores_a_second_study(
        tmp_path):
    """
    The end-to-end failure this module exists for, in miniature.

    Study one: parts of speech, a document-term matrix and its topic model,
    and a ridge fitted on all three. Study two uses a vocabulary that
    overlaps only partly -- so a matrix re-derived from it would have
    different columns, a refitted topic model different themes -- and is
    scored with the saved ridge. Before: refused over the missing columns.
    """
    pytest.importorskip("nltk")
    from taters.helpers.model_spec import describe, feature_plan
    from taters.pipelines.run_pipeline import run_preset
    from taters.ui.compose import compose

    first = _study(tmp_path / "one.csv", n=60, seed=1, words=(
        "the study found that participants who reported stress also slept "
        "less and felt tired because routines were disrupted although "
        "exercise helped them recover quickly").split())
    second = _study(tmp_path / "two.csv", n=24, seed=2, words=(
        "the survey showed that students who enjoyed music also read more "
        "and felt calm because evenings were quiet although homework kept "
        "them busy").split())

    fit = compose(["parts_of_speech", "doc_term_matrix", "topic_model_mem",
                   "stats_ridge_fit"],
                  source="csv", input_path=str(first), text_cols=["text"],
                  id_cols=["pid"], name="fit",
                  var_values={"stats_outcome_cols": ["openness"],
                              "stats_meta_carry": ["openness", "age"],
                              "stats_control_cols": ["age"],
                              "engine": "nltk"})
    work = tmp_path / "fit_run"
    work.mkdir()
    manifest = run_preset(fit, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]
    model = work / "stats_results" / "models" / "ridge__all__openness.json"
    info = describe(model)
    assert set(info.needs_tables) >= {"pos", "topic_model_mem"}
    plan = feature_plan(info)
    assert plan.replayable, plan.problems
    assert plan.controls == ("age",)

    score = compose(["parts_of_speech", "doc_term_matrix", "score_with_model"],
                    source="csv", input_path=str(second), text_cols=["text"],
                    id_cols=["pid"], name="score", model_plans=[plan],
                    var_values={"model_path": str(model), "engine": "nltk"})
    private = {s["call"] for s in score["steps"] if s.get("for_model")}
    assert "potato.text.apply_mem_model" in private
    assert "potato.text.build_doc_term_matrix" in private

    work2 = tmp_path / "score_run"
    work2.mkdir()
    manifest = run_preset(score, workers=1, work_dir=work2,
                          out_manifest=work2 / "run_manifest.json",
                          verbose=False)
    assert not manifest["errors"], manifest["errors"]
    scores = _read(work2 / "features" / "model_scores.csv")
    assert len(scores) == 24
    assert all(r["pred_openness"] for r in scores)

    # the private matrix has the model's columns, not the new corpus's
    fitted_cols = set(_read(work / "features" / "doc_term_matrix_count.csv")[0])
    replayed = next((work2 / "features" / "model_inputs").rglob(
        "doc_term_matrix_count.csv"))
    assert set(_read(replayed)[0]) == fitted_cols
    own = set(_read(work2 / "features" / "doc_term_matrix_count.csv")[0])
    assert own != fitted_cols, "the two vocabularies were meant to differ"
