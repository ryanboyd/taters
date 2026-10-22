"""
The scoring-side steps read and write database tables like files.

Each step runs twice on the same rows -- once from CSV files to a CSV, once
from tables in a SQLite database to a table in it -- and the two answers
have to agree cell for cell: a PCA fit, a saved model scored against the
run's feature tables (one model and several), a feature table averaged up a
level, and a wide embeddings table aggregated per group. The database copy
of the embeddings is wide enough to be stored packed, so the aggregation
also proves the packed layout reads back as plain cells.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

from csvhelpers import write_rows  # noqa: E402
from taters.helpers import db, tables  # noqa: E402
from taters.helpers.feature_average import average_feature_table  # noqa: E402
from taters.helpers.feature_gather import aggregate_features, make_plan  # noqa: E402
from taters.score_model import score_with_model  # noqa: E402
from taters.stats import pca  # noqa: E402
from taters.stats.ridge import fit_ridge_csv  # noqa: E402


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A sqlite connection called ``lab``, and a clean session either side."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    tables.close_sessions()
    db._SECRETS.clear()
    db.save_profile(db.Profile(name="lab", dialect="sqlite",
                               path=str(tmp_path / "lab.sqlite")))
    yield tmp_path
    tables.close_sessions()
    db._SECRETS.clear()


def _both(tmp_path, name, header, rows, hints=None):
    """The same rows as a CSV and as a table in ``lab``; returns both refs."""
    path = write_rows(tmp_path / f"{name}.csv", header, rows)
    ref = f"db://lab/main/study/{name}"
    with tables.write_table(ref, header, hints=hints) as w:
        w.writerows(rows)
    return path, ref


def _same_rows(a, b, *, floats=True):
    """Both tables hold the same cells in the same order; numbers compared
    as numbers, since a database hands back ``2.0`` where a file says
    ``2``."""
    ra = list(tables.read_rows(a))
    rb = list(tables.read_rows(b))
    assert tables.columns_of(a) == tables.columns_of(b)
    assert len(ra) == len(rb)
    for x, y in zip(ra, rb):
        for col in x:
            if x[col] == y[col]:
                continue
            assert floats, f"{col}: {x[col]!r} != {y[col]!r}"
            assert float(x[col]) == pytest.approx(float(y[col]), rel=1e-9, abs=1e-9), col


def _feature_rows(n=60, seed=3):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = f1 + f2 + rng.normal(scale=0.3, size=n)
    return f1, f2, f3


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def test_a_pca_fits_a_database_table_into_database_tables(lab):
    f1, f2, f3 = _feature_rows()
    header = ["text_id", "f1", "f2", "f3"]
    rows = [[f"d{i}", f"{f1[i]:.8f}", f"{f2[i]:.8f}", f"{f3[i]:.8f}"]
            for i in range(len(f1))]
    path, ref = _both(lab, "feats", header, rows)

    from_file = pca.fit_pca_csv(path, n_components=2, overwrite_existing=True,
                                out_model_json=lab / "file_pca_model.json")
    from_db = pca.fit_pca_csv(ref, n_components=2, overwrite_existing=True,
                              out_model_json=lab / "db_pca_model.json")

    assert tables.parse_ref(from_db).is_db
    assert str(from_db) == "db://lab/main/study/feats_pca_scores"
    _same_rows(from_file, from_db)
    _same_rows(lab / "feats_pca_loadings.csv", "db://lab/main/study/feats_pca_loadings")
    _same_rows(lab / "feats_pca_eigenvalues.csv", "db://lab/main/study/feats_pca_eigenvalues")
    # the model is not rows; it stays on disk, and reads back the same
    a = json.loads((lab / "file_pca_model.json").read_text(encoding="utf-8"))
    b = json.loads((lab / "db_pca_model.json").read_text(encoding="utf-8"))
    assert a["model"]["projection"] == b["model"]["projection"]
    assert a["features"] == b["features"] == ["f1", "f2", "f3"]

    # and the fit is skip-if-exists in the database too
    again = pca.fit_pca_csv(ref, n_components=2,
                            out_model_json=lab / "db_pca_model.json")
    assert again == from_db

    # applying the saved model to the database table reproduces the scores
    reapplied = pca.apply_pca_csv(lab / "db_pca_model.json", ref,
                                  out_scores_csv="db://lab/main/study/reapplied",
                                  overwrite_existing=True)
    _same_rows(reapplied, from_db)


def test_the_pca_model_defaults_beside_a_database_table_on_disk(lab):
    f1, f2, f3 = _feature_rows()
    header = ["text_id", "f1", "f2", "f3"]
    rows = [[f"d{i}", f"{f1[i]:.8f}", f"{f2[i]:.8f}", f"{f3[i]:.8f}"]
            for i in range(len(f1))]
    _, ref = _both(lab, "feats", header, rows)
    pca.fit_pca_csv(ref, n_components=2, overwrite_existing=True)
    model = tables.parse_ref(ref).local_dir() / "feats_pca_model.json"
    assert model.is_file()
    assert json.loads(model.read_text(encoding="utf-8"))["kind"] == "taters-pca-model"


# ---------------------------------------------------------------------------
# Scoring with a saved model
# ---------------------------------------------------------------------------

def _ridge(tmp_path, name="ridge", features=("f1", "f2")):
    """A ridge fitted on a CSV training table, since the fit is file-side
    here; what we are testing is scoring it."""
    rng = np.random.default_rng(11)
    n = 120
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    age = 40 + 8 * f1 - 2 * f2 + rng.normal(scale=1.0, size=n)
    train = write_rows(tmp_path / f"{name}_train.csv", ["text_id", "age", "f1", "f2"],
                       [[f"t{i}", f"{age[i]:.6f}", f"{f1[i]:.10f}", f"{f2[i]:.10f}"]
                        for i in range(n)])
    fit_ridge_csv(table_csv=train, outcome_cols=["age"],
                  feature_sets={"lang": list(features)},
                  out_dir=tmp_path / f"fit_{name}", verbose=False)
    model = tmp_path / f"fit_{name}" / "models" / "ridge__lang__age.json"
    if name != "ridge":
        doc = json.loads(model.read_text(encoding="utf-8"))
        doc["name"] = name
        model.write_text(json.dumps(doc), encoding="utf-8")
    return model


def _run_features(tmp_path, n=60):
    f1, f2, _ = _feature_rows(n)
    one = _both(tmp_path, "one", ["text_id", "f1"],
                [[f"d{i}", f"{f1[i]:.10f}"] for i in range(n)])
    two = _both(tmp_path, "two", ["text_id", "f2"],
                [[f"d{i}", f"{f2[i]:.10f}"] for i in range(n)])
    return [one[0], two[0]], [one[1], two[1]]


def test_one_model_scores_database_feature_tables_into_a_database_table(lab):
    model = _ridge(lab)
    files, refs = _run_features(lab)

    from_file = score_with_model(model_json=model, feature_csvs=files,
                                 out_csv=lab / "scores.csv",
                                 allow_unrecorded=True, verbose=False)
    from_db = score_with_model(model_json=model, feature_csvs=refs,
                               out_csv="db://lab/main/study/scores",
                               allow_unrecorded=True, verbose=False)
    assert tables.parse_ref(from_db).is_db
    assert tables.columns_of(from_db) == ["text_id", "pred_age"]
    _same_rows(from_file, from_db)
    # the joined table the scoring read is not left behind in the database
    assert not tables.exists("db://lab/main/study/scores_inputs/scoring_table")
    assert set(tables.tables_under("db://lab/main/study/")) == {
        tables.parse_ref(r) for r in [*refs, "db://lab/main/study/scores"]}

    kept = score_with_model(model_json=model, feature_csvs=refs,
                            out_csv="db://lab/main/study/kept", keep_inputs=True,
                            allow_unrecorded=True, verbose=False)
    assert tables.exists("db://lab/main/study/kept_inputs/scoring_table")
    assert (tables.parse_ref(kept).local_dir() / "kept_inputs"
            / "assemble_manifest.json").is_file()


def test_the_default_output_lands_in_the_features_database(lab):
    model = _ridge(lab)
    _, refs = _run_features(lab)
    out = score_with_model(model_json=model, feature_csvs=refs,
                           allow_unrecorded=True, verbose=False)
    assert str(out) == "db://lab/main/taters/features/model_scores/ridge__lang__age"
    assert tables.count(out) == 60


def test_several_models_merge_into_a_database_table_with_each_beside_it(lab):
    first = _ridge(lab)
    second = _ridge(lab, "second")
    files, refs = _run_features(lab)

    from_file = score_with_model(model_json=[first, second], feature_csvs=files,
                                 out_csv=lab / "all.csv",
                                 allow_unrecorded=True, verbose=False)
    from_db = score_with_model(model_json=[first, second], feature_csvs=refs,
                               out_csv="db://lab/main/study/all",
                               allow_unrecorded=True, verbose=False)
    assert tables.columns_of(from_db) == ["text_id", "ridge__lang__age__pred_age",
                                          "second__pred_age"]
    _same_rows(from_file, from_db)
    _same_rows(lab / "all" / "ridge__lang__age.csv", "db://lab/main/study/all/ridge__lang__age")
    _same_rows(lab / "all" / "second.csv", "db://lab/main/study/all/second")
    assert tables.read_sidecar("db://lab/main/study/all", "unscored") is None


def test_per_model_tables_that_do_not_line_up_are_merged_by_key(lab):
    """Every model reads the run's shared tables, so the per-model score
    tables line up row for row and the merge walks them in lockstep. When
    they do not -- pinned here on hand-written database tables, as the
    file test pins it on files -- the merge falls back to keying rows: an
    outer join in the first model's order, and the accounting beside the
    merged database table names the model."""
    from taters.helpers.model_spec import describe
    from taters.score_model import _merge_scores, _merge_unscored
    from taters.stats._fit_common import write_unscored

    infos = [describe(_ridge(lab)), describe(_ridge(lab, "second"))]
    a = "db://lab/main/study/all/ridge__lang__age"
    b = "db://lab/main/study/all/second"
    with tables.write_table(a, ["text_id", "pred_age"]) as w:
        w.writerows([["d2", "1"], ["d1", "2"]])
    with tables.write_table(b, ["text_id", "pred_age"]) as w:
        w.writerows([["d1", "3"], ["d9", "4"]])
    write_unscored(b, [("missing from feature table", "two", 1)], encoding="utf-8-sig")

    merged = "db://lab/main/study/all"
    _merge_scores(infos, [a, b], merged, key_cols=["text_id"], encoding="utf-8-sig")
    _merge_unscored(infos, [a, b], merged, encoding="utf-8-sig")
    rows = list(tables.read_rows(merged))
    assert tables.columns_of(merged) == ["text_id", "ridge__lang__age__pred_age",
                                         "second__pred_age"]
    assert [r["text_id"] for r in rows] == ["d2", "d1", "d9"]
    assert rows[0]["second__pred_age"] == "" and rows[2]["ridge__lang__age__pred_age"] == ""
    assert rows[1] == {"text_id": "d1", "ridge__lang__age__pred_age": "2",
                       "second__pred_age": "3"}
    account = tables.read_sidecar(merged, "unscored")
    assert account["columns"] == ["model", "reason", "detail", "rows"]
    assert account["entries"] == [["second", "missing from feature table", "two", 1]]

    # lined up, the lockstep walk gives the same table, and nothing unscored
    # means no accounting beside it
    with tables.write_table(b, ["text_id", "pred_age"]) as w:
        w.writerows([["d2", "5"], ["d1", "3"]])
    write_unscored(b, [], encoding="utf-8-sig")
    _merge_scores(infos, [a, b], merged, key_cols=["text_id"], encoding="utf-8-sig")
    _merge_unscored(infos, [a, b], merged, encoding="utf-8-sig")
    assert [(r["text_id"], r["second__pred_age"]) for r in tables.read_rows(merged)] == \
        [("d2", "5"), ("d1", "3")]
    assert tables.read_sidecar(merged, "unscored") is None


# ---------------------------------------------------------------------------
# Averaging a feature table up a level
# ---------------------------------------------------------------------------

def test_a_feature_table_averages_from_and_into_the_database(lab):
    keys_header = ["text_id", "conv", "speaker"]
    keys_rows = [["row_1", "c1", "alice"], ["row_2", "c1", "alice"],
                 ["row_3", "c1", "bob"], ["row_4", "c2", "alice"]]
    feat_header = ["text_id", "words", "grade", "label"]
    feat_rows = [["row_1", "3", "1", "x"], ["row_2", "2", "", "y"],
                 ["row_3", "1", "5", "x"], ["row_4", "4", "9", "z"]]
    keys_file, keys_ref = _both(lab, "keys", keys_header, keys_rows)
    feat_file, feat_ref = _both(lab, "words", feat_header, feat_rows)

    from_file = average_feature_table(
        in_csv=feat_file, out_csv=lab / "by_speaker.csv", keys_csv=keys_file,
        group_by=["conv", "speaker"], verbose=False)
    from_db = average_feature_table(
        in_csv=feat_ref, out_csv="db://lab/main/study/by_speaker", keys_csv=keys_ref,
        group_by=["conv", "speaker"], verbose=False)
    assert tables.parse_ref(from_db).is_db
    assert tables.columns_of(from_db) == ["text_id", "conv", "speaker",
                                          "rows_averaged", "words", "grade"]
    _same_rows(from_file, from_db)
    rows = {(r["conv"], r["speaker"]): r for r in tables.read_rows(from_db)}
    assert float(rows[("c1", "alice")]["words"]) == pytest.approx(2.5)
    assert float(rows[("c1", "alice")]["grade"]) == pytest.approx(1.0), \
        "a blank cell is skipped, not counted as zero"
    assert rows[("c1", "alice")]["rows_averaged"] == "2"

    # the second level reads the first's output straight from the database
    chained = average_feature_table(
        in_csv=from_db, out_csv="db://lab/main/study/by_conv", group_by=["conv"],
        verbose=False)
    by_conv = {r["conv"]: r for r in tables.read_rows(chained)}
    assert float(by_conv["c1"]["words"]) == pytest.approx(1.75)


# ---------------------------------------------------------------------------
# Aggregating a wide embeddings table
# ---------------------------------------------------------------------------

def test_a_wide_embeddings_table_aggregates_from_the_database(lab):
    """2,000 embedding columns is past what sqlite will hold as real
    columns, so the database copy is stored packed; the aggregation reads
    it back the same as the file and groups it the same."""
    width = 2000
    rng = np.random.default_rng(5)
    speakers = ["alice", "bob", "alice", "", "bob", "alice"]
    header = ["text_id", "speaker", "text", *(f"e{j}" for j in range(width))]
    rows = []
    for i, who in enumerate(speakers):
        vec = rng.normal(size=width)
        rows.append([f"u{i}", who, f"words {i}", *(f"{v:.6f}" for v in vec)])
    path, ref = _both(lab, "embeds", header, rows,
                      hints={"text_id": "text", "speaker": "text", "text": "text"})
    assert tables.columns_of(ref) == header

    plan = make_plan(group_by=["speaker"], per_file=False,
                     stats=["mean", "std", "count"], exclude_cols=["text_id", "text"])
    from_file = aggregate_features(root_dir=path, plan=plan, out_csv=lab / "agg.csv",
                                   verbose=False)
    from_db = aggregate_features(root_dir=ref, plan=plan, out_csv="db://lab/main/study/agg",
                                 verbose=False)
    assert tables.parse_ref(from_db).is_db
    cols = tables.columns_of(from_db)
    assert cols[:1] == ["speaker"]
    assert len(cols) == 1 + 3 * width
    _same_rows(from_file, from_db)

    got = {r["speaker"]: r for r in tables.read_rows(from_db)}
    assert list(got) == ["alice", "bob", ""], "sorted, the blank key last"
    alice = [i for i, who in enumerate(speakers) if who == "alice"]
    want = np.mean([[float(c) for c in rows[i][3:]] for i in alice], axis=0)
    assert float(got["alice"]["e0__mean"]) == pytest.approx(want[0], abs=1e-9)
    assert float(got["alice"]["e1999__mean"]) == pytest.approx(want[-1], abs=1e-9)
    assert got["alice"]["e7__count"] == "3"
    assert got[""]["e7__count"] == "1" and got[""]["e7__std"] == ""

    dropped = aggregate_features(root_dir=ref, plan=make_plan(
        group_by=["speaker"], per_file=False, stats=["mean"],
        exclude_cols=["text_id", "text"], dropna=True),
        out_csv="db://lab/main/study/agg_dropna", verbose=False)
    assert [r["speaker"] for r in tables.read_rows(dropped)] == ["alice", "bob"]


def test_running_statistics_agree_with_pandas(lab):
    """The streaming aggregation replaced a pandas groupby; the numbers it
    writes have to be the ones pandas writes, including a sample deviation
    and a blank where a statistic is undefined."""
    import pandas as pd

    rng = np.random.default_rng(8)
    header = ["speaker", "a", "b", "note"]
    rows = []
    for i in range(50):
        who = ["x", "y", "z"][i % 3]
        a = f"{rng.normal():.5f}" if i % 7 else ""
        rows.append([who, a, f"{rng.uniform(0, 10):.5f}", "words"])
    rows.append(["solo", "1.5", "", "words"])
    path = write_rows(lab / "feats" / "s.csv", header, rows)
    plan = make_plan(group_by=["speaker"], per_file=True,
                     stats=["mean", "std", "var", "sum", "count", "min", "max"])
    out = aggregate_features(root_dir=path.parent, plan=plan, out_csv=lab / "agg.csv",
                             verbose=False)
    got = pd.read_csv(out, dtype=str, keep_default_na=False)
    frame = pd.read_csv(path, dtype="object")
    frame["a"] = pd.to_numeric(frame["a"], errors="coerce")
    frame["b"] = pd.to_numeric(frame["b"], errors="coerce")
    want = frame.groupby("speaker").agg({"a": list(plan.stats), "b": list(plan.stats)})
    assert list(got.columns) == ["source", "speaker",
                                 *(f"{c}__{s}" for c in ("a", "b") for s in plan.stats)]
    assert list(got["speaker"]) == list(want.index)
    for _, row in got.iterrows():
        for c in ("a", "b"):
            for s in plan.stats:
                expect = want.loc[row["speaker"], (c, s)]
                cell = row[f"{c}__{s}"]
                if pd.isna(expect):
                    assert cell == "", (c, s, row["speaker"])
                else:
                    assert float(cell) == pytest.approx(float(expect), rel=1e-12), (c, s)
    assert not [c for c in got.columns if c.startswith("note__")], \
        "a column of words is not a feature"
