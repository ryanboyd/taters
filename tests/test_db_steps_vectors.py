"""
Word vectors with a database on both ends.

The texts sit in a SQLite table, the features and the neighbors table go
back into that database, and the model, its weights and the report stay
files. The same model then scores the same texts from the database into
another table, and the neighbors table can be asked for on its own. The
corpus here has none of the usual probe words, so the fallback to the most
frequent content words is what gets exercised -- the table and the clouds
would otherwise be empty.
"""
from __future__ import annotations

import random
import sqlite3
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from taters.helpers import db, tables  # noqa: E402
from taters.helpers.text_gather import csv_to_analysis_ready_csv  # noqa: E402
from taters.text import word_vectors as wv  # noqa: E402

needs_gensim = pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gensim") is None,
    reason="gensim is not installed (pip install taters[vectors])")

# two vocabularies, neither sharing a word with DEFAULT_PROBES or a stop list
KITCHEN = ["potato", "gravy", "onion", "carrot", "butter", "skillet", "pepper", "broth"]
OFFICE = ["boss", "memo", "deadline", "invoice", "printer", "meeting", "budget", "payroll"]
FUNCTION = ["the", "and", "a", "of"]


def _texts(n: int = 80):
    rng = random.Random(7)
    for i in range(n):
        theme = KITCHEN if i % 2 == 0 else OFFICE
        words = [rng.choice(theme) for _ in range(rng.randint(12, 24))]
        words += [rng.choice(FUNCTION) for _ in range(6)]
        rng.shuffle(words)
        yield f"p{i}", " ".join(words)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """The corpus as a sqlite table behind the profile ``lab``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    tables.close_sessions()
    db._SECRETS.clear()
    dbfile = tmp_path / "lab.sqlite"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE study(rowid_ INTEGER PRIMARY KEY, pid TEXT, text TEXT)")
    for i, (pid, text) in enumerate(_texts(), 1):
        con.execute("INSERT INTO study VALUES (?, ?, ?)", (i, pid, text))
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(dbfile)))
    yield csv_to_analysis_ready_csv(csv_path="db://lab/study", out_csv="db://lab/taters/texts",
                                    text_cols=["text"], id_cols=["pid"], verbose=False)
    tables.close_sessions()
    db._SECRETS.clear()


def _dicx(path: Path) -> Path:
    path.write_text("DicTerm,food,work\npotato,X,\ngrav*,2,\nboss,,X\nmemo,,X\n",
                    encoding="utf-8")
    return path


def _train(gathered, tmp_path, **kw):
    args = dict(analysis_csv=gathered, out_model_json=tmp_path / "models" / "wv.json",
                concept_dicts=[_dicx(tmp_path / "kitchen.dicx")],
                min_count=2, vector_size=8, epochs=3, seed=1, reproducible=True,
                overwrite_existing=True, verbose=False)
    args.update(kw)
    return wv.train_word_vectors(**args)


@needs_gensim
def test_training_from_a_database_writes_its_tables_back_and_its_model_to_disk(lab, tmp_path):
    out = _train(lab, tmp_path, out_features_csv="db://lab/taters/features/word_vectors")
    assert isinstance(out, tables.TableRef) and out.is_db
    assert str(out) == "db://lab/taters/features/word_vectors"
    assert tables.count(out) == 80
    columns = tables.columns_of(out)
    assert columns[:4] == ["text_id", "token_count", "in_vocab_count", "wv_1"]
    assert columns[-2:] == ["sim_kitchen__food", "sim_kitchen__work"]
    rows = {r["text_id"]: r for r in tables.read_rows(out)}
    assert rows["p0"]["in_vocab_count"] and float(rows["p0"]["wv_1"]) == float(rows["p0"]["wv_1"])
    # the kitchen texts sit nearer the food concept, the office texts nearer work
    lean = {k: float(r["sim_kitchen__food"]) - float(r["sim_kitchen__work"])
            for k, r in rows.items()}
    assert min(lean[f"p{i}"] for i in range(0, 80, 2)) > max(lean[f"p{i}"] for i in range(1, 80, 2))

    # the model and its paperwork are files, beside nothing in particular
    models = tmp_path / "models"
    assert {p.name for p in models.iterdir()} >= {"wv.json", "wv.npy", "wv_report.md"}
    assert not (models / "wv_neighbors.csv").exists(), "the neighbors went to the database"
    assert not list(models.glob(".wv_*")), "the scratch files are gone"

    # the neighbors table went beside the features, in the database
    neighbors = tables.parse_ref("db://lab/taters/features/wv_neighbors")
    assert tables.exists(neighbors)
    assert tables.columns_of(neighbors) == ["probe", "rank", "word", "similarity"]
    by_probe = {}
    for r in tables.read_rows(neighbors):
        by_probe.setdefault(r["probe"], []).append(r["word"])
    assert {"sim_kitchen__food", "sim_kitchen__work"} <= set(by_probe)
    probes = set(by_probe) - {"sim_kitchen__food", "sim_kitchen__work"}
    # none of the usual probes is in this corpus, so the most frequent
    # content words stand in -- and the function words do not
    assert probes and probes <= set(KITCHEN) | set(OFFICE), probes
    assert not probes & set(FUNCTION)
    assert all(by_probe[p] for p in probes), "every stand-in has neighbors"
    report = (models / "wv_report.md").read_text(encoding="utf-8")
    assert "None of the usual probe words are in this model, so the most frequent " \
           "content words stand in." in report
    assert "`wv_neighbors`" in report and "`word_vectors`" in report


@needs_gensim
def test_the_default_destinations_follow_the_source_into_the_database(lab, tmp_path):
    out = wv.train_word_vectors(analysis_csv=lab, min_count=2, vector_size=8, epochs=2,
                                seed=1, reproducible=True, verbose=False)
    assert str(out) == "db://lab/taters/features/word_vectors/texts"
    assert tables.exists("db://lab/taters/features/word_vectors/word_vectors_neighbors")
    model = tmp_path / "db_local" / "lab" / "taters" / "features" / "word_vectors" / \
        "models" / "word_vectors.json"
    assert model.is_file() and model.with_suffix(".npy").is_file()
    # the same call again is handed back, not retrained
    again = wv.train_word_vectors(analysis_csv=lab, min_count=2, vector_size=8, epochs=2,
                                  seed=1, reproducible=True, verbose=False)
    assert again == out


@needs_gensim
def test_applying_the_model_from_the_database_reproduces_the_training_features(lab, tmp_path):
    trained = _train(lab, tmp_path, out_features_csv="db://lab/taters/features/word_vectors")
    applied = wv.apply_word_vectors(model_json=tmp_path / "models" / "wv.json",
                                    analysis_csv=lab,
                                    out_features_csv="db://lab/taters/features/applied",
                                    verbose=False)
    assert isinstance(applied, tables.TableRef) and str(applied) == "db://lab/taters/features/applied"
    assert tables.columns_of(applied) == tables.columns_of(trained)
    assert tables.fingerprint(applied) == tables.fingerprint(trained)
    # and the default destination for a database source is a table too
    by_default = wv.apply_word_vectors(model_json=tmp_path / "models" / "wv.json",
                                       analysis_csv=lab, verbose=False)
    assert str(by_default) == "db://lab/taters/features/word_vectors_applied/texts"
    assert tables.count(by_default) == 80


@needs_gensim
def test_describe_writes_the_neighbors_table_where_it_is_asked_to(lab, tmp_path):
    _train(lab, tmp_path, out_features_csv="db://lab/taters/features/word_vectors")
    model = tmp_path / "models" / "wv.json"
    out = wv.describe_word_vectors(model, "db://lab/taters/neighbors", probes="potato, zork",
                                   top_neighbors=3, verbose=False)
    assert str(out) == "db://lab/taters/neighbors"
    rows = list(tables.read_rows(out))
    assert [r["word"] for r in rows if r["probe"] == "zork"] == [""], "an unknown probe is a blank row"
    assert len([r for r in rows if r["probe"] == "potato"]) == 3
    # asked again, the table is handed back; a file default lands beside the model
    assert wv.describe_word_vectors(model, "db://lab/taters/neighbors", verbose=False) == out
    beside = wv.describe_word_vectors(model, verbose=False)
    assert beside == model.with_name("wv_neighbors.csv") and beside.is_file()
    assert "the" not in beside.read_text(encoding="utf-8-sig").split("\n")[1].split(",")[0]


@needs_gensim
def test_a_file_run_with_the_fallback_probes_says_so_in_the_report(tmp_path):
    corpus = tmp_path / "c.csv"
    corpus.write_text("text_id,text\n" + "".join(f"{i},{t}\n" for i, t in _texts(40)),
                      encoding="utf-8")
    out = wv.train_word_vectors(csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
                                out_features_csv=tmp_path / "features" / "wv.csv",
                                min_count=2, vector_size=8, epochs=2, seed=1,
                                reproducible=True, verbose=False)
    assert out == tmp_path / "features" / "wv.csv" and out.is_file()
    models = tmp_path / "features" / "models"
    neighbors = (models / "word_vectors_neighbors.csv").read_text(encoding="utf-8-sig")
    probes = {line.split(",")[0] for line in neighbors.splitlines()[1:]}
    assert probes and probes <= set(KITCHEN) | set(OFFICE), probes
    report = (models / "word_vectors_report.md").read_text(encoding="utf-8")
    assert "most frequent content words stand in" in report
