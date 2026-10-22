"""
The corpus tools on a database: frequency list, document-term matrix and the
topic models, with the text in a SQLite table and every table they write
landing back in the database -- and the same numbers as the CSV twin.

The contract is the one every converted step has: a ``db://`` address goes
wherever a path went, the row-shaped outputs (the frequency list, the matrix,
the scores, the loadings) become tables, and the things that are not rows
(the model JSON) stay on disk under the run's ``db_local/`` folder.
"""
from __future__ import annotations

import csv
import random
import sqlite3
from pathlib import Path

import pytest

from taters.helpers import db, tables
from taters.text.analyze_ngram_frequencies import analyze_ngram_frequencies
from taters.text.build_doc_term_matrix import build_doc_term_matrix

FOOD = "bread butter cheese dinner kitchen recipe".split()
WORK = "office meeting deadline manager project report".split()

#: Keep a forty-document toy corpus from being filtered away entirely.
LOOSE = dict(min_freq=1, min_obs_pct=0.0, min_token_count=1)
VOCAB = dict(vocab_min_freq=0, vocab_min_obs_pct=0, vocab_rule="top_n",
             vocab_top_n=0)

DB_TEXT = "db://lab/main/posts"
GATHERED_DB = "db://lab/main/taters/gathered/posts"


def _docs(n: int = 40):
    rng = random.Random(0)
    for i in range(n):
        family = FOOD if i % 2 == 0 else WORK
        words = [rng.choice(family) for _ in range(30)]
        words.append(rng.choice(FOOD + WORK))          # a little leakage
        yield f"d{i:02d}", " ".join(words)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """The same corpus as a CSV and as a sqlite table behind a profile; the
    run folder is the working directory, as it is under the runner."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    tables.close_sessions()
    db._SECRETS.clear()
    rows = list(_docs())
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([("text_id", "text"), *rows])
    path = tmp_path / "lab.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE posts(text_id TEXT PRIMARY KEY, text TEXT)")
    con.executemany("INSERT INTO posts VALUES (?, ?)", rows)
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(path)))
    yield corpus
    tables.close_sessions()
    db._SECRETS.clear()


def _source(corpus: Path, where: str) -> dict:
    """The text-input arguments for one twin: the file or the table."""
    if where == "db":
        return dict(csv_path=DB_TEXT, text_cols=["text"], id_cols=["text_id"],
                    gathered_csv=GATHERED_DB)
    return dict(csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
                gathered_csv=corpus.parent / "gathered.csv")


def _freqs(corpus, where, out):
    return analyze_ngram_frequencies(
        **_source(corpus, where), out_features_csv=out, overwrite_existing=True,
        ngram_n=2, **LOOSE)


def _same_table(a, b) -> None:
    """Same header, same cells, row for row -- whatever each one lives in."""
    assert tables.columns_of(a) == tables.columns_of(b)
    assert tables.count(a) == tables.count(b) > 0
    assert tables.fingerprint(a) == tables.fingerprint(b)


def _scores(ref) -> dict:
    return {r["text_id"]: {k: float(v) for k, v in r.items() if k != "text_id"}
            for r in tables.read_rows(ref)}


# ---------------------------------------------------------------------------
# the frequency list and the matrix
# ---------------------------------------------------------------------------

def test_the_frequency_list_lands_in_the_database_with_the_files_numbers(lab, tmp_path):
    from_file = _freqs(lab, "file", tmp_path / "freqs.csv")
    from_db = _freqs(lab, "db", "db://lab/main/taters/features/ngram_frequencies")
    assert isinstance(from_file, Path) and from_file.is_file()
    assert tables.parse_ref(from_db).is_db and tables.exists(from_db)
    assert tables.exists(GATHERED_DB), "the gather went to the database too"
    _same_table(from_file, from_db)
    rows = {r["ngram"]: r for r in tables.read_rows(from_db)}
    assert set(FOOD + WORK) <= set(rows)
    assert any(" " in g and r["npmi"] for g, r in rows.items()), "bigrams scored"
    assert not (tmp_path / "features").exists(), "no row-level table touched the disk"


def test_the_matrix_reads_a_database_vocabulary_and_writes_a_database_table(lab, tmp_path):
    freq_file = _freqs(lab, "file", tmp_path / "freqs.csv")
    freq_db = _freqs(lab, "db", "db://lab/main/taters/features/ngram_frequencies")
    dtm_file = build_doc_term_matrix(
        freq_list_csv=freq_file, **_source(lab, "file"),
        out_features_csv=tmp_path / "dtm.csv", overwrite_existing=True, **VOCAB)
    dtm_db = build_doc_term_matrix(
        freq_list_csv=str(freq_db), **_source(lab, "db"),
        out_features_csv="db://lab/main/taters/features/doc_term_matrix",
        overwrite_existing=True, **VOCAB)
    assert tables.parse_ref(dtm_db).is_db and tables.exists(dtm_db)
    _same_table(dtm_file, dtm_db)
    header = tables.columns_of(dtm_db)
    assert header[:2] == ["text_id", "token_count"] and set(FOOD + WORK) <= set(header)
    first = next(tables.read_cells(dtm_db))
    assert first[0] == "d00" and first[1] == "31"

    # the default name carries the weighting, in the database as in a file
    by_default = build_doc_term_matrix(
        freq_list_csv=str(freq_db), **_source(lab, "db"), weighting="binary",
        overwrite_existing=True, **VOCAB)
    assert str(by_default) == "db://lab/main/taters/features/doc_term_matrix/posts_binary"
    assert tables.exists(by_default)

    # and the resume contract holds for a table: there, so handed back
    again = build_doc_term_matrix(
        freq_list_csv=str(freq_db), **_source(lab, "db"),
        out_features_csv="db://lab/main/taters/features/doc_term_matrix", **VOCAB)
    assert str(again) == str(dtm_db)


# ---------------------------------------------------------------------------
# the topic models
# ---------------------------------------------------------------------------

def _mem(corpus, where, out, **kw):
    from taters.text.topic_model_mem import topic_model_mem

    return topic_model_mem(
        **_source(corpus, where), out_features_csv=out, overwrite_existing=True,
        n_components=2, **LOOSE, vocab_min_freq=0, vocab_min_obs_pct=0,
        vocab_rule="top_n", vocab_top_n=0, **kw)


def test_mem_fits_and_applies_with_every_table_in_the_database(lab, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from taters.text.topic_model_mem import apply_mem_model

    on_file = _mem(lab, "file", tmp_path / "mem" / "mem.csv")
    on_db = _mem(lab, "db", "db://lab/main/taters/features/topic_model_mem/mem")
    assert tables.parse_ref(on_db).is_db and tables.exists(on_db)

    # the scores, the three small tables and the model's own matrix are all
    # tables, so they are all in the database, beside the scores
    for stem in ("mem_loadings", "mem_eigenvalues", "mem_theme_variance",
                 "mem_matrix/freq_list", "mem_matrix/dtm"):
        ref = f"db://lab/main/taters/features/topic_model_mem/{stem}"
        assert tables.exists(ref), stem
        twin = tmp_path / "mem" / f"{stem}.csv"
        assert twin.is_file(), f"the file twin {stem}"
        _same_table(twin, ref)
    _same_table(on_file, on_db)

    # the model is not rows, so it is on disk, under the run's db_local folder
    model = tmp_path / "db_local" / "lab" / "main" / "taters" / "features" / "topic_model_mem" / "mem_model.json"
    assert model.is_file()
    assert not (tmp_path / "features").exists() and not (tmp_path / "db:").exists()

    # applied to its own corpus, a saved model reproduces the fit -- the
    # roundtrip contract, now across a database as well
    applied = apply_mem_model(
        model_json=model, **_source(lab, "db"),
        out_features_csv="db://lab/main/taters/features/topic_model_mem/mem_applied",
        overwrite_existing=True)
    assert tables.exists(applied)
    assert _scores(applied) == _scores(on_db) == _scores(on_file)

    # and a database address for the model file is a file under db_local
    # too, which is what a pipeline template renders on a database run
    again = _mem(lab, "db", "db://lab/main/taters/features/topic_model_mem/mem2",
                 out_model_json="db://lab/main/taters/features/topic_model_mem/mem2_model.json")
    assert tables.exists(again)
    assert (tmp_path / "db_local" / "lab" / "main" / "taters" / "features" / "topic_model_mem"
            / "mem2_model.json").is_file()


def test_lda_fits_and_applies_through_the_database(lab, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from taters.text.topic_model_lda import apply_lda_model, topic_model_lda

    def fit(where, out):
        return topic_model_lda(
            **_source(lab, where), out_features_csv=out, overwrite_existing=True,
            n_topics=2, passes=3, **LOOSE, **VOCAB)

    on_file = fit("file", tmp_path / "lda" / "lda.csv")
    on_db = fit("db", "db://lab/main/taters/features/topic_model_lda/lda")
    _same_table(on_file, on_db)
    for stem in ("lda_loadings", "lda_top_terms", "lda_matrix/dtm"):
        ref = f"db://lab/main/taters/features/topic_model_lda/{stem}"
        assert tables.exists(ref), stem
        _same_table(tmp_path / "lda" / f"{stem}.csv", ref)
    model = tmp_path / "db_local" / "lab" / "main" / "taters" / "features" / "topic_model_lda" / "lda_model.json"
    assert model.is_file()

    applied = apply_lda_model(
        model_json=model, **_source(lab, "db"),
        out_features_csv="db://lab/main/taters/features/topic_model_lda/lda_applied",
        overwrite_existing=True)
    assert _scores(applied) == _scores(on_db)
    assert tables.exists("db://lab/main/taters/features/topic_model_lda/lda_applied_matrix/freq_list"), \
        "apply's frozen vocabulary went where its matrix goes: into the database"


# ---------------------------------------------------------------------------
# the word clouds read database tables
# ---------------------------------------------------------------------------

def test_word_clouds_draw_from_database_tables(lab, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    pytest.importorskip("PIL")
    from taters.figures import wordclouds as wc

    _mem(lab, "db", "db://lab/main/taters/features/topic_model_mem/mem")
    loadings = "db://lab/main/taters/features/topic_model_mem/mem_loadings"
    folder = wc.theme_wordclouds(loadings, tmp_path / "clouds", verbose=False)
    assert sorted(p.name for p in folder.glob("*.png")) == ["Theme_1.png", "Theme_2.png"]
    index = (folder / "index.md").read_text(encoding="utf-8")
    assert "mem_loadings" in index, \
        "the index names the table it was drawn from"

    freqs = _freqs(lab, "db", "db://lab/main/taters/features/ngram_frequencies")
    folder = wc.frequency_wordclouds(freqs, tmp_path / "freq_clouds", verbose=False)
    assert (folder / "top_words.png").is_file()

    # a db:// address for the picture folder (what a template renders on a
    # database run) is a folder under db_local, not a folder called "db:"
    folder = wc.frequency_wordclouds(
        freqs, "db://lab/main/taters/features/figures/wordclouds/ngram_frequencies",
        verbose=False)
    assert folder == (tmp_path / "db_local" / "lab" / "main" / "taters" / "features" / "figures"
                      / "wordclouds" / "ngram_frequencies")
    assert (folder / "top_words.png").is_file() and not (tmp_path / "db:").exists()


def test_the_loadings_of_a_database_feature_set_are_found_from_the_sidecar(lab, tmp_path):
    """The assemble sidecar records the features table's address; the
    loadings sit beside it, in the database."""
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from taters.figures.wordclouds import _theme_loadings_for

    features = "db://lab/main/taters/features/topic_model_mem/mem"
    _mem(lab, "db", features)
    stats_dir = tmp_path / "stats_results"
    stats_dir.mkdir()
    found = _theme_loadings_for("mem", ["Theme_1", "Theme_2"], {"mem": features},
                                stats_dir, "utf-8-sig")
    assert str(found) == "db://lab/main/taters/features/topic_model_mem/mem_loadings"
    assert _theme_loadings_for("mem", ["Theme_1"], {"mem": "db://lab/main/nowhere/mem"},
                               stats_dir, "utf-8-sig") is None
