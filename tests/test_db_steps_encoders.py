"""
The encoder-side text steps read from and write to a database.

Entropy, the Hugging Face classifier, the fine-tuned predictor and the two
encoder trainers all take a gathered table that may live in a database. Each
is run here on a sqlite table (with a database destination where the step
writes rows) and checked against the same step on the CSV copy. The scoring
steps are also checked for *how* they read: a 5,000-row table must reach the
model in bounded slices, never as one list.
"""
from __future__ import annotations

import csv
import random
import sqlite3
from pathlib import Path

import pytest

from csvhelpers import write_rows
from taters.helpers import db, tables
from taters.helpers.text_gather import csv_to_analysis_ready_csv
from taters.text.analyze_entropy import analyze_entropy

FOOD = "potato gravy butter salt dinner supper kitchen recipe".split()
WORK = "meeting deadline email boss office budget".split()
ROWS = [
    ["p1", "1", "A", "3.1", "ab ab ab ab ab ab cd cd cd cd cd cd"],
    ["p1", "2", "A", "5.0", "ab cd ab cd cd ab cd ab ab cd cd ab"],
    ["p2", "1", "B", "", "lonely words here and there"],
    ["p2", "2", "B", "7.2", ""],
    ["p3", "1", "A", "2.4", "third person speaks of a third thing"],
]
HEADER = ["pid", "session", "cond", "score", "text"]


@pytest.fixture
def both(tmp_path, monkeypatch):
    """The same rows as a CSV and as a sqlite table behind a profile."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    tables.close_sessions()
    db._SECRETS.clear()
    csv_path = write_rows(tmp_path / "study.csv", HEADER, ROWS)
    dbfile = tmp_path / "lab.sqlite"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE study(rowid_ INTEGER PRIMARY KEY, pid TEXT, session INTEGER, "
                "cond TEXT, score REAL, text TEXT)")
    for i, r in enumerate(ROWS, 1):
        con.execute("INSERT INTO study VALUES (?, ?, ?, ?, ?, ?)",
                    (i, r[0], int(r[1]), r[2], float(r[3]) if r[3] else None, r[4] or None))
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(dbfile)))
    yield csv_path, "db://lab/study"
    tables.close_sessions()
    db._SECRETS.clear()


def _gather(source, out, **kw):
    return csv_to_analysis_ready_csv(csv_path=source, out_csv=out, verbose=False,
                                     text_cols=["text"], id_cols=["pid", "session"], **kw)


def _same_rows(a, b):
    """Two tables agree: same columns, same words, numbers equal as numbers
    (a file keeps the digits it was written with, a database its own)."""
    assert tables.columns_of(a) == tables.columns_of(b)
    ra, rb = list(tables.read_rows(a)), list(tables.read_rows(b))
    assert len(ra) == len(rb) > 0
    for x, y in zip(ra, rb):
        for col in x:
            try:
                fx, fy = float(x[col]), float(y[col])
            except ValueError:
                assert x[col] == y[col], col
            else:
                assert fx == pytest.approx(fy, rel=1e-5), col


def _big_texts(path: Path, n: int) -> Path:
    rng = random.Random(0)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text"])
        for i in range(n):
            pool = FOOD if i % 2 else WORK
            w.writerow([f"d{i}", " ".join(rng.choice(pool) for _ in range(rng.randint(3, 9)))])
    return path


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_entropy_on_a_database_table_matches_the_file(both, tmp_path):
    csv_path, table = both
    from_file = _gather(csv_path, tmp_path / "g.csv")
    from_db = _gather(table, "db://lab/taters/gathered")
    file_out = analyze_entropy(analysis_csv=from_file, out_features_csv=tmp_path / "e.csv",
                               verbose=False)
    db_out = analyze_entropy(analysis_csv=from_db, out_features_csv="db://lab/taters/entropy",
                             verbose=False)
    assert isinstance(file_out, Path) and tables.parse_ref(db_out).is_db
    _same_rows(file_out, db_out)
    assert "ent_word_shannon_bits" in tables.columns_of(db_out)
    # the second call hands the table back rather than measuring it again
    assert analyze_entropy(analysis_csv=from_db, out_features_csv="db://lab/taters/entropy",
                           verbose=False) == db_out


def test_entropy_defaults_to_a_table_beside_a_database_source(both):
    _csv, table = both
    from_db = _gather(table, "db://lab/taters/gathered")
    out = analyze_entropy(analysis_csv=from_db, verbose=False)
    assert str(out) == "db://lab/taters/features/entropy/gathered"
    assert tables.exists(out) and tables.count(out) == 4


# ---------------------------------------------------------------------------
# The Hugging Face classifier
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def classifier(tmp_path_factory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_encoder import build_classifier

    return build_classifier(tmp_path_factory.mktemp("clf") / "tiny_clf",
                            labels=("food", "work"))


def test_the_classifier_scores_a_database_table_into_the_database(both, classifier, tmp_path):
    from taters.text.hf_classifier import apply_hf_classifier, import_hf_classifier

    csv_path, table = both
    manifest = import_hf_classifier(source=classifier, out_dir=tmp_path / "m",
                                    name="clf", outcome="topic")
    from_file = _gather(csv_path, tmp_path / "g.csv")
    from_db = _gather(table, "db://lab/taters/gathered")
    file_out = apply_hf_classifier(model_json=manifest, analysis_csv=from_file,
                                   out_features_csv=tmp_path / "s.csv", device="cpu",
                                   verbose=False, max_length=16)
    db_out = apply_hf_classifier(model_json=manifest, analysis_csv=from_db,
                                 out_features_csv="db://lab/taters/scored", device="cpu",
                                 verbose=False, max_length=16)
    assert tables.parse_ref(db_out).is_db
    _same_rows(file_out, db_out)
    assert tables.columns_of(db_out)[:4] == ["text_id", "token_count", "n_windows", "pred_topic"]


def test_the_classifier_reads_a_big_table_in_bounded_slices(classifier, tmp_path, monkeypatch):
    """5,000 rows reach the model in slices no larger than _TEXTS_PER_CALL,
    and every row comes out the other side."""
    from taters.text import hf_classifier as hf

    manifest = hf.import_hf_classifier(source=classifier, out_dir=tmp_path / "m",
                                       name="clf", outcome="topic")
    seen: list = []

    def fake_predict(model, tokenizer, texts, task, **kw):
        seen.append(len(texts))
        return ([[0.5, 0.5] for _ in texts], [1] * len(texts), [1] * len(texts))

    monkeypatch.setattr(hf, "_predict", fake_predict)
    out = hf.apply_hf_classifier(model_json=manifest, csv_path=_big_texts(tmp_path / "big.csv", 5000),
                                 text_cols=["text"], id_cols=["pid"],
                                 out_features_csv=tmp_path / "s.csv", device="cpu",
                                 verbose=False)
    assert seen and max(seen) <= hf._TEXTS_PER_CALL < 5000 and sum(seen) == 5000
    assert len(seen) == -(-5000 // hf._TEXTS_PER_CALL)
    assert tables.count(out) == 5000


# ---------------------------------------------------------------------------
# The fine-tuned predictor
# ---------------------------------------------------------------------------

def _study(path: Path, n=60, seed=1) -> Path:
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text", "score", "topic"])
        for i in range(n):
            food = i % 2 == 0
            words = [rng.choice(FOOD if food else WORK) for _ in range(rng.randint(10, 20))]
            score = (3.0 if food else 1.0) + rng.gauss(0, 0.3)
            # a handful of rows with no outcome at all, which the trainer leaves out
            w.writerow([f"d{i}", " ".join(words), "" if i % 15 == 7 else f"{score:.2f}",
                        "" if i % 15 == 7 else ("food" if food else "work")])
    return path


@pytest.fixture(scope="module")
def encoder(tmp_path_factory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_encoder import build

    return build(tmp_path_factory.mktemp("enc") / "tiny")


@pytest.fixture(scope="module")
def predictor(tmp_path_factory, encoder) -> Path:
    """One fitted predictor for the module; fitting is the slow part."""
    from taters.text.finetune_predictor import finetune_text_predictor

    root = tmp_path_factory.mktemp("fit")
    finetune_text_predictor(csv_path=_study(root / "s.csv"), text_cols=["text"], id_cols=["pid"],
                            outcome_cols=["score", "topic"], out_dir=root / "stats",
                            base_model=str(encoder), n_folds=2, epochs=1, max_length=32,
                            batch_size=8, learning_rate=1e-3, device="cpu", precision="fp32",
                            verbose=False)
    return root / "stats" / "models" / "text_predictor__score_topic.json"


def test_the_predictor_scores_a_database_table_into_the_database(both, predictor, tmp_path):
    from taters.text.finetune_predictor import apply_text_predictor

    csv_path, table = both
    from_file = _gather(csv_path, tmp_path / "g.csv")
    from_db = _gather(table, "db://lab/taters/gathered")
    file_out = apply_text_predictor(model_json=predictor, analysis_csv=from_file,
                                    out_features_csv=tmp_path / "s.csv", device="cpu",
                                    verbose=False)
    db_out = apply_text_predictor(model_json=predictor, analysis_csv=from_db,
                                  out_features_csv="db://lab/taters/predicted", device="cpu",
                                  verbose=False)
    assert tables.parse_ref(db_out).is_db
    _same_rows(file_out, db_out)
    assert {"pred_score", "pred_topic", "prob_topic"} <= set(tables.columns_of(db_out))
    # the default destination beside a database source is a table too
    default = apply_text_predictor(model_json=predictor, analysis_csv=from_db, device="cpu",
                                   verbose=False)
    assert str(default) == "db://lab/taters/features/text_predictor_applied"


def test_the_predictor_reads_a_big_table_in_bounded_slices(predictor, tmp_path, monkeypatch):
    from taters.text import finetune_predictor as fp

    seen: list = []

    def fake_predict(model, heads, tokenizer, texts, tasks, **kw):
        seen.append(len(texts))
        n = len(texts)
        return ({"score": [1.0] * n, "topic": [[0.5, 0.5]] * n}, [1] * n, [1] * n)

    monkeypatch.setattr(fp, "_predict", fake_predict)
    out = fp.apply_text_predictor(model_json=predictor, csv_path=_big_texts(tmp_path / "big.csv", 5000),
                                  text_cols=["text"], id_cols=["pid"],
                                  out_features_csv=tmp_path / "s.csv", device="cpu",
                                  verbose=False)
    assert seen and max(seen) <= fp._TEXTS_PER_CALL < 5000 and sum(seen) == 5000
    assert tables.count(out) == 5000


def test_the_predictor_trains_from_a_database_table(both, encoder, tmp_path):
    """The trainer reads its labeled rows out of the database; the model
    and its report stay on disk, which is where models live."""
    from taters.text.finetune_predictor import finetune_text_predictor

    _csv, _table = both
    study = _study(tmp_path / "s.csv")
    gathered = csv_to_analysis_ready_csv(csv_path=study, out_csv="db://lab/taters/study",
                                         text_cols=["text"], id_cols=["pid"],
                                         carry_cols=["score", "topic"], verbose=False)
    out = finetune_text_predictor(analysis_csv=gathered, outcome_cols=["score", "topic"],
                                  out_dir=tmp_path / "stats", base_model=str(encoder),
                                  n_folds=2, epochs=1, max_length=32, batch_size=8,
                                  device="cpu", precision="fp32", verbose=False)
    assert out == tmp_path / "stats" / "text_predictor_cv_metrics.csv" and out.is_file()
    assert (tmp_path / "stats" / "models" / "text_predictor__score_topic.json").is_file()


# ---------------------------------------------------------------------------
# The encoder trainers
# ---------------------------------------------------------------------------

def _corpus(path: Path, n=40, seed=1) -> Path:
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            words = [rng.choice(FOOD + ["the", "and", "a", "was", "good", "."])
                     for _ in range(rng.randint(12, 30))]
            w.writerow([f"d{i}", " ".join(words)])
    return path


def test_an_encoder_is_adapted_from_a_database_table(both, encoder, tmp_path):
    from taters.text.adapt_encoder import adapt_encoder

    corpus = csv_to_analysis_ready_csv(csv_path=_corpus(tmp_path / "c.csv"),
                                       out_csv="db://lab/taters/corpus", text_cols=["text"],
                                       id_cols=["text_id"], verbose=False)
    out = adapt_encoder(analysis_csv=corpus, out_model_json=tmp_path / "models" / "adapted.json",
                        base_model=str(encoder), epochs=1, max_length=32, batch_size=8,
                        grad_accum=1, learning_rate=5e-4, device="cpu", precision="fp32",
                        verbose=False)
    assert out == tmp_path / "models" / "adapted.json" and out.is_file()


def test_an_encoder_is_pretrained_from_a_database_table(both, tmp_path):
    pytest.importorskip("tokenizers")
    from taters.text.pretrain_encoder import MIN_VOCAB, pretrain_encoder

    corpus = csv_to_analysis_ready_csv(csv_path=_corpus(tmp_path / "c.csv"),
                                       out_csv="db://lab/taters/corpus", text_cols=["text"],
                                       id_cols=["text_id"], verbose=False)
    out = pretrain_encoder(analysis_csv=corpus, out_model_json=tmp_path / "models" / "scratch.json",
                           preset="custom", layers=2, hidden_size=32, attention_heads=2,
                           vocab_size=MIN_VOCAB, max_length=32, epochs=1, patience=2,
                           batch_size=8, grad_accum=1, learning_rate=5e-4, device="cpu",
                           precision="fp32", verbose=False)
    assert out == tmp_path / "models" / "scratch.json" and out.is_file()
