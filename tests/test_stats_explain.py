"""
Explaining a prediction model in words: the word correlates of its held-out
predictions and residuals, as tables first and clouds second.

The corpus is planted: one word's frequency follows the outcome, another's
follows what a deliberately imperfect model misses. The tables have to find
exactly those, and the clouds have to be drawn from the tables.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from taters.stats import explain as ex  # noqa: E402
from taters.stats.ridge import fit_ridge_csv  # noqa: E402
from csvhelpers import _read  # noqa: E402

FILLER = ["potato", "fluffy", "kitchen", "dinner", "friend", "family", "home",
          "work", "time", "people", "think", "good", "water", "table", "window"]


def _planted(tmp_path: Path, n=300, seed=3, set_name="embeds", cols=("e0", "e1", "e2", "e3")):
    """
    ``outcome`` is noisy ``e0``; so a model on e0..e3 predicts it imperfectly.
    "gravy" appears in proportion to the outcome, "butter" in proportion to
    the part of the outcome the model cannot see (the noise), so the first
    tracks the predictions and the second the residuals.
    """
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    e = nrng.normal(size=(n, len(cols)))
    noise = nrng.normal(scale=0.8, size=n)
    y = e[:, 0] + noise
    stats = tmp_path / "stats"
    stats.mkdir()
    table = stats / "analysis_table.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "outcome", *cols])
        for i in range(n):
            w.writerow([f"t{i}", f"{y[i]:.6f}", *(f"{v:.6f}" for v in e[i])])
    (stats / "analysis_table_sets.json").write_text(json.dumps({
        "key": ["text_id"], "metadata": ["outcome"], "sources": {},
        "sets": {set_name: list(cols)}}), encoding="utf-8")
    texts = tmp_path / "texts.csv"
    with texts.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            words = [rng.choice(FILLER) for _ in range(30)]
            words += ["gravy"] * max(0, int(round(4 + 3 * y[i])))
            words += ["butter"] * max(0, int(round(4 + 4 * noise[i])))
            rng.shuffle(words)
            w.writerow([f"t{i}", " ".join(words)])
    return stats, table, texts


def _fit(stats: Path, table: Path):
    fit_ridge_csv(table_csv=table, outcome_cols=["outcome"], feature_sets="per_table",
                  out_dir=stats, verbose=False)


def test_the_targets_are_the_held_out_predictions_and_the_residuals(tmp_path):
    stats, table, _texts = _planted(tmp_path)
    _fit(stats, table)
    targets = ex.targets_in(stats, opaque={"embeds": True})
    kinds = {(t.feature_set, t.kind): t for t in targets}
    assert set(kinds) == {("embeds", "prediction"), ("embeds", "residual")}
    pred, resid = kinds[("embeds", "prediction")], kinds[("embeds", "residual")]
    assert len(pred.values) == 300 and pred.opaque
    rows = {r["text_id"]: r for r in _read(stats / "ridge_predictions__embeds.csv")}
    for tid, v in list(resid.values.items())[:5]:
        assert v == pytest.approx(float(rows[tid]["outcome"]) - float(rows[tid]["oof_pred_outcome"]))


def test_the_planted_words_top_the_prediction_and_residual_correlates(tmp_path):
    stats, table, texts = _planted(tmp_path)
    _fit(stats, table)
    ex.explain_predictions(stats, texts=texts, workers=1, verbose=False)
    rows = _read(stats / "ridge_word_correlates.csv")
    assert list(rows[0]) == ["subset", "feature_set", "outcome", "target", "ngram", "r",
                             "p", "p_adj", "n_texts", "docs", "mean_relfreq"]
    pred = [r for r in rows if r["target"] == "prediction"]
    resid = [r for r in rows if r["target"] == "residual"]
    assert pred[0]["ngram"] == "gravy" and float(pred[0]["r"]) > 0.3, pred[:3]
    assert float(pred[0]["p_adj"]) < 0.05
    assert resid[0]["ngram"] == "butter" and float(resid[0]["r"]) > 0.3, resid[:3]
    assert float(resid[0]["p_adj"]) < 0.05
    # the fillers share the texts with the planted words, so their relative
    # frequency dips wherever "gravy" rises: a small negative r is real, not
    # a bug. what must hold is that nothing planted nowhere comes close
    fillers = [r for r in pred if r["ngram"] in FILLER]
    assert fillers and max(abs(float(r["r"])) for r in fillers) < 0.5 * float(pred[0]["r"])
    # the rows are sorted by |r| within a family, and every n-gram is in
    # enough texts to be worth a correlation
    assert [abs(float(r["r"])) for r in pred] == sorted((abs(float(r["r"])) for r in pred),
                                                        reverse=True)
    assert all(int(r["docs"]) >= 10 for r in rows)
    assert all(int(r["n_texts"]) == 300 for r in rows)
    manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tables"] == {"ridge_word_correlates": str(stats / "ridge_word_correlates.csv")}
    assert manifest["sets"][0]["explained"] is True and manifest["sets"][0]["opaque"] is True
    section = (stats / "_sections" / "55-prediction-words.md").read_text(encoding="utf-8")
    assert section.startswith("## What the predictions track in words")
    assert "*gravy*" in section and "*butter*" in section
    assert "ridge_word_correlates" in section


def test_a_set_that_is_words_already_is_left_alone_under_auto_and_done_under_all(tmp_path):
    stats, table, texts = _planted(tmp_path, set_name="dictionary",
                                   cols=("anger", "joy", "sadness", "fear"))
    _fit(stats, table)
    ex.explain_predictions(stats, texts=texts, workers=1, verbose=False)
    assert not (stats / "ridge_word_correlates.csv").exists()
    section = (stats / "_sections" / "55-prediction-words.md").read_text(encoding="utf-8")
    assert "words already" in section and "explain_predictions: all" in section
    manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sets"][0]["explained"] is False
    ex.explain_predictions(stats, texts=texts, which="all", workers=1, verbose=False)
    rows = _read(stats / "ridge_word_correlates.csv")
    assert rows and rows[0]["feature_set"] == "dictionary"


def test_off_writes_nothing_and_a_second_run_rewrites_nothing(tmp_path):
    stats, table, texts = _planted(tmp_path)
    _fit(stats, table)
    ex.explain_predictions(stats, texts=texts, which="off", verbose=False)
    assert not (stats / "explain_manifest.json").exists()
    ex.explain_predictions(stats, texts=texts, workers=1, verbose=False)
    before = {p: p.stat().st_mtime_ns for p in stats.rglob("*") if p.is_file()}
    ex.explain_predictions(stats, texts=texts, workers=1, verbose=False)
    after = {p: p.stat().st_mtime_ns for p in stats.rglob("*") if p.is_file()}
    assert before == after


def test_the_clouds_are_drawn_from_the_tables_and_the_opaque_coefficients_are_not(tmp_path):
    pytest.importorskip("PIL")
    from taters.figures import wordclouds as wc

    stats, table, texts = _planted(tmp_path)
    _fit(stats, table)
    ex.explain_predictions(stats, texts=texts, workers=1, verbose=False)
    folder = wc.stats_wordclouds(stats, verbose=False)
    pngs = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
    assert "ridge-regression/embeds/words/outcome__prediction__positive.png" in pngs
    assert "ridge-regression/embeds/words/outcome__residual__positive.png" in pngs
    assert "ridge-regression/embeds/outcome__positive.png" not in pngs, \
        "a cloud of e0..e3 says nothing, so auto leaves it out"
    section = (stats / "_sections" / "60-word-clouds.md").read_text(encoding="utf-8")
    assert "What its predictions track in words" in section
    assert "features are not words" in section and "wordcloud_kinds: all" in section
    # and asked for, every kind is drawn
    folder = wc.stats_wordclouds(stats, kinds="all", verbose=False)
    pngs = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
    assert "ridge-regression/embeds/outcome__positive.png" in pngs
    assert "ridge-regression/embeds/words/outcome__prediction__positive.png" in pngs


def test_a_target_that_covers_only_some_texts_is_correlated_over_those_alone(tmp_path):
    """The bug this guards: a combined feature set has held-out predictions
    for the rows every table covers, so a target can hold half the texts.
    Summing a word's frequency over every text while summing its product
    with y over half gave correlations of exactly ±1. Every sum is per
    target now, and the r agrees with scipy over exactly those texts."""
    from scipy.stats import pearsonr

    from taters.text.ngram_prep import make_token_stream

    stats, _table, texts = _planted(tmp_path)
    rows = _read(texts)
    y = {r["text_id"]: float(r["text_id"][1:]) % 7 + r["text"].count("gravy") * 0.3
         for r in rows if int(r["text_id"][1:]) % 2 == 0}
    target = ex.Target(analysis="ridge", feature_set="s", subset="", outcome="o",
                       kind="prediction", values=y, model=tmp_path / "none.json")
    out, counts = ex.word_correlates(texts, [target], workers=1, min_docs=1, min_doc_pct=0.0)
    assert counts["texts"] == 150 and counts["skipped"] == 150
    by_gram = {r["ngram"]: r for r in out}
    assert all(r["n_texts"] == 150 and r["docs"] <= 150 for r in out)
    assert all(abs(r["r"]) < 1.0 for r in out), "no coin-toss ±1 correlations"
    stream = make_token_stream()
    xs, ys = [], []
    for r in rows:
        if r["text_id"] not in y:
            continue
        toks = stream(r["text"])
        xs.append(toks.count("gravy") / len(toks))
        ys.append(y[r["text_id"]])
    r_scipy, _p = pearsonr(xs, ys)
    assert by_gram["gravy"]["r"] == pytest.approx(r_scipy, abs=1e-5)


def test_the_pass_runs_through_a_real_worker_pool(tmp_path):
    """The user's first real run died here: the pool's function was a
    closure, which cannot be pickled to a worker, and the tests had only
    ever run inline. Two workers, the same answer as one."""
    stats, _table, texts = _planted(tmp_path)
    rows = _read(texts)
    y = {r["text_id"]: float(r["text"].count("gravy")) for r in rows}
    target = ex.Target(analysis="ridge", feature_set="s", subset="", outcome="o",
                       kind="prediction", values=y, model=tmp_path / "none.json")
    one, _ = ex.word_correlates(texts, [target], workers=1, min_docs=5, min_doc_pct=0.0)
    two, counts = ex.word_correlates(texts, [target], workers=2, min_docs=5, min_doc_pct=0.0)
    assert counts["texts"] == 300
    assert [(r["ngram"], r["r"]) for r in two] == [(r["ngram"], r["r"]) for r in one]


def test_a_target_with_too_few_texts_gets_no_correlates_and_is_named(tmp_path):
    stats, table, texts = _planted(tmp_path)
    rows = _read(texts)
    few = {r["text_id"]: 1.0 * i for i, r in enumerate(rows[:6])}
    target = ex.Target(analysis="ridge", feature_set="s", subset="", outcome="o",
                       kind="prediction", values=few, model=tmp_path / "none.json")
    out, counts = ex.word_correlates(texts, [target], workers=1, min_docs=1, min_doc_pct=0.0)
    assert out == [] and counts["thin"] == [("o (prediction)", 6)]


def test_the_pass_holds_the_vocabulary_not_the_texts(tmp_path):
    """Memory is one record per distinct n-gram: ten times the texts over the
    same words must not cost ten times the memory."""
    import tracemalloc

    def corpus(n: int) -> Path:
        rng = random.Random(1)
        path = tmp_path / f"texts_{n}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["text_id", "text"])
            for i in range(n):
                w.writerow([f"t{i}", " ".join(rng.choice(FILLER) for _ in range(20))])
        return path

    def peak(n: int) -> int:
        target = ex.Target(analysis="ridge", feature_set="s", subset="", outcome="o",
                           kind="prediction", values={f"t{i}": float(i % 7) for i in range(n)},
                           model=tmp_path / "none.json")
        tracemalloc.start()
        ex.word_correlates(corpus(n), [target], workers=1, min_docs=1, min_doc_pct=0.0)
        _, high = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return high

    small, big = peak(400), peak(4000)
    assert big < 3 * small, (small, big)


def test_a_database_run_puts_the_tables_in_the_database(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    from taters.figures import wordclouds as wc
    from taters.helpers import db, tables

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    tables.close_sessions()
    db._SECRETS.clear()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(tmp_path / "lab.sqlite")))
    stats, table, texts = _planted(tmp_path)
    _fit(stats, table)
    gathered = "db://lab/main/taters/gathered/texts"
    with tables.write_table(gathered, ["text_id", "text"]) as w:
        for r in _read(texts):
            w.writerow([r["text_id"], r["text"]])
    try:
        ex.explain_predictions(stats, texts=gathered, out_tables="db://lab/main/taters/stats",
                               workers=1, verbose=False)
        manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
        address = manifest["tables"]["ridge_word_correlates"]
        assert address == "db://lab/main/taters/stats/ridge_word_correlates"
        assert tables.exists(address)
        assert not (stats / "ridge_word_correlates.csv").exists()
        rows = list(tables.read_rows(address))
        assert rows[0]["ngram"] == "gravy"
        folder = wc.stats_wordclouds(stats, verbose=False)
        pngs = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
        assert "ridge-regression/embeds/words/outcome__prediction__positive.png" in pngs
    finally:
        tables.close_sessions()


def test_a_manifest_written_on_windows_is_read_with_its_slashes_made_the_same(tmp_path):
    """The user's run folder is written on Windows and read from WSL: the
    manifest's `stats_results\\ridge_word_correlates.csv` is one long file
    name on this side, and every explanation cloud was silently skipped."""
    stats = tmp_path / "stats_results"
    stats.mkdir()
    (stats / "explain_manifest.json").write_text(json.dumps({
        "tables": {"ridge_word_correlates": "stats_results\\ridge_word_correlates.csv",
                   "db_one": "db://lab/main/taters/stats/x"}}), encoding="utf-8")
    doc = ex.read_manifest(stats)
    assert doc["tables"]["ridge_word_correlates"] == "stats_results/ridge_word_correlates.csv"
    assert doc["tables"]["db_one"] == "db://lab/main/taters/stats/x"
