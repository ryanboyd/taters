"""
Token attribution over word vectors: the same contract as the encoders,
with no context, no windows and no model to download.

A text's word-vector row is a weighted mean of its words' vectors, so a
ridge over it decomposes, exactly, into one share per word occurrence. The
model is the four-word, two-dimensional matrix `test_word_vectors` writes
by hand, so every share can be checked against arithmetic, and the steps
are the pipeline's own (apply, assemble, fit), so the records the
attribution walks are exactly what a run leaves behind.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from taters.stats import attribution as attr  # noqa: E402
from taters.stats import explain as ex  # noqa: E402
from taters.stats._common import opaque_sets  # noqa: E402
from taters.stats.assemble import assemble_analysis_table  # noqa: E402
from taters.stats.classify import fit_classifier_csv  # noqa: E402
from taters.stats.ridge import fit_ridge_csv  # noqa: E402
from taters.text import word_vectors as wv  # noqa: E402
from taters.text.word_vectors import apply_word_vectors  # noqa: E402
from csvhelpers import _read  # noqa: E402
from test_word_vectors import VECTORS, VOCAB, _model, _zoo  # noqa: E402


def _corpus(path: Path, n=40, seed=3) -> Path:
    """Texts of the model's four words plus words it does not know; the
    score follows cat and car with a little noise, and a class follows
    the score's sign."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text", "score", "side"])
        for i in range(n):
            k = rng.randint(3, 8)
            words = [rng.choice(VOCAB + ["zebra", "lamp"]) for _ in range(k)]
            if i % 7 == 0:
                words += ["dog", "dog"]
            text = " ".join(words)
            score = text.count("cat") - text.count("car") + rng.gauss(0, 0.3)
            w.writerow([f"t{i}", text, f"{score:.4f}", "high" if score > 0 else "low"])
    return path


def _pipeline(tmp_path: Path, *, weighting="tokens", normalize_words=False, pca="off",
              concept_dicts=(), bookkeeping="aside", analysis="ridge", n=40):
    src = _corpus(tmp_path / "study.csv", n=n)
    model = _model(tmp_path / "m", weighting=weighting, concept_dicts=concept_dicts)
    features = tmp_path / "features"
    out = apply_word_vectors(
        model_json=model, csv_path=src, text_cols=["text"], id_cols=["text_id"],
        out_features_csv=features / "word_vectors.csv", normalize_words=normalize_words,
        rounding=10, verbose=False)
    stats = tmp_path / "stats_results"
    assemble_analysis_table(feature_csvs=[str(out)], metadata_csv=src, key_cols=["text_id"],
                            metadata_cols=["score", "side"], bookkeeping=bookkeeping,
                            out_dir=stats, verbose=False)
    if analysis == "ridge":
        fit_ridge_csv(table_csv=stats / "analysis_table.csv", outcome_cols=["score"],
                      feature_sets="per_table", pca=pca,
                      pca_components=2 if pca != "off" else 0,
                      out_dir=stats, rounding=10, verbose=False)
    else:
        fit_classifier_csv(table_csv=stats / "analysis_table.csv", outcome_cols=["side"],
                           feature_sets="per_table", out_dir=stats, verbose=False)
    return stats, src, model


def _explain(stats: Path, texts: Path, **kw):
    return ex.explain_predictions(stats, texts=texts, device="cpu", workers=1,
                                  verbose=False, min_docs=1, min_doc_pct=0.0, **kw)


def _by_text(rows):
    out: dict = {}
    for r in rows:
        out.setdefault(r["text_id"], []).append(r)
    return out


def test_a_word_vector_set_is_opaque_and_auto_explains_it(tmp_path):
    """`wv_37` says nothing to a reader: the set counts as opaque by its
    record's call (the step's function name, not the recipe id, which is
    what the rule matched before and never found)."""
    stats, texts, _model = _pipeline(tmp_path)
    (name, is_opaque), = opaque_sets(stats).items()
    assert is_opaque, name
    _explain(stats, texts)
    assert (stats / "ridge_token_attributions.csv").exists()


@pytest.mark.parametrize("weighting,normalize_words,pca", [
    ("tokens", False, "off"),
    ("types", False, "off"),
    ("sif", True, "off"),
    ("tokens", True, "all"),
])
def test_the_shares_add_up_to_the_prediction_exactly(tmp_path, weighting, normalize_words, pca):
    stats, texts, _m = _pipeline(tmp_path, weighting=weighting,
                                 normalize_words=normalize_words, pca=pca)
    _explain(stats, texts, top_texts=3)
    (summary,) = _read(stats / "ridge_attribution_summary.csv")
    assert summary["kind"] == "word_vectors" and summary["exact"] == "True"
    assert summary["model"] == "wv" and weighting in summary["method"]
    assert ("normalized" in summary["method"]) == normalize_words
    assert float(summary["max_gap"]) < 1e-9, summary
    rows = _read(stats / "ridge_attributed_texts.csv")
    by_text = _by_text(rows)
    assert len(by_text) == 6
    constant = float(summary["constant"])
    for group in by_text.values():
        total = sum(float(r["contribution"]) for r in group)
        assert total == pytest.approx(float(group[0]["recomputed"]) - constant, abs=1e-6)
        assert float(group[0]["recomputed"]) == pytest.approx(float(group[0]["prediction"]),
                                                               abs=1e-6)
        # no sentences or windows here: a unit is a token, counted from 1
        assert {r["sentence"] for r in group} == {""} and {r["window"] for r in group} == {""}
        assert [int(r["position"]) for r in group] == list(range(1, len(group) + 1))
        assert attr.SPECIALS not in {r["word"] for r in group}
    # a word the model does not know is in the text with no share
    unknown = [r for r in rows if r["word"] in ("zebra", "lamp")]
    assert unknown and all(float(r["contribution"]) == 0.0 for r in unknown)
    words = {r["word"]: r for r in _read(stats / "ridge_token_attributions.csv")}
    assert "cat" in words and "car" in words
    assert float(words["cat"]["mean_contribution"]) > 0 > float(words["car"]["mean_contribution"])


def test_under_types_weighting_a_repeated_word_shares_one_share_equally(tmp_path):
    """The step counts each distinct word once; the attribution gives each
    occurrence an equal part of that one share, so the repeats are shaded
    alike and the text still adds up."""
    stats, texts, _m = _pipeline(tmp_path, weighting="types")
    _explain(stats, texts, top_texts=40)
    rows = _read(stats / "ridge_attributed_texts.csv")
    for group in _by_text(rows).values():
        dogs = [float(r["contribution"]) for r in group if r["word"] == "dog"]
        if len(dogs) > 1:
            assert all(d == pytest.approx(dogs[0], abs=1e-9) for d in dogs)
            break
    else:
        pytest.fail("no attributed text repeated a word")


def test_the_directions_table_ranks_the_whole_vocabulary(tmp_path):
    """Every word the model knows, by the push a text of that one word
    would give; a word's share in a text is its direction over the text's
    weight sum, so the two tables agree."""
    stats, texts, _m = _pipeline(tmp_path)
    _explain(stats, texts, top_texts=2)
    directions = _read(stats / "ridge_word_directions.csv")
    assert list(directions[0]) == ["subset", "feature_set", "outcome", "target", "word",
                                   "direction", "count", "rank"]
    assert sorted(r["word"] for r in directions) == sorted(VOCAB)
    assert [int(r["rank"]) for r in directions] == [1, 2, 3, 4]
    by_word = {r["word"]: float(r["direction"]) for r in directions}
    rows = _read(stats / "ridge_attributed_texts.csv")
    for group in _by_text(rows).values():
        known = [r for r in group if r["word"] in by_word]
        for r in known:
            assert float(r["contribution"]) * len(known) == pytest.approx(by_word[r["word"]],
                                                                          abs=1e-6)
    manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
    assert "ridge_word_directions" in manifest["tables"]


def test_a_cosine_column_keeps_its_term_in_the_features_row(tmp_path):
    """`sim_zoo__pets` is a cosine, not a sum over the words. Its term is
    read from the analysis table into one `[features]` row per text, so
    the text still adds up and the summary says which columns were folded."""
    stats, texts, _m = _pipeline(tmp_path, concept_dicts=[_zoo(tmp_path)])
    _explain(stats, texts, top_texts=2)
    (summary,) = _read(stats / "ridge_attribution_summary.csv")
    assert "sim_zoo__pets" in summary["note"] and attr.FEATURES in summary["note"]
    rows = _read(stats / "ridge_attributed_texts.csv")
    constant = float(summary["constant"])
    for group in _by_text(rows).values():
        features = [r for r in group if r["word"] == attr.FEATURES]
        assert len(features) == 1 and features[0]["sentence"] == ""
        total = sum(float(r["contribution"]) for r in group)
        assert total == pytest.approx(float(group[0]["recomputed"]) - constant, abs=1e-6)
        assert float(group[0]["recomputed"]) == pytest.approx(float(group[0]["prediction"]),
                                                               abs=1e-6)
    words = {r["word"] for r in _read(stats / "ridge_token_attributions.csv")}
    assert attr.FEATURES not in words


def test_bookkeeping_columns_in_the_set_are_sums_over_the_tokens_too(tmp_path):
    """With `token_count` and `in_vocab_count` kept as features, every
    token adds one to the first and the known ones one to the second, so
    a word the model does not know can carry a share."""
    stats, texts, _m = _pipeline(tmp_path, bookkeeping="features")
    sets = json.loads((stats / "analysis_table_sets.json").read_text(encoding="utf-8"))
    (cols,) = sets["sets"].values()
    assert "token_count" in cols
    _explain(stats, texts, top_texts=3)
    (summary,) = _read(stats / "ridge_attribution_summary.csv")
    assert float(summary["max_gap"]) < 1e-9 and attr.FEATURES not in summary["note"]
    rows = _read(stats / "ridge_attributed_texts.csv")
    constant = float(summary["constant"])
    for group in _by_text(rows).values():
        total = sum(float(r["contribution"]) for r in group)
        assert total == pytest.approx(float(group[0]["recomputed"]) - constant, abs=1e-6)
    unknown = [float(r["contribution"]) for r in rows if r["word"] == "zebra"]
    assert unknown and any(c != 0.0 for c in unknown)


def test_a_classifier_is_attributed_on_its_logit(tmp_path):
    stats, texts, _m = _pipeline(tmp_path, analysis="classifier", n=60)
    _explain(stats, texts, top_texts=2)
    summary = _read(stats / "classifier_attribution_summary.csv")
    assert {r["target"] for r in summary} == {"logit_high", "logit_low"}
    assert all(r["kind"] == "word_vectors" and float(r["max_gap"]) < 1e-9 for r in summary)
    directions = _read(stats / "classifier_word_directions.csv")
    high = {r["word"]: float(r["direction"]) for r in directions if r["target"] == "logit_high"}
    low = {r["word"]: float(r["direction"]) for r in directions if r["target"] == "logit_low"}
    # two classes, one fit: the other side is its negation
    for word in VOCAB:
        assert high[word] == pytest.approx(-low[word], abs=1e-6)


def test_the_model_is_found_again_by_name_and_digest(tmp_path):
    """A run folder moved whole: the record carries the manifest's name and
    digest, not its path, and the file is found under the run root."""
    stats, texts, model = _pipeline(tmp_path)
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for f in model.parent.iterdir():
        f.rename(moved / f.name)
    _explain(stats, texts, top_texts=1)
    (summary,) = _read(stats / "ridge_attribution_summary.csv")
    assert summary["exact"] == "True"


def test_a_missing_matrix_gets_correlates_only_and_says_why(tmp_path):
    stats, texts, model = _pipeline(tmp_path)
    (model.parent / "wv.npy").unlink()
    model.unlink()
    _explain(stats, texts)
    assert (stats / "ridge_word_correlates.csv").exists()
    assert not (stats / "ridge_token_attributions.csv").exists()
    manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
    status = manifest["sets"][0]["attributions"]
    assert status.startswith("not attributed: the word-vector model could not be opened"), status
    # the carried manifest was written under the run, where a matrix put
    # beside it would be found on the next run
    assert (tmp_path / "assets" / "model_json" / "wv.json").is_file()


def test_the_unit_vectors_are_the_text_vector_in_parts(tmp_path):
    model = wv._load_model(_model(tmp_path / "m"))
    tokens = ["cat", "zebra", "dog", "dog", "car"]
    for weighting in ("tokens", "types", "sif"):
        for normalize in (False, True):
            units = wv.unit_vectors(model, tokens, weighting=weighting, normalize_words=normalize)
            vector, in_vocab = wv._text_vector(model, tokens, weighting, normalize)
            assert units.shape == (5, 2) and in_vocab == 4
            assert np.allclose(units.sum(axis=0), vector)
            assert not units[1].any(), "an unknown word adds nothing"
    weights = wv._token_weights(model, tokens, "types")
    assert weights.tolist() == [1.0, 0.0, 0.5, 0.5, 1.0]
    sif = wv._token_weights(model, tokens, "sif")
    assert sif[0] > sif[4], "car is the commonest word, so it weighs least"
    assert np.allclose(VECTORS[0], wv.unit_vectors(model, ["cat"], weighting="tokens",
                                                   normalize_words=False)[0])
