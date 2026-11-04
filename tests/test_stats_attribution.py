"""
Token attribution: the exact share of every token in a prediction made over
mean-pooled embeddings.

The encoder is the tiny offline BERT; the claim under test is arithmetic,
not meaning -- that the shares of a text's positions, special tokens
included, add up to the prediction the ridge makes from the text's vector,
through every step the extraction pipeline takes (windows, layer choice,
sentence weighting, normalization, a PCA in front), and that the tables and
pictures are drawn from exactly those numbers.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from taters.helpers.feature_gather import feature_gather  # noqa: E402
from taters.stats import attribution as attr  # noqa: E402
from taters.stats import explain as ex  # noqa: E402
from taters.stats.assemble import assemble_analysis_table  # noqa: E402
from taters.stats.ridge import fit_ridge_csv  # noqa: E402
from taters.text.transformer_embeddings import extract_transformer_embeddings  # noqa: E402
from csvhelpers import _read  # noqa: E402
from tiny_encoder import WORDS, build  # noqa: E402

CONTENT = [w for w in WORDS if w.isalpha() and len(w) > 2 and not w.startswith("[")]


@pytest.fixture(scope="module")
def encoder(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("enc") / "tiny")


def _corpus(path: Path, n=40, seed=2, long_every=0) -> Path:
    """Short texts of the tiny vocabulary, two or three sentences each; every
    ``long_every``-th text gets one sentence longer than the window."""
    rng = random.Random(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text", "score"])
        for i in range(n):
            sents = []
            for _ in range(rng.randint(2, 3)):
                k = rng.randint(3, 7)
                if long_every and i % long_every == 0:
                    k = 24
                sents.append(" ".join(rng.choice(CONTENT) for _ in range(k)) + ".")
            text = " ".join(sents)
            score = text.count("gravy") + 0.5 * text.count("happy") + rng.gauss(0, 0.3)
            w.writerow([f"t{i}", text, f"{score:.4f}"])
    return path


def _pipeline(tmp_path: Path, encoder: Path, *, pooling="mean", weighting="equal",
              normalize=False, layers="second_to_last", pca="off", long_every=0,
              max_length=16, n=40):
    """The pipeline's own steps, in order: extract, merge, assemble, fit --
    so the sidecars and records are exactly what a run leaves behind."""
    src = _corpus(tmp_path / "study.csv", n=n, long_every=long_every)
    features = tmp_path / "features"
    raw = extract_transformer_embeddings(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "gathered" / "texts.csv",
        model_name_or_path=str(encoder), device="cpu", precision="fp32",
        layers=layers, pooling=pooling, sentence_weighting=weighting,
        normalize_l2=normalize, max_length=max_length,
        out_features_csv=features / "transformer_embeddings.csv", verbose=False)
    merged = feature_gather(root_dir=str(raw), aggregate=True, group_by=["text_id"],
                            per_file=False, stats=["mean"], add_source_path=False,
                            exclude_cols=["text", "token_count", "sentence_count"],
                            out_csv=features / "transformer_embeddings_aggregated.csv",
                            verbose=False)
    stats = tmp_path / "stats_results"
    assemble_analysis_table(feature_csvs=[str(merged)], metadata_csv=src,
                            key_cols=["text_id"], metadata_cols=["score"],
                            out_dir=stats, verbose=False)
    fit_ridge_csv(table_csv=stats / "analysis_table.csv", outcome_cols=["score"],
                  feature_sets="per_table", pca=pca, pca_components=3 if pca != "off" else 0,
                  out_dir=stats, rounding=10, verbose=False)
    return stats, tmp_path / "gathered" / "texts.csv"


def _explain(stats: Path, texts: Path, **kw):
    return ex.explain_predictions(stats, texts=texts, device="cpu", workers=1,
                                  verbose=False, min_docs=1, min_doc_pct=0.0, **kw)


def _summary(stats: Path) -> dict:
    (row,) = _read(stats / "ridge_attribution_summary.csv")
    return row


def test_the_merged_columns_point_back_at_the_raw_dimensions():
    cols = ["e_1", "e_2", "e_3"]
    assert attr.column_index("e_2__mean", "embeds", cols) == 1
    assert attr.column_index("embeds__e_3__mean", "embeds", cols) == 2
    assert attr.column_index("e_2", "embeds", cols) == 1
    assert attr.column_index("age", "embeds", cols) is None
    # a word-vector set: the dimension by its own name, and a cosine
    # column is not mistaken for a dimension called `sim_zoo`
    wv = ["wv_1", "wv_2", "token_count", "in_vocab_count"]
    assert attr.column_index("wv_2", "vectors", wv) == 1
    assert attr.column_index("vectors__wv_1", "vectors", wv) == 0
    assert attr.column_index("sim_zoo__pets", "vectors", wv) is None


def test_a_source_path_from_another_machine_is_found_again_by_name(tmp_path, encoder):
    """A sidecar written on Windows names C:\\... paths; read from WSL (or
    after the folder moved) they resolve to nothing, and the table is right
    there under features/. The resolver falls back to the file's name under
    the run folder, the way the theme loadings already do."""
    stats, _texts = _pipeline(tmp_path, encoder, n=20)
    sets = json.loads((stats / "analysis_table_sets.json").read_text(encoding="utf-8"))
    (name, real), = sets["sources"].items()
    sets["sources"][name] = "C:\\Users\\someone\\study\\features\\" + Path(real).name
    (stats / "analysis_table_sets.json").write_text(json.dumps(sets), encoding="utf-8")
    lost = attr.set_origin(name, sets["sources"])
    assert lost.problem and "no readable provenance record" in lost.problem
    found = attr.set_origin(name, sets["sources"], root=stats.parent)
    assert not found.problem and found.kind == "transformer" and found.raw_table is not None


def test_word_pieces_merge_into_the_word_a_person_reads():
    merged = attr.merge_pieces(["[CLS]", "the", "cat", "##s", "run", "##ning", "[SEP]"],
                               specials=["[CLS]", "[SEP]"])
    assert merged == [(attr.SPECIALS, [0]), ("the", [1]), ("cats", [2, 3]),
                      ("running", [4, 5]), (attr.SPECIALS, [6])]
    bpe = attr.merge_pieces(["<s>", "ĠThe", "Ġcat", "s", "Ġran", "</s>"],
                            specials=["<s>", "</s>"])
    assert bpe == [(attr.SPECIALS, [0]), ("the", [1]), ("cats", [2, 3]), ("ran", [4]),
                   (attr.SPECIALS, [5])]
    # byte-level BPE writes punctuation without the leading space too; it is
    # its own piece, not the tail of the word before it ("summer." was a word)
    punct = attr.merge_pieces(["<s>", "Ġfun", ",", "Ġsummer", ".", "Ġdon", "'t", "</s>"],
                              specials=["<s>", "</s>"])
    assert [w for w, _p in punct] == [attr.SPECIALS, "fun", ",", "summer", ".", "don't",
                                      attr.SPECIALS]
    # RoBERTa writes no space marker on a sentence's first word, so a
    # one-word sentence shows no marker at all; the scheme comes from the
    # tokenizer, not the sentence ("ag" and "reed" were words in a real run)
    first = ["<s>", "Ag", "reed", ".", "</s>"]
    assert [w for w, _p in attr.merge_pieces(first, specials=["<s>", "</s>"], scheme="bpe")] \
        == [attr.SPECIALS, "agreed", ".", attr.SPECIALS]
    assert [w for w, _p in attr.merge_pieces(first, specials=["<s>", "</s>"])] \
        == [attr.SPECIALS, "ag", "reed", ".", attr.SPECIALS], "the guess, without the tokenizer"


def test_the_piece_scheme_is_read_from_the_tokenizer():
    class Bpe:
        def get_vocab(self):
            return {"<s>": 0, "Ġthe": 1, "the": 2}

    class WordPiece:
        def get_vocab(self):
            return {"[CLS]": 0, "the": 1, "##s": 2}

    assert attr.piece_scheme(Bpe()) == "bpe"
    assert attr.piece_scheme(WordPiece()) == "wordpiece"


@pytest.mark.parametrize("weighting,normalize,layers,pca", [
    ("equal", False, "second_to_last", "off"),
    ("tokens", True, "last4_concat", "off"),
    ("equal", True, "second_to_last", "all"),
])
def test_the_shares_add_up_to_the_prediction_exactly(tmp_path, encoder, weighting,
                                                     normalize, layers, pca):
    """The decomposition: every position's share, special tokens included,
    summed over the text, equals the ridge's prediction from the text's
    vector rebuilt from the same token states -- through windows, the
    layer choice, token weighting, normalization and a PCA in front."""
    stats, texts = _pipeline(tmp_path, encoder, weighting=weighting, normalize=normalize,
                             layers=layers, pca=pca, long_every=5)
    _explain(stats, texts, top_texts=3)
    summary = _summary(stats)
    assert summary["exact"] == "True" and summary["kind"] == "transformer"
    assert float(summary["max_gap"]) < 1e-5, summary
    assert layers in summary["method"] and ("L2" in summary["method"]) == normalize
    assert summary["model"].endswith("tiny")
    rows = _read(stats / "ridge_attributed_texts.csv")
    by_text: dict = {}
    for r in rows:
        by_text.setdefault(r["text_id"], []).append(r)
    assert len(by_text) == 6, "three highest and three lowest"
    constant = float(summary["constant"])
    for tid, group in by_text.items():
        total = sum(float(r["contribution"]) for r in group)
        assert total == pytest.approx(float(group[0]["recomputed"]) - constant, abs=1e-5)
        # and the recomputed prediction is the kept model's own, up to the
        # six decimals the stored vectors were rounded to
        assert float(group[0]["recomputed"]) == pytest.approx(float(group[0]["prediction"]),
                                                               abs=5e-3)
        assert any(r["word"] == attr.SPECIALS for r in group), "the specials carry a share too"
        ranks = {r["rank"] for r in group}
        assert len(ranks) == 1 and ranks.pop() != ""
    # a long sentence was windowed: a text with window 2 exists
    assert any(int(r["window"]) > 1 for r in rows)


def test_the_word_table_sums_the_shares_per_word_and_the_clouds_draw_it(tmp_path, encoder):
    pytest.importorskip("PIL")
    from taters.figures import wordclouds as wc

    stats, texts = _pipeline(tmp_path, encoder)
    _explain(stats, texts, top_texts=2)
    words = _read(stats / "ridge_token_attributions.csv")
    assert list(words[0]) == ["subset", "feature_set", "outcome", "target", "word",
                              "mean_contribution", "total_contribution", "sd", "count",
                              "docs", "share_of_abs_total"]
    assert {r["target"] for r in words} == {"prediction"}
    assert attr.SPECIALS not in {r["word"] for r in words}
    by_word = {r["word"]: r for r in words}
    assert "gravy" in by_word and int(by_word["gravy"]["count"]) >= int(by_word["gravy"]["docs"])
    assert [abs(float(r["mean_contribution"])) for r in words] == \
        sorted((abs(float(r["mean_contribution"])) for r in words), reverse=True)
    manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
    assert {"ridge_word_correlates", "ridge_token_attributions", "ridge_attributed_texts",
            "ridge_attribution_summary"} <= set(manifest["tables"])
    assert manifest["sets"][0]["attributions"].startswith("attributed over 40 text(s)")
    section = (stats / "_sections" / "55-prediction-words.md").read_text(encoding="utf-8")
    assert "**Token attributions.**" in section and "attributed over 40" in section

    folder = wc.stats_wordclouds(stats, verbose=False)
    pngs = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.png"))
    set_name = words[0]["feature_set"]
    assert f"ridge-regression/{set_name}/words/score__attribution__positive.png" in pngs
    texts_pngs = [p for p in pngs if p.startswith(f"ridge-regression/{set_name}/texts/")]
    assert len(texts_pngs) == 4, texts_pngs
    section = (stats / "_sections" / "60-word-clouds.md").read_text(encoding="utf-8")
    assert "word by word" in section and "/texts/" in section


def test_cls_pooling_gets_correlates_only_and_says_why(tmp_path, encoder):
    stats, texts = _pipeline(tmp_path, encoder, pooling="cls")
    _explain(stats, texts)
    assert (stats / "ridge_word_correlates.csv").exists()
    assert not (stats / "ridge_token_attributions.csv").exists()
    manifest = json.loads((stats / "explain_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sets"][0]["attributions"].startswith("not attributed: the embeddings pool by cls")
    section = (stats / "_sections" / "55-prediction-words.md").read_text(encoding="utf-8")
    assert "pool by cls" in section


def test_correlates_only_can_be_asked_for(tmp_path, encoder):
    stats, texts = _pipeline(tmp_path, encoder)
    _explain(stats, texts, how="correlates")
    assert (stats / "ridge_word_correlates.csv").exists()
    assert not (stats / "ridge_token_attributions.csv").exists()


def test_a_sentence_transformers_model_is_attributed_through_its_own_step(tmp_path, encoder):
    """The other embedding step: a sentence-transformers model built from
    the same tiny encoder with mean pooling and a Normalize module, whose
    token embeddings come back from `encode(output_value=None)`."""
    st = pytest.importorskip("sentence_transformers")
    from sentence_transformers.base.modules.transformer import Transformer
    from sentence_transformers.sentence_transformer.modules import Normalize, Pooling

    from taters.text.extract_sentence_embeddings import extract_sentence_embeddings

    t = Transformer(str(encoder), max_seq_length=16)
    model = st.SentenceTransformer(
        modules=[t, Pooling(t.get_word_embedding_dimension(), pooling_mode="mean"), Normalize()],
        device="cpu")
    model_dir = tmp_path / "st_model"
    model.save(str(model_dir))

    src = _corpus(tmp_path / "study.csv", n=30)
    features = tmp_path / "features"
    raw = extract_sentence_embeddings(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "gathered" / "texts.csv", model_name=str(model_dir),
        device="cpu", normalize_l2=True,
        out_features_csv=features / "sentence_embeddings.csv", verbose=False)
    merged = feature_gather(root_dir=str(raw), aggregate=True, group_by=["text_id"],
                            per_file=False, stats=["mean"], add_source_path=False,
                            exclude_cols=["text"],
                            out_csv=features / "sentence_embeddings_aggregated.csv",
                            verbose=False)
    stats = tmp_path / "stats_results"
    assemble_analysis_table(feature_csvs=[str(merged)], metadata_csv=src,
                            key_cols=["text_id"], metadata_cols=["score"],
                            out_dir=stats, verbose=False)
    fit_ridge_csv(table_csv=stats / "analysis_table.csv", outcome_cols=["score"],
                  feature_sets="per_table", out_dir=stats, rounding=10, verbose=False)
    _explain(stats, tmp_path / "gathered" / "texts.csv", top_texts=2)
    summary = _summary(stats)
    assert summary["kind"] == "sentence" and summary["exact"] == "True"
    assert float(summary["max_gap"]) < 1e-5, summary
    rows = _read(stats / "ridge_attributed_texts.csv")
    by_text: dict = {}
    for r in rows:
        by_text.setdefault(r["text_id"], []).append(r)
    for group in by_text.values():
        total = sum(float(r["contribution"]) for r in group)
        assert total == pytest.approx(float(group[0]["recomputed"]) - float(summary["constant"]),
                                      abs=1e-5)
        assert float(group[0]["recomputed"]) == pytest.approx(float(group[0]["prediction"]),
                                                               abs=1e-4)
