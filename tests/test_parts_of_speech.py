"""
Part-of-speech features: the content coder's shape, tags as the categories.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from taters.helpers.nltk_data import ensure_pos_tagger, ensure_universal_tagset
from taters.text.analyze_parts_of_speech import analyze_parts_of_speech

if not ensure_pos_tagger(verbose=False):        # pragma: no cover
    pytest.skip("NLTK tagger data unavailable", allow_module_level=True)


def _corpus(tmp_path, docs):
    src = tmp_path / "corpus.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text_id", "text"])
        for i, t in enumerate(docs):
            w.writerow([f"d{i}", t])
    return src


def _pos(tmp_path, docs, **kwargs):
    out = analyze_parts_of_speech(
        csv_path=_corpus(tmp_path, docs), text_cols=["text"],
        id_cols=["text_id"], gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "pos.csv",
        overwrite_existing=True, **kwargs)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return out, {r["text_id"]: r for r in rows}


def test_relative_frequencies_are_the_default_and_sum_to_one(tmp_path):
    # "the cat sat" comes out as DT NN VBD, so three tags with one of each
    _, by_id = _pos(tmp_path, ["the cat sat"])
    row = by_id["d0"]
    assert row["token_count"] == "3"
    tags = {k: float(v) for k, v in row.items()
            if k.startswith("pos_")}
    assert tags == {"pos_DT": pytest.approx(1 / 3, abs=1e-4),
                    "pos_NN": pytest.approx(1 / 3, abs=1e-4),
                    "pos_VBD": pytest.approx(1 / 3, abs=1e-4)}
    assert sum(tags.values()) == pytest.approx(1.0, abs=1e-3)


def test_counts_mode_writes_integers(tmp_path):
    _, by_id = _pos(tmp_path, ["the cat sat on the mat"],
                    relative_freq=False)
    assert by_id["d0"]["pos_DT"] == "2"
    assert by_id["d0"]["pos_NN"] == "2"


def test_syntactic_ngrams_get_their_own_columns_per_order(tmp_path):
    """All orders 1..N, each normalized by its own window count -- an order-2
    column in a 3-token doc divides by 2, not 3."""
    _, by_id = _pos(tmp_path, ["the cat sat"], sngram_n=2)
    row = by_id["d0"]
    assert float(row["pos_DT_NN"]) == pytest.approx(1 / 2, abs=1e-4)
    assert float(row["pos_NN_VBD"]) == pytest.approx(1 / 2, abs=1e-4)
    assert float(row["pos_DT"]) == pytest.approx(1 / 3, abs=1e-4)


def test_a_document_missing_a_sequence_scores_zero_not_blank(tmp_path):
    _, by_id = _pos(tmp_path, ["the cat sat", "running quickly forward"])
    assert float(by_id["d1"]["pos_DT"]) == 0.0
    assert by_id["d1"]["pos_DT"] != ""


def test_the_universal_tagset_is_available(tmp_path):
    if not ensure_universal_tagset(verbose=False):
        pytest.skip("universal tagset data unavailable")
    _, by_id = _pos(tmp_path, ["the cat sat"], tagset="universal")
    row = by_id["d0"]
    assert float(row["pos_NOUN"]) == pytest.approx(1 / 3, abs=1e-4)
    assert "pos_NN" not in row


def test_an_unknown_tagset_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tagset"):
        _pos(tmp_path, ["hi there"], tagset="klingon")


def test_sngram_n_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sngram_n"):
        _pos(tmp_path, ["hi there"], sngram_n=0)


def test_tagged_text_is_off_by_default_and_word_tag_when_on(tmp_path):
    tagged = tmp_path / "tagged.csv"
    _pos(tmp_path, ["the cat sat"])
    assert not tagged.exists()

    _pos(tmp_path, ["the cat sat"], tagged_text_csv=tagged)
    rows = list(csv.DictReader(open(tagged, encoding="utf-8-sig")))
    assert rows[0]["tagged_text"] == "the_DT cat_NN sat_VBD"


def test_a_zero_token_document_writes_zeros_not_a_crash(tmp_path):
    """
    The CSV gatherer drops rows whose text is empty, but a hand-built
    analysis CSV (or text that tokenizes to nothing) can still deliver a
    zero-token document -- which must not divide by zero.
    """
    ready = tmp_path / "ready.csv"
    ready.write_text("text_id,text\nd0,\nd1,the cat sat\n", encoding="utf-8")
    out = analyze_parts_of_speech(
        analysis_csv=ready, out_features_csv=tmp_path / "pos.csv",
        overwrite_existing=True)
    with open(out, encoding="utf-8-sig", newline="") as f:
        by_id = {r["text_id"]: r for r in csv.DictReader(f)}
    assert by_id["d0"]["token_count"] == "0"
    assert by_id["d0"]["pos_DT"] == "0"


def test_a_passthrough_id_column_rides_along_once(tmp_path):
    """Two id columns compose text_id together, so each is a real part of
    the identity and rides along -- once. One id column IS text_id, and is
    not repeated beside it (the shared rule in resolve_passthrough_columns)."""
    src = tmp_path / "multi.csv"
    src.write_text("speaker,turn,text\nA,1,the cat sat\nB,2,a dog ran\n",
                   encoding="utf-8")
    out = analyze_parts_of_speech(
        csv_path=src, text_cols=["text"], id_cols=["speaker", "turn"],
        gathered_csv=tmp_path / "g.csv", out_features_csv=tmp_path / "p.csv",
        overwrite_existing=True)
    with open(out, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header[:4] == ["text_id", "speaker", "turn", "token_count"]
    assert header.count("speaker") == 1

    alone = analyze_parts_of_speech(
        csv_path=src, text_cols=["text"], id_cols=["speaker"],
        gathered_csv=tmp_path / "g1.csv", out_features_csv=tmp_path / "p1.csv",
        overwrite_existing=True)
    with open(alone, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header[:2] == ["text_id", "token_count"], header[:3]


def test_the_cli_runs_end_to_end(tmp_path, capsys):
    from taters.text import analyze_parts_of_speech as mod

    src = _corpus(tmp_path, ["the cat sat"])
    mod.main([
        "--csv-path", str(src), "--id-col", "text_id",
        "--gathered-csv", str(tmp_path / "g.csv"),
        "--out-features-csv", str(tmp_path / "pos.csv"),
        "--counts", "--overwrite-existing",
    ])
    assert "pos.csv" in capsys.readouterr().out
    rows = list(csv.DictReader(open(tmp_path / "pos.csv", encoding="utf-8-sig")))
    assert rows[0]["pos_NN"] == "1"


def test_the_facade_reaches_it(tmp_path):
    from taters.Taters import Taters

    out = Taters().analyze_parts_of_speech(
        csv_path=_corpus(tmp_path, ["the cat sat"]), text_cols=["text"],
        id_cols=["text_id"], gathered_csv=tmp_path / "g.csv",
        out_features_csv=tmp_path / "pos.csv", overwrite_existing=True)
    assert Path(out).is_file()


def test_the_pos_features_run_end_to_end_on_stanza(tmp_path):
    from taters.helpers.stanza_data import ensure_stanza

    if not ensure_stanza(verbose=False):
        pytest.skip("stanza or its 'en' model unavailable")

    _, by_id = _pos(tmp_path, ["the cats were running"],
                    engine="stanza", device="cpu", tagset="universal")
    row = by_id["d0"]
    assert float(row["pos_NOUN"]) == pytest.approx(1 / 4, abs=1e-4)
    assert float(row["pos_VERB"]) + float(row.get("pos_AUX", 0)) == \
        pytest.approx(2 / 4, abs=1e-4)
