"""Tests for the shared contract of the text analyzers.

The five text analyzers (dictionaries, archetypes, readability, lexical
richness, sentence embeddings) all accept input three ways and all write a wide
features CSV. That shared surface is tested here using the two analyzers that
need no optional dependencies or model downloads; the heavier ones get the same
treatment in the `slow` tests.

Testing the *shared* behavior once, in one place, is deliberate: it is the part
most likely to drift apart between modules.
"""

import csv
from pathlib import Path

import pytest

from taters import Taters
from taters.text.analyze_lexical_richness import analyze_lexical_richness

# textstat is an optional extra; skip this module's readability tests without it.
textstat = pytest.importorskip("textstat", reason="readability needs textstat")
from taters.text.analyze_readability import analyze_readability  # noqa: E402

ANALYZERS = [analyze_readability, analyze_lexical_richness]
ANALYZER_IDS = ["readability", "lexical_richness"]


def read_rows(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def transcript(tmp_path) -> Path:
    p = tmp_path / "transcript.csv"
    p.write_text(
        "source,speaker,text\n"
        "s1,alice,The quick brown fox jumps over the lazy dog every single morning.\n"
        "s1,bob,Colorless green ideas sleep furiously and then wake up again.\n"
        "s1,alice,Another sentence from alice with several more words in it.\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Input modes — shared by every analyzer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_accepts_a_prebuilt_analysis_ready_csv(analyzer, analysis_ready_csv, tmp_path):
    out = analyzer(analysis_csv=analysis_ready_csv, out_features_csv=tmp_path / "f.csv")
    rows = read_rows(out)
    assert len(rows) == 2
    assert [r["text_id"] for r in rows] == ["a", "b"]


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_accepts_a_raw_csv(analyzer, transcript, tmp_path):
    out = analyzer(
        csv_path=transcript, text_cols=["text"], out_features_csv=tmp_path / "f.csv"
    )
    assert len(read_rows(out)) == 3


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_accepts_a_folder_of_txt_files(analyzer, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.txt").write_text("A document with a reasonable number of words.", encoding="utf-8")
    (corpus / "two.txt").write_text("Another document, also with words in it.", encoding="utf-8")

    out = analyzer(txt_dir=corpus, out_features_csv=tmp_path / "f.csv")
    assert {r["text_id"] for r in read_rows(out)} == {"one", "two"}


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_requires_exactly_one_input_mode(analyzer, transcript, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        analyzer(out_features_csv=tmp_path / "f.csv")           # neither
    with pytest.raises(ValueError, match="exactly one"):
        analyzer(                                               # both
            csv_path=transcript, txt_dir=tmp_path, out_features_csv=tmp_path / "f.csv"
        )


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_missing_analysis_csv_raises(analyzer, tmp_path):
    with pytest.raises(FileNotFoundError):
        analyzer(analysis_csv=tmp_path / "nope.csv", out_features_csv=tmp_path / "f.csv")


# ---------------------------------------------------------------------------
# Grouping and pass-through columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_group_by_collapses_rows(analyzer, transcript, tmp_path):
    out = analyzer(
        csv_path=transcript,
        text_cols=["text"],
        id_cols=["source", "speaker"],
        group_by=["source", "speaker"],
        out_features_csv=tmp_path / "f.csv",
    )
    rows = read_rows(out)
    assert len(rows) == 2                              # alice's two turns merged
    assert {r["speaker"] for r in rows} == {"alice", "bob"}


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_id_columns_are_carried_into_the_features_file(analyzer, transcript, tmp_path):
    out = analyzer(
        csv_path=transcript,
        text_cols=["text"],
        id_cols=["source", "speaker"],
        group_by=["source", "speaker"],
        out_features_csv=tmp_path / "f.csv",
    )
    header = list(read_rows(out)[0])
    assert header[0] == "text_id"
    assert "source" in header and "speaker" in header


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_default_output_path_lands_under_features(analyzer, transcript, sandbox):
    out = Path(analyzer(csv_path=transcript, text_cols=["text"]))
    assert out.parent.parent == sandbox / "features"
    assert out.is_file()


# ---------------------------------------------------------------------------
# Readability specifics
# ---------------------------------------------------------------------------

def test_readability_emits_the_documented_metrics(analysis_ready_csv, tmp_path):
    out = analyze_readability(
        analysis_csv=analysis_ready_csv, out_features_csv=tmp_path / "f.csv"
    )
    header = set(read_rows(out)[0])
    assert {
        "flesch_reading_ease", "flesch_kincaid_grade", "smog_index",
        "gunning_fog", "syllable_count", "lexicon_count", "sentence_count",
    } <= header


def test_readability_scores_are_numeric_and_ordered_sensibly(tmp_path):
    """Simple prose should read as easier than dense academic prose."""
    src = tmp_path / "texts.csv"
    src.write_text(
        "text_id,text\n"
        "easy,The cat sat on the mat. The dog ran. It was fun.\n"
        "hard,\"Notwithstanding the epistemological ramifications, the "
        "phenomenological interpretation remains fundamentally indeterminate.\"\n",
        encoding="utf-8",
    )
    out = analyze_readability(analysis_csv=src, out_features_csv=tmp_path / "f.csv")
    scores = {r["text_id"]: float(r["flesch_reading_ease"]) for r in read_rows(out)}
    assert scores["easy"] > scores["hard"]


def test_readability_survives_empty_text(tmp_path):
    src = tmp_path / "texts.csv"
    src.write_text("text_id,text\nempty,\n", encoding="utf-8")
    out = analyze_readability(analysis_csv=src, out_features_csv=tmp_path / "f.csv")
    assert len(read_rows(out)) == 1        # a row, not a crash


# ---------------------------------------------------------------------------
# Lexical richness specifics
# ---------------------------------------------------------------------------

def test_lexical_richness_emits_every_metric(analysis_ready_csv, tmp_path):
    out = analyze_lexical_richness(
        analysis_csv=analysis_ready_csv, out_features_csv=tmp_path / "f.csv"
    )
    header = set(read_rows(out)[0])
    assert {
        "ttr", "rttr", "cttr", "herdan_c", "summer_s", "dugast", "maas",
        "yule_k", "yule_i", "herdan_vm", "simpson_d",
    } <= header


def test_lexical_richness_values_are_plausible(tmp_path):
    src = tmp_path / "texts.csv"
    src.write_text(
        "text_id,text\n"
        "repetitive,the the the the the the the the\n"
        "varied,alpha bravo charlie delta echo foxtrot golf hotel\n",
        encoding="utf-8",
    )
    out = analyze_lexical_richness(analysis_csv=src, out_features_csv=tmp_path / "f.csv")
    ttrs = {r["text_id"]: float(r["ttr"]) for r in read_rows(out)}
    assert ttrs["varied"] == pytest.approx(1.0)
    assert ttrs["repetitive"] == pytest.approx(0.125)


def test_lexical_richness_leaves_undefined_metrics_blank(tmp_path):
    """Short text makes windowed metrics undefined; they must be empty, not 0."""
    src = tmp_path / "texts.csv"
    src.write_text("text_id,text\ntiny,just three words\n", encoding="utf-8")
    out = analyze_lexical_richness(analysis_csv=src, out_features_csv=tmp_path / "f.csv")
    row = read_rows(out)[0]
    mattr_col = next(c for c in row if c.startswith("mattr"))
    assert row[mattr_col] == ""


def test_lexical_richness_is_reachable_through_the_facade(transcript, tmp_path):
    out = Taters().text.analyze_lexical_richness(
        csv_path=transcript, text_cols=["text"], out_features_csv=tmp_path / "f.csv"
    )
    assert Path(out).is_file()
