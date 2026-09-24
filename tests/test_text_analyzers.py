"""Tests for the shared contract of the text analyzers.

The five text analyzers (dictionaries, archetypes, readability, lexical
richness, sentence embeddings) all accept input three ways and all write a wide
features CSV. That shared surface is tested here using the two analyzers that
need no optional dependencies or model downloads; the heavier ones get the same
treatment in the `slow` tests.

Testing the *shared* behavior once, in one place, is deliberate: it is the part
most likely to drift apart between modules.
"""

from pathlib import Path

import pytest

from taters import Taters
from taters.text.analyze_lexical_richness import analyze_lexical_richness

# textstat is an optional extra, so without it we skip the readability tests here
textstat = pytest.importorskip("textstat", reason="readability needs textstat")
from taters.text.analyze_readability import analyze_readability  # noqa: E402
from csvhelpers import read_rows  # noqa: E402

ANALYZERS = [analyze_readability, analyze_lexical_richness]
ANALYZER_IDS = ["readability", "lexical_richness"]


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
# input modes (every analyzer shares these)
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
# grouping and pass-through columns
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
# readability specifics
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
# lexical richness specifics
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


def test_an_ampersand_entity_is_handled_the_same_whatever_else_is_present():
    """
    The `&amp;` replacement sat inside the loop over the *other* entities,
    so it ran zero times when `&amp;` was the only one: "cats &amp; dogs"
    tokenized as &, amp, ; and "cats &amp; dogs &eacute;" as "and".
    """
    from taters.text.happierfuntokenizing import Tokenizer

    tok = Tokenizer()
    alone = tok.tokenize("cats &amp; dogs")
    beside = tok.tokenize("cats &amp; dogs &eacute;")
    assert "and" in alone and "amp" not in alone, alone
    assert [t for t in alone if t != "and"] == \
        [t for t in beside if t not in ("and", "é")]


@pytest.mark.parametrize("analyzer", [
    pytest.param("readability", id="readability"),
    pytest.param("lexical_richness", id="lexical_richness"),
    pytest.param("word_count", id="word_count"),
])
def test_a_semicolon_spreadsheet_goes_through_whole(analyzer, tmp_path):
    """
    The gatherer writes the analysis-ready table with commas whatever the
    source used. Three analyzers passed the *source* delimiter on to that
    table, so a ";" spreadsheet -- ordinary in half of Europe -- failed with
    "Expected columns 'text_id' and 'text' ... found ['text_id,id,text']" on
    the one step almost every study picks; parts of speech, which reads with
    the default, worked. No test ran a non-comma file end to end.
    """
    from taters.text.analyze_word_count import analyze_word_count

    fn = {"readability": analyze_readability,
          "lexical_richness": analyze_lexical_richness,
          "word_count": analyze_word_count}[analyzer]
    src = tmp_path / "semi.csv"
    src.write_text("pid;text\n"
                   "p1;The quick brown fox jumps over the lazy dog today.\n"
                   "p2;Colorless green ideas sleep furiously every night.\n",
                   encoding="utf-8")
    out = fn(csv_path=src, text_cols=["text"], id_cols=["pid"], delimiter=";",
             gathered_csv=tmp_path / "g.csv",
             out_features_csv=tmp_path / f"{analyzer}.csv")
    rows = read_rows(out)
    assert [r["text_id"] for r in rows] == ["p1", "p2"]
    measures = {k: v for k, v in rows[0].items() if k != "text_id"}
    # some of the richness measures want more than ten tokens and come back
    # blank on purpose. all we care about is that the row got scored at all
    assert any(v not in ("", None) for v in measures.values()), \
        "the measures came through blank"


# ---------------------------------------------------------------------------
# what rides along beside text_id
# ---------------------------------------------------------------------------

def test_the_column_text_id_was_composed_from_is_not_carried_twice():
    """
    With one id column the gatherer writes its value *as* text_id, so
    carrying the column too put two identical columns -- text_id and
    ResponseId -- into every feature table of every run (a real report).
    Two composing columns are each a real part of the identity and stay.
    """
    from taters.helpers.row_map import resolve_passthrough_columns as pt

    header = ["text_id", "ResponseId", "text"]
    assert pt(header, id_cols=["ResponseId"]) == []
    # ...even when we name it outright, since it IS text_id
    assert pt(header, pass_through_cols=["ResponseId"], id_cols=["ResponseId"]) == []
    two = ["text_id", "pid", "wave", "text"]
    assert pt(two, id_cols=["pid", "wave"]) == ["pid", "wave"]
    # a grouping column gets folded into text_id the same way
    assert pt(["text_id", "author", "text", "group_count"], group_by=["author"]) == []
    assert pt(["text_id", "author", "sub", "text", "group_count"],
              group_by=["author", "sub"]) == ["author", "sub"]


def test_the_gatherers_row_count_is_never_carried_as_a_feature():
    """`group_count` is how many rows were combined -- a fact about the
    gather, not a measurement -- and the fall-back that carried every spare
    column would have handed it to the statistics as a predictor."""
    from taters.helpers.row_map import resolve_passthrough_columns as pt

    header = ["text_id", "condition", "text", "group_count", "source_col"]
    assert pt(header) == ["condition", "source_col"]


def test_a_missing_pass_through_column_is_refused_by_name():
    from taters.helpers.row_map import resolve_passthrough_columns as pt

    with pytest.raises(ValueError, match="nope"):
        pt(["text_id", "text"], pass_through_cols=["nope"], analysis_ready="t.csv")


@pytest.mark.parametrize("analyzer", ANALYZERS, ids=ANALYZER_IDS)
def test_a_single_id_column_appears_once_in_the_features(analyzer, tmp_path):
    """The rule, applied: the feature table names the row by text_id and
    does not repeat the column that value came from."""
    src = tmp_path / "survey.csv"
    src.write_text("ResponseId,text\n"
                   "R1,The quick brown fox jumps over the lazy dog today.\n"
                   "R2,Colorless green ideas sleep furiously every night.\n",
                   encoding="utf-8")
    out = analyzer(csv_path=src, text_cols=["text"], id_cols=["ResponseId"],
                   gathered_csv=tmp_path / "g.csv",
                   out_features_csv=tmp_path / "f.csv")
    rows = read_rows(out)
    assert [r["text_id"] for r in rows] == ["R1", "R2"]
    assert "ResponseId" not in rows[0], "the identity column was carried twice"


def test_the_three_input_modes_are_resolved_in_one_place():
    """
    The accept-or-gather block -- exactly one of csv_path / txt_dir unless
    analysis_csv, announce, call one of two gatherers with a dozen forwarded
    settings -- was copied into eleven modules and had started to drift. A
    change to the gather contract is now one edit, and this pins that.
    """
    src = Path(__file__).resolve().parents[1] / "src" / "taters"
    sentence = "Provide exactly one of csv_path or txt_dir"
    owners = sorted(str(f.relative_to(src)) for f in src.rglob("*.py")
                    if sentence in f.read_text(encoding="utf-8"))
    assert owners == ["helpers/text_gather.py"], owners
    callers = [f for f in (src / "text").glob("*.py")
               if "resolve_analysis_ready(" in f.read_text(encoding="utf-8")]
    # eleven analyzers, plus word vectors, transformer embeddings, the
    # encoder adapter and the fine-tuned predictor
    # 16 with hf_classifier (an imported Hugging Face classifier reads text
    # through the same gather as everything else)
    # 17 with VADER sentiment
    # 18 with LDA topics (topic_model_mem was already counted -- `apply_mem_model`
    #    gathered text long before the fit did)
    # 19 with NMF topics
    # 20 with the topic-count sweep
    # 21 with pretraining an encoder from scratch
    # 22 with entropy
    assert len(callers) == 23, sorted(f.name for f in callers)


def test_every_readability_metric_still_exists_on_textstat():
    """
    `_score_text` looks each metric up with getattr and records None when it
    is not there, which is right for one metric textstat drops and disastrous
    for a version that moves them all. textstat 1.0 is exactly that: its
    public surface is three classes (Text, Sentence, Word) and not one of
    these functions. Nothing would raise -- every readability column would
    come back empty, for every row, and the run would report success.

    The version ceiling in pyproject holds us to 0.7.x, the series we test
    against. This is what notices if that ceiling is ever raised without
    porting the analyzer: it reads the metric list out of the module and
    asks textstat itself whether each one is still callable.
    """
    import textstat

    from taters.text.analyze_readability import METRICS

    # this used to dig the list out of the module with `ast`, because it lived
    # inside the function and `analyze_readability` is decorated -- its
    # __code__ belongs to the wrapper in provenance.py. the list is a module
    # constant now (the column registry needs to read it too), so we just
    # import it.
    listed = list(METRICS)
    assert len(listed) >= 15, f"the metric list moved or shrank: {listed}"

    missing = [m for m in listed if not callable(getattr(textstat, m, None))]
    assert not missing, (
        f"textstat {getattr(textstat, '__version__', '?')} no longer provides "
        f"{missing}; readability would silently write empty columns")
