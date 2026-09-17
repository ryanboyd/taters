"""
The n-gram frequency list, its shared preparation, and the doc-term matrix.

The correctness core here is NPMI. Three silent-result bugs in the C# plugin
these steps replace are each pinned by a test: per-order probability spaces
(under which NPMI is not bounded by 1), totals computed after filtering (under
which NPMI drifted with the user's filter settings), and n-grams at the very
end of a document never matching in the DTM scan.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from taters.text.analyze_ngram_frequencies import HEADER, analyze_ngram_frequencies
from taters.text.build_doc_term_matrix import build_doc_term_matrix
from taters.text.ngram_prep import (
    iter_ngrams,
    load_stoplist,
    make_token_stream,
    split_for_collocation,
    stoplisted,
    tags_of,
    words_of,
)


def _corpus(tmp_path, docs, name="corpus.csv"):
    src = tmp_path / name
    src.write_text("text_id,text\n" +
                   "".join(f"d{i},{t}\n" for i, t in enumerate(docs)),
                   encoding="utf-8")
    return src


def _freq_list(tmp_path, docs, **kwargs):
    src = _corpus(tmp_path, docs)
    defaults = dict(min_freq=1, min_obs_pct=0, min_token_count=1, ngram_n=2)
    defaults.update(kwargs)
    out = analyze_ngram_frequencies(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "freqs.csv",
        overwrite_existing=True, **defaults)
    with open(out, encoding="utf-8-sig", newline="") as f:
        return out, {r["ngram"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# NPMI, done correctly
# ---------------------------------------------------------------------------

def test_perfect_collocation_npmi_is_one(tmp_path):
    _, rows = _freq_list(tmp_path, ["snow leopard"] * 5)
    assert float(rows["snow leopard"]["npmi"]) == 1.0
    # and logDice sits at its own theoretical maximum
    assert float(rows["snow leopard"]["logdice"]) == 14.0


def test_npmi_matches_the_hand_computed_value(tmp_path):
    """Bouma (2009) over one sample space: p(g) = freq(g) / corpus tokens."""
    _, rows = _freq_list(tmp_path, ["the cat sat", "the dog sat", "a cat ran"])
    # N = 9 tokens; f("the cat") = 1, f("the") = 2, f("cat") = 2
    p_xy, p_x, p_y = 1 / 9, 2 / 9, 2 / 9
    expected = math.log(p_xy / (p_x * p_y)) / -math.log(p_xy)
    assert float(rows["the cat"]["npmi"]) == pytest.approx(expected, abs=1e-4)
    assert -1.0 <= float(rows["the cat"]["npmi"]) <= 1.0


def test_npmi_does_not_drift_with_the_frequency_filter(tmp_path):
    """
    The totals bug: the C# plugin computed its probability denominators from
    the rows that *survived* filtering, so tightening min_freq changed every
    surviving NPMI value. Statistics must come from the full counts; filters
    only choose which rows reach the file.
    """
    docs = ["snow leopard hunts", "snow leopard sleeps", "a fox sleeps",
            "a fox hunts", "a fox runs"]
    _, loose = _freq_list(tmp_path, docs, min_freq=1)
    _, tight = _freq_list(tmp_path, docs, min_freq=2)

    assert "snow leopard" in tight
    assert tight["snow leopard"]["npmi"] == loose["snow leopard"]["npmi"]
    # the tight run had better have filtered something, or this proves nothing
    assert len(tight) < len(loose)


def test_the_stoplist_drops_rows_without_touching_the_statistics(tmp_path):
    """
    Stoplists filter finished rows, never the token stream: removing "the"
    before counting would manufacture the bigram "cat sat" out of "the cat
    sat" and shift every probability under the surviving rows.
    """
    stop = tmp_path / "stop.txt"
    stop.write_text("the\n", encoding="utf-8")
    # we want "cat sat" to be an *imperfect* pair (each word also shows up on
    # its own). for a perfect pair NPMI is 1.0 under any token count, and a
    # mutant that shrank the sample space slipped right past us
    docs = ["the cat sat", "the cat sat", "a cat naps", "he sat down"]

    _, plain = _freq_list(tmp_path, docs)
    _, stopped = _freq_list(tmp_path, docs, stoplist_paths=[stop])

    assert "the" not in stopped and "the cat" not in stopped
    assert "cat sat" in stopped
    assert stopped["cat sat"]["npmi"] == plain["cat sat"]["npmi"]
    assert "the cat" in plain      # the plain run proves the row existed


def test_higher_orders_use_the_chain_split():
    assert split_for_collocation("new york city") == ("new york", "city")
    assert split_for_collocation("snow leopard") == ("snow", "leopard")


# ---------------------------------------------------------------------------
# the list itself
# ---------------------------------------------------------------------------

def test_all_orders_up_to_n_are_listed_and_ranked_by_frequency(tmp_path):
    out, rows = _freq_list(tmp_path, ["b a b a", "b c b c"])
    with open(out, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.reader(f))
    assert raw[0] == HEADER
    # b shows up 4 times so it has to be rank 1, and ranks run contiguously from 1
    assert raw[1][1] == "b" and raw[1][0] == "1"
    assert [r[0] for r in raw[1:]] == [str(i) for i in range(1, len(raw))]
    orders = {int(r["phrase_length"]) for r in rows.values()}
    assert orders == {1, 2}


def test_documents_and_obs_pct_count_documents_not_occurrences(tmp_path):
    _, rows = _freq_list(tmp_path, ["cat cat cat", "cat dog", "dog dog"])
    assert rows["cat"]["frequency"] == "4"
    assert rows["cat"]["documents"] == "2"
    assert float(rows["cat"]["obs_pct"]) == pytest.approx(2 / 3 * 100, abs=1e-3)
    assert float(rows["cat"]["idf"]) == pytest.approx(math.log(3 / 2), abs=1e-4)


def test_short_documents_are_skipped_entirely(tmp_path):
    """A skipped document must not count toward document totals either, or
    obs_pct silently deflates."""
    _, rows = _freq_list(tmp_path, ["cat dog bird fish", "cat"],
                         min_token_count=2)
    assert rows["cat"]["documents"] == "1"
    assert float(rows["cat"]["obs_pct"]) == 100.0


def test_a_corpus_with_no_usable_document_refuses(tmp_path):
    with pytest.raises(ValueError, match="min_token_count"):
        _freq_list(tmp_path, ["hi", "yo"], min_token_count=10)


def test_lemmatization_merges_and_happens_before_the_stoplist(tmp_path):
    """
    "cats" and "cat" become one row -- and a stoplist entry "cat" catches
    the text's "cats", which is only true when lemmatization runs first.
    """
    from taters.helpers.nltk_data import ensure_wordnet

    if not ensure_wordnet(verbose=False):
        pytest.skip("WordNet data unavailable")

    docs = ["cats sat here", "cat sat there"]
    _, plain = _freq_list(tmp_path, docs, lemmatize=True)
    assert plain["cat"]["frequency"] == "2"
    assert "cats" not in plain

    stop = tmp_path / "stop.txt"
    stop.write_text("cat\n", encoding="utf-8")
    _, stopped = _freq_list(tmp_path, docs, lemmatize=True,
                            stoplist_paths=[stop])
    assert "cat" not in stopped and "cats" not in stopped
    assert "sit" in stopped     # "sat" lemmatizes too (verb pass), and stays


# ---------------------------------------------------------------------------
# shared preparation
# ---------------------------------------------------------------------------

def test_the_shipped_stoplists_load_despite_bom_and_crlf():
    shipped = (Path(__file__).parent.parent / "src" / "taters" / "data"
               / "library" / "stoplists")
    stopset = load_stoplist([shipped / "stopwords-en.txt", shipped / "_chars.txt"])
    assert "a" in stopset            # the BOM-carrying first lines survive
    assert "`" in stopset
    assert "" not in stopset
    assert not any("\r" in s for s in stopset)


def test_a_stoplist_folder_expands_and_a_missing_file_refuses(tmp_path):
    (tmp_path / "one.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Beta\n", encoding="utf-8")
    assert load_stoplist([tmp_path]) == {"alpha", "beta"}
    with pytest.raises(FileNotFoundError):
        load_stoplist([tmp_path / "gone.txt"])


def test_stoplisted_drops_on_any_constituent_token():
    stop = {"the"}
    assert stoplisted("the", stop)
    assert stoplisted("the cat", stop)
    assert not stoplisted("cat sat", stop)
    assert not stoplisted("cat sat", set())


def test_iter_ngrams_covers_every_window():
    assert list(iter_ngrams(["a", "b", "c"], 2)) == ["a b", "b c"]
    assert list(iter_ngrams(["a"], 2)) == []


def test_the_token_stream_lemmatizes_nouns_then_verbs():
    from taters.helpers.nltk_data import ensure_wordnet

    if not ensure_wordnet(verbose=False):
        pytest.skip("WordNet data unavailable")
    stream = make_token_stream(lemmatize=True)
    assert stream("The cats were running") == ["the", "cat", "be", "run"]


# ---------------------------------------------------------------------------
# the doc-term matrix
# ---------------------------------------------------------------------------

def _dtm(tmp_path, docs, freq_csv, **kwargs):
    src = _corpus(tmp_path, docs, name="dtm_in.csv")
    out = build_doc_term_matrix(
        freq_list_csv=freq_csv,
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "dtm_gathered.csv",
        out_features_csv=tmp_path / "dtm.csv",
        overwrite_existing=True, **kwargs)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return out, {r["text_id"]: r for r in rows}


def test_the_longest_ngram_wins_and_consumes_its_tokens(tmp_path):
    """BUTTER's own worked example: with a vocabulary of 'health' and 'health
    behaviors', the sentence 'I am interested in health, and I study health
    behaviors' scores each exactly once -- the token inside the matched bigram
    is not also counted as a unigram."""
    vocab = tmp_path / "vocab.csv"
    vocab.write_text("ngram,frequency,obs_pct,idf\n"
                     "health,2,100,0.0\n"
                     "health behaviors,1,100,0.0\n", encoding="utf-8")
    docs = ["i am interested in health and i study health behaviors"]
    _, by_id = _dtm(tmp_path, docs, vocab)

    assert by_id["d0"]["health"] == "1"
    assert by_id["d0"]["health behaviors"] == "1"


def test_an_ngram_ending_the_document_still_matches(tmp_path):
    """The C# scanner's window guard was off by one: a document exactly as
    long as the n-gram could never match it."""
    freq_csv, _ = _freq_list(tmp_path, ["snow leopard"] * 3)
    _, by_id = _dtm(tmp_path, ["snow leopard"], freq_csv)
    assert by_id["d0"]["snow leopard"] == "1"


def test_weighting_methods_agree_with_their_definitions(tmp_path):
    docs = ["cat cat dog", "dog dog dog"]
    freq_csv, freqs = _freq_list(tmp_path, docs, ngram_n=1)

    _, count = _dtm(tmp_path, docs, freq_csv, weighting="count")
    assert count["d0"]["cat"] == "2" and count["d0"]["token_count"] == "3"

    _, binary = _dtm(tmp_path, docs, freq_csv, weighting="binary")
    assert binary["d0"]["cat"] == "1" and binary["d1"]["cat"] == "0"

    _, rel = _dtm(tmp_path, docs, freq_csv, weighting="relfreq")
    assert float(rel["d0"]["cat"]) == pytest.approx(2 / 3, abs=1e-4)

    _, tfidf = _dtm(tmp_path, docs, freq_csv, weighting="tfidf")
    assert float(tfidf["d0"]["cat"]) == pytest.approx(
        2 * float(freqs["cat"]["idf"]), abs=1e-3)


def test_top_n_subsets_the_vocabulary_and_keeps_ties(tmp_path):
    docs = ["cat cat cat dog dog bird"]
    freq_csv, _ = _freq_list(tmp_path, docs, ngram_n=1)
    out, _ = _dtm(tmp_path, docs, freq_csv, vocab_top_n=1)
    with open(out, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header == ["text_id", "token_count", "cat"]

    # two terms tied at the cut-off both survive, since choosing between tied
    # terms should never be arbitrary
    docs2 = ["cat cat dog dog bird"]
    freq_csv2, _ = _freq_list(tmp_path, docs2, ngram_n=1)
    out2, _ = _dtm(tmp_path, docs2, freq_csv2, vocab_top_n=1)
    with open(out2, encoding="utf-8-sig", newline="") as f:
        header2 = next(csv.reader(f))
    assert header2 == ["text_id", "token_count", "cat", "dog"]


def test_a_file_that_is_not_a_frequency_list_is_named_and_refused(tmp_path):
    imposter = tmp_path / "other.csv"
    imposter.write_text("text_id,words\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frequency list"):
        build_doc_term_matrix(
            freq_list_csv=imposter,
            csv_path=_corpus(tmp_path, ["hi there"]),
            text_cols=["text"], id_cols=["text_id"],
            out_features_csv=tmp_path / "dtm.csv", overwrite_existing=True)


def test_a_vocabulary_no_term_survives_is_refused(tmp_path):
    docs = ["cat dog"]
    freq_csv, _ = _freq_list(tmp_path, docs, ngram_n=1)
    # `vocab_min_freq` only gets read when it's the chosen rule. exactly one
    # rule applies, so naming it is part of asking for it
    with pytest.raises(ValueError, match="survived"):
        _dtm(tmp_path, docs, freq_csv, vocab_rule="min_freq",
             vocab_min_freq=99)


# ---------------------------------------------------------------------------
# the pipeline wiring
# ---------------------------------------------------------------------------

def test_the_dtm_pulls_the_frequency_list_in_and_after_it():
    """
    Selecting only the DTM must auto-include the frequency step and order it
    first -- on the text path too, where the transcript requirement (and, for
    a while, every other requirement with it) evaporates.
    """
    from taters.ui.compose import resolve_selection

    for source in ("csv", "txt_dir", "media"):
        ids = [r.id for r in resolve_selection(["doc_term_matrix"], source=source)]
        assert "ngram_frequencies" in ids, source
        assert ids.index("ngram_frequencies") < ids.index("doc_term_matrix"), source


def test_the_two_steps_share_one_lemmatize_variable():
    from taters.ui.compose import compose

    preset = compose(["doc_term_matrix"], name="x", source="csv",
                     input_path="texts.csv")
    steps = {s["call"]: s for s in preset["steps"]}
    freq = steps["potato.text.analyze_ngram_frequencies"]
    dtm = steps["potato.text.build_doc_term_matrix"]
    assert freq["with"]["lemmatize"] == "{{var:lemmatize}}"
    assert dtm["with"]["lemmatize"] == "{{var:lemmatize}}"
    # lemmatized by default: for these steps the question is always "which
    # words does this text use", and cats/cat are the same word for that
    assert preset["vars"]["lemmatize"] is True
    # and the DTM reads the frequency step's actual output, wherever it went
    assert dtm["with"]["freq_list_csv"] == "{{ngram_freqs}}"
    # vanilla by default: no stoplist unless somebody picked one
    assert "stoplist_paths" not in freq["with"]


# ---------------------------------------------------------------------------
# from the terminal
# ---------------------------------------------------------------------------

def test_both_clis_run_end_to_end(tmp_path, capsys):
    """`python -m taters.text.analyze_ngram_frequencies` and the DTM CLI,
    driven exactly as a terminal would."""
    from taters.text import analyze_ngram_frequencies as freq_mod
    from taters.text import build_doc_term_matrix as dtm_mod

    src = _corpus(tmp_path, ["snow leopard hunts", "snow leopard sleeps"])
    freq_mod.main([
        "--csv-path", str(src), "--id-col", "text_id",
        "--gathered-csv", str(tmp_path / "g.csv"),
        "--out-features-csv", str(tmp_path / "freqs.csv"),
        "--ngram-n", "2", "--min-freq", "1", "--min-obs-pct", "0",
        "--min-token-count", "1", "--overwrite-existing",
    ])
    assert "freqs.csv" in capsys.readouterr().out

    dtm_mod.main([
        "--freq-list-csv", str(tmp_path / "freqs.csv"),
        "--csv-path", str(src), "--id-col", "text_id",
        "--gathered-csv", str(tmp_path / "g2.csv"),
        "--out-features-csv", str(tmp_path / "dtm.csv"),
        "--weighting", "binary", "--overwrite-existing",
    ])
    with open(tmp_path / "dtm.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["snow leopard"] == "1"


def test_the_taters_facade_reaches_both_functions(tmp_path):
    """potato.text.* in a preset resolves through the Taters class; both new
    steps must be reachable there, lazily, like every other module."""
    from taters.Taters import Taters

    t = Taters()
    src = _corpus(tmp_path, ["snow leopard hunts", "snow leopard rests"])
    out = t.analyze_ngram_frequencies(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "g.csv",
        out_features_csv=tmp_path / "freqs.csv",
        ngram_n=2, min_freq=1, min_obs_pct=0, min_token_count=1,
        overwrite_existing=True)
    dtm = t.build_doc_term_matrix(
        freq_list_csv=out, csv_path=src, text_cols=["text"],
        id_cols=["text_id"], gathered_csv=tmp_path / "g2.csv",
        out_features_csv=tmp_path / "dtm.csv", overwrite_existing=True)
    assert Path(dtm).is_file()


def test_function_words_survive_lemmatization_unmangled():
    """
    Found by an end-to-end run: a noun-first lemmatizer turns "was" into "wa",
    "us" into "u", "does" into "doe" -- and those garbage lemmas then TOP a
    frequency list, being the most frequent words in the language. POS-guided
    lemmatization leaves words WordNet has no category for alone.
    """
    from taters.helpers.nltk_data import ensure_pos_tagger, ensure_wordnet

    if not (ensure_wordnet(verbose=False) and ensure_pos_tagger(verbose=False)):
        pytest.skip("NLTK data unavailable")

    stream = make_token_stream(lemmatize=True)
    tokens = stream("it was good for us and it does suit his cats")
    assert "wa" not in tokens and "u" not in tokens and "doe" not in tokens
    assert "be" in tokens          # "was", properly, as a verb
    assert "us" in tokens and "his" in tokens
    assert "cat" in tokens and "do" in tokens


def test_a_term_named_text_id_cannot_clobber_the_id_column(tmp_path):
    """
    Found by adversarial probing: "text_id" is a plausible token, and a term
    column by that name duplicated the header -- reading the DTM back by
    column name then returned the term's *count* where the document id should
    be. Colliding term columns get a trailing underscore instead.
    """
    docs = ["text_id token_count text_id", "text_id token_count appears"]
    freq_csv, _ = _freq_list(tmp_path, docs, ngram_n=1)
    out, by_id = _dtm(tmp_path, docs, freq_csv)

    with open(out, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert len(header) == len(set(header)), f"duplicate columns: {header}"
    assert by_id["d0"]["text_id"] == "d0"
    assert by_id["d0"]["text_id_"] == "2"       # the term, disambiguated
    assert by_id["d0"]["token_count_"] == "1"


# ---------------------------------------------------------------------------
# POS-tagged terms
# ---------------------------------------------------------------------------

def _tagger_ready():
    from taters.helpers.nltk_data import ensure_pos_tagger
    return ensure_pos_tagger(verbose=False)


def test_pos_tagged_terms_keep_the_verb_and_the_noun_apart(tmp_path):
    """'we felt the felt' carries both: felt/VBD and felt/NN become separate
    rows, each naming its tag in the pos column."""
    if not _tagger_ready():
        pytest.skip("NLTK tagger data unavailable")
    out, _rows = _freq_list(tmp_path, ["we felt the felt", "we felt the felt"],
                            ngram_n=1, pos_tagged=True)
    # a dict keyed by ngram collapses the twins, so we read the raw file instead
    with open(out, encoding="utf-8-sig", newline="") as f:
        raw = [r for r in csv.DictReader(f) if r["ngram"] == "felt"]
    assert sorted(r["pos"] for r in raw) == ["NN", "VBD"]
    assert all(r["frequency"] == "2" for r in raw)


def test_the_pos_column_exists_only_when_asked_for(tmp_path):
    from taters.text.analyze_ngram_frequencies import header_for

    assert "pos" not in header_for(False) and header_for(False) == HEADER
    assert header_for(True).index("pos") == 2

    out, _ = _freq_list(tmp_path, ["the cat sat"])
    with open(out, encoding="utf-8-sig", newline="") as f:
        assert "pos" not in next(csv.reader(f))


def test_a_stoplist_still_matches_the_word_half_of_tagged_terms(tmp_path):
    if not _tagger_ready():
        pytest.skip("NLTK tagger data unavailable")
    stop = tmp_path / "stop.txt"
    stop.write_text("the\n", encoding="utf-8")
    _, rows = _freq_list(tmp_path, ["the cat sat", "the cat ran"],
                         ngram_n=1, pos_tagged=True, stoplist_paths=[stop])
    assert "the" not in rows
    assert "cat" in rows


def test_the_dtm_separates_tagged_twins_into_readable_columns(tmp_path):
    if not _tagger_ready():
        pytest.skip("NLTK tagger data unavailable")
    docs = ["we felt the felt", "we felt the felt"]
    freq_csv, _ = _freq_list(tmp_path, docs, ngram_n=1, pos_tagged=True)
    out, by_id = _dtm(tmp_path, docs, freq_csv, pos_tagged=True)

    assert by_id["d0"]["felt (VBD)"] == "1"
    assert by_id["d0"]["felt (NN)"] == "1"


def test_a_pos_mismatch_is_refused_in_both_directions(tmp_path):
    if not _tagger_ready():
        pytest.skip("NLTK tagger data unavailable")
    docs = ["the cat sat here"]
    plain_csv, _ = _freq_list(tmp_path, docs, ngram_n=1)
    with pytest.raises(ValueError, match="must agree"):
        _dtm(tmp_path, docs, plain_csv, pos_tagged=True)

    tagged_csv, _ = _freq_list(tmp_path, docs, ngram_n=1, pos_tagged=True)
    with pytest.raises(ValueError, match="must agree"):
        _dtm(tmp_path, docs, tagged_csv, pos_tagged=False)


def test_the_two_steps_share_the_pos_tagged_variable_too():
    from taters.ui.compose import compose

    preset = compose(["doc_term_matrix"], name="x", source="csv",
                     input_path="texts.csv")
    steps = {s["call"]: s for s in preset["steps"]}
    freq = steps["potato.text.analyze_ngram_frequencies"]
    dtm = steps["potato.text.build_doc_term_matrix"]
    assert freq["with"]["pos_tagged"] == "{{var:pos_tagged}}"
    assert dtm["with"]["pos_tagged"] == "{{var:pos_tagged}}"
    assert preset["vars"]["pos_tagged"] is False


# ---------------------------------------------------------------------------
# the Stanza engine
# ---------------------------------------------------------------------------

#: What it takes to run one of these: a real Stanza neural pipeline,
#: downloaded and loaded into memory. That is precisely what `slow` is for in
#: this suite ("needs real media, ML models, or minutes of runtime"), and
#: these seven tests were about a sixth of the default run's wall clock.
#: They still run, under `pytest -m slow`, and they have to stay real -- there
#: is no way to fake context-aware lemmatization or a GPU device placement.
#: The code paths that need no model are covered separately and fast, by
#: `test_the_two_engines_agree_when_there_is_nothing_to_tag` above and by the
#: tests that monkeypatch `_stanza_pipeline`.
#:
#: `stanza_ready` (conftest) probes availability lazily. It used to be a
#: module-level `skipif`, which built a throwaway Stanza pipeline during
#: COLLECTION on every pytest invocation -- see that fixture's docstring.
_STANZA_MARKS = (pytest.mark.slow,
                 pytest.mark.usefixtures("stanza_ready"))


def needs_stanza(fn):
    """Both marks at once, for a test that needs the real Stanza model."""
    for mark in _STANZA_MARKS:
        fn = mark(fn)
    return fn


def test_the_invalid_engine_combinations_are_refused():
    with pytest.raises(ValueError, match="engine"):
        make_token_stream(engine="spacy")
    with pytest.raises(ValueError, match="tokenizer"):
        make_token_stream(tokenizer="whitespace")
    with pytest.raises(ValueError, match="needs engine='stanza'"):
        make_token_stream(engine="nltk", tokenizer="stanza")


def test_plain_potts_needs_no_engine_at_all():
    """With nothing to tag or lemmatize, both engines yield Potts's tokens --
    identical, and without loading any model."""
    nltk_stream = make_token_stream(engine="nltk")
    stanza_stream = make_token_stream(engine="stanza")   # no model needed
    text = "Check :-) it's #great and http://x.co today"
    assert nltk_stream(text) == stanza_stream(text)


@needs_stanza
def test_stanza_lemmas_pass_the_canaries_wordnet_needed_pos_tags_for(tmp_path):
    """Context-aware lemmatization: both 'felt's lemmatize to 'feel', 'were'
    to 'be', and the function words come through unmangled."""
    stream = make_token_stream(lemmatize=True, engine="stanza", device="cpu")
    tokens = stream("we felt the felt and the cats were running")
    assert tokens.count("feel") == 2
    assert "cat" in tokens and "be" in tokens and "run" in tokens
    assert "we" in tokens and "the" in tokens
    assert "wa" not in tokens and "u" not in tokens


@needs_stanza
def test_stanza_tagged_stream_keeps_the_house_shape():
    """Same TAG_SEP contract as NLTK: word halves, Penn-shaped tags."""
    stream = make_token_stream(pos_tagged=True, engine="stanza", device="cpu")
    tokens = stream("the cats were running")
    assert words_of(" ".join(tokens)) == "the cats were running"
    tags = tags_of(" ".join(tokens)).split(" ")
    assert tags[0] == "DT" and tags[1] == "NNS"


@needs_stanza
def test_stanzas_own_tokenizer_keeps_urls_and_emoticons_whole():
    from taters.text.ngram_prep import make_tagged_stream

    tagged = make_tagged_stream(engine="stanza", tokenizer="stanza",
                                device="cpu")
    words = [w for w, _t in tagged("Check http://x.co :-) it's GREAT")]
    assert "http://x.co" in words
    assert ":-)" in words
    assert "great" in words, "stanza preserves case; the stream must fold it"


@needs_stanza
def test_an_empty_document_does_not_crash_stanza():
    """Found by probing: stanza raises IndexError on an empty pretokenized
    sentence, so the stream must not hand it one."""
    stream = make_token_stream(lemmatize=True, engine="stanza", device="cpu")
    assert stream("") == []


# `pytest.param` won't take `usefixtures` (on purpose), so the stanza case only
# carries the `slow` mark here and asks for the availability probe from inside
# the test. that keeps the probe lazy in the way that matters: it runs for the
# stanza parameter only, and only when that parameter hasn't been deselected
@pytest.mark.parametrize("engine", ["nltk",
                                    pytest.param("stanza",
                                                 marks=pytest.mark.slow)])
def test_both_engines_honor_both_tagsets(engine, request):
    from taters.text.ngram_prep import make_tagged_stream

    if engine == "stanza":
        request.getfixturevalue("stanza_ready")

    penn = make_tagged_stream(engine=engine, tagset="penn", device="cpu")
    tags = {t for _w, t in penn("the cat sat quickly")}
    assert "DT" in tags and "NN" in tags

    universal = make_tagged_stream(engine=engine, tagset="universal",
                                   device="cpu")
    tags = {t for _w, t in universal("the cat sat quickly")}
    assert "NOUN" in tags and "DET" in tags


@needs_stanza
def test_stanza_runs_on_the_cpu_and_on_the_gpu():
    import torch

    from taters.text.ngram_prep import _stanza_pipeline

    cpu = _stanza_pipeline("en", pretokenized=True, device="cpu")
    assert str(next(cpu.processors["pos"].trainer.model.parameters()).device) == "cpu"

    if not torch.cuda.is_available():
        pytest.skip("stanza GPU check uses the GPU")
    gpu = _stanza_pipeline("en", pretokenized=True, device="cuda")
    assert str(next(gpu.processors["pos"].trainer.model.parameters()).device).startswith("cuda")


@needs_stanza
def test_the_frequency_list_runs_end_to_end_on_stanza(tmp_path):
    """Lemmatized, tagged, CPU: the two 'felt's merge as the lemma 'feel' but
    stay apart by tag -- exactly the pos_tagged promise, on the new engine."""
    _, rows = _freq_list(tmp_path, ["we felt the felt", "we felt the felt"],
                         ngram_n=1, lemmatize=True, pos_tagged=True,
                         engine="stanza", device="cpu")
    out = tmp_path / "freqs.csv"
    with open(out, encoding="utf-8-sig", newline="") as f:
        feels = [r for r in csv.DictReader(f) if r["ngram"] == "feel"]
    assert len(feels) == 2
    assert all(r["frequency"] == "2" for r in feels)


def test_the_dtm_defaults_to_the_top_500_terms(tmp_path):
    """Requested default: an untouched DTM keeps the 500 most frequent terms
    -- a workable matrix width, and the vocabulary a topic model or an
    open-vocabulary prediction gets to work with -- rather than every term
    in the corpus. It was 250, which cut into a corpus's ordinary words."""
    import inspect

    from taters.ui import recipes as _r

    sig = inspect.signature(build_doc_term_matrix)
    assert sig.parameters["vocab_top_n"].default == 500
    # the matrix step reads it from a variable, and the 500 lives there.
    #
    # MEM used to read that same variable, because it scanned the matrix this
    # step built and the two had to agree about the vocabulary. It builds its
    # own now, so it keeps its own 500 on the function instead -- and the
    # generative models ask for more, because they have more topics to separate
    # and starve on a small vocabulary.
    recipe = _r.by_id("doc_term_matrix")
    assert recipe.with_["vocab_top_n"] == "{{var:vocab_top_n}}"
    assert recipe.vars["vocab_top_n"]["default"] == 500

    from taters.text.topic_model_lda import topic_model_lda
    from taters.text.topic_model_mem import topic_model_mem

    assert "vocab_top_n" not in _r.by_id("topic_model_mem").with_, (
        "MEM takes the matrix step's vocabulary setting again")
    assert inspect.signature(topic_model_mem).parameters["vocab_top_n"].default == 500
    assert inspect.signature(topic_model_lda).parameters["vocab_top_n"].default == 2000

    # and in practice: 501 terms with distinct frequencies -> 500 columns
    words = " ".join(f"w{i:03d} " * i for i in range(1, 502))
    freq_csv, _ = _freq_list(tmp_path, [words], ngram_n=1)
    out, _ = _dtm(tmp_path, [words], freq_csv)
    with open(out, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert len(header) == 2 + 500
    assert "w001" not in header, "the single rarest term is the one dropped"


# ---------------------------------------------------------------------------
# bounded memory: the disk path writes the identical file
# ---------------------------------------------------------------------------

def _spied_runs(tmp_path, monkeypatch, docs, **kwargs):
    """Run the analyzer twice -- unlimited RAM, then a starved budget -- and
    return (plain_path, spilled_path, spy_instances)."""
    import taters.text.analyze_ngram_frequencies as anf
    from taters.helpers.spill_counter import SpillCounter

    made = []

    class Spy(SpillCounter):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            made.append(self)

    monkeypatch.setattr(anf, "SpillCounter", Spy)

    src = _corpus(tmp_path, docs)
    shared = dict(csv_path=src, text_cols=["text"], id_cols=["text_id"],
                  min_freq=1, min_obs_pct=0, min_token_count=1,
                  overwrite_existing=True, **kwargs)
    plain = anf.analyze_ngram_frequencies(
        gathered_csv=tmp_path / "g1.csv",
        out_features_csv=tmp_path / "plain.csv", **shared)
    assert made[-1].spills == 0, "the reference run must be the in-memory path"
    spilled = anf.analyze_ngram_frequencies(
        gathered_csv=tmp_path / "g2.csv",
        out_features_csv=tmp_path / "spilled.csv",
        max_ram_mb=0, **shared)
    assert made[-1].spills > 0, "the disk path never ran -- vacuous comparison"
    return plain, spilled, made


def test_a_starved_ram_budget_writes_the_identical_file(tmp_path, monkeypatch):
    """The whole contract in one line: spilling changes WHERE counts live,
    never what the file says. The corpus deliberately holds both "a b" and
    "a bx" so a sloppy prefix match (string startswith without the space)
    would grab the wrong left count for "a bx"'s extensions."""
    docs = [
        "the cat sat on the mat and the cat ran off",
        "a b c a b c a bx y a bx y a b c",
        "the mat sat on a cat and a b c went by",
        "peanut butter peanut butter jelly time a bx y",
    ] * 3
    plain, spilled, _made = _spied_runs(tmp_path, monkeypatch, docs, ngram_n=3)
    assert Path(spilled).read_bytes() == Path(plain).read_bytes()
    assert len(Path(plain).read_text(encoding="utf-8-sig").splitlines()) > 40


def test_the_disk_path_survives_pos_tags_in_the_grams(tmp_path, monkeypatch):
    """pos_tagged grams carry \\x1f, which sorts below space: raw-string
    ordering of the merged stream would break the prefix stack exactly here."""
    docs = [
        "the quick dog sat on the mat",
        "the quick dog ran and the cat sat",
        "a quick cat sat on a dog",
    ] * 2
    plain, spilled, _made = _spied_runs(tmp_path, monkeypatch, docs,
                                        ngram_n=2, pos_tagged=True)
    assert Path(spilled).read_bytes() == Path(plain).read_bytes()


# ---------------------------------------------------------------------------
# tokens may not contain the n-gram separator
# ---------------------------------------------------------------------------

def test_page_ranges_and_phone_shapes_cannot_poison_the_grams(tmp_path):
    """The Potts phone-number pattern matches "pp. 1467- 1470" as ONE token
    with a space in it -- which the gram machinery then reads as a bigram
    whose halves were never counted. Found as a KeyError on the first real
    corpus of academic PDFs; the split keeps the pieces as tokens."""
    docs = ["see pp. 1467- 1470 for details and call 555 123 4567 today"] * 3
    _out, rows = _freq_list(tmp_path, docs, ngram_n=2)
    assert "1467-" in rows and "1470" in rows, "the halves are real unigrams"
    assert not any(g.count(" ") + 1 > 2 for g in rows), (
        "a spaced token would masquerade as a higher-order gram"
    )


def test_the_potts_stream_never_emits_a_token_with_a_space():
    stream = make_token_stream(False, False)
    for text in ("pp. 1467- 1470.", "call 555 123 4567", "plain words here"):
        assert all(" " not in tok for tok in stream(text)), text


def test_stanza_multiword_tokens_are_flattened_with_their_tag(monkeypatch):
    """Stanza's own tokenizer can emit multi-word tokens in some languages;
    the parts must become separate tokens, each carrying the word's tag."""
    from taters.text import ngram_prep as prep

    class Word:
        text = "Hà Nội"
        lemma = None
        upos = "PROPN"
        xpos = None

    monkeypatch.setattr(prep, "_stanza_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(prep, "_stanza_words",
                        lambda pipe, tok, text: iter([Word()]))
    # tokenizer="stanza" forces the stanza path, while plain potts short-circuits it
    plain = prep.make_token_stream(False, False, engine="stanza",
                                   tokenizer="stanza")
    assert plain("whatever") == ["hà", "nội"]
    tagged = prep.make_token_stream(False, True, engine="stanza")
    assert tagged("whatever") == [f"hà{prep.TAG_SEP}PROPN",
                                  f"nội{prep.TAG_SEP}PROPN"]


# ---------------------------------------------------------------------------
# parallel counting
# ---------------------------------------------------------------------------

def test_parallel_counting_writes_the_identical_file(tmp_path):
    """The pool tokenizes and tallies; the parent folds results in submission
    order -- so the file is byte-identical whatever the worker count. Explicit
    workers, because automatic deliberately stays serial on a small corpus."""
    docs = ["the cat sat on the mat and a bx y went by " * 3] * 8
    src = _corpus(tmp_path, docs)
    kwargs = dict(csv_path=src, text_cols=["text"], id_cols=["text_id"],
                  min_freq=1, min_obs_pct=0, min_token_count=1, ngram_n=2,
                  overwrite_existing=True)
    serial = analyze_ngram_frequencies(
        gathered_csv=tmp_path / "g1.csv",
        out_features_csv=tmp_path / "serial.csv", workers=1, **kwargs)
    pooled = analyze_ngram_frequencies(
        gathered_csv=tmp_path / "g2.csv",
        out_features_csv=tmp_path / "pooled.csv", workers=2, **kwargs)
    assert Path(pooled).read_bytes() == Path(serial).read_bytes()


def test_counting_reports_a_named_phase_with_documents_in_flight(tmp_path):
    """The counting pass is a fan-out phase like gathering and scoring, so it
    announces itself the same way: a named message plus the documents
    currently in flight, which is what draws the live sub-bars."""
    events = []

    def sink(done, total, message=None, unit=None, **extra):
        events.append((message, extra.get("inflight")))

    src = _corpus(tmp_path, ["the cat sat on the mat again"] * 4)
    analyze_ngram_frequencies(
        csv_path=src, text_cols=["text"], id_cols=["text_id"],
        gathered_csv=tmp_path / "g.csv", out_features_csv=tmp_path / "f.csv",
        min_freq=1, min_obs_pct=0, min_token_count=1,
        overwrite_existing=True, on_progress=sink)
    counting = [(m, i) for m, i in events if m == "counting n-grams"]
    assert counting, "the counting phase never announced itself"
    assert any(i for _m, i in counting), "no document was ever shown in flight"


def test_counting_workers_policy(monkeypatch):
    """Stanza-with-a-model stays at one process; automatic stays serial on a
    small corpus; an explicit ask is honored either way."""
    from taters.helpers import parallel_map as pm
    from taters.text.analyze_ngram_frequencies import _counting_workers

    monkeypatch.setattr(pm.os, "cpu_count", lambda: 8)
    base = dict(engine="nltk", tokenizer="potts",
                lemmatize=False, pos_tagged=False)

    assert _counting_workers(0, 10_000, **base) == 6, "auto = 8 - 8//4"
    assert _counting_workers(0, 10, **base) == 1, "small corpus: spawn tax"
    assert _counting_workers(3, 10, **base) == 3, "an explicit ask is honored"
    for engaged in (dict(engine="stanza", tokenizer="potts", lemmatize=True,
                         pos_tagged=False),
                    dict(engine="stanza", tokenizer="potts", lemmatize=False,
                         pos_tagged=True),
                    dict(engine="stanza", tokenizer="stanza", lemmatize=False,
                         pos_tagged=False)):
        assert _counting_workers(0, 10_000, **engaged) == 1, engaged
    plain_stanza = dict(engine="stanza", tokenizer="potts",
                        lemmatize=False, pos_tagged=False)
    assert _counting_workers(0, 10_000, **plain_stanza) == 6, (
        "nothing needs the model, so the potts shortcut pools"
    )


def test_the_dtm_filename_carries_its_weighting(tmp_path, monkeypatch):
    """Rerunning with a different weighting must write a SIBLING file: with a
    bare default name and overwrite off, the second run silently returned the
    first run's numbers under the new run's intent."""
    monkeypatch.chdir(tmp_path)
    docs = ["the cat sat on the mat", "a cat ran up a tree today"]
    freq_csv, _ = _freq_list(tmp_path, docs, ngram_n=1)

    outs = {}
    for weighting in ("count", "binary", "relfreq", "tfidf"):
        outs[weighting] = build_doc_term_matrix(
            freq_list_csv=freq_csv, analysis_csv=tmp_path / "gathered.csv",
            weighting=weighting, overwrite_existing=False)   # defaults on purpose
    names = {w: Path(p).name for w, p in outs.items()}
    assert len(set(names.values())) == 4, names
    for w, name in names.items():
        assert w in name, f"{name} does not say it is the {w} matrix"

    # and the contents really do differ (the old bug: the same file four times)
    contents = {Path(p).read_bytes() for p in outs.values()}
    assert len(contents) == 4

    # the TUI writes through the same convention, driven by the shared var
    from taters.ui import recipes as _r
    template = _r.by_id("doc_term_matrix").with_["out_features_csv"]
    assert "{{var:weighting}}" in template


def test_the_stanza_download_cannot_scribble_on_the_terminal(monkeypatch, capfd):
    """The model fetch runs while the live display owns the screen, so its
    raw tqdm stderr bars sliced through the progress bars (a real report).
    ensure_stanza must swallow the download's stream chatter and ask the hub
    to keep its bars off."""
    import sys
    import types

    recorded = {}

    class _DownloadMethod:
        NONE = "none"

    def _pipeline(**_kw):
        raise RuntimeError("no model yet")

    def _download(lang, verbose=False):
        import os

        recorded["env"] = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        print("default.zip:  42%|####      | raw tqdm chatter", file=sys.stderr)

    fake = types.ModuleType("stanza")
    fake.Pipeline = _pipeline
    fake.download = _download
    core = types.ModuleType("stanza.pipeline.core")
    core.DownloadMethod = _DownloadMethod
    pipeline_pkg = types.ModuleType("stanza.pipeline")
    pipeline_pkg.core = core
    fake.pipeline = pipeline_pkg
    monkeypatch.setitem(sys.modules, "stanza", fake)
    monkeypatch.setitem(sys.modules, "stanza.pipeline", pipeline_pkg)
    monkeypatch.setitem(sys.modules, "stanza.pipeline.core", core)

    from taters.helpers.stanza_data import ensure_stanza

    assert ensure_stanza("en", verbose=False) is False   # fake never "works"
    assert recorded["env"] == "1", "the hub's off switch was not set"
    _out, err = capfd.readouterr()
    assert "tqdm" not in err and "default.zip" not in err, (
        "download chatter reached the terminal"
    )


# ---------------------------------------------------------------------------
# one vocabulary rule at a time
# ---------------------------------------------------------------------------

def _graded_vocab(tmp_path):
    """A frequency list where the three rules would each keep a different
    set, so applying more than one is visible in the result."""
    vocab = tmp_path / "graded.csv"
    vocab.write_text(
        "ngram,frequency,obs_pct,idf\n"
        "everywhere,100,100,0.0\n"     # common and widespread
        "common,90,20,0.0\n"           # common, but concentrated
        "spread,5,80,0.0\n"            # rare, but widespread
        "rare,2,10,0.0\n",             # neither
        encoding="utf-8")
    return vocab


def _kept(tmp_path, vocab, **kwargs):
    docs = ["everywhere common spread rare"]
    _out, rows = _dtm(tmp_path, docs, vocab, **kwargs)
    return {c for c in next(iter(rows.values()))
            if c not in ("text_id", "token_count")}


def test_exactly_one_vocabulary_rule_applies(tmp_path):
    """
    The three rules used to be applied *together*, so working out what a
    corpus would be left with meant intersecting a minimum frequency, a
    minimum percent-of-documents and a top-N in your head -- and nothing on
    screen said they combined at all (a real report: "it gets super
    confusing to try to decide how all three of those can intersect with
    each other").

    Each rule is checked here with the settings of the other two left at
    values that *would* exclude terms if they were still being applied. If
    any of them leaks, one of these sets comes back smaller.
    """
    vocab = _graded_vocab(tmp_path)

    # top_n: rank by frequency, keep two. min_obs_pct=90 would drop `common`,
    # and min_freq=50 would drop `spread`, and neither should get to bite
    assert _kept(tmp_path, vocab, vocab_rule="top_n", vocab_top_n=2,
                 vocab_min_obs_pct=90, vocab_min_freq=50) == \
        {"everywhere", "common"}

    # min_obs_pct: widespread terms, whatever their frequency. top_n=1 and
    # min_freq=50 would each cut this down if they were still being applied
    assert _kept(tmp_path, vocab, vocab_rule="min_obs_pct",
                 vocab_min_obs_pct=50, vocab_top_n=1,
                 vocab_min_freq=50) == {"everywhere", "spread"}

    # min_freq: common terms, however concentrated they are
    assert _kept(tmp_path, vocab, vocab_rule="min_freq", vocab_min_freq=50,
                 vocab_top_n=1, vocab_min_obs_pct=90) == \
        {"everywhere", "common"}


def test_a_vocabulary_rule_that_keeps_nothing_says_which_rule(tmp_path):
    """The refusal has to name the rule that emptied the vocabulary and its
    setting; "no term survived the filters" left the user to work out which
    of four numbers to change."""
    vocab = _graded_vocab(tmp_path)
    with pytest.raises(ValueError) as e:
        _kept(tmp_path, vocab, vocab_rule="min_freq", vocab_min_freq=10_000)
    assert "frequency >= 10000" in str(e.value)
    assert "vocab_rule" in str(e.value)

    with pytest.raises(ValueError, match="vocab_rule must be one of"):
        _kept(tmp_path, vocab, vocab_rule="whatever")


def test_the_ranking_setting_only_affects_the_top_n_rule(tmp_path):
    """`vocab_rank_by` is a setting *of* top_n. Ranking by share-of-documents
    keeps a different pair than ranking by raw count, and neither of the
    other two rules looks at it at all."""
    vocab = _graded_vocab(tmp_path)
    assert _kept(tmp_path, vocab, vocab_rule="top_n", vocab_top_n=2,
                 vocab_rank_by="frequency") == {"everywhere", "common"}
    assert _kept(tmp_path, vocab, vocab_rule="top_n", vocab_top_n=2,
                 vocab_rank_by="obs_pct") == {"everywhere", "spread"}
    # the other rules ignore it: same answer whichever way we set it
    for rank_by in ("frequency", "obs_pct"):
        assert _kept(tmp_path, vocab, vocab_rule="min_freq",
                     vocab_min_freq=50, vocab_rank_by=rank_by) == \
            {"everywhere", "common"}


def test_the_word_counting_steps_lemmatize_by_default(tmp_path):
    """For the frequency list, the matrix and the topic model the question is
    always "which words does this text use", and `cats`/`cat` are the same
    word for that purpose. Left off, a topic model spends part of its
    vocabulary on inflections of words it already has and a frequency list
    splits one word's count across its forms."""
    import inspect

    from taters.text.topic_model_mem import topic_model_mem
    from taters.ui import recipes as _r

    # the frequency list and the matrix share one variable, so those two
    # cannot disagree -- the matrix scans text against the frequency list's
    # vocabulary, and a mismatch there fails silently rather than loudly
    for rid in ("ngram_frequencies", "doc_term_matrix"):
        recipe = _r.by_id(rid)
        assert recipe.with_["lemmatize"] == "{{var:lemmatize}}", rid
        assert recipe.vars["lemmatize"]["default"] is True, rid
    assert len({_r.by_id(rid).with_["lemmatize"]
                for rid in ("ngram_frequencies", "doc_term_matrix")}) == 1

    # every topic model builds its own frequency list and matrix, so it gets
    # its own variable -- one study can want LDA over lemmatized unigrams and
    # NMF over raw bigrams, and sharing this made that impossible. what still
    # has to hold is the default: on, for every one of them.
    for rid, var in (("topic_model_mem", "mem_lemmatize"),
                     ("topic_model_lda", "lda_lemmatize"),
                     ("topic_model_nmf", "nmf_lemmatize")):
        recipe = _r.by_id(rid)
        assert recipe.with_["lemmatize"] == f"{{{{var:{var}}}}}", rid
        assert recipe.vars[var]["default"] is True, rid
    # the functions themselves keep their own conservative defaults, since the
    # app's answer is a pipeline decision, not a change to the API
    for fn in (analyze_ngram_frequencies, build_doc_term_matrix,
               topic_model_mem):
        assert "lemmatize" in inspect.signature(fn).parameters


# ---------------------------------------------------------------------------
# punctuation isn't vocabulary
# ---------------------------------------------------------------------------

def test_punctuation_is_not_counted_unless_asked(tmp_path):
    """A frequency list whose top term is "." is counting sentence
    boundaries, not words -- and the period was the one term a real topic
    model found constant and set aside with a warning. Off by default; on
    request, the punctuation (and with it the emoticons) come back."""
    # no commas in the documents, since the helper writes an unquoted CSV
    docs = ["I like potato. I like gravy :)", "Potato gravy -- and butter."]
    _out, words = _freq_list(tmp_path, docs, ngram_n=1)
    assert {".", "-", "--", ":)"}.isdisjoint(words)
    assert {"potato", "gravy", "like"} <= set(words)

    _out, kept = _freq_list(tmp_path, docs, ngram_n=1, keep_punctuation=True)
    assert {".", ":)"} <= set(kept)


def test_the_matrix_scans_with_the_same_punctuation_rule_as_its_vocabulary(
        tmp_path):
    """The two steps must agree or the scan fails silently. With the rule
    shared, a token count on the matrix side excludes what the list side
    never counted."""
    docs = ["potato. gravy. butter.", "potato potato!"]
    freq, _ = _freq_list(tmp_path, docs, ngram_n=1)
    _out, rows = _dtm(tmp_path, docs, freq)
    assert [int(rows[k]["token_count"]) for k in ("d0", "d1")] == [3, 2]
    assert "." not in rows["d0"]


def test_is_word_keeps_anything_with_a_letter_or_digit():
    from taters.text.ngram_prep import TAG_SEP, is_word

    assert all(is_word(t) for t in ("don't", "#tag", "2nd", f"felt{TAG_SEP}VBD"))
    assert not any(is_word(t) for t in (".", "--", ":)", f".{TAG_SEP}."))
