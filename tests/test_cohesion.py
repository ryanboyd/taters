"""
Cohesion indices: TAACO's measure definitions, minus TAACO's bugs.

Arithmetic canaries feed `cohesion_row`/`overlap_indices` hand-built
(word, lemma, tag) triples, so the numbers are checkable by hand and immune
to tagger drift. End-to-end tests then pin the fixes that TAACO's audit
motivated: NA instead of 0.0 for undefined indices, punctuation (and ``%``)
excluded from every class, paragraphs on blank lines rather than every
newline, and window-two overlap built without mutating shared state.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from taters.text.analyze_cohesion import (
    analyze_cohesion,
    classify_sentence,
    cohesion_row,
    header_columns,
    overlap_indices,
    _mattr,
)


def _run(tmp_path, rows, **kwargs):
    # our fast tests run model-free (house convention); the semantic tests opt
    # back in with a fake encoder, and the real model is @slow territory.
    kwargs.setdefault("semantic_model", "none")
    # and on the CPU. even with a fake encoder these tests were spinning up
    # CUDA, which costs real seconds and buys us nothing, since nothing here is
    # measuring device placement.
    kwargs.setdefault("device", "cpu")
    src = tmp_path / "ready.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text_id", "text"])
        w.writerows(rows)
    out = analyze_cohesion(analysis_csv=src,
                           out_features_csv=tmp_path / "cohesion.csv",
                           overwrite_existing=True, **kwargs)
    with open(out, encoding="utf-8-sig", newline="") as f:
        return {r["text_id"]: r for r in csv.DictReader(f)}, Path(out)


# ---------------------------------------------------------------------------
# The overlap engine, by hand
# ---------------------------------------------------------------------------

def test_overlap_triple_matches_hand_computation():
    segs = [["a", "b"], ["b", "c"], ["c", "d"]]
    # window 1: {a,b}&{b,c} -> 1, {b,c}&{c,d} -> 1; source types are 2+2.
    assert overlap_indices(segs, 1) == (0.5, 1.0, 1.0)
    # window 2: {a,b}&{b,c,c,d} -> 1; one comparison, 2 source types.
    assert overlap_indices(segs, 2) == (0.5, 1.0, 1.0)


def test_overlap_normalizes_by_source_types_not_tokens():
    """TAACO's stated definition: the denominator is the TYPE count of the
    earlier segments. A token-count denominator would divide by 3 here."""
    prop, _per, _bin = overlap_indices([["a", "a", "b"], ["b"]], 1)
    assert prop == 0.5                     # 1 overlap / 2 source types


def test_overlap_is_na_not_zero_when_undefined():
    """TAACO bug 4.18: one-segment documents scored 0.0 'cohesion'. NA and 0
    mean different things, and 5.6% of TAACO's own sample corpus was
    poisoned this way."""
    assert overlap_indices([["a"]], 1) == (None, None, None)
    assert overlap_indices([["a"], ["b"]], 2) == (None, None, None)
    assert overlap_indices([], 1) == (None, None, None)


def test_window_two_does_not_bleed_between_comparisons():
    """TAACO bug 4.2: it appended segment i+2 INTO its shared list, so the
    second comparison saw a corrupted source. Here segment 2's source types
    must still be exactly {b} -- an overlap of b with [c]+[d] is 0."""
    prop, per, binary = overlap_indices([["a"], ["b"], ["c"], ["d"]], 2)
    assert (prop, per, binary) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Word classes, by hand
# ---------------------------------------------------------------------------

def test_classes_follow_the_documented_rules():
    triples = [
        ("the", "the", "DT"), ("%", "%", "NN"), ("cats", "cat", "NNS"),
        ("were", "be", "VBD"), ("running", "run", "VBG"),
        ("quickly", "quickly", "RB"), ("very", "very", "RB"),
        ("home", "home", "NN"),
    ]
    c = classify_sentence(triples)
    assert "%" not in c["all"], "TAACO bug 4.19: % is not a word, let alone a noun"
    assert c["noun"] == ["cat", "home"]
    assert c["verb"] == ["run"], "content verbs only: 'be' is a function word"
    assert "be" in c["fw"]
    assert c["adv"] == ["quickly", "very"]
    assert "quickly" in c["cw"] and "very" not in c["cw"], (
        "deadjectival adverbs are content words; 'very' is not"
    )
    assert c["argument"] == ["cat", "home"]


def test_demonstratives_split_by_attendedness():
    attended = classify_sentence([("that", "that", "DT"), ("car", "car", "NN")])
    assert attended["pronoun"] == [], "'that car' is a determiner use"
    unattended = classify_sentence([("she", "she", "PRP"),
                                    ("likes", "like", "VBZ"),
                                    ("that", "that", "DT")])
    assert unattended["pronoun"] == ["she", "that"], (
        "bare 'that' is pronominal and joins the pronoun class (TAACO rule)"
    )
    assert unattended["argument"] == ["she", "that"]


# ---------------------------------------------------------------------------
# Whole-document arithmetic, by hand
# ---------------------------------------------------------------------------

def _doc():
    """Two paragraphs, three sentences, all tags unambiguous:
    P1: 'the cat chased the dog' / 'the dog saw the cat'
    P2: 'birds sing'."""
    s1 = [("the", "the", "DT"), ("cat", "cat", "NN"),
          ("chased", "chase", "VBD"), ("the", "the", "DT"),
          ("dog", "dog", "NN")]
    s2 = [("the", "the", "DT"), ("dog", "dog", "NN"), ("saw", "see", "VBD"),
          ("the", "the", "DT"), ("cat", "cat", "NN")]
    s3 = [("birds", "bird", "NNS"), ("sing", "sing", "VBP")]
    return [[s1, s2], [s3]]


def test_cohesion_row_matches_hand_computation():
    nwords, values, _s, _p = cohesion_row(_doc(), mattr_window=50)
    row = dict(zip(header_columns([]), values))   # no connective lists needed here
    assert nwords == 12
    # lemma types: the,cat,chase,dog,see,bird,sing = 7 types out of 12 tokens.
    assert row["lemma_ttr"] == pytest.approx(7 / 12)
    # content: cat,chase,dog + dog,see,cat + bird,sing = 8 tokens, 6 types.
    assert row["lexical_density_tokens"] == pytest.approx(8 / 12)
    assert row["content_ttr"] == pytest.approx(6 / 8)
    assert row["noun_ttr"] == pytest.approx(3 / 5)   # cat,dog,dog,cat,bird
    # sentence overlap, all lemmas: s1 types {the,cat,chase,dog} (4) vs s2 ->
    # {the,cat,dog} = 3; s2 types {the,dog,see,cat} (4) vs s3 -> 0.
    assert row["adjacent_overlap_all_sent"] == pytest.approx(3 / 8)
    assert row["adjacent_overlap_all_sent_div_seg"] == pytest.approx(1.5)
    assert row["adjacent_overlap_binary_all_sent"] == pytest.approx(0.5)
    # window two: one comparison, s1 vs s2+s3 -> 3 of 4 source types.
    assert row["adjacent_overlap_2_all_sent"] == pytest.approx(3 / 4)
    # paragraph overlap: P1 types {the,cat,chase,dog,see} (5) vs P2 -> 0.
    assert row["adjacent_overlap_all_para"] == pytest.approx(0.0)
    assert row["adjacent_overlap_2_all_para"] is None, "needs three paragraphs"
    # givenness: no pronouns; repeated content lemmas = cat,dog twice each ->
    # 4 repeated tokens out of 8 content tokens (we fixed the denominator here;
    # TAACO divided by ALL 12 words -- bug 4.13).
    assert row["pronoun_density"] == pytest.approx(0.0)
    assert row["pronoun_noun_ratio"] == pytest.approx(0.0)
    assert row["repeated_content_lemmas"] == pytest.approx(4 / 8)
    # bigram TTR within sentences: 4 + 4 + 1 = 9 grams; "the cat" and
    # "the dog" each occur twice, so 7 types.
    assert row["bigram_lemma_ttr"] == pytest.approx(7 / 9)


def test_mattr_by_hand():
    assert _mattr(list("aabb"), 2) == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert _mattr(list("ab"), 50) == pytest.approx(1.0), "short docs: plain TTR"
    assert _mattr([], 50) is None


# ---------------------------------------------------------------------------
# End to end: the TAACO-bug pins that need the real pipeline
# ---------------------------------------------------------------------------

def test_percent_signs_do_not_create_noun_cohesion(tmp_path):
    """nltk (like spaCy) tags % as NN. The real nouns of these two sentences
    are disjoint, so any noun overlap could only come from % -- TAACO
    reports exactly that; we must report zero."""
    rows, _ = _run(tmp_path, [
        ("pct", "Prices rose 50 % on Monday. Values fell 3 % on Tuesday."),
    ])
    assert float(rows["pct"]["adjacent_overlap_noun_sent"]) == 0.0
    assert float(rows["pct"]["adjacent_overlap_all_sent"]) > 0.0, (
        "'on' genuinely repeats, so the all-lemma index proves the "
        "sentences were compared at all"
    )


def test_undefined_indices_are_empty_cells_not_zeros(tmp_path):
    rows, _ = _run(tmp_path, [
        ("one_sent", "A single sentence lives here."),
        ("one_para", "First sentence here. Second sentence here."),
    ])
    r = rows["one_sent"]
    assert r["adjacent_overlap_all_sent"] == ""
    assert r["adjacent_overlap_all_para"] == ""
    assert r["lemma_ttr"] != "", "document-level indices still exist"
    r = rows["one_para"]
    assert r["adjacent_overlap_all_sent"] != ""
    assert r["adjacent_overlap_all_para"] == "", "one paragraph: NA, not 0.0"


def test_paragraphs_break_on_blank_lines_not_every_newline(tmp_path):
    """TAACO treats ANY newline as a paragraph boundary, so hard-wrapped text
    (every PDF extraction) turns each visual line into a 'paragraph'. Ours:
    blank-line runs only."""
    wrapped = "The cat sat quietly.\nThe cat slept deeply."       # one paragraph
    blank = "The cat sat quietly.\n\nThe cat slept deeply."       # two of them
    rows, _ = _run(tmp_path, [("wrapped", wrapped), ("blank", blank)])
    assert rows["wrapped"]["adjacent_overlap_all_para"] == ""
    assert rows["blank"]["adjacent_overlap_all_para"] != ""
    # and the same words still count at sentence level in both.
    assert rows["wrapped"]["adjacent_overlap_all_sent"] == \
        rows["blank"]["adjacent_overlap_all_sent"]


def test_header_shape_and_taaco_naming(tmp_path):
    rows, out = _run(tmp_path, [("d", "A cat sat. A dog ran.")])
    with open(out, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header[:2] == ["text_id", "nwords"]
    assert len(header) == 2 + len(header_columns())
    for name in ("lemma_ttr", "adjacent_overlap_cw_sent",
                 "adjacent_overlap_binary_2_argument_para",
                 "repeated_content_and_pronoun_lemmas"):
        assert name in header, f"TAACO column name {name} missing"


# ---------------------------------------------------------------------------
# Connectives
# ---------------------------------------------------------------------------

def test_connective_counting_by_hand():
    from taters.text.analyze_cohesion import count_connectives

    sents = [
        [("so", "IN"), ("so", "IN"), ("yet", "RB"), ("yet", "CC")],
        [("as", "IN"), ("a", "DT"), ("result", "NN")],
        [("ends", "VBZ"), ("in", "IN")],
        [("fact", "NN"), ("is", "VBZ")],
    ]
    # adjacent repeats each count: TAACO's `" so ".count()` on the joined
    # document came back with 1 for "so so" (bug 4.11, half of it).
    assert count_connectives(sents, [(("so",), None)]) == 2
    # we use a tag constraint instead of TAACO's dependency caveats: "not yet"
    # (RB) stays out of opposition, the conjunction "yet" (CC) counts.
    assert count_connectives(sents, [(("yet",), frozenset({"CC"}))]) == 1
    # phrases match within a sentence...
    assert count_connectives(sents, [(("as", "a", "result"), None)]) == 1
    # ...and never across sentence boundaries: TAACO matched "in fact" inside
    # "...ends in. Fact is..." (bug 4.11, the other half of it).
    assert count_connectives(sents, [(("in", "fact"), None)]) == 0


def test_shipped_lists_load_and_order_canonically():
    from taters.text.analyze_cohesion import (_CONNECTIVE_ORDER,
                                              load_connective_lists)

    lists = load_connective_lists()
    names = [n for n, _ in lists]
    assert names == list(_CONNECTIVE_ORDER), "shipped set, TAACO's order"
    assert all(entries for _n, entries in lists)
    # make sure the constraint syntax parsed: opposition should carry yet/CC.
    opposition = dict(lists)["opposition"]
    assert ((("yet",), frozenset({"CC"}))) in opposition


def test_connective_columns_flow_end_to_end(tmp_path):
    rows, out = _run(tmp_path, [
        ("d", "The plan failed because the funding stopped. "
              "However, the team continued and eventually succeeded."),
    ])
    r = rows["d"]
    assert float(r["all_causal"]) > 0, "'because' is causal"
    assert float(r["opposition"]) > 0, "'however' is oppositional"
    assert float(r["all_connective"]) >= float(r["basic_connectives"])
    # this is incidence, not a count, so it's bounded by 1 per word.
    assert 0 < float(r["all_connective"]) < 1


def test_a_custom_connectives_list_becomes_a_column(tmp_path):
    custom = tmp_path / "hedges.txt"
    custom.write_text("# my own category\nmaybe\nperhaps\nsort of\n",
                      encoding="utf-8")
    rows, out = _run(tmp_path,
                     [("d", "Maybe it works. Perhaps it is sort of fine.")],
                     connective_lists=[custom])
    r = rows["d"]
    assert "hedges" in r
    assert float(r["hedges"]) == pytest.approx(3 / 9, abs=1e-3)
    assert "all_causal" not in r, (
        "an explicit list selection replaces the shipped set, same as "
        "dictionaries"
    )


def test_demonstrative_columns_by_hand(tmp_path):
    rows, _ = _run(tmp_path, [("d", "That car is fast. I like that.")])
    r = rows["d"]
    # 7 counted words: that,car,is,fast + i,like,that.
    assert float(r["all_demonstratives"]) == pytest.approx(2 / 7, abs=1e-3)
    assert float(r["attended_demonstratives"]) == pytest.approx(1 / 7, abs=1e-3)
    assert float(r["unattended_demonstratives"]) == pytest.approx(1 / 7, abs=1e-3)


def test_connectives_is_a_library_kind():
    from taters.helpers.library import kind_by_id

    kind = kind_by_id("connectives")
    assert kind.suffixes == (".txt",)


# ---------------------------------------------------------------------------
# Synonym overlap (WordNet)
# ---------------------------------------------------------------------------

def test_synonym_overlap_by_hand():
    from taters.text.analyze_cohesion import synonym_overlap

    # car/automobile share a WordNet synset; dog/cat don't.
    count, prop = synonym_overlap([["car"], ["automobile"]], "n")
    assert count == 1.0 and prop == 1.0
    count, prop = synonym_overlap([["dog"], ["cat"]], "n")
    assert count == 0.0 and prop == 0.0
    # TAACO's count double-dips: two synonymous targets = 2 hits for one
    # source type. the `_prop` companion stays bounded at 1.
    count, prop = synonym_overlap([["car"], ["automobile", "auto"]], "n")
    assert count == 2.0 and prop == 1.0
    # an out-of-vocabulary lemma is its own synonym set (TAACO's rule), so
    # exact repetition still overlaps.
    count, prop = synonym_overlap([["flibbertigib"], ["flibbertigib"]], "n")
    assert count == 1.0 and prop == 1.0
    # too few segments: we want NA here, not zero.
    assert synonym_overlap([["car"]], "n") == (None, None)


def test_synonym_columns_flow_end_to_end(tmp_path):
    rows, _ = _run(tmp_path, [
        ("syn", "The car stopped abruptly. The automobile started again."),
    ])
    r = rows["syn"]
    assert float(r["syn_overlap_sent_noun"]) > 0
    assert 0 < float(r["syn_overlap_sent_noun_prop"]) <= 1
    assert r["syn_overlap_para_noun"] == "", "one paragraph: NA"


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def test_adjacent_semantic_by_hand():
    import numpy as np

    from taters.text.analyze_cohesion import adjacent_semantic

    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sim1, sim2 = adjacent_semantic(e)
    # pairs: (0,1)=1.0, (1,2)=0.0 -> mean 0.5; window two: e0 vs the
    # normalized mean of e1+e2 = [1,1]/sqrt(2) -> cos = 1/sqrt(2).
    assert sim1 == pytest.approx(0.5)
    assert sim2 == pytest.approx(1 / 2 ** 0.5)
    assert adjacent_semantic(np.ones((1, 2))) == (None, None)
    sim1, sim2 = adjacent_semantic(np.ones((2, 2)) / 2 ** 0.5)
    assert sim1 == pytest.approx(1.0) and sim2 is None


class _FakeEncoder:
    """Encodes each text to a normalized 2-d vector by its first word, so
    similarity is 1 for same-first-word texts and 0 otherwise -- enough to
    prove the wiring without downloading a model (house convention: fast
    tests use fakes; real models are @slow)."""

    def encode(self, texts, **_kw):
        import numpy as np

        return np.array([[1.0, 0.0] if t.split()[0] == "the" else [0.0, 1.0]
                         for t in texts])


def test_semantic_columns_splice_into_position(tmp_path, monkeypatch):
    class _FakeST:
        def __init__(self, *a, **k):
            self.inner = _FakeEncoder()

        def encode(self, *a, **k):
            return self.inner.encode(*a, **k)

    import sys
    import types

    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeST)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    rows, out = _run(tmp_path, [
        ("d", "The cat sat. The dog ran. A bird sang."),
    ], semantic_model="fake-model")
    r = rows["d"]
    # sentences: the/the/a -> sims (1, 0) -> mean 0.5
    assert float(r["semantic_1_all_sent"]) == pytest.approx(0.5, abs=1e-3)
    assert r["semantic_1_all_para"] == "", "one paragraph: NA"
    # and the splice landed in the right columns: the neighbors keep their
    # meanings (a misplaced splice would shift every later column by four).
    assert 0 <= float(r["basic_connectives"]) < 1
    assert r["syn_overlap_para_noun"] == ""

    # with semantic_model="none" the columns shouldn't exist at all.
    rows2, out2 = _run(tmp_path, [("d", "A cat. A dog.")],
                       semantic_model="none")
    assert "semantic_1_all_sent" not in rows2["d"]


# ---------------------------------------------------------------------------
# The measures guide
# ---------------------------------------------------------------------------

def test_every_column_is_documented_in_the_guide():
    """The near-parity decision came with a promise: a comprehensive guide.
    This is the invariant that keeps it comprehensive -- every column the
    analyzer can write must appear in COHESION_MEASURES.md (family patterns
    count: adjacent_overlap_* columns are documented as a scheme)."""
    from taters.text.analyze_cohesion import header_columns, measures_guide

    guide = measures_guide()
    schemes = ("adjacent_overlap", "syn_overlap", "semantic_",
               "_ttr", "_mattr")
    missing = []
    for col in header_columns(semantic=True):
        if col in guide:
            continue
        if any(s in col for s in schemes):
            continue
        missing.append(col)
    assert not missing, f"undocumented columns: {missing}"
    # and the scheme sections actually exist.
    for anchor in ("adjacent_overlap_<class>_<unit>", "syn_overlap_<unit>",
                   "semantic_1_all_sent", "lemma_mattr"):
        assert anchor in guide, f"guide lost its {anchor} section"


def test_the_cli_explains_itself(capsys):
    from taters.text.analyze_cohesion import main

    main(["--explain"])
    out = capsys.readouterr().out
    assert "cohesion" in out.lower() and "adjacent_overlap" in out


def test_the_stanza_model_load_announces_itself(monkeypatch):
    """While the stanza model downloads (hundreds of MB), the step used to
    keep saying "reading the input" -- the phase must name what is actually
    happening, and it must land BEFORE the build starts."""
    import taters.text.analyze_cohesion as coh

    events = []

    def sink(done, total, message=None, unit=None, **_kw):
        events.append(message)

    def fake_stream(**_kw):
        return lambda text: []

    monkeypatch.setattr(coh, "make_sentence_stream", fake_stream)
    import csv as _csv
    import tempfile
    from pathlib import Path as _P
    d = _P(tempfile.mkdtemp())
    src = d / "ready.csv"
    with src.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["text_id", "text"])
        w.writerow(["a", "hello there"])
        w.writerow(["b", "general kenobi"])

    coh.analyze_cohesion(analysis_csv=src, out_features_csv=d / "o.csv",
                         engine="stanza", semantic_model="none",
                         overwrite_existing=True, on_progress=sink)
    assert any(e and "stanza" in e for e in events), (
        "no phase named the model load"
    )
    stanza_idx = next(i for i, e in enumerate(events) if e and "stanza" in e)
    measuring_idx = next(i for i, e in enumerate(events)
                         if e == "measuring cohesion")
    assert stanza_idx < measuring_idx


def test_cohesion_asks_for_the_wordnet_data_before_it_uses_wordnet():
    """
    `_adverb_is_content` decides whether an `-ly` adverb came from an
    adjective by asking WordNet. On a machine that has never downloaded the
    corpus that lookup raises LookupError, the `except Exception` around it
    swallows the error, and **every deadjectival adverb silently becomes a
    function word** -- wrong cohesion numbers, no warning, no failure. CI found
    it; no local run could, because a developer machine has the corpus.

    So the module has to ask for the data first, the way `nltk_data`'s own
    docstring says anything needing WordNet should. This asserts the ask
    happens, which is the part that cannot be observed from the output.
    """
    from taters.text import analyze_cohesion as ac

    asked = []
    real = ac._wordnet_ready
    ac._WORDNET.clear()
    try:
        ac._wordnet_ready = lambda ensure: asked.append(ensure) or True
        ac.classify_sentence([("quickly", "quickly", "RB")])
    finally:
        ac._wordnet_ready = real
        ac._WORDNET.clear()

    assert asked, "the WordNet data was used without ever being ensured"
    assert asked[0].__name__ == "ensure_wordnet", asked[0]


def test_a_missing_wordnet_does_not_pretend_every_adverb_is_a_function_word():
    """
    The fallback still has to be safe -- we cannot download on a machine with
    no network -- but it must be reached by *deciding*, not by an exception
    escaping from a lookup nobody guarded.
    """
    from taters.text import analyze_cohesion as ac

    real = ac._wordnet_ready
    ac._WORDNET.clear()
    try:
        ac._wordnet_ready = lambda ensure: False       # the corpus is not there
        classes = ac.classify_sentence([("quickly", "quickly", "RB"),
                                        ("cats", "cat", "NNS")])
    finally:
        ac._wordnet_ready = real
        ac._WORDNET.clear()

    assert classes["adv"] == ["quickly"], "it is still an adverb"
    assert "quickly" not in classes["cw"], "without WordNet we cannot claim it is content"
    assert classes["noun"] == ["cat"], "the rest of the sentence is unaffected"
