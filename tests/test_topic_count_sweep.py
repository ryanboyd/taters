"""
Choosing a topic count: the sweep, the curve, and the caveat it ships with.

The interesting test here is the one that plants a known answer. Four disjoint
word families, forty documents each drawn from one of them -- so the right
number of topics is four, and a sweep that cannot find four on a corpus built
this cleanly cannot be trusted on a real one.

The rest is about not misleading anybody: the vocabulary has to be shared
across fits (otherwise the scores compare nothing), and the report has to say
out loud that coherence is a guide rather than an answer.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from taters.text.topic_count_sweep import sweep_topic_count

def matrix_of(out) -> Path:
    """Where a step put the matrix it built for itself.

    Derived from the output's *stem*, not a fixed `matrix/` beside it: two
    steps writing into one folder would otherwise share one matrix folder,
    which is how three topic models came to build over each other's and how an
    apply left a frozen vocabulary for a later fit to pick up.
    """
    out = Path(out)
    return out.with_name(f"{out.stem}_matrix")


#: Four families of ten, so a forty-term vocabulary. Ten rather than six for a
#: concrete reason: coherence is scored over each topic's top words, and at six
#: a sweep up to eight topics cannot give every topic four distinct ones -- the
#: lists overlap and every score measures words from different topics. The
#: module warns about exactly that, and a fixture that tripped the warning would
#: be testing the warning rather than the sweep.
FAMILIES = [
    "bread butter cheese dinner kitchen recipe pasta soup salad pudding".split(),
    "office meeting deadline manager project report invoice budget memo client".split(),
    "river forest mountain valley glacier meadow canyon prairie tundra lagoon".split(),
    "guitar drums piano violin trumpet cello banjo oboe harp mandolin".split(),
]

LOOSE = dict(min_freq=1, min_obs_pct=0.0, min_token_count=1,
             vocab_min_freq=0, vocab_min_obs_pct=0)


@pytest.fixture
def four_families(tmp_path) -> Path:
    """Eighty documents, each drawn from exactly one of four word families."""
    rng = random.Random(0)
    path = tmp_path / "corpus.csv"
    rows = [("text_id", "text")]
    for i in range(80):
        family = FAMILIES[i % len(FAMILIES)]
        rows.append((f"d{i:02d}", " ".join(rng.choice(family) for _ in range(25))))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def sweep(corpus, tmp_path, **kwargs):
    defaults = dict(csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
                    k_values="2,4,8", passes=6, top_terms=4,
                    out_csv=tmp_path / "sweep.csv", overwrite_existing=True,
                    **LOOSE)
    defaults.update(kwargs)
    return Path(sweep_topic_count(**defaults))


def scores(path) -> dict:
    return {int(r["n_topics"]): float(r["coherence"])
            for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


# ---------------------------------------------------------------------------
# Does it find a planted answer?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["lda", "nmf"])
def test_the_planted_number_of_topics_wins(engine, four_families, tmp_path):
    """Constructed so the answer is not a matter of taste: four disjoint word
    families, so four topics is right and two and eight are both wrong."""
    got = scores(sweep(four_families, tmp_path, engine=engine))
    assert max(got, key=got.get) == 4, got


def test_the_lda_curve_is_a_peak_and_not_a_slope(four_families, tmp_path):
    """
    Four has to win by enough to see. LDA under-fitting is severely punished by
    coherence -- merging two families into one topic gives a topic whose top
    words genuinely do not co-occur -- so the curve on this corpus runs roughly
    -0.02, 0.95, 0.81 and the peak is unmistakable.
    """
    got = scores(sweep(four_families, tmp_path, engine="lda"))
    assert got[2] < got[4] > got[8], got
    assert got[4] - got[2] > 0.3, got


def test_the_nmf_curve_is_much_flatter_and_that_is_worth_knowing(four_families,
                                                                 tmp_path):
    """
    Not a defect, and a real limit on how much weight a sweep can carry.

    NMF at too few topics still produces topics whose *top* words come from one
    family -- it merges families in the tail, not the head -- so coherence
    barely notices. On this corpus it runs about 0.931, 0.939, 0.930: four still
    wins, by less than a hundredth. Anybody reading an NMF sweep should treat a
    flat curve as "coherence cannot separate these" rather than as a verdict,
    which is why the report says to read the words.
    """
    got = scores(sweep(four_families, tmp_path, engine="nmf"))
    assert max(got, key=got.get) == 4, got
    assert got[4] - min(got[2], got[8]) < 0.1, (
        "the NMF curve came out sharply peaked; if that is now true the "
        "docstring above is stale and should be rewritten", got)


# ---------------------------------------------------------------------------
# What it writes
# ---------------------------------------------------------------------------

def test_it_writes_the_numbers_the_curve_and_the_words(four_families, tmp_path):
    out = sweep(four_families, tmp_path)
    assert out.is_file()
    assert out.with_name("sweep_coherence.png").is_file()

    report = out.with_name("sweep_report.md").read_text(encoding="utf-8-sig")
    assert "4 topics" in report

    # every family should be visible somewhere in the listed topics -- but not
    # any *particular* word of it. a topic's top four are whichever four the fit
    # weighted highest, so requiring a chosen representative ("guitar") fails on
    # a run that picked banjo, harp, trumpet and drums instead, which is the
    # same family recovered just as well.
    for family in FAMILIES:
        assert any(word in report for word in family), (
            f"no topic's words mention any of {family[:4]}...")


def test_the_report_says_coherence_is_not_the_answer(four_families, tmp_path):
    """
    The number is seductive and it is not a verdict -- it rewards topics whose
    words co-occur, which is related to but not the same as topics worth
    interpreting. A report that just announced a winner would be inviting
    somebody to skip the only step that matters.
    """
    out = sweep(four_families, tmp_path)
    report = out.with_name("sweep_report.md").read_text(encoding="utf-8-sig").lower()
    assert "read the words" in report
    assert "a guide, not a verdict" in report


def test_every_fit_is_scored_over_one_shared_vocabulary(four_families, tmp_path):
    """
    Rebuilding the matrix per topic count would mean each coherence was
    computed over a different word list, and comparing those numbers would
    compare nothing. One matrix, built once, beside the results.
    """
    out = sweep(four_families, tmp_path)
    matrices = list(matrix_of(out).glob("dtm*.csv"))
    assert len(matrices) == 1, matrices


def test_the_matrix_weighting_follows_the_engine(four_families, tmp_path):
    """
    The sweep fits through `_topics` directly, so it never passes through
    `topic_model_lda`'s refusal of anything but counts -- it makes the same
    choice itself, on one line. That line therefore needs its own test, or LDA
    would quietly sweep over a tf-idf matrix and report coherence for a model
    that is not defined over one.

    Checked by looking at the matrix rather than at the setting: counts are
    whole numbers, tf-idf cells are not.
    """
    for engine, whole_numbers in (("lda", True), ("nmf", False)):
        out = sweep(four_families, tmp_path, engine=engine, k_values="4",
                    out_csv=tmp_path / f"{engine}.csv")
        rows = list(csv.reader(
            open(matrix_of(out) / "dtm.csv", encoding="utf-8-sig")))
        cells = [float(c) for row in rows[1:6] for c in row[2:] if c]
        assert cells, "the matrix came out empty"
        assert all(c == int(c) for c in cells) is whole_numbers, (
            f"{engine} got the wrong kind of matrix", cells[:8])


# ---------------------------------------------------------------------------
# What it refuses, and what it warns about
# ---------------------------------------------------------------------------

def test_asking_for_more_words_than_the_vocabulary_can_give_warns(four_families,
                                                                  tmp_path):
    """
    The trap that caught me while building this. Coherence is computed over
    each topic's top `top_terms` words; if the vocabulary cannot give every
    topic that many *distinct* words, the lists spill into each other and every
    score is measuring words from different topics against each other. Every
    count then scores badly and the winner is arbitrary -- confident nonsense,
    with no error anywhere.
    """
    with pytest.warns(UserWarning, match="different topics"):
        sweep(four_families, tmp_path, k_values="8", top_terms=10,
              vocab_top_n=20)


@pytest.mark.parametrize("engine", ["lda", "nmf"])
def test_a_topic_count_too_big_for_the_corpus_is_skipped_not_fatal(engine,
                                                                   four_families,
                                                                   tmp_path):
    """
    The one place an unfittable count is the question rather than a mistake.
    `fit_lda` refuses more topics than documents and NNDSVD refuses more than
    the matrix has dimensions; both fired inside the loop, so the default
    5,10,20,40 on a small corpus threw away the good fits *and* the run,
    after paying for the matrix. The sweep skips what it cannot fit, says so,
    and writes the rest.
    """
    with pytest.warns(UserWarning, match="skipped"):
        out = sweep(four_families, tmp_path, engine=engine,
                    k_values="2,4,500", vocab_top_n=40)
    assert sorted(scores(out)) == [2, 4]
    report = next(tmp_path.glob("*report*.md")).read_text(encoding="utf-8")
    assert "500" in report and "skipped" in report


def test_a_sweep_with_nothing_it_can_fit_says_what_the_corpus_can_support(
        four_families, tmp_path):
    """Skipping everything is not a silent empty run: it names the ceiling."""
    with pytest.raises(ValueError, match="the most this corpus can support"):
        sweep(four_families, tmp_path, k_values="500,900")


@pytest.mark.parametrize("bad,message", [
    ("1,2", "not a topic model"),
    ("", "no topic counts"),
    ("five", "not a number"),
])
def test_a_topic_count_that_is_not_one_is_refused_by_name(bad, message,
                                                          four_families,
                                                          tmp_path):
    with pytest.raises(ValueError, match=message):
        sweep(four_families, tmp_path, k_values=bad)


def test_counts_can_be_written_as_a_list_or_a_string(four_families, tmp_path):
    """A pipeline file gives a list; a command line gives a string."""
    as_string = scores(sweep(four_families, tmp_path, k_values="2,4"))
    as_list = scores(sweep(four_families, tmp_path, k_values=[4, 2],
                           out_csv=tmp_path / "b.csv"))
    assert as_string == as_list


def test_an_unknown_engine_is_refused():
    with pytest.raises(ValueError, match="unknown engine"):
        sweep_topic_count(analysis_csv="x.csv", engine="mallet")


def test_c_v_coherence_is_refused_with_a_reason():
    """It is the metric people ask for by name, and it is the one that would
    cost a dependency -- so the refusal explains rather than just listing."""
    with pytest.raises(ValueError, match="sliding windows"):
        sweep_topic_count(analysis_csv="x.csv", metric="c_v")
