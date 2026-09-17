"""
The arithmetic behind LDA and NMF, tested on matrices with a known answer.

No corpus, no CSV, no tokenizer: `text/_topics.py` is deliberately the numbers
on their own, so a failure here means the mathematics is wrong rather than that
some plumbing upstream of it is. The file-level behavior is tested separately.

On what can and cannot be pinned
--------------------------------
PCA has one right answer and `test_stats_pca.py` checks it against scikit-learn
to machine precision. **Neither of these methods is like that.** LDA's posterior
has no closed form and variational Bayes finds a local optimum that depends on
the initialization; NMF's factorization is not unique -- swap two topics, or
scale W up and H down, and it is the same model. Pinning digits would pin an
accident.

So these test the properties that actually have to hold:

* planted topics come back out, which is the test a person would run by hand;
* the objective moves the right way, because a model that is not learning is
  the failure that looks most like success;
* inference with the topics held fixed is the *same code* the fit uses, which
  is what makes a saved model reproduce its training scores exactly;
* the same input and seed give the same answer, twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from taters.text import _topics as topics

V_PER_FAMILY = 6
VOCAB = [f"w{i}" for i in range(2 * V_PER_FAMILY)]
FAMILY_A = range(0, V_PER_FAMILY)
FAMILY_B = range(V_PER_FAMILY, 2 * V_PER_FAMILY)


def planted(n_docs: int = 60, leak: int = 1, seed: int = 0):
    """Documents drawn from one of two disjoint word families.

    `leak` puts a few words from anywhere into every document, so the families
    are strongly separated but not perfectly -- a matrix with literal zero
    overlap is easier than anything real and hides division-by-zero bugs.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_docs, len(VOCAB)), dtype=float)
    for d in range(n_docs):
        family = FAMILY_A if d % 2 == 0 else FAMILY_B
        for _ in range(40):
            counts[d, rng.choice(list(family))] += 1
        for _ in range(leak):
            counts[d, rng.integers(0, len(VOCAB))] += 1
    return counts


def batches_of(counts, size: int = 16):
    """`fit_lda` wants a callable so it can walk the corpus more than once."""
    return lambda: (counts[i:i + size] for i in range(0, len(counts), size))


def recovered_families(components) -> set:
    """Which planted family each topic's heaviest terms belong to."""
    found = set()
    for terms in topics.top_terms(components, VOCAB, n=4):
        indices = [VOCAB.index(t) for t in terms]
        found.add("A" if all(i in FAMILY_A for i in indices)
                  else "B" if all(i in FAMILY_B for i in indices) else "mixed")
    return found


def _best_match(ours, theirs) -> float:
    """Weakest correlation once each of our topics is paired with its closest
    of theirs. Topic order is arbitrary in both, so a positional comparison
    would fail on a relabeling that changed nothing."""
    ours = ours / ours.sum(axis=1, keepdims=True)
    theirs = theirs / theirs.sum(axis=1, keepdims=True)
    worst = 1.0
    for row in ours:
        worst = min(worst, max(float(np.corrcoef(row, other)[0, 1])
                               for other in theirs))
    return worst


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------

def test_lda_recovers_two_planted_families():
    """The test a person would run by hand, and the one that matters most."""
    counts = planted()
    lam, _trace = topics.fit_lda(batches_of(counts), len(VOCAB), 2,
                                 passes=12, seed=7)
    assert recovered_families(lam) == {"A", "B"}


def test_lda_perplexity_falls():
    """A model that is not learning is the failure that looks most like
    success: it writes every file, reports no error, and the numbers are of
    nothing."""
    counts = planted()
    _lam, trace = topics.fit_lda(batches_of(counts), len(VOCAB), 2,
                                 passes=10, seed=7)
    assert trace[-1] < trace[0], trace


def test_a_document_is_assigned_to_the_family_it_was_drawn_from():
    counts = planted()
    lam, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=12, seed=7)
    theta = topics.infer_lda(lam, counts[:2])

    assert np.allclose(theta.sum(axis=1), 1.0), "proportions must sum to 1"
    # documents 0 and 1 come from different families, so their heaviest topic
    # must differ -- whichever way round the topics happened to come out
    assert int(np.argmax(theta[0])) != int(np.argmax(theta[1]))
    assert theta[0].max() > 0.8 and theta[1].max() > 0.8


def test_the_same_seed_gives_the_same_model_twice():
    """Reproducibility is the argument for hand-rolling this rather than taking
    a dependency whose threading decides the answer."""
    counts = planted()
    first, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=5, seed=3)
    again, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=5, seed=3)
    assert np.array_equal(first, again)


def test_a_different_seed_is_allowed_to_give_a_different_model():
    """The counterpart: if the seed changed nothing, it would not be doing its
    job and the reproducibility claim above would be vacuous."""
    counts = planted()
    first, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=2, seed=3)
    other, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=2, seed=99)
    assert not np.array_equal(first, other)


def test_the_batch_size_does_not_decide_the_answer():
    """Batching is a memory strategy, not a modeling choice. Different batch
    sizes are different update schedules, so the models are not identical --
    but they must find the same topics."""
    counts = planted()
    small, _ = topics.fit_lda(batches_of(counts, 8), len(VOCAB), 2, passes=12, seed=7)
    large, _ = topics.fit_lda(batches_of(counts, 64), len(VOCAB), 2, passes=12, seed=7)
    assert recovered_families(small) == recovered_families(large) == {"A", "B"}

    # and not merely "the same two families" -- the distributions themselves
    # have to line up. this is what notices if a batch's statistics stop being
    # scaled up to the whole corpus before they are mixed in: without that, a
    # run of many small batches weights the prior far more heavily than a run
    # of one big one, and the two quietly become different models.
    assert _best_match(small, large) > 0.99, "batch size changed the model"


def test_the_fitted_topics_account_for_every_word_in_the_corpus():
    """
    The topic-word matrix is `eta` plus the sufficient statistics, and those
    statistics are word counts redistributed over topics -- so their total has
    to come back to the number of words that were actually there.

    This is what pins the one line that scales a batch's statistics up to the
    whole corpus before mixing them in. Drop it and the topics still come out
    the same *shape* -- same words, same order, correlating at 0.99 with the
    right answer -- but the model believes it saw one batch's worth of evidence
    instead of a corpus's. Every test here that normalizes away magnitude sails
    straight past that, which is exactly why this one does not.
    """
    counts = planted()
    lam, _ = topics.fit_lda(batches_of(counts, 16), len(VOCAB), 2,
                            passes=12, seed=7)
    assert lam.sum() == pytest.approx(counts.sum(), rel=0.02), (
        f"the model accounts for {lam.sum():.0f} words out of {counts.sum():.0f}")


def test_handing_inference_the_hoisted_constant_changes_nothing():
    """
    `exp_elogbeta` depends only on lambda, and computing it is a `psi` and an
    `exp` over the whole k-by-V matrix -- while the E-step below it only
    touches a document's non-zero columns. So scoring a corpus recomputed the
    expensive part once a document and the cheap part once a document: 4.2 ms
    against 0.8 ms at k=20 over 2,000 terms, and worse as the vocabulary grows.

    Callers can now pass it in. It is the same array either way, so the two
    paths have to agree exactly -- not approximately, since nothing about this
    is a different computation.
    """
    counts = planted()
    lam, _trace = topics.fit_lda(batches_of(counts), len(VOCAB), 2,
                                 passes=6, seed=7)
    hoisted = np.exp(topics.dirichlet_expectation(np.asarray(lam, float)))
    for row in counts[:6]:
        recomputed = topics.infer_lda(lam, row)
        passed_in = topics.infer_lda(lam, row, exp_elogbeta=hoisted)
        assert np.array_equal(recomputed, passed_in)


def test_a_document_with_none_of_the_vocabulary_scores_without_exploding():
    """Real corpora have them: a row whose every word was filtered out. It has
    to come back as the prior rather than as NaN, and it must not contribute
    statistics for words that were never there."""
    counts = planted()
    lam, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=5, seed=7)

    empty = np.zeros((1, len(VOCAB)))
    theta = topics.infer_lda(lam, empty)
    assert np.all(np.isfinite(theta))
    assert theta[0] == pytest.approx([0.5, 0.5], abs=1e-9), (
        "with no evidence, an even split is the only honest answer")


def test_fewer_documents_than_topics_is_refused():
    with pytest.raises(ValueError, match="more documents than topics"):
        topics.fit_lda(batches_of(planted(3)), len(VOCAB), 8, passes=2)


def test_one_topic_is_refused():
    with pytest.raises(ValueError, match="at least 2 topics"):
        topics.fit_lda(batches_of(planted()), len(VOCAB), 1)


def test_a_learning_rate_that_would_not_converge_is_refused():
    """kappa outside (0.5, 1] breaks the convergence guarantee in Hoffman et
    al. 2010 -- it would still run, and the result would mean nothing."""
    with pytest.raises(ValueError, match="kappa"):
        topics.fit_lda(batches_of(planted()), len(VOCAB), 2, kappa=0.3)


def test_held_out_perplexity_prefers_the_model_that_saw_the_data():
    """A sanity check on `perplexity` itself: a fitted model has to be less
    surprised by the corpus than an unfitted one."""
    counts = planted()
    lam, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=12, seed=7)
    untrained = np.full_like(lam, 1.0)
    assert topics.perplexity(lam, counts) < topics.perplexity(untrained, counts)


# ---------------------------------------------------------------------------
# Choosing k
# ---------------------------------------------------------------------------

FOUR = [
    "bread butter cheese dinner kitchen recipe pasta soup salad pudding".split(),
    "office meeting deadline manager project report invoice budget memo client".split(),
    "river forest mountain valley glacier meadow canyon prairie tundra lagoon".split(),
    "guitar drums piano violin trumpet cello banjo oboe harp mandolin".split(),
]


def four_families(n_docs: int = 80, seed: int = 0):
    """Four disjoint families, so the planted answer is four and both two and
    eight are wrong -- in different ways, which is the point."""
    import random

    terms = [w for family in FOUR for w in family]
    where = {w: i for i, w in enumerate(terms)}
    rng = random.Random(seed)
    counts = np.zeros((n_docs, len(terms)))
    for d in range(n_docs):
        for _ in range(25):
            counts[d, where[rng.choice(FOUR[d % len(FOUR)])]] += 1
    return counts, terms


def test_exclusivity_is_one_when_topics_share_nothing():
    """A word's share of its own mass that sits in this topic. Disjoint
    topics own their words outright; overlapping ones do not."""
    apart = topics.exclusivity(np.array([[1.0, 1.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 1.0]]))
    assert np.allclose(apart[0][:2], 1.0)
    assert np.allclose(apart[0][2:], 0.0)

    shared = topics.exclusivity(np.array([[1.0, 1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0, 1.0]]))
    assert np.allclose(shared, 0.5)
    assert shared.min() >= 0.0 and shared.max() <= 1.0


def test_exclusivity_ignores_how_big_a_factor_happens_to_be():
    """NMF's factor scale is arbitrary -- W absorbs it -- so a factor with a
    large H must not look more exclusive for it. Rows are normalized before
    columns, which is what makes this true."""
    plain = topics.exclusivity(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]))
    scaled = topics.exclusivity(np.array([[50.0, 50.0, 0.0], [0.0, 1.0, 1.0]]))
    assert np.allclose(plain, scaled)


def test_the_balance_is_harmonic_so_one_score_cannot_cover_for_the_other():
    """
    The difference between a harmonic mean and an average, on the case that
    separates them. Perfect coherence with hopeless exclusivity is a bad
    model; middling on both is a real one. An average ranks the lopsided one
    *higher* (0.53 against 0.50); the harmonic mean ranks it far lower, which
    is the whole reason the pairing is not just added up.
    """
    lopsided = topics._balanced(1.0, 0.05)
    even = topics._balanced(0.0, 0.5)
    assert lopsided < even
    assert 0.0 <= lopsided <= 1.0 and 0.0 <= even <= 1.0


def test_the_balanced_rule_refuses_a_k_that_is_exclusive_but_incoherent():
    """
    The whole reason the two are paired, on real numbers. Fitting four
    planted families with two topics blends two families into each: the
    topics are still perfectly *disjoint* from each other, so exclusivity is
    about as high as it ever gets -- while coherence goes negative, because
    the words inside a topic genuinely do not co-occur.

    Exclusivity alone would take that model. The harmonic mean throws it out,
    which is what a harmonic mean is for: one term cannot cover for the
    other.
    """
    counts, terms = four_families()
    batches = batches_of(counts, 80)
    pair, single, n_docs = topics.codocument_counts(batches, len(terms))

    def fit(k):
        return topics.fit_lda(batches, len(terms), k, passes=10, seed=7)[0]

    rows, _words, best, _fit = topics.choose_k(
        counts=[2, 4, 8], fit=fit, terms=terms, pair=pair, single=single,
        n_docs=n_docs, rule="coherence_exclusivity", top_terms=4)
    by_k = {r["n_topics"]: r for r in rows}

    assert by_k[2]["exclusivity"] > by_k[4]["exclusivity"], \
        "the blended model should be the exclusive one -- the premise is gone"
    assert by_k[2]["coherence"] < 0 < by_k[4]["coherence"]
    assert by_k[2]["balanced"] < by_k[4]["balanced"]
    assert best == 4


@pytest.mark.parametrize("rule", ["coherence", "coherence_exclusivity"])
def test_both_rules_find_the_planted_number_of_topics(rule):
    counts, terms = four_families()
    batches = batches_of(counts, 80)
    pair, single, n_docs = topics.codocument_counts(batches, len(terms))

    def fit(k):
        return topics.fit_lda(batches, len(terms), k, passes=10, seed=7)[0]

    _rows, _words, best, components = topics.choose_k(
        counts=[2, 4, 8], fit=fit, terms=terms, pair=pair, single=single,
        n_docs=n_docs, rule=rule, top_terms=4)
    assert best == 4
    assert components.shape[0] == 4, "the winner's own fit should come back"


def test_the_winning_fit_comes_back_rather_than_being_thrown_away():
    """Refitting at the chosen count would be the same deterministic work
    twice, and at k=2000 that is not a rounding error."""
    counts, terms = four_families()
    batches = batches_of(counts, 80)
    pair, single, n_docs = topics.codocument_counts(batches, len(terms))
    seen = []

    def fit(k):
        seen.append(k)
        return topics.fit_lda(batches, len(terms), k, passes=6, seed=7)[0]

    _rows, _words, best, components = topics.choose_k(
        counts=[2, 4], fit=fit, terms=terms, pair=pair, single=single,
        n_docs=n_docs, top_terms=4)
    assert seen == [2, 4], "each count fitted exactly once"
    assert best == 4, "the wrong count won, so the fit handed back is the wrong one"
    again = topics.fit_lda(batches, len(terms), 4, passes=6, seed=7)[0]
    assert np.allclose(components, again), "that is not the winner's own fit"


def test_the_balanced_rule_refuses_umass_by_name():
    """UMass is unbounded below, so balancing it would mean rescaling against
    whichever counts are in the sweep -- and the winner would move when you
    added a candidate."""
    with pytest.raises(ValueError, match="needs metric='npmi'"):
        topics.choose_k(counts=[2], fit=lambda k: np.ones((k, 3)),
                        terms=["a", "b", "c"], pair=np.ones((3, 3)),
                        single=np.ones(3), n_docs=4,
                        rule="coherence_exclusivity", metric="umass")


def test_an_unknown_selection_rule_is_refused():
    with pytest.raises(ValueError, match="unknown k selection rule"):
        topics.choose_k(counts=[2], fit=lambda k: np.ones((k, 3)),
                        terms=["a", "b", "c"], pair=np.ones((3, 3)),
                        single=np.ones(3), n_docs=4, rule="perplexity")


def test_counts_a_corpus_cannot_support_are_separated_not_fatal():
    fittable, skipped = topics.fittable_counts([2, 5, 40], n_docs=12,
                                               n_terms=30, engine="lda")
    assert fittable == [2, 5] and skipped == [40]
    # NMF is bounded by the narrower of the two dimensions
    fittable, skipped = topics.fittable_counts([2, 5, 40], n_docs=100,
                                               n_terms=8, engine="nmf")
    assert fittable == [2, 5] and skipped == [40]


def test_topic_counts_can_be_written_as_a_list_or_a_string():
    assert topics.parse_counts("5, 10;20") == [5, 10, 20]
    assert topics.parse_counts([20, 5, 5]) == [5, 20]
    with pytest.raises(ValueError, match="not a number of topics"):
        topics.parse_counts("five")
    with pytest.raises(ValueError, match="not a topic model"):
        topics.parse_counts([1])


# ---------------------------------------------------------------------------
# NMF
# ---------------------------------------------------------------------------

def test_nmf_recovers_two_planted_families():
    counts = planted()
    _w, h, _err = topics.fit_nmf(counts, 2)
    assert recovered_families(h) == {"A", "B"}


def test_nmf_reconstruction_error_falls():
    counts = planted()
    _w, _h, err = topics.fit_nmf(counts, 2)
    assert err[-1] < err[0], err


def test_both_factors_actually_move_off_the_starting_point():
    """
    NNDSVD is a good enough start that it already separates a planted matrix
    on its own -- which means "the topics came out right" does not prove the
    multiplicative updates ran at all. Freezing H entirely still passed every
    other test in this file except the comparison against sklearn.

    So this asks the narrow question directly: did both factors move?
    """
    counts = planted()
    w0, h0 = topics.nndsvd(counts, 2)
    w, h, trace = topics.fit_nmf(counts, 2)

    assert not np.allclose(h, h0), "H never moved -- the topic update is dead"
    assert not np.allclose(w, w0), "W never moved -- the weight update is dead"
    assert trace[-1] < float(np.linalg.norm(counts - w0 @ h0)), (
        "the fit is no better than its own starting point")


def test_nmf_needs_no_seed_at_all():
    """NNDSVD is deterministic, so two runs on one corpus give one answer --
    no seed to record, and none to forget to record."""
    counts = planted()
    w1, h1, _ = topics.fit_nmf(counts, 2)
    w2, h2, _ = topics.fit_nmf(counts, 2)
    assert np.array_equal(w1, w2) and np.array_equal(h1, h2)


def test_both_divergences_find_the_families():
    counts = planted()
    for loss in ("frobenius", "kullback-leibler"):
        _w, h, _err = topics.fit_nmf(counts, 2, beta_loss=loss)
        assert recovered_families(h) == {"A", "B"}, loss


def test_a_negative_matrix_is_refused_by_name():
    """The non-negative part is not decoration. A standardized matrix has
    negative cells and would factor into nonsense without complaint."""
    with pytest.raises(ValueError, match="non-negative"):
        topics.fit_nmf(planted() - 1.0, 2)


def test_an_unknown_divergence_is_refused():
    with pytest.raises(ValueError, match="beta_loss"):
        topics.fit_nmf(planted(), 2, beta_loss="cosine")


def test_more_topics_than_the_matrix_can_hold_is_refused():
    with pytest.raises(ValueError, match="not that many dimensions"):
        topics.nndsvd(planted(4), 8)


def test_nndsvd_leaves_no_zeros_for_the_updates_to_get_stuck_on():
    """Zero is absorbing under multiplicative updates: a cell that starts at
    zero can never leave it, so the factorization would be permanently missing
    whatever NNDSVD happened to leave empty."""
    w, h = topics.nndsvd(planted(), 3)
    assert np.all(w > 0) and np.all(h > 0)


def test_applying_nmf_leaves_the_topics_alone():
    """Only W moves. Updating H here would be refitting to the new corpus,
    which is exactly what makes two studies incomparable."""
    counts = planted()
    _w, h, _ = topics.fit_nmf(counts, 2)
    before = h.copy()
    topics.infer_nmf(h, counts[:5])
    assert np.array_equal(h, before)


def test_inference_settles_against_the_topics_it_was_handed():
    """
    "Only W moves" is easy to assert and hard to actually check: every test
    that calls `infer_nmf` to work out the expected answer will drift in step
    with any bug inside it, and a version that quietly refits H per document
    passed the roundtrip, the subset comparison and the recompute-from-the-JSON
    test alike.

    This asks something those cannot: is the returned W a *fixed point for the
    H that was passed in*? One more multiplicative update against that same H
    must leave it where it is. A W that had settled against some internally
    drifted H would move here.
    """
    counts = planted()
    _w, h, _ = topics.fit_nmf(counts, 2)
    x = counts[:5]

    # asked to converge properly: the default tolerance stops when the
    # reconstruction settles, which leaves the weights still drifting in the
    # third decimal. that is fine for scoring and not fine for asserting a
    # fixed point, so this pays for the extra iterations.
    w = topics.infer_nmf(h, x, iters=5000, tol=1e-12)
    once_more = w * ((x @ h.T) / (w @ (h @ h.T) + 1e-100))

    # judged against the size of the solution rather than per element, because
    # inference stops when the *reconstruction* settles: a weight already down
    # near zero keeps wandering by large fractions of itself without moving the
    # error at all, and that is convergence working, not failing.
    assert np.allclose(once_more, w, atol=1e-3 * float(w.max()), rtol=0), (
        "inference did not settle against the topics it was given")


def test_inference_uses_the_divergence_the_model_was_fitted_with():
    """
    Found in review. `fit_nmf` honors `beta_loss`, and inference used to be
    least squares whatever the model said -- so a KL-fitted model's documents
    were scored with the Frobenius update, giving weights the fit never
    produced and which are optimal under a divergence nobody chose.

    The two updates have to disagree for that to matter, and they do.
    """
    counts = planted()
    _w, h, _ = topics.fit_nmf(counts, 2, beta_loss="kullback-leibler")
    x = counts[:6]

    kl = topics.infer_nmf(h, x, beta_loss="kullback-leibler")
    frobenius = topics.infer_nmf(h, x, beta_loss="frobenius")
    assert not np.allclose(kl, frobenius, rtol=0.05), (
        "the two divergences gave the same weights, so this test cannot tell "
        "whether the right one is being used")

    # and the KL answer settles against its own update, the way the fit's does
    wh = kl @ h + 1e-100
    once_more = kl * (((x / wh) @ h.T) / (h.sum(axis=1)[None, :] + 1e-100))
    assert np.allclose(once_more, kl, atol=1e-2 * float(kl.max()), rtol=0)


def test_an_unknown_divergence_is_refused_by_inference_too():
    counts = planted()
    _w, h, _ = topics.fit_nmf(counts, 2)
    with pytest.raises(ValueError, match="beta_loss"):
        topics.infer_nmf(h, counts[:3], beta_loss="cosine")


def test_nmf_memory_is_reported_before_the_work_not_after():
    """LDA streams and NMF does not, so the honest thing is to say how much it
    will want. A 20k-term matrix over 50k documents is not a small ask."""
    assert topics.nmf_memory_gb(50_000, 20_000, 50) > 20
    assert topics.nmf_memory_gb(100, 200, 5) < 0.01


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def test_a_coherent_topic_scores_above_an_incoherent_one():
    """Terms drawn from one planted family really do co-occur; terms taken one
    from each family really do not. If the metric cannot tell those apart it
    cannot guide anything."""
    counts = planted()
    pair, single, n_docs = topics.codocument_counts(batches_of(counts), len(VOCAB))

    coherent = list(FAMILY_A)[:4]
    mixed = [0, 1, V_PER_FAMILY, V_PER_FAMILY + 1]
    for metric in topics.COHERENCE_METRICS:
        scores = topics.coherence([coherent, mixed], pair, single, n_docs,
                                  metric=metric)
        assert scores[0] > scores[1], (metric, scores)


def test_npmi_stays_inside_minus_one_and_one():
    """
    The *normalized* in NPMI. Plain pointwise mutual information is unbounded
    and grows with corpus size and rarity, so two sweeps over different corpora
    -- or the same corpus at different topic counts, where the top terms get
    rarer as topics multiply -- cannot be compared on it.

    Ranking alone does not catch this: raw PMI separates a coherent topic from
    an incoherent one just as happily. The bound is the property that makes the
    number mean something on its own, so the bound is what gets tested.
    """
    counts = planted()
    pair, single, n_docs = topics.codocument_counts(batches_of(counts), len(VOCAB))

    every = [list(FAMILY_A)[:4], list(FAMILY_B)[:4],
             [0, 1, V_PER_FAMILY, V_PER_FAMILY + 1], [0, V_PER_FAMILY]]
    scores = topics.coherence(every, pair, single, n_docs, metric="npmi")
    assert all(-1.0 <= score <= 1.0 for score in scores), scores


def test_codocument_counts_are_about_presence_not_frequency():
    """Saying a word forty times in one document is still one document."""
    counts = np.zeros((2, len(VOCAB)))
    counts[0, 0] = 40
    counts[0, 1] = 1
    counts[1, 0] = 1
    pair, single, n_docs = topics.codocument_counts(batches_of(counts), len(VOCAB))
    assert n_docs == 2
    assert single[0] == 2 and single[1] == 1
    assert pair[0, 1] == 1


def test_an_unknown_coherence_metric_is_refused_by_name():
    counts = planted()
    pair, single, n_docs = topics.codocument_counts(batches_of(counts), len(VOCAB))
    with pytest.raises(ValueError, match="c_v|unknown coherence"):
        topics.coherence([[0, 1]], pair, single, n_docs, metric="c_v")


def test_top_terms_refuses_a_vocabulary_of_the_wrong_size():
    """Silently zipping a model against the wrong word list would label every
    topic with somebody else's words."""
    counts = planted()
    _w, h, _ = topics.fit_nmf(counts, 2)
    with pytest.raises(ValueError, match="same vocabulary"):
        topics.top_terms(h, VOCAB[:3])


# ---------------------------------------------------------------------------
# Checked against scikit-learn
# ---------------------------------------------------------------------------
#
# The established pattern here: sklearn is the oracle these are measured
# against, never a dependency they lean on (decision D9 in
# planning/stats-framework.md; see also test_stats_pca.py and
# test_stats_ridge.py, which do the same for PCA and ridge).
#
# The strength of the claim differs by method, and saying so is the point:
#
#   PCA  -- one right answer, checked to machine precision.
#   NMF  -- same updates from the same start, so the two should track closely.
#   LDA  -- no closed form, a local optimum, and topic order is arbitrary.
#           Digits here would pin an accident. What is checked is that the two
#           implementations find the *same topics*, matched up by correlation.

def test_our_lda_finds_the_same_topics_as_sklearns():
    """Not to machine precision -- see the note above. Same topics, matched by
    correlation, which is the strongest claim the method supports."""
    sklearn_lda = pytest.importorskip("sklearn.decomposition")

    counts = planted()
    ours, _ = topics.fit_lda(batches_of(counts), len(VOCAB), 2, passes=20, seed=7)
    theirs = sklearn_lda.LatentDirichletAllocation(
        n_components=2, random_state=0, max_iter=50,
        doc_topic_prior=0.1, topic_word_prior=0.01,
    ).fit(counts).components_

    assert _best_match(ours, theirs) > 0.95, (
        "our LDA and sklearn's disagree about what the topics are")


def test_our_nmf_finds_the_same_topics_as_sklearns():
    """NMF is the better-behaved of the two: same multiplicative updates from
    the same NNDSVD start, so these should track each other closely."""
    sklearn_decomp = pytest.importorskip("sklearn.decomposition")

    counts = planted()
    _w, ours, _ = topics.fit_nmf(counts, 2, iters=400)
    theirs = sklearn_decomp.NMF(
        n_components=2, init="nndsvd", max_iter=400, tol=1e-6,
    ).fit(counts).components_

    assert _best_match(ours, theirs) > 0.99, (
        "our NMF and sklearn's disagree about what the topics are")
