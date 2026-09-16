"""
The arithmetic behind LDA and NMF, with no opinions about files.

This is to `topic_model_lda` and `topic_model_nmf` what `stats/pca.py` is to
`topic_model_mem`: the numbers, separated from the plumbing, so they can be
tested against a known answer without a corpus or a CSV anywhere near them.

Why this is hand-written
------------------------
Because every other statistical method here is. Decision D9 in
`planning/stats-framework.md` settled it for ridge -- *"stays pure numpy (SVD)
for exact reproducibility; sklearn is used only in tests via importorskip"* --
and PCA, the correlations, the group differences and the classifiers all follow
it. scikit-learn is the oracle these are checked against, not a dependency they
lean on. Two practical consequences fall out of that and are worth having: LDA
works on Python 3.14, where the gensim-backed word-vector training does not, and
a fit is reproducible from a seed rather than from whatever a library's
threading happened to do.

What the two methods are
------------------------
**LDA** is a generative story: every document is a mixture over topics, every
topic a distribution over words, and fitting means finding the mixtures that
best explain the counts you actually saw. Solved here by *online variational
Bayes* (Hoffman, Blei & Bach 2010), which is the same algorithm scikit-learn
implements -- a per-document inner loop that settles each document's topic
mixture, and an outer update that nudges the topics themselves.

**NMF** asks something simpler and less probabilistic: split the matrix into two
non-negative pieces whose product is close to it. No generative story, no
priors, and on short texts the topics are often more readable for exactly that
reason. Solved by multiplicative updates (Lee & Seung 1999).

The memory shapes differ, and it matters
----------------------------------------
LDA's model is K by V -- topics by vocabulary -- and documents stream through in
batches, so nothing here grows with the size of the corpus. That is *better*
than the PCA path MEM uses, whose cross-product matrix is V by V and is what
`stats.pca.warn_if_wide` exists to warn about.

NMF is the other way round: the multiplicative updates need the whole D by V
matrix at once. :func:`nmf_memory_gb` is there to say so before a long read
rather than after an out-of-memory.
"""

from __future__ import annotations

from typing import Callable, Iterator, List, Sequence

__all__ = [
    "dirichlet_expectation", "fit_lda", "infer_lda", "perplexity",
    "nndsvd", "fit_nmf", "infer_nmf", "nmf_memory_gb",
    "top_terms", "coherence", "COHERENCE_METRICS",
]

#: What `coherence` can score. Both are computable from the matrix alone, in one
#: pass. C_v -- the one people usually quote -- is deliberately absent: it needs
#: sliding windows over the raw text rather than over the document-term matrix,
#: and wanting it is the usual reason people end up depending on gensim.
COHERENCE_METRICS = ("umass", "npmi")

#: Guards against log(0) and division by zero without perturbing any answer at
#: the scale the results are reported to.
_TINY = 1e-100


def dirichlet_expectation(alpha):
    """``E[log x]`` for ``x ~ Dirichlet(alpha)``: ``psi(a) - psi(sum(a))``.

    Handles a single vector or a stack of them; for a 2-D input the sum is over
    the last axis, which is the convention both the topic-word matrix and the
    document-topic matrix are stored in.
    """
    import numpy as np
    from scipy.special import psi

    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim == 1:
        return psi(alpha) - psi(alpha.sum())
    return psi(alpha) - psi(alpha.sum(axis=1))[:, None]


# ---------------------------------------------------------------------------
# LDA -- online variational Bayes
# ---------------------------------------------------------------------------

def _e_step(counts, exp_elogbeta, alpha: float, *, iters: int, tol: float):
    """
    Settle one document's topic mixture with the topics held still.

    Returns the document's ``gamma`` (its Dirichlet parameter over topics) and
    the sufficient statistics its words contribute to the topics. This is the
    whole of inference: :func:`infer_lda` is this and nothing else, which is why
    applying a saved model to the texts it was fitted on reproduces the training
    scores exactly rather than approximately.

    Only the words the document actually contains take part. A document with
    thirty distinct words does thirty columns of work regardless of how big the
    vocabulary is, which is what keeps this affordable at five thousand terms.
    """
    import numpy as np

    ids = np.nonzero(counts)[0]
    if ids.size == 0:
        # a document with none of the vocabulary in it. the prior is the only
        # honest answer, and it must not poison the topics with statistics from
        # words that were never there.
        k = exp_elogbeta.shape[0]
        return np.full(k, alpha, dtype=np.float64), ids, None

    cts = counts[ids].astype(np.float64)
    beta_d = exp_elogbeta[:, ids]                       # (K, n_words_here)

    gamma = np.ones(exp_elogbeta.shape[0], dtype=np.float64)
    exp_elogtheta = np.exp(dirichlet_expectation(gamma))
    norm = exp_elogtheta @ beta_d + _TINY

    for _ in range(max(1, iters)):
        previous = gamma
        gamma = alpha + exp_elogtheta * (beta_d @ (cts / norm))
        exp_elogtheta = np.exp(dirichlet_expectation(gamma))
        norm = exp_elogtheta @ beta_d + _TINY
        if float(np.mean(np.abs(gamma - previous))) < tol:
            break

    stats = np.outer(exp_elogtheta, cts / norm) * beta_d
    return gamma, ids, stats


def fit_lda(batches: Callable[[], Iterator], n_terms: int, k: int, *,
            alpha: float = 0.1, eta: float = 0.01, passes: int = 10,
            seed: int = 42, doc_iters: int = 100, tol: float = 1e-3,
            tau: float = 1.0, kappa: float = 0.7,
            on_progress=None, message: str = "fitting topics"):
    """
    Fit LDA to a corpus that arrives in batches.

    Parameters
    ----------
    batches : callable
        Called with no arguments and returns a fresh iterator of 2-D count
        arrays, each ``(rows, n_terms)``. A *callable* rather than an iterator
        because fitting makes several passes, and an iterator can only be
        walked once -- handing one in gets you a first pass and then silence.
    n_terms, k : int
        Vocabulary size and number of topics.
    alpha, eta : float
        Priors on the document-topic and topic-word distributions. Smaller
        makes each document (or topic) commit to fewer things.
    passes : int
        How many times to walk the corpus.
    seed : int
        Fixes the one random thing here, the initial topic-word matrix, which
        is what makes the whole fit reproducible.
    tau, kappa : float
        The learning schedule; a batch's influence decays as
        ``(tau + t) ** -kappa``. `kappa` must sit in (0.5, 1] for the updates to
        converge -- Hoffman et al. 2010, section 2.3.

    Returns
    -------
    (lam, trace)
        ``lam`` is the fitted topic-word matrix, ``(k, n_terms)``, unnormalized
        -- the Dirichlet parameters, not probabilities. ``trace`` lists the
        per-pass bound on held-in perplexity, which must fall; a run where it
        does not is a run that learned nothing.
    """
    import numpy as np

    if k < 2:
        raise ValueError(f"a topic model needs at least 2 topics, not {k}")
    if not 0.5 < kappa <= 1.0:
        raise ValueError(
            f"kappa={kappa} is outside (0.5, 1]; outside that range the online "
            "updates are not guaranteed to converge (Hoffman et al. 2010)")

    rng = np.random.default_rng(seed)
    lam = rng.gamma(100.0, 1.0 / 100.0, size=(k, n_terms))

    n_docs = 0
    for block in batches():
        n_docs += len(block)
    if n_docs < k:
        raise ValueError(
            f"{n_docs} document(s) cannot support {k} topics; a topic model "
            "needs more documents than topics, and in practice many more")

    trace: List[float] = []
    step = 0
    for _pass in range(max(1, passes)):
        exp_elogbeta = np.exp(dirichlet_expectation(lam))
        words_seen = 0.0
        bound = 0.0

        for block in batches():
            block = np.asarray(block, dtype=np.float64)
            stats = np.zeros_like(lam)
            for row in block:
                gamma, ids, contribution = _e_step(
                    row, exp_elogbeta, alpha, iters=doc_iters, tol=tol)
                if contribution is not None:
                    stats[:, ids] += contribution
                    bound += _row_bound(row, ids, gamma, exp_elogbeta, alpha)
                    words_seen += float(row.sum())

            # scale this batch's statistics up to the whole corpus before
            # mixing them in, which is what makes a batch update an estimate of
            # the full-corpus one rather than of itself
            scale = n_docs / max(1, len(block))
            rho = (tau + step) ** (-kappa)
            lam = (1.0 - rho) * lam + rho * (eta + scale * stats)
            exp_elogbeta = np.exp(dirichlet_expectation(lam))
            step += 1

        trace.append(float(np.exp(-bound / max(words_seen, 1.0))))
        if on_progress is not None:
            try:
                on_progress(_pass + 1, max(1, passes), message=message)
            except TypeError:
                on_progress(_pass + 1, max(1, passes))

    return lam, trace


def _row_bound(counts, ids, gamma, exp_elogbeta, alpha: float) -> float:
    """One document's contribution to the log-likelihood the trace reports."""
    import numpy as np

    if ids.size == 0:
        return 0.0
    exp_elogtheta = np.exp(dirichlet_expectation(gamma))
    per_word = exp_elogtheta @ exp_elogbeta[:, ids] + _TINY
    return float(np.sum(counts[ids] * np.log(per_word)))


def infer_lda(lam, counts, *, alpha: float = 0.1, iters: int = 100,
              tol: float = 1e-3, exp_elogbeta=None):
    """
    Topic proportions for documents, with the topics held fixed.

    This is what applying a saved model does, and it is deliberately the same
    inner loop the fit uses -- if the two ever drift apart, applying a model to
    its own training texts stops reproducing the training scores, which is the
    one test that would notice.

    ``exp_elogbeta`` is the one thing here that depends only on ``lam``, and
    it is a ``psi`` and an ``exp`` over the whole k-by-V matrix. The E-step
    itself only touches the document's non-zero columns, so recomputing this
    per document is most of the work: 7.1 ms a document against 1.3 ms at
    k=20 over a 2,000-term vocabulary, and the gap widens with the
    vocabulary. A caller scoring a whole corpus computes it once and passes
    it; anyone scoring one document can ignore it.

    Returns an array of ``(rows, k)`` proportions, each row summing to 1.
    """
    import numpy as np

    lam = np.asarray(lam, dtype=np.float64)
    if exp_elogbeta is None:
        exp_elogbeta = np.exp(dirichlet_expectation(lam))
    rows = np.atleast_2d(np.asarray(counts, dtype=np.float64))

    out = np.empty((rows.shape[0], lam.shape[0]), dtype=np.float64)
    for i, row in enumerate(rows):
        gamma, _ids, _stats = _e_step(row, exp_elogbeta, alpha,
                                      iters=iters, tol=tol)
        out[i] = gamma / gamma.sum()
    return out


def perplexity(lam, counts, *, alpha: float = 0.1) -> float:
    """
    Held-out perplexity: how surprised the model is by documents it did not
    see. Lower is better, and the only honest use of it is comparing models
    fitted on the same vocabulary.
    """
    import numpy as np

    rows = np.atleast_2d(np.asarray(counts, dtype=np.float64))
    theta = infer_lda(lam, rows, alpha=alpha)
    beta = np.asarray(lam, dtype=np.float64)
    beta = beta / beta.sum(axis=1)[:, None]

    total = float(rows.sum())
    if total <= 0:
        return float("inf")
    per_word = theta @ beta                              # (docs, terms)
    return float(np.exp(-np.sum(rows * np.log(per_word + _TINY)) / total))


# ---------------------------------------------------------------------------
# NMF -- non-negative matrix factorization
# ---------------------------------------------------------------------------

def nmf_memory_gb(n_docs: int, n_terms: int, k: int) -> float:
    """
    Roughly what :func:`fit_nmf` will want, in GB.

    LDA streams and NMF does not: the multiplicative updates need the whole
    matrix and a working copy of the same shape. Saying so before a long read
    is better than an out-of-memory after it -- the same courtesy
    `stats.pca.warn_if_wide` pays for the other direction.
    """
    cells = n_docs * n_terms * 3 + (n_docs + n_terms) * k
    return cells * 8 / 1024 ** 3


def nndsvd(x, k: int):
    """
    Non-negative double singular value decomposition: a starting point for NMF.

    Deterministic, which is the reason to prefer it over a random start here --
    NMF then needs no seed at all, and two runs on one corpus give one answer.
    Boutsidis & Gallopoulos (2008).
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if k > min(x.shape):
        raise ValueError(
            f"cannot ask for {k} topics from a {x.shape[0]}x{x.shape[1]} "
            "matrix; there are not that many dimensions in it")

    u, s, vt = np.linalg.svd(x, full_matrices=False)
    w = np.zeros((x.shape[0], k))
    h = np.zeros((k, x.shape[1]))

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(vt[0, :])

    for j in range(1, k):
        uj, vj = u[:, j], vt[j, :]
        up, un = np.maximum(uj, 0), np.maximum(-uj, 0)
        vp, vn = np.maximum(vj, 0), np.maximum(-vj, 0)
        np_, nn_ = np.linalg.norm(up) * np.linalg.norm(vp), \
            np.linalg.norm(un) * np.linalg.norm(vn)
        if np_ >= nn_:
            scale, u_part, v_part = np_, up / (np.linalg.norm(up) + _TINY), \
                vp / (np.linalg.norm(vp) + _TINY)
        else:
            scale, u_part, v_part = nn_, un / (np.linalg.norm(un) + _TINY), \
                vn / (np.linalg.norm(vn) + _TINY)
        w[:, j] = np.sqrt(s[j] * scale) * u_part
        h[j, :] = np.sqrt(s[j] * scale) * v_part

    # zeros are absorbing under multiplicative updates: a cell that starts at
    # zero can never move off it, so the factorization would be stuck with
    # whatever NNDSVD happened to leave empty.
    eps = float(np.mean(x)) * 1e-6 if x.size else _TINY
    w[w < eps] = eps
    h[h < eps] = eps
    return w, h


def fit_nmf(x, k: int, *, beta_loss: str = "frobenius", iters: int = 200,
            tol: float = 1e-4, on_progress=None, message: str = "fitting topics"):
    """
    Factor a non-negative matrix into ``W @ H`` by multiplicative updates.

    Parameters
    ----------
    x : array
        Documents by terms. Counts or tf-idf; both are non-negative, which is
        the only thing this requires.
    beta_loss : {"frobenius", "kullback-leibler"}
        What "close" means. Frobenius is least squares and is the faster and
        steadier of the two; KL matches the way counts actually vary and tends
        to give sparser, more readable topics on short texts.

    Returns
    -------
    (w, h, trace)
        ``w`` is documents by topics, ``h`` topics by terms, and ``trace`` the
        reconstruction error after each iteration -- which must fall.
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0):
        raise ValueError(
            "NMF needs a non-negative matrix; this one has negative cells, "
            "which usually means it was standardized somewhere upstream")
    if beta_loss not in ("frobenius", "kullback-leibler"):
        raise ValueError(f"unknown beta_loss {beta_loss!r}")

    w, h = nndsvd(x, k)
    trace: List[float] = []
    previous = None

    for step in range(max(1, iters)):
        if beta_loss == "frobenius":
            h *= (w.T @ x) / (w.T @ w @ h + _TINY)
            w *= (x @ h.T) / (w @ (h @ h.T) + _TINY)
            error = float(np.linalg.norm(x - w @ h))
        else:
            wh = w @ h + _TINY
            h *= (w.T @ (x / wh)) / (w.sum(axis=0)[:, None] + _TINY)
            wh = w @ h + _TINY
            w *= ((x / wh) @ h.T) / (h.sum(axis=1)[None, :] + _TINY)
            wh = w @ h + _TINY
            error = float(np.sum(x * np.log((x + _TINY) / wh) - x + wh))

        trace.append(error)
        if on_progress is not None:
            try:
                on_progress(step + 1, max(1, iters), message=message)
            except TypeError:
                on_progress(step + 1, max(1, iters))
        if previous is not None and abs(previous - error) <= tol * max(previous, _TINY):
            break
        previous = error

    return w, h, trace


def infer_nmf(h, x, *, beta_loss: str = "frobenius", iters: int = 200,
              tol: float = 1e-4):
    """
    Topic weights for documents, with the topics held fixed -- NMF's apply.

    Only ``W`` moves: ``H`` is the instrument and updating it here would be
    refitting the model to the new corpus, which is exactly what makes two
    studies incomparable.

    ``beta_loss`` has to match whatever :func:`fit_nmf` used, and for the same
    reason the fit takes it: the two divergences have different update rules
    and different optima. Scoring a KL-fitted model with the least-squares
    update gives every document the wrong ``W`` -- not the one the fit found,
    and optimal under a divergence nobody chose. Found in review; it used to
    be Frobenius always, whatever the model said.
    """
    import numpy as np

    if beta_loss not in ("frobenius", "kullback-leibler"):
        raise ValueError(f"unknown beta_loss {beta_loss!r}")

    h = np.asarray(h, dtype=np.float64)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))

    start = float(np.mean(x)) if x.size else 1.0
    w = np.full((x.shape[0], h.shape[0]), max(start, _TINY))
    previous = None
    # both of these depend only on H, which is frozen here by definition --
    # and they were being rebuilt on every one of the 200 iterations of every
    # document. `x @ h.T` is loop-invariant too, since only W moves.
    xht = x @ h.T
    hht = h @ h.T
    hsum = h.sum(axis=1)[None, :]
    for _ in range(max(1, iters)):
        if beta_loss == "frobenius":
            w *= xht / (w @ hht + _TINY)
            error = float(np.linalg.norm(x - w @ h))
        else:
            wh = w @ h + _TINY
            w *= ((x / wh) @ h.T) / (hsum + _TINY)
            wh = w @ h + _TINY
            error = float(np.sum(x * np.log((x + _TINY) / wh) - x + wh))
        if previous is not None and abs(previous - error) <= tol * max(previous, _TINY):
            break
        previous = error
    return w


# ---------------------------------------------------------------------------
# Reading the topics back
# ---------------------------------------------------------------------------

def top_terms(components, terms: Sequence[str], n: int = 10) -> List[List[str]]:
    """The ``n`` heaviest terms of each topic, heaviest first."""
    import numpy as np

    components = np.asarray(components, dtype=np.float64)
    if components.shape[1] != len(terms):
        raise ValueError(
            f"the model has {components.shape[1]} terms and {len(terms)} names "
            "were given; they have to be the same vocabulary")
    out = []
    for row in components:
        order = np.argsort(row)[::-1][:n]
        out.append([terms[i] for i in order])
    return out


def codocument_counts(batches: Callable[[], Iterator], n_terms: int):
    """
    How often each pair of terms appears in the same document, and how often
    each appears at all -- one pass, on presence rather than frequency.

    This is everything the coherence metrics need, which is why they cost a
    pass over the matrix rather than a second pass over the corpus.
    """
    import numpy as np

    pair = np.zeros((n_terms, n_terms), dtype=np.float64)
    single = np.zeros(n_terms, dtype=np.float64)
    n_docs = 0
    for block in batches():
        present = (np.asarray(block, dtype=np.float64) > 0).astype(np.float64)
        pair += present.T @ present
        single += present.sum(axis=0)
        n_docs += present.shape[0]
    return pair, single, n_docs


def coherence(topics: Sequence[Sequence[int]], pair, single, n_docs: int, *,
              metric: str = "npmi") -> List[float]:
    """
    Score each topic on whether its top terms actually turn up together.

    Parameters
    ----------
    topics : sequence of sequences of int
        Each topic's top terms, as indices into the vocabulary, best first.
    metric : {"npmi", "umass"}
        ``umass`` is Mimno et al. (2011): asymmetric, and reads the top term as
        the one the others should co-occur with. ``npmi`` is symmetric and
        normalized to [-1, 1], which makes scores comparable across topic
        counts -- the reason it is the default here, since comparing across
        topic counts is the entire point of a sweep.

    Coherence is a guide, not a verdict. It rewards topics whose words co-occur,
    which is *related to* but not the same as topics a person would find
    meaningful, and a sweep that picks the highest number without anybody
    reading the topics has learned nothing.
    """
    import numpy as np

    if metric not in COHERENCE_METRICS:
        raise ValueError(
            f"unknown coherence metric {metric!r}; have {COHERENCE_METRICS}")
    if n_docs <= 0:
        raise ValueError("no documents to score coherence over")

    scores: List[float] = []
    for terms in topics:
        terms = list(terms)
        values = []
        for i in range(1, len(terms)):
            for j in range(i):
                a, b = terms[i], terms[j]
                both = pair[a, b]
                if metric == "umass":
                    values.append(float(np.log((both + 1.0) / max(single[b], 1.0))))
                else:
                    p_both = both / n_docs
                    p_a, p_b = single[a] / n_docs, single[b] / n_docs
                    if p_both <= 0 or p_a <= 0 or p_b <= 0:
                        values.append(-1.0)
                        continue
                    pmi = np.log(p_both / (p_a * p_b))
                    values.append(float(pmi / -np.log(p_both)))
        scores.append(float(np.mean(values)) if values else 0.0)
    return scores


def stream_counts(csv_path, *, encoding: str, skip_cols: int,
                  chunk_rows: int = 4096) -> Iterator:
    """
    Walk a document-term matrix in blocks of rows, dropping the leading
    identifier columns.

    A generator, so callers that need several passes hand ``fit_lda`` a
    ``lambda: stream_counts(...)`` rather than the iterator itself.
    """
    import csv as _csv
    from pathlib import Path

    import numpy as np

    block: List[List[str]] = []
    with Path(csv_path).open("r", newline="", encoding=encoding) as f:
        reader = _csv.reader(f)
        next(reader, None)                              # the header is the caller's
        for row in reader:
            block.append(row[skip_cols:])
            if len(block) >= chunk_rows:
                yield np.asarray(block, dtype=np.float64)
                block = []
    if block:
        yield np.asarray(block, dtype=np.float64)


def read_matrix(csv_path, *, encoding: str, skip_cols: int):
    """The whole matrix at once, for NMF. See :func:`nmf_memory_gb`."""
    import numpy as np

    blocks = list(stream_counts(csv_path, encoding=encoding, skip_cols=skip_cols))
    if not blocks:
        return np.zeros((0, 0), dtype=np.float64)
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# The matrix each model builds for itself
# ---------------------------------------------------------------------------

def check_matrix_agrees(dtm_csv, columns, *, encoding: str) -> None:
    """
    Refuse a matrix whose columns are not the vocabulary we just derived.

    Both halves of a model's substrate honor ``overwrite_existing=False``, so
    a surviving ``dtm.csv`` is reused while the vocabulary is re-derived from
    the frequency list. Delete a results table to force a re-fit, interrupt a
    run, or change `vocab_top_n`, and the two come apart -- and nothing
    downstream can tell. `kept_terms = [terms[j] for j in kept]` then picks
    the wrong words (the top-N-by-obs_pct subset is not a prefix of the
    frequency-sorted list), and the model records a vocabulary its own
    projection does not match: wrong loadings, an unapplicable model, no error.

    MEM used to compare these headers and the check was lost when each model
    started building its own matrix. Reading one line is nothing next to a
    silently wrong model.
    """
    import csv as _csv
    from pathlib import Path

    with Path(dtm_csv).open("r", newline="", encoding=encoding) as fh:
        header = next(_csv.reader(fh), [])
    want = ["text_id", "token_count", *columns]
    if header != want:
        raise ValueError(
            f"{dtm_csv} does not match the vocabulary built beside it: it has "
            f"{max(0, len(header) - 2):,} term column(s) and the vocabulary has "
            f"{len(columns):,}. An earlier run left a matrix behind and this "
            "one reused it. Delete that folder, or pass "
            "`overwrite_existing=True`, and run again.")


def build_matrix(*, analysis_ready, out_dir, weighting: str, matrix_rounding: int,
                 text_settings: dict, ngram_settings: dict, vocab_settings: dict,
                 overwrite_existing: bool, workers, on_progress, encoding: str):
    """
    This model's own frequency list and document-term matrix, in its own folder.

    Every topic model calls this rather than sharing one matrix, because they
    do not want the same matrix. LDA is only defined over integer counts, NMF
    conventionally wants tf-idf, MEM takes either plus one-hot, and each wants
    a vocabulary sized for itself. Worse, the shared matrix's filename carried
    only the weighting -- so two models asking for different vocabularies from
    one corpus targeted the same file and rebuilt over each other, and whichever
    ran last won.

    They land beside the results rather than in a temp folder: somebody reading
    a topic model wants to see which vocabulary the topics came out of.

    Returns
    -------
    (freq_list_csv, dtm_csv, terms, columns, vocab)
        The two files, the vocabulary in matrix-column order, the column names
        (which differ from ``terms`` only when POS tags are in play), and the
        frequency-list rows behind them -- ``idf`` among them, which a model
        needs to recreate a tf-idf matrix later.
    """
    from pathlib import Path

    from .analyze_ngram_frequencies import analyze_ngram_frequencies
    from .build_doc_term_matrix import _load_vocabulary, build_doc_term_matrix
    from .ngram_prep import tags_of, words_of

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if on_progress is not None:
        from ..helpers.progress import announce
        announce(on_progress, "counting the vocabulary")
    freq_list_csv = analyze_ngram_frequencies(
        analysis_csv=analysis_ready, out_features_csv=out_dir / "freq_list.csv",
        overwrite_existing=overwrite_existing, workers=workers,
        on_progress=on_progress, encoding=encoding,
        **ngram_settings, **text_settings)

    if on_progress is not None:
        from ..helpers.progress import announce
        announce(on_progress, "building the matrix")
    dtm_csv = Path(build_doc_term_matrix(
        freq_list_csv=freq_list_csv, analysis_csv=analysis_ready,
        out_features_csv=out_dir / "dtm.csv",
        overwrite_existing=overwrite_existing, workers=workers,
        on_progress=on_progress, encoding=encoding,
        weighting=weighting, rounding=matrix_rounding,
        **vocab_settings, **text_settings))

    vocab = _load_vocabulary(
        Path(freq_list_csv), encoding=encoding,
        pos_tagged=text_settings.get("pos_tagged", False), **vocab_settings)
    terms = sorted(vocab, key=lambda g: (-vocab[g]["frequency"],
                                         words_of(g), tags_of(g)))
    from .build_doc_term_matrix import column_names
    columns = column_names(terms, text_settings.get("pos_tagged", False))
    check_matrix_agrees(dtm_csv, columns, encoding=encoding)
    return Path(freq_list_csv), dtm_csv, terms, columns, vocab
