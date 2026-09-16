"""
MEM builds its own document-term matrix, and doing so changed none of its numbers.

Why this file exists
--------------------
The matrix was being treated as universal when it is not. `topic_model_mem`
declared `requires={"doc_term_matrix_csv", "ngram_freq_csv"}`, so the composer
wired every topic model in a pipeline to **one** matrix built with **one** set of
settings -- and the three engines want different ones. LDA is only defined over
integer counts; NMF conventionally wants tf-idf; MEM takes either plus one-hot,
and wants a vocabulary sized for MEM rather than for LDA.

The clash was not hypothetical. The matrix's default path is
``features/doc_term_matrix/<stem>_<weighting>.csv`` -- the *weighting* is in the
filename but the vocabulary and tokenizer settings are not. Two topic models
asking for different vocabularies from one corpus therefore targeted the same
file, each noticed the record did not match its settings, and each rebuilt over
the other's matrix. They thrash, and whichever ran last wins.

So each topic model now builds its own, under its own folder.

What this file pins
-------------------
That the change was a plumbing change. The numbers below were produced by the
*previous* implementation -- the one handed a matrix built by separate steps --
and captured before a line of it moved. If MEM now computes anything different
from the same corpus at the same settings, these fail, and that is the whole
point: the reorganization was supposed to be invisible in the results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

def matrix_of(out) -> Path:
    """Where a step put the matrix it built for itself.

    Derived from the output's *stem*, not a fixed `matrix/` beside it: two
    steps writing into one folder would otherwise share one matrix folder,
    which is how three topic models came to build over each other's and how an
    apply left a frozen vocabulary for a later fit to pick up.
    """
    out = Path(out)
    return out.with_name(f"{out.stem}_matrix")


#: Theme scores from the implementation that read a matrix somebody else built.
#: Captured before the change, deliberately: a golden file written *after* a
#: refactor only proves the refactor agrees with itself.
GOLDEN = {
    "d00": (11, -0.599, -2.0024),
    "d01": (26, 3.4606, -2.0729),
    "d02": (21, -2.865, 1.8705),
    "d03": (32, -0.0092, 2.4324),
    "d04": (29, -0.4875, 2.5264),
    "d05": (27, 3.432, -1.8118),
    "d06": (7, -2.7676, -1.5835),
    "d07": (15, -0.1378, -1.8538),
    "d08": (33, -0.7344, 3.4402),
    "d09": (49, 3.6684, 3.7204),
    "d10": (7, -2.7393, -1.582),
    "d11": (18, -0.1681, -1.0827),
    "d12": (15, -0.6057, -0.9977),
    "d13": (28, 3.5143, -1.5752),
    "d14": (33, -2.9343, 4.843),
    "d15": (17, -0.1661, -1.3421),
    "d16": (17, -0.5524, -0.4923),
    "d17": (33, 3.4823, -0.2942),
    "d18": (11, -2.7742, -0.5788),
    "d19": (17, -0.0841, -1.3561),
    "d20": (15, -0.6304, -1.0185),
    "d21": (34, 3.5115, -0.0542),
    "d22": (15, -2.6961, 0.4304),
    "d23": (24, -0.1179, 0.4348),
}

FOOD = "bread butter cheese dinner kitchen recipe".split()
WORK = "office meeting deadline manager project report".split()


@pytest.fixture
def corpus(tmp_path) -> Path:
    """Two word families in varying proportions.

    Deliberately non-periodic: an earlier version of this corpus repeated every
    six documents, so four rows shared each answer and a change that shuffled
    the rows could have gone unnoticed. All twenty-four scores differ now.
    """
    path = tmp_path / "corpus.csv"
    rows = [("text_id", "text")]
    for i in range(24):
        food = FOOD[: 1 + (i * 5 + 1) % 6] * (1 + i % 5)
        work = WORK[: 1 + (i * 3 + 2) % 6] * (1 + (i + 2) % 4)
        rows.append((f"d{i:02d}", " ".join(food + work)))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def _scores(path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return {r["text_id"]: (int(r["token_count"]),
                           float(r["Theme_1"]), float(r["Theme_2"])) for r in rows}


def test_the_numbers_are_the_same_as_before_mem_owned_its_matrix(corpus, tmp_path):
    """
    The governing test for the change. Same corpus, same settings, same scores
    as the implementation that was handed a matrix built by separate steps.
    """
    from taters.text.topic_model_mem import topic_model_mem

    out = topic_model_mem(
        csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "scores.csv",
        out_model_json=tmp_path / "model.json",
        out_loadings_csv=tmp_path / "loadings.csv",
        out_eigenvalues_csv=tmp_path / "eigen.csv",
        min_freq=1, min_obs_pct=0.0, min_token_count=1,
        vocab_min_freq=1, vocab_min_obs_pct=0.0,
        n_components=2, overwrite_existing=True,
    )

    got = _scores(out)
    assert set(got) == set(GOLDEN), "the documents themselves changed"
    for text_id, (tokens, theme1, theme2) in GOLDEN.items():
        assert got[text_id][0] == tokens, f"{text_id}: token count moved"
        assert got[text_id][1] == pytest.approx(theme1, abs=5e-4), f"{text_id}: Theme_1"
        assert got[text_id][2] == pytest.approx(theme2, abs=5e-4), f"{text_id}: Theme_2"


def test_the_matrix_it_built_is_left_where_somebody_can_look_at_it(corpus, tmp_path):
    """
    The matrix is not a temp file. Somebody reading a topic model's output
    wants to see which vocabulary the themes came out of, and the frequency
    list that chose it -- so both are written beside the results rather than
    into a scratch directory that disappears.
    """
    from taters.text.topic_model_mem import topic_model_mem

    out = Path(topic_model_mem(
        csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "out" / "scores.csv",
        min_freq=1, min_obs_pct=0.0, min_token_count=1,
        vocab_min_freq=1, vocab_min_obs_pct=0.0,
        n_components=2, overwrite_existing=True,
    ))

    matrix_dir = matrix_of(out)
    assert matrix_dir.is_dir(), f"no matrix folder beside {out}"
    written = sorted(p.name for p in matrix_dir.glob("*.csv"))
    assert written, f"the matrix folder is empty: {matrix_dir}"


def test_two_topic_models_do_not_fight_over_one_matrix(corpus, tmp_path):
    """
    The bug the whole change exists for. Two models wanting different
    vocabularies from one corpus used to target the same matrix file, each
    rebuild clobbering the other. Their matrices must now be different files.
    """
    from taters.text.topic_model_mem import topic_model_mem

    outputs = []
    for name, top_n in (("narrow", 6), ("wide", 12)):
        outputs.append(Path(topic_model_mem(
            csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
            out_features_csv=tmp_path / name / "scores.csv",
            min_freq=1, min_obs_pct=0.0, min_token_count=1,
            vocab_min_freq=1, vocab_min_obs_pct=0.0, vocab_top_n=top_n,
            n_components=2, overwrite_existing=True,
        )))

    matrices = [sorted(matrix_of(out).glob("*.csv")) for out in outputs]
    assert all(matrices), "each model should have written its own matrix"
    flat = [p.resolve() for group in matrices for p in group]
    assert len(set(flat)) == len(flat), (
        "two models wrote to the same matrix file; that is the clobbering this "
        f"change removed: {flat}")


@pytest.mark.parametrize("engine", ["mem", "lda", "nmf"])
def test_a_matrix_that_stopped_matching_its_vocabulary_is_refused(engine,
                                                                  corpus,
                                                                  tmp_path):
    """
    MEM used to compare the matrix's header against the vocabulary it had
    just derived, and the check was lost when each model started building its
    own matrix -- LDA and NMF never had one at all.

    It matters because both halves honor ``overwrite_existing=False``: a
    surviving ``dtm.csv`` is reused while the vocabulary is re-derived. Where
    the settings changed, provenance notices and rebuilds; where the *file*
    drifted from the settings that describe it -- a half-written matrix from
    an interrupted run -- nothing did. The model then indexes the wrong terms
    and records a vocabulary its own projection does not match: no error, and
    results nobody can tell are wrong.
    """
    import importlib

    module = importlib.import_module(f"taters.text.topic_model_{engine}")
    fit = getattr(module, f"topic_model_{engine}")
    size = {"mem": dict(n_components=2)}.get(engine, dict(n_topics=2))
    common = dict(csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
                  min_freq=1, min_obs_pct=0.0, min_token_count=1,
                  vocab_min_freq=1, vocab_min_obs_pct=0.0, vocab_top_n=8,
                  **size)
    out = Path(fit(out_features_csv=tmp_path / "scores.csv", **common))

    # lop the last three term columns off the matrix, leaving its settings
    # record untouched -- so the rebuild gate sees nothing to redo
    dtm = matrix_of(out) / "dtm.csv"
    rows = [r.split(",")[:-3]
            for r in dtm.read_text(encoding="utf-8-sig").splitlines()]
    dtm.write_text("\n".join(",".join(r) for r in rows), encoding="utf-8")
    out.unlink()

    with pytest.raises(ValueError, match="does not match the vocabulary"):
        fit(out_features_csv=tmp_path / "scores.csv", **common)


def test_a_topic_model_ranks_its_vocabulary_by_spread_not_by_volume(tmp_path):
    """
    `vocab_top_n` has to rank by *something*, and for a topic model that
    something is how many documents use a term, not how many times it was
    said. A word one document repeats four hundred times tops the frequency
    ranking and cannot distinguish anything -- it only ever describes that one
    document. A word used once each across half the corpus is what a topic is
    made of.

    The standalone document-term matrix keeps ranking by frequency, because a
    *feature* table usually does want the commonest terms; this is a topic
    model's default, not a global one.
    """
    import random

    from taters.text.topic_model_lda import topic_model_lda

    rng = random.Random(0)
    path = tmp_path / "shouty.csv"
    rows = [("text_id", "text"), ("loud", " ".join(["shouty"] * 400))]
    for i in range(30):
        rows.append((f"d{i}", " ".join(
            ["spread"] + [rng.choice(FOOD[:4]) for _ in range(6)])))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    def vocabulary(**kwargs):
        out = Path(topic_model_lda(
            csv_path=path, text_cols=["text"], id_cols=["text_id"],
            n_topics=2, passes=4, min_freq=1, min_obs_pct=0.0,
            min_token_count=1, vocab_top_n=3,
            out_features_csv=tmp_path / f"{kwargs.get('vocab_rank_by', 'default')}.csv",
            overwrite_existing=True, **kwargs))
        header = (matrix_of(out) / "dtm.csv").read_text(
            encoding="utf-8-sig").splitlines()[0]
        return header.split(",")[2:]

    assert "shouty" not in vocabulary(), (
        "the default let one loud document buy the top of the vocabulary")
    assert "shouty" in vocabulary(vocab_rank_by="frequency"), (
        "ranking by frequency should still be reachable, and should still "
        "behave the old way -- otherwise this test proves nothing")


def test_only_the_topic_models_changed_how_they_rank(tmp_path):
    """The document-term matrix is a feature table in its own right, and its
    users generally do want the commonest terms."""
    import inspect

    from taters.text.build_doc_term_matrix import build_doc_term_matrix
    from taters.text.topic_count_sweep import sweep_topic_count
    from taters.text.topic_model_lda import topic_model_lda
    from taters.text.topic_model_mem import topic_model_mem
    from taters.text.topic_model_nmf import topic_model_nmf

    for fn in (topic_model_mem, topic_model_lda, topic_model_nmf,
               sweep_topic_count):
        got = inspect.signature(fn).parameters["vocab_rank_by"].default
        assert got == "obs_pct", (fn.__name__, got)
    assert inspect.signature(build_doc_term_matrix).parameters[
        "vocab_rank_by"].default == "frequency"


def test_the_vocabulary_settings_actually_reach_the_frequency_list(tmp_path):
    """
    Six settings became MEM's to pass on when it took over building the matrix
    -- `ngram_n`, `stoplist_paths`, `min_freq`, `min_obs_pct`,
    `min_token_count`, `min_npmi`. A parameter that is accepted and then not
    forwarded is the quietest kind of broken: the run succeeds, the settings
    are recorded in the model as though they applied, and the numbers are of
    something else entirely.

    So: one term appears exactly once in the corpus. Raising the frequency
    floor past it has to remove it from the matrix.
    """
    from taters.text.topic_model_mem import topic_model_mem

    path = tmp_path / "rare.csv"
    rows = [("text_id", "text")]
    for i in range(12):
        words = FOOD[: 1 + i % 6] * 2 + WORK[: 1 + (i + 3) % 6] * 2
        if i == 0:
            words = words + ["hapax"]          # said once, by one document
        rows.append((f"d{i:02d}", " ".join(words)))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    def terms_for(min_freq):
        out = Path(topic_model_mem(
            csv_path=path, text_cols=["text"], id_cols=["text_id"],
            out_features_csv=tmp_path / f"f{min_freq}" / "scores.csv",
            min_freq=min_freq, min_obs_pct=0.0, min_token_count=1,
            vocab_min_freq=0, vocab_min_obs_pct=0, vocab_top_n=500,
            n_components=2, overwrite_existing=True))
        header = (matrix_of(out) / "dtm.csv").read_text(
            encoding="utf-8-sig").splitlines()[0]
        return set(header.split(","))

    assert "hapax" in terms_for(1), "a floor of 1 should keep a term said once"
    assert "hapax" not in terms_for(2), (
        "min_freq did not reach the frequency list -- MEM accepted the setting "
        "and built its vocabulary without it")


def test_a_saved_model_still_scores_new_text_the_same_way(corpus, tmp_path):
    """
    The roundtrip spine, unchanged by any of this: applying a model to the
    texts it was fitted on reproduces the training scores. Apply never read a
    matrix file, so this is what proves fit and apply still agree about the
    vocabulary now that fit builds one.
    """
    from taters.text.topic_model_mem import apply_mem_model, topic_model_mem

    fitted = topic_model_mem(
        csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "scores.csv",
        out_model_json=tmp_path / "model.json",
        min_freq=1, min_obs_pct=0.0, min_token_count=1,
        vocab_min_freq=1, vocab_min_obs_pct=0.0,
        n_components=2, overwrite_existing=True,
    )
    replayed = apply_mem_model(
        model_json=tmp_path / "model.json",
        csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "replay.csv", overwrite_existing=True,
    )
    assert _scores(fitted) == _scores(replayed)
