"""
NMF as a step: the files it writes, the model it saves, and what it refuses.

The arithmetic lives in `test_topics_core.py`. What is under test here is the
plumbing around it, and the handful of places NMF genuinely differs from its
sibling `topic_model_nmf`:

* it defaults to **tf-idf** where LDA demands counts, and refuses the two
  weightings that would quietly destroy what it is looking for;
* its factor weights do **not** sum to one, which is the property that makes
  them easier to use as ordinary predictors than LDA's topic shares;
* it has **no seed**, because NNDSVD is derived from the matrix itself.

The spine is the same as everywhere else: applying a saved model to the texts
it was fitted on must reproduce the training scores exactly.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from taters.text.topic_model_nmf import (MODEL_FORMAT, MODEL_KIND,
                                         apply_nmf_model, topic_model_nmf)


def matrix_of(out) -> Path:
    """Where a step put the matrix it built for itself.

    Derived from the output's *stem*, not a fixed `matrix/` beside it: two
    steps writing into one folder would otherwise share one matrix folder,
    which is how three topic models came to build over each other's and how an
    apply left a frozen vocabulary for a later fit to pick up.
    """
    out = Path(out)
    return out.with_name(f"{out.stem}_matrix")


FOOD = "bread butter cheese dinner kitchen recipe".split()
WORK = "office meeting deadline manager project report".split()

#: Settings that keep the toy corpora below from being filtered away entirely.
#: Real defaults drop terms seen fewer than five times or in under a tenth of
#: documents, which on forty short documents is most of the vocabulary.
LOOSE = dict(min_freq=1, min_obs_pct=0.0, min_token_count=1,
             vocab_min_freq=0, vocab_min_obs_pct=0)


@pytest.fixture
def corpus(tmp_path) -> Path:
    """Forty documents, each drawn from one of two disjoint word families."""
    rng = random.Random(0)
    path = tmp_path / "corpus.csv"
    rows = [("text_id", "text")]
    for i in range(40):
        family = FOOD if i % 2 == 0 else WORK
        words = [rng.choice(family) for _ in range(30)]
        words.append(rng.choice(FOOD + WORK))          # a little leakage
        rows.append((f"d{i:02d}", " ".join(words)))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def fit(corpus, tmp_path, **kwargs):
    defaults = dict(csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
                    n_topics=2, overwrite_existing=True, **LOOSE)
    defaults.update(kwargs)
    defaults.setdefault("out_features_csv", tmp_path / "topics.csv")
    defaults.setdefault("out_model_json", tmp_path / "model.json")
    return Path(topic_model_nmf(**defaults))


def read(path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return {r["text_id"]: {k: v for k, v in r.items() if k != "text_id"}
            for r in rows}


# ---------------------------------------------------------------------------
# The roundtrip contract
# ---------------------------------------------------------------------------

def test_applying_a_model_to_its_training_texts_reproduces_the_scores(corpus,
                                                                      tmp_path):
    """
    The spine. Fitting and applying share one inference routine and one way of
    rebuilding the matrix; if either ever forks, this is what says so.
    """
    trained = fit(corpus, tmp_path)
    replayed = apply_nmf_model(
        model_json=tmp_path / "model.json", csv_path=corpus,
        text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "replay.csv", overwrite_existing=True)

    assert read(trained) == read(replayed)


def test_applying_to_a_subset_uses_the_models_topics_not_the_subsets(corpus,
                                                                    tmp_path):
    """
    A model applied to five documents must score them exactly as it scored
    them inside forty. Anything that re-derived the vocabulary, or refitted,
    from the corpus in front of it would drift here -- and that drift is what
    makes two studies incomparable.
    """
    trained = read(fit(corpus, tmp_path))

    subset = tmp_path / "subset.csv"
    rows = list(csv.DictReader(open(corpus, encoding="utf-8-sig")))[:5]
    with subset.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id", "text"])
        writer.writeheader()
        writer.writerows(rows)

    scored = read(apply_nmf_model(
        model_json=tmp_path / "model.json", csv_path=subset,
        text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "subset_out.csv", overwrite_existing=True))

    assert set(scored) <= set(trained)
    for text_id, values in scored.items():
        assert values == trained[text_id], text_id


def test_text_the_model_has_never_seen_scores_zero_rather_than_guessing(corpus,
                                                                        tmp_path):
    """
    A document with none of the model's vocabulary gets zero of every factor,
    and that is the right answer rather than a failure: the best non-negative
    reconstruction of an empty row is an empty row, and the model genuinely has
    nothing to say about words it never saw.

    Worth contrasting with LDA, which returns an even split over topics for the
    same input -- because its answer has to be a distribution, so "no evidence"
    comes out as "equally likely to be anything" instead of "nothing".
    """
    fit(corpus, tmp_path)

    strange = tmp_path / "strange.csv"
    strange.write_text("text_id,text\nx1,zzz qqq wumpus\n", encoding="utf-8")
    out = apply_nmf_model(
        model_json=tmp_path / "model.json", csv_path=strange,
        text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "strange_out.csv", overwrite_existing=True)

    row = list(read(out).values())[0]
    values = [float(row[k]) for k in row if k.startswith("Factor_")]
    assert all(v == v for v in values), "NaN in the scores"
    assert values == [0.0] * len(values), values


def test_applying_a_model_does_not_poison_a_later_fit(corpus, tmp_path):
    """
    Found in review, and reproduced before it was fixed. `apply` rebuilds the
    matrix from the *model's* vocabulary, and writes that vocabulary out as a
    frequency list for `build_doc_term_matrix` to read. That file is
    deliberately not provenance-recorded -- it describes the model, not this
    corpus -- and it used to land in the same `matrix/` folder the fit uses.

    So: fit on corpus A, apply that model to corpus B, then fit on B. The
    second fit found the frozen list, saw no record to disagree with, reused
    it, and produced a model of B's documents over *A's words*. No error, no
    warning, a full results table of the wrong thing.
    """
    import json

    first = fit(corpus, tmp_path, out_features_csv=tmp_path / "home" / "fit.csv",
                out_model_json=tmp_path / "home" / "m.json")

    other = tmp_path / "other.csv"
    other.write_text(
        "text_id,text\n" + "".join(f"o{i},zebra yak xerus walrus\n"
                                   for i in range(12)), encoding="utf-8")

    apply_nmf_model(model_json=tmp_path / "home" / "m.json", csv_path=other,
          text_cols=["text"], id_cols=["text_id"],
          out_features_csv=tmp_path / "home" / "applied.csv",
          overwrite_existing=True)

    # now fit on the *other* corpus, with `overwrite_existing` left at its
    # default -- which is the whole point, because that is when a stale file
    # gets reused. it has to be the other corpus: the frozen list holds the
    # model's vocabulary, so re-fitting the corpus the model came from looks
    # identical whether the bug is there or not.
    fit(other, tmp_path, out_features_csv=tmp_path / "home" / "fit2.csv",
        out_model_json=tmp_path / "home" / "m2.json", overwrite_existing=False)

    terms = json.loads(
        (tmp_path / "home" / "m2.json").read_text(encoding="utf-8"))["matrix"]["terms"]
    assert "zebra" in terms, (
        "the second fit modeled the applied model's vocabulary instead of its "
        f"own corpus's: {terms[:6]}")
    assert not set(terms) & set(FOOD + WORK), terms
    assert first.is_file()


# ---------------------------------------------------------------------------
# What it writes
# ---------------------------------------------------------------------------

def test_the_weights_are_weights_and_not_proportions(corpus, tmp_path):
    """
    The difference from LDA that matters downstream. LDA's topic shares sum to
    one, which makes them compositional and awkward in a regression. NMF's do
    not: a document can be a lot of one factor without being less of another.
    """
    totals = []
    for row in read(fit(corpus, tmp_path)).values():
        values = [float(v) for k, v in row.items() if k.startswith("Factor_")]
        assert all(v >= 0 for v in values), "non-negative is the whole method"
        totals.append(sum(values))

    assert not all(t == pytest.approx(1.0, abs=1e-3) for t in totals), (
        "these came out summing to one, which would mean they are shares of a "
        "whole -- that is LDA's contract, not this one")


def test_the_two_planted_families_come_out_as_the_two_topics(corpus, tmp_path):
    """The test a person would run by hand: are the topics the right topics?"""
    out = fit(corpus, tmp_path)
    rows = list(csv.DictReader(
        open(out.with_name(f"{out.stem}_top_terms.csv"), encoding="utf-8-sig")))

    leading = {}
    for row in rows:
        if int(row["rank"]) <= 6:
            leading.setdefault(row["topic"], []).append(row["term"])

    families = {name: ("food" if all(t in FOOD for t in terms)
                       else "work" if all(t in WORK for t in terms) else "mixed")
                for name, terms in leading.items()}
    assert set(families.values()) == {"food", "work"}, families


def test_the_loadings_are_in_the_shape_the_word_clouds_already_read(corpus,
                                                                   tmp_path):
    """`figures.wordclouds.theme_wordclouds` reads `term`, an optional `pos`,
    then one column per topic. Writing that shape is what lets every topic
    model share one word-cloud step instead of having its own."""
    out = fit(corpus, tmp_path)
    header = open(out.with_name(f"{out.stem}_loadings.csv"),
                  encoding="utf-8-sig").readline().strip().split(",")
    assert header[0] == "term"
    assert header[1:] == ["Factor_1", "Factor_2"]


def test_the_matrix_it_built_sits_beside_the_results(corpus, tmp_path):
    out = fit(corpus, tmp_path)
    matrix = matrix_of(out)
    assert (matrix / "dtm.csv").is_file()
    assert (matrix / "freq_list.csv").is_file()


def test_two_runs_agree_without_anybody_setting_a_seed(corpus, tmp_path):
    """
    NNDSVD starts from the matrix's own singular vectors, so there is nothing
    random to pin down. That is worth a test rather than a comment: it is the
    reason this model file has no `seed` field to forget to record.
    """
    first = fit(corpus, tmp_path, out_features_csv=tmp_path / "a.csv",
                out_model_json=tmp_path / "a.json")
    second = fit(corpus, tmp_path, out_features_csv=tmp_path / "b.csv",
                 out_model_json=tmp_path / "b.json")
    assert first.read_text(encoding="utf-8-sig") == \
        second.read_text(encoding="utf-8-sig")


def test_the_model_records_no_seed_because_there_is_nothing_random():
    import inspect
    assert "seed" not in inspect.signature(topic_model_nmf).parameters


def test_the_scores_come_from_the_saved_model_and_the_right_matrix_row(corpus,
                                                                      tmp_path):
    """
    Fit and apply share one scoring routine, so the roundtrip cannot catch a
    bug inside it -- both sides would be wrong in the same way. Worse, that
    routine scores one document at a time, so even comparing a subset against
    the whole corpus sees nothing: each row is scored in isolation either way.
    (MEM has this same test, for this same reason.)

    So this goes around the outside: take the factors out of the JSON, take one
    document's counts out of the matrix on disk, run inference, and check the
    number that was written. It pins the plumbing -- the right model, the right
    row, the right column -- rather than the arithmetic, which is pinned in
    test_topics_core.py.
    """
    import numpy as np

    from taters.text import _topics

    out = fit(corpus, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    h = np.asarray(model["model"]["components"], dtype=float)

    matrix = list(csv.reader(open(matrix_of(out) / "dtm.csv",
                                  encoding="utf-8-sig")))
    header, first = matrix[0], matrix[1]
    assert header[:2] == ["text_id", "token_count"]

    expected = _topics.infer_nmf(h, [float(c or 0) for c in first[2:]])[0]
    written = read(out)[first[0]]
    for i, value in enumerate(expected, start=1):
        assert float(written[f"Factor_{i}"]) == pytest.approx(value, abs=5e-4)


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------

def test_tfidf_is_the_default_because_it_is_what_nmf_wants(corpus, tmp_path):
    """Without it the first factor goes to whatever words are merely common,
    because those cells hold the most mass to explain."""
    fit(corpus, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert model["matrix"]["weighting"] == "tfidf"


def test_counts_are_allowed_so_the_two_engines_can_share_a_matrix(corpus,
                                                                  tmp_path):
    """Anybody comparing NMF against LDA wants them over the same matrix."""
    fit(corpus, tmp_path, weighting="count")
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert model["matrix"]["weighting"] == "count"


@pytest.mark.parametrize("weighting", ["binary", "relfreq"])
def test_the_weightings_that_would_destroy_the_signal_are_refused(weighting,
                                                                  corpus,
                                                                  tmp_path):
    """
    Both would factor without complaint. `binary` throws away how much a word
    was used; `relfreq` rescales every document to the same total, which is
    the very thing the factorization is trying to see.
    """
    with pytest.raises(ValueError, match="tf-idf or raw counts"):
        fit(corpus, tmp_path, weighting=weighting)


def test_both_divergences_run_and_are_recorded(corpus, tmp_path):
    for loss in ("frobenius", "kullback-leibler"):
        fit(corpus, tmp_path, beta_loss=loss,
            out_features_csv=tmp_path / f"{loss}.csv",
            out_model_json=tmp_path / f"{loss}.json")
        model = json.loads((tmp_path / f"{loss}.json").read_text(encoding="utf-8"))
        assert model["model"]["beta_loss"] == loss


def test_a_model_from_a_newer_taters_is_refused(corpus, tmp_path):
    fit(corpus, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    model["format"] = MODEL_FORMAT + 1
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ValueError, match="format"):
        apply_nmf_model(model_json=newer, csv_path=corpus, text_cols=["text"],
                        out_features_csv=tmp_path / "x.csv",
                        overwrite_existing=True)


def test_a_model_of_the_wrong_kind_is_refused_by_name(corpus, tmp_path):
    """MEM models and LDA models are both `.json` in the same library folder."""
    fit(corpus, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    model["kind"] = "taters-mem-model"
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ValueError, match=MODEL_KIND):
        apply_nmf_model(model_json=wrong, csv_path=corpus, text_cols=["text"],
                        out_features_csv=tmp_path / "x.csv",
                        overwrite_existing=True)


def test_a_model_whose_topics_and_vocabulary_disagree_is_refused(corpus,
                                                                 tmp_path):
    """A truncated or hand-edited model would otherwise score the wrong terms,
    silently, because numpy is happy to broadcast almost anything."""
    fit(corpus, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    model["model"]["components"] = [row[:-3] for row in model["model"]["components"]]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ValueError, match="disagree in size"):
        apply_nmf_model(model_json=broken, csv_path=corpus, text_cols=["text"],
                        out_features_csv=tmp_path / "x.csv",
                        overwrite_existing=True)


# ---------------------------------------------------------------------------
# The model file as an instrument
# ---------------------------------------------------------------------------

def test_the_model_carries_everything_needed_to_rebuild_the_instrument(corpus,
                                                                      tmp_path):
    fit(corpus, tmp_path)
    model = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))

    assert model["kind"] == MODEL_KIND
    assert model["format"] == MODEL_FORMAT
    assert set(model["text"]) >= {"lemmatize", "pos_tagged", "engine",
                                  "tokenizer", "stanza_lang", "keep_punctuation"}
    assert set(model["matrix"]) >= {"weighting", "terms", "idf", "rounding"}
    assert set(model["model"]) >= {"topics", "components", "beta_loss", "iters"}
    assert len(model["model"]["components"]) == 2


def test_the_model_does_not_record_where_it_happened_to_run(corpus, tmp_path):
    """`device` is a runtime choice, not part of the instrument. Recording it
    would make one model fitted on a GPU and the same model fitted on a CPU
    compare as two different measurements."""
    fit(corpus, tmp_path)
    text = (tmp_path / "model.json").read_text(encoding="utf-8")
    assert '"device"' not in text


def test_applying_takes_no_vocabulary_settings_at_all():
    """
    The instrument principle, enforced on the signature. If apply accepted
    `vocab_top_n` or `lemmatize`, somebody would pass one, and the model would
    be measuring something other than what it was fitted to measure.
    """
    import inspect

    taken = set(inspect.signature(apply_nmf_model).parameters)
    forbidden = {"lemmatize", "pos_tagged", "engine", "tokenizer",
                 "keep_punctuation", "weighting", "vocab_top_n", "vocab_rule",
                 "vocab_min_freq", "vocab_min_obs_pct", "vocab_rank_by",
                 "ngram_n", "min_freq", "min_obs_pct", "n_topics",
                 "beta_loss", "iters", "tol"}
    assert not (taken & forbidden), sorted(taken & forbidden)


def test_the_step_is_registered_where_the_rest_of_taters_can_find_it():
    """A measure nobody can reach from the catalog is a measure nobody has."""
    from taters.helpers.feature_columns import overlaps
    from taters.text.topic_model_lda import FEATURE_COLUMNS as lda
    from taters.text.topic_model_mem import FEATURE_COLUMNS as mem
    from taters.text.topic_model_nmf import FEATURE_COLUMNS

    assert FEATURE_COLUMNS.patterns == ("Factor_{n}",)
    assert not overlaps([FEATURE_COLUMNS, lda, mem]), (
        "the three topic models must not write the same column names")
