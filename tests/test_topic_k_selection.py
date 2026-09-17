"""
Choosing how many topics, the same way for all three models.

Why this file exists
--------------------
MEM could choose its own component count and LDA and NMF could not. The one
tool that helped was a *separate* checklist step that fitted nothing, wrote a
curve, and left you to read a number off a chart and retype it -- over a
vocabulary it had built independently of the model you then ran, so you could
choose a count for a corpus you had not modeled and nothing would notice.

Now ``0`` means "choose for me" on all three, over that model's own matrix, at
the moment of fitting. What is pinned here is that the three agree: the same
rules, the same evidence files, the same refusals.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

#: Four disjoint word families, so the planted answer is four. Two is wrong by
#: blending and eight is wrong by fragmenting, which are the two failures the
#: paired rules are meant to catch -- one each.
FAMILIES = [
    "bread butter cheese dinner kitchen recipe pasta soup salad pudding".split(),
    "office meeting deadline manager project report invoice budget memo client".split(),
    "river forest mountain valley glacier meadow canyon prairie tundra lagoon".split(),
    "guitar drums piano violin trumpet cello banjo oboe harp mandolin".split(),
]

LOOSE = dict(min_freq=1, min_obs_pct=0.0, min_token_count=1,
             vocab_min_freq=0, vocab_min_obs_pct=0, vocab_top_n=40)


@pytest.fixture
def corpus(tmp_path) -> Path:
    rng = random.Random(0)
    path = tmp_path / "corpus.csv"
    rows = [("text_id", "text")]
    for i in range(80):
        rows.append((f"d{i:02d}",
                     " ".join(rng.choice(FAMILIES[i % 4]) for _ in range(25))))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path


def fit(model: str, corpus: Path, out: Path, **kwargs):
    import importlib

    module = importlib.import_module(f"taters.text.topic_model_{model}")
    count = "n_components" if model == "mem" else "n_topics"
    settings = dict(csv_path=corpus, text_cols=["text"], id_cols=["text_id"],
                    out_features_csv=out, overwrite_existing=True, **LOOSE)
    settings.setdefault(count, 0)
    settings.update(kwargs)
    return Path(getattr(module, f"topic_model_{model}")(**settings))


def columns(path: Path) -> list:
    header = open(path, encoding="utf-8-sig").readline().strip().split(",")
    return [c for c in header if c.split("_")[0] in ("Topic", "Factor", "Theme")]


# ---------------------------------------------------------------------------
# The rules, on all three models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["mem", "lda", "nmf"])
@pytest.mark.parametrize("rule", ["coherence", "coherence_exclusivity"])
def test_a_count_of_zero_chooses_one_and_leaves_the_evidence(model, rule,
                                                             corpus, tmp_path):
    """
    The whole feature in one test: ask for no particular count, get a model
    with some, and get the table and pictures that say why -- beside the
    results rather than in another step's folder.
    """
    out = fit(model, corpus, tmp_path / f"{model}.csv",
              k_selection=rule, k_values="2,4,8", top_terms=4)

    picked = columns(out)
    assert 2 <= len(picked) <= 8

    stem = tmp_path / f"{model}_k_selection"
    rows = list(csv.DictReader(open(stem.with_suffix(".csv"),
                                    encoding="utf-8-sig")))
    assert [int(r["n_topics"]) for r in rows] == [2, 4, 8]
    # every metric, whichever rule is deciding -- the point of the table is
    # that somebody can disagree with the rule
    for row in rows:
        for column in ("coherence", "exclusivity", "balanced", "worst_topic"):
            assert row[column] not in ("", None), column
    chosen = [int(r["n_topics"]) for r in rows if r["chosen"]]
    assert chosen == [len(picked)], "the table and the model disagree about k"

    assert stem.with_suffix(".png").exists()
    assert (tmp_path / f"{model}_k_selection_report.md").exists()
    tradeoff = tmp_path / f"{model}_k_tradeoff.png"
    assert tradeoff.exists() is (rule == "coherence_exclusivity"), (
        "the two-scale chart belongs to the rule that balances two scales")


@pytest.mark.parametrize("model", ["lda", "nmf"])
def test_the_generative_models_find_the_planted_number_of_families(model, corpus,
                                                                   tmp_path):
    """Constructed so the answer is not a matter of taste: four disjoint word
    families, so four topics is right and two and eight are both wrong."""
    out = fit(model, corpus, tmp_path / f"{model}.csv",
              k_selection="coherence_exclusivity", k_values="2,4,8",
              top_terms=4)
    assert len(columns(out)) == 4


def test_an_explicit_count_skips_the_sweep_entirely(corpus, tmp_path):
    """The expensive path is opt-in. A number means that number, and nothing
    is fitted to find out."""
    out = fit("lda", corpus, tmp_path / "lda.csv", n_topics=3)
    assert len(columns(out)) == 3
    assert not list(tmp_path.glob("*_k_selection*")), \
        "a sweep ran even though the count was given"


# ---------------------------------------------------------------------------
# MEM keeps its own two rules, and gains the other two
# ---------------------------------------------------------------------------

def test_mem_defaults_to_parallel_analysis(corpus, tmp_path):
    """
    The rule that works out the noise floor for *this* matrix rather than
    assuming one. It matters because a document-term matrix is wide: at 938
    documents by 515 terms, chance alone produces eigenvalues up to about 3,
    so a fixed cutoff of 1.5 keeps 122 themes where parallel analysis keeps
    82 -- and most of that difference is noise.
    """
    out = fit("mem", corpus, tmp_path / "mem.csv")
    model = json.loads((tmp_path / "mem_model.json").read_text(encoding="utf-8"))
    retention = model["model"]["retention"]
    assert retention["rule"] == "parallel"
    assert len(retention["thresholds"]) == len(retention["unrotated_eigenvalues"])
    assert columns(out), "no themes at all"


def test_the_kaiser_cutoff_is_the_users_to_set(corpus, tmp_path):
    """
    The textbook 1.0 keeps every eigenvalue above one feature's worth of
    variance, which on a wide vocabulary is most of them. Raising it is the
    whole point of exposing it, so it has to actually bite.
    """
    low = fit("mem", corpus, tmp_path / "low.csv", k_selection="kaiser",
              kaiser_cutoff=0.5)
    high = fit("mem", corpus, tmp_path / "high.csv", k_selection="kaiser",
               kaiser_cutoff=8.0)
    assert len(columns(low)) > len(columns(high))

    # and it is inert under any other rule, which is what the options screen
    # promises by only showing it for Kaiser
    other = fit("mem", corpus, tmp_path / "other.csv", k_selection="parallel",
                kaiser_cutoff=8.0)
    assert len(columns(other)) == len(columns(
        fit("mem", corpus, tmp_path / "plain.csv", k_selection="parallel")))


def test_mem_records_which_rule_chose_and_scores_only_the_positive_pole(corpus,
                                                                        tmp_path):
    """
    A MEM theme is bipolar: a term loading -0.7 defines its other end as
    strongly as +0.7 defines this one. Scoring both ends together would mix
    words that anti-correlate by construction, so every theme would score
    badly and the comparison across counts would be noise. Only the positive
    pole counts, and the loadings still carry both.
    """
    out = fit("mem", corpus, tmp_path / "mem.csv",
              k_selection="coherence_exclusivity", k_values="2,4", top_terms=4)
    model = json.loads((tmp_path / "mem_model.json").read_text(encoding="utf-8"))
    assert model["model"]["retention"]["rule"] == "coherence_exclusivity"

    loadings = list(csv.DictReader(open(tmp_path / "mem_loadings.csv",
                                        encoding="utf-8-sig")))
    first = [float(r["Theme_1"]) for r in loadings]
    assert min(first) < 0 < max(first), \
        "the loadings should still carry both poles; only the scoring is one-sided"
    assert columns(out)


def test_mem_scores_the_words_it_really_modeled_when_terms_get_dropped(tmp_path):
    """
    The likeliest bug in this whole feature, and it is silent.

    MEM drops constant columns before decomposing, so its loadings are indexed
    over the *kept* terms while the co-occurrence counts are indexed over
    every term. Put a word in every document exactly once and the two lists
    stop lining up: score without mapping back and every theme is judged on
    words it does not actually load on, with no error anywhere and a plausible
    number at the end.

    So: the words the sweep reports for the winning count have to be the words
    the finished model actually loads positively on.
    """
    rng = random.Random(1)
    src = tmp_path / "corpus.csv"
    rows = [("text_id", "text")]
    for i in range(80):
        words = [rng.choice(FAMILIES[i % 4]) for _ in range(25)]
        words.append("ubiquitous")        # same count in every document
        rows.append((f"d{i:02d}", " ".join(words)))
    with src.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    out = fit("mem", src, tmp_path / "mem.csv", k_selection="coherence",
              k_values="2,4", top_terms=3)

    loadings = list(csv.DictReader(open(tmp_path / "mem_loadings.csv",
                                        encoding="utf-8-sig")))
    kept_terms = {r["term"] for r in loadings}
    assert "ubiquitous" not in kept_terms, (
        "the constant term should have been dropped; the premise is gone")

    report = (tmp_path / "mem_k_selection_report.md").read_text(encoding="utf-8")
    chosen = len(columns(out))
    section = report.split(f"### {chosen} topics")[1].split("###")[0]
    listed = [line.split(".", 1)[1] for line in section.splitlines()
              if line[:1].isdigit()]
    assert listed, "the report named no words at all"
    reported = {w.strip() for line in listed for w in line.split(",")}
    assert reported <= kept_terms, (
        f"scored words the model does not carry: {sorted(reported - kept_terms)}")

    # and every theme's reported words are the ones it actually loads on --
    # by *signed* loading, which is what "the positive pole" means. Ranking by
    # magnitude instead would let a strong negative displace a weaker positive,
    # mixing two ends that anti-correlate by construction. (On this corpus the
    # two rankings happen to agree, because varimax gives each theme a positive
    # block bigger than `top_terms`; the assertion is the guarantee, not a
    # difference you can usually see.)
    for theme, line in zip(columns(out), listed):
        ranked = sorted(loadings, key=lambda r: -float(r[theme]))
        best = {r["term"] for r in ranked[:3]}
        got = {w.strip() for w in line.split(",")}
        assert got == best, (
            f"{theme} was scored on {sorted(got)} but loads on {sorted(best)}")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_the_balanced_rule_will_not_run_on_umass(corpus, tmp_path):
    with pytest.raises(ValueError, match="needs metric='npmi'"):
        fit("lda", corpus, tmp_path / "lda.csv",
            k_selection="coherence_exclusivity", coherence_metric="umass",
            k_values="2,4")


@pytest.mark.parametrize("model", ["mem", "lda", "nmf"])
def test_counts_too_big_for_the_corpus_are_skipped_not_fatal(model, corpus,
                                                             tmp_path):
    """Skipping is the honest answer for a sweep: the run has already paid for
    the matrix, and the counts that *do* fit still have something to say."""
    with pytest.warns(UserWarning, match="skipped"):
        out = fit(model, corpus, tmp_path / f"{model}.csv",
                  k_selection="coherence", k_values="2,4,500", top_terms=4)
    rows = list(csv.DictReader(open(tmp_path / f"{model}_k_selection.csv",
                                    encoding="utf-8-sig")))
    assert [int(r["n_topics"]) for r in rows] == [2, 4]
    assert columns(out)
    report = (tmp_path / f"{model}_k_selection_report.md").read_text(encoding="utf-8")
    assert "500" in report and "skipped" in report
