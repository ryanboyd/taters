"""
VADER sentiment as a feature extractor.

VADER itself is somebody else's tested code, so what is under test here is the
wiring: that the four scores arrive under names that survive a merge, that a
custom lexicon actually changes the answer (and a missing one is refused where
the user can see it), that workers and the inline path agree, and that the
step composes into a preset like every other text measure.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("vaderSentiment")

from taters.text.analyze_sentiment_vader import (METRICS,  # noqa: E402
                                                 analyze_sentiment_vader)

SENTENCES = [
    ("glad", "I love this, it is absolutely wonderful!!!"),
    ("sad", "This is terrible and I hate every part of it."),
    ("flat", "The meeting is scheduled for three o'clock."),
]


def _corpus(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text"])
        for pid, text in SENTENCES:
            w.writerow([pid, text])
    return path


def _scores(out: Path) -> dict:
    with Path(out).open(encoding="utf-8-sig", newline="") as fh:
        return {r["text_id"]: r for r in csv.DictReader(fh)}


def _run(tmp_path, name="out.csv", **kw) -> dict:
    return _scores(analyze_sentiment_vader(
        csv_path=_corpus(tmp_path / "in.csv"), text_cols=["text"],
        id_cols=["pid"], out_features_csv=tmp_path / name,
        overwrite_existing=True, **kw))


# ---------------------------------------------------------------------------
# The measure
# ---------------------------------------------------------------------------

def test_the_four_scores_arrive_under_names_that_survive_a_merge(tmp_path):
    """
    A feature table gets joined beside a hundred other columns, where bare
    `neg` and `pos` would be anybody's guess. The prefix is the whole reason
    these are not named the way VADER names them internally.
    """
    rows = _run(tmp_path)
    assert list(rows["glad"]) == ["text_id", *METRICS]
    assert METRICS == ("vader_neg", "vader_neu", "vader_pos", "vader_compound")


def test_the_bands_sum_to_one_and_compound_runs_from_minus_one_to_one(tmp_path):
    """The contract VADER's own documentation states. Worth pinning because
    it is what makes the three bands comparable across texts of any length."""
    for row in _run(tmp_path).values():
        bands = sum(float(row[f"vader_{k}"]) for k in ("neg", "neu", "pos"))
        assert bands == pytest.approx(1.0, abs=0.01), row
        assert -1.0 <= float(row["vader_compound"]) <= 1.0


def test_praise_scores_positive_and_abuse_negative(tmp_path):
    """
    A sanity check on the wiring rather than on VADER: if the text and the
    scores ever came apart -- rows written in a different order from the one
    they were scored in, say -- this is what would catch it.
    """
    rows = _run(tmp_path)
    assert float(rows["glad"]["vader_compound"]) > 0.5
    assert float(rows["sad"]["vader_compound"]) < -0.5
    assert float(rows["flat"]["vader_compound"]) == pytest.approx(0.0, abs=0.05)


def test_rounding_is_honored(tmp_path):
    """Four places by default; a study that wants two says so."""
    rows = _run(tmp_path, rounding=2)
    for row in rows.values():
        for column in METRICS:
            decimals = row[column].split(".")[-1] if "." in row[column] else ""
            assert len(decimals) <= 2, (column, row[column])


# ---------------------------------------------------------------------------
# The parameters VADER itself takes
# ---------------------------------------------------------------------------

def test_a_custom_lexicon_changes_the_answer(tmp_path):
    """
    Swapping the lexicon is the one substantive knob VADER has, and the
    reason to expose it: a researcher with a domain lexicon wants their
    valences, not the defaults. If the parameter were accepted and ignored,
    the scores would look perfectly plausible and be the stock ones.
    """
    lexicon = tmp_path / "mine.txt"
    # VADER's format: token, mean valence, standard deviation, ratings
    lexicon.write_text("terrible\t3.9\t0.5\t[3, 4, 4, 4]\n"
                       "hate\t2.7\t0.6\t[3, 2, 3, 3]\n", encoding="utf-8")

    stock = _run(tmp_path, name="stock.csv")
    mine = _run(tmp_path, name="mine.csv", lexicon_file=lexicon)
    assert float(stock["sad"]["vader_compound"]) < 0
    assert float(mine["sad"]["vader_compound"]) > 0, \
        "the custom lexicon calls these words positive; it was ignored"


def test_a_lexicon_that_is_not_there_is_refused_by_its_own_name(tmp_path):
    """
    VADER joins a lexicon name onto its *own* package folder, so a relative
    name that does not exist fails with a path inside site-packages:

        No such file or directory: '.../site-packages/vaderSentiment/mine.txt'

    which sends somebody hunting through site-packages for a file they know
    is sitting in their project. Checking first means the refusal names what
    they typed, and which parameter they typed it into.
    """
    with pytest.raises(FileNotFoundError) as err:
        _run(tmp_path, lexicon_file="not-a-real-lexicon.txt")
    message = str(err.value)
    assert "not-a-real-lexicon.txt" in message
    assert "lexicon_file" in message, "say which parameter was wrong"
    assert "site-packages" not in message, "that is not where they put it"


# ---------------------------------------------------------------------------
# The wiring the rest of Taters relies on
# ---------------------------------------------------------------------------

def test_workers_and_the_inline_path_give_the_same_numbers(tmp_path):
    """
    The analyzer holds a parsed lexicon and is built per worker rather than
    pickled. A worker that silently built a *different* analyzer -- ignoring
    the custom lexicon in its initargs, say -- would only show up as a
    disagreement like this one.
    """
    lexicon = tmp_path / "mine.txt"
    lexicon.write_text("meeting\t2.5\t0.5\t[2, 3, 3, 2]\n", encoding="utf-8")
    inline = _run(tmp_path, name="one.csv", workers=1, lexicon_file=lexicon)
    pooled = _run(tmp_path, name="many.csv", workers=4, lexicon_file=lexicon)
    assert inline == pooled


def test_the_step_composes_into_a_preset_like_any_other_text_measure(tmp_path):
    """It is a feature table with a text input, so it has to reach the
    checklist and compose with the shared text-input block."""
    from preset_checks import check_preset

    from taters.ui import compose as comp
    from taters.ui import recipes as rec

    recipe = rec.by_id("sentiment_vader")
    assert recipe.feature_table and recipe.text_input
    assert recipe.user_facing, "it is nothing if nobody can pick it"

    preset = comp.compose(["sentiment_vader"], providers={}, overrides={},
                          var_values={}, name="vibes", file_type="csv",
                          root_dir=None, source="csv",
                          input_path=str(_corpus(tmp_path / "in.csv")),
                          text_cols=["text"], id_cols=["pid"],
                          text_mode="concat", group_by=(), delimiter=",",
                          level="row", model_plans=())
    assert check_preset(preset) == []
    step = next(s for s in preset["steps"]
                if s["call"].endswith("analyze_sentiment_vader"))
    assert step["with"]["out_features_csv"].endswith("sentiment_vader.csv")
