"""
Tests for `taters.text.analyze_with_norms`.

The thing being guarded here is not "does it run" but "is it the right kind of
average". A norm set and a content-coding dictionary are the same file format
holding the same kind of numbers, and scoring one with the other's arithmetic
returns a plausible float in plausible units that is simply wrong -- there is
nothing about the output to notice. So the two scorers are checked against each
other on identical input and required to disagree, and the mean is checked
against arithmetic anybody can do on paper rather than against last week's
output.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from taters.text.analyze_with_norms import analyze_with_norms

# cat 5.0, dog 3.0, "ice cream" 4.0. Emo is deliberately blank for `dog`, so a
# per-category denominator is the only way to get Emo right.
NORMS = """DicTerm,Conc,Emo
cat,5.0,2.0
dog,3.0,
ice cream,4.0,1.5
"""

TEXT = "the cat ate ice cream with a dog"     # 8 words


@pytest.fixture()
def norm_file(tmp_path) -> Path:
    p = tmp_path / "Test Norms.csv"
    p.write_text(NORMS, encoding="utf-8")
    return p


def _score(tmp_path, norm_file, text=TEXT) -> dict:
    src = tmp_path / "texts.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["t1", text])
    out = analyze_with_norms(
        csv_path=src, text_cols=["text"], norm_paths=[norm_file],
        out_features_csv=tmp_path / "out.csv",
        gathered_csv=tmp_path / "gathered.csv",
        overwrite_existing=True, workers=1)
    with Path(out).open(encoding="utf-8-sig", newline="") as fh:
        return next(iter(csv.DictReader(fh)))


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_a_rating_is_the_mean_of_the_words_that_had_one(tmp_path, norm_file):
    """(5.0 + 4.0 + 3.0) / 3. Not a rate, not a sum."""
    row = _score(tmp_path, norm_file)

    assert float(row["Test_Norms__Conc"]) == pytest.approx(4.0, abs=1e-4)


def test_a_multi_word_entry_is_one_observation_not_two(tmp_path, norm_file):
    """
    "ice cream" spans two words and is rated once, so it contributes 4.0 to
    the mean once. Counting it twice would pull the mean to 4.0625 here -- a
    difference small enough to look like rounding and large enough to be wrong.
    """
    row = _score(tmp_path, norm_file)

    assert float(row["Test_Norms__Conc"]) != pytest.approx(4.0625, abs=1e-3)


def test_coverage_counts_words_rather_than_entries(tmp_path, norm_file):
    """
    Three entries matched, but they cover four of the eight words, and what a
    reader needs to know is how much of the *text* was rated.
    """
    row = _score(tmp_path, norm_file)

    assert float(row["Test_Norms__Conc_Coverage"]) == pytest.approx(0.5)


def test_each_rating_gets_its_own_denominator(tmp_path, norm_file):
    """
    `dog` has a concreteness but no emotion rating, so it belongs in one mean
    and not the other. A single shared denominator would put Emo at 2.3333
    instead of 1.75, and understate nothing visibly.
    """
    row = _score(tmp_path, norm_file)

    assert float(row["Test_Norms__Emo"]) == pytest.approx(1.75, abs=1e-4)
    assert float(row["Test_Norms__Emo_Coverage"]) == pytest.approx(0.375)


def test_a_rating_of_zero_is_a_rating_and_not_an_absence(tmp_path):
    """
    The bug this guards: contentcoder drops weight-0 terms on load, because in
    a word-counting dictionary 0 means "not in this category". In a norm set it
    means somebody rated the word and the answer was zero -- a third of the
    Lancaster sensorimotor words have a gustatory strength of exactly 0. Drop
    those and the mean is taken over only the words that taste of something.
    """
    p = tmp_path / "Zeros.csv"
    p.write_text("DicTerm,Taste\ncat,0.0\ndog,0.0\nlemon,4.0\n", encoding="utf-8")

    row = _score(tmp_path, p, text="the cat and the dog and the lemon")

    # (0 + 0 + 4) / 3, not 4 / 1
    assert float(row["Zeros__Taste"]) == pytest.approx(1.3333, abs=1e-3)
    assert float(row["Zeros__Taste_Coverage"]) == pytest.approx(3 / 8, abs=1e-3)


# ---------------------------------------------------------------------------
# Empty is not zero
# ---------------------------------------------------------------------------

def test_a_text_with_no_rated_words_gets_an_empty_cell(tmp_path, norm_file):
    """
    Nought is an answer; no answer is not. A text with no rated words has no
    concreteness, and writing 0.0 would drag down every average that touched
    it -- exactly the rows somebody would then want to filter out and could
    not find.
    """
    row = _score(tmp_path, norm_file, text="nothing here is in the table")

    assert row["Test_Norms__Conc"] == ""
    assert float(row["Test_Norms__Conc_Coverage"]) == 0.0


# ---------------------------------------------------------------------------
# The two scorers must disagree
# ---------------------------------------------------------------------------

def test_the_norm_scorer_and_the_dictionary_scorer_do_not_agree(tmp_path,
                                                                norm_file):
    """
    The whole reason for two libraries and two steps. Handed the same file and
    the same text, a rate and a mean are different numbers; if these ever match
    then one of the two is not doing its job and nothing else here would say so.
    """
    from taters.text.analyze_with_dictionaries import analyze_with_dictionaries

    src = tmp_path / "texts.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["t1", TEXT])

    as_norms = _score(tmp_path, norm_file)
    out = analyze_with_dictionaries(
        csv_path=src, text_cols=["text"], dict_paths=[norm_file],
        out_features_csv=tmp_path / "dict_out.csv",
        gathered_csv=tmp_path / "dict_gathered.csv",
        overwrite_existing=True, workers=1)
    with Path(out).open(encoding="utf-8-sig", newline="") as fh:
        as_dict = next(iter(csv.DictReader(fh)))

    mean = float(as_norms["Test_Norms__Conc"])
    rate = float(as_dict["Test_Norms__Conc"])

    assert mean == pytest.approx(4.0, abs=1e-4)
    assert rate != pytest.approx(mean, abs=0.5)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_a_file_with_an_intercept_row_is_refused_by_name(tmp_path):
    """
    An intercept is a fitted model's constant term. Scoring it as a plain mean
    drops the constant without saying so, so the file is refused rather than
    quietly mis-scored.
    """
    p = tmp_path / "Has Intercept.csv"
    p.write_text("DicTerm,Score\n_intercept,23.2\ncat,5.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="_intercept"):
        analyze_with_norms(
            analysis_csv=_analysis_csv(tmp_path), norm_paths=[p],
            out_features_csv=tmp_path / "out.csv", overwrite_existing=True)


def test_the_shipped_norm_library_is_what_the_norms_step_reads(tmp_path):
    """The kind's folder and the step's asset declaration have to agree, or
    the built-ins are invisible to the only step that can use them."""
    from taters.helpers.library import KINDS

    assert KINDS["norms"].suffixes == (".csv",)
    assert analyze_with_norms.__provenance__["assets"]["norm_paths"] == "norms"


def _analysis_csv(tmp_path) -> Path:
    p = tmp_path / "ready.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["t1", TEXT])
    return p
