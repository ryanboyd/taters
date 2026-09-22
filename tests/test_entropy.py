"""
Tests for `taters.text.analyze_entropy`.

Almost all of this is arithmetic with a right answer, so almost all of these
tests check against one rather than against last week's output. Eight equally
common words hold exactly three bits; a text of one repeated word holds none;
Hill numbers at order 2 are Simpson's D inverted, which the lexical-richness
module already computes, so the two are checked against each other rather than
both against a hard-coded number.

The estimator tests are the ones that matter. A bias correction that is subtly
wrong still returns a plausible float, in the right units, correlated with the
right things -- there is nothing about the output to notice. So each one is
driven from samples where the true entropy is known by construction, and
checked to land near it while the uncorrected estimate does not.
"""
from __future__ import annotations

import csv
from math import log2
from pathlib import Path

import pytest

from taters.text.analyze_entropy import (analyze_entropy, berger_parker,
                                         chao1, chao_shen_bits, coverage,
                                         grassberger_bits, hill_number,
                                         miller_madow_bits, nsb_bits,
                                         renyi_bits, shannon_bits, tsallis)


def _rows(path) -> dict:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as fh:
        return {r["text_id"]: r for r in csv.DictReader(fh)}


def _draw(alphabet: int, n: int, seed: int = 0) -> list:
    """`n` tokens drawn uniformly from `alphabet` equally likely types, so
    the true entropy is exactly log2(alphabet)."""
    import random

    rng = random.Random(seed)
    counts = {}
    for _ in range(n):
        k = rng.randrange(alphabet)
        counts[k] = counts.get(k, 0) + 1
    return sorted(counts.values())


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_eight_equally_common_types_hold_exactly_three_bits():
    assert shannon_bits([1] * 8) == pytest.approx(3.0)
    assert shannon_bits([5] * 8) == pytest.approx(3.0), "counts, not proportions"


def test_one_repeated_type_holds_nothing_and_never_reports_minus_zero():
    """`-sum(...)` of all-zero terms is -0.0, which is a real float and an
    ugly thing to find in a results table."""
    value = shannon_bits([12])

    assert value == 0.0
    assert not str(value).startswith("-")


def test_an_empty_text_is_zero_rather_than_an_error():
    assert shannon_bits([]) == 0.0
    assert coverage([]) == 0.0
    assert berger_parker([]) == 0.0


def test_the_effective_number_of_types_is_the_entropy_in_units_people_have():
    """4.2 bits is not a quantity anybody has intuitions about; "18 equally
    common words" is the same number and is."""
    assert hill_number([1] * 8, 1.0) == pytest.approx(8.0)
    assert hill_number([1] * 16, 1.0) == pytest.approx(16.0)


def test_the_orders_of_the_profile_are_the_measures_they_should_be():
    """
    Type-token ratio, Shannon and Simpson are one curve read at three
    places. At q=0 the answer is the type count; at q=infinity it is the
    commonest type's share inverted.
    """
    counts = [1, 1, 2, 6]

    assert hill_number(counts, 0.0) == pytest.approx(4.0), "q=0 is richness"
    assert renyi_bits(counts, 1.0) == pytest.approx(shannon_bits(counts))
    assert hill_number(counts, float("inf")) == pytest.approx(
        1.0 / berger_parker(counts)), "q=inf is dominance, inverted"


def test_order_two_agrees_with_the_simpson_index_we_already_ship():
    """
    Checked against the other module rather than a constant, so if either
    drifts the pair stops agreeing and this says so. Simpson's D is the
    probability two draws match; its inverse is the q=2 Hill number.
    """
    from taters.text.analyze_lexical_richness import simpson_d

    tokens = ["a"] * 6 + ["b"] * 3 + ["c"] * 1
    counts = [6, 3, 1]

    assert hill_number(counts, 2.0) == pytest.approx(1.0 / simpson_d(tokens),
                                                     rel=0.2)


def test_a_flat_distribution_is_perfectly_even_and_a_lopsided_one_is_not():
    flat = shannon_bits([4, 4, 4, 4]) / log2(4)
    lopsided = shannon_bits([97, 1, 1, 1]) / log2(4)

    assert flat == pytest.approx(1.0)
    assert lopsided < 0.2


def test_tsallis_falls_back_to_shannon_at_order_one():
    counts = [5, 3, 2]
    assert tsallis(counts, 1.0) == pytest.approx(shannon_bits(counts) * 0.6931,
                                                 rel=1e-3)


# ---------------------------------------------------------------------------
# The bias corrections -- the part with a right answer nobody can eyeball
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("estimator", [
    shannon_bits, miller_madow_bits, chao_shen_bits, grassberger_bits, nsb_bits])
def test_every_estimator_is_right_when_there_is_enough_text(estimator):
    """
    Well sampled, they must all agree with the truth and with each other.
    An estimator that is wrong here is wrong outright rather than biased.
    """
    counts = [1000] * 64                      # 64 types, 64000 tokens

    assert estimator(counts) == pytest.approx(log2(64), abs=0.01)


def test_the_plain_estimate_is_badly_low_on_a_short_text():
    """
    The reason the corrections are here at all. 200 tokens drawn from 1000
    equally likely types really do hold log2(1000) bits; the plug-in estimate
    says seven and a half, because it can only see what turned up.
    """
    counts = _draw(alphabet=1000, n=200)

    assert shannon_bits(counts) < log2(1000) - 2.0


@pytest.mark.parametrize("estimator,tolerance", [
    (miller_madow_bits, 2.0),      # only knows how many types it saw
    (grassberger_bits, 1.2),
    (chao_shen_bits, 0.3),
    (nsb_bits, 0.5),
])
def test_each_correction_closes_the_gap_it_is_supposed_to(estimator, tolerance):
    """
    Each one is checked against what it claims to do rather than against a
    saved number: closer to the truth than the plug-in estimate, and within
    its own known accuracy. The tolerances differ because the estimators
    genuinely differ -- Miller-Madow only corrects the leading term.
    """
    counts = _draw(alphabet=1000, n=200)
    truth = log2(1000)

    assert abs(estimator(counts) - truth) < abs(shannon_bits(counts) - truth)
    assert estimator(counts) == pytest.approx(truth, abs=tolerance)


def test_no_estimate_can_exceed_the_entropy_of_a_flat_alphabet():
    """
    The bug that shipped in the first draft: NSB summed only over the types
    it had seen and left out the prior weight on the ones it had not, which
    let it return 12 bits for a distribution whose maximum is 9.97.
    """
    counts = _draw(alphabet=1000, n=200)
    ceiling = log2(chao1(counts))

    assert nsb_bits(counts) <= ceiling + 0.5


def test_the_alphabet_nsb_needs_is_estimated_rather_than_guessed():
    """
    NSB is the one estimator with a free parameter: it wants to know how many
    types the text *could* have used, and the answer moves by three bits
    across plausible guesses -- more than any two other estimators differ. So
    the guess is not left to anybody; Chao1 makes it from the data.
    """
    counts = _draw(alphabet=1000, n=200)

    assert nsb_bits(counts, alphabet=len(counts)) < nsb_bits(counts)
    assert nsb_bits(counts, alphabet=50000) > nsb_bits(counts)
    assert nsb_bits(counts) == pytest.approx(log2(1000), abs=0.5)


def test_coverage_says_how_much_of_the_distribution_was_actually_seen():
    """Low coverage is the signal that the numbers above it are estimates.
    Every word appearing once means the text told you almost nothing."""
    assert coverage([1] * 50) == 0.0
    assert coverage([50, 50]) == 1.0


def test_chao1_estimates_the_types_that_never_turned_up():
    """Many singletons and few doubletons mean a lot more vocabulary is out
    there; no singletons mean you have probably seen it all."""
    assert chao1([1] * 20 + [5]) > 21
    assert chao1([10, 10, 10]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Structure, which is a different question from variety
# ---------------------------------------------------------------------------

def test_a_formulaic_text_and_a_varied_one_can_share_a_vocabulary(tmp_path):
    """
    The point of the conditional entropies. Both texts use the same words the
    same number of times, so every single-token measure is identical; one is
    a repeated phrase and the other is not, and only the order-2 measures can
    tell.
    """
    sheet = tmp_path / "t.csv"
    sheet.write_text(
        "id,text\n"
        "formulaic,\"ab ab ab ab ab ab cd cd cd cd cd cd\"\n"
        "varied,\"ab cd ab cd cd ab cd ab ab cd cd ab\"\n",
        encoding="utf-8")
    out = analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                          out_features_csv=tmp_path / "e.csv", verbose=False)
    rows = _rows(out)

    assert rows["formulaic"]["ent_word_shannon_bits"] == \
        rows["varied"]["ent_word_shannon_bits"], "same words, same counts"
    assert float(rows["formulaic"]["ent_word_conditional2_bits"]) < \
        float(rows["varied"]["ent_word_conditional2_bits"]), \
        "the repeated phrase is more predictable and the pair entropy must say so"


def test_compression_rates_repetitive_text_as_cheaper(tmp_path):
    """The crudest estimate of the same thing, and the only one that assumes
    nothing about tokens."""
    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\n"
                     "same,\"" + "hello world " * 40 + "\"\n"
                     "varied,\"" + " ".join(f"w{i}" for i in range(80)) + "\"\n",
                     encoding="utf-8")
    out = analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                          out_features_csv=tmp_path / "e.csv", verbose=False)
    rows = _rows(out)

    for column in ("ent_compress_zlib_bpc", "ent_compress_bz2_bpc",
                   "ent_compress_lzma_bpc"):
        assert float(rows["same"][column]) < float(rows["varied"][column]), column


# ---------------------------------------------------------------------------
# The table it writes
# ---------------------------------------------------------------------------

def test_both_units_are_measured_and_characters_need_no_tokenizer(tmp_path):
    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\none,\"the cat sat on the mat\"\n", encoding="utf-8")
    out = analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                          out_features_csv=tmp_path / "e.csv", verbose=False)
    row = _rows(out)["one"]

    assert float(row["ent_word_tokens"]) == 6
    assert float(row["ent_char_tokens"]) > 6
    assert float(row["ent_char_shannon_bits"]) > 0


def test_the_token_counts_are_bookkeeping_rather_than_predictors():
    """
    Entropy is the most length-sensitive family of measures in common use, so
    the length is exactly what somebody needs to hold constant -- and exactly
    what nobody should regress on by accident.
    """
    declared = analyze_entropy.__provenance__["bookkeeping"]

    assert "ent_word_tokens" in declared
    assert "ent_char_tokens" in declared


def test_every_declared_column_is_actually_written(tmp_path):
    """
    The registry is read by `assemble` and by the collision check, so a
    declaration that does not match the file is worse than none.
    """
    from taters.text.analyze_entropy import FEATURE_COLUMNS

    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\none,\"the cat sat on the mat and the cat ran\"\n",
                     encoding="utf-8")
    out = analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                          out_features_csv=tmp_path / "e.csv", verbose=False)
    written = set(_rows(out)["one"])

    assert set(FEATURE_COLUMNS.names) <= written


def test_how_deep_the_block_entropies_go_is_a_setting(tmp_path):
    """And it rides in the column name, because `block2` and `block4` are not
    the same measure and must not share a column."""
    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\none,\"" + "a b c d " * 20 + "\"\n", encoding="utf-8")
    out = analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                          max_order=4, out_features_csv=tmp_path / "e.csv",
                          verbose=False)
    row = _rows(out)["one"]

    assert "ent_word_block4_bits" in row
    assert "ent_word_conditional4_bits" in row


def test_a_table_that_is_already_there_is_left_alone(tmp_path):
    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\none,\"the cat sat\"\n", encoding="utf-8")
    out = tmp_path / "e.csv"
    out.write_text("already,here\n", encoding="utf-8")
    analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                    out_features_csv=out, verbose=False)

    assert out.read_text(encoding="utf-8").startswith("already")


def test_an_order_below_one_is_refused_by_name(tmp_path):
    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\none,\"the cat sat\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_order"):
        analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                        max_order=0, out_features_csv=tmp_path / "e.csv",
                        verbose=False)


def test_the_conditional_entropy_is_the_block_entropy_minus_the_one_below(tmp_path):
    """
    Pinned as arithmetic rather than as a comparison, because a conditional
    entropy that had quietly become a block entropy would still rank two
    texts the same way and still look like bits.

    H(X_n | X_1..X_{n-1}) = H_n - H_{n-1}: what the next unit still costs once
    you know the ones before it.
    """
    sheet = tmp_path / "t.csv"
    sheet.write_text("id,text\none,\"" + "the cat sat on the mat and a dog ran "
                     * 12 + "\"\n", encoding="utf-8")
    out = analyze_entropy(csv_path=sheet, text_cols=["text"], id_cols=["id"],
                          out_features_csv=tmp_path / "e.csv", verbose=False)
    row = _rows(out)["one"]

    h1 = float(row["ent_word_shannon_bits"])
    h2 = float(row["ent_word_block2_bits"])
    h3 = float(row["ent_word_block3_bits"])
    assert float(row["ent_word_conditional2_bits"]) == pytest.approx(h2 - h1,
                                                                     abs=1e-4)
    assert float(row["ent_word_conditional3_bits"]) == pytest.approx(h3 - h2,
                                                                     abs=1e-4)
    # and the rate is the deepest conditional, not the deepest block
    assert float(row["ent_word_rate_bits"]) == pytest.approx(h3 - h2, abs=1e-4)
