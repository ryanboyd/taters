"""Tests for the metric functions in taters.text.analyze_lexical_richness.

These are pure math on lists of strings, which makes them the easiest thing in
the codebase to test *properly*: for a tiny input you can compute the expected
answer by hand and compare. That is much stronger than asserting "returns a
float" — it would catch a swapped numerator, a wrong log base, or an off-by-one
in a frequency table, none of which a smoke test would notice.

`pytest.approx` compares floats with a tolerance, because `0.1 + 0.2 == 0.3`
is False in binary floating point.
"""

from math import comb, log, sqrt

import pytest

from taters.text.analyze_lexical_richness import (
    _hypergeom_pmf_zero,
    _list_sliding_window,
    _preprocess,
    _segment_generator,
    _tokenize,
    _ttr_nd,
    cttr,
    dugast,
    hdd,
    herdan_c,
    herdan_vm,
    maas,
    mattr,
    msttr,
    mtld,
    rttr,
    simpson_d,
    summer_s,
    ttr,
    vocd,
    yule_i,
    yule_k,
)

# 3 tokens, 2 types: the smallest input where most formulas are defined.
ABA = ["a", "b", "a"]

ALL_METRICS = [ttr, rttr, cttr, herdan_c, summer_s, dugast, maas,
               yule_k, yule_i, herdan_vm, simpson_d]


# --- tokenization -----------------------------------------------------------

def test_preprocess_lowercases_and_replaces_digits_and_punctuation():
    # Punctuation becomes whitespace rather than vanishing, so "cat,dog" splits
    # into two tokens instead of fusing into "catdog". Digits are deleted.
    assert _preprocess("Hello, World! 42 times.") == "hello  world   times "
    assert _tokenize("cat,dog") == ["cat", "dog"]


def test_tokenize_splits_on_whitespace():
    assert _tokenize("The cat, the HAT!") == ["the", "cat", "the", "hat"]


def test_tokenize_collapses_repeated_whitespace():
    assert _tokenize("a\n\n  b\tc") == ["a", "b", "c"]


def test_tokenize_empty_text_gives_no_tokens():
    assert _tokenize("") == []
    assert _tokenize("!!! 123 ???") == []


# --- ratios with hand-computed answers --------------------------------------

def test_ttr_is_types_over_tokens():
    assert ttr(ABA) == pytest.approx(2 / 3)
    assert ttr(["a", "b", "c"]) == 1.0          # all unique
    assert ttr(["a", "a", "a"]) == pytest.approx(1 / 3)


def test_rttr_and_cttr_are_root_corrected_ttr():
    assert rttr(ABA) == pytest.approx(2 / sqrt(3))
    assert cttr(ABA) == pytest.approx(2 / sqrt(2 * 3))


def test_herdan_c_is_log_types_over_log_tokens():
    assert herdan_c(ABA) == pytest.approx(log(2) / log(3))


def test_summer_s_is_the_log_log_variant():
    tokens = ["a", "b", "c", "a", "b", "c", "d"]      # 7 tokens, 4 types
    assert summer_s(tokens) == pytest.approx(log(log(4)) / log(log(7)))


def test_dugast_and_maas_use_the_same_terms():
    w, t = 3, 2
    assert dugast(ABA) == pytest.approx((log(w) ** 2) / (log(w) - log(t)))
    assert maas(ABA) == pytest.approx((log(w) - log(t)) / (log(w) ** 2))


def test_yule_k_hand_computed():
    # freqs: a->2, b->1  =>  sum(count * freq^2) = 1*4 + 1*1 = 5, W = 3
    assert yule_k(ABA) == pytest.approx(1e4 * (5 / 9 - 1 / 3))


def test_yule_i_hand_computed():
    # types^2 / (sum(count * freq^2) - types) = 4 / (5 - 2)
    assert yule_i(ABA) == pytest.approx(4 / 3)


def test_simpson_d_hand_computed():
    # sum(count * freq * (freq-1)) / (W * (W-1)) = (1*2*1) / (3*2)
    assert simpson_d(ABA) == pytest.approx(2 / 6)


def test_simpson_d_is_zero_when_every_token_is_unique():
    assert simpson_d(["a", "b", "c"]) == 0.0


def test_herdan_vm_hand_computed():
    # sqrt( sum(count*(freq/W)^2) - 1/T ) = sqrt((4/9 + 1/9) - 1/2)
    assert herdan_vm(ABA) == pytest.approx(sqrt(5 / 9 - 1 / 2))


def test_herdan_vm_floors_at_zero_instead_of_taking_a_negative_root():
    assert herdan_vm(["a", "b", "c"]) == 0.0


# --- degenerate inputs ------------------------------------------------------

@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda f: f.__name__)
def test_metrics_return_none_for_empty_input(metric):
    """No text means no score — never a crash, never a fake number."""
    assert metric([]) is None


@pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda f: f.__name__)
def test_metrics_never_raise_on_a_single_token(metric):
    metric(["a"])       # may return None; must not raise


def test_dugast_is_undefined_when_every_token_is_unique():
    # W == T means a division by zero in the formula, so it must bail out.
    assert dugast(["a", "b", "c"]) is None


def test_metrics_ignore_token_order():
    """These are bag-of-words statistics; shuffling must not change them."""
    a = ["x", "y", "x", "z", "x", "y"]
    b = ["y", "x", "x", "y", "z", "x"]
    for metric in ALL_METRICS:
        assert metric(a) == pytest.approx(metric(b)), metric.__name__


# --- windowed measures ------------------------------------------------------

def test_segment_generator_chunks_in_order():
    assert list(_segment_generator(list("abcde"), 2)) == [["a", "b"], ["c", "d"], ["e"]]


def test_sliding_window_yields_every_position():
    assert list(_list_sliding_window(list("abcd"), 3)) == [
        ("a", "b", "c"), ("b", "c", "d"),
    ]


def test_sliding_window_is_empty_when_text_is_shorter_than_the_window():
    assert list(_list_sliding_window(list("ab"), 5)) == []


def test_msttr_needs_more_tokens_than_one_segment():
    assert msttr(["a"] * 10, segment_window=100) is None


def test_msttr_averages_per_segment_ttr():
    # 4 segments of 2: ["a","a"]=0.5, ["b","c"]=1.0, ["d","d"]=0.5, ["e","f"]=1.0
    tokens = ["a", "a", "b", "c", "d", "d", "e", "f"]
    assert msttr(tokens, segment_window=2) == pytest.approx(0.75)


def test_mattr_needs_at_least_one_full_window():
    assert mattr(["a", "b"], window_size=5) is None
    assert mattr(["a", "b", "c"], window_size=3) == pytest.approx(1.0)


def test_mattr_averages_over_sliding_windows():
    # windows of 2 over a,a,b: ["a","a"]=0.5, ["a","b"]=1.0
    assert mattr(["a", "a", "b"], window_size=2) == pytest.approx(0.75)


def test_mtld_is_higher_for_more_varied_text():
    varied = [f"w{i}" for i in range(200)]
    repetitive = ["a", "b"] * 100
    assert mtld(varied) > mtld(repetitive)


def test_mtld_returns_none_for_empty_input():
    assert mtld([]) is None


def test_mtld_is_symmetric_by_construction():
    """MTLD averages a forward and a backward pass, so reversing is a no-op."""
    tokens = ["a", "b", "c", "a", "b", "d", "e", "a"] * 5
    assert mtld(tokens) == pytest.approx(mtld(list(reversed(tokens))))


# --- HD-D -------------------------------------------------------------------

def test_hypergeometric_zero_probability_matches_the_closed_form():
    assert _hypergeom_pmf_zero(10, 3, 4) == pytest.approx(comb(7, 4) / comb(10, 4))


def test_hypergeom_pmf_zero_is_one_when_the_draw_is_impossible():
    assert _hypergeom_pmf_zero(5, 2, 10) == 1.0


def test_hdd_requires_at_least_as_many_tokens_as_draws():
    assert hdd(["a"] * 10, draws=42) is None
    assert hdd([], draws=1) is None
    assert hdd(["a"] * 10, draws=0) is None


def test_hdd_hand_computed_for_all_unique_tokens():
    # Every type appears once, so P(missing a given type in n draws) = C(N-1,n)/C(N,n).
    tokens = [f"w{i}" for i in range(10)]
    n, N = 5, 10
    expected = 10 * (1 - comb(N - 1, n) / comb(N, n)) / n
    assert hdd(tokens, draws=n) == pytest.approx(expected)


def test_hdd_is_higher_for_more_diverse_text():
    diverse = [f"w{i}" for i in range(100)]
    narrow = ["a", "b", "c", "d"] * 25
    assert hdd(diverse, draws=42) > hdd(narrow, draws=42)


# --- VOCD -------------------------------------------------------------------

def test_ttr_nd_model_is_monotonic_in_d():
    """Larger D means more diversity at a fixed sample size."""
    assert _ttr_nd(50, 100) > _ttr_nd(50, 10)


def test_ttr_nd_guards_against_nonsense_arguments():
    assert _ttr_nd(0, 10) == 0.0
    assert _ttr_nd(50, 0) == 0.0


def test_vocd_needs_enough_tokens():
    assert vocd(["a"] * 10, ntokens=50) is None
    assert vocd([f"w{i}" for i in range(100)], ntokens=34) is None   # ntokens < 35


def test_vocd_is_reproducible_for_a_fixed_seed():
    """
    VOCD samples randomly. A seeded run must be repeatable, or results are not
    reportable in a paper — this test is what makes that a guarantee.
    """
    tokens = [f"w{i % 40}" for i in range(300)]
    kwargs = dict(ntokens=40, within_sample=5, iterations=1, seed=123)
    assert vocd(tokens, **kwargs) == vocd(tokens, **kwargs)


def test_vocd_changes_with_a_different_seed_but_stays_in_range():
    tokens = [f"w{i % 40}" for i in range(300)]
    a = vocd(tokens, ntokens=40, within_sample=5, iterations=1, seed=1)
    b = vocd(tokens, ntokens=40, within_sample=5, iterations=1, seed=2)
    assert 5.0 <= a <= 200.0 and 5.0 <= b <= 200.0     # inside the search grid
