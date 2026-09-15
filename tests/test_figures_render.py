"""
Tests for the word-cloud renderer, `taters.figures.render`.

The layout is pure Python and takes a *measurer* -- a function from (text,
size) to (width, height) -- so most of these tests never touch a font: they
hand in arithmetic and check geometry (no two boxes overlap, everything is
inside the canvas, heavier words are bigger, the same seed draws the same
picture). Only the last few need Pillow, and they skip without it.
"""

from __future__ import annotations

import sys

import pytest

from taters.figures.render import (BLUE_HUE, RED_HUE, Layout, _as_pairs,
                                   _overlaps, color_for, fit_budget, layout,
                                   scale_sizes, shade)


def fake_measure(text: str, size: int):
    """A monospace-ish ruler: width grows with the letters, height is the size."""
    return (int(len(text) * size * 0.6), int(size))


def _words(n: int):
    return [(f"w{i:02d}", 1.0 - i / n) for i in range(n)]


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

def test_scale_sizes_is_monotone_in_weight():
    """A heavier word is never drawn smaller than a lighter one: the size is
    the first thing the eye reads, and an inversion would say the opposite
    of the number."""
    weights = [0.05 * i for i in range(1, 21)]
    sizes = scale_sizes(weights, 16, 96)
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_scale_sizes_spans_min_to_max():
    """The strongest word gets the largest type and the weakest the smallest,
    so the whole range of sizes is used whatever the weights' units."""
    weights = [i / 30 for i in range(1, 31)]
    sizes = scale_sizes(weights, 16, 96)
    assert max(sizes) == 96 and min(sizes) == 16


def test_scale_sizes_single_word_gets_max_font():
    assert scale_sizes([0.3], 16, 96) == [96]


def test_scale_sizes_all_equal_weights_share_one_middle_size():
    """Nothing to rank, so nothing is biggest: one size, between the ends."""
    sizes = scale_sizes([0.4] * 12, 16, 96)
    assert len(set(sizes)) == 1
    assert 16 < sizes[0] < 96


def test_scale_sizes_few_words_get_a_floor():
    """Five significant features are a result worth seeing, not a huddle of
    16-pixel words in the middle of an empty picture."""
    sizes = scale_sizes([1.0, 0.5, 0.01], 16, 96)
    assert min(sizes) >= 48


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

def test_positive_weights_are_blue_and_negative_red():
    r, g, b = color_for(1.0, 1.0)
    assert b > r and b > g
    r, g, b = color_for(-1.0, 1.0)
    assert r > g and r > b


def test_shade_darkens_with_strength():
    """Strength is read twice, in size and in ink -- a strong word that had to
    shrink to fit still reads as strong."""
    for hue in (BLUE_HUE, RED_HUE):
        assert sum(shade(hue, 0.9)) < sum(shade(hue, 0.1))


# ---------------------------------------------------------------------------
# Layout, without a font
# ---------------------------------------------------------------------------

def _lay(n=60, width=600, height=400, **kw) -> Layout:
    words = _words(n)
    sizes = scale_sizes([w for _, w in words], 10, 40)
    return layout(words, sizes, fake_measure, width=width, height=height, **kw)


def test_layout_places_no_two_boxes_overlapping():
    lay = _lay()
    boxes = [p.box for p in lay.placed]
    assert len(boxes) >= 40, "the fake canvas should hold most of the words"
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not _overlaps(a, b), (a, b)


def test_layout_keeps_every_box_inside_the_canvas_and_below_the_caption():
    lay = _lay(top_margin=50)
    for p in lay.placed:
        x0, y0, x1, y1 = p.box
        assert x0 >= 0 and y0 >= 50 and x1 <= 600 and y1 <= 400, p


def test_layout_keeps_heavier_words_bigger():
    lay = _lay(n=20, width=800, height=600)
    by_text = {p.text: p for p in lay.placed}
    assert by_text["w00"].size > by_text["w19"].size
    assert by_text["w00"].h > by_text["w19"].h


def test_layout_is_deterministic_under_a_seed():
    """The same words draw the same picture, so a re-run of an unchanged
    analysis does not produce a different-looking figure."""
    a, b = _lay(seed=3), _lay(seed=3)
    assert [p.box for p in a.placed] == [p.box for p in b.placed]
    c = _lay(seed=4)
    assert [p.box for p in a.placed] != [p.box for p in c.placed]


def test_layout_reports_the_words_that_did_not_fit():
    """A word with no room is named, not drawn on top of another or silently
    lost -- the caller says so in the report."""
    words = [("a very long phrase indeed", 1.0), ("another long phrase", 0.9),
             ("yet another one here", 0.8), ("and one more", 0.7), ("last", 0.6)]
    lay = layout(words, [30] * 5, fake_measure, width=120, height=40, min_font=30)
    assert lay.dropped
    assert len(lay.placed) + len(lay.dropped) == 5


def test_layout_shrinks_a_word_before_dropping_it():
    """Too big for the space is a reason to shrink, and only failing that to
    drop; a 40-pixel word on a 60-pixel-high canvas lands smaller."""
    lay = layout([("wide word", 1.0)], [40], fake_measure, width=200, height=60,
                 min_font=10)
    assert not lay.dropped
    assert lay.placed[0].size < 40


def test_phrases_stay_whole():
    """"going to be" is one feature and one word to a cloud; it is never
    split into three."""
    pairs = _as_pairs([("going to be", 0.9), ("  padded  ", 0.5)], 10)
    assert pairs == [("going to be", 0.9), ("padded", 0.5)]


def test_as_pairs_drops_blanks_zeros_and_duplicates():
    pairs = _as_pairs([("", 1.0), ("x", 0.0), ("y", "nan"), ("z", 0.2),
                       ("z", -0.5), ("q", "abc")], 10)
    assert pairs == [("z", -0.5)]


def test_layout_centers_the_cloud_in_the_drawing_area():
    """The finished cloud sits in the middle of the space below the caption,
    wherever the spiral happened to start."""
    lay = _lay(n=10, width=800, height=600, top_margin=100)
    x0 = min(p.x for p in lay.placed)
    x1 = max(p.x + p.w for p in lay.placed)
    y0 = min(p.y for p in lay.placed)
    y1 = max(p.y + p.h for p in lay.placed)
    assert abs((x0 + x1) / 2 - 400) <= 2
    assert abs((y0 + y1) / 2 - 350) <= 2


# ---------------------------------------------------------------------------
# The area budget
# ---------------------------------------------------------------------------

def test_the_budget_shrinks_sizes_that_cannot_fit():
    """Eighty words at up to 96 pixels need more than the canvas has; scaled
    to the budget they all land instead of 36 of them being dropped."""
    words = _words(80)
    sizes = scale_sizes([w for _, w in words], 16, 96)
    fitted = fit_budget(words, sizes, fake_measure, width=400, height=300,
                        min_font=16)
    assert max(fitted) < max(sizes)
    assert min(fitted) >= 16
    # one shared factor for every word the minimum size didn't catch.
    ratios = [f / s for f, s in zip(fitted, sizes) if f > 16]
    assert max(ratios) - min(ratios) < 0.1, "every word scaled by the same factor"


def test_the_budget_grows_a_sparse_cloud_up_to_a_cap():
    """Five words at the weight-scaled sizes sat in the middle of an empty
    picture; they grow, but the largest stops at a quarter of the height."""
    words = _words(5)
    sizes = scale_sizes([w for _, w in words], 16, 96)
    fitted = fit_budget(words, sizes, fake_measure, width=1200, height=800,
                        top_margin=0)
    assert max(fitted) > max(sizes)
    assert max(fitted) <= 800 // 4


# ---------------------------------------------------------------------------
# With Pillow
# ---------------------------------------------------------------------------

def test_render_writes_a_png_of_the_requested_size_that_is_not_blank(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    from taters.figures.render import render_wordcloud

    out = render_wordcloud(_words(30), tmp_path / "c.png", title="thirty words",
                           width=600, height=400)
    assert out.is_file() and not (tmp_path / "c.png.part").exists()
    img = Image.open(out)
    assert img.size == (600, 400)
    dark = sum(1 for px in img.getdata() if px != (255, 255, 255))
    assert dark > 8000, "the words were not drawn"
    assert PIL  # so the linter doesn't think the import is unused


def test_render_without_pillow_raises_a_sentence_that_says_what_to_do(monkeypatch):
    """A missing library gets one plain sentence naming the pip line, and
    callers can ask for it up front to skip the figures rather than fail."""
    from taters.figures import render as r

    monkeypatch.setitem(sys.modules, "PIL", None)
    reason = r.pillow_missing_reason()
    assert reason and "pip install pillow" in reason
    with pytest.raises(ImportError, match="pip install pillow"):
        r.render_wordcloud([("a", 1.0)], "x.png", title="t")
