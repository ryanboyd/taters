"""The shaded-text figure: a text with each word on a background colored by
its share in a prediction."""
from __future__ import annotations

import pytest

from taters.figures import highlights as hl


def test_the_background_runs_from_white_to_a_clear_tint_by_sign():
    assert hl.background_for(0.0, 1.0) == (255, 255, 255)
    up, down = hl.background_for(1.0, 1.0), hl.background_for(-1.0, 1.0)
    assert up != down and up != (255, 255, 255)
    r, g, b = up
    assert b > r, "a positive share is blue"
    r, g, b = down
    assert r > b, "a negative share is red"
    faint = hl.background_for(0.05, 1.0)
    assert sum(faint) > sum(up), "a smaller share is paler"


def test_a_highlighted_text_is_written_and_wraps(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    words = [(f"word{i}", (i % 5 - 2) * 0.1) for i in range(120)]
    out = hl.render_highlighted_text(words, tmp_path / "t.png", title="Text t1, predicted 3.2",
                                     legend="blue up, red down", width=600)
    assert out.is_file()
    with Image.open(out) as img:
        w, h = img.size
    assert w == 600 and h > 400, "many words wrap onto many lines"
    short = hl.render_highlighted_text(words[:5], tmp_path / "s.png", title="short", width=600)
    with Image.open(short) as img:
        assert img.size[1] < h


def test_a_very_long_text_is_cut_and_says_so(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    words = [("word", 0.1)] * 2000
    out = hl.render_highlighted_text(words, tmp_path / "long.png", title="long", width=400,
                                     max_lines=5)
    with Image.open(out) as img:
        # the caption band, five lines of text, the "and N more" note, and
        # the margins: nothing like the 2000 words' worth
        from PIL import ImageFont
        from taters.figures.render import DEFAULT_FONT, _Fonts, caption_band
        band = caption_band(_Fonts(ImageFont, DEFAULT_FONT))
        assert img.size[1] < band + 7 * (22 + 14) + 32


def test_without_pillow_the_reason_is_a_sentence(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_pil(name, *a, **k):
        if name.startswith("PIL"):
            raise ImportError("no PIL")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_pil)
    with pytest.raises(ImportError):
        hl.render_highlighted_text([("a", 1.0)], "x.png", title="x")


def test_the_texts_own_line_breaks_are_kept(tmp_path):
    """A newline unit starts a new line and a blank line stays blank, so a
    text written in paragraphs is drawn in paragraphs."""
    pytest.importorskip("PIL")
    from PIL import Image

    flat = [("one", 0.1), ("two", 0.2), ("three", 0.3), ("four", 0.1)]
    broken = [("one", 0.1), ("two", 0.2), ("\n", 0.0), ("three", 0.3), ("four", 0.1)]
    spaced = [("one", 0.1), ("two", 0.2), ("\n\n", 0.0), ("three", 0.3), ("four", 0.1)]
    heights = []
    for name, words in (("flat", flat), ("broken", broken), ("spaced", spaced)):
        out = hl.render_highlighted_text(words, tmp_path / f"{name}.png", title="t", width=600)
        with Image.open(out) as img:
            heights.append(img.size[1])
    assert heights[0] < heights[1] < heights[2]


def test_the_page_holds_every_text_selectable_with_its_breaks_and_fetches_nothing(tmp_path):
    out = hl.render_highlighted_page(
        [("the highest predicted score (t2), 4.5", [("<b>", 0.5), ("fine", -0.2), ("\n\n", 0.0),
                                                     ("day", 0.1)]),
         ("the lowest predicted score (t1), 1.2", [("tired", -0.6)])],
        tmp_path / "score__prediction.html", heading="Ridge: score — texts", legend="blue up")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and '<meta charset="utf-8">' in text
    assert "&lt;b&gt;" in text and "<b>" not in text.split("<body>")[1], "escaped"
    assert "<br><br>" in text, "a blank line stays blank"
    assert text.count("<article>") == 2 and "tired" in text and "blue up" in text
    assert "http" not in text and "<script" not in text and "<link" not in text
    before = out.stat().st_mtime_ns
    hl.render_highlighted_page(
        [("the highest predicted score (t2), 4.5", [("<b>", 0.5), ("fine", -0.2), ("\n\n", 0.0),
                                                     ("day", 0.1)]),
         ("the lowest predicted score (t1), 1.2", [("tired", -0.6)])],
        out, heading="Ridge: score — texts", legend="blue up")
    assert out.stat().st_mtime_ns == before, "unchanged content is left alone"
