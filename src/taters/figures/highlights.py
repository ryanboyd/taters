"""
A text with its words shaded by how much each pushed a prediction.

The companion to the word clouds for a model explained by token
attribution: the cloud says which words push the score across the corpus,
this says what one text's score was made of, word by word, in order. Blue
pushes the score up, red down, and the ink darkens with the size of the
share, on the same scale :mod:`render` uses for the clouds. Drawn from the
``<analysis>_attributed_texts`` table; the table is the record.
"""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from .render import (BLUE_HUE, DEFAULT_FONT, RED_HUE, _Fonts, _ink, _pillow,
                     caption_band, draw_caption)

__all__ = ["render_highlighted_text", "background_for"]

Word = Tuple[str, float]


def background_for(weight: float, max_abs: float) -> Tuple[int, int, int]:
    """The shade behind a word: blue for a positive share, red for a
    negative one, from near-white for a tiny share to a clear tint for the
    text's strongest, so black text stays readable on all of them."""
    strength = min(1.0, abs(weight) / max_abs) if max_abs > 0 else 0.0
    if strength < 1e-9:
        return (255, 255, 255)
    lightness = 0.97 - strength * (0.97 - 0.68)
    hue = BLUE_HUE if weight > 0 else RED_HUE
    r, g, b = colorsys.hls_to_rgb((hue % 360.0) / 360.0, lightness, 0.8)
    return (round(r * 255), round(g * 255), round(b * 255))


def render_highlighted_text(words: Sequence[Word], out_png, *, title: str,
                            legend: Optional[str] = None, width: int = 1200,
                            font_size: int = 22, font_path=None,
                            max_lines: int = 40,
                            on_progress=None, verbose: bool = False) -> Path:
    """
    Draw ``words`` in reading order, each on a background shaded by its
    weight, wrapped to ``width``; the picture is as tall as the text needs,
    up to ``max_lines`` lines (the rest is said in a last line).
    """
    Image, ImageDraw, ImageFont = _pillow()
    out_png = Path(out_png)
    announce(on_progress, f"drawing highlighted text: {title}")
    fonts = _Fonts(ImageFont, Path(font_path) if font_path else DEFAULT_FONT)
    body = fonts[font_size]
    margin = 16
    gap = 8
    line_height = font_size + 14
    max_abs = max((abs(w) for _t, w in words), default=0.0)

    # the caption band is the same height as on the clouds, whatever the
    # title takes, so the shaded texts line up with their clouds on a page
    band = caption_band(fonts)

    # then the words, wrapped: each is a box the width of its ink plus a
    # little air, and a line breaks when the next one would not fit
    boxes = []
    x, y = margin, band
    lines_used = 1
    cut = False
    for token, weight in words:
        text = str(token)
        if not text:
            continue
        left, top, right, bottom = _ink(body, text)
        w = (right - left) + 10
        if x + w > width - margin and x > margin:
            x = margin
            y += line_height
            lines_used += 1
            if lines_used > max_lines:
                cut = True
                break
        boxes.append((text, weight, x, y, w, left, top))
        x += w + gap
    height = y + line_height + margin + (line_height if cut else 0)

    img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_caption(draw, fonts, width, title, legend)
    for text, weight, bx, by, w, left, top in boxes:
        draw.rectangle([bx, by, bx + w, by + line_height - 4],
                       fill=background_for(weight, max_abs))
        # anchored at the ascender, not at each word's own ink top: lining
        # up ink tops hoisted "on", "my" and "as" above their neighbors,
        # since a word with no ascender has a lower top. same baseline for
        # every word is what reading expects
        draw.text((bx + 5 - left, by + 4), text, font=body, fill=(20, 20, 20))
    if cut:
        note = fonts[15]
        draw.text((margin, y + line_height), f"… and {len(words) - len(boxes)} more words",
                  font=note, fill=(110, 110, 110))
    with atomic_write(out_png, mode="wb") as fh:
        img.save(fh, format="PNG", optimize=True)
    return out_png
