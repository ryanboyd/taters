"""
Draw a word cloud: words sized by a weight, colored by its sign and strength.

The layout is ours -- a spiral outward from the center, horizontal words
only, no two boxes overlapping -- and it is pure Python, so it is tested
without a font engine by handing it a fake measurer. Pillow is used for the
two things that need a font: measuring a word at a size, and drawing the
PNG. It is imported inside the functions that need it, so importing this
module costs nothing and a machine without Pillow gets one plain sentence
instead of a traceback.

What the picture means is fixed here, once, for every cloud in Taters: size
follows the magnitude of the weight, shade darkens with it, blue is a
positive weight and red a negative one. A caller that wants a cloud of one
sign filters its words first; a caller with signed loadings passes them as
they are and gets both colors in one picture.
"""
from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce

__all__ = ["Placement", "Layout", "BLUE_HUE", "RED_HUE", "DEFAULT_FONT", "FILL",
           "pillow_missing_reason", "scale_sizes", "shade", "color_for",
           "fit_budget", "layout", "render_wordcloud"]

Word = Tuple[str, float]
RGB = Tuple[int, int, int]
Measure = Callable[[str, int], Tuple[int, int]]

#: Hue (degrees) for a positive weight -- a clear blue -- and for a negative
#: one -- a red. Direction is the whole point of a two-cloud comparison, so it
#: gets the strongest visual channel there is.
BLUE_HUE = 215.0
RED_HUE = 5.0

#: The one font that ships with Taters. Bold, because a cloud is read at a
#: glance and small words in a light face vanish. Bitstream Vera license; see
#: ``fonts/LICENSE`` beside it.
DEFAULT_FONT = Path(__file__).resolve().parent / "fonts" / "DejaVuSans-Bold.ttf"

#: What a caller is told when Pillow is not installed. Said once, in words,
#: instead of an ImportError from the middle of a drawing routine.
_PILLOW_MISSING = (
    "Pillow is needed to draw word clouds but is not installed. Install it "
    "with `pip install pillow` (a normal `pip install taters` includes it). "
    "The analysis results are complete; only the figures were skipped.")


@dataclass(frozen=True)
class Placement:
    """One word on the canvas: where its ink goes and how big it is."""

    text: str
    weight: float
    size: int
    x: int
    y: int
    w: int
    h: int

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` of the ink, canvas pixels."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)


@dataclass(frozen=True)
class Layout:
    """Where every word landed, and which did not fit at any legible size."""

    placed: List[Placement]
    dropped: List[str]
    width: int
    height: int
    top_margin: int


def pillow_missing_reason() -> Optional[str]:
    """``None`` when Pillow can be imported, else the sentence to show instead
    of a figure."""
    try:
        _pillow()
    except ImportError as e:
        return str(e)
    return None


def _pillow():
    """Pillow's three modules, imported on demand."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:            # pragma: no cover - exercised via monkeypatch
        raise ImportError(_PILLOW_MISSING) from e
    return Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# sizes and colors: these are pure functions of the weights
# ---------------------------------------------------------------------------

def scale_sizes(weights: Sequence[float], min_font: int, max_font: int, *,
                few: int = 8, few_floor: float = 0.5) -> List[int]:
    """
    Font size for each weight: linear in its magnitude, between a floor and
    ``max_font``.

    Linear, not by rank, because the picture is meant to show *how much*
    stronger the top word is, not just that it is first. The floor rises to
    ``few_floor * max_font`` when there are ``few`` words or fewer, so a cloud
    of five significant features fills its frame instead of huddling in the
    middle at the minimum size. One word gets ``max_font``; weights that are
    all equal share one middle size, since there is nothing to rank.
    """
    mags = [abs(float(w)) for w in weights]
    n = len(mags)
    if n == 0:
        return []
    if n == 1:
        return [int(max_font)]
    lo = int(min_font)
    if n <= few:
        lo = max(lo, round(few_floor * max_font))
    wmin, wmax = min(mags), max(mags)
    if wmax == wmin:
        return [(lo + int(max_font)) // 2] * n
    span = wmax - wmin
    return [round(lo + (max_font - lo) * (v - wmin) / span) for v in mags]


def shade(hue: float, strength: float) -> RGB:
    """
    A color of the given hue whose lightness falls as ``strength`` rises.

    Weak words come out pale and strong ones dark, so the eye reads strength
    twice -- in the size and in the ink -- and a strong word that had to
    shrink to fit still reads as strong.
    """
    s = min(1.0, max(0.0, float(strength)))
    lightness = 0.72 - s * (0.72 - 0.26)
    r, g, b = colorsys.hls_to_rgb((hue % 360.0) / 360.0, lightness, 0.65)
    return (round(r * 255), round(g * 255), round(b * 255))


def color_for(weight: float, max_abs: float) -> RGB:
    """Blue for a positive weight, red for a negative one, darker the larger
    it is relative to the cloud's strongest word."""
    strength = abs(weight) / max_abs if max_abs > 0 else 1.0
    return shade(BLUE_HUE if weight > 0 else RED_HUE, strength)


# ---------------------------------------------------------------------------
# layout: pure Python, with the text measurer passed in
# ---------------------------------------------------------------------------

def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Whether two boxes share any area. Touching edges do not count."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inside(box: Tuple[int, int, int, int],
            area: Tuple[int, int, int, int]) -> bool:
    return (box[0] >= area[0] and box[1] >= area[1]
            and box[2] <= area[2] and box[3] <= area[3])


def _pad_of(size: int) -> int:
    """Breathing room around a word, growing with its size so big words do
    not sit flush against their neighbors."""
    return max(2, size // 16)


def _padded(p: Placement) -> Tuple[int, int, int, int]:
    pad = _pad_of(p.size)
    return (p.x - pad, p.y - pad, p.x + p.w + pad, p.y + p.h + pad)


def _spiral_search(bw: int, bh: int, cx: float, cy: float, r_max: float,
                   theta0: float, area: Tuple[int, int, int, int],
                   placed: Sequence[Placement], max_steps: int,
                   aspect: float = 1.0) -> Optional[Tuple[int, int]]:
    """
    The first spot on an outward spiral where a ``bw`` by ``bh`` box fits.

    An Archimedean spiral: the radius grows by one step per turn, so the
    search sweeps the canvas evenly instead of leaving gaps between rings.
    The step is a quarter of the box height, so big words stride and small
    words probe the gaps the big ones left -- but never finer than a
    fortieth of the canvas radius. A spiral with a 5-pixel step needs some
    sixty thousand steps to reach the edge of a 1200 by 800 canvas, and
    capped at a few thousand it gave up at a radius of 180 pixels: every
    small word was declared unplaceable while three quarters of the canvas
    stood empty (34 of 80 dropped, on the first real picture). Returns
    ``None`` once the spiral leaves the canvas or runs out of steps.
    ``aspect`` stretches the spiral sideways: a round spiral in a frame
    nearly twice as wide as it is tall filled the height and left the sides
    empty, so the spiral is an ellipse of the frame's own shape.
    """
    step = max(2.0, bh / 4.0, r_max / 40.0)
    pitch = step / (2 * math.pi)
    theta, r = theta0, 0.0
    taken = [_padded(p) for p in placed]
    for _ in range(max_steps):
        x0 = round(cx + r * aspect * math.cos(theta) - bw / 2)
        y0 = round(cy + r * math.sin(theta) - bh / 2)
        box = (x0, y0, x0 + bw, y0 + bh)
        if _inside(box, area) and not any(_overlaps(box, t) for t in taken):
            return (x0, y0)
        dtheta = step / max(r, step)
        theta += dtheta
        r += pitch * dtheta
        if r > r_max:
            return None
    return None


def _recentred(placed: List[Placement],
               area: Tuple[int, int, int, int]) -> List[Placement]:
    """Shift the finished cloud so its bounding box is centered in the area.
    Shifted, never scaled: the sizes were measured and must stay true."""
    if not placed:
        return placed
    x0 = min(p.x for p in placed)
    y0 = min(p.y for p in placed)
    x1 = max(p.x + p.w for p in placed)
    y1 = max(p.y + p.h for p in placed)
    dx = round((area[0] + area[2]) / 2 - (x0 + x1) / 2)
    dy = round((area[1] + area[3]) / 2 - (y0 + y1) / 2)
    # clamp it, so that the shift never pushes a word off the canvas.
    dx = max(area[0] - x0, min(area[2] - x1, dx))
    dy = max(area[1] - y0, min(area[3] - y1, dy))
    return [replace(p, x=p.x + dx, y=p.y + dy) for p in placed]


def layout(words: Sequence[Word], sizes: Sequence[int], measure: Measure, *,
           width: int, height: int, top_margin: int = 0, min_font: int = 12,
           shrink: float = 0.85, seed: int = 0, max_steps: int = 6000) -> Layout:
    """
    Place every word on a ``width`` by ``height`` canvas below ``top_margin``.

    Largest first, each on a spiral out from the center, so the strongest
    words hold the middle and the rest fill in around them. A word that finds
    no room at its size shrinks by ``shrink`` and tries again, down to
    ``min_font``; below that it is dropped and named in ``Layout.dropped``
    rather than drawn illegibly or on top of something. ``measure(text,
    size)`` returns the ink width and height of a word; the renderer passes a
    real font, the tests pass arithmetic. The only randomness is the angle
    each spiral starts at, from ``seed``, so the same input draws the same
    picture every time.
    """
    if len(words) != len(sizes):
        raise ValueError("layout needs one size per word")
    rng = random.Random(seed)
    area = (0, int(top_margin), int(width), int(height))
    cx = width / 2.0
    cy = top_margin + (height - top_margin) / 2.0
    # the spiral is stretched to the frame's shape, so its radius is
    # measured against the height and the stretch reaches the width
    aspect = max(0.25, min(4.0, width / max(1, height - top_margin)))
    r_max = math.hypot(width / aspect, height - top_margin) / 2.0
    order = sorted(range(len(words)), key=lambda i: (-sizes[i], words[i][0]))
    placed: List[Placement] = []
    dropped: List[str] = []
    for i in order:
        text, weight = words[i]
        size = int(sizes[i])
        theta0 = rng.uniform(0.0, 2 * math.pi)
        while True:
            w, h = measure(text, size)
            pad = _pad_of(size)
            bw, bh = w + 2 * pad, h + 2 * pad
            spot = None
            if bw <= width and bh <= height - top_margin:
                spot = _spiral_search(bw, bh, cx, cy, r_max, theta0, area,
                                      placed, max_steps, aspect)
            if spot is not None:
                placed.append(Placement(text, weight, size,
                                        spot[0] + pad, spot[1] + pad, w, h))
                break
            if size <= min_font:
                dropped.append(text)
                break
            size = max(int(min_font), int(size * shrink))
    return Layout(_recentred(placed, area), dropped, int(width), int(height),
                  int(top_margin))


# ---------------------------------------------------------------------------
# the picture itself
# ---------------------------------------------------------------------------

def _as_pairs(words: Union[Sequence[Word], Mapping[str, float]],
              max_words: int) -> List[Word]:
    """
    Clean the input down to the words worth drawing.

    Blank labels, weights that are not numbers and weights of exactly zero
    are dropped; a label given twice keeps its larger magnitude; the rest are
    sorted by magnitude and cut at ``max_words``. Labels are stripped and
    nothing else -- a phrase like "going to be" is one word to a cloud.
    """
    items = words.items() if isinstance(words, Mapping) else words
    best: dict = {}
    for text, weight in items:
        label = str(text).strip()
        try:
            value = float(weight)
        except (TypeError, ValueError):
            continue
        if not label or not math.isfinite(value) or value == 0.0:
            continue
        if label not in best or abs(value) > abs(best[label]):
            best[label] = value
    ranked = sorted(best.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    return ranked[:max(0, int(max_words))]


#: How much of the drawing area the words' ink may add up to before their
#: sizes are scaled down. A spiral packs to roughly half the area; asking for
#: more drops words instead of drawing them.
FILL = 0.42
#: The share a sparse cloud is grown towards -- lower, since large words
#: pack less tightly on the spiral than small ones.
GROW_FILL = 0.28


def fit_budget(words: Sequence[Word], sizes: Sequence[int], measure: Measure,
               *, width: int, height: int, top_margin: int = 0,
               min_font: int = 12, fill: float = FILL) -> List[int]:
    """
    Sizes scaled so the words can actually fit on the canvas.

    :func:`scale_sizes` knows the weights and nothing about the canvas; eighty
    words at up to 96 px need more than 1200 by 800 pixels, and the layout's
    only answers are shrinking one word at a time or dropping it (36 of 80
    dropped, on the first real picture). Measuring the ink at the proposed
    sizes and scaling every size by the same factor keeps the picture's
    proportions -- the strongest word is still that much larger -- while
    making room for all of them.

    The same rule grows a sparse cloud: five significant features at the
    weight-scaled sizes sat in the middle of an otherwise empty picture, so
    when the ink falls short of the budget every size is scaled up, until
    the largest word is a quarter of the drawing height. Never below
    ``min_font``.
    """
    if not words:
        return list(sizes)
    ink = 0.0
    for (text, _w), size in zip(words, sizes):
        w, h = measure(text, int(size))
        pad = _pad_of(int(size))
        ink += (w + 2 * pad) * (h + 2 * pad)
    room = fill * width * max(1, height - top_margin)
    if ink <= 0:
        return [int(s) for s in sizes]
    factor = math.sqrt(room / ink)
    if factor > 1.0:
        # when we grow, we aim lower than we'd allow when shrinking. this is
        # because a spiral packs big words less tightly than small ones, and
        # growing to the full budget made the tail shrink back down one word
        # at a time (three seconds a cloud). we also stop once the largest
        # word is a quarter of the drawing height, so that one word doesn't
        # turn into a banner.
        factor = math.sqrt(GROW_FILL * room / (fill * ink))
        cap = max(1, height - top_margin) // 4
        factor = min(factor, cap / max(max(sizes), 1))
        if factor <= 1.0:
            return [int(s) for s in sizes]
    return [max(int(min_font), int(round(s * factor))) for s in sizes]


#: How far across the drawing area a cloud reaches in its longer direction.
#: Every cloud is scaled to this, so a picture of five words and a picture
#: of eighty fill the same frame; the ink budget above is only the first
#: guess the loop starts from.
EXTENT = 0.94
#: The largest word may be this share of the drawing height. A picture of
#: three words filled to the extent would otherwise be three banners.
MAX_WORD_SHARE = 0.4
#: How many layouts the fit loop may try before it keeps the best it saw.
FIT_PASSES = 5


def _reach(lay: Layout) -> float:
    """How much of the drawing area the cloud's bounding box spans, in the
    direction it spans most: 1.0 is edge to edge."""
    if not lay.placed:
        return 0.0
    boxes = [_padded(p) for p in lay.placed]
    bw = max(b[2] for b in boxes) - min(b[0] for b in boxes)
    bh = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    return max(bw / max(1, lay.width), bh / max(1, lay.height - lay.top_margin))


def fit_layout(words: Sequence[Word], sizes: Sequence[int], measure: Measure, *,
               width: int, height: int, top_margin: int = 0, min_font: int = 12,
               seed: int = 0, extent: float = EXTENT,
               passes: int = FIT_PASSES) -> Layout:
    """
    The layout scaled so the cloud spans ``extent`` of the drawing area.

    :func:`fit_budget` guesses sizes from the ink; the spiral then packs
    them as it packs them, and a sparse cloud came out small in the middle
    of its frame while a dense one reached the edges. The owner wants every
    figure the same shape, so after each layout the bounding box is
    measured and every size is scaled by the same factor towards the
    extent, up to ``passes`` times, keeping the proportions between words.
    A layout that dropped words is scaled down rather than up. The best
    layout seen is returned: no words dropped first, then the reach closest
    to the extent. The largest word stops at :data:`MAX_WORD_SHARE` of the
    drawing height and nothing goes below ``min_font``.
    """
    sizes = [int(s) for s in sizes]
    cap = max(int(min_font), int((height - top_margin) * MAX_WORD_SHARE))
    best: Optional[Tuple[Tuple[int, float], Layout]] = None
    for _ in range(max(1, int(passes))):
        lay = layout(words, sizes, measure, width=width, height=height,
                     top_margin=top_margin, min_font=min_font, seed=seed)
        reach = _reach(lay)
        score = (len(lay.dropped), abs(reach - extent))
        if best is None or score < best[0]:
            best = (score, lay)
        if not words:
            break
        factor = extent / reach if reach > 0 else 0.8
        if lay.dropped:
            factor = min(factor, 0.9)
        elif abs(factor - 1.0) < 0.03:
            break
        factor = min(factor, cap / max(max(sizes), 1))
        scaled = [max(int(min_font), int(round(s * factor))) for s in sizes]
        if scaled == sizes:
            break
        sizes = scaled
    return best[1]


class _Fonts:
    """One FreeType face per size, loaded on first use -- ``truetype`` is the
    slow call, and a cloud asks for a few dozen sizes many times over."""

    def __init__(self, ImageFont, path: Path):
        self._ImageFont = ImageFont
        self._path = str(path)
        self._cache: dict = {}

    def __getitem__(self, size: int):
        size = int(size)
        if size not in self._cache:
            self._cache[size] = self._ImageFont.truetype(self._path, size)
        return self._cache[size]


#: Labels longer than this are shown on two lines. A dictionary category
#: like ``Policy_Position_Laver_Garry__Values_Liberal`` is one word to the
#: layout, half the canvas wide at any readable size, and was being dropped.
LABEL_BREAK = 22


def display_label(text: str) -> str:
    """
    The label as the picture shows it: the table's name made readable.

    ``__`` (the dictionary-category separator) becomes ``: ``, ``_`` a
    space, and a label past :data:`LABEL_BREAK` cells is broken into two
    lines at the space nearest its middle. The table keeps the raw name;
    this is presentation only.
    """
    shown = str(text).replace("__", ": ").replace("_", " ").strip()
    if len(shown) <= LABEL_BREAK or " " not in shown:
        return shown
    middle = len(shown) // 2
    cuts = [i for i, ch in enumerate(shown) if ch == " "]
    at = min(cuts, key=lambda i: abs(i - middle))
    return shown[:at] + "\n" + shown[at + 1:]


def _ink(font, text: str) -> Tuple[int, int, int, int]:
    """The tight box around the ink of ``text``: ``(left, top, right,
    bottom)`` relative to the drawing anchor. Descenders are inside it, so a
    box that does not overlap another does not touch it on the page either."""
    if "\n" in text:
        # a two-line label: the union of its lines' boxes, stacked
        lines = text.split("\n")
        boxes = [font.getbbox(ln) for ln in lines]
        gap = max(2, int(font.size * 0.15))
        line_h = max(b[3] - b[1] for b in boxes) + gap
        left = min(b[0] for b in boxes)
        right = max(b[2] for b in boxes)
        top = boxes[0][1]
        bottom = top + line_h * (len(lines) - 1) + (boxes[-1][3] - boxes[-1][1])
        return (int(left), int(top), int(right), int(bottom))
    return tuple(int(v) for v in font.getbbox(text))


def _shorten(font, text: str, max_width: int) -> str:
    """Cut a caption line to fit, with an ellipsis, rather than run off."""
    if font.getlength(text) <= max_width:
        return text
    while text and font.getlength(text + "…") > max_width:
        text = text[:-1]
    return text.rstrip() + "…"


def _wrap(font, text: str, max_width: int, max_lines: int) -> List[str]:
    """Break a caption at spaces into at most ``max_lines`` lines that fit,
    the last one shortened if it still does not. A legend that names where
    a theme mattered ran off the right edge as one line."""
    words = str(text).split()
    lines: List[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if current and font.getlength(trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return [_shorten(font, ln, max_width) for ln in lines]


DEFAULT_LEGEND = "Size and shade follow the weight; blue = positive, red = negative."

#: The caption band: the title's type size and the lines reserved for it,
#: then the legend's. The band is the same height on every picture whether
#: the title takes one line or two, so the drawing area under it is the
#: same size on every figure and a longer name never shrinks the cloud.
TITLE_SIZE, TITLE_LINES = 22, 2
LEGEND_SIZE, LEGEND_LINES = 15, 2
CAPTION_GAP = 6
MARGIN = 16
TITLE_INK = (40, 40, 40)
LEGEND_INK = (110, 110, 110)


def _line_height(font) -> int:
    """One caption line: the face's ascent plus descent, which is the same
    for every line of text in that face, unlike the ink of a given line."""
    ascent, descent = font.getmetrics()
    return int(ascent + descent)


def caption_band(fonts: "_Fonts") -> int:
    """The height of the caption band, the same for every picture."""
    band = MARGIN
    for size, lines in ((TITLE_SIZE, TITLE_LINES), (LEGEND_SIZE, LEGEND_LINES)):
        band += lines * (_line_height(fonts[size]) + CAPTION_GAP)
    return band + MARGIN // 2


def draw_caption(draw, fonts: "_Fonts", width: int, title: str,
                 legend: Optional[str]) -> None:
    """Write the title and legend at the top, each line on the face's own
    line height from the top-left anchor, so a line without descenders
    sits where a line with them would."""
    y = MARGIN
    for text, size, color, lines in ((title, TITLE_SIZE, TITLE_INK, TITLE_LINES),
                                     (legend, LEGEND_SIZE, LEGEND_INK, LEGEND_LINES)):
        font = fonts[size]
        step = _line_height(font) + CAPTION_GAP
        for line in (_wrap(font, str(text), width - 2 * MARGIN, lines) if text else []):
            draw.text((MARGIN, y), line, font=font, fill=color)
            y += step


def render_wordcloud(words: Union[Sequence[Word], Mapping[str, float]],
                     out_png: Union[str, Path], *, title: str,
                     legend: Optional[str] = None, width: int = 1200,
                     height: int = 800, max_words: int = 80, min_font: int = 16,
                     max_font: int = 96, seed: int = 0,
                     font_path: Optional[Union[str, Path]] = None,
                     on_progress: Optional[Callable[..., None]] = None,
                     verbose: bool = False) -> Path:
    """
    Draw one word cloud to a PNG.

    Parameters
    ----------
    words : sequence of (label, weight) or mapping
        The words and their signed weights. Blue for positive, red for
        negative, size and shade by magnitude; the strongest ``max_words``
        are drawn.
    out_png : str or Path
        Where the picture goes. Written atomically.
    title : str
        Drawn across the top of the picture, so it still says what it is
        once pasted into a slide without the report.
    legend : str, optional
        The second caption line. Defaults to the sentence that explains size
        and color; pass ``""`` for none.
    width, height : int
        Canvas size in pixels.
    max_words, min_font, max_font : int
        How many words at most, and the smallest and largest type sizes.
    seed : int
        Fixes the layout, so the same words draw the same picture.
    font_path : str or Path, optional
        A TrueType font to use instead of the bundled DejaVu Sans Bold.
    on_progress, verbose
        The usual: a phase announcement, and a line naming any words that
        did not fit.

    Returns
    -------
    Path
        ``out_png``.

    Raises
    ------
    ImportError
        When Pillow is not installed, with a sentence that says what to do.
        Callers that would rather skip the figure check
        :func:`pillow_missing_reason` first.
    """
    Image, ImageDraw, ImageFont = _pillow()
    out_png = Path(out_png)
    announce(on_progress, f"drawing word cloud: {title}")

    fonts = _Fonts(ImageFont, Path(font_path) if font_path else DEFAULT_FONT)
    pairs = [(display_label(t), w) for t, w in _as_pairs(words, max_words)]
    legend = DEFAULT_LEGEND if legend is None else legend

    img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # the caption band is a fixed height, so the cloud is laid out in the
    # same frame on every picture; the loop then scales the cloud to that
    # frame, so five words and eighty fill it alike
    band = caption_band(fonts)

    def measure(text: str, size: int) -> Tuple[int, int]:
        left, top, right, bottom = _ink(fonts[size], text)
        return (right - left, bottom - top)

    sizes = scale_sizes([w for _, w in pairs], min_font, max_font)
    sizes = fit_budget(pairs, sizes, measure, width=width, height=height,
                       top_margin=band, min_font=min_font)
    lay = fit_layout(pairs, sizes, measure, width=width, height=height,
                     top_margin=band, min_font=min_font, seed=seed)

    draw_caption(draw, fonts, width, title, legend)
    max_abs = max((abs(w) for _, w in pairs), default=1.0)
    for p in lay.placed:
        font = fonts[p.size]
        left, top, _r, _b = _ink(font, p.text)
        color = color_for(p.weight, max_abs)
        if "\n" in p.text:
            # each line on its own baseline, centered under the other
            lines = p.text.split("\n")
            gap = max(2, int(font.size * 0.15))
            boxes = [font.getbbox(ln) for ln in lines]
            line_h = max(b[3] - b[1] for b in boxes) + gap
            widest = max(b[2] - b[0] for b in boxes)
            for i, (ln, b) in enumerate(zip(lines, boxes)):
                dx = (widest - (b[2] - b[0])) // 2
                draw.text((p.x - b[0] + dx, p.y - top + i * line_h), ln, font=font,
                          fill=color)
        else:
            draw.text((p.x - left, p.y - top), p.text, font=font, fill=color)

    if verbose and lay.dropped:
        print(f"[figures] {len(lay.dropped)} word(s) did not fit in "
              f"{out_png.name} and were left out: "
              f"{', '.join(lay.dropped[:6])}{'…' if len(lay.dropped) > 6 else ''}")
    with atomic_write(out_png, mode="wb") as fh:
        # the scratch file has no extension, so PIL can't guess the format.
        img.save(fh, format="PNG", optimize=True)
    return out_png
