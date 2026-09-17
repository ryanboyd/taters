"""
Small charts for training reports: line charts, scatter plots, heat tables.

A training report has three pictures a paper wants: a loss curve (a line
per fold or per series over epochs), a predicted-versus-observed scatter,
and a confusion matrix as a colored table. None of them needs a plotting
library -- the axes, ticks, legend and points are a few hundred lines of
arithmetic and Pillow, the same dependency the word clouds already use, so
a headless machine draws them and a machine without Pillow gets one plain
sentence instead of a traceback.

The layout arithmetic (tick choice, data-to-pixel mapping) is pure Python
and tested without a font engine; Pillow is imported inside the functions
that draw, as in :mod:`taters.figures.render`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (List, Literal, Mapping, Optional, Sequence, Tuple,
                    Union)

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from .render import (DEFAULT_FONT, _Fonts, _ink, _pillow, _wrap,
                     pillow_missing_reason)

__all__ = ["Axes", "nice_ticks", "log_ticks", "line_chart",
           "dual_axis_line_chart", "scatter_chart", "heat_table",
           "PALETTE", "pillow_missing_reason"]

PathLike = Union[str, Path]
Point = Tuple[float, float]
RGB = Tuple[int, int, int]

#: Series colors, in order: a blue, an orange, a green, a red, a purple, a
#: brown, a pink, a gray -- distinguishable in print and to most color-
#: blind readers (the Tableau 10 hues).
PALETTE: Tuple[RGB, ...] = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207))

_INK = (40, 40, 40)
_GRID = (225, 225, 225)
_MUTED = (110, 110, 110)


# ---------------------------------------------------------------------------
# pure layout -- no drawing in here, just numbers
# ---------------------------------------------------------------------------

def nice_ticks(lo: float, hi: float, n: int = 6) -> List[float]:
    """
    Round tick positions covering ``[lo, hi]``: about ``n`` of them, at a
    step of 1, 2, 2.5 or 5 times a power of ten, the first at or below
    ``lo`` and the last at or above ``hi``. A flat range gets a unit
    around its value, so a constant series still has an axis.
    """
    lo, hi = float(lo), float(hi)
    if not math.isfinite(lo) or not math.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        pad = abs(lo) * 0.1 or 0.5
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(1, int(n) - 1)
    power = 10 ** math.floor(math.log10(raw))
    step = next((m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw), 10 * power)
    first = math.floor(lo / step) * step
    ticks = []
    t = first
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    if ticks[-1] < hi:
        ticks.append(round(t, 10))
    return ticks


#: How many pixels a nominal pixel is. Everything positional in this module is
#: written at this density and multiplied by `scale` on the way out, so a
#: chart asked for at 300 dpi comes out the same shape rather than the same
#: size with microscopic text.
_SCREEN_DPI = 96

#: Which way an x axis counts. `integer` is the old `integer_x=True`: round
#: numbers only, for a count of something. `log` is for a range that spans
#: decades -- a topic-count sweep from 5 to 2000 wastes most of its width on a
#: linear axis.
XScale = Literal["linear", "integer", "log"]


def _px(value: float, scale: float = 1.0) -> int:
    """
    A layout constant at the drawing scale.

    ``_px(v, 1.0) == v`` exactly for every integer constant in this module,
    which is the whole backward-compatibility story: an unscaled chart is
    pixel-identical to the one this module drew before it could scale at all.
    """
    return max(1, int(round(float(value) * scale)))


def log_ticks(lo: float, hi: float, *,
              minors: Sequence[float] = (1.0, 2.0, 5.0),
              max_ticks: int = 12) -> List[float]:
    """
    Round tick positions covering ``[lo, hi]`` on a log axis: values of the
    form *m*x10^n, the first at or below ``lo`` and the last at or above
    ``hi`` -- the same contract :func:`nice_ticks` advertises, so the frame
    can take its data range from the ticks either way.

    A 1-2-5 ladder rather than decades alone, because decades are wasteful
    exactly where this is wanted: 5 to 2000 in decades is 1, 10, 100, 1000,
    10000, which pads the axis out past both ends and throws away a third of
    the width. The 1-2-5 ladder lands its first and last ticks *on the data*.

    Thins itself when the span is wide -- first dropping the 2s, then down to
    decades -- rather than printing forty labels over each other.
    """
    lo, hi = float(lo), float(hi)
    if hi < lo:
        lo, hi = hi, lo
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0:
        raise ValueError(
            f"a log axis needs positive values; got a low of {lo!r}. "
            "Use x_scale='linear' for data that reaches zero.")
    if hi == lo:
        lo, hi = lo / 2.0, hi * 2.0

    def ladder(steps: Sequence[float]) -> List[float]:
        rungs = []
        first = int(math.floor(math.log10(lo))) - 1
        last = int(math.ceil(math.log10(hi))) + 1
        for power in range(first, last + 1):
            for step in steps:
                rungs.append(round(float(step) * (10.0 ** power), 12))
        return sorted(set(rungs))

    coarsest: List[float] = []
    for steps in (tuple(minors),
                  tuple(m for m in minors if m != 2.0),
                  (1.0,)):
        if not steps:
            continue
        rungs = ladder(steps)
        below = [t for t in rungs if t <= lo * (1.0 + 1e-12)]
        above = [t for t in rungs if t >= hi * (1.0 - 1e-12)]
        if not below or not above:
            continue
        coarsest = [t for t in rungs if below[-1] <= t <= above[0]]
        if len(coarsest) <= max_ticks:
            return coarsest
    return coarsest


def _x_ticks(lo: float, hi: float, kind: XScale, n: int = 8) -> List[float]:
    """One place the three x scales are chosen between, so that the charts
    that share this frame cannot drift apart about what `integer` means."""
    if kind == "log":
        return log_ticks(lo, hi)
    if kind not in ("linear", "integer"):
        raise ValueError(
            f"unknown x_scale {kind!r}; one of 'linear', 'integer', 'log'")
    ticks = nice_ticks(lo, hi, n)
    if kind == "integer":
        ticks = sorted({int(round(t)) for t in ticks if t == int(t)}
                       or {int(lo), int(hi)})
        if len(ticks) == 1:
            ticks = [ticks[0] - 1, ticks[0], ticks[0] + 1]
    return list(ticks)


def _aligned_ticks(lo: float, hi: float, rows: int) -> List[float]:
    """
    Exactly ``rows`` ticks covering ``[lo, hi]``.

    For the right-hand scale of a two-axis chart: with six grid lines on the
    left and five labels on the right, the right-hand numbers float *between*
    the grid lines and the whole chart reads as broken. Asking for a count
    rather than a step fixes the rows to each other; the axis range stretches
    upward to absorb the difference, which costs nothing but white space.
    """
    rows = max(2, int(rows))
    ticks = nice_ticks(lo, hi, rows)
    for fewer in range(rows, 1, -1):
        candidate = nice_ticks(lo, hi, fewer)
        if len(candidate) <= rows:
            ticks = candidate
            break
    while len(ticks) < rows:
        step = (ticks[-1] - ticks[-2]) if len(ticks) > 1 else 1.0
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


@dataclass(frozen=True)
class Axes:
    """The plotting rectangle in pixels and the data range it shows: the
    one mapping every mark in a chart goes through."""

    left: int
    top: int
    right: int
    bottom: int
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    #: Defaulted last, so the eight positional arguments still bind.
    x_log: bool = False

    def _fx(self, x: float) -> float:
        """The x value as the axis counts it. Under a log scale anything at or
        below the low end lands *on* the low end rather than raising: a stray
        zero should put a mark on the left edge, not throw from inside a draw
        loop after the chart is half painted."""
        x = float(x)
        if not self.x_log:
            return x
        floor = self.x_lo if self.x_lo > 0 else 1e-12
        return math.log10(x if x > floor else floor)

    def px(self, x: float) -> int:
        lo, hi = self._fx(self.x_lo), self._fx(self.x_hi)
        span = (hi - lo) or 1.0
        return int(round(self.left + (self._fx(x) - lo) / span * (self.right - self.left)))

    def with_y(self, y_lo: float, y_hi: float) -> "Axes":
        """The same rectangle and the same x mapping, a different y range --
        the right-hand axis of a two-scale chart."""
        return replace(self, y_lo=float(y_lo), y_hi=float(y_hi))

    def py(self, y: float) -> int:
        span = (self.y_hi - self.y_lo) or 1.0
        return int(round(self.bottom - (float(y) - self.y_lo) / span * (self.bottom - self.top)))

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


def _fmt_tick(value: float) -> str:
    if abs(value) >= 1000 or value == int(value):
        return f"{int(round(value)):d}" if abs(value) < 1e15 else f"{value:.2g}"
    text = f"{value:.4g}"
    return text


def _range(values: Sequence[float], *, include_zero: bool = False) -> Tuple[float, float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return (0.0, 1.0)
    lo, hi = min(finite), max(finite)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    return lo, hi


# ---------------------------------------------------------------------------
# the frame that every chart shares
# ---------------------------------------------------------------------------

def _frame(draw, fonts, *, width: int, height: int, title: str, x_label: str,
           y_label: str, x_ticks: Sequence[float], y_ticks: Sequence[float],
           x_tick_labels: Optional[Sequence[str]] = None,
           legend: Sequence[Tuple[str, RGB]] = (),
           x_log: bool = False,
           y2_ticks: Sequence[float] = (), y2_label: str = "",
           y_color: Optional[RGB] = None, y2_color: Optional[RGB] = None,
           scale: float = 1.0) -> Axes:
    """
    Title, axes, grid, ticks, labels and legend; returns the axes.

    ``y2_ticks`` adds a second scale down the right-hand edge, for a chart
    whose two series share an x but nothing else. The grid stays on the *left*
    axis alone -- two interleaved sets of grid lines is noise, and the reader
    only needs one set to read positions against.

    ``scale`` multiplies every distance and every font size together, so a
    chart drawn at four times the size is the same chart rather than the same
    layout with unreadable text. At 1.0 it is the identity (see :func:`_px`).
    """
    s = scale
    margin = _px(18, s)
    title_font = fonts[_px(20, s)]
    label_font = fonts[_px(15, s)]
    tick_font = fonts[_px(13, s)]
    y = margin
    if title:
        for line in _wrap(title_font, title, width - 2 * margin, 2):
            left_, top_, right_, bottom_ = _ink(title_font, line)
            draw.text((margin - left_, y - top_), line, font=title_font, fill=_INK)
            y += (bottom_ - top_) + _px(6, s)
    top = y + _px(10, s)
    # leave room on the left for the widest tick label plus the axis label.
    widest = max((tick_font.getlength(_fmt_tick(t)) for t in y_ticks),
                 default=_px(20, s))
    left = int(margin + (_px(22, s) if y_label else 0) + widest + _px(10, s))
    bottom = int(height - margin - (_px(26, s) if x_label else 0) - _px(22, s))
    right = width - margin
    # the right-hand tick column sits *inboard* of the legend, so the two do
    # not draw over each other when a chart has both.
    gutter = 0
    if y2_ticks:
        widest2 = max(tick_font.getlength(_fmt_tick(t)) for t in y2_ticks)
        gutter = int(widest2 + _px(14, s) + (_px(22, s) if y2_label else 0))
        right -= gutter
    legend_rows = list(legend)
    if legend_rows:
        right -= int(max(label_font.getlength(name) for name, _ in legend_rows)
                     + _px(40, s))
    axes = Axes(left, top, right, bottom, min(x_ticks), max(x_ticks),
                min(y_ticks), max(y_ticks), x_log=x_log)
    # now the grid and the ticks.
    for t in y_ticks:
        py = axes.py(t)
        draw.line([(left, py), (right, py)], fill=_GRID, width=1)
        text = _fmt_tick(t)
        l_, t_, r_, b_ = _ink(tick_font, text)
        draw.text((left - _px(8, s) - (r_ - l_) - l_, py - (b_ - t_) // 2 - t_),
                  text, font=tick_font, fill=y_color or _MUTED)
    labels = list(x_tick_labels) if x_tick_labels is not None else [_fmt_tick(t) for t in x_ticks]
    for t, text in zip(x_ticks, labels):
        px = axes.px(t)
        draw.line([(px, top), (px, bottom)], fill=_GRID, width=1)
        l_, t_, r_, b_ = _ink(tick_font, text)
        draw.text((px - (r_ - l_) // 2 - l_, bottom + _px(6, s) - t_), text,
                  font=tick_font, fill=_MUTED)
    if y2_ticks:
        # a tick mark and a left-aligned label, no grid line
        right_axes = axes.with_y(min(y2_ticks), max(y2_ticks))
        for t in y2_ticks:
            py = right_axes.py(t)
            draw.line([(right, py), (right + _px(5, s), py)],
                      fill=y2_color or _MUTED, width=1)
            text = _fmt_tick(t)
            l_, t_, r_, b_ = _ink(tick_font, text)
            draw.text((right + _px(8, s) - l_, py - (b_ - t_) // 2 - t_), text,
                      font=tick_font, fill=y2_color or _MUTED)
    draw.rectangle([left, top, right, bottom], outline=(150, 150, 150), width=1)
    if x_label:
        l_, t_, r_, b_ = _ink(label_font, x_label)
        draw.text(((left + right) // 2 - (r_ - l_) // 2 - l_,
                   height - margin - (b_ - t_) - t_),
                  x_label, font=label_font, fill=_INK)
    if y_label:
        # the y label is rotated, so we draw it on its own little image and
        # paste that in.
        _paste_rotated(draw, label_font, y_label, 90, margin - 2,
                       (top + bottom) // 2, y_color or _INK, s)
    if y2_label:
        # just outboard of its own tick numbers, not out at the image edge:
        # the label belongs to the scale it names, and the legend owns the
        # far right. Reads the other way round, which is the usual setting.
        _paste_rotated(draw, label_font, y2_label, -90,
                       right + gutter - _px(22, s), (top + bottom) // 2,
                       y2_color or _INK, s)
    # lastly, the legend, off to the right of everything.
    ly = top
    for name, color in legend_rows:
        x0 = right + gutter + _px(12, s)
        draw.rectangle([x0, ly + _px(3, s), x0 + _px(14, s), ly + _px(15, s)],
                       fill=color)
        draw.text((x0 + _px(20, s), ly), name, font=label_font, fill=_INK)
        ly += _px(22, s)
    return axes


def _paste_rotated(draw, font, text: str, angle: int, x: int, y_center: int,
                   color: RGB, scale: float = 1.0) -> None:
    """One rotated axis label, drawn on its own tile and pasted. Shared by the
    left and right y labels so the two cannot drift apart."""
    Image, ImageDraw, _ImageFont = _pillow()
    l_, t_, r_, b_ = _ink(font, text)
    pad = _px(2, scale)
    tile = Image.new("RGBA", (r_ - l_ + 2 * pad, b_ - t_ + 2 * pad), (255, 255, 255, 0))
    ImageDraw.Draw(tile).text((pad - l_, pad - t_), text, font=font, fill=color)
    tile = tile.rotate(angle, expand=True)
    draw._image.paste(tile, (int(x), y_center - tile.height // 2), tile)


def _open(width: int, height: int, font_path: Optional[PathLike], *,
          scale: float = 1.0):
    Image, ImageDraw, ImageFont = _pillow()
    img = Image.new("RGB", (_px(width, scale), _px(height, scale)),
                    (255, 255, 255))
    draw = ImageDraw.Draw(img)
    fonts = _Fonts(ImageFont, Path(font_path) if font_path else DEFAULT_FONT)
    return img, draw, fonts


def _save(img, out_png: Path, *, dpi: Optional[int] = None) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    options = {"format": "PNG", "optimize": True}
    if dpi is not None:
        # a pHYs chunk, so Word and LaTeX place the figure at the size it was
        # drawn for instead of assuming 72 dpi. Absent by default, which keeps
        # every existing chart byte-identical.
        options["dpi"] = (int(dpi), int(dpi))
    with atomic_write(out_png, mode="wb") as fh:
        img.save(fh, **options)
    return out_png


# ---------------------------------------------------------------------------
# the charts themselves
# ---------------------------------------------------------------------------

def _drawing_scale(dpi: Optional[int], scale: Optional[float]) -> float:
    """
    How much to multiply every distance by.

    An explicit ``scale`` wins. Otherwise a ``dpi`` implies one, because 900
    pixels tagged 300 dpi is a three-inch figure, which is almost never what
    anyone means by print quality -- at ``dpi / 96`` the nominal width instead
    means "nine inches, at whatever density you asked for".
    """
    if scale is not None:
        return float(scale)
    if dpi is not None:
        return float(dpi) / _SCREEN_DPI
    return 1.0


def _draw_series(draw, axes: Axes, series: Mapping[str, Sequence[Point]],
                 colors: Sequence[RGB], *, scale: float = 1.0) -> None:
    """One line per series through its points, with a dot at each. Shared by
    the one-scale and two-scale charts so the two cannot drift apart."""
    radius = _px(4, scale)
    for (_name, pts), color in zip(series.items(), colors):
        pixels = [(axes.px(x), axes.py(y)) for x, y in pts
                  if y is not None and math.isfinite(float(y))]
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=_px(3, scale), joint="curve")
        for px, py in pixels:
            draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                         fill=color)


def line_chart(series: Mapping[str, Sequence[Point]], out_png: PathLike, *,
               title: str = "", x_label: str = "", y_label: str = "",
               width: int = 900, height: int = 560, include_zero: bool = False,
               x_scale: XScale = "integer", dpi: Optional[int] = None,
               scale: Optional[float] = None,
               font_path: Optional[PathLike] = None,
               on_progress=None) -> Path:
    """
    One line per series through its ``(x, y)`` points, in ``PALETTE`` order.

    Built for loss curves: ``{"fold 1 train": [(1, 0.9), (2, 0.7)], "fold 1
    validation": [...]}``. ``x_scale`` picks how x counts -- ``integer`` for
    epochs, ``linear`` for a measured quantity, ``log`` for a range spanning
    decades. ``include_zero`` forces the y axis to start at zero.

    ``width`` and ``height`` are nominal pixels at 96 dpi; ``dpi`` tags the
    file and scales the drawing to match, so the figure keeps its proportions.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    s = _drawing_scale(dpi, scale)
    img, draw, fonts = _open(width, height, font_path, scale=s)
    xs = [p[0] for pts in series.values() for p in pts]
    ys = [p[1] for pts in series.values() for p in pts]
    x_lo, x_hi = _range(xs)
    y_lo, y_hi = _range(ys, include_zero=include_zero)
    x_ticks = _x_ticks(x_lo, x_hi, x_scale)
    y_ticks = nice_ticks(y_lo, y_hi, 6)
    legend = [(name, PALETTE[i % len(PALETTE)]) for i, name in enumerate(series)]
    axes = _frame(draw, fonts, width=img.width, height=img.height, title=title,
                  x_label=x_label, y_label=y_label, x_ticks=x_ticks,
                  y_ticks=y_ticks, legend=legend if len(series) > 1 else (),
                  x_log=(x_scale == "log"), scale=s)
    _draw_series(draw, axes, series, [c for _n, c in legend], scale=s)
    return _save(img, Path(out_png), dpi=dpi)


def dual_axis_line_chart(left: Mapping[str, Sequence[Point]],
                         right: Mapping[str, Sequence[Point]],
                         out_png: PathLike, *,
                         title: str = "", x_label: str = "",
                         y_label: str = "", y2_label: str = "",
                         width: int = 900, height: int = 560,
                         include_zero: bool = False,
                         include_zero_right: bool = False,
                         x_scale: XScale = "linear",
                         dpi: Optional[int] = None,
                         scale: Optional[float] = None,
                         font_path: Optional[PathLike] = None,
                         on_progress=None) -> Path:
    """
    Two groups of series over one x, on **independent** y scales -- what
    matplotlib calls a twin axis.

    For two quantities whose shapes you want to compare but whose units have
    nothing to do with each other: coherence falling while exclusivity climbs,
    where forcing both onto one scale would flatten whichever has the smaller
    range into a straight line.

    A separate function rather than a flag on :func:`line_chart`, because
    which series belongs to which axis is structural, not styling: as a
    parameter it would make ``y_label``, ``include_zero`` and the y range all
    ambiguous, and would let a caller name a series that is not there.

    The left scale's ticks and label are drawn in the first series' color and
    the right scale's in its own, since color is the strongest cue available;
    the legend says "(left)" and "(right)" as well, for print in grayscale.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    if not left or not right:
        raise ValueError("a two-scale chart needs at least one series on each "
                         "side; got "
                         f"{len(left)} left and {len(right)} right")
    s = _drawing_scale(dpi, scale)
    img, draw, fonts = _open(width, height, font_path, scale=s)
    xs = [p[0] for group in (left, right) for pts in group.values() for p in pts]
    x_lo, x_hi = _range(xs)
    y_lo, y_hi = _range([p[1] for pts in left.values() for p in pts],
                        include_zero=include_zero)
    y2_lo, y2_hi = _range([p[1] for pts in right.values() for p in pts],
                          include_zero=include_zero_right)
    x_ticks = _x_ticks(x_lo, x_hi, x_scale)
    y_ticks = nice_ticks(y_lo, y_hi, 6)
    # the same number of rows on both sides, or the right-hand numbers float
    # between the left-hand grid lines and the chart reads as broken
    y2_ticks = _aligned_ticks(y2_lo, y2_hi, len(y_ticks))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(left) + len(right))]
    legend = ([(f"{name} (left)", colors[i]) for i, name in enumerate(left)]
              + [(f"{name} (right)", colors[len(left) + i])
                 for i, name in enumerate(right)])
    axes = _frame(draw, fonts, width=img.width, height=img.height, title=title,
                  x_label=x_label, y_label=y_label, x_ticks=x_ticks,
                  y_ticks=y_ticks, legend=legend, x_log=(x_scale == "log"),
                  y2_ticks=y2_ticks, y2_label=y2_label,
                  y_color=colors[0], y2_color=colors[len(left)], scale=s)
    _draw_series(draw, axes, left, colors[:len(left)], scale=s)
    _draw_series(draw, axes.with_y(min(y2_ticks), max(y2_ticks)), right,
                 colors[len(left):], scale=s)
    return _save(img, Path(out_png), dpi=dpi)


def scatter_chart(points: Sequence[Point], out_png: PathLike, *,
                  title: str = "", x_label: str = "", y_label: str = "",
                  identity: bool = True, note: str = "", width: int = 720,
                  height: int = 720, dpi: Optional[int] = None,
                  scale: Optional[float] = None,
                  font_path: Optional[PathLike] = None,
                  on_progress=None) -> Path:
    """
    Points on square axes; with ``identity`` a dashed y = x line, so a
    predicted-versus-observed plot shows at a glance how far from perfect
    the predictions fall. ``note`` (r, R²) is written inside the axes.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    s = _drawing_scale(dpi, scale)
    img, draw, fonts = _open(width, height, font_path, scale=s)
    finite = [(float(x), float(y)) for x, y in points
              if x is not None and y is not None
              and math.isfinite(float(x)) and math.isfinite(float(y))]
    lo, hi = _range([v for p in finite for v in p]) if finite else (0.0, 1.0)
    ticks = nice_ticks(lo, hi, 6)
    axes = _frame(draw, fonts, width=img.width, height=img.height, title=title,
                  x_label=x_label, y_label=y_label, x_ticks=ticks,
                  y_ticks=ticks, scale=s)
    if identity:
        a, b = (axes.px(ticks[0]), axes.py(ticks[0])), (axes.px(ticks[-1]), axes.py(ticks[-1]))
        steps = 40
        for i in range(0, steps, 2):
            x0 = a[0] + (b[0] - a[0]) * i / steps
            y0 = a[1] + (b[1] - a[1]) * i / steps
            x1 = a[0] + (b[0] - a[0]) * (i + 1) / steps
            y1 = a[1] + (b[1] - a[1]) * (i + 1) / steps
            draw.line([(x0, y0), (x1, y1)], fill=(170, 170, 170),
                      width=_px(2, s))
    radius = _px(4, s)
    color = PALETTE[0] + (150,)
    overlay = img.copy().convert("RGBA")
    from PIL import ImageDraw as _ImageDraw

    layer = _ImageDraw.Draw(overlay, "RGBA")
    for x, y in finite:
        px, py = axes.px(x), axes.py(y)
        layer.ellipse([px - radius, py - radius, px + radius, py + radius],
                      fill=color)
    img.paste(overlay.convert("RGB"))
    if note:
        font = fonts[_px(15, s)]
        draw = _ImageDraw.Draw(img)
        left, top, right, bottom = _ink(font, note)
        draw.text((axes.left + _px(10, s) - left, axes.top + _px(8, s) - top),
                  note, font=font, fill=_INK)
    return _save(img, Path(out_png), dpi=dpi)


def heat_cell_color(value: float, lo: float, hi: float) -> RGB:
    """White at ``lo`` through to a saturated blue at ``hi``; monotone in
    the value, so a darker cell is always a larger number."""
    if hi <= lo or not math.isfinite(float(value)):
        strength = 0.0
    else:
        strength = min(1.0, max(0.0, (float(value) - lo) / (hi - lo)))
    r = int(round(255 - strength * (255 - 31)))
    g = int(round(255 - strength * (255 - 119)))
    b = int(round(255 - strength * (255 - 180)))
    return (r, g, b)


def heat_table(rows: Sequence[Sequence[float]], out_png: PathLike, *,
               row_labels: Sequence[str], col_labels: Sequence[str],
               title: str = "", row_title: str = "", col_title: str = "",
               cell_format: str = "{:g}", dpi: Optional[int] = None,
               scale: Optional[float] = None,
               font_path: Optional[PathLike] = None,
               on_progress=None) -> Path:
    """
    A matrix as a colored table: each cell shaded by its value and printed
    with it. Built for confusion matrices (rows observed, columns
    predicted), where the diagonal should be the dark one.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    s = _drawing_scale(dpi, scale)
    n_rows, n_cols = len(rows), len(col_labels)
    # this one sizes itself from its contents rather than taking a width,
    # so `scale` multiplies the cell metrics and the rest follows
    cell_w, cell_h, margin = _px(110, s), _px(56, s), _px(18, s)
    Image, ImageDraw, ImageFont = _pillow()
    fonts = _Fonts(ImageFont, Path(font_path) if font_path else DEFAULT_FONT)
    label_font = fonts[_px(14, s)]
    cell_font = fonts[_px(16, s)]
    title_font = fonts[_px(20, s)]
    label_w = int(max((label_font.getlength(str(x)) for x in row_labels), default=_px(40, s))) + _px(20, s)
    label_w += _px(26, s) if row_title else 0
    header_h = _px(30, s) + (_px(26, s) if col_title else 0)
    title_h = _px(34, s) if title else 0
    width = margin * 2 + label_w + cell_w * n_cols
    height = margin * 2 + title_h + header_h + cell_h * n_rows
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = margin
    if title:
        l_, t_, r_, b_ = _ink(title_font, title)
        draw.text((margin - l_, y - t_), title, font=title_font, fill=_INK)
        y += title_h
    x0 = margin + label_w
    if col_title:
        l_, t_, r_, b_ = _ink(label_font, col_title)
        draw.text((x0 + (cell_w * n_cols) // 2 - (r_ - l_) // 2 - l_, y - t_),
                  col_title, font=label_font, fill=_MUTED)
        y += _px(26, s)
    for j, label in enumerate(col_labels):
        l_, t_, r_, b_ = _ink(label_font, str(label))
        draw.text((x0 + j * cell_w + cell_w // 2 - (r_ - l_) // 2 - l_, y - t_),
                  str(label), font=label_font, fill=_INK)
    y += _px(30, s)
    values = [float(v) for r in rows for v in r if v is not None and math.isfinite(float(v))]
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    if row_title:
        tile_font = label_font
        l_, t_, r_, b_ = _ink(tile_font, row_title)
        pad = _px(2, s)
        tile = Image.new("RGBA", (r_ - l_ + 2 * pad, b_ - t_ + 2 * pad),
                         (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((pad - l_, pad - t_), row_title,
                                  font=tile_font, fill=_MUTED)
        tile = tile.rotate(90, expand=True)
        img.paste(tile, (margin, y + (cell_h * n_rows) // 2 - tile.height // 2), tile)
    for i, (label, row) in enumerate(zip(row_labels, rows)):
        cy = y + i * cell_h
        l_, t_, r_, b_ = _ink(label_font, str(label))
        draw.text((x0 - _px(10, s) - (r_ - l_) - l_, cy + cell_h // 2 - (b_ - t_) // 2 - t_),
                  str(label), font=label_font, fill=_INK)
        for j in range(n_cols):
            value = row[j] if j < len(row) else None
            cx = x0 + j * cell_w
            fill = heat_cell_color(value, lo, hi) if value is not None else (245, 245, 245)
            draw.rectangle([cx, cy, cx + cell_w, cy + cell_h], fill=fill,
                           outline=(255, 255, 255), width=2)
            text = "" if value is None else cell_format.format(value)
            if text:
                l_, t_, r_, b_ = _ink(cell_font, text)
                dark = sum(fill) < 380
                draw.text((cx + cell_w // 2 - (r_ - l_) // 2 - l_,
                           cy + cell_h // 2 - (b_ - t_) // 2 - t_), text,
                          font=cell_font, fill=(255, 255, 255) if dark else _INK)
    return _save(img, Path(out_png), dpi=dpi)
