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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce
from .render import (DEFAULT_FONT, _Fonts, _ink, _pillow, _wrap,
                     pillow_missing_reason)

__all__ = ["Axes", "nice_ticks", "line_chart", "scatter_chart", "heat_table",
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

    def px(self, x: float) -> int:
        span = (self.x_hi - self.x_lo) or 1.0
        return int(round(self.left + (float(x) - self.x_lo) / span * (self.right - self.left)))

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
           legend: Sequence[Tuple[str, RGB]] = ()) -> Axes:
    """Title, axes, grid, ticks, labels and legend; returns the axes."""
    margin = 18
    title_font, label_font, tick_font = fonts[20], fonts[15], fonts[13]
    y = margin
    if title:
        for line in _wrap(title_font, title, width - 2 * margin, 2):
            left, top, right, bottom = _ink(title_font, line)
            draw.text((margin - left, y - top), line, font=title_font, fill=_INK)
            y += (bottom - top) + 6
    top = y + 10
    # leave room on the left for the widest tick label plus the axis label.
    widest = max((tick_font.getlength(_fmt_tick(t)) for t in y_ticks), default=20)
    left = int(margin + (22 if y_label else 0) + widest + 10)
    bottom = int(height - margin - (26 if x_label else 0) - 22)
    right = width - margin
    legend_rows = list(legend)
    if legend_rows:
        right -= int(max(label_font.getlength(name) for name, _ in legend_rows) + 40)
    axes = Axes(left, top, right, bottom, min(x_ticks), max(x_ticks),
                min(y_ticks), max(y_ticks))
    # now the grid and the ticks.
    for t in y_ticks:
        py = axes.py(t)
        draw.line([(left, py), (right, py)], fill=_GRID, width=1)
        text = _fmt_tick(t)
        l_, t_, r_, b_ = _ink(tick_font, text)
        draw.text((left - 8 - (r_ - l_) - l_, py - (b_ - t_) // 2 - t_), text,
                  font=tick_font, fill=_MUTED)
    labels = list(x_tick_labels) if x_tick_labels is not None else [_fmt_tick(t) for t in x_ticks]
    for t, text in zip(x_ticks, labels):
        px = axes.px(t)
        draw.line([(px, top), (px, bottom)], fill=_GRID, width=1)
        l_, t_, r_, b_ = _ink(tick_font, text)
        draw.text((px - (r_ - l_) // 2 - l_, bottom + 6 - t_), text,
                  font=tick_font, fill=_MUTED)
    draw.rectangle([left, top, right, bottom], outline=(150, 150, 150), width=1)
    if x_label:
        l_, t_, r_, b_ = _ink(label_font, x_label)
        draw.text(((left + right) // 2 - (r_ - l_) // 2 - l_, height - margin - (b_ - t_) - t_),
                  x_label, font=label_font, fill=_INK)
    if y_label:
        # the y label is rotated, so we draw it on its own little image and
        # paste that in.
        Image, ImageDraw, _ImageFont = _pillow()
        l_, t_, r_, b_ = _ink(label_font, y_label)
        tile = Image.new("RGBA", (r_ - l_ + 4, b_ - t_ + 4), (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((2 - l_, 2 - t_), y_label, font=label_font, fill=_INK)
        tile = tile.rotate(90, expand=True)
        draw._image.paste(tile, (margin - 2, (top + bottom) // 2 - tile.height // 2), tile)
    # lastly, the legend, off to the right of the axes.
    ly = top
    for name, color in legend_rows:
        draw.rectangle([right + 12, ly + 3, right + 26, ly + 15], fill=color)
        draw.text((right + 32, ly), name, font=label_font, fill=_INK)
        ly += 22
    return axes


def _open(width: int, height: int, font_path: Optional[PathLike]):
    Image, ImageDraw, ImageFont = _pillow()
    img = Image.new("RGB", (int(width), int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    fonts = _Fonts(ImageFont, Path(font_path) if font_path else DEFAULT_FONT)
    return img, draw, fonts


def _save(img, out_png: Path) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_png, mode="wb") as fh:
        img.save(fh, format="PNG", optimize=True)
    return out_png


# ---------------------------------------------------------------------------
# the charts themselves
# ---------------------------------------------------------------------------

def line_chart(series: Mapping[str, Sequence[Point]], out_png: PathLike, *,
               title: str = "", x_label: str = "", y_label: str = "",
               width: int = 900, height: int = 560, include_zero: bool = False,
               integer_x: bool = True, font_path: Optional[PathLike] = None,
               on_progress=None) -> Path:
    """
    One line per series through its ``(x, y)`` points, in ``PALETTE`` order.

    Built for loss curves: ``{"fold 1 train": [(1, 0.9), (2, 0.7)], "fold 1
    validation": [...]}``. With ``integer_x`` the x ticks fall on whole
    numbers (epochs). ``include_zero`` forces the y axis to start at zero.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    img, draw, fonts = _open(width, height, font_path)
    xs = [p[0] for pts in series.values() for p in pts]
    ys = [p[1] for pts in series.values() for p in pts]
    x_lo, x_hi = _range(xs)
    y_lo, y_hi = _range(ys, include_zero=include_zero)
    x_ticks = nice_ticks(x_lo, x_hi, 8)
    if integer_x:
        x_ticks = sorted({int(round(t)) for t in x_ticks if t == int(t)} or {int(x_lo), int(x_hi)})
        if len(x_ticks) == 1:
            x_ticks = [x_ticks[0] - 1, x_ticks[0], x_ticks[0] + 1]
    y_ticks = nice_ticks(y_lo, y_hi, 6)
    legend = [(name, PALETTE[i % len(PALETTE)]) for i, name in enumerate(series)]
    axes = _frame(draw, fonts, width=width, height=height, title=title,
                  x_label=x_label, y_label=y_label, x_ticks=x_ticks,
                  y_ticks=y_ticks, legend=legend if len(series) > 1 else ())
    for i, (name, pts) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        pixels = [(axes.px(x), axes.py(y)) for x, y in pts
                  if y is not None and math.isfinite(float(y))]
        if len(pixels) > 1:
            draw.line(pixels, fill=color, width=3, joint="curve")
        for px, py in pixels:
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
    return _save(img, Path(out_png))


def scatter_chart(points: Sequence[Point], out_png: PathLike, *,
                  title: str = "", x_label: str = "", y_label: str = "",
                  identity: bool = True, note: str = "", width: int = 720,
                  height: int = 720, font_path: Optional[PathLike] = None,
                  on_progress=None) -> Path:
    """
    Points on square axes; with ``identity`` a dashed y = x line, so a
    predicted-versus-observed plot shows at a glance how far from perfect
    the predictions fall. ``note`` (r, R²) is written inside the axes.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    img, draw, fonts = _open(width, height, font_path)
    finite = [(float(x), float(y)) for x, y in points
              if x is not None and y is not None
              and math.isfinite(float(x)) and math.isfinite(float(y))]
    lo, hi = _range([v for p in finite for v in p]) if finite else (0.0, 1.0)
    ticks = nice_ticks(lo, hi, 6)
    axes = _frame(draw, fonts, width=width, height=height, title=title,
                  x_label=x_label, y_label=y_label, x_ticks=ticks, y_ticks=ticks)
    if identity:
        a, b = (axes.px(ticks[0]), axes.py(ticks[0])), (axes.px(ticks[-1]), axes.py(ticks[-1]))
        steps = 40
        for i in range(0, steps, 2):
            x0 = a[0] + (b[0] - a[0]) * i / steps
            y0 = a[1] + (b[1] - a[1]) * i / steps
            x1 = a[0] + (b[0] - a[0]) * (i + 1) / steps
            y1 = a[1] + (b[1] - a[1]) * (i + 1) / steps
            draw.line([(x0, y0), (x1, y1)], fill=(170, 170, 170), width=2)
    color = PALETTE[0] + (150,)
    overlay = img.copy().convert("RGBA")
    from PIL import ImageDraw as _ImageDraw

    layer = _ImageDraw.Draw(overlay, "RGBA")
    for x, y in finite:
        px, py = axes.px(x), axes.py(y)
        layer.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color)
    img.paste(overlay.convert("RGB"))
    if note:
        font = fonts[15]
        draw = _ImageDraw.Draw(img)
        left, top, right, bottom = _ink(font, note)
        draw.text((axes.left + 10 - left, axes.top + 8 - top), note, font=font, fill=_INK)
    return _save(img, Path(out_png))


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
               cell_format: str = "{:g}", font_path: Optional[PathLike] = None,
               on_progress=None) -> Path:
    """
    A matrix as a colored table: each cell shaded by its value and printed
    with it. Built for confusion matrices (rows observed, columns
    predicted), where the diagonal should be the dark one.
    """
    announce(on_progress, f"drawing chart: {title or Path(out_png).name}")
    n_rows, n_cols = len(rows), len(col_labels)
    cell_w, cell_h, margin = 110, 56, 18
    Image, ImageDraw, ImageFont = _pillow()
    fonts = _Fonts(ImageFont, Path(font_path) if font_path else DEFAULT_FONT)
    label_font, cell_font, title_font = fonts[14], fonts[16], fonts[20]
    label_w = int(max((label_font.getlength(str(x)) for x in row_labels), default=40)) + 20
    label_w += 26 if row_title else 0
    header_h = 30 + (26 if col_title else 0)
    title_h = 34 if title else 0
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
        y += 26
    for j, label in enumerate(col_labels):
        l_, t_, r_, b_ = _ink(label_font, str(label))
        draw.text((x0 + j * cell_w + cell_w // 2 - (r_ - l_) // 2 - l_, y - t_),
                  str(label), font=label_font, fill=_INK)
    y += 30
    values = [float(v) for r in rows for v in r if v is not None and math.isfinite(float(v))]
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    if row_title:
        tile_font = label_font
        l_, t_, r_, b_ = _ink(tile_font, row_title)
        tile = Image.new("RGBA", (r_ - l_ + 4, b_ - t_ + 4), (255, 255, 255, 0))
        ImageDraw.Draw(tile).text((2 - l_, 2 - t_), row_title, font=tile_font, fill=_MUTED)
        tile = tile.rotate(90, expand=True)
        img.paste(tile, (margin, y + (cell_h * n_rows) // 2 - tile.height // 2), tile)
    for i, (label, row) in enumerate(zip(row_labels, rows)):
        cy = y + i * cell_h
        l_, t_, r_, b_ = _ink(label_font, str(label))
        draw.text((x0 - 10 - (r_ - l_) - l_, cy + cell_h // 2 - (b_ - t_) // 2 - t_),
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
    return _save(img, Path(out_png))
