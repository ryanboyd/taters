"""
Tests for `taters.figures.charts`: the small chart renderer training reports
use. The layout arithmetic is pure and checked against numbers; the drawing
tests need Pillow and skip without it, checking the pictures by their pixels
(a line of the series color exists, the diagonal cell is the dark one).
"""
from __future__ import annotations

import pytest

from taters.figures import charts


def test_nice_ticks_cover_the_range_at_a_round_step():
    ticks = charts.nice_ticks(0.13, 0.87, 6)
    assert ticks[0] <= 0.13 and ticks[-1] >= 0.87
    steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
    assert len(steps) == 1 and steps.pop() in (0.1, 0.2, 0.25, 0.5)
    assert 4 <= len(ticks) <= 9
    assert charts.nice_ticks(1, 6, 6) == [1, 2, 3, 4, 5, 6]


def test_a_flat_range_still_gets_an_axis_and_reversed_bounds_are_fine():
    ticks = charts.nice_ticks(2.0, 2.0)
    assert ticks[0] < 2.0 < ticks[-1]
    assert charts.nice_ticks(5, 1) == charts.nice_ticks(1, 5)


def test_axes_map_data_to_pixels_monotonically_and_within_the_frame():
    ax = charts.Axes(50, 20, 450, 320, 0.0, 10.0, -1.0, 1.0)
    assert ax.px(0.0) == 50 and ax.px(10.0) == 450 and ax.px(5.0) == 250
    assert ax.py(-1.0) == 320 and ax.py(1.0) == 20, "y grows upward"
    assert ax.contains(ax.px(3.3), ax.py(0.2))
    assert not ax.contains(ax.px(11.0), ax.py(0.0))


def test_heat_colors_are_monotone_in_the_value():
    shades = [sum(charts.heat_cell_color(v, 0, 10)) for v in range(11)]
    assert shades == sorted(shades, reverse=True), "darker with larger values"
    assert charts.heat_cell_color(0, 0, 10) == (255, 255, 255)
    assert charts.heat_cell_color(10, 0, 10) == charts.PALETTE[0]
    assert charts.heat_cell_color(3, 5, 5) == (255, 255, 255), "a flat range is all white"


# --------------------------------------------------------------- with Pillow

PIL = pytest.importorskip("PIL")


def _colors(path):
    from PIL import Image

    with Image.open(path) as img:
        return set(img.convert("RGB").getdata())


def test_a_line_chart_draws_each_series_in_its_own_color(tmp_path):
    out = charts.line_chart({"train": [(1, 1.0), (2, 0.5), (3, 0.3)],
                             "validation": [(1, 1.2), (2, 0.7), (3, 0.6)]},
                            tmp_path / "loss.png", title="Loss", x_label="epoch",
                            y_label="loss")
    colors = _colors(out)
    assert charts.PALETTE[2] not in colors, "no third series was asked for"
    from collections import Counter

    from PIL import Image

    with Image.open(out) as img:
        assert img.size == (900, 560)
        counts = Counter(img.convert("RGB").getdata())
    # a drawn line is hundreds of pixels; the legend swatch alone is under 200,
    # so this is how we tell a plotted series from one that only shows up in the key.
    assert counts[charts.PALETTE[0]] > 500 and counts[charts.PALETTE[1]] > 500


def test_a_scatter_puts_its_points_inside_the_axes(tmp_path):
    from PIL import Image

    pts = [(x, x * 1.1) for x in range(0, 50, 2)]
    out = charts.scatter_chart(pts, tmp_path / "s.png", title="fit", note="r = 0.9",
                               x_label="observed", y_label="predicted")
    with Image.open(out) as img:
        rgb = img.convert("RGB")
        blueish = [(x, y) for x in range(rgb.width) for y in range(rgb.height)
                   if rgb.getpixel((x, y))[2] > 150 and rgb.getpixel((x, y))[0] < 150]
    assert blueish, "no points were drawn"
    xs = [x for x, _ in blueish]
    ys = [y for _, y in blueish]
    assert min(xs) > 40 and max(xs) < rgb.width - 10
    assert min(ys) > 30 and max(ys) < rgb.height - 40


def test_a_heat_table_darkens_its_largest_cell(tmp_path):
    from PIL import Image

    out = charts.heat_table([[9, 1], [2, 5]], tmp_path / "h.png",
                            row_labels=["a", "b"], col_labels=["a", "b"],
                            title="confusion", row_title="observed", col_title="predicted")
    with Image.open(out) as img:
        rgb = img.convert("RGB")
        assert charts.PALETTE[0] in set(rgb.getdata()), "the 9 cell is the saturated one"
        assert rgb.width > 200 and rgb.height > 100


def test_charts_say_what_is_missing_without_pillow():
    assert charts.pillow_missing_reason() is None
