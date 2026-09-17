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


# ---------------------------------------------------------------------------
# Scaling, log axes and the second scale
# ---------------------------------------------------------------------------

def test_a_scale_of_one_is_the_identity_on_every_layout_constant():
    """
    The whole backward-compatibility story rests on this. Every distance and
    font size in the module is written at scale 1 and multiplied on the way
    out, so if `_px` is not exactly the identity there, every chart taters has
    ever drawn shifts by a pixel and the drawing tests below start lying.
    """
    for value in (2, 3, 4, 5, 6, 8, 10, 12, 13, 14, 15, 16, 18, 20, 22, 26,
                  30, 34, 40, 56, 110):
        assert charts._px(value, 1.0) == value, value


def test_log_ticks_land_on_the_data_rather_than_on_round_decades():
    """
    Decades over 5..2000 give 1, 10, 100, 1000, 10000 -- which pads the axis
    past both ends of the data and throws away a third of the width. The
    1-2-5 ladder puts its first and last tick *on* the sweep's own endpoints.
    """
    assert charts.log_ticks(5, 2000) == [5, 10, 20, 50, 100, 200, 500,
                                         1000, 2000]
    assert charts.log_ticks(0.001, 0.01)[0] <= 0.001
    for tick in charts.log_ticks(3, 900):
        mantissa = tick / 10 ** round(__import__("math").log10(tick) // 1)
        assert round(mantissa, 6) in (1.0, 2.0, 5.0), tick


def test_log_ticks_thin_themselves_rather_than_printing_forty_labels():
    assert len(charts.log_ticks(1e-3, 1e6)) <= 12


def test_a_log_axis_refuses_data_that_reaches_zero():
    """Silently dropping the offending point would draw a chart that lies
    about its data, so this is refused with the way out named."""
    with pytest.raises(ValueError, match="positive values"):
        charts.log_ticks(0, 100)
    with pytest.raises(ValueError, match="positive values"):
        charts.log_ticks(-5, 5)


def test_a_log_axis_places_the_geometric_middle_in_the_middle():
    ax = charts.Axes(0, 0, 100, 50, 1.0, 100.0, 0.0, 1.0, x_log=True)
    assert ax.px(10) == 50
    assert ax.px(1) == 0 and ax.px(100) == 100
    # and a stray non-positive value lands on the edge instead of throwing
    # from inside a draw loop with the chart half painted
    assert ax.px(0) == 0
    assert ax.px(-3) == 0


def test_a_second_y_range_keeps_the_same_rectangle_and_the_same_x():
    ax = charts.Axes(10, 5, 110, 55, 1.0, 100.0, 0.0, 1.0, x_log=True)
    other = ax.with_y(-50.0, 50.0)
    assert (other.left, other.top, other.right, other.bottom) == (10, 5, 110, 55)
    assert other.px(10) == ax.px(10)
    assert other.py(0.0) == (5 + 55) // 2


def test_the_two_scales_of_a_dual_axis_chart_are_independent(tmp_path):
    """
    The property the chart exists for. One series ranges over 0.01 and the
    other over 100; on a single shared scale the small one collapses to a
    flat line. Both must use most of the plot height.
    """
    ks = [5, 10, 20, 50, 100]
    out = charts.dual_axis_line_chart(
        {"tiny": [(k, 0.400 + 0.0001 * k) for k in ks]},
        {"huge": [(k, 100.0 - k) for k in ks]},
        tmp_path / "twin.png", title="Two scales", x_label="k",
        y_label="tiny", y2_label="huge", x_scale="log")

    from PIL import Image

    with Image.open(out) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        rows: dict = {}
        for i, color in enumerate(rgb.getdata()):
            x = i % width
            # only the plot interior: the legend swatches and the right-hand
            # tick labels are drawn in the series colors too, and counting
            # those made this pass even with both lines on one scale
            if not (0.15 * width < x < 0.65 * width):
                continue
            if color in (charts.PALETTE[0], charts.PALETTE[1]):
                rows.setdefault(color, []).append(i // width)
    assert len(rows) == 2, (
        "a series is missing from the plot area; it was drawn on the other "
        "series' scale and fell off the chart")
    for color, ys in rows.items():
        assert len(ys) > 200, color
        assert (max(ys) - min(ys)) / height > 0.25, (
            f"{color} collapsed to a flat line; the scales are not independent")


def test_a_dual_axis_chart_needs_something_on_each_side():
    with pytest.raises(ValueError, match="each side"):
        charts.dual_axis_line_chart({}, {"a": [(1, 1)]}, "x.png")


def test_dpi_is_written_only_when_it_is_asked_for(tmp_path):
    """A pHYs chunk makes Word and LaTeX place the figure at the size it was
    drawn for; its absence by default is what keeps old charts unchanged."""
    from PIL import Image

    series = {"a": [(1, 1.0), (2, 2.0)]}
    plain = charts.line_chart(series, tmp_path / "plain.png")
    with Image.open(plain) as img:
        assert "dpi" not in img.info
        assert img.size == (900, 560)

    printed = charts.line_chart(series, tmp_path / "print.png", dpi=300)
    with Image.open(printed) as img:
        assert round(img.info["dpi"][0]) == 300
        # and the drawing scaled with it, rather than staying a 3-inch figure
        assert img.size == (2812, 1750)


def test_scaling_grows_the_whole_drawing_not_just_the_canvas(tmp_path):
    from PIL import Image

    series = {"a": [(1, 1.0), (2, 2.0)]}
    small = charts.line_chart(series, tmp_path / "s.png", title="Title here")
    big = charts.line_chart(series, tmp_path / "b.png", title="Title here",
                            scale=2.0)
    with Image.open(big) as img:
        assert img.size == (1800, 1120)
    # ink, not just canvas: twice the size should carry well over twice the
    # non-white pixels, since strokes and glyphs thicken too
    def inked(path):
        with Image.open(path) as img:
            return sum(1 for c in img.convert("RGB").getdata() if c != (255, 255, 255))
    assert inked(big) > 2 * inked(small)


def test_an_unknown_x_scale_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown x_scale"):
        charts._x_ticks(1, 10, "logarithmic")


def test_the_right_hand_scale_gets_as_many_rows_as_the_left():
    """Five labels against six grid lines leaves the right-hand numbers
    floating between the lines, and the chart reads as broken."""
    for rows in (4, 5, 6, 7):
        ticks = charts._aligned_ticks(0.0, 0.83, rows)
        assert len(ticks) == rows
        assert ticks[0] <= 0.0 and ticks[-1] >= 0.83


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
