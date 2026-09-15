"""Tests for the hashbrowns hashbrowns.

The frame function is pure, so most of this is arithmetic on its output. The
two tests that open an application run it against a pipe input and a dummy
output, which is how prompt_toolkit expects to be driven without a terminal.
"""

from __future__ import annotations

import io
import re
import time

import numpy as np
import pytest
from prompt_toolkit.application import create_app_session, get_app_or_none
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from taters.ui import hashbrowns

STYLE = re.compile(r"^fg:#[0-9a-f]{6} bg:#[0-9a-f]{6}$")


def rendered(width: int, height: int, tick: int) -> list[str]:
    """Render a frame and hand back its plain text rows."""
    return hashbrowns.to_text(hashbrowns.frame(width, height, tick)).split("\n")


@pytest.mark.parametrize(
    "width,height", [(80, 24), (1, 1), (2, 3), (79, 23), (120, 31), (200, 50)]
)
def test_a_frame_fills_the_terminal_exactly(width, height):
    """Whatever the size, we get height rows of width cells and nothing spare."""
    lines = rendered(width, height, 300)
    assert len(lines) == height
    assert {len(line) for line in lines} == {width}


def test_a_frame_is_drawn_from_nothing_but_half_blocks_and_newlines():
    """Two pixels per cell only works if every cell really is a half block."""
    fragments = hashbrowns.frame(40, 12, 700)
    assert all(text == "\n" or set(text) == {hashbrowns.HALF_BLOCK} for _style, text in fragments)
    assert sum(1 for _style, text in fragments if text == "\n") == 11


def test_every_painted_fragment_carries_a_foreground_and_a_background_color():
    """The top pixel is the foreground and the bottom one is the background."""
    for style, text in hashbrowns.frame(40, 12, 700):
        if text == "\n":
            assert style == ""
        else:
            assert STYLE.match(style), style


@pytest.mark.parametrize("second", [0.5, 8, 22, 35, 48, 60, 72, 84])
def test_consecutive_ticks_render_different_frames(second):
    """Something moves in every scene, on every single frame."""
    tick = int(second * hashbrowns.FPS)
    assert hashbrowns.frame(60, 18, tick) != hashbrowns.frame(60, 18, tick + 1)


def test_the_palette_cycles_back_after_a_full_turn():
    """A plasma shimmers because its palette wraps instead of running out."""
    for role in ("main", "skin", "wild"):
        for index in (0, 37, 200):
            assert hashbrowns.palette_color(role, index) == hashbrowns.palette_color(
                role, index + hashbrowns.PALETTE_SIZE
            )
        assert len({hashbrowns.palette_color(role, i) for i in range(0, 256, 16)}) > 8


def test_the_loop_comes_round_without_a_join():
    """
    The last scene scrolls up into black, and the first one comes up out of
    it, so the frame before the wrap and the frame after it are the same
    picture and the demo never has a moment where it is obviously starting
    again.

    That picture is black on purpose. It used to be the storm's dim sky,
    which meant an unlit cloud sliding in a row at a time -- gray bands
    rather than weather. Scrolling into the dark and waking the storm out of
    it is both nicer to look at and an exact join, which is why this asks for
    the two frames to match rather than merely to be close.
    """
    last = hashbrowns.render(80, 24, int(hashbrowns.DEMO_SECONDS * hashbrowns.FPS) - 1)
    first = hashbrowns.render(80, 24, 0)
    assert np.abs(last - first).max() < 0.02
    assert first.max() == 0.0, "the loop no longer meets in the dark"

    # and it does not stay dark: the weather is up within a couple of seconds
    assert hashbrowns.render(80, 24, int(2.0 * hashbrowns.FPS)).max() > 0.05


def test_the_screen_is_full_of_color_once_the_demo_gets_going():
    """A frame mid-scene should be busy, not a flat wash."""
    styles = {style for style, text in hashbrowns.frame(80, 24, 400) if text != "\n"}
    assert len(styles) > 50


def test_the_running_order_covers_every_second_of_the_demo():
    """Each scene owns the stretch of the timeline between its neighbors."""
    assert hashbrowns._scene_at(0.0) == (0, 0.0)
    for index, (_name, length) in enumerate(hashbrowns.SCENES):
        start = hashbrowns._STARTS[index]
        assert hashbrowns._scene_at(start + length / 2) == (index, start)
    assert hashbrowns.DEMO_SECONDS == sum(length for _name, length in hashbrowns.SCENES)


def test_a_terminal_with_no_room_in_it_renders_nothing():
    """Sizes can arrive as zero while a terminal is being resized."""
    assert hashbrowns.frame(0, 10, 0) == []
    assert hashbrowns.frame(10, 0, 0) == []
    assert hashbrowns.frame(-4, -4, 9) == []


def test_the_size_is_read_from_the_output_for_every_frame():
    """A resize has to land on the next frame, not smear until we restart."""

    class Shrinking:
        def __init__(self):
            self.sizes = [Size(rows=10, columns=30), Size(rows=6, columns=12)]

        def get_size(self):
            return self.sizes.pop(0)

    output = Shrinking()
    first = hashbrowns.to_text(hashbrowns.frame_for_output(output, 100)).split("\n")
    second = hashbrowns.to_text(hashbrowns.frame_for_output(output, 101)).split("\n")
    assert (len(first), len(first[0])) == (10, 30)
    assert (len(second), len(second[0])) == (6, 12)


def test_the_font_has_a_glyph_for_every_character_the_demo_prints():
    """A missing glyph would come out as a hole in whatever line it is in."""
    spoken = ("".join(hashbrowns.SCROLL_TEXTS) + "".join(hashbrowns.BOOT_LINES)
              + "HASHBROWNS HASH BROWNS ONE POTATO JULIA GREETINGS")
    assert not {c for c in hashbrowns._glyphs(spoken) if c not in hashbrowns._FONT}


def test_the_potato_turns_without_changing_shape():
    """A rotation matrix that is not orthonormal would shear the potato."""
    matrix = hashbrowns._rotation(1.7)
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5)


def test_the_potato_is_longer_than_it_is_thick():
    """A potato is an oval that grew, not a ball with bumps on it."""
    x, y, z = hashbrowns._potato_mesh()
    assert np.ptp(x) > 1.4 * np.ptp(y)
    assert np.ptp(x) > 1.4 * np.ptp(z)


def test_the_potato_surface_is_not_a_smooth_ellipsoid():
    """Dividing out its own axes would leave a plain ellipsoid at exactly one."""
    x, y, z = hashbrowns._potato_mesh()
    ax, ay, az = hashbrowns.POTATO_AXES
    unit = np.sqrt((x / ax) ** 2 + (y / ay) ** 2 + (z / az) ** 2)
    assert unit.std() > 0.03
    assert np.ptp(unit) > 0.15


def test_the_potato_closes_up_where_it_comes_round():
    """Longitude wraps, so a swell at a fraction of a turn leaves a crease."""
    latitudes = np.linspace(-1.2, 1.2, 40)
    # the shape hands back points and nothing else. eyes and mottling used to
    # be carved into the radius here and come back alongside them; they are
    # painted on in `_potato_solid` now, because a real dent foreshortens into
    # a crease when the turntable catches it edge-on
    start = hashbrowns._potato_shape(latitudes, np.zeros_like(latitudes))
    end = hashbrowns._potato_shape(latitudes, np.full_like(latitudes, 2.0 * np.pi))
    assert np.abs(start - end).max() < 1e-5


def test_the_potato_comes_to_a_single_point_at_each_pole():
    """Every longitude meets there, so anything that varies with it must fade."""
    around = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    for pole in (-np.pi / 2, np.pi / 2):
        points = hashbrowns._potato_shape(np.full_like(around, pole), around)
        assert np.ptp(points, axis=1).max() < 1e-5


def test_the_demo_stops_itself_when_the_time_limit_runs_out():
    """Callers that cannot press a key still get their terminal back."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        start = time.monotonic()
        hashbrowns.run(max_seconds=0.3)
        elapsed = time.monotonic() - start
        assert get_app_or_none() is None
    assert 0.3 <= elapsed < 20.0


def test_escape_drops_the_demo_back_where_it_came_from():
    """The whole point is that it gets out of the way the moment you ask."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x1b")
        start = time.monotonic()
        hashbrowns.run(max_seconds=20)
        assert time.monotonic() - start < 8.0


def test_an_interrupt_drops_the_demo_back_too():
    """Ctrl-C is the other reflex, and in raw mode it arrives as a keypress."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\x03")
        start = time.monotonic()
        hashbrowns.run(max_seconds=20)
        assert time.monotonic() - start < 8.0


def test_any_ordinary_key_changes_the_colors_instead_of_quitting():
    """Leaning on the keyboard should repaint the demo, not end it."""
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("xyz 123")
        start = time.monotonic()
        hashbrowns.run(max_seconds=1.2)
        assert time.monotonic() - start >= 1.2


def test_a_broken_frame_still_hands_the_terminal_back(monkeypatch):
    """An exception must not leave the screen stuck in full-screen mode."""

    def explode(*args, **kwargs):
        raise RuntimeError("burnt")

    monkeypatch.setattr(hashbrowns, "frame_for_output", explode)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        with pytest.raises(RuntimeError, match="burnt"):
            hashbrowns.run(max_seconds=20)
        assert get_app_or_none() is None


def test_the_demo_takes_its_own_thread_when_something_else_owns_the_loop(monkeypatch):
    """A demo launched from a key binding cannot nest inside the running loop."""
    from prompt_toolkit.application import Application

    seen: list[dict] = []
    monkeypatch.setattr(Application, "run", lambda self, **kwargs: seen.append(kwargs))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        monkeypatch.setattr(hashbrowns, "_loop_is_busy", lambda: True)
        hashbrowns.run(max_seconds=1)
        monkeypatch.setattr(hashbrowns, "_loop_is_busy", lambda: False)
        hashbrowns.run(max_seconds=1)
    assert [call["in_thread"] for call in seen] == [True, False]


def replay(text: str, width: int, height: int) -> list[list[tuple]]:
    """Replay escape sequences into a grid of (foreground, background) pairs.

    This is a small terminal, good enough to prove that what we write lands
    where and in the color we meant.
    """
    cells = [[None] * width for _ in range(height)]
    fg = bg = None
    x = y = 0
    i = 0
    token = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
    while i < len(text):
        found = token.match(text, i)
        if found:
            args, command = found.group(1), found.group(2)
            if command == "H":
                parts = (args or "1;1").split(";")
                y = int(parts[0]) - 1
                x = int(parts[1]) - 1 if len(parts) > 1 else 0
            elif command == "m":
                nums = [int(v) for v in args.split(";") if v.isdigit()]
                while nums:
                    if nums[0] == 38:
                        fg, nums = tuple(nums[:5]), nums[5:]
                    elif nums[0] == 48:
                        bg, nums = tuple(nums[:5]), nums[5:]
                    else:
                        nums = nums[1:]
            i = found.end()
            continue
        if text[i] == hashbrowns.HALF_BLOCK:
            cells[y][x] = (fg, bg)
            x += 1
        i += 1
    return cells


def test_the_escape_sequences_paint_every_cell_of_the_screen():
    """The fast path has to cover the whole terminal, not most of it."""
    keys = hashbrowns._pack(hashbrowns.render(30, 9, 900))
    cells = replay(hashbrowns.ansi(keys, None, True), 30, 9)
    assert all(cell is not None for row in cells for cell in row)


def test_the_escape_sequences_agree_with_the_fragments():
    """Both paths draw the same picture, so either one can be trusted."""
    canvas = hashbrowns.render(24, 7, 1500)
    cells = replay(hashbrowns.ansi(hashbrowns._pack(canvas), None, True), 24, 7)
    styles = [
        style
        for style, text in hashbrowns._fragments(canvas)
        if text != "\n"
        for _ in text
    ]
    painted = [cell for row in cells for cell in row]
    assert len(painted) == len(styles) == 24 * 7
    for (fg, bg), style in zip(painted, styles):
        assert style == "fg:#%02x%02x%02x bg:#%02x%02x%02x" % (fg[2:] + bg[2:])


def test_a_frame_that_did_not_change_is_not_repainted():
    """Rows nobody touched cost us nothing, which is most of a quiet scene."""
    keys = hashbrowns._pack(hashbrowns.render(40, 12, 600))
    assert hashbrowns.ansi(keys, keys, True) == ""
    assert hashbrowns.ansi(keys, None, True) != ""


def test_only_the_columns_that_changed_get_repainted():
    """A one cell change must not redraw the row it happens to be on."""
    keys = hashbrowns._pack(hashbrowns.render(40, 12, 600))
    changed = keys.copy()
    changed[4, 20] = ~changed[4, 20] & 0x3FFFFFFF
    painted = hashbrowns.ansi(changed, keys, True)
    assert painted.count(hashbrowns.HALF_BLOCK) == 1
    assert "\x1b[5;21H" in painted


def test_a_terminal_without_true_color_gets_the_256_color_sequences():
    """Old terminals still get a picture, just a coarser one."""
    keys = hashbrowns._pack(hashbrowns.render(20, 6, 900))
    assert "38;5;" in hashbrowns.ansi(keys, None, False)
    assert "38;2;" not in hashbrowns.ansi(keys, None, False)
    assert "38;2;" in hashbrowns.ansi(keys, None, True)


def test_we_only_write_our_own_escapes_to_an_output_that_speaks_them():
    """A dummy output has no terminal behind it, so it goes the slow way."""
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.output.vt100 import Vt100_Output

    assert hashbrowns._direct_depth(DummyOutput()) is None
    out = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80),
                       default_color_depth=ColorDepth.DEPTH_24_BIT)
    assert hashbrowns._direct_depth(out) is True
    out = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80),
                       default_color_depth=ColorDepth.DEPTH_8_BIT)
    assert hashbrowns._direct_depth(out) is False
    out = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80),
                       default_color_depth=ColorDepth.DEPTH_4_BIT)
    assert hashbrowns._direct_depth(out) is None


@pytest.mark.parametrize("scheme", range(len(hashbrowns.SCHEMES)))
def test_every_color_scheme_draws_a_different_picture(scheme):
    """Six schemes, six looks, and none of them blank."""
    here = hashbrowns.frame(60, 18, 1500, scheme)
    assert len({style for style, text in here if text != "\n"}) > 40
    assert here != hashbrowns.frame(60, 18, 1500, (scheme + 1) % len(hashbrowns.SCHEMES))


def test_only_escape_and_an_interrupt_are_bound_to_quitting(monkeypatch):
    """Everything else has to land on the colors, including the arrow keys."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.keys import Keys

    captured: list = []
    monkeypatch.setattr(Application, "run", lambda self, **kwargs: captured.append(self))
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        hashbrowns.run(max_seconds=1)
    bindings = captured[0].key_bindings
    # prompt_toolkit runs the last match, so the specific binding wins
    for key in (Keys.Escape, Keys.ControlC):
        assert bindings.get_bindings_for_keys((key,))[-1].handler.__name__ == "_quit"
    for key in ("x", " ", Keys.Up, Keys.Enter):
        assert bindings.get_bindings_for_keys((key,))[-1].handler.__name__ == "_next_scheme"
    assert bindings.get_bindings_for_keys((Keys.Down,))[-1].handler.__name__ == "_previous_scheme"


@pytest.mark.parametrize("message", ["HI", "TATER", "A" * 500])
def test_the_scroller_crosses_the_screen_whatever_length_the_message_is(message):
    """Editing a message needs no other change: the speed comes out of its length.

    Whether it is two letters or five hundred, it comes in from the right, gets
    all the way across, and is gone by the time its run ends.
    """
    grid = hashbrowns._grid(40, 24)
    seen = {"right": False, "left": False, "gone": False}
    for step in range(41):
        canvas = hashbrowns._canvas(grid)
        hashbrowns._scroller(canvas, grid, step * 0.25, message, start=0.0, seconds=10.0)
        ink = canvas.sum(axis=2).sum(axis=0)
        seen["right"] |= bool(ink[-8:].sum() > 0)
        seen["left"] |= bool(ink[:8].sum() > 0)
        seen["gone"] |= bool(step > 2 and ink.sum() == 0)
    assert all(seen.values()), seen


@pytest.mark.parametrize("width,pixels", [(90, 100), (200, 120)])
def test_the_potato_is_solid_enough_to_have_no_holes_in_it(width, pixels):
    """The hero shot only works if the dense sample really covers the surface.

    A stray dark pixel here and there is a hairline crack nobody sees. What
    ruins it is a run of them: that reads as a hole straight through the potato,
    and it is what an unsampled cap at a pole of the grid looks like when the
    pole turns toward us.
    """
    grid = hashbrowns._grid(width, pixels)
    widest = 0
    interior_seen = 0
    for turn in range(8):
        canvas = hashbrowns._canvas(grid)
        hashbrowns._solid_potato(canvas, grid, 4.0 + turn * 1.7)
        lit = canvas.sum(axis=2) > 0.0
        # everything between the first and last lit pixel of a row is inside the
        # potato, so anything dark in there is something we can see through.
        left = np.maximum.accumulate(lit, axis=1)
        right = np.maximum.accumulate(lit[:, ::-1], axis=1)[:, ::-1]
        interior = left & right
        # the top and bottom rows of the silhouette curve away from us, and a
        # dip between two lit slivers there is the shape, not a hole
        edges = np.flatnonzero(lit.any(axis=1))
        interior[: edges[0] + 3] = False
        interior[edges[-1] - 2:] = False
        interior_seen = max(interior_seen, int(interior.sum()))
        holes = interior & ~lit
        runs = np.diff(np.concatenate([[0], holes.ravel().view(np.int8), [0]]))
        starts = np.flatnonzero(runs == 1)
        stops = np.flatnonzero(runs == -1)
        if starts.size:
            widest = max(widest, int((stops - starts).max()))
    assert interior_seen > 900
    assert widest <= 2


@pytest.mark.parametrize("width,pixels", [(80, 48), (100, 52), (200, 120)])
def test_the_vignette_leaves_the_edges_of_a_wide_terminal_readable(width, pixels):
    """The logo runs nearly the full width, so the sides cannot go dark."""
    vignette = hashbrowns._grid(width, pixels)["vign"][..., 0]
    middle = pixels // 2
    assert vignette[middle, width // 2] > 0.98
    assert vignette[middle, 0] > 0.6
    assert vignette[middle, -1] > 0.6
    assert vignette[0, 0] > 0.5
    assert vignette[0, 0] < vignette[middle, width // 2]


def test_the_scroller_is_outlined_so_it_never_blends_into_the_background():
    """It crosses every background the demo has and still has to be readable."""
    grid = hashbrowns._grid(60, 48)
    canvas = hashbrowns._canvas(grid)
    canvas[:] = 0.55  # a flat mid-bright wash, the worst case for a rainbow
    hashbrowns._scroller(canvas, grid, 6.0, start=0.0, seconds=30.0)
    assert (canvas.sum(axis=2) < 0.2).any()


def test_the_prologue_plays_once_at_the_front_of_the_running_order():
    """It is the opening titles: before the loader, and not again after that."""
    assert hashbrowns.SCENES[0] == ("storm", hashbrowns.STORM_SECONDS)
    assert hashbrowns.STORM_SECONDS == 28.2
    assert [name for name, _length in hashbrowns.SCENES].count("storm") == 1
    assert hashbrowns.SCENES[1][0] == "boot"
    assert hashbrowns._scene_at(hashbrowns.STORM_SECONDS / 2) == (0, 0.0)


def test_the_prologue_leaves_the_rest_of_the_demo_where_it_was():
    """An intro is only an intro: everything after it still plays in order."""
    rest = hashbrowns.SCENES[1:]
    assert hashbrowns.DEMO_SECONDS == hashbrowns.STORM_SECONDS + sum(length for _n, length in rest)
    for index, (_name, length) in enumerate(rest, start=1):
        start = hashbrowns._STARTS[index]
        assert start >= hashbrowns.STORM_SECONDS
        assert hashbrowns._scene_at(start + length / 2) == (index, start)


def test_the_prologue_opens_on_a_nearly_black_sky():
    """Clouds first, and between the flashes you can only just see them."""
    quiet = min(hashbrowns.render(80, 24, int(u * hashbrowns.FPS)).mean()
                for u in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5))
    assert quiet < 0.05


def test_the_lightning_flashes_and_then_dies_away_again():
    """Sheet lightning is a strobe, not a light somebody left on."""
    lit = [hashbrowns.render(60, 18, int(u * hashbrowns.FPS)).mean() for u in np.arange(0.3, 5.0, 0.05)]
    assert max(lit) > 8.0 * min(lit)
    assert float(np.median(lit)) < max(lit) / 3.0


def test_the_camera_pans_down_from_the_sky_into_the_fire():
    """The bottom of the screen is empty sky before the pan and fire after it."""
    sky = min(hashbrowns.render(80, 24, int(u * hashbrowns.FPS))[-12:].mean()
              for u in (2.2, 3.0, 3.8, 4.6, 7.0, 9.0))
    # the camera holds on the weather for nine and a half seconds and then
    # takes another three and a half getting down, so the fire is late
    fire = hashbrowns.render(80, 24, int(15.5 * hashbrowns.FPS))[-12:].mean()
    assert sky < 0.06
    assert fire > 0.25


def test_the_bolt_reaches_the_potato_it_is_aimed_at():
    """A bolt that stops short of the thing it hits is a bolt that missed."""
    grid = hashbrowns._grid(80, 48)
    canvas = hashbrowns._canvas(grid)
    hashbrowns._strike(canvas, grid, hashbrowns._HIT_AT + 0.15, (40.0, 30.0))
    lit = canvas.sum(axis=2) > 0.0
    assert lit[:4].any()
    assert lit[26:31].any()


def _storm_mean(dt: float) -> float:
    """How bright the whole frame is, `dt` seconds after the bolt lands."""
    return hashbrowns.render(80, 24,
                             int((hashbrowns._HIT_AT + dt) * hashbrowns.FPS)).mean()


def test_the_potato_cracks_open_slowly_instead_of_flashing_white():
    """
    The bolt used to land and the next frame was white, which read as a cut.
    Now it breaks open the way carbonite does: seams first, then light out of
    them, and only at the end the whole skin. That means it has to still be
    mostly dark a good second after the hit, and take its time getting bright.
    """
    assert _storm_mean(-0.6) < 0.3, "the storm is dark before the bolt"
    assert _storm_mean(1.0) < 0.35, "it cut straight to white again"

    # brighter every time you look, across the whole seven seconds
    steps = [_storm_mean(dt) for dt in (1.0, 2.5, 4.0, 5.5, 7.0)]
    assert steps == sorted(steps), steps
    assert steps[-1] - steps[0] > 0.05, "nothing much happened"
    # and most of that time is spent cracked rather than blown out: halfway
    # through, the scene is still a lit potato in a dark world
    assert _storm_mean(3.5) < 0.45, "it floods long before the seven seconds"

    # the light has to be coming out of the *potato*. measured over the whole
    # frame the fire alone brightens enough to look like progress, so this
    # takes a box around where the camera leaves it.
    def potato(dt):
        return hashbrowns.render(
            80, 24, int((hashbrowns._HIT_AT + dt) * hashbrowns.FPS)
        )[14:40, 22:58].mean()

    assert potato(6.5) > 0.85, "the potato never comes up to full glow"
    assert potato(6.5) - potato(0.5) > 0.5, "it is the fire doing the work"


def test_the_cracks_open_gradually_and_end_up_covering_the_potato():
    """
    `_cracks` walks a threshold down a field of seams. Early on only the
    sharpest are above it, which is a hairline; by the end everything is,
    which is a potato made of light. A linear walk floods it in the first
    second and there is nothing to watch, so the curve is the thing here.
    """
    pytest.importorskip("numpy")
    pts, _normals, _eyes, _mottle = hashbrowns._potato_solid(1)
    share = [float(hashbrowns._cracks(pts, glow).mean()) for glow in
             (0.0, 0.15, 0.35, 0.55, 0.75, 1.0)]
    assert share[0] == 0.0, "nothing is lit before the bolt lands"
    assert share == sorted(share), share
    assert share[1] < 0.06, "the first seams are hairlines, not patches"
    assert share[3] < 0.5, "over half the skin is gone by the middle"
    assert share[-1] > 0.98, "it never finishes as one lit thing"


def test_the_seams_stay_on_the_skin_while_the_potato_turns():
    """
    They are cracks in the potato, so they are computed from where each point
    sits on it rather than from where it happens to be on screen. Taken in
    screen space they would swim across the surface as it rotates.
    """
    np = pytest.importorskip("numpy")
    pts, _n, _e, _m = hashbrowns._potato_solid(1)
    assert not np.array_equal(hashbrowns._cracks(pts, 0.45),
                              hashbrowns._cracks(hashbrowns._rotation(2.0) @ pts, 0.45)), \
        "the seam field ignores where the points are"

    # and the potato hands it the points as they sit on the potato, not as the
    # rotation has left them -- so catch what it is actually called with
    seen = []
    real = hashbrowns._cracks
    try:
        hashbrowns._cracks = lambda given, glow: seen.append(given) or real(given, glow)
        grid = hashbrowns._grid(120, 80)
        hashbrowns._solid_potato(hashbrowns._canvas(grid), grid, 3.7,
                                 scale=0.3, glow=0.5)
    finally:
        hashbrowns._cracks = real
    assert seen, "_cracks was never called"
    home = {tuple(np.round(c, 4)) for c in pts.T}
    got = [tuple(np.round(c, 4)) for c in seen[0].T]
    assert got, "no points were handed over"
    # every point it measured is one of the potato's own, untouched by the spin
    assert sum(c in home for c in got) == len(got), "the seams were taken in screen space"


def test_the_glow_spreads_out_of_the_potato_to_swallow_the_scene():
    """Once the skin is all light, that light leaves the potato behind and
    takes the frame with it -- and it comes from where the potato stands."""
    pytest.importorskip("numpy")
    grid = hashbrowns._grid(120, 60)
    # the disc saturates, so there is no single brightest pixel to find --
    # what says where it started is which places it has reached yet
    early = hashbrowns._swallow(grid, (30.0, 20.0), 0.18)
    assert early is not None
    assert early[20, 30] > 0.9, "not lit where the potato is standing"
    assert early[55, 110] < 0.1, "already at the far corner"
    assert early.mean() < 0.5, "it covers everything before it has grown"

    assert hashbrowns._swallow(grid, (30.0, 20.0), 1.0).min() > 0.99, \
        "it never reaches the corners"
    assert hashbrowns._swallow(grid, (30.0, 20.0), 0.0) is None


def test_the_glare_lets_go_before_the_loader_has_to_type_through_it():
    """The prologue ends in light and the crossfade carries it into the
    loader -- but at full brightness the machine types behind a sheet."""
    assert _storm_mean(8.6) > 0.85, "it never gets to the whiteout"
    assert hashbrowns._glare_fade(hashbrowns.STORM_SECONDS) < 0.4
    assert hashbrowns._glare_fade(hashbrowns._HIT_AT) == 1.0


def test_the_loader_types_itself_out_instead_of_arriving_all_at_once():
    """It is a machine waking up, and the whole charm of it is the typing."""
    boot = hashbrowns._STARTS[1]
    early = hashbrowns.render(80, 24, int((boot + 0.9) * hashbrowns.FPS))
    late = hashbrowns.render(80, 24, int((boot + 4.0) * hashbrowns.FPS))
    # 0.9 rather than 0.35: the loader's own plasma backdrop sits just above
    # the lower threshold, so that counted background as typing and moved with
    # whatever absolute time the scene happened to start at
    assert (early.sum(axis=2) > 0.9).sum() < (late.sum(axis=2) > 0.9).sum() / 2


def test_the_loader_blinks_a_cursor_once_it_has_finished_typing():
    """A cursor that is always on is not a cursor, it is a block."""
    boot = hashbrowns._STARTS[1]
    last = hashbrowns.BOOT_LINES[-1]
    x0, gap = hashbrowns._centered_at(80, last)
    # the column just past the end of the last line holds the cursor and
    # nothing else, which keeps the moving scanline out of the count
    left = min(x0 + len(last) * (8 + gap), 80 - 8)
    cursor = slice(left, left + 8)
    lit = [(hashbrowns.render(80, 24, int((boot + 3.7 + step * 0.05) * hashbrowns.FPS))[:, cursor]
            .sum(axis=2) > 1.2).sum() for step in range(10)]
    assert max(lit) - min(lit) >= 40


def test_the_scroller_waits_for_its_scene_to_be_up_before_it_sets_off():
    """Otherwise you meet the message halfway through its own first word."""
    grid = hashbrowns._grid(100, 52)
    band = hashbrowns._scroller_top(grid)
    began = hashbrowns._STARTS[hashbrowns._ORDER["plasma"]] + hashbrowns.SCROLL_DELAY
    for when, wanted in ((began - 0.2, False), (began + 3.0, True)):
        canvas = hashbrowns._canvas(grid)
        hashbrowns._scroller(canvas, grid, when)
        # numpy's booleans are not python's, so ask for one plainly
        assert bool(canvas[band:].sum() > 0.0) is wanted


def test_the_scroller_walks_in_from_the_right_hand_edge():
    """The front of the message arrives first, not the middle of it."""
    grid = hashbrowns._grid(100, 52)
    canvas = hashbrowns._canvas(grid)
    began = hashbrowns._STARTS[hashbrowns._ORDER["plasma"]] + hashbrowns.SCROLL_DELAY
    hashbrowns._scroller(canvas, grid, began + 1.4)
    lit = (canvas.sum(axis=2) > 0.0).any(axis=0)
    assert not lit[:40].any()
    assert lit[60:].any()


@pytest.mark.parametrize("name", [name for name, _length in hashbrowns.SCENES
                                  if name not in ("storm", "boot")])
def test_the_scenes_that_carry_the_scroller_are_the_ones_that_say_they_do(name, monkeypatch):
    """The running order and the drawing have to agree about where it rests."""
    grid = hashbrowns._grid(100, 52)
    when = hashbrowns._STARTS[hashbrowns._ORDER[name]] + 5.0
    drawn = hashbrowns._SCENE_FUNCS[name](grid, when).copy()
    monkeypatch.setattr(hashbrowns, "_scroller", lambda *args, **kwargs: None)
    without = hashbrowns._SCENE_FUNCS[name](grid, when)
    band = hashbrowns._scroller_top(grid)
    carries = not np.array_equal(drawn[band:], without[band:])
    assert carries is (name in hashbrowns.SCROLL_SCENES)


def test_the_scroller_comes_back_in_from_the_edge_every_time_it_returns():
    """It rests through the hero shots, so each return is a fresh entrance."""
    runs = []
    for index, (name, _length) in enumerate(hashbrowns.SCENES):
        if name in hashbrowns.SCROLL_SCENES and (
            index == 0 or hashbrowns.SCENES[index - 1][0] not in hashbrowns.SCROLL_SCENES
        ):
            runs.append(hashbrowns._STARTS[index])
    assert len(runs) > 1  # it really does take a rest somewhere in the middle
    # and a scene in the middle of a run does not restart the message
    assert hashbrowns._scroll_run(hashbrowns._STARTS[hashbrowns._ORDER["tunnel"]] + 1.0)[1] == (
        runs[0] + hashbrowns.SCROLL_DELAY)
    grid = hashbrowns._grid(100, 52)
    for begins in runs:
        run, began, room = hashbrowns._scroll_run(begins + 1.0)
        assert began == begins + hashbrowns.SCROLL_DELAY
        # each run sets its own speed, so we ask each one at the point it has
        # come the same distance in rather than after the same length of time
        span = hashbrowns._text_bits(hashbrowns.SCROLL_TEXTS[run]).shape[1]
        canvas = hashbrowns._canvas(grid)
        hashbrowns._scroller(canvas, grid, began + 60.0 * room / (span + 100))
        lit = (canvas.sum(axis=2) > 0.0).any(axis=0)
        assert not lit[:50].any()
        assert lit[60:].any()


def test_the_potato_has_blunt_ends_rather_than_pointed_ones():
    """Tapering smoothly to a point is what makes a shape read as an egg.

    A plain ellipsoid of these axes is down to 60 per cent of its width by the
    outer fifth of its length. A potato is still most of the way out there.
    """
    x, y, z = hashbrowns._potato_mesh()
    girth = np.sqrt(y**2 + z**2)
    ends = np.abs(x) > 0.8 * np.abs(x).max()
    assert girth[ends].max() > 0.75 * girth.max()


def test_every_pass_of_the_scroller_gets_its_own_message():
    """Three runs, three messages, and none of them repeats another's words."""
    runs = [index for index, (name, _length) in enumerate(hashbrowns.SCENES)
            if name in hashbrowns.SCROLL_SCENES
            and (index == 0 or hashbrowns.SCENES[index - 1][0] not in hashbrowns.SCROLL_SCENES)]
    assert len(hashbrowns.SCROLL_TEXTS) == len(runs)
    assert len({text.strip() for text in hashbrowns.SCROLL_TEXTS}) == len(runs)
    for pass_number, index in enumerate(runs):
        assert hashbrowns._scroll_run(hashbrowns._STARTS[index] + 1.0)[0] == pass_number


def test_no_message_ends_up_going_faster_than_anybody_can_read():
    """The speed falls out of the length, so a long message is the way it bites."""
    for index, text in enumerate(hashbrowns.SCROLL_TEXTS):
        start = next(hashbrowns._STARTS[i] for i, (name, _l) in enumerate(hashbrowns.SCENES)
                     if name in hashbrowns.SCROLL_SCENES
                     and (i == 0 or hashbrowns.SCENES[i - 1][0] not in hashbrowns.SCROLL_SCENES)
                     and sum(1 for j in range(i)
                             if hashbrowns.SCENES[j][0] in hashbrowns.SCROLL_SCENES
                             and (j == 0 or hashbrowns.SCENES[j - 1][0] not in hashbrowns.SCROLL_SCENES)
                             ) == index)
        _run, _began, room = hashbrowns._scroll_run(start + 1.0)
        # 200 columns is a generous terminal, and a wider one has further to go
        span = hashbrowns._text_bits(text).shape[1] + 200
        assert span / room <= hashbrowns.SCROLL_READABLE_SPEED


def test_a_heart_is_one_glyph_even_though_you_type_two_characters():
    """So that "<3" comes out as a heart and nothing miscounts the line."""
    assert len(hashbrowns._glyphs("WE <3 SPUDS")) == len("WE <3 SPUDS") - 1
    heart = hashbrowns._text_bits(hashbrowns.HEART)
    plain = hashbrowns._text_bits("<")
    assert heart.shape[1] == plain.shape[1]
    # the widest row of a heart is at the top, unlike anything else we draw
    assert heart[0].sum() == 4
    assert heart[1].sum() == heart[2].sum() == 8


def test_the_scroller_is_gone_before_the_potato_comes_apart():
    """The sign off wants the screen to itself, so the message finishes early."""
    _run, began, room = hashbrowns._scroll_run(hashbrowns.DEMO_SECONDS - 1.0)
    assert began + room == hashbrowns.DEMO_SECONDS - hashbrowns.FAREWELL_SECONDS
    grid = hashbrowns._grid(100, 52)
    canvas = hashbrowns._canvas(grid)
    hashbrowns._scroller(canvas, grid, began + room + 0.5)
    assert canvas[hashbrowns._scroller_top(grid):].sum() == 0.0


def test_the_farewell_only_shows_once_the_potato_is_coming_apart(monkeypatch):
    """It arrives out of the blast rather than sitting there through the scene."""
    said = []
    drawn = hashbrowns._centered

    def watched(canvas, text, *args, **kwargs):
        said.append(text)
        return drawn(canvas, text, *args, **kwargs)

    monkeypatch.setattr(hashbrowns, "_centered", watched)
    grid = hashbrowns._grid(100, 52)
    ends = hashbrowns.DEMO_SECONDS - hashbrowns.FAREWELL_SECONDS
    hashbrowns._sc_finale(grid, ends - 2.0)
    assert [line for line in said if line in hashbrowns.FAREWELL] == []
    said.clear()
    hashbrowns._sc_finale(grid, ends + 2.5)
    assert [line for line in said if line in hashbrowns.FAREWELL] == list(hashbrowns.FAREWELL)


def test_the_potato_comes_apart_and_leaves_nothing_behind():
    """It shears into a storm that sweeps past us, and then the screen is clear."""
    grid = hashbrowns._grid(100, 52)

    def spread(burst):
        canvas = hashbrowns._canvas(grid)
        hashbrowns._chrome_potato(canvas, grid, 150.0, burst=burst)
        rows, cols = np.nonzero(canvas.sum(axis=2) > 0.0)
        return float(np.hypot(cols.std(), rows.std())) if rows.size else 0.0

    whole = spread(0.0)
    assert whole > 0.0
    assert spread(2.0) > whole * 1.6
    blown = hashbrowns._canvas(grid)
    hashbrowns._chrome_potato(blown, grid, 150.0, burst=hashbrowns.FAREWELL_SECONDS)
    assert blown.max() == 0.0


def test_the_chrome_potato_is_hard_edged_the_way_metal_is():
    """Polished metal reads as metal because of the edges between light and dark.

    A lit surface slides from one tone to the next; a mirror cuts between them,
    so neighboring pixels on the chrome one are far further apart than on the
    matte one.
    """
    grid = hashbrowns._grid(100, 52)

    def step(canvas):
        value = canvas.max(axis=2)
        lit = value > 0.0
        return float(np.abs(np.diff(value, axis=1))[lit[:, 1:] & lit[:, :-1]].mean())

    chrome = hashbrowns._canvas(grid)
    hashbrowns._chrome_potato(chrome, grid, 150.0)
    matte = hashbrowns._canvas(grid)
    hashbrowns._solid_potato(matte, grid, 150.0, scale=0.42)
    assert step(chrome) > 3.0 * step(matte)


def test_the_potato_has_spots_rather_than_being_evenly_darker():
    """Patches that overlap into one wash stop reading as spots at all.

    Two different marks live on the skin and they are not interchangeable:
    `_potato_spots` paints many small shallow patches of tan, `_potato_eyes`
    paints fewer, bigger, darker ones. So the deepest spot stays inside the
    band the spots declare, and an eye always goes deeper than any of them.
    """
    _pts, _normals, eyes, mottle = hashbrowns._potato_solid(1)
    assert 0.5 < mottle.max() <= 0.7, "a spot is a shallow patch of tan"
    assert eyes.max() > mottle.max(), "an eye is deeper than any spot"
    # counted at a depth that is well inside a spot rather than at its darkest
    # point: these are shallow, so asking for half-dark would clip most of
    # every one of them and make the skin look barer than it is
    assert 0.05 < float((mottle > 0.2).mean()) < 0.35, "patches, not a wash"


def test_the_spots_reach_the_potato_we_actually_draw(monkeypatch):
    """It is only skin if it survives the trip from the model to the screen."""
    grid = hashbrowns._grid(110, 60)
    try:
        hashbrowns._CACHE.clear()
        spotted = hashbrowns._canvas(grid)
        hashbrowns._solid_potato(spotted, grid, 56.0, scale=0.42)
        hashbrowns._CACHE.clear()
        monkeypatch.setattr(hashbrowns, "_potato_spots",
                            lambda pts, count=22: np.zeros(pts.shape[1], np.float32))
        plain = hashbrowns._canvas(grid)
        hashbrowns._solid_potato(plain, grid, 56.0, scale=0.42)
    finally:
        hashbrowns._CACHE.clear()
    assert not np.array_equal(spotted, plain)
    assert spotted.sum() < plain.sum()  # and what changed went darker, not lighter


def test_the_tunnel_is_muted_enough_to_read_the_scroller_over():
    """Same pattern as before; it just stops shouting over the message."""
    grid = hashbrowns._grid(100, 52)
    band = hashbrowns._scroller_top(grid)
    when = hashbrowns._STARTS[hashbrowns._ORDER["tunnel"]] + 5.0
    behind = hashbrowns._tunnel(grid, when)[band:].max(axis=2)
    ink = hashbrowns._canvas(grid)
    hashbrowns._scroller(ink, grid, when)
    lit = ink[band:].max(axis=2)
    assert float(np.median(behind)) < 0.6 * float(lit[lit > 0.5].mean())
    # and it is still a pattern rather than a flat wash
    assert float(behind.std()) > 0.08


def test_the_paint_rate_eases_off_on_a_big_terminal():
    """A frame costs bytes by the cell, so a big window costs more to send."""
    assert hashbrowns.paint_rate(80 * 24) == hashbrowns.FPS
    assert hashbrowns.paint_rate(hashbrowns.CELLS_AT_FULL_RATE) == hashbrowns.FPS
    assert hashbrowns.paint_rate(200 * 60) == hashbrowns.EASED_FPS
    # never faster than the usual rate, never slower than the eased one
    rates = [hashbrowns.paint_rate(cells) for cells in range(500, 60000, 500)]
    assert rates == sorted(rates, reverse=True)
    assert min(rates) == hashbrowns.EASED_FPS
    assert max(rates) == hashbrowns.FPS


def test_a_rate_asked_for_outright_is_honored_whatever_the_size():
    """Easing off is a default, not a policy imposed on the caller."""
    for cells in (80 * 24, 200 * 60, 400 * 120):
        assert hashbrowns.paint_rate(cells, 60) == 60.0
        assert hashbrowns.paint_rate(cells, 5) == 5.0


def test_frames_we_have_already_missed_are_not_chased():
    """A terminal that cannot keep up leaves us behind, and behind is behind.

    Retrying immediately would make pictures nobody sees, at full tilt, for as
    long as the terminal stays slow.
    """
    period = 1.0 / 30.0
    # a frame due in the future is left where it is
    assert hashbrowns._next_due(110.0, 105.0, period) == 110.0
    # one already past is given up on, and the next is a whole period away
    assert hashbrowns._next_due(100.0, 105.0, period) == pytest.approx(105.0 + period)
    # so however far behind we fall, the wait never collapses to nothing
    due = 0.0
    for now in (10.0, 20.0, 30.0):
        due = hashbrowns._next_due(due + period, now, period)
        assert due - now == pytest.approx(period)


def test_the_spots_reach_every_side_of_the_potato():
    """Thrown at random they leave whole sides of it bare.

    Spots at random directions put the emptiest patch forty degrees or more
    from the nearest one, which is most of a face with nothing on it. Spacing
    them out first and then knocking them off their marks brings that down to
    thirty without making them look laid out.
    """
    points, _normals, _eyes, mottle = hashbrowns._potato_solid(1)
    facing = points / np.sqrt((points * points).sum(axis=0))
    # a point is inside a spot, not necessarily at its darkest: see the depth
    # band in `_potato_spots`
    spotted = facing[:, mottle > 0.2]
    # for a sample of the surface, how far away is the nearest spot?
    nearest = (facing[:, ::37].T @ spotted).max(axis=1)
    worst = np.degrees(np.arccos(np.clip(nearest.min(), -1.0, 1.0)))
    assert worst < 35.0


def test_the_pan_in_scrolls_the_terminal_rather_than_wiping_it():
    """Whatever the caller had on screen has to leave by the top, not vanish."""
    keys = hashbrowns._pack(hashbrowns.render(40, 12, 0))
    painted = hashbrowns._pan_in_row(keys, 3, 40, 12, True)
    assert painted.startswith("\x1b[S")  # ask the terminal to scroll up a line
    assert "\x1b[12;1H" in painted  # and fill the gap that opens at the bottom
    assert painted.count(hashbrowns.HALF_BLOCK) == 40
    assert "\x1b[2J" not in painted  # nothing here wipes anything


def test_we_keep_off_the_alternate_screen_when_we_paint_for_ourselves(monkeypatch):
    """There is nothing to push off the top of a buffer we just switched to."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.output.vt100 import Vt100_Output

    seen: list = []
    monkeypatch.setattr(Application, "run", lambda self, **kwargs: seen.append(self))
    monkeypatch.setattr(hashbrowns, "_pan_in", lambda *args, **kwargs: None)
    terminal = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80),
                            default_color_depth=ColorDepth.DEPTH_24_BIT)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=terminal):
        hashbrowns.run(max_seconds=1)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        hashbrowns.run(max_seconds=1)
    assert [app.full_screen for app in seen] == [False, True]


def test_it_hands_the_terminal_back_the_way_it_found_it(monkeypatch):
    """We turned the cursor and the autowrap off, so we turn them back on."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.output.vt100 import Vt100_Output

    monkeypatch.setattr(Application, "run", lambda self, **kwargs: None)
    monkeypatch.setattr(hashbrowns, "_pan_in", lambda *args, **kwargs: None)
    paper = io.StringIO()
    terminal = Vt100_Output(paper, lambda: Size(rows=24, columns=80),
                            default_color_depth=ColorDepth.DEPTH_24_BIT)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=terminal):
        hashbrowns.run(max_seconds=1)
    written = paper.getvalue()
    assert "\x1b[?7h" in written  # autowrap back on
    assert "\x1b[?25h" in written  # and the cursor back
    assert "\x1b[2J" in written  # and nothing of ours left on the screen


def test_the_demo_scrolls_into_a_dark_sky_and_the_storm_starts_after():
    """
    Lighting the cloud on the way in was tried and did not look right, so the
    approach is deliberately dark: the finale pans into an unlit sky and the
    first flash comes once the prologue proper has started.

    What makes that worth the wait is the length of the hold -- nine and a
    half seconds of weather before the camera tilts down, where it used to be
    five and a half and the storm barely got going before the descent.
    """
    schedule = hashbrowns._flashes()
    assert all(start >= 0.0 for start, *_rest in schedule), \
        "something is lighting the scroll-in"
    assert schedule[0][0] == 0.5

    assert hashbrowns._PAN_AT > 9.0, "the hold on the sky is back to a glance"
    # and the extra time thickens the lightning rather than spreading the same
    # few bolts thinner, because the schedule is written in terms of the hit
    assert len(schedule) >= 18
    during_hold = [s for s, *_r in schedule if s < hashbrowns._PAN_AT]
    assert len(during_hold) >= 7, during_hold


def test_the_pan_back_into_the_storm_moves_every_frame():
    """
    It used to round to whole rows, and a whole row is a big step: thirty-one
    frames out of seventy-eight did not move at all and the rest jumped one.
    That reads as juddering rather than as a camera move.
    """
    np = pytest.importorskip("numpy")
    start = hashbrowns.DEMO_SECONDS - hashbrowns.LOOP_BACK_SECONDS
    # stop before LOOP_SETTLE_SECONDS, where the camera has arrived on purpose
    last = hashbrowns.DEMO_SECONDS - hashbrowns.LOOP_SETTLE_SECONDS - 0.1
    frames = [hashbrowns.render(80, 24, int(t * hashbrowns.FPS))
              for t in np.arange(start + 0.2, last, 1.0 / hashbrowns.FPS)]
    still = sum(1 for a, b in zip(frames, frames[1:])
                if float(np.abs(b - a).mean()) < 1e-4)
    assert still <= 2, f"{still} frames of {len(frames) - 1} did not move"
