"""hashbrowns: a potato-themed demoscene production for the terminal.

Two half-block pixels per cell give us a canvas of ``width`` by ``height * 2``
roughly square pixels, which is room enough for eleven scenes of the classics:
a loader, a plasma with copper bars, a texture tunnel, Sierpinski drawn two
ways, a lit potato turning in the dark, a Julia set, a starfield, calculator
curves, a rotozoomer, a twister and a finale. Every effect is vector math over
one numpy RGB array. Six color schemes fill the same five palette roles, so up
and down repaint the whole production without touching an effect.

The public surface is deliberately small:

* ``frame(width, height, tick, scheme)`` is pure. A size and a frame counter go
  in, prompt_toolkit fragments come out, so it can be tested without a terminal.
* ``run()`` opens its own full-screen application and exits on escape.
* ``python -m taters.ui.hashbrowns`` runs the whole thing standalone.

There are two ways out to the screen. ``frame`` builds prompt_toolkit fragments
and is what the app would use; ``ansi`` builds the escape sequences itself and
only repaints what changed, which is what ``run`` uses on a terminal we know
speaks vt100. The renderer is the expensive part of a frame, not the math, and
skipping it is the difference between twenty frames a second and thirty.
"""

from __future__ import annotations

import time

# the demo clock is fixed so the production runs at the same speed whatever
# refresh rate the terminal can actually keep up with.
FPS = 30
PALETTE_SIZE = 256

# one cell holds two pixels: the upper half block paints foreground on top and
# lets the background color show through underneath.
HALF_BLOCK = "▀"

# near black, but not quite, so outlined letters still read as drawn.
INK = (0.02, 0.0, 0.01)

# the potato's three axes, longest first. the shape and its test share them.
POTATO_AXES = (1.42, 0.77, 0.92)

# blank columns between glyphs. the font fills all eight of its columns on
# letters like A and M, so without this the words run together at any scale.
GAP = 1

# the running order, in seconds. it loops forever, so the last scene fades down
# and the first fades up.
SCENES = (
    ("storm", 28.2),  # must match STORM_SECONDS below
    ("boot", 5.5),
    ("plasma", 11.0),
    ("tunnel", 11.0),
    ("sierpinski", 11.0),
    ("spud", 12.0), # quiet time, no scroll text
    ("julia", 10.0), # quiet time, no scroll text
    ("stars", 19.0),
    ("curves", 20.0),
    ("roto", 9.0),
    ("twister", 22.0), # begin third phase
    ("finale", 22.0),
)
CROSSFADE = 0.9
_STARTS = tuple(sum(d for _n, d in SCENES[:i]) for i in range(len(SCENES)))
_ORDER = {name: i for i, (name, _length) in enumerate(SCENES)}
DEMO_SECONDS = sum(d for _n, d in SCENES)


def _since(name: str, t: float) -> float:
    """Return how long the named scene has been up at demo time ``t``.

    Scenes are handed the demo clock, not their own, because a plasma that
    restarts at every cut looks wrong. Anything that does have to start when its
    scene does, like the loader typing itself out, asks here instead.
    """
    return t - _STARTS[_ORDER[name]]

# How long the scroller waits after its scene arrives before it sets off.
SCROLL_DELAY = 2.2

# The scenes that carry the scroller. The hero shots go without it on purpose,
# and every time it comes back it comes back from the right hand edge rather
# than halfway through a word nobody saw the start of.
SCROLL_SCENES = frozenset(
    {"plasma", "tunnel", "sierpinski", "stars", "curves", "twister", "finale"}
)


def _scroll_run(t: float) -> tuple[int, float, float]:
    """Return which pass of the scroller this is, when it set off, and its room.

    The scroller rests through the scenes that do not carry it, so a pass is one
    unbroken run of the scenes that do. We walk out to both ends of the run the
    moment lands in: the front is where this message came in from the right, and
    the back is how much time it has to get all the way across.
    """
    first, _start = _scene_at(t % DEMO_SECONDS)
    while first > 0 and SCENES[first - 1][0] in SCROLL_SCENES:
        first -= 1
    last = first
    while last + 1 < len(SCENES) and SCENES[last + 1][0] in SCROLL_SCENES:
        last += 1
    run = sum(
        1
        for i in range(first)
        if SCENES[i][0] in SCROLL_SCENES and (i == 0 or SCENES[i - 1][0] not in SCROLL_SCENES)
    )
    began = _STARTS[first] + SCROLL_DELAY
    ends = _STARTS[last] + SCENES[last][1]
    if last == len(SCENES) - 1:
        ends -= FAREWELL_SECONDS  # the sign off wants the screen to itself
    return run, began, ends - began

# What the loader types out. Every line has to fit the narrow axis: at eighty
# columns we get ten glyphs, so it talks in short sentences.
# What the last scene says on the way out, and how long it keeps the screen to
# itself to say it. The scroller gives those seconds back out of its own run, so
# the message is gone before the potato comes apart.
FAREWELL = ("THANKS FOR", "WATCHING!", "[ESC] TO EXIT")
FAREWELL_SECONDS = 6.0

# The tail of that, spent panning back into the sky the demo opened on, and the
# moment it holds there before coming round. The join is seamless because what
# pans in is the opening frame itself, drawn at the time it is about to be.
LOOP_BACK_SECONDS = 2.6
LOOP_SETTLE_SECONDS = 0.35

BOOT_LINES = ("TATERS", "PRESENTS:", "HASHBROWNS!", "ALL SPUDS,", "NO DUDS!")

# One message for each unbroken run of scenes that carries the scroller. Each
# one plays through once, at whatever speed gets it finished exactly as its run
# ends, so a message is never cut off and never starts over halfway read. Write
# them as long or as short as you like; there is a test that none of them ends
# up going faster than anybody can read.
SCROLL_TEXTS = (
    "   HASHBROWNS:  ALL SPUDS, NO DUDS!         "
    "PO-TA-TOES? BOIL 'EM, MASH 'EM, STICK 'EM IN A STEW?         "
    "FOR ALL THE WORD NERDS AND DATA DORKS OUT THERE DOING AWESOME WORK!   ",

    "   GREETZ TO:   <3 ZELDA <3         NATALIE P.         "
    "JAMIE PENNEBAKER         KATE BLACKBURN         THE REALM LAB         "
    "H. ANDREW SCHWARTZ  &  THE HLAB         MATTHIAS MEHL         DAVE MARKOWITZ         "
    "WEIXI WANG         MICHELLE DROUIN         ADAM K. FETTERMAN",

    "   MORE GREETZ:   CINDY K. CHUNG         PAOLA PASCA         "
    "TABEA MEIER         HEATHER LEFFEW         MORTEZA DEHGHANI         "
    "ANDREA HORN         SVAPS         VIVIAN TA-JOHNSON         "
    "SAL GIORGI         OSCAR KJELL         THE ROWDY BOYS         "
    "DAVID GARCIA         KATIE HOEMANN     &     M. ATARI!   ",
)

# Nobody can read a scroller much past this, so it is what the test holds the
# messages to rather than anything the drawing enforces.
SCROLL_READABLE_SPEED = 105.0

# an 8x8 font, one hex byte per row, high bit on the left. we ship our own so
# this module stays standalone, and eight pixels is the classic demo size.
_FONT = {
    ' ': '0000000000000000', 'A': '3c66c3c3ffc3c300', 'B': 'fcc6c6fcc6c6fc00',
    'C': '3e63c0c0c0633e00', 'D': 'f8ccc6c6c6ccf800', 'E': 'ffc0c0fcc0c0ff00',
    'F': 'ffc0c0fcc0c0c000', 'G': '3e63c0cfc3633d00', 'H': 'c3c3c3ffc3c3c300',
    'I': '7e18181818187e00', 'J': '0f060606c6c67c00', 'K': 'c6ccd8f0d8ccc600',
    'L': 'c0c0c0c0c0c0ff00', 'M': 'c3e7ffdbc3c3c300', 'N': 'c3e3f3dbcfc7c300',
    'O': '3c66c3c3c3663c00', 'P': 'fcc6c6fcc0c0c000', 'Q': '3c66c3c3db6e3b00',
    'R': 'fcc6c6fcd8ccc600', 'S': '3e63301806c67c00', 'T': 'ff18181818181800',
    'U': 'c3c3c3c3c3663c00', 'V': 'c3c3c366663c1800', 'W': 'c3c3dbffffe7c300',
    'X': 'c3663c183c66c300', 'Y': 'c3663c1818181800', 'Z': 'ff060c183060ff00',
    '0': '3c66c7dbe3663c00', '1': '1878181818187e00', '2': '7cc6061c3060ff00',
    '3': '7cc6063c06c67c00', '4': '0e1e3666ff060f00', '5': 'ffc0fc0606c67c00',
    '6': '3c66c0fcc6c67c00', '7': 'ffc60c1830303000', '8': '7cc6c67cc6c67c00',
    '9': '7cc6c63f06663c00', '.': '0000000000181800', ',': '0000000018183000',
    '!': '1818181818001800', '?': '7cc60c1818001800', '-': '000000ff00000000',
    '=': '0000ff00ff000000', '+': '001818ff18180000', '*': '00db7e3c7edb0000',
    ':': '0018180018180000', "'": '1818300000000000', '/': '060c183060c00000',
    '(': '0c18303030180c00', ')': '30180c0c0c183000', '<': '0e38e0380e000000',
    '>': 'e0380e38e0000000', '[': '3c30303030303c00', ']': '3c0c0c0c0c0c3c00',
    '%': 'c3c60c183063c300', '"': '3636360000000000', '@': '3c66dedec0633e00',
    '#': '6666ff66ff666600', '^': '183c66c300000000', '_': '00000000000000ff',
    '&': '386c6c3876cc7b00', ';': '0018180018183000',
    # a heart, which "<3" gets folded into on its way to being drawn
    '\x01': '66ffffff7e3c1800',
}

# What you type to get the heart. Two characters in, one glyph out, so anything
# measuring a line has to count glyphs rather than letters.
HEART = "<3"
_HEART_GLYPH = "\x01"

# Palette stops are (index, r, g, b), and the last one lands back on the first
# so a ramp can cycle forever without a seam. Every scheme fills the same five
# roles, which is how one keypress can repaint the entire production:
#   main  the plasma, the copper bars, the twister, the logo fill
#   skin  the potato, the tunnel, the hashbrown tile
#   hot   fire and the griddle, always running black to white
#   cool  the dark backgrounds behind the geometry
#   wild  the scroller, a full hue wheel unless a scheme says otherwise
_HUE = "hue"  # generated from three phase-shifted sines rather than stops

SCHEMES = (
    ("electric", {
        "main": ((0, 14, 0, 30), (36, 176, 0, 200), (72, 40, 60, 255), (108, 0, 226, 226),
                 (144, 70, 255, 90), (180, 255, 216, 0), (216, 255, 40, 130), (255, 14, 0, 30)),
        "skin": ((0, 30, 6, 44), (70, 210, 40, 140), (130, 255, 140, 40), (190, 255, 246, 190),
                 (255, 30, 6, 44)),
        "hot": ((0, 0, 0, 0), (60, 130, 0, 170), (120, 255, 0, 140), (180, 255, 130, 60),
                (228, 255, 250, 200), (255, 255, 255, 255)),
        "cool": ((0, 4, 0, 20), (90, 44, 10, 124), (170, 120, 60, 255), (230, 232, 220, 255),
                 (255, 4, 0, 20)),
        "wild": _HUE,
    }),
    ("hashbrown", {
        "main": ((0, 10, 4, 14), (40, 86, 20, 40), (80, 196, 64, 24), (120, 246, 146, 32),
                 (160, 255, 208, 96), (200, 255, 248, 214), (228, 150, 64, 20), (255, 10, 4, 14)),
        "skin": ((0, 28, 12, 4), (70, 150, 72, 16), (130, 236, 158, 44), (190, 255, 220, 130),
                 (255, 28, 12, 4)),
        "hot": ((0, 0, 0, 0), (60, 110, 12, 0), (120, 226, 72, 0), (180, 255, 168, 24),
                (228, 255, 238, 170), (255, 255, 255, 255)),
        "cool": ((0, 2, 2, 10), (90, 30, 46, 96), (170, 96, 150, 214), (230, 210, 234, 255),
                 (255, 2, 2, 10)),
        "wild": _HUE,
    }),
    ("acid", {
        "main": ((0, 0, 16, 4), (50, 0, 200, 60), (100, 190, 255, 0), (150, 0, 255, 200),
                 (200, 220, 0, 255), (255, 0, 16, 4)),
        "skin": ((0, 4, 24, 0), (70, 60, 200, 0), (130, 190, 255, 40), (190, 240, 255, 200),
                 (255, 4, 24, 0)),
        "hot": ((0, 0, 0, 0), (60, 0, 110, 40), (120, 40, 230, 40), (180, 190, 255, 60),
                (228, 240, 255, 200), (255, 255, 255, 255)),
        "cool": ((0, 0, 8, 8), (90, 0, 70, 80), (170, 40, 190, 190), (230, 210, 255, 250),
                 (255, 0, 8, 8)),
        "wild": _HUE,
    }),
    ("vapor", {
        "main": ((0, 20, 0, 40), (48, 120, 0, 180), (96, 255, 60, 190), (144, 80, 200, 255),
                 (192, 255, 220, 250), (255, 20, 0, 40)),
        "skin": ((0, 26, 6, 50), (70, 170, 40, 160), (130, 255, 120, 200), (190, 210, 245, 255),
                 (255, 26, 6, 50)),
        "hot": ((0, 0, 0, 0), (60, 90, 0, 130), (120, 236, 40, 190), (180, 120, 200, 255),
                (228, 240, 240, 255), (255, 255, 255, 255)),
        "cool": ((0, 8, 0, 26), (90, 60, 10, 110), (170, 150, 90, 220), (230, 240, 210, 255),
                 (255, 8, 0, 26)),
        "wild": _HUE,
    }),
    ("ice", {
        "main": ((0, 0, 6, 24), (50, 0, 70, 160), (100, 0, 180, 230), (150, 120, 240, 255),
                 (200, 245, 252, 255), (255, 0, 6, 24)),
        "skin": ((0, 2, 14, 30), (70, 20, 100, 170), (130, 110, 200, 240), (190, 235, 250, 255),
                 (255, 2, 14, 30)),
        "hot": ((0, 0, 0, 0), (60, 0, 60, 120), (120, 0, 160, 220), (180, 150, 230, 255),
                (228, 235, 250, 255), (255, 255, 255, 255)),
        "cool": ((0, 0, 2, 14), (90, 10, 40, 90), (170, 70, 140, 200), (230, 220, 244, 255),
                 (255, 0, 2, 14)),
        "wild": _HUE,
    }),
    ("phosphor", {
        # one gun, one color, the way the machine in the loader would have had it.
        "main": ((0, 0, 4, 0), (60, 20, 90, 10), (120, 90, 220, 40), (180, 200, 255, 140),
                 (220, 240, 255, 210), (255, 0, 4, 0)),
        "skin": ((0, 0, 8, 0), (70, 30, 120, 20), (130, 120, 230, 60), (190, 220, 255, 180),
                 (255, 0, 8, 0)),
        "hot": ((0, 0, 0, 0), (60, 10, 70, 10), (120, 60, 190, 40), (180, 170, 250, 120),
                (228, 235, 255, 200), (255, 255, 255, 255)),
        "cool": ((0, 0, 4, 0), (90, 10, 50, 10), (170, 60, 150, 50), (230, 200, 250, 190),
                 (255, 0, 4, 0)),
        "wild": ((0, 0, 30, 0), (64, 120, 255, 60), (128, 230, 255, 170), (192, 120, 255, 60),
                 (255, 0, 30, 0)),
    }),
)
SCHEME_NAMES = tuple(name for name, _ramps in SCHEMES)

# the scheme the frame being drawn right now is using. frame() sets it from its
# own argument before anything draws, so the same arguments still give the same
# picture; it is a global only so every effect does not have to pass it along.
_SCHEME = [0]

_NUMPY = None
_GRIDS: dict[tuple[int, int], dict] = {}
_RAMPS: dict[str, object] = {}
_BITS: dict[str, object] = {}
_STYLES: dict[int, str] = {}
_ESCAPES: dict[tuple[bool, int], str] = {}
_CACHE: dict[str, object] = {}


def _np():
    """Return numpy, importing it the first time anyone draws.

    The app imports this module just to register a key binding, so we keep the
    heavy import out of module scope and inside the drawing path.
    """
    global _NUMPY
    if _NUMPY is None:
        import numpy

        _NUMPY = numpy
    return _NUMPY


# ---------------------------------------------------------------- palettes --


def _ramp(role: str):
    """Return the 256x3 float palette for a role, built once per scheme."""
    key = (_SCHEME[0], role)
    ramp = _RAMPS.get(key)
    if ramp is None:
        np = _np()
        stops = SCHEMES[_SCHEME[0]][1][role]
        index = np.arange(PALETTE_SIZE, dtype=np.float32)
        if stops is _HUE:
            # a full hue wheel, and three phase-shifted sines are the cheapest
            # rainbow there is.
            a = index * (2.0 * np.pi / PALETTE_SIZE)
            ramp = np.stack([0.5 + 0.5 * np.sin(a + p) for p in (0.0, 2.094, 4.189)], axis=1)
        else:
            xs = [s[0] for s in stops]
            ramp = np.stack(
                [np.interp(index, xs, [s[c + 1] for s in stops]) / 255.0 for c in range(3)],
                axis=1,
            )
        _RAMPS[key] = ramp = ramp.astype(np.float32)
    return ramp


def palette_color(role: str, index: int, scheme: int = 0) -> tuple[int, int, int]:
    """Return one palette entry as 0..255 ints, wrapping the index.

    The palettes cycle, which is what makes a plasma shimmer, so the index is
    taken modulo the palette size rather than clamped.
    """
    _SCHEME[0] = int(scheme) % len(SCHEMES)
    rgb = _ramp(role)[int(index) % PALETTE_SIZE]
    return tuple(int(round(float(c) * 255.0)) for c in rgb)


# ------------------------------------------------------------------- grids --


def _grid(w: int, h2: int) -> dict:
    """Return the precomputed coordinate grids for a canvas of w by h2 pixels.

    Every effect works from these, so we build the meshgrids once per size and
    keep only per-frame arithmetic in the hot path.
    """
    key = (w, h2)
    grid = _GRIDS.get(key)
    if grid is None:
        np = _np()
        # a resize storm would otherwise pin every intermediate size in memory.
        if len(_GRIDS) > 6:
            _GRIDS.clear()
        x, y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h2, dtype=np.float32)
        )
        # half-block pixels are about square, so we only have to correct for the
        # canvas being wider than it is tall.
        aspect = max(w, 1) / max(h2, 1)
        nx = (x / max(w - 1, 1) * 2.0 - 1.0) * aspect
        ny = y / max(h2 - 1, 1) * 2.0 - 1.0
        rad = np.sqrt(nx * nx + ny * ny)
        grid = {
            "w": w, "h2": h2, "x": x, "y": y, "nx": nx, "ny": ny, "rad": rad,
            "ang": np.arctan2(ny, nx),
            "cols": np.arange(w, dtype=np.float32),
            "rows": np.arange(h2, dtype=np.float32),
            # A vignette sells the curved glass of a tube monitor for one
            # multiply. We measure it against the distance to the corner rather
            # than in raw units, or a wide terminal puts its own left and right
            # edges past the end of the falloff and swallows the logo. The cube
            # keeps the middle of the screen flat and only rolls off near the
            # glass, which is a tube rather than a spotlight.
            "vign": np.clip(
                1.03 - 0.34 * (rad / max(float(np.sqrt(aspect * aspect + 1.0)), 1e-6)) ** 3,
                0.0, 1.0,
            ).astype(np.float32)[..., None],
        }
        _GRIDS[key] = grid
    return grid


def _canvas(g):
    """Return a fresh black canvas shaped (h2, w, 3)."""
    np = _np()
    return np.zeros((g["h2"], g["w"], 3), dtype=np.float32)


# -------------------------------------------------------------------- text --


def _glyphs(text: str) -> str:
    """Return the text as the glyphs it will actually be drawn with."""
    return text.upper().replace(HEART, _HEART_GLYPH)


def _text_bits(text: str, gap: int = GAP):
    """Return a boolean bitmap of shape (8, glyphs * (8 + gap)) for the text."""
    key = (text, gap)
    bits = _BITS.get(key)
    if bits is None:
        np = _np()
        blank = _FONT[" "]
        rows = [_FONT.get(c, blank) for c in _glyphs(text)] or [blank]
        # unpack each glyph's eight hex bytes into eight bits, most significant
        # bit first, which is the order the font was authored in.
        data = np.array(
            [[int(g[r * 2:r * 2 + 2], 16) for r in range(8)] for g in rows], dtype=np.uint8
        )
        bits = np.unpackbits(data, axis=1).reshape(len(rows), 8, 8).transpose(1, 0, 2)
        if gap:
            bits = np.concatenate([bits, np.zeros((8, len(rows), gap), np.uint8)], axis=2)
        bits = bits.reshape(8, -1).astype(bool)
        if len(_BITS) > 24:
            _BITS.clear()
        _BITS[key] = bits
    return bits


def _blit(canvas, bits, x0: int, y0: int, color, dy=None):
    """Paint the set pixels of a bitmap onto the canvas at (x0, y0).

    ``color`` is either one rgb triple or a callable taking the pixel columns
    and rows and returning one color per pixel, which is how text gets filled
    with a gradient. ``dy`` is an optional per-source-column vertical offset,
    which is all a sine scroller really is.
    """
    np = _np()
    h2, w = canvas.shape[:2]
    ys, xs = np.nonzero(bits)
    px = xs + x0
    py = ys + y0
    if dy is not None:
        py = py + dy[xs]
    m = (px >= 0) & (px < w) & (py >= 0) & (py < h2)
    px, py = px[m], py[m]
    canvas[py, px] = color(px, py, xs[m], ys[m]) if callable(color) else color


def _dilate(bits):
    """Grow a bitmap by one pixel in each direction, which is an outline.

    One dilated blit underneath beats five offset copies on top of each other,
    and the scroller redraws its outline every frame.
    """
    grown = bits.copy()
    grown[1:, :] |= bits[:-1, :]
    grown[:-1, :] |= bits[1:, :]
    grown[:, 1:] |= bits[:, :-1]
    grown[:, :-1] |= bits[:, 1:]
    return grown


def _text(canvas, text: str, x0: int, y0: int, color, sx: int = 1, sy: int = 1, dy=None,
          gap: int = GAP, outline: bool = False):
    """Draw text at a pixel position, optionally scaled up per axis."""
    np = _np()
    bits = _text_bits(text, gap)
    if sx > 1 or sy > 1:
        bits = np.repeat(np.repeat(bits, sy, axis=0), sx, axis=1)
    if outline:
        _blit(canvas, _dilate(bits), x0, y0, INK, dy)
    _blit(canvas, bits, x0, y0, color, dy)


def _centered_at(w: int, text: str, sx: int = 1, gap: int = GAP) -> tuple[int, int]:
    """Return where a centered line starts, and the spacing it settled for."""
    count = len(_glyphs(text))
    if count * (8 + gap) * sx > w:
        gap = 0
    return (w - count * (8 + gap) * sx) // 2, gap


def _centered(canvas, text: str, y0: int, color, sx: int = 1, sy: int = 1, outline: bool = False,
              gap: int = GAP):
    """Draw text centered horizontally, which is where logos belong.

    The outline is a ring of near-black copies underneath. Without it a lit
    letter disappears the moment the plasma behind it goes bright. A line that
    is too wide to fit gives up its letter spacing before it gives up letters.
    """
    x0, gap = _centered_at(canvas.shape[1], text, sx, gap)
    _text(canvas, text, x0, y0, color, sx, sy, gap=gap, outline=outline)


def _fit(text: str, w: int, limit: int = 3) -> int:
    """Return the largest horizontal scale at which the text still fits."""
    return max(1, min(limit, w // max(len(_glyphs(text)) * (8 + GAP), 1)))


def _name_lines(w: int) -> tuple[str, ...]:
    """Stack the name when it will not fit across the screen in one line."""
    return ("HASHBROWNS",) if w >= 10 * (8 + GAP) else ("HASH", "BROWNS")


# ----------------------------------------------------------------- effects --


def _plasma(g, t: float, role: str = "main", drift: float = 40.0):
    """Three summed sine fields plus a radial one, mapped through a palette."""
    np = _np()
    x, y = g["x"], g["y"]
    cx, cy = g["w"] * 0.5, g["h2"] * 0.5
    v = (
        np.sin(x * 0.17 + t * 1.7)
        + np.sin(y * 0.21 - t * 1.1)
        + np.sin((x + y) * 0.11 + t * 0.9)
        + np.sin(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * 0.24 - t * 2.3)
    )
    idx = ((v * 0.125 + 0.5) * 255.0 + t * drift).astype(np.int32) % PALETTE_SIZE
    return _ramp(role)[idx]


def _copper(canvas, g, t: float, count: int = 5, thickness: float = 4.0):
    """Add copper bars: fat sine-driven gradients sliding up and down."""
    np = _np()
    rows = g["rows"][:, None]
    ramp = _ramp("main")
    for i in range(count):
        center = (0.5 + 0.40 * np.sin(t * (0.7 + 0.11 * i) + i * 1.9)) * g["h2"]
        d = np.abs(rows - center) / thickness
        mask = np.clip(1.0 - d * d, 0.0, 1.0).astype(np.float32)
        color = ramp[int(t * 28.0 + i * 43.0) % PALETTE_SIZE]
        canvas += mask[..., None] * color * 0.55


def _plot(canvas, px, py, color, lighten: bool = False, cover=None):
    """Scatter points onto the canvas, dropping whatever falls off the edge.

    Stars lighten what is already there, because a dim star written straight
    over a bright plasma would read as a hole rather than a star.
    """
    np = _np()
    h2, w = canvas.shape[:2]
    xi = px.astype(np.int32)
    yi = py.astype(np.int32)
    m = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h2)
    yi, xi = yi[m], xi[m]
    canvas[yi, xi] = np.maximum(canvas[yi, xi], color[m]) if lighten else color[m]
    if cover is not None:
        cover[yi, xi] = True


def _star_field(count: int = 700):
    """Return the fixed star positions, seeded once so frames stay pure."""
    stars = _CACHE.get("stars")
    if stars is None:
        np = _np()
        rng = np.random.default_rng(0x5EEDCAFE)
        stars = (
            rng.uniform(-1.0, 1.0, count).astype(np.float32),
            rng.uniform(-1.0, 1.0, count).astype(np.float32),
            rng.uniform(0.05, 1.0, count).astype(np.float32),
            rng.integers(1, 4, count).astype(np.float32),  # three depth layers
        )
        _CACHE["stars"] = stars
    return stars


def _stars(canvas, g, t: float, speed: float = 0.13):
    """Fly through three layers of stars, nearest and brightest last."""
    np = _np()
    sx, sy, sz, layer = _star_field()
    z = (sz - t * speed * layer) % 1.0 + 0.04
    order = np.argsort(-z)  # paint far to near so the close ones win the cell
    z = z[order]
    cx, cy = g["w"] * 0.5, g["h2"] * 0.5
    px = cx + sx[order] / z * cx
    py = cy + sy[order] / z * cy
    bright = np.clip(1.15 - z, 0.0, 1.0) ** 1.3
    tint = _ramp("skin")[(160 + (layer[order] * 26).astype(np.int32)) % PALETTE_SIZE]
    _plot(canvas, px, py, tint * bright[:, None], lighten=True)


def _potato_shape(th, ph):
    """Return points on the potato surface.

    A potato is an oval that grew: longer than it is thick, one end fatter than
    the other, a slight bend along its length. Building it in that order is what
    makes a vegetable. Piling noise onto a sphere, which is the obvious thing to
    do, only ever makes a lump.

    Eyes and skin mottling used to be carved in here as actual dents in the
    radius. That looked right face-on, but a dent is real geometry, and real
    geometry viewed edge-on foreshortens: a round dimple caught side-on by the
    turntable flattens into a thin dark crease running across the potato. Both
    are painted on as flat shading instead now, by ``_potato_eyes`` and
    ``_potato_spots`` in ``_potato_solid`` below, which never distorts however
    the potato is turned.
    """
    np = _np()
    ax, ay, az = POTATO_AXES
    x = np.cos(th) * np.cos(ph)
    y = np.sin(th)
    z = np.cos(th) * np.sin(ph)
    # Broad swells, at different frequencies and phases so nothing lines up
    # into a pumpkin. Anything that varies with longitude has to come round to
    # where it started after one turn, so those frequencies are whole numbers:
    # 1.7 of a turn leaves a step at the seam, and the step shows up as a crease
    # running down the potato. And they fade out toward the poles, or the two
    # points where every longitude meets would each pucker into a ring.
    swell = (
        1.0
        + 0.072 * np.sin(3.0 * ph + 2.4) * np.cos(th)
        + 0.055 * np.cos(2.3 * th + 2.1)
        + 0.046 * np.sin(1.0 * ph + 4.1) * np.cos(0.7 * th) * np.cos(th)
        + 0.028 * np.sin(2.0 * ph + 0.9) * np.cos(th) * np.cos(1.3 * th)
    )
    radius = swell
    x, y, z = x * radius, y * radius, z * radius
    # How thick it is along its length. A smooth taper on its own gives you an
    # egg; what makes a potato is that the outline wanders as you run down it,
    # with a shoulder here and a waist there. It only depends on how far along
    # we are, so it leaves the poles alone.
    girth = (1.0 - 0.06 * x
             + 0.095 * np.sin(2.3 * x + 1.1)
             - 0.042 * np.cos(2.6 * x + 1.7)
             # and the ends stay fat rather than tapering to a point, which is
             # the difference between a potato and an egg.
             ) * (1.0 + 0.30 * x * x)
    y, z = y * girth, z * girth
    # and it does not lie flat: a gentle bend in both of the short axes
    y = y + 0.085 * (x * x - 0.45)
    z = z + 0.055 * (x * x * x - 0.3 * x)
    return np.stack([x * ax, y * ay, z * az]).astype(np.float32)


def _potato_mesh():
    """Return the wireframe potato as a 3xN array of points.

    We sample latitude rings and meridians densely instead of rasterizing
    lines, because scattering points is one numpy call and a wire mesh is just
    points that happen to sit on the wires.
    """
    mesh = _CACHE.get("mesh")
    if mesh is None:
        np = _np()
        th_r = np.linspace(-1.35, 1.35, 9)
        ph_r = np.linspace(0.0, 2.0 * np.pi, 120)
        th_m = np.linspace(-np.pi / 2, np.pi / 2, 64)
        ph_m = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)
        th = np.concatenate([np.repeat(th_r, ph_r.size), np.tile(th_m, ph_m.size)])
        ph = np.concatenate([np.tile(ph_r, th_r.size), np.repeat(ph_m, th_m.size)])
        _CACHE["mesh"] = mesh = _potato_shape(th, ph)
    return mesh


def _potato_blotches(pts, count: int, seed: int, width, depth):
    """Return how strongly ``count`` round marks darken each point.

    Spots and eyes are the same trick at two different sizes: marks spread
    evenly over the sphere by the golden angle, then each one knocked off its
    mark so the result does not read as a pattern. This only ever paints a
    shading value onto a point that is already sitting on the potato's real
    surface, so a mark stays round from every angle instead of distorting the
    geometry the way a carved-in dent does when the potato turns it edge-on.
    """
    np = _np()
    rng = np.random.default_rng(seed)
    facing = pts / np.sqrt((pts * pts).sum(axis=0))
    # Marks thrown down at random leave whole sides of a potato bare: with two
    # dozen of them the emptiest patch had its nearest mark sixty five degrees
    # away. So we space them evenly first, by the golden angle, and then knock
    # each one off its mark so the result does not read as a pattern.
    index = np.arange(count) + 0.5
    up = 1.0 - 2.0 * index / count
    ring = np.sqrt(np.maximum(1.0 - up * up, 0.0))
    around = np.pi * (1.0 + 5.0**0.5) * index
    marks = np.stack([ring * np.cos(around), up, ring * np.sin(around)])
    marks = marks + rng.normal(0.0, 0.22, marks.shape)
    marks /= np.sqrt((marks * marks).sum(axis=0))
    # small and far apart: marks that overlap stop reading as individual marks
    # and just come out as the whole potato being a bit darker.
    widths = rng.uniform(*width, count).astype(np.float32)
    depths = rng.uniform(*depth, count).astype(np.float32)
    blotch = np.zeros(pts.shape[1], np.float32)
    for i in range(count):
        away = 1.0 - (facing * marks[:, i : i + 1]).sum(axis=0)
        blotch = np.maximum(blotch, depths[i] * np.exp(-((away / widths[i]) ** 2)))
    return blotch


def _potato_spots(pts, count: int = 34):
    """Return how blotchy the skin is: many small, shallow patches of tan."""
    return _potato_blotches(pts, count, 0x50075, (0.004, 0.016), (0.35, 0.7))


def _potato_eyes(pts, count: int = 10):
    """Return how dark the eyes are: fewer, bigger, and deeper than a spot."""
    return _potato_blotches(pts, count, 0xE7E5, (0.012, 0.022), (0.7, 0.95))


def _potato_solid(detail: int = 1):
    """Return a dense sample of the potato surface, with normals and eye depth.

    Dense enough that the front half covers the screen once we throw the back
    half away, which is cheaper than any depth buffer. ``detail`` says how
    finely to sample: a wide terminal draws a bigger potato out of the same
    points, so it gets more of them rather than letting the stretch open cracks.
    """
    key = ("solid", detail)
    solid = _CACHE.get(key)
    if solid is None:
        np = _np()
        rings = 110 + 70 * detail
        around = 2 * rings
        # Latitudes are spaced by angle, not by area. Spacing them by area uses
        # the points more evenly, but it leaves the first ring a long way down
        # from the pole with nothing inside it, and that unsampled cap shows up
        # as a round hole in the potato every time a pole turns toward us.
        # Spacing by angle wastes some points around the poles and never opens
        # one. We sample the two poles as well and throw them away below, so
        # that the differences which give us the normals are central everywhere:
        # a one sided difference on the last ring reaches for a neighbor that is
        # not there and leaves a crease across the potato.
        th = np.linspace(-np.pi / 2, np.pi / 2, rings + 2, dtype=np.float32)
        ph = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False, dtype=np.float32)
        tt, pp = np.meshgrid(th, ph, indexing="ij")
        pts = _potato_shape(tt.ravel(), pp.ravel())
        grid = pts.reshape(3, rings + 2, around)
        mottle = _potato_spots(pts)
        eyes = _potato_eyes(pts)
        # Real normals, taken from the surface itself by crossing its two
        # tangents. The normalized position would do for a ball, and a potato
        # that is half again as long as it is thick is not a ball: using it
        # lights the ends as if they bulged straight at you.
        along_rings = grid[:, 2:, :] - grid[:, :-2, :]
        along_around = np.roll(grid, -1, axis=2) - np.roll(grid, 1, axis=2)
        normals = np.cross(along_around[:, 1:-1, :], along_rings, axis=0)
        normals /= np.maximum(np.sqrt((normals * normals).sum(axis=0)), 1e-9)
        grid = grid[:, 1:-1, :]
        # and face them outward, whichever way the cross product happened to go
        outward = np.where((normals * grid).sum(axis=0) < 0.0, -1.0, 1.0)
        normals = (normals * outward).reshape(3, -1).astype(np.float32)
        pts = grid.reshape(3, -1).copy()
        trim = slice(around, -around)  # the two pole rings we just threw away
        eyes = eyes[trim].copy()
        mottle = mottle[trim].copy()
        _CACHE[key] = solid = (pts, normals, eyes, mottle)
    return solid


def _close_gaps(body, cover, rounds: int = 2):
    """Fill any pixel that has painted neighbors on both sides.

    However finely we sample the surface, projecting it stretches some of the
    grid further than a pixel and opens hairline cracks. Averaging across them
    is far cheaper than sampling finely enough that they never happen, and each
    round reaches one pixel further in from either side of a crack.
    """
    np = _np()
    for _ in range(rounds):
        for axis in (1, 0):
            gap = ~cover & np.roll(cover, 1, axis) & np.roll(cover, -1, axis)
            # the edges have no neighbor on one side, whatever roll says
            gap[:, 0] = gap[:, -1] = gap[0, :] = gap[-1, :] = False
            rows, cols = np.nonzero(gap)
            if rows.size == 0:
                continue
            if axis == 1:
                body[rows, cols] = 0.5 * (body[rows, cols - 1] + body[rows, cols + 1])
            else:
                body[rows, cols] = 0.5 * (body[rows - 1, cols] + body[rows + 1, cols])
            cover[rows, cols] = True


def _rotation(t: float, rates: tuple = (0.61, 0.43, 0.27)):
    """Return a rotation matrix turning about all three axes at the given rates."""
    np = _np()
    out = np.eye(3, dtype=np.float32)
    for axis, a in enumerate((t * rates[0], t * rates[1], t * rates[2])):
        c, s = float(np.cos(a)), float(np.sin(a))
        m = np.eye(3, dtype=np.float32)
        i, j = (axis + 1) % 3, (axis + 2) % 3
        m[i, i] = m[j, j] = c
        m[i, j], m[j, i] = -s, s
        out = out @ m
    return out


def _potato(canvas, g, t: float, scale: float = 0.42, dist: float = 3.4):
    """Tumble the wireframe potato and project it onto the canvas."""
    np = _np()
    pts = _rotation(t) @ _potato_mesh()
    z = pts[2] + dist
    persp = dist / np.maximum(z, 0.2)
    size = min(g["w"], g["h2"]) * scale
    order = np.argsort(-z)  # painter's algorithm stands in for hidden surfaces
    px = g["w"] * 0.5 + pts[0][order] * persp[order] * size
    py = g["h2"] * 0.5 + pts[1][order] * persp[order] * size
    # near points glow at the crisp end of the palette, far ones sink into skin.
    shade = np.clip((dist + 1.2 - z[order]) / 2.4, 0.0, 1.0)
    color = _ramp("skin")[(40 + shade * 165).astype(np.int32) % PALETTE_SIZE]
    _plot(canvas, px, py, color * shade[:, None])


def _tunnel(g, t: float):
    """The classic texture tunnel: angle across, one over radius into depth."""
    np = _np()
    rad = np.maximum(g["rad"], 0.02)
    u = g["ang"] * (7.0 / np.pi) + t * 0.55 + 0.5 * np.sin(t * 0.45)
    v = 1.3 / rad + t * 1.9
    checker = ((np.floor(u) + np.floor(v)) % 2.0)
    speckle = 0.5 + 0.5 * np.sin(u * 6.1) * np.sin(v * 5.3)
    val = checker * 0.55 + speckle * 0.45
    idx = (val * 255.0 + t * 30.0).astype(np.int32) % PALETTE_SIZE
    # the far end of a tunnel is dark, which is the whole illusion.
    canvas = _ramp("skin")[idx] * np.clip(rad * 1.9, 0.0, 1.0)[..., None]
    # The pattern at full strength is a wall of bright color, and the scroller
    # has to cross it. Taking it down and part of the way toward gray keeps
    # every stripe where it was and gives the message something to read against.
    canvas *= 0.58
    return canvas * 0.78 + canvas.mean(axis=2, keepdims=True) * 0.22


def _tile():
    """Return a 64x64 shredded-hashbrown texture, generated once."""
    tile = _CACHE.get(("tile", _SCHEME[0]))
    if tile is None:
        np = _np()
        rng = np.random.default_rng(0xBADA55)
        # chunky low-res noise stretched out gives us shreds rather than static.
        noise = np.repeat(np.repeat(rng.random((8, 8)), 8, 0), 8, 1).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(64, dtype=np.float32), np.arange(64, dtype=np.float32),
                             indexing="ij")
        v = 0.5 + 0.5 * np.sin(xx * 0.55 + noise * 6.0) * np.sin(yy * 0.22 + noise * 3.0)
        tile = _ramp("skin")[(v * 255.0).astype(np.int32) % PALETTE_SIZE]
        _CACHE[("tile", _SCHEME[0])] = tile
    return tile


def _roto(g, t: float):
    """A rotozoomer over the hashbrown tile, rotating and breathing."""
    np = _np()
    # a step of much more than one texel per pixel turns the tile into static,
    # so the zoom breathes around one rather than swinging past two.
    zoom = 1.05 + 0.55 * np.sin(t * 0.6)
    a = t * 0.7
    c, s = np.cos(a), np.sin(a)
    x = g["x"] - g["w"] * 0.5
    y = g["y"] - g["h2"] * 0.5
    u = ((x * c - y * s) * zoom + t * 9.0).astype(np.int32) % 64
    v = ((x * s + y * c) * zoom + t * 4.0).astype(np.int32) % 64
    # fancy indexing hands back a fresh array, which the scene then draws on.
    return _tile()[v, u]


def _twister(canvas, g, t: float):
    """A twisting square column, four faces shaded by which way they face."""
    np = _np()
    w, h2 = g["w"], g["h2"]
    width = min(w * 0.20, h2 * 0.40)
    a = t * 1.5 + g["rows"] * 0.11
    ramp = _ramp("main")
    corners = [(w * 0.5 + width * np.cos(a + k * np.pi / 2), np.sin(a + k * np.pi / 2))
               for k in range(4)]
    faces = []
    for k in range(4):
        (x0, z0), (x1, z1) = corners[k], corners[(k + 1) % 4]
        faces.append((np.minimum(x0, x1), np.maximum(x0, x1), (z0 + z1) * 0.5,
                      ramp[(int(t * 20.0) + k * 62) % PALETTE_SIZE]))
    for r in range(h2):
        # the nearest face goes down last, or the column turns inside out.
        for left, right, depth, color in sorted(faces, key=lambda f: f[2][r]):
            if depth[r] <= 0.0:
                continue
            x0, x1 = max(int(left[r]), 0), min(int(right[r]) + 1, w)
            if x1 > x0:
                canvas[r, x0:x1] = color * (0.22 + 0.78 * depth[r])


def _sizzle(canvas, g, t: float, height: float = 0.42):
    """Add the griddle: hot fat rising off the bottom of the screen."""
    np = _np()
    h2 = g["h2"]
    up = np.clip((g["y"] - h2 * (1.0 - height)) / (h2 * height), 0.0, 1.0)
    n = (
        np.sin(g["x"] * 0.21 + t * 1.1)
        + np.sin(g["x"] * 0.09 - t * 1.7 + g["y"] * 0.2)
        + np.sin((g["x"] * 0.13 + g["y"] * 0.31) - t * 2.6)
    )
    v = np.clip(up * 1.45 + n * 0.20 - 0.30, 0.0, 1.0)
    canvas += _ramp("hot")[(v * 255.0).astype(np.int32)] * v[..., None]


def _scroller_top(g) -> int:
    """Return the topmost pixel row the scroller can reach at the bottom.

    Anything else drawing near the bottom asks first, because the scroller is
    the one thing that is always there and it must never be written over.
    """
    sy = 2 if g["h2"] >= 56 else 1
    amp = max(2.0, g["h2"] * 0.06)
    return int(g["h2"] - 8 * sy - 1 - 2 * amp)


def _scroller(canvas, g, t: float, text=None, start=None, seconds=None):
    """Run this pass of the greetings along the bottom on a sine wave.

    Each pass gets its own message and crosses once. Editing a message needs no
    other change: we work out the speed that gets it from off the right hand
    edge to off the left in the time its run has, so a longer message simply
    goes by faster.
    """
    np = _np()
    run, began, room = _scroll_run(t)
    text = SCROLL_TEXTS[run % len(SCROLL_TEXTS)] if text is None else text
    start = began if start is None else start
    seconds = room if seconds is None else seconds
    w, h2 = g["w"], g["h2"]
    sy = 2 if h2 >= 56 else 1
    bits = _text_bits(text)
    span = bits.shape[1]
    # it starts a screen's width out to the right, so the first thing anybody
    # sees is the front of the message arriving rather than the middle of it.
    off = int(max(t - start, 0.0) * (span + w) / max(seconds, 0.1)) - w
    index = np.arange(w) + off
    src = np.clip(index, 0, span - 1)
    amp = max(2.0, h2 * 0.06)
    dy = (amp * np.sin(np.arange(w) * 0.09 + t * 2.2)).astype(np.int32)
    base = _scroller_top(g) + int(amp)
    ramp = _ramp("wild")

    def color(px, py, xs, ys):
        # we keep the hue but lift the value, because a mid-bright rainbow over
        # a mid-bright plasma is exactly the case you cannot read.
        return 0.25 + 0.75 * ramp[(px * 2 + int(t * 50.0)) % PALETTE_SIZE]

    # we cut the visible slice out before scaling it up, because the whole
    # message is thousands of columns wide and only this many are on screen.
    visible = bits[:, src]
    on_screen = (index >= 0) & (index < span)
    if not on_screen.all():
        # in front of the message and behind it there is nothing, not a repeat
        visible = visible.copy()
        visible[:, ~on_screen] = False
    if sy > 1:
        visible = np.repeat(visible, sy, axis=0)
    # the same outline the logo gets. the scroller crosses every background the
    # demo has, so it cannot rely on any of them being dark.
    _blit(canvas, _dilate(visible), 0, base, INK, dy)
    _blit(canvas, visible, 0, base, color, dy)


def _logo(canvas, g, t: float, beat: float):
    """Stamp the big logo, filled with a gradient that slides through gold."""
    np = _np()
    w, h2 = g["w"], g["h2"]
    # the logo owns the top half and the scroller owns the bottom, so we only
    # split the name across two lines when there is height to spend on it.
    budget = int(h2 * 0.5)
    lines = ("HASH", "BROWNS") if budget >= 26 else _name_lines(w)
    sx = _fit(max(lines, key=len), w)
    sy = max(1, min(sx * 2, budget // (17 if len(lines) == 2 else 8)))
    ramp = _ramp("main")
    bob = int(np.sin(t * 1.7) * max(1.0, h2 * 0.03) - beat * 2.0)
    top = int(h2 * 0.08) + bob

    def fill(px, py, xs, ys):
        # we stay in the hot end of the palette so the letters never go muddy
        # against the plasma behind them.
        return ramp[110 + (py * 5 - int(t * 70.0)) % 95] * (0.85 + 0.15 * beat)

    for i, text in enumerate(lines):
        _centered(canvas, text, top + i * 9 * sy, fill, sx, sy, outline=True)


def _cracks(pts, glow: float):
    """How lit each surface point is as the potato splits open.

    Three sine ridges crossed over each other give a network of thin seams --
    ``1 - |sin|`` is near one only where the wave passes through zero, which is
    a line rather than a band. The seams are computed from each point's own
    place on the potato rather than from the screen, so they stay put on the
    skin while it turns.

    ``glow`` walks a threshold down through that field: at the start only the
    sharpest seams are above it and you get hairline cracks, and by the end
    everything is above it and the whole surface is light.
    """
    np = _np()
    x, y, z = pts
    seams = np.maximum.reduce([
        1.0 - np.abs(np.sin(11.3 * x + 6.1 * y - 4.7 * z)),
        1.0 - np.abs(np.sin(9.7 * z + 2.9 * y - 7.3 * x)),
        1.0 - np.abs(np.sin(6.7 * x + 8.9 * y - 10.1 * z)),
    ])
    # the threshold comes down a steep curve, so most of the seven seconds is
    # spent on hairlines and the skin only gives way near the end. a linear
    # walk floods the potato in the first second and there is nothing to watch.
    edge = 1.0 - glow ** 3.6
    lit = np.clip((seams - edge) / 0.02, 0.0, 1.0)
    # and the last stretch floods what is left, so it finishes as one lit
    # thing rather than a bundle of bright lines
    return np.maximum(lit, max(0.0, (glow - 0.88) / 0.12))


def _solid_potato(canvas, g, t: float, scale: float = 0.30, dist: float = 5.5,
                  center=None, light=(0.45, 0.62, 0.64), role: str = "skin",
                  ambient: float = 0.24, glow: float = 0.0, emit=None):
    """Light a dense potato and splat it, nearest points last.

    ``light`` is how much the key light weighs each of sideways, upright and
    toward us, so the prologue can turn it upside down and have the fire light
    the potato from underneath.

    ``glow`` cracks it open (see :func:`_cracks`); pass an ``emit`` array to
    get back where the light came out, which is what the prologue blooms.
    """
    np = _np()
    size = min(g["w"], g["h2"]) * scale
    # A bigger terminal draws a bigger potato out of the same points, so the
    # sample gets finer with it up to a point. Past that the cost runs away
    # faster than the coverage improves, and it is cheaper to widen the reach of
    # the crack filling below than to keep buying points.
    pts, normals, eyes, mottle = _potato_solid(sum(size >= step for step in (16, 30)))
    # mostly a spin about the upright axis, with only a slow wobble on the
    # others, so the potato keeps showing us its long side rather than rolling
    # end on and looking like an egg.
    rot = _rotation(t, (0.11, 0.85, 0.07))
    p = rot @ pts
    n = rot @ normals
    # negative z is toward us, so this is how much of each point we can see.
    facing = -n[2]
    # we keep a little past the horizon so the silhouette stays round rather
    # than showing the edge of the sample.
    keep = facing > -0.3
    home = pts[:, keep]          # where each point sits on the potato itself
    p, n, eyes, facing = p[:, keep], n[:, keep], eyes[keep], facing[keep]
    mottle = mottle[keep]
    z = p[2] + dist
    order = np.argsort(-z)  # far first, so the near surface lands on top
    persp = dist / np.maximum(z, 0.2)
    cx, cy = center if center is not None else (g["w"] * 0.5, g["h2"] * 0.5)
    px = cx + p[0][order] * persp[order] * size
    py = cy + p[1][order] * persp[order] * size
    # one key light over our left shoulder, a tight highlight, and a rim
    lam = np.clip(-n[0] * light[0] - n[1] * light[1] + facing * light[2], 0.0, 1.0)[order]
    shade = ambient + 0.68 * lam + 0.22 * lam**24 + 0.18 * (1.0 - facing[order]) ** 3
    # the eye marks go dark, which is what makes them read as eyes, and the
    # patches of rougher skin go brown, which is what makes the rest read as
    # a potato
    shade *= 1.0 - 0.62 * eyes[order]
    shade *= 1.0 - 0.45 * mottle[order]
    shade = np.clip(shade, 0.0, 1.0)
    # the palettes cycle, so their brightest entry is partway along and the far
    # end runs back down to dark. we stop short of that, or a lit face comes out
    # looking like a gray sticker stuck on the potato.
    color = _ramp(role)[(15 + shade * 172).astype(np.int32)]
    # we draw into a buffer of our own so that closing the cracks in the potato
    # cannot smear anything that was already on the screen behind it.
    tone = color * (0.35 + 0.65 * shade)[:, None]
    if glow > 0.0:
        # light replaces skin wherever a seam has opened. it is added rather
        # than blended so a crack over a dark face still reads as a crack.
        lit = _cracks(home[:, order], glow)[:, None]
        tone = tone * (1.0 - lit) + lit * (_ramp("hot")[236] * 0.45 + 0.55)
    body = _canvas(g)
    cover = np.zeros(body.shape[:2], bool)
    _plot(body, px, py, tone, cover=cover)
    _close_gaps(body, cover, rounds=max(2, int(size / 9)))
    canvas[cover] = body[cover]
    if emit is not None:
        # the bright half of what we just drew: the seams, for the bloom
        emit[cover] = np.maximum(emit[cover], body[cover].max(axis=1))


def _chaos_points(count: int = 14000):
    """Sierpinski by the chaos game: keep jumping halfway to a random corner."""
    pts = _CACHE.get("chaos")
    if pts is None:
        np = _np()
        rng = np.random.default_rng(0x51E49A)
        corners = np.array([[0.0, -0.98], [-0.95, 0.82], [0.95, 0.82]], np.float32)
        picks = corners[rng.integers(0, 3, count)]
        walk = np.zeros((count, 2), np.float32)
        here = np.zeros(2, np.float32)
        # each step depends on the one before it, so this is the one loop we
        # cannot vectorize. it runs once and then lives in the cache.
        for i in range(count):
            here = (here + picks[i]) * 0.5
            walk[i] = here
        _CACHE["chaos"] = pts = walk.T.copy()
    return pts


def _rule90(w: int, rows: int = 256):
    """Return rule 90 run from a single live cell, which draws the same triangle."""
    key = ("rule90", w)
    table = _CACHE.get(key)
    if table is None:
        np = _np()
        out = np.zeros((rows, w), bool)
        line = np.zeros(w, bool)
        line[w // 2] = True
        for r in range(rows):
            out[r] = line
            line = np.roll(line, 1) ^ np.roll(line, -1)
        _CACHE[key] = table = out
    return table


def _julia(g, t: float, iterations: int = 26):
    """Escape time for a Julia set whose constant walks round a circle."""
    np = _np()
    a = t * 0.23
    cr, ci = 0.7885 * np.cos(a), 0.7885 * np.sin(a)
    zoom = 1.45 + 0.35 * np.sin(t * 0.31)
    zx, zy = g["nx"] * zoom, g["ny"] * zoom
    count = np.zeros(zx.shape, np.float32)
    alive = np.ones(zx.shape, bool)
    # points that escaped keep getting iterated into infinity, which is noisy
    # but harmless: the comparison below just leaves them dead.
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(iterations):
            zx2, zy2 = zx * zx, zy * zy
            alive &= (zx2 + zy2) < 16.0
            count += alive
            zx, zy = zx2 - zy2 + cr, 2.0 * zx * zy + ci
    idx = (count * 11.0 + t * 40.0).astype(np.int32) % PALETTE_SIZE
    canvas = _ramp("main")[idx].copy()
    # the inside of the set is the one part that never moves, so keep it dark
    canvas[count >= iterations] *= 0.10
    return canvas


def _curve(canvas, g, t: float):
    """A hypotrochoid and a Lissajous figure, the two things every graphing
    calculator ever drew, with a pulse running round them."""
    np = _np()
    zoom = 0.42 * min(g["w"], g["h2"])
    cx, cy = g["w"] * 0.5, g["h2"] * 0.5
    a = t * 0.4
    rc, rs = np.cos(a), np.sin(a)
    ramp = _ramp("wild")

    def draw(x, y, bright):
        px = cx + (x * rc - y * rs) * zoom
        py = cy + (x * rs + y * rc) * zoom
        idx = (np.arange(x.size) * 2 + int(t * 50.0)) % PALETTE_SIZE
        _plot(canvas, px, py, ramp[idx] * bright[:, None], lighten=True)

    n = 4200
    th = np.linspace(0.0, 2.0 * np.pi * 7.0, n, dtype=np.float32)
    small = 0.29 + 0.12 * np.sin(t * 0.23)
    reach = 0.62 + 0.22 * np.sin(t * 0.17)
    k = (1.0 - small) / small
    pulse = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(np.arange(n) * (6.0 * np.pi / n) - t * 4.0))
    draw((1.0 - small) * np.cos(th) + reach * np.cos(k * th),
         (1.0 - small) * np.sin(th) - reach * np.sin(k * th), pulse)

    m = 1800
    u = np.linspace(0.0, 2.0 * np.pi, m, dtype=np.float32)
    draw(0.82 * np.sin(3.0 * u + t * 0.5), 0.82 * np.sin(4.0 * u),
         np.full(m, 0.34, np.float32))


def _caption(canvas, g, text: str, t: float, y: int = 2):
    """A small line of text, for the scenes that give the scroller a rest."""
    _centered(canvas, text, y, _ramp("wild")[int(t * 40.0) % PALETTE_SIZE] * 0.9, outline=True)


# --------------------------------------------------------------- the storm --

# The prologue runs on its own little timeline, in seconds from its first frame.
STORM_SECONDS = 28.2
# The camera holds on the sky for a long while before it tilts down. It used
# to be 5.6 seconds, which was not enough weather to be worth arriving in --
# the storm barely got going before the descent started. The flash schedule
# is written in terms of `_HIT_AT`, so stretching the prologue thickens the
# lightning to match rather than spreading the same dozen bolts thinner.
_PAN_AT, _PAN_FOR = 9.6, 3.6  # when the camera tilts down, and how long it takes
_HIT_AT = 18.3  # when the bolt lands on the potato


def _smoothstep(x: float) -> float:
    """Ease from 0 to 1, so the camera move starts and stops instead of jerking."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _flashes():
    """Return the storm's lightning as (when, where across, where up, how hard).

    Seeded once, so the same bolt lands in the same place every time the demo
    comes round. They crowd together toward the end, which is what makes the
    storm feel like it is building up to something.
    """
    schedule = _CACHE.get("flashes")
    if schedule is None:
        np = _np()
        rng = np.random.default_rng(0xB017)
        when, gap = [], 0.5
        while gap < _HIT_AT - 0.3:
            when.append((gap, float(rng.uniform(0.1, 0.9)), float(rng.uniform(-0.1, 0.7)),
                         float(rng.uniform(0.55, 1.25))))
            # the storm closes in, so the wait between flashes shortens
            gap += float(rng.uniform(0.5, 2.3)) * (1.0 - 0.55 * gap / _HIT_AT)
        _CACHE["flashes"] = schedule = tuple(when)
    return schedule


def _lightning(u: float, g):
    """Return how bright the sky is right now, and where the light is coming from."""
    np = _np()
    best, where = 0.0, (g["w"] * 0.5, 0.0)
    for start, across, up, power in _flashes():
        age = u - start
        if not 0.0 <= age < 1.1:
            continue
        # a real flash strobes two or three times before it dies away
        amount = power * np.exp(-age * 5.5) * (0.55 + 0.45 * np.sin(age * 47.0))
        amount = max(float(amount), 0.0)
        if amount > best:
            best, where = amount, (across * g["w"], up * g["h2"])
    return best, where


def _clouds(g, wy, u: float):
    """Return how thick the cloud is at every pixel, drifting as time passes."""
    np = _np()
    x = g["x"]
    # five octaves, each finer and weaker than the last, which is the cheapest
    # turbulence there is and enough to pass for weather.
    n = (
        1.00 * np.sin(x * 0.045 + u * 0.10) * np.cos(wy * 0.075 - u * 0.05)
        + 0.55 * np.sin(x * 0.105 - u * 0.16 + wy * 0.062 + 1.3)
        + 0.31 * np.sin(x * 0.215 + wy * 0.145 + u * 0.24 + 2.1)
        + 0.18 * np.cos(x * 0.430 - wy * 0.295 - u * 0.33)
        + 0.10 * np.sin(x * 0.810 + wy * 0.560 + u * 0.40)
    )
    # the weather gathers in a band and thins out above and below it
    band = np.clip(1.35 - np.abs(wy - g["h2"] * 0.30) / g["h2"], 0.0, 1.0)
    # a hard edge is what makes a cloud instead of a fog bank
    return np.clip((n * 0.42 + 0.46) * band * 1.5 - 0.34, 0.0, 1.0).astype(np.float32)


def _sky(canvas, g, wy, u: float, flash: float, where):
    """Paint the storm: near black sky, darker clouds, lit from behind."""
    np = _np()
    density = _clouds(g, wy, u)
    sky = np.float32((0.018, 0.023, 0.052))
    cloud = np.float32((0.004, 0.004, 0.009))
    canvas += sky * (1.0 - density)[..., None] + cloud * density[..., None]
    if flash <= 0.001:
        return
    # the light pools around wherever it went off, and the cloud it is behind
    # scatters it: that is what turns a flat black sky into shapes.
    fall = np.exp(-(((g["x"] - where[0]) / (g["w"] * 0.55)) ** 2
                    + ((wy - where[1]) / (g["h2"] * 0.75)) ** 2))
    # the light comes off the edges hardest, where the cloud is thin enough to
    # shine through but thick enough to catch anything at all.
    edge = density * (1.0 - density) * 4.0
    glow = flash * fall * (0.10 + 0.55 * density + 0.70 * edge) * 0.62
    canvas += (_ramp("cool")[232] * 0.55 + 0.45) * glow[..., None]


def _hellfire(canvas, g, wy, u: float):
    """Flames licking up off the floor of the world, and the glow off them."""
    np = _np()
    x, h2 = g["x"], g["h2"]
    floor = 2.0 * h2
    lick = ((0.55 + 0.45 * np.sin(x * 0.185 + u * 2.9))
            * (0.62 + 0.38 * np.sin(x * 0.083 - u * 1.7 + np.cos(x * 0.031 + u) * 2.0))
            * (0.70 + 0.30 * np.sin(x * 0.427 + u * 4.3)))
    tall = h2 * (0.18 + 0.62 * lick)
    up = (floor - wy) / tall
    churn = (np.sin(x * 0.37 + wy * 0.21 + u * 5.2)
             * np.sin(x * 0.150 - wy * 0.11 - u * 3.4))
    heat = np.clip((1.0 - up) * (0.82 + 0.42 * churn) + 0.08, 0.0, 1.0)
    heat = np.where((up > -0.35) & (up < 1.25), heat, 0.0).astype(np.float32)
    canvas += _ramp("hot")[(heat * 255.0).astype(np.int32)] * heat[..., None]
    # everything above the fire catches some of it, which is what puts the
    # potato in a room rather than in front of a picture of flames.
    canvas += _ramp("hot")[150] * np.clip(1.0 - (floor - wy) / h2, 0.0, 1.0)[
        ..., None] * 0.20


def _bolt_path():
    """Return a lightning bolt as a normalized polyline, built by subdivision.

    Halving the displacement at every step is the oldest trick there is for a
    jagged line, and it is the one that looks like lightning.
    """
    path = _CACHE.get("bolt")
    if path is None:
        np = _np()
        rng = np.random.default_rng(0xB0175)
        xs = np.array([0.0, 0.06])
        ys = np.array([0.0, 1.0])
        spread = 0.30
        for _ in range(9):
            mid_x = 0.5 * (xs[:-1] + xs[1:]) + rng.normal(0.0, spread, xs.size - 1)
            mid_y = 0.5 * (ys[:-1] + ys[1:])
            grown_x = np.empty(xs.size * 2 - 1)
            grown_y = np.empty(ys.size * 2 - 1)
            grown_x[0::2], grown_x[1::2] = xs, mid_x
            grown_y[0::2], grown_y[1::2] = ys, mid_y
            xs, ys = grown_x, grown_y
            spread *= 0.52
        _CACHE["bolt"] = path = (xs.astype(np.float32), ys.astype(np.float32))
    return path


def _strike(canvas, g, u: float, target):
    """Draw the bolt coming down onto the potato, for the moment it lasts."""
    np = _np()
    age = u - _HIT_AT
    if not -0.02 <= age < 0.60:
        return
    xs, ys = _bolt_path()
    top = -3.0
    px = target[0] + xs * g["w"] * 0.42
    py = top + ys * (target[1] - top)
    # it comes down fast, then flickers where it stands
    reached = np.clip(age / 0.10, 0.0, 1.0)
    live = ys <= reached
    flicker = float(np.clip(1.3 - age * 1.1, 0.0, 1.0)) * (0.68 + 0.32 * np.sin(age * 90.0))
    core = (_ramp("cool")[232] * 0.4 + 0.6) * max(flicker, 0.0)
    for dx, dy, level in ((1, 0, 0.35), (-1, 0, 0.35), (0, 1, 0.35), (0, -1, 0.35), (0, 0, 1.0)):
        _plot(canvas, px[live] + dx, py[live] + dy,
              np.tile(core * level, (int(live.sum()), 1)), lighten=True)


#: How long the potato takes to crack open and come up to full glow, and then
#: how long that glow takes to spread out and swallow the scene. The crack is
#: the whole point of the prologue, so it gets its time: rushed, it reads as a
#: flicker rather than as something breaking.
_CRACK_FOR = 7.0
_SWALLOW_FOR = 1.6


def _bloom(emit, rounds: int = 4):
    """Spread a mask of bright pixels outward into a soft halo.

    This is what puts light *outside* the potato: the seams are only a few
    pixels wide, and without somewhere for that light to go they read as paint
    on the skin rather than as something shining out of a gap.
    """
    np = _np()
    out = emit.astype(np.float32)
    for _ in range(rounds):
        out = 0.22 * (out
                      + np.roll(out, 1, 0) + np.roll(out, -1, 0)
                      + np.roll(out, 1, 1) + np.roll(out, -1, 1))
    return out


def _swallow(g, center, amount: float):
    """A disc of light growing out of the potato until it covers everything."""
    np = _np()
    if amount <= 0.0:
        return None
    dx = g["x"] - center[0]
    dy = g["y"] - center[1]
    rad = np.sqrt(dx * dx + dy * dy)
    # far enough to reach the far corner from wherever the potato is standing
    full = float(np.hypot(g["w"], g["h2"]))
    reach = full * _smoothstep(amount) * 1.15
    # a soft shoulder, so the edge of the light is a horizon rather than a hoop
    return np.clip(1.0 - (rad - reach) / max(full * 0.22, 1.0), 0.0, 1.0)


def _whiteout(u: float) -> float:
    """Return how much of the screen the flat flash has taken over.

    The glow does nearly all of it; this only closes the last gaps once the
    light has already reached the corners, and then stays up so the crossfade
    into the loader is a fade out of white rather than a cut.
    """
    age = u - _HIT_AT - _CRACK_FOR - _SWALLOW_FOR * 0.72
    if age < 0.0:
        return 0.0
    return _smoothstep(min(age / 0.9, 1.0)) * _glare_fade(u)


def _glare_fade(u: float) -> float:
    """How much of the light is left as the prologue hands over.

    It begins letting go before the scene is over, so what the crossfade
    blends into the loader is a softening glare rather than a sheet -- left at
    full, the machine spends its first second typing behind white. Both the
    spreading glow and the flat flash are held to this, or whichever one was
    not faded simply keeps the screen white on its own.
    """
    return 1.0 - 0.72 * _smoothstep(max(0.0, (u - (STORM_SECONDS - 1.3)) / 1.3))


#: How long the weather takes to come up out of the black it was scrolled
#: into. The loop hands over two black screens meeting, so there is no join
#: to hide -- and the first flash arrives out of the dark rather than sitting
#: there waiting to be noticed.
_WAKE_FOR = 1.1


def _sc_storm(g, u: float):
    """The prologue: a storm, a pan down into the fire, and then the strike."""
    np = _np()
    canvas = _canvas(g)
    h2 = g["h2"]
    cam = h2 * _smoothstep((u - _PAN_AT) / _PAN_FOR)
    wy = g["y"] + cam
    flash, where = _lightning(u, g)
    _sky(canvas, g, wy, u, flash, (where[0], where[1] - cam * 0.0))
    potato = (g["w"] * 0.5, 2.0 * h2 - h2 * 0.42 - cam)
    # the bolt lands and the potato starts coming apart: hairline seams first,
    # then light out of them, then the whole skin
    glow = float(np.clip((u - _HIT_AT) / _CRACK_FOR, 0.0, 1.0))
    if cam > h2 * 0.02:
        _hellfire(canvas, g, wy, u)
        emit = np.zeros((h2, g["w"]), dtype=np.float32) if glow > 0.0 else None
        # the camera leans in while it cracks. at the size it sits at for the
        # storm the potato is barely twenty pixels across, and a hairline on
        # twenty pixels is not a crack, it is a stray bright dot.
        near = 0.19 + 0.30 * _smoothstep(glow)
        # lit from underneath by what it is sitting over, and turning slowly
        _solid_potato(canvas, g, u * 0.35, scale=near, dist=6.5,
                      center=potato, light=(0.12, -0.94, 0.22), role="hot",
                      ambient=0.03, glow=glow, emit=emit)
        if emit is not None:
            # the beams: what escaped the seams, spread into the air around it
            halo = _bloom(emit, rounds=3)[..., None] * (0.35 + 1.5 * glow)
            canvas += (1.0 - np.clip(canvas, 0.0, 1.0)) * np.clip(halo, 0.0, 1.0)
    _strike(canvas, g, u, potato)
    # and once it is all light, that light leaves the potato behind
    spread = _swallow(g, potato, (u - _HIT_AT - _CRACK_FOR) / _SWALLOW_FOR)
    if spread is not None:
        canvas += ((1.0 - np.clip(canvas, 0.0, 1.0))
                   * (spread * _glare_fade(u))[..., None])
    white = _whiteout(u)
    if white > 0.0:
        canvas += (1.0 - np.clip(canvas, 0.0, 1.0)) * white
    if u < _WAKE_FOR:
        canvas *= _smoothstep(max(u, 0.0) / _WAKE_FOR)
    return canvas


# ------------------------------------------------------------------ scenes --


def _beat(t: float) -> float:
    """Return a 120 bpm pulse that decays between beats, for things to pump to."""
    return (1.0 - (t * 2.0) % 1.0) ** 4


def _sc_boot(g, t: float):
    """The loader: an old machine waking up and announcing itself."""
    canvas = _plasma(g, t, "cool", 16.0) * 0.16
    u = _since("boot", t)
    lines = BOOT_LINES
    budget = (u - 0.6) * 26.0  # characters revealed so far, at 26 per second
    step = 9
    top = max(0, (g["h2"] - len(lines) * step) // 2)
    ramp = _ramp("skin")
    for i, line in enumerate(lines):
        y = top + i * step
        if budget <= 0 or y + 8 > g["h2"]:
            break
        shown = line[: int(min(budget, len(line)))]
        color = ramp[(210 if i == 2 else 150) - i * 8]
        _centered(canvas, shown, y, color * (0.6 + 0.4 * _beat(t)))
        # the cursor blinks wherever the typing got to, and then sits at the
        # end of the last line blinking away until the loader hands over.
        typing = budget < len(line) + 6
        if (typing or i == len(lines) - 1) and (u * 3.0) % 1.0 < 0.6:
            x0, gap = _centered_at(g["w"], line)
            # a line that fills the width would push the cursor off the edge,
            # so it backs up rather than getting cut in half out there.
            x = min(x0 + len(_glyphs(shown)) * (8 + gap), g["w"] - 8)
            canvas[y:y + 8, max(x, 0):max(min(x + 8, g["w"]), 0)] = color
        budget -= len(line) + 6
    # one scanline sweeps down the tube, because the tube is warming up.
    sweep = int((t * 0.6 % 1.0) * g["h2"])
    canvas[sweep:sweep + 1] += 0.25
    return canvas


def _sc_plasma(g, t: float):
    """Plasma, copper bars, logo, scroller. The opening statement."""
    canvas = _plasma(g, t) * 0.85
    _copper(canvas, g, t)
    _logo(canvas, g, t, _beat(t))
    _scroller(canvas, g, t)
    return canvas


def _sc_tunnel(g, t: float):
    """Down the fryer pipe, with a small potato tumbling ahead of us."""
    canvas = _tunnel(g, t)
    _potato(canvas, g, t, scale=0.20, dist=3.0)
    _scroller(canvas, g, t)
    return canvas


def _sc_stars(g, t: float):
    """Deep space, three star layers, and the potato in full wireframe."""
    canvas = _plasma(g, t, "cool", 9.0) * 0.10
    _stars(canvas, g, t)
    _potato(canvas, g, t, scale=0.40)
    _scroller(canvas, g, t)
    return canvas


def _sc_sierpinski(g, t: float):
    """Two ways to draw the same triangle: an automaton and a random walk."""
    np = _np()
    canvas = _plasma(g, t, "cool", 8.0) * 0.10
    # rule 90 climbing the screen makes the backdrop, and it is the same shape
    table = _rule90(g["w"])
    rows = (g["rows"].astype(np.int32) + int(t * 14.0)) % table.shape[0]
    canvas += table[rows][..., None] * _ramp("cool")[196] * 0.45
    pts = _chaos_points()
    a = t * 0.5
    c, s = np.cos(a), np.sin(a)
    zoom = (0.36 + 0.05 * np.sin(t * 0.7)) * min(g["w"], g["h2"])
    _plot(canvas,
          g["w"] * 0.5 + (pts[0] * c - pts[1] * s) * zoom,
          g["h2"] * 0.5 + (pts[0] * s + pts[1] * c) * zoom,
          _ramp("wild")[(np.arange(pts.shape[1]) * 3 + int(t * 60.0)) % PALETTE_SIZE],
          lighten=True)
    _scroller(canvas, g, t)
    return canvas


def _sc_spud(g, t: float):
    """The hero shot. One potato, lit, turning in the dark."""
    canvas = _plasma(g, t, "cool", 7.0) * 0.09
    _stars(canvas, g, t, speed=0.05)
    _solid_potato(canvas, g, t)
    # the outline needs a pixel either side, so the long caption only goes up
    # when there is room for it.
    _caption(canvas, g, "110 CALORIES" if g["w"] >= 92 else "110 KCAL", t)
    return canvas


def _sc_julia(g, t: float):
    """A Julia set with its constant walking round a circle."""
    canvas = _julia(g, t)
    _caption(canvas, g, "MARIS PIPER", t)
    return canvas


def _sc_curves(g, t: float):
    """Calculator geometry: a hypotrochoid and a Lissajous figure."""
    canvas = _plasma(g, t, "cool", 6.0) * 0.08
    _curve(canvas, g, t)
    _scroller(canvas, g, t)
    return canvas


def _sc_roto(g, t: float):
    """A rotozoomer over the shredded tile, logo riding on top."""
    canvas = _roto(g, t)
    _copper(canvas, g, t, count=3, thickness=2.2)
    _logo(canvas, g, t, _beat(t))
    return canvas


def _sc_twister(g, t: float):
    """The twister, standing in the rising heat of the griddle."""
    canvas = _plasma(g, t, "cool", 11.0) * 0.20
    _twister(canvas, g, t)
    _sizzle(canvas, g, t)
    _scroller(canvas, g, t)
    return canvas


def _burst_drift(count: int):
    """Return a fixed shove for every point on the potato, seeded once."""
    drift = _CACHE.get(("drift", count))
    if drift is None:
        np = _np()
        rng = np.random.default_rng(0xB0057)
        _CACHE[("drift", count)] = drift = rng.normal(0.0, 1.0, (3, count)).astype(np.float32)
    return drift


def _chrome_potato(canvas, g, t: float, burst: float = 0.0, scale: float = 0.42,
                   dist: float = 5.2):
    """Turn the potato in polished metal, and blow it apart when asked.

    Nothing here is lit: every point works out where the world would be if it
    could see itself in the surface, and looks up what is there. That is what
    makes it read as chrome rather than as a shiny plastic potato.
    """
    np = _np()
    size = min(g["w"], g["h2"]) * scale
    pts, normals, _eyes, _mottle = _potato_solid(sum(size >= step for step in (16, 30)))
    rot = _rotation(t, (0.09, 0.62, 0.05))
    p = rot @ pts
    n = rot @ normals
    facing = -n[2]
    keep = facing > -0.3
    p, n, facing = p[:, keep], n[:, keep], facing[keep]
    if burst > 0.0:
        # it comes apart as a wave crossing it, not all in one go, and the
        # pieces keep the normals they had so they still catch the light.
        shove = _burst_drift(pts.shape[1])[:, keep]
        gone = np.clip(burst - (0.55 + 0.45 * p[0] / 1.6) * 0.7, 0.0, None)
        p = p + (n * 1.2 + shove * 0.8) * gone
        p[2] = p[2] - gone * 1.5  # and drifts toward us on the way out
    z = p[2] + dist
    order = np.argsort(-z)
    persp = dist / np.maximum(z, 0.35)
    px = g["w"] * 0.5 + p[0][order] * persp[order] * size
    py = g["h2"] * 0.5 + p[1][order] * persp[order] * size
    # the direction the view bounces off into, which is what it reflects
    bounce_x = 2.0 * facing * n[0]
    bounce_y = 2.0 * facing * n[1]
    bounce_z = 2.0 * facing * n[2] + 1.0
    # The environment it is reflecting: a dark floor, a bright sky, a hard line
    # where they meet and a few studio strips above it. Metal reads as metal
    # because of the edges between those, not because of any shading, so
    # everything here is deliberately abrupt.
    up = -bounce_y
    sky = np.clip(up * 2.6 + 0.42, 0.0, 1.0)
    horizon = np.exp(-((up * 8.0) ** 2)) * 0.95
    strips = (np.sin(up * 10.0 + bounce_x * 1.4 + t * 0.3) > 0.74) * 0.8
    glint = np.clip(-bounce_y * 0.5 - bounce_x * 0.5 + bounce_z * 0.7, 0.0, 1.0) ** 26
    edge = (1.0 - np.clip(facing, 0.0, 1.0)) ** 3
    tone = np.clip(sky * 0.42 + horizon + strips + glint * 1.4 + edge * 0.5 + 0.03,
                   0.0, 1.0)[order]
    ramp = _ramp("main")
    color = ramp[(20 + tone * 168).astype(np.int32)] * (0.12 + 0.88 * tone)[:, None]
    if burst > 0.0:
        color = color * np.clip(1.25 - burst * 0.22, 0.0, 1.0)
    body = _canvas(g)
    cover = np.zeros(body.shape[:2], bool)
    _plot(body, px, py, color, cover=cover)
    if burst <= 0.0:
        # while it is whole it has to look solid; once it is shrapnel the gaps
        # between the pieces are the point.
        _close_gaps(body, cover, rounds=max(2, int(size / 9)))
    canvas[cover] = body[cover]


def _sc_finale(g, t: float):
    """The sign off: one potato in chrome, and then it does not survive it."""
    np = _np()
    _run, began, room = _scroll_run(t)
    # the burst starts the moment the message has finished crossing
    burst = max(0.0, t - (began + room))
    canvas = _plasma(g, t, "cool", 5.0) * 0.06
    _chrome_potato(canvas, g, t, burst)
    if burst > 0.0:
        # and the words arrive out of the blast rather than sitting there all
        # the way through it, once there is enough potato out of the way to
        # read them against
        show = float(np.clip((burst - 1.6) * 0.9, 0.0, 1.0))
        if show > 0.0:
            sx = _fit(max(FAREWELL, key=len), g["w"], limit=2)
            top = max(2, g["h2"] // 2 - 9 * sx)
            for i, line in enumerate(FAREWELL):
                _centered(canvas, line, top + i * 10 * sx, _ramp("main")[186] * show, sx, sx,
                          outline=True)
    _scroller(canvas, g, t)
    # and then the camera carries on down, out of this and back into the storm
    pan = _smoothstep(
        (t - (DEMO_SECONDS - LOOP_BACK_SECONDS))
        / (LOOP_BACK_SECONDS - LOOP_SETTLE_SECONDS)
    )
    if pan > 0.0:
        # we scroll up into black, not into the sky. drawing the storm here
        # meant a dim unlit cloud sliding in a row at a time, which read as
        # gray bands rather than as weather -- and lighting it on the way in
        # was worse. the weather comes up out of the dark once we have
        # arrived, in `_sc_storm`.
        sky = _canvas(g)
        h2 = g["h2"]

        def slid(shift: int):
            """The two pictures stacked, with the join `shift` rows up."""
            out = _canvas(g)
            shift = max(0, min(shift, h2))
            if shift < h2:
                out[: h2 - shift] = canvas[shift:]
            if shift > 0:
                out[h2 - shift:] = sky[:shift]
            return out

        # the pan used to round to whole rows, and a whole row is a big step:
        # over the two and a bit seconds it takes, thirty-one frames out of
        # seventy-eight did not move at all and the rest jumped a row. that
        # reads as juddering rather than as a camera move. blending the two
        # neighboring offsets buys back the positions in between.
        exact = pan * h2
        low = int(exact)
        frac = exact - low
        canvas = slid(low) if frac < 1e-3 else (
            slid(low) * (1.0 - frac) + slid(low + 1) * frac)
    return canvas


_SCENE_FUNCS = {
    "storm": _sc_storm, "boot": _sc_boot, "plasma": _sc_plasma, "tunnel": _sc_tunnel,
    "sierpinski": _sc_sierpinski, "spud": _sc_spud, "julia": _sc_julia,
    "stars": _sc_stars, "curves": _sc_curves, "roto": _sc_roto,
    "twister": _sc_twister, "finale": _sc_finale,
}


def _scene_at(now: float) -> tuple[int, float]:
    """Return the index of the scene playing at ``now`` and when it started."""
    index = 0
    for i, start in enumerate(_STARTS):
        if now >= start:
            index = i
    return index, _STARTS[index]


# ------------------------------------------------------------- the frame -----


def _post(canvas, g, t: float):
    """Apply the whole-screen finish: vignette, scanlines, and a clamp."""
    np = _np()
    canvas *= g["vign"]
    # every second pixel row is the background half of a cell, and dimming it
    # gives us the scanline gap a real tube had.
    canvas[1::2] *= 0.90
    np.clip(canvas, 0.0, 1.0, out=canvas)


def _rgb(packed: int) -> tuple[int, int, int]:
    """Expand five bits per channel back into 0..255."""
    return (
        ((packed >> 10) & 31) * 255 // 31,
        ((packed >> 5) & 31) * 255 // 31,
        (packed & 31) * 255 // 31,
    )


def _hex(packed: int) -> str:
    """Expand a packed color into a #rrggbb string."""
    return "#%02x%02x%02x" % _rgb(packed)


def _xterm(packed: int) -> int:
    """Return the nearest xterm-256 index, for terminals without true color."""
    r, g, b = _rgb(packed)
    if r == g == b:
        # the gray ramp is finer than the color cube, so neutrals go there
        if r < 8:
            return 16
        if r > 238:
            return 231
        return 232 + (r - 8) * 24 // 231
    return 16 + 36 * (r * 5 // 255) + 6 * (g * 5 // 255) + (b * 5 // 255)


def _style(key: int) -> str:
    """Return the prompt_toolkit style for a packed top/bottom color pair."""
    style = _STYLES.get(key)
    if style is None:
        if len(_STYLES) > 30000:
            # the cache is only a speed trick, so dropping it costs us nothing.
            _STYLES.clear()
        style = f"fg:{_hex(key >> 15)} bg:{_hex(key & 0x7FFF)}"
        _STYLES[key] = style
    return style


def _escape(key: int, truecolor: bool) -> str:
    """Return the raw escape sequence that sets both colors of one cell."""
    cache_key = (truecolor, key)
    esc = _ESCAPES.get(cache_key)
    if esc is None:
        if len(_ESCAPES) > 30000:
            _ESCAPES.clear()
        top, bottom = key >> 15, key & 0x7FFF
        if truecolor:
            esc = "\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm" % (_rgb(top) + _rgb(bottom))
        else:
            esc = "\x1b[38;5;%d;48;5;%dm" % (_xterm(top), _xterm(bottom))
        _ESCAPES[cache_key] = esc
    return esc


def _pack(canvas):
    """Quantize to five bits per channel and pack each cell's two pixels into one int.

    Five bits is plenty for a gradient, it keeps the color caches small, and it
    makes runs of identical cells long enough to be worth collapsing.
    """
    np = _np()
    q = (np.clip(canvas, 0.0, 1.0) * 31.0 + 0.5).astype(np.int32)
    packed = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    return (packed[0::2] << 15) | packed[1::2]


def _runs(row, first: int, last: int):
    """Return the (key, start, stop) runs of identical cells in part of a row."""
    np = _np()
    span = row[first:last]
    cuts = np.flatnonzero(span[1:] != span[:-1]) + 1
    starts = [0, *cuts.tolist()]
    stops = [*cuts.tolist(), last - first]
    return zip(span[starts].tolist(), starts, stops)


def _fragments(canvas) -> list[tuple[str, str]]:
    """Turn an rgb canvas into prompt_toolkit fragments, one run at a time."""
    keys = _pack(canvas)
    height, width = keys.shape
    out: list[tuple[str, str]] = []
    for r in range(height):
        for key, start, stop in _runs(keys[r], 0, width):
            out.append((_style(key), HALF_BLOCK * (stop - start)))
        if r != height - 1:
            out.append(("", "\n"))
    return out


def ansi(keys, previous=None, truecolor: bool = True) -> str:
    """Turn packed cell colors into the escape sequences that paint them.

    Rows that match ``previous`` are skipped entirely and the rest are trimmed
    to the columns that actually changed, so a quiet scene costs almost nothing
    to redraw. Writing this ourselves is a lot faster than asking a renderer to
    diff a screen full of characters that all changed anyway.
    """
    np = _np()
    height, width = keys.shape
    # no autowrap: a glyph in the bottom right corner must not scroll the screen.
    out = ["\x1b[?7l"]
    changed = None
    if previous is not None and previous.shape == keys.shape:
        changed = keys != previous
    for r in range(height):
        first, last = 0, width
        if changed is not None:
            cols = np.flatnonzero(changed[r])
            if cols.size == 0:
                continue
            first, last = int(cols[0]), int(cols[-1]) + 1
        out.append("\x1b[%d;%dH" % (r + 1, first + 1))
        last_escape = None
        for key, start, stop in _runs(keys[r], first, last):
            esc = _escape(key, truecolor)
            # two runs can quantize down to the same escape on a 256 color
            # terminal, and repeating it would only cost bytes.
            if esc != last_escape:
                out.append(esc)
                last_escape = esc
            out.append(HALF_BLOCK * (stop - start))
    return "".join(out) if len(out) > 1 else ""


def render(width: int, height: int, tick: int, scheme: int = 0):
    """Draw one frame and return the finished rgb canvas, shaped (height*2, width, 3).

    This is where the running order lives: pick the scene for the moment, draw
    it, blend the one before it if we are still inside a crossfade, then put the
    whole picture through the vignette and the scanlines.
    """
    _SCHEME[0] = int(scheme) % len(SCHEMES)
    g = _grid(width, height * 2)
    now = (tick / FPS) % DEMO_SECONDS
    index, start = _scene_at(now)
    local = now - start
    canvas = _SCENE_FUNCS[SCENES[index][0]](g, now)
    if index > 0 and local < CROSSFADE:
        k = local / CROSSFADE
        canvas *= k
        canvas += _SCENE_FUNCS[SCENES[index - 1][0]](g, now) * (1.0 - k)
        canvas += (1.0 - k) ** 3 * 0.35  # a flash on the cut, for the drama
    # the first scene neither fades up nor blends: the last one has already
    # panned into it, so the loop comes round without a join to hide.
    _post(canvas, g, now)
    return canvas


def frame(width: int, height: int, tick: int, scheme: int = 0) -> list[tuple[str, str]]:
    """Render one frame of the demo as prompt_toolkit fragments.

    The result covers exactly ``height`` rows of exactly ``width`` cells, with a
    newline fragment between rows and none after the last. It depends on nothing
    but its arguments, so the caller owns the clock and the tests do not need a
    terminal.
    """
    if width < 1 or height < 1:
        return []
    return _fragments(render(width, height, tick, scheme))


def to_text(fragments) -> str:
    """Join fragments back into plain text, which is how you eyeball a frame."""
    return "".join(text for _style, text in fragments)


def frame_for_output(output, tick: int, scheme: int = 0) -> list[tuple[str, str]]:
    """Render a frame sized to whatever the output says the terminal is now.

    We ask the output every single frame rather than once at startup, because a
    resize mid-demo has to be picked up or the picture smears.
    """
    size = output.get_size()
    return frame(size.columns, size.rows, tick, scheme)


# ------------------------------------------------------------------ runner --


def _direct_depth(output):
    """Return True for true color, False for 256, or None if we must not paint raw.

    We only write our own escape sequences to an output we know speaks them.
    Anything else goes back through prompt_toolkit, which is slower but knows
    how to talk to whatever it is.
    """
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.output.vt100 import Vt100_Output

    speaks_vt100 = isinstance(output, Vt100_Output)
    try:  # only importable on windows, where it wraps a vt100 output of its own
        from prompt_toolkit.output.windows10 import Windows10_Output

        speaks_vt100 = speaks_vt100 or isinstance(output, Windows10_Output)
    except Exception:
        pass
    if not speaks_vt100:
        return None
    depth = output.get_default_color_depth()
    if depth == ColorDepth.DEPTH_24_BIT:
        return True
    if depth == ColorDepth.DEPTH_8_BIT:
        return False
    return None


# A frame costs bytes in proportion to the number of cells in it, and on a
# remote terminal those bytes are the whole story: a maximized window costs six
# times an eighty by twenty-four one at the same rate. Past a certain size we
# ease off rather than asking the link to carry all of it.
CELLS_AT_FULL_RATE = 5000  # about a hundred columns by fifty rows
EASED_FPS = 20


def paint_rate(cells: int, fps=None) -> float:
    """Return how many frames a second to aim for on a terminal of this size.

    A rate asked for outright is honored whatever the size; left alone, this
    eases from the usual rate down to ``EASED_FPS`` as the terminal grows past
    ``CELLS_AT_FULL_RATE``, and no further. Twenty is about where this content
    still reads as motion: below fifteen the scroller starts to strobe, because
    it moves two or three pixels a frame.
    """
    if fps:
        return float(fps)
    if cells <= CELLS_AT_FULL_RATE:
        return float(FPS)
    over = min((cells - CELLS_AT_FULL_RATE) / CELLS_AT_FULL_RATE, 1.0)
    return FPS - (FPS - EASED_FPS) * over


def _next_due(due: float, now: float, period: float) -> float:
    """Return when the next frame is due, giving up on any we have missed.

    A terminal that cannot keep up leaves us behind, and the frames we are
    behind by are gone. Chasing them only makes pictures nobody will see, at
    full tilt, so we let them go and aim at the next one instead.
    """
    return now + period if due < now else due


# How long we take pushing whatever was on the terminal up and off the top.
SCROLL_IN_SECONDS = 2.6


def _pan_in(output, input_, truecolor: bool, scheme: int, clock):
    """Push whatever is on the terminal up and off the top, coming in under it.

    This runs before the application starts, because prompt_toolkit erases from
    the cursor down on its first render and that would take the caller's screen
    with it. The terminal is put in raw mode for the duration so that anything
    typed at it does not echo across the picture.
    """
    size = output.get_size()
    width, height = size.columns, size.rows
    if width < 1 or height < 1:
        return None
    with input_.raw_mode():
        output.write_raw("\x1b[?25l\x1b[?7l")  # no cursor, and no autowrap
        began = time.monotonic()
        shape = None
        for row in range(height):
            keys = _pack(render(width, height, clock(), scheme))
            shape = keys.shape
            output.write_raw(_pan_in_row(keys, row, width, height, truecolor))
            output.flush()
            due = began + SCROLL_IN_SECONDS * (row + 1) / height
            time.sleep(max(0.0, due - time.monotonic()))
    return shape


def _pan_in_row(keys, row: int, width: int, height: int, truecolor: bool) -> str:
    """Return the escapes that scroll the terminal up a line and fill the gap.

    We do not know what the caller had on screen and we do not have to: the
    terminal still has it. We ask it to scroll, and paint our own next row into
    the gap that opens at the bottom. Their screen genuinely pans away instead
    of being wiped, which is the whole point of doing it this way, and it costs
    them one screenful of scrollback.
    """
    painted = ["\x1b[S", "\x1b[%d;1H" % height]
    for key, first, last in _runs(keys[row], 0, width):
        painted.append(_escape(key, truecolor))
        painted.append(HALF_BLOCK * (last - first))
    return "".join(painted)


def _loop_is_busy() -> bool:
    """Say whether another prompt_toolkit application already owns the loop."""
    from prompt_toolkit.application.current import get_app_or_none

    other = get_app_or_none()
    return other is not None and other.is_running


def run(*, fps=None, max_seconds: float | None = None, scheme: int = 0) -> None:
    """Run the demo full-screen until any key is pressed.

    A second prompt_toolkit application takes over the terminal and hands it
    back on the way out. Escape or an interrupt quits; every other key moves on
    to the next color scheme. ``max_seconds`` gives it a time limit, which is
    what the tests and
    ``--seconds`` use. ``fps`` pins the redraw rate; left alone it eases off on
    a big terminal, where every frame costs proportionally more to send.

    It is safe to call this from inside another application's key binding.
    asyncio has no nested event loops, so when something else is already
    running we put the demo on a thread of its own and block the caller until
    it finishes, which leaves the outer application paused but intact.
    """
    import asyncio

    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app_session
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    bindings = KeyBindings()
    stopped: list[bool] = []
    failed: list[BaseException] = []
    current = [int(scheme) % len(SCHEMES)]

    def bail() -> None:
        # exiting is only legal once, and only while the app is still running.
        if not stopped and app.is_running:
            stopped.append(True)
            app.exit()

    # Escape and an interrupt are the ways out. Everything else is a color
    # scheme, which means leaning on the keyboard repaints the demo rather than
    # ending it. We do not ask for escape eagerly: a lone one has to be told
    # apart from the start of an arrow key, and prompt_toolkit does that by
    # waiting a moment to see what follows.
    @bindings.add(Keys.Escape)
    @bindings.add(Keys.ControlC)
    @bindings.add(Keys.SIGINT)
    def _quit(event) -> None:
        bail()

    @bindings.add(Keys.Down)
    def _previous_scheme(event) -> None:
        current[0] = (current[0] - 1) % len(SCHEMES)

    @bindings.add(Keys.Any)
    def _next_scheme(event) -> None:
        current[0] = (current[0] + 1) % len(SCHEMES)

    # numpy, the palettes, the font and the potato mesh all get built here
    # rather than on the first frame, so we do not take the screen over and
    # then sit on a black rectangle while it happens.
    frame(80, 24, 0, current[0])
    start = time.monotonic()

    def tick() -> int:
        return int((time.monotonic() - start) * FPS)

    async def painter() -> None:
        """Paint straight to the terminal, on our own clock."""
        output = app.output
        previous = None
        shape = panned
        due = time.monotonic()
        while True:
            try:
                size = output.get_size()
                period = 1.0 / paint_rate(max(size.columns * size.rows, 1), fps)
                if size.columns > 0 and size.rows > 0:
                    keys = _pack(render(size.columns, size.rows, tick(), current[0]))
                    if keys.shape != shape:
                        # a resize invalidates every cell we thought we knew
                        shape = keys.shape
                        previous = None
                        output.write_raw("\x1b[2J")
                    painted = ansi(keys, previous, truecolor)
                    previous = keys
                    if painted:
                        output.write_raw(painted)
                        output.flush()
            except BaseException as exc:  # a broken frame must not strand the terminal
                failed.append(exc)
                bail()
                return
            now = time.monotonic()
            due = _next_due(due + period, now, period)
            await asyncio.sleep(max(0.002, due - now))

    def fragments() -> list[tuple[str, str]]:
        try:
            return frame_for_output(app.output, tick(), current[0])
        except BaseException as exc:
            failed.append(exc)
            bail()
            return []

    def pre_run() -> None:
        # the time limit rides on the event loop rather than on the frame
        # callback, so it still fires if the terminal is too slow to redraw.
        if max_seconds is not None:
            asyncio.get_running_loop().call_later(max_seconds, bail)
        if truecolor is not None:
            app.create_background_task(painter())

    from prompt_toolkit.output.defaults import create_output

    # Whether we paint for ourselves decides the rest of the shape of this: if
    # we do, there is no alternate screen, because we want the caller's own
    # screen sitting there to push off the top rather than hidden behind a
    # buffer that gets handed straight back at the end.
    truecolor = _direct_depth(get_app_session().output or create_output())
    empty = FormattedTextControl(lambda: [])
    app = Application(
        layout=Layout(Window(empty, always_hide_cursor=True)),
        key_bindings=bindings,
        full_screen=truecolor is None,
        mouse_support=False,
    )
    panned = None
    if truecolor is None:
        # prompt_toolkit waits the refresh interval between renders and then
        # pays the render cost on top, so we ask for roughly twice the rate we
        # want and let the renderer be the thing that slows it down.
        app.layout = Layout(Window(FormattedTextControl(fragments), always_hide_cursor=True))
        app.refresh_interval = 1.0 / max(paint_rate(80 * 24, fps) * 2, 1)
    else:
        panned = _pan_in(app.output, app.input, truecolor, current[0], tick)
    try:
        app.run(pre_run=pre_run, in_thread=_loop_is_busy())
    finally:
        # Whatever happened, we hand the terminal back in a usable state. When
        # we painted it ourselves that means undoing what we turned off and
        # clearing up after us: their own screen went off the top on the way in,
        # so leaving the last frame sitting there helps nobody.
        if truecolor is not None:
            app.output.write_raw("\x1b[?7h\x1b[?25h\x1b[2J\x1b[H")
        app.output.reset_attributes()
        app.output.flush()
    if failed:
        raise failed[0]


def _scheme_index(value: str) -> int:
    """Turn a scheme name or number from the command line into an index."""
    if value.isdigit():
        return int(value) % len(SCHEMES)
    matches = [i for i, name in enumerate(SCHEME_NAMES) if name.startswith(value.lower())]
    if not matches:
        raise SystemExit(f"unknown scheme {value!r}; pick from {', '.join(SCHEME_NAMES)}")
    return matches[0]


def main(argv=None) -> int:
    """Run the demo from the command line, which is how it gets developed."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="taters.ui.hashbrowns",
        description="run the hashbrowns demo. any key changes the colors; "
                    "escape or ctrl-c quits.",
    )
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long")
    parser.add_argument("--fps", type=int, default=None,
                        help="pin the redraw rate; by default it eases off on a big terminal")
    parser.add_argument("--scheme", default="0",
                        help="color scheme to start on: " + ", ".join(SCHEME_NAMES))
    parser.add_argument("--still", type=int, default=None, metavar="TICK",
                        help="print one frame as plain text instead of running")
    parser.add_argument("--size", default="80x24", metavar="WxH", help="size for --still")
    args = parser.parse_args(argv)
    scheme = _scheme_index(args.scheme)
    if args.still is not None:
        width, height = (int(v) for v in args.size.lower().split("x"))
        print(to_text(frame(width, height, args.still, scheme)))
        return 0
    run(fps=args.fps, max_seconds=args.seconds, scheme=scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
