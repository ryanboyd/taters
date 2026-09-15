"""
Every non-ASCII character the wizard prints, named once.

The problem this solves is invisible on Windows: the console there does **font
fallback**, quietly borrowing a glyph from another installed font when the one
you picked hasn't got it. So `↩`, `✓`, `⚠` and the rest render perfectly in
`cmd` while being absent from *every* monospace font Windows ships -- I checked
the `cmap` tables, and Courier New, Consolas and Lucida Console have none of
them. PuTTY does no fallback. It draws the missing-glyph box, and a menu full of
`↩ Back` turns into a menu full of `□ Back`.

That makes it our bug rather than the user's font. So the default set here
sticks to characters present in all three of those fonts, which is about as low
as the bar goes on Windows, and `TATERS_FANCY_GLYPHS=1` opts back up for a
terminal that can draw more.

Two things worth knowing before changing any of this:

* **Detection is impossible, and it is worth knowing why** so nobody tries. The
  usual trick is to print a character and ask the terminal where the cursor
  ended up -- but a missing-glyph box still advances exactly one column, so
  "drew it" and "drew a box" are indistinguishable from this end.
* **`sys.platform` tells you nothing**, and is the tempting wrong answer. It
  reports where Taters is running, not where the terminal is; somebody on a Mac
  over SSH to a Linux box reads as `linux` and their font is a total mystery
  either way.

Glyphs resolve on attribute access rather than at import, so a test can flip the
setting without reimporting half the UI::

    from .. import glyphs
    prompter.note(f"{glyphs.TICK} Saved")
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

__all__ = ["FANCY_ENV", "FANCY_KEY", "chosen", "fancy", "forget", "NAMES"]

#: Set this to opt into the nicer set, whatever the saved setting says.
FANCY_ENV = "TATERS_FANCY_GLYPHS"

#: Where the user's choice lives in ``settings.json``.
FANCY_KEY = "fancy_glyphs"

#: name -> (what we'd like to draw, what every stock Windows font can draw).
#:
#: The plain column is not an "ASCII mode" -- every one of these is a real
#: character with a real glyph in Courier New, Consolas and Lucida Console. The
#: screen should look deliberate on PuTTY, not degraded.
NAMES: Dict[str, Tuple[str, str]] = {
    "BACK":     ("↩", "←"),   # the way out of a submenu
    "TICK":     ("✓", "√"),   # done, saved, use-this-one
    "CROSS":    ("✗", "×"),   # finished with problems
    "DELETE":   ("✕", "×"),   # the destructive row in the library
    "WARNING":  ("⚠", "‼"),   # the one before deleting a folder
    "GEAR":     ("⚙", "*"),   # change how a model is applied
    "PENCIL":   ("✎", "›"),   # rename
    "ACTIVE":   ("▸", "►"),   # the step currently running
    "INDENT":   ("↳", "└"),   # a sub-item under the row above it
    "EXPORT":   ("⇩", "↓"),   # copy out of the library
    "KEYBOARD": ("⌨", ">"),   # type a path instead of browsing
    "IMPORT":   ("＋", "+"),   # copy into the library
    "ARC_TL":   ("╭", "┌"),   # the banner's corners. the arcs are the reason
    "ARC_TR":   ("╮", "┐"),   # the frame looks soft rather than boxy, and
    "ARC_BL":   ("╰", "└"),   # they are missing from Courier New and Lucida
    "ARC_BR":   ("╯", "┘"),   # though (unusually) present in Consolas.
}

_CACHE: Dict[str, bool] = {}


def chosen() -> bool:
    """
    The stored preference, read fresh: the environment, then the setting, then
    off. Never raises -- a damaged settings file costs a preference, not a
    session.

    Separate from :func:`fancy` so the settings screen can show the choice the
    moment it is made, while the screens already drawn keep the glyphs they were
    built with. Those two genuinely differ until the next start, and showing the
    stale one would read as the toggle not having worked.
    """
    raw = os.environ.get(FANCY_ENV, "").strip().lower()
    if raw:
        return raw not in ("0", "false", "no")
    try:
        from ..helpers.settings import load_settings
        return bool(load_settings().get(FANCY_KEY, False))
    except Exception:
        return False


def fancy() -> bool:
    """
    Whether to draw the nicer set. Cached: the settings screen says the choice
    takes effect at the next start, so re-reading the file on every glyph a
    progress display paints would buy nothing at all.
    """
    if "on" not in _CACHE:
        _CACHE["on"] = chosen()
    return _CACHE["on"]


def forget() -> None:
    """Drop the cached answer. For tests, and for the settings screen."""
    _CACHE.clear()


def __getattr__(name: str) -> str:
    try:
        pretty, plain = NAMES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no glyph named {name!r}") from None
    return pretty if fancy() else plain
