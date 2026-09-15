"""
One place that builds a ``rich`` Console, and one decision inside it: how many
colors to send.

Rich works this out from ``TERM``. That is the right instinct and the wrong
answer over SSH, because the number in ``TERM`` is what the *client announced*,
not what it can do -- and PuTTY, MobaXterm, Git Bash and plain ``ssh`` all
announce a bare ``xterm``. Rich reads no color suffix, concludes 16 colors, and
snaps our palette onto the nearest ANSI slots. The potato's four shades collapse
to two, the eyes and the whole frame land on bright-black (``#555555`` in
PuTTY's default palette, which on a black background is very nearly nothing),
and the border's slow hue drift dies completely -- every hue in the cycle maps
to the same slot. That is what a user sees as "mostly black with a couple of
blocks of color".

Nothing about the terminal is actually wrong. We just never sent it the bytes.

So we prefer 256 for any real terminal, and the rule is copied deliberately
rather than invented: it is exactly what ``prompt_toolkit`` already does for the
menus in this same app, and has done in every version we have ever shipped --

    if is_dumb_terminal(term):           return DEPTH_1_BIT
    if term in ("linux", "eterm-color"): return DEPTH_4_BIT
    return DEFAULT                       # 256

-- which is why on PuTTY the questionary menu looks right and only the banner
came out gray. Matching it means the two halves of the UI stop disagreeing, and
it means this is a rule with years of mileage behind it rather than a guess.

Two exclusions carried over from it, both real:

* **dumb terminals** get nothing, as before.
* **``linux``** -- the framebuffer console you get at Ctrl-Alt-F2 -- and
  ``eterm-color`` genuinely stop at 16, so they keep 16. This is the case a
  blanket upgrade would have broken, and it is a Linux-only case, which is
  exactly where nobody would have gone looking for it.

And one rule of our own, which matters more than it looks: **we only ever
upgrade a color system rich already detected.** When rich says ``None`` -- not a
terminal, ``NO_COLOR`` set, output redirected to a file -- we pass that straight
through and never name a system. Handing rich an explicit ``color_system`` makes
it emit escapes regardless of where the output is going, so the alternative is
escape codes in somebody's piped log.
"""

from __future__ import annotations

import os
from typing import Any, Optional

__all__ = ["COLOR_ENV", "COLOR_KEY", "CHOICES", "SIXTEEN_COLOR_TERMS",
           "resolve_color_system", "asked_for", "make_console"]

#: Wins over the saved setting, for a shared machine or a one-off session.
COLOR_ENV = "TATERS_COLOR"

#: Where the user's choice lives in ``settings.json``.
COLOR_KEY = "terminal_color"

#: What we accept, in the order the settings screen offers them. These are
#: rich's own names, apart from "auto" and "none", so the value doubles as the
#: argument we hand to Console.
CHOICES = ("auto", "truecolor", "256", "16", "none")

#: Terminals that really do stop at 16, so the preference for 256 skips them.
#: Same two names prompt_toolkit excludes.
SIXTEEN_COLOR_TERMS = frozenset({"linux", "eterm-color"})

_RICH_NAMES = {"truecolor": "truecolor", "256": "256", "16": "standard"}


def resolve_color_system(detected: Optional[str], term: str,
                         asked: str = "auto") -> Optional[str]:
    """
    How many colors to send, given what rich worked out on its own.

    Parameters
    ----------
    detected : str or None
        What rich's own detection returned -- its `Console.color_system`, so
        one of "standard", "256", "truecolor", "windows", or None for "this is
        not a terminal, send nothing".
    term : str
        The ``TERM`` environment variable, used only to spot the handful of
        terminals that really are limited to 16 colors.
    asked : str
        The user's preference: one of :data:`CHOICES`. Anything unrecognized is
        treated as "auto", because a typo in a setting should not blank
        somebody's screen.

    Returns
    -------
    str or None
        What to pass to ``Console(color_system=...)``.
    """
    if asked in _RICH_NAMES:
        return _RICH_NAMES[asked]
    if asked == "none":
        return None
    # "auto", or something we don't recognize.

    if detected != "standard":
        # everything except "standard" passes straight through, and that
        # deliberately includes None. rich returns None for "this is not a
        # terminal" -- output redirected to a file, NO_COLOR set, a dumb
        # terminal -- and naming a color system anyway makes it emit escapes
        # regardless of where they are going, which turns somebody's piped log
        # into a screenful of `\x1b[38;5;`. "standard" is the only answer rich
        # reaches by having nothing to go on, so it is the only one worth
        # second-guessing.
        return detected
    if term.strip().lower() in SIXTEEN_COLOR_TERMS:
        return detected
    return "256"


def asked_for() -> str:
    """The user's color preference: the environment, then the setting, then
    "auto". Never raises -- a damaged settings file costs a preference, not a
    session."""
    raw = os.environ.get(COLOR_ENV, "").strip().lower()
    if raw:
        return raw if raw in CHOICES else "auto"
    try:
        from ..helpers.settings import load_settings
        saved = str(load_settings().get(COLOR_KEY, "auto")).strip().lower()
        return saved if saved in CHOICES else "auto"
    except Exception:
        return "auto"


def make_console(**kwargs: Any) -> Any:
    """
    A rich Console with the color depth sorted out. Takes rich's own arguments.

    Built twice on purpose when the answer changes: we ask rich what it would
    have done rather than reimplementing its detection, so if rich learns to
    recognize a terminal we have never heard of, we inherit that instead of
    overriding it. Console construction reads environment variables and calls
    ``isatty``; nothing is written, so the throwaway costs nothing worth
    optimizing away.
    """
    from rich.console import Console

    probe = Console(**kwargs)
    chosen = resolve_color_system(probe.color_system, os.environ.get("TERM", ""),
                                  asked_for())
    if chosen == probe.color_system:
        return probe
    return Console(**{**kwargs, "color_system": chosen})
