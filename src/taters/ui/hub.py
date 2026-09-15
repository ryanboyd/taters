"""
The front door: "What would you like to do?"

The wizard used to open with "Where is your data?", which quietly assumed the
answer to a question nobody had asked -- that you were here to extract features
from files. Running a saved pipeline and managing pipelines involve no data at
all, and extracting features *and running analyses* starts from a spreadsheet's
columns rather than from a folder.

So the first question is about intent, and the answer chooses a
:class:`~taters.ui.tasks.Task`. Data comes up inside the task that needs it.

Backing out
-----------
Cancelling *inside* a task returns here rather than ending the session: a
mistyped path should not cost someone the whole run. Cancelling at this menu
exits. That is the difference between ctrl-c meaning "not that" and ctrl-c
meaning "I am done".
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .prompts import MEASURE, Cancelled, Choice, GoBack, Prompter, QuitRequested
from .tasks import TaskContext, all_tasks

__all__ = ["run_hub", "banner", "version", "title", "FAREWELL"]

_TAGLINE = "Takes All Things, Extracts Relevant Stuff"
#: The mottos the banner can open with. One is chosen per *launch*: the same
#: session keeps its motto through every repaint (the banner redraws on every
#: screen, and a line that changed under the reader's eyes would read as a
#: glitch), while the next launch may greet them differently.
_MOTTOS = (
    "Let's get mashing!",
    "It's Mashin' Time!",
    "May the Spuds be with you.",
    "You had me at 'potato.'",
    "You're gonna need a bigger (gravy) boat.",
    "I feel the need — the need for spud.",
    "Use the fork, Luke.",
    "Here's looking at you, spud.",
    "Frankly, my dear, I don't give a yam.",
)


def _pick_motto() -> str:
    """One motto, at random. Its own function so tests can reseed and call."""
    import random

    return random.choice(_MOTTOS)


_ENCOURAGEMENT = _pick_motto()

#: Printed on the way out. A tool that has just spent an hour of someone's
#: compute budget can afford to be warmer than "Bye."
FAREWELL = "So long, and thanks for all the spuds!"


def version() -> str:
    """
    The installed version, or "" if it cannot be determined.

    Running from a source tree that was never installed has no distribution
    metadata, and a wizard that refuses to start because it cannot name itself
    would be a poor trade.
    """
    try:
        from importlib.metadata import version as _version
        return _version("taters")
    except Exception:
        try:
            from .. import __version__
            return str(__version__)
        except Exception:
            return ""


def title() -> str:
    """Short name plus version, for the progress rail."""
    v = version()
    return f"Taters v{v}" if v else "Taters"


# our palette. one rule here: brighter = more important. the name is the
# brightest thing, the frame is the dimmest, everything else sits in between,
# so that the eye lands on what the screen is for and not on the box around it.
#
# we keep the frame colourless. it used to be a warm brown, which put the most
# saturated color on the least informative thing on screen and made the header
# look like orange furniture around some text.
_SPUD = "#e8c47d"      # potato flesh: butter, not terracotta
_SKIN = "#7a5c3e"      # the eyes -- brown, and dark enough to read as holes
_SPUD_LIT = "#f3dca8"  # the side facing the light
_SPUD_DIM = "#c9a55f"  # ...and the side turned away from it
_LEAF = "#8bc34a"      # the encouragement -- the one spot of green we kept
_NAME = "bold #ffd479"  # brightest thing on the screen, and the only bold one
_TAG_STYLE = "#b8b8b8"
# the frame slowly drifts through a muted spectrum instead of sitting on one
# color, so that an hour-long session doesn't look like a screenshot of itself.
# every screen repaints the banner, so the tint moves as you work: two questions
# in a row look about the same, but the top and bottom of a long job are
# plainly different.
#
# we hold saturation and value low on purpose. same rule as above (brightness
# carries importance), and the frame is the least informative thing on screen;
# drifting it through anything vivid would make the box compete with what's
# inside it.
# these two numbers are set by what survives a 256-color terminal, not just by
# taste. our first go was 0.22/0.42, which looked right in truecolor but
# quantized to the *same* palette slot for every hue on anything older -- so the
# drift was invisible on most terminals. value is what decides it: below about
# 0.5 the whole spectrum collapses into one gray no matter the saturation.
_BORDER_CYCLE_SECONDS = 120.0
_BORDER_SATURATION = 0.35
_BORDER_VALUE = 0.55


def border_style(at: Optional[float] = None) -> str:
    """
    The frame's color right now, as a hex string.

    Parameters
    ----------
    at : float, optional
        A point on the cycle in seconds. Defaults to the monotonic clock, which
        is what makes it move; tests pass a value to look at a fixed moment.
    """
    import colorsys
    import time

    seconds = time.monotonic() if at is None else at
    hue = (seconds / _BORDER_CYCLE_SECONDS) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, _BORDER_SATURATION, _BORDER_VALUE)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )

# a potato. three rows, because that's what a potato is -- an oval with a
# couple of eyes in it. we tried four rows and a sprout first, and it read as a
# shrub in a pot: the green grabbed the eye and the thing under it was too tall
# and too square to pass for the vegetable.
#
# two things do the work here. `▗▄▄▄▖` and `▝▀▀▀▘` round the ends off with
# half-blocks (full blocks give square corners, and a potato has none). and the
# three shades run left to right, lit to shadowed. that's what keeps a flat
# block of color from looking like a brick.
#
# two eyes, on the middle row only. the top and bottom rows are half-height
# blocks, so a dot placed there lands in the empty half of the cell and looks
# like it's floating off the potato instead of sitting in it.
#
# the top and bottom rows are also inset by a column at each end while the body
# runs full width. that one-column difference is the whole silhouette: it's
# what makes the outline bulge in the middle and read as an oval. we tried all
# three rows the same width (to fit a `>_` terminal screen into the body like
# the project's logo has) and it came out as a rounded rectangle. the screen
# needs a bigger potato than three rows can give it.
_SPUD_ART = (
    f" [{_SPUD_LIT}]▗▄▄[/][{_SPUD}]▄[/][{_SPUD_DIM}]▄▄▖[/] ",
    f"[{_SPUD_LIT}]▐█[/][{_SKIN}]•[/][{_SPUD}]███[/][{_SKIN}]•[/][{_SPUD_DIM}]█▌[/]",
    f" [{_SPUD}]▝▀▀[/][{_SPUD_DIM}]▀▀▀▘[/] ",
)
_ART_WIDTH = 9         # the widest row, in columns


def _visible(markup: str) -> str:
    """
    The text as it will appear on screen, with the style tags removed.

    Padding has to be measured on this rather than on the raw string: a tag
    occupies no columns when rendered and plenty when counted, and a box whose
    right edge is computed from the raw length is a box that looks broken on
    every line that happens to carry a color.
    """
    import re

    return re.sub(r"\[/?[^\]]*\]", "", markup)


def banner(width: int = MEASURE - 2) -> str:
    """
    The header: a potato, and what this program is.

    Laid out as art beside text rather than as a stack inside a box. The box is
    still there, but as a light rounded frame -- the heavy double rule this used
    to draw made a tool for mashing audio look like a compliance report.

    Built rather than hard-coded because the version sits inside it, and a
    version string of a different length would otherwise push the right-hand
    border out of line.
    """
    text_rows = (
        f"[{_NAME}]TATERS[/] [dim]{version_line()}[/dim]",
        f"[{_TAG_STYLE}]{_TAGLINE}[/]",
        f"[italic {_LEAF}]{_ENCOURAGEMENT}[/]",
    )

    gap = "   "
    inner = max(
        width - 6,
        max(_ART_WIDTH + len(gap) + len(_visible(row)) for row in text_rows),
    )

    frame = border_style()

    def row(art: str, text: str) -> str:
        art_pad = " " * (_ART_WIDTH - len(_visible(art)))
        body = f"{art}{art_pad}{gap}{text}"
        pad = " " * (inner - len(_visible(body)))
        return f"  [{frame}]│[/]  {body}{pad}  [{frame}]│[/]"

    top = f"  [{frame}]╭" + "─" * (inner + 4) + "╮[/]"
    bottom = f"  [{frame}]╰" + "─" * (inner + 4) + "╯[/]"
    return "\n".join(
        ["", top, *(row(a, t) for a, t in zip(_SPUD_ART, text_rows)), bottom, ""]
    )


def version_line() -> str:
    """The version on its own line, or a note that there is no metadata."""
    v = version()
    return f"v{v}" if v else "(running from source)"


def _farewell(prompter: Prompter) -> None:
    """
    Sign off: the banner, then the goodbye, and nothing else.

    The live renderer wipes the screen per question, so by the time quit is
    chosen the banner sits a whole menu above the cursor -- the farewell then
    landed under the dead menu frame, alone. Ending on a fresh screen of
    potato-over-goodbye leaves the session signed off the way it opened.
    """
    prompter.clear()
    # wrap=False for the same reason as on the opening screen: the banner is
    # rich markup, and the 64-column wrap counts the markup as text and folds
    # the border mid-tag.
    prompter.note(banner(), wrap=False)
    prompter.note(f"  {FAREWELL}\n", style="dim")


def run_hub(prompter: Prompter, *, cwd: Optional[Path] = None) -> bool:
    """
    Show the menu, run what is chosen, and come back for the next thing.

    Parameters
    ----------
    prompter : Prompter
        Where the questions go.
    cwd : Path, optional
        The working folder: pipelines are saved as subfolders of it. Defaults to the
        current directory.

    Returns
    -------
    bool
        False if anything that ran finished with problems. `taters` turns that
        into a non-zero exit, so a script driving this still learns of a
        failure even though the session may have done several things.
    """
    ctx = TaskContext(prompter=prompter, cwd=Path(cwd or Path.cwd()))

    # the renderer redraws the header on every screen, so it has to own it --
    # and then it has to be the *only* thing that draws it, otherwise the banner
    # shows up twice on the opening screen.
    owns_header = hasattr(prompter, "set_header")
    if owns_header:
        # we hand over the function itself, not banner(): the live renderer
        # calls it on every paint, and that's what lets the border color
        # actually drift.
        prompter.set_header(banner)
    if hasattr(prompter, "_title"):
        prompter._title = title()

    prompter.clear()
    if not owns_header:
        # wrap=False here: the banner is rich markup, and `note`'s 64-column
        # wrap counts the markup characters as text -- it folds the border
        # mid-tag and the plain renderer spits out garbage.
        prompter.note(banner(), wrap=False)
    prompter.note("  Ctrl-C backs out at any point. Nothing is written until you say so.\n",
                  style="dim")

    everything_ok = True
    first = True
    while True:
        tasks = all_tasks()
        choices = [
            Choice(task.id, task.label, task.help, disabled=task.blocked(ctx))
            for task in tasks
        ]
        choices.append(Choice("quit", "Quit", ""))

        # "Anything else?" sounds like a follow-up to whatever just happened.
        # coming back from a submenu that's neither accurate nor reassuring --
        # the point is that you're back at the top.
        question = "What would you like to do?" if first else "Main menu"
        first = False

        try:
            picked = str(prompter.select(question, choices))
        except (GoBack, Cancelled):
            # at the front door, back *is* out. everywhere else Esc means "undo
            # the last question", but there's no question above this one -- and
            # a keypress that visibly does nothing looks like we ignored it,
            # which is worse than either answer.
            _farewell(prompter)
            return everything_ok

        if picked == "quit":
            _farewell(prompter)
            return everything_ok

        task = next(t for t in tasks if t.id == picked)
        try:
            try:
                if task.run(ctx) is False:
                    everything_ok = False
            finally:
                # the rail belongs to the task that put it up. without this the
                # wizard's stages stayed on screen above the main menu, still
                # describing a pipeline we weren't building anymore.
                prompter.reset_stages()
        except QuitRequested as quit_:
            # the user picked quit on purpose from a finished screen, so no
            # "backed out" message: the work is done and saying otherwise would
            # be a lie. the verdict rides along on the exception -- raising
            # skipped the task's `return False`, so a failed run's quit used to
            # exit 0.
            _farewell(prompter)
            return everything_ok and quit_.ok
        except (Cancelled, GoBack):
            # Esc at the first question of a task means "not this one after
            # all", which is the same thing as backing out of it.
            prompter.note("\n  Backed out. Nothing was changed.\n", style="dim")
        prompter.note("")
