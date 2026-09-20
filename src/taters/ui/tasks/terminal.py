"""
"How this looks in your terminal" -- the two things we cannot detect.

Both switches here exist because the terminal will not tell us the truth. Over
SSH the client announces a terminal type that says nothing about what it can
actually draw: PuTTY and friends announce a bare ``xterm`` while happily
rendering 24-bit color, and no terminal anywhere reports which characters its
font has. So we pick sensible defaults, and put the override where somebody who
can see their own screen can reach it.
"""
from __future__ import annotations

from .. import glyphs
from ..prompts import Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK"]

_BACK = "back"

_COLOR_HELP = {
    "auto": "Match the terminal, preferring 256 colors on anything that looks "
            "capable. Right almost everywhere.",
    "truecolor": "24-bit color. Every modern terminal does this; a few older "
                 "ones show nothing rather than something.",
    "256": "What `auto` settles on for most terminals. Safe everywhere.",
    "16": "For a terminal that really is limited, like a Linux text console.",
    "none": "No color at all.",
}


def _run(ctx: TaskContext):
    import os

    from ...helpers.settings import clear_setting, save_setting
    from ..console import CHOICES, COLOR_ENV, COLOR_KEY, asked_for
    from ..glyphs import FANCY_ENV, FANCY_KEY, chosen

    prompter = ctx.prompter
    while True:
        color = asked_for()
        forced = os.environ.get(COLOR_ENV, "").strip().lower()

        prompter.note(f"\n  Colors: {color}"
                      f"\n  Characters: {'fancy' if chosen() else 'plain'}",
                      style="cyan", wrap=False)
        if forced:
            prompter.reason(f"{COLOR_ENV} is set in this environment, and it wins "
                            "over anything chosen here.")
        prompter.reason(
            "A washed-out or nearly black potato means your terminal announces "
            "fewer colors than it can draw. Empty boxes instead of arrows and "
            "ticks mean your font has no glyph for them.")

        rows = [Choice(f"color:{name}", f"Colors: {name}", _COLOR_HELP[name])
                for name in CHOICES]
        rows.append(Choice("glyphs", f"Characters: switch to "
                                     f"{'plain' if chosen() else 'fancy'}",
                           "Plain sticks to characters every terminal font has. "
                           "Fancy looks better where the font can draw it."))
        rows.append(Choice(_BACK, f"{glyphs.BACK} Back", tone="nav"))

        try:
            picked = str(prompter.select("How this looks in your terminal:", rows))
        except GoBack:
            return None
        if picked == _BACK:
            return None

        if picked == "glyphs":
            # stored only when it differs from the default, so that a settings
            # file from an older Taters -- which has no key here -- keeps the
            # default rather than being read as an explicit choice.
            if chosen():
                clear_setting(FANCY_KEY)
            else:
                save_setting(FANCY_KEY, True)
            if os.environ.get(FANCY_ENV, "").strip():
                prompter.reason(f"{FANCY_ENV} is set here and wins over this.")
        else:
            save_setting(COLOR_KEY, picked.split(":", 1)[1])

        prompter.note("  Saved. It takes effect the next time Taters starts.",
                      style="green")


TASK = Task(
    id="terminal",
    label="How this looks in your terminal",
    help="Colors and characters, for when the banner looks wrong over SSH",
    run=_run,
)
