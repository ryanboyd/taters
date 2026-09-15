"""
"Check for new versions" -- the one switch behind the note under the banner.

There is exactly one thing to decide here, so this screen is mostly an
explanation and a toggle. The explanation earns its space: an outbound
request nobody asked for is the kind of thing that starts a conversation with
IT, so this says plainly what gets sent (nothing but the request itself), how
often (once a day, in the background), and that the answer shown on the menu
came from a file rather than from the network just now.
"""
from __future__ import annotations

from .. import glyphs
from ..prompts import Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK"]

_ON, _OFF, _BACK = "on", "off", "back"


def _run(ctx: TaskContext):
    import os

    from ...helpers.settings import load_settings, save_setting
    from ...helpers.update_check import PYPI_JSON, SETTING_KEY, SKIP_ENV

    prompter = ctx.prompter
    while True:
        saved = load_settings().get(SETTING_KEY, {})
        if not isinstance(saved, dict):
            # settings.json is a plain file somebody can open in an editor, and
            # the screen that fixes a mangled setting should not be the one
            # that falls over on it.
            saved = {}
        forced_off = os.environ.get(SKIP_ENV, "").strip().lower() not in ("", "0", "false", "no")
        on = not saved.get("off")

        state = "on" if on else "off"
        latest = str(saved.get("latest", "") or "")
        prompter.note(f"\n  Checking for new versions is {state}."
                      + (f"\n  Last answer from PyPI: v{latest}" if latest else ""),
                      style="cyan", wrap=False)
        prompter.reason(
            f"When on, Taters asks {PYPI_JSON} for the current version at most once a "
            "day, in the background, and shows the version number under the menu if "
            "yours is older. Nothing about you or your data is sent, and the menu "
            "never waits for the answer.")
        if forced_off:
            prompter.reason(f"{SKIP_ENV} is set in this environment, so the check is off "
                            "no matter what is chosen here. Unset it to choose here.")

        rows = []
        if not on:
            rows.append(Choice(_ON, "Turn the check on",
                               "Ask PyPI once a day and note a newer version under the menu."))
        else:
            rows.append(Choice(_OFF, "Turn the check off",
                               "No requests, and no note. Nothing else changes."))
        rows.append(Choice(_BACK, f"{glyphs.BACK} Back", tone="nav"))

        try:
            picked = str(prompter.select("Checking for new versions:", rows))
        except GoBack:
            return None
        if picked == _BACK:
            return None

        # `off` is stored rather than `on` so that a settings file written by an
        # older Taters -- which has no key here at all -- reads as on, which is
        # the default we chose.
        save_setting(SETTING_KEY, {**saved, "off": picked == _OFF})
        prompter.note("  Saved.", style="green")


TASK = Task(
    id="update_check",
    label="Check for new versions",
    help="Whether Taters looks up the current version on PyPI",
    run=_run,
)
