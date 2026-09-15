"""
Tests for the two things a terminal will not tell us: how many colors it can
draw, and which characters its font has.

Both defaults were chosen against a real failure -- the banner rendering as a
mostly-black potato over PuTTY, and `↩ Back` rendering as `□ Back` -- so the
tests here are mostly about not breaking the *other* platforms while fixing
that. The Linux framebuffer console really is a 16-color terminal, a redirected
log really must not fill up with escape codes, and both of those are invisible
on the Windows machine where the bug was reported.
"""

from __future__ import annotations

import pytest

from taters.helpers import settings as settings_mod
from taters.ui import glyphs
from taters.ui.console import (CHOICES, COLOR_ENV, COLOR_KEY, SIXTEEN_COLOR_TERMS,
                               asked_for, make_console, resolve_color_system)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(COLOR_ENV, raising=False)
    monkeypatch.delenv(glyphs.FANCY_ENV, raising=False)
    glyphs.forget()
    yield
    glyphs.forget()


# ---------------------------------------------------------------------------
# How many colors to send
# ---------------------------------------------------------------------------

def test_a_terminal_that_announces_no_colors_still_gets_256():
    """PuTTY, MobaXterm, Git Bash and plain ssh all announce a bare `xterm`.
    That is what the *client* said, not what it can do -- and taking it at face
    value is what turned the banner into a gray smear."""
    assert resolve_color_system("standard", "xterm") == "256"


def test_the_linux_console_keeps_its_sixteen_colors():
    """The framebuffer console at Ctrl-Alt-F2 genuinely stops at 16. This is
    the case a blanket upgrade would have broken, and it only exists on Linux
    -- which is exactly where nobody would have gone looking for it."""
    assert resolve_color_system("standard", "linux") == "standard"


def test_the_sixteen_color_exemptions_match_prompt_toolkits():
    """The rule is copied from prompt_toolkit rather than invented, because
    prompt_toolkit renders the menus in this same app -- and when the two
    disagreed, the menu came out in color and the banner came out gray."""
    from prompt_toolkit.output.vt100 import Vt100_Output
    import inspect

    source = inspect.getsource(Vt100_Output.get_default_color_depth)
    for term in SIXTEEN_COLOR_TERMS:
        assert term in source, (
            f"{term!r} is exempt here but not in prompt_toolkit; the two halves "
            "of the UI would disagree about it")


def test_a_redirected_log_is_never_given_escape_codes():
    """Rich reports None when the output is not a terminal, and naming a color
    system anyway makes it emit escapes regardless of where they are going. So
    None passes straight through, always."""
    assert resolve_color_system(None, "xterm") is None
    assert resolve_color_system(None, "xterm-256color") is None


def test_a_terminal_that_already_said_what_it_can_do_is_left_alone():
    """Nothing here should second-guess a detection that had something to go
    on -- including Windows, which never reaches this problem at all."""
    assert resolve_color_system("256", "xterm-256color") == "256"
    assert resolve_color_system("truecolor", "xterm-256color") == "truecolor"
    assert resolve_color_system("windows", "") == "windows"


@pytest.mark.parametrize("asked,expected", [
    ("truecolor", "truecolor"), ("256", "256"), ("16", "standard"), ("none", None)])
def test_an_explicit_choice_wins_over_the_detection(asked, expected):
    assert resolve_color_system("standard", "xterm", asked) == expected


def test_a_typo_in_the_setting_falls_back_rather_than_blanking_the_screen():
    assert resolve_color_system("standard", "xterm", "sixteen") == "256"


def test_the_environment_beats_the_saved_setting(monkeypatch):
    settings_mod.save_setting(COLOR_KEY, "16")
    monkeypatch.setenv(COLOR_ENV, "truecolor")
    assert asked_for() == "truecolor"


def test_the_saved_setting_is_used_when_the_environment_is_quiet():
    settings_mod.save_setting(COLOR_KEY, "16")
    assert asked_for() == "16"


def test_every_offered_choice_is_one_the_resolver_understands():
    """The settings screen builds its rows from CHOICES, so a name in one and
    not the other is a row that silently does nothing."""
    for name in CHOICES:
        resolve_color_system("standard", "xterm", name)  # must not raise
    assert "auto" in CHOICES


def test_the_console_the_wizard_builds_honors_all_of_this(monkeypatch):
    monkeypatch.setenv(COLOR_ENV, "16")
    assert make_console().color_system == "standard"
    monkeypatch.setenv(COLOR_ENV, "none")
    assert make_console().color_system is None


def test_the_potato_keeps_four_distinct_shades_at_256():
    """The whole point of the fix. At 16 colors two of the four snap together
    and the eyes land on near-black; the banner was built to survive 256, and
    this is what says so."""
    from rich.color import Color, ColorSystem
    from taters.ui.hub import _SKIN, _SPUD, _SPUD_DIM, _SPUD_LIT

    shades = [_SPUD_LIT, _SPUD, _SPUD_DIM, _SKIN]
    at_256 = {Color.parse(c).downgrade(ColorSystem.EIGHT_BIT).number for c in shades}
    at_16 = {Color.parse(c).downgrade(ColorSystem.STANDARD).number for c in shades}
    assert len(at_256) == 4, at_256
    assert len(at_16) < 4, "if 16 colors were enough this whole change is pointless"


def test_the_border_still_drifts_at_256():
    """At 16 colors every hue in the two-minute cycle maps to the same slot, so
    the frame stops moving entirely. 256 has to keep it alive."""
    from rich.color import Color, ColorSystem
    from taters.ui.hub import _BORDER_CYCLE_SECONDS, border_style

    at = [border_style(_BORDER_CYCLE_SECONDS * i / 8) for i in range(8)]
    slots = {Color.parse(c).downgrade(ColorSystem.EIGHT_BIT).number for c in at}
    assert len(slots) >= 4, slots


# ---------------------------------------------------------------------------
# Which characters to draw
# ---------------------------------------------------------------------------

#: Every character we print has to exist in all three of the monospace fonts
#: Windows ships, because PuTTY does no font fallback and draws a box for
#: anything its font is missing. This list is not from memory -- it is the
#: intersection of the actual `cmap` tables of Courier New, Consolas and Lucida
#: Console, checked with a script. Widening it means checking the new character
#: against those three fonts first, not adding it because it looks ordinary:
#: `↩`, `✓` and `⚠` all look ordinary and are in none of them.
STOCK_FONT_SAFE = {chr(c) for c in range(0x20, 0x7f)} | set(
    "·×–—•…‹›«»"
    "←↑→↓─│═║╔╗╚╝"
    "┌┐└┘├┤┬┴┼"
    "░▒▓▀▄█▌▐"
    "▲▼►◄○■▬√‼°±§¶"
)


def test_the_plain_glyphs_are_all_drawable_in_a_stock_windows_font():
    """The bug this fixes: `↩` is in *none* of Courier New, Consolas or Lucida
    Console, so PuTTY drew a box. Anything we fall back to has to be a
    character those fonts actually have."""
    for name, (_pretty, plain) in glyphs.NAMES.items():
        for ch in plain:
            assert ch in STOCK_FONT_SAFE, (
                f"{name}'s plain form {plain!r} uses U+{ord(ch):04X}, which is "
                "not known to exist in the stock Windows monospace fonts")


def test_plain_is_the_default_because_nothing_can_detect_a_font():
    """A terminal cannot be asked what its font has -- a missing-glyph box
    still advances one column -- so the safe set has to be what everybody gets
    without choosing anything."""
    assert glyphs.fancy() is False
    assert glyphs.BACK == "←"


def test_the_fancier_set_is_one_setting_away(monkeypatch):
    monkeypatch.setenv(glyphs.FANCY_ENV, "1")
    glyphs.forget()
    assert glyphs.BACK == "↩"
    assert glyphs.TICK == "✓"


def test_the_environment_can_also_turn_it_back_off(monkeypatch):
    settings_mod.save_setting(glyphs.FANCY_KEY, True)
    monkeypatch.setenv(glyphs.FANCY_ENV, "0")
    glyphs.forget()
    assert glyphs.fancy() is False


def test_every_glyph_has_both_forms_and_they_differ():
    for name, pair in glyphs.NAMES.items():
        pretty, plain = pair
        assert pretty and plain, name
        assert pretty != plain, f"{name} would not need an entry here"


def test_asking_for_a_glyph_that_does_not_exist_is_an_attribute_error():
    """A typo at a call site should fail loudly at that call site, not quietly
    print the word 'None' into somebody's menu."""
    with pytest.raises(AttributeError):
        glyphs.NOT_A_GLYPH


def test_a_damaged_settings_file_costs_a_preference_not_a_session():
    settings_mod.save_setting(glyphs.FANCY_KEY, {"not": "a bool"})
    glyphs.forget()
    assert glyphs.fancy() is True   # truthy dict, but it must not raise
    settings_mod.save_setting(COLOR_KEY, ["nonsense"])
    assert asked_for() == "auto"


# ---------------------------------------------------------------------------
# Where the menus use them
# ---------------------------------------------------------------------------

def test_no_menu_prints_a_character_a_stock_font_cannot_draw(tmp_path):
    """The end-to-end version: walk the wizard's own menus and check every
    label. This is what would have caught the original bug."""
    from taters.ui import hub
    from taters.ui.prompts import ScriptedPrompter

    p = ScriptedPrompter(["settings", "back", "quit"])
    hub.run_hub(p, cwd=tmp_path)

    drawn = [c.label for _q, rows in p.offered for c in rows]
    drawn += [c.help for _q, rows in p.offered for c in rows if c.help]
    drawn += list(p.output) + list(p.reasons)
    for text in drawn:
        for ch in text:
            if ord(ch) < 0x80:
                continue
            assert ch in STOCK_FONT_SAFE, (
                f"U+{ord(ch):04X} ({ch!r}) reaches the screen in {text!r}, and "
                "PuTTY would draw a box for it")


def test_both_potatoes_are_the_same_width_as_the_banner_expects():
    """The banner pads its right edge from the art's width, so a silhouette
    that is a column out puts the frame out of line on three rows."""
    import re

    from taters.ui.hub import _ART_WIDTH, _art

    for fancy in (True, False):
        for row in _art(fancy):
            assert len(re.sub(r"\[/?[^\]]*\]", "", row)) == _ART_WIDTH


def test_the_plain_potato_gives_up_its_corners_rather_than_its_shading():
    """Losing the quadrant blocks costs the rounded ends. It must not also cost
    the three-shade lighting, which is what stops a flat block of color looking
    like a brick."""
    from taters.ui.hub import _SKIN, _SPUD, _SPUD_DIM, _SPUD_LIT, _art

    plain = "".join(_art(False))
    for shade in (_SPUD_LIT, _SPUD, _SPUD_DIM, _SKIN):
        assert shade in plain


def test_the_terminal_screen_is_offered_under_settings():
    from taters.ui.tasks import settings as settings_task

    ids = [t.id for t in settings_task.entries()]
    assert "terminal" in ids
    assert ids[-1] == "hashbrowns", ids


def test_choosing_a_color_depth_saves_it(tmp_path):
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, terminal

    p = ScriptedPrompter(["color:truecolor", "back"])
    terminal.TASK.run(TaskContext(prompter=p, cwd=tmp_path))
    assert asked_for() == "truecolor"


def test_the_glyph_toggle_reads_back_immediately_even_though_the_ui_waits(tmp_path):
    """The screens already drawn keep the glyphs they were built with, but the
    toggle itself has to show the new value -- otherwise it reads as broken."""
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, terminal

    p = ScriptedPrompter(["glyphs", "back"])
    terminal.TASK.run(TaskContext(prompter=p, cwd=tmp_path))
    assert glyphs.chosen() is True
    assert any("Characters: fancy" in line for line in p.output), p.output


def test_turning_the_glyphs_back_off_forgets_the_setting_rather_than_storing_false(tmp_path):
    """Plain is the default, so "off" is the absence of a setting. Storing
    False instead would be indistinguishable to read but would mean a settings
    file that disagrees with a future change of default."""
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, terminal

    p = ScriptedPrompter(["glyphs", "glyphs", "back"])
    terminal.TASK.run(TaskContext(prompter=p, cwd=tmp_path))
    assert glyphs.FANCY_KEY not in settings_mod.load_settings()
