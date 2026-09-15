"""
Tests for the "newer version available" note under the main menu.

Two properties matter more than the rest, and they are the reason this is a
file of its own rather than three assertions in test_hub.py:

* **The menu never waits on the network.** Whatever PyPI is doing, drawing the
  first screen has to be instant. So the note is read from a cached answer and
  the refresh happens in a thread whose result is for next time.
* **Off means off.** Somebody running this on data that cannot leave the
  building needs the switch to stop the request, not merely hide the note.

Everything here fakes `urlopen`; nothing in this file touches the network.
"""

from __future__ import annotations

import json
import time
from io import BytesIO

import pytest

from taters.helpers import settings as settings_mod
from taters.helpers import update_check as uc


@pytest.fixture(autouse=True)
def allowed(monkeypatch):
    """The suite turns the check off for every test; these tests are about it.

    Also pins the installed version, so the assertions do not move every time
    the real version is bumped.
    """
    monkeypatch.delenv(uc.SKIP_ENV, raising=False)
    monkeypatch.setattr(uc, "_installed", lambda: "0.7.2")


def fake_pypi(monkeypatch, version="0.7.3", *, boom=None):
    """Stand in for PyPI. Records the calls so a test can assert on silence."""
    calls = []

    def urlopen(request, timeout=None):
        calls.append(getattr(request, "full_url", request))
        if boom is not None:
            raise boom
        body = json.dumps({"info": {"version": version}}).encode("utf-8")

        class Response(BytesIO):
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        return Response(body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return calls


def cache(latest, *, at=None, off=False):
    saved = {"latest": latest, "at": time.time() if at is None else at}
    if off:
        saved["off"] = True
    settings_mod.save_setting(uc.SETTING_KEY, saved)


# ---------------------------------------------------------------------------
# The note itself
# ---------------------------------------------------------------------------

def test_a_newer_cached_version_becomes_one_plain_line():
    """The wording the user asked for: a fact, with no verb aimed at anybody."""
    cache("0.7.3")
    assert uc.note() == "newer version available: v0.7.3"


def test_nothing_is_said_before_the_first_check_has_ever_run():
    """A fresh install has no cached answer, and inventing one would mean
    blocking the menu on a request."""
    assert uc.note() == ""


def test_the_same_version_is_not_reported_as_newer():
    cache("0.7.2")
    assert uc.note() == ""


def test_an_older_cached_answer_is_not_reported_as_newer():
    """The cache can legitimately trail the installed version -- somebody
    upgrades, and yesterday's answer is still sitting in settings.json."""
    cache("0.7.1")
    assert uc.note() == ""


def test_versions_compare_as_numbers_rather_than_as_text():
    """The bug this pins: "0.7.10" sorts *before* "0.7.9" as a string, so a
    text comparison goes quiet exactly when there is something to say."""
    cache("0.7.10")
    assert uc.note() == "newer version available: v0.7.10"


def test_a_release_candidate_on_pypi_is_passed_over_in_silence(monkeypatch):
    """Telling somebody already on an rc that an older final is "newer" is
    worse than saying nothing, and the comparison for that is not worth a
    dependency."""
    monkeypatch.setattr(uc, "_installed", lambda: "0.8.0rc1")
    cache("0.7.9")
    assert uc.note() == ""


def test_running_from_a_source_tree_is_never_nagged(monkeypatch):
    """No distribution metadata means no version to compare -- and a developer
    does not need telling that PyPI is behind their working tree."""
    monkeypatch.setattr(uc, "_installed", lambda: "")
    cache("9.9.9")
    assert uc.note() == ""


def test_reading_the_note_does_not_touch_the_network(monkeypatch):
    """The whole point: the first screen is drawn from a file, not a socket."""
    calls = fake_pypi(monkeypatch)
    cache("0.7.3")
    assert uc.note()
    assert calls == [], "the menu made a request while drawing itself"


def test_a_damaged_cache_costs_a_note_and_nothing_else():
    settings_mod.save_setting(uc.SETTING_KEY, "not a dictionary")
    assert uc.note() == ""


# ---------------------------------------------------------------------------
# Off means off
# ---------------------------------------------------------------------------

def test_the_environment_switch_stops_the_note(monkeypatch):
    monkeypatch.setenv(uc.SKIP_ENV, "1")
    cache("0.7.3")
    assert uc.note() == ""


def test_the_environment_switch_stops_the_request(monkeypatch):
    """Hiding the note while still calling PyPI would be the worst of both."""
    monkeypatch.setenv(uc.SKIP_ENV, "1")
    calls = fake_pypi(monkeypatch)
    assert uc.refresh_in_background() is None
    assert calls == []


def test_the_saved_setting_stops_the_request(monkeypatch):
    cache("0.7.3", off=True)
    calls = fake_pypi(monkeypatch)
    assert uc.refresh_in_background() is None
    assert calls == []
    assert uc.note() == ""


def test_switching_it_off_is_a_setting_rather_than_a_deletion(monkeypatch):
    """`off` is stored, not `on`, so a settings.json written by an older Taters
    -- which has no key here at all -- reads as on."""
    settings_mod.save_setting(uc.SETTING_KEY, {"latest": "0.7.3"})
    assert uc.note() == "newer version available: v0.7.3"


# ---------------------------------------------------------------------------
# The refresh
# ---------------------------------------------------------------------------

def test_the_refresh_writes_the_answer_for_next_time(monkeypatch):
    calls = fake_pypi(monkeypatch, "0.9.0")
    assert uc.note() == "", "nothing cached yet"
    uc.refresh_in_background().join(10)
    assert calls, "the refresh never asked"
    assert uc.note() == "newer version available: v0.9.0"


def test_the_refresh_runs_where_it_cannot_hold_up_the_menu(monkeypatch):
    """A daemon thread: quitting Taters two seconds in must not wait on a
    socket nobody is reading any more."""
    fake_pypi(monkeypatch)
    thread = uc.refresh_in_background()
    assert thread.daemon
    thread.join(10)


def test_a_fresh_answer_is_not_asked_for_again(monkeypatch):
    """Twenty launches in an afternoon is one request, not twenty."""
    cache("0.7.3")
    calls = fake_pypi(monkeypatch)
    uc.refresh_in_background().join(10)
    assert calls == []


def test_a_day_old_answer_is_asked_for_again(monkeypatch):
    cache("0.7.3", at=time.time() - uc.MAX_AGE_SECONDS - 60)
    calls = fake_pypi(monkeypatch, "0.9.0")
    uc.refresh_in_background().join(10)
    assert calls, "a stale answer was kept forever"
    assert uc.note() == "newer version available: v0.9.0"


def test_an_unchanged_version_still_resets_the_clock(monkeypatch):
    """Without stamping the time on a no-news answer, a quiet month is one
    request per launch rather than one per day."""
    cache("0.7.2", at=time.time() - uc.MAX_AGE_SECONDS - 60)
    fake_pypi(monkeypatch, "0.7.2")
    uc.refresh_in_background().join(10)

    calls = fake_pypi(monkeypatch, "0.7.2")
    uc.refresh_in_background().join(10)
    assert calls == []


def test_switching_it_off_mid_request_is_not_undone_by_the_answer(monkeypatch):
    """The thread sits on a socket for up to five seconds. That is ample time
    to open Settings and switch this off, and the answer arriving afterwards
    must not write the old snapshot -- and the `off` with it -- back."""
    def urlopen(request, timeout=None):
        cache("0.7.3", off=True)   # the user, while the request is in flight
        body = json.dumps({"info": {"version": "0.9.0"}}).encode("utf-8")

        class Response(BytesIO):
            def __enter__(self): return self
            def __exit__(self, *exc): return False

        return Response(body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    uc.refresh_in_background().join(10)
    assert settings_mod.load_settings()[uc.SETTING_KEY]["off"] is True


def test_a_network_failure_is_swallowed_whole(monkeypatch):
    """There is nothing here worth interrupting anybody's run for."""
    fake_pypi(monkeypatch, boom=OSError("no route to host"))
    uc.refresh_in_background().join(10)
    assert uc.note() == ""


def test_nonsense_from_pypi_is_not_cached(monkeypatch):
    fake_pypi(monkeypatch, "not-a-version")
    uc.refresh_in_background().join(10)
    assert uc.note() == ""


# ---------------------------------------------------------------------------
# Where it shows up
# ---------------------------------------------------------------------------

def test_the_note_is_printed_above_the_menu(monkeypatch, tmp_path):
    from taters.ui import hub
    from taters.ui.prompts import ScriptedPrompter

    monkeypatch.setattr(uc, "note", lambda: "newer version available: v9.9.9")
    monkeypatch.setattr(uc, "refresh_in_background", lambda: None)

    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=tmp_path)
    printed = [i for i, line in enumerate(p.output) if "v9.9.9" in line]
    menu = [i for i, ask in enumerate(p.asked) if ask[1] == "What would you like to do?"]
    assert printed and menu, (p.output, p.asked)


def test_no_line_is_printed_when_there_is_nothing_to_report(monkeypatch, tmp_path):
    """An up-to-date install gets a menu with no extra furniture on it."""
    from taters.ui import hub
    from taters.ui.prompts import ScriptedPrompter

    monkeypatch.setattr(uc, "note", lambda: "")
    monkeypatch.setattr(uc, "refresh_in_background", lambda: None)

    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=tmp_path)
    assert not [line for line in p.output if "newer version" in line]


def test_the_switch_is_offered_under_settings():
    from taters.ui.tasks import settings as settings_task

    ids = [t.id for t in settings_task.entries()]
    assert "update_check" in ids
    assert ids.index("update_check") < ids.index("hashbrowns"), (
        "the treat still has to sit last", ids)


def test_turning_it_off_from_the_menu_saves_the_setting(tmp_path):
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, update_check as screen

    cache("0.7.3")
    p = ScriptedPrompter(["off", "back"])
    screen.TASK.run(TaskContext(prompter=p, cwd=tmp_path))

    assert settings_mod.load_settings()[uc.SETTING_KEY]["off"] is True
    assert uc.note() == ""


def test_a_mangled_setting_can_still_be_fixed_from_the_screen(tmp_path):
    """settings.json is a plain file somebody can open in an editor. The screen
    that fixes a mangled setting must not be the one that falls over on it."""
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, update_check as screen

    settings_mod.save_setting(uc.SETTING_KEY, "not a dictionary")
    p = ScriptedPrompter(["off", "back"])
    screen.TASK.run(TaskContext(prompter=p, cwd=tmp_path))
    assert settings_mod.load_settings()[uc.SETTING_KEY]["off"] is True


def test_turning_it_back_on_keeps_the_answer_already_cached(tmp_path):
    """Switching off and on again should not cost a launch with no note on it."""
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, update_check as screen

    cache("0.7.3", off=True)
    p = ScriptedPrompter(["on", "back"])
    screen.TASK.run(TaskContext(prompter=p, cwd=tmp_path))
    assert uc.note() == "newer version available: v0.7.3"
