"""
Tests for the task hub and the tasks behind it.

The hub replaced a single hard-coded flow that began "Where is your data?".
That question assumed an answer nobody had given — that you were here to
extract features — and two of the three things you can now do never touch a
data folder at all.

The tests that matter most here are the destructive ones. `manage` is the only
part of the wizard that deletes anything, and it sits one keystroke away from
"Export" in a menu aimed at people who do not read carefully. So: built-ins are
unreachable from every destructive action, and delete needs the name typed out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from taters.ui import hub
from taters.ui.browse import _TYPE as BROWSE_TYPE
from taters.ui.prompts import Cancelled, GoBack, ScriptedPrompter
from taters.ui.tasks import Task, TaskContext, all_tasks, manage, run_saved
from wizard_helpers import browse_to


@pytest.fixture()
def workspace(tmp_path):
    """A working folder with one pipeline of the user's own."""
    pipelines = tmp_path / "pipelines"
    pipelines.mkdir()
    (pipelines / "my_run.yaml").write_text(yaml.safe_dump({
        "meta": {"id": "my_run", "title": "My run", "summary": "A test pipeline",
                 "variables": {"overwrite_existing": {"default": False, "desc": "Redo work"}}},
        "vars": {"overwrite_existing": False},
        "steps": [{"scope": "global", "call": "potato.text.analyze_readability",
                   "save_as": "r", "with": {"txt_dir": "."}}],
    }, sort_keys=False), encoding="utf-8")
    return tmp_path


def ctx_for(prompter, cwd) -> TaskContext:
    return TaskContext(prompter=prompter, cwd=Path(cwd))


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------

def test_the_first_question_is_about_intent_not_about_files(workspace):
    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=workspace)
    assert p.asked[0] == ("select", "What would you like to do?")


def test_every_registered_task_is_offered(workspace):
    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=workspace)

    offered = {c.value for c in p.offered_choices("What would you like to do?")}
    assert {t.id for t in all_tasks()} <= offered
    assert "quit" in offered


def test_a_task_with_nothing_to_work_on_is_grayed_out_not_hidden(tmp_path):
    """
    Hiding "Manage saved pipelines" until one exists would leave a new user
    unable to discover that saved pipelines are a thing. Showing it with a
    reason teaches them.
    """
    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=tmp_path)

    choices = {c.value: c for c in p.offered_choices("What would you like to do?")}
    # managing pipelines lives under settings now; the front page keeps the
    # doing-verbs, with the data row first.
    assert set(choices) == {"data", "extract", "extract_analyze", "train",
                            "run_saved", "settings", "quit"}
    assert choices["extract"].disabled == ""
    assert choices["data"].disabled == ""


def test_hashbrowns_sits_last_under_settings(workspace):
    """
    A treat, not a task: it goes where nobody trips over it on the way to
    work. That used to be the bottom of the front page, directly above Quit;
    it is now the bottom of Settings, which is further out of the way still
    -- and keeps the front page to the verbs that do something.
    """
    from taters.ui.tasks import settings as settings_task

    assert "hashbrowns" not in {t.id for t in all_tasks()}
    rows = settings_task.entries()
    assert rows[-1].id == "hashbrowns", [t.id for t in rows]
    assert (rows[-1].label, rows[-1].help) == (
        "Hashbrowns",
        "When you need something to sizzle your mind... Press [esc] when you need to rejoin reality...")

    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=workspace)
    values = [c.value for c in p.offered_choices("What would you like to do?")]
    assert values[-1] == "quit" and values[-2] == "settings"


def test_choosing_hashbrowns_runs_the_demo_and_comes_back_to_the_menu(workspace, monkeypatch):
    """The row hands the terminal to the demo and takes it back: nothing ran
    that could succeed or fail, so the menu simply reappears."""
    from taters.ui import hashbrowns as demo

    calls = []
    monkeypatch.setattr(demo, "run", lambda **kw: calls.append(kw))
    # it lives under settings now, so getting there is two rows and coming
    # back out is one
    p = ScriptedPrompter(["settings", "hashbrowns", "back", "quit"])
    ok = hub.run_hub(p, cwd=workspace)
    assert calls == [{}], "the demo runs once, with its own defaults"
    assert [q for _, q in p.asked] == [
        "What would you like to do?", "Settings and tools",
        # the settings menu reappears after the demo hands the terminal back,
        # and backing out of it lands on the finish screen like any other task
        "Settings and tools", "Main menu"]
    assert ok is not False


def test_choosing_a_task_runs_it(workspace, monkeypatch):
    seen = []
    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="manage", label="Manage", help="", run=lambda ctx: seen.append(ctx.cwd))
    ])
    p = ScriptedPrompter(["manage", "quit"])
    hub.run_hub(p, cwd=workspace)
    assert seen == [workspace]


def test_backing_out_of_a_task_returns_to_the_menu(workspace, monkeypatch):
    """
    A mistyped path should cost you that task, not the whole session.
    """
    def explode(ctx):
        raise Cancelled()

    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="manage", label="Manage", help="", run=explode)
    ])
    p = ScriptedPrompter(["manage", "quit"])
    hub.run_hub(p, cwd=workspace)

    assert [q for _, q in p.asked] == ["What would you like to do?", "Main menu"]
    assert "Backed out" in p.text_output


def test_a_failed_run_survives_the_menu_as_a_nonzero_result(workspace, monkeypatch):
    """
    The hub can do several things in a session. A script watching the exit code
    still has to learn that one of them failed.
    """
    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="t", label="T", help="", run=lambda ctx: False)
    ])
    p = ScriptedPrompter(["t", "quit"])
    assert hub.run_hub(p, cwd=workspace) is False


def test_a_clean_session_reports_success(workspace):
    assert hub.run_hub(ScriptedPrompter(["quit"]), cwd=workspace) is True


# ---------------------------------------------------------------------------
# Running a saved pipeline
# ---------------------------------------------------------------------------

def test_the_shipped_pipelines_are_offered(workspace):
    """
    `conversation_video` and `single_speaker_media` encode a lot of hard-won
    wiring, and until the hub existed the only way to reach either was to know
    it existed and type a CLI flag.
    """
    from taters.pipelines.run_pipeline import available_presets, is_builtin_preset

    builtin = next(path for path, _ in available_presets() if is_builtin_preset(path))
    p = ScriptedPrompter([str(builtin)])
    chosen, _meta = run_saved.choose_preset(p)

    assert chosen == builtin
    labels = " | ".join(c.label for c in p.offered_choices("Which pipeline?"))
    assert "Conversation video" in labels
    assert "Single-speaker media" in labels
    assert "(built in)" in labels


def test_a_pipeline_that_reads_its_own_input_is_not_asked_for_a_folder(workspace):
    """
    A text pipeline is GLOBAL-only; asking it for a --root_dir would be asking
    for something the runner will never use.
    """
    preset = yaml.safe_load((workspace / "pipelines" / "my_run.yaml").read_text())
    assert run_saved._needs_inputs(preset) is False
    assert run_saved._needs_inputs(
        {"steps": [{"scope": "item", "call": "x"}]}
    ) is True


def test_a_boolean_setting_does_not_come_back_as_a_string():
    """
    `overwrite_existing` answered "false" and stored as the *string* "false"
    is worse than useless: every truthiness test downstream then passes.
    """
    assert run_saved._coerce("false", False) is False
    assert run_saved._coerce("yes", False) is True
    assert run_saved._coerce("4", 1) == 4
    assert run_saved._coerce("0.5", 1.0) == 0.5
    assert run_saved._coerce("hello", "world") == "hello"


def test_unparseable_numbers_are_left_alone_rather_than_crashing():
    assert run_saved._coerce("lots", 1) == "lots"


def test_a_setting_is_explained_where_it_is_actually_answered(workspace):
    """
    The table of settings is drawn above "Change any of these?", and answering
    that question wipes the screen. So the description has to be repeated at
    the prompt itself, or the only thing on screen when someone types a value
    is a bare parameter name.
    """
    preset = yaml.safe_load((workspace / "pipelines" / "my_run.yaml").read_text())
    p = ScriptedPrompter([True, False])
    run_saved.ask_var_overrides(p, preset)

    assert any("Redo work" in line for line in p.output)


def test_a_yes_no_setting_is_not_asked_as_free_text(workspace):
    """
    `overwrite_existing` offered as a text box pre-filled with "False" invites
    the one edit that does not work: clearing it means "leave it alone".
    """
    preset = yaml.safe_load((workspace / "pipelines" / "my_run.yaml").read_text())
    p = ScriptedPrompter([True, True])
    assert run_saved.ask_var_overrides(p, preset) == {"overwrite_existing": True}
    assert ("confirm", "overwrite_existing?") in p.asked


def test_changing_one_setting_does_not_delete_all_the_others(workspace, monkeypatch):
    """
    A regression with teeth. `run_preset` takes `vars_ctx` as the *whole*
    variable context, not a patch over the preset's own `vars` block. Handing
    it only the settings the user changed therefore deleted every one they
    left alone, and the run died on the first template that mentioned an
    untouched variable:

        Templating failed (likely missing artifact):
            "Variable 'overwrite_existing' not found"

    Nothing about the composed preset was wrong -- the file on disk still
    declared all six variables -- which is exactly what made it hard to see.
    """
    from taters.pipelines import run_pipeline
    from taters.ui.prompts import QuitRequested

    (workspace / "pipelines" / "two_vars.yaml").write_text(yaml.safe_dump({
        "meta": {
            "id": "two_vars", "title": "Two vars", "summary": "",
            "variables": {
                "whisper_model": {"default": "base.en", "desc": "Model size"},
                "overwrite_existing": {"default": False, "desc": "Redo work"},
            },
        },
        "vars": {"whisper_model": "base.en", "overwrite_existing": False},
        "steps": [{"scope": "global", "call": "potato.text.analyze_readability",
                   "save_as": "r",
                   "with": {"txt_dir": ".",
                            "overwrite_existing": "{{var:overwrite_existing}}"}}],
    }, sort_keys=False), encoding="utf-8")

    seen = {}

    def fake_run_preset(preset, **kwargs):
        seen.update(kwargs)
        return {"items": [], "errors": []}

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(run_pipeline, "run_preset", fake_run_preset)

    path = str(workspace / "pipelines" / "two_vars.yaml")
    p = ScriptedPrompter([
        path,           # which pipeline
        True,           # change any of these?
        "tiny",         # whisper_model
        False,          # overwrite_existing -- left alone
        True,           # run it now?
        "quit",         # what now?
    ])
    with pytest.raises(QuitRequested):
        run_saved._run(ctx_for(p, workspace))

    assert seen["vars_ctx"] == {"whisper_model": "tiny", "overwrite_existing": False}


# ---------------------------------------------------------------------------
# Managing pipelines -- the destructive half
# ---------------------------------------------------------------------------

def test_built_in_pipelines_are_never_offered_for_a_destructive_action(workspace):
    """
    The shipped presets live inside the installed package. Deleting one would
    damage the install in a way that survives until the next `pip install`, and
    the person most likely to try it is the one least likely to know that.
    """
    p = ScriptedPrompter([])
    ctx = ctx_for(p, workspace)
    own = {path.name for path in ctx.own_pipelines()}

    assert own == {"my_run.yaml"}
    assert not any("presets" in str(path) for path in ctx.own_pipelines())


def test_deleting_requires_the_name_typed_out(workspace):
    p = ScriptedPrompter([str(workspace / "pipelines" / "my_run.yaml"), "not the name"])
    ctx = ctx_for(p, workspace)
    manage._delete(ctx, ctx.own_pipelines())

    assert (workspace / "pipelines" / "my_run.yaml").exists()
    assert "Nothing was deleted" in p.text_output


def test_deleting_with_the_right_name_removes_the_file(workspace):
    p = ScriptedPrompter([str(workspace / "pipelines" / "my_run.yaml"), "My run"])
    ctx = ctx_for(p, workspace)
    manage._delete(ctx, ctx.own_pipelines())

    assert not (workspace / "pipelines" / "my_run.yaml").exists()


def test_renaming_moves_title_id_and_filename_together(workspace):
    """
    Renaming only the file leaves `--preset <id>` pointing at the old name;
    renaming only the title leaves the menu disagreeing with the folder.
    """
    p = ScriptedPrompter([str(workspace / "pipelines" / "my_run.yaml"), "Pilot study"])
    ctx = ctx_for(p, workspace)
    manage._rename(ctx, ctx.own_pipelines())

    target = workspace / "pipelines" / "pilot_study.yaml"
    assert target.exists()
    assert not (workspace / "pipelines" / "my_run.yaml").exists()

    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["meta"]["title"] == "Pilot study"
    assert data["meta"]["id"] == "pilot_study"


def test_importing_something_that_is_not_a_pipeline_is_refused(workspace, tmp_path):
    junk = tmp_path / "notes.yaml"
    junk.write_text("just: a mapping\n", encoding="utf-8")
    assert "no 'steps:' list" in manage.looks_like_a_preset(junk)

    broken = tmp_path / "broken.yaml"
    broken.write_text("steps: [\n", encoding="utf-8")
    assert "not readable as YAML" in manage.looks_like_a_preset(broken)

    good = workspace / "pipelines" / "my_run.yaml"
    assert manage.looks_like_a_preset(good) == ""


def test_importing_a_clashing_name_does_not_overwrite(workspace, tmp_path):
    incoming = tmp_path / "my_run.yaml"
    incoming.write_text((workspace / "pipelines" / "my_run.yaml").read_text(), encoding="utf-8")

    (workspace / "my_run").mkdir()          # so the name is already taken

    p = ScriptedPrompter([BROWSE_TYPE, str(incoming)])
    ctx = ctx_for(p, workspace)
    manage._import(ctx, ctx.own_pipelines())

    # it gets imported into a folder of its own, sidestepping the clash rather
    # than silently overwriting whatever was already there.
    assert (workspace / "my_run-2" / "my_run-2.yaml").exists()
    assert "renamed from" in p.text_output


def test_exporting_does_not_move_the_original(workspace, tmp_path):
    destination = tmp_path / "outbox"
    destination.mkdir()

    # we browse to the destination rather than typing it: "Type a path
    # instead", then the path.
    p = ScriptedPrompter([
        str(workspace / "pipelines" / "my_run.yaml"),
        BROWSE_TYPE, str(destination),
    ])
    ctx = ctx_for(p, workspace)
    manage._export(ctx, ctx.own_pipelines())

    assert (destination / "my_run.yaml").exists()
    assert (workspace / "pipelines" / "my_run.yaml").exists()


def test_duplicating_a_built_in_leaves_the_original_untouched(workspace):
    from taters.pipelines.run_pipeline import available_presets, is_builtin_preset

    builtin = next(p for p, _ in available_presets() if is_builtin_preset(p))
    before = builtin.read_bytes()

    p = ScriptedPrompter([str(builtin), "My copy"])
    ctx = ctx_for(p, workspace)
    manage._duplicate(ctx, ctx.own_pipelines())

    assert builtin.read_bytes() == before
    assert (workspace / "my_copy" / "my_copy.yaml").exists()


# ---------------------------------------------------------------------------
# One folder per pipeline
# ---------------------------------------------------------------------------

def test_a_pipeline_in_its_own_folder_is_found(workspace):
    """
    The wizard now writes `readability_test/readability_test.yaml`. Both that
    and the older flat `./pipelines/` layout have to be listed -- the old ones
    did not stop being pipelines.
    """
    folder = workspace / "essay_run"
    folder.mkdir()
    (folder / "essay_run.yaml").write_text(
        yaml.safe_dump({"meta": {"id": "essay_run", "title": "Essay run"},
                        "steps": [{"scope": "global", "call": "x"}]}),
        encoding="utf-8")

    ctx = ctx_for(ScriptedPrompter([]), workspace)
    found = {p.name for p in ctx.own_pipelines()}
    assert found == {"my_run.yaml", "essay_run.yaml"}


def test_a_stray_yaml_in_a_folder_is_not_mistaken_for_a_pipeline(workspace):
    """
    The rule is deliberately narrow -- a folder holding a YAML of the *same
    name*. Anything looser sweeps up every config.yaml in the working folder.
    """
    folder = workspace / "notes"
    folder.mkdir()
    (folder / "config.yaml").write_text("steps: []\n", encoding="utf-8")

    ctx = ctx_for(ScriptedPrompter([]), workspace)
    assert not any("config" in p.name for p in ctx.own_pipelines())


def test_deleting_a_pipeline_folder_warns_about_the_results_inside(workspace):
    """
    "Delete a pipeline" sounds like it removes a small config file. When the
    pipeline owns a folder it can remove hours of extraction, so the warning
    has to name what goes with it.
    """
    folder = workspace / "essay_run"
    (folder / "features").mkdir(parents=True)
    (folder / "features" / "readability.csv").write_text("a,b\n", encoding="utf-8")
    (folder / "essay_run.yaml").write_text(
        yaml.safe_dump({"meta": {"id": "essay_run", "title": "Essay run"},
                        "steps": [{"scope": "global", "call": "x"}]}),
        encoding="utf-8")

    p = ScriptedPrompter([str(folder / "essay_run.yaml"), "folder", "Essay run"])
    ctx = ctx_for(p, workspace)
    manage._delete(ctx, ctx.own_pipelines())

    assert "WARNING" in p.text_output
    assert "features" in p.text_output
    assert not folder.exists()


def test_renaming_a_pipeline_moves_its_folder_too(workspace):
    folder = workspace / "essay_run"
    (folder / "features").mkdir(parents=True)
    (folder / "essay_run.yaml").write_text(
        yaml.safe_dump({"meta": {"id": "essay_run", "title": "Essay run"},
                        "steps": [{"scope": "global", "call": "x"}]}),
        encoding="utf-8")

    p = ScriptedPrompter([str(folder / "essay_run.yaml"), "Pilot study"])
    ctx = ctx_for(p, workspace)
    manage._rename(ctx, ctx.own_pipelines())

    moved = workspace / "pilot_study"
    assert (moved / "pilot_study.yaml").exists()
    assert (moved / "features").is_dir(), "the run's outputs moved with it"
    assert not folder.exists()


def test_the_banner_is_drawn_once_on_the_opening_screen(workspace, monkeypatch):
    """
    The renderer redraws the header on every screen, so the hub must not also
    print it -- that put two banners on the opening screen.
    """
    from taters.ui.live import LivePrompter

    drawn = []

    class Probe(LivePrompter):
        def __init__(self):
            self._stages = []
            self._header = ""
            self._screen_notes = []
            self._title = "Taters"

        def clear(self):
            pass

        def note(self, text, *, style="", wrap=True):
            drawn.append(text)

        def select(self, question, choices, *, default=None):
            drawn.append("<menu>")
            return "quit"

    hub.run_hub(Probe(), cwd=workspace)
    banners = [i for i, line in enumerate(drawn) if "Takes All Things" in line]
    menu = drawn.index("<menu>")
    assert not [i for i in banners if i < menu], (
        "the hub printed the banner on the opening screen even though the "
        "renderer owns it"
    )
    # the sign-off is the one exception: no question follows it, so nothing
    # will ever repaint the header -- the hub has to print the banner itself.
    assert len(banners) == 1


def test_a_renderer_without_a_header_still_gets_a_banner(workspace):
    """The plain renderer has no screen to redraw, so the hub prints it."""
    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=workspace)
    # once on the opening screen, once above the farewell.
    assert sum("Takes All Things" in line for line in p.output) == 2


def test_the_banner_carries_the_version_beside_the_name():
    """
    Beside it, not below it. The version used to sit on a row of its own, which
    spent a line of a five-line header on a number -- and read as a separate
    fact rather than as part of what the program calls itself.
    """
    from taters.ui.hub import banner, version, version_line

    drawn = banner()
    assert version() in drawn
    assert any("TATERS" in line and version_line() in line
               for line in drawn.splitlines())


def test_the_banner_still_names_itself_when_there_is_no_metadata(monkeypatch):
    """Running from an uninstalled source tree should not blank the header."""
    from taters.ui import hub as hub_mod

    monkeypatch.setattr(hub_mod, "version", lambda: "")
    assert "running from source" in hub_mod.banner()
    assert "TATERS" in hub_mod.banner()


def test_the_banner_encourages():
    from taters.ui.hub import banner

    from taters.ui.hub import _ENCOURAGEMENT, _MOTTOS

    assert _ENCOURAGEMENT in _MOTTOS
    assert _ENCOURAGEMENT in banner()


def test_the_encouragement_is_italic():
    """Markup, so it reads as an aside rather than as another fact."""
    import io
    import re

    from rich.console import Console

    from taters.ui.hub import banner

    buf = io.StringIO()
    Console(file=buf, width=80, force_terminal=True).print(banner())
    raw = buf.getvalue()

    # we look for `3` as one of the parameters of an SGR escape rather than the
    # bare `\x1b[3m`: the encouragement is italic *and* colored, and rich emits
    # those as a single escape (`\x1b[3;38;5;113m`).
    italics = [seq for seq in re.findall(r"\x1b\[([0-9;]*)m", raw)
               if "3" in seq.split(";")]
    assert italics, f"no italic escape was emitted: {raw!r}"
    from taters.ui.hub import _ENCOURAGEMENT
    assert _ENCOURAGEMENT in re.sub(r"\x1b\[[0-9;]*m", "", raw)


def test_leaving_is_not_curt():
    from taters.ui.hub import FAREWELL

    assert FAREWELL == "So long, and thanks for all the spuds!"


@pytest.mark.parametrize("fake_version", ["0.2.1", "10.20.30rc1", ""])
def test_the_banner_box_stays_square_whatever_the_version(monkeypatch, fake_version):
    """
    A version string of a different length pushing the right border out of
    line is a small thing that makes a whole application look broken.

    Measured on the *rendered* width, not the raw string: the markup that makes
    one line italic occupies no columns on screen but plenty in the text, so
    comparing raw lengths would fail on a box that is actually straight.
    """
    import io
    import re

    from rich.console import Console

    from taters.ui import hub as hub_mod

    monkeypatch.setattr(hub_mod, "version", lambda: fake_version)

    buf = io.StringIO()
    Console(file=buf, width=200, force_terminal=True).print(hub_mod.banner())
    plain = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())

    widths = {len(line) for line in plain.splitlines() if line.strip()}
    assert len(widths) == 1, plain


def test_returning_to_the_top_says_so(workspace):
    """
    "Anything else?" reads as a follow-up to whatever just happened. Coming back
    from a submenu that is neither accurate nor reassuring — the point is that
    you are back at the top.
    """
    p = ScriptedPrompter(["settings", "back", "quit"])
    hub.run_hub(p, cwd=workspace)

    questions = [q for _, q in p.asked]
    assert questions[0] == "What would you like to do?"
    assert "Main menu" in questions
    assert "Anything else?" not in questions


# ---------------------------------------------------------------------------
# Esc at the front door
#
# everywhere else Esc means "undo the last question" -- that's the whole point
# of keeping `GoBack` separate from `Cancelled`, so a typo costs you one answer
# rather than the session. the main menu is the one place with no question
# above it, and there the same key was getting swallowed and just redrawing the
# menu: a keypress that visibly does nothing, which looks like we ignored it.
# ---------------------------------------------------------------------------

class _EscapesOnce(ScriptedPrompter):
    """A prompter that presses Esc at its first question, then behaves."""

    escaped: bool = False

    def select(self, question, choices, *, default=None, transient=False):
        if not self.escaped:
            self.escaped = True
            self.asked.append(("select", question))
            raise GoBack()
        return super().select(question, choices, default=default,
                              transient=transient)


def test_escape_at_the_main_menu_leaves(workspace):
    p = _EscapesOnce([])
    result = hub.run_hub(p, cwd=workspace)

    assert result is True
    assert hub.FAREWELL in p.text_output
    assert len(p.asked) == 1, "the menu was redrawn instead of closing"


def test_escape_inside_a_task_still_only_backs_out_of_that_task(workspace, monkeypatch):
    """
    The distinction that has to survive. Esc is "undo the last question"
    everywhere it has a question to undo, and only the front door treats it as
    the way out.
    """
    def explode(ctx):
        raise GoBack()

    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="manage", label="Manage", help="", run=explode)
    ])
    p = ScriptedPrompter(["manage", "quit"])
    hub.run_hub(p, cwd=workspace)

    assert [q for _, q in p.asked] == ["What would you like to do?", "Main menu"]
    assert "Backed out" in p.text_output


def test_ctrl_c_at_the_main_menu_still_leaves(workspace):
    """Unchanged -- and now the same as Esc, which is what people expect."""
    class _Interrupts(ScriptedPrompter):
        def select(self, question, choices, *, default=None, transient=False):
            self.asked.append(("select", question))
            raise Cancelled()

    p = _Interrupts([])
    assert hub.run_hub(p, cwd=workspace) is True
    assert hub.FAREWELL in p.text_output


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------

def test_the_banner_has_a_potato_in_it():
    """
    Not decoration for its own sake. This is a tool for people who do not write
    code, and the first screen sets whether it feels approachable or like
    something that will blame them for holding it wrong.
    """
    import re

    from taters.ui.hub import _ART_WIDTH, _art, banner

    # we don't pin the exact glyphs -- anyone should be free to redraw the
    # potato, and the plain build has to redraw it anyway because the quadrant
    # blocks that round the ends are missing from every stock Windows font. what
    # has to survive is the silhouette: the body runs the full width and the
    # ends are inset, and that difference is the whole reason it reads as an
    # oval rather than as a brick.
    for fancy in (True, False):
        top, body, bottom = (re.sub(r"\[/?[^\]]*\]", "", row) for row in _art(fancy))
        assert len(body.strip()) == _ART_WIDTH, "the body should run the full width"
        assert len(top.strip()) < len(body.strip()), "the top is not inset"
        assert len(bottom.strip()) < len(body.strip()), "the bottom is not inset"
    assert "•" in banner(), "the potato has no eyes"


def test_the_art_sits_beside_the_text_rather_than_above_it():
    import re

    from taters.ui.hub import banner

    for line in banner().splitlines():
        plain = re.sub(r"\[/?[^\]]*\]", "", line)
        if "TATERS" in plain:
            assert "▄" in plain, "the name is not on the same row as the art"
            return
    raise AssertionError("no row carried the name")


def test_the_potato_and_the_text_are_the_same_height():
    """
    Three rows each. An earlier potato was four rows tall with a sprout on top,
    which left a blank row under the text and made the box look like it had
    been padded out.
    """
    import re

    from taters.ui import hub as hub_mod
    from taters.ui.hub import banner

    plain = [re.sub(r"\[/?[^\]]*\]", "", line) for line in banner().splitlines()]
    rows = [line for line in plain if line.startswith("  │")]

    assert len(rows) == len(hub_mod._SPUD_ART) == 3
    assert all(r.strip() for r in rows), "a row of the banner is empty"


def test_the_banner_has_no_sprout():
    """
    The green drew the eye before the potato did, and `\\|/` above a square
    body read as a shrub rather than a vegetable.
    """
    from taters.ui import hub as hub_mod
    from taters.ui.hub import banner

    assert "\\|/" not in banner()
    assert not any("|" in row for row in hub_mod._SPUD_ART)


def test_the_banner_is_colored():
    """
    Style tags, not escape codes: `banner()` returns markup for whichever
    renderer is in front, and one of them paints it into a live header rather
    than printing it.
    """
    from taters.ui.hub import banner

    drawn = banner()
    assert drawn.count("[#") >= 4, "the header is monochrome"


def test_the_border_is_not_a_compliance_report():
    """A light frame. The heavy double rule was the wrong register."""
    from taters.ui import glyphs
    from taters.ui.hub import banner

    drawn = banner()
    # the corners come from glyphs rather than being spelled out: the arcs are
    # missing from Courier New and Lucida Console, so the plain build squares
    # them off. light either way is the point.
    assert glyphs.ARC_TL in drawn and glyphs.ARC_BL in drawn
    assert "╔" not in drawn and "═" not in drawn


@pytest.mark.parametrize("fake_version", ["0.2.1", "10.20.30rc1", ""])
def test_the_art_does_not_break_the_box(monkeypatch, fake_version):
    """
    The half-block characters are single-width, but padding is computed after
    stripping style tags -- and a tag counts plenty of characters and no
    columns. Getting that wrong bends the right-hand border on exactly the
    lines that carry color, which is all of them.
    """
    import io
    import re

    from rich.console import Console

    from taters.ui import hub as hub_mod

    monkeypatch.setattr(hub_mod, "version", lambda: fake_version)

    buf = io.StringIO()
    Console(file=buf, width=200, force_terminal=True,
            highlight=False).print(hub_mod.banner())
    plain = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())

    widths = {len(line) for line in plain.splitlines() if line.strip()}
    assert len(widths) == 1, plain


def test_the_frame_is_never_the_brightest_thing():
    """
    The rule the palette is built on: brightness carries importance, and the
    frame is the least informative element on the screen. It drifts through a
    spectrum now, so this checks the whole cycle rather than one value -- the
    color is free to move, the muting is not.
    """
    from taters.ui import hub as hub_mod

    for at in range(0, int(hub_mod._BORDER_CYCLE_SECONDS), 5):
        hex_color = hub_mod.border_style(at)
        channels = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
        assert max(channels) <= 160, f"the frame is too bright at t={at}: {hex_color}"


def test_the_frame_drifts_rather_than_sitting_still():
    """A session that lasts an hour should not look like a screenshot of itself."""
    from taters.ui import hub as hub_mod

    period = hub_mod._BORDER_CYCLE_SECONDS
    seen = {hub_mod.border_style(at) for at in range(0, int(period), 10)}
    assert len(seen) > 6, f"the frame barely moves across a cycle: {sorted(seen)}"

    # and it comes back round rather than running off somewhere.
    assert hub_mod.border_style(0) == hub_mod.border_style(period)


def test_the_drift_survives_a_256_color_terminal():
    """
    The bug the first version had. At a low enough value every hue quantizes to
    the *same* palette slot, so the drift looked right in truecolor and did
    nothing at all on anything older -- which is most terminals.
    """
    from rich.color import Color

    from taters.ui import hub as hub_mod

    period = hub_mod._BORDER_CYCLE_SECONDS
    slots = {Color.parse(hub_mod.border_style(at)).downgrade(2).number
             for at in range(0, int(period), 15)}
    assert len(slots) >= 4, f"only {len(slots)} distinct 256-color slots: {slots}"


def test_the_cycle_is_slow_enough_to_be_calm():
    """Movement you notice between screens, not a flicker you notice at all."""
    from taters.ui import hub as hub_mod

    assert hub_mod._BORDER_CYCLE_SECONDS >= 60


def test_the_name_is_the_brightest_thing():
    from taters.ui import hub as hub_mod

    assert "bold" in hub_mod._NAME
    assert not any("bold" in style for style in
                   (hub_mod._TAG_STYLE, hub_mod._SPUD, hub_mod.border_style(0)))


def test_the_potato_is_shaded_rather_than_flat():
    """
    Three shades running left to right, lit to shadowed. A single flat color
    over block characters reads as a brick; the gradient is what makes it look
    round, which is most of what makes it look like a potato.
    """
    from taters.ui import hub as hub_mod

    art = "".join(hub_mod._SPUD_ART)
    for shade in (hub_mod._SPUD_LIT, hub_mod._SPUD, hub_mod._SPUD_DIM):
        assert shade in art, f"the potato does not use {shade}"
    assert len({hub_mod._SPUD_LIT, hub_mod._SPUD, hub_mod._SPUD_DIM}) == 3


def test_the_eyes_sit_on_the_solid_row_only():
    """
    The top and bottom rows are half-height blocks, so a dot placed there lands
    in the empty half of the cell and reads as floating off the potato rather
    than sitting in it.
    """
    from taters.ui import hub as hub_mod

    top, middle, bottom = hub_mod._SPUD_ART
    assert "•" not in top and "•" not in bottom, "an eye is on a half-block row"
    assert middle.count("•") == 2


def test_the_outline_bulges_in_the_middle():
    """
    What makes it an oval rather than a rounded rectangle, and the thing that
    was lost when a `>_` screen was fitted into the body: the top and bottom
    rows are inset by a column at each end while the body runs full width. With
    all three the same width it reads as a square with soft corners.
    """
    import re

    from taters.ui import hub as hub_mod

    widths = [len(re.sub(r"\[/?[^\]]*\]", "", row).rstrip())
              for row in hub_mod._SPUD_ART]
    top, middle, bottom = widths
    assert top < middle and bottom < middle, (
        f"the potato is a rectangle: rows are {widths} wide"
    )

def test_the_rail_is_cleared_when_a_task_finishes(workspace, monkeypatch):
    """
    From a real report: finishing the wizard and coming back to the main menu
    left its stages on screen, still describing a pipeline that was no longer
    being built. The rail belongs to the task that raised it.
    """
    def _raises_a_rail(ctx):
        ctx.prompter.stage("source", "Source", status="done")
        ctx.prompter.stage("features", "Features", status="active")
        return True

    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="t", label="T", help="", run=_raises_a_rail)
    ])
    p = ScriptedPrompter(["t", "quit"])
    hub.run_hub(p, cwd=workspace)

    assert p.stages == [], "the task's rail outlived the task"


def test_the_rail_is_cleared_even_when_a_task_backs_out(workspace, monkeypatch):
    """Backing out is the commoner way to leave, and left the same wreckage."""
    def _backs_out(ctx):
        ctx.prompter.stage("source", "Source", status="active")
        raise GoBack()

    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="t", label="T", help="", run=_backs_out)
    ])
    p = ScriptedPrompter(["t", "quit"])
    hub.run_hub(p, cwd=workspace)

    assert p.stages == []


# ---------------------------------------------------------------------------
# Renaming a pipeline that owns its folder (code review issue 3)
# ---------------------------------------------------------------------------


def _folder_pipeline(root, pid, title):
    """A wizard-style pipeline: <root>/<id>/<id>.yaml, the shape run_hub saves."""
    folder = root / pid
    folder.mkdir()
    (folder / f"{pid}.yaml").write_text(yaml.safe_dump({
        "meta": {"id": pid, "title": title, "summary": "s", "variables": {}},
        "vars": {},
        "steps": [{"scope": "global", "call": "potato.text.analyze_readability",
                   "save_as": "r", "with": {"txt_dir": "."}}],
    }, sort_keys=False), encoding="utf-8")
    return folder / f"{pid}.yaml"


def test_renaming_to_a_same_slug_title_moves_nothing(tmp_path):
    """
    From the code review: 'My pipeline' -> 'My Pipeline!' slugifies to the same
    id, and _unique then treated the pipeline's own folder as a collision --
    moving it to my_pipeline-2/ around an unrenamed my_pipeline.yaml, after
    which no menu could see the pipeline at all.
    """
    path = _folder_pipeline(tmp_path, "my_pipeline", "My pipeline")
    p = ScriptedPrompter([str(path), "My Pipeline!"])
    ctx = ctx_for(p, tmp_path)

    manage._rename(ctx, ctx.own_pipelines())

    assert path.exists(), "folder or file moved for a display-only rename"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["meta"]["title"] == "My Pipeline!"
    assert data["meta"]["id"] == "my_pipeline"
    assert [pl.stem for pl in ctx.own_pipelines()] == ["my_pipeline"]


def test_a_real_rename_keeps_folder_file_and_id_in_lockstep(tmp_path):
    path = _folder_pipeline(tmp_path, "my_pipeline", "My pipeline")
    p = ScriptedPrompter([str(path), "Pilot study"])
    ctx = ctx_for(p, tmp_path)

    manage._rename(ctx, ctx.own_pipelines())

    target = tmp_path / "pilot_study" / "pilot_study.yaml"
    assert target.exists()
    assert target.parent.name == target.stem
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["meta"]["id"] == "pilot_study"
    assert not (tmp_path / "my_pipeline").exists()


def test_renaming_onto_a_taken_name_suffixes_all_three_together(tmp_path):
    """
    The suffix _unique adds must reach the folder, the file stem AND meta.id --
    a folder called pilot_study-2 holding pilot_study.yaml is invisible to
    own_pipelines and --list-presets, which is the same vanishing act as the
    same-slug case by another door.
    """
    _folder_pipeline(tmp_path, "pilot_study", "Pilot study")
    path = _folder_pipeline(tmp_path, "my_pipeline", "My pipeline")
    p = ScriptedPrompter([str(path), "Pilot study"])
    ctx = ctx_for(p, tmp_path)

    manage._rename(ctx, ctx.own_pipelines())

    target = tmp_path / "pilot_study-2" / "pilot_study-2.yaml"
    assert target.exists()
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["meta"]["id"] == "pilot_study-2"
    assert sorted(pl.stem for pl in ctx.own_pipelines()) == [
        "pilot_study", "pilot_study-2"]


# ---------------------------------------------------------------------------
# Hand-edited YAML must never take the session down (code review issue 9)
# ---------------------------------------------------------------------------


def test_renaming_a_broken_yaml_explains_instead_of_crashing(tmp_path):
    """
    The listers tolerate unreadable files so they stay visible and fixable --
    which means the actions that then open one meet a stray tab or a truncated
    write, which used to climb as a raw YAMLError and end the session.
    """
    pipelines = tmp_path / "pipelines"
    pipelines.mkdir()
    bad = pipelines / "broken.yaml"
    bad.write_text("meta:\n\ttitle: tabs are illegal in yaml\n",
                   encoding="utf-8")

    p = ScriptedPrompter([str(bad)])
    manage._rename(ctx_for(p, tmp_path), [bad])

    assert any("Cannot read broken.yaml" in line for line in p.output)
    assert bad.exists(), "the broken file must be left alone to fix by hand"


def test_a_null_meta_key_is_survivable(tmp_path):
    """`meta:\\n` with nothing under it parses to None; setdefault then handed
    None back and None['title'] was a session-ending TypeError."""
    pipelines = tmp_path / "pipelines"
    pipelines.mkdir()
    f = pipelines / "nullmeta.yaml"
    f.write_text(
        "meta:\nsteps:\n  - scope: global\n    call: potato.text.analyze_readability\n"
        "    save_as: r\n    with: {txt_dir: '.'}\n",
        encoding="utf-8")

    p = ScriptedPrompter([str(f), "Given a name"])
    manage._rename(ctx_for(p, tmp_path), [f])

    data = yaml.safe_load((pipelines / "given_a_name.yaml").read_text(encoding="utf-8"))
    assert data["meta"]["title"] == "Given a name"


def test_making_a_copy_of_a_broken_yaml_explains_instead_of_crashing(tmp_path, monkeypatch):
    pipelines = tmp_path / "pipelines"
    pipelines.mkdir()
    bad = pipelines / "broken.yaml"
    bad.write_text("steps: [\n", encoding="utf-8")   # like a write that got cut off
    # _duplicate imports this lazily, so we have to patch it at its source.
    monkeypatch.setattr("taters.pipelines.run_pipeline.available_presets",
                        lambda root=None: [(bad, {})])

    p = ScriptedPrompter([str(bad)])
    manage._duplicate(ctx_for(p, tmp_path), [bad])

    assert any("Cannot read broken.yaml" in line for line in p.output)


def test_a_pipeline_saved_under_dash_dash_dir_is_offered_to_run(tmp_path, monkeypatch):
    """
    From the code review (issue 11): `taters --dir X` saved pipelines under X
    (ctx.cwd) while "Run a saved pipeline" listed presets from Path.cwd() --
    a just-saved pipeline appeared in Manage but could never be run.
    """
    from taters.ui.tasks.run_saved import choose_preset

    _folder_pipeline(tmp_path, "field_study", "Field study")
    monkeypatch.chdir(tmp_path.parent)          # pretend the shell is somewhere else

    p = ScriptedPrompter(["__never_reached__"])
    try:
        choose_preset(p, tmp_path)
    except AssertionError:
        pass    # the scripted answer isn't an actual choice; the offer is the test

    offered = [c.label for _q, choices in p.offered for c in choices]
    assert any("Field study" in label for label in offered), offered


def test_escape_inside_a_manage_action_returns_to_the_manage_menu(workspace):
    """
    From the code review (issue 14): the comment promised "backing out of one
    action returns to this menu", but only Cancelled was caught -- Esc at the
    rename prompt (GoBack) climbed out of Manage entirely.
    """
    from taters.ui.prompts import GoBack

    class EscAtText(ScriptedPrompter):
        def text(self, question, *, default=None):
            raise GoBack()

    p = EscAtText([
        "rename", str(workspace / "pipelines" / "my_run.yaml"),
        "done",
    ])
    manage.TASK.run(ctx_for(p, workspace))      # this has to return, not raise

    asked = [q for _k, q in p.asked]
    assert asked.count("What would you like to do with them?") == 2, (
        "Esc inside the action did not come back to the Manage menu"
    )


def test_the_plain_renderers_banner_survives_the_note_wrap(monkeypatch):
    """
    From the code review (issue 16): a renderer without `set_header` gets the
    banner as a note -- whose 64-column wrap counts rich markup as text and
    folded the border mid-tag into garbage.
    """
    import io

    from rich.console import Console

    from taters.ui import glyphs
    from taters.ui.prompts import QuestionaryPrompter

    p = QuestionaryPrompter()
    buf = io.StringIO()
    p._console = Console(file=buf, force_terminal=True, width=100,
                         highlight=False)
    monkeypatch.setattr(hub, "all_tasks", lambda: [])
    p.select = lambda q, choices, **k: "quit"

    hub.run_hub(p, cwd=Path("."))

    import re as _re
    drawn = _re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", buf.getvalue())
    lines = [ln for ln in drawn.splitlines() if "─" in ln or "│" in ln]
    assert lines, "no banner border was drawn at all"
    ends = (glyphs.ARC_TR, glyphs.ARC_BR, "│")
    assert all(ln.rstrip().endswith(ends) for ln in lines), (
        "a border line was folded mid-markup"
    )


def test_quitting_from_a_failed_runs_finish_screen_reports_failure(workspace, monkeypatch):
    """
    From the code review (issue 17): tasks report failure by returning False,
    but QuitRequested skipped the return entirely -- a failed run followed by
    "Finish" exited 0, and anything scripted around the TUI read success.
    """
    from taters.ui.prompts import QuitRequested

    def failed_task(ctx):
        raise QuitRequested(ok=False)

    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="t", label="T", help="", run=failed_task)
    ])
    p = ScriptedPrompter(["t"])

    assert hub.run_hub(p, cwd=workspace) is False


def test_quitting_from_a_good_run_still_reports_success(workspace, monkeypatch):
    from taters.ui.prompts import QuitRequested

    monkeypatch.setattr(hub, "all_tasks", lambda: [
        Task(id="t", label="T", help="",
             run=lambda ctx: (_ for _ in ()).throw(QuitRequested(ok=True)))
    ])
    p = ScriptedPrompter(["t"])

    assert hub.run_hub(p, cwd=workspace) is True


# ---------------------------------------------------------------------------
# One run path, wherever a run starts (code review issue 19)
# ---------------------------------------------------------------------------


def test_running_a_saved_pipeline_asks_about_workers(workspace, monkeypatch):
    """
    From the code review: the run-execution block existed twice and had
    drifted -- the wizard asked how many files to work on at once while this
    path silently hardcoded four. Same pipeline, different parallelism,
    depending on which menu launched it.
    """
    import taters.ui.wizard as wiz

    seen = {}

    def fake_execute(prompter, preset, **kwargs):
        seen.update(kwargs)
        return True, {"errors": [], "items": []}

    monkeypatch.setattr(wiz, "execute_preset", fake_execute)
    # an item-scoped preset, so the workers question applies.
    pipelines = workspace / "pipelines"
    f = pipelines / "fanout.yaml"
    f.write_text(yaml.safe_dump({
        "meta": {"id": "fanout", "title": "Fan out", "summary": "s",
                 "variables": {}},
        "vars": {},
        "steps": [{"scope": "item", "call": "potato.audio.convert_to_wav",
                   "save_as": "wav", "with": {"input_path": "{{input}}"}}],
    }, sort_keys=False), encoding="utf-8")
    media = workspace / "clips"
    media.mkdir()
    (media / "a.wav").write_bytes(b"RIFF")

    p = ScriptedPrompter([
        str(f),                     # which pipeline
        "audio", *browse_to(media), # what kind, and where
        False,                      # change any settings? no
        True,                       # run it now?
        "1",                        # how many at once <- the question under test
    ])
    run_saved.TASK.run(ctx_for(p, workspace))

    assert any("at once" in q for _k, q in p.asked), "workers never asked"
    assert seen.get("workers") == 1
    assert "vars_ctx" in seen, "saved runs must keep their merged variables"


def test_a_global_only_saved_pipeline_is_not_asked_about_workers(workspace, monkeypatch):
    import taters.ui.wizard as wiz

    seen = {}
    monkeypatch.setattr(wiz, "execute_preset",
                        lambda prompter, preset, **kw: (seen.update(kw) or (True, {})))

    p = ScriptedPrompter([
        str(workspace / "pipelines" / "my_run.yaml"),
        False,                  # change any settings? no
        True,                   # run it now? (text preset: no inputs question)
    ])
    run_saved.TASK.run(ctx_for(p, workspace))

    assert not any("at once" in q for _k, q in p.asked)
    # 0 means "let the dial decide": the runner resolves it to the preset's
    # shared `workers` variable, else one process per core. it must NOT be 1
    # -- that would silently force the text analyzers' pools down to serial.
    assert seen.get("workers") == 0


def test_a_stepless_preset_is_refused_politely_when_run(workspace):
    """Round-2 issue 31: `meta:` but no steps passed every check, then
    run_preset raised ValueError('Preset has no steps') into a dead session."""
    pipelines = workspace / "pipelines"
    f = pipelines / "stepless.yaml"
    f.write_text("meta:\n  id: stepless\n  title: Stepless\nsteps: []\n",
                 encoding="utf-8")

    p = ScriptedPrompter([str(f)])
    with pytest.raises(Cancelled):
        run_saved.TASK.run(ctx_for(p, workspace))

    assert any("no runnable steps" in line for line in p.output)


def test_a_preset_without_scope_lines_is_still_asked_about_workers(
        workspace, monkeypatch):
    """Round-2 issue 33: the runner defaults a missing scope to 'item'; the
    fans-out check did not, so hand-written presets ran single-threaded."""
    import taters.ui.wizard as wiz

    seen = {}
    monkeypatch.setattr(wiz, "execute_preset",
                        lambda prompter, preset, **kw: (seen.update(kw) or (True, {})))
    pipelines = workspace / "pipelines"
    f = pipelines / "noscope.yaml"
    f.write_text(yaml.safe_dump({
        "meta": {"id": "noscope", "title": "No scope", "variables": {}},
        "vars": {},
        "steps": [{"call": "potato.audio.convert_to_wav",
                   "save_as": "wav", "with": {"input_path": "{{input}}"}},
                  None],       # a stray '- ' in YAML parses to None
    }, sort_keys=False), encoding="utf-8")
    media = workspace / "clips"
    media.mkdir()
    (media / "a.wav").write_bytes(b"RIFF")

    p = ScriptedPrompter([
        str(f), "audio", *browse_to(media), False, True, "1",
    ])
    run_saved.TASK.run(ctx_for(p, workspace))

    assert any("at once" in q for _k, q in p.asked), "workers never asked"
    assert seen.get("workers") == 1


def test_coerce_gives_null_defaulted_variables_their_types_back():
    """Round-2 cut list: `num_speakers: null` answered as '2' stayed the string
    '2', and 'none' reached the diarization subprocess as a speaker count of
    literally 'none'."""
    from taters.ui.tasks.run_saved import _coerce

    assert _coerce("2", None) == 2
    assert _coerce("0.5", None) == 0.5
    assert _coerce("none", None) is None
    assert _coerce("null", None) is None
    assert _coerce("", None) is None
    assert _coerce("en", None) == "en"      # actual text stays text


def test_manage_sees_the_same_pipelines_run_sees(workspace):
    """Round-2 cut list: discovery rules lived in three copies and had drifted
    -- the runner searches pipelines/ recursively, Manage's copy did not, so a
    nested preset appeared under Run but not under Manage."""
    nested = workspace / "pipelines" / "archive"
    nested.mkdir()
    (nested / "old_study.yaml").write_text(yaml.safe_dump({
        "meta": {"id": "old_study", "title": "Old study"},
        "steps": [{"scope": "global", "call": "potato.text.analyze_readability",
                   "save_as": "r", "with": {"txt_dir": "."}}],
    }, sort_keys=False), encoding="utf-8")

    ctx = ctx_for(ScriptedPrompter([]), workspace)
    assert any(p.name == "old_study.yaml" for p in ctx.own_pipelines())


# ---------------------------------------------------------------------------
# Changing settings on a saved pipeline (requested directly)
# ---------------------------------------------------------------------------


def _wizard_style_preset(root, media):
    """A preset shaped exactly as the wizard saves it (steps map to recipes)."""
    import taters.ui.wizard as wiz
    from wizard_helpers import browse_to

    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe", "speaker",
        False, "Saved run", "save",
    ])
    return wiz.run_wizard(p, cwd=root).preset_path


def test_a_saved_pipeline_gets_the_step_chooser_not_a_flat_march(
        workspace, media, monkeypatch):
    """
    Requested directly: saying yes to "change any settings?" jumped straight
    into the flat global-variable march -- no way to pick a step, none of the
    per-step settings. A wizard-made preset maps back onto its recipes, so it
    gets the wizard's own options screen.
    """
    import taters.ui.wizard as wiz

    path = _wizard_style_preset(workspace, media)
    executed = {}
    monkeypatch.setattr(wiz, "execute_preset",
                        lambda prompter, preset, **kw: (executed.update(
                            {"preset": preset, **kw}) or (True, {})))

    p = ScriptedPrompter([
        str(path),
        "audio", *browse_to(media),
        True,                           # change settings
        "readability",                  # <- the step chooser, not a flat march
        "out_features_csv", "features/readable.csv",
        ":done", ":done",
        True, "1",                      # run it now; workers
    ])
    run_saved.TASK.run(ctx_for(p, workspace))

    assert any(q == "What would you like to change?" for _k, q in p.asked)
    step = next(s for s in executed["preset"]["steps"]
                if s["call"].endswith("analyze_readability"))
    assert step["with"]["out_features_csv"] == "features/readable.csv", (
        "the per-step edit never reached the run"
    )
    # ...and only for this run: the file on disk stays untouched.
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    disk_step = next(s for s in on_disk["steps"]
                     if s["call"].endswith("analyze_readability"))
    assert disk_step["with"]["out_features_csv"] != "features/readable.csv"


def test_variable_edits_still_reach_the_run_as_overrides(
        workspace, media, monkeypatch):
    import taters.ui.wizard as wiz

    path = _wizard_style_preset(workspace, media)
    executed = {}
    monkeypatch.setattr(wiz, "execute_preset",
                        lambda prompter, preset, **kw: (executed.update(kw)
                                                        or (True, {})))
    p = ScriptedPrompter([
        str(path), "audio", *browse_to(media),
        True, ":shared", "overwrite_existing", True, ":done", ":done",
        True, "1",
    ])
    run_saved.TASK.run(ctx_for(p, workspace))

    assert executed["vars_ctx"]["overwrite_existing"] is True


def test_a_hand_written_preset_falls_back_to_the_flat_editor(workspace):
    """my_run.yaml's save_as does not match any recipe, so the per-step screen
    cannot address it -- the flat editor takes over, with a single gate."""
    p = ScriptedPrompter([
        str(workspace / "pipelines" / "my_run.yaml"),
        True,                           # change settings (the one gate)
        True,                           # overwrite_existing? (flat editor)
        True, "1",                      # run; workers... (never reached: mock)
    ])
    import taters.ui.wizard as wiz
    seen = {}
    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(wiz, "execute_preset",
                   lambda prompter, preset, **kw: (seen.update(kw) or (True, {})))
        run_saved.TASK.run(ctx_for(p, workspace))

    gates = [q for _k, q in p.asked if "Change any" in q]
    assert gates == ["Change any settings first?"], f"double gate: {gates}"
    assert seen["vars_ctx"]["overwrite_existing"] is True


def test_quitting_signs_off_under_the_banner(workspace):
    """
    The farewell used to print under the dead menu frame, a whole screen below
    the banner. Quit repaints instead -- clear, potato, goodbye -- so the
    session ends the way it opened.
    """
    p = ScriptedPrompter(["quit"])
    hub.run_hub(p, cwd=workspace)

    goodbye = next(i for i, line in enumerate(p.output)
                   if hub.FAREWELL in str(line))
    assert "TATERS" in str(p.output[goodbye - 1]), "no banner above the farewell"
    assert p.output[goodbye - 2] == "<clear>", "the old screen was left behind"


def test_escape_at_the_main_menu_signs_off_the_same_way(workspace):
    p = _EscapesOnce([])
    hub.run_hub(p, cwd=workspace)

    goodbye = next(i for i, line in enumerate(p.output)
                   if hub.FAREWELL in str(line))
    assert "TATERS" in str(p.output[goodbye - 1])


def _pipeline_with_results(workspace):
    folder = workspace / "essay_run"
    (folder / "features").mkdir(parents=True)
    (folder / "features" / "readability.csv").write_text("a,b\n", encoding="utf-8")
    (folder / "essay_run.yaml").write_text(
        yaml.safe_dump({"meta": {"id": "essay_run", "title": "Essay run"},
                        "steps": [{"scope": "global", "call": "x"}]}),
        encoding="utf-8")
    return folder


def test_deleting_can_spare_the_results(workspace):
    """
    Deleting the whole folder is a choice, not the default meaning of
    "delete a pipeline": the file alone can go, and the results stay put.
    """
    folder = _pipeline_with_results(workspace)
    p = ScriptedPrompter([str(folder / "essay_run.yaml"), "file", "Essay run"])
    ctx = ctx_for(p, workspace)
    manage._delete(ctx, ctx.own_pipelines())

    assert not (folder / "essay_run.yaml").exists()
    assert (folder / "features" / "readability.csv").exists(), (
        "deleting just the pipeline file must not touch the results")


def test_never_mind_backs_out_of_the_delete_with_everything_intact(workspace):
    folder = _pipeline_with_results(workspace)
    p = ScriptedPrompter([str(folder / "essay_run.yaml"), "back"])
    ctx = ctx_for(p, workspace)
    manage._delete(ctx, ctx.own_pipelines())

    assert (folder / "essay_run.yaml").exists()
    assert (folder / "features" / "readability.csv").exists()


def test_a_mistyped_name_gets_its_own_loud_screen(workspace):
    """A dim one-liner on the way back to the menu read as "maybe it worked?".
    The refusal is now a screen of its own: red, explicit, and paused so it
    cannot scroll past unread."""
    p = ScriptedPrompter([str(workspace / "pipelines" / "my_run.yaml"),
                          "not the name"])
    ctx = ctx_for(p, workspace)
    manage._delete(ctx, ctx.own_pipelines())

    assert (workspace / "pipelines" / "my_run.yaml").exists()
    assert any("not this pipeline's name" in line for line in p.output)
    assert any("'not the name'" in line for line in p.output), (
        "the screen should show what was typed next to the real name")
    assert ("pause", ) == tuple(k for k, _ in p.asked if k == "pause"), (
        "the refusal must wait to be read, not scroll away"
    )


def test_the_motto_varies_by_launch_but_not_within_one():
    """
    Requested: a random motto per *execution*. Two properties, both pinned:
    the pick genuinely varies (across reseeded picks), and one session's
    banner says the same thing on every repaint -- a line that changed under
    the reader's eyes would read as a glitch.
    """
    import random

    from taters.ui.hub import _MOTTOS, _pick_motto, banner

    picks = set()
    for seed in range(30):
        random.seed(seed)
        picks.add(_pick_motto())
    random.seed()                       # leave global randomness alone
    assert len(picks) >= 3, "the pick does not actually vary"
    assert picks <= set(_MOTTOS)

    # stable within the session: the banner re-renders per screen, and every
    # render has to carry the same motto. we compare *mottos*, not raw lines
    # -- the border color drifts with time, so comparing whole lines flaked
    # whenever eight renders straddled a color boundary.
    mottos_shown = {next(m for m in _MOTTOS if m in banner())
                    for _ in range(8)}
    assert len(mottos_shown) == 1
