"""
What every wizard test file shares: the scripted drivers and the machine fixture.

The wizard tests were one 4,000-line file, interleaved by the order bugs were
fixed rather than by subject, with these helpers defined once at the top and
imported from it by two other test modules. They live here now, and the tests
are split by stage (test_wizard_flow / _sources / _options / _analysis).

Not a pytest file: imported like `preset_checks`. `clean_machine` is an
autouse fixture; a wizard test module imports it to activate it.
"""

from __future__ import annotations

import pytest

from taters.ui import wizard as wiz
from taters.ui.browse import _TYPE as BROWSE_TYPE
from taters.ui.prompts import GoBack, ScriptedPrompter

__all__ = ["clean_machine", "browse_to", "run", "EscapingPrompter", "_tuning",
           "_escaping_tuning", "_engine_tuning", "_asked", "BROWSE_TYPE"]


@pytest.fixture(autouse=True)
def clean_machine(monkeypatch):
    """
    Pretend every optional dependency is installed and ffmpeg is on PATH.

    Without this, the number of questions the wizard asks depends on what
    happens to be installed in the environment running the tests, and every
    scripted answer list would shift by one on a different machine.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda recipe: [])
    monkeypatch.setattr(wiz.shutil, "which", lambda name: f"/usr/bin/{name}")


def browse_to(path) -> list:
    """
    The answers that pick `path` in the file browser.

    Folders and files are chosen by browsing now, not typed, so a single path
    answer becomes two: "Type a path instead", then the path. Tests take that
    shortcut rather than walking the tree, because what they are testing is the
    flow around the question, not the browser -- `test_browse.py` covers that.
    """
    return [BROWSE_TYPE, str(path)]


def run(answers, cwd) -> tuple:
    """Drive the wizard with a scripted answer list."""
    prompter = ScriptedPrompter(list(answers))
    return wiz.run_wizard(prompter, cwd=cwd), prompter


class EscapingPrompter(ScriptedPrompter):
    """A scripted prompter where the answer `"__esc__"` presses Esc instead."""

    def _next(self, kind, question, default=None):
        value = super()._next(kind, question, default)
        if value == "__esc__":
            from taters.ui.prompts import GoBack
            raise GoBack()
        return value

    def select(self, question, choices, *, default=None, transient=False,
               toggle_values=(), ticked=None, navigate=None, breadcrumb=None,
               cycle=None):
        self.offered.append((question, list(choices)))
        return self._next("select", question, default)


def _tuning(answers):
    """Drive `ask_tuning` over two real steps and hand back the prompter."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    p = ScriptedPrompter(answers)
    steps = [_r.by_id("transcribe"), _r.by_id("sentence_embeddings")]
    var_specs = compose(["sentence_embeddings"],
                        providers={"transcript_csv": "transcribe"},
                        name="x")["meta"]["variables"]
    return p, wiz.ask_tuning(p, steps, var_specs)


def _escaping_tuning(answers):
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    p = EscapingPrompter(answers)
    steps = [_r.by_id("transcribe"), _r.by_id("sentence_embeddings")]
    var_specs = compose(["sentence_embeddings"],
                        providers={"transcript_csv": "transcribe"},
                        name="x")["meta"]["variables"]
    # we own these dicts (not the wizard), so whatever's left in them after an
    # Esc out of the top of the screen is exactly what `run_wizard` would see too
    overrides, var_values = {}, {}
    try:
        wiz.ask_tuning(p, steps, var_specs, overrides, var_values)
        backed_out = False
    except GoBack:
        backed_out = True
    return p, (overrides, var_values), backed_out


def _engine_tuning(answers):
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("parts_of_speech")]
    var_specs = compose(["parts_of_speech"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter(answers)
    _overrides, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False)
    return p, var_values


def _asked(p, question):
    return [q for _k, q in p.asked if q == question]
