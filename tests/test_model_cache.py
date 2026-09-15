"""
Tests for the model-cache setting (`taters.helpers.settings`) and its
screen: where downloaded models go, in what order the rules decide, that
the choice persists and reaches the environment every downloading library
reads, and that the setup check says so.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from taters.helpers import settings as st
from taters.ui.prompts import ScriptedPrompter
from taters.ui.tasks import TaskContext
from taters.ui.tasks import model_cache as task
from wizard_helpers import browse_to


@pytest.fixture()
def clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    for var in ("TATERS_MODEL_CACHE", "HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_the_rules_decide_in_order_most_explicit_first(clean_env, monkeypatch):
    assert st.model_cache_source() == (Path.home() / ".cache" / "huggingface" / "hub", "default")
    monkeypatch.setenv("HF_HOME", str(clean_env / "hf"))
    assert st.model_cache_source() == (clean_env / "hf" / "hub", "hub environment")
    monkeypatch.setenv("HF_HUB_CACHE", str(clean_env / "hub"))
    assert st.model_cache_source() == (clean_env / "hub", "hub environment")
    st.save_setting(st.MODEL_CACHE_KEY, str(clean_env / "chosen"))
    assert st.model_cache_source() == (clean_env / "chosen", "setting"), \
        "a folder chosen in Taters beats the hub library's own environment"
    monkeypatch.setenv("TATERS_MODEL_CACHE", str(clean_env / "admin"))
    assert st.model_cache_source() == (clean_env / "admin", "environment"), \
        "an administrator's variable beats everything"


def test_settings_persist_in_the_taters_home_and_survive_damage(clean_env):
    path = st.save_setting("model_cache", "/models")
    assert path == clean_env / "home" / "settings.json"
    st.save_setting("other", 3)
    assert st.load_settings() == {"model_cache": "/models", "other": 3}
    st.clear_setting("model_cache")
    st.clear_setting("never_set")
    assert st.load_settings() == {"other": 3}
    path.write_text("{not json", encoding="utf-8")
    assert st.load_settings() == {}, "a damaged file must not stop Taters starting"


def test_applying_the_choice_exports_it_for_every_downloading_library(clean_env, monkeypatch):
    assert st.apply_model_cache() is None and "HF_HUB_CACHE" not in os.environ, \
        "no Taters-level choice: the hub library's environment is left alone"
    st.save_setting(st.MODEL_CACHE_KEY, str(clean_env / "chosen"))
    assert st.apply_model_cache() == clean_env / "chosen"
    assert os.environ["HF_HUB_CACHE"] == str(clean_env / "chosen")
    assert os.environ["TRANSFORMERS_CACHE"] == str(clean_env / "chosen")


def test_the_cache_scan_and_the_loader_follow_the_setting(clean_env, monkeypatch):
    from taters.text._transformer_common import cached_hub_encoders

    snap = clean_env / "chosen" / "models--my--enc" / "snapshots" / "a"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "bert", "num_hidden_layers": 2,
                                                  "hidden_size": 32}), encoding="utf-8")
    st.save_setting(st.MODEL_CACHE_KEY, str(clean_env / "chosen"))
    assert cached_hub_encoders() == [("my/enc", "downloaded · 2 layers · 32 wide")]


def test_the_screen_saves_a_folder_and_can_forget_it(clean_env):
    folder = clean_env / "shared" / "models"
    p = ScriptedPrompter(["new", str(folder), "back"])       # folder isn't there yet
    task.TASK.run(TaskContext(prompter=p, cwd=clean_env))
    assert st.load_settings()[st.MODEL_CACHE_KEY] == str(folder.resolve())
    assert folder.is_dir() and os.environ.get("HF_HUB_CACHE") == str(folder.resolve())
    assert any("Saved. Models download to" in line for line in p.output)

    existing = clean_env / "elsewhere"
    existing.mkdir()
    p = ScriptedPrompter(["change", *browse_to(existing), "back"])
    task.TASK.run(TaskContext(prompter=p, cwd=clean_env))
    assert st.load_settings()[st.MODEL_CACHE_KEY] == str(existing.resolve())

    q = ScriptedPrompter(["default", "back"])
    task.TASK.run(TaskContext(prompter=q, cwd=clean_env))
    assert st.MODEL_CACHE_KEY not in st.load_settings()
    offered = [c.value for c in q.offered[0][1]]
    assert offered == ["change", "new", "default", "back"]
    offered_after = [c.value for c in q.offered[1][1]]
    assert offered_after == ["change", "new", "back"], "nothing to forget any more"


def test_an_administrators_variable_is_explained_and_not_overridden(clean_env, monkeypatch):
    monkeypatch.setenv("TATERS_MODEL_CACHE", str(clean_env / "admin"))
    p = ScriptedPrompter(["back"])
    task.TASK.run(TaskContext(prompter=p, cwd=clean_env))
    assert any("TATERS_MODEL_CACHE is set" in r for r in p.reasons)
    assert "default" not in [c.value for c in p.offered[0][1]]


def test_the_setup_check_names_the_folder_and_how_it_was_chosen(clean_env):
    from taters.ui.tasks import settings as settings_menu
    from taters.ui.tasks.gpu import _system_section

    st.save_setting(st.MODEL_CACHE_KEY, str(clean_env / "chosen"))
    row = next(f for f in _system_section().findings if f.layer == "Downloaded models")
    assert str(clean_env / "chosen") in row.detail and "chosen in Settings" in row.detail
    assert "nothing downloaded yet" in row.detail
    # it moved a level down, in with everything else Taters keeps on disk
    from taters.ui.tasks import manage_data

    assert "manage_data" in [t.id for t in settings_menu.entries()]
    assert "model_cache" in [t.id for t in manage_data.entries()]
