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


@pytest.fixture()
def empty_list(monkeypatch):
    """The folder screen is reached through the list; an empty cache keeps
    these tests off this machine's real one."""
    from taters.helpers import hub_models

    monkeypatch.setattr(hub_models, "cached_repos", lambda cache=None: [])


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


def _folder_screens(p):
    return [rows for q, rows in p.offered if q == "Model folder:"]


def test_the_screen_saves_a_folder_and_can_forget_it(clean_env, empty_list):
    folder = clean_env / "shared" / "models"
    p = ScriptedPrompter([":folder", "new", str(folder), ":back"])   # folder isn't there yet
    task.TASK.run(TaskContext(prompter=p, cwd=clean_env))
    assert st.load_settings()[st.MODEL_CACHE_KEY] == str(folder.resolve())
    assert folder.is_dir() and os.environ.get("HF_HUB_CACHE") == str(folder.resolve())
    assert any("Saved. Models download to" in line for line in p.output)

    existing = clean_env / "elsewhere"
    existing.mkdir()
    p = ScriptedPrompter([":folder", "change", *browse_to(existing), ":back"])
    task.TASK.run(TaskContext(prompter=p, cwd=clean_env))
    assert st.load_settings()[st.MODEL_CACHE_KEY] == str(existing.resolve())

    q = ScriptedPrompter([":folder", "default", ":folder", "back", ":back"])
    task.TASK.run(TaskContext(prompter=q, cwd=clean_env))
    assert st.MODEL_CACHE_KEY not in st.load_settings()
    first, after = _folder_screens(q)
    assert [c.value for c in first] == ["change", "new", "default", "back"]
    assert [c.value for c in after] == ["change", "new", "back"], "nothing to forget any more"


def test_an_administrators_variable_is_explained_and_not_overridden(clean_env, monkeypatch,
                                                                     empty_list):
    monkeypatch.setenv("TATERS_MODEL_CACHE", str(clean_env / "admin"))
    p = ScriptedPrompter([":folder", "back", ":back"])
    task.TASK.run(TaskContext(prompter=p, cwd=clean_env))
    assert any("TATERS_MODEL_CACHE is set" in r for r in p.reasons)
    (rows,) = _folder_screens(p)
    assert "default" not in [c.value for c in rows]


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



# ---------------------------------------------------------------------------
# the list of what is downloaded
# ---------------------------------------------------------------------------

def _repo(cache: Path, repo_id: str, files: dict, *, kind: str = "model",
          refs=("main",), commit: str = "a" * 40) -> Path:
    """One repository laid out the way the hub library lays it out: the
    bytes in blobs/, the snapshot files linked to them, refs naming the
    commit."""
    import hashlib

    folder = cache / f"{kind}s--{repo_id.replace('/', '--')}"
    snap = folder / "snapshots" / commit
    snap.mkdir(parents=True)
    (folder / "blobs").mkdir(exist_ok=True)
    for name, body in files.items():
        data = body.encode("utf-8") if isinstance(body, str) else body
        blob = folder / "blobs" / hashlib.sha256(data + commit.encode()).hexdigest()
        blob.write_bytes(data)
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).symlink_to(blob)
    for ref in refs:
        (folder / "refs" / ref).parent.mkdir(parents=True, exist_ok=True)
        (folder / "refs" / ref).write_text(commit, encoding="utf-8")
    return folder


def _llm(cache):
    return _repo(cache, "org/chat-1b", {
        "config.json": json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                                   "num_hidden_layers": 2, "hidden_size": 32}),
        "tokenizer.json": "{}", "model.safetensors": b"w" * 5000})


@pytest.fixture()
def cache(clean_env, monkeypatch):
    root = clean_env / "hub"
    root.mkdir()
    monkeypatch.setenv("TATERS_MODEL_CACHE", str(root))
    return root


def test_every_repository_says_what_it_is_to_taters(cache):
    from taters.helpers import hub_models as hm

    _llm(cache)
    _repo(cache, "kredor/punctuate-all", {
        "config.json": json.dumps({"architectures": ["XLMRobertaForTokenClassification"],
                                   "model_type": "xlm-roberta"}), "model.safetensors": "w"})
    _repo(cache, "someone/diffusion", {"config.json": json.dumps({"model_type": "unet"}),
                                       "model.safetensors": "w" * 300})
    _repo(cache, "dell/stories", {"data.parquet": "rows"}, kind="dataset")
    _repo(cache, "org/enc-config-only", {"config.json": json.dumps(
        {"architectures": ["BertModel"], "model_type": "bert", "num_hidden_layers": 2,
         "hidden_size": 8}), "tokenizer.json": "{}"})
    _repo(cache, "Systran/faster-whisper-base.en", {"config.json": "{}", "model.bin": "w"})
    by = {r.hub_id: r for r in hm.cached_repos()}
    assert (by["org/chat-1b"].role, by["org/chat-1b"].uses) == \
        ("step", "Generate responses with an LLM")
    assert by["kredor/punctuate-all"].role == "part", "a step's own part, never offered"
    assert by["Systran/faster-whisper-base.en"].uses == "Transcript (one speaker)"
    assert by["someone/diffusion"].role == "other"
    assert (by["dell/stories"].role, by["dell/stories"].kind) == ("other", "dataset")
    assert not by["org/enc-config-only"].complete and by["org/chat-1b"].complete
    roles = [r.role for r in hm.cached_repos()]
    assert roles == sorted(roles, key=["step", "part", "other"].index), "steps first"


def test_only_a_version_no_branch_points_to_is_a_leftover(cache):
    """transformers loads a model's safetensors from the hub's conversion
    branch, so a version a branch names is in use even when it is not
    main (a real cache: CLIP's)."""
    from taters.helpers import hub_models as hm

    folder = _llm(cache)
    _repo(cache, "org/chat-1b", {"model.safetensors": b"x" * 7000}, refs=("pr/3",),
          commit="b" * 40)
    _repo(cache, "org/chat-1b", {"model.safetensors": b"y" * 9000}, refs=(), commit="c" * 40)
    (repo,) = [r for r in hm.cached_repos() if r.hub_id == "org/chat-1b"]
    assert repo.current == "a" * 40 and repo.leftovers == ["c" * 40]
    assert repo.old_size == 9000
    assert hm.delete_old_versions([repo]) == 9000
    assert not (folder / "snapshots" / ("c" * 40)).exists()
    assert (folder / "snapshots" / ("b" * 40)).exists() and (folder / "snapshots" / ("a" * 40)).exists()


def test_the_saved_pipelines_that_name_a_model_are_found(cache, clean_env):
    from taters.helpers import hub_models as hm

    _llm(cache)
    _repo(cache, "Systran/faster-whisper-base.en", {"config.json": "{}", "model.bin": "w"})
    mine = clean_env / "work" / "study"
    mine.mkdir(parents=True)
    (mine / "study.yaml").write_text(
        "meta: {id: study, title: My study}\n"
        "vars: {llm_model: org/chat-1b, whisper_model: base.en}\nsteps: []\n",
        encoding="utf-8")
    by = {r.hub_id: r for r in hm.cached_repos()}
    assert hm.pipelines_naming(by["org/chat-1b"], clean_env / "work") == ["My study"]
    named = hm.pipelines_naming(by["Systran/faster-whisper-base.en"], clean_env / "work")
    assert "My study" in named, "a Whisper model is named by its short name"
    assert all(n == "My study" or n.endswith("(built in)") for n in named)


def test_deleting_a_model_a_pipeline_uses_warns_by_name_and_asks(cache, clean_env):
    folder = _llm(cache)
    mine = clean_env / "work" / "study"
    mine.mkdir(parents=True)
    (mine / "study.yaml").write_text(
        "meta: {id: study, title: My study}\nvars: {llm_model: org/chat-1b}\nsteps: []\n",
        encoding="utf-8")
    p = ScriptedPrompter(["\x00space:org/chat-1b", ":delete", False, ":back"])
    task.manage_model_cache(p, clean_env / "work")
    said = " ".join(p.output)
    assert "org/chat-1b is named by 1 saved pipeline(s): My study" in said
    assert any("even though saved pipelines use some of them" in q for _k, q in p.asked)
    assert folder.exists(), "no: nothing deleted"
    p = ScriptedPrompter(["\x00space:org/chat-1b", ":delete", True, ":back"])
    task.manage_model_cache(p, clean_env / "work")
    assert not folder.exists() and any("Deleted 1 model(s)" in line for line in p.output)


def test_deleting_something_taters_does_not_use_says_another_program_may_need_it(cache):
    folder = _repo(cache, "someone/diffusion", {"config.json": json.dumps({"model_type": "unet"})})
    p = ScriptedPrompter(["\x00space:someone/diffusion", ":delete", True, ":back"])
    task.manage_model_cache(p, cache)
    assert any("not used by Taters. Another program on this computer may need it" in line
               for line in p.output)
    assert not folder.exists()


def test_several_models_export_at_once_as_plain_folders(cache, clean_env):
    _llm(cache)
    _repo(cache, "other/chat-1b", {"config.json": "{}", "model.safetensors": b"z" * 100})
    _repo(cache, "org/enc", {"config.json": "{}", "sub/vocab.txt": "a b c"})
    dest = clean_env / "usb"
    dest.mkdir()
    p = ScriptedPrompter(["\x00space:org/chat-1b", "\x00space:other/chat-1b",
                          "\x00space:org/enc", ":export", *browse_to(dest), ":back"])
    task.manage_model_cache(p, clean_env)
    made = sorted(x.name for x in dest.iterdir())
    assert made == ["chat-1b", "enc", "other--chat-1b"], "a name already taken gets its owner"
    copied = dest / "chat-1b" / "model.safetensors"
    assert copied.is_file() and not copied.is_symlink() and copied.stat().st_size == 5000
    assert (dest / "enc" / "sub" / "vocab.txt").read_text(encoding="utf-8") == "a b c"
    assert not [x for x in dest.iterdir() if x.name.endswith(".copying")]


def test_an_export_that_does_not_fit_copies_nothing(cache, clean_env, monkeypatch):
    import shutil
    from collections import namedtuple

    _llm(cache)
    dest = clean_env / "small"
    dest.mkdir()
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10, 10, 10))
    p = ScriptedPrompter(["\x00space:org/chat-1b", ":export", *browse_to(dest), ":back"])
    task.manage_model_cache(p, clean_env)
    assert list(dest.iterdir()) == []
    assert any("Nothing was copied" in line for line in p.output)


def test_the_list_says_what_each_row_is_and_offers_leftovers_only_when_there_are_some(cache):
    _llm(cache)
    _repo(cache, "someone/diffusion", {"config.json": json.dumps({"model_type": "unet"})})
    p = ScriptedPrompter([":back"])
    task.manage_model_cache(p, cache)
    (question, rows), = [(q, r) for q, r in p.offered if q == "Downloaded models:"]
    by = {c.value: c for c in rows}
    assert by["org/chat-1b"].help.startswith("Used by Generate responses with an LLM")
    assert by["someone/diffusion"].help.startswith("Not used by Taters")
    assert ":leftovers" not in by
    assert any("One of these is not used by Taters" in r for r in p.reasons)
    assert by["someone/diffusion"].annotation.endswith(" · not used by Taters")
