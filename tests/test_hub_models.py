"""
A Hugging Face model offered for download when it is chosen: the "already
here" check on a cache laid out by hand, and the wizard's conversation with
the hub stubbed -- not here, download; gated, token, download; a token the
hub refuses; an approval still pending; no such model; no network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from taters.helpers import hub_models as hm
from taters.ui import library as uilib
from taters.ui.prompts import ScriptedPrompter


def _cached(cache: Path, hub_id: str, files: dict) -> Path:
    folder = cache / ("models--" + hub_id.replace("/", "--"))
    snap = folder / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (folder / "refs").mkdir()
    (folder / "refs" / "main").write_text("abc123", encoding="utf-8")
    for name, body in files.items():
        (snap / name).write_text(body, encoding="utf-8")
    return snap


def test_a_model_counts_as_here_only_when_every_weight_file_is(tmp_path):
    whole = {"config.json": "{}", "tokenizer.json": "{}", "model.safetensors": "w"}
    _cached(tmp_path, "org/whole", whole)
    assert hm.is_downloaded("org/whole", tmp_path)
    index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    _cached(tmp_path, "org/half", {"config.json": "{}", "tokenizer.json": "{}",
                                   "model.safetensors.index.json": index,
                                   "model-1.safetensors": "w"})
    assert not hm.is_downloaded("org/half", tmp_path), "a stopped download is not a model"
    _cached(tmp_path, "org/config-only", {"config.json": "{}"})
    assert not hm.is_downloaded("org/config-only", tmp_path)
    assert not hm.is_downloaded("org/never", tmp_path)


class _Sibling:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


def test_the_plan_fetches_what_a_run_needs_and_nothing_else(monkeypatch):
    import huggingface_hub

    class Info:
        siblings = [_Sibling("config.json", 1_000), _Sibling("model.safetensors", 2_000_000_000),
                    _Sibling("tokenizer.json", 9_000_000), _Sibling("pytorch_model.bin", 2e9),
                    _Sibling("original/consolidated.00.pth", 6e9), _Sibling("README.md", 10)]

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: Info())
    plan = hm.plan_download("org/m")
    assert plan.state == "ok"
    assert plan.files == ["config.json", "model.safetensors", "tokenizer.json"]
    assert plan.gigabytes == "2.0 GB"


@pytest.fixture
def hub(monkeypatch):
    """The hub, scripted: what `access` answers in turn, what the token
    check says, and a record of downloads."""
    state = {"access": ["ok"], "downloads": [], "tokens": [], "bad": set()}
    monkeypatch.setattr(hm, "is_downloaded", lambda name, cache=None: False)
    monkeypatch.setattr(hm, "plan_download", lambda name: hm.DownloadPlan(
        name, files=["config.json"], size=6_400_000_000))
    monkeypatch.setattr(hm, "access", lambda name: state["access"].pop(0)
                        if len(state["access"]) > 1 else state["access"][0])

    def save(token):
        if token in state["bad"]:
            raise ValueError("Invalid user token.")
        state["tokens"].append(token)

    def fetch(plan, on_bytes=None, every=0.5):
        state["downloads"].append(plan.hub_id)
        if on_bytes:
            on_bytes(plan.size)
        return Path("snap")

    monkeypatch.setattr(hm, "save_token", save)
    monkeypatch.setattr(hm, "download", fetch)
    monkeypatch.delenv("TATERS_FAKE_LLM", raising=False)
    return state


def test_a_model_already_here_asks_nothing(monkeypatch):
    monkeypatch.setattr(hm, "is_downloaded", lambda name, cache=None: True)
    monkeypatch.delenv("TATERS_FAKE_LLM", raising=False)
    p = ScriptedPrompter([])
    assert uilib.make_local(p, "Qwen/Qwen3-1.7B") and p.asked == []


def test_a_model_not_here_is_offered_with_its_size_and_downloaded(hub):
    p = ScriptedPrompter([True])
    assert uilib.make_local(p, "Qwen/Qwen3-4B")
    assert hub["downloads"] == ["Qwen/Qwen3-4B"]
    assert any("about 6.4 GB to download" in line for line in p.output)
    assert any("Downloaded Qwen/Qwen3-4B" in line for line in p.output)


def test_declining_leaves_it_for_the_first_run(hub):
    p = ScriptedPrompter([False])
    assert uilib.make_local(p, "Qwen/Qwen3-4B") and hub["downloads"] == []
    assert any("downloaded when the pipeline first runs" in line for line in p.output)


def test_a_gated_model_asks_for_a_token_saves_it_and_downloads(hub):
    hub["access"] = ["gated", "ok"]
    p = ScriptedPrompter([True, "hf_good"])
    assert uilib.make_local(p, "meta-llama/Llama-3.2-3B-Instruct")
    assert hub["tokens"] == ["hf_good"] and hub["downloads"] == ["meta-llama/Llama-3.2-3B-Instruct"]
    said = " ".join(p.output)
    assert "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct" in said
    assert "settings/tokens" in said and "Signed in to Hugging Face" in said


def test_a_refused_token_is_asked_for_again_and_enter_skips(hub):
    hub["access"] = ["gated"]
    hub["bad"] = {"hf_typo"}
    p = ScriptedPrompter([True, "hf_typo", ""])
    assert uilib.make_local(p, "meta-llama/Llama-3.2-3B-Instruct")
    assert hub["tokens"] == [] and hub["downloads"] == []
    said = " ".join(p.output)
    assert "That token was not accepted (Invalid user token.)" in said
    assert "Without access it will stop there" in said


def test_an_approval_still_pending_can_be_checked_again_or_traded_for_another_model(hub):
    hub["access"] = ["pending", "ok"]
    p = ScriptedPrompter([True, "again"])
    assert uilib.make_local(p, "meta-llama/Llama-3.2-3B-Instruct")
    assert hub["downloads"] == ["meta-llama/Llama-3.2-3B-Instruct"]
    hub["access"] = ["pending"]
    hub["downloads"].clear()
    p = ScriptedPrompter([True, "other"])
    assert not uilib.make_local(p, "meta-llama/Llama-3.2-3B-Instruct"), "pick again"
    assert hub["downloads"] == []


def test_no_such_model_picks_again_and_no_network_keeps_the_pick(hub, monkeypatch):
    monkeypatch.setattr(hm, "plan_download", lambda name: hm.DownloadPlan(name, state="missing"))
    p = ScriptedPrompter([])
    assert not uilib.make_local(p, "nobody/nothing")
    monkeypatch.setattr(hm, "plan_download", lambda name: hm.DownloadPlan(
        name, state="offline", why="Max retries exceeded"))
    p = ScriptedPrompter([])
    assert uilib.make_local(p, "Qwen/Qwen3-4B") and p.asked == []
    assert any("Could not reach Hugging Face" in line for line in p.output)


def test_folders_servers_and_dry_runs_are_left_alone(hub, tmp_path, monkeypatch):
    p = ScriptedPrompter([])
    assert uilib.make_local(p, str(tmp_path))
    assert uilib.make_local(p, "http://localhost:11434/v1#qwen")
    monkeypatch.setenv("TATERS_FAKE_LLM", "1")
    assert uilib.make_local(p, "Qwen/Qwen3-4B")
    assert p.asked == [] and hub["downloads"] == []


def test_the_picker_offers_the_download_and_picks_again_after_a_wrong_name(hub, monkeypatch):
    monkeypatch.setattr(hm, "plan_download", lambda name: hm.DownloadPlan(
        name, state="missing") if name == "nobody/nothing" else hm.DownloadPlan(
        name, files=["config.json"], size=1_000_000_000))
    p = ScriptedPrompter([":type", "nobody/nothing", "Qwen/Qwen3-0.6B", True])
    assert uilib.pick_generative_model(p, "") == "Qwen/Qwen3-0.6B"
    assert hub["downloads"] == ["Qwen/Qwen3-0.6B"]


def test_a_repository_with_nothing_runnable_is_said_so_and_never_fetched_whole(monkeypatch):
    """A GGUF-only repository: the old plan had no files and a size of 0 MB,
    and `allow_patterns=None` would have fetched every multi-GB export in it."""
    import huggingface_hub

    class Info:
        siblings = [_Sibling("README.md", 10), _Sibling("Qwen2.5-7B-Q4_K_M.gguf", 4_700_000_000),
                    _Sibling("Qwen2.5-7B-Q8_0.gguf", 8_100_000_000)]

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: Info())
    plan = hm.plan_download("someone/Qwen2.5-7B-GGUF")
    assert plan.state == "unusable" and plan.files == []
    assert ".gguf" in plan.why and "Ollama" in plan.why
    with pytest.raises(ValueError, match="holds no weights"):
        hm.download(plan)
    with pytest.raises(ValueError, match="nothing to download"):
        hm.download(hm.DownloadPlan("org/m"))      # no files: never the whole repository


def test_the_wizard_says_why_a_repository_cannot_be_used_and_asks_again(hub, monkeypatch):
    monkeypatch.setattr(hm, "plan_download", lambda name: hm.DownloadPlan(
        name, state="unusable", why="org/gguf holds no weights a step can run"))
    p = ScriptedPrompter([])
    assert uilib.make_local(p, "org/gguf") is False, "pick another model"
    assert any("holds no weights" in line for line in p.output)
