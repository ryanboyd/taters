"""
The local language-model backend on a tiny causal model: the memory plan
from a config, the precision ladder and its refusals, the chat template
with its thinking switch, batched generation equal to one-at-a-time, the
out-of-memory fallback, and the resume of a half-finished table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from taters.helpers import tables  # noqa: E402
from taters.text import _generation_local as gl  # noqa: E402
from taters.text import generate_with_llm as al  # noqa: E402
from taters.text._generation import Conversation  # noqa: E402
from csvhelpers import _read  # noqa: E402
from tiny_llm import build  # noqa: E402

GB = gl.GB


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("llm") / "tiny")


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

def test_a_checkpoint_folder_is_counted_exactly_and_the_estimate_is_close(tiny):
    import torch
    from transformers import AutoModelForCausalLM

    facts = gl.model_facts(str(tiny))
    assert facts.kind == "folder" and not facts.problem and facts.exact
    model = AutoModelForCausalLM.from_pretrained(str(tiny), torch_dtype=torch.float32)
    real = sum(p.numel() for p in model.parameters())
    assert facts.params == real
    assert abs(gl.estimate_params(facts.config) - real) / real < 0.05
    assert gl.context_of(facts.config) == 128
    assert gl.kv_bytes_per_token(facts.config) == 2 * 2 * 2 * 16 * 2


def test_a_missing_or_wrong_model_is_a_sentence_not_a_traceback(tmp_path):
    assert "no language model was named" in gl.model_facts("").problem
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no config.json" in gl.model_facts(str(empty)).problem


def test_a_gated_model_says_how_to_get_access_however_the_refusal_is_wrapped(monkeypatch,
                                                                           tmp_path):
    """transformers wraps the hub's gated error in a plain OSError; a gated
    Llama used to come out as "the hub could not be reached" (a real run)."""
    import huggingface_hub
    import transformers
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: object())
    name = "meta-llama/Llama-3.2-3B-Instruct"

    def wrapped(*a, **k):
        try:
            raise GatedRepoError("401 Client Error. Cannot access gated repo")
        except GatedRepoError as inner:
            raise OSError("You are trying to access a gated repo.") from inner

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", wrapped)
    problem = gl.model_facts(name).problem
    assert problem.startswith(f"{name} is gated") and "hf auth login" in problem
    assert "could not be reached" not in problem
    # the words alone are enough when the chain was lost on the way
    plain = gl._hub_problem(name, "cache", OSError("Cannot access gated repo for url ..."))
    assert "is gated" in plain
    login = gl._hub_problem(name, "cache", OSError("401 Client Error: Unauthorized"))
    assert "would not let this machine download" in login
    offline = gl._hub_problem(name, "cache", OSError("Max retries exceeded\nmore detail"))
    assert "could not be reached (OSError: Max retries exceeded)" in offline


def _llama_config(billions: float) -> dict:
    """A Llama-shaped config with roughly this many parameters."""
    H = 4096
    return {"model_type": "llama", "hidden_size": H, "num_hidden_layers": int(billions * 4),
            "vocab_size": 32000, "num_attention_heads": 32, "num_key_value_heads": 8,
            "intermediate_size": 11008, "max_position_embeddings": 8192,
            "tie_word_embeddings": False, "torch_dtype": "bfloat16"}


def test_the_ladder_picks_sixteen_bit_then_four_bit_then_refuses(monkeypatch):
    facts = gl.ModelFacts("m", "m", "hub", _llama_config(7), gl.estimate_params(_llama_config(7)),
                          False)
    monkeypatch.setattr(gl, "model_facts", lambda *a, **k: facts)
    monkeypatch.setattr(gl, "_bf16_capable", lambda: True)
    monkeypatch.setattr(gl, "_quantization_available", lambda: False)
    # a 24 GB card: the 7B model fits at bf16 with room for a batch
    big = gl.plan_load("m", gpu=(22 * GB, 24 * GB), ram=(32 * GB, 64 * GB))
    assert big.precision == "bf16" and big.device == "cuda:0" and big.batch > 1
    assert not big.refusal and "GB at bf16" in big.describe()
    # an 8 GB card without bitsandbytes: refused, with the three fixes
    small = gl.plan_load("m", gpu=(7 * GB, 8 * GB), ram=(32 * GB, 64 * GB))
    assert "does not fit this GPU" in small.refusal
    assert "smaller model" in small.refusal and 'taters[llm]' in small.refusal
    assert "device: cpu" in small.refusal
    # the same card with bitsandbytes: 4-bit
    monkeypatch.setattr(gl, "_quantization_available", lambda: True)
    quant = gl.plan_load("m", gpu=(7 * GB, 8 * GB), ram=(32 * GB, 64 * GB))
    assert quant.precision == "nf4" and quant.device == "cuda:0"
    assert any("4-bit" in n for n in quant.notes)
    # offload is only ever asked for
    spill = gl.plan_load("m", gpu=(2 * GB, 8 * GB), ram=(32 * GB, 64 * GB), offload=True)
    assert spill.device == "auto" and any("spill" in n for n in spill.notes)
    # a named precision that cannot be had says what to install
    monkeypatch.setattr(gl, "_quantization_available", lambda: False)
    named = gl.plan_load("m", precision="nf4", gpu=(7 * GB, 8 * GB), ram=(32 * GB, 64 * GB))
    assert 'pip install "taters[llm]"' in named.refusal


def test_the_cpu_plan_needs_the_ram_and_warns_about_speed(monkeypatch):
    facts = gl.ModelFacts("m", "m", "hub", _llama_config(7), gl.estimate_params(_llama_config(7)),
                          False)
    monkeypatch.setattr(gl, "model_facts", lambda *a, **k: facts)
    cpu = gl.plan_load("m", device="cpu", gpu=(None, None), ram=(40 * GB, 64 * GB))
    assert cpu.device == "cpu" and cpu.precision == "bf16" and not cpu.refusal
    assert any("tens of seconds" in n for n in cpu.notes)
    thin = gl.plan_load("m", device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    assert "GB of RAM" in thin.refusal
    asked = gl.plan_load("m", device="cuda", gpu=(None, None), ram=(40 * GB, 64 * GB))
    assert "no CUDA device" in asked.refusal


def test_the_batch_follows_the_memory_left_after_the_weights():
    cfg = _llama_config(7)
    kv = gl.kv_bytes_per_token(cfg)
    few = gl.batch_for(10 * GB, 8 * GB, 12 * GB, kv, 512, 256, cfg)
    many = gl.batch_for(40 * GB, 8 * GB, 48 * GB, kv, 512, 256, cfg)
    assert 1 <= few < many <= 64
    assert gl.batch_for(6 * GB, 8 * GB, 8 * GB, kv, 512, 256, cfg) == 1


def test_free_memory_reads_something_on_this_machine():
    avail, total = gl.free_memory()
    assert total is None or total > 0


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _texts(tmp_path: Path, n: int = 6) -> Path:
    words = ["the cat sat on the mat", "a dog ran to the sea", "potato gravy and butter",
             "the meeting was today", "good email from the boss", "very happy people today",
             "fish swam in the sea and the sky was good", "bad budget"]
    p = tmp_path / "c.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(n):
            w.writerow([f"t{i}", words[i % len(words)]])
    return p


def test_batched_greedy_generation_equals_one_at_a_time(tiny):
    gen = gl.LocalGenerator(str(tiny), device="cpu", batch_size=4, max_new_tokens=8)
    assert gen.kind == "transformers" and gen.context_length == 128
    convs = [Conversation(prefix=[("system", "you rate cats")], user=t)
             for t in ("the cat sat", "a dog ran to the sea and the sky", "potato")]
    together = gen.generate(convs, max_new_tokens=6)
    alone = [gen.generate([c], max_new_tokens=6)[0] for c in convs]
    assert together == alone and len(together) == 3
    assert all(isinstance(r, str) for r in together)
    assert gen.count_tokens(convs[1].messages()) > gen.count_tokens(convs[2].messages())
    gen.close()


def test_the_thinking_switch_reaches_a_template_that_has_one(tiny):
    off = gl.LocalGenerator(str(tiny), device="cpu", thinking=False, max_new_tokens=4)
    on = gl.LocalGenerator(str(tiny), device="cpu", thinking=True, max_new_tokens=4)
    messages = [{"role": "user", "content": "the cat"}]
    assert off.template_kwargs == {"enable_thinking": False}
    assert on.count_tokens(messages) < off.count_tokens(messages), \
        "thinking off opens the reply with an empty think block"
    assert "thinking on" in on.describe() and "thinking on" not in off.describe()


def test_the_gpus_out_of_memory_halves_the_batch(tiny, monkeypatch):
    gen = gl.LocalGenerator(str(tiny), device="cpu", batch_size=4, max_new_tokens=4)
    real = gen.model.generate
    sizes = []

    def flaky(*args, **kwargs):
        n = kwargs["input_ids"].shape[0]
        sizes.append(n)
        if n > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return real(*args, **kwargs)

    monkeypatch.setattr(gen.model, "generate", flaky)
    convs = [Conversation(user=f"text {i}") for i in range(4)]
    replies = gen.generate(convs, max_new_tokens=4)
    assert len(replies) == 4 and all(r is not None for r in replies)
    assert sizes[0] == 4 and max(sizes[1:]) <= 2
    # the size that got through is remembered: the next call doesn't pay
    # the out-of-memory and the halving all over again
    assert gen.batch_size == 2
    del sizes[:]
    gen.generate(convs, max_new_tokens=4)
    assert sizes and max(sizes) <= 2, "the second call started at the halved size"


# ---------------------------------------------------------------------------
# the step on the real backend, and the resume
# ---------------------------------------------------------------------------

def test_the_step_runs_the_tiny_model_end_to_end(tiny, tmp_path):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood from 1 to 7 as JSON.\n", encoding="utf-8")
    out = al.generate_with_llm(parse_json=True, prompt_paths=[p], model=str(tiny), device="cpu",
                                csv_path=_texts(tmp_path), text_cols=["text"],
                                id_cols=["text_id"], out_features_csv=tmp_path / "f" / "llm.csv",
                                max_new_tokens=6, verbose=False)
    rows = _read(out)
    assert len(rows) == 6 and "Mood__parse_ok" in rows[0] and "Mood__reply" in rows[0]
    assert all(r["Mood__error"] == "" for r in rows)
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "backend: transformers" in report and "parameters" in report
    record = json.loads((tmp_path / "f" / "llm_settings.json").read_text(encoding="utf-8"))
    assert record["instrument"]["model"] == str(tiny)


def _long_corpus(tmp_path):
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["long", " ".join(["the cat sat on the mat."] * 300)])
        w.writerow(["short", "bad budget"])
    return src


def test_a_long_text_is_read_with_notes_on_a_real_tokenizer(tmp_path_factory, tmp_path):
    """Through a real tokenizer and chat template, on a 1,024-token model: the
    long text is read part by part with notes, every call fits (the model
    would fail on one that did not), and the one answer is written."""
    roomy = build(tmp_path_factory.mktemp("llm") / "roomy", max_positions=1024)
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood from 1 to 7 as JSON.\n", encoding="utf-8")
    out = al.generate_with_llm(parse_json=True, prompt_paths=[p], model=str(roomy), device="cpu",
                                csv_path=_long_corpus(tmp_path), text_cols=["text"],
                                id_cols=["text_id"], out_features_csv=tmp_path / "f" / "llm.csv",
                                max_new_tokens=4, long_texts="notes", verbose=False)
    rows = {r["text_id"]: r for r in _read(out)}
    n = int(rows["long"]["Mood__chunks"])
    assert n > 2 and rows["short"]["Mood__chunks"] == "1"
    assert len(json.loads(rows["long"]["Mood__chunk_replies"])) >= n
    assert not rows["long"]["Mood__error"].startswith("the answer:")


def test_notes_on_a_model_too_small_for_them_is_a_sentence(tiny, tmp_path):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood from 1 to 7 as JSON.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too small to read long texts with notes"):
        al.generate_with_llm(prompt_paths=[p], model=str(tiny), device="cpu",
                              csv_path=_long_corpus(tmp_path), text_cols=["text"],
                              id_cols=["text_id"], out_features_csv=tmp_path / "f" / "llm.csv",
                              max_new_tokens=4, long_texts="notes", verbose=False)


def test_an_interrupted_run_resumes_without_answering_a_text_twice(tiny, tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood from 1 to 7 as JSON.\n", encoding="utf-8")
    src = _texts(tmp_path, n=8)
    answered: list = []

    class Dies(gl.LocalGenerator):
        def generate(self, conversations, **kw):
            texts = [c.user for c in conversations]
            answered.extend(texts)
            if any("very happy people" in t for t in texts) and not getattr(self, "revived", False):
                raise KeyboardInterrupt
            return super().generate(conversations, **kw)

    state = {"revived": False}

    def fresh(*a, **k):
        gen = Dies(str(tiny), device="cpu", batch_size=2, max_new_tokens=4)
        gen.revived = state["revived"]
        return gen

    monkeypatch.setattr(al, "generator_for", fresh)
    kwargs = dict(prompt_paths=[p], model=str(tiny), device="cpu", csv_path=src,
                  text_cols=["text"], id_cols=["text_id"],
                  out_features_csv=tmp_path / "f" / "llm.csv", max_new_tokens=4, verbose=False)
    with pytest.raises(KeyboardInterrupt):
        al.generate_with_llm(parse_json=True, **kwargs)
    part = tmp_path / "f" / "llm.csv.part"
    assert part.is_file() and not (tmp_path / "f" / "llm.csv").exists()
    kept = list(csv.DictReader(part.open(encoding="utf-8-sig")))
    assert 0 < len(kept) < 8
    first_pass = list(answered)
    state["revived"] = True
    answered.clear()
    out = al.generate_with_llm(parse_json=True, **kwargs)
    rows = _read(out)
    assert [r["text_id"] for r in rows] == [f"t{i}" for i in range(8)]
    assert not part.exists() and not (tmp_path / "f" / "llm_partial.json").exists()
    # the second pass answered only what the first had not written
    assert len(answered) + len(kept) <= 8 + 1, "one smoke reply at most on top"
    assert not any(t in first_pass[:len(kept)] for t in answered[1:])


def test_append_table_starts_over_when_the_settings_changed(tmp_path):
    ref = tmp_path / "t.csv"
    with tables.append_table(ref, ["text_id", "x"], settings={"v": 1}) as (w, done):
        assert done == set()
        w.writerow({"text_id": "a", "x": 1})
        w.flush()
        raise_later = True
    assert ref.is_file() and raise_later
    # a run that stops keeps its scratch; the same settings pick it up
    ref2 = tmp_path / "u.csv"
    try:
        with tables.append_table(ref2, ["text_id", "x"], settings={"v": 1}) as (w, done):
            w.writerow({"text_id": "a", "x": 1})
            w.flush()
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    with tables.append_table(ref2, ["text_id", "x"], settings={"v": 1}) as (w, done):
        assert done == {"a"}
        w.writerow({"text_id": "b", "x": 2})
    assert [r["text_id"] for r in _read(ref2)] == ["a", "b"]
    # different settings: the scratch is not trusted
    try:
        with tables.append_table(ref2, ["text_id", "x"], settings={"v": 2}) as (w, done):
            w.writerow({"text_id": "c", "x": 3})
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    with tables.append_table(ref2, ["text_id", "x"], settings={"v": 3}) as (w, done):
        assert done == set()


# ---------------------------------------------------------------------------
# several copies of the model, on CPU workers here
# ---------------------------------------------------------------------------

def test_two_workers_answer_the_same_as_one_and_in_order(tiny):
    """Two processes, a copy of the tiny model in each, answer one call's
    conversations between them; the replies come back in order and equal
    what one copy says (greedy decoding), and a dead worker's chunk is
    answered by the other."""
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    par = gl.ParallelGenerator(str(tiny), [None, None], batch_size=2, max_new_tokens=4,
                               plan=plan)
    try:
        assert par.batch_size == 4 and "2 copies" in par.describe()
        convs = [Conversation(user=t) for t in ("the cat sat", "a dog ran", "potato gravy",
                                                "the meeting", "bad budget")]
        got = par.generate(convs, max_new_tokens=4)
        one = gl.LocalGenerator(str(tiny), device="cpu", batch_size=4, max_new_tokens=4)
        assert got == one.generate(convs, max_new_tokens=4)
        # a worker that dies mid-run: its chunk is handed to the other
        par.workers[0].terminate()
        par.workers[0].join()
        par.jobs[0] = par.jobs[1]       # a job sent to the dead worker lands on the live one
        got2 = par.generate(convs[:2], max_new_tokens=4)
        assert got2 == one.generate(convs[:2], max_new_tokens=4)
    finally:
        par.close()


def test_a_worker_killed_with_a_chunk_in_its_own_queue_does_not_hang_the_call(tiny):
    """The real hazard on a server: a worker process is killed (the system's
    memory killer, a crash in a kernel) while a chunk sits in *its* queue.
    Nothing will ever answer that chunk, and the parent used to wait for it
    forever. Now it notices the dead process and hands the chunk on."""
    import threading

    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    par = gl.ParallelGenerator(str(tiny), [None, None], batch_size=2, max_new_tokens=4,
                               plan=plan)
    try:
        convs = [Conversation(user=t) for t in ("the cat sat", "a dog ran", "potato gravy",
                                                "the meeting", "bad budget")]
        par.workers[0].terminate()
        par.workers[0].join()
        out: list = []
        t = threading.Thread(target=lambda: out.append(par.generate(convs, max_new_tokens=4)),
                             daemon=True)
        t.start()
        t.join(90)
        assert not t.is_alive(), "the call hung on the dead worker"
        one = gl.LocalGenerator(str(tiny), device="cpu", batch_size=4, max_new_tokens=4)
        assert out == [one.generate(convs, max_new_tokens=4)]
        assert par._dead == {0}
        # later calls skip the dead worker outright instead of waiting on it
        t0 = __import__("time").time()
        assert par.generate(convs[:2], max_new_tokens=4) == one.generate(convs[:2],
                                                                         max_new_tokens=4)
        assert __import__("time").time() - t0 < 60
    finally:
        par.close()


def test_every_worker_dead_is_an_error_not_a_wait(tiny):
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    par = gl.ParallelGenerator(str(tiny), [None], batch_size=2, max_new_tokens=4, plan=plan)
    try:
        par.workers[0].terminate()
        par.workers[0].join()
        with pytest.raises(RuntimeError, match="died|every worker"):
            par.generate([Conversation(user="x")], max_new_tokens=4)
        with pytest.raises(RuntimeError, match="every worker has died"):
            par.generate([Conversation(user="x")], max_new_tokens=4)
    finally:
        par.close()


def test_the_parent_counts_tokens_the_way_the_workers_read_them(tiny):
    """The step sizes cuts and budgets with the parent's count; a word count
    stood in before, and a cut sized on words can overflow the model."""
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    par = gl.ParallelGenerator(str(tiny), [None], batch_size=2, max_new_tokens=4, plan=plan)
    try:
        one = gl.LocalGenerator(str(tiny), device="cpu", batch_size=1, max_new_tokens=4)
        msgs = [{"role": "system", "content": "be brief"},
                {"role": "user", "content": "the cat sat on the mat, didn't it? " * 20}]
        assert par.count_tokens(msgs) == one.count_tokens(msgs)
        assert par.thinking is False, "the smoke test reads this on every backend"
    finally:
        par.close()


def test_worker_device_ids_go_through_the_parents_own_visible_devices():
    """torch numbers what it can see from 0: under CUDA_VISIBLE_DEVICES=2,3
    the parent's card 1 is the machine's 3, and a worker told "1" would
    take a card the parent was never given."""
    assert gl._physical_ids([[0], [1]], visible="2,3") == [["2"], ["3"]]
    assert gl._physical_ids([[0, 1], [2, 3]], visible="4,5,6,7") == [["4", "5"], ["6", "7"]]
    assert gl._physical_ids([[0], [1]], visible="") == [["0"], ["1"]]
    assert gl._physical_ids([[1], None], visible="GPU-aaaa, GPU-bbbb") == [["GPU-bbbb"], None]


def test_the_batch_is_sized_from_the_card_the_model_is_on():
    assert gl._device_index("cuda:1") == 1
    assert gl._device_index("cuda") == 0


def test_devices_for_a_model_that_needs_a_gpu_is_empty_on_the_cpu(tiny):
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    assert gl.devices_for(plan) == []


# ---------------------------------------------------------------------------
# the layout over several GPUs: copies, shards, copies of shards
# ---------------------------------------------------------------------------

def _facts_7b():
    cfg = _llama_config(7)
    return gl.ModelFacts("m", "m", "hub", cfg, gl.estimate_params(cfg), False)


def test_a_model_that_fits_one_card_gets_a_copy_on_every_card_that_fits(monkeypatch):
    monkeypatch.setattr(gl, "model_facts", lambda *a, **k: _facts_7b())
    monkeypatch.setattr(gl, "_bf16_capable", lambda: True)
    monkeypatch.setattr(gl, "_sharding_available", lambda: True)
    cards = [(22 * GB, 24 * GB), (22 * GB, 24 * GB), (3 * GB, 24 * GB), (22 * GB, 24 * GB)]
    plan = gl.plan_load("m", gpus=cards, ram=(64 * GB, 128 * GB))
    assert plan.device == "cuda:0" and plan.groups == [[0], [1], [3]]
    assert "3 GPUs, a copy on each" in plan.describe()
    assert gl.devices_for(plan) == [[0], [1], [3]]


def test_a_model_that_fits_only_several_cards_is_sharded_and_copied(monkeypatch):
    """A 40B-shaped model (80 GB at bf16) over eight 24 GB cards: four
    cards hold it, so two copies run, each sharded across four."""
    cfg = dict(_llama_config(7), hidden_size=8192, num_hidden_layers=40,
               num_attention_heads=64, num_key_value_heads=8, intermediate_size=28672)
    big = gl.ModelFacts("big", "big", "hub", cfg, gl.estimate_params(cfg), False)
    monkeypatch.setattr(gl, "model_facts", lambda *a, **k: big)
    monkeypatch.setattr(gl, "_bf16_capable", lambda: True)
    monkeypatch.setattr(gl, "_sharding_available", lambda: True)
    monkeypatch.setattr(gl, "_quantization_available", lambda: False)
    cards = [(23 * GB, 24 * GB)] * 8
    plan = gl.plan_load("big", gpus=cards, ram=(64 * GB, 128 * GB))
    assert plan.device == "shards" and not plan.refusal
    assert len(plan.groups) == 2 and all(len(g) == 4 for g in plan.groups)
    assert "2 copies, each sharded across 4 GPUs" in plan.describe()
    assert any("sharded across 4" in n for n in plan.notes)
    # two cards: one sharded copy, one process
    plan2 = gl.plan_load("big", gpus=cards[:5], ram=(64 * GB, 128 * GB))
    assert plan2.device == "shards" and len(plan2.groups) == 1
    assert "sharded across 4 GPUs" in plan2.describe()
    # without accelerate there is no sharding, and the refusal says so
    monkeypatch.setattr(gl, "_sharding_available", lambda: False)
    plan3 = gl.plan_load("big", gpus=cards, ram=(64 * GB, 128 * GB))
    assert "GPUs together" in plan3.refusal and "accelerate" in plan3.refusal


def test_cards_too_small_even_together_are_refused(monkeypatch):
    monkeypatch.setattr(gl, "model_facts", lambda *a, **k: _facts_7b())
    monkeypatch.setattr(gl, "_bf16_capable", lambda: True)
    monkeypatch.setattr(gl, "_sharding_available", lambda: True)
    monkeypatch.setattr(gl, "_quantization_available", lambda: False)
    plan = gl.plan_load("m", gpus=[(5 * GB, 8 * GB), (5 * GB, 8 * GB)], ram=(64 * GB, 128 * GB))
    assert "does not fit this GPU" in plan.refusal and "2 GPUs together" in plan.refusal


def test_device_groups_arithmetic():
    cards = [(10 * GB, 12 * GB), (10 * GB, 12 * GB), (10 * GB, 12 * GB)]
    assert gl.device_groups(4 * GB, 10 * 1024 ** 2, cards, can_shard=True) == [[0], [1], [2]]
    assert gl.device_groups(14 * GB, 10 * 1024 ** 2, cards, can_shard=True) == [[0, 1]]
    assert gl.device_groups(14 * GB, 10 * 1024 ** 2, cards, can_shard=False) == []
    assert gl.device_groups(40 * GB, 10 * 1024 ** 2, cards, can_shard=True) == []


def test_temperature_zero_is_greedy_whatever_the_model_recommends(tmp_path_factory):
    """A model saved with transformers 4.50 or later carries a recommendation
    (Qwen3's: sample at 0.6, top_k 20), and transformers put it in place of
    every setting left at its own default -- "do not sample" among them. A
    run at temperature 0 sampled, unseeded: one question, three answers (a
    real run). The model here recommends sampling hot; the answers must
    not move."""
    import json as _json

    folder = build(tmp_path_factory.mktemp("llm") / "hot")
    cfg = folder / "generation_config.json"
    doc = _json.loads(cfg.read_text(encoding="utf-8")) if cfg.is_file() else {}
    doc.update({"do_sample": True, "temperature": 1.8, "top_k": 50, "top_p": 1.0,
                "transformers_version": "4.57.0"})
    cfg.write_text(_json.dumps(doc), encoding="utf-8")
    gen = gl.LocalGenerator(str(folder), device="cpu", max_new_tokens=12, verbose=False)
    if not gen._exactly_as_given:
        pytest.skip("this transformers does not replace defaults")
    conv = [Conversation(user="the cat sat on the mat")]
    replies = {gen.generate(conv, max_new_tokens=12, temperature=0.0)[0] for _ in range(4)}
    assert len(replies) == 1


def test_a_streamed_reply_is_the_same_reply_as_one_generated_whole(tiny):
    """Chat streams; greedy decoding means the pieces join up to exactly the
    reply the step would have written."""
    gen = gl.LocalGenerator(str(tiny), device="cpu", batch_size=1, max_new_tokens=12)
    conv = Conversation(prefix=[("system", "answer briefly")], user="the cat sat on")
    (whole,) = gen.generate([conv], max_new_tokens=12)
    pieces = list(gen.stream(conv, max_new_tokens=12))
    assert "".join(pieces).strip() == whole.strip()


def test_a_reply_left_partway_stops_the_model(tiny):
    """Ctrl-C in a chat abandons the stream; the model must stop writing
    rather than carry on to the cap underneath the next reply."""
    import threading

    gen = gl.LocalGenerator(str(tiny), device="cpu", batch_size=1, max_new_tokens=400)
    before = threading.active_count()
    stream = gen.stream(Conversation(user="the cat sat on"), max_new_tokens=400)
    next(stream)
    stream.close()
    assert threading.active_count() == before, "the generating thread was left running"


def test_decoding_settings_for_a_chat_and_the_step_left_as_it_was(tiny):
    gen = gl.LocalGenerator(str(tiny), device="cpu", batch_size=1, max_new_tokens=8)
    step = gen._config(8, 0.7, 0.95)
    assert step.top_k == 50 and step.repetition_penalty == 1.0, "the step: transformers' defaults"
    chat = gen._config(8, 0.7, 0.9, top_k=0, repetition_penalty=1.2)
    assert chat.top_k == 0 and chat.top_p == 0.9 and chat.repetition_penalty == 1.2
    greedy = gen._config(8, 0.0, 0.9, top_k=40, repetition_penalty=1.2)
    assert greedy.do_sample is False and greedy.repetition_penalty == 1.2
    pieces = list(gen.stream(Conversation(user="the cat"), max_new_tokens=8, temperature=0.9,
                             top_k=5, seed=3, repetition_penalty=1.3))
    assert pieces, "sampled with every setting, it still answers"


def test_the_plan_made_to_choose_a_backend_is_the_one_the_model_loads_with(tiny, monkeypatch):
    """generator_for plans to choose between one process and many; the model
    then planned again from scratch, a second trip to the hub every load."""
    from taters.text._generation import generator_for

    calls = []
    real = gl.plan_load
    monkeypatch.setattr(gl, "plan_load", lambda *a, **k: calls.append(1) or real(*a, **k))
    gen = generator_for("local", str(tiny), device="auto", max_new_tokens=4, batch_size=1)
    try:
        assert isinstance(gen, gl.LocalGenerator)
        assert len(calls) == 1, "planned once, and handed to the model"
    finally:
        gen.close()


def test_the_many_gpu_branch_reports_progress_the_way_the_step_listens(tiny, monkeypatch):
    """The step's callback takes (done, total, message). The branch that
    picks several GPUs called it with one argument: a TypeError that only
    a machine with several cards ever reached, before any model loaded."""
    from taters.text._generation import generator_for

    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    plan.device, plan.groups = "cuda:0", [[0], [1]]
    monkeypatch.setattr(gl, "plan_load", lambda *a, **k: plan)
    built = []

    class Stub:
        def __init__(self, model, groups, **kw):
            built.append((groups, kw.get("plan") is plan))

    monkeypatch.setattr(gl, "ParallelGenerator", Stub)
    heard = []
    gen = generator_for("local", str(tiny), max_new_tokens=4,
                        on_progress=lambda done, total, message: heard.append(message))
    assert isinstance(gen, Stub) and built == [([[0], [1]], True)]
    assert any(m.startswith("loading") for m in heard)


def test_the_one_time_download_carries_the_steps_token_and_progress(tiny, monkeypatch):
    """A gated model refuses without the token; the helper used to try
    without it, fail silently, and leave N workers downloading at once."""
    from taters.helpers import hub_models as hm
    from taters.text import _generation as g

    seen = {}
    monkeypatch.setattr(hm, "is_downloaded", lambda name, cache=None: False)

    def plan_download(name, token=None):
        seen["plan_token"] = token
        return hm.DownloadPlan(name, files=["model.safetensors"], size=4_000)

    monkeypatch.setattr(hm, "plan_download", plan_download)

    def download(plan, *, on_bytes=None, every=0.5, token=None):
        seen["download_token"] = token
        if on_bytes:
            on_bytes(2_000)
        return Path("/cache/x")

    monkeypatch.setattr(hm, "download", download)
    heard = []
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    plan.facts.kind = "hub"
    g._fetch_once("org/gated", plan, lambda d, t, m: heard.append((d, t)), "hf_secret")
    assert seen == {"plan_token": "hf_secret", "download_token": "hf_secret"}
    assert (2_000, 4_000) in heard, "progress in bytes, against the size"


def test_a_failed_one_time_download_is_said_not_swallowed(tiny, monkeypatch):
    from taters.helpers import hub_models as hm
    from taters.text import _generation as g

    monkeypatch.setattr(hm, "is_downloaded", lambda name, cache=None: False)
    monkeypatch.setattr(hm, "plan_download", lambda name, token=None: hm.DownloadPlan(
        name, files=["model.safetensors"], size=1))

    def refuse(plan, **kw):
        raise RuntimeError("GatedRepoError: access denied")

    monkeypatch.setattr(hm, "download", refuse)
    heard = []
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    plan.facts.kind = "hub"
    g._fetch_once("org/gated", plan, lambda d, t, m: heard.append(m), None)
    assert any("each copy fetches for itself" in m for m in heard)


def test_a_late_answer_to_a_re_sent_chunk_is_used_not_thrown_away(tiny):
    """A slow worker's answer arriving after its chunk went to another is
    still the answer to those rows; whichever comes first fills them."""
    plan = gl.plan_load(str(tiny), device="cpu", gpu=(None, None), ram=(8 * GB, 16 * GB))
    par = gl.ParallelGenerator(str(tiny), [None], batch_size=2, max_new_tokens=4, plan=plan)
    try:
        par._superseded = {7: 9}
        par._serial = 9
        # the parent is told job 9 is pending; the answer comes labeled 7
        msgs = [("done", 7, ["late reply"])]
        par._next = lambda waiting: msgs.pop(0)
        par.jobs[0] = type("Q", (), {"put": lambda self, job: None})()
        par._serial = 8                     # send() will number the one chunk 9
        assert par.generate([Conversation(user="x")], max_new_tokens=4) == ["late reply"]
    finally:
        par.close()
