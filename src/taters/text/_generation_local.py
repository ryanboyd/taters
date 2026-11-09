"""
The local backend: a causal language model loaded here through transformers.

Nothing heavy happens before the plan. The model is resolved (a folder,
the cache, the hub), its size is read from the checkpoint's own metadata
or worked out from its config, the free GPU and CPU memory are measured,
and ``precision: auto`` picks the first of 16-bit, 4-bit (when
bitsandbytes is installed) and nothing that fits -- "nothing" being a
refusal that names the three fixes, never a load that spills to the CPU
unasked and runs for a day. The same plan sizes the batch from the memory
left after the weights, and the measured weights after loading size it
again. Then generation is the plain thing done carefully: the chat
template tokenized directly, left padding, batches sorted by length, a
fresh generation config (greedy unless asked), the GPU's out-of-memory
halving the batch, and the reasoning switch passed to templates that
have one.
"""
from __future__ import annotations

import json
import math
import os
import queue
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ._generation import Conversation, Generator

__all__ = ["LoadPlan", "plan_load", "model_facts", "estimate_params", "free_memory",
           "LocalGenerator", "ParallelGenerator", "devices_for", "device_groups",
           "all_gpu_memory", "PRECISIONS"]

PRECISIONS = ("auto", "bf16", "fp16", "fp32", "int8", "nf4")
GB = 1024 ** 3
#: what one parameter costs at each precision; the quantized ones carry
#: their scales, and the embeddings and output head stay at 16-bit
_BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0, "int8": 1.0, "nf4": 0.6}
_CUDA_CONTEXT = 0.5 * GB
#: what a card keeps back beyond the weights and the context: a 6 GB laptop
#: card refused a 3.2 GB model with a full gigabyte held in reserve, and
#: ran it fine. fragmentation needs a little; the OOM halving is the net
_RESERVE_SHARE = 0.05
_RESERVE_MIN = 256 * 1024 ** 2
#: the reply length the batch is planned on, whatever the cap
_TYPICAL_REPLY = 512


# ---------------------------------------------------------------------------
# Memory on this machine
# ---------------------------------------------------------------------------

def free_memory() -> Tuple[Optional[int], Optional[int]]:
    """``(available, total)`` RAM in bytes from the standard library alone,
    or ``None`` where the platform does not say."""
    try:
        if sys.platform.startswith("linux"):
            total = avail = None
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
            return avail, total
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = Status()
            status.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys), int(status.ullTotalPhys)
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * size
        except (ValueError, OSError):
            avail = None
        return avail, pages * size
    except Exception:
        return None, None


def gpu_memory(index: int = 0) -> Tuple[Optional[int], Optional[int]]:
    """``(free, total)`` bytes on one CUDA device, or ``None`` without one."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info(index)
        return int(free), int(total)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# What the model is
# ---------------------------------------------------------------------------

def estimate_params(config: Dict[str, Any]) -> int:
    """
    Parameters from a config alone, within a few percent: embeddings,
    per-layer attention (grouped-query aware) and the MLP (gated, three
    matrices, unless the family is known to use two), the output head
    unless it is tied to the embeddings, experts multiplied in.
    """
    H = int(config.get("hidden_size") or config.get("n_embd") or 0)
    L = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    V = int(config.get("vocab_size") or 0)
    heads = int(config.get("num_attention_heads") or config.get("n_head") or 1)
    kv = int(config.get("num_key_value_heads") or heads)
    d = int(config.get("head_dim") or (H // max(1, heads)))
    inter = int(config.get("intermediate_size") or config.get("n_inner") or 4 * H)
    gated = str(config.get("model_type", "")) not in ("gpt2", "opt", "phi", "gpt_neox", "bloom")
    mlp = (3 if gated else 2) * H * inter
    experts = int(config.get("num_local_experts") or config.get("num_experts") or 1)
    attention = H * heads * d + 2 * H * kv * d + heads * d * H
    per_layer = attention + mlp * experts + 2 * H
    head = 0 if config.get("tie_word_embeddings", False) else V * H
    return V * H + L * per_layer + head


def kv_bytes_per_token(config: Dict[str, Any], dtype_bytes: int = 2) -> int:
    L = int(config.get("num_hidden_layers") or config.get("n_layer") or 0)
    heads = int(config.get("num_attention_heads") or config.get("n_head") or 1)
    kv = int(config.get("num_key_value_heads") or heads)
    H = int(config.get("hidden_size") or config.get("n_embd") or 0)
    d = int(config.get("head_dim") or (H // max(1, heads)))
    return 2 * L * kv * d * dtype_bytes


def context_of(config: Dict[str, Any]) -> Optional[int]:
    scaling = config.get("rope_scaling") or {}
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        if config.get(key):
            n = int(config[key])
            original = (scaling.get("original_max_position_embeddings")
                        if isinstance(scaling, dict) else None)
            return max(n, int(original)) if original else n
    return None


def _params_from_safetensors(folder: Path, config: Dict[str, Any]) -> Optional[int]:
    """The exact count from a checkpoint folder: the index's total bytes
    over the checkpoint dtype, or a single file's header summed."""
    def counted(file: Path) -> int:
        with file.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n).decode("utf-8"))
        total = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            shape = entry.get("shape") or []
            total += math.prod(int(s) for s in shape) if shape else 0
        return total

    index = folder / "model.safetensors.index.json"
    try:
        if index.is_file():
            doc = json.loads(index.read_text(encoding="utf-8"))
            # the shards' own headers, not the index's byte total over the
            # config's dtype: a checkpoint saved in bf16 under a config that
            # says float32 would count half its parameters
            shards = sorted({str(f) for f in (doc.get("weight_map") or {}).values()})
            if shards and all((folder / f).is_file() for f in shards):
                return sum(counted(folder / f) for f in shards) or None
            return None
        single = folder / "model.safetensors"
        if single.is_file():
            return counted(single) or None
    except (OSError, ValueError, struct.error):
        return None
    return None


@dataclass
class ModelFacts:
    source: str                     # what from_pretrained is given
    label: str                      # how it is named to a person
    kind: str                       # "folder" | "cached" | "hub"
    config: Dict[str, Any]
    params: int
    exact: bool                     # counted from the checkpoint, not estimated
    problem: str = ""


def _hub_problem(name: str, cache: str, error: BaseException) -> str:
    """
    What to do about a model the hub would not hand over, in a sentence.

    The hub's own errors say gated or not found, but transformers wraps
    them in a plain OSError on the way out, so the error and everything it
    was raised from are looked at, and the message last: a gated Llama
    used to come out as "the hub could not be reached", which sent people
    to check a network that was fine (a real run).
    """
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:  # pragma: no cover - transformers needs it
        GatedRepoError = RepositoryNotFoundError = ()  # type: ignore[assignment]
    chain: List[BaseException] = []
    at: Optional[BaseException] = error
    while at is not None and at not in chain:
        chain.append(at)
        at = at.__cause__ or at.__context__
    text = " ".join(str(x) for x in chain).lower()
    login = "sign in with `hf auth login` (or set HF_TOKEN), and run again"
    if any(isinstance(x, GatedRepoError) for x in chain) or "gated repo" in text:
        return (f"{name} is gated: its owner has to approve you first. Open "
                f"https://huggingface.co/{name}, accept the license with your account, then "
                f"{login}. Or pick a model that needs no approval, such as Qwen/Qwen3-1.7B")
    if any(isinstance(x, RepositoryNotFoundError) for x in chain) or "repository not found" in text:
        return (f"no model called {name!r} on the Hugging Face hub; check the name "
                f"(owner/model), or give a checkpoint folder")
    if "401" in text or "unauthorized" in text:
        return f"the Hugging Face hub would not let this machine download {name}: {login}"
    return (f"{name} is not in the model cache ({cache}) and the hub could not be reached "
            f"({type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}); "
            f"connect to the internet, or download it first")


def model_facts(model: str, *, hf_token: Optional[str] = None) -> ModelFacts:
    """
    Resolve a model without loading it: a checkpoint folder, else the
    Hugging Face cache, else the hub. Every failure is a sentence that
    says what to do (a gated repository, a wrong name, no network).
    """
    from ..helpers.settings import model_cache_dir

    name = str(model or "").strip()
    if not name:
        return ModelFacts("", "", "hub", {}, 0, False,
                          problem="no language model was named (the `model` setting)")
    path = Path(name)
    if path.is_dir():
        cfg_path = path / "config.json"
        if not cfg_path.is_file():
            return ModelFacts(name, path.name, "folder", {}, 0, False,
                              problem=f"{path} holds no config.json, so it is not a "
                                      f"transformers checkpoint folder")
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except ValueError as e:
            return ModelFacts(name, path.name, "folder", {}, 0, False,
                              problem=f"{cfg_path} is not readable JSON ({e})")
        counted = _params_from_safetensors(path, config)
        return ModelFacts(name, path.name, "folder", config,
                          counted or estimate_params(config), counted is not None)
    cache = str(model_cache_dir())
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(name, "config.json", cache_dir=cache)
    except Exception:
        cached = None
    if isinstance(cached, str) and Path(cached).is_file():
        config = json.loads(Path(cached).read_text(encoding="utf-8"))
        counted = _params_from_safetensors(Path(cached).parent, config)
        return ModelFacts(name, name, "cached", config, counted or estimate_params(config),
                          counted is not None)
    try:
        from huggingface_hub import HfApi
        from transformers import AutoConfig

        info = HfApi().model_info(name, files_metadata=True, token=hf_token)
        # before anything downloads: see settle_links
        from ..helpers.hub_models import settle_links

        settle_links()
        config = AutoConfig.from_pretrained(name, cache_dir=cache, token=hf_token).to_dict()
    except Exception as e:  # noqa: BLE001 - every kind becomes a sentence
        return ModelFacts(name, name, "hub", {}, 0, False, problem=_hub_problem(name, cache, e))
    st = getattr(info, "safetensors", None)
    total = int(getattr(st, "total", 0) or 0) if st is not None else 0
    return ModelFacts(name, name, "hub", config, total or estimate_params(config), total > 0)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass
class LoadPlan:
    """What loading the model will cost and where it goes, decided before
    any weight is read. ``refusal`` is the sentence when nothing fits.
    ``groups`` is the layout over the machine's GPUs: one list per copy of
    the model, each the devices that copy is sharded across (``[[0], [1]]``
    is a copy per card; ``[[0, 1], [2, 3]]`` two copies, each over two)."""

    facts: ModelFacts
    precision: str = "bf16"
    device: str = "cpu"             # "cuda:0" | "cpu" | "shards" (across GPUs) | "auto" (offloaded)
    groups: List[List[int]] = field(default_factory=list)
    weights: int = 0                # bytes
    kv_per_token: int = 0
    context_length: Optional[int] = None
    gpu_free: Optional[int] = None
    gpu_total: Optional[int] = None
    ram_free: Optional[int] = None
    ram_total: Optional[int] = None
    batch: int = 1
    notes: List[str] = field(default_factory=list)
    refusal: str = ""

    def describe(self) -> str:
        if self.device == "shards":
            k = len(self.groups[0]) if self.groups else 2
            where = (f"{len(self.groups)} copies, each sharded across {k} GPUs"
                     if len(self.groups) > 1 else f"sharded across {k} GPUs")
        elif self.device.startswith("cuda"):
            where = (f"{len(self.groups)} GPUs, a copy on each" if len(self.groups) > 1
                     else "the GPU")
        elif self.device == "auto":
            where = "every device, spilling to the CPU"
        else:
            where = "the CPU"
        n = self.facts.params
        size = f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"
        return (f"{self.facts.label}: {size} parameters, {self.weights / GB:.1f} GB at "
                f"{self.precision}, on {where}; batch {int(self.batch)}"
                + (f"; free GPU memory {self.gpu_free / GB:.1f} of {self.gpu_total / GB:.1f} GB"
                   if self.gpu_free is not None else "")
                + (f"; free RAM {self.ram_free / GB:.1f} GB" if self.ram_free is not None else ""))


def _sharding_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("accelerate") is not None


def _quantization_available() -> bool:
    import importlib.util

    return (importlib.util.find_spec("bitsandbytes") is not None
            and importlib.util.find_spec("accelerate") is not None)


def _bf16_capable() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0))
    except Exception:
        return False


def all_gpu_memory() -> List[Tuple[int, int]]:
    """``(free, total)`` for every CUDA device, in index order."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [tuple(int(v) for v in torch.cuda.mem_get_info(i))
                for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def device_groups(weights: int, per_row: int, gpus: Sequence[Tuple[int, int]], *,
                  can_shard: bool) -> List[List[int]]:
    """
    How to lay one model out over these GPUs: ``[[i], [j], …]`` when a
    copy fits one card (every card that fits gets one), else
    ``[[a, b], [c, d]]``: the smallest number of cards whose free memory
    holds the weights, and as many such groups as the machine has, else
    nothing. A copy wants its weights, the CUDA context, a margin and a
    small batch's cache; a shard wants the context on every card.
    """
    margin = [max(_RESERVE_MIN, int(_RESERVE_SHARE * total)) for _free, total in gpus]
    need_one = [weights + _CUDA_CONTEXT + margin[i] + per_row for i in range(len(gpus))]
    singles = [i for i, (free, _t) in enumerate(gpus) if free >= need_one[i]]
    if singles:
        return [[i] for i in singles]
    if not can_shard or len(gpus) < 2:
        return []
    # the usable memory per card once its context and margin are set aside
    usable = {i: free - _CUDA_CONTEXT - margin[i] for i, (free, _t) in enumerate(gpus)}
    order = sorted(usable, key=lambda i: -usable[i])
    for k in range(2, len(gpus) + 1):
        if sum(usable[i] for i in order[:k]) >= weights + per_row:
            break
    else:
        return []
    groups: List[List[int]] = []
    pool = list(order)
    while len(pool) >= k:
        group = pool[:k]
        if sum(usable[i] for i in group) < weights + per_row:
            break
        groups.append(sorted(group))
        pool = pool[k:]
    return groups


def weights_bytes(facts: ModelFacts, precision: str) -> int:
    cfg = facts.config
    V = int(cfg.get("vocab_size") or 0)
    H = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    embeddings = V * H * (1 if cfg.get("tie_word_embeddings", False) else 2)
    body = max(0, facts.params - embeddings)
    per = _BYTES_PER_PARAM[precision]
    head_per = 4.0 if precision == "fp32" else 2.0
    return int(body * per + embeddings * head_per)


def batch_for(plan_free: int, weights: int, total: Optional[int], kv_per_token: int,
              prompt_tokens: int, max_new_tokens: int, config: Dict[str, Any],
              cap: int = 64) -> int:
    """How many conversations fit beside the weights: the memory left after
    the weights, the CUDA context and a margin, over what one row costs
    (its cache for prompt plus reply, and the prefill's largest activation)."""
    margin = max(_RESERVE_MIN, int(_RESERVE_SHARE * (total or plan_free)))
    budget = plan_free - weights - _CUDA_CONTEXT - margin
    inter = int(config.get("intermediate_size") or 4 * int(config.get("hidden_size") or 1))
    per_row = kv_per_token * (prompt_tokens + max_new_tokens) + prompt_tokens * 3 * inter * 2
    if budget <= 0 or per_row <= 0:
        return 1
    return max(1, min(cap, budget // per_row))


def plan_load(model: str, *, precision: str = "auto", device: str = "auto",
              max_new_tokens: int = 256, prompt_tokens: int = 512, offload: bool = False,
              batch_size: int = 0, hf_token: Optional[str] = None,
              gpu: Optional[Tuple[Optional[int], Optional[int]]] = None,
              ram: Optional[Tuple[Optional[int], Optional[int]]] = None,
              gpus: Optional[Sequence[Tuple[int, int]]] = None) -> LoadPlan:
    """
    Decide precision, device and batch before loading.

    ``auto`` precision is 16-bit when the weights plus a small batch fit the
    GPU's free memory, else 4-bit when bitsandbytes is installed and that
    fits, else a refusal naming the fixes; a named precision is tried as
    named and refused the same way. ``device: cpu`` plans for RAM at
    16-bit (fp32 when asked) with a speed note. ``offload`` turns the
    refusal into a sharded load that spills to the CPU, since the owner of
    the machine asked for it. ``gpu`` and ``ram`` override the measurement
    (tests plan machines they do not have).
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, not {precision!r}")
    facts = model_facts(model, hf_token=hf_token)
    plan = LoadPlan(facts=facts)
    if facts.problem:
        plan.refusal = facts.problem
        return plan
    cfg = facts.config
    plan.context_length = context_of(cfg)
    plan.kv_per_token = kv_bytes_per_token(cfg, 2)
    if gpus is not None and gpu is None:
        gpu = gpus[0] if gpus else (None, None)
    gpu_free, gpu_total = gpu if gpu is not None else gpu_memory(0)
    ram_free, ram_total = ram if ram is not None else free_memory()
    plan.gpu_free, plan.gpu_total, plan.ram_free, plan.ram_total = (
        gpu_free, gpu_total, ram_free, ram_total)
    want_gpu = device != "cpu" and gpu_free is not None
    if device == "cuda" and gpu_free is None:
        plan.refusal = "device='cuda' was asked for and no CUDA device is usable"
        return plan
    sixteen = "bf16" if _bf16_capable() else "fp16"
    if sixteen == "fp16" and str(cfg.get("torch_dtype")) == "bfloat16" and want_gpu:
        plan.notes.append("the checkpoint is bfloat16 and this GPU has no bfloat16, so it "
                          "runs in float16; a few models overflow there (answers become "
                          "blank), in which case use precision fp32 or a newer GPU")
    if not want_gpu:
        plan.device = "cpu"
        plan.precision = "fp32" if precision == "fp32" else ("bf16" if precision in
                                                             ("auto", "bf16", "fp16") else precision)
        if plan.precision in ("int8", "nf4"):
            plan.refusal = "4-bit and 8-bit loading need a CUDA GPU; on the CPU use auto"
            return plan
        if precision == "fp16":
            plan.notes.append("fp16 is not a CPU format; running in bf16 instead")
        plan.weights = weights_bytes(facts, plan.precision)
        if ram_free is not None and plan.weights * 1.3 > ram_free:
            plan.refusal = (f"{facts.label} needs about {plan.weights / GB:.1f} GB of RAM at "
                            f"{plan.precision} and {ram_free / GB:.1f} GB is free. Use a "
                            f"smaller model, or a machine with a GPU")
            return plan
        plan.batch = int(batch_size) if batch_size else 4
        if facts.params > 4e9:
            plan.notes.append("a model this size on the CPU answers a text in tens of "
                              "seconds; a 1-4B model or a GPU is the practical choice")
        else:
            plan.notes.append("running on the CPU: expect a few seconds per text")
        return plan
    ladder = ([sixteen, "nf4"] if precision == "auto" else
              [sixteen if precision in ("bf16", "fp16") else precision])
    # the cards: as given (a test planning a machine it does not have), the
    # one card given, or the machine's
    if gpus is not None:
        cards = list(gpus)
    elif gpu is not None:
        cards = [(gpu_free, gpu_total or gpu_free)]
    else:
        cards = all_gpu_memory() or [(gpu_free, gpu_total or gpu_free)]
    # a cap is generous on purpose and rarely reached; the batch is sized
    # on a typical reply, and the out-of-memory halving covers the rest
    typical = min(int(max_new_tokens), _TYPICAL_REPLY)
    per_row = plan.kv_per_token * (prompt_tokens + typical)
    for choice in ladder:
        if choice in ("nf4", "int8") and not _quantization_available():
            if precision != "auto":
                plan.refusal = (f"{choice} loading needs bitsandbytes and accelerate: "
                                f"pip install \"taters[llm]\"")
                return plan
            continue
        weights = weights_bytes(facts, choice)
        groups = device_groups(weights, per_row, cards,
                               can_shard=choice not in ("nf4", "int8") and _sharding_available())
        if not groups:
            continue
        plan.precision, plan.weights, plan.groups = choice, weights, groups
        if len(groups[0]) == 1:
            first = groups[0][0]
            free, total = cards[first]
            plan.device, plan.gpu_free, plan.gpu_total = f"cuda:{first}", free, total
            plan.batch = int(batch_size) if batch_size else int(batch_for(
                free, weights, total, plan.kv_per_token, prompt_tokens, typical, cfg))
        else:
            plan.device = "shards"
            pooled = sum(cards[i][0] for i in groups[0])
            plan.gpu_free = pooled
            plan.gpu_total = sum(cards[i][1] for i in groups[0])
            plan.batch = int(batch_size) if batch_size else int(batch_for(
                pooled, weights, plan.gpu_total, plan.kv_per_token, prompt_tokens,
                typical, cfg))
            plan.notes.append(f"the model does not fit one GPU, so it is sharded across "
                              f"{len(groups[0])} (layers on each, one working at a time)")
        if choice == "nf4":
            plan.notes.append("loaded in 4-bit (nf4) to fit the GPU; answers may differ "
                              "slightly from the 16-bit model")
        return plan
    weights16 = weights_bytes(facts, sixteen)
    if offload:
        plan.precision, plan.device, plan.weights = sixteen, "auto", weights16
        plan.batch = int(batch_size) if batch_size else 1
        plan.notes.append("the model does not fit the GPU, so layers spill to the CPU "
                          "(offload): expect it to be many times slower")
        return plan
    fixes = [f"a smaller model (this one needs about {weights16 / GB:.1f} GB at {sixteen}, "
             f"and the GPU has {gpu_free / GB:.1f} GB free"
             + (f"; {len(cards)} GPUs together {sum(c[0] for c in cards) / GB:.1f} GB)"
                if len(cards) > 1 else ")")]
    if len(cards) > 1 and not _sharding_available():
        fixes.append("pip install \"taters[llm]\" for accelerate, which lets the model be "
                     "sharded across your GPUs")
    if not _quantization_available():
        fixes.append("pip install \"taters[llm]\" for 4-bit loading, which needs about "
                     f"{weights_bytes(facts, 'nf4') / GB:.1f} GB")
    fixes.append("device: cpu with enough RAM, or offload: true, both slow")
    plan.refusal = f"{facts.label} does not fit this GPU. Fixes: " + "; ".join(fixes)
    return plan


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

def _tokenizer_for(source: str, kind: str, hf_token: Optional[str], thinking: bool):
    """The model's tokenizer and the template switches it takes: shared by
    the one-process backend and the parent of the many-process one, so
    both count tokens the same way."""
    from transformers import AutoTokenizer

    from ..helpers.hub_models import settle_links
    from ..helpers.settings import model_cache_dir

    cache: Dict[str, Any] = {} if kind == "folder" else {"cache_dir": str(model_cache_dir())}
    if hf_token:
        cache["token"] = hf_token
    # before anything downloads: see settle_links
    settle_links()
    tokenizer = AutoTokenizer.from_pretrained(source, **cache)
    template = tokenizer.chat_template or ""
    kwargs = {"enable_thinking": bool(thinking)} if "enable_thinking" in template else {}
    return tokenizer, kwargs


def _template_ids(tokenizer, messages: Sequence[Dict[str, str]],
                  template_kwargs: Dict[str, Any]) -> List[int]:
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, return_dict=False,
            **template_kwargs)
        return list(ids) if not isinstance(ids, dict) else list(ids["input_ids"])
    # no template: the plain fallback every chat model understands least badly
    text = "\n\n".join(
        (f"{m['role'].capitalize()}: {m['content']}" if m["role"] != "system"
         else m["content"]) for m in messages) + "\n\nAssistant:"
    return list(tokenizer(text)["input_ids"])


def _device_index(device: str) -> int:
    """``cuda:1`` -> 1; a bare ``cuda`` is card 0."""
    return int(device.split(":", 1)[1]) if ":" in device else 0


class LocalGenerator(Generator):
    kind = "transformers"

    def __init__(self, model: str, *, device: str = "auto", precision: str = "auto",
                 offload: bool = False, batch_size: int = 0, hf_token: Optional[str] = None,
                 max_new_tokens: int = 256, thinking: bool = False, verbose: bool = False,
                 on_progress: Optional[Callable[..., None]] = None,
                 prompt_tokens: int = 512, plan: Optional[LoadPlan] = None):
        from ..helpers.progress import announce
        from ._transformer_common import set_threads, torch_missing_reason

        missing = torch_missing_reason()
        if missing:
            raise ImportError(missing)
        import torch
        from transformers import AutoModelForCausalLM

        from ..helpers.settings import model_cache_dir

        # the plan as handed in (generator_for made one to choose a backend;
        # making it again meant a second trip to the hub), else made here
        self.plan = plan if plan is not None else plan_load(
            model, precision=precision, device=device, max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens, offload=offload, batch_size=batch_size,
            hf_token=hf_token)
        if self.plan.refusal:
            raise ValueError(self.plan.refusal)
        facts = self.plan.facts
        self.model_label = facts.label
        self.context_length = self.plan.context_length
        self.thinking = thinking
        self.verbose = verbose
        announce(on_progress, "loading " + self.plan.describe())
        if verbose:
            print(f"[llm] {self.plan.describe()}")
            for note in self.plan.notes:
                print(f"[llm] {note}")
        set_threads(0)
        cache = {} if facts.kind == "folder" else {"cache_dir": str(model_cache_dir())}
        if hf_token:
            cache["token"] = hf_token
        self.tokenizer, self.template_kwargs = _tokenizer_for(facts.source, facts.kind,
                                                              hf_token, thinking)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # the template gives ids and `pad` lines them up; the tokenizer's
        # advice to call it on text instead does not apply, and printed once
        try:
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        except Exception:
            pass
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            self.plan.precision, torch.bfloat16)
        kwargs: Dict[str, Any] = dict(torch_dtype=dtype, low_cpu_mem_usage=True, **cache)
        try:
            kwargs["attn_implementation"] = "sdpa"
        except Exception:
            pass
        if self.plan.precision in ("nf4", "int8"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
                if self.plan.precision == "nf4" else BitsAndBytesConfig(load_in_8bit=True))
            # on the card the plan chose: card 0 may be the busy one
            kwargs["device_map"] = {"": _device_index(self.plan.device)
                                    if self.plan.device.startswith("cuda") else 0}
        elif self.plan.device == "auto":
            kwargs["device_map"] = "auto"
        elif self.plan.device == "shards":
            # the layers spread over the cards of this copy's group and nowhere
            # else: a spill to the CPU is what `offload` is for
            group = self.plan.groups[0] if self.plan.groups else list(range(len(all_gpu_memory())))
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = {**{i: int(free - _CUDA_CONTEXT
                                               - max(_RESERVE_MIN, int(_RESERVE_SHARE * total)))
                                       for i, (free, total) in enumerate(all_gpu_memory())
                                       if i in group},
                                    "cpu": 0}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(facts.source, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(facts.source, **kwargs)
        if "device_map" not in kwargs:
            self.model.to(self.plan.device)
        self.model.eval()
        self.device_name = self.plan.device
        self.reason = "; ".join(self.plan.notes)
        # the weights as loaded, not as planned: size the batch from the truth
        if self.plan.device.startswith("cuda") and len(self.plan.groups) <= 1:
            try:
                # the card the model is on: on a machine where card 0 is
                # busy the plan picks another, and card 0's memory says
                # nothing about this one's
                card = _device_index(self.plan.device)
                allocated = int(torch.cuda.memory_allocated(card))
                free, total = gpu_memory(card)
                if free is not None and not batch_size:
                    # `free` already excludes the CUDA context that batch_for
                    # subtracts, so it is handed back here
                    self.plan.batch = batch_for(free + allocated + _CUDA_CONTEXT, allocated,
                                                total, self.plan.kv_per_token, prompt_tokens,
                                                min(int(max_new_tokens), _TYPICAL_REPLY),
                                                facts.config)
                self.plan.weights = allocated or self.plan.weights
            except Exception:
                pass
        self.batch_size = max(1, int(self.plan.batch))
        self.max_new_tokens = int(max_new_tokens)

    def describe(self) -> str:
        return (f"transformers {self.plan.describe()}"
                + ("; thinking on" if self.template_kwargs.get("enable_thinking") else ""))

    def _config(self, max_new_tokens: int, temperature: float, top_p: float,
                top_k: Optional[int] = None, repetition_penalty: float = 1.0):
        """The decoding settings for a call: greedy at temperature 0, sampled
        above it -- the same for a batch and for a streamed reply. top-k only
        means something when sampling; the repetition penalty applies either
        way. top_k None leaves transformers' own default (50), which is what
        the step has always had; a chat passes its setting, 0 meaning no limit."""
        from transformers import GenerationConfig

        sampled = float(temperature) > 0
        extra: Dict[str, Any] = {}
        if sampled:
            extra.update(temperature=float(temperature), top_p=float(top_p))
            if top_k is not None:
                extra["top_k"] = int(top_k)
        if float(repetition_penalty) != 1.0:
            extra["repetition_penalty"] = float(repetition_penalty)
        return GenerationConfig(
            do_sample=sampled, max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.model.generation_config.eos_token_id, **extra)

    def set_thinking(self, on: bool) -> None:
        """Switch the model's thinking on or off for the calls that follow,
        where its chat template has the switch."""
        if "enable_thinking" in self.template_kwargs:
            self.template_kwargs["enable_thinking"] = bool(on)
        self.thinking = bool(on)

    def stream(self, conversation: Conversation, *, max_new_tokens: int,
               temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
               top_k: int = 0, repetition_penalty: float = 1.0) -> Iterator[str]:
        """
        One reply as it is written, through transformers' text streamer: the
        model generates on a thread of its own and the pieces come back as
        they are decoded. A failure on that thread ends the stream and is
        raised here, rather than leaving the reader waiting for a piece that
        will never come.
        """
        import threading

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        ids = self._ids(conversation.messages())
        inputs = torch.tensor([ids], device=self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        config = self._config(max_new_tokens, temperature, top_p, top_k, repetition_penalty)
        failed: List[BaseException] = []
        stop = threading.Event()

        class _Stop(StoppingCriteria):
            # Ctrl-C mid-reply leaves the reader; without this the model kept
            # writing the abandoned reply to the cap, under the next one
            def __call__(self, input_ids, scores, **kwargs):
                return stop.is_set()

        def work() -> None:
            try:
                if float(temperature) > 0:
                    torch.manual_seed(int(seed))
                with torch.inference_mode():
                    self.model.generate(input_ids=inputs,
                                        attention_mask=torch.ones_like(inputs),
                                        generation_config=config, streamer=streamer,
                                        stopping_criteria=StoppingCriteriaList([_Stop()]),
                                        **self._exactly_as_given)
            except BaseException as e:  # noqa: BLE001 - handed to the reader
                failed.append(e)
                streamer.end()

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            # finished, failed, or left (Ctrl-C, or the reader closed us):
            # either way the model stops before anything else is asked of it
            stop.set()
            worker.join(timeout=60)
        if failed:
            raise failed[0]

    @property
    def _exactly_as_given(self) -> Dict[str, Any]:
        """
        What tells ``generate`` to use the settings it is handed, and only
        those. From transformers 4.50, for a model saved by 4.50 or later,
        every setting left at transformers' own default is replaced by the
        model's recommendation -- and "do not sample" *is* the default, so
        Qwen3's ``do_sample=True, temperature=0.6, top_k=20`` came back:
        a run at temperature 0 sampled, unseeded, and the same question got
        three different answers (a real run). Older versions have neither
        the replacing nor the switch, so the switch is passed only where it
        exists.
        """
        import inspect

        try:
            known = "use_model_defaults" in inspect.signature(self.model.generate).parameters
        except (TypeError, ValueError):
            known = False
        return {"use_model_defaults": False} if known else {}

    def _ids(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        return _template_ids(self.tokenizer, messages, self.template_kwargs)

    def count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        return len(self._ids(messages))

    def generate(self, conversations: Sequence[Conversation], *, max_new_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 42
                 ) -> List[Optional[str]]:
        import torch

        from ._transformer_common import run_with_oom_fallback

        if not conversations:
            return []
        ids = [self._ids(c.messages()) for c in conversations]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        sampled = float(temperature) > 0
        config = self._config(max_new_tokens, temperature, top_p)
        replies: List[Optional[str]] = [None] * len(ids)

        def run(size: int) -> None:
            for start in range(0, len(order), size):
                chunk = order[start:start + size]
                batch = self.tokenizer.pad({"input_ids": [ids[i] for i in chunk]},
                                           padding_side="left", return_tensors="pt")
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                if sampled:
                    torch.manual_seed(int(seed) + start)
                with torch.inference_mode():
                    out = self.model.generate(**batch, generation_config=config,
                                              **self._exactly_as_given)
                width = batch["input_ids"].shape[1]
                texts = self.tokenizer.batch_decode(out[:, width:], skip_special_tokens=True)
                for i, text in zip(chunk, texts):
                    replies[i] = text.strip()
            # this size got through, so it's the size from now on; before
            # this every call paid the out-of-memory and the halving again
            self.batch_size = size

        run_with_oom_fallback(run, self.batch_size, verbose=self.verbose, what="a batch")
        return replies

    def close(self) -> None:
        try:
            import torch

            del self.model
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Several GPUs: a copy of the model on each, the rows shared between them
# ---------------------------------------------------------------------------

def devices_for(plan: LoadPlan) -> List[List[int]]:
    """The plan's layout when it means more than one process: a copy per
    group of cards. One group, or none, means one process."""
    return list(plan.groups) if plan.device == "shards" or plan.device.startswith("cuda") else []


def _physical_ids(devices: Sequence[Optional[Sequence[int]]],
                  visible: Optional[str] = None) -> List[Optional[List[str]]]:
    """
    What each worker's ``CUDA_VISIBLE_DEVICES`` should say for these torch
    indices. torch numbers the cards it can see from 0, so when the parent
    itself runs under ``CUDA_VISIBLE_DEVICES=2,3`` (a shared server), its
    card 1 is the machine's card 3, and a worker told "1" would take a
    card the parent was never given. ``visible`` stands in for the
    environment in tests.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    names = [n.strip() for n in str(raw).split(",") if n.strip()] if raw else []
    out: List[Optional[List[str]]] = []
    for group in devices:
        if group is None:
            out.append(None)
        else:
            out.append([names[i] if i < len(names) else str(i) for i in group])
    return out


def _worker(rank: int, devices: Optional[Sequence[str]], spec: Dict[str, Any], jobs,
            results) -> None:
    """One process: pin itself to its cards before torch is imported, load
    the model once (sharded across them when it takes more than one), then
    answer chunks of conversations until told to stop."""
    if devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    try:
        gen = LocalGenerator(spec["model"], device="cuda" if devices is not None else "cpu",
                             precision=spec["precision"], offload=False,
                             batch_size=spec["batch_size"], hf_token=spec["hf_token"],
                             max_new_tokens=spec["max_new_tokens"], thinking=spec["thinking"],
                             verbose=False, prompt_tokens=spec["prompt_tokens"])
    except BaseException as e:  # noqa: BLE001 - the parent reports it
        results.put(("dead", rank, f"{type(e).__name__}: {e}"))
        return
    results.put(("ready", rank, gen.batch_size))
    while True:
        job = jobs.get()
        if job is None:
            return
        index, conversations, kwargs = job
        try:
            replies = gen.generate(conversations, **kwargs)
            results.put(("done", index, replies))
        except BaseException as e:  # noqa: BLE001
            results.put(("failed", index, f"{type(e).__name__}: {e}"))


class ParallelGenerator(Generator):
    """
    The local backend over several GPUs at once.

    Each GPU gets its own process with its own copy of the model, pinned
    through ``CUDA_VISIBLE_DEVICES`` before torch loads, so a worker is an
    ordinary :class:`LocalGenerator` that believes it has one card. A
    call splits its conversations into one chunk per worker, hands them
    out, and puts the replies back in order; a chunk whose worker fails is
    tried once more on another before the error is raised. The batch size
    reported to the step is the sum of the workers', so one call keeps
    every card busy. Rows are never written here: the step writes, as
    always, in order.
    """

    kind = "transformers"

    def __init__(self, model: str, devices: Sequence[Optional[Sequence[int]]], *,
                 precision: str = "auto",
                 batch_size: int = 0, hf_token: Optional[str] = None, max_new_tokens: int = 256,
                 thinking: bool = False, prompt_tokens: int = 512, verbose: bool = False,
                 plan: Optional[LoadPlan] = None):
        import multiprocessing as mp

        self.plan = plan
        self.model_label = plan.facts.label if plan else str(model)
        self.context_length = plan.context_length if plan else None
        self.thinking = bool(thinking)
        self.devices = [None if d is None else list(d) for d in devices]
        self.device_name = ("cuda" if any(d is not None for d in self.devices) else "cpu")
        # the tokenizer here too, so the step's budgets and cuts are counted
        # the way the workers will read them; a word count stood in before,
        # and a cut sized on words can overflow the model on long texts
        self.template_kwargs: Dict[str, Any] = {}
        counted = ""
        try:
            source = plan.facts.source if plan else str(model)
            kind = plan.facts.kind if plan else "hub"
            self.tokenizer, self.template_kwargs = _tokenizer_for(source, kind, hf_token,
                                                                  thinking)
        except Exception as e:  # noqa: BLE001 - the word count is the fallback, and said
            self.tokenizer = None
            counted = (f"; token counts are estimates from word counts (the tokenizer could "
                       f"not be loaded here: {type(e).__name__})")
            if verbose:
                print(f"[llm] {counted[2:]}")
        self._serial = 0            # job ids, unique across calls
        self._dead: Set[int] = set()
        self._superseded: Dict[int, int] = {}   # a re-sent job's old id -> its new one
        width = max((len(d) for d in self.devices if d), default=1)
        self.reason = (f"{len(self.devices)} copies of the model, one per device" if width == 1
                       else f"{len(self.devices)} copies of the model, each sharded across "
                            f"{width} GPUs") + counted
        ctx = mp.get_context("spawn")
        self.jobs = [ctx.Queue() for _ in self.devices]
        self.results = ctx.Queue()
        spec = {"model": model, "precision": precision, "batch_size": batch_size,
                "hf_token": hf_token, "max_new_tokens": max_new_tokens, "thinking": thinking,
                "prompt_tokens": prompt_tokens}
        physical = _physical_ids(self.devices)
        self.workers = [ctx.Process(target=_worker, args=(r, d, spec, self.jobs[r], self.results),
                                    daemon=True) for r, d in enumerate(physical)]
        for w in self.workers:
            w.start()
        sizes: Dict[int, int] = {}
        loading = set(range(len(self.workers)))
        while loading:
            kind, rank, payload = self._next(lambda: loading)
            if kind in ("dead", "died"):
                self.close()
                raise RuntimeError(f"worker {rank} on device {self._where(rank)} could not "
                                   f"load the model: {payload}")
            if kind == "ready":
                sizes[rank] = int(payload)
                loading.discard(rank)
        self.per_worker = [sizes[r] for r in range(len(self.workers))]
        self.batch_size = max(1, sum(self.per_worker))

    def _where(self, rank: int) -> str:
        d = self.devices[rank]
        return "cpu" if d is None else "cuda:" + "+".join(str(i) for i in d)

    def _next(self, waiting_on: Callable[[], Set[int]]) -> Tuple[str, Any, Any]:
        """
        The next message from a worker. A worker that dies with work
        outstanding (killed by the system for memory, a crash inside a CUDA
        kernel) never sends, so without this the parent waited on an empty
        queue forever -- an unattended overnight run that hangs instead of
        failing. The wait is timed, and a dead process is reported as the
        message it will never send.
        """
        while True:
            try:
                return self.results.get(timeout=1.0)
            except queue.Empty:
                for r in sorted(waiting_on()):
                    if r not in self._dead and not self.workers[r].is_alive():
                        self._dead.add(r)
                        code = self.workers[r].exitcode
                        return ("died", r,
                                f"worker {r} on {self._where(r)} died (exit code {code}): "
                                f"killed, most likely for running out of memory")

    def describe(self) -> str:
        where = ", ".join(self._where(r) for r in range(len(self.devices)))
        return (f"transformers, {len(self.devices)} copies of {self.model_label} on {where}, "
                f"{self.batch_size} conversations per round")

    def count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        if self.tokenizer is not None:
            return len(_template_ids(self.tokenizer, messages, self.template_kwargs))
        # no tokenizer could be loaded here: words times a bit, on the high side
        return int(sum(len(m["content"].split()) for m in messages) * 1.3) + 8

    def generate(self, conversations: Sequence[Conversation], *, max_new_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 42
                 ) -> List[Optional[str]]:
        if not conversations:
            return []
        kwargs = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                      seed=seed)
        live = [r for r in range(len(self.workers)) if r not in self._dead]
        if not live:
            raise RuntimeError("every worker has died; nothing is left to answer with")
        # chunks in proportion to each live worker's batch, in order
        total = sum(self.per_worker[r] for r in live)
        chunks: List[Tuple[int, int, int]] = []     # (worker, start, end)
        at = 0
        for n_th, r in enumerate(live):
            size = self.per_worker[r]
            n = max(1, round(len(conversations) * size / total)) if n_th < len(live) - 1 \
                else len(conversations) - at
            if at >= len(conversations):
                break
            chunks.append((r, at, min(len(conversations), at + n)))
            at = min(len(conversations), at + n)
        replies: List[Optional[str]] = [None] * len(conversations)
        pending: Dict[int, Tuple[int, int, int, int]] = {}      # job id -> (worker, s, e, tries)

        def send(r: int, start: int, end: int, tries: int) -> None:
            self._serial += 1
            self.jobs[r].put((self._serial, list(conversations[start:end]), kwargs))
            pending[self._serial] = (r, start, end, tries)

        for r, start, end in chunks:
            send(r, start, end, 1)
        tired = set()       # workers that failed a chunk this call: no retries go to them
        while pending:
            # `who` is the job id for done/failed, and the worker's rank for died
            kind, who, payload = self._next(lambda: {p[0] for p in pending.values()})
            if kind == "done":
                # a late answer from a slow worker to a job since re-sent is
                # still the answer to those rows: the first good one counts,
                # and the other, when it comes, finds nothing pending
                while who in self._superseded:
                    who = self._superseded[who]
                if who in pending:
                    r, start, end, _tries = pending.pop(who)
                    replies[start:end] = payload
                continue
            # a failed chunk goes once more to another worker, else it is the
            # error; a dead worker's chunks, all of them, the same way
            if kind == "died":
                lost = [i for i, p in pending.items() if p[0] == who]
            else:
                lost = [who] if who in pending else []
            for i in lost:
                r, start, end, tries = pending.pop(i)
                tired.add(r)
                others = [w for w in range(len(self.workers))
                          if w not in self._dead and w not in tired]
                if tries >= 2 or not others:
                    raise RuntimeError(f"a worker could not answer a batch: {payload}")
                send(others[0], start, end, tries + 1)
                self._superseded[i] = self._serial
        return replies

    def close(self) -> None:
        for q in getattr(self, "jobs", []):
            try:
                q.put(None)
            except Exception:
                pass
        for w in getattr(self, "workers", []):
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
