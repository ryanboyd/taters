"""
One contract for "give a language model a conversation, get a reply".

The "Generate responses with an LLM" step (:mod:`generate_with_llm`) does not care where a
reply comes from: a model loaded into this process through transformers,
a local server speaking the OpenAI chat API, or canned answers in a
test. Each is a :class:`Generator`: it says what it is (``describe``),
how long a conversation is in its own tokens (``count_tokens``), how
much context it has, and it turns a batch of conversations into a batch
of replies. Everything about memory plans, batching, devices and retries
lives inside the backend that needs it; the step only hands over
conversations and writes rows.

A conversation is carried as a *prefix* (the system turn and any few-shot
examples, the same for every text of one prompt) and the final user turn,
so a backend that can reuse a shared prefix is free to, without the step
knowing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = ["Turn", "Conversation", "Generator", "FakeGenerator", "generator_for",
           "FAKE_LLM_ENV", "CURATED_GENERATIVE_MODELS", "cached_hub_generative_models"]

Turn = Tuple[str, str]      # (role, content)

#: A JSON file of canned replies for tests and dry runs: ``{"default":
#: "...", "contains": {"substring of the text": "reply", ...}}``. The
#: fake backend is chosen whenever this is set, whatever ``backend`` says,
#: so a pipeline can be driven end to end with no model on the machine.
FAKE_LLM_ENV = "TATERS_FAKE_LLM"


@dataclass
class Conversation:
    """One request: the prompt's shared turns, then the user turn made
    for this text. ``key`` is whatever the caller needs back beside the
    reply (a text id and a prompt name)."""

    prefix: List[Turn] = field(default_factory=list)
    user: str = ""
    key: Any = None

    def messages(self) -> List[Dict[str, str]]:
        return [{"role": r, "content": c} for r, c in self.prefix] + [
            {"role": "user", "content": self.user}]


class Generator:
    """
    The contract. ``generate`` takes a batch and returns one reply per
    conversation, in order; a reply that could not be made is ``None``
    and the backend has already retried what it could. ``count_tokens``
    is the backend's own count, used to cut a text to fit the context.
    """

    kind: str = ""
    model_label: str = ""
    device_name: str = ""
    reason: str = ""
    context_length: Optional[int] = None
    #: how many conversations a call should carry; the step batches to it
    batch_size: int = 1
    #: after a call: one entry per conversation, the reason when its reply
    #: is ``None``; a backend that raises instead loses the whole batch
    errors: List[Optional[str]] = []

    def describe(self) -> str:  # pragma: no cover - the backends
        return self.model_label

    def count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        return sum(len(m["content"].split()) for m in messages)

    def generate(self, conversations: Sequence[Conversation], *, max_new_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 42
                 ) -> List[Optional[str]]:  # pragma: no cover - the backends
        raise NotImplementedError

    def stream(self, conversation: Conversation, *, max_new_tokens: int,
               temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
               top_k: int = 0, repetition_penalty: float = 1.0) -> Iterator[str]:
        """
        One reply, as it is written: pieces of text, in order, that join
        into the reply. For a chat, where a whole answer after a silence
        reads as a hang. A backend that cannot stream hands the reply over
        in one piece; one that cannot answer raises.
        """
        (reply,) = self.generate([conversation], max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_p=top_p, seed=seed)
        if reply is None:
            why = (getattr(self, "errors", None) or [None])[0]
            raise RuntimeError(why or "the model gave no reply")
        yield reply

    def close(self) -> None:
        return None


class FakeGenerator(Generator):
    """
    Replies from a table, for tests and dry runs.

    ``reply`` is a callable from the user turn to the reply, or a mapping
    with ``default`` and ``contains`` (substring -> reply) the way the
    :data:`FAKE_LLM_ENV` file spells it. Every call is logged in
    ``calls`` so a test can assert how many conversations were made and
    that none was made twice.
    """

    kind = "fake"

    def __init__(self, reply: Any = None, *, model_label: str = "fake model",
                 context_length: Optional[int] = None, batch_size: int = 4):
        self.model_label = model_label
        self.device_name = "cpu"
        self.context_length = context_length
        self.batch_size = batch_size
        self.calls: List[List[Conversation]] = []
        if callable(reply):
            self._reply = reply
        else:
            table = dict(reply or {})
            default = str(table.get("default", ""))
            contains = {str(k): str(v) for k, v in (table.get("contains") or {}).items()}

            def by_table(user: str) -> str:
                for needle, answer in contains.items():
                    if needle in user:
                        return answer
                return default

            self._reply = by_table

    def describe(self) -> str:
        return f"{self.model_label}: canned replies (no model was run)"

    def generate(self, conversations: Sequence[Conversation], *, max_new_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 42
                 ) -> List[Optional[str]]:
        self.calls.append(list(conversations))
        return [self._reply(c.user) for c in conversations]

    def stream(self, conversation: Conversation, *, max_new_tokens: int,
               temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
               top_k: int = 0, repetition_penalty: float = 1.0) -> Iterator[str]:
        """The canned reply a word at a time, the way a model writes."""
        (reply,) = self.generate([conversation], max_new_tokens=max_new_tokens)
        words = str(reply).split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "


def generator_for(backend: str, model: str, *, endpoint_url: str = "",
                  device: str = "auto", precision: str = "auto", offload: bool = False,
                  batch_size: int = 0, concurrency: int = 4, hf_token: Optional[str] = None,
                  max_new_tokens: int = 256, thinking: bool = False, verbose: bool = False,
                  on_progress: Optional[Callable[..., None]] = None,
                  one_copy: bool = False) -> Generator:
    """
    The backend for a model name: canned replies when :data:`FAKE_LLM_ENV`
    is set, else a local server when ``backend`` is ``endpoint`` (or the
    model is spelled as a URL), else the model loaded here.
    """
    fake = os.environ.get(FAKE_LLM_ENV)
    if fake:
        table: Any = {}
        path = Path(fake)
        if path.is_file():
            table = json.loads(path.read_text(encoding="utf-8"))
        return FakeGenerator(table, model_label=f"fake ({model or 'no model'})")
    backend = str(backend or "auto").strip().lower()
    url = endpoint_url or (model if str(model).startswith(("http://", "https://")) else "")
    if backend == "endpoint" or (backend == "auto" and url):
        from ._generation_endpoint import EndpointGenerator

        return EndpointGenerator(url, model=model, concurrency=concurrency,
                                 max_new_tokens=max_new_tokens, verbose=verbose)
    if str(model).startswith(("http://", "https://")):
        raise ValueError(f"{model} is a server address; use backend='endpoint' (or 'auto')")
    from ._generation_local import (LocalGenerator, ParallelGenerator, devices_for,
                                    plan_load)

    if device != "cpu" and not offload:
        plan = plan_load(model, precision=precision, device=device,
                         max_new_tokens=max_new_tokens, batch_size=batch_size,
                         hf_token=hf_token)
        groups = devices_for(plan) if not plan.refusal else []
        # a copy per card is for many texts at once; a chat is one
        # conversation, and a copy in a worker process cannot stream
        if len(groups) > 1 and not one_copy:
            from ..helpers.progress import announce

            _fetch_once(model, plan, on_progress, hf_token)
            # through announce: the step's callback takes (done, total,
            # message), and a bare one-argument call here was a TypeError
            # that only a machine with several GPUs ever reached
            announce(on_progress, f"loading {plan.describe()}")
            return ParallelGenerator(model, groups, precision=precision,
                                     batch_size=batch_size, hf_token=hf_token,
                                     max_new_tokens=max_new_tokens, thinking=thinking,
                                     verbose=verbose, plan=plan)
    else:
        plan = None
    return LocalGenerator(model, device=device, precision=precision, offload=offload,
                         batch_size=batch_size, hf_token=hf_token,
                         max_new_tokens=max_new_tokens, thinking=thinking, verbose=verbose,
                         on_progress=on_progress, plan=plan)


def _fetch_once(model: str, plan, on_progress, hf_token: Optional[str] = None) -> None:
    """The weights into the cache from this process, before the workers
    start: otherwise every worker downloads the same files at once into the
    same folder, and the first run on a fresh server does N downloads'
    worth of waiting (and, on Windows, N chances at the link race). The
    step's token goes with it, since a gated model refuses without one. A
    failure is said, then left to the workers to report in the hub's words."""
    from ..helpers.progress import announce

    try:
        from ..helpers import hub_models as hm

        if plan.facts.kind == "folder" or hm.is_downloaded(model):
            return
        wanted = hm.plan_download(model, token=hf_token)
        if wanted.state != "ok":
            return
        label = f"downloading {model} ({wanted.gigabytes}) once, before the copies load"
        announce(on_progress, label)
        if on_progress is not None:
            hm.download(wanted, token=hf_token,
                        on_bytes=lambda done: on_progress(done, wanted.size or None, label))
        else:
            hm.download(wanted, token=hf_token)
    except Exception as e:  # noqa: BLE001 - the workers say what went wrong, with the hub's words
        announce(on_progress, f"the download could not be done once here ({type(e).__name__}: "
                              f"{str(e).splitlines()[0] if str(e) else ''}); each copy "
                              f"fetches for itself")


#: Instruction-tuned models worth offering by name, smallest first, with
#: what they cost: the size said in parameters and in memory at 16-bit,
#: which is what decides whether a laptop runs them.
CURATED_GENERATIVE_MODELS: Tuple[Tuple[str, str], ...] = (
    ("Qwen/Qwen3-0.6B",
     "0.6B parameters, about 1.2 GB at 16-bit. The smallest that follows an "
     "instruction: fine for yes/no and simple ratings, runs on a CPU."),
    ("Qwen/Qwen3-1.7B",
     "1.7B parameters, about 3.4 GB at 16-bit. The default: fits a laptop GPU, "
     "a fair rater; can reason before answering."),
    ("Qwen/Qwen3-4B",
     "4B parameters, about 8 GB at 16-bit, 2.5 GB at 4-bit. Noticeably better "
     "judgments; a desktop GPU, or a laptop with taters[llm]."),
    ("meta-llama/Llama-3.2-3B-Instruct",
     "3B parameters, about 6 GB at 16-bit. Gated: Meta approves each account, and "
     "the wizard asks for your Hugging Face token."),
    ("Qwen/Qwen3-8B",
     "8B parameters, about 16 GB at 16-bit, 5 GB at 4-bit (taters[llm]). "
     "A desktop GPU; the best judgments on this list."),
    ("meta-llama/Llama-3.1-8B-Instruct",
     "8B parameters, about 16 GB at 16-bit, 5.5 GB at 4-bit. Gated."),
)


def cached_hub_generative_models() -> List[Tuple[str, str]]:
    """The causal language models already in the Hugging Face cache:
    ``(hub id, note)``, offline and cheap."""
    from ._transformer_common import cached_hub_configs

    out: List[Tuple[str, str]] = []
    for hub_id, config in cached_hub_configs():
        archs = [str(a) for a in (config.get("architectures") or [])]
        if not any(a.endswith("ForCausalLM") for a in archs):
            continue
        layers = config.get("num_hidden_layers") or config.get("n_layer")
        width = config.get("hidden_size") or config.get("n_embd")
        note = f"{layers} layers, width {width}" if layers and width else "causal model"
        out.append((hub_id, note))
    return sorted(out)
