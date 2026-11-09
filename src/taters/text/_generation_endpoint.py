"""
The server backend: a local model behind the OpenAI chat API.

Ollama, LM Studio, vLLM and llama.cpp's server all speak the same few
JSON fields on ``/v1/chat/completions``, so one small client through the
standard library covers them, with no package to install. The server
loads and batches the model its own way; this side sends one conversation
per request, a few at a time, waits patiently (a reply can take a minute
on a slow machine), and retries the failures that mean "busy" rather than
"wrong": 429, 5xx, a dropped connection, a timeout.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Sequence

from ._generation import Conversation, Generator

__all__ = ["EndpointGenerator", "split_server_model", "server_context"]

#: where servers put a model's context window in their model listing, the
#: one the server is running with first: LM Studio's loaded length, vLLM's
#: max_model_len, llama.cpp's n_ctx, then the model's own maximum. Ollama's
#: listing says nothing, and its own API reports the model's trained
#: maximum rather than the smaller window it actually runs with, so it is
#: left unknown rather than guessed.
_CONTEXT_KEYS = ("loaded_context_length", "max_model_len", "n_ctx", "context_length",
                 "max_context_length", "context_window")


def _context_in(entry) -> Optional[int]:
    """The first context-window number in a model listing's entry, looked
    for at any depth (llama.cpp nests it under ``meta``)."""
    if isinstance(entry, dict):
        for key in _CONTEXT_KEYS:
            value = entry.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                return int(value)
        for value in entry.values():
            found = _context_in(value)
            if found:
                return found
    return None


def _model_listing(base: str, api_key: str = "", timeout: float = 10) -> list:
    req = urllib.request.Request(f"{base}/models", headers=(
        {"Authorization": f"Bearer {api_key}"} if api_key else {}))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        doc = json.loads(resp.read().decode("utf-8") or "{}")
    return [m for m in (doc.get("data") or []) if isinstance(m, dict)]


def server_context(base: str, model: str = "", api_key: str = "") -> Optional[int]:
    """How many tokens the server's model reads at once, when the server's
    model listing says; ``None`` when it does not, or cannot be reached."""
    try:
        entries = _model_listing(base.rstrip("/"), api_key, timeout=5)
    except Exception:  # noqa: BLE001 - unknown is an answer
        return None
    named = [m for m in entries if str(m.get("id", "")) in (model, model.split(":", 1)[0])]
    for entry in named or entries[:1]:
        found = _context_in(entry)
        if found:
            return found
    return None

_RETRIES = 5


def split_server_model(model: str, endpoint_url: str = "") -> tuple:
    """``("http://host/v1", "name")`` from the ways a server model is
    spelled: a URL, ``URL#name`` (the wizard's spelling), or a name with
    the URL given separately."""
    text = str(model or "").strip()
    url = str(endpoint_url or "").strip()
    name = text
    if text.startswith(("http://", "https://")):
        url, _sep, name = text.partition("#")
    return url.rstrip("/"), name.strip()


class EndpointGenerator(Generator):
    kind = "endpoint"

    def __init__(self, url: str, *, model: str = "", concurrency: int = 4,
                 max_new_tokens: int = 256, timeout: Optional[float] = None,
                 api_key: str = "", verbose: bool = False):
        base, name = split_server_model(model, url)
        if not base:
            raise ValueError("the endpoint backend needs the server's URL (endpoint_url), "
                             "such as http://localhost:11434/v1")
        self.base = base
        self.model = name
        self.concurrency = max(1, int(concurrency))
        # a slow server makes a token a second; the cap keeps a hung one from
        # holding every retry for an hour
        self.timeout = float(timeout) if timeout else min(600.0, max(120.0, float(max_new_tokens)))
        self.errors: List[Optional[str]] = []
        self.api_key = api_key
        self.verbose = verbose
        self.model_label = f"{name or 'the server default'} at {base}"
        self.device_name = "server"
        self.batch_size = self.concurrency
        self.context_length = None
        self.reason = ""
        self._preflight()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _preflight(self) -> None:
        """The server answers, and knows the model when it lists models."""
        try:
            req = urllib.request.Request(f"{self.base}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as resp:
                doc = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                return      # a server without the listing; the first reply will tell
            raise ValueError(f"the server at {self.base} answered {e.code} to /models; check "
                             f"the URL (it usually ends in /v1)") from None
        except Exception as e:  # noqa: BLE001 - every kind becomes a sentence
            raise ValueError(f"no server answered at {self.base} ({type(e).__name__}: {e}). "
                             f"Start it (Ollama: `ollama serve`; LM Studio: the server tab), "
                             f"or check the URL") from None
        entries = [m for m in (doc.get("data") or []) if isinstance(m, dict)]
        ids = [str(m.get("id", "")) for m in entries]
        # Ollama lists `llama3:latest` and answers to `llama3`
        known = set(ids) | {i.split(":", 1)[0] for i in ids}
        if self.model and ids and self.model not in known:
            self.reason = (f"the server does not list {self.model!r}; it has "
                           f"{', '.join(ids[:6])}" + (" …" if len(ids) > 6 else ""))
        if not self.model and ids:
            self.model = ids[0]
            self.model_label = f"{self.model} at {self.base}"
        # the window the server runs the model with, when its listing says
        mine = [m for m in entries if str(m.get("id", "")) in (self.model,
                                                                self.model.split(":", 1)[0])]
        for entry in mine or entries[:1]:
            found = _context_in(entry)
            if found:
                self.context_length = found
                break

    def describe(self) -> str:
        return (f"OpenAI-style chat API at {self.base}, model {self.model or 'default'}, "
                f"{self.concurrency} request(s) in flight")

    def count_tokens(self, messages: Sequence[Dict[str, str]]) -> int:
        # no tokenizer here: the server's. four characters a token is the
        # usual rule; it only sizes a cut or a piece, and only when the
        # model's context is known (said by the server, or set by hand)
        return sum(len(m["content"]) for m in messages) // 4

    def _body(self, conv: Conversation, *, max_new_tokens: int, temperature: float,
              top_p: float, seed: int, stream: bool = False, top_k: int = 0,
              repetition_penalty: float = 1.0) -> bytes:
        body = {"model": self.model, "messages": conv.messages(),
                "max_tokens": int(max_new_tokens), "temperature": float(temperature),
                "stream": bool(stream)}
        if temperature > 0:
            body["top_p"] = float(top_p)
            body["seed"] = int(seed)
            if int(top_k) > 0:
                body["top_k"] = int(top_k)
        # not in the OpenAI spec, but what vLLM, llama.cpp and LM Studio read;
        # sent only when asked for, so a strict server never sees it otherwise
        if float(repetition_penalty) != 1.0:
            body["repetition_penalty"] = float(repetition_penalty)
        return json.dumps(body).encode("utf-8")

    def stream(self, conversation: Conversation, *, max_new_tokens: int,
               temperature: float = 0.0, top_p: float = 1.0, seed: int = 42,
               top_k: int = 0, repetition_penalty: float = 1.0) -> Iterator[str]:
        """
        One reply as the server writes it: the OpenAI streaming format, a
        ``data:`` line per piece. Reasoning a server reports apart from the
        answer (``reasoning_content``) comes back inside a ``<think>``
        block, the way a local model writes it, so the reader treats both
        alike. Not retried: half a reply already on the screen cannot be
        taken back.
        """
        data = self._body(conversation, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p, seed=seed, stream=True,
                          top_k=top_k, repetition_penalty=repetition_penalty)
        req = urllib.request.Request(f"{self.base}/chat/completions", data=data,
                                     headers=self._headers(), method="POST")
        thinking = False
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        doc = json.loads(payload)
                    except ValueError:
                        continue
                    choices = doc.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        if not thinking:
                            thinking = True
                            yield "<think>"
                        yield str(reasoning)
                    content = delta.get("content")
                    if content:
                        if thinking:
                            thinking = False
                            yield "</think>"
                        yield str(content)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"the server refused the request (HTTP {e.code})") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise RuntimeError(f"the server at {self.base} stopped answering "
                               f"({type(e).__name__}: {e})") from None
        if thinking:
            yield "</think>"

    def _one(self, conv: Conversation, *, max_new_tokens: int, temperature: float,
             top_p: float, seed: int) -> str:
        data = self._body(conv, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_p=top_p, seed=seed)
        last = ""
        for attempt in range(_RETRIES + 1):
            try:
                req = urllib.request.Request(f"{self.base}/chat/completions", data=data,
                                             headers=self._headers(), method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    doc = json.loads(resp.read().decode("utf-8"))
                choices = doc.get("choices") or []
                message = (choices[0].get("message") or {}) if choices else {}
                content = message.get("content")
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                # a reasoning parser on the server hands the thinking over
                # apart from the answer: back together as the think block a
                # local model writes, so the step splits it off the same way
                # (and keeps it when asked); the stream does the same
                think = f"<think>{reasoning}</think>" if reasoning else ""
                if content is None:
                    return think
                return think + str(content).strip()
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code not in (408, 409, 429) and e.code < 500:
                    detail = ""
                    try:
                        detail = e.read().decode("utf-8", errors="replace")[:200]
                    except Exception:
                        pass
                    raise RuntimeError(f"the server refused the request ({last}): "
                                       f"{detail or e.reason}") from None
                wait = e.headers.get("Retry-After") if e.headers else None
                delay = float(wait) if wait and str(wait).isdigit() else None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                delay = None
            if attempt == _RETRIES:
                break
            time.sleep(delay if delay is not None else (2 ** attempt) + random.random())
        raise RuntimeError(f"the server at {self.base} did not answer after {_RETRIES} "
                           f"retries ({last})")

    def generate(self, conversations: Sequence[Conversation], *, max_new_tokens: int,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 42
                 ) -> List[Optional[str]]:
        if not conversations:
            return []
        self.errors = [None] * len(conversations)
        if len(conversations) == 1:
            try:
                return [self._one(conversations[0], max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_p=top_p, seed=seed)]
            except Exception as e:  # noqa: BLE001
                self.errors = [f"{type(e).__name__}: {e}"]
                return [None]
        # one failure does not cost the other replies in flight: it is None
        # with its reason beside it, the way the contract says
        replies: List[Optional[str]] = [None] * len(conversations)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._one, c, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_p=top_p, seed=seed)
                       for c in conversations]
            for i, f in enumerate(futures):
                try:
                    replies[i] = f.result()
                except Exception as e:  # noqa: BLE001
                    self.errors[i] = f"{type(e).__name__}: {e}"
        return replies
