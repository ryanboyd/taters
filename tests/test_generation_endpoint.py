"""
The server backend against a small OpenAI-style server run in a thread:
the preflight, a reply, a 429 that is retried, a 500 that is retried, a
400 that is not, and the step end to end through it.
"""
from __future__ import annotations

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from taters.text import generate_with_llm as al  # noqa: E402
from taters.text._generation import Conversation  # noqa: E402
from taters.text._generation_endpoint import EndpointGenerator, split_server_model  # noqa: E402
from csvhelpers import _read  # noqa: E402


class _Server:
    """Answers like Ollama: /v1/models lists one model; /v1/chat/completions
    replies with the user text's last word, after whatever failures the
    test scripted (a list of status codes to send first)."""

    def __init__(self, failures=()):
        self.failures = list(failures)
        self.requests: list = []
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "tiny-chat"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                doc = json.loads(self.rfile.read(n).decode("utf-8"))
                with server.lock:
                    server.requests.append(doc)
                    code = server.failures.pop(0) if server.failures else 200
                if code != 200:
                    self.send_response(code)
                    if code == 429:
                        self.send_header("Retry-After", "0")
                    self.end_headers()
                    self.wfile.write(b'{"error": "scripted"}')
                    return
                user = doc["messages"][-1]["content"]
                if doc.get("stream"):
                    # the way a reasoning server streams: its thinking apart
                    # (reasoning_content), then the answer a piece at a time
                    deltas = [{"role": "assistant"}, {"reasoning_content": "the user "},
                              {"reasoning_content": "said hi"}, {"content": "Hello"},
                              {"content": " back"}]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for d in deltas:
                        self.wfile.write(b"data: " + json.dumps(
                            {"choices": [{"delta": d}]}).encode() + b"\n\n")
                    self.wfile.write(b": keep-alive\n\ndata: not json\n\ndata: [DONE]\n\n")
                    return
                reply = json.dumps({"mood": len(user.split()) % 7 + 1})
                body = json.dumps({"choices": [{"message": {"role": "assistant",
                                                             "content": reply}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.stop()


def test_the_servers_spelling_is_read_either_way():
    assert split_server_model("http://h:1/v1#qwen") == ("http://h:1/v1", "qwen")
    assert split_server_model("qwen", "http://h:1/v1/") == ("http://h:1/v1", "qwen")
    assert split_server_model("http://h:1/v1") == ("http://h:1/v1", "")


def test_no_server_is_a_sentence_naming_the_fix():
    with pytest.raises(ValueError, match="no server answered.*ollama serve"):
        EndpointGenerator("http://127.0.0.1:9/v1", model="x")


def test_replies_come_back_in_order_and_busy_answers_are_retried(server):
    server.failures[:] = [429, 500]
    gen = EndpointGenerator(server.url, model="tiny-chat", concurrency=3)
    assert gen.kind == "endpoint" and gen.reason == ""
    convs = [Conversation(prefix=[("system", "rate")], user=f"text number {i} here")
             for i in range(5)]
    replies = gen.generate(convs, max_new_tokens=16)
    assert len(replies) == 5 and all(r.startswith('{"mood":') for r in replies)
    assert len(server.requests) == 7, "five answers, two retried"
    assert server.requests[0]["model"] == "tiny-chat"
    assert server.requests[0]["temperature"] == 0.0 and "top_p" not in server.requests[0]
    assert gen.describe().startswith("OpenAI-style chat API")


def test_a_model_the_server_does_not_have_is_said_and_a_bad_request_is_not_retried(server):
    gen = EndpointGenerator(server.url, model="nope")
    assert "does not list 'nope'" in gen.reason
    assert EndpointGenerator(server.url, model="tiny-chat").reason == ""
    server.failures[:] = [400]
    assert gen.generate([Conversation(user="x")], max_new_tokens=8) == [None]
    assert gen.errors[0].startswith("RuntimeError: the server refused the request")
    assert len(server.requests) == 1
    # one bad request among several costs only its own reply
    server.failures[:] = [400]
    got = gen.generate([Conversation(user=f"t {i}") for i in range(3)], max_new_tokens=8)
    assert got.count(None) == 1 and sum(1 for g in got if g) == 2
    assert sum(1 for e in gen.errors if e) == 1


def test_the_step_answers_through_a_server(server, tmp_path):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood from 1 to 7 as JSON.\n", encoding="utf-8")
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(4):
            w.writerow([f"t{i}", " ".join(["word"] * (i + 1))])
    out = al.generate_with_llm(parse_json=True, prompt_paths=[p], model=f"{server.url}#tiny-chat",
                                csv_path=src, text_cols=["text"], id_cols=["text_id"],
                                out_features_csv=tmp_path / "f" / "llm.csv", verbose=False)
    rows = _read(out)
    assert [r["Mood__parse_ok"] for r in rows] == ["1"] * 4
    assert all(r["Mood__mood"] for r in rows)
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "backend: endpoint" in report and server.url in report
    assert "context length was not known, so every text was sent whole" in report


def test_a_server_reads_long_texts_in_pieces_once_its_context_is_said(server, tmp_path):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood from 1 to 7 as JSON.\n", encoding="utf-8")
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["long", " ".join(f"Sentence {k} says something about the day." for k in range(60))])
        w.writerow(["short", "A short one."])
    out = al.generate_with_llm(parse_json=True, prompt_paths=[p], model=f"{server.url}#tiny-chat",
                                csv_path=src, text_cols=["text"], id_cols=["text_id"],
                                out_features_csv=tmp_path / "f" / "llm.csv",
                                long_texts="pieces", context_length=300, verbose=False)
    rows = {r["text_id"]: r for r in _read(out)}
    assert int(rows["long"]["Mood__chunks"]) > 1 and rows["short"]["Mood__chunks"] == "1"
    assert rows["long"]["Mood__parse_ok"] == "1"
    assert len(server.requests) == 1 + int(rows["long"]["Mood__chunks"]) + 1, \
        "the smoke test, every piece, and the short text"


def test_a_reasoning_only_reply_comes_back_as_a_think_block(monkeypatch):
    from taters.text._generation_endpoint import EndpointGenerator as EG

    class Quiet(EG):
        def _preflight(self):
            pass

    gen = Quiet("http://127.0.0.1:9/v1", model="m")
    import json as _json
    import urllib.request

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return _json.dumps(self.body).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Resp(
        {"choices": [{"message": {"content": None, "reasoning_content": "thinking…"}}]}))
    assert gen.generate([Conversation(user="x")], max_new_tokens=8) == ["<think>thinking…</think>"]


def test_a_server_that_lists_its_window_is_read(tmp_path):
    """vLLM's model listing says how much the model reads at once
    (`max_model_len`); llama.cpp nests an `n_ctx` under `meta`."""
    from taters.text._generation_endpoint import _context_in

    assert _context_in({"id": "m", "max_model_len": 32768}) == 32768
    assert _context_in({"id": "m", "meta": {"n_ctx": 8192}}) == 8192
    assert _context_in({"id": "m", "owned_by": "me"}) is None
    assert _context_in({"id": "m", "context_length": True}) is None


def test_the_step_announces_a_window_it_does_not_know(server, tmp_path):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    src = tmp_path / "c.csv"
    src.write_text("text_id,text\nt1,hello there\n", encoding="utf-8")
    said = []
    al.generate_with_llm(prompt_paths=[p], model=f"{server.url}#tiny-chat", csv_path=src,
                          text_cols=["text"], id_cols=["text_id"],
                          out_features_csv=tmp_path / "f" / "llm.csv", verbose=False,
                          on_progress=lambda *a, **k: said.append(" ".join(map(str, a))))
    assert any("context window is not known" in s for s in said)


def test_a_streamed_reply_arrives_in_pieces_with_its_thinking_tagged(server):
    gen = EndpointGenerator(server.url, model="tiny-chat")
    pieces = list(gen.stream(Conversation(prefix=[("system", "be brief")], user="hi"),
                             max_new_tokens=32))
    assert len(pieces) > 2, "a piece at a time, not the whole reply at the end"
    assert "".join(pieces) == "<think>the user said hi</think>Hello back"
    sent = server.requests[-1]
    assert sent["stream"] is True and sent["messages"][0] == {"role": "system",
                                                              "content": "be brief"}


def test_top_k_and_the_penalty_are_sent_only_when_asked_for(server):
    gen = EndpointGenerator(server.url, model="tiny-chat")
    conv = Conversation(user="hi")
    list(gen.stream(conv, max_new_tokens=8))
    plain = server.requests[-1]
    assert "top_k" not in plain and "repetition_penalty" not in plain and "seed" not in plain
    list(gen.stream(conv, max_new_tokens=8, temperature=0.5, top_k=20, seed=9,
                    repetition_penalty=1.15))
    asked = server.requests[-1]
    assert (asked["top_k"], asked["seed"], asked["repetition_penalty"]) == (20, 9, 1.15)
