"""
The LLM step under a few hundred made-up conditions, each from a fixed seed.

Each scenario draws a corpus (texts of every length and shape: no spaces at
all, emoji, blank, punctuation only, CRLF and tabs, one enormous "word"), a
set of prompts (the text placed once, twice, or not at all; a system turn
and worked examples; a placeholder filled from a column that can be
enormous too), a context window from barely enough up, a long-text mode, a
token count (by words or by characters), reasoning on or off, a batch size,
and a model that misbehaves at random: no reply, an empty one, one far past
its cap, braces and quotes, a reasoning block that never closes, a whole
batch that raises.

What must hold every time, whatever was drawn:

* no call is ever handed more than the context holds (the fake asserts it);
* the run ends, within a bound on the calls it makes;
* every text with any text in it has exactly one row;
* a refusal, when there is one, is one of the sentences the step means to
  say -- never a traceback.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from taters.text import generate_with_llm as al
from taters.text._generation import FakeGenerator
from csvhelpers import _read

#: the refusals the step means to give; anything else is a bug
_REFUSALS = ("too small to read long texts with notes", "is too small for the prompt")

_WORDS = ("the cat sat on a mat and then it rained for days while everyone waited "
          "patiently for news about the harvest").split()


class Chaos(FakeGenerator):
    """A model that is strict about the context and careless about everything
    else, by the scenario's dice."""

    def __init__(self, rng, context, *, by_chars, think, batch_size, max_calls):
        super().__init__(context_length=context, batch_size=batch_size)
        self.rng = rng
        self.by_chars = by_chars
        self.thinking = think
        self.max_calls = max_calls
        self.calls_made = 0

    def count_tokens(self, messages):
        if self.by_chars:
            return sum(len(m["content"]) for m in messages) // 4
        return sum(len(m["content"].split()) for m in messages)

    def reply(self, cap):
        r = self.rng.random()
        if r < 0.05:
            return None
        if r < 0.10:
            return ""
        if r < 0.15:
            return " ".join(self.rng.choice(_WORDS) for _ in range(cap * 5))   # ignores the cap
        if r < 0.20:
            return 'notes with {text} and {{braces}} and "quotes" and """ fences'
        if r < 0.25 and self.thinking:
            return "<think>going on and on"
        body = json.dumps({"score": self.rng.randint(1, 7), "why": "because"})
        return ("<think>hm</think>" + body) if self.thinking else body

    def generate(self, conversations, *, max_new_tokens, **kw):
        self.calls_made += len(conversations)
        assert self.calls_made <= self.max_calls, "the run did not finish"
        if len(conversations) > 1 and self.rng.random() < 0.05:
            raise RuntimeError("the whole batch failed")
        out = []
        for c in conversations:
            used = self.count_tokens(c.messages())
            assert used + max_new_tokens <= self.context_length, (
                f"overflow: {used} + {max_new_tokens} > {self.context_length}")
            out.append(self.reply(max_new_tokens))
        return out


def _text(rng: random.Random) -> str:
    shape = rng.choice(["normal", "long", "huge", "spaceless", "emoji", "blank", "punct",
                        "crlf", "one_word", "short"])
    words = lambda n: " ".join(rng.choice(_WORDS) for _ in range(n))  # noqa: E731
    if shape == "normal":
        return ". ".join(words(rng.randint(3, 20)) for _ in range(rng.randint(1, 8))) + "."
    if shape == "long":
        return ". ".join(words(rng.randint(5, 30)) for _ in range(rng.randint(20, 60))) + "."
    if shape == "huge":
        return ". ".join(words(rng.randint(5, 30)) for _ in range(rng.randint(80, 200))) + "."
    if shape == "spaceless":
        return "这是一个没有空格的很长的中文句子。" * rng.randint(5, 120)
    if shape == "emoji":
        return " ".join(rng.choice(["🥔", "😀", "🌧️", "ok", "é"]) for _ in range(rng.randint(5, 400)))
    if shape == "blank":
        return rng.choice(["", "   ", "\n\n\t"])
    if shape == "punct":
        return "!?.,;:" * rng.randint(1, 200)
    if shape == "crlf":
        return "\r\n".join(words(rng.randint(5, 40)) for _ in range(rng.randint(2, 40)))
    if shape == "one_word":
        return "x" * rng.randint(50, 6000)
    return words(rng.randint(1, 4))


def _prompts(rng: random.Random, folder: Path):
    kinds = rng.sample(["plain", "twice", "examples", "placeholder"], rng.randint(1, 3))
    out = []
    for kind in kinds:
        if kind == "plain":
            p = folder / "Plain.txt"
            p.write_text("Rate it as JSON.\n", encoding="utf-8")
        elif kind == "twice":
            p = folder / "Twice.txt"
            p.write_text("Read this:\n{text}\nAnd again, carefully:\n{text}\nRate it.\n",
                         encoding="utf-8")
        elif kind == "examples":
            p = folder / "Examples.csv"
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["role", "content"])
                w.writerows([("system", "You rate texts. " * rng.randint(1, 10)),
                             ("user", "Rate: a sunny day"), ("assistant", '{{"score": 6}}'),
                             ("user", 'Rate this text:\n"""\n{text}\n"""')])
        else:
            p = folder / "Placeholder.txt"
            p.write_text("The writer is from {group}. Rate it.\n\n{text}\n", encoding="utf-8")
        out.append(p)
    return out


SEEDS = list(range(240))


@pytest.mark.parametrize("seed", SEEDS)
def test_whatever_is_drawn_no_call_overflows_and_every_text_gets_its_row(seed, tmp_path,
                                                                          monkeypatch):
    rng = random.Random(seed)
    n_texts = rng.randint(1, 7)
    texts = [_text(rng) for _ in range(n_texts)]
    groups = [rng.choice(["Ohio", "", "a place " * rng.choice([1, 50, 400])])
              for _ in range(n_texts)]
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text", "group"])
        for i, (t, g) in enumerate(zip(texts, groups)):
            w.writerow([f"t{i}", t, g])
    prompts = _prompts(rng, tmp_path)
    mode = rng.choice(["cut", "pieces", "notes", "notes"])
    think = rng.random() < 0.3
    gen = Chaos(rng, rng.choice([120, 300, 600, 1200, 4000]), by_chars=rng.random() < 0.3,
                think=think, batch_size=rng.choice([1, 2, 3, 8]), max_calls=20000)
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: gen)
    try:
        out = al.generate_with_llm(
            prompt_paths=prompts, model="m", csv_path=src, text_cols=["text"],
            id_cols=["text_id"], pass_through_cols=["group"],
            out_features_csv=tmp_path / "f" / "llm.csv", long_texts=mode,
            max_new_tokens=rng.choice([8, 40, 200]), keep_reasoning=think,
            reasoning_tokens=rng.choice([0, 20, 100]), parse_json=rng.random() < 0.5,
            verbose=False)
    except al.TruncatedReplies:
        return        # the guard did its job on a model that never finishes
    except ValueError as e:
        assert any(r in str(e) for r in _REFUSALS), f"seed {seed}: {e}"
        return
    rows = _read(out)
    got = [r["text_id"] for r in rows]
    # the gather leaves out a row with no text at all, before this step
    wanted = {f"t{i}" for i, t in enumerate(texts) if t.strip()}
    assert len(got) == len(set(got)), f"seed {seed}: a text has two rows"
    assert wanted <= set(got) <= {f"t{i}" for i in range(n_texts)}, seed
    for r in rows:
        for p in prompts:
            col = al.read_prompt(p).column
            if mode != "cut":
                n = int(r[f"{col}__chunks"])
                record = json.loads(r[f"{col}__chunk_replies"])
                assert isinstance(record, list)
                # no parts at all only for a row that could not be asked, and
                # then its error says so
                assert n >= 1 or "nothing was asked" in r[f"{col}__error"], (seed, r)
                if mode == "pieces":
                    assert len(record) == n
