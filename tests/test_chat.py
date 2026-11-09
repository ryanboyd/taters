"""
Chat mode: the reply split into thinking and answer as it streams, the
conversation fitted to the model's window, the CSV export, and the screens
under Wrangle Language Models, end to end with a canned model.
"""
from __future__ import annotations

import csv
from datetime import datetime

import pytest

from taters.text import chat
from taters.text._generation import FAKE_LLM_ENV, FakeGenerator
from taters.text.chat import ChatSession, ThinkFilter
from taters.ui.prompts import GoBack, ScriptedPrompter


# ---------------------------------------------------------------------------
# thinking and answer, as the reply arrives
# ---------------------------------------------------------------------------

def _run(pieces, expect=False):
    f = ThinkFilter(expect_thinking=expect)
    parts = [p for piece in pieces for p in f.feed(piece)] + f.close()
    out = {"thinking": "", "answer": ""}
    for kind, text in parts:
        out[kind] += text
    return out


@pytest.mark.parametrize("pieces", [
    ["<think>hm, let me see</think>The answer is 5"],
    ["<thi", "nk>hm, let", " me see</th", "ink>The answer", " is 5"],
    list("<think>hm, let me see</think>The answer is 5"),        # a character at a time
])
def test_tags_cut_across_pieces_are_still_tags(pieces):
    assert _run(pieces) == {"thinking": "hm, let me see", "answer": "The answer is 5"}


def test_a_template_that_opens_the_block_itself_is_read_from_the_close():
    assert _run(["hm, let", " me see</think>", "5"], expect=True) == \
        {"thinking": "hm, let me see", "answer": "5"}


def test_a_model_told_to_think_that_does_not_gives_only_an_answer():
    assert _run(["just ", "the answer"], expect=True) == {"thinking": "", "answer": "just the answer"}


def test_an_unclosed_block_is_thinking():
    assert _run(["<think>still going"]) == {"thinking": "still going", "answer": ""}


def test_text_that_only_looks_like_a_tag_start_is_given_back():
    assert _run(["a < b and x <", "y"]) == {"thinking": "", "answer": "a < b and x <y"}
    assert _run(["ends with <"]) == {"thinking": "", "answer": "ends with <"}


def test_a_model_not_told_to_think_streams_straight_through():
    f = ThinkFilter()
    assert f.feed("Hello ") == [("answer", "Hello ")], "nothing held back for no reason"


# ---------------------------------------------------------------------------
# the conversation
# ---------------------------------------------------------------------------

def _echo(user):
    return f"you said {user}"


def test_each_turn_carries_the_whole_conversation_before_it():
    gen = FakeGenerator(_echo)
    s = ChatSession(gen, model="m", system="be kind")
    assert "".join(s.send("hello")) == "you said hello"
    "".join(s.send("again"))
    last = gen.calls[-1][0]
    assert last.messages() == [
        {"role": "system", "content": "be kind"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "you said hello"},
        {"role": "user", "content": "again"}]
    assert [m.role for m in s.messages] == ["user", "assistant", "user", "assistant"]


def test_a_long_chat_lets_go_of_its_oldest_exchanges_but_never_the_system_prompt():
    # the fake counts words; 40 of room, 10 kept for the answer
    gen = FakeGenerator(_echo, context_length=48)
    s = ChatSession(gen, model="m", system="sys", max_new_tokens=10)
    for i in range(6):
        "".join(s.send(f"message number {i} here"))
    sent = gen.calls[-1][0].messages()
    assert sent[0] == {"role": "system", "content": "sys"}
    assert sent[-1]["content"] == "message number 5 here"
    assert gen.count_tokens(sent) <= 48 - 10 - 8
    assert s.out_of_view > 0 and s.out_of_view % 2 == 0, "whole exchanges only"
    assert len(s.messages) == 12, "the transcript keeps every message"
    assert sent[1]["role"] == "user", "the history starts on a question, not a reply"


def test_a_message_too_long_to_send_is_taken_back_out():
    gen = FakeGenerator(_echo, context_length=30)
    s = ChatSession(gen, model="m", max_new_tokens=10)
    with pytest.raises(ValueError, match="longer than the model can read"):
        "".join(s.send("word " * 40))
    assert s.messages == [] and gen.calls == []


def test_a_window_that_is_not_known_sends_everything():
    gen = FakeGenerator(_echo, context_length=None)
    s = ChatSession(gen, model="m")
    for _ in range(3):
        "".join(s.send("word " * 500))
    assert len(gen.calls[-1][0].messages()) == 5 and s.out_of_view == 0


@pytest.mark.parametrize("thinking, kept", [("keep", "pondering"), ("drop", ""), ("off", "")])
def test_reasoning_is_kept_only_when_asked(thinking, kept):
    gen = FakeGenerator(lambda u: "<think>pondering</think>Yes.")
    s = ChatSession(gen, model="m", thinking=thinking)
    "".join(s.send("well?"))
    assert s.messages[-1].content == "Yes."
    assert s.messages[-1].reasoning == kept


def test_thinking_gets_its_room_on_top_of_the_answer():
    assert ChatSession(FakeGenerator(_echo), model="m", max_new_tokens=100,
                       thinking="keep", reasoning_tokens=50).reply_budget == 150
    assert ChatSession(FakeGenerator(_echo), model="m", max_new_tokens=100,
                       thinking="off", reasoning_tokens=50).reply_budget == 100


def test_the_export_is_every_line_with_its_time(tmp_path):
    gen = FakeGenerator(lambda u: '<think>hmm</think>She said "hi",\nthen left.')
    s = ChatSession(gen, model="org/model", system="you are terse", thinking="keep")
    "".join(s.send("what happened?"))
    out = s.export_csv(tmp_path / "deep" / "chat.csv")
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "a BOM so Excel reads it as UTF-8"
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(chat.CSV_COLUMNS)
    assert [r["role"] for r in rows] == ["system", "user", "assistant"]
    assert rows[2]["content"] == 'She said "hi",\nthen left.'
    assert rows[2]["reasoning"] == "hmm" and rows[1]["reasoning"] == ""
    assert {r["model"] for r in rows} == {"org/model"}
    for r in rows:
        stamp = datetime.fromisoformat(r["timestamp"])
        assert stamp.tzinfo is not None, "the offset is in the stamp"


def test_a_system_prompt_comes_from_a_library_prompt_or_any_file(tmp_path):
    with_system = tmp_path / "Judge.csv"
    with_system.write_text("role,content\nsystem,You are a judge.\nuser,Rate {text}\n",
                           encoding="utf-8")
    assert chat.system_prompt_from_file(with_system)[0] == "You are a judge."
    question_only = tmp_path / "Rate.csv"
    question_only.write_text("role,content\nuser,Rate this text: {text}\n", encoding="utf-8")
    text, where = chat.system_prompt_from_file(question_only)
    assert text == "Rate this text:" and "no system turn" in where
    plain = tmp_path / "persona.md"
    plain.write_text("﻿You are a pirate.\n\nAlways.\n", encoding="utf-8")
    assert chat.system_prompt_from_file(plain) == ("You are a pirate.\n\nAlways.", "persona.md")


def test_the_default_export_name_is_the_start_time():
    assert chat.default_export_name(datetime(2026, 1, 2, 3, 4, 5)) == "chat-20260102-030405.csv"


# ---------------------------------------------------------------------------
# the screens, end to end with canned replies
# ---------------------------------------------------------------------------

class Prompter(ScriptedPrompter):
    """A scripted GoBack() is Esc pressed on that question."""

    def _next(self, kind, question, default=None):
        value = super()._next(kind, question, default)
        if isinstance(value, GoBack):
            raise value
        return value

@pytest.fixture
def canned(monkeypatch, tmp_path):
    table = tmp_path / "replies.json"
    table.write_text('{"default": "I am a canned model.", '
                     '"contains": {"pirate": "<think>arr</think>Ahoy."}}', encoding="utf-8")
    monkeypatch.setenv(FAKE_LLM_ENV, str(table))
    from taters.ui import library

    picked = []
    monkeypatch.setattr(library, "pick_generative_model",
                        lambda p, current="", for_step="", later="":
                        picked.append((for_step, later)) or "org/tiny-chat")
    return picked


def test_chat_is_a_verb_under_wrangle_language_models():
    from taters.ui.train import VERBS

    assert "chat" in [v for v, _l, _h in VERBS]


def test_a_whole_chat_from_the_menu_to_the_saved_csv(canned, tmp_path):
    from taters.ui.train import run_train

    out = tmp_path / "saved.csv"
    p = Prompter([
        "chat",
        "system", ":type", "You are a pirate.",   # system prompt, from the settings
        "thinking", "keep",                   # one setting changed
        ":start",
        "hello", "", "talk like a pirate", "/quit",
        True, str(out),                       # save it, there
    ])
    p.answers.append(GoBack())   # back on the verb menu after the chat; Esc leaves
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    assert [q for k, q in p.asked if k == "select"].count("What would you like to do?") == 2
    assert canned == [("Chat", "It will be downloaded when the chat starts.")], \
        "a download left for later is said in a chat's terms, not a pipeline's"
    assert "I am a canned model." in p.chat_text and "Ahoy." in p.chat_text
    assert "arr" in p.chat_text, "kept thinking is shown"
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["role"], r["content"]) for r in rows] == [
        ("system", "You are a pirate."), ("user", "hello"),
        ("assistant", "I am a canned model."), ("user", "talk like a pirate"),
        ("assistant", "Ahoy.")]
    assert rows[-1]["reasoning"] == "arr"
    assert [q for q in p.asked if q[0] == "chat"] == [("chat", "You")] * 4


def test_esc_in_the_chat_leaves_and_declining_saves_nothing(canned, tmp_path):
    from taters.ui.chat import run_chat

    p = Prompter([":start", "hi", None, False])
    run_chat(p, cwd=tmp_path)
    assert "I am a canned model." in p.chat_text
    assert list(tmp_path.glob("*.csv")) == []


def test_a_chat_with_nothing_said_offers_no_export(canned, tmp_path):
    from taters.ui.chat import run_chat

    p = Prompter([":start", "/exit"])
    run_chat(p, cwd=tmp_path)      # would run out of answers if it asked
    assert not any("CSV" in q for _k, q in p.asked)


def test_a_failed_reply_is_said_and_the_chat_goes_on(canned, tmp_path, monkeypatch):
    from taters.ui.chat import run_chat

    calls = []

    def flaky(self, conv, **kw):
        calls.append(conv.user)
        if len(calls) == 1:
            raise RuntimeError("the server went away")
        yield "fine now"

    monkeypatch.setattr(FakeGenerator, "stream", flaky)
    p = Prompter([":start", "one", "two", "/exit", False])
    run_chat(p, cwd=tmp_path)
    assert "could not answer: RuntimeError: the server went away" in p.chat_text
    assert "fine now" in p.chat_text


def test_the_model_leads_straight_to_the_settings_and_esc_there_goes_back_to_it(
        canned, tmp_path, monkeypatch):
    from taters.text import _generation
    from taters.ui import library
    from taters.ui.chat import run_chat

    loads = []
    monkeypatch.setattr(_generation, "generator_for",
                        lambda *a, **k: loads.append(1) or FakeGenerator(_echo))
    p = Prompter([GoBack(), ":start", "/exit"])
    run_chat(p, cwd=tmp_path)
    assert p.asked[0] == ("select", "Chat settings:"), "no screen between the model and these"
    assert len(canned) == 2, "Esc on the settings went back to the model picker"
    assert loads == [1], "loaded once, only after Start"

    monkeypatch.setattr(library, "pick_generative_model", lambda *a, **k: None)
    with pytest.raises(GoBack):
        run_chat(Prompter([]), cwd=tmp_path)
    assert loads == [1], "Esc on the model loads nothing"


def test_settings_are_checked_and_shown(canned, tmp_path):
    from taters.ui.chat import run_chat

    p = Prompter(["temperature", "hot", "5", "0.7",
                          "max_new_tokens", "200", ":start", "/exit"])
    run_chat(p, cwd=tmp_path)
    shown = "\n".join(p.output)
    assert "is not a number" in shown and "Between 0.0 and 2.0" in shown
    last_screen = [c.label for c in [cs for q, cs in p.offered if q == "Chat settings:"][-1]]
    assert any("0.7" in label for label in last_screen)
    assert any("200" in label for label in last_screen)


def test_sampling_settings_show_only_when_answers_vary_and_reach_the_model(canned, tmp_path,
                                                                            monkeypatch):
    from taters.ui.chat import run_chat

    seen = {}
    real = FakeGenerator.stream

    def spy(self, conv, **kw):
        seen.update(kw)
        yield from real(self, conv, **kw)

    monkeypatch.setattr(FakeGenerator, "stream", spy)
    p = Prompter(["temperature", "0", "temperature", "0.8", "top_p", "0.85", "top_k", "30",
                  "seed", "7", "repetition_penalty", "1.2", ":start", "hi", "/exit", False])
    run_chat(p, cwd=tmp_path)
    screens = [[c.value for c in cs] for q, cs in p.offered if q == "Chat settings:"]
    assert "top_p" not in screens[1] and "seed" not in screens[1], "greedy: nothing to sample"
    assert "repetition_penalty" in screens[1], "the penalty applies either way"
    assert {"top_p", "top_k", "seed"} <= set(screens[-1])
    assert (seen["temperature"], seen["top_p"], seen["top_k"], seen["seed"],
            seen["repetition_penalty"]) == (0.8, 0.85, 30, 7, 1.2)


def test_a_chat_samples_by_default_with_a_fresh_seed_each_time(canned, tmp_path, monkeypatch):
    from taters.ui.chat import run_chat

    seen = []
    real = FakeGenerator.stream

    def spy(self, conv, **kw):
        seen.append(kw)
        yield from real(self, conv, **kw)

    monkeypatch.setattr(FakeGenerator, "stream", spy)
    for _ in range(2):
        run_chat(Prompter([":start", "hi", "/exit", False]), cwd=tmp_path)
    first, second = seen
    assert (first["temperature"], first["top_p"], first["top_k"],
            first["repetition_penalty"]) == (0.7, 0.9, 40, 1.15)
    assert first["seed"] != second["seed"], "a chat is not a measure: a new seed each time"


def test_a_penalty_below_one_is_refused(canned, tmp_path):
    from taters.ui.chat import run_chat

    p = Prompter(["repetition_penalty", "0.5", "1.1", ":start", "/exit"])
    run_chat(p, cwd=tmp_path)
    assert "Between 1.0 and 2.0" in "\n".join(p.output)
    assert any("1.05 to 1.2 is typical" in r for r in p.reasons), \
        "what to type is said above the box, not only on the row"


def test_every_number_setting_says_whats_typical_and_its_range():
    from taters.ui.chat import _NUMBERS
    from taters.ui.prompts import wrap_description

    for key, (label, help_text, _kind, low, high) in _NUMBERS.items():
        assert "Range" in help_text or key == "seed", key
        assert ("typical" in help_text or "suits" in help_text or "about" in help_text
                or key == "seed"), key
        lines = wrap_description(help_text, width=110, max_lines=9).count("\n") + 1
        assert lines <= 2, f"{key}: keep it to two lines on a 110-column terminal"


def test_replies_are_labeled_poe_unless_named(canned, tmp_path):
    from taters.ui.chat import run_chat

    p = Prompter([":start", "hi", "/exit", False])
    run_chat(p, cwd=tmp_path)
    assert "Poe › I am a canned model." in p.chat_text
    assert "Chatting with Poe (tiny-chat)" in p.chat_text, "the model is still named once"

    out = tmp_path / "named.csv"
    p = Prompter(["bot_name", "Spud", ":start", "hi", "/exit", True, str(out)])
    run_chat(p, cwd=tmp_path)
    assert "Spud › I am a canned model." in p.chat_text
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[-1]["role"] == "assistant" and rows[-1]["model"] == "org/tiny-chat"

    p = Prompter(["bot_name", "   ", ":start", "hi", "/exit", False])
    run_chat(p, cwd=tmp_path)
    assert "Poe › " in p.chat_text, "an empty name puts the default back"


# ---------------------------------------------------------------------------
# the transcript after a reply that failed or was stopped
# ---------------------------------------------------------------------------

def test_a_reply_that_fails_partway_is_kept_as_far_as_it_got():
    def half(conv, **kw):
        yield "Paris is "
        raise RuntimeError("the server went away")

    gen = FakeGenerator(_echo)
    gen.stream = half
    s = ChatSession(gen, model="m")
    with pytest.raises(RuntimeError):
        list(s.send("capital of France?"))
    assert [(m.role, m.content.strip()) for m in s.messages] == [
        ("user", "capital of France?"), ("assistant", "Paris is")]


def test_a_reply_that_never_started_takes_the_question_back_out_so_turns_alternate():
    """Gemma's and Llama's templates refuse two user turns in a row: one
    failed reply used to leave [user, user] and every later send failed."""
    def dead(conv, **kw):
        raise RuntimeError("no server")
        yield  # unreachable; makes this a generator

    gen = FakeGenerator(_echo)
    gen.stream = dead
    s = ChatSession(gen, model="m")
    with pytest.raises(RuntimeError):
        list(s.send("hi"))
    assert s.messages == []
    del gen.stream       # the class's stream again
    list(s.send("hello again"))
    assert [m.role for m in s.messages] == ["user", "assistant"]
    assert gen.calls[-1][0].messages() == [{"role": "user", "content": "hello again"}]


def test_a_reply_stopped_by_the_reader_is_recorded_as_far_as_it_got():
    """Ctrl-C in the chat closes the generator partway through."""
    gen = FakeGenerator(lambda u: "one two three four")
    s = ChatSession(gen, model="m")
    stream = s.send("count")
    assert next(stream) == "one " and next(stream) == "two "
    stream.close()
    assert [(m.role, m.content.strip()) for m in s.messages] == [("user", "count"),
                                                                  ("assistant", "one two")]


def test_ctrl_c_during_a_reply_stops_it_and_keeps_what_was_said(canned, tmp_path, monkeypatch):
    from taters.ui.chat import run_chat

    def interrupted(self, conv, **kw):
        yield "Ahoy "
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeGenerator, "stream", interrupted)
    out = tmp_path / "c.csv"
    p = Prompter([":start", "hi", "/exit", True, str(out)])
    run_chat(p, cwd=tmp_path)
    assert "(stopped)" in p.chat_text
    with out.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["role"], r["content"].strip()) for r in rows] == [("user", "hi"),
                                                                  ("assistant", "Ahoy")]


def test_a_reply_that_never_started_tells_the_user_to_say_it_again(canned, tmp_path,
                                                                     monkeypatch):
    from taters.ui.chat import run_chat

    def dead(self, conv, **kw):
        raise RuntimeError("no server")
        yield  # unreachable; makes this a generator

    monkeypatch.setattr(FakeGenerator, "stream", dead)
    p = Prompter([":start", "hi", "/exit"])
    run_chat(p, cwd=tmp_path)       # nothing kept, so no export is offered
    assert "Your message was not kept; say it again." in p.chat_text


def test_a_broken_prompt_csv_is_refused_not_used_as_the_system_prompt(tmp_path):
    """Its raw text would make a nonsense system prompt and nothing would say
    so; a .txt, which is plain text whatever else it is, is still read whole."""
    bad = tmp_path / "Broken.csv"
    bad.write_bytes(b'role,content\nsystem,"unbalanced\nuser,{text}\n')
    with pytest.raises(ValueError, match="Broken.csv could not be read as a prompt file"):
        chat.system_prompt_from_file(bad)
    plain = tmp_path / "notes.txt"
    plain.write_text("You are terse.\n\nAlways.\n", encoding="utf-8")
    assert chat.system_prompt_from_file(plain)[0] == "You are terse.\n\nAlways."


def test_the_screen_says_a_prompt_file_could_not_be_read_and_asks_again(canned, tmp_path):
    from taters.ui.chat import run_chat

    bad = tmp_path / "Broken.csv"
    bad.write_bytes(b'role,content\nsystem,"unbalanced\nuser,{text}\n')
    p = Prompter(["system", ":file", str(bad), ":none", ":start", "/exit"])
    run_chat(p, cwd=tmp_path)
    assert any("Broken.csv could not be read" in line for line in p.output)


def test_an_empty_first_piece_does_not_count_as_a_reply(canned, tmp_path, monkeypatch):
    """The screen and the transcript agree on whether the question was kept:
    the session decides, the screen reads it back."""
    from taters.ui.chat import run_chat

    def blank_then_dead(self, conv, **kw):
        yield ""
        raise RuntimeError("gone")

    monkeypatch.setattr(FakeGenerator, "stream", blank_then_dead)
    p = Prompter([":start", "hi", "/exit"])
    run_chat(p, cwd=tmp_path)       # no export offered: nothing was kept
    assert "Your message was not kept" in p.chat_text
