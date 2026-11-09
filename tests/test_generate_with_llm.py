"""
Tests for `taters.text.generate_with_llm`: prompts, replies, and the step end to end.

No model is run here. The step is driven with canned replies through the
`TATERS_FAKE_LLM` switch, or a fake generator handed in, so what is under
test is the prompt files, the filling, the reading of replies, the key
dictionary, the columns, the report and the record -- the parts a user
actually configures.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from taters.text import generate_with_llm as al  # noqa: E402
from taters.text._generation import Conversation, FakeGenerator  # noqa: E402
from csvhelpers import _read  # noqa: E402


# ---------------------------------------------------------------------------
# prompt files
# ---------------------------------------------------------------------------

def _csv_prompt(path: Path, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["role", "content"])
        w.writerows(rows)
    return path


def test_a_csv_prompt_reads_its_turns(tmp_path):
    p = _csv_prompt(tmp_path / "Mood.csv", [
        ("system", "You rate moods."),
        ("user", "Rate: hello"), ("assistant", '{"mood": 4}'),
        ("user", "Rate this from 1 to 7:\n\n{text}"),
    ])
    prompt = al.read_prompt(p)
    assert prompt.name == "Mood" and prompt.system == "You rate moods."
    assert prompt.examples == [("Rate: hello", '{"mood": 4}')]
    assert prompt.user.endswith("{text}")
    assert al.prompt_problem(p) == ""
    label, note = al.describe_prompt(p)
    assert label == "Mood" and "1 example" in note and "a system turn" in note


def test_a_txt_prompt_is_the_whole_file(tmp_path):
    p = tmp_path / "Summary.txt"
    p.write_text("Summarize this in one sentence.\nKeep it plain.\n", encoding="utf-8")
    assert al.read_prompt(p).user == "Summarize this in one sentence.\nKeep it plain."
    assert al.describe_prompt(p)[1] == "one question"


def test_a_declared_fields_line_is_refused_with_what_to_do_instead(tmp_path):
    """Fields used to be declared and an instruction added; the prompt is
    now sent as written, so a fields line would reach the model as text."""
    txt = tmp_path / "Old.txt"
    txt.write_text("fields: mood: number 1-7\nRate the mood.\n", encoding="utf-8")
    assert "ask for the JSON you want in the question itself" in al.prompt_problem(txt)
    old = _csv_prompt(tmp_path / "Old.csv", [("user", "Rate it."), ("fields", "mood: number")])
    assert "`fields` row" in al.prompt_problem(old)


@pytest.mark.parametrize("rows,why", [
    ([("user", "a"), ("assistant", "b")], "last turn must be a `user`"),
    ([("system", "s"), ("user", "a"), ("system", "t"), ("user", "b")], "system turn comes first"),
    ([("user", "a"), ("user", "b"), ("user", "c")], "user/assistant pairs"),
    ([("judge", "a"), ("user", "b")], "not one of"),
])
def test_a_broken_prompt_says_what_is_wrong(tmp_path, rows, why):
    p = _csv_prompt(tmp_path / "bad.csv", rows)
    assert why in al.prompt_problem(p)


def test_placeholders_fill_from_the_row_and_the_text_is_appended_when_absent(tmp_path):
    p = tmp_path / "Age.txt"
    p.write_text("The writer is {age} years old. Rate their mood.\n", encoding="utf-8")
    prompt = al.read_prompt(p)
    assert al.placeholders(prompt) == {"age"}
    conv = al.fill(prompt, "I feel great {{really}}", {"age": "34"})
    assert conv.user == "The writer is 34 years old. Rate their mood.\n\nI feel great {{really}}"
    placed = al.Prompt(name="x", path=None, user="Text: {text}. Done {{ok}}.")
    assert al.fill(placed, "hi").user == "Text: hi. Done {ok}."


def test_the_prompt_reaches_the_model_as_written():
    """Nothing is added: no format instruction, whatever the prompt says
    about JSON."""
    asks = al.Prompt(name="p", path=None, system="Be brief.",
                     user='Reply as JSON: {{"mood": 1-7}}.\n\n{text}')
    conv = al.fill(asks, "a sunny day")
    assert conv.user == 'Reply as JSON: {"mood": 1-7}.\n\na sunny day'
    assert conv.prefix == [("system", "Be brief.")]


# ---------------------------------------------------------------------------
# replies
# ---------------------------------------------------------------------------

def test_reasoning_is_split_off_and_an_unclosed_block_means_the_budget_ran_out():
    assert al.split_reasoning("<think>\nhmm\n</think>\n{\"a\": 1}") == ("hmm", '{"a": 1}', False)
    assert al.split_reasoning("<think>still thinking") == ("still thinking", "", True)
    assert al.split_reasoning("plain") == ("", "plain", False)
    # a template that opened the block itself: only the close is in the reply
    assert al.split_reasoning("hmm, so\n</think>\n{\"a\": 1}") == ("hmm, so", '{"a": 1}', False)
    # a second block that never closed, after one that did
    assert al.split_reasoning("<think>a</think>{\"x\": 1}<think>and on") == \
        ("a\n\nand on", '{"x": 1}', True)


def test_an_answer_is_read_as_json_with_the_usual_repairs():
    reply = ("Sure! Here is my answer:\n```json\n{'Mood': 6, \"Sure \": \"Yes\", side: 'For',"
             " \"Why it\": \"it is upbeat\",}\n```")
    values, cut = al.parse_answer(reply)
    assert values == {"mood": 6, "sure": "Yes", "side": "For", "why_it": "it is upbeat"}
    assert not cut
    assert al.parse_answer("I cannot say.") == (None, False)
    assert al.parse_answer("") == (None, False)
    apostrophes, _ = al.parse_answer('{"why": "the writer\'s job isn\'t easy", "n": 3,}')
    assert apostrophes == {"why": "the writer's job isn't easy", "n": 3}


def test_a_reply_cut_inside_its_json_still_yields_what_came_before_the_cut():
    """A real run: the reply budget ran out inside the `reason` string and
    the braces never closed. What was written before the cut is read, and
    the answer says it was cut."""
    cut = '{\n  "emotion": "anger",\n  "intensity": 4,\n  "reason": "The text expresses frus'
    values, unterminated = al.parse_answer(cut)
    assert unterminated and values["emotion"] == "anger" and values["intensity"] == 4
    assert values["reason"].startswith("The text expresses")


def test_known_keys_read_a_reply_whose_quotes_a_model_forgot_to_escape():
    """A real run: `"why": "...as "crazy" and "backwards"..."` is not JSON,
    and the quote repair fixes it. A quoted word followed by a comma looks
    like the end of the string, so no repair can tell; with the keys the
    run's other replies used, each value is read up to where the next key
    starts, so the quotes inside do no harm."""
    fixed = ('{\n  "Tone": "cold",\n  "why": "The writer describes them as "crazy" and '
             '"backwards". They feel trapped."\n}')
    assert al.parse_answer(fixed)[0] == {
        "tone": "cold", "why": 'The writer describes them as "crazy" and "backwards". '
                               'They feel trapped.'}
    reply = '{"Tone": "cold", "why": "they said "no", then they left"}'
    assert al.parse_answer(reply) == (None, False), "no keys known: nothing to read by"
    values, _ = al.parse_answer(reply, keys=["tone", "why"])
    assert values == {"tone": "cold", "why": 'they said "no", then they left'}


def test_an_answer_that_is_json_but_not_an_object_is_one_value():
    assert al.parse_answer("5") == ({"value": 5}, False)
    assert al.parse_answer("```json\n[1, 2]\n```") == ({"value": [1, 2]}, False)
    assert al.parse_answer("true") == ({"value": True}, False)
    assert al.parse_answer("five") == (None, False)


def test_the_dictionary_orders_keys_by_how_many_replies_used_them():
    book = al.KeyBook()
    for values in ({"why": "a", "mood": 1}, {"mood": 2, "sure": True}, None,
                   {"mood": 3, "why": "b", "reply": "x"}):
        book.see(values)
    assert book.keys() == ["mood", "why", "sure", "reply"], "ties keep the order first seen"
    assert book.answers == 4 and book.parsed == 3
    prompt = al.Prompt(name="Mood 1-7", path=None, user="q")
    assert book.columns(prompt) == ["Mood_1_7__mood", "Mood_1_7__why", "Mood_1_7__sure",
                                    "Mood_1_7__json_reply", "Mood_1_7__parse_ok"], \
        "a key spelled like a column the step writes itself gets a prefix"
    (mood, *_rest) = book.describe(prompt)
    assert mood == {"key": "mood", "column": "Mood_1_7__mood", "replies": 3, "share": 0.75,
                    "kinds": {"number": 3}}


def test_a_keys_values_across_pieces_combine_by_what_they_are():
    assert al._combine([2, 6, None]) == 4.0
    assert al._combine([True, False, True, True]) == 0.75
    assert al._combine(["joy", "joy", "anger"]) == "joy | anger"
    assert al._combine([1, "two"]) == "1 | two", "mixed values are kept as text"
    assert al._combine([None, ""]) is None


def test_a_prompts_column_prefix_is_its_stem_made_safe():
    assert al.Prompt(name="Sentiment 1-7", path=None, user="q").column == "Sentiment_1_7"
    assert al.Prompt(name="  ", path=None, user="q").column == "prompt"


def test_the_reasoning_column_comes_before_the_reply():
    prompt = al.Prompt(name="Mood", path=None, user="q")
    assert al.columns_for(prompt, keep_reasoning=True) == [
        "Mood__reasoning", "Mood__reply", "Mood__truncated", "Mood__error"]
    assert al.columns_for(prompt, keep_reasoning=False, in_pieces=True) == [
        "Mood__reply", "Mood__truncated", "Mood__chunks", "Mood__chunk_replies", "Mood__error"]


# ---------------------------------------------------------------------------
# the step, with canned replies
# ---------------------------------------------------------------------------

def _corpus(path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text", "age"])
        w.writerow(["t1", "I love this sunny day", "30"])
        w.writerow(["t2", "Everything is awful and I hate it", "41"])
        w.writerow(["t3", "The meeting is at three", "52"])
    return path


def _fake(tmp_path: Path, table: dict) -> Path:
    p = tmp_path / "fake.json"
    p.write_text(json.dumps(table), encoding="utf-8")
    return p


def _mood_and_gist(tmp_path: Path):
    mood = _csv_prompt(tmp_path / "Mood.csv", [
        ("system", "You rate moods."), ("user", "Rate the mood 1-7 as JSON:\n\n{text}")])
    gist = tmp_path / "Gist.txt"
    gist.write_text("Say why in {age}-year-old terms.\n", encoding="utf-8")
    return mood, gist


_CANNED = {
    "default": '{"why": "no idea"}',
    "contains": {"30-year-old": '{"Why": "it is bright", "tags": ["sun", "day"]}',
                 "as JSON:\n\nI love": '{"mood": 7, "sure": true}',
                 "as JSON:\n\nEverything": '{"Mood ": 1, "sure": false}',
                 "as JSON:\n\nThe meeting": "<think>hmm</think>4"}}


def test_with_parsing_off_the_replies_are_kept_as_written(tmp_path, monkeypatch):
    mood, gist = _mood_and_gist(tmp_path)
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, _CANNED)))
    out = al.generate_with_llm(
        prompt_paths=[mood, gist], model="fake-model", csv_path=_corpus(tmp_path / "c.csv"),
        text_cols=["text"], id_cols=["text_id"], pass_through_cols=["age"],
        out_features_csv=tmp_path / "f" / "llm.csv", keep_reasoning=True, verbose=False)
    rows = _read(out)
    assert list(rows[0]) == ["text_id", "age", "Mood__reasoning", "Mood__reply",
                             "Mood__truncated", "Mood__error", "Gist__reasoning", "Gist__reply",
                             "Gist__truncated", "Gist__error"]
    by = {r["text_id"]: r for r in rows}
    assert by["t1"]["Mood__reply"] == '{"mood": 7, "sure": true}'
    assert by["t3"]["Mood__reasoning"] == "hmm" and by["t3"]["Mood__reply"] == "4"
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "Replies were kept as written; nothing was read out of them." in report
    assert "## Prompt: Mood" in report and "You rate moods." in report
    assert "The prompts were sent as written" in report and "canned replies" in report
    assert not (tmp_path / "f" / "llm_keys.json").exists()
    # the record carries the prompts' content, like a dictionary's
    record = json.loads((tmp_path / "f" / "llm_settings.json").read_text(encoding="utf-8"))
    assert {e["name"] for e in record["assets"]["prompt_paths"]} == {"Mood.csv", "Gist.txt"}
    assert record["instrument"]["parse_json"] is False


def test_with_parsing_on_every_key_the_replies_used_becomes_a_column(tmp_path, monkeypatch):
    mood, gist = _mood_and_gist(tmp_path)
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, _CANNED)))
    out = al.generate_with_llm(
        prompt_paths=[mood, gist], model="fake-model", csv_path=_corpus(tmp_path / "c.csv"),
        text_cols=["text"], id_cols=["text_id"], pass_through_cols=["age"], parse_json=True,
        out_features_csv=tmp_path / "f" / "llm.csv", keep_reasoning=True, verbose=False)
    rows = _read(out)
    # keys merged across spellings, most-used first, right after the reply
    assert list(rows[0]) == ["text_id", "age", "Mood__reasoning", "Mood__reply",
                             "Mood__mood", "Mood__sure", "Mood__value", "Mood__parse_ok",
                             "Mood__truncated", "Mood__error",
                             "Gist__reasoning", "Gist__reply", "Gist__why", "Gist__tags",
                             "Gist__parse_ok", "Gist__truncated", "Gist__error"]
    by = {r["text_id"]: r for r in rows}
    assert (by["t1"]["Mood__mood"], by["t1"]["Mood__sure"]) == ("7", "1")
    assert (by["t2"]["Mood__mood"], by["t2"]["Mood__sure"]) == ("1", "0")
    assert by["t3"]["Mood__value"] == "4" and by["t3"]["Mood__mood"] == "", \
        "a bare number is valid JSON; the reasoning is never read"
    assert all(r["Mood__parse_ok"] == "1" for r in rows)
    assert by["t1"]["Gist__why"] == "it is bright" and by["t2"]["Gist__why"] == "no idea"
    assert json.loads(by["t1"]["Gist__tags"]) == ["sun", "day"], "a list is kept as JSON text"
    keys = json.loads((tmp_path / "f" / "llm_keys.json").read_text(encoding="utf-8"))
    (m, g) = keys["prompts"]
    assert [k["key"] for k in m["keys"]] == ["mood", "sure", "value"] and m["answers"] == 3
    assert g["keys"][1] == {"key": "tags", "column": "Gist__tags", "replies": 1,
                            "share": round(1 / 3, 4), "kinds": {"list": 1}}
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "read as JSON: 3 (100%)" in report
    assert "| `mood` | `Mood__mood` | 2 | 67% | number 2 |" in report
    assert "each reply's answer (not its reasoning) was read as JSON" in report
    record = json.loads((tmp_path / "f" / "llm_settings.json").read_text(encoding="utf-8"))
    assert record["instrument"]["parse_json"] is True


def test_a_reply_broken_past_repair_is_read_with_the_keys_the_others_used(tmp_path,
                                                                           monkeypatch):
    """The reason the keys are collected as the run goes: a reply whose
    JSON cannot be loaded is still read, for the keys the other replies
    used."""
    p = tmp_path / "Tone.txt"
    p.write_text("Give the tone and why, as JSON.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, {
        "default": '{"tone": "warm", "why": "kind words"}',
        "contains": {"awful": '{"tone": "cold", "why": "calls it "awful" twice"}'}})))
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                                text_cols=["text"], id_cols=["text_id"], parse_json=True,
                                out_features_csv=tmp_path / "llm.csv", verbose=False)
    by = {r["text_id"]: r for r in _read(out)}
    assert by["t2"]["Tone__tone"] == "cold" and by["t2"]["Tone__parse_ok"] == "1"
    assert by["t2"]["Tone__why"] == 'calls it "awful" twice'


def test_a_placeholder_the_table_lacks_is_refused_before_any_model_loads(tmp_path, monkeypatch):
    p = tmp_path / "Age.txt"
    p.write_text("The writer is {age}.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", "1")
    with pytest.raises(ValueError, match=r"fills \{age\}, which is not a column"):
        al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                              text_cols=["text"], id_cols=["text_id"],
                              out_features_csv=tmp_path / "llm.csv", verbose=False)


def test_a_prompt_with_a_fields_line_is_refused_before_any_model_loads(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("fields: mood: number 1-7\nRate the mood.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", "1")
    with pytest.raises(ValueError, match="turn on llm_parse_json"):
        al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                              text_cols=["text"], id_cols=["text_id"],
                              out_features_csv=tmp_path / "llm.csv", verbose=False)


def test_a_reply_that_fails_is_a_row_not_a_stopped_run(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood as JSON.\n", encoding="utf-8")

    class Flaky(FakeGenerator):
        def generate(self, conversations, **kw):
            if len(conversations) > 1:
                raise RuntimeError("batch failed")
            if "awful" in conversations[0].user:
                raise RuntimeError("the model choked")
            return ['{"mood": 5}']

    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Flaky(batch_size=4))
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                                text_cols=["text"], id_cols=["text_id"], parse_json=True,
                                out_features_csv=tmp_path / "llm.csv", verbose=False)
    by = {r["text_id"]: r for r in _read(out)}
    assert by["t1"]["Mood__mood"] == "5" and by["t1"]["Mood__parse_ok"] == "1"
    assert by["t2"]["Mood__mood"] == "" and by["t2"]["Mood__parse_ok"] == "0"
    assert by["t2"]["Mood__error"] == "RuntimeError: the model choked"


def test_a_long_text_is_cut_to_the_models_context_and_flagged(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    seen = []

    class Small(FakeGenerator):
        def generate(self, conversations, **kw):
            seen.extend(c.user for c in conversations)
            return ["3"] * len(conversations)

    monkeypatch.setattr(al, "generator_for",
                        lambda *a, **k: Small(context_length=60, batch_size=8))
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["long", " ".join(f"w{i}" for i in range(200))])
        w.writerow(["short", "two words"])
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=src, text_cols=["text"],
                                id_cols=["text_id"], out_features_csv=tmp_path / "llm.csv",
                                max_new_tokens=8, verbose=False)
    by = {r["text_id"]: r for r in _read(out)}
    assert by["long"]["Mood__truncated"] == "1" and by["short"]["Mood__truncated"] == "0"
    long_turn = next(u for u in seen if "w0 " in u)
    assert "w199" not in long_turn and "w0" in long_turn


#: The built-in prompts that ask for JSON, and the columns their answer
#: format gives -- the lists docs/guides/bundled-prompts.md promises.
_STOCK = {
    "Big Five (BFI-2)": [f"{d}_{part}" for d in (
        "extraversion", "agreeableness", "conscientiousness", "negative_emotionality",
        "open_mindedness") for part in ("evidence", "score")],
    "Moral foundations (MFQ-2)": ["rationale", "care", "equality", "proportionality", "loyalty",
                                  "authority", "purity"],
    "Schwartz basic values": ["rationale", "self_direction", "stimulation", "hedonism",
                              "achievement", "power", "security", "conformity", "tradition",
                              "benevolence", "universalism"],
    "Agency and communion": ["rationale", "agency", "agency_themes", "communion",
                             "communion_themes"],
    "Depressive symptoms (PHQ-9)": [f"{s}_{part}" for s in (
        "anhedonia", "depressed_mood", "sleep", "fatigue", "appetite", "worthlessness_guilt",
        "concentration", "psychomotor", "suicidal_ideation")
        for part in ("evidence", "excerpt", "score")],
    "Core affect": ["rationale", "positive", "negative", "arousal"],
    "Discrete emotions (appraisal)": ["rationale", "anger", "fear", "sadness", "disgust", "joy",
                                      "surprise"],
}


def _shipped(name: str) -> Path:
    from taters.helpers.library import KINDS, _shipped_dir

    return _shipped_dir(KINDS["prompts"]) / f"{name}.csv"


@pytest.mark.parametrize("name", sorted(_STOCK))
def test_a_built_in_prompt_reaches_the_model_as_its_authors_wrote_it(name):
    """Literal braces are doubled in the files, Taters' escape, so the model
    sees the answer format as JSON -- the Big Five's ends in `}}`, which
    unescaped would have reached the model as one brace. Its own format
    and every worked example parse into the columns the docs list."""
    import re

    path = _shipped(name)
    assert al.prompt_problem(path) == ""
    prompt = al.read_prompt(path)
    assert al.placeholders(prompt) == {"text"}
    conv = al.fill(prompt, "SAMPLE TEXT")
    assert conv.user.rstrip().endswith('SAMPLE TEXT\n"""')
    for turn in [conv.user] + [t for _r, t in conv.prefix]:
        assert "{{" not in turn and "}}}" not in turn, "a brace left doubled"
    shown = re.search(r"^\{.*\}$", conv.user, re.MULTILINE).group(0)
    values, _cut = al.parse_answer(shown)
    assert sorted(values) == sorted(_STOCK[name]), name
    # the worked examples as the model sees them, braces undoubled
    for role, answer in conv.prefix:
        if role == "assistant":
            got, _cut = al.parse_answer(answer)
            assert sorted(got) == sorted(_STOCK[name]), (name, answer)


def test_nested_answers_are_one_column_per_inner_value():
    values, _ = al.parse_answer('{"Extraversion": {"evidence": "talks a lot", "Score": 4}, '
                                '"themes": ["a", "b"]}')
    assert values == {"extraversion_evidence": "talks a lot", "extraversion_score": 4,
                      "themes": ["a", "b"]}


def test_the_library_knows_prompts_and_ships_ten(tmp_path, monkeypatch):
    from taters.helpers import library as lib

    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    kind = lib.KINDS["prompts"]
    assert kind.suffixes == (".csv", ".txt") and kind.deep_check is not None
    names = {e.name for e in lib.entries(kind)}
    assert {"Sentiment 1-7.csv", "Emotion with reason.csv", "One-sentence summary.txt"} <= names
    assert {f"{n}.csv" for n in _STOCK} <= names and len(names) == 10
    for entry in lib.entries(kind):
        assert al.prompt_problem(entry) == "", entry.name
    label, note = kind.describe_entry(lib.kind_dir(kind) / "Emotion with reason.csv")
    assert label == "Emotion with reason" and "2 example(s)" in note


def test_the_recipe_is_wired_like_the_other_feature_steps():
    from taters.ui import recipes as r

    recipe = r.by_id("generate_with_llm")
    assert recipe.feature_table and recipe.library == {"prompt_paths": "prompts"}
    assert recipe.encoder_param == "model" and recipe.encoder_kind == "generative"
    assert recipe.gpu_use == "gpu_one_model"
    assert recipe.with_["out_features_csv"].endswith("/llm.csv")
    for name, spec in r.by_id("generate_with_llm").vars.items():
        assert spec["desc"], name
    # long texts and parsing are ordinary settings of the step, on the
    # options screen with their defaults
    from taters.ui.introspect import describe, load_target

    params = {q.name: q for q in describe(load_target(recipe.target)).params}
    assert params["long_texts"].choices == ["cut", "notes", "pieces"]
    assert recipe.with_["long_texts"] == "{{var:llm_long_texts}}"
    assert recipe.vars["llm_long_texts"]["default"] == "cut"
    assert recipe.with_["context_length"] == "{{var:llm_context_length}}"
    assert recipe.with_["parse_json"] == "{{var:llm_parse_json}}"
    assert recipe.vars["llm_parse_json"]["default"] is False


def test_the_fake_generator_answers_by_substring():
    gen = FakeGenerator({"default": "d", "contains": {"sun": "bright"}})
    got = gen.generate([Conversation(user="a sunny day"), Conversation(user="rain")],
                       max_new_tokens=8)
    assert got == ["bright", "d"] and len(gen.calls) == 1


def test_a_composed_pipeline_runs_the_step_and_the_parsed_answers_reach_the_statistics(
        tmp_path, monkeypatch):
    """The whole road: the wizard's composer builds a preset with the step
    and a ridge, the runner runs it with canned replies, the parsed key is
    a feature the ridge fits on, and the bookkeeping columns are kept
    aside."""
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from taters.pipelines.run_pipeline import run_preset
    from taters.ui.compose import compose

    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    src = tmp_path / "study.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text", "openness"])
        for i in range(30):
            w.writerow([f"p{i}", f"text number {i} about {'cats' if i % 2 else 'dogs'}",
                        f"{3 + (i % 5) * 0.3:.2f}"])
    prompt = tmp_path / "Mood.txt"
    prompt.write_text('Rate the mood. Answer as JSON: {{"mood": 1-7}}.\n', encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, {
        "default": '{"mood": 2}', "contains": {"cats": '{"mood": 6}'}})))
    preset = compose(
        ["generate_with_llm", "stats_ridge_fit"], source="csv", input_path=str(src),
        text_cols=["text"], id_cols=["pid"], name="llm study",
        var_values={"llm_parse_json": True, "stats_outcome_cols": ["openness"],
                    "stats_meta_carry": ["openness"], "llm_prompts": str(prompt),
                    "llm_model": "fake-model", "explain_predictions": "off"})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    answers = _read(work / "features" / "llm.csv")
    assert len(answers) == 30 and {r["Mood__mood"] for r in answers} == {"6", "2"}
    sets = json.loads((work / "stats_results" / "analysis_table_sets.json").read_text(
        encoding="utf-8"))
    assert sets["sets"]["llm"] == ["Mood__mood"], sets["sets"]
    assert set(sets["bookkeeping"]["llm"]) == {"Mood__parse_ok", "Mood__truncated"}, \
        "the text bookkeeping columns are dropped as non-numeric, the numeric ones kept aside"
    coefs = _read(work / "stats_results" / "ridge_coefficients.csv")
    assert any(r["predictor"] == "Mood__mood" for r in coefs)


def test_an_interrupted_run_learns_the_keys_of_the_rows_it_kept(tmp_path, monkeypatch):
    """The dictionary is rebuilt from the rows an earlier run kept, so a
    key only those rows used is still a column, and every row is in the
    finished table once."""
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood as JSON.\n", encoding="utf-8")
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(12):
            w.writerow([f"t{i}", f"text {i}"])

    class Dies(FakeGenerator):
        def generate(self, conversations, **kw):
            if any("text 8" in c.user for c in conversations):
                raise KeyboardInterrupt
            return ['{"early": 1}'] * len(conversations)

    class Lives(FakeGenerator):
        def generate(self, conversations, **kw):
            return ['{"late": 2}'] * len(conversations)

    run = dict(prompt_paths=[p], model="m", csv_path=src, text_cols=["text"],
               id_cols=["text_id"], out_features_csv=tmp_path / "f" / "llm.csv",
               parse_json=True, verbose=False)
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Dies(batch_size=4))
    with pytest.raises(KeyboardInterrupt):
        al.generate_with_llm(**run)
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Lives(batch_size=4))
    out = al.generate_with_llm(**run)
    rows = _read(out)
    assert sorted(r["text_id"] for r in rows) == sorted(f"t{i}" for i in range(12))
    assert {"Mood__early", "Mood__late"} <= set(rows[0])
    assert sum(r["Mood__early"] == "1" for r in rows) == 8
    assert sum(r["Mood__late"] == "2" for r in rows) == 4
    assert sorted(x.name for x in (tmp_path / "f").iterdir()) == [
        "llm.csv", "llm_keys.json", "llm_report.md", "llm_settings.json"]


# ---------------------------------------------------------------------------
# long texts
# ---------------------------------------------------------------------------

def _words(text):
    return len(text.split())


def test_a_long_text_is_split_between_sentences_and_a_long_sentence_between_words():
    text = "One two three. Four five six. Seven eight nine ten eleven twelve thirteen."
    assert al.pieces_of(text, _words, 8) == [
        "One two three. Four five six.", "Seven eight nine ten eleven twelve thirteen."]
    run_on = " ".join(f"w{i}" for i in range(20))
    pieces = al.pieces_of(run_on, _words, 8)
    assert [len(x.split()) for x in pieces] == [8, 8, 4] and " ".join(pieces) == run_on
    assert al.pieces_of(text, _words, 100) == [text], "a text that fits is one piece"
    assert al.pieces_of("", _words, 8) == [""] and al.pieces_of(text, _words, None) == [text]


def _sentences(tag: str, n: int, words: int) -> str:
    return " ".join(f"{tag}{k} " + " ".join(["filler"] * (words - 2)) + " done."
                    for k in range(1, n + 1))


def _room_for(prompt_path, room: int) -> int:
    """A context that leaves ``room`` tokens for the text after the prompt
    and an 8-token reply, as the fake counts tokens (words)."""
    prompt = al.read_prompt(prompt_path)
    empty = FakeGenerator().count_tokens(al.fill(prompt, "").messages())
    return empty + 16 + room


def _pieces_run(tmp_path, monkeypatch, *, parse_json):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    replies = {"A1 ": '{"mood": 2, "happy": true, "kind": "a"}',
               "A4 ": '{"mood": 6, "happy": false, "kind": "b"}',
               "B1 ": '{"mood": 5, "happy": true, "kind": "b"}',
               "B4 ": "no idea",
               "brief text": '{"mood": 3, "happy": false, "kind": "b"}'}
    calls = []

    class Small(FakeGenerator):
        def generate(self, conversations, **kw):
            calls.extend(conversations)
            return super().generate(conversations, **kw)

    # three six-word sentences to a piece
    context = _room_for(p, 24)
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Small(
        {"contains": replies, "default": "{}"}, context_length=context, batch_size=8))
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["long", _sentences("A", 6, 6)])
        w.writerow(["broken", _sentences("B", 6, 6)])
        w.writerow(["brief", "a brief text"])
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=src, text_cols=["text"],
                                id_cols=["text_id"], out_features_csv=tmp_path / "f" / "llm.csv",
                                max_new_tokens=8, long_texts="pieces", parse_json=parse_json,
                                verbose=False)
    return {r["text_id"]: r for r in _read(out)}, calls


def test_pieces_are_asked_about_on_their_own_and_kept_side_by_side(tmp_path, monkeypatch):
    rows, calls = _pieces_run(tmp_path, monkeypatch, parse_json=False)
    long = rows["long"]
    assert long["Mood__chunks"] == "2" and long["Mood__truncated"] == "0"
    assert json.loads(long["Mood__chunk_replies"]) == [
        '{"mood": 2, "happy": true, "kind": "a"}', '{"mood": 6, "happy": false, "kind": "b"}']
    assert long["Mood__reply"] == ('{"mood": 2, "happy": true, "kind": "a"}\n---\n'
                                   '{"mood": 6, "happy": false, "kind": "b"}')
    assert "Mood__mood" not in long, "nothing is parsed with the switch off"
    assert all("A1 " not in c.user or "A4 " not in c.user for c in calls), \
        "each piece is its own conversation"
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "Texts read in pieces: 2 (pieces per text: median 2, most 2)" in report
    assert "each piece was asked about in its own conversation" in report


def test_the_pieces_answers_combine_key_by_key_when_parsed(tmp_path, monkeypatch):
    rows, _calls = _pieces_run(tmp_path, monkeypatch, parse_json=True)
    long = rows["long"]
    assert (long["Mood__mood"], long["Mood__happy"], long["Mood__kind"]) == ("4.0", "0.5", "a | b")
    assert long["Mood__parse_ok"] == "1"
    broken = rows["broken"]
    assert broken["Mood__parse_ok"] == "0" and broken["Mood__mood"] == "5.0", \
        "a piece that did not parse is left out, and the row says so"
    assert rows["brief"]["Mood__chunks"] == "1" and rows["brief"]["Mood__mood"] == "3"


class Reader(FakeGenerator):
    """
    A model for reading long texts with notes, strict about the one thing
    that must never happen: a call whose input and reply cap do not fit the
    context. Every call is checked and kept. A note names the part it read
    (`saw M4`); a condensing returns `condensed`; an answer counts the parts
    its notes cover. ``say`` overrides a reply by kind or part number.
    """

    def __init__(self, context, *, say=None, think=False, batch_size=8):
        super().__init__(context_length=context, batch_size=batch_size)
        self.say = say or {}
        self.think = think
        self.log = []

    def kind(self, c):
        if "Now write your notes on part" in c.user:
            return "note"
        if "Rewrite these notes as one shorter set" in c.user:
            return "condense"
        return "final"

    def generate(self, conversations, *, max_new_tokens, **kw):
        out = []
        for c in conversations:
            used = self.count_tokens(c.messages())
            assert used + max_new_tokens <= self.context_length, \
                f"a {self.kind(c)} call of {used} tokens + {max_new_tokens} overflows"
            kind = self.kind(c)
            self.log.append((kind, c, max_new_tokens))
            if kind == "note":
                k = re.search(r"Part (\d+) of \d+:", c.user).group(1)
                tag = re.search(r'"""\n(\S+)', c.user.split(f"Part {k} of")[-1])
                reply = self.say.get(("note", int(k)), f"saw {tag.group(1) if tag else '?'}")
            elif kind == "condense":
                reply = self.say.get("condense", "condensed")
            else:
                covered = c.user.count("saw ") + 100 * c.user.count("condensed")
                reply = self.say.get("final", json.dumps({"parts": covered}))
            if self.think and reply is not None:
                reply = "<think>let me see</think>" + reply
            out.append(reply)
        return out


def _long(path: Path, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text", "group"])
        for r in rows:
            w.writerow(r)
    return path


def _notes_run(tmp_path, monkeypatch, gen, rows, prompt="Rate it.\n", **kw):
    p = tmp_path / "Mood.txt"
    p.write_text(prompt, encoding="utf-8")
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: gen)
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_long(tmp_path / "c.csv", rows),
                                text_cols=["text"], id_cols=["text_id"],
                                out_features_csv=tmp_path / "f" / "llm.csv", max_new_tokens=8,
                                long_texts="notes", parse_json=True, **kw)
    return {r["text_id"]: r for r in _read(out)}


def test_notes_read_every_part_in_order_then_answer_once(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    # room enough that nothing is condensed: that has tests of its own
    gen = Reader(_room_for(p, 900))
    rows = _notes_run(tmp_path, monkeypatch, gen, [
        ["long", _sentences("M", 20, 50), "a"], ["short", "M3 a short text", "a"]],
        verbose=False)
    long = rows["long"]
    n = int(long["Mood__chunks"])
    assert n > 1 and long["Mood__error"] == ""
    assert long["Mood__parts"] == str(n), "the one answer saw the notes on every part"
    notes = json.loads(long["Mood__chunk_replies"])
    assert notes == [f"saw M{k}" for k in range(1, n + 1)][:len(notes)] or len(notes) == n
    assert rows["short"]["Mood__chunks"] == "1" and rows["short"]["Mood__parts"] == "0"
    # the first call is the smoke test on the first text, before the run
    calls = [(kind, c, cap) for kind, c, cap in gen.log[1:] if "M1 " in c.user
             or "M2 " in c.user or "saw M" in c.user]
    kinds = [kind for kind, _c, _cap in calls]
    assert kinds == ["note"] * n + ["final"], kinds
    # a note sees the notes so far and one part, never an earlier part itself
    third = calls[2][1].user
    assert "Part 1:\nsaw M" in third and "Part 2:\nsaw M" in third and "M1 filler" not in third
    assert "Rate it." in third, "the question is in every call"
    assert all(64 <= cap <= 512 or cap == calls[0][2] for _k, _c, cap in calls[:-1])
    final = calls[-1][1].user
    assert final.startswith("Rate it.") and "[This text was too long to read at once" in final
    assert all(f"Part {k}:" in final for k in range(1, n + 1)) and "filler" not in final, \
        "the answer comes from the notes, not the last part itself"
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "took notes on what it showed about the question" in report


def test_examples_and_the_system_turn_go_where_they_belong(tmp_path, monkeypatch):
    """A note-taking call keeps the system turn (the coder's role) but not
    the worked examples, whose answers would teach the answer format; the
    one answer gets the prompt exactly as written."""
    p = _csv_prompt(tmp_path / "Mood.csv", [
        ("system", "You rate moods."), ("user", "Rate: a sunny day"), ("assistant", '{"parts": 9}'),
        ("user", "Rate the mood of this, from {group}:\n\n{text}")])
    gen = Reader(_room_for(p, 300) + 20)
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: gen)
    al.generate_with_llm(prompt_paths=[p], model="m",
                          csv_path=_long(tmp_path / "c.csv", [["long", _sentences("M", 7, 50), "Ohio"]]),
                          text_cols=["text"], id_cols=["text_id"], pass_through_cols=["group"],
                          out_features_csv=tmp_path / "f" / "llm.csv", max_new_tokens=8,
                          long_texts="notes", verbose=False)
    notes = [c for kind, c, _cap in gen.log if kind == "note"]
    final = [c for kind, c, _cap in gen.log if kind == "final" and "too long" in c.user]
    assert notes and final
    assert all(c.prefix == [("system", "You rate moods.")] for c in notes)
    assert all("from Ohio" in c.user for c in notes), "placeholders are filled in every call"
    assert [r for r, _t in final[0].prefix] == ["system", "user", "assistant"]
    assert "from Ohio" in final[0].user


def test_notes_that_outgrow_their_half_are_condensed_and_reading_goes_on(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 450), say={"condense": "condensed"})
    # forty parts in a small room: the floor of 64 tokens a note cannot fit
    # forty times, so the notes are condensed along the way
    for k in range(1, 41):
        gen.say[("note", k)] = " ".join(["word"] * 60)
    rows = _notes_run(tmp_path, monkeypatch, gen,
                      [["long", _sentences("M", 40, 40), "a"]], verbose=False)
    long = rows["long"]
    record = json.loads(long["Mood__chunk_replies"])
    assert any(r.startswith("[Parts 1-") and "(condensed)" in r for r in record)
    assert sum(1 for kind, _c, _cap in gen.log if kind == "note") == int(long["Mood__chunks"])
    assert sum(1 for kind, _c, _cap in gen.log[1:] if kind == "final") == 1, \
        "one answer (the first call is the smoke test)"
    report = (tmp_path / "f" / "llm_report.md").read_text(encoding="utf-8")
    assert "texts whose notes were condensed to fit: 1" in report


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_a_condensing_that_fails_is_done_without_the_model_and_cannot_loop(tmp_path, monkeypatch,
                                                                           bad):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 450), say={"condense": bad})
    for k in range(1, 41):
        gen.say[("note", k)] = " ".join(["word"] * 60)
    rows = _notes_run(tmp_path, monkeypatch, gen,
                      [["long", _sentences("M", 40, 40), "a"]], verbose=False)
    record = json.loads(rows["long"]["Mood__chunk_replies"])
    assert any("(cut short)" in r for r in record)
    assert len(gen.log) < 200, "reading finished"
    assert sum(1 for kind, _c, _cap in gen.log[1:] if kind == "final") == 1, \
        "one answer (the first call is the smoke test)"


def test_a_part_that_gets_no_notes_is_said_and_the_answer_is_still_given(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 300), say={("note", 2): None, ("note", 3): ""})
    rows = _notes_run(tmp_path, monkeypatch, gen, [["long", _sentences("M", 7, 50), "a"]],
                      verbose=False)
    long = rows["long"]
    assert long["Mood__error"].startswith("1 of ") and "part 2: no reply" in long["Mood__error"]
    record = json.loads(long["Mood__chunk_replies"])
    assert record[1] == "(this part could not be read)"
    assert record[2] == "(nothing noted in this part)"
    assert long["Mood__parts"] != "", "the answer was still given"


def test_an_answer_that_fails_is_the_rows_error(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 300), say={"final": None})
    rows = _notes_run(tmp_path, monkeypatch, gen, [["long", _sentences("M", 7, 50), "a"],
                                                   ["short", "a short one", "a"]], verbose=False)
    assert rows["long"]["Mood__error"] == "the answer: no reply"
    assert rows["short"]["Mood__error"] == "no reply"


def test_a_server_that_ignores_the_cap_is_kept_to_it(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 300))
    for k in range(1, 20):
        gen.say[("note", k)] = " ".join(["blah"] * 2000)
    rows = _notes_run(tmp_path, monkeypatch, gen, [["long", _sentences("M", 7, 50), "a"]],
                      verbose=False)
    record = json.loads(rows["long"]["Mood__chunk_replies"])
    assert all(len(r.split()) <= 512 for r in record)


def test_a_thinking_models_notes_leave_its_reasoning_out_and_get_room_for_it(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 300) + 50, think=True, say={("note", 2): "<think>and on and"})
    gen.thinking = True
    rows = _notes_run(tmp_path, monkeypatch, gen, [["long", _sentences("M", 7, 50), "a"]],
                      keep_reasoning=True, reasoning_tokens=40, verbose=False)
    record = json.loads(rows["long"]["Mood__chunk_replies"])
    assert not any("let me see" in r for r in record), "reasoning is never a note"
    assert record[1] == "(no notes: the reply ran out of room)"
    assert "ran out of room inside the model's reasoning" in rows["long"]["Mood__error"]
    caps = [cap for kind, _c, cap in gen.log if kind == "note"]
    assert caps and all(cap > 40 for cap in caps), "a note's cap includes room to think"


def test_a_text_without_spaces_is_read_in_parts_and_none_of_it_is_lost(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")

    class Chars(Reader):
        def count_tokens(self, messages):
            return sum(len(m["content"]) for m in messages) // 4

    gen = Chars(400)
    text = "这是一个没有空格的很长的中文句子" * 240
    seen = []
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: gen)
    p2 = tmp_path / "Rate2.txt"
    p2.write_text("Rate it.\n", encoding="utf-8")
    al.generate_with_llm(prompt_paths=[p2], model="m",
                          csv_path=_long(tmp_path / "c.csv", [["cjk", text, "a"]]),
                          text_cols=["text"], id_cols=["text_id"],
                          out_features_csv=tmp_path / "f" / "llm.csv", max_new_tokens=8,
                          long_texts="notes", verbose=False)
    for kind, c, _cap in gen.log:
        if kind == "note":
            seen.append(re.search(r'Part \d+ of \d+:\n"""\n(.*)\n"""\n\nNow write', c.user,
                                  re.DOTALL).group(1))
    assert len(seen) > 1 and "".join(seen) == text


def test_what_a_model_writes_in_its_notes_is_never_read_as_markup(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 900), say={("note", 1): 'braces {text} {{x}} and "quotes" """'})
    rows = _notes_run(tmp_path, monkeypatch, gen, [["long", _sentences("M", 20, 50), "a"]],
                      verbose=False)
    final = [c for kind, c, _cap in gen.log if kind == "final"][-1].user
    assert 'braces {text} {{x}} and "quotes" """' in final
    assert rows["long"]["Mood__error"] == ""


def test_a_context_too_small_for_notes_says_what_to_use_instead(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 40))
    with pytest.raises(ValueError, match="too small to read long texts with notes.*cut or pieces"):
        _notes_run(tmp_path, monkeypatch, gen, [["long", _sentences("M", 7, 50), "a"]],
                   verbose=False)


def test_the_old_name_conversation_reads_with_notes(tmp_path, monkeypatch):
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")
    gen = Reader(_room_for(p, 300))
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: gen)
    out = al.generate_with_llm(prompt_paths=[p], model="m",
                                csv_path=_long(tmp_path / "c.csv", [["long", _sentences("M", 7, 50), "a"]]),
                                text_cols=["text"], id_cols=["text_id"],
                                out_features_csv=tmp_path / "f" / "llm.csv", max_new_tokens=8,
                                long_texts="conversation", verbose=False)
    assert any(kind == "note" for kind, _c, _cap in gen.log)
    assert int(_read(out)[0]["Rate__chunks"]) > 1


def test_many_texts_are_read_side_by_side_and_each_call_carries_one_cap(tmp_path, monkeypatch):
    """Texts of different lengths in one group: a step's calls are batched
    across texts, and a batch never mixes reply caps."""
    p = tmp_path / "Rate.txt"
    p.write_text("Rate it.\n", encoding="utf-8")

    class Batches(Reader):
        def generate(self, conversations, *, max_new_tokens, **kw):
            self.sizes = getattr(self, "sizes", []) + [len(conversations)]
            return super().generate(conversations, max_new_tokens=max_new_tokens, **kw)

    gen = Batches(_room_for(p, 300), batch_size=8)
    rows = _notes_run(tmp_path, monkeypatch, gen, [
        ["a", _sentences("M", 7, 50), "x"], ["b", _sentences("M", 12, 50), "x"],
        ["c", "short", "x"], ["d", _sentences("M", 3, 50), "x"]], verbose=False)
    assert max(gen.sizes) > 1, "notes for several texts went in one call"
    assert all(r["Mood__error"] == "" for r in rows.values())
    assert int(rows["b"]["Mood__chunks"]) > int(rows["a"]["Mood__chunks"])


def test_long_texts_is_one_of_three(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", "1")
    with pytest.raises(ValueError, match="cut, notes or pieces"):
        al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                              text_cols=["text"], id_cols=["text_id"], long_texts="chunk",
                              out_features_csv=tmp_path / "llm.csv", verbose=False)


def test_a_prompt_is_found_in_the_library_by_name_with_or_without_its_suffix(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TATERS_FAKE_LLM", "1")
    for name in ("Sentiment 1-7", "Sentiment 1-7.csv"):
        out = al.generate_with_llm(prompt_paths=name, model="m",
                                    csv_path=_corpus(tmp_path / "c.csv"), text_cols=["text"],
                                    id_cols=["text_id"], out_features_csv=tmp_path / "llm.csv",
                                    overwrite_existing=True, verbose=False)
        assert "Sentiment_1_7__reply" in _read(out)[0]
    with pytest.raises(FileNotFoundError, match="not found"):
        al.generate_with_llm(prompt_paths=["No such prompt"], model="m",
                              csv_path=_corpus(tmp_path / "c.csv"), text_cols=["text"],
                              id_cols=["text_id"], out_features_csv=tmp_path / "x.csv",
                              verbose=False)


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------

def test_a_reply_that_hit_the_token_cap_is_said_in_the_error_column(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood as JSON.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, {
        "default": '{"mood": 4, "why": "it goes on and on and'})))
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                                text_cols=["text"], id_cols=["text_id"], parse_json=True,
                                out_features_csv=tmp_path / "llm.csv", verbose=False)
    rows = _read(out)
    assert all(r["Mood__mood"] == "4" for r in rows), "what came before the cut is read"
    assert all(r["Mood__error"] == "reply hit max_new_tokens before it finished" for r in rows)
    report = (tmp_path / "llm_report.md").read_text(encoding="utf-8")
    assert "replies that hit max_new_tokens: 3" in report


def test_reasoning_that_never_closed_is_a_cut_reply_with_parsing_off(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, {"default": "<think>so, well"})))
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                                text_cols=["text"], id_cols=["text_id"], keep_reasoning=True,
                                out_features_csv=tmp_path / "llm.csv", verbose=False)
    rows = _read(out)
    assert all(r["Mood__error"] == "reply hit max_new_tokens before it finished" for r in rows)
    assert all(r["Mood__reasoning"] == "so, well" and r["Mood__reply"] == "" for r in rows)


def test_kept_reasoning_gets_its_own_room_in_the_reply_budget(tmp_path, monkeypatch):
    """A thinking model spent the whole answer budget thinking and every
    reply was cut (a real run): with reasoning kept, the budget handed to
    the model is the answer's plus the reasoning allowance."""
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    seen = []

    class Watch(FakeGenerator):
        thinking = True

        def generate(self, conversations, *, max_new_tokens, **kw):
            seen.append(max_new_tokens)
            return ["<think>hmm</think>{\"mood\": 3}"] * len(conversations)

    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Watch(batch_size=8))
    al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                          text_cols=["text"], id_cols=["text_id"], keep_reasoning=True,
                          max_new_tokens=100, reasoning_tokens=500,
                          out_features_csv=tmp_path / "llm.csv", verbose=False)
    assert set(seen) == {600}, seen
    seen.clear()
    al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                          text_cols=["text"], id_cols=["text_id"], keep_reasoning=False,
                          max_new_tokens=100, reasoning_tokens=500, overwrite_existing=True,
                          out_features_csv=tmp_path / "llm.csv", verbose=False)
    assert set(seen) <= {100, 64}, seen


def test_a_run_whose_replies_keep_hitting_the_cap_stops_early_and_keeps_its_rows(tmp_path, monkeypatch):
    """Ten thousand truncated answers is the worst outcome a run can have.
    When more than half of the first twenty replies hit the cap, the run
    stops with the setting to raise, and the rows so far wait for the
    resume."""
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood as JSON.\n", encoding="utf-8")
    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        for i in range(100):
            w.writerow([f"t{i}", f"text {i}"])
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, {
        "default": '{"mood": 4, "why": "it goes on and'})))
    with pytest.raises(al.TruncatedReplies, match="hit the reply cap.*llm_max_new_tokens"):
        al.generate_with_llm(prompt_paths=[p], model="m", csv_path=src, text_cols=["text"],
                              id_cols=["text_id"], out_features_csv=tmp_path / "f" / "llm.csv",
                              parse_json=True, verbose=False)
    part = tmp_path / "f" / "llm.csv.part"
    assert part.is_file() and not (tmp_path / "f" / "llm.csv").exists()
    kept = list(csv.DictReader(part.open(encoding="utf-8-sig")))
    assert 20 <= len(kept) < 100


def test_the_text_is_served_before_the_reply_on_a_small_context(tmp_path, monkeypatch):
    """With a generous reply cap and a 4k-context model, the reply budget
    was taken off first and the text got nothing. The text keeps at least
    a third of the room; the reply budget shrinks; a context with no room
    for either is refused in a sentence."""
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood of the text below from 1 to 7, and answer with a JSON object "
                 "holding the key mood and nothing else at all.\n", encoding="utf-8")
    seen = {}

    class Small(FakeGenerator):
        def generate(self, conversations, *, max_new_tokens, **kw):
            seen["budget"] = max_new_tokens
            seen["words"] = max(len(c.user.split()) for c in conversations)
            return ["3"] * len(conversations)

    src = tmp_path / "c.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text_id", "text"])
        w.writerow(["long", " ".join(f"w{i}" for i in range(2000))])
    # words count as tokens in the fake: a 600-token context, a ~30-token prompt
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Small(context_length=600, batch_size=4))
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=src, text_cols=["text"],
                                id_cols=["text_id"], out_features_csv=tmp_path / "llm.csv",
                                max_new_tokens=1024, verbose=False)
    (row,) = _read(out)
    assert row["Mood__truncated"] == "1" and row["Mood__reply"] == "3"
    assert seen["budget"] < 1024 and seen["words"] >= 256, seen
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: Small(context_length=40, batch_size=4))
    with pytest.raises(ValueError, match="context .* too small for the prompt"):
        al.generate_with_llm(prompt_paths=[p], model="m", csv_path=src, text_cols=["text"],
                              id_cols=["text_id"], out_features_csv=tmp_path / "x.csv",
                              verbose=False)


def test_a_quote_the_model_forgot_to_escape_never_leaves_a_piece_of_the_answer():
    """A real run (Big Five, Qwen3-1.7B): the model quoted the text inside an
    evidence note and closed one score with a stray quote. The outer object
    would not load, and the first inner object that did -- one domain's
    evidence and score -- was taken for the whole answer. Both slips are
    repaired, and only the outermost object ever counts."""
    reply = ('{"extraversion": {"evidence": "mostly introspective", "score": 3}, '
             '"agreeableness": {"evidence": "a lack of concern (e.g., "All the liberals can '
             'go") for others", "score": 2}, "open_mindedness": {"evidence": "no clear '
             'evidence", "score": 3"}}')
    values, cut = al.parse_answer(reply)
    assert not cut and sorted(values) == [
        "agreeableness_evidence", "agreeableness_score", "extraversion_evidence",
        "extraversion_score", "open_mindedness_evidence", "open_mindedness_score"]
    assert values["agreeableness_evidence"] == \
        'a lack of concern (e.g., "All the liberals can go") for others'
    assert values["open_mindedness_score"] == 3
    # a quote that does end a string is left alone, digits before it or not
    assert al.parse_answer('{"why": "rated 3", "n": 2}')[0] == {"why": "rated 3", "n": 2}


# ---------------------------------------------------------------------------
# what fits, and the settings that decide it
# ---------------------------------------------------------------------------

def _summary(tmp_path, monkeypatch, **settings):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    base = dict(model="Qwen/Qwen3-1.7B", prompt_paths=["Sentiment 1-7"], max_new_tokens=1024,
                thinking="off", reasoning_tokens=4096, long_texts="cut", context_length=0)
    base.update(settings)
    return al.settings_summary(base)


def test_the_summary_says_what_the_window_leaves_for_a_text(tmp_path, monkeypatch):
    said = _summary(tmp_path, monkeypatch, context_length=40960)
    assert said.startswith("Qwen3-1.7B reads 40,960 tokens at once (as set).")
    assert "an answer may use up to 1,024." in said and "a longer one is cut to fit." in said
    assert "words) for each text" in said


def test_thinking_counts_in_the_answers_room_and_a_small_window_is_said(tmp_path, monkeypatch):
    said = _summary(tmp_path, monkeypatch, context_length=4096, thinking="drop",
                    long_texts="notes")
    assert "up to 5,120 with its thinking" in said and "read in parts, with notes" in said
    assert "answers are held to" in said and "thinking included" in said


def test_a_window_with_no_room_says_what_to_change(tmp_path, monkeypatch):
    said = _summary(tmp_path, monkeypatch, context_length=60)
    assert "no room for a text and an answer" in said


def test_a_server_that_does_not_say_is_said(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "context_window", lambda *a: (None, "the server does not say"))
    said = _summary(tmp_path, monkeypatch, model="http://127.0.0.1:9/v1#qwen")
    assert said.startswith("The server does not say how much its model reads at once")


def test_the_summary_and_the_run_share_one_budget(tmp_path, monkeypatch):
    """The screen's numbers come from `answer_budget`; so do the run's."""
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    seen = {}

    class Watch(FakeGenerator):
        def generate(self, conversations, *, max_new_tokens, **kw):
            seen.setdefault("cap", max_new_tokens)
            seen["cap"] = max(seen["cap"], max_new_tokens)
            return ["3"] * len(conversations)

    gen = Watch(context_length=600, batch_size=4)
    monkeypatch.setattr(al, "generator_for", lambda *a, **k: gen)
    al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                          text_cols=["text"], id_cols=["text_id"], max_new_tokens=1024,
                          out_features_csv=tmp_path / "llm.csv", verbose=False)
    prompt_tokens = gen.count_tokens(al.fill(al.read_prompt(p), "").messages())
    assert seen["cap"] == al.answer_budget(600, prompt_tokens, 1024)[0]


@pytest.mark.parametrize("thinking,column,room", [
    ("off", False, False), ("drop", False, True), ("keep", True, True)])
def test_thinking_off_drop_keep(tmp_path, monkeypatch, thinking, column, room):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    made = {}

    def make(*a, **k):
        made.update(k)
        return FakeGenerator({"default": "<think>hm</think>5"}, batch_size=4)

    monkeypatch.setattr(al, "generator_for", make)
    out = al.generate_with_llm(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
                                text_cols=["text"], id_cols=["text_id"], thinking=thinking,
                                max_new_tokens=100, reasoning_tokens=500,
                                out_features_csv=tmp_path / "llm.csv", verbose=False)
    row = _read(out)[0]
    assert ("Mood__reasoning" in row) is column and row["Mood__reply"] == "5"
    assert made["thinking"] is (thinking != "off")
    assert made["max_new_tokens"] == (600 if room else 100)


def test_the_old_keep_reasoning_switch_still_keeps_and_a_wrong_value_is_said(tmp_path, monkeypatch):
    p = tmp_path / "Mood.txt"
    p.write_text("Rate the mood.\n", encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", str(_fake(tmp_path, {"default": "<think>hm</think>5"})))
    run = dict(prompt_paths=[p], model="m", csv_path=_corpus(tmp_path / "c.csv"),
               text_cols=["text"], id_cols=["text_id"], verbose=False)
    out = al.generate_with_llm(keep_reasoning=True, out_features_csv=tmp_path / "a.csv", **run)
    assert _read(out)[0]["Mood__reasoning"] == "hm"
    with pytest.raises(ValueError, match="thinking is off, drop or keep"):
        al.generate_with_llm(thinking="sometimes", out_features_csv=tmp_path / "b.csv", **run)


def test_the_settings_screen_speaks_plainly_and_hides_what_does_not_apply():
    import taters.ui.wizard as w
    from taters.ui.recipes import by_id as recipe_of

    recipe = recipe_of("generate_with_llm")
    spec = w._spec_for(recipe)
    if spec is None:
        pytest.skip("the step does not import here")
    rows = {c.value: c for c in w.step_rows(recipe, spec, dict(recipe.vars), {}, {})}
    for name, row in rows.items():
        assert not row.label.startswith(name.replace("_", " ") + " "), \
            f"{name} is shown by its parameter name: {row.label}"
    assert "keep_reasoning" not in rows, "one setting for thinking, not two"
    assert "reasoning_tokens" not in rows, "room to think is hidden while thinking is off"
    assert rows["thinking"].help.startswith("Let the model reason before answering")
    assert list(rows)[:4] == ["prompt_paths", "model", "long_texts", "max_new_tokens"]
    shown = w.step_rows(recipe, spec, dict(recipe.vars), {}, {"llm_thinking": "drop"})
    assert "reasoning_tokens" in [c.value for c in shown]


def test_a_pipeline_saved_under_the_old_name_still_finds_its_step():
    """The step was `ask_language_model` before it was renamed; a saved
    pipeline names its steps by call, so the old call must still open in
    the wizard and still run."""
    from taters import Taters
    from taters.ui import recipes as r

    recipe = r.by_id("generate_with_llm")
    assert r.answers_call(recipe, "potato.text.ask_language_model")
    assert r.answers_call(recipe, "potato.text.generate_with_llm")
    assert callable(Taters().text.ask_language_model)


def test_a_failed_window_lookup_is_not_remembered_for_the_session(monkeypatch):
    """Offline when the settings screen first opened, then online: the
    screen used to keep saying the window could not be looked up until
    the process restarted, because the failure was cached with the answers."""
    import types

    from taters.text import _generation_local as gl

    seq = [types.SimpleNamespace(problem="the hub could not be reached", config={}),
           types.SimpleNamespace(problem="", config={"max_position_embeddings": 4096})]
    monkeypatch.setattr(gl, "model_facts", lambda model, **k: seq.pop(0))
    monkeypatch.setattr(al, "_FAILURE_TTL", 0.0)
    name = "org/only-in-this-test"
    assert al.context_window(name) == (None, "the hub could not be reached")
    assert al.context_window(name) == (4096, "from the model's config")
    assert al.context_window(name) == (4096, "from the model's config"), "the answer is kept"
    assert seq == []


def test_a_failed_window_lookup_is_not_repeated_on_every_redraw(monkeypatch):
    """The other half: a screen redrawn per keystroke must not wait out the
    hub's timeout each time, so a failure is remembered for a short while."""
    import types

    from taters.text import _generation_local as gl

    calls = []
    monkeypatch.setattr(gl, "model_facts", lambda model, **k: calls.append(1) or
                        types.SimpleNamespace(problem="offline", config={}))
    name = "org/only-in-this-other-test"
    for _ in range(5):
        assert al.context_window(name) == (None, "offline")
    assert calls == [1], "asked once; the redraws read the remembered failure"
