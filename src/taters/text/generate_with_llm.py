"""
Generate responses with an LLM: a stored prompt over every text, the replies
kept as features.

A prompt is a library asset -- a ``.csv`` of ``role,content`` turns (a
system turn, few-shot examples as user/assistant pairs, the final user
turn) or a ``.txt`` holding one user turn -- and the model gets it exactly
as written, with the text and any ``{column}`` filled in. Taters adds
nothing to it.

Every reply is kept as it came, with the model's reasoning (a ``<think>``
block) split off into its own column when it is kept, or dropped. With
``parse_json`` on, each reply's answer -- never its reasoning -- is tried
as JSON as the run goes, only to learn which keys the model uses; once
every text is answered, the replies are read against that dictionary of
keys as well as they can be, and each key becomes a column of the same
table, ``<prompt>__<key>``. The dictionary is written out beside the
report, so which columns exist, and how often each key was there, can be
checked.

``{text}`` in the final user turn is the text; ``{column}`` any other
column of the input table; ``{{`` and ``}}`` are literal braces. Several
prompts run in one model load, text by text, into one wide table.

Where the replies come from is :mod:`_generation`'s business: a model
loaded here, a local server, or canned answers for a dry run.
"""
from __future__ import annotations

import csv
import functools
import time
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple,
                    Union)

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec
from ..helpers.progress import announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready
from ._generation import Conversation, Generator, generator_for

__all__ = ["generate_with_llm", "Prompt", "read_prompt", "prompt_problem",
           "describe_prompt", "parse_answer", "split_reasoning", "placeholders", "fill",
           "KeyBook", "key_name", "FEATURE_COLUMNS"]

PathLike = Union[str, Path]

FEATURE_COLUMNS = ColumnSpec(
    label="Language model answers",
    names=(),
    dynamic="the replies, named <prompt>__reply after the prompt file; with parse_json, "
            "<prompt>__<key> for every key the model's JSON replies used",
)

ROLES = ("system", "user", "assistant")
_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_ .-]*)\}(?!\})")
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>(.*)$", re.DOTALL)
_FIELDS_GONE = ("declared fields are no longer read: ask for the JSON you want in the question "
                "itself, and turn on llm_parse_json to have its keys made into columns")


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

@dataclass
class Prompt:
    """A prompt as read from its file: the shared turns, the final user
    turn, and the file it came from."""

    name: str
    path: Optional[Path]
    system: str = ""
    examples: List[Tuple[str, str]] = field(default_factory=list)
    user: str = ""

    @property
    def column(self) -> str:
        """The stem as a column prefix: letters, digits and single
        underscores, so `Sentiment 1-7` names `Sentiment_1_7__reply`."""
        return re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_") or "prompt"

    @property
    def prefix(self) -> List[Tuple[str, str]]:
        turns: List[Tuple[str, str]] = []
        if self.system:
            turns.append(("system", self.system))
        for u, a in self.examples:
            turns += [("user", u), ("assistant", a)]
        return turns


def read_prompt(path: PathLike) -> Prompt:
    """
    A prompt from its file.

    ``.csv``: header ``role,content``; at most one ``system`` (first);
    ``user``/``assistant`` alternating with the last turn ``user``. ``.txt``:
    the whole file is the user turn. Raises ``ValueError`` with the reason
    when the file is not a prompt.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    name = path.stem
    if path.suffix.lower() == ".txt":
        lines = text.splitlines()
        if lines and lines[0].lower().startswith("fields:"):
            raise ValueError(f"{path.name} starts with a `fields:` line; {_FIELDS_GONE}")
        user = text.strip()
        if not user:
            raise ValueError(f"{path.name} holds no question")
        return Prompt(name=name, path=path, user=user)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"{path.name} is not a .csv or .txt prompt")
    rows = list(csv.DictReader(text.splitlines(keepends=True)))
    header = [h.strip().lower() for h in (rows[0].keys() if rows else [])]
    if not rows or header[:2] != ["role", "content"]:
        raise ValueError(f"{path.name} needs the columns `role,content`")
    system = ""
    turns: List[Tuple[str, str]] = []
    for i, r in enumerate(rows, start=2):
        role = str(r.get("role") or "").strip().lower()
        content = str(r.get("content") or "").strip()
        if not role and not content:
            continue
        if role == "fields":
            raise ValueError(f"{path.name} row {i} is a `fields` row; {_FIELDS_GONE}")
        if role not in ROLES:
            raise ValueError(f"{path.name} row {i}: role {role!r} is not one of "
                             f"{', '.join(ROLES)}")
        if role == "system":
            if turns or system:
                raise ValueError(f"{path.name} row {i}: the system turn comes first, once")
            system = content
            continue
        turns.append((role, content))
    if not turns or turns[-1][0] != "user":
        raise ValueError(f"{path.name}: the last turn must be a `user` turn, the question")
    examples: List[Tuple[str, str]] = []
    body = turns[:-1]
    if len(body) % 2:
        raise ValueError(f"{path.name}: few-shot examples come in user/assistant pairs")
    for j in range(0, len(body), 2):
        if body[j][0] != "user" or body[j + 1][0] != "assistant":
            raise ValueError(f"{path.name}: few-shot examples come in user/assistant pairs")
        examples.append((body[j][1], body[j + 1][1]))
    return Prompt(name=name, path=path, system=system, examples=examples, user=turns[-1][1])


def prompt_problem(path: PathLike) -> str:
    """Why this file is not a prompt, or ``""``: the library's deep check."""
    try:
        prompt = read_prompt(path)
    except (OSError, ValueError, csv.Error) as e:
        return str(e)
    bad = [p for p in placeholders(prompt) if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p)]
    if bad:
        return f"placeholder {{{bad[0]}}} is not a column name"
    return ""


def describe_prompt(path: PathLike) -> Tuple[str, str]:
    """The library row: the stem, and what the prompt holds."""
    path = Path(path)
    try:
        prompt = read_prompt(path)
    except (OSError, ValueError, csv.Error) as e:
        return path.stem, f"cannot be read: {e}"
    parts = [f"{len(prompt.examples)} example(s)" if prompt.examples else "one question"]
    if prompt.system:
        parts.append("a system turn")
    fills = sorted(placeholders(prompt) - {"text"})
    if fills:
        parts.append("fills " + ", ".join("{" + c + "}" for c in fills))
    return path.stem, " · ".join(parts)


def text_slots(prompt: Prompt) -> int:
    """How many times the text goes into the conversation: every
    ``{text}`` in every turn, and once more at the end of the question
    when the question does not place it."""
    turns = [prompt.system, prompt.user] + [t for pair in prompt.examples for t in pair]
    found = sum(1 for t in turns for m in _PLACEHOLDER.finditer(t)
                if m.group(1).strip() == "text")
    placed = any(m.group(1).strip() == "text" for m in _PLACEHOLDER.finditer(prompt.user))
    return max(1, found + (0 if placed else 1))


def placeholders(prompt: Prompt) -> set:
    """Every ``{name}`` the prompt's turns fill from a row, ``text`` included."""
    found = set()
    for text in [prompt.system, prompt.user] + [t for pair in prompt.examples for t in pair]:
        found.update(m.group(1).strip() for m in _PLACEHOLDER.finditer(text))
    return found


def _render(template: str, values: Dict[str, str]) -> str:
    """Placeholders filled from ``values``; ``{{``/``}}`` in the template
    are literal braces. A value is inserted as it is: a text's own braces
    are the text's, not markup."""
    out: List[str] = []
    at = 0
    for m in _PLACEHOLDER.finditer(template):
        out.append(template[at:m.start()].replace("{{", "{").replace("}}", "}"))
        out.append(values.get(m.group(1).strip(), ""))
        at = m.end()
    out.append(template[at:].replace("{{", "{").replace("}}", "}"))
    return "".join(out)


def fill(prompt: Prompt, text: str, row: Optional[Dict[str, Any]] = None) -> Conversation:
    """The conversation for one text, the prompt as written: placeholders
    filled from the row, and ``{text}`` appended as a last paragraph when
    the turn does not place it."""
    values = {str(k): ("" if v is None else str(v)) for k, v in (row or {}).items()}
    values["text"] = text
    prefix = [(role, _render(content, values)) for role, content in prompt.prefix]
    placed = any(m.group(1).strip() == "text" for m in _PLACEHOLDER.finditer(prompt.user))
    user = prompt.user if placed else prompt.user + "\n\n{text}"
    return Conversation(prefix=prefix, user=_render(user, values))


# ---------------------------------------------------------------------------
# The reply
# ---------------------------------------------------------------------------

def split_reasoning(reply: str) -> Tuple[str, str, bool]:
    """``(reasoning, answer, cut)``: every ``<think>`` block apart from the
    rest; ``cut`` when a block never closed (the budget ran out), in which
    case the answer is what came before it. A reply that only closes a
    block -- the chat template opened it, as some reasoning models' do --
    has everything before ``</think>`` as reasoning, not as the answer."""
    reply = reply or ""
    parts: List[str] = [m.group(1).strip() for m in _THINK.finditer(reply)]
    rest = _THINK.sub("", reply)
    if "</think>" in rest and "<think>" not in rest.split("</think>", 1)[0]:
        head, _sep, rest = rest.partition("</think>")
        parts.insert(0, head.strip())
    cut = False
    m = _THINK_OPEN.search(rest)
    if m:
        parts.append(m.group(1).strip())
        rest = rest[:m.start()]
        cut = True
    return "\n\n".join(x for x in parts if x), rest.strip(), cut


def _find_json(text: str) -> Optional[str]:
    """The first balanced ``{…}`` in the text, fences ignored."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _outermost_json(text: str) -> Optional[str]:
    """The object that starts at the answer's first ``{``, when it closes;
    ``None`` otherwise. Fences ignored."""
    found = _find_json(text)
    if found is None:
        return None
    bare = re.sub(r"```(?:json)?", "", text)
    return found if bare.find("{") == bare.find(found) else None


def _escape_stray_quotes(text: str) -> str:
    """
    A quote inside a string that does not end it -- the model quoted the
    text it was rating, ``"evidence": "they say "fuck off" twice"`` -- made
    into ``\\"``. A quote ends a string only when what follows it is a
    comma, a closing brace or bracket, a colon, or the end; any other quote
    met inside a string is the model's, and is escaped. A quote straight
    after a number, outside a string (``"score": 3"}``), is dropped.
    """
    out: List[str] = []
    in_str = escape = False
    for i, ch in enumerate(text):
        if not in_str:
            if ch == '"' and "".join(out).rstrip()[-1:].isdigit():
                # `"score": 3"}` -- a quote straight after a number, outside
                # any string, is never JSON; dropped (a real run)
                continue
            in_str = ch == '"'
            out.append(ch)
            continue
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            rest = text[i + 1:].lstrip()
            if not rest or rest[0] in ",}]:":
                in_str = False
            else:
                out.append("\\")
        out.append(ch)
    return "".join(out)


def _close_unterminated(text: str) -> Optional[str]:
    """An object the reply never closed (the token budget ran out inside
    it, a real run): the open string and braces closed, so the fields
    written before the cut can still be read. ``None`` without an object."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for ch in text[start:]:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    body = text[start:].rstrip()
    if in_str:
        body += '"'
    body = re.sub(r",\s*$", "", body)
    # a key left without its value ("reason": ) cannot be closed into sense
    body = re.sub(r',\s*"[^"]*"\s*:?\s*$', "", body)
    return body + "}" * max(0, depth)


def _repair(text: str) -> str:
    """The small repairs a model's JSON tends to need: single quotes,
    trailing commas, unquoted keys, Python-spelled booleans."""
    # single quotes become double only where they delimit a key or a value
    # (after `{`, `,` or `:`), never inside a string: "the writer's job
    # isn't easy" keeps its apostrophes
    fixed = re.sub(r"(?<=[{,:])(\s*)'([^'\n]*)'(?=\s*[:,}\]])", r'\1"\2"', text)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):", r'\1"\2"\3:', fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)
    fixed = re.sub(r"\bNone\b", "null", fixed)
    return fixed


def _load_object(text: str) -> Optional[dict]:
    for candidate in (text, _repair(text)):
        try:
            doc = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(doc, dict):
            return {str(k).strip().lower(): v for k, v in doc.items()}
    return None


def key_name(key: Any) -> str:
    """A key as a column suffix: lower case, runs of anything but letters
    and digits as one underscore, so "Sentiment", "sentiment " and
    "Sentiment Score" are "sentiment", "sentiment" and "sentiment_score"."""
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_") or "key"


def _keys_by_regex(text: str, keys: Sequence[str]) -> Optional[dict]:
    """
    Known keys read straight out of a reply that is not valid JSON -- a
    model wrote `"crazy"` inside a string without escaping it (a real
    run). Each key's value runs to wherever the next known key starts, or
    to the end of the object, so quotes inside a text value do no harm.
    The keys are the ones the run's other replies used. ``None`` when no
    known key is there at all.
    """
    text = re.sub(r"```(?:json)?", "", text)

    def spelled(key: str) -> str:
        # the dictionary's spelling; the reply's may use spaces or capitals
        return r"""["']?""" + r"[\s_-]*".join(re.escape(part) for part in key.split("_")) \
            + r"""["']?\s*:"""

    found: List[Tuple[int, int, str]] = []
    for key in keys:
        m = re.search(spelled(key), text, re.IGNORECASE)
        if m:
            found.append((m.start(), m.end(), key))
    if not found:
        return None
    found.sort()
    out: dict = {}
    for n, (_start, end, key) in enumerate(found):
        stop = found[n + 1][0] if n + 1 < len(found) else len(text)
        cell = text[end:stop].strip()
        cell = re.sub(r"[\s,]*}+\s*$", "", cell).strip().rstrip(",").strip()
        if len(cell) >= 2 and cell[0] == cell[-1] and cell[0] in "\"'":
            cell = cell[1:-1]
        try:
            out[key] = json.loads(cell) if cell[:1] not in "\"'" else cell
        except ValueError:
            out[key] = cell.strip()
    return out


def parse_answer(answer: str, keys: Sequence[str] = ()) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    What a reply's answer (its reasoning already split off) holds as JSON:
    ``(values, unterminated)``. ``values`` maps :func:`key_name` keys to
    what the model wrote, or is ``None`` when nothing could be read. In
    order: the JSON object the answer starts with, repaired when it needs
    the usual repairs, quotes the model forgot to escape included; an object the reply never closed (the token budget
    ran out inside it), closed so what came before the cut is read --
    ``unterminated`` then; a whole answer that is JSON but not an object
    (a bare ``5``, a list), as ``{"value": …}``; and last, ``keys`` read
    straight out of the text, for JSON broken past repairing. A nested
    object is flattened into one key per inner value
    (``extraversion_score``); a list stays one value.
    """
    answer = (answer or "").strip()
    if not answer:
        return None, False
    # the outermost object, as written, then with stray quotes escaped. never
    # a piece of it: when a quote the model forgot to escape broke the outer
    # object, the first inner object that still parsed -- one Big Five
    # domain's {"evidence", "score"} -- used to be taken for the answer (a
    # real run)
    tries = (answer, _escape_stray_quotes(answer))
    doc = None
    found = None
    for text in tries:
        found = _outermost_json(text)
        doc = _load_object(found) if found else None
        if doc is not None:
            break
    unterminated = False
    if doc is None and found is None:
        for text in tries:
            closed = _close_unterminated(text)
            doc = _load_object(closed) if closed else None
            if doc is not None:
                unterminated = True
                break
    if doc is None and found is None:
        bare = re.sub(r"^```(?:json)?|```$", "", answer).strip()
        try:
            value = json.loads(bare)
        except ValueError:
            value = None
        if value is not None and not isinstance(value, dict):
            return {"value": value}, False
    if doc is None and keys:
        doc = _keys_by_regex(answer, keys)
    if doc is None:
        return None, False
    out: Dict[str, Any] = {}
    _flatten(doc, "", out)
    return out, unterminated


def _flatten(doc: Dict[str, Any], prefix: str, out: Dict[str, Any]) -> None:
    """Nested objects as one key per inner value, the path joined by
    underscores: ``{"extraversion": {"score": 4}}`` is
    ``extraversion_score``, a column of numbers rather than a cell of
    JSON. Lists stay whole."""
    for k, v in doc.items():
        name = f"{prefix}_{key_name(k)}" if prefix else key_name(k)
        if isinstance(v, dict) and v:
            _flatten(v, name, out)
        else:
            out.setdefault(name, v)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "true/false"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "text"


def _cell(value: Any, rounding: int) -> Any:
    """A parsed value as a table cell: true/false as 1/0, a number as a
    number, a list or an object as its JSON text, null as blank."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value, rounding)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _combine(values: Sequence[Any]) -> Any:
    """One value from a key's values across a long text's pieces, read each
    on its own: numbers averaged, true/false as the share true, anything
    else as its distinct values joined in the order first seen."""
    got = [v for v in values if v is not None and v != ""]
    if not got:
        return None
    if all(isinstance(v, bool) for v in got):
        return sum(1 for v in got if v) / len(got)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in got):
        return sum(float(v) for v in got) / len(got)
    seen: List[str] = []
    for v in got:
        t = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
        if t not in seen:
            seen.append(t)
    return _PIECE_JOIN.join(seen)


#: column suffixes the step writes itself; a key the model spells the same
#: way gets a `json_` prefix rather than overwrite them
_OWN = {"reply", "reasoning", "parse_ok", "truncated", "chunks", "chunk_replies", "error"}


class KeyBook:
    """
    The keys one prompt's replies used, learned as the run goes: how many
    answers held each, the order they were first seen, and the kinds of
    value each held. Its keys become the prompt's parsed columns, and it is
    what the regex fallback reads a broken reply with.
    """

    def __init__(self) -> None:
        self.answers = 0
        self.parsed = 0
        self.counts: Dict[str, int] = {}
        self.order: Dict[str, int] = {}
        self.kinds: Dict[str, Dict[str, int]] = {}

    def see(self, values: Optional[Dict[str, Any]]) -> None:
        self.answers += 1
        if values is None:
            return
        self.parsed += 1
        for k, v in values.items():
            self.order.setdefault(k, len(self.order))
            self.counts[k] = self.counts.get(k, 0) + 1
            kinds = self.kinds.setdefault(k, {})
            kinds[_kind(v)] = kinds.get(_kind(v), 0) + 1

    def keys(self) -> List[str]:
        """Most answers first; ties in the order first seen."""
        return sorted(self.counts, key=lambda k: (-self.counts[k], self.order[k]))

    @staticmethod
    def suffix(key: str) -> str:
        return f"json_{key}" if key in _OWN else key

    def columns(self, prompt: Prompt) -> List[str]:
        return [f"{prompt.column}__{self.suffix(k)}" for k in self.keys()] + [
            f"{prompt.column}__parse_ok"]

    def describe(self, prompt: Prompt) -> List[Dict[str, Any]]:
        n = self.answers or 1
        return [{"key": k, "column": f"{prompt.column}__{self.suffix(k)}",
                 "replies": self.counts[k], "share": round(self.counts[k] / n, 4),
                 "kinds": dict(sorted(self.kinds[k].items(), key=lambda kv: -kv[1]))}
                for k in self.keys()]


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def columns_for(prompt: Prompt, *, keep_reasoning: bool, in_pieces: bool = False) -> List[str]:
    """The columns one prompt writes during the run. The reasoning, when
    kept, comes before the reply it led to; with long texts read in
    pieces, how many pieces and each piece's answer follow. Parsed columns
    are added after the run (see :class:`KeyBook`)."""
    p = prompt.column
    cols = [f"{p}__reasoning"] if keep_reasoning else []
    cols += [f"{p}__reply", f"{p}__truncated"]
    if in_pieces:
        cols += [f"{p}__chunks", f"{p}__chunk_replies"]
    cols.append(f"{p}__error")
    return cols


# ---------------------------------------------------------------------------
# Long texts
# ---------------------------------------------------------------------------

#: how a long text's pieces are joined into one cell
_PIECE_JOIN = " | "
_REPLY_JOIN = "\n---\n"


def pieces_of(text: str, measure: Callable[[str], int], budget: Optional[int]) -> List[str]:
    """
    A text in pieces that each fit ``budget`` by ``measure``, split between
    sentences; a sentence longer than the budget is split between words.
    One piece -- the text itself -- when it fits, or nothing measures it.

    Sentences are measured once each and summed (less the cost of an empty
    turn, which every measure carries), so a long text costs one pass of
    the tokenizer, not one per candidate piece; a piece the sum got wrong
    is split again by its true measure.
    """
    text = text or ""
    if budget is None or not text.strip() or measure(text) <= budget:
        return [text]
    from ._sentences import split_sentences

    if budget <= 0:
        return [text]
    overhead = measure("")
    room = max(1, budget - overhead)
    pieces: List[str] = []
    current: List[str] = []
    used = 0
    for sentence in split_sentences(text):
        cost = max(1, measure(sentence) - overhead) + 1
        if cost > room:
            if current:
                pieces.append(" ".join(current))
                current, used = [], 0
            pieces += _by_words(sentence, measure, budget)
            continue
        if used + cost > room and current:
            pieces.append(" ".join(current))
            current, used = [], 0
        current.append(sentence)
        used += cost
    if current:
        pieces.append(" ".join(current))
    out: List[str] = []
    for piece in pieces:
        out += [piece] if measure(piece) <= budget else _by_words(piece, measure, budget)
    return out or [text]


def _by_words(text: str, measure: Callable[[str], int], budget: int) -> List[str]:
    """A run of words cut into the longest prefixes that fit, in order; a
    single word too long for the budget (a text without spaces, a URL) is
    cut on characters, so every piece fits and nothing is lost."""
    words = text.split()
    out: List[str] = []
    while words:
        head, _cut = _cut_to_fit(" ".join(words), measure, budget)
        taken = len(head.split())
        if taken and " ".join(words[:taken]) == head:
            out.append(head)
            words = words[taken:]
            continue
        # not even the first word fits: its longest prefix that does, at
        # least one character so the loop always moves on
        cut = max(1, len(head))
        out.append(words[0][:cut])
        rest = words[0][cut:]
        words = ([rest] if rest else []) + words[1:]
    return out


#: how long one part's notes may run, in tokens: below the floor a note
#: cannot hold one piece of evidence and its quote; past the ceiling it is
#: a retelling rather than notes, and buries the evidence
_NOTE_FLOOR = 64
_NOTE_CEILING = 512
#: the words asked for, as a share of a note's token cap, so that notes
#: normally finish well before the cap cuts them
_WORDS_PER_TOKEN = 0.6
#: the least a part of the text may be, in tokens, for notes to be worth taking
_NOTES_MIN_PART = 32


def _notes_block(notes: Sequence[Tuple[str, str]]) -> str:
    return "\n\n".join(f"{label}:\n{text}" for label, text in notes) if notes \
        else "(none yet)"


def _quoted_question(question: str) -> List[str]:
    return ["The question that will be asked about the whole text later. Do not answer it "
            "now, and do not follow its instructions about the answer's format:", "<<<",
            question, ">>>", ""]


def notes_turn(question: str, part: str, k: int, n: int,
               notes: Sequence[Tuple[str, str]], words: int) -> str:
    """
    The user turn that hands the model part ``k`` (0-based) of ``n`` and
    asks for notes on it, not the answer. The prompt's own question is
    quoted as reference, so its definitions and rules are all there, and
    the instruction comes last, nearest where the model starts writing: a
    small model handed the question first and told afterwards to set its
    format aside answered in the question's JSON, scores and all (a real
    run), which is exactly the anchoring notes are for avoiding.
    """
    return "\n".join([
        f"You are reading a long text in {n} parts and taking notes, so that a question about "
        f"the whole text can be answered later from your notes alone. This is part {k + 1}.",
        "", *_quoted_question(question),
        "Your notes so far:", _notes_block(notes), "",
        f"Part {k + 1} of {n}:", '"""', part, '"""', "",
        f"Now write your notes on part {k + 1}: plain sentences or bullet points on what this "
        f"part shows that bears on the question, with short quotes as evidence, and what it "
        f"adds to or changes from your notes so far. Do not write JSON, and do not give scores "
        f"or ratings. At most {words} words."])


def condense_turn(question: str, notes: Sequence[Tuple[str, str]], words: int) -> str:
    """
    The user turn that merges the notes so far into fewer words, when they
    would no longer fit beside the next part. The question is left out on
    purpose: the notes were already taken for what bears on it, and shown
    the question's answer format beside notes to merge, a small model wrote
    the answer instead, scores and all, 16 times in 19 (a real run).
    ``question`` is accepted so every call is built the same way.
    """
    del question
    return "\n".join([
        "Below are notes taken while reading a long text, part by part. They have grown too "
        "long to keep.", "",
        "Notes:", _notes_block(notes), "",
        f"Rewrite these notes as one shorter set of at most {words} words: plain sentences or "
        f"bullet points that keep every piece of evidence and every quote, and drop "
        f"repetition. Do not write JSON, and do not give scores or ratings."])


def notes_for_answer(notes: Sequence[Tuple[str, str]], n: int) -> str:
    """What goes in the text's place for the one final answer: the notes on
    every part, in order, said to be notes."""
    return (f"[This text was too long to read at once. It was read in {n} parts, and below "
            f"are notes on each part, taken while reading it in order. Give your answer for "
            f"the whole text from these notes.]\n\n" + _notes_block(notes))


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

_TEXTS_PER_SLICE = 256
#: after this many replies to one prompt, more than half of them cut off
#: at the cap stops the run: ten thousand truncated answers is the worst
#: outcome a run can have, and the rows so far are kept for the resume
_CUT_GUARD_AFTER = 20
_CUT_GUARD_SHARE = 0.5
#: the least a text and a reply each get of a model's context
_TEXT_FLOOR = 256
_REPLY_FLOOR = 8


class TruncatedReplies(RuntimeError):
    """Most replies hit the token cap; the run stopped early on purpose."""


def prompt_files(prompt_paths: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    """The prompt files named: each a path, or a name in the prompts library
    (with or without its suffix); a folder means every prompt in it."""
    from ..helpers.library import KINDS, expand, kind_dir

    if isinstance(prompt_paths, (str, Path)):
        prompt_paths = [prompt_paths]
    given: List[Path] = []
    for item in prompt_paths or []:
        path = Path(str(item))
        if not path.exists():
            shelf = kind_dir(KINDS["prompts"])
            found = [shelf / path.name] + [shelf / f"{path.name}{suffix}"
                                            for suffix in KINDS["prompts"].suffixes]
            shelved = next((f for f in found if f.exists()), None)
            if shelved is None:
                raise FileNotFoundError(
                    f"prompt {item!s} not found, neither as a path nor in your prompts "
                    f"library ({kind_dir(KINDS['prompts'])})")
            path = shelved
        given.append(path)
    return [Path(f) for f in expand(KINDS["prompts"], [str(p) for p in given])]


def answer_budget(context: int, prompt_tokens: int, asked: int) -> Tuple[int, int]:
    """
    How one model call's context is shared: ``(reply, text)`` tokens, for a
    window of ``context`` tokens, a prompt of ``prompt_tokens`` and a reply
    allowance of ``asked``. The text is served first -- at least a third
    of what is left after the prompt, never less than 256 tokens when there
    is that much -- and the reply shrinks to what remains, because a
    generous reply cap on a small window left the text no room at all. The
    run and the settings screen both ask this, so they cannot disagree.
    """
    room = context - prompt_tokens - 8
    floor = max(room // 3, min(_TEXT_FLOOR, room // 2))
    reply = max(0, min(asked, room - floor))
    return reply, max(0, room - reply)


def context_window(model: str, endpoint_url: str = "", given: int = 0) -> Tuple[Optional[int], str]:
    """
    How many tokens the model reads at once, and where that came from:
    the setting, when it is given; a server's own listing, when it says;
    the model's config otherwise. ``(None, why)`` when nothing can tell.
    An answer is remembered for the session. A failure to get one is
    remembered for :data:`_FAILURE_TTL` seconds only: long enough that a
    screen redrawn on every keystroke does not wait out the hub's timeout
    each time, short enough that it recovers once the network (or the
    download) does.
    """
    if int(given or 0) > 0:
        return int(given), "as set"
    key = (model, endpoint_url)
    failed = _window_failures.get(key)
    if failed and time.monotonic() - failed[0] < _FAILURE_TTL:
        return None, failed[1]
    try:
        found = _context_window(model, endpoint_url)
    except LookupError as e:
        _window_failures[key] = (time.monotonic(), str(e))
        return None, str(e)
    _window_failures.pop(key, None)
    return found


_FAILURE_TTL = 30.0
_window_failures: Dict[Tuple[str, str], Tuple[float, str]] = {}


@functools.lru_cache(maxsize=32)
def _context_window(model: str, endpoint_url: str) -> Tuple[Optional[int], str]:
    """The lookup itself; raises ``LookupError`` (which the cache does not
    keep) when the model or the server could not be asked."""
    from ._generation_endpoint import server_context, split_server_model

    url = endpoint_url or (model if str(model).startswith(("http://", "https://")) else "")
    if url:
        base, name = split_server_model(model, url)
        found = server_context(base, name)
        if found:
            return found, "as the server reports"
        raise LookupError("the server does not say")
    from ._generation_local import context_of, model_facts

    facts = model_facts(model)
    if facts.problem:
        raise LookupError(facts.problem)
    found = context_of(facts.config)
    return (found, "from the model's config") if found else (None, "its config does not say")



def settings_summary(settings: Dict[str, Any]) -> str:
    """
    The settings screen's few sentences on how a call's window is shared
    for these settings: what the model reads at once, what the prompts and
    the answer take, and what is left for each text. Worked out with
    :func:`answer_budget`, as the run does; the prompts are counted at
    about four characters a token, so the sentence says "about".
    """
    model = str(settings.get("model") or "")
    if not model:
        return ""
    thinking = str(settings.get("thinking") or "off")
    if settings.get("keep_reasoning") and thinking == "off":
        thinking = "keep"
    asked = int(settings.get("max_new_tokens") or 0) + (
        int(settings.get("reasoning_tokens") or 0) if thinking != "off" else 0)
    try:
        prompts = [read_prompt(p) for p in prompt_files(settings.get("prompt_paths") or [])]
    except (OSError, ValueError):
        prompts = []
    prompt_tokens = max((sum(len(m["content"]) for m in fill(p, "").messages()) // 4
                         for p in prompts), default=0)
    context, source = context_window(model, str(settings.get("endpoint_url") or ""),
                                     int(settings.get("context_length") or 0))
    if context is None:
        if source == "the server does not say":
            return ("The server does not say how much its model reads at once, so texts are "
                    "sent whole and the server cuts what does not fit. Set the context window "
                    "here to have Taters fit them instead.")
        return (f"The model's context window could not be looked up ({source}); the run "
                f"works it out when it loads the model.")
    reply, text = answer_budget(context, prompt_tokens, asked)
    name = model.rstrip("/").split("/")[-1].split("#")[-1]

    def about(n: int) -> str:
        # estimates, rounded like estimates
        return f"{round(n, -2 if n >= 1000 else -1):,}"

    what = {"cut": "cut to fit", "notes": "read in parts, with notes",
            "pieces": "read in pieces"}.get(str(settings.get("long_texts") or "cut"),
                                            "cut to fit")
    lines = [f"{name} reads {context:,} tokens at once ({source}). The prompts take about "
             f"{about(prompt_tokens)}, and an answer may use up to {asked:,}"
             + (" with its thinking" if thinking != "off" else "") + "."]
    if text <= 0 or reply < _REPLY_FLOOR:
        lines.append("That leaves no room for a text and an answer: shorten the prompts, lower "
                     "the longest answer, or use a model with a larger window.")
        return " ".join(lines)
    lines.append(f"That leaves about {about(text)} tokens (roughly {about(int(text * 0.75))} "
                 f"words) for each text; a longer one is {what}.")
    if reply < asked:
        lines.append(f"The window is small for these settings, so answers are held to "
                     f"{reply:,} tokens"
                     + (", thinking included: a long line of reasoning may be cut off"
                        if thinking != "off" else "") + ".")
    return " ".join(lines)


def _cut_to_fit(text: str, measure: Callable[[str], int], budget: int) -> Tuple[str, bool]:
    """The longest prefix of ``text`` whose measured size fits ``budget``,
    on word boundaries -- or on characters when not even the first word
    fits, which is any text written without spaces (Chinese, Japanese,
    Thai) and any long URL or encoded blob: cut by words alone, those came
    out as nothing at all. ``(text, False)`` when it fits whole."""
    if budget <= 0:
        return "", True
    if measure(text) <= budget:
        return text, False
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(" ".join(words[:mid])) <= budget:
            lo = mid
        else:
            hi = mid - 1
    if lo:
        return " ".join(words[:lo]), True
    first = words[0] if words else text
    lo, hi = 0, len(first)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(first[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return first[:lo], True


@records_settings(
    binding=TEXT_INPUT, grain=TEXT_GRAIN,
    assets={"prompt_paths": "prompts"},
    outputs=("out_features_csv", "out_report_md"),
    bookkeeping=("*__parse_ok", "*__reply", "*__reasoning", "*__truncated", "*__chunks",
                 "*__chunk_replies", "*__error"))
def generate_with_llm(
    *,
    prompt_paths: Sequence[PathLike],
    model: str = "",
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_features_csv: Optional[PathLike] = None,
    out_report_md: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    workers: int = 0,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    delimiter: str = ",",
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    joiner: str = " ",
    num_buckets: int = 64,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = "*.txt",
    id_from: str = "stem",
    include_source_path: bool = False,
    pass_through_cols: Sequence[str] = (),
    backend: Literal["auto", "transformers", "endpoint"] = "auto",
    endpoint_url: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    parse_json: bool = False,
    thinking: Literal["off", "drop", "keep"] = "off",
    reasoning_tokens: int = 4096,
    keep_reasoning: bool = False,
    long_texts: Literal["cut", "notes", "pieces"] = "cut",
    context_length: int = 0,
    precision: Literal["auto", "bf16", "fp16", "fp32", "int8", "nf4"] = "auto",
    offload: bool = False,
    batch_size: int = 0,
    concurrency: int = 4,
    hf_token: Optional[str] = None,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    rounding: int = 4,
) -> Path:
    """
    Run stored prompts over every text and keep the answers as columns.

    Parameters
    ----------
    prompt_paths : list of str or Path
        Prompt files from the library (``.csv`` of role/content turns, or a
        ``.txt`` holding one question), or folders of them. Every prompt
        runs on every text in one model load.
    model : str
        The language model: a Hugging Face name or checkpoint folder of an
        instruction-tuned causal model, or the URL of a local server
        speaking the OpenAI chat API (Ollama, LM Studio, vLLM).
    csv_path, txt_dir, analysis_csv, gathered_csv
        The text input: a spreadsheet, a folder of .txt files, or a table
        already gathered (see the other text steps).
    text_cols : list of str
        The spreadsheet column(s) holding the text.
    id_cols : list of str
        The column(s) that identify a row; they compose ``text_id``.
    mode : {"concat", "separate"}
        Several text columns: joined into one text, or asked about separately.
    group_by : list of str
        Combine rows sharing these columns into one text first.
    pattern : str
        Which files to read from a folder of texts.
    overwrite_existing : bool
        Redo the step even when its output exists.
    workers : int
        CPU threads for the local model; 0 means all but one.
    delimiter, encoding, joiner, num_buckets, max_open_bucket_files, tmp_root, recursive, id_from, include_source_path
        The usual.
    out_features_csv : str or Path, optional
        Where the answers go, one row per text: ``features/llm.csv``.
    out_report_md : str or Path, optional
        The report: prompts verbatim, the model and how it was loaded,
        parse rates, example replies. Beside the features by default.
    pass_through_cols : list of str
        Columns of the input carried into the output -- and the columns a
        prompt's ``{placeholder}`` may fill.
    backend : {"auto", "transformers", "endpoint"}
        Where replies come from: a model loaded here, or a local server;
        ``auto`` picks the server when ``model`` is a URL.
    endpoint_url : str
        The server's base URL for the endpoint backend.
    max_new_tokens : int, default 1024
        The longest reply, in the model's tokens. Generous on purpose: a
        model stops when its answer is done, so an unused cap costs
        nothing, while a cap that is hit leaves a reply nobody can use.
    temperature : float, default 0
        0 is greedy decoding: the same answer every time for the same
        model and settings. Above 0 samples, with ``seed``.
    top_p : float, default 1.0
        Nucleus sampling when sampling.
    seed : int, default 42
        The sampling seed; recorded either way.
    parse_json : bool, default False
        After every text is answered, read each reply's answer (never its
        reasoning) as JSON and add a column per key the model used,
        ``<prompt>__<key>``, right after the prompt's reply, with
        ``<prompt>__parse_ok``. The keys are learned from the replies as
        the run goes; ``llm_keys.json`` beside the report lists them with
        how many replies held each. Nothing is added to the prompt: ask for
        the JSON you want in it. Off, the replies are kept as written.
    thinking : {"off", "drop", "keep"}, default "off"
        Whether the model thinks before it answers (its ``<think>`` block).
        ``off`` asks it not to, where the model allows; ``drop`` lets it
        think and keeps only the answer; ``keep`` also writes the thinking
        in a column before the answer. Thinking costs time and tokens, and
        may judge better.
    reasoning_tokens : int, default 4096
        Room for the thinking on top of ``max_new_tokens`` when the model
        thinks: a model that thinks before answering writes hundreds or
        thousands of tokens first, and a budget sized for the answer alone
        cut every reply off inside its reasoning (a real run).
    keep_reasoning : bool, default False
        The old switch for ``thinking="keep"``, read as it when ``thinking``
        is left off, so saved pipelines keep working.
    long_texts : {"cut", "notes", "pieces"}, default "cut"
        A text longer than the model's context minus the prompt and the
        reply. ``cut`` keeps its start and flags the row (``__truncated``).
        ``notes`` splits it between sentences into parts and reads them in
        order, asking for notes on each part (with the notes so far in
        view) rather than an answer, then answers once from the notes on
        every part, with the prompt as written. Half the room goes to a
        part and half to the notes on all of them, so a note runs 64 to
        512 tokens; notes that would outgrow their half are condensed by
        the model first. ``pieces`` splits it the same way and asks about
        each piece in a fresh conversation; the reply is the pieces'
        answers joined, and with ``parse_json`` each key's values are
        combined: numbers averaged, true/false as the share true, anything
        else as its distinct values joined. Both add ``__chunks`` (how many
        parts) and ``__chunk_replies`` (each part's notes, or each piece's
        answer, as a JSON list). ``conversation`` is the old name of
        ``notes`` and is read as it.
    context_length : int, default 0
        The model's context in tokens, when the backend cannot tell: a
        server does not say, so without this nothing is cut or split there
        (a text is sent whole). A number here also overrides what a model
        loaded here says. 0 leaves it to the backend.
    precision : {"auto", "bf16", "fp16", "fp32", "int8", "nf4"}, default "auto"
        How the local model is loaded; ``auto`` picks what fits.
    offload : bool, default False
        Allow a model that does not fit the GPU to spill to CPU memory
        (slow); off, the step refuses instead and says what would fit.
    batch_size : int, default 0
        Conversations per model call; 0 sizes it from free memory.
    concurrency : int, default 4
        Requests in flight at once for the endpoint backend.
    hf_token : str, optional
        A Hugging Face token for a gated model.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where the local model runs.
    rounding : int, default 4
        Decimals for numeric answers.

    Returns
    -------
    Path
        The features table.
    """
    from ..helpers import tables

    if keep_reasoning and thinking == "off":
        thinking = "keep"          # the old switch, from a saved pipeline
    if thinking not in ("off", "drop", "keep"):
        raise ValueError(f"thinking is off, drop or keep, not {thinking!r}")
    keep_reasoning = thinking == "keep"
    think = thinking != "off"
    files = prompt_files(prompt_paths)
    if not files:
        raise ValueError("generate_with_llm needs at least one prompt file (prompt_paths)")
    prompts = [read_prompt(p) for p in files]
    names = [p.column for p in prompts]
    if len(set(names)) != len(names):
        raise ValueError(f"two prompts share a name, and the columns would collide: "
                         f"{', '.join(n for n in names if names.count(n) > 1)}")
    if long_texts == "conversation":
        # its name before it took notes instead of re-answering every part
        long_texts = "notes"
    if long_texts not in ("cut", "notes", "pieces"):
        raise ValueError(f"long_texts is cut, notes or pieces, not {long_texts!r}")

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers, carry_cols=list(pass_through_cols))
    out_ref = tables.output_ref(out_features_csv, source=analysis_ready,
                                default=("features", "llm"))
    # the report is a file whatever the table is; on a database run it goes
    # under the run's local folder, the way the word-vector report does
    from ._topics import local_file

    if out_ref.is_db:
        report_path = (local_file(out_report_md, out_ref, "llm_report.md") if out_report_md
                       else out_ref.local_dir() / "llm_report.md")
    else:
        report_path = (Path(out_report_md) if out_report_md
                       else out_ref.path.with_name(f"{out_ref.stem}_report.md"))
    if not overwrite_existing and tables.exists(out_ref):
        if verbose:
            print(f"Language model answers already exist; returning existing: {out_ref}")
        return tables.handle(out_ref)

    # the placeholders every prompt fills must be columns the table has,
    # said before any model is loaded
    available = set(tables.columns_of(analysis_ready))
    for prompt in prompts:
        missing = sorted((placeholders(prompt) - {"text"}) - available)
        if missing:
            raise ValueError(
                f"prompt {prompt.name!r} fills {{{missing[0]}}}, which is not a column of "
                f"the input. The columns are: {', '.join(sorted(available))}. Name it in "
                f"pass_through_cols, or fix the placeholder.")

    announce(on_progress, f"loading the language model ({model or 'none named'})")
    gen = generator_for(backend, model, endpoint_url=endpoint_url, device=device,
                        precision=precision, offload=offload, batch_size=batch_size,
                        concurrency=concurrency, hf_token=hf_token,
                        max_new_tokens=int(max_new_tokens)
                        + (int(reasoning_tokens) if think else 0),
                        thinking=think, verbose=verbose, on_progress=on_progress)
    if int(context_length or 0) > 0:
        gen.context_length = int(context_length)
    # what the partial table was made with: a change in any of it starts over
    # the reply budget: the answer's, plus room to think when the model thinks
    budget = int(max_new_tokens) + (int(reasoning_tokens) if think else 0)
    settings = {"prompts": [{"name": p.name, "text": p.path.read_text(encoding="utf-8-sig")
                             if p.path else ""} for p in prompts],
                # not `describe()`: it quotes the free memory, which is never
                # the same twice, and a resume must match
                "model": model, "backend": gen.kind,
                "max_new_tokens": budget, "temperature": temperature,
                "parse_json": bool(parse_json),
                "top_p": top_p, "seed": seed, "thinking": thinking,
                "long_texts": long_texts, "context_length": int(context_length or 0),
                "rounding": rounding,
                "pass_through_cols": list(pass_through_cols)}
    try:
        return _run(gen, prompts, analysis_ready, out_ref, report_path, model=model,
                    pass_through_cols=list(pass_through_cols), encoding=encoding,
                    delimiter=delimiter, max_new_tokens=budget,
                    temperature=temperature, top_p=top_p, seed=seed,
                    keep_reasoning=keep_reasoning, long_texts=long_texts, rounding=rounding,
                    settings=settings, verbose=verbose, on_progress=on_progress,
                    reasoning_room=int(reasoning_tokens) if think else 0,
                    parse_json=bool(parse_json), thinking=thinking)
    finally:
        gen.close()


def _run(gen: Generator, prompts: Sequence[Prompt], analysis_ready, out_ref, report_path: Path,
         *, model: str, pass_through_cols: List[str], encoding: str, delimiter: str,
         max_new_tokens: int, temperature: float, top_p: float, seed: int,
         keep_reasoning: bool, long_texts: str, rounding: int, settings: dict,
         verbose: bool, on_progress, reasoning_room: int = 0, parse_json: bool = False,
         thinking: str = "off") -> Path:
    from ..helpers import tables
    from ..helpers.gpu import device_note
    from ..helpers.progress import Ticker, count_rows

    in_pieces = long_texts != "cut"
    header = ["text_id", *pass_through_cols]
    for prompt in prompts:
        header += columns_for(prompt, keep_reasoning=keep_reasoning, in_pieces=in_pieces)
    total = count_rows(analysis_ready, on_progress=on_progress)
    announce(on_progress, device_note(
        f"asking {gen.model_label} {len(prompts)} prompt(s) about {total} text(s)",
        gen.device_name, gen.reason))
    ticker = Ticker(on_progress, total)
    stats = {p.name: {"n": 0, "ok": 0, "cut": 0, "errors": 0, "reply_cut": 0, "lengths": [],
                      "examples": [], "in_pieces": 0, "pieces": [], "condensed": 0}
             for p in prompts}
    stats_seen: Dict[str, int] = {}          # rows counted from an earlier run
    books = {p.name: KeyBook() for p in prompts}
    guard: Dict[str, dict] = {}
    asked_tokens = int(max_new_tokens)
    # what the prompt costs in tokens, without the text, per prompt: the
    # context minus that is shared between the text and the reply. the
    # text is served first -- at least a third of the room, never less
    # than 256 tokens -- and the reply budget shrinks to what is left,
    # because a generous reply cap on a small-context model was leaving
    # the text no room at all
    budgets: Dict[str, Optional[int]] = {}
    reply_budget = int(max_new_tokens)
    for prompt in prompts:
        if not gen.context_length:
            budgets[prompt.name] = None
            continue
        empty = fill(prompt, "", {c: "" for c in pass_through_cols}).messages()
        # the most a reply could have here; see `answer_budget`
        allowed, _text = answer_budget(gen.context_length, gen.count_tokens(empty), 10 ** 9)
        if allowed < _REPLY_FLOOR:
            raise ValueError(
                f"the model's context ({gen.context_length} tokens) is too small for the "
                f"prompt {prompt.name!r} ({gen.count_tokens(empty)} tokens before the text): "
                f"no room is left for a text and a reply. Shorten the prompt, or use a model "
                f"with a longer context.")
        reply_budget = min(reply_budget, allowed)
    if reply_budget < int(max_new_tokens):
        announce(on_progress, f"the model's context leaves {reply_budget} tokens for a reply "
                              f"({max_new_tokens} were allowed); long replies will be cut")
    if not gen.context_length:
        # said at the start, not only in the report: a server that does not
        # report its window cuts long texts by its own rule, silently
        announce(on_progress, "the model's context window is not known (the server does not "
                              "say), so texts are sent whole and the server cuts what does not "
                              "fit; set llm_context_length to have Taters fit them")
    for prompt in prompts:
        if budgets.get(prompt.name, 0) is not None:
            empty = fill(prompt, "", {c: "" for c in pass_through_cols}).messages()
            # a text placed twice costs twice: its room is shared between
            # the places, or a prompt quoting it twice overflowed the context
            budgets[prompt.name] = max(0, gen.context_length - gen.count_tokens(empty)
                                       - reply_budget - 8) // text_slots(prompt)
    max_new_tokens = reply_budget
    by_name = {p.name: p for p in prompts}
    punkt = [False]

    def measure(text: str) -> int:
        return gen.count_tokens([{"role": "user", "content": text}])

    blanks = {c: "" for c in pass_through_cols}

    def row_cost(row: dict, prompt: Prompt) -> int:
        """What this row's own values add to the prompt, in tokens: the
        rooms are measured with every other placeholder empty, and a long
        value in one of them took that much from the text's room unseen."""
        if not (placeholders(prompt) - {"text"}):
            return 0
        return max(0, gen.count_tokens(fill(prompt, "", row).messages())
                   - gen.count_tokens(fill(prompt, "", blanks).messages()))

    def too_long(row: dict, prompt: Prompt) -> str:
        """Why nothing can be asked for this row, or "": the prompt with the
        row's own values, and no text at all, already leaves no room for a
        reply -- a placeholder filled from a column holding a whole essay."""
        if not gen.context_length or not (placeholders(prompt) - {"text"}):
            return ""
        need = gen.count_tokens(fill(prompt, "", row).messages())
        if need + _REPLY_FLOOR + 8 <= gen.context_length and (
                long_texts != "notes" or need + max_new_tokens + 8 <= gen.context_length):
            return ""
        return (f"the prompt with this row's values takes {need} tokens of the model's "
                f"{gen.context_length}, which leaves no room for the text and a reply; "
                f"nothing was asked")

    def split(row: dict, prompt: Prompt) -> Tuple[List[str], bool]:
        """The text as the model will read it: whole, cut to fit (and
        flagged), or in pieces that each fit; no pieces at all when the
        row's own values leave no room (see `too_long`)."""
        text = str(row.get("text") or "")
        budget = budgets[prompt.name]
        if too_long(row, prompt):
            return [], False
        if budget is None or not text.strip():
            return [text], False
        extra = row_cost(row, prompt)
        budget = max(1, budget - extra // text_slots(prompt))
        if long_texts == "cut":
            shown, truncated = _cut_to_fit(text, measure, budget)
            return [shown], truncated
        if measure(text) <= budget:
            return [text], False
        if not punkt[0]:
            from ..helpers.nltk_data import ensure_punkt

            ensure_punkt(verbose=False)
            punkt[0] = True
        if long_texts == "notes":
            # half the room for a part, the other half for the notes on
            # every part together (see `geometry`), less this row's values
            budget = max(1, geometry(prompt)["part"] - extra)
        return pieces_of(text, measure, budget), False

    def opening(row: dict, prompt: Prompt, pieces: List[str]) -> Conversation:
        """The first text as the plain prompt asks about it, for the smoke
        test: cut to the plain prompt's own room, since a part sized for
        note-taking beside a large reply cap did not fit there (a sweep)."""
        first = pieces[0]
        budget = budgets[prompt.name]
        if budget is not None and first.strip():
            room = max(1, budget - row_cost(row, prompt) // text_slots(prompt))
            first = _cut_to_fit(first, measure, room)[0]
        return fill(prompt, first, row)

    geometries: Dict[str, dict] = {}

    def geometry(prompt: Prompt) -> dict:
        """
        How a text read in parts with notes shares the context, per prompt:
        a note-taking call holds the question, the notes so far, one part,
        and the note it writes; the final call holds the prompt as written
        with every part's notes in the text's place, and the reply. The
        notes get at most what the final call has room for and at most half
        of what a note-taking call has; a part gets the rest of that call,
        less the longest note. Measured on the real conversations, empty.
        """
        if prompt.name in geometries:
            return geometries[prompt.name]
        ctx = int(gen.context_length or 0)
        blank = {c: "" for c in pass_through_cols}
        question = fill(prompt, _IN_PARTS, blank).user
        system = [t for t in fill(prompt, "", blank).prefix if t[0] == "system"]
        note_cost = gen.count_tokens(Conversation(
            prefix=system, user=notes_turn(question, "", 0, 1, [], 0)).messages())
        final_cost = gen.count_tokens(fill(prompt, notes_for_answer([], 1), blank).messages())
        room_notes = ctx - 8 - note_cost - reasoning_room
        # the notes go wherever the text goes, as often as it does
        room_final = (ctx - 8 - final_cost - max_new_tokens) // text_slots(prompt)
        total = min(room_final, room_notes // 2)
        cap = min(_NOTE_CEILING, total // 2)
        part = room_notes - total - cap
        if cap < 16 or part < _NOTES_MIN_PART:
            raise ValueError(
                f"the model's context ({ctx} tokens) is too small to read long texts with "
                f"notes for the prompt {prompt.name!r}: the question takes {note_cost} tokens "
                f"and the answer {max_new_tokens}, which leaves no room for a part of the text "
                f"and notes on it. Use long_texts cut or pieces, a shorter prompt, or a model "
                f"with a longer context.")
        geometries[prompt.name] = {"notes": total, "cap": cap, "part": part}
        return geometries[prompt.name]

    def ask(conversations: List[Conversation], size: int, budget: Optional[int] = None
            ) -> Tuple[Dict[Any, Optional[str]], Dict[Any, str]]:
        """Replies by key, a model call per ``size`` conversations, and the
        reason beside every key that got none. ``budget`` is the reply cap,
        the step's own unless a call asks for less (a note)."""
        cap = int(budget or max_new_tokens)
        replies: Dict[Any, Optional[str]] = {}
        why: Dict[Any, str] = {}
        for start in range(0, len(conversations), size):
            batch = conversations[start:start + size]
            try:
                got = gen.generate(batch, max_new_tokens=cap,
                                   temperature=temperature, top_p=top_p, seed=seed)
                said = list(getattr(gen, "errors", None) or [])
            except Exception as e:  # noqa: BLE001 - one bad batch is not a lost run
                got, said = [], []
                for conv in batch:
                    try:
                        got += gen.generate([conv], max_new_tokens=cap,
                                            temperature=temperature, top_p=top_p, seed=seed)
                        said += list(getattr(gen, "errors", None) or [None])[:1]
                    except Exception as e1:  # noqa: BLE001
                        got.append(None)
                        said.append(f"{type(e1).__name__}: {e1}")
                del e
            # a backend that could not answer one conversation of a batch
            # hands back None for it and says why in `errors`
            said += [None] * (len(batch) - len(said))
            for conv, reply, reason in zip(batch, got, said):
                replies[conv.key] = reply
                if reply is None:
                    why[conv.key] = str(reason) if reason else "no reply"
        return replies, why

    def separately(rows: List[dict], plans: dict, size: int) -> dict:
        """Every piece its own conversation; one batch run over them all."""
        conversations: List[Conversation] = []
        for (i, name), (pieces, _cut) in plans.items():
            for j, piece in enumerate(pieces):
                conv = fill(by_name[name], piece, rows[i])
                conv.key = (i, name, j)
                conversations.append(conv)
        replies, why = ask(conversations, size)
        out = {}
        for (i, name), (pieces, _cut) in plans.items():
            keys = [(i, name, j) for j in range(len(pieces))]
            parts = [_read(replies.get(k)) for k in keys]
            out[(i, name)] = (parts, [(k[2], why[k]) for k in keys if k in why], 0,
                              [q["answer"] for q in parts])
        return out

    def notes_of(notes: List[Tuple[str, str]]) -> int:
        """What the notes so far take, in tokens."""
        return measure(_notes_block(notes)) - measure("") if notes else 0

    def squeeze(notes: List[Tuple[str, str]], label: str, budget: int) -> List[Tuple[str, str]]:
        """
        The notes cut to ``budget`` tokens without the model, label and all:
        the fallback when condensing fails or does not shrink them enough,
        so reading always moves on. The label is counted in the budget --
        cut first and labeled after, the notes came out a label too long,
        every time, and the reader cut them forever (a real hang).
        """
        joined = "\n\n".join(text for _l, text in notes)
        room = max(0, budget)
        for _ in range(4):
            text, _cut = _cut_to_fit(joined, measure, max(1, room))
            out = [(label, text)]
            over = notes_of(out) - budget
            if over <= 0 or room <= 1:
                return out
            room -= over
        return [(label, "")]

    def with_notes(rows: List[dict], plans: dict, size: int) -> dict:
        """
        Every long text read in order, a part at a time, taking notes on
        each with the notes so far in view, then answered once from all
        the notes. A step at a time across every text of the group: each
        text has one call to make (a note, a condensing, or its answer),
        and the calls of a step are batched by the reply cap they need.
        """
        state: Dict[Any, dict] = {}
        for (i, name), (pieces, _cut) in plans.items():
            prompt = by_name[name]
            n = len(pieces)
            st = {"pieces": list(pieces), "n": n, "k": 0, "notes": [], "record": [],
                  "errs": [], "condensed": 0, "parts_cut": 0, "final": None, "done": False}
            if n > 1:
                geo = geometry(prompt)
                st["geo"] = geo
                # the notes on all the parts share half the room; a part's
                # share is kept between the floor and the ceiling, and the
                # notes are condensed when the floor would not fit
                st["cap"] = max(min(_NOTE_FLOOR, geo["cap"]), min(geo["cap"], geo["notes"] // n))
                st["question"] = fill(prompt, _IN_PARTS, rows[i]).user
                st["system"] = [t for t in fill(prompt, "", rows[i]).prefix if t[0] == "system"]
            state[(i, name)] = st
        limit = int(gen.context_length or 0) - 8

        def fits(conv: Conversation, cap: int) -> bool:
            return gen.count_tokens(conv.messages()) + cap <= limit

        def next_call(key, st):
            """
            The one call this text makes next. When the notes so far will
            not fit beside it, the model condenses them first; when even
            that left them too long at the same point, they are cut to the
            room left, and when the part itself is what does not fit, the
            part is cut -- each of those ends in a call, so reading always
            moves on.
            """
            i, name = key
            prompt = by_name[name]
            if st["n"] == 1:
                return "final", fill(prompt, st["pieces"][0], rows[i]), max_new_tokens
            geo, notes = st["geo"], st["notes"]
            where = st["k"]
            asked = int(geo["notes"] // 2 * _WORDS_PER_TOKEN)

            def condense():
                st["condensing_at"] = where
                return ("condense", Conversation(prefix=st["system"], user=condense_turn(
                    st["question"], notes, asked)), geo["notes"] // 2 + reasoning_room)

            if where < st["n"]:
                cap = st["cap"] + reasoning_room
                words = max(1, int(st["cap"] * _WORDS_PER_TOKEN))

                def note(ns):
                    return Conversation(prefix=st["system"], user=notes_turn(
                        st["question"], st["pieces"][where], where, st["n"], ns, words))

                conv = note(notes)
                if notes_of(notes) + st["cap"] <= geo["notes"] and fits(conv, cap):
                    return "note", conv, cap
                if notes and st.get("condensing_at") != where:
                    return condense()
                # condensed already at this part and still too long: cut the
                # notes to the room the part leaves, then the part if need be
                bare = note([])
                room = min(geo["notes"] - st["cap"],
                           limit - cap - gen.count_tokens(bare.messages()))
                if notes:
                    st["notes"] = notes = squeeze(notes, notes[0][0], room)
                    conv = note(notes)
                if not fits(conv, cap):
                    used = gen.count_tokens(conv.messages()) - measure(st["pieces"][where])
                    st["pieces"][where] = _cut_to_fit(st["pieces"][where], measure,
                                                      max(1, limit - cap - used))[0]
                    st["parts_cut"] += 1
                    conv = note(notes)
                return "note", conv, cap
            conv = fill(prompt, notes_for_answer(notes, st["n"]), rows[i])
            if fits(conv, max_new_tokens) or not notes:
                return "final", conv, max_new_tokens
            if st.get("condensing_at") != where:
                return condense()
            bare = fill(prompt, notes_for_answer([("Notes", "")], st["n"]), rows[i])
            room = limit - max_new_tokens - gen.count_tokens(bare.messages())
            st["notes"] = notes = squeeze(notes, notes[0][0], room)
            return "final", fill(prompt, notes_for_answer(notes, st["n"]), rows[i]), max_new_tokens

        while not all(st["done"] for st in state.values()):
            calls: Dict[int, List[Tuple[Any, str, Conversation]]] = {}
            for key, st in state.items():
                if st["done"]:
                    continue
                kind, conv, budget = next_call(key, st)
                conv.key = key
                calls.setdefault(int(budget), []).append((key, kind, conv))
            for budget, batch in calls.items():
                replies, why = ask([c for _k, _kind, c in batch], size, budget)
                for key, kind, conv in batch:
                    st = state[key]
                    part = _read(replies.get(key))
                    if kind == "final":
                        st["final"] = part
                        if part["reply"] is None:
                            st["errs"].append(("final", why.get(key, "no reply")))
                        st["done"] = True
                        continue
                    text = part["answer"].strip()
                    if kind == "condense":
                        covered = f"Parts 1-{st['k']}"
                        if part["reply"] is None or not text:
                            st["notes"] = squeeze(st["notes"], f"{covered} (cut short)",
                                                  st["geo"]["notes"] // 2)
                        else:
                            st["notes"] = [(f"{covered} (condensed)", text)]
                        st["record"].append(f"[{st['notes'][0][0]}] {st['notes'][0][1]}")
                        st["condensed"] += 1
                        continue
                    k = st["k"]
                    if part["reply"] is None:
                        st["errs"].append((k, why.get(key, "no reply")))
                        text = "(this part could not be read)"
                    elif part["cut"] and not text:
                        st["errs"].append((k, "its notes ran out of room inside the model's "
                                              "reasoning"))
                        text = "(no notes: the reply ran out of room)"
                    elif not text:
                        text = "(nothing noted in this part)"
                    elif measure(text) - measure("") > 2 * st["cap"]:
                        # a server that does not honor the cap: kept to it here
                        text = _cut_to_fit(text, measure, st["cap"])[0]
                    st["notes"].append((f"Part {k + 1}", text))
                    st["record"].append(text)
                    st["k"] += 1
        out = {}
        for key, st in state.items():
            final = st["final"] or _read(None)
            out[key] = ([final], st["errs"], st["condensed"], st["record"])
        return out

    def answer(rows: List[dict], writer) -> None:
        # a group of texts at a time, each group one model call wide, and
        # its rows written as soon as it is answered: what the resume can
        # pick up is what was written, so the write has to be that fine
        size = max(1, int(gen.batch_size or 1))
        per = max(1, size // max(1, len(prompts)))
        for start in range(0, len(rows), per):
            _answer_group(rows[start:start + per], writer, size)
            writer.flush()

    def learn(prompt: Prompt, answers: Sequence[str]) -> bool:
        """The keys these answers use, into the prompt's dictionary; True
        when one of them is a JSON object the token cap cut off."""
        if not parse_json:
            return False
        book = books[prompt.name]
        cut = False
        for a in answers:
            values, unterminated = parse_answer(a, book.keys())
            book.see(values)
            cut = cut or unterminated
        return cut

    def _answer_group(rows: List[dict], writer, size: int) -> None:
        plans = {(i, prompt.name): split(r, prompt)
                 for i, r in enumerate(rows) for prompt in prompts}
        askable = {k: v for k, v in plans.items() if v[0]}
        results = (with_notes if long_texts == "notes" else separately)(rows, askable, size) \
            if askable else {}
        for key, (pieces, _cut) in plans.items():
            if not pieces:
                why = too_long(rows[key[0]], by_name[key[1]])
                results[key] = ([_read(None)], [("final", why)], 0, [])
        for i, r in enumerate(rows):
            cells: List[Any] = [r.get("text_id", "")] + [r.get(c, "") for c in pass_through_cols]
            for prompt in prompts:
                key = (i, prompt.name)
                pieces, truncated = plans[key]
                parts, errs, condensed, record = results[key]
                n = len(pieces)
                if long_texts == "notes" or len(parts) == 1:
                    reply, reasoning = parts[-1]["answer"], parts[-1]["reasoning"]
                else:
                    reply = _REPLY_JOIN.join(q["answer"] for q in parts)
                    reasoning = _REPLY_JOIN.join(q["reasoning"] for q in parts if q["reasoning"])
                error = _error_of(errs, n, long_texts)
                # the dictionary learns from what becomes the row's answer: every
                # piece read on its own, or the one answer given from the notes
                answers = ([q["answer"] for q in parts] if long_texts == "pieces"
                           else [parts[-1]["answer"]])
                reply_cut = learn(prompt, answers) or any(q["cut"] for q in parts)
                if not error and reply_cut:
                    error = "reply hit max_new_tokens before it finished"
                cells += ([reasoning] if keep_reasoning else []) + [reply, int(truncated)]
                if in_pieces:
                    cells += [n, json.dumps(record, ensure_ascii=False)]
                cells.append(error)
                s = stats[prompt.name]
                s["n"] += 1
                s["cut"] += int(truncated)
                s["errors"] += int(bool(error))
                s["reply_cut"] += int(reply_cut)
                s["in_pieces"] += int(n > 1)
                if n > 1:
                    s["pieces"].append(n)
                s["condensed"] += int(bool(condensed))
                if (s["n"] >= _CUT_GUARD_AFTER and s["n"] - stats_seen.get(prompt.name, 0)
                        >= _CUT_GUARD_AFTER
                        and s["reply_cut"] / s["n"] > _CUT_GUARD_SHARE):
                    guard[prompt.name] = s
                if parts[-1]["reply"] is not None:
                    s["lengths"].append(len(reply))
                    if len(s["examples"]) < 3:
                        s["examples"].append(reply[:300])
            writer.writerow(cells)
        ticker.tick(len(rows))
        if guard:
            name, s = next(iter(guard.items()))
            writer.flush()
            raise TruncatedReplies(
                f"{s['reply_cut']} of the first {s['n']} replies to '{name}' hit the reply cap "
                f"before finishing, so the run stopped here rather than write thousands of "
                f"unusable answers. The {s['n']} rows so far are kept and the next run picks up "
                f"from them. "
                + (f"The model's context limited replies to {max_new_tokens} tokens; use a "
                   f"model with a longer context, or a shorter prompt"
                   if max_new_tokens < asked_tokens else
                   f"Raise llm_max_new_tokens (now {asked_tokens - reasoning_room})"
                   + (f" or llm_reasoning_tokens (now {reasoning_room}, the room for the "
                      f"model's thinking)" if reasoning_room else
                      ", or set llm_thinking to drop if the model thinks before answering"))
                + ", and run again.")

    def answers_in(row: Dict[str, Any], prompt: Prompt) -> List[str]:
        """What a finished row holds to parse for one prompt: each piece's
        answer for a long text read in pieces, else the reply."""
        c = prompt.column
        if long_texts == "pieces":
            try:
                each = json.loads(row.get(f"{c}__chunk_replies") or "[]")
            except ValueError:
                each = []
            if isinstance(each, list) and len(each) > 1:
                return [str(a) for a in each]
        return [str(row.get(f"{c}__reply") or "")]

    def finish(names: List[str], rows: Iterator[Dict[str, str]]):
        """After the run: every reply read against its prompt's dictionary,
        each key a column right after the prompt's reply."""
        added = {p.name: books[p.name].columns(p) for p in prompts}
        header: List[str] = []
        for c in names:
            header.append(c)
            for prompt in prompts:
                if c == f"{prompt.column}__reply":
                    header += added[prompt.name]
        announce(on_progress, "reading the replies as JSON")

        def each():
            for row in rows:
                out: Dict[str, Any] = dict(row)
                for prompt in prompts:
                    book = books[prompt.name]
                    keys = book.keys()
                    found = [parse_answer(a, keys)[0] for a in answers_in(row, prompt)]
                    ok = bool(found) and all(v is not None for v in found)
                    for k in keys:
                        values = [v.get(k) if v else None for v in found]
                        value = values[0] if len(values) == 1 else _combine(values)
                        out[f"{prompt.column}__{book.suffix(k)}"] = _cell(value, rounding)
                    out[f"{prompt.column}__parse_ok"] = int(ok)
                    stats[prompt.name]["ok"] += int(ok)
                yield out
        return header, each()

    hints = {c: "text" for c in ["text_id", *pass_through_cols]}
    for prompt in prompts:
        for c in columns_for(prompt, keep_reasoning=keep_reasoning, in_pieces=in_pieces):
            if c.endswith(("__reply", "__reasoning", "__error", "__chunk_replies")):
                hints[c] = "text"
    smoked = False
    with tables.append_table(out_ref, header, key="text_id", encoding=encoding,
                             settings=settings, hints=hints,
                             finish=finish if parse_json else None) as (writer, done):
        if done:
            announce(on_progress, f"picking up where the last run stopped: {len(done)} "
                                  f"text(s) already answered")
            ticker.tick(len(done))
            # the report counts them too, and the dictionary learns their keys
            for kept in getattr(writer, "kept", []):
                for prompt in prompts:
                    c = prompt.column
                    s = stats[prompt.name]
                    s["n"] += 1
                    stats_seen[prompt.name] = stats_seen.get(prompt.name, 0) + 1
                    s["cut"] += int(str(kept.get(f"{c}__truncated", "")) == "1")
                    s["errors"] += int(bool(kept.get(f"{c}__error", "")))
                    reply = kept.get(f"{c}__reply", "")
                    if reply:
                        s["lengths"].append(len(reply))
                    chunks = str(kept.get(f"{c}__chunks", "") or "")
                    if chunks.isdigit() and int(chunks) > 1:
                        s["in_pieces"] += 1
                        s["pieces"].append(int(chunks))
                    learn(prompt, answers_in(kept, prompt))
        pending: List[dict] = []
        for row in tables.read_rows(analysis_ready, encoding=encoding, delimiter=delimiter):
            if str(row.get("text_id", "")) in done:
                continue
            if not smoked:
                # the first text alone, before the long run: a template that
                # does not apply, a reply with no JSON in it when JSON is
                # wanted, a GPU that cannot hold one row -- all cheaper to
                # meet here. a row that cannot be asked at all is not the one
                first = split(row, prompts[0])[0]
                if first:
                    smoked = True
                    _smoke(gen, prompts[0], opening(row, prompts[0], first),
                           max_new_tokens=max_new_tokens, parse_json=parse_json,
                           on_progress=on_progress, verbose=verbose)
            pending.append(row)
            if len(pending) >= _TEXTS_PER_SLICE:
                answer(pending, writer)
                pending = []
        if pending:
            answer(pending, writer)
    _write_report(report_path, prompts, gen, stats, model=model, out_ref=out_ref,
                  max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                  seed=seed, thinking=thinking, long_texts=long_texts,
                  books=books if parse_json else None)
    if verbose:
        for prompt in prompts:
            s = stats[prompt.name]
            print(f"[llm] {prompt.name}: {s['n']} replies"
                  + (f", {s['ok']} read as JSON ({len(books[prompt.name].keys())} key(s))"
                     if parse_json else "")
                  + (f", {s['cut']} text(s) cut to fit" if s["cut"] else "")
                  + (f", {s['in_pieces']} text(s) read in pieces" if s["in_pieces"] else "")
                  + (f", {s['errors']} error(s)" if s["errors"] else ""))
    return tables.handle(out_ref)


def _error_of(errs: Sequence[Tuple[Any, str]], n: int, long_texts: str) -> str:
    """The error column's sentence for one text: the reply that could not
    be made, or the pieces or parts that got none."""
    if not errs:
        return ""
    final = [why for where, why in errs if where == "final"]
    if final:
        return final[0] if n <= 1 else f"the answer: {final[0]}"
    j, why = errs[0]
    if n == 1:
        return why
    if long_texts == "notes":
        return (f"{len(errs)} of {n} parts have no notes, and the answer was given "
                f"without them; part {j + 1}: {why}")
    return f"{len(errs)} of {n} pieces got no reply; piece {j + 1}: {why}"


#: what the text's place says in a note-taking call: the text is below
_IN_PARTS = "(The text is given in parts below.)"


def _read(reply: Optional[str]) -> Dict[str, Any]:
    """A reply as the table keeps it: the reasoning apart from the answer,
    and whether the reasoning never closed (the budget ran out in it)."""
    reasoning, answer, cut = split_reasoning(reply or "")
    return {"reply": reply, "reasoning": reasoning, "answer": answer, "cut": cut}


def _smoke(gen: Generator, prompt: Prompt, conv: Conversation, *, max_new_tokens: int,
           parse_json: bool, on_progress, verbose: bool) -> None:
    """One reply for the first text and the first prompt, with a short
    budget: what is wrong shows up now, in a sentence, not an hour in."""
    try:
        (reply,) = gen.generate([conv], max_new_tokens=min(64, int(max_new_tokens))
                                if not getattr(gen, "thinking", False) else int(max_new_tokens))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"the model could not answer the first text ({type(e).__name__}: "
                           f"{e}); nothing was written") from e
    part = _read(reply)
    if parse_json and not part["cut"] and parse_answer(part["answer"])[0] is None \
            and _close_unterminated(part["answer"]) is None:
        announce(on_progress, f"the first reply to '{prompt.name}' held no JSON; continuing, "
                              f"and how many replies did is in the report")
        if verbose:
            print(f"[llm] first reply to {prompt.name!r}: {reply!r}")


def _long_text_sentence(long_texts: str, context: Optional[int]) -> str:
    """The methods paragraph's sentence on texts longer than the model's context."""
    if not context:
        return ("The model's context length was not known, so every text was sent whole "
                "(a server truncates what does not fit by its own rule).")
    if long_texts == "pieces":
        return ("Texts longer than the model's context were split between sentences into "
                "pieces that fit; each piece was asked about in its own conversation, and "
                "every piece's answer is kept. When replies were read as JSON, each key's "
                "values across a text's pieces were combined: numbers averaged, true/false "
                "as the share true, anything else as its distinct values joined.")
    if long_texts == "notes":
        return ("Texts longer than the model's context were split between sentences into "
                "parts and read in order: for each part, the model took notes on what it "
                "showed about the question, with the notes on earlier parts in view, and "
                "gave no answer; the question was then answered once, from the notes on every "
                "part, with the prompt exactly as written. When the notes outgrew the room "
                "they were given, the model condensed them before reading on. Every part's "
                "notes are kept beside the answer.")
    return "Texts longer than the model's context were cut to fit, and flagged."


def _write_report(path: Path, prompts: Sequence[Prompt], gen: Generator, stats: dict, *,
                  model: str, out_ref, max_new_tokens: int, temperature: float, top_p: float,
                  seed: int, thinking: str, long_texts: str,
                  books: Optional[Dict[str, KeyBook]] = None) -> None:
    from ..stats._common import taters_version

    lines = ["# Language model answers", "",
             "## Methods paragraph", "",
             f"Each text was given to a language model ({gen.describe()}) with "
             f"{len(prompts)} stored prompt(s), and the reply was kept as features. Decoding "
             + ("was greedy (temperature 0), so a reply is the same for the same model, "
                "settings and hardware" if temperature == 0 else
                f"sampled at temperature {temperature:g} (top_p {top_p:g}, seed {seed})")
             + f"; replies were capped at {max_new_tokens} tokens"
             + (" (the answer's budget plus room for the reasoning)" if thinking != "off"
                else "")
             + ". "
             + {"keep": "The model reasoned before answering, and its reasoning was kept in "
                        "its own column, before the answer. ",
                "drop": "The model reasoned before answering; its reasoning was discarded "
                        "and only the answer kept. ",
                "off": "The model was asked not to reason before answering, where it "
                       "allows that, and any reasoning it wrote was discarded. "}[thinking]
             + "The prompts were sent as written, quoted below. "
             + ("After the run, each reply's answer (not its reasoning) was read as JSON, "
                "and every key the model used became a column, with keys that differed only "
                "in case or spacing taken as one; lists and nested objects were kept as JSON "
                "text, and a reply whose JSON was broken past repair was read for the keys "
                "the other replies used. " if books is not None else
                "Replies were kept as written; nothing was read out of them. ")
             + _long_text_sentence(long_texts, gen.context_length),
             "", "## Model", "",
             f"- model: `{model}`", f"- backend: {gen.kind}", f"- how: {gen.describe()}",
             f"- device: {gen.device_name or 'n/a'}"
             + (f" ({gen.reason})" if gen.reason else ""),
             f"- context length: {gen.context_length or 'unknown'}",
             ""]
    for prompt in prompts:
        s = stats[prompt.name]
        lines += [f"## Prompt: {prompt.name}", ""]
        if prompt.path:
            lines.append(f"From `{prompt.path.name}`.")
        if prompt.system:
            lines += ["", "**System:**", "", "> " + prompt.system.replace("\n", "\n> ")]
        for k, (u, a) in enumerate(prompt.examples, start=1):
            lines += ["", f"**Example {k}, user:**", "", "> " + u.replace("\n", "\n> "),
                      "", f"**Example {k}, assistant:**", "", "> " + a.replace("\n", "\n> ")]
        lines += ["", "**User:**", "", "> " + prompt.user.replace("\n", "\n> ")]
        n = s["n"] or 1
        lengths = sorted(s["lengths"])
        median = lengths[len(lengths) // 2] if lengths else 0
        lines += ["", f"Replies: {s['n']}; "
                      + (f"read as JSON: {s['ok']} ({100 * s['ok'] / n:.0f}%); "
                         if books is not None else "")
                      + (f"texts cut to fit: {s['cut']}; " if long_texts == "cut" else "")
                      + f"replies that hit max_new_tokens: "
                      f"{s['reply_cut']}; errors: {s['errors']}; reply length (characters) "
                      f"median {median}, longest {lengths[-1] if lengths else 0}."]
        if long_texts != "cut":
            counts = sorted(s["pieces"])
            lines += ["", f"Texts read in pieces: {s['in_pieces']}"
                          + (f" (pieces per text: median {counts[len(counts) // 2]}, most "
                             f"{counts[-1]})" if counts else "")
                          + (f"; texts whose notes were condensed to fit: {s['condensed']}"
                             if long_texts == "notes" else "")
                          + "."]
        if books is not None:
            found = books[prompt.name].describe(prompt)
            lines += ["", "Keys the replies used (each a column):", ""]
            if found:
                lines += ["| key | column | answers with it | share | values |",
                          "|---|---|---|---|---|"]
                lines += [f"| `{k['key']}` | `{k['column']}` | {k['replies']} | "
                          f"{100 * k['share']:.0f}% | "
                          + ", ".join(f"{kind} {c}" for kind, c in k["kinds"].items()) + " |"
                          for k in found]
            else:
                lines.append("None: no reply held JSON.")
        if s["examples"]:
            lines += ["", "Example replies:", ""]
            lines += [f"- {e!r}" for e in s["examples"]]
        lines.append("")
    lines += ["## Versions", "", f"- taters {taters_version()}", ""]
    lines += [f"Table: `{out_ref}`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    if books is not None:
        # the dictionary itself, beside the report: which columns exist and why
        doc = {"prompts": [{"prompt": p.name, "answers": books[p.name].answers,
                            "keys": books[p.name].describe(p)} for p in prompts]}
        with atomic_write(path.with_name(path.stem.replace("_report", "") + "_keys.json"),
                          mode="w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)


CLI = CliSpec({"generate": generate_with_llm},
              description="Run stored prompts over every text with a language model and "
                          "keep the answers as features.")


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
