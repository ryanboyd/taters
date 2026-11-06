"""
Chat with a language model: the conversation, without the screens.

The same models as the "Generate responses with an LLM" step, through the
same :class:`~taters.text._generation.Generator`, but one conversation that
grows turn by turn instead of one question per text. This module keeps the
transcript (every line timestamped), fits each call to the model's context
window, splits a streamed reply into its thinking and its answer, and writes
the transcript out as a CSV. The screens are :mod:`taters.ui.chat`.

A long chat outgrows any model's window. The system prompt always stays in
view; the oldest exchanges go first, and :attr:`ChatSession.out_of_view`
says how many messages the model can no longer see. The transcript keeps all
of them, and so does the export.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from ._generation import Conversation, Generator

__all__ = ["ChatMessage", "ChatSession", "ThinkFilter", "EXIT_WORDS", "CSV_COLUMNS"]

PathLike = Union[str, Path]

#: What ends a chat when it is the whole message. A slash, so that a message
#: that is just the word "exit" can still be sent.
EXIT_WORDS = ("/exit", "/quit")
CSV_COLUMNS = ("timestamp", "role", "model", "content", "reasoning")


def _now() -> datetime:
    """Local time with its offset, so a transcript read elsewhere is exact."""
    return datetime.now().astimezone()


@dataclass
class ChatMessage:
    """One line of the transcript: who said it, when, and what; a reply's
    reasoning beside it when the model thought before answering."""

    role: str
    content: str
    when: datetime = field(default_factory=_now)
    reasoning: str = ""


class ThinkFilter:
    """
    A streamed reply split into thinking and answer as it arrives:
    :meth:`feed` takes the next piece and returns ``(kind, text)`` parts,
    kind ``"thinking"`` or ``"answer"``. A tag can arrive cut across two
    pieces (``"<thi"`` then ``"nk>"``), so text that could still be the
    start of one is held back until the next piece settles it.

    ``expect_thinking`` is for a model told to think. Some chat templates
    open the thinking block themselves, so the reply starts mid-thought and
    only *closes* the block; until a tag says which it is, the text is held
    as not yet known -- shown as thinking if ``</think>`` comes, as answer
    if the reply ends without one. A model not told to think streams
    straight through.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self, expect_thinking: bool = False) -> None:
        self._held = ""                     # the end of a piece that may begin a tag
        self._thinking = False
        self._undecided = bool(expect_thinking)
        self._pending: List[str] = []       # text whose kind is not known yet

    def _split_tail(self, text: str) -> Tuple[str, str]:
        """``text`` less any ending that could still begin a tag."""
        keep = 0
        for tag in (self.OPEN, self.CLOSE):
            for n in range(1, len(tag)):
                if text.endswith(tag[:n]):
                    keep = max(keep, n)
        return (text[:-keep], text[-keep:]) if keep else (text, "")

    def feed(self, piece: str) -> List[Tuple[str, str]]:
        text = self._held + (piece or "")
        self._held = ""
        out: List[Tuple[str, str]] = []
        while text:
            opened, closed = text.find(self.OPEN), text.find(self.CLOSE)
            first = min((i for i in (opened, closed) if i != -1), default=-1)
            if first == -1:
                body, self._held = self._split_tail(text)
                if body:
                    if self._undecided:
                        self._pending.append(body)
                    else:
                        out.append(("thinking" if self._thinking else "answer", body))
                break
            before = text[:first]
            if first == closed:
                # the block closes: what came before it was thinking, even
                # when no opening tag was ever seen
                out += [("thinking", t) for t in self._pending]
                self._pending = []
                if before:
                    out.append(("thinking", before))
                self._thinking = False
                text = text[first + len(self.CLOSE):]
            else:
                out += [("answer", t) for t in self._pending]
                self._pending = []
                if before:
                    out.append(("thinking" if self._thinking else "answer", before))
                self._thinking = True
                text = text[first + len(self.OPEN):]
            self._undecided = False
        return out

    def close(self) -> List[Tuple[str, str]]:
        """What is still held when the reply ends: undecided text was the
        answer after all; an unclosed block was thinking."""
        out = [("thinking" if self._thinking else "answer", t) for t in self._pending]
        self._pending = []
        if self._held:
            out.append(("thinking" if self._thinking else "answer", self._held))
            self._held = ""
        return out


class ChatSession:
    """
    One conversation with one model.

    ``system`` is the system prompt (empty for none). ``max_new_tokens`` is
    the longest answer; ``thinking`` is ``off``, ``drop`` or ``keep``, and
    when the model thinks each reply gets ``reasoning_tokens`` more room.
    :meth:`send` returns the reply as it is written; the transcript has
    both sides once it ends.
    """

    def __init__(self, gen: Generator, *, model: str, system: str = "",
                 max_new_tokens: int = 1024, temperature: float = 0.0, top_p: float = 1.0,
                 seed: int = 42, thinking: str = "off", reasoning_tokens: int = 4096,
                 top_k: int = 0, repetition_penalty: float = 1.0):
        self.gen = gen
        self.model = model
        self.system = (system or "").strip()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.thinking = thinking
        self.reasoning_tokens = int(reasoning_tokens)
        self.started = _now()
        self.messages: List[ChatMessage] = []
        #: messages the model can no longer see, oldest first (still in the
        #: transcript and the export)
        self.out_of_view = 0

    @property
    def reply_budget(self) -> int:
        return self.max_new_tokens + (self.reasoning_tokens if self.thinking != "off" else 0)

    def _conversation(self, keep_from: int) -> Conversation:
        prefix = [("system", self.system)] if self.system else []
        history = self.messages[keep_from:-1]
        prefix += [(m.role, m.content) for m in history]
        return Conversation(prefix=prefix, user=self.messages[-1].content)

    def fit(self) -> Conversation:
        """
        The conversation for the newest message, as much of the history as
        the window holds beside it and the reply: whole exchanges are left
        out from the oldest, the system prompt never. ``ValueError`` when
        the newest message alone does not fit. A window that is not known
        (a server that does not say) takes everything.
        """
        context = getattr(self.gen, "context_length", None)
        if not context:
            self.out_of_view = 0
            return self._conversation(0)
        limit = int(context) - self.reply_budget - 8
        start = 0
        while True:
            conv = self._conversation(start)
            if self.gen.count_tokens(conv.messages()) <= limit:
                self.out_of_view = start
                return conv
            if start >= len(self.messages) - 1:
                raise ValueError(
                    f"that message is longer than the model can read beside its answer "
                    f"(it reads {int(context):,} tokens at once, {self.reply_budget:,} of "
                    f"them kept for the answer)")
            start = min(start + 2, len(self.messages) - 1)   # an exchange at a time

    def send(self, text: str) -> Iterator[str]:
        """
        Say ``text`` and stream the reply. The user's line goes in the
        transcript at once; the reply goes in when it has finished, its
        thinking split off. A message that cannot be sent is taken back out,
        so the transcript never holds a question nothing answered unless the
        model failed while answering it.
        """
        self.messages.append(ChatMessage("user", text))
        try:
            conv = self.fit()
        except ValueError:
            self.messages.pop()
            raise
        pieces: List[str] = []
        for piece in self.gen.stream(conv, max_new_tokens=self.reply_budget,
                                     temperature=self.temperature, top_p=self.top_p,
                                     seed=self.seed, top_k=self.top_k,
                                     repetition_penalty=self.repetition_penalty):
            pieces.append(piece)
            yield piece
        from .generate_with_llm import split_reasoning

        reasoning, answer, _cut = split_reasoning("".join(pieces))
        self.messages.append(ChatMessage("assistant", answer,
                                         reasoning=reasoning if self.thinking == "keep" else ""))

    def export_csv(self, path: PathLike) -> Path:
        """
        The whole transcript as a CSV: a row per message, oldest first, with
        when it was said (local time and its offset), who said it, the model,
        and the text; the system prompt first when there is one; a reply's
        reasoning in its own column when the model's thinking was kept.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows: List[ChatMessage] = []
        if self.system:
            rows.append(ChatMessage("system", self.system, when=self.started))
        rows += self.messages
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_COLUMNS)
            for m in rows:
                w.writerow([m.when.isoformat(timespec="seconds"), m.role, self.model,
                            m.content, m.reasoning])
        return path


def system_prompt_from_file(path: PathLike) -> Tuple[str, str]:
    """
    A system prompt read from a file, and a word on where it came from: a
    prompt from the library gives its system turn, or its question when it
    has none (``{text}`` taken out); any other file is used as it is.
    """
    path = Path(path)
    if path.suffix.lower() in (".csv", ".txt"):
        try:
            from .generate_with_llm import read_prompt

            prompt = read_prompt(path)
        except (OSError, ValueError):
            prompt = None
        if prompt is not None and prompt.system:
            return prompt.system, f"the system turn of {path.stem}"
        if prompt is not None and path.suffix.lower() == ".csv":
            question = prompt.user.replace("{text}", "").strip()
            return question, f"the question of {path.stem} (it has no system turn)"
    return path.read_text(encoding="utf-8-sig").strip(), path.name


def default_export_name(when: Optional[datetime] = None) -> str:
    return f"chat-{(when or _now()).strftime('%Y%m%d-%H%M%S')}.csv"
