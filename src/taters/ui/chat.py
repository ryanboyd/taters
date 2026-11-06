"""
"Chat with a language model": the screens around :mod:`taters.text.chat`.

Under Wrangle Language Models, beside training and adapting, because it is
something done *with a model* rather than with a dataset. The parts are the
ones the "Generate responses with an LLM" step already has -- the model
picker (with its download and sign-in), the prompts library, the loader --
so this module is only the conversation:

1. which model;
2. the settings screen, straight after: the system prompt (none, typed or
   pasted, from the prompts library, or from a file), the bot's name, how
   much answers vary (and, when they do, top-p, top-k and the seed), a
   repetition penalty, thinking, the longest answer;
3. the chat itself, outside the screens, so the transcript stays in the
   terminal: replies stream as they are written; ``/exit``, ``/quit``, Esc
   or Ctrl-C leave;
4. on the way out, the whole conversation saved as a CSV, if wanted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from . import glyphs
from .prompts import Cancelled, Choice, GoBack, Prompter

__all__ = ["run_chat"]

_START, _SYSTEM, _TEMPERATURE, _THINKING, _LONGEST, _ROOM = (
    ":start", "system", "temperature", "thinking", "max_new_tokens", "reasoning_tokens")
_TOP_P, _TOP_K, _SEED, _REPEAT = "top_p", "top_k", "seed", "repetition_penalty"
_BOT = "bot_name"

#: The settings a chat starts with. Unlike the step, a chat isn't a measure,
#: so it samples by default: some variety without going off the rails, about
#: what Ollama and the Qwen3 model cards suggest for chatting, plus a
#: repetition penalty for small models that loop. Not past 1.15: it counts
#: every word in view, the user's and earlier turns too, so a long chat
#: penalizes more and more ordinary words the higher it goes. The seed is None here and
#: drawn fresh for each chat (shown on the screen, so one can be repeated).
#: Poe, as in poe-tato.
DEFAULTS: Dict[str, Any] = {"system": "", "bot_name": "Poe", "temperature": 0.7,
                            "top_p": 0.9, "top_k": 40, "seed": None,
                            "repetition_penalty": 1.15, "thinking": "off",
                            "max_new_tokens": 1024, "reasoning_tokens": 4096}

#: setting -> (label, help, kind, low, high) for the number rows. The help is
#: shown on the row and again above the box you type into, so each one says
#: what it does, what's typical, and the range.
_NUMBERS = {
    _TEMPERATURE: ("how much answers vary (temperature)",
                   "0 = the same answer every time. 0.7 is typical for chatting; above 1.2 "
                   "gets erratic. Range 0 to 2.", float, 0.0, 2.0),
    _TOP_P: ("only the likeliest words (top-p)",
             "Picks only among the likeliest words that together hold this share of the "
             "odds. 1.0 = off; 0.8 to 0.95 is typical. Range 0.01 to 1.", float, 0.01, 1.0),
    _TOP_K: ("only the top k words (top-k)",
             "Picks only among this many of the likeliest next words. 0 = off; 20 to 50 "
             "is typical. Range 0 to 1000.", int, 0, 1000),
    _SEED: ("seed",
            "A new one is drawn for each chat. Set it to repeat one: the same seed and "
            "settings give the same answers. Any whole number.", int, 0, 2**31 - 1),
    _REPEAT: ("discourage repeating (repetition penalty)",
              "Makes words already in the chat less likely, so a model doesn't loop. 1.0 = "
              "off; 1.05 to 1.2 is typical; past 1.3 answers garble. Range 1 to 2.",
              float, 1.0, 2.0),
    _LONGEST: ("longest answer (tokens)",
               "The most one answer can be; a token is about 3/4 of a word, so 1024 is "
               "about 750 words. Range 16 to 32768.", int, 16, 32768),
    _ROOM: ("room to think (tokens)",
            "Extra tokens for thinking before each answer. 4096 suits most questions; hard "
            "problems may want 8192 or more. Range 0 to 65536.", int, 0, 65536),
}
_THINKING_HELP = ("off = answer straight away; drop = think first, show only the answer; "
                  "keep = also show the thinking, and save it in the export.")
_NONE, _TYPE, _LIBRARY, _FILE = ":none", ":type", ":library", ":file"


def _ask_system_prompt(prompter: Prompter, current: str) -> Optional[str]:
    """The system prompt, or None when backed out of."""
    from ..text.chat import system_prompt_from_file

    rows = [
        Choice(_NONE, "No system prompt", "The model answers as itself."),
        Choice(_TYPE, "Type or paste one…", "One line; for a long one, use a file."),
        Choice(_LIBRARY, "From my prompts library…",
               "A prompt's system turn, or its question when it has none."),
        Choice(_FILE, "From a text file…", "The whole file is the system prompt."),
    ]
    while True:
        try:
            picked = str(prompter.select("System prompt:", rows))
        except GoBack:
            return None
        try:
            if picked == _NONE:
                return ""
            if picked == _TYPE:
                text = str(prompter.text("System prompt:", default=current)).strip()
                if text:
                    return text
                continue
            if picked == _LIBRARY:
                chosen = _pick_library_prompt(prompter)
                if chosen is None:
                    continue
                text, where = system_prompt_from_file(chosen)
            else:
                raw = str(prompter.path("Text file with the system prompt:")).strip()
                if not raw:
                    continue
                text, where = system_prompt_from_file(Path(raw).expanduser())
        except GoBack:
            continue
        except OSError as e:
            prompter.note(f"  That file could not be read: {e}", style="yellow")
            continue
        if not text:
            prompter.note("  That holds no text.", style="yellow")
            continue
        prompter.note(f"  {glyphs.TICK} Using {where}.", style="green")
        return text


def _pick_library_prompt(prompter: Prompter) -> Optional[Path]:
    """One prompt from the library. The step's picker ticks several and
    offers "use all of them", which means nothing for a system prompt, so
    a chat just gets a plain list to pick one from."""
    from ..helpers import library as lib

    kind = lib.kind_by_id("prompts")
    found = lib.entries(kind)
    if not found:
        prompter.note("  Your prompts library is empty.", style="yellow")
        return None
    rows = []
    for path in found:
        label, note = lib.display_name(path), ""
        if kind.describe_entry is not None:
            try:
                label, note = kind.describe_entry(path)
            except Exception:  # noqa: BLE001 - a row that can't describe itself is still a row
                pass
        rows.append(Choice(str(path), label, note))
    try:
        return Path(str(prompter.select("Which prompt?", rows)))
    except GoBack:
        return None


def _settings(prompter: Prompter, model: str, state: Dict[str, Any]) -> bool:
    """The settings screen before the chat starts. False when backed out of."""
    while True:
        short = state["system"].replace("\n", " ")
        shown = (short[:48] + "…") if len(short) > 48 else (short or "none")
        rows = [Choice(_START, f"{glyphs.TICK} Start chatting with "
                               f"{model.rstrip('/').split('/')[-1].split('#')[-1]}",
                       "Type /exit or /quit, or press Esc, to finish.", tone="good"),
                Choice(_SYSTEM, f"system prompt — {shown}",
                       "What the model is told before the conversation starts."),
                Choice(_BOT, f"bot's name — {state[_BOT]}",
                       "What the model's replies are labeled with on screen. The export "
                       "still says assistant, with the model's own name beside it."),
                _row(_TEMPERATURE, state)]
        if float(state["temperature"]) > 0:
            # these only shape sampling, so they're shown when there is some
            rows += [_row(k, state, under=True) for k in (_TOP_P, _TOP_K, _SEED)]
        rows += [_row(_REPEAT, state),
                 Choice(_THINKING, f"think before answering — {state['thinking']}",
                        _THINKING_HELP),
                 _row(_LONGEST, state)]
        if state["thinking"] != "off":
            rows.append(_row(_ROOM, state, under=True))
        try:
            picked = str(prompter.select("Chat settings:", rows))
        except GoBack:
            return False
        try:
            if picked == _START:
                return True
            if picked == _SYSTEM:
                text = _ask_system_prompt(prompter, state["system"])
                if text is not None:
                    state["system"] = text
            elif picked == _BOT:
                # empty puts the default back
                name = str(prompter.text("bot's name:", default=state[_BOT])).strip()
                state[_BOT] = name or DEFAULTS[_BOT]
            elif picked == _THINKING:
                prompter.reason(_THINKING_HELP)
                state["thinking"] = str(prompter.select("think before answering:", [
                    Choice("off", "off", "Answer straight away."),
                    Choice("drop", "drop", "Think first; show only the answer."),
                    Choice("keep", "keep", "Think first; show the thinking, dimmed.")],
                    default=state["thinking"]))
            elif picked in _NUMBERS:
                label, help_text, kind, low, high = _NUMBERS[picked]
                prompter.reason(help_text)
                state[picked] = _number(prompter, f"{label}:", state[picked], kind, low, high)
        except GoBack:
            continue


def _row(key: str, state: Dict[str, Any], *, under: bool = False) -> Choice:
    label, help_text = _NUMBERS[key][:2]
    return Choice(key, f"{'  └ ' if under else ''}{label} — {state[key]}", help_text)


def _number(prompter: Prompter, question: str, current, kind, low, high):
    while True:
        raw = str(prompter.text(question, default=str(current))).strip()
        try:
            value = kind(raw)
        except ValueError:
            prompter.note(f"  {raw!r} is not a number.", style="yellow")
            continue
        if not low <= value <= high:
            prompter.note(f"  Between {low} and {high}, please.", style="yellow")
            continue
        return value


def _converse(prompter: Prompter, session, label: str, model_label: str = "") -> None:
    """The chat itself, until the person leaves."""
    from ..text.chat import EXIT_WORDS, ThinkFilter

    prompter.clear()
    who = f"{label} ({model_label})" if model_label and model_label != label else label
    prompter.chat_write(f"Chatting with {who}. Type /exit or /quit, or press Esc, to finish.",
                        style="dim", end="\n\n")
    out_of_view = 0
    while True:
        line = prompter.chat_input("You")
        if line is None or line.strip().lower() in EXIT_WORDS:
            return
        if not line.strip():
            continue
        # a blank line between what you typed and the reply
        prompter.chat_write("", end="\n")
        prompter.chat_write(f"{label} › ", style="bold cyan")
        filt = ThinkFilter(expect_thinking=session.thinking != "off")
        said_thinking = False
        try:
            for piece in session.send(line):
                for kind, text in filt.feed(piece):
                    said_thinking = _show(prompter, kind, text, session.thinking, said_thinking)
            for kind, text in filt.close():
                said_thinking = _show(prompter, kind, text, session.thinking, said_thinking)
            prompter.chat_write("", end="\n\n")
        except ValueError as e:
            prompter.chat_write(f"({e}.)", style="yellow", end="\n\n")
        except KeyboardInterrupt:
            prompter.chat_write(" (stopped)", style="dim", end="\n\n")
        except Exception as e:  # noqa: BLE001 - one failed reply is not a lost chat
            prompter.chat_write(f"(The model could not answer: {type(e).__name__}: {e})",
                                style="red", end="\n\n")
        if session.out_of_view > out_of_view:
            out_of_view = session.out_of_view
            prompter.chat_write(
                f"(The oldest {out_of_view} message(s) no longer fit in what the model reads "
                f"at once, so it no longer sees them. They stay in the transcript and the "
                f"export.)", style="dim", end="\n\n")


def _show(prompter: Prompter, kind: str, text: str, thinking: str, said: bool) -> bool:
    """One part of a reply on screen: the answer as it is, thinking dimmed
    when it is kept, and a single "thinking…" when it is not."""
    if kind == "answer":
        if said and thinking == "keep":
            # the thinking we just showed gets its own lines, the answer starts fresh
            prompter.chat_write("", end="\n")
        prompter.chat_write(text)
        return False
    if thinking == "keep":
        prompter.chat_write(text, style="dim italic")
        return True
    if not said:
        prompter.chat_write("(thinking…) ", style="dim")
    return True


def _offer_export(prompter: Prompter, session, cwd: Path) -> None:
    from ..text.chat import default_export_name

    if not session.messages:
        return
    try:
        if not prompter.confirm("Save this chat to a CSV file?", default=True):
            return
        raw = str(prompter.path("Where should the CSV go?",
                                default=str(cwd / default_export_name(session.started)))).strip()
    except (GoBack, Cancelled):
        return
    if not raw:
        return
    try:
        written = session.export_csv(Path(raw).expanduser())
    except OSError as e:
        prompter.note(f"  The chat could not be saved: {e}", style="red")
        return
    prompter.note(f"  {glyphs.TICK} Saved {len(session.messages)} message(s) to {written}.",
                  style="green")


def run_chat(prompter: Prompter, *, cwd: Path) -> None:
    """Pick a model and a system prompt, chat, and offer to save the chat.
    Esc on the first screens goes back; nothing is loaded until the chat
    starts."""
    from ..text._generation import generator_for
    from ..text.chat import ChatSession
    from .library import pick_generative_model
    from .recipes import _LLM_VARS

    import random

    state: Dict[str, Any] = dict(DEFAULTS)
    state[_SEED] = random.randint(0, 2**31 - 1)
    while True:
        model = pick_generative_model(prompter, str(_LLM_VARS["llm_model"]["default"]),
                                      for_step="Chat",
                                      later="It will be downloaded when the chat starts.")
        if model is None:
            raise GoBack()
        # Esc on the settings is back to the model, the screen before it
        if _settings(prompter, model, state):
            break
    think = state["thinking"] != "off"
    prompter.working(f"Loading {model} (a minute or two the first time)…")
    try:
        gen = generator_for("auto", model, thinking=think, one_copy=True,
                            max_new_tokens=int(state["max_new_tokens"])
                            + (int(state["reasoning_tokens"]) if think else 0))
    except Exception as e:  # noqa: BLE001 - every kind is a sentence
        prompter.note(f"  The model could not be loaded: {e}", style="red")
        return
    label = model.rstrip("/").split("/")[-1].split("#")[-1]
    session = ChatSession(gen, model=model, system=state["system"],
                          max_new_tokens=int(state["max_new_tokens"]),
                          temperature=float(state["temperature"]),
                          top_p=float(state["top_p"]), top_k=int(state["top_k"]),
                          seed=int(state["seed"]),
                          repetition_penalty=float(state["repetition_penalty"]),
                          thinking=state["thinking"],
                          reasoning_tokens=int(state["reasoning_tokens"]))
    try:
        _converse(prompter, session, str(state[_BOT]), label)
    finally:
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            pass
    _offer_export(prompter, session, cwd)
