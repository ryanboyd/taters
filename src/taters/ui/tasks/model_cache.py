"""
"Where downloaded models are kept": one folder, chosen once.

Every transformer, sentence-transformers and Whisper model Taters uses is
downloaded by the Hugging Face hub library into its cache, under the
user's home by default. On a shared server that is one copy per user of
every encoder on a partition that is small on purpose; on a laptop with a
small system drive it is the wrong drive. Until now the fix was an
environment variable in every shell. This screen shows where models go and
why, lets the user pick another folder, and remembers it across sessions
(``settings.json`` in the Taters home). See :mod:`taters.helpers.settings`
for the order of precedence.
"""
from __future__ import annotations

from pathlib import Path

from .. import glyphs
from ..prompts import Choice, GoBack
from . import Task, TaskContext

__all__ = ["TASK"]

_CHANGE, _NEW, _DEFAULT, _BACK = "change", "new", "default", "back"


def _run(ctx: TaskContext):
    from ...helpers.settings import (MODEL_CACHE_ENV, MODEL_CACHE_KEY,
                                     apply_model_cache, clear_setting,
                                     describe_model_cache, model_cache_source,
                                     save_setting)
    from ..browse import browse_for_folder

    prompter = ctx.prompter
    while True:
        path, source = model_cache_source()
        prompter.note(f"\n  Downloaded models are kept in:\n    {describe_model_cache()}",
                      style="cyan", wrap=False)
        if source == "environment":
            prompter.reason(
                f"{MODEL_CACHE_ENV} is set in this environment, and it wins over "
                f"any folder chosen here. Unset it to choose a folder in Taters.")
        rows = [Choice(_CHANGE, "Choose an existing folder…",
                       "Models download there from now on; ones already downloaded "
                       "elsewhere are not moved."),
                Choice(_NEW, "Type a folder to create…",
                       "A folder that does not exist yet; Taters creates it.")]
        if source == "setting":
            rows.append(Choice(_DEFAULT, "Go back to the default",
                               "Forget the folder chosen here and follow the "
                               "environment, or the usual place under your home."))
        rows.append(Choice(_BACK, f"{glyphs.BACK} Back", tone="nav"))
        try:
            picked = str(prompter.select("Model folder:", rows))
        except GoBack:
            return None
        if picked == _BACK:
            return None
        if picked == _DEFAULT:
            clear_setting(MODEL_CACHE_KEY)
            prompter.note("  Forgotten. New downloads follow the default from the next "
                          "start of Taters.", style="green")
            continue
        try:
            if picked == _NEW:
                raw = str(prompter.path("Folder to create for downloaded models:",
                                        default=str(path))).strip()
                if not raw:
                    continue
                chosen = Path(raw)
            else:
                chosen = browse_for_folder(
                    prompter, question="Which folder should hold downloaded models?",
                    start=path if path.is_dir() else Path.cwd())
        except GoBack:
            continue
        chosen = Path(chosen).expanduser().resolve()
        try:
            chosen.mkdir(parents=True, exist_ok=True)
            probe = chosen / ".taters-write-test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            prompter.note(f"  That folder cannot be written to: {e}", style="red")
            continue
        save_setting(MODEL_CACHE_KEY, str(chosen))
        applied = apply_model_cache()
        prompter.note(f"  {glyphs.TICK} Saved. Models download to {applied or chosen} from now on"
                      f"{'' if applied else ' (from the next start of Taters)'}.",
                      style="green")


TASK = Task(
    id="model_cache",
    label="Where downloaded models are kept",
    help="The folder transformer, embedding and Whisper models download to; "
         "change it once for a shared server or a small system drive",
    run=_run,
)
