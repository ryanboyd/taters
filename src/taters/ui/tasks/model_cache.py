"""
"Downloaded models": the Hugging Face cache, listed, and kept tidy.

Every transformer, sentence-transformers, Whisper and language model Taters
uses is downloaded by the Hugging Face hub library into its cache, under the
user's home by default -- and that cache is shared with every other program
on the machine that uses the hub. This screen lists what is in it, says what
each entry is to Taters (a model a step can use, a part a step downloads for
itself, or something Taters never touches), and exports or deletes the
ticked ones. Deleting a model a saved pipeline names asks first, by name.

It is also where the download folder is chosen (``settings.json`` in the
Taters home; see :mod:`taters.helpers.settings` for the order of
precedence), which used to be this screen's only job.

Storage lives in :mod:`taters.helpers.hub_models`; this is the conversation,
in the same dialect as the library screens: tick rows, then an action.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

from .. import glyphs
from ..browse import browse_for_folder, human_size
from ..prompts import Choice, GoBack, Prompter
from . import Task, TaskContext

__all__ = ["TASK", "manage_model_cache"]

_CHANGE, _NEW, _DEFAULT, _BACK = "change", "new", "default", "back"
# the list's own rows carry a colon, so no model name (`roberta-base` has no
# owner) can be mistaken for one
_EXPORT, _DELETE, _LEFTOVERS, _FOLDER = ":export", ":delete", ":leftovers", ":folder"
_LIST_BACK = ":back"


def _used(repo) -> str:
    if repo.role == "step":
        steps = [s for s in repo.uses.split("; ") if s]
        named = ", ".join(steps[:2]) + (f", and {len(steps) - 2} more" if len(steps) > 2 else "")
        return f"Used by {named}" if named else "A model a step can use"
    if repo.role == "part":
        return f"Part of a step: {repo.uses}"
    return "Not used by Taters" + (f" ({repo.uses})" if repo.uses.startswith("a ") else "")


def _tag(repo) -> str:
    """The size, and a word for anything that is not simply a model a step
    uses, so those can be seen without moving onto them."""
    extra = {"part": "part of a step", "other": "not used by Taters"}.get(repo.role, "")
    if not repo.complete:
        extra = "incomplete"
    return human_size(repo.size) + (f" · {extra}" if extra else "")


def _row_help(repo) -> str:
    when = time.strftime("%b %d, %Y", time.localtime(repo.last_used)) if repo.last_used else ""
    parts = [_used(repo)]
    if not repo.complete:
        parts.append("not fully downloaded; a run fetches the rest")
    if when:
        parts.append(f"last used {when}")
    return " · ".join(parts)


def manage_model_cache(prompter: Prompter, cwd: Optional[Path] = None) -> None:
    """
    List the cache, tick entries, export or delete them. The same dialect
    as the library screens: entries are tick rows, and the actions apply
    to whatever is ticked. Esc leaves; every action returns to the list.
    """
    from ...helpers import hub_models as hm
    from ...helpers.settings import describe_model_cache

    ticked: set = set()
    pointer: Optional[str] = None
    while True:
        try:
            with prompter.scanning("Reading the model cache") as seen:
                seen("listing what is downloaded")
                repos = hm.cached_repos()
        except Exception as e:  # noqa: BLE001 - an unreadable cache is a sentence
            prompter.note(f"  The model cache could not be read: {e}", style="red")
            repos = []
        by_id = {r.hub_id: r for r in repos}
        ticked &= set(by_id)
        prompter.note(f"\n  Downloaded models are kept in:\n    {describe_model_cache()}",
                      style="cyan", wrap=False)
        others = [r for r in repos if r.role == "other"]
        if others:
            prompter.reason(
                ("One of these is" if len(others) == 1 else f"{len(others)} of these are")
                + f" not used by Taters ({human_size(sum(r.size for r in others))}): the "
                f"cache is shared with other programs on this computer, which may need "
                f"{'it' if len(others) == 1 else 'them'}.")
        leftover = sum(r.old_size for r in repos)
        choices = [Choice(_LIST_BACK, f"{glyphs.BACK} Back", tone="nav")]
        if repos:
            choices.append(Choice(_EXPORT, f"{glyphs.EXPORT} Export the ticked models…",
                                  "Copy them to a folder as plain model folders, to load "
                                  "by path or carry to another computer."))
            choices.append(Choice(_DELETE, f"{glyphs.DELETE} Delete the ticked models",
                                  "Frees the space; anything a step uses downloads again "
                                  "the next time it is needed.", tone="danger"))
        if leftover:
            choices.append(Choice(_LEFTOVERS, f"{glyphs.DELETE} Free {human_size(leftover)} "
                                              f"of leftover versions",
                                  "Old versions left behind when a model was updated; "
                                  "the versions in use stay."))
        choices.append(Choice(_FOLDER, f"{glyphs.GEAR} Change the download folder…",
                              "Where new downloads go; ones already downloaded stay put."))
        for r in repos:
            mark = "x" if r.hub_id in ticked else " "
            choices.append(Choice(r.hub_id, f"[{mark}] {r.hub_id}", _row_help(r),
                                  annotation=_tag(r)))
        if not repos:
            prompter.reason("Nothing is downloaded yet. Models arrive the first time a "
                            "step needs one.")
        try:
            picked = str(prompter.select(
                "Downloaded models:", choices,
                default=pointer if any(c.value == pointer for c in choices) else None,
                toggle_values=list(by_id), ticked=ticked))
        except GoBack:
            return
        pointer = picked
        if picked == _LIST_BACK:
            return
        if picked == _FOLDER:
            change_folder(prompter)
        elif picked == _LEFTOVERS:
            _free_leftovers(prompter, repos)
        elif picked in (_EXPORT, _DELETE) and not ticked:
            prompter.note("  Nothing is ticked yet — press \\[space] on a model to tick it.",
                          style="yellow")
        elif picked == _EXPORT:
            _export(prompter, [by_id[h] for h in sorted(ticked)])
        elif picked == _DELETE:
            if _delete(prompter, [by_id[h] for h in sorted(ticked)], cwd):
                ticked.clear()
        else:
            ticked.symmetric_difference_update({picked})


def _delete(prompter: Prompter, repos: List, cwd: Optional[Path] = None) -> bool:
    """Delete the ticked repositories after saying what each one is to
    Taters, which saved pipelines name it, and how much is freed. True
    when they were deleted."""
    from ...helpers import hub_models as hm

    total = sum(r.size for r in repos)
    prompter.note("  To delete:", style="bold")
    for r in repos:
        prompter.note(f"    {r.hub_id}  ({human_size(r.size)})", wrap=False)
    named = {}
    with prompter.scanning("Checking your saved pipelines") as seen:
        for r in repos:
            seen(r.hub_id)
            found = hm.pipelines_naming(r, cwd)
            if found:
                named[r.hub_id] = found
    for r in repos:
        if r.hub_id in named:
            prompter.note(f"  {glyphs.WARNING} {r.hub_id} is named by "
                          f"{len(named[r.hub_id])} saved pipeline(s): "
                          f"{', '.join(named[r.hub_id])}. They will download it again "
                          f"when they run.", style="yellow")
        elif r.role == "part":
            prompter.note(f"  {r.hub_id} is part of a step ({r.uses}); that step downloads "
                          f"it again the next time it runs.", style="yellow")
        elif r.role == "other":
            prompter.note(f"  {glyphs.WARNING} {r.hub_id} is not used by Taters. Another "
                          f"program on this computer may need it.", style="yellow")
    question = (f"Delete {len(repos)} model(s), freeing {human_size(total)}"
                + (", even though saved pipelines use some of them?" if named else "?"))
    try:
        if not prompter.confirm(question, default=False):
            return False
    except GoBack:
        return False
    try:
        freed = hm.delete_repos(repos)
    except Exception as e:  # noqa: BLE001 - a locked file on Windows, say
        prompter.note(f"  Could not delete everything: {e}", style="red")
        return False
    prompter.note(f"  {glyphs.TICK} Deleted {len(repos)} model(s); freed {human_size(freed)}.",
                  style="green")
    return True


def _free_leftovers(prompter: Prompter, repos: List) -> None:
    from ...helpers import hub_models as hm

    leftover = hm.old_versions(repos)
    size = sum(r.old_size for r in repos)
    for r in {r.hub_id: r for r, _h in leftover}.values():
        prompter.note(f"    {r.hub_id}: {len(r.leftovers)} old version(s), "
                      f"{human_size(r.old_size)}", wrap=False)
    try:
        if not prompter.confirm(f"Delete these old versions, freeing {human_size(size)}? "
                                f"The versions in use stay.", default=True):
            return
    except GoBack:
        return
    try:
        freed = hm.delete_old_versions(repos)
    except Exception as e:  # noqa: BLE001
        prompter.note(f"  Could not delete everything: {e}", style="red")
        return
    prompter.note(f"  {glyphs.TICK} Freed {human_size(freed)}.", style="green")


def _snapshot_size(repo) -> int:
    snap = repo.snapshot
    if snap is None or not snap.is_dir():
        return 0
    total = 0
    for f in snap.rglob("*"):
        try:
            if f.is_file():
                total += f.resolve().stat().st_size
        except OSError:
            continue
    return total


def _export(prompter: Prompter, repos: List) -> None:
    """Copy the ticked models out as plain folders, one per model."""
    import shutil

    from ...helpers import hub_models as hm

    try:
        dest = browse_for_folder(prompter, question="Which folder should the copies go in?",
                                 start=Path.home())
    except GoBack:
        return
    dest = Path(dest)
    sizes = {r.hub_id: _snapshot_size(r) for r in repos}
    need = sum(sizes.values())
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        free = None
    if free is not None and need > free:
        prompter.note(f"  The copies need {human_size(need)} and {dest} has "
                      f"{human_size(free)} free. Nothing was copied.", style="red")
        return
    made = []
    for r in repos:
        target = hm.export_folder_name(r, dest)
        size = sizes[r.hub_id]
        if not r.complete:
            prompter.note(f"  {r.hub_id} is not fully downloaded here; the copy has what "
                          f"there is, and loading it may still download the rest.",
                          style="yellow")
        try:
            with prompter.scanning(f"Copying {r.hub_id}") as seen:
                hm.export_repo(r, target, on_bytes=lambda done, size=size: seen(
                    f"{human_size(done)} of {human_size(size)}"))
        except KeyboardInterrupt:
            prompter.note("  Stopped; the unfinished copy was removed.", style="yellow")
            return
        except Exception as e:  # noqa: BLE001 - one failed copy does not stop the rest
            prompter.note(f"  {r.hub_id} could not be copied: {e}", style="red")
            continue
        made.append((r, target))
        prompter.note(f"  {glyphs.TICK} {r.hub_id} → {target}", style="green", wrap=False)
    if made:
        prompter.reason("Each folder is a complete model. Type its path wherever a step asks "
                        "for a model, or copy it to another computer.")


def change_folder(prompter: Prompter) -> None:
    """Where new downloads go: pick a folder, type one to create, or go back
    to the default. What is already downloaded stays where it is."""
    from ...helpers.settings import (MODEL_CACHE_ENV, MODEL_CACHE_KEY,
                                     apply_model_cache, clear_setting,
                                     describe_model_cache, model_cache_source,
                                     save_setting)

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
        return
    if picked == _BACK:
        return
    if picked == _DEFAULT:
        clear_setting(MODEL_CACHE_KEY)
        prompter.note("  Forgotten. New downloads follow the default from the next "
                      "start of Taters.", style="green")
        return
    try:
        if picked == _NEW:
            raw = str(prompter.path("Folder to create for downloaded models:",
                                    default=str(path))).strip()
            if not raw:
                return
            chosen = Path(raw)
        else:
            chosen = browse_for_folder(
                prompter, question="Which folder should hold downloaded models?",
                start=path if path.is_dir() else Path.cwd())
    except GoBack:
        return
    chosen = Path(chosen).expanduser().resolve()
    try:
        chosen.mkdir(parents=True, exist_ok=True)
        probe = chosen / ".taters-write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        prompter.note(f"  That folder cannot be written to: {e}", style="red")
        return
    save_setting(MODEL_CACHE_KEY, str(chosen))
    applied = apply_model_cache()
    prompter.note(f"  {glyphs.TICK} Saved. Models download to {applied or chosen} from now on"
                  f"{'' if applied else ' (from the next start of Taters)'}.",
                  style="green")


def _run(ctx: TaskContext):
    manage_model_cache(ctx.prompter, ctx.cwd)


TASK = Task(
    id="model_cache",
    label="Manage downloaded models",
    help="See the models downloaded to this computer, export or delete them, "
         "and choose where new ones go",
    run=_run,
)
