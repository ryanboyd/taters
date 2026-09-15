"""
"Manage saved pipelines" -- rename, duplicate, export, import, delete.

A pipeline is an ordinary YAML file, so all of this is file management. It
exists in the wizard anyway because the audience for the wizard is exactly the
audience that should not have to find `./pipelines/` in a file browser to
rename something.

One rule runs through the whole module: **the pipelines that ship with Taters
are read-only.** They can be copied and run, never renamed or deleted. They
live inside the installed package, so "delete" would mean damaging the install
in a way that survives until the next `pip install` -- and the person most
likely to try it is the one least likely to know that.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, List

import yaml

from ...helpers.atomic import atomic_write
from ..compose import slugify
from ..prompts import Cancelled, Choice, Prompter, GoBack
from . import Task, TaskContext

__all__ = ["TASK", "looks_like_a_preset"]


def looks_like_a_preset(path: Path) -> str:
    """
    Why ``path`` is not a usable pipeline file, or "" if it is.

    Deliberately shallow: parseable YAML with a list of steps. A deeper check
    belongs to the runner, which reports what it cannot resolve far better than
    a guess here could, and refusing a slightly unusual but working file would
    be worse than accepting it.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        return f"it is not readable as YAML ({type(e).__name__})"

    if not isinstance(data, dict):
        return "it does not contain a YAML mapping"
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return "it has no 'steps:' list"
    return ""


def _title_of(path: Path) -> str:
    # the runner's own metadata reader, not a third copy of it: its defaulting
    # rules (title from the filename when the file doesn't say) are the ones
    # every list in the app shows.
    from ...pipelines.run_pipeline import _load_preset_meta

    return str(_load_preset_meta(path).get("title") or path.stem)


def _pick(prompter: Prompter, paths: List[Path], question: str) -> Path:
    choices = [Choice(str(p), _title_of(p), p.name) for p in paths]
    return Path(str(prompter.select(question, choices)))


def _unique(path: Path) -> Path:
    """A path that does not exist yet, by adding -2, -3, … before the suffix."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise Cancelled()


def load_preset_or_explain(prompter, path: Path) -> Optional[dict]:
    """
    A preset dict, or None after saying why not.

    The preset listers tolerate unreadable files so they stay visible and
    fixable -- which means every action that then *opens* one meets hand-edited
    YAML: a stray tab, a truncated write, an explicit `meta:` null. Those used
    to climb as raw YAMLError/TypeError and take the whole session down.
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        prompter.note(f"  Cannot read {Path(path).name}: {e}", style="yellow")
        return None
    if not isinstance(data, dict):
        prompter.note(f"  {Path(path).name} is not a pipeline file.",
                      style="yellow")
        return None
    steps = data.get("steps")
    if not isinstance(steps, list) or not any(steps):
        # runnable means steps. a `meta:`-only or truncated file passed every
        # check here and then run_preset raised ValueError('Preset has no
        # steps'), which nothing caught -- so the session died over a file the
        # polite path was built for.
        prompter.note(f"  {Path(path).name} has no runnable steps.",
                      style="yellow")
        return None
    return data


def _rename(ctx: TaskContext, paths: List[Path]) -> None:
    path = _pick(ctx.prompter, paths, "Rename which one?")
    data = load_preset_or_explain(ctx.prompter, path)
    if data is None:
        return
    old_title = str((data.get("meta") or {}).get("title") or path.stem)

    new_title = str(ctx.prompter.text("New name:", default=old_title)).strip()
    if not new_title or new_title == old_title:
        ctx.prompter.note("  Unchanged.", style="dim")
        return

    # the title, the id, the filename and the folder all have to agree --
    # `own_pipelines` and `--list-presets` recognize a pipeline by its folder
    # name matching the YAML stem. so we settle on one final slug first and
    # derive everything from it. deriving them separately is how a rename to a
    # same-slug title ("My pipeline" -> "My Pipeline!") once moved the folder
    # to my_pipeline-2/ while the file inside kept my_pipeline.yaml, and the
    # pipeline vanished from every menu.
    data["meta"] = data.get("meta") or {}
    data["meta"]["title"] = new_title

    new_id = slugify(new_title)
    if new_id == path.stem:
        # only the display name changed. nothing on disk needs to move, and
        # asking _unique about our own folder would invent a collision.
        data["meta"]["id"] = new_id
        with atomic_write(path, encoding="utf-8") as fh:
            fh.write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        ctx.prompter.note(f"  Renamed to '{new_title}'.", style="green")
        return

    if ctx.owns_folder(path):
        folder = _unique(path.parent.with_name(new_id))
        final = folder.name             # new_id, or new_id-2 if that was taken
        data["meta"]["id"] = final
        path.parent.rename(folder)
        target = folder / f"{final}{path.suffix}"
        (folder / path.name).rename(target)
    else:
        target = _unique(path.with_name(f"{new_id}{path.suffix}"))
        data["meta"]["id"] = target.stem
        if target != path:
            path.rename(target)

    with atomic_write(target, encoding="utf-8") as fh:
        fh.write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    ctx.prompter.note(f"  Renamed to '{new_title}' ({target.parent.name}/{target.name}).",
                      style="green")


def _duplicate(ctx: TaskContext, paths: List[Path]) -> None:
    """
    Copy, including from a built-in.

    This is the intended way to adapt a shipped pipeline: take a copy, and the
    original stays pristine.
    """
    from ...pipelines.run_pipeline import available_presets

    everything = [p for p, _ in available_presets(ctx.cwd)]
    path = _pick(ctx.prompter, everything, "Make a copy of which one?")

    data = load_preset_or_explain(ctx.prompter, path)
    if data is None:
        return
    suggested = f"{str((data.get('meta') or {}).get('title') or path.stem)} (copy)"
    new_title = str(ctx.prompter.text("Name for the copy:", default=suggested)).strip() or suggested

    new_id = slugify(new_title)
    # `or {}`, not setdefault: an explicit `meta:` null in hand-edited YAML
    # makes setdefault hand back None, and None["title"] is a session-ending
    # TypeError.
    data["meta"] = data.get("meta") or {}
    data["meta"]["title"] = new_title
    data["meta"]["id"] = new_id

    folder = _unique(ctx.cwd / new_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{folder.name}.yaml"
    data["meta"]["id"] = folder.name
    with atomic_write(target, encoding="utf-8") as fh:
        fh.write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    ctx.prompter.note(f"  Saved as {target}.", style="green")


def _export(ctx: TaskContext, paths: List[Path]) -> None:
    from ..browse import browse_for_folder

    path = _pick(ctx.prompter, paths, "Export which one?")
    folder = browse_for_folder(ctx.prompter, question="Save a copy where?")
    destination = folder / path.name

    if destination.exists() and not ctx.prompter.confirm(
        f"{destination} exists. Overwrite it?", default=False
    ):
        raise Cancelled()

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    ctx.prompter.note(f"  Exported to {destination}.", style="green")


def _import(ctx: TaskContext, paths: List[Path]) -> None:
    from ..browse import browse_for_file

    source = browse_for_file(ctx.prompter, question="Which file do you want to import?",
                             suffixes=(".yaml", ".yml"))
    problem = looks_like_a_preset(source)
    if problem:
        ctx.prompter.note(f"  That does not look like a pipeline: {problem}.", style="yellow")
        return

    folder = _unique(ctx.cwd / source.stem)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{folder.name}{source.suffix}"
    shutil.copy2(source, target)
    ctx.prompter.note(f"  Imported as {target}.", style="green")
    if folder.name != source.stem:
        ctx.prompter.note(
            f"  (renamed from {source.stem}, because that name was taken)", style="dim"
        )


def _delete(ctx: TaskContext, paths: List[Path]) -> None:
    prompter = ctx.prompter
    path = _pick(prompter, paths, "Delete which one?")
    title = _title_of(path)

    # a pipeline with a folder of its own keeps its *results* there --
    # transcripts, features, hours of extraction. "Delete a pipeline" sounds
    # like it removes a small config file, so which of the two they mean is a
    # question we ask, not an assumption we make.
    whole_folder = False
    if ctx.owns_folder(path):
        prompter.reason(
            f"'{title}' lives in its own folder, which also holds everything "
            "it has produced."
        )
        what = str(prompter.select(
            "Delete what?",
            [Choice("file", "Just the pipeline file",
                    "The folder and every result in it stay put."),
             Choice("folder", "The whole folder — pipeline, results, everything",
                    "Nothing in it survives.", tone="danger"),
             Choice("back", "↩ Never mind", tone="nav")],
        ))
        if what == "back":
            return
        whole_folder = what == "folder"

    # typed confirmation, not y/n. this is the only irreversible thing in the
    # whole wizard, and it sits one keystroke away from "Export" in the menu.
    if whole_folder:
        extras = sorted(
            child.name for child in path.parent.iterdir() if child != path
        )
        prompter.note(f"  ⚠ WARNING — this deletes the folder {path.parent} "
                      "and EVERYTHING in it.", style="bold red")
        if extras:
            prompter.note(f"  That includes: {', '.join(extras)}", style="red")
    else:
        prompter.note(f"  About to delete {path}.", style="yellow")
    prompter.note("  This cannot be undone.", style="yellow")

    typed = str(prompter.text(f"Type the name to confirm ({title}):",
                              default="")).strip()
    if typed != title:
        # loud, and a screen of its own: the old one-line dim note on the way
        # back to the menu read as "maybe it worked?" -- the opposite of what
        # someone standing next to a delete button needs to hear.
        from rich.markup import escape

        prompter.note("  ✗ That is not this pipeline's name. "
                      "Nothing was deleted.", style="bold red")
        prompter.note(f"  You typed '{escape(typed)}' — the pipeline is called "
                      f"'{escape(title)}'.", style="red")
        prompter.note(f"  '{escape(title)}' and all of its files are untouched.",
                      style="red")
        prompter.pause()
        return

    if whole_folder:
        shutil.rmtree(path.parent)
        prompter.note(f"  Deleted {path.parent.name}/.", style="green")
    else:
        path.unlink()
        prompter.note(f"  Deleted {path.name}.", style="green")


_ACTIONS = [
    ("rename", "Rename a pipeline", "Changes its name, its id and its filename together", _rename),
    ("duplicate", "Make a copy", "Including of the pipelines that ship with Taters", _duplicate),
    ("export", "Export a copy", "Save one somewhere else, to share or back up", _export),
    ("import", "Import a pipeline", "Bring in a .yaml someone sent you", _import),
    ("delete", "Delete a pipeline", "Permanent", _delete),
]

# what each action needs before we can offer it. import and duplicate work
# from an empty folder; the rest need something of the user's own to act on.
_NEEDS_OWN = {"rename", "export", "delete"}


def _run(ctx: TaskContext) -> None:
    while True:
        own = ctx.own_pipelines()
        choices = [
            Choice(
                action_id, label, help_text,
                disabled="you have no saved pipelines yet"
                         if action_id in _NEEDS_OWN and not own else "",
            )
            for action_id, label, help_text, _ in _ACTIONS
        ]
        choices.append(Choice("done", "↩ Back", "Return to the main menu",
                              tone="nav"))

        picked = str(ctx.prompter.select("What would you like to do with them?", choices))
        if picked == "done":
            return

        handler = next(fn for aid, _, _, fn in _ACTIONS if aid == picked)
        try:
            handler(ctx, own)
        except (Cancelled, GoBack):
            # backing out of one action returns to this menu, not out of the
            # task -- otherwise a mistyped path costs you the whole session.
            # GoBack included: Esc at a prompt inside an action is the same
            # "not this one after all", and letting it climb dropped the user
            # all the way out to Settings, one level further than they asked.
            ctx.prompter.note("  Cancelled.", style="dim")


def _blocked(ctx: TaskContext) -> str:
    from ...pipelines.run_pipeline import available_presets

    return "" if available_presets(ctx.cwd) else "no pipelines exist yet"


TASK = Task(
    id="manage",
    label="Manage saved pipelines",
    help="Rename, copy, import, export or delete",
    run=_run,
    unavailable_because=_blocked,
)
