"""
Pick a folder or a file by looking at them, rather than by typing a path.

Typing a path is the single most error-prone thing the wizard asks for. It is
also the one place where a mistake is invisible until it is too late: a typo in
a folder name produces "no files found", which reads as "there is nothing here"
rather than "you are looking in the wrong place". People who do not live in a
terminal do not necessarily know where they are, what the working directory is,
or that `~` means anything.

So this is a browser built out of the ordinary ``select`` prompt: entries are
the folders and files you can see, plus the moves you can make from here. It
needs nothing from the renderer that the scripted prompter cannot also do,
which is why it is testable without a terminal.

Typing is still available, because for someone who *does* know the path,
browsing to it is the slow way round.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Tuple, List, Optional, Sequence

from .prompts import Choice, Prompter

__all__ = ["browse_for_folder", "browse_for_file", "browse_and_tick", "short_path", "human_size"]

# enough rows to see what's there, few enough to actually move through. a media
# folder can hold thousands of files, and a select list that long is worse than
# no list at all.
_MAX_ENTRIES = 200

def short_path(path: Path, keep: int = 2) -> str:
    """
    A path short enough to sit in a question, with the end kept.

    The end is the part that identifies it -- ``…/scratchpad/hub`` says where
    you are; the first sixty characters of a temp directory do not. Home is
    abbreviated to ``~`` for the same reason.
    """
    path = Path(path)
    try:
        path = Path("~") / path.relative_to(Path.home())
    except ValueError:
        pass

    parts = path.parts
    if len(parts) <= keep + 1:
        return str(path)
    return "…/" + "/".join(parts[-keep:])


_CHOOSE = "\x00choose"
_UP = "\x00up"
_TYPE = "\x00type"


def _entries(folder: Path, *, want_files: Optional[Sequence[str]]) -> tuple:
    """
    Subfolders and (optionally) matching files, sorted, hidden ones last.

    Returns ``(folders, files, truncated)``. Unreadable folders come back empty
    rather than raising: a permissions error should narrow the choices, not end
    the wizard.
    """
    folders: List[Path] = []
    files: List[Path] = []
    hidden_folders: List[Path] = []
    hidden_files: List[Path] = []
    try:
        for child in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            hidden = child.name.startswith(".")
            if child.is_dir():
                (hidden_folders if hidden else folders).append(child)
            elif want_files is not None and child.suffix.lower() in want_files:
                (hidden_files if hidden else files).append(child)
    except (PermissionError, OSError):
        return [], [], False
    # dot-folders go at the end rather than nowhere. we used to drop them
    # outright, which meant nobody could browse to ~/.cache/huggingface/hub to
    # import a model they'd already downloaded -- the folder just wasn't there
    folders += hidden_folders
    files += hidden_files

    truncated = len(folders) + len(files) > _MAX_ENTRIES
    if truncated:
        # folders and files split the budget. we used to give folders absolute
        # priority, so a directory with 250 subfolders and 300 matching files
        # showed 200 folders and *one* file -- which reads as "my files are
        # missing" rather than "this list is long".
        keep_folders = min(len(folders),
                           _MAX_ENTRIES - min(len(files), _MAX_ENTRIES // 2))
        folders = folders[:keep_folders]
        files = files[:_MAX_ENTRIES - keep_folders]
    return folders, files, truncated


def human_size(n: int) -> str:
    """A file size short enough to sit in a column."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return ""


def _count_hint(folder: Path, want_files: Optional[Sequence[str]],
                cap: int = 500) -> str:
    """
    How many interesting files a folder holds, shown next to its name.

    Capped: this runs for every folder in the listing on every render, and an
    uncapped sweep of a directory with fifty thousand entries -- on a network
    share, or WSL's /mnt/c -- stalled the whole browser to draw one hint.
    Past the cap the hint says so and stops counting.
    """
    if want_files is None:
        return ""
    try:
        entries = folder.iterdir()
        head = list(itertools.islice(entries, cap))
        capped = next(entries, None) is not None
        n = sum(1 for e in head
                if e.is_file() and e.suffix.lower() in want_files)
    except (PermissionError, OSError):
        return ""
    if not n:
        return ""
    if capped:
        return f"{n}+ files"
    return f"{n} file" if n == 1 else f"{n} files"


def _listing(box: dict, *, want_files, count_suffixes,
             ticked: Optional[set] = None) -> Tuple[List[Choice], set]:
    """
    The rows of one folder, for either browser.

    Up first, then (when choosing a folder) the affirmative act, then folders
    with a count of interesting files, then files with their sizes -- ticked
    or not -- then "type a path". Everything goes in the label, nothing in
    `help`: names and sizes need to line up in columns, and the annotation
    rides on the Choice, where the renderer gives it a color of its own --
    that is what makes "4 files" spottable while scanning.

    The two browsers each carried a copy of this that differed in three
    details (a choose row, tick marks, the note when a folder is empty); the
    details are parameters now. Returns the rows and the set of tickable
    values (files), which is empty when ``ticked`` is None.
    """
    here = box["here"]
    folders, files, truncated = _entries(here, want_files=want_files)
    marked = ticked is not None
    # the tick mark is part of the label, so it's part of the width. when we
    # measured without it, every file row's annotation drifted 4 columns to the
    # right of the folders'.
    names = ([f"{f.name}/" for f in folders]
             + [(f"[x] {f.name}" if marked else f.name) for f in files])
    column = min(max((len(n) for n in names), default=0), 38 if marked else 34)

    def row(name: str) -> str:
        # pad so every annotation starts in the same column.
        return name.ljust(column) + "  "

    choices: List[Choice] = []
    if here.parent != here:
        # parent first: browsing is mostly walking, and "up" is the one move you
        # make from anywhere. (we dropped the "Home folder" row: the launch
        # location covers it, and every action row pushes the folders down.)
        choices.append(Choice(_UP, f"↑ Up to {short_path(here.parent, keep=1)}",
                              tone="nav"))
    if want_files is None:
        # the "yes, do this" row, green so it reads as an action rather than one
        # more folder.
        here_hint = _count_hint(here, count_suffixes)
        choices.append(Choice(_CHOOSE,
                              row("✓ Use this folder") if here_hint
                              else "✓ Use this folder",
                              annotation=here_hint, tone="good"))
    for folder in folders:
        hint = _count_hint(folder, count_suffixes)
        # a dot-folder says so in the annotation column, so that ~/.cache
        # sitting at the bottom of the list doesn't read like a stray file
        note = hint or ("hidden" if folder.name.startswith(".") else "")
        choices.append(Choice(str(folder),
                              row(f"{folder.name}/") if note else f"{folder.name}/",
                              annotation=note))
    for file in files:
        try:
            size = human_size(file.stat().st_size)
        except OSError:
            size = ""
        if marked:
            mark = "x" if str(file) in ticked else " "
            label = row(f"[{mark}] {file.name}")
        else:
            label = row(file.name) if size else file.name
        choices.append(Choice(str(file), label, annotation=size))
    choices.append(Choice(_TYPE, "⌨ Type a path instead"))

    # this goes on the breadcrumb rather than printed as a note, because notes
    # under a transient prompt forced a repaint, and the message belongs to the
    # listing anyway.
    if truncated:
        box["note"] = f"showing the first {_MAX_ENTRIES} entries"
    elif want_files is not None and not files and not folders:
        box["note"] = "nothing matching in this folder"
    else:
        box["note"] = ""
    return choices, ({str(f) for f in files} if marked else set())


def _crumb(box: dict) -> str:
    path = short_path(box["here"])
    return f"{path}   · {box['note']}" if box["note"] else path


def _browse(
    prompter: Prompter,
    *,
    question: str,
    start: Path,
    want_files: Optional[Sequence[str]],
    count_suffixes: Optional[Sequence[str]],
    type_prompt: str,
) -> Path:
    """
    Shared loop.

    ``want_files=None`` means "choosing a folder"; ``count_suffixes`` annotates
    each folder with how many interesting files it holds, which is what turns
    "which of these eleven folders was it?" into an answerable question.
    """
    box = {"here": Path(start).expanduser().resolve(), "note": ""}
    if not box["here"].is_dir():
        box["here"] = Path.cwd().resolve()

    def listing() -> List[Choice]:
        choices, _toggles = _listing(box, want_files=want_files,
                                     count_suffixes=count_suffixes)
        return choices

    def crumb() -> str:
        return _crumb(box)

    def navigate(value: str) -> Optional[dict]:
        # enter on a folder (or Up) means "swap this screen for that folder's",
        # and we answer that inside the running prompt -- the exit-and-rebuild
        # this replaces made every step of a walk flash. anything else is a
        # real answer and comes back to the loop below.
        if value == _UP:
            target = box["here"].parent
        else:
            target = Path(value)
            if value in (_CHOOSE, _TYPE) or not target.is_dir():
                return None
        box["here"] = target
        return {"choices": listing(),
                # start the pointer on the "use this folder" row: that's how
                # most visits end, and starting on "Up" made the common case
                # two keys instead of one.
                "default": _CHOOSE if want_files is None else None}

    # paint the screen furniture once; navigation happens inside the prompt.
    prompter.repaint()

    while True:
        picked = str(prompter.select(
            question, listing(), transient=True,
            default=_CHOOSE if want_files is None else None,
            navigate=navigate, breadcrumb=crumb,
        ))

        if picked == _CHOOSE:
            return box["here"]
        if picked == _UP:
            # only a renderer that can't swap in place ever hands back a
            # navigation value; this loop is its fallback, one screen per step.
            box["here"] = box["here"].parent
            continue
        if picked == _TYPE:
            raw = str(prompter.path(type_prompt, default="")).strip()
            if not raw:
                continue
            typed = Path(raw).expanduser()
            if want_files is None and typed.is_dir():
                return typed.resolve()
            if want_files is not None and typed.is_file():
                return typed.resolve()
            prompter.note(f"  '{typed}' is not something I can see. Still browsing.",
                          style="yellow")
            prompter.repaint()
            continue

        chosen = Path(picked)
        if chosen.is_dir():
            box["here"] = chosen    # fallback renderers only, as with _UP
            continue
        return chosen


def browse_for_folder(prompter: Prompter, *, question: str = "Which folder?",
                      start: Optional[Path] = None,
                      want_files: Optional[Sequence[str]] = None) -> Path:
    """
    Walk the filesystem and return a folder.

    Parameters
    ----------
    want_files : sequence of str, optional
        Suffixes worth counting, e.g. ``(".txt",)``. Shown beside each folder,
        so the right one can be recognized rather than remembered.
    """
    return _browse(
        prompter,
        question=question,
        start=start or Path.cwd(),
        want_files=None,
        count_suffixes=tuple(s.lower() for s in want_files) if want_files else None,
        type_prompt="Path to the folder:",
    )


def browse_and_tick(prompter: Prompter, *, question: str = "Which files?",
                    start: Optional[Path] = None,
                    suffixes: Sequence[str] = (".csv",)) -> List[Path]:
    """
    A file browser that is also the selector: walk folders, tick files where
    they are, confirm once.

    Space ticks a file in place -- no redraw, no flash -- and the selection
    survives walking between folders, so a set spread over several folders is
    still one trip. Enter means proceed: on a folder it opens it, on a file it
    finishes -- with the ticked set if anything is ticked, or with just that
    file (the quick single-file path). There is no "Import N files" menu row to
    hunt for; two earlier shapes hid either the files or the way out.

    Returns the chosen files. Esc raises GoBack, like every other screen.
    """
    suffixes = tuple(x.lower() for x in suffixes)
    box = {"here": Path(start or Path.cwd()).expanduser().resolve(),
           "note": "", "toggles": set()}
    if not box["here"].is_dir():
        box["here"] = Path.cwd().resolve()
    ticked: set = set()          # str paths, the same strings the rows carry

    def listing() -> List[Choice]:
        choices, toggles = _listing(box, want_files=suffixes,
                                    count_suffixes=suffixes, ticked=ticked)
        box["toggles"] = toggles
        return choices

    def crumb() -> str:
        return _crumb(box)

    def navigate(value: str) -> Optional[dict]:
        # enter on a folder (or Up) swaps the rows inside the running prompt;
        # the ticks live in `ticked` and ride out the walk untouched. a file or
        # an action row is a real answer and exits.
        if value == _UP:
            target = box["here"].parent
        else:
            target = Path(value)
            if value == _TYPE or not target.is_dir():
                return None
        box["here"] = target
        return {"choices": listing(), "toggle_values": box["toggles"]}

    # paint the screen furniture once; navigation happens inside the prompt.
    prompter.repaint()
    while True:
        choices = listing()
        picked = str(prompter.select(
            question, choices, transient=True,
            # space ticks in place -- the prompt keeps running, so there's no
            # redraw and the pointer doesn't move at all.
            toggle_values=box["toggles"],
            ticked=ticked,
            navigate=navigate, breadcrumb=crumb,
        ))

        if picked == _UP:
            # only a renderer that can't swap in place ever hands back a
            # navigation value; this loop is its fallback, one screen per step.
            box["here"] = box["here"].parent
            continue
        if picked == _TYPE:
            raw = str(prompter.path("Type a path (a folder to open, or a file "
                                    "to add):", default="")).strip()
            if not raw:
                continue
            typed = Path(raw).expanduser()
            if typed.is_dir():
                box["here"] = typed.resolve()
            elif typed.is_file() and typed.suffix.lower() in suffixes:
                # typing a file's path IS choosing it: we proceed with it (plus
                # whatever's already ticked) rather than making the user find
                # the row they just named and hit enter again.
                return sorted({Path(v) for v in ticked} | {typed.resolve()})
            else:
                prompter.note(f"  '{typed}' is not something I can use. "
                              "Still browsing.", style="yellow")
                prompter.repaint()
            continue

        chosen = Path(picked)
        if chosen.is_dir():
            box["here"] = chosen    # fallback renderers only, as with _UP
            continue
        # enter on a file proceeds with everything ticked plus the pointed file
        # itself -- we never silently drop ticks, and with nothing ticked this
        # is the quick single-file path.
        return sorted({Path(v) for v in ticked} | {chosen})


def browse_for_file(prompter: Prompter, *, question: str = "Which file?",
                    start: Optional[Path] = None,
                    suffixes: Sequence[str] = (".csv",)) -> Path:
    """Walk the filesystem and return a file with one of ``suffixes``."""
    suffixes = tuple(s.lower() for s in suffixes)
    return _browse(
        prompter,
        question=question,
        start=start or Path.cwd(),
        want_files=suffixes,
        count_suffixes=suffixes,
        type_prompt="Path to the file:",
    )
