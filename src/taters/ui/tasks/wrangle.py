"""
"Wrangle data" -- the gather helpers, as a front-door conversation.

Collecting text into a tidy spreadsheet is a job in its own right, not just
the invisible first step of a feature pipeline: a folder of documents into
one CSV, or a spreadsheet aggregated by any column or combination of columns
(everything one author wrote, everything posted in one subreddit, one row
per author-within-subreddit). This task asks those questions in plain words
and hands the answers to :mod:`taters.helpers.text_gather` -- the same code
every pipeline's gather step runs, so a file wrangled here is byte-for-byte
what a pipeline would have produced.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from ...helpers.doc_text import DOCUMENT_PATTERN
from .. import glyphs
from ..columns import looks_numeric, peek_csv
from ..prompts import Cancelled, Choice, GoBack, ask_at_least_one
from . import Task, TaskContext

# the tickable "this spreadsheet has no text" row. NUL-prefixed like the other
# UI sentinels so it can never collide with a real column name.
_NO_TEXT = "\x00no_text"


def _peek(path: Path, n: int = 200):
    """The spreadsheet's columns and a sample of rows. Shared with the
    wizard's analysis stage, which asks the same kind of column questions --
    see :mod:`taters.ui.columns`."""
    return peek_csv(path, n)


def _looks_numeric(rows, column: str) -> bool:
    """Whether a sampled column is all numbers -- the test for offering
    "average this column" at all."""
    return looks_numeric(rows, column)


def _ask_output(prompter, default: Path) -> tuple[Path, bool]:
    """Where the spreadsheet goes; an existing file needs an explicit yes."""
    while True:
        raw = prompter.path("Where should the finished spreadsheet go?",
                            default=str(default))
        out = Path(raw).expanduser()
        if out.suffix.lower() != ".csv":
            out = out.with_suffix(".csv")
        if not out.exists():
            return out, False
        if prompter.confirm(f"{out.name} already exists. Overwrite it?",
                            default=False):
            return out, True
        # neither overwrite nor a new name yet, so we ask again.


def _with_display(prompter, label: str, fn, **kwargs):
    """
    Run one gather with the same live bars a pipeline step gets.

    The gather reports progress and in-flight documents through the standard
    ``on_progress`` channel; a fabricated one-step run feeds that into the
    RunDisplay so folder reads show per-file sub-bars. The scripted prompter
    has no console, and then the work simply runs without decoration.

    Only gathers that *take* ``on_progress`` get one: the CSV gather has no
    such parameter, and passing it anyway crashed the whole task with a
    TypeError mid-run. That gather shows a working spinner instead.
    """
    import inspect

    try:
        takes_progress = "on_progress" in inspect.signature(fn).parameters
    except (TypeError, ValueError):        # pragma: no cover - C callables
        takes_progress = False

    console = getattr(prompter, "_console", None)
    if console is None:
        return fn(**kwargs)

    from ..run_display import RunDisplay, reporter_for

    with RunDisplay(console) as display:
        emit = reporter_for(display)
        emit("run_start", steps=1, inputs=[])
        emit("step_start", index=1, total=1, call=label, scope="global")

        def on_progress(done, total, message=None, unit=None, **extra):
            emit("step_progress", index=1, call=label, done=done,
                 total=total, message=message, unit=unit, **extra)

        if takes_progress:
            kwargs["on_progress"] = on_progress
        try:
            result = fn(**kwargs)
        except Exception as e:
            emit("step_done", index=1, call=label, status="error", error=str(e))
            raise
        emit("step_done", index=1, call=label, status="ok")
        emit("run_done", manifest={}, manifest_path="")
    return result


def _wrangle_folder(ctx: TaskContext) -> Optional[Path]:
    from ...helpers.text_gather import txt_folder_to_analysis_ready_csv
    from ..browse import browse_for_folder

    prompter = ctx.prompter
    prompter.reason("Every readable document in the folder (.txt, .docx, "
                    ".doc, .pdf) becomes one row: its name, and its text.")
    folder = browse_for_folder(
        prompter, question="Which folder holds the documents?",
        start=ctx.cwd, want_files=(".txt", ".docx", ".doc", ".pdf"))
    recursive = prompter.confirm("Include documents inside subfolders too?",
                                 default=True)
    out, overwrite = _ask_output(prompter,
                                 ctx.cwd / f"{folder.name}_wrangled.csv")
    result = _with_display(
        prompter, "wrangle: gather documents",
        txt_folder_to_analysis_ready_csv,
        root_dir=folder, out_csv=out, recursive=recursive,
        pattern=DOCUMENT_PATTERN, id_from="stem", include_source_path=True,
        overwrite_existing=overwrite)
    return Path(result)


def _wrangle_csv(ctx: TaskContext) -> Optional[Path]:
    from ...helpers.text_gather import csv_to_analysis_ready_csv
    from ..browse import browse_for_file

    prompter = ctx.prompter
    src = browse_for_file(prompter, question="Where is the spreadsheet?",
                          start=ctx.cwd, suffixes=(".csv", ".tsv", ".txt"))
    columns, sample = _peek(src)
    if not columns:
        prompter.note("  That file has no header row to read columns from.",
                      style="yellow")
        raise Cancelled()

    picks = ask_at_least_one(
        prompter, "Which column(s) hold the text?",
        [Choice(c, c) for c in columns] +
        [Choice(_NO_TEXT, "None — there's no text to wrangle",
                "Just combine or summarize the other columns.")],
        thing="one text column (or 'None')")
    # ticking "None" alongside real columns is a contradiction; the real
    # columns win, since ticking them was the more intentional move.
    text_cols = [c for c in picks if c != _NO_TEXT]

    others = [c for c in columns if c not in text_cols]
    group_by: Optional[List[str]] = None
    id_cols: Optional[List[str]] = None
    carry: List[str] = []
    averages: List[str] = []
    if others:
        prompter.reason(
            "Two shapes are possible. Keep every row as it is — one row of "
            "output per row of your spreadsheet. Or combine rows that belong "
            "together, so that (say) everything one person wrote becomes a "
            "single row of text.")
        shape = str(prompter.select(
            "How should the finished spreadsheet be shaped?",
            [Choice("as_is", "One row out for every row in",
                    "Each row keeps its own text."),
             Choice("combine", "Combine rows that belong together",
                    "One row per person, per category, per whatever you "
                    "pick — texts joined together.")]))
        if shape == "combine":
            prompter.reason(
                "Tick the column(s) that say which rows belong together. One "
                "column (say, username) gives one row per person. Ticking "
                "two (say, subreddit and username) gives one row per person "
                "within each subreddit.")
            group_by = ask_at_least_one(
                prompter, "Which column(s) say which rows belong together?",
                [Choice(c, c) for c in others], thing="one column")
            rest = [c for c in others if c not in group_by]
            numeric = [c for c in rest if _looks_numeric(sample, c)]
            if numeric and prompter.confirm(
                    "Some columns look like numbers. Average them for each "
                    "combined row (e.g., a post's score becomes a person's "
                    "average score)?", default=False):
                averages = ask_at_least_one(
                    prompter, "Average which column(s)?",
                    [Choice(c, c) for c in numeric], thing="one column")
            rest = [c for c in rest if c not in averages]
            if rest and prompter.confirm(
                    "Bring any other columns along unchanged?",
                    default=False):
                prompter.reason(
                    "A brought-along column keeps its value where every "
                    "combined row agrees on it; where the rows disagree, the "
                    "cell is left blank — there is no single honest value "
                    "to write.")
                carry = ask_at_least_one(
                    prompter, "Bring along which column(s)?",
                    [Choice(c, c) for c in rest], thing="one column")
        else:
            picked = str(prompter.select(
                "Does one of these columns name each row — an ID, like a "
                "filename or participant number?",
                [Choice("", "No — just number the rows",
                        "Rows come out as row_1, row_2, …")] +
                [Choice(c, c) for c in others]))
            id_cols = [picked] if picked else None
            rest = [c for c in others if c != picked]
            if rest and prompter.confirm(
                    "Keep any of the other columns in the output?",
                    default=False):
                carry = ask_at_least_one(
                    prompter, "Keep which column(s)?",
                    [Choice(c, c) for c in rest], thing="one column")

    mode = "concat"
    if len(text_cols) > 1:
        mode = str(prompter.select(
            "Several text columns — how should they combine?",
            [Choice("concat", "Join them into one text per row",
                    "One row, the columns' text stitched together."),
             Choice("separate", "Keep each column as its own row",
                    "One row per column, with a source_col saying which.")]))

    out, overwrite = _ask_output(prompter,
                                 ctx.cwd / f"{src.stem}_wrangled.csv")
    result = _with_display(
        prompter, "wrangle: gather spreadsheet",
        csv_to_analysis_ready_csv,
        csv_path=src, out_csv=out, text_cols=list(text_cols),
        id_cols=list(id_cols) if id_cols else None,
        group_by=list(group_by) if group_by else None,
        carry_cols=carry or None,
        agg_cols={c: "mean" for c in averages} or None,
        mode=mode, overwrite_existing=overwrite,
        # the live display owns the screen; the gather's own prints would slice
        # right through it the way the HF download bars once did.
        verbose=False)
    return Path(result)


def _run(ctx: TaskContext) -> Optional[bool]:
    prompter = ctx.prompter
    try:
        source = str(prompter.select(
            "What are we wrangling?",
            [Choice("folder", "A folder of documents",
                    "Every .txt, .docx, .doc and .pdf becomes one row of a "
                    "new spreadsheet."),
             Choice("csv", "A spreadsheet (CSV)",
                    "Pick the text column(s), optionally combine rows by any "
                    "column(s) — one row per subreddit, per author, per "
                    "author-within-subreddit…")]))
        out = (_wrangle_folder if source == "folder" else _wrangle_csv)(ctx)
    except (GoBack, Cancelled):
        raise Cancelled()
    if out is None:
        return None

    from ...helpers.progress import count_rows

    # count_rows returns 0 when there's no sink (its pass only exists to feed a
    # progress bar), but the summary actually wants the number, so we give it
    # one.
    rows = max(0, count_rows(out, on_progress=lambda *a, **k: None))
    prompter.note(f"\n  {glyphs.TICK} Wrangled {rows:,} row(s) into:\n    {out}",
                  style="green", wrap=False)
    with out.open("r", newline="", encoding="utf-8-sig") as f:
        has_text = "text" in (next(csv.reader(f), []))
    if has_text:
        prompter.note("  Any Taters text step can read this file directly — "
                      "and so can R, Excel, or pandas.", style="dim")
    else:
        prompter.note("  R, Excel, or pandas can read this file directly.",
                      style="dim")
    prompter.pause()
    return True


TASK = Task(
    id="wrangle",
    label="Wrangle data",
    help="Collect text into a tidy spreadsheet: a folder of documents to "
         "CSV, or a CSV aggregated by any column(s).",
    run=_run,
)
