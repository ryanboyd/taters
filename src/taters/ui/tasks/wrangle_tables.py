"""
The merge, stack and reshape conversations of "Wrangle data".

Three flows, each a chain of questions through the wizard's
``_ask_in_order`` so Esc steps back one question with the earlier answers
kept, and each with the same shape: pick the file(s), pick the columns,
*read back what will happen* -- with real counts, from a pass over the
files that writes nothing -- and only then ask where the result goes. The
read-back is the whole point. "12,340 of 12,500 rows match" is the sentence
that catches a key typed as P001 on one side and 1 on the other before an
analysis is built on top of it, and nothing else in the flow can.

The work itself is :mod:`taters.helpers.reshape`; this file only asks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...helpers import reshape
from ..columns import Inspection, inspect_csv
from ..prompts import Cancelled, Choice, GoBack, Prompter, ask_at_least_one
from . import TaskContext
from .wrangle import _ask_output, _with_display

__all__ = ["merge_flow", "stack_flow", "reshape_flow"]

#: The tickable "no, number the rows" row. NUL-prefixed like the other UI
#: sentinels so it can never collide with a real column name.
_NO_ID = "\x00no_id"

#: Past this many distinct values a column cannot become columns: the
#: helper refuses it, so the picker grays it out with the number instead.
_MAX_WIDE_COLUMNS = 1000

_SUFFIXES = (".csv", ".tsv", ".txt")


@dataclass
class _Table:
    """A file the flow has opened: how to read it, and what the scan found."""

    fmt: reshape.TableFormat
    insp: Inspection
    kinds: Dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.fmt.path

    @property
    def name(self) -> str:
        return self.fmt.name

    @property
    def columns(self) -> List[str]:
        return list(self.fmt.columns)

    def unique(self, column: str) -> int:
        return self.insp.unique_counts([column]).get(column, 0)

    def is_unique(self, column: str) -> bool:
        """One value per row, as far as the scan could tell."""
        stats = self.insp.stats.get(column)
        if stats is None or not stats.filled or stats.unique_over:
            return False
        return stats.unique == stats.filled == self.insp.scanned

    def is_empty(self, column: str) -> bool:
        return self.kinds.get(column) == "empty"


def _fold(name: str) -> str:
    return name.strip().casefold()


def _progress(seen: Callable[[int], None]):
    """The planners' progress callback, feeding the screen's running count."""
    def on_progress(done, total=None, message=None, **_extra):
        if done:
            seen(int(done))
    return on_progress


def _open_file(prompter: Prompter, question: str, *, start: Path,
               cache: Dict[Any, _Table], forbid: Sequence[Path] = ()) -> _Table:
    """
    Browse to a spreadsheet, work out how to read it, scan it out loud.

    Anything wrong with the file -- empty, not a CSV, an encoding it cannot
    read -- is said and the browser comes back, rather than cancelling the
    whole flow for one wrong click. A ragged file is offered as-is with a
    warning, since the pass that follows pads and trims the rows anyway.
    """
    from ..browse import browse_for_file

    taken = {Path(f).resolve() for f in forbid}
    while True:
        path = browse_for_file(prompter, question=question, start=start,
                               suffixes=_SUFFIXES)
        if path.resolve() in taken:
            prompter.note("  That's the same file -- pick the other one.",
                          style="yellow")
            continue
        try:
            stat = path.stat()
            key = (path.resolve(), stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = (path.resolve(), 0, 0)
        if key in cache:
            return cache[key]
        try:
            fmt = reshape.detect_format(path)
        except (ValueError, OSError) as e:
            prompter.note(f"  {e}", style="yellow")
            continue
        try:
            with prompter.scanning(f"Reading {path.name}") as seen:
                insp = inspect_csv(path, delimiter=fmt.delimiter,
                                   encoding=fmt.encoding, on_rows=seen)
        except Exception as e:
            prompter.note(f"  Could not read {path.name}: {e}", style="yellow")
            continue
        for line in fmt.notes():
            prompter.note(f"  {line}", style="dim")
        trouble = insp.trouble()
        if trouble:
            prompter.note(f"  {trouble}", style="yellow")
            if not prompter.confirm("Use it anyway?", default=False):
                continue
        table = _Table(fmt=fmt, insp=insp,
                       kinds={c: insp.kind(c) for c in fmt.columns})
        cache[key] = table
        return table


def _rows(table: _Table, columns: Sequence[str], *,
          checked: Sequence[str] = (),
          help_of: Optional[Callable[[str], str]] = None,
          disabled: Optional[Dict[str, str]] = None) -> List[Choice]:
    """Column rows the way every other picker draws them: kind and count."""
    from ..wizard import _kind_rows

    return _kind_rows(columns, table.kinds, checked=checked, help_of=help_of,
                      counts=table.insp.unique_counts(columns),
                      disabled=disabled)


def _cycle(table: _Table, columns: Sequence[str]):
    from ..wizard import _cycler

    return _cycler(table.kinds, table.insp, columns,
                   counts=table.insp.unique_counts(columns))


def _in_order(steps: Sequence[Callable[[], bool]]) -> None:
    from ..wizard import _ask_in_order

    _ask_in_order(steps)


def _id_like(name: str) -> bool:
    n = _fold(name)
    return n in ("id", "pid", "sid", "uid", "participant", "subject", "key") \
        or n.endswith("_id") or n.endswith("id") and len(n) <= 6 \
        or "participant" in n or "subject" in n


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _suggest_key(left: _Table, right: _Table) -> List[str]:
    """
    The one column most likely to be the key: in both files, unique on the
    right (that is what "one row per participant" looks like), unique on
    the left as a tie-breaker, called something like an id. Nothing when
    the files share no column name.
    """
    right_names = {_fold(c): c for c in right.columns}
    best: Optional[Tuple[int, int, str]] = None
    for i, c in enumerate(left.columns):
        other = right_names.get(_fold(c))
        if other is None or left.is_empty(c) or right.is_empty(other):
            continue
        score = (2 if right.is_unique(other) else 0) \
            + (1 if left.is_unique(c) else 0) + (1 if _id_like(c) else 0)
        if best is None or score > best[0]:
            best = (score, -i, c)
    return [best[2]] if best else []


def merge_flow(ctx: TaskContext) -> Optional[Path]:
    prompter = ctx.prompter
    cache: Dict[Any, _Table] = {}
    plans: Dict[Any, reshape.MergeReport] = {}
    state: Dict[str, Any] = {"left": None, "right": None, "keys": [],
                             "right_keys": {}, "cols": None, "key_match": "trim",
                             "how": "left", "report": None}

    def left() -> _Table:
        return state["left"]

    def right() -> _Table:
        return state["right"]

    def ask_left() -> bool:
        prompter.reason(
            "Merging adds columns from one spreadsheet onto the rows of "
            "another, matching rows by an ID they share. Start with the file "
            "whose rows you want to keep -- say, the trials.")
        state["left"] = _open_file(
            prompter, "Where is the spreadsheet whose rows we're keeping?",
            start=ctx.cwd, cache=cache)
        return True

    def ask_right() -> bool:
        prompter.reason(
            "Now the one with the columns to add -- say, one row per "
            "participant with their age and condition.")
        state["right"] = _open_file(
            prompter, "Where is the spreadsheet with the columns to add?",
            start=ctx.cwd, cache=cache, forbid=[left().path])
        return True

    def ask_keys() -> bool:
        right_names = {_fold(c): c for c in right().columns}
        checked = state["keys"] or _suggest_key(left(), right())

        def help_of(c: str) -> str:
            other = right_names.get(_fold(c))
            if other is None:
                return f"not in {right().name}"
            parts = [f"also in {right().name}"]
            if right().is_unique(other):
                parts.append("one row each there")
            if not left().is_unique(c):
                parts.append("repeats here")
            return " · ".join(parts)

        prompter.reason(
            "Tick the column(s) that identify the same thing in both files: "
            "a participant ID, or participant and session together.")
        disabled = {c: "empty" for c in left().columns if left().is_empty(c)}
        state["keys"] = ask_at_least_one(
            prompter, "Which column(s) match rows between the two files?",
            _rows(left(), left().columns, checked=checked, help_of=help_of,
                  disabled=disabled),
            thing="one column", cycle=_cycle(left(), left().columns))
        return True

    def ask_right_keys() -> bool:
        """Only for a key the right file does not have under the same name."""
        right_names = {_fold(c): c for c in right().columns}
        mapping: Dict[str, str] = {}
        asked = False
        for k in state["keys"]:
            auto = right_names.get(_fold(k))
            if auto is not None:
                mapping[k] = auto
                continue
            free = [c for c in right().columns if c not in mapping.values()]
            earlier = state["right_keys"].get(k)
            mapping[k] = str(prompter.select(
                f"In {right().name}, which column holds {k}?",
                _rows(right(), free),
                default=earlier if earlier in free else None,
                cycle=_cycle(right(), free)))
            asked = True
        state["right_keys"] = mapping
        return asked

    def ask_cols() -> bool:
        used = set(state["right_keys"].values())
        candidates = [c for c in right().columns if c not in used]
        if len(candidates) <= 1:
            state["cols"] = None
            return False
        if prompter.confirm(f"Bring over every other column from "
                            f"{right().name}?", default=state["cols"] is None):
            state["cols"] = None
            return True
        state["cols"] = ask_at_least_one(
            prompter, "Bring over which column(s)?",
            _rows(right(), candidates, checked=state["cols"] or ()),
            thing="one column", cycle=_cycle(right(), candidates))
        return True

    def plan(key_match: str) -> reshape.MergeReport:
        key = (left().path, right().path, tuple(state["keys"]),
               tuple(sorted(state["right_keys"].items())),
               tuple(state["cols"] or ()), key_match)
        if key not in plans:
            with prompter.scanning(f"Matching {left().name} against "
                                   f"{right().name}") as seen:
                plans[key] = reshape.plan_merge(
                    left_csv=left().path, right_csv=right().path,
                    on=state["keys"],
                    right_on=[state["right_keys"][k] for k in state["keys"]],
                    right_cols=state["cols"], key_match=key_match,
                    left_delimiter=left().fmt.delimiter,
                    right_delimiter=right().fmt.delimiter,
                    left_encoding=left().fmt.encoding,
                    right_encoding=right().fmt.encoding,
                    on_progress=_progress(seen))
        return plans[key]

    def read_back(report: reshape.MergeReport) -> None:
        for line in report.sentences(state["how"]):
            prompter.note(f"  {line}", style="cyan")

    def ask_keys_and_check() -> bool:
        """
        The key questions and the read-back, as one step: a merge that
        cannot go ahead comes straight back to the keys, which is the only
        answer that can fix it.
        """
        while True:
            _in_order([ask_keys, ask_right_keys, ask_cols])
            report = plan(state["key_match"])
            read_back(report)
            blocked = ""
            if report.relationship == "many-to-many":
                blocked = "The files can't be matched as they are. What now?"
            elif not report.matched and report.left.rows > report.left.blank_keys:
                blocked = "Nothing matched. What now?"
            if blocked:
                answer = str(prompter.select(blocked, [
                    Choice("back", "Go back and pick the matching columns again"),
                    Choice("cancel", "Cancel"),
                ]))
                if answer == "cancel":
                    raise Cancelled()
                continue
            if report.loose_extra_matches and state["key_match"] != "loose":
                l, r = report.loose_examples[0] if report.loose_examples else ("", "")
                prompter.reason(
                    f"Some IDs only differ in how they're written -- {l} in "
                    f"{left().name} is {r} in {right().name}."
                    + (f" {left().name} has IDs that differ only in capitals "
                       f"or spacing, so capitals matter there."
                       if report.left.loose_collisions else "")
                    + (f" {right().name} has IDs that differ only in capitals "
                       f"or spacing, so capitals matter there."
                       if report.right.loose_collisions else ""))
                safe = not (report.left.loose_collisions
                            or report.right.loose_collisions)
                if prompter.confirm(
                        f"Match those too? That's "
                        f"{report.loose_extra_matches:,} more rows.",
                        default=safe):
                    state["key_match"] = "loose"
                    report = plan("loose")
                    read_back(report)
            state["report"] = report
            return True

    def ask_how() -> bool:
        report: reshape.MergeReport = state["report"]
        left_out = report.left.rows_unmatched + report.left.blank_keys
        right_out = report.right.rows_unmatched + report.right.blank_keys
        if not left_out and not right_out:
            state["how"] = "left"
            return False
        rows = report.rows_out
        options = [Choice(
            "left", f"All {rows['left']:,} rows of {left().name}",
            f"The {left_out:,} without a match get empty {right().name} columns."
            if left_out else "Every row matched.")]
        if left_out:
            options.append(Choice(
                "inner", f"Only the {rows['inner']:,} rows that matched",
                f"The {left_out:,} without a match are left out."))
        if right_out:
            options.append(Choice(
                "outer", f"Everything -- {rows['outer']:,} rows",
                f"Also adds the {right_out:,} rows of {right().name} with no "
                f"match, as rows of their own."))
        prompter.reason("Not every row found a partner. Choose what happens "
                        "to the ones that didn't.")
        state["how"] = str(prompter.select(
            "Which rows should the merged spreadsheet have?", options,
            default=state["how"] if any(o.value == state["how"] for o in options)
            else "left"))
        return True

    def ask_out() -> bool:
        default = ctx.cwd / f"{left().path.stem}_with_{right().path.stem}.csv"
        state["out"], state["overwrite"] = _ask_output(
            prompter, default, forbid=[left().path, right().path])
        return True

    _in_order([ask_left, ask_right, ask_keys_and_check, ask_how, ask_out])
    result = _with_display(
        prompter, "wrangle: merge", reshape.merge_spreadsheets,
        left_csv=left().path, right_csv=right().path, on=list(state["keys"]),
        right_on=[state["right_keys"][k] for k in state["keys"]],
        right_cols=state["cols"], how=state["how"],
        key_match=state["key_match"], out_csv=state["out"],
        overwrite_existing=state["overwrite"],
        left_delimiter=left().fmt.delimiter,
        right_delimiter=right().fmt.delimiter,
        left_encoding=left().fmt.encoding, right_encoding=right().fmt.encoding,
        verbose=False)
    return Path(result)


# ---------------------------------------------------------------------------
# Stack
# ---------------------------------------------------------------------------

def stack_flow(ctx: TaskContext) -> Optional[Path]:
    from ..browse import browse_and_tick

    prompter = ctx.prompter
    state: Dict[str, Any] = {"files": [], "loose": False, "source": True,
                             "plan": None}

    def ask_files() -> bool:
        prompter.reason(
            "Stacking puts the rows of several spreadsheets one under "
            "another. Columns with the same name line up; a column only some "
            "files have is left empty in the others.")
        while True:
            picked = browse_and_tick(
                prompter, question="Which spreadsheets should be stacked? "
                "(Space ticks; Enter when done.)",
                start=ctx.cwd, suffixes=_SUFFIXES)
            files: List[Path] = []
            seen = set()
            for p in picked:
                r = Path(p).resolve()
                if r not in seen:
                    seen.add(r)
                    files.append(Path(p))
            if len(files) < 2:
                prompter.note("  Tick at least two files -- space ticks a "
                              "file, Enter finishes.", style="yellow")
                continue
            state["files"] = files
            return True

    def plan(count: bool) -> reshape.StackReport:
        return reshape.plan_stack(
            csv_paths=state["files"],
            match_columns="loose" if state["loose"] else "exact",
            source_col="source" if state["source"] else "",
            count_rows=count,
            on_progress=_progress(state["seen"]) if count else None)

    def ask_check() -> bool:
        try:
            headers = plan(count=False)
        except (ValueError, OSError) as e:
            prompter.note(f"  {e}", style="yellow")
            raise GoBack()
        if headers.near_duplicates and not state["loose"]:
            shown = "; ".join(" / ".join(g) for g in headers.near_duplicates[:3])
            prompter.reason(f"Some column names differ only in capitals or "
                            f"spacing: {shown}.")
            state["loose"] = bool(prompter.confirm(
                "Treat each of those as one column?", default=True))
        with prompter.scanning(f"Reading {len(state['files'])} files") as seen:
            state["seen"] = seen
            report = plan(count=True)
        prompter.table(
            "What's in each file",
            [[f.file, f"{f.rows:,}", str(len(f.columns)),
              ", ".join(f.only_here) or "--"] for f in report.files],
            headers=["file", "rows", "columns", "only here"])
        for f in report.files:
            if f.encoding in ("cp1252", "latin-1", "utf-16"):
                prompter.note(f"  {f.file} was read as {f.encoding} text.",
                              style="dim")
        if report.disjoint_files:
            prompter.reason(
                f"{', '.join(report.disjoint_files)} shares no columns with "
                f"the rest, so stacking gives a checkerboard of blanks. Did "
                f"you mean to match them up by an ID (merge) instead?")
            if not prompter.confirm("Stack them anyway?", default=False):
                raise GoBack()
        state["plan"] = report
        return True

    def ask_source() -> bool:
        report: reshape.StackReport = state["plan"]
        if report.source_col and report.source_col != "source":
            prompter.reason(f"It'll be called {report.source_col}, since "
                            f"source is taken.")
        state["source"] = bool(prompter.confirm(
            "Add a column saying which file each row came from?",
            default=state["source"]))
        return True

    def ask_out() -> bool:
        parents = [str(Path(f).resolve().parent) for f in state["files"]]
        common = Path(os.path.commonpath(parents))
        stem = common.name or state["files"][0].stem
        state["out"], state["overwrite"] = _ask_output(
            prompter, ctx.cwd / f"{stem}_stacked.csv", forbid=state["files"])
        return True

    _in_order([ask_files, ask_check, ask_source, ask_out])
    result = _with_display(
        prompter, "wrangle: stack", reshape.stack_spreadsheets,
        csv_paths=[str(f) for f in state["files"]], out_csv=state["out"],
        overwrite_existing=state["overwrite"],
        source_col="source" if state["source"] else "",
        match_columns="loose" if state["loose"] else "exact", verbose=False)
    return Path(result)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------

def _shared_prefix(columns: Sequence[str]) -> str:
    """
    What the ticked columns all start with, trimmed back to a boundary.

    ``score_t1, score_t2`` share ``score_t``; the boundary rule is what stops
    ``time1, time2`` from losing ``time`` and ``pretest, posttest`` from
    sharing nothing useful. A prefix is worth offering only when what it
    leaves behind still names something.
    """
    common = os.path.commonprefix(list(columns))
    for n in range(len(common), 1, -1):
        p = common[:n]
        if p[-1] in "_-. " or all(len(c) > n and c[n].isdigit() for c in columns):
            if all(c[n:] for c in columns):
                return p
    return ""


def reshape_flow(ctx: TaskContext) -> Optional[Path]:
    prompter = ctx.prompter
    cache: Dict[Any, _Table] = {}
    state: Dict[str, Any] = {
        "table": None, "direction": "longer",
        # wide -> long
        "value_cols": [], "keep": None, "prefix": "", "names_to": "variable",
        "values_to": "value", "drop_blank": False,
        # long -> wide
        "names_from": "", "values_from": "", "keys": [], "carry": [],
        "on_duplicate": "refuse", "plan": None,
    }

    def table() -> _Table:
        return state["table"]

    def ask_file() -> bool:
        state["table"] = _open_file(prompter, "Where is the spreadsheet?",
                                    start=ctx.cwd, cache=cache)
        return True

    def ask_direction() -> bool:
        state["direction"] = str(prompter.select(
            "Which way are we flipping it?", [
                Choice("longer", "Columns into rows (wide to long)",
                       "score_t1, score_t2, score_t3 become one score column, "
                       "plus a column saying which time."),
                Choice("wider", "Rows into columns (long to wide)",
                       "One row per participant per condition becomes one row "
                       "per participant, with a column per condition."),
            ], default=state["direction"]))
        return True

    # ---- wide -> long --------------------------------------------------

    def ask_value_cols() -> bool:
        if state["direction"] != "longer":
            return False
        prompter.reason(
            "Tick the columns that hold the same kind of measurement at "
            "different times or in different conditions. Each one becomes "
            "its own row.")
        cols = table().columns
        while True:
            picked = ask_at_least_one(
                prompter, "Which columns should become rows?",
                _rows(table(), cols, checked=state["value_cols"]),
                thing="two columns", cycle=_cycle(table(), cols))
            if len(picked) >= 2:
                state["value_cols"] = picked
                return True
            prompter.note("  Tick at least two -- one column is already a "
                          "column.", style="yellow")

    def ask_keep() -> bool:
        if state["direction"] != "longer":
            return False
        others = [c for c in table().columns if c not in state["value_cols"]]
        if not others:
            state["keep"] = []
            return False
        prompter.reason("The other columns ride along on every new row. "
                        "Untick any you don't want in the result.")
        state["keep"] = list(prompter.checkbox(
            "Keep which of the other columns?",
            _rows(table(), others,
                  checked=others if state["keep"] is None else state["keep"]),
            cycle=_cycle(table(), others)))
        return True

    def ask_mixed() -> bool:
        if state["direction"] != "longer":
            return False
        kinds = {c: table().kinds.get(c) for c in state["value_cols"]}
        numbers = [c for c, k in kinds.items() if k == "numbers"]
        words = [c for c, k in kinds.items() if k in ("labels", "text")]
        if not numbers or not words:
            return False
        prompter.reason(
            f"{', '.join(words)} hold text while {', '.join(numbers)} hold "
            f"numbers; they'll share one column, which R and Excel will read "
            f"as text.")
        if not prompter.confirm("Stack them together anyway?", default=False):
            raise GoBack()
        return True

    def ask_prefix() -> bool:
        if state["direction"] != "longer":
            return False
        prefix = _shared_prefix(state["value_cols"])
        if not prefix:
            state["prefix"] = ""
            return False
        first = state["value_cols"][0]
        if prompter.confirm(
                f"Drop the shared start '{prefix}' from the names "
                f"({first} becomes {first[len(prefix):]})?", default=True):
            state["prefix"] = prefix
        else:
            state["prefix"] = ""
        return True

    def ask_names() -> bool:
        if state["direction"] != "longer":
            return False
        prompter.reason("Two new columns: one says which of the old columns "
                        "a row came from, the other holds its value.")
        taken = set(state["keep"] or [])
        while True:
            names_to = str(prompter.text(
                "What should the column saying which one be called?",
                default=state["names_to"])).strip() or "variable"
            if names_to in taken:
                prompter.note(f"  There's already a column called {names_to} "
                              f"-- pick another name.", style="yellow")
                continue
            break
        while True:
            values_to = str(prompter.text(
                "And the column holding the values?",
                default=state["values_to"])).strip() or "value"
            if values_to in taken or values_to == names_to:
                prompter.note(f"  There's already a column called {values_to} "
                              f"-- pick another name.", style="yellow")
                continue
            break
        state["names_to"], state["values_to"] = names_to, values_to
        return True

    def ask_blank() -> bool:
        if state["direction"] != "longer":
            return False
        scanned = table().insp.scanned
        blanks = any((table().insp.stats.get(c) is not None
                      and table().insp.stats[c].filled < scanned)
                     for c in state["value_cols"])
        if not blanks:
            state["drop_blank"] = False
            return False
        state["drop_blank"] = not prompter.confirm(
            "Some of those cells are empty. Keep a row for them anyway?",
            default=not state["drop_blank"])
        return True

    # ---- long -> wide --------------------------------------------------

    def ask_names_from() -> bool:
        if state["direction"] != "wider":
            return False
        prompter.reason("Pick the column whose values become the new column "
                        "names -- a condition, a time point, a question number.")
        cols = table().columns
        disabled = {}
        for c in cols:
            n = table().unique(c)
            if table().is_empty(c):
                disabled[c] = "empty"
            elif n > _MAX_WIDE_COLUMNS:
                disabled[c] = (f"{n:,}{'+' if table().insp.stats[c].unique_over else ''} "
                               f"values -- too many to become columns")
        state["names_from"] = str(prompter.select(
            "Which column's values become columns?",
            _rows(table(), cols, disabled=disabled),
            default=state["names_from"] or None, cycle=_cycle(table(), cols)))
        return True

    def ask_values_from() -> bool:
        if state["direction"] != "wider":
            return False
        cols = [c for c in table().columns if c != state["names_from"]]
        state["values_from"] = str(prompter.select(
            "Which column holds the values that go in them?",
            _rows(table(), cols),
            default=state["values_from"] if state["values_from"] in cols else None,
            cycle=_cycle(table(), cols)))
        return True

    def ask_keys() -> bool:
        if state["direction"] != "wider":
            return False
        cols = [c for c in table().columns
                if c not in (state["names_from"], state["values_from"])]
        if not cols:
            prompter.note("  Every column is being pivoted; nothing is left to "
                          "say which rows belong together.", style="yellow")
            raise GoBack()
        prompter.reason(
            "Tick the column(s) that say which rows belong together. "
            "Participant alone gives one row per participant; participant "
            "and session, one row per participant per session.")
        checked = state["keys"] or [c for c in cols if _id_like(c)][:1]
        state["keys"] = ask_at_least_one(
            prompter, "Which column(s) say which rows belong together?",
            _rows(table(), cols, checked=checked), thing="one column",
            cycle=_cycle(table(), cols))
        return True

    def ask_carry() -> bool:
        if state["direction"] != "wider":
            return False
        others = [c for c in table().columns
                  if c not in (state["names_from"], state["values_from"])
                  and c not in state["keys"]]
        if not others:
            state["carry"] = []
            return False
        with prompter.scanning(f"Counting rows per "
                               f"{' + '.join(state['keys'])}") as seen:
            grain = table().insp.grain(state["keys"], track=others, on_rows=seen)
        steady = [c for c in others if grain.constant(c)]
        if not steady:
            state["carry"] = []
            return False
        if not prompter.confirm("Bring any other columns along unchanged?",
                                default=bool(state["carry"])):
            state["carry"] = []
            return True
        prompter.reason(
            "Only the columns with one value per "
            f"{' + '.join(state['keys'])} are offered; the others would have "
            "nothing single to say.")
        state["carry"] = ask_at_least_one(
            prompter, "Bring along which column(s)?",
            _rows(table(), steady, checked=state["carry"]), thing="one column",
            cycle=_cycle(table(), steady))
        return True

    def ask_check_wide() -> bool:
        if state["direction"] != "wider":
            return False
        with prompter.scanning(f"Reading {table().name}") as seen:
            report = reshape.plan_long_to_wide(
                csv_path=table().path, id_cols=state["keys"],
                names_from=state["names_from"], values_from=state["values_from"],
                carry_cols=state["carry"], on_duplicate="refuse",
                max_columns=_MAX_WIDE_COLUMNS,
                delimiter=table().fmt.delimiter, encoding=table().fmt.encoding,
                on_progress=_progress(seen))
        for line in report.sentences():
            prompter.note(f"  {line}", style="cyan")
        if report.too_many_columns or report.varying_carry:
            prompter.note(f"  {report.refusal()}", style="yellow")
            raise GoBack()
        if report.collisions:
            k, name, values = report.collision_examples[0]
            prompter.reason(
                f"{k} has {len(values)} rows for {state['names_from']} = {name}. "
                f"Esc goes back to add a column that tells them apart.")
            options = []
            if report.collisions_numeric:
                options.append(Choice("mean", "Average them"))
            options += [Choice("first", "Keep the first one"),
                        Choice("last", "Keep the last one")]
            state["on_duplicate"] = str(prompter.select(
                "What should we do when there's more than one?", options,
                default=state["on_duplicate"]
                if any(o.value == state["on_duplicate"] for o in options)
                else options[0].value))
        else:
            state["on_duplicate"] = "refuse"
        state["plan"] = report
        return True

    def ask_out() -> bool:
        if state["direction"] == "longer":
            report = reshape.plan_wide_to_long(
                csv_path=table().path, value_cols=state["value_cols"],
                id_cols=state["keep"], names_to=state["names_to"],
                values_to=state["values_to"], strip_prefix=state["prefix"],
                drop_blank=state["drop_blank"],
                delimiter=table().fmt.delimiter, encoding=table().fmt.encoding)
            for line in report.sentences():
                prompter.note(f"  {line}", style="cyan")
            suffix = "_long"
        else:
            suffix = "_wide"
        state["out"], state["overwrite"] = _ask_output(
            prompter, ctx.cwd / f"{table().path.stem}{suffix}.csv",
            forbid=[table().path])
        return True

    _in_order([ask_file, ask_direction,
               ask_value_cols, ask_keep, ask_mixed, ask_prefix, ask_names, ask_blank,
               ask_names_from, ask_values_from, ask_keys, ask_carry, ask_check_wide,
               ask_out])
    common = dict(csv_path=table().path, out_csv=state["out"],
                  overwrite_existing=state["overwrite"],
                  delimiter=table().fmt.delimiter, encoding=table().fmt.encoding,
                  verbose=False)
    if state["direction"] == "longer":
        result = _with_display(
            prompter, "wrangle: wide to long", reshape.wide_to_long,
            value_cols=state["value_cols"], id_cols=state["keep"],
            names_to=state["names_to"], values_to=state["values_to"],
            strip_prefix=state["prefix"], drop_blank=state["drop_blank"],
            **common)
    else:
        result = _with_display(
            prompter, "wrangle: long to wide", reshape.long_to_wide,
            id_cols=state["keys"], names_from=state["names_from"],
            values_from=state["values_from"], carry_cols=state["carry"],
            on_duplicate=state["on_duplicate"], max_columns=_MAX_WIDE_COLUMNS,
            **common)
    return Path(result)
