"""
The two database errands under "Wrangle data": a table (or a join of
tables) out to a spreadsheet, and a spreadsheet into a table.

Copying out needs only permission to read, which is what most lab accounts
have; it is also the way a person whose account cannot write still gets
their data into a run. Copying in needs permission to create a table, and
writes into a shared database, so it reads back exactly what it is about to
do and asks before doing it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...helpers import db, sources, tables
from ..db_source import (JoinState, _rows_word, ask_columns, inspection_for,
                         list_tables, pick_connection, plan_joins)
from ..prompts import Choice, GoBack
from . import TaskContext
from .wrangle import _ask_output, _with_display

__all__ = ["db_out_flow", "db_in_flow"]

_NONE = "\x00none"


def _in_order(steps):
    from ..wizard import _ask_in_order

    _ask_in_order(steps)


def db_out_flow(ctx: TaskContext) -> Optional[Path]:
    """
    Tables out of a database, step by step: each table as its own
    spreadsheet, or several joined into one; every column or chosen ones;
    where the files go. The join path is the run's own column conversation.
    """
    from ..browse import browse_for_folder
    from ..prompts import ask_at_least_one

    prompter = ctx.prompter
    state: Dict[str, Any] = {"picked": None, "shape": "", "names": [], "keep_all": True,
                             "kept": None, "folder": None, "replace": False,
                             "chosen": None, "only_tables": None, "base": "", "join": None}

    def ask_connection() -> bool:
        state["picked"] = pick_connection(prompter, need_write=False)
        state["tables"] = list_tables(prompter, state["picked"])
        return True

    def ask_shape() -> bool:
        prompter.reason("A plain copy, one spreadsheet per table -- or columns from "
                        "several tables beside each other, matched on an ID they share.")
        state["shape"] = str(prompter.select("What do you want out?", [
            Choice("dump", "Each table as its own spreadsheet",
                   "One CSV file per table, named after it"),
            Choice("join", "One spreadsheet from several tables, joined",
                   "Pick columns across tables; the join is worked out for you"),
        ], default=state["shape"] or None))
        return True

    # ---- each table as its own file --------------------------------------
    def ask_tables() -> bool:
        if state["shape"] != "dump":
            return False
        usable = [t for t in state["tables"] if t.columns]
        if not usable:
            prompter.note(f"  {state['picked'].name} has no tables this account can see.",
                          style="yellow")
            raise GoBack()
        ticked = set(state["names"])
        rows = [Choice("\x00all", f"Every table ({len(usable)})",
                       "Tick this to take them all",
                       children=tuple(t.name for t in usable))]
        rows += [Choice(t.name, t.name + (" (view)" if t.kind == "view" else ""),
                        (f"text: {', '.join(t.text_columns[:4])}" if t.text_columns
                         else "no long-text columns"),
                        checked=t.name in ticked,
                        annotation=f"{_rows_word(t)}  {len(t.columns)} columns", depth=1)
                for t in usable]
        picked = [v for v in ask_at_least_one(prompter, "Which tables?", rows, thing="one table")
                  if v != "\x00all"]
        if picked != state["names"]:
            state["kept"] = None
        state["names"] = picked
        return True

    def ask_keep() -> bool:
        if state["shape"] != "dump":
            return False
        state["keep_all"] = prompter.confirm("Keep every column of each table?",
                                             default=state["keep_all"])
        if state["keep_all"]:
            state["kept"] = None
            return True
        wanted = [t for t in state["tables"] if t.name in state["names"]]
        before = (state["kept"].values if state["kept"]
                  else [f"{t.name}.{c.name}" for t in wanted for c in t.columns])
        state["kept"] = ask_columns(prompter, state["picked"], wanted,
                                    question="Which columns do you want to keep?",
                                    checked=before, only_tables=state["names"])
        return True

    def ask_folder() -> bool:
        if state["shape"] != "dump":
            return False
        while True:
            folder = browse_for_folder(prompter, question="Where should the spreadsheets go?",
                                       start=state["folder"] or ctx.cwd)
            files = [folder / f"{n}.csv" for n in state["names"]]
            exist = [f.name for f in files if f.exists()]
            state["folder"] = folder
            if not exist:
                state["replace"] = False
                return True
            prompter.note(f"  {len(exist)} of these already exist in {folder}: "
                          f"{', '.join(exist[:5])}{' …' if len(exist) > 5 else ''}.",
                          style="yellow")
            if prompter.confirm("Replace them?", default=False):
                state["replace"] = True
                return True

    # ---- one spreadsheet, joined -----------------------------------------
    def ask_which() -> bool:
        if state["shape"] != "join":
            return False
        prompter.reason("Columns from several tables are joined for you, on the links "
                        "the database declares; the read-back says how the rows matched.")
        before = state["chosen"].values if state["chosen"] else []
        state["chosen"] = ask_columns(
            prompter, state["picked"], state["tables"],
            question="Which columns do you want in the spreadsheet?",
            checked=before, only_tables=state["only_tables"])
        state["only_tables"] = state["chosen"].tables
        if state["chosen"].values != before:
            state["join"] = None
        return True

    def ask_join() -> bool:
        if state["shape"] != "join" or state["join"] is not None:
            return False
        js, asked = plan_joins(
            prompter, state["picked"], state["tables"], state["chosen"],
            base=state["base"], allow_fan_out=True,
            row_question="Which table should each row of the spreadsheet come from?")
        state["join"], state["base"] = js, js.base.name
        return asked

    def ask_out() -> bool:
        if state["shape"] != "join":
            return False
        js: JoinState = state["join"]
        spec = js.source()
        insp = inspection_for(prompter, js)
        cols = list(insp.columns)
        joined = [j["table"] for j in js.joins if j.get("cols") is None]
        tail = ""
        if joined:
            missing = sum(r.left.rows_unmatched for r in js.reports.get(str(spec.to_dict()), [])[-1:])
            tail = (f", with {', '.join(joined)}' columns beside it"
                    + (f" ({missing:,} have none)" if missing else ""))
        prompter.note(f"  {insp.scanned:,} rows × {len(cols)} columns: every row of "
                      f"{js.base.name}{tail}.", style="cyan")
        sql, _params = sources.render(spec, js.picked.session, columns=cols)
        prompter.note(f"  The query: {sql}", style="dim", wrap=False)
        name = js.base.name + (f"_with_{joined[0]}" if joined else "")
        state["out"], state["overwrite"] = _ask_output(prompter, ctx.cwd / f"{name}.csv")
        state["spec"], state["columns"] = spec, cols
        return True

    _in_order([ask_connection, ask_shape, ask_tables, ask_keep, ask_folder,
               ask_which, ask_join, ask_out])
    picked = state["picked"]
    if state["shape"] == "join":
        js: JoinState = state["join"]
        result = _with_display(
            prompter, f"wrangle: copy out of {js.picked.name}", tables.export_table,
            source=state["spec"].to_dict(), out_csv=state["out"],
            columns=state["columns"], overwrite_existing=state["overwrite"], verbose=False)
        return Path(result)

    # the dump: one file per table, the read-back first
    by_name = {t.name: t for t in state["tables"]}
    kept = state["kept"].by_table if state["kept"] else {}
    lines = []
    for n in state["names"]:
        t = by_name[n]
        cols = kept.get(n, [c.name for c in t.columns]) if state["kept"] else \
            [c.name for c in t.columns]
        lines.append(f"{n}.csv ({_rows_word(t)}, {len(cols)} of {len(t.columns)} columns)"
                     if state["kept"] else f"{n}.csv ({_rows_word(t)}, {len(cols)} columns)")
    prompter.note(f"  {len(state['names'])} table(s) → {state['folder']}:", style="cyan")
    for line in lines:
        prompter.note(f"      {line}", style="cyan", wrap=False)
    written = []
    for n in state["names"]:
        if state["kept"] and n not in kept:
            prompter.note(f"  {n}: no columns kept, so no file.", style="dim")
            continue
        cols = kept.get(n) if state["kept"] else None
        out = state["folder"] / f"{n}.csv"
        _with_display(
            prompter, f"wrangle: copy {n} out of {picked.name}", tables.export_table,
            source={"connection": picked.name, "from": {"table": n}}, out_csv=out,
            columns=cols, overwrite_existing=state["replace"], verbose=False)
        written.append(out)
    prompter.note(f"\n  ✓ Wrote {len(written)} file(s) into {state['folder']}.",
                  style="green", wrap=False)
    prompter.note("  R, Excel, pandas and every Taters spreadsheet flow can read them.",
                  style="dim")
    prompter.pause()
    return None


def db_in_flow(ctx: TaskContext) -> Optional[Path]:
    from .wrangle_tables import _Table, _open_file

    prompter = ctx.prompter
    cache: Dict[Any, _Table] = {}
    state: Dict[str, Any] = {"picked": None, "table": None, "name": "", "key": ""}

    def ask_connection() -> bool:
        state["picked"] = pick_connection(prompter, need_write=True)
        return True

    def ask_file() -> bool:
        state["table"] = _open_file(prompter, "Where is the spreadsheet?",
                                    start=ctx.cwd, cache=cache)
        return True

    def ask_name() -> bool:
        picked = state["picked"]
        session: db.Session = picked.session
        default = state["name"] or db.sanitize_table_name(state["table"].path.stem)
        while True:
            typed = str(prompter.text("What should the table be called?",
                                      default=default)).strip()
            name = db.sanitize_table_name(typed)
            if name != typed:
                prompter.note(f"  It will be called {name} -- letters, numbers and "
                              f"underscores only.", style="dim")
            owned = db.registered(session, name)
            if session.d.table_exists(session, name, picked.profile.schema):
                if owned is None:
                    prompter.note(f"  {picked.name} already has a table called {name} "
                                  f"that Taters did not make.", style="yellow")
                    answer = str(prompter.select("What now?", [
                        Choice("rename", "Pick a different name"),
                        Choice("back", "Back")]))
                    if answer == "back":
                        raise GoBack()
                    continue
                if not picked.probe.drop:
                    prompter.note(f"  {name} is already there and this account cannot "
                                  f"replace tables.", style="yellow")
                    continue
                if not prompter.confirm(f"{name} is already there (Taters made it). "
                                        f"Replace it?", default=False):
                    continue
                state["replace"] = True
            else:
                state["replace"] = False
            state["name"] = name
            return True

    def ask_key() -> bool:
        table: _Table = state["table"]
        from ..db_source import _id_like, column_rows

        unique = [c for c in table.columns if table.is_unique(c)]
        guess = next((c for c in unique if _id_like(c)), unique[0] if unique else _NONE)
        rows = column_rows(table.insp, table.columns,
                           disabled={c: "repeats" for c in table.columns if c not in unique})
        rows.insert(0, Choice(_NONE, "None -- no key", "Rows are stored as they are."))
        state["key"] = str(prompter.select(
            "Which column identifies each row? (it becomes the table's key)", rows,
            default=state["key"] or guess))
        if state["key"] == _NONE:
            state["key"] = ""
        return True

    def ask_write() -> bool:
        table: _Table = state["table"]
        numbers = [c for c in table.columns if table.kinds.get(c) == "numbers"]
        rest = "the rest as text" if len(numbers) < len(table.columns) else "all as numbers"
        stored = (f"{', '.join(numbers[:6])}"
                  + (f" and {len(numbers) - 6} more" if len(numbers) > 6 else "")
                  + f" are stored as numbers; {rest}." if numbers else "Every column is stored as text.")
        prompter.note(f"  {table.insp.scanned:,} rows × {len(table.columns)} columns → "
                      f"{state['name']} in {state['picked'].name}. {stored}", style="cyan")
        if not prompter.confirm("Write it?", default=True):
            raise GoBack()
        return True

    _in_order([ask_connection, ask_file, ask_name, ask_key, ask_write])
    table: _Table = state["table"]
    ref = tables.parse_ref(f"db://{state['picked'].name}/{state['name']}")
    _with_display(
        prompter, f"wrangle: copy into {state['picked'].name}", tables.import_table,
        csv_path=table.path, out=str(ref), unique_key=state["key"],
        delimiter=table.fmt.delimiter, encoding=table.fmt.encoding,
        overwrite_existing=bool(state.get("replace")), verbose=False)
    n = tables.count(ref)
    prompter.note(f"\n  ✓ Wrote {n:,} rows into {state['picked'].name}: {state['name']}.",
                  style="green", wrap=False)
    prompter.note("  Runs on this database can use its columns now, from "
                  "\"Which columns do you need?\"", style="dim")
    prompter.pause()
    return None
