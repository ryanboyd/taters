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


def _file_stem(table: str) -> str:
    """A table's name as a file name: separators and dots that could point
    elsewhere become underscores; the name otherwise stays recognizable."""
    import re

    stem = re.sub(r"[\\/:*?\"<>|\x00]", "_", str(table)).strip()
    stem = re.sub(r"^\.+", "_", stem)
    return stem or "table"


def db_out_flow(ctx: TaskContext) -> Optional[Path]:
    """
    Tables out of a database, step by step. The plain copy -- each table as
    its own spreadsheet -- starts from which connection(s), then which
    database(s) on them, then one screen of their tables arranged under
    each database, where ticking a database takes every table in it; then
    every column or chosen ones; then where the files go. The join path is the run's own column conversation, on
    one database.
    """
    from ..browse import browse_for_folder
    from ..db_source import login, open_database
    from ..prompts import ask_at_least_one
    from .db_connections import INSTALL_DRIVERS, add_connection, driver_missing

    prompter = ctx.prompter
    state: Dict[str, Any] = {"shape": "", "conns": [], "logged_by": {}, "dbs": [],
                             "picked_by": {}, "tables_by": {},
                             "names": [], "keep_all": True, "kept": {}, "folder": None,
                             "replace": False,
                             "picked": None, "chosen": None, "only_tables": None,
                             "base": "", "join": None}

    def ask_shape() -> bool:
        prompter.reason("A plain copy, one spreadsheet per table -- or columns from "
                        "several tables beside each other, matched on an ID they share.")
        state["shape"] = str(prompter.select("What do you want out?", [
            Choice("dump", "Each table as its own spreadsheet",
                   "One CSV file per table, named after it; from one database or several"),
            Choice("join", "One spreadsheet from several tables, joined",
                   "Pick columns across one database's tables; the join is worked out for you"),
        ], default=state["shape"] or None))
        return True

    # ---- each table as its own file --------------------------------------
    def ask_connections() -> bool:
        if state["shape"] != "dump":
            return False
        while True:
            profiles = db.list_profiles()
            if not profiles:
                prompter.reason("Taters needs to know where the database server is and "
                                "how to log in. This is saved for next time; the "
                                "password is not.")
                if add_connection(prompter, test=False) is None:
                    raise GoBack()
                continue
            rows = []
            for pr in profiles:
                missing = driver_missing(pr.dialect)
                rows.append(Choice(pr.name, pr.name, pr.label(), checked=pr.name in state["conns"],
                                   disabled=(f"needs {INSTALL_DRIVERS}" if missing else "")))
            rows.append(Choice("\x00add", "+ Add a connection…",
                               "A server and a login, or a database file"))
            picked = list(ask_at_least_one(prompter, "Which connection(s)?", rows,
                                           thing="one connection"))
            if "\x00add" in picked:
                add_connection(prompter, test=False)
                state["conns"] = [n for n in picked if n != "\x00add"]
                continue
            # log in to each now, so a wrong password is found here and the
            # databases can be listed on the next screen
            logged_by: Dict[str, Any] = {}
            for name in picked:
                found = state["logged_by"].get(name) or login(prompter, db.get_profile(name))
                if found is None:
                    break
                logged_by[name] = found
            else:
                if picked != state["conns"]:
                    state["dbs"], state["names"], state["kept"] = [], [], {}
                state["conns"] = picked
                state["logged_by"] = logged_by
                return True

    def ask_databases() -> bool:
        """Which database(s) on each connection: one screen, the databases
        arranged under their connection, a SQLite file taken as is."""
        if state["shape"] != "dump":
            return False
        rows = []
        ticked = set(state["dbs"])
        auto = []
        for name in state["conns"]:
            logged = state["logged_by"][name]
            if logged.profile.dialect == "sqlite":
                auto.append(f"{name}/main")
                continue
            listed = logged.databases
            if not listed:
                prompter.note(f"  {name}: this login cannot list any databases; one can "
                              f"still be typed in.", style="yellow")
            rows.append(Choice(f"\x00conn:{name}", name, f"every database on {name}",
                               annotation=f"{len(listed)} databases",
                               children=tuple(f"{name}/{d}" for d in listed)))
            for d in listed:
                rows.append(Choice(f"{name}/{d}", d, checked=f"{name}/{d}" in ticked, depth=1))
            rows.append(Choice(f"\x00type:{name}", "Type a name…", "One not listed here",
                               depth=1))
        if not rows:
            picked = auto
        else:
            if len(state["conns"]) > 1:
                prompter.reason("Tick a connection to take every database on it, or open "
                                "it up and tick the ones you want.")
            picked = [v for v in ask_at_least_one(prompter, "Which database(s)?", rows,
                                                  thing="one database")
                      if not v.startswith("\x00conn:")]
            typed = [v for v in picked if v.startswith("\x00type:")]
            picked = [v for v in picked if not v.startswith("\x00type:")]
            for v in typed:
                name = v.partition(":")[2]
                d = str(prompter.text(f"Which database on {name}?")).strip()
                if d:
                    picked.append(f"{name}/{d}")
            picked = auto + picked
        # open each: a database this login cannot read is found here
        picked_by: Dict[str, Any] = {}
        for value in picked:
            name, _, d = value.partition("/")
            found = state["picked_by"].get(value) or open_database(
                prompter, state["logged_by"][name], d)
            if found is None:
                continue
            picked_by[value] = found
        if not picked_by:
            if not rows:
                raise GoBack()
            return ask_databases()
        if list(picked_by) != state["dbs"]:
            state["names"], state["kept"] = [], {}
        state["dbs"] = list(picked_by)
        state["picked_by"] = picked_by
        for value, found in picked_by.items():
            if value not in state["tables_by"]:
                state["tables_by"][value] = list_tables(prompter, found)
        return bool(rows)

    def _heading(value: str) -> str:
        """``study7 on lab`` for a database; the connection's name for a file."""
        return state["picked_by"][value].label

    def ask_tables() -> bool:
        if state["shape"] != "dump":
            return False
        rows = []
        ticked = set(state["names"])
        any_table = False
        for value in state["dbs"]:
            usable = [t for t in state["tables_by"][value] if t.columns]
            if not usable:
                prompter.note(f"  {_heading(value)} has no tables this account can see.",
                              style="yellow")
                continue
            any_table = True
            rows.append(Choice(f"\x00db:{value}", _heading(value),
                               f"every table of {_heading(value)}",
                               annotation=f"{len(usable)} tables",
                               children=tuple(f"{value}/{t.name}" for t in usable)))
            width = max(len(t.name) for t in usable)
            for t in usable:
                v = f"{value}/{t.name}"
                rows.append(Choice(v, t.name.ljust(width) + (" (view)" if t.kind == "view" else ""),
                                   (f"text: {', '.join(t.text_columns[:4])}" if t.text_columns
                                    else "no long-text columns"),
                                   checked=v in ticked,
                                   annotation=f"{_rows_word(t)}  {len(t.columns)} columns",
                                   depth=1))
        if not any_table:
            raise GoBack()
        prompter.reason("Tick a database to take every table in it, or open it up and "
                        "tick the tables you want.")
        picked = [v for v in ask_at_least_one(prompter, "Which tables?", rows, thing="one table")
                  if not v.startswith("\x00db:")]
        if picked != state["names"]:
            state["kept"] = {}
        state["names"] = picked
        return True

    def ask_keep() -> bool:
        if state["shape"] != "dump":
            return False
        state["keep_all"] = prompter.confirm("Keep every column of each table?",
                                             default=state["keep_all"])
        if state["keep_all"]:
            state["kept"] = {}
            return True
        # one column screen per database, each with everything ticked: the
        # work is unticking what you don't want
        for value in state["dbs"]:
            mine = [v.rpartition("/")[2] for v in state["names"] if v.startswith(value + "/")]
            wanted = [t for t in state["tables_by"][value] if t.name in mine]
            if not wanted:
                continue
            before = (state["kept"][value].values if value in state["kept"]
                      else [f"{t.name}.{c.name}" for t in wanted for c in t.columns])
            state["kept"][value] = ask_columns(
                prompter, state["picked_by"][value], wanted,
                question=(f"Which columns of {_heading(value)}'s tables do you want to keep?"
                          if len(state["dbs"]) > 1 else "Which columns do you want to keep?"),
                checked=before, only_tables=mine, ask_tables=False)
        return True

    def _subfolders() -> Dict[str, str]:
        """A folder per database when there are several: the database's
        name, or connection-database when two share a name."""
        if len(state["dbs"]) <= 1:
            return {v: "" for v in state["dbs"]}
        out = {}
        for value in state["dbs"]:
            name, _, d = value.partition("/")
            stem = _file_stem(name if d == "main" else d)
            out[value] = stem
        counts = {}
        for stem in out.values():
            counts[stem] = counts.get(stem, 0) + 1
        for value, stem in list(out.items()):
            if counts[stem] > 1:
                out[value] = f"{value.partition('/')[0]}-{stem}"
        return out

    def _files() -> Dict[str, Path]:
        """Where each ticked table goes: ``<table>.csv`` in the folder, or
        under a folder per database when there are several."""
        out = {}
        subs = _subfolders()
        for value in state["names"]:
            where, _, table = value.rpartition("/")
            base = state["folder"] / subs[where] if subs[where] else state["folder"]
            # a table's name comes from the database, not from the person:
            # one called "../../something" must not climb out of the folder
            out[value] = base / f"{_file_stem(table)}.csv"
        return out

    def ask_folder() -> bool:
        if state["shape"] != "dump":
            return False
        while True:
            folder = browse_for_folder(prompter, question="Where should the spreadsheets go?",
                                       start=state["folder"] or ctx.cwd)
            state["folder"] = folder
            files = _files()
            exist = [f.relative_to(folder).as_posix() for f in files.values() if f.exists()]
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
    def ask_connection() -> bool:
        if state["shape"] != "join":
            return False
        while True:
            state["picked"] = pick_connection(prompter, need_write=False)
            try:
                state["tables"] = list_tables(prompter, state["picked"])
            except GoBack:
                continue        # the server could not answer; said, so pick again
            return True

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

    _in_order([ask_shape, ask_connections, ask_databases, ask_tables, ask_keep, ask_folder,
               ask_connection, ask_which, ask_join, ask_out])
    if state["shape"] == "join":
        js: JoinState = state["join"]
        result = _with_display(
            prompter, f"wrangle: copy out of {js.picked.name}", tables.export_table,
            source=state["spec"].to_dict(), out_csv=state["out"],
            columns=state["columns"], overwrite_existing=state["overwrite"], verbose=False)
        return Path(result)

    # the dump: one file per table, the read-back first
    files = _files()
    prompter.note(f"  {len(files)} table(s) → {state['folder']}:", style="cyan")
    plan = []
    for value, out in files.items():
        where, _, table = value.rpartition("/")
        name, _, database = where.partition("/")
        info = next(t for t in state["tables_by"][where] if t.name == table)
        cols = None
        if state["kept"]:
            cols = state["kept"].get(where).by_table.get(table) if where in state["kept"] else None
            if not cols:
                prompter.note(f"      {out.relative_to(state['folder']).as_posix()}: no columns "
                              f"kept, so no file.", style="dim")
                continue
            words = f"{len(cols)} of {len(info.columns)} columns"
        else:
            words = f"{len(info.columns)} columns"
        prompter.note(f"      {out.relative_to(state['folder']).as_posix()} ({_rows_word(info)}, "
                      f"{words})", style="cyan", wrap=False)
        plan.append((name, database, table, out, cols))
    where = sorted({_heading(f"{n}/{d}") for n, d, _t, _o, _c in plan})
    written = _with_display(
        prompter, f"wrangle: copy {len(plan)} table(s) out of {', '.join(where)}",
        _dump_all, plan=plan, replace=state["replace"])
    prompter.note(f"\n  ✓ Wrote {written} file(s) into {state['folder']}.",
                  style="green", wrap=False)
    prompter.note("  R, Excel, pandas and every Taters spreadsheet flow can read them.",
                  style="dim")
    prompter.pause()
    return None


def _dump_all(*, plan, replace: bool, on_progress=None) -> int:
    """
    Every table in ``plan`` to its file, under one progress bar that counts
    tables, with a sub-bar beneath it for the table in hand counting its
    rows as they stream out. One bar per table, each reading 1/1, told a
    person nothing about how far along the whole job was.
    """
    n = len(plan)
    done = 0

    def say(i=None, table="", done_rows=None, total_rows=None, finished=False):
        if on_progress is None:
            return
        extra = {}
        if table:
            extra = dict(item=i, item_name=table, item_done=done_rows, item_total=total_rows,
                         item_unit="rows", item_finished=finished)
        on_progress(done, n, message=table or None, unit="tables", **extra)

    say()
    for i, (name, database, table, out, cols) in enumerate(plan):
        out.parent.mkdir(parents=True, exist_ok=True)

        def sub(d=None, t=None, message=None, unit=None, _i=i, _table=table, **_extra):
            say(_i, _table, d, t)
        tables.export_table(
            source={"connection": name, "database": database, "from": {"table": table}},
            out_csv=out, columns=cols, overwrite_existing=replace, verbose=False,
            on_progress=sub)
        done += 1
        say(i, table, finished=True)
    return done


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
                    prompter.note(f"  {picked.label} already has a table called {name} "
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
                      f"{state['name']} in {state['picked'].label}. {stored}", style="cyan")
        if not prompter.confirm("Write it?", default=True):
            raise GoBack()
        return True

    _in_order([ask_connection, ask_file, ask_name, ask_key, ask_write])
    table: _Table = state["table"]
    ref = tables.parse_ref(f"{state['picked'].root}/{state['name']}")
    _with_display(
        prompter, f"wrangle: copy into {state['picked'].name}", tables.import_table,
        csv_path=table.path, out=str(ref), unique_key=state["key"],
        delimiter=table.fmt.delimiter, encoding=table.fmt.encoding,
        overwrite_existing=bool(state.get("replace")), verbose=False)
    n = tables.count(ref)
    prompter.note(f"\n  ✓ Wrote {n:,} rows into {state['picked'].label}: {state['name']}.",
                  style="green", wrap=False)
    prompter.note("  Runs on this database can use its columns now, from "
                  "\"Which columns do you need?\"", style="dim")
    prompter.pause()
    return None
