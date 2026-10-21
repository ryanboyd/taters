"""
The database conversation: which connection, which columns, and the joins
that follow from them. Shared by the source stage of the wizard and by
Wrangle data, so a person learns it once.

One screen lists every column of every table as ``table.column``; the
person ticks what the run needs. Which table is the base and how the others
join on is worked out from the catalog -- the links the database declares,
then same-named key columns -- and read back in a sentence, computed on
the server: how many rows matched, which did not, what repeats. A question
is only asked when the catalog cannot settle it: which table is one row
per text when several hold text, or how two tables match when nothing
links them. A join that would repeat text rows is not refused, but it is
said loudly and asked about, because every statistic afterward would count
those rows more than once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..helpers import db, sources
from .columns import Inspection
from .db_columns import DbInspection, inspect_source
from .prompts import Cancelled, Choice, GoBack, Prompter, ask_at_least_one

__all__ = ["Picked", "pick_connection", "offer_host_key", "list_tables", "Chosen", "ask_columns",
           "JoinState", "plan_joins", "source_of", "inspection_for", "column_rows",
           "cycle_for", "can_write_here", "qualified"]

_ADD = "\x00add"

#: Past this many columns across the tables, the column screen is narrowed
#: to the tables the person names first; under it, everything fits on one.
MANY_COLUMNS = 150


@dataclass
class Picked:
    """A connection the person chose, logged in and probed."""

    profile: db.Profile
    session: db.Session
    probe: db.Probe

    @property
    def name(self) -> str:
        return self.profile.name


def pick_connection(prompter: Prompter, *, need_write: bool = False,
                    start: str = "") -> Optional[Picked]:
    """
    "Which database?", the password if one is needed, and the probe.

    ``need_write`` is a destination: an account that cannot make tables is
    turned away with the reason and a way out. A source only has to read,
    so a read-only account gets one dim sentence and carries on.
    """
    from .tasks.db_connections import INSTALL_DRIVERS, add_connection, driver_missing

    while True:
        profiles = db.list_profiles()
        if not profiles:
            prompter.reason("Taters needs to know where the database is and how to "
                            "log in. This is saved for next time; the password is not.")
            profile = add_connection(prompter)
            if profile is None:
                raise GoBack()
        else:
            rows = []
            for p in profiles:
                missing = driver_missing(p.dialect)
                rows.append(Choice(p.name, p.name, p.label(),
                                   disabled=(f"needs {INSTALL_DRIVERS}" if missing else "")))
            rows.append(Choice(_ADD, "+ Add a connection…",
                               "A server and a login, or a database file"))
            picked = str(prompter.select("Which database?", rows,
                                         default=start if any(
                                             p.name == start for p in profiles) else None))
            if picked == _ADD:
                profile = add_connection(prompter)
                if profile is None:
                    continue
            else:
                profile = db.get_profile(picked)
        found = _connect(prompter, profile)
        if found is None:
            continue
        if need_write and not found.probe.can_write:
            prompter.reason(
                f"{profile.user or 'This account'} can read {profile.name} but cannot "
                f"create tables there, and this needs to write one. Ask whoever "
                f"manages the database to grant CREATE"
                + (f" on {profile.database}" if profile.database else "")
                + ", or pick a connection that has it.")
            answer = str(prompter.select(f"{profile.name} can't take new tables. "
                                         f"What now?", [
                Choice("again", "Pick a different connection"),
                Choice("back", "Back"),
            ]))
            found.session.close()
            if answer == "back":
                raise GoBack()
            continue
        if not need_write and not found.probe.can_write:
            prompter.note(f"  {profile.user or 'This account'} can read {profile.name} "
                          f"but cannot create tables there, so the results of a run "
                          f"will go to files, or to another database.", style="dim")
        return found


def can_write_here(prompter: Prompter, profile: db.Profile) -> bool:
    """Whether a connection can take this run's tables, said in words if not."""
    found = _connect(prompter, profile)
    if found is None:
        return False
    found.session.close()
    if found.probe.can_write:
        return True
    prompter.note(
        f"  {profile.user or 'This account'} can read {profile.name} but cannot "
        f"create tables there. Ask whoever manages the database to grant CREATE"
        + (f" on {profile.database}" if profile.database else "")
        + ", or pick somewhere else for the results.", style="yellow")
    return False


def offer_host_key(prompter: Prompter, exc: db.HostKeyUnknown, *,
                   save: bool = True) -> Optional[db.Profile]:
    """
    The ssh host is new to this computer: show the key it presented and
    ask. Returns the profile to go on with when accepted (changed only for
    a PuTTY key, which carries the fingerprint), None when declined.
    """
    p = exc.profile
    if not exc.keys:
        prompter.note(f"  {exc}", style="yellow")
        return None
    prompter.reason(f"{p.ssh_host} is new to this computer. ssh remembers each "
                    f"server's key so that a different machine answering to the same "
                    f"name is noticed. This is the key it showed:")
    for k in exc.keys:
        prompter.note(f"    {k.type}  {k.fingerprint}", style="cyan", wrap=False)
    # the reason stays two lines (the screen's rule); the how-to-check is
    # the notes under it, which a person reads in order anyway
    prompter.reason("Accepting means trusting that this is really that server: compare "
                    "the fingerprint above with one you got some other way.")
    prompter.note("  Whoever runs the server can tell you theirs, and PuTTY or ssh showed "
                  "it the first time you logged in by hand. If they match, accept.",
                  style="dim")
    prompter.note("  If you have nothing to compare it with, or you have connected before "
                  "and the key has changed, say no and ask whoever runs the server.",
                  style="dim")
    if not prompter.confirm(f"Accept {p.ssh_host}'s key and continue?", default=False):
        return None
    updated = db.accept_host_keys(p, exc.keys, save=save)
    prompter.note("  Accepted. " + (f"Recorded in {db.known_hosts_path()}, where ssh keeps them."
                                    if not db.uses_putty(p) else
                                    "Taters hands it to plink each time."), style="green")
    return updated


def _connect(prompter: Prompter, profile: db.Profile) -> Optional[Picked]:
    """Log in (asking for the password once, re-asking once), then probe."""
    from .tasks.db_connections import driver_missing

    missing = driver_missing(profile.dialect)
    if missing:
        prompter.note(f"  {profile.name} is a {profile.engine_label} database, and its "
                      f"driver is not installed. "
                      f"{missing.partition(':')[2].strip().replace('[', chr(92) + '[')}",
                      style="yellow")
        return None
    for attempt in range(3):
        try:
            password = db.password_for(profile, ask=prompter.secret)
            with prompter.scanning(f"Connecting to {profile.name}", unit="checks") as step:
                probe = db.probe(profile, password, on_step=step)
            if not probe.select:
                raise db.DbError("; ".join(probe.notes) or "could not connect")
            session = db.connect(profile, password)
        except db.HostKeyUnknown as e:
            accepted = offer_host_key(prompter, e)
            if accepted is None:
                return None
            profile = accepted
            continue
        except db.DbError as e:
            text = str(e)
            if "did not accept that password" in text and profile.password == "ask":
                db.forget_password(profile.name)
                if attempt == 0:
                    prompter.note(f"  {text}", style="yellow")
                    continue
                prompter.reason("Still not accepted, so Taters stopped before the "
                                "account gets locked. Check the username in Settings "
                                "> Database connections, or ask whoever manages the "
                                "database.")
                return None
            if "did not accept that password" in text and profile.password == "env":
                prompter.note(f"  {profile.name} did not accept the password in "
                              f"{db.password_env(profile.name)}.", style="yellow")
                answer = str(prompter.select("What now?", [
                    Choice("type", "Type it in this once"),
                    Choice("other", "Pick a different connection")]))
                if answer == "type":
                    db.remember_password(profile.name, prompter.secret(
                        f"Password for {profile.user} on {profile.name}:"))
                    continue
                return None
            prompter.note(f"  {text}", style="yellow")
            return None
        prompter.note(f"  Connected to {profile.name}"
                      + (f" as {profile.user}." if profile.user else "."), style="green")
        return Picked(profile=profile, session=session, probe=probe)
    return None


# ---------------------------------------------------------------------------
# The catalog, and the one column screen
# ---------------------------------------------------------------------------

def _rows_word(info: db.TableInfo) -> str:
    if info.rows is None:
        return "? rows"
    return f"{'about ' if not info.rows_exact else ''}{info.rows:,} rows"


def list_tables(prompter: Prompter, picked: Picked) -> List[db.TableInfo]:
    with prompter.scanning(f"Looking at the tables in {picked.name}", unit="tables") as seen:
        found = picked.session.d.list_tables(picked.session, picked.profile.schema)
        seen(len(found))
    return found


def qualified(table: str, column: str) -> str:
    """How a column is named on the screen and in a script: ``table.column``,
    the way SQL itself names one."""
    return f"{table}.{column}"


def _split_qualified(value: str, tables: Sequence[db.TableInfo]) -> Tuple[str, str]:
    """``table.column`` back to its parts; a column name may hold dots, a
    table name may not, so the first dot that follows a known table wins."""
    for t in sorted(tables, key=lambda t: -len(t.name)):
        if value.startswith(t.name + "."):
            return t.name, value[len(t.name) + 1:]
    table, _, column = value.partition(".")
    return table, column


@dataclass
class Chosen:
    """What the column screen settled: the columns wanted, per table, in the
    catalog's order."""

    by_table: Dict[str, List[str]]

    @property
    def tables(self) -> List[str]:
        return [t for t, cols in self.by_table.items() if cols]

    @property
    def values(self) -> List[str]:
        return [qualified(t, c) for t, cols in self.by_table.items() for c in cols]

    def __len__(self) -> int:
        return sum(len(c) for c in self.by_table.values())


def _column_help(t: db.TableInfo, c: db.ColumnInfo) -> str:
    parts = []
    if c.name in t.primary_key:
        parts.append("the table's key")
    link = next((f"→ {rt}.{rc}" for col, rt, rc in t.foreign_keys if col == c.name), None)
    if link:
        parts.append(link)
    if c.is_long_text:
        parts.append("long text")
    return " · ".join(parts)


def ask_columns(prompter: Prompter, picked: Picked, tables: Sequence[db.TableInfo], *,
                question: str, checked: Sequence[str] = (),
                only_tables: Optional[List[str]] = None) -> Chosen:
    """
    Every column of every table on one screen, as ``table.column`` under a
    heading per table, with the declared type beside it and what the
    catalog knows (the key, a declared link, long text) as its help.

    A database with hundreds of columns gets one question first -- which
    tables are involved -- so the screen stays a screen. ``only_tables``
    carries that answer across an Esc. Returns what was ticked.
    """
    usable = [t for t in tables if t.rows != 0 and t.columns]
    if not usable:
        prompter.note(f"  {picked.name} has no tables with rows that "
                      f"{picked.profile.user or 'this account'} can see.", style="yellow")
        raise GoBack()
    total = sum(len(t.columns) for t in usable)
    ticked = set(checked)
    if total > MANY_COLUMNS:
        want = set(only_tables or []) or {_split_qualified(v, usable)[0] for v in ticked}
        rows = [Choice(t.name, t.name + (" (view)" if t.kind == "view" else ""),
                       (f"text: {', '.join(t.text_columns[:4])}" if t.text_columns
                        else "no long-text columns"),
                       checked=t.name in want,
                       annotation=f"{_rows_word(t)}  {len(t.columns)} columns")
                for t in sorted(usable, key=lambda t: (0 if t.text_columns else 1,
                                                       t.name.lower()))]
        prompter.reason(f"{picked.name} has {total:,} columns across {len(usable)} "
                        f"tables, so first: which tables hold the ones you need?")
        names = ask_at_least_one(prompter, "Which tables are involved?", rows,
                                 thing="one table")
        usable = [t for t in usable if t.name in set(names)]
    rows = []
    # three columns, lined up: the column, the table it is in, its type.
    # table names have underscores of their own, so a joined-up
    # "table_column" was unreadable; the value a script sees is still
    # table.column, the way SQL names it
    col_w = max(len(c.name) for t in usable for c in t.columns)
    tab_w = max(len(t.name) for t in usable)
    for t in usable:
        rows.append(Choice(f"\x00table:{t.name}", t.name + (" (view)" if t.kind == "view" else ""),
                           f"every column of {t.name}",
                           annotation=f"{_rows_word(t)}  {len(t.columns)} columns",
                           children=tuple(qualified(t.name, c.name) for c in t.columns)))
        for c in t.columns:
            name = qualified(t.name, c.name)
            rows.append(Choice(name, c.name.ljust(col_w), _column_help(t, c),
                               checked=name in ticked,
                               annotation=f"{t.name.ljust(tab_w)}  {c.type.lower() or '?'}",
                               depth=1))
    picked_values = [v for v in ask_at_least_one(prompter, question, rows, thing="one column")
                     if not v.startswith("\x00table:")]
    by_table: Dict[str, List[str]] = {}
    for t in usable:
        cols = [c.name for c in t.columns if qualified(t.name, c.name) in set(picked_values)]
        if cols:
            by_table[t.name] = cols
    return Chosen(by_table=by_table)


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------

@dataclass
class JoinState:
    """What the join conversation has settled so far."""

    picked: Picked
    base: db.TableInfo
    tables: List[db.TableInfo]
    joins: List[Dict[str, Any]] = field(default_factory=list)
    #: The columns the person asked for, when the source is narrowed to them.
    chosen: Optional[Chosen] = None
    #: `Inspection`s per source shape, so Esc never rescans.
    inspections: Dict[str, DbInspection] = field(default_factory=dict)
    reports: Dict[str, List[sources.MergeReport]] = field(default_factory=dict)

    def source(self, upto: Optional[int] = None) -> sources.Source:
        return source_of(self, upto=upto)

    def table(self, name: str) -> db.TableInfo:
        return next(t for t in self.tables if t.name == name)

    def owner(self, column: str) -> str:
        """Which table an output column came from."""
        for name, alias, col in self.output_columns():
            if name == column:
                return self.base.name if alias == "t0" else next(
                    j["table"] for j in self.joins if j["alias"] == alias)
        return self.base.name

    def output_columns(self) -> List[Tuple[str, str, str]]:
        """``(output name, alias, column)`` the source will produce."""
        return _output_columns(self)


def _output_columns(state: JoinState) -> List[Tuple[str, str, str]]:
    """
    The output columns, bare-named unless two tables share a name (then
    ``table__column`` for the second, the spreadsheet merge's rule). With a
    `chosen` set it is those columns plus the base table's key, which every
    row needs whether or not it was ticked; without one, everything.
    """
    taken: Dict[str, str] = {}
    out: List[Tuple[str, str, str]] = []
    right_keys = {(j["alias"], rc) for j in state.joins for _lc, rc in
                  ((lk, rk) for lk, rk in zip(j["keys"], [j["right_keys"][k] for k in j["keys"]]))}

    def take(table: str, alias: str, col: str) -> None:
        name = col
        if name in taken:
            name = f"{table}{sources.SEPARATOR}{col}"
            k = 2
            while name in taken:
                name = f"{table}{sources.SEPARATOR}{col}_{k}"
                k += 1
        taken[name] = alias
        out.append((name, alias, col))

    wanted = state.chosen.by_table if state.chosen else None
    for col in state.base.column_names:
        if wanted is None or col in wanted.get(state.base.name, []) \
                or col in state.base.primary_key:
            take(state.base.name, "t0", col)
    for j in state.joins:
        t = state.table(j["table"])
        for col in t.column_names:
            if (j["alias"], col) in right_keys:
                continue
            if wanted is None:
                if j.get("cols") is None or col in j["cols"]:
                    take(t.name, j["alias"], col)
            elif col in wanted.get(t.name, []):
                take(t.name, j["alias"], col)
    return out


def source_of(state: JoinState, *, upto: Optional[int] = None) -> sources.Source:
    joins = state.joins if upto is None else state.joins[:upto]
    narrowed = state.chosen is not None or any(j.get("cols") for j in joins)
    columns = {name: f"{alias}.{col}" for name, alias, col in _output_columns(state)
               if upto is None or alias == "t0" or alias in {j["alias"] for j in joins}} \
        if narrowed else {}
    return sources.Source(
        connection=state.picked.name, table=state.base.name, alias="t0",
        joins=tuple(sources.Join(table=j["table"], alias=j["alias"], how="left",
                                 on=tuple(j["on"]))
                    for j in joins),
        columns=columns, schema=state.picked.profile.schema)


def inspection_for(prompter: Prompter, state: JoinState,
                   spec: Optional[sources.Source] = None) -> DbInspection:
    """The column facts for the current shape, read once per shape."""
    spec = spec or state.source()
    key = str(spec.to_dict())
    if key not in state.inspections:
        label = " with ".join([spec.table] + [j.table for j in spec.joins])
        with prompter.scanning(f"Reading {label}", unit="columns") as seen:
            state.inspections[key] = inspect_source(spec, state.picked.session,
                                                    on_rows=seen)
    return state.inspections[key]


def column_rows(insp: Inspection, columns: Sequence[str], *,
                labels: Optional[Dict[str, str]] = None,
                checked: Sequence[str] = (),
                help_of: Optional[Callable[[str], str]] = None,
                disabled: Optional[Dict[str, str]] = None) -> List[Choice]:
    """``labels`` maps an output column to the table it came from (see
    `_labels`); the table is shown in a column of its own beside the name,
    lined up, rather than folded into the name."""
    from .wizard import _kind_rows

    kinds = {c: insp.kind(c) for c in columns}
    owners = {c: t for c, t in (labels or {}).items() if t}
    return _kind_rows(columns, kinds, checked=checked, help_of=help_of,
                      counts=insp.unique_counts(columns), disabled=disabled,
                      owner_of=(lambda c: owners.get(c, "")) if owners else None)


def cycle_for(insp: Inspection, kinds: Dict[str, str], columns: Sequence[str]):
    from .wizard import _cycler

    return _cycler(kinds, insp, columns, counts=insp.unique_counts(columns))


def _id_like(name: str) -> bool:
    n = name.strip().casefold()
    return (n in ("id", "pid", "sid", "uid", "participant", "subject", "key")
            or n.endswith("_id") or (n.endswith("id") and len(n) <= 6)
            or "participant" in n or "subject" in n)


def _stems(table: str) -> List[str]:
    """What a column naming this table might be called before ``_id``:
    ``users`` -> users, user; ``smoke_users`` -> smoke_users, smoke_user,
    users, user."""
    out: List[str] = []
    for base in (table, table.rpartition("_")[2]):
        for stem in (base, base[:-1] if base.endswith("s") else base):
            if stem and stem not in out:
                out.append(stem)
    return out


def _link(state: JoinState, right: db.TableInfo
          ) -> Optional[Tuple[str, str, str, str]]:
    """
    How a new table hangs off the tables already in play: ``(left table,
    left column, right column, why)``. A link the database declares, in
    either direction; then ``<table>_id`` against the other table's key;
    then a same-named column that is the key of one side (not both: two
    tables each with their own ``id`` share nothing). None when nothing fits.
    """
    in_play = [state.base] + [state.table(j["table"]) for j in state.joins]
    for t in in_play:
        for col, ref_table, ref_col in t.foreign_keys:
            if ref_table == right.name:
                return t.name, col, ref_col, "the database declares it"
        for col, ref_table, ref_col in right.foreign_keys:
            if ref_table == t.name:
                return t.name, ref_col, col, "the database declares it"
    right_cols = {c.lower(): c for c in right.column_names}
    right_key = right.primary_key[0] if len(right.primary_key) == 1 else right_cols.get("id")
    for t in in_play:
        if right_key:
            for col in t.column_names:
                if col.lower() in {f"{stem}_id".lower() for stem in _stems(right.name)}:
                    return t.name, col, right_key, f"{col} names {right.name}"
        t_cols = {c.lower(): c for c in t.column_names}
        t_key = t.primary_key[0] if len(t.primary_key) == 1 else t_cols.get("id")
        if t_key:
            for col in right.column_names:
                if col.lower() in {f"{stem}_id".lower() for stem in _stems(t.name)}:
                    return t.name, t_key, col, f"{col} names {t.name}"
    for t in in_play:
        for col in t.column_names:
            other = right_cols.get(col.lower())
            if other is None:
                continue
            here_key = tuple(t.primary_key) == (col,)
            there_key = tuple(right.primary_key) == (other,)
            if here_key != there_key or (not here_key and _id_like(col)):
                return t.name, col, other, "they share it"
    return None


def _via(state: JoinState, right: db.TableInfo, chosen: Sequence[str]
         ) -> Optional[db.TableInfo]:
    """A table nobody asked for that links the ones in play to ``right``:
    the bridge a database puts between posts and tags."""
    in_play = {state.base.name} | {j["table"] for j in state.joins}
    for t in state.tables:
        if t.name in in_play or t.name in chosen or t.rows == 0:
            continue
        trial = JoinState(picked=state.picked, base=state.base, tables=state.tables,
                          joins=list(state.joins))
        if _link(trial, t) is None:
            continue
        trial.joins.append({"table": t.name, "alias": "x", "on": [], "keys": [],
                            "right_keys": {}, "cols": []})
        if _link(trial, right) is not None:
            return t
    return None


def _ask_link(prompter: Prompter, state: JoinState, right: db.TableInfo
              ) -> Tuple[str, str, str]:
    """When the catalog has no idea: which column here, which column there."""
    in_play = [state.base] + [state.table(j["table"]) for j in state.joins]
    prompter.reason(f"Nothing in {state.picked.name} says how rows of {right.name} "
                    f"match rows of {state.base.name}: no declared link, no column "
                    f"they share by name. Pick the column on each side that holds "
                    f"the same ID.")
    left_rows = [Choice(f"{t.name}.{c.name}", qualified(t.name, c.name),
                        "the table's key" if c.name in t.primary_key else "",
                        annotation=c.type.lower())
                 for t in in_play for c in t.columns]
    left_pick = str(prompter.select(f"Which column matches rows to {right.name}?",
                                    left_rows))
    left_table, left_col = _split_qualified(left_pick, in_play)
    guess = next((c.name for c in right.columns if c.name.lower() == left_col.lower()),
                 next((c.name for c in right.columns if c.name.lower() == "id"), None))
    right_col = str(prompter.select(
        f"In {right.name}, which column holds {left_col}?",
        [Choice(c.name, c.name, "the table's key" if c.name in right.primary_key else "",
                annotation=c.type.lower()) for c in right.columns],
        default=guess))
    return left_table, left_col, right_col


def _alias_of(state: JoinState, table: str) -> str:
    if table == state.base.name:
        return "t0"
    return next(j["alias"] for j in state.joins if j["table"] == table)


def plan_joins(prompter: Prompter, picked: Picked, tables: Sequence[db.TableInfo],
               chosen: Chosen, *, base: str = "", allow_fan_out: bool,
               row_question: str = "Which table has one row per piece of text?"
               ) -> Tuple[JoinState, bool]:
    """
    The tables behind a column choice, joined. Returns the state and whether
    a question was asked on the way.

    The base table is the one the rows come from: the only table chosen,
    the only chosen table with long text, or else it is asked. Every other
    chosen table is joined on the link the catalog gives, through a bridge
    table when one is needed, or on columns the person names when nothing
    links them. Each join is read back with its counts before the next.
    """
    asked = False
    by_name = {t.name: t for t in tables}
    names = chosen.tables
    if base and base in names:
        base_name = base
    elif len(names) == 1:
        base_name = names[0]
    else:
        # the links the database declares point from the many side to the
        # one side (posts -> users): the table nothing else points at is
        # the one the rows come from. failing that, the one with the text
        referenced = {rt for n in names for _c, rt, _rc in by_name[n].foreign_keys
                      if rt in names}
        pointing = [n for n in names if n not in referenced
                    and any(rt in names for _c, rt, _rc in by_name[n].foreign_keys)]
        with_text = [n for n in names if any(c in by_name[n].text_columns
                                             for c in chosen.by_table[n])]
        if len(pointing) == 1:
            base_name = pointing[0]
        elif len(with_text) == 1:
            base_name = with_text[0]
        else:
            rows = [Choice(n, n, f"{', '.join(chosen.by_table[n][:4])}",
                           annotation=_rows_word(by_name[n])) for n in names]
            prompter.reason("The run makes one row of measures per row of this table; "
                            "the other tables' columns are matched onto it.")
            base_name = str(prompter.select(row_question, rows, default=base or None))
            asked = True
    state = JoinState(picked=picked, base=by_name[base_name], tables=list(tables),
                      chosen=chosen)
    for name in names:
        if name == base_name:
            continue
        right = by_name[name]
        link = _link(state, right)
        if link is None:
            bridge = _via(state, right, names)
            if bridge is not None:
                _add_join(prompter, state, bridge, _link(state, bridge), pass_through=True,
                          allow_fan_out=allow_fan_out)
                link = _link(state, right)
        if link is None:
            link = (*_ask_link(prompter, state, right), "as you said")
            asked = True
        if _add_join(prompter, state, right, link, allow_fan_out=allow_fan_out):
            asked = True
    after = inspection_for(prompter, state)
    prompter.note(f"  Working table: {after.scanned:,} rows — "
                  + ", ".join(_column_counts(state)) + ".", style="cyan")
    return state, asked


def _add_join(prompter: Prompter, state: JoinState, right: db.TableInfo,
              link: Tuple[str, str, str, str], *, allow_fan_out: bool,
              pass_through: bool = False) -> bool:
    """One table joined on, read back, and (when rows would repeat) asked
    about. Returns whether a question was asked."""
    left_table, left_col, right_col, why = link
    alias = f"t{len(state.joins) + 1}"
    state.joins.append({"table": right.name, "alias": alias,
                        "on": [[f"{_alias_of(state, left_table)}.{left_col}", right_col]],
                        "keys": [left_col], "right_keys": {left_col: right_col},
                        "cols": [] if pass_through else None})
    spec = state.source()
    key = str(spec.to_dict())
    if key not in state.reports:
        with prompter.scanning(f"Matching {left_table} against {right.name}", unit="checks"):
            state.reports[key] = sources.join_report(spec, state.picked.session)
    report = state.reports[key][-1]
    prompter.note(f"  {right.name} joined on {qualified(left_table, left_col)} = "
                  f"{qualified(right.name, right_col)} ({why})"
                  + (", bringing no columns of its own" if pass_through else "") + ".",
                  style="cyan")
    for line in report.sentences():
        prompter.note(f"  {line}", style="cyan")
    asked = False
    if report.relationship == "many-to-many" or not report.matched:
        state.joins.pop()
        state.reports.pop(key, None)
        blocked = ("The tables can't be joined as they are. What now?"
                   if report.relationship == "many-to-many" else "Nothing matched. What now?")
        answer = str(prompter.select(blocked, [
            Choice("columns", "Pick the columns again"),
            Choice("say", f"Say which columns match {state.base.name} to {right.name}"),
            Choice("cancel", "Cancel")]))
        if answer == "cancel":
            raise Cancelled()
        if answer == "columns":
            raise GoBack()
        return _add_join(prompter, state, right, (*_ask_link(prompter, state, right),
                                                  "as you said"),
                         allow_fan_out=allow_fan_out, pass_through=pass_through) or True
    if report.relationship == "one-to-many":
        left = report.left
        eg = "; ".join(f"{k} × {n}" for k, n in _fan_out_examples(report))
        prompter.note(
            f"  Some rows in {left.file} match several rows in {right.name} "
            f"(e.g. {eg}). Joining repeats those rows: {left.rows:,} rows become "
            f"{report.rows_out['left']:,}"
            + (", and each repeated row is measured once per match, so every "
               "statistic will count it more than once. That usually means "
               "picking a different key, or a summary table with one row per "
               f"{left.file}." if not allow_fan_out else "."),
            style="yellow")
        if not allow_fan_out:
            asked = True
            if not prompter.confirm("Repeat those rows and measure each copy anyway?",
                                    default=False):
                state.joins.pop()
                state.reports.pop(key, None)
                answer = str(prompter.select("What now?", [
                    Choice("columns", "Pick the columns again"),
                    Choice("say", f"Say which columns match {state.base.name} to "
                                  f"{right.name}"),
                    Choice("cancel", "Cancel")]))
                if answer == "cancel":
                    raise Cancelled()
                if answer == "columns":
                    raise GoBack()
                return _add_join(prompter, state, right,
                                 (*_ask_link(prompter, state, right), "as you said"),
                                 allow_fan_out=allow_fan_out, pass_through=pass_through) or True
    return asked


def _labels(state: JoinState) -> Dict[str, str]:
    """Which table each output column came from, for the pickers to show
    beside the name: every column once there is more than one table, so the
    column of table names reads as one; nothing when there is only one."""
    if not state.joins:
        return {}
    out: Dict[str, str] = {}
    for name, alias, col in state.output_columns():
        out[name] = state.base.name if alias == "t0" else next(
            j["table"] for j in state.joins if j["alias"] == alias)
    return out


def _column_counts(state: JoinState) -> List[str]:
    cols = state.output_columns()
    parts = [f"{state.base.name}' {sum(1 for _n, a, _c in cols if a == 't0')} columns"]
    for j in state.joins:
        n = sum(1 for _n, a, _c in cols if a == j["alias"])
        if n or j.get("cols") is None:
            parts.append(f"{n} from {j['table']}")
    return parts


def _fan_out_examples(report: sources.MergeReport) -> List[Tuple[str, int]]:
    return list(report.fan_out_examples) or [(k, 2) for k in report.right.key_samples[:2]]
