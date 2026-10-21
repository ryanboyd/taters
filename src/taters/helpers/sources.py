"""
A database source: one table, or a table with others joined on, as a thing
a preset can hold and the wizard can read back before anything is run.

Not SQL text. A :class:`Source` is a base table plus zero or more joins on
named keys, with the output columns named, and it is rendered into SQL for
whichever engine the connection speaks -- from quoted identifiers only,
every one checked against the catalog, so there is nothing to inject and a
column that has been dropped since the pipeline was made is found before a
step runs. Saved in YAML it reads like this::

    source:
      from: {table: posts, as: p}
      join:
        - {table: users, as: u, how: left, on: [[p.user_id, u.id]]}
      columns: {id: p.id, body: p.body, age: u.age}
      order_by: [p.id]

A raw ``query:`` is the escape hatch for people who have SQL and know it is
right; it is wrapped as a subquery and never edited.

:func:`join_report` is the read-back: for each join, the same counts the
spreadsheet merge computes (matched, unmatched with examples, keys repeating
on both sides, the rows the join would produce), from ``COUNT`` queries on
the server, poured into the merge's own :class:`MergeReport` so the
sentences a person sees are the ones they already know.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .feature_columns import SEPARATOR
from .reshape import ColumnClash, MergeReport, SideSummary

__all__ = ["Join", "Source", "coerce", "render", "resolve_columns",
           "join_report", "guarded", "query_problem", "HOW"]

HOW = ("left", "inner")

_ALIAS = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Join:
    table: str
    alias: str
    how: str = "left"
    #: Pairs of ``alias.column`` on the left (anything already joined) and
    #: ``column`` or ``alias.column`` on this table.
    on: Tuple[Tuple[str, str], ...] = ()
    allow_many_to_many: bool = False
    schema: str = ""

    def to_dict(self) -> dict:
        d: dict = {"table": self.table, "as": self.alias, "how": self.how,
                   "on": [list(p) for p in self.on]}
        if self.allow_many_to_many:
            d["allow_many_to_many"] = True
        if self.schema:
            d["schema"] = self.schema
        return d


@dataclass(frozen=True)
class Source:
    connection: str
    table: str = ""
    alias: str = ""
    joins: Tuple[Join, ...] = ()
    #: Output name -> ``alias.column``. Empty means every column of every
    #: table, base first, a name that repeats prefixed ``<table>__``.
    columns: Dict[str, str] = field(default_factory=dict)
    order_by: Tuple[str, ...] = ()
    query: str = ""
    schema: str = ""

    def __post_init__(self):
        if not self.connection:
            raise ValueError("a source needs a connection name")
        if not self.table and not self.query:
            raise ValueError("a source is a table (from:) or a query:")
        if self.query:
            problem = query_problem(self.query)
            if problem:
                raise ValueError(problem)
        if self.table and not self.alias:
            object.__setattr__(self, "alias", "t0")
        for j in self.joins:
            if j.how not in HOW:
                raise ValueError(f"a join is 'left' or 'inner', not {j.how!r}")
        for a in [self.alias, *(j.alias for j in self.joins)]:
            if a and not _ALIAS.match(a):
                raise ValueError(f"an alias is letters, digits and _: {a!r}")

    @property
    def tables(self) -> List[Tuple[str, str]]:
        """``(alias, table)`` for the base and every join, in order."""
        return [(self.alias, self.table)] + [(j.alias, j.table) for j in self.joins]

    def describe(self) -> str:
        """``posts LEFT JOIN users LEFT JOIN outcomes`` (or ``a query``)."""
        if self.query:
            return "a query"
        return " ".join([self.table] + [f"{j.how.upper()} JOIN {j.table}"
                                        for j in self.joins])

    def to_dict(self) -> dict:
        if self.query:
            d: dict = {"connection": self.connection, "query": self.query}
        else:
            d = {"connection": self.connection,
                 "from": {"table": self.table, "as": self.alias}}
            if self.schema:
                d["from"]["schema"] = self.schema
            if self.joins:
                d["join"] = [j.to_dict() for j in self.joins]
        if self.columns:
            d["columns"] = dict(self.columns)
        if self.order_by:
            d["order_by"] = list(self.order_by)
        return d

    @classmethod
    def from_dict(cls, doc: Mapping, *, connection: str = "") -> "Source":
        base = doc.get("from") or {}
        if isinstance(base, str):
            base = {"table": base}
        joins = []
        for i, j in enumerate(doc.get("join") or []):
            on = tuple((str(a), str(b)) for a, b in (j.get("on") or []))
            joins.append(Join(table=str(j["table"]), alias=str(j.get("as") or f"t{i + 1}"),
                              how=str(j.get("how") or "left"), on=on,
                              allow_many_to_many=bool(j.get("allow_many_to_many")),
                              schema=str(j.get("schema") or "")))
        return cls(connection=str(doc.get("connection") or connection),
                   table=str(base.get("table") or ""), alias=str(base.get("as") or ""),
                   joins=tuple(joins),
                   columns={str(k): str(v) for k, v in (doc.get("columns") or {}).items()},
                   order_by=tuple(str(c) for c in (doc.get("order_by") or [])),
                   query=str(doc.get("query") or ""),
                   schema=str(base.get("schema") or doc.get("schema") or ""))


def coerce(value: Any, *, connection: str = "") -> Source:
    """A :class:`Source` from a spec, a mapping, a ``db://`` string or a table name."""
    from .tables import DB_SCHEME, parse_ref

    if isinstance(value, Source):
        return value
    if isinstance(value, Mapping):
        return Source.from_dict(value, connection=connection)
    text = str(value).strip()
    if text.startswith(DB_SCHEME) or text.startswith("db:/"):
        ref = parse_ref(text)
        return Source(connection=ref.profile, table=ref.logical)
    if not connection:
        raise ValueError(f"{text!r} needs a connection name")
    return Source(connection=connection, table=text)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _split(ref: str, default_alias: str) -> Tuple[str, str]:
    alias, dot, col = ref.partition(".")
    return (alias, col) if dot else (default_alias, ref)


def resolve_columns(spec: Source, session) -> List[Tuple[str, str, str]]:
    """
    ``(output name, alias, column)`` for every output column, checked
    against the catalog. With no ``columns`` mapping: every column of every
    table, base first; a name present in several tables is kept bare for
    the first and prefixed ``<table>__`` after that, merge's rule.
    """
    if spec.query:
        with session.read_only():
            return [(c, "", c) for c in session.columns_of_query(spec.query)]
    known: Dict[str, List[str]] = {}
    for alias, table in spec.tables:
        try:
            info = session.d.describe(session, table, schema=spec.schema)
        except Exception:
            raise ValueError(f"{spec.connection} has no table called {table}") from None
        known[alias] = info.column_names
    out: List[Tuple[str, str, str]] = []
    if spec.columns:
        for name, ref in spec.columns.items():
            alias, col = _split(ref, spec.alias)
            if alias not in known:
                raise ValueError(f"{ref!r}: no table is called {alias!r} in this source")
            if col not in known[alias]:
                raise ValueError(f"{dict(spec.tables)[alias]} no longer has a column "
                                 f"called {col}")
            out.append((name, alias, col))
        return out
    # a join's right-hand key repeats the left one row for row, so it is
    # matched, not carried -- the same rule the spreadsheet merge follows
    matched = {(j.alias, _split(b, j.alias)[1]) for j in spec.joins for _a, b in j.on}
    taken: Dict[str, str] = {}
    for alias, table in spec.tables:
        for col in known[alias]:
            if (alias, col) in matched:
                continue
            name = col
            if name in taken:
                name = f"{table}{SEPARATOR}{col}"
                k = 2
                while name in taken:
                    name = f"{table}{SEPARATOR}{col}_{k}"
                    k += 1
            taken[name] = alias
            out.append((name, alias, col))
    return out


def _from_clause(spec: Source, d, *, upto: Optional[int] = None) -> str:
    """``FROM base AS p LEFT JOIN users AS u ON ...`` for the first ``upto`` joins."""
    q = d.quote
    parts = [f"{d.qualified(spec.table, spec.schema)} AS {q(spec.alias)}"]
    joins = spec.joins if upto is None else spec.joins[:upto]
    for j in joins:
        conds = []
        for left, right in j.on:
            la, lc = _split(left, spec.alias)
            ra, rc = _split(right, j.alias)
            conds.append(f"{q(la)}.{q(lc)} = {q(ra)}.{q(rc)}")
        parts.append(f"{j.how.upper()} JOIN {d.qualified(j.table, j.schema or spec.schema)} "
                     f"AS {q(j.alias)} ON {' AND '.join(conds) or '1=1'}")
    return "FROM " + " ".join(parts)


def render(spec: Source, session, *, columns: Optional[Sequence[str]] = None,
           order_by_exprs: Optional[Sequence[str]] = None
           ) -> Tuple[str, Tuple[Any, ...]]:
    """
    The SELECT for a source, from quoted identifiers only.

    ``columns`` narrows the output to those names (in that order).
    ``order_by_exprs`` replaces the spec's own ordering with ready SQL
    expressions (the gather uses it to sort by its group keys). The result
    carries no parameters today, but the pair is the shape every caller
    handles, so a WHERE pushed down later changes nothing upstream.
    """
    d = session.d
    if spec.query:
        sql = f"SELECT * FROM ({spec.query}) AS q"
        if columns:
            sql = f"SELECT {', '.join(d.quote(c) for c in columns)} FROM ({spec.query}) AS q"
        return sql, ()
    resolved = resolve_columns(spec, session)
    if columns:
        by_name = {n: (a, c) for n, a, c in resolved}
        missing = [c for c in columns if c not in by_name]
        if missing:
            raise ValueError(f"the source has no column {missing[0]!r}")
        resolved = [(n, *by_name[n]) for n in columns]
    selects = ", ".join(f"{d.quote(a)}.{d.quote(c)} AS {d.quote(n)}" for n, a, c in resolved)
    sql = f"SELECT {selects} {_from_clause(spec, d)}"
    if order_by_exprs:
        sql += " ORDER BY " + ", ".join(order_by_exprs)
    elif spec.order_by:
        keys = []
        for ref in spec.order_by:
            a, c = _split(ref, spec.alias)
            keys.append(f"{d.quote(a)}.{d.quote(c)}")
        sql += " ORDER BY " + ", ".join(keys)
    return sql, ()


def expression_of(spec: Source, session, name: str) -> str:
    """The quoted ``"alias"."column"`` behind an output column name."""
    d = session.d
    for n, a, c in resolve_columns(spec, session):
        if n == name:
            return f"{d.quote(a)}.{d.quote(c)}" if a else d.quote(c)
    raise ValueError(f"the source has no column {name!r}")


# ---------------------------------------------------------------------------
# The read-back
# ---------------------------------------------------------------------------

def _key_exprs(d, alias: str, cols: Sequence[str]) -> List[str]:
    return [f"{d.quote(alias)}.{d.quote(c)}" for c in cols]


def _blank_any(d, exprs: Sequence[str]) -> str:
    return "(" + " OR ".join(d.blank(e) for e in exprs) + ")"


def join_report(spec: Source, session, *, upto: Optional[int] = None
                ) -> List[MergeReport]:
    """
    One :class:`MergeReport` per join: the left side is the base table with
    the earlier joins applied, the right side is the table being added.
    """
    if spec.query:
        return []
    d = session.d
    reports: List[MergeReport] = []
    joins = spec.joins if upto is None else spec.joins[:upto]
    for k, j in enumerate(joins):
        left_from = _from_clause(spec, d, upto=k)
        right_from = f"FROM {d.qualified(j.table, j.schema or spec.schema)} AS {d.quote(j.alias)}"
        left_refs = [_split(a, spec.alias) for a, _b in j.on]
        right_refs = [_split(b, j.alias) for _a, b in j.on]
        # the left key may sit on the base or on an earlier join; each pair
        # names its own alias, and the side summary needs one alias to count
        # under, so the counts are expressed through the full left FROM
        left_keys = [f"{a}.{c}" for a, c in left_refs]
        right_keys = [c for _a, c in right_refs]
        lk = [f"{d.quote(a)}.{d.quote(c)}" for a, c in left_refs]
        rk = [f"{d.quote(j.alias)}.{d.quote(c)}" for c in right_keys]
        match = " AND ".join(f"{a} = {b}" for a, b in zip(lk, rk))
        left = _side_any(session, label=_left_label(spec, k), sql_from=left_from,
                         mine=lk, other_from=right_from, theirs=rk, keys=left_keys)
        right = _side_any(session, label=j.table, sql_from=right_from, mine=rk,
                          other_from=left_from, theirs=lk, keys=right_keys)
        inner = int(session.one(f"SELECT COUNT(*) {left_from} INNER JOIN "
                                f"{d.qualified(j.table, j.schema or spec.schema)} AS "
                                f"{d.quote(j.alias)} ON {match}") or 0)
        rows_out = {"inner": inner, "left": inner + (left.rows - left.rows_matched)}
        rows_out["outer"] = rows_out["left"] + (right.rows - right.rows_matched)
        m2m = _m2m(session, d, lk, rk, left_from, right_from, left_refs, right_keys)
        left_multi = _multi(session, d, lk, left_from, rk, right_from)
        right_multi = _multi(session, d, rk, right_from, lk, left_from)
        fan_out: List[Tuple[str, int]] = []
        if right_multi:
            rkeys = ", ".join(rk)
            for row in session.all(d.limit(
                    f"SELECT {rkeys}, COUNT(*) AS n {right_from} WHERE NOT "
                    f"{_blank_any(d, rk)} AND EXISTS (SELECT 1 {left_from} WHERE {match}) "
                    f"GROUP BY {rkeys} HAVING COUNT(*) > 1 ORDER BY n DESC", 3)):
                *key, n = row
                fan_out.append((" | ".join(str(v) for v in key), int(n)))
        if m2m["count"]:
            relationship = "many-to-many"
        elif right_multi:
            relationship = "one-to-many"
        elif left_multi:
            relationship = "many-to-one"
        else:
            relationship = "one-to-one"
        clashes, columns_out = _clashes(session, spec, k, j, match, left_from)
        reports.append(MergeReport(
            left=left, right=right, on=tuple(c for _a, c in left_refs),
            right_on=tuple(right_keys),
            key_match="exact", relationship=relationship,
            many_to_many_keys=m2m["count"], many_to_many_examples=tuple(m2m["examples"]),
            rows_out=rows_out, loose_extra_matches=0, loose_examples=(),
            clashes=clashes, columns_out=columns_out, what="table",
            fan_out_examples=tuple(fan_out)))
    return reports


def query_problem(query: str) -> str:
    """Why a typed query cannot be a source, or "" when it reads as one
    SELECT. The session it runs on is made read-only as well (see
    :func:`guarded`); this is the cheap first check that says so in words."""
    text = _strip_comments(str(query)).strip()
    if ";" in text:
        return "a query source is one SELECT statement; ';' is not allowed in it"
    if not re.match(r"(?is)^\s*(select|with)\b", text):
        return "a query source has to start with SELECT (or WITH); it is read-only"
    return ""


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"--[^\n]*", " ", text)


def guarded(spec: Source, session):
    """The context a source's SQL runs in: read-only when the source is a
    typed query, nothing special when it is tables Taters rendered itself."""
    import contextlib

    return session.read_only() if spec.query else contextlib.nullcontext()


def _left_label(spec: Source, k: int) -> str:
    return spec.table if k == 0 else spec.table + " + " + " + ".join(
        j.table for j in spec.joins[:k])


def _side_any(session, *, label, sql_from, mine, other_from, theirs, keys) -> SideSummary:
    """Like :func:`_side`, with the key expressions already quoted."""
    d = session.d
    blank = _blank_any(d, mine)
    match = " AND ".join(f"{m} = {t}" for m, t in zip(mine, theirs))
    exists = f"EXISTS (SELECT 1 {other_from} WHERE {match})"
    one = session.one
    keylist = ", ".join(mine)
    rows = one(f"SELECT COUNT(*) {sql_from}")
    blanks = one(f"SELECT COUNT(*) {sql_from} WHERE {blank}")
    distinct = one(f"SELECT COUNT(*) FROM (SELECT {keylist} {sql_from} WHERE NOT {blank} "
                   f"GROUP BY {keylist}) AS g")
    duplicated = one(f"SELECT COUNT(*) FROM (SELECT {keylist} {sql_from} WHERE NOT {blank} "
                     f"GROUP BY {keylist} HAVING COUNT(*) > 1) AS g")
    matched = one(f"SELECT COUNT(*) {sql_from} WHERE NOT {blank} AND {exists}")
    unmatched = one(f"SELECT COUNT(*) {sql_from} WHERE NOT {blank} AND NOT {exists}")
    unmatched_keys = one(f"SELECT COUNT(*) FROM (SELECT {keylist} {sql_from} WHERE NOT "
                         f"{blank} AND NOT {exists} GROUP BY {keylist}) AS g")
    examples = [" | ".join(str(v) for v in r) for r in session.all(
        d.limit(f"SELECT {keylist} {sql_from} WHERE NOT {blank} AND NOT {exists} "
                f"GROUP BY {keylist}", 5))]
    samples = [" | ".join(str(v) for v in r) for r in session.all(
        d.limit(f"SELECT {keylist} {sql_from} WHERE NOT {blank}", 3))]
    return SideSummary(
        file=label, encoding="", delimiter="", rows=int(rows or 0), keys=tuple(keys),
        renamed_headers=(), ragged_short=0, ragged_long=0,
        blank_keys=int(blanks or 0), distinct_keys=int(distinct or 0),
        duplicated_keys=int(duplicated or 0), rows_matched=int(matched or 0),
        rows_unmatched=int(unmatched or 0), unmatched_keys=int(unmatched_keys or 0),
        unmatched_examples=tuple(examples), loose_collisions=0,
        loose_collision_examples=(), mangled_ids=0, key_samples=tuple(samples))


def _multi(session, d, mine, sql_from, theirs, other_from) -> int:
    """How many keys on this side appear on several rows *and* match the other."""
    blank = _blank_any(d, mine)
    match = " AND ".join(f"{m} = {t}" for m, t in zip(mine, theirs))
    keylist = ", ".join(mine)
    return int(session.one(
        f"SELECT COUNT(*) FROM (SELECT {keylist} {sql_from} WHERE NOT {blank} AND EXISTS "
        f"(SELECT 1 {other_from} WHERE {match}) GROUP BY {keylist} HAVING COUNT(*) > 1) "
        f"AS g") or 0)


def _m2m(session, d, lk, rk, left_from, right_from, left_refs, right_keys) -> dict:
    """Keys repeating on both sides: how many, and the three worst."""
    lkeys, rkeys = ", ".join(lk), ", ".join(rk)
    la = [f"a.{d.quote(c)}" for _al, c in left_refs]
    rb = [f"b.{d.quote(c)}" for c in right_keys]
    on = " AND ".join(f"{x} = {y}" for x, y in zip(la, rb))
    sql = (f"SELECT {', '.join(la)}, a.n AS na, b.n AS nb FROM (SELECT {lkeys}, COUNT(*) AS n {left_from} "
           f"WHERE NOT {_blank_any(d, lk)} GROUP BY {lkeys} HAVING COUNT(*) > 1) AS a JOIN "
           f"(SELECT {rkeys}, COUNT(*) AS n {right_from} WHERE NOT {_blank_any(d, rk)} "
           f"GROUP BY {rkeys} HAVING COUNT(*) > 1) AS b ON {on}")
    count = int(session.one(f"SELECT COUNT(*) FROM ({sql}) AS m") or 0)
    examples = []
    if count:
        for row in session.all(d.limit(sql + " ORDER BY na * nb DESC", 3)):
            *key, nl, nr = row
            examples.append((" | ".join(str(v) for v in key), int(nl), int(nr)))
    return {"count": count, "examples": examples}


def _clashes(session, spec: Source, k: int, j: Join, match: str, left_from: str
             ) -> Tuple[Tuple[ColumnClash, ...], Tuple[str, ...]]:
    """Same-named non-key columns on both sides, and whether they agree."""
    d = session.d
    left_cols: List[Tuple[str, str]] = []           # (name, alias)
    for alias, table in spec.tables[:k + 1]:
        info = session.d.describe(session, table, schema=spec.schema)
        for c in info.column_names:
            if c not in [n for n, _a in left_cols]:
                left_cols.append((c, alias))
    right_info = session.d.describe(session, j.table, schema=j.schema or spec.schema)
    right_key_cols = {_split(b, j.alias)[1] for _a, b in j.on}
    clashes: List[ColumnClash] = []
    names_out = [n for n, _a in left_cols]
    right_from = f"{d.qualified(j.table, j.schema or spec.schema)} AS {d.quote(j.alias)}"
    for c in right_info.column_names:
        if c in right_key_cols:
            continue
        owner = next((a for n, a in left_cols if n == c), None)
        if owner is None:
            names_out.append(c)
            continue
        l, r = f"{d.quote(owner)}.{d.quote(c)}", f"{d.quote(j.alias)}.{d.quote(c)}"
        n = int(session.one(
            f"SELECT COUNT(*) {left_from} INNER JOIN {right_from} ON {match} WHERE NOT "
            f"{d.blank(l)} AND NOT {d.blank(r)} AND {d.as_text(l)} <> {d.as_text(r)}") or 0)
        example = None
        if n:
            keys = ", ".join(f"{d.quote(j.alias)}.{d.quote(c)}" for c in sorted(right_key_cols))
            row = session.all(d.limit(
                f"SELECT {l}, {r}, {keys} {left_from} INNER JOIN {right_from} ON {match} "
                f"WHERE NOT {d.blank(l)} AND NOT {d.blank(r)} AND {d.as_text(l)} <> "
                f"{d.as_text(r)}", 1))
            if row:
                example = (" | ".join(str(v) for v in row[0][2:]), str(row[0][0]),
                           str(row[0][1]))
            new = f"{j.table}{SEPARATOR}{c}"
            names_out.append(new)
            clashes.append(ColumnClash(c, False, n, example, (c, new)))
        else:
            clashes.append(ColumnClash(c, True, 0, None, (c,)))
    return tuple(clashes), tuple(names_out)
