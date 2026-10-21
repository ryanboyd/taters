"""
The column questions, answered by a database instead of a scan.

Every screen that asks about columns -- which look like numbers, which could
be labels, how many groups a column makes, how many rows a filter keeps --
asks an :class:`~taters.ui.columns.Inspection`. This is one whose answers
come from ``COUNT`` queries on the server rather than from a pass over a
file, built from a :class:`~taters.helpers.sources.Source` (a table, or a
table with others joined on). The screens cannot tell the difference, which
is the point: grouping, row filters, the kind arrows and the controls offer
all work on a database without a line of them changing.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..helpers import sources as _sources
from ..helpers.row_filter import FILTER_OPS
from .columns import MAX_LABELS, UNIQUE_CEILING, ColumnStats, Grain, Inspection

__all__ = ["DbInspection", "inspect_source"]


class _Guarded:
    """A session whose reads run inside ``read_only()`` when asked to."""

    def __init__(self, session, *, read_only: bool):
        self._session = session
        self._read_only = read_only

    def __getattr__(self, name):
        return getattr(self._session, name)

    def _call(self, name, *args, **kwargs):
        if not self._read_only:
            return getattr(self._session, name)(*args, **kwargs)
        with self._session.read_only():
            return getattr(self._session, name)(*args, **kwargs)

    def all(self, *a, **k):
        return self._call("all", *a, **k)

    def one(self, *a, **k):
        return self._call("one", *a, **k)

    def stream(self, *a, **k):
        if not self._read_only:
            yield from self._session.stream(*a, **k)
            return
        with self._session.read_only():
            yield from self._session.stream(*a, **k)


class DbInspection(Inspection):
    """An Inspection whose questions go to the database."""

    def __init__(self, source: _sources.Source, session, *,
                 limit: int = 0, on_rows: Optional[Callable[[int], None]] = None):
        self.source = source
        # a typed query (the YAML escape hatch) gets the rights a SELECT
        # needs and no more, on every call the inspector makes
        self.session = _Guarded(session, read_only=bool(source.query))
        self.limit = int(limit or 0)
        resolved = _sources.resolve_columns(source, session)
        self._expr: Dict[str, str] = {}
        d = session.d
        for name, alias, col in resolved:
            self._expr[name] = (f"{d.quote(alias)}.{d.quote(col)}" if alias
                                else d.quote(col))
        names = [n for n, _a, _c in resolved]
        sql, params = _sources.render(source, session)
        self._sql, self._params = sql, params
        if source.query:
            self._from = f"FROM ({source.query}) AS q"
        else:
            self._from = _sources._from_clause(source, d)
        scanned = int(session.one(f"SELECT COUNT(*) FROM ({sql}) AS q", params) or 0)
        stats = {}
        for i, name in enumerate(names):
            stats[name] = self._stats(name)
            if on_rows is not None:
                on_rows(i + 1)
        super().__init__(columns=names, rows=[], scanned=scanned, stats=stats,
                         path=None, delimiter=None)

    # ---- one column's record -----------------------------------------

    def _where(self) -> str:
        return ""

    def _stats(self, name: str) -> ColumnStats:
        d, s = self.session.d, self.session
        e = self._expr[name]
        filled = int(s.one(f"SELECT COUNT(*) {self._from} WHERE NOT {d.blank(e)}") or 0)
        if not filled:
            return ColumnStats(filled=0, all_numeric=True, distinct=[], counts={},
                               unique=0, unique_over=False)
        non_numeric = int(s.one(f"SELECT COUNT(*) {self._from} WHERE NOT {d.blank(e)} "
                                f"AND NOT {d.numeric(e)}") or 0)
        unique = int(s.one(f"SELECT COUNT(DISTINCT {d.as_text(e)}) {self._from} WHERE NOT "
                           f"{d.blank(e)}") or 0)
        counts: Optional[Dict[str, int]] = None
        distinct: List[str] = []
        if unique <= MAX_LABELS + 1:
            rows = s.all(f"SELECT {d.as_text(e)}, COUNT(*) {self._from} WHERE NOT "
                         f"{d.blank(e)} GROUP BY {d.as_text(e)} ORDER BY COUNT(*) DESC")
            counts = {str(v).strip(): int(n) for v, n in rows}
            distinct = list(counts)
        else:
            distinct = [str(v).strip() for (v,) in s.all(d.limit(
                f"SELECT DISTINCT {d.as_text(e)} {self._from} WHERE NOT {d.blank(e)}",
                MAX_LABELS + 1))]
        return ColumnStats(filled=filled, all_numeric=non_numeric == 0,
                           distinct=distinct, counts=counts,
                           unique=min(unique, UNIQUE_CEILING + 1),
                           unique_over=unique > UNIQUE_CEILING)

    # ---- the questions that need more than one column ---------------------

    def grain(self, keys: Sequence[str], *, track: Sequence[str] = (),
              on_rows: Optional[Callable[[int], None]] = None) -> Grain:
        keys = tuple(keys)
        wanted = frozenset(track) - set(keys)
        have = self._grains.get(keys)
        if have is not None and wanted <= have.tracked:
            return have
        d, s = self.session.d, self.session
        kexprs = [self._expr[k] for k in keys]
        klist = ", ".join(kexprs)
        # the server does the counting of the counts: a row per group would
        # be half a million tuples for an id-like key
        agg = s.all(f"SELECT COUNT(*), COALESCE(SUM(n), 0), COALESCE(MIN(n), 0), "
                    f"COALESCE(MAX(n), 0) FROM (SELECT COUNT(*) AS n {self._from} "
                    f"GROUP BY {klist}) AS g")
        n_groups, n_rows, smallest, largest = [int(v or 0) for v in agg[0]]
        varies = set()
        for i, c in enumerate(sorted(wanted)):
            e = self._expr.get(c)
            if e is None:
                continue
            n = int(s.one(f"SELECT COUNT(*) FROM (SELECT {klist} {self._from} WHERE NOT "
                          f"{d.blank(e)} GROUP BY {klist} HAVING COUNT(DISTINCT "
                          f"{d.as_text(e)}) > 1) AS g") or 0)
            if n:
                varies.add(c)
            if on_rows is not None:
                on_rows(i + 1)
        found = Grain(keys=keys, groups=n_groups, rows=n_rows,
                      smallest=smallest, largest=largest,
                      tracked=wanted | (have.tracked if have else frozenset()),
                      varies=frozenset(varies | (have.varies if have else set())))
        self._grains[keys] = found
        return found

    def survivors(self, filters: Sequence[Sequence[object]], *,
                  on_rows: Optional[Callable[[int], None]] = None) -> int:
        if not filters:
            return self.scanned
        where, params = self.where_clause(filters)
        return int(self.session.one(f"SELECT COUNT(*) {self._from} WHERE {where}",
                                    params) or 0)

    def where_clause(self, filters: Sequence[Sequence[object]]
                     ) -> Tuple[str, Tuple[Any, ...]]:
        """``[column, op, value]`` triples as a WHERE, parameterized."""
        d = self.session.d
        parts: List[str] = []
        params: List[Any] = []
        for spec in filters:
            column, op, value = str(spec[0]), str(spec[1]), spec[2]
            if op not in FILTER_OPS:
                raise ValueError(f"unknown filter operator {op!r}")
            e = self._expr.get(column)
            if e is None:
                raise ValueError(f"no column called {column!r}")
            notblank = f"NOT {d.blank(e)}"
            if op in ("in", "not_in"):
                values = [str(v).strip() for v in (value or ())]
                if not values:
                    parts.append(notblank if op == "in" else "1=1")
                    continue
                marks = ", ".join("?" for _ in values)
                clause = f"TRIM({d.as_text(e)}) IN ({marks})"
                parts.append(f"({notblank} AND {clause if op == 'in' else 'NOT ' + clause})")
                params.extend(values)
            elif op in ("==", "!="):
                try:
                    number = float(str(value).strip())
                    clause = f"{d.numeric(e)} AND CAST({e} AS {d.float_type}) = ?"
                    params.append(number)
                except ValueError:
                    clause = f"TRIM({d.as_text(e)}) = ?"
                    params.append(str(value).strip())
                parts.append(f"({notblank} AND {'' if op == '==' else 'NOT '}({clause}))")
            else:
                number = float(str(value).strip())
                parts.append(f"({notblank} AND {d.numeric(e)} AND CAST({e} AS "
                             f"{d.float_type}) {op} ?)")
                params.append(number)
        return " AND ".join(parts) or "1=1", tuple(params)

    def distinct(self, column: str, limit: int = 12) -> List[str]:
        s = self.stats.get(column)
        if s is not None and (s.counts is not None or limit <= MAX_LABELS):
            return s.distinct[:limit + 1]
        d = self.session.d
        e = self._expr[column]
        return [str(v).strip() for (v,) in self.session.all(d.limit(
            f"SELECT DISTINCT {d.as_text(e)} {self._from} WHERE NOT {d.blank(e)}",
            limit + 1))]

    # ---- what the statistics stage needs ---------------------------------

    def column_tally(self, column: str, *,
                     on_rows: Optional[Callable[[int], None]] = None):
        """What the vetting checks need from one column, counted by the
        server: rows per value (the most common first, capped), how many
        cells are filled and numeric, and the values that are not numbers."""
        from .columns import TALLY_CAP, Tally

        d, s = self.session.d, self.session
        e = self._expr[column]
        t = Tally()
        for v, n in s.all(d.limit(
                f"SELECT {d.as_text(e)}, COUNT(*) AS n {self._from} WHERE NOT "
                f"{d.blank(e)} GROUP BY {d.as_text(e)} ORDER BY n DESC", TALLY_CAP + 1)):
            if len(t.counts) >= TALLY_CAP:
                t.truncated = True
                break
            t.counts[str(v).strip()] = int(n)
        if on_rows is not None:
            on_rows(1)
        t.filled = int(s.one(f"SELECT COUNT(*) {self._from} WHERE NOT {d.blank(e)}") or 0)
        t.numeric = int(s.one(f"SELECT COUNT(*) {self._from} WHERE NOT {d.blank(e)} AND "
                              f"{d.numeric(e)}") or 0)
        if t.numeric < t.filled:
            for v, n in s.all(d.limit(
                    f"SELECT {d.as_text(e)}, COUNT(*) AS n {self._from} WHERE NOT "
                    f"{d.blank(e)} AND NOT {d.numeric(e)} GROUP BY {d.as_text(e)} "
                    f"ORDER BY n DESC", 50)):
                t.bad[str(v).strip()] = int(n)
        if on_rows is not None:
            on_rows(2)
        return t

    def column_values(self, column: str, *,
                      on_rows: Optional[Callable[[int], None]] = None) -> List[str]:
        """Every value of one column, in the source's order, as text."""
        from ..helpers import db

        sql, params = _sources.render(self.source, self.session, columns=[column])
        out: List[str] = []
        for row in self.session.stream(sql, params):
            out.append(db.cell_text(row[0]))
            if on_rows is not None and len(out) % 2000 == 0:
                on_rows(len(out))
        if on_rows is not None:
            on_rows(len(out))
        return out

    def duplicate_keys(self, keys: Sequence[str]) -> Tuple[int, List[Tuple[str, int]]]:
        """How many key values repeat, and the three worst with their counts."""
        d, s = self.session.d, self.session
        klist = ", ".join(self._expr[k] for k in keys)
        n = int(s.one(f"SELECT COUNT(*) FROM (SELECT {klist} {self._from} GROUP BY {klist} "
                      f"HAVING COUNT(*) > 1) AS g") or 0)
        worst = []
        if n:
            for row in s.all(d.limit(f"SELECT {klist}, COUNT(*) AS n {self._from} GROUP BY "
                                     f"{klist} HAVING COUNT(*) > 1 ORDER BY n DESC", 3)):
                *key, count = row
                worst.append((" | ".join(str(v) for v in key), int(count)))
        return n, worst


def inspect_source(source: _sources.Source, session, *, limit: int = 0,
                   on_rows: Optional[Callable[[int], None]] = None) -> DbInspection:
    """A :class:`DbInspection` for a source, the way ``inspect_csv`` makes one."""
    return DbInspection(source, session, limit=limit, on_rows=on_rows)
