"""
A table is a table: one way to say where one lives, and one way to read or
write it, whether that is a CSV file or a table in a database.

Everything in Taters that produces or consumes a table used to take a
``Path``. A :class:`TableRef` is the one value that replaces it. For a file
it *is* path-like -- ``open()``, ``Path()`` and ``os.fspath()`` all work, so
nothing that still thinks in files has to change -- and for a database
table it is ``db://<connection>/<logical/path>``, which is not path-like on
purpose: a call site that has not learned about databases fails with a
``TypeError`` the moment it is handed one, instead of quietly creating a
folder called ``db:`` and writing a CSV into it.

The canonical string form goes everywhere a path string went: YAML presets,
``{{save_as}}`` artifacts, lists of feature tables, command-line flags. A
trailing ``.csv`` on a database address is dropped, so a template written
as ``{{var:features_dir}}/readability.csv`` renders either way.
"""
from __future__ import annotations

import csv
import re
import os
from contextlib import contextmanager
from pathlib import Path
from typing import (Tuple, Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Sequence, Union)

from .atomic import atomic_write
from .cliargs import CliSpec
from .csvio import widen_csv_field_limit
from .progress import Ticker, announce
from .provenance import records_settings

__all__ = [
    "TableRef", "parse_ref", "is_ref", "exists", "columns_of", "count",
    "modified_at", "read_rows", "read_str_table", "write_table",
    "read_sidecar", "write_sidecar", "delete_sidecar", "fingerprint",
    "require_local", "export_table", "import_table", "output_ref", "handle", "read_cells",
    "delete_table", "tables_under",
    "DB_SCHEME",
]

widen_csv_field_limit()

DB_SCHEME = "db://"
RefLike = Union[str, Path, "TableRef"]


class TableRef:
    """
    Where a table lives. A file path, or ``db://<connection>/<logical>``.

    Only the file kind is path-like. ``logical`` is the slash-separated
    name inside the database (``taters/study7/features/readability``); the
    physical table name is derived from it by the database layer.
    """

    __slots__ = ("_path", "profile", "logical")

    def __init__(self, path: Optional[Path] = None, *, profile: str = "",
                 logical: str = ""):
        if path is None and not (profile and logical):
            raise ValueError("a TableRef is a path, or a connection and a logical name")
        self._path = Path(path) if path is not None else None
        self.profile = profile
        self.logical = logical.strip("/")

    # ---- what kind ------------------------------------------------------

    @property
    def is_db(self) -> bool:
        return self._path is None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise TypeError(f"{self} is a database table, not a file")
        return self._path

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    # ---- names ----------------------------------------------------------

    @property
    def parts(self) -> List[str]:
        if self._path is not None:
            return list(self._path.parts)
        return self.logical.split("/")

    @property
    def stem(self) -> str:
        """The table's name: a file's stem, or the last logical segment."""
        if self._path is not None:
            return self._path.stem
        return self.parts[-1]

    @property
    def name(self) -> str:
        """What to call the table in a sentence: a file's name with its
        extension, a database table's logical name."""
        if self._path is not None:
            return self._path.name
        return self.parts[-1]

    @property
    def parent(self) -> str:
        if self._path is not None:
            return str(self._path.parent)
        return "/".join(self.parts[:-1])

    def sibling(self, stem: str, suffix: str = ".csv") -> "TableRef":
        """Another table beside this one: ``features/x.csv`` -> ``features/y.csv``."""
        if self._path is not None:
            return TableRef(self._path.with_name(stem + (suffix if suffix else "")))
        return TableRef(profile=self.profile, logical="/".join([*self.parts[:-1], stem]))

    def local_dir(self, work_dir: Optional[Path] = None) -> Path:
        """
        A folder on disk for things that are not rows: a model file, a
        figure, a report fragment beside this table. For a file it is the
        file's folder; for a database table it is ``db_local/<connection>/
        <logical parent>`` under the work folder.
        """
        if self._path is not None:
            return self._path.parent
        base = Path(work_dir) if work_dir is not None else Path.cwd()
        return base / "db_local" / self.profile / Path(*self.parts[:-1]) \
            if len(self.parts) > 1 else base / "db_local" / self.profile

    # ---- plumbing -------------------------------------------------------

    def __str__(self) -> str:
        if self._path is not None:
            return str(self._path)
        return f"{DB_SCHEME}{self.profile}/{self.logical}"

    def __repr__(self) -> str:
        return f"TableRef({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TableRef):
            return str(self) == str(other)
        if isinstance(other, (str, Path)):
            try:
                return str(self) == str(parse_ref(other))
            except (TypeError, ValueError):
                return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(str(self))

    def __reduce__(self):
        return (parse_ref, (str(self),))


def is_ref(value: Any) -> bool:
    """Whether ``value`` could name a table: a path-like or a db address."""
    return isinstance(value, (TableRef, Path)) or (
        isinstance(value, str) and bool(value.strip()))


def parse_ref(value: RefLike) -> TableRef:
    """
    A :class:`TableRef` from a path, a string, or a ref.

    Accepts the ``db:/connection/...`` spelling too, which is what a
    ``db://`` string becomes after one careless ``Path()`` on the way.
    """
    if isinstance(value, TableRef):
        return value
    if isinstance(value, Path):
        text = str(value)
        if not (text.startswith("db:/") or text.startswith("db:\\")):
            return TableRef(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise TypeError(f"not a table: {value!r}")
    if text.startswith(DB_SCHEME) or text.startswith("db:/") or text.startswith("db:\\"):
        rest = text[3:].replace("\\", "/").lstrip("/")
        profile, _, logical = rest.partition("/")
        if logical.lower().endswith(".csv"):
            logical = logical[:-4]
        if not profile or not logical:
            raise ValueError(f"a database table is db://<connection>/<name>, not {text!r}")
        # the logical name becomes a folder under the run folder for the
        # things that are not rows, so it cannot be allowed to climb out
        if not re.match(r"^[a-z][a-z0-9_-]*$", profile):
            raise ValueError(f"{profile!r} is not a connection name (lowercase letters, "
                             f"digits, _ or -)")
        parts = logical.split("/")
        if any(not seg or seg in (".", "..") for seg in parts):
            raise ValueError(f"a table's name cannot have empty or '..' parts: {text!r}")
        return TableRef(profile=profile, logical=logical)
    return TableRef(Path(text))


# ---------------------------------------------------------------------------
# Sessions, held once per connection name for the life of a process
# ---------------------------------------------------------------------------

_SESSIONS: Dict[Tuple[str, str], Any] = {}


def _session(ref: TableRef, role: str = "read"):
    """
    The open session for a ref's connection, opened on first use.

    Reads and writes get separate connections (``role``): a gather streams
    its source through an unbuffered cursor while the loader inserts the
    rows it produces, and MySQL refuses a second command on a connection
    whose cursor still has rows pending ("Commands out of sync").
    """
    from . import db

    s = _SESSIONS.get((ref.profile, role))
    if s is None:
        s = db.session_for(ref.profile)
        _SESSIONS[(ref.profile, role)] = s
    return s


def close_sessions() -> None:
    from . import db

    for s in _SESSIONS.values():
        try:
            s.close()
        except Exception:
            pass
    _SESSIONS.clear()
    db.close_tunnels()


def _entry(ref: TableRef):
    """The registry row for a database ref, or None when nothing is there.

    The physical name is a tidied version of the logical one, so two logical
    names can land on one table (``Posts`` and ``posts``); only the one the
    registry says it holds is it."""
    from . import db

    entry = db.registered(_session(ref), db.sanitize_table_name(ref.logical))
    return entry if entry is not None and entry.logical == ref.logical else None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def exists(ref: RefLike) -> bool:
    r = parse_ref(ref)
    if not r.is_db:
        return r.path.is_file()
    return _entry(r) is not None


def columns_of(ref: RefLike, *, encoding: str = "utf-8-sig",
               delimiter: str = ",") -> List[str]:
    r = parse_ref(ref)
    if r.is_db:
        entry = _entry(r)
        if entry is None:
            raise FileNotFoundError(f"{r} is not in the database")
        return entry.logical_columns
    with r.path.open("r", newline="", encoding=encoding) as fh:
        return [str(c) for c in (next(csv.reader(fh, delimiter=delimiter), None) or [])]


def count(ref: RefLike, *, encoding: str = "utf-8-sig",
          on_progress: Optional[Callable[..., None]] = None) -> int:
    """How many rows, by the cheapest honest means."""
    r = parse_ref(ref)
    if r.is_db:
        entry = _entry(r)
        return entry.n_rows if entry is not None else 0
    from .progress import count_rows

    return count_rows(r.path, encoding=encoding, on_progress=on_progress)


def modified_at(ref: RefLike) -> Optional[float]:
    """When the table was last written, as a timestamp; None when absent."""
    import datetime as _dt

    r = parse_ref(ref)
    if not r.is_db:
        try:
            return r.path.stat().st_mtime
        except OSError:
            return None
    entry = _entry(r)
    if entry is None or not entry.written_at:
        return None
    try:
        return _dt.datetime.fromisoformat(entry.written_at).timestamp()
    except ValueError:
        return None


def read_rows(ref: RefLike, *, columns: Optional[Sequence[str]] = None,
              encoding: str = "utf-8-sig", delimiter: str = ",",
              on_rows: Optional[Callable[[int], None]] = None
              ) -> Iterator[Dict[str, str]]:
    """Every row as a dict of text cells, in the table's own order."""
    r = parse_ref(ref)
    if r.is_db:
        from . import db

        entry = _entry(r)
        if entry is None:
            raise FileNotFoundError(f"{r} is not in the database")
        yield from db.read_rows(_session(r), entry, columns=columns, on_rows=on_rows)
        return
    n = 0
    with r.path.open("r", newline="", encoding=encoding) as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            n += 1
            if columns is not None:
                row = {c: row.get(c) or "" for c in columns}
            else:
                row = {k: (v or "") for k, v in row.items() if k is not None}
            yield row
            if on_rows is not None and n % 2000 == 0:
                on_rows(n)
    if on_rows is not None:
        on_rows(n)


def read_cells(ref: RefLike, *, columns: Optional[Sequence[str]] = None,
               encoding: str = "utf-8-sig", delimiter: str = ",",
               on_rows: Optional[Callable[[int], None]] = None
               ) -> Iterator[List[str]]:
    """
    Every row as a list of text cells, in ``columns`` order (the header's
    when none are named). The positional twin of :func:`read_rows`, for the
    matrix-shaped readers (a document-term matrix, a PCA) where a dict per
    row of five thousand cells is most of the work.
    """
    r = parse_ref(ref)
    if r.is_db:
        from . import db

        entry = _entry(r)
        if entry is None:
            raise FileNotFoundError(f"{r} is not in the database")
        yield from db.read_cells(_session(r), entry, columns=columns, on_rows=on_rows)
        return
    n = 0
    with r.path.open("r", newline="", encoding=encoding) as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = [str(c) for c in (next(reader, None) or [])]
        if columns is None:
            width = len(header)
            pick = None
        else:
            wanted = [str(c) for c in columns]
            missing = [c for c in wanted if c not in header]
            if missing:
                raise ValueError(f"{r} has no column {missing[0]!r}")
            pick = [header.index(c) for c in wanted]
        for row in reader:
            n += 1
            if pick is None:
                cells = [(v or "") for v in row]
                if len(cells) < width:
                    cells += [""] * (width - len(cells))
                yield cells[:width]
            else:
                yield [(row[i] if i < len(row) else "") or "" for i in pick]
            if on_rows is not None and n % 2000 == 0:
                on_rows(n)
    if on_rows is not None:
        on_rows(n)


def read_str_table(ref: RefLike, *, encoding: str = "utf-8-sig",
                   columns: Optional[Sequence[str]] = None):
    """The whole table as an all-string DataFrame (what the statistics
    read), or only ``columns`` of it, in the table's own order."""
    import pandas as pd

    r = parse_ref(ref)
    wanted = [str(c) for c in columns] if columns is not None else None
    if not r.is_db:
        return pd.read_csv(r.path, dtype=str, keep_default_na=False, encoding=encoding,
                           usecols=(lambda c: c in set(wanted)) if wanted is not None else None)
    cols = [c for c in columns_of(r) if wanted is None or c in set(wanted)]
    # in blocks, so the peak is one block plus the frame rather than every
    # row as a Python list beside the frame built from it
    frames = []
    block: List[List[str]] = []
    for cells in read_cells(r, columns=cols):
        block.append(cells)
        if len(block) >= 50_000:
            frames.append(pd.DataFrame(block, columns=cols, dtype=str))
            block = []
    if block:
        frames.append(pd.DataFrame(block, columns=cols, dtype=str))
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=str) for c in cols})
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class _Writer:
    """What :func:`write_table` yields: ``writerow`` for a dict or a sequence."""

    def __init__(self, fieldnames: Sequence[str]):
        self.fieldnames = [str(c) for c in fieldnames]
        self.rows = 0
        self.fingerprint: Optional[str] = None
        self._sink: Optional[Callable[[List[Any]], None]] = None

    def _cells(self, row: Any) -> List[Any]:
        if isinstance(row, Mapping):
            return [row.get(c, "") for c in self.fieldnames]
        cells = list(row)
        if len(cells) != len(self.fieldnames):
            raise ValueError(f"row has {len(cells)} cells, the header {len(self.fieldnames)}")
        return cells

    def writerow(self, row: Any) -> None:
        assert self._sink is not None
        self._sink(self._cells(row))
        self.rows += 1

    def writerows(self, rows: Any) -> None:
        for row in rows:
            self.writerow(row)


@contextmanager
def write_table(ref: RefLike, fieldnames: Sequence[str], *,
                encoding: str = "utf-8-sig",
                hints: Optional[Mapping[str, str]] = None,
                on_rows: Optional[Callable[[int], None]] = None,
                fingerprint: bool = False,
                unique_key: str = ""):
    """
    Open a table for writing; it appears only once complete.

    For a file this is exactly ``atomic_write`` plus ``csv.DictWriter`` --
    the bytes are what they always were. For a database table the rows are
    loaded, typed and swapped in by :func:`taters.helpers.db.write_rows`.
    ``hints`` names columns whose type is known (``"text"``, ``"int"``,
    ``"float"``). The writer's ``fingerprint`` is set on exit when asked
    for (a database table always has one; a file's costs a second pass over
    the cells, so it is opt-in).
    """
    r = parse_ref(ref)
    writer = _Writer(fieldnames)
    if r.is_db:
        from . import db

        loader = db.Loader(_session(r, "write"), r.logical, writer.fieldnames, hints=hints,
                           on_rows=on_rows, unique_key=unique_key)
        writer._sink = loader.add
        try:
            yield writer
        except BaseException:
            loader.abort()
            raise
        entry = loader.finish()
        writer.fingerprint = entry.fingerprint
        return
    from .db import Fingerprint

    fp = Fingerprint(writer.fieldnames) if fingerprint else None
    with atomic_write(r.path, "w", newline="", encoding=encoding) as fh:
        out = csv.DictWriter(fh, fieldnames=writer.fieldnames)
        out.writeheader()

        def sink(cells: List[Any]) -> None:
            out.writerow(dict(zip(writer.fieldnames, cells)))
            if fp is not None:
                fp.add("" if v is None else str(v) for v in cells)
            if on_rows is not None and writer.rows % 2000 == 1999:
                on_rows(writer.rows + 1)
        writer._sink = sink
        yield writer
    if fp is not None:
        writer.fingerprint = fp.hexdigest()
    if on_rows is not None:
        on_rows(writer.rows)


def fingerprint(ref: RefLike, *, encoding: str = "utf-8-sig",
                delimiter: str = ",") -> str:
    """The table's fingerprint from its rows (see ``db.Fingerprint``)."""
    from . import db

    r = parse_ref(ref)
    if r.is_db:
        entry = _entry(r)
        if entry is None:
            raise FileNotFoundError(f"{r} is not in the database")
        return db.fingerprint_table(_session(r), entry)
    cols = columns_of(r, encoding=encoding, delimiter=delimiter)
    fp = db.Fingerprint(cols)
    for row in read_rows(r, encoding=encoding, delimiter=delimiter):
        fp.add(row.get(c, "") for c in cols)
    return fp.hexdigest()


# ---------------------------------------------------------------------------
# Sidecars: the small JSON documents that ride beside a table
# ---------------------------------------------------------------------------

def _sidecar_file(r: TableRef, kind: str) -> Path:
    return r.path.with_name(f"{r.path.stem}_{kind}.json")


def read_sidecar(ref: RefLike, kind: str) -> Optional[dict]:
    """``<stem>_<kind>.json`` beside a file, or the database's sidecar row."""
    import json

    r = parse_ref(ref)
    if r.is_db:
        from . import db

        entry = _entry(r)
        return db.read_sidecar(_session(r), entry.table_name, kind) if entry else None
    try:
        with _sidecar_file(r, kind).open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def write_sidecar(ref: RefLike, kind: str, doc: Mapping) -> None:
    import json

    r = parse_ref(ref)
    if r.is_db:
        from . import db

        entry = _entry(r)
        if entry is None:
            raise FileNotFoundError(f"{r} is not in the database, so nothing can ride beside it")
        db.write_sidecar(_session(r), entry.table_name, kind, doc)
        return
    with atomic_write(_sidecar_file(r, kind), mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)


def delete_sidecar(ref: RefLike, kind: str) -> None:
    r = parse_ref(ref)
    if r.is_db:
        from . import db

        entry = _entry(r)
        if entry is not None:
            db.delete_sidecar(_session(r), entry.table_name, kind)
        return
    _sidecar_file(r, kind).unlink(missing_ok=True)


def output_ref(given: Any, *, source: RefLike, default: Sequence[str]) -> TableRef:
    """
    Where an analyzer's table goes: what the caller said, or the default
    beside the source. ``default`` is the folder path under the run folder
    (``("features", "readability")``); the table takes the source's name.
    A database source defaults to the same connection, under
    ``taters/<default...>/<name>``, so anything in gives anything out only
    when somebody asks for it.
    """
    if given:
        return parse_ref(given)
    src = parse_ref(source)
    if src.is_db:
        return TableRef(profile=src.profile,
                        logical="/".join(["taters", *default, src.stem]))
    return TableRef(Path.cwd().joinpath(*default) / f"{src.stem}.csv")


def handle(ref: RefLike):
    """What an analyzer returns: the ``Path`` for a file (unchanged for every
    caller that ever read one), the ref itself for a database table."""
    r = parse_ref(ref)
    return r.path if not r.is_db else r


def delete_table(ref: RefLike) -> bool:
    """
    Remove a table Taters made, and whatever rides beside it: a file and
    its sidecars, or a database table with its registry and sidecar rows.
    A table Taters did not make is left alone. Returns whether anything
    was removed.
    """
    r = parse_ref(ref)
    if r.is_db:
        from . import db

        entry = _entry(r)
        if entry is None:
            return False
        db.drop_table(_session(r), entry)
        return True
    if not r.path.is_file():
        return False
    r.path.unlink()
    for kind in ("settings", "sets", "unscored"):
        _sidecar_file(r, kind).unlink(missing_ok=True)
    return True


def tables_under(root: RefLike) -> List[TableRef]:
    """Every table Taters wrote under a database root such as
    ``db://lab/taters/study7/``; nothing for a folder (its files are its own)."""
    r = parse_ref(str(root).rstrip("/") + "/x")
    if not r.is_db:
        return []
    from . import db

    prefix = "/".join(r.parts[:-1])
    return [TableRef(profile=r.profile, logical=e.logical)
            for e in db.registered_under(_session(r), prefix)]


def require_local(ref: RefLike, what: str) -> Path:
    """The path of a file ref, or a plain refusal for a database one."""
    r = parse_ref(ref)
    if r.is_db:
        raise TypeError(f"{what} works on spreadsheets only for now; {r} is a "
                        f"table in a database. Copy it out with Wrangle data first.")
    return r.path


# ---------------------------------------------------------------------------
# Copying a database table (or a query) out to a spreadsheet
# ---------------------------------------------------------------------------

@records_settings(
    binding=("source", "connection", "columns"),
    outputs=("out_csv",),
    redo_without_record=True)
def export_table(
    *,
    source: Any,
    out_csv: Any,
    connection: str = "",
    columns: Optional[Sequence[str]] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Copy a database table, or a join of tables, into a CSV file.

    Parameters
    ----------
    source : str or mapping
        ``db://<connection>/<table>`` for one table, or a source spec (a base
        table plus joins; see :func:`taters.helpers.sources.render`) for
        several. A plain table name works when ``connection`` is given.
    out_csv : str or Path
        Where the spreadsheet goes.
    connection : str, optional
        The connection a bare table name or a spec belongs to.
    columns : sequence of str, optional
        Only these columns, in this order. All of them by default.
    overwrite_existing : bool, default False
        Rebuild the file if it is already there.
    verbose : bool, default True
        Print what was copied.
    on_progress : callable, optional
        Progress callback.

    Returns
    -------
    pathlib.Path
        ``out_csv``.
    """
    from . import db, sources

    out = Path(out_csv)
    if out.is_file() and not overwrite_existing:
        if verbose:
            print(f"[export] {out} exists; leaving it (overwrite_existing=False)")
        return out
    spec = sources.coerce(source, connection=connection)
    profile = db.get_profile(spec.connection)
    if profile.export == "forbidden":
        raise db.NotPermitted(f"{profile.name} is marked as not to be copied out of "
                              f"(export: forbidden in its connection settings)")
    session = db.session_for(spec.connection)
    with session, sources.guarded(spec, session):
        sql, params = sources.render(spec, session)
        announce(on_progress, f"reading from {spec.connection}")
        total = session.one(f"SELECT COUNT(*) FROM ({sql}) AS q", params)
        ticker = Ticker(on_progress, total=int(total or 0))
        seen = [0]

        def on_rows(n: int) -> None:
            ticker.tick(n - seen[0])
            seen[0] = n
        names, rows = db.stream_query(session, sql, params, on_rows=on_rows)
        wanted = [str(c) for c in columns] if columns else names
        missing = [c for c in wanted if c not in names]
        if missing:
            raise ValueError(f"the query has no column {missing[0]!r}; it has: "
                             f"{', '.join(names)}")
        with write_table(out, wanted) as w:
            for row in rows:
                w.writerow([row[c] for c in wanted])
    if verbose:
        print(f"[export] {spec.describe()} -> {out} ({w.rows:,} rows, {len(wanted)} columns)")
    return out


@records_settings(
    binding=("csv_path", "delimiter", "encoding", "unique_key"),
    outputs=("out",),
    redo_without_record=True)
def import_table(
    *,
    csv_path: Any,
    out: Any,
    unique_key: str = "",
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> TableRef:
    """
    Copy a spreadsheet into a database table.

    Parameters
    ----------
    csv_path : str or Path
        The spreadsheet. Its encoding and separator are detected unless given.
    out : str
        ``db://<connection>/<table>``.
    unique_key : str, optional
        A column that identifies each row; it becomes the table's key, and a
        repeated value stops the write.
    delimiter, encoding : str, optional
        How to read the file; detected when omitted.
    overwrite_existing : bool, default False
        Replace a table Taters made earlier under this name.
    verbose : bool, default True
        Print what was copied.
    on_progress : callable, optional
        Progress callback.

    Returns
    -------
    TableRef
        ``out``.
    """
    from . import reshape

    r = parse_ref(out)
    if not r.is_db:
        raise ValueError("out names a database table, db://<connection>/<table>")
    if exists(r) and not overwrite_existing:
        if verbose:
            print(f"[import] {r} exists; leaving it (overwrite_existing=False)")
        return r
    fmt = reshape.detect_format(csv_path, encoding=encoding, delimiter=delimiter)
    announce(on_progress, f"reading {fmt.name}")
    total = count(fmt.path, encoding=fmt.encoding)
    ticker = Ticker(on_progress, total=total)
    seen = [0]

    def on_rows(n: int) -> None:
        ticker.tick(n - seen[0])
        seen[0] = n
    with write_table(r, list(fmt.columns), unique_key=unique_key) as w:
        for row in reshape._Reader(fmt, on_rows=on_rows):
            w.writerow(row)
    if verbose:
        print(f"[import] {fmt.name} -> {r} ({w.rows:,} rows, {len(fmt.columns)} columns)")
    return r


CLI = CliSpec({"export": export_table, "import": import_table},
              description="Copy a database table out to a CSV, or a CSV into a database.")


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
