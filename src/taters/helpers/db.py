"""
Database connections: where they are, how to log in, what they allow, and
how a table gets written into one.

Three engines behind one small seam. SQLite is the standard library and a
real target for a single person's study; MySQL (and MariaDB) and PostgreSQL
are what a lab's data lives in. Each is a :class:`Dialect` of a few dozen
lines -- how to quote a name, what a placeholder looks like, how to list
tables, how to swap one table for another atomically -- over the DB-API
connection its driver gives. The drivers are imported inside functions, so
importing Taters never imports a database driver, and a missing one is a
sentence naming the extra rather than an import error at startup.

**Profiles** are the saved part: a name, the engine, where the server is,
which database and which user. Never the password. That comes from
``TATERS_DB_PASSWORD_<NAME>``, or from a masked prompt once per session,
and lives in this process's memory and nowhere else -- not in the
environment (a child process would inherit it), not in the run log, not in
the manifest, not in ``settings.json`` (which gets synced). Pool workers
never connect at all: the parent reads the rows and ships text to them.

**Writing** is copy-then-swap. Rows load into a scratch table, get their
types, and the finished table replaces the old one in one rename, so a run
killed halfway leaves the previous table exactly as it was. Every table
Taters writes carries a hidden ``_taters_row`` so it reads back in the
order it was written, and a row in ``_taters_tables`` saying what it is and
a fingerprint of its rows, which is how provenance tells "unchanged" from
"somebody edited this".
"""
from __future__ import annotations

import atexit
import contextlib
import datetime as _dt
import decimal
import hashlib
import json
import math
import multiprocessing
import os
import re
import secrets as _secrets
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Sequence, Tuple)

from . import settings as _settings

__all__ = [
    "DIALECTS", "Profile", "Probe", "TableInfo", "ColumnInfo", "Session",
    "list_profiles", "get_profile", "save_profile", "delete_profile",
    "password_env", "password_for", "forget_password", "remember_password",
    "connect", "probe", "DriverMissing", "DbError", "NotPermitted",
    "cell_text", "cell_token", "Fingerprint", "sanitize_table_name", "Loader",
    "REGISTRY", "SIDECARS", "ROW_COLUMN",
]

#: The engines, by the name a profile stores.
DIALECTS = ("sqlite", "mysql", "postgres")

#: Where the profiles live in ``settings.json``.
CONNECTIONS_KEY = "connections"

#: The two bookkeeping tables Taters keeps in a database it writes to.
REGISTRY = "_taters_tables"
SIDECARS = "_taters_sidecars"

#: The hidden column every written table carries: its rows' write order.
ROW_COLUMN = "_taters_row"

#: The one column a *packed* table keeps its measures in: a JSON array per
#: row, in the registry's column order. A table with more columns than the
#: engine allows (embeddings, a document-term matrix) is stored this way;
#: its identity columns stay real, so joins on text_id still work.
PACKED_COLUMN = "_taters_packed"

#: Columns that stay text whatever they look like, and stay real columns in
#: a packed table: "1" and "01" are two different rows.
ID_COLUMNS = ("text_id", "source_col", "source_path")

#: Rows per insert batch.
_BATCH = 1000

#: Longest physical table name we make. Under MySQL's 64 and Postgres's 63
#: with room for the scratch suffixes (``__tnew``, ``__told``, ``__tload``).
_NAME_LIMIT = 48

#: The engines' column-count ceilings (a bit under the real ones).
_MAX_COLUMNS = {"sqlite": 1900, "mysql": 1000, "postgres": 1500}

#: A plain integer: digits with no leading zero (so ``007`` is text).
_INTEGER = re.compile(r"^[+-]?(?:0|[1-9]\d*)$")
#: A plain decimal or exponent number without a leading zero before a digit.
_NUMBER = re.compile(r"^[+-]?(?:(?:0|[1-9]\d*)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_NON_FINITE = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}

_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
#: What may go on an ssh command line as a host or a login name.
_SSH_WORD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DbError(RuntimeError):
    """Something the database said, in words a person can act on."""


class DriverMissing(DbError):
    """The engine's driver is not installed; the message names the extra."""


class NotPermitted(DbError):
    """The account cannot do what the step needs."""


class HostKeyUnknown(DbError):
    """The ssh host is not trusted yet. ``keys`` are what it presented
    (see :func:`host_keys`), so the person can be shown them and asked."""

    def __init__(self, message: str, profile: "Profile", keys: Sequence["HostKey"] = ()):
        super().__init__(message)
        self.profile = profile
        self.keys = list(keys)


# ---------------------------------------------------------------------------
# Profiles and secrets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """
    A saved way to reach a database. Everything but the password.

    ``password`` records *how* the password is obtained -- ``"ask"`` (a
    masked prompt, once per session), ``"env"`` (``TATERS_DB_PASSWORD_<NAME>``),
    ``"server"`` (read from the person's own ``~/.my.cnf`` or ``~/.pgpass``
    on the ssh host, each session) or ``"none"`` -- never the password
    itself. A ``password`` value that is anything else is dropped when the
    profile is loaded.
    """

    name: str
    dialect: str
    host: str = ""
    port: int = 0
    user: str = ""
    database: str = ""
    schema: str = ""
    path: str = ""
    ssl: str = "required"
    ssl_ca: str = ""
    password: str = "ask"
    export: str = "allowed"
    #: A machine to tunnel through when the database is not on the network:
    #: ``host``/``port`` are then the database as seen from that machine
    #: (usually localhost). The tunnel is the system's own ``ssh``, so the
    #: keys, agent and config a person already logs in with are what it uses.
    ssh_host: str = ""
    ssh_port: int = 0
    ssh_user: str = ""
    ssh_key: str = ""
    #: The ssh host's key fingerprint(s) accepted from inside Taters, for
    #: plink, which otherwise only trusts what PuTTY has cached. OpenSSH
    #: keeps its own list (known_hosts), so this stays empty for it.
    ssh_hostkey: str = ""
    #: The database's unix socket on the ssh host. Set, the tunnel goes to
    #: the socket instead of a port, and the server sees the connection as
    #: the ssh user -- which is what lets a database that trusts the local
    #: login (socket authentication) be used with no password at all.
    socket: str = ""

    _FIELDS = ("name", "dialect", "host", "port", "user", "database", "schema",
               "path", "ssl", "ssl_ca", "password", "export",
               "ssh_host", "ssh_port", "ssh_user", "ssh_key", "ssh_hostkey", "socket")

    def __post_init__(self):
        if self.dialect not in DIALECTS:
            raise ValueError(f"dialect must be one of {DIALECTS}, not {self.dialect!r}")
        if not _PROFILE_NAME.match(self.name):
            raise ValueError(f"a connection name is lowercase letters, digits, "
                             f"_ or -, starting with a letter: {self.name!r}")
        if self.password not in ("ask", "env", "none", "server"):
            object.__setattr__(self, "password", "ask")
        # these become arguments to ssh/plink: a value starting with "-"
        # would be read as an option (ProxyCommand runs a local command),
        # so only the characters a host name or a login can hold are allowed
        for label, value in (("ssh host", self.ssh_host), ("ssh user", self.ssh_user)):
            if value and not _SSH_WORD.match(value):
                raise ValueError(f"an {label} is letters, digits, dots, underscores and "
                                 f"dashes, and cannot start with a dash: {value!r}")

    @property
    def engine_label(self) -> str:
        return {"sqlite": "SQLite", "mysql": "MySQL", "postgres": "PostgreSQL"}[self.dialect]

    @property
    def tunneled(self) -> bool:
        return bool(self.ssh_host)

    @property
    def ssh_address(self) -> str:
        """``user@host:port`` for the tunnel, as a person would type it."""
        port = f":{self.ssh_port}" if self.ssh_port and self.ssh_port != 22 else ""
        return (f"{self.ssh_user}@" if self.ssh_user else "") + f"{self.ssh_host}{port}"

    @property
    def server(self) -> str:
        """``db.lab.edu`` (with the port when it is not the usual one), or the
        socket or address as seen through the ssh tunnel."""
        port = f":{self.port}" if self.port and self.port != DEFAULT_PORT[self.dialect] else ""
        if self.tunneled:
            return (self.socket if self.socket else f"{self.host or 'localhost'}{port}") \
                + f" via ssh {self.ssh_address}"
        return f"{self.host}{port}"

    @property
    def where(self) -> str:
        """``study7 on db.lab.edu`` when pointed at a database, else just the
        server; or the file's name."""
        if self.dialect == "sqlite":
            return Path(self.path).name or self.path
        return f"{self.database} on {self.server}" if self.database else self.server

    def label(self) -> str:
        """One line for a picker: engine, who at where, and how it logs in. A
        connection is the server and the login; the database comes later."""
        parts = [self.engine_label]
        if self.dialect == "sqlite":
            parts.append(self.where)
        else:
            parts.append(f"{self.user}@{self.server}" if self.user else self.server)
            parts.append({"ask": "asks for its password",
                          "env": f"password from {password_env(self.name)}",
                          "server": "password from your saved login on the ssh host",
                          "none": ("no password, by socket login" if self.socket
                                   else "no password")}[self.password])
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "name"}


DEFAULT_PORT = {"sqlite": 0, "mysql": 3306, "postgres": 5432}


def password_env(name: str) -> str:
    """``lab-db`` -> ``TATERS_DB_PASSWORD_LAB_DB``."""
    return "TATERS_DB_PASSWORD_" + re.sub(r"[^A-Z0-9]", "_", name.upper())


def _profile_from(name: str, doc: Mapping) -> Optional[Profile]:
    fields = {k: doc.get(k) for k in Profile._FIELDS if k in doc and k != "name"}
    try:
        return Profile(name=name, **{k: (int(v) if k in ("port", "ssh_port") else str(v))
                                     for k, v in fields.items() if v is not None})
    except (ValueError, TypeError):
        return None


def list_profiles() -> List[Profile]:
    """Every saved connection, by name. A damaged entry is skipped."""
    doc = _settings.load_settings().get(CONNECTIONS_KEY) or {}
    out = []
    for name, entry in sorted(doc.items()):
        if isinstance(entry, Mapping):
            p = _profile_from(str(name), entry)
            if p is not None:
                out.append(p)
    return out


def get_profile(name: str) -> Profile:
    for p in list_profiles():
        if p.name == name:
            return p
    known = ", ".join(p.name for p in list_profiles()) or "none"
    raise KeyError(f"no connection called {name!r} (saved: {known})")


def save_profile(profile: Profile) -> Path:
    """Write a profile, keeping the others. Never a password."""
    doc = dict(_settings.load_settings().get(CONNECTIONS_KEY) or {})
    entry = profile.to_dict()
    doc[profile.name] = entry
    return _settings.save_setting(CONNECTIONS_KEY, doc)


def delete_profile(name: str) -> None:
    doc = dict(_settings.load_settings().get(CONNECTIONS_KEY) or {})
    doc.pop(name, None)
    forget_password(name)
    if doc:
        _settings.save_setting(CONNECTIONS_KEY, doc)
    else:
        _settings.clear_setting(CONNECTIONS_KEY)


#: Passwords typed this session, by profile name. Process memory only.
_SECRETS: Dict[str, str] = {}


def remember_password(name: str, value: str) -> None:
    _SECRETS[name] = value


def forget_password(name: str) -> None:
    _SECRETS.pop(name, None)


def password_for(profile: Profile,
                 ask: Optional[Callable[[str], str]] = None) -> Optional[str]:
    """
    The password for a profile, from the environment, then this session's
    memory, then ``ask(question)`` -- or None when the profile needs none.

    ``ask`` is the masked prompt; without one (a script, a worker), a
    password that is not already known raises and names the variable to
    set, so a headless run fails at once rather than at step seven.
    """
    if profile.dialect == "sqlite" or profile.password == "none":
        return None
    env = os.environ.get(password_env(profile.name))
    if env:
        return env
    if profile.name in _SECRETS:
        return _SECRETS[profile.name]
    if profile.password == "server":
        # the person's own saved login on the ssh host, read over the same
        # ssh login each session and kept only in memory, like a typed one
        hints = remote_login_hints(profile)
        if hints.password is None:
            raise NotPermitted(
                f"{profile.name} is set to take its password from your saved login on "
                f"{profile.ssh_host}, and {hints.problem or 'none was found there'}. Set "
                f"the connection to ask instead (Settings > Database connections).")
        _SECRETS[profile.name] = hints.password
        return hints.password
    if ask is None:
        raise NotPermitted(
            f"{profile.name} needs a password and none is known. Set "
            f"{password_env(profile.name)} in the environment, or run the "
            f"app, which asks for it.")
    value = ask(f"Password for {profile.user} on {profile.name}:")
    _SECRETS[profile.name] = value
    return value


# ---------------------------------------------------------------------------
# Values as text, and the fingerprint
# ---------------------------------------------------------------------------

def cell_text(value: Any) -> str:
    """
    A database value as the text a CSV export of the same table would hold.

    ``str()`` rather than ``isoformat()`` for dates, so ``2024-01-01
    12:00:00`` comes out the way a spreadsheet export writes it and an id
    built from a query equals one built from a file. Floats use ``repr``,
    the shortest text that reads back to the same number.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, decimal.Decimal):
        return format(value, "f")
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return str(value)
    return str(value)


def cell_token(text: str) -> bytes:
    """
    One cell canonicalized by *value*, so a CSV and a database copy of the
    same rows fingerprint the same whatever the column's type: blank or
    non-finite is empty, an integer is ``i:<n>``, another number is
    ``f:<repr(float)>``, and anything else is the text itself.
    """
    s = text.strip()
    if not s or s.lower() in _NON_FINITE:
        return b"\x00"
    if _INTEGER.match(s):
        return b"i:" + str(int(s)).encode()
    if _NUMBER.match(s):
        try:
            f = float(s)
        except ValueError:
            return b"s:" + s.encode("utf-8")
        if not math.isfinite(f):
            return b"\x00"
        if f == int(f) and abs(f) < 2 ** 53:
            return b"i:" + str(int(f)).encode()
        return b"f:" + repr(f).encode()
    return b"s:" + s.encode("utf-8")


class Fingerprint:
    """
    A table's identity from its rows, independent of their order and of
    how the engine typed them.

    Each row hashes to a digest; the table's fingerprint is a hash over the
    column names, the row count and the *sum* of the row digests. A sum
    rather than an exclusive-or, so two identical rows do not cancel out.
    """

    __slots__ = ("columns", "rows", "_sum")

    def __init__(self, columns: Sequence[str]):
        self.columns = [str(c) for c in columns]
        self.rows = 0
        self._sum = 0

    def add(self, cells: Iterable[str]) -> None:
        h = hashlib.sha256()
        for cell in cells:
            token = cell_token(cell)
            h.update(len(token).to_bytes(4, "big"))
            h.update(token)
        self._sum = (self._sum + int.from_bytes(h.digest(), "big")) % (1 << 256)
        self.rows += 1

    def hexdigest(self) -> str:
        h = hashlib.sha256(b"taters-table-v1")
        h.update(json.dumps(self.columns, ensure_ascii=False).encode("utf-8"))
        h.update(str(self.rows).encode())
        h.update(self._sum.to_bytes(32, "big"))
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Dialects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str          # the engine's declared type, uppercased, or ""

    @property
    def is_long_text(self) -> bool:
        t = self.type.upper()
        if t in ("TEXT", "MEDIUMTEXT", "LONGTEXT", "CLOB", "JSON"):
            return True
        m = re.match(r"(?:VAR)?CHAR(?:ACTER VARYING)?\((\d+)\)", t)
        return bool(m and int(m.group(1)) >= 256)


@dataclass(frozen=True)
class TableInfo:
    name: str
    kind: str = "table"               # "table" | "view"
    rows: Optional[int] = None        # exact or estimated; None when unknown
    rows_exact: bool = False
    columns: Tuple[ColumnInfo, ...] = ()
    primary_key: Tuple[str, ...] = ()
    #: (column here, other table, column there), as the database declares.
    foreign_keys: Tuple[Tuple[str, str, str], ...] = ()
    schema: str = ""

    @property
    def column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    @property
    def text_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.is_long_text]


def _split_placeholders(sql: str) -> str:
    """``?`` placeholders become ``%s`` outside quoted names and strings, and
    a literal ``%`` anywhere (``LIKE '\\_taters\\_%'``) becomes ``%%``, since the
    format-style drivers read every ``%`` as a placeholder, quoted or not."""
    out = []
    quote = None
    for ch in sql:
        if ch == "%":
            out.append("%%")
        elif quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ('"', "'", "`"):
            quote = ch
            out.append(ch)
        elif ch == "?":
            out.append("%s")
        else:
            out.append(ch)
    return "".join(out)


class Dialect:
    """What differs between engines. The base class is SQLite."""

    name = "sqlite"
    paramstyle = "qmark"
    extra = ""                      # the pip extra that installs the driver
    text_type = "TEXT"
    int_type = "INTEGER"
    float_type = "REAL"
    max_columns = _MAX_COLUMNS["sqlite"]

    # ---- connecting ---------------------------------------------------

    def connect(self, profile: Profile, password: Optional[str]):
        import sqlite3

        path = Path(os.path.expandvars(profile.path)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(path), isolation_level=None,
                              check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def version(self, session: "Session") -> str:
        return "SQLite " + session.one("SELECT sqlite_version()")

    def encrypted(self, session: "Session") -> Optional[bool]:
        return None                 # a file: the question does not apply

    def driver_errors(self) -> tuple:
        """The exception classes the driver raises for something the server
        said, so a session can turn them into words. Not a programming
        error: a TypeError from a bad parameter stays what it is."""
        import sqlite3

        return (sqlite3.Error,)

    def translate(self, sql: str) -> str:
        return sql

    # ---- names ----------------------------------------------------------

    def quote(self, name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def qualified(self, name: str, schema: str = "") -> str:
        return (self.quote(schema) + "." if schema else "") + self.quote(name)

    # ---- catalog --------------------------------------------------------

    def list_tables(self, session: "Session", schema: str = "") -> List[TableInfo]:
        rows = session.all("SELECT name, type FROM sqlite_master WHERE type IN "
                           "('table', 'view') AND name NOT LIKE 'sqlite_%' "
                           "AND name NOT LIKE '_taters_%' ORDER BY name")
        return [self.describe(session, name, kind=kind) for name, kind in rows]

    def describe(self, session: "Session", name: str, *, kind: str = "table",
                 schema: str = "") -> TableInfo:
        cols = session.all(f"PRAGMA table_info({self.quote(name)})")
        columns = tuple(ColumnInfo(str(c[1]), str(c[2] or "").upper()) for c in cols)
        pk = tuple(str(c[1]) for c in sorted((c for c in cols if c[5]), key=lambda c: c[5]))
        fks = tuple((str(f[3]), str(f[2]), str(f[4])) for f in
                    session.all(f"PRAGMA foreign_key_list({self.quote(name)})"))
        n = session.one(f"SELECT COUNT(*) FROM {self.quote(name)}")
        return TableInfo(name=name, kind=kind, rows=int(n), rows_exact=True,
                         columns=columns, primary_key=pk, foreign_keys=fks)

    def table_exists(self, session: "Session", name: str, schema: str = "") -> bool:
        return bool(session.one("SELECT COUNT(*) FROM sqlite_master WHERE name = ?",
                                (name,)))

    # ---- writing --------------------------------------------------------

    def create(self, session: "Session", name: str, columns: Sequence[Tuple[str, str]],
               *, schema: str = "") -> None:
        body = ", ".join(f"{self.quote(c)} {t}" for c, t in columns)
        session.run(f"CREATE TABLE {self.qualified(name, schema)} "
                    f"({self.quote(ROW_COLUMN)} INTEGER PRIMARY KEY, {body})")

    def drop(self, session: "Session", name: str, schema: str = "") -> None:
        session.run(f"DROP TABLE IF EXISTS {self.qualified(name, schema)}")

    def cast(self, column: str, kind: str) -> str:
        """``kind`` is ``"int"`` or ``"float"``; non-finite words become NULL."""
        q = self.quote(column)
        guard = (f"CASE WHEN LOWER(TRIM({q})) IN ('nan','inf','-inf','+inf',"
                 f"'infinity','-infinity','+infinity') OR TRIM({q}) = '' THEN NULL ELSE ")
        target = self.int_type if kind == "int" else self.float_type
        return f"{guard}CAST({q} AS {target}) END"

    def swap(self, session: "Session", name: str, new: str, old: str,
             schema: str = "") -> None:
        """``new`` becomes ``name``; what was ``name`` becomes ``old``. Atomic."""
        q = lambda n: self.qualified(n, schema)  # noqa: E731
        session.run("BEGIN IMMEDIATE")
        try:
            if self.table_exists(session, name, schema):
                session.run(f"ALTER TABLE {q(name)} RENAME TO {self.quote(old)}")
            session.run(f"ALTER TABLE {q(new)} RENAME TO {self.quote(name)}")
            session.run("COMMIT")
        except Exception:
            session.run("ROLLBACK")
            raise

    def upsert(self, session: "Session", table: str, row: Mapping[str, Any],
               keys: Sequence[str], schema: str = "") -> None:
        cols = list(row)
        marks = ", ".join("?" for _ in cols)
        names = ", ".join(self.quote(c) for c in cols)
        session.run(f"INSERT OR REPLACE INTO {self.qualified(table, schema)} "
                    f"({names}) VALUES ({marks})", tuple(row[c] for c in cols))

    def index(self, session: "Session", table: str, column: str,
              schema: str = "") -> None:
        idx = f"{table}__{column}__ix"[:60]
        session.run(f"CREATE INDEX IF NOT EXISTS {self.quote(idx)} ON "
                    f"{self.qualified(table, schema)} ({self.quote(column)})")

    def sort_key(self, quoted: str) -> str:
        """An expression that puts equal values together, blanks with NULLs."""
        return f"COALESCE(CAST({quoted} AS TEXT), '')"

    def as_text(self, quoted: str) -> str:
        return f"CAST({quoted} AS TEXT)"

    def blank(self, quoted: str) -> str:
        """True when the cell is NULL or empty after trimming."""
        return f"({quoted} IS NULL OR TRIM({self.as_text(quoted)}) = '')"

    def numeric(self, quoted: str) -> str:
        """True when a non-blank cell reads as a number. An offer, not a
        contract: the wizard shows the verdict and the arrows can change it."""
        t = f"TRIM({self.as_text(quoted)})"
        return (f"(typeof({quoted}) IN ('integer', 'real') OR "
                f"({t} GLOB '*[0-9]*' AND NOT {t} GLOB '*[^0-9.eE+-]*'))")

    def limit(self, sql: str, n: int) -> str:
        return f"{sql} LIMIT {int(n)}"

    def stream_cursor(self, con):
        return con.cursor()

    def stream_scope(self, con):
        """What a streaming read has to run inside: nothing, for most engines."""
        return contextlib.nullcontext()

    #: Statements that put the session into, and back out of, read-only mode.
    read_only_on = "PRAGMA query_only = 1"
    read_only_off = "PRAGMA query_only = 0"

    def list_databases(self, session: "Session") -> List[str]:
        """The databases on the server this account can see; none for a file."""
        return []

    def count_tables(self, session: "Session", schema: str = "") -> int:
        """How many tables the account can see, in one query -- the probe
        wants the number, not a description of each."""
        return int(session.one("SELECT COUNT(*) FROM sqlite_master WHERE type IN "
                               "('table', 'view') AND name NOT LIKE 'sqlite_%' "
                               "AND name NOT LIKE '_taters_%'") or 0)

    def executemany(self, cur, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        cur.executemany(sql, rows)


class MySQLDialect(Dialect):
    name = "mysql"
    paramstyle = "format"
    extra = "mysql"
    text_type = "LONGTEXT"
    int_type = "SIGNED"
    float_type = "DOUBLE"
    max_columns = _MAX_COLUMNS["mysql"]

    def connect(self, profile: Profile, password: Optional[str]):
        try:
            import pymysql
        except ImportError:
            raise DriverMissing('MySQL needs its driver: pip install "taters[db]"') from None
        # the context is built here rather than left to the driver: handed a
        # dict with no CA, pymysql quietly verifies nothing, so "verify" was
        # "required" with extra steps. whether the link really is encrypted
        # is checked after login, in connect(), for every engine
        host, port = _endpoint(profile)
        try:
            con = pymysql.connect(host=host, port=port,
                                  user=profile.user, password=password or "",
                                  database=profile.database or None, charset="utf8mb4",
                                  ssl=_ssl_context(profile), autocommit=True,
                                  read_timeout=3600, write_timeout=3600)
        except (pymysql.err.MySQLError, OSError) as e:
            # every failure to log in is said in words: the driver's own
            # exceptions (a wrong password is one kind, a port that is not
            # speaking MySQL another) and the socket's
            raise DbError(_mysql_words(e, profile)) from None
        with con.cursor() as cur:
            cur.execute("SET SESSION sql_mode = 'STRICT_ALL_TABLES'")
            cur.execute("SET SESSION net_write_timeout = 3600")
        return con

    def version(self, session: "Session") -> str:
        return "MySQL " + str(session.one("SELECT VERSION()"))

    def encrypted(self, session: "Session") -> Optional[bool]:
        row = session.all("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
        return bool(row and row[0][1]) if row else None

    def driver_errors(self) -> tuple:
        import pymysql

        return (pymysql.err.MySQLError,)

    def translate(self, sql: str) -> str:
        return _split_placeholders(sql)

    def quote(self, name: str) -> str:
        return "`" + str(name).replace("`", "``") + "`"

    read_only_on = "SET SESSION TRANSACTION READ ONLY"
    read_only_off = "SET SESSION TRANSACTION READ WRITE"

    def list_databases(self, session: "Session") -> List[str]:
        system = {"information_schema", "mysql", "performance_schema", "sys"}
        return [str(r[0]) for r in session.all("SHOW DATABASES")
                if str(r[0]).lower() not in system]

    def count_tables(self, session: "Session", schema: str = "") -> int:
        return int(session.one(
            "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() "
            "AND TABLE_NAME NOT LIKE '\\_taters\\_%'") or 0)

    def list_tables(self, session: "Session", schema: str = "") -> List[TableInfo]:
        # four queries for the whole catalog, however many tables there are:
        # one per table was hundreds of round trips through a tunnel, and a
        # COUNT(*) on every view could take minutes on its own. views show
        # as "? rows"; a table's count is the engine's estimate
        kinds = {str(n): ("view" if "VIEW" in str(t) else "table", r) for n, t, r in session.all(
            "SELECT TABLE_NAME, TABLE_TYPE, TABLE_ROWS FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME NOT LIKE '\\_taters\\_%' "
            "ORDER BY TABLE_NAME")}
        columns: Dict[str, List[ColumnInfo]] = {n: [] for n in kinds}
        pks: Dict[str, List[str]] = {n: [] for n in kinds}
        for t, c, typ, key in session.all(
                "SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, COLUMN_KEY FROM "
                "information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION"):
            if str(t) in columns:
                columns[str(t)].append(ColumnInfo(str(c), str(typ).upper()))
                if key == "PRI":
                    pks[str(t)].append(str(c))
        fks: Dict[str, List[Tuple[str, str, str]]] = {n: [] for n in kinds}
        for t, a, b, c in session.all(
                "SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME "
                "FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = DATABASE() "
                "AND REFERENCED_TABLE_NAME IS NOT NULL ORDER BY TABLE_NAME, ORDINAL_POSITION"):
            if str(t) in fks:
                fks[str(t)].append((str(a), str(b), str(c)))
        return [TableInfo(name=n, kind=kind,
                          rows=(int(r) if r is not None and kind == "table" else None),
                          rows_exact=False, columns=tuple(columns[n]),
                          primary_key=tuple(pks[n]), foreign_keys=tuple(fks[n]))
                for n, (kind, r) in kinds.items()]

    def describe(self, session: "Session", name: str, *, kind: str = "table",
                 schema: str = "", estimate: Optional[int] = None) -> TableInfo:
        cols = session.all(
            "SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_KEY FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
            (name,))
        columns = tuple(ColumnInfo(str(c[0]), str(c[1]).upper()) for c in cols)
        pk = tuple(str(c[0]) for c in cols if c[2] == "PRI")
        fks = tuple((str(a), str(b), str(c)) for a, b, c in session.all(
            "SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME FROM "
            "information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = DATABASE() AND "
            "TABLE_NAME = ? AND REFERENCED_TABLE_NAME IS NOT NULL", (name,)))
        return TableInfo(name=name, kind=kind, rows=estimate, rows_exact=False,
                         columns=columns, primary_key=pk, foreign_keys=fks)

    def table_exists(self, session: "Session", name: str, schema: str = "") -> bool:
        return bool(session.one(
            "SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = "
            "DATABASE() AND TABLE_NAME = ?", (name,)))

    def create(self, session: "Session", name: str, columns: Sequence[Tuple[str, str]],
               *, schema: str = "") -> None:
        body = ", ".join(f"{self.quote(c)} {t}" for c, t in columns)
        session.run(f"CREATE TABLE {self.quote(name)} ({self.quote(ROW_COLUMN)} BIGINT "
                    f"PRIMARY KEY, {body}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 "
                    f"COLLATE=utf8mb4_bin")

    def swap(self, session: "Session", name: str, new: str, old: str,
             schema: str = "") -> None:
        if self.table_exists(session, name):
            session.run(f"RENAME TABLE {self.quote(name)} TO {self.quote(old)}, "
                        f"{self.quote(new)} TO {self.quote(name)}")
        else:
            session.run(f"RENAME TABLE {self.quote(new)} TO {self.quote(name)}")

    def upsert(self, session: "Session", table: str, row: Mapping[str, Any],
               keys: Sequence[str], schema: str = "") -> None:
        cols = list(row)
        names = ", ".join(self.quote(c) for c in cols)
        marks = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{self.quote(c)} = VALUES({self.quote(c)})"
                            for c in cols if c not in keys)
        session.run(f"INSERT INTO {self.quote(table)} ({names}) VALUES ({marks}) "
                    f"ON DUPLICATE KEY UPDATE {updates}", tuple(row[c] for c in cols))

    def index(self, session: "Session", table: str, column: str,
              schema: str = "") -> None:
        idx = f"{table}__{column}__ix"[:60]
        session.run(f"CREATE INDEX {self.quote(idx)} ON {self.quote(table)} "
                    f"({self.quote(column)}(191))")

    def sort_key(self, quoted: str) -> str:
        return f"CAST(COALESCE({quoted}, '') AS BINARY)"

    def as_text(self, quoted: str) -> str:
        return f"CAST({quoted} AS CHAR)"

    def numeric(self, quoted: str) -> str:
        return (f"(TRIM({self.as_text(quoted)}) REGEXP "
                f"'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$')")

    def stream_cursor(self, con):
        import pymysql.cursors
        return con.cursor(pymysql.cursors.SSCursor)


class PostgresDialect(Dialect):
    name = "postgres"
    paramstyle = "format"
    extra = "postgres"
    text_type = "TEXT"
    int_type = "BIGINT"
    float_type = "DOUBLE PRECISION"
    max_columns = _MAX_COLUMNS["postgres"]

    def connect(self, profile: Profile, password: Optional[str]):
        try:
            import psycopg
        except ImportError:
            raise DriverMissing('PostgreSQL needs its driver: pip install "taters[db]"') from None
        sslmode = {"verify": "verify-full", "off": "disable"}.get(
            "off" if profile.tunneled else profile.ssl, "require")
        # no database yet means "log in and look around": postgres always
        # needs one, so the maintenance database stands in for the moment
        host, port = _endpoint(profile)
        kwargs: Dict[str, Any] = dict(host=host, port=port,
                                      user=profile.user, password=password or "",
                                      dbname=profile.database or "postgres",
                                      sslmode=sslmode, autocommit=True)
        if profile.ssl_ca:
            kwargs["sslrootcert"] = profile.ssl_ca
        try:
            con = psycopg.connect(**kwargs)
        except (psycopg.Error, OSError) as e:
            raise DbError(_postgres_words(e, profile)) from None
        if profile.schema:
            with con.cursor() as cur:
                cur.execute(f"SET search_path TO {self.quote(profile.schema)}")
        return con

    def version(self, session: "Session") -> str:
        return str(session.one("SHOW server_version")).split()[0].join(["PostgreSQL ", ""])

    def encrypted(self, session: "Session") -> Optional[bool]:
        try:
            return bool(session.one("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()"))
        except Exception:
            return None

    def driver_errors(self) -> tuple:
        import psycopg

        return (psycopg.Error,)

    def translate(self, sql: str) -> str:
        return _split_placeholders(sql)

    def list_tables(self, session: "Session", schema: str = "") -> List[TableInfo]:
        # four queries for the whole catalog; see MySQLDialect.list_tables
        kinds = {str(n): ("view" if k in ("v", "m") else "table", r) for n, k, r in session.all(
            "SELECT c.relname, c.relkind, c.reltuples FROM pg_class c JOIN pg_namespace n "
            "ON n.oid = c.relnamespace WHERE n.nspname = COALESCE(NULLIF(?, ''), "
            "current_schema()) AND c.relkind IN ('r', 'v', 'm', 'p') AND c.relname "
            "NOT LIKE '\\_taters\\_%' ORDER BY c.relname", (schema,))}
        columns: Dict[str, List[ColumnInfo]] = {n: [] for n in kinds}
        for t, c, typ, length in session.all(
                "SELECT table_name, column_name, data_type, character_maximum_length FROM "
                "information_schema.columns WHERE table_schema = COALESCE(NULLIF(?, ''), "
                "current_schema()) ORDER BY table_name, ordinal_position", (schema,)):
            if str(t) in columns:
                columns[str(t)].append(ColumnInfo(
                    str(c), (f"VARCHAR({length})" if length else str(typ)).upper()))
        pks: Dict[str, List[str]] = {n: [] for n in kinds}
        for t, c in session.all(
                "SELECT c.relname, a.attname FROM pg_index i JOIN pg_class c ON c.oid = "
                "i.indrelid JOIN pg_namespace n ON n.oid = c.relnamespace JOIN pg_attribute a "
                "ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) WHERE i.indisprimary "
                "AND n.nspname = COALESCE(NULLIF(?, ''), current_schema()) ORDER BY c.relname, "
                "array_position(i.indkey, a.attnum)", (schema,)):
            if str(t) in pks:
                pks[str(t)].append(str(c))
        fks: Dict[str, List[Tuple[str, str, str]]] = {n: [] for n in kinds}
        for t, a, b, c in session.all(
                "SELECT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name FROM "
                "information_schema.table_constraints tc JOIN information_schema."
                "key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND "
                "tc.table_schema = kcu.table_schema JOIN information_schema."
                "constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name "
                "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = "
                "COALESCE(NULLIF(?, ''), current_schema()) ORDER BY tc.table_name, "
                "kcu.ordinal_position", (schema,)):
            if str(t) in fks:
                fks[str(t)].append((str(a), str(b), str(c)))
        return [TableInfo(name=n, kind=kind,
                          rows=(int(r) if r is not None and r >= 0 and kind == "table" else None),
                          rows_exact=False, columns=tuple(columns[n]),
                          primary_key=tuple(pks[n]), foreign_keys=tuple(fks[n]), schema=schema)
                for n, (kind, r) in kinds.items()]

    def describe(self, session: "Session", name: str, *, kind: str = "table",
                 schema: str = "", estimate: Optional[int] = None) -> TableInfo:
        cols = session.all(
            "SELECT column_name, data_type, character_maximum_length FROM "
            "information_schema.columns WHERE table_schema = COALESCE(NULLIF(?, ''), "
            "current_schema()) AND table_name = ? ORDER BY ordinal_position",
            (schema, name))
        columns = tuple(ColumnInfo(str(c[0]), (f"VARCHAR({c[2]})" if c[2] else str(c[1])).upper())
                        for c in cols)
        pk = tuple(str(r[0]) for r in session.all(
            "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid = "
            "i.indrelid AND a.attnum = ANY(i.indkey) WHERE i.indrelid = ?::regclass "
            "AND i.indisprimary ORDER BY array_position(i.indkey, a.attnum)",
            (self.qualified(name, schema),)))
        fks = tuple((str(a), str(b), str(c)) for a, b, c in session.all(
            "SELECT kcu.column_name, ccu.table_name, ccu.column_name FROM "
            "information_schema.table_constraints tc JOIN information_schema."
            "key_column_usage kcu ON tc.constraint_name = kcu.constraint_name AND "
            "tc.table_schema = kcu.table_schema JOIN information_schema."
            "constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name "
            "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = ?", (name,)))
        return TableInfo(name=name, kind=kind, rows=estimate, rows_exact=False,
                         columns=columns, primary_key=pk, foreign_keys=fks, schema=schema)

    def table_exists(self, session: "Session", name: str, schema: str = "") -> bool:
        return bool(session.one(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = "
            "COALESCE(NULLIF(?, ''), current_schema()) AND table_name = ?", (schema, name)))

    def create(self, session: "Session", name: str, columns: Sequence[Tuple[str, str]],
               *, schema: str = "") -> None:
        body = ", ".join(f"{self.quote(c)} {t}" for c, t in columns)
        session.run(f"CREATE TABLE {self.qualified(name, schema)} "
                    f"({self.quote(ROW_COLUMN)} BIGINT PRIMARY KEY, {body})")

    def cast(self, column: str, kind: str) -> str:
        q = self.quote(column)
        target = self.int_type if kind == "int" else self.float_type
        return (f"CASE WHEN LOWER(TRIM({q})) IN ('nan','inf','-inf','+inf','infinity',"
                f"'-infinity','+infinity') OR TRIM({q}) = '' THEN NULL ELSE "
                f"CAST({q} AS {target}) END")

    def swap(self, session: "Session", name: str, new: str, old: str,
             schema: str = "") -> None:
        q = lambda n: self.qualified(n, schema)  # noqa: E731
        session.run("BEGIN")
        try:
            if self.table_exists(session, name, schema):
                session.run(f"ALTER TABLE {q(name)} RENAME TO {self.quote(old)}")
            session.run(f"ALTER TABLE {q(new)} RENAME TO {self.quote(name)}")
            session.run("COMMIT")
        except Exception:
            session.run("ROLLBACK")
            raise

    def upsert(self, session: "Session", table: str, row: Mapping[str, Any],
               keys: Sequence[str], schema: str = "") -> None:
        cols = list(row)
        names = ", ".join(self.quote(c) for c in cols)
        marks = ", ".join("?" for _ in cols)
        updates = ", ".join(f"{self.quote(c)} = EXCLUDED.{self.quote(c)}"
                            for c in cols if c not in keys)
        session.run(f"INSERT INTO {self.qualified(table, schema)} ({names}) VALUES "
                    f"({marks}) ON CONFLICT ({', '.join(self.quote(k) for k in keys)}) "
                    f"DO UPDATE SET {updates}", tuple(row[c] for c in cols))

    def index(self, session: "Session", table: str, column: str,
              schema: str = "") -> None:
        idx = f"{table}__{column}__{_secrets.token_hex(3)}"[:60]
        session.run(f"CREATE INDEX {self.quote(idx)} ON {self.qualified(table, schema)} "
                    f"({self.quote(column)})")

    def sort_key(self, quoted: str) -> str:
        return f"COALESCE(CAST({quoted} AS TEXT), '') COLLATE \"C\""

    def numeric(self, quoted: str) -> str:
        return (f"(TRIM({self.as_text(quoted)}) ~ "
                f"'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$')")

    def stream_cursor(self, con):
        return con.cursor(name=f"taters_{_secrets.token_hex(4)}")

    def stream_scope(self, con):
        # a named (server-side) cursor only lives inside a transaction
        return con.transaction()

    read_only_on = "SET default_transaction_read_only = on"
    read_only_off = "SET default_transaction_read_only = off"

    def list_databases(self, session: "Session") -> List[str]:
        return [str(r[0]) for r in session.all(
            "SELECT datname FROM pg_database WHERE NOT datistemplate AND datname <> "
            "'postgres' ORDER BY datname")]

    def count_tables(self, session: "Session", schema: str = "") -> int:
        return int(session.one(
            "SELECT COUNT(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = COALESCE(NULLIF(?, ''), current_schema()) AND c.relkind IN "
            "('r', 'v', 'm', 'p') AND c.relname NOT LIKE '\\_taters\\_%'", (schema,)) or 0)


#: Open tunnels by profile name: the ssh process and the local port it
#: forwards. One per profile, shared by every session on it, closed with the
#: sessions (see ``tables.close_sessions``) and at exit.
_TUNNELS: Dict[str, Tuple[Any, int]] = {}


def uses_putty(profile: Profile) -> bool:
    """A ``.ppk`` key is PuTTY's format, which OpenSSH cannot read; the
    tunnel is then opened with PuTTY's own command-line client, plink,
    which also uses Pageant and the host keys PuTTY has already accepted."""
    return profile.ssh_key.lower().endswith(".ppk")


def plink_path() -> Optional[str]:
    """Where plink is: on the PATH, or in PuTTY's usual Windows folders."""
    import shutil

    found = shutil.which("plink")
    if found:
        return found
    for folder in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                   os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")):
        candidate = Path(folder) / "PuTTY" / "plink.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def tunnel_command(profile: Profile, local_port: int) -> List[str]:
    """
    The invocation that forwards ``local_port`` to the database as the SSH
    host sees it: OpenSSH's ``ssh``, or PuTTY's ``plink`` for a ``.ppk``
    key. Batch mode either way -- a key, an agent or Pageant, never a
    prompt, since there is no terminal to answer one in the middle of a run.
    """
    # a socket on the far end instead of a port: the server then sees the
    # ssh user, not a network client (OpenSSH only; plink cannot forward one)
    far = profile.socket if profile.socket else \
        f"{profile.host or 'localhost'}:{profile.port or DEFAULT_PORT[profile.dialect]}"
    target = f"127.0.0.1:{local_port}:{far}"
    who = (f"{profile.ssh_user}@" if profile.ssh_user else "") + profile.ssh_host
    key = str(Path(profile.ssh_key).expanduser()) if profile.ssh_key else ""
    if uses_putty(profile):
        if profile.socket:
            raise DbError(f"{profile.name} logs in over the database's socket on "
                          f"{profile.ssh_host}, which PuTTY's plink cannot forward. Use an "
                          f"OpenSSH key for this connection (PuTTYgen: Conversions > Export "
                          f"OpenSSH key), or give it the database's port and a password.")
        cmd = [plink_path() or "plink", "-ssh", "-batch", "-v", "-N", "-L", target]
        if profile.ssh_port and profile.ssh_port != 22:
            cmd += ["-P", str(profile.ssh_port)]
        for fp in [f.strip() for f in profile.ssh_hostkey.split(",") if f.strip()]:
            cmd += ["-hostkey", fp]
        cmd += ["-i", key, who]
        return cmd
    cmd = ["ssh", "-v", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
           "-o", "ServerAliveInterval=30", "-L", target]
    if profile.ssh_port and profile.ssh_port != 22:
        cmd += ["-p", str(profile.ssh_port)]
    if key:
        cmd += ["-i", key]
    cmd.append(who)
    return cmd


@dataclass(frozen=True)
class LoginHints:
    """What the ssh host's own database client would do: the login saved
    in ``~/.my.cnf`` / ``~/.pgpass`` (user, password), the database's local
    socket if there is one, and the ssh user. Nothing here is shown or
    saved; a password found is used for the session and forgotten."""

    user: str = ""
    # kept out of repr, so no traceback, log line or debug print of a
    # LoginHints can ever carry the password
    password: Optional[str] = field(default=None, repr=False)
    socket: str = ""
    source: str = ""            # which file the login came from, if any
    os_user: str = ""
    problem: str = ""           # why nothing could be read, in words

    @property
    def found(self) -> bool:
        return bool(self.password is not None or self.socket)


#: One command, run on the ssh host, that prints the three things worth
#: knowing with markers between them. Only the person's own files, and only
#: the places the database's socket usually lives.
_LOOK = ("echo --mycnf--; cat ~/.my.cnf 2>/dev/null; echo --pgpass--; cat ~/.pgpass 2>/dev/null; "
         "echo --sockets--; for s in /run/mysqld/mysqld.sock /var/run/mysqld/mysqld.sock "
         "/var/lib/mysql/mysql.sock /tmp/mysql.sock; do [ -S \"$s\" ] && echo mysql=$s; done; "
         "for s in /var/run/postgresql/.s.PGSQL.5432 /run/postgresql/.s.PGSQL.5432 "
         "/tmp/.s.PGSQL.5432; do [ -S \"$s\" ] && echo postgres=$s; done; echo --user--; id -un")


def remote_command(profile: Profile, command: str, timeout: float = 30.0) -> Tuple[int, str, str]:
    """Run one shell command on the ssh host, in batch mode, through the
    same program and key the tunnel uses. ``(exit code, stdout, stderr)``."""
    import subprocess

    who = (f"{profile.ssh_user}@" if profile.ssh_user else "") + profile.ssh_host
    key = str(Path(profile.ssh_key).expanduser()) if profile.ssh_key else ""
    if uses_putty(profile):
        cmd = [plink_path() or "plink", "-ssh", "-batch"]
        if profile.ssh_port and profile.ssh_port != 22:
            cmd += ["-P", str(profile.ssh_port)]
        for fp in [f.strip() for f in profile.ssh_hostkey.split(",") if f.strip()]:
            cmd += ["-hostkey", fp]
        cmd += ["-i", key, who, command]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes"]
        if profile.ssh_port and profile.ssh_port != 22:
            cmd += ["-p", str(profile.ssh_port)]
        if key:
            cmd += ["-i", key]
        cmd += [who, command]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    return done.returncode, done.stdout, done.stderr


def remote_login_hints(profile: Profile) -> LoginHints:
    """
    Look on the ssh host for how its own database client logs in, with the
    person's consent already given: their ``~/.my.cnf`` (MySQL) or
    ``~/.pgpass`` (PostgreSQL), and whether the database's socket is there.
    A failure to log in comes back as ``problem``, never as an exception,
    because this is a look, not the connection.
    """
    import subprocess

    try:
        code, out, err = remote_command(profile, _LOOK)
    except (OSError, subprocess.SubprocessError) as e:
        return LoginHints(problem=f"could not run a command on {profile.ssh_host}: {e}")
    if code != 0 and "--mycnf--" not in out:
        if _host_key_unknown(err):
            raise HostKeyUnknown(_ssh_words(err, profile), profile, keys=host_keys(profile, err))
        return LoginHints(problem=_ssh_words(err, profile))
    return parse_login_hints(out, profile)


def parse_login_hints(text: str, profile: Profile) -> LoginHints:
    """The look's output as :class:`LoginHints`; separated out so it can be
    tested without a server."""
    sections: Dict[str, List[str]] = {}
    current = ""
    for line in text.splitlines():
        if line.strip() in ("--mycnf--", "--pgpass--", "--sockets--", "--user--"):
            current = line.strip().strip("-")
            sections[current] = []
        elif current:
            sections[current].append(line)
    os_user = next((ln.strip() for ln in sections.get("user", []) if ln.strip()), "")
    sockets = dict(ln.split("=", 1) for ln in sections.get("sockets", []) if "=" in ln)
    socket = sockets.get(profile.dialect, "")
    user, password, source = "", None, ""
    if profile.dialect == "mysql":
        client = _my_cnf_client("\n".join(sections.get("mycnf", [])))
        user = client.get("user", "")
        password = client.get("password")
        socket = client.get("socket", socket) or socket
        source = "~/.my.cnf" if client else ""
    elif profile.dialect == "postgres":
        hit = _pgpass_match("\n".join(sections.get("pgpass", [])), profile)
        if hit:
            user, password, source = hit[0], hit[1], "~/.pgpass"
    return LoginHints(user=user, password=password, socket=socket, source=source,
                      os_user=os_user)


def _my_cnf_client(text: str) -> Dict[str, str]:
    """The ``[client]`` section of a ``.my.cnf`` (``[mysql]`` counts too),
    quotes stripped, later lines winning."""
    out: Dict[str, str] = {}
    section = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        if section not in ("client", "mysql"):
            continue
        key, sep, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip().strip("'\"") if sep else ""
        if key in ("user", "password", "socket", "host", "port"):
            out[key] = value
    return out


def _pgpass_match(text: str, profile: Profile) -> Optional[Tuple[str, str]]:
    """``(user, password)`` from the first ``.pgpass`` line that fits this
    database: host and database may be ``*``, and a local socket counts as
    localhost."""
    wanted_db = profile.database or ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) < 5:
            continue
        host, _port, dbname, user, password = parts[0], parts[1], parts[2], parts[3], ":".join(parts[4:])
        if host not in ("*", "localhost", "127.0.0.1", profile.host or "localhost"):
            continue
        if dbname not in ("*", wanted_db) and wanted_db:
            continue
        if profile.user and user not in ("*", profile.user):
            continue
        return (user if user != "*" else profile.user), password
    return None


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _port_open(port: int) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(0.3)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


#: How long a tunnel gets to come up. A university login can take a while
#: (a slow gateway, a key agent being asked), and a minute costs nothing
#: when it works; when it does not, the story of where it stalled is what
#: comes back, not just the number.
TUNNEL_TIMEOUT = 60.0


def open_tunnel(profile: Profile, timeout: float = TUNNEL_TIMEOUT) -> int:
    """
    The local port the database is reachable on through the SSH host,
    opening the tunnel on first use. A tunnel that cannot be opened says
    why in ssh's own words, which name the key, the host or the permission
    that was the problem; one that stalls says how far it got.
    """
    import shutil
    import subprocess
    import threading
    import time
    from collections import deque

    have = _TUNNELS.get(profile.name)
    if have is not None and have[0].poll() is None and _port_open(have[1]):
        return have[1]
    if have is not None:
        _TUNNELS.pop(profile.name, None)
    if uses_putty(profile):
        if plink_path() is None:
            raise DbError(f"{profile.name} logs in with a PuTTY key ({Path(profile.ssh_key).name}), "
                          f"which needs PuTTY's plink, and it was not found on this computer. "
                          f"Install PuTTY (plink comes with it), or convert the key to "
                          f"OpenSSH's format in PuTTYgen (Conversions > Export OpenSSH key) "
                          f"and pick that file instead.")
    elif shutil.which("ssh") is None:
        raise DbError(f"{profile.name} is reached through ssh, and no ssh program was "
                      f"found on this computer. Install OpenSSH (Windows: Settings > "
                      f"Apps > Optional features > OpenSSH Client) and try again.")
    port = _free_port()
    proc = subprocess.Popen(tunnel_command(profile, port), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # what ssh says, read as it says it: a full pipe would stall ssh itself,
    # and the last lines are the story when nothing else is
    said: deque = deque(maxlen=200)

    def collect() -> None:
        for raw in iter(proc.stderr.readline, b""):
            said.append(raw.decode("utf-8", "replace").rstrip())
    threading.Thread(target=collect, daemon=True).start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            time.sleep(0.2)                    # let the reader drain the tail
            log = "\n".join(said)
            if _host_key_unknown(log):
                # the one refusal a person can settle on the spot: here is
                # the key the server showed, do you accept it
                raise HostKeyUnknown(_ssh_words(log, profile), profile,
                                     keys=host_keys(profile, log))
            raise DbError(_ssh_words(log, profile))
        if _port_open(port):
            _TUNNELS[profile.name] = (proc, port)
            return port
        time.sleep(0.2)
    proc.kill()
    raise DbError(_stall_words("\n".join(said), profile, timeout))


def _host_key_unknown(log: str) -> bool:
    low = log.lower()
    return ("host key is not cached" in low or "host key verification failed" in low
            or "no ed25519 host key is known" in low or "host key is known" in low
            and "strict checking" in low)


@dataclass(frozen=True)
class HostKey:
    """One key an ssh host presented: its type, the key itself (base64, as
    a known_hosts line carries it) and the fingerprint people compare."""

    type: str
    blob: str
    fingerprint: str


def _fingerprint(blob_b64: str) -> str:
    """``SHA256:...`` the way ssh-keygen -l prints it: unpadded base64 of
    the SHA-256 of the raw key."""
    import base64

    raw = base64.b64decode(blob_b64)
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


def host_keys(profile: Profile, log: str = "") -> List[HostKey]:
    """
    The keys the ssh host presents, by asking it (``ssh-keyscan``), or from
    what plink already printed when it refused: its complaint names the
    key's type and fingerprint. Empty when neither is to be had.
    """
    import shutil
    import subprocess

    found: List[HostKey] = []
    port = profile.ssh_port or 22
    if shutil.which("ssh-keyscan"):
        try:
            out = subprocess.run(["ssh-keyscan", "-p", str(port), "-T", "10", profile.ssh_host],
                                 capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and not line.startswith("#"):
                try:
                    found.append(HostKey(parts[1], parts[2], _fingerprint(parts[2])))
                except Exception:
                    continue
    if not found and not uses_putty(profile) and shutil.which("ssh"):
        # no ssh-keyscan (or it got nothing): ssh itself will write the key
        # it is shown into a known_hosts file of our choosing, and a
        # throwaway file is one nothing trusts. the login is expected to
        # fail; the file is what was wanted
        found = _keys_via_ssh(profile)
    if not found:
        # plink: "ssh-ed25519 255 SHA256:AbC..." somewhere in its refusal
        m = re.search(r"((?:ssh|ecdsa)-[\w@.-]+)\s+\d+\s+(SHA256:[A-Za-z0-9+/=]+)", log)
        if m:
            found.append(HostKey(m.group(1), "", m.group(2)))
    return found


def _keys_via_ssh(profile: Profile) -> List[HostKey]:
    import subprocess
    import tempfile

    found: List[HostKey] = []
    who = (f"{profile.ssh_user}@" if profile.ssh_user else "") + profile.ssh_host
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "known_hosts"
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
               "-o", f"UserKnownHostsFile={scratch}", "-o", "PasswordAuthentication=no",
               "-o", "PubkeyAuthentication=no", "-o", "ConnectTimeout=10"]
        if profile.ssh_port and profile.ssh_port != 22:
            cmd += ["-p", str(profile.ssh_port)]
        cmd += [who, "exit"]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                           stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return found
        if scratch.is_file():
            for line in scratch.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 3 and not line.startswith("#"):
                    try:
                        found.append(HostKey(parts[1], parts[2], _fingerprint(parts[2])))
                    except Exception:
                        continue
    return found


def login_by_hand(profile: Profile) -> int:
    """
    Run the by-hand login right here, in this terminal, with nothing
    captured: ssh (or plink) shows the server's key and asks whether to
    trust it, exactly as it would at a shell prompt, and remembers the
    answer where it always does. For when the key could not be fetched to
    show inside the app. Returns the program's exit code.
    """
    import subprocess

    who = (f"{profile.ssh_user}@" if profile.ssh_user else "") + profile.ssh_host
    key = str(Path(profile.ssh_key).expanduser()) if profile.ssh_key else ""
    if uses_putty(profile):
        cmd = [plink_path() or "plink", "-ssh"]
        if profile.ssh_port and profile.ssh_port != 22:
            cmd += ["-P", str(profile.ssh_port)]
        cmd += ["-i", key, who, "exit"]
    else:
        cmd = ["ssh"]
        if profile.ssh_port and profile.ssh_port != 22:
            cmd += ["-p", str(profile.ssh_port)]
        if key:
            cmd += ["-i", key]
        cmd += [who, "exit"]
    try:
        return subprocess.run(cmd).returncode
    except OSError:
        return 255


def known_hosts_path() -> Path:
    return Path.home() / ".ssh" / "known_hosts"


def accept_host_keys(profile: Profile, keys: Sequence[HostKey], *,
                     save: bool = True) -> Profile:
    """
    Trust these keys for this host from now on: OpenSSH's way is a line
    per key in ``~/.ssh/known_hosts``, the same line ``ssh`` would have
    written after a "yes"; plink's is a fingerprint on the profile, handed
    to it as ``-hostkey`` every time. Returns the profile to use (changed
    only for plink).
    """
    keys = [k for k in keys if k.fingerprint]
    if not keys:
        raise DbError(f"nothing to accept: {profile.ssh_host} showed no key")
    if uses_putty(profile):
        have = [f for f in profile.ssh_hostkey.split(",") if f.strip()]
        fps = have + [k.fingerprint for k in keys if k.fingerprint not in have]
        updated = replace(profile, ssh_hostkey=",".join(fps))
        if save:
            save_profile(updated)
        return updated
    port = profile.ssh_port or 22
    name = profile.ssh_host if port == 22 else f"[{profile.ssh_host}]:{port}"
    path = known_hosts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = [f"{name} {k.type} {k.blob}" for k in keys if k.blob]
    if not lines:
        raise DbError(f"{profile.ssh_host}'s key could not be fetched to record it "
                      f"(ssh-keyscan is missing); log in once by hand with "
                      f"`{ssh_hint(profile)}` instead.")
    new = [ln for ln in lines if ln not in existing.splitlines()]
    if new:
        with path.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write("\n".join(new) + "\n")
    return profile


def _stall_words(log: str, p: Profile, timeout: float) -> str:
    """Where a tunnel that never came up got to, from ssh's own account."""
    tool = "plink" if uses_putty(p) else "ssh"
    low = log.lower()
    logged_in = "access granted" in low or "authenticated to" in low \
        or "authentication succeeded" in low
    connected = "connected to" in low or "connection established" in low \
        or "server version" in low or "using ssh protocol" in low
    lines = [ln for ln in log.strip().splitlines() if ln.strip()]
    tail = [ln for ln in lines if not ln.lower().startswith("debug")][-4:] or lines[-2:]
    story = ("\n    ".join(tail)) if tail else f"{tool} said nothing"
    if logged_in:
        head = (f"{tool} logged in to {p.ssh_address}, but the tunnel to "
                f"{p.host or 'localhost'}:{p.port or DEFAULT_PORT[p.dialect]} never opened "
                f"in {int(timeout)} seconds. The server may not allow port forwarding "
                f"(AllowTcpForwarding), or something on this computer may be blocking "
                f"the local port.")
    elif connected:
        head = (f"{tool} reached {p.ssh_address} but did not finish logging in within "
                f"{int(timeout)} seconds. That usually means it is waiting on something it "
                f"cannot answer: a passphrase (load the key into "
                f"{'Pageant' if tool == 'plink' else 'your ssh agent'}), or a second "
                f"factor the server asks for interactively.")
    else:
        head = (f"{tool} did not reach {p.ssh_address} within {int(timeout)} seconds. If "
                f"it is only reachable on campus, connect to the VPN first.")
    return f"{head} Its last words:\n    {story}"


def close_tunnels() -> None:
    for name, (proc, _port) in list(_TUNNELS.items()):
        try:
            proc.kill()
        except Exception:
            pass
        _TUNNELS.pop(name, None)


# an ssh left running after Taters exits would hold the port and the login
# open for nobody; the interpreter takes the tunnels down with it
atexit.register(close_tunnels)


def ssh_hint(p: Profile) -> str:
    """The command a person would type to log in to this host by hand:
    ``ssh -p 2200 me@host``, or with PuTTY ``plink -P 2200 -i key.ppk me@host``
    -- the port goes on a flag, not after a colon."""
    who = (f"{p.ssh_user}@" if p.ssh_user else "") + p.ssh_host
    if uses_putty(p):
        port = f"-P {p.ssh_port} " if p.ssh_port and p.ssh_port != 22 else ""
        return f"plink {port}-i {Path(p.ssh_key).name} {who}"
    port = f"-p {p.ssh_port} " if p.ssh_port and p.ssh_port != 22 else ""
    return f"ssh {port}{who}"


def _ssh_words(err: str, p: Profile) -> str:
    """ssh's (or plink's) complaint, in a sentence that says what to do."""
    low = err.lower()
    where = p.ssh_address
    tool = "plink" if uses_putty(p) else "ssh"
    # the log is verbose on purpose (see tunnel_command); the lines a person
    # should see are the ones that are not ssh talking to itself
    spoken = [ln for ln in err.strip().splitlines()
              if ln.strip() and not ln.lower().startswith("debug")]
    last = spoken[-1] if spoken else "no message"
    # the key file itself first: plink says "unable to use key file" and
    # then, having nothing left to try, "no supported authentication"
    if "unable to use key file" in low or "passphrase" in low:
        first = next((ln for ln in err.strip().splitlines()
                      if "unable to use key" in ln.lower() or "passphrase" in ln.lower()),
                     next((ln for ln in err.strip().splitlines() if "key file" in ln.lower()),
                          last))
        return (f"{tool} could not use the key {p.ssh_key}: {first.strip()}. A key with "
                f"a passphrase has to be loaded in "
                f"{'Pageant' if tool == 'plink' else 'your ssh agent'} first, since "
                f"Taters cannot type it in; a key that cannot be opened is the wrong file.")
    if "permission denied" in low or "server refused our key" in low \
            or "no supported authentication" in low:
        return (f"{tool} could not log in to {where}: the key was not accepted. Taters "
                f"uses your keys exactly as {tool} does, never a typed password; check "
                f"that `{ssh_hint(p)}` works without a prompt"
                + (f", and that {p.ssh_key} is the right key." if p.ssh_key else "."))
    if "host key is not cached" in low or "host key verification failed" in low:
        return (f"{tool} does not know {p.ssh_host} yet. Log in once by hand"
                + (" with PuTTY" if tool == "plink" else f" with `{ssh_hint(p)}`")
                + " to accept its key, then try again.")
    if "could not resolve" in low or "no route" in low or "connection refused" in low \
            or "timed out" in low or "network is unreachable" in low \
            or "network error" in low or "unable to open connection" in low \
            or "host does not exist" in low:
        return (f"Could not reach {where}: {last}. If it is only reachable on campus, "
                f"connect to the VPN first.")
    if "forward" in low and ("failed" in low or "refused" in low or "prohibited" in low):
        return (f"{tool} logged in to {where}, but it refused to forward a port to "
                f"{p.host or 'localhost'}:{p.port or DEFAULT_PORT[p.dialect]}. The database "
                f"may listen on another port there, or the server may not allow "
                f"tunnels (AllowTcpForwarding).")
    return f"{tool} to {where} failed: {last}"


def _endpoint(profile: Profile) -> Tuple[str, int]:
    """Where the driver connects: the database itself, or the near end of
    its tunnel."""
    if profile.tunneled:
        return "127.0.0.1", open_tunnel(profile)
    return profile.host or "localhost", profile.port or DEFAULT_PORT[profile.dialect]


def _ssl_context(profile: Profile):
    """The TLS context a server connection uses, or None for ``ssl: off``.

    ``required`` encrypts and trusts whatever certificate the server shows,
    which stops a passive listener and nothing more; ``verify`` checks the
    certificate against the system's trust store, or ``ssl_ca``, and the
    host name, which is what stops an impostor."""
    import ssl as _ssl

    if profile.ssl == "off" or profile.tunneled:
        return None
    ctx = _ssl.create_default_context(cafile=profile.ssl_ca or None)
    if profile.ssl == "verify":
        ctx.check_hostname = True
        ctx.verify_mode = _ssl.CERT_REQUIRED
    else:
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
    return ctx


_DIALECTS = {"sqlite": Dialect(), "mysql": MySQLDialect(), "postgres": PostgresDialect()}


def dialect_for(name: str) -> Dialect:
    return _DIALECTS[name]


def _not_a_database(e: Exception, p: Profile, port: int) -> Optional[str]:
    """When the port answered, but not in the database's protocol."""
    text = str(e)
    if "Packet sequence number wrong" not in text and "server sent" not in text:
        return None
    # the fourth byte of what came back is read as a sequence number; an
    # SSH server's banner starts "SSH-", whose fourth byte is 45
    who = "an SSH server" if "got 45" in text else "something that is not a database"
    return (f"{p.host} answered on port {port}, but as {who}, not as {p.engine_label}. "
            f"Check the port. If the database is only reachable through an SSH "
            f"tunnel, open the tunnel first (for example ssh -L {port}:localhost:{port} "
            f"you@{p.host}) and give Taters localhost as the address.")


def _nothing_behind_the_tunnel(p: Profile) -> str:
    return (f"ssh logged in to {p.ssh_address}, but nothing answered at "
            f"{p.host or 'localhost'}:{p.port or DEFAULT_PORT[p.dialect]} on that machine. "
            f"Check the database's address as seen from there (often localhost) and its "
            f"port -- on the machine, `ss -ltn` or the database's own config says where "
            f"it listens.")


def _mysql_words(e: Exception, p: Profile) -> str:
    code = getattr(e, "args", [None])[0]
    odd = _not_a_database(e, p, p.port or DEFAULT_PORT["mysql"])
    if odd:
        return odd
    if p.tunneled and code in (2013, 2006, 2003):
        return _nothing_behind_the_tunnel(p)
    if code in (1045,):
        return f"{p.name} did not accept that password for {p.user}."
    if code in (3159,) or "SSL" in str(e).upper() and "not" in str(e).lower():
        return _no_ssl(p)
    if code in (2003, 2005):
        return (f"Could not reach {p.host} on port {p.port or 3306}. If the database "
                f"is only reachable on campus, connect to the VPN first.")
    if code in (1049,):
        return f"{p.host} has no database called {p.database}."
    return f"{p.name}: {e}"


def _no_ssl(p: Profile) -> str:
    return (f"{p.host} does not offer an encrypted connection, and {p.name} is set to "
            f"require one. For a server inside a trusted network, the connection can "
            f"be set to go without encryption.")


def _postgres_words(e: Exception, p: Profile) -> str:
    text = str(e)
    odd = _not_a_database(e, p, p.port or DEFAULT_PORT["postgres"])
    if odd:
        return odd
    if p.tunneled and ("closed the connection" in text or "connection refused" in text.lower()
                       or "could not connect" in text):
        return _nothing_behind_the_tunnel(p)
    if "password authentication failed" in text:
        return f"{p.name} did not accept that password for {p.user}."
    if "server does not support SSL" in text or "SSL is not enabled" in text:
        return _no_ssl(p)
    if "could not connect" in text or "Connection refused" in text or "timeout" in text:
        return (f"Could not reach {p.host} on port {p.port or 5432}. If the database "
                f"is only reachable on campus, connect to the VPN first.")
    if "does not exist" in text and "database" in text:
        return f"{p.host} has no database called {p.database}."
    return f"{p.name}: {text.strip()}"


# ---------------------------------------------------------------------------
# A session: a connection with its dialect
# ---------------------------------------------------------------------------

class Session:
    """One open connection, its dialect, and the small verbs the code uses."""

    def __init__(self, profile: Profile, con, dialect: Dialect):
        self.profile = profile
        self.con = con
        self.d = dialect

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    @contextlib.contextmanager
    def _said(self):
        """A driver's exception becomes a :class:`DbError` that says, in
        words, what the server said and whose problem it is. A programming
        error (a bad parameter, a bug) is not the server's and passes."""
        try:
            yield
        except DbError:
            raise
        except self.d.driver_errors() as e:
            raise DbError(query_words(e, self.profile)) from e

    def run(self, sql: str, params: Sequence[Any] = ()) -> None:
        cur = self.con.cursor()
        try:
            with self._said():
                cur.execute(self.d.translate(sql), tuple(params))
        finally:
            cur.close()

    def all(self, sql: str, params: Sequence[Any] = ()) -> List[tuple]:
        cur = self.con.cursor()
        try:
            with self._said():
                cur.execute(self.d.translate(sql), tuple(params))
                return [tuple(r) for r in cur.fetchall()]
        finally:
            cur.close()

    def one(self, sql: str, params: Sequence[Any] = ()) -> Any:
        rows = self.all(sql, params)
        return rows[0][0] if rows else None

    def stream(self, sql: str, params: Sequence[Any] = (),
               batch: int = _BATCH) -> Iterator[tuple]:
        """Rows one at a time without holding the result set."""
        with self.d.stream_scope(self.con):
            cur = self.d.stream_cursor(self.con)
            try:
                with self._said():
                    cur.execute(self.d.translate(sql), tuple(params))
                    while True:
                        chunk = cur.fetchmany(batch)
                        if not chunk:
                            break
                        for row in chunk:
                            yield tuple(row)
            finally:
                cur.close()

    @contextlib.contextmanager
    def read_only(self):
        """Run a block with the session unable to write: a query somebody
        typed into a YAML file gets exactly the rights a SELECT needs."""
        self.run(self.d.read_only_on)
        try:
            yield self
        finally:
            try:
                self.run(self.d.read_only_off)
            except Exception:
                pass

    def columns_of_query(self, sql: str, params: Sequence[Any] = ()) -> List[str]:
        cur = self.con.cursor()
        try:
            with self._said():
                cur.execute(self.d.translate(self.d.limit(f"SELECT * FROM ({sql}) AS q", 0)),
                            tuple(params))
                return [str(c[0]) for c in (cur.description or [])]
        finally:
            cur.close()

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        cur = self.con.cursor()
        try:
            with self._said():
                self.d.executemany(cur, self.d.translate(sql), [tuple(r) for r in rows])
        finally:
            cur.close()


def connect(profile: Profile, password: Optional[str] = None,
            ask: Optional[Callable[[str], str]] = None,
            on_step: Optional[Callable[[str], None]] = None) -> Session:
    """
    Open a session. Refuses to run inside a pool worker: the parent reads
    the rows and ships text, so a worker never needs, or holds, a password.
    ``on_step`` hears what is being done as it is done -- a tunnel being
    opened, a login -- so a screen never just sits there.
    """
    if multiprocessing.parent_process() is not None:
        raise DbError("a database is read by the main process only; workers "
                      "receive rows from it")
    d = dialect_for(profile.dialect)
    if password is None:
        password = password_for(profile, ask)
    say = on_step or (lambda text: None)
    if profile.tunneled:
        tunnel = _TUNNELS.get(profile.name)
        if tunnel is None or tunnel[0].poll() is not None:
            say(f"Opening the ssh tunnel to {profile.ssh_address}")
            open_tunnel(profile)
    if profile.dialect == "sqlite":
        say(f"Opening {profile.where}")
    else:
        say(f"Logging in to {profile.database or 'the server'} as {profile.user}")
    session = Session(profile, d.connect(profile, password), d)
    if profile.dialect != "sqlite" and profile.ssl != "off" and not profile.tunneled:
        # the driver can be talked out of TLS by a server (or a man in the
        # middle) that claims not to support it; a profile that asked for
        # encryption gets it or gets nothing
        if d.encrypted(session) is False:
            session.close()
            raise DbError(_no_ssl(profile))
    return session


# ---------------------------------------------------------------------------
# Probe: what the account may do, by trying
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    select: bool = False
    create: bool = False
    drop: bool = False
    registry: bool = False
    version: str = ""
    tables: int = 0
    encrypted: Optional[bool] = None
    notes: Tuple[str, ...] = ()

    @property
    def can_read(self) -> bool:
        return self.select

    @property
    def can_write(self) -> bool:
        return self.select and self.create and self.drop and self.registry

    def sentences(self, profile: Profile) -> List[str]:
        """The test's read-back, one line each."""
        if not self.select:
            return [f"Could not read from {profile.name}: {'; '.join(self.notes) or 'no permission to select'}."]
        enc = {True: "yes", False: "no", None: "n/a (a file)"}[self.encrypted]
        head = (f"Connected to {profile.name} ({self.version}, {profile.where})"
                + (f" as {profile.user}" if profile.user else "") + f". Encrypted: {enc}.")
        if profile.dialect != "sqlite" and profile.database:
            head = (f"Connected to {profile.name} ({self.version}, {profile.server}) as "
                    f"{profile.user}, on database {profile.database}. Encrypted: {enc}.")
        out = [head, f"Can read: yes — {self.tables} table(s).",
               f"Can create tables: {'yes' if self.create else 'no'}.",
               f"Can replace tables it made: {'yes' if self.drop else 'no'}."]
        if not self.can_write:
            out.append(f"Wrangle data can copy tables out of {profile.name} into "
                       f"spreadsheets, and a run can read from it; its results will "
                       f"have to go to files or to a database that allows new tables."
                       + (f" Ask whoever manages the database to grant CREATE on "
                          f"{profile.database}, or add a second connection that has it."
                          if profile.dialect != "sqlite" else ""))
        elif not self.drop:
            out.append("Runs work. A re-run writes new tables beside the old ones "
                       "instead of replacing them.")
        return out


def probe(profile: Profile, password: Optional[str] = None,
          ask: Optional[Callable[[str], str]] = None,
          on_step: Optional[Callable[[str], None]] = None) -> Probe:
    """Try each thing a run needs and say which worked, by name. ``on_step``
    hears each thing as it is tried, so nothing here is a silent wait."""
    notes: List[str] = []
    say = on_step or (lambda text: None)
    try:
        session = connect(profile, password, ask, on_step=on_step)
    except HostKeyUnknown:
        raise                       # the one refusal a person can settle on the spot
    except DbError as e:
        return Probe(notes=(str(e),))
    with session:
        d = session.d
        try:
            say("Asking the server its version")
            session.one("SELECT 1")
            version = d.version(session)
            say("Counting the tables this account can see")
            tables = d.count_tables(session, profile.schema)
            encrypted = d.encrypted(session)
        except Exception as e:
            return Probe(notes=(f"{type(e).__name__}: {e}",))
        name = f"_taters_probe_{_secrets.token_hex(4)}"
        create = drop = False
        try:
            say("Checking whether this account can create a table")
            d.create(session, name, [("x", d.text_type)], schema=profile.schema)
            create = True
        except Exception as e:
            notes.append(f"create: {_short(e)}")
        if create:
            try:
                say("...and remove it again")
                d.drop(session, name, profile.schema)
                drop = True
            except Exception as e:
                notes.append(f"drop: {_short(e)}")
        registry = False
        if create:
            try:
                say("Setting up the small registry Taters keeps of its tables")
                ensure_registry(session)
                registry = True
            except Exception as e:
                notes.append(f"registry: {_short(e)}")
    return Probe(select=True, create=create, drop=drop, registry=registry,
                 version=version, tables=tables, encrypted=encrypted,
                 notes=tuple(notes))


def query_words(e: Exception, p: Profile) -> str:
    """
    What the server said while a query ran, in words a person can act on.
    The login explainers cover not getting in; this covers the things that
    go wrong once in -- the server's disk filling up, the connection
    dropping, a table the account may not read -- which are almost always
    somebody else's to fix, and the sentence says whose.
    """
    text = str(e)
    low = text.lower()
    code = e.args[0] if getattr(e, "args", None) and isinstance(e.args[0], int) else None
    ask = ("Ask whoever runs the database server" if p.dialect != "sqlite"
           else "Free some space on this computer")
    if (code in (1021, 1114) or "errno: 28" in low or "no space left" in low
            or "disk is full" in low or "disk full" in low):
        return (f"{p.engine_label} on {p.name} has run out of disk space"
                + (" (its temp space, which it needs to answer a query)"
                   if code == 1021 or "/tmp" in text else "")
                + f". Nothing is wrong on this computer. {ask}; until then "
                f"the database cannot answer.")
    if code in (2013, 2006) or "server has gone away" in low or \
            "closed the connection unexpectedly" in low or "connection was killed" in low:
        return (f"The connection to {p.name} dropped mid-query"
                + (" (the ssh tunnel may have closed)" if p.tunneled else "")
                + ". Try again; if it keeps happening on one table, the server may "
                "be killing long queries.")
    if code in (1142, 1143, 1044, 1045) or "permission denied" in low or "access denied" in low \
            or "command denied" in low:
        return (f"{p.user or 'This account'} is not allowed to do that on {p.name}: "
                f"{_short(e)}. Ask whoever runs the database for the right.")
    if code in (1205, 1213) or "lock wait timeout" in low or "deadlock" in low \
            or "database is locked" in low:
        return (f"{p.name} is busy: another session holds a lock on what this needs. "
                f"Try again in a moment.")
    if code in (1146,) or ("no such table" in low) or ("does not exist" in low and
                                                       "relation" in low):
        return f"{p.name} has no such table any more: {_short(e)}."
    return f"{p.engine_label} on {p.name} said: {_short(e)}"


def _short(e: Exception) -> str:
    return str(e).strip().splitlines()[0][:160] if str(e).strip() else type(e).__name__


# ---------------------------------------------------------------------------
# The registry, names, and the writer
# ---------------------------------------------------------------------------

def ensure_registry(session: Session) -> None:
    d, schema = session.d, session.profile.schema
    long_text = d.text_type
    if not d.table_exists(session, REGISTRY, schema):
        session.run(f"CREATE TABLE {d.qualified(REGISTRY, schema)} ("
                    f"{d.quote('table_name')} VARCHAR(64) PRIMARY KEY, "
                    f"{d.quote('logical')} VARCHAR(512), {d.quote('layout')} VARCHAR(8), "
                    f"{d.quote('columns')} {long_text}, {d.quote('n_rows')} BIGINT, "
                    f"{d.quote('fingerprint')} VARCHAR(64), {d.quote('written_at')} VARCHAR(32), "
                    f"{d.quote('taters_version')} VARCHAR(32))")
    if not d.table_exists(session, SIDECARS, schema):
        session.run(f"CREATE TABLE {d.qualified(SIDECARS, schema)} ("
                    f"{d.quote('table_name')} VARCHAR(64), {d.quote('kind')} VARCHAR(32), "
                    f"{d.quote('doc')} {long_text}, PRIMARY KEY ({d.quote('table_name')}, "
                    f"{d.quote('kind')}))")


def sanitize_table_name(logical: str) -> str:
    """
    ``taters/study7/features/readability`` -> ``taters__study7__features__readability``.

    Lowercase letters, digits and underscores; past 48 characters the name
    is shortened deterministically with a hash, never by chance.
    """
    parts = [p for p in str(logical).replace("\\", "/").split("/") if p]
    clean = []
    for p in parts:
        s = re.sub(r"[^a-z0-9_]+", "_", p.lower()).strip("_") or "t"
        if s[0].isdigit():
            s = "t_" + s
        clean.append(s)
    name = "__".join(clean) or "t"
    if len(name) > _NAME_LIMIT:
        name = name[:_NAME_LIMIT - 9] + "_" + hashlib.sha1(logical.encode("utf-8")).hexdigest()[:8]
    return name


@dataclass
class Registered:
    """One row of the registry, as Python."""

    table_name: str
    logical: str
    layout: str
    columns: List[List[str]]          # [[logical, physical, type], ...]
    n_rows: int
    fingerprint: str
    written_at: str
    taters_version: str = ""

    @property
    def logical_columns(self) -> List[str]:
        return [c[0] for c in self.columns]

    @property
    def physical_columns(self) -> List[str]:
        return [c[1] for c in self.columns]

    @property
    def table_columns(self) -> List[str]:
        """The columns the table really has, besides the row number: every
        physical column, plus the packed column when there is one."""
        real = [c[1] for c in self.columns if c[1]]
        return real + [PACKED_COLUMN] if self.layout == "packed" else real


def registered(session: Session, physical: str) -> Optional[Registered]:
    d, schema = session.d, session.profile.schema
    if not d.table_exists(session, REGISTRY, schema):
        return None
    rows = session.all(
        f"SELECT table_name, logical, layout, columns, n_rows, fingerprint, written_at, "
        f"taters_version FROM {d.qualified(REGISTRY, schema)} WHERE table_name = ?",
        (physical,))
    if not rows:
        return None
    r = rows[0]
    try:
        cols = json.loads(r[3] or "[]")
    except ValueError:
        cols = []
    return Registered(table_name=str(r[0]), logical=str(r[1] or ""), layout=str(r[2] or "wide"),
                      columns=cols, n_rows=int(r[4] or 0), fingerprint=str(r[5] or ""),
                      written_at=str(r[6] or ""), taters_version=str(r[7] or ""))


def registered_under(session: Session, prefix: str) -> List[Registered]:
    """Every registered table whose logical path starts with ``prefix``."""
    d, schema = session.d, session.profile.schema
    if not d.table_exists(session, REGISTRY, schema):
        return []
    like = prefix.rstrip("/") + "/%"
    rows = session.all(f"SELECT table_name FROM {d.qualified(REGISTRY, schema)} WHERE "
                       f"logical LIKE ? ORDER BY logical", (like,))
    out = []
    for (name,) in rows:
        r = registered(session, str(name))
        if r:
            out.append(r)
    return out


def _column_name(name: str, ordinal: int, taken: set) -> str:
    """A physical column name: verbatim when it fits, else a stable alias."""
    s = str(name)
    ok = (0 < len(s.encode("utf-8")) <= 60 and "\x00" not in s and not s.endswith(" ")
          and s.lower() not in taken)
    if not ok:
        s = f"c{ordinal}_{hashlib.sha1(str(name).encode('utf-8')).hexdigest()[:6]}"
    taken.add(s.lower())
    return s


class _TypeTracker:
    """Per-column: could every non-blank cell be an integer? a number?"""

    __slots__ = ("is_int", "is_num", "filled", "maxlen")

    def __init__(self, n: int):
        self.is_int = [True] * n
        self.is_num = [True] * n
        self.filled = [0] * n
        self.maxlen = [0] * n

    def see(self, i: int, cell: str) -> None:
        v = cell.strip()
        if not v:
            return
        self.filled[i] += 1
        if len(v) > self.maxlen[i]:
            self.maxlen[i] = len(v)
        if self.is_int[i] and not _INTEGER.match(v):
            self.is_int[i] = False
        if self.is_num[i] and not (_NUMBER.match(v) or v.lower() in _NON_FINITE):
            self.is_num[i] = False

    def kind(self, i: int) -> str:
        if not self.filled[i]:
            return "text"
        if self.is_int[i]:
            return "int"
        if self.is_num[i]:
            return "float"
        return "text"


class Loader:
    """
    Rows pushed in one at a time; a typed table swapped in at the end.

    ``add`` takes a row, ``finish`` makes the table and returns its registry
    entry, ``abort`` drops the scratch. Nothing is buffered beyond one
    insert batch, so a table of any size costs the same memory.

    ``hints`` names columns whose type is known in advance (``"text"``,
    ``"int"``, ``"float"``); when every column is hinted the rows go
    straight into the typed table. Otherwise they load as text and are
    copied with casts on the server, so a 500k-row table is never read back
    into Python to learn its types. Refuses to replace a table Taters did
    not make.
    """

    def __init__(self, session: Session, logical: str, columns: Sequence[str], *,
                 hints: Optional[Mapping[str, str]] = None,
                 on_rows: Optional[Callable[[int], None]] = None,
                 unique_key: str = ""):
        d, schema = session.d, session.profile.schema
        self.session, self.d, self.schema = session, d, schema
        self.logical = logical
        self.unique_key = str(unique_key or "")
        if self.unique_key and self.unique_key not in [str(c) for c in columns]:
            raise DbError(f"{logical} has no column {self.unique_key!r} to make its key")
        self.physical = sanitize_table_name(logical)
        self.columns = [str(c) for c in columns]
        self.on_rows = on_rows
        self.hints = {str(k): str(v) for k, v in (hints or {}).items()}
        # past the engine's column limit the measures go into one JSON
        # column and only the identity columns stay real. which ones are
        # "identity" has to be decided up front, before a row is seen: the
        # ids by name, and whatever the caller hinted is text
        self.packed = len(self.columns) + 1 > d.max_columns
        if self.packed:
            self.real = [c for c in self.columns
                         if c in ID_COLUMNS or self.hints.get(c) == "text"]
            if len(self.real) + 2 > d.max_columns:
                raise DbError(f"{logical} has {len(self.real):,} text columns, more "
                              f"than {session.profile.engine_label} can hold in one "
                              f"table, and text cannot be packed")
            if self.unique_key and self.unique_key not in self.real:
                raise DbError(f"{logical} is too wide for {session.profile.engine_label} "
                              f"to hold as columns, and its key {self.unique_key!r} "
                              f"would be packed away with the measures")
        else:
            self.real = list(self.columns)
        self.real_index = [self.columns.index(c) for c in self.real]
        self.packed_index = [i for i, c in enumerate(self.columns) if c not in set(self.real)]
        ensure_registry(session)
        old = registered(session, self.physical)
        if d.table_exists(session, self.physical, schema) and \
                (old is None or old.logical != logical):
            raise DbError(f"{session.profile.name} already has a table called "
                          f"{self.physical}"
                          + (" that Taters did not make" if old is None
                             else f" holding {old.logical}")
                          + "; it will not be replaced. Pick another name.")
        taken: set = {ROW_COLUMN, PACKED_COLUMN}
        self.physical_cols = [_column_name(c, i, taken) for i, c in enumerate(self.real)]
        self.all_hinted = all(c in self.hints for c in self.real)
        # scratch names nobody could have made on purpose: fixed suffixes
        # (`posts__tnew`) could be somebody's real table, and dropping those
        # "to clear the way" was dropping their data
        tmp = f"_taters_tmp_{_secrets.token_hex(4)}"
        self.load = tmp + "_load"
        self.new = tmp + "_new"
        self.told = tmp + "_old"
        self.swapped = False
        self.kinds = [self.hints.get(c, "text") for c in self.real]
        self.fp = Fingerprint(self.columns)
        self.tracker = _TypeTracker(len(self.real))
        target = self.new if self.all_hinted else self.load
        extra = [(PACKED_COLUMN, d.text_type)] if self.packed else []
        if self.all_hinted:
            d.create(session, self.new,
                     [(p, self._typed(k)) for p, k in zip(self.physical_cols, self.kinds)]
                     + extra, schema=schema)
        else:
            d.create(session, self.load,
                     [(p, d.text_type) for p in self.physical_cols] + extra, schema=schema)
        stored = [ROW_COLUMN, *self.physical_cols] + ([PACKED_COLUMN] if self.packed else [])
        self.names = ", ".join(d.quote(c) for c in stored)
        marks = ", ".join("?" for _ in stored)
        self.insert = (f"INSERT INTO {d.qualified(target, schema)} ({self.names}) "
                       f"VALUES ({marks})")
        self.batch: List[tuple] = []
        self.n = 0
        self.done = False

    def _typed(self, kind: str) -> str:
        d = self.d
        return {"int": d.int_type if d.name != "mysql" else "BIGINT",
                "float": d.float_type, "text": d.text_type}[kind]

    def add(self, row: Sequence[Any]) -> None:
        cells = [cell_text(v) for v in row]
        if len(cells) != len(self.columns):
            raise DbError(f"a row for {self.logical} has {len(cells)} cells, not "
                          f"{len(self.columns)}")
        self.fp.add(cells)
        real = [cells[i] for i in self.real_index] if self.packed else cells
        for i, cell in enumerate(real):
            self.tracker.see(i, cell)
        self.n += 1
        if self.all_hinted:
            values = [_typed_value(c, k) for c, k in zip(real, self.kinds)]
        else:
            values = [c if c != "" else None for c in real]
        if self.packed:
            values.append(json.dumps([_packed_cell(cells[i]) for i in self.packed_index],
                                     ensure_ascii=False, separators=(",", ":")))
        self.batch.append((self.n, *values))
        if len(self.batch) >= _BATCH:
            self._flush()

    def _flush(self) -> None:
        if self.batch:
            self.session.executemany(self.insert, self.batch)
            self.batch.clear()
        if self.on_rows is not None:
            self.on_rows(self.n)

    def abort(self) -> None:
        d, session, schema = self.d, self.session, self.schema
        for scratch in (self.load, self.new):
            try:
                d.drop(session, scratch, schema)
            except Exception:
                pass
        # a swap that died halfway has moved the old table aside; put it back
        try:
            if d.table_exists(session, self.told, schema) and \
                    not d.table_exists(session, self.physical, schema):
                d.swap(session, self.physical, self.told, self.told + "_x", schema)
            elif d.table_exists(session, self.told, schema):
                d.drop(session, self.told, schema)
        except Exception:
            pass

    def finish(self) -> Registered:
        from .. import __version__

        d, schema, session = self.d, self.schema, self.session
        self._flush()
        if not self.all_hinted:
            self.kinds = [self.hints.get(c) or self.tracker.kind(i)
                          for i, c in enumerate(self.real)]
            # an id is text whatever it looks like: "1" and "01" are two rows
            self.kinds = ["text" if c in ID_COLUMNS else k
                          for c, k in zip(self.real, self.kinds)]
            d.create(session, self.new,
                     [(p, self._typed(k)) for p, k in zip(self.physical_cols, self.kinds)]
                     + ([(PACKED_COLUMN, d.text_type)] if self.packed else []), schema=schema)
            selects = [d.quote(ROW_COLUMN)]
            for p, k in zip(self.physical_cols, self.kinds):
                selects.append(d.cast(p, k) if k != "text" else d.quote(p))
            if self.packed:
                selects.append(d.quote(PACKED_COLUMN))
            session.run(f"INSERT INTO {d.qualified(self.new, schema)} ({self.names}) "
                        f"SELECT {', '.join(selects)} FROM {d.qualified(self.load, schema)}")
            d.drop(session, self.load, schema)
        if "text_id" in self.real:
            try:
                d.index(session, self.new,
                        self.physical_cols[self.real.index("text_id")], schema)
            except Exception:
                pass                # an index is a nicety; the table is the point
        if self.unique_key:
            # this one is a promise, not a nicety: a repeated value has to stop
            # the write, since the whole point of a key is that it identifies
            col = self.physical_cols[self.real.index(self.unique_key)]
            idx = f"{self.new}__{col}__uq"[:60]
            session.run(f"CREATE UNIQUE INDEX {d.quote(idx)} ON "
                        f"{d.qualified(self.new, schema)} ({d.quote(col)}"
                        f"{'(191)' if d.name == 'mysql' else ''})")
        # checked again, right before the swap: a table could have appeared
        # at this name since __init__ looked, and it is not ours to replace
        old = registered(session, self.physical)
        if d.table_exists(session, self.physical, schema) and \
                (old is None or old.logical != self.logical):
            self.abort()
            raise DbError(f"{session.profile.name} now has a table called {self.physical} "
                          f"that Taters did not make; nothing was replaced.")
        d.swap(session, self.physical, self.new, self.told, schema)
        self.swapped = True
        # the registry keeps every logical column in order; a packed one has
        # no physical column of its own, so its slot says so
        where = dict(zip(self.real, zip(self.physical_cols, self.kinds)))
        layout = "packed" if self.packed else "wide"
        entry = Registered(
            table_name=self.physical, logical=self.logical, layout=layout,
            columns=[[c, *where[c]] if c in where else [c, "", "packed"]
                     for c in self.columns],
            n_rows=self.n, fingerprint=self.fp.hexdigest(),
            written_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            taters_version=str(__version__))
        d.upsert(session, REGISTRY, {
            "table_name": self.physical, "logical": self.logical, "layout": layout,
            "columns": json.dumps(entry.columns, ensure_ascii=False), "n_rows": self.n,
            "fingerprint": entry.fingerprint, "written_at": entry.written_at,
            "taters_version": entry.taters_version}, keys=["table_name"], schema=schema)
        d.drop(session, self.told, schema)
        self.done = True
        return entry


def write_rows(session: Session, logical: str, columns: Sequence[str],
               rows: Iterable[Sequence[Any]], *,
               hints: Optional[Mapping[str, str]] = None,
               on_rows: Optional[Callable[[int], None]] = None) -> Registered:
    """Load ``rows`` into the table named by ``logical``; see :class:`Loader`."""
    loader = Loader(session, logical, columns, hints=hints, on_rows=on_rows)
    try:
        for row in rows:
            loader.add(row)
        return loader.finish()
    except BaseException:
        loader.abort()
        raise


def _packed_cell(cell: str) -> Any:
    """A cell as it goes into the packed JSON array: a number when it reads
    as one (and has no leading zero to lose), null when blank, else text."""
    v = cell.strip()
    if not v:
        return None
    if _INTEGER.match(v):
        return int(v)
    if _NUMBER.match(v):
        f = float(v)
        return f if math.isfinite(f) else None
    if v.lower() in _NON_FINITE:
        return None
    return cell


def _unpack(doc: Any, n: int) -> List[str]:
    """The packed array back to ``n`` text cells, blanks for anything short."""
    try:
        values = json.loads(doc) if isinstance(doc, (str, bytes, bytearray)) else (doc or [])
    except ValueError:
        values = []
    cells = [cell_text(v) for v in values]
    return (cells + [""] * n)[:n]


def _typed_value(cell: str, kind: str) -> Any:
    v = cell.strip() if kind != "text" else cell
    if kind == "text":
        return cell if cell != "" else None
    if not v or v.lower() in _NON_FINITE:
        return None
    if kind == "int":
        try:
            return int(v)
        except ValueError:
            return int(float(v))
    f = float(v)
    return f if math.isfinite(f) else None


def read_cells(session: Session, entry: Registered, *,
               columns: Optional[Sequence[str]] = None,
               on_rows: Optional[Callable[[int], None]] = None) -> Iterator[List[str]]:
    """The rows of a table Taters wrote, in write order, as lists of text
    cells in ``columns`` order (the table's own when none are named)."""
    d, schema = session.d, session.profile.schema
    wanted = [str(c) for c in columns] if columns else entry.logical_columns
    logical = entry.logical_columns
    missing = [c for c in wanted if c not in logical]
    if missing:
        raise DbError(f"{entry.logical} has no column {missing[0]!r}")
    phys = {c[0]: c[1] for c in entry.columns if c[1]}
    packed = entry.layout == "packed"
    packed_order = [c[0] for c in entry.columns if not c[1]] if packed else []
    # select what the table really has; the packed column is unpacked here
    # and the wanted cells picked out, so a reader never sees the layout
    real_wanted = [c for c in wanted if c in phys]
    need_packed = packed and any(c not in phys for c in wanted)
    selected = [d.quote(ROW_COLUMN), *(d.quote(phys[c]) for c in real_wanted)]
    if need_packed:
        selected.append(d.quote(PACKED_COLUMN))
    sql = (f"SELECT {', '.join(selected)} FROM {d.qualified(entry.table_name, schema)} "
           f"WHERE {d.quote(ROW_COLUMN)} > ? ORDER BY {d.quote(ROW_COLUMN)}")
    last = 0
    n = 0
    while True:
        rows = session.all(d.limit(sql, _BATCH), (last,))
        if not rows:
            break
        for row in rows:
            last = int(row[0])
            n += 1
            have = dict(zip(real_wanted, (cell_text(v) for v in row[1:1 + len(real_wanted)])))
            if need_packed:
                have.update(zip(packed_order, _unpack(row[-1], len(packed_order))))
            yield [have.get(c, "") for c in wanted]
        if on_rows is not None:
            on_rows(n)
    if on_rows is not None:
        on_rows(n)


def read_rows(session: Session, entry: Registered, *,
              columns: Optional[Sequence[str]] = None,
              on_rows: Optional[Callable[[int], None]] = None) -> Iterator[Dict[str, str]]:
    """The rows of a table Taters wrote, in write order, as dicts of text cells."""
    wanted = [str(c) for c in columns] if columns else entry.logical_columns
    for cells in read_cells(session, entry, columns=wanted, on_rows=on_rows):
        yield dict(zip(wanted, cells))


def fingerprint_table(session: Session, entry: Registered) -> str:
    """Recompute a written table's fingerprint from its rows."""
    fp = Fingerprint(entry.logical_columns)
    for cells in read_cells(session, entry):
        fp.add(cells)
    return fp.hexdigest()


def drop_table(session: Session, entry: Registered) -> None:
    """Remove a table Taters made, and its registry and sidecar rows."""
    d, schema = session.d, session.profile.schema
    d.drop(session, entry.table_name, schema)
    session.run(f"DELETE FROM {d.qualified(REGISTRY, schema)} WHERE table_name = ?",
                (entry.table_name,))
    session.run(f"DELETE FROM {d.qualified(SIDECARS, schema)} WHERE table_name = ?",
                (entry.table_name,))


def read_sidecar(session: Session, physical: str, kind: str) -> Optional[dict]:
    d, schema = session.d, session.profile.schema
    if not d.table_exists(session, SIDECARS, schema):
        return None
    doc = session.one(f"SELECT doc FROM {d.qualified(SIDECARS, schema)} WHERE "
                      f"table_name = ? AND kind = ?", (physical, kind))
    if doc is None:
        return None
    try:
        parsed = json.loads(doc)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def write_sidecar(session: Session, physical: str, kind: str, doc: Mapping) -> None:
    ensure_registry(session)
    session.d.upsert(session, SIDECARS,
                     {"table_name": physical, "kind": kind,
                      "doc": json.dumps(doc, sort_keys=True, ensure_ascii=False)},
                     keys=["table_name", "kind"], schema=session.profile.schema)


def delete_sidecar(session: Session, physical: str, kind: str) -> None:
    d, schema = session.d, session.profile.schema
    if d.table_exists(session, SIDECARS, schema):
        session.run(f"DELETE FROM {d.qualified(SIDECARS, schema)} WHERE table_name = ? "
                    f"AND kind = ?", (physical, kind))


def stream_query(session: Session, sql: str, params: Sequence[Any] = (), *,
                 on_rows: Optional[Callable[[int], None]] = None
                 ) -> Tuple[List[str], Iterator[Dict[str, str]]]:
    """A query's column names and its rows as text cells, streamed."""
    columns = session.columns_of_query(sql, params)

    def rows() -> Iterator[Dict[str, str]]:
        n = 0
        for row in session.stream(sql, params):
            n += 1
            yield {c: cell_text(v) for c, v in zip(columns, row)}
            if on_rows is not None and n % _BATCH == 0:
                on_rows(n)
        if on_rows is not None:
            on_rows(n)
    return columns, rows()


def session_for(name: str, database: str = "",
                ask: Optional[Callable[[str], str]] = None) -> Session:
    """
    A session by connection name, on one of its databases; the usual entry
    from the table layer. A connection is a server and a login; which
    database to use on it is chosen each time, so it is passed here. Empty
    means the connection's own default (a SQLite file has only the one).
    """
    profile = get_profile(name)
    return connect(for_database(profile, database), ask=ask)


def for_database(profile: Profile, database: str = "") -> Profile:
    """The profile pointed at one database on its server. ``main`` and ""
    mean the file itself for SQLite; "" means the saved default otherwise."""
    if profile.dialect == "sqlite":
        return profile if profile.database == "main" else with_fields(profile, database="main")
    if not database or database == profile.database:
        return profile
    return with_fields(profile, database=database)


def database_of(profile: Profile) -> str:
    """The database segment of a table address on this profile: ``main``
    for a file, else whatever database the profile is pointed at."""
    return "main" if profile.dialect == "sqlite" else profile.database


def describe_profile(profile: Profile) -> dict:
    return {"name": profile.name, **profile.to_dict()}


def with_fields(profile: Profile, **changes) -> Profile:
    return replace(profile, **changes)


_ = field  # keep the import used for dataclass defaults in future edits
