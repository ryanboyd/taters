"""
"Database connections" -- where the databases are and how Taters logs in.

A connection is a name, an engine, where the server is, which database and
which user. It is saved so nobody types a hostname twice. The password is
not saved, ever: Taters asks for it, with dots, once per session, or reads
it from ``TATERS_DB_PASSWORD_<NAME>`` for scripts. That is a concrete choice
rather than caution for its own sake -- the settings file lives in a folder
people sync, and a synced password is a leaked one.

"Test it" does not read the account's grants, it tries: can it read, can it
make a table, can it drop one. Then it says, in a sentence, what that means
for a run -- because a person who can only read should learn that here, not
at the end of a long extraction.
"""
from __future__ import annotations

import importlib.util
import re
from typing import Any, Dict, List, Optional

from ...helpers import db
from .. import glyphs
from ..prompts import Choice, GoBack, Prompter
from . import Task, TaskContext

__all__ = ["TASK", "add_connection", "test_connection", "driver_missing", "INSTALL_DRIVERS"]

_BACK = "back"
_ADD = "\x00add"


class _Abandon(Exception):
    """"Never mind" on the connection attempt: the whole add is dropped."""

_KINDS = (
    ("sqlite", "A SQLite file", "One file on this computer or a shared drive -- .sqlite, .db"),
    ("mysql", "MySQL or MariaDB", "A server, a database on it, a username"),
    ("postgres", "PostgreSQL", "A server, a database on it, a username"),
)

_DRIVER = {"mysql": "pymysql", "postgres": "psycopg"}

#: The one command that installs both server drivers.
INSTALL_DRIVERS = 'pip install "taters[db]"'
#: The same, safe to print as a note: rich would read "[databases]" as
#: markup and print nothing, which is how an install hint once vanished.
INSTALL_DRIVERS_NOTE = INSTALL_DRIVERS.replace("[", "\\[")


def driver_missing(dialect: str) -> str:
    """The install command, or "" when the engine's driver is importable."""
    module = _DRIVER.get(dialect)
    if module is None or importlib.util.find_spec(module) is not None:
        return ""
    return f"Not installed. To add it:  {INSTALL_DRIVERS}"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", text.strip().lower()).strip("-")
    if not s or not s[0].isalpha():
        s = "db-" + s if s else "db"
    return s


def add_connection(prompter: Prompter,
                   existing: Optional[db.Profile] = None) -> Optional[db.Profile]:
    """
    The questions that make a profile, through `_ask_in_order` so Esc steps
    back one with the earlier answers kept. A server is connected to as soon
    as the login is known, before anything is saved, so a wrong password or
    an unreachable host is found here and not at the end of a run; the
    database is then picked from the server's own list. Returns the saved
    profile, or None when the person backed out of the first question.
    """
    from ..wizard import _ask_in_order

    state: Dict[str, Any] = {
        "dialect": existing.dialect if existing else "",
        "path": existing.path if existing else "",
        "host": (f"{existing.host}:{existing.port}" if existing and existing.port
                 and existing.port != db.DEFAULT_PORT.get(existing.dialect)
                 else (existing.host if existing else "localhost")),
        "database": existing.database if existing else "",
        "user": existing.user if existing else "",
        "password": existing.password if existing else "ask",
        "name": existing.name if existing else "",
        "ssl": existing.ssl if existing else "required",
        "ssh": (existing.ssh_address if existing and existing.ssh_host else ""),
        "ssh_key": existing.ssh_key if existing else "",
        "ssh_hostkey": existing.ssh_hostkey if existing else "",
        "databases": None,          # what the server listed, once connected
        "unreached": False,         # saved without a successful connection
    }

    def ask_kind() -> bool:
        while True:
            rows = []
            for value, label, help_text in _KINDS:
                missing = driver_missing(value)
                rows.append(Choice(value, label, missing or help_text))
            state["dialect"] = str(prompter.select("What kind of database is it?", rows,
                                                   default=state["dialect"] or None))
            missing = driver_missing(state["dialect"])
            if not missing:
                return True
            # not disabled, because they may be about to install it; but it
            # cannot go further until they have, so this says exactly that
            prompter.note(f"  Taters cannot talk to {dict((v, l) for v, l, _h in _KINDS)[state['dialect']]} "
                          f"until its driver is installed. In the same environment "
                          f"Taters runs in:\n    {INSTALL_DRIVERS_NOTE}\n  then come back here.",
                          style="yellow")

    def ask_where() -> bool:
        if state["dialect"] == "sqlite":
            from pathlib import Path

            from ..browse import browse_for_file, browse_for_folder

            current = Path(state["path"]).expanduser() if state["path"] else None
            start = current.parent if current and current.parent.is_dir() else None
            if prompter.confirm("Does the database file exist already?",
                                default=current.is_file() if current else True):
                state["path"] = str(browse_for_file(
                    prompter, question="Which database file?", start=start,
                    suffixes=(".sqlite", ".sqlite3", ".db", ".s3db")))
                return True
            # a new one: where, then what to call it. it is made the first
            # time something is written, so nothing lands on disk from here
            folder = browse_for_folder(prompter, question="Where should the new "
                                                          "database file go?", start=start)
            while True:
                name = str(prompter.text("What should the file be called?",
                                         default=current.name if current else "lab.sqlite")
                           ).strip()
                if not name:
                    prompter.note("  A file name is needed.", style="yellow")
                    continue
                if not Path(name).suffix:
                    name += ".sqlite"
                state["path"] = str(folder / name)
                prompter.note(f"  {state['path']} will be created the first time "
                              f"something is written to it.", style="dim")
                return True
        prompter.reason("Some databases are not on the network at all: you log in to "
                        "a machine over ssh and run the client there. Taters can do the "
                        "same, through a tunnel it opens with your own ssh keys.")
        how = str(prompter.select("How is the database reached?", [
            Choice("direct", "Over the network", "The server has an address this "
                                                 "computer can connect to"),
            Choice("ssh", "Through an ssh host", "Log in to a machine first; the "
                                                 "database is reachable from there"),
        ], default="ssh" if state["ssh"] else "direct"))
        if how == "ssh":
            ask_ssh_host()
            ask_ssh_key()
            if state["host"] == "localhost" or not state["host"]:
                state["host"] = "localhost"
            ask_db_address()
            return True
        state["ssh"] = ""
        ask_server_address()
        return True

    # the pieces of "where", each askable again on its own from the retry
    # menu: a forgotten ssh port must not mean starting over
    def ask_ssh_host() -> None:
        new = str(prompter.text("Which ssh host? (user@host, or user@host:port)",
                                default=state["ssh"])).strip()
        if new != state["ssh"]:
            state["ssh_hostkey"] = ""          # a different machine, a different key
        state["ssh"] = new

    def ask_ssh_key() -> None:
        prompter.reason("Most people don't need this: Taters uses the same keys your "
                        "ssh does. Say yes for a file you normally pass with -i, or "
                        "PuTTY's .ppk.")
        if prompter.confirm("Do you log in to that host with a particular key file?",
                            default=bool(state["ssh_key"])):
            from pathlib import Path

            from ..browse import browse_for_file

            ssh_dir = Path.home() / ".ssh"
            prompter.reason("The private key: a file like id_ed25519 or id_rsa, "
                            "a .pem, or PuTTY's .ppk -- never the .pub one.")
            state["ssh_key"] = str(browse_for_file(
                prompter, question="Which key file?",
                start=(Path(state["ssh_key"]).expanduser().parent
                       if state["ssh_key"] else ssh_dir if ssh_dir.is_dir() else None),
                suffixes=("", ".pem", ".key", ".ppk")))
        else:
            state["ssh_key"] = ""

    def ask_db_address() -> None:
        state["host"] = str(prompter.text(
            f"What is the database's address, as seen from "
            f"{state['ssh'].partition('@')[2] or state['ssh']}? (host, or host:port)",
            default=state["host"])).strip() or "localhost"

    def ask_server_address() -> None:
        state["host"] = str(prompter.text(
            "What is the server's address? (host, or host:port)",
            default=state["host"])).strip() or "localhost"

    def ask_user() -> bool:
        if state["dialect"] == "sqlite":
            return False
        state["user"] = str(prompter.text("Which username?",
                                          default=state["user"])).strip()
        return True

    def ask_name() -> bool:
        from pathlib import Path

        suggested = state["name"] or _slug(
            Path(state["path"]).stem if state["dialect"] == "sqlite"
            else (state["ssh"].rpartition("@")[2] if state["ssh"] else state["host"])
            .partition(":")[0])
        while True:
            typed = str(prompter.text("What should Taters call this connection?",
                                      default=suggested)).strip()
            name = _slug(typed)
            if existing is None or name != existing.name:
                if any(p.name == name for p in db.list_profiles()):
                    prompter.note(f"  There is already a connection called {name}.",
                                  style="yellow")
                    continue
            if name != typed:
                prompter.note(f"  Saved as {name}.", style="dim")
            state["name"] = name
            return True

    def ask_password() -> bool:
        if state["dialect"] == "sqlite":
            return False
        prompter.reason("Taters never writes passwords to disk.")
        state["password"] = str(prompter.select(
            "How should Taters get the password?", [
                Choice("ask", "Ask me each time",
                       "Typed with dots; kept only until Taters closes"),
                Choice("env", "From an environment variable",
                       f"{db.password_env(state['name'])} -- set it in your shell "
                       f"profile; useful for scripts"),
                Choice("none", "No password",
                       "For servers that trust this computer's login"),
            ], default=state["password"]))
        return True

    def _trial() -> db.Profile:
        host, _, port = state["host"].partition(":")
        try:
            port_n = int(port) if port.strip() else 0
        except ValueError:
            port_n = 0
        ssh_user, _, ssh_rest = state["ssh"].rpartition("@")
        ssh_host, _, ssh_port = ssh_rest.partition(":")
        try:
            ssh_port_n = int(ssh_port) if ssh_port.strip() else 0
        except ValueError:
            ssh_port_n = 0
        return db.Profile(
            name=state["name"], dialect=state["dialect"], host=host.strip(),
            port=port_n, user=state["user"], database=state["database"],
            path=state["path"], password=state["password"],
            schema=existing.schema if existing else "",
            ssl=state["ssl"],
            ssl_ca=existing.ssl_ca if existing else "",
            export=existing.export if existing else "allowed",
            ssh_host=ssh_host.strip(), ssh_port=ssh_port_n, ssh_user=ssh_user.strip(),
            ssh_key=state["ssh_key"], ssh_hostkey=state["ssh_hostkey"])

    def ask_connect() -> bool:
        if state["dialect"] == "sqlite":
            return False
        state["unreached"] = False
        wrong = 0
        while True:
            trial = db.with_fields(_trial(), database="")
            try:
                password = db.password_for(trial, ask=prompter.secret)
            except db.DbError as e:
                prompter.note(f"  {e}", style="yellow")
                password = None
                probe = None
                text = str(e)
            if password is not None:
                reaching = (f"Connecting through ssh {trial.ssh_address} to "
                            f"{trial.host or 'localhost'}" if trial.tunneled
                            else f"Connecting to {trial.host}")
                try:
                    with prompter.scanning(reaching, unit="checks") as step:
                        probe = db.probe(trial, password, on_step=step)
                except db.HostKeyUnknown as e:
                    from ..db_source import offer_host_key

                    accepted = offer_host_key(prompter, e, save=False)
                    if accepted is not None:
                        state["ssh_hostkey"] = accepted.ssh_hostkey
                        continue
                    probe = db.Probe(notes=(str(e),))
                if probe.select:
                    with db.connect(trial, password) as session:
                        state["databases"] = session.d.list_databases(session)
                    enc = {True: "yes", False: "no", None: "unknown"}[probe.encrypted]
                    prompter.note(f"  Connected to {trial.host or 'localhost'}"
                                  + (f" through ssh {trial.ssh_address}" if trial.tunneled else "")
                                  + f" as {trial.user} ({probe.version}). Encrypted: "
                                  + ("yes, by the tunnel." if trial.tunneled else f"{enc}."),
                                  style="green")
                    return True
                text = "; ".join(probe.notes) or "could not connect"
                if "did not accept that password" in text and state["password"] == "ask":
                    db.forget_password(trial.name)
                    wrong += 1
                    prompter.note(f"  {text}", style="yellow")
                    if wrong < 2:
                        continue
                    prompter.reason("Twice now, so Taters stopped before the account "
                                    "gets locked.")
                else:
                    prompter.note(f"  {text}", style="yellow")
            no_ssl = "does not offer an encrypted connection" in (text if password else "")
            tunneled = bool(state["ssh"])
            answer = str(prompter.select("Could not connect. What now?", [
                *([Choice("plain", "Connect without encryption",
                          "Only for a server inside a trusted network: the password "
                          "and the data travel in the clear")] if no_ssl else []),
                Choice("again", "Try again"),
                *([Choice("ssh", "Change the ssh host", f"Now {state['ssh']}"),
                   Choice("key", "Change the key file",
                          f"Now {state['ssh_key'] or 'your usual keys'}"),
                   Choice("host", "Change the database's address, as seen from there",
                          f"Now {state['host']}")]
                  if tunneled else
                  [Choice("host", "Change the server address", f"Now {state['host']}")]),
                Choice("user", "Change the username", f"Now {state['user']}"),
                Choice("save", "Save it anyway and sort it out later",
                       "The database has to be typed in, since the server could not list them"),
                Choice("back", f"{glyphs.BACK} Never mind", tone="nav"),
            ]))
            if answer == "back":
                raise _Abandon()
            if answer == "plain":
                state["ssl"] = "off"
            if answer == "ssh":
                ask_ssh_host()
            elif answer == "key":
                ask_ssh_key()
            if answer == "save":
                state["unreached"] = True
                state["databases"] = None
                return True
            if answer == "host":
                ask_db_address() if tunneled else ask_server_address()
            elif answer == "user":
                state["user"] = str(prompter.text("Which username?",
                                                  default=state["user"])).strip()
            wrong = 0

    def ask_database() -> bool:
        if state["dialect"] == "sqlite":
            return False
        listed = state["databases"] or []
        if not listed:
            if state["databases"] is not None:
                prompter.note(f"  {state['user']} cannot see any database on "
                              f"{state['host']}; type the name of the one to use.",
                              style="yellow")
            state["database"] = str(prompter.text("Which database on that server?",
                                                  default=state["database"])).strip()
            return True
        rows = [Choice(name, name) for name in listed]
        rows.append(Choice("\x00type", "Type a name…", "One not listed here"))
        prompter.reason("One database per connection. To use another on the same "
                        "server, add a second connection.")
        picked = str(prompter.select("Which database?", rows,
                                     default=state["database"] if state["database"] in listed
                                     else None))
        if picked == "\x00type":
            state["database"] = str(prompter.text("Which database on that server?",
                                                  default=state["database"])).strip()
        else:
            state["database"] = picked
        return True

    try:
        _ask_in_order([ask_kind, ask_where, ask_user, ask_name, ask_password,
                       ask_connect, ask_database])
    except (GoBack, _Abandon):
        return None

    profile = _trial()
    if existing is not None and existing.name != profile.name:
        db.delete_profile(existing.name)
    db.save_profile(profile)
    prompter.note("  Saved.", style="green")
    if state["unreached"]:
        prompter.note(f"  Not tested: {profile.name} could not be reached. Settings > "
                      f"Database connections > Test it, once it can be.", style="dim")
    elif profile.dialect == "sqlite":
        if prompter.confirm("Test it now?", default=True):
            test_connection(prompter, profile)
    else:
        # already logged in once above; this is the same login on the chosen
        # database, and it says what the account may do there
        test_connection(prompter, profile)
    return profile


def test_connection(prompter: Prompter, profile: db.Profile) -> db.Probe:
    """Try the connection and read back what the account can do."""
    missing = driver_missing(profile.dialect)
    if missing:
        prompter.note(f"  {missing.replace('[', chr(92) + '[')}", style="yellow")
        return db.Probe(notes=(missing,))
    try:
        password = db.password_for(profile, ask=prompter.secret)
    except db.DbError as e:
        prompter.note(f"  {e}", style="yellow")
        return db.Probe(notes=(str(e),))
    with prompter.scanning(f"Connecting to {profile.name}", unit="checks") as step:
        result = db.probe(profile, password, on_step=step)
    if not result.select:
        db.forget_password(profile.name)
    for i, line in enumerate(result.sentences(profile)):
        prompter.note(f"  {line}", style="green" if i == 0 and result.select
                      else ("yellow" if not result.select else "dim"))
    return result


def _run(ctx: TaskContext):
    prompter = ctx.prompter
    while True:
        profiles = db.list_profiles()
        if profiles:
            prompter.note("\n  Connections: " + ", ".join(
                f"{p.name} ({p.engine_label})" for p in profiles), style="cyan",
                wrap=False)
        else:
            prompter.note("\n  No connections yet.", style="cyan", wrap=False)
        rows: List[Choice] = [Choice(p.name, p.name, p.label()) for p in profiles]
        rows.append(Choice(_ADD, "+ Add a connection…",
                           "A server and a login, or a database file"))
        rows.append(Choice(_BACK, f"{glyphs.BACK} Back", tone="nav"))
        try:
            picked = str(prompter.select("Database connections:", rows))
        except GoBack:
            return None
        if picked == _BACK:
            return None
        if picked == _ADD:
            add_connection(prompter)
            continue
        profile = db.get_profile(picked)
        try:
            what = str(prompter.select(f"What about {profile.name}?", [
                Choice("test", "Test it", "Connect, and say what the account can do"),
                Choice("edit", "Change it", "The same questions, with the current "
                                            "answers filled in"),
                Choice("forget", "Forget it", "Tables already in the database are "
                                              "not touched"),
                Choice(_BACK, f"{glyphs.BACK} Back", tone="nav"),
            ]))
        except GoBack:
            continue
        if what == "test":
            test_connection(prompter, profile)
        elif what == "edit":
            add_connection(prompter, existing=profile)
        elif what == "forget":
            if prompter.confirm(f"Forget {profile.name}? Tables already in the "
                                f"database are not touched.", default=False):
                db.delete_profile(profile.name)
                prompter.note("  Forgotten.", style="green")


TASK = Task(
    id="db_connections",
    label="Database connections",
    help="Add, test or forget the databases Taters reads from and writes results to",
    run=_run,
)
