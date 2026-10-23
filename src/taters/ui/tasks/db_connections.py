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
                   existing: Optional[db.Profile] = None, *,
                   test: bool = True) -> Optional[db.Profile]:
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
        "socket": existing.socket if existing else "",
        "looked": None,             # what the ssh host's own client would do, once looked
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
            # the address is asked after the look: a socket found there is
            # the address, and the question would only confuse
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
            state["looked"], state["socket"] = None, ""
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

    def ask_ssh_check() -> bool:
        """
        Log in over ssh now, before a single question about the database:
        if the machine cannot be reached there is nothing to ask yet. Then,
        with the person's say-so, see how the database's own client logs in
        there -- a saved login in their ~/.my.cnf or ~/.pgpass, or the
        database's local socket, which lets the ssh login itself serve.
        Nothing read here is shown or written down.
        """
        if state["dialect"] == "sqlite" or not state["ssh"]:
            state["looked"] = None
            return False
        from ..db_source import offer_host_key

        while True:
            trial = db.with_fields(_trial(), database="")
            try:
                with prompter.scanning(f"Logging in to {trial.ssh_address} over ssh",
                                       unit="checks"):
                    code, out, err = db.remote_command(trial, "id -un")
                if code != 0 and db._host_key_unknown(err):
                    raise db.HostKeyUnknown(db._ssh_words(err, trial), trial,
                                            keys=db.host_keys(trial, err))
                if code != 0:
                    raise db.DbError(db._ssh_words(err, trial))
            except db.HostKeyUnknown as e:
                accepted = offer_host_key(prompter, e, save=False)
                if accepted is None:
                    raise _Abandon()
                state["ssh_hostkey"] = accepted.ssh_hostkey
                continue
            except (db.DbError, OSError) as e:
                prompter.note(f"  {e}", style="yellow")
                answer = str(prompter.select("Could not log in over ssh. What now?", [
                    Choice("again", "Try again"),
                    Choice("ssh", "Change the ssh host", f"Now {state['ssh']}"),
                    Choice("key", "Change the key file",
                           f"Now {state['ssh_key'] or 'your usual keys'}"),
                    Choice("back", f"{glyphs.BACK} Never mind", tone="nav"),
                ]))
                if answer == "back":
                    raise _Abandon()
                if answer == "ssh":
                    ask_ssh_host()
                elif answer == "key":
                    ask_ssh_key()
                continue
            os_user = out.strip().splitlines()[-1].strip() if out.strip() else ""
            prompter.note(f"  Logged in to {trial.ssh_host}"
                          + (f" as {os_user}" if os_user else "") + ".", style="green")
            break

        prompter.reason("On that machine the database may already know you: a login "
                        "saved in your home folder, or a local socket that trusts your "
                        "ssh login. Taters can look; it shows and saves nothing it finds.")
        if not prompter.confirm("Look on the ssh host for how the database logs you in?",
                                default=True):
            state["looked"] = None
            return True
        with prompter.scanning(f"Looking on {trial.ssh_host}", unit="checks"):
            hints = db.remote_login_hints(trial)
        state["looked"] = hints
        if hints.problem:
            prompter.note(f"  Could not look: {hints.problem}", style="yellow")
            return True
        engine = dict((v, l) for v, l, _h in _KINDS)[state["dialect"]]
        if hints.password is not None:
            who = hints.user or state["user"] or hints.os_user
            prompter.note(f"  Found a saved login for {who} in {hints.source} there. Taters "
                          f"can read it fresh each session, so the password is never "
                          f"copied anywhere.", style="green")
            state["user"] = state["user"] or hints.user
            if state["password"] in ("ask", "none"):
                state["password"] = "server"
        if hints.socket and not uses_plink():
            prompter.note(f"  {engine} there listens on its local socket "
                          f"({hints.socket}), where it trusts whoever is logged in. "
                          f"Taters can connect through it as {hints.os_user or 'you'}, "
                          f"with no password.", style="green")
            state["socket"] = hints.socket
            if hints.password is None:
                state["user"] = state["user"] or hints.os_user
                state["password"] = "none"
        elif hints.socket and uses_plink():
            prompter.note(f"  {engine} there listens on its local socket, which PuTTY's "
                          f"plink cannot forward; an OpenSSH key would let Taters use it.",
                          style="dim")
        if hints.password is None and not hints.socket:
            prompter.note("  Nothing saved there; Taters will ask for the password.",
                          style="dim")
        return True

    def uses_plink() -> bool:
        return state["ssh_key"].lower().endswith(".ppk")

    def ask_address_after_look() -> bool:
        if state["dialect"] == "sqlite" or not state["ssh"] or state["socket"]:
            return False
        ask_db_address()
        return True

    def ask_user() -> bool:
        if state["dialect"] == "sqlite":
            return False
        state["user"] = str(prompter.text("Which username?",
                                          default=state["user"])).strip()
        return True

    def ask_name() -> bool:
        from pathlib import Path

        # the login names the connection -- me-db-lab-edu -- since one
        # person may well have two accounts on one server, and the database
        # is not part of it: that is picked each time the connection is used
        server = (state["ssh"].rpartition("@")[2] if state["ssh"] else state["host"]) \
            .partition(":")[0]
        suggested = state["name"] or _slug(
            Path(state["path"]).stem if state["dialect"] == "sqlite"
            else "-".join(w for w in (state["user"], server) if w))
        while True:
            if state["dialect"] != "sqlite":
                prompter.reason("A connection is the server and your login on it. Which "
                                "database to use is asked each time you use it.")
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
        looked = state["looked"]
        rows = [
            Choice("ask", "Ask me each time",
                   "Typed with dots; kept only until Taters closes"),
            Choice("env", "From an environment variable",
                   f"{db.password_env(state['name'])} -- set it in your shell "
                   f"profile; useful for scripts"),
        ]
        if looked is not None and looked.password is not None:
            rows.append(Choice("server", f"From your saved login on {state['ssh'].rpartition('@')[2]}",
                               f"Read from {looked.source} there each session; never copied"))
        rows.append(Choice("none", "No password",
                           "Socket login on the ssh host, or a server that trusts this "
                           "computer" if state["socket"] else
                           "For servers that trust this computer's login"))
        state["password"] = str(prompter.select(
            "How should Taters get the password?", rows, default=state["password"]))
        return True

    def _trial() -> db.Profile:
        # before the name is asked (the look on the ssh host runs first), a
        # stand-in name keeps the profile valid; nothing is saved under it
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
            name=state["name"] or "new", dialect=state["dialect"], host=host.strip(),
            port=port_n, user=state["user"], database=state["database"],
            path=state["path"], password=state["password"],
            schema=existing.schema if existing else "",
            ssl=state["ssl"],
            ssl_ca=existing.ssl_ca if existing else "",
            export=existing.export if existing else "allowed",
            ssh_host=ssh_host.strip(), ssh_port=ssh_port_n, ssh_user=ssh_user.strip(),
            ssh_key=state["ssh_key"], ssh_hostkey=state["ssh_hostkey"],
            socket=state["socket"] if state["ssh"] else "")

    def ask_connect() -> bool:
        if state["dialect"] == "sqlite":
            return False
        state["unreached"] = False
        wrong = 0
        while True:
            trial = db.with_fields(_trial(), database="")
            try:
                password = db.password_for(trial, ask=prompter.secret)
            except db.HostKeyUnknown as e:
                from ..db_source import offer_host_key

                accepted = offer_host_key(prompter, e, save=False)
                if accepted is None:
                    raise _Abandon()
                state["ssh_hostkey"] = accepted.ssh_hostkey
                continue
            except db.DbError as e:
                prompter.note(f"  {e}", style="yellow")
                password = None
                probe = None
                text = str(e)
            # "no password" is an answer, not a failure to get one
            if password is not None or trial.password == "none":
                reaching = (f"Connecting through ssh {trial.ssh_address} to "
                            f"{trial.socket or trial.host or 'localhost'}" if trial.tunneled
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
                    n = len(state["databases"])
                    prompter.note(f"  Connected to {trial.host or 'localhost'}"
                                  + (f" through ssh {trial.ssh_address}" if trial.tunneled else "")
                                  + f" as {trial.user} ({probe.version}). Encrypted: "
                                  + ("yes, by the tunnel." if trial.tunneled else f"{enc}.")
                                  + (f" {n} database(s) this login can see: "
                                     f"{', '.join(state['databases'][:6])}"
                                     + (" …" if n > 6 else "") + "." if n else
                                     " This login cannot list any databases; the one to "
                                     "use can still be typed in when it is needed."),
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
                *([Choice("host", "Change the database's address, as seen from there",
                          f"Now {state['socket'] or state['host']}")]
                  if tunneled else
                  [Choice("host", "Change the server address", f"Now {state['host']}")]),
                Choice("user", "Change the username", f"Now {state['user']}"),
                Choice("save", "Save it anyway and sort it out later",
                       "Settings > Database connections > Test it, once it can be reached"),
                Choice("back", f"{glyphs.BACK} Never mind", tone="nav"),
            ]))
            if answer == "back":
                raise _Abandon()
            if answer == "plain":
                state["ssl"] = "off"
            if answer == "save":
                state["unreached"] = True
                state["databases"] = None
                return True
            if answer == "host":
                if tunneled:
                    state["socket"] = ""      # a typed address replaces a found socket
                    ask_db_address()
                else:
                    ask_server_address()
            elif answer == "user":
                state["user"] = str(prompter.text("Which username?",
                                                  default=state["user"])).strip()
            wrong = 0

    try:
        _ask_in_order([ask_kind, ask_where, ask_ssh_check, ask_address_after_look, ask_user,
                       ask_name, ask_password, ask_connect])
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
    elif not test:
        pass
    elif profile.dialect == "sqlite":
        if prompter.confirm("Test it now?", default=True):
            test_connection(prompter, profile)
    else:
        # already logged in once above; what the account may do is a
        # question about one database, so pick one and say
        prompter.reason("What this login may do differs by database: read one, "
                        "make tables in another. Pick one to check now; any other "
                        "is checked when you use it.")
        test_connection(prompter, profile)
    return profile


def test_connection(prompter: Prompter, profile: db.Profile) -> db.Probe:
    """Log in, pick a database, and read back what the account can do there."""
    from ..db_source import _connect

    missing = driver_missing(profile.dialect)
    if missing:
        prompter.note(f"  {missing.replace('[', chr(92) + '[')}", style="yellow")
        return db.Probe(notes=(missing,))
    try:
        found = _connect(prompter, profile)
    except GoBack:
        found = None
    if found is None:
        db.forget_password(profile.name)
        return db.Probe(notes=("could not connect",))
    found.session.close()
    for i, line in enumerate(found.probe.sentences(found.profile)):
        prompter.note(f"  {line}", style="green" if i == 0 else "dim")
    return found.probe


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
