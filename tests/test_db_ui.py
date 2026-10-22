"""
The database conversations, scripted: connections under Settings, the
source conversation (connection, table, joins), and the two Wrangle flows.

A temporary SQLite database stands in for the lab's server, since SQLite is
an engine Taters supports outright; the server-only behavior (passwords,
drivers) is exercised through fakes.
"""
from __future__ import annotations

import sqlite3

import pytest

from csvhelpers import read_rows, write_rows
from taters.helpers import db, tables
from taters.ui.prompts import ScriptedPrompter
from taters.ui.tasks import TaskContext
from taters.ui.tasks.db_connections import TASK as CONNECTIONS
from taters.ui.tasks.wrangle import TASK as WRANGLE


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(h))
    tables.close_sessions()
    db._SECRETS.clear()
    yield h
    tables.close_sessions()


@pytest.fixture
def lab(home, tmp_path):
    path = tmp_path / "lab.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE users(id INTEGER PRIMARY KEY, age INTEGER, cond TEXT, created_at TEXT);
    CREATE TABLE posts(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id),
                       body TEXT, created_at TEXT, score REAL);
    CREATE TABLE outcomes(post_id INTEGER, wave INTEGER, phq9 REAL);
    CREATE TABLE outcome_summary(post_id INTEGER PRIMARY KEY, phq9 REAL);
    CREATE TABLE empty_one(x TEXT);
    CREATE TABLE notes(ref INTEGER, note TEXT);
    INSERT INTO users VALUES (1, 30, 'A', 'u1'), (2, 25, 'B', 'u2'), (3, 50, 'C', 'u3');
    INSERT INTO posts VALUES (10, 1, 'hello there', 'p1', 1.5), (11, 1, 'again', 'u1', 2.0),
                             (12, 2, 'yo', 'p3', NULL), (13, 9, 'orphan', 'p4', 3.5);
    INSERT INTO outcomes VALUES (10, 1, 5), (10, 2, 7), (11, 1, 3);
    INSERT INTO outcome_summary VALUES (10, 6), (11, 3), (12, 9);
    INSERT INTO notes VALUES (10, 'n1'), (12, 'n2');
    """)
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(path)))
    return path


def _asked(p, kind=None):
    return [q for k, q in p.asked if kind is None or k == kind]


# ---------------------------------------------------------------------------
# Settings: Database connections
# ---------------------------------------------------------------------------

def test_a_sqlite_connection_is_added_tested_and_forgotten(home, tmp_path):
    from taters.ui.browse import _TYPE

    file = tmp_path / "study.sqlite"
    p = ScriptedPrompter([
        "\x00add",                        # + Add a connection
        "sqlite",
        False,                            # the file does not exist yet
        _TYPE, str(tmp_path),             # where it goes: the folder browser
        "study",                          # what to call it (.sqlite is added)
        "study",                          # name
        True,                             # test it now
        "study", "forget", True,          # then forget it
        "back",
    ])
    assert CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path)) is None

    out = " ".join(p.output)
    assert f"{file} will be created the first time something is written" in out
    assert "Saved." in out and "Connected to study (SQLite" in out
    assert "Can create tables: yes." in out and "Forgotten." in out
    assert "Connecting to study — Counting the tables this account can see" in p.output, \
        "each step of the test is named while it runs"
    assert db.list_profiles() == []
    assert "Connecting to study" in p.output, "the test runs out loud"


class _FakeSession:
    """What a server answers once logged in: its list of databases."""

    def __init__(self, databases):
        self.d = self
        self._databases = databases

    def list_databases(self, _session):
        return list(self._databases)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        pass

    def close(self):
        pass


def test_a_server_connection_is_a_login_and_the_database_is_picked_when_it_is_used(
        home, tmp_path, monkeypatch):
    """Setting up asks for the server and the login, nothing about a
    database; the test that follows picks one from the server's own list,
    and that pick is remembered as the default for next time."""
    seen = []

    def fake_probe(profile, password=None, ask=None, on_step=None):
        seen.append((profile.host, profile.port, profile.user, profile.database, password))
        return db.Probe(select=True, create=True, drop=True, registry=True,
                        version="MySQL 8.4", tables=0, encrypted=True)
    monkeypatch.setattr(db, "probe", fake_probe)
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["study7", "archive"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    p = ScriptedPrompter([
        "\x00add", "mysql", "direct", "db.lab.edu:3307", "ro",
        "Lab DB!",                        # a name that needs tidying
        "ask", "hunter2",                 # typed, with dots
        "study7",                         # the test: which database to check
        "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))

    (prof,) = db.list_profiles()
    assert (prof.name, prof.host, prof.port, prof.database, prof.user, prof.password) == \
        ("lab-db", "db.lab.edu", 3307, "study7", "ro", "ask")
    assert seen[0][:4] == ("db.lab.edu", 3307, "ro", ""), "logged in before any database was named"
    assert seen[1][3] == "study7", "then tested on the one picked"
    asked = [q for _k, q in p.asked]
    assert asked.index("What should Taters call this connection?") < \
        asked.index("Which database?"), "the connection is named for the login, not a database"
    assert "Connected to db.lab.edu as ro (MySQL 8.4). Encrypted: yes." in p.text_output
    assert "2 database(s) this login can see: study7, archive." in p.text_output
    listed = [c.value for c in p.offered_choices("Which database?")]
    assert listed == ["study7", "archive", "\x00type"]
    assert "Which database on that server?" not in asked
    assert "on database study7" in p.text_output
    assert "hunter2" not in p.text_output and "hunter2" not in repr(p.presented)
    saved = (home / "settings.json").read_text(encoding="utf-8")
    assert "hunter2" not in saved and '"password": "ask"' in saved


def test_a_database_behind_an_ssh_host_is_set_up_from_the_same_questions(
        home, tmp_path, monkeypatch):
    seen = []

    def fake_probe(profile, password=None, ask=None, on_step=None):
        seen.append(profile)
        return db.Probe(select=True, create=True, drop=True, registry=True,
                        version="MySQL 8.4", tables=0, encrypted=False)
    monkeypatch.setattr(db, "probe", fake_probe)
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["study"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    monkeypatch.setattr(db, "remote_command",
                        lambda profile, command, timeout=30.0: (0, "me\n", ""))
    p = ScriptedPrompter([
        "\x00add", "mysql",
        "ssh",                            # how is the database reached
        "me@myserver.example.edu",    # the ssh host
        False,                            # a particular key file? no, the usual keys
        False,                            # look on the ssh host for the login? no
        "localhost",                      # the database, as seen from there
        "me", "myserver", "ask", "pw",
        "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    (prof,) = db.list_profiles()
    assert (prof.ssh_host, prof.ssh_user, prof.ssh_port, prof.ssh_key) == \
        ("myserver.example.edu", "me", 0, "")
    assert (prof.host, prof.user, prof.database) == ("localhost", "me", "study")
    assert seen[0].tunneled, "the tunnel is part of the first login"
    assert "Logged in to myserver.example.edu as me." in p.text_output
    asked_all = [q for _k, q in p.asked]
    assert asked_all.index("Look on the ssh host for how the database logs you in?") < \
        asked_all.index("Which username?"), "the machine first, the database after"
    assert "Connecting through ssh me@myserver.example.edu to localhost" in p.output
    assert "Connected to localhost through ssh me@myserver.example.edu as me " \
           "(MySQL 8.4). Encrypted: yes, by the tunnel." in p.text_output
    asked = [q for _k, q in p.asked]
    assert "What is the database's address, as seen from myserver.example.edu? " \
           "(host, or host:port)" in asked


def test_an_unknown_host_key_is_offered_and_accepted_in_the_add_flow(home, tmp_path,
                                                                     monkeypatch):
    """The host key comes up at the ssh check, before any database question,
    and accepting it carries on with the same login."""
    logins = []

    def fake_remote(profile, command, timeout=30.0):
        logins.append(profile.ssh_hostkey)
        if not profile.ssh_hostkey:
            return 255, "", "Host key verification failed."
        return 0, "me\n", ""
    monkeypatch.setattr(db, "remote_command", fake_remote)
    monkeypatch.setattr(db, "host_keys", lambda profile, log="": [
        db.HostKey("ssh-ed25519", "AAAA", "SHA256:abc")])
    monkeypatch.setattr(db, "probe", lambda profile, password=None, ask=None, on_step=None:
                        db.Probe(select=True, create=True, drop=True, registry=True,
                                 version="MySQL 8.4", tables=0, encrypted=False))
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["study"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    recorded = []
    monkeypatch.setattr(db, "accept_host_keys",
                        lambda profile, keys, save=True: (recorded.append((keys, save)),
                                                          db.with_fields(profile, ssh_hostkey="SHA256:abc"))[1])
    p = ScriptedPrompter([
        "\x00add", "mysql", "ssh", "me@h.edu", False,
        True,                             # accept h.edu's key and continue
        False,                            # look on the ssh host? no
        "localhost", "me", "h", "ask", "pw", "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    asked = [q for _k, q in p.asked]
    assert "Accept h.edu's key and continue?" in asked
    assert asked.index("Accept h.edu's key and continue?") < asked.index("Which username?")
    assert "ssh-ed25519  SHA256:abc" in p.text_output
    assert "compare the fingerprint above with one you got some other way" in \
        " ".join(p.reasons) + " " + p.text_output
    assert recorded and recorded[0][1] is False, "not saved before the profile exists"
    assert logins == ["", "SHA256:abc"], "refused, accepted, logged in"
    assert "Logged in to h.edu as me." in p.text_output
    assert db.get_profile("h").ssh_hostkey == "SHA256:abc"

def test_the_retry_menu_lets_the_ssh_host_be_fixed_without_starting_over(home, tmp_path,
                                                                           monkeypatch):
    """A forgotten ssh port is found at the ssh check -- before a single
    question about the database -- and fixed there, alone."""
    seen = []

    def fake_remote(profile, command, timeout=30.0):
        seen.append(profile.ssh_port)
        if profile.ssh_port != 130:
            return 255, "", "ssh: connect to host h.edu port 22: Connection refused"
        return 0, "me\n", ""
    monkeypatch.setattr(db, "remote_command", fake_remote)
    monkeypatch.setattr(db, "probe", lambda profile, password=None, ask=None, on_step=None:
                        db.Probe(select=True, create=True, drop=True, registry=True,
                                 version="MySQL 8.4", tables=0, encrypted=False))
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["study"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    p = ScriptedPrompter([
        "\x00add", "mysql", "ssh", "me@h.edu", False,
        "ssh", "me@h.edu:130",            # could not log in: fix the ssh host alone
        False,                            # look? no
        "localhost", "me", "h", "ask", "pw", "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    menu = {c.value: c for c in p.offered_choices("Could not log in over ssh. What now?")}
    assert [v for v in menu] == ["again", "ssh", "key", "back"]
    assert menu["ssh"].help == "Now me@h.edu"
    assert "connect to the VPN first" in p.text_output
    asked = [q for _k, q in p.asked]
    assert asked.index("Could not log in over ssh. What now?") < asked.index("Which username?"), \
        "nothing about the database was asked while the machine was unreachable"
    assert seen == [0, 130]
    assert db.get_profile("h").ssh_port == 130

def _hints_probe(monkeypatch, hints):
    """A server that cannot be reached, with a look that returns ``hints``."""
    monkeypatch.setattr(db, "remote_login_hints", lambda profile: hints)
    monkeypatch.setattr(db, "remote_command",
                        lambda profile, command, timeout=30.0: (0, "me\n", ""))
    seen = []

    def fake_probe(profile, password=None, ask=None, on_step=None):
        seen.append(profile)
        return db.Probe(select=True, create=True, drop=True, registry=True,
                        version="MySQL 8.4", tables=0, encrypted=False)
    monkeypatch.setattr(db, "probe", fake_probe)
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["study"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    return seen


def test_the_look_finds_the_socket_and_the_login_needs_no_password(home, tmp_path, monkeypatch):
    """The server that trusts whoever is logged in: the look finds the
    socket, the tunnel goes to it, and there is no password to ask for."""
    seen = _hints_probe(monkeypatch, db.LoginHints(socket="/run/mysqld/mysqld.sock",
                                                   os_user="me"))
    p = ScriptedPrompter([
        "\x00add", "mysql", "ssh", "me@h.edu", False,
        True,                             # look on the ssh host? yes
        "me",                             # username (pre-filled from the ssh user)
        "h", "none",                      # name; no password (the default now)
        "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    (prof,) = db.list_profiles()
    assert (prof.socket, prof.password, prof.user) == ("/run/mysqld/mysqld.sock", "none", "me")
    said = p.text_output
    assert "listens on its local socket (/run/mysqld/mysqld.sock)" in said
    assert "with no password" in said
    asked = [q for _k, q in p.asked]
    assert not any("as seen from" in q for q in asked), "a socket is the address"
    assert seen[0].socket == "/run/mysqld/mysqld.sock", "the login went through the socket"
    assert "no password, by socket login" in prof.label()


def test_the_look_finds_a_saved_login_and_reads_it_each_session(home, tmp_path, monkeypatch):
    _hints_probe(monkeypatch, db.LoginHints(user="lab_ro", password="s3cret",
                                            source="~/.my.cnf", os_user="me"))
    p = ScriptedPrompter([
        "\x00add", "mysql", "ssh", "me@h.edu", False,
        True,                             # look
        "localhost",                      # the database's address there
        "lab_ro",                         # username, pre-filled from the file
        "h", "server",                    # name; password from the saved login
        "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    (prof,) = db.list_profiles()
    assert (prof.password, prof.user, prof.socket) == ("server", "lab_ro", "")
    said = p.text_output
    assert "Found a saved login for lab_ro in ~/.my.cnf there" in said
    assert "s3cret" not in said and "s3cret" not in repr(p.presented)
    rows = {c.value: c for c in p.offered_choices("How should Taters get the password?")}
    assert rows["server"].label == "From your saved login on h.edu"
    assert "never copied" in rows["server"].help
    saved = (home / "settings.json").read_text(encoding="utf-8")
    assert "s3cret" not in saved and '"password": "server"' in saved


def test_a_look_that_finds_nothing_falls_back_to_asking(home, tmp_path, monkeypatch):
    _hints_probe(monkeypatch, db.LoginHints(os_user="me"))
    p = ScriptedPrompter([
        "\x00add", "mysql", "ssh", "me@h.edu", False, True,
        "localhost", "me", "h", "ask", "pw", "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    assert "Nothing saved there; Taters will ask for the password." in p.text_output
    rows = {c.value: c for c in p.offered_choices("How should Taters get the password?")}
    assert "server" not in rows


def test_the_key_file_is_picked_with_the_file_browser(home, tmp_path, monkeypatch):
    from taters.ui.browse import _TYPE

    monkeypatch.setattr(db, "probe", lambda profile, password=None, ask=None, on_step=None: db.Probe(
        select=True, create=True, drop=True, registry=True, version="MySQL 8.4",
        tables=0, encrypted=False))
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["study"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    keys = tmp_path / "keys"
    keys.mkdir()
    key = keys / "id_lab"
    key.write_text("private", encoding="utf-8")
    (keys / "id_lab.pub").write_text("public", encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".ssh").mkdir()
    monkeypatch.setattr(db, "remote_command",
                        lambda profile, command, timeout=30.0: (0, "me\n", ""))
    p = ScriptedPrompter([
        "\x00add", "mysql", "ssh", "me@h.edu",
        True,                             # a particular key file
        _TYPE, str(key),                  # the browser: type its path
        False,                            # look on the ssh host? no
        "localhost", "me", "h", "ask", "pw", "study", "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    (prof,) = db.list_profiles()
    assert prof.ssh_key == str(key.resolve())
    assert "Which key file?" in [q for _k, q in p.asked]
    assert "never the .pub one" in " ".join(p.reasons)


def test_a_server_that_cannot_be_reached_is_said_so_before_anything_is_saved(
        home, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "probe", lambda profile, password=None, ask=None, on_step=None: db.Probe(
        notes=("Could not reach db.lab.edu on port 3307. If the database is only "
               "reachable on campus, connect to the VPN first.",)))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    p = ScriptedPrompter([
        "\x00add", "mysql", "direct", "db.lab.edu:3307", "ro", "lab", "ask", "pw",
        "save",                           # could not connect: save it anyway
        "back",
    ])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    (prof,) = db.list_profiles()
    assert (prof.database, prof.password) == ("", "ask"), "no database until one is used"
    said = p.text_output
    assert "connect to the VPN first" in said
    assert "Could not connect. What now?" in [q for _k, q in p.asked]
    assert "Not tested: lab could not be reached." in said
    rows = {c.value: c for c in p.offered_choices("How should Taters get the password?")}
    assert rows["env"].help.startswith("TATERS_DB_PASSWORD_LAB "), "named from the connection"


def test_a_wrong_password_is_asked_again_once_then_stopped(home, tmp_path, monkeypatch):
    tries = []

    def fake_probe(profile, password=None, ask=None, on_step=None):
        tries.append(password)
        return db.Probe(notes=(f"{profile.name} did not accept that password for ro.",))
    monkeypatch.setattr(db, "probe", fake_probe)
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    p = ScriptedPrompter(["\x00add", "mysql", "direct", "h", "ro", "lab", "ask",
                          "wrong1", "wrong2",
                          "back",                         # never mind: the add is dropped
                          "back"])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    assert tries == ["wrong1", "wrong2"]
    assert "stopped before the account gets locked" in " ".join(p.reasons)
    assert db.list_profiles() == [], "nothing saved"


def test_a_missing_driver_is_named_in_the_choice_text_and_stops_the_kind(home, tmp_path,
                                                                          monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "pymysql" else real(name, *a, **k))
    from taters.ui.browse import _TYPE

    existing = tmp_path / "x.sqlite"
    existing.write_bytes(b"")
    p = ScriptedPrompter(["\x00add", "mysql",      # picked, though the driver is missing
                          "sqlite", True,          # an existing file, picked in the browser
                          _TYPE, str(existing), "x",
                          False, "back"])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))
    rows = {c.value: c for c in p.offered_choices("What kind of database is it?")}
    assert rows["mysql"].help == 'Not installed. To add it:  pip install "taters[db]"'
    assert not rows["mysql"].disabled, "they may be about to install it"
    assert 'pip install "taters[db]"' in p.text_output.replace("\\[", "[")
    assert "until its driver is installed" in p.text_output
    assert "Which database file?" in [q for _k, q in p.asked]
    assert db.get_profile("x").path == str(existing.resolve())


def test_a_typed_password_is_asked_with_dots_and_recorded_nowhere(home, tmp_path, monkeypatch):
    db.save_profile(db.Profile(name="srv", dialect="mysql", host="h", user="u",
                               database="d", password="ask"))
    seen = {}

    def fake_probe(profile, password=None, ask=None, on_step=None):
        seen["password"] = password
        return db.Probe(select=True, create=True, drop=True, registry=True,
                        version="MySQL 8.0", tables=3, encrypted=True)
    monkeypatch.setattr(db, "probe", fake_probe)
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _FakeSession(["d", "e"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    p = ScriptedPrompter(["srv", "test", "hunter2", "d", "back", "back"])
    CONNECTIONS.run(TaskContext(prompter=p, cwd=tmp_path))

    assert seen["password"] == "hunter2"
    assert ("secret", "Password for u on srv:") in p.asked
    assert "hunter2" not in p.text_output and "hunter2" not in repr(p.presented)
    assert "Connected to srv (MySQL 8.0, h) as u, on database d. Encrypted: yes." in p.text_output


# ---------------------------------------------------------------------------
# Wrangle: tables out to a spreadsheet
# ---------------------------------------------------------------------------

def _cols(table, *names):
    return [f"{table}.{n}" for n in names]


def test_a_join_of_tables_is_copied_out_with_the_read_back(lab, tmp_path):
    out = tmp_path / "posts_with_users.csv"
    p = ScriptedPrompter([
        "db_out", "join", "lab", ["posts", "users"],
        _cols("posts", "id", "user_id", "body", "created_at", "score")
        + _cols("users", "id", "age", "cond", "created_at"),       # one screen
        str(out),
    ])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True

    rows = read_rows(out)
    assert list(rows[0]) == ["id", "user_id", "body", "created_at", "score", "age", "cond",
                             "users__created_at"]
    assert [r["age"] for r in rows] == ["30", "30", "25", ""]
    said = p.text_output
    assert "users joined on posts.user_id = users.id (the database declares it)." in said
    assert "3 of 4 rows in posts match a row in users; 1 do not (e.g. 9)." in said
    assert "1 of 3 rows in users have no match in posts (e.g. 3)." in said
    assert "created_at is in both tables and differs on 2 matched rows" in said
    assert "4 rows × 8 columns: every row of posts, with users' columns beside it (1 have none)." in said
    assert 'The query: SELECT "t0"."id" AS "id"' in said
    assert "Which table should each row" not in " ".join(q for _k, q in p.asked), \
        "the declared link says which table the rows come from"
    screen = {c.value: c for c in p.offered_choices("Which columns do you want")}
    assert screen["posts.user_id"].help == "→ users.id"
    assert screen["users.id"].help == "the table's key"
    assert screen["\x00table:posts"].annotation == "4 rows  5 columns"
    assert screen["\x00table:posts"].children[:2] == ("posts.id", "posts.user_id")
    assert "\x00table:empty_one" not in screen, "a table with no rows is not offered"
    assert "Wrangled 4 row(s)" in said


def test_a_fan_out_is_allowed_when_copying_out_and_read_back_as_a_repeat(lab, tmp_path):
    out = tmp_path / "o.csv"
    p = ScriptedPrompter([
        "db_out", "join", "lab", ["posts", "outcomes"],
        _cols("posts", "id", "body") + _cols("outcomes", "wave", "phq9"),
        str(out),
    ])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert len(read_rows(out)) == 5, "post 10 twice"
    assert list(read_rows(out)[0]) == ["id", "body", "wave", "phq9"]
    said = p.text_output
    assert "outcomes joined on posts.id = outcomes.post_id (post_id names posts)." in said
    assert "Joining repeats those rows: 4 rows become 5." in said
    assert "measure each copy anyway" not in " ".join(q for _k, q in p.asked)


def test_a_single_table_is_copied_out_with_only_some_columns(lab, tmp_path):
    out = tmp_path / "u.csv"
    p = ScriptedPrompter(["db_out", "join", "lab", ["users"], _cols("users", "id", "age"), str(out)])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert list(read_rows(out)[0]) == ["id", "age"]
    assert "1 rows ×" not in p.text_output
    assert "3 rows × 2 columns: every row of users." in p.text_output


def test_tables_with_nothing_linking_them_are_asked_about(lab, tmp_path):
    out = tmp_path / "o.csv"
    p = ScriptedPrompter([
        "db_out", "join", "lab", ["posts", "notes"],
        _cols("posts", "id", "body") + _cols("notes", "note"),
        "posts",                          # both hold text and nothing links them: which is the row
        "posts.id",                       # which column matches rows to notes
        "ref",                            # which column there holds id
        str(out),
    ])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert [r["note"] for r in read_rows(out)] == ["n1", "", "n2", ""]
    asked = [q for _k, q in p.asked]
    assert "Which table should each row of the spreadsheet come from?" in asked
    assert "Which column matches rows to notes?" in asked
    assert "notes joined on posts.id = notes.ref (as you said)." in p.text_output


def test_a_table_keyed_as_post_id_links_to_posts_by_name(lab, tmp_path):
    out = tmp_path / "o.csv"
    p = ScriptedPrompter(["db_out", "join", "lab", ["posts", "outcome_summary"],
                          _cols("posts", "id", "body") + _cols("outcome_summary", "phq9"),
                          str(out)])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert [r["phq9"] for r in read_rows(out)] == ["6.0", "3.0", "9.0", ""]
    assert "outcome_summary joined on posts.id = outcome_summary.post_id (post_id names posts)." \
        in p.text_output


def test_every_table_can_be_dumped_to_its_own_spreadsheet(lab, tmp_path):
    from taters.ui.browse import _TYPE

    folder = tmp_path / "dump"
    folder.mkdir()
    p = ScriptedPrompter([
        "db_out", "dump",
        ["lab"],                          # which connection(s); a file is its own database
        ["lab/main/users", "lab/main/posts", "lab/main/outcomes"],   # which tables, under it
        True,                             # keep every column
        _TYPE, str(folder),               # where the spreadsheets go
    ])
    WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))
    assert sorted(q.name for q in folder.glob("*.csv")) == ["outcomes.csv", "posts.csv",
                                                             "users.csv"]
    assert list(read_rows(folder / "users.csv")[0]) == ["id", "age", "cond", "created_at"]
    assert len(read_rows(folder / "posts.csv")) == 4
    said = p.text_output
    assert "3 table(s) →" in said and "users.csv (3 rows, 4 columns)" in said
    assert "✓ Wrote 3 file(s) into" in said
    conns = {c.value: c for c in p.offered_choices("Which connection(s)?")}
    assert "lab" in conns and "\x00add" in conns
    assert "Which database(s)?" not in [q for _k, q in p.asked], "a file has only the one"
    rows = {c.value: c for c in p.offered_choices("Which tables?")}
    # the database is a heading whose children are its tables: ticking it
    # takes them all, opening it up takes some
    assert rows["\x00db:lab/main"].children == (
        "lab/main/empty_one", "lab/main/notes", "lab/main/outcome_summary",
        "lab/main/outcomes", "lab/main/posts", "lab/main/users")
    assert rows["\x00db:lab/main"].label == "lab"
    assert rows["lab/main/posts"].annotation == "4 rows  5 columns"
    assert rows["lab/main/posts"].depth == 1 and rows["lab/main/posts"].label.rstrip() == "posts"


def test_a_connection_added_on_the_way_is_not_asked_for_its_database_twice(home, tmp_path,
                                                                            monkeypatch):
    """No connections yet, so the dump flow sets one up on the spot. The
    setup's own test would ask which database to check, and the dump asks
    which database(s) next: the same question twice in a row, so the
    flow skips the test and the database is asked once."""
    class _Server(_FakeSession):
        def list_tables(self, _session, _schema=""):
            return [db.TableInfo(name="posts", rows=3, rows_exact=True,
                                 columns=(db.ColumnInfo("id", "INTEGER"),
                                          db.ColumnInfo("body", "TEXT")),
                                 primary_key=("id",))]
    monkeypatch.setattr(db, "probe", lambda profile, password=None, ask=None, on_step=None:
                        db.Probe(select=True, create=True, drop=True, registry=True,
                                 version="MySQL 8.4", tables=0, encrypted=True))
    monkeypatch.setattr(db, "connect", lambda profile, password=None, ask=None, on_step=None:
                        _Server(["study", "archive"]))
    monkeypatch.setattr("taters.ui.tasks.db_connections.driver_missing", lambda d: "")
    p = ScriptedPrompter([
        "db_out", "dump",
        "mysql", "direct", "h", "me", "lab", "ask", "pw",   # set up, since there are none
        ["lab"],                          # which connection(s)
        ["lab/study"],                    # which database(s): asked once
    ])
    # the script ends at the table screen on purpose: what matters is the
    # road there, and a fake server cannot be copied out of
    with pytest.raises(AssertionError, match="Which tables"):
        WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))
    asked = [q for _k, q in p.asked]
    assert "Which database?" not in asked, "the setup's test is skipped here"
    assert asked.count("Which database(s)?") == 1
    assert "Test it now?" not in asked
    assert db.get_profile("lab").database == "", "nothing picked for it behind the scenes"


def test_a_dump_can_keep_only_some_columns_and_asks_before_replacing(lab, tmp_path):
    from taters.ui.browse import _TYPE

    folder = tmp_path / "dump"
    folder.mkdir()
    (folder / "users.csv").write_text("old", encoding="utf-8")
    p = ScriptedPrompter([
        "db_out", "dump", ["lab"], ["lab/main/users", "lab/main/posts"],
        False,                            # not every column
        ["users.id", "users.age", "posts.id", "posts.body"],
        _TYPE, str(folder),               # users.csv is already there
        True,                             # replace it
    ])
    WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))
    assert list(read_rows(folder / "users.csv")[0]) == ["id", "age"]
    assert list(read_rows(folder / "posts.csv")[0]) == ["id", "body"]
    said = p.text_output
    assert "1 of these already exist" in said and "users.csv" in said
    assert "Replace them?" in [q for _k, q in p.asked]
    assert "users.csv (3 rows, 2 of 4 columns)" in said
    screen = {c.value: c for c in p.offered_choices("Which columns do you want to keep?")}
    assert all(c.checked for c in screen.values() if not c.value.startswith("\x00")), \
        "every column starts ticked; unticking is the work"
    assert "\x00table:outcomes" not in screen, "only the tables being dumped"


def test_tables_from_several_databases_land_in_a_folder_each(lab, tmp_path):
    """Two databases on one screen, each a heading over its tables; the
    files go under a folder per database so same-named tables never collide."""
    from taters.ui.browse import _TYPE

    other = tmp_path / "archive.sqlite"
    con = sqlite3.connect(other)
    con.executescript("""
    CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT);
    CREATE TABLE waves(wave INTEGER, label TEXT);
    INSERT INTO users VALUES (1, 'old ann');
    INSERT INTO waves VALUES (1, 'baseline'), (2, 'follow-up');
    """)
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="archive", dialect="sqlite", path=str(other)))
    folder = tmp_path / "dump"
    folder.mkdir()
    p = ScriptedPrompter([
        "db_out", "dump",
        ["lab", "archive"],
        # the live screen turns a ticked database heading into its tables;
        # this is what that returns for "all of archive, and lab's users"
        ["lab/main/users", "archive/main/users", "archive/main/waves"],
        True, _TYPE, str(folder),
    ])
    WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))
    assert sorted(q.relative_to(folder).as_posix() for q in folder.rglob("*.csv")) == \
        ["archive/users.csv", "archive/waves.csv", "lab/users.csv"]
    assert list(read_rows(folder / "archive" / "users.csv")[0]) == ["id", "name"]
    assert list(read_rows(folder / "lab" / "users.csv")[0]) == ["id", "age", "cond", "created_at"]
    rows = {c.value: c for c in p.offered_choices("Which tables?")}
    assert rows["\x00db:archive/main"].children == ("archive/main/users", "archive/main/waves")
    assert rows["\x00db:archive/main"].annotation == "2 tables"
    said = p.text_output
    assert "3 table(s) →" in said and "archive/waves.csv (2 rows, 2 columns)" in said
    assert "Connected to archive." in said and "Connected to lab." in said


# ---------------------------------------------------------------------------
# Wrangle: a spreadsheet into a table
# ---------------------------------------------------------------------------

def test_a_spreadsheet_goes_into_a_table_with_a_key(lab, tmp_path):
    src = write_rows(tmp_path / "Participants 2024.csv", ["pid", "age", "note"],
                     [["p1", "30", "x"], ["p2", "41", "Zoé"]])
    p = ScriptedPrompter([
        "db_in", "lab", str(src),
        "Participants 2024",              # tidied to participants_2024
        "pid",                            # the key
        True,                             # write it
    ])
    WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))

    said = p.text_output
    assert "It will be called participants_2024" in said
    assert "2 rows × 3 columns → participants_2024 in lab. age are stored as numbers; the rest as text." in said
    assert "✓ Wrote 2 rows into lab: participants_2024." in said
    ref = tables.parse_ref("db://lab/main/participants_2024")
    assert list(tables.read_rows(ref)) == [{"pid": "p1", "age": "30", "note": "x"},
                                           {"pid": "p2", "age": "41", "note": "Zoé"}]
    s = db.session_for("lab")
    assert s.all("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND "
                 "tbl_name = 'participants_2024' AND sql LIKE '%UNIQUE%'") == [(1,)]
    s.close()


def test_a_table_taters_did_not_make_cannot_be_replaced_from_the_flow(lab, tmp_path):
    src = write_rows(tmp_path / "users.csv", ["id", "x"], [["1", "a"]])
    p = ScriptedPrompter(["db_in", "lab", str(src), "users", "rename", "users_import",
                          "id", True])
    WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))
    assert "already has a table called users that Taters did not make" in p.text_output
    assert tables.exists("db://lab/main/users_import")


def test_a_read_only_account_is_turned_away_from_writing_but_not_reading(lab, tmp_path, monkeypatch):
    real = db.probe

    def read_only(profile, password=None, ask=None, on_step=None):
        pr = real(profile, password, ask)
        return db.Probe(select=pr.select, create=False, drop=False, registry=False,
                        version=pr.version, tables=pr.tables, encrypted=pr.encrypted)
    monkeypatch.setattr(db, "probe", read_only)
    from taters.ui.prompts import Cancelled

    p = ScriptedPrompter(["db_in", "lab", "back"])
    with pytest.raises(Cancelled):          # backing out of the first question
        WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path))
    assert "lab can't take new tables. What now?" in _asked(p, "select")
    assert "cannot create tables there" in " ".join(p.reasons)

    out = tmp_path / "u.csv"
    p = ScriptedPrompter(["db_out", "join", "lab", ["users"], ["users.id"], str(out)])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert "results of a run will go to files, or to another database" in p.text_output


def test_esc_inside_the_join_conversation_steps_back_one_question(lab, tmp_path):
    from wizard_helpers import EscapingPrompter

    out = tmp_path / "o.csv"
    p = EscapingPrompter([
        "db_out", "join", "lab", ["posts", "users"], ["posts.id", "posts.body", "users.age"],
        "__esc__",                        # on the output path: back to the tables
        ["posts", "users"], ["posts.id", "posts.body"], str(out),
    ])
    assert WRANGLE.run(TaskContext(prompter=p, cwd=tmp_path)) is True
    assert sum(q.startswith("Which columns do you want") for q in _asked(p)) == 2
    assert list(read_rows(out)[0]) == ["id", "body"], "the second pick is the one used"


def test_a_dumped_table_name_cannot_climb_out_of_the_folder():
    from taters.ui.tasks.wrangle_db import _file_stem

    stem = _file_stem("../../.bashrc")
    assert "/" not in stem and ".." not in stem.split("_")[0] and not stem.startswith(".")
    assert _file_stem("posts") == "posts" and _file_stem("a:b") == "a_b"
