"""
The table layer: one address for a table wherever it lives, one way to read
and write it, and a database behind it that behaves like a file would.

SQLite is the engine here. It is the standard library, so every test runs
everywhere; the MySQL and PostgreSQL dialects run the same checks when a
server is offered through ``TATERS_TEST_MYSQL_URL`` / ``TATERS_TEST_POSTGRES_URL``
(see ``tests/README.md``).
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from csvhelpers import read_rows, write_rows
from taters.helpers import db, sources, tables
from taters.helpers.tables import parse_ref


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh TATERS_HOME, so profiles never leak between tests."""
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(h))
    tables.close_sessions()
    db._SECRETS.clear()
    yield h
    tables.close_sessions()


@pytest.fixture
def lab(home, tmp_path):
    """A small sqlite database with posts, users (one row each), outcomes (a
    fan-out), and a profile called ``lab`` pointing at it."""
    path = tmp_path / "lab.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE users(id INTEGER PRIMARY KEY, age INTEGER, cond TEXT, created_at TEXT);
    CREATE TABLE posts(id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(id),
                       body TEXT, created_at TEXT, score REAL);
    CREATE TABLE outcomes(post_id INTEGER, wave INTEGER, phq9 REAL);
    CREATE TABLE outcome_summary(post_id INTEGER PRIMARY KEY, phq9 REAL);
    CREATE TABLE nokey(x TEXT, y TEXT);
    CREATE VIEW recent AS SELECT id, body FROM posts WHERE id > 10;
    INSERT INTO users VALUES (1, 30, 'A', 'u1'), (2, 25, 'B', 'u2'), (3, 50, 'C', 'u3');
    INSERT INTO posts VALUES (10, 1, 'hello there', 'p1', 1.5), (11, 1, 'again', 'u1', 2.0),
                             (12, 2, 'yo', 'p3', NULL), (13, 9, 'orphan', 'p4', 3.5);
    INSERT INTO outcomes VALUES (10, 1, 5), (10, 2, 7), (11, 1, 3);
    INSERT INTO outcome_summary VALUES (10, 6), (11, 3), (12, 9);
    INSERT INTO nokey VALUES ('a', '1'), ('a', '2');
    """)
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(path)))
    return path


# ---------------------------------------------------------------------------
# TableRef
# ---------------------------------------------------------------------------

def test_a_file_ref_is_a_path_and_a_db_ref_is_not(tmp_path):
    f = parse_ref(tmp_path / "features" / "readability.csv")
    assert not f.is_db and Path(f) == tmp_path / "features" / "readability.csv"
    assert f.stem == "readability"

    d = parse_ref("db://lab/main/taters/study7/features/readability")
    assert d.is_db and d.profile == "lab" and d.stem == "readability"
    assert d.parts == ["taters", "study7", "features", "readability"]
    with pytest.raises(TypeError):
        Path(d)
    with pytest.raises(TypeError):
        open(d)
    assert str(d) == "db://lab/main/taters/study7/features/readability"


def test_a_db_ref_survives_a_template_suffix_and_a_careless_path(tmp_path):
    assert parse_ref("db://lab/main/features/readability.csv") == parse_ref("db://lab/main/features/readability")
    assert parse_ref(Path("db:/lab/features/readability")).is_db, "what Path() leaves behind"
    assert parse_ref("db://lab/main/x").sibling("y") == parse_ref("db://lab/main/y")
    assert parse_ref(tmp_path / "a" / "x.csv").sibling("y") == parse_ref(tmp_path / "a" / "y.csv")
    with pytest.raises(ValueError):
        parse_ref("db://lab")


def test_a_ref_pickles_hashes_and_compares_by_its_string():
    import pickle
    r = parse_ref("db://lab/main/a/b")
    assert pickle.loads(pickle.dumps(r)) == r
    assert len({r, parse_ref("db://lab/main/a/b")}) == 1
    assert r == "db://lab/main/a/b"


# ---------------------------------------------------------------------------
# Files through the layer: bytes unchanged
# ---------------------------------------------------------------------------

def test_write_table_to_a_file_is_byte_identical_to_the_csv_module(tmp_path):
    rows = [{"text_id": "a", "n": "1"}, {"text_id": "b", "n": ""}]
    theirs = tmp_path / "theirs.csv"
    with theirs.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["text_id", "n"])
        w.writeheader()
        w.writerows(rows)
    ours = tmp_path / "ours.csv"
    with tables.write_table(ours, ["text_id", "n"]) as w:
        w.writerows(rows)
    assert ours.read_bytes() == theirs.read_bytes()
    assert tables.columns_of(ours) == ["text_id", "n"] and tables.count(ours) == 2
    assert list(tables.read_rows(ours)) == rows
    assert tables.exists(ours) and not tables.exists(tmp_path / "nope.csv")


def test_file_sidecars_keep_their_old_names(tmp_path):
    t = write_rows(tmp_path / "cohesion.csv", ["text_id"], [["a"]])
    tables.write_sidecar(t, "settings", {"x": 1})
    assert (tmp_path / "cohesion_settings.json").is_file()
    assert tables.read_sidecar(t, "settings") == {"x": 1}
    tables.delete_sidecar(t, "settings")
    assert tables.read_sidecar(t, "settings") is None


# ---------------------------------------------------------------------------
# What the server says mid-query, in words
# ---------------------------------------------------------------------------

def test_a_drivers_error_mid_query_becomes_a_sentence(lab):
    """The real report: MariaDB's temp disk filled while it answered the
    catalog query, and the person got a forty-line traceback ending in
    'Disk full (/tmp/#sql_1f551_0.MAI)'. Every session verb now turns the
    driver's exception into one sentence that says whose problem it is."""
    s = db.connect(db.get_profile("lab"))
    with pytest.raises(db.DbError, match="has no such table any more") as info:
        s.all("SELECT * FROM nope")
    assert "nope" in str(info.value)
    with pytest.raises(db.DbError, match="SQLite on lab said"):
        s.run("THIS IS NOT SQL")
    s.close()


def test_the_servers_troubles_are_said_as_whose_problem_they_are():
    p = db.Profile(name="lab", dialect="mysql", host="h", user="me",
                   ssh_host="box.example.edu", ssh_user="me")

    def err(code, text):
        e = Exception(code, text)
        return e
    full = db.query_words(err(1021, "Disk full (/tmp/#sql_1f551_0.MAI); waiting for someone "
                                    "to free some space... (errno: 28 \"No space left on device\")"), p)
    assert full.startswith("MySQL on lab has run out of disk space (its temp space")
    assert "Nothing is wrong on this computer" in full and "Ask whoever runs" in full
    assert "#sql_1f551" not in full, "the server's file name is noise to a person"
    gone = db.query_words(err(2013, "Lost connection to MySQL server during query"), p)
    assert gone.startswith("The connection to lab dropped mid-query (the ssh tunnel may have closed)")
    denied = db.query_words(err(1142, "SELECT command denied to user 'me'@'%' for table 'x'"), p)
    assert denied.startswith("me is not allowed to do that on lab")
    pg = db.Profile(name="pg", dialect="postgres", host="h", user="me")
    assert "run out of disk space" in db.query_words(
        Exception("could not write to file: No space left on device"), pg)
    assert db.query_words(Exception("something odd"), pg) == "PostgreSQL on pg said: something odd"


# ---------------------------------------------------------------------------
# Profiles and secrets
# ---------------------------------------------------------------------------

def test_profiles_are_saved_without_passwords_and_read_back(home, monkeypatch):
    p = db.Profile(name="lab-db", dialect="mysql", host="db.lab.edu", user="ro",
                   database="study7", password="env")
    db.save_profile(p)
    assert [x.name for x in db.list_profiles()] == ["lab-db"]
    saved = (home / "settings.json").read_text(encoding="utf-8")
    assert "password" in saved and "secret" not in saved, "only *how*, never the value"
    assert db.get_profile("lab-db").label() == \
        "MySQL · ro@db.lab.edu · password from TATERS_DB_PASSWORD_LAB_DB"

    monkeypatch.setenv("TATERS_DB_PASSWORD_LAB_DB", "s3cret")
    assert db.password_for(db.get_profile("lab-db")) == "s3cret"
    monkeypatch.delenv("TATERS_DB_PASSWORD_LAB_DB")
    with pytest.raises(db.NotPermitted, match="TATERS_DB_PASSWORD_LAB_DB"):
        db.password_for(db.get_profile("lab-db"))
    asked = []
    assert db.password_for(db.get_profile("lab-db"), ask=lambda q: asked.append(q) or "typed") == "typed"
    assert asked == ["Password for ro on lab-db:"]
    assert db.password_for(db.get_profile("lab-db")) == "typed", "remembered for the session"
    db.delete_profile("lab-db")
    assert db.list_profiles() == []


def test_a_bad_profile_name_or_dialect_is_refused(home):
    with pytest.raises(ValueError):
        db.Profile(name="Lab DB", dialect="sqlite", path="x")
    with pytest.raises(ValueError):
        db.Profile(name="lab", dialect="oracle")


# ---------------------------------------------------------------------------
# The database writer
# ---------------------------------------------------------------------------

def test_a_table_is_typed_swapped_in_and_read_back_in_order(lab):
    ref = parse_ref("db://lab/main/taters/s7/features/read")
    rows = [["a", "1", "1.5", "hi"], ["b", "007", "nan", ""], ["c", "3", "", "yo"]]
    with tables.write_table(ref, ["text_id", "n", "x", "note"]) as w:
        w.writerows(rows)

    assert tables.exists(ref) and tables.count(ref) == 3
    assert tables.columns_of(ref) == ["text_id", "n", "x", "note"]
    got = list(tables.read_rows(ref))
    assert got[1] == {"text_id": "b", "n": "007", "x": "", "note": ""}, \
        "007 stays text; nan is NULL and reads back blank"
    s = db.session_for("lab")
    entry = db.registered(s, db.sanitize_table_name("taters/s7/features/read"))
    assert [c[2] for c in entry.columns] == ["text", "text", "float", "text"]
    assert entry.table_name == "taters__s7__features__read"
    schema = s.one("SELECT sql FROM sqlite_master WHERE name = ?", (entry.table_name,))
    assert '"x" REAL' in schema and '"_taters_row" INTEGER PRIMARY KEY' in schema
    s.close()

    with tables.write_table(ref, ["text_id", "n"]) as w:
        w.writerow(["z", "9"])
    assert list(tables.read_rows(ref)) == [{"text_id": "z", "n": "9"}], "replaced whole"


def test_hinted_columns_skip_the_typed_copy_and_still_type(lab):
    ref = parse_ref("db://lab/main/t/hinted")
    with tables.write_table(ref, ["text_id", "n", "f"],
                            hints={"text_id": "text", "n": "int", "f": "float"}) as w:
        w.writerow(["a", "4", "2.5"])
    s = db.session_for("lab")
    assert s.all('SELECT "n", "f" FROM "t__hinted"') == [(4, 2.5)]
    s.close()


def test_a_table_taters_did_not_make_is_never_replaced(lab):
    s = db.session_for("lab")
    s.run('CREATE TABLE "mine" ("a" TEXT)')
    s.close()
    with pytest.raises(db.DbError, match="Taters did not make"):
        with tables.write_table(parse_ref("db://lab/main/mine"), ["a"]) as w:
            w.writerow(["x"])
    s = db.session_for("lab")
    assert s.d.table_exists(s, "mine") and not s.d.table_exists(s, "mine__tload")
    s.close()


def test_an_interrupted_load_leaves_the_old_table_and_no_scratch(lab):
    ref = parse_ref("db://lab/main/t/keep")
    with tables.write_table(ref, ["a"]) as w:
        w.writerow(["old"])
    with pytest.raises(RuntimeError):
        with tables.write_table(ref, ["a"]) as w:
            w.writerow(["new"])
            raise RuntimeError("boom")
    assert list(tables.read_rows(ref)) == [{"a": "old"}]
    s = db.session_for("lab")
    names = [n for (n,) in s.all("SELECT name FROM sqlite_master WHERE type = 'table'")]
    assert not any(n.endswith(("__tload", "__tnew", "__told")) for n in names)
    s.close()


def test_sanitized_names_are_deterministic_and_short():
    assert db.sanitize_table_name("taters/study7/features/readability") == \
        "taters__study7__features__readability"
    assert db.sanitize_table_name("Taters/My Study!/7features") == "taters__my_study__t_7features"
    long = db.sanitize_table_name("a" * 100)
    assert len(long) <= 48 and long == db.sanitize_table_name("a" * 100)
    assert long != db.sanitize_table_name("a" * 99)


def test_a_table_too_wide_for_the_engine_is_packed_and_reads_back_the_same(lab, tmp_path):
    """Embeddings and document-term matrices have thousands of columns; the
    engines allow about a thousand. The measures go into one JSON column,
    the ids stay real columns, and nothing reading the table can tell."""
    cols = ["text_id", "source_col", *(f"d{i}" for i in range(2500))]
    rows = [[f"r{n}", "text", *(f"{(n * 7 + i) % 13 / 4:.3g}" for i in range(2500))]
            for n in range(5)]
    rows[2][5] = ""                                   # a blank survives the trip
    rows[3][6] = "007"                                # and so does an id-like cell
    ref = parse_ref("db://lab/main/taters/s/features/embeddings")
    hints = {"text_id": "text", "source_col": "text", **{f"d{i}": "float" for i in range(2500)}}
    with tables.write_table(ref, cols, hints=hints) as w:
        w.writerows(rows)
    entry = tables._entry(ref)
    assert entry.layout == "packed"
    assert entry.table_columns == ["text_id", "source_col", db.PACKED_COLUMN]
    assert tables.columns_of(ref) == cols
    back = list(tables.read_cells(ref))
    assert [r[0] for r in back] == [f"r{n}" for n in range(5)]
    assert back[2][5] == "" and back[3][6] == "007"
    assert [float(x) for x in back[1][2:]] == [float(x) for x in rows[1][2:]]
    # a few columns by name, from the real and the packed side alike
    picked = list(tables.read_rows(ref, columns=["d7", "text_id", "d2499"]))
    assert picked[4] == {"d7": rows[4][9], "text_id": "r4", "d2499": rows[4][-1]}
    # the fingerprint is over the logical cells, so a CSV copy agrees
    csv_ref = tmp_path / "emb.csv"
    with tables.write_table(csv_ref, cols) as w:
        w.writerows(rows)
    assert tables.fingerprint(csv_ref) == entry.fingerprint == tables.fingerprint(ref)
    # and the statistics' DataFrame reads it like any other table
    df = tables.read_str_table(ref)
    assert list(df.columns) == cols and df.shape == (5, 2502)


def test_a_packed_table_cannot_key_on_a_packed_column(lab):
    cols = [f"c{i}" for i in range(2000)]
    with pytest.raises(db.DbError, match="would be packed away"):
        with tables.write_table(parse_ref("db://lab/main/wide"), cols, unique_key="c5"):
            pass
    # no hints at all: every column that is not an id gets packed, and the
    # text that lands there still comes back as text
    with tables.write_table(parse_ref("db://lab/main/wide"), cols) as w:
        w.writerow(["x y"] + ["1"] * 1999)
    assert list(tables.read_cells("db://lab/main/wide")) == [["x y"] + ["1"] * 1999]


# ---------------------------------------------------------------------------
# Fingerprints: a file and a database copy of the same rows agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cells", [
    ["a", "3", "3.0", "007"],
    ["b", "-0.0", "nan", "x y"],
    ["c", "", "1e3", "  padded  "],
])
def test_the_fingerprint_of_a_file_equals_its_database_copy(lab, tmp_path, cells):
    f = write_rows(tmp_path / "t.csv", ["text_id", "n", "x", "s"], [cells, ["z", "1", "2", "3"]])
    ref = parse_ref("db://lab/main/t/copy")
    with tables.write_table(ref, ["text_id", "n", "x", "s"]) as w:
        for row in read_rows(f):
            w.writerow(row)
    assert tables.fingerprint(f) == tables.fingerprint(ref) == w.fingerprint


def test_the_fingerprint_ignores_row_order_but_not_duplicates():
    a = db.Fingerprint(["x"])
    b = db.Fingerprint(["x"])
    for cells in (["1"], ["2"]):
        a.add(cells)
    for cells in (["2"], ["1"]):
        b.add(cells)
    assert a.hexdigest() == b.hexdigest()
    c = db.Fingerprint(["x"])
    for cells in (["1"], ["1"]):
        c.add(cells)
    d = db.Fingerprint(["x"])
    d.add(["1"])
    assert c.hexdigest() != d.hexdigest()


def test_cell_tokens_canonicalize_by_value():
    assert db.cell_token("3") == db.cell_token("3.0") == db.cell_token(" 3 ")
    assert db.cell_token("007") != db.cell_token("7"), "a leading zero is text"
    assert db.cell_token("") == db.cell_token("nan") == db.cell_token("inf")
    assert db.cell_token("12345678901234567") != db.cell_token("12345678901234568")


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def test_the_probe_reports_what_the_account_may_do(lab):
    p = db.get_profile("lab")
    pr = db.probe(p)
    assert pr.can_read and pr.can_write and pr.version.startswith("SQLite")
    lines = pr.sentences(p)
    assert lines[0].startswith("Connected to lab (SQLite")
    assert "Can read: yes" in lines[1] and "Can create tables: yes." in lines
    s = db.session_for("lab")
    assert not [n for (n,) in s.all("SELECT name FROM sqlite_master") if "probe" in n], \
        "the probe cleans up after itself"
    s.close()


def test_a_read_only_probe_says_where_results_can_go(lab, monkeypatch):
    p = db.get_profile("lab")
    pr = db.Probe(select=True, create=False, drop=False, registry=False,
                  version="SQLite 3", tables=5)
    text = " ".join(pr.sentences(p))
    assert "Can create tables: no." in text
    assert "results will have to go to files or to a database that allows new tables" in text


# ---------------------------------------------------------------------------
# Sources: a table plus joins, rendered and read back
# ---------------------------------------------------------------------------

def _spec(*joins, columns=None):
    return sources.Source(connection="lab", table="posts", alias="p", joins=tuple(joins),
                          columns=columns or {}, order_by=("p.id",))


def test_a_source_round_trips_through_yaml_shape_and_renders_quoted_sql(lab):
    spec = _spec(sources.Join("users", "u", on=(("p.user_id", "id"),)))
    again = sources.Source.from_dict(spec.to_dict())
    assert again == spec
    s = db.session_for("lab")
    sql, params = sources.render(spec, s)
    assert sql.startswith('SELECT "p"."id" AS "id", "p"."user_id" AS "user_id"')
    assert 'LEFT JOIN "users" AS "u" ON "p"."user_id" = "u"."id"' in sql
    assert sql.endswith('ORDER BY "p"."id"') and params == ()
    names = [n for n, _a, _c in sources.resolve_columns(spec, s)]
    assert names == ["id", "user_id", "body", "created_at", "score", "age", "cond",
                     "users__created_at"], "a repeated name is prefixed, merge's rule"
    s.close()


def test_a_dropped_column_or_table_is_found_before_anything_runs(lab):
    s = db.session_for("lab")
    with pytest.raises(ValueError, match="no table called gone"):
        sources.resolve_columns(sources.Source(connection="lab", table="gone"), s)
    with pytest.raises(ValueError, match="no longer has a column called nope"):
        sources.resolve_columns(_spec(columns={"x": "p.nope"}), s)
    s.close()


def test_the_join_report_reads_back_the_merge_sentences(lab):
    spec = _spec(sources.Join("users", "u", on=(("p.user_id", "id"),)),
                 sources.Join("outcomes", "o", on=(("p.id", "post_id"),)))
    s = db.session_for("lab")
    users, outcomes = sources.join_report(spec, s)
    s.close()

    assert users.relationship == "many-to-one"
    assert users.left.rows_matched == 3 and users.left.unmatched_examples == ("9",)
    assert users.right.rows_unmatched == 1 and users.right.unmatched_examples == ("3",)
    assert users.rows_out == {"inner": 3, "left": 4, "outer": 5}
    lines = users.sentences()
    assert lines[0] == "3 of 4 rows in posts match a row in users; 1 do not (e.g. 9)."
    assert lines[1] == "1 of 3 rows in users have no match in posts (e.g. 3)."
    clash = next(c for c in users.clashes if c.name == "created_at")
    assert not clash.agrees and clash.output_names == ("created_at", "users__created_at")

    assert outcomes.relationship == "one-to-many", "post 10 has two outcome rows"
    assert outcomes.rows_out["left"] == 5 and outcomes.left.file == "posts + users"
    assert any("repeat once per match" in line for line in outcomes.sentences())


def test_many_to_many_is_named_with_the_worst_keys(lab):
    s = db.session_for("lab")
    s.run("INSERT INTO posts VALUES (14, 1, 'more', 'p5', 1.0)")
    spec = sources.Source(connection="lab", table="posts", alias="p", joins=(
        sources.Join("outcomes", "o", on=(("p.user_id", "wave"),)),))
    (report,) = sources.join_report(spec, s)
    s.close()
    assert report.relationship == "many-to-many"
    assert report.many_to_many_examples[0][0] == "1"
    assert "appear more than once in both" in report.refusal()


# ---------------------------------------------------------------------------
# Export: a table or a join, out to a spreadsheet
# ---------------------------------------------------------------------------

def test_export_writes_a_join_to_csv_with_only_select_needed(lab, tmp_path):
    spec = _spec(sources.Join("users", "u", on=(("p.user_id", "id"),)))
    out = tables.export_table(source=spec.to_dict(), out_csv=tmp_path / "out.csv",
                              columns=["id", "body", "age"], verbose=False)
    rows = read_rows(out)
    assert list(rows[0]) == ["id", "body", "age"]
    assert [r["age"] for r in rows] == ["30", "30", "25", ""], "the orphan's age is blank"
    assert tables.export_table(source="db://lab/main/users", out_csv=tmp_path / "u.csv",
                               verbose=False).is_file()
    assert len(read_rows(tmp_path / "u.csv")) == 3


def test_export_is_on_the_command_line(lab, tmp_path):
    from taters.helpers.tables import main
    out = tmp_path / "cli.csv"
    main(["export", "--source", "db://lab/main/users", "--out-csv", str(out), "--verbose", "false"])
    assert out.is_file() and len(read_rows(out)) == 3


def test_import_taters_loads_no_database_driver():
    import subprocess
    import sys
    code = ("import sys, taters; import taters.helpers.tables, taters.helpers.db; "
            "print(any(m in sys.modules for m in ('pymysql', 'psycopg')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_connect_refuses_inside_a_pool_worker(lab, monkeypatch):
    import multiprocessing
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    with pytest.raises(db.DbError, match="main process"):
        db.connect(db.get_profile("lab"))


def test_a_port_that_is_not_a_database_is_said_in_words_not_a_traceback(home, monkeypatch):
    """From a real session: an SSH host given as the database server. The
    driver raised an error type nothing caught, and the app fell over with a
    traceback instead of saying what answered."""
    pymysql = pytest.importorskip("pymysql")

    def refuse(**kwargs):
        raise pymysql.err.InternalError(
            "Packet sequence number wrong - got 45 expected 0")
    monkeypatch.setattr(pymysql, "connect", refuse)
    profile = db.Profile(name="myserver", dialect="mysql", host="myserver.example.edu",
                         user="me", password="ask")
    result = db.probe(profile, "pw")
    assert not result.select
    said = " ".join(result.notes)
    assert "answered on port 3306, but as an SSH server, not as MySQL" in said
    assert "SSH tunnel" in said and "localhost" in said


# ---------------------------------------------------------------------------
# A database behind an ssh host
# ---------------------------------------------------------------------------

def test_a_tunneled_profile_round_trips_and_names_the_ssh_host(home):
    prof = db.Profile(name="myserver", dialect="mysql", host="localhost", user="me",
                      database="study", ssh_host="myserver.example.edu", ssh_user="me",
                      ssh_port=2200, ssh_key="~/.ssh/id_lab")
    db.save_profile(prof)
    (back,) = db.list_profiles()
    assert back == prof and back.tunneled
    assert back.ssh_address == "me@myserver.example.edu:2200"
    assert back.where == "study on localhost via ssh me@myserver.example.edu:2200"
    cmd = db.tunnel_command(back, 51234)
    assert cmd[0] == "ssh" and "-N" in cmd and "BatchMode=yes" in cmd
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:51234:localhost:3306"
    assert cmd[cmd.index("-p") + 1] == "2200"
    assert cmd[cmd.index("-i") + 1].endswith(".ssh/id_lab") and "~" not in cmd[cmd.index("-i") + 1]
    assert cmd[-1] == "me@myserver.example.edu"
    assert db._ssl_context(back) is None, "the tunnel is the encryption"


def test_ssh_complaints_become_sentences_that_say_what_to_do():
    p = db.Profile(name="lab", dialect="postgres", host="localhost", ssh_host="h.edu",
                   ssh_user="me", ssh_key="/k")
    assert "the key was not accepted" in db._ssh_words("me@h.edu: Permission denied (publickey).", p)
    assert "`ssh me@h.edu` works without a prompt" in db._ssh_words("Permission denied", p)
    on_port = db.with_fields(p, ssh_port=2200)
    assert "`ssh -p 2200 me@h.edu`" in db._ssh_words("Permission denied", on_port), \
        "the port goes on a flag; user@host:port is not something ssh accepts"
    assert "VPN" in db._ssh_words("ssh: connect to host h.edu port 22: Connection timed out", p)
    assert "accept its key" in db._ssh_words("Host key verification failed.", p)
    assert "AllowTcpForwarding" in db._ssh_words(
        "channel 2: open failed: administratively prohibited: open failed\n"
        "ssh: forwarding failed", p)


def test_a_tunnel_that_cannot_open_is_refused_in_words(home, monkeypatch, tmp_path):
    """A fake ssh that exits the way the real one does on a bad key."""
    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\necho 'me@h.edu: Permission denied (publickey).' >&2\nexit 255\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:" + __import__("os").environ["PATH"])
    p = db.Profile(name="lab", dialect="mysql", host="localhost", user="u", password="ask",
                   ssh_host="h.edu", ssh_user="me")
    with pytest.raises(db.DbError, match="the key was not accepted"):
        db.open_tunnel(p, timeout=5)
    assert db._TUNNELS == {}


@pytest.mark.skipif(not __import__("os").environ.get("TATERS_TEST_SSH_KEY"),
                    reason="needs the docker ssh+mysql pair; see tests/README.md")
def test_a_database_behind_ssh_is_reached_through_the_tunnel(home):
    import os

    prof = db.Profile(name="t", dialect="mysql", host="taters-mysql", port=3306,
                      user="taters", database="taters_test", password="env",
                      ssh_host="127.0.0.1", ssh_port=2222, ssh_user="taters",
                      ssh_key=os.environ["TATERS_TEST_SSH_KEY"])
    db.save_profile(prof)
    os.environ["TATERS_DB_PASSWORD_T"] = "taterspw"
    try:
        result = db.probe(prof)
        assert result.select and result.can_write, result.notes
        assert "t" in db._TUNNELS
        ref = parse_ref("db://t/taters_test/taters/sshtest/x")
        with tables.write_table(ref, ["text_id", "a"]) as w:
            w.writerow(["r1", "1"])
        assert list(tables.read_rows(ref)) == [{"text_id": "r1", "a": "1"}]
    finally:
        tables.close_sessions()
        os.environ.pop("TATERS_DB_PASSWORD_T", None)
    assert db._TUNNELS == {}, "closed with the sessions"
    # a tunnel that opens onto nothing is said as such, not as a lost connection
    nowhere = db.with_fields(prof, name="t4", host="nowhere.invalid", password="ask")
    said = " ".join(db.probe(nowhere, "taterspw").notes)
    assert "nothing answered at nowhere.invalid:3306 on that machine" in said
    tables.close_sessions()


def test_a_putty_key_opens_the_tunnel_with_plink(home, monkeypatch, tmp_path):
    """A .ppk is PuTTY's format, which OpenSSH cannot read: the tunnel is
    then plink's, with its own flags, its own hint and its own complaints."""
    p = db.Profile(name="lab", dialect="mysql", host="localhost", user="u", password="ask",
                   ssh_host="h.edu", ssh_user="me", ssh_port=2200, ssh_key="~/keys/lab.ppk")
    assert db.uses_putty(p) and not db.uses_putty(db.with_fields(p, ssh_key="~/keys/lab"))
    monkeypatch.setattr(db, "plink_path", lambda: "/opt/putty/plink")
    cmd = db.tunnel_command(p, 51234)
    assert cmd[0] == "/opt/putty/plink" and cmd[1:5] == ["-ssh", "-batch", "-v", "-N"]
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:51234:localhost:3306"
    assert cmd[cmd.index("-P") + 1] == "2200"
    assert cmd[cmd.index("-i") + 1].endswith("keys/lab.ppk") and cmd[-1] == "me@h.edu"
    assert db.ssh_hint(p) == "plink -P 2200 -i lab.ppk me@h.edu"
    assert "the key was not accepted" in db._ssh_words("Server refused our key", p)
    assert "Pageant" in db._ssh_words("Unable to use key file \"lab.ppk\" (unable to open file)", p)
    assert "with PuTTY to accept its key" in db._ssh_words(
        "The server's host key is not cached in the registry.", p)
    assert "Could not reach" in db._ssh_words("FATAL ERROR: Network error: Connection refused", p)
    # without plink, the way out is named: install PuTTY, or convert the key
    monkeypatch.setattr(db, "plink_path", lambda: None)
    with pytest.raises(db.DbError, match="plink.*not found.*Export OpenSSH key"):
        db.open_tunnel(p, timeout=5)


def test_plink_is_found_in_puttys_usual_windows_folder(monkeypatch, tmp_path):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "x86"))
    assert db.plink_path() is None
    (tmp_path / "PuTTY").mkdir()
    (tmp_path / "PuTTY" / "plink.exe").write_bytes(b"")
    assert db.plink_path() == str(tmp_path / "PuTTY" / "plink.exe")


@pytest.mark.parametrize("program, lines, expect", [
    ("plink", ['Using username "me".', "Access granted"],
     "logged in to me@h.edu, but the tunnel to localhost:3306 never opened"),
    ("ssh", ["debug1: Connecting to h.edu port 22.", "debug1: Connection established."],
     "reached me@h.edu but did not finish logging in"),
    ("ssh", [], "did not reach me@h.edu"),
])
def test_a_tunnel_that_stalls_says_how_far_it_got(home, monkeypatch, tmp_path, program,
                                                   lines, expect):
    """A login that hangs -- a passphrase nobody can type, a second factor,
    a server that quietly refuses to forward -- used to be a bare "did not
    open within 20 seconds". Now it is the story, from the program's own
    account of what it was doing."""
    fake = tmp_path / program
    body = "\n".join(f"echo '{ln}' >&2" for ln in lines)
    fake.write_text(f"#!/bin/sh\n{body}\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:" + __import__("os").environ["PATH"])
    monkeypatch.setattr(db, "plink_path", lambda: str(fake))
    p = db.Profile(name="lab", dialect="mysql", host="localhost", user="u", password="ask",
                   ssh_host="h.edu", ssh_user="me",
                   ssh_key=str(tmp_path / ("k.ppk" if program == "plink" else "k")))
    with pytest.raises(db.DbError) as caught:
        db.open_tunnel(p, timeout=1.5)
    said = str(caught.value)
    assert expect in said
    assert "Its last words:" in said
    if lines:
        assert lines[-1] in said, "the program's own account is quoted"
    assert db._TUNNELS == {}


def test_an_unknown_host_key_is_shown_and_can_be_accepted(home, monkeypatch, tmp_path):
    """The one refusal a person can settle on the spot. ssh's way is a line
    in known_hosts (what a "yes" by hand would have written); plink's is a
    fingerprint the profile hands it every time."""
    import base64
    import subprocess

    fake = tmp_path / "ssh"
    fake.write_text("#!/bin/sh\necho 'Host key verification failed.' >&2\nexit 255\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:" + __import__("os").environ["PATH"])
    blob = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519" + b"\x00" * 36).decode()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout=f"# h.edu:2200 SSH-2.0\n[h.edu]:2200 ssh-ed25519 {blob}\n", stderr=""))
    monkeypatch.setattr(db, "known_hosts_path", lambda: tmp_path / "known_hosts")
    p = db.Profile(name="lab", dialect="mysql", host="localhost", user="u", password="ask",
                   ssh_host="h.edu", ssh_user="me", ssh_port=2200)
    with pytest.raises(db.HostKeyUnknown) as caught:
        db.open_tunnel(p, timeout=5)
    (key,) = caught.value.keys
    assert key.type == "ssh-ed25519" and key.fingerprint.startswith("SHA256:")
    assert key.fingerprint == db._fingerprint(blob)
    # probe lets this one through, so a screen can ask about it
    with pytest.raises(db.HostKeyUnknown):
        db.probe(p, "pw")
    same = db.accept_host_keys(p, caught.value.keys)
    assert same == p
    assert (tmp_path / "known_hosts").read_text() == f"[h.edu]:2200 ssh-ed25519 {blob}\n"
    db.accept_host_keys(p, caught.value.keys)
    assert (tmp_path / "known_hosts").read_text().count(blob) == 1, "never twice"
    # plink: the fingerprint rides on the profile and on the command line
    putty = db.with_fields(p, ssh_key="/k.ppk")
    monkeypatch.setattr(db, "plink_path", lambda: "/opt/plink")
    kept = db.accept_host_keys(putty, caught.value.keys)
    assert kept.ssh_hostkey == key.fingerprint
    assert db.get_profile("lab").ssh_hostkey == key.fingerprint, "saved"
    cmd = db.tunnel_command(kept, 5555)
    assert cmd[cmd.index("-hostkey") + 1] == key.fingerprint


def test_without_keyscan_the_key_is_fetched_through_ssh_itself(tmp_path, monkeypatch):
    """A machine with ssh but no ssh-keyscan (or one keyscan cannot reach):
    ssh is told to write the key it is shown into a throwaway known_hosts,
    and that is what the person is shown. Nothing real trusts it until
    they say so."""
    import base64
    import shutil

    blob = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519" + b"\x00" * 36).decode()
    fake = tmp_path / "ssh"
    # the fake finds the UserKnownHostsFile= option and writes a line there,
    # then fails the login the way the real one would with no key offered
    fake.write_text("#!/bin/sh\nfor a in \"$@\"; do case \"$a\" in UserKnownHostsFile=*) "
                    "f=\"${a#UserKnownHostsFile=}\";; esac; done\n"
                    f"echo '[h.edu]:2200 ssh-ed25519 {blob}' > \"$f\"\n"
                    "echo 'Permission denied (publickey).' >&2\nexit 255\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:" + __import__("os").environ["PATH"])
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which",
                        lambda name, *a, **k: None if name == "ssh-keyscan" else real_which(name, *a, **k))
    p = db.Profile(name="lab", dialect="mysql", host="localhost", user="u",
                   ssh_host="h.edu", ssh_user="me", ssh_port=2200)
    (key,) = db.host_keys(p)
    assert key.type == "ssh-ed25519" and key.blob == blob
    assert key.fingerprint == db._fingerprint(blob)


def test_the_fingerprint_matches_ssh_keygen(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ssh-keygen"):
        pytest.skip("no ssh-keygen here")
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "k")],
                   check=True)
    blob = (tmp_path / "k.pub").read_text().split()[1]
    expected = subprocess.run(["ssh-keygen", "-lf", str(tmp_path / "k.pub")],
                              capture_output=True, text=True, check=True).stdout.split()[1]
    assert db._fingerprint(blob) == expected


def test_plinks_refusal_carries_the_fingerprint_itself():
    p = db.Profile(name="lab", dialect="mysql", host="localhost", ssh_host="h.edu",
                   ssh_key="/k.ppk")
    log = ("The host key is not cached for this server:\n  h.edu (port 22)\n"
           "You have no guarantee that the server is the computer you think it is.\n"
           "The server's ssh-ed25519 key fingerprint is:\n"
           "  ssh-ed25519 255 SHA256:abcDEF123/+xyz\n"
           "Connection abandoned.")
    (key,) = db.host_keys(p, log) if not __import__("shutil").which("ssh-keyscan") else \
        [k for k in db.host_keys(db.with_fields(p, ssh_host="nowhere.invalid"), log)]
    assert key.fingerprint == "SHA256:abcDEF123/+xyz" and key.type == "ssh-ed25519"


def test_the_probe_says_what_it_is_doing_at_each_step(lab):
    steps = []
    result = db.probe(db.get_profile("lab"), on_step=steps.append)
    assert result.can_write
    assert steps == [
        "Opening lab.sqlite",
        "Asking the server its version",
        "Counting the tables this account can see",
        "Checking whether this account can create a table",
        "...and remove it again",
        "Setting up the small registry Taters keeps of its tables",
    ]
    assert result.tables == 6, "counted in one query, not described one by one"


# ---------------------------------------------------------------------------
# What the ssh host's own client knows
# ---------------------------------------------------------------------------

def test_the_look_reads_the_saved_mysql_login_and_the_socket():
    p = db.Profile(name="lab", dialect="mysql", ssh_host="h.edu", ssh_user="me")
    out = ("--mycnf--\n[client]\nuser = lab_ro\npassword = \"p@ss:word\"\n# a comment\n"
           "[mysqld]\nuser = mysql\n--pgpass--\n--sockets--\nmysql=/run/mysqld/mysqld.sock\n"
           "--user--\nme\n")
    h = db.parse_login_hints(out, p)
    assert (h.user, h.password, h.socket, h.source, h.os_user) == \
        ("lab_ro", "p@ss:word", "/run/mysqld/mysqld.sock", "~/.my.cnf", "me")
    assert h.found
    nothing = db.parse_login_hints("--mycnf--\n--pgpass--\n--sockets--\n--user--\nme\n", p)
    assert not nothing.found and nothing.os_user == "me"


def test_the_look_reads_pgpass_for_the_right_database():
    p = db.Profile(name="lab", dialect="postgres", database="study", ssh_host="h.edu")
    out = ("--mycnf--\n--pgpass--\nlocalhost:5432:other:bob:nope\n"
           "localhost:5432:study:ann:yes:colon\n*:*:*:fallback:z\n--sockets--\n"
           "postgres=/var/run/postgresql/.s.PGSQL.5432\n--user--\nann\n")
    h = db.parse_login_hints(out, p)
    assert (h.user, h.password, h.source) == ("ann", "yes:colon", "~/.pgpass")
    assert h.socket == "/var/run/postgresql/.s.PGSQL.5432"


def test_a_socket_tunnel_forwards_to_the_socket_and_plink_cannot(home):
    p = db.Profile(name="lab", dialect="mysql", user="me", password="none", ssh_host="h.edu",
                   ssh_user="me", socket="/run/mysqld/mysqld.sock")
    cmd = db.tunnel_command(p, 5555)
    assert cmd[cmd.index("-L") + 1] == "127.0.0.1:5555:/run/mysqld/mysqld.sock"
    assert "localhost" not in cmd[cmd.index("-L") + 1]
    with pytest.raises(db.DbError, match="plink cannot forward"):
        db.tunnel_command(db.with_fields(p, ssh_key="/k.ppk"), 5555)
    assert p.where == " on /run/mysqld/mysqld.sock via ssh me@h.edu".replace(" on", "", 1) \
        or "via ssh me@h.edu" in p.where


def test_the_server_password_mode_reads_the_saved_login_each_session(home, monkeypatch):
    p = db.Profile(name="lab", dialect="mysql", host="localhost", user="lab_ro",
                   password="server", ssh_host="h.edu", ssh_user="me")
    db.save_profile(p)
    calls = []
    monkeypatch.setattr(db, "remote_login_hints", lambda profile: (
        calls.append(profile.name), db.LoginHints(user="lab_ro", password="s3cret",
                                                  source="~/.my.cnf"))[1])
    assert db.password_for(p) == "s3cret"
    assert db.password_for(p) == "s3cret" and calls == ["lab"], "once per session"
    db.forget_password("lab")
    monkeypatch.setattr(db, "remote_login_hints", lambda profile: db.LoginHints(os_user="me"))
    with pytest.raises(db.NotPermitted, match="saved login"):
        db.password_for(p)
    assert "s3cret" not in (home / "settings.json").read_text(encoding="utf-8")


@pytest.mark.skipif(not __import__("os").environ.get("TATERS_TEST_SSH_KEY"),
                    reason="needs the docker ssh container; see tests/README.md")
def test_a_database_that_trusts_the_ssh_login_needs_no_password_through_its_socket(home):
    """The test container runs a MariaDB on its socket only, where the user
    taters logs in by socket authentication: the shape of a lab server where
    `mysql` just launches once you are logged in."""
    import os

    key = os.environ["TATERS_TEST_SSH_KEY"]
    look = db.Profile(name="lk", dialect="mysql", database="socket_test", password="ask",
                      ssh_host="127.0.0.1", ssh_port=2222, ssh_user="taters", ssh_key=key)
    hints = db.remote_login_hints(look)
    assert hints.socket.endswith("mysqld.sock") and hints.os_user == "taters"
    prof = db.Profile(name="sk", dialect="mysql", user="taters", database="socket_test",
                      password="none", ssh_host="127.0.0.1", ssh_port=2222, ssh_user="taters",
                      ssh_key=key, socket=hints.socket)
    db.save_profile(prof)
    try:
        result = db.probe(prof)
        assert result.select and result.can_write, result.notes
        ref = parse_ref("db://sk/socket_test/taters/sock/x")
        with tables.write_table(ref, ["text_id", "a"]) as w:
            w.writerow(["r1", "1"])
        assert list(tables.read_rows(ref)) == [{"text_id": "r1", "a": "1"}]
    finally:
        tables.close_sessions()


def test_an_ssh_host_that_reads_as_an_option_is_refused():
    """ssh -oProxyCommand=... runs a local command: a host or a login that
    starts with a dash must never reach the command line."""
    for bad in ("-oProxyCommand=calc", "-v", "h e.edu", "h;edu"):
        with pytest.raises(ValueError, match="ssh host"):
            db.Profile(name="lab", dialect="mysql", ssh_host=bad)
    with pytest.raises(ValueError, match="ssh user"):
        db.Profile(name="lab", dialect="mysql", ssh_host="h.edu", ssh_user="-oX")
    assert db.Profile(name="lab", dialect="mysql", ssh_host="my-server.example.edu",
                      ssh_user="me_2").ssh_address == "me_2@my-server.example.edu"


def test_a_found_password_never_shows_in_a_repr():
    h = db.LoginHints(user="u", password="s3cret", source="~/.my.cnf")
    assert "s3cret" not in repr(h) and "s3cret" not in str(h)
