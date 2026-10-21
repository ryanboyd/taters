"""
The database layer against real servers: MySQL and PostgreSQL, when
``TATERS_TEST_MYSQL_URL`` / ``TATERS_TEST_POSTGRES_URL`` point at one (see
tests/README.md for the two docker commands). Skipped otherwise. SQLite
covers the same code every day; this is what catches the things only a
driver and a server can -- a placeholder the driver reads as a format
string, a cursor that needs a transaction, a column alias one engine
refuses -- each of which was found exactly this way.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import pytest

from taters.helpers import db, sources, tables

URLS = {"mysql": os.environ.get("TATERS_TEST_MYSQL_URL", ""),
        "postgres": os.environ.get("TATERS_TEST_POSTGRES_URL", "")}


@pytest.fixture(params=["mysql", "postgres"])
def live(request, tmp_path, monkeypatch):
    """A profile called ``t`` for one real server, password from the URL."""
    kind = request.param
    url = URLS[kind]
    if not url:
        pytest.skip(f"set TATERS_TEST_{kind.upper()}_URL to run against a real {kind}")
    pytest.importorskip("pymysql" if kind == "mysql" else "psycopg")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    tables.close_sessions()
    db._SECRETS.clear()
    u = urlparse(url)
    prof = db.Profile(name="t", dialect=kind, host=u.hostname or "127.0.0.1",
                      port=u.port or 0, user=u.username or "", database=u.path.lstrip("/"),
                      password="env", ssl="off")
    db.save_profile(prof)
    monkeypatch.setenv("TATERS_DB_PASSWORD_T", u.password or "")
    s = db.connect(prof)
    for name in ("live_posts", "live_users"):
        s.run(f"DROP TABLE IF EXISTS {name}")
    s.run("CREATE TABLE live_users(id INTEGER PRIMARY KEY, age INTEGER)")
    s.run("CREATE TABLE live_posts(id INTEGER PRIMARY KEY, user_id INTEGER, body TEXT, "
          "FOREIGN KEY (user_id) REFERENCES live_users(id))")
    s.run("INSERT INTO live_users VALUES (1, 30), (2, 40)")
    s.run("INSERT INTO live_posts VALUES (10, 1, 'hello there'), (11, 1, 'again'), "
          "(12, NULL, 'orphan')")
    s.close()
    tables.close_sessions()
    yield prof
    tables.close_sessions()


def test_the_probe_lists_tables_and_can_write(live):
    result = db.probe(live)
    assert result.select and result.can_write, result.notes
    s = db.connect(live)
    names = [t.name for t in s.d.list_tables(s, live.schema)]
    s.close()
    assert {"live_posts", "live_users"} <= set(names)


def test_a_table_is_written_typed_and_read_back(live):
    ref = tables.parse_ref("db://t/taters/live/features/x")
    with tables.write_table(ref, ["text_id", "a", "b"]) as w:
        w.writerow(["r1", "1", "x"])
        w.writerow(["r2", "2.5", ""])
    assert list(tables.read_rows(ref)) == [{"text_id": "r1", "a": "1.0", "b": "x"},
                                           {"text_id": "r2", "a": "2.5", "b": ""}]
    assert tables.delete_table(ref) and not tables.exists(ref)


def test_a_wide_table_is_packed_on_the_server(live):
    ref = tables.parse_ref("db://t/taters/live/features/emb")
    cols = ["text_id"] + [f"d{i}" for i in range(1500)]
    with tables.write_table(ref, cols, hints={"text_id": "text",
                                              **{c: "float" for c in cols[1:]}}) as w:
        w.writerow(["r1"] + ["0.5"] * 1500)
    assert tables._entry(ref).layout == "packed"
    assert len(next(tables.read_cells(ref))) == 1501
    tables.delete_table(ref)


def test_the_join_report_and_the_gather_agree_with_sqlite(live):
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    spec = sources.coerce({"connection": "t", "from": {"table": "live_posts", "as": "p"},
                           "join": [{"table": "live_users", "as": "u",
                                     "on": [["p.user_id", "u.id"]]}]})
    s = db.connect(live)
    (report,) = sources.join_report(spec, s)
    s.close()
    assert report.relationship == "many-to-one"
    assert report.left.rows_matched == 2 and report.right.rows_unmatched == 1
    out = csv_to_analysis_ready_csv(csv_path=spec.to_dict(),
                                    out_csv="db://t/taters/live/gathered/texts",
                                    text_cols=["body"], id_cols=["id"], carry_cols=["age"],
                                    verbose=False, overwrite_existing=True)
    rows = list(tables.read_rows(out))
    assert [r["text_id"] for r in rows] == ["10", "11", "12"]
    assert [r["age"] for r in rows] == ["30", "30", ""]
    tables.delete_table(out)


def test_a_large_gather_streams_and_writes_on_the_same_profile(live):
    """Reading through an unbuffered cursor while the loader inserts needs
    two connections; MySQL refuses a second command on one."""
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    s = db.connect(live)
    s.run("DROP TABLE IF EXISTS live_big")
    s.run("CREATE TABLE live_big(id INTEGER PRIMARY KEY, grp INTEGER, body TEXT)")
    s.executemany("INSERT INTO live_big VALUES (?, ?, ?)",
                  [(i, i % 7, f"text number {i}") for i in range(1, 3501)])
    s.close()
    tables.close_sessions()
    out = csv_to_analysis_ready_csv(
        csv_path={"connection": "t", "from": {"table": "live_big", "as": "b"},
                  "order_by": ["b.id"]},
        out_csv="db://t/taters/live/gathered/big", text_cols=["body"], group_by=["grp"],
        verbose=False, overwrite_existing=True)
    assert tables.count(out) == 7
    assert next(tables.read_rows(out))["group_count"] == "500"
    tables.delete_table(out)
