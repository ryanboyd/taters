"""
The gather from a database agrees with the gather from a file, row for row.

Same text, once in a CSV and once in a SQLite table: ``csv_to_analysis_ready_csv``
on each must yield tables with the same fingerprint, across every shape the
gather has -- grouped and not, texts joined or kept separate, columns carried
along, numbers summarized, rows filtered. Then the analyzers run on the
database copy, which is what the pipeline will do.
"""
from __future__ import annotations

import sqlite3

import pytest

from csvhelpers import read_rows, write_rows
from taters.helpers import db, tables
from taters.helpers.text_gather import csv_to_analysis_ready_csv


ROWS = [
    # pid, session, cond, score, text
    ["p1", "1", "A", "3", "first words"],
    ["p1", "2", "A", "5", "second words"],
    ["p2", "1", "B", "", "lonely words"],
    ["p2", "2", "B", "7", ""],
    ["p3", "1", "A", "2", "third person"],
]
HEADER = ["pid", "session", "cond", "score", "text"]


@pytest.fixture
def both(tmp_path, monkeypatch):
    """The same rows as a CSV and as a sqlite table behind a profile."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    tables.close_sessions()
    db._SECRETS.clear()
    csv_path = write_rows(tmp_path / "study.csv", HEADER, ROWS)
    dbfile = tmp_path / "lab.sqlite"
    con = sqlite3.connect(dbfile)
    con.execute("CREATE TABLE study(rowid_ INTEGER PRIMARY KEY, pid TEXT, session INTEGER, "
                "cond TEXT, score REAL, text TEXT)")
    for i, r in enumerate(ROWS, 1):
        con.execute("INSERT INTO study VALUES (?, ?, ?, ?, ?, ?)",
                    (i, r[0], int(r[1]), r[2], float(r[3]) if r[3] else None, r[4] or None))
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(dbfile)))
    yield csv_path, "db://lab/main/study"
    tables.close_sessions()


def _gather(source, out, **kw):
    return csv_to_analysis_ready_csv(csv_path=source, out_csv=out, verbose=False, **kw)


@pytest.mark.parametrize("kw", [
    dict(text_cols=["text"], id_cols=["pid", "session"]),
    dict(text_cols=["text"], id_cols=["pid", "session"], carry_cols=["cond", "score"]),
    dict(text_cols=["text"], group_by=["pid"]),
    dict(text_cols=["text"], group_by=["pid"], carry_cols=["cond", "session"],
         agg_cols=["score"]),
    dict(text_cols=["text"], group_by=["cond"], mode="separate"),
    dict(text_cols=["text"], id_cols=["pid", "session"],
         row_filters=[["cond", "in", ["A"]]]),
    dict(text_cols=[], group_by=["pid"], agg_cols={"score": "sum"}),
], ids=["rows", "rows+carry", "grouped", "grouped+carry+agg", "separate", "filtered",
        "no-text"])
def test_a_file_and_a_table_gather_identically(both, tmp_path, kw):
    csv_path, table = both
    if "group_by" in kw and "id_cols" not in kw:
        kw = {**kw}
    from_file = _gather(csv_path, tmp_path / "from_file.csv", **kw)
    from_db = _gather(table, "db://lab/main/taters/gathered",
                      **({**kw} if "group_by" not in kw else
                         {**kw}))
    assert tables.columns_of(from_file) == tables.columns_of(from_db)
    assert tables.fingerprint(from_file) == tables.fingerprint(from_db)
    assert len(read_rows(from_file)) == tables.count(from_db) > 0


def test_the_grouped_order_follows_the_table_key(both, tmp_path):
    """Texts are joined in key order, so the result is the same every run."""
    _csv, table = both
    out = _gather(table, "db://lab/main/taters/g", text_cols=["text"], group_by=["pid"])
    rows = {r["text_id"]: r for r in tables.read_rows(out)}
    assert rows["p1"]["text"] == "first words second words"
    assert rows["p1"]["group_count"] == "2" and rows["p2"]["group_count"] == "1"


def test_a_table_without_a_key_needs_id_cols(both, tmp_path):
    _csv, _table = both
    s = db.session_for("lab")
    s.run("CREATE TABLE nokey AS SELECT pid, text FROM study")
    s.close()
    with pytest.raises(ValueError, match="numbered by its position"):
        _gather("db://lab/main/nokey", "db://lab/main/taters/x", text_cols=["text"])
    with pytest.raises(ValueError, match="order_by"):
        _gather("db://lab/main/nokey", "db://lab/main/taters/x", text_cols=["text"], group_by=["pid"])
    out = _gather("db://lab/main/nokey", "db://lab/main/taters/y", text_cols=["text"], id_cols=["pid"])
    assert tables.count(out) == 4


def test_the_base_key_is_the_default_id(both):
    _csv, table = both
    out = _gather(table, "db://lab/main/taters/keyed", text_cols=["text"])
    assert [r["text_id"] for r in tables.read_rows(out)] == ["1", "2", "3", "5"], \
        "rowid_ is the key; the textless row 4 is left out"


def test_a_gather_to_a_file_from_a_database_is_allowed(both, tmp_path):
    """Anything in, anything out: a file destination for a database source."""
    _csv, table = both
    out = _gather(table, tmp_path / "out.csv", text_cols=["text"], id_cols=["pid", "session"])
    assert out == tmp_path / "out.csv" and len(read_rows(out)) == 4


def test_the_gathered_table_carries_a_provenance_record(both):
    from taters.helpers.provenance import read

    _csv, table = both
    out = _gather(table, "db://lab/main/taters/recorded", text_cols=["text"],
                  id_cols=["pid", "session"])
    rec = read(out)
    assert rec and rec["defines_text"] and rec["table"] == "recorded"
    assert rec["output"]["rows"] == 4 and rec["output"]["location"] == str(out)
    # the same call again is handed back and keeps its record
    again = _gather(table, "db://lab/main/taters/recorded", text_cols=["text"],
                    id_cols=["pid", "session"])
    assert again == out and read(again)


def test_an_analyzer_reads_a_gathered_table_in_the_database(both):
    from taters.text.analyze_readability import analyze_readability

    _csv, table = both
    gathered = _gather(table, "db://lab/main/taters/texts", text_cols=["text"],
                       id_cols=["pid", "session"])
    out = analyze_readability(analysis_csv=str(gathered),
                              out_features_csv="db://lab/main/taters/readability")
    assert tables.exists(out)
    rows = list(tables.read_rows(out))
    assert len(rows) == 4 and rows[0]["text_id"] == "p1 | 1"
    assert "flesch_reading_ease" in rows[0]


@pytest.fixture
def joined(both, tmp_path):
    """The study table plus an outcomes table that fans out (two waves per
    participant) and a notes table whose keys repeat on both sides."""
    dbfile = tmp_path / "lab.sqlite"
    con = sqlite3.connect(dbfile)
    con.executescript("""
    CREATE TABLE outcomes(rid INTEGER, wave INTEGER, phq9 REAL);
    INSERT INTO outcomes VALUES (1, 1, 2.0), (1, 2, 3.0), (2, 1, 4.0);
    CREATE TABLE notes(cond TEXT, note TEXT);
    INSERT INTO notes VALUES ('A', 'first'), ('A', 'second'), ('B', 'only');
    """)
    con.commit()
    con.close()
    tables.close_sessions()
    return both


def test_a_fan_out_join_is_warned_about_in_counts_only(joined, tmp_path, capsys):
    spec = {"connection": "lab", "from": {"table": "study"},
            "join": [{"table": "outcomes", "on": [["rowid_", "rid"]]}]}
    out = csv_to_analysis_ready_csv(csv_path=spec, out_csv=tmp_path / "g.csv",
                                    text_cols=["text"], id_cols=["pid", "session", "wave"],
                                    verbose=True)
    said = capsys.readouterr().out
    assert "some rows in study match several rows in outcomes" in said
    assert "rows become" in said
    assert " 1," not in said and "(e.g." not in said, "the run log names counts, never keys"
    ids = [r["text_id"] for r in read_rows(out)]
    assert ids.count("p1 | 1 | 1") == 1 and ids.count("p1 | 1 | 2") == 1, "both copies measured"


def test_a_many_to_many_join_is_refused_at_run_time_unless_allowed(joined, tmp_path):
    spec = {"connection": "lab", "from": {"table": "study"},
            "join": [{"table": "notes", "on": [["cond", "cond"]]}]}
    with pytest.raises(ValueError, match="repeat on both sides.*allow_many_to_many"):
        _gather(spec, tmp_path / "g.csv", text_cols=["text"], id_cols=["pid", "session", "note"])
    spec["join"][0]["allow_many_to_many"] = True
    out = _gather(spec, tmp_path / "g2.csv", text_cols=["text"],
                  id_cols=["pid", "session", "note"])
    assert tables.exists(out)
