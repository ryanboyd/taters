"""
The wizard on a database: the source stage's own questions, the join
conversation inside it, the results question, and a pipeline composed and
run from the answers -- against a SQLite database standing in for the lab's.
"""
from __future__ import annotations

import random
import sqlite3

import pytest

from taters.helpers import db, tables
from taters.ui import wizard as wiz
from taters.ui.prompts import ScriptedPrompter
from wizard_helpers import clean_machine  # noqa: F401  (autouse: no extras missing)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    tables.close_sessions()
    db._SECRETS.clear()
    path = tmp_path / "lab.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE participants(pid TEXT PRIMARY KEY, condition TEXT, openness REAL, age INTEGER);
    CREATE TABLE posts(id INTEGER PRIMARY KEY, pid TEXT REFERENCES participants(pid),
                       body TEXT, created_at TEXT);
    CREATE TABLE outcomes(post_id INTEGER, wave INTEGER, phq9 REAL);
    """)
    rng = random.Random(4)
    for i in range(30):
        group = "ABC"[i % 3]
        con.execute("INSERT INTO participants VALUES (?, ?, ?, ?)",
                    (f"p{i}", group, round(rng.gauss(3, 1), 4), rng.randrange(18, 70)))
        words = 8 + 6 * "ABC".index(group) + rng.randrange(3)
        text = " ".join(rng.choice(["potato", "gravy", "butter", "delightful", "fluffy"])
                        for _ in range(words))
        con.execute("INSERT INTO posts VALUES (?, ?, ?, ?)", (i + 1, f"p{i}", text, "t"))
    for i in range(1, 6):
        con.execute("INSERT INTO outcomes VALUES (?, 1, 2.0)", (i,))
        con.execute("INSERT INTO outcomes VALUES (?, 2, 3.0)", (i,))
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(path)))
    yield path
    tables.close_sessions()


def _asked(p, kind=None):
    return [q for k, q in p.asked if kind is None or k == kind]


def _cols(table, *names):
    return [f"{table}.{n}" for n in names]


_POSTS_AND_PEOPLE = (_cols("posts", "id", "pid", "body")
                     + _cols("participants", "condition", "openness", "age"))


def test_a_database_source_is_offered_and_built_from_its_tables(lab, tmp_path):
    p = ScriptedPrompter([
        "db", "lab",
        ["posts", "participants"],        # which tables, under the database
        _POSTS_AND_PEOPLE,                # then their columns, on one screen
        ["readability"], "row",
        "lab",                            # where should the results go
        [],                               # no statistics
        False,                            # change any settings first? no
        "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    kinds = {c.value: c for c in p.offered_choices("What kind of data do you have?")}
    assert kinds["db"].label.startswith("Tables in a database")
    asked = _asked(p)
    assert asked.index("Which tables?") < asked.index("Which columns do you need?")
    hier = {c.value: c for c in p.offered_choices("Which tables?")}
    assert hier["\x00db:lab"].children == ("outcomes", "participants", "posts")
    assert hier["posts"].depth == 1 and "body" in hier["posts"].help
    assert "Which table has one row per piece of text?" not in asked, \
        "posts references participants, so posts is the row table"
    assert "Which column(s) identify each row?" not in asked, "the key is the id"
    assert "Which column(s) hold the text" not in asked, "body is the one long text"
    said = p.text_output
    assert "participants joined on posts.pid = participants.pid (the database declares it)." in said
    assert "Every row in posts with a pid matches a row in participants." in said
    assert "Each row is identified by posts's key, id." in said
    assert "The text is body from posts, the one long-text column you picked." in said
    screen = {c.value: c for c in p.offered_choices("Which columns do you need?")}
    # three lined-up columns -- name, table, type -- never a name with the
    # table glued on, since table names have underscores of their own
    assert screen["posts.body"].label.rstrip() == "body"
    assert screen["posts.body"].annotation.startswith("posts")
    assert screen["participants.openness"].annotation.split()[0] == "participants"
    assert screen["posts.body"].help == "long text"
    assert screen["posts.pid"].help.startswith("→ participants.pid")
    assert "Where should the results go?" in _asked(p, "select")
    preset = result.preset
    assert preset["vars"]["connection"] == "lab"
    spec = preset["vars"]["source"]
    assert spec["join"][0]["table"] == "participants"
    assert list(spec["columns"]) == ["id", "pid", "body", "condition", "openness", "age"]
    assert preset["vars"]["results_root"].startswith("db://lab/main/taters/")
    assert preset["meta"]["id"] == "study"


def test_a_fan_out_join_is_warned_about_and_confirmed(lab, tmp_path):
    p = ScriptedPrompter([
        "db", "lab",
        ["posts", "outcomes"],
        _cols("posts", "id", "body") + _cols("outcomes", "wave", "phq9"),
        False,                            # don't repeat the rows
        "columns",                        # pick the columns again instead
        ["posts", "participants"], _POSTS_AND_PEOPLE,
        ["readability"], "row", "", [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    said = p.text_output
    assert "outcomes joined on posts.id = outcomes.post_id (post_id names posts)." in said
    assert "Some rows in posts match several rows in outcomes" in said
    assert "measured once per match, so every statistic will count it more than once" in said
    assert "Repeat those rows and measure each copy anyway?" in _asked(p, "confirm")
    assert sum(q == "Which columns do you need?" for q in _asked(p)) == 2
    assert result.preset["vars"]["source"]["join"][0]["table"] == "participants"
    assert result.preset["vars"]["results_root"] == "", "results to the folder"


def test_the_key_is_kept_even_when_it_was_not_ticked(lab, tmp_path):
    p = ScriptedPrompter([
        "db", "lab", ["posts", "participants"],
        _cols("posts", "body") + _cols("participants", "age"),
        ["readability"], "row", "", [], False, "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    spec = result.preset["vars"]["source"]
    assert list(spec["columns"]) == ["id", "body", "age"]
    assert "Each row is identified by posts's key, id." in p.text_output


def test_statistics_on_a_database_source_vet_outcomes_from_the_server(lab, tmp_path):
    p = ScriptedPrompter([
        "db", "lab", ["posts", "participants"], _POSTS_AND_PEOPLE,
        ["readability"], "row",
        "lab",
        ["stats_correlations"], False, ["openness"], False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert "Reading openness" in p.output, "the column is read out loud"
    outcomes = {c.value: c for c in p.offered_choices("Which column(s) hold the outcomes?")}
    assert outcomes["openness"].label == "openness", "the name stays the name"
    assert outcomes["openness"].annotation.split()[0] == "participants", "the table beside it"
    assert result.preset["vars"]["stats_outcome_cols"] == ["openness"]
    assert result.preset["vars"]["results_root"].startswith("db://lab/main/")


def test_the_composed_database_pipeline_runs(lab, tmp_path):
    from taters.pipelines.run_pipeline import run_preset

    p = ScriptedPrompter([
        "db", "lab", ["posts", "participants"], _POSTS_AND_PEOPLE,
        ["readability"], "row", "lab",
        ["stats_group_differences"], False, "condition", False, "fdr_bh", False,
        ":done", "Study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    work = tmp_path / result.preset["meta"]["id"]
    work.mkdir(exist_ok=True)
    manifest = run_preset(result.preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    root = result.preset["vars"]["results_root"]
    assert tables.exists(root + "stats_results/analysis_table")
    assert not (work / "features").exists()
    assert (work / "stats_results" / "report.md").is_file()

    # the finish screen names the tables, and the folder for the report
    q = ScriptedPrompter(["quit"])
    with pytest.raises(Exception):
        wiz.finish_screen(q, ok=True, manifest=manifest, folder=work,
                          manifest_path=work / "run_manifest.json")
    said = q.text_output
    assert "Results are in lab" in said and "features__readability" in said
    assert "The report and figures are in" in said
