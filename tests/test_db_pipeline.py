"""
A whole pipeline on a database: composed by the real composer, run by the
real runner, with the text in one table and the outcomes in another, and
every table the run makes landing back in the database -- or, when asked,
in the run folder. SQLite stands in for the lab's server.
"""
from __future__ import annotations

import random
import sqlite3

import pytest

from csvhelpers import _read
from taters.helpers import db, tables
from taters.pipelines.run_pipeline import run_preset
from taters.ui.compose import compose


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
    CREATE TABLE participants(pid TEXT PRIMARY KEY, condition TEXT, openness REAL);
    CREATE TABLE posts(id INTEGER PRIMARY KEY, pid TEXT REFERENCES participants(pid),
                       body TEXT);
    """)
    rng = random.Random(9)
    for i in range(24):
        group = "ABC"[i % 3]
        con.execute("INSERT INTO participants VALUES (?, ?, ?)",
                    (f"p{i}", group, round(rng.gauss(3, 1), 4)))
        words = 8 + 6 * "ABC".index(group) + rng.randrange(3)
        text = " ".join(rng.choice(["potato", "gravy", "butter", "delightful", "fluffy"])
                        for _ in range(words))
        con.execute("INSERT INTO posts VALUES (?, ?, ?)", (i + 1, f"p{i}", text))
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(path)))
    yield path
    tables.close_sessions()


SOURCE = {"connection": "lab",
          "from": {"table": "posts", "as": "p"},
          "join": [{"table": "participants", "as": "u", "how": "left",
                    "on": [["p.pid", "u.pid"]]}],
          "order_by": ["p.id"]}


def _preset(results_root: str, selected=None):
    return compose(
        selected or ["readability", "stats_group_differences", "stats_correlations"],
        source="db", source_spec=SOURCE, connection="lab", results_root=results_root,
        text_cols=["body"], id_cols=["id"], name="study stats",
        var_values={"stats_group_col": "condition",
                    "stats_outcome_cols": ["openness"],
                    "stats_meta_carry": ["condition", "openness"]})


def test_a_database_source_composes_to_a_preset_that_reads_the_spec(lab):
    preset = _preset("db://lab/taters/study_stats/")
    assert preset["vars"]["connection"] == "lab"
    assert preset["vars"]["source"]["from"]["table"] == "posts"
    assert preset["vars"]["results_root"] == "db://lab/taters/study_stats/"
    assert preset["vars"]["features_dir"] == "db://lab/taters/study_stats/features"
    gathers = [s for s in preset["steps"]
               if s["call"] == "potato.helpers.csv_to_analysis_ready_csv"]
    assert gathers and all(s["with"]["csv_path"] == "{{var:source}}" for s in gathers)
    assert all("delimiter" not in s["with"] for s in gathers)


def test_database_in_database_out(lab, tmp_path):
    preset = _preset("db://lab/taters/study_stats/")
    work = tmp_path / "run"
    work.mkdir()

    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]

    assert not (work / "features").exists() and not (work / "gathered").exists(), \
        "no row-level table touched the disk"
    table = tables.parse_ref("db://lab/taters/study_stats/stats_results/analysis_table")
    assert tables.exists(table)
    rows = list(tables.read_rows(table))
    assert len(rows) == 24
    assert {"text_id", "condition", "openness"} <= set(rows[0])
    assert any("flesch" in c or "reading" in c for c in rows[0]), "readability joined in"
    assert tables.exists("db://lab/taters/study_stats/features/readability")
    assert tables.exists("db://lab/taters/study_stats/gathered/texts")
    assert tables.exists("db://lab/taters/study_stats/gathered/metadata")

    stats = work / "stats_results"
    diffs = {r["feature"]: r for r in _read(stats / "group_differences.csv")}
    assert diffs and all(r["method"] == "anova" for r in diffs.values())
    assert _read(stats / "correlations_pearson.csv")
    report = (stats / "report.md").read_text(encoding="utf-8")
    assert "## Group differences" in report and "## Correlations" in report
    assert not (stats / "analysis_table.csv").exists(), "the row-level table is in the database"
    from taters.helpers.provenance import read
    assert read("db://lab/taters/study_stats/features/readability"), "recorded, and verified"


def test_database_in_files_out(lab, tmp_path):
    """Anything in, anything out: the same source, results in the folder."""
    preset = _preset("")
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    table = _read(work / "stats_results" / "analysis_table.csv")
    assert len(table) == 24 and {"condition", "openness"} <= set(table[0])
    assert (work / "features" / "readability.csv").is_file()
    assert (work / "gathered" / "texts.csv").is_file()


def test_a_rerun_hands_back_the_tables_it_made(lab, tmp_path):
    preset = _preset("db://lab/taters/again/")
    work = tmp_path / "run"
    work.mkdir()
    run_preset(preset, workers=1, work_dir=work, out_manifest=work / "m.json", verbose=False)
    s = db.session_for("lab")
    before = db.registered(s, db.sanitize_table_name("taters/again/features/readability"))
    s.close()
    manifest = run_preset(preset, workers=1, work_dir=work, out_manifest=work / "m.json",
                          verbose=False)
    assert not manifest["errors"]
    s = db.session_for("lab")
    after = db.registered(s, db.sanitize_table_name("taters/again/features/readability"))
    s.close()
    assert after.written_at == before.written_at, "not rewritten"


def test_spreadsheet_in_database_out(lab, tmp_path):
    """Anything in, anything out, the other way: a one-off CSV lands in the
    lab database with its features and analysis table beside it."""
    from csvhelpers import write_rows

    rows = [["id", "condition", "openness", "body"]]
    rng = random.Random(3)
    for i in range(24):
        group = "ABC"[i % 3]
        words = 8 + 6 * "ABC".index(group) + rng.randrange(3)
        rows.append([str(i + 1), group, f"{rng.gauss(3, 1):.4f}",
                     " ".join(rng.choice(["potato", "gravy", "butter"]) for _ in range(words))])
    sheet = write_rows(tmp_path / "study.csv", rows[0], rows[1:])
    preset = compose(
        ["readability", "stats_group_differences"], source="csv", input_path=str(sheet),
        results_root="db://lab/taters/sheet_run/", text_cols=["body"], id_cols=["id"],
        name="sheet run",
        var_values={"stats_group_col": "condition", "stats_meta_carry": ["condition"]})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    assert not (work / "features").exists() and not (work / "gathered").exists()
    assert tables.exists("db://lab/taters/sheet_run/gathered/texts")
    assert tables.exists("db://lab/taters/sheet_run/features/readability")
    table = list(tables.read_rows("db://lab/taters/sheet_run/stats_results/analysis_table"))
    assert len(table) == 24 and "condition" in table[0]
    assert (work / "stats_results" / "report.md").is_file()


def test_folder_of_documents_in_database_out(lab, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(6):
        (docs / f"d{i}.txt").write_text("potato gravy " * (3 + i), encoding="utf-8")
    preset = compose(["readability"], source="txt_dir", input_path=str(docs),
                     results_root="db://lab/taters/docs_run/", name="docs run")
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    gathered = list(tables.read_rows("db://lab/taters/docs_run/gathered/texts"))
    assert sorted(r["text_id"] for r in gathered) == [f"d{i}" for i in range(6)]
    assert tables.exists("db://lab/taters/docs_run/features/readability")


def test_deleting_a_pipeline_offers_to_drop_its_tables(lab, tmp_path):
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import TaskContext, manage

    preset = _preset("db://lab/taters/study_stats/", selected=["readability"])
    folder = tmp_path / "study_stats"
    folder.mkdir()
    run_preset(preset, workers=1, work_dir=folder,
               out_manifest=folder / "run_manifest.json", verbose=False)
    assert tables.exists("db://lab/taters/study_stats/features/readability")
    import yaml
    path = folder / "study_stats.yaml"
    path.write_text(yaml.safe_dump(preset, sort_keys=False), encoding="utf-8")

    # keep them: said, and left alone
    p = ScriptedPrompter([str(path), "file", preset["meta"]["title"], False])
    manage._delete(TaskContext(prompter=p, cwd=tmp_path), [path])
    assert "table(s) in lab are still there" in p.text_output
    assert "Remove them too?" in [q for _k, q in p.asked]
    assert tables.exists("db://lab/taters/study_stats/features/readability")

    # and again, dropping them
    path.write_text(yaml.safe_dump(preset, sort_keys=False), encoding="utf-8")
    p = ScriptedPrompter([str(path), "file", preset["meta"]["title"], True])
    manage._delete(TaskContext(prompter=p, cwd=tmp_path), [path])
    assert "Removed" in p.text_output
    assert not tables.exists("db://lab/taters/study_stats/features/readability")
    assert not tables.exists("db://lab/taters/study_stats/gathered/texts")
