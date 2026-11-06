"""
Every text step, once, on a database: composed by the real composer from a
table in SQLite, run by the real runner, with every row-shaped output
landing back in the database. The steps that need a saved model, a trained
encoder or a GPU are left to their own suites; what this guards is that no
step still reaches for a file when it was handed a table.
"""
from __future__ import annotations

import random
import sqlite3

import pytest

from taters.helpers import db, tables
from taters.pipelines.run_pipeline import run_preset
from taters.ui import recipes as _recipes
from taters.ui.compose import compose

#: Steps whose inputs are a fitted or downloaded model, or whose cost is a
#: GPU: covered by tests/test_db_steps_*.py with fakes and tiny models.
NEEDS_A_MODEL = {
    "archetypes", "sentence_embeddings", "transformer_embeddings",
    "hf_classifier_apply", "text_predictor_apply", "word_vectors_apply",
    "topic_model_lda_apply", "topic_model_nmf_apply", "topic_model_mem_apply",
    "adapt_encoder", "pretrain_encoder", "finetune_text_predictor",
    "score_with_model",
}


def _text_steps():
    return sorted(r.id for r in _recipes.RECIPES
                  if r.text_input and "db" in r.sources and r.id not in NEEDS_A_MODEL)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TATERS_HOME", str(home))
    tables.close_sessions()
    db._SECRETS.clear()
    path = tmp_path / "lab.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE posts(id INTEGER PRIMARY KEY, cond TEXT, body TEXT)")
    rng = random.Random(11)
    words = ["potato", "gravy", "butter", "delightful", "fluffy", "kitchen", "dinner",
             "friend", "family", "home", "work", "time", "people", "think", "good"]
    for i in range(40):
        n = 12 + rng.randrange(12)
        text = " ".join(rng.choice(words) for _ in range(n)) + "."
        con.execute("INSERT INTO posts VALUES (?, ?, ?)", (i + 1, "AB"[i % 2], text))
    con.commit()
    con.close()
    db.save_profile(db.Profile(name="lab", dialect="sqlite", path=str(path)))
    yield path
    tables.close_sessions()


def test_every_text_step_is_offered_on_a_database():
    """The catalog no longer holds a step back from a database source."""
    held_back = [r.id for r in _recipes.RECIPES if r.text_input and "db" not in r.sources]
    assert held_back == []


@pytest.mark.parametrize("step", _text_steps())
def test_a_text_step_runs_from_a_table_into_a_table(lab, tmp_path, step, monkeypatch):
    recipe = _recipes.by_id(step)
    if step == "generate_with_llm":
        # the plumbing is under test, not a model: canned replies
        monkeypatch.setenv("TATERS_FAKE_LLM", "1")
    for extra in recipe.extras or ():
        probes = __import__("taters.ui.wizard", fromlist=["EXTRA_PROBES"]).EXTRA_PROBES.get(extra, ())
        for module in probes:
            pytest.importorskip(module)
    # the two steps that read a library asset get a tiny one of their own,
    # and the topic models a count the forty-text corpus can support
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "cats.dic").write_text("%\n1\tFood\n%\npotato\t1\ngravy\t1\n",
                                     encoding="utf-8")
    (assets / "norms.csv").write_text("DicTerm,Warmth\npotato,5.0\nbutter,3.0\n",
                                      encoding="utf-8")
    preset = compose([step], source="db",
                     source_spec={"connection": "lab", "from": {"table": "posts", "as": "p"}},
                     connection="lab", results_root="db://lab/main/taters/sweep/",
                     text_cols=["body"], id_cols=["id"], name="sweep",
                     var_values={"dictionaries_path": str(assets / "cats.dic"),
                                 "norms_path": str(assets / "norms.csv"),
                                 "lda_topics": 3, "nmf_factors": 3,
                                 "mem_components": 3, "mem_k_selection": "fixed"})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    assert not (work / "features").exists(), "no row-level table touched the disk"
    assert not (work / "gathered").exists()
    made = [t for t in tables.tables_under("db://lab/main/taters/sweep/")
            if t.logical.startswith("taters/sweep/features/")]
    assert made, f"{step} wrote no feature table into the database"
    assert all(tables.count(t) > 0 for t in made)


def test_a_prompt_fills_a_placeholder_from_a_database_column(lab, tmp_path, monkeypatch):
    """{cond} in the prompt is filled from the table's own column, row by
    row, the way a spreadsheet column fills it, and the column rides along
    into the answers table."""
    import json

    prompt = tmp_path / "Fit.txt"
    prompt.write_text("The writer was in condition {cond}. "
                      "Rate the fit.\n", encoding="utf-8")
    fake = tmp_path / "fake.json"
    fake.write_text(json.dumps({"default": "{}", "contains": {
        "condition A.": '{"fit": 2}', "condition B.": '{"fit": 6}'}}), encoding="utf-8")
    monkeypatch.setenv("TATERS_FAKE_LLM", str(fake))
    preset = compose(["generate_with_llm"], source="db",
                     source_spec={"connection": "lab", "from": {"table": "posts", "as": "p"}},
                     connection="lab", results_root="db://lab/main/taters/fill/",
                     text_cols=["body"], id_cols=["id"], name="fill",
                     overrides={"generate_with_llm": {"prompt_paths": [str(prompt)],
                                            "pass_through_cols": ["id", "cond"]}},
                     var_values={"llm_parse_json": True})
    work = tmp_path / "run"
    work.mkdir()
    manifest = run_preset(preset, workers=1, work_dir=work,
                          out_manifest=work / "run_manifest.json", verbose=False)
    assert not manifest["errors"], manifest["errors"]
    made = [t.logical for t in tables.tables_under("db://lab/main/taters/fill/")]
    assert not [m for m in made if "stage" in m], "the parse pass drops its stage"
    assert not (work / "features").exists(), "no row-level table touched the disk"
    (answers,) = [t for t in tables.tables_under("db://lab/main/taters/fill/")
                  if t.logical.endswith("features/llm")]
    rows = list(tables.read_rows(answers))
    assert len(rows) == 40
    assert {(str(r["cond"]), str(r["Fit__fit"])) for r in rows} == {("A", "2"), ("B", "6")}
