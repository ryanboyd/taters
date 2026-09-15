"""
Tests for "Train a model": the front-page task, the flow in `taters.ui.train`
and the wizard hooks it rests on (a preselected step, a text-only source
question).

The first question is the task's own; everything after it is the ordinary
wizard, so these tests check the seam -- that the checklist is not asked,
that the pipeline composed carries the training step and its companions,
that a missing extra grays the kind out with its install command -- and the
import path, which writes a model and offers the library.
"""
from __future__ import annotations

import json

import pytest

from taters.ui import wizard as wiz
from taters.ui.prompts import GoBack, ScriptedPrompter
from taters.ui.train import run_train
from wizard_helpers import EscapingPrompter, browse_to

CHECKLIST = "Each row here is a table of measures"


def _calls(preset):
    return [s["call"].rsplit(".", 1)[-1] for s in preset["steps"]]


def test_the_train_task_sits_after_the_extraction_verbs():
    from taters.ui.tasks import all_tasks

    ids = [t.id for t in all_tasks()]
    # the extraction verbs first, and only then training -- a run that exists
    # for the model it leaves behind rather than for the numbers it produces
    # now. (wrangling and analyzing a spreadsheet you already have share the
    # "data" row above them; neither extracts anything.)
    assert ids.index("train") == ids.index("extract_analyze") + 1
    assert ids.index("extract_analyze") == ids.index("extract") + 1
    assert ids.index("train") < ids.index("run_saved")
    assert "analyze_spreadsheet" not in ids, "it moved under the data row"


def test_training_word_vectors_runs_the_wizard_with_the_step_already_chosen(study_csv, tmp_path):
    """No checklist: what to train was the task's first question. The
    pipeline saved is the ordinary kind -- the training step, its clouds,
    the descriptives -- so it re-runs like any other."""
    pytest.importorskip("gensim")
    p = ScriptedPrompter([
        "word_vectors",
        "csv", *browse_to(study_csv), ["text"], True, ["pid"],
        "row",                      # we still get asked the level question
        False,                      # no settings changes
        "Vectors", "save",
    ])
    assert run_train(p, cwd=tmp_path) is None, "saved, not run"
    assert not any(CHECKLIST in r for r in p.reasons), "the checklist was skipped"
    import yaml

    saved = next(tmp_path.glob("*/*.yaml"))
    preset = yaml.safe_load(saved.read_text(encoding="utf-8"))
    calls = _calls(preset)
    assert "train_word_vectors" in calls
    assert "neighbor_wordclouds" in calls and "describe_features" in calls
    kinds = next(cs for q, cs in p.offered if q == "What kind of data do you have?")
    assert [c.value for c in kinds] == ["txt_dir", "csv"], "text only: nothing to transcribe"


def test_a_kind_this_python_cannot_have_is_grayed_with_the_python_not_a_pip_command(
        tmp_path, monkeypatch):
    """A pip command that installs nothing is not a fix; the Train menu says
    which Python would have it."""
    import importlib.util

    from taters.ui.tasks import gpu

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "gensim" else real(name, *a, **k))
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "gensim")
    p = EscapingPrompter(["__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    rows = {c.value: c for c in p.offered[0][1]}
    assert rows["word_vectors"].disabled.startswith("not available for Python")
    assert "pip install" not in rows["word_vectors"].disabled


def test_a_kind_whose_extra_is_missing_is_grayed_with_the_install_command(tmp_path, monkeypatch):
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    # Esc at the very first question takes us out to the menu, same as in the
    # wizard. the hub catches it (GoBack and Cancelled both)
    p = EscapingPrompter(["__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    rows = {c.value: c for c in p.offered[0][1]}
    assert rows["word_vectors"].disabled == 'needs pip install "taters[vectors]"'
    assert not rows["import_vectors"].disabled, "text formats need numpy alone"


def test_esc_at_the_level_question_returns_to_the_source_not_to_a_checklist(study_csv, tmp_path):
    pytest.importorskip("gensim")
    p = EscapingPrompter([
        "csv", *browse_to(study_csv), ["text"], True, ["pid"],
        "__esc__",                  # back out from the level; there's no checklist
        "csv", *browse_to(study_csv), ["text"], True, ["pid"],
        "row", False, "Again", "save",
    ])
    res = wiz.run_wizard(p, cwd=tmp_path, banner=False, analyses=False,
                         preselected=["word_vectors_train"], text_only=True)
    assert [q for _k, q in p.asked].count("What kind of data do you have?") == 2
    assert "train_word_vectors" in _calls(res.preset)


def test_importing_vectors_writes_the_model_and_offers_the_library(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    from taters.helpers import library as lib

    glove = tmp_path / "glove.6B.50d.txt"
    glove.write_text("cat 1 0\ndog 0.9 0.1\ncar 0 1\n", encoding="utf-8")
    # a dictionary sitting in the library, so the import flow offers it as a concept
    pets = tmp_path / "pets.dicx"
    pets.write_text("DicTerm,pets\ncat,X\ndog,X\n", encoding="utf-8")
    lib.import_into(lib.KINDS["dictionaries"], pets)
    p = ScriptedPrompter([
        "import_vectors", *browse_to(glove), "auto",
        "glove-tiny",               # the model's name
        str((lib.kind_dir(lib.KINDS["dictionaries"]) / "pets.dicx").resolve()),  # a concept dictionary
        True,                       # yes, add it to the library
        False,                      # don't rename it or its columns
    ])
    assert run_train(p, cwd=tmp_path) is True
    model = tmp_path / "models" / "glove-tiny.json"
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert doc["vocabulary"] == ["cat", "dog", "car"]
    assert [d["name"] for d in doc["apply"]["concept_dicts"]] == ["pets"]
    assert doc["apply"]["concept_dicts"][0]["categories"]["pets"] == [["cat", 1.0], ["dog", 1.0]]
    assert (tmp_path / "models" / "glove-tiny.npy").is_file()
    assert (tmp_path / "models" / "glove-tiny_report.md").is_file()
    assert [e.name for e in lib.entries(lib.KINDS["models"])] == ["glove-tiny.json"]
    assert (lib.kind_dir(lib.KINDS["models"]) / "glove-tiny.npy").is_file()
    assert any("Saved glove-tiny [word vectors]" in line for line in p.output)


def test_an_empty_dictionary_library_asks_nothing_about_concepts(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    glove = tmp_path / "g.txt"
    glove.write_text("cat 1 0\n", encoding="utf-8")
    p = ScriptedPrompter(["import_vectors", *browse_to(glove), "auto", "g", False])
    assert run_train(p, cwd=tmp_path) is True
    assert not any(q.startswith("Which content-coding dictionaries") for _k, q in p.asked)
    doc = json.loads((tmp_path / "models" / "g.json").read_text(encoding="utf-8"))
    assert doc["apply"]["concept_dicts"] == []


def test_an_unreadable_vectors_file_is_a_red_note_not_a_traceback(tmp_path):
    bad = tmp_path / "vectors.txt"
    bad.write_text("", encoding="utf-8")
    p = ScriptedPrompter(["import_vectors", *browse_to(bad), "glove", "x", ""])
    assert run_train(p, cwd=tmp_path) is False
    assert any("Not imported" in line and "no readable word vectors" in line
               for line in p.output)


def test_a_concept_the_vectors_do_not_know_is_refused_before_saving(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    from taters.helpers import library as lib

    glove = tmp_path / "g.txt"
    glove.write_text("cat 1 0\n", encoding="utf-8")
    zoo = tmp_path / "zoo.dicx"
    zoo.write_text("DicTerm,exotic\nzebra,X\n", encoding="utf-8")
    lib.import_into(lib.KINDS["dictionaries"], zoo)
    p = ScriptedPrompter(["import_vectors", *browse_to(glove), "auto", "g",
                          str((lib.kind_dir(lib.KINDS["dictionaries"]) / "zoo.dicx").resolve())])
    assert run_train(p, cwd=tmp_path) is False
    assert any("'exotic'" in line for line in p.output)


def test_fine_tuning_asks_which_columns_to_predict_and_offers_only_spreadsheets(study_csv, tmp_path):
    """The predictor needs outcome columns, which only a spreadsheet has,
    and which are its own question -- asked after the level, from the
    file's columns, split into measurements and categories by kind."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    p = ScriptedPrompter([
        "finetune_predictor",
        "distilroberta-base",           # the encoder, before anything else
        "csv", *browse_to(study_csv), ["text"], True, ["pid"],
        "row",
        ["openness", "condition"],      # the columns we want to predict
        ":done",                        # picking something opens up the options list
        "Predict", "save",
    ])
    assert run_train(p, cwd=tmp_path) is None
    questions = [q for _k, q in p.asked]
    assert questions.index("Which encoder?") < questions.index(
        "What kind of data do you have?"), "the encoder is the first thing asked"
    kinds = next(cs for q, cs in p.offered if q == "What kind of data do you have?")
    assert [c.value for c in kinds] == ["csv"]
    offered = next(cs for q, cs in p.offered if q.startswith("Which column(s) should the model"))
    assert [c.value for c in offered] == ["condition", "openness"], "not the text or the id"
    import yaml

    preset = yaml.safe_load(next(tmp_path.glob("*/*.yaml")).read_text(encoding="utf-8"))
    step = next(s for s in preset["steps"] if s["call"].endswith("finetune_text_predictor"))
    # the answers ride along as the pipeline's variables (the step reads them
    # through {{var:...}}). that's what lets a saved pipeline be re-run and
    # edited later
    assert step["with"]["outcome_cols"] == "{{var:predictor_outcomes}}"
    variables = preset["meta"]["variables"]
    assert variables["predictor_outcomes"]["default"] == ["openness", "condition"]
    assert variables["predictor_categorical"]["default"] == ["condition"], \
        "condition holds labels, so it is a category; openness holds numbers"
    # the encoder we picked is the pipeline's, and the model is named after it
    assert variables["encoder"]["default"] == "distilroberta-base"
    assert variables["predictor_name"]["default"] == "distilroberta-base-finetuned"
    assert step["with"]["name"] == "{{var:predictor_name}}"


def test_adapting_asks_for_the_encoder_first_and_names_the_result_after_it(study_csv, tmp_path):
    """
    "It wasn't even obvious that a model had already been selected": the
    encoder used to be one row of the options screen, defaulted silently.
    Now it is the question right after "what would you like to train", and
    the adapted encoder's default name says where it came from.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    p = ScriptedPrompter([
        "adapt_encoder",
        "distilroberta-base",
        "csv", *browse_to(study_csv), ["text"], True, ["pid"],
        "row",
        ":done", "Adapt", "save",
    ])
    assert run_train(p, cwd=tmp_path) is None
    questions = [q for _k, q in p.asked]
    assert questions.index("Which encoder?") < questions.index("What kind of data do you have?")
    import yaml

    preset = yaml.safe_load(next(tmp_path.glob("*/*.yaml")).read_text(encoding="utf-8"))
    variables = preset["meta"]["variables"]
    assert variables["encoder"]["default"] == "distilroberta-base"
    assert variables["encoder_name"]["default"] == "distilroberta-base-adapted"


def test_esc_at_the_encoder_question_returns_to_the_kind_question(tmp_path):
    """The screen before the encoder is the list of things to train, so Esc
    goes there -- not out of the task, and not on into the wizard with a
    default nobody chose."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    p = EscapingPrompter(["adapt_encoder", "__esc__", "__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    kinds = [q for _k, q in p.asked if q == "What would you like to train?"]
    assert len(kinds) == 2, "the kind question was asked again after Esc"


def test_the_name_built_from_an_encoder_is_its_short_form():
    from taters.ui.train import encoder_stem

    assert encoder_stem("distilroberta-base") == "distilroberta-base"
    assert encoder_stem("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2"
    assert encoder_stem("/home/me/library/encoders/forum_adapted.json") == "forum_adapted"
    assert encoder_stem("C:\\models\\my.encoder\\") == "my-encoder"
    assert encoder_stem("") == "encoder"


def test_the_encoder_setting_opens_the_picker_on_the_options_screen(tmp_path, monkeypatch):
    """Not a text box: the encoder setting of every transformer step lists
    what is on the machine, and the answer lands in the shared variable."""
    from taters.text import _transformer_common as tc
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    monkeypatch.setattr(tc, "cached_hub_encoders",
                        lambda cache_dir=None: [("sentence-transformers/all-MiniLM-L6-v2",
                                                 "downloaded")])
    for recipe_id, param in (("transformer_embeddings", "model_name_or_path"),
                             ("adapt_encoder", "base_model"),
                             ("finetune_text_predictor", "base_model")):
        assert _r.by_id(recipe_id).encoder_param == param
    steps = [_r.by_id("transformer_embeddings")]
    var_specs = compose(["transformer_embeddings"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter([
        "transformer_embeddings", "model_name_or_path",
        "sentence-transformers/all-MiniLM-L6-v2",
        ":done", ":done",
    ])
    _overrides, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False)
    assert var_values["encoder"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert any(q == "Which encoder?" for _k, q in p.asked)


def test_every_transformer_kind_is_grayed_without_torch(tmp_path, monkeypatch):
    """The check keyed on words in the target module name and missed the
    fine-tuning entry, whose module is called finetune_predictor."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torch" else real(name, *a, **k))
    p = EscapingPrompter(["__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    rows = {c.value: c for c in p.offered[0][1]}
    assert rows["adapt_encoder"].disabled.startswith("needs torch")
    assert rows["finetune_predictor"].disabled.startswith("needs torch")
    assert not rows["import_vectors"].disabled


def test_the_concept_dictionary_setting_opens_the_library_picker_empty(tmp_path, monkeypatch):
    """Word vectors take their concepts from the dictionaries library, and the
    recipe declares an *empty* default: the picker opens with nothing ticked
    (dictionary steps open full), and what is ticked lands as the step's
    concept_dicts."""
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    from taters.helpers import library as lib
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    kind = lib.KINDS["dictionaries"]
    for name in ("pets.dicx", "moods.dicx"):
        (tmp_path / name).write_text("DicTerm,c\ncat,X\n", encoding="utf-8")
        lib.import_into(kind, tmp_path / name)
    pets = str((lib.kind_dir(kind) / "pets.dicx").resolve())
    steps = [_r.by_id("word_vectors_train")]
    var_specs = compose(["word_vectors_train"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter(["word_vectors_train", "concept_dicts", pets, ":done", ":done"])
    overrides, _vars = wiz.ask_tuning(p, steps, var_specs, ask_gate=False)
    assert overrides["word_vectors_train"]["concept_dicts"] == [pets]
    picker = next(cs for q, cs in p.offered if q.startswith("Which content-coding dictionaries"))
    assert all(c.label.startswith("[ ]") for c in picker if c.value.endswith(".dicx")), \
        "nothing pre-ticked: concepts are optional"
    # ...whereas the dictionary analyzer's own picker still opens with everything
    # ticked
    steps = [_r.by_id("dictionaries")]
    var_specs = compose(["dictionaries"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    q = ScriptedPrompter(["dictionaries", "dict_paths", pets, ":done", ":done"])
    wiz.ask_tuning(q, steps, var_specs, ask_gate=False)
    picker = next(cs for qq, cs in q.offered if qq.startswith("Which content-coding dictionaries"))
    assert all(c.label.startswith("[x]") for c in picker if c.value.endswith(".dicx"))
