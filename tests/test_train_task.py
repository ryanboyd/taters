"""
Tests for "Train a model": the front-page task, the flow in `taters.ui.train`
and the wizard hooks it rests on (a preselected step, a text-only source
question).

The first two questions -- what to do, and with what -- are the task's own;
everything after them is the ordinary wizard, so these tests check the seam
-- that the checklist is not asked,
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
        "scratch", "word_vectors",
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

    from taters.text import _transformer_common as _common
    from taters.ui.tasks import gpu

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "gensim" else real(name, *a, **k))
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "gensim")
    # torch counts as present whatever this machine has: the verb's fate
    # below turns on it, and the point here is gensim's
    monkeypatch.setattr(_common, "torch_missing_reason", lambda: "")
    p = EscapingPrompter(["scratch", "__esc__", "__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    rows = {c.value: c for c in p.offered[1][1]}
    assert rows["word_vectors"].disabled.startswith("not available for Python")
    assert "pip install" not in rows["word_vectors"].disabled
    # the verb above it is not grayed: a transformer from scratch still can run
    verbs = {c.value: c for c in p.offered[0][1]}
    assert not verbs["scratch"].disabled


def test_a_kind_whose_extra_is_missing_is_grayed_with_the_install_command(tmp_path, monkeypatch):
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    # Esc at the very first question takes us out to the menu, same as in the
    # wizard. the hub catches it (GoBack and Cancelled both)
    p = EscapingPrompter(["scratch", "__esc__", "wrangle", "__esc__", "__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    kinds = next(cs for q, cs in p.offered if q == "With what?")
    rows = {c.value: c for c in kinds}
    assert rows["word_vectors"].disabled == 'needs pip install "taters[vectors]"'
    shelves = next(cs for q, cs in p.offered if q == "Which models?")
    assert not {c.value: c for c in shelves}["import_vectors"].disabled, \
        "text formats need numpy alone"


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
    # a dictionary in the library, which the import flow must *not* ask about:
    # bringing a model in is not applying it, and concept dictionaries are an
    # apply setting, set per model later under Settings
    pets = tmp_path / "pets.dicx"
    pets.write_text("DicTerm,pets\ncat,X\ndog,X\n", encoding="utf-8")
    lib.import_into(lib.KINDS["dictionaries"], pets)
    p = ScriptedPrompter([
        "wrangle", "import_vectors", *browse_to(glove), "auto",
        "glove-tiny",               # the model's name
        True,                       # yes, add it to the library
        False,                      # don't rename it or its columns
    ])
    assert run_train(p, cwd=tmp_path) is True
    assert not any(q.startswith("Which content-coding dictionaries") for _k, q in p.asked), \
        "the import flow asked about concept dictionaries"
    model = tmp_path / "models" / "glove-tiny.json"
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert doc["vocabulary"] == ["cat", "dog", "car"]
    assert doc["apply"]["concept_dicts"] == [], "none until somebody sets them per model"
    assert (tmp_path / "models" / "glove-tiny.npy").is_file()
    assert (tmp_path / "models" / "glove-tiny_report.md").is_file()
    assert [e.name for e in lib.entries(lib.KINDS["models"])] == ["glove-tiny.json"]
    assert (lib.kind_dir(lib.KINDS["models"]) / "glove-tiny.npy").is_file()
    assert any("Saved glove-tiny [word vectors]" in line for line in p.output)


def test_an_unreadable_vectors_file_is_a_red_note_not_a_traceback(tmp_path):
    bad = tmp_path / "vectors.txt"
    bad.write_text("", encoding="utf-8")
    p = ScriptedPrompter(["wrangle", "import_vectors", *browse_to(bad), "glove", "x", ""])
    assert run_train(p, cwd=tmp_path) is False
    assert any("Not imported" in line and "no readable word vectors" in line
               for line in p.output)


def test_fine_tuning_asks_which_columns_to_predict_and_offers_only_spreadsheets(study_csv, tmp_path):
    """The predictor needs outcome columns, which only a spreadsheet has,
    and which are its own question -- asked after the level, from the
    file's columns, split into measurements and categories by kind."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    p = ScriptedPrompter([
        "predict", "finetune_predictor",
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
        "What kind of data do you have?"), "the encoder comes before the wizard's questions"
    assert questions[:3] == ["What would you like to do?", "With what?", "Which encoder?"]
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
        "adapt", "adapt_encoder",
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


def test_esc_walks_back_one_screen_at_a_time(tmp_path):
    """The screen before the encoder is the kind, and the screen before that
    is the verb, so Esc goes there -- not out of the task, and not on into
    the wizard with a default nobody chose."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    p = EscapingPrompter(["adapt", "adapt_encoder", "__esc__",   # encoder -> kind
                          "__esc__",                              # kind -> verb
                          "__esc__"])                             # verb -> the hub
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    questions = [q for _k, q in p.asked]
    assert questions.count("With what?") == 2, "the kind was asked again after Esc"
    assert questions.count("What would you like to do?") == 2, "and the verb after that"


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

    from taters.ui import wizard as _wizard

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torch" else real(name, *a, **k))
    # word vectors count as installed here whatever this machine has -- the
    # point is torch's absence alone, and CI has no gensim either
    monkeypatch.setattr(_wizard, "missing_extras", lambda recipe: [])
    monkeypatch.setattr(_wizard, "unavailable_reason", lambda recipe: "")
    p = EscapingPrompter(["scratch", "__esc__", "__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    verbs = {c.value: c for c in p.offered[0][1]}
    # a verb with nothing left under it is grayed with the reason
    assert verbs["adapt"].disabled.startswith("needs torch")
    assert verbs["predict"].disabled.startswith("needs torch")
    # ...and one with a kind that still runs is not, though that kind is
    kinds = {c.value: c for c in p.offered[1][1]}
    assert not verbs["scratch"].disabled
    assert not kinds["word_vectors"].disabled
    assert kinds["scratch_transformer"].disabled.startswith("needs torch")
    assert not verbs["wrangle"].disabled


def test_a_verb_with_nothing_left_under_it_is_grayed_with_the_first_reason(tmp_path, monkeypatch):
    """
    Neither gensim nor torch -- the CI machine. Both rows under "from scratch"
    are grayed, so the verb is too, and it borrows its first row's reason
    rather than inventing a third. Import/export never needs either.
    """
    import importlib.util

    from taters.ui import wizard as _wizard

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "torch" else real(name, *a, **k))
    monkeypatch.setattr(_wizard, "missing_extras",
                        lambda recipe: ["vectors"] if recipe.id == "word_vectors_train" else [])
    monkeypatch.setattr(_wizard, "unavailable_reason", lambda recipe: "")
    p = EscapingPrompter(["__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    verbs = {c.value: c for c in p.offered[0][1]}
    assert verbs["scratch"].disabled == 'needs pip install "taters[vectors]"'
    assert verbs["adapt"].disabled.startswith("needs torch")
    assert not verbs["wrangle"].disabled


def test_the_first_screen_offers_the_four_verbs_and_the_second_is_always_asked(tmp_path):
    """
    Intention first, material second -- and the second question is asked even
    when it has one answer. A menu of one row still tells you exactly what
    you are about to do, and it is the scaffolding the next option slots
    into; skipping it would make "adapt" and "fine-tune" feel like different
    kinds of screen from "train from scratch".
    """
    p = EscapingPrompter(["adapt", "__esc__", "predict", "__esc__", "__esc__"])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    verbs = [c.value for c in p.offered[0][1]]
    assert verbs == ["scratch", "adapt", "predict", "wrangle"]
    kinds = [[c.value for c in cs] for q, cs in p.offered if q == "With what?"]
    assert kinds == [["adapt_encoder"], ["finetune_predictor"]], \
        "a one-row kind question was skipped"


def test_import_or_export_hands_off_to_the_library_screen_for_the_right_shelf(tmp_path, monkeypatch):
    """The fourth verb builds nothing new: the library screen already imports,
    exports, renames and deletes, per shelf. This just has to reach it with
    the shelf the person chose."""
    from taters.ui import library as ui_lib

    seen = []
    monkeypatch.setattr(ui_lib, "manage_library", lambda prompter, kind: seen.append(kind.id))
    assert run_train(ScriptedPrompter(["wrangle", ":encoders"]), cwd=tmp_path) is None
    assert run_train(ScriptedPrompter(["wrangle", ":models"]), cwd=tmp_path) is None
    assert seen == ["encoders", "models"]


def test_training_from_scratch_asks_the_size_and_warns_about_the_cost(study_csv, tmp_path):
    """
    The heavy-duty option is offered, not withheld -- someone with four GPUs
    may well want it -- but nobody should start it by accident. So after the
    source comes the size, then an estimate in hours for this many texts on
    this machine, then a confirm; and the size and a name built from it land
    in the pipeline's variables.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    p = ScriptedPrompter([
        "scratch", "scratch_transformer",
        "csv", *browse_to(study_csv), ["text"], True, ["pid"], "row",
        "small",                        # which size
        True,                           # go ahead
        ":done", "Scratch", "save",
    ])
    assert run_train(p, cwd=tmp_path) is None
    questions = [q for _k, q in p.asked]
    assert "Which encoder?" not in questions, "there is no encoder to start from"
    assert questions.index("Which size?") > questions.index("What kind of data do you have?")
    assert "Go ahead with training from scratch?" in questions
    assert any("random weights" in r for r in p.reasons)
    assert any("estimate" in r and ("hours" in r or "minutes" in r) for r in p.reasons), \
        "the cost was not said in wall-clock terms"
    import yaml

    preset = yaml.safe_load(next(tmp_path.glob("*/*.yaml")).read_text(encoding="utf-8"))
    assert "pretrain_encoder" in _calls(preset)
    variables = preset["meta"]["variables"]
    assert variables["pretrain_preset"]["default"] == "small"
    assert variables["encoder_name"]["default"] == "small-scratch"


def test_declining_the_warning_backs_out_rather_than_training(study_csv, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    p = EscapingPrompter([
        "scratch", "scratch_transformer",
        "csv", *browse_to(study_csv), ["text"], True, ["pid"], "row",
        "small", False,                 # think again
        "__esc__", "__esc__", "__esc__",   # source -> kind -> verb -> out
    ])
    with pytest.raises(GoBack):
        run_train(p, cwd=tmp_path)
    assert not list(tmp_path.glob("*/*.yaml")), "a pipeline was composed anyway"


def test_training_word_vectors_does_not_offer_concept_dictionaries(tmp_path, monkeypatch):
    """
    Training a model and applying it are two different things. The concept
    dictionaries a word-vector model scores texts against are an apply
    setting -- set per saved model under Settings, changeable later without
    retraining -- so the training step's options screen does not offer them,
    and the training run's own features carry the vectors alone.
    """
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    from taters.helpers import library as lib
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    (tmp_path / "pets.dicx").write_text("DicTerm,c\ncat,X\n", encoding="utf-8")
    lib.import_into(lib.KINDS["dictionaries"], tmp_path / "pets.dicx")
    recipe = _r.by_id("word_vectors_train")
    assert "concept_dicts" in recipe.hidden and "concept_dicts" not in recipe.library
    var_specs = compose(["word_vectors_train"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter(["word_vectors_train", ":done", ":done"])
    wiz.ask_tuning(p, [recipe], var_specs, ask_gate=False)
    rows = [c.value for c in next(cs for q, cs in p.offered if "change a setting" in q)]
    assert "concept_dicts" not in rows
    step = compose(["word_vectors_train"], name="x", source="csv",
                   input_path="t.csv")["steps"]
    assert not any("concept_dicts" in (st.get("with") or {}) for st in step)


def test_an_empty_curated_default_is_shown_as_none_not_as_everything(tmp_path, monkeypatch):
    """
    From a real session: a library row read "all 35 of your content-coding
    dictionaries" while its curated default applied none of them. An empty
    curated default is falsy, so it fell through to the wording for "no
    curated default at all", which means everything. The two are different
    settings and the row has to say which. No shipped recipe declares an
    empty default any more, so one is built here.
    """
    import dataclasses

    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    from taters.helpers import library as lib
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    kind = lib.KINDS["dictionaries"]
    for name in ("pets.dicx", "moods.dicx"):
        (tmp_path / name).write_text("DicTerm,c\ncat,X\n", encoding="utf-8")
        lib.import_into(kind, tmp_path / name)

    base = _r.by_id("dictionaries")
    var_specs = compose(["dictionaries"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    # as shipped: no curated default, so an untouched row really means everything
    q = ScriptedPrompter(["dictionaries", ":done", ":done"])
    wiz.ask_tuning(q, [base], var_specs, ask_gate=False)
    rows = {c.value: c for c in next(cs for qq, cs in q.offered if "change a setting" in qq)}
    assert "all 2 of your" in rows["dict_paths"].label

    # the same step with an *empty* curated default must say none, not all
    empty = dataclasses.replace(base, library_defaults={**base.library_defaults,
                                                        "dict_paths": ()})
    p = ScriptedPrompter(["dictionaries", ":done", ":done"])
    wiz.ask_tuning(p, [empty], var_specs, ask_gate=False)
    rows = {c.value: c for c in next(cs for qq, cs in p.offered if "change a setting" in qq)}
    assert "none (optional)" in rows["dict_paths"].label, rows["dict_paths"].label
    assert "of your" not in rows["dict_paths"].label
