"""
One description of a saved model, whatever kind it is.

The layer these tests guard exists because every screen that offers models
used to sniff JSON keys for itself. What is checked here is not the sniffing
but the two promises built on it: that a listing can tell a ridge from a
topic model *without* opening either one's loader, and that a name the user
chose reaches the column headings in their results.
"""

import json
from pathlib import Path

import pytest

from taters.helpers.model_spec import (LIBRARY_KIND, UnknownModel, describe,
                                       describe_all, output_label,
                                       rename_model)


def _ridge(path, outcomes=("age",), predictors=("f1", "f2")):
    path.write_text(json.dumps({
        "kind": "taters-ridge-model", "format": 2,
        "predictors": list(predictors),
        "outcomes": {o: {"kept": [0], "mu": [0.0], "sigma": [1.0],
                         "coef": [1.0], "intercept": 0.0}
                     for o in outcomes},
    }), encoding="utf-8")
    return path


def _classifier(path, classes=("no", "yes")):
    path.write_text(json.dumps({
        "kind": "taters-classifier-model", "format": 2,
        "predictors": ["f1"],
        "outcomes": {"cond": {"kept": [0], "mu": [0.0], "sigma": [1.0],
                              "classes": list(classes),
                              "models": [{"intercept": 0.0, "coef": [1.0]},
                                         None]}},
    }), encoding="utf-8")
    return path


def _mem(path, k=3):
    path.write_text(json.dumps({
        "kind": "taters-mem-model", "format": 1,
        "model": {"themes": [f"Theme_{i + 1}" for i in range(k)]},
    }), encoding="utf-8")
    return path


def test_every_kind_of_model_says_what_it_is_and_what_it_needs(tmp_path):
    """The listing has to distinguish them: every model file is a `.json` of
    no informative size, so a menu of file stems is a menu of one repeated
    word. Without the type and the needs, a user picks a topic model for a
    step that wanted a ridge, and the type label is also the only thing
    telling them which of two `ridge__all.json` files is which."""
    ridge = describe(_ridge(tmp_path / "ridge__all.json"))
    clf = describe(_classifier(tmp_path / "classifier__all.json"))
    mem = describe(_mem(tmp_path / "themes_model.json"))

    assert (ridge.type_id, ridge.needs) == ("ridge", "features")
    assert (clf.type_id, clf.needs) == ("classifier", "features")
    assert (mem.type_id, mem.needs) == ("mem", "text")
    assert ridge.display() == "ridge__all [ridge]"
    assert mem.display() == "themes_model [MEM topic model]"
    # what each one actually adds to the results, which isn't the same thing as
    # its outputs: one classifier outcome turns into four columns
    assert ridge.columns == ("pred_age",)
    assert clf.columns == ("pred_cond", "prob_cond", "p_cond_no",
                           "p_cond_yes")
    assert mem.n_outputs == 3 and mem.bulk_outputs
    assert ridge.inputs == ("f1", "f2") and mem.inputs == ()
    assert {ridge.library_kind, clf.library_kind} == {LIBRARY_KIND}


def test_a_file_that_is_not_a_model_is_refused_by_what_it_claims_to_be(
        tmp_path):
    """`describe` is what a listing calls, so its refusals are what a user
    reads when a file will not import. Naming the kind the file claims to be
    is the difference between "that is a PCA model, not a scorer" and a
    blank row."""
    other = tmp_path / "pca.json"
    other.write_text('{"kind": "taters-pca-model", "format": 1}',
                     encoding="utf-8")
    with pytest.raises(UnknownModel) as e:
        describe(other)
    assert "taters-pca-model" in str(e.value)
    assert e.value.kind == "taters-pca-model"

    plain = tmp_path / "settings.json"
    plain.write_text('{"theme": "dark"}', encoding="utf-8")
    with pytest.raises(UnknownModel, match="not a Taters model file at all"):
        describe(plain)

    junk = tmp_path / "junk.json"
    junk.write_text("not json at all", encoding="utf-8")
    with pytest.raises(UnknownModel, match="not readable"):
        describe(junk)


def test_one_unreadable_file_does_not_empty_the_whole_listing(tmp_path):
    """`describe_all` backs the menus. A folder holding one damaged file must
    still list the good ones -- a listing that raised would take the screen
    down, and the user would have no way to reach the delete action that
    would fix it."""
    good = _ridge(tmp_path / "good.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    found = describe_all([good, bad, tmp_path / "absent.json"])
    assert [i.name for i in found] == ["good"]


def test_a_renamed_model_writes_the_columns_the_user_asked_for(tmp_path):
    """The whole reason for naming: a ridge that predicts age from a blog
    corpus arrives as `ridge__all` writing `pred_age`, and scored against a
    dataset that already has an `age` column the results are ambiguous in a
    way no error reports. The name has to reach the column heading."""
    path = _ridge(tmp_path / "ridge__all.json")
    info = rename_model(path, name="age_blogs", outputs=["age_blogs"])
    assert info.name == "age_blogs"
    assert info.columns == ("pred_age_blogs",)
    assert info.display() == "age_blogs [ridge]"
    # and the scoring functions all read it through this one accessor, so a
    # renamed model gives us renamed columns no matter how it gets called
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert output_label(doc, "age") == "age_blogs"
    assert output_label(doc, "never_fitted") == "never_fitted"


def test_renaming_twice_renames_rather_than_accumulating(tmp_path):
    """The rename is keyed on the *fitted* outcome, not on the previous
    label. Keyed on the label, a second rename would write a second entry
    and leave the first one deciding the column name -- so the user's latest
    answer would be the one that did nothing."""
    path = _ridge(tmp_path / "r.json")
    rename_model(path, outputs=["first"])
    info = rename_model(path, outputs=["second"])
    assert info.columns == ("pred_second",)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["output_names"] == {"age": "second"}


def test_a_topic_model_is_renamed_by_prefix_and_a_ridge_is_not(tmp_path):
    """A hundred themes are not named one at a time, and one prefix is what
    distinguishes one topic model's columns from another's. The reverse is
    refused too: a prefix on a two-outcome ridge would name its columns
    `x_1` and `x_2`, throwing away which is which."""
    mem = _mem(tmp_path / "m.json", k=3)
    info = rename_model(mem, name="fb_topics", prefix="fb_topics")
    assert info.columns == ("fb_topics_1", "fb_topics_2", "fb_topics_3")
    # re-importing should default to the prefix it already has, not quietly
    # revert the naming to `Theme`
    assert describe(mem).outputs[0].rsplit("_", 1)[0] == "fb_topics"

    with pytest.raises(ValueError, match="name them individually"):
        rename_model(_ridge(tmp_path / "r.json"), prefix="x")
    with pytest.raises(ValueError, match="not both"):
        rename_model(mem, outputs=["a", "b", "c"], prefix="x")


def test_a_name_that_would_break_the_results_file_is_refused(tmp_path):
    """These become column headings. A comma or a newline would split the
    row when the results are read back, and the researcher would find it in
    their analysis rather than here -- so it is refused while they are
    holding the file."""
    path = _ridge(tmp_path / "r.json")
    for bad in ("a,b", "a\tb", 'a"b', "a\nb", "   "):
        with pytest.raises(ValueError):
            rename_model(path, outputs=[bad])
    with pytest.raises(ValueError, match="model name cannot be blank"):
        rename_model(path, name="  ")
    # refused, and nothing written. the file still says what it said before
    assert describe(path).columns == ("pred_age",)


def test_two_outputs_cannot_be_given_the_same_name(tmp_path):
    """One column would overwrite the other, and the file would look fine --
    the shape is right, one of the two predictions is simply gone."""
    path = _ridge(tmp_path / "r.json", outcomes=("age", "openness"))
    with pytest.raises(ValueError, match="cannot share a name"):
        rename_model(path, outputs=["same", "same"])
    assert describe(path).columns == ("pred_age", "pred_openness")


def test_a_wrong_number_of_names_is_refused_rather_than_zipped(tmp_path):
    """Zipped against a shorter list, the extra outcomes would keep their old
    names and the user would think they had renamed everything."""
    path = _ridge(tmp_path / "r.json", outcomes=("age", "openness"))
    with pytest.raises(ValueError, match="2 outcome"):
        rename_model(path, outputs=["only_one"])


def test_the_library_row_for_a_model_says_which_kind_it_is(tmp_path):
    """Every model file is a `.json` of no informative size, so a listing of
    file stems and suffixes is a menu of one repeated word -- and picking
    wrong means a step that wanted a ridge being handed a topic model. The
    row carries the kind and what the model will add."""
    from taters.helpers.library import KINDS

    row = KINDS["models"].describe_entry
    label, note = row(_ridge(tmp_path / "ridge__all.json"))
    assert label == "ridge__all [ridge]"
    assert note.startswith("adds age")
    # this fixture doesn't record how its features were measured, so the row
    # needs to say so right here, while you can still pick a different model,
    # and not an hour into a run
    assert "cannot reproduce its features" in note

    label, note = row(_mem(tmp_path / "m.json", k=40))
    assert label == "m [MEM topic model]"
    # forty theme names wouldn't fit and wouldn't help anybody
    assert note == "adds 40 columns, Theme_1…"

    # a damaged file should still show up by name instead of taking the whole
    # screen down
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert row(bad) == ("bad", ".json · not a readable model")


def test_a_model_of_any_kind_passes_the_one_import_gate(tmp_path):
    """The three kinds share one library folder, because "score this with a
    model I already have" is one thing to want. So one gate admits all of
    them and refuses everything else, delegating to whichever loader the
    file's own declaration points at."""
    from taters.helpers.library import KINDS, asset_problem

    kind = KINDS["models"]
    assert asset_problem(_ridge(tmp_path / "r.json"), kind) == ""
    assert asset_problem(_classifier(tmp_path / "c.json"), kind) == ""

    other = tmp_path / "pca.json"
    other.write_text('{"kind": "taters-pca-model", "format": 1}',
                     encoding="utf-8")
    assert "taters-pca-model" in asset_problem(other, kind)

    # structurally broken. the real loader catches this, not the cheap
    # description, so the gate is exactly as strict as run time is
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({
        "kind": "taters-ridge-model", "format": 2, "predictors": ["f1"],
        "outcomes": {"age": {"kept": [0], "mu": [0.0], "sigma": [1.0],
                             "coef": [1.0, 2.0], "intercept": 0.0}},
    }), encoding="utf-8")
    assert "mismatched sizes" in asset_problem(broken, kind)


# ---------------------------------------------------------------------------
# the contract for adding a kind of model
# ---------------------------------------------------------------------------

def test_every_registered_model_kind_declares_everything_it_needs():
    """
    The registry is what makes a new kind of model — a fine-tuned
    transformer, a boosted tree, whatever comes next — an addition rather
    than a surgery. Everything downstream reads it: the library folder, the
    naming flow at import, the wizard row, the provenance gate, and the
    scoring dispatch.

    So a half-added kind has to fail here rather than in someone's results.
    The dispatch in particular used to be `if classifier ... else ridge`,
    which would have scored an unregistered feature-reading model *as a
    ridge*, reading coefficients that were not in the file.
    """
    from taters.helpers.model_spec import BY_ID, MODEL_TYPES, scorer

    assert MODEL_TYPES, "no model kinds registered"
    for tag, spec in MODEL_TYPES.items():
        assert tag.startswith("taters-") and tag.endswith("-model"), tag
        assert spec.kind_tag == tag, f"{spec.id}: kind_tag disagrees with its key"
        assert spec.needs in ("features", "text"), \
            f"{spec.id}: needs={spec.needs!r} is neither features nor text"
        assert callable(spec.read), f"{spec.id}: no reader"
        # it has to resolve, and to something callable. otherwise a typo in the
        # dotted path would only turn up when somebody actually tried to score
        assert callable(scorer(spec.id)), f"{spec.id}: scorer does not resolve"
    assert set(BY_ID) == {s.id for s in MODEL_TYPES.values()}


def test_an_unregistered_kind_refuses_rather_than_guessing():
    """The whole value of routing through the registry: the failure mode of
    forgetting an entry is a named refusal, not a silent mis-dispatch."""
    from taters.helpers.model_spec import scorer

    with pytest.raises(ValueError) as e:
        scorer("some-future-transformer")
    assert "no scoring function is registered" in str(e.value)
    assert "MODEL_TYPES" in str(e.value)


def test_every_model_file_the_toolkit_writes_has_a_registered_kind():
    """
    The registry is only useful if it is complete, and nothing until now
    noticed a kind that gets *written* but never registered.

    That is exactly how LDA and NMF shipped unusable: both fits saved a
    perfectly good model file, and the import gate refused it by name --
    "not a model this build can score" -- so the apply steps could never be
    reached. The failure was at the seam, so neither side's tests saw it.

    So: every ``taters-...-model`` tag spelled anywhere in the source has to
    be a kind the registry knows. The scan is deliberately blunt rather than
    looking for ``"kind":`` -- LDA and NMF write ``"kind": MODEL_KIND``, so
    a scan for the literal spelling would have missed the two modules that
    prompted the test.

    PCA is the one exemption, and deliberately so: in this application a PCA
    is never a standalone instrument, it is a component of one. MEM's themes
    *are* its mu/sigma/projection; a ridge or classifier fitted on components
    carries its reduction inside its own model file and replays it when it
    scores. `fit_pca_csv` can save a freestanding one, but that is a
    Python-API tool with no wizard presence, and the registry is the list of
    instruments a saved file can be scored with. Registering it stays a live
    option if it ever gets a recipe of its own.
    """
    import re

    from taters.helpers.model_spec import MODEL_TYPES

    src = Path(__file__).resolve().parent.parent / "src" / "taters"
    written = {}
    for py in sorted(src.rglob("*.py")):
        if py.name == "model_spec.py":
            continue      # the registry itself, where naming a tag proves nothing
        for tag in re.findall(r'"(taters-[a-z0-9-]+-model)"',
                              py.read_text(encoding="utf-8")):
            written.setdefault(tag, []).append(py.name)

    assert written, "no model files are written anywhere -- the scan is broken"
    unregistered = {tag: mods for tag, mods in written.items()
                    if tag not in MODEL_TYPES and tag != "taters-pca-model"}
    assert not unregistered, (
        "these kinds are written but not in MODEL_TYPES, so the import gate "
        f"refuses them: {unregistered}")
    for tag in ("taters-mem-model", "taters-lda-model", "taters-nmf-model"):
        assert tag in written, f"{tag} is registered but nothing writes it"


def test_the_gate_applies_to_feature_models_and_not_to_text_models():
    """
    `needs` is the single word that decides whether a model's features have
    to be settings-checked, and it is read from the registry — so a new kind
    inherits the right behavior by declaring one string.

    A text model re-derives its own instrument from its own file (the MEM
    pattern), so there is nothing to compare and asking would be a permanent
    false refusal.
    """
    from taters.helpers.model_spec import MODEL_TYPES

    by_needs = {}
    for spec in MODEL_TYPES.values():
        by_needs.setdefault(spec.needs, []).append(spec.id)
    assert sorted(by_needs["features"]) == ["classifier", "ridge"]
    assert by_needs["text"] == ["mem", "lda", "nmf", "word_vectors",
                                "text_predictor", "hf_classifier"]


def test_the_description_carries_the_control_recipe(tmp_path):
    """The scorer used to re-open the model file for its `controls` -- a
    second read of a file `describe` had just parsed -- so the description
    now carries them, and the scorer reads nothing after describing."""
    import json

    from taters.helpers.model_spec import describe
    from taters.score_model import _control_spec

    doc = {"kind": "taters-ridge-model", "format": 2, "name": "age",
           "predictors": ["a", "b"], "outcomes": {"age": {"kept": [0, 1],
           "mu": [0, 0], "sigma": [1, 1], "coef": [1, 1], "intercept": 0}},
           "controls": [{"name": "gender", "levels": ["f", "m"]}]}
    path = tmp_path / "ridge__all.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    info = describe(path)
    assert list(info.controls) == doc["controls"]
    path.unlink()                       # so nothing can sneak a read of the file
    assert _control_spec(info) == doc["controls"]


# ---------------------------------------------------------------------------
# class labels and per-model apply settings
# ---------------------------------------------------------------------------

def test_a_class_can_be_relabeled_and_every_column_follows(tmp_path):
    """A model fitted on a 0/1 column predicts "0" and "1" -- correct, and
    unreadable a month later. The relabel lives in the file, keyed by the
    class as fitted, and describe() lists the columns as they will be
    written."""
    from taters.helpers.model_spec import describe, edit_model

    path = _classifier(tmp_path / "c.json", classes=("0", "1"))
    info = edit_model(path, class_names={"cond": {"0": "control", "1": "patient"}})
    assert info.classes == {"cond": ("control", "patient")}
    assert info.class_names == {"cond": {"0": "control", "1": "patient"}}
    assert set(info.columns) >= {"p_cond_control", "p_cond_patient"}
    assert "p_cond_0" not in info.columns

    # these merge one class at a time, and a class you name yourself loses its
    # relabel
    info = edit_model(path, class_names={"cond": {"1": "case"}})
    assert info.classes == {"cond": ("control", "case")}
    info = edit_model(path, class_names={"cond": {"0": "0", "1": ""}})
    assert info.classes == {"cond": ("0", "1")} and info.class_names == {}
    assert "class_names" not in json.loads(path.read_text(encoding="utf-8"))
    assert describe(path).classes == {"cond": ("0", "1")}


def test_a_relabel_that_could_not_be_told_apart_is_refused(tmp_path):
    from taters.helpers.model_spec import edit_model

    path = _classifier(tmp_path / "c.json", classes=("0", "1"))
    with pytest.raises(ValueError, match="cannot share the label"):
        edit_model(path, class_names={"cond": {"0": "same", "1": "same"}})
    with pytest.raises(ValueError, match="no class 'yes'"):
        edit_model(path, class_names={"cond": {"yes": "x"}})
    with pytest.raises(ValueError, match="no categorical outcome 'age'"):
        edit_model(path, class_names={"age": {"0": "x"}})
    with pytest.raises(ValueError, match="predicts no categories"):
        edit_model(_ridge(tmp_path / "r.json"), class_names={"age": {"0": "x"}})
    with pytest.raises(ValueError, match="cannot contain"):
        edit_model(path, class_names={"cond": {"0": "a,b"}})
    # none of the above should have touched the file
    assert "class_names" not in json.loads(path.read_text(encoding="utf-8"))


def test_apply_settings_come_from_the_registry_under_the_models_own(tmp_path, monkeypatch):
    """A kind registers the settings that govern how it is applied; a model
    file carries its own values in an `apply` block; an applier reads the
    merge. Typed-in values are coerced, choices checked, unknown keys
    refused by name -- and a damaged value in the file falls back to the
    default rather than blocking scoring."""
    from taters.helpers import model_spec as ms

    fake = ms.ModelType(
        id="fake", label="fake vectors", kind_tag="taters-fake-model",
        needs="text", read=lambda doc: (["v"], ["v_1"], []),
        apply_settings={
            "weighting": ms.ApplySetting("tokens", "How words are averaged.",
                                         choices=("tokens", "types")),
            "batch_size": ms.ApplySetting(32, "Texts per batch.", kind="int"),
            "normalize": ms.ApplySetting(False, "Unit length?", kind="bool"),
        })
    monkeypatch.setitem(ms.MODEL_TYPES, fake.kind_tag, fake)
    monkeypatch.setitem(ms.BY_ID, "fake", fake)
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"kind": fake.kind_tag, "format": 1}), encoding="utf-8")

    assert ms.describe(path).apply == {"weighting": "tokens", "batch_size": 32,
                                       "normalize": False}
    info = ms.edit_model(path, apply={"weighting": "types", "batch_size": "8",
                                      "normalize": "yes"})
    assert info.apply == {"weighting": "types", "batch_size": 8, "normalize": True}
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["apply"] == {"weighting": "types", "batch_size": 8, "normalize": True}

    with pytest.raises(ValueError, match="one of tokens, types"):
        ms.edit_model(path, apply={"weighting": "sif"})
    with pytest.raises(ValueError, match="whole number"):
        ms.edit_model(path, apply={"batch_size": "8.5"})
    with pytest.raises(ValueError, match="yes or no"):
        ms.edit_model(path, apply={"normalize": "maybe"})
    with pytest.raises(ValueError, match="not a setting of a fake vectors model"):
        ms.edit_model(path, apply={"color": "blue"})
    with pytest.raises(ValueError, match="no settings for how it is applied"):
        ms.edit_model(_ridge(tmp_path / "r.json"), apply={"x": 1})

    doc["apply"]["batch_size"] = "many"
    doc["apply"]["from_the_future"] = 1
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert ms.apply_defaults(doc) == {"weighting": "types", "batch_size": 32,
                                      "normalize": True}
    assert ms.apply_defaults({"kind": "taters-ridge-model"}) == {}
