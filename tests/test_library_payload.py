"""
Tests for models that carry weights beside their manifest.

A trained model is a small JSON that says what it is, plus weights that do
not belong in a JSON: a word-vector matrix, a transformer checkpoint. The
manifest names them under ``"payload"`` and the library treats the set as
one entry -- imported, exported, renamed and deleted together, never listed
apart. Everything here runs under a temporary ``TATERS_HOME``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taters.helpers import library as lib
from taters.helpers.model_spec import (ENCODER_KIND, MODEL_TYPES, BY_ID,
                                       encoder_problem, describe_encoder,
                                       library_kind_for, models_produced,
                                       one_model_path)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "tathome"))
    return tmp_path / "tathome"


def _encoder(folder: Path, stem: str = "clinical", *, weights=b"w" * 64,
             evaluation=None) -> Path:
    """A structurally complete encoder: manifest + payload folder."""
    payload = folder / f"{stem}.encoder"
    payload.mkdir(parents=True, exist_ok=True)
    (payload / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (payload / "model.safetensors").write_bytes(weights)
    (payload / "vocab.txt").write_text("[PAD]\n[UNK]\nhello\n", encoding="utf-8")
    doc = {"kind": ENCODER_KIND, "format": 1, "name": stem,
           "payload": [payload.name], "base_model": "distilroberta-base",
           "num_hidden_layers": 6,
           "evaluation": evaluation or {"perplexity_before": 15.0,
                                        "perplexity_after": 7.5}}
    manifest = folder / f"{stem}.json"
    manifest.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return manifest


def _mem_model(folder: Path, stem: str = "themes") -> Path:
    """A minimal MEM model file (the topic-model kind), no payload."""
    from taters.text.topic_model_mem import MODEL_FORMAT

    doc = {"kind": "taters-mem-model", "format": MODEL_FORMAT, "name": stem,
           "text": {"lemmatize": False, "pos_tagged": False, "engine": "nltk",
                    "tokenizer": "potts", "stanza_lang": "en",
                    "keep_punctuation": False},
           "matrix": {"weighting": "count", "rounding": 4, "terms": ["a", "b"],
                      "columns": ["a", "b"], "idf": [0.1, 0.2]},
           "model": {"n_documents": 3, "rotation": True, "themes": ["Theme_1"],
                     "kept": [0, 1], "mu": [0.0, 0.0], "sigma": [1.0, 1.0],
                     "projection": [[0.5], [0.5]], "eigenvalues": [1.0],
                     "pct_variance": [50.0]}}
    manifest = folder / f"{stem}.json"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# The payload travels with the manifest
# ---------------------------------------------------------------------------

def test_import_export_rename_and_delete_carry_the_payload(home, tmp_path):
    """One entry, however many files: after every operation the weights sit
    beside the manifest under the manifest's name, and nothing is left
    behind after a delete."""
    kind = lib.KINDS["encoders"]
    manifest = _encoder(tmp_path / "run")
    landed = lib.import_into(kind, manifest)
    assert landed == lib.kind_dir(kind) / "clinical.json"
    assert (lib.kind_dir(kind) / "clinical.encoder" / "model.safetensors").is_file()
    assert [e.name for e in lib.entries(kind)] == ["clinical.json"]

    renamed = lib.rename(kind, "clinical", "notes-v2")
    assert renamed.name == "notes-v2.json"
    assert (lib.kind_dir(kind) / "notes-v2.encoder").is_dir()
    assert not (lib.kind_dir(kind) / "clinical.encoder").exists()
    doc = json.loads(renamed.read_text(encoding="utf-8"))
    assert doc["payload"] == ["notes-v2.encoder"], "the manifest follows its weights"

    out = lib.export_to(kind, "notes-v2", tmp_path / "out")
    assert (tmp_path / "out" / "notes-v2.encoder" / "config.json").is_file()
    assert out.is_file()

    lib.delete(kind, "notes-v2")
    assert lib.entries(kind) == []
    assert not (lib.kind_dir(kind) / "notes-v2.encoder").exists()


def test_a_payload_that_is_not_beside_the_manifest_is_refused_by_name(home, tmp_path):
    manifest = _encoder(tmp_path / "run")
    import shutil
    shutil.rmtree(tmp_path / "run" / "clinical.encoder")
    with pytest.raises(ValueError, match="clinical.encoder"):
        lib.import_into(lib.KINDS["encoders"], manifest)


def test_a_collision_on_the_payload_alone_is_still_a_collision(home, tmp_path):
    """Two manifests with different names but the same weights folder name
    would overwrite each other's weights silently."""
    kind = lib.KINDS["encoders"]
    lib.import_into(kind, _encoder(tmp_path / "a"))
    other = _encoder(tmp_path / "b", stem="other")
    doc = json.loads(other.read_text(encoding="utf-8"))
    (tmp_path / "b" / "clinical.encoder").mkdir()
    for f in (tmp_path / "b" / "other.encoder").iterdir():
        f.rename(tmp_path / "b" / "clinical.encoder" / f.name)
    doc["payload"] = ["clinical.encoder"]
    other.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(lib.LibraryCollision):
        lib.import_into(kind, other)


def test_nothing_inside_a_payload_is_ever_a_library_entry(home, tmp_path):
    """A checkpoint folder holds a config.json and a tokenizer.json; listing
    them as models made "one model" into three, and a step that was handed
    the folder refused with 'N model files found'."""
    kind = lib.KINDS["encoders"]
    lib.import_into(kind, _encoder(tmp_path / "run"))
    folder = lib.kind_dir(kind)
    (folder / "clinical.encoder" / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert [e.name for e in lib.entries(kind)] == ["clinical.json"]
    assert [p.name for p in lib.expand(kind, [folder])] == ["clinical.json"]
    assert [p.name for p in lib.model_files(folder)] == ["clinical.json"]
    assert one_model_path(folder).name == "clinical.json"


def test_the_weights_are_found_in_the_library_by_their_digest(home, tmp_path):
    """A model carried into another run travels as its manifest text; the
    loader finds its weights here by content, even under another name."""
    from taters.helpers.provenance import file_digest

    kind = lib.KINDS["encoders"]
    lib.import_into(kind, _encoder(tmp_path / "run", weights=b"exact bytes" * 8))
    lib.rename(kind, "clinical", "renamed")
    payload = lib.kind_dir(kind) / "renamed.encoder"
    digests = {"model.safetensors": file_digest(payload / "model.safetensors")}
    assert lib.find_payload(kind, "anything.encoder", digests) == payload
    assert lib.find_payload(kind, "anything.encoder",
                            {"model.safetensors": "0" * 64}) is None
    assert lib.find_payload(kind, "x.npy", digests) is None, "wrong shape of payload"


def test_the_row_of_a_model_with_weights_says_how_big_it_is(home, tmp_path):
    kind = lib.KINDS["encoders"]
    manifest = lib.import_into(kind, _encoder(tmp_path / "run", weights=b"x" * 2048))
    label, note = kind.describe_entry(manifest)
    assert label == "clinical [distilroberta-base]"
    assert "6 layers" in note and "15.0 → 7.5" in note
    assert "KB" in note


# ---------------------------------------------------------------------------
# The encoders kind and its gate
# ---------------------------------------------------------------------------

def test_the_encoder_gate_names_what_is_missing(tmp_path):
    manifest = _encoder(tmp_path)
    assert encoder_problem(manifest) == ""
    (tmp_path / "clinical.encoder" / "vocab.txt").unlink()
    assert "tokenizer" in encoder_problem(manifest)
    (tmp_path / "clinical.encoder" / "model.safetensors").unlink()
    assert "weights" in encoder_problem(manifest)
    mem = _mem_model(tmp_path)
    assert "not a text encoder" in encoder_problem(mem)
    assert describe_encoder(manifest)[0] == "clinical [distilroberta-base]"


def test_an_encoder_from_a_newer_taters_is_refused_with_the_remedy(tmp_path):
    manifest = _encoder(tmp_path)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["format"] = 99
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    assert "Update Taters" in encoder_problem(manifest)


# ---------------------------------------------------------------------------
# What a run produced, and where each thing belongs
# ---------------------------------------------------------------------------

def test_models_produced_finds_every_kind_and_nothing_inside_a_payload(tmp_path):
    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    _encoder(run / "models", stem="enc")
    _mem_model(run / "models", stem="themes")
    (run / "models" / "enc.encoder" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (run / "run_manifest.json").write_text('{"steps": []}', encoding="utf-8")
    found = {p.name: label for p, label in models_produced(run)}
    assert set(found) == {"enc.json", "themes.json"}
    assert "text encoder" in found["enc.json"]
    assert "MEM topic model" in found["themes.json"]
    assert library_kind_for(run / "models" / "enc.json") == "encoders"
    assert library_kind_for(run / "models" / "themes.json") == "models"


def test_every_registered_model_kind_has_an_import_gate_loader():
    """An unregistered id passed the deep check silently; now the registry
    and the loaders table are pinned to each other."""
    assert set(lib.MODEL_LOADERS) == set(BY_ID)
    assert set(BY_ID) == {s.id for s in MODEL_TYPES.values()}


# ---------------------------------------------------------------------------
# On screen: Settings, the picker, the finish screen
# ---------------------------------------------------------------------------

def _ridge(path: Path, name: str = "ridge__all") -> Path:
    """A minimal ridge model file -- a scoring model with no payload."""
    path.write_text(json.dumps({
        "kind": "taters-ridge-model", "format": 2, "name": name,
        "predictors": ["f1", "f2"],
        "outcomes": {"age": {"kept": [0], "mu": [0.0], "sigma": [1.0],
                             "coef": [1.0], "intercept": 0.0}},
    }), encoding="utf-8")
    return path


def test_settings_lists_a_screen_for_saved_models_and_one_for_encoders():
    """Two help strings pointed at a "Manage my library" that did not exist,
    and KINDS["models"] was unreachable from any menu. They live under
    Settings -> Manage Taters data now, with everything else Taters keeps."""
    from taters.ui.tasks import manage_data

    by_id = {t.id: t.label for t in manage_data.entries()}
    assert by_id["models"] == "Manage saved models"
    assert by_id["encoders"] == "Manage text encoders"


def test_a_model_step_takes_several_models_and_starts_with_none_ticked(home, tmp_path):
    """Dictionaries open with everything ticked; models open with nothing,
    because "score with every model I ever saved" is not a default anyone
    means. Space ticks, enter approves the ticked plus the pointed one, and
    "Use all of them" is still one row away."""
    from taters.ui.library import pick_from_library
    from taters.ui.prompts import ScriptedPrompter

    kind = lib.KINDS["models"]
    lib.import_into(kind, _ridge(tmp_path / "ridge__all.json"))
    lib.import_into(kind, _ridge(tmp_path / "ridge__b.json", name="ridge__b"))
    a = str((lib.kind_dir(kind) / "ridge__all.json").resolve())
    b = str((lib.kind_dir(kind) / "ridge__b.json").resolve())

    p = ScriptedPrompter([f"\x00space:{a}", b])
    assert pick_from_library(p, kind) == sorted([a, b])
    question, rows = p.offered[0]
    assert question.startswith("Which saved models should this step use?")
    assert all(c.label.startswith("[ ]") for c in rows if c.value in (a, b))
    assert ":all" in {c.value for c in rows}
    assert pick_from_library(ScriptedPrompter([b]), kind) == [b]


def test_the_finish_screen_offers_to_keep_a_model_the_run_produced(home, tmp_path):
    """A run that fitted a model used to end with the file sitting in
    ``models/``; the next study could not pick it. The row appears only
    when there is something to keep, imports on a yes, and disappears
    once taken up."""
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.wizard import finish_screen

    run = tmp_path / "run"
    (run / "models").mkdir(parents=True)
    _ridge(run / "models" / "ridge__all.json")
    (run / "run_manifest.json").write_text('{"items": []}', encoding="utf-8")

    p = ScriptedPrompter(["keep",
                          True,        # add ridge__all to your library?
                          False,       # name the model and its columns?
                          "menu"])
    finish_screen(p, ok=True, manifest={"items": [], "errors": []},
                  folder=run, manifest_path=run / "run_manifest.json")

    first, second = [rows for q, rows in p.offered if q == "What now?"]
    keep = next(c for c in first if c.value == "keep")
    assert keep.label == "Add 1 model from this run to my library"
    assert "ridge" in keep.help
    assert [c.value for c in second] == ["menu", "quit"]
    assert [e.name for e in lib.entries(lib.KINDS["models"])] == ["ridge__all.json"]
    assert any("Add ridge" in q or "to your library?" in q for _k, q in p.asked)


def test_a_run_that_produced_no_model_has_no_keep_row(home, tmp_path):
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.wizard import finish_screen

    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    (run / "features" / "readability.csv").write_text("a,b\n", encoding="utf-8")
    p = ScriptedPrompter(["menu"])
    finish_screen(p, ok=True, manifest={"items": [], "errors": []},
                  folder=run, manifest_path=run / "m.json")
    assert [c.value for c in p.offered[0][1]] == ["menu", "quit"]


def test_the_import_offer_files_each_model_where_it_belongs(home, tmp_path):
    """An encoder goes to the encoders kind with its weights and is not
    asked the scoring model's naming questions; a ridge goes to models and
    is; a collision asks before replacing."""
    from taters.ui.library import offer_library_import
    from taters.ui.prompts import ScriptedPrompter

    run = tmp_path / "run" / "models"
    enc = _encoder(run, stem="enc")
    _ridge(run / "ridge__all.json")
    produced = models_produced(tmp_path / "run")

    p = ScriptedPrompter([True,                  # add the encoder
                          True,                  # add the ridge
                          True, "age-model", "years"])   # name it and its column
    landed = offer_library_import(p, produced)
    assert [x.name for x in landed] == ["enc.json", "ridge__all.json"]
    assert landed[0].parent == lib.kind_dir(lib.KINDS["encoders"])
    assert landed[1].parent == lib.kind_dir(lib.KINDS["models"])
    assert (lib.kind_dir(lib.KINDS["encoders"]) / "enc.encoder").is_dir()
    from taters.helpers.model_spec import describe
    info = describe(landed[1])
    assert info.name == "age-model" and list(info.outputs) == ["years"]
    asked = [q for _k, q in p.asked]
    assert sum("Name this" in q for q in asked) == 1, "encoders have no columns to name"

    # now the same encoder again: a collision, and we decline it.
    p = ScriptedPrompter([True, False])
    assert offer_library_import(p, [(enc, "enc (text encoder)")]) == []
    assert any("already in your library" in q for _k, q in p.asked)
