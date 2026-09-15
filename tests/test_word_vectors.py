"""
Tests for `taters.text.word_vectors`: train, import, apply, describe.

Most of the module is numpy over a matrix the tests write by hand -- four
words in two dimensions, so every mean and cosine can be checked against
arithmetic -- and the model file that carries it. Training itself needs
gensim and is exercised at the end on a toy corpus with two vocabularies;
those tests skip where gensim is not installed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from csvhelpers import _read  # noqa: E402

from taters.text import word_vectors as wv  # noqa: E402
from taters.text.word_vectors import (MODEL_KIND, apply_word_vectors,  # noqa: E402
                                      import_word_vectors, nearest_neighbors)

VOCAB = ["cat", "dog", "fish", "car"]
VECTORS = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
COUNTS = [10, 5, 2, 20]


def _dicx(path: Path, categories, rows) -> Path:
    """A LIWC-22 dictionary: columns are categories, rows are terms, cells
    are weights (X = 1)."""
    lines = ["DicTerm," + ",".join(categories)]
    for term, cells in rows:
        lines.append(term + "," + ",".join("" if c is None else str(c) for c in cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _zoo(tmp_path, name="zoo.dicx") -> Path:
    """pets = cat, dog (both X); wheels = car (X)."""
    return _dicx(tmp_path / name, ["pets", "wheels"],
                 [("cat", ["X", None]), ("dog", ["X", None]), ("car", [None, "X"])])


def _model(folder: Path, *, counts=COUNTS, concept_dicts=(), weighting="tokens",
           stem="wv", lowercase=True) -> Path:
    """A hand-built word-vector model: manifest + .npy, as the trainer writes.
    ``concept_dicts`` are dictionary files whose content is stored, as the
    trainer stores it."""
    from taters.helpers.provenance import file_digest
    from taters.text._concept_dicts import concept_dicts_to_json, load_concept_dicts

    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / f"{stem}.npy"
    np.save(weights, VECTORS)
    doc = {"kind": MODEL_KIND, "format": 1, "name": stem,
           "payload": [weights.name],
           "payload_digests": {weights.name: file_digest(weights)},
           "text": {"lemmatize": False, "pos_tagged": False, "engine": "nltk",
                    "tokenizer": "potts", "stanza_lang": "en",
                    "keep_punctuation": False, "lowercase": lowercase},
           "vocabulary": VOCAB, "counts": counts, "dim": 2,
           "dim_labels": ["wv_1", "wv_2"],
           "training": {"source": "trained", "family": "word2vec"},
           "apply": {"weighting": weighting, "normalize_words": False,
                     "concept_dicts": concept_dicts_to_json(load_concept_dicts(list(concept_dicts)))}}
    manifest = folder / f"{stem}.json"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    return manifest


def _corpus(path: Path, texts) -> Path:
    path.write_text("text_id,text\n" + "".join(f"t{i},{t}\n" for i, t in enumerate(texts)),
                    encoding="utf-8")
    return path


def _apply(tmp_path, model, texts, **kw):
    out = apply_word_vectors(model_json=model, csv_path=_corpus(tmp_path / "c.csv", texts),
                             text_cols=["text"], id_cols=["text_id"],
                             out_features_csv=tmp_path / "out.csv",
                             overwrite_existing=True, verbose=False, **kw)
    return _read(out)


# ---------------------------------------------------------------------------
# the mean vector
# ---------------------------------------------------------------------------

def test_a_texts_vector_is_the_mean_of_its_words_and_unknown_words_are_counted_out(tmp_path):
    rows = _apply(tmp_path, _model(tmp_path / "m"), ["cat dog dog zebra"])
    r = rows[0]
    assert (r["token_count"], r["in_vocab_count"]) == ("4", "3")
    assert float(r["wv_1"]) == pytest.approx((1.0 + 0.9 + 0.9) / 3, abs=1e-4)
    assert float(r["wv_2"]) == pytest.approx(0.2 / 3, abs=1e-4)
    assert list(r) == ["text_id", "token_count", "in_vocab_count", "wv_1", "wv_2"]


def test_types_weighting_counts_each_distinct_word_once(tmp_path):
    rows = _apply(tmp_path, _model(tmp_path / "m"), ["cat dog dog"], weighting="types")
    assert float(rows[0]["wv_1"]) == pytest.approx(0.95, abs=1e-4)
    assert rows[0]["in_vocab_count"] == "3", "the count is of tokens, whatever the weighting"


def test_sif_weighting_downweights_the_commonest_words(tmp_path):
    """a / (a + p(word)): `car` (20 of 37) weighs far less than `fish` (2 of
    37), so a text of one of each leans to fish."""
    rows = _apply(tmp_path, _model(tmp_path / "m"), ["fish car"], weighting="sif")
    a = wv.SIF_A
    w_fish, w_car = a / (a + 2 / 37), a / (a + 20 / 37)
    expected = (w_fish * VECTORS[2] + w_car * VECTORS[3]) / (w_fish + w_car)
    assert float(rows[0]["wv_1"]) == pytest.approx(expected[0], abs=1e-4)
    assert float(rows[0]["wv_2"]) == pytest.approx(expected[1], abs=1e-4)


def test_sif_is_refused_by_name_for_a_model_without_counts(tmp_path):
    with pytest.raises(ValueError, match="no word counts"):
        _apply(tmp_path, _model(tmp_path / "m", counts=None), ["cat"], weighting="sif")


def test_a_text_with_no_known_word_gets_blanks_not_zeros(tmp_path):
    """A zero vector would sit at the center of every concept and count as
    a real measurement in the statistics."""
    rows = _apply(tmp_path, _model(tmp_path / "m", concept_dicts=[_zoo(tmp_path)]),
                  ["zebra giraffe", "...", "cat"])
    assert rows[0]["in_vocab_count"] == "0" and rows[0]["wv_1"] == "" \
        and rows[0]["sim_zoo__pets"] == ""
    # punctuation on its own isn't a word (the gather drops a truly empty text)
    assert rows[1]["token_count"] == "0" and rows[1]["wv_2"] == ""
    assert rows[2]["wv_1"] == "1.0"


def test_normalizing_words_evens_out_long_vectors(tmp_path):
    """fish has norm 0.71, cat norm 1: unnormalised, cat pulls the mean its
    way; normalized, both count alike."""
    plain = _apply(tmp_path, _model(tmp_path / "m"), ["cat fish"])
    even = _apply(tmp_path, _model(tmp_path / "m"), ["cat fish"], normalize_words=True)
    assert float(plain[0]["wv_2"]) == pytest.approx(0.25, abs=1e-4)
    assert float(even[0]["wv_2"]) == pytest.approx((0 + 0.5 / np.sqrt(0.5)) / 2, abs=1e-4)


def test_words_are_read_as_the_model_reads_them(tmp_path):
    """Text is lower-cased on the way in, so "Cat" finds cat -- and the
    manifest records that, so a model file says how it reads."""
    rows = _apply(tmp_path, _model(tmp_path / "m"), ["Cat CAT"])
    assert rows[0]["in_vocab_count"] == "2"
    assert wv._load_model(_model(tmp_path / "m")).text_settings["lowercase"] is True


# ---------------------------------------------------------------------------
# concepts
# ---------------------------------------------------------------------------

def test_a_dictionary_is_read_with_its_categories_weights_and_wildcards(tmp_path):
    """Through contentcoder, so a .dicx, a .csv and a 2007 .dic all mean
    what they mean to the dictionary analyzer; X is 1.0 and a number is
    the weight as written."""
    from taters.text._concept_dicts import load_concept_dicts

    path = _dicx(tmp_path / "moods.dicx", ["happy", "calm"],
                 [("happy", ["2", None]), ("glad*", ["X", None]), ("calm", ["X", "1"]),
                  ("very calm", [None, "3"])])
    (d,) = load_concept_dicts([path])
    assert d.name == "moods"
    assert d.categories["happy"] == [("happy", 2.0), ("glad*", 1.0), ("calm", 1.0)]
    assert d.categories["calm"] == [("calm", 1.0), ("very calm", 3.0)]
    as_csv = _dicx(tmp_path / "moods.csv", ["happy"], [("happy", ["X"])])
    assert load_concept_dicts([as_csv])[0].categories == {"happy": [("happy", 1.0)]}
    (tmp_path / "folder").mkdir()
    _dicx(tmp_path / "folder" / "a.dicx", ["x"], [("a", ["X"])])
    _dicx(tmp_path / "folder" / "b.dicx", ["y"], [("b", ["X"])])
    assert [d.name for d in load_concept_dicts([tmp_path / "folder"])] == ["a", "b"]
    (tmp_path / "words.dic").write_text("cat\ndog\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_concept_dicts([tmp_path / "words.dic"])
    with pytest.raises(FileNotFoundError):
        load_concept_dicts([tmp_path / "nowhere.dicx"])


def test_similarity_to_a_category_is_the_cosine_to_its_terms_weighted_mean(tmp_path):
    """pets = cat, dog at weight 1 each: mean(cat, dog) = (0.95, 0.05). The
    text "car" is nearly orthogonal to it, "cat" nearly aligned."""
    rows = _apply(tmp_path, _model(tmp_path / "m", concept_dicts=[_zoo(tmp_path)]),
                  ["car", "cat"])
    pets = VECTORS[:2].mean(axis=0)
    cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    assert float(rows[0]["sim_zoo__pets"]) == pytest.approx(cos(VECTORS[3], pets), abs=1e-4)
    assert float(rows[1]["sim_zoo__pets"]) == pytest.approx(cos(VECTORS[0], pets), abs=1e-4)
    assert float(rows[0]["sim_zoo__wheels"]) == pytest.approx(1.0, abs=1e-4)
    assert list(rows[0])[-2:] == ["sim_zoo__pets", "sim_zoo__wheels"]


def test_cell_weights_and_wildcards_shape_the_category_vector(tmp_path):
    """A `2` pulls the vector twice as far as a `1`; a wildcard that matches
    two words is still one term with one vote; a phrase is the mean of its
    words; a negative weight pushes away."""
    from taters.text._concept_dicts import resolve_concepts
    from taters.text.word_vectors import _load_model, _stream_for

    model = _load_model(_model(tmp_path / "m"))
    stream = _stream_for(model.text_settings, "cpu")

    def vector_of(rows):
        d = _dicx(tmp_path / "w.dicx", ["c"], rows)
        from taters.text._concept_dicts import load_concept_dicts

        cols, vecs, report = resolve_concepts(model.index, model.vectors,
                                              load_concept_dicts([d]), stream, True)
        assert cols == ["sim_w__c"]
        return vecs[0], report[0]

    v, _ = vector_of([("cat", ["2"]), ("dog", ["1"])])
    assert np.allclose(v, (2 * VECTORS[0] + VECTORS[1]) / 3)
    # c* matches cat and car -> one term, their mean; dog is the other term
    v, report = vector_of([("c*", ["X"]), ("dog", ["X"])])
    assert np.allclose(v, ((VECTORS[0] + VECTORS[3]) / 2 + VECTORS[1]) / 2)
    assert report["n_matched"] == 2 and report["missed"] == []
    # a phrase is the mean of its words
    v, _ = vector_of([("cat dog", ["X"])])
    assert np.allclose(v, (VECTORS[0] + VECTORS[1]) / 2)
    # a negative weight (hand-built: the dictionary reader keeps whatever
    # weight contentcoder hands over) pushes away, normalized by |weights|
    from collections import OrderedDict

    from taters.text._concept_dicts import ConceptDict

    hand = ConceptDict("w", OrderedDict({"c": [("cat", 1.0), ("car", -1.0)]}))
    _cols, vecs, _r = resolve_concepts(model.index, model.vectors, [hand], stream, True)
    assert np.allclose(vecs[0], (VECTORS[0] - VECTORS[3]) / 2)


def test_a_category_the_model_does_not_know_is_refused_and_a_partial_one_reported(tmp_path):
    from taters.text._concept_dicts import load_concept_dicts, resolve_concepts
    from taters.text.word_vectors import _load_model, _stream_for

    model = _load_model(_model(tmp_path / "m"))
    stream = _stream_for(model.text_settings, "cpu")
    nothing = _dicx(tmp_path / "far.dicx", ["exotic"], [("zebra", ["X"]), ("giraffe", ["X"])])
    with pytest.raises(ValueError, match="none of the 2 term\\(s\\) of 'exotic' in far"):
        resolve_concepts(model.index, model.vectors, load_concept_dicts([nothing]), stream, True)
    from collections import OrderedDict

    from taters.text._concept_dicts import ConceptDict

    # weights are normalized by their absolute sum, so 1 and -1 is a
    # contrast, not a cancellation; only all-zero weights leave no direction
    cancel = ConceptDict("zero", OrderedDict({"z": [("cat", 0.0), ("dog", 0.0)]}))
    with pytest.raises(ValueError, match="add up to zero"):
        resolve_concepts(model.index, model.vectors, [cancel], stream, True)
    partial = _dicx(tmp_path / "some.dicx", ["pets"], [("cat", ["X"]), ("unicorn", ["X"])])
    cols, vecs, rows = resolve_concepts(model.index, model.vectors,
                                        load_concept_dicts([partial]), stream, True)
    assert rows[0]["missed"] == ["unicorn"] and rows[0]["n_matched"] == 1
    assert np.allclose(vecs[0], VECTORS[0])
    with pytest.raises(ValueError, match="'exotic'"):
        _apply(tmp_path, _model(tmp_path / "n"), ["cat"], concept_dicts=[nothing])


def test_the_models_own_apply_settings_are_used_unless_the_call_overrides(tmp_path):
    """The model file says types weighting and carries the zoo dictionary;
    a plain call honors both, and the call that names its own weighting
    and its own (empty) dictionary list wins."""
    model = _model(tmp_path / "m", weighting="types", concept_dicts=[_zoo(tmp_path)])
    rows = _apply(tmp_path, model, ["cat dog dog"])
    assert float(rows[0]["wv_1"]) == pytest.approx(0.95, abs=1e-4)
    assert "sim_zoo__pets" in rows[0]
    rows = _apply(tmp_path, model, ["cat dog dog"], weighting="tokens", concept_dicts=[])
    assert float(rows[0]["wv_1"]) == pytest.approx(2.8 / 3, abs=1e-4)
    assert "sim_zoo__pets" not in rows[0]


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------

def test_neighbors_are_sorted_by_similarity_and_never_the_probe_itself(tmp_path):
    found = nearest_neighbors(_model(tmp_path / "m"), ["cat", "zebra"], top_n=2)
    assert [w for w, _ in found["cat"]] == ["dog", "fish"]
    assert found["cat"][0][1] > found["cat"][1][1]
    assert found["zebra"] == []


def test_neighbors_come_out_the_same_in_chunks(tmp_path, monkeypatch):
    """The matrix is searched a slice at a time so a large model is never
    copied; a slice smaller than the vocabulary must change nothing."""
    whole = nearest_neighbors(_model(tmp_path / "m"), ["car"], top_n=3)
    monkeypatch.setattr(wv, "_CHUNK_ROWS", 1)
    assert nearest_neighbors(_model(tmp_path / "m"), ["car"], top_n=3) == whole


def test_the_neighbors_table_shows_an_unknown_probe_as_a_blank_row(tmp_path):
    out = wv.describe_word_vectors(_model(tmp_path / "m"), tmp_path / "n.csv",
                                   probes="cat, zebra", top_neighbors=1, verbose=False)
    rows = _read(out)
    assert list(rows[0]) == ["probe", "rank", "word", "similarity"]
    assert (rows[0]["probe"], rows[0]["word"]) == ("cat", "dog")
    assert [r["word"] for r in rows if r["probe"] == "zebra"] == [""]


# ---------------------------------------------------------------------------
# the model file and its weights
# ---------------------------------------------------------------------------

def test_the_gate_refuses_what_is_not_a_word_vector_model(tmp_path):
    model = _model(tmp_path / "m")
    doc = json.loads(model.read_text(encoding="utf-8"))
    doc["kind"] = "taters-mem-model"
    (tmp_path / "m" / "other.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="not a word-vector model"):
        wv._load_model(tmp_path / "m" / "other.json")
    doc["kind"] = MODEL_KIND
    doc["format"] = 99
    model.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="Update Taters"):
        wv._load_model(model)


def test_weights_that_are_not_the_models_are_refused(tmp_path):
    """The digest is checked once at load: a matrix that does not belong to
    this vocabulary would score every text with someone else's meanings."""
    model = _model(tmp_path / "m")
    np.save(tmp_path / "m" / "wv.npy", VECTORS * 2)
    with pytest.raises(ValueError, match="not the matrix"):
        wv._load_model(model)
    doc = json.loads(model.read_text(encoding="utf-8"))
    doc.pop("payload_digests")
    np.save(tmp_path / "m" / "wv.npy", VECTORS[:3])
    model.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="3x2 but the manifest describes 4 words"):
        wv._load_model(model)


def test_a_manifest_without_its_weights_names_both_fixes(tmp_path):
    model = _model(tmp_path / "m")
    (tmp_path / "m" / "wv.npy").unlink()
    with pytest.raises(ValueError, match="not beside it and not in your library"):
        wv._load_model(model)


def test_a_manifest_that_traveled_alone_finds_its_weights_in_the_library(tmp_path, monkeypatch):
    """A model carried inside a ridge's record is materialized as its
    manifest text alone; the loader finds the matrix by digest in the
    library, even under another name."""
    from taters.helpers import library as lib

    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    model = _model(tmp_path / "m")
    lib.import_into(lib.KINDS["models"], model)
    lib.rename(lib.KINDS["models"], "wv", "renamed")
    alone = tmp_path / "elsewhere" / "wv.json"
    alone.parent.mkdir()
    shutil.copy(model, alone)
    loaded = wv._load_model(alone)
    assert loaded.vectors.shape == (4, 2)
    rows = _apply(tmp_path, alone, ["cat"])
    assert rows[0]["wv_1"] == "1.0"


def test_the_registry_describes_and_edits_a_word_vector_model(tmp_path):
    """Outputs are the dimensions (renamable by prefix), the columns add one
    per concept, and the apply block is editable through the registry."""
    from taters.helpers.model_spec import describe, edit_model

    model = _model(tmp_path / "m", concept_dicts=[_zoo(tmp_path)])
    info = describe(model)
    assert (info.type_id, info.needs, info.bulk_outputs) == ("word_vectors", "text", True)
    assert info.outputs == ("wv_1", "wv_2")
    assert info.columns == ("wv_1", "wv_2", "sim_zoo__pets", "sim_zoo__wheels")
    assert info.apply["weighting"] == "tokens" and info.apply["normalize_words"] is False
    assert [d["name"] for d in info.apply["concept_dicts"]] == ["zoo"]
    assert info.apply["concept_dicts"][0]["categories"]["pets"] == [["cat", 1.0], ["dog", 1.0]]
    # editing hands over dictionary *files*; the model keeps their content
    only_pets = _dicx(tmp_path / "pets.dicx", ["pets"], [("cat", ["X"])])
    info = edit_model(model, prefix="emb", apply={"weighting": "types",
                                                  "concept_dicts": [str(only_pets)]})
    assert info.outputs == ("emb_1", "emb_2") and info.columns[-1] == "sim_pets__pets"
    rows = _apply(tmp_path, model, ["cat dog dog"])
    assert list(rows[0])[3:] == ["emb_1", "emb_2", "sim_pets__pets"]
    assert float(rows[0]["emb_1"]) == pytest.approx(0.95, abs=1e-4), "types, from the file"
    assert edit_model(model, apply={"concept_dicts": []}).columns == ("emb_1", "emb_2")
    with pytest.raises(ValueError, match="one of tokens, types, sif"):
        edit_model(model, apply={"weighting": "mean"})


def test_the_library_import_gate_is_the_loader(tmp_path, monkeypatch):
    from taters.helpers import library as lib

    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "home"))
    model = _model(tmp_path / "m")
    assert lib.asset_problem(model, lib.KINDS["models"]) == ""
    np.save(tmp_path / "m" / "wv.npy", VECTORS * 2)
    assert "not the matrix" in lib.asset_problem(model, lib.KINDS["models"])


def test_score_with_model_scores_a_word_vector_model_from_text(tmp_path):
    from taters.score_model import score_with_model

    out = score_with_model(model_json=_model(tmp_path / "m", concept_dicts=[_zoo(tmp_path)]),
                           csv_path=_corpus(tmp_path / "c.csv", ["cat car"]),
                           out_csv=tmp_path / "scored.csv", verbose=False)
    rows = _read(out)
    assert {"wv_1", "wv_2", "sim_zoo__pets"} <= set(rows[0])
    assert float(rows[0]["wv_1"]) == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# importing pre-trained vectors (the text formats only need numpy)
# ---------------------------------------------------------------------------

def test_glove_text_is_imported_case_folded_and_capped(tmp_path):
    """The first occurrence of a word wins when case folding merges two, the
    cap keeps the top of the file (these files list words by frequency),
    and the model that results scores text like a trained one."""
    src = tmp_path / "glove.txt"
    src.write_text("Cat 1 0\ncat 0 1\ndog 0.9 0.1\ncar 0 1\nfish 0.5 0.5\n", encoding="utf-8")
    model = import_word_vectors(src, tmp_path / "models" / "glove.json", max_vocab=3,
                                verbose=False)
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert doc["vocabulary"] == ["cat", "dog", "car"]
    assert doc["counts"] is None and doc["dim"] == 2
    assert doc["training"]["source"] == "imported" and doc["training"]["format"] == "glove"
    assert (tmp_path / "models" / "glove.npy").is_file()
    rows = _apply(tmp_path, model, ["cat"])
    assert (rows[0]["wv_1"], rows[0]["wv_2"]) == ("1.0", "0.0")
    assert (tmp_path / "models" / "glove_report.md").is_file()
    assert (tmp_path / "models" / "glove_neighbors.csv").is_file()


def test_word2vec_text_is_recognized_by_its_header_line(tmp_path):
    src = tmp_path / "vectors.txt"
    src.write_text("2 3\nalpha 1 2 3\nbeta 4 5 6\n", encoding="utf-8")
    assert wv._sniff_format(src) == "word2vec_text"
    model = import_word_vectors(src, tmp_path / "v.json", verbose=False)
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert doc["vocabulary"] == ["alpha", "beta"] and doc["dim"] == 3
    src.write_text("alpha 1 2 3\n", encoding="utf-8")
    assert wv._sniff_format(src) == "glove"


def test_an_imported_model_refuses_sif_and_an_unreadable_file_by_name(tmp_path):
    src = tmp_path / "glove.txt"
    src.write_text("cat 1 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="carry no word counts"):
        import_word_vectors(src, tmp_path / "v.json", weighting="sif", verbose=False)
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no readable word vectors"):
        import_word_vectors(tmp_path / "empty.txt", tmp_path / "v.json", verbose=False)
    with pytest.raises(FileNotFoundError):
        import_word_vectors(tmp_path / "nowhere.txt", tmp_path / "v.json", verbose=False)


# ---------------------------------------------------------------------------
# training (gensim)
# ---------------------------------------------------------------------------


FOOD = "potato gravy butter salt dinner supper kitchen recipe".split()
WORK = "meeting deadline email boss office budget report client".split()


def _two_topic_corpus(path: Path, n=200, seed=3) -> Path:
    import random

    rng = random.Random(seed)
    texts = []
    for i in range(n):
        theme = FOOD if i % 2 == 0 else WORK
        words = [rng.choice(theme) for _ in range(rng.randint(15, 40))]
        words += [rng.choice(["the", "and", "a"]) for _ in range(8)]
        rng.shuffle(words)
        texts.append(" ".join(words))
    return _corpus(path, texts)


def _train(tmp_path, **kw):
    args = dict(csv_path=_two_topic_corpus(tmp_path / "corpus.csv"),
                text_cols=["text"], id_cols=["text_id"],
                out_features_csv=tmp_path / "features" / "word_vectors.csv",
                min_count=2, vector_size=12, epochs=6, seed=1, reproducible=True,
                overwrite_existing=True, verbose=False)
    args.update(kw)
    return wv.train_word_vectors(**args)


needs_gensim = pytest.mark.skipif(
    pytest.importorskip("importlib.util").find_spec("gensim") is None,
    reason="gensim is not installed (pip install taters[vectors])")


@needs_gensim
def test_training_writes_the_model_its_features_the_neighbors_and_the_report(tmp_path):
    kitchen = _dicx(tmp_path / "kitchen.dicx", ["food", "work"],
                    [("potato", ["X", None]), ("grav*", ["2", None]),
                     ("boss", [None, "X"]), ("office", [None, "X"])])
    out = _train(tmp_path, concept_dicts=[kitchen], probes="potato, boss")
    models = tmp_path / "features" / "models"
    assert {p.name for p in models.iterdir()} == {
        "word_vectors.json", "word_vectors.npy", "word_vectors_neighbors.csv",
        "word_vectors_report.md", "word_vectors_loss.png"}
    rows = _read(out)
    assert len(rows) == 200
    assert list(rows[0])[:3] == ["text_id", "token_count", "in_vocab_count"]
    assert list(rows[0])[-2:] == ["sim_kitchen__food", "sim_kitchen__work"]
    # food texts should sit nearer the food concept, work texts nearer work
    food = [float(r["sim_kitchen__food"]) - float(r["sim_kitchen__work"]) for r in rows[0::2]]
    work = [float(r["sim_kitchen__food"]) - float(r["sim_kitchen__work"]) for r in rows[1::2]]
    assert min(food) > max(work)
    doc = json.loads((models / "word_vectors.json").read_text(encoding="utf-8"))
    t = doc["training"]
    assert t["n_documents"] == 200 and t["coverage"] == pytest.approx(1.0)
    assert len(t["loss_per_epoch"]) == 6 and t["gensim_version"]
    report = (models / "word_vectors_report.md").read_text(encoding="utf-8")
    assert "## Methods paragraph" in report and "skip-gram" in report
    assert "covered 100.0% of the corpus tokens" in report
    assert "| 6 |" in report, "the loss table runs to the last epoch"
    assert "![training loss](word_vectors_loss.png)" in report
    assert "**potato**" in report and "gravy" in report
    assert "## Concepts" in report and "| sim_kitchen__food | 2 | 2 | 3 |" in report, \
        "two terms matched (grav* is one term), weight 1 + 2"
    assert "**sim_kitchen__food**" in report, "the category's own neighbors are shown"
    neighbors = _read(tmp_path / "features" / "models" / "word_vectors_neighbors.csv")
    assert {r["probe"] for r in neighbors} >= {"potato", "boss", "sim_kitchen__food",
                                                "sim_kitchen__work"}
    assert "gensim" in report and "numpy" in report


@needs_gensim
def test_the_neighbors_of_a_food_word_are_food_words(tmp_path):
    _train(tmp_path, probes="potato")
    rows = _read(tmp_path / "features" / "models" / "word_vectors_neighbors.csv")
    top = [r["word"] for r in rows if r["probe"] == "potato"][:5]
    assert set(top) <= set(FOOD) - {"potato"}


@needs_gensim
def test_applying_the_model_to_its_training_texts_reproduces_the_features(tmp_path):
    """The contract that carries the feature: one row-maker and one
    tokenizing path for fit and apply."""
    kitchen = _dicx(tmp_path / "kitchen.dicx", ["food"], [("potato", ["X"]), ("grav*", ["X"])])
    out = _train(tmp_path, concept_dicts=[kitchen])
    again = apply_word_vectors(
        model_json=tmp_path / "features" / "models" / "word_vectors.json",
        csv_path=tmp_path / "corpus.csv", text_cols=["text"], id_cols=["text_id"],
        out_features_csv=tmp_path / "again.csv", verbose=False)
    assert again.read_bytes() == out.read_bytes()


@needs_gensim
def test_the_same_seed_on_one_thread_gives_the_same_vectors_and_another_does_not(tmp_path):
    _train(tmp_path)
    first = np.load(tmp_path / "features" / "models" / "word_vectors.npy")
    _train(tmp_path)
    assert np.array_equal(np.load(tmp_path / "features" / "models" / "word_vectors.npy"), first)
    _train(tmp_path, seed=2)
    assert not np.array_equal(np.load(tmp_path / "features" / "models" / "word_vectors.npy"), first)


@needs_gensim
def test_cbow_and_fasttext_train_and_say_so(tmp_path):
    _train(tmp_path, algorithm="cbow")
    doc = json.loads((tmp_path / "features" / "models" / "word_vectors.json").read_text(encoding="utf-8"))
    assert doc["training"]["algorithm"] == "cbow"
    assert "CBOW" in (tmp_path / "features" / "models" / "word_vectors_report.md").read_text(encoding="utf-8")
    _train(tmp_path, family="fasttext")
    doc = json.loads((tmp_path / "features" / "models" / "word_vectors.json").read_text(encoding="utf-8"))
    assert doc["training"]["family"] == "fasttext"
    assert doc["training"]["loss_per_epoch"] == []
    assert np.load(tmp_path / "features" / "models" / "word_vectors.npy").shape[1] == 12
    report = (tmp_path / "features" / "models" / "word_vectors_report.md").read_text(encoding="utf-8")
    assert "reports no training loss for this family" in report
    assert not (tmp_path / "features" / "models" / "word_vectors_loss.png").exists()


@needs_gensim
def test_a_corpus_with_nothing_above_min_count_is_refused_before_training(tmp_path):
    with pytest.raises(ValueError, match="not enough to train on"):
        _train(tmp_path, min_count=10_000)
    assert not (tmp_path / "features" / "models" / ".word_vectors_tokens.txt").exists()


@needs_gensim
def test_a_bad_family_or_weighting_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="family must be one of"):
        _train(tmp_path, family="glove")
    with pytest.raises(ValueError, match="weighting must be one of"):
        _train(tmp_path, weighting="mean")


@needs_gensim
def test_the_text_settings_travel_with_the_model(tmp_path):
    """A model trained lemmatised reads new text lemmatised: "dinners" and
    "dinner" are one word to it."""
    _train(tmp_path, lemmatize=True)
    model = tmp_path / "features" / "models" / "word_vectors.json"
    doc = json.loads(model.read_text(encoding="utf-8"))
    assert doc["text"]["lemmatize"] is True
    rows = _apply(tmp_path, model, ["dinners potatoes"])
    assert rows[0]["in_vocab_count"] == "2"


def test_training_without_gensim_says_what_to_install(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "gensim" else real(name, *a, **k))
    with pytest.raises(ImportError, match=r"taters\[vectors\]"):
        wv.train_word_vectors(csv_path=_corpus(tmp_path / "c.csv", ["cat"]),
                              text_cols=["text"], verbose=False)
