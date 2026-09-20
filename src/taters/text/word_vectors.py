"""
Word vectors: train them on your own texts, bring them from elsewhere, and
turn them into features.

A word vector model learns, from nothing but co-occurrence, that *grave*
sits near *death* and far from *picnic*. Trained on the corpus under study
it captures how *these* writers use words -- what "home" means in a
bereavement forum is not what it means in a real-estate listing -- and a
pre-trained set (GloVe, word2vec, fastText) brings a general-purpose sense
of the language to a corpus too small to train on. Either way the model is
a matrix: one row per word, one column per dimension.

Three things are done with it here:

* **The mean vector per text** -- ``wv_1 .. wv_k`` -- the classic bag-of-
  vectors representation, a compact numeric fingerprint that a ridge or
  classifier can learn from.
* **Similarity to concepts** -- ``sim_mydict__death``, ``sim_mydict__work``
  -- the cosine between a text's vector and the weighted mean vector of a
  category of a LIWC-22 dictionary (``.dic``, ``.dicx``, ``.csv``: one
  column per category, wildcards and phrases as LIWC reads them, cells as
  weights), so a hypothesis ("these texts dwell on mortality") becomes one
  column per category. See :mod:`taters.text._concept_dicts`.
* **Nearest neighbors** of chosen words, as a table and as word clouds, so
  the model can be inspected and reported: the neighbors are the evidence
  that it learned what you think it learned.

Two commitments shape the module:

* **A fitted model is a reusable instrument.** The manifest (``.json``)
  records the vocabulary, the text settings that produced it, how it was
  trained and how it should be applied; the matrix travels beside it as
  ``<stem>.npy`` (a *payload*, in the library's terms -- moved, renamed and
  deleted with the manifest). Applying the model to its own training texts
  reproduces the training features exactly.
* **Memory-safe on an ordinary laptop.** The matrix is memory-mapped, never
  copied whole; nearest neighbors are computed in row chunks; training
  streams the tokenized corpus from a scratch file (gensim keeps only the
  vocabulary and the matrix in memory). gensim is needed only to *train*
  and to read the binary word2vec/fastText formats; applying, describing
  and importing text formats need numpy alone.

fastText's subword vectors are not kept (the bucket matrix is hundreds of
megabytes and mostly noise for feature extraction); a word outside the
vocabulary is skipped for both families, and the ``in_vocab_count`` column
says how many words each text lost that way.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Literal, Optional,
                    Sequence, Tuple, Union)

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.model_spec import one_model_path
from ..helpers.progress import announce
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready
from ._concept_dicts import (concept_dicts_from_json, concept_dicts_to_json,
                             load_concept_dicts, resolve_concepts)
from .ngram_prep import make_token_stream

__all__ = ["train_word_vectors", "apply_word_vectors", "import_word_vectors",
           "describe_word_vectors", "nearest_neighbors",
           "MODEL_KIND", "MODEL_FORMAT", "WEIGHTINGS", "FAMILIES",
           "ALGORITHMS", "VECTOR_FORMATS"]

PathLike = Union[str, Path]

#: The tag a word-vector manifest carries, and its format, bumped on any
#: incompatible change so an older Taters refuses a newer model.
MODEL_KIND = "taters-word-vectors-model"
MODEL_FORMAT = 1

#: One column per vector dimension, plus whatever the concept dictionaries add.
FEATURE_COLUMNS = ColumnSpec(
    label="Word vectors",
    patterns=("wv_{n}",),
    dynamic="one column per concept-dictionary category, named from the file",
)

WEIGHTINGS = ("tokens", "types", "sif")
FAMILIES = ("word2vec", "fasttext")
ALGORITHMS = ("skipgram", "cbow")
VECTOR_FORMATS = ("auto", "glove", "word2vec_text", "word2vec_bin", "fasttext_bin")

#: The smoothing constant of SIF weighting (Arora, Liang & Ma, 2017): a word
#: weighs a / (a + p(word)), so the commonest words count least.
SIF_A = 1e-3

#: Rows of the matrix handled at once when searching for neighbors: bounds
#: the working copy to a few tens of megabytes whatever the vocabulary.
_CHUNK_ROWS = 65536

#: The probe words whose neighbors the report and the clouds show when none
#: are named: a dozen common content words, spelled out so they can be read,
#: argued with and replaced (``probes``). Any not in a model's vocabulary is
#: skipped. Function words are not here on purpose -- the neighbors of "the"
#: are the same in every corpus and diagnostic of none -- and no stoplist is
#: involved in deciding that: the list is the whole of the rule.
DEFAULT_PROBES: Tuple[str, ...] = (
    "time", "people", "work", "life", "home", "family", "money", "love",
    "food", "friend", "think", "good",
)


# ---------------------------------------------------------------------------
# The loaded model
# ---------------------------------------------------------------------------

@dataclass
class _Loaded:
    """A model read and vetted: its manifest, its matrix (memory-mapped),
    and the lookups every scorer needs."""

    path: Path
    doc: dict
    vectors: "object"                # an np.ndarray or np.memmap, V x k
    index: Dict[str, int]
    counts: Optional["object"]       # an np.ndarray of ints, or None

    @property
    def dim(self) -> int:
        return int(self.doc["dim"])

    @property
    def dim_labels(self) -> List[str]:
        return [str(x) for x in self.doc["dim_labels"]]

    @property
    def text_settings(self) -> dict:
        return dict(self.doc["text"])

    def total_count(self) -> float:
        return float(self.counts.sum()) if self.counts is not None else 0.0


def _payload_path(manifest: Path, doc: dict) -> Path:
    """
    The matrix beside the manifest, or the copy the library holds.

    A model replayed inside another run travels as its manifest text alone
    -- that is what a ridge fitted on these features carries -- so the
    matrix is looked for in the library by the digest the manifest records.
    The refusal names both fixes.
    """
    names = [str(n) for n in (doc.get("payload") or [])]
    if not names:
        raise ValueError(
            f"{manifest.name} names no weights file (its 'payload' list is "
            f"empty). Re-train or re-import the word vectors to write one.")
    beside = manifest.with_name(Path(names[0]).name)
    if beside.is_file():
        return beside
    digests = doc.get("payload_digests") or {}
    try:
        from ..helpers.library import KINDS, find_payload

        found = find_payload(KINDS["models"], names[0], digests)
    except Exception:
        found = None
    if found is not None:
        return found
    raise ValueError(
        f"{manifest.name} needs its weights file {names[0]}, which is not "
        f"beside it and not in your library. Put {names[0]} next to the "
        f"model file, or add the model to your library (Settings → Manage "
        f"saved models) so any run can find it.")


def _load_model(model_json: PathLike) -> _Loaded:
    """
    Load and *vet* a word-vector model: the import gate's authority too.

    Every refusal names what is wrong. The matrix is memory-mapped, so a
    200,000 x 300 model costs a few kilobytes to open; its digest is checked
    once here (a few hundred megabytes hash in well under a second) because
    a matrix that does not belong to this vocabulary would score every text
    with someone else's meanings and never say so.
    """
    import numpy as np

    from ..helpers.provenance import file_digest

    path = Path(model_json)
    if not path.exists():
        raise FileNotFoundError(f"model_json not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    kind = doc.get("kind")
    if kind != MODEL_KIND:
        what = f"a {kind!r} file" if kind else "not a Taters model file at all"
        raise ValueError(
            f"{path.name} is not a word-vector model ({what}). Only models "
            f"written by the word-vectors steps can be applied here.")
    if int(doc.get("format", 0)) > MODEL_FORMAT:
        raise ValueError(
            f"{path.name} was written by a newer Taters (model format "
            f"{doc.get('format')}; this build reads up to {MODEL_FORMAT}). "
            f"Update Taters to use it.")

    def _broken(why: str) -> ValueError:
        return ValueError(f"{path.name} is damaged or incomplete: {why}. "
                          f"Re-train or re-import the word vectors to write "
                          f"a fresh one.")

    for key in ("text", "vocabulary", "dim", "dim_labels"):
        if key not in doc:
            raise _broken(f"it lacks {key!r}")
    text = doc["text"]
    for key in ("lemmatize", "engine", "tokenizer", "stanza_lang",
                "keep_punctuation", "lowercase"):
        if key not in text:
            raise _broken(f"the text settings lack {key!r}")
    vocabulary = [str(w) for w in doc["vocabulary"]]
    dim = int(doc["dim"])
    if len(doc["dim_labels"]) != dim:
        raise _broken(f"it names {len(doc['dim_labels'])} dimensions for a "
                      f"{dim}-dimensional model")
    weights = _payload_path(path, doc)
    recorded = (doc.get("payload_digests") or {}).get(weights.name) \
        or (doc.get("payload_digests") or {}).get(str(doc["payload"][0]))
    if recorded and file_digest(weights) != recorded:
        raise ValueError(
            f"{weights.name} is not the matrix {path.name} was written with "
            f"(its contents differ from the recorded digest). Restore the "
            f"pair from the run that wrote them, or re-train.")
    try:
        vectors = np.load(weights, mmap_mode="r")
    except (OSError, ValueError) as e:
        raise _broken(f"its matrix could not be read ({e})") from None
    if vectors.ndim != 2 or vectors.shape != (len(vocabulary), dim):
        raise _broken(
            f"its matrix is {'x'.join(map(str, vectors.shape))} but the "
            f"manifest describes {len(vocabulary)} words x {dim} dimensions")
    counts = doc.get("counts")
    if counts is not None:
        if len(counts) != len(vocabulary):
            raise _broken("its word counts do not match its vocabulary")
        counts = np.asarray(counts, dtype=np.float64)
    index = {w: i for i, w in enumerate(vocabulary)}
    if len(index) != len(vocabulary):
        raise _broken("its vocabulary lists a word twice")
    return _Loaded(path=path, doc=doc, vectors=vectors, index=index, counts=counts)


def _stream_for(text: dict, device: str) -> Callable[[str], List[str]]:
    """The tokenizer the model was built with, rebuilt from its record."""
    return make_token_stream(
        bool(text["lemmatize"]), False, engine=str(text["engine"]),
        tokenizer=str(text["tokenizer"]), stanza_lang=str(text["stanza_lang"]),
        device=device, keep_punctuation=bool(text["keep_punctuation"]))


def _tokens_of(stream, lowercase: bool, text: str) -> List[str]:
    """One text's tokens as the model sees them -- the single tokenizing
    path fit, apply and the concept seeds all run."""
    tokens = stream(text)
    return [t.lower() for t in tokens] if lowercase else list(tokens)


#: Per-process state for the tokenizing workers (spawn cannot pickle the
#: stream's closures, so each worker builds its own once).
_TOKEN_WORKER: Dict[str, object] = {}


def _init_token_worker(stream_args: dict, lowercase: bool) -> None:
    _TOKEN_WORKER["stream"] = make_token_stream(**stream_args)
    _TOKEN_WORKER["lowercase"] = lowercase


def _tokens_in_worker(pair):
    _tid, text = pair
    return _tokens_of(_TOKEN_WORKER["stream"], bool(_TOKEN_WORKER["lowercase"]), text)


def _stream_args(text: dict, device: str) -> dict:
    return dict(lemmatize=bool(text["lemmatize"]), pos_tagged=False,
                engine=str(text["engine"]), tokenizer=str(text["tokenizer"]),
                stanza_lang=str(text["stanza_lang"]), device=device,
                keep_punctuation=bool(text["keep_punctuation"]))


# ---------------------------------------------------------------------------
# Scoring one text
# ---------------------------------------------------------------------------

def _weights_for(model: _Loaded, rows: Sequence[int], weighting: str):
    """One weight per in-vocabulary token (or type), by the weighting rule."""
    import numpy as np

    if weighting == "types":
        return np.ones(len(rows), dtype=np.float64)
    if weighting == "sif":
        if model.counts is None:
            raise ValueError(
                f"{model.path.name} has no word counts (imported vectors "
                f"carry none), so 'sif' weighting cannot be used with it. "
                f"Use 'tokens' or 'types'.")
        total = model.total_count() or 1.0
        p = model.counts[list(rows)] / total
        return SIF_A / (SIF_A + p)
    return np.ones(len(rows), dtype=np.float64)


def _text_vector(model: _Loaded, tokens: Sequence[str], weighting: str,
                 normalize_words: bool):
    """The weighted mean vector of a text's in-vocabulary words, and how
    many words that was. ``None`` when nothing was in the vocabulary."""
    import numpy as np

    rows = [model.index[t] for t in tokens if t in model.index]
    if not rows:
        return None, 0
    in_vocab = len(rows)
    if weighting == "types":
        rows = sorted(set(rows))
    block = np.asarray(model.vectors[rows], dtype=np.float64)
    if normalize_words:
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        block = block / norms
    w = _weights_for(model, rows, weighting)
    return (w[:, None] * block).sum(axis=0) / w.sum(), in_vocab


def _cosine(a, b) -> Optional[float]:
    import numpy as np

    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _feature_row(model: _Loaded, tokens: Sequence[str], *, weighting: str,
                 normalize_words: bool, concept_vectors: Sequence,
                 rounding: int) -> list:
    """
    ``[token_count, in_vocab_count, wv_1..wv_k, sim_<concept>...]`` for one
    text -- the one row-maker fit and apply share, so applying a model to
    its training texts reproduces the training features exactly. A text
    with no word in the vocabulary gets blank cells, not zeros: a zero
    vector would sit at the center of every concept.
    """
    vector, in_vocab = _text_vector(model, tokens, weighting, normalize_words)
    cells: list = [len(tokens), in_vocab]
    if vector is None:
        return cells + [None] * (model.dim + len(concept_vectors))
    cells += [round(float(v), rounding) for v in vector]
    for cv in concept_vectors:
        sim = _cosine(vector, cv)
        cells.append(None if sim is None else round(sim, rounding))
    return cells


def _concept_vectors(model: _Loaded, dicts, stream, lowercase: bool):
    """
    The concept columns, their vectors, and the report rows saying how each
    category met the vocabulary -- see :func:`resolve_concepts`. A category
    no word of which the model knows is refused there, by name.
    """
    return resolve_concepts(model.index, model.vectors, list(dicts), stream, lowercase)


def _header(model: _Loaded, concept_columns: Sequence[str]) -> List[str]:
    return ["text_id", "token_count", "in_vocab_count", *model.dim_labels,
            *concept_columns]


# ---------------------------------------------------------------------------
# Nearest neighbors
# ---------------------------------------------------------------------------

def _row_norms(model: _Loaded):
    """The norm of every row, computed in chunks so the matrix is never
    copied whole."""
    import numpy as np

    n = len(model.index)
    norms = np.empty(n, dtype=np.float64)
    for start in range(0, n, _CHUNK_ROWS):
        block = np.asarray(model.vectors[start:start + _CHUNK_ROWS], dtype=np.float64)
        norms[start:start + block.shape[0]] = np.linalg.norm(block, axis=1)
    norms[norms == 0] = 1.0
    return norms


def nearest_neighbors(model_json: PathLike, words: Sequence[str], *,
                       top_n: int = 20, device: str = "auto"
                       ) -> Dict[str, List[Tuple[str, float]]]:
    """
    The ``top_n`` most similar vocabulary words to each word, by cosine.

    A word is looked up as the model tokenizes it (lower-cased, lemmatized
    if the model was); a word not in the vocabulary maps to an empty list.
    The probe itself is never among its own neighbors. Computed in row
    chunks, so a large model is searched without a second copy in memory.
    """
    model = _load_model(one_model_path(model_json))
    stream = _stream_for(model.text_settings, device)
    return _neighbors(model, words, stream, top_n=top_n)


def _neighbors(model: _Loaded, words: Sequence[str], stream, *, top_n: int,
                norms=None) -> Dict[str, List[Tuple[str, float]]]:
    import numpy as np

    lowercase = bool(model.text_settings.get("lowercase", True))
    vocabulary = list(model.index)
    if norms is None:
        norms = _row_norms(model)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for word in words:
        forms = _tokens_of(stream, lowercase, word) or [word]
        hit = next((model.index[f] for f in forms if f in model.index), None)
        if hit is None:
            out[word] = []
            continue
        q = np.asarray(model.vectors[hit], dtype=np.float64)
        qn = float(np.linalg.norm(q)) or 1.0
        sims = np.empty(len(vocabulary), dtype=np.float64)
        for start in range(0, len(vocabulary), _CHUNK_ROWS):
            block = np.asarray(model.vectors[start:start + _CHUNK_ROWS], dtype=np.float64)
            sims[start:start + block.shape[0]] = block @ q
        sims /= norms * qn
        sims[hit] = -np.inf
        k = min(int(top_n), len(vocabulary) - 1)
        if k <= 0:
            out[word] = []
            continue
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out[word] = [(vocabulary[i], float(sims[i])) for i in top]
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _write_manifest(path: Path, weights: Path, *, name: str, text: dict,
                    vocabulary: Sequence[str], counts: Optional[Sequence[int]],
                    dim: int, training: dict, concept_dicts, weighting: str,
                    normalize_words: bool) -> None:
    from ..helpers.provenance import file_digest

    doc = {
        "kind": MODEL_KIND, "format": MODEL_FORMAT, "name": name,
        "payload": [weights.name],
        "payload_digests": {weights.name: file_digest(weights)},
        "text": dict(text),
        "vocabulary": list(vocabulary),
        "counts": None if counts is None else [int(c) for c in counts],
        "dim": int(dim),
        "dim_labels": [f"wv_{i + 1}" for i in range(int(dim))],
        "training": training,
        # how the model gets applied to new data. people can edit these per
        # model in Settings, and the applier reads them unless a call overrides
        # them. the dictionaries travel *inside* the model (terms and weights),
        # so a model applied on another machine measures the same concepts
        "apply": {"weighting": weighting, "normalize_words": bool(normalize_words),
                  "concept_dicts": concept_dicts_to_json(concept_dicts)},
    }
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


def _settings_of(model: _Loaded, weighting, normalize_words, concept_dicts):
    """The apply settings: the call's when given, else the model's own. A
    call may hand over dictionary *files*; the model holds their content."""
    from ..helpers.model_spec import apply_defaults

    defaults = apply_defaults(model.doc)
    weighting = weighting if weighting is not None else defaults.get("weighting", "tokens")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {', '.join(WEIGHTINGS)}, "
                         f"not {weighting!r}")
    normalize_words = (bool(normalize_words) if normalize_words is not None
                       else bool(defaults.get("normalize_words", False)))
    if concept_dicts is not None:
        dicts = load_concept_dicts(list(concept_dicts))
    else:
        dicts = concept_dicts_from_json(defaults.get("concept_dicts") or [])
    return weighting, normalize_words, dicts


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _gensim_missing() -> str:
    import importlib.util

    if importlib.util.find_spec("gensim") is None:
        return ("Training word vectors needs gensim, which is not installed. "
                "Install it with:  pip install \"taters[vectors]\"")
    return ""


#: gensim reports within-epoch progress only as a log line, once a second:
#: ``EPOCH 2 - PROGRESS: at 37.50% examples, 81234 words/s, ...``. The epoch
#: in it is zero-based.
_PROGRESS_LINE = re.compile(r"EPOCH (\d+) - PROGRESS: at ([\d.]+)%")


class _TrainingProgress:
    """
    Turns gensim's training into progress reports the runner can show.

    Two sources, because neither is enough alone. gensim's callbacks fire at
    the start and end of every epoch -- deterministic, so the display always
    knows which epoch it is on -- but say nothing in between, and an epoch
    over a large corpus is minutes of silence. Within an epoch gensim reports
    only through its logger, once a second, at INFO, which the root logger
    drops by default; so for the duration of training a handler is attached
    to that logger, its level lowered, and both put back afterwards, the way
    :mod:`~taters.helpers.doc_text` borrows pypdf's logger.

    Progress is reported as ``(done, total, message)`` with ``total`` the
    epochs times a hundred, so a percentage inside epoch 3 of 6 lands where
    it should on one bar.
    """

    def __init__(self, on_progress, epochs: int, family: str):
        import logging

        from gensim.models.callbacks import CallbackAny2Vec

        self._report = on_progress
        self._epochs = int(epochs)
        self._family = family
        self._epoch = 0                       # the one-based epoch under way
        outer = self

        class _Cb(CallbackAny2Vec):
            def on_epoch_begin(self, model):
                outer._epoch += 1
                outer._say(0.0)

            def on_epoch_end(self, model):
                outer._say(100.0)

        class _Handler(logging.Handler):
            def emit(self, record):
                try:
                    m = _PROGRESS_LINE.search(record.getMessage())
                except Exception:          # pragma: no cover - a malformed record
                    return
                if m:
                    # gensim counts epochs from zero; the callback above is the
                    # authority on which epoch we are in, but a line that names
                    # one is taken at its word
                    outer._epoch = int(m.group(1)) + 1
                    outer._say(min(100.0, float(m.group(2))))

        self.callback = _Cb()
        self._handler = _Handler()
        self._logger = logging.getLogger("gensim.models.word2vec")
        self._previous_level = self._logger.level

    def _say(self, percent: float) -> None:
        if self._report is None or self._epoch < 1:
            return
        done = (self._epoch - 1) * 100 + percent
        self._report(int(round(done)), self._epochs * 100,
                     f"training {self._family}: epoch {self._epoch}/{self._epochs}, "
                     f"{percent:.0f}%")

    def __enter__(self):
        import logging

        self._logger.addHandler(self._handler)
        if self._logger.getEffectiveLevel() > logging.INFO:
            self._logger.setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._previous_level)
        return False


class _EpochLoss:
    """Collects gensim's running loss after each epoch as a per-epoch list."""

    def __init__(self):
        from gensim.models.callbacks import CallbackAny2Vec

        outer = self

        class _Cb(CallbackAny2Vec):
            def __init__(self):
                self.previous = 0.0

            def on_epoch_end(self, model):
                total = float(model.get_latest_training_loss())
                outer.losses.append(total - self.previous)
                self.previous = total

        self.losses: List[float] = []
        self.callback = _Cb()


@records_settings(
    binding=TEXT_INPUT, grain=TEXT_GRAIN,
    # the concept dictionaries are word lists: their *content* is what the
    # features depend on, so it gets hashed and carried like dict_paths is
    assets={"concept_dicts": "dictionaries"},
    outputs=("out_features_csv", "out_model_json", "out_neighbors_csv",
             "out_report_md"),
    # the vectors are fitted to this corpus, so the honest way to measure them
    # on another is to apply the saved model. retraining on the new texts
    # would give us different dimensions, and a ridge fitted on these ones
    # couldn't be scored at all
    replay=(f"{__name__}:apply_word_vectors", {"model_json": "out_model_json"}),
    bookkeeping=("token_count", "in_vocab_count"))
def train_word_vectors(
    *,
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_features_csv: Optional[PathLike] = None,
    out_model_json: Optional[PathLike] = None,
    out_neighbors_csv: Optional[PathLike] = None,
    out_report_md: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    workers: int = 0,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,
    device: str = "auto",

    # ----- how we read the text (we record these into the model) -----
    lemmatize: bool = False,
    keep_punctuation: bool = False,
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",

    # ----- the model -----
    name: Optional[str] = None,
    family: Literal["word2vec", "fasttext"] = "word2vec",
    algorithm: Literal["skipgram", "cbow"] = "skipgram",
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    negative: int = 5,
    seed: int = 42,
    reproducible: bool = False,

    # ----- how we apply it, to this corpus now and to others later -----
    concept_dicts: Sequence[PathLike] = (),
    weighting: Literal["tokens", "types", "sif"] = "tokens",
    normalize_words: bool = False,
    probes: str = "",
    top_neighbors: int = 20,
    rounding: int = 4,
) -> Path:
    """
    Train word vectors on the texts, save the model, and write its features.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as every other text analyzer: a spreadsheet
        of texts, a folder of documents, or a prebuilt analysis-ready CSV.
    out_features_csv : str or Path, optional
        The features -- one row per text -- default
        ``./features/word_vectors.csv``.
    overwrite_existing : bool, default False
        If False and the output files exist, return them untouched.
    workers : int, default 0
        Parallel processes for reading and tokenizing texts. 0 means
        automatic: three-quarters of the logical cores.
    text_cols : sequence of str, default ("text",)
        When gathering from a CSV, the column(s) holding the text.
    id_cols : sequence of str, optional
        Columns that identify each row when gathering from a CSV.
    mode : {"concat", "separate"}, default "concat"
        With several text columns: join them into one text per row, or
        treat each as its own text.
    group_by : sequence of str, optional
        Columns to combine rows by before analyzing (one text per group).
    pattern : str, default every document type
        Which files to read when gathering from a folder of documents.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where Stanza runs, if the stanza engine is used -- a runtime choice,
        deliberately not stored in the model.
    out_model_json : str or Path, optional
        The model manifest; its matrix lands beside it as ``<stem>.npy``.
        Default ``<features folder>/models/word_vectors.json``.
    out_neighbors_csv, out_report_md : str or Path, optional
        The nearest-neighbors table and the training report, default
        beside the model as ``<stem>_neighbors.csv`` and ``<stem>_report.md``.
    lemmatize, keep_punctuation, engine, tokenizer, stanza_lang
        How text becomes words, recorded in the model so new texts are read
        the same way. Text is always lower-cased (every tokenizer here does
        it): *Death* and *death* are one word to a study of meaning, and a
        vocabulary that kept both would spend its counts twice.
    name : str, optional
        The model's name in menus; default the manifest's file stem.
    family : {"word2vec", "fasttext"}
        word2vec learns a vector per word; fastText also learns from
        character n-grams during training, which helps with rare and
        misspelt words, though only whole-word vectors are kept here.
    algorithm : {"skipgram", "cbow"}
        Skip-gram predicts context from a word and does better on small
        corpora and rare words; CBOW is faster and slightly better on very
        large ones.
    vector_size, window, min_count, epochs, negative
        The usual: dimensions; context words either side; the fewest
        occurrences a word needs to get a vector; passes over the corpus;
        negative samples per positive.
    seed, reproducible
        The random seed. Training runs in parallel threads and is then
        reproducible only in distribution; ``reproducible=True`` trains on
        one thread so the same corpus and seed give the same vectors (slower).
    concept_dicts : sequence of str or Path
        LIWC-22 dictionaries (``.dic``, ``.dicx``, ``.csv``; a folder means
        every dictionary in it). Every category becomes a
        ``sim_<dictionary>__<category>`` column: the cosine between a text's
        vector and the category's vector, the weighted mean of its terms'
        vectors (cells are weights, ``X`` = 1; wildcards and phrases as LIWC
        reads them). The dictionaries' terms and weights are stored in the
        model, and can be changed later in Settings.
    weighting : {"tokens", "types", "sif"}
        How words are averaged into a text's vector: every occurrence
        (``tokens``), each distinct word once (``types``), or smooth
        inverse frequency (``sif``: the commonest words count least).
    normalize_words : bool
        Scale every word vector to unit length before averaging, so a
        frequent word with a long vector does not dominate.
    probes : str
        Comma-separated words whose nearest neighbors the report shows;
        every concept category's neighbors are shown as well. Empty:
        :data:`DEFAULT_PROBES`, a dozen common content words, skipping any
        the model did not learn. Nothing about training depends on this; it
        only decides which words the clouds are about.
    top_neighbors : int
        Neighbors per probe in the table and the clouds.
    rounding : int
        Decimal places written.

    Returns
    -------
    Path
        ``out_features_csv``.

    Notes
    -----
    Nothing survives below ``min_count`` on a tiny corpus, and the refusal
    says so before gensim is asked. The report beside the model has the
    methods paragraph, the settings, the corpus coverage, the loss per
    epoch, the neighbors, and the package versions -- what a paper needs.
    """
    import numpy as np

    started = time.time()
    missing = _gensim_missing()
    if missing:
        raise ImportError(missing)
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(FAMILIES)}, not {family!r}")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}, "
                         f"not {algorithm!r}")
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {', '.join(WEIGHTINGS)}, "
                         f"not {weighting!r}")
    if int(vector_size) < 2 or int(min_count) < 1 or int(epochs) < 1:
        raise ValueError("vector_size must be at least 2, min_count and epochs "
                         "at least 1")
    dicts = load_concept_dicts(list(concept_dicts))

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers)

    out_features_csv = Path(out_features_csv) if out_features_csv else \
        Path.cwd() / "features" / "word_vectors.csv"
    out_model_json = Path(out_model_json) if out_model_json else \
        out_features_csv.parent / "models" / "word_vectors.json"
    out_neighbors_csv = Path(out_neighbors_csv) if out_neighbors_csv else \
        out_model_json.with_name(out_model_json.stem + "_neighbors.csv")
    out_report_md = Path(out_report_md) if out_report_md else \
        out_model_json.with_name(out_model_json.stem + "_report.md")
    if not overwrite_existing and out_features_csv.is_file() and out_model_json.is_file():
        if verbose:
            print("Word-vector features already exist; returning existing file.")
        return out_features_csv
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    out_model_json.parent.mkdir(parents=True, exist_ok=True)

    lowercase = True
    text_settings = {"lemmatize": bool(lemmatize), "pos_tagged": False,
                     "engine": engine, "tokenizer": tokenizer,
                     "stanza_lang": stanza_lang,
                     "keep_punctuation": bool(keep_punctuation),
                     "lowercase": lowercase}

    # 1) first, we tokenize every text once, out to a scratch file that gensim
    #    can stream from. only the type counts stay in memory
    if engine == "stanza":
        announce(on_progress, "loading the stanza pipeline (first use downloads its model)")
    stream = _stream_for(text_settings, device)
    tokens_path = out_model_json.with_name(f".{out_model_json.stem}_tokens.txt")
    ids: List[str] = []
    type_counts: Counter = Counter()
    n_tokens = 0
    from ..helpers.row_map import map_text_rows
    from .ngram_prep import pooled_text_workers

    with atomic_write(tokens_path, mode="w", encoding="utf-8") as fh:
        for row, tokens in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pooled_text_workers(
                    workers, n_rows, engine=engine, tokenizer=tokenizer,
                    lemmatize=bool(lemmatize), pos_tagged=False),
                message="tokenizing texts", on_progress=on_progress,
                inline_fn=lambda pair: _tokens_of(stream, bool(lowercase), pair[1]),
                pool_fn=_tokens_in_worker, initializer=_init_token_worker,
                initargs=(_stream_args(text_settings, device), bool(lowercase))):
            ids.append(str(row.get("text_id", "")))
            clean = [t.replace(" ", "_") for t in tokens if t.strip()]
            type_counts.update(clean)
            n_tokens += len(clean)
            fh.write(" ".join(clean) + "\n")

    surviving = sum(1 for c in type_counts.values() if c >= int(min_count))
    if surviving < 2:
        tokens_path.unlink(missing_ok=True)
        raise ValueError(
            f"only {surviving} word(s) occur at least {min_count} times in "
            f"these {len(ids)} text(s) ({n_tokens} tokens, {len(type_counts)} "
            f"distinct), which is not enough to train on. Lower min_count "
            f"or use a larger corpus.")

    # 2) now we train. gensim streams the scratch file, so what sits in memory
    #    is the vocabulary and the matrix, never the corpus itself
    import gensim

    announce(on_progress, f"training {family} ({algorithm}) for {epochs} epoch(s)")
    threads = 1 if reproducible else max(1, (os.cpu_count() or 2) - 1)
    loss = _EpochLoss()
    progress = _TrainingProgress(on_progress, int(epochs), family)
    common = dict(vector_size=int(vector_size), window=int(window),
                  min_count=int(min_count), sg=1 if algorithm == "skipgram" else 0,
                  negative=int(negative), epochs=int(epochs), seed=int(seed),
                  workers=threads)
    if family == "fasttext":
        # gensim's fastText doesn't keep a running loss, so there's no curve
        # for us to report. the report says so rather than showing zeros
        from gensim.models import FastText

        with progress:
            trained = FastText(corpus_file=str(tokens_path),
                               callbacks=[progress.callback], **common)
    else:
        from gensim.models import Word2Vec

        with progress:
            trained = Word2Vec(corpus_file=str(tokens_path), compute_loss=True,
                               callbacks=[loss.callback, progress.callback], **common)
    kv = trained.wv
    vocabulary = list(kv.index_to_key)
    counts = [int(kv.get_vecattr(w, "count")) for w in vocabulary]
    matrix = np.asarray(kv.vectors, dtype=np.float32)
    losses = [float(x) for x in loss.losses if math.isfinite(x)]
    if not any(losses):
        losses = []          # no loss from fastText; we say so instead of zeros

    # 3) save: the matrix goes beside the manifest, and the manifest names it
    weights = out_model_json.with_suffix(".npy")
    with atomic_write(weights, mode="wb") as fh:
        np.save(fh, matrix)
    covered = sum(c for w, c in type_counts.items() if w in kv.key_to_index)
    oov = [(w, c) for w, c in type_counts.most_common() if w not in kv.key_to_index][:15]
    training = {
        "source": "trained", "family": family, "algorithm": algorithm,
        "vector_size": int(vector_size), "window": int(window),
        "min_count": int(min_count), "epochs": int(epochs),
        "negative": int(negative), "seed": int(seed),
        "reproducible": bool(reproducible), "threads": threads,
        "n_documents": len(ids), "n_tokens": n_tokens,
        "n_types": len(type_counts), "vocabulary_size": len(vocabulary),
        "coverage": (covered / n_tokens) if n_tokens else 0.0,
        "top_oov": oov, "loss_per_epoch": losses,
        "gensim_version": gensim.__version__,
        "wall_seconds": None,     # we fill this in after the report
    }
    model_name = str(name or out_model_json.stem)
    _write_manifest(out_model_json, weights, name=model_name, text=text_settings,
                    vocabulary=vocabulary, counts=counts, dim=int(vector_size),
                    training=training, concept_dicts=dicts,
                    weighting=weighting, normalize_words=normalize_words)
    model = _load_model(out_model_json)

    # 4) features for the training corpus. these go through the very same
    #    row-maker that apply uses, and from the same tokens
    concept_columns, concept_vecs, concept_rows = _concept_vectors(
        model, dicts, stream, bool(lowercase))
    announce(on_progress, "writing the features")
    with atomic_write(out_features_csv, mode="w", newline="", encoding=encoding) as out, \
            tokens_path.open("r", encoding="utf-8") as fh:
        writer = csv.writer(out)
        writer.writerow(_header(model, concept_columns))
        for text_id, line in zip(ids, fh):
            tokens = line.rstrip("\n").split(" ") if line.strip() else []
            writer.writerow([text_id, *_feature_row(
                model, tokens, weighting=weighting,
                normalize_words=normalize_words, concept_vectors=concept_vecs,
                rounding=rounding)])
    tokens_path.unlink(missing_ok=True)

    # 5) lastly, the neighbors (of the probe words, and of every concept
    #    category) and the report
    probe_words = _probe_words(probes, vocabulary, counts)
    neighbors = _neighbors(model, probe_words, stream, top_n=int(top_neighbors))
    neighbors.update(_neighbors_of_vectors(model, concept_columns, concept_vecs,
                                             top_n=int(top_neighbors)))
    _write_neighbors(out_neighbors_csv, neighbors, encoding=encoding,
                      rounding=rounding)
    training["wall_seconds"] = round(time.time() - started, 1)
    doc = json.loads(out_model_json.read_text(encoding="utf-8"))
    doc["training"] = training
    with atomic_write(out_model_json, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    _write_report(out_report_md, doc, out_model_json, weights, neighbors,
                  concept_rows, features_csv=out_features_csv,
                  neighbors_csv=out_neighbors_csv, model=model)
    if verbose:
        print(f"[word_vectors] {len(vocabulary)} words x {vector_size} dims "
              f"from {len(ids)} text(s); coverage "
              f"{100 * training['coverage']:.1f}% -> {out_model_json}")
        for row in concept_rows:
            if row["missed"]:
                print(f"[word_vectors] note: {row['column']}: {len(row['missed'])} of "
                      f"{row['n_terms']} term(s) not in the vocabulary")
    return out_features_csv


def _probe_words(probes: str, vocabulary: Sequence[str],
                 counts: Optional[Sequence[int]] = None) -> List[str]:
    """
    The words whose neighbors the report shows: those asked for, or -- with
    none asked -- :data:`DEFAULT_PROBES`, in that order, skipping any the
    model never learned.

    An earlier version took the corpus's most frequent words, and every
    cloud was function words. A fixed list needs no explaining: it is on the
    options screen, and it is the whole rule. ``counts`` is accepted for the
    callers that have it and no longer used.
    """
    out: List[str] = []
    for w in [p.strip() for p in str(probes or "").split(",")]:
        if w and w not in out:
            out.append(w)
    if out:
        return out
    known = set(str(w) for w in vocabulary)
    return [w for w in DEFAULT_PROBES if w in known]


def _neighbors_of_vectors(model: _Loaded, names: Sequence[str], vectors,
                           *, top_n: int) -> Dict[str, List[Tuple[str, float]]]:
    """The words closest to each concept's vector -- a concept is a point
    in the space like any word, and its neighbors are the plainest check
    that the dictionary's category means what its name says."""
    import numpy as np

    vocabulary = list(model.index)
    norms = _row_norms(model)
    out: Dict[str, List[Tuple[str, float]]] = {}
    for name, q in zip(names, vectors):
        q = np.asarray(q, dtype=np.float64)
        qn = float(np.linalg.norm(q)) or 1.0
        sims = np.empty(len(vocabulary), dtype=np.float64)
        for start in range(0, len(vocabulary), _CHUNK_ROWS):
            block = np.asarray(model.vectors[start:start + _CHUNK_ROWS], dtype=np.float64)
            sims[start:start + block.shape[0]] = block @ q
        sims /= norms * qn
        k = min(int(top_n), len(vocabulary))
        if k <= 0:
            out[name] = []
            continue
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        out[name] = [(vocabulary[i], float(sims[i])) for i in top]
    return out


def _write_neighbors(path: Path, neighbors: Dict[str, List[Tuple[str, float]]],
                      *, encoding: str, rounding: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", newline="", encoding=encoding) as fh:
        writer = csv.writer(fh)
        writer.writerow(["probe", "rank", "word", "similarity"])
        for probe, found in neighbors.items():
            if not found:
                writer.writerow([probe, "", "", ""])
            for rank, (word, sim) in enumerate(found, start=1):
                writer.writerow([probe, rank, word, round(sim, rounding)])


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

@records_settings(
    binding=TEXT_INPUT, grain=TEXT_GRAIN, assets={"model_json": None},
    outputs=("out_features_csv",), bookkeeping=("token_count", "in_vocab_count"))
def apply_word_vectors(
    *,
    model_json: PathLike,
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    workers: int = 0,
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
    encoding: str = "utf-8-sig",
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,
    device: str = "auto",
    rounding: int = 4,
    weighting: Optional[Literal["tokens", "types", "sif"]] = None,
    normalize_words: Optional[bool] = None,
    concept_dicts: Optional[Sequence[PathLike]] = None,
) -> Path:
    """
    Score new texts with a saved word-vector model.

    The model's own settings (its ``apply`` block: weighting, word
    normalization, concepts) are used unless the call gives its own, so a
    pipeline that applies a model gets the settings the model was saved
    with -- and changed in Settings afterwards -- without knowing them.

    Parameters
    ----------
    model_json
        A word-vector manifest written by :func:`train_word_vectors` or
        :func:`import_word_vectors`; its matrix is read from beside it, or
        from the library when the manifest traveled alone.
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as the other text analyzers.
    out_features_csv : str or Path, optional
        Default ``./features/word_vectors_applied.csv``.
    overwrite_existing : bool, default False
        If False and the output files exist, return them untouched.
    workers : int, default 0
        Parallel processes for reading and tokenizing texts. 0 means
        automatic: three-quarters of the logical cores.
    text_cols : sequence of str, default ("text",)
        When gathering from a CSV, the column(s) holding the text.
    id_cols : sequence of str, optional
        Columns that identify each row when gathering from a CSV.
    mode : {"concat", "separate"}, default "concat"
        With several text columns: join them into one text per row, or
        treat each as its own text.
    group_by : sequence of str, optional
        Columns to combine rows by before analyzing (one text per group).
    pattern : str, default every document type
        Which files to read when gathering from a folder of documents.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where Stanza runs, if the stanza engine is used -- a runtime choice,
        deliberately not stored in the model.
    rounding : int, default 4
        Decimal places written.
    weighting, normalize_words, concept_dicts
        Overrides for the model's own apply settings; see
        :func:`train_word_vectors`. ``concept_dicts`` are dictionary files;
        left out, the dictionaries stored in the model apply.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id, token_count, in_vocab_count,
        wv_1..wv_k, sim_<dictionary>__<category>...``.
    """
    model = _load_model(one_model_path(model_json))
    weighting, normalize_words, dicts = _settings_of(
        model, weighting, normalize_words, concept_dicts)
    text = model.text_settings
    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers)
    out_features_csv = Path(out_features_csv) if out_features_csv else \
        Path.cwd() / "features" / "word_vectors_applied.csv"
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        if verbose:
            print("Word-vector features already exist; returning existing file.")
        return out_features_csv

    if text["engine"] == "stanza":
        announce(on_progress, "loading the stanza pipeline (first use downloads its model)")
    stream = _stream_for(text, device)
    lowercase = bool(text.get("lowercase", True))
    concept_columns, concept_vecs, concept_rows = _concept_vectors(
        model, dicts, stream, lowercase)
    from ..helpers.row_map import map_text_rows
    from .ngram_prep import pooled_text_workers

    with atomic_write(out_features_csv, mode="w", newline="", encoding=encoding) as out:
        writer = csv.writer(out)
        writer.writerow(_header(model, concept_columns))
        for row, tokens in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pooled_text_workers(
                    workers, n_rows, engine=str(text["engine"]),
                    tokenizer=str(text["tokenizer"]),
                    lemmatize=bool(text["lemmatize"]), pos_tagged=False),
                message="scoring word vectors", on_progress=on_progress,
                inline_fn=lambda pair: _tokens_of(stream, lowercase, pair[1]),
                pool_fn=_tokens_in_worker, initializer=_init_token_worker,
                initargs=(_stream_args(text, device), lowercase)):
            clean = [t.replace(" ", "_") for t in tokens if t.strip()]
            writer.writerow([row.get("text_id", ""), *_feature_row(
                model, clean, weighting=weighting, normalize_words=normalize_words,
                concept_vectors=concept_vecs, rounding=rounding)])
    if verbose:
        for row in concept_rows:
            if row["missed"]:
                print(f"[word_vectors] note: {row['column']}: {len(row['missed'])} of "
                      f"{row['n_terms']} term(s) not in the vocabulary")
    return out_features_csv


# ---------------------------------------------------------------------------
# Import pre-trained vectors
# ---------------------------------------------------------------------------

def _sniff_format(path: Path) -> str:
    """glove / word2vec_text / word2vec_bin / fasttext_bin from the name and
    the first bytes."""
    name = path.name.lower()
    if name.endswith(".bin") and ("fasttext" in name or name.startswith("cc.")
                                  or name.startswith("wiki.")):
        return "fasttext_bin"
    with path.open("rb") as fh:
        head = fh.read(4096)
    if head[:4] == b"\xba\x16\x2d\x2f":          # fastText's magic number
        return "fasttext_bin"
    try:
        first = head.split(b"\n", 1)[0].decode("utf-8")
    except UnicodeDecodeError:
        return "word2vec_bin"
    parts = first.split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        # a `V k` header, followed by either text rows or binary blocks
        rest = head.split(b"\n", 1)[1] if b"\n" in head else b""
        try:
            rest.decode("utf-8")
            return "word2vec_text"
        except UnicodeDecodeError:
            return "word2vec_bin"
    if name.endswith(".bin"):
        return "word2vec_bin"
    return "glove"


def _read_text_vectors(path: Path, *, header: bool, max_vocab: int,
                       lowercase: bool, encoding: str):
    """
    GloVe / word2vec text rows -> (words, matrix). Streams the file; keeps
    at most ``max_vocab`` words (the first ones, which these files list by
    frequency), the first occurrence of a word when lower-casing merges two.
    """
    import numpy as np

    words: List[str] = []
    rows: List[List[float]] = []
    seen: Dict[str, int] = {}
    dim: Optional[int] = None
    with path.open("r", encoding=encoding, errors="replace") as fh:
        if header:
            fh.readline()
        for line in fh:
            parts = line.rstrip("\n").rsplit(" ", maxsplit=-1)
            if len(parts) < 3:
                continue
            # some GloVe releases have words with spaces in them, so we take
            # the last `dim` fields as the numbers and whatever's left as the
            # word
            if dim is None:
                dim = len(parts) - 1
            word = " ".join(parts[:len(parts) - dim]) if len(parts) - dim > 1 else parts[0]
            values = parts[len(parts) - dim:]
            if len(values) != dim:
                continue
            if lowercase:
                word = word.lower()
            if word in seen:
                continue
            try:
                rows.append([float(v) for v in values])
            except ValueError:
                continue
            seen[word] = len(words)
            words.append(word)
            if len(words) >= int(max_vocab):
                break
    if not words:
        raise ValueError(f"{path.name} holds no readable word vectors")
    return words, np.asarray(rows, dtype=np.float32)


def import_word_vectors(
    vectors_path: PathLike,
    out_model_json: PathLike,
    *,
    format: Literal["auto", "glove", "word2vec_text", "word2vec_bin",
                    "fasttext_bin"] = "auto",
    max_vocab: int = 200000,
    name: Optional[str] = None,
    concept_dicts: Sequence[PathLike] = (),
    weighting: Literal["tokens", "types"] = "tokens",
    normalize_words: bool = False,
    lemmatize: bool = False,
    keep_punctuation: bool = False,
    engine: Literal["nltk", "stanza"] = "nltk",
    tokenizer: Literal["potts", "stanza"] = "potts",
    stanza_lang: str = "en",
    probes: str = "",
    top_neighbors: int = 20,
    out_report_md: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    encoding: str = "utf-8",
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Bring pre-trained vectors (GloVe, word2vec, fastText) in as a model.

    Parameters
    ----------
    vectors_path
        The vectors file: GloVe text (``word v1 v2 ...``), word2vec text
        (a ``V k`` header line first) or binary, or a fastText ``.bin``.
        The binary formats need gensim (``pip install "taters[vectors]"``).
    out_model_json
        Where the manifest goes; the matrix lands beside it as ``.npy``.
    format
        ``auto`` sniffs the file; name it when the sniff is wrong.
    max_vocab
        The most words kept, from the top of the file (these files list
        words by frequency). 200,000 x 300 floats is 240 MB, the most an
        ordinary laptop should be asked to hold. Case is folded, keeping
        the first occurrence of a word, because every tokenizer here
        lower-cases text: a vocabulary with both *Cat* and *cat* would
        never see the first.
    lemmatize, keep_punctuation, engine, tokenizer, stanza_lang
        How texts will be tokenized when the model is applied. Off for
        lemmatizing by default: pre-trained vectors were learned on
        inflected words, and "ran" is in the vocabulary while "run" alone
        would miss it.
    name, concept_dicts, weighting, normalize_words, probes, top_neighbors
        As for :func:`train_word_vectors`. ``sif`` weighting is not offered:
        imported vectors carry no counts.

    Returns
    -------
    Path
        ``out_model_json``.
    """
    import numpy as np

    started = time.time()
    src = Path(vectors_path)
    if not src.is_file():
        raise FileNotFoundError(f"vectors file not found: {src}")
    out_model_json = Path(out_model_json)
    if out_model_json.is_file() and not overwrite_existing:
        if verbose:
            print("Word-vector model already exists; returning existing file.")
        return out_model_json
    if format not in VECTOR_FORMATS:
        raise ValueError(f"format must be one of {', '.join(VECTOR_FORMATS)}, "
                         f"not {format!r}")
    if weighting not in ("tokens", "types"):
        raise ValueError("imported vectors carry no word counts, so weighting "
                         "must be 'tokens' or 'types'")
    dicts = load_concept_dicts(list(concept_dicts))
    lowercase = True
    fmt = _sniff_format(src) if format == "auto" else format
    announce(on_progress, f"reading {src.name} as {fmt}")
    if fmt in ("glove", "word2vec_text"):
        words, matrix = _read_text_vectors(
            src, header=(fmt == "word2vec_text"), max_vocab=max_vocab,
            lowercase=lowercase, encoding=encoding)
    else:
        missing = _gensim_missing()
        if missing:
            raise ImportError(missing.replace("Training word vectors", "Reading binary vectors"))
        if fmt == "fasttext_bin":
            from gensim.models.fasttext import load_facebook_vectors

            kv = load_facebook_vectors(str(src))
        else:
            from gensim.models import KeyedVectors

            kv = KeyedVectors.load_word2vec_format(str(src), binary=True,
                                                   limit=int(max_vocab))
        words, keep = [], []
        seen = set()
        for i, w in enumerate(kv.index_to_key[:int(max_vocab)]):
            w = w.lower() if lowercase else w
            if w in seen:
                continue
            seen.add(w)
            words.append(w)
            keep.append(i)
        matrix = np.asarray(kv.vectors[keep], dtype=np.float32)
    dim = int(matrix.shape[1])
    out_model_json.parent.mkdir(parents=True, exist_ok=True)
    weights = out_model_json.with_suffix(".npy")
    with atomic_write(weights, mode="wb") as fh:
        np.save(fh, matrix)
    text_settings = {"lemmatize": bool(lemmatize), "pos_tagged": False,
                     "engine": engine, "tokenizer": tokenizer,
                     "stanza_lang": stanza_lang,
                     "keep_punctuation": bool(keep_punctuation),
                     "lowercase": bool(lowercase)}
    training = {"source": "imported", "format": fmt, "file": src.name,
                "file_size": src.stat().st_size, "vocabulary_size": len(words),
                "max_vocab": int(max_vocab), "wall_seconds": None}
    _write_manifest(out_model_json, weights, name=str(name or out_model_json.stem),
                    text=text_settings, vocabulary=words, counts=None, dim=dim,
                    training=training, concept_dicts=dicts,
                    weighting=weighting, normalize_words=normalize_words)
    model = _load_model(out_model_json)
    stream = _stream_for(text_settings, "cpu")
    # this refuses any category the vectors don't know about, before we save
    # anything as a model that somebody will try to use
    concept_columns, concept_vecs, concept_rows = _concept_vectors(
        model, dicts, stream, bool(lowercase))
    probe_words = _probe_words(probes, words, None)
    neighbors = _neighbors(model, probe_words, stream, top_n=int(top_neighbors))
    neighbors.update(_neighbors_of_vectors(model, concept_columns, concept_vecs,
                                             top_n=int(top_neighbors)))
    neighbors_csv = out_model_json.with_name(out_model_json.stem + "_neighbors.csv")
    _write_neighbors(neighbors_csv, neighbors, encoding="utf-8-sig", rounding=4)
    training["wall_seconds"] = round(time.time() - started, 1)
    doc = json.loads(out_model_json.read_text(encoding="utf-8"))
    doc["training"] = training
    with atomic_write(out_model_json, mode="w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    report = Path(out_report_md) if out_report_md else \
        out_model_json.with_name(out_model_json.stem + "_report.md")
    _write_report(report, doc, out_model_json, weights, neighbors, concept_rows,
                  features_csv=None, neighbors_csv=neighbors_csv, model=model)
    if verbose:
        print(f"[word_vectors] imported {len(words)} words x {dim} dims from "
              f"{src.name} -> {out_model_json}")
    return out_model_json


# ---------------------------------------------------------------------------
# Describe: neighbors table for any model
# ---------------------------------------------------------------------------

def describe_word_vectors(
    model_json: PathLike,
    out_neighbors_csv: Optional[PathLike] = None,
    *,
    probes: str = "",
    top_neighbors: int = 20,
    device: str = "auto",
    rounding: int = 4,
    encoding: str = "utf-8-sig",
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Write the nearest-neighbors table of a saved model.

    ``probes`` are comma-separated words; empty means the model's most
    frequent words. Every concept category the model carries is listed
    too, as the words closest to its vector. The table has one row
    per (probe, neighbor): ``probe, rank, word, similarity``; a probe not
    in the vocabulary gets one row with blanks, so its absence is visible.
    """
    path = one_model_path(model_json)
    model = _load_model(path)
    out = Path(out_neighbors_csv) if out_neighbors_csv else \
        path.with_name(path.stem + "_neighbors.csv")
    if out.is_file() and not overwrite_existing:
        if verbose:
            print("Neighbors table already exists; returning existing file.")
        return out
    announce(on_progress, "finding nearest neighbors")
    stream = _stream_for(model.text_settings, device)
    dicts = concept_dicts_from_json((model.doc.get("apply") or {}).get("concept_dicts") or [])
    words = _probe_words(probes, list(model.index),
                         model.counts if model.counts is not None else None)
    neighbors = _neighbors(model, words, stream, top_n=int(top_neighbors))
    if dicts:
        columns, vecs, _rows = _concept_vectors(model, dicts, stream,
                                                bool(model.text_settings.get("lowercase", True)))
        neighbors.update(_neighbors_of_vectors(model, columns, vecs,
                                                 top_n=int(top_neighbors)))
    _write_neighbors(out, neighbors, encoding=encoding, rounding=rounding)
    if verbose:
        print(f"[word_vectors] neighbors of {len(words)} word(s) -> {out}")
    return out


# ---------------------------------------------------------------------------
# The training report
# ---------------------------------------------------------------------------

def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("taters", "numpy", "gensim"):
        try:
            module = __import__(mod)
            out[mod] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            continue
    return out


def _md_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return lines


def _methods_paragraph(doc: dict, model: _Loaded) -> str:
    t = doc.get("training") or {}
    text = doc["text"]
    how = []
    how.append("lower-cased" if text.get("lowercase", True) else "case-preserving")
    how.append("lemmatized" if text.get("lemmatize") else "unlemmatized")
    how.append(f"{text.get('tokenizer')} tokenizer")
    if text.get("keep_punctuation"):
        how.append("punctuation kept")
    reading = ", ".join(how)
    apply = doc.get("apply") or {}
    dicts = concept_dicts_from_json(apply.get("concept_dicts") or [])
    n_cats = sum(len(d.categories) for d in dicts)
    tail = (f" Each text was represented by the {apply.get('weighting', 'tokens')}-"
            f"weighted mean of its word vectors"
            + (" (each word vector scaled to unit length first)"
               if apply.get("normalize_words") else "")
            + (f", and by its cosine similarity to the {n_cats} categor"
               f"{'ies' if n_cats != 1 else 'y'} of the dictionar"
               f"{'ies' if len(dicts) != 1 else 'y'} "
               f"{', '.join(d.name for d in dicts)} (each category's vector the "
               f"weighted mean of its terms' vectors, wildcards expanded against "
               f"the vocabulary)" if dicts else "") + ".")
    if t.get("source") == "imported":
        return (f"Pre-trained word vectors ({t.get('format')}, from "
                f"{t.get('file')}) were imported into Taters; the {t.get('vocabulary_size')} "
                f"most frequent words were kept ({doc['dim']} dimensions). Texts were "
                f"read {reading}; words outside the vocabulary were skipped."
                + tail)
    fam = "word2vec" if t.get("family") == "word2vec" else "fastText"
    alg = "skip-gram" if t.get("algorithm") == "skipgram" else "CBOW"
    return (f"{fam} vectors ({alg} with negative sampling, {t.get('negative')} "
            f"negatives; {doc['dim']} dimensions; window {t.get('window')}; "
            f"minimum count {t.get('min_count')}; {t.get('epochs')} epochs; "
            f"seed {t.get('seed')}) were trained with gensim "
            f"{t.get('gensim_version')} on {t.get('n_documents')} texts "
            f"({t.get('n_tokens')} tokens, {t.get('n_types')} distinct words; "
            f"texts read {reading}). The vocabulary of {t.get('vocabulary_size')} "
            f"words covered {100 * float(t.get('coverage') or 0):.1f}% of the "
            f"corpus tokens; words outside it were skipped."
            + (" Training ran on one thread, so the result is reproducible "
               "from the seed." if t.get("reproducible") else
               f" Training ran on {t.get('threads')} threads, so the result is "
               f"reproducible in distribution but not bit for bit.")
            + tail)


def _write_report(path: Path, doc: dict, manifest: Path, weights: Path,
                  neighbors: Dict[str, List[Tuple[str, float]]],
                  concept_rows: Sequence[dict], *, features_csv: Optional[Path],
                  neighbors_csv: Path, model: _Loaded) -> None:
    """
    ``<stem>_report.md``: what a paper needs to report the model.

    A methods paragraph ready to paste, the settings, the corpus coverage
    and the words that fell out of the vocabulary, the training loss per
    epoch (a plot when the chart renderer is available), the neighbors
    of every probe, the spread of vector norms, and the versions.
    """
    import numpy as np

    t = doc.get("training") or {}
    text = doc["text"]
    apply = doc.get("apply") or {}
    lines = [f"# Word vectors: {doc.get('name', manifest.stem)}", "",
             "## Methods paragraph", "", _methods_paragraph(doc, model), ""]
    lines += ["## Settings", ""]
    settings: List[Tuple[str, object]] = [
        ("model file", manifest.name), ("weights", f"{weights.name} "
                                                   f"({weights.stat().st_size / 1e6:.1f} MB)"),
        ("dimensions", doc["dim"]), ("vocabulary", len(doc["vocabulary"]))]
    if t.get("source") == "imported":
        settings += [("source", f"{t.get('file')} ({t.get('format')})"),
                     ("words kept (max)", t.get("max_vocab"))]
    else:
        settings += [(k, t.get(k)) for k in
                     ("family", "algorithm", "window", "min_count", "epochs",
                      "negative", "seed", "reproducible", "threads")]
    settings += [(f"text: {k}", v) for k, v in text.items() if k != "pos_tagged"]
    settings += [(f"apply: {k}", v) for k, v in apply.items()]
    lines += _md_table(["setting", "value"], settings) + [""]

    if t.get("source") != "imported":
        lines += ["## Corpus and coverage", "",
                  f"{t.get('n_documents')} texts, {t.get('n_tokens')} tokens, "
                  f"{t.get('n_types')} distinct words. {t.get('vocabulary_size')} "
                  f"words met the minimum count and got a vector, covering "
                  f"{100 * float(t.get('coverage') or 0):.1f}% of all tokens.", ""]
        oov = t.get("top_oov") or []
        if oov:
            lines += ["The most frequent words *without* a vector (below the "
                      "minimum count):", ""]
            lines += _md_table(["word", "count"], oov) + [""]
        losses = t.get("loss_per_epoch") or []
        lines += ["## Training loss", ""]
        if losses:
            lines += ["gensim's running training loss, per epoch (lower is "
                      "better; a curve that flattens has converged):", ""]
            lines += _md_table(["epoch", "loss"],
                               [(i + 1, f"{v:.1f}") for i, v in enumerate(losses)]) + [""]
            plot = _loss_plot(path.with_name(manifest.stem + "_loss.png"), losses)
            if plot is not None:
                lines += [f"![training loss]({plot.name})", ""]
        else:
            lines += ["gensim reports no training loss for this family "
                      "(fastText), so none is shown.", ""]
            # otherwise a plot left behind by an earlier word2vec run in this
            # folder would get read as this model's
            path.with_name(manifest.stem + "_loss.png").unlink(missing_ok=True)

    lines += ["## Nearest neighbors", "",
              f"The words closest to each probe by cosine similarity, from "
              f"`{neighbors_csv.name}`. These are the evidence that the model "
              f"learned what the study assumes it learned.", ""]
    for probe, found in neighbors.items():
        if not found:
            lines += [f"**{probe}** — not in the vocabulary.", ""]
            continue
        shown = ", ".join(f"{w} ({s:.2f})" for w, s in found[:10])
        lines += [f"**{probe}** — {shown}", ""]
    if concept_rows:
        lines += ["## Concepts", "",
                  "Each category of each concept dictionary, how many of its terms "
                  "the model knows (a wildcard counts as one term however many words "
                  "it matches), the weight behind the category's vector, and the "
                  "terms that missed. The nearest neighbors above, under the "
                  "column's name, are what the category means to this model.", ""]
        lines += _md_table(["column", "terms", "matched", "weight", "missed"],
                           [(r["column"], r["n_terms"], r["n_matched"],
                             f"{r['weight_sum']:g}",
                             ", ".join(r["missed"][:12]) + (" …" if len(r["missed"]) > 12 else ""))
                            for r in concept_rows]) + [""]

    norms = _row_norms(model)
    lines += ["## Vector norms", "",
              f"min {float(norms.min()):.3f}, median {float(np.median(norms)):.3f}, "
              f"max {float(norms.max()):.3f}. Frequent words tend to have longer "
              f"vectors; `normalize_words` evens that out before averaging.", ""]
    if features_csv is not None:
        lines += ["## Outputs", "", f"- `{features_csv.name}` — one row per text: "
                  f"`token_count`, `in_vocab_count`, `wv_1..wv_{doc['dim']}`"
                  + (", the `sim_*` columns" if apply.get("concept_dicts") else "") + ".",
                  f"- `{neighbors_csv.name}` — the neighbors table.", ""]
    lines += ["## Provenance", ""]
    prov = [("wall time (s)", t.get("wall_seconds")),
            ("machine", f"{platform.system()} {platform.machine()}, "
                        f"{os.cpu_count()} logical cores")]
    prov += list(_versions().items())
    lines += _md_table(["item", "value"], prov) + [""]
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def _loss_plot(path: Path, losses: Sequence[float]) -> Optional[Path]:
    """The per-epoch loss as a line chart, when the chart renderer and
    Pillow are available; None otherwise (the table stands on its own)."""
    try:
        from ..figures.charts import line_chart
    except ImportError:
        return None
    try:
        return line_chart(
            {"training loss": [(i + 1, float(v)) for i, v in enumerate(losses)]},
            path, title="Training loss per epoch", x_label="epoch", y_label="loss")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# command line -- we derive this from the functions above; see
# helpers.cliargs.CliSpec
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"train": train_word_vectors, "apply": apply_word_vectors,
     "import": import_word_vectors, "describe": describe_word_vectors},
    description="Word vectors: train word2vec/fastText on your texts, import "
                "pre-trained vectors, score texts with a saved model, or list "
                "a model's nearest neighbors.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
