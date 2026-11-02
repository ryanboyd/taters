"""
The exact share of every token in a prediction made over a feature set
that is a sum over its tokens.

A text's embedding is a mean of means: sentences of token vectors, averaged
per sentence (per window, for a sentence longer than the encoder's window),
then averaged per text, then perhaps scaled to unit length. A text's
word-vector row is simpler still: a weighted mean of its words' vectors. A
ridge over either -- with or without PCA in front -- is linear in it (see
`_fit_common.effective_weights`). So the prediction minus its constant is a
sum of one term per **unit** -- one position in one window for an encoder,
one token occurrence for word vectors -- and the terms add up to the
prediction exactly.

The contract that makes the three feature sets one case: a backend reads a
text into units, each carrying the vector it *added* to the text's row,
with every scalar of the backend's own pooling (sentence weights, window
means, per-sentence norms, the final L2, a word's a/Σa) folded in, so that
the row is exactly the sum of the units. :func:`attribute` then knows
nothing about encoders or vocabularies: the share of a unit is its vector
dotted with the model's effective weights, the shares are summed per word
and audited against the recomputed prediction.

It is exact only when the pooling really is a sum: the transformer step's
``pooling="mean"``, a sentence-transformers model whose modules are exactly
``Transformer -> Pooling(mean) [-> Normalize]``, any word-vector model. A
model that pools by ``cls`` puts the whole share on one position; ``max``
is not a sum at all; a ``Dense`` layer bends the space. Those are refused
with the reason, and the caller falls back to word correlates. A column of
the set that is not a sum over units (a cosine to a concept) keeps its term
in one ``[features]`` row per text, so the text still adds up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from ..helpers import tables

__all__ = ["Origin", "set_origin", "column_index", "merge_pieces", "SPECIALS", "FEATURES",
           "Unit", "Backend", "backend_for", "ATTRIBUTABLE_STEPS", "Attribution",
           "attribute"]

#: The row a text's special tokens ([CLS], [SEP], <s>, </s>) are summed into:
#: they carry a share of the mean like any position, so the sum of a text's
#: rows is its prediction; they are not words, so the word table skips them.
SPECIALS = "[specials]"
#: The row holding the term of the set's columns that are not sums over
#: units (a cosine between the text and a concept): read from the analysis
#: table rather than decomposed, so a text's rows still add up.
FEATURES = "[features]"

_EMBED_COL = re.compile(r"^e_?\d+$")


# ---------------------------------------------------------------------------
# Where the set came from
# ---------------------------------------------------------------------------

@dataclass
class Origin:
    """The step that made a feature set, as its provenance record, and the
    raw table whose columns name the dimensions. ``problem`` says why the
    walk failed, when it did."""

    call: str = ""
    record: Dict[str, Any] = field(default_factory=dict)
    raw_table: Optional[Any] = None
    root: Optional[Path] = None
    problem: str = ""

    @property
    def step(self) -> str:
        """The step's function name: what the backend registry is keyed by."""
        return self.call.rsplit(":", 1)[-1]

    @property
    def kind(self) -> str:
        cls = _BACKENDS.get(self.step)
        return cls.kind if cls else ""

    @property
    def instrument(self) -> Dict[str, Any]:
        return dict(self.record.get("instrument") or {})


def _resolve(address: Optional[str], root: Optional[Path]) -> Optional[Any]:
    """
    The table an address names, as a ref -- found again under ``root`` by
    its name when the recorded path does not resolve here.

    A sidecar written on Windows and read from WSL (or a run folder moved
    whole) records a path that does not exist on this side, and the table
    is right there under ``features/``. The same fallback the theme
    loadings use.
    """
    if not address:
        return None
    text = str(address).replace("\\", "/")
    try:
        ref = tables.parse_ref(text)
        if ref.is_db or ref.path.is_file():
            return ref
    except (TypeError, ValueError):
        return None
    if root is not None:
        name = text.rsplit("/", 1)[-1]
        for found in sorted(Path(root).rglob(name)):
            if found.is_file():
                return tables.parse_ref(found)
    return None


def _record(address: Optional[str], root: Optional[Path] = None) -> Optional[dict]:
    from ..helpers import provenance

    ref = _resolve(address, root)
    if ref is None:
        return None
    try:
        return provenance.read(ref)
    except Exception:
        return None


def set_origin(set_name: str, sources: Dict[str, str],
               root: Optional[Path] = None) -> Origin:
    """
    Walk from a feature set to the step that made it.

    The assemble sidecar names the set's source table; on a pipeline run
    that may be the merge step's output, whose record binds ``root_dir``
    to the raw table, whose record names the step and its settings. Each
    link that is missing is a sentence, not a guess. Which steps can be
    attributed is :data:`ATTRIBUTABLE_STEPS`.
    """
    address = sources.get(set_name)
    if not address:
        return Origin(problem=f"the analysis table does not say which table "
                              f"{set_name} came from")
    rec = _record(address, root)
    if rec is None:
        return Origin(problem=f"{set_name}'s source table has no readable "
                              f"provenance record")
    call = str(rec.get("call") or "")
    raw = address
    if "feature_gather" in call:
        raw = str((rec.get("binding") or {}).get("root_dir") or "")
        if not raw:
            return Origin(problem=f"the merge record of {set_name} does not "
                                  f"name the table it merged")
        rec = _record(raw, root)
        if rec is None:
            return Origin(problem=f"the raw table behind {set_name} ({raw}) has no "
                                  f"readable provenance record")
        call = str(rec.get("call") or "")
    origin = Origin(call=call, record=rec, raw_table=_resolve(raw, root), root=root)
    if origin.step not in _BACKENDS:
        return Origin(problem=f"{set_name} was not made by a step whose features are a "
                              f"sum over the tokens ({origin.step or 'unknown step'})")
    if origin.raw_table is None:
        return Origin(problem=f"the raw table behind {set_name} ({raw}) cannot be found")
    return origin


def column_index(name: str, set_name: str, columns: Sequence[str]) -> Optional[int]:
    """
    Which unit column an analysis-table column is, or ``None`` when it is
    not one (a control, a cosine column: not a sum over units).

    The analysis table spells a dimension ``e12__mean`` (the merge step's
    tail) or ``wv_3``, behind, perhaps, the assemble step's ``<set>__``
    collision prefix; the raw table spells it ``e12`` / ``e_13`` / ``wv_3``.
    The exact name is tried first, then the name before the ``__`` tail,
    so ``sim_zoo__pets`` is not mistaken for a dimension called ``sim_zoo``.
    """
    from ..helpers.feature_columns import SEPARATOR

    bare = name
    prefix = f"{set_name}{SEPARATOR}"
    if bare.startswith(prefix):
        bare = bare[len(prefix):]
    known = list(columns)
    if bare in known:
        return known.index(bare)
    head, _sep, _tail = bare.partition(SEPARATOR)
    if head in known:
        return known.index(head)
    return None


# ---------------------------------------------------------------------------
# Tokens into words
# ---------------------------------------------------------------------------

def merge_pieces(tokens: Sequence[str], specials: Sequence[str] = ()
                 ) -> List[Tuple[str, List[int]]]:
    """
    Group word pieces into the words a person reads: ``[(word, positions)]``.

    WordPiece continues a word with ``##``; byte-level BPE starts one with
    ``Ġ`` (a leading space) and continues without it; SentencePiece starts
    one with ``▁``. A special token is a word of its own named
    :data:`SPECIALS`. The positions are what the shares are summed over.
    """
    special = set(specials)
    out: List[Tuple[str, List[int]]] = []
    bpe = any(t.startswith("Ġ") or t.startswith("▁") for t in tokens)

    def wordy(text: str) -> bool:
        return any(ch.isalnum() for ch in text)

    for i, tok in enumerate(tokens):
        if tok in special:
            out.append((SPECIALS, [i]))
            continue
        if tok.startswith("##") and out and out[-1][0] != SPECIALS:
            out[-1] = (out[-1][0] + tok[2:], out[-1][1] + [i])
            continue
        if bpe:
            if tok.startswith("Ġ") or tok.startswith("▁"):
                out.append((tok[1:], [i]))
            elif (out and out[-1][0] != SPECIALS and not tok.startswith("<")
                  # a piece with no leading space continues the word -- unless
                  # it is punctuation, which byte-level BPE also writes without
                  # the space and which is not part of the word before it
                  # ("summer." and "fun," were turning up as words)
                  and wordy(tok) and wordy(out[-1][0])):
                out[-1] = (out[-1][0] + tok, out[-1][1] + [i])
            else:
                out.append((tok, [i]))
            continue
        out.append((tok, [i]))
    return [(w.lower(), p) for w, p in out if w]


# ---------------------------------------------------------------------------
# The contract: a text is a sum of units
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    """
    One thing the pooling summed over, with what it added to the text's
    row: ``vector`` is its share of the set's unit columns, every scalar
    of the pooling folded in, so a text's row is the sum of its units'
    vectors. ``token`` is the backend's spelling (the pieces of a word,
    joined), ``word`` the word a person reads; ``position`` counts from 1
    where the backend counts, ``sentence``/``window`` only where it reads
    by sentence.
    """

    position: int
    token: str
    word: str
    vector: Any
    sentence: Optional[int] = None
    window: Optional[int] = None


class Backend:
    """
    One step's way of reading a text into units.

    ``columns`` are the raw table's unit columns, in the raw spelling, in
    the order of the vectors' entries; ``exact``/``why`` whether the
    decomposition holds and, when it does not, why in a sentence;
    ``describe()`` the method in one sentence for the summary table.
    ``vocabulary()`` is the units a model knows without a text, for a
    model that has them (word vectors), else ``None``.
    """

    kind: str = ""
    exact: bool = True
    why: str = ""
    columns: List[str]
    model_label: str = ""
    device_name: str = ""
    reason: str = ""

    def __init__(self, origin: Origin, *, device: str, batch_size: int,
                 verbose: bool):  # pragma: no cover - the subclasses
        self.columns = []

    def describe(self) -> str:  # pragma: no cover
        return ""

    def read(self, texts: Sequence[str]) -> List[List[Unit]]:  # pragma: no cover
        raise NotImplementedError

    def vocabulary(self) -> Optional[Iterator[Tuple[List[str], List[Optional[int]], Any]]]:
        """``(words, counts, vectors[n, d])`` in chunks, or ``None``."""
        return None


def _raw_columns(origin: Origin, pattern: "re.Pattern[str]") -> List[str]:
    header = tables.columns_of(origin.raw_table)
    return [c for c in header if pattern.match(c)]


def _units_from_states(sentences: List[List[Tuple[List[str], Any]]], *,
                       sentence_norms: List[float], token_counts: List[int],
                       weighting: str, normalize: bool, specials: Sequence[str]
                       ) -> List[Unit]:
    """
    The encoders' units: per sentence, its windows of ``(tokens, states
    [positions, d])``, folded the way the extraction pooled them -- mean
    over positions per window, mean over windows per sentence, the
    sentence's norm when the model normalizes, sentences weighted equally
    or by token count, and the text's L2 when the step normalized. Word
    pieces are merged into one unit per word (the pieces' vectors summed).
    """
    import numpy as np

    S = len(sentences)
    if S == 0:
        return []
    if weighting == "tokens":
        tc = np.array([max(1, c) for c in token_counts], dtype=np.float64)
        u = tc / tc.sum()
    else:
        u = np.full(S, 1.0 / S)
    # the text vector first, from the same states, for the L2 scale
    x = None
    for s_i, wins in enumerate(sentences):
        v = None
        for _toks, arr in wins:
            m = arr.astype(np.float64).mean(axis=0)
            v = m if v is None else v + m
        v = v / (max(1, len(wins)) * sentence_norms[s_i])
        x = u[s_i] * v if x is None else x + u[s_i] * v
    scale = 1.0
    if normalize:
        n = float(np.linalg.norm(x))
        if n > 1e-12:
            scale = 1.0 / n
    units: List[Unit] = []
    for s_i, wins in enumerate(sentences):
        for w_i, (tokens, arr) in enumerate(wins):
            weight = u[s_i] / (len(wins) * arr.shape[0]) / sentence_norms[s_i] * scale
            V = arr.astype(np.float64) * weight
            for word, positions in merge_pieces(tokens, specials):
                units.append(Unit(
                    position=positions[0], token=" ".join(tokens[p] for p in positions),
                    word=word, vector=V[positions].sum(axis=0),
                    sentence=s_i + 1, window=w_i + 1))
    return units


class _TransformerBackend(Backend):
    kind = "transformer"

    def __init__(self, origin: Origin, *, device: str, batch_size: int, verbose: bool):
        from ..text._transformer_common import load_encoder, set_threads

        inst = origin.instrument
        self.columns = _raw_columns(origin, _EMBED_COL)
        self.model_label = str(inst.get("model_name") or inst.get("model_name_or_path") or "")
        self.layers = str(inst.get("layers") or "second_to_last")
        self.pooling = str(inst.get("pooling") or "mean")
        self.max_length = int(inst.get("max_length") or 512)
        self.stride = int(inst.get("stride") or 32)
        self.weighting = str(inst.get("sentence_weighting") or "equal")
        self.normalize = bool(inst.get("normalize_l2", False))
        self.batch_size = int(batch_size)
        if self.pooling != "mean":
            self.exact = False
            self.why = (f"the embeddings pool by {self.pooling}, which is not a mean "
                        f"over the tokens, so a prediction over them is not a sum of "
                        f"token shares")
            return
        set_threads(0)
        self.model, self.tokenizer, _res, self.device_name, self.reason = load_encoder(
            self.model_label, device=device, verbose=verbose)
        self.specials = list(self.tokenizer.all_special_tokens)

    def describe(self) -> str:
        return (f"{self.model_label}: mean over the tokens of layer {self.layers}, sentences "
                f"weighted {'by token count' if self.weighting == 'tokens' else 'equally'}"
                + (", the text vector L2-normalized" if self.normalize else ""))

    def read(self, texts: Sequence[str]) -> List[List[Unit]]:
        from ..text._sentences import split_sentences
        from ..text._transformer_common import encode_sentences

        per_text = [split_sentences(t) for t in texts]
        flat = [s for sents in per_text for s in sents]
        if not flat:
            return [[] for _ in texts]
        _vecs, counts, per_sentence = encode_sentences(
            self.model, self.tokenizer, flat, layers=self.layers, pooling="mean",
            max_length=self.max_length, batch_size=self.batch_size,
            device_name=self.device_name, precision="fp32", stride=self.stride,
            return_tokens=True)
        out: List[List[Unit]] = []
        at = 0
        for sents in per_text:
            windows, norms, tc = [], [], []
            for _ in sents:
                windows.append([(self.tokenizer.convert_ids_to_tokens(ids), arr)
                                for ids, arr in per_sentence[at]])
                norms.append(1.0)
                tc.append(int(counts[at]))
                at += 1
            out.append(_units_from_states(
                windows, sentence_norms=norms, token_counts=tc, weighting=self.weighting,
                normalize=self.normalize, specials=self.specials))
        return out


class _SentenceTransformerBackend(Backend):
    kind = "sentence"

    def __init__(self, origin: Origin, *, device: str, batch_size: int, verbose: bool):
        from sentence_transformers import SentenceTransformer

        from ..helpers.gpu import resolve_device

        inst = origin.instrument
        self.columns = _raw_columns(origin, _EMBED_COL)
        self.model_label = str(inst.get("model_name") or "")
        self.normalize = bool(inst.get("normalize_l2", False))
        self.batch_size = int(batch_size)
        resolved, self.reason = resolve_device(device, backend="torch")
        self.device_name = resolved
        self.model = SentenceTransformer(self.model_label, device=resolved)
        names = [type(m).__name__ for m in self.model._modules.values()]
        self.modules = " -> ".join(names)
        pooling = next((m for m in self.model._modules.values()
                        if type(m).__name__ == "Pooling"), None)
        mode = getattr(pooling, "pooling_mode", None) if pooling is not None else None
        self.normalizes = "Normalize" in names
        allowed = (names == ["Transformer", "Pooling"]
                   or names == ["Transformer", "Pooling", "Normalize"])
        if not allowed or mode != "mean":
            self.exact = False
            self.why = (f"the model's modules are {self.modules}"
                        + (f" with pooling {mode!r}" if pooling is not None else "")
                        + ", not a plain mean over the tokens, so a prediction over its "
                          "vectors is not a sum of token shares")
            return
        self.specials = list(self.model.tokenizer.all_special_tokens)

    def describe(self) -> str:
        return (f"{self.model_label}: {self.modules}, sentences weighted equally"
                + (", the text vector L2-normalized" if self.normalize else ""))

    def read(self, texts: Sequence[str]) -> List[List[Unit]]:
        import numpy as np

        from ..text._sentences import split_sentences

        per_text = [split_sentences(t) for t in texts]
        flat = [s for sents in per_text for s in sents]
        if not flat:
            return [[] for _ in texts]
        # the per-token vectors before pooling, trimmed to the attended
        # positions, and the ids from the model's own tokenizer with the
        # same truncation. (`output_value=None`, which would hand both back
        # at once, trips over a string field in this sentence-transformers
        # version once a batch has more than four items.)
        embs = self.model.encode(flat, output_value="token_embeddings",
                                 batch_size=self.batch_size, show_progress_bar=False,
                                 convert_to_numpy=False)
        feats = self.model.tokenize(flat)
        out: List[List[Unit]] = []
        at = 0
        for sents in per_text:
            windows, norms, tc = [], [], []
            for _ in sents:
                mask = feats["attention_mask"][at].bool()
                ids = [int(v) for v in feats["input_ids"][at][mask].tolist()]
                arr = embs[at].detach().cpu().float().numpy().astype(np.float32)
                n = min(len(ids), arr.shape[0])
                ids, arr = ids[:n], arr[:n]
                windows.append([(self.model.tokenizer.convert_ids_to_tokens(ids), arr)])
                norm = float(np.linalg.norm(arr.mean(axis=0))) if self.normalizes else 1.0
                norms.append(norm if norm > 1e-12 else 1.0)
                tc.append(len(ids) - 2)
                at += 1
            out.append(_units_from_states(
                windows, sentence_norms=norms, token_counts=tc, weighting="equal",
                normalize=self.normalize, specials=self.specials))
        return out


#: The two bookkeeping columns of a word-vector row, which are sums over
#: the tokens too (1 each, and 1 per token the model knows), so they ride
#: along as the last two entries of every unit's vector.
_WV_BOOKKEEPING = ("token_count", "in_vocab_count")


class _WordVectorBackend(Backend):
    kind = "word_vectors"

    def __init__(self, origin: Origin, *, device: str, batch_size: int, verbose: bool):
        import numpy as np

        from ..text import word_vectors as wv

        self.np = np
        self.wv = wv
        self.columns = []
        inst = origin.instrument
        entry = _model_entry(origin.record)
        if entry is None:
            self.exact = False
            self.why = "its record does not carry the word-vector model it was made with"
            return
        path = _model_file(entry, origin.root)
        if path is None:
            self.exact = False
            self.why = (f"the model file {entry.get('name')} is not under this run or in "
                        f"the library")
            return
        try:
            self.model = wv._load_model(path)
            self.weighting, self.normalize_words, _dicts = wv._settings_of(
                self.model, inst.get("weighting"), inst.get("normalize_words"), None)
        except (OSError, ValueError) as e:
            self.exact = False
            self.why = f"the word-vector model could not be opened: {e}"
            return
        self.model_label = str(self.model.doc.get("name") or path.stem)
        self.columns = list(self.model.dim_labels) + list(_WV_BOOKKEEPING)
        self.text = self.model.text_settings
        self.lowercase = bool(self.text.get("lowercase", True))
        self.stream = wv._stream_for(self.text, device)
        self.device_name = "cpu"

    def describe(self) -> str:
        family = str((self.model.doc.get("training") or {}).get("family") or "word vectors")
        return (f"{self.model_label} ({family}, {len(self.model.index)} words): "
                f"{self.weighting} weighting over the words the model knows"
                + (", each word vector normalized" if self.normalize_words else ""))

    def _tokens(self, text: str) -> List[str]:
        # the one tokenizing path apply runs, cleaning included
        return [t.replace(" ", "_") for t in self.wv._tokens_of(self.stream, self.lowercase, text)
                if t.strip()]

    def read(self, texts: Sequence[str]) -> List[List[Unit]]:
        np = self.np
        out: List[List[Unit]] = []
        for text in texts:
            tokens = self._tokens(text)
            U = self.wv.unit_vectors(self.model, tokens, weighting=self.weighting,
                                     normalize_words=self.normalize_words)
            units = []
            for i, tok in enumerate(tokens):
                known = 1.0 if tok in self.model.index else 0.0
                units.append(Unit(position=i + 1, token=tok, word=tok,
                                  vector=np.concatenate([U[i], [1.0, known]])))
            out.append(units)
        return out

    def vocabulary(self):
        """Every word the model knows, as the unit a one-word text would
        be: its (normalized) vector with the two bookkeeping entries."""
        np = self.np
        words = [str(w) for w in self.model.doc["vocabulary"]]
        counts = self.model.counts
        step = self.wv._CHUNK_ROWS
        for start in range(0, len(words), step):
            block = np.asarray(self.model.vectors[start:start + step], dtype=np.float64)
            if self.normalize_words:
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                block = block / norms
            tail = np.ones((block.shape[0], 2), dtype=np.float64)
            chunk_counts = ([int(c) for c in counts[start:start + step]] if counts is not None
                            else [None] * block.shape[0])
            yield words[start:start + step], chunk_counts, np.hstack([block, tail])


def _model_entry(record: Dict[str, Any]) -> Optional[dict]:
    """The word-vector model a record carries: the apply step's asset, or
    the train step's replay asset. Name, digest and the manifest text."""
    for holder in (record.get("assets"), (record.get("replay") or {}).get("assets")):
        entries = (holder or {}).get("model_json") or []
        if entries and isinstance(entries[0], dict):
            return entries[0]
    return None


def _model_file(entry: dict, root: Optional[Path]) -> Optional[Path]:
    """
    The manifest file for a carried model: a file of its name under the
    run root whose digest matches, else the carried text written under
    ``<root>/assets/`` (the way a pipeline materializes a model's lists),
    from where the matrix is found beside it or in the library by digest.
    """
    from ..helpers.atomic import atomic_write
    from ..helpers.provenance import text_digest

    name = Path(str(entry.get("name") or "")).name
    wanted = entry.get("sha256")
    if not name:
        return None
    base = Path(root) if root is not None else Path.cwd()
    for found in sorted(base.rglob(name)):
        if not found.is_file():
            continue
        try:
            if not wanted or text_digest(found.read_text(encoding="utf-8-sig",
                                                         errors="replace")) == wanted:
                return found
        except OSError:
            continue
    text = entry.get("text")
    if text is None:
        return None
    target = base / "assets" / "model_json" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(target, mode="w", encoding="utf-8") as fh:
        fh.write(str(text))
    return target


_BACKENDS: Dict[str, type] = {
    "extract_transformer_embeddings": _TransformerBackend,
    "extract_sentence_embeddings": _SentenceTransformerBackend,
    "train_word_vectors": _WordVectorBackend,
    "apply_word_vectors": _WordVectorBackend,
}
#: The steps whose feature sets are a sum over the tokens: what can be
#: attributed, and (with the model-scoring steps) what counts as opaque.
ATTRIBUTABLE_STEPS: Tuple[str, ...] = tuple(_BACKENDS)


def backend_for(origin: Origin, *, device: str = "auto", batch_size: int = 16,
                verbose: bool = False) -> Backend:
    """The backend that reads an origin's texts into units; loading the
    model is its constructor's business, and ``exact``/``why`` say whether
    the decomposition holds."""
    if origin.problem or origin.step not in _BACKENDS:
        raise ValueError(origin.problem or f"no backend for {origin.step!r}")
    return _BACKENDS[origin.step](origin, device=device, batch_size=batch_size,
                                  verbose=verbose)


# ---------------------------------------------------------------------------
# The attribution itself
# ---------------------------------------------------------------------------

@dataclass
class _WordSums:
    total: float = 0.0
    sumsq: float = 0.0
    count: int = 0
    docs: int = 0
    last_doc: str = ""


@dataclass
class Attribution:
    """
    What one pass produced for one feature set.

    ``words[target][word]`` are the running sums per word; ``texts`` the
    full breakdown of the chosen texts as rows; ``audit`` the largest gap
    between a text's summed shares and its recomputed prediction, per
    target.
    """

    exact: bool
    why: str = ""
    words: Dict[str, Dict[str, _WordSums]] = field(default_factory=dict)
    texts: List[dict] = field(default_factory=list)
    audit: Dict[str, float] = field(default_factory=dict)
    n_texts: int = 0
    skipped: int = 0
    device: str = ""


def _wordy(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def attribute(texts: Any, *, backend: Backend,
              weights: Dict[str, Any],            # target label -> (w_eff[d], constant)
              predicted: Dict[str, Dict[str, float]],  # target label -> text_id -> ŷ
              chosen_ids: Sequence[str],
              extra: Optional[Dict[str, Dict[str, float]]] = None,  # label -> text_id -> term
              texts_per_call: int = 16, encoding: str = "utf-8-sig",
              on_progress: Optional[Callable[..., None]] = None) -> Attribution:
    """
    One pass over ``texts`` for one feature set, every target at once.

    ``weights`` maps a target label to its effective weight vector over
    the backend's unit columns and its constant. For each text the backend
    reads the units, every unit's share is ``vector · w``, the shares are
    summed per word (every target) and, for the texts in ``chosen_ids``,
    kept unit by unit. ``extra`` is the per-text term of the set's columns
    that are not sums over units, kept as one :data:`FEATURES` row. The
    audit compares each text's summed shares with ``w · Σ vectors``, which
    has to agree to floating point.
    """
    import numpy as np

    from ..helpers.progress import FlightReporter, count_rows

    labels = list(weights)
    out = Attribution(exact=True)
    if not labels:
        return out
    if not backend.exact:
        return Attribution(exact=False, why=backend.why)
    out.device = backend.device_name
    d = len(backend.columns)
    W = np.zeros((d, len(labels)), dtype=np.float64)
    for j, label in enumerate(labels):
        W[:, j] = np.asarray(weights[label][0], dtype=np.float64)
    constants = np.array([float(weights[label][1]) for label in labels])
    for label in labels:
        out.words[label] = {}
    chosen = set(str(c) for c in chosen_ids)
    extra = extra or {}

    texts_ref = tables.parse_ref(texts)
    total = count_rows(texts_ref, on_progress=on_progress)
    reporter = FlightReporter(on_progress, total, "attributing the predictions to tokens")

    def rows():
        for row in tables.read_rows(texts_ref, encoding=encoding, columns=["text_id", "text"]):
            yield str(row.get("text_id", "") or ""), str(row.get("text", "") or "")

    def handle(batch: List[Tuple[str, str]]) -> None:
        read = backend.read([text for _tid, text in batch])
        for (text_id, _text), units in zip(batch, read):
            reporter.start(text_id)
            if not units:
                out.skipped += 1
                reporter.finish(text_id)
                reporter.consumed()
                continue
            V = np.stack([u.vector for u in units])          # [units, d]
            C = V @ W                                         # [units, targets]
            base = V.sum(axis=0) @ W
            add = np.array([float(extra.get(label, {}).get(text_id, 0.0)) for label in labels])
            recomputed = base + add + constants
            gaps = np.abs(C.sum(axis=0) - base)
            # the words they make: one share per word per occurrence
            word_shares: Dict[str, np.ndarray] = {}
            word_hits: Dict[str, int] = {}
            for u, c in zip(units, C):
                # punctuation and specials carry a share like any unit and
                # stay in the text's breakdown; the word table is words,
                # the n-gram steps' own rule (`is_word`)
                if u.word != SPECIALS and _wordy(u.word):
                    word_shares[u.word] = word_shares.get(u.word, 0.0) + c
                    word_hits[u.word] = word_hits.get(u.word, 0) + 1
            for j, label in enumerate(labels):
                out.audit[label] = max(out.audit.get(label, 0.0), float(gaps[j]))
                table = out.words[label]
                for word, share in word_shares.items():
                    rec = table.get(word)
                    if rec is None:
                        rec = table[word] = _WordSums()
                    v = float(share[j])
                    rec.total += v
                    rec.sumsq += v * v
                    rec.count += word_hits[word]
                    if rec.last_doc != text_id:
                        rec.docs += 1
                        rec.last_doc = text_id
            if text_id in chosen:
                for j, label in enumerate(labels):
                    common = {"target": label, "text_id": text_id,
                              "prediction": predicted.get(label, {}).get(text_id),
                              "recomputed": float(recomputed[j])}
                    for u, c in zip(units, C):
                        out.texts.append({**common, "sentence": u.sentence, "window": u.window,
                                          "position": u.position, "token": u.token,
                                          "word": u.word, "contribution": float(c[j])})
                    if label in extra and text_id in extra[label]:
                        out.texts.append({**common, "sentence": None, "window": None,
                                          "position": len(units) + 1, "token": FEATURES,
                                          "word": FEATURES, "contribution": float(add[j])})
            out.n_texts += 1
            reporter.finish(text_id)
            reporter.consumed()

    batch: List[Tuple[str, str]] = []
    for text_id, text in rows():
        if not any(text_id in predicted.get(label, {}) for label in labels):
            out.skipped += 1
            reporter.consumed()
            continue
        batch.append((text_id, text))
        if len(batch) >= texts_per_call:
            handle(batch)
            batch = []
    if batch:
        handle(batch)
    return out
