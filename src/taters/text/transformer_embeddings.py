"""
Embeddings from any transformer encoder, sentence by sentence, averaged
per text.

Conceptually the sentence-transformers step: each text is split into
sentences with the same splitter, every sentence is encoded, and the
text's vector is the mean of its sentence vectors. What this step adds is
the *choice of encoder and of reading*: any Hugging Face encoder, an
encoder adapted to your corpus in Taters, or the encoder inside a
fine-tuned predictor; which hidden layers are read and how token vectors
are pooled. A sentence-transformers model has been trained to put a
sentence's meaning in one place; a plain encoder has not, so the defaults
here -- the second-to-last layer, mean pooling -- are the ones that
transfer best as frozen features (the last layer is specialized to
predicting masked words; ``[CLS]`` means little in a model never trained to
use it).

Long sentences are windowed with overlap and their windows averaged, never
cut; sentences from many texts are batched together for throughput; a GPU
that runs out of memory halves the batch and carries on. A text with no
sentence gets blank cells, not zeros.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, List, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.csvio import widen_csv_field_limit
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.gpu import device_note
from ..helpers.progress import Ticker, announce, count_rows
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready
from ._sentences import split_sentences
from ._transformer_common import (CURATED_ENCODERS, encode_sentences,
                                  load_encoder, round_cells, set_threads,
                                  torch_missing_reason)
from ..helpers.feature_columns import ColumnSpec

#: `e_1, e_2, ...` -- see the note in `extract_sentence_embeddings` about why
#: these are spelled differently from its `e0, e1, ...`.
FEATURE_COLUMNS = ColumnSpec(label="Transformer embeddings", patterns=("e_{n}",))

widen_csv_field_limit()

__all__ = ["extract_transformer_embeddings", "DEFAULT_ENCODER"]

PathLike = Union[str, Path]

DEFAULT_ENCODER = CURATED_ENCODERS[0][0]

#: How many sentences to gather from consecutive texts before encoding: a
#: batch drawn from many short texts keeps the GPU full where one text at a
#: time (the sentence-transformers step's shape) leaves it idle.
_SENTENCES_PER_CALL = 256


def _doc_vector(vectors, token_counts: Sequence[int], weighting: str):
    """A text's vector from its sentences': the plain mean (``equal``, the
    sentence-transformers step's rule) or weighted by sentence length in
    tokens (``tokens``), so a long sentence counts for more."""
    import numpy as np

    if vectors.shape[0] == 0:
        return None
    if weighting == "tokens":
        w = np.asarray([max(1, int(c)) for c in token_counts], dtype=np.float64)
        return (vectors.astype(np.float64) * w[:, None]).sum(axis=0) / w.sum()
    return vectors.astype(np.float64).mean(axis=0)


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  bookkeeping=("token_count", "sentence_count"))
def extract_transformer_embeddings(
    *,
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
    delimiter: str = ",",
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,
    pass_through_cols: Optional[Sequence[str]] = None,

    # ----- the encoder and how it is read -----
    model_name_or_path: str = DEFAULT_ENCODER,
    layers: str = "second_to_last",
    pooling: Literal["mean", "cls", "max"] = "mean",
    sentence_weighting: Literal["equal", "tokens"] = "equal",
    max_length: int = 512,
    batch_size: int = 32,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    precision: Literal["auto", "fp32", "fp16"] = "auto",
    normalize_l2: bool = False,
    rounding: int = 6,
) -> Path:
    """
    Embed every text with a transformer encoder: sentence vectors, averaged.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv, gathered_csv
        The same input contract as every other text analyzer: a spreadsheet
        of texts, a folder of documents, or a prebuilt analysis-ready CSV.
    out_features_csv : str or Path, optional
        Default ``./features/transformer_embeddings.csv``.
    overwrite_existing : bool, default False
        If False and the output exists, return it untouched.
    workers : int, default 0
        Parallel processes for reading documents during the gather, and the
        CPU threads torch may use. 0 means automatic.
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
    pass_through_cols : sequence of str, optional
        Columns of the source carried into the output beside ``text_id``.
    model_name_or_path : str
        A Hugging Face encoder name (``distilroberta-base``,
        ``sentence-transformers/all-MiniLM-L6-v2``, ``roberta-base``,
        ``bert-base-uncased``, ``microsoft/deberta-v3-base``), a checkpoint
        folder, a Taters text encoder file (adapted to your corpus), or a
        Taters fine-tuned predictor file (its encoder is used). Downloaded
        on first use.
    layers : str, default "second_to_last"
        Which hidden layers become the token vectors: ``second_to_last``
        (the recommended reading of an encoder that was not fine-tuned: the
        last layer is specialized to predicting masked words), ``last``,
        ``last4_mean``, ``last4_concat`` (four times the width), or a list
        like ``-1,-2`` (averaged).
    pooling : {"mean", "cls", "max"}, default "mean"
        How a sentence's token vectors become one: the mean over its real
        tokens, the ``[CLS]`` position, or the element-wise maximum.
    sentence_weighting : {"equal", "tokens"}, default "equal"
        How a text's sentence vectors become one: each sentence equally
        (what the sentence-transformers step does) or weighted by length in
        tokens.
    max_length : int, default 512
        The most tokens the encoder reads at once. A longer sentence is
        windowed with overlap and its windows averaged, never cut.
    batch_size : int, default 32
        Sentences per forward pass; halved automatically if the GPU runs out
        of memory.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where the encoder runs.
    precision : {"auto", "fp32", "fp16"}, default "auto"
        Half precision on a GPU (auto), always full, or always half.
    normalize_l2 : bool, default False
        Scale each text's vector to unit length, for cosine comparisons.
    rounding : int, default 6
        Decimal places written.

    Returns
    -------
    Path
        ``out_features_csv``: ``text_id[, pass-through], token_count,
        sentence_count, e_1..e_d``.
    """
    import numpy as np

    missing = torch_missing_reason()
    if missing:
        raise ImportError(missing)
    if sentence_weighting not in ("equal", "tokens"):
        raise ValueError("sentence_weighting must be 'equal' or 'tokens', not "
                         f"{sentence_weighting!r}")
    from ..helpers.nltk_data import ensure_punkt

    ensure_punkt(verbose=verbose)

    from ..helpers.row_map import resolve_passthrough_columns

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers, carry_cols=list(pass_through_cols or []) or None,
        verbose=verbose)
    out_features_csv = Path(out_features_csv) if out_features_csv else \
        Path.cwd() / "features" / "transformer_embeddings.csv"
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        if verbose:
            print("Transformer embeddings output already exists; returning existing file.")
        return out_features_csv

    announce(on_progress, f"loading the encoder ({model_name_or_path})")
    set_threads(workers)
    model, tokenizer, resolved, device_name, reason = load_encoder(
        model_name_or_path, device=device, verbose=verbose)
    with analysis_ready.open("r", newline="", encoding=encoding) as rf:
        header_fields = csv.DictReader(rf, delimiter=delimiter).fieldnames or []
    passthrough = resolve_passthrough_columns(
        header_fields, pass_through_cols=pass_through_cols, id_cols=id_cols,
        group_by=group_by, analysis_ready=analysis_ready)

    total = count_rows(analysis_ready, on_progress=on_progress)
    # which device we ended up on, said where it stays put: the ticks below
    # carry no message, so this is the label somebody watches for the whole
    # encode. it used to be a print behind `verbose`, which the app never
    # sets -- so an hour-long run on the CPU looked exactly like a fast one
    # on the card, and the only way to find out was to watch the GPU meter
    announce(on_progress, device_note("encoding", device_name, reason))
    ticker = Ticker(on_progress, total)
    dim: Optional[int] = None

    def encode(sentences: List[str]):
        return encode_sentences(
            model, tokenizer, sentences, layers=layers, pooling=pooling,
            max_length=max_length, batch_size=batch_size,
            device_name=device_name, precision=precision, verbose=verbose)

    def flush(pending, writer):
        """Encode the buffered texts' sentences in one go and write a row
        per text -- the batch spans texts, the averaging never does."""
        nonlocal dim
        flat = [s for _row, sents in pending for s in sents]
        vectors, counts = encode(flat)
        if dim is None:
            dim = int(vectors.shape[1]) if vectors.shape[0] else int(
                model.config.hidden_size) * (4 if layers == "last4_concat" else 1)
        at = 0
        for row, sents in pending:
            k = len(sents)
            vecs, cnts = vectors[at:at + k], counts[at:at + k]
            at += k
            doc = _doc_vector(vecs, cnts, sentence_weighting)
            cells: List[object] = [row.get("text_id", "")]
            cells += [row.get(c, "") for c in passthrough]
            cells += [int(sum(cnts)), k]
            if doc is None:
                cells += [None] * dim
            else:
                if normalize_l2:
                    n = float(np.linalg.norm(doc))
                    if n > 1e-12:
                        doc = doc / n
                cells += round_cells(doc, rounding)
            writer.writerow(cells)
            ticker.tick()

    with atomic_write(out_features_csv, newline="", encoding=encoding) as out, \
            analysis_ready.open("r", newline="", encoding=encoding) as rf:
        reader = csv.DictReader(rf, delimiter=delimiter)
        writer = csv.writer(out)
        # we can't write the header until we know the embedding width, and we
        # don't know that until the first forward pass. so, we buffer rows until
        # the first flush and write the header then
        header_written = False
        pending: list = []
        buffered = 0
        for row in reader:
            sents = split_sentences(row.get("text") or "")
            pending.append((row, sents))
            buffered += len(sents)
            if buffered >= _SENTENCES_PER_CALL:
                if not header_written:
                    _probe_dim(pending, encode, model, layers)
                    dim = _probe_dim.dim
                    writer.writerow(_header(passthrough, dim))
                    header_written = True
                flush(pending, writer)
                pending, buffered = [], 0
        if not header_written:
            _probe_dim(pending, encode, model, layers)
            dim = _probe_dim.dim
            writer.writerow(_header(passthrough, dim))
        if pending:
            flush(pending, writer)
    if verbose:
        print(f"[transformer_embeddings] {total} text(s) embedded with {resolved.label} "
              f"(layers={layers}, pooling={pooling}) on {device_name} -> {out_features_csv}")
    return out_features_csv


def _header(passthrough: Sequence[str], dim: int) -> List[str]:
    return ["text_id", *passthrough, "token_count", "sentence_count",
            *(f"e_{i + 1}" for i in range(int(dim)))]


def _probe_dim(pending, encode, model, layers: str) -> None:
    """The output width, from the first real sentence when there is one
    (a concatenation of layers is wider than the model) and from the
    config otherwise."""
    first = next((s for _row, sents in pending for s in sents), None)
    if first is not None:
        vectors, _counts = encode([first])
        _probe_dim.dim = int(vectors.shape[1])
    else:
        _probe_dim.dim = int(model.config.hidden_size) * (
            4 if str(layers) == "last4_concat" else 1)


_probe_dim.dim = 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# command line -- we derive this from the function above; see
# helpers.cliargs.CliSpec
# ---------------------------------------------------------------------------

CLI = CliSpec(
    extract_transformer_embeddings,
    description="Embed texts with any transformer encoder: every sentence "
                "encoded, the text's vector their mean.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
