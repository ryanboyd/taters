from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Literal, Union, Sequence, Iterable, Tuple
import csv
import numpy as np

from ..helpers.nltk_data import ensure_punkt
from ..helpers.atomic import atomic_write
from ..helpers.gpu import device_note, resolve_device
from ..helpers.progress import Ticker, announce, count_rows
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec
from ..helpers.csvio import widen_csv_field_limit

widen_csv_field_limit()

PathLike = Union[str, Path]


def _ensure_nltk_punkt(verbose: bool = True) -> bool:
    """
    Ensure the NLTK sentence tokenizer is available.

    Thin wrapper kept for backwards compatibility; the implementation is shared
    with the other modules that need sentence splitting.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print status messages about tokenizer availability.

    Returns
    -------
    bool
        ``True`` if NLTK's tokenizer is usable; ``False`` if a regex fallback
        should be used instead.
    """
    return ensure_punkt(verbose=verbose)

def _split_sentences(text: str) -> list[str]:
    """
    Split a text string into sentences.

    The one splitter every sentence-by-sentence step shares
    (:mod:`taters.text._sentences`), kept under its old name here for the
    callers that import it from this module.
    """
    from ._sentences import split_sentences

    return split_sentences(text)


def _iter_items_from_csv(path: Path, *, id_col: str = "text_id", text_col: str = "text",
                         encoding: str = "utf-8-sig", delimiter: str = ",") -> Iterable[Tuple[str, str]]:
    """
    Stream ``(text_id, text)`` pairs from an analysis-ready CSV.

    Parameters
    ----------
    path : pathlib.Path
        Path to the CSV file.
    id_col : str, default="text_id"
        Name of the identifier column in the CSV.
    text_col : str, default="text"
        Name of the text column in the CSV.
    encoding : str, default="utf-8-sig"
        File encoding.
    delimiter : str, default=","
        Field delimiter.

    Yields
    ------
    tuple of (str, str)
        The ``(text_id, text)`` for each row. Missing text values are emitted
        as empty strings.

    Raises
    ------
    ValueError
        If the required ``id_col`` and ``text_col`` are not present in the header.
    """

    with path.open("r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if id_col not in reader.fieldnames or text_col not in reader.fieldnames:
            raise ValueError(f"Expected columns '{id_col}' and '{text_col}' in {path}; found {reader.fieldnames}")
        for row in reader:
            yield str(row[id_col]), (row.get(text_col) or "")

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",))
def extract_sentence_embeddings(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,
    gathered_csv: Optional[Union[str, Path]] = None,

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,
    workers: int = 0,

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",
    delimiter: str = ",",

    # ====== CSV GATHER OPTIONS (when csv_path is provided) ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[Union[str, Path]] = None,

    # ====== TXT FOLDER GATHER OPTIONS (when txt_dir is provided) ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== SentenceTransformer options ======
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    device: Optional[str] = "auto",
    batch_size: int = 32,
    normalize_l2: bool = True,       # set True if you want unit-length vectors
    rounding: Optional[int] = None,   # None = full precision; 6 is about float32
    show_progress: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
    pass_through_cols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Path:
    """
    Average sentence embeddings per row of text and write a wide features CSV.

    Supports three mutually exclusive input modes:

    1. ``analysis_csv`` — Use a prebuilt file with columns ``text_id`` and ``text``.
    2. ``csv_path`` — Gather from a CSV using ``text_cols`` (and optional
    ``id_cols``/``group_by``) to build an analysis-ready CSV.
    3. ``txt_dir`` — Gather from a folder of ``.txt`` files.

    For each row, the text is split into sentences (NLTK if available; otherwise
    a regex fallback). Each sentence is embedded with a Sentence-Transformers
    model and the vectors are averaged into one row-level embedding. Optionally,
    vectors are L2-normalized. The output CSV schema is:

    ``text_id[, <pass_through_cols...>], e0, e1, ..., e{D-1}``

    If ``out_features_csv`` is omitted, the default is
    ``./features/sentence-embeddings/<analysis_ready_filename>``. When
    ``overwrite_existing`` is ``False`` and the output exists, the function
    returns the existing path without recomputation.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Source CSV to gather from. Mutually exclusive with ``txt_dir`` and ``analysis_csv``.
    txt_dir : str or pathlib.Path, optional
        Folder of ``.txt`` files to gather from. Mutually exclusive with the other modes.
    analysis_csv : str or pathlib.Path, optional
        Prebuilt analysis-ready CSV containing exactly ``text_id`` and ``text``.
    gathered_csv : str or pathlib.Path, optional
        Where to write the intermediate "analysis-ready" table built from
        ``csv_path`` or ``txt_dir``.
    workers : int, default=0
        Parallel processes for reading documents. ``0`` means automatic: three-quarters of the
        logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.

        By default it lands beside the *source* -- which means analyzing a
        spreadsheet in someone's Downloads folder writes a file into their
        Downloads folder. Pass this to keep the intermediate with the rest of a
        run's output instead. Ignored when ``analysis_csv`` is given, because
        then no gathering happens.
    out_features_csv : str or pathlib.Path, optional
        Output features CSV path. If ``None``, a default path is derived from the
        analysis-ready filename under ``./features/sentence-embeddings/``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and return it.
        This also controls the intermediate analysis-ready CSV: when ``True``, it is rebuilt
        from the current source instead of reusing a stale copy from an earlier run.
    verbose : bool, default True
        Print incidental notices -- which model is loading, which requested
        pass-through columns the input did not have. The pipeline runner passes
        False when a live display owns the screen.
    pass_through_cols : Sequence[str], optional
        Column names from the analysis-ready CSV to copy into the output
        alongside ``text_id`` (e.g., ``["source","speaker"]``). **Any names
        given in ``id_cols`` are always included automatically**, even if not
        listed here. Missing columns are ignored with a warning.


    encoding : str, default="utf-8-sig"
        CSV I/O encoding.
    delimiter : str, default=","
        CSV field delimiter.

    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV: column(s) containing text.
    id_cols : Sequence[str], optional
        When gathering from a CSV: optional ID columns to carry through.
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior if multiple ``text_cols`` are provided. ``"concat"`` joins
        them with ``joiner``; ``"separate"`` creates one row per column.
    group_by : Sequence[str], optional
        Optional grouping keys used during CSV gathering (e.g., ``["speaker"]``).
    joiner : str, default=" "
        Separator used when concatenating text in ``"concat"`` mode.
    num_buckets : int, default=512
        Number of temporary hash buckets for scalable gathering.
    max_open_bucket_files : int, default=64
        Maximum number of bucket files kept open concurrently during gathering.
    tmp_root : str or pathlib.Path, optional
        Root directory for temporary gathering artifacts.

    recursive : bool, default=True
        When gathering from a text folder, recurse into subdirectories.
    pattern : str, default="*.txt"
        Glob pattern for selecting text files.
    id_from : {"stem", "name", "path"}, default="stem"
        How to derive ``text_id`` when gathering from a text folder.
    include_source_path : bool, default=True
        Whether to include the absolute source path as an additional column when
        gathering from a text folder.

    model_name : str, default="sentence-transformers/all-roberta-large-v1"
        Sentence-Transformers model name or path.
    device : {"auto", "cuda", "cpu"} | None, default "auto"
        Where to run the embedding model. "auto" uses the GPU when torch reports
        one that works and falls back to the CPU when it does not; "cuda"
        insists and raises if it cannot; "cpu" never touches the GPU. Previously
        there was no way to ask: sentence-transformers takes the GPU whenever
        torch reports one, which is fine until it is the third model in a
        pipeline to do so.
    batch_size : int, default=32
        Batch size for model encoding.
    normalize_l2 : bool, default=True
        If ``True``, L2-normalize each row's final vector.
    rounding : int or None, default=None
        If provided, round floats to this many decimals (useful for smaller files).
    show_progress : bool, default=False
        Print the model's own encoding progress bar. Suppressed whenever
        ``on_progress`` is given, because two bars fighting over the same
        lines is worse than either alone.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)`` so a UI can show a
        real bar instead of a spinner. Injected automatically by the pipeline
        runner for any step function that declares this parameter. See
        :mod:`taters.helpers.progress` for the contract.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Raises
    ------
    FileNotFoundError
        If an input file or directory does not exist.
    ImportError
        If ``sentence-transformers`` is not installed.
    ValueError
        If input modes are misconfigured (e.g., multiple or none provided),
        or if the analysis-ready CSV lacks ``text_id``/``text``.

    Examples
    --------
    Compute row-level embeddings from a transcript CSV, grouped by speaker:

    >>> analyze_with_sentence_embeddings(
    ...     csv_path="transcripts/session.csv",
    ...     text_cols=["text"], id_cols=["speaker"], group_by=["speaker"],
    ...     model_name="sentence-transformers/all-roberta-large-v1",
    ...     normalize_l2=True
    ... )
    PosixPath('.../features/sentence-embeddings/session.csv')

    Notes
    -----
    - Rows with no recoverable sentences produce **empty** feature cells (not zeros).
    - The embedding dimensionality ``D`` is taken from the model and used to
    construct header columns ``e0..e{D-1}``.
    """

    def _merge_cols(preferred: Optional[Sequence[str]], ensure: Optional[Sequence[str]]) -> list[str]:
        """
        Merge two sequences while preserving order and removing duplicates.
        'preferred' order is kept; any 'ensure' items not present are appended.
        """
        out: list[str] = []
        seen: set[str] = set()
        for seq in (preferred or []), (ensure or []):
            for c in seq:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
        return out

    # pre-check that nltk's sent_tokenizer is usable. we call this purely for
    # the side effect (it downloads `punkt` if it's missing), and the sentence
    # splitter falls back on its own if it's still not there, so there's no
    # return value worth keeping
    _ensure_nltk_punkt(verbose=verbose)

    # we resolve these BEFORE the gather, because the gather is what has to
    # preserve them. `pass_through_cols` used to only get applied when reading
    # the analysis-ready CSV back, but the gather that produced it wrote
    # `text_id` and `text` and nothing else. so every requested column came
    # out present-but-empty, and anything grouping on them downstream silently
    # collapsed into one meaningless bucket
    pt_cols: list[str] = _merge_cols(pass_through_cols, id_cols)

    analysis_ready = resolve_analysis_ready(
        csv_path=csv_path, txt_dir=txt_dir, analysis_csv=analysis_csv,
        gathered_csv=gathered_csv, text_cols=text_cols, id_cols=id_cols,
        mode=mode, group_by=group_by, delimiter=delimiter, encoding=encoding,
        joiner=joiner, num_buckets=num_buckets,
        max_open_bucket_files=max_open_bucket_files, tmp_root=tmp_root,
        recursive=recursive, pattern=pattern, id_from=id_from,
        include_source_path=include_source_path,
        overwrite_existing=overwrite_existing, on_progress=on_progress,
        workers=workers,
        carry_cols=pt_cols or None, verbose=verbose)

    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "sentence-embeddings" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_features_csv).is_file():
        if verbose:
            print("Sentence embedding feature output file already exists; returning existing file.")
        return out_features_csv

    # 2) load model
    #
    # we import this here rather than at module scope. it used to be guarded up
    # there, which keeps a missing install from breaking the import -- but
    # that's not lazy, and sentence-transformers takes about fourteen seconds
    # to load. the wizard imports this module just to read its signature when
    # it builds the options screen, and was paying that every single time
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required. Install with `pip install sentence-transformers`."
        ) from e
    # we resolve the device ourselves rather than leaving it to
    # sentence-transformers, which grabs CUDA whenever torch reports it and
    # gives us no way to find out afterwards what it picked. a step that
    # silently moves to the GPU is a step that can silently run it out of memory
    resolved, fallback_reason = resolve_device(device, backend="torch")
    if verbose:
        print(f"Loading sentence-transformer model: {model_name} on {resolved}")
        if fallback_reason:
            print(f"[sentence-embeddings] {fallback_reason}")
    # and on the run display too, not only under `verbose` -- see the same
    # announcement in transformer_embeddings for why
    announce(on_progress, device_note("embedding", resolved, fallback_reason))
    model = SentenceTransformer(model_name, device=resolved)
    dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 768)())

    # 3) header
    header = ["text_id"] + pt_cols + [f"e{i}" for i in range(dim)]


    # 4) stream rows → split → encode → average → (optional) L2 normalize → write
    if verbose:
        print("Extracting embeddings...")
    with atomic_write(out_features_csv, newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # open the analysis-ready CSV as dicts so we can read the extra cols
        with analysis_ready.open("r", newline="", encoding=encoding) as rf:
            reader = csv.DictReader(rf, delimiter=delimiter)
            # light validation: warn if any requested pass-through column is missing
            missing = [c for c in pt_cols if c not in (reader.fieldnames or [])]
            if missing and verbose:
                print(f"[sentence-embeddings] WARNING: pass-through columns missing in source: {missing}")

            _ticker = Ticker(on_progress,
                             count_rows(analysis_ready, on_progress=on_progress))

            for row in reader:
                text_id = str(row.get("text_id", ""))
                text = (row.get("text") or "")
                pt_vals = [row.get(c, "") for c in pt_cols]

                sents = _split_sentences(text)
                if not sents:
                    vec = None
                else:
                    emb = model.encode(
                        sents,
                        batch_size=batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=show_progress and on_progress is None,
                    )
                    vec = emb.mean(axis=0).astype(np.float32, copy=False)

                if vec is None:
                    values = [""] * dim
                else:
                    if normalize_l2:
                        n = float(np.linalg.norm(vec))
                        if n > 1e-12:
                            vec = vec / n
                    values = [float(x) for x in vec.tolist()]
                    if rounding is not None:
                        values = [round(v, int(rounding)) for v in values]

                writer.writerow([text_id] + pt_vals + values)
                _ticker.tick()


    return out_features_csv



# --- CLI ------------------------------------------------------------



#: The name the module, its recipe and the facade used to disagree on: the
#: module is `extract_sentence_embeddings`, the function was
#: `analyze_with_sentence_embeddings`, the facade a third spelling. One name
#: now; the old one stays callable.
analyze_with_sentence_embeddings = extract_sentence_embeddings

# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    extract_sentence_embeddings,
    description='Average sentence embeddings per row (Sentence-Transformers).',
    aliases={
        'csv_path': ['--csv'],
        'out_features_csv': ['--out'],
    },
    legacy={
        '--no-include-source-path': ['--include-source-path', 'false'],
        '--no-recursive': ['--recursive', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
