from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Union, Sequence, Iterable, Tuple
import csv
import re
import sys
import numpy as np
import nltk

# Allow very large CSV fields (handles huge text safely).
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

# Lazy import to keep startup light.
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore

from ..helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)

PathLike = Union[str, Path]


def _ensure_nltk_punkt(verbose: bool = True) -> bool:
    """
    Ensure the NLTK sentence tokenizer is available.

    Checks for the presence of NLTK's ``punkt`` (and, for newer NLTK builds,
    ``punkt_tab``). If missing, attempts a quiet download. Prints a short
    status message when ``verbose`` is True.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print status messages about tokenizer availability.

    Returns
    -------
    bool
        ``True`` if NLTK's tokenizer is usable; ``False`` if a regex fallback
        should be used instead.

    Notes
    -----
    This helper does not load heavy NLP models. It only ensures that sentence
    segmentation can proceed.
    """

    try:
        nltk.data.find("tokenizers/punkt")
        ok = True
    except LookupError:
        if verbose:
            print("Downloading NLTK 'punkt' tokenizer ...")
        try:
            nltk.download("punkt", quiet=True)
            nltk.data.find("tokenizers/punkt")
            ok = True
        except LookupError:
            # Some setups expose data under 'punkt_tab'
            try:
                if verbose:
                    print("Trying NLTK 'punkt_tab' ...")
                nltk.download("punkt_tab", quiet=True)
                nltk.data.find("tokenizers/punkt_tab")
                ok = True
            except LookupError:
                ok = False

    if verbose:
        if ok:
            print("Sentence tokenizer available: using NLTK sent_tokenize.")
        else:
            print("Sentence tokenizer NOT available: using regex fallback.")
    return ok

def _split_sentences(text: str) -> list[str]:
    """
    Split a text string into sentences.

    Prefers ``nltk.tokenize.sent_tokenize`` if available; otherwise falls back
    to a lightweight regex that splits on end punctuation followed by whitespace.

    Parameters
    ----------
    text : str
        Input text. ``None``/empty values are treated as empty strings.

    Returns
    -------
    list of str
        List of non-empty, stripped sentences. Returns an empty list when the
        input is empty or contains no sentence-like chunks.

    Notes
    -----
    The regex fallback is intentionally simple and language-agnostic; it may
    under-segment or over-segment compared to NLTK's tokenizer.
    """

    txt = (text or "").strip()
    if not txt:
        return []
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore
        return [s for s in sent_tokenize(txt) if s.strip()]
    except Exception:
        # crude but dependency-free: split on end punctuation + whitespace
        parts = re.split(r"(?<=[.!?])\s+", txt)
        return [p.strip() for p in parts if p.strip()]

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

def analyze_with_sentence_embeddings(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default

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
    pattern: str = "*.txt",
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== SentenceTransformer options ======
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    batch_size: int = 32,
    normalize_l2: bool = True,       # set True if you want unit-length vectors
    rounding: Optional[int] = None,   # None = full precision; e.g., 6 for ~float32-ish text
    show_progress: bool = False,
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

    ``text_id, e0, e1, ..., e{D-1}``

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
    out_features_csv : str or pathlib.Path, optional
        Output features CSV path. If ``None``, a default path is derived from the
        analysis-ready filename under ``./features/sentence-embeddings/``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and return it.

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
    batch_size : int, default=32
        Batch size for model encoding.
    normalize_l2 : bool, default=True
        If ``True``, L2-normalize each row's final vector.
    rounding : int or None, default=None
        If provided, round floats to this many decimals (useful for smaller files).
    show_progress : bool, default=False
        Show a progress bar during embedding.

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

    # pre-check that nltk sent_tokenizer is usable
    use_nltk = _ensure_nltk_punkt(verbose=True)

    # 1) analysis-ready CSV
    if analysis_csv is not None:
        analysis_ready = Path(analysis_csv)
        if not analysis_ready.exists():
            raise FileNotFoundError(f"analysis_csv not found: {analysis_ready}")
    else:
        if (csv_path is None) == (txt_dir is None):
            raise ValueError("Provide exactly one of csv_path or txt_dir (or pass analysis_csv).")
        if csv_path is not None:
            analysis_ready = Path(
                csv_to_analysis_ready_csv(
                    csv_path=csv_path,
                    text_cols=list(text_cols),
                    id_cols=list(id_cols) if id_cols else None,
                    mode=mode,
                    group_by=list(group_by) if group_by else None,
                    delimiter=delimiter,
                    encoding=encoding,
                    joiner=joiner,
                    num_buckets=num_buckets,
                    max_open_bucket_files=max_open_bucket_files,
                    tmp_root=tmp_root,
                )
            )
        else:
            analysis_ready = Path(
                txt_folder_to_analysis_ready_csv(
                    root_dir=txt_dir,
                    recursive=recursive,
                    pattern=pattern,
                    encoding=encoding,
                    id_from=id_from,
                    include_source_path=include_source_path,
                )
            )

    # 1b) default output path
    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "sentence-embeddings" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_features_csv).is_file():
        print("Sentence embedding feature output file already exists; returning existing file.")
        return out_features_csv

    # 2) load model
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required. Install with `pip install sentence-transformers`."
        )
    print(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 768)())

    # 3) header
    header = ["text_id"] + [f"e{i}" for i in range(dim)]

    # 4) stream rows → split → encode → average → (optional) L2 normalize → write
    def _norm(v: np.ndarray) -> np.ndarray:
        if not normalize_l2:
            return v
        n = float(np.linalg.norm(v))
        return v if n < 1e-12 else (v / n)

    print("Extracting embeddings...")
    with out_features_csv.open("w", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for text_id, text in _iter_items_from_csv(analysis_ready, encoding=encoding, delimiter=delimiter):
            sents = _split_sentences(text)
            if not sents:
                vec = None
            else:
                emb = model.encode(
                    sents,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=show_progress,
                )
                # Average across sentences → one vector
                vec = emb.mean(axis=0).astype(np.float32, copy=False)
            
            
            # L2 and rounding only if we have a vector
            if vec is None:
                values = [""] * dim  # <- write empty cells, not zeros/NaNs
            else:
                if normalize_l2:
                    n = float(np.linalg.norm(vec))
                    if n > 1e-12:
                        vec = vec / n
                if rounding is not None:
                    values = [round(float(x), int(rounding)) for x in vec.tolist()]
                else:
                    values = [float(x) for x in vec.tolist()]

            writer.writerow([text_id] + values)

    return out_features_csv



# --- CLI ------------------------------------------------------------
def _build_arg_parser():
    """
    Create an ``argparse.ArgumentParser`` for the sentence-embedding CLI.

    Defines mutually exclusive input sources (``--csv``, ``--txt-dir``,
    ``--analysis-csv``), output/overwrite flags, CSV/TXT gathering options,
    and Sentence-Transformers parameters.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    import argparse
    p = argparse.ArgumentParser(
        description="Average sentence embeddings per row (Sentence-Transformers)."
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", dest="csv_path", help="Source CSV to gather from")
    src.add_argument("--txt-dir", dest="txt_dir", help="Folder of .txt files to gather from")
    src.add_argument("--analysis-csv", dest="analysis_csv",
                     help="Use an existing analysis-ready CSV (skip gathering)")

    p.add_argument("--out", dest="out_features_csv", default=None,
                   help="Output CSV (default: ./features/sentence-embeddings/<gathered_name>)")
    p.add_argument("--overwrite_existing", type=bool, default=False,
                    help="Do you want to overwrite the output file if it already exists?")

    # I/O
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--delimiter", default=",")

    # CSV gather options
    p.add_argument("--text-col", dest="text_cols", action="append",
                   help="Text column (repeatable). Default: --text-col text")
    p.add_argument("--id-col", dest="id_cols", action="append",
                   help="ID column(s) to carry through (repeatable)")
    p.add_argument("--mode", choices=["concat", "separate"], default="concat")
    p.add_argument("--group-by", dest="group_by", action="append",
                   help="Group by column(s) (repeatable)")
    p.add_argument("--joiner", default=" ")
    p.add_argument("--num-buckets", type=int, default=512)
    p.add_argument("--max-open-bucket-files", type=int, default=64)
    p.add_argument("--tmp-root", default=None)

    # TXT gather options
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--id-from", choices=["stem", "name", "path"], default="stem")
    p.add_argument("--include-source-path", action="store_true", default=True)
    p.add_argument("--no-include-source-path", dest="include_source_path", action="store_false")

    # Model options
    p.add_argument("--model-name", default="sentence-transformers/all-roberta-large-v1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--normalize-l2", action="store_true", default=True,
                   help="L2-normalize the final vector per row")
    p.add_argument("--rounding", type=int, default=None,
                   help="Round floats to N decimals (omit for full precision)")
    p.add_argument("--show-progress", action="store_true", default=False)

    return p

def main():
    """
    Command-line entry point for row-level sentence embeddings.

    Parses CLI arguments via :func:`_build_arg_parser`, normalizes list-like
    defaults (e.g., ``--text-col``, ``--id-col``, ``--group-by``), invokes
    :func:`analyze_with_sentence_embeddings`, and prints the resulting path.

    Examples
    --------
    $ python -m taters.text.extract_sentence_embeddings \\
        --csv transcripts/session.csv \\
        --text-col text --id-col speaker --group-by speaker \\
        --model-name sentence-transformers/all-roberta-large-v1 \\
        --normalize-l2
    """
    args = _build_arg_parser().parse_args()

    # Defaults for list-ish args
    text_cols = args.text_cols if args.text_cols else ["text"]
    id_cols = args.id_cols if args.id_cols else None
    group_by = args.group_by if args.group_by else None

    out = analyze_with_sentence_embeddings(
        csv_path=args.csv_path,
        txt_dir=args.txt_dir,
        analysis_csv=args.analysis_csv,
        out_features_csv=args.out_features_csv,
        overwrite_existing=args.overwrite_existing,
        encoding=args.encoding,
        delimiter=args.delimiter,
        text_cols=text_cols,
        id_cols=id_cols,
        mode=args.mode,
        group_by=group_by,
        joiner=args.joiner,
        num_buckets=args.num_buckets,
        max_open_bucket_files=args.max_open_bucket_files,
        tmp_root=args.tmp_root,
        recursive=args.recursive,
        pattern=args.pattern,
        id_from=args.id_from,
        include_source_path=args.include_source_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_l2=args.normalize_l2,
        rounding=args.rounding,
        show_progress=args.show_progress,
    )
    print(str(out))

if __name__ == "__main__":
    main()
