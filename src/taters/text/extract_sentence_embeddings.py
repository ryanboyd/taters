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
    Ensure NLTK sentence tokenizer data is available.
    Tries 'punkt' first; falls back to 'punkt_tab' (used in newer NLTK builds).
    Returns True if sent_tokenize is usable; False otherwise.
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
    """Prefer NLTK sent_tokenize if available; fallback to a simple regex."""
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
    Build/accept an analysis-ready CSV (columns: text_id,text) and write
    one average sentence-embedding vector per row:
        text_id, e0, e1, ..., e{D-1}

    If out_features_csv is not provided, defaults to:
        ./features/sentence-embeddings/<analysis_ready_filename>
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
