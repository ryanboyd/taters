# taters/text/analyze_readability.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Sequence, Literal, Iterable, Dict, Any
import csv
import sys


from ..helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)


# ---- textstat loader ---------------------------------------------------------

def _require_textstat():
    """
    Import and return the `textstat` module or raise a helpful error if missing.
    """
    try:
        import textstat  # type: ignore
        return textstat
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            'The "textstat" package is required for readability analysis.\n'
            'Install it via the optional extra:\n\n'
            '    pip install "taters[readability]"\n\n'
            "or add `textstat` to your environment."
        ) from e


# ---- Core API ----------------------------------------------------------------

def analyze_readability(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,  # if provided, gathering is skipped

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS ======
    # Only used when csv_path is provided
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[Union[str, Path]] = None,

    # ====== TXT FOLDER GATHER OPTIONS ======
    # Only used when txt_dir is provided
    recursive: bool = True,
    pattern: str = "*.txt",
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,
) -> Path:
    """
    Compute per-row readability metrics using `textstat` and write a wide features CSV.

    The function supports exactly one of three input modes:

    1. ``analysis_csv`` — Use a prebuilt file with at least columns ``text_id`` and ``text``.
    2. ``csv_path`` — Gather text from an arbitrary CSV using ``text_cols`` (and optional
       ``id_cols``/``group_by``) to produce an analysis-ready file.
    3. ``txt_dir`` — Gather text from a folder of ``.txt`` files.

    If ``out_features_csv`` is omitted, the default output path is
    ``./features/readability/<analysis_ready_filename>``. All metrics below are computed
    for every row. Non-numeric metrics (e.g., ``text_standard``) are retained as strings.

    Metrics (columns)
    -----------------
    The following metrics are emitted as columns (subject to `textstat` availability):

    - ``flesch_reading_ease``
    - ``smog_index``
    - ``flesch_kincaid_grade``
    - ``coleman_liau_index``
    - ``automated_readability_index``
    - ``dale_chall_readability_score``
    - ``difficult_words``
    - ``linsear_write_formula``
    - ``gunning_fog``
    - ``text_standard``                 (string label)
    - ``spache_readability``            (for shorter/children texts; may be None)
    - ``readability_consensus``         (string label)
    - ``syllable_count``                (on entire text)
    - ``lexicon_count``                 (word count)
    - ``sentence_count``
    - ``char_count``
    - ``avg_sentence_length``
    - ``avg_syllables_per_word``
    - ``avg_letter_per_word``

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Source CSV to gather from. Mutually exclusive with ``txt_dir`` and ``analysis_csv``.
    txt_dir : str or pathlib.Path, optional
        Folder containing ``.txt`` files to gather from. Mutually exclusive with other modes.
    analysis_csv : str or pathlib.Path, optional
        Prebuilt analysis-ready CSV with columns ``text_id`` and ``text`` (additional columns
        such as ``source``/``speaker`` will be copied through to the output).
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/readability/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and return the path.
    encoding : str, default="utf-8-sig"
        Text encoding used for reading/writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, name(s) of the column(s) containing text.
    id_cols : Sequence[str] or None, optional
        Optional ID columns to carry into grouping when gathering from CSV.
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple text columns are provided. ``"concat"`` joins them
        using ``joiner``; ``"separate"`` creates one row per column.
    group_by : Sequence[str] or None, optional
        Optional grouping keys used during CSV gathering (e.g., ``["speaker"]``).
    delimiter : str, default=","
        Delimiter for reading/writing CSV files.
    joiner : str, default=" "
        Separator used when concatenating multiple text chunks in ``"concat"`` mode.
    num_buckets : int, default=512
        Number of temporary hash buckets used during scalable CSV gathering.
    max_open_bucket_files : int, default=64
        Maximum number of bucket files kept open concurrently during gathering.
    tmp_root : str or pathlib.Path or None, optional
        Root directory for temporary gathering artifacts.
    recursive : bool, default=True
        When gathering from a text folder, recurse into subdirectories.
    pattern : str, default="*.txt"
        Glob pattern for selecting text files when gathering from a folder.
    id_from : {"stem", "name", "path"}, default="stem"
        How to derive ``text_id`` for gathered ``.txt`` files.
    include_source_path : bool, default=True
        If ``True``, include the absolute source path as an additional column when gathering
        from a text folder.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Raises
    ------
    FileNotFoundError
        If an input is missing.
    ValueError
        If input modes are misconfigured or required columns are absent.
    RuntimeError
        If ``textstat`` is not installed.

    Notes
    -----
    - All rows are processed; blank or missing text yields benign defaults (metrics may be 0 or None).
    - Additional columns present in the analysis-ready CSV (beyond ``text``) are copied through
      to the output (e.g., ``source``, ``speaker``, ``group_count``), aiding joins/aggregation.
    """
    textstat = _require_textstat()

    # 1) Accept or produce the analysis-ready CSV (must have: text_id, text)
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

    # 2) Decide default features path if not provided:
    #    <cwd>/features/readability/<analysis_ready_filename>
    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "readability" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print(f"Readability output file already exists; returning existing file: {out_features_csv}")
        return out_features_csv

    # 3) Stream analysis-ready CSV and compute metrics per row
    metrics = [
        # readability indices
        "flesch_reading_ease",
        "smog_index",
        "flesch_kincaid_grade",
        "coleman_liau_index",
        "automated_readability_index",
        "dale_chall_readability_score",
        "difficult_words",
        "linsear_write_formula",
        "gunning_fog",
        # labels / consensus
        "text_standard",
        "spache_readability",
        "readability_consensus",
        # counts/derived
        "syllable_count",
        "lexicon_count",
        "sentence_count",
        "char_count",
        "avg_sentence_length",
        "avg_syllables_per_word",
        "avg_letter_per_word",
    ]

    # open input/output
    with analysis_ready.open("r", newline="", encoding=encoding) as fin, \
         out_features_csv.open("w", newline="", encoding=encoding) as fout:
        reader = csv.DictReader(fin, delimiter=delimiter)

        if "text_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns 'text_id' and 'text' in {analysis_ready}; "
                f"found {reader.fieldnames}"
            )

        # Carry through any non-text columns (e.g., id cols, source, speaker)
        passthrough_cols = [c for c in reader.fieldnames if c != "text"]

        # Output header = passthrough + metrics
        fieldnames = passthrough_cols + metrics
        writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()

        # Helpers to call textstat functions safely
        def _call_metric(name: str, txt: str) -> Any:
            # Guard against missing attributes across textstat versions
            fn = getattr(textstat, name, None)
            if fn is None:
                # Fallback: approximate consensus with text_standard if available
                if name == "readability_consensus" and hasattr(textstat, "text_standard"):
                    try:
                        return textstat.text_standard(txt)
                    except Exception:
                        return None
                return None
            try:
                return fn(txt)
            except Exception:
                return None


        for row in reader:
            txt = (row.get("text") or "").strip()
            out_row: Dict[str, Any] = {k: row.get(k) for k in passthrough_cols}
            for m in metrics:
                out_row[m] = _call_metric(m, txt)
            writer.writerow(out_row)

    return out_features_csv


# ---- CLI ---------------------------------------------------------------------

def _build_arg_parser():
    """
    Create an ``argparse.ArgumentParser`` for the readability CLI.

    Mirrors the CSV/TXT gathering and output options used elsewhere in Taters.
    """
    import argparse
    p = argparse.ArgumentParser(
        description="Compute textstat readability metrics for an analysis-ready CSV."
    )

    # Input source (choose one)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", dest="csv_path", help="Source CSV to gather from")
    src.add_argument("--txt-dir", dest="txt_dir", help="Folder of .txt files to gather from")
    src.add_argument("--analysis-csv", dest="analysis_csv",
                     help="Use an existing analysis-ready CSV (skip gathering)")

    # Output
    p.add_argument("--out", dest="out_features_csv", default=None,
                   help="Output CSV (default: ./features/readability/<gathered_name>)")
    p.add_argument("--overwrite_existing", type=lambda s: str(s).lower() == "true", default=False,
                   help="Overwrite output if it exists (true/false). Default: false")

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

    return p


def main():
    r"""
    Command-line entry point for readability metrics.

    Examples
    --------
    On a prebuilt analysis-ready CSV:

    $ python -m taters.text.analyze_readability --analysis-csv transcripts.csv

    Gather from a transcript CSV and group by speaker before scoring:

    $ python -m taters.text.analyze_readability \
        --csv transcripts/session.csv \
        --text-col text --id-col source --id-col speaker \
        --group-by source --group-by speaker --mode concat
    """
    args = _build_arg_parser().parse_args()

    # Defaults for list-ish args
    text_cols = args.text_cols if args.text_cols else ["text"]
    id_cols = args.id_cols if args.id_cols else None
    group_by = args.group_by if args.group_by else None

    out = analyze_readability(
        csv_path=args.csv_path,
        txt_dir=args.txt_dir,
        analysis_csv=args.analysis_csv,
        out_features_csv=args.out_features_csv,
        overwrite_existing=args.overwrite_existing,
        encoding=args.encoding,
        text_cols=text_cols,
        id_cols=id_cols,
        mode=args.mode,
        group_by=group_by,
        delimiter=args.delimiter,
        joiner=args.joiner,
        num_buckets=args.num_buckets,
        max_open_bucket_files=args.max_open_bucket_files,
        tmp_root=args.tmp_root,
        recursive=args.recursive,
        pattern=args.pattern,
        id_from=args.id_from,
        include_source_path=args.include_source_path,
    )
    print(str(out))


if __name__ == "__main__":
    main()
