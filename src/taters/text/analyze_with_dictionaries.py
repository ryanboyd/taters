from pathlib import Path
from typing import Optional, Literal, Union, Sequence, Iterable, Tuple
import csv

from .dictionary_analyzers import multi_dict_analyzer as mda
from ..helpers.find_files import find_files
from ..helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)

def analyze_with_dictionaries(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,  # if provided, gathering is skipped

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default

    # ----- Dictionaries -----
    dict_paths: Sequence[Union[str, Path]], # LIWC2007 (.dic) or LIWC-22 format (.dicx, .csv)

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

    # ====== ANALYZER OPTIONS (passed through to ContentCoder) ======
    relative_freq: bool = True,
    drop_punct: bool = True,
    rounding: int = 4,
    retain_captures: bool = False,
    wildcard_mem: bool = True,
) -> Path:
    """
    Compute LIWC-style dictionary features for text rows and write a wide features CSV.

    The function supports exactly one of three input modes:

    1. ``analysis_csv`` — Use a prebuilt file with columns ``text_id`` and ``text``.
    2. ``csv_path`` — Gather text from an arbitrary CSV using ``text_cols`` (and optional
    ``id_cols``/``group_by``) to produce an analysis-ready file.
    3. ``txt_dir`` — Gather text from a folder of ``.txt`` files.

    If ``out_features_csv`` is omitted, the default output path is
    ``./features/dictionary/<analysis_ready_filename>``. Multiple dictionaries are supported;
    passing a directory discovers all ``.dic``, ``.dicx``, and ``.csv`` dictionary files
    recursively in a stable order. Global columns (e.g., word counts, punctuation) are emitted
    once (from the first dictionary) and each dictionary contributes a namespaced block.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Source CSV to gather from. Mutually exclusive with ``txt_dir`` and ``analysis_csv``.
    txt_dir : str or pathlib.Path, optional
        Folder containing ``.txt`` files to gather from. Mutually exclusive with other modes.
    analysis_csv : str or pathlib.Path, optional
        Prebuilt analysis-ready CSV with exactly two columns: ``text_id`` and ``text``.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/dictionary/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and return the path.
    dict_paths : Sequence[str or pathlib.Path]
        One or more dictionary inputs (files or directories). Supported extensions:
        ``.dic``, ``.dicx``, ``.csv``. Directories are expanded recursively.
    encoding : str, default="utf-8-sig"
        Text encoding used for reading/writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, name(s) of the column(s) containing text.
    id_cols : Sequence[str] or None, optional
        Optional ID columns to carry into grouping when gathering from CSV.
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple text columns are provided. ``"concat"`` joins them
        into one text field using ``joiner``; ``"separate"`` creates one row per column.
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
    relative_freq : bool, default=True
        Emit relative frequencies instead of raw counts, when supported by the dictionary engine.
    drop_punct : bool, default=True
        Drop punctuation prior to analysis (dictionary-dependent).
    rounding : int, default=4
        Decimal places to round numeric outputs. Use ``None`` to disable rounding.
    retain_captures : bool, default=False
        Pass-through flag to the underlying analyzer to retain capture groups, if applicable.
    wildcard_mem : bool, default=True
        Pass-through optimization flag for wildcard handling in the analyzer.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Raises
    ------
    FileNotFoundError
        If input files/folders or any dictionary file cannot be found.
    ValueError
        If input modes are misconfigured (e.g., multiple sources provided or none),
        required columns are missing from the analysis-ready CSV, or unsupported
        dictionary extensions are encountered.

    Examples
    --------
    Run on a transcript CSV, grouped by speaker:

    >>> analyze_with_dictionaries(
    ...     csv_path="transcripts/session.csv",
    ...     text_cols=["text"], id_cols=["speaker"], group_by=["speaker"],
    ...     dict_paths=["dictionaries/liwc/LIWC-22 Dictionary (2022-01-27).dicx"]
    ... )
    PosixPath('.../features/dictionary/session.csv')

    Notes
    -----
    If ``overwrite_existing`` is ``False`` and the output exists, the existing file path
    is returned without recomputation.
    """


    # 1) Produce or accept the analysis-ready CSV (must have columns: text_id,text)
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

    # 1b) Decide default features path if not provided:
    #     <cwd>/features/dictionary/<analysis_ready_filename>
    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "dictionary" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_features_csv).is_file():
        print("Dictionary content coding output file already exists; returning existing file.")
        return out_features_csv


    # 2) Validate dictionaries
    def _expand_dict_inputs(paths):
        """
        Normalize dictionary inputs into a unique, ordered list of files.

        Parameters
        ----------
        paths : Iterable[Union[str, pathlib.Path]]
            Files or directories. Directories are expanded recursively to files with
            extensions ``.dic``, ``.dicx``, or ``.csv``.

        Returns
        -------
        list[pathlib.Path]
            Deduplicated, resolved file paths in stable order.

        Raises
        ------
        FileNotFoundError
            If a referenced file or directory does not exist.
        ValueError
            If a file has an unsupported extension or if no dictionary files are found.
        """

        out = []
        seen = set()
        for p in map(Path, paths):
            if p.is_dir():
                # Find .dic/.dicx/.csv under this folder (recursive), stable order
                found = find_files(
                    root_dir=p,
                    extensions=[".dic", ".dicx", ".csv"],
                    recursive=True,
                    absolute=True,
                    sort=True,
                )
                for f in found:
                    fp = Path(f).resolve()
                    if fp.suffix.lower().lstrip(".") in {"dic", "dicx", "csv"}:
                        if fp not in seen:
                            out.append(fp)
                            seen.add(fp)
            else:
                if not p.exists():
                    raise FileNotFoundError(f"Dictionary path not found: {p}")
                fp = p.resolve()
                if fp.suffix.lower().lstrip(".") not in {"dic", "dicx", "csv"}:
                    raise ValueError(f"Unsupported dictionary extension: {fp.name}")
                if fp not in seen:
                    out.append(fp)
                    seen.add(fp)
        if not out:
            raise ValueError("No dictionary files found. Supply .dic/.dicx/.csv files or folders containing them.")
        return out

    dict_paths = _expand_dict_inputs(dict_paths)

        # 3) Stream the analysis-ready CSV into the analyzer → features CSV
    def _iter_items_from_csv_with_meta(
        path: Path,
        *,
        id_col: str = "text_id",
        text_col: str = "text",
        pass_through_cols: Optional[Sequence[str]] = None,
    ) -> Iterable[Tuple[str, str, dict]]:
        """
        Stream (text_id, text, meta) from an analysis-ready CSV.

        Parameters
        ----------
        path : pathlib.Path
            Path to the analysis-ready CSV file.
        id_col : str, default="text_id"
            Identifier column.
        text_col : str, default="text"
            Text column.
        pass_through_cols : Sequence[str] or None
            Extra columns to fetch per row and forward to the analyzer.

        Yields
        ------
        tuple[str, str, dict]
            (text_id, text, meta_dict) where meta_dict maps each pass-through column
            to its string value ('' if missing).
        """
        wanted = list(pass_through_cols or [])
        with path.open("r", newline="", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = reader.fieldnames or []
            if id_col not in fields or text_col not in fields:
                raise ValueError(
                    f"Expected columns '{id_col}' and '{text_col}' in {path}; found {fields}"
                )
            # If id_cols were requested, enforce they exist up-front (fail fast)
            missing = [c for c in wanted if c not in fields]
            if missing:
                raise ValueError(
                    f"Requested id_cols not present in analysis-ready CSV {path}: {missing}"
                )

            for row in reader:
                tid = str(row.get(id_col, "") or "")
                text = str(row.get(text_col, "") or "")
                meta = {c: str(row.get(c, "") or "") for c in wanted}
                yield tid, text, meta

    # Use multi_dict_analyzer as the middle layer (new API)
    mda.analyze_texts_to_csv(
        items=_iter_items_from_csv_with_meta(analysis_ready, pass_through_cols=id_cols or []),
        dict_files=dict_paths,
        out_csv=out_features_csv,
        relative_freq=relative_freq,
        drop_punct=drop_punct,
        rounding=rounding,
        retain_captures=retain_captures,
        wildcard_mem=wildcard_mem,
        id_col_name="text_id",
        pass_through_cols=list(id_cols or []),  # <-- inject id_cols right after text_id
        encoding=encoding,
    )


    return out_features_csv



# --- CLI ------------------------------------------------------------
def _build_arg_parser():
    """
    Create an ``argparse.ArgumentParser`` for the dictionary coding CLI.

    The parser exposes three mutually exclusive input modes (``--csv``, ``--txt-dir``,
    ``--analysis-csv``), output/overwrite options, repeatable ``--dict`` arguments
    (accepting files or directories), gathering parameters for CSV/TXT inputs,
    and analyzer pass-through options.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    import argparse
    p = argparse.ArgumentParser(
        description="ContentCoder: multi-dictionary coding into one CSV (globals once + per-dict blocks)."
    )

    # Input source (choose one)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", dest="csv_path", help="Source CSV to gather from")
    src.add_argument("--txt-dir", dest="txt_dir", help="Folder of .txt files to gather from")
    src.add_argument("--analysis-csv", dest="analysis_csv",
                     help="Use an existing analysis-ready CSV (skip gathering)")

    # Output
    p.add_argument("--out", dest="out_features_csv", default=None,
                   help="Output CSV (default: ./features/dictionary/<gathered_name>)")
    p.add_argument("--overwrite_existing", type=bool, default=False,
                   help="Do you want to overwrite the output file if it already exists?")

    # Dictionaries (repeatable)
    p.add_argument("--dict", dest="dict_paths", action="append", required=True,
                   help="Path to a .dicx dictionary (repeat this flag for multiple)")

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

    # Analyzer options (pass-through to ContentCoder)
    p.add_argument("--relative-freq", action="store_true", default=True)
    p.add_argument("--no-relative-freq", dest="relative_freq", action="store_false")
    p.add_argument("--drop-punct", action="store_true", default=True)
    p.add_argument("--no-drop-punct", dest="drop_punct", action="store_false")
    p.add_argument("--retain-captures", action="store_true", default=False)
    p.add_argument("--wildcard-mem", action="store_true", default=True)
    p.add_argument("--no-wildcard-mem", dest="wildcard_mem", action="store_false")
    p.add_argument("--rounding", type=int, default=4)

    return p

def main():
    r"""
    Command-line entry point for multi-dictionary content coding.

    Parses CLI arguments via :func:`_build_arg_parser`, normalizes list-like defaults,
    invokes :func:`analyze_with_dictionaries`, and prints the resulting output path.

    Examples
    --------
    Basic usage on a CSV with grouping by speaker:

    $ python -m taters.text.analyze_with_dictionaries \
        --csv transcripts/session.csv \
        --text-col text --id-col speaker --group-by speaker \
        --dict dictionaries/liwc/LIWC-22\ Dictionary\ (2022-01-27).dicx

    Notes
    -----
    Boolean flags include positive/negative pairs (e.g., ``--recursive`` /
    ``--no-recursive``, ``--relative-freq`` / ``--no-relative-freq``) to make
    CLI behavior explicit.
    """

    args = _build_arg_parser().parse_args()

    # Defaults for list-ish args
    text_cols = args.text_cols if args.text_cols else ["text"]
    id_cols = args.id_cols if args.id_cols else None
    group_by = args.group_by if args.group_by else None

    out = analyze_with_dictionaries(
        csv_path=args.csv_path,
        txt_dir=args.txt_dir,
        analysis_csv=args.analysis_csv,
        out_features_csv=args.out_features_csv,
        overwrite_existing=args.overwrite_existing,
        dict_paths=args.dict_paths,
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
        relative_freq=args.relative_freq,
        drop_punct=args.drop_punct,
        rounding=args.rounding,
        retain_captures=args.retain_captures,
        wildcard_mem=args.wildcard_mem,
    )
    print(str(out))

if __name__ == "__main__":
    main()
