from pathlib import Path
from typing import Optional, Literal, Union, Sequence, Iterable, Tuple
import csv

from .dictionary_analyzers import multi_archetype_analyzer as maa
from ..helpers.find_files import find_files
from ..helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)

def analyze_with_archetypes(
    *,
    # ----- Input source (choose exactly one, OR pass analysis_csv to skip gathering) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,   # <- NEW: skip gathering if provided

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default

    # ----- Archetype CSVs (one or more) -----
    archetype_csvs: Sequence[Union[str, Path]],

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

    # ====== Archetyper scoring options ======
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    mean_center_vectors: bool = True,
    fisher_z_transform: bool = False,
    rounding: int = 4,
) -> Path:
    """
    Compute archetype scores for text rows and write a wide, analysis-ready features CSV.

    This function supports three input modes:

    1. ``analysis_csv`` — Use a prebuilt CSV with exactly two columns: ``text_id`` and ``text``.
    2. ``csv_path`` — Gather text from an arbitrary CSV by specifying ``text_cols`` (and optionally
    ``id_cols`` and ``group_by``) to construct an analysis-ready CSV on the fly.
    3. ``txt_dir`` — Gather text from a folder of ``.txt`` files.

    Archetype scoring is delegated to a middle layer that embeds text with a Sentence-Transformers
    model and evaluates cosine similarity to one or more archetype CSVs. If ``out_features_csv`` is
    omitted, the default path is ``./features/archetypes/<analysis_ready_filename>``.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Source CSV for gathering. Mutually exclusive with ``txt_dir`` and ``analysis_csv``.
    txt_dir : str or pathlib.Path, optional
        Folder of ``.txt`` files to gather from. Mutually exclusive with the other input modes.
    analysis_csv : str or pathlib.Path, optional
        Precomputed analysis-ready CSV containing exactly the columns ``text_id`` and ``text``.
    out_features_csv : str or pathlib.Path, optional
        Output path for the features CSV. If ``None``, defaults to
        ``./features/archetypes/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip recomputation and return the existing path.
    archetype_csvs : Sequence[str or pathlib.Path]
        One or more archetype CSVs (name → seed phrases). Directories are allowed and expanded
        recursively to all ``.csv`` files.
    encoding : str, default="utf-8-sig"
        Text encoding for CSV I/O.
    delimiter : str, default=","
        Field delimiter for CSV I/O.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV: column(s) that contain text. Used only if ``csv_path`` is provided.
    id_cols : Sequence[str], optional
        When gathering from a CSV: optional ID columns to carry into grouping (e.g., ``["speaker"]``).
    mode : {"concat", "separate"}, default="concat"
        Gathering behavior when multiple ``text_cols`` are provided. ``"concat"`` joins into a single
        text field; ``"separate"`` creates one row per text column.
    group_by : Sequence[str], optional
        Optional grouping keys used during gathering (e.g., ``["speaker"]``). In ``"concat"`` mode,
        members are concatenated into one row per group.
    joiner : str, default=" "
        Separator used when concatenating multiple text chunks.
    num_buckets : int, default=512
        Number of temporary hash buckets used for scalable CSV gathering.
    max_open_bucket_files : int, default=64
        Maximum number of bucket files to keep open concurrently during gathering.
    tmp_root : str or pathlib.Path, optional
        Root directory for temporary files used by gathering.
    recursive : bool, default=True
        When gathering from a text folder, whether to recurse into subdirectories.
    pattern : str, default="*.txt"
        Filename glob used when gathering from a text folder.
    id_from : {"stem", "name", "path"}, default="stem"
        How to derive the ``text_id`` when gathering from a text folder.
    include_source_path : bool, default=True
        Whether to include the absolute source path as an additional column when gathering from a text folder.
    model_name : str, default="sentence-transformers/all-roberta-large-v1"
        Sentence-Transformers model used to embed text for archetype scoring.
    mean_center_vectors : bool, default=True
        If ``True``, mean-center embedding vectors prior to scoring.
    fisher_z_transform : bool, default=False
        If ``True``, apply the Fisher z-transform to correlations.
    rounding : int, default=4
        Number of decimal places to round numeric outputs. Use ``None`` to disable rounding.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Raises
    ------
    FileNotFoundError
        If an input file or folder does not exist, or an archetype CSV path is invalid.
    ValueError
        If required arguments are incompatible or missing (e.g., no input mode chosen),
        or if the analysis-ready CSV lacks ``text_id``/``text`` columns.

    Examples
    --------
    Run on a transcript CSV, grouped by speaker:

    >>> analyze_with_archetypes(
    ...     csv_path="transcripts/session.csv",
    ...     text_cols=["text"],
    ...     id_cols=["speaker"],
    ...     group_by=["speaker"],
    ...     archetype_csvs=["dictionaries/archetypes"],
    ...     model_name="sentence-transformers/all-roberta-large-v1",
    ... )
    PosixPath('.../features/archetypes/session.csv')

    Notes
    -----
    If ``out_features_csv`` exists and ``overwrite_existing=False``, the existing path is returned
    without recomputation. Directories passed in ``archetype_csvs`` are expanded recursively to
    all ``.csv`` files and deduplicated before scoring.
    """


    # 1) Use analysis-ready CSV if given; otherwise gather from csv_path or txt_dir
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
    #     <analysis_ready_dir>/features/archetypes/<analysis_ready_filename>
    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "archetypes" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_features_csv).is_file():
        print("Archetypes output file already exists; returning existing file.")
        return out_features_csv


    # 2) Resolve/validate archetype CSVs
    # Allow passing either:
    #   • one or more CSV files, or
    #   • one or more directories containing CSVs (recursively).
    #
    # We lean on the shared find_files helper to avoid redundancy.

    # 2) Resolve/validate archetype CSVs
    resolved_archetype_csvs: list[Path] = []

    for src in archetype_csvs:
        src_path = Path(src)
        if src_path.is_dir():
            # find all *.csv under this folder (recursive)
            found = find_files(
                root_dir=src_path,
                extensions=[".csv"],
                recursive=True,
                absolute=True,
                sort=True,
            )
            resolved_archetype_csvs.extend(Path(f) for f in found)
        else:
            resolved_archetype_csvs.append(src_path)

    # De-dup, normalize, and sort
    archetype_csvs = sorted({p.resolve() for p in resolved_archetype_csvs})

    if not archetype_csvs:
        raise ValueError(
            "No archetype CSVs found. Pass one or more CSV files, or a directory containing CSV files with your archetypes."
        )
    for p in archetype_csvs:
        if not p.exists():
            raise FileNotFoundError(f"Archetype CSV not found: {p}")



    # 3) Stream (text_id, text) → middle layer → features CSV
    def _iter_items_from_csv(path: Path, *, id_col: str = "text_id", text_col: str = "text") -> Iterable[Tuple[str, str]]:
        with path.open("r", newline="", encoding=encoding) as f:
            """
            Stream ``(text_id, text)`` pairs from an analysis-ready CSV.

            Parameters
            ----------
            path : pathlib.Path
                Path to the analysis-ready CSV containing at least ``text_id`` and ``text``.
            id_col : str, default="text_id"
                Name of the identifier column to read.
            text_col : str, default="text"
                Name of the text column to read.

            Yields
            ------
            tuple of (str, str)
                The ``(text_id, text)`` for each row. Missing text values are emitted as empty strings.

            Raises
            ------
            ValueError
                If the required columns are not present in the CSV header.
            """

            reader = csv.DictReader(f, delimiter=delimiter)
            if id_col not in reader.fieldnames or text_col not in reader.fieldnames:
                raise ValueError(
                    f"Expected columns '{id_col}' and '{text_col}' in {path}; found {reader.fieldnames}"
                )
            for row in reader:
                yield str(row[id_col]), (row.get(text_col) or "")

    maa.analyze_texts_to_csv(
        items=_iter_items_from_csv(analysis_ready),
        archetype_csvs=archetype_csvs,
        out_csv=out_features_csv,
        model_name=model_name,
        mean_center_vectors=mean_center_vectors,
        fisher_z_transform=fisher_z_transform,
        rounding=rounding,
        encoding=encoding,
        delimiter=delimiter,
        id_col_name="text_id",
    )

    return out_features_csv



# --- CLI ------------------------------------------------------------
def _build_arg_parser():
    """
    Create and configure an ``argparse.ArgumentParser`` for the archetype CLI.

    The parser exposes three mutually exclusive input modes (``--csv``, ``--txt-dir``,
    ``--analysis-csv``), output and overwrite flags, repeatable ``--archetype`` paths,
    I/O parameters, gathering options for CSV/TXT inputs, and archetype scoring options.

    Returns
    -------
    argparse.ArgumentParser
        A parser with subcommands/flags matching the function parameters of
        :func:`analyze_with_archetypes`.
    """

    import argparse
    p = argparse.ArgumentParser(
        description="Archetype scoring into a single CSV (globals once + per-archetype blocks)."
    )

    # Input source (choose one)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", dest="csv_path", help="Source CSV to gather from")
    src.add_argument("--txt-dir", dest="txt_dir", help="Folder of .txt files to gather from")
    src.add_argument("--analysis-csv", dest="analysis_csv",
                     help="Use an existing analysis-ready CSV (skip gathering)")

    # Output
    p.add_argument("--out", dest="out_features_csv", default=None,
                   help="Output CSV (default: ./features/archetypes/<gathered_name>)")
    p.add_argument("--overwrite_existing", type=bool, default=False,
                    help="Do you want to overwrite the output file if it already exists?")

    # Archetype CSVs (repeatable)
    p.add_argument("--archetype", dest="archetype_csvs", action="append", required=True,
                   help="Path to an archetype CSV (repeat for multiple)")

    # I/O
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--delimiter", default=",")

    # CSV gather options
    p.add_argument("--text-col", dest="text_cols", action="append",
                   help="Text column (repeatable). Default: --text-col text")
    p.add_argument("--id-col", dest="id_cols", action="append",
                   help="ID column(s) (repeatable)")
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

    # Archetyper scoring options
    p.add_argument("--model-name", default="sentence-transformers/all-roberta-large-v1")
    p.add_argument("--mean-center-vectors", action="store_true", default=True)
    p.add_argument("--no-mean-center-vectors", dest="mean_center_vectors", action="store_false")
    p.add_argument("--fisher-z-transform", action="store_true", default=False)
    p.add_argument("--rounding", type=int, default=4)

    return p


def main():
    """
    Command-line entry point for archetype scoring.

    Parses arguments using :func:`_build_arg_parser`, normalizes list-like defaults,
    invokes :func:`analyze_with_archetypes`, and prints the resulting output path.

    Notes
    -----
    This function is executed when the module is run as a script:

        python -m taters.text.analyze_with_archetypes \
            --analysis-csv transcripts/X/X.csv \
            --archetype dictionaries/archetypes \
            --model-name sentence-transformers/all-roberta-large-v1
    """

    args = _build_arg_parser().parse_args()

    # Defaults for list-ish args
    text_cols = args.text_cols if args.text_cols else ["text"]
    id_cols = args.id_cols if args.id_cols else None
    group_by = args.group_by if args.group_by else None

    out = analyze_with_archetypes(
        csv_path=args.csv_path,
        txt_dir=args.txt_dir,
        analysis_csv=args.analysis_csv,
        out_features_csv=args.out_features_csv,
        overwrite_existing=args.overwrite_existing,
        archetype_csvs=args.archetype_csvs,
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
        mean_center_vectors=args.mean_center_vectors,
        fisher_z_transform=args.fisher_z_transform,
        rounding=args.rounding,
    )
    print(str(out))


if __name__ == "__main__":
    main()
