"""
Word norms, one row per document.

A content-coding dictionary asks "how much of this text is about X", and the
answer is a rate: what share of the words landed in the category. A norm set
asks something different. Concreteness norms, valence norms, age of acquisition
-- those attach a *rating* to a word, and the number people want is the average
rating of the words that had one. Not a rate. The two are nowhere near each
other: score Brysbaert concreteness as a rate and you get about 0.3 instead of
about 3.2, and it shrinks as the text gets longer, which is the opposite of
what a rating should do.

That is the whole reason this is a separate step with its own library rather
than a flag on the dictionary step. A norm file and a dictionary file look
identical -- both are a term per row with a number per column -- so nothing
about the file itself would stop somebody pointing one at the wrong scorer and
getting a plausible, wrong answer. Keeping them in separate libraries is what
makes that mistake impossible instead of merely unlikely.

Every mean comes with a **coverage** column beside it: the share of the text
that had a rating at all. A mean is only worth the number of words it was taken
over, and two texts can both come back at 3.2 with one of them having had forty
rated words and the other three. Coverage is the column that tells them apart,
and it is bookkeeping rather than a measure, so the statistics stage keeps it
out of the predictors.

A text with no rated words in it gets an **empty cell, not a zero**. It does
not have a concreteness of nought; it does not have one at all, and writing 0.0
there would drag every average that touched it downward.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import (Callable, Iterable, Literal, Optional, Sequence, Tuple,
                    Union)

from .dictionary_analyzers import multi_dict_analyzer as mda
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.feature_columns import ColumnSpec
from ..helpers.find_files import find_files
from ..helpers.progress import count_rows
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready

__all__ = ["analyze_with_norms", "FEATURE_COLUMNS"]

#: Accepted on the command line and when a folder is expanded. Norm sets are
#: `.csv` rather than `.dicx` so that a glance at the folder says which library
#: a file belongs to.
NORM_SUFFIXES = {"csv"}

FEATURE_COLUMNS = ColumnSpec(
    label="Word norms",
    dynamic="one column per rating in the norm file the user picked, plus a "
            "coverage column for each",
)


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  # WC is the denominator the coverage columns are shares of.
                  # The coverage columns are named after whatever ratings the
                  # user's file happens to hold, so what gets declared is their
                  # shape rather than their names -- they are counts of how much
                  # text was rated, and a model that regressed an outcome on
                  # "how much of this text was in the concreteness table" would
                  # be fitting the corpus rather than the construct.
                  bookkeeping=("WC", "*" + mda.COVERAGE_SUFFIX),
                  assets={"norm_paths": "norms"})
def analyze_with_norms(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) --
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,
    gathered_csv: Optional[Union[str, Path]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,
    workers: int = 0,

    # ----- Norm sets -----
    norm_paths: Sequence[Union[str, Path]],

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS ======
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
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== ANALYZER OPTIONS ======
    drop_punct: bool = True,
    rounding: int = 4,
    wildcard_mem: bool = True,
) -> Path:
    """
    Score texts against one or more word-norm tables and write a features CSV.

    Each norm file contributes one column per rating, holding the **mean rating
    of the words in the text that had one**, plus a companion coverage column
    holding the share of the text that was rated. A text with no rated words
    gets an empty cell rather than a zero.

    The input modes are the same three every text step accepts:

    1. ``analysis_csv`` -- a prebuilt file with ``text_id`` and ``text``.
    2. ``csv_path`` -- gather text from an arbitrary CSV.
    3. ``txt_dir`` -- gather text from a folder of documents.

    Parameters
    ----------
    csv_path : str or pathlib.Path, optional
        Source CSV to gather from. Mutually exclusive with ``txt_dir`` and
        ``analysis_csv``.
    txt_dir : str or pathlib.Path, optional
        Folder of documents to gather from.
    analysis_csv : str or pathlib.Path, optional
        Prebuilt analysis-ready CSV with ``text_id`` and ``text``.
    gathered_csv : str or pathlib.Path, optional
        Where to write the intermediate analysis-ready table.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)``.
    out_features_csv : str or pathlib.Path, optional
        Output path. Defaults to ``./features/norms/<analysis_ready_name>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, skip and return the existing path.
    workers : int, default=0
        Parallel scoring processes. ``0`` means automatic.
    norm_paths : Sequence[str or pathlib.Path]
        One or more norm files (``.csv``) or folders containing them.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSV files.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, the column(s) holding the text.
    id_cols : Sequence[str] or None, optional
        Columns to carry through beside ``text_id``.
    mode : {"concat", "separate"}, default="concat"
        With several text columns, join them into one text or score each
        column as its own row.
    group_by : Sequence[str] or None, optional
        Join the text of rows that share these columns before scoring.
    delimiter : str, default=","
        Delimiter for reading and writing CSV files.
    joiner : str, default=" "
        What goes between chunks of text when they are joined.
    num_buckets : int, default=512
        Temporary hash buckets used while gathering a large CSV.
    max_open_bucket_files : int, default=64
        How many bucket files stay open at once while gathering.
    tmp_root : str or pathlib.Path or None, optional
        Where the temporary gathering files go.
    recursive : bool, default=True
        When gathering from a folder, look in subfolders too.
    pattern : str
        Which files to pick up when gathering from a folder.
    id_from : {"stem", "name", "path"}, default="stem"
        How to name each gathered document.
    include_source_path : bool, default=True
        Keep each document's path as a column when gathering from a folder.
    drop_punct : bool, default=True
        Drop punctuation before matching.
    rounding : int, default=4
        Decimal places for numeric output.
    wildcard_mem : bool, default=True
        Cache wildcard matches. Faster, and changes no output.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Raises
    ------
    FileNotFoundError
        If an input or norm path does not exist.
    ValueError
        If input modes are misconfigured, or a norm file cannot be read.

    Examples
    --------
    >>> analyze_with_norms(
    ...     csv_path="data/essays.csv", text_cols=["text"],
    ...     norm_paths=["Concreteness - Brysbaert.csv"],
    ... )
    PosixPath('.../features/norms/essays.csv')

    Notes
    -----
    There is no ``relative_freq`` option here, unlike the dictionary step. A
    mean rating is a mean rating; there is no rate version of it to ask for.
    """
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

    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "norms" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print("Word norm output file already exists; returning existing file.")
        return out_features_csv

    norm_files = _expand_norm_inputs(norm_paths)

    total_rows = count_rows(analysis_ready, on_progress=on_progress)

    from ..helpers.row_map import resolve_passthrough_columns

    with analysis_ready.open("r", newline="", encoding=encoding) as _fh:
        _header = csv.DictReader(_fh).fieldnames or []
    passthrough = resolve_passthrough_columns(
        _header, id_cols=id_cols, group_by=group_by,
        analysis_ready=analysis_ready)

    mda.analyze_texts_to_csv(
        items=_iter_items(analysis_ready, encoding=encoding,
                          delimiter=delimiter, pass_through_cols=passthrough),
        dict_files=norm_files,
        out_csv=out_features_csv,
        # the whole point: means over the matched words, with coverage beside
        weighted_mean=True,
        label="word norms",
        drop_punct=drop_punct,
        rounding=rounding,
        wildcard_mem=wildcard_mem,
        id_col_name="text_id",
        pass_through_cols=passthrough,
        workers=workers,
        on_progress=on_progress,
        total_hint=total_rows,
        encoding=encoding,
        verbose=on_progress is None,
    )

    return out_features_csv


def _expand_norm_inputs(paths: Sequence[Union[str, Path]]) -> list:
    """
    Normalize norm inputs into a unique, ordered list of readable files.

    Each one is put through the library's own norms check rather than merely
    having its extension looked at, because the rule that matters -- no
    leftover ``_intercept`` row -- is invisible from the filename. A file can
    reach this function by CLI path without ever passing through the library's
    import screen, and that is exactly the path on which a bad file would
    otherwise go unnoticed.
    """
    from ..helpers.library import KINDS

    check = KINDS["norms"].deep_check
    out, seen = [], set()
    for p in map(Path, paths):
        if p.is_dir():
            found = find_files(root_dir=p, extensions=[".csv"], recursive=True,
                               absolute=True, sort=True)
            candidates = [Path(f).resolve() for f in found]
        else:
            if not p.exists():
                raise FileNotFoundError(f"Norm file not found: {p}")
            if p.suffix.lower().lstrip(".") not in NORM_SUFFIXES:
                raise ValueError(f"Unsupported norm file extension: {p.name}")
            candidates = [p.resolve()]

        for fp in candidates:
            if fp in seen:
                continue
            problem = check(fp) if check else ""
            if problem:
                raise ValueError(problem)
            out.append(fp)
            seen.add(fp)

    if not out:
        raise ValueError(
            "No norm files found. Supply .csv rating tables, or folders "
            "containing them.")
    return out


def _iter_items(path: Path, *, encoding: str, delimiter: str,
                pass_through_cols: Sequence[str]) -> Iterable[Tuple[str, str, dict]]:
    """Stream ``(text_id, text, meta)`` out of an analysis-ready CSV."""
    wanted = list(pass_through_cols or [])
    with path.open("r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames or []
        if "text_id" not in fields or "text" not in fields:
            raise ValueError(
                f"Expected columns 'text_id' and 'text' in {path}; "
                f"found {fields}")
        missing = [c for c in wanted if c not in fields]
        if missing:
            raise ValueError(
                f"Requested id_cols not present in analysis-ready CSV "
                f"{path}: {missing}")
        for row in reader:
            yield (str(row.get("text_id", "") or ""),
                   str(row.get("text", "") or ""),
                   {c: str(row.get(c, "") or "") for c in wanted})


# --- CLI --------------------------------------------------------------------

CLI = CliSpec(
    analyze_with_norms,
    description="Word norms: mean rating per text, with coverage, into one CSV.",
    aliases={
        "csv_path": ["--csv"],
        "norm_paths": ["--norms"],
        "out_features_csv": ["--out"],
    },
    legacy={
        "--no-drop-punct": ["--drop-punct", "false"],
        "--no-include-source-path": ["--include-source-path", "false"],
        "--no-recursive": ["--recursive", "false"],
        "--no-wildcard-mem": ["--wildcard-mem", "false"],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
