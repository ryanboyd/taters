from pathlib import Path
from typing import Callable, Optional, Literal, Union, Sequence, Iterable, Tuple
import csv

from .dictionary_analyzers import multi_dict_analyzer as mda
from ..helpers.find_files import find_files
from ..helpers.progress import count_rows
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  # this is the word count that the categories are percentages of
                  bookkeeping=("WC",),
                  assets={"dict_paths": "dictionaries"})
def analyze_with_dictionaries(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,  # if given, we skip gathering
    gathered_csv: Optional[Union[str, Path]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,
    workers: int = 0,  # if the file already exists, let's not overwrite by default

    # ----- Dictionaries -----
    dict_paths: Sequence[Union[str, Path]], # LIWC2007 (.dic) or LIWC-22 (.dicx, .csv)

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS ======
    # these only matter when csv_path is provided
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
    # these only matter when txt_dir is provided
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
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
    gathered_csv : str or pathlib.Path, optional
        Where to write the intermediate "analysis-ready" table built from
        ``csv_path`` or ``txt_dir``.

        By default it lands beside the *source* -- which means analyzing a
        spreadsheet in someone's Downloads folder writes a file into their
        Downloads folder. Pass this to keep the intermediate with the rest of a
        run's output instead. Ignored when ``analysis_csv`` is given, because
        then no gathering happens.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)`` so a UI can show a
        real bar instead of a spinner. Injected automatically by the pipeline
        runner for any step function that declares this parameter. See
        :mod:`taters.helpers.progress` for the contract.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/dictionary/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and return the path.
        This also controls the intermediate analysis-ready CSV: when ``True``, it is rebuilt
        from the current source instead of reusing a stale copy from an earlier run.
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
    workers : int, default=0
        Parallel processes for reading documents and scoring texts. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.
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
        out_features_csv = Path.cwd() / "features" / "dictionary" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and Path(out_features_csv).is_file():
        print("Dictionary content coding output file already exists; returning existing file.")
        return out_features_csv


    # 2) validate the dictionaries
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
                # find .dic/.dicx/.csv under this folder (recursive), stable order
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

        # 3) stream the analysis-ready CSV into the analyzer → features CSV
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
            # if id_cols were requested, make sure they exist up-front (fail fast)
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


    # we use multi_dict_analyzer as the middle layer (new API). it pulls the
    # generator above lazily, writes as it goes, and owns the progress story:
    # "scoring documents", with one sub-bar per document in flight on displays
    # that can show it
    total_rows = count_rows(analysis_ready, on_progress=on_progress)

    # there's one shared rule for what rides along beside text_id -- see
    # resolve_passthrough_columns: same order of preference that every per-row
    # analyzer uses, and the same two columns that never get carried
    from ..helpers.row_map import resolve_passthrough_columns

    with analysis_ready.open("r", newline="", encoding=encoding) as _fh:
        _header = csv.DictReader(_fh).fieldnames or []
    passthrough = resolve_passthrough_columns(
        _header, id_cols=id_cols, group_by=group_by,
        analysis_ready=analysis_ready)

    mda.analyze_texts_to_csv(
        items=_iter_items_from_csv_with_meta(analysis_ready, pass_through_cols=passthrough),
        dict_files=dict_paths,
        out_csv=out_features_csv,
        relative_freq=relative_freq,
        drop_punct=drop_punct,
        rounding=rounding,
        retain_captures=retain_captures,
        wildcard_mem=wildcard_mem,
        id_col_name="text_id",
        pass_through_cols=passthrough,  # these land right after text_id
        workers=workers,
        on_progress=on_progress,
        total_hint=total_rows,
        encoding=encoding,
        verbose=on_progress is None,
    )


    return out_features_csv



# --- CLI ------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_with_dictionaries,
    description='ContentCoder: multi-dictionary coding into one CSV (globals once + per-dict blocks).',
    aliases={
        'csv_path': ['--csv'],
        'dict_paths': ['--dict'],
        'out_features_csv': ['--out'],
    },
    legacy={
        '--no-drop-punct': ['--drop-punct', 'false'],
        '--no-include-source-path': ['--include-source-path', 'false'],
        '--no-recursive': ['--recursive', 'false'],
        '--no-relative-freq': ['--relative-freq', 'false'],
        '--no-wildcard-mem': ['--wildcard-mem', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
