# taters/text/analyze_readability.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union, Sequence, Literal, Dict, Any
import csv


from ..helpers.atomic import atomic_write
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec


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


# ---- Per-document scoring (runs inline or in a worker process) ---------------

def _score_text(metrics: Sequence[str], txt: str) -> list:
    """Every metric's value for one text, best-effort per metric."""
    textstat = _require_textstat()
    values = []
    for name in metrics:
        fn = getattr(textstat, name, None)
        if fn is None:
            values.append(None)
            continue
        try:
            values.append(fn(txt))
        except Exception:
            values.append(None)
    return values


#: Per-process state for scoring workers: the metric list arrives once via
#: the initializer, not pickled per text.
_READ_WORKER: Dict[str, Any] = {}


def _init_readability_worker(metrics: Sequence[str]) -> None:
    _READ_WORKER["metrics"] = list(metrics)


def _readability_in_worker(pair):
    _tid, text = pair
    return _score_text(_READ_WORKER["metrics"], (text or "").strip())


# ---- Core API ----------------------------------------------------------------

@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  # these are just raw sizes; the indices we compute from them
                  # are the actual measures
                  bookkeeping=("lexicon_count", "sentence_count", "char_count",
                               "syllable_count", "difficult_words"))
def analyze_readability(
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
    workers: int = 0,

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

    # ====== NEW: passthrough control (optional) ======
    pass_through_cols: Optional[Sequence[str]] = None
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
        real bar instead of a spinner.

        ``total`` is the row count of the analysis-ready table, which is fully
        written before measuring starts, so it is known up front. Until it is,
        ``total`` is ``None`` and ``done`` is a running tally -- the long silent
        passes (reading the input, counting its rows) report through the same
        callback with a ``message`` saying which one is running.

        Injected automatically by the pipeline runner for any step function that
        declares this parameter.
    out_features_csv : str or pathlib.Path, optional
        Output file path. If ``None``, defaults to
        ``./features/readability/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        If ``False`` and the output file already exists, skip processing and return the path.
        This also controls the intermediate analysis-ready CSV: when ``True``, it is rebuilt
        from the current source instead of reusing a stale copy from an earlier run.
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
        Column separator of the *input* spreadsheet. The gathered table and
        the output are always comma-separated.
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

    pass_through_cols : Sequence[str] or None, optional
        Extra input columns to copy into the output beside ``text_id``.
    workers : int, default=0
        Parallel processes for reading documents. ``0`` means automatic:
        three-quarters of the logical cores; ``1`` turns parallelism off. Output files are
        identical whatever the worker count.

    Returns
    -------
    pathlib.Path
        Path to the written features CSV.

    Output layout
    -------------
    The output CSV starts with:
        text_id, <pass-through columns...>, <metrics...>

    Pass-through behavior:
    - If `pass_through_cols` is provided, those columns are included immediately
      after `text_id` in that order.
    - Else if `id_cols` were supplied during gathering, they are included in that order.
    - Else (backward compatible), *all* non-`text` columns in the analysis-ready CSV
      are copied through (e.g., `source`, `speaker`, etc.).

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
    _require_textstat()   # fail early, with a message that says what to do

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
        out_features_csv = Path.cwd() / "features" / "readability" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print(f"Readability output file already exists; returning existing file: {out_features_csv}")
        return out_features_csv

    # 3) our list of metrics
    metrics = [
        "flesch_reading_ease",
        "smog_index",
        "flesch_kincaid_grade",
        "coleman_liau_index",
        "automated_readability_index",
        "dale_chall_readability_score",
        "difficult_words",
        "linsear_write_formula",
        "gunning_fog",
        "text_standard",
        "spache_readability",
        "syllable_count",
        "lexicon_count",
        "sentence_count",
        "char_count",
        "avg_sentence_length",
        "avg_syllables_per_word",
        "avg_letter_per_word",
    ]

    # 4) figure out the output's shape from the input's header alone. the
    # gatherer writes the analysis-ready table with commas no matter what the
    # source used, so `delimiter` stops here and does NOT get passed on. when
    # we did pass it on, a ";" spreadsheet blew up with "Expected columns
    # 'text_id' and 'text' ... found ['text_id,id,text']" -- on the one step
    # that almost every study picks
    with analysis_ready.open("r", newline="", encoding=encoding) as fin:
        header_fields = csv.DictReader(fin).fieldnames or []

    if "text_id" not in header_fields or "text" not in header_fields:
        raise ValueError(
            f"Expected columns 'text_id' and 'text' in {analysis_ready}; found {header_fields}"
        )

    # there's one shared rule for what rides along beside text_id -- see
    # resolve_passthrough_columns for the order of preference and for the two
    # columns it never carries
    from ..helpers.row_map import resolve_passthrough_columns

    passthrough_cols = resolve_passthrough_columns(
        header_fields, pass_through_cols=pass_through_cols, id_cols=id_cols,
        group_by=group_by, analysis_ready=analysis_ready)
    fieldnames = ["text_id", *passthrough_cols, *metrics]

    # 5) scoring is CPU-bound per row and the rows don't depend on each other,
    # so we run it on the shared pooled-row driver. that gives us results in
    # file order no matter the worker count, plus one live sub-bar per
    # document in flight. `_score_text` is the one scorer that both the inline
    # path and the workers run
    from ..helpers.parallel_map import pool_workers
    from ..helpers.row_map import map_text_rows

    with atomic_write(out_features_csv, newline="", encoding=encoding) as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row, values in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pool_workers(workers, n_rows),
                message="measuring readability", on_progress=on_progress,
                inline_fn=lambda pair: _score_text(metrics, (pair[1] or "").strip()),
                pool_fn=_readability_in_worker,
                initializer=_init_readability_worker,
                initargs=(list(metrics),)):
            out_row: Dict[str, Any] = {
                "text_id": row.get("text_id"),
                **{k: row.get(k, "") for k in passthrough_cols},
                **dict(zip(metrics, values)),
            }
            writer.writerow(out_row)
    return out_features_csv


# ---- CLI ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_readability,
    description='Compute textstat readability metrics for an analysis-ready CSV.',
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
