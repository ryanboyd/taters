# taters/text/analyze_sentiment_vader.py
"""
VADER sentiment: a rule-based sentiment measure, one row per text.

VADER is a lexicon plus a small set of rules -- boosters ("very"), negation,
punctuation, capitalization, contrastive "but" -- tuned on social media and
serviceable well beyond it. It is not a model: there is nothing to train and
nothing to fit, so the same text always scores the same, and the scoring is
fast enough that a corpus of a hundred thousand rows is a coffee break rather
than an afternoon.

Four columns come out of it, named as VADER names them but prefixed, because
``neg`` and ``pos`` alone would be anybody's guess in a feature table with two
hundred columns beside them:

``vader_neg``, ``vader_neu``, ``vader_pos``
    The share of the text falling in each band. They sum to 1.
``vader_compound``
    The single summary score, -1 (most negative) to +1 (most positive). This
    is the one most analyses use. It is *not* the mean of the other three --
    it is a normalized sum of the valences, so a long mild text and a short
    vehement one can land in the same place.

A text is scored whole, the way the other per-text measures here work. VADER
was designed on sentence-length material, so a long document's compound score
is a blunter instrument than a tweet's -- if sentence-level scores matter to
your question, split the text into sentences before gathering and each will
get its own row.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.text_gather import resolve_analysis_ready

PathLike = Union[str, Path]

#: The four scores, in the order VADER's own documentation lists them, with a
#: prefix so they survive being merged beside everything else.
METRICS = ("vader_neg", "vader_neu", "vader_pos", "vader_compound")

#: VADER's keys for the same four, in the same order.
_KEYS = ("neg", "neu", "pos", "compound")


def _require_vader():
    """The analyzer class, or a refusal that says what to install."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer
    except Exception as e:  # pragma: no cover - only without the package
        raise RuntimeError(
            'The "vaderSentiment" package is required for VADER sentiment.\n'
            "It ships with Taters, so a missing copy usually means a partial "
            "install:\n\n"
            "    pip install vaderSentiment\n"
        ) from e


def _build(lexicon_file: Optional[PathLike], emoji_lexicon: Optional[PathLike]):
    """
    One analyzer, with custom lexicons where they were asked for.

    VADER resolves its defaults relative to its own package folder and takes
    them as *names*, not paths, so handing it an absolute path to a file
    somewhere else works only because it falls back to the name as given. We
    check the file ourselves first: its own failure for a missing lexicon is
    an IOError naming a path inside site-packages, which sends people looking
    in the wrong place entirely.
    """
    analyzer = _require_vader()
    kwargs: Dict[str, Any] = {}
    for name, value in (("lexicon_file", lexicon_file),
                        ("emoji_lexicon", emoji_lexicon)):
        if value is None:
            continue
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        kwargs[name] = str(path)
    return analyzer(**kwargs)


def _score_text(analyzer, text: str, rounding: int) -> list:
    """The four scores for one text, in METRICS order."""
    scores = analyzer.polarity_scores(text or "")
    return [round(float(scores.get(k, 0.0)), rounding) for k in _KEYS]


#: Per-process state for scoring workers. The analyzer parses a 7,500-line
#: lexicon on construction, so it is built once per worker in the initializer
#: rather than per row -- and never pickled, which it would not survive.
_VADER_WORKER: Dict[str, Any] = {}


def _init_vader_worker(lexicon_file, emoji_lexicon, rounding: int) -> None:
    _VADER_WORKER["analyzer"] = _build(lexicon_file, emoji_lexicon)
    _VADER_WORKER["rounding"] = int(rounding)


def _vader_in_worker(pair):
    _tid, text = pair
    return _score_text(_VADER_WORKER["analyzer"], (text or "").strip(),
                       _VADER_WORKER["rounding"])


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",))
def analyze_sentiment_vader(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) --
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    on_progress: Optional[Callable[..., None]] = None,

    # ----- Output -----
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    workers: int = 0,

    # ----- VADER options -----
    lexicon_file: Optional[PathLike] = None,
    emoji_lexicon: Optional[PathLike] = None,
    rounding: int = 4,

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
    tmp_root: Optional[PathLike] = None,

    # ====== TXT FOLDER GATHER OPTIONS ======
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== passthrough control (optional) ======
    pass_through_cols: Optional[Sequence[str]] = None,
) -> Path:
    """
    Score each row's text with VADER and write a wide features CSV.

    The function supports exactly one of three input modes:

    1. ``analysis_csv`` -- a prebuilt file with at least ``text_id`` and ``text``.
    2. ``csv_path`` -- gather text from an arbitrary CSV using ``text_cols``
       (and optional ``id_cols``/``group_by``).
    3. ``txt_dir`` -- gather text from a folder of documents.

    If ``out_features_csv`` is omitted the default is
    ``./features/sentiment_vader/<analysis_ready_filename>``.

    Columns
    -------
    ``vader_neg``, ``vader_neu``, ``vader_pos``
        The proportion of the text in each band; these sum to 1.
    ``vader_compound``
        The summary score, -1 to +1, normalized so that texts of different
        lengths are comparable. Most analyses want this one.

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
        Where to write the intermediate analysis-ready table. By default it
        lands beside the *source*; pass this to keep it with the run's output.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)`` so a UI can show
        a real bar. Injected by the pipeline runner.
    out_features_csv : str or pathlib.Path, optional
        Output path. Defaults to
        ``./features/sentiment_vader/<analysis_ready_filename>``.
    overwrite_existing : bool, default=False
        When ``False`` and the output exists, skip and return the path.
    workers : int, default=0
        Scoring processes; 0 picks a sensible number for the row count.
    lexicon_file : str or pathlib.Path, optional
        A replacement VADER lexicon, in VADER's own tab-separated format
        (``token<TAB>mean valence<TAB>standard deviation<TAB>[ratings]``).
        Omit it for the lexicon VADER ships, which is what published VADER
        scores are computed with -- a custom lexicon makes the numbers yours
        rather than VADER's, which is sometimes exactly what is wanted and
        should always be reported.
    emoji_lexicon : str or pathlib.Path, optional
        A replacement emoji description file, same format. Omit for VADER's.
    rounding : int, default=4
        Decimal places for each score.
    encoding : str, default="utf-8-sig"
        Encoding for reading and writing CSVs.
    text_cols : Sequence[str], default=("text",)
        When gathering from a CSV, the column(s) holding text.
    id_cols : Sequence[str], optional
        ID columns carried into grouping when gathering from a CSV.
    mode : {"concat", "separate"}, default="concat"
        With several text columns: join them, or give each its own row.
    group_by : Sequence[str], optional
        Grouping keys used during CSV gathering.
    delimiter : str, default=","
        Column separator of the *input* spreadsheet.
    joiner : str, default=" "
        String joining several text columns under ``mode="concat"``.
    num_buckets, max_open_bucket_files, tmp_root
        Gathering internals for very large inputs.
    recursive : bool, default=True
        Recurse into subfolders under ``txt_dir``.
    pattern : str
        Filename pattern for ``txt_dir``.
    id_from : {"stem", "name", "path"}, default="stem"
        How a file becomes a ``text_id``.
    include_source_path : bool, default=True
        Carry each file's path into the gathered table.
    pass_through_cols : Sequence[str], optional
        Columns to copy from the analysis-ready table into the output.

    Returns
    -------
    pathlib.Path
        The features CSV.

    Raises
    ------
    FileNotFoundError
        If an input, or a named lexicon, is missing.
    ValueError
        If input modes are misconfigured or required columns are absent.
    RuntimeError
        If ``vaderSentiment`` is not installed.

    Notes
    -----
    Blank text scores zero on all four, which is VADER's own answer for it
    rather than a value invented here.
    """
    _require_vader()   # fail early, with a message that says what to do

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
        out_features_csv = (Path.cwd() / "features" / "sentiment_vader"
                            / analysis_ready.name)
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_features_csv.is_file():
        print("VADER output file already exists; returning existing file: "
              f"{out_features_csv}")
        return out_features_csv

    # the gatherer writes the analysis-ready table with commas whatever the
    # source used, so `delimiter` stops here and is NOT passed on
    with analysis_ready.open("r", newline="", encoding=encoding) as fin:
        header_fields = csv.DictReader(fin).fieldnames or []
    if "text_id" not in header_fields or "text" not in header_fields:
        raise ValueError(
            f"Expected columns 'text_id' and 'text' in {analysis_ready}; "
            f"found {header_fields}")

    from ..helpers.row_map import resolve_passthrough_columns

    passthrough_cols = resolve_passthrough_columns(
        header_fields, pass_through_cols=pass_through_cols, id_cols=id_cols,
        group_by=group_by, analysis_ready=analysis_ready)
    fieldnames = ["text_id", *passthrough_cols, *METRICS]

    # one analyzer for the inline path; the workers build their own in the
    # initializer, because it holds a parsed lexicon and would not pickle
    inline = _build(lexicon_file, emoji_lexicon)

    from ..helpers.parallel_map import pool_workers
    from ..helpers.row_map import map_text_rows

    with atomic_write(out_features_csv, newline="", encoding=encoding) as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row, values in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pool_workers(workers, n_rows),
                message="scoring sentiment", on_progress=on_progress,
                inline_fn=lambda pair: _score_text(
                    inline, (pair[1] or "").strip(), rounding),
                pool_fn=_vader_in_worker,
                initializer=_init_vader_worker,
                initargs=(lexicon_file, emoji_lexicon, rounding)):
            out_row: Dict[str, Any] = {
                "text_id": row.get("text_id"),
                **{k: row.get(k, "") for k in passthrough_cols},
                **dict(zip(METRICS, values)),
            }
            writer.writerow(out_row)
    return out_features_csv


# ---------------------------------------------------------------------------
# command line -- derived from the function above; see helpers.cliargs.CliSpec
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_sentiment_vader,
    description="Score texts for sentiment with VADER and write a features CSV.",
    aliases={
        "csv_path": ["--csv"],
        "out_features_csv": ["--out"],
    },
    legacy={
        "--no-include-source-path": ["--include-source-path", "false"],
        "--no-recursive": ["--recursive", "false"],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
