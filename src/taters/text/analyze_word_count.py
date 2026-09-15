# taters/text/analyze_word_count.py
"""
How many words each text has -- the one measure almost every study needs and
nothing else here provides on its own.

It exists because of filtering. "Ignore anything under 25 words" is the
commonest thing a researcher wants to do before analyzing language, and a
three-word answer makes most language measures meaningless. Until now that
filter could only name a column some *other* step happened to write, so
asking for it meant knowing that readability calls its word count
``lexicon_count`` -- and picking it without readability selected cost a whole
run (a real report).

The counting is contentcoder's, not a fresh regex. It is already a declared
dependency, its tokenizer is the one the dictionary step scores with, and a
word count that disagrees with the dictionary categories computed from the
same text would be its own small mystery. contentcoder needs *a* dictionary
to construct, so it gets a placeholder one that matches nothing: the counts
this reads -- ``WC`` -- are computed before any category is scored.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.doc_text import DOCUMENT_PATTERN
from ..helpers.text_gather import (resolve_analysis_ready)
from ..helpers.provenance import TEXT_GRAIN, TEXT_INPUT, records_settings
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]

#: A dictionary that matches nothing, so contentcoder can be constructed
#: without one of the user's. Two categories' worth of header and a single
#: entry no text contains: the word counts are computed regardless of what
#: the dictionary holds, and a placeholder keeps this step from needing an
#: import from the library.
_EMPTY_DICTIONARY = "%\n1\tPlaceholder\n%\nzzzz_taters_placeholder_zzzz\t1\n"

#: What the output column is called. Plain words, because it is offered in a
#: filter menu to people who do not know what `lexicon_count` means.
WORD_COUNT_COLUMN = "word_count"


def _coder():
    """One contentcoder instance, built quietly.

    It prints "Dictionary loaded." on construction, which lands in the middle
    of a live progress display; the dictionary step muzzles it the same way.
    """
    import contextlib
    import io

    try:
        from contentcoder.ContentCoder import ContentCoder  # type: ignore
    except Exception:  # pragma: no cover - contentcoder is a core dependency
        try:
            from ContentCoder import ContentCoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "The 'contentcoder' package is required to count words.\n"
                "It ships with Taters; reinstall with `pip install taters`."
            ) from e

    with contextlib.redirect_stdout(io.StringIO()):
        return ContentCoder(dictString=_EMPTY_DICTIONARY,
                            fileEncoding="utf-8-sig")


_CODER = None


def _init_worker() -> None:
    """Build the coder once per worker process, not once per row."""
    global _CODER
    _CODER = _coder()


def _count_in_worker(pair) -> int:
    return _count_words(_CODER, (pair[1] or ""))


def _count_words(coder, text: str) -> int:
    """One text's word count. Empty text is zero words, not a missing value:
    a blank answer really does have none, and blanking it would quietly
    exempt it from every length filter."""
    text = (text or "").strip()
    if not text:
        return 0
    return int(coder.Analyze(text, relativeFreq=False)["WC"])


@records_settings(binding=TEXT_INPUT, grain=TEXT_GRAIN,
                  outputs=("out_features_csv",),
                  # the whole table is just a length to filter on, not a feature
                  bookkeeping=("word_count",))
def analyze_word_count(
    *,
    csv_path: Optional[PathLike] = None,
    txt_dir: Optional[PathLike] = None,
    analysis_csv: Optional[PathLike] = None,
    gathered_csv: Optional[PathLike] = None,
    out_features_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    on_progress: Optional[Callable[..., None]] = None,
    workers: int = 0,
    encoding: str = "utf-8-sig",
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: str = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[PathLike] = None,
    recursive: bool = True,
    pattern: str = DOCUMENT_PATTERN,
    id_from: str = "stem",
    include_source_path: bool = True,
    pass_through_cols: Optional[Sequence[str]] = None,
) -> Path:
    """
    Count the words in each text.

    Parameters
    ----------
    csv_path, txt_dir, analysis_csv
        Exactly one input: a spreadsheet, a folder of documents, or an
        already-gathered analysis-ready table.
    out_features_csv
        Where the counts go. Defaults to
        ``features/word_count/<gathered name>``.
    overwrite_existing
        When False (default) and the output exists, it is returned untouched.
    workers
        How many texts to count at once; 0 means choose automatically.
    text_cols, id_cols, mode, group_by, delimiter, joiner
        How to read a spreadsheet: which columns hold the text, which
        identify a row, whether to join several text columns or measure them
        separately, and which columns to combine rows on. Set for you by the
        pipeline's gather settings.
    pattern, recursive, id_from, include_source_path
        How to read a folder of documents: which files to match, whether to
        descend into subfolders, what to build ``text_id`` from, and whether
        to record each file's path.
    pass_through_cols
        Columns from the gathered table to copy into the output, so the
        counts join back to the rest of your data. Defaults to the id
        columns, or to everything but the text.
    gathered_csv, num_buckets, max_open_bucket_files, tmp_root, encoding
        Where the intermediate gathered table goes and how it is built.

    Returns
    -------
    Path
        A CSV of ``text_id``, any carried columns, and ``word_count``.

    Notes
    -----
    The count is contentcoder's ``WC``: the same tokenization the
    dictionary-scoring step uses, so a word count sitting beside dictionary
    percentages was computed on the same words those percentages came from.
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
        out_features_csv = (Path.cwd() / "features" / "word_count"
                            / analysis_ready.name)
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite_existing and out_features_csv.is_file():
        print(f"Word count output file already exists; returning existing "
              f"file: {out_features_csv}")
        return out_features_csv

    with analysis_ready.open("r", newline="", encoding=encoding) as f:
        header_fields = next(csv.reader(f), [])

    from ..helpers.row_map import resolve_passthrough_columns

    passthrough_cols = resolve_passthrough_columns(
        header_fields, pass_through_cols=pass_through_cols, id_cols=id_cols,
        group_by=group_by, analysis_ready=analysis_ready)
    fieldnames = ["text_id", *passthrough_cols, WORD_COUNT_COLUMN]

    from ..helpers.parallel_map import pool_workers
    from ..helpers.row_map import map_text_rows

    inline = _coder()
    with atomic_write(out_features_csv, newline="", encoding=encoding) as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row, count in map_text_rows(
                analysis_ready, encoding=encoding,
                workers=lambda n_rows: pool_workers(workers, n_rows),
                message="counting words", on_progress=on_progress,
                inline_fn=lambda pair: _count_words(inline, pair[1]),
                pool_fn=_count_in_worker,
                initializer=_init_worker,
                initargs=()):
            out_row: Dict[str, Any] = {
                "text_id": row.get("text_id"),
                **{k: row.get(k, "") for k in passthrough_cols},
                WORD_COUNT_COLUMN: count,
            }
            writer.writerow(out_row)
    return out_features_csv


# ---------------------------------------------------------------------------
# command line -- we derive this from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases and legacy flags are the spellings that
# the old hand-written parser used; we keep them so that every documented
# invocation still works
# ---------------------------------------------------------------------------

CLI = CliSpec(
    analyze_word_count,
    description='Count the words in each text, with contentcoder.',
    aliases={
        'csv_path': ['--csv'],
        'out_features_csv': ['--out'],
    },
    legacy={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
