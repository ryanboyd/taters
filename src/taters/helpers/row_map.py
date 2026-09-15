"""
One ordered, pooled pass over an analysis-ready CSV, with live sub-bars.

Every per-row text scorer has the same shape: read ``text_id,text`` rows,
compute something CPU-bound per row, write results in file order, and show
which documents are in flight. This driver owns that shape once -- the
dictionary scorer, the n-gram counter, readability, lexical richness, the
DTM scan and the MEM apply all differ only in the function they run.

Determinism contract inherited from `ordered_parallel_map`: results come
back in file order whatever the worker count, so output files are identical
at workers=1 and workers=8. Memory stays bounded: at most the pool's
in-flight window of rows (2x workers) is held at once.
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

from .parallel_map import ordered_parallel_map
from .progress import FlightReporter, count_rows

__all__ = ["map_text_rows", "resolve_passthrough_columns"]


def map_text_rows(
    analysis_ready: Path,
    *,
    encoding: str,
    workers,
    message: str,
    on_progress,
    inline_fn: Callable[[Tuple[str, str]], Any],
    pool_fn: Optional[Callable[[Tuple[str, str]], Any]] = None,
    initializer: Optional[Callable] = None,
    initargs: Tuple = (),
    delimiter: str = ",",
) -> Iterator[Tuple[dict, Any]]:
    """
    Yield ``(row, result)`` for every row of the table, in file order.

    ``inline_fn`` / ``pool_fn`` receive ``(text_id, text)`` -- the picklable
    unit of work -- and the full row dict rides back alongside each result so
    callers keep their passthrough columns. ``workers`` is either a resolved
    process count or a callable given the row total (so a caller can apply
    `pool_workers`' small-job rule, or a model-aware rule, without paying a
    second counting pass). The wired :class:`FlightReporter` names the phase
    (``message``) and the documents in flight, which is what draws one
    sub-bar per document under the step's row.
    """
    total = count_rows(analysis_ready, on_progress=on_progress)
    if callable(workers):
        workers = workers(total)
    reporter = FlightReporter(on_progress, total, message)
    with Path(analysis_ready).open("r", newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames or []
        if "text_id" not in fields or "text" not in fields:
            raise ValueError(
                f"Expected columns 'text_id' and 'text' in {analysis_ready}; "
                f"found {fields}"
            )

        pending: deque = deque()

        def pairs():
            for row in reader:
                pending.append(row)
                yield (str(row.get("text_id", "") or ""),
                       str(row.get("text", "") or ""))

        for result in ordered_parallel_map(
                inline_fn, pairs(),
                workers=workers,
                pool_fn=pool_fn,
                initializer=initializer,
                initargs=initargs,
                on_start=lambda pair: reporter.start(pair[0]),
                on_finish=lambda pair: reporter.finish(pair[0])):
            reporter.consumed()
            yield pending.popleft(), result


def resolve_passthrough_columns(header_fields, *, pass_through_cols=None,
                                id_cols=None, group_by=None,
                                analysis_ready="") -> list:
    """
    Which columns of the analysis-ready table ride along beside ``text_id``.

    One rule for every per-row analyzer, where there used to be three copies
    of it (readability, lexical richness, word count) and three shorter ones
    that disagreed with them (dictionaries, archetypes, parts of speech).

    The order of preference: the columns the caller named; else the id
    columns; else the grouping columns; else every column that is not the
    text, the id, or the gatherer's own ``group_count`` -- a count of rows
    combined, which is not a measurement of anything and, carried into a
    feature table, was one keystroke from becoming a predictor.

    Two columns are never carried, whoever asks. ``text_id`` itself, which
    is written first by hand. And the one column ``text_id`` was composed
    from: with a single id column (or a single grouping column) the gatherer
    writes that value *as* ``text_id``, so carrying the source column too
    put two identical columns -- ``text_id`` and ``ResponseId`` -- into
    every feature table of every run (a real report). With two or more
    composing columns each is a genuine part of the identity and stays.

    ``source_col`` always rides along when present: it is what tells a
    headline row from a body row under ``mode="separate"``, and without it
    those rows are indistinguishable.
    """
    header = list(header_fields or [])
    fields = set(header)
    composers = list(group_by or []) or list(id_cols or [])
    requested = list(pass_through_cols or [])
    if not requested:
        requested = list(id_cols or []) or list(group_by or [])
    if not requested:
        requested = [c for c in header
                     if c not in ("text", "text_id", "group_count")]
    redundant = {"text_id"}
    if len(composers) == 1:
        redundant.add(composers[0])
    requested = [c for c in requested if c not in redundant]
    if "source_col" in fields and "source_col" not in requested:
        requested.append("source_col")
    missing = [c for c in requested if c not in fields]
    if missing:
        raise ValueError(
            f"Requested pass-through columns not present in analysis-ready "
            f"CSV {analysis_ready}: {missing}")
    return list(dict.fromkeys(requested))
