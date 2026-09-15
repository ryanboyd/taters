"""
Row-level progress reporting for analysis steps.

A GLOBAL pipeline step is a single call: the runner hands over and gets control
back at the end, so it cannot count anything from outside. A step that *can*
count itself says so by declaring an ``on_progress`` parameter, which the
pipeline runner then injects automatically.

The contract
------------
``on_progress(done, total, message=None, unit=None)``

* ``total`` is an int once the size of the job is known, and ``done`` is a
  position within it.
* ``total`` is ``None`` while the size is still being worked out. ``done`` is
  then a running tally, and ``message`` says which pass is running -- reading
  the input, counting its rows. On a large file those passes take long enough
  to look like a hang, and a number climbing is the only visible proof that
  anything is happening.
* ``unit`` names what is being counted when it is not rows. The only value with
  a meaning today is ``"seconds"``, used by transcription, where the numbers are
  a position in the recording: ``252/771`` would be read as segments, which is
  not what it is, while ``4:12/12:51`` cannot be misread.

The last two arguments are optional in both directions -- a sink that predates
them still works, because nothing is obliged to send them.

This module exists so the five text analyzers share one implementation of that
pattern rather than five copies that drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
from .csvio import widen_csv_field_limit

__all__ = ["count_rows", "Ticker", "announce", "FlightReporter"]

# (the counting pass reports every 1,000 records -- see count_rows. records,
# not lines: one record can be a whole book, so our old 50,000-line interval
# could mean we never report at all for an entire corpus.)


def announce(on_progress: Optional[Callable[..., None]], message: str) -> None:
    """Name the phase that is about to run, with no size known yet."""
    if on_progress is not None:
        on_progress(0, None, message)


def count_rows(
    path: Path,
    *,
    on_progress: Optional[Callable[..., None]] = None,
    encoding: str = "utf-8-sig",
    every: int = 1_000,
) -> int:
    """
    Count the data *records* in a CSV, reporting progress as it goes.

    Records, not lines: document text carries embedded newlines inside its
    quoted field, and counting physical lines told a 2,300-paper run it had
    3.2 million rows to do -- a denominator so wrong the bar read as broken.

    Always counts, and only *reports* when something is watching. It used
    to return 0 with no sink, on the theory that the count was only ever a
    progress bar's denominator -- but :func:`taters.helpers.row_map.map_text_rows`
    sizes its worker pool from it, so every direct API call (no sink) ran
    on one worker while the same call from the wizard ran on twelve. One
    ``csv.reader`` sweep is far cheaper than the work it sizes.

    The header is not counted. A file that cannot be read comes back as 0
    rather than raising -- a progress bar is not worth failing a run over.
    """
    import csv

    widen_csv_field_limit()

    seen = 0
    try:
        with Path(path).open("r", encoding=encoding, newline="") as fh:
            for seen, _ in enumerate(csv.reader(fh), 1):
                if on_progress is not None and seen % every == 0:
                    on_progress(seen, None, "counting rows")
    except (OSError, UnicodeDecodeError, csv.Error):
        return 0

    return max(0, seen - 1)


class Ticker:
    """
    Counts work as it happens and reports it.

    Safe to use unconditionally: with no ``on_progress`` every method is a
    no-op, so the analysis code reads the same whether anyone is watching.
    """

    __slots__ = ("_on_progress", "_total", "_done")

    def __init__(self, on_progress: Optional[Callable[..., None]], total: int = 0) -> None:
        self._on_progress = on_progress
        self._total = int(total)
        self._done = 0
        if on_progress is not None:
            on_progress(0, self._total or None, None)

    def tick(self, n: int = 1, message: Optional[str] = None) -> None:
        """
        Record ``n`` more units of work done.

        ``message`` names the unit -- "scoring <text_id>" -- for steps whose
        rows are wildly uneven: one two-million-character document can take
        minutes where its neighbors take milliseconds, and a bar that just
        sits there unnamed reads as a hang rather than as one slow paper.
        """
        if self._on_progress is None:
            return
        self._done += n
        self._on_progress(self._done, self._total or None, message)

    @property
    def done(self) -> int:
        return self._done


def _sink_takes(sink, name: str) -> bool:
    """Whether an on_progress sink can accept keyword ``name``.

    The documented contract is (done, total, message, unit) with the last two
    optional *in both directions* -- so anything newer, like ``inflight``, is
    sent only to sinks that can take it. Sending it blindly would break every
    older callback."""
    import inspect

    try:
        params = inspect.signature(sink).parameters
    except (TypeError, ValueError):
        return False
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class FlightReporter:
    """
    Progress for a phase whose work is spread over parallel workers.

    Reports the usual (done, total, message) plus -- to sinks that accept it
    -- ``inflight``: the names currently being worked on, so a display can
    show one sub-bar per file the way item steps do for ffmpeg or Whisper.
    ``start``/``finish`` may be called from executor threads; ``consumed``
    is the parent's in-order tally. All three re-report immediately.
    """

    def __init__(self, on_progress: Optional[Callable[..., None]],
                 total: int, message: str) -> None:
        import threading

        self._on_progress = on_progress
        self._total = int(total)
        self._message = message
        self._done = 0
        self._inflight: list = []
        self._lock = threading.Lock()
        self._takes_inflight = (on_progress is not None
                                and _sink_takes(on_progress, "inflight"))
        self._report()

    def start(self, name: str) -> None:
        if self._on_progress is None:
            return
        with self._lock:
            self._inflight.append(name)
        self._report()

    def finish(self, name: str) -> None:
        if self._on_progress is None:
            return
        with self._lock:
            try:
                self._inflight.remove(name)
            except ValueError:
                pass
        self._report()

    def consumed(self, n: int = 1) -> None:
        if self._on_progress is None:
            return
        self._done += n
        self._report()

    def _report(self) -> None:
        if self._on_progress is None:
            return
        if self._takes_inflight:
            with self._lock:
                names = list(self._inflight)
            self._on_progress(self._done, self._total or None, self._message,
                              None, inflight=names)
        else:
            self._on_progress(self._done, self._total or None, self._message)
