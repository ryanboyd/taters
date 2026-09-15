"""
Order-preserving parallel map for CPU-bound per-item work.

The determinism contract: for the same input sequence, the output sequence is
**identical whatever the worker count** -- one worker, eight, or the inline
serial path. No binning or reassembly is needed for that: results are yielded
in *submission* order (a deque of futures consumed from the left), so a slow
item simply holds the line while its successors finish behind it.

Two deliberate choices:

* ``spawn``, not ``fork``. The parent process may have torch (or CUDA) loaded
  by the time a text step runs, and forking such a process is a documented
  source of deadlocks. Spawned workers import only what their function needs.
* A bounded in-flight window rather than ``executor.map``, which consumes its
  entire input iterable up front -- for a corpus of large documents that means
  holding every text in the task queue at once.

This is for pure-Python, CPU-bound work (document parsing, dictionary
scoring). It is the wrong tool for model-backed steps -- a process per worker
would put a model copy per worker in memory; their lever is batching.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple, TypeVar

__all__ = ["max_workers", "auto_workers", "resolve_workers", "pool_workers",
           "ordered_parallel_map"]

T = TypeVar("T")
R = TypeVar("R")


def _worker_pid(_item) -> int:
    """Test support: proves work actually crossed a process boundary."""
    return os.getpid()


def _spawn_safe() -> bool:
    """
    Can a spawned child re-import this program's ``__main__``?

    Spawn re-runs the main module in every worker, and when the parent is a
    REPL, a ``python - <<script`` heredoc, or anything else without an
    importable main, every worker dies on startup with a FileNotFoundError
    for ``<stdin>`` and the pool comes back broken -- found the hard way.
    Those callers silently get the serial path instead, which is merely slow.
    """
    import sys

    main = sys.modules.get("__main__")
    if main is None:
        return False
    if getattr(main, "__spec__", None) is not None:
        return True
    main_file = getattr(main, "__file__", None)
    try:
        return bool(main_file) and Path(main_file).exists()
    except OSError:
        return False


def _in_worker() -> bool:
    """
    Is this process itself a multiprocessing child?

    Spawn re-imports the parent's ``__main__``; a user script that calls an
    analyzer at top level without an ``if __name__ == "__main__"`` guard is
    then re-executed *inside every worker* -- and if those re-executions
    opened pools of their own, the result is a fork bomb (reproduced while
    benchmarking). A child therefore always takes the serial path: the
    accidental re-run is wasteful, but it terminates.
    """
    import multiprocessing

    return multiprocessing.parent_process() is not None

def max_workers() -> int:
    """
    The most workers a request may ask for: this machine's logical cores.

    One process per logical core is where more stops helping CPU-bound
    Python; past it, extra workers only add spawn cost and contention. This
    is the ceiling the wizard shows and ``resolve_workers`` enforces.
    """
    return os.cpu_count() or 1


def auto_workers() -> int:
    """
    The automatic worker count: three-quarters of the machine.

    "Leave a quarter for the human." Saturating every logical core keeps a
    laptop's foreground apps gasping for the whole run; a batch job the user
    launched should finish fast, not own the machine. ``max(1, ...)`` so a
    small machine still gets its one worker.
    """
    cores = max_workers()
    return max(1, cores - cores // 4)


def resolve_workers(requested: int) -> int:
    """
    Turn a ``workers`` request into a process count.

    ``0`` (the default everywhere) means automatic: three-quarters of the
    logical cores (``auto_workers``). Anything below zero is treated as 1.
    An explicit positive number is honored up to the machine's core count
    (``max_workers``) -- beyond that, more processes than cores is strictly
    worse for this CPU-bound work, so the ask is clamped, not obeyed.
    """
    if requested > 0:
        return min(requested, max_workers())
    if requested < 0:
        return 1
    return auto_workers()


#: Below this many items, automatic parallelism stays serial: spawning a pool
#: costs seconds (each worker is a fresh interpreter importing its world), and
#: a small job is finished serially before the pool would come up.
MIN_ITEMS_TO_POOL = 25


def pool_workers(requested: int, n_items: int) -> int:
    """
    ``resolve_workers``, minus the spawn tax on small jobs.

    Automatic (``0``) stays serial under ``MIN_ITEMS_TO_POOL`` items. An
    explicit ask is honored regardless -- someone who typed a number decided,
    and ten enormous documents may well be worth ten workers.
    """
    if requested == 0 and n_items < MIN_ITEMS_TO_POOL:
        return 1
    return resolve_workers(requested)


def ordered_parallel_map(
    inline_fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    pool_fn: Optional[Callable[[T], R]] = None,
    initializer: Optional[Callable] = None,
    initargs: Tuple = (),
    on_start: Optional[Callable[[T], None]] = None,
    on_finish: Optional[Callable[[T], None]] = None,
) -> Iterator[R]:
    """
    Yield ``fn(item)`` for every item, in input order.

    Parameters
    ----------
    inline_fn
        Used when ``workers <= 1``: the plain serial path, no processes, no
        pickling -- byte-for-byte the behavior before parallelism existed.
    pool_fn
        The function workers run; must be module-level (spawn pickles it by
        name). Defaults to ``inline_fn``. Separate because workers often read
        per-process state built by ``initializer`` where the inline path uses
        state the caller already holds.
    initializer, initargs
        Run once per worker on start -- the place to build the expensive
        per-process state (loaded dictionaries) exactly once, not per item.
    on_start, on_finish
        Lifecycle hooks, for showing the work: ``on_start(item)`` fires when
        an item is handed to a worker (or begun inline), ``on_finish(item)``
        when its computation completes -- which, under parallelism, is NOT the
        order results are yielded in: a fast item behind a slow one finishes
        early and its hook says so, which is what lets a display retire its
        bar instead of showing fifteen "working" files behind one straggler.
        ``on_finish`` may run on an executor-internal thread; keep it small
        and thread-safe.
    """
    if workers <= 1 or not _spawn_safe() or _in_worker():
        for item in items:
            if on_start is not None:
                on_start(item)
            result = inline_fn(item)
            if on_finish is not None:
                on_finish(item)
            yield result
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    fn = pool_fn if pool_fn is not None else inline_fn
    context = multiprocessing.get_context("spawn")
    window = workers * 2

    def _submit(pool, item):
        if on_start is not None:
            on_start(item)
        future = pool.submit(fn, item)
        if on_finish is not None:
            future.add_done_callback(lambda _f, _item=item: on_finish(_item))
        return future

    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=initializer,
                             initargs=initargs) as pool:
        inflight: deque = deque()
        for item in items:
            inflight.append(_submit(pool, item))
            if len(inflight) >= window:
                yield inflight.popleft().result()
        while inflight:
            yield inflight.popleft().result()
