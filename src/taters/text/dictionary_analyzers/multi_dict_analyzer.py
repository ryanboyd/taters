from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union
import csv
import re
import warnings
from ...helpers.atomic import atomic_write
from ...helpers.csvio import widen_csv_field_limit

# we only ever touch ContentCoder's public API
from contentcoder.ContentCoder import ContentCoder

PathLike = Union[str, Path]

widen_csv_field_limit()


# ---- globals: we write these once, from the FIRST dictionary only -----------
GLOBAL_ONLY_FIELDS = {
    "WC",
    "BigWords",
    "Numbers",
    "AllPunct",
    "Period",
    "Comma",
    "QMark",
    "Exclam",
    "Apostro",
}

# ---- helpers ---------------------------------------------------------------

def _prefix_from_path(p: PathLike) -> str:
    stem = Path(p).stem
    return re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_") or "dict"

def _load_coders(dict_files: Sequence[PathLike]) -> List[Tuple[str, ContentCoder]]:
    """
    Load every readable dictionary; skip the broken ones out loud.

    One malformed file used to fail the whole step, so a run over eight
    dictionaries produced *zero* features because of one bare word list saved
    as `.dic`. A broken dictionary now costs exactly itself: it is skipped
    with a warning naming it and the reason (the run display collects
    warnings and prints them after the run), and the rest are scored. Only
    when NOTHING loads does the step fail -- there is no result to stand
    behind at that point.
    """
    coders: List[Tuple[str, ContentCoder]] = []
    problems: List[str] = []
    for d in dict_files:
        d = Path(d)
        try:
            _check_dictionary_shape(d)
            cc = ContentCoder(dicFilename=str(d), fileEncoding="utf-8-sig")
        except ValueError as e:
            problems.append(str(e))
            continue
        except Exception as e:
            # contentcoder's own errors are bare ("list index out of range")
            # and never tell you WHICH file, so we name it ourselves
            problems.append(
                f"'{d.name}' could not be read as a dictionary "
                f"({type(e).__name__}: {e})."
            )
            continue
        coders.append((_prefix_from_path(d), cc))

    if not coders:
        raise ValueError(
            "none of the dictionaries could be read: " + " | ".join(problems)
        )
    for problem in problems:
        warnings.warn(
            f"{problem} Skipped; continuing with the "
            f"{len(coders)} readable dictionar"
            f"{'y' if len(coders) == 1 else 'ies'}.",
            stacklevel=2,
        )
    return coders


def _check_dictionary_shape(path: Path) -> None:
    """
    Refuse a file that cannot be a dictionary, saying why.

    Delegates to the library's canonical checker (the same rule that now
    refuses these files at import time), because a file can still arrive here
    by CLI path without ever passing through the library. Deliberately without
    a kind: the kind's deep check *is* the ContentCoder construction on the
    next line of the caller, and running it twice per dictionary buys nothing.
    """
    from ...helpers.library import asset_problem

    problem = asset_problem(path)
    if problem:
        raise ValueError(problem)

def _partition_indices(header: Sequence[str], *, keep_globals: bool) -> Tuple[List[int], List[int]]:
    """
    Return (global_idxs, per_dict_idxs) for a single dictionary header.
    - global_idxs: indices for WC/BigWords/Numbers/punctuation — included only when keep_globals=True
    - per_dict_idxs: indices for 'Dic' + categories (i.e., everything NOT in GLOBAL_ONLY_FIELDS)
    """
    global_idxs: List[int] = []
    per_idxs: List[int] = []
    for i, name in enumerate(header):
        if name in GLOBAL_ONLY_FIELDS:
            if keep_globals:
                global_idxs.append(i)
        else:
            per_idxs.append(i)
    return global_idxs, per_idxs

# ---- public API ------------------------------------------------------------


def _score_cells(text: str, first_cc, g0, p0, plans, opts) -> List:
    """
    The feature cells for one text -- globals, then per-dictionary blocks.

    The one scoring implementation, shared by the serial path and the worker
    processes: identical output whatever the worker count, by construction.
    """
    relative_freq, drop_punct, rounding, retain_captures, wildcard_mem = opts
    res0 = first_cc.Analyze(
        text, relativeFreq=relative_freq, dropPunct=drop_punct,
        retainCaptures=retain_captures, returnTokens=False,
        wildcardMem=wildcard_mem,
    )
    v0 = list(first_cc.GetResultsArray(res0, rounding=rounding))
    cells: List = [v0[i] for i in g0]
    cells.extend(v0[i] for i in p0)
    for _pref, per_idxs, _h, cc in plans:
        res = cc.Analyze(
            text, relativeFreq=relative_freq, dropPunct=drop_punct,
            retainCaptures=retain_captures, returnTokens=False,
            wildcardMem=wildcard_mem,
        )
        v = list(cc.GetResultsArray(res, rounding=rounding))
        cells.extend(v[i] for i in per_idxs)
    return cells


def _plans_for(coders):
    """The static analysis plan: the first coder's split, and the rest."""
    first_pref, first_cc = coders[0]
    h0 = list(first_cc.GetResultsHeader())
    g0, p0 = _partition_indices(h0, keep_globals=True)
    plans: List[Tuple[str, List[int], List[str], ContentCoder]] = []
    for pref, cc in coders[1:]:
        h = list(cc.GetResultsHeader())
        _, p = _partition_indices(h, keep_globals=False)
        plans.append((pref, p, h, cc))
    return first_pref, first_cc, h0, g0, p0, plans


#: Per-worker scoring state, built once by the pool initializer.
_WORKER_STATE: dict = {}


def _init_score_worker(dict_files: List[str], opts: tuple) -> None:
    """Load the dictionaries once per worker -- quietly: the parent process
    already validated them and voiced any skip-and-warn messages once.

    Quiet includes stdout: contentcoder prints "Dictionary loaded." on every
    construction, and a worker's print goes to the *inherited* file
    descriptor -- straight past the parent's live display, which can only
    redirect its own process's stdout. Eight workers x two dictionaries
    scribbled sixteen lines across the progress bars (a real run)."""
    import contextlib
    import io

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        coders = _load_coders(dict_files)
    _first_pref, first_cc, _h0, g0, p0, plans = _plans_for(coders)
    _WORKER_STATE.update(first_cc=first_cc, g0=g0, p0=p0, plans=plans,
                         opts=opts)


def _score_in_worker(pair) -> List:
    _tid, text = pair
    s = _WORKER_STATE
    return _score_cells(text, s["first_cc"], s["g0"], s["p0"], s["plans"],
                        s["opts"])


def analyze_texts_to_csv(
    items: Iterable[Union[Tuple[str, str], Tuple[str, str, dict]]],
    dict_files: Sequence[PathLike],
    out_csv: PathLike,
    *,
    relative_freq: bool = True,
    drop_punct: bool = True,
    rounding: int = 4,
    retain_captures: bool = False,
    wildcard_mem: bool = True,
    id_col_name: str = "text_id",
    pass_through_cols: Sequence[str] = (),
    newline: str = "",
    encoding: str = "utf-8-sig",
    verbose: bool = True,
    workers: int = 0,
    on_progress=None,
    total_hint: int = 0,
) -> Path:
    """
    Write one wide CSV with per-text features.

    Header layout
    -------------
    [id_col_name] + [<pass_through_cols...>] + [globals_once] + [per-dict blocks]

    Parameters
    ----------
    items : Iterable[tuple]
        Yields either (text_id, text) or (text_id, text, meta_dict) where meta_dict
        provides values for pass_through_cols. Missing keys are emitted as empty strings.
    dict_files : sequence of path-like
        Paths to dictionaries (.dic/.dicx/.csv). Directories are expanded upstream.
    out_csv : path-like
        Output CSV path.
    relative_freq, drop_punct, rounding, retain_captures, wildcard_mem : see caller.
    id_col_name : str, default="text_id"
        Name for the identifier column.
    pass_through_cols : sequence of str, default=()
        Additional columns to inject immediately after id_col_name.
    newline, encoding : CSV writer options.

    workers : int, default=0
        Parallel scoring processes. Content coding is CPU-bound and grows
        super-linearly with text length, so a corpus of long documents scores
        several times faster in parallel. ``0`` means automatic: three-quarters
        of the logical cores; ``1`` turns parallelism off. The output file is
        **identical whatever the worker count**: rows are scored and written
        in input order.

    Notes
    -----
    - Streaming/bounded memory: rows are computed and written in order, with
      at most a small window of texts in flight.
    - Backward-compatible: callers that provide only (text_id, text) continue to work.
    """
    from ...helpers.parallel_map import ordered_parallel_map, resolve_workers

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        coders = _load_coders(dict_files)
    else:
        # under a live display, contentcoder's "Dictionary loaded." lines would
        # land right in the middle of the bars, so we swallow stdout here. its
        # warnings still get through untouched
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            coders = _load_coders(dict_files)
    if not coders:
        raise ValueError("No dictionaries provided.")

    dict_names = [pref for pref, _ in coders]
    first_pref, first_cc, h0, g0, p0, plans = _plans_for(coders)

    # ---- now we build the header plan (it's static from here on) -----------
    header: List[str] = [id_col_name]
    header.extend(list(pass_through_cols))  # up front, in the order we were asked
    header.extend([h0[i] for i in g0])
    header.extend([f"{first_pref}__{h0[i]}" for i in p0])
    for pref, _per_idxs, h, _cc in plans:
        _, p = _partition_indices(h, keep_globals=False)
        header.extend([f"{pref}__{h[i]}" for i in p])

    opts = (relative_freq, drop_punct, rounding, retain_captures, wildcard_mem)

    # the metas stay here in the parent; only (text_id, text) crosses over to
    # a worker. the id rides along so that the display can name the document
    # each worker is chewing on. `ordered_parallel_map` only pulls inputs a
    # small window ahead of the results it has handed back, so this list never
    # runs away from us, and result i always pairs up with ids_meta[i]
    from ...helpers.progress import FlightReporter

    ids_meta: List[Tuple[str, dict]] = []
    reporter = FlightReporter(on_progress, total_hint, "scoring documents") \
        if on_progress is not None else None

    def pairs() -> Iterable[Tuple[str, str]]:
        for it in items:
            if len(it) == 2:
                text_id, text = it  # type: ignore[misc]
                meta = {}
            elif len(it) == 3:
                text_id, text, meta = it  # type: ignore[misc]
                if not isinstance(meta, dict):
                    raise ValueError("Third element of items must be a dict of pass-through column values.")
            else:
                raise ValueError("Each item must be (text_id, text) or (text_id, text, meta_dict).")
            ids_meta.append((str(text_id), meta))
            yield (str(text_id), text)

    cells_stream = ordered_parallel_map(
        lambda pair: _score_cells(pair[1], first_cc, g0, p0, plans, opts),
        pairs(),
        workers=resolve_workers(workers),
        pool_fn=_score_in_worker,
        initializer=_init_score_worker,
        initargs=([str(d) for d in dict_files], opts),
        on_start=None if reporter is None else (lambda pair: reporter.start(pair[0])),
        on_finish=None if reporter is None else (lambda pair: reporter.finish(pair[0])),
    )

    # ---- lastly, we stream the rows out ------------------------------------
    with atomic_write(out_csv, newline=newline, encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, cells in enumerate(cells_stream):
            text_id, meta = ids_meta[i]
            if reporter is not None:
                reporter.consumed()
            if verbose:
                print(
                    f"Analyzing with dictionaries: {text_id}\n\t" + "\n\t".join(dict_names),
                    flush=True,
                )
            row_out: List[Union[str, float]] = [text_id]
            if pass_through_cols:
                row_out.extend([str(meta.get(c, "")) for c in pass_through_cols])
            row_out.extend(cells)
            writer.writerow(row_out)

    return out_csv

