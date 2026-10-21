from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import re
from . import tables
from .progress import Ticker, announce
from .provenance import records_settings
from .cliargs import CliSpec
from .feature_columns import ColumnSpec

FEATURE_COLUMNS = ColumnSpec(
    label="Gathered per-file tables",
    dynamic="`<column>__<stat>` from whatever the gathered tables held",
)

PathLike = Union[str, Path]

#: The statistics with a running form: one pass over the rows, a few numbers
#: per group per column, nothing held. Anything else (a median, a quantile)
#: needs every value and goes through pandas -- see `aggregate_features`.
RUNNING_STATS = ("mean", "std", "var", "sum", "count", "min", "max")


# ---------------------------
# File discovery (single root)
# ---------------------------

def _iter_csv_files(
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
):
    """
    Yield CSV files under a root directory, optionally searching recursively.

    Parameters
    ----------
    root_dir : PathLike
        Root directory to search, or a single CSV file. If a file is passed
        and it matches `pattern`, it is yielded.
    pattern : str, default="*.csv"
        Glob pattern for file discovery.
    recursive : bool, default=True
        If True, search with ``Path.rglob``; otherwise use ``Path.glob`` in the
        top-level folder only.

    Yields
    ------
    pathlib.Path
        Absolute paths to files that match the pattern.

    Notes
    -----
    If `root_dir` does not exist, nothing is yielded. If `root_dir` is a file
    and matches `pattern`, it is yielded even though the typical use is a folder.
    """

    root = Path(root_dir)
    if not root.exists():
        return
    if root.is_file():
        # not the intended mode, but let's be permissive: if it's a file and
        # matches, yield it.
        if root.match(pattern):
            yield root.resolve()
        return
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def _sources(root_dir: PathLike, *, pattern: str, recursive: bool) -> List[tables.TableRef]:
    """
    The tables under a root: the files matching ``pattern`` under a folder
    (or the one file), or, for a database address, that table itself when
    it exists and otherwise every table Taters wrote under it as a prefix.
    """
    root = tables.parse_ref(root_dir)
    if not root.is_db:
        return [tables.TableRef(p) for p in
                _iter_csv_files(root.path, pattern=pattern, recursive=recursive)]
    if tables.exists(root):
        return [root]
    import fnmatch

    # the glob was written for files, so we match it against the table's
    # name as if it were one
    found = [t for t in tables.tables_under(root)
             if fnmatch.fnmatch(t.stem + ".csv", pattern)]
    if not recursive:
        found = [t for t in found if t.parent == root.logical]
    return found


def _default_out(root: tables.TableRef, out_csv: Optional[PathLike]) -> tables.TableRef:
    """``<root_dir_parent>/<root_dir_name>.csv`` beside a folder; beside a
    database table, ``<table>_gathered``."""
    if out_csv is not None:
        return tables.parse_ref(out_csv)
    if root.is_db:
        return root.sibling(f"{root.stem}_gathered")
    return tables.TableRef(root.path.parent / f"{root.path.name}.csv")


# ---------------------------
# Loading / concatenation
# ---------------------------

def _next_available(base: str, existing: set) -> str:
    """Return base.N with the smallest N>=1 not in existing."""
    n = 1
    candidate = f"{base}.{n}"
    while candidate in existing:
        n += 1
        candidate = f"{base}.{n}"
    return candidate


def _origin_columns(columns: Sequence[str], *,
                    add_source_path: bool = False) -> Tuple[List[str], List[str]]:
    """
    One table's columns as they will be named in the gathered table.

    Returns ``(lead, renamed)``: the pipeline's own ``"source"`` (and
    ``"source_path"`` when asked for) that lead the table, and the input
    columns in their own order, an input column by one of those names
    renamed ``<name>.N`` so nothing is overwritten.
    """
    cols = [str(c) for c in columns]

    def _make_room(name: str) -> None:
        nonlocal cols
        if name not in cols:
            return
        taken = set(cols)
        renamed = []
        for col in cols:
            if col == name:
                col = _next_available(name, taken)
                taken.add(col)
            renamed.append(col)
        cols = renamed

    # the pipeline's own `source` (and `source_path`) lead the table; if an
    # input column has the same name, we keep it (renamed) rather than clobber it.
    _make_room("source")
    lead = ["source"]
    if add_source_path:
        _make_room("source_path")
        lead.append("source_path")
    return lead, cols


def _origin_values(ref: tables.TableRef, lead: Sequence[str]) -> List[str]:
    """What the lead columns hold for one table: its stem, and where it is."""
    values = [ref.stem]
    if "source_path" in lead:
        values.append(str(ref.path.resolve()) if not ref.is_db else str(ref))
    return values


def _union(headers: Sequence[Sequence[str]]) -> List[str]:
    """Every column once, in order of first appearance -- what stacking
    tables with different columns gives."""
    seen: Dict[str, bool] = {}
    for header in headers:
        for col in header:
            seen.setdefault(col, True)
    return list(seen)


def _read_plans(files: Sequence[tables.TableRef], *, delimiter: str, encoding: str,
                add_source_path: bool, verbose: bool, tag: str) -> list:
    """
    Each table's header, named for the gathered table: ``(ref, lead,
    renamed)``. A table whose header cannot be read is warned about and
    left out, as it always was.
    """
    plans = []
    for ref in files:
        try:
            columns = tables.columns_of(ref, encoding=encoding, delimiter=delimiter)
            if not columns:
                raise ValueError("no header row")
        except Exception as e:
            if verbose:
                print(f"[{tag}] WARNING: failed to read {ref}: {e}")
            continue
        lead, renamed = _origin_columns(columns, add_source_path=add_source_path)
        plans.append((ref, lead, renamed))
    return plans


# ---------------------------------
# Figure out what we're going to do
# ---------------------------------

def make_plan(
    *,
    group_by: Sequence[str],
    per_file: bool = True,
    stats: Sequence[str] = ("mean", "std"),
    exclude_cols: Sequence[str] = (),
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    dropna: bool = False,
) -> AggregationPlan:
    """
    Create an :class:`AggregationPlan` from simple arguments.

    Parameters
    ----------
    group_by : Sequence[str]
        Grouping key(s) to use (e.g., ``["speaker"]``).
    per_file : bool, default=True
        If True, group within files by including ``"source"`` in group keys.
    stats : Sequence[str], default=("mean", "std")
        Statistical reductions to compute per numeric column.
    exclude_cols : Sequence[str], default=()
        Columns to drop prior to feature selection.
    include_regex : str or None, default=None
        Regex to include feature columns by name.
    exclude_regex : str or None, default=None
        Regex to exclude feature columns by name.
    dropna : bool, default=False
        Drop rows with NA in any group key. Off by default so rows with a
        missing key are still reported rather than silently discarded.

    Returns
    -------
    AggregationPlan
        A configured plan instance for :func:`aggregate_features`.
    """

    return AggregationPlan(
        group_by=tuple(group_by),
        per_file=per_file,
        stats=tuple(stats),
        exclude_cols=tuple(exclude_cols),
        include_regex=include_regex,
        exclude_regex=exclude_regex,
        dropna=dropna,
    )


def _promote_inner_keys(columns: Sequence[str], keys: Sequence[str]) -> List[str]:
    """
    When both a file-level key (e.g., 'source') and an inner CSV key
    that was renamed to 'source.N' exist, promote the inner key to the
    base name and demote/rename the file-level key to 'file_<key>'.

    This is intended for per_file=False: we want to aggregate across files
    using the *inner* keys rather than the injected file stem. Works on the
    column names alone and returns them renamed, in the same positions.
    """
    def _unique(col: str, existing: set) -> str:
        if col not in existing:
            return col
        i = 1
        while f"{col}.{i}" in existing:
            i += 1
        return f"{col}.{i}"

    cols = [str(c) for c in columns]
    existing = set(cols)

    for base in keys:
        # go looking for numbered variants like base.1, base.2, ...
        numbered = [c for c in cols if c == f"{base}.1" or c.startswith(f"{base}.")]
        has_base = base in existing
        if numbered:
            numbered.sort()
            inner_col = numbered[0]              # first after sorting, so it's stable
            if has_base:
                # demote the file-level base to 'file_<base>' (uniquified if needed)
                demoted = _unique(f"file_{base}", existing)
                cols = [demoted if c == base else c for c in cols]
                existing.discard(base)
                existing.add(demoted)
            # now we promote the inner column up to the base name
            cols = [base if c == inner_col else c for c in cols]
            existing.discard(inner_col)
            existing.add(base)

    return cols


# -----------------------------------------------------
# Gather features sub-functions, abstracted and unified
# -----------------------------------------------------

def feature_gather(
    *,
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
    # toggle aggregation; when True you must pass a plan (or plan_args below)
    aggregate: bool = False,
    plan: Optional[AggregationPlan] = None,
    # optional “quick plan” args (only used if plan=None and aggregate=True)
    group_by: Optional[Sequence[str]] = None,
    per_file: bool = True,
    stats: Sequence[str] = ("mean", "std"),
    exclude_cols: Sequence[str] = (),
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    dropna: bool = False,
    # output
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Single entry point to concatenate or aggregate feature CSVs from one folder.

    If ``aggregate=False``, CSVs are concatenated with origin metadata
    (see :func:`gather_csvs_to_one`). If ``aggregate=True``, numeric feature
    columns are aggregated per the provided or constructed plan
    (see :func:`aggregate_features`).

    Parameters
    ----------
    root_dir : PathLike
        Folder containing per-item CSVs (or a single CSV file), or a table
        in a database (``db://<connection>/<table>``).
    pattern : str, default="*.csv"
        Glob pattern for selecting CSV files.
    recursive : bool, default=True
        Recurse into subdirectories when True.
    delimiter : str, default=","
        CSV delimiter.
    encoding : str, default="utf-8-sig"
        CSV encoding.
    add_source_path : bool, default=False
        If True, include a ``"source_path"`` column in outputs.
    aggregate : bool, default=False
        Toggle aggregation mode. If False, files are concatenated.
    plan : AggregationPlan or None, default=None
        Explicit plan for aggregation. Required if ``aggregate=True`` and
        ``group_by`` is not given.
    group_by : Sequence[str] or None, default=None
        Quick-plan keys. Used only when ``aggregate=True`` and ``plan`` is None.
    per_file : bool, default=True
        Quick-plan flag; include ``"source"`` in grouping keys to aggregate per file.
    stats : Sequence[str], default=("mean", "std")
        Quick-plan statistics to compute per numeric column.
    exclude_cols : Sequence[str], default=()
        Quick-plan columns to drop before numeric selection.
    include_regex : str or None, default=None
        Quick-plan regex to include feature columns by name.
    exclude_regex : str or None, default=None
        Quick-plan regex to exclude feature columns by name.
    dropna : bool, default=False
        Quick-plan NA handling for group keys. When True, rows whose group key
        is missing are dropped before aggregating.
    out_csv : PathLike or None, default=None
        Output table. If None, defaults to
        ``<root_dir_parent>/<root_dir_name>.csv``.
    overwrite_existing : bool, default=False
        If False and `out_csv` exists, the existing path is returned without
        recomputation.

    Returns
    -------
    pathlib.Path
        Path to the resulting CSV (the ref, for a database table).

    Raises
    ------
    ValueError
        If ``aggregate=True`` and neither ``plan`` nor ``group_by`` is provided.

    See Also
    --------
    gather_csvs_to_one : Concatenate CSVs with origin metadata.
    aggregate_features : Aggregate numeric columns according to a plan.
    """

    if not aggregate:
        return gather_csvs_to_one(
            on_progress=on_progress,
            root_dir=root_dir,
            pattern=pattern,
            recursive=recursive,
            delimiter=delimiter,
            encoding=encoding,
            add_source_path=add_source_path,
            out_csv=out_csv,
            overwrite_existing=overwrite_existing,
            verbose=verbose,
        )

    # aggregate=True
    if plan is None:
        if not group_by:
            raise ValueError("When aggregate=True, you must provide 'plan' or 'group_by'.")
        plan = make_plan(
            group_by=group_by,
            per_file=per_file,
            stats=stats,
            exclude_cols=exclude_cols,
            include_regex=include_regex,
            exclude_regex=exclude_regex,
            dropna=dropna,
        )

    return aggregate_features(
        on_progress=on_progress,
        root_dir=root_dir,
        pattern=pattern,
        recursive=recursive,
        delimiter=delimiter,
        encoding=encoding,
        add_source_path=add_source_path,
        plan=plan,
        out_csv=out_csv,
        overwrite_existing=overwrite_existing,
        verbose=verbose,
    )


#: The gather steps turn per-item acoustics and per-utterance embeddings into
#: one joinable table, and their own settings are plumbing. The ones that
#: decided the numbers -- the Whisper model, the embedding model, the
#: acoustics thresholds -- live upstream, which is why `root_dir` is binding:
#: the chain walk follows it and folds in the records beside the files it
#: read. One tuple for both writers: only the aggregating one carried it, so
#: the plain concatenation (the acoustics summary) wrote no record and a
#: model fitted on it could never be checked.
GATHER_BINDING = ("root_dir", "pattern", "recursive", "delimiter",
                  "encoding", "add_source_path")


@records_settings(binding=GATHER_BINDING, outputs=("out_csv",))
def gather_csvs_to_one(
    *,
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Concatenate many CSVs into a single CSV with origin metadata.

    Each input table is streamed through, a leading ``"source"`` column is
    inserted (and optionally ``"source_path"``), and rows are appended. The
    final table's columns are the union of every input's, in order of first
    appearance, with ``"source"`` (and, if present, ``"source_path"``)
    leading.

    Parameters
    ----------
    root_dir : PathLike
        Folder containing CSVs, a single CSV file, or a database table.
    pattern : str, default="*.csv"
        Glob pattern for selecting files.
    recursive : bool, default=True
        Recurse into subdirectories when True.
    delimiter : str, default=","
        CSV delimiter.
    encoding : str, default="utf-8-sig"
        CSV encoding for read/write.
    add_source_path : bool, default=False
        If True, include absolute path in ``"source_path"``.
    out_csv : PathLike or None, default=None
        Output path. If None, defaults to
        ``<root_dir_parent>/<root_dir_name>.csv``.
    overwrite_existing : bool, default=False
        If False and `out_csv` exists, return it without recomputation.

    Returns
    -------
    pathlib.Path
        Path to the written CSV (the ref, for a database table).

    Raises
    ------
    FileNotFoundError
        If no files match the pattern under `root_dir`.
    RuntimeError
        If files were found but none could be read successfully.

    Notes
    -----
    Cells are copied as the text they are. Column order from inputs is
    preserved after the leading origin columns.
    """

    root = tables.parse_ref(root_dir)
    out_ref = _default_out(root, out_csv)

    if tables.exists(out_ref) and not overwrite_existing:
        if verbose:
            print(f"Aggregated feature output file already exists; returning existing file: {out_ref}")
        return tables.handle(out_ref)

    files = _sources(root, pattern=pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {root}")

    # we tick once per table read: a big embeddings folder took a minute under
    # a bare spinner, which looks an awful lot like a hang.
    announce(on_progress, "reading feature tables")
    ticker = Ticker(on_progress, len(files))
    plans = _read_plans(files, delimiter=delimiter, encoding=encoding,
                        add_source_path=add_source_path, verbose=verbose, tag="gather")
    if not plans:
        raise RuntimeError("No CSVs could be read successfully.")

    # the headers decide the shape; the rows then stream straight through
    # from each table into the one output, so nothing is held but the row
    # in hand. make sure 'source' is first (and 'source_path' next, if we have it)
    lead = ["source"] + (["source_path"] if add_source_path else [])
    header = lead + [c for c in _union([[*ld, *cols] for _, ld, cols in plans])
                     if c not in lead]
    at = {c: i for i, c in enumerate(header)}
    with tables.write_table(out_ref, header, encoding=encoding,
                            hints={c: "text" for c in lead}) as writer:
        for ref, ld, cols in plans:
            origin = _origin_values(ref, ld)
            slots = [at[c] for c in cols]
            lead_slots = [at[c] for c in ld]
            for cells in tables.read_cells(ref, encoding=encoding, delimiter=delimiter):
                row = [""] * len(header)
                for i, v in zip(lead_slots, origin):
                    row[i] = v
                for i, v in zip(slots, cells):
                    row[i] = v
                writer.writerow(row)
            ticker.tick(message="reading feature tables")
    return tables.handle(out_ref)


# ---------------------------
# Aggregation
# ---------------------------

@dataclass
class AggregationPlan:
    """
    Plan describing how numeric feature columns should be aggregated.

    Parameters
    ----------
    group_by : Sequence[str]
        One or more column names used as grouping keys (e.g., ``["speaker"]``).
    per_file : bool, default=True
        If True, include ``"source"`` in the grouping keys to aggregate within
        each input file; if False, aggregate across all files globally.
    stats : Sequence[str], default=("mean", "std")
        Statistical reductions to compute for each numeric feature column.
        ``mean``, ``std``, ``var``, ``sum``, ``count``, ``min`` and ``max``
        run in one streaming pass; any other name (``"median"``, say) is
        passed to ``pandas.DataFrame.agg`` and holds the table in memory.
    exclude_cols : Sequence[str], default=()
        Columns to drop before filtering/selecting numeric features (e.g.,
        timestamps or free text).
    include_regex : str or None, default=None
        Optional regex; if provided, only columns matching this pattern are kept
        (after excluding `exclude_cols`).
    exclude_regex : str or None, default=None
        Optional regex; if provided, columns matching this pattern are removed
        (after applying `include_regex`, if any).
    dropna : bool, default=False
        Whether to drop rows with NA in any of the group-by keys before grouping.
        The default keeps them, so rows with a missing key land in their own
        clearly-labeled group instead of vanishing from the output.

    Notes
    -----
    This plan is consumed by :func:`aggregate_features`. Column filtering happens
    before numeric selection; only columns that remain and can be coerced to numeric
    will be aggregated.
    """

    group_by: Sequence[str]
    per_file: bool = True
    stats: Sequence[str] = ("mean", "std")
    exclude_cols: Sequence[str] = ()
    include_regex: Optional[str] = None
    exclude_regex: Optional[str] = None
    dropna: bool = False


def _filter_columns(
    columns: Sequence[str],
    *,
    exclude_cols: Sequence[str],
    include_regex: Optional[str],
    exclude_regex: Optional[str],
    must_keep: Sequence[str] = (),
) -> List[str]:
    """
    Filter column names prior to numeric feature selection.

    Parameters
    ----------
    columns : Sequence[str]
        The gathered table's columns: group keys and feature columns.
    exclude_cols : Sequence[str]
        Columns to drop unconditionally.
    include_regex : str or None
        If provided, keep only columns whose names match this regex.
    exclude_regex : str or None
        If provided, drop columns whose names match this regex (applied after
        `include_regex`).
    must_keep : Sequence[str] or None

    Returns
    -------
    list of str
        The columns that survive, the group keys first.
    """

    cols = list(columns)
    keep = [c for c in cols if c not in set(exclude_cols)]
    if include_regex:
        rx = re.compile(include_regex)
        keep = [c for c in keep if rx.search(c)]
    if exclude_regex:
        rx = re.compile(exclude_regex)
        keep = [c for c in keep if not rx.search(c)]
    # always hang onto the group keys (dropping duplicates, but keeping order).
    # we only re-add keys that actually exist: naming a missing column here
    # would beat the caller's much clearer "Missing group-by columns in
    # data: [...]" error to the punch.
    present = set(cols)
    return list(dict.fromkeys([k for k in must_keep if k in present] + keep))


def _resolve_keys(base_keys: Sequence[str], columns: Sequence[str]) -> List[str]:
    """A group key by its own name, or by its numbered variant ('k.1') when
    the name itself was taken; left as is when neither exists, so the
    missing-keys error below can name it."""
    cols = set(columns)
    resolved = []
    for k in base_keys:
        if k in cols:
            resolved.append(k)
            continue
        prefix = f"{k}."
        candidates = [c for c in columns if c == f"{k}.1" or c.startswith(prefix)]
        # sorted first, so we pick the same one every time
        resolved.append(sorted(candidates)[0] if candidates else k)
    return resolved


def _floats(cells: Sequence[str]):
    """A row's feature cells as float64, NaN where a cell is blank or not
    a number -- what ``to_numeric(errors="coerce")`` made of them."""
    import numpy as np

    try:
        return np.array([c if c.strip() else "nan" for c in cells], dtype=np.float64)
    except ValueError:
        out = np.empty(len(cells), dtype=np.float64)
        for i, c in enumerate(cells):
            try:
                out[i] = float(c)
            except ValueError:
                out[i] = np.nan
        return out


class _Running:
    """
    One group's running moments over the feature columns: a count, a sum,
    a mean with its M2 (Welford, so the deviation needs no second pass),
    a min and a max per column, folded in one row at a time with blank
    cells skipped per column. The memory of an aggregation is then the
    number of groups times a few vectors, however many rows went in.
    """

    __slots__ = ("n", "total", "mean", "m2", "lo", "hi")

    def __init__(self, width: int):
        import numpy as np

        self.n = np.zeros(width, dtype=np.int64)
        self.total = np.zeros(width)
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.lo = np.full(width, np.nan)
        self.hi = np.full(width, np.nan)

    def add(self, x) -> None:
        import numpy as np

        ok = ~np.isnan(x)
        if not ok.any():
            return
        xv = x[ok]
        self.n[ok] += 1
        self.total[ok] += xv
        delta = xv - self.mean[ok]
        self.mean[ok] += delta / self.n[ok]
        self.m2[ok] += delta * (xv - self.mean[ok])
        self.lo[ok] = np.fmin(self.lo[ok], xv)
        self.hi[ok] = np.fmax(self.hi[ok], xv)

    def stat(self, name: str):
        """The named statistic per column, NaN where it is undefined --
        the same answers ``pandas.DataFrame.agg`` gives (sample deviation,
        a sum of nothing is 0, a count is a count)."""
        import numpy as np

        n = self.n
        if name == "count":
            return n
        if name == "sum":
            return self.total
        with np.errstate(invalid="ignore", divide="ignore"):
            if name == "mean":
                return np.where(n > 0, self.total / np.maximum(n, 1), np.nan)
            if name in ("std", "var"):
                var = np.where(n > 1, self.m2 / np.maximum(n - 1, 1), np.nan)
                return np.sqrt(var) if name == "std" else var
        if name == "min":
            return self.lo
        if name == "max":
            return self.hi
        raise KeyError(name)


def _cell(value, stat: str) -> str:
    """A statistic as the text pandas would have written: a count as an
    integer, a float as its repr, nothing for NaN."""
    import numpy as np

    value = float(value)
    if np.isnan(value):
        return ""
    return str(int(value)) if stat == "count" else repr(value)


def _group_rows(plans, header: Sequence[str], group_keys: Sequence[str],
                feature_cols: Sequence[str], *, delimiter: str, encoding: str,
                dropna: bool, ticker: Ticker) -> Iterator[Tuple[tuple, "object"]]:
    """
    Every row of every table as ``(group, features)``: the group key
    values as text and the feature cells as a float vector. Streams, one
    table after another, ticking the table off when it is done.
    """
    at = {c: i for i, c in enumerate(header)}
    key_at = [at[k] for k in group_keys]
    feat_at = [at[c] for c in feature_cols]
    for ref, lead, cols in plans:
        # where this table's cells land in the gathered header, and the
        # lead values it carries
        slots = {at[c]: j for j, c in enumerate(cols)}
        origin = dict(zip((at[c] for c in lead), _origin_values(ref, lead)))
        key_src = [(slots.get(i), origin.get(i, "")) for i in key_at]
        feat_src = [slots.get(i) for i in feat_at]
        for cells in tables.read_cells(ref, encoding=encoding, delimiter=delimiter):
            group = tuple(cells[j] if j is not None else fixed for j, fixed in key_src)
            if dropna and any(not g for g in group):
                continue
            yield group, _floats([cells[j] if j is not None else "" for j in feat_src])
        ticker.tick(message="reading feature tables")


@records_settings(binding=GATHER_BINDING, outputs=("out_csv",))
def aggregate_features(
    *,
    root_dir: PathLike,
    pattern: str = "*.csv",
    recursive: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    add_source_path: bool = False,
    plan: AggregationPlan,
    out_csv: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Discover files, read, concatenate, and aggregate numeric columns per plan.

    This function consolidates tables from a single folder (or one table in
    a database), filters columns, coerces candidate features to numeric,
    groups by the specified keys, and computes the requested statistics.
    Output columns for aggregated features are flattened with the pattern
    ``"{column}__{stat}"``.

    Parameters
    ----------
    root_dir : PathLike
        Folder containing per-item CSVs, a single CSV file, or a database
        table (``db://<connection>/<table>``).
    pattern : str, default="*.csv"
        Glob pattern for selecting files.
    recursive : bool, default=True
        Recurse into subdirectories when True.
    delimiter : str, default=","
        CSV delimiter.
    encoding : str, default="utf-8-sig"
        CSV encoding for read/write.
    add_source_path : bool, default=False
        If True, include absolute path in ``"source_path"`` prior to filtering.
    plan : AggregationPlan
        Aggregation configuration (group keys, stats, filters, NA handling).
    out_csv : PathLike or None, default=None
        Output path. If None, defaults to
        ``<root_dir_parent>/<root_dir_name>.csv``.
    overwrite_existing : bool, default=False
        If False and `out_csv` exists, return it without recomputation.

    Returns
    -------
    pathlib.Path
        Path to the written CSV of aggregated features (the ref, for a
        database table).

    Raises
    ------
    FileNotFoundError
        If no files match the pattern under `root_dir`.
    RuntimeError
        If files were found but none could be read successfully.
    ValueError
        If required group-by columns are missing,
        or if no numeric columns remain after filtering,
        or if per-file grouping is requested but the ``"source"`` column is absent.

    Notes
    -----
    Group keys are preserved as leading columns in the output. The output places
    ``"source"`` (and optionally ``"source_path"``) first when present. The
    statistics in :data:`RUNNING_STATS` are folded in one pass with a few
    numbers per group held; any other statistic goes through pandas and
    holds the whole table.
    """
    import numpy as np

    root = tables.parse_ref(root_dir)
    out_ref = _default_out(root, out_csv)

    if tables.exists(out_ref) and not overwrite_existing:
        if verbose:
            print(f"Aggregated feature output file already exists; returning existing file: {out_ref}")
        return tables.handle(out_ref)

    files = _sources(root, pattern=pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {root}")

    announce(on_progress, "reading feature tables")
    ticker = Ticker(on_progress, len(files))
    plans = _read_plans(files, delimiter=delimiter, encoding=encoding,
                        add_source_path=add_source_path, verbose=verbose, tag="aggregate")
    if not plans:
        raise RuntimeError("No CSVs could be read successfully.")

    header = _union([[*lead, *cols] for _, lead, cols in plans])

    # if we're aggregating across files (per_file=False), we promote the inner
    # keys (e.g., 'source.1' -> 'source') and demote the file-level keys. the
    # names move; the positions don't, which is what lets each table's cells
    # still land in the right slot
    if not plan.per_file:
        header = _promote_inner_keys(header, plan.group_by)

    # build the (base) group keys from the plan
    group_keys = list(plan.group_by)
    if plan.per_file:
        if "source" not in header:
            raise ValueError("source column is missing; cannot group per_file.")
        group_keys = ["source"] + group_keys

    # resolve any collisions against the columns we actually have
    group_keys = _resolve_keys(group_keys, header)

    # now we filter, but we ALWAYS keep the group keys
    kept = _filter_columns(
        header,
        exclude_cols=tuple(plan.exclude_cols) + ("source_path",),
        include_regex=plan.include_regex,
        exclude_regex=plan.exclude_regex,
        must_keep=group_keys,
    )

    missing = [k for k in group_keys if k not in kept]
    if missing:
        raise ValueError(f"Missing group-by columns in data: {missing}")

    # everything that isn't a key is a candidate feature; which of them are
    # numeric we only know once we've seen the cells
    feature_cols = [c for c in kept if c not in set(group_keys)]
    if not feature_cols:
        raise ValueError("No numeric columns available for aggregation after filtering.")

    stats = [str(s) for s in plan.stats]
    # the promotion renamed the gathered header by position, so each table's
    # own columns have to be mapped through the header they were unioned
    # into, not through the renamed one. _group_rows does that from `plans`
    # against the pre-promotion names, so hand it those
    pre = _union([[*lead, *cols] for _, lead, cols in plans])
    rows = _group_rows(plans, pre, [pre[header.index(k)] for k in group_keys],
                       [pre[header.index(c)] for c in feature_cols],
                       delimiter=delimiter, encoding=encoding, dropna=plan.dropna,
                       ticker=ticker)

    width = len(feature_cols)
    seen = np.zeros(width, dtype=np.int64)
    if all(s in RUNNING_STATS for s in stats):
        # one pass, a few vectors per group, and the rows are gone. this is
        # what keeps a million utterance embeddings from being a million
        # rows of 768 floats in RAM
        groups: Dict[tuple, _Running] = {}
        for group, x in rows:
            acc = groups.get(group)
            if acc is None:
                acc = groups[group] = _Running(width)
            acc.add(x)
            seen += ~np.isnan(x)
        results = {g: {s: acc.stat(s) for s in stats} for g, acc in groups.items()}
    else:
        # a statistic with no running form (a median, a quantile) needs
        # every value, so these go through pandas and hold the table
        import pandas as pd

        keys_held: List[tuple] = []
        held: list = []
        for group, x in rows:
            keys_held.append(group)
            held.append(x)
            seen += ~np.isnan(x)
        values = np.vstack(held) if held else np.zeros((0, width))
        frame = pd.DataFrame(values, columns=feature_cols)
        for i, k in enumerate(group_keys):
            frame.insert(i, k, [g[i] if g[i] else np.nan for g in keys_held])
        agg = frame.groupby(group_keys, dropna=plan.dropna).agg(
            {c: stats for c in feature_cols})
        results = {}
        for key, row in agg.iterrows():
            key = key if isinstance(key, tuple) else (key,)
            group = tuple("" if isinstance(k, float) and np.isnan(k) else str(k) for k in key)
            results[group] = {s: np.array([row[(c, s)] for c in feature_cols])
                              for s in stats}

    # a column of pure text parses to nothing anywhere -- if we kept it we'd
    # spit out meaningless "<textcol>__mean" columns full of blanks. any
    # column where at least one value parsed as a number, we keep
    usable = [j for j in range(width) if seen[j] > 0]
    if not usable:
        raise ValueError("No numeric columns available for aggregation after filtering.")

    # make sure 'source' (and 'source_path' if present) lead the output
    lead = [c for c in ("source", "source_path") if c in group_keys]
    key_order = lead + [k for k in group_keys if k not in lead]
    key_pos = [group_keys.index(k) for k in key_order]
    out_header = key_order + [f"{feature_cols[j]}__{s}" for j in usable for s in stats]
    hints = {**{k: "text" for k in key_order},
             **{f"{feature_cols[j]}__{s}": ("int" if s == "count" else "float")
                for j in usable for s in stats}}

    # groups come out sorted on their keys, a blank key last, which is the
    # order a groupby wrote them in
    ordered = sorted(results, key=lambda g: tuple((not k, k) for k in g))
    with tables.write_table(out_ref, out_header, encoding=encoding, hints=hints) as writer:
        for group in ordered:
            stat = results[group]
            writer.writerow([*(group[i] for i in key_pos),
                             *(_cell(stat[s][j], s) for j in usable for s in stats)])
    return tables.handle(out_ref)


# ---------------------------
# Minimal CLI
# ---------------------------


# ---------------------------------------------------------------------------
# Command line -- derived from the functions above; see helpers.cliargs.CliSpec.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    {"gather": gather_csvs_to_one, "aggregate": aggregate_features,
     "run": feature_gather},
    description="Concatenate or aggregate the feature tables under a folder.",
    legacy={"--no-recursive": ["--recursive", "false"]},
    # we build the plan from the other arguments (`group_by`, `stats`, ...);
    # there's no sensible way to type one out on a command line.
    skip=("plan",),
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
