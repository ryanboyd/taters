"""
Descriptive statistics for every feature table a run writes.

The first thing anyone does with a new measure is look at its distribution:
how many texts have it, what its mean and spread are, whether it is skewed
into a corner, whether a suspicious maximum is one text or a hundred. Until
now that meant opening each CSV in a spreadsheet and doing it by hand, per
column, per table. This step does it once for every numeric column of every
feature table -- count, missing, mean, standard deviation, quartiles, range,
skewness, kurtosis, zeros, distinct values -- and writes one table of
descriptives per feature table under ``stats_descriptives/``, whether or not
any statistics were asked for.

The numbers are the textbook sample statistics (standard deviation with
n-1; the bias-adjusted skewness and excess kurtosis that SPSS, R's
``psych::describe`` with ``type=2`` and Excel report), so they can be pasted
into a methods table beside anyone else's. Every cell is computed from the
values that are present; a blank cell is missing, and is counted as such.

One file per table, written only when the table is newer than it, so a
second run of an unchanged pipeline writes nothing -- the resume contract.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.progress import announce
from ._common import looks_numeric, numeric_column, reusable

__all__ = ["describe_features", "describe_values", "COLUMNS", "SKIP_COLUMNS"]

PathLike = Union[str, Path]

#: The descriptives, in the order they are written.
COLUMNS = ["variable", "n", "missing", "mean", "sd", "min", "q1", "median",
           "q3", "max", "skew", "kurtosis", "zeros", "unique"]

#: Identifier columns that are never described even when they parse as
#: numbers -- a mean participant id is not a statistic.
SKIP_COLUMNS = frozenset({"text_id", "source", "speaker", "source_col"})


def describe_values(values) -> Dict[str, Optional[float]]:
    """
    The descriptives of one column, from its values with blanks as NaN.

    ``sd`` needs two values, ``skew`` three and ``kurtosis`` four; below
    that they are ``None`` rather than a division by zero dressed as a
    number. Skewness is the bias-adjusted sample statistic (G1) and kurtosis
    the bias-adjusted *excess* kurtosis (G2), zero for a normal distribution
    -- the pair a methods table expects.
    """
    import numpy as np

    x = np.asarray(values, dtype=float)
    present = x[~np.isnan(x)]
    n = int(present.size)
    out: Dict[str, Optional[float]] = {
        "n": n, "missing": int(x.size - n), "mean": None, "sd": None,
        "min": None, "q1": None, "median": None, "q3": None, "max": None,
        "skew": None, "kurtosis": None, "zeros": int(np.sum(present == 0)),
        "unique": int(np.unique(present).size),
    }
    if n == 0:
        return out
    mean = float(present.mean())
    out["mean"] = mean
    q1, median, q3 = (float(v) for v in np.percentile(present, [25, 50, 75]))
    out.update(min=float(present.min()), q1=q1, median=median, q3=q3,
               max=float(present.max()))
    if n >= 2:
        sd = float(present.std(ddof=1))
        out["sd"] = sd
        if sd > 0:
            z = (present - mean) / sd
            if n >= 3:
                out["skew"] = float(np.sum(z ** 3) * n / ((n - 1) * (n - 2)))
            if n >= 4:
                m4 = float(np.sum(z ** 4))
                out["kurtosis"] = float(
                    (n * (n + 1) * m4 / ((n - 1) * (n - 2) * (n - 3)))
                    - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        else:
            out["skew"] = 0.0 if n >= 3 else None
            out["kurtosis"] = None
    return out


def _fmt(value, rounding: int) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(round(float(value), rounding))


def _describe_table(path, encoding: str) -> List[List[object]]:
    """
    One row of descriptives per numeric column of a feature table.

    Read in blocks: each column's values are parsed to floats as they go
    and only the floats are kept, so the table is never held as strings.
    A column that stops parsing in a later block is dropped, the same
    verdict `looks_numeric` gives over the whole column.
    """
    import numpy as np

    from ..helpers.tables import columns_of, read_cells

    names = [c for c in columns_of(path, encoding=encoding)]
    wanted = [c for c in names if c not in SKIP_COLUMNS]
    parts: Dict[str, List[object]] = {c: [] for c in wanted}
    seen: Dict[str, bool] = {c: False for c in wanted}
    block: Dict[str, List[str]] = {c: [] for c in wanted}

    def flush() -> None:
        for c in list(parts):
            values = block[c]
            if not values:
                continue
            if any(str(v or "").strip() for v in values):
                if not looks_numeric(values):
                    del parts[c]
                    continue
                seen[c] = True
            parts[c].append(numeric_column(values, column=c))
        for c in block:
            block[c].clear()

    n = 0
    for cells in read_cells(path, columns=wanted, encoding=encoding):
        for c, v in zip(wanted, cells):
            if c in parts:
                block[c].append(v)
        n += 1
        if n % 50_000 == 0:
            flush()
    flush()
    rows: List[List[object]] = []
    for column in wanted:
        if column not in parts or not seen[column]:
            continue
        values = np.concatenate(parts[column]) if parts[column] else np.empty(0)
        stats = describe_values(values)
        rows.append([column] + [stats[k] for k in COLUMNS[1:]])
    return rows


def _write_if_changed(path: Path, text: str) -> None:
    text = text.rstrip() + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write(text)


def describe_features(
    feature_csvs: Sequence[PathLike],
    out_dir: PathLike = "stats_descriptives",
    *,
    overwrite_existing: bool = False,
    encoding: str = "utf-8-sig",
    rounding: int = 4,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Write a table of descriptive statistics for every feature table.

    Parameters
    ----------
    feature_csvs : sequence of str or Path
        The feature tables to describe -- one row per text, identifier
        columns first, measures after. Every Taters feature file has this
        shape; the pipeline hands over every table the run wrote.
    out_dir : str or Path, default "stats_descriptives"
        Where the descriptives go: ``<out_dir>/<table stem>.csv`` for each
        table, and a ``README.md`` listing them.
    overwrite_existing : bool, default False
        Recompute a table's descriptives even when they are newer than it.
    encoding : str, default "utf-8-sig"
        The tables' encoding, and the descriptives'.
    rounding : int, default 4
        Decimal places written.
    verbose, on_progress
        The usual.

    Returns
    -------
    Path
        ``out_dir``.

    Notes
    -----
    Every numeric column is described except the identifiers (``text_id``,
    ``source``, ``speaker``). Columns holding words are left out, since a
    mean of them means nothing. Count columns the steps write beside their
    measures (``token_count``, ``WC``) are described like any other -- they
    are the first thing to look at when a measure looks strange.
    """
    from ..helpers import tables

    out_dir = Path(out_dir)
    # feature tables live wherever the run put them; the descriptives are
    # bookkeeping and stay in `out_dir` on disk
    paths = [tables.parse_ref(p) for p in feature_csvs]
    if not paths:
        if verbose:
            print("[describe] no feature tables to describe.")
        return out_dir
    written = reused = 0
    index: List[str] = []
    for i, path in enumerate(paths):
        if on_progress is not None:
            on_progress(i, len(paths), f"describing {path.stem}")
        if not tables.exists(path):
            raise FileNotFoundError(f"feature table not found: {path}")
        out = out_dir / f"{path.stem}.csv"
        if reusable(out, path, overwrite_existing=overwrite_existing,
                    verbose=verbose, what=f"the descriptives for {path.stem}"):
            reused += 1
        else:
            rows = _describe_table(path, encoding)
            out_dir.mkdir(parents=True, exist_ok=True)
            with atomic_write(out, mode="w", newline="", encoding=encoding) as fh:
                writer = csv.writer(fh)
                writer.writerow(COLUMNS)
                for row in rows:
                    writer.writerow([row[0]] + [_fmt(v, rounding) for v in row[1:]])
            written += 1
        index.append(out.name)
    announce(on_progress, "writing the descriptives index")
    lines = ["# Descriptive statistics", "",
             "One table per feature table: for every numeric measure, how many "
             "texts have it (`n`) and how many do not (`missing`), the mean and "
             "standard deviation (n-1), the minimum, quartiles and maximum, the "
             "skewness and excess kurtosis (bias-adjusted, as SPSS and R's "
             "psych report them; zero for a normal distribution), how many "
             "values are exactly zero, and how many distinct values there are. "
             "Identifier columns are left out.", ""]
    lines += [f"- `{name}`" for name in index]
    _write_if_changed(out_dir / "README.md", "\n".join(lines))
    if verbose:
        print(f"[describe] {written} table(s) described, {reused} already "
              f"current -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# command line -- we derive it from the function above; see helpers.cliargs.CliSpec.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    describe_features,
    description="Descriptive statistics for every numeric column of one or "
                "more feature tables.",
    aliases={},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
