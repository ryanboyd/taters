"""
Average a feature table up to a coarser row.

The other way round from :mod:`text_gather`. That one joins the *texts* that
belong together and measures the result once; this one measures every row on
its own and then averages the *numbers*. They are different analyses, not two
routes to one answer -- joining a speaker's words before measuring versus
averaging their per-utterance scores differ by around 35% on vocabulary
measures -- so which one happened is the user's choice and is recorded.

One table in, one table out, because a feature table's record has to sit
beside it: :func:`taters.stats.assemble.assemble_analysis_table` reads the
record of every table it joins to work out which columns are bookkeeping, and
:mod:`taters.score_model` reads it again to check a saved model against the
features it was fitted on. A step that wrote a folder full of tables would
leave all of them unrecorded.

Averaging repeats. Utterances to speakers to conversations is two calls, the
second reading the first's output, and it is not the same as one call straight
to conversations: the two-step version weights speakers equally, the one-step
version weights whoever talked most. That is the reason this takes a table
rather than a corpus -- so the chain is visible in the pipeline instead of
hidden in an argument.

Nothing here renames a column. A model is fitted on column names and has to be
applicable to text that arrives at any shape, so ``flesch_reading_ease`` stays
``flesch_reading_ease`` whether it describes one utterance or a conversation's
mean of means. The pipeline and its manifest are what record the difference;
see the ``defines_text`` note in :mod:`taters.helpers.provenance`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

from .atomic import atomic_write
from .cliargs import CliSpec
from .feature_columns import ColumnSpec
from .progress import announce
from .provenance import records_settings
from .text_gather import _compose_id

__all__ = ["average_feature_table", "GROUP_FALLBACK_ID", "COUNT_COLUMN"]

FEATURE_COLUMNS = ColumnSpec(
    label="Averaged feature tables",
    dynamic="the columns of the table being averaged, under their own names",
)

PathLike = Union[str, Path]

#: What the row key becomes when every grouping value is blank. The text
#: gather does the same (``_compose_id(key) or "group"``), and the two have to
#: agree exactly or the analysis table's join finds nothing.
GROUP_FALLBACK_ID = "group"

#: How many rows went into each average. Useful to filter on ("I don't trust a
#: speaker I have two turns from") and never a predictor, so it is declared
#: bookkeeping below.
COUNT_COLUMN = "rows_averaged"

#: What one row represents -- recorded, and deliberately never compared, so a
#: model fitted on conversation means can still score utterances.
_GRAIN = ("group_by", "split_col", "min_rows")


def _numeric(value: str) -> Optional[float]:
    """The cell as a number, or None if it is not one."""
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_rows(path: Path, *, encoding: str, delimiter: str):
    """Header and rows of a CSV, as strings."""
    import csv

    from .csvio import widen_csv_field_limit

    widen_csv_field_limit()
    with path.open("r", newline="", encoding=encoding) as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = list(reader.fieldnames or [])
        return header, [dict(row) for row in reader]


def _key_lookup(keys_csv: Path, *, key_cols: Sequence[str],
                group_by: Sequence[str], encoding: str, delimiter: str) -> dict:
    """
    ``{row key: (group values)}`` from the key map.

    The first average needs this because a feature table carries `text_id` and
    its own numbers and nothing else -- there is no column on it saying which
    speaker the utterance belonged to. Later averages do not: their input
    already carries the columns it was grouped on.
    """
    header, rows = _read_rows(keys_csv, encoding=encoding, delimiter=delimiter)
    missing = [c for c in list(key_cols) + list(group_by) if c not in header]
    if missing:
        raise ValueError(
            f"{keys_csv.name} has no column(s) {missing}, so it cannot say "
            f"which rows belong together. It needs {list(key_cols)} to match "
            f"the feature table on, and {list(group_by)} to group by.")
    lookup = {}
    for row in rows:
        lookup[tuple(str(row.get(c, "")) for c in key_cols)] = \
            tuple(str(row.get(c, "")) for c in group_by)
    return lookup


@records_settings(
    # which table was read, and what it was matched against. recorded for
    # diagnosis, never compared: a model exists to meet a different corpus.
    binding=("in_csv", "keys_csv", "key_cols", "delimiter", "encoding"),
    grain=_GRAIN,
    outputs=("out_csv",),
    # averaging is not measuring: the record takes the identity of whatever
    # measured `in_csv`, so a model fitted on averages can be scored on the
    # un-averaged table, and the measure's own bookkeeping columns are not
    # forgotten on the way up.
    reshapes="in_csv",
    # cheap enough that an existing copy with no record gets redone rather
    # than trusted -- same reasoning as the two text gathers.
    redo_without_record=True,
    bookkeeping=(COUNT_COLUMN,),
    # this step decides what a row *is*, not how anything was measured. its
    # settings stay out of the chain downstream steps compare, which is what
    # lets a model fitted on averages score un-averaged text. the module
    # docstring in `provenance` names this case outright.
    defines_text=True)
def average_feature_table(
    *,
    in_csv: PathLike,
    out_csv: PathLike,
    group_by: Sequence[str],
    keys_csv: Optional[PathLike] = None,
    key_cols: Sequence[str] = ("text_id",),
    split_col: str = "",
    min_rows: int = 1,
    overwrite_existing: bool = False,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    on_progress: Optional[Callable[..., None]] = None,
    verbose: bool = True,
) -> Path:
    """
    One row per group, each number the mean of the rows that went into it.

    Parameters
    ----------
    in_csv : str or Path
        The feature table to average: one row per text, a row key, and numeric
        columns. Non-numeric columns are dropped, since there is no mean of a
        word.
    out_csv : str or Path
        Where to write the averaged table.
    group_by : sequence of str
        The columns whose shared values make a group. They come from
        ``keys_csv`` on the first average and from ``in_csv`` itself on any
        later one.
    keys_csv : str or Path, optional
        A table saying which row belongs to which group: the row key, and the
        grouping columns. Needed for the first average of a feature table,
        whose only non-numeric column is its key. Omit it once the input
        already carries the grouping columns.
    key_cols : sequence of str, default ``("text_id",)``
        What identifies a row in both tables, and the name the averaged row
        key is written under. The averaged key is the group's values joined
        with ``" | "``, exactly as the text gather composes a grouped id, so
        the two tables join.
    split_col : str, default ""
        A column that splits a row legitimately and must survive the average
        rather than be grouped away -- ``source_col`` when each text column
        was measured separately. Added to the grouping when present.
    min_rows : int, default 1
        Leave out groups assembled from fewer rows than this. ``1`` keeps
        everything.
    overwrite_existing : bool, default False
        Rebuild the table if it is already there.
    delimiter, encoding : str
        How to read and write the CSVs.
    on_progress : callable, optional
        Progress callback.
    verbose : bool, default True
        Print a line about what was averaged.

    Returns
    -------
    pathlib.Path
        ``out_csv``.

    Notes
    -----
    A group whose values are all blank gets the row key ``"group"``, matching
    :func:`taters.helpers.text_gather.csv_to_analysis_ready_csv`. A blank cell
    is skipped rather than counted as zero, so a column measured on only some
    of a speaker's turns is the mean of the turns it was measured on; the
    count column reports the rows in the group, not the rows behind any one
    number.
    """
    src = Path(in_csv)
    out = Path(out_csv)
    keys = list(group_by) + ([split_col] if split_col else [])
    if not keys:
        raise ValueError(
            "group_by is empty, so there is nothing to average by. Averaging "
            "a table by nothing would copy it.")
    if out.exists() and not overwrite_existing:
        if verbose:
            print(f"[average] exists, leaving alone: {out}")
        return out
    if not src.is_file():
        raise FileNotFoundError(f"in_csv not found: {src}")

    announce(on_progress, f"averaging {src.name} by {', '.join(keys)}")
    header, rows = _read_rows(src, encoding=encoding, delimiter=delimiter)
    if not header:
        raise ValueError(f"{src.name} has no header row.")

    lookup = None
    if keys_csv is not None:
        lookup = _key_lookup(Path(keys_csv), key_cols=key_cols,
                             group_by=group_by, encoding=encoding,
                             delimiter=delimiter)
        missing = [c for c in key_cols if c not in header]
        if missing:
            raise ValueError(
                f"{src.name} has no column(s) {missing}, so its rows cannot "
                f"be matched to the ones in {Path(keys_csv).name}.")
    else:
        absent = [c for c in keys if c not in header]
        if absent:
            raise ValueError(
                f"{src.name} has no column(s) {absent} to group by. Averaging "
                f"a table by a column it does not carry needs keys_csv, which "
                f"says which row belongs to which group.")

    # numeric columns are whatever parses somewhere. a column of words gives
    # no number anywhere and is dropped; one that's numeric in some rows and
    # blank in others is kept and averaged over the rows that had a value.
    # the count from a previous average is not a measurement, so it is not
    # averaged again -- and leaving it in put the column in the header twice,
    # once as a mean of counts and once as this level's own count.
    skip = set(keys) | set(key_cols) | {COUNT_COLUMN}
    candidates = [c for c in header if c not in skip]
    sums = {}
    counts = {}
    members = {}
    order: List[tuple] = []
    for row in rows:
        if lookup is not None:
            found = lookup.get(tuple(str(row.get(c, "")) for c in key_cols))
            if found is None:
                # a row the key map has never heard of. skipping it beats
                # inventing a group for it, and the count column is what
                # shows the rows went missing.
                continue
            group = found + ((str(row.get(split_col, "")),) if split_col else ())
        else:
            group = tuple(str(row.get(c, "")) for c in keys)
        if group not in members:
            members[group] = 0
            sums[group] = {}
            counts[group] = {}
            order.append(group)
        members[group] += 1
        for col in candidates:
            value = _numeric(row.get(col, ""))
            if value is None:
                continue
            sums[group][col] = sums[group].get(col, 0.0) + value
            counts[group][col] = counts[group].get(col, 0) + 1

    kept_cols = [c for c in candidates if any(counts[g].get(c) for g in order)]
    if not kept_cols:
        raise ValueError(
            f"{src.name} has no numeric column to average -- only "
            f"{', '.join(header) or 'nothing'}.")

    survivors = [g for g in order if members[g] >= max(1, int(min_rows))]
    if not survivors:
        raise ValueError(
            f"every group in {src.name} was assembled from fewer than "
            f"{min_rows} row(s), so nothing is left to write.")

    out.parent.mkdir(parents=True, exist_ok=True)
    id_col = list(key_cols)[0]
    out_header = [id_col, *keys, COUNT_COLUMN, *kept_cols]
    import csv as _csv

    with atomic_write(out, mode="w", newline="", encoding="utf-8") as fh:
        writer = _csv.writer(fh, delimiter=delimiter)
        writer.writerow(out_header)
        for group in survivors:
            row_id = _compose_id(group) or GROUP_FALLBACK_ID
            values = []
            for col in kept_cols:
                n = counts[group].get(col, 0)
                values.append(f"{sums[group][col] / n:.10g}" if n else "")
            writer.writerow([row_id, *group, members[group], *values])

    if verbose:
        dropped = len(order) - len(survivors)
        tail = f", {dropped} group(s) under {min_rows} row(s) left out" \
            if dropped else ""
        print(f"[average] {src.name}: {len(rows)} row(s) -> "
              f"{len(survivors)} group(s) by {', '.join(keys)}{tail}")
    return out


CLI = CliSpec(
    {"run": average_feature_table},
    description="Average a feature table up to one row per group.",
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
