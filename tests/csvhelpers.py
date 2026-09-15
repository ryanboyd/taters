"""
The two CSV helpers every results test needs, once.

`_read` was defined identically in nine test files and `_write` in five (and
`read_rows` in four more, and `_write_table` in two). Not a pytest file: it is
imported like `preset_checks`.
"""

import csv
from pathlib import Path


def read_rows(path) -> list:
    """Every row of a CSV as a dict, the way the analyzers write them."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path, header, rows) -> Path:
    """A CSV from a header and rows, parent folders made; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_table(tmp_path, header, rows, name="analysis_table.csv") -> Path:
    """An analysis table under `tmp_path`, by name."""
    return write_rows(Path(tmp_path) / name, header, rows)


# the names the test files have always used.
_read = read_rows
_write = write_rows
_write_table = write_table
