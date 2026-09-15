"""The one line every CSV reader here needs before reading a text table."""

import csv
import sys


def widen_csv_field_limit() -> None:
    """
    Let ``csv`` read cells longer than its 128 KB default.

    A transcript or a gathered document is one cell, and the default limit
    turns a long one into ``_csv.Error: field larger than field limit``
    several minutes into a run. This three-line guard had been copied into six
    modules; the ``OverflowError`` branch is for platforms whose C ``long`` is
    32 bits, where ``sys.maxsize`` is too large to store.
    """
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:                    # pragma: no cover - platform quirk
        csv.field_size_limit(2**31 - 1)
