"""Small argparse helpers shared by the module CLIs.

The main export is :func:`str2bool`, which exists because
``add_argument("--flag", type=bool)`` does the wrong thing in argparse: it runs
Python's truthiness rule over the *string*, so ``--flag false`` yields ``True``.
"""

from __future__ import annotations

import argparse
from typing import Any

_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


def str2bool(value: Any) -> bool:
    """
    Parse a CLI string into a bool, raising on anything ambiguous.

    Parameters
    ----------
    value : Any
        Value from the command line (e.g., ``"true"``, ``"0"``, ``"yes"``).
        Actual booleans are passed through unchanged.

    Returns
    -------
    bool
        The parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value is not a recognized boolean spelling. Failing loudly is
        deliberate: silently reading ``--overwrite_existing maybe`` as ``False``
        is how people lose a day of compute.
    """
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (true/false, yes/no, 1/0), got {value!r}"
    )


def add_bool_argument(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str | None = None,
    default: bool = False,
    help: str | None = None,
) -> argparse.Action:
    """
    Add a flag that works bare (``--overwrite_existing``) *and* with an explicit
    value (``--overwrite_existing false``).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.
    *flags : str
        Option strings, e.g. ``"--overwrite_existing"``.
    dest : str, optional
        Destination attribute name; argparse's default is used when omitted.
    default : bool, default=False
        Value when the flag is absent.
    help : str, optional
        Help text.

    Returns
    -------
    argparse.Action
        The action that was added.
    """
    kwargs: dict[str, Any] = dict(
        nargs="?", const=True, default=default, type=str2bool, help=help
    )
    if dest is not None:
        kwargs["dest"] = dest
    return parser.add_argument(*flags, **kwargs)
