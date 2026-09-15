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


# ---------------------------------------------------------------------------
# A command line derived from the function it drives
# ---------------------------------------------------------------------------

_UNSET = object()


def _first_sentence(text: str) -> str:
    """The opening sentence of a numpydoc description, safe for argparse.

    argparse formats help strings with ``%``, so a description that says
    "keep terms in at least 5% of texts" crashed ``--help`` with "unsupported
    format character" -- which is how the topic model's command line went
    unrunnable without any test noticing.
    """
    text = " ".join((text or "").split())
    for stop in (". ", ".\n"):
        if stop in text:
            text = text[: text.index(stop) + 1]
            break
    return text.replace("%", "%%")


class CliSpec:
    """
    Build a module's command line from the function it drives.

    Every public analysis function is keyword-only and documented in
    numpydoc, and the wizard already renders its options screen from exactly
    that (see :func:`taters.ui.introspect.describe`). The hand-written
    parsers beside those functions had drifted in every way a parser can:
    twenty-nine of them, in two flag dialects (``--out-dir`` here,
    ``--out_dir`` there), with booleans that could be switched on but not
    off, and each missing between two and eighteen of its function's
    parameters -- the stats CLIs had no controls or PCA, the text CLIs no
    ``--workers``, the scorer none of the safety valves its own refusals told
    people to pass. Deriving the parser from the signature makes parity a
    property rather than a chore.

    One flag per parameter, ``--spelled-with-dashes`` (the underscored
    spelling is accepted too). Booleans take an optional ``true``/``false``
    and are on when given bare. List parameters repeat (``--outcome-col a
    --outcome-col b``, and the plural form) or take commas. Parameters are
    passed to the function only when given, so its own defaults apply.

    Parameters
    ----------
    targets
        The function, or ``{subcommand: function}`` for a module with more
        than one (``fit`` / ``apply``).
    aliases
        ``{parameter: [old flag, ...]}`` -- the spellings earlier parsers
        used, kept so every documented invocation still works.
    legacy
        ``{old flag: replacement}`` applied to ``argv`` before parsing, for
        flags that meant more than a spelling: ``--no-zscore`` becomes
        ``--zscore false``. A replacement is a list of tokens, or a callable
        taking the flag's value (which is consumed) and returning tokens.
    parsers
        ``{parameter: callable}`` to turn a string into the value the
        function wants, where the type cannot be inferred (a filter written
        as JSON).
    skip
        Parameters never offered (``on_progress`` always is).
    positional
        Parameters also accepted positionally, in this order, for the CLIs
        that took them that way.
    """

    def __init__(self, targets, *, description: str, aliases=None, legacy=None,
                 parsers=None, skip=(), positional=(), rewrite=None,
                 echo=print):
        self.targets = targets if isinstance(targets, dict) else {None: targets}
        self.description = description
        self.aliases = dict(aliases or {})
        self.legacy = dict(legacy or {})
        self.parsers = dict(parsers or {})
        self.skip = set(skip) | {"on_progress"}
        self.positional = tuple(positional)
        self.rewrite = rewrite
        self.echo = echo

    # -- building ----------------------------------------------------------
    def _add(self, parser: argparse.ArgumentParser, fn) -> None:
        from ..ui.introspect import describe

        spec = describe(fn)
        for param in spec.params:
            name = param.name
            if name in self.skip:
                continue
            flags = [f"--{name.replace('_', '-')}"]
            if "_" in name:
                flags.append(f"--{name}")
            if param.widget == "list" and name.endswith("s"):
                singular = f"--{name[:-1].replace('_', '-')}"
                if singular not in flags:
                    flags.append(singular)
            for old in self.aliases.get(name, ()):
                if old not in flags:
                    flags.append(old)
            help_text = _first_sentence(param.desc) or None
            if name in self.positional:
                parser.add_argument(name, nargs="?", default=_UNSET,
                                    help=help_text)
                # the flag form too, so scripts can be explicit about it.
                parser.add_argument(*flags, dest=name, default=_UNSET,
                                    help=argparse.SUPPRESS)
                continue
            # argparse copies an append action's default and appends to the
            # copy, so we can't use the unset sentinel here: None means
            # "never given" for lists.
            if name == "device":
                # the override hatch has to look identical everywhere; see
                # gpu.add_device_argument, whose contract we're keeping here.
                from .gpu import DEVICE_CHOICES, DEVICE_DEFAULT, DEVICE_HELP

                parser.add_argument(*flags, dest=name, choices=DEVICE_CHOICES,
                                    default=DEVICE_DEFAULT, help=DEVICE_HELP)
            elif name in self.parsers:
                parser.add_argument(*flags, dest=name, action="append",
                                    default=None, help=help_text)
            elif param.widget == "bool":
                default = param.default if param.has_default else False
                add_bool_argument(parser, *flags, dest=name,
                                  default=bool(default), help=help_text)
            elif param.widget == "list":
                parser.add_argument(*flags, dest=name, action="append",
                                    default=None, help=help_text)
            elif param.widget == "int":
                parser.add_argument(*flags, dest=name, type=_int_or_none,
                                    default=_UNSET, help=help_text)
            elif param.widget == "float":
                parser.add_argument(*flags, dest=name, type=_float_or_none,
                                    default=_UNSET, help=help_text)
            elif param.widget == "choice" and param.choices:
                parser.add_argument(*flags, dest=name,
                                    choices=[str(c) for c in param.choices],
                                    default=_UNSET, help=help_text)
            else:
                parser.add_argument(*flags, dest=name, default=_UNSET,
                                    help=help_text)

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.description)
        if list(self.targets) == [None]:
            self._add(parser, self.targets[None])
        else:
            sub = parser.add_subparsers(dest="_command", required=True)
            for command, fn in self.targets.items():
                child = sub.add_parser(command, help=_first_sentence(
                    (fn.__doc__ or "").strip().split("\n")[0]))
                self._add(child, fn)
        return parser

    def parsers_for_test(self) -> dict:
        """``{subcommand or None: parser}`` -- what the parity test inspects."""
        out = {}
        for command, fn in self.targets.items():
            child = argparse.ArgumentParser()
            self._add(child, fn)
            out[command] = child
        return out

    # -- running -----------------------------------------------------------
    def _rewritten(self, argv):
        argv = list(argv)
        if self.rewrite is not None:
            argv = list(self.rewrite(argv))
        out = []
        i = 0
        while i < len(argv):
            token = argv[i]
            if token in self.legacy:
                replacement = self.legacy[token]
                if callable(replacement):
                    value = argv[i + 1] if i + 1 < len(argv) else ""
                    out += list(replacement(value))
                    i += 2
                else:
                    out += list(replacement)
                    i += 1
                continue
            out.append(token)
            i += 1
        return out

    def kwargs(self, argv=None):
        """The function to call and the keyword arguments the line meant."""
        import sys

        argv = self._rewritten(sys.argv[1:] if argv is None else argv)
        args = vars(self.parser().parse_args(argv))
        fn = self.targets[args.pop("_command", None)]
        from ..ui.introspect import describe

        widgets = {p.name: p.widget for p in describe(fn).params}
        kwargs = {}
        for name, value in args.items():
            if value is _UNSET or name not in widgets:
                continue
            if value is None and (widgets[name] == "list" or name in self.parsers):
                continue
            if name in self.parsers:
                kwargs[name] = [self.parsers[name](v) for v in value]
            elif widgets[name] == "list":
                pieces = []
                for v in value:
                    pieces += [s.strip() for s in str(v).split(",") if s.strip()]
                kwargs[name] = pieces
            elif isinstance(value, str) and value.lower() in ("none", "null"):
                kwargs[name] = None
            else:
                kwargs[name] = value
        return fn, kwargs

    def run(self, argv=None) -> int:
        fn, kwargs = self.kwargs(argv)
        out = fn(**kwargs)
        if out is not None:
            self.echo(out if isinstance(out, str) else
                      (str(out) if not isinstance(out, (dict, list)) else
                       "\n".join(str(v) for v in
                                 (out.values() if isinstance(out, dict) else out))))
        return 0


def _int_or_none(text: str):
    return None if str(text).lower() in ("none", "null") else int(text)


def _float_or_none(text: str):
    return None if str(text).lower() in ("none", "null") else float(text)
