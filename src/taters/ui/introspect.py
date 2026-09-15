"""
Turn a Taters function into a description a user interface can render.

Every analysis function in this package is a plain Python function with type
annotations and a numpydoc docstring. That is already most of a form: the
signature says what the fields are called and what type they hold, and the
``Parameters`` block says what each one means in prose. This module joins the
two into :class:`ParamSpec` objects so a UI never has to hard-code a list of
options for each module.

Deliberately UI-agnostic: nothing here imports ``questionary``, ``rich``, or
anything else terminal-shaped. The console wizard renders these specs, and a
future GUI or web front end can render the same ones -- they serialize to JSON
cleanly.

What this module can and cannot tell you
----------------------------------------
It gives you the **knobs**: names, types, defaults, help text, enumerated
choices. It cannot give you the **wiring** -- nothing in
``analyze_vocal_acoustics``'s signature says its ``transcript_csv`` should be
fed the output of an earlier transcription step. That part is declared by hand
in :mod:`taters.ui.recipes`.
"""

from __future__ import annotations

import importlib
import inspect
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "clean_doc",
    "ParamSpec",
    "FunctionSpec",
    "describe",
    "load_target",
    "parse_numpydoc_params",
    "widget_for",
]


# a sentinel for "this parameter has no default", which is not the same thing
# as "its default is None" -- we use None defaults all over the place.
class _Empty:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<required>"


EMPTY = _Empty()


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------

# every numpydoc section heading we might run into, so the Parameters block
# knows where to stop. otherwise "Returns" and everything after it gets
# swallowed up as if it were more parameter documentation.
_SECTION_NAMES = (
    "Parameters", "Returns", "Yields", "Receives", "Other Parameters",
    "Raises", "Warns", "Warnings", "See Also", "Notes", "References",
    "Examples", "Attributes", "Methods",
)
_SECTION_RE = re.compile(
    r"^[ \t]*(" + "|".join(_SECTION_NAMES) + r")[ \t]*\n[ \t]*-{3,}[ \t]*$",
    re.MULTILINE,
)

# a parameter entry header. we've got three different spellings in this
# codebase:
#
#   sample_rate : int, default 16000          <- the common case
#   root_dir                                  <- helpers/find_files.py, no type
#   include_globs / exclude_globs             <- two names sharing one entry
#
# the names group is loose about "/" so we catch the third form, and the type
# half is optional so we catch the second.
_RST_ROLE_RE = re.compile(r":[a-z:]+:`~?([^`]+)`")
_RST_LITERAL_RE = re.compile(r"``([^`]+)``")


def clean_doc(text: str) -> str:
    """
    Turn a numpydoc parameter description into a sentence a person can read.

    Two things are in the way. Docstrings are hard-wrapped at some column, so a
    description arrives as several short lines and anything showing only the
    first gets half a sentence. And they carry reStructuredText markup --
    ``None``, :func:`some.thing` -- which renders as literal backticks in a
    terminal.

    Done here rather than in the wizard so every front end gets readable text
    from the same place, and so the raw markup never has to be handled twice.

    Paragraph breaks are kept; a run of lines within a paragraph is joined.
    """
    if not text:
        return ""

    body = inspect.cleandoc(text)
    body = _RST_ROLE_RE.sub(r"\1", body)      # :func:`x.y` -> x.y
    body = _RST_LITERAL_RE.sub(r"\1", body)   # ``None``    -> None

    paragraphs = []
    for chunk in re.split(r"\n\s*\n", body):
        joined = " ".join(line.strip() for line in chunk.splitlines() if line.strip())
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs).strip()


_PARAM_HEADER_RE = re.compile(
    r"^(?P<names>\*{0,2}\w+(?:[ \t]*[/,][ \t]*\*{0,2}\w+)*)"
    r"(?:[ \t]*:[ \t]*(?P<type>.*))?$"
)

# literal choice sets inside a numpydoc type string: {16,24,32}, {"ms", "s"},
# {"concat", "separate"}. quotes are optional and we strip them.
_CHOICES_RE = re.compile(r"\{([^{}]+)\}")


@dataclass
class ParamDoc:
    """The docstring half of a parameter: its prose and its declared type."""

    name: str
    type_str: str = ""
    desc: str = ""


def parse_numpydoc_params(doc: Optional[str]) -> Dict[str, ParamDoc]:
    """
    Pull the ``Parameters`` block out of a numpydoc docstring.

    Parameters
    ----------
    doc : str or None
        A raw docstring. ``None`` and docstrings with no ``Parameters`` section
        both yield an empty dict rather than raising -- an undocumented
        function should still be usable, just with less help text.

    Returns
    -------
    dict[str, ParamDoc]
        Keyed by parameter name. An ``a / b`` entry is expanded into one key
        per name, both pointing at the same description.
    """
    if not doc:
        return {}

    doc = inspect.cleandoc(doc)

    # find the Parameters heading, then cut at whatever section comes next.
    body: Optional[str] = None
    for match in _SECTION_RE.finditer(doc):
        if match.group(1) != "Parameters":
            continue
        start = match.end()
        nxt = _SECTION_RE.search(doc, start)
        body = doc[start:nxt.start()] if nxt else doc[start:]
        break
    if body is None:
        return {}

    out: Dict[str, ParamDoc] = {}
    current: List[str] = []          # description lines for the entry in hand
    names: List[str] = []
    type_str = ""

    def flush() -> None:
        if not names:
            return
        desc = clean_doc("\n".join(current))
        for name in names:
            out[name] = ParamDoc(name=name, type_str=type_str, desc=desc)

    for line in body.splitlines():
        if not line.strip():
            current.append("")
            continue
        # indented lines carry on the description of the entry above.
        if line[:1] in (" ", "\t"):
            current.append(line)
            continue
        header = _PARAM_HEADER_RE.match(line.rstrip())
        if not header:
            # not a header we recognize -- treat it as more description so we
            # never silently drop text.
            current.append(line)
            continue
        flush()
        # numpydoc's own form is `a, b : type`; we've also got `a / b` in this
        # tree. both mean several names sharing one description, and a comma
        # header that didn't match here used to contribute nothing -- both
        # parameters reached the wizard with no help text at all.
        names = [n.strip().lstrip("*")
                 for n in re.split(r"[/,]", header.group("names"))]
        type_str = (header.group("type") or "").strip()
        current = []
    flush()
    return out


def _choices_from_type_str(type_str: str) -> Optional[List[Any]]:
    """
    Lift an enumerated choice set out of a numpydoc type string.

    ``bit_depth : {16,24,32}`` and ``time_unit : {"ms", "s"}`` both declare a
    closed set, which a UI should render as a picker rather than a free-text
    box. Values are returned as ints when they parse as ints, else as strings
    with any surrounding quotes removed.
    """
    match = _CHOICES_RE.search(type_str or "")
    if not match:
        return None
    values: List[Any] = []
    for raw in match.group(1).split(","):
        token = raw.strip().strip("'\"")
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            values.append(token)
    return values or None


def _choices_from_annotation(annotation: Any) -> Optional[List[Any]]:
    """
    The values a ``Literal[...]`` annotation allows, ``Optional`` or not.

    The signature is the other place a closed set is declared -- and the one
    that cannot drift from the code, since the type checker reads it. A
    parameter typed ``Literal["aside", "features"]`` with a docstring header
    of just its name was a text box asking someone to spell "features"
    (a real report); the annotation said all along what the answers were.
    """
    inner, _ = _unwrap_optional(annotation)
    if typing.get_origin(inner) is typing.Literal:
        values = list(typing.get_args(inner))
        return values or None
    return None


#: Words in a type string that, beside a ``{...}`` set, say the set is not the
#: whole story: ``{"tiny", "base"} or str`` takes a model folder too. ``None``
#: is not on the list -- ``{"auto", "cuda"} | None`` is still a closed set,
#: one of whose answers is "unset".
_OPEN_WORDS_RE = re.compile(r"\b(str|path|folder|directory|file)\b", re.I)


def _open_ended(type_str: str) -> bool:
    """Whether a type string offers its ``{...}`` set as examples rather than
    the only answers -- see :data:`_OPEN_WORDS_RE`."""
    rest = _CHOICES_RE.sub("", type_str or "")
    return bool(_OPEN_WORDS_RE.search(rest))


# ---------------------------------------------------------------------------
# Annotation -> widget
# ---------------------------------------------------------------------------

def _unwrap_optional(annotation: Any) -> Tuple[Any, bool]:
    """
    Reduce ``Optional[X]`` / ``X | None`` to ``(X, True)``.

    Returns ``(annotation, False)`` for anything else. Unions of several real
    types (``str | Path``) collapse to their first non-``None`` member, which
    is enough to pick a widget.
    """
    origin = typing.get_origin(annotation)
    if origin is not typing.Union and str(origin) != "types.UnionType":
        return annotation, False
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    optional = len(args) != len(typing.get_args(annotation))
    if not args:
        return annotation, optional
    return args[0], optional


def _names_a_sequence(annotation: Any) -> bool:
    """Whether a Union names a list-like member anywhere in it (None aside)."""
    origin = typing.get_origin(annotation)
    if origin is not typing.Union and str(origin) != "types.UnionType":
        return False
    for member in typing.get_args(annotation):
        m_origin = typing.get_origin(member)
        if m_origin in (list, tuple, set, frozenset) or member in (list, tuple, set):
            return True
        if getattr(m_origin, "__name__", "") in {"Sequence", "Iterable", "Collection"}:
            return True
    return False


def widget_for(annotation: Any, name: str = "", default: Any = EMPTY) -> str:
    """
    Choose a rendering hint for one parameter.

    Parameters
    ----------
    annotation : Any
        The evaluated annotation (see :func:`describe`, which resolves string
        annotations for you).
    name : str, optional
        The parameter name. Used only to spot path-ish parameters that are
        annotated as bare ``str``: anything ending in ``_path``, ``_dir``,
        ``_csv`` or ``_wav``.
    default : Any, optional
        The default value, used as a last resort when there is no annotation.

    Returns
    -------
    str
        One of ``text``, ``path``, ``dir``, ``int``, ``float``, ``bool``,
        ``list``.

    Notes
    -----
    ``choice`` is not returned here -- it comes from the docstring's literal
    set, not the annotation, so :func:`describe` applies it afterwards.
    """
    inner, _ = _unwrap_optional(annotation)

    # "one or many" -- Union[PathLike, Sequence[PathLike]] -- is a list as
    # far as a screen or a command line cares: one value is a list of one.
    # we have to look at the union *before* it gets collapsed to its first
    # member below, otherwise this reads as a plain str and a repeated
    # --model-json flag quietly keeps only the last one
    if _names_a_sequence(annotation):
        return "list"

    # concrete scalar types settle it outright. the name heuristic below only
    # gets a say for string-ish parameters -- `include_source_path` is a bool
    # despite the suffix, and guessing from the name there would give us a
    # filesystem prompt for a yes/no question.
    if inner is bool:
        return "bool"
    if inner is int:
        return "int"
    if inner is float:
        return "float"

    lowered = name.lower()
    if lowered.endswith("_dir") or lowered in {"root_dir", "out_dir", "tmp_root"}:
        return "dir"
    if lowered.endswith(("_path", "_csv", "_wav", "_file")):
        return "path"

    if inner is Path or inner is str:
        return "text"

    origin = typing.get_origin(inner)
    if origin in (list, tuple, set, frozenset) or inner in (list, tuple, set):
        return "list"
    if origin is not None and origin in (Sequence, typing.Sequence):
        return "list"
    # typing.Sequence[str] and friends surface as collections.abc.Sequence
    if getattr(origin, "__name__", "") in {"Sequence", "Iterable", "Collection"}:
        return "list"

    # no usable annotation, so we fall back to the shape of the default value.
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, (list, tuple, set)):
        return "list"
    return "text"


# ---------------------------------------------------------------------------
# The public shape
# ---------------------------------------------------------------------------

@dataclass
class ParamSpec:
    """One renderable field: everything a UI needs to draw a single input."""

    name: str
    annotation: Any = None
    annotation_str: str = ""
    default: Any = EMPTY
    required: bool = False
    kw_only: bool = False
    desc: str = ""
    widget: str = "text"
    choices: Optional[List[Any]] = None
    #: The choices are the usual answers, not the only ones: ``{"tiny",
    #: "base"} or str`` names the stock Whisper models and still takes the
    #: path of a model folder. A picker for such a setting keeps a
    #: "something else" row that opens a text box; a closed set has none.
    open_ended: bool = False

    @property
    def has_default(self) -> bool:
        return not isinstance(self.default, _Empty)

    def as_dict(self) -> dict:
        """A JSON-friendly view, for a web UI or an MCP schema."""
        return {
            "name": self.name,
            "type": self.annotation_str,
            "default": None if not self.has_default else _plain(self.default),
            "required": self.required,
            "desc": self.desc,
            "widget": self.widget,
            "choices": self.choices,
            "open_ended": self.open_ended,
        }


@dataclass
class FunctionSpec:
    """A whole function: its summary line and its renderable parameters."""

    name: str
    qualname: str
    summary: str = ""
    doc: str = ""
    params: List[ParamSpec] = field(default_factory=list)

    def get(self, name: str) -> Optional[ParamSpec]:
        return next((p for p in self.params if p.name == name), None)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "qualname": self.qualname,
            "summary": self.summary,
            "params": [p.as_dict() for p in self.params],
        }


def _plain(value: Any) -> Any:
    """Coerce a default into something ``json`` and ``yaml`` both accept."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _annotation_str(raw: Any) -> str:
    """A short human-readable spelling of an annotation."""
    if raw is inspect.Parameter.empty:
        return ""
    if isinstance(raw, str):
        return raw
    return (
        getattr(raw, "__name__", None)
        or str(raw).replace("typing.", "")
    )


def describe(func: Callable) -> FunctionSpec:
    """
    Build a :class:`FunctionSpec` from a live callable.

    Parameters
    ----------
    func : Callable
        Any Taters analysis function. ``**kwargs``-forwarding facade methods
        are the wrong input here -- pass the function they forward *to*, which
        :func:`load_target` will fetch for you.

    Returns
    -------
    FunctionSpec
        Parameters in declaration order. ``*args``/``**kwargs`` are omitted:
        they are not renderable fields.

    Notes
    -----
    Signatures are read with ``eval_str=True``. Every module in this package
    starts with ``from __future__ import annotations``, which makes annotations
    plain strings at runtime; without ``eval_str`` you would get the string
    ``"Optional[Union[str, Path]]"`` instead of a type to dispatch on. If
    evaluation fails -- a name that only exists under ``TYPE_CHECKING``, say --
    we fall back to the unevaluated signature and the string annotations still
    give a usable, if coarser, result.
    """
    try:
        sig = inspect.signature(func, eval_str=True)
    except Exception:
        sig = inspect.signature(func)

    doc = inspect.getdoc(func) or ""
    docs = parse_numpydoc_params(doc)
    summary = doc.strip().splitlines()[0].strip() if doc.strip() else ""

    params: List[ParamSpec] = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        pdoc = docs.get(p.name, ParamDoc(name=p.name))
        default = EMPTY if p.default is inspect.Parameter.empty else p.default
        annotation = None if p.annotation is inspect.Parameter.empty else p.annotation

        spec = ParamSpec(
            name=p.name,
            annotation=annotation,
            annotation_str=_annotation_str(p.annotation) or pdoc.type_str,
            default=default,
            required=isinstance(default, _Empty),
            kw_only=p.kind is inspect.Parameter.KEYWORD_ONLY,
            desc=pdoc.desc,
            widget=widget_for(annotation, p.name, default),
        )
        # the docstring's ``{...}`` set first, since that's written for the
        # reader; the annotation's ``Literal[...]`` when the docstring doesn't
        # have one. either way we know the answers, and a known set is a
        # picker, never a box to spell the answer into.
        choices = (_choices_from_type_str(pdoc.type_str)
                   or _choices_from_annotation(annotation))
        if choices:
            spec.choices = choices
            spec.widget = "choice"
            spec.open_ended = _open_ended(pdoc.type_str)
        params.append(spec)

    return FunctionSpec(
        name=func.__name__,
        qualname=f"{func.__module__}.{func.__name__}",
        summary=summary,
        doc=doc,
        params=params,
    )


def load_target(target: str) -> Callable:
    """
    Import and return the function named by a ``"module:function"`` string.

    Parameters
    ----------
    target : str
        E.g. ``"taters.audio.convert_to_wav:convert_audio_to_wav"``. A dotted
        form without the colon is also accepted, with the last segment taken
        as the attribute name.

    Returns
    -------
    Callable

    Raises
    ------
    ImportError
        Propagated unchanged from the import. Callers are expected to catch
        this and translate it into an install hint -- most Taters modules pull
        heavy optional dependencies, and this function is the point where that
        cost is paid, which is exactly why recipes name their target as a
        string instead of importing it at module load.
    """
    module_name, _, attr = target.partition(":")
    if not attr:
        module_name, _, attr = target.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)
