"""
What every feature table calls its columns, declared once and checked at build
time.

The problem this solves only shows up when two measures meet. Feature tables get
joined side by side into one analysis table, and if two of them use the same
column name, something has to give. `stats.assemble` already handles that: it
refuses two tables with the same stem, and renames any column used by two tables
to ``<stem>__<column>`` in *every* table that has it, so the outcome does not
depend on file order.

That is a good safety net and it stays. But it is **reactive**, and reactive
renaming has a property that is poison for research: the name depends on what
else ran. A column is ``Topic_1`` in a pipeline with one topic model and
``lda_topics__Topic_1`` in a pipeline with two. The same instrument, the same
corpus, two different column names -- so a script written against one study
silently fails against the next, and two results tables cannot be compared
without knowing what else was in each run.

So the names we choose have to be disjoint *up front*. Each feature module
declares what it writes; :func:`overlaps` finds any two declarations that could
produce the same name; and `tests/test_feature_columns.py` fails the build if
any do, naming both offenders. Collisions among shipped measures therefore never
happen, the reactive rename never fires for them, and ``Topic_1`` is ``Topic_1``
in every pipeline forever.

Three kinds of column name exist, and only two of them can be policed here:

* **Fixed** -- names we choose and always write (``flesch_reading_ease``).
* **Patterned** -- names we choose whose tail is a number or a setting
  (``Theme_{n}``, ``msttr_{n}``). Declared as a pattern, matched as one.
* **Dynamic** -- names that come out of the user's own data: the categories in
  their dictionary, their archetype names, however many dimensions their encoder
  has. We cannot know these and do not pretend to. They are declared with a
  sentence saying where they come from, and they are exactly what the reactive
  rename in `assemble` is for.

Adding a measure means adding a `FEATURE_COLUMNS` to its module. The test finds
it through the recipe catalog rather than through a list kept here, so forgetting
is a failure rather than a silence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

__all__ = ["ColumnSpec", "SEPARATOR", "pattern_regex", "overlaps",
           "DECLARING_MODULES", "registry", "reduced_name"]

#: How a namespaced column is spelled when `assemble` has to disambiguate one.
#: Double underscore rather than a dot, for two reasons: it is already what the
#: rest of the codebase uses (`score_model`, `_concept_dicts`, `feature_gather`),
#: and a dot inside a column name has to be quoted in an R or patsy formula,
#: which is exactly where these tables end up.
SEPARATOR = "__"

#: Values a pattern is rendered with when checking two patterns against each
#: other. Small, and includes a multi-digit case because `Theme_1` and
#: `Theme_10` are both real.
_PROBES = ("1", "2", "10", "42")


@dataclass(frozen=True)
class ColumnSpec:
    """
    One module's claim about the columns it writes.

    Parameters
    ----------
    label : str
        What to call this measure in a collision message. A person reads it.
    names : tuple of str
        Column names always written, spelled exactly.
    patterns : tuple of str
        Names whose tail varies with a setting, with a placeholder where the
        varying part goes. ``{n}`` is a number (``"Theme_{n}"``, ``"msttr_{n}"``)
        and ``{*}`` is anything (``"pos_{*}"``, where the tail is a tag). Exactly
        one placeholder per pattern; the rest is matched literally.

        The distinction earns its keep: sentence embeddings write ``e0, e1, …``
        and transformer embeddings write ``e_1, e_2, …``, which differ by one
        character and do not collide -- but a placeholder meaning "anything"
        would report them as if they did, and the noise would train somebody to
        ignore this check.
    reduces_to : str
        What a PCA over these columns produces, as a singular noun. Reducing a
        topic model's topics does not give you "components", it gives you
        something worth naming -- so the topic models say ``"Supertopic"`` and
        everything else keeps ``"Component"``. It is declared here rather than
        worked out by the statistics stage because this is where the codebase
        already knows what a column *is*.
    dynamic : str
        Non-empty when this module *also* writes columns whose names come from
        the user's data rather than from us. The text says where they come from.
        It is documentation, not an exemption: whatever is declared in ``names``
        and ``patterns`` is still checked. The undeclarable rest is what
        `assemble`'s reactive rename exists for.

    Notes
    -----
    Bookkeeping columns are left out on purpose. Several modules write
    ``token_count`` beside their measures, and declaring it would report a
    collision on a column no analysis ever treats as a feature -- `assemble`
    keeps those aside from the feature sets and renames them harmlessly if two
    tables carry one. Declare what somebody would analyze.
    """

    label: str
    names: Tuple[str, ...] = ()
    patterns: Tuple[str, ...] = ()
    reduces_to: str = "Component"
    dynamic: str = ""
    _module: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not (self.names or self.patterns or self.dynamic):
            raise ValueError(
                f"{self.label!r} declares no columns at all. A measure that "
                "writes nothing we can name should say so in `dynamic`.")
        for pattern in self.patterns:
            if pattern.count("{n}") + pattern.count("{*}") != 1:
                raise ValueError(
                    f"{self.label!r} pattern {pattern!r} needs exactly one "
                    "'{n}' or '{*}' -- that is the part allowed to vary.")


def pattern_regex(pattern: str) -> "re.Pattern[str]":
    """A pattern as a regex: everything literal but the one placeholder.

    ``{n}`` becomes ``\\d+`` and ``{*}`` becomes ``.+``. Use ``{n}`` whenever
    the varying part really is a number, because it is what tells ``e{n}`` and
    ``e_{n}`` apart -- two real patterns in this codebase that would otherwise
    read as the same one.
    """
    placeholder, expansion = ("{n}", r"\d+") if "{n}" in pattern else ("{*}", ".+")
    head, _, tail = pattern.partition(placeholder)
    return re.compile(f"{re.escape(head)}{expansion}{re.escape(tail)}$")


def _renderings(pattern: str) -> List[str]:
    """Sample names a pattern could produce, for checking it against another.

    `{*}` gets a couple of non-numeric probes too, because a tag pattern like
    `pos_{*}` has to be seen to overlap `pos_{n}`.
    """
    probes = _PROBES if "{n}" in pattern else _PROBES + ("NN", "x_y")
    placeholder = "{n}" if "{n}" in pattern else "{*}"
    return [pattern.replace(placeholder, probe) for probe in probes]


def _collide(a: ColumnSpec, b: ColumnSpec) -> List[str]:
    """Every column name these two could both produce."""
    # `dynamic` is not a get-out: a module can declare some names and still have
    # others come from the user's data (cohesion does -- fifteen fixed measures
    # plus one column per connective list). What it declares is still checked;
    # `dynamic` only says there are more names here that nothing could check.
    hits: List[str] = []

    shared = set(a.names) & set(b.names)
    hits.extend(sorted(shared))

    for spec, other in ((a, b), (b, a)):
        for pattern in spec.patterns:
            rx = pattern_regex(pattern)
            hits.extend(name for name in other.names if rx.fullmatch(name))
            for theirs in other.patterns:
                if any(rx.fullmatch(rendered) for rendered in _renderings(theirs)):
                    hits.append(f"{pattern} / {theirs}")

    return sorted(set(hits))


def overlaps(specs: Sequence[ColumnSpec]) -> List[Tuple[str, str, List[str]]]:
    """
    Every pair of declarations that could write the same column name.

    Returns
    -------
    list of (label, label, names)
        One entry per colliding pair, with the names they collide on. Empty
        when the declarations are disjoint, which is the only acceptable state
        for the measures Taters ships.
    """
    found: List[Tuple[str, str, List[str]]] = []
    for i, first in enumerate(specs):
        for second in specs[i + 1:]:
            shared = _collide(first, second)
            if shared:
                found.append((first.label, second.label, shared))
    return found


def by_module(specs: Sequence[ColumnSpec]) -> Dict[str, ColumnSpec]:
    """Declarations keyed by the module that made them, for error messages."""
    return {spec._module or spec.label: spec for spec in specs}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: Every module that declares a `FEATURE_COLUMNS`. Spelled out here rather than
#: discovered, for two reasons: importing is cheap and scanning is not, and a
#: runtime registry should not depend on the wizard's recipe catalog (that would
#: point `helpers` at `ui`, which is backwards).
#:
#: Forgetting to add one is caught rather than tolerated --
#: `tests/test_feature_columns.py` checks this list against the recipe catalog,
#: which is where new feature steps get registered anyway.
DECLARING_MODULES: Tuple[str, ...] = (
    "taters.helpers.feature_gather",
    "taters.helpers.text_gather",
    "taters.text.analyze_entropy",
    "taters.score_model",
    "taters.text.analyze_cohesion",
    "taters.text.analyze_lexical_richness",
    "taters.text.analyze_parts_of_speech",
    "taters.text.analyze_readability",
    "taters.text.analyze_sentiment_vader",
    "taters.text.analyze_with_archetypes",
    "taters.text.analyze_with_dictionaries",
    "taters.text.analyze_with_norms",
    "taters.text.analyze_word_count",
    "taters.text.build_doc_term_matrix",
    "taters.text.extract_sentence_embeddings",
    "taters.text.finetune_predictor",
    "taters.text.hf_classifier",
    "taters.text.topic_model_lda",
    "taters.text.topic_model_mem",
    "taters.text.topic_model_nmf",
    "taters.text.transformer_embeddings",
    "taters.text.word_vectors",
)

_REGISTRY: Dict[str, ColumnSpec] = {}


def registry() -> Dict[str, ColumnSpec]:
    """Every declaration, keyed by module. Built once, and never raises.

    A module that cannot be imported (an optional dependency is missing) is
    skipped rather than fatal: the registry is used to *name* things nicely,
    and a missing encoder should not stop a run that never wanted one.
    """
    if not _REGISTRY:
        import importlib

        for path in DECLARING_MODULES:
            try:
                spec = getattr(importlib.import_module(path), "FEATURE_COLUMNS", None)
            except Exception:                   # pragma: no cover - optional extras
                continue
            if spec is not None:
                _REGISTRY[path] = spec
    return dict(_REGISTRY)


def reduced_name(columns: Sequence[str], *, set_name: str = "") -> str:
    """
    What to call a component built from these columns: "Supertopic", or
    "Component" when nothing more specific is known.

    Matched on the column names themselves rather than threaded down from the
    pipeline, because by the time the statistics stage reduces a feature set,
    the set is just a list of column names in a table -- whatever produced it
    is long out of scope.
    """
    trimmed = []
    for column in columns:
        prefix = f"{set_name}{SEPARATOR}"
        trimmed.append(column[len(prefix):]
                       if set_name and column.startswith(prefix) else column)

    if not trimmed:
        return "Component"

    for spec in registry().values():
        if spec.reduces_to == "Component":
            continue
        literal = set(spec.names)
        checks = [pattern_regex(p) for p in spec.patterns]
        # *every* column has to belong, not just one. a feature set that mixes
        # a topic model with readability indices reduces to something that is
        # genuinely not a supertopic, and the combined "all" set -- which the
        # statistics stage builds by default -- is exactly that mixture. an
        # `any` here named its components supertopics on the strength of one
        # matching column.
        if all(c in literal or any(rx.fullmatch(c) for rx in checks)
               for c in trimmed):
            return spec.reduces_to
    return "Component"
