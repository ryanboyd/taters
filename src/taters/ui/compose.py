"""
Turn a set of chosen features into a runnable preset.

This module is pure: it takes recipe ids and option overrides, and returns a
dict shaped exactly like the YAML in ``taters/pipelines/presets/``. It touches
no files, imports no analysis code, and runs in microseconds -- which is what
makes it worth testing hard. ``tests/test_compose.py`` puts its output through
the same validator that guards the shipped presets, so the composer
structurally cannot emit a step naming a parameter that does not exist.

The job has three parts:

1. **Closure.** The user checks "Readability"; that needs the merged transcript
   table, which needs a transcript, which needs a WAV. Four steps from one tick.
2. **Ordering.** Dependencies first, and -- matching how both shipped presets
   are laid out -- every per-file step before every run-once step.
3. **Metadata.** A full ``meta:`` block, so a composed preset is a first-class
   citizen: ``--list-presets`` and ``--describe-preset`` work on it exactly as
   they do on the built-ins.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .recipes import (
    BASE_VARS,
    CAPABILITIES,
    RECIPES,
    SOURCE_VARS,
    SOURCES,
    TEXT_INPUT_KEYS,
    Recipe,
    by_id,
    providers_of,
    DEFAULT_LEVEL,
    TEXT_IDENTITY,
    level_aware,
    level_by_id,
    text_binding,
)

__all__ = ["compose", "resolve_selection", "pending_choices", "feature_tables",
           "ComposeError", "slugify"]


class ComposeError(RuntimeError):
    """A selection that cannot be turned into a runnable pipeline."""


def slugify(text: str) -> str:
    """Reduce a title to a safe preset id / filename stem."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "custom_pipeline"


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

def _requires(recipe: Recipe, source: str) -> frozenset:
    """
    What a recipe needs *given where the text is coming from*.

    A text analyzer requires the merged transcript table only on the media
    path. Point it at a folder of essays and that requirement evaporates --
    along with transcription, WAV conversion and ffmpeg. This is what
    collapses a 7-step audio pipeline into a 1-step text one.

    Only the *transcript* requirement evaporates, though. A text step can also
    depend on another text step's artifact -- the doc-term matrix reads the
    n-gram frequency list -- and dropping the whole requires set on the text
    path silently severed that edge: the frequency step was neither
    auto-included nor ordered first.
    """
    if source != "media" and recipe.text_input:
        return recipe.requires - {"unified_transcripts_csv"}
    return recipe.requires


def resolve_selection(
    selected: Sequence[str],
    *,
    providers: Optional[Dict[str, str]] = None,
    source: str = "media",
) -> List[Recipe]:
    """
    Expand a user's picks into the full, ordered list of steps to run.

    Parameters
    ----------
    selected : sequence of str
        Recipe ids the user checked off.
    providers : dict[str, str], optional
        How to satisfy a capability that more than one recipe can provide, e.g.
        ``{"transcript_csv": "diarize"}``. When a needed capability has exactly
        one provider it is chosen automatically; when it has several and no
        choice was given, the *first in catalog order* wins. The wizard always
        asks rather than relying on that fallback, but a scripted caller should
        not have to.
    source : {"media", "txt_dir", "csv"}, default "media"
        Where the text comes from. Anything but ``"media"`` means the text
        already exists, so the text analyzers stop requiring a transcript and
        the whole audio half of the catalog drops out of the closure.

    Returns
    -------
    list[Recipe]
        Every step to run, dependencies included, in execution order: `item`
        steps first (each after whatever it depends on), then `global` steps.

    Raises
    ------
    ComposeError
        If a requirement has no provider at all, or if the graph is cyclic.

    Notes
    -----
    Capability satisfaction is checked against the *chosen set*, not against
    the catalog. That distinction matters: ``transcript_csv`` has two providers,
    and pulling in both would give the run two transcription steps writing to
    the same ``save_as``.
    """
    if source not in SOURCES:
        raise ComposeError(
            f"unknown source {source!r}. Known: {', '.join(sorted(SOURCES))}"
        )

    try:
        unsupported = sorted({r for r in selected
                              if source not in by_id(r).sources})
    except KeyError as e:
        # `by_id` throws a KeyError on a typo'd id, but we promise ComposeError
        # here (and that's what callers catch), so let's convert it.
        raise ComposeError(str(e).strip("'\"")) from None
    if unsupported:
        raise ComposeError(
            f"{unsupported} cannot run on {source!r} input "
            f"({SOURCES[source]}). Ask recipes.user_facing({source!r}) for what can."
        )

    providers = dict(providers or {})

    # if the user explicitly picked a provider, that settles it -- we drop any
    # *other* producer of that capability from the selection. otherwise, ticking
    # both "Transcript (one speaker)" and "Transcript with speaker labels" puts
    # both in the pipeline even after we asked them to choose, and since they
    # share `save_as: diar`, the second one silently clobbers the first for
    # every step downstream.
    rejected = {
        r.id
        for capability, keep in providers.items()
        for r in providers_of(capability)
        if r.id != keep
    }
    trimmed = [r for r in selected if r not in rejected]
    if selected and not trimmed:
        raise ComposeError(
            "the provider choice removed everything that was selected: "
            f"{sorted(set(selected) & rejected)} contradict "
            f"{ {k: v for k, v in providers.items()} }"
        )
    selected = trimmed

    chosen: Dict[str, Recipe] = {}

    # first, let's figure out what the user's own ticks produce, up front. the
    # walk used to only look at what it had *already added*, so the result
    # depended on tick order: ["readability", "diarize"] hit readability's need
    # for a transcript first, auto-picked `transcribe`, and then added the
    # user's `diarize` on top -- two transcript steps sharing save_as="diar",
    # silently clobbering each other. the reverse order worked fine. ugh.
    selected_producers: Dict[str, str] = {}
    for rid in selected:
        for capability in by_id(rid).produces:
            if (capability in selected_producers
                    and selected_producers[capability] != rid):
                # two ticked producers and nobody chose between them. if we
                # shipped both they'd share a save_as and silently clobber each
                # other downstream. the wizard always asks first
                # (resolve_providers), so if we get here it's a programmatic
                # caller's mistake -- let's make it a loud one.
                raise ComposeError(
                    f"'{selected_producers[capability]}' and '{rid}' both "
                    f"produce '{capability}'. Pass providers="
                    f"{{'{capability}': <one of them>}} to choose."
                )
            selected_producers.setdefault(capability, rid)

    def add(recipe_id: str, trail: tuple = ()) -> None:
        if recipe_id in chosen:
            return
        if recipe_id in trail:
            cycle = " -> ".join(trail[trail.index(recipe_id):] + (recipe_id,))
            raise ComposeError(f"recipes form a cycle: {cycle}")
        recipe = by_id(recipe_id)
        chosen[recipe_id] = recipe
        trail = trail + (recipe_id,)

        for capability in sorted(_requires(recipe, source)):
            # already covered by something the user picked directly? skip it.
            if any(capability in r.produces for r in chosen.values()):
                continue
            options = providers_of(capability)
            if not options:
                raise ComposeError(
                    f"'{recipe_id}' needs '{capability}', which nothing produces"
                )
            picked = providers.get(capability)
            if picked is None:
                # a producer the user actually ticked beats the catalog's first
                # option, whether or not the walk has gotten to it yet.
                picked = selected_producers.get(capability, options[0].id)
                providers[capability] = picked
            elif picked not in {o.id for o in options}:
                raise ComposeError(
                    f"'{picked}' was chosen for '{capability}' but does not "
                    f"produce it (options: {', '.join(o.id for o in options)})"
                )
            add(picked, trail)

        for follow_on in recipe.auto_with:
            add(follow_on, trail)

    for recipe_id in selected:
        add(recipe_id)

    # lastly, every feature table gets descriptive stats whether they asked or
    # not. the first thing anyone does with a new measure is look at its
    # distribution, and that used to mean opening each CSV by hand. one rule
    # here beats sticking an `auto_with` on thirteen recipes.
    if any(getattr(r, "feature_table", False) for r in chosen.values()):
        add("describe_features")

    return _order(chosen.values(), source)


def pending_choices(selected: Sequence[str], *, source: str = "media") -> Dict[str, List[Recipe]]:
    """
    Capabilities this selection needs that more than one recipe could satisfy.

    Parameters
    ----------
    selected : sequence of str
        The recipe ids the user ticked.
    source : {"media", "txt_dir", "csv"}, default "media"
        Where the text comes from. On a text source there is nothing to
        transcribe, so this returns empty and the wizard skips the question.

    Returns
    -------
    dict[str, list[Recipe]]
        Capability -> the recipes that could provide it. Empty when nothing is
        ambiguous.

    Notes
    -----
    The need is resolved *transitively*, which is the whole reason this is not
    a one-line check over ``requires``. Someone who ticks only "Readability
    scores" has not asked for a transcript and does not mention one anywhere in
    their selection -- but readability reads the merged transcript table, and
    that merge step is what needs a transcript. Ask them how to make one
    anyway, because the two answers differ by a multi-gigabyte install.

    A capability whose producer the user ticked directly is not returned: they
    have already answered.
    """
    steps = resolve_selection(selected, source=source)
    picked = set(selected)

    out: Dict[str, List[Recipe]] = {}
    for capability in CAPABILITIES:
        options = providers_of(capability)
        if len(options) < 2:
            continue
        if not any(capability in _requires(step, source) for step in steps):
            continue
        if any(option.id in picked for option in options):
            continue
        out[capability] = options
    return out


def _order(recipes: Iterable[Recipe], source: str = "media") -> List[Recipe]:
    """
    Sort steps so every step follows the ones it depends on.

    Kahn's algorithm, with catalog position as the tiebreak so the same
    selection always produces the same file. `item` steps are emitted before
    `global` ones: the runner fans `item` steps out across inputs and puts a
    barrier before each `global` step, so a `global` step that reads what the
    per-file steps wrote has to come after all of them.
    """
    pool = {r.id: r for r in recipes}
    catalog_pos = {r.id: i for i, r in enumerate(RECIPES)}
    available: Set[str] = set()
    ordered: List[Recipe] = []

    # "item" sorts before "global"; within a scope, we go by catalog order.
    def rank(recipe: Recipe) -> tuple:
        return (0 if recipe.scope == "item" else 1, catalog_pos[recipe.id])

    remaining = sorted(pool.values(), key=rank)
    while remaining:
        ready = [r for r in remaining if _requires(r, source) <= available]
        if not ready:
            stuck = ", ".join(sorted(r.id for r in remaining))
            raise ComposeError(f"cannot order steps; unsatisfied: {stuck}")
        # we never want a global step jumping ahead of an item step that's also
        # ready -- rank() already handles that, and `ready` keeps the order.
        pick = ready[0]
        remaining.remove(pick)
        ordered.append(pick)
        available |= pick.produces
    return ordered


# ---------------------------------------------------------------------------
# Preset assembly
# ---------------------------------------------------------------------------

def _collect_vars(steps: Sequence[Recipe]) -> Dict[str, dict]:
    """
    Merge each step's variable contributions, first writer winning.

    First-wins matters for exactly one variable: ``whisper_model``. Plain
    transcription defaults it to ``base.en`` and diarization to ``base``, and
    both come before the embedding step in execution order -- so whichever
    transcript producer the user chose sets it, and the embedding step inherits
    that choice instead of overriding it.
    """
    merged: Dict[str, dict] = {k: dict(v) for k, v in BASE_VARS.items()}
    for recipe in steps:
        for name, spec in recipe.vars.items():
            merged.setdefault(name, dict(spec))
    return merged


# what identifies one utterance row of the transcript table: the file, the
# voice, and when. unique per row (whisper stamps each segment's start), and
# together they're our join key back to the transcript itself.
_MEDIA_ROW_ID = ("source", "speaker", "start_time")


# which variable holds the input path, per source. no "media" entry because
# there the path is a run-time argument (--root_dir), not a preset variable.
_INPUT_VAR = {"txt_dir": "input_dir", "csv": "input_csv"}


def _suffixed(path: str, level_id: str) -> str:
    """`features/readability.csv` -> `features/readability_by-utterance.csv`."""
    head, dot, ext = path.rpartition(".")
    if not dot:
        return f"{path}_by-{level_id}"
    return f"{head}_by-{level_id}.{ext}"


def _is_pointless_merge(recipe: Recipe, source: str, level: Optional[str],
                        group_by: Sequence[str]) -> bool:
    """
    Whether this merge would have nothing to merge and is dropped outright.

    At the raw level -- one result row per utterance, per spreadsheet row, per
    text file -- an ``aggregate: True`` step keyed on what already makes a row
    unique is a copy of its input with ``__mean`` stapled to the column names.
    Worse than useless: `aggregate_features` keeps only its group keys among
    the non-numeric columns, so the copy *loses* the source/speaker/id columns
    the per-row output exists to carry. No step beats a step that subtracts.
    """
    if not (level_aware(recipe) and recipe.with_.get("aggregate") is True):
        return False
    spec = level_by_id(source, _level_for(source, level, group_by))
    keys = list(group_by) if spec.group_by is None else list(spec.group_by)
    return not keys


def _level_for(source: str, level: Optional[str],
               group_by: Sequence[str]) -> Optional[str]:
    """
    The level to use, given what the caller actually said.

    Naming grouping columns for a spreadsheet *is* choosing the grouped level,
    so a caller that passes `group_by` and no level gets what it plainly meant
    rather than having its argument silently ignored.
    """
    if level is None and source == "csv" and group_by:
        return "group"
    return level


def _bind_source(
    recipe: Recipe,
    source: str,
    *,
    text_cols: Sequence[str],
    id_cols: Sequence[str],
    text_mode: str = "concat",
    group_by: Sequence[str] = (),
    level: Optional[str] = None,
    metadata_group_by: Optional[Sequence[str]] = None,
    gathered_artifact: str = "",
) -> dict:
    """
    Render a step, rewiring its input for a non-media source.

    The rewire *replaces* the whole input group rather than patching it. A text
    analyzer takes exactly one of ``analysis_csv`` / ``csv_path`` / ``txt_dir``
    and ignores the arguments belonging to the other two, so a surviving
    ``csv_path: "{{transcripts_all}}"`` next to a new ``txt_dir`` would not
    error -- it would quietly read a transcript file that this run never
    produced.

    ``metadata_group_by`` is for a run that joins the texts at one grain and
    then averages the measures up to a coarser one: the text steps still
    gather at the joining keys, while the metadata has to arrive at the grain
    the statistics actually see. Left unset both use ``group_by``, which is
    every run that does not average.
    """
    step = recipe.to_step()

    spec = level_by_id(source, _level_for(source, level, group_by))
    # for a spreadsheet the level doesn't name any columns (the user picks
    # them), so we use the caller's `group_by` instead.
    keys = list(group_by) if spec.group_by is None else list(spec.group_by)

    if source == "csv" and recipe.keys_like_metadata:
        # a table that joins on text_id has to build it EXACTLY the way the
        # text gather does, or the join finds nothing: same group keys when
        # rows get combined, same id columns otherwise. we're copying
        # text_binding's csv branch here on purpose -- if the two ever
        # disagree, the join breaks. the recipes say which tables need it
        # rather than us guessing from what they produce, because the
        # spreadsheet feature table needs the same treatment and produces
        # something else entirely.
        meta_keys = list(metadata_group_by) \
            if metadata_group_by is not None else keys
        if meta_keys:
            step["with"]["group_by"] = meta_keys
        elif id_cols:
            step["with"]["id_cols"] = list(id_cols)

    if source != "media" and recipe.text_input:
        kept = {k: v for k, v in step["with"].items() if k not in TEXT_INPUT_KEYS}
        kept.update(text_binding(
            source,
            text_cols=text_cols,
            id_cols=id_cols,
            pass_through="pass_through_cols" in recipe.with_,
            text_mode=text_mode,
            group_by=keys,
            gathered_artifact=gathered_artifact,
        ))
        step["with"] = kept

    elif source == "media" and level_aware(recipe):
        # the level decides the grain of the text features and of the merges
        # that finish them off. audio steps keep whatever grain they declare --
        # see `level_aware` for why "one row per utterance" doesn't mean
        # anything for a per-speaker WAV.
        if "group_by" in recipe.with_:
            if keys:
                step["with"]["group_by"] = keys
            elif step["with"].get("aggregate") is True:
                # a merge with `aggregate: True` and no key collapses
                # *everything* into a single row. at the raw level there's
                # nothing to combine, so we group on what already makes a row
                # unique and it becomes a tidy copy.
                step["with"]["group_by"] = TEXT_IDENTITY
            else:
                # the raw level. we used to just drop group_by, which left
                # id_cols=["source", "speaker"] and gave every utterance by one
                # speaker the same text_id -- N identical-looking rows with no
                # utterance index. start_time is what tells them apart, and
                # carrying it also gives the row a join key back to the
                # transcript.
                del step["with"]["group_by"]
                if "id_cols" in step["with"]:
                    step["with"]["id_cols"] = list(_MEDIA_ROW_ID)
        elif not keys and "pass_through_cols" in step["with"]:
            # the pass-through analyzer (sentence embeddings) at the raw
            # level: its merge only keeps its group keys, so everything a
            # per-utterance row needs to be identifiable has to ride along.
            step["with"]["pass_through_cols"] = list(_MEDIA_ROW_ID)

    # two runs at different levels write different tables. if they shared a
    # filename the second would silently overwrite the first, and nothing in
    # the file would say which level made it. the default level keeps its plain
    # name so existing output paths (and the shipped presets) don't change.
    if level_aware(recipe) and spec.id != DEFAULT_LEVEL[source]:
        for key in ("out_features_csv", "out_csv"):
            if isinstance(step["with"].get(key), str):
                step["with"][key] = _suffixed(step["with"][key], spec.id)

    step["with"].update(recipe.source_with.get(source, {}))

    # what makes one row of an analyzer's output unique, for a merge that's
    # meant to tidy rather than collapse. `text_id` alone isn't enough under
    # `mode="separate"`, since that one emits a row per text column on purpose.
    # (merges whose keys would make them pure copies never get here --
    # `compose` drops them; see `_is_pointless_merge`.)
    identity = ["text_id"] + (["source_col"] if text_mode == "separate" else [])
    for key, value in list(step["with"].items()):
        if value == TEXT_IDENTITY:
            step["with"][key] = identity
    return step


def feature_tables(steps, source: str = "media", level: Optional[str] = None,
                   group_by: Sequence[str] = (),
                   picked: Optional[Sequence[str]] = None) -> List[tuple]:
    """
    The steps whose output joins into an analysis table, and what to call it.

    Returns ``(recipe, name)`` pairs, in step order.

    ``picked`` is what the user actually ticked. A feature table pulled in
    only as another step's input does not join: the topic model needs a
    document-term matrix, and someone who asked for topic scores has not
    asked for every one of the matrix's five thousand term columns to be
    correlated with their outcome as well. Without the list, every feature
    table in ``steps`` joins, which is what a hand-written pipeline means.

    Two steps can describe one table. "Sentence embeddings" writes a row per
    utterance; "Merge sentence embeddings" averages those to a row per unit
    of analysis -- and only one of them is the per-text table the statistics
    can join. Whichever survives, the *name* comes from the measure rather
    than from the plumbing: a screen offering "Merge sentence embeddings"
    alongside "Sentence embeddings" reads as two feature sets, which is a
    thing they have never been (a real report).

    One rule, one answer: the composer wires the assemble step from this and
    the wizard builds its picker from it, so the tables offered are exactly
    the tables used. They were computed separately, and disagreed.
    """
    kept = [r for r in steps
            if getattr(r, "feature_table", False)
            and not _is_pointless_merge(r, source, level, group_by)]
    superseded = {}
    for r in kept:
        for other in kept:
            if (other is not r and (other.requires & r.produces)
                    # only a *merge* of the measure supersedes it. a step
                    # that eats one table to compute a different one (the
                    # topic model reading the document-term matrix) is a
                    # second table, not the same one re-shaped. if we folded
                    # the two together we'd end up labeling topic scores
                    # "Document-term matrix", which is just wrong.
                    and level_aware(other)
                    and other.with_.get("aggregate") is True):
                # `other` collects what `r` produced, so `other` is the
                # table -- but under `r`'s name, since that's the measure.
                superseded[r.id] = other.id
    names = {r.id: r.label for r in kept}
    for lost, winner in superseded.items():
        names[winner] = names.get(lost, names[winner])
    if picked is not None:
        wanted = set(picked)
        # a merge survives if the measure it merges was picked: the user
        # ticked "Sentence embeddings", and the per-text table is the merge.
        kept = [r for r in kept
                if r.id in wanted
                or any(lost in wanted for lost, winner in superseded.items()
                       if winner == r.id)]
    return [(r, names[r.id]) for r in kept if r.id not in superseded]


#: The setting that names a feature step's output file, whichever of the two
#: spellings the step uses. Its stem is the table's name in every analysis.
_OUTPUT_KEYS = ("out_features_csv", "out_csv")


def table_names(steps, source: str = "media", level: Optional[str] = None,
                group_by: Sequence[str] = (),
                picked: Optional[Sequence[str]] = None, *,
                overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                var_values: Optional[Dict[str, Any]] = None,
                var_specs: Optional[Dict[str, dict]] = None) -> List[tuple]:
    """
    What each feature table will be called in the analyses, with the step it
    comes from.

    Returns ``(name, label)`` pairs in step order -- ``("dictionary",
    "Dictionary categories")`` -- for the tables :func:`feature_tables` says
    will join. The name is the stem of the file the step writes, which is how
    the assemble step names a feature set and how ``pca`` and
    ``unverified_ok`` refer to one. It is worked out the way the composer
    will write it: the step bound to this source and level (a non-default
    level suffixes the filename), the user's own override of the output path
    on top, and every ``{{var:...}}`` in it rendered from the live variable
    values -- so a matrix under ``weighting: tfidf`` is offered as
    ``doc_term_matrix_tfidf``, which is the name the run will use.

    This exists so the wizard can *offer* the names instead of asking for
    them to be typed: "off, all, or the name of a feature set" was a text
    box, and the names it wanted were file stems nobody had seen yet.
    """
    overrides = overrides or {}
    var_values = var_values or {}
    var_specs = _collect_vars(steps) if var_specs is None else var_specs

    def live(name: str) -> str:
        if name in var_values:
            return str(var_values[name])
        return str(var_specs.get(name, {}).get("default", ""))

    out: List[tuple] = []
    for recipe, label in feature_tables(steps, source, level, group_by,
                                        picked=picked):
        step = _bind_source(recipe, source, text_cols=("text",), id_cols=(),
                            group_by=group_by, level=level)
        with_ = {**step["with"], **overrides.get(recipe.id, {})}
        template = next((with_[k] for k in _OUTPUT_KEYS
                         if isinstance(with_.get(k), str)), None)
        name = recipe.save_as
        if template is not None:
            rendered = re.sub(r"\{\{var:([^}]+)\}\}",
                              lambda m: live(m.group(1).strip()), template)
            stem = rendered.replace("\\", "/").rsplit("/", 1)[-1]
            stem = stem.rsplit(".", 1)[0] if "." in stem else stem
            # if we can't render the template (another artifact's name in
            # the path) it's not a name worth offering; the artifact key is
            # at least a stable word for the table.
            if stem and "{{" not in stem:
                name = stem
        out.append((name, label))
    return out


def _feature_table_templates(steps, source: str, level: Optional[str],
                             group_by: Sequence[str],
                             picked: Sequence[str]) -> List[str]:
    """The artifact templates the assemble step joins."""
    return ["{{" + r.save_as + "}}"
            for r, _ in feature_tables(steps, source, level, group_by,
                                       picked=picked)]


def _apply_overrides(step: dict, overrides: Dict[str, Any]) -> dict:
    """Write per-step parameter overrides into a step's ``with:`` block."""
    if overrides:
        step["with"] = {**step["with"], **overrides}
    return step


def compose(
    selected: Sequence[str],
    *,
    providers: Optional[Dict[str, str]] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    var_values: Optional[Dict[str, Any]] = None,
    name: str = "My pipeline",
    file_type: str = "any",
    root_dir: Optional[str] = None,
    notes: str = "",
    source: str = "media",
    input_path: Optional[str] = None,
    text_cols: Sequence[str] = ("text",),
    id_cols: Sequence[str] = (),
    text_mode: str = "concat",
    group_by: Sequence[str] = (),
    delimiter: str = ",",
    level: Optional[str] = None,
    feature_levels: Sequence[Sequence[str]] = (),
    row_filters: Sequence[Sequence[Any]] = (),
    model_plans: Sequence[Any] = (),
) -> dict:
    """
    Build a complete preset dict from a set of chosen features.

    Parameters
    ----------
    selected : sequence of str
        Recipe ids the user checked off. Prerequisites are added for you.
    providers : dict[str, str], optional
        Capability -> recipe id, for capabilities with more than one provider.
        In practice this is ``{"transcript_csv": "transcribe" | "diarize"}``.
    overrides : dict[str, dict], optional
        Per-step parameter overrides, keyed by recipe id:
        ``{"transcribe": {"beam_size": 1}}``. These are written straight into
        the step's ``with:`` block, so they beat the ``{{var:...}}`` templates.
    var_values : dict[str, Any], optional
        Overrides for the preset's ``vars:`` block, keyed by variable name.
        Unknown names are kept -- a preset may legitimately carry a variable
        that no shipped recipe declares.
    name : str, default "My pipeline"
        Human title. The preset id is its slug.
    file_type : {"audio", "video", "any"}, default "any"
        Recorded in ``meta.inputs`` and in the generated ``cli_example``, so
        re-running from the command line does not require remembering it.
    root_dir : str, optional
        The input folder, baked into ``cli_example`` so that line is
        copy-pasteable rather than a template to fill in.
    notes : str, optional
        Free text appended to ``meta.notes``.
    source : {"media", "txt_dir", "csv"}, default "media"
        Where the text comes from. The two non-media sources rewire the text
        analyzers to read the user's own files, and drop transcription and
        everything under it.
    input_path : str, optional
        The folder of ``.txt`` files or the spreadsheet, depending on
        ``source``. Stored as the ``input_dir`` / ``input_csv`` variable rather
        than baked into each step, so the preset can be re-pointed with
        ``--var`` instead of edited.
    text_cols, id_cols : sequence of str
        For ``source="csv"``: which columns hold the text, and which columns
        identify a row. Ignored for the other sources, where the answer is not
        the user's to give.
    text_mode : {"concat", "separate"}
        For ``source="csv"`` with more than one text column: measure them
        joined together, or one at a time.
    row_filters : sequence of [column, operator, value], optional
        Which spreadsheet rows to keep, judged on the spreadsheet's own
        columns, before anything is measured: ``[["age", ">=", 18]]``. Any
        makes the text gather an explicit first step and points every text
        analyzer at the table it writes, because an analyzer gathers for
        itself and has no argument for this.
    feature_levels : sequence of sequences of str, optional
        Average the measures up, once per entry, in order: ``[("conv",
        "speaker"), ("conv",)]`` measures every row, averages to one row per
        speaker per conversation, then averages those to one row per
        conversation. Each entry may only name columns the entry before it
        grouped on, because the rest are gone by then. Empty means the rows
        are analyzed as extraction produced them, which is what every run did
        before this existed.

    Returns
    -------
    dict
        A preset with ``meta``, ``vars``, and ``steps`` keys, ready to hand to
        ``yaml.safe_dump`` or straight to
        :func:`taters.pipelines.run_pipeline.run_preset`.

    Raises
    ------
    ComposeError
        Propagated from :func:`resolve_selection`.
    """
    if not selected:
        raise ComposeError("nothing selected -- pick at least one thing to extract")

    steps = resolve_selection(selected, providers=providers, source=source)
    overrides = dict(overrides or {})
    var_values = dict(var_values or {})
    steps = _wire_controls(steps, model_plans, source=source,
                           overrides=overrides, var_values=var_values)

    # the stats assemble step joins the selected feature tables, and only we
    # (the composer) can see which ones those are. we inject them as a default
    # UNDER any caller override, so the wizard's "which tables should feed the
    # analysis" narrowing (just an ordinary override) still wins.
    tables = _feature_table_templates(steps, source, level, group_by,
                                      picked=selected)
    for r in steps:
        if getattr(r, "consumes_feature_tables", False):
            # a table we only joined so a filter could name its column isn't
            # a feature. this bit us once: a word count computed for "drop
            # texts under 25 words" got left in the feature list and showed
            # up as a ridge *predictor* -- a result that looks fine and isn't.
            filter_only = set(overrides.get(r.id, {}).get("filter_csvs") or [])
            # never itself, either: the scoring step's own output is a feature
            # table too, and a step listed among its own inputs would sit
            # waiting on an artifact that it's the one producing.
            mine = "{{" + r.save_as + "}}"
            wanted = [tpl for tpl in tables
                      if tpl not in filter_only and tpl != mine]
            if not wanted and not getattr(r, "feature_tables_optional", False):
                raise ComposeError(
                    f"'{r.label}' needs at least one feature table -- select "
                    f"a feature to extract first")
            overrides[r.id] = {"feature_csvs": wanted,
                               **overrides.get(r.id, {})}

    chosen_level = level_by_id(source, _level_for(source, level, group_by))

    var_specs = _collect_vars(steps)
    for key, spec in SOURCE_VARS[source].items():
        var_specs.setdefault(key, dict(spec))
    if input_path is not None and source != "media":
        var_specs[_INPUT_VAR[source]]["default"] = input_path
    if source == "csv":
        # the caller sniffs this from the file itself, because the extension
        # is only a hint: plenty of tab-separated files are named .csv or .txt.
        var_specs["csv_delimiter"]["default"] = delimiter
    for key, value in (var_values or {}).items():
        if key in var_specs:
            var_specs[key]["default"] = value
        else:
            var_specs[key] = {"default": value, "desc": ""}

    preset_id = slugify(name)
    chosen_ids = {r.id for r in steps}
    extras = sorted({e for r in steps for e in r.extras})
    needs_ffmpeg = any(r.needs_ffmpeg for r in steps)

    picked_labels = [by_id(i).label for i in selected if i in chosen_ids]
    added = [r.label for r in steps if r.id not in set(selected)]

    summary = "Built by the Taters wizard. Extracts: " + "; ".join(picked_labels) + "."
    note_lines = [
        ("Generated by the Taters setup wizard. It is an ordinary preset -- "
         "edit it, re-run it, or hand it to someone else."),
        "",
        f"You asked for: {', '.join(picked_labels)}.",
    ]
    if added:
        note_lines += ["", f"Added automatically as prerequisites: {', '.join(added)}."]
    # we write this down because it's the single decision that changes the
    # numbers most and otherwise leaves no trace in the output -- the same
    # measure can differ by a third between levels.
    note_lines += ["", f"One row of results describes: {chosen_level.label.lower()} "
                       f"-- {chosen_level.help}"]
    note_lines += ["", "Safe to re-run; steps short-circuit if their outputs exist."]
    if notes:
        note_lines += ["", notes]

    meta = {
        "id": preset_id,
        "title": name,
        "summary": summary,
        "inputs": _inputs_meta(source, file_type, input_path),
        "requirements": {
            "cpu": True,
            "gpu/cuda": "optional",
            "ffmpeg": needs_ffmpeg,
            "extras": extras,
        },
        "variables": {
            k: {"default": v.get("default"), "desc": v.get("desc", "")}
            for k, v in var_specs.items()
        },
        "tags": sorted({t for r in steps for t in _tags_for(r)}),
        "level": chosen_level.id,
        "version": 1,
        "notes": "\n".join(note_lines),
        "cli_example": _cli_example(preset_id, source, file_type, root_dir),
    }

    # what one row of the analysis table will be. averaging moves it: the
    # text steps still gather at the joining keys, but the outcomes have to
    # arrive at the grain the statistics see or the join has nothing to match.
    averaging = [list(lv) for lv in (feature_levels or ()) if lv]
    final_grain = averaging[-1] if averaging else None

    # leaving rows out before anything is measured turns the gather into a
    # step of its own; every text analyzer then reads what it wrote.
    pre_gather = None
    keep = [list(f) for f in (row_filters or ())]
    if source == "csv" and keep:
        pre_gather = _pre_gather_step(
            text_cols=text_cols, id_cols=id_cols, text_mode=text_mode,
            group_by=group_by, row_filters=keep,
            # whatever the analyzers pass through, plus whatever the
            # averaging will need to group by, has to survive the gather
            carry=sorted({*group_by, *id_cols,
                          *(c for lv in averaging for c in lv)}))

    built = [
        _apply_overrides(
            _bind_source(r, source, text_cols=text_cols, id_cols=id_cols,
                         text_mode=text_mode, group_by=group_by,
                         level=level, metadata_group_by=final_grain,
                         gathered_artifact=_GATHERED if pre_gather else ""),
            overrides.get(r.id, {}),
        )
        for r in steps
        if not _is_pointless_merge(r, source, level, group_by)
    ]
    built = _splice_averaging(
        built, steps, averaging, source=source, text_cols=text_cols,
        id_cols=id_cols, text_mode=text_mode, group_by=group_by, level=level,
        picked=selected, overrides=overrides, var_specs=var_specs)
    built = _splice_model_extractions(
        built, steps, model_plans, source=source, text_cols=text_cols,
        id_cols=id_cols, text_mode=text_mode, group_by=group_by, level=level,
        overrides=overrides, var_specs=var_specs)
    if pre_gather:
        # last, and at the front: the splices above walk `built` alongside the
        # resolved recipe list, and a step with no recipe behind it would put
        # the two out of step with each other.
        built = [pre_gather] + built

    return {
        "meta": meta,
        "vars": {k: v.get("default") for k, v in var_specs.items()},
        "steps": built,
    }



#: Where a model's private feature tables go. Skipped by the finish screen so
#: tables nobody asked for do not crowd the list of results.
from ..helpers.model_spec import MODEL_WORK_DIR  # noqa: E402  (shared with the scorer)


def _wire_controls(steps, model_plans, *, source, overrides, var_values):
    """
    Carry a model's control columns from the spreadsheet to the scoring step.

    A ridge fitted with ``age`` and ``gender`` held constant needs those two
    columns to score anything, and no feature step produces them: they are
    the spreadsheet's own. The statistics already have a step for exactly
    this -- the metadata gather, which wrangles the spreadsheet without its
    text so its ``text_id`` composes like the features' -- so when any model
    needs controls that step is added, placed ahead of the scoring step,
    told to carry the control columns, and its table handed to the scorer.

    A real run refused at the very last step over this, after every feature
    had been measured for the model correctly.
    """
    needed = sorted({str(c) for p in model_plans if p is not None
                     for c in getattr(p, "controls", ())})
    scoring = [r for r in steps if getattr(r, "library", None)
               and "models" in r.library.values()]
    if not needed or not scoring:
        return steps
    models = ", ".join(sorted(p.model for p in model_plans if p is not None
                              and getattr(p, "controls", ())))
    if source != "csv":
        raise ComposeError(
            f"{models} was fitted with control column(s) "
            f"{', '.join(needed)}, which come from a spreadsheet's own "
            f"columns, and this run reads {SOURCES[source]}, which has "
            f"none. Score a spreadsheet that carries them, or use a model "
            f"fitted without controls.")
    gather = by_id("stats_gather_metadata")
    steps = [r for r in steps if r.id != gather.id]
    at = min(i for i, r in enumerate(steps) if r in scoring)
    steps.insert(at, gather)
    for r in scoring:
        overrides[r.id] = {**overrides.get(r.id, {}),
                           "metadata_csv": "{{" + gather.save_as + "}}"}
    carry = [str(c) for c in (var_values.get("stats_meta_carry") or [])]
    var_values["stats_meta_carry"] = carry + [c for c in needed
                                              if c not in carry]
    return steps


#: The folder an averaged table lands in. `by-conv-speaker` says what one row
#: of it is; a level number or an `_avg2` suffix says only how many steps ago
#: it happened, which is not a thing anyone wants to know from a filename.
def _by_dir(keys: Sequence[str]) -> str:
    safe = ["".join(c if (c.isalnum() or c in "-_") else "_" for c in str(k))
            for k in keys]
    return "by-" + "-".join(safe)


#: The artifact a pre-measure gather writes, when there is one.
_GATHERED = "gathered_texts"


def _pre_gather_step(*, text_cols, id_cols, text_mode, group_by, row_filters,
                     carry) -> dict:
    """
    The text gather, as its own visible step, so rows can be left out first.

    Normally every text analyzer gathers for itself and the first one to run
    writes the table the rest reuse. That is fine until somebody wants to
    leave rows out before anything is measured: the analyzers take no
    argument for it, and adding one to all twenty of them to carry a setting
    none of them uses is the wrong shape. So the gather comes out in front,
    where the setting belongs and where the manifest shows it happened.

    It matters most when rows are being joined. A row inside somebody else's
    joined text cannot be removed afterwards, so "before" is the only time
    this can be asked at all.
    """
    step = {
        "scope": "global",
        "call": "potato.helpers.csv_to_analysis_ready_csv",
        "save_as": _GATHERED,
        "with": {
            "csv_path": "{{var:input_csv}}",
            "out_csv": "gathered/texts.csv",
            "text_cols": list(text_cols),
            "mode": text_mode,
            "delimiter": "{{var:csv_delimiter}}",
            "row_filters": [list(f) for f in row_filters],
            "overwrite_existing": "{{var:overwrite_existing}}",
        },
    }
    if carry:
        step["with"]["carry_cols"] = list(carry)
    if group_by:
        step["with"]["group_by"] = list(group_by)
    elif id_cols:
        step["with"]["id_cols"] = list(id_cols)
    return step


def _splice_averaging(built, steps, feature_levels, *, source, text_cols,
                      id_cols, text_mode, group_by, level, picked,
                      overrides, var_specs):
    """
    Measure every row, then average the numbers -- one step per table, per level.

    The other order from the text gather, which joins the texts that belong
    together and measures the result once. Both are defensible and they give
    different numbers, so this is a thing the user says rather than something
    inferred, and it lands in the preset where anyone can read it back.

    Three things about the shape, each of which was the alternative design:

    * **One step per table per level, not one step over a folder.** A feature
      table's provenance record sits beside it, and `assemble_analysis_table`
      reads every joined table's record to find its bookkeeping columns while
      `score_model` reads it to check a saved model. A single step writing a
      folder full of tables would leave all of them unrecorded, and the
      decorator writes one record per returned file by design.
    * **Plain step dicts, never Recipes.** Registered in the catalog they
      would appear as editable rows on the options screen, be counted by
      `feature_tables()`, and need a `save_as` known before the tables are.
      Splicing after resolution makes them invisible by arithmetic, the same
      trick `_splice_model_extractions` uses and for the same reasons.
    * **A key map, because a feature table cannot say who said it.** All it
      carries is `text_id` and numbers. So the first average reads a table of
      the spreadsheet's own grouping columns, keyed the way the text gather
      keys one; later averages group on columns their own input already
      carries.

    The averaged file keeps its table's stem and changes folder, so a model
    fitted on `by-conv/readability.csv` and one fitted on `readability.csv`
    agree about what the table is called -- which is what `score_model`
    compares records by.
    """
    levels = [tuple(str(c) for c in lv) for lv in (feature_levels or ()) if lv]
    if not levels or source != "csv":
        return built
    tables = feature_tables(steps, source, level, group_by, picked=picked)
    if not tables:
        return built
    # the measure's own name, with the joining level's filename suffix taken
    # off: `_suffixed` put it there to say what one row of the *raw* table
    # was, and the folder this one lands in says what one row of it is. left
    # on, the analyses would name a feature set `readability_by-group` inside
    # a folder called `by-conv`, which reads as two answers to one question.
    joined_at = level_by_id(source, _level_for(source, level, group_by)).id
    suffix = f"_by-{joined_at}"
    stems = {}
    for save_as, name in zip(
            [r.save_as for r, _ in tables],
            [n for n, _ in table_names(steps, source, level, group_by,
                                       picked=picked, overrides=overrides,
                                       var_specs=var_specs)], strict=True):
        stems[save_as] = name[:-len(suffix)] \
            if suffix != "_by-" and name.endswith(suffix) else name

    kept = [r for r in steps
            if not _is_pointless_merge(r, source, level, group_by)]
    # ahead of everything that reads a feature table, which is the statistics
    # and the descriptives both -- the descriptives sit in the extract stage
    # and would otherwise be asked to read a table written after them.
    reading = {"stats_assemble", "describe_features"}
    at = min([i for i, r in enumerate(kept)
              if r.stage == "analyze" or r.id in reading] or [len(kept)])
    split_col = "source_col" if text_mode == "separate" else ""

    keys_art = "row_group_keys"
    placing = ({"group_by": list(group_by)} if group_by
               else {"id_cols": list(id_cols)} if id_cols else {})
    spliced = [{
        "scope": "global",
        "call": "potato.helpers.csv_to_analysis_ready_csv",
        "save_as": keys_art,
        "with": {
            "csv_path": "{{var:input_csv}}",
            "out_csv": "gathered/row_group_keys.csv",
            "text_cols": [],
            "delimiter": "{{var:csv_delimiter}}",
            "carry_cols": sorted({c for lv in levels for c in lv}),
            "overwrite_existing": "{{var:overwrite_existing}}",
            **placing,
        },
    }]

    latest = {r.save_as: "{{" + r.save_as + "}}" for r, _ in tables}
    for depth, lv in enumerate(levels):
        for recipe, _label in tables:
            art = f"{recipe.save_as}_by_{depth + 1}"
            spliced.append({
                "scope": "global",
                "call": "potato.helpers.average_feature_table",
                "save_as": art,
                "with": {
                    "in_csv": latest[recipe.save_as],
                    "out_csv": (f"{{{{var:features_dir}}}}/{_by_dir(lv)}/"
                                f"{stems[recipe.save_as]}.csv"),
                    "group_by": list(lv),
                    "split_col": split_col,
                    "overwrite_existing": "{{var:overwrite_existing}}",
                    # only the first average needs telling which row belongs
                    # to which group; after that the grouping columns are in
                    # the table it just wrote.
                    **({"keys_csv": "{{" + keys_art + "}}"} if depth == 0
                       else {}),
                },
            })
            latest[recipe.save_as] = "{{" + art + "}}"

    # the statistics and the descriptives both read the grain the analyses
    # actually use. a report whose descriptives say "4000 rows" about a model
    # fitted on 40 conversations is worse than no descriptives.
    swap = {"{{" + r.save_as + "}}": latest[r.save_as] for r, _ in tables}
    for recipe, step in zip(kept, built, strict=True):
        if recipe.id not in reading:
            continue
        current = step.get("with", {}).get("feature_csvs")
        if isinstance(current, list):
            step["with"]["feature_csvs"] = [swap.get(t, t) for t in current]

    return built[:at] + spliced + built[at:]


def _splice_model_extractions(built, steps, model_plans, *, source, text_cols,
                              id_cols, text_mode, group_by, level,
                              overrides, var_specs):
    """
    Add a private extraction for each saved model whose settings differ.

    A model fitted on cohesion measured one way cannot be scored on cohesion
    measured another way -- the numbers move and the column names do not. So
    when the two disagree, this measures the feature a second time with the
    model's own settings, into a private path, and points the scoring step at
    that instead. When they agree, nothing is added and both share one
    extraction.

    Three things make this safe, and each was a hole in an earlier design:

    * **A clone is a plain step dict, never a Recipe.** Registered in the
      catalog it would trip the single-provider check, appear as an editable
      row on the options screen, and be counted by ``feature_tables()`` --
      which would join a model's private table into the user's own analysis
      table. Splicing after resolution makes it invisible by arithmetic
      rather than by vigilance, because every screen reads the resolved
      *Recipe* list and this is not in it.
    * **Its settings are literals from the model, never variables.**
      ``lemmatize`` is deliberately one shared variable across three steps and
      cannot hold two values; a clone carries ``lemmatize: false`` outright
      while the user's step keeps ``{{var:lemmatize}}`` reading ``true``.
      Reading a variable is how the weakest earlier design silently replayed
      with the *user's* value for anything the model happened not to state.
    * **The reuse decision may only be wrong toward extracting twice.** It
      compares declared settings and cannot see a later ``--var``, a
      hand-edited preset or library drift -- so the run-time gate in
      ``score_model`` is what actually guarantees correctness, and this is
      only an optimization on top of it.
    """

    plans = [p for p in model_plans if p is not None]
    if not plans:
        return built

    scoring = [r for r in steps if getattr(r, "library", None)
               and "models" in r.library.values()]
    if not scoring:
        return built

    by_target = {r.target: r for r in steps}
    wanted: List[str] = []
    clones: List[dict] = []

    for plan in plans:
        for table in plan.tables:
            if table.replay:
                # this one was fitted to its corpus, so the user's own step
                # (however it's set) measures something else. always applied.
                clone = _replay_step(plan, table, source=source,
                                     text_cols=text_cols, id_cols=id_cols,
                                     text_mode=text_mode, group_by=group_by,
                                     level=level)
                clones.append(clone)
                wanted.append("{{" + clone["save_as"] + "}}")
                continue
            recipe = by_target.get(table.target)
            explicit = next((s for s in built
                             if s.get("call") == (recipe.call if recipe else None)),
                            None)
            if recipe is not None and explicit is not None \
                    and _settings_agree(recipe, explicit, table,
                                        var_specs):
                # R6: the user's already measuring it exactly this way, so
                # we just reuse their table.
                wanted.append("{{" + recipe.save_as + "}}")
                continue
            if recipe is None:
                raise ComposeError(
                    f"{plan.model} was fitted on the {table.stem!r} feature "
                    f"table, which this run does not produce and which "
                    f"cannot be added automatically. Select the step that "
                    f"produces it, measured with: "
                    f"{_readable(table.instrument)}.")
            if not _replayable(recipe):
                raise ComposeError(
                    f"{plan.model} was fitted on {recipe.label!r}, which "
                    f"cannot be measured a second time inside one run. "
                    f"Extract it explicitly with these settings and score "
                    f"against that: {_readable(table.instrument)}.")
            clone = _clone_step(recipe, plan, table, source=source,
                                text_cols=text_cols, id_cols=id_cols,
                                text_mode=text_mode, group_by=group_by,
                                level=level)
            clones.append(clone)
            wanted.append("{{" + clone["save_as"] + "}}")

    if not clones and not wanted:
        return built

    # two models with one slug ("Age Blogs" and "age_blogs") would write two
    # private steps under one `save_as`, and the second would silently replace
    # the first's table. exact duplicates (the same model listed twice)
    # collapse into one; an actual name clash we refuse, by name.
    unique: Dict[str, dict] = {}
    for clone in clones:
        key = str(clone["save_as"])
        seen = unique.get(key)
        if seen is None:
            unique[key] = clone
            continue
        if seen["with"].get("out_features_csv") != clone["with"].get("out_features_csv") \
                or seen.get("for_model") != clone.get("for_model"):
            raise ComposeError(
                f"two saved models are both called {seen.get('for_model')!r} "
                f"(or spell the same name differently), so their private "
                f"feature tables would overwrite each other. Rename one under "
                f"Settings → Manage saved models.")
    clones = list(unique.values())

    # the scoring step reads exactly the model's tables and nothing else. we
    # write this straight into the rendered steps: `built` was rendered before
    # we got here, so adding to `overrides` at this point reached nothing --
    # the clone got measured and then ignored, and the run-time gate refused
    # the user's own table. so the whole thing was broken, and every test was
    # green because none of them checked the scoring step's wiring. we also
    # dedupe, since two models sharing one table name it twice.
    scoring_calls = {r.call for r in scoring}
    for step in built:
        if step.get("call") in scoring_calls:
            step["with"]["feature_csvs"] = list(dict.fromkeys(wanted))

    # splice them in before the step that eats them, after everything they need.
    at = min((i for i, s in enumerate(built)
              if s.get("call") in scoring_calls), default=len(built))
    return built[:at] + clones + built[at:]


def _replayable(recipe) -> bool:
    """Whether one step can be measured twice in a single run.

    A global text analyzer can: it reads the shared gathered table and writes
    one CSV. Everything else cannot -- an item-scoped chain would mean
    re-measuring audio behind the user's back, and a step that reads another
    step's artifact would need that step duplicated too. The one artifact a
    clone may read is an asset the model carries: the document-term matrix
    reads a frequency list, and the model has the very list it was fitted
    against, so the clone scans against that and needs no second
    frequency step.
    """
    if not (recipe.feature_table and recipe.text_input
            and recipe.scope == "global"
            and not getattr(recipe, "consumes_feature_tables", False)):
        return False
    supplied = set(TEXT_INPUT_KEYS) | set(_asset_params(recipe))
    return not any(_reads_artifact(v) for k, v in recipe.with_.items()
                   if k not in supplied)


def _reads_artifact(value) -> bool:
    """Whether a `with:` value names another step's output (`{{name}}`, not
    `{{var:name}}`)."""
    return isinstance(value, str) and bool(
        re.search(r"\{\{(?!var:)[^{}]+\}\}", value))


def _replay_step(plan, table, *, source, text_cols, id_cols, text_mode,
                 group_by, level) -> dict:
    """
    One private step that *applies* what the fit-time step fitted.

    Built from the catalog's applying recipe -- never a selectable one --
    with the fitted file written from the model at run time. The instrument
    check the ordinary clone makes has no meaning here: the fitted file is
    the instrument, and it is carried whole.
    """
    target = str((table.replay or {}).get("target") or "")
    recipe = next((r for r in RECIPES if r.target == target), None)
    if recipe is None:
        raise ComposeError(
            f"{plan.model} was fitted on the {table.stem!r} feature table, "
            f"which is reproduced by applying a saved model with {target}, "
            f"and this build of Taters has no step for that.")
    step = _bind_source(recipe, source, text_cols=text_cols, id_cols=id_cols,
                        text_mode=text_mode, group_by=group_by, level=level)
    _carry_assets(step, recipe, plan, table.replay.get("assets") or {})
    step["why"] = (f"applies the fitted "
                   f"{', '.join(sorted(table.replay.get('assets') or {}))}")
    return _finish_private(step, recipe, plan, table)


def _carry_assets(step: dict, recipe, plan, assets) -> None:
    """
    Point a private step at the model's own copy of each asset it reads.

    The step names the model file and which entries to take; the runner
    writes them beside the step's output and binds the parameter. A clone
    used to keep the recipe's value here -- the *user's* dictionaries --
    and the model's embedded lists were carried and never read, so the
    "portable model" replayed with whatever the machine happened to have.
    """
    take = {k: [e.get("name") for e in v] for k, v in assets.items() if v}
    if not take:
        return
    if not getattr(plan, "path", ""):
        raise ComposeError(
            f"{plan.model} carries word lists or fitted files its private "
            f"extraction needs, but the plan does not say which model file "
            f"to read them from.")
    for key in take:
        step["with"].pop(key, None)
    step["assets"] = {"from": str(plan.path), "take": take}


def _finish_private(step: dict, recipe, plan, table) -> dict:
    """The tail every private extraction shares: where it writes, what it
    is called, and whose it is."""
    # the digest goes in the directory so two models needing identical settings
    # share one extraction, and a re-fit under the same name can't
    # short-circuit onto the old private table. the fit-time stem goes in the
    # filename, because the stem is what fixed the predictor names.
    step["with"]["out_features_csv"] = (
        f"{{{{var:features_dir}}}}/{MODEL_WORK_DIR}/{plan.slug}/"
        f"{table.digest[:8]}/{table.stem}.csv")
    step["save_as"] = f"{recipe.save_as}__for_{plan.slug}"
    step["for_model"] = plan.model
    step["label"] = f"{recipe.label} — for {plan.model}"
    return step


def _clone_step(recipe, plan, table, *, source, text_cols, id_cols, text_mode,
                group_by, level) -> dict:
    """One private extraction, at the model's settings."""
    step = _bind_source(recipe, source, text_cols=text_cols, id_cols=id_cols,
                        text_mode=text_mode, group_by=group_by, level=level)
    declared = _instrument_keys(recipe)
    missing = [k for k in declared if k not in table.instrument]
    if missing:
        raise ComposeError(
            f"{plan.model} does not record {', '.join(sorted(missing))} for "
            f"the {table.stem!r} table, so it cannot be measured again "
            f"exactly as it was fitted. Re-fit the model with a build of "
            f"Taters that records every setting.")
    for key in declared:
        step["with"][key] = table.instrument[key]
    _carry_assets(step, recipe, plan, table.assets or {})
    step["why"] = _readable(table.instrument)
    return _finish_private(step, recipe, plan, table)


def _instrument_defaults(recipe) -> Dict[str, Any]:
    """Each instrument setting's value when nobody states it.

    A step's ``with:`` block states only some of them; the rest are function
    defaults, and a comparison that read an unstated key as "no value" would
    decide the settings differ every time and extract twice on every run.
    Safe, and wasteful enough to defeat the point of comparing at all.
    """
    import inspect

    from ..pipelines.run_pipeline import resolve_call

    from ..Taters import Taters

    fn = resolve_call(recipe.target, Taters())
    params = inspect.signature(fn).parameters
    return {k: params[k].default for k in _instrument_keys(recipe)
            if k in params}


def _declared(recipe) -> Optional[dict]:
    """The analyzer's own provenance declaration, or None when it has none."""
    from ..pipelines.run_pipeline import resolve_call

    from ..Taters import Taters

    fn = resolve_call(recipe.target, Taters())
    return getattr(fn, "__provenance__", None)


def _asset_params(recipe) -> Dict[str, Optional[str]]:
    """``{parameter: library kind}`` for the word lists and fitted files the
    recipe's analyzer reads, as it declares them."""
    declared = _declared(recipe)
    return dict((declared or {}).get("assets") or {})


def _instrument_keys(recipe) -> List[str]:
    """The settings of this recipe that decide the numbers.

    Read from the analyzer's own provenance declaration -- the residue after
    input, grain and output paths -- so the clone carries exactly what the
    record carries and the two cannot drift apart.
    """
    from ..pipelines.run_pipeline import resolve_call

    from ..Taters import Taters

    fn = resolve_call(recipe.target, Taters())
    declared = getattr(fn, "__provenance__", None)
    if not declared:
        return []
    import inspect

    from ..helpers.provenance import PLUMBING

    skip = (set(declared["binding"]) | set(declared["grain"])
            | set(declared["outputs"]) | set(declared["assets"])
            | set(PLUMBING))
    return [k for k in inspect.signature(fn).parameters if k not in skip]


def _settings_agree(recipe, step: dict, table, var_specs=None) -> bool:
    """Whether the user's own step already measures it the model's way.

    Compares the *declared* settings, resolving `{{var:x}}` against the
    value that will actually be used -- reusing because a catalog default
    matched, when the user had changed it, is the one way this decision can
    be wrong in the dangerous direction.
    """
    # `var_specs` is our one authoritative map of what a variable will actually
    # be: compose folds every answer into `var_specs[key]["default"]` before we
    # get here, so an answered variable already reads as its answer and an
    # unanswered one reads as its default. checking `var_values` separately
    # would be reading the same fact twice -- and reading ONLY the catalog
    # default is the subtle way to reuse the user's table when they'd actually
    # changed the setting.
    values = {k: v.get("default") for k, v in (var_specs or {}).items()}
    defaults = _instrument_defaults(recipe)
    for key in _instrument_keys(recipe):
        if key not in table.instrument:
            return False
        # not in the `with:` block means the function default applies -- which
        # is exactly what the model recorded on its side, since the record is
        # taken after defaults get bound.
        mine = step["with"][key] if key in step["with"] \
            else defaults.get(key)
        if isinstance(mine, str) and mine.startswith("{{var:"):
            name = mine[len("{{var:"):-2]
            if name not in values:
                return False        # can't resolve it: extract twice, to be safe
            mine = values[name]
        if mine != table.instrument[key]:
            return False
    # now the word lists too, by content. two dictionary steps with identical
    # settings and different dictionaries make different columns, and when we
    # only compared settings we reused the user's table for a model fitted on
    # other lists -- caught at run time, but as a dead end, since the user's
    # only way past was to change their own dictionaries.
    from ..helpers.provenance import asset_manifest

    for key, kind in _asset_params(recipe).items():
        want = (table.assets or {}).get(key)
        if want is None:
            return False        # the model didn't record it: extract twice
        # a list parameter comes as a list of templates
        # (`["{{var:dictionaries_path}}"]`), so we resolve each element.
        mine, known = _resolve_vars(step["with"].get(key), values)
        if not known:
            return False        # can't resolve, or it's another step's output
        have = asset_manifest(kind, mine, embed=False)
        if {(e.get("name"), e.get("sha256")) for e in want} != \
                {(e.get("name"), e.get("sha256")) for e in have}:
            return False
    return True


def _resolve_vars(value, values: Dict[str, Any]):
    """
    A `with:` value with every `{{var:x}}` replaced by its effective value.

    Returns ``(resolved, known)``; ``known`` is False when a variable has no
    value here or the value names another step's output, either of which
    means the comparison cannot be made and the safe answer is "differs".
    """
    if isinstance(value, list):
        out = []
        for item in value:
            item, known = _resolve_vars(item, values)
            if not known:
                return None, False
            out.append(item)
        return out, True
    if isinstance(value, str) and value.startswith("{{var:") \
            and value.endswith("}}"):
        name = value[len("{{var:"):-2]
        if name not in values:
            return None, False
        return values[name], True
    if _reads_artifact(value):
        return None, False
    return value, True


def _readable(instrument) -> str:
    """Settings as a person would say them."""
    return ", ".join(f"{k}={v!r}" for k, v in sorted(instrument.items()))


def _inputs_meta(source: str, file_type: str, input_path: Optional[str]) -> Dict[str, str]:
    """What `--describe-preset` shows under "inputs"."""
    if source == "txt_dir":
        return {"text files": f"{input_path or '<a folder>'}/*.txt"}
    if source == "csv":
        return {"spreadsheet": input_path or "<a .csv file>"}
    return {"file type": f"{file_type} files"}


def _cli_example(preset_id: str, source: str, file_type: str, root_dir: Optional[str]) -> str:
    """
    A copy-pasteable re-run command.

    Text presets are GLOBAL-only, so `run_preset` skips input discovery and
    `--root_dir` is not merely unnecessary but misleading -- the input path
    lives in a variable instead.
    """
    base = f"python -m taters.pipelines.run_pipeline --preset {preset_id}"
    if source == "media":
        return (
            f"python -m taters.pipelines.run_pipeline "
            f"--root_dir {root_dir or '<your-folder>'} --file_type {file_type} "
            f"--preset {preset_id} --workers 4"
        )
    return base


def _tags_for(recipe: Recipe) -> Set[str]:
    """
    Tags for the preset index: what the step declares, plus what its module
    says about it.

    This used to name `transcribe` and `diarize` outright -- the only place in
    the composer that knew a recipe by id, and the one seam a new step could
    not reach without editing this function.
    """
    tags = set(recipe.tags)
    if ".audio." in recipe.call:
        tags.add("audio")
    if ".text." in recipe.call:
        tags.add("text")
    if "embeddings" in recipe.id:
        tags.add("embeddings")
    return tags
