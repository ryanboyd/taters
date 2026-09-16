"""
Two feature tables must never agree on a column name.

Feature tables get joined side by side into one analysis table. `stats.assemble`
already copes when two of them share a column name -- it renames the column in
every table that has it -- but that fix is *reactive*, and reactive renaming has
a property that is poison for research: **the name depends on what else ran.**
A column is `Topic_1` in a pipeline with one topic model and
`lda_topics__Topic_1` in a pipeline with two. Same instrument, same corpus, two
different names, so a script written against one study fails against the next
and two results tables cannot be compared without knowing what was in each run.

So the names Taters chooses have to be disjoint up front. Each feature module
declares what it writes; this file fails the build if any two declarations could
produce the same name, and if any feature-producing module forgot to declare at
all.

The list of modules is read out of the recipe catalog rather than kept here, on
purpose: a list kept here would be one more thing to remember, and the whole
point is that forgetting is caught.
"""

from __future__ import annotations

import importlib

import pytest

from taters.helpers.feature_columns import ColumnSpec, overlaps, pattern_regex
from taters.ui import recipes as rec


def _feature_modules():
    """Every module behind a recipe that writes a joinable feature table."""
    found = {}
    for recipe in rec.RECIPES:
        if not recipe.feature_table:
            continue
        module_path = str(recipe.target).split(":", 1)[0]
        found.setdefault(module_path, []).append(recipe.id)
    return found


def _declared():
    """(module path, spec) for every feature module that declares its columns."""
    specs = []
    for module_path in sorted(_feature_modules()):
        try:
            module = importlib.import_module(module_path)
        except Exception:                       # pragma: no cover - optional extras
            continue
        spec = getattr(module, "FEATURE_COLUMNS", None)
        if spec is not None:
            specs.append((module_path, spec))
    return specs


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def test_no_two_measures_can_write_the_same_column():
    """
    The whole point of the file. If this fails, two measures would land on the
    same column name in an analysis table that has both -- and `assemble` would
    paper over it by renaming, which is how a column's name starts depending on
    what else was in the pipeline.
    """
    specs = [spec for _path, spec in _declared()]
    clashes = overlaps(specs)
    assert not clashes, "\n".join(
        f"{a} and {b} both write: {', '.join(names)}" for a, b, names in clashes)


def test_every_feature_table_module_declares_its_columns():
    """
    Read from the recipe catalog rather than a list kept here, so a new
    extractor cannot quietly skip the check by not being added to something.
    """
    undeclared = []
    for module_path, recipe_ids in sorted(_feature_modules().items()):
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue                            # an optional extra is not installed
        if getattr(module, "FEATURE_COLUMNS", None) is None:
            undeclared.append(f"{module_path} (recipes: {', '.join(recipe_ids)})")
    assert not undeclared, (
        "these write feature tables but do not declare their columns; add a "
        "FEATURE_COLUMNS (see helpers/feature_columns.py):\n  "
        + "\n  ".join(undeclared))


def test_a_declaration_matches_what_the_module_actually_writes():
    """
    A declaration that has drifted from the code is worse than none, because it
    reads as a guarantee. These four write fixed names and are cheap to run, so
    the claim is checked against the header rather than trusted.
    """
    from taters.text.analyze_readability import FEATURE_COLUMNS as readability
    from taters.text.analyze_readability import METRICS as readability_metrics
    from taters.text.analyze_sentiment_vader import FEATURE_COLUMNS as vader
    from taters.text.analyze_sentiment_vader import METRICS as vader_metrics
    from taters.text.analyze_word_count import FEATURE_COLUMNS as counts
    from taters.text.analyze_word_count import WORD_COUNT_COLUMN

    assert readability.names == tuple(readability_metrics)
    assert vader.names == tuple(vader_metrics)
    assert counts.names == (WORD_COUNT_COLUMN,)


def test_the_cohesion_declaration_comes_from_the_header_writer():
    """Cohesion builds its header with `header_columns`, and declares from the
    same call -- so the two cannot drift even as TAACO measures come and go."""
    from taters.text.analyze_cohesion import FEATURE_COLUMNS, header_columns

    assert FEATURE_COLUMNS.names == tuple(header_columns(connective_names=(),
                                                         semantic=True))


def test_importing_a_feature_module_reads_no_files():
    """
    Cohesion's connective lists live in files, and declaring from them the easy
    way (`header_columns()` with its default) would make importing the module do
    disk I/O -- on every import, in every test, for a constant. The declaration
    passes an empty list for exactly that reason; this is what notices if
    somebody removes it.
    """
    import taters.text.analyze_cohesion as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "header_columns(connective_names=(), semantic=True)" in source, (
        "the cohesion declaration should pass an explicit empty connective list "
        "so that importing the module does not read the connective files")


def test_a_declared_pattern_matches_the_names_it_claims_to_describe():
    """
    Found in review. Parts of speech declared `pos_{n}` -- a *number* -- while
    writing `pos_NN` and `pos_DT_NN`. A pattern that matches none of its own
    columns is worse than no pattern: `overlaps()` cannot see a future measure
    colliding with it, which is the single job the registry has.

    Checked against a name each module really produces, rather than against
    the pattern's own shape, which would just restate it.
    """
    from taters.helpers.feature_columns import pattern_regex, registry

    examples = {
        "pos_{*}": "pos_NN",
        "Theme_{n}": "Theme_3",
        "Topic_{n}": "Topic_3",
        "Factor_{n}": "Factor_3",
        "wv_{n}": "wv_3",
        "e{n}": "e3",
        "e_{n}": "e_3",
        "msttr_{n}": "msttr_50",
        "mattr_{n}": "mattr_50",
        "mtld_{*}": "mtld_0_72",
        "hdd_{n}": "hdd_42",
        "vocd_{n}": "vocd_50",
    }
    for module, spec in registry().items():
        for pattern in spec.patterns:
            assert pattern in examples, (
                f"{module} declares {pattern!r}; add a real column name for it "
                "here so the pattern is checked against something")
            assert pattern_regex(pattern).fullmatch(examples[pattern]), (
                f"{module}'s {pattern!r} does not match {examples[pattern]!r}, "
                "a name it actually writes")


def test_everything_that_strips_the_collision_prefix_uses_the_same_separator():
    """
    `assemble` renames a collided column to `<table><SEP><column>`, and the
    word clouds strip that prefix back off before labeling anything. The two
    have to agree, and when the separator changed from `.` to `__` the clouds
    silently stopped stripping: every cloud drawn from a collided column would
    have read `readability__word_count` instead of `word_count`, with nothing
    raising anywhere.

    So the strippers import the separator rather than spelling it out, and this
    is what says so.
    """
    from taters.figures.wordclouds import _label
    from taters.helpers.feature_columns import SEPARATOR

    assert _label(f"readability{SEPARATOR}word_count", "readability") == "word_count"
    assert _label("flesch_reading_ease", "readability") == "flesch_reading_ease"
    assert _label(f"cohesion{SEPARATOR}word_count", "readability") == \
        f"cohesion{SEPARATOR}word_count", "only its own prefix comes off"


# ---------------------------------------------------------------------------
# What a reduction of these columns is called
# ---------------------------------------------------------------------------

def test_the_registry_covers_every_module_the_catalog_knows_about():
    """
    `DECLARING_MODULES` is spelled out so the registry works at runtime without
    `helpers` having to import the wizard's catalog -- which would point the
    layering backwards. The cost is a list that could go stale, so this is what
    stops it: the catalog is where feature steps get registered anyway.
    """
    from taters.helpers.feature_columns import DECLARING_MODULES

    missing = sorted(set(_feature_modules()) - set(DECLARING_MODULES))
    assert not missing, (
        "these declare feature columns but are not in DECLARING_MODULES, so "
        "nothing at runtime can find them:\n  " + "\n  ".join(missing))


def test_reducing_a_topic_model_gives_supertopics():
    """`Component_3` is one more thing to look up. A PCA over a topic model's
    topics produces something worth naming, so it gets named."""
    from taters.helpers.feature_columns import reduced_name

    assert reduced_name(["Topic_1", "Topic_2"]) == "Supertopic"
    assert reduced_name(["Theme_1", "Theme_2"]) == "Supertopic"
    assert reduced_name(["Factor_1", "Factor_2"]) == "Supertopic"


def test_reducing_anything_else_still_gives_components():
    from taters.helpers.feature_columns import reduced_name

    assert reduced_name(["posemo", "negemo"]) == "Component"
    assert reduced_name(["flesch_reading_ease", "smog_index"]) == "Component"
    assert reduced_name([]) == "Component"


def test_a_mixed_feature_set_is_not_a_topic_model():
    """
    The statistics stage builds a combined "all" set out of every feature
    table, so a run with topics *and* readability reduces a mixture -- and a
    component of that mixture is genuinely not a supertopic. Requiring one
    matching column rather than all of them named it one anyway.
    """
    from taters.helpers.feature_columns import reduced_name

    assert reduced_name(["Topic_1", "flesch_reading_ease"]) == "Component"
    assert reduced_name(["Topic_1", "Theme_1"]) == "Component", (
        "two different topic models are still not one topic model")


def test_the_collision_prefix_does_not_hide_a_topic_model():
    """If one of the set's columns collided and got renamed, the set is still
    a topic model."""
    from taters.helpers.feature_columns import SEPARATOR, reduced_name

    columns = [f"lda_topics{SEPARATOR}Topic_1", f"lda_topics{SEPARATOR}Topic_2"]
    assert reduced_name(columns, set_name="lda_topics") == "Supertopic"


def test_the_statistics_stage_names_them_by_what_they_reduce():
    """The end of the wire: `component_names` takes the noun, and the analysis
    stage asks the registry for it."""
    from taters.stats.pca import component_names

    assert component_names(2) == ["Component_1", "Component_2"]
    assert component_names(2, "Supertopic") == ["Supertopic_1", "Supertopic_2"]


def test_a_real_reduction_over_topics_writes_supertopic_columns(tmp_path):
    """
    End to end, because everything above this can pass while the statistics
    stage carries on calling `component_names` without asking anybody. Reduce
    a table whose features are topics, and the results have to say
    `Supertopic_1`.
    """
    import csv as _csv
    import random

    from taters.stats.correlations import analyze_correlations

    rng = random.Random(0)
    table = tmp_path / "analysis_table.csv"
    topics = [f"Topic_{i + 1}" for i in range(5)]
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["text_id", "outcome", *topics])
        for row in range(40):
            base = rng.random()
            writer.writerow([f"d{row}", row % 5,
                             *[round(base + rng.random() * 0.3, 4) for _ in topics]])

    # name the set, the way the assemble step's sidecar does
    import json
    (tmp_path / "analysis_table_sets.json").write_text(
        json.dumps({"sets": {"lda_topics": topics}}), encoding="utf-8")

    analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                         feature_sets="per_table",
                         out_dir=tmp_path / "stats", pca="all",
                         pca_components=2, verbose=False)

    loadings = sorted((tmp_path / "stats").glob("*pca_loadings*.csv"))
    assert loadings, "the reduction wrote no loadings at all"
    header = _csv.DictReader(
        open(loadings[0], encoding="utf-8-sig")).fieldnames or []

    # named for what it is, and prefixed by where it came from. the prefix is
    # unconditional even though there is only one set here and nothing to tell
    # it apart from -- the name is the only thing that travels with a column
    # into a paper or somebody else's table.
    assert "lda_topics_Supertopic_1" in header, header
    assert not any("Component" in h for h in header), header


def test_only_the_topic_models_claim_anything_other_than_components():
    """
    A guard on the whole idea, not just on today's three. Reducing readability
    indices, or dictionary categories, or embedding dimensions gives components
    -- because that is what they are. If a fourth measure ever starts calling
    its reductions something else, that should be a deliberate edit here rather
    than a surprise in somebody's results table.
    """
    from taters.helpers.feature_columns import registry

    claimed = {spec.label: spec.reduces_to for spec in registry().values()
               if spec.reduces_to != "Component"}
    assert claimed == {"Topic model (LDA)": "Supertopic",
                       "Topic model (MEM)": "Supertopic",
                       "Topic model (NMF)": "Supertopic"}, claimed


def test_a_real_reduction_over_readability_still_writes_components(tmp_path):
    """The other half of the previous test, end to end: eighteen readability
    indices reduce to components, and nothing calls them supertopics."""
    import csv as _csv
    import json
    import random

    from taters.stats.correlations import analyze_correlations
    from taters.text.analyze_readability import METRICS

    rng = random.Random(0)
    metrics = list(METRICS[:6])
    table = tmp_path / "analysis_table.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["text_id", "outcome", *metrics])
        for row in range(40):
            base = rng.random()
            writer.writerow([f"d{row}", row % 5,
                             *[round(base + rng.random() * 0.3, 4) for _ in metrics]])
    (tmp_path / "analysis_table_sets.json").write_text(
        json.dumps({"sets": {"readability": metrics}}), encoding="utf-8")

    analyze_correlations(table_csv=table, outcome_cols=["outcome"],
                         feature_sets="per_table", out_dir=tmp_path / "stats",
                         pca="all", pca_components=2, verbose=False)

    header = _csv.DictReader(open(
        sorted((tmp_path / "stats").glob("*pca_loadings*.csv"))[0],
        encoding="utf-8-sig")).fieldnames or []
    assert "readability_Component_1" in header, header
    assert not any("Supertopic" in h for h in header), header


# ---------------------------------------------------------------------------
# The declaration format itself
# ---------------------------------------------------------------------------

def test_a_declaration_that_names_nothing_is_refused():
    """An empty spec would pass the disjointness check by saying nothing."""
    with pytest.raises(ValueError, match="no columns at all"):
        ColumnSpec(label="Ghost")


def test_a_pattern_needs_exactly_one_varying_part():
    with pytest.raises(ValueError, match="exactly one"):
        ColumnSpec(label="Bad", patterns=("theme_{n}_{n}",))
    with pytest.raises(ValueError, match="exactly one"):
        ColumnSpec(label="Bad", patterns=("theme",))


def test_a_pattern_catches_a_literal_that_would_fall_inside_it():
    """`Theme_{n}` and a literal `Theme_3` are the same column."""
    clashes = overlaps([ColumnSpec(label="A", patterns=("Theme_{n}",)),
                        ColumnSpec(label="B", names=("Theme_3",))])
    assert clashes and clashes[0][2] == ["Theme_3"]


def test_two_patterns_with_the_same_shape_collide():
    clashes = overlaps([ColumnSpec(label="LDA", patterns=("Topic_{n}",)),
                        ColumnSpec(label="NMF", patterns=("Topic_{n}",))])
    assert clashes, "two engines numbering their topics the same way must clash"


def test_patterns_that_differ_only_after_the_number_are_told_apart():
    """`msttr_{n}` and `mattr_{n}` are one letter apart and must not collide."""
    assert not overlaps([ColumnSpec(label="A", patterns=("msttr_{n}",)),
                         ColumnSpec(label="B", patterns=("mattr_{n}",))])


def test_a_multi_digit_index_is_still_caught():
    """`Theme_1` and `Theme_10` are both real columns; a check that only ever
    rendered a single digit would miss half of them."""
    rx = pattern_regex("Theme_{n}")
    assert rx.fullmatch("Theme_10") and rx.fullmatch("Theme_1")


def test_dynamic_names_do_not_excuse_the_declared_ones():
    """A module can have both. Saying 'some of my columns come from the user'
    must not switch off checking for the ones that do not."""
    clashes = overlaps([
        ColumnSpec(label="A", names=("lemma_ttr",), dynamic="plus user lists"),
        ColumnSpec(label="B", names=("lemma_ttr",)),
    ])
    assert clashes and clashes[0][2] == ["lemma_ttr"]
