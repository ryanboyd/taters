"""
Tests for the recipe catalog — the hand-declared wiring behind the wizard.

The catalog is the one part of the wizard that is not derived from the code, so
it is the one part that can drift away from it. These tests pin it down: every
`call` must resolve, every parameter named in a `with_` block or a `hidden`
list must exist on the real function, and the capability graph must actually be
satisfiable.

Rename a parameter in `src/` and this file fails immediately, rather than a
user finding out mid-run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from preset_checks import templates_in, underlying

from taters.ui import recipes as _r
from taters.ui.introspect import describe, load_target
from taters.ui.recipes import (
    BASE_VARS,
    CAPABILITIES,
    RECIPES,
    SOURCES,
    TEXT_INPUT_KEYS,
    by_id,
    providers_of,
    text_binding,
    user_facing,
)


def pytest_generate_tests(metafunc):
    """One test per recipe, so a failure names the recipe that broke."""
    if "recipe" in metafunc.fixturenames:
        metafunc.parametrize("recipe", RECIPES, ids=[r.id for r in RECIPES])


# --- catalog-wide -----------------------------------------------------------

def test_ids_are_unique():
    ids = [r.id for r in RECIPES]
    assert len(ids) == len(set(ids))


def test_something_is_offered_to_the_user():
    assert user_facing(), "the feature checklist would be empty"


def test_every_feature_step_sits_under_exactly_one_heading():
    """
    The promise the headings make. A step in no category vanishes from the
    checklist entirely -- `categories_for` builds the screen from the table,
    not from the catalog -- and a step in two would be offered twice and
    ticked by two different headings.

    So a new extractor that names no home fails here rather than silently
    going missing from the one screen that offers it.
    """
    from taters.ui.recipes import FEATURE_CATEGORIES, SOURCES

    placed = [m for c in FEATURE_CATEGORIES for m in c.members]
    assert len(placed) == len(set(placed)), \
        f"listed under two headings: {sorted({m for m in placed if placed.count(m) > 1})}"

    every = {r.id for source in SOURCES for r in user_facing(source)}
    assert set(placed) == every, (
        f"no heading offers: {sorted(every - set(placed))}; "
        f"headings name unknown or non-offered steps: {sorted(set(placed) - every)}")


def test_every_heading_says_what_is_under_it():
    """The help text is what someone reads before opening a branch; a heading
    without one is a bare word on a screen built to stop being cryptic."""
    from taters.ui.recipes import FEATURE_CATEGORIES

    for category in FEATURE_CATEGORIES:
        assert category.label and category.help, category.id
        assert category.members, f"{category.id} is an empty heading"


def test_a_heading_is_dropped_where_it_has_nothing_to_offer():
    """The two audio headings are simply absent for a folder of essays --
    shown empty, they would advertise things that source can never do."""
    from taters.ui.recipes import categories_for

    on_text = {c.id for c, _members in categories_for("csv")}
    on_media = {c.id for c, _members in categories_for("media")}
    assert {"transcription", "voice"} <= on_media
    assert not ({"transcription", "voice"} & on_text)
    assert on_text < on_media


def test_every_requirement_has_a_producer():
    """A capability nothing produces is a step that can never run."""
    produced = {c for r in RECIPES for c in r.produces}
    required = {c for r in RECIPES for c in r.requires}
    assert required <= produced, f"unsatisfiable: {sorted(required - produced)}"


def test_every_capability_is_described():
    """`CAPABILITIES` supplies the wording for the wizard's prompts."""
    used = {c for r in RECIPES for c in (r.requires | r.produces)}
    assert used <= set(CAPABILITIES), f"undescribed: {sorted(used - set(CAPABILITIES))}"


def test_auto_with_targets_exist():
    for recipe in RECIPES:
        for follow_on in recipe.auto_with:
            by_id(follow_on)      # raises with the valid ids if it's a typo


def test_transcripts_have_exactly_two_producers():
    """
    The wizard's one real branch. If this ever changes, `resolve_providers`
    needs to know — a third option is fine, but silently going down to one
    would mean the "how should we transcribe?" question stops being asked.
    """
    assert [r.id for r in providers_of("transcript_csv")] == ["transcribe", "diarize"]


def test_by_id_names_the_alternatives_when_it_fails():
    """The error has to be actionable; the bad id alone is not enough to fix it."""
    with pytest.raises(KeyError, match="transcribe"):
        by_id("transcirbe")


# --- per recipe -------------------------------------------------------------

def test_scope_is_valid(recipe):
    assert recipe.scope in {"item", "global"}


def test_call_resolves_through_the_facade(recipe):
    """`potato.audio.foo` must be a real method on a real Taters instance."""
    assert callable(underlying(recipe.call))


def test_target_points_at_the_same_function_as_the_call(recipe):
    """
    A recipe names its function twice: once as a `call` for the runner, once as
    a `target` the wizard imports to read options from. They must agree, or the
    options screen describes a different function than the one that runs.
    """
    try:
        target = load_target(recipe.target)
    except ImportError as exc:
        pytest.skip(f"{recipe.id} needs an optional dependency: {exc}")
    assert target is underlying(recipe.call)


def test_template_parameters_exist_on_the_function(recipe):
    """Catches a renamed parameter before it reaches a preset."""
    try:
        spec = describe(load_target(recipe.target))
    except ImportError as exc:
        pytest.skip(f"{recipe.id} needs an optional dependency: {exc}")
    unknown = set(recipe.with_) - set(spec.names)
    assert not unknown, f"{recipe.id} passes parameters that do not exist: {sorted(unknown)}"


def test_hidden_parameters_exist_on_the_function(recipe):
    try:
        spec = describe(load_target(recipe.target))
    except ImportError as exc:
        pytest.skip(f"{recipe.id} needs an optional dependency: {exc}")
    unknown = set(recipe.hidden) - set(spec.names)
    assert not unknown, f"{recipe.id} hides options that do not exist: {sorted(unknown)}"


def test_required_parameters_are_supplied(recipe):
    """
    A step whose `with:` omits a required argument fails at call time, deep
    inside a thread pool, with a traceback nobody wants to read.
    """
    try:
        spec = describe(load_target(recipe.target))
    except ImportError as exc:
        pytest.skip(f"{recipe.id} needs an optional dependency: {exc}")
    missing = [p.name for p in spec.params if p.required and p.name not in recipe.with_]
    assert not missing, f"{recipe.id} never supplies required {missing}"


def test_var_templates_are_declared_somewhere(recipe):
    """
    Every `{{var:x}}` a recipe uses must be contributed by some recipe or by
    `BASE_VARS`, or the composed preset references a variable it never defines.
    """
    declared = set(BASE_VARS)
    for other in RECIPES:
        declared |= set(other.vars)

    # templates_in() walks nested lists/dicts and copes with templates buried in
    # a longer string, e.g. "{{var:features_dir}}/acoustics"
    used = {expr.split(":", 1)[1]
            for expr in templates_in(recipe.with_)
            if expr.startswith("var:")}
    assert used <= declared, f"{recipe.id} uses undeclared vars: {sorted(used - declared)}"


def test_user_facing_recipes_explain_themselves(recipe):
    """The checklist is the whole interface; a blank line there is useless."""
    if not recipe.user_facing:
        return
    assert recipe.label and recipe.help
    assert not recipe.label.endswith("."), "labels read better without a full stop"


def test_a_step_that_can_use_the_gpu_is_told_where_to_run(recipe):
    """
    Declaring `gpu_use` and then not forwarding `device` means the step runs
    wherever its own default says -- "auto", i.e. CUDA if there is any -- no
    matter what the person set. Found in review on `topic_model_mem`, which
    started tokenizing when it took over building its own matrix and had its
    `gpu_use` corrected without its `with_` block catching up.
    """
    if recipe.resolved_gpu_use == "cpu":
        return
    try:
        names = set(describe(load_target(recipe.target)).names)
    except ImportError:
        pytest.skip(f"{recipe.id} needs an optional dependency")
    if "device" not in names:
        return
    assert "device" in recipe.with_, (
        f"{recipe.id} declares gpu_use={recipe.resolved_gpu_use!r} and takes "
        "`device`, but never passes it")


def test_each_topic_model_writes_into_a_folder_of_its_own():
    """
    Every topic model builds its own matrix into `<its output folder>/matrix`.
    While they all wrote straight into `features/`, that resolved to one
    `features/matrix` for all of them -- so a run with two topic models had
    them building over each other's matrix, which is the exact clobbering the
    per-model matrix was introduced to stop.

    The unit test for that passed throughout, because it gave each model its
    own output folder by hand. The catalog did not.
    """
    folders = {}
    for rid in ("topic_model_mem", "topic_model_lda", "topic_model_nmf",
                ):
        recipe = by_id(rid)
        out = recipe.with_.get("out_features_csv") or recipe.with_["out_csv"]
        parent = str(out).rsplit("/", 1)[0]
        assert parent != "{{var:features_dir}}", (
            f"{rid} writes straight into the features folder, so its matrix "
            "would land in the one every other topic model uses")
        clash = folders.setdefault(parent, rid)
        assert clash == rid, f"{rid} and {clash} share the folder {parent}"


def test_every_file_a_topic_model_writes_lands_in_that_same_folder():
    """A model whose loadings went one place and whose scores went another
    would leave somebody hunting; the word-cloud step reads the loadings by
    path, so it has to agree too."""
    for engine in ("mem", "lda", "nmf"):
        home = f"{{{{var:features_dir}}}}/topic_model_{engine}/"
        # the apply too, not just the fit. it was checking only the fits, and
        # MEM's apply was quietly writing to `features/topic_model_mem_applied
        # .csv` -- loose in the features folder, outside the model's own.
        for rid in (f"topic_model_{engine}", f"topic_model_{engine}_apply"):
            recipe = by_id(rid)
            for key, value in recipe.with_.items():
                if key.startswith("out_") and isinstance(value, str):
                    assert value.startswith(home), (rid, key, value)
        clouds = by_id(f"topic_model_{engine}_wordclouds")
        assert str(clouds.with_["loadings_csv"]).startswith(home), engine


def test_the_declared_stop_lists_actually_become_paths():
    """
    The declaration is two fields on a recipe; the *effect* comes from
    `_library_defaults`, which turns them into an override before the preset
    is composed. That is three hops, and the catalog test above only checks the
    first one -- a step could declare the picker and still apply nothing.

    So this asks the question the user asked: does the frequency list this step
    builds actually get the shipped stop words?
    """
    from taters.ui.wizard import _library_defaults

    for rid in ("ngram_frequencies", "topic_model_mem", "topic_model_lda",
                "topic_model_nmf"):
        overrides: dict = {}
        _library_defaults([by_id(rid)], overrides)
        got = [Path(p).name for p in
               overrides.get(rid, {}).get("stoplist_paths", [])]
        assert "stopwords-en.txt" in got, (
            f"{rid} would build its vocabulary with no stop list, so every "
            f"topic comes out as 'the, and, of'. Got: {got}")


def test_the_topic_models_do_not_share_a_vocabulary_setting():
    """
    Each topic model builds its own matrix, precisely because they want
    different ones -- LDA needs counts, NMF wants tf-idf, and one study can
    reasonably want LDA over lemmatized unigrams and NMF over raw bigrams.

    The settings that decide what the vocabulary *is* were shared back when
    they all scanned one matrix built by somebody else, where a disagreement
    would have been a lie. That reason is gone, and sharing now means changing
    `lemmatize` for one model silently changes it for the others.
    """
    models = ["topic_model_mem", "topic_model_lda", "topic_model_nmf",
              ]
    vocabulary = ("lemmatize", "pos_tagged", "keep_punctuation", "ngram_n")

    seen: dict = {}
    for rid in models:
        recipe = by_id(rid)
        for setting in vocabulary:
            var = recipe.with_.get(setting)
            assert isinstance(var, str) and var.startswith("{{var:"), (
                f"{rid} does not offer {setting} as a setting at all")
            shared_with = seen.setdefault(var, rid)
            assert shared_with == rid, (
                f"{rid} and {shared_with} both read {var} for {setting}; "
                "changing it on one screen would change the other's model")

    # and the settings that shape the matrix itself. these are the
    # document-term-matrix step's variables, which MEM read back when it
    # scanned the matrix that step built. A topic model that took them again
    # would not only follow another step's screen -- it would override its own
    # defaults, which differ on purpose: a topic model ranks its vocabulary by
    # how many documents use a term, a feature table by raw count.
    matrix_shape = ("weighting", "vocab_rank_by", "vocab_rule", "vocab_top_n",
                    "vocab_min_freq", "vocab_min_obs_pct")
    from taters.ui.recipes import _MATRIX_SHAPE_WITH

    for rid in models:
        recipe = by_id(rid)
        for setting in matrix_shape:
            wired = recipe.with_.get(setting)
            assert wired != _MATRIX_SHAPE_WITH.get(setting), (
                f"{rid} reads the document-term-matrix step's {setting!r}; it "
                "builds its own matrix, so that setting is not shared any more "
                "-- and taking it would override this step's own default")


def test_the_tokenizer_settings_are_still_shared_on_purpose():
    """
    The counterpart, so the split above reads as a decision rather than as
    something half-done. Which toolkit splits and tags the text is a fact
    about the corpus and the machine -- and Stanza's model download is shared
    whatever any one step asked for.
    """
    for rid in ("topic_model_mem", "topic_model_lda", "topic_model_nmf"):
        recipe = by_id(rid)
        assert recipe.with_["engine"] == "{{var:engine}}", rid
        assert recipe.with_["tokenizer"] == "{{var:tokenizer}}", rid


def test_every_step_that_can_take_stop_lists_is_wired_to_the_library():
    """
    A step that accepts `stoplist_paths` and does not declare the library
    binding gets no stop list at all unless somebody types a path -- and the
    most frequent words in any corpus are function words, so its output comes
    out as "the, and, of" with no error anywhere.

    That is exactly what happened when the topic models started building their
    own frequency lists: they had inherited the n-gram step's stop lists, and
    the wiring did not come across with the rest. A real user found it within
    minutes, so it is a catalog rule now rather than a thing to remember.
    """
    unwired = []
    for recipe in RECIPES:
        try:
            names = set(describe(load_target(recipe.target)).names)
        except ImportError:
            continue
        if "stoplist_paths" not in names:
            continue
        if recipe.library.get("stoplist_paths") != "stoplists":
            unwired.append(f"{recipe.id} (no library picker)")
        elif not recipe.library_defaults.get("stoplist_paths"):
            unwired.append(f"{recipe.id} (picker, but nothing applied by default)")
    assert not unwired, (
        "these take stop lists but would apply none:\n  " + "\n  ".join(unwired))


def test_every_relabeled_setting_is_a_real_setting_somewhere():
    """
    `SETTING_LABELS` renames settings on the options screen. A name in it that
    no step actually takes is a label nobody will ever see -- most likely a
    parameter that was renamed and left its friendlier wording behind.
    """
    from taters.ui.recipes import SETTING_LABELS

    every = set()
    for recipe in RECIPES:
        try:
            every |= set(describe(load_target(recipe.target)).names)
        except ImportError:
            continue
        every |= set(recipe.vars)

    orphans = sorted(set(SETTING_LABELS) - every)
    assert not orphans, f"labeled but offered by nothing: {orphans}"


def test_a_step_that_relabels_a_setting_really_has_it(recipe):
    """Same check for a step's own overrides, which are the ones most likely
    to name something that has since moved."""
    if not recipe.labels:
        return
    try:
        names = set(describe(load_target(recipe.target)).names)
    except ImportError:
        pytest.skip(f"{recipe.id} needs an optional dependency")
    unknown = sorted(set(recipe.labels) - names - set(recipe.vars))
    assert not unknown, f"{recipe.id} relabels settings it does not take: {unknown}"


def test_a_setting_that_does_nothing_under_the_chosen_rule_is_not_shown():
    """
    The vocabulary is cut one of three ways and each way reads one threshold,
    so two of the three thresholds are always inert. Showing all of them is
    most of why this screen was confusing -- and the inert ones are precisely
    the ones whose names collide with the frequency-list thresholds above
    (`vocab_min_freq` beside `min_freq`, four rows apart and doing different
    jobs at different stages).
    """
    from taters.ui.recipes import gate_of

    for recipe in RECIPES:
        if "vocab_rule" not in (recipe.with_ or {}):
            continue
        assert gate_of(recipe, "vocab_top_n") == ("vocab_rule", "==", "top_n")
        assert gate_of(recipe, "vocab_rank_by") == ("vocab_rule", "==", "top_n")
        assert gate_of(recipe, "vocab_min_freq") == ("vocab_rule", "==", "min_freq")
        assert gate_of(recipe, "vocab_min_obs_pct") == \
            ("vocab_rule", "==", "min_obs_pct")


def test_the_two_vocabulary_stages_do_not_read_alike_on_screen():
    """
    `min_freq` and `vocab_min_freq` are different settings at different
    stages, and as bare parameter names they read as a typo for each other.
    Whatever they are called, the labels have to be tellable apart.
    """
    from taters.ui.recipes import SETTING_LABELS

    pairs = [("min_freq", "vocab_min_freq"),
             ("min_obs_pct", "vocab_min_obs_pct")]
    for first, second in pairs:
        a, b = SETTING_LABELS.get(first), SETTING_LABELS.get(second)
        assert a and b, f"{first}/{second} still show as raw parameter names"
        assert a != b
        # and neither is the other with a word bolted on the front
        assert not b.endswith(a) and not a.endswith(b), (a, b)


def test_a_per_step_label_names_a_setting_that_step_actually_has():
    """
    `Recipe.labels` renames a setting on the options screen where the shared
    word would describe the wrong thing -- `engine` meaning which topic model
    rather than who tags the text. A label keyed by a name the step does not
    have is dead text nobody will ever see, and the mistake is invisible.

    Nothing in the catalog needs an override at the moment (the step that did
    has been folded into the topic models), so this is currently vacuous --
    deliberately so, since it is the guard for the next one.
    """
    for recipe in RECIPES:
        spec = describe(load_target(recipe.target))
        for name in recipe.labels:
            assert name in spec, f"{recipe.id}: labels[{name!r}] is not a setting"


# ---------------------------------------------------------------------------
# source-aware wiring
# ---------------------------------------------------------------------------

def test_every_recipe_declares_at_least_one_source():
    for recipe in RECIPES:
        assert recipe.sources, f"{recipe.id} is offered for no source at all"
        unknown = set(recipe.sources) - set(SOURCES)
        assert not unknown, f"{recipe.id} names unknown source(s) {sorted(unknown)}"


def test_only_text_steps_claim_to_handle_text():
    """
    A recipe that says it works on a folder of essays but shells out to ffmpeg
    would be offered to someone it can only fail for.
    """
    for recipe in RECIPES:
        if set(recipe.sources) - {"media"}:
            assert not recipe.needs_ffmpeg, f"{recipe.id} needs ffmpeg but claims text sources"
            assert recipe.scope == "global", f"{recipe.id} is item-scoped but claims text sources"


def test_text_input_recipes_are_the_ones_wired_to_transcripts():
    """
    `text_input` marks the steps whose input binding gets rewritten. If a step
    reads the merged transcript table and is *not* marked, it would keep that
    wiring on a text source and read a file the run never produced.
    """
    for recipe in RECIPES:
        reads_transcripts = "unified_transcripts_csv" in recipe.requires
        assert reads_transcripts == recipe.text_input, (
            f"{recipe.id}: text_input={recipe.text_input} but "
            f"requires unified_transcripts_csv = {reads_transcripts}"
        )


def test_text_binding_never_mixes_two_input_modes():
    """
    The analyzers take exactly one of `analysis_csv` / `csv_path` / `txt_dir`.
    Two at once is not an error they raise; it is an error they ignore.
    """
    for source in ("txt_dir", "csv"):
        binding = text_binding(source, text_cols=["t"], id_cols=["i"], pass_through=True)
        present = {"csv_path", "txt_dir", "analysis_csv"} & set(binding)
        assert len(present) == 1, f"{source} produced {sorted(present)}"


def test_text_binding_keys_are_all_declared_as_input_keys():
    """
    `TEXT_INPUT_KEYS` is the strip-list applied before a rebind. A key the
    binding sets but the list does not name would survive from the media wiring
    and then be overwritten -- harmless -- but a key the *media* wiring sets and
    the list does not name would survive and be read. Keeping the two in step is
    what makes the rebind total.
    """
    for source in ("txt_dir", "csv"):
        binding = text_binding(source, text_cols=["t"], id_cols=["i"], pass_through=True)
        assert set(binding) <= set(TEXT_INPUT_KEYS), (
            f"{source} sets {sorted(set(binding) - set(TEXT_INPUT_KEYS))}, "
            "which the strip-list does not cover"
        )


def test_media_binding_is_empty_so_transcript_wiring_is_left_alone():
    assert text_binding("media") == {}


def test_text_binding_rejects_an_unknown_source():
    with pytest.raises(KeyError, match="unknown source"):
        text_binding("telepathy")


def test_every_text_input_key_is_a_real_parameter():
    """
    The strip-list names arguments by hand. A rename in `src/` would leave a
    stale entry that silently stops stripping the thing it was meant to strip.
    """
    for recipe in RECIPES:
        if not recipe.text_input:
            continue
        params = {p.name for p in describe(load_target(recipe.target)).params}
        for key in TEXT_INPUT_KEYS & set(recipe.with_):
            assert key in params, f"{recipe.id}: with_[{key!r}] is not a parameter"


def test_source_with_patches_name_real_parameters():
    for recipe in RECIPES:
        if not recipe.source_with:
            continue
        params = {p.name for p in describe(load_target(recipe.target)).params}
        for source, patch in recipe.source_with.items():
            assert source in SOURCES, f"{recipe.id}: unknown source {source!r}"
            for key in patch:
                assert key in params, f"{recipe.id}.source_with[{source!r}]: {key!r} is not a parameter"


def test_text_sources_do_not_promise_results_per_speaker():
    """
    "per speaker per file" is exactly right for a recorded conversation and
    simply untrue for a folder of essays. The help line under a checkbox is
    most of what a non-programmer has to go on, so it has to match the source
    they chose.
    """
    for recipe in RECIPES:
        if not recipe.user_facing or set(recipe.sources) == {"media"}:
            continue
        shown = recipe.text_help or recipe.help
        assert "speaker" not in shown.lower(), (
            f"{recipe.id} offers itself for text but its help says: {shown!r}"
        )


def test_gathered_csv_is_a_real_parameter_on_every_text_analyzer():
    """
    It is set by `text_binding` for the non-media sources, so a rename in
    `src/` would silently stop redirecting the intermediate rather than error.
    """
    for recipe in RECIPES:
        if not recipe.text_input:
            continue
        params = {p.name for p in describe(load_target(recipe.target)).params}
        assert "gathered_csv" in params, recipe.id


# ---------------------------------------------------------------------------
# how many files a GPU step may work on at once
#
# this is the one thing no amount of inspection can work out from the outside:
# `transcribe` and `whisper_embeddings` have nearly identical signatures but
# differ fourfold in how their VRAM scales. we measured on four files with `tiny`:
#
#   transcribe          272 MiB flat at 1, 2 and 4 workers   (one shared model)
#   whisper_embeddings  240 -> 479 -> 893 MiB, and slower every step up
#
# so we declare it per step, once, and whoever writes the step is on the hook
# ---------------------------------------------------------------------------

def test_every_recipe_declares_what_it_does_to_the_gpu(recipe):
    """
    Explicitly, including the CPU ones. Inference exists as a safety net for
    presets written before the field, not as the way the catalog works -- a
    reader should be able to see the answer without deducing it.
    """
    from taters.helpers.gpu import GPU_USE

    assert recipe.gpu_use in GPU_USE, (
        f"{recipe.id} declares gpu_use={recipe.gpu_use!r}; expected one of {GPU_USE}"
    )


def test_a_step_that_shares_one_model_may_run_two_files(recipe):
    from taters.helpers.gpu import WORKER_CAP

    assert WORKER_CAP["gpu_one_model"] == 2
    assert WORKER_CAP["gpu_model_each"] == 1
    assert WORKER_CAP["cpu"] is None


def test_an_undeclared_gpu_step_is_capped_at_one(recipe):
    """
    The safety net. A new module whose author forgot the field costs speed, not
    a crashed batch -- so forgetting fails in the recoverable direction.
    """
    from dataclasses import replace as _replace

    if not any(isinstance(v, str) and "{{var:device}}" in v
               for v in recipe.with_.values()):
        pytest.skip(f"{recipe.id} does not hand a device to anything")

    undeclared = _replace(recipe, gpu_use=None)
    assert undeclared.resolved_gpu_use == "gpu_model_each"
    assert undeclared.worker_cap == 1


def test_a_cpu_step_is_never_capped(recipe):
    """
    `--workers 8` has to keep meaning 8 for ffmpeg conversion and the merges.
    Capping those would make the one concurrency knob a lie.
    """
    if recipe.resolved_gpu_use != "cpu":
        pytest.skip(f"{recipe.id} uses the GPU")
    assert recipe.worker_cap is None


def test_the_cap_reaches_the_step_a_preset_will_carry(recipe):
    """
    Written into the YAML rather than left implicit, so a preset someone edits
    by hand -- or hands to a colleague -- carries the limit with it instead of
    depending on a catalog they may not have.
    """
    step = recipe.to_step()
    if recipe.scope != "item" or recipe.worker_cap is None:
        assert "max_workers" not in step
    else:
        assert step["max_workers"] == recipe.worker_cap


def test_a_global_step_carries_no_ceiling(recipe):
    """A `global` step is one call, so a ceiling on it states nothing."""
    if recipe.scope == "global":
        assert "max_workers" not in recipe.to_step()


def test_the_gpu_steps_are_the_ones_we_think_they_are():
    """
    A canary on the catalog as a whole. If a step starts or stops using the GPU
    and nobody updates its declaration, this is where it shows up -- rather than
    in someone's out-of-memory error.
    """
    from taters.ui.recipes import RECIPES

    on_gpu = {r.id for r in RECIPES if r.resolved_gpu_use != "cpu"}
    assert on_gpu == {
        "transcribe",            # one shared CTranslate2 model
        "diarize",               # a subprocess per file: NeMo, Demucs, aligner
        "whisper_embeddings",    # a subprocess per file: its own encoder
        "archetypes",            # ArchetypeQuantifier loads a sentence-transformer
        "sentence_embeddings",   # loads a sentence-transformer
        "transformer_embeddings",  # loads an encoder
        "adapt_encoder",           # trains one
        "finetune_text_predictor", # trains one with heads
        "text_predictor_apply", "hf_classifier_apply",    # loads the fine-tuned one
        # the n-gram family under engine=stanza: one global call, one model.
        # under the default nltk engine they never touch the GPU, but the
        # declaration is about what the step *can* cost
        "ngram_frequencies",
        "doc_term_matrix",
        "parts_of_speech",
        "score_with_model",             # the saved model may be stanza-built
        # the topic models all build their own matrix, which means they all
        # tokenize -- and under engine="stanza" that is a model on the GPU.
        # `topic_model_mem` only joined this list when the matrix moved inside
        # it; before that somebody else did the tokenizing and it really was
        # pure linear algebra.
        "topic_model_mem", "topic_model_mem_apply",
        "topic_model_lda", "topic_model_lda_apply",
        "topic_model_nmf", "topic_model_nmf_apply",
        "word_vectors_train",           # likewise: tokenizing under stanza
        "word_vectors_apply",
        "cohesion",              # stanza prep and/or the embedding model
    }


# ---------------------------------------------------------------------------
# analysis levels
# ---------------------------------------------------------------------------


def test_every_level_groups_on_columns_the_source_actually_produces():
    """
    A `group_by` naming a column that is not there is a run-time error several
    minutes in, not a no-op. `media` transcripts carry `source` (added by the
    gather) and `speaker` (written by the transcriber); a spreadsheet's columns
    belong to the user, which is what `group_by=None` means.
    """
    available = {"media": {"source", "speaker"}, "txt_dir": {"text_id"}}

    for source, levels in _r.LEVELS.items():
        for level in levels:
            if level.group_by is None:
                assert source == "csv", (
                    f"{source}/{level.id} leaves its columns to the user, but "
                    "only a spreadsheet can do that"
                )
                continue
            unknown = set(level.group_by) - available.get(source, set())
            assert not unknown, f"{source}/{level.id} groups on {sorted(unknown)}"


def test_every_source_has_a_default_level_that_exists():
    for source in _r.SOURCES:
        assert source in _r.DEFAULT_LEVEL, source
        # raises if the default names a level the source doesn't have
        assert _r.level_by_id(source, None).id == _r.DEFAULT_LEVEL[source]


def test_level_ids_are_unique_within_a_source():
    for source, levels in _r.LEVELS.items():
        ids = [level.id for level in levels]
        assert len(ids) == len(set(ids)), f"{source}: {ids}"


def test_the_level_governs_the_text_features_and_nothing_else():
    """
    Derived from the capability graph rather than declared, so a new module
    needs no wizard work. The audio features must stay out: there is no
    per-utterance WAV, so "one row per utterance" cannot mean anything for them.
    """
    aware = {r.id for r in _r.RECIPES if _r.level_aware(r)}

    assert {"readability", "lexical_richness", "dictionaries", "archetypes",
            "sentence_embeddings", "gather_sentence_embeddings"} <= aware
    assert not ({"acoustics", "gather_whisper_embeddings", "convert_to_wav",
                 "transcribe", "diarize", "gather_transcripts"} & aware)


def test_every_per_file_table_has_something_that_collects_it():
    """
    An `item` step runs once per input and writes one CSV per input. Something
    `global` has to collect those, or the run leaves a pile of per-file files
    and no feature table -- and nothing about that failure is visible: every
    step reports success, because every step did succeed.

    `auto_with` is how a step says so, and it has no other structural check on
    it, so forgetting it on a new extractor is the mistake this exists to catch.

    A golden list rather than a rule, because whether per-file output is
    acceptable on its own is a judgment the catalog cannot make. Transcripts
    are: one .csv per recording in `transcripts/` is a thing people came for.
    Per-file acoustic summaries are not -- nobody wants 400 one-row files. So a
    *new* name appearing here is not automatically wrong, but it does mean
    somebody has to decide, which is the point.
    """
    from taters.ui.compose import resolve_selection

    uncollected = set()
    for recipe in RECIPES:
        tables = {c for c in recipe.produces if c.endswith("_csv")}
        if recipe.scope != "item" or not tables:
            continue
        # through the closure, not the catalog. a gather that exists but isn't
        # named anywhere never joins the pipeline, and asserting that it merely
        # exists would pass whether or not `auto_with` mentions it
        pipeline = resolve_selection([recipe.id],
                                     providers={"transcript_csv": recipe.id}
                                     if "transcript_csv" in recipe.produces else {})
        if not any(x.scope == "global" and (tables & x.requires) for x in pipeline):
            uncollected.add(recipe.id)

    assert uncollected == {"transcribe", "diarize"}, (
        f"{sorted(uncollected)} write a table per file with nothing to collect "
        "them. If that is a new feature extractor, name a gather step in its "
        "`auto_with`; if its per-file output stands on its own, add it here."
    )


def test_two_recipes_share_a_save_as_only_as_alternatives():
    """
    `transcribe` and `diarize` both save as `diar`, deliberately: downstream
    steps reference `{{diar...}}` without caring which produced it, and
    `resolve_selection` drops whichever the user did not choose.

    That is only safe while they are alternatives. Two steps that could both
    run with the same `save_as` would have the second overwrite the first for
    everything downstream, which is silent and total.
    """
    from collections import Counter

    for save_as, count in Counter(r.save_as for r in RECIPES).items():
        if count == 1:
            continue
        sharers = [r for r in RECIPES if r.save_as == save_as]
        shared_caps = set.intersection(*(set(r.produces) for r in sharers))
        assert shared_caps, (
            f"{[r.id for r in sharers]} share save_as={save_as!r} but produce "
            "nothing in common, so they are not alternatives and would clobber "
            "each other"
        )
        # ...and the capability they share has to be one the user gets asked
        # about, otherwise nothing would ever drop the losers
        for capability in shared_caps:
            assert len(_r.providers_of(capability)) == count, capability


#: Runtime libraries that cost seconds to import and that the options screen
#: has no use for. Reading a function's parameter names must not pull these in.
_HEAVY = ("torch", "sentence_transformers", "nemo", "transformers",
          "archetypes.archetypes", "faster_whisper", "nltk")


def test_reading_a_steps_settings_does_not_load_its_runtime(recipe):
    """
    From a real report: opening the options for "Archetype similarity" looked
    like the wizard had frozen. It had not -- it was importing
    `archetypes.archetypes`, which imports sentence-transformers, which imports
    torch: **twenty seconds** of nothing, to find out what the parameters are
    called.

    The options screen only ever needs a signature and a docstring. Anything a
    step needs in order to *run* belongs inside the function. This matters more
    with every module built on a pre-trained model, so it is checked per recipe
    rather than left as a note.

    Run in a subprocess: `sys.modules` is process-wide and something else in
    the suite will already have imported half of this list.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import sys
        from taters.ui.introspect import load_target, describe
        describe(load_target({recipe.target!r}))
        heavy = [m for m in {_HEAVY!r} if m in sys.modules]
        print(",".join(heavy))
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=300)
    if r.returncode != 0:
        pytest.skip(f"{recipe.id} is not installed here: {r.stderr.strip()[-120:]}")

    loaded = [m for m in r.stdout.strip().split(",") if m]
    assert not loaded, (
        f"reading {recipe.id}'s settings imported {loaded}. Move that import "
        "inside the function that needs it -- the options screen only reads the "
        "signature, and this is what makes it look frozen."
    )


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------


def test_library_declarations_name_real_kinds_and_real_parameters(recipe):
    """
    `library={"dict_paths": "dictionaries"}` is the whole hookup for a step
    that consumes importable assets. Both halves are checkable: the kind must
    exist in the catalog, and the parameter must exist on the function and be
    wired in the step's `with:` -- a typo in either place would otherwise
    surface as a picker that writes to nowhere.
    """
    from taters.helpers.library import kind_by_id

    for param_name, kind_id in recipe.library.items():
        kind_by_id(kind_id)                       # raises, naming valid kinds
        if param_name not in recipe.library_defaults:
            # optional library params (see Recipe.library_defaults) stay out
            # of `with:` on purpose: untouched means the step runs without
            # one. the mandatory ones have to be wired, otherwise the picker
            # writes to nowhere
            assert param_name in recipe.with_, (recipe.id, param_name)
        spec = describe(load_target(recipe.target))
        assert param_name in {p.name for p in spec.params}, (recipe.id, param_name)


def test_library_default_entries_actually_ship(recipe):
    """
    `library_defaults` names files by their exact filename; a typo would
    surface as a picker that quietly starts with nothing ticked. Every named
    entry must exist in the package's shipped seed folder for that kind.
    """
    from taters.helpers.library import _shipped_dir, kind_by_id

    for param_name, names in recipe.library_defaults.items():
        assert param_name in recipe.library, (recipe.id, param_name)
        shipped = _shipped_dir(kind_by_id(recipe.library[param_name]))
        for name in names:
            assert (shipped / name).is_file(), (recipe.id, param_name, name)


def test_the_dictionary_consumers_declare_their_kinds():
    """The known consumers; a golden pin so the hookup cannot quietly drop."""
    assert _r.by_id("dictionaries").library == {"dict_paths": "dictionaries"}
    assert _r.by_id("archetypes").library == {"archetype_csvs": "archetypes"}
    assert _r.by_id("ngram_frequencies").library == {"stoplist_paths": "stoplists"}
    assert _r.by_id("doc_term_matrix").library == {}


def test_no_step_hardcodes_a_knob_the_pipeline_offers_as_a_variable(recipe):
    """
    From the code review (issues 6 and 7): the embeddings merge hardcoded
    `overwrite_existing: False`, so `--var overwrite_existing=true` regenerated
    every output except the aggregate -- stale numbers presented as fresh. And
    every gather wrote its `out_csv` to a literal `features/...` while reading
    `root_dir` from `{{var:features_dir}}` -- redirecting the variable made the
    run fail or split output across two folders.

    Pinned as a rule, not per instance: any parameter for which the pipeline
    declares a variable must read that variable, everywhere it appears.
    """
    for key, wanted_var in (("overwrite_existing", "overwrite_existing"),):
        if key in recipe.with_:
            assert recipe.with_[key] == f"{{{{var:{wanted_var}}}}}", (
                f"{recipe.id}.{key} hardcodes {recipe.with_[key]!r}"
            )
    for key, value in recipe.with_.items():
        if isinstance(value, str) and value.startswith("features/"):
            raise AssertionError(
                f"{recipe.id}.{key} writes to a literal 'features/' path; "
                "use {{var:features_dir}} so redirecting the variable moves "
                "every output together"
            )


def test_a_gather_scans_exactly_where_its_producer_writes(recipe):
    """
    The whisper extractor once relied on its internal default output folder
    while its gather read `{{var:features_dir}}/whisper-embeddings` -- identical
    at the default, split the moment anyone redirected the variable, failing
    the run with "No files matched". The producer must name the same location,
    as the same template, so the two cannot drift apart.
    """
    for gather_id in recipe.auto_with:
        gather = _r.by_id(gather_id)
        root = gather.with_.get("root_dir")
        if not isinstance(root, str) or not root.startswith("{{var:"):
            continue    # e.g. "{{sent_embeds}}": wired to the step output itself
        assert root in recipe.with_.values(), (
            f"'{gather_id}' scans {root!r} but '{recipe.id}' never writes "
            "there -- pass the same template to the producer's output "
            "directory parameter"
        )


@pytest.mark.skipif(importlib.util.find_spec("parselmouth") is None,
                    reason="needs praat-parselmouth (pip install taters[vocalacoustics])")
def test_the_acoustics_mode_description_offers_only_real_modes():
    """Round-2 cut list: the description offered "full", which the analyzer's
    Mode literal does not contain -- it silently behaved as "simple", the
    quietest possible way to not give someone the features they asked for."""
    import re

    from taters.audio.analyze_vocal_acoustics import Mode
    from typing import get_args

    desc = _r.by_id("acoustics").vars["acoustics_mode"]["desc"]
    offered = set(re.findall(r"[a-z]+", desc.split(":", 1)[1].lower())) - {"or", "and"}
    real = set(get_args(Mode))
    assert offered, "the description stopped naming the options at all"
    assert offered <= real, f"description offers {offered - real} which do not exist"


def _skip_if_target_needs_something_missing(recipe):
    """
    Some targets import a package at module scope that this machine does not
    have -- acoustics needs parselmouth, and split_by_speaker needs pydub,
    which is broken on 3.13 without `audioop-lts`. Loading them to read their
    parameters then fails for a reason that is about the environment rather
    than about the recipe, so the test steps aside.

    Two checks, because the two cases look different. A declared `extra` gets
    a message naming the pip command, which is the useful thing to print. An
    undeclared one -- pydub is a *base* dependency, so it has no extra to
    name -- only shows up when the import is actually tried, which is what
    every sibling test in this file already does.
    """
    from taters.ui.wizard import EXTRA_PROBES

    for extra, probes in EXTRA_PROBES.items():
        if extra in recipe.extras and any(
                importlib.util.find_spec(m) is None for m in probes):
            pytest.skip(f"{recipe.id} needs pip install \"taters[{extra}]\"")
    try:
        load_target(recipe.target)
    except ImportError as exc:
        pytest.skip(f"{recipe.id} needs an optional dependency: {exc}")


def test_every_offered_setting_explains_itself(recipe):
    """
    Reported: "most of the parameters don't seem to have a description
    attached, meaning you'd have to already know what they are for". The
    options screen shows each setting's docstring line; a parameter that is
    offered but undocumented is a question the user cannot answer.
    """
    from taters.ui.wizard import describe, is_wired, load_target

    _skip_if_target_needs_something_missing(recipe)
    spec = describe(load_target(recipe.target))
    bare = [p.name for p in spec.params
            if not p.desc and not p.required and not is_wired(recipe, p.name)]
    assert not bare, f"{recipe.id} offers undocumented settings: {bare}"


def test_param_when_gates_name_real_settings(recipe):
    """A typo'd gate would hide a row forever (or never): both halves of every
    `param_when` entry must exist on the target function."""
    if not recipe.param_when:
        return
    from taters.ui.recipes import gate_of

    spec = describe(load_target(recipe.target))
    names = set(spec.names)
    for param in recipe.param_when:
        gate_param, op, _value = gate_of(recipe, param)   # raises on a bad shape
        assert op in ("==", "!=")
        assert param in names, (recipe.id, param)
        assert gate_param in names, (recipe.id, gate_param)
        assert gate_param not in recipe.param_when, (
            recipe.id, "a gate must not itself be gated")


def test_the_engine_aware_steps_share_the_engine_vars():
    """One answer per pipeline for engine/tokenizer/stanza_lang: a vocabulary
    prepared one way is silently unfindable by a step reading another way."""
    for rid in ("ngram_frequencies", "doc_term_matrix", "parts_of_speech"):
        rec = _r.by_id(rid)
        for var in ("engine", "tokenizer", "stanza_lang"):
            assert rec.with_[var] == f"{{{{var:{var}}}}}", (rid, var)
            assert var in rec.vars, (rid, var)
        assert rec.with_["device"] == "{{var:device}}", rid


# ---------------------------------------------------------------------------
# stages: the feature checklist and the statistics stage are separate screens
# ---------------------------------------------------------------------------

def test_the_feature_checklist_never_offers_an_analysis():
    """
    Mixing "extract cohesion" with "run an ANOVA" on one screen buries both:
    the first is about what to measure, the second about what to conclude,
    and the second cannot even be answered until the first is. `stage` keeps
    them apart, and `user_facing()` is where that has to bite.
    """
    for source in SOURCES:
        offered = {r.id for r in _r.user_facing(source)}
        analyses = {r.id for r in RECIPES if r.stage == "analyze"}
        assert not (offered & analyses), \
            f"{source}: analysis steps leaked onto the feature checklist"


def test_the_analysis_stage_offers_exactly_the_analyses():
    offered = [r.id for r in _r.user_facing("csv", stage="analyze")]
    # reducing the features to components is NOT a row here: it's a setting on
    # each analysis, so it can differ between them and per feature set
    assert offered == ["stats_group_differences", "stats_correlations",
                       "stats_ridge_fit", "stats_classify_fit"]
    # and nothing at all where there's no metadata to analyze
    assert _r.user_facing("txt_dir", stage="analyze") == []


def test_every_stats_step_is_a_tail_global_on_the_csv_path():
    """The stats steps read feature tables the run just wrote; an item-scoped
    or media-source one could not."""
    for recipe in RECIPES:
        if recipe.stage != "analyze":
            continue
        assert recipe.scope == "global", recipe.id
        assert recipe.sources == ("csv",), recipe.id
        assert not recipe.text_input, recipe.id
        assert "stats" in recipe.tags, recipe.id


def test_word_count_is_a_filter_ingredient_not_a_feature_to_pick():
    """Nobody sets out to extract a word count: it is a thing you filter on,
    and the filter question adds the step. Offering it alongside cohesion
    and readability would put a one-column table on a list of analyses."""
    for source in SOURCES:
        offered = {r.id for r in _r.user_facing(source)}
        assert "word_count" not in offered, source

    # ...but it's still a real, joinable step the pipeline can run
    recipe = _r.by_id("word_count")
    assert recipe.feature_table and recipe.scope == "global"
    assert set(recipe.sources) == {"media", "txt_dir", "csv"}


# ---------------------------------------------------------------------------
# closed sets have to say so, or the options screen turns into a guessing game
# ---------------------------------------------------------------------------

def test_a_setting_with_a_closed_set_offers_it_rather_than_a_text_box():
    """The report: the correlations step's "method" was a free-text box, so
    there was no way to learn that pearson, spearman and both were the
    words it knew. A numpydoc type of {"a", "b"} turns it into a picker."""
    expected = {
        ("taters.stats.correlations:analyze_correlations", "method"):
            ["pearson", "spearman", "both"],
        ("taters.stats.group_differences:analyze_group_differences",
         "posthoc"):
            ["auto", "tukey", "games_howell", "bonferroni", "none"],
    }
    for (target, name), choices in expected.items():
        spec = describe(load_target(target))
        param = next(p for p in spec.params if p.name == name)
        assert param.widget == "choice", f"{target}.{name} is a text box"
        assert param.choices == choices


def test_the_correction_menu_cannot_drift_from_the_methods_that_exist():
    """The docstring's set and the implementation's set are two lists of the
    same thing, and two lists of the same thing drift. Adding a correction
    without offering it would leave it unreachable from the wizard."""
    from taters.stats._common import P_ADJUST_METHODS

    for target in ("taters.stats.correlations:analyze_correlations",
                   "taters.stats.group_differences:analyze_group_differences"):
        spec = describe(load_target(target))
        param = next(p for p in spec.params if p.name == "p_adjust")
        assert set(param.choices) == set(P_ADJUST_METHODS), target


def test_every_step_that_takes_a_device_offers_the_devices():
    """`device` is a *shared* setting, and the shared row borrows the spec of
    whichever step happens to reference it first -- so one step documenting
    it as bare `str` made the row a free-text box for everybody, depending on
    what else was selected. Five did. Declaring the set everywhere makes the
    donor irrelevant."""
    from taters.helpers.gpu import DEVICE_CHOICES

    checked = 0
    for recipe in RECIPES:
        if not any(isinstance(v, str) and "{{var:device}}" in v
                   for v in recipe.with_.values()):
            continue
        try:
            spec = describe(load_target(recipe.target))
        except ImportError:
            continue
        param = next((p for p in spec.params if p.name == "device"), None)
        if param is None:
            continue
        checked += 1
        assert param.widget == "choice", \
            f"{recipe.id} offers 'device' as a text box"
        assert set(param.choices) <= set(DEVICE_CHOICES), \
            f"{recipe.id} offers devices that are not devices: {param.choices}"
    assert checked >= 8, "the test stopped finding the steps it guards"


def test_every_merge_recipe_says_which_columns_are_not_features():
    """The Whisper merge averaged `start_time`/`end_time` into predictors
    because, unlike the sentence-embedding merge, it named no exclusions.
    A merge that aggregates every numeric column has to say which are not
    measures."""
    merges = [r for r in RECIPES if r.feature_table
              and r.target.endswith("feature_gather:feature_gather")]
    assert merges, "the catalog has merge steps"
    for r in merges:
        assert r.with_.get("exclude_cols"), f"{r.id} names no exclude_cols"
    whisper = by_id("gather_whisper_embeddings").with_["exclude_cols"]
    assert {"start_time", "end_time"} <= set(whisper)
    acoustics = by_id("gather_acoustics").with_["exclude_cols"]
    assert {"start_s_mean", "end_s_mean", "segment_index_mean"} <= set(acoustics)


def test_gate_rules_read_the_way_they_are_written():
    """`(gate, value)` shows the row while the gate equals the value;
    `(gate, "!=", value)` while it is anything else; an unknown live value
    shows the row -- hiding a setting wrongly is the one mistake the screen
    must never make."""
    from taters.ui.introspect import EMPTY
    from taters.ui.recipes import Recipe, gate_holds, gate_of

    r = Recipe(id="r", label="r", help="", call="c", target="t", scope="global",
               save_as="s", with_={},
               param_when={"stanza_lang": ("engine", "stanza"),
                           "pca_components": ("pca", "!=", "off")})
    eq, ne = gate_of(r, "stanza_lang"), gate_of(r, "pca_components")
    assert eq == ("engine", "==", "stanza") and ne == ("pca", "!=", "off")
    assert gate_of(r, "engine") is None
    assert gate_holds(eq, "stanza") and not gate_holds(eq, "nltk")
    assert gate_holds(ne, "all") and gate_holds(ne, ["liwc"]) and not gate_holds(ne, "off")
    assert gate_holds(eq, EMPTY) and gate_holds(ne, None)
    bad = Recipe(id="b", label="b", help="", call="c", target="t", scope="global",
                 save_as="s", with_={}, param_when={"x": ("y", "~", "z")})
    with pytest.raises(ValueError, match="param_when"):
        gate_of(bad, "x")


def test_every_analysis_brings_the_word_clouds_and_then_the_report():
    """
    The clouds are drawn from the analyses' tables and shown by the report,
    so they sit between the two: every analysis pulls the cloud step in
    beside the report, and the catalog places it after the last analysis
    and before the report, which is what orders the ready global steps.
    """
    from taters.ui.recipes import RECIPES, by_id

    analyses = [r for r in RECIPES if r.stage == "analyze" and r.user_facing
                and r.outcome_kind]
    assert len(analyses) == 4
    for r in analyses:
        assert r.auto_with == ("stats_wordclouds", "stats_report"), r.id
    ids = [r.id for r in RECIPES]
    clouds, report = ids.index("stats_wordclouds"), ids.index("stats_report")
    assert clouds < report
    assert clouds > max(ids.index(r.id) for r in analyses)
    assert by_id("stats_wordclouds").user_facing is False


def test_the_text_stage_figure_steps_come_with_their_tables():
    """The topic model brings its theme clouds and the frequency list its
    corpus cloud; neither is on the checklist, and each needs exactly the
    table its producer writes."""
    from taters.ui.recipes import by_id

    assert "topic_model_mem_wordclouds" in by_id("topic_model_mem").auto_with
    assert "mem_loadings_csv" in by_id("topic_model_mem").produces
    assert by_id("topic_model_mem_wordclouds").requires == {"mem_loadings_csv"}
    assert "ngram_frequency_wordclouds" in by_id("ngram_frequencies").auto_with
    assert by_id("ngram_frequency_wordclouds").requires == {"ngram_freq_csv"}
    for rid in ("topic_model_mem_wordclouds", "ngram_frequency_wordclouds",
                "stats_wordclouds"):
        r = by_id(rid)
        assert r.user_facing is False
        assert r.with_["enabled"] == "{{var:wordclouds}}"


def test_the_retention_rule_is_a_shared_setting_on_the_analyses_and_the_topic_model():
    """Parallel analysis or Kaiser is one answer for the run's reductions,
    and one for the topic model -- everyday settings both, shown only while
    the count is automatic (or the reduction is on)."""
    from taters.ui.recipes import (RECIPES, by_id, gate_of,
                                   _STATS_ANALYSIS_WITH)

    assert _STATS_ANALYSIS_WITH["pca_retain"] == "{{var:stats_pca_retain}}"
    for r in RECIPES:
        if r.stage == "analyze" and r.user_facing and r.outcome_kind:
            assert r.vars["stats_pca_retain"]["default"] == "parallel", r.id
            assert gate_of(r, "pca_retain") == ("pca", "!=", "off"), r.id
    mem = by_id("topic_model_mem")
    assert mem.with_["k_selection"] == "{{var:mem_k_selection}}"
    assert mem.with_["n_components"] == "{{var:mem_components}}"
    # parallel analysis, same as the analyses use: a document-term matrix is
    # wide, and on a wide matrix chance alone makes large eigenvalues, so a
    # fixed cutoff cannot know where the noise floor is.
    assert mem.vars["mem_k_selection"]["default"] == "parallel"
    assert mem.vars["mem_kaiser_cutoff"]["default"] == 1.5
    assert gate_of(mem, "k_selection") == ("n_components", "==", "0")


def test_combining_the_tables_is_a_shared_setting_shown_only_when_together():
    from taters.ui.recipes import by_id, gate_of

    for rid in ("stats_ridge_fit", "stats_classify_fit"):
        r = by_id(rid)
        assert r.with_["set_combos"] == "{{var:stats_set_combos}}", rid
        assert r.vars["stats_set_combos"]["default"] == "subsets", rid
        assert gate_of(r, "set_combos") == ("feature_sets", "!=", "per_table"), rid


def test_every_feature_table_names_its_own_output_file():
    """
    A feature table's file stem is the feature set's name in every result
    -- the metrics, the coefficients, the models, the word clouds. A step
    that leaves the file name to its analyzer inherits whatever the input
    was called, and the sentence embeddings came out as a feature set named
    "texts" after the gathered table (a real report). Every feature step
    names its file, under the features folder, with a stem that says what
    it measures.
    """
    from taters.ui.recipes import RECIPES

    for r in RECIPES:
        if not r.feature_table or r.consumes_feature_tables:
            continue
        named = r.with_.get("out_features_csv") or r.with_.get("out_csv")
        assert isinstance(named, str) and named.startswith("{{var:features_dir}}/"), r.id
        stem = named.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        assert stem and "texts" not in stem and "{{" not in stem.replace(
            "{{var:weighting}}", ""), r.id
