"""
Tests for the composer: chosen features in, runnable preset out.

The headline test here is `test_composed_presets_pass_the_shipped_validator`.
It runs every combination the composer can produce through `preset_checks`, the
same module that guards `conversation_video.yaml` and `single_speaker_media.yaml`.
That means the wizard structurally cannot emit a pipeline that names a function
which does not exist, passes a parameter that was renamed, or references an
artifact no earlier step produced.

The other headline is the round-trip: both shipped presets are reproduced
exactly from a recipe selection. Those two files are the known-good wiring for
this whole project, so "the composer agrees with them" is the strongest
statement available about whether its output will actually run.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import yaml
from preset_checks import check_preset, assert_valid_preset

from taters.ui.compose import ComposeError, compose, resolve_selection, slugify
from taters.ui.recipes import by_id, user_facing

PRESET_DIR = Path(__file__).parent.parent / "src" / "taters" / "pipelines" / "presets"


def ids_of(steps) -> list[str]:
    return [r.id for r in steps]


# ---------------------------------------------------------------------------
# Dependency closure
# ---------------------------------------------------------------------------

def test_a_single_pick_pulls_in_what_it_needs():
    """
    Ticking one box has to produce a runnable pipeline. Readability reads the
    merged transcript table, which needs transcripts, which need a WAV — four
    steps from one tick, none of which a user should have to know about.
    """
    steps = ids_of(resolve_selection(["readability"]))
    # the fifth one is the descriptives of the table, which every run with a
    # feature table gets.
    assert steps == ["convert_to_wav", "transcribe", "gather_transcripts",
                     "readability", "describe_features"]


def test_prerequisites_are_shared_not_duplicated():
    """Five text measures need one merge step between them, not five."""
    steps = ids_of(resolve_selection(
        ["dictionaries", "readability", "lexical_richness", "archetypes",
         "sentence_embeddings"]
    ))
    assert steps.count("gather_transcripts") == 1
    assert steps.count("convert_to_wav") == 1


def test_auto_with_pulls_in_the_follow_up_gather():
    """
    A gather step does not *provide* anything the feature step needs — it tidies
    up after it — so `requires` cannot pull it in. `auto_with` does.
    """
    assert "gather_acoustics" in ids_of(resolve_selection(["acoustics"]))
    assert "gather_sentence_embeddings" in ids_of(resolve_selection(["sentence_embeddings"]))


def test_item_steps_come_before_global_steps():
    """
    The runner fans item steps out across files and puts a barrier before each
    global step. A global step that reads what the per-file steps wrote must
    therefore come after all of them.
    """
    for recipe_id in [r.id for r in user_facing()]:
        scopes = [r.scope for r in resolve_selection([recipe_id])]
        assert scopes == sorted(scopes, key=lambda s: 0 if s == "item" else 1), recipe_id


def test_a_step_never_precedes_something_it_depends_on():
    steps = resolve_selection([r.id for r in user_facing()],
                              providers={"transcript_csv": "diarize"})
    available: set[str] = set()
    for recipe in steps:
        assert recipe.requires <= available, f"{recipe.id} runs before its inputs exist"
        available |= recipe.produces


def test_ordering_is_deterministic():
    """The same answers must always give the same file, in any tick order."""
    a = ids_of(resolve_selection(["readability", "acoustics", "transcribe"]))
    b = ids_of(resolve_selection(["acoustics", "transcribe", "readability"]))
    assert a == b


# ---------------------------------------------------------------------------
# Choosing between providers
# ---------------------------------------------------------------------------

def test_the_transcript_producer_is_honored():
    assert "diarize" in ids_of(resolve_selection(
        ["readability"], providers={"transcript_csv": "diarize"}))
    assert "transcribe" in ids_of(resolve_selection(
        ["readability"], providers={"transcript_csv": "transcribe"}))


def test_only_one_transcript_producer_ends_up_in_the_pipeline():
    """
    Both write to `save_as: diar`. Two of them in one run would have the second
    silently clobber the first for every downstream step.
    """
    steps = ids_of(resolve_selection(["acoustics", "readability", "whisper_embeddings"],
                                     providers={"transcript_csv": "diarize"}))
    assert steps.count("diarize") == 1
    assert "transcribe" not in steps


def test_an_impossible_provider_choice_is_rejected():
    with pytest.raises(ComposeError, match="does not produce"):
        resolve_selection(["readability"], providers={"transcript_csv": "acoustics"})


def test_picking_a_producer_directly_needs_no_provider_hint():
    assert "diarize" in ids_of(resolve_selection(["diarize", "readability"]))


def test_empty_selection_is_refused():
    with pytest.raises(ComposeError, match="nothing selected"):
        compose([])


# ---------------------------------------------------------------------------
# The round trip against the shipped presets
# ---------------------------------------------------------------------------

ROUND_TRIPS = {
    "single_speaker_media": (
        ["transcribe", "acoustics", "whisper_embeddings", "dictionaries",
         "readability", "lexical_richness", "archetypes", "sentence_embeddings"],
        {"transcript_csv": "transcribe"},
    ),
    "conversation_video": (
        ["diarize", "split_by_speaker", "acoustics", "whisper_embeddings",
         "dictionaries", "readability", "lexical_richness", "archetypes",
         "sentence_embeddings"],
        {"transcript_csv": "diarize"},
    ),
}


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_the_composer_reproduces_the_shipped_presets(name):
    """
    Both shipped presets are hand-written, hand-tuned, and known to work. If
    the composer can rebuild them step for step, its output is wired the same
    way theirs is — which is a much stronger claim than any assertion this file
    could make on its own.
    """
    selected, providers = ROUND_TRIPS[name]
    shipped = yaml.safe_load((PRESET_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    built = compose(selected, providers=providers, name=name)

    assert ([(s["scope"], s["call"]) for s in built["steps"]]
            == [(s["scope"], s["call"]) for s in shipped["steps"]])


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_the_composer_reproduces_the_shipped_step_parameters(name):
    """
    Beyond the step list: every `with:` block should match too. One documented
    exception, below — the composer is deliberately more explicit than
    `conversation_video.yaml` is.
    """
    selected, providers = ROUND_TRIPS[name]
    shipped = yaml.safe_load((PRESET_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    built = compose(selected, providers=providers, name=name)

    # strict=True: if the step counts ever diverge we want this to fail rather
    # than silently comparing only the shorter prefix.
    for want, got in zip(shipped["steps"], built["steps"], strict=True):
        for key, value in (want.get("with") or {}).items():
            assert got["with"].get(key) == value, f"{want['call']}.{key}"


def test_the_transcript_directory_is_wired_explicitly():
    """
    Both transcript producers must be *told* where to write, not left to their
    default of `./transcripts`.

    The gather step downstream reads `{{var:transcripts_dir}}`. If the producer
    relies on its default instead, the two agree only until somebody passes
    `--var transcripts_dir=...`, at which point the transcripts go one place and
    the gather looks in another and the run quietly produces nothing.
    `conversation_video.yaml` had exactly that gap; it is fixed, and this keeps
    it fixed for both producers.
    """
    for producer, call in [("diarize", "diarize_with_thirdparty"),
                           ("transcribe", "transcribe_with_whisper")]:
        built = compose([producer], providers={"transcript_csv": producer}, name="x")
        step = next(s for s in built["steps"] if s["call"].endswith(call))
        assert step["with"]["out_dir"] == "{{var:transcripts_dir}}", producer


# ---------------------------------------------------------------------------
# The validator, applied to composed output
# ---------------------------------------------------------------------------

def _selections():
    """Every single pick, every pair, and one all-in run per transcript source."""
    ids = [r.id for r in user_facing()]
    for one in ids:
        yield [one]
    for pair in itertools.combinations(ids, 2):
        yield list(pair)
    yield ids


@pytest.mark.parametrize("selection", list(_selections()),
                         ids=lambda s: "+".join(s) if len(s) < 3 else "all")
@pytest.mark.parametrize("producer", ["transcribe", "diarize"])
def test_composed_presets_pass_the_shipped_validator(selection, producer):
    """
    The important one.

    `check_preset` is the same validation `test_presets.py` runs over the
    presets that ship with Taters: every `call` resolves to a real function,
    every `with:` key is a real parameter of it, every required parameter is
    supplied, and every `{{artifact}}` was produced by an earlier step that the
    current scope can see.

    Running it over the composer's whole output space means a wizard user
    cannot be handed a broken pipeline — whatever they tick.
    """
    # a selection that names a transcript producer has already answered the
    # question; forcing the other one would be a contradiction the wizard never
    # actually builds (it drops the unchosen producer before composing).
    others = {"transcribe", "diarize"} - {producer}
    if others & set(selection):
        pytest.skip(f"{sorted(others & set(selection))} contradicts producer={producer}")

    preset = compose(selection, providers={"transcript_csv": producer},
                     name="test pipeline", file_type="audio")
    problems = check_preset(preset)
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Variables and overrides
# ---------------------------------------------------------------------------

def test_base_variables_are_always_present():
    preset = compose(["transcribe"], name="x")
    assert {"device", "overwrite_existing"} <= set(preset["vars"])


def test_only_the_variables_that_are_used_are_declared():
    """
    A `vars:` block full of settings that nothing reads is noise in
    --describe-preset, and misleading: changing one would appear to do nothing.
    """
    preset = compose(["transcribe"], name="x")
    assert "dictionaries_path" not in preset["vars"]
    assert "acoustics_mode" not in preset["vars"]


def test_the_chosen_transcript_producer_sets_the_whisper_model_default():
    """
    Both producers declare `whisper_model`, with different defaults — `base.en`
    for plain transcription, `base` for diarization, which handles other
    languages. Whichever one is in the pipeline must win, and the embedding
    step downstream inherits it rather than overriding it.
    """
    assert compose(["transcribe", "whisper_embeddings"],
                   providers={"transcript_csv": "transcribe"},
                   name="x")["vars"]["whisper_model"] == "base.en"
    assert compose(["diarize", "whisper_embeddings"],
                   providers={"transcript_csv": "diarize"},
                   name="x")["vars"]["whisper_model"] == "base"


def test_var_values_override_defaults():
    preset = compose(["transcribe"], var_values={"device": "cuda"}, name="x")
    assert preset["vars"]["device"] == "cuda"
    assert preset["meta"]["variables"]["device"]["default"] == "cuda"


def test_an_unknown_var_value_is_kept_and_documented():
    preset = compose(["transcribe"], var_values={"custom_thing": 7}, name="x")
    assert preset["vars"]["custom_thing"] == 7
    assert "custom_thing" in preset["meta"]["variables"]


def test_step_overrides_land_in_the_with_block():
    preset = compose(["transcribe"], overrides={"transcribe": {"beam_size": 1}}, name="x")
    step = next(s for s in preset["steps"] if s["call"].endswith("transcribe_with_whisper"))
    assert step["with"]["beam_size"] == 1


def test_overrides_do_not_leak_between_compositions():
    """
    `Recipe.with_` is shared class-level state. If a step were mutated in place
    rather than copied, one composition would silently poison the next.
    """
    compose(["transcribe"], overrides={"transcribe": {"beam_size": 99}}, name="x")
    clean = compose(["transcribe"], name="y")
    step = next(s for s in clean["steps"] if s["call"].endswith("transcribe_with_whisper"))
    assert "beam_size" not in step["with"]
    assert by_id("transcribe").with_.get("beam_size") is None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_is_complete_enough_for_describe_preset():
    """`--describe-preset` and the docs index both read these fields."""
    preset = compose(["transcribe", "readability"], name="My run", file_type="audio")
    meta = preset["meta"]
    for key in ("id", "title", "summary", "inputs", "requirements", "variables",
                "tags", "version", "notes", "cli_example"):
        assert meta.get(key) not in (None, ""), f"meta.{key} is empty"


def test_the_notes_say_what_was_added_automatically():
    """
    A user who ticked two boxes and got seven steps deserves to know why,
    especially when they open the file six months later.
    """
    notes = compose(["readability"], name="x")["meta"]["notes"]
    assert "Readability scores" in notes
    assert "Convert media to WAV" in notes
    assert "prerequisites" in notes


def test_requirements_report_the_extras_that_are_actually_needed():
    # readability declares no extra: textstat is a base dependency, so there's
    # nothing more to install to run it.
    meta = compose(["acoustics", "readability"], name="x")["meta"]
    assert set(meta["requirements"]["extras"]) == {"vocalacoustics"}
    assert compose(["readability"], name="x")["meta"]["requirements"]["extras"] == []
    assert compose(["transcribe"], name="x")["meta"]["requirements"]["extras"] == []


def test_the_cli_example_is_copy_pasteable():
    example = compose(["transcribe"], name="My run", file_type="audio",
                      root_dir="/data/talks")["meta"]["cli_example"]
    assert "--root_dir /data/talks" in example
    assert "--file_type audio" in example
    assert "--preset my_run" in example


@pytest.mark.parametrize("title,expected", [
    ("My run", "my_run"),
    ("Lecture  Analysis!", "lecture_analysis"),
    ("  ", "custom_pipeline"),
    ("2024 interviews", "2024_interviews"),
])
def test_slugify_produces_usable_filenames(title, expected):
    assert slugify(title) == expected


def test_the_preset_id_matches_the_title_slug():
    """`--preset <id>` has to work on the file the wizard wrote."""
    preset = compose(["transcribe"], name="Lecture Pilot")
    assert preset["meta"]["id"] == "lecture_pilot"


def test_a_contradictory_provider_choice_is_refused():
    """
    Selecting diarization while forcing plain transcription as the transcript
    producer leaves nothing to run. Silently composing an empty pipeline would
    be far worse than saying so.
    """
    with pytest.raises(ComposeError, match="removed everything"):
        compose(["diarize"], providers={"transcript_csv": "transcribe"}, name="x")


def test_the_unchosen_producer_is_dropped_from_the_selection():
    """
    Both producers write to `save_as: diar`, so only one can be in a pipeline.
    Once a choice is made, the other goes -- even if it was ticked.
    """
    steps = ids_of(resolve_selection(["transcribe", "diarize", "readability"],
                                     providers={"transcript_csv": "transcribe"}))
    assert "transcribe" in steps and "diarize" not in steps


# ---------------------------------------------------------------------------
# Which choices still need making
# ---------------------------------------------------------------------------

def test_pending_choices_sees_a_transitive_need():
    """
    "Readability scores" never mentions a transcript. It reads the merged
    transcript table, and the *merge step* is what needs one — so the
    requirement only appears once prerequisites are resolved.
    """
    from taters.ui.compose import pending_choices
    assert "transcript_csv" in pending_choices(["readability"])
    assert "transcript_csv" in pending_choices(["acoustics"])


def test_pending_choices_is_quiet_when_the_user_already_decided():
    from taters.ui.compose import pending_choices
    assert pending_choices(["transcribe", "readability"]) == {}
    assert pending_choices(["diarize", "readability"]) == {}


def test_pending_choices_is_quiet_when_nothing_is_ambiguous():
    """`split_by_speaker` needs a transcript too, but it is not itself a choice."""
    from taters.ui.compose import pending_choices
    assert pending_choices(["transcribe", "split_by_speaker"]) == {}


# ---------------------------------------------------------------------------
# Text and spreadsheet sources
# ---------------------------------------------------------------------------
#
# the media path assumes there's no text until Whisper makes some. these tests
# cover the other two sources: the text already exists, so transcription, WAV
# conversion and ffmpeg all drop out, and the analyzers read the user's files
# directly. a whole lot of people only ever want this.

TEXT_SOURCES = ["txt_dir", "csv"]


def _text_kwargs(source: str) -> dict:
    """The source-specific arguments a caller would pass alongside a selection."""
    if source == "csv":
        return {"source": "csv", "input_path": "responses.csv",
                "text_cols": ["response"], "id_cols": ["pid"]}
    return {"source": "txt_dir", "input_path": "./essays"}


@pytest.mark.parametrize("source", TEXT_SOURCES)
def test_a_text_source_needs_no_transcription(source):
    """
    The whole point. On media, "readability" pulls in three prerequisites; on
    text it stands alone, which is the difference between a two-second run and
    a multi-gigabyte install.
    """
    assert ids_of(resolve_selection(["readability"])) == [
        "convert_to_wav", "transcribe", "gather_transcripts", "readability",
        "describe_features",
    ]
    assert ids_of(resolve_selection(["readability"], source=source)) == [
        "readability", "describe_features"]


@pytest.mark.parametrize("source", TEXT_SOURCES)
def test_a_text_preset_has_no_item_steps(source):
    """
    Every text analyzer is GLOBAL-scoped, so `run_preset` skips input discovery
    and `root_dir` is never required. That is what lets the wizard point at a
    single spreadsheet rather than a folder.
    """
    preset = compose(user_facing_ids(source), name="x", **_text_kwargs(source))
    assert all(step["scope"] == "global" for step in preset["steps"])
    assert "--root_dir" not in preset["meta"]["cli_example"]


def user_facing_ids(source: str) -> list[str]:
    return [r.id for r in user_facing(source)]


@pytest.mark.parametrize("source", TEXT_SOURCES)
def test_audio_features_are_not_offered_for_text(source):
    assert "acoustics" not in user_facing_ids(source)
    assert "transcribe" not in user_facing_ids(source)
    with pytest.raises(ComposeError, match="cannot run on"):
        compose(["acoustics"], name="x", **_text_kwargs(source))


@pytest.mark.parametrize("source", TEXT_SOURCES)
def test_the_transcript_wiring_is_replaced_not_merely_supplemented(source):
    """
    The analyzers accept exactly one of `analysis_csv` / `csv_path` / `txt_dir`
    and ignore arguments belonging to the other modes. A leftover
    `csv_path: "{{transcripts_all}}"` sitting beside a new `txt_dir` would not
    raise -- it would quietly read a transcript this run never produced. So the
    binding is rebuilt from scratch, and this asserts nothing survived.
    """
    preset = compose(["readability"], name="x", **_text_kwargs(source))
    with_ = preset["steps"][0]["with"]

    assert "analysis_csv" not in with_
    if source == "txt_dir":
        assert with_["txt_dir"] == "{{var:input_dir}}"
        assert "csv_path" not in with_
        # group_by belongs to the transcript case: many utterance rows per
        # speaker. here one file is one text, so it mustn't carry over.
        assert "group_by" not in with_
    else:
        assert with_["csv_path"] == "{{var:input_csv}}"
        assert with_["text_cols"] == ["response"]
        assert "txt_dir" not in with_


def test_the_input_path_travels_as_a_variable():
    """
    Baked into five separate steps, re-pointing a preset at next term's essays
    would mean editing YAML. As a variable it is one `--var`.
    """
    preset = compose(["readability", "dictionaries"], name="x",
                     source="txt_dir", input_path="./essays")
    assert preset["vars"]["input_dir"] == "./essays"
    readers = [s for s in preset["steps"] if "txt_dir" in s["with"]]
    assert len(readers) == 2
    assert all(s["with"]["txt_dir"] == "{{var:input_dir}}" for s in readers)


def test_the_embedding_gather_exists_only_where_it_combines_something():
    """
    On media (speaker level) the aggregate genuinely combines utterances.
    Essays are already one row per text, so the "merge" was a copy of its
    input with __mean stapled on -- and, worse, `aggregate_features` drops
    every non-numeric column that is not a group key, so the copy *lost*
    columns. A step that subtracts is worse than no step (round-2 issues
    26/27): raw-level merges are now dropped at compose time.
    """
    media = compose(["sentence_embeddings"], name="x")
    text = compose(["sentence_embeddings"], name="x", source="txt_dir", input_path="./e")

    gather_media = [s for s in media["steps"] if s["save_as"] == "sent_embeds_agg"][0]
    assert "speaker" in gather_media["with"]["group_by"]
    assert not [s for s in text["steps"] if s["save_as"] == "sent_embeds_agg"]


@pytest.mark.parametrize("source", TEXT_SOURCES)
@pytest.mark.parametrize("n", [1, 2])
def test_composed_text_presets_pass_the_shipped_validator(source, n):
    """
    The same guarantee the media path gets, over the text sources: every
    combination goes through the validator that guards the shipped presets.
    """
    ids = user_facing_ids(source)
    for selection in itertools.combinations(ids, n):
        preset = compose(list(selection), name="test", **_text_kwargs(source))
        problems = check_preset(preset)
        assert not problems, f"{source} {selection}:\n" + "\n".join(problems)


@pytest.mark.parametrize("source", TEXT_SOURCES)
def test_every_text_feature_at_once_still_validates(source):
    preset = compose(user_facing_ids(source), name="test", **_text_kwargs(source))
    assert not check_preset(preset)


@pytest.mark.parametrize("source", TEXT_SOURCES)
def test_the_intermediate_table_stays_inside_the_run(source):
    """
    The analyzers build an "analysis-ready" table before measuring anything,
    and by default write it beside the *source*. Analyzing a spreadsheet in
    someone's Downloads folder then leaves a file in their Downloads folder --
    which may not even be writable.

    `gathered_csv` is a relative path, so it resolves inside the folder
    `run_preset(work_dir=...)` runs in: the pipeline's own.
    """
    preset = compose(user_facing_ids(source), name="x", **_text_kwargs(source))

    for step in preset["steps"]:
        with_ = step["with"]
        if "csv_path" in with_ or "txt_dir" in with_:
            assert with_.get("gathered_csv") == "gathered/texts.csv", step["call"]
            assert not Path(with_["gathered_csv"]).is_absolute()


def test_the_media_path_is_left_alone():
    """
    On media the analyzers read the merged transcript table, which the run
    already wrote inside its own folder -- so the intermediate lands there too,
    and the shipped presets stay byte-identical to what the round-trip asserts.
    """
    preset = compose(["readability"], name="x")
    step = next(s for s in preset["steps"] if "readability" in s["call"])
    assert "gathered_csv" not in step["with"]


# ---------------------------------------------------------------------------
# GPU worker limits reach the composed preset
# ---------------------------------------------------------------------------

def test_a_composed_gpu_step_carries_its_ceiling():
    """
    Written into the YAML, not left to the runner's guess. A composed preset is
    a file someone keeps, edits, and sends to a colleague -- so the limit has to
    travel with it rather than depend on a recipe catalog at the far end.
    """
    preset = compose(["whisper_embeddings"],
                     providers={"transcript_csv": "transcribe"}, name="x")

    caps = {s["call"]: s.get("max_workers") for s in preset["steps"]}
    assert caps["potato.audio.extract_whisper_embeddings"] == 1
    assert caps["potato.audio.transcribe_with_whisper"] == 2
    assert caps["potato.audio.convert_to_wav"] is None, "ffmpeg is not GPU work"


def test_a_composed_cpu_pipeline_names_no_ceilings():
    preset = compose(["readability"], providers={"transcript_csv": "transcribe"},
                     source="csv", input_path="essays.csv", name="x")

    for step in preset["steps"]:
        if step["call"] != "potato.audio.transcribe_with_whisper":
            assert "max_workers" not in step, step["call"]


def test_the_composed_ceiling_survives_the_real_validator():
    """
    `max_workers` is a new step key. The shipped-preset validator has to accept
    it -- and reject a nonsensical one -- or every composed preset fails the
    checks that exist to catch typos.
    """
    from preset_checks import check_preset

    preset = compose(["whisper_embeddings"],
                     providers={"transcript_csv": "transcribe"}, name="x")
    assert check_preset(preset) == []

    broken = {**preset, "steps": [{**preset["steps"][0], "max_workers": 0}]}
    assert any("at least 1" in p for p in check_preset(broken))


# ---------------------------------------------------------------------------
# What a spreadsheet row means
#
# a CSV is the one source where "one result per input" is ambiguous: a row can
# be the unit, or many rows can belong to one participant. getting this wrong
# is invisible -- every output is well-formed either way, just at a grain
# nobody actually chose.
# ---------------------------------------------------------------------------


def _csv_preset(**kw):
    from taters.ui.compose import compose
    kw.setdefault("text_cols", ["body"])
    return compose(["readability", "sentence_embeddings"], providers={},
                   name="d", source="csv", input_path="/x.csv", **kw)


def _with_of(preset, fragment):
    return next(s["with"] for s in preset["steps"] if fragment in s["call"])


def test_grouping_reaches_every_analyzer_not_just_one():
    """
    The whole point of asking. If one analyzer groups and another does not, the
    two feature tables have different row counts and cannot be joined.
    """
    preset = _csv_preset(group_by=["participant_id"])

    for fragment in ("analyze_readability", "extract_sentence_embeddings"):
        assert _with_of(preset, fragment)["group_by"] == ["participant_id"]


def test_grouping_replaces_the_row_identifier_rather_than_joining_it():
    """
    `text_gather` composes `text_id` from `id_cols` *only when not grouping*,
    so sending both would silently ignore one of them.
    """
    preset = _csv_preset(group_by=["pid"], id_cols=["row_key"])
    block = _with_of(preset, "analyze_readability")

    assert block["group_by"] == ["pid"]
    assert "id_cols" not in block


def test_without_grouping_the_row_stays_the_unit():
    preset = _csv_preset(id_cols=["row_key"])
    block = _with_of(preset, "analyze_readability")

    assert block["id_cols"] == ["row_key"]
    assert "group_by" not in block


def test_separate_columns_survive_to_the_output():
    """
    Measured, on real numbers, in round 1: with `mode="separate"` a headline
    embedding of 1.0 and a body embedding of 9.0 once merged into a single row
    of 5.0. There is no raw-level merge any more (round-2 issues 26/27), so the
    analyzer's own output is the deliverable -- and it must carry `source_col`,
    or the two rows are indistinguishable.
    """
    preset = _csv_preset(text_cols=["headline", "body"], text_mode="separate",
                         id_cols=["pid"])

    analyzer = _with_of(preset, "extract_sentence_embeddings")
    assert "source_col" in analyzer["pass_through_cols"]
    assert "pid" in analyzer["pass_through_cols"]
    # the ids build text_id here exactly the way they do for every other table
    # -- see test_every_text_table_composes_its_id_the_same_way.
    assert analyzer["id_cols"] == ["pid"]
    assert not [s for s in preset["steps"] if "feature_gather" in s["call"]]


def test_concat_does_not_carry_a_column_that_will_not_exist():
    """`source_col` is only written in separate mode; asking for it otherwise
    would make the analyzer carry a column the gather never produced."""
    preset = _csv_preset(text_cols=["headline", "body"], text_mode="concat",
                         id_cols=["pid"])

    assert _with_of(preset, "extract_sentence_embeddings")["pass_through_cols"] \
        == ["pid"]


def test_the_delimiter_is_a_variable_not_a_hardcoded_comma():
    """The picker offers .tsv, and a comma reader turns one into a single
    column whose name is the entire header line."""
    preset = _csv_preset()

    assert _with_of(preset, "analyze_readability")["delimiter"] \
        == "{{var:csv_delimiter}}"
    assert preset["vars"]["csv_delimiter"] == ","


# ---------------------------------------------------------------------------
# Analysis level
#
# what a row of results describes used to be fixed per recipe, and inconsistent
# between recipes: the four text analyzers joined a speaker's words and
# measured once, sentence embeddings measured each utterance and averaged.
# neither was a choice, and the two differ by about a third on vocabulary
# measures.
# ---------------------------------------------------------------------------

LEVEL_TEXT_STEPS = ("analyze_readability", "analyze_lexical_richness",
                    "analyze_with_dictionaries", "analyze_with_archetypes")


def _media(level=None, extra=()):
    from taters.ui.compose import compose
    return compose(["readability", "lexical_richness", "dictionaries",
                    "archetypes", "sentence_embeddings", *extra],
                   providers={"transcript_csv": "transcribe"}, name="x",
                   level=level)


def _step(preset, fragment, aggregate=None):
    for s in preset["steps"]:
        if fragment in s["call"] and (aggregate is None
                                      or s["with"].get("aggregate") is aggregate):
            return s
    raise AssertionError(f"no step matching {fragment!r}")


def test_no_level_is_exactly_the_speaker_level():
    """
    The shipped presets and the CLI depend on the default being what Taters has
    always done. Anything else silently changes numbers in existing pipelines.
    """
    assert _media() == _media("speaker")


def test_every_text_analyzer_follows_the_level_together():
    """
    One level per run, or the feature tables have different row counts and
    cannot be joined -- the exact defect this replaced on the CSV path.
    """
    for level, want in [("speaker", ["source", "speaker"]),
                        ("conversation", ["source"])]:
        preset = _media(level)
        for call in LEVEL_TEXT_STEPS:
            assert _step(preset, call)["with"]["group_by"] == want, (level, call)
        # ...and the merge that finishes sentence embeddings collapses on the
        # same key, otherwise that one feature lands at a different grain than
        # the rest.
        assert _step(preset, "feature_gather",
                     aggregate=True)["with"]["group_by"] == want


def test_the_utterance_level_leaves_no_grouping_on_any_analyzer():
    preset = _media("utterance")

    for call in LEVEL_TEXT_STEPS:
        assert "group_by" not in _step(preset, call)["with"], call


def test_the_raw_level_ships_no_merge_at_all():
    """
    `aggregate: True` with no key collapses *everything* into one row, and
    keyed on the unique row id it is a lossy copy (non-key text columns are
    dropped). Neither is a merge, so at the raw level the step is dropped.
    """
    preset = _media("utterance")

    assert not [s for s in preset["steps"]
                if s["with"].get("aggregate") is True
                and "sentence" in str(s["with"].get("root_dir", ""))]
    # ...while the audio merges, whose grain the level doesn't govern, stay put.
    assert [s for s in preset["steps"] if s["with"].get("aggregate") is True] == []


def test_per_utterance_rows_are_unique_and_traceable():
    """
    The round-2 reproduction: at the utterance level every utterance by one
    speaker shared a text_id ("file|SPEAKER_00") with no utterance index, and
    the embeddings output kept only synthetic row ids -- neither joinable to
    anything. start_time is what tells utterances apart, and carrying it is
    also the join key back to the transcript.
    """
    preset = _media("utterance")

    for call in LEVEL_TEXT_STEPS:
        assert _step(preset, call)["with"]["id_cols"] == [
            "source", "speaker", "start_time"]
    emb = _step(preset, "extract_sentence_embeddings")
    assert emb["with"]["pass_through_cols"] == [
        "source", "speaker", "start_time"]


def test_the_level_does_not_touch_the_audio_features():
    """
    `acoustics` groups by `speaker` within one file because it is item-scoped,
    and `gather_whisper_embeddings` aggregates audio segments. "One row per
    utterance" has no meaning for either -- there is no per-utterance WAV.
    """
    preset = _media("utterance", extra=("acoustics", "whisper_embeddings"))

    assert _step(preset, "analyze_vocal_acoustics")["with"]["group_by"] == ["speaker"]
    whisper = next(s for s in preset["steps"]
                   if "feature_gather" in s["call"]
                   and "whisper" in str(s["with"].get("out_csv", "")))
    assert whisper["with"]["group_by"] == ["speaker"]


def test_a_non_default_level_does_not_overwrite_the_default_level_output():
    """Two levels are two different tables; one filename would lose one of them."""
    default = _step(_media(), "analyze_readability")["with"]["out_features_csv"]
    other = _step(_media("utterance"), "analyze_readability")["with"]["out_features_csv"]

    assert default == "{{var:features_dir}}/readability.csv"
    assert other == "{{var:features_dir}}/readability_by-utterance.csv"


def test_the_level_is_recorded_in_the_preset():
    """So `--describe-preset` and a reader six months later can both see it."""
    preset = _media("conversation")

    assert preset["meta"]["level"] == "conversation"
    assert "whole conversation (transcript)" in preset["meta"]["notes"]


def test_naming_csv_group_columns_chooses_the_grouped_level():
    """A caller that says what to group by has plainly chosen to group."""
    from taters.ui.compose import compose
    preset = compose(["readability"], providers={}, name="x", source="csv",
                     input_path="/x.csv", text_cols=["body"],
                     group_by=["participant_id"])

    assert preset["meta"]["level"] == "group"
    assert _step(preset, "analyze_readability")["with"]["group_by"] == ["participant_id"]


def test_the_merged_transcripts_live_with_the_transcripts():
    """
    Requested directly: the unified table used to land in the pipeline root --
    the module's own default leaking through -- as the one loose file among
    the tidy per-purpose folders. It belongs beside what it merges.
    """
    preset = _media()
    merge = next(s for s in preset["steps"]
                 if "unified" in str(s.get("save_as", "")) or
                 str(s["with"].get("out_csv", "")).endswith("all_transcripts.csv"))

    assert merge["with"]["out_csv"] == "{{var:transcripts_dir}}/all_transcripts.csv"


# ---------------------------------------------------------------------------
# Tick order must not matter (round-2 issue 24)
# ---------------------------------------------------------------------------


def test_the_selection_composes_the_same_pipeline_in_any_order():
    """
    From the second review: ["readability", "diarize"] reached readability's
    transcript need before the user's diarize, auto-added `transcribe`, and
    shipped BOTH producers -- two steps sharing save_as="diar", silently
    clobbering each other. The reverse order was correct.
    """
    from itertools import permutations

    from taters.ui.compose import resolve_selection

    for sel in permutations(["readability", "diarize"]):
        steps = resolve_selection(list(sel))
        ids = [r.id for r in steps]
        assert "transcribe" not in ids, f"{sel} pulled in a second producer"
        assert ids.count("diarize") == 1


def test_no_composed_pipeline_ever_holds_two_steps_with_one_save_as():
    """The invariant behind 24, pinned wider than the one pair: two steps
    writing the same binding means the second silently wins downstream."""
    from itertools import permutations

    from taters.ui.compose import resolve_selection
    from taters.ui.recipes import user_facing

    ids = [r.id for r in user_facing("media")]
    for pair in permutations(ids, 2):
        try:
            steps = resolve_selection(list(pair))
        except Exception:
            continue        # some pairs really do conflict; not this test's job
        names = [r.save_as for r in steps]
        assert len(names) == len(set(names)), (pair, names)


def test_two_undecided_producers_are_a_loud_error_not_a_silent_clobber():
    from taters.ui.compose import ComposeError, resolve_selection

    with pytest.raises(ComposeError) as err:
        resolve_selection(["transcribe", "diarize"])
    assert "providers=" in str(err.value)


# ---------------------------------------------------------------------------
# A model's own private extraction
# ---------------------------------------------------------------------------

def _plan(stem="pos", target=None, digest="abcd1234", slug="age_blogs",
          **instrument):
    """A synthetic feature plan, so these tests need no fitted model."""
    from taters.helpers.model_spec import FeaturePlan, TablePlan
    from taters.ui.recipes import by_id

    target = target or by_id("parts_of_speech").target
    # we use the pipeline's own values, so a plan built with no overrides
    # describes exactly what this run would measure -- that's what lets us test
    # the reuse-versus-clone decision in both directions.
    from taters.ui.compose import _instrument_defaults

    settings = dict(_instrument_defaults(by_id("parts_of_speech")))
    settings.update(instrument)
    return FeaturePlan(
        model=f"{slug} [ridge]", slug=slug,
        tables=(TablePlan(stem=stem, target=target, instrument=settings,
                          assets={}, digest=digest, grain={}),))


def _csv_kwargs(**extra):
    kwargs = dict(source="csv", input_path="study.csv", text_cols=["text"],
                  id_cols=["pid"], name="t")
    kwargs.update(extra)
    return kwargs


def test_a_model_needing_other_settings_gets_its_own_extraction():
    """
    The whole point. The user measures parts of speech their way; the model
    was fitted on parts of speech measured another way. Both must happen, and
    the model's must not overwrite or contaminate the user's.
    """
    preset = compose(["parts_of_speech", "score_with_model"],
                     model_plans=[_plan(rounding=6)], **_csv_kwargs())
    pos = [s for s in preset["steps"]
           if s["call"].endswith("analyze_parts_of_speech")]
    assert len(pos) == 2, "the model's extraction did not happen"

    mine = next(s for s in pos if not s.get("for_model"))
    theirs = next(s for s in pos if s.get("for_model"))
    # separate artifacts: the runner keys results by save_as, so sharing one
    # would mean each clobbering the other.
    assert mine["save_as"] != theirs["save_as"]
    from taters.ui.recipes import by_id
    parent = by_id("parts_of_speech").save_as
    assert theirs["save_as"] == f"{parent}__for_age_blogs"
    # separate files, with the settings digest in the directory so a re-fit
    # can't short-circuit onto the previous private table.
    assert mine["with"]["out_features_csv"] != theirs["with"]["out_features_csv"]
    assert "model_inputs/age_blogs/abcd1234/pos.csv" in \
        theirs["with"]["out_features_csv"]
    # the model's settings, not the user's.
    assert theirs["with"]["rounding"] == 6
    # and the scoring step reads the clone, not the user's table. this used to
    # get written into `overrides` after the steps had already been rendered,
    # so the clone was measured and then ignored, the gate refused the user's
    # table, and this test stayed green because it never looked.
    scoring = next(s for s in preset["steps"]
                   if s["call"].endswith("score_with_model"))
    assert scoring["with"]["feature_csvs"] == [
        "{{" + theirs["save_as"] + "}}"]
    assert_valid_preset(preset)


def test_two_models_sharing_one_table_do_not_name_it_twice():
    """A template listed twice would join the same table to itself."""
    preset = compose(["parts_of_speech", "score_with_model"],
                     model_plans=[_plan(rounding=6), _plan(rounding=6)],
                     **_csv_kwargs())
    scoring = next(s for s in preset["steps"]
                   if s["call"].endswith("score_with_model"))
    tables = scoring["with"]["feature_csvs"]
    assert len(tables) == len(set(tables)) == 1, tables


def test_the_models_extraction_carries_literals_not_variables():
    """
    A clone cannot read pipeline variables, and this is not tidiness.
    `lemmatize` is deliberately ONE shared variable across three steps and
    cannot hold two values at once -- so a clone that read variables would
    silently measure with the *user's* value, which is the bug it exists to
    prevent.
    """
    preset = compose(["parts_of_speech", "score_with_model"],
                     model_plans=[_plan(rounding=6, engine="nltk")],
                     **_csv_kwargs())
    theirs = next(s for s in preset["steps"] if s.get("for_model"))
    for key in ("rounding", "engine", "tagset", "relative_freq"):
        value = theirs["with"][key]
        assert not (isinstance(value, str) and value.startswith("{{var:")), \
            f"{key} is a variable, so the model's value can be overridden"


def test_matching_settings_share_one_extraction():
    """R6: no needless double work. If the user is already measuring it
    exactly the model's way, one extraction serves both."""
    preset = compose(["parts_of_speech", "score_with_model"],
                     model_plans=[_plan(rounding=4)], **_csv_kwargs())
    pos = [s for s in preset["steps"]
           if s["call"].endswith("analyze_parts_of_speech")]
    assert len(pos) == 1, "the same settings were extracted twice"
    assert not pos[0].get("for_model")
    scoring = next(s for s in preset["steps"]
                   if s["call"] == "potato.score_with_model")
    from taters.ui.recipes import by_id
    assert scoring["with"]["feature_csvs"] == \
        ["{{" + by_id("parts_of_speech").save_as + "}}"]


def test_a_changed_answer_is_not_mistaken_for_the_default():
    """The reuse decision resolves `{{var:x}}` against the *answered* value.
    Comparing against the catalog default would reuse the user's table
    whenever a default happened to match the model -- while the user had
    actually changed it."""
    preset = compose(["ngram_frequencies", "parts_of_speech",
                      "score_with_model"],
                     model_plans=[_plan(rounding=4)],
                     var_values={"engine": "stanza"}, **_csv_kwargs())
    pos = [s for s in preset["steps"]
           if s["call"].endswith("analyze_parts_of_speech")]
    # the model wants engine='nltk'; the user answered 'stanza'. so, two runs.
    assert len(pos) == 2
    assert next(s for s in pos if s.get("for_model"))["with"]["engine"] == "nltk"


def test_table_names_are_the_stems_the_analyses_will_see():
    """
    `pca` and `unverified_ok` name feature sets by the stem of the file each
    step writes -- `doc_term_matrix_count`, not "Document-term matrix". The
    wizard offers those names instead of asking for them to be typed, so
    they have to be worked out the way the run will: the matrix's weighting
    is in its filename, and a renamed output is renamed here too.
    """
    from taters.ui.compose import table_names

    steps = [by_id("doc_term_matrix"), by_id("dictionaries"), by_id("readability")]
    plain = table_names(steps, "csv")
    assert [n for n, _l in plain] == ["doc_term_matrix_count", "dictionary",
                                      "readability"]
    assert plain[0][1] == by_id("doc_term_matrix").label

    changed = table_names(
        steps, "csv", var_values={"weighting": "tfidf"},
        overrides={"readability": {"out_features_csv": "out/mine.csv"}})
    assert [n for n, _l in changed] == ["doc_term_matrix_tfidf", "dictionary",
                                        "mine"]


def test_the_models_extraction_is_invisible_to_everything_else():
    """
    R4, and by arithmetic rather than vigilance: every screen reads the
    resolved *Recipe* list, and a clone is a plain step dict that is not in
    it. The one that matters most is `feature_tables()` -- a clone counted
    there would join the model's private table into the user's own analysis
    table, so their results would silently contain a second set of
    parts-of-speech columns measured to somebody else's settings.
    """
    from taters.ui.compose import feature_tables, resolve_selection

    selected = ["parts_of_speech", "score_with_model", "stats_correlations"]
    preset = compose(selected, model_plans=[_plan(rounding=6)],
                     var_values={"stats_outcome_cols": ["age"]},
                     **_csv_kwargs())
    steps = resolve_selection(selected, source="csv")
    names = [name for _r, name in feature_tables(steps, "csv")]
    assert not any("for" in n and "age_blogs" in n for n in names)

    assemble = next(s for s in preset["steps"]
                    if s["call"] == "potato.stats.assemble_analysis_table")
    joined = assemble["with"]["feature_csvs"]
    assert not any("for_age_blogs" in tpl for tpl in joined), \
        "a model's private table leaked into the user's analysis table"
    assert_valid_preset(preset)


def test_a_step_that_cannot_be_measured_twice_is_refused():
    """An item-scoped chain would mean re-measuring audio behind the user's
    back through a fan-out one global call cannot express. Refused with the
    settings printed, so the researcher can extract it explicitly."""
    from taters.ui.recipes import by_id

    plan = _plan(stem="acoustics_summary",
                 target=by_id("acoustics").target, rounding=6)
    with pytest.raises(ComposeError) as e:
        compose(["parts_of_speech", "score_with_model"], model_plans=[plan],
                **_csv_kwargs())
    assert "cannot be measured a second time" in str(e.value) \
        or "does not produce" in str(e.value)


def test_a_model_that_omits_a_setting_is_refused_not_guessed():
    """Filling a gap with this run's value is how a wrong answer gets
    produced quietly. Recording effective settings is what makes the replay
    constructible, so a gap means the model is too old to replay."""
    from taters.helpers.model_spec import FeaturePlan, TablePlan
    from taters.ui.recipes import by_id

    thin = FeaturePlan(
        model="old [ridge]", slug="old",
        tables=(TablePlan(stem="pos", target=by_id("parts_of_speech").target,
                          instrument={"rounding": 6},   # and nothing else
                          assets={}, digest="dddd", grain={}),))
    with pytest.raises(ComposeError) as e:
        compose(["parts_of_speech", "score_with_model"], model_plans=[thin],
                **_csv_kwargs())
    assert "does not record" in str(e.value)


def test_every_text_table_composes_its_id_the_same_way():
    """
    The join is on `text_id`, so every table a run writes has to compose it
    from the same columns. The sentence-embedding step used to keep a
    synthetic `row_<n>` while the metadata gather used the id column the
    user picked, and a real run (embeddings + a classifier over 938 texts)
    died at the very last step: "no key value appears in every input".
    """
    preset = compose(["sentence_embeddings", "readability", "stats_classify_fit"],
                     source="csv", input_path="study.csv", text_cols=["text"],
                     id_cols=["pid"], name="t",
                     var_values={"stats_class_cols": ["condition"]})
    embeddings = _with_of(preset, "extract_sentence_embeddings")
    readability = _with_of(preset, "analyze_readability")
    metadata = next(s["with"] for s in preset["steps"]
                    if s["call"] == "potato.helpers.csv_to_analysis_ready_csv"
                    and s["with"].get("text_cols") == [])
    assert embeddings["id_cols"] == readability["id_cols"] \
        == metadata["id_cols"] == ["pid"]
    assert "group_by" not in embeddings and "group_by" not in metadata


def test_the_two_gathers_really_do_agree_on_text_id(tmp_path):
    """The binding is only wiring; this runs the gather both ways -- as the
    embedding step is bound and as the metadata step is bound -- on a real
    spreadsheet and compares the ids they wrote."""
    import csv

    from taters.helpers.text_gather import csv_to_analysis_ready_csv
    from taters.ui.recipes import text_binding

    sheet = tmp_path / "s.csv"
    sheet.write_text("pid,condition,text\nR_1,a,hello there\nR_2,b,more words\n",
                     encoding="utf-8")
    text = text_binding("csv", text_cols=["text"], id_cols=["pid"],
                        pass_through=True)
    texts = csv_to_analysis_ready_csv(
        csv_path=sheet, out_csv=tmp_path / "texts.csv",
        text_cols=text["text_cols"], id_cols=text.get("id_cols"))
    meta = csv_to_analysis_ready_csv(
        csv_path=sheet, out_csv=tmp_path / "meta.csv", text_cols=[],
        id_cols=["pid"], carry_cols=["condition"])

    def ids(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as fh:
            return [r["text_id"] for r in csv.DictReader(fh)]
    assert ids(texts) == ids(meta) == ["R_1", "R_2"]


def test_both_prediction_steps_read_one_cross_validation_answer():
    """Fold count and stratification are pipeline-wide: the ridge and the
    classifier used to take them as two unrelated per-step settings."""
    preset = compose(["readability", "stats_ridge_fit", "stats_classify_fit"],
                     source="csv", input_path="s.csv", text_cols=["text"],
                     id_cols=["pid"], name="t",
                     var_values={"stats_outcome_cols": ["openness"],
                                 "stats_class_cols": ["condition"],
                                 "stats_n_folds": 10, "stats_stratify": False})
    ridge = _with_of(preset, "fit_ridge_csv")
    classifier = _with_of(preset, "fit_classifier_csv")
    assert ridge["n_folds"] == classifier["n_folds"] == "{{var:stats_n_folds}}"
    assert ridge["stratify"] == classifier["stratify"] == "{{var:stats_stratify}}"
    assert preset["vars"]["stats_n_folds"] == 10
    assert preset["vars"]["stats_stratify"] is False


def test_every_run_with_a_feature_table_describes_it_and_no_other_run_does():
    """
    The descriptives step is added by one rule here rather than an
    `auto_with` on thirteen recipes: whenever a feature table is in the run
    it comes along, fed every table the analyses would see, and a run that
    writes no table (a transcript alone) does not get it.
    """
    from taters.ui.compose import feature_tables

    steps = resolve_selection(["readability", "dictionaries"], source="csv")
    assert ids_of(steps)[-1] == "describe_features"
    preset = compose(["readability", "dictionaries"], source="csv",
                     input_path="s.csv", text_cols=["text"], name="x")
    describe = next(s for s in preset["steps"]
                    if s["call"] == "potato.stats.describe_features")
    wanted = ["{{" + r.save_as + "}}" for r, _ in feature_tables(steps, "csv")]
    assert describe["with"]["feature_csvs"] == wanted and len(wanted) == 2
    assert describe["with"]["out_dir"] == "{{var:descriptives_dir}}"
    assert preset["vars"]["descriptives_dir"] == "stats_descriptives"

    assert "describe_features" not in ids_of(
        resolve_selection(["transcribe"], providers={"transcript_csv": "transcribe"}))


def test_two_models_with_one_slug_are_refused_not_clobbered():
    """Two plans that slug alike but are different models (or spell one
    name two ways) produced two private steps under one save_as, the
    second silently replacing the first's table. A plain duplicate of one
    model collapses to one step; a clash is refused by name."""
    from taters.ui.compose import ComposeError, compose

    # rounding=2 differs from the pipeline's default, so we measure a private
    # table (identical settings would reuse the user's own step instead)
    same = [_plan(slug="age_blogs", digest="abcd1234", rounding=2),
            _plan(slug="age_blogs", digest="abcd1234", rounding=2)]
    preset = compose(["parts_of_speech", "score_with_model"], model_plans=same,
                     **_csv_kwargs())
    private = [s for s in preset["steps"] if s.get("for_model")]
    assert len(private) == 1
    clash = [_plan(slug="age_blogs", digest="abcd1234", rounding=2),
             _plan(slug="age_blogs", digest="ef567890", rounding=3)]
    with pytest.raises(ComposeError, match="both called"):
        compose(["parts_of_speech", "score_with_model"], model_plans=clash,
                **_csv_kwargs())


# ---------------------------------------------------------------------------
# Measure each row, then average the numbers
# ---------------------------------------------------------------------------

AVERAGE = "potato.helpers.average_feature_table"


def _chained(levels, *, group_by=(), picks=("readability", "stats_correlations"),
             source="csv", text_mode="concat"):
    return compose(list(picks), source=source, text_cols=["text"],
                   group_by=list(group_by), feature_levels=levels,
                   text_mode=text_mode,
                   var_values={"input_csv": "turns.csv"}, name="chain")


def _averaging(preset) -> list:
    return [s for s in preset["steps"] if s["call"] == AVERAGE]


def _by_id(preset, save_as):
    return next(s for s in preset["steps"] if s.get("save_as") == save_as)


def test_a_run_that_averages_nothing_is_exactly_what_it_always_was():
    """The default has to be untouched: every saved pipeline, every shipped
    preset and every other test in this file runs through this path."""
    plain = _chained([])

    assert _averaging(plain) == []
    assert not [s for s in plain["steps"] if s.get("save_as") == "row_group_keys"]


def test_averaging_adds_one_step_per_table_per_level(tmp_path):
    """
    One step per table rather than one step over a folder, because a feature
    table's provenance record sits beside it and the decorator writes one
    record per file it returns. A step that wrote a folder full of tables
    would leave every one of them unrecorded, and both the assemble step and
    the model gate read those records.
    """
    preset = _chained([("conv", "speaker"), ("conv",)],
                      picks=("readability", "dictionaries",
                             "stats_correlations"))
    steps = _averaging(preset)

    assert len(steps) == 4, "two tables at two levels"
    assert_valid_preset(preset)


def test_each_level_reads_what_the_level_before_it_wrote(tmp_path):
    """
    Averaging turns to speakers and then speakers to conversations is not
    the same arithmetic as averaging turns straight to conversations, so the
    steps have to chain rather than each read the raw table.
    """
    preset = _chained([("conv", "speaker"), ("conv",)])
    first, second = _averaging(preset)

    assert first["with"]["in_csv"] == "{{readability_features}}"
    assert second["with"]["in_csv"] == "{{" + first["save_as"] + "}}"
    assert second["with"]["group_by"] == ["conv"]


def test_only_the_first_level_is_told_which_row_belongs_to_which_group():
    """
    A feature table carries `text_id` and numbers, and nothing on it says
    who said it -- so the first average needs a key map. After that the
    grouping columns are in the table the previous level wrote.
    """
    preset = _chained([("conv", "speaker"), ("conv",)])
    first, second = _averaging(preset)

    assert first["with"]["keys_csv"] == "{{row_group_keys}}"
    assert "keys_csv" not in second["with"]


def test_the_key_map_is_keyed_the_way_the_text_was_gathered():
    """
    It has to compose `text_id` exactly as the run's feature tables did, or
    it can place none of their rows. Rows measured one at a time are keyed
    on the id columns; rows whose text was joined first are keyed on the
    joining columns.
    """
    per_row = compose(["readability", "stats_correlations"], source="csv",
                      text_cols=["text"], id_cols=["turn_id"], group_by=[],
                      feature_levels=[("conv",)],
                      var_values={"input_csv": "t.csv"}, name="a")
    joined = _chained([("conv",)], group_by=["conv", "speaker"])

    assert _by_id(per_row, "row_group_keys")["with"]["id_cols"] == ["turn_id"]
    assert _by_id(joined, "row_group_keys")["with"]["group_by"] == \
        ["conv", "speaker"]
    assert _by_id(joined, "row_group_keys")["with"]["text_cols"] == [], \
        "the key map is not supposed to carry any text"


def test_the_outcomes_arrive_at_the_grain_the_statistics_will_see():
    """
    The text steps still gather at the joining keys, but the metadata has to
    be one row per *analyzed* row or the join matches nothing. This is the
    one place where the two grains in a run are genuinely different.
    """
    joined = _chained([("conv",)], group_by=["conv", "speaker"])

    assert _by_id(joined, "stats_metadata")["with"]["group_by"] == ["conv"]
    gather = next(s for s in joined["steps"]
                  if s.get("save_as") == "readability_features")
    assert gather["with"]["group_by"] == ["conv", "speaker"], \
        "the text stopped being joined at the level the user asked for"


def test_the_statistics_and_the_descriptives_both_read_the_averaged_tables():
    """
    A report whose descriptives say "4000 rows" about a model fitted on 40
    conversations is worse than no descriptives, so both readers move to the
    grain the analyses use -- and the averaging has to be spliced in ahead
    of the descriptives, which sit in the extract stage.
    """
    preset = _chained([("conv", "speaker"), ("conv",)])
    last = _averaging(preset)[-1]["save_as"]
    calls = [s["call"] for s in preset["steps"]]

    for reader in ("potato.stats.assemble_analysis_table",
                   "potato.stats.describe_features"):
        step = next(s for s in preset["steps"] if s["call"] == reader)
        assert step["with"]["feature_csvs"] == ["{{" + last + "}}"], reader
        assert calls.index(AVERAGE) < calls.index(reader), \
            f"{reader} reads a table written after it"


def test_an_averaged_table_is_named_for_its_measure_and_filed_by_its_grain():
    """
    `by-conv/readability.csv` rather than `readability_avg2.csv`: the folder
    says what one row is and the stem stays the measure's own name, which is
    what the analyses call a feature set and what a saved model compares its
    tables by.
    """
    preset = _chained([("conv", "speaker"), ("conv",)])
    outs = [s["with"]["out_csv"] for s in _averaging(preset)]

    assert outs == ["{{var:features_dir}}/by-conv-speaker/readability.csv",
                    "{{var:features_dir}}/by-conv/readability.csv"]


def test_the_joining_levels_filename_suffix_is_not_carried_onto_the_average():
    """
    A non-default level suffixes the raw table's filename to say what one
    row of *it* was. Left on the averaged copy, the analyses would name a
    feature set `readability_by-group` inside a folder called `by-conv`,
    which reads as two answers to one question.
    """
    preset = _chained([("conv",)], group_by=["conv", "speaker"])

    assert _averaging(preset)[0]["with"]["out_csv"] == \
        "{{var:features_dir}}/by-conv/readability.csv"


def test_a_column_that_splits_a_row_on_purpose_is_kept_through_the_average():
    """
    Measuring each text column separately means a row per text column on
    purpose, and the analysis table joins on `text_id` *and* that column.
    Averaging it away would leave duplicate keys, which assemble refuses.
    """
    separate = compose(["readability", "stats_correlations"], source="csv",
                       text_cols=["q1", "q2"], text_mode="separate",
                       group_by=[], feature_levels=[("conv",)],
                       var_values={"input_csv": "t.csv"}, name="s")

    assert _averaging(separate)[0]["with"]["split_col"] == "source_col"
    assert _averaging(_chained([("conv",)]))[0]["with"]["split_col"] == ""


def test_averaging_is_ignored_where_there_is_no_spreadsheet_to_average_by():
    """The grouping columns are a spreadsheet's own. A folder of documents
    has none, and the media path already averages per speaker its own way."""
    docs = compose(["readability"], source="txt_dir",
                   feature_levels=[("conv",)], name="d")

    assert _averaging(docs) == []


def test_an_early_gather_carries_whatever_a_later_average_will_group_by():
    """
    The gather writes the analysis-ready table the analyzers read, so a
    column a later average groups on has to survive it. Left out, the
    averaging step would have nothing to group on and the run would die
    after every feature had already been measured.

    The wizard only offers the early gather when rows are being joined, and
    then the averaging columns are a subset of the joining ones -- but
    `compose` is a public entry point and this combination is legal through
    it, so it is wired rather than assumed.
    """
    preset = compose(["readability", "stats_correlations"], source="csv",
                     text_cols=["text"], group_by=[],
                     row_filters=[["arm", "in", ["A"]]],
                     feature_levels=[("conv", "speaker")],
                     var_values={"input_csv": "t.csv"}, name="api")
    gather = next(s for s in preset["steps"]
                  if s.get("save_as") == "gathered_texts")

    assert set(gather["with"]["carry_cols"]) >= {"conv", "speaker"}
    assert_valid_preset(preset)
