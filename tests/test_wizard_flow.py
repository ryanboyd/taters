"""
Tests for the setup wizard, driven end to end without a terminal (see
wizard_helpers for how). If a test fails with 'ran out of answers', the
message names the question that went unanswered -- insert an answer at
that position.

This file: the run from start to finish -- the happy path, backing out, input
validation, transcription, preflight, options gate, running, the entry point,
saving over an existing pipeline, dropping a provider at preflight.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from preset_checks import check_preset

from taters.ui import wizard as wiz
from taters.ui.prompts import (Cancelled, ScriptedPrompter)
from taters.ui.recipes import by_id
from wizard_helpers import (clean_machine,  # noqa: F401  (autouse fixture)
                            browse_to, run, EscapingPrompter, _tuning,
                            _escaping_tuning, _engine_tuning, _asked)


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------

def test_a_minimal_run_writes_a_preset(media, tmp_path):
    result, _ = run([
        "audio",                    # what kind of data
        *browse_to(media),                 # where is it
        ["transcribe"],             # what do you want out of it
        False,                      # change any options? -> no
        "Lecture pilot",            # name
        "save",                     # keep it, run it later
    ], cwd=tmp_path)

    # each pipeline owns a folder named after it, holding the preset and
    # everything the run produces
    assert result.preset_path == tmp_path / "lecture_pilot" / "lecture_pilot.yaml"
    assert result.folder == tmp_path / "lecture_pilot"
    assert result.preset_path.exists()
    assert result.ran is False
    assert result.root_dir == media.resolve()
    assert len(result.inputs) == 2


def test_the_written_file_is_a_valid_preset(media, tmp_path):
    """
    The wizard's output goes through the same validator as the presets that
    ship with Taters — reading it back off disk, so YAML round-tripping is
    covered too.
    """
    result, _ = run(["audio", *browse_to(media), ["transcribe", "readability"],
                     "speaker", False, "Round trip", "save"], cwd=tmp_path)
    on_disk = yaml.safe_load(result.preset_path.read_text(encoding="utf-8"))
    problems = check_preset(on_disk)
    assert not problems, "\n".join(problems)


def test_the_preset_lands_where_the_runner_looks_for_it(media, tmp_path, monkeypatch):
    """
    `./pipelines` is the second entry in `run_pipeline._get_preset_dirs()`, so a
    file written there is immediately usable as `--preset <id>`. That is the
    whole reason the wizard writes a file instead of just running.
    """
    from taters.pipelines.run_pipeline import resolve_preset_path

    result, _ = run(["audio", *browse_to(media), ["transcribe"],
                     False, "Findable", "save"], cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    assert resolve_preset_path("findable").resolve() == result.preset_path.resolve()


def test_prerequisites_are_added_and_explained(media, tmp_path):
    """A user who ticks one box gets four steps, and is told which were added."""
    result, prompter = run(["audio", *browse_to(media), ["readability"],
                            "transcribe",       # how should we make a transcript?
                            "speaker", False, "Why", "save"], cwd=tmp_path)
    assert len(result.preset["steps"]) == 5      # the fifth describes the table
    assert "added for you" in prompter.text_output
    assert "you picked" in prompter.text_output


# ---------------------------------------------------------------------------
# backing out
# ---------------------------------------------------------------------------

def test_discarding_writes_nothing(media, tmp_path):
    """
    The way out is still there -- it is just called what it does. This used to
    be "Save this pipeline?" answered no, which threw the session away while
    sounding like it only declined to write a file.
    """
    with pytest.raises(Cancelled):
        run(["audio", *browse_to(media), ["transcribe"], False, "Nope", "discard"],
            cwd=tmp_path)
    assert not (tmp_path / "pipelines").exists()


def test_the_last_question_offers_the_three_things_you_might_want(media, tmp_path):
    """
    One question, not two. Naming a pipeline is already a decision to keep it,
    so "Save this pipeline?" read as a formality -- and saving then running is
    one act, which was costing two answers.
    """
    _, prompter = run(["audio", *browse_to(media), ["transcribe"], False,
                       "Three ways", "save"], cwd=tmp_path)

    last = [choices for q, choices in prompter.offered if q == "What next?"]
    assert last, "the closing question is not a choice"
    assert [c.value for c in last[-1]] == ["run", "save", "discard"]


def test_nothing_asks_whether_to_save_after_you_named_it(media, tmp_path):
    _, prompter = run(["audio", *browse_to(media), ["transcribe"], False,
                       "No double ask", "save"], cwd=tmp_path)

    asked = " ".join(q for _kind, q in prompter.asked).lower()
    assert "save this pipeline?" not in asked
    assert "run it now?" not in asked


def test_declining_to_run_still_leaves_the_preset(media, tmp_path):
    result, prompter = run(["audio", *browse_to(media), ["transcribe"],
                            False, "Saved only", "save"], cwd=tmp_path)
    assert result.preset_path.exists()
    assert result.ran is False
    assert result.manifest is None
    assert "Run it whenever you are ready" in prompter.text_output


def test_the_saved_preset_comes_with_the_command_to_rerun_it(media, tmp_path):
    _, prompter = run(["audio", *browse_to(media), ["transcribe"],
                       False, "Rerun me", "save"], cwd=tmp_path)
    assert "--preset rerun_me" in prompter.text_output
    assert "run_pipeline" in prompter.text_output


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def test_a_folder_that_does_not_exist_is_re_asked(media, tmp_path):
    """Typos in a path are the most common thing to get wrong; do not proceed."""
    result, prompter = run([
        "audio",                              # what kind
        *browse_to(tmp_path / "nope"),        # typed a path that isn't there
        *browse_to(media),                    # ...browser stayed open, try again
        ["transcribe"], False, "Second try", "save",
    ], cwd=tmp_path)
    assert result.preset_path.exists()
    assert "not something I can see" in prompter.text_output


def test_a_folder_with_no_matching_files_is_re_asked(tmp_path, media):
    """
    Discovery runs before anything else is asked. Finding out you matched zero
    files after picking options and waiting for a model download is a miserable
    way to learn you chose the wrong file type.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    result, prompter = run([
        "video", *browse_to(empty),        # nothing there
        True,                       # try again? -> yes, from the top
        "audio", *browse_to(media),        # so the *type* can be corrected too
        ["transcribe"], False, "Found it", "save",
    ], cwd=tmp_path)
    assert "No video files" in prompter.text_output
    assert result.preset_path.exists()


def test_giving_up_on_an_empty_folder_cancels(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Cancelled):
        run(["audio", *browse_to(empty), False], cwd=tmp_path)


def test_ticking_nothing_is_re_asked(media, tmp_path):
    result, prompter = run([
        "audio", *browse_to(media),
        [],                         # nothing ticked
        ["transcribe"],             # try again
        False, "Eventually", "save",
    ], cwd=tmp_path)
    assert "at least one" in prompter.text_output
    assert result.preset_path.exists()


# ---------------------------------------------------------------------------
# choosing how to transcribe
# ---------------------------------------------------------------------------

def test_a_transcript_producer_is_asked_for_when_none_was_ticked(media, tmp_path):
    """
    Acoustic measures need a transcript but do not say how to make one. The
    user has to choose, because the two options differ enormously in what they
    cost to install.
    """
    result, prompter = run([
        "audio", *browse_to(media),
        ["acoustics"],
        "diarize",                  # how should we produce a transcript?
        False, "Diarized", "save",
    ], cwd=tmp_path)
    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.audio.diarize_with_thirdparty" in calls
    assert any("needs a transcript" in line for line in prompter.output)


def test_ticking_both_producers_forces_a_choice(media, tmp_path):
    """
    Both write to `save_as: diar`. Running both would have the second silently
    clobber the first for every step downstream.
    """
    result, prompter = run([
        "audio", *browse_to(media),
        ["transcribe", "diarize"],
        "transcribe",               # pick one
        False, "Just one", "save",
    ], cwd=tmp_path)
    calls = [s["call"] for s in result.preset["steps"]]
    assert "potato.audio.diarize_with_thirdparty" not in calls
    assert "more than one way" in prompter.text_output


def test_no_choice_is_asked_when_the_user_already_picked_one(media, tmp_path):
    _, prompter = run(["audio", *browse_to(media), ["transcribe", "readability"],
                       "speaker", False, "No question", "save"], cwd=tmp_path)
    assert not any("How should Taters produce" in q for _, q in prompter.asked)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def test_a_transformer_step_asks_for_its_encoder_at_the_features_stage(tmp_path, monkeypatch):
    """
    "Transformer embeddings (any encoder)" took its default encoder silently
    and offered a change only on the options screen, where nobody looked --
    so people embedded with a model they never chose. The picker is asked
    right after the checklist now; the answer is the pipeline's variable.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = tmp_path / "s.csv"
    path.write_text("id,text\n1,hello there\n2,bye now\n", encoding="utf-8")
    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], True, ["id"],
        ["transformer_embeddings"], "row",
        "distilroberta-base",           # the encoder question, at the features stage
        ":done", "Emb", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=False)

    questions = [q for _k, q in p.asked]
    assert "Which encoder?" in questions
    # at the features stage: before the settings, naming and save questions
    assert questions.index("Which encoder?") < len(questions) - 3
    assert result.preset["meta"]["variables"]["encoder"]["default"] == "distilroberta-base"


def test_a_missing_extra_is_named_with_its_install_command(media, tmp_path, monkeypatch):
    monkeypatch.setattr(wiz, "missing_extras",
                        lambda r: list(r.extras))
    result, prompter = run([
        "audio", *browse_to(media), ["acoustics"], "transcribe",
        "keep",                     # keep 'Acoustic measures' anyway
        False, "Optimistic", "save",
    ], cwd=tmp_path)

    # in the question's description, where it sits next to the choice it's about,
    # and with its brackets intact. rich reads `[vocalacoustics]` as a style tag,
    # so unescaped this printed as `pip install "taters"`: a command that looks
    # right, runs cleanly, and installs nothing that was actually missing
    helps = " ".join(c.help for _q, choices in prompter.offered for c in choices)
    assert 'pip install "taters[vocalacoustics]"' in helps
    assert any("analyze_vocal_acoustics" in s["call"] for s in result.preset["steps"])


def test_an_extra_this_python_cannot_have_is_not_offered_as_a_pip_command(monkeypatch):
    """
    On Python 3.14 our metadata does not ask for gensim (there are no wheels),
    so `pip install "taters[vectors]"` succeeds and installs nothing. Printing
    that command sends someone round a loop; the honest hint is the Python
    version.
    """
    from taters.ui import recipes
    from taters.ui.prompts import ScriptedPrompter
    from taters.ui.tasks import gpu

    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "gensim")

    prompter = ScriptedPrompter(["keep"])
    dropped = wiz.preflight(prompter, [recipes.by_id("word_vectors_train")])

    assert dropped == []
    helps = " ".join(c.help for _q, choices in prompter.offered for c in choices)
    assert "Not available for Python" in helps
    assert "gensim" in helps
    assert "3.13 or older" in helps
    assert "pip install" not in helps, "a command that installs nothing is not a fix"


def test_a_step_can_be_dropped_when_its_extra_is_missing(media, tmp_path, monkeypatch):
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    result, _ = run([
        "audio", *browse_to(media), ["acoustics", "readability"], "transcribe", "speaker",
        "drop",                     # drop 'Acoustic measures'
        # readability never gets asked about: it declares no extra, since
        # textstat ships in the base install
        False, "Trimmed", "save",
    ], cwd=tmp_path)
    calls = [s["call"] for s in result.preset["steps"]]
    assert not any("acoustics" in c for c in calls)
    assert any("readability" in c for c in calls)


def test_dropping_everything_cancels(media, tmp_path, monkeypatch):
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    with pytest.raises(Cancelled):
        run(["audio", *browse_to(media), ["acoustics"], "transcribe", "drop"],
            cwd=tmp_path)


def test_diarization_warns_about_its_github_dependencies(media, tmp_path, monkeypatch):
    """
    `pip install taters[diarization]` is not sufficient on its own — three of
    its dependencies only exist as git URLs, which cannot be declared on PyPI.
    Someone who follows the printed command and then hits an ImportError has
    every right to be annoyed.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    with pytest.raises(Cancelled):
        run(["audio", *browse_to(media), ["diarize"], "drop"], cwd=tmp_path)


def test_a_missing_ffmpeg_is_reported_before_anything_is_written(media, tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(wiz.shutil, "which", lambda name: None)
    with pytest.raises(Cancelled):
        run(["audio", *browse_to(media), ["transcribe"], False], cwd=tmp_path)
    assert not (tmp_path / "pipelines").exists()


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------

def test_changing_a_templated_option_edits_the_variable_not_the_step(media, tmp_path):
    """
    Most recipe templates read `{{var:whisper_model}}` rather than holding a
    literal. Editing one should move the *variable*, so the change lands in
    `vars:` where it is named, documented, and overridable with `--var` later —
    rather than being buried inside one step's `with:` block.
    """
    result, _ = run([
        "audio", *browse_to(media), ["transcribe"],
        True,                       # change any settings?
        "transcribe",               #   which part of the pipeline
        "whisper_model", "small.en",
        "language", "en",
        ":done",                    #   done with this step -> back to the list
        ":done",                    #   done with the whole screen
        "Tuned", "save",
    ], cwd=tmp_path)

    assert result.preset["vars"]["whisper_model"] == "small.en"
    assert result.preset["vars"]["language"] == "en"
    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("transcribe_with_whisper"))
    assert step["with"]["whisper_model"] == "{{var:whisper_model}}"


def test_changing_a_step_local_option_edits_the_step(media, tmp_path):
    """`beam_size` has no variable behind it, so it is written into the step."""
    result, _ = run([
        "audio", *browse_to(media), ["transcribe"],
        True, "transcribe",
        "beam_size", "1",           # -> greedy
        ":done", ":done",
        "Greedy", "save",
    ], cwd=tmp_path)

    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("transcribe_with_whisper"))
    assert step["with"]["beam_size"] == 1
    assert "beam_size" not in result.preset["vars"]


def test_unchanged_options_are_not_written(media, tmp_path):
    """Answering a question by pressing enter should leave no trace."""
    result, _ = run([
        "audio", *browse_to(media), ["transcribe"],
        True, "transcribe",
        "whisper_model", "base.en",         # re-typed, identical
        "beam_size", "5",
        ":done", ":done",
        "Untouched", "save",
    ], cwd=tmp_path)
    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("transcribe_with_whisper"))
    assert set(step["with"]) == set(by_id("transcribe").with_)


def test_saying_no_to_options_asks_nothing_further(media, tmp_path):
    _, prompter = run(["audio", *browse_to(media), ["transcribe"],
                       False, "Quick", "save"], cwd=tmp_path)
    assert not any(kind == "checkbox" and "which steps" in q.lower()
                   for kind, q in prompter.asked)


# --- what is offered, and what is withheld ---------------------------------

def test_wiring_parameters_are_never_offered_as_options():
    """
    `transcript_csv: "{{pick:diar.raw_files.csv}}"` is the wiring that makes the
    pipeline a pipeline. Letting someone edit it in an options screen would let
    them quietly disconnect their own run.
    """
    acoustics = by_id("acoustics")
    assert wiz.is_wired(acoustics, "transcript_csv")
    assert wiz.is_wired(acoustics, "wav_path")
    assert wiz.is_wired(by_id("convert_to_wav"), "input_path")


def test_variable_backed_parameters_are_offered():
    """`{{var:...}}` points at a setting that exists precisely to be changed."""
    assert not wiz.is_wired(by_id("transcribe"), "whisper_model")
    assert not wiz.is_wired(by_id("transcribe"), "overwrite_existing")


def test_alternate_input_parameters_are_withheld():
    """
    `txt_dir` and `analysis_csv` are mutually exclusive with the `csv_path` the
    pipeline wires in — setting one detaches the step from the run.
    """
    assert wiz.is_wired(by_id("readability"), "txt_dir")
    assert wiz.is_wired(by_id("readability"), "analysis_csv")


def test_internal_plumbing_is_withheld():
    assert wiz.is_wired(by_id("whisper_embeddings"), "run_in_subprocess")
    assert wiz.is_wired(by_id("transcribe"), "verbose")


@pytest.mark.parametrize("widget,text,expected", [
    ("int", "5", 5),
    ("float", "1.5", 1.5),
    ("list", "a, b ,c", ["a", "b", "c"]),
    ("text", "hello", "hello"),
    ("text", "", None),
    ("text", "none", None),
    ("choice", "24", 24),
])
def test_typed_input_is_parsed_back_to_its_type(widget, text, expected):
    """
    Everything comes back off a terminal as a string, but the preset is YAML —
    writing `"5"` where an int belongs is the kind of thing that only fails much
    later, inside the analysis code.
    """
    from taters.ui.introspect import ParamSpec
    param = ParamSpec(name="p", widget=widget, choices=[16, 24, 32])
    assert wiz._parse_as(text, param) == expected


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def test_saying_yes_runs_the_composed_preset(media, tmp_path, monkeypatch):
    """
    The wizard hands its preset straight to `run_preset` rather than
    reimplementing execution, so this checks the handover: the same preset, the
    folder the user named, and a progress callback attached.
    """
    seen = {}

    def fake_run_preset(preset, **kwargs):
        seen["preset"] = preset
        seen["kwargs"] = kwargs
        kwargs["on_event"]("step_start", index=1, total=1,
                           call="potato.audio.convert_to_wav", scope="item", items=2)
        kwargs["on_event"]("item_done", index=1, item=0,
                           input=str(media / "one.wav"), status="ok", error=None)
        kwargs["on_event"]("item_done", index=1, item=1,
                           input=str(media / "two.wav"), status="ok", error=None)
        return {"items": [{"status": "ok", "errors": []}] * 2, "errors": [], "globals": {}}

    monkeypatch.setattr("taters.pipelines.run_pipeline.run_preset", fake_run_preset)

    # "1" answers the concurrency question (this preset fans out over files), and
    # "menu" answers the finished screen: do something else rather than quit
    result, prompter = run(["audio", *browse_to(media), ["transcribe"],
                            False, "Go", "run", "1", "menu"], cwd=tmp_path)

    assert result.ran is True
    assert result.ok is True
    assert seen["preset"] is result.preset
    assert seen["kwargs"]["root_dir"] == media.resolve()
    assert seen["kwargs"]["file_type"] == "audio"
    assert seen["kwargs"]["verbose"] is False      # the wizard reports instead
    assert "2/2 done" in prompter.text_output
    assert "Everything succeeded" in prompter.text_output


def test_failures_are_surfaced_rather_than_swallowed(media, tmp_path, monkeypatch):
    def fake_run_preset(preset, **kwargs):
        kwargs["on_event"]("step_start", index=1, total=1, call="x", scope="item", items=1)
        kwargs["on_event"]("item_done", index=1, item=0, input="one.wav",
                           status="error", error="ffmpeg exploded")
        return {
            "items": [{"status": "error", "errors": ["ffmpeg exploded"]}],
            "errors": [],
            "globals": {},
        }

    monkeypatch.setattr("taters.pipelines.run_pipeline.run_preset", fake_run_preset)
    result, prompter = run(["audio", *browse_to(media), ["transcribe"],
                            False, "Broken", "run", "1", "menu"], cwd=tmp_path)

    assert result.ok is False
    # we collect the failure detail during the run and report it on the finished
    # screen. this is because writing into the area a live progress display owns
    # corrupts it, and a per-file error reads better next to the final count anyway
    assert "ffmpeg exploded" in prompter.text_output
    assert "Finished with problems" in prompter.text_output
    assert "1 file(s) failed" in prompter.text_output


def test_a_progress_callback_bug_cannot_take_down_a_run():
    """
    `run_preset` swallows exceptions from `on_event` on purpose. A display bug
    must not destroy a run that may be hours old.
    """
    from taters.pipelines.run_pipeline import run_preset

    def exploding(name, **payload):
        raise RuntimeError("boom")

    manifest = run_preset({"steps": [{"scope": "global", "call": "builtins:dict"}]},
                          on_event=exploding, verbose=False)
    assert manifest["errors"] == []


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def test_main_exits_cleanly_when_the_user_backs_out(monkeypatch, capsys):
    monkeypatch.setattr("taters.ui.hub.run_hub",
                        lambda prompter, cwd=None: (_ for _ in ()).throw(Cancelled()))
    monkeypatch.setattr("taters.ui.live.LivePrompter", lambda: object())
    assert wiz.main([]) == 130
    assert "Cancelled" in capsys.readouterr().out


def test_main_reports_a_failed_run_with_a_nonzero_exit(monkeypatch):
    """
    A script driving `taters` still has to learn that the work failed, even
    though the hub may have done several things before the user quit.
    """
    monkeypatch.setattr("taters.ui.live.LivePrompter", lambda: object())
    monkeypatch.setattr("taters.ui.hub.run_hub", lambda prompter, cwd=None: False)
    assert wiz.main([]) == 1


def test_main_returns_zero_on_success(monkeypatch):
    monkeypatch.setattr("taters.ui.live.LivePrompter", lambda: object())
    monkeypatch.setattr("taters.ui.hub.run_hub", lambda prompter, cwd=None: True)
    assert wiz.main([]) == 0


def test_a_mistyped_number_is_re_asked_rather_than_crashing(media, tmp_path):
    """
    Typing "abc" into a number field is an ordinary slip. This wizard exists for
    people who would not read a `ValueError` traceback as anything but a crash,
    so it has to say what it wanted and ask again.
    """
    result, prompter = run([
        "audio", *browse_to(media), ["transcribe"],
        True, "transcribe",
        "beam_size",
        "abc",                      # rejected
        "2",                        # accepted
        ":done", ":done",
        "Fat fingers", "save",
    ], cwd=tmp_path)

    assert "is not a whole number" in prompter.text_output
    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("transcribe_with_whisper"))
    assert step["with"]["beam_size"] == 2


def test_a_transcript_producer_is_asked_for_even_when_the_need_is_indirect(media,
                                                                          tmp_path):
    """
    Regression: ticking only "Readability scores" never mentions a transcript.
    Readability reads the *merged transcript table*, and it is the merge step
    that needs one — so the requirement is only visible after prerequisites are
    resolved.

    Before this was fixed, the wizard silently defaulted to single-speaker
    transcription. Someone analyzing group interviews would have got the wrong
    pipeline without ever being asked, and the two options differ by a
    multi-gigabyte install.
    """
    result, prompter = run([
        "audio", *browse_to(media),
        ["readability", "lexical_richness"],
        "diarize", "speaker",
        False, "Indirect", "save",
    ], cwd=tmp_path)

    assert any("How should Taters produce" in q for _, q in prompter.asked)
    assert any("diarize_with_thirdparty" in s["call"] for s in result.preset["steps"])


# ---------------------------------------------------------------------------
# saving over an existing pipeline (code review issue 2)
# ---------------------------------------------------------------------------


def _one_readability_run(media, *, title_answers):
    return ScriptedPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe", "speaker",
        False, *title_answers,
    ])


def test_saving_a_second_pipeline_with_the_same_name_asks_first(media, tmp_path):
    """
    From the code review: the second save silently replaced the first
    definition, and a later run then resumed against the first run's outputs --
    a results folder quietly mixing two different pipelines.
    """
    wiz.run_wizard(_one_readability_run(
        media, title_answers=["Twice", "save"]), cwd=tmp_path)

    p = _one_readability_run(media, title_answers=["Twice", "both", "save"])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("already exists" in q for _k, q in p.asked)
    assert result.preset["meta"]["title"] == "Twice (2)"
    assert result.preset_path.parent.name == result.preset_path.stem == "twice_2"
    # the first pipeline is untouched
    first = tmp_path / "twice" / "twice.yaml"
    assert first.exists()


def test_replacing_is_a_deliberate_choice_and_still_works(media, tmp_path):
    wiz.run_wizard(_one_readability_run(
        media, title_answers=["Twice", "save"]), cwd=tmp_path)

    p = _one_readability_run(media, title_answers=["Twice", "replace", "save"])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset_path == tmp_path / "twice" / "twice.yaml"
    assert not (tmp_path / "twice_2").exists()


def test_choosing_rename_returns_to_the_naming_question(media, tmp_path):
    wiz.run_wizard(_one_readability_run(
        media, title_answers=["Twice", "save"]), cwd=tmp_path)

    p = _one_readability_run(
        media, title_answers=["Twice", "rename", "Fresh name", "save"])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["meta"]["title"] == "Fresh name"


def test_a_data_folder_is_never_offered_for_replacement(media, tmp_path):
    """
    'Replace it' is only an honest offer for an actual pipeline. Naming the
    pipeline after an existing plain folder (someone's input data, say) must
    not offer to write into it.
    """
    (tmp_path / "interviews").mkdir()

    p = _one_readability_run(
        media, title_answers=["Interviews", "both", "save"])
    result = wiz.run_wizard(p, cwd=tmp_path)

    question, choices = next((q, c) for q, c in p.offered
                             if "already exists" in q)
    assert "not a pipeline" in question
    assert "replace" not in {c.value for c in choices}
    assert result.preset_path.parent.name == "interviews_2"
    assert not (tmp_path / "interviews" / "interviews.yaml").exists()


# ---------------------------------------------------------------------------
# dropping a provider at preflight (code review issue 4)
# ---------------------------------------------------------------------------


def test_a_provider_this_python_cannot_have_is_grayed_in_the_transcript_question(
        media, tmp_path, monkeypatch):
    """Diarization on a Python with no NeMo release is not a way to get a
    transcript, so the provider question grays it the way the checklist
    does -- rather than offering it and refusing two screens later."""
    from taters.ui.prompts import GoBack
    from taters.ui.tasks import gpu
    from wizard_helpers import EscapingPrompter

    monkeypatch.setattr(wiz, "missing_extras",
                        lambda r: ["diarization"] if "diarization" in r.extras else [])
    monkeypatch.setattr(gpu, "unavailable_here", lambda dist: dist == "nemo-toolkit")
    # Esc out from the provider question: what we're after is what it offered
    p = EscapingPrompter(["audio", *browse_to(media), ["readability"]] + ["__esc__"] * 6)
    with pytest.raises(GoBack):
        wiz.run_wizard(p, cwd=tmp_path)
    offered = {c.value: c for c in p.offered_choices("How should Taters produce")}
    assert offered["diarize"].disabled.startswith("not available for Python")
    assert "nemo-toolkit" in offered["diarize"].disabled
    assert not offered["transcribe"].disabled


def test_dropping_a_provider_at_preflight_actually_removes_it(
        media, tmp_path, monkeypatch):
    """
    From the code review: the drop filtered only `selected`, then the
    re-resolve reused the stale providers dict and put the very step the user
    declined straight back into the pipeline -- which then failed at run time
    on the missing install.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"],
        "diarize",                  # how to make a transcript
        "speaker",
        "drop",                     # diarization isn't installed, so leave it out
        False, "No nemo", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    calls = [s["call"] for s in result.preset["steps"]]
    assert not any("diarize" in c for c in calls), calls
    assert any(c.endswith("transcribe_with_whisper") for c in calls), calls
    # ...and we say the substitution out loud rather than slipping it in
    assert any("Transcript (one speaker)" in line for line in p.output)


def test_dropping_the_chosen_of_two_ticked_providers_does_not_crash(
        media, tmp_path, monkeypatch):
    """
    The other half: with both transcript producers ticked and the dropped one
    chosen at the collision question, the old code emptied the selection and
    let ComposeError climb uncaught -- a raw traceback instead of a menu.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["transcribe", "diarize"],
        "diarize",                  # they collide; keep diarize
        "drop",                     # ...which turns out not to be installed
        False, "Fallback", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    calls = [s["call"] for s in result.preset["steps"]]
    assert not any("diarize" in c for c in calls), calls
    assert any(c.endswith("transcribe_with_whisper") for c in calls), calls


def test_an_unresolvable_selection_backs_out_instead_of_crashing(
        media, tmp_path, monkeypatch):
    """
    Nothing above the wizard catches ComposeError, so letting it climb ended
    the session with a traceback. With the provider purge in place no known UI
    flow still reaches either raise site -- these pin the conversions so the
    next flow that does gets a clean back-out, not a crash.
    """
    from taters.ui.compose import ComposeError

    def explode(*a, **k):
        raise ComposeError("no way to satisfy this")

    # inside `_derive` this gets converted to Cancelled, so back to the main menu
    monkeypatch.setattr(wiz, "resolve_selection", explode)
    src = wiz.SourceSpec(source="media", path=media, file_type="audio",
                         inputs=list(media.iterdir()))
    p = ScriptedPrompter([])
    with pytest.raises(Cancelled):
        wiz._derive(p, src, ["readability"], {"transcript_csv": "transcribe"})
    assert any("no way to satisfy this" in line for line in p.output)


def test_an_unresolvable_selection_at_the_checklist_re_asks_it(
        media, tmp_path, monkeypatch):
    """The features stage resolves the selection too (for the level question);
    a failure there is a wrong answer to the checklist, so it re-asks."""
    from taters.ui.compose import ComposeError

    real = wiz.resolve_selection
    calls = []

    def once(*a, **k):
        if not calls:
            calls.append(1)
            raise ComposeError("not this combination")
        return real(*a, **k)

    monkeypatch.setattr(wiz, "resolve_selection", once)
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe",   # boom
        ["readability"], "transcribe", "speaker",                    # again
        False, "Second try", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("not this combination" in line for line in p.output)
    assert result.preset["meta"]["title"] == "Second try"


def test_a_header_with_spaces_is_offered_raw_and_works_end_to_end(tmp_path):
    """
    From the code review (issue 5): `_read_header` used to strip column names
    before offering them, but the gather matches DictReader fieldnames
    verbatim -- so the very common header `id, text` composed a pipeline that
    always died with "Missing columns: ['text']". The value the wizard stores
    has to be what the file actually says; only the label gets tidied.
    """
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    path = tmp_path / "survey.csv"
    path.write_text("id, text\n1, hello there\n2, more words\n",
                    encoding="utf-8")

    header = wiz._read_header(path)
    assert header == ["id", " text"]

    p = ScriptedPrompter([
        "csv", *browse_to(path), [" text"], True, ["id"],
        ["readability"], "row", [], False, "Spaced", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_readability"))
    assert step["with"]["text_cols"] == [" text"]
    assert step["with"]["id_cols"] == ["id"]
    # ...and the labels the user actually read were the tidy ones
    offered = next(cs for q, cs in p.offered if "hold the text" in q)
    assert [c.label for c in offered] == ["id", "text"]

    # the proof: the stored names reach the gather and match
    out = csv_to_analysis_ready_csv(
        csv_path=path, out_csv=tmp_path / "g.csv",
        text_cols=[" text"], id_cols=["id"])
    assert "hello there" in out.read_text(encoding="utf-8")


def test_the_finish_screen_puts_the_runs_verdict_on_the_quit(tmp_path):
    """The other half of code-review issue 17: the exception must carry ok=False
    out of a failed run, or the hub has nothing truthful to return."""
    from taters.ui.prompts import QuitRequested

    p = ScriptedPrompter(["quit"])
    with pytest.raises(QuitRequested) as caught:
        wiz.finish_screen(p, ok=False,
                          manifest={"errors": ["boom"], "items": []},
                          folder=tmp_path,
                          manifest_path=tmp_path / "run_manifest.json")
    assert caught.value.ok is False


def test_the_manifest_records_the_exact_reproduction_command(media, tmp_path):
    """
    Requested directly: a run built by clicking through menus should be
    re-runnable from a terminal by pasting one line from its manifest, with
    every chosen parameter spelled out -- not reverse-engineered.
    """
    import json

    recorded = {}

    def fake_run_preset(preset, **kwargs):
        recorded.update(kwargs)
        manifest = {"preset": kwargs.get("preset_name"), "errors": [],
                    "items": [], "command": kwargs.get("command")}
        out = kwargs.get("out_manifest")
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    import taters.pipelines.run_pipeline as rp
    real = rp.run_preset
    try:
        rp.run_preset = fake_run_preset
        p = ScriptedPrompter([
            "audio", *browse_to(media), ["readability"], "transcribe",
            "speaker", False, "Repro run", "run",
            "1",                        # workers
            "menu",                     # leave the finish screen
        ])
        wiz.run_wizard(p, cwd=tmp_path)
    finally:
        rp.run_preset = real

    command = recorded.get("command", "")
    assert command.startswith("python -m taters.pipelines.run_pipeline")
    assert "--preset-file" in command and "repro_run.yaml" in command
    assert f"--root_dir {media}" in command or f'--root_dir "{media}"' in command
    assert "--file_type audio" in command
    assert "--workers 1" in command


def test_variable_overrides_ride_the_reproduction_command():
    from taters.ui.wizard import repro_command

    cmd = repro_command(Path("/x/my run/my run.yaml"), root_dir=None,
                        file_type="any", workers=2,
                        overrides={"whisper_model": "small.en"})

    assert '--preset-file "/x/my run/my run.yaml"' in cmd, "spaces need quoting"
    assert "--var whisper_model=small.en" in cmd
    assert "--root_dir" not in cmd, "a text pipeline has no input folder"


def test_escape_at_the_finish_screen_is_just_do_something_else(tmp_path):
    """
    Round-2 issue 28: there is nothing above "What now?" to go back to -- the
    run already happened. Esc used to climb to the hub, print "Backed out.
    Nothing was changed." over a folder of fresh outputs, and discard a failed
    run's verdict so the process exited 0.
    """
    p = EscapingPrompter(["__esc__"])

    wiz.finish_screen(p, ok=False, manifest={"errors": ["boom"], "items": []},
                      folder=tmp_path,
                      manifest_path=tmp_path / "run_manifest.json")
    # returned normally, so the caller keeps its verdict and goes back to the menu


def test_grouping_a_single_column_csv_falls_back_instead_of_crashing(tmp_path):
    """
    Round-2 issue 29: 'Each group' with no spare columns built an empty
    checkbox, and questionary dies on one -- taking the whole TUI with it.
    """
    path = tmp_path / "solo.csv"
    path.write_text("text\nhello there\nmore words\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"],      # no id question: no columns left
        ["readability"], "group",               # nothing left to group by
        False, "Solo", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["meta"]["level"] == "row"
    assert any("nothing left to group" in r.lower() for r in p.reasons)


def test_escape_at_preflight_keeps_the_session_and_the_step(
        media, tmp_path, monkeypatch):
    """
    Round-2 issue 32: Esc at "Keep X in the pipeline?" climbed uncaught to the
    hub -- one keypress after answering source, columns and features threw the
    whole session away. Esc is "not this question", and the non-destructive
    reading is to keep the step.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    p = EscapingPrompter([
        "audio", *browse_to(media), ["readability"], "diarize", "speaker",
        "__esc__",                  # Esc at "Keep 'Speaker diarization'?"
        False, "Kept it", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    calls = [s["call"] for s in result.preset["steps"]]
    assert any("diarize" in c for c in calls), "Esc should keep, not drop"
    assert result.preset["meta"]["title"] == "Kept it"


def test_a_semicolon_csv_gets_its_real_delimiter(tmp_path):
    """Round-2 cut list: the sniff only knew tab-vs-comma, so a European-locale
    CSV got ',' baked into the preset and parsed as one giant column."""
    path = tmp_path / "euro.csv"
    path.write_text("id;text\n1;hello there\n2;more words\n", encoding="utf-8")

    assert wiz._delimiter_of(path) == ";"
    assert wiz._read_header(path) == ["id", "text"]


def test_uppercase_txt_files_are_found_like_the_browser_promises(tmp_path):
    """Round-2 cut list: the browser counts case-insensitively, the source
    asker globbed '*.txt' exactly -- '12 files' at browse, none at confirm."""
    folder = tmp_path / "essays"
    folder.mkdir()
    (folder / "ONE.TXT").write_text("hello", encoding="utf-8")
    (folder / "two.txt").write_text("there", encoding="utf-8")

    p = ScriptedPrompter([*browse_to(folder)])
    spec = wiz._ask_txt_source(p)

    assert spec is not None
    assert sorted(f.name for f in spec.inputs) == ["ONE.TXT", "two.txt"]


def test_backing_out_of_naming_shows_no_phantom_change_count(media, tmp_path):
    """Round-2 cut list: the whole-library default was injected into the live
    overrides dict at compose time, so Esc at naming brought the user back to
    an options screen claiming "(1 changed)" about an edit they never made."""
    p = EscapingPrompter([
        "audio", *browse_to(media), ["dictionaries"], "transcribe", "speaker",
        "keep",                     # empty library; keep the step
        False,                      # no settings changes
        "Clean",                    # name it -- compose runs HERE
        "__esc__",                  # Esc at "What next?" -> back to naming
        "__esc__",                  # Esc at naming -> back to options
        False,                      # ...which had better still be the plain gate
        "Clean", "replace", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    gates = [q for _k, q in p.asked if q == "Change any settings first?"]
    assert len(gates) == 2, (
        "with zero real changes, backing out must land on the gate, not on a "
        "list claiming something changed"
    )


def test_escape_at_the_workers_question_says_saved_not_nothing_changed(
        media, tmp_path):
    """Round-2 cut list: the preset is already on disk by this question; Esc
    used to climb to the hub's 'Backed out. Nothing was changed.'"""
    p = EscapingPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe", "speaker",
        False, "Half run", "run",
        "__esc__",                  # Esc at "how many at once?"
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset_path.exists()
    assert result.ran is False
    assert any("Saved, not run" in line for line in p.output)


def test_a_typoed_recipe_id_is_a_compose_error(tmp_path):
    from taters.ui.compose import ComposeError, resolve_selection

    with pytest.raises(ComposeError) as err:
        resolve_selection(["readabillity"])
    assert "readabillity" in str(err.value)


def test_a_composite_template_shows_its_rendered_value_not_not_set():
    """Round-2 cut list: "{{var:features_dir}}/whisper-embeddings" displayed as
    "not set" on the options screen -- not what a step with a perfectly good
    destination deserves."""
    from taters.ui.introspect import ParamSpec

    recipe = by_id("whisper_embeddings")
    param = ParamSpec(name="output_dir", annotation="str", default=None,
                      required=False, kw_only=True, desc="", widget="text",
                      choices=())
    current, var_name, _w = wiz._current_value(
        recipe, param, {"features_dir": {"default": "features"}})

    assert current == "features/whisper-embeddings"
    assert var_name is None, "a composite is derived, not owned by one variable"


def test_a_failed_global_step_says_why_on_the_finish_screen(tmp_path):
    """
    From a real failed run: the dictionaries step died, the manifest recorded
    why, and the screen said only "Finished with problems: a step failed" --
    per-ITEM failures had lines, step-level errors had nowhere. The one
    sentence explaining an empty features/ folder was only in the JSON.
    """
    p = ScriptedPrompter(["menu"])
    wiz.finish_screen(
        p, ok=False,
        manifest={"items": [], "errors": [
            "potato.text.analyze_with_dictionaries failed: 'invective.dic' "
            "has no %...% category header"]},
        folder=tmp_path, manifest_path=tmp_path / "run_manifest.json")

    assert any("category header" in line for line in p.output), (
        "the step-level error never reached the screen"
    )


def test_gather_plumbing_is_not_offered_as_step_settings():
    """
    Reported from the frequency list's menu: "recursive" and "pattern"-style
    plumbing read as questions, are irrelevant on a CSV path, and -- edited on
    one of two steps sharing a gathered table -- would silently disagree with
    the file the first step already wrote. The source stage owns them.
    """
    from taters.ui import recipes as _r
    from taters.ui.wizard import _editable_names, describe, load_target

    recipe = _r.by_id("ngram_frequencies")
    spec = describe(load_target(recipe.target))
    everyday, rest = _editable_names(recipe, spec)
    offered = set(everyday) | set(rest)
    for plumbing in ("recursive", "id_from", "include_source_path", "joiner",
                     "num_buckets", "max_open_bucket_files", "tmp_root"):
        assert plumbing not in offered, plumbing
    assert "min_freq" in offered
    assert "ngram_n" in offered


def test_a_shared_setting_edits_from_the_step_screen_pipeline_wide():
    """
    The step screen offers `lemmatize` even when the DTM shares it; editing
    it there explains itself and lands once in the preset's vars -- both
    steps update together, and only the Shared row takes the credit.
    """
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("ngram_frequencies"), _r.by_id("doc_term_matrix")]
    var_specs = compose(["doc_term_matrix"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    # toggled to False, which is the non-default answer now: lemmatizing is on
    # by default for the three word-counting steps, and only a value that
    # differs from the default gets recorded as an override
    p = ScriptedPrompter(["ngram_frequencies", "lemmatize", False,
                          ":done", ":done", ":done"])
    overrides, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False)

    assert var_values == {"lemmatize": False}
    assert overrides.get("ngram_frequencies", {}) == {}
    assert any("shared setting" in r for r in p.reasons)
    row = next(c for _q, cs in p.offered for c in cs if c.value == "lemmatize")
    assert row.annotation == "(shared)"


def test_engine_gated_settings_hide_until_the_engine_is_chosen():
    """
    `stanza_lang` and `device` are noise while the engine is nltk. The rows
    appear the moment the engine changes -- the menu re-evaluates its gates
    against the live value on every repaint.
    """
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("parts_of_speech")]
    var_specs = compose(["parts_of_speech"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter([
        "parts_of_speech",              # open the step
        "engine", "stanza",             # flip the gate
        ":done", ":done",
    ])
    wiz.ask_tuning(p, steps, var_specs, ask_gate=False)

    menus = [choices for q, choices in p.offered if "change a setting" in q]
    first, second = menus[0], menus[-1]
    assert "stanza_lang" not in {c.value for c in first}
    assert "device" not in {c.value for c in first}
    assert "stanza_lang" in {c.value for c in second}
    assert "device" in {c.value for c in second}
    # ...indented, and sitting directly under the engine row they depend on
    values = [c.value for c in second]
    assert values.index("stanza_lang") > values.index("engine")
    assert all(c.label.startswith(wiz.INDENT) for c in second
               if c.value in ("stanza_lang", "device"))


def test_choosing_stanza_without_the_package_warns_at_the_moment_of_choice(
        monkeypatch):
    """Preflight has already run by the time the options screen exists, so the
    warning belongs to the answer itself -- not to a run an hour later."""
    import builtins
    import sys

    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    real_import = builtins.__import__

    def no_stanza(name, *args, **kwargs):
        if name == "stanza":
            raise ImportError("not here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_stanza)
    monkeypatch.delitem(sys.modules, "stanza", raising=False)

    steps = [_r.by_id("parts_of_speech")]
    var_specs = compose(["parts_of_speech"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter(["parts_of_speech", "engine", "stanza",
                          ":done", ":done"])
    _o, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False)

    assert var_values["engine"] == "stanza", "the answer itself must stand"
    assert any("pip install taters[stanza]" in line for line in p.output)


def test_choosing_stanza_announces_the_slow_import(monkeypatch):
    """Importing stanza pulls in torch and takes seconds; the screen must say
    it is working before the wait, not sit still looking hung."""
    import sys

    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    monkeypatch.delitem(sys.modules, "stanza", raising=False)

    steps = [_r.by_id("parts_of_speech")]
    var_specs = compose(["parts_of_speech"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    p = ScriptedPrompter(["parts_of_speech", "engine", "stanza",
                          ":done", ":done"])
    wiz.ask_tuning(p, steps, var_specs, ask_gate=False)

    assert any("takes a moment" in line for line in p.output)


def test_workers_is_one_shared_setting_across_the_text_steps():
    """The dial appears once, under Shared settings, wearing the shared tag on
    each step -- not as eight independent per-step copies that silently do not
    affect each other."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("ngram_frequencies"), _r.by_id("doc_term_matrix")]
    shared = wiz.shared_variables(steps)
    assert "workers" in shared

    preset = compose(["doc_term_matrix"], name="x", source="csv",
                     input_path="t.csv")
    assert preset["vars"]["workers"] == 0
    # the text steps all read the one variable. (the word-cloud step that comes
    # with the frequency list has no workers dial, since it draws one picture)
    dialed = [s for s in preset["steps"] if "workers" in s["with"]]
    assert len(dialed) >= 2
    for step in dialed:
        assert step["with"]["workers"] == "{{var:workers}}"


def test_flipping_the_engine_back_rescues_an_orphaned_stanza_tokenizer():
    """engine=nltk + tokenizer=stanza composes a pipeline that only fails at
    run time; validity cascades, with a note saying what moved and why."""
    p, var_values = _engine_tuning([
        "parts_of_speech",
        "engine", "stanza",
        "tokenizer", "stanza",
        "engine", "nltk",
        ":done", ":done",
    ])
    assert var_values["engine"] == "nltk"
    assert var_values["tokenizer"] == "potts"
    assert any("switched back to potts" in line for line in p.output)


def test_choosing_the_stanza_tokenizer_brings_its_engine_along():
    """Intent flows upward: picking stanza's tokenizer IS picking stanza."""
    p, var_values = _engine_tuning([
        "parts_of_speech",
        "tokenizer", "stanza",
        ":done", ":done",
    ])
    assert var_values["engine"] == "stanza"
    assert var_values["tokenizer"] == "stanza"
    assert any("engine switched to stanza" in line for line in p.output)


def test_choosing_stanza_does_not_drag_the_tokenizer_with_it():
    """Preferences do NOT cascade: potts is the deliberate default under both
    engines (comparable counts, emoticons survive), so engine=stanza leaves
    the tokenizer exactly where it was."""
    _p, var_values = _engine_tuning([
        "parts_of_speech",
        "engine", "stanza",
        ":done", ":done",
    ])
    assert var_values["engine"] == "stanza"
    assert "tokenizer" not in var_values


def test_ask_workers_offers_the_machine_and_leaves_a_quarter_for_the_human(monkeypatch):
    """The recommendation is three-quarters of the logical cores (the same
    number `workers: 0` resolves to), the ceiling choice names ALL of them,
    and a typed answer is clamped to the 1..cores range."""
    from taters.helpers import parallel_map as pm

    monkeypatch.setattr(pm.os, "cpu_count", lambda: 8)

    p = ScriptedPrompter(["6"])
    assert wiz.ask_workers(p, fans_out=True) == 6
    _q, choices = p.offered[0]
    values = [str(c.value) for c in choices]
    assert values == ["6", "1", "8"], "recommended, gentle, and the whole machine"
    assert any("8" in c.label and "8" == str(c.value) for c in choices)

    # the select only offers in-range values, so picking the ceiling is the
    # loudest ask a user can make here (a typed `--workers 999` gets clamped by
    # resolve_workers, and is tested with it)
    p = ScriptedPrompter(["8"])
    assert wiz.ask_workers(p, fans_out=True) == 8

    # on a machine where the recommendation IS the ceiling, there's no third choice
    monkeypatch.setattr(pm.os, "cpu_count", lambda: 1)
    p = ScriptedPrompter(["1"])
    assert wiz.ask_workers(p, fans_out=True) == 1
    _q, choices = p.offered[0]
    assert [str(c.value) for c in choices] == ["1", "1"], \
        "no 'everything this machine has' row when the recommendation is it"


def test_the_finish_question_itself_carries_the_verdict(tmp_path):
    """From a real run: the red failure lines scrolled past and the screen a
    user actually sat on -- folder listing plus "What now?" -- read as
    success. The question is the one line the cursor is on, so it says so."""
    p = ScriptedPrompter(["menu"])
    wiz.finish_screen(p, ok=False,
                      manifest={"items": [], "errors": ["boom"]},
                      folder=tmp_path, manifest_path=tmp_path / "m.json")
    assert any("problems" in q for _k, q in p.asked), p.asked

    p = ScriptedPrompter(["menu"])
    wiz.finish_screen(p, ok=True, manifest={"items": [], "errors": []},
                      folder=tmp_path, manifest_path=tmp_path / "m.json")
    assert any(q == "What now?" for _k, q in p.asked), (
        "a clean run must not cry wolf"
    )


def test_a_step_crash_names_the_exception_type(tmp_path):
    """A bare KeyError stringifies to just its key: a real run reported
    `failed: '1467-'`, which told the user nothing. The type is the clue."""
    from taters.pipelines import run_pipeline as rp

    class Boom:
        def boom(self, **kwargs):
            raise KeyError("1467-")

    status, _globals, err = rp.run_global_step(
        step={"scope": "global", "call": "potato.boom", "with": {}},
        potato=Boom(), globals_ctx={}, vars_ctx={},
        manifest_path=tmp_path / "m.json", quiet=True)
    assert status == "error"
    assert "KeyError" in err["error"] and "1467-" in err["error"], err


def test_gather_plumbing_is_pipeline_handled_for_text_steps():
    """Reported from the cohesion step's menu: "which files to search for"
    was offered even though the pipeline gathers all text before the step
    fires. One universal rule -- compose REPLACES every text step's input
    group (TEXT_INPUT_KEYS) with its own wiring -- so those settings are
    withheld from every text step's options, and a grayed row says where
    they went."""
    from taters.ui import recipes as _r
    from taters.ui.introspect import describe, load_target

    for rid in ("cohesion", "readability", "dictionaries", "ngram_frequencies"):
        recipe = _r.by_id(rid)
        spec = describe(load_target(recipe.target))
        everyday, rest = wiz._editable_names(recipe, spec)
        offered = set(everyday) | set(rest)
        leaked = offered & set(_r.TEXT_INPUT_KEYS)
        assert not leaked, f"{rid} offers pipeline-handled settings: {leaked}"
        # the real settings are still there
        assert offered, rid

    # and a non-text step is untouched by the rule
    recipe = _r.by_id("transcribe")
    spec = describe(load_target(recipe.target))
    everyday, rest = wiz._editable_names(recipe, spec)
    assert set(everyday) | set(rest), "transcribe still offers its settings"


def test_the_options_menu_says_where_the_plumbing_went():
    p, _values = _engine_tuning([
        "parts_of_speech",
        ":done", ":done",
    ])
    _q, choices = next(
        (q, cs) for q, cs in p.offered if "change a setting" in q)
    note = [c for c in choices if c.value == "_pipeline_handled_"]
    assert note, "no grayed row explaining the hidden input settings"
    assert note[0].disabled and "gather" in note[0].disabled
    assert "pattern" not in {c.value for c in choices}


def test_shared_annotation_wears_parentheses():
    """"model: base.en shared" read as a model NAMED "base.en shared" -- the
    tag needs its parentheses to read as a tag."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose
    from taters.ui.introspect import describe, load_target

    recipe = _r.by_id("parts_of_speech")
    spec = describe(load_target(recipe.target))
    var_specs = compose(["parts_of_speech"], name="x", source="csv",
                        input_path="t.csv")["meta"]["variables"]
    choice = wiz._setting_choice(recipe, spec.get("engine"), var_specs,
                                 {}, {}, shared_marker=True)
    assert choice.annotation == "(shared)"
    plain = wiz._setting_choice(recipe, spec.get("engine"), var_specs,
                                {}, {}, shared_marker=False)
    assert plain.annotation == ""


def test_a_gated_change_survives_its_gate_being_closed():
    """Turn PCA on, set its component count, turn it off, turn it back on:
    the count is still what it was set to. Hiding a row must never discard
    its answer."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("stats_correlations")]
    var_specs = compose(["readability", "stats_correlations"], source="csv",
                        input_path="t.csv", text_cols=["text"], name="x",
                        var_values={"stats_outcome_cols": ["o"]})["meta"]["variables"]
    p = ScriptedPrompter([
        "stats_correlations",
        "pca", "all",
        "pca_components", "3",
        "pca_max_missing", "0.25",
        "pca", "off",
        "pca", "all",
        ":done", ":done",
    ])
    overrides, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False)
    assert var_values["stats_pca_components"] == 3
    assert overrides["stats_correlations"]["pca_max_missing"] == 0.25
    menus = [choices for q, choices in p.offered if "change a setting" in q]
    closed = [c.value for c in menus[-2]]          # right after pca -> off
    assert "pca_components" not in closed and "pca_max_missing" not in closed
    reopened = {c.value: c.label for c in menus[-1]}
    assert reopened["pca_components"].endswith("— 3")
    assert reopened["pca_max_missing"].endswith("— 0.25")


def test_a_hidden_row_cannot_be_picked():
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("stats_correlations")]
    var_specs = compose(["readability", "stats_correlations"], source="csv",
                        input_path="t.csv", text_cols=["text"], name="x",
                        var_values={"stats_outcome_cols": ["o"]})["meta"]["variables"]
    p = ScriptedPrompter(["stats_correlations", "pca_components", "3", ":done", ":done"])
    with pytest.raises(AssertionError, match="not among"):
        wiz.ask_tuning(p, steps, var_specs, ask_gate=False)


def test_the_meaning_tuned_step_asks_which_model_the_way_the_raw_one_does(tmp_path):
    """
    Sentence embeddings took its model silently and offered a change only on
    the options screen, where nobody looked for it -- while the raw-encoder
    step asked right after the checklist. Now both ask, and the answer is an
    ordinary preset variable of its own.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = tmp_path / "s.csv"
    path.write_text("id,text\n1,hello there\n2,bye now\n", encoding="utf-8")
    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], True, ["id"],
        ["sentence_embeddings"], "row",
        "sentence-transformers/all-mpnet-base-v2",   # the model question
        ":done", "Emb", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=False)

    questions = [q for _k, q in p.asked]
    assert "Which meaning-tuned model?" in questions
    assert "Which encoder?" not in questions, "the raw-encoder question was asked instead"
    variables = result.preset["meta"]["variables"]
    assert variables["sentence_model"]["default"] == "sentence-transformers/all-mpnet-base-v2"
    assert "encoder" not in variables


def test_with_both_embedding_steps_ticked_the_two_model_questions_cannot_be_confused(tmp_path):
    """
    The case that motivated the wording: two model menus in a row. Each has
    its own question, each says which step it is for, each says the other
    step asks separately, and the two answers land in two variables.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = tmp_path / "s.csv"
    path.write_text("id,text\n1,hello there\n2,bye now\n", encoding="utf-8")
    p = ScriptedPrompter([
        "csv", *browse_to(path), ["text"], True, ["id"],
        ["sentence_embeddings", "transformer_embeddings"], "row",
        "sentence-transformers/all-mpnet-base-v2",   # meaning-tuned model
        "distilroberta-base",                        # raw encoder
        ":done", "Emb", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path, analyses=False)

    questions = [q for _k, q in p.asked]
    assert "Which meaning-tuned model?" in questions
    assert "Which encoder?" in questions
    assert questions.count("Which meaning-tuned model?") == 1
    assert questions.count("Which encoder?") == 1

    reasons = [" ".join(r.split()) for r in p.reasons]
    tuned = [r for r in reasons if "runs on a meaning-tuned model" in r]
    raw = [r for r in reasons if "runs on a raw encoder" in r]
    assert len(tuned) == 1 and "Sentence embeddings (meaning-tuned model)" in tuned[0]
    assert len(raw) == 1 and "Transformer embeddings (raw, any encoder)" in raw[0]
    assert "asks for its own model separately" in tuned[0]
    assert "asks for its own model separately" in raw[0]
    # and the line right above each menu names the step too
    assert any(r.startswith("The meaning-tuned model for 'Sentence embeddings") for r in reasons)
    assert any(r.startswith("The encoder for 'Transformer embeddings") for r in reasons)

    variables = result.preset["meta"]["variables"]
    assert variables["sentence_model"]["default"] == "sentence-transformers/all-mpnet-base-v2"
    assert variables["encoder"]["default"] == "distilroberta-base"
