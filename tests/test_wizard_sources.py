"""
Tests for the setup wizard, driven end to end without a terminal (see
wizard_helpers for how). If a test fails with 'ran out of answers', the
message names the question that went unanswered -- insert an answer at
that position.

This file: the source stage -- folders of documents, spreadsheets and their
columns -- and the level question that follows the feature checklist.
"""

from __future__ import annotations


import yaml
from preset_checks import check_preset

from taters.ui import wizard as wiz
from taters.ui.prompts import (ScriptedPrompter)
from wizard_helpers import (clean_machine,  # noqa: F401  (autouse fixture)
                            browse_to, run, EscapingPrompter, _tuning,
                            _escaping_tuning, _engine_tuning, _asked)


# ---------------------------------------------------------------------------
# text and spreadsheet sources
# ---------------------------------------------------------------------------
#
# these drive the whole wizard for somebody who already has text. the shape of
# the question list is really the assertion here: no transcription question, no
# ffmpeg warning, and for a spreadsheet, a question about which column to read


def test_a_folder_of_text_files_needs_no_transcription_question(tmp_path, essays):
    p = ScriptedPrompter([
        "txt_dir", *browse_to(essays), ["readability"], False, "Essay study", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    questions = " | ".join(q for _, q in p.asked).lower()
    assert "transcript" not in questions
    assert result.source == "txt_dir"
    # our one measure, plus the descriptives that every feature table gets
    assert [s["call"] for s in result.preset["steps"]] == [
        "potato.text.analyze_readability", "potato.stats.describe_features"]


def test_a_spreadsheet_asks_which_column_holds_the_text(tmp_path, survey_csv):
    """
    The analyzers default to `text_cols=("text",)`. A file whose column is
    called `response` would analyze nothing and report no error, which is the
    most likely way for a first run to fail silently. So it has to be asked.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(survey_csv), ["response"], True, ["pid"],
        ["readability"], "row", [], False, "Survey", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    questions = [q for _, q in p.asked]
    assert any("hold the text you want analyzed" in q for q in questions)
    assert result.preset["steps"][0]["with"]["text_cols"] == ["response"]
    assert result.preset["steps"][0]["with"]["id_cols"] == ["pid"]
    # just one column, so there's nothing to decide about combining them
    assert not any("How should they be measured" in q for q in questions)
    assert result.preset["steps"][0]["with"]["mode"] == "concat"


def test_the_column_choices_come_from_the_file(tmp_path, survey_csv):
    """The offered columns must be the real header, not a guess."""
    header = wiz._read_header(survey_csv)
    assert header == ["pid", "response", "condition"]


def test_a_csv_with_a_bom_does_not_grow_a_phantom_column_name(tmp_path):
    """
    Spreadsheets exported from Excel carry a byte-order mark. Read as plain
    utf-8 the first column comes back as '﻿pid', which then fails to match
    anything the user picked.
    """
    path = tmp_path / "excel.csv"
    path.write_bytes("﻿pid,response\np1,hello\n".encode("utf-8"))
    assert wiz._read_header(path) == ["pid", "response"]


def test_a_file_that_is_not_a_csv_is_refused_rather_than_composed(tmp_path):
    junk = tmp_path / "notes.docx"
    junk.write_bytes(b"\x50\x4b\x03\x04\x00\x00\x00\x00")
    assert wiz._read_header(junk) == []


def test_an_empty_text_folder_loops_instead_of_composing_nothing(tmp_path, essays):
    """
    Same contract as the media path: find out now, not after picking options.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    p = ScriptedPrompter([
        "txt_dir", *browse_to(empty),      # nothing in here
        True,                       # try again? -> yes, back to the top
        "txt_dir", *browse_to(essays),
        ["readability"], False, "Essay study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert any("No documents" in line for line in p.output)


def test_a_text_preset_is_saved_and_validates(tmp_path, essays):
    p = ScriptedPrompter([
        "txt_dir", *browse_to(essays), ["readability", "lexical_richness"],
        False, "Essay measures", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    written = yaml.safe_load(result.preset_path.read_text(encoding="utf-8"))
    assert not check_preset(written)
    assert written["vars"]["input_dir"] == str(essays)
    # GLOBAL-only, so the re-run line had better not tell them to pass --root_dir
    assert "--root_dir" not in written["meta"]["cli_example"]


def test_audio_features_are_absent_from_the_checklist_for_text(tmp_path, essays):
    p = ScriptedPrompter([
        "txt_dir", *browse_to(essays), ["readability"], False, "Essay study", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    offered = [c.value for c in p.offered_choices("Which features do you want to extract?")]
    assert "acoustics" not in offered
    assert "whisper_embeddings" not in offered
    assert "readability" in offered


def test_the_review_table_counts_in_english():
    """"1 steps" in a summary reads as a bug even when nothing is wrong."""
    assert wiz._plural(1, "step") == "1 step"
    assert wiz._plural(0, "step") == "0 steps"
    assert wiz._plural(3, "file") == "3 files"


def test_the_checklist_help_matches_the_chosen_source(tmp_path, essays, media):
    """The same recipe describes itself differently for essays than for audio."""
    text_run = ScriptedPrompter([
        "txt_dir", *browse_to(essays), ["readability"], False, "E", "save"])
    wiz.run_wizard(text_run, cwd=tmp_path)

    media_run = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe", "speaker", False, "M", "save"])
    wiz.run_wizard(media_run, cwd=tmp_path)

    def help_for(prompter):
        return next(c.help for c in prompter.offered_choices("Which features do you want to extract?")
                    if c.value == "readability")

    assert "speaker" in help_for(media_run)
    assert "speaker" not in help_for(text_run)
    assert "one row per text" in help_for(text_run)


def test_several_text_columns_prompt_for_how_to_combine_them(tmp_path):
    """
    A spreadsheet is the one source where a row can carry several pieces of
    text -- an open-ended answer and a follow-up, a headline and a body. Both
    readings are legitimate, so it has to be asked rather than assumed.
    """
    path = tmp_path / "survey.csv"
    path.write_text(
        "pid,answer,followup\n"
        'p1,"First thought.","On reflection, more."\n',
        encoding="utf-8",
    )

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer", "followup"], "separate", True, ["pid"],
        ["readability"], "row",
        [],                       # we get offered statistics now, and say no
        False, "Survey", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    with_ = result.preset["steps"][0]["with"]
    assert with_["text_cols"] == ["answer", "followup"]
    assert with_["mode"] == "separate"


def test_combining_several_columns_is_also_offered(tmp_path):
    path = tmp_path / "survey.csv"
    path.write_text("pid,answer,followup\np1,a,b\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["answer", "followup"], "concat", False,
        ["readability"], "row", [], False, "Survey", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)
    assert result.preset["steps"][0]["with"]["mode"] == "concat"


def test_ticking_no_text_column_is_re_asked(tmp_path, survey_csv):
    """A spreadsheet with no text column selected has nothing to measure."""
    p = ScriptedPrompter([
        "csv", *browse_to(survey_csv), [], ["response"], False,
        ["readability"], "row", [], False, "Survey", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)
    assert "pick at least one column" in p.text_output


def test_a_text_column_is_not_also_offered_as_an_identifier(tmp_path, survey_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(survey_csv), ["response"], True, ["pid"],
        ["readability"], "row", [], False, "Survey", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    offered = [c.value for c in p.offered_choices("Which one(s)?")]
    assert "response" not in offered
    assert {"pid", "condition"} <= set(offered)


# ---------------------------------------------------------------------------
# analysis level, from the wizard's side
# ---------------------------------------------------------------------------


def test_media_is_asked_what_a_row_should_describe(tmp_path, media):
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe",
        "conversation", False, "Talk", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["meta"]["level"] == "conversation"
    step = next(s for s in result.preset["steps"]
                if "readability" in s["call"])
    assert step["with"]["group_by"] == ["source"]


def test_a_folder_of_text_files_is_not_asked(tmp_path, essays):
    """One text per file already. A question with one answer wastes a screen."""
    p = ScriptedPrompter([
        "txt_dir", *browse_to(essays), ["readability"], False, "E", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert not any("one row of results" in q.lower() for _kind, q in p.asked)


def test_a_pipeline_with_no_text_features_is_not_asked(tmp_path, media):
    """
    Acoustics has a grain of its own -- one row per speaker's audio -- that the
    level neither sets nor should. Asking would imply otherwise.
    """
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["acoustics"], "transcribe",
        False, "Sound", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert not any("one row of results" in q.lower() for _kind, q in p.asked)


def test_the_level_defaults_to_the_speaker(tmp_path, media):
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe",
        "speaker", False, "Default", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    offered = [(q, choices) for q, choices in p.offered
               if "one row of results" in q.lower()]
    assert offered, "the question was not asked"
    # most to least granular, with the pointer starting on the speaker. the order
    # is for reading, the default is for just pressing enter
    assert [c.value for c in offered[0][1]] == [
        "utterance", "speaker", "conversation"]
    assert p.select_defaults["What should one row of results describe?"] \
        == "speaker"


def test_backing_out_of_the_level_returns_to_the_features(media, tmp_path):
    """It is part of "what do you want out of it", so back is the checklist."""
    p = EscapingPrompter([
        "audio", *browse_to(media), ["readability"], "transcribe",
        "__esc__",                          # Esc at the level question
        ["readability"], "transcribe", "speaker",
        False, "Second", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    asked = [q for _kind, q in p.asked]
    # we match on "out of it" specifically: the unit-of-analysis question starts
    # with the same four words, and a plain prefix match was counting both
    assert len([q for q in asked if q.startswith("Which features do you want to extract")]) == 2
    assert result.preset["meta"]["level"] == "speaker"


def test_the_review_says_how_each_step_combines_its_rows(tmp_path, media):
    """
    The default is deliberately mixed -- words joined for the text analyzers,
    scores averaged for sentence embeddings -- and nothing in the output files
    says which. The review is the one place it is visible.
    """
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability", "sentence_embeddings"],
        "transcribe", "speaker", False, "Mixed", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    table = next(t for t in p.tables if "Mixed" in t[0])
    _title, rows, headers = table
    assert "Measures" in headers
    measures = {r[1]: r[headers.index("Measures")] for r in rows}
    assert measures["Readability scores"] == "words joined"
    assert measures["Merge sentence embeddings"] == "scores averaged"


def test_the_review_does_not_claim_a_merge_that_did_not_combine_anything(tmp_path, media):
    """
    At the utterance level the merge groups on the per-row identifier, so it
    collapses nothing -- it is a tidy copy. Calling that "scores averaged"
    would claim a combining step that never happened.
    """
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability", "sentence_embeddings"],
        "transcribe", "utterance", False, "Raw", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    _title, rows, headers = next(t for t in p.tables if "Raw" in t[0])
    measures = {r[1]: r[headers.index("Measures")] for r in rows}
    assert measures["Readability scores"] == "each row"
    # the merge shouldn't show up at all any more: a raw-level merge combined
    # nothing and lost columns, so compose drops it (round-2 issues 26/27)
    assert "Merge sentence embeddings" not in measures


def test_a_slow_step_says_so_before_it_blocks(tmp_path, media):
    """
    From a real report: opening the options for archetypes looked like a freeze.
    It was a twenty-second import with nothing on screen. The imports are lazy
    now, but a module built on a pre-trained model can still take a moment, and
    a blank finished screen reads as a hang whatever the cause.
    """
    wiz._SPEC_CACHE.clear()
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["archetypes"], "transcribe", "speaker",
        "keep",                     # empty library; keep the step anyway
        True, "archetypes", ":done", ":done", "Slow", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    said = [line for line in p.output if "Reading" in line and "settings" in line]
    assert said, "nothing on screen while the settings were being read"


def test_a_signature_is_only_read_once(tmp_path, media):
    """
    Opening the same step's options twice should not pay the import twice --
    and should not have to rely on `sys.modules` to avoid it.
    """
    wiz._SPEC_CACHE.clear()
    calls = []
    real = wiz.load_target
    try:
        wiz.load_target = lambda target: (calls.append(target), real(target))[1]
        p = ScriptedPrompter([
            "audio", *browse_to(media), ["archetypes"], "transcribe", "speaker",
            "keep",                 # empty library; keep the step anyway
            True, "archetypes", ":done", "archetypes", ":done", ":done",
            "Twice", "save",
        ])
        wiz.run_wizard(p, cwd=tmp_path)
    finally:
        wiz.load_target = real

    archetype_reads = [t for t in calls if "archetypes" in t]
    assert len(archetype_reads) == 1, f"read {len(archetype_reads)} times"
