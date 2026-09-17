"""
Tests for the setup wizard, driven end to end without a terminal (see
wizard_helpers for how). If a test fails with 'ran out of answers', the
message names the question that went unanswered -- insert an answer at
that position.

This file: the options screen -- what it explains, what it offers and withholds,
where the shared settings come from, Esc inside it, and backing out of the
review stage.
"""

from __future__ import annotations


import pytest

from taters.ui import wizard as wiz
from taters.ui.introspect import ParamSpec
from taters.ui.prompts import (Cancelled, ScriptedPrompter)
from taters.ui.recipes import by_id
from wizard_helpers import (clean_machine,  # noqa: F401  (autouse fixture)
                            browse_to, run, EscapingPrompter, _tuning,
                            _escaping_tuning, _engine_tuning, _asked)


# ---------------------------------------------------------------------------
# explaining the options
# ---------------------------------------------------------------------------

def test_an_option_is_explained_before_it_is_asked_for(tmp_path):
    """
    `rounding`, `relative_freq`, `mean_center_vectors` mean nothing to someone
    who has never opened the module -- and that is exactly who this is for. A
    question with only a name on it is a question they cannot answer.
    """
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    p = ScriptedPrompter([True, "dictionaries", "relative_freq", False, ":done", ":done"])
    try:
        wiz.ask_tuning(p, [_r.by_id("dictionaries")],
                       compose(["dictionaries"], name="x")["meta"]["variables"])
    except AssertionError:
        pass

    shown = p.text_output
    # `relative_freq` is one of the three offered by default, and it's an opaque
    # name that means nothing without its sentence
    assert "relative frequencies instead of raw counts" in shown
    assert "Now: yes" in shown, "the present value has to be visible to leave alone"


def test_a_whole_description_is_shown_not_just_its_first_line(tmp_path):
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    p = ScriptedPrompter([True, "dictionaries", "out_features_csv", "", ":done", ":done"])
    wiz.ask_tuning(p, [_r.by_id("dictionaries")],
                   compose(["dictionaries"], name="x")["meta"]["variables"])

    # this description wraps in the source, and the tail used to get dropped
    assert "defaults to ./features/dictionary/" in p.text_output


def test_the_allowed_answers_are_named_for_a_choice(tmp_path):
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    # ("mode" was our specimen here until we withheld the input-plumbing
    # settings from text steps, since the pipeline gathers before they run)
    p = ScriptedPrompter([True, "ngram_frequencies", "engine", "nltk",
                          ":done", ":done"])
    wiz.ask_tuning(p, [_r.by_id("ngram_frequencies")],
                   compose(["ngram_frequencies"], name="x")["meta"]["variables"])

    assert "One of: nltk, stanza" in p.text_output


def test_runner_infrastructure_is_never_offered_as_an_option(tmp_path):
    """
    `on_progress` is injected by the runner and `gathered_csv` is wiring the
    wizard sets itself. Their docstrings are written for whoever maintains the
    pipeline, and answering either would break the run.
    """
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    p = ScriptedPrompter([True, "dictionaries", ":done", ":done"])
    wiz.ask_tuning(p, [_r.by_id("dictionaries")],
                   compose(["dictionaries"], name="x")["meta"]["variables"])

    # every setting the step will ever offer is on that expanded menu
    offered = {c.value for _q, choices in p.offered for c in choices}
    assert "on_progress" not in offered
    assert "gathered_csv" not in offered


def test_a_boolean_reads_as_yes_or_no_not_true_or_false():
    assert wiz._describe_value(True) == "yes"
    assert wiz._describe_value(False) == "no"
    assert wiz._describe_value(None) == "not set"
    assert wiz._describe_value([]) == "empty"
    assert wiz._describe_value(["a", "b"]) == "a, b"


# ---------------------------------------------------------------------------
# the shape of the options screen
#
# the old flow asked one yes/no per step and then, for each yes, "show every
# option, not just the common ones?", which asks somebody to choose between
# two lists they haven't seen. nobody knows what a "common option" is before
# they've been shown one
# ---------------------------------------------------------------------------


def test_the_steps_are_offered_as_a_list_to_pick_from():
    p, _ = _tuning([True, ":done"])

    kinds = [kind for kind, _q in p.asked]
    assert kinds == ["confirm", "select"]
    offered = {c.value for _q, choices in p.offered for c in choices}
    assert {"transcribe", "sentence_embeddings"} <= offered


def test_nobody_is_asked_to_choose_between_lists_they_cannot_see():
    """The question this whole screen was redesigned to remove."""
    p, _ = _tuning([True, "transcribe", ":done", ":done"])

    asked = " ".join(q.lower() for _kind, q in p.asked)
    assert "common" not in asked
    assert "show every option" not in asked


def test_a_setting_shows_what_it_is_set_to_before_you_open_it():
    """
    The question someone is actually answering here is not "what settings
    exist" but "is anything set to something I do not want". A list of bare
    names makes them open each one to find out.
    """
    p, _ = _tuning([True, "transcribe", ":done", ":done"])

    labels = [c.label for _q, choices in p.offered for c in choices]
    assert any(label.startswith("whisper model — ") for label in labels)
    assert any(label.startswith("beam size — 5") for label in labels)


def test_a_boolean_setting_reads_as_yes_or_no_in_the_list():
    p, _ = _tuning([True, "transcribe", ":done", ":done"])

    labels = [c.label for _q, choices in p.offered for c in choices]
    assert any(label == "vad filter — yes" for label in labels)


def test_the_list_updates_after_a_setting_is_changed():
    """A menu still showing the old value reads as an edit that did not take."""
    p, (_ov, var_values) = _tuning(
        [True, "transcribe", "whisper_model", "small.en", ":done", ":done"])

    assert var_values["whisper_model"] == "small.en"
    menus = [choices for q, choices in p.offered if "change a setting" in q]
    before = [c.label for c in menus[0] if c.value == "whisper_model"][0]
    after = [c.label for c in menus[-1] if c.value == "whisper_model"][0]
    assert before != after
    assert after.endswith("small.en")


def test_you_can_change_several_settings_without_leaving_the_step():
    p, (overrides, var_values) = _tuning([
        True, "transcribe",
        "whisper_model", "small.en",
        "beam_size", "1",
        ":done", ":done",
    ])
    assert var_values["whisper_model"] == "small.en"
    assert overrides["transcribe"]["beam_size"] == 1


def test_every_setting_is_on_the_one_list():
    """
    There is no "Show N more settings" row any more. It existed because a long
    list ran off the bottom of the screen; the list scrolls now, so hiding
    settings behind a keypress is a worse answer than showing them.
    """
    p, spec = _tuning([True, "transcribe", ":done", ":done"])

    menu = [choices for q, choices in p.offered if "change a setting" in q][0]
    assert not [c for c in menu if c.value == ":more"]
    assert not [c for c in menu if c.label.startswith("Show ")]


def test_the_everyday_settings_still_come_first():
    """
    One list, but not an unordered one: what most people change is still at the
    top, and the rest follows it rather than being hidden behind it.
    """
    from taters.ui import wizard as w

    p, _ = _tuning([True, "transcribe", ":done", ":done"])
    recipe = by_id("transcribe")
    spec = w._spec_for(recipe, p)
    # shared-bound settings get listed on the step too now (marked "shared"),
    # so the expected order is just the full editable list
    everyday, rest = w._editable_names(recipe, spec)

    menu = [choices for q, choices in p.offered if "change a setting" in q][0]
    listed = [c.value for c in menu if not c.value.startswith(":")]

    assert rest, "transcribe needs both kinds for this to prove anything"
    assert listed[:len(everyday)] == list(everyday)
    assert set(listed) == set(everyday) | set(rest)


def test_every_step_menu_offers_a_way_out():
    p, _ = _tuning([True, "transcribe", ":done", ":done"])

    menu = [choices for q, choices in p.offered if "change a setting" in q][0]
    # first, not last: at the bottom of a scrolling list the way out was
    # invisible until you scrolled, so people didn't know it existed. green for
    # the same reason the browser's "✓ Use this folder" is green: one color
    # language, where the forward act leads and is green
    assert menu[0].value == ":done"
    assert "Done" in menu[0].label
    assert menu[0].tone == "good"


def test_only_the_step_you_picked_is_opened():
    p, _ = _tuning([True, "sentence_embeddings", ":done", ":done"])

    # the top-level list is a select too, now that it's a menu you come back to,
    # so what we care about is that no *other* step's settings got opened
    step_menus = [q for kind, q in p.asked
                  if kind == "select" and "change a setting" in q]
    assert step_menus and all("Sentence embeddings" in q for q in step_menus)


def test_finishing_one_step_returns_to_the_list_rather_than_leaving():
    """
    The bug this menu was rebuilt for. Changing one step's settings is not
    evidence that you are finished with the whole screen -- someone who opened
    the shared settings, changed one, and pressed "Done with shared settings"
    found their pipeline being written, with no way back to the step they had
    not thought of yet.
    """
    p, _ = _tuning([True, "transcribe", ":done", "sentence_embeddings", ":done",
                    ":done"])

    step_menus = [q for kind, q in p.asked
                  if kind == "select" and "change a setting" in q]
    assert any("Transcript" in q for q in step_menus)
    assert any("Sentence embeddings" in q for q in step_menus), (
        "the second step was unreachable after finishing the first"
    )


def test_the_list_says_which_parts_you_have_already_changed():
    """
    With the menu returning here after every step, nothing else on screen
    distinguishes the parts you have visited from the parts you have not.
    """
    p, _ = _tuning([True, "transcribe", "beam_size", "1", ":done",
                    ":done"])

    top = [choices for q, choices in p.offered if "What would you like" in q][-1]
    labels = {c.value: c.label for c in top}
    assert labels["transcribe"].endswith("(1 changed)")
    assert "changed" not in labels["sentence_embeddings"]


def test_the_way_out_says_what_it_will_save():
    p, _ = _tuning([True, "transcribe", "beam_size", "1", ":done",
                    ":done"])

    top = [choices for q, choices in p.offered if "What would you like" in q][-1]
    done = next(c for c in top if c.value == ":done")
    assert "1 change" in done.label


def test_a_shared_change_is_credited_to_one_row_only():
    """
    A step *reads* `whisper_model`, but the shared entry is where you changed
    it. Crediting both would make the row counts fail to add up to the total on
    the "Done" line, which reads as a bug in the tally.
    """
    # `device` is the one both of these steps read, so it's the one that lands
    # in the shared entry rather than under either of them
    p, _ = _tuning([True, ":shared", "device", "cpu", ":done", ":done"])

    top = [choices for q, choices in p.offered if "What would you like" in q][-1]
    labels = {c.value: c.label for c in top}
    assert "(1 changed)" in labels[":shared"]
    assert "changed" not in labels["transcribe"]
    assert "1 change" in labels[":done"]


def test_wiring_never_reaches_the_menu():
    """
    `csv_path: "{{transcripts_all}}"` is what makes the pipeline a pipeline.
    Now that every setting is listed at once, the full list must still not be
    a way to disconnect your own run.
    """
    p, _ = _tuning([True, "sentence_embeddings", ":done", ":done"])

    offered = {c.value for _q, choices in p.offered for c in choices}
    assert "csv_path" not in offered
    assert "analysis_csv" not in offered


# ---------------------------------------------------------------------------
# where the split comes from
#
# it used to be a `tunable` tuple hand-written on each recipe. that read better
# for some steps and worse for others, went stale quietly whenever a function
# grew a parameter, and had to be written again for every new module. now we
# derive it, so a new module needs no wizard work at all
# ---------------------------------------------------------------------------

def test_recipes_carry_no_hand_written_list_of_everyday_settings():
    from dataclasses import fields

    from taters.ui.recipes import Recipe

    assert "tunable" not in {f.name for f in fields(Recipe)}


def test_what_the_recipe_wrote_down_is_what_gets_shown_first():
    """
    Not a claim about which settings matter -- nothing in a signature or a
    docstring tells you that. It is a record of the settings someone actually
    made a decision about, which is the closest thing to a signal that exists,
    and it costs nobody any curation.
    """
    from taters.ui import recipes as _r
    from taters.ui.introspect import describe, load_target

    recipe = _r.by_id("sentence_embeddings")
    everyday, rest = wiz._editable_names(recipe, describe(load_target(recipe.target)))

    assert "model_name" in everyday and "normalize_l2" in everyday
    assert everyday == [n for n in recipe.with_ if n in everyday]   # recipe order
    assert "batch_size" in rest, "a plain signature default belongs under 'more'"


def test_a_setting_the_function_lost_is_not_offered():
    """A `with_` key that no longer exists must vanish rather than crash a menu."""
    from dataclasses import replace as _replace

    from taters.ui import recipes as _r
    from taters.ui.introspect import describe, load_target

    recipe = _r.by_id("readability")
    broken = _replace(recipe, with_={**recipe.with_, "renamed_away": 1})
    everyday, rest = wiz._editable_names(broken, describe(load_target(recipe.target)))

    assert "renamed_away" not in everyday + rest


# --- settings that belong to the pipeline, not to one step ------------------

def test_a_variable_several_steps_read_is_counted_as_shared():
    """
    `overwrite_existing` is referenced by thirteen of fifteen recipes. Offering
    it on every step's menu asks one question over and over and implies a local
    answer: changing `device` under "Transcript" changes it for the embeddings
    step too.
    """
    from taters.ui import recipes as _r

    steps = [_r.by_id("transcribe"), _r.by_id("whisper_embeddings"),
             _r.by_id("sentence_embeddings")]
    shared = wiz.shared_variables(steps)

    assert "overwrite_existing" in shared
    assert "whisper_model" in shared, "transcribe and whisper_embeddings both read it"
    assert "speaker_label" not in shared, "only one step reads it"


def test_a_shared_setting_stays_on_the_step_menu_marked_shared():
    """
    Requested after `lemmatize` "went missing": a setting bound to a shared
    variable stays on every step that reads it, wearing the green "shared"
    tag, and editing it there edits the one pipeline-wide value.
    """
    from taters.ui import recipes as _r
    from taters.ui.introspect import describe, load_target

    recipe = _r.by_id("transcribe")
    spec = describe(load_target(recipe.target))
    everyday, rest = wiz._editable_names(recipe, spec)
    assert "overwrite_existing" in everyday + rest

    row = wiz._setting_choice(recipe, spec.get("overwrite_existing"),
                              {}, {}, {}, shared_marker=True)
    assert row.annotation == "(shared)"


def test_the_shared_settings_get_a_row_of_their_own():
    p, _ = _tuning([True, ":done"])

    steps_menu = p.offered[0][1]
    # row 0 is the green Done, and the shared entry leads the *content* rows
    assert steps_menu[0].value == ":done"
    assert steps_menu[1].value == ":shared"
    assert "overwrite existing" in steps_menu[1].help


def test_a_shared_setting_is_named_after_the_variable_not_the_parameter():
    """
    `transcripts_dir` is one setting; the transcribe step calls its end of it
    `out_dir` and the merge step calls it `root_dir`. The name that is true for
    everyone is the variable's.
    """
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("transcribe"), _r.by_id("gather_transcripts")]
    p = ScriptedPrompter([True, ":shared", ":done", ":done"])
    wiz.ask_tuning(p, steps, compose(["transcribe"], name="x")["meta"]["variables"])

    rows = [(c.value, c.label) for _q, choices in p.offered for c in choices]
    assert "transcripts_dir" in {v for v, _l in rows}
    assert "out_dir" not in {v for v, _l in rows}
    # and it has to *read* that way too, not just be filed under it
    labels = [label for value, label in rows if value == "transcripts_dir"]
    assert labels and labels[0].startswith("transcripts dir — ")


def test_changing_a_shared_setting_writes_it_once_to_the_variables():
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("transcribe"), _r.by_id("whisper_embeddings")]
    p = ScriptedPrompter([True, ":shared", "whisper_model", "small.en", ":done", ":done"])
    overrides, var_values = wiz.ask_tuning(
        p, steps, compose(["whisper_embeddings"],
                          providers={"transcript_csv": "transcribe"},
                          name="x")["meta"]["variables"])

    assert var_values == {"whisper_model": "small.en"}
    assert overrides == {}, "a shared setting is never written into one step"


def test_a_merge_step_will_not_let_you_rearrange_what_it_merges():
    """
    `feature_gather` is told how to fold a table down, and the wizard works that
    out from what the step before it produced. Grouping row-level embeddings by
    a key that is unique per row yields a file of the right shape, with real
    numbers, that has aggregated nothing -- a bug that has already shipped here
    once, and not one to hand back to the user as a menu entry.
    """
    from taters.ui import recipes as _r

    for rid in ("gather_sentence_embeddings", "gather_whisper_embeddings"):
        recipe = _r.by_id(rid)
        for name in ("group_by", "aggregate", "stats", "per_file", "exclude_cols"):
            if name in recipe.with_:
                assert wiz.is_wired(recipe, name), f"{rid}.{name} is editable"
        assert not wiz.is_wired(recipe, "out_csv"), "where it lands is a fair question"


def test_escape_at_the_first_question_leaves_the_wizard(media, tmp_path):
    """
    From a real report: Esc on "What kind of data do you have?" did nothing
    visible. It was caught and the same question re-asked, on the reasoning
    that nothing comes before the first stage -- but something does. The menu
    that launched the wizard is the previous screen, and `run_hub` already
    treats a task backing out as "not this one after all".
    """
    from taters.ui.prompts import GoBack

    class _EscapesAtTheFirstQuestion(ScriptedPrompter):
        def select(self, question, choices, *, default=None, transient=False,
               toggle_values=(), ticked=None, navigate=None, breadcrumb=None):
            self.asked.append(("select", question))
            raise GoBack()

    p = _EscapesAtTheFirstQuestion([])
    with pytest.raises(GoBack):
        wiz.run_wizard(p, cwd=tmp_path)

    assert len(p.asked) == 1, "the same question was asked again instead of leaving"


def test_escape_later_in_the_wizard_rewinds_a_stage_rather_than_leaving(media, tmp_path):
    """
    The distinction that has to survive. Inside the wizard, Esc rewinds to the
    previous stage; only the very first question means out. Backing out of the
    feature list must land back on "What kind of data do you have?", not drop
    the session.
    """
    from taters.ui.prompts import GoBack

    class _BacksOutOfTheFeatureList(ScriptedPrompter):
        backed_out = False

        def checkbox(self, question, choices):
            if not self.backed_out:
                self.backed_out = True
                self.asked.append(("checkbox", question))
                raise GoBack()
            return super().checkbox(question, choices)

    p = _BacksOutOfTheFeatureList([
        "audio", *browse_to(media),          # first pass through the source stage
        "audio", *browse_to(media),          # and again, after the rewind
        ["transcribe"], False, "Rewound", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset_path is not None, "the wizard did not survive a rewind"
    kinds = [q for _kind, q in p.asked if "kind of data" in q]
    assert len(kinds) == 2, f"the source stage was not re-asked: {p.asked}"


def test_the_install_command_survives_being_displayed(media, tmp_path, monkeypatch):
    """
    From a real screen: the command printed as `pip install "taters"`. It went
    through rich as a note, and rich reads `[vocalacoustics]` as a style tag --
    so the one part that mattered was the part removed, leaving a command that
    looks right, runs cleanly, and installs nothing.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    _result, prompter = run([
        "audio", *browse_to(media), ["acoustics"], "transcribe",
        "keep", False, "Brackets", "save",
    ], cwd=tmp_path)

    shown = " ".join(c.help for _q, choices in prompter.offered for c in choices)
    assert 'pip install "taters[vocalacoustics]"' in shown
    assert "\\[" not in shown, "a rich escape leaked into text rich never renders"


def test_a_missing_extra_is_explained_where_it_is_asked_about(media, tmp_path,
                                                              monkeypatch):
    """
    Attached to the question, not printed before the screen was drawn. As a
    note it landed above the progress rail -- a paragraph of context a long way
    from the question it was about, and above the banner at that.
    """
    monkeypatch.setattr(wiz, "missing_extras", lambda r: list(r.extras))
    _result, prompter = run([
        "audio", *browse_to(media), ["acoustics"], "transcribe",
        "keep", False, "Attached", "save",
    ], cwd=tmp_path)

    asked = [(q, choices) for q, choices in prompter.offered if "Keep '" in q]
    assert asked, "the question is no longer a list with a description"
    _question, choices = asked[0]
    assert [c.value for c in choices] == ["keep", "drop"]
    # on both options, so it stays on screen whichever one is highlighted
    assert all("Not installed" in c.help for c in choices)

    # and it's not floating around in the notes any more
    assert "Not installed" not in prompter.text_output


# ---------------------------------------------------------------------------
# Esc inside the options screen
#
# somebody reported that Esc "doesn't go back to the previous screen". there
# were no `GoBack` handlers anywhere in the tuning flow, so Esc unwound all the
# way out of `ask_tuning` to `run_wizard`, which caught it and threw away every
# change that had been made. backing out of one setting discarded all of them
# ---------------------------------------------------------------------------


def test_escape_while_answering_a_setting_keeps_the_other_changes():
    """
    The worst of it. Esc on one setting used to unwind the whole screen and
    discard everything already changed -- which is the opposite of going back.
    """
    _p, (overrides, var_values), _back = _escaping_tuning([
        True, "transcribe",
        "whisper_model", "small.en",     # changed, and must survive
        "language", "__esc__",           # backed out of
        ":done", ":done",
    ])

    assert var_values["whisper_model"] == "small.en"
    assert "language" not in var_values, "a setting backed out of was still written"


def test_escape_on_a_steps_menu_returns_to_the_list_of_steps():
    p, _, _back = _escaping_tuning([True, "transcribe", "__esc__",
                             "sentence_embeddings", ":done", ":done"])

    opened = [q for q, _c in p.offered if "change a setting" in q]
    assert any("Transcript" in q for q in opened)
    assert any("Sentence embeddings" in q for q in opened), (
        "Esc left the options screen instead of going up one level"
    )


def test_escape_on_the_list_of_steps_leaves_with_the_changes_intact():
    """
    The top of this screen, so back means out of it -- but out with the work
    kept. Discarding here would make Esc a destructive key in the one place it
    reads as the safe one.
    """
    _p, (overrides, var_values), backed_out = _escaping_tuning([
        True, "transcribe", "whisper_model", "small.en", ":done",
        "__esc__",
    ])

    assert backed_out, "Esc at the top of this screen has to go backwards"
    assert var_values["whisper_model"] == "small.en"


def test_escape_at_the_opening_question_changes_nothing():
    _p, (overrides, var_values), backed_out = _escaping_tuning(["__esc__"])
    assert backed_out
    assert overrides == {} and var_values == {}


def test_escape_while_naming_goes_back_to_the_options_screen(media, tmp_path):
    """
    From a real report. Esc at "Name this pipeline" used to *accept* the default
    name and carry straight on to the review table -- the one direction Esc
    should never take anyone.
    """
    p = EscapingPrompter([
        "audio", *browse_to(media), ["transcribe"],
        False,                      # change any settings? -> no
        "__esc__",                  # Esc while naming
        False,                      # ...back at the options screen
        "Second time", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    asked = [q for _kind, q in p.asked]
    assert asked.count("Change any settings first?") == 2, (
        "Esc did not return to the options screen"
    )
    assert result.preset["meta"]["title"] == "Second time"


def test_escape_at_the_closing_question_goes_back_to_naming(media, tmp_path):
    """
    One step back, not out. Naming, the review table and this question are one
    stage, so Esc here rewinds within it rather than abandoning the pipeline.
    """
    p = EscapingPrompter([
        "audio", *browse_to(media), ["transcribe"], False,
        "First name", "__esc__",    # Esc at "What next?"
        "Renamed", "save",          # asked to name it again
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    named = [q for _kind, q in p.asked if q == "Name this pipeline:"]
    assert len(named) == 2, "Esc did not return to the naming question"
    assert result.preset["meta"]["title"] == "Renamed"
    assert p.asked.count(("select", "What next?")) == 2


def test_the_paragraphs_of_a_description_are_kept_apart():
    """
    A docstring marks its own paragraph breaks and the screen used to throw
    them away, printing them back to back. A two-paragraph description then
    arrived as one unbroken slab -- the "blocks of text running together" the
    user reported.
    """
    p = ScriptedPrompter([])
    spec = ParamSpec(name="model", annotation="str", default="tiny",
                     required=False, kw_only=True,
                     desc="First paragraph, which says one thing.\n\n"
                          "Second paragraph, which says another.",
                     widget="text", choices=())
    wiz._explain(p, spec, "tiny")

    out = p.output
    first = out.index("    First paragraph, which says one thing.")
    second = out.index("    Second paragraph, which says another.")
    assert "" in out[first + 1:second], "the paragraph break was dropped"


def test_the_current_value_gets_its_own_line():
    """
    "Leave it alone" is usually the right answer, so the present value is the
    fact that decides it. It used to be the tail of a dim parenthesis at the
    end of the prose, in the prose's own style -- the thing you most needed to
    see was the hardest thing on screen to find.
    """
    p = ScriptedPrompter([])
    spec = ParamSpec(name="mode", annotation="str", default="concat",
                     required=False, kw_only=True, desc="Some prose.",
                     widget="choice", choices=("concat", "separate"))
    wiz._explain(p, spec, "concat")

    assert "    Now: concat" in p.output
    assert "    One of: concat, separate" in p.output
    # on its own line, not tacked onto the prose
    assert not any("Some prose" in line and "concat" in line for line in p.output)


# ---------------------------------------------------------------------------
# backing out of the review stage
#
# reported twice, as one bug each way round: Esc at "Name this pipeline" "backs
# up to an odd place", and "hitting esc on that screen takes us back to 'name
# this pipeline'". three faults piled up here. `ask_tuning` built its overrides
# fresh on every call, so stepping back past it discarded every setting the
# user had made; it re-asked the yes/no gate rather than reopening the list
# they were working in; and it *returned* on Esc instead of raising, so Esc
# there went forward to naming and the two screens bounced with no way out
# ---------------------------------------------------------------------------


def test_backing_out_of_naming_keeps_the_settings_already_changed(media, tmp_path):
    """The silent half of the bug: the work was thrown away on the way past."""
    p = EscapingPrompter([
        "audio", *browse_to(media), ["transcribe"],
        True, "transcribe", "whisper_model", "small.en", ":done", ":done",
        "__esc__",                  # Esc while naming -> back to options
        ":done",                    # ...the list is open; leave it alone
        "Kept", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["vars"]["whisper_model"] == "small.en", (
        "backing out of naming discarded the settings"
    )


def test_backing_out_of_naming_reopens_the_list_not_the_yes_no_gate(media, tmp_path):
    """
    "An odd place": the gate is a question they have already answered, and it
    sits one screen further from the list they were actually working in.
    """
    p = EscapingPrompter([
        "audio", *browse_to(media), ["transcribe"],
        True, "transcribe", "whisper_model", "small.en", ":done", ":done",
        "__esc__",
        ":done",
        "Kept", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    asked = [q for _kind, q in p.asked]
    gates = [q for q in asked if q == "Change any settings first?"]
    assert len(gates) == 1, "the gate was asked again on the way back"

    # whatever comes straight after the Esc is where the Esc landed
    after = asked[asked.index("Name this pipeline:") + 1]
    assert after == "What would you like to change?", (
        f"backing out of naming landed on {after!r}"
    )


def test_escape_out_of_the_options_screen_goes_back_to_features(media, tmp_path):
    """
    The other half: Esc at the top of the options screen used to carry *on* to
    naming, so Esc there and Esc at naming bounced against each other. Back
    from options is the features stage -- one stage backwards, like everywhere.
    """
    p = EscapingPrompter([
        "audio", *browse_to(media), ["transcribe"],
        False,                      # no settings changes
        "__esc__",                  # Esc while naming -> back to options
        "__esc__",                  # Esc at the options gate -> back to features
        ["transcribe"],             # asked what we want again
        False, "Second time", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    features = [q for kind, q in p.asked
                if q.startswith("Which features do you want to extract")]
    assert len(features) == 2, "Esc out of options did not reach the features stage"
    assert result.preset["meta"]["title"] == "Second time"


def test_the_pipeline_is_not_rebuilt_on_the_way_back_from_naming(media, tmp_path):
    """
    `preflight` asks whether to keep a step whose extra is missing. Redoing the
    derived work on every pass would ask that again on the way back from
    naming, which would be its own small mystery.
    """
    calls = []
    real = wiz.preflight
    try:
        wiz.preflight = lambda prompter, steps: (calls.append(1), real(prompter, steps))[1]
        p = EscapingPrompter([
            "audio", *browse_to(media), ["transcribe"],
            False, "__esc__", False, "Named", "save",
        ])
        wiz.run_wizard(p, cwd=tmp_path)
    finally:
        wiz.preflight = real

    assert len(calls) == 1, f"preflight ran {len(calls)} times"


def test_the_transcript_question_says_why_it_is_being_asked(media, tmp_path):
    """
    Ticking readability alone needs a transcript, transitively, without ever
    mentioning one. The question is a surprise unless it explains itself -- and
    a `reason` is what puts that explanation next to the question rather than
    at the top of the screen in the same gray as everything else.
    """
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["readability"],
        "transcribe",               # the question under test
        "speaker", False, "Named", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert any("needs a transcript" in r for r in p.reasons), p.reasons


def test_a_missing_folder_explains_itself_before_asking_to_try_again(media, tmp_path):
    """Same treatment for every question that only exists because something went
    wrong -- otherwise "Try a different folder or file type?" is a non-sequitur."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    p = ScriptedPrompter([
        "audio", *browse_to(empty),
        False,                      # don't try again
    ])
    with pytest.raises(Cancelled):
        wiz.run_wizard(p, cwd=tmp_path)

    assert any("No audio files under" in r for r in p.reasons), p.reasons


def test_a_spreadsheet_is_asked_how_its_rows_should_be_counted(tmp_path, survey_csv):
    """
    Asked outright rather than inferred. The old flow guessed from the id
    column, and naming one whose values repeat gave several rows the same
    `text_id` *without* combining them -- so the id identified nothing, and the
    sentence-embedding merge averaged those rows while every other feature left
    them apart.
    """
    p = ScriptedPrompter([
        "csv", *browse_to(survey_csv), ["response"], False,
        ["readability"], "group", ["pid"], [], False, "Survey", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("what should one row of results describe" in q.lower()
               for _kind, q in p.asked)
    assert result.preset["steps"][0]["with"]["group_by"] == ["pid"]


def test_choosing_per_row_never_sets_a_grouping(tmp_path, survey_csv):
    p = ScriptedPrompter([
        "csv", *browse_to(survey_csv), ["response"], True, ["pid"],
        ["readability"], "row", [], False, "Survey", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    block = result.preset["steps"][0]["with"]
    assert "group_by" not in block
    assert block["id_cols"] == ["pid"]


def test_asking_to_group_by_nothing_is_re_asked(tmp_path, survey_csv):
    """Empty is not an answer here: it would silently fall back to per-row."""
    p = ScriptedPrompter([
        "csv", *browse_to(survey_csv), ["response"], False,
        ["readability"], "group", [], ["pid"], [], False, "Survey", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["steps"][0]["with"]["group_by"] == ["pid"]


def test_a_tab_separated_file_is_read_as_columns(tmp_path):
    """
    The picker offers .tsv. Read with a comma reader it yields one column whose
    name is the whole header line, which the wizard then offers as the text.
    """
    path = tmp_path / "posts.tsv"
    path.write_text("pid\tbody\np01\thello there\n", encoding="utf-8")

    assert wiz._read_header(path) == ["pid", "body"]
    assert wiz._delimiter_of(path) == "\t"


def test_a_tab_separated_file_carries_its_delimiter_into_the_preset(tmp_path):
    path = tmp_path / "posts.tsv"
    path.write_text("pid\tbody\np01\thello there\n", encoding="utf-8")

    p = ScriptedPrompter([
        "csv", *browse_to(path), ["body"], False,
        ["readability"], "row", [], False, "Tabbed", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset["vars"]["csv_delimiter"] == "\t"


# ---------------------------------------------------------------------------
# dependent settings: under their gate, only shown when they apply, never
# wrongly hidden
# ---------------------------------------------------------------------------

def _stats_setup():
    """The four analyses composed on a spreadsheet, and their variables."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    ids = ["stats_group_differences", "stats_correlations", "stats_ridge_fit",
           "stats_classify_fit"]
    preset = compose(["readability", *ids], source="csv", input_path="s.csv",
                     text_cols=["text"], id_cols=["pid"], name="x",
                     var_values={"stats_group_col": "condition",
                                 "stats_outcome_cols": ["openness"],
                                 "stats_class_cols": ["condition"]})
    return [_r.by_id(i) for i in ids], preset["meta"]["variables"]


def _rows(recipe, var_specs, overrides=None, var_values=None):
    from taters.ui import wizard as w

    spec = w._spec_for(recipe)
    return w.step_rows(recipe, spec, var_specs, overrides or {}, var_values or {})


from taters.ui.recipes import RECIPES  # noqa: E402


@pytest.mark.parametrize("recipe", [r for r in RECIPES], ids=lambda r: r.id)
def test_every_setting_a_step_offers_is_on_its_menu(recipe):
    """Catalog-wide: take the gated rows out and the menu is exactly the
    editable list, everyday first -- so no setting can go missing from a
    menu by accident, whatever the gates do."""
    from taters.ui import recipes as _r
    from taters.ui import wizard as w
    from taters.ui.compose import compose

    spec = w._spec_for(recipe)
    if spec is None:
        pytest.skip(f"{recipe.id} is not installed here")
    everyday, rest = w._editable_names(recipe, spec)
    var_specs = {**{k: dict(v) for k, v in compose(
        ["readability"], source="csv", input_path="s.csv", text_cols=["text"],
        name="x")["meta"]["variables"].items()}, **recipe.vars}
    listed = [c.value for c in w.step_rows(recipe, spec, var_specs, {}, {})]
    ungated = [n for n in everyday + rest if _r.gate_of(recipe, n) is None]
    assert [n for n in listed if _r.gate_of(recipe, n) is None] == ungated
    assert set(ungated) <= set(listed)


def test_every_gate_hides_and_reveals_its_dependents():
    """For every declared gate in the catalog: the dependent row sits
    indented directly under its gate when the gate holds, and is absent
    when it does not."""
    from taters.ui import recipes as _r
    from taters.ui import wizard as w

    seen = 0
    for recipe in RECIPES:
        if not recipe.param_when:
            continue
        spec = w._spec_for(recipe)
        if spec is None:
            continue
        var_specs = {k: dict(v) for k, v in recipe.vars.items()}
        for name in recipe.param_when:
            gates = _r.gates_of(recipe, name)

            def _state(gate, satisfy):
                _p, op, value = gate
                if satisfy:
                    return value if op == "==" else "__anything_else__"
                return f"not-{value}" if op == "==" else value

            def _apply(break_at=None):
                """Satisfy every condition, then break one on top -- on top,
                because two conditions can name the same gate and the later
                write would otherwise undo the break."""
                overrides, var_values = {}, {}
                order = list(range(len(gates)))
                if break_at is not None:
                    order = [i for i in order if i != break_at] + [break_at]
                for i in order:
                    gate = gates[i]
                    state = _state(gate, satisfy=(i != break_at))
                    var = w._var_behind(recipe, gate[0])
                    if var:
                        var_values[var] = state
                    else:
                        overrides.setdefault(recipe.id, {})[gate[0]] = state
                return overrides, var_values

            # every condition satisfied shows the row; breaking any one hides it
            cases = [(_apply(), True)] + [(_apply(i), False)
                                          for i in range(len(gates))]
            for (overrides, var_values), present in cases:
                gate_param, _op, state = gates[0][0], None, None
                rows = w.step_rows(recipe, spec, var_specs, overrides, var_values)
                values = [c.value for c in rows]
                assert (name in values) is present, (recipe.id, name, state)
                assert len(values) == len(set(values)), "a row listed twice"
                if present and gate_param not in values:
                    # a gate the menu never shows (`feature_sets` gets answered
                    # by the analysis stage) still hides its dependents, it
                    # just has no row for them to sit under
                    seen += 1
                    continue
                if present:
                    at = values.index(name)
                    assert rows[at].label.startswith(w.INDENT), (recipe.id, name)
                    gate_at = values.index(gate_param)
                    between = values[gate_at + 1:at]
                    assert all(_r.gate_of(recipe, b) and _r.gate_of(recipe, b)[0]
                               == gate_param for b in between), \
                        f"{name} is not directly under {gate_param}"
                    seen += 1
    assert seen >= 6, "the catalog should have gated settings to check"


def test_an_unknown_gate_fails_open():
    """A gate the step's signature does not have cannot hide anything."""
    from dataclasses import replace

    steps, var_specs = _stats_setup()
    recipe = replace(steps[1], param_when={"pca_components": ("no_such", "x")})
    assert "pca_components" in [c.value for c in _rows(recipe, var_specs)]


def test_pca_settings_appear_only_when_pca_is_on():
    from taters.ui import wizard as w

    steps, var_specs = _stats_setup()
    for recipe in steps:
        off = [c.value for c in _rows(recipe, var_specs)]
        assert "pca" in off
        assert "pca_components" not in off and "pca_rotation" not in off
        on = _rows(recipe, var_specs, var_values={"stats_pca": "all"})
        values = [c.value for c in on]
        assert values.index("pca_components") == values.index("pca") + 1
        assert on[values.index("pca_components")].label.startswith(w.INDENT)
        if "pca_max_missing" in recipe.param_when:
            assert "pca_max_missing" in values and "pca_max_missing" not in off


def test_the_shared_section_and_the_step_menus_agree():
    """The same gate, the same answer, on both menus -- the shared section
    used to show the component count while every step hid it."""
    from taters.ui import wizard as w

    steps, var_specs = _stats_setup()
    shared = w.shared_variables(steps)
    for pca, expect in (("off", False), ("all", True)):
        var_values = {"stats_pca": pca}
        shared_vars = [var for var, _r_, _p, _c in w.shared_rows(
            shared, var_specs, {}, var_values)]
        assert ("stats_pca_components" in shared_vars) is expect
        assert ("stats_pca_rotation" in shared_vars) is expect
        for recipe in steps:
            step = [c.value for c in _rows(recipe, var_specs, var_values=var_values)]
            assert ("pca_components" in step) is expect
        if expect:
            assert shared_vars.index("stats_pca_components") == \
                shared_vars.index("stats_pca") + 1
            rows = w.shared_rows(shared, var_specs, {}, var_values)
            choice = next(c for v, _r_, _p, c in rows if v == "stats_pca_components")
            assert choice.label.startswith(w.INDENT)


def test_a_steps_own_word_for_a_setting_cannot_leak_onto_a_shared_row():
    """
    A step's `labels` are keyed by its *parameter*; the shared section keys
    its rows by the *variable*. Where those two names differ, looking the
    label up by the variable read one step's private word for one of its
    parameters onto a row about something else entirely.

    The sweep is the case that has both: `engine` is which topic model to
    sweep, and `engine_nlp` is the shared tagging engine. The shared row
    said "which topic model — nltk".
    """
    from dataclasses import replace

    from taters.ui import recipes as _r
    from taters.ui import wizard as w

    # a step that calls one of its own parameters something private, mapped to
    # a shared variable of a different name. Built here rather than taken from
    # the catalog: the step that used to need this was folded into the topic
    # models, and a regression test for a real bug should not disappear with
    # whichever recipe happened to exercise it.
    owner = _r.by_id("parts_of_speech")
    # this step calls its own `engine` parameter something private
    owner = replace(owner, labels={"engine": "which topic model"})
    spec = w._spec_for(owner, None)

    # the shared row is for a *different* parameter that happens to read a
    # variable named `engine`. Looking the label up by the variable rather
    # than the parameter put this step's private word on that row.
    shared = w._setting_choice(owner, spec.get("tokenizer"), {}, {}, {},
                               key="engine")
    assert "topic model" not in shared.label, (
        "a step's private word for its own parameter leaked onto the shared "
        f"row for a different one: {shared.label!r}")
    assert shared.label.startswith("tagging engine")

    # while the step's own row still gets the word it asked for
    own = w._setting_choice(owner, spec.get("engine"), {}, {}, {})
    assert own.label.startswith("which topic model")


# ---------------------------------------------------------------------------
# picked, not typed
# ---------------------------------------------------------------------------

def _typed(p):
    """The free-text questions a scripted run was asked."""
    return [q for kind, q in p.asked if kind == "text"]


def test_pca_offers_the_run_s_feature_sets_instead_of_a_box_to_spell_them_in():
    """
    "off, all, or the name of a feature set" was a text box, and the names it
    wanted were file stems the user had never seen (a real report: "I had to
    type the name of a feature set, or all, or none"). The tables this run
    will make are known, so the third answer opens a tick list of them and
    the value stored is exactly the names the analyses match on.
    """
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    ids = ["dictionaries", "readability", "stats_correlations"]
    steps = [_r.by_id(i) for i in ids]
    var_specs = compose(ids, source="csv", input_path="s.csv", text_cols=["text"],
                        name="x", var_values={"stats_outcome_cols": ["o"]}
                        )["meta"]["variables"]
    p = ScriptedPrompter(["stats_correlations", "pca", ":some", ["dictionary"],
                          ":done", ":done"])
    _o, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False,
                                    source="csv", selected=ids)
    assert var_values["stats_pca"] == ["dictionary"]
    assert not _typed(p)
    offered = next(cs for q, cs in p.offered if q.startswith("Which feature sets"))
    assert [c.value for c in offered] == ["dictionary", "readability"]
    # the step's own name is on the row too: "dictionary" alone would leave
    # somebody guessing which step wrote it
    assert offered[0].label.startswith(_r.by_id("dictionaries").label)


def test_pca_off_and_all_are_the_first_two_rows():
    """The two plain answers stay one keypress away, and the tick list only
    opens for "some" -- with nothing ticked reading as off, said aloud."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    ids = ["readability", "stats_correlations"]
    steps = [_r.by_id(i) for i in ids]
    var_specs = compose(ids, source="csv", input_path="s.csv", text_cols=["text"],
                        name="x", var_values={"stats_outcome_cols": ["o"]}
                        )["meta"]["variables"]
    p = ScriptedPrompter(["stats_correlations", "pca", "all",
                          "pca", ":some", [], ":done", ":done"])
    _o, var_values = wiz.ask_tuning(p, steps, var_specs, ask_gate=False,
                                    source="csv", selected=ids)
    menu = next(cs for q, cs in p.offered if q == "pca:")
    assert [c.value for c in menu][:2] == ["off", "all"]
    assert var_values["stats_pca"] == "off"
    assert "stays off" in p.text_output


def test_unverified_ok_is_ticked_from_the_feature_tables():
    """A list of table names is the same kind of answer as `pca`'s: tick the
    tables, never spell their stems."""
    from taters.ui import recipes as _r
    from taters.ui.compose import compose

    steps = [_r.by_id("dictionaries"), _r.by_id("score_with_model")]
    var_specs = compose(["dictionaries"], source="csv", input_path="s.csv",
                        text_cols=["text"], name="x")["meta"]["variables"]
    p = ScriptedPrompter(["score_with_model", "unverified_ok", ["dictionary"],
                          ":done", ":done"])
    overrides, _v = wiz.ask_tuning(p, steps, var_specs, ask_gate=False,
                                   source="csv")
    assert overrides["score_with_model"]["unverified_ok"] == ["dictionary"]
    assert not _typed(p)


def test_every_table_shaped_setting_in_the_catalog_has_a_picker():
    """
    The pickers are keyed by parameter name. A new step that took a `pca`
    or `unverified_ok` under another meaning, or a new table-valued setting
    under a new name, would get the wrong screen or a text box -- so the
    names in the registry are pinned to the steps that have them.
    """
    holders = {}
    for recipe in RECIPES:
        spec = wiz._spec_for(recipe)
        if spec is None:
            continue
        for name in wiz._TABLE_PICKERS:
            if spec.get(name) is not None:
                holders.setdefault(name, set()).add(recipe.id)
    assert holders["pca"] == {"stats_correlations", "stats_group_differences",
                              "stats_ridge_fit", "stats_classify_fit"}
    assert holders["unverified_ok"] == {"score_with_model"}


@pytest.mark.parametrize("recipe_id,param", [
    ("stats_assemble", "bookkeeping"),
    ("stats_classify_fit", "control_combos"),
    ("stats_ridge_fit", "control_combos"),
    ("topic_model_mem", "engine"),
    ("topic_model_mem", "tokenizer"),
    ("topic_model_mem", "vocab_rule"),
    ("topic_model_mem", "vocab_rank_by"),
    ("transcribe", "whisper_model"),
    ("transcribe", "compute_type"),
    ("whisper_embeddings", "model_name"),
    ("whisper_embeddings", "compute_type"),
])
def test_a_setting_with_a_known_set_of_answers_is_a_picker(recipe_id, param):
    """
    Each of these asked to be typed: `none_and_all`, `min_obs_pct`,
    `int8_float16`. Their answers were always a short list -- in the
    signature or in the prose -- and a list is picked from.
    """
    from taters.ui.recipes import by_id as _by_id

    spec = wiz._spec_for(_by_id(recipe_id))
    if spec is None:
        pytest.skip(f"{recipe_id} is not installed here")
    p = spec.get(param)
    assert p is not None, f"{recipe_id} no longer has {param}"
    assert p.widget == "choice" and p.choices, (recipe_id, param)


def test_an_open_set_keeps_a_row_for_typing_something_else():
    """
    The stock Whisper models are a list; a model folder on disk is not on
    it. "something else" is the last row, and opens the text box with the
    current value -- so the list costs nothing that typing could do.
    """
    p, (_o, var_values) = _tuning([True, "transcribe", "whisper_model",
                                   ":other", "models/my-ct2", ":done", ":done"])
    menu = next(cs for q, cs in p.offered if q == "whisper model:")
    assert menu[-1].value == wiz._OTHER
    assert "small.en" in {c.value for c in menu}
    assert var_values["whisper_model"] == "models/my-ct2"
    assert _typed(p) == ["whisper model:"]


def test_a_setting_that_may_stay_unset_offers_automatic_as_a_row():
    """
    `compute_type` is None by default and the code then picks float16 or
    int8 for the machine. From a list there is no other way to say "leave
    it" -- so it is the first row, and choosing it stores None, not "".
    """
    p, (overrides, _v) = _tuning([True, "transcribe", "compute_type", "float32",
                                  "compute_type", "", ":done", ":done"])
    menu = next(cs for q, cs in p.offered if q == "compute type:")
    assert menu[0].value == "" and "automatic" in menu[0].label
    assert menu[-1].value != wiz._OTHER          # a closed set: no typing row
    assert overrides["transcribe"]["compute_type"] is None
    assert not _typed(p)


def test_turning_translation_on_moves_off_an_english_only_model():
    """
    Asking for a translation *is* asking for a multilingual model: an
    English-only `.en` model cannot translate, and a pipeline saying both
    would only fail at run time. The model follows, and the screen says so.
    The reverse holds too: picking an `.en` model turns translation off.
    """
    p, (_o, var_values) = _tuning([True, "transcribe", "translate", True,
                                   ":done", ":done"])
    assert var_values["translate"] is True
    assert var_values["whisper_model"] == "base"
    assert "English-only and cannot translate" in p.text_output

    p, (_o, var_values) = _tuning([True, "transcribe", "translate", True,
                                   "whisper_model", "small.en", ":done", ":done"])
    assert var_values["whisper_model"] == "small.en"
    assert var_values["translate"] is False
    assert "translate switched off" in p.text_output


def test_translate_is_a_yes_no_on_the_transcript_menu():
    """The switch the request asked for: a true/false the user can set under
    the transcript step, not a variable only a hand-written preset reaches."""
    from taters.ui.recipes import by_id as _by_id

    recipe = _by_id("transcribe")
    spec = wiz._spec_for(recipe)
    if spec is None:
        pytest.skip("faster-whisper is not installed here")
    rows = [c.value for c in _rows(recipe, {}, {}, {})]
    assert "translate" in rows
    assert spec.get("translate").widget == "bool"
    # among the everyday settings the recipe wired, not one keypress further in
    # "the rest", since it's the kind of decision somebody makes about a run
    everyday, _rest = wiz._editable_names(recipe, spec)
    assert "translate" in everyday
