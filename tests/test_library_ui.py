"""
The library on screen: the manager in Settings, and the picker in the wizard.

The storage itself is covered in test_library.py; here the subject is the
conversation -- what lands in a preset when someone ticks two of three
dictionaries, what an empty library says, and that "Manage…" mid-pick is the
same flow Settings opens.

Every test runs under the hermetic ``TATERS_HOME`` from conftest, so "the
library" always starts empty and holds exactly what the test imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taters.helpers import library as lib
from taters.ui import wizard as wiz
from taters.ui.library import manage_library, pick_from_library
from taters.ui.prompts import Choice, ScriptedPrompter  # noqa: F401
from taters.ui.tasks import TaskContext
from taters.ui.tasks import library as library_task

# browsing helpers we share with the wizard tests.
from wizard_helpers import EscapingPrompter, browse_to


VALID_DIC = "%\n1\tCats\n%\ncat\t1\n"   # the smallest .dic that's still LIWC-shaped


@pytest.fixture()
def dicts():
    return lib.KINDS["dictionaries"]


@pytest.fixture()
def three(tmp_path, dicts):
    """Three imported dictionaries; returns their library paths, sorted."""
    for name in ("anger.dicx", "joy.dic", "mood.csv"):
        f = tmp_path / name
        # .dic files have to be LIWC-shaped now -- the import gate refuses bare
        # word lists (we had a run die on one).
        f.write_text(VALID_DIC if name.endswith(".dic") else name,
                     encoding="utf-8")
        lib.import_into(dicts, f)
    return lib.entries(dicts)


def _key(entry: Path) -> str:
    return str(entry.resolve())


# ---------------------------------------------------------------------------
# The picker
# ---------------------------------------------------------------------------

def test_everything_starts_ticked_when_nothing_was_chosen_before(dicts, three):
    # enter on an entry approves the selection; with everything ticked the
    # union adds nothing, so a bare enter means "use them all".
    p = ScriptedPrompter([_key(three[0])])
    picked = pick_from_library(p, dicts)

    assert picked == sorted(_key(e) for e in three)
    labels = [c.label for c in p.offered_choices("Which content-coding")]
    assert all(label.startswith("[x] ") for label in labels
               if label.startswith("["))


def test_space_unticks_one_and_enter_approves_the_rest(dicts, three):
    p = ScriptedPrompter(["\x00space:" + _key(three[1]), _key(three[0])])

    picked = pick_from_library(p, dicts)

    assert picked == sorted(_key(e) for e in (three[0], three[2]))


def test_select_none_then_enter_on_one_returns_exactly_that_one(dicts, three):
    p = ScriptedPrompter([":none", _key(three[0])])

    assert pick_from_library(p, dicts) == [_key(three[0])]


def test_use_all_of_them_approves_outright(dicts, three):
    """
    Reported: "Select all" ticked things and relooped, which -- with
    everything already ticked -- changed nothing visible and read as a broken
    blink. Its intent is "use all of them", so now it approves outright.
    """
    p = ScriptedPrompter([
        "\x00space:" + _key(three[0]), "\x00space:" + _key(three[1]),
        ":all",
    ])

    assert pick_from_library(p, dicts) == sorted(_key(e) for e in three)
    row = next(c for _q, cs in p.offered for c in cs if c.value == ":all")
    assert row.tone == "good" and "Use all" in row.label


def test_untick_everything_swaps_in_place_without_a_reloop(dicts, three):
    """"Untick everything" flips the marks inside the running prompt (the
    browser's no-flash swap); the scripted prompter records the swapped
    listing under the same, single select."""
    p = ScriptedPrompter([":none", _key(three[0])])
    picked = pick_from_library(p, dicts)

    assert picked == [_key(three[0])]
    listings = [cs for q, cs in p.offered if q.startswith("Which")]
    assert len(listings) == 2, "the untick must swap, not exit and re-ask"
    unticked = [c.label for c in listings[1] if c.label.startswith("[")]
    assert all(label.startswith("[ ] ") for label in unticked)
    # here's how we tell a swap from an exit-and-reloop: a reloop re-asks with
    # the pointer parked on ":none"; a swap never calls select again.
    question = next(q for q in p.select_defaults if q.startswith("Which"))
    assert p.select_defaults[question] != ":none"


def test_enter_approves_the_ticked_plus_the_pointed(dicts, three):
    """The import browser's union: the pointed entry counts as chosen, which
    is also what makes an empty approval impossible -- a step reading zero
    dictionaries fails minutes into a run, and "none of these" is Esc."""
    p = ScriptedPrompter([":none", "\x00space:" + _key(three[1]),
                          _key(three[2])])

    assert pick_from_library(p, dicts) == sorted(
        _key(e) for e in (three[1], three[2]))


def test_a_folder_in_current_opens_with_everything_ticked(dicts, three):
    """
    The reported bug: an untouched pipeline's saved preset carries the whole
    library *folder* as its value, and the picker intersected that one path
    against entry file paths -- the settings row said "all 10 selected" and
    the screen opened with ten empty boxes. A folder means everything in it,
    exactly as the analyzers read it.
    """
    p = ScriptedPrompter([_key(three[0])])

    picked = pick_from_library(p, dicts, current=[str(lib.kind_dir(dicts))])

    assert picked == sorted(_key(e) for e in three)
    labels = [c.label for c in p.offered_choices("Which content-coding")]
    assert all(label.startswith("[x] ") for label in labels
               if label.startswith("["))


def test_the_pointer_starts_on_the_first_ticked_entry(dicts, three):
    """A bare enter must approve the selection exactly as it stands, so the
    pointer cannot start on an unticked row the union would then add."""
    p = ScriptedPrompter([_key(three[1])])
    pick_from_library(p, dicts, current=[_key(three[1])])

    question = next(q for q in p.select_defaults if q.startswith("Which"))
    assert p.select_defaults[question] == _key(three[1])


def test_escape_keeps_whatever_it_was(dicts, three):
    p = EscapingPrompter(["__esc__"])

    assert pick_from_library(p, dicts, current=[_key(three[0])]) is None


def test_manage_from_the_picker_offers_the_new_import_immediately(
        dicts, three, tmp_path):
    """The point of sharing the flow: import mid-pick, and it is just there."""
    new = tmp_path / "surprise.dicx"
    new.write_text("x", encoding="utf-8")
    p = ScriptedPrompter([
        ":manage",
        ":import", "\x00type", str(new),        # typing a file IS choosing it
        ":back",                                 # leave the manager
        ":none",
        _key(lib.kind_dir(dicts) / "surprise.dicx"),    # enter approves it
    ])

    picked = pick_from_library(p, dicts)

    assert picked == [str((lib.kind_dir(dicts) / "surprise.dicx").resolve())]


def test_an_empty_library_explains_and_offers_an_import(dicts, tmp_path):
    new = tmp_path / "first.dicx"
    new.write_text("x", encoding="utf-8")
    p = ScriptedPrompter([
        "import", "\x00type", str(new),         # typing a file IS choosing it
        _key(lib.kind_dir(dicts) / "first.dicx"),   # a normal pick: approve it
    ])

    picked = pick_from_library(p, dicts)

    assert picked == [str((lib.kind_dir(dicts) / "first.dicx").resolve())]
    assert any("no content-coding dictionaries yet" in r.lower()
               for r in p.reasons)


def test_only_the_models_screen_offers_the_hugging_face_import(dicts):
    from taters.ui import library as ui_lib

    p = ScriptedPrompter([ui_lib._BACK])
    manage_library(p, dicts)
    values = [c.value for _q, cs in p.offered for c in cs]
    assert ui_lib._IMPORT_HF not in values, "a dictionary is not a Hugging Face model"

    p = ScriptedPrompter([ui_lib._BACK])
    manage_library(p, lib.KINDS["models"])
    values = [c.value for _q, cs in p.offered for c in cs]
    assert ui_lib._IMPORT_HF in values


def test_a_classifier_folder_is_imported_named_and_lands_in_the_library(tmp_path):
    """
    The Settings route end to end: pick "a checkpoint folder", browse to it,
    say what it predicts and what to call it, and the model is in the
    library with the columns it will write. A user who downloaded a
    sentiment model for something else should be scoring with it a minute
    later.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_encoder import build_classifier

    from taters.helpers.model_spec import describe
    from taters.ui import library as ui_lib

    folder = build_classifier(tmp_path / "sent", labels=("neg", "pos"))
    p = ScriptedPrompter([
        ui_lib._IMPORT_HF,
        ui_lib._HF_FOLDER, *browse_to(folder),
        "sentiment",            # what it predicts: the column stem
        "twitter_sentiment",    # what to call it
        ui_lib._BACK,
    ])
    manage_library(p, lib.KINDS["models"])
    entries = lib.entries(lib.KINDS["models"])
    assert [e.name for e in entries] == ["twitter_sentiment.json"]
    info = describe(entries[0])
    assert info.type_id == "hf_classifier"
    assert info.columns == ("pred_sentiment", "prob_sentiment", "p_sentiment_neg",
                            "p_sentiment_pos")
    assert (entries[0].parent / "twitter_sentiment.hfmodel" / "config.json").is_file(), \
        "the checkpoint traveled into the library"
    assert any("adds pred_sentiment" in n for n in p.output)


def test_a_bare_encoder_offered_as_a_classifier_is_refused_on_the_spot(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tiny_encoder import build

    from taters.ui import library as ui_lib

    folder = build(tmp_path / "enc")
    p = ScriptedPrompter([ui_lib._IMPORT_HF, ui_lib._HF_FOLDER, *browse_to(folder),
                          ui_lib._BACK])
    manage_library(p, lib.KINDS["models"])
    assert lib.entries(lib.KINDS["models"]) == []
    assert any("Train a model" in n for n in p.output), "the refusal says where it belongs"


def test_backing_out_of_an_empty_library_changes_nothing(dicts):
    p = ScriptedPrompter([":back"])

    assert pick_from_library(p, dicts) is None


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------

def test_rename_and_delete_through_the_manager(dicts, three):
    p = ScriptedPrompter([
        "anger.dicx", ":rename", "fury",        # tick it, rename the tick
        "fury.dicx", ":delete", True,           # tick it, delete the tick
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.stem for e in lib.entries(dicts)] == ["joy", "mood"]


def test_export_through_the_manager(dicts, three, tmp_path):
    out = tmp_path / "exported"
    out.mkdir()
    p = ScriptedPrompter([
        "joy.dic", ":export", "\x00type", str(out),
        ":back",
    ])
    manage_library(p, dicts)

    assert (out / "joy.dic").read_text(encoding="utf-8") == VALID_DIC


def test_a_colliding_import_asks_before_replacing(dicts, three, tmp_path):
    newer = tmp_path / "mood.csv"
    newer.write_text("newer words", encoding="utf-8")
    p = ScriptedPrompter([
        ":import", "\x00type", str(newer), False,   # replace? no
        ":import", "\x00type", str(newer), True,    # replace? yes
        ":back",
    ])
    manage_library(p, dicts)

    stored = next(e for e in lib.entries(dicts) if e.stem == "mood")
    assert stored.read_text(encoding="utf-8") == "newer words"


def test_the_settings_task_reaches_the_same_manager(dicts, three, tmp_path):
    p = ScriptedPrompter([
        "dictionaries",             # which library?
        "anger.dicx", ":delete", True,  # tick, then delete the tick
        ":back", ":back",
    ])
    library_task.TASK.run(TaskContext(prompter=p, cwd=Path(tmp_path)))

    assert [e.stem for e in lib.entries(dicts)] == ["joy", "mood"]


# ---------------------------------------------------------------------------
# Through the wizard
# ---------------------------------------------------------------------------

def test_a_picked_selection_lands_in_the_preset_as_explicit_paths(
        dicts, three, media, tmp_path):
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["dictionaries"], "transcribe", "speaker",
        True, "dictionaries",
        "dict_paths",               # opens the picker, everything ticked
        "\x00space:" + _key(three[2]),  # untick mood
        _key(three[0]),                 # enter approves -> anger + joy
        ":done", ":done",
        "Two dicts", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_with_dictionaries"))
    assert step["with"]["dict_paths"] == sorted(_key(e) for e in three[:2])


def test_an_untouched_step_reads_the_whole_library_folder(
        dicts, three, media, tmp_path):
    """
    "Always available" without opening the options screen: the default is the
    kind's library folder, which the analyzer expands recursively -- so a
    dictionary imported next month is used by this pipeline next month.
    """
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["dictionaries"], "transcribe", "speaker",
        False, "Untouched", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_with_dictionaries"))
    assert step["with"]["dict_paths"] == [str(lib.kind_dir(dicts))]


def test_the_settings_row_counts_rather_than_listing_paths(
        dicts, three, media, tmp_path):
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["dictionaries"], "transcribe", "speaker",
        True, "dictionaries", ":done", ":done",
        "Counted", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    row = next(c for _q, choices in p.offered for c in choices
               if c.value == "dict_paths")
    assert "all 3 of your content-coding dictionaries" in row.label


def test_preflight_offers_an_import_when_the_library_is_empty(
        dicts, media, tmp_path):
    first = tmp_path / "first.dicx"
    first.write_text("x", encoding="utf-8")
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["dictionaries"], "transcribe", "speaker",
        "import",                       # the contingency…
        "\x00type", str(first),         # …drops straight into the browser:
                                        # no second "Import files…" menu after
                                        # the user just said "import one now"
        False, "Rescued", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert any("none imported" in r for r in p.reasons)
    assert [e.stem for e in lib.entries(dicts)] == ["first"]
    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_with_dictionaries"))
    assert step["with"]["dict_paths"] == [str(lib.kind_dir(dicts))]


def test_preflight_can_drop_the_step_instead(dicts, media, tmp_path):
    p = ScriptedPrompter([
        "audio", *browse_to(media), ["dictionaries", "readability"],
        "transcribe", "speaker",
        "drop",                         # leave dictionaries out
        False, "Dropped", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    calls = [s["call"] for s in result.preset["steps"]]
    assert not any(c.endswith("analyze_with_dictionaries") for c in calls)
    assert any(c.endswith("analyze_readability") for c in calls)


def test_the_manager_deletes_the_row_that_was_picked_not_its_twin(
        dicts, tmp_path):
    """
    From the code review (issue 1), through the real screen: two entries that
    share a stem are two distinct rows, and deleting one leaves the other.
    """
    for name, text in (("foo.csv", "the csv"), ("foo.dic", VALID_DIC)):
        f = tmp_path / name
        f.write_text(text, encoding="utf-8")
        lib.import_into(dicts, f)

    p = ScriptedPrompter(["foo.dic", ":delete", True, ":back"])
    manage_library(p, dicts)

    survivors = lib.entries(dicts)
    assert [e.name for e in survivors] == ["foo.csv"]
    assert survivors[0].read_text(encoding="utf-8") == "the csv"
    # ...and we could tell the two rows apart when they were offered.
    rows = [c.value for c in p.offered_choices(f"{dicts.label}:")]
    assert "foo.csv" in rows and "foo.dic" in rows


def test_exporting_onto_an_existing_file_asks_first(dicts, three, tmp_path):
    """From the code review (issue 10): a copy that silently overwrites is a
    delete nobody asked for."""
    out = tmp_path / "exported"
    out.mkdir()
    (out / "joy.dic").write_text("precious older copy", encoding="utf-8")

    p = ScriptedPrompter([
        "joy.dic", ":export", "\x00type", str(out), False,  # replace? no
        ":back",
    ])
    manage_library(p, dicts)
    assert (out / "joy.dic").read_text(encoding="utf-8") == "precious older copy"

    p = ScriptedPrompter([
        "joy.dic", ":export", "\x00type", str(out), True,   # replace? yes
        ":back",
    ])
    manage_library(p, dicts)
    assert (out / "joy.dic").read_text(encoding="utf-8") == VALID_DIC


def test_a_filesystem_error_is_a_note_not_a_crash(dicts, three, monkeypatch):
    """A file the OS will not release (Windows says no a lot) is that entry's
    problem, not the session's."""
    def refuse(*a, **k):
        raise PermissionError("held open by another program")

    monkeypatch.setattr(lib, "delete", refuse)
    p = ScriptedPrompter(["anger.dicx", ":delete", True, ":back"])

    manage_library(p, dicts)          # this has to return, not raise

    assert any("held open by another program" in line for line in p.output)


def test_several_files_import_in_one_trip(dicts, tmp_path):
    """
    Requested directly, twice: dictionaries arrive as folders of related
    files, and both earlier shapes (one browse per file; folder-then-checkbox)
    hid the files at the moment they mattered. The browser IS the selector:
    files are toggle rows and the selection survives navigation.
    """
    src = tmp_path / "liwc-set"
    src.mkdir()
    for name in ("affect.dicx", "social.dicx", "cognition.dic"):
        (src / name).write_text(
            VALID_DIC if name.endswith(".dic") else name, encoding="utf-8")
    (src / "README.txt").write_text("not a dictionary", encoding="utf-8")

    p = ScriptedPrompter([
        ":import", "\x00type", str(src),                # navigate in
        "\x00space:" + str(src / "affect.dicx"),        # space ticks in place
        str(src / "cognition.dic"),                      # enter: proceed w/ both
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.name for e in lib.entries(dicts)] == ["affect.dicx",
                                                    "cognition.dic"]
    # ...and the browser never offered up the README as a tickable row.
    rows = next(cs for q, cs in p.offered if "[enter] imports" in q)
    assert not any("README" in c.label for c in rows)


def test_several_entries_export_in_one_trip(dicts, three, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    p = ScriptedPrompter([
        "anger.dicx", "mood.csv",               # tick two
        ":export", "\x00type", str(out),        # one destination for both
        ":back",
    ])
    manage_library(p, dicts)

    assert sorted(f.name for f in out.iterdir()) == ["anger.dicx", "mood.csv"]


def test_a_collision_mid_batch_only_costs_that_file(dicts, tmp_path):
    """One clash must not abort the rest of the batch."""
    src = tmp_path / "set"
    src.mkdir()
    (src / "a.dicx").write_text("new a", encoding="utf-8")
    (src / "b.dicx").write_text("new b", encoding="utf-8")
    old = tmp_path / "a.dicx"
    old.write_text("old a", encoding="utf-8")
    lib.import_into(dicts, old)

    p = ScriptedPrompter([
        ":import", "\x00type", str(src),
        "\x00space:" + str(src / "a.dicx"),
        str(src / "b.dicx"),            # enter: proceed with a + b
        False,                          # keep the existing a.dicx
        ":back",
    ])
    manage_library(p, dicts)

    stored = {e.name: e.read_text(encoding="utf-8") for e in lib.entries(dicts)}
    assert stored == {"a.dicx": "old a", "b.dicx": "new b"}
    assert any("Imported 1 of 2" in line for line in p.output)


def test_a_selection_can_span_folders_in_one_trip(dicts, tmp_path):
    """The point of the browser being the selector: ticks survive walking, so
    a set spread over a whole DICTIONARIES tree is still one import."""
    a = tmp_path / "liwc"
    b = tmp_path / "nrc"
    a.mkdir(); b.mkdir()
    (a / "affect.dicx").write_text("a", encoding="utf-8")
    (b / "color.csv").write_text("b", encoding="utf-8")

    p = ScriptedPrompter([
        ":import",
        "\x00type", str(tmp_path),                     # land above both
        str(a), "\x00space:" + str(a / "affect.dicx"), # walk in, space-tick
        "\x00up",                                      # walk back out
        str(b), str(b / "color.csv"),                  # walk in, enter: go
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.name for e in lib.entries(dicts)] == ["affect.dicx", "color.csv"]
    # no "Import N files" row to hunt for any more: enter on a file just proceeds.
    assert not any(c.value == "\x00choose"
                   for _q, cs in p.offered for c in cs)


def test_several_dictionaries_delete_with_one_confirm(dicts, three):
    """
    Requested directly: "I don't want to have to delete them one-by-one."
    Tick as many as you like; one red action row, one confirm naming them all.
    """
    p = ScriptedPrompter([
        "anger.dicx", "joy.dic",        # tick two
        ":delete", True,                # one confirm for both
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.name for e in lib.entries(dicts)] == ["mood.csv"]
    confirm = next(q for k, q in p.asked if k == "confirm" and "Delete" in q)
    assert "2" in confirm and "anger" in confirm and "joy" in confirm
    # the action row gets the danger tone -- red, not one more green row.
    delete_rows = [c for _q, cs in p.offered for c in cs if c.value == ":delete"]
    assert delete_rows and all(c.tone == "danger" for c in delete_rows)


def test_declining_the_batch_delete_keeps_every_tick_and_file(dicts, three):
    p = ScriptedPrompter([
        "anger.dicx", "joy.dic", ":delete", False,      # think better of it
        ":delete", True,                                # ticks survived; go
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.name for e in lib.entries(dicts)] == ["mood.csv"]


def test_rename_needs_exactly_one_tick_and_says_so(dicts, three):
    """The action rows are static now (space ticks in place, so rows cannot
    appear and vanish mid-prompt); the exactly-one rule moved into the answer."""
    p = ScriptedPrompter([
        "\x00space:anger.dicx", "\x00space:joy.dic",
        ":rename",                      # two ticked: refused with a reason
        ":back",
    ])
    manage_library(p, dicts)

    assert any("exactly one" in line for line in p.output)
    assert [e.stem for e in lib.entries(dicts)] == ["anger", "joy", "mood"]

def test_import_wears_the_green_and_export_does_not(dicts, three):
    """Requested directly: the forward act on this screen is bringing files
    IN. Export moving files does not make it the green one."""
    p = ScriptedPrompter([":back"])
    manage_library(p, dicts)

    rows = {c.value: c for _q, cs in p.offered for c in cs}
    assert rows[":import"].tone == "good"
    assert rows[":export"].tone == ""
    assert rows[":delete"].tone == "danger"


def test_a_bad_file_in_an_import_batch_costs_only_itself(dicts, tmp_path):
    """The import gate refuses the broken file with its reason on screen; the
    rest of the batch still arrives."""
    src = tmp_path / "mixed"
    src.mkdir()
    (src / "good.dic").write_text(VALID_DIC, encoding="utf-8")
    (src / "wordlist.dic").write_text("angry\nsad\n", encoding="utf-8")

    p = ScriptedPrompter([
        ":import", "\x00type", str(src),
        "\x00space:" + str(src / "good.dic"),
        str(src / "wordlist.dic"),          # enter: proceed with both
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.name for e in lib.entries(dicts)] == ["good.dic"]
    assert any("category header" in line for line in p.output)
    assert any("Imported 1 of 2" in line for line in p.output)


def test_import_announces_the_work_before_the_results(dicts, tmp_path):
    """
    Importing now means parsing, and the first file also pays for importing
    the parser itself -- on a slow drive the screen would sit silent looking
    hung. The 'Checking and importing…' line must land *before* the results,
    which is the whole point of prompter.working().
    """
    f = tmp_path / "fine.dic"
    f.write_text(VALID_DIC, encoding="utf-8")

    p = ScriptedPrompter([":import", "\x00type", str(f), ":back"])
    manage_library(p, dicts)

    working = next(i for i, line in enumerate(p.output)
                   if "Checking and importing 1 file…" in line)
    imported = next(i for i, line in enumerate(p.output)
                    if "Imported 1 of 1" in line)
    assert working < imported


def test_a_subtly_broken_dic_in_a_batch_costs_only_itself(dicts, tmp_path):
    """Same contract as the headerless case, one layer deeper: a file only the
    real parser can refuse is named on screen and the rest still arrives."""
    pytest.importorskip("contentcoder")
    src = tmp_path / "mixed"
    src.mkdir()
    (src / "good.dic").write_text(VALID_DIC, encoding="utf-8")
    # passes the heuristics, but references a category the header never defines.
    (src / "subtle.dic").write_text("%\n1\tCats\n%\ncat\t1\t9\n",
                                    encoding="utf-8")

    p = ScriptedPrompter([
        ":import", "\x00type", str(src),
        "\x00space:" + str(src / "good.dic"),
        str(src / "subtle.dic"),            # enter: proceed with both
        ":back",
    ])
    manage_library(p, dicts)

    assert [e.name for e in lib.entries(dicts)] == ["good.dic"]
    assert any("subtle.dic" in line and "parser" in line for line in p.output)
    assert any("Imported 1 of 2" in line for line in p.output)


def test_a_saved_folder_default_reads_as_all_on_the_settings_row(dicts, three):
    """run_saved seeds its overrides from the saved preset, whose untouched
    default is the library *folder* -- the row must say "all N of your…", the
    same thing the picker's ticks then show, or the two screens contradict
    each other about one setting."""
    shown = wiz._describe_library_value("dictionaries",
                                        [str(lib.kind_dir(dicts))])
    assert shown == "all 3 of your content-coding dictionaries"


# ---------------------------------------------------------------------------
# Optional library params: the stoplists
# ---------------------------------------------------------------------------

def _essay_folder(tmp_path):
    folder = tmp_path / "essays"
    folder.mkdir()
    for i in range(3):
        (folder / f"e{i}.txt").write_text(
            "Words upon words, and then a few more of them for good measure.",
            encoding="utf-8")
    return folder


def test_an_untouched_ngram_step_applies_the_three_default_stoplists(tmp_path):
    """
    Untouched, punctuation + English apply -- what nearly everyone wants,
    without setting them by hand each time. Never the whole folder: all 22
    languages at once would let German's "die" silently strip English text.
    """
    p = ScriptedPrompter([
        "txt_dir", *browse_to(_essay_folder(tmp_path)),
        ["ngram_frequencies"], False, "Defaulted", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_ngram_frequencies"))
    stoplists = lib.KINDS["stoplists"]
    expected = sorted(str((lib.kind_dir(stoplists) / n).resolve()) for n in
                      ("_chars.txt", "_chars_extended.txt", "stopwords-en.txt"))
    assert sorted(step["with"]["stoplist_paths"]) == expected


def test_the_stoplist_picker_starts_with_punctuation_and_english(tmp_path):
    """Opening the picker offers all 24 built-ins with exactly the three
    defaults ticked; a bare enter on one of them approves those three."""
    stoplists = lib.KINDS["stoplists"]
    en = lib.kind_dir(stoplists) / "stopwords-en.txt"
    p = ScriptedPrompter([
        "txt_dir", *browse_to(_essay_folder(tmp_path)),
        ["ngram_frequencies"],
        True, "ngram_frequencies", "stoplist_paths",
        _key(en),                       # enter approves the default ticks
        ":done", ":done", "Stopped", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    labels = [c.label for c in p.offered_choices("Which stop word lists")]
    ticked = {label[4:] for label in labels if label.startswith("[x] ")}
    assert ticked == {"_chars", "_chars_extended", "stopwords-en"}
    assert sum(1 for label in labels if label.startswith("[ ] ")) == 21

    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_ngram_frequencies"))
    expected = sorted(str((lib.kind_dir(stoplists) / n).resolve()) for n in
                      ("_chars.txt", "_chars_extended.txt", "stopwords-en.txt"))
    assert step["with"]["stoplist_paths"] == expected


def test_the_stoplist_row_names_the_default_set_when_untouched(tmp_path):
    """The row and the picker must tell the same story: these three apply."""
    p = ScriptedPrompter([
        "txt_dir", *browse_to(_essay_folder(tmp_path)),
        ["ngram_frequencies"],
        True, "ngram_frequencies", ":done", ":done", "Rowcheck", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    row = next(c for _q, choices in p.offered for c in choices
               if c.value == "stoplist_paths")
    assert "the default set" in row.label
    assert "stopwords-en" in row.label and "_chars" in row.label


def test_preflight_ignores_an_empty_stoplist_library(tmp_path):
    """A step with curated defaults runs fine without them, so an empty
    stoplist library is not a contingency to fix -- unlike dictionaries,
    where an empty library means the step reads nothing. Deleted defaults
    are simply not injected: the run is vanilla, never broken."""
    stoplists = lib.KINDS["stoplists"]
    for e in lib.entries(stoplists):
        lib.delete(stoplists, e.name)

    p = ScriptedPrompter([
        "txt_dir", *browse_to(_essay_folder(tmp_path)),
        ["ngram_frequencies"], False, "NoLists", "save",
    ])
    result = wiz.run_wizard(p, cwd=tmp_path)

    assert result.preset is not None
    assert not any("none imported" in r for r in p.reasons)
    step = next(s for s in result.preset["steps"]
                if s["call"].endswith("analyze_ngram_frequencies"))
    assert "stoplist_paths" not in step["with"]


def test_stoplists_import_and_export_through_the_same_manager(tmp_path):
    """
    Stop lists are an ordinary library kind: the Settings manager imports a
    user's own .txt where they sit and exports the built-ins back out --
    nothing stoplist-specific to learn.
    """
    stoplists = lib.KINDS["stoplists"]
    mine = tmp_path / "my-jargon.txt"
    mine.write_text("uh\num\n", encoding="utf-8")
    out = tmp_path / "exported"
    out.mkdir()

    p = ScriptedPrompter([
        ":import", "\x00type", str(mine),
        "stopwords-en.txt", ":export", "\x00type", str(out),
        ":back",
    ])
    manage_library(p, stoplists)

    names = {e.name for e in lib.entries(stoplists)}
    assert "my-jargon.txt" in names, "the user's own list was not imported"
    assert (out / "stopwords-en.txt").exists(), "a built-in did not export"
    # and the analyzer's loader can use the imported list right away.
    from taters.text.ngram_prep import load_stoplist
    assert load_stoplist([lib.kind_dir(stoplists) / "my-jargon.txt"]) == {"uh", "um"}


def test_the_dictionaries_entry_offers_only_dictionary_kinds(tmp_path):
    """Stop lists, models and encoders have their own Settings rows, so the
    dictionaries chooser must not offer them -- one door per thing.

    Checked against the shelf list rather than a frozen set: adding a shelf of
    word lists behind this one door is allowed, and the rule being guarded is
    that the shelves with their own door stay out."""
    p = ScriptedPrompter([":back"])
    library_task.TASK.run(TaskContext(prompter=p, cwd=tmp_path))

    kinds = {c.value for c in p.offered_choices("Which library?")}
    assert kinds == set(library_task._DICTIONARY_KINDS) | {":back"}
    assert not kinds & {"stoplists", "models", "encoders"}


def test_the_stoplists_entry_opens_the_manager_directly(tmp_path):
    """One kind means nothing to choose: the task lands straight in the
    manager, not on a single-option 'Which library?' screen."""
    p = ScriptedPrompter([":back"])
    library_task.STOPLISTS_TASK.run(TaskContext(prompter=p, cwd=tmp_path))

    questions = [q for _, q in p.asked]
    assert not any(q.startswith("Which library") for q in questions)
    assert any("Stop word lists" in q for q in questions)


def test_manage_taters_data_lists_both_library_tasks():
    from taters.ui.tasks import manage_data, settings

    ids = [t.id for t in manage_data.entries()]
    assert "library" in ids and "stoplists" in ids
    # and the row that holds them is the one Settings offers
    assert "manage_data" in {t.id for t in settings.entries()}


def test_a_hostile_rename_shows_its_reason_and_keeps_the_file(dicts, three):
    """The refusal must reach the screen -- an uncaught ValueError took the
    whole manager down, and a swallowed one read as a silent failure."""
    p = ScriptedPrompter([
        "joy.dic", ":rename", "evil[/]",
        ":back",
    ])
    manage_library(p, dicts)

    assert any("cannot contain" in line for line in p.output)
    assert "joy.dic" in {e.name for e in lib.entries(dicts)}


def test_the_default_stoplist_ticks_explain_themselves(tmp_path):
    """The row says the default set applies; the picker opens with the same
    three ticked and says so -- the two screens must tell one story."""
    p = ScriptedPrompter([
        "txt_dir", *browse_to(_essay_folder(tmp_path)),
        ["ngram_frequencies"],
        True, "ngram_frequencies", "stoplist_paths",
        _key(lib.kind_dir(lib.KINDS["stoplists"]) / "stopwords-en.txt"),
        ":done", ":done", "Explained", "save",
    ])
    wiz.run_wizard(p, cwd=tmp_path)

    assert any("default set" in r for r in p.reasons)


# ---------------------------------------------------------------------------
# Naming a model at import
# ---------------------------------------------------------------------------

def _ridge_file(tmp_path, name="ridge__all.json", outcome="age"):
    import json

    path = tmp_path / name
    path.write_text(json.dumps({
        "kind": "taters-ridge-model", "format": 2, "predictors": ["f1"],
        "outcomes": {outcome: {"kept": [0], "mu": [0.0], "sigma": [1.0],
                               "coef": [1.0], "intercept": 0.0}},
    }), encoding="utf-8")
    return path


def _mem_file(tmp_path, k=3):
    import json

    path = tmp_path / "themes_model.json"
    path.write_text(json.dumps({
        "kind": "taters-mem-model", "format": 1,
        "model": {"themes": [f"Theme_{i + 1}" for i in range(k)]},
    }), encoding="utf-8")
    return path


def test_a_model_is_named_at_import_and_the_name_reaches_its_columns(
        tmp_path):
    """Asked at import because this is the one moment the user knows what the
    model is. Every ridge file is called `ridge__all` and writes `pred_age`,
    so two of them scored in one run collide, and one scored against a
    dataset that already has an `age` column is worse than a collision --
    both readings are plausible and nothing reports it."""
    from taters.helpers.model_spec import describe
    from taters.ui.library import _name_a_model

    path = _ridge_file(tmp_path)
    p = ScriptedPrompter([True, "age_blogs", "age_blogs"])
    _name_a_model(p, path)

    info = describe(path)
    assert info.name == "age_blogs"
    assert info.columns == ("pred_age_blogs",)
    assert info.display() == "age_blogs [ridge]"


def test_declining_the_naming_leaves_the_model_exactly_as_it_was(tmp_path):
    """The whole step is optional, and the defaults are what the file already
    says -- so enter through it changes nothing, and answering "no" does not
    even ask."""
    from taters.helpers.model_spec import describe
    from taters.ui.library import _name_a_model

    path = _ridge_file(tmp_path)
    before = path.read_text(encoding="utf-8")
    _name_a_model(ScriptedPrompter([False]), path)
    assert path.read_text(encoding="utf-8") == before

    # and accepting but keeping every default is, in effect, also a no-op.
    _name_a_model(ScriptedPrompter([True, "ridge__all", "age"]), path)
    assert describe(path).columns == ("pred_age",)


def test_a_topic_models_themes_are_named_by_prefix(tmp_path):
    """Forty themes are not named one at a time, and one prefix is what
    tells one topic model's columns from another's."""
    from taters.helpers.model_spec import describe
    from taters.ui.library import _name_a_model

    path = _mem_file(tmp_path, k=3)
    _name_a_model(ScriptedPrompter([True, "fb_topics", "fb_topics"]), path)
    info = describe(path)
    assert info.name == "fb_topics"
    assert info.columns == ("fb_topics_1", "fb_topics_2", "fb_topics_3")


def test_a_name_that_would_break_the_results_file_is_reported_not_swallowed(
        tmp_path):
    """A comma becomes a column heading and splits the row when the results
    are read back. Refused -- and the reason has to be shown, or the rename
    reads as having silently worked."""
    from taters.helpers.model_spec import describe
    from taters.ui.library import _name_a_model

    path = _ridge_file(tmp_path)
    p = ScriptedPrompter([True, "fine", "bad,name"])
    _name_a_model(p, path)
    assert describe(path).columns == ("pred_age",)
    assert any("column heading" in str(n) for n in p.output), \
        f"the refusal was never shown: {p.output}"


def test_naming_is_never_offered_for_something_that_is_not_a_model(tmp_path):
    """A dictionary's filename already says what it is, and being asked to
    name its output columns would be a question with no answer."""
    from taters.ui.library import _name_a_model

    plain = tmp_path / "anger.dicx"
    plain.write_text(VALID_DIC, encoding="utf-8")
    p = ScriptedPrompter([])
    _name_a_model(p, plain)      # no questions asked, so nothing to script


def test_exporting_a_model_says_what_is_embedded_in_it(tmp_path, monkeypatch):
    """
    A model carries the word lists it was fitted with, which is what makes it
    portable -- and means the file contains them. The moment that matters is
    sharing: exporting, emailing, posting to a repository. So the warning
    lives there rather than at fit time, because what is licensed and what is
    confidential is a judgment about the material that only the person
    sharing it can make.
    """
    import json

    from taters.ui.library import _warn_about_embedded_content

    kind = lib.KINDS["models"]
    model = tmp_path / "age_blogs.json"
    model.write_text(json.dumps({
        "kind": "taters-ridge-model", "format": 2, "predictors": ["a"],
        "outcomes": {"age": {"kept": [0], "mu": [0.0], "sigma": [1.0],
                             "coef": [1.0], "intercept": 0.0}},
        "assets": {"connectives": [{"name": "causal.txt", "sha256": "x",
                                    "text": "because\nsince\n"}]},
    }), encoding="utf-8")
    lib.import_into(kind, model)

    p = ScriptedPrompter([])
    _warn_about_embedded_content(p, kind, ["age_blogs.json"])
    said = " ".join(str(o) for o in p.output)
    assert "carry the word lists" in said
    assert "causal.txt" in said
    assert "publish or send them on" in said


def test_a_model_carrying_nothing_gets_no_warning(tmp_path):
    """The warning has to stay meaningful. A model with no embedded lists is
    the common case, and a caution printed every time is one nobody reads."""
    import json

    from taters.ui.library import _warn_about_embedded_content

    kind = lib.KINDS["models"]
    model = tmp_path / "plain.json"
    model.write_text(json.dumps({
        "kind": "taters-ridge-model", "format": 2, "predictors": ["a"],
        "outcomes": {"age": {"kept": [0], "mu": [0.0], "sigma": [1.0],
                             "coef": [1.0], "intercept": 0.0}},
    }), encoding="utf-8")
    lib.import_into(kind, model)

    p = ScriptedPrompter([])
    _warn_about_embedded_content(p, kind, ["plain.json"])
    assert not p.output

    # and it never fires for a kind that isn't a model at all.
    p = ScriptedPrompter([])
    _warn_about_embedded_content(p, lib.KINDS["dictionaries"], ["anger.dicx"])
    assert not p.output


def test_a_renderer_without_in_place_ticks_can_still_pick_a_subset(dicts, three):
    """
    Under --plain, `QuestionaryPrompter.select` bound no space key and read
    no `ticked`, so the only reachable selections were all, or exactly one
    after "Untick everything": two of five dictionaries was impossible. Now
    enter on an entry flips its mark and re-asks, and an approve row exists
    because enter cannot mean approve there.
    """
    from taters.ui.prompts import QuestionaryPrompter

    class Plain(QuestionaryPrompter):
        def __init__(self, answers):
            super().__init__()
            self.answers = list(answers)
            self.shown = []
            self.rows = []

        def select(self, question, choices, **kw):
            self.rows = [c.value for c in choices]
            return super().select(question, choices, **kw)

        def _ask(self, q):
            self.shown.append(q)
            answer = self.answers.pop(0)
            # an actual person can only pick a row that's on the screen.
            assert answer in self.rows, f"{answer!r} is not offered: {self.rows}"
            return answer

    p = Plain([_key(three[1]), ":use-ticked"])     # untick one, approve the rest
    picked = pick_from_library(p, dicts)
    assert picked == sorted(_key(e) for e in (three[0], three[2]))
    assert len(p.shown) == 2, "the list was not re-asked after the tick"

    # and the scripted prompter, which ticks in place, never gets offered that row.
    s = ScriptedPrompter([_key(three[0])])
    pick_from_library(s, dicts)
    assert not any(c.value == ":use-ticked" for _q, cs in s.offered for c in cs)


def test_a_scoring_step_asks_which_model_when_the_library_holds_several(tmp_path):
    """With two saved models the old default -- the whole folder -- failed
    minutes into the run with "2 model files found"; now the features stage
    asks, and the answer lands in the preset. One model: nothing asked."""
    import json

    from wizard_helpers import browse_to

    models = lib.KINDS["models"]

    def _mem(path, name):
        # a text model: it re-derives its own instrument, so the compose-time
        # provenance gate has nothing to ask it and the picker is the only
        # question in play.
        path.write_text(json.dumps({
            "kind": "taters-mem-model", "format": 1, "name": name,
            "text": {"lemmatize": False, "pos_tagged": False, "engine": "nltk",
                     "tokenizer": "potts", "stanza_lang": "en",
                     "keep_punctuation": False},
            "matrix": {"weighting": "count", "rounding": 4, "terms": ["a", "b"],
                       "columns": ["a", "b"], "idf": [0.1, 0.2]},
            "model": {"n_documents": 3, "rotation": True, "themes": ["Theme_1"],
                      "kept": [0, 1], "mu": [0.0, 0.0], "sigma": [1.0, 1.0],
                      "projection": [[0.5], [0.5]], "eigenvalues": [1.0],
                      "pct_variance": [50.0]}}), encoding="utf-8")
        return path

    study = tmp_path / "s.csv"
    study.write_text("pid,text\n" + "".join(f"p{i},some words here {i}\n" for i in range(6)),
                     encoding="utf-8")
    lib.import_into(models, _mem(tmp_path / "themes_a.json", "a"))
    one = ScriptedPrompter(["csv", *browse_to(study), ["text"], True, ["pid"],
                            ["score_with_model"], "row", False, "One", "save"])
    wiz.run_wizard(one, cwd=tmp_path, banner=False, analyses=False)
    assert not any(q.startswith("Which saved models") for _k, q in one.asked)

    lib.import_into(models, _mem(tmp_path / "themes_b.json", "b"))
    target = str((lib.kind_dir(models) / "themes_b.json").resolve())
    two = ScriptedPrompter(["csv", *browse_to(study), ["text"], True, ["pid"],
                            ["score_with_model"], "row",
                            target,             # the new question
                            # making a choice counts as changing a setting, so
                            # the options screen opens on its list, not its gate.
                            ":done", "Two", "save"])
    res = wiz.run_wizard(two, cwd=tmp_path, banner=False, analyses=False)
    assert any(q.startswith("Which saved models") for _k, q in two.asked)
    step = next(s for s in res.preset["steps"] if s["call"].endswith("score_with_model"))
    assert step["with"]["model_json"] == [target]


# ---------------------------------------------------------------------------
# Changing how a saved model is applied
# ---------------------------------------------------------------------------

def _classifier_stub(path, classes=("0", "1")):
    import json

    path.write_text(json.dumps({
        "kind": "taters-classifier-model", "format": 2, "name": "cond",
        "predictors": ["f1"],
        "outcomes": {"cond": {"kept": [0], "mu": [0.0], "sigma": [1.0],
                              "classes": list(classes),
                              "models": [{"intercept": 0.0, "coef": [1.0]}, None]}},
    }), encoding="utf-8")
    return path


def _vectors_stub(folder):
    import json

    import numpy as np

    from taters.helpers.provenance import file_digest

    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / "wv.npy"
    np.save(weights, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    doc = {"kind": "taters-word-vectors-model", "format": 1, "name": "wv",
           "payload": ["wv.npy"], "payload_digests": {"wv.npy": file_digest(weights)},
           "text": {"lemmatize": False, "pos_tagged": False, "engine": "nltk",
                    "tokenizer": "potts", "stanza_lang": "en",
                    "keep_punctuation": False, "lowercase": True},
           "vocabulary": ["cat", "dog"], "counts": [3, 2], "dim": 2,
           "dim_labels": ["wv_1", "wv_2"], "training": {"source": "trained"},
           "apply": {"weighting": "tokens", "normalize_words": False, "concept_dicts": []}}
    (folder / "wv.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder / "wv.json"


def test_the_models_manager_relabels_classes_one_at_a_time(tmp_path):
    """The complaint was results reading "outcome 0, outcome 1". The row
    walks the classes of the ticked model with the current label as the
    default, and the file carries the answer for every later run."""
    from taters.helpers.model_spec import describe

    models = lib.KINDS["models"]
    lib.import_into(models, _classifier_stub(tmp_path / "classifier__all.json"))
    p = ScriptedPrompter([
        "classifier__all.json", ":apply",
        "classes:cond", "control", "patient",       # relabel both classes
        ":done", ":back",
    ])
    manage_library(p, models)
    info = describe(lib.kind_dir(models) / "classifier__all.json")
    assert info.classes == {"cond": ("control", "patient")}
    assert set(info.columns) >= {"p_cond_control", "p_cond_patient"}
    offered = next(cs for q, cs in p.offered if q == "Change what?")
    assert [c.value for c in offered] == [":done", "name", "outputs", "classes:cond"], \
        "a classifier has no apply settings, so none are offered"
    assert any("Saved." in line for line in p.output)


def test_the_models_manager_edits_apply_settings_with_the_right_widget(dicts, three, tmp_path):
    """A choice is a list, a yes/no a confirm, a library setting the library
    picker (opening empty), a name a text box -- and a refused value is a
    yellow note, not a traceback."""
    from taters.helpers.model_spec import describe

    models = lib.KINDS["models"]
    lib.import_into(models, _vectors_stub(tmp_path / "v"))
    joy = str((lib.kind_dir(dicts) / "joy.dic").resolve())
    p = ScriptedPrompter([
        "wv.json", ":apply",
        "apply:weighting", "types",
        "apply:normalize_words", True,
        "apply:concept_dicts", joy,                # enter on the entry: that one
        "name", "my vectors",
        "outputs", "emb",                          # a prefix: the dims are a family
        ":done", ":back",
    ])
    manage_library(p, models)
    info = describe(lib.kind_dir(models) / "wv.json")
    assert info.apply["weighting"] == "types" and info.apply["normalize_words"] is True
    assert [d["name"] for d in info.apply["concept_dicts"]] == ["joy"]
    assert info.apply["concept_dicts"][0]["categories"] == {"Cats": [["cat", 1.0]]}
    assert info.name == "my vectors" and info.outputs == ("emb_1", "emb_2")
    assert info.columns[-1] == "sim_joy__Cats"
    kinds = {q: k for k, q in p.asked}
    assert kinds["weighting:"] == "select" and kinds["normalize_words?"] == "confirm"
    picker = next(cs for q, cs in p.offered if q.startswith("Which content-coding dictionaries"))
    assert all(c.label.startswith("[ ]") for c in picker if c.value == joy), "opens empty"
    offered = next(cs for q, cs in p.offered if q == "weighting:")
    assert [c.value for c in offered] == ["tokens", "types", "sif"]

    q = ScriptedPrompter(["wv.json", ":apply", "apply:weighting", "sif", ":done", ":back"])
    manage_library(q, models)
    assert describe(lib.kind_dir(models) / "wv.json").apply["weighting"] == "sif"
    # a bad value never reaches the file: sif with no counts gets refused at
    # apply time, but a value outside the choices gets refused right here
    r = ScriptedPrompter(["wv.json", ":apply", "apply:normalize_words", True, ":done", ":back"])
    manage_library(r, models)
    assert describe(lib.kind_dir(models) / "wv.json").apply["normalize_words"] is True


def test_the_apply_row_belongs_to_models_alone_and_needs_one_tick(dicts, three, tmp_path):
    p = ScriptedPrompter([":back"])
    manage_library(p, dicts)
    assert ":apply" not in {c.value for c in p.offered[0][1]}

    models = lib.KINDS["models"]
    lib.import_into(models, _classifier_stub(tmp_path / "a.json"))
    lib.import_into(models, _classifier_stub(tmp_path / "b.json"))
    p = ScriptedPrompter([":apply", "a.json", "b.json", ":apply", ":back"])
    manage_library(p, models)
    notes = [line for line in p.output if "ticked" in line]
    assert any("Nothing is ticked yet" in n for n in notes)
    assert any("exactly one file ticked" in n for n in notes)


# ---------------------------------------------------------------------------
# Choosing an encoder
# ---------------------------------------------------------------------------

def _encoder_stub(folder, stem="clinical"):
    import json

    payload = folder / f"{stem}.encoder"
    payload.mkdir(parents=True, exist_ok=True)
    (payload / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    (payload / "model.safetensors").write_bytes(b"w" * 16)
    (payload / "vocab.txt").write_text("[PAD]\n", encoding="utf-8")
    doc = {"kind": "taters-encoder", "format": 1, "name": stem, "payload": [payload.name],
           "base_model": "distilroberta-base", "num_hidden_layers": 6,
           "evaluation": {"perplexity_before": 15.0, "perplexity_after": 7.5}}
    (folder / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder / f"{stem}.json"


def _predictor_stub(path):
    """A structurally complete predictor: the import gate wants an encoder
    folder and the heads' weights beside the manifest."""
    import json

    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    payload = path.parent / "x.predictor"
    (payload / "encoder").mkdir(parents=True, exist_ok=True)
    (payload / "encoder" / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
    save_file({"openness.weight": torch.zeros(1, 4), "openness.bias": torch.zeros(1)},
              str(payload / "heads.safetensors"))
    path.write_text(json.dumps({
        "kind": "taters-text-predictor-model", "format": 1, "name": "openness_model",
        "payload": ["x.predictor"], "encoder": {"layers": "last", "pooling": "mean",
                                                 "max_length": 256},
        "outcomes": {"openness": {"task": "regression", "mean": 0.0, "std": 1.0}}}),
        encoding="utf-8")
    return path


def test_the_hub_cache_lists_downloaded_encoders_and_nothing_else(tmp_path):
    """A Whisper or a Qwen checkpoint shares the cache; offering it would
    fail at load time. Only configs naming an encoder architecture count."""
    import json

    from taters.text._transformer_common import cached_hub_encoders

    def _cached(name, cfg):
        snap = tmp_path / f"models--{name}" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    _cached("distilroberta-base", {"model_type": "roberta", "num_hidden_layers": 6,
                                   "hidden_size": 768})
    _cached("sentence-transformers--all-MiniLM-L6-v2", {"model_type": "bert",
                                                        "num_hidden_layers": 6,
                                                        "hidden_size": 384})
    _cached("openai--whisper-tiny", {"model_type": "whisper"})
    _cached("Qwen--Qwen3.5-0.8B", {"model_type": "qwen3_5"})
    (tmp_path / "models--broken").mkdir()          # no snapshots at all
    found = cached_hub_encoders(tmp_path)
    assert found == [("distilroberta-base", "downloaded · 6 layers · 768 wide"),
                     ("sentence-transformers/all-MiniLM-L6-v2",
                      "downloaded · 6 layers · 384 wide")]
    assert cached_hub_encoders(tmp_path / "nowhere") == []


def test_the_encoder_picker_offers_the_library_the_cache_and_the_usual_names(tmp_path, monkeypatch):
    from taters.text import _transformer_common as tc
    from taters.ui.library import pick_encoder

    lib.import_into(lib.KINDS["encoders"], _encoder_stub(tmp_path / "e"))
    predictor = tmp_path / "text_predictor__openness.json"
    lib.import_into(lib.KINDS["models"], _predictor_stub(predictor))
    monkeypatch.setattr(tc, "cached_hub_encoders",
                        lambda cache_dir=None: [("distilroberta-base", "downloaded · 6 layers")])
    enc = str((lib.kind_dir(lib.KINDS["encoders"]) / "clinical.json").resolve())

    p = ScriptedPrompter([enc])
    assert pick_encoder(p) == enc
    rows = p.offered[0][1]
    values = [c.value for c in rows]
    assert values[0] == enc, "your own encoders first"
    assert values[1] == str((lib.kind_dir(lib.KINDS["models"]) / "text_predictor__openness.json").resolve())
    assert "warm start" in rows[1].help
    # each row says which kind it is: a plain encoder, or a sentence-transformers
    # model trained so that similar texts get similar vectors
    assert values[2] == "distilroberta-base" and rows[2].help.startswith("raw encoder · downloaded")
    assert "sentence-transformers/all-MiniLM-L6-v2" in values
    minilm = next(c for c in rows if c.value == "sentence-transformers/all-MiniLM-L6-v2")
    assert minilm.help.startswith("meaning-tuned · not downloaded yet")
    assert values[-1] == ":type"
    assert values.count("distilroberta-base") == 1, "cached and curated: listed once"

    typed = ScriptedPrompter([":type", "my-org/my-encoder"])
    assert pick_encoder(typed, current="distilroberta-base") == "my-org/my-encoder"
    assert p.select_defaults.get("Which encoder?") is None
    again = ScriptedPrompter(["distilroberta-base"])
    pick_encoder(again, current="distilroberta-base")
    assert again.select_defaults["Which encoder?"] == "distilroberta-base", "the current one is pointed at"


def test_the_hub_cache_tells_meaning_tuned_models_by_their_marker_file(tmp_path):
    """
    A sentence-transformers checkpoint carries `config_sentence_transformers.json`
    beside its weights; a raw encoder does not. Reading that, rather than
    the ``sentence-transformers/`` prefix, is what lets a meaning-tuned model
    from any organization -- or a folder someone saved themselves -- count.
    And a raw encoder must not sneak in: the step this feeds promises a model
    trained for meaning.
    """
    import json

    from taters.text._transformer_common import (SENTENCE_MODEL_MARKER,
                                                 cached_hub_sentence_models)

    def _cached(name, cfg, *, tuned):
        snap = tmp_path / f"models--{name}" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        if tuned:
            (snap / SENTENCE_MODEL_MARKER).write_text("{}", encoding="utf-8")

    _cached("distilroberta-base", {"model_type": "roberta", "num_hidden_layers": 6,
                                   "hidden_size": 768}, tuned=False)
    _cached("sentence-transformers--all-MiniLM-L6-v2",
            {"model_type": "bert", "num_hidden_layers": 6, "hidden_size": 384},
            tuned=True)
    # not under the sentence-transformers org, and still meaning-tuned
    _cached("someone--their-own-model",
            {"model_type": "roberta", "num_hidden_layers": 12, "hidden_size": 768},
            tuned=True)

    found = cached_hub_sentence_models(tmp_path)
    assert found == [
        ("sentence-transformers/all-MiniLM-L6-v2", "downloaded · 6 layers · 384-wide vectors"),
        ("someone/their-own-model", "downloaded · 12 layers · 768-wide vectors"),
    ]
    assert "distilroberta-base" not in dict(found), "a raw encoder was offered as meaning-tuned"


def test_the_meaning_tuned_picker_is_its_own_menu_not_the_encoder_one_again(tmp_path, monkeypatch):
    """
    Both embedding steps ask for a model, one after the other, and the two
    screens used to be the same menu with different rows -- which is how a
    raw encoder ends up chosen for the meaning-tuned step. So the question is
    worded differently, the step is named above it, and only meaning-tuned
    models are offered: no library shelf (an encoder adapted here is raw),
    and no raw encoder from the cache however many are downloaded.
    """
    from taters.text import _transformer_common as tc
    from taters.ui.library import pick_encoder

    lib.import_into(lib.KINDS["encoders"], _encoder_stub(tmp_path / "e"))
    monkeypatch.setattr(tc, "cached_hub_sentence_models",
                        lambda cache_dir=None: [("sentence-transformers/all-MiniLM-L6-v2",
                                                 "downloaded · 6 layers · 384-wide vectors")])
    monkeypatch.setattr(tc, "cached_hub_encoders",
                        lambda cache_dir=None: [("distilroberta-base", "downloaded · 6 layers")])

    p = ScriptedPrompter(["sentence-transformers/all-mpnet-base-v2"])
    picked = pick_encoder(p, "sentence-transformers/all-roberta-large-v1",
                          meaning_tuned=True,
                          for_step="Sentence embeddings (meaning-tuned model)")
    assert picked == "sentence-transformers/all-mpnet-base-v2"

    question, rows = p.offered[0]
    assert question == "Which meaning-tuned model?"
    assert question != "Which encoder?"
    values = [c.value for c in rows]
    assert values[0] == "sentence-transformers/all-MiniLM-L6-v2", "what is downloaded comes first"
    assert rows[0].help.startswith("meaning-tuned · downloaded")
    assert "sentence-transformers/all-roberta-large-v1" in values
    assert "distilroberta-base" not in values, "a raw encoder was offered"
    assert not any(v.endswith(".json") for v in values), "the library's raw encoders were offered"
    assert values[-1] == ":type"
    assert p.select_defaults["Which meaning-tuned model?"] == \
        "sentence-transformers/all-roberta-large-v1", "the current one is pointed at"
    assert any("Sentence embeddings (meaning-tuned model)" in r for r in p.reasons), \
        "the line above the menu does not say which step this model is for"

    typed = ScriptedPrompter([":type", "my-org/my-sentence-model"])
    assert pick_encoder(typed, meaning_tuned=True) == "my-org/my-sentence-model"
    assert any(q == "Model name or folder:" for _k, q in typed.asked)
