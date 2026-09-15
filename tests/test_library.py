"""
The library: user-imported assets that outlive any one project.

Everything runs under a temporary ``TATERS_HOME`` -- the environment override
exists precisely so these tests never touch a real home directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taters.helpers import library as lib


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("TATERS_HOME", str(tmp_path / "tathome"))
    return tmp_path / "tathome"


@pytest.fixture()
def dicts():
    return lib.KINDS["dictionaries"]


VALID_DIC = "%\n1\tCats\n%\ncat\t1\n"   # the smallest .dic that's still LIWC-shaped


def _a_file(tmp_path, name="mood.dicx", text="mood words"):
    f = tmp_path / name
    # .dic files have to be LIWC-shaped: the import gate refuses bare word
    # lists, because we once had a run die on one an hour after importing it.
    if name.endswith(".dic") and text == "mood words":
        text = VALID_DIC
    f.write_text(text, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Where it lives
# ---------------------------------------------------------------------------

def test_the_env_override_moves_the_whole_library(home, dicts):
    assert lib.library_home() == home / "library"
    assert lib.kind_dir(dicts) == home / "library" / "dictionaries"


def test_without_the_override_the_library_is_under_the_users_home(monkeypatch):
    monkeypatch.delenv("TATERS_HOME", raising=False)
    assert lib.library_home() == Path.home() / ".taters" / "library"


def test_an_unknown_kind_names_the_valid_ones():
    with pytest.raises(KeyError) as err:
        lib.kind_by_id("classifiers")
    assert "archetypes" in str(err.value) and "dictionaries" in str(err.value)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def test_import_list_export_rename_delete_round_trip(home, dicts, tmp_path):
    src = _a_file(tmp_path)

    stored = lib.import_into(dicts, src)
    assert [e.stem for e in lib.entries(dicts)] == ["mood"]
    assert stored.read_text(encoding="utf-8") == "mood words"

    out = lib.export_to(dicts, "mood", tmp_path / "exported")
    assert out.read_text(encoding="utf-8") == "mood words"
    assert out.name == "mood.dicx"

    renamed = lib.rename(dicts, "mood", "moods-2026")
    assert renamed.stem == "moods-2026"
    assert [e.stem for e in lib.entries(dicts)] == ["moods-2026"]

    lib.delete(dicts, "moods-2026")
    assert lib.entries(dicts) == []


def test_the_wrong_format_is_refused_with_the_right_ones_named(home, tmp_path):
    arch = lib.KINDS["archetypes"]
    with pytest.raises(ValueError) as err:
        lib.import_into(arch, _a_file(tmp_path, "mood.dicx"))
    assert ".csv" in str(err.value)


def test_an_import_never_silently_overwrites(home, dicts, tmp_path):
    lib.import_into(dicts, _a_file(tmp_path, "mood.dicx", "first"))
    with pytest.raises(lib.LibraryCollision):
        lib.import_into(dicts, _a_file(tmp_path, "mood.dicx", "second"))
    # ...unless the caller says so (this is what the UI's "replace?" question
    # feeds into).
    lib.import_into(dicts, _a_file(tmp_path, "mood.dicx", "second"), replace=True)
    assert lib.entries(dicts)[0].read_text(encoding="utf-8") == "second"


def test_a_rename_cannot_change_the_format(home, dicts, tmp_path):
    """The suffix says how to parse the file; a rename must not lie about it."""
    lib.import_into(dicts, _a_file(tmp_path, "mood.dicx"))

    renamed = lib.rename(dicts, "mood", "sneaky.csv")

    assert renamed.name == "sneaky.dicx"


def test_a_rename_onto_an_existing_entry_is_refused(home, dicts, tmp_path):
    lib.import_into(dicts, _a_file(tmp_path, "a.dicx"))
    lib.import_into(dicts, _a_file(tmp_path, "b.dicx"))
    with pytest.raises(lib.LibraryCollision):
        lib.rename(dicts, "a", "b")


def test_a_stray_file_of_the_wrong_format_is_ignored_not_an_error(home, dicts):
    """The library is a plain folder people may touch by hand."""
    (lib.kind_dir(dicts) / "notes.txt").write_text("hi", encoding="utf-8")
    assert lib.entries(dicts) == []


def test_a_missing_entry_names_what_does_exist(home, dicts, tmp_path):
    lib.import_into(dicts, _a_file(tmp_path, "mood.dicx"))
    with pytest.raises(KeyError) as err:
        lib.delete(dicts, "moood")
    assert "mood" in str(err.value)


# ---------------------------------------------------------------------------
# Seeding the built-ins
# ---------------------------------------------------------------------------

def test_builtins_are_seeded_when_a_kind_is_first_used(home, dicts, monkeypatch, tmp_path):
    shipped = tmp_path / "shipped" / "library" / "dictionaries"
    shipped.mkdir(parents=True)
    (shipped / "builtin.dicx").write_text("shipped words", encoding="utf-8")
    (shipped / "README.md").write_text("not a dictionary", encoding="utf-8")
    monkeypatch.setattr(lib, "_shipped_dir", lambda kind: shipped)

    got = lib.entries(dicts)

    assert [e.stem for e in got] == ["builtin"]
    assert got[0].read_text(encoding="utf-8") == "shipped words"


def test_deleting_a_builtin_is_respected(home, dicts, monkeypatch, tmp_path):
    """
    The contingency for an empty library is messaging in the UI, not quietly
    putting back files the user just removed.
    """
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    (shipped / "builtin.dicx").write_text("shipped", encoding="utf-8")
    monkeypatch.setattr(lib, "_shipped_dir", lambda kind: shipped)

    lib.delete(dicts, "builtin")

    assert lib.entries(dicts) == []


# ---------------------------------------------------------------------------
# Twins: two formats, one stem
# ---------------------------------------------------------------------------

def test_a_bare_stem_shared_by_two_files_is_never_guessed(home, dicts, tmp_path):
    """
    From the code review (issue 1): `foo.csv` and `foo.dic` can both exist, and
    resolving a stem to the first sorted match deleted the file the user did
    not pick. Ambiguity has to be an error, never a guess.
    """
    lib.import_into(dicts, _a_file(tmp_path, "foo.csv", "the csv"))
    lib.import_into(dicts, _a_file(tmp_path, "foo.dic", VALID_DIC))

    with pytest.raises(KeyError) as err:
        lib.delete(dicts, "foo")
    assert "foo.csv" in str(err.value) and "foo.dic" in str(err.value)
    assert len(lib.entries(dicts)) == 2, "an ambiguous name deleted something"


def test_a_full_filename_reaches_exactly_that_file(home, dicts, tmp_path):
    lib.import_into(dicts, _a_file(tmp_path, "foo.csv", "the csv"))
    lib.import_into(dicts, _a_file(tmp_path, "foo.dic", VALID_DIC))

    lib.delete(dicts, "foo.dic")

    survivors = lib.entries(dicts)
    assert [e.name for e in survivors] == ["foo.csv"]
    assert survivors[0].read_text(encoding="utf-8") == "the csv"


def test_a_unique_stem_still_works_as_a_name(home, dicts, tmp_path):
    """The friendly form stays valid while it cannot lie."""
    lib.import_into(dicts, _a_file(tmp_path, "solo.dicx"))
    lib.rename(dicts, "solo", "duet")
    assert [e.name for e in lib.entries(dicts)] == ["duet.dicx"]


def test_a_headerless_dic_is_refused_at_import_time(home, dicts, tmp_path):
    """
    Requested after it bit twice: a bare word list saved as .dic detonates a
    run an hour after it was imported. Refused here instead, while the user is
    holding the file and can act on it.
    """
    bad = tmp_path / "invectives.dic"
    bad.write_text("abnormal\nabusive\n", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        lib.import_into(dicts, bad)
    assert "invectives.dic" in str(err.value)
    assert "category header" in str(err.value)
    assert lib.entries(dicts) == []


def test_an_empty_file_is_refused_at_import_time(home, dicts, tmp_path):
    empty = tmp_path / "hollow.dicx"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError) as err:
        lib.import_into(dicts, empty)
    assert "empty" in str(err.value)


# ---------------------------------------------------------------------------
# The deep check: import asks the parser that will actually read the file
# ---------------------------------------------------------------------------

#: Passes every shape heuristic (non-empty, two % marks) but a word line
#: references category 9, which the header never defines -- the kind of break
#: only the real parser can see.
SUBTLY_BROKEN_DIC = "%\n1\tCats\n%\ncat\t1\t9\n"


def test_import_asks_the_real_parser(home, dicts, tmp_path):
    """
    The heuristics catch what we thought of; the parser catches everything it
    will choke on at run time. Import-time strictness must equal run-time
    reality, or a file sails into the library only to be skipped later.
    """
    pytest.importorskip("contentcoder")
    bad = _a_file(tmp_path, "subtle.dic", SUBTLY_BROKEN_DIC)

    with pytest.raises(ValueError) as err:
        lib.import_into(dicts, bad)
    assert "subtle.dic" in str(err.value)
    assert "parser" in str(err.value)
    assert lib.entries(dicts) == []


def test_the_parser_gate_stays_off_the_screen(home, dicts, tmp_path, capsys):
    """contentcoder prints 'Dictionary loaded.' on success; unmuzzled, that
    lands in the middle of whatever screen asked for the import."""
    pytest.importorskip("contentcoder")
    lib.import_into(dicts, _a_file(tmp_path, "fine.dic", VALID_DIC))
    assert "Dictionary loaded" not in capsys.readouterr().out


def test_without_a_kind_the_check_stays_heuristic(tmp_path):
    """The analyzers call asset_problem() kind-less on purpose: they construct
    the real parser on the very next line, and parsing twice buys nothing."""
    bad = _a_file(tmp_path, "subtle.dic", SUBTLY_BROKEN_DIC)
    assert lib.asset_problem(bad) == ""


def test_the_deep_check_is_the_kinds_own(home, tmp_path):
    """Dispatch goes through kind.deep_check, not a hardcoded format rule --
    that is what lets a future classifier kind bring its own loader."""
    import dataclasses

    fussy = dataclasses.replace(
        lib.KINDS["archetypes"],
        deep_check=lambda p: f"'{p.name}' displeases the deep check.",
    )
    with pytest.raises(ValueError, match="displeases the deep check"):
        lib.import_into(fussy, _a_file(tmp_path, "themes.csv", "a,b\n"))


def test_archetypes_pay_no_parser_at_import():
    """Loading an archetype dictionary means loading a sentence-transformer
    model -- not a price an import screen pays, so the kind declares no deep
    check and its imports stay heuristic-only."""
    assert lib.KINDS["archetypes"].deep_check is None


def test_a_missing_parser_falls_back_to_the_heuristics(home, dicts, tmp_path,
                                                       monkeypatch):
    """Without contentcoder installed, a gate that refused everything would be
    worse than the heuristics alone: the heuristic verdict stands."""
    import sys

    monkeypatch.setitem(sys.modules, "contentcoder", None)
    monkeypatch.setitem(sys.modules, "contentcoder.ContentCoder", None)
    kept = lib.import_into(dicts, _a_file(tmp_path, "fine.dic", VALID_DIC))
    assert kept.exists()


def test_the_stoplists_kind_seeds_the_builtin_lists(home):
    """22 languages plus the two character lists arrive on first use, exactly
    like the other kinds' built-ins -- import once, available everywhere."""
    names = {e.name for e in lib.entries(lib.KINDS["stoplists"])}
    assert {"stopwords-en.txt", "_chars.txt", "_chars_extended.txt"} <= names
    assert len(names) == 24


def test_a_rename_with_a_path_separator_is_refused_not_mangled(home, dicts, tmp_path):
    """Found by fuzzing: Path(new).stem keeps only what follows the last
    separator, so "a/b" silently became "b" and "[bold red]evil[/]" became
    "]". A separator in a *name* is always a mistake; refuse it whole."""
    lib.import_into(dicts, _a_file(tmp_path, "mood.dicx"))
    for hostile in ("a/b", "evil[/]", "back\\slash"):
        with pytest.raises(ValueError, match="cannot contain"):
            lib.rename(dicts, "mood", hostile)
    assert [e.name for e in lib.entries(dicts)] == ["mood.dicx"]


def test_a_missing_entry_is_named_in_the_singular():
    """The message chopped the last letter of the kind's label: "no
    content-coding dictionarie called ..." (a real report)."""
    from taters.helpers.library import _singular

    assert _singular("content-coding dictionaries") == "content-coding dictionary"
    assert _singular("saved models") == "saved model"
    assert _singular("connective lists") == "connective list"
