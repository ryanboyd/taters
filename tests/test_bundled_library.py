"""
The dictionaries and archetypes that ship inside the package.

These are not test fixtures -- they are the real research instruments users
get "preinstalled," and a broken one is invisible until somebody's run dies
an hour in. So every shipped file is opened by the parser that will actually
read it at run time, and every one has to be cited in the docs, because a
measure nobody can attribute is a measure nobody can publish.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from taters.helpers import library as lib

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "src" / "taters" / "data" / "library"
#: The docs are not in the repository (`/docs` is gitignored), so a fresh
#: clone has none and the two tests that read this skip themselves. They are
#: still worth having: they run where the docs live, which is where somebody
#: adds a dictionary and forgets to cite it.
CITATIONS = ROOT / "docs" / "guides" / "bundled-dictionaries.md"


def _shipped(kind_id: str) -> list[Path]:
    kind = lib.KINDS[kind_id]
    return sorted(f for f in (SHIPPED / kind_id).iterdir()
                  if f.is_file() and f.suffix.lower() in kind.suffixes)


#: The .dicx files are deliberately kept out of the repository (see the
#: `.gitignore` entry for them): they ship inside the wheel, which is built
#: from the working tree, but a clone has none and neither does CI. So the
#: tests that actually measure them skip there rather than failing, the same
#: way the docs tests do -- and still run in full on a tree that has them,
#: which is the tree a release is built from.
needs_dictionaries = pytest.mark.skipif(
    not any((SHIPPED / "dictionaries").glob("*.dicx")),
    reason="the bundled .dicx files are not in this checkout; they ship in the wheel",
)


# ---------------------------------------------------------------------------
# Do they load?
# ---------------------------------------------------------------------------

@needs_dictionaries
def test_every_shipped_dictionary_loads_in_the_real_parser():
    """
    Seeding copies these in with shutil.copy2, which walks straight past the
    import gate every user file has to pass. This is that gate, run at build
    time instead: whatever ships has already been through the parser.
    """
    pytest.importorskip("contentcoder")
    files = _shipped("dictionaries")
    assert len(files) >= 35, "the shipped dictionaries went missing"
    broken = {f.name: lib._contentcoder_problem(f) for f in files}
    assert not {k: v for k, v in broken.items() if v}


def test_every_shipped_archetype_loads_in_the_real_parser():
    """
    Same gate, other shelf. Archetypes have no deep_check of their own
    because the collection is cheap to build but the analyzer around it is
    not -- the loader itself, though, costs nothing and takes no model.
    """
    archetypes = pytest.importorskip("archetypes.archetypes")
    for f in _shipped("archetypes"):
        collection = archetypes.ArchetypeCollection()
        collection.add_archetypes_from_CSV(
            filepath=str(f), file_encoding="utf-8-sig", file_has_headers=True)
        assert collection.archetype_names, f"{f.name} defines no archetypes"
        for name, seeds in collection.archetype_sentences.items():
            # an empty seed still gets encoded, and every text then sits some
            # fixed distance from whatever the model makes of "". the loader
            # takes it without a word, so this is where it gets caught
            assert seeds, f"{f.name}: '{name}' has no seed sentences"
            blank = [i for i, s in enumerate(seeds) if not s.strip()]
            assert not blank, f"{f.name}: '{name}' has blank seeds at {blank}"


@needs_dictionaries
def test_every_shipped_dictionary_has_terms_under_named_categories():
    """
    A .dicx is a CSV whose header names the categories and whose first column
    holds the terms. contentcoder accepts a header-only file without
    complaint, and one would extract a column of zeroes all day.
    """
    csv.field_size_limit(10 ** 9)
    for f in _shipped("dictionaries"):
        with f.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            terms = sum(1 for row in reader if row and row[0].strip())
        assert header[0].strip().lower() == "dicterm", f"{f.name}: {header[:1]}"
        assert len(header) > 1, f"{f.name} names no categories"
        assert terms > 0, f"{f.name} has no terms"


# ---------------------------------------------------------------------------
# Do they arrive, and can they be credited?
# ---------------------------------------------------------------------------

def test_the_shipped_dictionaries_and_archetypes_seed_a_fresh_library(shipped_library):
    """
    "Preinstalled" is the whole point: a user who has never imported anything
    should find these already on the shelf, under the names we shipped. The
    rest of the suite runs with these hidden (see conftest), so this is the
    one place the real seeding gets exercised.
    """
    for kind_id in ("dictionaries", "archetypes"):
        arrived = {e.name for e in lib.entries(lib.KINDS[kind_id])}
        assert arrived == {f.name for f in _shipped(kind_id)}


def _headings() -> list[str]:
    return [line[4:].strip() for line in
            CITATIONS.read_text(encoding="utf-8").splitlines()
            if line.startswith("### ")]


def _stems() -> set[str]:
    return {f.stem for kind in ("dictionaries", "archetypes")
            for f in _shipped(kind)}


def _section_for(stem: str, headings: list[str]) -> list[str]:
    """
    The headings that could be this file's section.

    A heading may say more than the file name does -- "Situational 8
    DIAMONDS" for `Situational 8.dicx` -- so an exact match wins, and
    otherwise the name has to be the start of the heading. That rule alone
    would let "Moral Foundations" claim "Moral Foundations 2.0", which is
    why the exact hit short-circuits.
    """
    if stem in headings:
        return [stem]
    return [h for h in headings if h.startswith(stem + " ")]


@pytest.mark.skipif(not CITATIONS.exists(),
                    reason="docs are not present in this checkout")
def test_every_shipped_asset_is_cited_in_the_docs():
    """
    These are other people's research. Adding one without a citation leaves a
    user with a number they cannot attribute in a paper, so the docs page is
    part of shipping the file, not an optional follow-up.
    """
    headings = _headings()
    for kind_id in ("dictionaries", "archetypes"):
        for f in _shipped(kind_id):
            found = _section_for(f.stem, headings)
            assert len(found) == 1, \
                f"{f.name} matches {found or 'no section'} in {CITATIONS.name}"


@needs_dictionaries
@pytest.mark.skipif(not CITATIONS.exists(),
                    reason="docs are not present in this checkout")
def test_each_cited_dictionary_still_ships():
    """The other direction: a citation for a file we have since renamed or
    dropped sends the reader looking for something that is not there."""
    headings = _headings()
    stems = _stems()
    claimed = {h for stem in stems for h in _section_for(stem, headings)}
    assert set(headings) == claimed, \
        f"cited but not shipped: {sorted(set(headings) - claimed)}"


def test_archetypes_are_managed_through_the_dictionaries_row_not_a_second_one():
    """
    Both shelves are reachable from Settings, and they share one row: the
    dictionaries entry asks which library first. A second row for archetypes
    would say the same thing twice and grow a menu that has to stay short --
    which is what happened, briefly, when these first shipped.
    """
    from taters.ui.tasks.library import _DICTIONARY_KINDS
    from taters.ui.tasks.manage_data import entries

    assert _DICTIONARY_KINDS == ("dictionaries", "archetypes")
    rows = [row.id for row in entries()]
    assert "library" in rows
    assert "archetypes" not in rows, "archetypes already have a door; this is a second one"


# ---------------------------------------------------------------------------
# Upgrades: what a release that changes the built-ins does to a library
# that has been in use
# ---------------------------------------------------------------------------

@pytest.fixture()
def shelf(tmp_path, monkeypatch):
    """
    A pretend package shelf we can add to, edit and re-release.

    The real built-ins are 13 MB and their whole point is that they do not
    change from under a test; what needs exercising here is the reconcile, so
    these stand in for them. Returns a callable that (re)writes the shipped
    folder, standing for cutting a release.
    """
    shipped = tmp_path / "shipped" / "dictionaries"
    shipped.mkdir(parents=True)
    from taters.helpers import library

    monkeypatch.setattr(library, "_shipped_dir", lambda kind: shipped)

    def release(**files: str):
        for f in shipped.iterdir():
            f.unlink()
        for name, text in files.items():
            (shipped / f"{name}.dicx").write_text(text, encoding="utf-8")
        return shipped

    return release


def _names(kind_id: str = "dictionaries") -> set[str]:
    return {e.name for e in lib.entries(lib.KINDS[kind_id])}


def _text(name: str) -> str:
    return (lib.kind_dir(lib.KINDS["dictionaries"]) / name).read_text(encoding="utf-8")


def test_a_release_that_adds_a_dictionary_reaches_a_library_already_in_use(shelf):
    """
    The bug this exists for: seeding fired only when the folder did not exist,
    so everything we ever added afterwards reached new installs and nobody
    else. A user of two years' standing gets the new one.
    """
    shelf(alpha="DicTerm,A\na,X\n")
    assert _names() == {"alpha.dicx"}

    shelf(alpha="DicTerm,A\na,X\n", beta="DicTerm,B\nb,X\n")
    assert _names() == {"alpha.dicx", "beta.dicx"}


def test_a_builtin_the_user_deleted_stays_deleted_across_upgrades(shelf):
    """
    Deleting is a decision, and an update that quietly undoes it is worse than
    one that never arrives -- you would delete it again every release.
    """
    shelf(alpha="DicTerm,A\na,X\n")
    lib.delete(lib.KINDS["dictionaries"], "alpha.dicx")
    assert _names() == set()

    shelf(alpha="DicTerm,A\na,X\nmore,X\n", beta="DicTerm,B\nb,X\n")
    assert _names() == {"beta.dicx"}, "the deleted one came back"


def test_a_corrected_dictionary_replaces_an_untouched_copy(shelf):
    """The other half of keeping people current: fixing a category in a
    shipped dictionary has to reach the people already using it."""
    shelf(alpha="DicTerm,A\na,X\n")
    assert "typo" not in _text("alpha.dicx")

    shelf(alpha="DicTerm,A\na,X\ntypo,X\n")
    assert "typo" in _text("alpha.dicx")


def test_a_copy_the_user_edited_is_never_overwritten_by_an_update(shelf):
    """
    Somebody who tuned a category for their own study has done real work on
    that file. An update that discards it silently is data loss, so the edit
    makes the file theirs and we stop managing it.
    """
    shelf(alpha="DicTerm,A\na,X\n")
    mine = lib.kind_dir(lib.KINDS["dictionaries"]) / "alpha.dicx"
    mine.write_text("DicTerm,A\nmy_own_word,X\n", encoding="utf-8")

    shelf(alpha="DicTerm,A\na,X\ntypo,X\n")
    assert _text("alpha.dicx") == "DicTerm,A\nmy_own_word,X\n"


def test_a_users_own_file_of_the_same_name_is_left_alone(shelf, tmp_path):
    """
    Names collide: somebody's own `Honor.dicx` predates ours shipping one. The
    file that is already there wins, now and in every later release.
    """
    theirs = tmp_path / "alpha.dicx"
    theirs.write_text("DicTerm,Mine\nmine,X\n", encoding="utf-8")
    shelf()                                   # nothing shipped yet
    lib.import_into(lib.KINDS["dictionaries"], theirs)

    shelf(alpha="DicTerm,A\na,X\n")
    assert _text("alpha.dicx") == "DicTerm,Mine\nmine,X\n"
    shelf(alpha="DicTerm,A\na,X\ntypo,X\n")
    assert _text("alpha.dicx") == "DicTerm,Mine\nmine,X\n"


def test_a_file_already_seeded_before_the_ledger_existed_is_adopted(shelf):
    """
    Every library built before the ledger has built-ins in it and no record of
    them -- the stop lists and connectives, which have shipped for ages. A
    file whose bytes are still exactly what we ship is plainly one of ours, so
    it gets adopted on the next look and later releases can update it, rather
    than it counting as the user's work forever.
    """
    shelf(alpha="DicTerm,A\na,X\n")
    assert _names() == {"alpha.dicx"}          # seeded the old way
    (lib.library_home() / lib._LEDGER_NAME).unlink()
    lib.entries(lib.KINDS["dictionaries"])     # rebuilds the record by content

    shelf(alpha="DicTerm,A\na,X\ntypo,X\n")
    assert "typo" in _text("alpha.dicx")


def test_an_unchanged_shelf_costs_no_hashing(shelf, monkeypatch):
    """
    kind_dir() runs on every library access, and the built-ins are 13 MB. The
    name-and-size stamp is what keeps a reconcile from happening on each one;
    without it every menu that lists dictionaries would hash the lot.
    """
    shelf(alpha="DicTerm,A\na,X\n")
    lib.entries(lib.KINDS["dictionaries"])

    calls = []
    monkeypatch.setattr(lib, "_digest", lambda p: calls.append(p) or "x")
    for _ in range(3):
        lib.entries(lib.KINDS["dictionaries"])
    assert calls == []


def test_an_interrupted_install_leaves_no_half_written_dictionary(shelf):
    """
    Seeding copies files into somebody's library, and a copy that dies midway
    -- Ctrl-C on a slow first run, a full disk, a closed laptop -- must not
    leave a truncated dictionary under a name claiming it is whole. It would
    parse as far as the cut and then fail in the middle of an analysis.

    So the copy goes to a scratch name and is renamed: at every instant the
    destination is the old file or the new one, never half of either. And
    because the ledger is only written at the end, the next run simply tries
    again.
    """
    import shutil as _shutil

    old, new = "DicTerm,A\na,X\n", "DicTerm,A\na,X\nb,X\nc,X\n"
    shelf(alpha=old)
    assert _names() == {"alpha.dicx"}          # installed cleanly once
    folder = lib.kind_dir(lib.KINDS["dictionaries"])

    # a release that would update it, interrupted mid-copy. the ledger keeps
    # the old digest, which is what says the copy on disk is still ours to
    # replace -- that is the branch this exercises
    shelf(alpha=new)

    def die(src, dst):
        Path(dst).write_text("DicTerm,A\na,", encoding="utf-8")   # a torn file
        raise KeyboardInterrupt("laptop lid")

    # its own MonkeyPatch rather than the `monkeypatch` fixture: undo() on the
    # fixture reverts *every* patch in the test, the autouse hermetic
    # TATERS_HOME included, and the assertions then read -- and seed -- the
    # developer's own library. Which is exactly what happened while writing this
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_shutil, "copy2", die)
        with pytest.raises(KeyboardInterrupt):
            lib.entries(lib.KINDS["dictionaries"])

        # look now, while the copy is still broken: this is the instant the
        # user would be left in if they closed the lid
        assert (folder / "alpha.dicx").read_text(encoding="utf-8") == old, \
            "the destination was written in place"
        assert sorted(p.name for p in folder.iterdir()) == ["alpha.dicx"], \
            "a scratch file was left behind"

    # and the run after it finishes the job, because nothing was recorded
    assert _text("alpha.dicx") == new
