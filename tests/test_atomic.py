"""
Tests for atomic output writes.

The bug these prevent is unusually nasty because it looks like success.
Analysis steps stream output row by row, so the output file carried its final
name from the first row onwards. Steps also *skip work whose output already
exists* -- that is what makes a long pipeline resumable. Put those together and
an interrupted run leaves a truncated file that the next run accepts as
finished: three rows where three hundred thousand were asked for, no error.

Interruption is not exotic here. Ctrl-C, a full disk, a power cut, or the
cancel key the TUI is heading towards all land in the same place.
"""

from __future__ import annotations

import csv

import pytest

from taters.helpers.atomic import SCRATCH_SUFFIX, atomic_write


def test_the_destination_does_not_exist_until_the_write_finishes(tmp_path):
    target = tmp_path / "out.csv"
    with atomic_write(target, newline="", encoding="utf-8") as fh:
        fh.write("id\n1\n")
        assert not target.exists(), "the real name appeared before the write finished"
    assert target.exists()


def test_an_interrupted_write_leaves_nothing_behind(tmp_path):
    target = tmp_path / "out.csv"

    with pytest.raises(KeyboardInterrupt):
        with atomic_write(target, newline="", encoding="utf-8") as fh:
            fh.write("id\n1\n")
            raise KeyboardInterrupt

    assert not target.exists()
    assert list(tmp_path.iterdir()) == [], "a scratch file was left behind"


def test_keyboard_interrupt_is_caught_even_though_it_is_not_an_exception(tmp_path):
    """
    KeyboardInterrupt inherits from BaseException, not Exception. Catching only
    Exception would leave a scratch file behind in the single most likely case.
    """
    target = tmp_path / "out.csv"
    with pytest.raises(BaseException):
        with atomic_write(target) as fh:
            fh.write("partial")
            raise KeyboardInterrupt
    assert not (tmp_path / f"out.csv{SCRATCH_SUFFIX}").exists()


def test_an_existing_file_is_untouched_by_a_failed_write(tmp_path):
    """A failed refresh must not destroy the good copy that was already there."""
    target = tmp_path / "out.csv"
    target.write_text("id\ngood\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with atomic_write(target, encoding="utf-8") as fh:
            fh.write("id\nhalf")
            raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "id\ngood\n"


def test_a_successful_write_replaces_what_was_there(tmp_path):
    target = tmp_path / "out.csv"
    target.write_text("old\n", encoding="utf-8")
    with atomic_write(target, encoding="utf-8") as fh:
        fh.write("new\n")
    assert target.read_text(encoding="utf-8") == "new\n"


def test_missing_parent_directories_are_created(tmp_path):
    target = tmp_path / "features" / "deep" / "out.csv"
    with atomic_write(target, encoding="utf-8") as fh:
        fh.write("x\n")
    assert target.exists()


def test_the_scratch_file_sits_beside_the_destination(tmp_path):
    """
    A rename is only indivisible within one filesystem. A scratch file in /tmp
    could land on another device, where the rename quietly degrades to a copy
    and the guarantee is lost.
    """
    target = tmp_path / "out.csv"
    with atomic_write(target) as fh:
        scratch = next(p for p in tmp_path.iterdir() if p.name.endswith(SCRATCH_SUFFIX))
        assert scratch.parent == target.parent
        fh.write("x")


# ---------------------------------------------------------------------------
# End to end, through a real analyzer
# ---------------------------------------------------------------------------

@pytest.fixture()
def rows(tmp_path):
    src = tmp_path / "in.csv"
    with src.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "text"])
        for i in range(100):
            w.writerow([f"r{i}", "A sentence with several plain words in it."])
    return src


def test_an_interrupted_analysis_is_redone_rather_than_resumed(tmp_path, rows):
    """
    The whole point. Before this, an interrupted run left a truncated features
    file, and the next run returned it as finished.
    """
    pytest.importorskip("textstat")
    from taters.text.analyze_readability import analyze_readability

    out = tmp_path / "features.csv"

    class Stop(Exception):
        pass

    def stop_partway(done, total, message=None):
        if done == 30:
            raise Stop

    with pytest.raises(Stop):
        analyze_readability(
            csv_path=rows, gathered_csv=tmp_path / "g.csv", out_features_csv=out,
            text_cols=["text"], id_cols=["row"], mode="concat",
            on_progress=stop_partway,
        )

    assert not out.exists(), "a partial features file survived the interruption"

    # now we re-run the way the wizard does: overwrite_existing defaults to False,
    # so if a partial file had survived it would've been handed back as-is.
    analyze_readability(
        csv_path=rows, gathered_csv=tmp_path / "g.csv", out_features_csv=out,
        text_cols=["text"], id_cols=["row"], mode="concat",
    )
    written = len(out.read_text(encoding="utf-8-sig").splitlines()) - 1
    assert written == 100


def test_an_interrupted_gather_is_redone_too(tmp_path, rows):
    """
    The gathered intermediate is skip-if-exists as well, so it needs the same
    protection -- a truncated one would silently shrink every later analysis.
    """
    from taters.helpers.text_gather import csv_to_analysis_ready_csv

    gathered = tmp_path / "g.csv"
    csv_to_analysis_ready_csv(csv_path=rows, out_csv=gathered,
                              text_cols=["text"], id_cols=["row"], mode="concat")
    assert len(gathered.read_text(encoding="utf-8-sig").splitlines()) - 1 == 100
    assert not (tmp_path / f"g.csv{SCRATCH_SUFFIX}").exists()
