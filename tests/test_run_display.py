"""
Tests for the live run display and the finished screen.

These exist because of a specific failure report: a single-step pipeline sat on
an unchanging line for minutes and looked hung. The user's instinct then is
Ctrl-C, which is the one action guaranteed to throw the work away. Visible
proof of life is not decoration here.
"""

from __future__ import annotations

import contextlib
import io
import time

import pytest
from rich.console import Console

from taters.ui.run_display import RunDisplay, _short_call, reporter_for
from taters.ui.prompts import QuitRequested, ScriptedPrompter
from taters.ui.wizard import _outputs_in, finish_screen


@pytest.fixture()
def display():
    console = Console(file=io.StringIO(), width=80)
    with RunDisplay(console) as d:
        yield d


def test_an_item_step_gets_a_determinate_bar(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.convert_to_wav",
       scope="item", items=4)

    task = display._progress.tasks[-1]
    assert task.total == 4


def test_a_global_step_pulses_rather_than_sitting_at_zero(display):
    """
    A GLOBAL step is one call and cannot report how far through it is. A bar
    frozen at 0% reads as stuck; an indeterminate one reads as working.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global", items=None)

    task = display._progress.tasks[-1]
    assert task.total is None


def test_overall_progress_advances_once_per_step(display):
    ev = reporter_for(display)
    ev("run_start", steps=3, inputs=[])
    for i in range(1, 4):
        ev("step_start", index=i, total=3, call="potato.x.y", scope="global")
        ev("step_done", index=i, call="potato.x.y", status="ok")

    overall = display._progress.tasks[0]
    assert overall.completed == 3
    assert overall.total == 3


def test_failures_are_collected_rather_than_printed_mid_run(display):
    """
    Writing into the area a live display owns corrupts it, and a per-file error
    reads better at the end next to the final count.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.convert_to_wav",
       scope="item", items=2)
    ev("item_done", index=1, item=0, input="/data/one.mp4", status="error",
       error="ffmpeg exploded")
    ev("item_done", index=1, item=1, input="/data/two.mp4", status="ok")

    assert display.failures == ["one.mp4 — ffmpeg exploded"]


def test_a_step_error_is_collected_too(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability", scope="global")
    ev("step_done", index=1, call="potato.text.analyze_readability",
       status="error", error="no such column")

    assert display.failures == ["analyze readability — no such column"]


def test_the_step_label_is_readable():
    assert _short_call("potato.text.analyze_readability") == "analyze readability"
    assert _short_call("convert_to_wav") == "convert to wav"


def test_events_without_a_display_are_harmless():
    """The scripted prompter has no console, so there is no display to drive."""
    reporter_for(None)("step_start", index=1, total=1, call="x", scope="global")


# ---------------------------------------------------------------------------
# the finished screen
# ---------------------------------------------------------------------------

def test_a_clean_run_says_so_and_names_the_folder(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "readability.csv").write_text("a,b\n", encoding="utf-8")

    p = ScriptedPrompter(["menu"])
    finish_screen(p, ok=True, manifest={"items": [{"status": "ok"}], "errors": []},
                  folder=tmp_path, manifest_path=tmp_path / "run_manifest.json")

    assert "Finished. Everything succeeded" in p.text_output
    assert str(tmp_path) in p.text_output
    assert "features/readability.csv" in p.text_output.replace("\\", "/")


def test_choosing_finish_ends_the_session(tmp_path):
    """
    Distinct from backing out: the work is done, and reporting it as cancelled
    would be a lie about a run that succeeded.
    """
    p = ScriptedPrompter(["quit"])
    with pytest.raises(QuitRequested):
        finish_screen(p, ok=True, manifest={"items": [], "errors": []},
                      folder=tmp_path, manifest_path=tmp_path / "m.json")


def test_failures_are_shown_on_the_finished_screen(tmp_path):
    p = ScriptedPrompter(["menu"])
    finish_screen(p, ok=False,
                  manifest={"items": [{"status": "error"}], "errors": []},
                  folder=tmp_path, manifest_path=tmp_path / "m.json",
                  failures=["one.mp4 — ffmpeg exploded"])

    assert "Finished with problems" in p.text_output
    assert "ffmpeg exploded" in p.text_output


def test_a_long_list_of_failures_is_capped(tmp_path):
    p = ScriptedPrompter(["menu"])
    finish_screen(p, ok=False, manifest={"items": [], "errors": []},
                  folder=tmp_path, manifest_path=tmp_path / "m.json",
                  failures=[f"file{i}.mp4 — boom" for i in range(12)])

    assert "and 7 more" in p.text_output


def test_only_result_files_are_listed(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "features" / "out.csv").write_text("x", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "scratch.wav").write_bytes(b"RIFF")

    listed = _outputs_in(tmp_path)
    assert any("out.csv" in row for row in listed)
    assert not any("run_manifest" in row for row in listed)
    assert not any("scratch.wav" in row for row in listed)


# ---------------------------------------------------------------------------
# real progress from a step that can count itself
# ---------------------------------------------------------------------------

def test_an_indeterminate_task_shows_no_count_rather_than_a_question_mark(display):
    """
    rich's own M-of-N column renders `0/?` when the total is unknown, which
    reads as a bug rather than as "this step cannot count itself". The spinner
    and the clock already carry the liveness.
    """
    from taters.ui.run_display import _CountColumn

    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global")

    task = display._progress.tasks[-1]
    assert str(_CountColumn().render(task)) == ""


def test_a_step_that_counts_itself_gets_a_real_bar(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global")
    ev("step_progress", index=1, call="x", done=0, total=400)
    ev("step_progress", index=1, call="x", done=137, total=400)

    task = display._progress.tasks[-1]
    assert task.total == 400
    assert task.completed == 137


def test_the_runner_offers_progress_to_a_step_that_wants_it():
    from taters.pipelines.run_pipeline import _accepts_on_progress

    def counts(a, on_progress=None):
        pass

    def cannot(a):
        pass

    def facade(**kwargs):
        pass

    assert _accepts_on_progress(counts) is True
    assert _accepts_on_progress(cannot) is False
    # the Taters facade forwards **kwargs, which hides the real signature. it's
    # safe to offer anyway, since `_forward` drops it again for targets that
    # can't take it
    assert _accepts_on_progress(facade) is True


def test_the_facade_drops_progress_for_a_target_that_cannot_take_it():
    """
    `_forward` is strict about unexpected arguments on purpose -- a typo in a
    preset should be reported, not ignored. `on_progress` is the exception: it
    is injected by the runner, not written by anyone, so a target that does not
    support it should simply not get it.
    """
    from taters.Taters import _forward

    def plain(x):
        return x

    assert _forward(plain, {"x": 1, "on_progress": lambda *_: None}) == 1

    with pytest.raises(TypeError, match="unexpected keyword|Allowed params"):
        _forward(plain, {"x": 1, "typo": 2})


def test_an_analyzer_reports_row_progress(tmp_path):
    """
    The end-to-end shape the user asked for: a bar that moves through the rows
    of a spreadsheet rather than a spinner that only says "still going".
    """
    pytest.importorskip("textstat")
    import csv as _csv

    from taters.text.analyze_readability import analyze_readability

    source = tmp_path / "many.csv"
    with source.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["row", "text"])
        for i in range(25):
            w.writerow([f"r{i}", "A sentence with several plain words in it."])

    seen = []
    analyze_readability(
        csv_path=source,
        gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "features.csv",
        text_cols=["text"],
        id_cols=["row"],
        mode="concat",
        overwrite_existing=True,
        on_progress=lambda done, total, message=None: seen.append((done, total, message)),
    )

    # the phases before measuring report with total=None and a message, then the
    # measuring itself reports a real position
    phases = [m for _, total, m in seen if total is None and m]
    assert "reading the input" in phases

    counted = [(d, t) for d, t, _ in seen if t is not None]
    assert counted[0] == (0, 25), "the total is known before the first row"
    assert counted[-1] == (25, 25)
    assert [d for d, _ in counted] == sorted(d for d, _ in counted)


# ---------------------------------------------------------------------------
# the silent phases before measuring starts
# ---------------------------------------------------------------------------

def test_the_phase_is_named_while_the_total_is_unknown(display):
    """
    On a large spreadsheet the passes *before* measuring -- reading it, then
    counting its rows -- take long enough to look like a hang. There is nothing
    else on that line to explain the wait, so it has to say so.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global")
    ev("step_progress", index=1, call="x", done=0, total=None,
       message="reading the input")

    assert display._progress.tasks[-1].description == "analyze readability · reading the input"


def test_a_running_tally_is_shown_while_counting(display):
    """
    The number climbing is the only proof that a long silent pass is doing
    anything -- which is the whole reason to report it before the total exists.
    """
    from taters.ui.run_display import _CountColumn

    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global")
    ev("step_progress", index=1, call="x", done=250000, total=None,
       message="counting rows")

    rendered = str(_CountColumn().render(display._progress.tasks[-1]))
    assert rendered == "250,000…"


def test_the_phase_label_is_dropped_once_measuring_starts(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global")
    ev("step_progress", index=1, call="x", done=0, total=None, message="counting rows")
    ev("step_progress", index=1, call="x", done=10, total=400, message=None)

    task = display._progress.tasks[-1]
    assert task.description == "analyze readability"
    assert task.total == 400


def test_big_numbers_are_grouped(display):
    from taters.ui.run_display import _CountColumn

    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.x.y", scope="global")
    ev("step_progress", index=1, call="x", done=137204, total=300000)

    assert str(_CountColumn().render(display._progress.tasks[-1])) == "137,204/300,000"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_counting_reports_before_it_finishes(tmp_path):
    """
    A tally that only arrives at the end is no better than silence. The pass
    reports every 1,000 records (records, not lines: one record can be a whole
    book), so this uses a file big enough to cross that many times.

    5,000 rows, not the 60,000 this used to write. Every extra row was paid
    for twice -- once writing the file, once running real readability over it
    -- for 9.5 seconds that proved nothing the fifth crossing had not already
    proved. The assertion below is now *stronger* than the one those 60,000
    rows bought: it pins the whole sequence of tallies rather than only the
    first.
    """
    pytest.importorskip("textstat")
    import csv as _csv

    from taters.text.analyze_readability import analyze_readability

    source = tmp_path / "big.csv"
    with source.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["row", "text"])
        for i in range(5_000):
            w.writerow([f"r{i}", "Short text."])

    seen = []
    analyze_readability(
        csv_path=source,
        gathered_csv=tmp_path / "gathered.csv",
        out_features_csv=tmp_path / "features.csv",
        text_cols=["text"], id_cols=["row"], mode="concat",
        overwrite_existing=True,
        on_progress=lambda done, total, message=None: seen.append((done, total, message)),
    )

    tallies = [d for d, total, m in seen if total is None and m == "counting rows"]
    assert tallies, "the counting pass reported nothing"
    # every thousand, in order, not just the first. a counter that fired once
    # and then went quiet would've passed the old assertion
    assert tallies == [1_000, 2_000, 3_000, 4_000, 5_000]


# ---------------------------------------------------------------------------
# one bar per concurrent file
# ---------------------------------------------------------------------------

def test_a_bar_appears_for_each_file_in_flight(display):
    """
    Without per-file bars a viewer sees "3 of 40 done" and no sign of which
    files are moving. On a long transcription that is the difference between
    "working" and "hung".
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=40)
    for i in range(4):
        ev("item_start", index=1, call="x", item=i, input=f"/data/lecture_{i}.mp4")

    assert len(display._slots) == 4
    labels = [t.description for t in display._progress.tasks]
    assert any("lecture_2" in label for label in labels)


def test_the_rows_are_bounded_by_workers_not_by_files(display):
    """
    Forty files across four workers is four rows, not forty. The display
    follows what is actually in flight.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=40)

    inflight = []
    peak = 0
    for i in range(40):
        ev("item_start", index=1, call="x", item=i, input=f"/data/f{i}.mp4")
        inflight.append(i)
        peak = max(peak, len(display._slots))
        if len(inflight) >= 4:
            done = inflight.pop(0)
            ev("item_done", index=1, item=done, input=f"/data/f{done}.mp4", status="ok")

    assert peak == 4


def test_a_high_worker_count_is_summarized_rather_than_listed(display):
    """
    Sixteen workers would otherwise push the whole display down the screen.
    Four named, the rest counted -- the height stays fixed whatever the setting.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=200)
    for i in range(16):
        ev("item_start", index=1, call="x", item=i, input=f"/data/f{i:02d}.mp4")

    assert len(display._slots) == 4
    assert len(display._inflight) == 16
    labels = [t.description for t in display._progress.tasks]
    assert any("…and 12 more files" in label for label in labels)


def test_a_freed_slot_is_given_to_a_waiting_file(display):
    """
    And to the *oldest* waiting file. A file running far longer than its
    neighbors is the one worth seeing, and newest-first would hide it.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=200)
    for i in range(8):
        ev("item_start", index=1, call="x", item=i, input=f"/data/f{i:02d}.mp4")

    assert [display._inflight[k] for k in display._slots] == [
        "f00", "f01", "f02", "f03"]

    ev("item_done", index=1, item=0, input="/data/f00.mp4", status="ok")
    assert [display._inflight[k] for k in display._slots] == [
        "f01", "f02", "f03", "f04"]


def test_the_summary_disappears_when_it_no_longer_applies(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=8)
    for i in range(8):
        ev("item_start", index=1, call="x", item=i, input=f"/data/f{i}.mp4")
    assert display._overflow is not None

    for i in range(8):
        ev("item_done", index=1, item=i, input=f"/data/f{i}.mp4", status="ok")

    assert display._overflow is None
    assert display._slots == {}
    assert display._inflight == {}


def test_a_finished_file_gives_its_bar_back(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=10)
    ev("item_start", index=1, call="x", item=0, input="/data/a.mp4")
    ev("item_start", index=1, call="x", item=1, input="/data/b.mp4")
    ev("item_done", index=1, item=0, input="/data/a.mp4", status="ok")

    assert set(display._slots) == {1}
    assert display._progress.tasks[1].completed == 1, "the step bar still advanced"


def test_slots_do_not_survive_a_step_boundary(display):
    """A file bar from the previous step is stale the moment the step changes."""
    ev = reporter_for(display)
    ev("run_start", steps=2, inputs=[])
    ev("step_start", index=1, total=2, call="potato.audio.x", scope="item", items=2)
    ev("item_start", index=1, call="x", item=0, input="/data/a.mp4")
    ev("step_start", index=2, total=2, call="potato.text.y", scope="global")

    assert display._slots == {}


def test_the_runner_announces_a_file_when_a_worker_picks_it_up(tmp_path):
    """
    `item_start` fires from inside the worker, which is what makes a bar per
    concurrent slot possible. Threads share memory so the callback works;
    processes would have to pickle it, so it is not offered there.
    """
    from taters.pipelines.run_pipeline import _run_item_step

    seen = []
    step = {"scope": "item", "call": "nope.does.not.exist", "with": {}}

    # the step itself can't resolve, which is fine. what we care about is that
    # the announcement happens on entry, before any work gets attempted. a file
    # that fails still has to show up as "started", or its bar never shows at all
    with contextlib.suppress(Exception):
        _run_item_step(3, tmp_path / "x.wav", step, object(), {}, {}, {},
                       lambda i, path: seen.append((i, path)))

    assert seen == [(3, str(tmp_path / "x.wav"))]


def test_the_worker_runs_fine_without_a_callback(tmp_path):
    """The process engine passes None; that must not be a special case."""
    from taters.pipelines.run_pipeline import _run_item_step

    step = {"scope": "item", "call": "nope.does.not.exist", "with": {}}

    # no callback, and its absence shouldn't change the code path at all: the
    # same failure comes back just like it did before per-file bars existed
    with pytest.raises(Exception):
        _run_item_step(0, tmp_path / "x.wav", step, object(), {}, {}, {}, None)


# ---------------------------------------------------------------------------
# concurrency, and the bugs a single-threaded test can never see
# ---------------------------------------------------------------------------

def test_two_workers_starting_at_once_do_not_orphan_a_bar(display):
    """
    A real report: a four-file run showed the same filename three times, under
    a step that had finished minutes earlier.

    `item_start` fires from inside the worker thread, so `_fill_slots` runs
    concurrently, and its "not in `_slots`, so `add_task`" is a check-then-act.
    Two workers both find a file unslotted, both add a bar, and the second
    assignment to `_slots[item]` drops the first bar's id on the floor. Nothing
    holds it, so nothing removes it -- not `item_done`, not the step boundary.

    The window between the check and the act is a few bytecodes wide, so the
    GIL hides it from a test that merely starts threads together. Slowing
    `add_task` down widens it to something reproducible; the hazard is the
    same one, just visible.
    """
    import threading
    import time

    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.convert_to_wav",
       scope="item", items=4)

    real_add = display._progress.add_task

    def slow_add(*args, **kwargs):
        time.sleep(0.02)
        return real_add(*args, **kwargs)

    display._progress.add_task = slow_add

    ready = threading.Barrier(4)

    def announce(i):
        ready.wait()
        ev("item_start", index=1, call="x", item=i, input=f"/data/{i}.mkv")

    threads = [threading.Thread(target=announce, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # overall, the step bar, and exactly one bar per file. anything more is a
    # bar no dictionary points at, and that's a bar that never goes away
    assert len(display._progress.tasks) == 2 + 4
    assert len(display._slots) == 4


def test_a_step_boundary_removes_bars_the_display_lost_track_of(display):
    """
    Defense in depth for the same failure. Even if a bar goes missing from
    `_slots`, it must not survive into the next step: a bar naming a file that
    is not being worked on is worse than no bar at all.
    """
    ev = reporter_for(display)
    ev("run_start", steps=2, inputs=[])
    ev("step_start", index=1, total=2, call="potato.audio.x", scope="item", items=2)
    ev("item_start", index=1, call="x", item=0, input="/data/a.mkv")

    # let's fake the orphan: a real bar that nothing has a handle on any more
    display._progress.add_task("  ↳ ghost.mkv", total=None)
    display._slots.clear()

    ev("step_start", index=2, total=2, call="potato.text.y", scope="global")

    descriptions = [t.description for t in display._progress.tasks]
    assert not any("ghost" in d for d in descriptions)
    assert not any("↳" in d for d in descriptions)


def test_files_are_counted_as_they_finish_not_after_the_pool_drains(tmp_path):
    """
    The other half of the same screenshot: `transcribe with whisper  0/4` after
    two and a half minutes, with all four files still listed as in flight.

    The runner used to collect every future's result and only *then* replay
    `item_done` for the successes. So the step's counter read 0/N for the whole
    step and jumped to N/N at the end -- on a step that takes minutes per file,
    exactly the "is it hung?" signal this display exists to remove.

    The shape of the test is what makes it meaningful: one file finishes at
    once while another is still working, and the assertion is that the finished
    one was *reported* while the other was still going.
    """
    import threading
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    release = threading.Event()
    slow_finished = threading.Event()
    reported_early = []

    def fake_item_step(i, p, step, potato, item_artifacts, globals_ctx,
                       vars_ctx, on_start=None, on_progress=None, quiet=False):
        if on_start is not None:
            on_start(i, str(p))
        if i == 0:
            # we hold this open until the other file's been reported, so "reported
            # while still running" is a fact rather than a matter of timing
            release.wait(timeout=2)
            slow_finished.set()
        return (i, "ok", {}, {})

    def on_event(name, **payload):
        if name == "item_done" and payload["item"] == 1:
            reported_early.append(not slow_finished.is_set())
            release.set()

    preset = {"steps": [{"scope": "item", "call": "potato.audio.convert_to_wav",
                         "save_as": "wav", "with": {"input_path": "{{input}}"}}]}
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"")

    with mock.patch.object(run_pipeline, "_run_item_step", fake_item_step):
        manifest = run_pipeline.run_preset(
            preset, root_dir=tmp_path, file_type="any", workers=2,
            out_manifest=tmp_path / "m.json", on_event=on_event, verbose=False,
        )

    assert reported_early == [True], "item_done waited for the whole pool to drain"
    assert [i["status"] for i in manifest["items"]] == ["ok", "ok"]


# ---------------------------------------------------------------------------
# progress within a single file
#
# an ITEM step gets counted from the outside (files finished out of files
# found), which says nothing at all while one file is running. so a four-file
# transcription read `0/4` until the first file *completed*, and somebody's
# two-hour stall looked exactly like a working run right up until they gave
# up on it
# ---------------------------------------------------------------------------

def slot_task(display, item):
    """
    The bar for one in-flight file.

    By id, not by position: `Progress.tasks` is a list, and the two stop
    agreeing the moment a task is removed -- which is exactly what the
    promotion test does on purpose.
    """
    return display._progress._tasks[display._slots[item]]


def test_a_file_reporting_its_position_shows_it(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=2)
    ev("item_start", index=1, item=0, input="/data/lecture.mkv")
    ev("item_progress", index=1, item=0, input="/data/lecture.mkv",
       done=252, total=771, message=None, unit="seconds")

    task = slot_task(display, 0)
    assert (task.completed, task.total) == (252, 771)


def test_a_phase_with_no_size_yet_is_named_rather_than_counted(display):
    """
    Loading a model and scanning for speech both take real time and neither has
    a denominator. Zero out of a known total reads as stalled; a pulsing bar
    with a phase name reads as work.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=1)
    ev("item_start", index=1, item=0, input="/data/lecture.mkv")
    ev("item_progress", index=1, item=0, input="/data/lecture.mkv",
       done=0, total=None, message="loading tiny on cpu (int8)", unit=None)

    task = slot_task(display, 0)
    assert task.total is None
    assert "loading tiny on cpu (int8)" in task.description


def test_a_file_promoted_into_a_slot_keeps_the_progress_it_already_had(display):
    """
    With more files in flight than rows on screen, a file can start reporting
    long before it has a bar. A bar that appears at zero and immediately jumps
    is worse than one that appears where the work actually is.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=6)
    for i in range(5):
        ev("item_start", index=1, item=i, input=f"/data/f{i}.mkv")

    hidden = next(i for i in range(5) if i not in display._slots)
    ev("item_progress", index=1, item=hidden, input=f"/data/f{hidden}.mkv",
       done=300, total=771, message=None, unit="seconds")
    # free up a slot so the hidden file gets promoted into it
    shown = next(iter(display._slots))
    ev("item_done", index=1, item=shown, input=f"/data/f{shown}.mkv", status="ok")

    assert slot_task(display, hidden).completed == 300


def test_time_based_progress_is_rendered_as_a_clock(display):
    """
    `252/771` on a transcription invites exactly the wrong reading -- segments,
    not seconds. `4:12/12:51` cannot be misread.
    """
    from taters.ui.run_display import _CountColumn

    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=1)
    ev("item_start", index=1, item=0, input="/data/lecture.mkv")
    ev("item_progress", index=1, item=0, input="/data/lecture.mkv",
       done=252, total=771, message=None, unit="seconds")

    task = slot_task(display, 0)
    assert str(_CountColumn().render(task)) == "4:12/12:51"


def test_countable_work_is_still_rendered_as_a_tally(display):
    """The default: rows are counted, not clocked."""
    from taters.ui.run_display import _CountColumn

    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_readability",
       scope="global")
    ev("step_progress", index=1, call="x", done=4000, total=90000)

    assert str(_CountColumn().render(display._progress.tasks[-1])) == "4,000/90,000"


def test_a_finished_file_stops_being_tracked(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=1)
    ev("item_start", index=1, item=0, input="/data/lecture.mkv")
    ev("item_progress", index=1, item=0, input="/data/lecture.mkv",
       done=252, total=771, message=None, unit="seconds")
    ev("item_done", index=1, item=0, input="/data/lecture.mkv", status="ok")

    assert display._itemprog == {}


def test_a_warning_is_held_back_rather_than_written_over_the_display():
    """
    A warning goes to stderr, so silencing a step's `verbose` does not silence
    it -- and pandas and sentence-transformers both warn during an ordinary
    run. Landing in the region a live display owns corrupts it just as a print
    does, so they are collected, deduplicated, and shown once at the end.
    """
    import warnings

    buf = io.StringIO()
    with RunDisplay(Console(file=buf, width=80)) as d:
        for _ in range(3):
            warnings.warn("frame is fragmented", UserWarning)
        assert d.notices == ["UserWarning: frame is fragmented"]

    assert "frame is fragmented" in buf.getvalue()
    # and the hook gets handed back, so a later warning is nobody's business here
    assert warnings.showwarning.__name__ != "_collect"


# ---------------------------------------------------------------------------
# what the runner hands a step
# ---------------------------------------------------------------------------

def _one_item_preset(call: str) -> dict:
    return {"steps": [{"scope": "item", "call": call, "with": {"path": "{{input}}"}}]}


def test_an_item_step_is_given_a_progress_sink(tmp_path):
    """
    `on_progress` used to be wired for GLOBAL steps only, so the longest thing
    in most pipelines -- transcription, an ITEM step -- was the one thing that
    could not say how far along it was.
    """
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    seen = []
    (tmp_path / "a.wav").write_bytes(b"")

    def step(path, on_progress=None, **kw):
        assert on_progress is not None
        on_progress(30, 120, None, "seconds")
        return path

    with mock.patch.object(run_pipeline, "resolve_call", lambda call, potato: step):
        run_pipeline.run_preset(
            _one_item_preset("taters.fake:step"), root_dir=tmp_path, file_type="any",
            out_manifest=tmp_path / "m.json", verbose=False, vars_ctx={},
            on_event=lambda n, **p: seen.append((n, p)),
        )

    progress = [p for n, p in seen if n == "item_progress"]
    assert progress and progress[0]["done"] == 30
    assert progress[0]["total"] == 120
    assert progress[0]["unit"] == "seconds"


def test_a_quiet_run_stops_the_steps_printing_too(tmp_path):
    """
    `run_preset(verbose=False)` used to silence only the runner's own lines. The
    step functions kept their own `verbose=True` default and printed straight
    into the region the live display owns -- 638 lines for a four-file
    transcription, over the top of the bars meant to replace them.
    """
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    (tmp_path / "a.wav").write_bytes(b"")
    got = {}

    def step(path, verbose=True, **kw):
        got["verbose"] = verbose
        return path

    with mock.patch.object(run_pipeline, "resolve_call", lambda call, potato: step):
        run_pipeline.run_preset(
            _one_item_preset("taters.fake:step"), root_dir=tmp_path, file_type="any",
            out_manifest=tmp_path / "m.json", verbose=False, vars_ctx={},
            on_event=lambda n, **p: None,
        )

    assert got["verbose"] is False


def test_a_loud_run_leaves_a_quiet_step_alone(tmp_path):
    """
    The asymmetry matters. Silence is forced down; noise is not. Pushing the
    runner's `verbose=True` into every step would make a plain CLI run start
    printing things it has never printed, for any step whose own default is
    quieter than the runner's.
    """
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    (tmp_path / "a.wav").write_bytes(b"")
    got = {}

    def step(path, verbose=False, **kw):
        got["verbose"] = verbose
        return path

    with mock.patch.object(run_pipeline, "resolve_call", lambda call, potato: step):
        run_pipeline.run_preset(
            _one_item_preset("taters.fake:step"), root_dir=tmp_path, file_type="any",
            out_manifest=tmp_path / "m.json", verbose=True, vars_ctx={},
        )

    assert got["verbose"] is False


def test_a_step_that_sets_verbose_itself_wins(tmp_path):
    """Someone who wrote `verbose: true` into a preset step meant it."""
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    (tmp_path / "a.wav").write_bytes(b"")
    got = {}

    def step(path, verbose=True, **kw):
        got["verbose"] = verbose
        return path

    preset = _one_item_preset("taters.fake:step")
    preset["steps"][0]["with"]["verbose"] = True

    with mock.patch.object(run_pipeline, "resolve_call", lambda call, potato: step):
        run_pipeline.run_preset(
            preset, root_dir=tmp_path, file_type="any",
            out_manifest=tmp_path / "m.json", verbose=False, vars_ctx={},
            on_event=lambda n, **p: None,
        )

    assert got["verbose"] is True


def test_nothing_listening_means_no_progress_sink_to_pay_for(tmp_path):
    """
    Offering a step a progress sink is not free -- counting rows for a
    denominator is a whole extra pass over the input. A plain `--preset` run on
    the command line has never paid for that and must not start.
    """
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    (tmp_path / "a.wav").write_bytes(b"")
    got = {}

    def step(path, on_progress=None, **kw):
        got["on_progress"] = on_progress
        return path

    with mock.patch.object(run_pipeline, "resolve_call", lambda call, potato: step):
        run_pipeline.run_preset(
            _one_item_preset("taters.fake:step"), root_dir=tmp_path, file_type="any",
            out_manifest=tmp_path / "m.json", verbose=True, vars_ctx={},
        )

    assert got["on_progress"] is None


# ---------------------------------------------------------------------------
# what a row says, and how much room it takes
# ---------------------------------------------------------------------------

def test_a_row_names_the_item_not_the_file_the_step_has_open(display):
    """
    The event carries the file the *pipeline* was handed. By the time Whisper
    runs, that is not what Whisper is reading -- `convert_to_wav` has been and
    gone, and the step's input is a `.wav`. A row reading
    "transcribe with whisper ↳ lecture.mkv" states something untrue, so the
    extension goes and the stem, which stays true for every step, remains.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
       scope="item", items=1)
    ev("item_start", index=1, item=0, input="/vids/lecture.mkv")

    description = slot_task(display, 0).description
    assert "lecture" in description
    assert ".mkv" not in description


def test_a_long_name_does_not_squeeze_out_the_bar_and_the_clock():
    """
    From a real screenshot: four rows of Crash Course filenames, each one long
    enough to push everything else off the edge, leaving four identical
    `━━━━ 0:0…` stubs. The bar and the clock are the parts that show the run is
    moving; the fiftieth character of a filename is not.
    """
    from rich.cells import cell_len

    console = Console(file=io.StringIO(), width=100)
    with RunDisplay(console) as d:
        ev = reporter_for(d)
        ev("run_start", steps=1, inputs=[])
        ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=1)
        ev("item_start", index=1, item=0,
           input="/vids/Cathedrals and Universities： Crash Course History of "
                 "Science #11 [0wDlLwLIFeI].mkv")

        assert cell_len(slot_task(d, 0).description) <= 100 - 40


def test_a_shortened_name_keeps_both_ends():
    """
    Which end identifies a file depends on the corpus: a numbered series differs
    at the tail, a dated export at the head. Cutting from the middle keeps
    whichever it is, so four rows of near-identical names stay distinguishable.
    """
    from taters.ui.run_display import _fit

    name = "Cathedrals and Universities： Crash Course History of Science #11 [0wDlLwLIFeI]"
    short = _fit(name, 40)
    assert short.startswith("Cathedrals")
    assert short.endswith("0wDlLwLIFeI]")
    assert "…" in short


def test_a_name_that_already_fits_is_left_exactly_alone():
    from taters.ui.run_display import _fit

    assert _fit("lecture", 40) == "lecture"


def test_fullwidth_characters_are_measured_as_two_columns():
    """
    The names that make shortening necessary are the ones most likely to contain
    fullwidth punctuation: a title downloaded from the web arrives with `：` in
    place of a colon the filesystem would refuse, and each one takes two columns
    while counting as a single character. Measuring with `len` overshoots on
    exactly the rows that were already too long.
    """
    from rich.cells import cell_len

    from taters.ui.run_display import _fit

    assert cell_len(_fit("：" * 40, 20)) <= 20


# ---------------------------------------------------------------------------
# a failure you can see while it's still relevant
#
# this one got reported to us: four files went into a transcription step, one
# dropped out after two seconds, and the run then stalled with three still in
# flight. the reason that one file failed was sitting in `failures`, waiting
# for a summary the run never got to
# ---------------------------------------------------------------------------

def test_a_failed_file_is_reported_when_it_fails_not_only_at_the_end():
    buf = io.StringIO()
    with RunDisplay(Console(file=buf, width=100)) as d:
        ev = reporter_for(d)
        ev("run_start", steps=1, inputs=[])
        ev("step_start", index=1, total=1, call="potato.audio.transcribe_with_whisper",
           scope="item", items=4)
        ev("item_start", index=1, item=0, input="/vids/one.mkv")
        ev("item_done", index=1, item=0, input="/vids/one.mkv", status="error",
           error="cuDNN could not be loaded")

        # on screen right away, while the other three are still running
        assert "cuDNN could not be loaded" in buf.getvalue()
        assert "one.mkv" in buf.getvalue()


def test_a_reported_failure_is_still_counted_for_the_summary():
    """Said twice, on purpose: once when it happens, once next to the count."""
    buf = io.StringIO()
    with RunDisplay(Console(file=buf, width=100)) as d:
        ev = reporter_for(d)
        ev("run_start", steps=1, inputs=[])
        ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=2)
        ev("item_done", index=1, item=0, input="/vids/one.mkv", status="error",
           error="boom")

        assert d.failures == ["one.mkv — boom"]


def test_a_successful_file_says_nothing(display):
    """Only failures earn a line. A per-file success is what the bar is for."""
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=2)
    ev("item_done", index=1, item=0, input="/vids/one.mkv", status="ok")

    assert display.failures == []


def test_a_bar_that_has_already_gone_does_not_cost_the_count(display):
    """
    `emit` swallows whatever this raises, so an exception between removing the
    bar and advancing the counter would freeze the count in place -- which is
    indistinguishable from the hang the whole display exists to rule out.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.audio.x", scope="item", items=2)
    ev("item_start", index=1, item=0, input="/vids/one.mkv")

    # whip the bar away behind the display's back, then finish the file
    display._progress.remove_task(display._slots[0])
    ev("item_done", index=1, item=0, input="/vids/one.mkv", status="ok")

    assert display._progress._tasks[display._step].completed == 1


def test_a_failed_file_reaches_the_manifest_while_the_step_is_still_running(tmp_path):
    """
    The manifest used to be written once, at the end of each step. A step where
    one file has already failed is exactly the step most likely to stall on the
    rest -- and a step that never ends never writes one, so the reason for the
    failure reaches disk nowhere at all. That is the position a real run left a
    user in: one file gone after two seconds, three still listed as working, and
    nothing anywhere saying why.

    So the assertion is specifically that the error is readable from disk *while
    another file is still in flight*, not merely that it lands eventually.
    """
    import json
    import threading
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    released = threading.Event()
    seen_on_disk = []
    manifest_path = tmp_path / "m.json"

    def fake_item_step(i, p, step, potato, item_artifacts, globals_ctx,
                       vars_ctx, on_start=None, on_progress=None, quiet=False):
        if i == 0:
            return (i, "error", {}, {"error": "cuDNN could not be loaded"})
        released.wait(timeout=2)        # still running while item 0 is recorded
        return (i, "ok", {}, {})

    def on_event(name, **payload):
        if name == "item_done" and payload["item"] == 0:
            # read it back the way a worried user would: from the file itself
            seen_on_disk.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            released.set()

    preset = {"steps": [{"scope": "item", "call": "taters.fake:step",
                         "with": {"path": "{{input}}"}}]}
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"")

    with mock.patch.object(run_pipeline, "_run_item_step", fake_item_step):
        run_pipeline.run_preset(preset, root_dir=tmp_path, file_type="any", workers=2,
                                out_manifest=manifest_path, on_event=on_event,
                                verbose=False)

    assert seen_on_disk, "item 0 never reported"
    errors = [e for item in seen_on_disk[0]["items"] for e in item["errors"]]
    assert any("cuDNN could not be loaded" in e for e in errors)


# ---------------------------------------------------------------------------
# how many files a step actually runs at once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("step,workers,expected", [
    # a step that touches no model gets the whole of --workers. this is the one
    # that makes the feature acceptable: capping ffmpeg and the merges would
    # make the single concurrency knob a lie
    ({"with": {"input_path": "{{input}}"}}, 8, 8),
    # a ceiling the wizard wrote, for a step that shares one model
    ({"with": {}, "max_workers": 2}, 8, 2),
    # and for a step that loads a model per worker
    ({"with": {}, "max_workers": 1}, 8, 1),
    # a ceiling is a maximum, never a minimum: --workers 2 stays 2
    ({"with": {}, "max_workers": 4}, 2, 2),
    # no ceiling declared, but it hands `device` to something. this is the safety
    # net for presets written before `max_workers` existed
    ({"with": {"device": "{{var:device}}"}}, 8, 1),
    # `workers:` is an instruction, not a preference: it beats both the guess...
    ({"with": {"device": "{{var:device}}"}, "workers": 6}, 2, 6),
    # ...and an explicit ceiling. this is the escape hatch for a big card
    ({"with": {}, "max_workers": 1, "workers": 3}, 8, 3),
])
def test_how_many_files_a_step_runs_at_once(step, workers, expected):
    from taters.pipelines.run_pipeline import _workers_for

    assert _workers_for(step, workers) == expected


def test_a_capped_step_says_why_it_is_not_using_every_worker():
    """
    Someone who passed `--workers 8` and sees two files moving should be told
    why, rather than concluding the setting was ignored.
    """
    from taters.pipelines.run_pipeline import _workers_for

    said = []
    _workers_for({"with": {}, "max_workers": 2}, 8, said.append)

    assert said and "2 at a time" in said[0]
    assert "GPU" in said[0]
    assert "--workers 8 still applies elsewhere" in said[0]


def test_nothing_is_said_when_the_cap_changes_nothing():
    from taters.pipelines.run_pipeline import _workers_for

    said = []
    _workers_for({"with": {}, "max_workers": 4}, 2, said.append)
    _workers_for({"with": {"input_path": "{{input}}"}}, 8, said.append)

    assert said == []


def test_the_cap_is_not_conditional_on_a_gpu_actually_being_there():
    """
    Deliberate. Resolving the device here would mean asking a question the
    runner cannot answer honestly -- CTranslate2 reports a usable CUDA device
    and then fails at the first encode -- and the capped steps are no faster
    with extra workers on a CPU either. Predictable beats clever.
    """
    from taters.pipelines.run_pipeline import _workers_for

    assert _workers_for({"with": {"device": "cpu"}, "max_workers": 1}, 8) == 1


def test_a_cap_is_obeyed_by_a_real_run(tmp_path):
    """
    The clamp has to reach the pool, not just the arithmetic. Counts how many
    workers are ever in flight at once over eight files with a ceiling of two.
    """
    import threading
    import unittest.mock as mock

    from taters.pipelines import run_pipeline

    lock = threading.Lock()
    live, peak = [0], [0]

    def fake_item_step(i, p, step, potato, item_artifacts, globals_ctx,
                       vars_ctx, on_start=None, on_progress=None, quiet=False):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return (i, "ok", {}, {})

    for n in range(8):
        (tmp_path / f"f{n}.wav").write_bytes(b"")
    preset = {"steps": [{"scope": "item", "call": "taters.fake:step",
                         "max_workers": 2, "with": {"path": "{{input}}"}}]}

    with mock.patch.object(run_pipeline, "_run_item_step", fake_item_step):
        manifest = run_pipeline.run_preset(
            preset, root_dir=tmp_path, file_type="any", workers=8,
            out_manifest=tmp_path / "m.json", verbose=False, vars_ctx={},
        )

    assert peak[0] <= 2, f"{peak[0]} files were in flight despite max_workers: 2"
    assert [i["status"] for i in manifest["items"]] == ["ok"] * 8


def test_a_line_separates_what_has_happened_from_what_is_happening():
    """
    Without one, everything the run prints -- a step's own output, a warning, a
    subprocess traceback relayed by the parent -- lands flush against the top
    bar and the two read as one block. That is worst exactly when it matters: a
    reported user paste had a stack trace whose last line abutted a spinner.
    """
    import re

    buf = io.StringIO()
    console = Console(file=buf, width=70, force_terminal=True, color_system=None)
    with RunDisplay(console) as d:
        ev = reporter_for(d)
        ev("run_start", steps=2, inputs=[])
        ev("step_start", index=1, total=2, call="potato.audio.x", scope="item", items=2)
        ev("item_done", index=1, item=0, input="/v/one.mkv", status="error",
           error="Embedding subprocess failed with code 1")
        d._progress.refresh()

    plain = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", buf.getvalue())
    lines = [ln for ln in plain.split("\n") if ln.strip()]

    rules = [i for i, ln in enumerate(lines) if set(ln.strip()) == {"─"}]
    assert rules, "nothing separates the printed output from the bars"

    # the failure sits above the line, the bars below it
    joined_above = "\n".join(lines[:rules[0]])
    joined_below = "\n".join(lines[rules[0]:])
    assert "failed with code 1" in joined_above
    assert "Overall" in joined_below


def test_error_text_with_brackets_survives_the_display(tmp_path):
    """
    Round-2 issue 30: failure and warning text is interpolated into rich
    markup. Unescaped, a message containing '[/bold]' raised MarkupError
    inside __exit__ -- crashing while unwinding after a possibly hours-long
    run -- and milder bracketed tokens were silently deleted from the very
    line the user needed to read.
    """
    import io

    from rich.console import Console

    from taters.ui.run_display import RunDisplay, reporter_for

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100)
    with RunDisplay(console) as display:
        report = reporter_for(display)
        report("item_done", item="clip.wav", status="error",
               error="dict [KeyError: 'text[0]'] and a stray [/bold] tag")
        display.notices.append("warning with [brackets] in it")
    # __exit__ survived, so now the text had better be there verbatim. we strip
    # the ANSI styling first, since rich's highlighter wraps brackets in their
    # own spans, which is fine. what can't happen is the bracketed text *vanishing*
    import re

    drawn = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", buf.getvalue()).replace("\n", "")
    assert "[/bold] tag" in drawn
    assert "[brackets]" in drawn
    assert "[KeyError: 'text[0]']" in drawn


def test_a_long_document_name_cannot_push_the_bar_off_the_row(display):
    """
    Reported from a real run: the gather phase now names each document, and a
    paper titled like a conference proceedings widened the description past
    the terminal -- hiding the bar and the clock, the two things that prove
    the run is moving. The step row is fitted like the per-file rows.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1,
       call="potato.text.analyze_with_dictionaries", scope="global")
    ev("step_progress",
       done=1, total=214,
       message="reading Advances on P2P, Parallel, Grid, Cloud and Internet "
               "Computing Proceedings of the 13th International Conference.pdf")

    task = display._progress.tasks[-1]
    # label (~30) + the fitted message: far under the 130 characters the raw
    # title would've taken, and short phase names stay whole (see the pulses
    # test above, still asserting "reading the input" verbatim)
    assert len(task.description) <= 75, task.description
    assert "…" in task.description
    # _fit clips from the middle and keeps both ends: the start says what's
    # happening, and the extension tells one long-titled paper from another
    assert task.description.startswith("analyze with dictionaries · reading ")
    assert task.description.endswith(".pdf")


def test_count_rows_counts_records_not_physical_lines(tmp_path):
    """
    From a real run: document text carries embedded newlines inside its quoted
    CSV field, and line-counting told a 2,300-paper corpus it was 3.2 million
    rows -- a denominator so wrong the bar read as broken.
    """
    from taters.helpers.progress import count_rows

    f = tmp_path / "gathered.csv"
    f.write_text('text_id,text\n'
                 'a,"one\ntwo\nthree lines of text"\n'
                 'b,"another\nmultiline document"\n',
                 encoding="utf-8")
    assert count_rows(f, on_progress=lambda *a, **k: None) == 2


def test_a_named_tick_reaches_the_step_row(display):
    """One slow document must be visibly *someone's fault*: the analyzers name
    the row they are scoring, and the display shows it."""
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1,
       call="potato.text.analyze_with_dictionaries", scope="global")
    ev("step_progress", done=20, total=2300,
       message="scoring Advances on P2P Proceedings")

    task = display._progress.tasks[-1]
    # fitted for the narrow test console (mid-clipped), but still visibly a name
    assert "· scoring " in task.description
    assert task.description.endswith("oceedings")


def test_inflight_documents_get_sub_bars_like_item_steps(display):
    """
    Requested: a parallel phase inside a global step shows one row per
    document being worked on, the same way ffmpeg and Whisper item steps do
    -- and the rows retire the moment their document finishes, so fifteen
    "working" bars never sit behind one straggler.
    """
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1,
       call="potato.text.analyze_with_dictionaries", scope="global")

    ev("step_progress", done=0, total=40, message="gathering documents",
       inflight=["alpha.pdf", "beta.docx"])
    labels = [t.description for t in display._progress.tasks]
    assert any("alpha.pdf" in d for d in labels)
    assert any("beta.docx" in d for d in labels)

    ev("step_progress", done=1, total=40, message="gathering documents",
       inflight=["beta.docx", "gamma.txt"])
    labels = [t.description for t in display._progress.tasks]
    assert not any("alpha.pdf" in d for d in labels), "finished bars retire"
    assert any("gamma.txt" in d for d in labels)


def test_a_flood_of_inflight_documents_folds_into_and_n_more(display):
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1,
       call="potato.text.analyze_with_dictionaries", scope="global")
    ev("step_progress", done=0, total=99, message="gathering documents",
       inflight=[f"doc{i:02d}.pdf" for i in range(14)])

    labels = [t.description for t in display._progress.tasks]
    named = [d for d in labels if ".pdf" in d]
    assert len(named) == 8
    assert any("and 6 more" in d for d in labels)


def test_the_step_clock_survives_a_phase_ending_at_full(display):
    """A step runs phases over one bar (count the rows, then score them) and
    each phase ends at N/N. rich stamps finished_time the first time completed
    reaches total and the elapsed clock freezes there for good -- reported
    from a real run whose step timer stood still while the sub-bars ticked.
    Still receiving step_progress means the step is not finished: unfreeze."""
    ev = reporter_for(display)
    ev("run_start", steps=1, inputs=[])
    ev("step_start", index=1, total=1, call="potato.text.analyze_with_dictionaries",
       scope="global")
    ev("step_progress", index=1, call="x", done=15_000, total=15_000,
       message="counting rows")                      # phase one ends at N/N
    ev("step_progress", index=1, call="x", done=10, total=15_000,
       message="scoring documents")                  # phase two begins

    task = next(t for t in display._progress.tasks if t.id == display._step)
    assert task.finished_time is None, "the elapsed clock froze at phase end"
    assert task.completed == 10


def test_the_display_silences_huggingface_progress_bars_while_live():
    """Model downloads (stanza, sentence-transformers) draw raw tqdm bars on
    stderr via huggingface-hub, which sliced through the live bars in a real
    run. While the display owns the terminal, the hub's own off switch is
    set -- and restored afterwards, whatever it was."""
    import io
    import os

    from rich.console import Console

    from taters.ui.run_display import RunDisplay

    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    with RunDisplay(Console(file=io.StringIO(), width=80)):
        assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    try:
        with RunDisplay(Console(file=io.StringIO(), width=80)):
            assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
        assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "0", (
            "a user's own setting must come back exactly as it was"
        )
    finally:
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)


def test_a_cut_message_never_dangles_a_parenthesis():
    """
    "sorting rows into groups (pass 1 of 2)" fitted into a narrow row read
    "sorting row…pass 1 of 2)" -- the middle cut ate the opening paren and
    the orphan looked like a rendering bug (a real report). Orphaned parens
    are dropped whenever a message had to be cut.
    """
    console = Console(file=io.StringIO(), width=60)
    with RunDisplay(console) as display:
        ev = reporter_for(display)
        ev("run_start", steps=1, inputs=[])
        ev("step_start", index=1, total=1,
           call="wrangle: gather spreadsheet", scope="global")
        ev("step_progress", index=1, call="x", done=63191, total=634043,
           message="sorting rows into groups (pass 1 of 2)")
        d = display._progress.tasks[-1].description
    assert "…" in d, "the message was not cut; the test lost its subject"
    assert d.count("(") == d.count(")"), f"unbalanced parenthesis in {d!r}"


def test_an_uncut_message_keeps_its_parentheses():
    console = Console(file=io.StringIO(), width=300)
    with RunDisplay(console) as display:
        ev = reporter_for(display)
        ev("run_start", steps=1, inputs=[])
        ev("step_start", index=1, total=1,
           call="wrangle: gather spreadsheet", scope="global")
        ev("step_progress", index=1, call="x", done=1, total=10,
           message="sorting rows into groups (pass 1 of 2)")
        d = display._progress.tasks[-1].description
    assert d.endswith("sorting rows into groups (pass 1 of 2)")
