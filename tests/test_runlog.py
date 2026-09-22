"""
The run log: everything a run printed, in one file.

Why this file exists
--------------------
A real run failed and the manifest recorded one string -- a wrapper's
``Could not import module 'RobertaModel'`` -- while the exception it had been
raised ``from``, the one naming the module that was actually missing, was
discarded at the failure site. Nothing recorded the Python version, the build
of torch, or a line of what the step printed first. Working it out took an hour
of picking through the filesystem, and by then the environment was gone.

So what is pinned here is the difference between those two outcomes: that the
short reason still reaches the manifest **unchanged**, that the whole cause
chain reaches the log, and that the log holds what the run printed -- including
the output of child processes, which is where a library's real complaint
usually lands.

And, just as firmly, that none of it can break a run. A log is a diagnostic;
the moment it is load-bearing it has become the bug.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from taters.helpers import runlog
from taters.helpers.runlog import RunLog, clean_stream_text, exception_text


# ---------------------------------------------------------------------------
# Turning terminal output into something readable
# ---------------------------------------------------------------------------

def test_escape_sequences_are_stripped_and_redraws_collapse():
    """
    A log is read in a text editor, where an escape sequence is mojibake and a
    progress bar is thousands of near-identical lines. A carriage return is a
    redraw, so only the frame that was on screen at the end survives.
    """
    assert clean_stream_text("\x1b[1;32mstyled\x1b[0m done") == ["styled done"]
    assert clean_stream_text("\x1b]0;a title\x07real text") == ["real text"]
    assert clean_stream_text("\x1b[2K\x1b[1Gstep 3\x1b[?25h") == ["step 3"]

    bar = "  0%|    | 0/9\r 55%|##  | 5/9\r100%|####| 9/9"
    assert clean_stream_text(bar) == ["100%|####| 9/9"]


def test_text_survives_the_cleaning():
    """Stripping escapes must not damage the words, which are the point."""
    assert clean_stream_text("naive -- 938 texts embedded") == \
        ["naive -- 938 texts embedded"]
    assert clean_stream_text("café — done ✓") == ["café — done ✓"]


def test_the_header_names_the_build_of_torch_without_importing_it():
    """
    The single most useful line in the header, and the one the incident
    turned on: ``2.11.0+cu128`` and ``2.14.0+cpu`` are different afternoons.

    Reading it must not *import* torch, which would add seconds to every run
    and, on a broken install, would fail in the middle of writing the log.
    """
    lines = runlog.environment_lines(work_dir="/tmp/x", preset="p", workers=4)
    blob = "\n".join(lines)
    assert blob.startswith("# taters run log -- started ")
    assert "# torch " in blob, "the header does not mention torch at all"
    assert "# preset" in blob and "# python" in blob and "# platform" in blob

    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys;sys.path.insert(0,'src');"
         "from taters.helpers import runlog;runlog.environment_lines();"
         "print('torch' in sys.modules)"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert probe.stdout.strip() == "False", \
        "writing the header imported torch, which it must never do"


# ---------------------------------------------------------------------------
# The cause chain -- the payload the incident needed
# ---------------------------------------------------------------------------

def test_a_wrapped_exception_keeps_the_reason_it_was_raised_from():
    """
    The incident in one test. A library catches an import error and re-raises
    its own vague one ``from`` it; the vague message is all the manifest ever
    held. ``exception_text`` keeps both, and the inner one names the module.
    """
    try:
        try:
            raise ModuleNotFoundError("No module named 'torch.fx'")
        except ModuleNotFoundError as inner:
            raise ModuleNotFoundError(
                "Could not import module 'RobertaModel'.") from inner
    except ModuleNotFoundError as outer:
        text = exception_text(outer)

    assert "torch.fx" in text, "the real reason was lost"
    assert "Could not import module 'RobertaModel'" in text
    assert "direct cause" in text, "the chain was flattened to one exception"


def test_a_failing_step_keeps_its_short_reason_and_gains_a_traceback():
    """
    Two promises at once. The string the manifest has always carried is
    **byte-identical** -- six tests elsewhere match on its exact shape, and a
    saved pipeline's failure message is something people quote. The traceback
    is additive, and nothing but the log reads it.
    """
    from taters.pipelines.run_pipeline import run_global_step

    def boom(**_kw):
        try:
            raise ModuleNotFoundError("No module named 'torch.fx'")
        except ModuleNotFoundError as inner:
            raise ModuleNotFoundError(
                "Could not import module 'RobertaModel'. "
                "Are this object's requirements defined correctly?") from inner

    potato = types.SimpleNamespace(
        text=types.SimpleNamespace(extract_transformer_embeddings=boom))

    status, _globals, err = run_global_step(
        step={"call": "potato.text.extract_transformer_embeddings", "with": {}},
        potato=potato, globals_ctx={}, vars_ctx={},
        manifest_path=Path("run_manifest.json"))

    assert status == "error"
    assert err["error"] == (
        "potato.text.extract_transformer_embeddings failed: ModuleNotFoundError: "
        "Could not import module 'RobertaModel'. Are this object's requirements "
        "defined correctly?")
    assert "torch.fx" in err["trace"]
    assert "direct cause" in err["trace"]


# ---------------------------------------------------------------------------
# Writing, and holding lines until there is somewhere to put them
# ---------------------------------------------------------------------------

def test_what_was_said_before_the_run_started_is_in_the_run_s_log(tmp_path):
    """
    A session answers questions for several screens before anything runs, and
    a wrong answer three screens back is as much a cause of failure as a bad
    import. There is no folder to write to yet, so those lines are held and
    flushed into the file once there is one -- in order, ahead of the run.
    """
    log = RunLog()
    log.line("ui", "ask   Which features? -> readability")
    log.line("ui", "ask   Work folder? -> here")

    path = log.open_run(tmp_path / "work", preset="p")
    log.line("run", "start  1 step(s)")
    log.close_run()

    body = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]
    assert "ask   Which features? -> readability" in body[0]
    assert "ask   Work folder? -> here" in body[1]
    assert "start  1 step(s)" in body[2]


def test_the_log_lands_in_a_timestamped_file_so_a_rerun_keeps_the_evidence(tmp_path):
    """
    The failed run and the run that was meant to prove the fix are the two
    things you want to compare, so the second must not overwrite the first.
    """
    first = RunLog().open_run(tmp_path / "work", preset="p")
    second = RunLog().open_run(tmp_path / "work", preset="p")
    assert first.parent.name == "logs"
    assert first.name.startswith("run-") and first.suffix == ".log"
    # same folder, and the earlier file is still there
    assert first.parent == second.parent
    assert first.is_file()


def test_a_traceback_is_indented_under_a_gutter_not_given_forty_timestamps(tmp_path):
    """Forty identical clocks down the side of a traceback help nobody, and a
    block that is visibly subordinate is what makes the skeleton skimmable."""
    log = RunLog()
    path = log.open_run(tmp_path / "work")
    log.block("trace", "Traceback (most recent call last):\n  File x\nBoom: no")
    log.close_run()

    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if "Traceback" in ln or "File x" in ln or "Boom" in ln]
    assert "trace" in lines[0] and "Traceback" in lines[0]
    assert lines[1].lstrip().startswith("|"), "continuation lines lost the gutter"
    assert lines[2].lstrip().startswith("|")


def test_an_enormous_line_is_truncated_rather_than_written_whole(tmp_path):
    """One line of a base64 blob can be megabytes wide and nobody reads past
    the first screenful."""
    log = RunLog()
    path = log.open_run(tmp_path / "work")
    log.line("out", "x" * 50_000)
    log.close_run()
    longest = max(len(ln) for ln in path.read_text(encoding="utf-8").splitlines())
    assert longest < 3000
    assert "chars)" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Capturing what gets printed
# ---------------------------------------------------------------------------

def test_the_log_holds_what_a_step_printed_including_a_child_process(tmp_path):
    """
    The reason there are two capture layers rather than one. Replacing
    ``sys.stdout`` catches Python's own printing; redirecting the descriptor
    underneath catches what never passes through it -- an extension module, a
    raw write, or a child process, which is where a library's real complaint
    usually turns up.
    """
    log = RunLog()
    path = log.open_run(tmp_path / "work")
    with log.capture_streams():
        print("a step said this")
        os.write(1, b"written straight to the descriptor\n")
        subprocess.run([sys.executable, "-c", "print('a child said this')"])
        sys.stderr.write("and this went to stderr\n")
    log.close_run()

    text = path.read_text(encoding="utf-8")
    assert "a step said this" in text
    assert "written straight to the descriptor" in text
    assert "a child said this" in text, "child-process output was not captured"
    assert "and this went to stderr" in text
    assert "  err  " in text, "stderr was not labeled as its own channel"


def test_a_live_display_draws_on_the_screen_and_not_into_the_log(tmp_path):
    """
    The trap that would have made this feature worse than useless. A bare rich
    ``Console`` holds no file of its own and resolves ``sys.stdout`` *at write
    time* -- which, during a run, is the tee. Left alone it would put every
    spinner frame in the log and, worse, the progress display would be drawing
    somewhere the user cannot see. Pinning is an assignment to ``console.file``
    and nothing else will do it.
    """
    from rich.console import Console

    console = Console()
    assert console._file is None, "the premise is gone; rich now holds a file"

    log = RunLog(console=console)
    path = log.open_run(tmp_path / "work")
    with log.capture_streams():
        console.print("SPINNER FRAME")
        print("real step output")
    log.close_run()

    text = path.read_text(encoding="utf-8")
    assert "real step output" in text
    assert "SPINNER FRAME" not in text, \
        "the live display is being written into the log instead of the screen"
    assert console._file is None, "the console was left pinned after the run"


def test_no_escape_sequences_reach_the_file(tmp_path):
    log = RunLog()
    path = log.open_run(tmp_path / "work")
    with log.capture_streams():
        print("\x1b[31mred\x1b[0m")
    log.close_run()
    assert "\x1b" not in path.read_text(encoding="utf-8")


def test_the_streams_are_put_back_even_when_the_work_raises(tmp_path):
    """
    Whatever else happens, the terminal has to be usable afterwards. Getting
    the teardown order wrong yields ``Bad file descriptor`` from inside
    unrelated code much later, which is a miserable thing to debug.
    """
    log = RunLog()
    log.open_run(tmp_path / "work")

    before_out, before_err = sys.stdout, sys.stderr
    before_isatty = sys.stdout.isatty()

    with pytest.raises(RuntimeError):
        with log.capture_streams():
            print("some output first")
            raise RuntimeError("the step blew up")

    log.close_run()
    assert sys.stdout is before_out, "sys.stdout was not restored"
    assert sys.stderr is before_err, "sys.stderr was not restored"
    assert sys.stdout.isatty() == before_isatty
    print("and printing still works")          # must not raise
    os.write(1, b"")                           # the descriptor is still live


def test_capturing_does_nothing_when_there_is_no_file_to_write_to():
    """No handle, no redirection. This is what keeps the descriptors alone
    during the menus, where a pipe would change how the prompts render."""
    log = RunLog()
    before = sys.stdout
    with log.capture_streams():
        assert sys.stdout is before, "the streams were hijacked with nowhere to write"


# ---------------------------------------------------------------------------
# It must never be the thing that breaks a run
# ---------------------------------------------------------------------------

def test_a_log_that_cannot_be_opened_is_not_an_error(tmp_path):
    """
    The most likely moment for the log to be writing is while something else
    has already gone wrong. It gets one shot and then stays out of the way.
    """
    blocked = tmp_path / "wall"
    blocked.write_text("I am a file, not a folder", encoding="utf-8")

    log = RunLog()
    assert log.open_run(blocked / "work") is None
    log.line("run", "the run carries on")      # must not raise
    log.block("trace", "nor here")
    with log.capture_streams():
        pass
    log.close_run()


def test_a_broken_sink_gives_up_instead_of_raising(tmp_path):
    """A failed write disables the log for good rather than failing twice per
    line for the rest of a long run."""
    log = RunLog()
    log.open_run(tmp_path / "work")
    log._handle.close()                        # the disk went away, in effect
    log.line("run", "this cannot be written")
    assert not log.active, "the sink kept trying after it broke"
    log.line("run", "and this is quietly dropped")
    log.close_run()


def test_the_kill_switch_turns_everything_off(tmp_path, monkeypatch):
    """One environment variable, for anyone whose terminal this upsets."""
    monkeypatch.setenv("TATERS_RUNLOG", "0")
    log = RunLog()
    assert not log.active
    assert log.open_run(tmp_path / "work") is None
    assert not list((tmp_path / "work").glob("**/*.log"))


# ---------------------------------------------------------------------------
# Through the runner
# ---------------------------------------------------------------------------

def _preset_that_fails():
    def fine(**_kw):
        print("[vader] 938 rows scored")
        return "features/sentiment_vader.csv"

    def boom(**_kw):
        print("about to load distilroberta-base")
        try:
            raise ModuleNotFoundError("No module named 'torch.fx'")
        except ModuleNotFoundError as inner:
            raise ModuleNotFoundError(
                "Could not import module 'RobertaModel'.") from inner

    return ({"steps": [
        {"call": "potato.text.analyze_sentiment_vader", "scope": "global",
         "save_as": "vader"},
        {"call": "potato.text.extract_transformer_embeddings", "scope": "global"},
    ]}, types.SimpleNamespace(text=types.SimpleNamespace(
        analyze_sentiment_vader=fine, extract_transformer_embeddings=boom)))


def test_a_run_leaves_one_file_that_answers_why_it_failed(tmp_path, monkeypatch):
    """
    The whole feature, end to end, on the shape of the incident: the header
    says what it ran against, the output of the step that died is there, and
    the reason names the module that was actually missing.
    """
    import taters.pipelines.run_pipeline as rp

    preset, potato = _preset_that_fails()
    monkeypatch.setattr(rp, "Taters", lambda: potato)

    work = tmp_path / "mypipe"
    manifest = rp.run_preset(
        preset, work_dir=work, preset_name="predicting_personality", workers=2,
        out_manifest=work / "run_manifest.json", verbose=False,
        run_log=RunLog())

    assert manifest["log"], "the manifest does not point at the log"
    text = Path(manifest["log"]).read_text(encoding="utf-8")

    assert "# torch " in text                              # the environment
    assert "[vader] 938 rows scored" in text               # a step that worked
    assert "about to load distilroberta-base" in text      # the one that did not
    assert "torch.fx" in text                              # the real reason
    assert "direct cause" in text

    # and the manifest keeps its own short version, untouched
    assert manifest["errors"] == [
        "potato.text.extract_transformer_embeddings failed: ModuleNotFoundError: "
        "Could not import module 'RobertaModel'."]


def test_a_run_nobody_asked_to_log_writes_nothing(tmp_path, monkeypatch):
    """
    Every existing caller of ``run_preset`` -- and there are around forty in
    this suite -- passes no log and must keep behaving exactly as it did,
    including leaving no new files behind.
    """
    import taters.pipelines.run_pipeline as rp

    preset, potato = _preset_that_fails()
    monkeypatch.setattr(rp, "Taters", lambda: potato)

    work = tmp_path / "mypipe"
    manifest = rp.run_preset(preset, work_dir=work, workers=1,
                             out_manifest=work / "run_manifest.json",
                             verbose=False)
    assert manifest["log"] is None
    assert not list(work.glob("**/*.log"))
    assert not (work / "logs").exists()


# ---------------------------------------------------------------------------
# The session transcript
# ---------------------------------------------------------------------------

def test_a_note_is_recorded_once_however_often_the_screen_is_redrawn(tmp_path):
    """
    The bug this whole guard exists for, and it is silent.

    The live renderer redraws everything above the question on every screen,
    and it does that by replaying remembered notes through the very method
    that records them. Without the flag, a note said on screen one is written
    again on screens two, three and four -- so the file grows with the square
    of the session while the terminal looks perfectly normal.
    """
    pytest.importorskip("questionary")
    from taters.ui.live import LivePrompter

    prompter = LivePrompter()
    log = RunLog()
    path = log.open_run(tmp_path / "work")
    prompter._runlog = log

    prompter.note("  Found 412 texts under /data/blogs")
    for _ in range(4):
        prompter.repaint()
    log.close_run()

    text = path.read_text(encoding="utf-8")
    assert text.count("Found 412 texts") == 1, (
        f"the note was recorded {text.count('Found 412 texts')} times; "
        "the repaint guard is not holding")


def test_the_questions_and_the_answers_given_are_both_recorded(tmp_path):
    """
    A transcript of answers is the point of logging the menus at all: the run
    manifest records the settings that came out, but not that someone was
    asked "which folder?" and typed the wrong one.
    """
    from taters.ui.prompts import QuestionaryPrompter

    prompter = QuestionaryPrompter.__new__(QuestionaryPrompter)   # no terminal
    log = RunLog()
    path = log.open_run(tmp_path / "work")
    prompter._runlog = log

    prompter._log_answer("ask", "Which folder holds your texts?", "/data/blogs")
    prompter._log_answer("tick", "Which features?", ["readability", "vader"])
    log.close_run()

    text = path.read_text(encoding="utf-8")
    assert "ask  Which folder holds your texts? -> /data/blogs" in text
    assert "tick Which features? -> readability, vader" in text


def test_nothing_is_recorded_when_no_log_is_running():
    """
    Every test in this suite builds prompters without a log, and ~3,000 of
    them would break if the recording helpers assumed one was there.
    """
    from taters.ui.prompts import QuestionaryPrompter

    prompter = QuestionaryPrompter.__new__(QuestionaryPrompter)
    assert prompter._runlog is None
    assert prompter._replaying is False
    prompter._log_ui("goes nowhere")                          # must not raise
    assert prompter._log_answer("ask", "q", "a") == "a"        # still returns


def test_output_goes_through_a_live_display_rather_than_across_it(tmp_path):
    """
    A regression, and a visible one: the progress bars came apart.

    While a live display is up, rich swaps ``sys.stdout`` for a proxy of its
    own that prints above the region the bars occupy. Capturing output by
    replacing that proxy, and writing to the terminal directly, put every
    library's chatter through the middle of a bar -- half a message, then the
    rest of the rule, then a redrawn bar at a different width.

    So while something else owns the screen, captured output is handed back to
    it rather than written past it. The log gets the line either way; this is
    about what the person watching sees.
    """
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TextColumn

    console = Console(force_terminal=True, width=60)
    progress = Progress(TextColumn("{task.description}"), BarColumn(),
                        console=console)
    log = RunLog(console=console)
    path = log.open_run(tmp_path / "work")

    progress.start()
    try:
        proxy = sys.stdout
        assert type(proxy).__module__.startswith("rich."), \
            "rich no longer proxies stdout for a live display; premise is gone"
        with log.capture_streams():
            print("a library said something")
            assert sys.stdout._terminal is proxy, (
                "captured output is being written past the live display "
                "instead of through it -- this is what breaks the bars")
    finally:
        progress.stop()
        log.close_run()

    assert "a library said something" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Putting the terminal back afterwards
# ---------------------------------------------------------------------------

def test_the_next_question_measures_the_terminal_again_after_a_run():
    """
    A real run died at the finish screen on Windows, with every step
    succeeded and every file written, on
    ``NoConsoleScreenBufferError: No Windows console found``.

    The chain: the log redirects file descriptor 1 into a pipe so a child
    process's output reaches it, ``dup2`` closes whatever that descriptor
    pointed at, and on Windows that is the console handle ``prompt_toolkit``
    read once at start-up and cached. Printing survives, because the
    descriptor is put back. The next *prompt* does not, because it asks the
    closed handle how big the screen is.

    So the cached measurement is dropped after every run. Asserted through
    the session prompt_toolkit really uses rather than through a stub, since
    a stub would not have caught the original bug either.
    """
    from prompt_toolkit.application.current import get_app_session

    from taters.ui.prompts import QuestionaryPrompter

    session = get_app_session()
    before = session._output
    try:
        session._output = object()
        QuestionaryPrompter().forget_console()
        assert session._output is None, \
            "prompt_toolkit is still holding the terminal it measured"
    finally:
        session._output = before


def test_forgetting_the_console_never_takes_a_finished_run_down_with_it():
    """
    It runs after the results are on disk, so anything it raises would lose a
    run that had already succeeded. A terminal that cannot be reset is worth
    less than that.
    """
    import sys

    from taters.ui.prompts import QuestionaryPrompter

    saved = sys.modules.get("prompt_toolkit.application.current")
    sys.modules["prompt_toolkit.application.current"] = None
    try:
        QuestionaryPrompter().forget_console()      # must not raise
    finally:
        if saved is None:
            sys.modules.pop("prompt_toolkit.application.current", None)
        else:
            sys.modules["prompt_toolkit.application.current"] = saved


def test_a_run_from_the_wizard_always_resets_the_terminal_before_asking(tmp_path):
    """
    Wired at the one place a run hands control back to the interface, so it
    cannot be forgotten by a caller that runs a pipeline some other way.
    """
    import inspect

    from taters.ui import wizard

    source = inspect.getsource(wizard.execute_preset)
    assert "forget_console()" in source
    assert source.index("forget_console()") < source.index("finish_screen("), \
        "the terminal has to be re-measured before the next question, not after"
