"""Tests for taters.helpers.proc.run_and_stream.

This helper decides what you see while a long child process (diarization,
Whisper) is running, and what the error message says when one dies. Both are
worth pinning down; a regression here is invisible until you are three hours
into a batch wondering whether anything is happening.

`capsys` is a built-in pytest fixture that captures whatever the test printed.
"""

import subprocess
import sys
import textwrap

import pytest

from taters.helpers.proc import run_and_stream


def python_child(body: str) -> list[str]:
    """Build a command that runs a snippet in a fresh interpreter."""
    return [sys.executable, "-c", textwrap.dedent(body)]


# --- exit codes -------------------------------------------------------------

def test_successful_child_returns_zero():
    code, _ = run_and_stream(python_child("print('fine')"))
    assert code == 0


def test_failing_child_returns_its_exit_code():
    code, _ = run_and_stream(python_child("import sys; sys.exit(3)"))
    assert code == 3


def test_a_failing_child_does_not_raise():
    """Callers decide what a non-zero exit means; the helper just reports it."""
    code, tail = run_and_stream(python_child("import sys; sys.exit(1)"))
    assert code == 1 and isinstance(tail, str)


# --- streaming --------------------------------------------------------------

def test_output_is_printed_as_it_arrives(capsys):
    run_and_stream(python_child("print('line one'); print('line two')"))
    assert capsys.readouterr().out.splitlines() == ["line one", "line two"]


def test_prefix_labels_every_line(capsys):
    run_and_stream(python_child("print('a'); print('b')"), prefix="[diarize:clip] ")
    out = capsys.readouterr().out.splitlines()
    assert out == ["[diarize:clip] a", "[diarize:clip] b"]


def test_stderr_is_interleaved_with_stdout(capsys):
    run_and_stream(python_child("""
        import sys
        print('to stdout', flush=True)
        print('to stderr', file=sys.stderr, flush=True)
    """))
    out = capsys.readouterr().out
    assert "to stdout" in out and "to stderr" in out


def test_stream_false_prints_nothing_but_still_captures(capsys):
    code, tail = run_and_stream(python_child("print('quiet please')"), stream=False)
    assert capsys.readouterr().out == ""
    assert "quiet please" in tail
    assert code == 0


# --- tail capture -----------------------------------------------------------

def test_tail_contains_the_child_output():
    _, tail = run_and_stream(python_child("print('first'); print('second')"), stream=False)
    assert tail.splitlines() == ["first", "second"]


def test_tail_is_capped_at_the_requested_number_of_lines():
    """A child that dies after 10k lines of progress bars must not build a
    10k-line exception message."""
    _, tail = run_and_stream(
        python_child("[print(i) for i in range(500)]"), stream=False, tail_lines=10
    )
    lines = tail.splitlines()
    assert len(lines) == 10
    assert lines == [str(i) for i in range(490, 500)]      # the *last* ten


def test_empty_output_gives_an_empty_tail():
    _, tail = run_and_stream(python_child("pass"), stream=False)
    assert tail == ""


def test_undecodable_bytes_do_not_crash_the_reader():
    """Model tooling emits progress bars and stray bytes; never blow up on them."""
    code, tail = run_and_stream(
        python_child(r"""
            import sys
            sys.stdout.buffer.write(b'ok \xff\xfe bad bytes\n')
            sys.stdout.buffer.flush()
        """),
        stream=False,
    )
    assert code == 0 and "bad bytes" in tail


# --- environment and working directory --------------------------------------

def test_cwd_is_passed_to_the_child(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _, tail = run_and_stream(
        python_child("import os; print(os.getcwd())"), cwd=work, stream=False
    )
    assert tail.strip().endswith("work")


def test_env_is_passed_to_the_child():
    _, tail = run_and_stream(
        python_child("import os; print(os.environ.get('TATERS_TEST_VAR'))"),
        env={"TATERS_TEST_VAR": "hello", "PATH": "/usr/bin"},
        stream=False,
    )
    assert tail.strip() == "hello"


def test_child_stdin_is_closed_so_it_cannot_hang(capsys):
    """
    ffmpeg and friends will happily block forever waiting on a TTY. The helper
    must hand the child a closed stdin so a batch run cannot deadlock.
    """
    code, tail = run_and_stream(
        python_child("import sys; print(repr(sys.stdin.read()))"), stream=False
    )
    assert code == 0 and tail.strip() == "''"


# --- timeout ----------------------------------------------------------------

def test_timeout_kills_the_child_and_raises():
    with pytest.raises(subprocess.TimeoutExpired):
        run_and_stream(
            python_child("import time; print('start', flush=True); time.sleep(30)"),
            timeout=1,
            stream=False,
        )
