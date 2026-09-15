"""Run a child process while showing its output as it happens.

``subprocess.run(..., capture_output=True)`` gives great error messages and
terrible ergonomics: a twenty-minute transcription prints nothing until it is
over. :func:`run_and_stream` does both — every line is echoed as it arrives
(prefixed, so concurrent workers stay legible) and the tail is retained for the
exception message if the child fails.
"""

from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


def _utf8_stdio(env: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """
    Make the child write UTF-8, whatever the console's code page is.

    A child that prints a path is a child that can be killed by the path. On
    Windows, `print()` encodes to the active code page -- cp1252 for most
    Western installs -- and a filename containing a character it cannot map
    raises `UnicodeEncodeError` and takes the process down with a non-zero exit.

    Media filenames are full of exactly those characters: anything downloaded
    from the web arrives with fullwidth stand-ins (`：`, `？`, `／`) for the
    punctuation a filesystem refuses. One real run lost a whole embeddings step
    that way -- and lost it on the line announcing success, after the CSV had
    already been written, so the work was done and thrown away.

    `PYTHONIOENCODING` covers the child's own streams; `PYTHONUTF8` covers
    anything it opens without naming an encoding. Neither overrides a value the
    caller set deliberately.
    """
    merged: Dict[str, str] = dict(env) if env is not None else dict(os.environ)
    merged.setdefault("PYTHONIOENCODING", "utf-8")
    merged.setdefault("PYTHONUTF8", "1")
    return merged


def run_and_stream(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[int] = None,
    prefix: str = "",
    stream: bool = True,
    tail_lines: int = 200,
) -> Tuple[int, str]:
    """
    Run `cmd`, echoing its combined stdout/stderr line by line.

    Parameters
    ----------
    cmd : Sequence[str]
        Command and arguments.
    cwd : Path, optional
        Working directory for the child.
    env : Mapping[str, str], optional
        Environment for the child.
    timeout : int, optional
        Seconds before the child is killed. ``None`` waits indefinitely.
    prefix : str, default=""
        Prepended to each echoed line, e.g. ``"[diarize:session] "``. Useful
        when several items are processed concurrently.
    stream : bool, default=True
        Echo output as it arrives. When ``False`` the output is still captured
        for the return value, just not printed.
    tail_lines : int, default=200
        How many trailing output lines to retain for the caller (typically to
        build an error message).

    Returns
    -------
    tuple[int, str]
        The child's exit code and the retained tail of its output.

    Raises
    ------
    subprocess.TimeoutExpired
        If `timeout` elapses; the child is killed before the exception escapes.
    """
    tail: deque[str] = deque(maxlen=max(1, tail_lines))

    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=_utf8_stdio(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )

    def _pump() -> None:
        """Drain the child's output on a background thread."""
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            tail.append(line)
            if stream:
                print(f"{prefix}{line}", flush=True)

    # the reader has to run off the main thread: iterating the pipe blocks until
    # the child closes it, so we can't wait for output and enforce `timeout` on
    # the same thread. (select() would be POSIX-only; this works on Windows too.)
    reader = threading.Thread(target=_pump, name="taters-proc-reader", daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout)
    except BaseException:
        # this covers TimeoutExpired and KeyboardInterrupt alike: we never want
        # to leave an orphaned ffmpeg/whisper process behind.
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        raise
    finally:
        reader.join(timeout=5)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass

    return proc.returncode, "\n".join(tail)
