"""Run a child process while showing its output as it happens.

``subprocess.run(..., capture_output=True)`` gives great error messages and
terrible ergonomics: a twenty-minute transcription prints nothing until it is
over. :func:`run_and_stream` does both — every line is echoed as it arrives
(prefixed, so concurrent workers stay legible) and the tail is retained for the
exception message if the child fails.
"""

from __future__ import annotations

import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple


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
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if stream:
                print(f"{prefix}{line}", flush=True)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    return proc.returncode, "\n".join(tail)
