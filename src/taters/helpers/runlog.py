"""
Everything a run printed, in one file, so a failure can be read rather than reconstructed.

Why this exists
---------------
A run failed with one line in the manifest::

    potato.text.extract_transformer_embeddings failed: ModuleNotFoundError:
    Could not import module 'RobertaModel'. Are this object's requirements
    defined correctly?

That message is a wrapper's, not the cause's. The library raised it ``from e``,
so the real reason was one attribute away -- and it was discarded, because every
failure site in the runner flattens an exception to ``type(e).__name__`` plus
``str(e)`` and lets the object go. Nothing recorded the Python version, the
build of torch that was installed, or a single line of what the step printed
before it died. Working out what happened took an hour of picking through the
filesystem, and the environment that produced it had been replaced by then.

So: one log file per run, holding what the run printed, the full traceback
chain when something raises, and the versions it all ran against.

This is not a replacement for the run manifest. The manifest stays the
structured record -- which step ran, which input failed, where the outputs
landed -- and is the thing to read first. The log is the verbatim one, for when
the manifest's summary is not enough.

What it is not
--------------
Not :func:`~taters.helpers.atomic.atomic_write`. Every other artifact in Taters
is written that way, and it would be wrong here: that helper unlinks its
scratch file on ``BaseException`` and re-raises, which is exactly the moment a
log has to survive. This writes plainly and flushes as it goes, the way the run
manifest does, so that an interrupted run still leaves behind whatever had been
recorded.

Not a transcript of the screen either. Live progress bars are deliberately
absent; what is kept is the structure of the run and everything that was
printed.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "RunLog",
    "session_fallback_path",
    "describe_event",
    "exception_text",
    "clean_stream_text",
    "environment_lines",
]

PathLike = Union[str, Path]

#: One fixed-width column per channel, so that ``grep -E '  (trace|err)  '``
#: picks out a whole category without also matching the message text.
CHANNEL_WIDTH = 5

#: Timestamp, two spaces, channel, two spaces. Continuation lines pad to the
#: same width and mark themselves with a gutter instead of repeating the clock.
_PREFIX_WIDTH = 12 + 2 + CHANNEL_WIDTH + 2
_GUTTER = " " * (_PREFIX_WIDTH - 2) + "| "

# we strip these rather than keep them: the log gets opened in a text editor,
# and Notepad renders an escape sequence as mojibake rather than as color.
_ANSI = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"           # CSI -- colors, cursor moves, erase
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC -- window titles
    r"|\x1b[()][0-9A-B]"                   # charset selection
    r"|\x1b[=>]"                           # keypad mode
)

# the packages whose versions actually explain a failure. torch leads because
# its build tag is the single most useful thing in here -- `2.14.0+cpu` versus
# `2.11.0+cu128` is the difference between two very different afternoons.
_INTERESTING: Tuple[str, ...] = (
    "torch", "torchaudio", "transformers", "tokenizers",
    "sentence-transformers", "numpy", "scipy", "scikit-learn",
    "gensim", "nemo-toolkit", "praat-parselmouth", "pillow",
)

# a session can sit in the menus for a long time before it starts a run, and
# every line it says is held in memory until there is a folder to write to. so
# the buffer forgets its oldest lines rather than growing without limit.
_MAX_BUFFERED_LINES = 5000

# a progress bar can redraw thousands of times without ever emitting a newline.
# we hold partial output only up to this much before giving up and writing it.
_MAX_PARTIAL_BYTES = 64 * 1024

# one line of a base64 blob can be megabytes wide, and nobody reads past the
# first screenful of one anyway.
_MAX_LINE_CHARS = 2000

# a step that prints once per row can print for a very long time. structured
# lines keep going past this; captured output stops, with a line saying so, so
# that a chatty step cannot quietly fill the disk.
_MAX_STREAM_BYTES = 64 * 1024 * 1024

#: Set ``TATERS_RUNLOG=0`` to turn all of this off.
_OFF_VALUES = {"0", "false", "no", "off"}

# how often a long step is allowed to say "still going".
_HEARTBEAT_SECONDS = 10.0


def logging_wanted() -> bool:
    """Whether logging is switched on for this process."""
    return os.environ.get("TATERS_RUNLOG", "").strip().lower() not in _OFF_VALUES


# ---------------------------------------------------------------------------
# Pure text handling -- no files, no descriptors
# ---------------------------------------------------------------------------

def clean_stream_text(raw: str) -> List[str]:
    """
    Turn a chunk of captured terminal output into plain log lines.

    Escape sequences are removed, and a carriage return is treated as the
    redraw it is: only the last frame of ``\\r``-separated output survives. A
    progress bar that repainted two thousand times therefore contributes one
    line showing where it finished, instead of two thousand near-identical ones.

    Parameters
    ----------
    raw : str
        Captured output, which may contain escape sequences, carriage returns
        and newlines.

    Returns
    -------
    list of str
        One entry per line, right-stripped. Blank entries are kept so that a
        caller can decide whether spacing matters.
    """
    lines = []
    for line in _ANSI.sub("", raw).split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]
        lines.append(line.rstrip())
    return lines


def describe_event(name: str, payload: dict) -> str:
    """One line for one of the runner's progress events."""
    get = payload.get
    if name == "run_start":
        return (f"start  {get('steps')} step(s), "
                f"{len(get('inputs') or [])} input(s)")
    if name == "step_start":
        return (f"step   {get('index')}/{get('total')}  {get('call')}  "
                f"(scope={get('scope')}, items={get('items')})")
    if name == "step_done":
        tail = f": {get('error')}" if get("error") else ""
        status = str(get("status") or "?").upper() if get("status") != "ok" else "ok"
        return f"step   {get('index')} -> {status}{tail}"
    if name == "item_start":
        return f"item   {get('item')}  {get('input')}"
    if name == "item_done":
        tail = f": {get('error')}" if get("error") else ""
        status = str(get("status") or "?")
        return f"item   {get('item')} -> {status if status == 'ok' else status.upper()}{tail}"
    if name == "step_progress":
        unit = get("unit") or ""
        total = get("total")
        done = f"{get('done')}/{total}" if total else f"{get('done')}"
        return f"prog   {get('index')}  {done} {unit}".rstrip()
    if name == "run_done":
        return f"done   manifest at {get('manifest_path')}"
    return f"{name}  {payload}"


def _prints_above_a_live_display(stream) -> bool:
    """
    Whether this stream is rich's stand-in for one it has taken over.

    A live progress display owns a region of the screen, and rich keeps
    anything printed while it is up from landing inside that region by
    swapping ``sys.stdout`` for a proxy of its own that prints *above* it.
    Writing past that proxy -- straight at the terminal -- is what puts a
    library's chatter through the middle of a progress bar.
    """
    return type(stream).__module__.startswith("rich.")


def session_fallback_path() -> Path:
    """
    Where a session that never started a run leaves its record.

    A crash in the menus has no work folder to write to, so it falls back to
    the same home folder the rest of Taters already uses for state.
    """
    from .settings import _home

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return _home() / "logs" / f"session-{stamp}.log"


def exception_text(exc: BaseException) -> str:
    """
    The whole traceback for an exception, chained causes and all.

    This is the function the incident needed. A library that wraps an import
    failure re-raises ``from e``, so its own message says only that something
    could not be imported -- while the exception it was raised *from* names the
    module that was actually missing. :func:`traceback.format_exception` walks
    ``__cause__`` and ``__context__`` for us and writes the "The above
    exception was the direct cause of the following exception" separators, so
    the entire chain comes back as one string.

    A string is also the only shape that survives the trip out of a worker: an
    item step is caught inside a thread or a spawned process, and a traceback
    object neither pickles nor outlives the frame it came from.

    Parameters
    ----------
    exc : BaseException
        The exception to render.

    Returns
    -------
    str
        The formatted traceback, ending in a newline.
    """
    return "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__))


def _installed(distribution: str) -> Optional[str]:
    """The installed version of a distribution, without importing it."""
    try:
        return _dist_version(distribution)
    except PackageNotFoundError:
        return None
    except Exception:
        # importlib.metadata can trip over a half-removed distribution. a
        # missing version line is not worth losing the log over.
        return None


def _version_lines(distributions: Sequence[str], width: int = 78) -> List[str]:
    """``# torch 2.11.0+cu128 | transformers 4.57.6 | ...``, wrapped."""
    parts = [f"{name} {_installed(name) or '-'}" for name in distributions]
    lines: List[str] = []
    current = "#"
    for part in parts:
        candidate = f"{current} {part}" if current == "#" else f"{current} | {part}"
        if len(candidate) > width and current != "#":
            lines.append(current)
            current = f"# {part}"
        else:
            current = candidate
    if current != "#":
        lines.append(current)
    return lines


def environment_lines(
    *,
    work_dir: Optional[PathLike] = None,
    preset: Optional[str] = None,
    workers: Optional[int] = None,
    command: Optional[str] = None,
    started: Optional[datetime] = None,
) -> List[str]:
    """
    The header block: when, where, on what, against which versions.

    Every value here was missing from the incident that prompted this module,
    and between them they answer "was this even the environment I think it
    was?" before any of the run's own output is read.

    Parameters
    ----------
    work_dir : str or pathlib.Path, optional
        The folder the run owns.
    preset : str, optional
        Name of the preset being run.
    workers : int, optional
        The resolved parallelism budget.
    command : str, optional
        A terminal command that reproduces the run.
    started : datetime.datetime, optional
        Overrides the clock, for tests.

    Returns
    -------
    list of str
        Comment lines, followed by one blank line.
    """
    when = (started or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")

    rows: List[Tuple[str, str]] = [
        ("taters", _installed("taters") or "(not installed)"),
        ("python", f"{platform.python_version()}  ({sys.executable})"),
        ("platform", f"{sys.platform}  {platform.platform(terse=True)}"),
    ]
    if work_dir is not None:
        rows.append(("work dir", str(work_dir)))
    if preset:
        rows.append(("preset", str(preset)))
    if workers is not None:
        rows.append(("workers", str(workers)))
    if command:
        rows.append(("command", str(command)))

    label = max(len(key) for key, _ in rows)
    out = [f"# taters run log -- started {when}", "#"]
    out.extend(f"# {key:<{label}}  {value}" for key, value in rows)
    out.append("#")
    out.extend(_version_lines(_INTERESTING))
    out.append("")
    return out


# ---------------------------------------------------------------------------
# The sink
# ---------------------------------------------------------------------------

class RunLog:
    """
    A run's log file, plus the machinery for capturing what is printed into it.

    Lines can be recorded before there is anywhere to put them: a session that
    starts in the menus buffers what it says until :meth:`open_run` is given a
    folder, at which point the buffer is flushed into the new file ahead of the
    run's own output. That is what makes the answers someone gave three screens
    back part of the record of the run they led to.

    Nothing here raises. A log that breaks a run would be worse than no log at
    all, and the most likely moment for it to be writing is while something
    else has already gone wrong -- so the first write failure switches the sink
    off for good and the run carries on without it.

    Parameters
    ----------
    enabled : bool, default True
        ``False`` makes every method a no-op, which is how a caller turns
        logging off without branching at each call site.
    console : rich.console.Console, optional
        The console a live display draws on, pinned to the real terminal for
        as long as output is being captured. See :meth:`capture_streams` for
        why leaving it unpinned would put the display in the log and nothing
        on the screen.
    """

    def __init__(self, *, enabled: bool = True, console=None) -> None:
        self._lock = threading.RLock()
        self._console = console
        self._last_heartbeat = 0.0
        self._buffer: List[str] = []
        self._handle = None
        self._path: Optional[Path] = None
        self._enabled = bool(enabled) and logging_wanted()
        self._broken = False
        self._pumps: List["_Pump"] = []
        self._stream_bytes = 0
        self._stream_capped = False
        self._ever_opened = False

    # -- state ------------------------------------------------------------

    @property
    def path(self) -> Optional[Path]:
        """Where the current run is being logged, or ``None``."""
        return self._path

    @property
    def ever_opened(self) -> bool:
        """Whether any run of this session got as far as writing a file."""
        return self._ever_opened

    @property
    def active(self) -> bool:
        """Whether anything is being recorded at all."""
        return self._enabled and not self._broken

    # -- recording --------------------------------------------------------

    def line(self, channel: str, text: str) -> None:
        """
        Record one line on one channel. Thread-safe, and never raises.

        Parameters
        ----------
        channel : str
            A short category -- ``ui``, ``run``, ``out``, ``err``, ``warn`` or
            ``trace``. Kept to one padded column so a channel can be grepped.
        text : str
            The message. Newlines in it are handled, but :meth:`block` is the
            better fit for anything deliberately multi-line.
        """
        if not self.active:
            return
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        head, *rest = str(text).split("\n")
        self._emit(f"{stamp}  {channel:<{CHANNEL_WIDTH}}  {head}")
        for extra in rest:
            self._emit(f"{_GUTTER}{extra}".rstrip())

    def block(self, channel: str, text: str) -> None:
        """
        Record a multi-line blob -- a traceback, or a captured chunk.

        The first line is timestamped like any other; the rest are indented
        under a gutter rather than each carrying its own clock, because forty
        identical timestamps down the side of a traceback help nobody.
        """
        if not self.active:
            return
        lines = clean_stream_text(str(text))
        while lines and not lines[-1]:
            lines.pop()
        if not lines:
            return
        self.line(channel, lines[0])
        for extra in lines[1:]:
            self._emit(f"{_GUTTER}{extra}".rstrip())

    def event(self, name: str, payload: dict) -> None:
        """
        Record one of the runner's progress events.

        Not all of them. ``item_progress`` fires many times a second and says
        nothing worth reading afterwards, so it is dropped. ``step_progress``
        is kept to one heartbeat every ten seconds, because "still going" is
        the one thing that tells a slow step apart from a hung one when you are
        reading this hours later.
        """
        if not self.active:
            return
        try:
            if name == "item_progress":
                return
            if name == "step_progress":
                if not payload.get("done") and not payload.get("total"):
                    return          # nothing counted yet; no news to report
                now = time.monotonic()
                if now - self._last_heartbeat < _HEARTBEAT_SECONDS:
                    return
                self._last_heartbeat = now
            self.line("run", describe_event(name, payload))
        except Exception:
            pass

    def stream(self, channel: str, raw: str) -> None:
        """
        Record captured terminal output, one log line per output line.

        Captured output is the one channel with no natural limit -- a step that
        prints per row prints as long as there are rows -- so it is the one
        that gets a budget. Structured lines are unaffected when it runs out.
        """
        if not self.active or self._stream_capped:
            return
        self._stream_bytes += len(raw)
        if self._stream_bytes > _MAX_STREAM_BYTES:
            self._stream_capped = True
            self.line("log", f"captured output passed {_MAX_STREAM_BYTES // (1024 * 1024)} MB; "
                             "no more of it will be recorded (the run is unaffected)")
            return
        for cleaned in clean_stream_text(raw):
            if cleaned:
                self.line(channel, cleaned)

    def _emit(self, formatted: str) -> None:
        """Put one finished line where it belongs. Never raises."""
        if len(formatted) > _MAX_LINE_CHARS:
            dropped = len(formatted) - _MAX_LINE_CHARS
            formatted = f"{formatted[:_MAX_LINE_CHARS]}... (+{dropped} chars)"
        with self._lock:
            if self._broken or not self._enabled:
                return
            try:
                if self._handle is None:
                    self._buffer.append(formatted)
                    excess = len(self._buffer) - _MAX_BUFFERED_LINES
                    if excess > 0:
                        del self._buffer[:excess]
                else:
                    self._handle.write(formatted + "\n")
                    self._handle.flush()
            except Exception:
                # one failure is enough to know this is not going to work.
                self._broken = True

    # -- the file ---------------------------------------------------------

    def open_run(self, work_dir: PathLike, **context) -> Optional[Path]:
        """
        Start a file for this run and flush anything said before it.

        The name carries a timestamp so that re-running keeps the evidence of
        the run that failed, rather than overwriting it with the run that was
        meant to prove the fix.

        Parameters
        ----------
        work_dir : str or pathlib.Path
            The run's folder. The log lands in ``<work_dir>/logs/``.
        **context
            Passed to :func:`environment_lines` for the header.

        Returns
        -------
        pathlib.Path or None
            Where the log is being written, or ``None`` if it could not be
            opened -- in which case lines keep buffering and nothing breaks.
        """
        if not self.active:
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(work_dir) / "logs" / f"run-{stamp}.log"
        # outside the lock on purpose -- see `close_run` for why holding it
        # across a thread join would be a deadlock.
        self.close_run()
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("w", encoding="utf-8", errors="replace")
            except OSError:
                # no folder to write to is not fatal; we keep buffering, and a
                # caller that has a fallback location can still ask for it.
                return None
            try:
                for header in environment_lines(work_dir=work_dir, **context):
                    handle.write(header + "\n")
                for buffered in self._buffer:
                    handle.write(buffered + "\n")
                handle.flush()
            except Exception:
                self._broken = True
                try:
                    handle.close()
                except OSError:
                    pass
                return None
            self._buffer.clear()
            self._handle = handle
            self._path = path
            self._ever_opened = True
        return path

    def close_run(self) -> None:
        """
        Finish the current file. Lines said afterwards buffer again.

        The pumps are stopped *before* the lock is taken, and that ordering is
        load-bearing: stopping one joins its thread, and that thread wants this
        same lock to record what it has just drained. Holding the lock across
        the join would be waiting on a thread that is waiting on us.
        """
        self.stop_capture()
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.flush()
                    self._handle.close()
                except Exception:
                    # not just OSError: a handle that has already been closed
                    # under us raises ValueError, and closing down is the one
                    # moment where refusing to fail actually matters.
                    pass
            self._handle = None
            self._path = None

    def write_buffer_to(self, path: PathLike, **context) -> Optional[Path]:
        """
        Dump whatever is buffered to ``path``.

        This is the safety net for a session that never got as far as a run --
        a crash in the menus still leaves its answers somewhere readable.
        """
        if not self._enabled:
            return None
        with self._lock:
            if not self._buffer:
                return None
            target = Path(path)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("w", encoding="utf-8", errors="replace") as fh:
                    for header in environment_lines(**context):
                        fh.write(header + "\n")
                    for buffered in self._buffer:
                        fh.write(buffered + "\n")
            except OSError:
                return None
            return target

    # -- capturing what gets printed --------------------------------------

    @contextmanager
    def capture_streams(self, *, console=None) -> Iterator[None]:
        """
        Tee everything printed into the log, for the duration of the block.

        Two layers, because neither is enough on its own. ``sys.stdout`` is
        replaced so that Python-level printing is recorded, and it writes to a
        held-open copy of the real terminal rather than through the descriptor,
        so nothing is logged twice. Underneath that, descriptors 1 and 2 are
        redirected into pipes, which catches what never passes through
        ``sys.stdout`` at all: extension modules, raw writes, and the output of
        child processes.

        Only ever wrap work. While a descriptor is a pipe ``isatty`` is false,
        and the interactive prompts render themselves differently when they
        believe they are not on a terminal -- so this must not be held open
        across a question.

        Parameters
        ----------
        console : rich.console.Console, optional
            The live display's console, which gets pinned to the real terminal
            for the duration. This is not optional in practice for anything
            with a progress display: a bare ``Console`` holds ``_file = None``
            and resolves ``sys.stdout`` *at write time*, so without pinning it
            would resolve to the tee installed here and every spinner frame
            would be recorded as though it were output. Pinning is an
            assignment to ``console.file``; nothing else will do it.
        """
        if self._handle is None or not self.active:
            yield
            return

        console = console if console is not None else self._console
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        had_console_file = getattr(console, "_file", None) if console is not None else None
        pinned: List = []
        try:
            for fd, channel in ((1, "out"), (2, "err")):
                try:
                    pump = _Pump(fd, self, channel)
                except OSError:
                    continue      # no such descriptor (pythonw, a closed fd)
                self._pumps.append(pump)
                terminal = os.fdopen(os.dup(pump.saved), "w", buffering=1,
                                     errors="replace")
                pinned.append(terminal)
                # if a live display has already taken this stream over, we hand
                # our writes back to it rather than to the raw terminal: it is
                # the thing that knows how to print above the bars instead of
                # through them. its own writes go to the console we pinned
                # above, so they never come back round into the capture.
                was = saved_stdout if fd == 1 else saved_stderr
                onward = was if _prints_above_a_live_display(was) else terminal
                tee = _TeeStream(onward, self, channel)
                if fd == 1:
                    sys.stdout = tee
                    if console is not None:
                        try:
                            console.file = terminal
                        except Exception:
                            pass
                else:
                    sys.stderr = tee
            yield
        finally:
            if console is not None:
                try:
                    # back to None means "follow sys.stdout again", which is
                    # how rich arrived.
                    console.file = had_console_file
                except Exception:
                    pass
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                    if isinstance(stream, _TeeStream):
                        stream.drain()
                except Exception:
                    pass
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
            self.stop_capture()
            for terminal in pinned:
                try:
                    terminal.close()
                except OSError:
                    pass

    def stop_capture(self) -> None:
        """Put the descriptors back. Safe to call more than once."""
        pumps, self._pumps = self._pumps, []
        for pump in pumps:
            pump.stop()


# ---------------------------------------------------------------------------
# The two capture layers
# ---------------------------------------------------------------------------

class _TeeStream:
    """
    A stand-in for ``sys.stdout`` that also records what passes through.

    It writes to a held-open copy of the terminal, *not* to the descriptor it
    replaced -- which is now a pipe feeding the same log. Going straight to the
    terminal is what keeps a single ``print`` from being recorded twice.
    """

    def __init__(self, terminal, log: RunLog, channel: str) -> None:
        self._terminal = terminal
        self._log = log
        self._channel = channel
        self._partial = ""

    def write(self, text: str) -> int:
        try:
            self._terminal.write(text)
        except Exception:
            pass
        self._partial += text
        while "\n" in self._partial:
            head, self._partial = self._partial.split("\n", 1)
            self._log.stream(self._channel, head)
        if len(self._partial) > _MAX_PARTIAL_BYTES:
            self._log.stream(self._channel, self._partial)
            self._partial = ""
        return len(text)

    def flush(self) -> None:
        try:
            self._terminal.flush()
        except Exception:
            pass

    def drain(self) -> None:
        """Record anything written without a closing newline."""
        if self._partial:
            self._log.stream(self._channel, self._partial)
            self._partial = ""

    def isatty(self) -> bool:
        # the terminal we hold really is one, and saying so is what keeps rich
        # and questionary rendering the way they would without us here.
        try:
            return bool(self._terminal.isatty())
        except Exception:
            return False

    def __getattr__(self, name: str):
        return getattr(self._terminal, name)


def _point_windows_handle_at(fd: int):
    """
    Point Windows' own standard handle at a redirected descriptor.

    On Windows ``os.dup2`` moves the C runtime's descriptor and nothing else.
    ``subprocess`` and ``multiprocessing`` hand a child the *OS handle* from
    ``GetStdHandle``, which ``dup2`` never touched -- so without this a child
    process keeps printing to the real console and its output, which is very
    often the interesting part, is the one thing the log misses.

    Returns whatever was there before, for restoring, or ``None`` on any
    platform or in any situation where this does not apply.
    """
    if sys.platform != "win32":
        return None
    slot = {1: -11, 2: -12}.get(fd)          # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
    if slot is None:
        return None
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        previous = kernel32.GetStdHandle(slot)
        kernel32.SetStdHandle(slot, ctypes.c_void_p(msvcrt.get_osfhandle(fd)))
        return previous
    except Exception:
        # best effort. losing child output is a smaller problem than refusing
        # to log at all, so this never propagates.
        return None


def _restore_windows_handle(fd: int, previous) -> None:
    """Undo :func:`_point_windows_handle_at`."""
    if previous is None or sys.platform != "win32":
        return
    slot = {1: -11, 2: -12}.get(fd)
    if slot is None:
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetStdHandle(slot, ctypes.c_void_p(previous))
    except Exception:
        pass


class _Pump:
    """
    One redirected descriptor, copied to the terminal and into the log.

    This is the layer that catches what Python never sees: an extension module
    writing through the C runtime, an ``os.write``, or a child process that
    inherited the descriptor.
    """

    def __init__(self, fd: int, log: RunLog, channel: str) -> None:
        self.fd = fd
        self._log = log
        self._channel = channel
        self.saved = os.dup(fd)             # raises OSError if fd is closed
        read_fd, write_fd = os.pipe()
        try:
            os.dup2(write_fd, fd)
        finally:
            os.close(write_fd)
        # from here the descriptor is a pipe, so every failure has to put it
        # back before it propagates. a redirected descriptor with nothing
        # reading it fills after about 64 KB and then blocks the next print
        # forever -- a hung run, which is far worse than no log.
        try:
            self._reader = os.fdopen(read_fd, "rb", 0)
            self._std_handle = _point_windows_handle_at(fd)
            self._thread = threading.Thread(
                target=self._drain, name=f"taters-log-fd{fd}", daemon=True)
            self._thread.start()
        except Exception:
            os.dup2(self.saved, fd)
            for closer in (lambda: os.close(self.saved),
                           lambda: os.close(read_fd)):
                try:
                    closer()
                except OSError:
                    pass
            raise

    def _drain(self) -> None:
        """
        Read until end of file, whatever else goes wrong.

        Only a closed pipe ends this loop. That is not defensiveness for its
        own sake: the pipe holds about 64 KB, and if we stopped reading, the
        next thing a step printed would block on a full buffer and never
        return. A bug in here has to degrade to "the output was not recorded",
        never to "the run hung" -- so everything inside the loop is swallowed
        and we keep reading regardless.
        """
        partial = b""
        while True:
            try:
                chunk = self._reader.read(4096)
            except (OSError, ValueError):
                break                      # the descriptor is gone; so are we
            if not chunk:
                break
            try:
                os.write(self.saved, chunk)
            except OSError:
                pass                       # the terminal went away; keep draining
            try:
                partial += chunk
                if b"\n" in partial:
                    *whole, partial = partial.split(b"\n")
                    for raw in whole:
                        self._record(raw)
                elif len(partial) > _MAX_PARTIAL_BYTES:
                    self._record(partial)
                    partial = b""
            except Exception:
                partial = b""              # lose the text, not the loop
        if partial:
            self._record(partial)

    def _record(self, raw: bytes) -> None:
        try:
            self._log.stream(self._channel, raw.decode("utf-8", "replace"))
        except Exception:
            pass

    def stop(self) -> None:
        """
        Drain, then put the descriptor back.

        The order matters and the wrong one is not subtle: closing the
        descriptor is what gives the reader its end-of-file, so it has to come
        before the join, and restoring has to come after -- otherwise the
        reader is still holding a descriptor that has been reassigned and the
        whole thing ends in ``Bad file descriptor``.
        """
        _restore_windows_handle(self.fd, self._std_handle)
        try:
            os.close(self.fd)
        except OSError:
            pass
        self._thread.join(timeout=5.0)
        try:
            os.dup2(self.saved, self.fd)
        except OSError:
            pass
        for closer in (lambda: os.close(self.saved), self._reader.close):
            try:
                closer()
            except OSError:
                pass
