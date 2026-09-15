"""
Write a file under a scratch name, and give it its real name only when finished.

Analysis steps stream their output row by row, which means the output file
exists -- with its final name -- from the first row onwards. Anything that
interrupts the run leaves that half-written file behind looking exactly like a
completed one.

That would be survivable on its own. What makes it a real hazard is the other
half of the design: steps skip work whose output already exists, because that
is what makes a long pipeline resumable. So a truncated file is not merely
wrong, it is *sticky* -- the next run sees it, decides the step is done, and
returns it. You get 3 rows where you asked for 300,000, with no error.

The fix is to write to ``<name>.part`` and rename it at the end. A rename is
indivisible as far as the filesystem is concerned: there is no moment at which
the destination exists half-renamed. Every other part of the write can be
interrupted; that step cannot. So the real name only ever refers to a complete
file, and an interrupted run leaves nothing for the next one to mistake for
finished work.

This covers interruption generally -- a cancelled run, a full disk, a power
cut, Ctrl-C -- not just any one of them.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Iterator, Union

__all__ = ["atomic_write", "SCRATCH_SUFFIX"]

# we keep this next to the destination on purpose. a rename is only atomic
# within one filesystem; a scratch file in /tmp could land on a different
# device, where `os.replace` quietly degrades to a copy and we lose the guarantee.
SCRATCH_SUFFIX = ".part"


@contextmanager
def atomic_write(
    path: Union[str, Path],
    mode: str = "w",
    **open_kwargs: Any,
) -> Iterator[IO]:
    """
    Open a file for writing that only appears at ``path`` once complete.

    Parameters
    ----------
    path : str or pathlib.Path
        Where the finished file should end up. Parent directories are created.
    mode : str, default "w"
        As :func:`open`. Must be a writing mode.
    **open_kwargs
        Passed straight through to :func:`open` -- ``newline``, ``encoding``
        and so on.

    Yields
    ------
    IO
        The handle to write to. It refers to the scratch file, not to ``path``.

    Notes
    -----
    On an exception the scratch file is removed and ``path`` is left exactly as
    it was -- which for a first run means absent, so the step is retried rather
    than resumed from a partial file.

    Not safe for two processes writing the same destination at once: they would
    share a scratch name. Nothing in Taters does that, since a GLOBAL step runs
    once and ITEM steps write per-input paths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_name(path.name + SCRATCH_SUFFIX)

    handle = scratch.open(mode, **open_kwargs)
    try:
        yield handle
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt is the most likely
        # way we land here, and that's exactly the case this exists for.
        handle.close()
        scratch.unlink(missing_ok=True)
        raise
    else:
        handle.close()
        os.replace(scratch, path)
