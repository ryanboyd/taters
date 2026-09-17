"""
"A newer version is out" -- said once, quietly, and never in the way.

The rules this is built around, in order of importance:

1. **The menu never waits for the network.** The note is read from a cached
   answer, which is instant; the refresh that produces that answer runs in a
   background thread and its result is for *next* launch. A slow network, a
   proxy that blackholes the request, no network at all -- none of it can
   delay the first screen or stop Taters starting.
2. **It can be turned off, and turning it off is honored everywhere.** Some
   people run this on data that cannot leave the building, and an unexplained
   outbound connection is a conversation with IT nobody wants. Setting
   ``TATERS_NO_UPDATE_CHECK=1``, or the saved setting, stops it dead -- no
   thread, no request.
3. **It cannot break anything.** Every path here is wrapped. A failure leaves
   no note and no complaint; there is nothing here worth interrupting a run
   for.
4. **It says one thing.** "Newer version available: v0.7.3" under the banner,
   in dim text. Not an exhortation, not something to dismiss.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Optional, Tuple

from .settings import load_settings, save_setting

__all__ = ["note", "refresh_in_background", "SKIP_ENV", "SETTING_KEY"]

#: Set this to anything truthy and nothing here reaches the network.
SKIP_ENV = "TATERS_NO_UPDATE_CHECK"

#: Where the last answer lives, under the usual settings file.
SETTING_KEY = "update_check"

#: How long an answer stays good. A day: new releases are not hourly news, and
#: this is the difference between one request per day and one per launch for
#: somebody who opens Taters twenty times in an afternoon.
MAX_AGE_SECONDS = 24 * 60 * 60

#: PyPI's own metadata endpoint. No tracking, no redirect, one small JSON body.
PYPI_JSON = "https://pypi.org/pypi/taters/json"

_TIMEOUT = 5.0


def _release(text: str) -> Optional[Tuple[int, ...]]:
    """``"0.7.10"`` -> ``(0, 7, 10)``; anything unrecognizable -> None.

    Deliberately not `packaging.version`: it is not a declared dependency, and
    the comparison needed here is "is the number bigger", over versions this
    project actually issues. Anything with a suffix (rc, dev, post) fails to
    parse and the note simply does not appear, which is the right outcome --
    telling somebody on 0.8.0rc1 that 0.7.9 is "newer" is worse than silence.
    """
    if not re.fullmatch(r"\d+(\.\d+)*", text.strip()):
        return None
    return tuple(int(part) for part in text.strip().split("."))


def _disabled() -> bool:
    if os.environ.get(SKIP_ENV, "").strip().lower() not in ("", "0", "false", "no"):
        return True
    try:
        return bool(load_settings().get(SETTING_KEY, {}).get("off"))
    except Exception:
        return False


def _installed() -> str:
    try:
        from importlib.metadata import version
        return version("taters")
    except Exception:
        return ""


def note() -> str:
    """The line to print under the banner, or "" -- read from cache, instantly.

    Returns "" when: the check is off, Taters is running from a source tree
    with no version metadata, nothing has been cached yet (the first launch),
    the cached answer is not newer, or anything at all goes wrong.
    """
    try:
        if _disabled():
            return ""
        here = _release(_installed())
        if here is None:
            # no metadata, or a version this cannot compare. running from a
            # checkout is the normal case for that, and a developer does not
            # need telling that PyPI is behind them.
            return ""
        seen = load_settings().get(SETTING_KEY, {}).get("latest", "")
        there = _release(str(seen))
        if there is None or there <= here:
            return ""
        return f"Newer version available: v{seen}"
    except Exception:
        return ""


def _fetch_and_store() -> None:
    """One request, one settings write. Never raises -- it runs unattended."""
    try:
        import json
        import time
        from urllib.request import Request, urlopen

        saved = load_settings().get(SETTING_KEY, {})
        if time.time() - float(saved.get("at", 0) or 0) < MAX_AGE_SECONDS:
            return

        request = Request(PYPI_JSON, headers={"Accept": "application/json"})
        with urlopen(request, timeout=_TIMEOUT) as response:   # noqa: S310
            latest = json.load(response)["info"]["version"]

        if _release(str(latest)) is None:
            return

        # re-read rather than reusing the snapshot from before the request. we
        # were just sitting on a socket for up to five seconds, which is ample
        # time for somebody to have opened Settings and switched this off --
        # and writing back the old snapshot would quietly switch it on again.
        saved = load_settings().get(SETTING_KEY, {})
        if saved.get("off"):
            return
        # `at` is stamped even when the version is unchanged, so a quiet week
        # is one request a day rather than one per launch.
        save_setting(SETTING_KEY, {**saved, "latest": str(latest), "at": time.time()})
    except Exception:
        pass


def refresh_in_background() -> Optional[threading.Thread]:
    """Start the refresh, or don't. Returns the thread for tests to join.

    A daemon thread: if somebody quits Taters two seconds after opening it,
    the interpreter exits without waiting on a socket nobody is reading.
    """
    try:
        if _disabled() or _release(_installed()) is None:
            return None
        thread = threading.Thread(target=_fetch_and_store, name="taters-update-check",
                                  daemon=True)
        thread.start()
        return thread
    except Exception:
        return None
