"""
Reading structure out of ffmpeg, and the parsers that make it a number.

ffmpeg already knows how to find scene changes, black frames and frozen frames
-- `scdet`, `blackdetect` and `freezedetect` have shipped with it for years.
Taters already requires ffmpeg and already drives it by subprocess, so the whole
structural half of video analysis costs nothing to install.

What it does cost is parsing, and that is the brittle part: the filters announce
themselves as loose text on stderr in formats nobody versioned. So the parsing
lives here as plain functions over a string, with no subprocess anywhere near
them, and the tests feed them captured real output. When a future ffmpeg changes
its wording, one of these fails with an obvious message instead of a feature
silently reporting zero cuts in everything.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

__all__ = [
    "SceneScore", "Span", "probe_video", "parse_scene_scores",
    "parse_black_spans", "parse_freeze_spans", "scene_scores", "black_spans",
    "freeze_spans", "FfmpegMissing",
]

#: One frame's scene-change score: (frame index, time in seconds, score 0-100).
SceneScore = Tuple[int, float, float]

#: One detected interval: (start seconds, end seconds, duration seconds).
Span = Tuple[float, float, float]


class FfmpegMissing(RuntimeError):
    """ffmpeg or ffprobe is not on PATH."""


# ---------------------------------------------------------------------------
# Parsers -- pure functions over text, which is the whole point
# ---------------------------------------------------------------------------

#: `metadata=print` writes the frame header and the value on separate lines, so
#: the frame number has to be carried forward to the line that follows it.
_FRAME_RE = re.compile(r"^frame:(\d+)\s+pts:(-?\d+)\s+pts_time:(-?[\d.]+)")
_SCORE_RE = re.compile(r"^lavfi\.scd\.score=([\d.]+)")

#: blackdetect puts the whole interval on one line.
_BLACK_RE = re.compile(
    r"black_start:(-?[\d.]+)\s+black_end:(-?[\d.]+)\s+black_duration:([\d.]+)")

#: freezedetect spreads one interval over three lines and, unlike blackdetect,
#: reports the start before it knows the end -- so a freeze running to the end
#: of the file has a start and no end at all.
_FREEZE_START_RE = re.compile(r"lavfi\.freezedetect\.freeze_start:\s*(-?[\d.]+)")
_FREEZE_END_RE = re.compile(r"lavfi\.freezedetect\.freeze_end:\s*(-?[\d.]+)")


def parse_scene_scores(text: str) -> List[SceneScore]:
    """
    Per-frame scene-change scores from ``metadata=print`` output.

    Returns ``(frame, seconds, score)`` per frame, in file order.
    """
    out: List[SceneScore] = []
    frame: Optional[int] = None
    when: float = 0.0
    for line in text.splitlines():
        line = line.strip()
        m = _FRAME_RE.match(line)
        if m:
            frame, when = int(m.group(1)), float(m.group(3))
            continue
        m = _SCORE_RE.match(line)
        if m and frame is not None:
            out.append((frame, when, float(m.group(1))))
            frame = None
    return out


def parse_black_spans(text: str) -> List[Span]:
    """Black intervals from ``blackdetect`` output."""
    return [(float(a), float(b), float(c))
            for a, b, c in _BLACK_RE.findall(text)]


def parse_freeze_spans(text: str, *, duration: Optional[float] = None) -> List[Span]:
    """
    Frozen intervals from ``freezedetect`` output.

    A freeze that runs to the end of the file gets a start and never an end, so
    it is closed at ``duration`` when we know it and dropped when we do not --
    inventing an end would put a made-up number in a results table.
    """
    starts = [float(x) for x in _FREEZE_START_RE.findall(text)]
    ends = [float(x) for x in _FREEZE_END_RE.findall(text)]

    spans: List[Span] = []
    for i, start in enumerate(starts):
        if i < len(ends):
            end = ends[i]
        elif duration is not None and duration > start:
            end = duration
        else:
            continue
        spans.append((start, end, max(0.0, end - start)))
    return spans


# ---------------------------------------------------------------------------
# Running ffmpeg
# ---------------------------------------------------------------------------

def _require(tool: str) -> str:
    found = shutil.which(tool)
    if not found:
        raise FfmpegMissing(
            f"{tool} is not on PATH. Taters needs it for every media step; "
            "see https://ffmpeg.org/download.html")
    return found


def _run(args: Sequence[str]) -> str:
    """Run a filter pass and hand back everything it said, both streams.

    The filters write to stderr and `metadata=print:file=-` writes to stdout, so
    a caller that only read one of them would silently get nothing.
    """
    proc = subprocess.run(list(args), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def probe_video(path: PathLike) -> Dict[str, Optional[float]]:
    """
    Duration, frame rate, size and frame count, via ffprobe.

    `nb_frames` is absent from plenty of containers, so it comes back as None
    rather than a guess; anything that needs a frame count can work it out from
    duration and fps and say that it did.
    """
    ffprobe = _require("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        blob = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        blob = {}

    streams = blob.get("streams") or [{}]
    stream = streams[0] if streams else {}
    fmt = blob.get("format") or {}

    fps: Optional[float] = None
    rate = stream.get("r_frame_rate") or ""
    if "/" in str(rate):
        num, _, den = str(rate).partition("/")
        try:
            fps = float(num) / float(den) if float(den) else None
        except (ValueError, ZeroDivisionError):
            fps = None

    def _num(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "duration_s": _num(fmt.get("duration")),
        "fps": fps,
        "width": _num(stream.get("width")),
        "height": _num(stream.get("height")),
        "n_frames": _num(stream.get("nb_frames")),
    }


def scene_scores(path: PathLike) -> List[SceneScore]:
    """
    Every frame's scene-change score.

    `scdet` is asked for threshold 0 on purpose: we want the whole distribution
    and will pick the cut threshold ourselves. Letting the filter decide would
    hand back a bare list of cuts and throw away the continuous measure, which
    is the more interesting half -- how much the picture is moving is a
    description of pacing, not just a list of where the edits are.
    """
    ffmpeg = _require("ffmpeg")
    return parse_scene_scores(_run([
        ffmpeg, "-hide_banner", "-nostdin", "-i", str(path),
        "-vf", "scdet=threshold=0,metadata=print:key=lavfi.scd.score:file=-",
        "-an", "-f", "null", "-",
    ]))


def black_spans(path: PathLike, *, min_duration: float = 0.1,
                pixel_threshold: float = 0.10) -> List[Span]:
    """Intervals that are (almost) entirely black."""
    ffmpeg = _require("ffmpeg")
    return parse_black_spans(_run([
        ffmpeg, "-hide_banner", "-nostdin", "-i", str(path),
        "-vf", f"blackdetect=d={min_duration}:pix_th={pixel_threshold}",
        "-an", "-f", "null", "-",
    ]))


def freeze_spans(path: PathLike, *, min_duration: float = 0.5,
                 noise: float = 0.001,
                 duration: Optional[float] = None) -> List[Span]:
    """Intervals where the picture stopped changing."""
    ffmpeg = _require("ffmpeg")
    return parse_freeze_spans(_run([
        ffmpeg, "-hide_banner", "-nostdin", "-i", str(path),
        "-vf", f"freezedetect=n={noise}:d={min_duration}",
        "-an", "-f", "null", "-",
    ]), duration=duration)
