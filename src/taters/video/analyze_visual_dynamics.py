"""
How a video is cut and paced, one row per file.

This is the structural half of video analysis: not what is in the picture, but
how the picture behaves. How often it cuts, how long the shots run, how much it
moves between cuts, how much of it is black or frozen.

**Average shot length** is the headline number and it has a real measurement
tradition behind it -- the cinemetrics literature, Barry Salt's shot-length
statistics, Cutting and colleagues on the pacing of Hollywood film. It is the
covariate anybody studying edited media reaches for first, and it is also the
thing you want to hold constant before claiming that something *else* about a
video predicts an outcome.

The continuous measure beside it is the more unusual one. ffmpeg's `scdet`
scores every frame for how different it is from the last, and asking for the
whole distribution rather than a list of cuts gives a description of visual
volatility: a locked-off interview and a handheld chase can have identical shot
lengths and nothing else in common.

Everything here comes from filters that ship inside ffmpeg, which Taters already
requires. Nothing new is installed and nothing can collide.

One honest limit, worth knowing before reading a cut count: `scdet` works on
luminance, so a cut between two shots of similar brightness can score low. A
transition from a red screen to a green one of the same luma is invisible to it.
On real footage that is rarely the situation; on synthetic or heavily graded
material it can be.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec
from ..helpers.progress import announce
from ..helpers.provenance import records_settings
from ._ffmpeg_probe import (Span, black_spans, freeze_spans, probe_video,
                            scene_scores)

PathLike = Union[str, Path]

__all__ = ["analyze_visual_dynamics", "FEATURE_COLUMNS", "BOOKKEEPING"]

#: What the file is, rather than what is in it. Useful to filter on -- nobody
#: wants a 3-second clip in with the feature films -- and meaningless as a
#: predictor, so the statistics stage keeps them out of the feature sets. A
#: model that discovered your outcome correlates with frame width has
#: discovered how you collected your videos.
BOOKKEEPING = ("vid_duration_s", "vid_fps", "vid_width", "vid_height",
               "vid_n_frames", "vid_frames_scored")

MEASURES = (
    "vid_cut_count", "vid_cuts_per_min",
    "vid_shot_len_mean", "vid_shot_len_sd", "vid_shot_len_median",
    "vid_shot_len_min", "vid_shot_len_max",
    "vid_change_mean", "vid_change_sd", "vid_change_max", "vid_change_p95",
    "vid_black_count", "vid_black_total_s", "vid_black_prop",
    "vid_freeze_count", "vid_freeze_total_s", "vid_freeze_prop",
)

FEATURE_COLUMNS = ColumnSpec(label="Visual dynamics",
                             names=tuple(MEASURES) + BOOKKEEPING)


def _sd(values: Sequence[float]) -> Optional[float]:
    """Sample SD, or nothing when one number cannot have a spread."""
    return statistics.stdev(values) if len(values) > 1 else None


def _percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def _total(spans: Sequence[Span]) -> float:
    return float(sum(d for _s, _e, d in spans))


def _shot_lengths(cut_times: Sequence[float], duration: Optional[float]) -> List[float]:
    """
    Shot durations from the cut times.

    A video with no cuts is one shot the length of the file, which is the right
    answer and not a missing one. Without a duration we cannot close the last
    shot, so it is left out rather than guessed at.
    """
    if duration is None or duration <= 0:
        return []
    edges = [0.0] + [t for t in cut_times if 0.0 < t < duration] + [float(duration)]
    return [b - a for a, b in zip(edges, edges[1:]) if b > a]


@records_settings(binding=("video_path",),
                  outputs=("out_csv",),
                  bookkeeping=BOOKKEEPING)
def analyze_visual_dynamics(
    *,
    video_path: PathLike,
    out_csv: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    overwrite_existing: bool = False,
    cut_threshold: float = 10.0,
    black_min_duration: float = 0.1,
    black_pixel_threshold: float = 0.10,
    freeze_min_duration: float = 0.5,
    freeze_noise: float = 0.001,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Measure how a video is cut and paced, and write one row of features.

    Parameters
    ----------
    video_path : str or pathlib.Path
        The video to measure. Anything ffmpeg can read.
    out_csv : str or pathlib.Path, optional
        Where to write. Defaults to ``<out_dir>/<video stem>.csv``.
    out_dir : str or pathlib.Path, optional
        Folder for the default output path. Defaults to
        ``./features/visual_dynamics``.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, skip and return the existing path.
    cut_threshold : float, default=10.0
        Scene-change score above which a frame counts as a cut, on ffmpeg's
        0--100 scale. Raise it if gentle transitions are being counted as cuts;
        lower it if real cuts are being missed. The per-frame scores are
        summarized in the output either way, so you can see where yours sit.
    black_min_duration : float, default=0.1
        Shortest run of black frames, in seconds, that counts as a black span.
    black_pixel_threshold : float, default=0.10
        How dark a pixel has to be to count as black, 0--1.
    freeze_min_duration : float, default=0.5
        Shortest run of unchanging frames, in seconds, that counts as frozen.
    freeze_noise : float, default=0.001
        How much change is tolerated while still calling the picture frozen.
    verbose : bool, default=True
        Print what is happening.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)``.

    Returns
    -------
    pathlib.Path
        Path to the written CSV.

    Notes
    -----
    Shot lengths are derived from the cut times, so a video with no detected
    cuts is reported as a single shot the length of the file.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if out_csv is None:
        base = Path(out_dir) if out_dir else (Path.cwd() / "features" / "visual_dynamics")
        out_csv = base / f"{video_path.stem}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_csv.is_file():
        if verbose:
            print(f"Visual dynamics output already exists; returning {out_csv}")
        return out_csv

    announce(on_progress, f"reading {video_path.name}")
    info = probe_video(video_path)
    duration = info.get("duration_s")

    # a media folder is usually a mix, and an .mp3 has no pictures to measure.
    # skipping it quietly beats failing the whole run, and beats writing a row
    # of empties that would look like a video we measured badly.
    if info.get("width") is None:
        if verbose:
            print(f"{video_path.name} has no video stream; nothing to measure.")
        return out_csv

    announce(on_progress, "scoring scene changes")
    scores = scene_scores(video_path)

    announce(on_progress, "finding black and frozen spans")
    blacks = black_spans(video_path, min_duration=black_min_duration,
                         pixel_threshold=black_pixel_threshold)
    freezes = freeze_spans(video_path, min_duration=freeze_min_duration,
                           noise=freeze_noise, duration=duration)

    values = [s for _f, _t, s in scores]
    cut_times = [t for _f, t, s in scores if s >= cut_threshold]
    shots = _shot_lengths(cut_times, duration)
    ordered = sorted(values)

    row: Dict[str, object] = {
        "vid_duration_s": duration,
        "vid_fps": info.get("fps"),
        "vid_width": info.get("width"),
        "vid_height": info.get("height"),
        "vid_n_frames": info.get("n_frames"),
        "vid_frames_scored": len(scores),

        "vid_cut_count": len(cut_times),
        "vid_cuts_per_min": (len(cut_times) / duration * 60.0)
                            if duration else None,

        "vid_shot_len_mean": statistics.fmean(shots) if shots else None,
        "vid_shot_len_sd": _sd(shots),
        "vid_shot_len_median": statistics.median(shots) if shots else None,
        "vid_shot_len_min": min(shots) if shots else None,
        "vid_shot_len_max": max(shots) if shots else None,

        "vid_change_mean": statistics.fmean(values) if values else None,
        "vid_change_sd": _sd(values),
        "vid_change_max": max(values) if values else None,
        "vid_change_p95": _percentile(ordered, 0.95),

        "vid_black_count": len(blacks),
        "vid_black_total_s": _total(blacks),
        "vid_black_prop": (_total(blacks) / duration) if duration else None,

        "vid_freeze_count": len(freezes),
        "vid_freeze_total_s": _total(freezes),
        "vid_freeze_prop": (_total(freezes) / duration) if duration else None,
    }

    header = ["source_path", "source"] + list(MEASURES) + list(BOOKKEEPING)
    with atomic_write(out_csv, newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow([str(video_path.resolve()), video_path.stem]
                        + [_fmt(row.get(c)) for c in MEASURES]
                        + [_fmt(row.get(c)) for c in BOOKKEEPING])

    if verbose:
        print(f"Visual dynamics: {video_path.name} -> {len(cut_times)} cuts, "
              f"{len(scores)} frames scored")
    return out_csv


def _fmt(value) -> object:
    """Round for the file, and leave "we could not measure this" empty."""
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


CLI = CliSpec(
    analyze_visual_dynamics,
    description="Measure how a video is cut and paced (shots, cuts, black, frozen).",
    aliases={"video_path": ["--video"], "out_csv": ["--out"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
