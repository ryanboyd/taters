"""
Pulling frames out of a video, with ffmpeg.

Every other way of doing this costs a dependency. `opencv-python` is 60 MB and
brings its own ffmpeg; `torchcodec` is pinned to an exact torch version;
`moviepy` drags imageio and a second ffmpeg binary. Taters already requires
ffmpeg and already drives it by subprocess, so decoding here is a subprocess
call and nothing else.

Doing it ourselves buys three things beyond the saved dependency. The sampling
rate becomes an explicit setting that gets recorded in the provenance, instead
of a library's hidden default. The frame count is known before any model runs,
so progress is a real bar rather than a spinner. And the timestamps are
arithmetic -- `fps=N` resamples to exactly N frames per second, so frame *k* is
at *k/N* seconds -- which is what lets a face detection be matched back to the
shot it happened in.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

PathLike = Union[str, Path]

__all__ = ["Frame", "extracted_frames", "extract_frames"]


@dataclass(frozen=True)
class Frame:
    """One sampled frame: where it is on disk, and when it happened."""

    index: int
    time_s: float
    path: Path


def extract_frames(video_path: PathLike, out_dir: PathLike, *,
                   fps: float = 1.0,
                   max_frames: Optional[int] = None,
                   width: Optional[int] = 640,
                   quality: int = 3,
                   on_progress: Optional[Callable[..., None]] = None) -> List[Frame]:
    """
    Sample frames at a fixed rate and write them as JPEGs.

    Parameters
    ----------
    video_path : str or pathlib.Path
        The video to sample.
    out_dir : str or pathlib.Path
        Where the frames go. Created if it does not exist.
    fps : float, default=1.0
        Frames per second to sample. One per second is plenty for measuring who
        is on screen and what their face is doing; raising it costs time
        linearly and buys temporal resolution.
    max_frames : int, optional
        Stop after this many. A guard against someone pointing this at a
        feature film at 25 fps.
    width : int, optional
        Scale to this width, preserving aspect. Face models work on small crops
        anyway, and the decode is most of the cost. ``None`` keeps full size.
    quality : int, default=3
        JPEG quality for ffmpeg's ``-q:v``, 2 (best) to 31.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)``.

    Returns
    -------
    list of Frame
        In time order. Empty if the video has no decodable video stream.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is not on PATH. Taters needs it for every media step; "
            "see https://ffmpeg.org/download.html")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chain = f"fps={fps}"
    if width:
        # -2 keeps the height even, which JPEG does not care about but every
        # subsequent encoder does
        chain += f",scale={int(width)}:-2:flags=bilinear"

    args = [ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y",
            "-i", str(video_path), "-vf", chain, "-an"]
    if max_frames:
        args += ["-frames:v", str(int(max_frames))]
    args += ["-q:v", str(int(quality)), str(out_dir / "frame_%06d.jpg")]

    if on_progress is not None:
        on_progress(0, 0, "extracting frames")
    subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                   errors="replace")

    # ffmpeg numbers from 1; the time of the k-th sampled frame is k/fps
    # because `fps=` resamples rather than selecting existing timestamps
    files = sorted(out_dir.glob("frame_*.jpg"))
    return [Frame(index=i, time_s=i / float(fps), path=p)
            for i, p in enumerate(files)]


@contextmanager
def extracted_frames(video_path: PathLike, **kwargs) -> Iterator[List[Frame]]:
    """
    :func:`extract_frames` into a temporary folder that cleans itself up.

    A minute of video at 1 fps is sixty JPEGs; a lecture is three thousand.
    Leaving those behind in someone's working directory would be rude, and
    keeping them in memory instead would mean holding a whole video's frames at
    once for no benefit -- the models read them one at a time.
    """
    tmp = Path(tempfile.mkdtemp(prefix="taters_frames_"))
    try:
        yield extract_frames(video_path, tmp, **kwargs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
