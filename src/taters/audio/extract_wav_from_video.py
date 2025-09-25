#!/usr/bin/env python3
# save as split_audio_streams.py

"""Extract all audio streams from a video/container into standalone WAV files.

This utility probes the container with `ffprobe`, lists audio streams (with
index and tags), and then maps each stream with `ffmpeg` to a separate PCM WAV.
It is useful for multi-track recordings (e.g., Zoom, OBS, ProRes with stems).
:contentReference[oaicite:1]{index=1}
"""


from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


class FFmpegNotFoundError(RuntimeError):
    pass


def _check_binaries():
    """Ensure ffmpeg and ffprobe are available."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegNotFoundError(
            "ffmpeg and/or ffprobe not found. Please install FFmpeg and make sure it's on your PATH."
        )


def _safe_slug(value: Optional[str]) -> str:
    """Make a filesystem-safe slug from tags like language/title."""
    if not value:
        return ""
    value = value.strip().lower()
    slug = re.sub(r"[^\w\-]+", "-", value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _probe_audio_streams(input_path: Path) -> list[dict]:
    """Return a list of audio stream dicts with index and tags via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,channels,channel_layout:stream_tags=language,title",
        "-of", "json",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    return data.get("streams", [])


def _build_wav_name(base: str, stream_idx: int, lang: Optional[str], title: Optional[str]) -> str:
    parts = [f"{base}", f"a{stream_idx}"]
    if (lang_s := _safe_slug(lang)):
        parts.append(lang_s)
    if (title_s := _safe_slug(title)):
        parts.append(title_s)
    return "_".join(parts) + ".wav"


def split_audio_streams_to_wav(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,     # <-- now optional
    sample_rate: int = 48000,
    bit_depth: int = 16,
    overwrite: bool = True,
) -> List[str]:
    """
    Extract each audio stream in a container to its own WAV file.

    Parameters
    ----------
    input_path : str | os.PathLike
        Video or audio container readable by FFmpeg.
    output_dir : str | os.PathLike | None, optional
        Destination directory. If None, defaults to ``./audio`` in the current
        working directory (predictable write location).
    sample_rate : int, default 48000
        Target sample rate for the output WAVs (Hz).
    bit_depth : {16,24,32}, default 16
        Output PCM bit depth (little-endian).
    overwrite : bool, default True
        If True, overwrite existing files. If False and a target exists,
        raises :class:`FileExistsError`.

    Returns
    -------
    list[str]
        Absolute paths to the created WAVs.

    Behavior
    --------
    - Output file names are constructed from the input base name and stream
      metadata: ``<stem>_a<index>[_<lang>][_<title>].wav`` with safe slugs.
    - Uses ``-map 0:a:<N>`` to select the N-th audio stream in the container.
    - Runs FFmpeg with ``-nostdin`` and quiet loglevel to avoid TTY lockups.

    Examples
    --------
    >>> split_audio_streams_to_wav("session.mp4")
    ['.../audio/session_a0_eng.wav', '.../audio/session_a1_eng.wav']
    """

    _check_binaries()

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Default predictable location when none is provided
    if output_dir is None:
        out_dir = Path.cwd() / "audio"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting audio streams from {in_path} to {out_dir} at {sample_rate} Hz, bit depth: {bit_depth}")

    streams = _probe_audio_streams(in_path)
    if not streams:
        raise ValueError("No audio streams found in input.")

    pcm_fmt_map = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
    if bit_depth not in pcm_fmt_map:
        raise ValueError("bit_depth must be one of {16, 24, 32}.")
    pcm_codec = pcm_fmt_map[bit_depth]

    created_files: List[str] = []
    base = in_path.stem

    for s in streams:
        idx = s.get("index")
        tags = s.get("tags", {}) or {}
        lang = tags.get("language")
        title = tags.get("title")

        print(f"Extracting audio stream:\n"
              f"index: {idx}\n"
              f"tags: {tags}\n"
              f"language: {lang}\n"
              f"title: {title}\n")

        out_name = _build_wav_name(base, idx, lang, title)
        out_path = out_dir / out_name

        ffmpeg_cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-map", f"0:a:{streams.index(s)}",  # Nth audio stream
            "-acodec", pcm_codec,
            "-ar", str(sample_rate),
            str(out_path),
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            if not overwrite and out_path.exists():
                raise FileExistsError(f"Target exists (use overwrite=True): {out_path}")
            raise RuntimeError(f"ffmpeg failed for stream {idx}: {result.stderr.strip()}")

        created_files.append(str(out_path))

    return created_files


# --- Optional CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split all audio streams from a video to individual WAV files.")
    parser.add_argument("input", help="Path to input video file")
    # Make output dir optional; default to <cwd>/audio/<input_stem>/
    parser.add_argument("output_dir", nargs="?", default=None, help="Directory for WAVs (default: ./audio/<stem>/)")
    parser.add_argument("--sr", type=int, default=48000, help="Output sample rate (default: 48000)")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[16, 24, 32], help="PCM bit depth (default: 16)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    paths = split_audio_streams_to_wav(
        args.input,
        args.output_dir,     # may be None → uses default path
        sample_rate=args.sr,
        bit_depth=args.bit_depth,
        overwrite=args.overwrite,
    )
    print("\n".join(paths))
