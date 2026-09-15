#!/usr/bin/env python3
# save as split_audio_streams.py

"""Extract all audio streams from a video/container into standalone WAV files.

This utility probes the container with `ffprobe`, lists audio streams (with
index and tags), and then maps each stream with `ffmpeg` to a separate PCM WAV.
It is useful for multi-track recordings (e.g., Zoom, OBS, ProRes with stems).
"""


from __future__ import annotations

import json
import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import List, Optional
from ..helpers.cliargs import CliSpec
from .convert_to_wav import PCM_CODECS, FFmpegNotFoundError, _check_ffmpeg  # noqa: F401  (re-exported)


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
    overwrite_existing: bool = False,
    *,
    overwrite: bool | None = None,   # deprecated alias for overwrite_existing
    verbose: bool = True,
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
    overwrite_existing : bool, default False
        If False (default) and a target WAV already exists, that stream is left
        alone and the existing path is returned, matching the rest of Taters.
        Set True to re-extract and replace.
    verbose : bool, default True
        Print each stream as it is extracted. The pipeline runner turns this
        off under its live display.
    overwrite : bool, optional
        Deprecated alias for `overwrite_existing`. Passing it emits a
        :class:`DeprecationWarning`. Note that this function used to default to
        overwriting; it now preserves existing files like every other writer.

    Returns
    -------
    list[str]
        Absolute paths to the WAVs for every audio stream, whether freshly
        written or already present.

    Behavior
    --------
    - Output file names are constructed from the input base name and stream
      metadata: ``<stem>_a<index>[_<lang>][_<title>].wav`` with safe slugs.
    - Existing outputs are decided in Python rather than by handing ffmpeg
      ``-n``: some ffmpeg builds refuse to overwrite but still exit 0, which
      would report a stale file as freshly written.
    - Uses ``-map 0:a:<N>`` to select the N-th audio stream in the container.
    - Runs FFmpeg with ``-nostdin`` and quiet loglevel to avoid TTY lockups.

    Examples
    --------
    >>> split_audio_streams_to_wav("session.mp4")
    ['.../audio/session_a0_eng.wav', '.../audio/session_a1_eng.wav']
    """

    if overwrite is not None:
        warnings.warn(
            "split_audio_streams_to_wav(overwrite=...) is deprecated; use "
            "overwrite_existing=... instead. Note the default also changed: "
            "existing WAVs are now kept rather than overwritten.",
            DeprecationWarning,
            stacklevel=2,
        )
        overwrite_existing = bool(overwrite)

    _check_ffmpeg()

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # no output dir given? we fall back to a predictable spot: ./audio
    if output_dir is None:
        out_dir = Path.cwd() / "audio"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Extracting audio streams from {in_path} to {out_dir} at "
              f"{sample_rate} Hz, bit depth: {bit_depth}")

    streams = _probe_audio_streams(in_path)
    if not streams:
        raise ValueError("No audio streams found in input.")

    if bit_depth not in PCM_CODECS:
        raise ValueError("bit_depth must be one of {16, 24, 32}.")
    pcm_codec = PCM_CODECS[bit_depth]

    created_files: List[str] = []
    base = in_path.stem

    for s in streams:
        idx = s.get("index")
        tags = s.get("tags", {}) or {}
        lang = tags.get("language")
        title = tags.get("title")

        if verbose:
            print(f"Extracting audio stream:\n"
                  f"index: {idx}\n"
                  f"tags: {tags}\n"
                  f"language: {lang}\n"
                  f"title: {title}\n")

        out_name = _build_wav_name(base, idx, lang, title)
        out_path = out_dir / out_name

        # we decide about existing files ourselves rather than leaning on
        # ffmpeg's "-n": some ffmpeg builds refuse to overwrite but still exit
        # 0, so we'd end up reporting a stale file as freshly written.
        if out_path.exists() and not overwrite_existing:
            if verbose:
                print(f"WAV already exists; returning existing file: {out_path}")
            created_files.append(str(out_path))
            continue

        ffmpeg_cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(in_path),
            "-map", f"0:a:{streams.index(s)}",  # Nth audio stream
            "-acodec", pcm_codec,
            "-ar", str(sample_rate),
            str(out_path),
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for stream {idx}: {result.stderr.strip()}")
        if not out_path.is_file():
            # ffmpeg can claim success without ever producing a file; we never
            # want to hand back a path that isn't there.
            raise RuntimeError(
                f"ffmpeg reported success but no file was written for stream {idx}: "
                f"{out_path}\n{result.stderr.strip()}"
            )

        created_files.append(str(out_path))

    return created_files


# --- Optional CLI ---


# ---------------------------------------------------------------------------
# Command line -- derived from the function(s) above; see helpers.cliargs.CliSpec.
# The aliases and legacy flags are the spellings the hand-written parser used,
# kept so every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    split_audio_streams_to_wav,
    description='Split every audio stream of a container into its own WAV.',
    aliases={
        'sample_rate': ['--sr'],
    },
    legacy={},
    positional=("input_path", "output_dir"),
    skip=("overwrite",),
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
