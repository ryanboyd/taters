# taters/audio/convert_to_wav.py
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union
from ..helpers.cliargs import CliSpec

class FFmpegNotFoundError(RuntimeError):
    pass


#: PCM bit depth -> ffmpeg codec name. Shared with the stream splitter, which
#: used to carry its own copy of this table and of the PATH check below.
PCM_CODECS = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}

def _check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegNotFoundError("ffmpeg and/or ffprobe not found on PATH.")

def convert_audio_to_wav(
    input_path: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    sample_rate: int = 16000,          # common for ASR
    bit_depth: int = 16,               # 16/24/32 signed PCM
    channels: int = 1,                 # 1=mono, 2=stereo
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default
    verbose: bool = True,
) -> Path:
    """
    Convert any FFmpeg-readable audio/video file to a linear PCM WAV.

    Parameters
    ----------
    input_path : str | Path
        Source media file (audio or video container). FFmpeg must be able to read it.
    output_path : str | Path | None, optional
        Target WAV path. If None, ``<output_dir>/<input_stem>.wav``.
    output_dir : str | Path | None, optional
        Where the WAV goes when ``output_path`` is not given. Defaults to
        ``<cwd>/audio``.
    sample_rate : int, default 16000
        Desired sample rate (Hz).
    bit_depth : {16,24,32}, default 16
        Output PCM bit depth; maps to ``pcm_s{bit_depth}le`` codec.
    channels : int, default 1
        Output channels: 1 for mono, 2 for stereo.
    overwrite_existing : bool, default False
        Overwrite `output_path` if it already exists.
    verbose : bool, default True
        Print incidental notices (such as "already exists, skipping"). The
        pipeline runner passes False when a live display owns the screen.

    Returns
    -------
    Path
        Path to the written WAV file.

    Raises
    ------
    FileNotFoundError
        If `input_path` does not exist.
    RuntimeError
        If FFmpeg/FFprobe are missing or the conversion fails.

    Notes
    -----
    - Video inputs are supported: the audio stream is extracted and converted.
    - For multi-channel sources and `channels is None`, channel layout is preserved.
    - We run FFmpeg with ``-nostdin`` to avoid TTY issues in pipelines.
    """

    _check_ffmpeg()

    in_path = Path(input_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    if output_path and output_dir:
        raise ValueError("Provide at most one of output_path or output_dir.")

    if output_path:
        out_path = Path(output_path).resolve()
    else:
        base = in_path.stem + ".wav"
        out_dir = Path(output_dir).resolve() if output_dir else Path.cwd() / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / base

    if not overwrite_existing and Path(out_path).is_file():
        if verbose:
            print("WAV file already exists; returning existing file.")
        return out_path

    pcm_map = PCM_CODECS
    if bit_depth not in pcm_map:
        raise ValueError("bit_depth must be one of {16, 24, 32}.")
    if channels not in (1, 2):
        raise ValueError("channels must be 1 (mono) or 2 (stereo).")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner", "-loglevel", "error",
        "-y" if overwrite_existing else "-n",
        "-i", str(in_path),
        "-vn",                        # ignore video
        "-acodec", pcm_map[bit_depth],
        "-ar", str(sample_rate),
        "-ac", str(channels),
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        if not overwrite_existing and out_path.exists():
            raise FileExistsError(f"Target exists (use overwrite_existing=True): {out_path}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    return out_path


# --- CLI --------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Command line -- derived from the function(s) above; see helpers.cliargs.CliSpec.
# The aliases and legacy flags are the spellings the hand-written parser used,
# kept so every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    convert_audio_to_wav,
    description='Convert one audio or video file to PCM WAV.',
    aliases={
        'output_dir': ['--out-dir'],
        'output_path': ['--out'],
        'sample_rate': ['--sr'],
    },
    legacy={},
    positional=("input_path",),
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
