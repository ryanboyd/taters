# taters/audio/convert_to_wav.py
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

class FFmpegNotFoundError(RuntimeError):
    pass

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
) -> Path:
    """
    Convert any FFmpeg-readable audio/video file to a linear PCM WAV.

    Parameters
    ----------
    input_path : str | Path
        Source media file (audio or video container). FFmpeg must be able to read it.
    output_path : str | Path | None, optional
        Target WAV path. If None, defaults to
        ``<cwd>/audio/<input_stem>.wav``.
    sample_rate : int, default 16000
        Desired sample rate (Hz).
    bit_depth : {16,24,32}, default 16
        Output PCM bit depth; maps to ``pcm_s{bit_depth}le`` codec.
    channels : int | None, default 1
        If provided, set number of output channels (e.g., 1=mono, 2=stereo).
        If None, keep original channel count.
    overwrite_existing : bool, default False
        Overwrite `output_path` if it already exists.

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
        print("WAV file already exists; returning existing file.")
        return out_path

    pcm_map = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
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
            raise FileExistsError(f"Target exists (use overwrite=True): {out_path}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    return out_path


# --- CLI --------------------------------------------------------------------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Convert any audio (or A/V) file to PCM WAV via ffmpeg.")
    p.add_argument("input", help="Input file (audio or video container)")
    p.add_argument("--out", dest="output_path", default=None,
                   help="Exact output .wav path (overrides --out-dir)")
    p.add_argument("--out-dir", dest="output_dir", default=None,
                   help="Directory for output (filename will be <input_stem>.wav)")
    p.add_argument("--sr", dest="sample_rate", type=int, default=16000, help="Sample rate (Hz)")
    p.add_argument("--bit-depth", type=int, choices=[16, 24, 32], default=16, help="PCM bit depth")
    p.add_argument("--channels", type=int, choices=[1, 2], default=1, help="1=mono, 2=stereo")
    p.add_argument("--overwrite_existing", type=bool, default=False, help="Overwrite existing output")
    return p

def main():
    args = _build_arg_parser().parse_args()
    out = convert_audio_to_wav(
        args.input,
        output_path=args.output_path,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        channels=args.channels,
        overwrite_existing=args.overwrite_existing,
    )
    print(str(out))

if __name__ == "__main__":
    main()
