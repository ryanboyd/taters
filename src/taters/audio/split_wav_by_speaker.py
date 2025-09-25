# taters/audio/speaker_split.py

from __future__ import annotations
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydub import AudioSegment


def _sanitize_speaker(name: str) -> str:
    name = (name or "SPEAKER_0").strip()
    name = name.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def make_speaker_wavs_from_csv(
    source_wav: Union[str, Path],
    transcript_csv_path: Union[str, Path],
    output_dir: Union[str, Path, None] = None,   # <- renamed + optional
    *,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default
    start_col: str = "start_time",
    end_col: str = "end_time",
    speaker_col: str = "speaker",
    time_unit: str = "ms",             # "ms" (default) or "s"
    silence_ms: int = 1000,            # silence length before/after each clip, in ms
    pre_silence_ms: Optional[int] = None,  # optional override for pre-silence
    post_silence_ms: Optional[int] = None, # optional override for post-silence
    sr: Optional[int] = 16000,         # resample; None = keep original rate
    mono: bool = True,                 # force mono output
    min_dur_ms: int = 50,              # skip ultra-short blips
) -> Dict[str, Path]:
    """
    Concatenate speaker-specific segments into per-speaker WAV files.

    The function reads a timestamped transcript, extracts the corresponding
    audio regions from `source_wav`, and concatenates them into one output WAV
    per unique speaker label. Optional silence can be inserted before/after each
    segment to avoid clicks or tight joins.

    Parameters
    ----------
    source_wav : str | Path
        Path to the source WAV.
    transcript_csv_path : str | Path
        CSV with timing and speaker columns (e.g., diarization output).
    output_dir : str | Path | None, optional
        Where to write the per-speaker files. If None, defaults to
        ``<cwd>/audio/speakers``.
    start_col, end_col, speaker_col : str
        Names of the columns in `transcript_csv_path` for segment start, end,
        and speaker ID/name.
    time_unit : {"ms","s"}, default "ms"
        Units for the start/end columns.
    silence_ms : int, default 1000
        If `pre_silence_ms`/`post_silence_ms` are None, this value is used for
        both sides of each segment. Set to 0 to disable padding.
    pre_silence_ms, post_silence_ms : int | None
        Explicit padding (ms) before/after each segment. Overrides `silence_ms`.
    sr : int | None, default 16000
        Resample output to this rate. If None, keep original rate.
    mono : bool, default True
        If True, downmix to mono.
    min_dur_ms : int, default 50
        Skip segments shorter than this duration (ms).

    Returns
    -------
    dict[str, Path]
        Mapping from friendly speaker label → output WAV path.

    Behavior
    --------
    - Input speaker labels are sanitized for filenames but a more readable label
      (without path-hostile characters) is preserved for naming.
    - Segments are sorted by start time per speaker before concatenation.
    - If a speaker ends up with zero valid segments, no file is written.

    Examples
    --------
    >>> make_speaker_wavs_from_csv(
    ...     source_wav="audio/session.wav",
    ...     transcript_csv_path="transcripts/session.csv",
    ...     time_unit="ms",
    ...     silence_ms=0,  # no padding
    ...     sr=16000,
    ...     mono=True,
    ... )
    """

    if time_unit not in ("ms", "s"):
        raise ValueError("time_unit must be 'ms' or 's'")

    # --- small helper for filename-friendly labels (keeps spaces) ---
    def _friendly_filename_label(name: str) -> str:
        s = (name or "").strip()
        s = s.replace("/", "_").replace("\\", "_")     # forbid path separators
        s = re.sub(r'[<>:"|?*]', "", s)               # trim problematic chars
        s = re.sub(r"\s+", " ", s)                    # collapse whitespace
        return s or "SPEAKER_0"

    # Resolve paths
    source_wav = Path(source_wav)
    transcript_csv_path = Path(transcript_csv_path)

    # Default predictable location if none provided
    out_dir = Path(output_dir) if output_dir is not None else (Path.cwd() / "audio_split" / source_wav.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = source_wav.stem

    # Load audio once
    audio = AudioSegment.from_file(source_wav)
    if sr:
        audio = audio.set_frame_rate(sr)
    if mono:
        audio = audio.set_channels(1)

    # Factor to convert CSV time to milliseconds
    factor = 1000.0 if time_unit == "s" else 1.0
    audio_len_ms = len(audio)

    # Read CSV
    with transcript_csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Collect segments by (sanitized) speaker key and remember friendly label
    segs_by_spk: Dict[str, List[tuple[int, int]]] = {}
    label_for_key: Dict[str, str] = {}

    for row in rows:
        try:
            start_raw = float(row[start_col])
            end_raw   = float(row[end_col])
            raw_spk   = str(row.get(speaker_col, "SPEAKER_0"))
        except Exception:
            continue

        start_ms = int(round(start_raw * factor))
        end_ms   = int(round(end_raw   * factor))
        if end_ms <= start_ms:
            continue
        if (end_ms - start_ms) < min_dur_ms:
            continue

        start_ms = _clamp(start_ms, 0, audio_len_ms)
        end_ms   = _clamp(end_ms,   0, audio_len_ms)
        if end_ms <= start_ms:
            continue

        spk_key = _sanitize_speaker(raw_spk)  # stable dict key
        segs_by_spk.setdefault(spk_key, []).append((start_ms, end_ms))
        label_for_key.setdefault(spk_key, _friendly_filename_label(raw_spk))

    # check to see if files already exist
    friendly = label_for_key.get(spk_key, spk_key)
    out_path = out_dir / f"{base_stem}_{friendly}.wav"

    if not overwrite_existing and Path(out_path).is_file():
        print("WAV file already exists; returning existing file.")
        return out_path

    # Sort by start time per speaker
    for spk_key in segs_by_spk:
        segs_by_spk[spk_key].sort(key=lambda t: (t[0], t[1]))

    # Silence chunks
    pre_ms  = silence_ms if pre_silence_ms  is None else pre_silence_ms
    post_ms = silence_ms if post_silence_ms is None else post_silence_ms
    pre_sil  = AudioSegment.silent(duration=max(0, pre_ms),  frame_rate=audio.frame_rate)
    post_sil = AudioSegment.silent(duration=max(0, post_ms), frame_rate=audio.frame_rate)
    if mono:
        pre_sil  = pre_sil.set_channels(1)
        post_sil = post_sil.set_channels(1)

    # Build one file per speaker; name = <source_stem>_<Friendly Label>.wav
    results: Dict[str, Path] = {}
    for spk_key, segs in segs_by_spk.items():
        out = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
        if mono:
            out = out.set_channels(1)

        for (s, e) in segs:
            clip = audio[s:e]
            if len(clip) < min_dur_ms:
                continue
            out += pre_sil + clip + post_sil

        if len(out) == 0:
            continue

        out.export(out_path, format="wav", codec="pcm_s16le")
        results[friendly] = out_path

    return results


# Optional CLI for ad-hoc use
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Create per-speaker WAVs from a timestamped CSV transcript.")
    p.add_argument("--wav", required=True, help="Source audio (wav)")
    p.add_argument("--transcript_csv_path", required=True, help="Transcript CSV (start_time,end_time,speaker,text)")
    # optional output dir; default to ./audio_split/<source_stem>/
    p.add_argument("--output_dir", required=False, default=None,
                   help="Output dir for per-speaker wavs (default: ./audio_split/<source_stem>/)")
    p.add_argument("--unit", choices=["ms", "s"], default="ms", help="Timestamp unit in CSV (default: ms)")
    p.add_argument("--sr", type=int, default=16000, help="Output sample rate (Hz)")
    p.add_argument("--silence-ms", type=int, default=1000, help="Silence before/after each clip in ms")
    p.add_argument("--pre-silence-ms", type=int, default=None, help="Override pre-silence (ms)")
    p.add_argument("--post-silence-ms", type=int, default=None, help="Override post-silence (ms)")
    p.add_argument("--overwrite_existing", action="store_true", help="Overwrite existing output")
    args = p.parse_args()

    paths = make_speaker_wavs_from_csv(
        source_wav=args.wav,
        transcript_csv_path=args.transcript_csv_path,
        output_dir=args.output_dir,   # <- new name + optional
        time_unit=args.unit,
        sr=args.sr,
        silence_ms=args.silence_ms,
        pre_silence_ms=args.pre_silence_ms,
        post_silence_ms=args.post_silence_ms,
        overwrite_existing=args.overwrite_existing,
    )
    print("Wrote:", {k: str(v) for k, v in paths.items()})
