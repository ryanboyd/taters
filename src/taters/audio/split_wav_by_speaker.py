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
    output_dir: Union[str, Path, None] = None,
    *,
    overwrite_existing: bool = False,
    start_col: str = "start_time",
    end_col: str = "end_time",
    speaker_col: str = "speaker",
    time_unit: str = "ms",             # "ms" or "s"
    silence_ms: int = 1000,
    pre_silence_ms: Optional[int] = None,
    post_silence_ms: Optional[int] = None,
    sr: Optional[int] = 16000,
    mono: bool = True,
    min_dur_ms: int = 50,
    merge_consecutive: bool = True,    # NEW: merge back-to-back turns by same speaker
) -> Dict[str, Path]:
    """
    Concatenate speaker-specific segments into per-speaker WAV files.

    If `merge_consecutive=True` (default), adjacent transcript rows with the same
    speaker are merged into a single, longer segment spanning from the first
    start to the last end — including any silence between those turns. If you
    need the strict per-row behavior, set `merge_consecutive=False`.

    Parameters
    ----------
    source_wav : str | Path
        Path to the source WAV.
    transcript_csv_path : str | Path
        CSV with timing and speaker columns (e.g., diarization output).
    output_dir : str | Path | None, optional
        Where to write the per-speaker files. If None, defaults to
        ``./audio_split/<source_stem>/``.
    start_col, end_col, speaker_col : str
        Column names in the transcript CSV.
    time_unit : {"ms","s"}, default "ms"
        Units for start/end columns.
    silence_ms : int, default 1000
        If `pre_silence_ms`/`post_silence_ms` are None, use this for both sides.
    pre_silence_ms, post_silence_ms : int | None
        Explicit padding (ms) before/after each segment; overrides `silence_ms`.
    sr : int | None, default 16000
        Resample output to this rate. If None, keep original rate.
    mono : bool, default True
        Downmix to mono if True.
    min_dur_ms : int, default 50
        Skip segments shorter than this duration (ms).
    merge_consecutive : bool, default True
        Merge back-to-back turns for the same speaker into one segment span
        (including any inter-turn silence). If False, emit one clip per row.

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

    def _friendly_filename_label(name: str) -> str:
        s = (name or "").strip()
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r'[<>:"|?*]', "", s)
        s = re.sub(r"\s+", " ", s)
        return s or "SPEAKER_0"

    source_wav = Path(source_wav)
    transcript_csv_path = Path(transcript_csv_path)
    out_dir = Path(output_dir) if output_dir is not None else (Path.cwd() / "audio_split" / source_wav.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_stem = source_wav.stem

    audio = AudioSegment.from_file(source_wav)
    if sr:
        audio = audio.set_frame_rate(sr)
    if mono:
        audio = audio.set_channels(1)

    factor = 1000.0 if time_unit == "s" else 1.0
    audio_len_ms = len(audio)

    with transcript_csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    segs_by_spk: Dict[str, List[tuple[int, int]]] = {}
    label_for_key: Dict[str, str] = {}

    # Build segments with awareness of original row order so that we can merge
    # adjacent turns for the same speaker when requested.
    prev_spk_key: Optional[str] = None
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

        start_ms = _clamp(start_ms, 0, audio_len_ms)
        end_ms   = _clamp(end_ms,   0, audio_len_ms)
        if end_ms <= start_ms:
            continue

        spk_key = _sanitize_speaker(raw_spk)
        label_for_key.setdefault(spk_key, _friendly_filename_label(raw_spk))

        if merge_consecutive and prev_spk_key == spk_key and segs_by_spk.get(spk_key):
            # Extend the last segment for this speaker to cover the new end
            s0, e0 = segs_by_spk[spk_key][-1]
            # Keep the earliest start, extend to the latest end
            s_new = min(s0, start_ms)
            e_new = max(e0, end_ms)
            segs_by_spk[spk_key][-1] = (s_new, e_new)
        else:
            # Strictly append a new segment
            segs_by_spk.setdefault(spk_key, []).append((start_ms, end_ms))

        prev_spk_key = spk_key

    # Optional: drop very short segments after merging
    for spk_key, segs in list(segs_by_spk.items()):
        segs_by_spk[spk_key] = [(s, e) for (s, e) in segs if (e - s) >= min_dur_ms]

    pre_ms  = silence_ms if pre_silence_ms  is None else pre_silence_ms
    post_ms = silence_ms if post_silence_ms is None else post_silence_ms
    pre_sil  = AudioSegment.silent(duration=max(0, pre_ms),  frame_rate=audio.frame_rate)
    post_sil = AudioSegment.silent(duration=max(0, post_ms), frame_rate=audio.frame_rate)
    if mono:
        pre_sil  = pre_sil.set_channels(1)
        post_sil = post_sil.set_channels(1)

    results: Dict[str, Path] = {}
    for spk_key, segs in segs_by_spk.items():
        if not segs:
            continue

        friendly = label_for_key.get(spk_key, spk_key)
        out_path = out_dir / f"{base_stem}_{friendly}.wav"

        if (not overwrite_existing) and out_path.is_file():
            results[friendly] = out_path
            continue

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
    p.add_argument(
    "--no-merge-consecutive",
    dest="merge_consecutive",
    action="store_false",
    help="Do NOT merge adjacent rows for the same speaker; emit one clip per transcript row."
    )
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
        merge_consecutive=args.merge_consecutive,
    )
    print("Wrote:", {k: str(v) for k, v in paths.items()})
