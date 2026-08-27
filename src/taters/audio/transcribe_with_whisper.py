"""Single-speaker transcription with faster-whisper.

This is the lightweight counterpart to
:func:`taters.audio.diarizer.whisper_diar_wrapper.run_whisper_diarization_repo`.
Both produce the same artifact — a timestamped utterance CSV with columns
``start_time,end_time,speaker,text`` in **milliseconds** — so anything
downstream (per-speaker WAVs, acoustics, Whisper embeddings, the text
analyzers) accepts either one without modification.

The difference is what they cost and what they can tell you:

============  ==========================  ==================================
              ``transcribe_with_whisper``  ``diarize_with_thirdparty``
============  ==========================  ==================================
Speakers      One (a fixed label)         Many, clustered automatically
Install       Base ``pip install taters`` ``[diarization]`` + three git installs
Runtime       faster-whisper only         Demucs, forced alignment,
                                          punctuation restoration, NeMo MSDD
Execution     In-process                  Subprocess against a vendored repo
============  ==========================  ==================================

Reach for this module when the recording has one voice — a lecture, an
interview recorded on a lapel mic, a voice memo, a podcast monologue — or
when you simply want a transcript and do not care who said what. Reach for
the diarizer when "who spoke when" is part of the question.

Unlike the embedding extractor, nothing here imports torch or transformers
(faster-whisper sits on CTranslate2), so there is no CUDA/Torch state to
collide with and no subprocess is needed to isolate it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

__all__ = ["TranscriptionOutputFiles", "transcribe_with_whisper", "main"]


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> str:
    """
    Resolve a device string to a concrete ``"cuda"`` or ``"cpu"``.

    ``"auto"`` (or ``None``) asks CTranslate2 how many CUDA devices it can see.
    We deliberately ask CTranslate2 rather than torch: faster-whisper does not
    depend on torch, and importing it here purely to answer this question would
    give the module a heavyweight dependency it does not otherwise need.
    """
    if device is None or str(device).strip().lower() in {"", "auto"}:
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"
    return str(device).strip().lower()


def _resolve_compute_type(compute_type: Optional[str], device: str) -> str:
    """
    Pick a sensible CTranslate2 compute type when the caller did not name one.

    ``float16`` is the right default on CUDA and is *not* supported on CPU,
    where CTranslate2 silently falls back to ``float32`` after a warning. Since
    this module is meant to work on a plain CPU-only install, we resolve to
    ``int8`` there instead — roughly 4x faster than float32 with negligible
    transcription differences.
    """
    if compute_type:
        return str(compute_type)
    return "float16" if device == "cuda" else "int8"


# Loading a Whisper model costs seconds and hundreds of MB. The pipeline runner
# fans item steps out across a ThreadPoolExecutor by default, so without a cache
# an 8-worker run would load the same model eight times. CTranslate2 releases
# the GIL during inference, so one shared model serves every thread.
_MODEL_CACHE: Dict[tuple, object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str, device: str, compute_type: str):
    """Return a (possibly cached) ``WhisperModel`` for this configuration."""
    key = (model_name, device, compute_type)
    # The lock is held across the load on purpose: concurrent first-callers
    # should queue behind one load rather than each building their own.
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:  # pragma: no cover - core dependency
                raise ImportError(
                    "faster-whisper is required for transcription. It ships with "
                    "the base install; reinstall with `pip install taters` if it "
                    "has gone missing."
                ) from e
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _MODEL_CACHE[key] = model
        return model


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionOutputFiles:
    """
    Where the transcription artifacts landed.

    Deliberately the same shape as
    :class:`~taters.audio.diarizer.whisper_diar_wrapper.DiarizationOutputFiles`
    so pipeline steps can pick the CSV out of either with the identical
    ``{{pick:<step>.raw_files.csv}}`` expression.

    Attributes
    ----------
    work_dir : pathlib.Path
        Per-file directory holding the artifacts (``<out_dir>/<stem>/``).
    raw_files : dict[str, pathlib.Path]
        Written outputs keyed by extension: ``"csv"``, and ``"srt"``/``"txt"``
        when those were requested.
    language : str | None
        Language Whisper detected (or the one that was forced), if known.
    duration : float | None
        Audio duration in seconds, as reported by Whisper.
    """
    work_dir: Path
    raw_files: Dict[str, Path] = field(default_factory=dict)
    language: Optional[str] = None
    duration: Optional[float] = None


def _write_utterance_csv(csv_path: Path, rows: List[dict], speaker_label: str) -> Path:
    """
    Write the ``start_time,end_time,speaker,text`` CSV, in milliseconds.

    Milliseconds are not an arbitrary choice: the vendored diarizer emits ms
    (its helpers multiply Whisper's seconds by 1000), and every downstream
    consumer defaults to ``time_unit="ms"``. Writing seconds here would give
    the two producers incompatible outputs that both parse without complaint.
    """
    import csv as _csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["start_time", "end_time", "speaker", "text"])
        for r in rows:
            w.writerow([
                f"{r['start_ms']:.3f}",
                f"{r['end_ms']:.3f}",
                speaker_label,
                r["text"],
            ])
    return csv_path


def _write_srt(srt_path: Path, rows: List[dict]) -> Path:
    """Write an SRT, reusing the renderer that already backs `convert_subtitles`."""
    from ..text.subtitle_parser import SubtitleSegment, render_to_srt

    segs = [
        SubtitleSegment(
            number=i,
            start_ms=int(round(r["start_ms"])),
            end_ms=int(round(r["end_ms"])),
            text=r["text"].strip(),
        )
        for i, r in enumerate(rows, start=1)
    ]
    return render_to_srt(segs, srt_path)


def _write_txt(txt_path: Path, rows: List[dict], speaker_label: str) -> Path:
    """
    Write the flat transcript.

    Formatted the way the diarizer's speaker-aware writer would render a
    single-speaker recording: one labelled block holding the full text.
    """
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    body = " ".join(r["text"].strip() for r in rows if r["text"].strip())
    with txt_path.open("w", encoding="utf-8-sig") as f:
        f.write(f"{speaker_label}: {body}\n")
    return txt_path


def _collect_existing(work_dir: Path, stem: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in ("csv", "srt", "txt"):
        p = work_dir / f"{stem}.{ext}"
        if p.exists():
            out[ext] = p
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transcribe_with_whisper(
    audio_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default
    whisper_model: str = "base.en",
    language: Optional[str] = None,
    device: Optional[str] = "auto",
    compute_type: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    initial_prompt: Optional[str] = None,
    speaker_label: str = "Speaker 0",
    write_srt: bool = True,
    write_txt: bool = True,
    verbose: bool = True,
) -> TranscriptionOutputFiles:
    """
    Transcribe an audio file with faster-whisper, treating it as one speaker.

    Produces the same ``start_time,end_time,speaker,text`` CSV (in milliseconds)
    that the diarizer produces, so the result is a drop-in substitute anywhere a
    transcript is consumed. Every row carries the same `speaker_label`, because
    no speaker clustering is performed — if you need to know who spoke when, use
    :func:`taters.audio.diarize_with_thirdparty` instead.

    Parameters
    ----------
    audio_path : str | Path
        Input audio. Anything faster-whisper can decode works; a 16 kHz mono WAV
        (what :func:`taters.audio.convert_to_wav` produces) is the safe choice.
    out_dir : str | Path | None, optional
        Base output directory. Artifacts land in ``<out_dir>/<stem>/``, matching
        the diarizer's layout. Defaults to ``./transcripts``.
    overwrite_existing : bool, default False
        If False and the CSV already exists, return the existing artifacts
        without re-running the model.
    whisper_model : str, default "base.en"
        faster-whisper model name (``tiny``, ``base``, ``small``, ``medium``,
        ``large-v3``, their ``.en`` variants) or a local CTranslate2 directory.
    language : str | None, optional
        Force a language code (e.g. ``"en"``). None auto-detects. Ignored in
        practice by ``.en`` models, which are English-only.
    device : {"auto","cuda","cpu"} | None, default "auto"
        Where to run. "auto" picks CUDA when CTranslate2 reports a usable device.
    compute_type : str | None, optional
        CTranslate2 compute type. None resolves to ``float16`` on CUDA and
        ``int8`` on CPU. Pass explicitly (e.g. ``"float32"``) to override.
    beam_size : int, default 5
        Decoder beam width. 1 is greedy and noticeably faster; 5 is the
        faster-whisper default and generally more accurate.
    vad_filter : bool, default True
        Run Silero VAD first and skip silent regions. Usually improves both
        speed and quality on long recordings, and avoids the well-known Whisper
        habit of hallucinating text during silence.
    initial_prompt : str | None, optional
        Optional context string to bias decoding — useful for seeding proper
        nouns, jargon, or spellings the model would otherwise mangle.
    speaker_label : str, default "Speaker 0"
        Value written to the ``speaker`` column of every row. Matches the
        diarizer's naming convention so grouped features line up.
    write_srt : bool, default True
        Also write ``<stem>.srt``.
    write_txt : bool, default True
        Also write ``<stem>.txt``.
    verbose : bool, default True
        Print progress as segments are decoded. Transcription is streamed, so
        this is the only feedback on a long file.

    Returns
    -------
    TranscriptionOutputFiles
        Work directory, written artifact paths, and the detected language and
        duration.

    Raises
    ------
    FileNotFoundError
        If `audio_path` does not exist.

    Examples
    --------
    >>> outs = transcribe_with_whisper("audio/lecture.wav", whisper_model="small.en")
    >>> outs.raw_files["csv"]
    PosixPath('.../transcripts/lecture/lecture.csv')

    See Also
    --------
    taters.audio.diarizer.whisper_diar_wrapper.run_whisper_diarization_repo :
        Multi-speaker alternative producing the same CSV schema.
    taters.audio.extract_whisper_embeddings :
        Turn the resulting transcript into per-segment encoder embeddings.
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = Path(out_dir).resolve() if out_dir is not None else (Path.cwd() / "transcripts")
    work_dir = out_dir / audio_path.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    csv_path = work_dir / f"{audio_path.stem}.csv"
    if not overwrite_existing and csv_path.is_file():
        if verbose:
            print("Transcript output file already exists; returning existing file.")
        return TranscriptionOutputFiles(
            work_dir=work_dir,
            raw_files=_collect_existing(work_dir, audio_path.stem),
        )

    resolved_device = _resolve_device(device)
    resolved_compute = _resolve_compute_type(compute_type, resolved_device)
    if verbose:
        print(
            f"Transcribing with faster-whisper "
            f"(model={whisper_model}, device={resolved_device}, compute_type={resolved_compute})"
        )

    model = _get_model(whisper_model, resolved_device, resolved_compute)

    # `transcribe` returns a lazy generator; the work happens as we iterate.
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    # Whisper's segment end times can overrun the actual audio — a 15.0 s clip
    # routinely reports a final segment ending at 16.9 s. Downstream consumers
    # (`split_wav_by_speaker`, `analyze_vocal_acoustics`) slice the WAV by these
    # numbers, so an out-of-range end silently yields a truncated or empty
    # segment. Clamp to the known duration rather than passing the overrun on.
    limit_ms = total * 1000.0 if total > 0 else None

    rows: List[dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start_ms = float(seg.start) * 1000.0
        end_ms = float(seg.end) * 1000.0
        if limit_ms is not None:
            if start_ms >= limit_ms:
                continue
            end_ms = min(end_ms, limit_ms)
        if end_ms <= start_ms:
            continue
        rows.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
        if verbose:
            pct = f" ({min(100.0, 100.0 * seg.end / total):5.1f}%)" if total > 0 else ""
            print(f"[transcribe:{audio_path.stem}]{pct} {seg.end:8.2f}s  {text}")

    raw_files: Dict[str, Path] = {"csv": _write_utterance_csv(csv_path, rows, speaker_label)}
    if write_srt:
        raw_files["srt"] = _write_srt(work_dir / f"{audio_path.stem}.srt", rows)
    if write_txt:
        raw_files["txt"] = _write_txt(work_dir / f"{audio_path.stem}.txt", rows, speaker_label)

    if verbose:
        print(f"Transcript CSV written to: {csv_path}  ({len(rows)} segments)")

    return TranscriptionOutputFiles(
        work_dir=work_dir,
        raw_files=raw_files,
        language=getattr(info, "language", None),
        duration=total or None,
    )


# ---------------------------------------------------------------------------
# CLI: python -m taters.audio.transcribe_with_whisper
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    from ..helpers.cliargs import add_bool_argument

    p = argparse.ArgumentParser(
        description="Taters: transcribe a single-speaker recording with faster-whisper."
    )
    p.add_argument("--audio_path", required=True, help="Path to input audio (e.g., WAV)")
    p.add_argument("--out_dir", default=None,
                   help="Base output directory (artifacts land in <out_dir>/<stem>/). "
                        "Default: ./transcripts under the current working dir")

    add_bool_argument(p, "--overwrite_existing", default=False,
                      help="Do you want to overwrite the output file if it already exists?")

    p.add_argument("--whisper_model", default="base.en",
                   help="Faster-Whisper model name (e.g., tiny, base.en, small, large-v3)")
    p.add_argument("--language", default=None, help="Force language (e.g., en). Omit to auto-detect")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                   help='Device: "auto" (default) picks CUDA if available else CPU')
    p.add_argument("--compute_type", default=None,
                   help="CTranslate2 compute type. Default: float16 on CUDA, int8 on CPU")
    p.add_argument("--beam_size", type=int, default=5, help="Decoder beam width (1 = greedy)")
    add_bool_argument(p, "--vad_filter", default=True,
                      help="Skip silence using Silero VAD before decoding")
    p.add_argument("--initial_prompt", default=None,
                   help="Optional context string to bias decoding (names, jargon, spellings)")
    p.add_argument("--speaker_label", default="Speaker 0",
                   help="Value written to the speaker column of every row")

    add_bool_argument(p, "--write_srt", default=True, help="Also write <stem>.srt")
    add_bool_argument(p, "--write_txt", default=True, help="Also write <stem>.txt")

    p.add_argument("--quiet", dest="verbose", action="store_false", default=True,
                   help="Do not print per-segment progress")
    return p


def main():
    args = _build_arg_parser().parse_args()

    outs = transcribe_with_whisper(
        audio_path=args.audio_path,
        out_dir=args.out_dir,
        overwrite_existing=args.overwrite_existing,
        whisper_model=args.whisper_model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        initial_prompt=args.initial_prompt,
        speaker_label=args.speaker_label,
        write_srt=args.write_srt,
        write_txt=args.write_txt,
        verbose=args.verbose,
    )

    print(f"Work dir: {outs.work_dir}")
    for k, v in outs.raw_files.items():
        print(f"{k.upper()}: {v}")


if __name__ == "__main__":
    main()
