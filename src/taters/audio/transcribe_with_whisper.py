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
from typing import Callable, Dict, List, Optional, Union

from ..helpers.gpu import (CPU_FALLBACK_NOTE, ensure_cuda_libraries, resolve_device)
from ..helpers.progress import announce
from ..helpers.cliargs import CliSpec

__all__ = ["TranscriptionOutputFiles", "transcribe_with_whisper", "main"]


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------

def _resolve_device(device: Optional[str]) -> str:
    """Resolve a device string to a concrete ``"cuda"`` or ``"cpu"``."""
    return resolve_device(device, backend="ctranslate2")[0]


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


# loading a Whisper model costs us seconds and hundreds of MB. the pipeline
# runner fans item steps out across a ThreadPoolExecutor by default, so without
# a cache an 8-worker run would load the same model eight times over.
# CTranslate2 releases the GIL during inference, so one shared model can serve
# every thread.
_MODEL_CACHE: Dict[tuple, object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _prove_it_works(model) -> None:
    """
    Run one encode, so a broken GPU install fails here instead of everywhere.

    CTranslate2 loads its CUDA libraries lazily, at the first encode -- not when
    the model is constructed. So a machine where cuBLAS cannot be loaded builds
    a ``WhisperModel`` quite happily and only raises deep inside the segment
    generator, minutes later, once per file.

    That timing is what makes it worth paying a few milliseconds to avoid. The
    pipeline hands one cached model to every worker, so by the time the first
    encode fails the other workers have already queued work on the same
    CTranslate2 replica -- and a replica whose job raised does not return
    results to the jobs behind it. The observed shape is one file failing after
    two seconds and the rest sitting at zero CPU forever, which reads as a hang
    with an unrelated error in it rather than as one cause.

    A second of silence is enough: it is one 30-second window, one encode, and
    every library the real work needs.
    """
    import numpy as np

    segments, _info = model.transcribe(
        np.zeros(16000, dtype=np.float32),
        language="en", beam_size=1, vad_filter=False,
    )
    # the generator is where the work actually happens, so we have to pull on it
    for _ in segments:
        break


def english_only(whisper_model: str) -> bool:
    """
    Whether a model name is one of Whisper's English-only variants.

    Those are the ``.en`` names (``base.en``, ``small.en``). A local model
    directory is taken at its word: a folder called ``my-model.en`` is
    presumably one too, and a folder without the suffix is assumed
    multilingual, because there is no way to ask without loading it.
    """
    return str(whisper_model).rstrip("/\\").endswith(".en")


def _get_model(model_name: str, device: str, compute_type: str) -> tuple:
    """
    Return a (possibly cached) working ``WhisperModel``, and what it runs on.

    Falls back to the CPU when the GPU turns out not to work, rather than
    failing the run: a transcription that takes longer than you hoped is worth
    much more than a stack trace naming a DLL. The fallback is decided once,
    here, under the cache lock -- so every worker gets the same answer and none
    of them queues work on a model that cannot serve it.

    Returns
    -------
    tuple
        ``(model, device, compute_type)`` -- the last two being what was
        actually used, which is not always what was asked for.
    """
    key = (model_name, device, compute_type)
    # we hold the lock across the whole load on purpose: if several callers show
    # up at once, they should queue behind one load rather than each build their
    # own.
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - core dependency
            raise ImportError(
                "faster-whisper is required for transcription. It ships with "
                "the base install; reinstall with `pip install taters` if it "
                "has gone missing."
            ) from e

        if device == "cuda":
            # the math libraries are usually there, just unreachable -- torch
            # bundles them, and nothing puts torch's lib directory on the search
            # path. pointing the loader at it costs us nothing, and it's the
            # difference between a GPU run and a CPU one.
            ensure_cuda_libraries()

        try:
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _prove_it_works(model)
        except Exception as e:
            if device != "cuda":
                raise
            # we say this loudly, every time. it's the difference between a
            # one-minute run and an hour-long one, and a run that quietly takes
            # the slow road is how somebody ends up staring at a progress bar
            # wondering whether it crashed.
            print(f"[transcribe] The GPU is visible but CTranslate2 cannot "
                  f"use it: {e}")
            print(f"[transcribe] {CPU_FALLBACK_NOTE}")
            print("[transcribe] Falling back to the CPU.")
            device = "cpu"
            compute_type = _resolve_compute_type(None, device)
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _prove_it_works(model)

        resolved = (model, device, compute_type)
        # we cache under both keys -- what was asked for and what we actually
        # delivered -- so a later caller asking for CUDA gets the working CPU
        # model instead of repeating the failed probe.
        _MODEL_CACHE[key] = resolved
        _MODEL_CACHE[(model_name, device, compute_type)] = resolved
        return resolved


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
    device : str | None
        The device transcription actually ran on -- ``"cuda"`` or ``"cpu"``,
        never ``"auto"``. Recorded because ``device="auto"`` resolving to the
        CPU is the difference between a one-minute run and an hour-long one,
        and it is otherwise invisible after the fact. ``None`` when the CSV
        already existed and nothing was run.
    compute_type : str | None
        The CTranslate2 compute type used, for the same reason.
    """
    work_dir: Path
    raw_files: Dict[str, Path] = field(default_factory=dict)
    language: Optional[str] = None
    duration: Optional[float] = None
    device: Optional[str] = None
    compute_type: Optional[str] = None


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
    single-speaker recording: one labeled block holding the full text.
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
    translate: bool = False,
    device: Optional[str] = "auto",
    compute_type: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    initial_prompt: Optional[str] = None,
    speaker_label: str = "Speaker 0",
    write_srt: bool = True,
    write_txt: bool = True,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
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
    whisper_model : {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2", "large-v3", "large-v3-turbo", "distill-large-v3"} or str, default "base.en"
        A faster-whisper model name -- the ``.en`` variants are English-only
        and a little better at it -- or the path of a local CTranslate2 model
        directory.
    language : str | None, optional
        Force a language code (e.g. ``"en"``). None auto-detects. Ignored in
        practice by ``.en`` models, which are English-only. With ``translate``
        it names the language being *spoken*; the output is English either way.
    translate : bool, default False
        Write the transcript in English whatever language is spoken. Whisper's
        multilingual models were trained to do this as a second task, so it
        costs nothing extra -- but only those models can: an English-only
        ``.en`` model has no other language to translate from, and asking one
        to is refused before anything is loaded. The timestamps still belong
        to the spoken audio, so the transcript is a translation of what was
        said when, not a transcript of what was said.
    device : {"auto","cuda","cpu"} | None, default "auto"
        Where to run. "auto" picks CUDA when CTranslate2 reports a usable device.
    compute_type : {"int8", "int8_float16", "int8_bfloat16", "int16", "float16", "bfloat16", "float32"} | None, optional
        CTranslate2 compute type. None resolves to ``float16`` on CUDA and
        ``int8`` on CPU. Pass explicitly (e.g. ``"float32"``) to override.
    beam_size : int, default 5
        Decoder beam width. 1 is greedy and noticeably faster; 5 is the
        faster-whisper default and generally more accurate.
    vad_filter : bool, default True
        Run Silero VAD first and skip silent regions. Usually improves both
        speed and quality on long recordings, and avoids the well-known Whisper
        habit of hallucinating text during silence.
    word_timestamps : bool, default True
        Align the decoded text to the audio and use the result for the segment
        boundaries. On by default because without it the boundaries are close to
        fiction: Whisper's decoder ends each segment where the next one starts,
        so a transcript reports near-continuous speech whatever the recording
        actually contains. Measured on a 771-second lecture, 126 of 133 segment
        gaps were exactly zero and the transcript claimed 750 seconds of speech;
        with alignment the same file yields 138 segments and 713 seconds, and
        the words underneath account for 681.

        That matters beyond tidiness. `split_wav_by_speaker` cuts the WAV on
        these times and `analyze_vocal_acoustics` measures the pieces -- pause
        features included -- so boundaries that overshoot mean acoustics
        measured over silence the segmentation invented.

        The cost is the alignment pass: roughly 20% slower (13.7s to 16.7s on
        that file). Turn it off when throughput matters more than knowing when
        anyone was actually speaking.
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
    on_progress : callable, optional
        Structured progress sink, ``on_progress(done, total, message)``, with
        both figures in **seconds of audio**. Injected automatically by the
        pipeline runner. This is the machine-readable counterpart to `verbose`:
        a UI owning the screen cannot let a step print into it, but a
        transcription is the longest thing in most pipelines and needs to be
        seen moving.

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
    if translate and english_only(whisper_model):
        # we refuse here, in plain words, rather than leave it to the decoder:
        # an English-only model has no translate token, and what comes back is
        # either an obscure tokenizer error or an English transcript of foreign
        # speech -- in other words, nonsense that looks like output.
        raise ValueError(
            f"translate=True needs a multilingual Whisper model, and "
            f"{whisper_model!r} is English-only. Use the same size without "
            f".en -- {whisper_model[:-3]!r} -- or any of tiny, base, small, "
            f"medium, large-v3.")

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

    resolved_device, fallback_reason = resolve_device(device, backend="ctranslate2")
    resolved_compute = _resolve_compute_type(compute_type, resolved_device)
    if verbose:
        print(
            f"Transcribing with faster-whisper "
            f"(model={whisper_model}, device={resolved_device}, compute_type={resolved_compute})"
        )
        if fallback_reason:
            print(f"[transcribe] {fallback_reason}")

    # we announce this before touching the model, not after. loading is where a
    # first run downloads the weights and where a broken CUDA install hangs or
    # stalls, and until this line there's nothing on screen to say which of
    # those is happening -- or that we picked the CPU.
    announce(on_progress,
             f"loading {whisper_model} on {resolved_device} ({resolved_compute})")
    model, resolved_device, resolved_compute = _get_model(
        whisper_model, resolved_device, resolved_compute)
    # and we say it again, because it may have changed: `_get_model` proves the
    # GPU works before handing it over and quietly moves to the CPU when it
    # doesn't, and a row still reading "cuda" after that is the wrong answer to
    # the only question anyone ever asks about a slow transcription.
    announce(on_progress,
             f"loaded {whisper_model} on {resolved_device} ({resolved_compute})")

    # VAD runs over the whole file inside `transcribe()`, before we get a single
    # segment back, so from the outside this phase is dead silent.
    announce(on_progress, "finding speech" if vad_filter else "starting")

    # `transcribe` hands back a lazy generator; the work happens as we iterate.
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        task="translate" if translate else "transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    # Whisper's segment end times can overrun the actual audio -- a 15.0 s clip
    # routinely reports a final segment ending at 16.9 s. downstream consumers
    # (`split_wav_by_speaker`, `analyze_vocal_acoustics`) slice the WAV by these
    # numbers, so an out-of-range end quietly gives us a truncated or empty
    # segment. so we clamp to the known duration rather than pass the overrun on.
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
        # we report position in the recording, not segment count: we know the
        # denominator from the very start (Whisper reports the duration up
        # front), whereas we only know the number of segments once decoding is
        # done.
        if on_progress is not None:
            on_progress(int(min(seg.end, total) if total > 0 else seg.end),
                        int(total) if total > 0 else None,
                        None,
                        "seconds")
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
        device=resolved_device,
        compute_type=resolved_compute,
    )


# ---------------------------------------------------------------------------
# CLI: python -m taters.audio.transcribe_with_whisper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Command line -- derived from the function(s) above; see helpers.cliargs.CliSpec.
# The aliases and legacy flags are the spellings the hand-written parser used,
# kept so every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    transcribe_with_whisper,
    description='Transcribe a single-speaker recording with faster-whisper.',
    aliases={},
    legacy={
        '--quiet': ['--verbose', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
