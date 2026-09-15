# -*- coding: utf-8 -*-

"""Subprocess worker that computes Whisper encoder embeddings.

This module is meant to be executed with ``python -m ...`` by the wrapper in
``extract_whisper_embeddings.py``. It avoids importing heavyweight torch
packages in the parent process and keeps CUDA state isolated.

Two entry functions implement I/O and shape-handling:

- :func:`export_segment_embeddings_csv` — transcript-driven, one vector per row.
- :func:`export_audio_embeddings_csv` — general WAVs; segmentation + optional pooling.

Both functions use `faster-whisper` (CTranslate2 backend) and `WhisperFeatureExtractor`
to produce encoder features, then pool the encoder outputs into fixed-length vectors.
"""


from __future__ import annotations

# keep Transformers from going anywhere near torch/TF/Flax in this module
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, List, Tuple, Literal

import numpy as np
import librosa
from ctranslate2 import StorageView
import ctranslate2
from faster_whisper import WhisperModel
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from ..helpers.gpu import add_device_argument, resolve_device


# ----------------------------
# Helpers
# ----------------------------

def _hf_repo_for(model_name: str) -> str:
    name = model_name.lower()
    table = {
        "tiny": "openai/whisper-tiny",
        "tiny.en": "openai/whisper-tiny.en",
        "base": "openai/whisper-base",
        "base.en": "openai/whisper-base.en",
        "small": "openai/whisper-small",
        "small.en": "openai/whisper-small.en",
        "medium": "openai/whisper-medium",
        "medium.en": "openai/whisper-medium.en",
        "large": "openai/whisper-large",
        "large-v1": "openai/whisper-large-v1",
        "large-v2": "openai/whisper-large-v2",
        "large-v3": "openai/whisper-large-v3",
        "distil-large-v2": "openai/whisper-large-v2",
        "distil-medium.en": "openai/whisper-medium.en",
    }
    return table.get(name, f"openai/whisper-{name}")

def _storageview_to_numpy(x) -> np.ndarray:
    if hasattr(x, "to_array"):
        try:
            return x.to_array()
        except Exception:
            pass
    if hasattr(x, "to"):
        try:
            xc = x.to("cpu")
            if hasattr(xc, "to_array"):
                return xc.to_array()
        except Exception:
            pass
    return np.asarray(x)

def _sv_to_numpy_cpu(sv) -> Optional[np.ndarray]:
    try:
        if hasattr(sv, "to_array"):
            arr = sv.to_array()
        else:
            arr = np.asarray(sv)
        if arr is None:
            return None
        if arr.dtype == object and arr.ndim <= 1:
            return None
        return arr
    except Exception:
        return None



def _encode_features_any_layout(ct2_model, feats: np.ndarray) -> Optional[np.ndarray]:
    """
    Try encoding with CT2 Whisper for both common layouts:
      1) [B, 80, T]
      2) [B, T, 80]
    Return a pooled [D] vector if successful, else None.
    """
    f = np.ascontiguousarray(feats.astype("float32", copy=False))
    if f.ndim == 2:  # [80, T] -> [1, 80, T]
        f = f[None, ...]
    if f.ndim != 3:
        return None

    candidates = [
        f,                         # [B, 80, T]
        np.transpose(f, (0, 2, 1)) # [B, T, 80]
    ]

    for cand in candidates:
        try:
            # *** THE IMPORTANT BIT: force CPU output so to_array() works reliably ***
            sv = ct2_model.encode(StorageView.from_array(cand), to_cpu=True)
        except Exception:
            continue

        arr = _sv_to_numpy_cpu(sv)  # this now gives us an actual ndarray
        if arr is None:
            continue

        # pool down to [D] by averaging every axis but the last (which we assume
        # is the hidden size)
        if arr.ndim == 1:
            vec = arr
        else:
            try:
                D = int(arr.shape[-1])
                vec = arr.reshape(-1, D).mean(axis=0)
            except Exception:
                vec = None

        if vec is None or vec.ndim != 1:
            continue

        if 128 <= vec.shape[0] <= 2048:  # sanity range for Whisper hidden sizes
            return vec

    return None



def _pick(primary: str, fieldnames: List[str], *alts: str) -> Optional[str]:
    for k in (primary, *alts):
        if k in fieldnames:
            return k
    return None

def _resolve_columns(fieldnames: List[str],
                     start_col: str, end_col: str, speaker_col: str) -> Tuple[str,str,str]:
    sc = _pick(start_col,   fieldnames, "start", "from", "t0", "start_ms", "start_sec")
    ec = _pick(end_col,     fieldnames, "end",   "to",   "t1", "end_ms",   "end_sec")
    pc = _pick(speaker_col, fieldnames, "speaker_label", "spk", "speaker_id", "speaker_name")
    if not (sc and ec and pc):
        raise ValueError(
            f"Could not find required columns. Have {fieldnames}. "
            f"Tried start={start_col}/..., end={end_col}/..., speaker={speaker_col}/..."
        )
    return sc, ec, pc

def _guess_time_unit(max_end: float, dur_s: float, n_samples: int) -> str:
    """
    Heuristically guess time unit of transcript values:
      - 's' if values look like seconds
      - 'ms' if values look like milliseconds
      - 'samples' if values look like raw sample indices
    """
    # we give each guess 10% slack
    if max_end <= dur_s * 1.1:
        return "s"
    if max_end <= (dur_s * 1000.0) * 1.1:
        return "ms"
    if max_end <= n_samples * 1.1:
        return "samples"
    # if nothing fits, we just assume seconds and hope it's not wildly off
    return "s"

def l2_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n



@dataclass
class EmbedConfig:
    model_name: str = "base"        # faster-whisper model name or ct2 dir
    device: str = "auto"            # "cuda", "cpu", or "auto"
    compute_type: str = "float16"   # "float16" (GPU), "int8" (CPU), etc.
    time_unit: str = "auto"         # "auto" | "ms" | "s" | "samples"


def _load_whisper(config: "EmbedConfig"):
    """
    Load the encoder on a device that has been shown to work.

    The same trap as in `transcribe_with_whisper`, one process further out:
    CTranslate2 reports a usable CUDA device from the driver alone and loads
    cuBLAS only at the first encode, so a broken GPU install builds this model
    without complaint and fails minutes later. Here that failure took a whole
    subprocess with it, which the parent could only report as a dead child.

    Returns
    -------
    tuple
        ``(model, device, compute_type)`` -- the last two being what was
        actually used, which is not always what was asked for.
    """
    import numpy as _np

    asked = (config.device or "auto").strip().lower()

    def build(device: str, compute_type: str):
        model = WhisperModel(config.model_name, device=device, compute_type=compute_type)
        # one 30-second window of silence: a single encode, which drags in every
        # library the real work is going to need. milliseconds, against a run
        # that takes minutes.
        model.encode(_np.zeros((80, 3000), dtype=_np.float32))
        return model

    if asked == "cuda":
        # the user named cuda explicitly, so we treat it as an instruction: a
        # failure here is an error, not a quiet demotion to something slower.
        return build("cuda", config.compute_type), "cuda", config.compute_type

    if asked != "cpu":
        device, reason = resolve_device(asked, backend="ctranslate2")
        if device == "cuda":
            try:
                return build("cuda", config.compute_type), "cuda", config.compute_type
            except Exception as e:
                reason = f"the GPU is visible but CTranslate2 cannot use it: {e}"
        print(f"[whisper-embeddings] {reason}", flush=True)
        print("[whisper-embeddings] Falling back to the CPU.", flush=True)

    # int8 rather than the caller's float16: CPU CTranslate2 doesn't support
    # float16 and quietly downgrades it to float32 after a warning.
    compute = config.compute_type if config.compute_type not in ("", "float16") else "int8"
    return build("cpu", compute), "cpu", compute


# ----------------------------
# Core
# ----------------------------

def export_segment_embeddings_csv(
    transcript_csv: str | Path,
    source_wav: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    config: EmbedConfig = EmbedConfig(),
    start_col: str = "start_time",
    end_col: str = "end_time",
    speaker_col: str = "speaker",
    apply_l2_normalization: bool = False,
    sr: int = 16000,
) -> Path:
    """
    Compute Whisper encoder embeddings for each transcript segment and write a CSV.

    Expected transcript columns (auto-resolved with fallbacks):
    - start_time (or: start, from, t0, start_ms, start_sec)
    - end_time   (or: end, to, t1, end_ms, end_sec)
    - speaker    (optional; fallbacks include speaker_label, spk, speaker_id, ...)

    Parameters
    ----------
    transcript_csv : str | Path
        CSV with segment timings (and optionally speaker labels).
    source_wav : str | Path
        Audio file to slice. Will be resampled to `sr`.
    output_dir : str | Path | None, optional
        Directory for the output CSV. If None, defaults to the WAV's parent.
    config : EmbedConfig, keyword-only
        Configuration for model name, device, compute type, and time unit.
    start_col, end_col, speaker_col : str
        Column name hints. The function will fall back to common aliases if the
        exact names are not present.
    sr : int, default 16000
        Sample rate for feature extraction (audio is resampled as needed).

    Returns
    -------
    Path
        Path to the written CSV: ``<output_dir>/<wav_stem>_embeddings.csv``

    Behavior
    --------
    - Attempts to infer time units ("s", "ms", "samples") when config.time_unit == "auto".
    - Skips invalid or tiny segments (< 2 samples after rounding).
    - Pools encoder outputs to a fixed-length vector (mean over time).
    - Writes header even if no valid segments remain (empty payload).

    See Also
    --------
    export_audio_embeddings_csv : transcript-free embeddings.
    """

    transcript_csv = Path(transcript_csv)
    source_wav = Path(source_wav)

    # figure out where the output goes (next to the WAV unless told otherwise)
    if output_dir is None:
        output_dir = source_wav.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # and the final output path
    output_csv = output_dir / f"{source_wav.stem}_embeddings.csv"

    # 1) load the audio (mono, at sr)
    audio, in_sr = librosa.load(str(source_wav), sr=sr, mono=True)
    n_samples = len(audio)
    dur_s = n_samples / float(sr)

    # 2) load faster-whisper and the ct2 model
    fw, resolved_device, resolved_compute = _load_whisper(config)
    try:
        ct2_model: ctranslate2.models.Whisper = fw.model  # type: ignore[attr-defined]
    except AttributeError:
        model_dir = getattr(fw, "model_dir", None) or getattr(fw, "_model_dir", None)
        if not model_dir:
            raise RuntimeError(
                "Could not access the underlying CTranslate2 model from faster-whisper. "
                "Consider passing a local CTranslate2 model directory as model_name."
            )
        ct2_model = ctranslate2.models.Whisper(str(model_dir), device=resolved_device,
                                              compute_type=resolved_compute)

    # 3) the feature extractor
    fe = WhisperFeatureExtractor.from_pretrained(_hf_repo_for(config.model_name))

    # 4) read the transcript and work out what time unit it's in
    if not transcript_csv.exists():
        raise FileNotFoundError(f"Transcript CSV not found: {transcript_csv}")

    rows_out: list[list[Any]] = []
    embed_dim: Optional[int] = None

    # first pass: peek at the header and a few rows so we can guess the units
    with transcript_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        sc, ec, pc = _resolve_columns(fields, start_col, end_col, speaker_col)

        # up to 100 rows is plenty to find a reasonable max end time
        sample_vals: List[float] = []
        for i, row in enumerate(reader):
            try:
                sample_vals.append(float(row[ec]))
            except Exception:
                pass
            if i >= 99:
                break

        # we'll re-open the file for the real pass
    # now we settle on a unit
    if config.time_unit not in {"auto", "ms", "s", "samples"}:
        raise ValueError("config.time_unit must be 'auto', 'ms', 's', or 'samples'")

    guessed_unit = None
    if config.time_unit == "auto":
        max_end = max(sample_vals) if sample_vals else 0.0
        guessed_unit = _guess_time_unit(max_end, dur_s, n_samples)
        unit = guessed_unit
    else:
        unit = config.time_unit

    if _os.environ.get("TATERS_DEBUG") == "1":
        print(f"[emb] audio duration: {dur_s:.3f}s @ {sr}Hz (samples={n_samples})")
        if guessed_unit:
            print(f"[emb] time unit guessed -> {guessed_unit}")
        print(f"[emb] time unit in use -> {unit}")

    # conversion lambdas: to seconds, and to a sample index
    if unit == "s":
        to_sec = lambda x: float(x)
        to_idx = lambda t: int(round(float(t) * sr))
    elif unit == "ms":
        to_sec = lambda x: float(x) * 0.001
        to_idx = lambda t: int(round(float(t) * sr * 0.001))
    elif unit == "samples":
        to_sec = lambda x: float(x) / float(sr)
        to_idx = lambda t: int(round(float(t)))
    else:
        raise RuntimeError("Unexpected time unit.")

    # the real pass
    n_total = n_parsed = n_kept = 0
    n_oob = n_too_short = n_shape_skip = 0

    with transcript_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        sc, ec, pc = _resolve_columns(fields, start_col, end_col, speaker_col)

        for row in reader:
            n_total += 1
            try:
                t0_sec = to_sec(row[sc])
                t1_sec = to_sec(row[ec])
            except Exception:
                continue
            if not (t1_sec > t0_sec):
                continue
            n_parsed += 1

            s = max(0, min(n_samples, to_idx(row[sc])))
            e = max(0, min(n_samples, to_idx(row[ec])))
            if e <= s:
                n_oob += 1
                continue

            # slice it out; skip anything ultra tiny after rounding (< 2 samples)
            if e - s < 2:
                n_too_short += 1
                continue

            clip = audio[s:e]

            # build the input features (float32, no torch)
            feats = fe(clip, sampling_rate=sr, return_tensors="np")["input_features"]

            # encode with CT2, trying both layouts, then pool down to [D]
            vec = _encode_features_any_layout(ct2_model, feats)

            # --- purely for debugging: raw candidate shapes for the first few rows ---
            if _os.environ.get("TATERS_DEBUG") == "1" and n_parsed <= 3:
                try:
                    a = np.ascontiguousarray(feats.astype("float32", copy=False))
                    a1 = a if a.ndim == 3 else a[None, ...]
                    a2 = np.transpose(a1, (0, 2, 1))
                    print(f"[emb] feats shapes tried: {getattr(a1, 'shape', None)} and {getattr(a2, 'shape', None)}")
                except Exception:
                    pass
            # ----------------------------------------------------------------

            if vec is None:
                n_shape_skip += 1
                continue

            if apply_l2_normalization:
                vec = l2_normalize(vec)

            if embed_dim is None:
                embed_dim = int(vec.shape[-1])

            speaker = row.get(pc, "SPEAKER_0")
            rows_out.append([row[sc], row[ec], speaker] + vec.tolist())
            n_kept += 1


    # 5) write the CSV (header even if we've got no rows)
    if embed_dim is None:
        header = ["start_time", "end_time", "speaker"]
    else:
        header = ["start_time", "end_time", "speaker"] + [f"e{i}" for i in range(embed_dim)]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_out)

    if _os.environ.get("TATERS_DEBUG") == "1":
        print(f"[emb] rows: total={n_total}, parsed={n_parsed}, kept={n_kept}, oob={n_oob}, tiny={n_too_short}, shape_skip={n_shape_skip}")
        print(f"[emb] columns: {header}")
        print(f"[emb] wrote: {output_csv}")

    return output_csv


# ----------------------------
# New: transcript-free embeddings
# ----------------------------

def export_audio_embeddings_csv(
    source_wav: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    config: EmbedConfig = EmbedConfig(),
    sr: int = 16000,
    strategy: Literal["windows", "nonsilent"] = "windows",
    window_s: float = 30.0,
    hop_s: float = 15.0,
    min_seg_s: float = 1.0,
    top_db: float = 30.0,
    apply_l2_normalization: bool = False,
    aggregate: Literal["none", "mean"] = "none",
) -> Path:
    """
    Compute Whisper encoder embeddings for an arbitrary WAV (no transcript).

    Parameters
    ----------
    source_wav : str | Path
        Input audio (any format `librosa` can read).
    output_dir : str | Path | None, optional
        Directory for the output CSV. Defaults to the WAV's parent if None.
    config : EmbedConfig, keyword-only
        Model/device/compute configuration.
    sr : int, default 16000
        Resample rate used by the feature extractor.
    strategy : {"windows","nonsilent"}, default "windows"
        - "windows": fixed windows with hop (overlap allowed).
        - "nonsilent": energy-based voice activity detection via librosa.
    window_s, hop_s : float
        Window length and hop size (seconds). Used by both strategies.
    min_seg_s : float
        Discard segments shorter than this length (seconds).
    top_db : float
        Silence threshold for "nonsilent". Higher → fewer segments.
    aggregate : {"none","mean"}, default "none"
        If "mean", write a single pooled vector over the whole file.

    Returns
    -------
    Path
        CSV path: ``<output_dir>/<wav_stem>_embeddings.csv``.

    Notes
    -----
    - When `aggregate="none"`, rows are ``start_time,end_time,SEGMENT_i,e0..``.
    - When `aggregate="mean"`, a single row ``0.000,<dur>,GLOBAL_MEAN,e0..`` is written.
    """

    source_wav = Path(source_wav)
    if output_dir is None:
        output_dir = source_wav.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{source_wav.stem}_embeddings.csv"

    # 1) load the audio
    y, in_sr = librosa.load(str(source_wav), sr=sr, mono=True)
    n = len(y)
    if n == 0:
        # nothing to do, so we write a header-only file and bail
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["start_time", "end_time", "speaker"])
        return out_csv

    # 2) load faster-whisper + ct2 + the feature extractor (same as transcript mode)
    fw, resolved_device, resolved_compute = _load_whisper(config)
    try:
        ct2_model: ctranslate2.models.Whisper = fw.model  # type: ignore[attr-defined]
    except AttributeError:
        model_dir = getattr(fw, "model_dir", None) or getattr(fw, "_model_dir", None)
        if not model_dir:
            raise RuntimeError(
                "Could not access the underlying CTranslate2 model from faster-whisper. "
                "Consider passing a local CTranslate2 model directory as model_name."
            )
        ct2_model = ctranslate2.models.Whisper(str(model_dir), device=resolved_device,
                                              compute_type=resolved_compute)

    fe = WhisperFeatureExtractor.from_pretrained(_hf_repo_for(config.model_name))

    # 3) build up the segments (in samples)
    segs: list[tuple[int, int]] = []
    win = max(1, int(round(window_s * sr)))
    hop = max(1, int(round(hop_s * sr)))
    min_len = max(1, int(round(min_seg_s * sr)))

    if strategy == "windows":
        if n <= win:
            segs = [(0, n)]
        else:
            s = 0
            while s < n:
                e = min(n, s + win)
                segs.append((s, e))
                if e >= n:
                    break
                s += hop
    elif strategy == "nonsilent":
        # basic energy-based VAD; no torch needed, and it's fast
        intervals = librosa.effects.split(y, top_db=top_db)
        for s, e in intervals:
            if e - s < min_len:
                continue
            # chop very long spans down into ~window_s chunks
            cur = s
            while cur < e:
                nxt = min(e, cur + win)
                if nxt - cur >= min_len:
                    segs.append((cur, nxt))
                cur = nxt
        if not segs:
            # found nothing? then the whole file is one segment
            segs = [(0, n)]
    else:
        raise ValueError("strategy must be 'windows' or 'nonsilent'")

    # 4) encode each segment
    rows_out: list[list[Any]] = []
    embed_dim: Optional[int] = None
    vectors: list[np.ndarray] = []

    for i, (s, e) in enumerate(segs):
        clip = y[s:e]
        feats = fe(clip, sampling_rate=sr, return_tensors="np")["input_features"]
        vec = _encode_features_any_layout(ct2_model, feats)
        if vec is None:
            continue
        if embed_dim is None:
            embed_dim = int(vec.shape[-1])
        vectors.append(vec)
        # keep a row per chunk unless we're aggregating
        if aggregate == "none":
            t0 = s / float(sr)
            t1 = e / float(sr)
            rows_out.append([f"{t0:.3f}", f"{t1:.3f}", f"SEGMENT_{i}"] + vec.tolist())

    # 5) aggregate, if we were asked to
    if vectors and aggregate == "mean":
        vec = np.vstack(vectors).mean(axis=0)
        if apply_l2_normalization:
            vec = l2_normalize(vec)
        embed_dim = int(vec.shape[-1])
        rows_out = [["0.000", f"{n/float(sr):.3f}", "GLOBAL_MEAN"] + vec.tolist()]

    # 6) write the CSV (header even if we've got no rows)
    if embed_dim is None:
        header = ["start_time", "end_time", "speaker"]
    else:
        header = ["start_time", "end_time", "speaker"] + [f"e{i}" for i in range(embed_dim)]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_out)

    if _os.environ.get("TATERS_DEBUG") == "1":
        print(f"[emb-any] segments={len(segs)}, kept={len(rows_out)}, aggregate={aggregate}")
        print(f"[emb-any] wrote: {out_csv}")

    return out_csv



# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    from taters.helpers.cliargs import add_bool_argument
    p = argparse.ArgumentParser(description="Export Whisper encoder embeddings.")
    # two modes: (A) per-transcript segments, (B) general audio
    p.add_argument("--transcript_csv", default=None, help="If provided, export per transcript segment.")
    p.add_argument("--source_wav", required=True)
    p.add_argument("--output_dir", default=None)

    # shared model opts
    p.add_argument("--model_name", default="base")
    add_device_argument(p)
    p.add_argument("--compute_type", default="float16")

    # transcript mode
    p.add_argument("--time_unit", default="auto", choices=("auto", "ms", "s", "samples"))

    # general-audio mode
    p.add_argument("--strategy", default="windows", choices=("windows", "nonsilent"))
    p.add_argument("--window_s", type=float, default=30.0)
    p.add_argument("--hop_s", type=float, default=15.0)
    p.add_argument("--min_seg_s", type=float, default=1.0)
    p.add_argument("--top_db", type=float, default=30.0)
    p.add_argument("--aggregate", default="none", choices=("none", "mean"))
    add_bool_argument(p, "--apply_l2_normalization", default=False)

    args = p.parse_args()

    cfg = EmbedConfig(
        model_name=args.model_name,
        device=args.device,
        compute_type=args.compute_type,
        time_unit=args.time_unit,
    )

    if args.transcript_csv:
        out = export_segment_embeddings_csv(
            transcript_csv=args.transcript_csv,
            source_wav=args.source_wav,
            output_dir=args.output_dir,
            config=cfg,
            apply_l2_normalization=args.apply_l2_normalization,
        )
    else:
        out = export_audio_embeddings_csv(
            source_wav=args.source_wav,
            output_dir=args.output_dir,
            config=cfg,
            strategy=args.strategy,
            window_s=args.window_s,
            hop_s=args.hop_s,
            min_seg_s=args.min_seg_s,
            top_db=args.top_db,
            aggregate=args.aggregate,
            apply_l2_normalization=args.apply_l2_normalization,
        )
    print(f"Wrote: {out}")

