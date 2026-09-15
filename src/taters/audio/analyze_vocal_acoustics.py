"""
Acoustic feature extraction (Praat/Parselmouth-based) with optional
per-turn analysis and OpenWillis-style "simple / tremor / advanced" modes.

This module targets parity with the OpenWillis vocal acoustics stack:
- Framewise tracks: f0, formants (F1–F4), loudness (intensity), HNR.
- Summary stats of those tracks (mean, std, range), with an option to
  summarize only on voiced segments longer than 100 ms.
- Phonation metrics via Praat (jitter/shimmer families, GNE).
- Pause metrics (SPIR, DurMED, DurMAD) using energy-based VAD.
- Cepstral features (MFCC mean/var; CPPS via Praat PowerCepstrogram).
- Optional tremor metrics (requires the "tremor.praat" script).
- Optional glottal features (HRF, NAQ, OQ) via DisVoice (if installed).

It also supports "per-turn" analysis using a transcript CSV, so you can compute
speaker-level or utterance-level acoustics aligned with your diarized segments.

Outputs
-------
1) Framewise CSV (optional): one row per analysis frame (or per frame per turn).
2) Summary CSV: one row per file (or per speaker/turn grouping), including
   pass-through metadata (e.g., source, speaker) if provided.

Dependencies
------------
- Required: parselmouth (Praat), numpy, pandas (for CSV I/O only), librosa
- Optional: DisVoice (glottal metrics), pysptk (DisVoice dependency)
- Optional: Praat tremor script file if you want tremor metrics

Notes on design choices
-----------------------
- f0 range (75–500 Hz) matches OpenWillis defaults. Out-of-range f0 frames are set to 0
  for "framewise", and are excluded from summary calculations (like OpenWillis).
- Voiced frames are derived from Praat/Parselmouth tracks. A "voiced-segment >=100ms"
  filter is available for summary statistics (again, matching OpenWillis semantics).
- Pause metrics (SPIR, DurMED, DurMAD) follow OpenWillis thresholds (50 ms < pause < 2 s).
- CPPS is computed via Praat PowerCepstrogram calls; if Praat/Parselmouth lacks the function
  in your local build, we skip with a warning.
- Tremor metrics require a Praat script (tremor.praat). Provide its path if you want them.

CLI
---
python -m taters.audio.analyze_acoustics \
  --wav audio/speaker.wav \
  --out-dir features/acoustics \
  --mode simple \
  --voiced-segments true \
  --transcript-csv transcripts/X/X.csv \
  --time-unit ms \
  --group-by speaker \
  --pass-through source speaker

"""
from __future__ import annotations

import soundfile as sf
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..helpers.atomic import atomic_write
from ..helpers.progress import announce

try:
    import parselmouth  # Praat-Python bridge
    from parselmouth.praat import call as praat_call
except Exception as e:
    raise ImportError("parselmouth is required for acoustic analysis. pip install praat-parselmouth") from e

try:
    import librosa
except Exception as e:
    raise ImportError("librosa is required for IO/VAD/MFCC. pip install librosa") from e

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None
    import warnings

# ------------------------
# Normalization functions
#-------------------------

def _pydub_resample(sig: "AudioSegment", sr: int) -> "AudioSegment":
    return sig.set_frame_rate(sr)

def _pydub_remove_dc(sig: "AudioSegment") -> "AudioSegment":
    # pydub gives us DC offset helpers on WAV/MP3; we handle mono and stereo
    try:
        if sig.channels == 1:
            off = sig.get_dc_offset(channel=1)
            if off:
                sig = sig.remove_dc_offset(channel=1, offset=off)
        else:
            off_l = sig.get_dc_offset(channel=1)
            off_r = sig.get_dc_offset(channel=2)
            if off_l:
                sig = sig.remove_dc_offset(channel=1, offset=off_l)
            if off_r:
                sig = sig.remove_dc_offset(channel=2, offset=off_r)
    except Exception:
        # some formats/backends don't report DC at all; we just move along quietly
        pass
    return sig

def _pydub_normalize(sig: "AudioSegment", target_dbfs: float = -20.0) -> "AudioSegment":
    # make sure we don't clip: we cap the gain at whatever headroom is left to 0 dBFS
    try:
        headroom = -sig.max_dBFS  # how far we can push it before clipping (positive)
        gain = target_dbfs - sig.dBFS
        if gain > headroom:
            gain = headroom
        return sig.apply_gain(gain)
    except Exception:
        return sig

def _pydub_to_float_mono(sig: "AudioSegment") -> tuple[np.ndarray, int]:
    # convert to mono float32 in [-1,1]; we keep the target sample rate
    if sig.channels > 1:
        sig = sig.set_channels(1)
    sr = sig.frame_rate
    # get_array_of_samples hands back int PCM, so we normalize by the sample width
    arr = np.array(sig.get_array_of_samples())
    # the scale falls out of the sample width
    max_int = float(1 << (8 * sig.sample_width - 1))
    y = (arr.astype(np.float32) / max_int).copy()
    return y, sr

def _min_required_duration_s(f0_min: float) -> float:
    """
    Praat intensity analysis requires ~6.4 / min_pitch seconds of audio.
    Praat's to_intensity() defaults to a 100 Hz min pitch internally,
    so we honor the stricter of:
      - 6.4 / 100
      - 6.4 / f0_min (our pitch floor)
      - and a hard floor of 0.064 s (the 100 Hz example)
    """
    rule_100 = 6.4 / 100.0
    rule_f0  = 6.4 / max(1e-6, float(f0_min))
    return max(0.064, rule_100, rule_f0)


def _pad_sound_to_min_duration(snd: parselmouth.Sound, min_dur_s: float) -> parselmouth.Sound:
    """Pad a parselmouth.Sound with silence (split on both ends) to reach min_dur_s."""
    dur = float(snd.get_total_duration())
    if dur >= min_dur_s:
        return snd
    sr = int(snd.sampling_frequency)
    needed = int(round((min_dur_s - dur) * sr))
    left = needed // 2
    right = needed - left
    pad_left = parselmouth.Sound(values=np.zeros((1, left), dtype=np.float64), sampling_frequency=sr)
    pad_right = parselmouth.Sound(values=np.zeros((1, right), dtype=np.float64), sampling_frequency=sr)

    # IMPORTANT: we use the class-level concatenator so we get a brand new, longer
    # Sound back
    try:
        concatenated = parselmouth.Sound.concatenate([pad_left, snd, pad_right])
    except Exception:
        # if that's not there, we go through Praat's "Concatenate" instead
        concatenated = parselmouth.praat.call([pad_left, snd, pad_right], "Concatenate")
    return concatenated


def _pad_numpy_to_min_duration(y: np.ndarray, sr: int, min_dur_s: float) -> np.ndarray:
    """Pad a 1D numpy signal with zeros (split on both ends) to reach min_dur_s."""
    cur = len(y)
    need = int(round(min_dur_s * sr))
    if cur >= need:
        return y
    add = need - cur
    left = add // 2
    right = add - left
    return np.pad(y, (left, right), mode="constant", constant_values=0.0)


# -------------------------
# Types and configuration
# -------------------------

Mode = Literal["simple", "tremor", "advanced"]  # OpenWillis-style option set


@dataclass
class FramewiseTracks:
    """Aligned framewise series at a fixed hop (e.g., 10 ms)."""
    times: np.ndarray          # seconds
    f0: np.ndarray             # Hz (0 for unvoiced)
    f1: np.ndarray             # Hz (nan if undefined)
    f2: np.ndarray             # Hz
    f3: np.ndarray             # Hz
    f4: np.ndarray             # Hz
    loudness_db: np.ndarray    # dB
    hnr_db: np.ndarray         # dB (may include -inf where undefined)


# -------------------------
# Helpers
# -------------------------

def _ensure_mono_sound(sound: parselmouth.Sound) -> parselmouth.Sound:
    """Convert to mono if stereo, keeping Praat-native Sound object."""
    if sound.get_number_of_channels() == 1:
        return sound
    # average the channels together (Praat's Mix)
    return praat_call(sound, "Convert to mono")


def _load_audio(
    wav_path: Union[str, Path],
    *,
    preprocess: bool = True,
    target_sr: int = 44100,
    target_dbfs: float = -20.0,
    remove_dc: bool = True,
) -> tuple[np.ndarray, int, "parselmouth.Sound"]:
    """
    Load audio → (optionally) OpenWillis-style preprocess → return:
      y  : float32 mono waveform in [-1, 1]
      sr : sample rate
      snd: parselmouth.Sound created from the same audio
    """
    wav_path = Path(wav_path)

    if preprocess and AudioSegment is not None:
        try:
            fmt = "wav" if str(wav_path).lower().endswith(".wav") else None
            sig = AudioSegment.from_file(str(wav_path), format=fmt)
            # 1) resample
            sig = _pydub_resample(sig, target_sr)
            # 2) knock out the DC offset
            if remove_dc:
                sig = _pydub_remove_dc(sig)
            # 3) loudness normalization
            sig = _pydub_normalize(sig, target_dbfs=target_dbfs)
            # then over to a numpy mono float
            y, sr = _pydub_to_float_mono(sig)
            # and a Praat Sound built from that same numpy
            snd = parselmouth.Sound(y, sr)
            return y, sr, snd
        except Exception as e:
            warnings.warn(f"[acoustics] pydub preprocessing failed ({e}); falling back to librosa.")
            # fall through to the librosa path below

    # fallback: plain librosa load (so no DC removal or volume normalization)
    y, sr = librosa.load(str(wav_path), sr=target_sr if preprocess else None, mono=True)
    snd = parselmouth.Sound(y, sr)
    return y, sr, snd


def _framewise_tracks(
    snd: parselmouth.Sound,
    *,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    time_step: Optional[float] = None,
    formant_max: float = 5500.0,
    formant_n: int = 4,
) -> FramewiseTracks:
    # fixed hop (10 ms unless told otherwise)
    ts = 0.01 if (time_step is None or time_step <= 0) else float(time_step)

    # pitch is what decides where the frame centers land
    pitch = snd.to_pitch(time_step=ts, pitch_floor=f0_min, pitch_ceiling=f0_max)
    times = pitch.xs()
    f0_vals = pitch.selected_array["frequency"]  # Hz (0 = unvoiced)

    # intensity (loudness), lined up on the same hop
    intensity = snd.to_intensity(time_step=ts)
    loudness_db = np.interp(times, intensity.xs(), intensity.values[0])

    # harmonicity (cc) -> HNR in dB
    harm = snd.to_harmonicity_cc(time_step=ts, minimum_pitch=f0_min)
    hnr_db = np.interp(times, harm.xs(), harm.values[0])  # may include -inf

    # formants via Burg (note that the kwarg is max_number_of_formants)
    max_formants = int(max(1, min(formant_n, 5)))
    formant = snd.to_formant_burg(
        time_step=ts,
        max_number_of_formants=float(max_formants),
        maximum_formant=formant_max,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )

    def _sample_formant(n: int) -> np.ndarray:
        vals = np.empty_like(times)
        vals.fill(np.nan)
        for i, t in enumerate(times):
            v = formant.get_value_at_time(n, float(t))
            vals[i] = np.nan if (v is None or np.isnan(v) or v <= 0) else v
        return vals

    # we always hand back f1..f4; any we didn't ask for get filled with NaN
    f1 = _sample_formant(1) if max_formants >= 1 else np.full_like(times, np.nan)
    f2 = _sample_formant(2) if max_formants >= 2 else np.full_like(times, np.nan)
    f3 = _sample_formant(3) if max_formants >= 3 else np.full_like(times, np.nan)
    f4 = _sample_formant(4) if max_formants >= 4 else np.full_like(times, np.nan)

    return FramewiseTracks(
        times=times,
        f0=f0_vals,
        f1=f1,
        f2=f2,
        f3=f3,
        f4=f4,
        loudness_db=loudness_db,
        hnr_db=hnr_db,
    )


def _voiced_mask_from_f0(f0: np.ndarray) -> np.ndarray:
    """Boolean mask where f0 > 0 (i.e., voiced frames)."""
    return f0 > 0


def _segments_from_mask(times: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    Convert a boolean mask over frames into contiguous (start, end) time segments.
    A segment starts when mask goes False→True and ends when it goes True→False.
    The final open segment, if any, is closed at the end of the series.
    """
    if len(times) == 0 or len(mask) == 0:
        return []
    if len(times) != len(mask):
        raise ValueError("times and mask must have the same length")

    segs: List[Tuple[float, float]] = []
    cur_on: Optional[float] = None
    # a stable hop estimate, for padding out the end
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01

    for i, on in enumerate(mask):
        if on and cur_on is None:
            # voicing just switched on: start a new segment
            cur_on = float(times[i])
        elif (not on) and cur_on is not None:
            # voicing switched off: end the segment midway between frames i-1
            # and i (if we can)
            if i > 0:
                t_end = float(0.5 * (times[i - 1] + times[i]))
            else:
                t_end = float(times[i])
            if t_end > cur_on:
                segs.append((cur_on, t_end))
            cur_on = None

    # if we're still inside a segment at the tail, close it off
    if cur_on is not None:
        segs.append((cur_on, float(times[-1]) + 0.5 * hop))

    return segs


def _stats_on_segments(values: np.ndarray, times: np.ndarray, segments: List[Tuple[float, float]]) -> Tuple[float,float,float]:
    """
    Compute mean/std/range of `values` restricted to given time segments.
    If no coverage, return (NaN, NaN, NaN).
    """
    if len(segments) == 0:
        return (np.nan, np.nan, np.nan)
    mask = np.zeros_like(values, dtype=bool)
    for (a,b) in segments:
        mask |= (times >= a) & (times <= b)
    vals = values[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    return float(np.mean(vals)), float(np.std(vals)), float(np.ptp(vals))  # ptp = max - min


def _summarize_framewise(
    tr: FramewiseTracks,
    voiced_only_longer_than: Optional[float] = None  # seconds; e.g., 0.10 = 100 ms
) -> Dict[str, float]:
    """
    Summaries (mean/std/range) of f0, F1–F4, loudness, HNR.
    If `voiced_only_longer_than` is set, we compute on voiced segments whose
    duration >= threshold; otherwise compute on all frames for loudness/HNR/formants,
    and on f0>0 for f0 summaries.
    """
    # first thing's first: which frames are voiced, going by f0
    voiced_mask = _voiced_mask_from_f0(tr.f0)
    voiced_segments = _segments_from_mask(tr.times, voiced_mask)

    if voiced_only_longer_than is not None:
        voiced_segments = [(a,b) for (a,b) in voiced_segments if (b-a) >= voiced_only_longer_than]

    # little helper: segment-aware stats or plain global ones
    def stat_block(vals: np.ndarray, voiced_only=False) -> Dict[str, float]:
        if voiced_only:
            m, s, r = _stats_on_segments(vals, tr.times, voiced_segments)
        else:
            v = vals[np.isfinite(vals)]
            if v.size == 0:
                m, s, r = (np.nan, np.nan, np.nan)
            else:
                m, s, r = (float(np.mean(v)), float(np.std(v)), float(np.ptp(v)))
        return {"mean": m, "std": s, "range": r}

    out: Dict[str, float] = {}
    # f0 stats only make sense on voiced frames (plus the segment-length
    # threshold, if one was asked for)
    out.update({f"f0_{k}": v for k, v in stat_block(tr.f0, voiced_only=True).items()})

    # formants, loudness, HNR: OpenWillis summarizes these on the same segments
    # when asked, so we do too
    for name, arr in [("f1", tr.f1), ("f2", tr.f2), ("f3", tr.f3), ("f4", tr.f4),
                      ("loudness", tr.loudness_db), ("hnr", tr.hnr_db)]:
        out.update({f"{name}_{k}": v for k, v in stat_block(arr, voiced_only=(voiced_only_longer_than is not None)).items()})

    # silence ratio: what fraction of frames are unvoiced
    out["silence_ratio"] = float(np.mean(~voiced_mask)) if tr.f0.size else np.nan
    return out


def _pause_metrics(
    y: np.ndarray,
    sr: int,
    min_pause_ms: float = 50.0,
    max_pause_s: float = 2.0,
    *,
    top_db: int = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """
    Pause-related metrics following OpenWillis (Kovac et al., 2023).

    Parameters
    ----------
    y, sr : waveform and sample rate
    min_pause_ms : float, default 50.0
        Minimum pause duration to count (in milliseconds).
    max_pause_s : float, default 2.0
        Maximum pause duration to count (in seconds).
    top_db : int, default 30
        Non-silence threshold (dB below peak) for librosa.effects.split.
        Larger → detects fewer speech segments (more 'silence').
    frame_length : int, default 2048
        Analysis window length (samples) for energy computation.
    hop_length : int, default 512
        Hop between frames (samples).

    Returns
    -------
    dict with SPIR, DurMED, DurMAD
    """
    # find the stretches that aren't silent
    non_silent = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    duration_s = len(y) / sr

    # the pauses are just whatever's left over between the speech
    speech_segments = [(start / sr, end / sr) for start, end in non_silent]
    pauses: List[Tuple[float, float]] = []
    if not speech_segments:
        pauses = [(0.0, duration_s)]
    else:
        t = 0.0
        for (s, e) in speech_segments:
            if s > t:
                pauses.append((t, s))
            t = e
        if t < duration_s:
            pauses.append((t, duration_s))

    # only keep pauses that fall within [min_pause_ms, max_pause_s]
    min_pause_s = min_pause_ms / 1000.0
    pauses = [(a, b) for (a, b) in pauses if (b - a) >= min_pause_s and (b - a) <= max_pause_s]

    if not pauses:
        return {"SPIR": 0.0, "DurMED": 0.0, "DurMAD": 0.0}

    pause_durs = np.array([b - a for (a, b) in pauses], dtype=float)
    speech_time = sum((b - a) for (a, b) in speech_segments) if speech_segments else 0.0
    spir = (len(pauses) / speech_time) if speech_time > 0 else 0.0
    dur_med = float(np.median(pause_durs))
    dur_mad = float(np.median(np.abs(pause_durs - dur_med)))
    return {"SPIR": float(spir), "DurMED": dur_med, "DurMAD": dur_mad}


def _phonation_metrics(snd: parselmouth.Sound, f0_min: float = 75.0, f0_max: float = 500.0) -> Dict[str, float]:
    """
    Jitter/shimmer family + GNE from Praat via parselmouth.
    Uses argument signatures compatible with common Praat builds.
    Period bounds are derived from the f0 range to avoid out-of-range artifacts.
    """
    # first we need a PointProcess; everything jitter/shimmer builds on it
    point_process = praat_call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)

    # turn the f0 bounds (Hz) into period bounds (seconds)
    minPeriod = 1.0 / float(f0_max)  # shortest plausible cycle
    maxPeriod = 1.0 / float(f0_min)  # longest plausible cycle
    maxPeriodFactor = 1.3            # Praat's standard factor

    # ----- JITTER -----
    # most Praat builds want the 5-arg form:
    # (from_time, to_time, minimumPeriod, maximumPeriod, maximumPeriodFactor)
    def _jit(method: str) -> float:
        return float(praat_call(point_process, method, 0, 0, minPeriod, maxPeriod, maxPeriodFactor))

    jitter_local = _jit("Get jitter (local)")
    jitter_rap   = _jit("Get jitter (rap)")
    jitter_ppq5  = _jit("Get jitter (ppq5)")
    jitter_ddp   = _jit("Get jitter (ddp)")

    # ----- SHIMMER -----
    # Praat's signatures move around between versions. we try the 6-arg flavor
    # first:
    # (from_time, to_time, minimumPeriod, maximumPeriod, maximumAmplitudeFactor,
    #  maximumPeriodFactor)
    # and if that blows up, we retry with the 8-arg flavor that adds
    # silenceThreshold/periodsPerWindow.
    def _shim(method: str) -> float:
        # 6-arg flavor
        try:
            return float(praat_call([snd, point_process], method,
                                    0, 0, minPeriod, maxPeriod, 1.3, 1.6))
        except Exception:
            # 8-arg flavor:
            # (from, to, minPeriod, maxPeriod, maxAmplitudeFactor, silenceThreshold,
            #  periodsPerWindow, maxPeriodFactor)
            return float(praat_call([snd, point_process], method,
                                    0, 0, minPeriod, maxPeriod, 1.3, 0.03, 1.0, 1.6))

    shimmer_local     = _shim("Get shimmer (local)")
    shimmer_local_db  = _shim("Get shimmer (local_dB)")
    shimmer_apq3      = _shim("Get shimmer (apq3)")
    shimmer_apq5      = _shim("Get shimmer (apq5)")
    shimmer_apq11     = _shim("Get shimmer (apq11)")
    shimmer_dda       = _shim("Get shimmer (dda)")

    # ----- GNE (optional; not every Praat build has it) -----
    gne = np.nan
    try:
        harm_gne = praat_call(snd, "To Harmonicity (gne)", f0_min, 0.01, 2.0)  # (min pitch, time step, periods/window)
        gne_vals = np.asarray(harm_gne.values[0], dtype=float)
        gne = float(np.nanmean(gne_vals)) if gne_vals.size else np.nan
    except Exception:
        warnings.warn("Praat 'To Harmonicity (gne)' not available; skipping GNE")

    return {
        "jitter_local": jitter_local,
        "jitter_rap": jitter_rap,
        "jitter_ppq5": jitter_ppq5,
        "jitter_ddp": jitter_ddp,
        "shimmer_local": shimmer_local,
        "shimmer_local_db": shimmer_local_db,
        "shimmer_apq3": shimmer_apq3,
        "shimmer_apq5": shimmer_apq5,
        "shimmer_apq11": shimmer_apq11,
        "shimmer_dda": shimmer_dda,
        "gne": gne,
    }


def _mfcc_stats(y: np.ndarray, sr: int, n_mfcc: int = 14) -> Dict[str, float]:
    """
    MFCC mean/variance (1..n_mfcc) over the whole signal (or segment).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    out = {}
    for i in range(n_mfcc):
        out[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        out[f"mfcc{i+1}_var"]  = float(np.var(mfcc[i]))
    return out


def _cpps_via_praat(snd: parselmouth.Sound, f0_min: float = 75.0) -> Dict[str, float]:
    """
    CPPS (Cepstral Peak Prominence, smoothed) via Praat PowerCepstrogram calls.
    This follows typical Praat recipes used in clinical voice literature.
    If unavailable in your build, we skip gracefully.
    """
    try:
        # these defaults are meant to be robust; we can expose them as knobs if
        # anyone ever needs to
        pcep = praat_call(snd, "To PowerCepstrogram", 0.01, f0_min, 0.0, 0.05, 50.0)  # (time step, f0min, qlow, qhigh, fast)
        cpps = float(praat_call(pcep, "Get CPPS", 0, 0, f0_min, 0.05, "Straight"))   # (t1,t2,f0min, ceiling, "Straight")
        return {"cpps": cpps}
    except Exception:
        warnings.warn("CPPS via Praat not available; skipping CPPS")
        return {"cpps": np.nan}


def _tremor_metrics_via_script(snd: parselmouth.Sound, tremor_script: Optional[Union[str, Path]]) -> Dict[str, float]:
    """
    Run an external Praat script (e.g., tremor.praat) that prints or returns
    a set of tremor-related values. We assume the script writes a TableOfReal
    to the Praat environment and we read values back.
    """
    if not tremor_script:
        return {}  # the user didn't ask for tremor
    try:
        tremor_script = str(tremor_script)
        # most tremor scripts expect the current sound to be the selection.
        # we pass arguments where needed; here we just lean on the defaults.
        # the usual pattern is: run script -> get a Table back -> read columns.
        # we call this for its side effect and throw the return value away:
        # what a tremor script hands back is script-specific and we don't
        # have an agreed contract to parse yet. binding it to `result` made
        # it look like one was coming, and left a variable nothing ever read.
        parselmouth.praat.run_file(tremor_script, -20, 2, "no")
        return {}  # stays empty until we pin down a specific script contract
    except Exception:
        warnings.warn("Tremor script execution failed; skipping tremor metrics")
        return {k: np.nan for k in ("FCoM","FTrC","FMoN","FTrF","FTrI","FTrP","FTrCIP","FTrPS","FCoHNR",
                                    "ACoM","ATrC","AMoN","ATrF","ATrI","ATrP","ATrCIP","ATrPS","ACoHNR")}


def _glottal_features_via_disvoice(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Optional: leverage DisVoice glottal module (HRF, NAQ, OQ) if available.
    We compute averages and variances over estimated cycles.
    """
    try:
        from disvoice.glottal import Glottal
        gl = Glottal()
        feats = gl.extract_features(y, sr)
        # DisVoice gives us a dict or an array; we summarize the key ones if
        # they're there.
        out: Dict[str, float] = {}
        # the names depend on the version, so we just probe the common keys:
        for key in ["HRF", "NAQ", "OQ"]:
            vals = feats.get(key)
            if vals is not None and len(vals) > 0:
                out[f"{key.lower()}_mean"] = float(np.mean(vals))
                out[f"{key.lower()}_var"] = float(np.var(vals))
        return out
    except Exception:
        warnings.warn("DisVoice not installed or failed; skipping glottal features (HRF/NAQ/OQ)")
        return {}


# -------------------------
# Main analysis for a clip
# -------------------------

def _analyze_clip(
    wav_path: Union[str, Path],
    *,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    summarize_on_voiced_segments_ms: Optional[int] = 100,
    mode: Mode = "simple",
    include_framewise: bool = True,
    n_mfcc: int = 14,
    tremor_script: Optional[Union[str, Path]] = None,
    preprocess: bool = True,
    target_sr: int = 44100,
    target_dbfs: float = -20.0,
    remove_dc: bool = True,
    # VAD / pause tuning
    pause_top_db: int = 30,
    pause_frame_length: int = 2048,
    pause_hop_length: int = 512,
) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    """
    Run full stack on a single WAV and return (framewise_df, summary_dict).
    framewise_df is None if include_framewise=False.
    """
    y, sr, snd = _load_audio(
        wav_path,
        preprocess=preprocess,
        target_sr=target_sr,
        target_dbfs=target_dbfs,
        remove_dc=remove_dc,
    )
    snd = _ensure_mono_sound(snd)

    # --- make sure the signal is long enough for Praat's pitch/intensity windows ---
    # Praat's rule of thumb: at least 6.4 / f0_min seconds (e.g., 64 ms at 100 Hz).
    min_dur_s = _min_required_duration_s(f0_min)

    if (len(y) / sr) < min_dur_s:
        y   = _pad_numpy_to_min_duration(y, sr, min_dur_s)
        snd = _pad_sound_to_min_duration(snd, min_dur_s)


    # belt and braces: if intensity still throws a duration error (rare edge
    # rounding), we pad a little more and retry inside the tracks call
    def _safe_framewise_tracks(snd_obj: parselmouth.Sound, **kwargs) -> FramewiseTracks:
        try:
            return _framewise_tracks(snd_obj, **kwargs)
        except Exception as e:
            # we only want to catch the window-length / intensity errors here
            msg = str(e)
            if "shorter than window length" in msg or "intensity analysis not performed" in msg:
                # pad out to the same min_dur_s (plus a hair of margin) and retry
                extra = 0.005  # 5 ms of safety
                snd_padded = _pad_sound_to_min_duration(snd_obj, min_dur_s + extra)
                return _framewise_tracks(snd_padded, **kwargs)
            raise

    # we go through the safe wrapper rather than calling _framewise_tracks directly
    tracks = _safe_framewise_tracks(snd, f0_min=f0_min, f0_max=f0_max)

    # the framewise DataFrame (only if asked for)
    frame_df: Optional[pd.DataFrame] = None
    if include_framewise:
        frame_df = pd.DataFrame({
            "time_s": tracks.times,
            "f0_hz": tracks.f0,
            "f1_hz": tracks.f1,
            "f2_hz": tracks.f2,
            "f3_hz": tracks.f3,
            "f4_hz": tracks.f4,
            "loudness_db": tracks.loudness_db,
            "hnr_db": tracks.hnr_db,
        })

    # now we summarize the framewise series
    voiced_th = (summarize_on_voiced_segments_ms / 1000.0) if summarize_on_voiced_segments_ms is not None else None
    summary: Dict[str, float] = _summarize_framewise(tracks, voiced_only_longer_than=voiced_th)

    # pause metrics and MFCCs, on the same *preprocessed* y,sr we use everywhere else
    summary.update(_pause_metrics(
        y, sr,
        top_db=pause_top_db,
        frame_length=pause_frame_length,
        hop_length=pause_hop_length,
    ))
    summary.update(_mfcc_stats(y, sr, n_mfcc=n_mfcc))

    # CPPS
    summary.update(_cpps_via_praat(snd, f0_min=f0_min))

    # phonation (jitter/shimmer/GNE)
    summary.update(_phonation_metrics(snd, f0_min=f0_min, f0_max=f0_max))

    # lastly, whatever extras the mode calls for
    if mode in ("tremor", "advanced"):
        summary.update(_tremor_metrics_via_script(snd, tremor_script))
    if mode == "advanced":
        summary.update(_glottal_features_via_disvoice(y, sr))

    return frame_df, summary


# -------------------------
# Per-turn/per-speaker analysis
# -------------------------


def _analyze_turns(
    wav_path: Union[str, Path],
    transcript_csv: Union[str, Path],
    *,
    time_unit: Literal["ms", "s"] = "ms",
    start_col: str = "start_time",
    end_col: str = "end_time",
    text_col: str = "text",
    extra_id_cols: Sequence[str] = (),
    group_by: Optional[Sequence[str]] = None,  # e.g., ["speaker"]
    mode: Mode = "simple",
    summarize_on_voiced_segments_ms: Optional[int] = 100,
    include_framewise: bool = False,  # framewise can be huge
    tremor_script: Optional[Union[str, Path]] = None,
    # the top-level knobs, so they make it all the way down
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    n_mfcc: int = 14,
    preprocess: bool = True, 
    target_sr: int = 44100, 
    target_dbfs: float = -20.0, 
    remove_dc: bool = True,
    # VAD/"pause" tuning
    pause_top_db: int = 30,
    pause_frame_length: int = 2048,
    pause_hop_length: int = 512,
    on_progress: Optional[Callable[..., None]] = None,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Analyze a single WAV guided by a transcript CSV (utterance-level rows).
    Returns (framewise_df_or_None, summary_df).

    Notes
    -----
    - If `text_col` exists in the transcript, its content is attached to each
      per-turn summary row as `utterance_text`.
    - When `group_by` is used, `utterance_text` is dropped prior to aggregation.
    """
    wav_path = Path(wav_path)

    # we load (and preprocess) once, then slice y up for each turn; the actual
    # analysis runs on the per-turn WAV slices
    y, sr, _snd = _load_audio(
        wav_path,
        preprocess=preprocess,
        target_sr=target_sr,
        target_dbfs=target_dbfs,
        remove_dc=remove_dc,
    )

    # read the transcript and make sure the timing columns are actually there
    df = pd.read_csv(transcript_csv)
    missing = [c for c in (start_col, end_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Transcript must have {missing} columns")

    # get the times into seconds
    factor = 0.001 if time_unit == "ms" else 1.0
    df["_start_s"] = pd.to_numeric(df[start_col], errors="coerce") * factor
    df["_end_s"]   = pd.to_numeric(df[end_col],   errors="coerce") * factor
    df = df.dropna(subset=["_start_s", "_end_s"])

    from tempfile import TemporaryDirectory
    rows: List[Dict[str, object]] = []
    framewise_rows: List[pd.DataFrame] = []

    announce(on_progress, "measuring turns")
    done = 0
    with TemporaryDirectory(prefix=".tmp_acoustic_slices_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for i, r in df.iterrows():
            done += 1
            if on_progress is not None:
                on_progress(done, len(df), "measuring turns")
            start_s = float(r["_start_s"])
            end_s   = float(r["_end_s"])

            # clamp to the audio bounds, and skip anything bogus or very short
            start_s = max(0.0, min(start_s, len(y) / sr))
            end_s   = max(0.0, min(end_s,   len(y) / sr))
            if not (end_s > start_s):
                continue
            if (end_s - start_s) < 0.020:  # under 20 ms is too short to bother with
                continue

            s_idx = int(round(start_s * sr))
            e_idx = int(round(end_s * sr))
            ys = y[s_idx:e_idx]

            # write the slice out as a 16-bit PCM WAV
            slice_path = tmpdir_path / f"slice_{i:06d}.wav"
            sf.write(str(slice_path), ys, sr, subtype="PCM_16")

            # and analyze it
            fdf, summ = _analyze_clip(
                slice_path,
                mode=mode,
                summarize_on_voiced_segments_ms=summarize_on_voiced_segments_ms,
                include_framewise=include_framewise,
                tremor_script=tremor_script,
                f0_min=f0_min,
                f0_max=f0_max,
                n_mfcc=n_mfcc,
                preprocess=False,          # the slice was already preprocessed above
                target_sr=sr,
                target_dbfs=target_dbfs,
                remove_dc=False,
                pause_top_db=pause_top_db,
                pause_frame_length=pause_frame_length,
                pause_hop_length=pause_hop_length,
            )

            # tack on the IDs and timing
            summ["start_s"] = start_s
            summ["end_s"]   = end_s
            summ["segment_index"] = int(i)

            # and the utterance text, if we've got it
            if text_col in df.columns:
                try:
                    summ["utterance_text"] = None if pd.isna(r[text_col]) else str(r[text_col])
                except Exception:
                    summ["utterance_text"] = None

            # any extra ID columns from the transcript row, if they're there
            for c in extra_id_cols:
                if c in r:
                    summ[c] = r[c]

            rows.append(summ)

            # framewise rows (no text on these, or they'd balloon)
            if include_framewise and fdf is not None and len(fdf):
                fdf = fdf.copy()
                fdf["segment_index"] = int(i)
                fdf["start_s"] = start_s
                fdf["end_s"]   = end_s
                for c in extra_id_cols:
                    if c in r:
                        fdf[c] = r[c]
                framewise_rows.append(fdf)

    # build the per-turn summary DataFrame
    summary_df = pd.DataFrame(rows)

    # aggregate by group if asked (e.g., per speaker)
    if group_by:
        group_by = list(group_by)
        if not summary_df.empty:
            # utterance_text has to go first; there's no meaningful way to aggregate it
            if "utterance_text" in summary_df.columns:
                summary_df = summary_df.drop(columns=["utterance_text"])
            num_cols = summary_df.select_dtypes(include=[np.number]).columns
            grouped = summary_df.groupby(group_by, dropna=False)[num_cols].agg(["mean", "std"])
            grouped.columns = ["_".join(col).strip() for col in grouped.columns.to_flat_index()]
            summary_df = grouped.reset_index()
        else:
            summary_df = pd.DataFrame(columns=list(group_by))

    framewise_df: Optional[pd.DataFrame] = None
    if include_framewise and framewise_rows:
        framewise_df = pd.concat(framewise_rows, axis=0, ignore_index=True)

    return framewise_df, summary_df


# -------------------------
# Public API (Taters-style)
# -------------------------

def analyze_acoustics(
    *,
    # Inputs (choose ONE of these two paths)
    wav_path: Optional[Union[str, Path]] = None,
    transcript_csv: Optional[Union[str, Path]] = None,  # if provided, we do per-turn analysis
    # Transcript options
    time_unit: Literal["ms","s"] = "ms",
    group_by: Optional[Sequence[str]] = None,       # e.g., ["speaker"]
    extra_id_cols: Sequence[str] = ("source","speaker"),
    # Output
    out_dir: Optional[Union[str, Path]] = None,
    out_framewise_csv: Optional[Union[str, Path]] = None,
    out_summary_csv: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,
    include_framewise: bool = True,
    # Analysis options
    mode: Mode = "simple",
    summarize_on_voiced_segments_ms: Optional[int] = 100,
    f0_min: float = 75.0,
    f0_max: float = 500.0,
    n_mfcc: int = 14,
    tremor_script: Optional[Union[str, Path]] = None,
    # preprocessing controls
    preprocess: bool = True,
    target_sr: int = 44100,
    target_dbfs: float = -20.0,
    remove_dc: bool = True,
    # VAD/"pause" tuning
    pause_top_db: int = 30,
    pause_frame_length: int = 2048,
    pause_hop_length: int = 512,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Dict[str, Optional[Path]]:
    """
    Extract acoustic features and write a summary CSV and (by default) a framewise CSV.

    This function computes a battery of speech/voice features using
    Praat/Parselmouth-style workflows with optional cepstral, tremor, and glottal
    measures. It supports two operating modes:

    1) Whole-file analysis
       Features are derived across the entire WAV. Summary statistics can be
       restricted to voiced segments longer than a threshold.

    2) Per-turn analysis (transcript-guided)
       The WAV is segmented using `start_time`/`end_time` from a transcript CSV,
       features are computed per segment, and (optionally) per-segment rows are
       aggregated via `group_by` (e.g., one row per speaker).

    Two artifacts can be written:
      • Summary CSV (always): means/SDs/ranges of framewise series; silence ratio;
        jitter/shimmer; MFCC means/variances; CPP (if available); optional tremor/glottal.
        With a transcript and `group_by`, the summary is aggregated per group.
      • Framewise CSV (default): one row per short-time frame (f0, F1–F4, loudness, HNR).
        Disable with `include_framewise=False` or set a custom path.

    Parameters
    ----------
    wav_path : str or pathlib.Path, optional
        Path to a WAV file (mono or stereo, PCM). Required for both whole-file
        and per-turn modes. If both `wav_path` and `transcript_csv` are ``None``,
        a ``ValueError`` is raised.
    transcript_csv : str or pathlib.Path, optional
        Path to a transcript CSV with at least `start_time`, `end_time` (and
        typically `speaker`). Intervals with non-positive duration are skipped.
        When provided, per-turn analysis is performed.
    time_unit : {"ms", "s"}, default "ms"
        Units for `start_time` and `end_time` in `transcript_csv`.
    group_by : sequence of str, optional
        Column names from `transcript_csv` used to aggregate per-turn summaries
        into higher-level rows (e.g., `["speaker"]`). If omitted, per-turn rows
        are written without aggregation.
    extra_id_cols : sequence of str, default ("source", "speaker")
        Identifier/metadata columns to pass through when present (and to use as
        grouping keys where applicable). These are not numerically aggregated.
    out_dir : str or pathlib.Path, optional
        Base directory for outputs if file paths are not given. Defaults to
        ``./features/acoustics`` (created if missing).
    out_framewise_csv : str or pathlib.Path, optional
        Path for the framewise CSV. If omitted and `include_framewise=True`,
        defaults to ``<out_dir>/<stem>_framewise.csv``.
    out_summary_csv : str or pathlib.Path, optional
        Path for the summary CSV. If omitted, defaults to
        ``<out_dir>/<stem>_summary.csv`` (or an equivalent name in per-turn mode).
    overwrite_existing : bool, default False
        If ``False`` and an output already exists, returns existing paths without
        recomputation. If ``True``, outputs are recomputed and overwritten.
    include_framewise : bool, default True
        If ``True``, also write the framewise table. Set to ``False`` to write
        only the summary.
    mode : {"simple", "tremor", "advanced"}, default "simple"
        Feature families to compute:
          - ``"simple"``: framewise f0, formants (F1–F4), loudness, HNR; summary stats;
            silence ratio; jitter/shimmer; MFCC means/variances; CPP (if available).
          - ``"tremor"``: everything in *simple* plus tremor metrics via a Praat script
            (requires `tremor_script`).
          - ``"advanced"``: everything in *tremor* plus glottal features (requires
            DisVoice and dependencies).
    summarize_on_voiced_segments_ms : int or None, default 100
        If an integer, summary statistics for framewise series are computed only
        on voiced segments whose duration is at least this many milliseconds.
        If ``None``, all frames are used.
    f0_min : float, default 75.0
        Minimum fundamental frequency (Hz) for pitch tracking. Out-of-range f0
        values are treated as unvoiced (0 in framewise; excluded from voiced summaries).
    f0_max : float, default 500.0
        Maximum fundamental frequency (Hz) for pitch tracking.
    n_mfcc : int, default 14
        Number of MFCC coefficients to summarize (means and variances).
    tremor_script : str or pathlib.Path, optional
        Path to a Praat tremor script. Required when ``mode in {"tremor","advanced"}``.
    preprocess : bool, default True
        If ``True`` (whole-file mode), resample to `target_sr`, optionally remove
        DC offset (`remove_dc`), and normalize level toward `target_dbfs` with
        headroom protection. In per-turn mode, slices are analyzed with consistent
        parameters and are not re-normalized per slice.
    target_sr : int, default 44100
        Target sample rate for preprocessing (whole-file mode).
    target_dbfs : float, default -20.0
        Target loudness (dBFS) for level normalization (whole-file mode).
    remove_dc : bool, default True
        If ``True``, attempt to remove DC offset during preprocessing (whole-file mode).
    pause_top_db : int, default 30
        Non-silence threshold for pause detection (higher → fewer speech segments).
        Passed to ``librosa.effects.split``.
    pause_frame_length : int, default 2048
        Frame length (samples) for pause detection.
    pause_hop_length : int, default 512
        Hop length (samples) for pause detection.
    verbose : bool, default True
        Print what the step is doing. The pipeline runner turns this off
        under its live display; without the parameter the skip-path message
        landed in the middle of that display.
    on_progress : callable, optional
        ``on_progress(done, total, message)``. Per-turn analysis is the
        longest CPU-only step in a conversation pipeline, and it reported
        nothing while it ran; now every measured turn ticks.

    Returns
    -------
    dict
        Mapping with:
        ``{"framewise_csv": pathlib.Path or None, "summary_csv": pathlib.Path}``.

    Raises
    ------
    ValueError
        If neither `wav_path` nor `transcript_csv` is provided; if `transcript_csv`
        is provided without `wav_path`; if required transcript columns are missing; or
        if `mode` requires unavailable dependencies (e.g., `tremor_script` for
        ``"tremor"``, DisVoice for ``"advanced"``).
    FileNotFoundError
        If provided paths do not exist.
    RuntimeError
        If feature extraction fails due to decoding errors, invalid audio, or
        downstream library issues.

    Notes
    -----
    **Framewise CSV** (written when `include_framewise=True`)  
    One row per short-time frame with: `frame_index`, `time_s`, `f0_hz`,
    `f1_hz`–`f4_hz`, `loudness_db`, `hnr_db`. In per-turn mode, also includes
    `segment_index`, `start_s`, `end_s`, and any `extra_id_cols` present.

    **Summary CSV** (always written)  
    Whole-file: one row.  
    Per-turn (no `group_by`): one row per interval.  
    Per-turn with `group_by`: one row per group (e.g., per speaker).  
    Columns include summary stats of framewise series (on all frames or voiced
    segments ≥ `summarize_on_voiced_segments_ms`), silence ratio, jitter/shimmer
    variants, MFCC means/variances, CPP (if available), optional tremor/glottal
    metrics, and any `extra_id_cols`/`group_by` columns.

    Performance
    -----------
    Per-turn analysis can be I/O intensive for long files with dense transcripts.
    Tremor/glottal metrics are substantially more expensive than *simple* mode.

    Examples
    --------
    Whole-file analysis with framewise output:

    >>> analyze_acoustics(
    ...     wav_path="session.wav",
    ...     out_dir="features/acoustics",
    ... )

    Per-turn analysis aggregated by speaker:

    >>> analyze_acoustics(
    ...     wav_path="session.wav",
    ...     transcript_csv="transcripts/session.csv",
    ...     time_unit="ms",
    ...     group_by=["speaker"],
    ...     extra_id_cols=["source", "speaker"],
    ...     out_summary_csv="features/acoustics/session_by_speaker.csv",
    ...     summarize_on_voiced_segments_ms=100,
    ...     mode="simple",
    ... )
    """

    if mode in ("tremor", "advanced") and not tremor_script:
        # we check this before touching any audio. the docstring has always
        # said the script is required for these modes, but the per-clip code
        # used to quietly return the simple set instead: someone who picked
        # "tremor" got no tremor columns and not a word about it, after the
        # whole run.
        raise ValueError(
            f"mode={mode!r} needs tremor_script=<path to a Praat tremor "
            f"script>; without one there are no tremor metrics to compute. "
            f"Use mode='simple' or supply the script.")
    if wav_path is None:
        raise ValueError("wav_path is required")

    wav_path = Path(wav_path)
    if out_dir is None:
        out_dir = Path("features") / "acoustics"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # default output paths
    stem = wav_path.stem
    if out_framewise_csv is None:
        out_framewise_csv = out_dir / f"{stem}_framewise.csv"
    else:
        out_framewise_csv = Path(out_framewise_csv)

    if out_summary_csv is None:
        suffix = "_by_" + "_".join(group_by) if (transcript_csv and group_by) else ""
        out_summary_csv = out_dir / f"{stem}_summary{suffix}.csv"
    else:
        out_summary_csv = Path(out_summary_csv)

    # if it's already there and we weren't told to overwrite, we're done
    if (not overwrite_existing) and out_summary_csv.exists():
        if verbose:
            print(f"[acoustics] Summary output already exists; returning "
                  f"existing file: {out_summary_csv}")
        return {"framewise_csv": out_framewise_csv if out_framewise_csv.exists() else None,
                "summary_csv": out_summary_csv}

    # this is where the magic happens
    if transcript_csv:
        framewise_df, summary_df = _analyze_turns(
            wav_path=wav_path,
            transcript_csv=transcript_csv,
            time_unit=time_unit,
            group_by=group_by,
            extra_id_cols=extra_id_cols,
            mode=mode,
            summarize_on_voiced_segments_ms=summarize_on_voiced_segments_ms,
            include_framewise=include_framewise,
            tremor_script=tremor_script,
            f0_min=f0_min,
            f0_max=f0_max,
            n_mfcc=n_mfcc,
            preprocess=preprocess,
            target_sr=target_sr,
            target_dbfs=target_dbfs,
            remove_dc=remove_dc,
            pause_top_db=pause_top_db,
            pause_frame_length=pause_frame_length,
            pause_hop_length=pause_hop_length,
            on_progress=on_progress,
        )
    else:
        announce(on_progress, "measuring the recording")
        fdf, summ = _analyze_clip(
            wav_path,
            mode=mode,
            summarize_on_voiced_segments_ms=summarize_on_voiced_segments_ms,
            include_framewise=include_framewise,
            tremor_script=tremor_script,
            f0_min=f0_min,
            f0_max=f0_max,
            n_mfcc=n_mfcc,
            preprocess=preprocess,
            target_sr=target_sr,
            target_dbfs=target_dbfs,
            remove_dc=remove_dc,
            pause_top_db=pause_top_db,
            pause_frame_length=pause_frame_length,
            pause_hop_length=pause_hop_length,
        )
        # build the summary DF; extra_id_cols could ride along here later if we
        # ever pull them out of the filename
        summary_df = pd.DataFrame([summ])
        framewise_df = fdf

    # write the outputs -- atomically, like every other skip-if-exists table: a
    # Ctrl-C mid-write used to leave a truncated summary behind that the next
    # run then handed back as finished.
    frame_path_out: Optional[Path] = None
    if include_framewise and framewise_df is not None:
        with atomic_write(out_framewise_csv, mode="w", newline="",
                          encoding="utf-8-sig") as fh:
            framewise_df.to_csv(fh, index=False)
        frame_path_out = out_framewise_csv

    with atomic_write(out_summary_csv, mode="w", newline="",
                      encoding="utf-8-sig") as fh:
        summary_df.to_csv(fh, index=False)

    return {"framewise_csv": frame_path_out, "summary_csv": out_summary_csv}


# -------------------------
# CLI
# -------------------------

def main():
    import argparse
    from ..helpers.cliargs import add_bool_argument

    BoolOpt = argparse.BooleanOptionalAction

    ap = argparse.ArgumentParser(
        description="Extract PRAAT/Parselmouth acoustic features (simple / tremor / advanced)"
    )
    ap.add_argument("--wav", dest="wav_path", required=True, help="Input WAV file (mono or stereo)")
    ap.add_argument("--transcript-csv", default=None, help="Optional transcript CSV for per-turn analysis")
    ap.add_argument("--time-unit", choices=["ms","s"], default="ms",
                    help="Units for start/end columns if transcript is provided")
    ap.add_argument("--group-by", nargs="*", default=None,
                    help="Aggregate per-turn summaries by these columns (e.g., speaker)")
    ap.add_argument("--pass-through", dest="extra_id_cols", nargs="*", default=["source","speaker"],
                    help="Metadata columns to carry into outputs")

    ap.add_argument("--out-dir", default=None, help="Output folder (default: ./features/acoustics)")
    ap.add_argument("--out-framewise-csv", default=None,
                    help="Optional explicit path for framewise CSV (defaults to <stem>_framewise.csv)")
    ap.add_argument("--out-summary-csv", default=None,
                    help="Optional explicit path for summary CSV (defaults to <stem>_summary[...].csv)")
    add_bool_argument(ap, "--overwrite_existing", default=False,
                      help="Recompute and overwrite outputs that already exist")

    ap.add_argument("--mode", choices=["simple","tremor","advanced"], default="simple")
    add_bool_argument(ap, "--voiced-segments", dest="voiced_segments", default=True,
                      help="Summaries on voiced segments ≥100ms (true/false)")
    ap.add_argument("--f0_min", type=float, default=75.0)
    ap.add_argument("--f0_max", type=float, default=500.0)
    ap.add_argument("--n_mfcc", type=int, default=14)
    ap.add_argument("--tremor_script", default=None,
                    help="Path to Praat tremor script (for 'tremor'/'advanced')")
    add_bool_argument(ap, "--preprocess", default=True,
                      help="OpenWillis-style preprocessing (resample/DC/normalize). true/false (default: true)")
    ap.add_argument("--target_sr", type=int, default=44100)
    ap.add_argument("--target_dbfs", type=float, default=-20.0)
    add_bool_argument(ap, "--remove_dc", default=True,
                      help="Remove DC offset during preprocessing (default: true)")
    ap.add_argument("--pause_top_db", type=int, default=30,
                help="VAD non-silence threshold (dB below peak). Default: 30")
    ap.add_argument("--pause_frame_length", type=int, default=2048,
                    help="Frame length (samples) for VAD. Default: 2048")
    ap.add_argument("--pause_hop_length", type=int, default=512,
                    help="Hop length (samples) for VAD. Default: 512")


    ap.add_argument(
        "--framewise",
        action=BoolOpt,
        default=True,
        help="Write framewise CSV (default: true). Use --no-framewise to disable."
    )

    args = ap.parse_args()

    summarize_ms = 100 if args.voiced_segments else None
    preproc = args.preprocess
    remdc = args.remove_dc

    res = analyze_acoustics(
        wav_path=args.wav_path,
        transcript_csv=args.transcript_csv,
        time_unit=args.time_unit,
        group_by=args.group_by,
        extra_id_cols=args.extra_id_cols,
        out_dir=args.out_dir,
        out_framewise_csv=args.out_framewise_csv,  # honored if given
        out_summary_csv=args.out_summary_csv,
        overwrite_existing=args.overwrite_existing,
        include_framewise=args.framewise,          # the on/off toggle
        mode=args.mode,
        summarize_on_voiced_segments_ms=summarize_ms,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        n_mfcc=args.n_mfcc,
        tremor_script=args.tremor_script,
        preprocess=preproc,
        target_sr=args.target_sr,
        target_dbfs=args.target_dbfs,
        remove_dc=remdc,
        pause_top_db=args.pause_top_db,
        pause_frame_length=args.pause_frame_length,
        pause_hop_length=args.pause_hop_length,
    )
    print("[acoustics] Wrote:", res)


if __name__ == "__main__":
    main()
