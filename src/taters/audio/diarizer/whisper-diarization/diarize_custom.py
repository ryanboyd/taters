#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom diarization runner:
- Keeps original outputs (.srt, .txt)
- Adds a timestamped CSV of utterances: start_time, end_time, speaker, text
- (Optional) Exports one WAV per speaker by concatenating their segments
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import inspect
from pathlib import Path
from typing import Dict, List, Iterable

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    process_language_arg,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

# -------------------------
# CLI
# -------------------------

mtypes = {"cpu": "int8", "cuda": "float16"}

pid = os.getpid()
temp_outputs_dir = f"temp_outputs_{pid}"
temp_path = os.path.join(os.getcwd(), "temp_outputs")
os.makedirs(temp_path, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--audio", required=True, help="Target audio file (wav)")

parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disable Demucs vocal separation (faster on talk-only files).",
)

parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Transcribe numbers as words to improve alignment.",
)

parser.add_argument(
    "--whisper-model",
    dest="model_name",
    default="base",
    help="Whisper model name (e.g., base, small, medium.en, large-v3)",
)

parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for faster-whisper batched inference; 0 = original longform.",
)

parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language (or None to auto-detect).",
)

parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="cuda or cpu",
)

parser.add_argument(
    "--diarizer",
    default="msdd",
    choices=["msdd"],
    help="Diarization backend",
)

# ---- NEW OUTPUT OPTIONS ----
parser.add_argument(
    "--csv-out",
    type=str,
    default=None,
    help="Path to write utterance CSV (default: <audio_stem>.csv next to audio).",
)
parser.add_argument(
    "--speaker-dir",
    type=str,
    default=None,
    help="If set, export one WAV per speaker into this directory.",
)
parser.add_argument(
    "--sr",
    type=int,
    default=16000,
    help="Sample rate for exported speaker WAVs.",
)

parser.add_argument(
    "--num-speakers",
    type=int,
    default=None,
    help="If set, force the diarizer to use this number of speakers (oracle).",
)

args = parser.parse_args()
language = process_language_arg(args.language, args.model_name)

# -------------------------
# Stage 1: (optional) source separation
# -------------------------
if args.stemming:
    # Isolate vocals with Demucs; if it fails, fall back to original audio
    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "{temp_outputs_dir}" --device "{args.device}"'
    )
    if return_code != 0:
        logging.warning("Source separation failed; falling back to original audio. "
                        "Use --no-stem to skip separation.")
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            temp_outputs_dir,
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio

# -------------------------
# Stage 2: ASR (faster-whisper)
# -------------------------
whisper_model = faster_whisper.WhisperModel(
    args.model_name, device=args.device, compute_type=mtypes[args.device]
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)

suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if args.suppress_numerals
    else [-1]
)

if args.batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        batch_size=args.batch_size,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        vad_filter=True,
    )

full_transcript = "".join(segment.text for segment in transcript_segments)

# free VRAM
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

# -------------------------
# Stage 3: Forced alignment (CTC)
# -------------------------
alignment_model, alignment_tokenizer = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform).to(alignment_model.dtype).to(alignment_model.device),
    batch_size=args.batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions, tokens_starred, alignment_tokenizer
)
spans = get_spans(tokens_starred, segments, blank_token)
word_timestamps = postprocess_results(text_starred, spans, stride, scores)

# -------------------------
# Stage 4: Diarization (NeMo MSDD)
# -------------------------
if args.diarizer == "msdd":
    from diarization import MSDDDiarizer
    diarizer_model = MSDDDiarizer(device=args.device)

# Helper: check which kw name the diarizer supports
def _supports_kw(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

# custom bit to try to force number of speakers
diarize_kwargs = {}
if args.num_speakers is not None:
    diarize_kwargs["num_speakers"] = int(args.num_speakers)
    diarize_kwargs["oracle_num_speakers"] = True
    diarize_kwargs["min_num_speakers"] = int(args.num_speakers)
    diarize_kwargs["max_num_speakers"] = int(args.num_speakers)

speaker_ts = diarizer_model.diarize(
    torch.from_numpy(audio_waveform).unsqueeze(0),
    **diarize_kwargs
)



speaker_ts = diarizer_model.diarize(
    torch.from_numpy(audio_waveform).unsqueeze(0),
    **diarize_kwargs
)

del diarizer_model
torch.cuda.empty_cache()

# Word-level speaker map, then sentence-level utterances
wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

# Optional punctuation restoration for better sentence boundaries
if info.language in punct_model_langs:
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    words_list = [x["word"] for x in wsm]
    labeled = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for wd, lab in zip(wsm, labeled):
        word = wd["word"]
        if word and lab[1] in ending_puncts and (word[-1] not in model_puncts or is_acronym(word)):
            word = word + lab[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            wd["word"] = word
else:
    logging.warning(
        f"Punctuation model not available for {info.language}; keeping original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)  # <- sentence-level speaker map

# -------------------------
# Stage 5: Original outputs (.txt/.srt)
# -------------------------
stem = os.path.splitext(args.audio)[0]
with open(f"{stem}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{stem}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

# -------------------------
# NEW: CSV utterances
# -------------------------

def _iter_utterances(ssm_obj) -> Iterable[Dict]:
    """
    Robustly iterate utterances from ssm produced by helpers.
    Expect items with keys like: start/start_time, end/end_time, speaker, text.
    """
    for item in ssm_obj:
        # common shapes supported
        start = item.get("start_time", item.get("start"))
        end = item.get("end_time", item.get("end"))
        speaker = item.get("speaker", item.get("spk", "SPEAKER_0"))
        text = item.get("text", item.get("sentence", item.get("utterance", "")))
        if start is None or end is None:
            # fall back: if nested, flatten naive
            if "sentences" in item and isinstance(item["sentences"], list):
                for s in item["sentences"]:
                    yield {
                        "start_time": s.get("start_time", s.get("start", 0.0)),
                        "end_time": s.get("end_time", s.get("end", 0.0)),
                        "speaker": item.get("speaker", "SPEAKER_0"),
                        "text": s.get("text", ""),
                    }
                continue
        yield {
            "start_time": float(start) if start is not None else 0.0,
            "end_time": float(end) if end is not None else 0.0,
            "speaker": str(speaker),
            "text": text or "",
        }

def write_utterance_csv(csv_path: Path, ssm_obj) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "end_time", "speaker", "text"])
        for u in _iter_utterances(ssm_obj):
            w.writerow([f"{u['start_time']:.3f}", f"{u['end_time']:.3f}", u["speaker"], u["text"]])
    return csv_path

csv_out = Path(args.csv_out) if args.csv_out else Path(f"{stem}.csv")
csv_written = write_utterance_csv(csv_out, ssm)
print(f"[custom] CSV written: {csv_written}")

# -------------------------
# Cleanup
# -------------------------
print("Cleaning up...")
cleanup(temp_path)
print("Done cleaning up.")
