<p align="center">
  <img src="https://github.com/ryanboyd/taters/blob/main/img/taters-small.png?raw=true" alt="Taters!"/>
</p>


# 🥔 **TATERS**: Takes All Things, Extracts Relevant Stuff

Taters is a Python toolkit and CLI for getting from raw media to analysis-ready data. Point it at video, audio, or text and it will extract WAV from video, transcribe it (with or without diarization), compute embeddings, run dictionary and archetype analyses, and gather the results into tidy datasets you can model or visualize.

* 🥔 Documentation: **[https://www.taters.wiki](https://www.taters.wiki)**
* 🥔 Status: early but usable. APIs will probably evolve; pin a version if you need stability.

---

## What Taters is (and is not)

* **Is:** A library + CLI with small, composable functions and an optional YAML pipeline runner. Predictable I/O, friendly defaults, and “do not overwrite unless asked.”
* **Is not:** A single black-box pipeline. You keep control of each step and can run pieces à la carte or all at once.
* **Is not:** Edible.

---

## A short example

### Python

```python
from taters import Taters
t = Taters()

# Pull audio from video
wavs = t.audio.extract_wavs_from_video(input_path="input.mp4")

# Transcribe (CSV/SRT/TXT). Swap in diarize_with_thirdparty for multi-speaker
# recordings — it returns the same shape, so nothing below changes.
asr = t.audio.transcribe_with_whisper(audio_path=wavs[0], device="auto")
transcript = asr.raw_files["csv"]    # also: asr.raw_files["srt"] / ["txt"]

# Features (defaults write under ./features/<kind>/)
t.audio.extract_whisper_embeddings(source_wav=wavs[0], transcript_csv=transcript)
t.text.analyze_with_dictionaries(csv_path=transcript, dict_paths=["dictionaries/liwc"])
t.text.analyze_with_archetypes(csv_path=transcript, archetype_csvs=["archetypes/Resilience.csv"])
```

### CLI

```bash
# Transcribe a single-speaker recording
python -m taters.audio.transcribe_with_whisper \
  --audio_path audio/lecture.wav --whisper_model small.en

# Whisper embeddings over non-silent spans, then mean-pool
python -m taters.audio.extract_whisper_embeddings \
  --source_wav audio/session.wav --strategy nonsilent --aggregate mean
```

For more examples, including per-speaker splits, sentence embeddings, and end-to-end pipelines, see the Guides in the documentation.

---

## Installation

Install into a fresh virtual environment. The install guide covers CPU and CUDA setups, FFmpeg, and the optional diarization extras:

**[https://www.taters.wiki/install-guide](https://www.taters.wiki/install-guide)**

---

## Pipelines

To batch a whole dataset, use the YAML runner to chain steps and control concurrency:

```bash
python -m taters.pipelines.run_pipeline \
  --root_dir videos --file_type video \
  --preset conversation_video \
  --workers 8 --var device=cuda
```

Details, presets, and how to write your own:

**[https://www.taters.wiki/guides/pipelines/](https://www.taters.wiki/guides/pipelines/)**

---

## Contributing

Bug reports and pull requests are welcome. If you are using Taters on real projects, feedback on rough edges and missing presets is especially useful.

---

## License

MIT. See `LICENSE` for details.