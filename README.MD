<p align="center">
  <img src="https://github.com/ryanboyd/taters/blob/main/img/taters-small.png?raw=true" alt="Taters!"/>
</p>


# **TATERS**: Takes All Things, Extracts Relevant Stuff

Taters is a broad-scope toolkit for researchers that can be used to extract features from multiple types of data (video, audio, text) into clean, analysis-ready artifacts and features. Think of it as a small, dependable kitchen crew for your data: you bring potatoes (files), it handles the peeling, chopping, and plating.

**Status:** active WIP. It works today, but expect some rough edges and breaking changes as the project grows.

---

## What Taters is (and isn't)

Taters is **a library and a CLI** for end-to-end A/V + text processing with predictable outputs. It's **not** a monolithic "black box" pipeline — each step is a clear, reusable function you can run on its own or string together with YAML presets.

---

### What you can do with it (high level)

With Taters, the goal is to have relatively standard (and standardized) processing pipelines for different types of data commonly used (or increasingly used) in the computational social sciences. If you want to extract language features from text data, you can do that. If you want to take audio features, have them machine-transcribed, and *then* extract language features, you can do that too. If you have video files where you want to extract the audio, then transcribe it, and *then* extract time-stamped language features (e.g., sentence embeddings, dictionary-based coding, etc.)... you can do that too. In essence, the goal of taters is to take a lot of the tedious tasks that are common for getting from raw data to analysis-ready data and stitch them together to save everyone a huge amount of time, energy, and spare us all the learning curve that goes along with that — all while keeping the advanced/super-user abilities of these methods available to even the most hardcore nerds.

Note: everything below is currently implemented, but is highly subject to change as the project evolves. There will be a lot of things to come as the project grows/changes over time.

* **Pull audio from video**: extract one or more WAV streams from containers.
* **Diarize + transcribe**: wrap a proven third-party stack to produce per-recording CSV/SRT/TXT.
* **Per-speaker WAVs**: build one WAV per speaker from a transcript CSV.
* **Embeddings**

  * **Whisper encoder embeddings** (segment-level from a transcript or general audio without one).
  * **Sentence embeddings** (mean per row) for any text dataset.
* **Text gatherer**: stream CSVs or folders of `.txt` into a single “analysis-ready” CSV, with optional grouping.
* **Feature extraction**

  * **Dictionary coding** across any number of ContentCoder dictionaries → one wide CSV with stable column order.
  * **Archetype scoring** with sentence-transformers → tidy, fixed columns.
* **Predictable outputs**: if you don't specify a path, Taters writes to `./features/<kind>/<filename>.csv`, where `<filename>` reflects how the text was gathered (e.g., grouped vs. concatenated).

---

## How you'll use it

### Python (quick sketch)

```python
from taters import Taters
t = Taters()

# 1) Audio from video
wavs = t.audio.extract_wavs_from_video(input_path="input.mp4")

# 2) Diarize (CSV/SRT/TXT)
diar = t.audio.diarize_with_thirdparty(audio_path=wavs[0], device="cuda")

# 3) Features (defaults write under ./features/<kind>/)
t.audio.extract_whisper_embeddings(source_wav=wavs[0], transcript_csv=diar["csv"])
t.text.analyze_with_dictionaries(csv_path=diar["csv"], dict_paths=["dicts/LIWC-22.dicx"])
t.text.analyze_with_archetypes(csv_path=diar["csv"], archetype_csvs=["archetypes/Resilience.csv"])
t.text.extract_sentence_embeddings(csv_path=diar["csv"], text_cols=["text"], id_cols=["speaker"], group_by=["speaker"])
```

### CLI (quick sketch)

Virtually every aspect of Taters is directly callable via the CLI. A huge to-do for the project is to develop comprehensive documentation of each facet of Taters, making it obvious/easy to determine which scripts accomplish what tasks, what information/parameters they require, etc. However, in the meantime, one of the major goals of Taters is to make everything relatively easy to get into — submodules/scripts use helpful argument parsers to give you all of the information you need to run any given function from the command line. A few examples of what this can look like:

```bash
# Diarize
python -m taters.audio.diarize_with_thirdparty \
  --audio_path audio/session.wav --device cuda

# Whisper embeddings (general audio; non-silent spans + mean pool)
python -m taters.audio.extract_whisper_embeddings \
  --source_wav audio/session.wav --strategy nonsilent --aggregate mean

# Gather text from CSV (auto names the output if --out omitted)
python -m taters.helpers.text_gather \
  --csv transcripts/session.csv --text-col text --group-by speaker --delimiter ,
```

What's really nice is that each function of Taters can be bolted together into full-blown pipelines for batch processing. Once a pipeline is built, you can do zillions of things to your dataset with a single, simple line of code. If you know what parameters you use to run any given function, your pipeline can/will take the same parameters for ultimate reproducibility. Well, "ultimate" is a bit of a judgement call, but it's pretty great.

Speaking of pipelines...

### Pipelines (do it all at once)

Presets live in YAML (e.g., `taters/pipelines/presets/`). Point at a dataset, choose a preset, and Taters will run the steps in order, using each step's output as the next step's input. You can override variables (like models, device, overwrite behavior) on the command line.

Pipeline presets are currently in an activate state of development and have the most room for expansion. More presets will be developed for different workflows but, on the bright side, it is already 100% possible to develop your own pipeline YAML file. Why would you want to spend your time doing anything else?

---

## Install (tidy version)

Use a fresh virtual environment. Seriously, a fresh virtual environment is strongly recommended.

```bash
python -m venv venv-taters
source venv-taters/bin/activate
```

### Quick path (when available)

```bash
pip install "taters[diarization,cuda]"
```

Then install the three git extras used by the diarization wrapper:

```bash
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

Install PyTorch built for **CUDA 12.4** (the stack ChopShop targets):

```bash
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

And ensure **FFmpeg** is on your `PATH` (Ubuntu: `sudo apt-get install ffmpeg`, macOS: `brew install ffmpeg`).

> Tip: If you hit CUDA/cuDNN loader errors, it usually means your runtime and wheel builds don't match. Keep CUDA **12.4**, `cu124` wheels, and cuDNN 9 aligned.

---

## Roadmap (short)

* More feature families
* More obviously composable pipelines (per-item + global phases, manifests, post-run aggregation).
* Rich gatherers/aggregators to unify outputs across large runs.
* Clear docs, examples, and ready-to-run presets.

If you try Taters on a real project, feedback on your flow and pain points is incredibly helpful.

---

## License & credits

MIT license. Built on top of excellent open-source projects (Faster-Whisper, sentence-transformers, ContentCoder, and an incredible [community diarization stack](https://github.com/MahmoudAshraf97/whisper-diarization).

*(Taters grew out of the earlier "ChopShop" prototype; many ideas and defaults carry over.)*

