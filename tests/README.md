# Taters test suite

## Running it

```bash
pip install -e ".[dev]"     # once: installs taters + pytest
pytest                      # the whole fast suite, a few seconds
```

If you want the `slow` tests too, install torch **before** Taters so
sentence-transformers does not pull a CPU-only build over your CUDA one:

```bash
python3 -m venv ~/.venvs/taters && source ~/.venvs/taters/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,readability,vocalacoustics,diarization]"
pip install git+https://github.com/MahmoudAshraf97/demucs.git \
            git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
            git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

(`cu128` is for Blackwell GPUs; see `docs/install-guide.md` for other cards.
Putting the venv outside the repo matters on WSL — pip on a `/mnt/c` mount is
extremely slow for packages with many small files.)

Useful variations:

| command | what it does |
|---|---|
| `pytest` | everything fast; `slow` tests are deselected |
| `pytest -v` | one line per test instead of a dot |
| `pytest -x` | stop at the first failure |
| `pytest --lf` | re-run only what failed last time |
| `pytest -k speaker` | only tests whose name contains "speaker" |
| `pytest tests/test_find_files.py` | one file |
| `pytest tests/test_find_files.py::test_video_group_matches_only_video_extensions` | one test |
| `pytest -m slow` | **only** the heavy tests (real media + models) |
| `pytest -m ""` | absolutely everything |

A dot is a pass, `F` is a failure, `s` is a skip, `E` is an error while setting
up. Failures print the source of the test, the values on both sides of the
failed `assert`, and anything the code printed.

## Reading a skip

A skip is not a failure — it means "this machine cannot check that". The two
common reasons:

* **`ffmpeg/ffprobe not found on PATH`** — install ffmpeg (`sudo apt install
  ffmpeg`). About 60 tests depend on it.
* **`needs an optional dependency: ...`** — that feature's extra is not
  installed (`pip install "taters[all]"` covers everything).

If you want the suite to be honest about what it actually verified, run
`pytest -rs`, which lists every skip and why.

## Layers

**Fast** (default). No models, no network, no repo media. Synthetic inputs
built on the fly: a three-second WAV, a two-second video, a handful of CSV
rows. These cover file discovery, text gathering, feature aggregation,
subtitle parsing, lexical-richness math, the templating engine, preset
validation, the facade, and the "don't overwrite unless asked" contract.

**ffmpeg** (`needs_ffmpeg`). Still fast, still synthetic, but shells out to
ffmpeg: WAV conversion, multi-track extraction, per-speaker splitting. They
assert on what ffprobe says about the output file, not on ffmpeg's exit code.

**Slow** (`slow`, opt-in). Real interview media from `./test_vids` and real
models: diarization, Whisper embeddings, vocal acoustics, and the full
`conversation_video` preset. Assertions are structural — column names, row
alignment, ordering, plausible ranges — never exact transcript text, which
changes legitimately between model versions.

## Adding a test

Put it in the file matching the module you are testing, name the function
`test_something_specific`, and use `assert`. If it writes files, do nothing
special: every test already runs inside its own empty temp directory.

```python
def test_my_new_thing(tmp_path):
    result = my_function(out_csv=tmp_path / "out.csv")
    assert Path(result).is_file()
```

Fixtures available everywhere (defined in `conftest.py`):

| fixture | what you get |
|---|---|
| `tmp_path` | an empty directory of your own (built into pytest) |
| `sandbox` | the working directory the test is running in — applied automatically |
| `repo_root` | absolute path to the project root |
| `tiny_wav` | a 3-second 16 kHz mono WAV (needs ffmpeg) |
| `tiny_video` | a 2-second video with one audio stream (needs ffmpeg) |
| `tiny_video_two_audio_streams` | a 2-second video with two tagged audio tracks |
| `transcript_csv` | a diarization-shaped `start_time,end_time,speaker,text` CSV |
| `analysis_ready_csv` | a `text_id,text` CSV |
| `real_audio_clip` | 30 seconds of real audio from `./test_vids` (slow tests) |

Mark anything heavy so it stays out of the default run:

```python
@pytest.mark.slow
@pytest.mark.needs_ffmpeg
def test_the_expensive_thing(real_audio_clip):
    ...
```

Markers must be registered in `pyproject.toml` — `--strict-markers` turns a
typo into an error rather than a silently ignored decoration.

## About `./test_vids`

The slow tests pick the **smallest** file there that genuinely has an audio
stream, then cut 30 seconds from it. Video-only downloads (yt-dlp's `.f137.mp4`
without a merge) are skipped automatically, and if nothing usable is present
the slow tests skip with an explanation rather than failing.
