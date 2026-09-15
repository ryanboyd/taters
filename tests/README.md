# Taters test suite

## Running it

```bash
pip install -e ".[dev]"     # once: installs taters + pytest
pytest                      # the whole fast suite, about 3-4 minutes
```

About 3,000 tests, of which all but a few dozen run by default. Most take milliseconds; the
time goes on a handful of unavoidable imports (`torch` takes 13 seconds to
import, `textstat` 7, `nltk` 6) and on the tests that run a whole pipeline end
to end. Those imports are paid **once per process**, which is why running the
files one at a time adds up to more than running them together.

Anything that needs a real neural model — a Stanza pipeline, a Whisper
transcription, a sentence-transformer — is marked `slow` and runs only when
you ask for it. That is the whole reason the default suite is minutes rather
than hours, and it is a real division rather than a way of hiding failures:
`pytest -m slow` runs exactly those tests and nothing else.

**If it takes much longer than ten minutes, something else on your machine is
competing for the CPU.** The suite is almost entirely CPU-bound, so a build,
another test run, or a video export happening at the same time can easily
double or triple it. Check with `pgrep -af pytest` before concluding the suite
itself got slower.

A single test that runs for more than five minutes is treated as a hang, not as
slowness: pytest prints the stack of every thread -- so you can see *which*
test stopped and where -- and then aborts the run. That is deliberate, and it is
there because a stalled run once sat silent for over an hour. The heavy tests
can legitimately exceed it, so `pytest -m slow` runs should turn it off with
`-o faulthandler_timeout=0`.

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
| `pytest -m slow -o faulthandler_timeout=0` | **only** the heavy tests (real media + models) |
| `pytest --durations=20` | which twenty tests took longest |
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
| `study_csv` | a study spreadsheet: `pid,condition,openness,text` (the wizard's analysis stage) |
| `survey_csv` | a two-row spreadsheet with a `response` text column and a `condition` |
| `essays` | a folder of three `.txt` documents |
| `media` | a folder with one tiny video, for a wizard run over media (needs ffmpeg) |
| `stanza_ready` | the Stanza English model, downloaded once per session (slow tests) |
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

## Shared helpers

Plain modules next to the tests, imported by name (the tests folder is on
`sys.path`):

| module | what it holds |
|---|---|
| `csvhelpers.py` | `read_rows`, `write_rows`, `write_table` — the CSV round trip every results test needs. The old names `_read`, `_write`, `_write_table` are aliases. |
| `wizard_helpers.py` | `browse_to`, `run`, `EscapingPrompter`, the tuning drivers and the `clean_machine` fixture that makes the wizard's question count independent of what is installed. Import `clean_machine` into a wizard test module to activate it. |
| `preset_checks.py` | static checks over a composed pipeline; `assert_valid_preset(preset)` fails naming every problem. |
| `mutcheck.py` | not a test: `python tests/mutcheck.py <src> <tests> <old> <new>` breaks the code the way named and reports whether the tests notice. Every behavior change ships with a test this has said CATCHES IT about. |
| `e2e_wizard.py` | not a test: `python tests/e2e_wizard.py [--media clip.mkv] [--flows A F] [--keep]` answers the wizard the way a person would, runs what it composed with the real analyzers, re-runs it from the command line, and scores a second study with the model it fitted. Minutes, not seconds; run it before a release. It found five gaps the unit tests had missed. |

The wizard tests are split by stage: `test_wizard_flow.py` (start to finish,
preflight, saving, running), `test_wizard_sources.py` (folders, spreadsheets,
the level question), `test_wizard_options.py` (the options screen and Esc
inside it) and `test_wizard_analysis.py` (the analysis stage, the two
extraction flows, and Esc meaning the previous question everywhere).

## About `./test_vids`

The slow tests pick the **smallest** file there that genuinely has an audio
stream, then cut 30 seconds from it. Video-only downloads (yt-dlp's `.f137.mp4`
without a merge) are skipped automatically, and if nothing usable is present
the slow tests skip with an explanation rather than failing.
