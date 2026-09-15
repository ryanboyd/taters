# Pipelines

A pipeline is a recipe: a short file that says, in order, what should happen
to your data. "Take every video in this folder. Pull the audio out.
Transcribe it. Score the transcripts with my dictionaries. Merge everything
into one table." Write that down once and the whole analysis becomes a
single command — today, next month when the reviewer asks for a change, and
on your collaborator's machine.

That last part is the real reason to care. A pipeline file *is* your methods
section, in executable form: every step, every setting, every model name, in
a plain-text file you can version, share, and rerun. When someone asks
"exactly how were these features produced?", the honest answer is a file,
not a memory.

## You probably don't need to write one

Run `taters`, answer its questions, and [the app](wizard.md) writes the
pipeline file for you — then offers to run it. Everything on this page is
what that file *means*, for the day you want to read one, tweak one, or
build one by hand.

## The anatomy, in plain words

A pipeline file (YAML — a format where indentation shows structure) has two
parts:

* **`vars:`** — the settings that might change between runs: which device to
  use, which Whisper model, whether to overwrite existing outputs. Anything
  here can be overridden at run time without touching the file.
* **`steps:`** — the recipe itself, run top to bottom. Each step names a
  Taters function and its inputs. Steps come in two scopes:
    * **item** steps run once *per input file* — convert this video,
      transcribe this recording. If you have 200 videos, an item step runs
      200 times (several at once, if you allow it).
    * **global** steps run *once for the whole job* — merge all the
      per-file outputs into one table, build a corpus-wide frequency list,
      fit a topic model.

The steps talk to each other through **artifacts**: a step can save its
result under a name (`save_as: wav`), and a later step can use it by putting
that name in double curly braces. That is the whole trick — outputs flow
into inputs by name, so reordering or swapping steps is editing text, not
rewriting code.

Here is a small, real one, annotated:

{% raw %}

```yaml
vars:
  device: "auto"              # cpu, cuda, or let Taters decide
  whisper_model: "base.en"

steps:
  # For every video: pull the audio out.
  - scope: item
    call: potato.audio.convert_to_wav
    save_as: wav                        # <- later steps can say {{wav}}
    with:
      input_path: "{{input}}"           # {{input}} = the current file

  # For every WAV: transcribe it.
  - scope: item
    call: potato.audio.transcribe_with_whisper
    save_as: transcript
    with:
      audio_path: "{{wav}}"             # <- the artifact from step 1
      whisper_model: "{{var:whisper_model}}"
      device: "{{var:device}}"

  # Once, at the end: merge every transcript into one table.
  - scope: global
    call: taters.helpers.feature_gather.feature_gather
    with:
      root_dir: "./transcripts"
      pattern: "*.csv"
```

The curly-brace expressions are filled in at run time: `{{input}}` is the
file currently being processed, `{{var:device}}` reads from the `vars:`
block (or your command-line override), `{{wav}}` is whatever the step named
`wav` produced, and `{{pick:name.some.field}}` reaches inside a saved
artifact when a step returns more than one thing.

{% endraw %}

## Running one

```bash
python -m taters.pipelines.run_pipeline \
  --preset-file my_pipeline.yaml \
  --root_dir ./videos --file_type video \
  --var device=cuda
```

`--root_dir` and `--file_type` tell the item steps what to fan out over
(a pipeline with only global steps needs neither). Any `--var key=value`
overrides the file's `vars:` — so the same pipeline runs on the laptop
(`device=cpu`) and the lab machine (`device=cuda`) without editing anything.
Two ready-made pipelines ship with Taters (`--list-presets` shows them):
`conversation_video` for multi-speaker recordings and `single_speaker_media`
for everything else.

## What happens when something goes wrong

Runs on real data hit real problems — one corrupt video, one empty PDF —
and a good pipeline treats those as data points, not disasters:

* If an **item** step fails on one file, that file is marked as failed and
  the run continues with the rest.
* If a **global** step fails, the run stops, loudly.
* Either way, the runner writes a **manifest** — a JSON file recording every
  input, every setting, what succeeded, what failed and why, and where every
  output landed. It is updated as the run goes, so a long job can be
  monitored, and it is the file to read when you ask "what exactly ran
  here?" The exit code is non-zero if anything failed, so pipelines are safe
  to put in scripts and schedulers.
* By default, steps refuse to overwrite existing outputs — so rerunning an
  interrupted pipeline picks up roughly where it left off instead of
  starting over.

## Habits that pay off

* **Start from the app.** Let `taters` write the file, then read it. It is
  the fastest way to learn the format, because it is describing choices you
  just made.
* **Keep the pipeline file with the project.** It is the reproducibility
  artifact: data + pipeline file + Taters version = the analysis.
* **Change settings through `vars`, not by editing steps** — the file stays
  stable, and the run manifest records which values were actually used.
* **Read the manifest before debugging anything else.** It usually already
  says what went wrong, and for which file.
