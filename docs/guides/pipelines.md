# Pipelines

Every other page in these guides answers "what can Taters measure?". This one
answers "how do I make it happen the same way twice?".

A pipeline is a recipe: a short file that says, in order, what should happen
to your data. *Take every video in this folder. Pull the audio out.
Transcribe it. Score the transcripts with my dictionaries. Merge everything
into one table.* Write that down once and the whole analysis becomes a single
command — today, next month when a reviewer asks for a change, and on your
collaborator's machine.

That last part is the real reason to care. A pipeline file **is** your methods
section, in executable form: every step, every setting, every model name, in
plain text you can version, share and rerun. When somebody asks "exactly how
were these features produced?", the honest answer is a file rather than a
memory of what you clicked eight months ago.

> **You probably don't need to write one.** Run `taters`, answer its
> questions, and [the app](wizard.md) writes the pipeline file for you — then
> offers to run it. Everything below is what that file *means*, for the day
> you want to read one, change one, or build one by hand.

---

## Why a file, when the app already works?

Clicking through the app is faster the first time. The file wins every time
after that, and for four reasons worth separating:

* **You will run it again.** New participants arrive, a transcript gets
  fixed, someone spots a bad recording. With a file you rerun one command
  instead of remembering twenty answers.
* **You will change one thing.** A pipeline makes "the same analysis, but
  with a different dictionary" a one-line edit, and makes it obvious to a
  reader exactly what differed between two runs.
* **Somebody else will need to run it.** A colleague, a reviewer, you in two
  years. Data plus pipeline file plus Taters version reproduces the analysis;
  a folder of output files does not.
* **It scales past your patience.** Two hundred videos is the same file as
  two, and the run keeps going while you do something else.

---

## The format of a pipeline file

A pipeline file is YAML — a plain-text format where indentation shows
structure — and it has two parts.

**`vars:`** holds the settings that might change between runs: which device
to use, which Whisper model, whether to overwrite existing outputs. Anything
here can be overridden when you run it, without touching the file.

**`steps:`** is the recipe itself, run top to bottom. Each step names a
Taters function and the inputs to hand it.

Here is a small, real one:

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

{% endraw %}

That is a complete pipeline. Three steps, and the ideas in it cover almost
everything you will meet in a longer one.

---

## Item and global: yup, it's a big difference

Every step runs in one of
two **scopes**, and which one it is changes what "run this step" means.

**`scope: item`** runs the step once *per input file*. Convert this video.
Transcribe this recording. If you point the pipeline at two hundred videos,
an item step happens two hundred times — several at once, if you allow it.
Inside an item step, {% raw %}`{{input}}`{% endraw %} means "the file being
worked on right now".

**`scope: global`** runs the step once for *the whole job*, after the item
steps that feed it. Merge every per-file transcript into one table. Build a
frequency list across the corpus. Fit a topic model on everything.

The reason this matters is that it decides what a step can see. An item step
sees one file and knows nothing about the others, so it cannot compute
anything corpus-wide — a word's frequency across your whole dataset is not a
fact about any single document. A global step sees everything, which is why
every "and now put it all together" step is global.

Most pipelines are a fan-out followed by a merge: item steps widen to do the
same work on every file, then global steps narrow back down to one table.

---

## How steps hand things to each other

Steps talk through **artifacts**. A step can save its result under a name:

{% raw %}
```yaml
    save_as: wav
```
{% endraw %}

and any later step can use it by putting that name in double curly braces:

{% raw %}
```yaml
      audio_path: "{{wav}}"
```
{% endraw %}

That is the whole trick, and it is why a pipeline is editable by a human:
outputs flow into inputs *by name*, so reordering steps, swapping one for
another, or inserting a new one is editing text rather than rewriting code.
Nothing hardcodes a path.

Four expressions get filled in when the pipeline runs:

{% raw %}

| Expression | Means |
|---|---|
| `{{input}}` | The file currently being processed (item steps only) |
| `{{wav}}` | Whatever the step named `wav` produced |
| `{{var:device}}` | A value from the `vars:` block, or your override |
| `{{pick:name.some.field}}` | Reaches inside a saved artifact, when a step returns more than one thing |

{% endraw %}

---

## Running a pipeline

```bash
python -m taters.pipelines.run_pipeline \
  --preset-file my_pipeline.yaml \
  --root_dir ./videos --file_type video \
  --var device=cuda
```

`--root_dir` and `--file_type` tell the item steps what to fan out over; a
pipeline made only of global steps needs neither. Every `--var key=value`
overrides the file's `vars:`, which is how the same pipeline runs on your
laptop with `device=cpu` and on the lab machine with `device=cuda` without
anybody editing anything. `--vars-file` does the same from a YAML file when
there are several, and `--workers` sets how much runs at once.

Two ready-made pipelines ship with Taters. `--list-presets` shows them and
`--describe-preset <name>` prints what one does, what it needs and every
variable it takes:

* **`conversation_video`** — for recordings with more than one person in
  them. Pulls the audio, works out who spoke when, transcribes per speaker,
  and measures the language of each.
* **`single_speaker_media`** — for everything else: one voice, one recording,
  straight to a transcript and its features.

Run either with `--preset conversation_video` instead of `--preset-file`.

---

## When something goes wrong

Runs on real data hit real problems — one corrupt video, one empty PDF — and
a pipeline treats those as data points rather than disasters:

* If an **item** step fails on one file, that file is marked failed and the
  run carries on with the rest. One bad recording out of two hundred does not
  cost you the other hundred and ninety-nine.
* If a **global** step fails, the run stops and makes a whole bunch of noise.
  A merge that half worked is worse than one that did not run.
* By default, steps refuse to overwrite existing outputs, so rerunning an
  interrupted pipeline picks up roughly where it stopped instead of starting
  over. Set `overwrite_existing` when you actually want it redone.

The exit code is non-zero if anything failed, which makes pipelines safe to
put in scripts and schedulers.

---

## The manifest: a log of what actually ran

Every run writes a **manifest** — a JSON file, `./run_manifest.json` unless
you pass `--out-manifest` — recording every input, every setting, what
succeeded, what failed and why, and where each output landed. It is updated
as the run goes, so a long job can be watched rather than waited on.

This is the file to read when you ask "what exactly happened here?", and it
is the other half of the reproducibility story. The pipeline file says what
you *intended*; the manifest says what actually took place, including the
values your overrides supplied and the failures you would otherwise never
hear about.

When you are debugging, read it first. It usually already says what went
wrong and for which file, which saves rerunning anything to find out.

---

## Good habits

* **Start from the app.** Let `taters` write the file, then open it. It is
  the fastest way to learn the format, because it is describing choices you
  just made and can still remember.
* **Keep the pipeline file with the project**, next to the data. It is the
  reproducibility artifact: data + pipeline + Taters version = the analysis.
* **Change settings through `vars`, not by editing steps.** The file stays
  stable, the diffs stay readable, and the manifest records which values were
  actually used.
* **Copy before you experiment.** The two shipped pipelines are read-only on
  purpose; copy one and edit the copy, so there is always a working version
  to go back to.
* **Read the manifest before debugging anything else.**

---

## Where to go next

* [The Taters app](wizard.md) — the thing that writes these files for you.
* [Analyzing Text](analyzing-text.md) and [Analyzing Audio](analyzing-audio.md)
  — what the individual steps actually measure.
* [Running Analyses](running-analyses.md) — what to do with the table at the
  end of it.
