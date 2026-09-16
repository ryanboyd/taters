# The Taters app

This is Taters. Install it — see the [install guide](../install-guide.md) —
then run:

```bash
taters
```

It walks you through choosing your data and what to measure, builds a pipeline
from your answers, saves it, and offers to run it. Nothing is written until you
say so.

You never have to write any Python. But everything the app does is also
available as a Python library and a set of command-line tools, so if you would
rather script it, chain steps together, or drop one of these analyses into
something bigger, none of that is hidden from you — the app writes an ordinary
pipeline file you can read, edit and re-run yourself. See
[Pipelines](pipelines.md) and the [API reference](../api/api-overview.md).

## What you can do

The opening question is what you want to do, not where your files are — several
of the answers do not involve new data at all.

| | |
|---|---|
| **Wrangle/Analyze data** | The two things you do to a spreadsheet you already have. *Wrangle data* collects text into a tidy spreadsheet — a folder of documents to CSV, or a CSV combined by any column(s). *Analyze data* runs the statistics over one: LIWC output from elsewhere, questionnaire scales, anything measured in another program. Tick which columns are the predictors and answer the same statistics questions; nothing is extracted. |
| **Extract features from my data** | Turn audio, video or text into measures. Builds a new pipeline. |
| **Extract features and run analyses** | The same, then compare groups, correlate with outcomes, or fit a prediction model. Needs a spreadsheet with those columns in it. |
| **Train a model** | Word vectors from your own texts, a language model adapted to them, or a transformer fine-tuned to predict your outcome columns — each saved with a report of how it was trained, kept in your library, and applied to new data from the checklist. Pre-trained word vectors can be brought in here too. |
| **Run a saved pipeline** | Re-run something you built earlier, or one of the pipelines that ship with Taters. |
| **Settings and tools** | Check whether your setup works, and manage everything Taters keeps between runs. |

Everything is chosen from a list. Move with the arrow keys and press enter, or
press the number beside an option to pick it outright. Yes/no questions take
`y`/`n` or `1`/`0` as single keypresses, <kbd>Esc</kbd> steps back exactly one
question if you change your mind (what you already answered stays filled in,
so changing one thing costs one keypress per question, not a re-typing of all of
them), and <kbd>Ctrl-C</kbd> leaves without writing anything.

Some screens let you pick *several* things — those rows wear `[ ]` boxes so
you can tell at a glance. <kbd>Space</kbd> ticks a box, and the **✓ Done**
row at the top is the only place <kbd>Enter</kbd> does anything — and even
there it waits until at least one box is ticked. A stray enter can never
quietly pick the wrong thing or skip you forward. On a list too long for the
screen, the ✓ Done row stays pinned at the top while the rest scrolls
underneath it, with a `▲ 4 more above` marker in between — your way out is
always in sight. The default answer on yes/no questions is labeled
`(default)` — that is the one enter picks if you change nothing.

## What it can read

What you pick here changes what you are offered next:

* **Audio** or **video** files in a folder. These need transcribing before
  there is any text to measure, so the app asks how — see
  [the one question it has to ask you](#the-one-question-it-has-to-ask-you).
* **Text files** (`.txt`) in a folder — one document per file.
* **A spreadsheet** with the text in one or more columns — one document per row.

There is also an "everything in a folder" option, which treats each file as
media and lets ffmpeg work out what it is.

The two text options skip transcription entirely: no ffmpeg, no models, no GPU.
On audio, asking for readability builds a four-step pipeline; on a folder of
essays it is a single step.

You browse to your data rather than typing a path (hidden folders such as
`.cache` are listed too, after the others, so a model in your Hugging Face
cache is reachable), and folders show how many
usable files they contain, so the right one can be recognized rather than
remembered. Typing a path is still offered for anyone who already knows it.

### Spreadsheets

A spreadsheet is the one source where a row can carry several pieces of text —
an open-ended answer and a follow-up, a headline and a body. So the app reads
your column names and asks which hold the text, and if you pick more than one,
whether to measure them **together** (one result per row) or **separately** (one
result per column per row).

It also asks which columns identify a row, and carries those through to the
output so the results join back to your original data.

This is asked rather than guessed because the underlying analysers default to a
column called `text` — a file whose column is called `response` would otherwise
analyze nothing and report no error.

## You only pick what you *want*, not what it takes

The checklist offers results, not steps. Tick "Readability scores" on a folder of
recordings and the app works out that it needs a transcript, which needs a
WAV, and adds both. The review screen before saving marks which steps you chose
and which were added for you.

The checklist is also filtered by your input: vocal acoustics is not offered for
a folder of essays, because there is no audio to measure.

It is grouped, too. Rows sit under headings — *Style & readability*, *Topics &
themes*, *Embeddings & vectors* and so on — arranged by what a measure tells
you rather than by how it is computed, so you can look for the question you
have instead of for the name of a method. That is why *Archetype similarity*
sits with the dictionaries: it runs on embeddings, but what it gives you is a
score against categories somebody defined in advance.

A heading has its own box. Ticking it takes everything underneath in one
press; untick one of its rows afterwards and the heading shows `[~]` to say it
is partly ticked. A heading whose rows are all unavailable — *Score with saved
models*, before you have saved any — is grayed out with them. Headings are not
themselves things Taters produces: what you end up with is always the rows.

## The one question it has to ask you

When something you picked needs a transcript, the app asks how to make one:

* **one speaker** — faster, works with the base install;
* **several speakers, labeled** — speaker diarization, which is a much larger
  install (see [installation troubleshooting](../install-troubleshooting.md#speaker-diarization)).

It asks even when you never mentioned transcription, because the two answers
differ by a multi-gigabyte download and by what the results mean.

## Statistics, if your data can support them

There are two extraction entries on the front page, and the difference is what
you came to do. **Extract features** stops at the measures and never mentions
statistics. **Extract features and run analyses** knows you have something to
test, so it only offers the spreadsheet source — the groups and outcomes have
to be columns sitting beside the text — and it insists you pick at least one
analysis rather than treating the question as optional.

In the analyses flow the app
asks which column holds your groups (for comparisons) or your outcomes (for
correlations), whether to analyze all your feature tables together or one at
a time, how to correct p-values for testing many features at once (including
not correcting them, if that is what your design calls for), and whether to
ignore any rows first — "only rows with at least 25 words", say. Filtering
asks where the variables come from (the run's own measures, your
spreadsheet's columns, or both), lets you tick as many as you want from one
list, then walks you through setting each one. A column of numbers takes an
operator and a value; a column of labels is a checkbox of the values to keep,
each with its row count — "only males and females" is two ticks, not two
typed conditions. Word count is always the
first option, and picking it adds the step that counts them — as a *filter*,
not as a feature: wanting to drop short texts is not the same as wanting
length as a predictor, so it gates the rows and stays out of the models.

On any tick screen, [enter] on a row other than **✓ Done** glides you back up
to Done in a quarter of a second — the list scrolls with the pointer — so a tick
made at row 480 of 500 does not end with 480 presses of the up arrow.

**Every column picker is a small table.** Beside each column name the app
shows what the column is treated as — `numbers`, `labels`, `text` — because
that decides what it can be used for: numbers are measurements (outcomes,
things to correlate, controls held constant as a slope), labels name groups,
classes and categorical controls, and text fits nowhere but the text
question. The treatment is detected from the file and can be wrong in one
common way: a category that happens to be numbered, a 1/2 gender code, reads
as numbers. Press → or ← on the row to change it, on any picker from the
very first one. Any column of numbers can be made labels — whether an
analysis can use fifty of them is said when the column is chosen, in words,
never refused by the arrow; labels can go back to numbers only when every
value is one; free text stays free text. A change holds for the rest of the
session, so a gender made labels while choosing a text column is still
labels when it comes up as a control — which is why there is no longer a
separate question about which controls are categories.

**The columns you pick are checked before anything runs.** A statistics run
fails in a handful of predictable ways, and every one of them is visible in
the spreadsheet on the first screen — so the app reads the whole column when
you choose it, not the two hundred rows it offered columns from, and says
so on the spot:

- An id column whose values repeat does not identify rows, and the run
  would stop when the features were joined. You are asked for the id again,
  with the repeating values named; or say there is no id and combine the
  rows that share one at the next question.
- A category to predict with a class of three rows cannot be classified —
  the classifier needs five in every fold. You are offered the classes it
  *can* learn as a checkbox, and keeping only those becomes a row filter.
  When no two classes are big enough, the question is asked again.
- A group of one is excluded from every comparison; a group of three is
  compared and should not be trusted. Same offer: keep the groups worth
  comparing.
- An outcome column that says "n/a" on row 812 is not a column of numbers.
  The cells are named, and the question is asked again.
- A control with a level of two rows — the "other" and "prefer not to say"
  of a real gender column — gets its own coefficient fitted to two people.
  You are offered the levels there are enough of.

Each check mirrors the refusal the step itself would have made ten minutes
later, after every feature had been extracted.

Each analysis then appears on the same options screen as everything else,
with its own real settings — which post-hoc test, how many folds, whether to
standardize — and the settings that apply to all of them (where the results
go, whether to keep the merged dataset, which correction to use) sit in the
shared section at the top. The run then ends with a `stats_results` folder: tidy tables plus
a plain-English report. The [statistics guide](stats.md) explains what those
tests do and how to read them.

If you chose to measure several text columns separately, each one is
analyzed separately too — one set of results per column — because a
participant contributes one row per column and pooling them would treat one
person's answers as independent observations.

When statistics are not possible the app says so rather than staying silent:
a folder of documents has no grouping columns to test, for instance. It says
so in yellow, on the spot, rather than leaving you to notice a missing
question.

## Changing options

A setting that only matters under another setting is shown only then, and
indented under it: the PCA component count and rotation appear under `pca`
once you turn it on, the Stanza language under `engine` once you choose
Stanza. The rule is written once per setting in the catalog, one evaluator
applies it to every menu (a step's own and the shared section alike), and it
fails open — a gate the app cannot read shows the row. A value you set while
a row was visible is kept when the row hides and is there again when it
comes back.

Typing is the last resort. Any setting with a known set of answers is picked
from a list: the ones whose function lists its values (`none_and_all`,
`min_obs_pct`, the Whisper model names), and the ones whose answers are this
run's own feature tables. `pca` offers off, all, or a tick list of the
feature sets the run will make, named as the results will name them, so you
never have to know that the matrix is called `doc_term_matrix_count`. A
setting that may be left unset has "automatic" as its first row, and one
whose list is only the usual answers (a Whisper model can also be a folder
on disk) keeps "something else" at the bottom, which opens the text box.

Every step has defaults that work, so the app asks once whether you want to
change anything and moves on if not.

If you do, each setting is shown with an explanation taken from the function's
own documentation, what it is currently set to, and the valid answers where
there is a short list of them. A handful of common settings are offered per
step, with the rest behind "show every option".

Settings that wire the pipeline together are never offered — changing one would
break the pipeline without saying so.

## What you end up with

Each pipeline gets a folder of its own, named after it:

```text
readability_test/
├── readability_test.yaml     the pipeline itself
├── features/                 the measures you asked for
│   └── figures/wordclouds/   word clouds of the themes and the frequent terms
├── gathered/                 intermediate tables
├── stats_descriptives/       descriptive statistics of every feature table
├── stats_results/            the statistics, when you asked for them
│   └── figures/wordclouds/   word clouds of every result, shown in report.md
└── run_manifest.json         what ran, what it produced, what failed
```

One folder per pipeline means three different runs do not overwrite each
other's results, and a whole analysis can be zipped up and sent to a colleague.

The pipeline file is an ordinary preset — the same format as the ones that ship
with Taters. You can edit it, re-run it from the command line, or hand it to
someone else. The app prints the exact command to re-run it.

## While it runs

Long steps report progress rather than sitting silent: how many files are done,
which files are being worked on right now, and for a large spreadsheet, how far
through the rows it is.

If you have many files, the app asks how many to work on at once. It only
asks when the pipeline actually fans out over files — analyzing a single
spreadsheet is one step over one table, where the setting would change nothing.

When it finishes you get a summary saying what succeeded, what failed, where the
results are, and the option to do something else or stop. An interrupted run
leaves no half-written result files behind, so re-running redoes the work
properly rather than resuming from a truncated file.

## Settings and tools

### Check my setup

Reports what Taters can actually use on this machine: your operating system and
Python version, the graphics card and driver if there is one, every relevant
package with its version, and — separately — whether PyTorch and the
transcription engine can each reach the GPU.

Those last two are reported separately on purpose, because they can differ: a
CPU-only PyTorch will not use your card even when transcription happily does.

When something is wrong it names the exact command to fix it. See
[the install guide](../install-troubleshooting.md#gpu-problems) for the usual
causes.

### Manage Taters data

Everything Taters keeps between runs shares one row, because they are all the
same errand: something stored on your disk that you want to rename, export or
throw away. Beneath it sit saved pipelines, dictionaries and archetypes, stop
lists, saved models, text encoders, and the downloaded model cache.

#### Manage saved pipelines

Rename, copy, import, export or delete your pipelines. The pipelines that ship
with Taters are read-only — copy one to adapt it.

Deleting asks you to type the pipeline's name, and says what else goes with it:
a pipeline owns its folder, so deleting it removes that run's results too.

#### Manage saved models, text encoders, dictionaries, archetypes, stop lists

Your library: the things a pipeline picks from rather than measures. Saved
models are anything a run fitted or trained — a topic model, a ridge, a
classifier, word vectors — with their weights beside them where they have
any; a model file that travels with a weights file is one entry here, moved,
renamed and deleted as one. Text encoders are language models adapted to your
texts, the base for embeddings and fine-tuning. Import, rename, export and
delete from these screens, and the finish screen of any run that produced a
model offers to add it here so the next study can pick it. *Import a
classifier from Hugging Face* brings in a finished text classifier or
regressor from the hub -- one already in this computer's Hugging Face cache,
a checkpoint folder, or a name to download -- as a saved model that scores
like the rest.

Dictionaries and archetypes arrive with a shelf already stocked: Taters ships
35 published content-coding dictionaries and 2 archetype dictionaries, listed
with their citations under
[Built-in dictionaries and archetypes](bundled-dictionaries.md). They behave
like anything you import — rename one, export it, delete one you will never
use. Upgrading Taters keeps the built-ins current without undoing your
decisions: one you deleted stays deleted, and one you edited stays yours.

For a saved model there is one more row, *Change how the ticked model is
applied*: its name, the names of its output columns, what each predicted
class is written as (a model trained on a 0/1 column predicts `0` and `1`
until you say `control` and `patient` here), and the settings it scores
with -- how a word-vector model averages words and which concepts it
measures, a fine-tuned predictor's batch size and whether it writes
per-class probabilities. They live in the model file, so every pipeline
that applies the model uses them from then on.

### Where downloaded models are kept

Every transformer, sentence-embedding and Whisper model is downloaded once
into a cache -- by default under your home folder. On a shared server, or a
machine with a small system drive, that is the wrong place, so this screen
shows where models go and why, lets you choose another folder, and remembers
it: the choice lives in `settings.json` in the Taters home, is applied every
time Taters starts, and every downloading library follows it. An
administrator can set `TATERS_MODEL_CACHE` in the environment instead, which
wins over any folder chosen here; the Hugging Face variables `HF_HUB_CACHE`
and `HF_HOME` are honored as they always were when neither is set. Models
already downloaded elsewhere are not moved. "Check my setup" shows the
folder, how it was chosen, and how much is in it.

### How this looks in your terminal

Two settings for the two things a terminal will not tell us the truth about.

**Colors.** When you connect over SSH, your client announces what kind of
terminal it is — and PuTTY, MobaXterm, Git Bash and plain `ssh` all announce a
bare `xterm`, which by convention means "16 colors". They are all perfectly
capable of far more; it is just what the label says. Taken at face value, the
potato in the banner collapses from four shades to two, the eyes and the frame
land on a gray so dark it reads as black, and the frame's slow color drift stops
moving entirely. So Taters prefers 256 colors on anything that looks like a real
terminal, which is the same thing the menus have always done. The exceptions are
deliberate: a Linux text console (`TERM=linux`) really does stop at 16 and keeps
16, and output that is not going to a terminal at all — a redirected log, a
pipe — gets no color codes whatever. Set it by hand here if you need to, or with
`TATERS_COLOR=truecolor` in the environment.

**Characters.** The wizard marks things with arrows and ticks. Windows consoles
quietly borrow a glyph from another font when the one you chose hasn't got it,
so those characters look fine there — but `↩`, `✓` and `⚠` are genuinely absent
from every monospace font Windows ships, and a terminal that does not do that
borrowing draws an empty box instead. There is no way to ask a terminal what its
font can draw, so Taters defaults to characters that every one of those fonts
has. If yours can do better, switch to fancy here or set
`TATERS_FANCY_GLYPHS=1`.

Both take effect the next time Taters starts.

### Check for new versions

When a version newer than yours is on PyPI, a single dim line appears under the
menu — `newer version available: v0.7.3` — and that is the whole feature. It is
a statement of fact, not a nag: nothing to dismiss, nothing that blocks a run,
and no complaint when you carry on using the version you have.

The check asks PyPI for the current version number at most once a day, in the
background, and writes the answer to `settings.json` in the Taters home. The
menu is drawn from that saved answer, never from a live request, so a slow
network, a proxy, or no connection at all cannot delay the opening screen. One
consequence worth knowing: a brand-new install shows nothing the first time,
because the first check's answer arrives after the menu is already on screen.

Nothing about you, your data, or your runs is sent — the request is for a
public version number and carries no identifying information. If you would
rather it made no outbound request at all, turn it off on this screen, or set
`TATERS_NO_UPDATE_CHECK=1` in the environment, which wins over the setting and
is the one to use when you are locking down a shared machine for everybody on
it.

## If something is missing

Some measures need optional extras. When you pick one you do not have installed,
the app says so and names the command, then asks whether to keep the step
anyway — a pipeline that is slightly ahead of your machine is a reasonable thing
to want.

## Re-running is cheap

A step reuses an existing output only if that output was *made the same
way*: the record beside every table says how, and when it disagrees with
the run — other id columns, another tokenizer setting, a different word list
— the step redoes the work and says so. Two pipelines saved into the same
folder can therefore no longer feed each other stale tables, which is how
one real run died at the join. A file with no record is left alone, with
one exception: the gathered text and metadata tables take seconds to make
and every join depends on them, so those are rebuilt rather than trusted.
The statistics follow a second rule: a result older than the table it was
computed from is redone, so a rebuilt table flows through to fresh
analyses and a fresh report.

Steps skip work whose output already exists, so a re-run after adding one more
measure only does the new part. To force everything to be redone, set
`overwrite_existing` when tuning options, or delete the results you want rebuilt.

## Prefer plain output?

`taters --plain` asks the same questions one at a time without the redrawn
screen, which is useful over a slow connection or in a terminal that renders it
badly.

## Where to go next

- [Install guide](../install-guide.md) — the optional extras and what needs them
- [Pipelines](pipelines.md) — the file the app writes, in full
- [Wizard API](../api/wizard.md) — the layers behind it, if you want to build on them
