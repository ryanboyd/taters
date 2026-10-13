# Utilities & Helpers

Between raw data and analysis sits a layer of unglamorous decisions that
quietly shape every number you will eventually publish: what counts as a
document, how scattered outputs become one dataset, and how you will prove —
months later — where each row came from. The helpers exist so those
decisions are made once, explicitly, instead of being improvised in a
different way for every project.

If one idea organizes this page, it is **the unit of analysis**. The same
interview corpus can be analyzed per utterance, per speaker, per session, or
per person-across-sessions, and the choice changes what your features mean
and what your statistics are entitled to claim. Taters makes that choice a
visible, repeatable setting rather than an accident of how files happened to
be arranged on disk.

* Wickham, H. (2014). Tidy data. *Journal of Statistical Software, 59*(10),
  1–23. (The "one row per observation" philosophy every helper here serves.)

---

## Finding your files

Every project starts with "which files are we even talking about?" — and ad
hoc answers (whatever was in the folder that day) are a quiet threat to
reproducibility. The file-discovery helper turns that question into an
explicit, repeatable specification: what types, which folders, what to
include and exclude, and (for media) whether a file *actually contains* a
playable stream rather than merely wearing the right extension.

**In Taters:** `find_files`, used internally by every pipeline's input
discovery and available on its own.

---

## From raw text to analysis-ready data

Every text method in Taters reads the same simple shape: one row per
document, `text_id` and `text`. The gathering helpers produce that shape
from the two ways text actually arrives — a folder of files (one document
each, in any mix of .txt, Word, and PDF), or a spreadsheet with text in one
or more columns.

The spreadsheet path is where the unit-of-analysis decision lives: you can
keep one row per original row, or *group* — say, everything each speaker
said across a session becomes one document — and the gatherer records how
many pieces went into each group, and can carry along the identifier
columns you will need for merging later. It streams, so a spreadsheet far
larger than memory is fine, and broken files in a folder (a PDF with no
machine-readable text, a mislabeled binary) are skipped with a warning
rather than taking the run down.

**In Taters:** the gather step at the front of every text pipeline; also
callable directly as `text_gather`.

---

## Merging, stacking and reshaping spreadsheets

A lot of data arrives as several files that have to become one before
anything can be measured: the trials in one spreadsheet and the participants
in another, one file per session, a wide file that should have been long.
Until now that meant leaving Taters for R or pandas. *Wrangle data* now does
the four things that cover nearly all of it, and every one of them reads the
files first, says in plain words what is going to happen -- with the real
numbers -- and only then writes.

Nothing here holds a file in memory. The stacks and the wide-to-long pass
stream; the merge and the pivot work through a temporary database on disk,
about the size of the file being matched against, which is removed afterward.
And nothing here reads a key as a number: a 17-digit ID is never rounded into
its neighbor.

### Matching two spreadsheets by an ID (merge)

Start with the file whose rows you want to keep -- the trials -- then the one
with the columns to add -- one row per participant. Tick the column(s) that
identify the same thing in both (Taters suggests the one that is in both
files and unique in the second). Then it reads both files and tells you what
matched:

```
12,340 of 12,500 rows in trials.csv match a row in participants.csv; 160 do not (e.g. p0091, p1203, p1377).
40 of 540 rows in participants.csv have no match in trials.csv (e.g. p0002, p0310).
age is in both files and agrees on every matched row, so it's kept once.
```

That read-back is the whole point. A key typed as `P001` on one side and `1`
on the other is found here, before an analysis is built on it, and when
nothing matches at all the two files' keys are shown side by side so you can
see why.

A few rules, each decided for a reason:

* **Keys are compared with surrounding spaces ignored, and nothing else.**
  When `007` on one side is `7` on the other, or `P01` is `p01`, Taters says
  so and asks whether to match those too -- and says how many rows it would
  add. It is a question rather than the default because loose matching can
  also fold two genuinely different IDs together, and the read-back warns
  when a file has IDs that differ only in case.
* **One participant, many trials is fine.** The merge says each participant
  row will repeat once per trial and how many rows come out. A key that
  repeats in *both* files is refused, with examples, because that almost
  always means one more column is needed to tell rows apart -- a session, a
  time point.
* **A blank ID matches nothing.** It is counted separately from "unmatched".
* **A column in both files** is kept once when the two agree on every
  matched row (a blank on either side counts as agreement, and is filled in
  from the other file); otherwise both are kept, the second as
  `<file>__<column>`, and the read-back gives an example of the disagreement.
* **Which rows come out is asked with real numbers:** all 12,500 trials
  (the 160 without a match get empty participant columns), only the 12,340
  that matched, or everything including the 40 participants with no trials.
  The question is skipped when every row matched.
* IDs that look like `1.23457E+11` are what Excel writes after rounding a
  long number. They are flagged, not matched; re-export that column as text.

### Putting spreadsheets one under another (stack)

Tick the files. Columns with the same name line up, a column only some files
have is left empty in the others, and a `source` column says which file each
row came from. Column names that differ only in capitals or spacing (`Age`,
`age`) are pointed out and merged on a yes. A file that shares no column with
the rest is pointed out too -- stacked, it is a checkerboard of blanks, and
nearly always what was wanted was a merge.

### Wide and long

`score_t1, score_t2, score_t3` become one `score` column plus a column saying
which time; or the other way round. Going long, the shared start of the
column names (`score_t`) is offered for dropping so the time column reads
`1, 2, 3`, and a row is kept for an empty cell unless you say otherwise, so a
balanced design stays visibly balanced. Going wide, Taters reads the file and
tells you how many rows and new columns will come out; a participant with
two rows for the same condition is shown and you choose whether to average
them or keep the first or last. A column with thousands of values is refused
as the thing to make columns from, because that is nearly always an ID or a
timestamp picked by mistake.

### Messy files

The files that need this are the messy ones, so the readers expect it. The
encoding is worked out from the file -- UTF-8 with or without a BOM, UTF-16
from Excel's "Unicode Text", then Windows cp1252 -- and the separator is
sniffed per file, so a semicolon file from a European Excel merges with a
comma file without being told. Duplicate or blank column names are made
unique and announced. Rows with too few or too many cells are padded or
trimmed, counted, and shown before you decide to go on. Writing the result
over one of the files being read is refused outright. Every output is
UTF-8 and comma-separated, which every other Taters step reads.

**In Taters:** *Wrangle data* in the app; also callable directly as
`merge_spreadsheets`, `stack_spreadsheets`, `wide_to_long` and
`long_to_wide`, each with a `plan_*` twin that reads the files and reports
without writing anything.

---

## Assembling the final dataset

A pipeline leaves you with several feature tables — dictionary scores here,
readability there, embeddings somewhere else — that all describe the same
documents. The feature gatherer is the last mile: it joins or stacks those
tables into the one wide CSV your statistics package expects, and can
*aggregate* on the way (mean embeddings per speaker, say), which is again
the unit-of-analysis decision in another costume. Averaging row-level
features to person-level features is a modeling choice with real
consequences — make it deliberately, and the gatherer makes it visible in
the output (group counts ride along).

**In Taters:** the merge steps a pipeline adds automatically when your
features need aligning; also callable directly as `feature_gather`.

---

## Practical habits worth stealing

* **Keep provenance columns.** Carrying `source_path` and identifier columns
  through the gather costs nothing and answers "where did this row come
  from?" a year later.
* **Let outputs land where they land.** Every tool writes to a predictable
  place (`./features/<kind>/...`) and refuses to overwrite unless asked —
  so an interrupted run resumes instead of restarting, and nothing silently
  replaces the results a draft was written from.
* **Decide the unit of analysis once, on purpose,** and let the pipeline's
  shared settings enforce it everywhere — rather than aggregating one way in
  the gather and another way in the merge.
