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
