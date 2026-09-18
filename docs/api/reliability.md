# Writing results safely

Three small modules that everything else depends on being correct. None is part
of the user interface; they exist because of failure modes that are invisible
until they have already cost someone their results, or their afternoon.

## Atomic writes

Analysis steps stream output row by row, and steps skip work whose output
already exists — that is what makes a long pipeline resumable. Together those
two facts make an interrupted run dangerous: a half-written file keeps its final
name, and the next run accepts it as finished.

Writing under a scratch name and renaming at the end removes that. A rename is
indivisible as far as the filesystem is concerned, so the real name only ever
refers to a complete file.

::: taters.helpers.atomic
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## Progress reporting

A GLOBAL pipeline step is a single call, so the runner cannot count it from
outside. A step that *can* count itself says so by declaring an `on_progress`
parameter, which the runner then injects automatically — a signature check
rather than a registry, so a new analyzer opts in without anything else needing
to know.

::: taters.helpers.progress
    options:
      members_order: source
      show_source: true
      show_root_heading: true

## The run log

A run used to record one line when a step failed: the exception's type and its
message. That is enough when the message is the reason, and useless when it is
a wrapper around the reason — a library reporting "could not import module
'RobertaModel'" while the exception it was raised *from*, naming the module
that was actually missing, went unrecorded. Nothing captured the environment
either, so a failure caused by the installed build of a dependency looked
identical to a failure in Taters.

So every run now writes a log: the environment, what was asked and answered,
each step, everything printed — including from child processes — and the full
chain of causes behind anything that raised. The manifest keeps the short
structured reason; this keeps the rest.

Deliberately **not** built on `atomic_write` above. That helper's contract is
that nothing appears at the destination unless the write finished, and it
unlinks its scratch file when an exception passes through — which is exactly
the moment a log has to survive. This writes plainly and flushes per line.

::: taters.helpers.runlog
    options:
      members_order: source
      show_source: true
      show_root_heading: true
