# Writing results safely

Two small modules that everything else depends on being correct. Neither is part
of the user interface; both exist because of failure modes that are invisible
until they have already cost someone their results.

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
