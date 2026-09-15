# Built-in library assets

Files placed under `<kind>/` here ship with the package and are copied into a
user's library (`~/.taters/library/<kind>/`) the first time that kind's folder
is created. Only files matching the kind's accepted suffixes are seeded (see
`taters/helpers/library.py:KINDS`).

Seeding happens once: a user who deletes a built-in keeps it deleted.
