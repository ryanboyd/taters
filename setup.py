"""Compatibility shim for installing Taters from a git clone.

All packaging metadata -- dependencies, extras, console scripts, the src/
layout -- lives in ``pyproject.toml``. This file deliberately holds none of
it: a bare ``setup()`` call tells setuptools to read the ``[project]`` table,
so there is exactly one place to edit and nothing here can drift out of sync.

This exists for muscle memory only. ``pip install -e .`` is the documented way
to install from a clone, and it beats ``python setup.py install`` on every axis
that matters:

  * ``setup.py`` cannot bootstrap itself. Since Python 3.12 a fresh venv ships
    pip and nothing else, so the import below raises ModuleNotFoundError until
    you ``pip install setuptools`` by hand. pip builds in an isolated
    environment and fetches setuptools on its own.
  * ``setup.py install`` copies the source, so later edits to the clone are
    ignored until you reinstall. ``-e`` links to the clone instead.
  * setuptools deprecated direct invocation and has promised build errors in
    some future release.

If that release lands, delete this file. Nothing else in the project reads it.
"""

from setuptools import setup

setup()
