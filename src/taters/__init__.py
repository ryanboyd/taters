"""Taters: Takes All Things, Extracts Relevant Stuff.

The package exposes the :class:`~taters.Taters.Taters` facade directly, so both
of these work identically::

    from taters import Taters
    from taters.Taters import Taters

Importing the package is cheap: the facade only pulls in heavy dependencies
(torch, transformers, parselmouth, ...) when you actually call a method.
"""

from __future__ import annotations

from .Taters import Taters
from .helpers.settings import apply_model_cache as _apply_model_cache

# first thing's first: if someone picked a model cache in Settings (or set
# TATERS_MODEL_CACHE), it has to be in the environment before any library
# that downloads models gets imported -- the hub library reads it once, when
# it first loads. cheap, too: this is one small JSON read.
_apply_model_cache()

try:  # pragma: no cover - depends on install method
    from importlib.metadata import version as _version

    __version__ = _version("taters")
except Exception:  # pragma: no cover - source checkouts without metadata
    __version__ = "unknown"

__all__ = ["Taters", "__version__"]
