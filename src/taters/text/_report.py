"""
The pieces every training report shares: a Markdown table, the package
versions, the machine, a plot that is skipped when the chart renderer is
not available.

A report is what makes a trained model reportable -- the methods paragraph
and the numbers a paper needs -- and the three trainers (word vectors,
adapted encoders, fine-tuned predictors) must state the same facts the same
way, so the helpers that produce them live once.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = ["md_table", "versions", "machine", "line_plot", "write_markdown"]


def md_table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> List[str]:
    """A GitHub-flavoured Markdown table, blanks for None."""
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return lines


def versions(*modules: str) -> Dict[str, str]:
    """``{package: version}`` for python, taters and whatever else is named,
    skipping anything not installed."""
    out = {"python": platform.python_version()}
    for mod in ("taters", *modules):
        try:
            module = __import__(mod)
            out[mod] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            continue
    return out


def machine(device_name: Optional[str] = None) -> str:
    """One line naming the hardware a run used, GPU included when torch can
    name it."""
    line = f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical cores"
    if device_name and device_name.startswith("cuda"):
        try:
            import torch

            line += f"; GPU {torch.cuda.get_device_name(0)}"
        except Exception:
            line += "; GPU"
    return line


def line_plot(path: Path, series, *, title: str, x_label: str, y_label: str,
              x_scale: str = "integer", dpi: Optional[int] = None,
              scale: Optional[float] = None) -> Optional[Path]:
    """A line chart when Pillow and the renderer are available; None when
    not, so a report stands on its tables alone."""
    try:
        from ..figures.charts import line_chart, pillow_missing_reason
    except ImportError:
        return None
    if pillow_missing_reason():
        return None
    try:
        return line_chart(series, path, title=title, x_label=x_label,
                          y_label=y_label, x_scale=x_scale, dpi=dpi,
                          scale=scale)
    except TypeError:
        # a signature that does not match is a programmer's mistake, and
        # swallowing it here turns it into a figure that silently stops being
        # drawn. Everything else -- a missing font, data that cannot be
        # ranged -- is the environment, and the report survives without it.
        raise
    except Exception:
        return None


def write_markdown(path: Path, lines: Sequence[str]) -> Path:
    from ..helpers.atomic import atomic_write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")
    return path
