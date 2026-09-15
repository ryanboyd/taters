"""
The plain-English report: one markdown file summarizing a run's statistics.

Each analysis writes its own section fragment into ``stats_results/_sections/``
as it runs (see :func:`taters.stats._common.write_section`); this module just
concatenates them in their fixed order under a dated header. It computes no
statistics of its own -- every number in the report was produced, and is owned,
by the module that understands it. That division is what lets a future
analysis join the report by writing one fragment, with no edits here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..helpers.atomic import atomic_write
from ._common import SECTIONS_DIR, taters_version
from ..helpers.cliargs import CliSpec

PathLike = Union[str, Path]


def write_stats_report(
    *,
    stats_dir: PathLike = "stats_results",
    title: Optional[str] = None,
    out_md: Optional[PathLike] = None,
    keep_table: bool = True,
    overwrite_existing: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Concatenate the analyses' section fragments into ``report.md``.

    Parameters
    ----------
    stats_dir
        The stats output folder holding ``_sections/``.
    title
        Heading for the report; defaults to "Statistical results".
    out_md
        Where to write; defaults to ``<stats_dir>/report.md``.
    keep_table
        Whether to leave the merged analysis table in place. Deleting it is
        this step's job rather than the assemble step's for the obvious
        reason: every analysis reads that table, so the only moment it is
        safe to remove is after the last one has run, and this is the step
        that runs last. The sidecar describing it goes with it -- a map of a
        file that is not there helps nobody -- while the row accounting in
        ``assemble_manifest.json`` stays, because how many rows were
        analyzed is part of the result.
    overwrite_existing
        Unlike the analyses, the report is cheap and derived, so the default
        skip is mostly about symmetry: pass True to rebuild after re-running
        any analysis.

    Returns
    -------
    Path
        The report file.
    """
    import datetime

    stats_dir = Path(stats_dir)
    out_path = Path(out_md) if out_md else stats_dir / "report.md"
    sections_dir = stats_dir / SECTIONS_DIR
    from ._common import reusable

    if reusable(out_path, *(sorted(sections_dir.glob("*.md"))
                            if sections_dir.is_dir() else []),
                overwrite_existing=overwrite_existing, verbose=verbose,
                what="the report"):
        if verbose:
            print(f"Report already exists; returning existing file: {out_path}")
        return out_path
    fragments = sorted(sections_dir.glob("*.md")) if sections_dir.is_dir() else []
    if not fragments:
        raise FileNotFoundError(
            f"no report sections found under {sections_dir}. Run at least "
            f"one analysis first -- each one writes its own section.")

    version = taters_version()
    header = [
        f"# {title or 'Statistical results'}",
        "",
        f"*Written {datetime.date.today().isoformat()}"
        + (f" by taters {version}" if version else " by taters")
        + f"; tables live beside this file in `{stats_dir.name}/`.*",
        "",
    ]
    body = []
    for fragment in fragments:
        body.append(fragment.read_text(encoding="utf-8").rstrip())
        body.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(out_path, mode="w", encoding="utf-8") as fh:
        fh.write("\n".join(header + body).rstrip() + "\n")
    if not keep_table:
        for name in ("analysis_table.csv", "analysis_table_sets.json"):
            target = stats_dir / name
            if target.is_file():
                target.unlink()
                if verbose:
                    print(f"[report] removed {name} (keep_table is off)")
    if verbose:
        print(f"[report] {len(fragments)} section(s) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# command line -- we derive it from the function(s) above; see
# helpers.cliargs.CliSpec. the aliases are the spellings the old hand-written
# parser used, and we keep them so that every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    write_stats_report,
    description="Gather every analysis's summary into one report.md.",
    aliases={
        'out_md': ['--out'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
