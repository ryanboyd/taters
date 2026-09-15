# docs/_macros.py
from __future__ import annotations
from pathlib import Path
import yaml

def define_env(env):
    """
    mkdocs-macros entrypoint. Defines macros:
      - pipelines_table(): compact table of presets
      - pipelines_cards(): collapsible cards of presets
      - pipelines_scan_report(): debug info about where we looked
    """

    # Resolve project root from mkdocs config (robust on Windows too)
    cfg_path = env.conf.get("config_file_path", ".")
    project_root = Path(cfg_path).resolve().parent

    # Candidate directories to search for presets (existence-checked below)
    candidate_dirs = [
        project_root / "pipelines",
        project_root / "presets",
        project_root / "taters" / "pipelines" / "presets",
        project_root / "src" / "taters" / "pipelines" / "presets",
    ]

    # --- helpers -------------------------------------------------------------

    def _scan_dirs() -> list[Path]:
        roots = [p for p in candidate_dirs if p.is_dir()]
        if not roots:
            print("[pipelines] No preset roots exist among:", ", ".join(str(p) for p in candidate_dirs))
        else:
            print("[pipelines] Scanning roots:", ", ".join(str(p) for p in roots))
        return roots

    def _iter_preset_files():
        """Yield all preset files found in any existing root."""
        for base in _scan_dirs():
            # Support *.yaml and *.yml
            for pattern in ("**/*.yaml", "**/*.yml"):
                for p in sorted(base.rglob(pattern)):
                    name = p.name.lower()
                    if name.startswith("~") or name.endswith(".tmp"):
                        continue
                    yield p

    def _load_meta(p: Path) -> dict:
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[pipelines] Failed to read {p}: {e}")
            data = {}
        meta = dict(data.get("meta", {}) or {})
        # Sensible defaults
        meta.setdefault("id", p.stem)
        meta.setdefault("title", p.stem.replace("_", " "))
        meta.setdefault("summary", "")
        meta.setdefault("tags", [])
        meta["file"] = str(p)
        return meta

    def _load_all_presets() -> list[dict]:
        files = list(_iter_preset_files())
        print(f"[pipelines] Found {len(files)} preset file(s).")
        items = [_load_meta(p) for p in files]
        items.sort(key=lambda m: (m.get("title") or m.get("id") or "").lower())
        return items

    # --- exposed macros ------------------------------------------------------

    def pipelines_table() -> str:
        """Return a Markdown table of all discovered presets."""
        items = _load_all_presets()
        if not items:
            return "_No presets found._"
        lines = ["| ID | Title | Tags |", "|---|---|---|"]
        for m in items:
            tags = ", ".join(m.get("tags") or [])
            lines.append(f"| `{m['id']}` | {m['title']} | {tags} |")
        return "\n".join(lines)

    def _fmt_kv_table(rows):
        """rows: sequence of (key, value) -> Markdown table (2 cols). Skips empty values."""
        lines = ["| Key | Value |", "|---|---|"]
        for k, v in rows:
            if v is None or v == "" or v == [] or v == {}:
                continue
            if isinstance(v, (list, tuple)):
                v = ", ".join(map(str, v))
            elif isinstance(v, dict):
                v = ", ".join(f"{kk}: {vv}" for kk, vv in v.items())
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    def _fmt_variables_table(vars_meta: dict | None):
        """Render variables map {name: {default, desc}} -> 3-col table."""
        if not vars_meta:
            return ""
        lines = ["| Variable | Default | Description |", "|---|---|---|"]
        for name, spec in sorted(vars_meta.items(), key=lambda kv: kv[0].lower()):
            default = spec.get("default", "")
            desc = spec.get("desc", "")
            lines.append(f"| `{name}` | `{default}` | {desc} |")
        return "\n".join(lines)


# --- replace your existing pipelines_cards() with this richer version ---
    def pipelines_cards() -> str:
        """
        Return collapsible cards (requires admonition + details) with rich meta info:
        summary, use_cases, inputs, outputs, requirements, variables, notes, and CLI example.
        """
        items = _load_all_presets()
        if not items:
            return "_No presets found._"

        blocks = []
        for m in items:
            title = m.get("title") or m.get("id") or "Preset"
            pid = m.get("id", "")
            tags = ", ".join(m.get("tags") or [])
            summary = (m.get("summary") or "").strip()

            # Optional sections
            use_cases = m.get("use_cases") or []
            inputs = m.get("inputs") or {}
            requirements = m.get("requirements") or {}
            variables = m.get("variables") or {}
            version = m.get("version") or ""
            authors = ", ".join(m.get("authors") or []) if isinstance(m.get("authors"), (list, tuple)) else (m.get("authors") or "")
            last_updated = m.get("last_updated") or ""
            notes = (m.get("notes") or "").strip()
            cli_example = (m.get("cli_example") or "").strip()

            # Build block
            b = []
            b.append(f'??? info "{title} (`{pid}`)"')

            # Summary line
            if summary:
                b.append(f"    {summary}")
                b.append("")

            # Tags / version / authors quick row
            quick = []
            if tags: quick.append(f"**Tags:** {tags}")
            if version: quick.append(f"**Version:** `{version}`")
            if authors: quick.append(f"**Authors:** {authors}")
            if last_updated: quick.append(f"**Last updated:** {last_updated}")
            if quick:
                b.append("    " + "  \n    ".join(quick))
                b.append("")

            # Use cases
            if use_cases:
                b.append("    **Use cases**")
                for u in use_cases:
                    b.append(f"    - {u}")
                b.append("")

            # Inputs (render as 2-col table)
            if inputs:
                b.append("    **Inputs**")
                b.append("    ")
                tbl = _fmt_kv_table(list(inputs.items()))
                for line in tbl.splitlines():
                    b.append("    " + line)
                b.append("")

            # Outputs (list of objects with path/desc)
            #if outputs:
            #    b.append("    **Outputs**")
            #    for o in outputs:
            #        path = o.get("path", "")
            #        desc = o.get("desc", "")
            #        if path and desc:
            #            b.append(f"    - `{path}` — {desc}")
            #        elif path:
            #            b.append(f"    - `{path}`")
            #    b.append("")

            # Requirements (table: cpu/gpu/ffmpeg/extras)
            if requirements:
                b.append("    **Requirements**")
                rows = []
                for key in requirements.keys():
                    if key not in ["extras"]:
                        rows.append((key, str(requirements.get(key))))
                extras = requirements.get("extras")
                if extras:
                    rows.append(("Extras", ", ".join(extras) if isinstance(extras, (list, tuple)) else str(extras)))
                b.append("    ")
                tbl = _fmt_kv_table(rows)
                for line in tbl.splitlines():
                    b.append("    " + line)
                b.append("")

            # Variables (structured table)
            if variables:
                b.append("    **Variables**")
                b.append("    ")
                vtbl = _fmt_variables_table(variables)
                for line in vtbl.splitlines():
                    b.append("    " + line)
                b.append("")

            # Notes
            if notes:
                b.append("    **Notes**")
                for line in notes.splitlines():
                    b.append("    " + line)
                b.append("")

            # CLI example
            if cli_example:
                b.append("    **CLI example**")
                b.append("    ")
                b.append("    ```bash")
                for line in cli_example.splitlines():
                    b.append("    " + line)
                b.append("    ```")
                b.append("")

            # File path
            #if file_path:
            #    b.append(f"    _Preset file_: `{file_path}`")

            blocks.append("\n".join(b))

        return "\n\n".join(blocks)


    def pipelines_scan_report() -> str:
        """Return a tiny debug block showing where we looked and what we found."""
        roots = _scan_dirs()
        files = list(_iter_preset_files())
        block = ["**Preset roots checked:**"]
        block += [f"- {r}" for r in roots] or ["- (none found)"]
        block += ["", f"**Files discovered:** {len(files)}"]
        block += [f"- {p}" for p in files] or ["- (none)"]
        return "\n".join(block)

    env.macro(pipelines_table)
    env.macro(pipelines_cards)
    env.macro(pipelines_scan_report)
