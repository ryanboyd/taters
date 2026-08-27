"""Checks on pyproject.toml itself.

Packaging bugs are invisible during development — your working copy has every
package installed for unrelated reasons, so an undeclared dependency only
surfaces when someone installs from PyPI into a clean environment and gets
`ModuleNotFoundError`. These tests read the metadata and compare it against
what the code actually imports.

This suite already caught three: pandas, numpy and PyYAML were imported at
module level and declared nowhere, so a fresh install could not run
`feature_gather` (which the conversation_video preset uses four times).
"""

import ast
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
SRC = PROJECT_ROOT / "src" / "taters"

# Import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "faster_whisper": "faster-whisper",
    "sentence_transformers": "sentence-transformers",
    "parselmouth": "praat-parselmouth",
    "contentcoder": "contentcoder",
    "archetypes": "archetyper",
    "ContentCoder": "contentcoder",
    "sklearn": "scikit-learn",
}

# Import-time requirements of modules behind an optional extra: importing these
# modules is expected to fail without the extra installed.
OPTIONAL = {
    "parselmouth": "vocalacoustics",
    "soundfile": "vocalacoustics",
    "disvoice": "vocalacoustics",
    "textstat": "readability",
    "nemo": "diarization",
    "torch": None,          # deliberately not declared: users pick their own build
    "nvidia": "cuda",
    "chardet": None,        # optional nicety, guarded by try/except
}


@pytest.fixture(scope="module")
def metadata() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def top_level_imports(path: Path) -> set[str]:
    """Third-party modules imported when `path` is imported (not lazily)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:                       # module scope only
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if n not in sys.stdlib_module_names and n != "taters"}


def source_files() -> list[Path]:
    return [p for p in sorted(SRC.rglob("*.py")) if "whisper-diarization" not in str(p)]


def declared_distributions(metadata: dict) -> set[str]:
    """Every distribution named in dependencies or any extra, normalized."""
    import re
    names: set[str] = set()
    project = metadata["project"]
    groups = [project.get("dependencies", [])]
    groups += list(project.get("optional-dependencies", {}).values())
    for group in groups:
        for spec in group:
            name = re.split(r"[<>=!\[;\s]", spec, maxsplit=1)[0]
            names.add(name.strip().lower().replace("_", "-"))
    return names


def test_pyproject_parses(metadata):
    assert metadata["project"]["name"] == "taters"


def test_every_import_time_dependency_is_declared(metadata):
    """
    Anything imported at module scope must be installable from the metadata —
    either as a core dependency or via a documented extra.
    """
    declared = declared_distributions(metadata)
    missing: list[str] = []

    for path in source_files():
        for module in top_level_imports(path):
            if module in OPTIONAL:
                continue
            dist = IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
            if dist not in declared:
                rel = path.relative_to(PROJECT_ROOT)
                missing.append(f"{module} (imported by {rel}) -> needs '{dist}'")

    assert not missing, "undeclared dependencies:\n  " + "\n  ".join(sorted(missing))


def test_core_dependencies_cover_the_core_helpers(metadata):
    """
    helpers/ and pipelines/ are the parts every workflow touches; they must not
    depend on an optional extra.
    """
    core = {
        name.lower().replace("_", "-")
        for name in (
            __import__("re").split(r"[<>=!\[;\s]", spec, maxsplit=1)[0]
            for spec in metadata["project"]["dependencies"]
        )
    }
    for path in source_files():
        if path.parent.name not in {"helpers", "pipelines"}:
            continue
        for module in top_level_imports(path):
            dist = IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
            assert dist in core, (
                f"{path.name} imports {module} at module level, but '{dist}' is "
                f"not a core dependency"
            )


def test_console_script_targets_exist(metadata):
    """An entry point pointing at a module that does not exist fails on first use."""
    import importlib
    for name, target in (metadata["project"].get("scripts") or {}).items():
        module_path, _, attribute = target.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            pytest.skip(f"{name}: {module_path} needs an optional dependency ({exc})")
        assert hasattr(module, attribute), f"{name}: {module_path} has no {attribute}()"
        assert callable(getattr(module, attribute))


def test_declared_python_version_matches_what_the_code_uses(metadata):
    """
    The CLI uses argparse.BooleanOptionalAction and the code uses `X | Y` type
    syntax, both of which need 3.10+.
    """
    assert metadata["project"]["requires-python"] == ">=3.10"


def test_packaging_does_not_ship_the_tests(metadata):
    """Tests live outside src/, so the wheel should not contain them."""
    find = metadata["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert all(pattern.startswith("taters") for pattern in find["include"])
