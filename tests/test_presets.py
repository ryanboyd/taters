"""Static validation of the shipped pipeline presets.

A preset is a YAML file full of function names and parameter names. Nothing
checks it until you run it — which, for a preset whose last step is an hour in,
is a bad time to discover a typo. These tests read every preset and confirm
that everything it names actually exists and lines up, in under a second.

They are also a guard on refactoring: rename a parameter in the Python and the
preset that still passes the old name fails here immediately.
"""

import inspect
import re
from pathlib import Path

import pytest
import yaml

from taters import Taters
from taters.pipelines.run_pipeline import _get_preset_dirs, resolve_call

BUILTIN_PRESETS = sorted((Path(__file__).parent.parent / "src" / "taters" /
                          "pipelines" / "presets").glob("*.yaml"))

# Expressions the runner resolves without any artifact existing.
SPECIAL_NAMES = {"input", "cwd", "run_manifest"}
TEMPLATE_RE = re.compile(r"\{\{([^}]+)\}\}")


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def underlying(call: str):
    """
    The real function a `call:` points at.

    `potato.*` targets go through the facade, which forwards with **kwargs — so
    to check parameter names we have to look at what it forwards *to*.
    """
    import importlib
    if not call.startswith("potato."):
        return resolve_call(call, Taters())
    namespace, method = call.split(".")[1:3]
    src = inspect.getsource(getattr(type(getattr(Taters(), namespace)), method))
    rel, name = re.search(r"from (\S+) import (\w+)", src).groups()
    try:
        module = importlib.import_module("taters" + rel)
    except ImportError as exc:
        pytest.skip(f"{call} needs an optional dependency: {exc}")
    return getattr(module, name)


def templates_in(value) -> list[str]:
    """Every {{expression}} appearing anywhere inside a nested structure."""
    found: list[str] = []
    if isinstance(value, dict):
        for v in value.values():
            found += templates_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            found += templates_in(v)
    elif isinstance(value, str):
        found += [m.strip() for m in TEMPLATE_RE.findall(value)]
    return found


def pytest_generate_tests(metafunc):
    """
    Generate one test per (preset, step) pair.

    This hook is pytest's programmatic version of @parametrize: it lets the
    parameters come from disk rather than being hard-coded, so a new preset is
    picked up automatically.
    """
    if "preset_path" in metafunc.fixturenames and "step" not in metafunc.fixturenames:
        metafunc.parametrize("preset_path", BUILTIN_PRESETS, ids=lambda p: p.stem)
    if "step" in metafunc.fixturenames:
        cases, ids = [], []
        for path in BUILTIN_PRESETS:
            for i, step in enumerate(load(path).get("steps", []) or [], start=1):
                cases.append((path, i, step))
                ids.append(f"{path.stem}-step{i}-{step.get('call', '?')}")
        metafunc.parametrize("preset_path,step_number,step", cases, ids=ids)


def test_at_least_one_preset_ships():
    assert BUILTIN_PRESETS, "no presets found in taters/pipelines/presets"


# --- whole-file checks ------------------------------------------------------

def test_preset_is_valid_yaml_with_steps(preset_path):
    data = load(preset_path)
    assert isinstance(data, dict)
    assert data.get("steps"), f"{preset_path.name} has no steps"


def test_preset_declares_metadata(preset_path):
    """--list-presets and the docs site both read this block."""
    meta = load(preset_path).get("meta", {})
    assert meta.get("id"), "meta.id is required for --preset lookups"
    assert meta.get("title")


def test_documented_variables_match_the_vars_block(preset_path):
    """
    `meta.variables` is what users see in --describe-preset; `vars` is what the
    runner actually uses. Documenting a variable that does not exist sends
    people chasing a setting that does nothing.
    """
    data = load(preset_path)
    documented = set((data.get("meta", {}) or {}).get("variables", {}) or {})
    actual = set(data.get("vars", {}) or {})
    assert documented <= actual, f"documented but unused: {sorted(documented - actual)}"


def test_every_artifact_reference_resolves(preset_path):
    """
    Walks the steps in order, tracking which names exist. Catches typos like
    `{{pick:diar.csv}}` when the artifact is really `diar.raw_files.csv`, and
    global steps reaching for item-scoped artifacts (which they cannot see).
    """
    data = load(preset_path)
    known_vars = set(data.get("vars", {}) or {})
    item_artifacts: set[str] = set()
    global_artifacts: set[str] = set()

    problems: list[str] = []
    for i, step in enumerate(data.get("steps", []) or [], start=1):
        scope = step.get("scope", "item")
        visible = (item_artifacts | global_artifacts) if scope == "item" else set(global_artifacts)

        for expr in templates_in(step.get("with", {})):
            if expr in SPECIAL_NAMES:
                continue
            if expr.startswith("var:"):
                name = expr.split(":", 1)[1]
                if name not in known_vars:
                    problems.append(f"step {i}: undefined variable {{{{var:{name}}}}}")
                continue
            if expr.startswith("global."):
                continue
            name = expr.split(":", 1)[1].split(".")[0] if expr.startswith("pick:") else expr.split(".")[0]
            if name not in visible:
                problems.append(
                    f"step {i} ({scope}): references '{name}', which is not a "
                    f"{'prior' if scope == 'item' else 'prior global'} artifact"
                )

        if "save_as" in step:
            (item_artifacts if scope == "item" else global_artifacts).add(step["save_as"])

    assert not problems, "\n".join(problems)


# --- per-step checks --------------------------------------------------------

def test_step_has_a_call_and_a_valid_scope(preset_path, step_number, step):
    assert step.get("call"), f"step {step_number} has no call"
    assert step.get("scope", "item") in {"item", "global"}


def test_step_call_target_exists(preset_path, step_number, step):
    assert callable(underlying(step["call"]))


def test_step_parameters_match_the_function_signature(preset_path, step_number, step):
    """The check that catches a renamed parameter before a batch run does."""
    func = underlying(step["call"])
    keys = list((step.get("with") or {}).keys())
    try:
        inspect.signature(func).bind_partial(**{k: None for k in keys})
    except TypeError as exc:
        pytest.fail(f"{step['call']}: {exc}")


def test_required_parameters_are_actually_supplied(preset_path, step_number, step):
    """A `require:` naming a key that is not in `with:` can never be satisfied."""
    supplied = set((step.get("with") or {}).keys())
    for name in step.get("require", []) or []:
        assert name in supplied, f"require '{name}' is not present in with:"


def test_step_engine_is_recognized(preset_path, step_number, step):
    assert step.get("engine", "thread") in {"thread", "process"}
