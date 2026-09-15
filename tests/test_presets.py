"""Static validation of the shipped pipeline presets.

A preset is a YAML file full of function names and parameter names. Nothing
checks it until you run it — which, for a preset whose last step is an hour in,
is a bad time to discover a typo. These tests read every preset and confirm
that everything it names actually exists and lines up, in under a second.

They are also a guard on refactoring: rename a parameter in the Python and the
preset that still passes the old name fails here immediately.
"""

import inspect
from pathlib import Path

import pytest
import yaml

# the actual checks live over in `preset_checks.py`, so that `test_compose.py`
# can run the very same validation over the presets the wizard builds
from preset_checks import check_artifact_references, underlying

BUILTIN_PRESETS = sorted((Path(__file__).parent.parent / "src" / "taters" /
                          "pipelines" / "presets").glob("*.yaml"))


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
    problems = check_artifact_references(load(preset_path))
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


# ---------------------------------------------------------------------------
# aggregations that can't actually aggregate
# ---------------------------------------------------------------------------

def test_no_preset_aggregates_by_a_key_that_is_unique_per_row(preset_path):
    """
    Both shipped presets used to group row-level sentence embeddings by
    ``["text_id", "source", "speaker"]``. `text_id` is unique per utterance, so
    every group had one member and the step emitted a copy of its input. The
    failure is invisible in the output -- right columns, real numbers -- which
    is what makes a static check worth having.
    """
    from preset_checks import check_aggregations

    problems = check_aggregations(load(preset_path))
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# concurrency limits on GPU steps
# ---------------------------------------------------------------------------

def test_a_gpu_step_carries_a_ceiling(preset_path):
    """
    A per-file step that hands `device` to a model must say how many copies of
    that model the GPU can be asked to hold. Without a ceiling, `--workers 8`
    means eight -- which is fine on `tiny` and an out-of-memory error on
    `large-v3`, halfway through a batch.

    The runner guesses 1 for an undeclared step, so this is not a correctness
    hole. But a shipped preset should not be relying on the guess: the guess
    exists for presets written by hand.
    """
    data = load(preset_path)
    for i, step in enumerate(data.get("steps") or [], 1):
        if step.get("scope", "item") != "item":
            continue
        params = step.get("with") or {}
        if not any(isinstance(v, str) and "{{var:device}}" in v
                   for v in params.values()):
            continue
        assert "max_workers" in step or "workers" in step, (
            f"step {i} ({step['call']}) puts a model on the GPU for every file "
            f"but names no worker limit"
        )


def test_no_preset_names_an_impossible_worker_count(preset_path):
    from preset_checks import check_step

    for i, step in enumerate(load(preset_path).get("steps") or [], 1):
        problems = [p for p in check_step(step, i)
                    if "workers" in p or "max_workers" in p]
        assert not problems, "\n".join(problems)
