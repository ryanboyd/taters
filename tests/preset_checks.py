"""
Reusable static checks for a preset, shared by two test modules.

`test_presets.py` runs these over the presets that ship with Taters.
`test_compose.py` runs the *same* checks over presets the wizard builds. That
sharing is the point: it means the composer cannot emit a step that names a
function which does not exist, or a parameter that was renamed, or an
``{{artifact}}`` that no earlier step produces. Those are exactly the mistakes
you otherwise discover an hour into a batch run.

This is a helper module, not a test module -- it defines no `test_*` functions,
so pytest imports it only when something asks for it.
"""

from __future__ import annotations

import importlib
import inspect
import re
from typing import Any, Callable, List

import pytest

from taters import Taters
from taters.pipelines.run_pipeline import resolve_call

# the runner can resolve these on its own, no artifact needed
SPECIAL_NAMES = {"input", "cwd", "run_manifest"}
TEMPLATE_RE = re.compile(r"\{\{([^}]+)\}\}")


def underlying(call: str) -> Callable:
    """
    The real function a `call:` points at.

    `potato.*` targets go through the facade, which forwards with **kwargs — so
    to check parameter names we have to look at what it forwards *to*.
    """
    if not call.startswith("potato."):
        return resolve_call(call, Taters())
    parts = call.split(".")[1:]
    if len(parts) == 1:
        # a top-level facade method. `potato.score_with_model` lives outside any
        # namespace on purpose: the whole point is that you shouldn't have to know
        # what kind of model you've got
        owner, method = Taters, parts[0]
    else:
        namespace, method = parts[:2]
        owner = type(getattr(Taters(), namespace))
    src = inspect.getsource(getattr(owner, method))
    rel, name = re.search(r"from (\S+) import (\w+)", src).groups()
    try:
        module = importlib.import_module("taters" + rel)
    except ImportError as exc:
        pytest.skip(f"{call} needs an optional dependency: {exc}")
    return getattr(module, name)


def templates_in(value: Any) -> List[str]:
    """Every {{expression}} appearing anywhere inside a nested structure."""
    found: List[str] = []
    if isinstance(value, dict):
        for v in value.values():
            found += templates_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            found += templates_in(v)
    elif isinstance(value, str):
        found += [m.strip() for m in TEMPLATE_RE.findall(value)]
    return found


def check_artifact_references(data: dict) -> List[str]:
    """
    Walk the steps in order, tracking which names exist at each point.

    Catches typos like `{{pick:diar.csv}}` when the artifact is really
    `diar.raw_files.csv`, undefined `{{var:...}}` lookups, and global steps
    reaching for item-scoped artifacts, which they cannot see.

    Returns
    -------
    list[str]
        One human-readable line per problem; empty when the preset is sound.
    """
    known_vars = set(data.get("vars", {}) or {})
    item_artifacts: set[str] = set()
    global_artifacts: set[str] = set()
    problems: List[str] = []

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
            name = (expr.split(":", 1)[1].split(".")[0] if expr.startswith("pick:")
                    else expr.split(".")[0])
            if name not in visible:
                problems.append(
                    f"step {i} ({scope}): references '{name}', which is not a "
                    f"{'prior' if scope == 'item' else 'prior global'} artifact"
                )

        if "save_as" in step:
            (item_artifacts if scope == "item" else global_artifacts).add(step["save_as"])

    return problems


def check_step(step: dict, number: int = 0) -> List[str]:
    """Structural and signature checks for a single step."""
    problems: List[str] = []
    label = f"step {number}" if number else "step"

    call = step.get("call")
    if not call:
        return [f"{label} has no call"]
    if step.get("scope", "item") not in {"item", "global"}:
        problems.append(f"{label}: invalid scope {step.get('scope')!r}")
    if step.get("engine", "thread") not in {"thread", "process"}:
        problems.append(f"{label}: unknown engine {step.get('engine')!r}")

    for key in ("workers", "max_workers"):
        if key in step:
            try:
                if int(step[key]) < 1:
                    problems.append(f"{label}: {key} must be at least 1")
            except (TypeError, ValueError):
                problems.append(f"{label}: {key} is not a number ({step[key]!r})")

    # `global` steps are a single call, so a worker ceiling on one means nothing.
    # harmless, but it tells us somebody misunderstood the key, and we'd rather
    # say so before it gets copy-pasted around
    if "max_workers" in step and step.get("scope", "item") != "item":
        problems.append(f"{label}: max_workers has no meaning on a global step")

    func = underlying(call)
    if not callable(func):
        return problems + [f"{label}: {call} is not callable"]

    keys = list((step.get("with") or {}).keys())
    try:
        inspect.signature(func).bind_partial(**{k: None for k in keys})
    except TypeError as exc:
        problems.append(f"{label} ({call}): {exc}")

    supplied = set(keys)
    for name in step.get("require", []) or []:
        if name not in supplied:
            problems.append(f"{label}: require '{name}' is not present in with:")

    return problems


def check_metadata(data: dict) -> List[str]:
    """`meta` is what --list-presets and the docs site read."""
    problems: List[str] = []
    meta = data.get("meta", {}) or {}
    if not meta.get("id"):
        problems.append("meta.id is required for --preset lookups")
    if not meta.get("title"):
        problems.append("meta.title is required")

    documented = set(meta.get("variables", {}) or {})
    actual = set(data.get("vars", {}) or {})
    extra = sorted(documented - actual)
    if extra:
        problems.append(f"meta.variables documents unused variables: {extra}")
    return problems


def check_aggregations(data: dict) -> List[str]:
    """
    Catch an aggregation whose grouping key can never group anything.

    From a real run: `feature_gather` with
    ``group_by: ["text_id", "source", "speaker"]`` over row-level sentence
    embeddings. `text_id` is unique per utterance there, so every group had
    exactly one member and the "aggregate" step wrote a copy of its input --
    right shape, right columns, real numbers, just not averaged. Nothing failed
    and nothing warned.

    The rule is a static one and holds either way round. If `text_id` already
    identifies a row, adding more keys cannot change the grouping, so they are
    noise. If it does not, then including it splits apart the very groups the
    other keys were chosen to merge. So `text_id` is legitimate only when it is
    the *whole* key.
    """
    problems: List[str] = []
    for i, step in enumerate(data.get("steps") or [], start=1):
        if not str(step.get("call", "")).endswith("feature_gather"):
            continue
        params = step.get("with") or {}
        if not params.get("aggregate"):
            continue
        group_by = list(params.get("group_by") or [])
        if "text_id" in group_by and len(group_by) > 1:
            others = [g for g in group_by if g != "text_id"]
            problems.append(
                f"step {i}: aggregating by {group_by} groups nothing -- text_id is "
                f"unique per row, so each group has one member. Drop it and group "
                f"by {others}."
            )
    return problems


def assert_valid_preset(data: dict) -> None:
    """
    Fail with every problem named, or pass silently.

    :func:`check_preset` returns its findings, and a bare
    ``check_preset(preset)`` at the end of a test therefore checks nothing --
    ten of them stood in the suite as the structural safety net for the
    statistics tail, and would have let a step referencing an artifact
    nothing produces sail through.
    """
    problems = check_preset(data)
    assert not problems, "\n".join(problems)


def check_preset(data: dict) -> List[str]:
    """
    Every static check, over a whole preset.

    Returns
    -------
    list[str]
        Empty when the preset would run. Assert on this directly; the strings
        are written to be readable in a failure message.
    """
    if not isinstance(data, dict):
        return ["preset is not a mapping"]
    steps = data.get("steps") or []
    if not steps:
        return ["preset has no steps"]

    problems = check_metadata(data)
    problems += check_artifact_references(data)
    problems += check_aggregations(data)
    for i, step in enumerate(steps, start=1):
        problems += check_step(step, i)
    return problems
