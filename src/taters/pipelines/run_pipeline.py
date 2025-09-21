#!/usr/bin/env python3
"""
Taters Pipeline Runner (robust templating + flexible call resolution)

- ITEM steps run once per input (fan-out concurrently).
- GLOBAL steps run once (barrier before/after).
- Templating preserves native types when the *entire* value is a single template
  (e.g., {{var:text_cols}} → list, not "['text']").
- Calls:
    * "potato.*"      → call via a Taters() instance (e.g., potato.text.analyze_with_dictionaries)
    * dotted path → import and call any function (e.g., taters.helpers.feature_gather.aggregate_features)

Usage example:
  python -m taters.pipelines.run_pipeline \
    --root_dir videos \
    --file_type video \
    --preset conversation_video \
    --workers 4 \
    --var device=cuda --var overwrite_existing=true
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib
import json
import re
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Taters entrypoint
from taters.Taters import Taters


# ============== JSON-safe casting ==============

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass
    return obj


# ============== Discovery ==============

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wmv", ".flv", ".webm", ".mpg", ".mpeg", ".3gp"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".aifc"}

def discover_inputs(root_dir: Path, kind: str) -> List[Path]:
    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    out: List[Path] = []
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if kind == "video" and ext in _VIDEO_EXTS:
            out.append(p)
        elif kind == "audio" and ext in _AUDIO_EXTS:
            out.append(p)
        elif kind == "any":
            out.append(p)
    return sorted(out)


# ============== Preset loading / vars ==============

def load_preset_by_name(name: str) -> dict:
    here = Path(__file__).parent
    path = here / "presets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_yaml_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merge_vars(base: dict, overlay: dict) -> dict:
    out = dict(base or {})
    out.update(overlay or {})
    return out

def parse_var_overrides(pairs: List[str]) -> dict:
    out: Dict[str, Any] = {}
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"--var expects key=value, got: {s}")
        k, v = s.split("=", 1)
        vs = v.strip()
        if vs.lower() in {"true", "false"}:
            out[k] = (vs.lower() == "true")
        elif vs.lower() in {"null", "none"}:
            out[k] = None
        else:
            try:
                out[k] = float(vs) if "." in vs else int(vs)
            except Exception:
                out[k] = v
    return out


# ============== Templating (native-type preserving) ==============

_VAR_RE = re.compile(r"\{\{([^}]+)\}\}")

def _deep_get(d: Any, dotted: str) -> Any:
    cur = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            raise KeyError(f"Could not resolve '{part}' in {_json_safe(cur)}")
    return cur

def _eval_expr(expr: str, *, item_ctx: dict, globals_ctx: dict, vars_ctx: dict, input_path: Path) -> Any:
    expr = expr.strip()
    if expr == "input":
        return str(input_path)
    if expr == "cwd":
        return str(Path.cwd())
    if expr.startswith("var:"):
        key = expr.split(":", 1)[1]
        if key not in vars_ctx:
            raise KeyError(f"Variable '{key}' not found")
        return vars_ctx[key]
    if expr.startswith("global."):
        key = expr.split(".", 1)[1]
        return _deep_get(globals_ctx, key) if key else globals_ctx
    if expr.startswith("pick:"):
        path = expr.split(":", 1)[1]
        if "." not in path:
            raise KeyError("pick: requires 'artifact.nested.path'")
        art, nested = path.split(".", 1)
        if art not in item_ctx:
            raise KeyError(f"Artifact '{art}' not found")
        return _deep_get(item_ctx[art], nested)
    if expr in item_ctx:
        return item_ctx[expr]
    # fallback: return raw template if unknown
    return "{{" + expr + "}}"

def render_value(val: Any, *, item_ctx: dict, globals_ctx: dict, vars_ctx: dict, input_path: Path) -> Any:
    """Render strings, lists, dicts. If a string is *exactly* one template, return native value."""
    if isinstance(val, dict):
        return {k: render_value(v, item_ctx=item_ctx, globals_ctx=globals_ctx, vars_ctx=vars_ctx, input_path=input_path)
                for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [render_value(v, item_ctx=item_ctx, globals_ctx=globals_ctx, vars_ctx=vars_ctx, input_path=input_path)
                for v in val]
    if not isinstance(val, str):
        return val

    # If the whole string is a single {{...}}, return the native value
    m = _VAR_RE.fullmatch(val.strip())
    if m:
        native = _eval_expr(m.group(1), item_ctx=item_ctx, globals_ctx=globals_ctx, vars_ctx=vars_ctx, input_path=input_path)
        return native

    # Otherwise do string substitution with each expr -> str(value)
    def _subst(match: re.Match) -> str:
        expr = match.group(1)
        v = _eval_expr(expr, item_ctx=item_ctx, globals_ctx=globals_ctx, vars_ctx=vars_ctx, input_path=input_path)
        return str(v)
    return _VAR_RE.sub(_subst, val)


# ============== Call resolver (no hard-coded mapping) ==============

def resolve_call(call_name: str, potato: Taters):
    """
    - 'potato.audio.convert_to_wav' → getattr chain on Taters instance
    - 'package.module:function' or 'package.module.func' → import and return function
    - 'package.module.Class.method' → import, getattr chain; must be callable
    """
    if call_name.startswith("potato."):
        obj: Any = potato
        for part in call_name.split(".")[1:]:
            if not hasattr(obj, part):
                raise AttributeError(f"{call_name}: '{part}' not found on {obj}")
            obj = getattr(obj, part)
        if not callable(obj):
            raise TypeError(f"{call_name} is not callable")
        return obj

    # Allow dotted import paths
    # Support both "pkg.mod:func" and "pkg.mod.func"
    mod_path, sep, tail = call_name.partition(":")
    if not sep:
        # split at last dot for function
        parts = call_name.rsplit(".", 1)
        if len(parts) == 2:
            mod_path, tail = parts
        else:
            raise KeyError(f"Cannot resolve call target: {call_name}")
    module = importlib.import_module(mod_path)
    target = module
    for attr in tail.split("."):
        if not hasattr(target, attr):
            raise AttributeError(f"{call_name}: '{attr}' not found in {target}")
        target = getattr(target, attr)
    if not callable(target):
        raise TypeError(f"{call_name} resolved to non-callable: {target}")
    return target


# ============== Step runners ==============

def run_item_step_for_one_input(
    *, step: dict, input_path: Path, potato: Taters, item_artifacts: Dict[str, Any],
    globals_ctx: Dict[str, Any], vars_ctx: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    call = step["call"]
    params = step.get("with", {})

    rendered = render_value(params, item_ctx=item_artifacts, globals_ctx=globals_ctx, vars_ctx=vars_ctx, input_path=input_path)

    # Required keys check (post-templating)
    for key in step.get("require", []):
        if key not in rendered or rendered[key] in (None, "", []):
            return ("error", {}, {"error": f"Missing required parameter '{key}' after templating"})

    func = resolve_call(call, potato)
    try:
        result = func(**rendered)
    except Exception as e:
        return ("error", {}, {"error": f"{call} failed: {e}"})

    out: Dict[str, Any] = {}
    if "save_as" in step:
        out[step["save_as"]] = result
    return ("ok", out, {})

def run_global_step(
    *, step: dict, potato: Taters, globals_ctx: Dict[str, Any], vars_ctx: Dict[str, Any], manifest_path: Path
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    call = step["call"]
    params = step.get("with", {})

    # Expose manifest path via vars
    vars_aug = dict(vars_ctx)
    vars_aug["run_manifest"] = str(manifest_path)

    rendered = render_value(params, item_ctx={}, globals_ctx=globals_ctx, vars_ctx=vars_aug, input_path=manifest_path)

    func = resolve_call(call, potato)
    try:
        result = func(**rendered)
    except Exception as e:
        return ("error", {}, {"error": f"{call} failed: {e}"})

    out: Dict[str, Any] = {}
    if "save_as" in step:
        out[step["save_as"]] = result
    return ("ok", out, {})


# ============== Main ==============

def main():
    ap = argparse.ArgumentParser(description="Taters Pipeline Runner (robust templating + flexible calls)")
    ap.add_argument("--root_dir", required=True, help="Folder to scan for inputs")
    ap.add_argument("--file_type", default="any", choices=["audio", "video", "any"], help="Input type filter")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", help="Preset name (taters/pipelines/presets/<name>.yaml)")
    group.add_argument("--preset-file", help="Path to preset YAML")
    ap.add_argument("--vars-file", help="YAML overrides for 'vars'")
    ap.add_argument("--var", action="append", default=[], help="Single override key=value (repeatable)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrency for ITEM steps")
    ap.add_argument("--out-manifest", default=None, help="Output run manifest (JSON). Default: ./run_manifest.json")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    inputs = discover_inputs(root_dir, args.file_type)
    print(f"[pipeline] Found {len(inputs)} '{args.file_type}' input(s) under {root_dir}")
    if not inputs:
        return

    preset = load_preset_by_name(args.preset) if args.preset else load_yaml_file(Path(args.preset_file))
    steps: List[dict] = preset.get("steps", [])
    if not steps:
        raise ValueError("Preset has no steps")

    vars_ctx = dict(preset.get("vars", {}))
    if args.vars_file:
        vars_ctx = merge_vars(vars_ctx, load_yaml_file(Path(args.vars_file)))
    vars_ctx = merge_vars(vars_ctx, parse_var_overrides(args.var))

    manifest: dict = {
        "preset": args.preset or str(args.preset_file),
        "root_dir": str(root_dir),
        "file_type": args.file_type,
        "vars": _json_safe(vars_ctx),
        "items": [{"input": str(p), "artifacts": {}, "status": "pending", "errors": []} for p in inputs],
        "globals": {}
    }
    out_manifest_path = Path(args.out_manifest or (Path.cwd() / "run_manifest.json"))

    potato = Taters()
    globals_ctx: Dict[str, Any] = {}

    for idx, step in enumerate(steps, 1):
        scope = step.get("scope", "item")
        print(f"[pipeline] Step {idx}/{len(steps)}: {step.get('call')}  (scope={scope})")

        if scope == "item":
            # Per-step overrides
            step_engine = step.get("engine", "thread")     # "thread" (default) or "process"
            step_workers = int(step.get("workers", args.workers))
            step_workers = max(1, step_workers)

            def _run_one(ix_and_path: Tuple[int, Path]):
                i, p = ix_and_path
                itm = manifest["items"][i]
                return (i, *run_item_step_for_one_input(
                    step=step, input_path=p, potato=potato,
                    item_artifacts=itm["artifacts"], globals_ctx=globals_ctx, vars_ctx=vars_ctx
                ))

            results: List[Tuple[int, str, Dict[str, Any], Dict[str, Any]]] = []

            # Choose executor backend per step
            ExecutorCls = cf.ProcessPoolExecutor if step_engine == "process" else cf.ThreadPoolExecutor

            # Important: keep a *single* pool for this step, so workers persist across items.
            # That way, heavyweight libs are imported once per worker process.
            with ExecutorCls(max_workers=step_workers) as pool:
                futures = [pool.submit(_run_one, (i, p)) for i, p in enumerate(inputs)]
                for fut in cf.as_completed(futures):
                    results.append(fut.result())


            for i, status, new_artifacts, err in results:
                itm = manifest["items"][i]
                if status == "ok":
                    for k, v in new_artifacts.items():
                        itm["artifacts"][k] = _json_safe(v)
                    if itm["status"] != "error":
                        itm["status"] = "ok"
                else:
                    itm["status"] = "error"
                    itm["errors"].append(err.get("error", "unknown error"))

        elif scope == "global":
            status, new_globals, err = run_global_step(
                step=step, potato=potato, globals_ctx=globals_ctx, vars_ctx=vars_ctx, manifest_path=out_manifest_path
            )
            if status != "ok":
                print(f"[pipeline] GLOBAL step failed: {err.get('error')}")
                break
            for k, v in new_globals.items():
                globals_ctx[k] = v
                manifest["globals"][k] = _json_safe(v)

        else:
            raise ValueError(f"Invalid scope: {scope}")

        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)

    print(f"[pipeline] Manifest written to: {out_manifest_path}")


if __name__ == "__main__":
    main()
