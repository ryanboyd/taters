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

# ============== JSON-safe casting ==============

def _json_safe(obj: Any) -> Any:
    """
    Convert arbitrary Python objects into JSON-serializable structures.

    This helper is used when writing the run manifest and step results
    to disk. It handles a few common non-JSON types and normalizes them:

    - `pathlib.Path` → `str`
    - `dataclasses.dataclass` → nested `dict` via `asdict(...)`
    - Arbitrary objects with `__dict__` → `vars(obj)` (best-effort)
    - `list` / `tuple` / `set` and `dict` → deep-converted recursively

    Parameters
    ----------
    obj : Any
        The object to normalize.

    Returns
    -------
    Any
        A structure that `json.dumps(...)` can serialize (strings, numbers,
        booleans, `None`, lists, and dicts). Any unknown objects fall
        through unchanged (letting `json` raise if it still cannot serialize).
    """
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
    """
    Recursively discover input files under a root folder.

    The preset's ITEM-scoped steps operate over a list of inputs. This
    function builds that list by scanning `root_dir` and selecting files by
    type:

    - kind == "video": only common video extensions (e.g., .mp4, .mov, .mkv)
    - kind == "audio": only common audio extensions (e.g., .wav, .mp3, .flac)
    - kind == "any":   all files

    Parameters
    ----------
    root_dir : Path
        Directory to scan (will be resolved to an absolute path).
    kind : {"audio","video","any"}
        Filter that determines which file extensions are included.

    Returns
    -------
    List[Path]
        Sorted list of absolute file paths.

    Raises
    ------
    FileNotFoundError
        If `root_dir` does not exist.
    """
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
    """
    Load a named pipeline preset from `taters/pipelines/presets/<name>.yaml`.

    Parameters
    ----------
    name : str
        Basename of a YAML preset in the `presets/` directory.

    Returns
    -------
    dict
        Parsed YAML as a Python dictionary. Returns `{}` for an empty file.

    Raises
    ------
    FileNotFoundError
        If the preset file does not exist.
    """
    here = Path(__file__).parent
    path = here / "presets" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_yaml_file(path: Path) -> dict:
    """
    Load a YAML file into a Python dictionary.

    Parameters
    ----------
    path : Path
        Full path to a YAML file.

    Returns
    -------
    dict
        Parsed YAML contents. Empty files yield `{}`.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merge_vars(base: dict, overlay: dict) -> dict:
    """
    Shallow-merge two variable dictionaries.

    Later sources of variables (e.g., `--vars-file`, then repeated `--var`
    overrides) should replace keys from earlier sources. This helper
    applies a simple `dict.update(...)` and returns a new dictionary.

    Parameters
    ----------
    base : dict
        The starting dictionary of variables.
    overlay : dict
        The dictionary whose keys override entries in `base`.

    Returns
    -------
    dict
        A new dictionary with merged keys/values.
    """
    out = dict(base or {})
    out.update(overlay or {})
    return out

def parse_var_overrides(pairs: List[str]) -> dict:
    """
    Parse `--var key=value` CLI overrides into typed Python values.

    Typing rules:
      - "true"/"false" (case-insensitive) → bool
      - "null"/"none" (case-insensitive)  → None
      - integer or float strings → numeric
      - all else → raw string

    Parameters
    ----------
    pairs : List[str]
        CLI arguments of the form `["k1=v1", "k2=v2", ...]`.

    Returns
    -------
    dict
        Mapping from variable name to parsed value.

    Raises
    ------
    ValueError
        If any entry does not contain an '=' separator.
    """
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
    """
    Resolve a dotted attribute/key path within nested dicts/objects.

    This is used by templating expressions like `{{global.some.nested.value}}`
    or `{{pick:artifact.path.to.field}}`.

    Resolution order per path segment:
      1) If `d` is a dict and has the key → descend by key
      2) Else if `d` has an attribute with that name → use `getattr`
      3) Otherwise → KeyError

    Parameters
    ----------
    d : Any
        Root object/dict to traverse.
    dotted : str
        Dotted path, e.g. `"a.b.c"`.

    Returns
    -------
    Any
        The resolved value.

    Raises
    ------
    KeyError
        If any path segment cannot be resolved.
    """
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
    """
    Evaluate a single templating expression and return a native Python value.

    Supported expressions:
      - `input`      → the current item's input path (string)
      - `cwd`        → current working directory (string)
      - `var:<key>`  → look up `vars[<key>]`
      - `global.<p>` → deep lookup on the globals context
      - `pick:<artifact>.<path>` → deep lookup inside an item artifact
      - `<artifact>` → direct lookup in the item's artifact dict

    Notes
    -----
    If the expression cannot be resolved by any rule, the raw `{{...}}`
    string is returned as a literal—this allows progressive templating or
    clearer error messages at call time.

    Parameters
    ----------
    expr : str
        Expression content without the surrounding `{{` and `}}`.
    item_ctx : dict
        Per-item artifact dictionary (things saved via `save_as`).
    globals_ctx : dict
        Global artifact dictionary (set by GLOBAL steps).
    vars_ctx : dict
        Variables merged from preset → vars file → `--var` flags.
    input_path : Path
        Path of the current input item, for `input`.

    Returns
    -------
    Any
        Native Python value that represents the expression.
    """
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
    """
    Render templating expressions within a value (str, list, or dict).

    Behavior:
      * If `val` is a dict or list/tuple, render recursively.
      * If `val` is a string and the ENTIRE string is a single template
        (e.g., `\"{{var:text_cols}}\"`), return the native value
        (list, int, bool, etc.) of that expression.
      * Otherwise, perform string substitution on all `{{...}}` occurrences
        and return the string result.

    This “native-type preserving” rule is crucial for YAML presets:
    ```
      text_cols: "{{var:text_cols}}"  # becomes a list, not the string "['text']"
    ```

    Parameters
    ----------
    val : Any
        The value to render (string, list/tuple, dict, or other).
    item_ctx : dict
        Per-item artifacts for `{{pick:...}}` and artifact references.
    globals_ctx : dict
        Globals for `{{global...}}` lookups.
    vars_ctx : dict
        Variables context for `{{var:...}}` lookups.
    input_path : Path
        The current input item (for `{{input}}`).

    Returns
    -------
    Any
        Rendered value, with native types preserved where applicable.
    """
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
    Resolve a call target from a preset step into an actual callable.

    Supported forms
    ---------------
    1) Taters instance methods (recommended):
       - `"potato.audio.convert_to_wav"`
       - `"potato.text.analyze_with_dictionaries"`
       The function is resolved via attribute chaining on a single
       `Taters()` instance created for the whole run.

    2) Dotted import paths:
       - `"package.module:function"`
       - `"package.module.func"`
       - `"package.module.Class.method"`
       The target is imported and attributes are resolved. The final target
       must be callable.

    Parameters
    ----------
    call_name : str
        Call string from the preset step's `call:` field.
    potato : Taters
        The shared `Taters` instance for resolving `"potato.*"` calls.

    Returns
    -------
    Callable
        The function/object that will be invoked for the step.

    Raises
    ------
    AttributeError, KeyError, TypeError
        If the target cannot be resolved or is not callable.
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
    """
    Execute a single ITEM-scoped step for one input path.

    Lifecycle
    ---------
    1) Template the step's `with:` parameters using `render_value(...)`.
    2) Validate any `require:` keys after templating (fail fast if missing).
    3) Resolve the callable (Taters method or import path).
    4) Invoke with keyword arguments.
    5) If the step specified `save_as: <name>`, store the return value under
       that name in the item's `artifacts` dict.

    Parameters
    ----------
    step : dict
        The step definition block from the preset.
    input_path : Path
        The current input file for ITEM scope.
    potato : Taters
        Shared Taters instance used to call `potato.*` targets.
    item_artifacts : Dict[str, Any]
        The current item's artifact dictionary (mutated across steps).
    globals_ctx : Dict[str, Any]
        Global artifacts (from GLOBAL steps).
    vars_ctx : Dict[str, Any]
        Merged variables.

    Returns
    -------
    Tuple[str, Dict[str, Any], Dict[str, Any]]
        A tuple `(status, new_artifacts, err)` where:
          - `status` is `"ok"` or `"error"`.
          - `new_artifacts` is a (possibly empty) dict of artifacts to merge.
          - `err` contains an `"error"` message on failure.
    """
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
    """
    Execute a single GLOBAL-scoped step (runs once per pipeline).

    Differences from ITEM steps
    ---------------------------
    - The templating `item_ctx` is empty.
    - The run manifest path is injected into `vars` as `run_manifest`,
      so presets can reference it in GLOBAL stages.
    - On success, any values from `save_as:` are merged into the `globals`
      artifact map.

    Parameters
    ----------
    step : dict
        The step definition block from the preset.
    potato : Taters
        Shared Taters instance used to call `potato.*` targets.
    globals_ctx : Dict[str, Any]
        Accumulated global artifacts (readable by later steps).
    vars_ctx : Dict[str, Any]
        Merged variables.
    manifest_path : Path
        Path where the JSON run manifest is (or will be) saved.

    Returns
    -------
    Tuple[str, Dict[str, Any], Dict[str, Any]]
        A tuple `(status, new_globals, err)` mirroring the ITEM step shape.
    """
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
    """
    Entry point for the Taters Pipeline Runner.

    Responsibilities
    ----------------
    - Parse CLI arguments (`--preset` or `--preset-file`, optional `--vars-file`
      and repeated `--var key=value` overrides, `--workers`, etc.).
    - Load the preset YAML and merge variables from three sources in order:
        1) preset `vars` block
        2) `--vars-file` (YAML)
        3) repeated `--var` CLI flags
    - Decide whether input discovery is required:
        * If the preset has any ITEM-scoped steps, `--root_dir` is required and
          files are discovered with `discover_inputs(...)`.
        * If there are only GLOBAL steps, discovery is skipped entirely.
    - Build a run manifest skeleton (preset name, inputs, vars, globals).
    - Create a single `Taters()` instance (shared across all steps in the run).
    - Execute each step in order:
        * ITEM steps: fan out across discovered inputs using a thread or process
          pool (configurable per step). A given step reuses one pool for all
          items to amortize worker startup.
        * GLOBAL steps: run once, in order, with a barrier between steps.
    - After each step, update and persist the JSON manifest so long-running runs
      are observable and resumable.
    - Print the final manifest path on completion.

    Concurrency Notes
    -----------------
    - The default executor for ITEM steps is a `ThreadPoolExecutor` (good for
      I/O-bound steps and for GPU inference that releases the GIL).
    - For heavy Python/CPU work, presets may set `engine: process` on a step to
      use a `ProcessPoolExecutor`. In that case, be mindful that a new Python
      process is spawned for each worker (model weights may be reloaded once per
      worker).

    Error Handling
    --------------
    - Individual ITEM step failures do not crash the pipeline; they mark that
      item as `"error"` in the manifest and continue.
    - GLOBAL step failures are terminal for the run (the loop breaks).

    Returns
    -------
    None
        The function exits the process after writing the manifest.
    """
    # ---------------------------
    # CLI
    # ---------------------------
    ap = argparse.ArgumentParser(description="Taters Pipeline Runner (robust templating + flexible calls)")
    ap.add_argument("--root_dir", default=None, help="Folder to scan for inputs (required only if preset has ITEM steps)")
    ap.add_argument("--file_type", default="any", choices=["audio", "video", "any"], help="Input type filter for discovery")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--preset", help="Preset name (taters/pipelines/presets/<name>.yaml)")
    group.add_argument("--preset-file", dest="preset_file", help="Path to preset YAML")
    ap.add_argument("--vars-file", dest="vars_file", help="YAML file with 'vars' overrides")
    ap.add_argument("--var", action="append", default=[], help="Single override key=value (repeatable)")
    ap.add_argument("--workers", type=int, default=4, help="Concurrency for ITEM steps")
    ap.add_argument("--out-manifest", dest="out_manifest", default=None,
                    help="Run manifest (JSON). Default: ./run_manifest.json")
    args = ap.parse_args()

    # ---------------------------
    # Load preset and vars first
    # ---------------------------
    preset = load_preset_by_name(args.preset) if args.preset else load_yaml_file(Path(args.preset_file))
    steps: List[dict] = preset.get("steps", []) or []
    if not steps:
        raise ValueError("Preset has no steps")

    vars_ctx: Dict[str, Any] = dict(preset.get("vars", {}) or {})
    if args.vars_file:
        vars_ctx = merge_vars(vars_ctx, load_yaml_file(Path(args.vars_file)))
    vars_ctx = merge_vars(vars_ctx, parse_var_overrides(args.var))

    # ---------------------------
    # Decide if discovery is needed
    # ---------------------------
    has_item_steps = any((step.get("scope", "item") == "item") for step in steps)

    inputs: List[Path] = []
    root_dir: Path | None = None
    if has_item_steps:
        if not args.root_dir:
            raise ValueError("--root_dir is required because this preset contains ITEM-scoped steps.")
        root_dir = Path(args.root_dir).resolve()
        inputs = discover_inputs(root_dir, args.file_type)
        print(f"[pipeline] Found {len(inputs)} '{args.file_type}' input(s) under {root_dir}")
        if not inputs:
            print("[pipeline] No inputs found; ITEM steps will be skipped.")
    else:
        print("[pipeline] Preset has only GLOBAL steps; skipping input discovery.")

    # ---------------------------
    # Build manifest skeleton
    # ---------------------------
    manifest: Dict[str, Any] = {
        "preset": args.preset or str(args.preset_file),
        "root_dir": str(root_dir) if root_dir else None,
        "file_type": args.file_type if has_item_steps else None,
        "vars": _json_safe(vars_ctx),
        "items": [{"input": str(p), "artifacts": {}, "status": "pending", "errors": []} for p in inputs],
        "globals": {},
        "errors": [],
    }
    out_manifest_path = Path(args.out_manifest or (Path.cwd() / "run_manifest.json"))

    # Create a single Taters instance for the whole run (correct class import)
    from taters.Taters import Taters  # ← IMPORTANT: import the class, not the module
    potato = Taters()
    globals_ctx: Dict[str, Any] = {}

    # ---------------------------
    # Execute steps
    # ---------------------------
    for idx, step in enumerate(steps, 1):
        scope = step.get("scope", "item")
        call_name = step.get("call")
        print(f"[pipeline] Step {idx}/{len(steps)}: {call_name}  (scope={scope})")

        if scope == "item":
            if not inputs:
                print(f"[pipeline] No inputs; skipping ITEM step: {call_name}")
                continue

            step_engine = step.get("engine", "thread")  # "thread" (default) or "process"
            step_workers = max(1, int(step.get("workers", args.workers)))

            def _run_one(ix_and_path: Tuple[int, Path]):
                i, p = ix_and_path
                itm = manifest["items"][i]
                status, new_artifacts, err = run_item_step_for_one_input(
                    step=step,
                    input_path=p,
                    potato=potato,
                    item_artifacts=itm["artifacts"],
                    globals_ctx=globals_ctx,
                    vars_ctx=vars_ctx,
                )
                return i, status, new_artifacts, err

            results: List[Tuple[int, str, Dict[str, Any], Dict[str, Any]]] = []
            Executor = cf.ProcessPoolExecutor if step_engine == "process" else cf.ThreadPoolExecutor
            with Executor(max_workers=step_workers) as pool:
                futures = [pool.submit(_run_one, (i, p)) for i, p in enumerate(inputs)]
                for fut in cf.as_completed(futures):
                    results.append(fut.result())

            for i, status, new_artifacts, err in results:
                itm = manifest["items"][i]
                if status == "ok":
                    for k, v in (new_artifacts or {}).items():
                        itm["artifacts"][k] = _json_safe(v)
                    if itm["status"] != "error":
                        itm["status"] = "ok"
                else:
                    itm["status"] = "error"
                    itm["errors"].append(err.get("error", "unknown error"))

        elif scope == "global":
            status, new_globals, err = run_global_step(
                step=step,
                potato=potato,
                globals_ctx=globals_ctx,
                vars_ctx=vars_ctx,
                manifest_path=out_manifest_path,
            )
            if status != "ok":
                print(f"[pipeline] GLOBAL step failed: {err.get('error')}")
                manifest["errors"].append(err.get("error", "unknown error"))
                break
            for k, v in (new_globals or {}).items():
                globals_ctx[k] = v
                manifest["globals"][k] = _json_safe(v)

        else:
            raise ValueError(f"Invalid scope: {scope}")

        # Persist manifest after each step
        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)

    print(f"[pipeline] Manifest written to: {out_manifest_path}")

if __name__ == "__main__":
    main()
