"""
"Run a saved pipeline" -- pick one that already exists and run it.

This is the task that makes the pipelines shipping with Taters actually
reachable. ``conversation_video`` and ``single_speaker_media`` encode a lot of
hard-won wiring, and until now the only way to reach either was to know they
existed and type a command-line flag -- which is precisely what the audience
for this wizard will not do.

It also closes the loop on the extract task: build a pipeline once, then re-run
it next week over a different folder without rebuilding or remembering
anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..prompts import Cancelled, Choice, Prompter
from . import Task, TaskContext

__all__ = ["TASK", "choose_preset", "ask_var_overrides"]


def _needs_inputs(preset: dict) -> bool:
    """Whether the preset fans out over files, and so needs a folder to scan."""
    return any(
        (step or {}).get("scope", "item") == "item"
        for step in (preset.get("steps") or [])
    )


def choose_preset(prompter: Prompter, cwd: Optional[Path] = None) -> tuple:
    """
    Offer every visible pipeline and return the ``(path, meta)`` chosen.

    Built-ins are labeled as such. Without that, "Conversation video" sitting
    beside "My pilot study" looks like something the user made and forgot.
    """
    from ...pipelines.run_pipeline import available_presets, is_builtin_preset

    # rooted where the app was pointed, not where the shell happens to be --
    # with `taters --dir X` the wizard saves under X, so we have to look there.
    found = available_presets(cwd)
    if not found:
        prompter.note("  No pipelines found.", style="yellow")
        raise Cancelled()

    choices = []
    for path, meta in found:
        tag = "  (built in)" if is_builtin_preset(path) else ""
        summary = str(meta.get("summary") or "").strip()
        choices.append(Choice(str(path), f"{meta.get('title', path.stem)}{tag}", summary))

    picked = prompter.select("Which pipeline?", choices)
    return next((p, m) for p, m in found if str(p) == str(picked))


def _recipes_behind(preset: dict):
    """
    Map each preset step back to the catalog recipe that made it, or None.

    A wizard-made preset matches on (save_as, call) -- the same rule the
    review table uses. A hand-written preset may not match at all, and that is
    fine: the caller falls back to the flat variable editor for those.
    """
    from ..recipes import RECIPES

    matched = []
    for step in preset.get("steps") or []:
        step = step or {}
        recipe = next((r for r in RECIPES
                       if r.save_as == step.get("save_as")
                       and r.call == step.get("call")), None)
        matched.append(recipe)
    return matched


def _tune_saved_preset(prompter: Prompter,
                       preset: dict) -> Tuple[Dict[str, Any], int]:
    """
    The wizard's own options screen, over a saved pipeline.

    Saying "yes" to changing settings used to march straight through the flat
    list of global variables -- no way to pick a step, none of the per-step
    settings at all. A wizard-made preset maps cleanly back onto its recipes,
    so it gets the full screen: choose the step, see current values, edit.
    Per-step edits are written into the loaded preset dict (this run only, and
    the file on disk is untouched); variable edits come back as the runtime
    overrides the caller already merges.
    """
    from ..compose import _apply_overrides
    from ..wizard import ask_tuning

    matched = _recipes_behind(preset)
    recipes = [r for r in matched if r is not None]
    if not recipes or len(recipes) != len(set(r.id for r in recipes)):
        # unknown or duplicated steps: the per-step screen can't address them
        # unambiguously, so the flat editor -- which can -- takes over.
        flat = ask_var_overrides(prompter, preset, ask_gate=False)
        return flat, len(flat)

    # the preset's own values are the current values: its vars overlay the
    # meta defaults, and any step setting that differs from the recipe's
    # template comes in pre-seeded as that step's override, so the screen
    # shows what this pipeline actually does -- not what the catalog would
    # default to.
    var_specs = {k: dict(v) for k, v in
                 ((preset.get("meta") or {}).get("variables") or {}).items()}
    for key, value in (preset.get("vars") or {}).items():
        var_specs.setdefault(key, {})["default"] = value

    overrides: Dict[str, Dict[str, Any]] = {}
    for step, recipe in zip(preset.get("steps") or [], matched):
        if recipe is None or not step:
            continue
        for key, value in (step.get("with") or {}).items():
            template = recipe.with_.get(key)
            if value != template and not (
                    isinstance(template, str) and template.startswith("{{")):
                overrides.setdefault(recipe.id, {})[key] = value
    seeded = {rid: dict(vals) for rid, vals in overrides.items()}

    var_values: Dict[str, Any] = {}
    ask_tuning(prompter, recipes, var_specs, overrides, var_values,
               ask_gate=False)

    # step edits land on the loaded dict; the seeds that came back unchanged
    # just write their own values over themselves, which is a no-op.
    step_edits = 0
    for step, recipe in zip(preset.get("steps") or [], matched):
        if recipe is None or not step:
            continue
        _apply_overrides(step, overrides.get(recipe.id, {}))
        before = seeded.get(recipe.id, {})
        step_edits += sum(1 for k, v in overrides.get(recipe.id, {}).items()
                          if before.get(k, object()) != v)
    return var_values, len(var_values) + step_edits


def ask_var_overrides(prompter: Prompter, preset: dict,
                      ask_gate: bool = True) -> Dict[str, Any]:
    """
    Offer to change the pipeline's settings before running it.

    Only the declared variables are offered. Everything else in a preset is
    wiring, and a UI that invites someone to edit wiring is a UI that lets them
    quietly disconnect their own pipeline.
    """
    declared = (preset.get("meta", {}) or {}).get("variables", {}) or {}
    current = preset.get("vars", {}) or {}
    if not declared:
        return {}

    rows = [
        [name, str(current.get(name, spec.get("default", ""))), str(spec.get("desc", ""))]
        for name, spec in declared.items()
    ]
    prompter.table("Settings", rows, ["Setting", "Value", "What it does"])

    if ask_gate and not prompter.confirm("Change any of these?", default=False):
        return {}

    out: Dict[str, Any] = {}
    for name, spec in declared.items():
        shown = current.get(name, spec.get("default", ""))
        _explain_var(prompter, str(spec.get("desc", "")), shown)

        # a yes/no setting asked as free text is a setting most people get
        # wrong: `overwrite_existing` shows up with "False" pre-filled, and the
        # obvious edit -- clearing it -- means "leave it alone", not "no".
        if isinstance(shown, bool):
            answer = bool(prompter.confirm(f"{name}?", default=shown))
            if answer != shown:
                out[name] = answer
            continue

        answer = str(prompter.text(f"{name}:",
                                   default="" if shown is None else str(shown))).strip()
        if answer == "" or answer == str(shown):
            continue
        out[name] = _coerce(answer, shown)
    return out


def _explain_var(prompter: Prompter, desc: str, current: Any) -> None:
    """
    Say what a setting is before asking for it.

    The table above the "Change any of these?" question carries the
    descriptions, but answering that question wipes the screen -- so by the
    time anyone is actually typing a value, the only thing left on screen is
    the variable's name. `speaker_label` and `overwrite_existing` are not
    self-explanatory to the person this menu exists for.
    """
    from ..wizard import _describe_value

    prompter.note("")
    if desc:
        for paragraph in desc.split("\n\n"):
            prompter.note(f"    {paragraph}", style="dim")
    value = _describe_value(current)
    if value:
        prompter.note(f"    (currently {value})", style="dim")


def _coerce(answer: str, like: Any) -> Any:
    """
    Match the type of the existing value.

    A preset variable that was `false` and comes back as the *string* "false"
    is worse than useless: every truthiness test downstream then passes.
    """
    if isinstance(like, bool):
        return answer.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(like, int) and not isinstance(like, bool):
        try:
            return int(answer)
        except ValueError:
            return answer
    if isinstance(like, float):
        try:
            return float(answer)
        except ValueError:
            return answer
    if like is None:
        # a null default gives us no type to match, but "none"/"null" typed
        # back has to mean null -- as a string it reached the diarization
        # subprocess as a speaker count of literally "none".
        if answer.strip().lower() in {"", "none", "null"}:
            return None
        for caster in (int, float):
            try:
                return caster(answer)
            except ValueError:
                continue
    return answer


def _ask_inputs(prompter: Prompter, preset: dict) -> tuple:
    """Ask for the folder and file type a fan-out pipeline needs."""
    from ...pipelines.run_pipeline import discover_inputs
    from ..wizard import FILE_TYPES, _ask_folder, _suffixes_for

    while True:
        file_type = str(prompter.select("What kind of files should it look for?", FILE_TYPES))
        root = _ask_folder(prompter, "Where is your data?", _suffixes_for(file_type))
        inputs = discover_inputs(root, file_type)
        if inputs:
            prompter.note(f"  Found {len(inputs)} file(s) under {root}.", style="green")
            return root, file_type, inputs

        prompter.note(f"  No {file_type} files under {root}.", style="yellow")
        if not prompter.confirm("Try again?", default=True):
            raise Cancelled()


def _run(ctx: TaskContext):

    from ...pipelines.run_pipeline import merge_vars
    from ..wizard import ask_workers, execute_preset, repro_command

    prompter = ctx.prompter
    prompter.stage("pick", "Pipeline", status="active")

    path, meta = choose_preset(prompter, ctx.cwd)
    from .manage import load_preset_or_explain
    preset = load_preset_or_explain(prompter, Path(path))
    if preset is None:
        raise Cancelled()
    prompter.stage("pick", "Pipeline", status="done", detail=str(meta.get("title", "")))

    summary = str(meta.get("summary") or "").strip()
    if summary:
        prompter.note(f"\n  {summary}", style="dim")

    root_dir: Optional[Path] = None
    file_type = "any"
    inputs: List[Path] = []
    if _needs_inputs(preset):
        prompter.stage("inputs", "Data", status="active")
        root_dir, file_type, inputs = _ask_inputs(prompter, preset)
        prompter.stage("inputs", "Data", status="done",
                       detail=f"{root_dir}  ({len(inputs)} file(s))")
    else:
        prompter.note("  This pipeline reads its input from its own settings.", style="dim")

    prompter.stage("settings", "Settings", status="active")
    tunable = any(_recipes_behind(preset)) or (
        (preset.get("meta") or {}).get("variables") or {})
    if tunable and prompter.confirm("Change any settings first?", default=False):
        overrides, changed = _tune_saved_preset(prompter, preset)
    else:
        overrides, changed = {}, 0
    prompter.stage("settings", "Settings", status="done",
                   detail="unchanged" if not changed else f"{changed} changed")

    if not prompter.confirm("Run it now?", default=True):
        raise Cancelled()

    workers = ask_workers(
        prompter,
        # the runner defaults a missing `scope:` to "item"
        # (run_pipeline._run_preset does the same) -- when we checked without
        # the default, hand-written presets that left it out ran
        # single-threaded over hundreds of files, silently. `(step or {})`
        # because a stray `- ` in YAML parses to None.
        fans_out=any((step or {}).get("scope", "item") == "item"
                     for step in preset.get("steps", [])))

    prompter.stage("run", "Run", status="active")

    # a pipeline with a folder of its own runs inside it, so a re-run puts its
    # results where the first run put them instead of in whatever directory
    # `taters` happened to be started from.
    folder = Path(path).parent if Path(path).parent.name == Path(path).stem else ctx.cwd

    ok, _manifest = execute_preset(
        prompter, preset,
        root_dir=root_dir,
        file_type=file_type,
        workers=workers,
        work_dir=folder,
        preset_name=str(meta.get("id", Path(path).stem)),
        # merged, not substituted: `run_preset` takes `vars_ctx` as the whole
        # variable context, so handing it only the settings someone changed
        # would wipe out every one they left alone -- and the first step to
        # reference an untouched variable then dies on "not found".
        vars_ctx=merge_vars(dict(preset.get("vars", {}) or {}), overrides),
        command=repro_command(Path(path), root_dir=root_dir,
                              file_type=file_type, workers=workers,
                              overrides=overrides),
    )
    return ok


TASK = Task(
    id="run_saved",
    label="Run a saved pipeline",
    help="Re-run something you built earlier, or one that ships with Taters",
    run=_run,
)
