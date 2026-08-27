"""Tests for taters.pipelines.run_pipeline.

The runner is mostly a small language: it resolves `{{...}}` expressions, looks
up call targets by name, and decides what counts as a failed step. All of that
is testable without running a single model, and it is worth testing carefully
because a templating mistake shows up as a confusing failure five steps later.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from taters import Taters
from taters.pipelines.run_pipeline import (
    _json_safe,
    _load_preset_meta,
    discover_inputs,
    load_preset_by_name,
    merge_vars,
    parse_var_overrides,
    render_value,
    resolve_call,
    resolve_preset_path,
    run_global_step,
    run_item_step_for_one_input,
)


def render(value, *, item=None, globals_=None, vars_=None, input_path="/in/file.mp4"):
    """Thin wrapper so the tests read cleanly."""
    return render_value(
        value,
        item_ctx=item or {},
        globals_ctx=globals_ or {},
        vars_ctx=vars_ or {},
        input_path=Path(input_path),
    )


# ---------------------------------------------------------------------------
# --var parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pair,expected",
    [
        ("device=cuda", {"device": "cuda"}),
        ("overwrite=true", {"overwrite": True}),
        ("overwrite=FALSE", {"overwrite": False}),
        ("num_speakers=null", {"num_speakers": None}),
        ("num_speakers=None", {"num_speakers": None}),
        ("workers=8", {"workers": 8}),
        ("threshold=0.72", {"threshold": 0.72}),
        ("model=base.en", {"model": "base.en"}),      # dotted string, not a float
        ("path=C:/data/x", {"path": "C:/data/x"}),
        ("empty=", {"empty": ""}),
    ],
)
def test_parse_var_overrides_types_values(pair, expected):
    assert parse_var_overrides([pair]) == expected


def test_parse_var_overrides_keeps_equals_signs_in_the_value():
    assert parse_var_overrides(["expr=a=b"]) == {"expr": "a=b"}


def test_parse_var_overrides_rejects_a_missing_equals():
    with pytest.raises(ValueError, match="key=value"):
        parse_var_overrides(["device"])


def test_merge_vars_lets_the_overlay_win():
    assert merge_vars({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_merge_vars_does_not_mutate_its_inputs():
    base = {"a": 1}
    merge_vars(base, {"a": 2})
    assert base == {"a": 1}


# ---------------------------------------------------------------------------
# JSON-safe conversion (used for the manifest)
# ---------------------------------------------------------------------------

@dataclass
class _Result:
    path: Path
    count: int


def test_json_safe_converts_paths_dataclasses_and_containers():
    value = {"a": Path("/tmp/x"), "b": [Path("/tmp/y"), {"c": Path("/tmp/z")}]}
    assert _json_safe(value) == {"a": "/tmp/x", "b": ["/tmp/y", {"c": "/tmp/z"}]}


def test_json_safe_flattens_dataclasses():
    assert _json_safe(_Result(path=Path("/tmp/x"), count=2)) == {"path": "/tmp/x", "count": 2}


def test_json_safe_output_actually_serializes():
    """The point of the helper is that json.dumps cannot choke on the manifest."""
    json.dumps(_json_safe({"p": Path("/a"), "r": _Result(Path("/b"), 1), "s": {1, 2}}))


def test_json_safe_leaves_primitives_alone():
    assert _json_safe([1, "two", 3.0, True, None]) == [1, "two", 3.0, True, None]


# ---------------------------------------------------------------------------
# Templating
# ---------------------------------------------------------------------------

def test_input_and_cwd_literals():
    assert render("{{input}}", input_path="/data/clip.mp4") == "/data/clip.mp4"
    assert render("{{cwd}}") == str(Path.cwd())


def test_var_lookup():
    assert render("{{var:device}}", vars_={"device": "cuda"}) == "cuda"


def test_missing_var_raises():
    with pytest.raises(KeyError, match="nope"):
        render("{{var:nope}}", vars_={})


def test_a_lone_template_preserves_the_native_type():
    """
    This is the subtle one. `text_cols: "{{var:text_cols}}"` has to arrive at the
    function as a real list, not the string "['text']".
    """
    assert render("{{var:cols}}", vars_={"cols": ["text"]}) == ["text"]
    assert render("{{var:n}}", vars_={"n": 8}) == 8
    assert render("{{var:flag}}", vars_={"flag": False}) is False
    assert render("{{var:nothing}}", vars_={"nothing": None}) is None


def test_a_template_inside_a_larger_string_becomes_text():
    out = render("{{var:dir}}/acoustics", vars_={"dir": "features"})
    assert out == "features/acoustics"


def test_several_templates_in_one_string():
    out = render("{{var:a}}-{{var:b}}", vars_={"a": "x", "b": "y"})
    assert out == "x-y"


def test_bare_artifact_name_resolves_from_the_item_first():
    assert render("{{wav}}", item={"wav": "/tmp/a.wav"}, globals_={"wav": "/other"}) == "/tmp/a.wav"


def test_bare_artifact_name_falls_back_to_globals():
    assert render("{{merged}}", globals_={"merged": "/tmp/all.csv"}) == "/tmp/all.csv"


def test_pick_walks_into_a_nested_artifact():
    diar = {"raw_files": {"csv": "/tmp/t.csv", "srt": "/tmp/t.srt"}}
    assert render("{{pick:diar.raw_files.csv}}", item={"diar": diar}) == "/tmp/t.csv"


def test_pick_also_walks_object_attributes():
    @dataclass
    class Out:
        work_dir: Path
    assert render("{{pick:d.work_dir}}", item={"d": Out(Path("/tmp/w"))}) == Path("/tmp/w")


def test_pick_on_a_missing_artifact_raises():
    with pytest.raises(KeyError):
        render("{{pick:nope.csv}}", item={})


def test_pick_requires_a_nested_path():
    with pytest.raises(KeyError, match="pick:"):
        render("{{pick:diar}}", item={"diar": {}})


def test_explicit_global_lookup():
    assert render("{{global.a.b}}", globals_={"a": {"b": 7}}) == 7


def test_unresolved_expressions_are_left_verbatim():
    """Progressive templating: an unknown name stays as-is so the eventual
    error message still shows what was not filled in."""
    assert render("{{mystery}}") == "{{mystery}}"


def test_templating_recurses_through_dicts_and_lists():
    out = render(
        {"a": ["{{var:x}}", {"b": "{{var:y}}"}], "c": 5},
        vars_={"x": 1, "y": "two"},
    )
    assert out == {"a": [1, {"b": "two"}], "c": 5}


def test_non_string_scalars_pass_through_untouched():
    assert render(42) == 42
    assert render(None) is None
    assert render(True) is True


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

@pytest.fixture
def media_tree(tmp_path) -> Path:
    root = tmp_path / "inputs"
    (root / "sub").mkdir(parents=True)
    for name in ["b.mp4", "a.mov", "clip.wav", "song.mp3", "notes.txt", "sub/deep.mkv"]:
        (root / name).touch()
    return root


def test_discover_inputs_filters_by_kind(media_tree):
    videos = {p.name for p in discover_inputs(media_tree, "video")}
    assert videos == {"a.mov", "b.mp4", "deep.mkv"}

    audio = {p.name for p in discover_inputs(media_tree, "audio")}
    assert audio == {"clip.wav", "song.mp3"}


def test_discover_inputs_any_takes_everything(media_tree):
    assert len(discover_inputs(media_tree, "any")) == 6


def test_discover_inputs_is_sorted_and_absolute(media_tree):
    found = discover_inputs(media_tree, "video")
    assert found == sorted(found)
    assert all(p.is_absolute() for p in found)


def test_discover_inputs_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_inputs(tmp_path / "nope", "video")


# ---------------------------------------------------------------------------
# Call resolution
# ---------------------------------------------------------------------------

def test_resolve_call_finds_a_facade_method():
    fn = resolve_call("potato.helpers.find_files", Taters())
    assert callable(fn)


def test_resolve_call_rejects_an_unknown_facade_path():
    with pytest.raises(AttributeError, match="nope"):
        resolve_call("potato.audio.nope", Taters())


def test_resolve_call_imports_a_dotted_path():
    fn = resolve_call("taters.helpers.find_files.find_files", Taters())
    from taters.helpers.find_files import find_files
    assert fn is find_files


def test_resolve_call_supports_the_colon_form():
    fn = resolve_call("taters.helpers.find_files:find_files", Taters())
    assert callable(fn)


def test_resolve_call_rejects_a_non_callable_target():
    with pytest.raises(TypeError, match="non-callable"):
        resolve_call("taters.helpers.find_files.AUDIO_EXTS", Taters())


# ---------------------------------------------------------------------------
# Preset resolution
# ---------------------------------------------------------------------------

def test_builtin_preset_is_found_by_name():
    assert resolve_preset_path("conversation_video").name == "conversation_video.yaml"


def test_builtin_preset_loads_with_steps():
    preset = load_preset_by_name("conversation_video")
    assert preset["steps"] and preset["meta"]["id"] == "conversation_video"


def test_project_local_preset_is_found(sandbox):
    """A YAML dropped in ./pipelines must be loadable, not just listable."""
    local = sandbox / "pipelines"
    local.mkdir()
    (local / "mine.yaml").write_text(
        "meta:\n  id: my_custom\nsteps:\n  - scope: global\n    call: x.y\n",
        encoding="utf-8",
    )
    assert resolve_preset_path("my_custom").name == "mine.yaml"    # by meta.id
    assert resolve_preset_path("mine").name == "mine.yaml"         # by file stem


def test_preset_name_with_an_extension_still_resolves(sandbox):
    local = sandbox / "pipelines"
    local.mkdir()
    (local / "mine.yml").write_text("steps: []\n", encoding="utf-8")
    assert resolve_preset_path("mine.yml").name == "mine.yml"


def test_unknown_preset_error_lists_what_is_available():
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_preset_path("not_a_real_preset")
    message = str(excinfo.value)
    assert "not_a_real_preset" in message
    assert "conversation_video" in message      # tells you what you could use


def test_preset_meta_defaults_are_filled_in(sandbox):
    p = sandbox / "bare.yaml"
    p.write_text("steps: []\n", encoding="utf-8")
    meta = _load_preset_meta(p)
    assert meta["id"] == "bare" and meta["title"] == "bare" and meta["tags"] == []


# ---------------------------------------------------------------------------
# Step execution and error isolation
# ---------------------------------------------------------------------------

def test_item_step_runs_and_saves_its_artifact(tmp_path):
    (tmp_path / "a.mp4").touch()
    step = {
        "call": "taters.helpers.find_files.find_files",
        "save_as": "found",
        "with": {"root_dir": "{{var:root}}", "file_type": "video"},
    }
    status, artifacts, err = run_item_step_for_one_input(
        step=step, input_path=tmp_path / "a.mp4", potato=Taters(),
        item_artifacts={}, globals_ctx={}, vars_ctx={"root": str(tmp_path)},
    )
    assert status == "ok" and err == {}
    assert [p.name for p in artifacts["found"]] == ["a.mp4"]


def test_item_step_reports_a_failure_instead_of_raising(tmp_path):
    """
    One bad file must not abort a 200-file run, so step failures come back as
    a status rather than an exception.
    """
    step = {
        "call": "taters.helpers.find_files.find_files",
        "with": {"root_dir": "/definitely/not/here"},
    }
    status, artifacts, err = run_item_step_for_one_input(
        step=step, input_path=tmp_path / "a.mp4", potato=Taters(),
        item_artifacts={}, globals_ctx={}, vars_ctx={},
    )
    assert status == "error" and artifacts == {}
    assert "failed" in err["error"]


def test_item_step_missing_upstream_artifact_is_reported_clearly(tmp_path):
    step = {
        "call": "taters.helpers.find_files.find_files",
        "with": {"root_dir": "{{pick:diar.raw_files.csv}}"},
    }
    status, _, err = run_item_step_for_one_input(
        step=step, input_path=tmp_path / "a.mp4", potato=Taters(),
        item_artifacts={}, globals_ctx={}, vars_ctx={},
    )
    assert status == "error"
    assert "missing artifact" in err["error"].lower()


@pytest.mark.parametrize("value", [None, "", []])
def test_require_catches_empty_parameters_after_templating(tmp_path, value):
    step = {
        "call": "taters.helpers.find_files.find_files",
        "require": ["root_dir"],
        "with": {"root_dir": "{{var:missing_thing}}"},
    }
    status, _, err = run_item_step_for_one_input(
        step=step, input_path=tmp_path / "a.mp4", potato=Taters(),
        item_artifacts={}, globals_ctx={}, vars_ctx={"missing_thing": value},
    )
    assert status == "error"
    assert "root_dir" in err["error"]


def test_global_step_exposes_the_manifest_path_as_a_var(tmp_path):
    step = {
        "call": "taters.pipelines.run_pipeline._json_safe",
        "save_as": "echoed",
        "with": {"obj": "{{var:run_manifest}}"},
    }
    status, new_globals, err = run_global_step(
        step=step, potato=Taters(), globals_ctx={}, vars_ctx={},
        manifest_path=tmp_path / "manifest.json",
    )
    assert status == "ok", err
    assert new_globals["echoed"] == str(tmp_path / "manifest.json")


def test_global_step_cannot_see_item_artifacts(tmp_path):
    step = {
        "call": "taters.pipelines.run_pipeline._json_safe",
        "with": {"obj": "{{some_item_artifact}}"},
    }
    status, new_globals, _ = run_global_step(
        step=step, potato=Taters(), globals_ctx={}, vars_ctx={},
        manifest_path=tmp_path / "manifest.json",
    )
    # Unresolved names stay verbatim rather than leaking another item's value.
    assert status == "ok"
