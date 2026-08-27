"""End-to-end runs of the pipeline runner.

The fast tests here drive the real CLI (`python -m taters.pipelines.run_pipeline`)
over synthetic CSVs, so they exercise argument parsing, preset discovery, step
sequencing, the manifest, and error isolation — everything except the models.
The slow test at the bottom runs the real `conversation_video` preset on real
media.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas", reason="the gather steps need pandas")


def run_pipeline(*args: str, cwd: Path, repo_root: Path) -> subprocess.CompletedProcess:
    """Invoke the runner exactly the way a user would."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "taters.pipelines.run_pipeline", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True,
    )


@pytest.fixture
def project(sandbox) -> Path:
    """A working directory with input CSVs and a project-local preset."""
    inputs = sandbox / "inputs"
    inputs.mkdir()
    (inputs / "a.csv").write_text(
        "speaker,text\nalice,first thing alice said\nbob,something from bob\n",
        encoding="utf-8",
    )
    (inputs / "b.csv").write_text(
        "speaker,text\nalice,more from alice later on\n", encoding="utf-8",
    )

    pipelines = sandbox / "pipelines"
    pipelines.mkdir()
    (pipelines / "smoke.yaml").write_text(
        "meta:\n"
        "  id: smoke\n"
        "  title: Gather and group\n"
        "vars:\n"
        "  overwrite_existing: true\n"
        "steps:\n"
        "  - scope: global\n"
        "    call: potato.helpers.feature_gather\n"
        "    save_as: merged\n"
        "    with:\n"
        "      root_dir: inputs\n"
        "      pattern: '*.csv'\n"
        "      out_csv: out/merged.csv\n"
        "      overwrite_existing: '{{var:overwrite_existing}}'\n"
        "  - scope: global\n"
        "    call: potato.helpers.csv_to_analysis_ready_csv\n"
        "    save_as: ready\n"
        "    with:\n"
        "      csv_path: '{{merged}}'\n"
        "      text_cols: ['text']\n"
        "      group_by: ['speaker']\n"
        "      out_csv: out/ready.csv\n"
        "      overwrite_existing: '{{var:overwrite_existing}}'\n",
        encoding="utf-8",
    )
    return sandbox


# --- discovery helpers ------------------------------------------------------

def test_list_presets_shows_builtin_and_local(project, repo_root):
    res = run_pipeline("--list-presets", cwd=project, repo_root=repo_root)
    assert res.returncode == 0, res.stderr
    assert "conversation_video" in res.stdout
    assert "smoke" in res.stdout


def test_describe_preset_prints_metadata(project, repo_root):
    res = run_pipeline("--describe-preset", "smoke", cwd=project, repo_root=repo_root)
    assert res.returncode == 0, res.stderr
    assert "Gather and group" in res.stdout


def test_missing_preset_argument_is_an_error(project, repo_root):
    res = run_pipeline(cwd=project, repo_root=repo_root)
    assert res.returncode != 0
    assert "--preset" in res.stderr


def test_unknown_preset_names_the_available_ones(project, repo_root):
    res = run_pipeline("--preset", "nonsense", cwd=project, repo_root=repo_root)
    assert res.returncode != 0
    assert "conversation_video" in (res.stderr + res.stdout)


# --- a real run -------------------------------------------------------------

def test_global_only_pipeline_runs_and_writes_outputs(project, repo_root):
    res = run_pipeline(
        "--preset", "smoke", "--out-manifest", "out/manifest.json",
        cwd=project, repo_root=repo_root,
    )
    assert res.returncode == 0, res.stderr

    merged = project / "out" / "merged.csv"
    ready = project / "out" / "ready.csv"
    assert merged.is_file() and ready.is_file()

    with ready.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert {r["speaker"] for r in rows} == {"alice", "bob"}
    # alice appears in both input files; her turns should be pooled.
    alice = next(r for r in rows if r["speaker"] == "alice")
    assert alice["group_count"] == "2"


def test_global_only_pipeline_skips_input_discovery(project, repo_root):
    res = run_pipeline("--preset", "smoke", cwd=project, repo_root=repo_root)
    assert "only GLOBAL steps" in res.stdout


def test_manifest_records_steps_and_artifacts(project, repo_root):
    run_pipeline(
        "--preset", "smoke", "--out-manifest", "out/manifest.json",
        cwd=project, repo_root=repo_root,
    )
    manifest = json.loads((project / "out" / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["preset"] == "smoke"
    assert manifest["errors"] == []
    assert set(manifest["globals"]) == {"merged", "ready"}
    # Artifacts are stored JSON-safe (Paths become strings).
    assert isinstance(manifest["globals"]["merged"], str)


def test_var_override_reaches_the_step(project, repo_root):
    """`--var` should beat the preset's own vars block."""
    out = project / "out" / "ready.csv"
    run_pipeline("--preset", "smoke", cwd=project, repo_root=repo_root)
    out.write_text("sentinel\n", encoding="utf-8")

    run_pipeline(
        "--preset", "smoke", "--var", "overwrite_existing=false",
        cwd=project, repo_root=repo_root,
    )
    assert out.read_text(encoding="utf-8") == "sentinel\n"

    run_pipeline(
        "--preset", "smoke", "--var", "overwrite_existing=true",
        cwd=project, repo_root=repo_root,
    )
    assert out.read_text(encoding="utf-8") != "sentinel\n"


def test_preset_file_path_also_works(project, repo_root):
    res = run_pipeline(
        "--preset-file", "pipelines/smoke.yaml", cwd=project, repo_root=repo_root
    )
    assert res.returncode == 0, res.stderr
    assert (project / "out" / "ready.csv").is_file()


def test_a_failing_global_step_stops_the_run_and_is_recorded(project, repo_root):
    (project / "pipelines" / "broken.yaml").write_text(
        "meta:\n  id: broken\n"
        "steps:\n"
        "  - scope: global\n"
        "    call: potato.helpers.feature_gather\n"
        "    with:\n"
        "      root_dir: no_such_folder\n"
        "      out_csv: out/nope.csv\n"
        "  - scope: global\n"
        "    call: potato.helpers.feature_gather\n"
        "    save_as: never_runs\n"
        "    with:\n"
        "      root_dir: inputs\n"
        "      out_csv: out/after.csv\n",
        encoding="utf-8",
    )
    res = run_pipeline(
        "--preset", "broken", "--out-manifest", "out/manifest.json",
        cwd=project, repo_root=repo_root,
    )
    assert "GLOBAL step failed" in res.stdout
    # The exit code has to reflect the failure, or a scheduled job silently
    # reports success while producing nothing.
    assert res.returncode != 0

    manifest = json.loads((project / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["errors"], "the failure should be recorded in the manifest"
    assert not (project / "out" / "after.csv").exists(), "run should have stopped"


def test_a_clean_run_exits_zero(project, repo_root):
    res = run_pipeline("--preset", "smoke", cwd=project, repo_root=repo_root)
    assert res.returncode == 0, res.stderr


def test_a_failed_item_makes_the_whole_run_exit_nonzero(sandbox, repo_root):
    """
    A batch where some files failed is not a successful batch. The run still
    completes every other item — only the exit code changes.
    """
    inputs = sandbox / "media"
    inputs.mkdir()
    (inputs / "good.wav").touch()
    (inputs / "bad.wav").touch()
    (inputs / "good.wav_dir").mkdir()

    pipelines = sandbox / "pipelines"
    pipelines.mkdir()
    (pipelines / "mixed.yaml").write_text(
        "meta:\n  id: mixed2\n"
        "steps:\n"
        "  - scope: item\n"
        "    call: taters.helpers.find_files.find_files\n"
        "    save_as: seen\n"
        "    with:\n"
        "      root_dir: '{{input}}_dir'\n"
        "      file_type: any\n",
        encoding="utf-8",
    )
    res = run_pipeline(
        "--root_dir", "media", "--file_type", "audio", "--preset", "mixed2",
        "--out-manifest", "out/manifest.json",
        cwd=sandbox, repo_root=repo_root,
    )
    assert res.returncode != 0
    assert "1 ok, 1 failed" in res.stdout

    manifest = json.loads((sandbox / "out" / "manifest.json").read_text(encoding="utf-8"))
    statuses = {Path(i["input"]).name: i["status"] for i in manifest["items"]}
    assert statuses == {"good.wav": "ok", "bad.wav": "error"}


# --- item steps -------------------------------------------------------------

def test_item_steps_fan_out_over_discovered_inputs(sandbox, repo_root):
    inputs = sandbox / "media"
    inputs.mkdir()
    for name in ["one.wav", "two.wav", "three.wav"]:
        (inputs / name).touch()

    pipelines = sandbox / "pipelines"
    pipelines.mkdir()
    (pipelines / "peritem.yaml").write_text(
        "meta:\n  id: peritem\n"
        "steps:\n"
        "  - scope: item\n"
        "    call: taters.helpers.find_files.find_files\n"
        "    save_as: seen\n"
        "    with:\n"
        "      root_dir: '{{cwd}}/media'\n"
        "      file_type: audio\n",
        encoding="utf-8",
    )
    res = run_pipeline(
        "--root_dir", "media", "--file_type", "audio", "--preset", "peritem",
        "--out-manifest", "out/manifest.json", "--workers", "2",
        cwd=sandbox, repo_root=repo_root,
    )
    assert res.returncode == 0, res.stderr
    assert "Found 3 'audio' input(s)" in res.stdout

    manifest = json.loads((sandbox / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 3
    assert all(item["status"] == "ok" for item in manifest["items"])


def test_item_step_requires_root_dir(project, repo_root):
    (project / "pipelines" / "needsitems.yaml").write_text(
        "meta:\n  id: needsitems\n"
        "steps:\n"
        "  - scope: item\n"
        "    call: taters.helpers.find_files.find_files\n"
        "    with: {root_dir: '{{input}}'}\n",
        encoding="utf-8",
    )
    res = run_pipeline("--preset", "needsitems", cwd=project, repo_root=repo_root)
    assert res.returncode != 0
    assert "--root_dir is required" in (res.stderr + res.stdout)


def test_one_bad_item_does_not_stop_the_others(sandbox, repo_root):
    """
    Error isolation is the whole reason item failures are returned rather than
    raised: a corrupt file in the middle of a dataset must not abort the batch.
    """
    inputs = sandbox / "media"
    inputs.mkdir()
    (inputs / "good.wav").touch()
    (inputs / "bad.wav").touch()

    pipelines = sandbox / "pipelines"
    pipelines.mkdir()
    # find_files raises FileNotFoundError for a root that does not exist; we
    # point at "<input>_dir", which exists only for the good file.
    (inputs / "good.wav_dir").mkdir()
    (pipelines / "mixed.yaml").write_text(
        "meta:\n  id: mixed\n"
        "steps:\n"
        "  - scope: item\n"
        "    call: taters.helpers.find_files.find_files\n"
        "    save_as: seen\n"
        "    with:\n"
        "      root_dir: '{{input}}_dir'\n"
        "      file_type: any\n",
        encoding="utf-8",
    )
    res = run_pipeline(
        "--root_dir", "media", "--file_type", "audio", "--preset", "mixed",
        "--out-manifest", "out/manifest.json",
        cwd=sandbox, repo_root=repo_root,
    )
    # Non-zero because one item failed — but the good one still ran, which is
    # the property being tested here.
    assert res.returncode != 0

    manifest = json.loads((sandbox / "out" / "manifest.json").read_text(encoding="utf-8"))
    statuses = {Path(i["input"]).name: i["status"] for i in manifest["items"]}
    assert statuses == {"good.wav": "ok", "bad.wav": "error"}
    bad = next(i for i in manifest["items"] if i["input"].endswith("bad.wav"))
    assert bad["errors"], "the failure reason should be recorded"


# --- the real thing ---------------------------------------------------------

@pytest.fixture
def resource_dirs(sandbox) -> dict:
    """
    Minimal LIWC-style dictionary and archetype files.

    The preset's defaults point at `dictionaries/liwc` and
    `dictionaries/archetypes`, which are a user's own resources and are not in
    the repo. Without something at those paths the dictionary step fails and
    takes the second half of the preset down with it — which is precisely how
    this test used to pass while verifying almost nothing.
    """
    liwc = sandbox / "resources" / "liwc"
    liwc.mkdir(parents=True)
    # LIWC2007 .dic format: category header between % lines, then word<TAB>ids.
    (liwc / "mini.dic").write_text(
        "%\n1\tposemo\n2\tnegemo\n%\n"
        "happy\t1\ngood\t1\nlove*\t1\nsad\t2\nbad\t2\nterrible\t2\n",
        encoding="utf-8",
    )

    archetypes = sandbox / "resources" / "archetypes"
    archetypes.mkdir(parents=True)
    (archetypes / "mini.csv").write_text(
        "archetype,sentence\n"
        "Warmth,I really care about the people around me.\n"
        "Warmth,We look after each other here.\n"
        "Competence,I know exactly how to get this done.\n"
        "Competence,The work was finished carefully and correctly.\n",
        encoding="utf-8",
    )
    return {"dictionaries": liwc, "archetypes": archetypes}


@pytest.mark.slow
@pytest.mark.needs_ffmpeg
@pytest.mark.needs_media
def test_conversation_video_preset_end_to_end(
    real_media_with_both_streams, resource_dirs, sandbox, repo_root
):
    """
    Run the shipped preset over a single short clip, start to finish.

    This is the test that proves the documented workflow works: video in,
    transcripts and eight feature tables out. It needs the full extras and
    takes minutes even on a GPU, which is why it is opt-in.
    """
    pytest.importorskip("faster_whisper", reason="needs the whisper stack")
    pytest.importorskip("nemo", reason="install with: pip install 'taters[diarization]'")
    pytest.importorskip("parselmouth", reason="needs praat-parselmouth")
    pytest.importorskip("textstat", reason="needs the readability extra")

    media = sandbox / "media"
    media.mkdir()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-t", "30", "-i", str(real_media_with_both_streams),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(media / "clip.mp4")],
        check=True, capture_output=True,
    )

    res = run_pipeline(
        "--root_dir", "media", "--file_type", "video",
        "--preset", "conversation_video",
        "--workers", "1",
        "--var", "whisper_model=tiny.en",
        "--var", "overwrite_existing=true",
        "--var", f"dictionaries_path={resource_dirs['dictionaries']}",
        "--var", f"archetypes_dict_path={resource_dirs['archetypes']}",
        "--out-manifest", "run_manifest.json",
        cwd=sandbox, repo_root=repo_root,
    )

    manifest = json.loads((sandbox / "run_manifest.json").read_text(encoding="utf-8"))
    # Check the manifest before the exit code: it says *what* broke.
    assert manifest["errors"] == [], manifest["errors"]
    assert manifest["items"] and manifest["items"][0]["status"] == "ok", manifest["items"]
    assert res.returncode == 0, res.stdout[-3000:]

    # Every step of the preset should have left its artifact behind.
    assert list((sandbox / "transcripts").rglob("*.csv")), "no transcript produced"
    assert (sandbox / "all_transcripts.csv").is_file()

    for relative in [
        "features/acoustics_summary.csv",
        "features/whisper-embeddings_aggregated.csv",
        "features/dictionary.csv",
        "features/readability.csv",
        "features/lexical-richness.csv",
        "features/archetypes.csv",
        "features/sentence-embeddings_aggregated.csv",
    ]:
        path = sandbox / relative
        assert path.is_file(), f"{relative} was not produced"
        assert path.stat().st_size > 0, f"{relative} is empty"

    # And the feature tables should describe the speakers we actually found.
    with (sandbox / "features" / "dictionary.csv").open(newline="", encoding="utf-8-sig") as f:
        dict_rows = list(csv.DictReader(f))
    assert dict_rows, "dictionary features are empty"
    columns = set(dict_rows[0])
    assert {"source", "speaker"} <= columns, "id columns were not carried through"
    # Global counts are unprefixed; each dictionary's categories are namespaced
    # by its filename, so our mini.dic contributes mini__posemo / mini__negemo.
    assert "WC" in columns
    assert {"mini__posemo", "mini__negemo"} <= columns, sorted(columns)
    assert all(float(r["WC"]) > 0 for r in dict_rows), "speakers with no words"


@pytest.mark.slow
@pytest.mark.needs_ffmpeg
@pytest.mark.needs_media
def test_failed_preset_run_reports_a_nonzero_exit_code(
    real_media_with_both_streams, sandbox, repo_root
):
    """
    The same preset *without* the dictionary resources must fail loudly.

    Regression: the runner used to print the failure, write the manifest and
    then exit 0, so any script or scheduler wrapping it saw a clean run.
    """
    pytest.importorskip("nemo", reason="install with: pip install 'taters[diarization]'")

    media = sandbox / "media"
    media.mkdir()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-t", "10", "-i", str(real_media_with_both_streams),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(media / "clip.mp4")],
        check=True, capture_output=True,
    )

    res = run_pipeline(
        "--root_dir", "media", "--file_type", "video",
        "--preset", "conversation_video", "--workers", "1",
        "--var", "whisper_model=tiny.en",
        "--var", "dictionaries_path=does/not/exist",
        "--out-manifest", "run_manifest.json",
        cwd=sandbox, repo_root=repo_root,
    )
    assert res.returncode != 0, "a failed pipeline must not report success"
    assert "GLOBAL ERROR" in res.stdout

    manifest = json.loads((sandbox / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["errors"], "the failure should be recorded in the manifest"
