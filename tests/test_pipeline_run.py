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


# --- actually running one ---------------------------------------------------

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
    # alice shows up in both input files, so her turns should get pooled
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
    # artifacts get stored JSON-safe (Paths become strings)
    assert isinstance(manifest["globals"]["merged"], str)


def test_var_override_reaches_the_step(project, repo_root):
    """`--var` should beat the preset's own vars block."""
    import os

    out = project / "out" / "ready.csv"
    run_pipeline("--preset", "smoke", cwd=project, repo_root=repo_root)
    # we date it into the past: a gathered table with its record gets kept under
    # overwrite_existing=false (its date stays) and remade under =true (its
    # date moves). a hand-written sentinel wouldn't do here, since a gathered
    # file with no record gets rebuilt on sight
    old = out.stat().st_mtime - 3600
    os.utime(out, (old, old))

    run_pipeline(
        "--preset", "smoke", "--var", "overwrite_existing=false",
        cwd=project, repo_root=repo_root,
    )
    assert out.stat().st_mtime == old

    run_pipeline(
        "--preset", "smoke", "--var", "overwrite_existing=true",
        cwd=project, repo_root=repo_root,
    )
    assert out.stat().st_mtime > old


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
    # the exit code has to reflect the failure, otherwise a scheduled job
    # quietly reports success while producing nothing
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
    # find_files raises FileNotFoundError for a root that doesn't exist, so we
    # point at "<input>_dir", which only exists for the good file
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
    # non-zero because one item failed, but the good one still ran, and that's
    # really what we're testing here
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
    # LIWC2007 .dic format: category header between % lines, then word<TAB>ids
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
    # check the manifest before the exit code, since it tells us *what* broke
    assert manifest["errors"] == [], manifest["errors"]
    assert manifest["items"] and manifest["items"][0]["status"] == "ok", manifest["items"]
    assert res.returncode == 0, res.stdout[-3000:]

    # every step of the preset should've left its artifact behind
    assert list((sandbox / "transcripts").rglob("*.csv")), "no transcript produced"
    # under `transcripts/`, which is where the preset's gather step writes it
    # (`out_csv: "{{var:transcripts_dir}}/all_transcripts.csv"`). this used to
    # look in the sandbox root and had been failing unnoticed since the preset
    # changed. a `slow` test nobody had run in a while, which is the hazard of
    # the opt-in layer, and the reason to run `pytest -m slow` before a release
    # rather than only when something feels wrong
    assert (sandbox / "transcripts" / "all_transcripts.csv").is_file()

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

    # and the feature tables should describe the speakers we actually found
    with (sandbox / "features" / "dictionary.csv").open(newline="", encoding="utf-8-sig") as f:
        dict_rows = list(csv.DictReader(f))
    assert dict_rows, "dictionary features are empty"
    columns = set(dict_rows[0])
    assert {"source", "speaker"} <= columns, "id columns were not carried through"
    # global counts are unprefixed, and each dictionary's categories get
    # namespaced by its filename, so our mini.dic gives us mini__posemo / mini__negemo
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


# --- --quiet ----------------------------------------------------------------
#
# the runner had no way to turn its output down. `run_preset` grew a `verbose`
# argument for the TUI's benefit (a live display can't share the screen with a
# step printing into it), but the command line got left with the one setting
# it always had. on a four-file transcription that's 638 lines, one per
# decoded segment

def test_quiet_suppresses_the_step_by_step_chatter(project, repo_root):
    loud = run_pipeline("--preset", "smoke", cwd=project, repo_root=repo_root)
    quiet = run_pipeline("--preset", "smoke", "--quiet", cwd=project, repo_root=repo_root)

    assert loud.returncode == 0 and quiet.returncode == 0, quiet.stderr
    assert "[pipeline] Step 1/2" in loud.stdout
    assert "Step 1/2" not in quiet.stdout


def test_quiet_reaches_the_steps_own_printing_too(project, repo_root):
    """
    Not just the runner's lines. The chatter that actually drowns a terminal
    comes from inside the steps, which each default to printing and never used
    to be told otherwise.
    """
    # `smoke` overwrites by default, so we turn that off so a second run
    # short-circuits and each step says so. that line comes from inside the step,
    # not the runner, which is exactly the kind --quiet had no way to reach
    resume = ("--preset", "smoke", "--var", "overwrite_existing=false")
    run_pipeline(*resume, cwd=project, repo_root=repo_root)
    loud = run_pipeline(*resume, cwd=project, repo_root=repo_root)
    quiet = run_pipeline(*resume, "--quiet", cwd=project, repo_root=repo_root)

    assert "already exists" in loud.stdout
    assert "already exists" not in quiet.stdout


def test_quiet_still_says_whether_it_worked(project, repo_root):
    """
    The outcome is not chatter. A quiet run that says nothing at all about
    whether it succeeded is a worse tool than a loud one.
    """
    res = run_pipeline("--preset", "smoke", "--quiet", cwd=project, repo_root=repo_root)
    assert res.returncode == 0
    assert "[pipeline] Items:" in res.stdout or not json.loads(
        (project / "run_manifest.json").read_text(encoding="utf-8"))["items"]


def test_quiet_still_names_the_files_that_failed(sandbox, repo_root):
    inputs = sandbox / "media"
    inputs.mkdir()
    (inputs / "good.wav").touch()
    (inputs / "bad.wav").touch()
    (inputs / "good.wav_dir").mkdir()

    pipelines = sandbox / "pipelines"
    pipelines.mkdir()
    (pipelines / "mixed.yaml").write_text(
        "meta:\n  id: mixed3\n"
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
        "--root_dir", "media", "--file_type", "audio", "--preset", "mixed3", "--quiet",
        cwd=sandbox, repo_root=repo_root,
    )

    assert res.returncode != 0, "a quiet run must still fail loudly"
    assert "1 ok, 1 failed" in res.stdout
    assert "bad.wav" in res.stdout


def test_a_plain_run_prints_what_it_always_printed(project, repo_root):
    """
    Adding `--quiet` must not change the default. This covers the runner's own
    lines; that no *step* starts printing differently is the job of
    `test_a_loud_run_leaves_a_quiet_step_alone`, which watches what the runner
    actually hands a step function.
    """
    res = run_pipeline("--preset", "smoke", cwd=project, repo_root=repo_root)
    assert "[pipeline] Step 1/2: potato.helpers.feature_gather" in res.stdout
    assert "[pipeline] Step 2/2: potato.helpers.csv_to_analysis_ready_csv" in res.stdout
    assert "[pipeline] Manifest written to:" in res.stdout



def test_a_relative_manifest_path_is_anchored_where_the_caller_stood(tmp_path, monkeypatch):
    """
    From the second review (issue 25): `run_preset` chdir'd into work_dir and
    only then resolved a relative out_manifest -- so `taters --dir data` wrote
    data/mypipe/data/mypipe/run_manifest.json while every screen pointed at the
    real path, which did not exist.
    """
    import json

    from taters.pipelines.run_pipeline import run_preset

    monkeypatch.chdir(tmp_path)
    preset = {
        "meta": {"id": "anchor", "title": "Anchor"},
        "vars": {},
        # one do-nothing global step, since an empty steps list gets refused outright
        "steps": [{"scope": "global", "call": "potato.helpers.find_files",
                   "save_as": "listing",
                   "with": {"root_dir": ".", "extensions": [".none"]}}],
    }
    rel = Path("data") / "anchor" / "run_manifest.json"

    run_preset(preset, out_manifest=rel, verbose=False,
               work_dir=Path("data") / "anchor")

    assert (tmp_path / rel).exists(), "the manifest is not where the caller said"
    assert not (tmp_path / "data" / "anchor" / "data").exists(), (
        "the relative path was re-anchored under work_dir"
    )
    assert json.loads((tmp_path / rel).read_text(encoding="utf-8"))["preset"] == "anchor"


# ---------------------------------------------------------------------------
# one parallelism dial
# ---------------------------------------------------------------------------

def test_the_workers_dial_resolves_in_priority_order(monkeypatch):
    """--workers beats the preset's `workers` variable beats automatic; 0 and
    None both mean "not decided here" at every layer. Every door goes through
    the one policy: automatic is three-quarters of the cores, and no answer
    exceeds the machine's core count."""
    from taters.helpers import parallel_map as pm
    from taters.pipelines.run_pipeline import _run_wide_workers

    monkeypatch.setattr(pm.os, "cpu_count", lambda: 8)
    assert _run_wide_workers(6, {"workers": 2}) == 6
    assert _run_wide_workers(None, {"workers": 2}) == 2
    assert _run_wide_workers(None, {"workers": 0}) == 6, "auto = 8 - 8//4"
    assert _run_wide_workers(0, {"workers": 3}) == 3
    assert _run_wide_workers(None, {}) == 6
    assert _run_wide_workers(None, {"workers": "not a number"}) == 6
    assert _run_wide_workers(99, {}) == 8, "the flag is clamped to the machine"
    assert _run_wide_workers(None, {"workers": 99}) == 8, "the var too"


def test_the_resolved_dial_feeds_the_var_templates_too(tmp_path):
    """
    The unification's whole point: `--workers N` and `vars: workers:` steer the
    SAME number, and the text steps' `{{var:workers}}` templates see the
    resolved value -- whichever door it came in by.
    """

    from taters.pipelines.run_pipeline import run_preset

    seen = {}

    def probe(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "out.csv")

    import taters.pipelines.run_pipeline as rp
    preset = {
        "meta": {"id": "t"},
        "vars": {"workers": 2},
        "steps": [{"scope": "global", "call": "helpers.find_files",
                   "save_as": "x", "with": {"workers": "{{var:workers}}"}}],
    }
    # resolve_call would want a real target, so we monkeypatch at the seam instead
    orig = rp.resolve_call
    rp.resolve_call = lambda name, potato: probe
    try:
        run_preset(preset, out_manifest=tmp_path / "m.json")
        assert seen["workers"] == 2, "vars.workers must reach the step"
        seen.clear()
        # 1, not 5: the dial is capped at the core count, so on a two-core
        # CI runner asking for 5 resolves to 4 and this read as a bug in the
        # override. one can never be capped, and still differs from the 2 in
        # `vars`, which is the whole thing being tested.
        run_preset(preset, workers=1, out_manifest=tmp_path / "m.json")
        assert seen["workers"] == 1, "--workers must override vars.workers"
    finally:
        rp.resolve_call = orig


def test_a_workers_signature_is_the_whole_authoring_contract(tmp_path):
    """
    Drop-in module story: a GLOBAL step whose function declares `workers` gets
    the run's resolved dial injected -- no recipe wiring, no `with:` line. An
    explicit `with:` value still wins, and a function without the parameter
    is called exactly as before.
    """
    import taters.pipelines.run_pipeline as rp

    seen = {}

    def takes_workers(*, workers=None, **kw):
        seen["workers"] = workers
        return "x"

    def takes_none():
        # a truly empty signature: **kwargs would count as "accepts workers",
        # which is the facade case, and that one's meant to be injectable
        seen["none"] = True
        return "y"

    preset = {
        "meta": {"id": "t"},
        "vars": {"workers": 3},
        "steps": [
            {"scope": "global", "call": "a", "save_as": "a", "with": {}},
            {"scope": "global", "call": "b", "save_as": "b",
             "with": {"workers": 2}},
            {"scope": "global", "call": "c", "save_as": "c", "with": {}},
        ],
    }
    funcs = {"a": takes_workers, "b": takes_workers, "c": takes_none}
    orig = rp.resolve_call
    rp.resolve_call = lambda name, potato: funcs[name]
    try:
        seen.clear()
        rp.run_preset(preset, out_manifest=tmp_path / "m.json")
    finally:
        rp.resolve_call = orig

    # step a: injected from the dial; step b: the preset's explicit 2 wins
    assert seen["none"] is True
    assert seen["workers"] == 2          # last call to takes_workers was b
    # rerun watching step a alone
    rp.resolve_call = lambda name, potato: funcs["a"]
    try:
        seen.clear()
        rp.run_preset({"meta": {"id": "t"}, "vars": {"workers": 3},
                       "steps": [{"scope": "global", "call": "a",
                                  "save_as": "a", "with": {}}]},
                      out_manifest=tmp_path / "m.json")
    finally:
        rp.resolve_call = orig
    assert seen["workers"] == 3


def test_item_steps_get_workers_one_because_fanout_is_their_parallelism(tmp_path):
    """An internal pool per fanned-out call would multiply into cores-squared;
    an ITEM function that declares `workers` is told 1."""
    import taters.pipelines.run_pipeline as rp

    seen = []

    def item_fn(*, input=None, workers=None, **kw):
        seen.append(workers)
        return {"ok": True}

    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "in" / "b.txt").write_text("x", encoding="utf-8")

    preset = {
        "meta": {"id": "t"},
        "vars": {"workers": 8},
        "steps": [{"scope": "item", "call": "f", "save_as": "f",
                   "with": {"input": "{{item.path}}"}}],
    }
    orig = rp.resolve_call
    rp.resolve_call = lambda name, potato: item_fn
    try:
        rp.run_preset(preset, root_dir=tmp_path / "in", file_type="any",
                      out_manifest=tmp_path / "m.json")
    finally:
        rp.resolve_call = orig

    assert seen and all(w == 1 for w in seen)
