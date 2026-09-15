from __future__ import annotations
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from importlib.resources import files, as_file
from contextlib import ExitStack

from ...helpers.gpu import resolve_device
from ...helpers.proc import run_and_stream
from ...helpers.cliargs import CliSpec

def _resolve_vendored_repo_dir() -> Path:
    # this gives us a Traversable pointing at the directory inside the wheel
    return files("taters.audio.diarizer").joinpath("whisper-diarization")

def _resolve_device(device: Optional[str]) -> str:
    """
    Resolve device selection through the shared rules.

    ``backend="torch"`` because that is what the vendored diarizer runs on --
    NeMo, Demucs and the aligner are all torch. This is the one place where
    asking torch rather than CTranslate2 is correct, and getting it wrong in
    either direction is how a run comes to believe in a GPU it cannot use.

    No probe: the work happens in a subprocess, so there is nothing here to run
    one inference on, and a failure surfaces as that subprocess's exit.
    """
    return resolve_device(device, backend="torch")[0]



@dataclass
class DiarizationOutputFiles:
    work_dir: Path
    raw_files: Dict[str, Path]    # {"srt": ..., "txt": ..., "csv": ...}
    speaker_wavs: Dict[str, Path] # kept for compatibility; always {} here

def _run_repo_script(
    repo_dir: Path,
    audio_path: Path,
    work_dir: Path,
    whisper_model: str,
    language: Optional[str],
    device: Optional[str],
    batch_size: int,
    no_stem: bool,
    suppress_numerals: bool,
    parallel: bool,
    timeout: Optional[int],
    use_custom: bool,
    csv_out: Optional[Path],
    num_speakers: Optional[int],
    verbose: bool = True,
) -> None:

    script = (
        "diarize_custom.py" if (use_custom and (repo_dir / "diarize_custom.py").exists())
        else ("diarize_parallel.py" if parallel else "diarize.py")
    )

    cmd = [sys.executable, str((repo_dir / script).resolve()), "-a", str(audio_path)]

    if whisper_model:
        cmd += ["--whisper-model", whisper_model]
    if language:
        cmd += ["--language", language]
    if device:
        cmd += ["--device", device]
    
    cmd += ["--batch-size", str(batch_size)]   # 0 == non-batched
    
    if no_stem:
        cmd += ["--no-stem"]
    if suppress_numerals:
        cmd += ["--suppress_numerals"]

    cmd += ["--csv-out", str(csv_out)]

    if num_speakers is not None:
        cmd += ["--num-speakers", str(num_speakers)]
    
    

    work_dir.mkdir(parents=True, exist_ok=True)

    # set up the child's environment: on CPU we make sure the GPUs are really
    # hidden; on CUDA we add the pip cuDNN path
    env = os.environ.copy()
    if (device or "").lower() == "cpu":
        env.update({"CUDA_VISIBLE_DEVICES": "", "USE_CUDA": "0", "FORCE_CPU": "1"})
    else:
        try:
            import pathlib

            import nvidia.cudnn
            cudnn_lib = str(pathlib.Path(nvidia.cudnn.__file__).with_name("lib"))
            env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        except Exception:
            pass

    # we stream the child's output live (prefixed, so concurrent items stay
    # readable) and hang onto the tail for the error message if it fails.
    returncode, tail = run_and_stream(
        cmd,
        cwd=work_dir,
        env=env,
        timeout=timeout,
        prefix=f"[diarize:{audio_path.stem}] ",
        stream=verbose,
    )
    if returncode != 0:
        raise RuntimeError(
            f"Diarization subprocess failed (exit {returncode}).\n"
            f"Last output:\n{tail.strip()}"
        )

def _guess_outputs_from_stem(work_dir: Path, stem: str) -> Dict[str, Path]:
    exts = ["srt", "txt", "csv"]
    out: Dict[str, Path] = {}
    for ext in exts:
        p = work_dir / f"{stem}.{ext}"
        if p.exists():
            out[ext] = p
    return out

def _cleanup_temps(work_dir: Path, keep_temp: bool) -> None:
    if keep_temp:
        return
    # diarize.py & demucs write under the CWD (and we run with cwd=work_dir)
    for d in work_dir.glob("temp_outputs*"):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

def run_whisper_diarization_repo(
    audio_path: str | Path,
    out_dir: Optional[str] | Optional[Path] | None = None,
    *,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default
    repo_dir: str | Path | None = None,      # ← now Optional
    whisper_model: str = "base.en",
    language: Optional[str] = None,
    device: Optional[str] = None,            # "cuda" / "cpu"
    batch_size: int = 0,
    no_stem: bool = False,
    suppress_numerals: bool = False,
    parallel: bool = False,
    timeout: Optional[int] = None,
    use_custom: bool = True,
    keep_temp: bool = False,
    num_speakers: Optional[int] = None,
    verbose: bool = True,
) -> DiarizationOutputFiles:
    """
    Run the vendored Whisper diarization scripts and normalize their outputs.

    Parameters
    ----------
    audio_path : str | Path
        Input audio (WAV recommended).
    out_dir : str | Path
        Output directory for transcript artifacts. If it does not exist, it will
        be created.
    repo_dir : str | Path | None
        Optional explicit location of the diarization repo. If None, the
        vendored copy is used.
    whisper_model : str, default "base.en"
        Whisper ASR model to use (e.g., "small", "base", "large-v3").
    language : str | None
        Language hint for Whisper (e.g., "en"); if None, autodetection is used.
    device : {"cpu","cuda"} | None
        Runtime device. If "cpu", environment variables are set to hide GPUs.
    batch_size : int, default 0
        Whisper batch size; 0 disables batching.
    no_stem : bool, default False
        Pass through to demucs/whisper scripts to disable vocal/instrument stems.
    suppress_numerals : bool, default False
        Heuristic to reduce spurious numeral tokens.
    parallel : bool, default False
        Use parallel diarization script if available.
    timeout : int | None
        Subprocess timeout in seconds; None means no timeout.
    use_custom : bool, default True
        Prefer the customized script if present (adds CSV emission and minor cleanup).
    keep_temp : bool, default False
        If False (default), temporary folders created by demucs/whisper are removed.
    num_speakers : int | None
        Force a fixed number of speakers, if the downstream diarizer supports it.
    verbose : bool, default True
        Echo the diarization subprocess's output as it runs, prefixed with
        ``[diarize:<stem>]``. Set False for quiet runs; failures still report
        the tail of the child's output either way.

    overwrite_existing : bool, default=False
        If ``False`` and the output already exists, skip the work and return
        the existing path.
    Returns
    -------
    DiarizationOutputFiles
        Paths to ``.txt``, ``.srt``, and ``.csv`` (if produced) in a per-file working
        directory, plus an (empty) ``speaker_wavs`` mapping for API compatibility.

    Notes
    -----
    - The function copies the input WAV to a per-file work directory before running,
      to ensure relative paths inside the third-party scripts resolve correctly.
    - If `device="cpu"`, CUDA is disabled in the child environment.
    - On success, the local WAV copy is deleted and temporary folders are tidied up.

    See Also
    --------
    taters.audio.split_wav_by_speaker.make_speaker_wavs_from_csv :
        Build per-speaker WAVs from the diarization CSV.
    """


    # first thing's first: pick a device if the user said "auto" (or nothing)
    resolved_device = _resolve_device(device)
    if verbose:
        print(f"Resolved device for whisper extraction: {resolved_device}")

    audio_path = Path(audio_path).resolve()
    # default transcripts folder lives under the current working dir
    out_dir = Path(out_dir).resolve() if out_dir is not None else (Path.cwd() / "transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # each file gets its own working folder so runs don't step on each other
    work_dir = out_dir / f"{audio_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    local_audio = work_dir / audio_path.name
    
    # the CSV lands at <work_dir>/<stem>.csv by default
    csv_path = work_dir / f"{local_audio.stem}.csv"
    if not overwrite_existing and Path(csv_path).is_file():
        print("Diarized transcript output file already exists; returning existing file.")
        raw = _guess_outputs_from_stem(work_dir, local_audio.stem)
        return DiarizationOutputFiles(work_dir=work_dir, raw_files=raw, speaker_wavs={})

    
    # copy the input audio in next to the outputs so the CLI can get by with
    # simple relative paths
    if not local_audio.exists():
        shutil.copy2(audio_path, local_audio)

    # find the vendored repo (unless the user pointed us at their own)
    with ExitStack() as stack:
        if repo_dir is None:
            repo_trav = _resolve_vendored_repo_dir()
            repo_dir_path = stack.enter_context(as_file(repo_trav))  # a real FS path
        else:
            repo_dir_path = Path(repo_dir).resolve()

        # make sure the script we want is actually in there
        script_name = ("diarize_custom.py" if (use_custom and (repo_dir_path / "diarize_custom.py").exists())
                       else ("diarize_parallel.py" if parallel else "diarize.py"))
        script_path = (repo_dir_path / script_name)
        if not script_path.exists():
            raise FileNotFoundError(f"Expected script not found: {script_path}")

        # now we run the repo script (cwd = work_dir so temp_outputs land there)
        _run_repo_script(
            repo_dir=repo_dir_path,
            audio_path=local_audio,
            work_dir=work_dir,
            whisper_model=whisper_model,
            language=language,
            device=resolved_device,
            batch_size=batch_size,
            no_stem=no_stem,
            suppress_numerals=suppress_numerals,
            parallel=parallel,
            timeout=timeout,
            use_custom=use_custom,
            csv_out=csv_path,
            num_speakers=num_speakers,
            verbose=verbose,
        )

    # tidy up the temp dirs
    _cleanup_temps(work_dir, keep_temp)

    # round up the outputs (.txt/.srt/.csv)
    raw = _guess_outputs_from_stem(work_dir, local_audio.stem)

    # lastly, get rid of our copy of the WAV now that we're done with it
    try:
        if local_audio.exists():
            local_audio.unlink()
    except Exception:
        pass

    return DiarizationOutputFiles(work_dir=work_dir, raw_files=raw, speaker_wavs={})

# ---------------- CLI: allow `python -m taters.audio.diarizer.whisper_diar_wrapper` -----


# ---------------------------------------------------------------------------
# Command line -- derived from the function(s) above; see helpers.cliargs.CliSpec.
# The aliases and legacy flags are the spellings the hand-written parser used,
# kept so every documented invocation still works.
# ---------------------------------------------------------------------------

CLI = CliSpec(
    run_whisper_diarization_repo,
    description='Speaker diarization through the vendored whisper-diarization scripts.',
    aliases={},
    legacy={
        '--no-custom': ['--use-custom', 'false'],
        '--quiet': ['--verbose', 'false'],
    },
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
