from __future__ import annotations
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from importlib.resources import files, as_file
from contextlib import ExitStack

def _resolve_vendored_repo_dir() -> Path:
    # This returns a Traversable pointing at the directory inside your wheel.
    return files("taters.audio.diarizer").joinpath("whisper-diarization")

def _resolve_device(device: Optional[str]) -> str:
    """
    Resolve device selection:
      - "auto" or None -> "cuda" if torch.cuda.is_available() else "cpu"
      - "cuda"/"cpu"   -> returned as-is
    We import torch only here to avoid importing it unnecessarily elsewhere.
    """
    if device is None or str(device).lower() in {"", "auto"}:
        try:
            import torch  # noqa
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device).lower()



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

    # Prepare environment (ensure CPU really hides GPUs; on CUDA add pip cuDNN path)
    env = os.environ.copy()
    if (device or "").lower() == "cpu":
        env.update({"CUDA_VISIBLE_DEVICES": "", "USE_CUDA": "0", "FORCE_CPU": "1"})
    else:
        try:
            import nvidia.cudnn, pathlib
            cudnn_lib = str(pathlib.Path(nvidia.cudnn.__file__).with_name("lib"))
            env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        except Exception:
            pass

    subprocess.run(cmd, cwd=work_dir, check=True, timeout=timeout, env=env, stdin=subprocess.DEVNULL)

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
    # diarize.py & demucs write under CWD (we run with cwd=work_dir)
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
    whisper_model : str, default "medium.en"
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


    # Decide device if user passed "auto" (or None)
    resolved_device = _resolve_device(device)
    print(f"Resolved device for whisper extraction: {resolved_device}")

    audio_path = Path(audio_path).resolve()
    # default transcripts folder next to current working dir
    out_dir = Path(out_dir).resolve() if out_dir is not None else (Path.cwd() / "transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Isolated working folder
    work_dir = out_dir / f"{audio_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    local_audio = work_dir / audio_path.name
    
    # CSV default: <work_dir>/<stem>.csv
    csv_path = work_dir / f"{local_audio.stem}.csv"
    if not overwrite_existing and Path(csv_path).is_file():
        print("Diarized transcript output file already exists; returning existing file.")
        raw = _guess_outputs_from_stem(work_dir, local_audio.stem)
        return DiarizationOutputFiles(work_dir=work_dir, raw_files=raw, speaker_wavs={})

    
    # Copy input audio next to outputs so the CLI can use simple relative paths
    if not local_audio.exists():
        shutil.copy2(audio_path, local_audio)

    # Resolve path to vendored repo (or use user-supplied path)
    with ExitStack() as stack:
        if repo_dir is None:
            repo_trav = _resolve_vendored_repo_dir()
            repo_dir_path = stack.enter_context(as_file(repo_trav))  # real FS path
        else:
            repo_dir_path = Path(repo_dir).resolve()

        # Validate the script exists inside the repo
        script_name = ("diarize_custom.py" if (use_custom and (repo_dir_path / "diarize_custom.py").exists())
                       else ("diarize_parallel.py" if parallel else "diarize.py"))
        script_path = (repo_dir_path / script_name)
        if not script_path.exists():
            raise FileNotFoundError(f"Expected script not found: {script_path}")

        # Run the repo script (cwd = work_dir so temp_outputs land there)
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
        )

    # Tidy temp dirs
    _cleanup_temps(work_dir, keep_temp)

    # Collect outputs (.txt/.srt/.csv)
    raw = _guess_outputs_from_stem(work_dir, local_audio.stem)

    # Remove the copied WAV now that we're done
    try:
        if local_audio.exists():
            local_audio.unlink()
    except Exception:
        pass

    return DiarizationOutputFiles(work_dir=work_dir, raw_files=raw, speaker_wavs={})

# ---------------- CLI: allow `python -m taters.audio.diarizer.whisper_diar_wrapper` -----

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Taters wrapper for MahmoudAshraf97/whisper-diarization"
    )
    # required I/O
    p.add_argument("--audio_path", required=True, help="Path to input audio (e.g., WAV)")
    p.add_argument("--out_dir", default=None, help="Directory to write outputs (work_dir/<stem>/...) "
                                                   "Default: ./transcripts under current working dir")

    p.add_argument("--overwrite_existing", type=bool, default=False,
                    help="Do you want to overwrite the output file if it already exists?")

    # optional repo dir (omit to use vendored copy)
    p.add_argument("--repo_dir", default=None, help="Path to whisper-diarization repo; omit to use vendored")

    # diarization controls
    p.add_argument("--whisper_model", default="medium.en", help="Faster-Whisper model name (e.g., base, medium.en)")
    p.add_argument("--language", default=None, help="Force language (e.g., en). Leave empty to auto-detect")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help='Device: "auto" (default) picks CUDA if available else CPU')
    p.add_argument("--batch_size", type=int, default=0, help="Whisper batch size (0 = non-batched)")

    p.add_argument("--no_stem", action="store_true", help="Disable source separation (no Demucs stem)")
    p.add_argument("--suppress_numerals", action="store_true", help="Suppress numerals in transcript")
    p.add_argument("--parallel", action="store_true", help="Use diarize_parallel.py if available")
    p.add_argument("--timeout", type=int, default=None, help="Kill run after N seconds")

    p.add_argument("--use_custom", action="store_true", default=True,
                   help="Use diarize_custom.py if present (default ON)")
    p.add_argument("--no-custom", dest="use_custom", action="store_false",
                   help="Force upstream script (diarize.py/diarize_parallel.py)")

    p.add_argument("--keep_temp", action="store_true", help="Keep temp_outputs* folders")
    p.add_argument("--num_speakers", type=int, default=None, help="Optional speaker count hint")

    return p


def main():
    import sys
    args = _build_arg_parser().parse_args()

    outs = run_whisper_diarization_repo(
        audio_path=args.audio_path,
        out_dir=args.out_dir,
        overwrite_existing=args.overwrite_existing,
        repo_dir=args.repo_dir,
        whisper_model=args.whisper_model,
        language=args.language,
        device=args.device,
        batch_size=args.batch_size,
        no_stem=args.no_stem,
        suppress_numerals=args.suppress_numerals,
        parallel=args.parallel,
        timeout=args.timeout,
        use_custom=args.use_custom,
        keep_temp=args.keep_temp,
        num_speakers=args.num_speakers,
    )

    print(f"Work dir: {outs.work_dir}")
    if outs.raw_files:
        for k, v in outs.raw_files.items():
            print(f"{k.upper()}: {v}")
    else:
        print("No output files detected.")

    # return code 0 for success
    sys.exit(0)


if __name__ == "__main__":
    main()
