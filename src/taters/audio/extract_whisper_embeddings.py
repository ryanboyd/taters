"""High-level, environment-safe wrapper for exporting Whisper encoder embeddings.

This module provides a single entry point, :func:`extract_whisper_embeddings`,
which (by default) launches a subprocess to extract embeddings using a dedicated
worker module. The subprocess approach avoids CUDA/Torch collisions with other
parts of your pipeline.

Two modes are supported:

1) Transcript-driven mode
   Pass ``transcript_csv`` to compute one embedding vector per transcript row
   (e.g., per diarized segment). The output is a CSV with columns
   ``start_time,end_time,speaker,e0..e{D-1}``.

2) General-audio mode
   Omit ``transcript_csv`` to analyze the raw WAV. You can segment by fixed
   windows or by non-silent regions; optionally aggregate to a single mean row.
"""

from __future__ import annotations
import os, sys, shlex, subprocess
from pathlib import Path
from typing import Optional, Literal, Union

def extract_whisper_embeddings(
    *,
    # required
    source_wav: Union[str, Path],

    # optional transcript-driven mode
    transcript_csv: Optional[Union[str, Path]] = None,
    time_unit: Literal["auto", "ms", "s", "samples"] = "auto",

    # general-audio mode (used when transcript_csv is None)
    strategy: Literal["windows", "nonsilent"] = "windows",
    window_s: float = 30.0,
    hop_s: float = 15.0,
    min_seg_s: float = 1.0,
    top_db: float = 30.0,
    aggregate: Literal["none", "mean"] = "none",

    # outputs
    output_dir: Optional[Union[str, Path]] = None,
    overwrite_existing: bool = False,  # if the file already exists, let's not overwrite by default

    # model/runtime
    model_name: str = "base",
    device: Literal["auto", "cuda", "cpu"] = "auto",
    compute_type: str = "float16",

    # execution strategy
    run_in_subprocess: bool = True,
    extra_env: Optional[dict] = None,
    verbose: bool = True,

    # where the extractor lives (python -m <module>)
    extractor_module: str = "taters.audio.extract_whisper_embeddings_subproc",
) -> Path:
    """
    Export Whisper encoder embeddings to a CSV file, using a subprocess by default.

    Parameters
    ----------
    source_wav : str | Path
        Path to the input WAV. Must be readable by `librosa`.
    transcript_csv : str | Path | None, optional
        If provided, enables transcript-driven mode. The CSV is expected to contain
        timestamp columns and (optionally) a speaker column. A row is emitted per
        transcript segment.
    time_unit : {"auto","ms","s","samples"}, default "auto"
        How to interpret timestamps in `transcript_csv`. In "auto", the worker
        heuristically infers the unit from max end time vs audio duration.
    strategy : {"windows","nonsilent"}, default "windows"
        General-audio mode only. "windows" uses fixed sized windows with overlap;
        "nonsilent" uses an energy-based splitter (librosa.effects.split).
    window_s, hop_s : float, default 30.0, 15.0
        General-audio mode only. Window length and hop (seconds).
    min_seg_s : float, default 1.0
        General-audio mode only. Skip segments shorter than this many seconds.
    top_db : float, default 30.0
        General-audio mode only ("nonsilent"). Threshold (dB) below reference to
        consider as silence. Smaller → more segments; larger → fewer.
    aggregate : {"none","mean"}, default "none"
        General-audio mode only. If "mean", a single pooled row is written covering
        the entire file; otherwise one row per segment.
    output_dir : str | Path | None, optional
        Directory for the output CSV. If None, defaults to
        ``./features/whisper-embeddings``.
    model_name : str, default "base"
        Model identifier passed through to the worker (e.g., "tiny", "base",
        "small", "large-v3" or a local CTranslate2 model directory).
    device : {"auto","cuda","cpu"}, default "auto"
        Runtime device. If "cpu", environment variables are set to disable CUDA
        in the child process.
    compute_type : str, default "float16"
        CTranslate2 compute type (e.g., "float16", "int8", "float32"); passed to
        the worker module.
    run_in_subprocess : bool, default True
        If True (recommended), runs extraction in a separate Python process to
        isolate Torch/CUDA state from the parent process.
    extra_env : dict | None, optional
        Additional environment variables to inject into the child process.
    verbose : bool, default True
        If True, print the launched command and the child's stdout.
    extractor_module : str, default "chopshop.audio.extract_whisper_embeddings_subproc"
        Dotted module path whose ``__main__`` implements the extractor CLI.

    Returns
    -------
    Path
        Path to the written embeddings CSV. Pattern:
        ``<output_dir>/<source_stem>_embeddings.csv``.

    Notes
    -----
    - The subprocess writes and exits. The parent returns once the file exists.
    - If `transcript_csv` is supplied, the worker runs in transcript mode; otherwise
      general-audio mode is used with the given segmentation strategy.
    - Failures in the child process are re-raised with the captured stdout/stderr
      to ease debugging.

    Examples
    --------
    Transcript per-segment embeddings:

    >>> extract_whisper_embeddings(
    ...     source_wav="audio/session.wav",
    ...     transcript_csv="transcripts/session.csv",
    ...     time_unit="ms",
    ...     model_name="small",
    ...     device="cuda",
    ... )

    Whole-file mean embedding:

    >>> extract_whisper_embeddings(
    ...     source_wav="audio/session.wav",
    ...     strategy="nonsilent",
    ...     aggregate="mean",
    ...     output_dir="features/whisper-embeddings",
    ... )
    """
    
    source_wav = Path(source_wav).resolve()
    # default to ./features/whisper-embeddings when not provided
    out_dir_final = (
        Path(output_dir).resolve()
        if output_dir
        else (Path.cwd() / "features" / "whisper-embeddings")
    )

    out_dir_final.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir_final / f"{source_wav.stem}_embeddings.csv"

    if not overwrite_existing and Path(output_csv).is_file():
        print("Whisper embedding feature output file already exists; returning existing file.")
        return output_csv

    if not run_in_subprocess:
        # ---- In-process path (only when you’re sure no Torch/CUDA conflicts) ----
        from ..audio.extract_whisper_embeddings import (  # type: ignore
            export_segment_embeddings_csv,
            export_audio_embeddings_csv,
            EmbedConfig,
        )
        cfg = EmbedConfig(model_name=model_name, device=device, compute_type=compute_type, time_unit=time_unit)
        if transcript_csv is not None:
            transcript_csv = Path(transcript_csv).resolve()
            return Path(
                export_segment_embeddings_csv(
                    transcript_csv=transcript_csv,
                    source_wav=source_wav,
                    output_dir=out_dir_final,
                    config=cfg,
                )
            )
        else:
            return Path(
                export_audio_embeddings_csv(
                    source_wav=source_wav,
                    output_dir=out_dir_final,
                    config=cfg,
                    strategy=strategy,
                    window_s=window_s,
                    hop_s=hop_s,
                    min_seg_s=min_seg_s,
                    top_db=top_db,
                    aggregate=aggregate,
                )
            )

    # ---- Subprocess path (recommended) ----
    env = os.environ.copy()
    # Keep Transformers from importing heavy backends in the child
    env.setdefault("TRANSFORMERS_NO_TORCH", "1")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")

    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    if device == "cpu":
        # Make sure the child won’t try CUDA
        env.update({"CUDA_VISIBLE_DEVICES": "", "USE_CUDA": "0", "FORCE_CPU": "1"})
    else:
        # Best-effort: prepend cuDNN wheel's lib dir if available
        try:
            import nvidia.cudnn, pathlib  # type: ignore
            cudnn_lib = str(pathlib.Path(nvidia.cudnn.__file__).with_name("lib"))
            env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        except Exception:
            pass

    cmd = [
        sys.executable, "-m", extractor_module,
        "--source_wav", str(source_wav),
        "--output_dir", str(out_dir_final),
        "--model_name", model_name,
        "--device", device,
        "--compute_type", compute_type,
    ]

    if transcript_csv is not None:
        transcript_csv = Path(transcript_csv).resolve()
        cmd += ["--transcript_csv", str(transcript_csv), "--time_unit", time_unit]
    else:
        cmd += [
            "--strategy", strategy,
            "--window_s", str(window_s),
            "--hop_s", str(hop_s),
            "--min_seg_s", str(min_seg_s),
            "--top_db", str(top_db),
            "--aggregate", aggregate,
        ]

    if verbose:
        print("Launching embedding subprocess:")
        print(" ", shlex.join(cmd))

    try:
        res = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if verbose and res.stdout:
            print(res.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Embedding subprocess failed with code {e.returncode}\n"
            f"CMD: {shlex.join(cmd)}\n"
            f"STDOUT:\n{(e.stdout or '').strip()}\n\n"
            f"STDERR:\n{(e.stderr or '').strip()}"
        ) from e

    if not output_csv.exists():
        raise FileNotFoundError(f"Expected embeddings CSV not found: {output_csv}")

    if verbose:
        print(f"Embeddings CSV written to: {output_csv}")

    return output_csv


# --- CLI support: run this module directly -----------------------------------
def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Taters: export Whisper encoder embeddings (env-safe wrapper)."
    )
    # required
    p.add_argument("--source_wav", required=True, help="Path to input WAV")

    # optional transcript-driven mode
    p.add_argument("--transcript_csv", default=None, help="Transcript CSV for segment-level embeddings")
    p.add_argument("--time_unit", default="auto", choices=("auto", "ms", "s", "samples"))

    # general-audio mode
    p.add_argument("--strategy", default="windows", choices=("windows", "nonsilent"))
    p.add_argument("--window_s", type=float, default=30.0)
    p.add_argument("--hop_s", type=float, default=15.0)
    p.add_argument("--min_seg_s", type=float, default=1.0)
    p.add_argument("--top_db", type=float, default=30.0)
    p.add_argument("--aggregate", default="none", choices=("none", "mean"))

    # outputs
    p.add_argument("--output_dir", default=None, 
                   help="Output directory for the CSV (default: ./features/whisper-embeddings)",
    )
    p.add_argument("--overwrite_existing", type=bool, default=False,
                    help="Do you want to overwrite the output file if it already exists?")

    # model/runtime
    p.add_argument("--model_name", default="base")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--compute_type", default="float16")

    # execution strategy
    p.add_argument("--run_in_subprocess", action="store_true", default=True,
                   help="(default) Run extractor in a subprocess")
    p.add_argument("--no-subprocess", dest="run_in_subprocess", action="store_false",
                   help="Run in-process (only if you're sure no CUDA/Torch conflicts)")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")

    # advanced
    p.add_argument("--extractor_module", default="taters.audio.extract_whisper_embeddings_subproc",
                   help="Python module to run for the actual extraction")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Call the same function this module exposes as a class method.
    # We don't use 'self' internally, so pass None.
    out = extract_whisper_embeddings(
        source_wav=args.source_wav,
        transcript_csv=args.transcript_csv,
        time_unit=args.time_unit,
        strategy=args.strategy,
        window_s=args.window_s,
        hop_s=args.hop_s,
        min_seg_s=args.min_seg_s,
        top_db=args.top_db,
        aggregate=args.aggregate,
        output_dir=args.output_dir,
        overwrite_existing=args.overwrite_existing,
        model_name=args.model_name,
        device=args.device,
        compute_type=args.compute_type,
        run_in_subprocess=args.run_in_subprocess,
        verbose=args.verbose,
        extractor_module=args.extractor_module,
    )
    print(str(out))

if __name__ == "__main__":
    main()
