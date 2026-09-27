"""
Turning a video into numbers that put similar-looking videos close together.

The visual counterpart to the sentence embeddings in the text side. Frames are
sampled with ffmpeg, each is embedded with a vision encoder, and the frames are
pooled into one vector per video.

This costs nothing to install. `transformers` and torch are already base
dependencies -- they arrive with `sentence-transformers` -- and the image
preprocessing goes through PIL, so **torchvision is not needed**, which is the
one dependency that would have fought the torch build people already have.

What these are good for is the same as any embedding: similarity, clustering,
and as predictors in a model. What they are not good for is explaining
anything. A CLIP vector says two videos look alike; it will not tell you what
about them.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, List, Literal, Optional, Union

import numpy as np

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec
from ..helpers.progress import announce
from ..helpers.provenance import records_settings
from ._frames import extracted_frames

PathLike = Union[str, Path]
Pooling = Literal["mean", "max", "mean_max"]

__all__ = ["extract_video_embeddings", "FEATURE_COLUMNS", "BOOKKEEPING"]

#: How many frames went into the vector. Not a measure: a model that fit it
#: would be fitting video length.
BOOKKEEPING = ("vemb_frames",)

FEATURE_COLUMNS = ColumnSpec(
    label="Video embeddings",
    dynamic="one column per dimension of the encoder you picked, plus the "
            "number of frames pooled",
    names=BOOKKEEPING,
)


def _pool(frames: np.ndarray, how: Pooling) -> np.ndarray:
    """
    Reduce per-frame vectors to one per video.

    Mean is the sensible default -- it is what the video is like on average.
    Max keeps the strongest response on each dimension, which is what you want
    if you are looking for whether something ever appeared rather than how
    typical it was. `mean_max` keeps both, at twice the width.
    """
    if how == "mean":
        return frames.mean(axis=0)
    if how == "max":
        return frames.max(axis=0)
    if how == "mean_max":
        return np.concatenate([frames.mean(axis=0), frames.max(axis=0)])
    raise ValueError(f"pooling must be mean, max or mean_max; got {how!r}")


@records_settings(binding=("video_path",),
                  outputs=("out_csv",),
                  bookkeeping=BOOKKEEPING)
def extract_video_embeddings(
    *,
    video_path: PathLike,
    out_csv: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    model_name: str = "openai/clip-vit-base-patch32",
    fps: float = 0.5,
    max_frames: Optional[int] = 600,
    pooling: Pooling = "mean",
    batch_size: int = 16,
    device: Literal["auto", "cuda", "cpu"] = "auto",
    normalize: bool = True,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Embed a video's frames with a vision encoder and pool them into one vector.

    Parameters
    ----------
    video_path : str or pathlib.Path
        The video to embed.
    out_csv : str or pathlib.Path, optional
        Where the row goes. Defaults to ``<out_dir>/<stem>.csv``.
    out_dir : str or pathlib.Path, optional
        Defaults to ``./features/video_embeddings``.
    model_name : str, default="openai/clip-vit-base-patch32"
        Any Hugging Face vision encoder. The default is small, fast on CPU and
        very widely used, which makes it the easiest to compare against other
        people's work.
    fps : float, default=0.5
        Frames sampled per second. One every two seconds is plenty for "what
        does this video look like"; raise it if you care about brief events.
    max_frames : int, optional
        Stop after this many frames.
    pooling : {"mean", "max", "mean_max"}, default="mean"
        How per-frame vectors become one vector.
    batch_size : int, default=16
        Frames per forward pass.
    device : str, default="auto"
        ``"auto"``, ``"cpu"``, ``"cuda"``.
    normalize : bool, default=True
        L2-normalize the result, so cosine similarity is a dot product.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, skip and return it.
    verbose : bool, default=True
        Print what is happening.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)``.

    Returns
    -------
    pathlib.Path
        Path to the written CSV: one row, one column per dimension.

    Notes
    -----
    Needs no torchvision. The image preprocessing runs through PIL, which is
    what keeps this free of the one dependency that pins an exact torch build.
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    base = Path(out_dir) if out_dir else (Path.cwd() / "features" / "video_embeddings")
    if out_csv is None:
        out_csv = base / f"{video_path.stem}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite_existing and out_csv.is_file():
        if verbose:
            print(f"Video embedding output already exists; returning {out_csv}")
        return out_csv

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    announce(on_progress, f"loading {model_name}")
    # use_fast=False is load-bearing, not a preference. The fast image
    # processors are torchvision-backed, and transformers is in the middle of
    # flipping the default to True -- which would quietly reintroduce the one
    # dependency this whole package was arranged to avoid, on some future
    # upgrade, with no error until somebody's torch got dragged along with it.
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    vectors: List[np.ndarray] = []
    with extracted_frames(video_path, fps=fps, max_frames=max_frames,
                          width=None, on_progress=on_progress) as frames:
        total = max(1, len(frames))
        for start in range(0, len(frames), batch_size):
            chunk = frames[start:start + batch_size]
            if on_progress is not None:
                on_progress(start, total, "embedding frames")
            images = [Image.open(f.path).convert("RGB") for f in chunk]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    out = model.get_image_features(**inputs)
                else:
                    hidden = model(**inputs).last_hidden_state
                    out = hidden.mean(dim=1)
            vectors.append(out.detach().cpu().numpy())

    if not vectors:
        raise ValueError(
            f"No frames could be read from {video_path.name}. It may have no "
            "video stream, or ffmpeg may not be able to decode it.")

    stacked = np.concatenate(vectors, axis=0)
    pooled = _pool(stacked, pooling)
    if normalize:
        norm = float(np.linalg.norm(pooled))
        if norm:
            pooled = pooled / norm

    header = (["source_path", "source"]
              + [f"vemb_{i:04d}" for i in range(pooled.shape[0])]
              + list(BOOKKEEPING))
    with atomic_write(out_csv, newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerow([str(video_path.resolve()), video_path.stem]
                   + [round(float(v), 6) for v in pooled]
                   + [int(stacked.shape[0])])

    if verbose:
        print(f"Video embeddings: {video_path.name} -> {pooled.shape[0]} dims "
              f"from {stacked.shape[0]} frames")
    return out_csv


CLI = CliSpec(
    extract_video_embeddings,
    description="Embed a video's frames with a vision encoder, pooled per video.",
    aliases={"video_path": ["--video"], "out_csv": ["--out"],
             "model_name": ["--model"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
