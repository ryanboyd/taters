"""
Who is on screen, and what their face is doing.

Frames are sampled with ffmpeg, faces are found with YuNet, each face gets an
emotion distribution from HSEmotion and an identity embedding from SFace, and
then the identities are worked out by tracking within shots and clustering the
tracks. The reasoning behind that last part is in :mod:`._identity`; the short
version is that clustering raw per-frame detections scatters one person across a
dozen identities, and tracking first is what stops it.

Three things this module refuses to do quietly, all of them the same mistake in
different clothes:

**It does not report action units.** Every open FACS implementation is either
licensed for non-commercial research only or pins a torch version that would
fight the install people already have. Approximating AUs from blendshapes and
calling them FACS would be worse than the gap.

**It does not write zero when it found nothing.** A video with no detectable
face has no mean emotion; it does not have a mean emotion of nought. Those cells
come out empty, so a row with no faces in it can be filtered out rather than
silently dragging an average down.

**It does not hide how much it saw.** Every mean is accompanied by the number of
frames it was taken over. A mean happiness across the 11% of frames where a face
was visible is a different quantity from one across 95%, and nothing else in the
output would tell them apart.

A warning to put next to any result from here: emotion recognition from faces is
contested. These models are trained on posed or crowd-labeled images, the
mapping from a facial configuration to a felt emotion is not one-to-one, and
Barrett et al. (2019) is the standard reference for why. What this measures is
facial appearance, which is not the same thing as what somebody felt.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import (Callable, Dict, List, Literal, Optional, Sequence, Tuple,
                    Union)

import numpy as np

from ..helpers.atomic import atomic_write
from ..helpers.cliargs import CliSpec
from ..helpers.feature_columns import ColumnSpec
from ..helpers.progress import announce
from ..helpers.provenance import records_settings
from . import _identity as ident
from ._ffmpeg_probe import scene_scores
from ._frames import extracted_frames
from ._onnx_faces import EMOTION_LABELS

PathLike = Union[str, Path]
Level = Literal["video", "face", "both"]

__all__ = ["analyze_faces", "FEATURE_COLUMNS", "BOOKKEEPING", "EMOTION_LABELS"]

#: How much was seen, rather than what was seen. Filterable, never a predictor:
#: a model that discovered your outcome tracks "how many frames had a face in
#: them" has discovered your camera setup.
BOOKKEEPING = (
    "face_frames_sampled", "face_frames_with_face", "face_detections",
    "face_per_frame_mean", "face_identities",
    "face_track_count", "face_frame_count", "face_quality_mean",
    "face_margin", "face_coverage",
)

_EMO = tuple(f"face_emo_{label}" for label in EMOTION_LABELS)
_EMO_SD = tuple(f"{c}_sd" for c in _EMO)

MEASURES = _EMO + _EMO_SD + (
    "face_box_area_mean", "face_box_area_sd",
    "face_center_x_mean", "face_center_y_mean",
    "face_interocular_mean", "face_roll_deg_mean", "face_roll_deg_sd",
)

FEATURE_COLUMNS = ColumnSpec(label="Faces",
                             names=tuple(MEASURES) + BOOKKEEPING)


def _sd(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else None


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _sharpness(crop: np.ndarray) -> float:
    """
    Variance of a Laplacian: high for a crisp face, near zero for a blurred one.

    Used only to weight which frames decide a track's identity, so a cheap
    proxy is the right amount of effort. A motion-blurred profile should not be
    what settles who somebody is.
    """
    if crop.size == 0:
        return 0.0
    grey = crop.astype(np.float64).mean(axis=2)
    if min(grey.shape) < 3:
        return 0.0
    lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
           - 4.0 * grey[1:-1, 1:-1])
    return float(lap.var())


def _roll_degrees(landmarks: Sequence[Tuple[float, float]]) -> Optional[float]:
    """
    Head tilt from the eye line, in degrees.

    Geometric, and labeled as such everywhere it surfaces: it is the angle
    between the eyes, not the calibrated head pose a fitted 3D model gives. It
    is honest about roll and says nothing about yaw or pitch.
    """
    if len(landmarks) < 2:
        return None
    (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
    return float(math.degrees(math.atan2(ry - ly, rx - lx)))


def _cut_frames(video_path: Path, fps: float, threshold: float) -> List[int]:
    """
    Where the cuts land, in *sampled frame* indices.

    The scene scores are computed on the real frame rate and the faces on our
    sampled one, so the times have to be mapped across. Without this a track
    would be allowed to walk straight through an edit.
    """
    try:
        scores = scene_scores(video_path)
    except Exception:
        return []
    return sorted({int(round(t * fps)) for _f, t, s in scores if s >= threshold})


@records_settings(binding=("video_path",),
                  outputs=("out_csv",),
                  bookkeeping=BOOKKEEPING)
def analyze_faces(
    *,
    video_path: PathLike,
    out_csv: Optional[PathLike] = None,
    out_dir: Optional[PathLike] = None,
    level: Level = "video",
    fps: float = 1.0,
    max_frames: Optional[int] = 3000,
    detect_threshold: float = 0.6,
    n_faces: Optional[int] = None,
    identity_threshold: float = 0.45,
    min_track_length: int = 2,
    max_gap_frames: int = 2,
    min_iou: float = 0.3,
    cut_threshold: float = 10.0,
    include_framewise: bool = False,
    overwrite_existing: bool = False,
    verbose: bool = True,
    on_progress: Optional[Callable[..., None]] = None,
) -> Path:
    """
    Find the faces in a video and measure what they are doing.

    Parameters
    ----------
    video_path : str or pathlib.Path
        The video to analyze.
    out_csv : str or pathlib.Path, optional
        Where the per-video row goes. Defaults to ``<out_dir>/<stem>.csv``.
    out_dir : str or pathlib.Path, optional
        Folder for the default paths. Defaults to ``./features/faces``.
    level : {"video", "face", "both"}, default="video"
        ``video`` writes one row for the whole file. ``face`` writes one row per
        person identified. ``both`` writes both.
    fps : float, default=1.0
        Frames sampled per second. One a second is ample for who is on screen;
        raise it for fast expression changes, at linear cost.
    max_frames : int, optional
        Stop after this many sampled frames.
    detect_threshold : float, default=0.6
        Minimum detector confidence for a face to count.
    n_faces : int, optional
        How many people are in the video, if you know. **Prefer this** --
        it is far more reliable than a threshold. Refused if it is smaller than
        the number of faces actually visible together in one frame.
    identity_threshold : float, default=0.45
        Used only when ``n_faces`` is not given: cosine distance above which two
        tracks are different people. Lower splits one person into several;
        higher merges two people into one.
    min_track_length : int, default=2
        Drop tracks shorter than this many frames. A one-frame track is usually
        a false detection.
    max_gap_frames : int, default=2
        How long a face may vanish and still continue the same track.
    min_iou : float, default=0.3
        Minimum box overlap between frames to continue a track.
    cut_threshold : float, default=10.0
        Scene-change score that counts as a cut. Tracks never cross one.
    include_framewise : bool, default=False
        Also write one row per detection, with its frame and time.
    overwrite_existing : bool, default=False
        If ``False`` and the output exists, skip and return it.
    verbose : bool, default=True
        Print what is happening.
    on_progress : callable, optional
        Called as ``on_progress(done, total, message=None)``.

    Returns
    -------
    pathlib.Path
        Path to the per-video CSV (or the per-face one when ``level="face"``).

    Raises
    ------
    FileNotFoundError
        If the video is not there.
    taters.video._identity.IdentityConflict
        If ``n_faces`` is smaller than the number of people demonstrably on
        screen at once.

    Notes
    -----
    Identities are worked out by tracking faces within shots and then clustering
    the tracks, never by clustering individual detections. **Verifying that the
    identities are the people you think they are is your job**: the output
    reports how many were found, how many frames each was seen in, and how close
    a call each assignment was, so that check is quick -- but it is still a
    check somebody has to make before publishing per-person results.
    """
    from PIL import Image

    from ._onnx_faces import Emotion, SFace, YuNet, align_face

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    base = Path(out_dir) if out_dir else (Path.cwd() / "features" / "faces")
    if out_csv is None:
        out_csv = base / f"{video_path.stem}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_face_csv = out_csv.with_name(f"{out_csv.stem}_by_face.csv")
    framewise_csv = out_csv.with_name(f"{out_csv.stem}_framewise.csv")

    if not overwrite_existing and out_csv.is_file():
        if verbose:
            print(f"Face output already exists; returning {out_csv}")
        return out_csv

    announce(on_progress, "finding cuts")
    cuts = _cut_frames(video_path, fps, cut_threshold)

    detector = YuNet(score_threshold=detect_threshold)
    emotion = Emotion()
    recognizer = SFace()

    detections: List[ident.Detection] = []
    rows: List[Dict[str, object]] = []
    n_sampled = 0

    with extracted_frames(video_path, fps=fps, max_frames=max_frames,
                          on_progress=on_progress) as frames:
        n_sampled = len(frames)
        total = max(1, n_sampled)
        for k, frame in enumerate(frames):
            if on_progress is not None:
                on_progress(k, total, "reading faces")
            image = np.asarray(Image.open(frame.path).convert("RGB"))
            for face in detector.detect(image):
                x, y, w, h = (int(round(v)) for v in face.box)
                x, y = max(0, x), max(0, y)
                crop = image[y:y + max(1, h), x:x + max(1, w)]
                if crop.size == 0:
                    continue

                probs = emotion.predict(crop)
                try:
                    embedding = recognizer.embed(
                        align_face(image, face.landmarks))
                except Exception:
                    embedding = None        # a crop at the edge can be empty

                detections.append(ident.Detection(
                    frame_index=frame.index, time_s=frame.time_s,
                    box=face.box, confidence=face.score,
                    embedding=embedding, sharpness=_sharpness(crop)))
                rows.append({
                    "frame": frame.index, "time_s": round(frame.time_s, 3),
                    "x": round(face.x, 1), "y": round(face.y, 1),
                    "w": round(face.w, 1), "h": round(face.h, 1),
                    "score": round(face.score, 4),
                    "roll_deg": _roll_degrees(face.landmarks),
                    **{f"emo_{k2}": round(v, 4) for k2, v in probs.items()},
                })

    announce(on_progress, "working out who is who")
    tracks = ident.build_tracks(
        detections, shot_boundaries=cuts, max_gap_frames=max_gap_frames,
        min_iou=min_iou, min_length=min_track_length)
    embeddings = [ident.track_embedding(t) for t in tracks]
    labels, margins = ident.cluster_tracks(
        embeddings, cannot_link=ident.cooccurring_pairs(tracks),
        n_faces=n_faces, threshold=identity_threshold)

    # Which identity each detection belongs to, for the per-face rollup.
    # Keyed on object identity rather than on the box coordinates: the rows
    # store rounded values for legibility, and matching a rounded float against
    # an unrounded one silently found nothing at all. `build_tracks` holds the
    # same Detection objects we passed it, so this is exact.
    row_of = {id(d): i for i, d in enumerate(detections)}
    for track, label in zip(tracks, labels):
        for d in track.detections:
            idx = row_of.get(id(d))
            if idx is not None:
                rows[idx]["face_id"] = label
    for row in rows:
        row.setdefault("face_id", None)

    _write_video_row(out_csv, video_path, rows, detections, tracks, labels,
                     n_sampled)
    if level in ("face", "both"):
        _write_face_rows(per_face_csv, video_path, rows, tracks, labels,
                         margins, n_sampled)
    if include_framewise:
        _write_framewise(framewise_csv, video_path, rows)

    if verbose:
        print(f"Faces: {video_path.name} -> {len(detections)} detections in "
              f"{n_sampled} frames, {len(set(labels))} identities")
    return out_csv if level != "face" else per_face_csv


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Mean and SD of each measure over a set of detections."""
    out: Dict[str, object] = {}
    for label in EMOTION_LABELS:
        vals = [float(r[f"emo_{label}"]) for r in rows]
        out[f"face_emo_{label}"] = _mean(vals)
        out[f"face_emo_{label}_sd"] = _sd(vals)

    areas = [float(r["w"]) * float(r["h"]) for r in rows]
    out["face_box_area_mean"] = _mean(areas)
    out["face_box_area_sd"] = _sd(areas)
    out["face_center_x_mean"] = _mean([float(r["x"]) + float(r["w"]) / 2 for r in rows])
    out["face_center_y_mean"] = _mean([float(r["y"]) + float(r["h"]) / 2 for r in rows])
    out["face_interocular_mean"] = _mean([float(r["w"]) * 0.45 for r in rows])
    rolls = [float(r["roll_deg"]) for r in rows if r.get("roll_deg") is not None]
    out["face_roll_deg_mean"] = _mean(rolls)
    out["face_roll_deg_sd"] = _sd(rolls)
    return out


def _fmt(value) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def _write_video_row(path: Path, video: Path, rows, detections, tracks,
                     labels, n_sampled: int) -> None:
    summary = _summarize(rows) if rows else {c: None for c in MEASURES}
    frames_seen = len({r["frame"] for r in rows})
    summary.update({
        "face_frames_sampled": n_sampled,
        "face_frames_with_face": frames_seen,
        "face_detections": len(rows),
        "face_per_frame_mean": (len(rows) / n_sampled) if n_sampled else None,
        "face_identities": len(set(labels)) if labels else 0,
        "face_coverage": (frames_seen / n_sampled) if n_sampled else None,
        "face_track_count": len(tracks),
        "face_frame_count": frames_seen,
        "face_quality_mean": _mean([float(r["score"]) for r in rows]),
        "face_margin": None,
    })
    header = ["source_path", "source"] + list(MEASURES) + list(BOOKKEEPING)
    with atomic_write(path, newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerow([str(video.resolve()), video.stem]
                   + [_fmt(summary.get(c)) for c in MEASURES]
                   + [_fmt(summary.get(c)) for c in BOOKKEEPING])


def _write_face_rows(path: Path, video: Path, rows, tracks, labels, margins,
                     n_sampled: int) -> None:
    by_face: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        if row.get("face_id") is not None:
            by_face.setdefault(int(row["face_id"]), []).append(row)

    tracks_per: Dict[int, int] = {}
    margin_per: Dict[int, float] = {}
    for label, margin in zip(labels, margins):
        tracks_per[label] = tracks_per.get(label, 0) + 1
        margin_per[label] = min(margin_per.get(label, float("inf")), margin)

    header = (["source_path", "source", "face_id"] + list(MEASURES)
              + list(BOOKKEEPING))
    with atomic_write(path, newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for face_id in sorted(by_face):
            mine = by_face[face_id]
            summary = _summarize(mine)
            seen = len({r["frame"] for r in mine})
            summary.update({
                "face_frames_sampled": n_sampled,
                "face_frames_with_face": seen,
                "face_detections": len(mine),
                "face_per_frame_mean": (len(mine) / n_sampled) if n_sampled else None,
                "face_identities": len(by_face),
                "face_coverage": (seen / n_sampled) if n_sampled else None,
                "face_track_count": tracks_per.get(face_id, 0),
                "face_frame_count": seen,
                "face_quality_mean": _mean([float(r["score"]) for r in mine]),
                # how close a call this identity was. A small number means the
                # nearest other person was nearly as good a match.
                "face_margin": (None if margin_per.get(face_id) in (None, float("inf"))
                                else margin_per[face_id]),
            })
            w.writerow([str(video.resolve()), video.stem, f"face_{face_id + 1}"]
                       + [_fmt(summary.get(c)) for c in MEASURES]
                       + [_fmt(summary.get(c)) for c in BOOKKEEPING])


def _write_framewise(path: Path, video: Path, rows) -> None:
    cols = ["frame", "time_s", "face_id", "x", "y", "w", "h", "score",
            "roll_deg"] + [f"emo_{label}" for label in EMOTION_LABELS]
    with atomic_write(path, newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["source", *cols])
        for row in rows:
            w.writerow([video.stem] + [
                _fmt(row.get(c) if c != "face_id"
                     else (None if row.get("face_id") is None
                           else f"face_{int(row['face_id']) + 1}"))
                for c in cols])


CLI = CliSpec(
    analyze_faces,
    description="Find faces in a video and measure emotion, geometry and identity.",
    aliases={"video_path": ["--video"], "out_csv": ["--out"]},
)


def main(argv=None) -> int:
    return CLI.run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
