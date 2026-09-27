"""
Working out which faces are the same person.

Naively, you embed every detected face and cluster the lot. That is the version
that scatters one person across a dozen identities, because a single face varies
more across poses and lighting than two faces of similar-looking people vary
from each other.

What the video face clustering literature actually does -- and what this
implements -- is three stages: **track locally, aggregate per track, cluster
tracks globally**. A track is a run of detections that are demonstrably the same
person because they are adjacent in time and overlap in space, so averaging over
one is averaging over a single identity by construction, and the thing being
clustered is a handful of stable track embeddings rather than thousands of noisy
frame ones.

Two constraints do most of the work here, and neither costs anything:

**Shot boundaries.** A cut to a new camera angle looks like smooth motion to a
box tracker, so a track that crosses one is usually two people. The visual
dynamics step already computes cuts, so we get this for free.

**Co-occurrence.** If two faces are visible in the same frame they are two
people, whatever their embeddings say. This is the single highest-value rule in
the module: it is certain rather than probabilistic, and it rules out exactly
the merge that a similarity threshold gets wrong.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "Detection", "Track", "iou", "build_tracks", "track_embedding",
    "cooccurring_pairs", "max_simultaneous_faces", "cluster_tracks",
    "IdentityConflict",
]


class IdentityConflict(ValueError):
    """The requested number of faces cannot be reconciled with the video."""


@dataclass
class Detection:
    """One face found in one frame."""

    frame_index: int
    time_s: float
    box: Tuple[float, float, float, float]      # x, y, w, h in pixels
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    sharpness: float = 1.0

    @property
    def area(self) -> float:
        return float(self.box[2]) * float(self.box[3])

    @property
    def quality(self) -> float:
        """
        How much this detection should count toward its track's identity.

        Confidence times the square root of the face's size times sharpness. A
        tiny, blurred, barely-detected face is still a real detection worth
        reporting -- it just should not be the thing that decides who the person
        is. The square root is there because area grows quadratically with
        distance and we want the weight to fall off with distance, not with its
        square.
        """
        return (max(0.0, float(self.confidence))
                * math.sqrt(max(0.0, self.area))
                * max(0.0, float(self.sharpness)))


@dataclass
class Track:
    """A run of detections that are the same person by construction."""

    detections: List[Detection] = field(default_factory=list)

    @property
    def frames(self) -> Set[int]:
        return {d.frame_index for d in self.detections}

    @property
    def start_frame(self) -> int:
        return min(d.frame_index for d in self.detections)

    @property
    def end_frame(self) -> int:
        return max(d.frame_index for d in self.detections)

    def __len__(self) -> int:
        return len(self.detections)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - overlap
    return float(overlap / union) if union > 0 else 0.0


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a, b) / (na * nb))


def _shot_of(frame_index: int, boundaries: Sequence[int]) -> int:
    """Which shot a frame belongs to, given the frame indices where cuts land."""
    shot = 0
    for b in boundaries:
        if frame_index >= b:
            shot += 1
        else:
            break
    return shot


def build_tracks(detections: Sequence[Detection], *,
                 shot_boundaries: Sequence[int] = (),
                 max_gap_frames: int = 2,
                 min_iou: float = 0.3,
                 min_similarity: float = 0.0,
                 min_length: int = 1) -> List[Track]:
    """
    Link detections across frames into tracks.

    Greedy, highest-overlap-first assignment per frame. Greedy is the right
    trade here: an optimal assignment differs from it only when two faces are
    nearly on top of each other, and in that case the embeddings are the thing
    that should decide, not the boxes.

    Parameters
    ----------
    detections : sequence of Detection
        All detections, any order.
    shot_boundaries : sequence of int, default=()
        Frame indices where a cut happens. A track never crosses one.
    max_gap_frames : int, default=2
        How many frames a face may go undetected and still continue the same
        track. Blinks, turns and momentary occlusions cost a frame or two.
    min_iou : float, default=0.3
        Minimum box overlap to continue a track.
    min_similarity : float, default=0.0
        Minimum embedding cosine similarity to continue a track, when
        embeddings are present. ``0.0`` means "do not use embeddings here".
    min_length : int, default=1
        Tracks shorter than this are dropped. A one-frame track is usually a
        false positive, and it is also the one most likely to be a bad crop.

    Returns
    -------
    list of Track
        In order of first appearance.
    """
    by_frame: Dict[int, List[Detection]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_index, []).append(d)

    boundaries = sorted(set(int(b) for b in shot_boundaries))
    open_tracks: List[Track] = []
    done: List[Track] = []

    for frame in sorted(by_frame):
        shot = _shot_of(frame, boundaries)

        # a track cannot survive a cut, and cannot survive a long absence
        still_open: List[Track] = []
        for tr in open_tracks:
            last = tr.detections[-1]
            if (_shot_of(last.frame_index, boundaries) != shot
                    or frame - last.frame_index > max_gap_frames):
                done.append(tr)
            else:
                still_open.append(tr)
        open_tracks = still_open

        # score every (track, detection) pair, then take them best-first
        candidates = []
        for ti, tr in enumerate(open_tracks):
            last = tr.detections[-1]
            for di, det in enumerate(by_frame[frame]):
                overlap = iou(last.box, det.box)
                if overlap < min_iou:
                    continue
                sim = _cosine(last.embedding, det.embedding)
                if sim is not None and min_similarity > 0.0 and sim < min_similarity:
                    continue
                candidates.append((overlap if sim is None else overlap + sim,
                                   ti, di))
        candidates.sort(reverse=True)

        used_tracks: Set[int] = set()
        used_dets: Set[int] = set()
        for _score, ti, di in candidates:
            if ti in used_tracks or di in used_dets:
                continue
            open_tracks[ti].detections.append(by_frame[frame][di])
            used_tracks.add(ti)
            used_dets.add(di)

        for di, det in enumerate(by_frame[frame]):
            if di not in used_dets:
                open_tracks.append(Track([det]))

    done.extend(open_tracks)
    done.sort(key=lambda t: (t.start_frame, t.detections[0].box[0]))
    return [t for t in done if len(t) >= min_length]


def track_embedding(track: Track) -> Optional[np.ndarray]:
    """
    One L2-normalized embedding for a track, weighted by detection quality.

    A plain mean lets the worst frame in a track pull as hard as the best, and
    the worst frame in a track is usually a motion-blurred profile. Weighting is
    the cheap version of what the recent work does with a learned attention
    model, and it addresses the same failure.

    Returns ``None`` when no detection in the track carried an embedding.
    """
    vecs = [(d.embedding, d.quality) for d in track.detections
            if d.embedding is not None]
    if not vecs:
        return None
    weights = np.array([max(w, 1e-9) for _v, w in vecs], dtype=np.float64)
    stack = np.stack([np.asarray(v, dtype=np.float64) for v, _w in vecs])
    mean = (stack * weights[:, None]).sum(axis=0) / weights.sum()
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def cooccurring_pairs(tracks: Sequence[Track]) -> Set[Tuple[int, int]]:
    """
    Track pairs that share a frame, and therefore cannot be the same person.

    Certain rather than probabilistic, which is what makes it worth more than
    any amount of embedding tuning.
    """
    pairs: Set[Tuple[int, int]] = set()
    frames = [t.frames for t in tracks]
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            if frames[i] & frames[j]:
                pairs.add((i, j))
    return pairs


def max_simultaneous_faces(detections: Sequence[Detection]) -> int:
    """
    The most faces ever visible at once.

    A hard lower bound on how many people are in the video, and the number that
    makes a too-small ``n_faces`` obviously wrong instead of quietly wrong.
    """
    counts: Dict[int, int] = {}
    for d in detections:
        counts[d.frame_index] = counts.get(d.frame_index, 0) + 1
    return max(counts.values()) if counts else 0


def cluster_tracks(embeddings: Sequence[Optional[np.ndarray]], *,
                   cannot_link: Set[Tuple[int, int]] = frozenset(),
                   n_faces: Optional[int] = None,
                   threshold: Optional[float] = 0.45,
                   ) -> Tuple[List[int], List[float]]:
    """
    Group track embeddings into identities.

    Exactly one of ``n_faces`` and ``threshold`` decides where clustering stops.
    Prefer ``n_faces`` whenever the number of people is actually known -- it is
    the reliable answer, and a threshold is a guess about footage quality.

    Parameters
    ----------
    embeddings : sequence of array or None
        One per track. A ``None`` gets its own identity: we could not tell who
        it was, and folding it into somebody is worse than admitting that.
    cannot_link : set of (int, int)
        Track pairs that must not share an identity. See
        :func:`cooccurring_pairs`.
    n_faces : int, optional
        How many people are in the video, when you know.
    threshold : float, optional
        Cosine distance above which two tracks are different people. Used only
        when ``n_faces`` is ``None``.

    Returns
    -------
    (labels, margins)
        ``labels[i]`` is the identity of track *i*. ``margins[i]`` is the cosine
        distance from that track to the nearest track of a *different*
        identity -- small means this assignment was a close call, and it is the
        number to look at before believing any per-person result.

    Raises
    ------
    IdentityConflict
        If ``n_faces`` is smaller than the number of mutually co-occurring
        tracks, which would force the algorithm to merge faces that are visible
        in the same frame.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(embeddings)
    if n == 0:
        return [], []
    if n == 1:
        return [0], [float("inf")]

    known = [i for i, e in enumerate(embeddings) if e is not None]
    if not known:
        return list(range(n)), [float("inf")] * n

    mat = np.stack([np.asarray(embeddings[i], dtype=np.float64) for i in known])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1.0, norms)
    dist = 1.0 - mat @ mat.T
    np.clip(dist, 0.0, 2.0, out=dist)
    np.fill_diagonal(dist, 0.0)

    pos = {t: k for k, t in enumerate(known)}
    blocked = [(pos[i], pos[j]) for i, j in cannot_link
               if i in pos and j in pos]
    for a, b in blocked:
        dist[a, b] = dist[b, a] = 1e6

    if n_faces is not None:
        floor = _largest_mutually_exclusive_set(blocked, len(known))
        if n_faces < floor:
            raise IdentityConflict(
                f"n_faces={n_faces}, but {floor} faces appear together in a "
                "single frame, so there are at least that many people. Raise "
                "n_faces, or leave it out and set a threshold instead.")
        k = max(1, min(int(n_faces), len(known)))
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                        linkage="average")
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(threshold),
            metric="precomputed", linkage="average")

    sub = list(model.fit_predict(dist))

    labels = [-1] * n
    for t, lab in zip(known, sub):
        labels[t] = int(lab)
    spare = (max(sub) + 1) if sub else 0
    for i in range(n):
        if labels[i] == -1:          # no embedding: its own identity
            labels[i] = spare
            spare += 1

    margins = _margins(dist, known, sub, n)
    return labels, margins


def _largest_mutually_exclusive_set(blocked: Sequence[Tuple[int, int]],
                                    n: int) -> int:
    """
    How many tracks are pairwise co-occurring -- a clique in the cannot-link
    graph, and therefore a hard floor on the number of people.

    Greedy rather than exact: finding a maximum clique is NP-hard, and being
    conservative here only means we fail to catch some bad `n_faces` values,
    never that we reject a good one.
    """
    if not blocked:
        return 1
    adj: Dict[int, Set[int]] = {}
    for a, b in blocked:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    best = 1
    for start in sorted(adj, key=lambda x: -len(adj[x])):
        clique = {start}
        for other in sorted(adj[start], key=lambda x: -len(adj.get(x, ()))):
            if all(other in adj.get(m, set()) for m in clique):
                clique.add(other)
        best = max(best, len(clique))
    return best


def _margins(dist: np.ndarray, known: Sequence[int], sub: Sequence[int],
             n: int) -> List[float]:
    """Distance from each track to the nearest track of another identity."""
    out = [float("inf")] * n
    for a, track_a in enumerate(known):
        others = [dist[a, b] for b, _tb in enumerate(known)
                  if sub[b] != sub[a] and dist[a, b] < 1e5]
        out[track_a] = float(min(others)) if others else float("inf")
    return out
