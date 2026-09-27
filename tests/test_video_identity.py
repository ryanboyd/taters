"""
Tests for `taters.video._identity`.

This is the part of video analysis with real error in it. A mis-linked track
silently merges two people's measurements into one row, and nothing in the
output looks wrong afterwards -- the numbers are all plausible, they are just
somebody else's.

So every test here has an answer known by construction: tracks built from a
fixed number of seeded identities must come back as that many, faces that share
a frame must never merge, and a threshold that is obviously too loose must be
caught by the constraint rather than by luck.

No models are involved. Embeddings are seeded vectors, which is what makes the
right answer knowable.
"""
from __future__ import annotations

import numpy as np
import pytest

from taters.video._identity import (Detection, IdentityConflict, Track,
                                    build_tracks, cluster_tracks,
                                    cooccurring_pairs, iou,
                                    max_simultaneous_faces, track_embedding)


def _person(seed: int, dim: int = 128) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim)
    return v / np.linalg.norm(v)


def _jitter(base: np.ndarray, seed: int, scale: float = 0.03) -> np.ndarray:
    v = base + np.random.default_rng(seed).normal(scale=scale, size=base.shape)
    return v / np.linalg.norm(v)


def _det(frame, x, y=0.0, w=50.0, h=50.0, emb=None, conf=1.0, sharp=1.0):
    return Detection(frame_index=frame, time_s=frame / 10.0, box=(x, y, w, h),
                     confidence=conf, embedding=emb, sharpness=sharp)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_iou_of_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_of_touching_boxes_is_zero_not_negative():
    """Edges that meet do not overlap, and a negative area would poison the
    tracker's scoring."""
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def test_a_face_that_stays_put_becomes_one_track():
    dets = [_det(f, x=10.0) for f in range(5)]

    tracks = build_tracks(dets)

    assert len(tracks) == 1
    assert len(tracks[0]) == 5


def test_two_faces_side_by_side_become_two_tracks():
    dets = [_det(f, x=0.0) for f in range(5)] + [_det(f, x=200.0) for f in range(5)]

    tracks = build_tracks(dets)

    assert len(tracks) == 2
    assert all(len(t) == 5 for t in tracks)


def test_a_track_never_crosses_a_cut():
    """
    The whole reason shot boundaries are threaded in. A cut to a new angle looks
    like smooth motion to a box tracker, so without this a two-shot interview
    becomes one person.
    """
    dets = [_det(f, x=10.0) for f in range(10)]

    without = build_tracks(dets)
    with_cut = build_tracks(dets, shot_boundaries=[5])

    assert len(without) == 1
    assert len(with_cut) == 2
    assert {t.start_frame for t in with_cut} == {0, 5}


def test_a_short_gap_is_bridged_but_a_long_one_is_not():
    """A blink or a turn costs a frame or two and should not end a track."""
    dets = [_det(0, 10.0), _det(1, 10.0), _det(4, 10.0)]

    bridged = build_tracks(dets, max_gap_frames=3)
    broken = build_tracks(dets, max_gap_frames=1)

    assert len(bridged) == 1
    assert len(broken) == 2


def test_a_face_that_moves_too_far_starts_a_new_track():
    dets = [_det(0, 0.0), _det(1, 500.0)]

    assert len(build_tracks(dets, min_iou=0.3)) == 2


def test_one_frame_tracks_can_be_dropped():
    """Usually a false positive, and always a bad crop to judge identity from."""
    dets = [_det(f, 10.0) for f in range(4)] + [_det(9, 300.0)]

    assert len(build_tracks(dets, min_length=1)) == 2
    assert len(build_tracks(dets, min_length=2)) == 1


# ---------------------------------------------------------------------------
# Track embeddings
# ---------------------------------------------------------------------------

def test_a_track_embedding_is_unit_length():
    a = _person(1)
    tr = Track([_det(f, 0.0, emb=_jitter(a, f)) for f in range(4)])

    assert np.linalg.norm(track_embedding(tr)) == pytest.approx(1.0)


def test_a_blurry_tiny_face_does_not_decide_who_the_person_is():
    """
    The quality weighting, stated as a test. One big sharp confident frame of
    person A and one tiny blurred frame of person B: the track should look like
    A, because B is the frame you would throw away if you were doing this by
    hand.
    """
    a, b = _person(1), _person(2)
    tr = Track([
        _det(0, 0.0, w=200, h=200, emb=a, conf=0.99, sharp=1.0),
        _det(1, 0.0, w=10, h=10, emb=b, conf=0.3, sharp=0.1),
    ])

    emb = track_embedding(tr)

    assert float(emb @ a) > 0.95
    assert float(emb @ a) > float(emb @ b)


def test_a_track_with_no_embeddings_has_none():
    assert track_embedding(Track([_det(0, 0.0)])) is None


# ---------------------------------------------------------------------------
# Co-occurrence: the constraint that is certain rather than probabilistic
# ---------------------------------------------------------------------------

def test_tracks_sharing_a_frame_are_flagged_as_different_people():
    t0 = Track([_det(0, 0.0), _det(1, 0.0)])
    t1 = Track([_det(1, 200.0), _det(2, 200.0)])
    t2 = Track([_det(9, 0.0)])

    assert cooccurring_pairs([t0, t1, t2]) == {(0, 1)}


def test_max_simultaneous_faces_is_the_floor_on_how_many_people_there_are():
    dets = [_det(0, 0.0), _det(0, 100.0), _det(0, 200.0), _det(5, 0.0)]

    assert max_simultaneous_faces(dets) == 3


def test_no_faces_means_no_floor():
    assert max_simultaneous_faces([]) == 0


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_six_tracks_of_three_people_resolve_to_three_identities():
    people = [_person(i) for i in range(3)]
    embs = [_jitter(people[i // 2], seed=100 + i) for i in range(6)]

    labels, _margins = cluster_tracks(embs, threshold=0.5)

    assert len(set(labels)) == 3
    assert labels[0] == labels[1] and labels[2] == labels[3]


def test_the_count_can_be_given_instead_of_a_threshold():
    people = [_person(i) for i in range(3)]
    embs = [_jitter(people[i // 2], seed=200 + i) for i in range(6)]

    labels, _ = cluster_tracks(embs, n_faces=3)

    assert len(set(labels)) == 3


def test_co_occurring_tracks_never_merge_however_alike_they_look():
    """
    Two tracks with *identical* embeddings that appear in the same frame are
    still two people. This is the case a similarity threshold gets wrong every
    time, and the constraint that fixes it.
    """
    same = _person(7)
    labels, _ = cluster_tracks([same, same.copy()],
                               cannot_link={(0, 1)}, threshold=0.9)

    assert labels[0] != labels[1]


def test_asking_for_fewer_people_than_appear_together_is_refused():
    """
    The trap: with a small enough `n_faces`, agglomerative clustering will
    happily merge faces visible in the same frame. Better to refuse than to
    publish that.
    """
    people = [_person(i) for i in range(3)]

    with pytest.raises(IdentityConflict, match="appear together"):
        cluster_tracks(people, cannot_link={(0, 1), (0, 2), (1, 2)}, n_faces=2)


def test_a_track_we_could_not_embed_gets_its_own_identity():
    """Folding an unknown face into somebody is worse than admitting it is
    unknown."""
    a = _person(1)
    labels, _ = cluster_tracks([a, _jitter(a, 5), None], threshold=0.5)

    assert labels[0] == labels[1]
    assert labels[2] not in (labels[0],)


def test_the_margin_says_how_close_a_call_each_identity_was():
    """
    The number to look at before believing a per-person result. Two well
    separated people have a big margin; two similar ones do not.
    """
    # two independent people sit ~0.87 apart in cosine distance; a lightly
    # jittered copy of one person sits ~0.06 away. Thresholds below each force
    # both pairs to split, so the margin is the distance itself.
    far = [_person(1), _person(2)]
    near = [_person(3), _jitter(_person(3), 9, scale=0.03)]

    _, far_margins = cluster_tracks(far, threshold=0.5)
    _, near_margins = cluster_tracks(near, threshold=0.01)

    assert min(near_margins) < 0.2, "a near-identical pair is a close call"
    assert min(far_margins) > 0.5, "two different people are not"
    assert min(far_margins) > min(near_margins)


def test_no_tracks_is_not_an_error():
    assert cluster_tracks([]) == ([], [])


def test_one_track_is_one_identity():
    labels, margins = cluster_tracks([_person(1)])

    assert labels == [0]
    assert margins == [float("inf")]


# ---------------------------------------------------------------------------
# The whole chain, on a constructed two-person scene
# ---------------------------------------------------------------------------

def test_a_two_person_scene_with_a_cut_resolves_to_two_people():
    """
    Two people, each appearing in both halves of a video with a cut in the
    middle, and both on screen together throughout. Four tracks, two identities,
    and the co-occurrence constraint is what stops it being one.
    """
    a, b = _person(11), _person(12)
    dets = []
    for f in range(10):
        dets.append(_det(f, x=0.0, emb=_jitter(a, seed=f)))
        dets.append(_det(f, x=300.0, emb=_jitter(b, seed=100 + f)))

    tracks = build_tracks(dets, shot_boundaries=[5])
    assert len(tracks) == 4, "two people, two shots"

    embs = [track_embedding(t) for t in tracks]
    labels, margins = cluster_tracks(embs, cannot_link=cooccurring_pairs(tracks),
                                     threshold=0.5)

    assert len(set(labels)) == 2
    assert all(m > 0.5 for m in margins), "two different people, comfortably"
