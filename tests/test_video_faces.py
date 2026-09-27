"""
Tests for `taters.video._onnx_faces` and `taters.video.analyze_faces`.

Split by what can be known without a model. YuNet's anchor decoding is
arithmetic -- an anchor at a known cell with a known offset lands at a known
pixel -- so it is tested against constructed arrays, which is the only way to
catch a decode that is off by half a cell and still produces boxes that look
about right on a photograph.

The tests that need the actual models are marked and skip when the weights are
not cached, so a machine with no network still runs the rest.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from taters.video._onnx_faces import (EMOTION_LABELS, FaceBox, decode_yunet,
                                      nms, softmax)
from taters.video.analyze_faces import (BOOKKEEPING, MEASURES, _roll_degrees,
                                        _sharpness, analyze_faces)


def _heads(n_per_stride, hits):
    """
    Build a full set of YuNet outputs with faces only where asked.

    `hits` maps (stride, flat anchor index) -> (cls, obj, box, kps).
    """
    out = {}
    for stride, n in n_per_stride.items():
        out[f"cls_{stride}"] = np.zeros((1, n, 1), dtype=np.float32)
        out[f"obj_{stride}"] = np.zeros((1, n, 1), dtype=np.float32)
        out[f"bbox_{stride}"] = np.zeros((1, n, 4), dtype=np.float32)
        out[f"kps_{stride}"] = np.zeros((1, n, 10), dtype=np.float32)
    for (stride, idx), (cls, obj, box, kps) in hits.items():
        out[f"cls_{stride}"][0, idx, 0] = cls
        out[f"obj_{stride}"][0, idx, 0] = obj
        out[f"bbox_{stride}"][0, idx] = box
        out[f"kps_{stride}"][0, idx] = kps
    return out


SIZES = {8: 6400, 16: 1600, 32: 400}


# ---------------------------------------------------------------------------
# Anchor decoding: the part that is arithmetic
# ---------------------------------------------------------------------------

def test_an_anchor_decodes_to_the_pixel_it_should():
    """
    Stride 8, 640-wide input, so 80 columns. Anchor 81 is row 1 column 1. With
    a zero offset and a log-size of 0 the box is one stride across, centered on
    that cell: center (8, 8), so the corner is (4, 4) and the size is 8.
    """
    outs = _heads(SIZES, {(8, 81): (1.0, 1.0, [0.0, 0.0, 0.0, 0.0], [0.0] * 10)})

    faces = decode_yunet(outs, input_size=(640, 640), score_threshold=0.5)

    assert len(faces) == 1
    f = faces[0]
    assert (f.x, f.y) == pytest.approx((4.0, 4.0))
    assert (f.w, f.h) == pytest.approx((8.0, 8.0))


def test_the_box_offset_moves_the_face_by_cells_not_pixels():
    """Half a cell at stride 8 is four pixels, not half a pixel. Getting this
    wrong puts every face slightly in the wrong place."""
    outs = _heads(SIZES, {(8, 81): (1.0, 1.0, [0.5, 0.0, 0.0, 0.0], [0.0] * 10)})

    f = decode_yunet(outs, input_size=(640, 640), score_threshold=0.5)[0]

    assert f.x == pytest.approx(8.0)          # center moved 4px, corner with it


def test_the_size_head_is_log_encoded():
    """`w = exp(dw) * stride`, so dw=ln(4) is four strides across."""
    outs = _heads(SIZES, {(8, 0): (1.0, 1.0, [0.0, 0.0, math.log(4), math.log(4)],
                                   [0.0] * 10)})

    f = decode_yunet(outs, input_size=(640, 640), score_threshold=0.5)[0]

    assert f.w == pytest.approx(32.0)
    assert f.h == pytest.approx(32.0)


def test_scaling_maps_back_to_the_original_image():
    """The frame is resized to feed the model, so the boxes have to come back."""
    outs = _heads(SIZES, {(8, 81): (1.0, 1.0, [0.0, 0.0, 0.0, 0.0], [0.0] * 10)})

    f = decode_yunet(outs, input_size=(640, 640), scale_x=2.0, scale_y=3.0,
                     score_threshold=0.5)[0]

    assert (f.x, f.y) == pytest.approx((8.0, 12.0))
    assert (f.w, f.h) == pytest.approx((16.0, 24.0))


def test_the_score_is_the_geometric_mean_of_the_two_heads():
    """Classification alone fires on things that are not faces; objectness
    alone fires on anything salient. YuNet combines them, and so do we."""
    outs = _heads(SIZES, {(8, 0): (0.64, 0.25, [0.0] * 4, [0.0] * 10)})

    f = decode_yunet(outs, input_size=(640, 640), score_threshold=0.1)[0]

    assert f.score == pytest.approx(0.4)      # sqrt(0.64 * 0.25)


def test_weak_detections_are_dropped():
    outs = _heads(SIZES, {(8, 0): (0.2, 0.2, [0.0] * 4, [0.0] * 10)})

    assert decode_yunet(outs, input_size=(640, 640), score_threshold=0.6) == []


def test_nothing_above_threshold_is_an_empty_list():
    assert decode_yunet(_heads(SIZES, {}), input_size=(640, 640)) == []


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def test_overlapping_detections_collapse_to_the_strongest():
    """
    A detector fires on several neighboring anchors for one face. Without
    suppression, one person is counted three times and every face count in the
    output is wrong.
    """
    boxes = [FaceBox(10, 10, 100, 100, 0.7), FaceBox(12, 12, 100, 100, 0.9),
             FaceBox(11, 11, 100, 100, 0.8)]

    kept = nms(boxes, iou_threshold=0.3)

    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_two_separate_faces_both_survive():
    boxes = [FaceBox(0, 0, 50, 50, 0.9), FaceBox(500, 500, 50, 50, 0.8)]

    assert len(nms(boxes, iou_threshold=0.3)) == 2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one_and_survives_big_logits():
    p = softmax(np.array([1000.0, 1000.0, 999.0]))

    assert p.sum() == pytest.approx(1.0)
    assert np.isfinite(p).all(), "the shift is what stops this overflowing"


def test_roll_is_zero_for_level_eyes_and_signed_for_a_tilt():
    assert _roll_degrees([(0, 0), (100, 0)]) == pytest.approx(0.0)
    assert _roll_degrees([(0, 0), (100, 100)]) == pytest.approx(45.0)
    assert _roll_degrees([(0, 100), (100, 0)]) == pytest.approx(-45.0)


def test_sharpness_separates_a_flat_patch_from_a_busy_one():
    """Only used to weight which frames decide an identity, so all it has to do
    is rank a blurred crop below a crisp one."""
    flat = np.full((32, 32, 3), 128, dtype=np.uint8)
    busy = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    busy = np.stack([busy] * 3, axis=-1)

    assert _sharpness(flat) == pytest.approx(0.0)
    assert _sharpness(busy) > _sharpness(flat)


def test_sharpness_of_an_empty_or_tiny_crop_is_zero_not_an_error():
    assert _sharpness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
    assert _sharpness(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0


# ---------------------------------------------------------------------------
# The output contract
# ---------------------------------------------------------------------------

def test_coverage_is_bookkeeping_and_never_a_measure():
    """
    A mean emotion over 11% of frames is a different quantity from one over
    95%, so the count rides along -- and a model must never be allowed to fit
    "how many frames had a face in them", which is a fact about the camera.
    """
    assert "face_coverage" in BOOKKEEPING
    assert "face_frames_with_face" in BOOKKEEPING
    assert not set(MEASURES) & set(BOOKKEEPING)
    assert set(BOOKKEEPING) == set(analyze_faces.__provenance__["bookkeeping"])


def test_every_emotion_gets_a_mean_and_a_spread():
    for label in EMOTION_LABELS:
        assert f"face_emo_{label}" in MEASURES
        assert f"face_emo_{label}_sd" in MEASURES


def test_the_eight_affectnet_categories_are_what_we_claim():
    assert set(EMOTION_LABELS) == {"anger", "contempt", "disgust", "fear",
                                   "happiness", "neutral", "sadness", "surprise"}


def test_a_missing_video_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_faces(video_path=tmp_path / "nope.mp4",
                      out_csv=tmp_path / "out.csv")


# ---------------------------------------------------------------------------
# The model download must survive parallel workers
# ---------------------------------------------------------------------------

def _fetch_in_child(args):
    """Runs in a separate process: fetch the URL into the given cache."""
    import os
    os.environ["TATERS_HOME"] = args[1]
    from taters.video._onnx_faces import _cached_url
    return str(_cached_url(args[0]))


def test_parallel_workers_share_one_download_instead_of_racing(tmp_path):
    """
    A pipeline run with --workers N starts N copies of the face step at once,
    and every one of them finds the emotion model uncached. The first version
    let them all download to the same .partial file: on Windows the losers
    died in the rename, then died again cleaning up a file another worker
    still had open. Three of four videos failed in a real run that way.

    Eight processes, one fresh cache, a local server counting requests. They
    must all come back with the same intact file, and the server must have
    been asked for it once -- the lock is what turns a race into a queue.
    """
    import http.server
    import multiprocessing
    import threading
    from pathlib import Path

    payload = b"not really a model, but 4096 bytes of one\n" * 100
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_a):     # keep the test output quiet
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/fake_model.onnx"
        cache = str(tmp_path / "home")
        with multiprocessing.get_context("spawn").Pool(8) as pool:
            results = pool.map(_fetch_in_child, [(url, cache)] * 8)
    finally:
        server.shutdown()

    assert len(set(results)) == 1, "every worker must land on the same file"
    got = Path(results[0])
    assert got.read_bytes() == payload, "and it must be intact"
    assert len(hits) == 1, f"downloaded {len(hits)} times; the lock should make it 1"
    leftovers = list(got.parent.glob("*.partial*"))
    assert not leftovers, f"temp files left behind: {leftovers}"
