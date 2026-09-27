"""
The three small ONNX models that do the looking.

All of this runs on ``onnxruntime``, which is already installed -- it arrives
with the audio stack -- so none of it costs a dependency. The models are fetched
from Hugging Face on first use and cached, exactly as the Whisper and
sentence-transformer models already are.

========================  ==================================  ==============
what                      model                               license
========================  ==================================  ==============
finding faces             YuNet (OpenCV Zoo)                  MIT
who a face is             SFace (OpenCV Zoo)                  Apache-2.0
what a face is doing      HSEmotion / EmotiEffLib             Apache-2.0
========================  ==================================  ==============

All three are usable commercially, which is the reason they were chosen over
better-known alternatives. OpenFace and Py-Feat's multitask model are both
"non-commercial research only", and shipping a step that quietly puts a user in
breach of a license is worse than not shipping it.

The awkward part is YuNet's output: it does not hand back boxes, it hands back
raw classification, objectness, box and keypoint heads at three strides, and
turning those into faces is anchor decoding plus non-maximum suppression. OpenCV
does that for you in C++, which is no use without OpenCV, so it is done here --
as a pure function over arrays, so it can be tested against known geometry
rather than only against whether a photo comes out looking right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FaceBox", "decode_yunet", "nms", "YuNet", "SFace", "Emotion",
           "MODELS", "EMOTION_LABELS", "load_session", "align_face",
           "softmax"]

#: Repo and filename on the Hugging Face hub, per model. Kept in one place so
#: that "where did this number come from" has a single answer.
MODELS: Dict[str, Tuple[str, str]] = {
    "detect": ("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx"),
    "identity": ("opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx"),
}

#: HSEmotion publishes its ONNX weights in its own repository rather than on
#: the hub, so this one is fetched by URL and cached beside the library. Same
#: Apache-2.0 terms; just a different address.
EMOTION_URL = ("https://raw.githubusercontent.com/HSE-asavchenko/"
               "face-emotion-recognition/main/models/affectnet_emotions/onnx/"
               "enet_b0_8_best_vgaf.onnx")

#: The eight AffectNet categories this model was trained on, in output order.
EMOTION_LABELS = ("anger", "contempt", "disgust", "fear", "happiness",
                  "neutral", "sadness", "surprise")

#: What the emotion model's backbone was trained with. Getting these wrong
#: gives confident, wrong answers rather than an error, which is the whole
#: reason they are written down here next to the model that needs them.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#: YuNet predicts at these strides, each over its own grid of anchors.
_STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class FaceBox:
    """One detected face in image coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: Tuple[Tuple[float, float], ...] = ()

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


# ---------------------------------------------------------------------------
# Decoding, as pure functions
# ---------------------------------------------------------------------------

def nms(boxes: Sequence[FaceBox], iou_threshold: float = 0.3) -> List[FaceBox]:
    """
    Keep the strongest of each overlapping group.

    A detector fires several times on one face -- neighboring anchors all see
    it -- so without this a single person is three or four detections and every
    per-frame face count is wrong.
    """
    from ._identity import iou

    kept: List[FaceBox] = []
    for cand in sorted(boxes, key=lambda b: -b.score):
        if all(iou(cand.box, k.box) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


def decode_yunet(outputs: Dict[str, np.ndarray], *,
                 input_size: Tuple[int, int],
                 scale_x: float = 1.0, scale_y: float = 1.0,
                 score_threshold: float = 0.6,
                 iou_threshold: float = 0.3) -> List[FaceBox]:
    """
    Turn YuNet's raw heads into faces.

    Parameters
    ----------
    outputs : dict of str to ndarray
        The session outputs, keyed as the model names them: ``cls_8``,
        ``obj_8``, ``bbox_8``, ``kps_8`` and the same at strides 16 and 32.
    input_size : (int, int)
        ``(width, height)`` the model was fed, which sets the anchor grid.
    scale_x, scale_y : float
        Multipliers back to original-image coordinates, for when the frame was
        resized to fit the model.
    score_threshold : float, default=0.6
        Minimum confidence. Classification and objectness are combined as their
        geometric mean, which is what YuNet's own postprocessing does.
    iou_threshold : float, default=0.3
        Overlap above which two detections are the same face.

    Returns
    -------
    list of FaceBox
        Strongest first.

    Notes
    -----
    The box head is an offset from its anchor's cell, in cell units, with the
    size log-encoded: ``cx = (col + dx) * stride`` and ``w = exp(dw) * stride``.
    Getting that wrong produces boxes that look plausible and sit in the wrong
    place, which is why it is tested against constructed inputs rather than
    only against a photograph.
    """
    in_w, in_h = input_size
    found: List[FaceBox] = []

    for stride in _STRIDES:
        cls = np.asarray(outputs[f"cls_{stride}"]).reshape(-1)
        obj = np.asarray(outputs[f"obj_{stride}"]).reshape(-1)
        bbox = np.asarray(outputs[f"bbox_{stride}"]).reshape(-1, 4)
        kps = np.asarray(outputs[f"kps_{stride}"]).reshape(-1, 10)

        cols = int(math.ceil(in_w / stride))
        rows = int(math.ceil(in_h / stride))
        n = min(len(cls), rows * cols)

        scores = np.sqrt(np.clip(cls[:n], 0, 1) * np.clip(obj[:n], 0, 1))
        for idx in np.nonzero(scores >= score_threshold)[0]:
            col, row = int(idx) % cols, int(idx) // cols
            dx, dy, dw, dh = bbox[idx]

            cx = (col + float(dx)) * stride
            cy = (row + float(dy)) * stride
            w = math.exp(float(dw)) * stride
            h = math.exp(float(dh)) * stride

            marks = tuple(
                (((col + float(kps[idx][2 * k])) * stride) * scale_x,
                 ((row + float(kps[idx][2 * k + 1])) * stride) * scale_y)
                for k in range(5))

            found.append(FaceBox(
                x=(cx - w / 2.0) * scale_x, y=(cy - h / 2.0) * scale_y,
                w=w * scale_x, h=h * scale_y,
                score=float(scores[idx]), landmarks=marks))

    return nms(found, iou_threshold=iou_threshold)


def align_face(image: np.ndarray, landmarks: Sequence[Tuple[float, float]],
               size: int = 112) -> np.ndarray:
    """
    Crop and straighten a face to the square a recognition model expects.

    Uses the two eyes to set rotation and scale. A recognition embedding is only
    comparable with another if both faces were aligned the same way -- feeding
    raw crops of a tilted head and a straight one makes the same person look
    like two, which is exactly the error this whole module is trying to avoid.
    """
    from PIL import Image

    (lx, ly), (rx, ry) = landmarks[0], landmarks[1]
    dx, dy = rx - lx, ry - ly
    eye_gap = math.hypot(dx, dy) or 1.0
    angle = math.degrees(math.atan2(dy, dx))

    cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
    # eyes about a third of the way down a crop about 2.6 eye-gaps wide, which
    # is roughly how the recognition training crops were framed
    half = eye_gap * 1.3
    pil = Image.fromarray(image).rotate(
        angle, center=(cx, cy), resample=Image.BILINEAR)
    box = (cx - half, cy - half * 0.75, cx + half, cy + half * 1.25)
    return np.asarray(pil.crop(box).resize((size, size), Image.BILINEAR))


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def load_session(which: str, *, providers: Optional[Sequence[str]] = None):
    """
    Fetch a model if needed and open an inference session.

    Raises a plain, actionable error when the model cannot be fetched -- this is
    the first thing that fails on a machine with no network, and "check your
    connection" beats a stack trace out of huggingface_hub.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    repo, filename = MODELS[which]
    try:
        path = hf_hub_download(repo_id=repo, filename=filename)
    except Exception as exc:                      # pragma: no cover - network
        raise RuntimeError(
            f"Could not fetch the {which} model ({repo}/{filename}). It is "
            "downloaded once and then cached, so this needs a working "
            f"connection the first time. Original error: {exc}") from exc

    ort.set_default_logger_severity(3)            # these models warn a lot
    return ort.InferenceSession(
        str(path), providers=list(providers or ["CPUExecutionProvider"]))


class YuNet:
    """Face detection. Boxes, a score, and five landmarks per face."""

    def __init__(self, *, score_threshold: float = 0.6,
                 iou_threshold: float = 0.3,
                 input_size: Optional[int] = None):
        self.session = load_session("detect")
        spec = self.session.get_inputs()[0]
        self.name = spec.name
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        # the published YuNet is exported at a fixed 640x640; later ones are
        # dynamic. Ask the model rather than assume, because feeding the wrong
        # size to a fixed export fails loudly and feeding it to a dynamic one
        # silently changes the anchor grid.
        declared = spec.shape[-1] if len(spec.shape) == 4 else None
        self.input_size = int(input_size or
                              (declared if isinstance(declared, int) else 640))

    def detect(self, image: np.ndarray) -> List[FaceBox]:
        """Find faces in an RGB uint8 array."""
        from PIL import Image

        h, w = image.shape[:2]
        side = self.input_size
        resized = np.asarray(
            Image.fromarray(image).resize((side, side), Image.BILINEAR))
        # the model wants BGR planes, which is what it was trained on
        blob = resized[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)

        outs = self.session.run(None, {self.name: blob})
        named = {o.name: v for o, v in zip(self.session.get_outputs(), outs)}
        return decode_yunet(named, input_size=(side, side),
                            scale_x=w / side, scale_y=h / side,
                            score_threshold=self.score_threshold,
                            iou_threshold=self.iou_threshold)


class SFace:
    """Face identity. A 128-d embedding per aligned face."""

    def __init__(self):
        self.session = load_session("identity")
        self.name = self.session.get_inputs()[0].name

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        """Embed a 112x112 aligned RGB crop, L2-normalized."""
        blob = aligned[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        vec = np.asarray(self.session.run(None, {self.name: blob})[0]).reshape(-1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    total = e.sum()
    return e / total if total else np.full_like(e, 1.0 / e.size)


def _cached_url(url: str) -> Path:
    """
    Download once into the Taters home and reuse it forever after.

    Deliberately not the Hugging Face cache: this model does not live on the
    hub, and putting a stray file in somebody else's cache directory is how you
    get a confusing `huggingface-cli scan-cache`.
    """
    import os
    import urllib.request

    from filelock import FileLock

    base = os.environ.get("TATERS_HOME")
    root = (Path(base) if base else Path.home() / ".taters") / "models"
    root.mkdir(parents=True, exist_ok=True)
    dst = root / url.rsplit("/", 1)[-1]
    if dst.is_file() and dst.stat().st_size > 0:
        return dst

    # A pipeline run with --workers N starts N copies of this at once, and the
    # first version of this function let all of them download to the same
    # .partial file. On Linux that merely wasted bandwidth; on Windows an open
    # file is locked, so every worker but the winner died in the rename -- and
    # then died again in the cleanup, trying to delete a file another worker
    # still had open. Three of four videos in a real run failed that way.
    #
    # So: one lock per model, and whoever holds it does the download while the
    # rest wait and then find the file already there. filelock comes in with
    # huggingface_hub, which is how the hub models avoid this same problem.
    with FileLock(str(dst) + ".lock"):
        if dst.is_file() and dst.stat().st_size > 0:
            return dst                   # somebody else finished while we waited

        # per-process temp name as well, so that even a lock that failed to do
        # its job could never put two writers on one file
        tmp = dst.with_name(f"{dst.name}.partial.{os.getpid()}")
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)         # atomic; only a complete file gets the name
        except Exception as exc:         # pragma: no cover - network
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass                     # the download error is the one worth raising
            raise RuntimeError(
                f"Could not fetch the emotion model from {url}. It is "
                f"downloaded once and then cached. Original error: {exc}") from exc
    return dst


class Emotion:
    """Facial emotion, as eight AffectNet probabilities."""

    labels = EMOTION_LABELS

    def __init__(self):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(
            str(_cached_url(EMOTION_URL)), providers=["CPUExecutionProvider"])
        self.name = self.session.get_inputs()[0].name

    def predict(self, crop: np.ndarray) -> Dict[str, float]:
        """
        Score one RGB face crop.

        Wants the detector's plain box, not the eye-aligned square the identity
        model wants -- this backbone was trained on whole-face crops, and
        handing it a tight aligned chip changes the answer.
        """
        from PIL import Image

        arr = np.asarray(
            Image.fromarray(crop).resize((224, 224), Image.BILINEAR),
            dtype=np.float32) / 255.0
        arr = (arr - np.asarray(_IMAGENET_MEAN, dtype=np.float32)) \
            / np.asarray(_IMAGENET_STD, dtype=np.float32)
        blob = arr.transpose(2, 0, 1)[None]

        logits = np.asarray(
            self.session.run(None, {self.name: blob})[0]).reshape(-1)
        probs = softmax(logits)
        return {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}
