"""Hand / fingertip localisation.

Two backends behind one function:

* **dnn**: OpenCV 5 DNN running the MediaPipe palm detector + hand-landmark ONNX
  models from opencv_zoo (Apache-2.0; fetch with `python scripts/fetch_models.py`).
  The palm detector proposes a rotated hand box and the landmark model returns 21
  keypoints. Each finger's TIP->DIP segment gives tip, axis and phalanx length.
* **classical**: FingerSafe-style skin/foreground mask -> largest contour ->
  convex hull + convexity defects, with a distance-transform palm centre. A single
  close-up finger (fingerphoto datasets) is handled by the same path.

`backend="auto"` uses the DNN when the models are present and falls back to classical.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gattaca.debug import DebugSink
from gattaca.pipeline.roi import finger_mask
from gattaca.types import FingertipDetection

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
PALM_MODEL = "palm_detection_mediapipe_2023feb.onnx"
HAND_MODEL = "handpose_estimation_mediapipe_2023feb.onnx"
# MediaPipe hand-landmark indices: (tip, dip) per finger
FINGERS = {"thumb": (4, 3), "index": (8, 7), "middle": (12, 11), "ring": (16, 15), "pinky": (20, 19)}


# --------------------------------------------------------------------------- classical
def _width_along_axis(dt: np.ndarray, tip: np.ndarray, axis: np.ndarray) -> float:
    """Finger width: grow along the axis until the walked length exceeds ~1.2 widths."""
    h, w = dt.shape
    best = 0.0
    for t in range(0, 2000, 2):
        x, y = (tip + axis * t).astype(int)
        if not (0 <= x < w and 0 <= y < h):
            break
        best = max(best, float(dt[y, x]))
        if t > 1.2 * 2 * best and best > 3:
            break
    return 2 * best


def _local_axis(mask: np.ndarray, tip: np.ndarray, radius: float) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    d = np.hypot(xs - tip[0], ys - tip[1])
    sel = d < radius
    if sel.sum() < 10:
        return np.array([0.0, 1.0])
    v = np.array([xs[sel].mean() - tip[0], ys[sel].mean() - tip[1]])
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 1.0])


def _touches_border(pt, shape, tol=3):
    h, w = shape[:2]
    return pt[0] <= tol or pt[1] <= tol or pt[0] >= w - 1 - tol or pt[1] >= h - 1 - tol


def detect_classical(img: np.ndarray, debug: DebugSink | None = None) -> list[FingertipDetection]:
    mask = finger_mask(img, debug=debug)
    if not mask.any():
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    palm_r = float(dt.max())
    py, px = np.unravel_index(int(dt.argmax()), dt.shape)
    palm = np.array([px, py], np.float64)
    hull_idx = cv2.convexHull(cnt, returnPoints=False).reshape(-1)
    pts = cnt.reshape(-1, 2).astype(np.float64)
    defects = cv2.convexityDefects(cnt, hull_idx) if len(hull_idx) > 3 else None
    # OpenCV 4.x returns (N,1,4), 5.x (N,4): normalise
    depth = np.zeros(0) if defects is None else np.asarray(defects).reshape(-1, 4)[:, 3] / 256.0
    deep = int((depth > 0.6 * palm_r).sum())
    cands: list[np.ndarray] = []
    if deep >= 1:  # hand: several fingers separated by deep defects
        hull_pts = pts[hull_idx]
        far = [p for p in hull_pts if np.linalg.norm(p - palm) > 1.6 * palm_r
               and not _touches_border(p, mask.shape)]
        far.sort(key=lambda p: -np.linalg.norm(p - palm))
        for p in far:
            if all(np.linalg.norm(p - q) > 0.8 * palm_r for q in cands):
                cands.append(p)
        cands = cands[:5]
    if not cands:  # single finger: tip = the axis extreme that does not touch the border
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        proj = (pts - mean) @ vt[0]
        ends = [pts[proj.argmin()], pts[proj.argmax()]]
        free = [e for e in ends if not _touches_border(e, mask.shape)]
        cands = [min(free or ends, key=lambda e: e[1])]
    dets = []
    for tip in cands:
        axis = _local_axis(mask, tip, 1.2 * palm_r if deep else 2.0 * palm_r)
        width = _width_along_axis(dt, tip + axis * 2, axis)
        if width < 8:
            continue
        axis = _local_axis(mask, tip, 1.2 * width)
        score = float(np.clip(np.linalg.norm(tip - palm) / (3 * palm_r), 0.3, 1.0))
        dets.append(FingertipDetection(tuple(tip), tuple(axis), width, score, mask, "classical"))
    return dets


# --------------------------------------------------------------------------- DNN
def _palm_anchors() -> np.ndarray:
    """SSD anchors of the 192x192 MediaPipe palm detector: 24x24x2 + 12x12x6 = 2016."""
    out = []
    for grid, per in ((24, 2), (12, 6)):
        for y in range(grid):
            for x in range(grid):
                out += [((x + 0.5) / grid, (y + 0.5) / grid)] * per
    return np.asarray(out, np.float32)


class DnnHandLandmarker:
    def __init__(self, model_dir: Path = MODEL_DIR, score_thr: float = 0.5, conf_thr: float = 0.6):
        self.palm = cv2.dnn.readNetFromONNX(str(model_dir / PALM_MODEL))
        self.hand = cv2.dnn.readNetFromONNX(str(model_dir / HAND_MODEL))
        self.anchors = _palm_anchors()
        self.score_thr, self.conf_thr = score_thr, conf_thr

    @staticmethod
    def available(model_dir: Path = MODEL_DIR) -> bool:
        return (model_dir / PALM_MODEL).exists() and (model_dir / HAND_MODEL).exists()

    def palms(self, bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
        h, w = bgr.shape[:2]
        s = 192 / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        top, left = (192 - nh) // 2, (192 - nw) // 2
        img = cv2.copyMakeBorder(cv2.resize(bgr, (nw, nh)), top, 192 - nh - top, left,
                                 192 - nw - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self.palm.setInput(blob)
        reg, score = self.palm.forward(self.palm.getUnconnectedOutLayersNames())
        score = 1 / (1 + np.exp(-score[0, :, 0].astype(np.float64)))
        reg = reg[0]
        c = reg[:, :2] / 192 + self.anchors
        wh = reg[:, 2:4] / 192
        boxes = np.concatenate([c - wh / 2, c + wh / 2], axis=1) * 192
        lms = (reg[:, 4:].reshape(-1, 7, 2) / 192 + self.anchors[:, None, :]) * 192
        undo = lambda p: (p - np.array([left, top])) / s  # noqa: E731
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        keep = cv2.dnn.NMSBoxes(xywh, score.tolist(), self.score_thr, 0.3)
        out = []
        for i in np.array(keep).reshape(-1):
            out.append((undo(boxes[i].reshape(2, 2)), undo(lms[i]), float(score[i])))
        return out

    def landmarks(self, bgr: np.ndarray, palm_box: np.ndarray, palm_lms: np.ndarray):
        """21 hand landmarks (source px) for one palm, or None below confidence."""
        wrist, middle = palm_lms[0], palm_lms[2]
        v = middle - wrist
        rot = np.degrees(np.arctan2(v[1], v[0])) + 90.0        # rotate so wrist->middle points up
        centre = palm_box.mean(axis=0)
        R = cv2.getRotationMatrix2D(tuple(centre), rot, 1.0)
        rl = (np.c_[palm_lms, np.ones(7)] @ R.T)
        lo, hi = rl.min(axis=0), rl.max(axis=0)
        wh = hi - lo
        c = (lo + hi) / 2 + np.array([0.0, -0.4]) * wh          # shift toward the fingers
        side = float(wh.max() * 3.0)
        S = 224.0 / side
        A = np.array([[S, 0, 112 - S * c[0]], [0, S, 112 - S * c[1]], [0, 0, 1]]) @ np.vstack([R, [0, 0, 1]])
        crop = cv2.warpAffine(bgr, A[:2], (224, 224), flags=cv2.INTER_AREA)
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self.hand.setInput(blob)
        lm, conf, _hand, _world = self.hand.forward(self.hand.getUnconnectedOutLayersNames())
        if float(conf.ravel()[0]) < self.conf_thr:
            return None
        pts = lm.reshape(21, 3)[:, :2]
        inv = np.linalg.inv(A)
        return (np.c_[pts, np.ones(21)] @ inv.T)[:, :2], float(conf.ravel()[0])

    def detect(self, bgr: np.ndarray) -> list[FingertipDetection]:
        mask = finger_mask(bgr)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5) if mask.any() else None
        dets = []
        for box, plms, pscore in self.palms(bgr):
            res = self.landmarks(bgr, box, plms)
            if res is None:
                continue
            lms, conf = res
            for name, (ti, di) in FINGERS.items():
                tip, dip = lms[ti], lms[di]
                seg = dip - tip
                length = float(np.linalg.norm(seg))
                if length < 6:
                    continue
                axis = seg / length
                width = 0.8 * length
                if dt is not None:  # refine width from the skin mask at mid-phalanx
                    mx, my = (tip + 0.5 * seg).astype(int)
                    if 0 <= my < dt.shape[0] and 0 <= mx < dt.shape[1] and dt[my, mx] > 0.25 * length:
                        width = float(2 * dt[my, mx])
                dets.append(FingertipDetection(tuple(tip), tuple(axis), width,
                                               pscore * conf, mask, f"dnn:{name}"))
        return dets


_dnn: DnnHandLandmarker | None = None


def detect_fingertips(img: np.ndarray, backend: str = "auto",
                      debug: DebugSink | None = None) -> list[FingertipDetection]:
    """Fingertip candidates, best first."""
    global _dnn
    dets: list[FingertipDetection] = []
    if backend in ("auto", "dnn") and img.ndim == 3 and DnnHandLandmarker.available():
        _dnn = _dnn or DnnHandLandmarker()
        dets = _dnn.detect(img)
    if not dets and backend in ("auto", "classical"):
        dets = detect_classical(img, debug=debug)
    if debug is not None and dets:
        vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for d in dets:
            t = np.array(d.tip)
            cv2.circle(vis, tuple(t.astype(int)), int(d.width / 2), (0, 255, 255), 2)
            cv2.arrowedLine(vis, tuple(t.astype(int)), tuple((t + np.array(d.axis) * d.width).astype(int)),
                            (0, 0, 255), 2)
        debug.save("detect_fingertips", vis)
    return sorted(dets, key=lambda d: -d.score)
