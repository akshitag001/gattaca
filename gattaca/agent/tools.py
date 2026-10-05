"""OpenCV stages exposed as agent tools. Each takes plain inputs and returns an
observation dict (JSON-safe) plus any heavy payload the loop keeps in its state."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from gattaca.analyze import analyze_photo
from gattaca.pipeline.detect import detect_fingertips
from gattaca.pipeline.enhance import EnhanceParams
from gattaca.scoring import Recoverability, score_finger
from gattaca.video.fuse import fuse
from gattaca.video.select import FrameScore, score_frame, select_best
from gattaca.video.track import track_box

RETUNE_PRESETS = [
    EnhanceParams(denoise="nlm", clahe_clip=3.0, gabor_sigma=0.6),
    EnhanceParams(denoise="none", clahe_clip=2.5, gabor_sigma=0.45, gabor_gamma=0.5, orient_sigma=9.0),
]


def finger_box(det, shape) -> tuple[float, float, float, float]:
    """Square box covering the distal phalanx of a detection."""
    tip, a = np.array(det.tip), np.array(det.axis)
    c = tip + a * 0.6 * det.width
    s = 1.7 * det.width
    x, y = c[0] - s / 2, c[1] - s / 2
    return (float(np.clip(x, 0, shape[1] - 2)), float(np.clip(y, 0, shape[0] - 2)), float(s), float(s))


def tool_detect(frame: np.ndarray):
    dets = detect_fingertips(frame)
    return {"fingers": len(dets), "labels": [d.source for d in dets]}, dets


def tool_track(frames, box, start):
    boxes, motion = track_box(frames, box, start)
    return {"tracked": sum(b is not None for b in boxes), "mean_motion_px": round(float(np.mean(motion)), 2)}, \
        (boxes, motion)


def tool_score_frames(frames, boxes, motion, indices) -> tuple[dict, list[FrameScore]]:
    scores = [score_frame(frames[i], boxes[i], i, float(motion[i])) for i in indices if boxes[i] is not None]
    best = max(scores, key=lambda s: s.score) if scores else None
    return {"scored": len(scores), "best_frame": best.index if best else None,
            "best_sharpness": round(best.sharpness, 3) if best else None}, scores


def tool_select(scores, k):
    idx = select_best(scores, k=k)
    return {"selected": idx}, idx


def _assess(img, params: EnhanceParams, finger: int = 0, pprp_override: float | None = None):
    fingers = analyze_photo(img, params=params, max_fingers=5)
    if not fingers:
        return {"fingers": 0, "band": None}, None, None
    fa = fingers[min(finger, len(fingers) - 1)]
    if pprp_override is not None:
        # fused images are upsampled: score with the *capture* PPRP so the resolution
        # feature is not inflated; the gain must show up in clarity / minutiae instead
        fa.analysis.quality = dataclasses.replace(fa.analysis.quality, pprp=pprp_override)
    rec: Recoverability = score_finger(fa)
    obs = {"fingers": len(fingers), "band": rec.band, "probability": round(rec.probability, 3),
           "confidence": round(rec.confidence, 3), "limiting_factor": rec.limiting_factor,
           "pprp": round(rec.pprp, 2), "clarity": round(fa.analysis.quality.clarity, 3),
           "minutiae_count": fa.analysis.quality.minutiae_count}
    return obs, rec, fa


def tool_assess_frame(frame, params=EnhanceParams(), finger=0):
    return _assess(frame, params, finger)


def tool_fuse(frames, boxes, indices, native_pprp: float, params=EnhanceParams()):
    ref = indices[0]
    x, y, w, h = boxes[ref]
    size = (int(w), int(h))
    crops = []
    for i in indices:
        bx, by, bw, bh = boxes[i]
        crops.append(cv2.getRectSubPix(frames[i], size, (bx + bw / 2, by + bh / 2)))
    fused, info = fuse(crops, 0, scale=2)
    obs, rec, fa = _assess(fused, params, 0, pprp_override=native_pprp)
    obs.update({"frames_fused": info["frames_used"]})
    return obs, rec, fa, fused
