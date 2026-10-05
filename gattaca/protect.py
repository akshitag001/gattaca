"""Stretch goal: one-click "protect my photo" (simplified, non-adversarial FingerSafe idea).

Inside each detected distal phalanx, the ridge frequency band is suppressed: a
low-pass at the measured ridge period removes the ridges, and smooth skin-like
low-frequency texture plus mild sensor-like grain are added back so the finger still
looks natural. The mask is feathered so no edge is visible. The same pipeline then
re-scores the result to show that recoverability really dropped.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca.analyze import RIDGES_ACROSS_FINGER
from gattaca.pipeline.detect import detect_fingertips
from gattaca.pipeline.roi import finger_mask


def fingertip_mask(img: np.ndarray, dets, length_ratio: float = 1.6, grow: int = 9) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros((h, w), np.uint8)
    fm = finger_mask(img)
    for d in dets:
        tip, a = np.array(d.tip), np.array(d.axis)
        perp = np.array([a[1], -a[0]])
        half = 0.75 * d.width
        L = length_ratio * d.width
        start = tip - a * 0.3 * d.width
        quad = np.array([start - perp * half, start + perp * half,
                         start + a * L + perp * half, start + a * L - perp * half])
        cv2.fillConvexPoly(out, quad.astype(np.int32), 255)
    if fm.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        out = cv2.bitwise_and(out, cv2.dilate(fm, k))
    return out


def protect_photo(img: np.ndarray, dets=None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (protected image, mask used)."""
    dets = dets if dets is not None else detect_fingertips(img)
    if not dets:
        return img.copy(), np.zeros(img.shape[:2], np.uint8)
    rng = np.random.default_rng(seed)
    period = max(np.median([d.width / RIDGES_ACROSS_FINGER for d in dets]), 1.0)
    feather_sigma = max(2.0, 1.5 * period)
    # grow past the silhouette so the feather fall-off lands outside the skin: a half-
    # strength edge would leave the (later unwrapped, i.e. magnified) edge ridges intact
    mask = fingertip_mask(img, dets, grow=int(3 * feather_sigma))
    f = img.astype(np.float32)
    low = cv2.GaussianBlur(f, (0, 0), 0.9 * period)                # removes the ridge band
    tex = cv2.GaussianBlur(rng.standard_normal(img.shape[:2]).astype(np.float32), (0, 0), 3 * period)
    tex = tex / (np.abs(tex).max() + 1e-6) * 6.0
    grain = rng.normal(0, 1.5, img.shape[:2]).astype(np.float32)
    add = (tex + grain)[..., None] if img.ndim == 3 else tex + grain
    prot = low + add
    feather = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather_sigma)
    feather = feather[..., None] if img.ndim == 3 else feather
    out = f * (1 - feather) + prot * feather
    return np.clip(out, 0, 255).astype(np.uint8), mask
