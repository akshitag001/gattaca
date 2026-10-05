"""Per-frame quality scoring and best-K selection (cf. IIITD multi-resolution selection).

Cheap per-frame proxies, computed only inside the tracked fingertip box:
* sharpness: variance of the Laplacian, normalised by local contrast
* ridge-band energy: fraction of high-pass spectral energy lying in the ridge-frequency
  ring, over ALL orientations. (Angular peakiness is NOT used here: motion blur
  leaves only ridges parallel to the motion, a single strong orientation, which
  would make the blurriest frames look best.)
* exposure: penalises dark or clipped crops
* motion: tracked inter-frame displacement (a proxy for motion blur)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gattaca.pipeline.ridgefreq import dominant_period


def ridge_band_fraction(g: np.ndarray, period: float) -> float:
    """Share of spectral energy (above 0.3x the ridge frequency) inside the ridge ring."""
    if not np.isfinite(period) or period <= 0 or min(g.shape) < 16:
        return 0.0
    f = g.astype(np.float32)
    f = (f - cv2.GaussianBlur(f, (0, 0), 2 * period)) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fft2(f)) ** 2
    fy = np.fft.fftfreq(g.shape[0])[:, None]
    fx = np.fft.fftfreq(g.shape[1])[None, :]
    r = np.hypot(fx, fy)
    f0 = 1.0 / period
    ring = np.abs(r - f0) < 0.25 * f0
    hp = r > 0.3 * f0
    return float(P[ring].sum() / (P[hp].sum() + 1e-9))


@dataclass
class FrameScore:
    index: int
    sharpness: float
    ridge_band: float
    pprp: float
    exposure: float
    motion: float
    score: float


def crop(frame: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    x, y, w, h = box
    x0, y0 = int(max(0, x - pad * w)), int(max(0, y - pad * h))
    x1, y1 = int(min(frame.shape[1], x + w * (1 + pad))), int(min(frame.shape[0], y + h * (1 + pad)))
    return frame[y0:y1, x0:x1]


def score_frame(frame: np.ndarray, box, index: int = 0, motion: float = 0.0) -> FrameScore:
    c = crop(frame, box)
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    if g.size < 64:
        return FrameScore(index, 0, 0, float("nan"), 0, motion, -np.inf)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    sharp = float(lap.var() / (g.astype(np.float32).var() + 1.0))
    pprp, _, _ = dominant_period(g)
    band = ridge_band_fraction(g, pprp)
    mean = float(g.mean())
    clipped = float(((g < 8) | (g > 247)).mean())
    exposure = float(np.clip(1 - abs(mean - 128) / 128, 0, 1) * (1 - clipped))
    s = np.log1p(sharp) + 2.0 * band + 0.5 * exposure - 0.05 * motion
    return FrameScore(index, sharp, band, float(pprp), exposure, motion, float(s))


def select_best(scores: list[FrameScore], k: int = 5, min_gap: int = 1) -> list[int]:
    """Top-k frame indices, at least `min_gap` apart (temporal diversity -> distinct sub-pixel offsets)."""
    chosen: list[int] = []
    for fs in sorted(scores, key=lambda s: -s.score):
        if not np.isfinite(fs.score):
            continue
        if all(abs(fs.index - c) >= min_gap for c in chosen):
            chosen.append(fs.index)
        if len(chosen) == k:
            break
    return chosen
