"""Controlled degradation ladders for the measurement study.

Every axis is expressed in physically meaningful units relative to the image's own
measured ridge period, so ladders mean the same thing across datasets:

* pprp      target pixels-per-ridge-period (area-integrating downsample, like a sensor)
* gauss     defocus: Gaussian sigma in ridge periods
* motion    linear motion blur length in ridge periods (random direction, seeded)
* jpeg      JPEG quality (platform recompression)
* exposure  under-exposure in stops, then 8-bit quantisation (shadows lose ridges)
* noise     additive sensor noise sigma in grey levels
* pprp_jpeg the pprp ladder followed by JPEG q=75 (a typical social-media re-encode)

Optical degradations (blur) are followed by mild sensor noise (sigma=1.5) because a
camera adds noise *after* the optics. Without it, a blurred 8-bit image is
unrealistically invertible.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SENSOR_NOISE = 1.5

LADDERS: dict[str, list[float]] = {
    "pprp": [10, 8, 6, 5, 4, 3.5, 3, 2.5, 2],
    "gauss": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
    "motion": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    "jpeg": [95, 80, 60, 45, 30, 15],
    "exposure": [-1, -2, -3, -4, -5],
    "noise": [2, 4, 8, 12, 16, 24],
    "pprp_jpeg": [10, 8, 6, 5, 4, 3.5, 3, 2.5],
}


@dataclass(frozen=True)
class Condition:
    axis: str
    level: float

    @property
    def label(self) -> str:
        return "clean" if self.axis == "none" else f"{self.axis}={self.level:g}"


def conditions(axes: list[str]) -> list[Condition]:
    out = [Condition("none", 0.0)]
    for a in axes:
        out += [Condition(a, float(v)) for v in LADDERS[a]]
    return out


def _noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def motion_kernel(length: float, angle: float) -> np.ndarray:
    n = int(np.ceil(length)) | 1
    k = np.zeros((n + 2, n + 2), np.float32)
    c = (n + 1) / 2
    dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
    cv2.line(k, (int(round((c - dx) * 16)), int(round((c - dy) * 16))),
             (int(round((c + dx) * 16)), int(round((c + dy) * 16))), 1.0, 1, cv2.LINE_AA, shift=4)
    return k / max(k.sum(), 1e-6)


def apply(img: np.ndarray, cond: Condition, native_pprp: float, seed: int = 0) -> np.ndarray:
    """Degrade `img` (uint8, gray or BGR). Returns None if the condition is not applicable
    (e.g. the target PPRP is above the image's native PPRP: we never upsample)."""
    rng = np.random.default_rng(seed)
    a, v = cond.axis, cond.level
    if a == "none":
        return img.copy()
    if a in ("pprp", "pprp_jpeg"):
        s = v / native_pprp
        if s > 1.0 + 1e-6:
            return None
        out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
        out = _noise(out, SENSOR_NOISE, rng)
        return _jpeg(out, 75) if a == "pprp_jpeg" else out
    if a == "gauss":
        return _noise(cv2.GaussianBlur(img, (0, 0), max(v * native_pprp, 0.1)), SENSOR_NOISE, rng)
    if a == "motion":
        k = motion_kernel(v * native_pprp, rng.uniform(0, np.pi))
        return _noise(cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT), SENSOR_NOISE, rng)
    if a == "jpeg":
        return _jpeg(img, int(v))
    if a == "exposure":
        lin = (img.astype(np.float32) / 255.0) ** 2.2 * (2.0 ** v)        # to linear, scale
        out = np.clip(255 * lin ** (1 / 2.2), 0, 255)                       # back to sRGB-ish
        return _noise(np.round(out).astype(np.uint8), SENSOR_NOISE, rng)
    if a == "noise":
        return _noise(img, v, rng)
    raise ValueError(f"unknown axis {a}")
