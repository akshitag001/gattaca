"""Debug sink: every stage can dump the intermediate image OpenCV produced."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_u8(img: np.ndarray) -> np.ndarray:
    """Min-max scale any image to uint8 for viewing."""
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
    if hi - lo < 1e-9:
        return np.zeros(f.shape, np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def orientation_to_bgr(theta: np.ndarray, strength: np.ndarray | None = None) -> np.ndarray:
    """Colour-code an orientation field (hue = angle, value = strength)."""
    hue = (np.mod(theta, np.pi) / np.pi * 179).astype(np.uint8)
    val = np.full(theta.shape, 255, np.uint8) if strength is None else to_u8(strength)
    hsv = cv2.merge([hue, np.full_like(hue, 255), val])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class DebugSink:
    """Writes numbered PNGs (`03_enhance_gabor.png`) so the stage order is visible."""

    def __init__(self, out_dir: str | Path, prefix: str = ""):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._n = 0

    def save(self, name: str, img: np.ndarray) -> Path:
        self._n += 1
        path = self.dir / f"{self.prefix}{self._n:02d}_{name}.png"
        cv2.imwrite(str(path), to_u8(img))
        return path


def dump(debug: DebugSink | None, name: str, img: np.ndarray) -> None:
    if debug is not None:
        debug.save(name, img)
