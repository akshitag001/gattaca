"""Finger segmentation and distal-phalanx crop (FingerSafe recipe).

Binary mask via thresholding + erosion/dilation -> largest contour -> oriented crop
around the fingertip with the tip at the top, at 1:1 pixel scale so that the
measured PPRP is the true PPRP of the photo.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca.debug import DebugSink, dump
from gattaca.types import FingertipDetection


def finger_mask(img: np.ndarray, debug: DebugSink | None = None) -> np.ndarray:
    """uint8 {0,255} mask of the largest skin/foreground region."""
    if img.ndim == 3:
        cr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 1]
        chan = cv2.GaussianBlur(cr, (0, 0), 3)
        # skin is high-Cr; Otsu separates it from most backgrounds
        _, m = cv2.threshold(chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # if Otsu split inside the skin (low-contrast scene), fall back to luminance
        if m.mean() / 255 < 0.03 or m.mean() / 255 > 0.97:
            gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 3)
            _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        gray = cv2.GaussianBlur(img, (0, 0), 3)
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(0.012 * min(img.shape[:2])) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.dilate(cv2.erode(m, se), se)            # open: drop specks
    m = cv2.erode(cv2.dilate(m, se, iterations=2), se, iterations=2)  # close: fill gaps
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if contours:
        big = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big) > 0.01 * m.size:
            cv2.drawContours(out, [big], -1, 255, cv2.FILLED)
    dump(debug, "roi_finger_mask", out)
    return out


def oriented_patch(img: np.ndarray, det: FingertipDetection, length_ratio: float = 2.0,
                   margin: float = 0.25):
    """Crop an upright patch (tip at top) around the distal phalanx at 1:1 scale.

    Returns (patch, mask_patch, M) where M is the 2x3 affine source -> patch."""
    a = np.asarray(det.axis, np.float64)
    a /= np.linalg.norm(a) + 1e-12
    perp = np.array([a[1], -a[0]])                 # right-handed, no mirroring
    w = det.width
    pw = int(round(w * (1 + 2 * margin)))
    ph = int(round(w * (length_ratio + margin)))
    tip = np.asarray(det.tip, np.float64)
    origin = tip - perp * (pw / 2) - a * (margin * w)
    M = np.array([[perp[0], perp[1], 0.0], [a[0], a[1], 0.0]])
    M[:, 2] = -M[:, :2] @ origin
    patch = cv2.warpAffine(img, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    src_mask = det.mask if det.mask is not None else finger_mask(img)
    mpatch = cv2.warpAffine(src_mask, M, (pw, ph), flags=cv2.INTER_NEAREST)
    # keep only the component containing the tip column (neighbouring fingers out)
    n, lab = cv2.connectedComponents(mpatch)
    if n > 2:
        cx = pw // 2
        col = lab[:, cx]
        ids = col[col > 0]
        if len(ids):
            mpatch = np.where(lab == np.bincount(ids).argmax(), 255, 0).astype(np.uint8)
    return patch, mpatch, M
