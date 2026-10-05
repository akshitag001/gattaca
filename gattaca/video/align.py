"""Sub-pixel registration of frames to a reference: ECC (cv2.findTransformECC) with an
ORB + RANSAC similarity fallback when ECC does not converge."""

from __future__ import annotations

import cv2
import numpy as np


def _gray32(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return g.astype(np.float32)


def ecc_align(ref: np.ndarray, img: np.ndarray, motion: int = cv2.MOTION_AFFINE,
              iters: int = 200, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Warp W (2x3) with img(W(x)) ~ ref(x), plus the ECC correlation (higher is better)."""
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    cc, warp = cv2.findTransformECC(_gray32(ref), _gray32(img), warp, motion, crit, None, 5)
    return warp, float(cc)


def orb_align(ref: np.ndarray, img: np.ndarray) -> np.ndarray | None:
    orb = cv2.ORB_create(1500)
    k1, d1 = orb.detectAndCompute(_gray32(ref).astype(np.uint8), None)
    k2, d2 = orb.detectAndCompute(_gray32(img).astype(np.uint8), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 8:
        return None
    src = np.float32([k1[x.queryIdx].pt for x in m])
    dst = np.float32([k2[x.trainIdx].pt for x in m])
    W, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    return W


def align(ref: np.ndarray, img: np.ndarray) -> tuple[np.ndarray, float, str]:
    """(warp ref->img coords, quality, method)."""
    try:
        W, cc = ecc_align(ref, img)
        if cc > 0.5:
            return W, cc, "ecc"
    except cv2.error:
        pass
    W = orb_align(ref, img)
    if W is not None:
        return W.astype(np.float32), 0.5, "orb"
    return np.eye(2, 3, dtype=np.float32), 0.0, "identity"


def warp_to_ref(img: np.ndarray, W: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(img, W, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)
