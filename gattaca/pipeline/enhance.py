"""Illumination normalisation + classical ridge enhancement.

CLAHE -> denoise -> local mean/contrast normalisation -> orientation field (gradient
structure tensor) -> local ridge period (block spectra) -> oriented Gabor bank.
The input is expected at (roughly) canonical scale: the analyzer resamples each ROI
so its measured period is CANONICAL_PERIOD_PX before calling `enhance`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gattaca import CANONICAL_PERIOD_PX
from gattaca.debug import DebugSink, dump, orientation_to_bgr
from gattaca.pipeline.ridgefreq import block_spectra
from gattaca.types import EnhanceResult


@dataclass(frozen=True)
class EnhanceParams:
    """Knobs the agent may retune (see gattaca/agent)."""

    clahe_clip: float = 2.0
    clahe_tile: int = 8
    denoise: str = "bilateral"         # "bilateral" | "nlm" | "none"
    orient_sigma: float = 7.0          # tensor smoothing (px at canonical scale)
    gabor_sigma: float = 0.5           # Gaussian sigma as a fraction of the period
    gabor_gamma: float = 0.6           # <1 elongates the kernel along the ridge
    n_orient: int = 16
    period_bands: tuple[float, ...] = (0.85, 1.0, 1.18)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def local_normalize(gray: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Zero local mean, unit local contrast; ridges positive (dark ridges -> +)."""
    g = gray.astype(np.float32)
    m = (mask > 0).astype(np.float32)
    msum = cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6
    mean = cv2.GaussianBlur(g * m, (0, 0), sigma) / msum
    d = (g - mean) * m
    var = cv2.GaussianBlur(d * d, (0, 0), sigma) / msum
    out = -d / np.sqrt(var + 1e-3)
    return np.clip(out, -4, 4) * m


def orientation_field(norm: np.ndarray, mask: np.ndarray, sigma: float, coh_sigma: float | None = None):
    """Ridge-normal angle in [0, pi) and coherence from the smoothed structure tensor.

    Coherence uses a wider window (default 2*sigma). Real ridges stay coherent over
    several periods (>0.5), while band-limited isotropic noise does not (<0.2) even
    when it locally looks ridge-like."""
    g = cv2.GaussianBlur(norm.astype(np.float32, copy=False), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    a, b, c = gx * gx, gy * gy, gx * gy
    gxx, gyy, gxy = (cv2.GaussianBlur(t, (0, 0), sigma) for t in (a, b, c))
    normal = np.mod(0.5 * np.arctan2(2 * gxy, gxx - gyy), np.pi).astype(np.float32)
    cs = coh_sigma or 2 * sigma
    wxx, wyy, wxy = (cv2.GaussianBlur(t, (0, 0), cs) for t in (a, b, c))
    coh = np.sqrt((wxx - wyy) ** 2 + 4 * wxy**2) / (wxx + wyy + 1e-6)
    coh = np.clip(coh, 0, 1).astype(np.float32) * (mask > 0)
    return normal, coh


def period_field(norm: np.ndarray, mask: np.ndarray, block: int = 32) -> np.ndarray:
    """Local ridge period map (px), interpolated from block spectra, clamped around the median."""
    h, w = norm.shape
    sp = block_spectra(norm, mask, block=block, stride=block // 2, min_fill=0.5)
    good = sp.snr >= 4
    if good.sum() < 1:
        return np.full((h, w), CANONICAL_PERIOD_PX, np.float32)
    med = float(np.median(sp.period[good]))
    gh, gw = max(1, h // (block // 2)), max(1, w // (block // 2))
    grid = np.full((gh, gw), np.nan, np.float32)
    for (cx, cy), p in zip(sp.centers[good], sp.period[good]):
        gy, gx = min(int(cy // (block // 2)), gh - 1), min(int(cx // (block // 2)), gw - 1)
        grid[gy, gx] = p
    grid = np.where(np.isnan(grid), med, grid)
    grid = cv2.medianBlur(grid, 3) if min(gh, gw) >= 3 else grid
    field = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    field = cv2.GaussianBlur(field, (0, 0), block / 2)
    return np.clip(field, 0.75 * med, 1.33 * med).astype(np.float32)


def gabor_bank_filter(norm: np.ndarray, normal: np.ndarray, period: np.ndarray,
                      mask: np.ndarray, p: EnhanceParams) -> np.ndarray:
    """Per-pixel selection from an (orientation x period) bank of zero-mean Gabor kernels."""
    med = float(np.median(period[mask > 0])) if (mask > 0).any() else CANONICAL_PERIOD_PX
    periods = np.array([med * b for b in p.period_bands], np.float32)
    obin = np.round(normal / np.pi * p.n_orient).astype(np.int32) % p.n_orient
    pbin = np.abs(period[..., None] - periods[None, None, :]).argmin(axis=2)
    out = np.zeros_like(norm)
    for pi, lam in enumerate(periods):
        sigma = p.gabor_sigma * lam
        k = int(2 * np.ceil(3 * sigma / p.gabor_gamma) + 1)
        for oi in range(p.n_orient):
            sel = (obin == oi) & (pbin == pi)
            if not sel.any():
                continue
            kern = cv2.getGaborKernel((k, k), sigma, oi * np.pi / p.n_orient, float(lam),
                                      p.gabor_gamma, 0, ktype=cv2.CV_32F)
            kern -= kern.mean()
            kern /= np.abs(kern).sum() + 1e-9
            out[sel] = cv2.filter2D(norm, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT)[sel]
    return out * (mask > 0)


def ridge_mask(norm: np.ndarray, coh: np.ndarray, mask: np.ndarray, period: float) -> np.ndarray:
    """Usable ridge area: inside the finger mask with locally coherent orientation."""
    c = cv2.GaussianBlur(coh, (0, 0), period)
    m = ((c > 0.3) & (mask > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * period) | 1,) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def enhance(img: np.ndarray, mask: np.ndarray | None = None,
            params: EnhanceParams = EnhanceParams(),
            debug: DebugSink | None = None) -> EnhanceResult:
    gray = to_gray(img)
    mask = np.full(gray.shape, 255, np.uint8) if mask is None else mask
    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip,
                            tileGridSize=(params.clahe_tile, params.clahe_tile))
    eq = clahe.apply(gray)
    dump(debug, "enhance_clahe", eq)
    if params.denoise == "nlm":
        eq = cv2.fastNlMeansDenoising(eq, None, h=7, templateWindowSize=5, searchWindowSize=15)
    elif params.denoise == "bilateral":
        eq = cv2.bilateralFilter(eq, 5, 20, 3)
    dump(debug, f"enhance_denoise_{params.denoise}", eq)
    norm = local_normalize(eq, mask, sigma=2 * CANONICAL_PERIOD_PX)
    dump(debug, "enhance_normalized", norm)
    normal, coh = orientation_field(norm, mask, params.orient_sigma)
    if debug is not None:
        debug.save("enhance_orientation", orientation_to_bgr(normal, coh))
    period = period_field(norm, mask)
    dump(debug, "enhance_period", period)
    rmask = ridge_mask(norm, coh, mask, float(np.median(period)))
    gab = gabor_bank_filter(norm, normal, period, rmask, params)
    dump(debug, "enhance_gabor", gab)
    return EnhanceResult(enhanced=gab, orientation=normal, coherence=coh, period=period,
                         mask=rmask, normalized=norm)
