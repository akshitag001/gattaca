"""Ridge-period estimation from block spectra: the PPRP measurement itself.

Each block is Hann-windowed and FFT'd, then the power spectrum is binned in polar
coordinates. A ridge pattern concentrates its energy at one radius (the ridge
frequency) *and one angle* (the ridge normal). Isotropic noise and texture spread
energy over all angles. So per radius we take the angular **max** (ridge energy)
and the angular **median** (background), choose the radius with the largest
ridge energy, and report `snr = max/median` there. Blocks with low snr are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_PERIOD = 2.2     # just above Nyquist; below this ridges alias and are unrecoverable
MAX_PERIOD = 32.0
N_ANGLE = 16


@dataclass
class BlockSpectra:
    centers: np.ndarray      # (n, 2) block centre (x, y)
    period: np.ndarray       # (n,) dominant period in px (nan if rejected)
    snr: np.ndarray          # (n,) angular peak / median at the peak radius
    normal: np.ndarray       # (n,) ridge-normal angle of the spectral peak [0, pi)


def _polar_index(n: int):
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    r = np.hypot(fx, fy)
    ang = np.mod(np.arctan2(fy, fx), np.pi)
    rbin = np.round(r * n).astype(np.int32)
    abin = np.minimum((ang / np.pi * N_ANGLE).astype(np.int32), N_ANGLE - 1)
    return rbin, abin


def block_spectra(img: np.ndarray, mask: np.ndarray | None, block: int = 64,
                  stride: int | None = None, pad: int = 2, min_fill: float = 0.7) -> BlockSpectra:
    """Dominant ridge period per block. `img` is any grayscale image (uint8 or float)."""
    g = img.astype(np.float32)
    # remove shading so it doesn't leak into the ridge band
    g = g - cv2.GaussianBlur(g, (0, 0), 12)
    h, w = g.shape
    stride = stride or block // 2
    m = np.ones_like(g, np.uint8) if mask is None else (mask > 0).astype(np.uint8)
    win = np.outer(np.hanning(block), np.hanning(block)).astype(np.float32)
    blocks, centers = [], []
    for y in range(0, max(h - block, 0) + 1, stride):
        for x in range(0, max(w - block, 0) + 1, stride):
            if m[y:y + block, x:x + block].mean() < min_fill:
                continue
            b = g[y:y + block, x:x + block]
            if b.shape != (block, block):
                continue
            b = (b - b.mean()) * win
            blocks.append(b)
            centers.append((x + block / 2, y + block / 2))
    n = block * pad
    if not blocks:
        e = np.zeros(0, np.float32)
        return BlockSpectra(np.zeros((0, 2), np.float32), e, e, e)
    stack = np.stack(blocks)
    P = np.abs(np.fft.fft2(stack, s=(n, n))) ** 2
    rbin, abin = _polar_index(n)
    rmin = int(np.floor(n / MAX_PERIOD))
    rmax = int(np.ceil(n / MIN_PERIOD))
    rmax = min(rmax, n // 2)
    nr = rmax + 2
    idx = (np.clip(rbin, 0, nr - 1) * N_ANGLE + abin).ravel()
    valid = (rbin.ravel() <= rmax)
    polar = np.zeros((len(stack), nr * N_ANGLE), np.float64)
    flatP = P.reshape(len(stack), -1)
    for i in range(len(stack)):           # angular max per (radius, angle) cell
        np.maximum.at(polar[i], idx[valid], flatP[i, valid])
    polar = polar.reshape(len(stack), nr, N_ANGLE)
    peak = polar.max(axis=2)
    bg = np.median(polar, axis=2) + 1e-9
    band = np.zeros(nr, bool)
    band[rmin:rmax + 1] = True
    # whiten: natural-image power falls ~1/f^2, which would otherwise make residual
    # shading win over fine ridges at low PPRP
    radius = np.arange(nr, dtype=np.float64)
    score = np.where(band[None, :], peak * radius[None, :] ** 2, -1.0)
    k = score.argmax(axis=1)
    # parabolic refinement of the radial peak
    kk = np.clip(k, 1, nr - 2)
    a, b, c = (score[np.arange(len(k)), kk - 1], score[np.arange(len(k)), kk],
               score[np.arange(len(k)), kk + 1])
    a, c = np.maximum(a, 0), np.maximum(c, 0)
    denom = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
    rpk = kk + np.clip(off, -0.5, 0.5)
    period = n / np.maximum(rpk, 1e-6)
    snr = peak[np.arange(len(k)), k] / bg[np.arange(len(k)), k]
    ang_bin = polar[np.arange(len(k)), k].argmax(axis=1)
    normal = (ang_bin + 0.5) * np.pi / N_ANGLE
    return BlockSpectra(np.asarray(centers, np.float32), period.astype(np.float32),
                        snr.astype(np.float32), normal.astype(np.float32))


def dominant_period(img: np.ndarray, mask: np.ndarray | None = None, block: int = 64,
                    snr_min: float = 4.0) -> tuple[float, float, BlockSpectra]:
    """Robust image-level ridge period (px) and spectral SNR. Returns (nan, snr, spectra)
    when no block shows a ridge peak."""
    h, w = img.shape[:2]
    if mask is not None and (mask > 0).any():
        # size blocks to the finger, not the frame: ~half the finger's inscribed width
        inscribed = 2 * float(cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5).max())
        extent = min(h, w, inscribed)
    else:
        extent = min(h, w)
    block = int(np.clip(2 ** int(np.log2(max(extent / 2, 16))), 16, block))
    sp = block_spectra(img, mask, block=block)
    if len(sp.period) == 0:
        return float("nan"), 0.0, sp
    good = sp.snr >= snr_min
    snr = float(np.median(np.sort(sp.snr)[-max(1, len(sp.snr) // 3):]))
    if not good.any():
        return float("nan"), snr, sp
    wts = np.log(sp.snr[good])
    order = np.argsort(sp.period[good])
    cw = np.cumsum(wts[order])
    period = float(sp.period[good][order][np.searchsorted(cw, cw[-1] / 2)])
    return period, snr, sp
