"""Multi-frame fusion: the *principle* of handheld multi-frame super-resolution
(Wronski et al. 2019), not their full algorithm.

Hand tremor shifts each frame by a different sub-pixel amount. After ECC alignment
we know every frame pixel's exact position in the reference frame, so we splat
it onto a finer grid (scale x) at that position with bilinear kernel weights
(kernel regression). Pixels that disagree with the warped reference, from local
motion or occlusion, are down-weighted (robustness term). Cells that receive no
samples fall back to the upsampled reference.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca.video.align import align, warp_to_ref


def _splat(acc: np.ndarray, wacc: np.ndarray, img: np.ndarray, X: np.ndarray, Y: np.ndarray, w: np.ndarray):
    """Bilinear splat of `img` samples at (X, Y) into channel-first `acc` (C, H, W) and `wacc`."""
    H, Wd = wacc.shape
    x0, y0 = np.floor(X).astype(np.int64), np.floor(Y).astype(np.int64)
    fx, fy = X - x0, Y - y0
    for dx, dy, k in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < Wd) & (yi >= 0) & (yi < H)
        idx = yi[ok] * Wd + xi[ok]
        ww = (k * w)[ok]
        wacc += np.bincount(idx, weights=ww, minlength=H * Wd).reshape(H, Wd)
        for c in range(acc.shape[0]):
            acc[c] += np.bincount(idx, weights=ww * img[..., c][ok], minlength=H * Wd).reshape(H, Wd)


def fuse(frames: list[np.ndarray], ref_index: int = 0, scale: int = 2, noise_sigma: float = 4.0,
         warps: list[np.ndarray] | None = None) -> tuple[np.ndarray, dict]:
    """Fuse aligned crops of the same region into one image `scale`x larger.

    Returns (fused uint8 image, info). `warps[i]` maps reference coords -> frame-i coords
    (the ECC convention). They are estimated if not given."""
    ref = frames[ref_index]
    h, w = ref.shape[:2]
    chans = ref.shape[2] if ref.ndim == 3 else 1
    if warps is None:
        warps, quality = [], []
        for i, f in enumerate(frames):
            if i == ref_index:
                warps.append(np.eye(2, 3, dtype=np.float32))
                quality.append(1.0)
                continue
            W, q, _ = align(ref, f)
            warps.append(W)
            quality.append(q)
    else:
        quality = [1.0] * len(frames)
    H, Wd = h * scale, w * scale
    acc = np.zeros((chans, H, Wd), np.float64)
    wacc = np.zeros((H, Wd), np.float64)
    ref_f = ref.astype(np.float32).reshape(h, w, chans)
    used = 0
    for f, Wm, q in zip(frames, warps, quality):
        if q < 0.5:
            continue
        fh, fw = f.shape[:2]
        ff = f.astype(np.float32).reshape(fh, fw, chans)
        inv = cv2.invertAffineTransform(Wm)                       # frame coords -> ref coords
        yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float64)
        rx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
        ry = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
        # robustness: compare the frame (warped to ref) with the reference, per pixel
        back = warp_to_ref(f, Wm, (w, h)).astype(np.float32).reshape(h, w, chans)
        diff = cv2.GaussianBlur(np.abs(back - ref_f).mean(axis=2), (0, 0), 1.5)
        rob_ref = np.exp(-(diff ** 2) / (2 * (3 * noise_sigma) ** 2))
        rob = cv2.remap(rob_ref.astype(np.float32), rx.astype(np.float32), ry.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        X = (rx + 0.5) * scale - 0.5
        Y = (ry + 0.5) * scale - 0.5
        _splat(acc, wacc, ff, X, Y, rob.astype(np.float64) * q)
        used += 1
    up = cv2.resize(ref, (Wd, H), interpolation=cv2.INTER_CUBIC).astype(np.float64).reshape(H, Wd, chans)
    fill = np.clip(0.3 - wacc, 0, None)[..., None]        # weak prior where samples are sparse
    out = (acc.transpose(1, 2, 0) + fill * up) / (wacc[..., None] + fill + 1e-9)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if chans == 1:
        out = out[..., 0]
    return out, {"frames_used": used, "align_quality": [round(float(q), 3) for q in quality]}
