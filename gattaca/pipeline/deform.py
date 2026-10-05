"""Deformation correction for contactless <-> contact matching (after Lin & Kumar 2018).

* `TPS`: thin-plate spline with regularisation, solved in normalised coordinates.
* `rtps_fit`: **RTPS-lite**, a robust, iteratively re-weighted TPS. Lin & Kumar add a
  localisation-error term over extra landmarks and iterate until that error has not
  decreased for three iterations. We have no hand-marked landmarks, so we use the
  matched minutiae themselves: each pair is scored against a stiffer TPS, pairs
  beyond 2.5 robust sigmas are dropped, and the fit is repeated with the same
  three-iteration stopping rule.
* `DeformationCorrectionModel`: the *generalised* DCM, a block-wise mean
  displacement field over normalised finger coordinates, learned from training pairs
  and then applied to unseen prints without needing their pairing.
* `warp_image`: OpenCV's ThinPlateSplineShapeTransformer, for visualising the
  correction on images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def _U(r2: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        u = r2 * np.log(r2)
    return np.nan_to_num(u)


@dataclass
class TPS:
    ctrl: np.ndarray       # (n, 2) normalised control points
    w: np.ndarray          # (n, 2) non-affine weights
    a: np.ndarray          # (3, 2) affine part
    mu: np.ndarray         # normalisation centre
    s: float               # normalisation scale

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray, lam: float = 0.05) -> "TPS":
        src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
        mu = src.mean(axis=0)
        s = float(np.sqrt(((src - mu) ** 2).sum(axis=1).mean())) or 1.0
        p = (src - mu) / s
        q = (dst - mu) / s
        n = len(p)
        K = _U(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
        P = np.c_[np.ones(n), p]
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K + lam * np.eye(n)
        L[:n, n:] = P
        L[n:, :n] = P.T
        rhs = np.zeros((n + 3, 2))
        rhs[:n] = q
        sol = np.linalg.lstsq(L, rhs, rcond=None)[0]
        return cls(p, sol[:n], sol[n:], mu, s)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        x = (np.asarray(pts, np.float64) - self.mu) / self.s
        U = _U(((x[:, None, :] - self.ctrl[None, :, :]) ** 2).sum(-1))
        y = np.c_[np.ones(len(x)), x] @ self.a + U @ self.w
        return y * self.s + self.mu


def rtps_fit(src: np.ndarray, dst: np.ndarray, lam: float = 0.05, stiff: float = 2.0,
             floor_px: float = 4.0, max_iter: int = 15) -> tuple[TPS, np.ndarray]:
    """Robust TPS. Returns (model, inlier mask)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    inl = np.ones(len(src), bool)
    best_err, stall, best = np.inf, 0, (TPS.fit(src, dst, lam), inl.copy())
    for _ in range(max_iter):
        if inl.sum() < 4:
            break
        rigid = TPS.fit(src[inl], dst[inl], stiff)
        res = np.linalg.norm(rigid(src) - dst, axis=1)
        mad = 1.4826 * np.median(res[inl])
        new = res < max(2.5 * mad, floor_px)
        err = float(res[new].mean()) if new.any() else np.inf
        if err < best_err - 1e-3:
            best_err, stall = err, 0
            best = (TPS.fit(src[new], dst[new], lam), new.copy())
        else:
            stall += 1
            if stall >= 3:  # Lin & Kumar: stop when E_error has not decreased in 3 iterations
                break
        if np.array_equal(new, inl):
            break
        inl = new
    return best


def warp_image(img: np.ndarray, src: np.ndarray, dst: np.ndarray, reg: float = 50.0) -> np.ndarray:
    """Warp `img` so that `src` landmarks move to `dst` (OpenCV TPS shape transformer)."""
    tps = cv2.createThinPlateSplineShapeTransformer(reg)
    s = np.asarray(src, np.float32).reshape(1, -1, 2)
    d = np.asarray(dst, np.float32).reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(s.shape[1])]
    tps.estimateTransformation(d, s, matches)   # OpenCV warps with the inverse mapping
    return tps.warpImage(img)


@dataclass
class DeformationCorrectionModel:
    """Generalised DCM: mean displacement per block of the normalised finger frame."""

    grid: int = 8
    disp: np.ndarray = field(default_factory=lambda: np.zeros((8, 8, 2)))
    counts: np.ndarray = field(default_factory=lambda: np.zeros((8, 8)))

    @staticmethod
    def frame(mask: np.ndarray) -> tuple[np.ndarray, float]:
        """Finger centre and half-width of the target (contact) print."""
        ys, xs = np.nonzero(mask)
        c = np.array([xs.mean(), ys.mean()])
        half = 0.5 * (np.percentile(xs, 98) - np.percentile(xs, 2))
        return c, max(half, 1.0)

    def _cell(self, pts: np.ndarray, c: np.ndarray, half: float) -> np.ndarray:
        u = (pts - c) / half                                   # ~[-1, 1] across the finger
        return np.clip(((u + 1.25) / 2.5 * self.grid), 0, self.grid - 1e-6)

    def add_pair(self, aligned_src: np.ndarray, dst: np.ndarray, target_mask: np.ndarray) -> None:
        """Accumulate residual displacements of matched, rigidly aligned minutiae."""
        c, half = self.frame(target_mask)
        cell = self._cell(aligned_src, c, half).astype(int)
        d = (dst - aligned_src) / half
        for (gx, gy), dv in zip(cell, d):
            self.disp[gy, gx] += dv
            self.counts[gy, gx] += 1

    def finalize(self, min_count: int = 3) -> "DeformationCorrectionModel":
        f = np.where(self.counts[..., None] >= min_count,
                     self.disp / np.maximum(self.counts[..., None], 1), 0.0)
        f = cv2.GaussianBlur(f.astype(np.float32), (3, 3), 0.8).astype(np.float64)
        return DeformationCorrectionModel(self.grid, f, self.counts.copy())

    def apply(self, aligned_src: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
        c, half = self.frame(target_mask)
        cell = self._cell(aligned_src, c, half)
        fx = cv2.remap(self.disp[..., 0].astype(np.float32), cell[:, 0:1].astype(np.float32) - 0.5,
                       cell[:, 1:2].astype(np.float32) - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fy = cv2.remap(self.disp[..., 1].astype(np.float32), cell[:, 0:1].astype(np.float32) - 0.5,
                       cell[:, 1:2].astype(np.float32) - 0.5, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned_src + np.c_[fx.ravel(), fy.ravel()] * half

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"grid": self.grid, "disp": self.disp.tolist(),
                                          "counts": self.counts.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> "DeformationCorrectionModel":
        d = json.loads(Path(path).read_text())
        return cls(d["grid"], np.array(d["disp"]), np.array(d["counts"]))
