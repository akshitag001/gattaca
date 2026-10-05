"""Minutiae matching, used ONLY to measure recovery against paired, consented ground truth.

1. Local descriptors: each minutia's k nearest neighbours as (distance, bearing
   relative to the minutia angle, neighbour angle relative to the minutia angle).
2. The top candidate correspondences seed rigid alignment hypotheses (rotation
   bounded: both prints are upright after rectification).
3. Each hypothesis is scored by mutual-nearest pairing within distance/angle
   tolerances. The best one is refined with cv2.estimateAffinePartial2D (RANSAC
   similarity, scale clamped). Scale normalisation from PPRP leaves only a few %
   of scale error.
4. Optional deformation correction: generalised DCM and/or pairwise RTPS-lite, then
   re-pairing.

Score = n^2 / (N_P * N_Q), as in Lin & Kumar 2018. All coordinates are at canonical
(500 ppi-equivalent) scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gattaca.pipeline.deform import DeformationCorrectionModel, rtps_fit

D_TOL = 14.0                 # px at canonical scale (~1.5 ridge periods)
A_TOL = np.deg2rad(30)
MAX_ROT = np.deg2rad(60)
K_NEIGH = 5
N_HYP = 40


def _angdiff(a, b):
    d = np.mod(a - b + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def descriptors(m: np.ndarray, k: int = K_NEIGH) -> np.ndarray:
    xy = np.c_[m["x"], m["y"]].astype(np.float64)
    th = m["angle"].astype(np.float64)
    n = len(m)
    out = np.full((n, k, 3), np.nan)
    if n < 2:
        return out
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    nn = np.argsort(d, axis=1)[:, :kk]
    for i in range(n):
        j = nn[i]
        v = xy[j] - xy[i]
        out[i, :kk, 0] = d[i, j]
        out[i, :kk, 1] = np.mod(np.arctan2(v[:, 1], v[:, 0]) - th[i], 2 * np.pi)
        out[i, :kk, 2] = np.mod(th[j] - th[i], 2 * np.pi)
    return out


def descriptor_similarity(dp: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """(n_p, n_q) similarity; each P neighbour is matched to its best Q neighbour."""
    a = dp[:, None, :, None, :]
    b = dq[None, :, None, :, :]
    pen = (np.abs(a[..., 0] - b[..., 0]) / 8.0
           + _angdiff(a[..., 1], b[..., 1]) / (np.pi / 8)
           + _angdiff(a[..., 2], b[..., 2]) / (np.pi / 8))
    s = np.exp(-np.nan_to_num(pen, nan=1e3))
    return s.max(axis=3).mean(axis=2)


def _pair(pa: np.ndarray, ta: np.ndarray, q: np.ndarray, tq: np.ndarray,
          d_tol: float, a_tol: float) -> np.ndarray:
    """Mutual-nearest pairs (i, j) within tolerance."""
    if len(pa) == 0 or len(q) == 0:
        return np.zeros((0, 2), int)
    d = np.hypot(*(pa[:, None, :] - q[None, :, :]).transpose(2, 0, 1))
    ok = (d < d_tol) & (_angdiff(ta[:, None], tq[None, :]) < a_tol)
    cost = np.where(ok, d, np.inf)
    bi = cost.argmin(axis=1)
    bj = cost.argmin(axis=0)
    i = np.flatnonzero(np.isfinite(cost[np.arange(len(pa)), bi]) & (bj[bi] == np.arange(len(pa))))
    return np.c_[i, bi[i]]


def _apply(M: np.ndarray, xy: np.ndarray, th: np.ndarray):
    rot = np.arctan2(M[1, 0], M[0, 0])
    return xy @ M[:, :2].T + M[:, 2], np.mod(th + rot, 2 * np.pi)


@dataclass
class MatchResult:
    score: float
    n_matched: int
    n_p: int
    n_q: int
    transform: np.ndarray = field(default_factory=lambda: np.eye(2, 3))
    pairs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))
    aligned_p: np.ndarray | None = None
    stage: str = "rigid"


def match(p: np.ndarray, q: np.ndarray, *, q_mask: np.ndarray | None = None,
          dcm: DeformationCorrectionModel | None = None, rtps: bool = False,
          d_tol: float = D_TOL, a_tol: float = A_TOL) -> MatchResult:
    """Match probe minutiae `p` (contactless) against gallery minutiae `q` (contact)."""
    n_p, n_q = len(p), len(q)
    if n_p < 3 or n_q < 3:
        return MatchResult(0.0, 0, n_p, n_q)
    pxy = np.c_[p["x"], p["y"]].astype(np.float64)
    qxy = np.c_[q["x"], q["y"]].astype(np.float64)
    pt, qt = p["angle"].astype(np.float64), q["angle"].astype(np.float64)
    sim = descriptor_similarity(descriptors(p), descriptors(q))
    rot = np.mod(qt[None, :] - pt[:, None] + np.pi, 2 * np.pi) - np.pi
    sim = np.where(np.abs(rot) <= MAX_ROT, sim, -1)
    flat = np.argsort(sim, axis=None)[::-1][:N_HYP]
    best_pairs, best_M = np.zeros((0, 2), int), np.eye(2, 3)
    for f in flat:
        i, j = np.unravel_index(f, sim.shape)
        if sim[i, j] <= 0:
            break
        r = rot[i, j]
        R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        M = np.c_[R, qxy[j] - R @ pxy[i]]
        pa, ta = _apply(M, pxy, pt)
        pairs = _pair(pa, ta, qxy, qt, d_tol, a_tol)
        if len(pairs) > len(best_pairs):
            best_pairs, best_M = pairs, M
    # refine with an OpenCV RANSAC similarity on the inlier pairs
    for _ in range(2):
        if len(best_pairs) < 3:
            break
        M, inl = cv2.estimateAffinePartial2D(pxy[best_pairs[:, 0]], qxy[best_pairs[:, 1]],
                                             method=cv2.RANSAC, ransacReprojThreshold=d_tol / 2)
        if M is None:
            break
        sc = np.hypot(M[0, 0], M[1, 0])
        if not 0.85 < sc < 1.15:
            break
        pa, ta = _apply(M, pxy, pt)
        pairs = _pair(pa, ta, qxy, qt, d_tol, a_tol)
        if len(pairs) >= len(best_pairs):
            best_pairs, best_M = pairs, M
    pa, ta = _apply(best_M, pxy, pt)
    stage = "rigid"
    if dcm is not None and q_mask is not None:
        pc = dcm.apply(pa, q_mask)
        pairs = _pair(pc, ta, qxy, qt, d_tol, a_tol)
        if len(pairs) >= len(best_pairs):
            best_pairs, pa, stage = pairs, pc, "dcm"
    if rtps and len(best_pairs) >= 6:
        tps, inl = rtps_fit(pa[best_pairs[:, 0]], qxy[best_pairs[:, 1]], lam=0.5)
        if inl.sum() >= 6:
            pw = tps(pa)
            pairs = _pair(pw, ta, qxy, qt, d_tol * 0.8, a_tol)
            if len(pairs) >= len(best_pairs):
                best_pairs, pa, stage = pairs, pw, stage + "+rtps"
    n = len(best_pairs)
    return MatchResult(n * n / (n_p * n_q), n, n_p, n_q, best_M, best_pairs, pa, stage)
