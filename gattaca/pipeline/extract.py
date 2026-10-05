"""Binarise -> thin (Zhang-Suen, cv2.ximgproc) -> crossing-number minutiae -> pruning.

Angle convention (polarity-robust): a ridge ending points *into* its ridge body and a
bifurcation points *toward its forks*. A ridge ending is the dual of a valley
bifurcation at almost the same spot, so with this convention both get nearly the
same angle. That matters for contactless photos, where ridges may image bright or dark.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca.debug import DebugSink, dump
from gattaca.types import BIFURCATION, ENDING, MINUTIA_DTYPE, EnhanceResult

# 8-neighbourhood in circular order (dy, dx), starting east, counter-clockwise on screen
_NB = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]


def _small_components(img: np.ndarray, min_area: int) -> np.ndarray:
    _, lab, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
    return np.isin(lab, small)


def binarize(enh: EnhanceResult, period: float) -> np.ndarray:
    b = ((enh.enhanced > 0) & (enh.mask > 0)).astype(np.uint8) * 255
    min_area = int(0.8 * period * period)
    b[_small_components(b, min_area)] = 0                                   # specks
    b[_small_components(255 - b, min_area) & (enh.mask > 0)] = 255           # pinholes
    return b


def thin(binary: np.ndarray) -> np.ndarray:
    return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)


def crossing_number(skel: np.ndarray) -> np.ndarray:
    s = (skel > 0).astype(np.int16)
    p = np.pad(s, 1)
    h, w = s.shape
    nbs = [p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in _NB]
    cn = sum(np.abs(nbs[i] - nbs[(i + 1) % 8]) for i in range(8)) // 2
    return np.where(s > 0, cn, 0)


def _trace(skel: np.ndarray, y: int, x: int, first: tuple[int, int], steps: int):
    """Follow the skeleton from (y, x) through neighbour `first` for up to `steps` px."""
    h, w = skel.shape
    prev, cur = (y, x), first
    for _ in range(steps - 1):
        nxt = None
        for dy, dx in _NB:
            q = (cur[0] + dy, cur[1] + dx)
            if q != prev and q != (y, x) and 0 <= q[0] < h and 0 <= q[1] < w and skel[q]:
                nxt = q
                break
        if nxt is None:
            break
        prev, cur = cur, nxt
    return cur


def _neighbours(skel, y, x):
    h, w = skel.shape
    return [(y + dy, x + dx) for dy, dx in _NB
            if 0 <= y + dy < h and 0 <= x + dx < w and skel[y + dy, x + dx]]


def minutia_angle(skel: np.ndarray, y: int, x: int, kind: int, steps: int) -> float | None:
    nbs = _neighbours(skel, y, x)
    if kind == ENDING:
        if len(nbs) != 1:
            return None
        ty, tx = _trace(skel, y, x, nbs[0], steps)
        return float(np.arctan2(ty - y, tx - x)) % (2 * np.pi)
    vecs = []
    for nb in nbs:
        ty, tx = _trace(skel, y, x, nb, steps)
        v = np.array([tx - x, ty - y], np.float64)
        n = np.linalg.norm(v)
        if n > 0:
            vecs.append(v / n)
    if len(vecs) < 3:
        return None
    best, pair = -2.0, (0, 1)
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            d = float(vecs[i] @ vecs[j])
            if d > best:
                best, pair = d, (i, j)
    v = vecs[pair[0]] + vecs[pair[1]]
    return float(np.arctan2(v[1], v[0])) % (2 * np.pi)


def extract_minutiae(enh: EnhanceResult, period: float | None = None,
                     debug: DebugSink | None = None) -> np.ndarray:
    period = float(np.median(enh.period[enh.mask > 0])) if period is None and (enh.mask > 0).any() \
        else (period or 9.0)
    binary = binarize(enh, period)
    dump(debug, "extract_binary", binary)
    skel = thin(binary) > 0
    dump(debug, "extract_skeleton", skel.astype(np.uint8) * 255)
    cn = crossing_number(skel)
    margin = int(round(1.2 * period))
    inner = cv2.erode(enh.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    ys, xs = np.nonzero(((cn == 1) | (cn == 3)) & (inner > 0))
    steps = max(4, int(round(period)))
    rows = []
    for y, x in zip(ys, xs):
        kind = ENDING if cn[y, x] == 1 else BIFURCATION
        ang = minutia_angle(skel, y, x, kind, steps)
        if ang is None:
            continue
        rows.append((x, y, ang, kind))
    if not rows:
        return np.zeros(0, MINUTIA_DTYPE)
    arr = np.array(rows, np.float64)
    # false-minutiae pruning: two minutiae closer than ~one ridge period are almost always
    # a broken ridge, a bridge or a spur, so drop both
    d = np.hypot(arr[:, None, 0] - arr[None, :, 0], arr[:, None, 1] - arr[None, :, 1])
    np.fill_diagonal(d, np.inf)
    keep = d.min(axis=1) > 0.9 * period
    arr = arr[keep]
    out = np.zeros(len(arr), MINUTIA_DTYPE)
    out["x"], out["y"], out["angle"], out["kind"] = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    # quality: orientation certainty x local ridge amplitude around the point
    amp = cv2.GaussianBlur(np.abs(enh.enhanced), (0, 0), period)
    ref = float(np.percentile(amp[enh.mask > 0], 90)) + 1e-6 if (enh.mask > 0).any() else 1.0
    coh = cv2.GaussianBlur(enh.coherence, (0, 0), period)
    xi, yi = out["x"].astype(int), out["y"].astype(int)
    out["quality"] = np.clip(coh[yi, xi] * np.minimum(amp[yi, xi] / ref, 1.0), 0, 1)
    if debug is not None:
        vis = cv2.cvtColor(skel.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        for m in out:
            c = (0, 0, 255) if m["kind"] == ENDING else (0, 200, 0)
            p = (int(m["x"]), int(m["y"]))
            cv2.circle(vis, p, 4, c, 1)
            q = (int(m["x"] + 10 * np.cos(m["angle"])), int(m["y"] + 10 * np.sin(m["angle"])))
            cv2.line(vis, p, q, c, 1)
        debug.save("extract_minutiae", vis)
    return out
