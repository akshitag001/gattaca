"""Pose / perspective correction of the upright fingertip patch.

1. Fit straight lines to the left and right finger edges (cv2.fitLine). Perspective
   tilt makes them non-parallel. Map the quadrilateral they bound onto a rectangle
   (getPerspectiveTransform + warpPerspective).
2. Find the distal crease (the darkest row band 0.9-1.9 widths below the tip) and crop
   there, as FingerSafe does, falling back to 1.3 widths.
3. Cylindrical unwrap (cv2.remap): a contactless fingertip is roughly a cylinder,
   so ridges near the edges are foreshortened. Unrolling the arc length restores
   their spacing, which brings the geometry closer to a rolled/contact print. This is
   the cylinder model in Lin & Kumar's DCM, applied on the photo side.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca.debug import DebugSink, dump
from gattaca.types import FingerROI

MAX_UNWRAP = 0.92  # |x/R| limit: beyond this the surface is too oblique to carry ridges


def _edge_lines(mask: np.ndarray, y0: int, y1: int):
    """(x_left(y), x_right(y)) as straight lines fitted over rows y0..y1."""
    left, right = [], []
    for y in range(y0, y1):
        xs = np.flatnonzero(mask[y])
        if len(xs) > 4:
            left.append((xs[0], y))
            right.append((xs[-1], y))
    if len(left) < 10:
        return None
    out = []
    for pts in (left, right):
        vx, vy, x0, yy0 = cv2.fitLine(np.float32(pts), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        out.append(lambda y, vx=vx, vy=vy, x0=x0, yy0=yy0: x0 + (y - yy0) * vx / (vy if abs(vy) > 1e-6 else 1e-6))
    return out


def _tip_row(mask: np.ndarray) -> int:
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[0]) if len(rows) else 0


def find_crease(gray: np.ndarray, mask: np.ndarray, tip_y: int, width: float) -> int | None:
    """Row of the distal interphalangeal crease, or None if no clear dark band."""
    g = gray.astype(np.float32)
    m = mask > 0
    prof = np.array([g[y][m[y]].mean() if m[y].sum() > 0.3 * width else np.nan
                     for y in range(g.shape[0])])
    lo, hi = int(tip_y + 0.9 * width), int(min(len(prof) - 2, tip_y + 1.9 * width))
    if hi - lo < 5 or np.isnan(prof[lo:hi]).all():
        return None
    p = prof.copy()
    p[np.isnan(p)] = np.nanmean(p)
    smooth = cv2.GaussianBlur(p.reshape(-1, 1).astype(np.float32), (1, 0), 2).ravel()
    base = cv2.GaussianBlur(p.reshape(-1, 1).astype(np.float32), (1, 0), width / 4).ravel()
    dip = base - smooth
    y = lo + int(np.argmax(dip[lo:hi]))
    noise = float(np.std(dip[lo:hi])) + 1e-6
    return y if dip[y] > 3 * noise and dip[y] > 2.0 else None


def cylinder_unwrap(img: np.ndarray, mask: np.ndarray, color: np.ndarray | None = None):
    """Unroll a vertical cylinder whose silhouette spans the mask width in each row."""
    h, w = mask.shape
    R = w / 2.0
    smax = R * np.arcsin(MAX_UNWRAP)
    out_w = int(round(2 * smax))
    s = np.linspace(-smax, smax, out_w, dtype=np.float32)
    map_x = np.tile(R + R * np.sin(s / R), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, out_w))
    warp = lambda im, interp: cv2.remap(im, map_x, map_y, interp, borderMode=cv2.BORDER_REPLICATE)  # noqa: E731
    return (warp(img, cv2.INTER_CUBIC), warp(mask, cv2.INTER_NEAREST),
            None if color is None else warp(color, cv2.INTER_CUBIC), map_x)


def rectify(patch: np.ndarray, mask: np.ndarray, M_src2patch: np.ndarray, width: float,
            unwrap: bool = True, debug: DebugSink | None = None) -> FingerROI:
    gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    color = patch if patch.ndim == 3 else None
    tip_y = _tip_row(mask)
    crease = find_crease(gray, mask, tip_y, width)
    bottom = crease if crease is not None else int(tip_y + 1.3 * width)
    bottom = min(bottom, mask.shape[0] - 1)
    lines = _edge_lines(mask, int(tip_y + 0.45 * width), bottom)
    H = np.eye(3)
    out_w = int(round(width))
    out_h = max(8, bottom - tip_y)
    if lines is not None:
        xl, xr = lines
        yt, yb = float(tip_y), float(bottom)
        quad = np.float32([[xl(yt), yt], [xr(yt), yt], [xr(yb), yb], [xl(yb), yb]])
        mean_w = float(np.mean([xr(yt) - xl(yt), xr(yb) - xl(yb)]))
        if mean_w > 0.4 * width:
            out_w = int(round(mean_w))
            rect = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
            H = cv2.getPerspectiveTransform(quad, rect)
    if np.allclose(H, np.eye(3)):
        x0 = int(max(0, mask.shape[1] / 2 - out_w / 2))
        H = np.array([[1, 0, -x0], [0, 1, -tip_y], [0, 0, 1]], np.float64)
    img_r = cv2.warpPerspective(gray, H, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m_r = cv2.warpPerspective(mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
    col_r = None if color is None else cv2.warpPerspective(color, H, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                                           borderMode=cv2.BORDER_REPLICATE)
    dump(debug, "rectify_perspective", img_r)
    if unwrap:
        img_r, m_r, col_r, _ = cylinder_unwrap(img_r, m_r, col_r)
        dump(debug, "rectify_unwrapped", img_r)
    # ROI px -> source px (ignores the unwrap remap; used only for drawing overlays)
    P = np.vstack([M_src2patch, [0, 0, 1]])
    to_source = np.linalg.inv(H @ P)
    if unwrap:  # approximate the unwrap by a horizontal scale for overlay placement
        S = np.diag([out_w / max(img_r.shape[1], 1), 1.0, 1.0])
        to_source = to_source @ S
    return FingerROI(image=img_r, mask=m_r, to_source=to_source, width_px=width, color=col_r)
