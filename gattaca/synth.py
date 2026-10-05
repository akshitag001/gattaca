"""Synthetic paired fingerprints for building and validating the pipeline.

The licensed datasets (see datasets/README.md) take days to approve. This module
generates an unlimited supply of *paired* data so every stage, the study and the
calibration code can run end to end today:

* a **master ridge pattern** per identity: SFinGe-style iterative, orientation-
  adaptive Gabor growth from noise over a Sherlock-Monro orientation field,
  so minutiae emerge naturally;
* a **contact impression**: elastic, pressure-distorted, inked rendering at 500 ppi
  (the ground truth);
* a **contactless photo**: the same master wrapped onto a cylindrical fingertip,
  shaded, low-contrast, skin-toned, with a knuckle crease, background, pose and noise.

Synthetic results validate the measurement *instrument*. They are not the
headline curve, which needs the real paired datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from gattaca import CANONICAL_PERIOD_PX

MASTER_SIZE = 480


def sherlock_monro(h: int, w: int, cores, deltas, base: float = 0.0) -> np.ndarray:
    """Ridge *direction* field (radians, mod pi) from singular points (Sherlock & Monro 1993)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = np.full((h, w), base, np.float64)
    for cx, cy in cores:
        theta += 0.5 * np.arctan2(yy - cy, xx - cx)
    for dx, dy in deltas:
        theta -= 0.5 * np.arctan2(yy - dy, xx - dx)
    return np.mod(theta, np.pi).astype(np.float32)


def _random_orientation(rng: np.random.Generator, n: int) -> np.ndarray:
    c = n / 2
    kind = rng.choice(["loop", "loop", "whorl", "arch"])
    jitter = lambda s: rng.uniform(-s, s)  # noqa: E731
    if kind == "loop":
        lean = rng.choice([-1, 1]) * rng.uniform(0.08, 0.2) * n
        core = (c + jitter(0.06 * n), c - 0.08 * n + jitter(0.05 * n))
        delta = (core[0] + lean, core[1] + rng.uniform(0.3, 0.42) * n)
        theta = sherlock_monro(n, n, [core], [delta], base=np.pi / 2 + jitter(0.15))
    elif kind == "whorl":
        y0 = c + jitter(0.05 * n)
        cores = [(c - 0.04 * n, y0 - 0.05 * n), (c + 0.04 * n, y0 + 0.03 * n)]
        deltas = [(c - 0.32 * n, y0 + 0.36 * n), (c + 0.32 * n, y0 + 0.36 * n)]
        theta = sherlock_monro(n, n, cores, deltas, base=np.pi / 2 + jitter(0.1))
    else:  # arch: smooth bump, no singularities
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        amp = rng.uniform(0.5, 0.9)
        sig = rng.uniform(0.18, 0.3) * n
        slope = -amp * (xx - c) / sig * np.exp(-((xx - c) ** 2) / (2 * sig**2)) * np.exp(
            -((yy - c * 0.9) ** 2) / (2 * (0.5 * n) ** 2)
        )
        theta = np.mod(np.arctan(slope), np.pi).astype(np.float32)
    return theta


def _gabor_bank(period: float, n_bins: int) -> list[np.ndarray]:
    sigma = 0.55 * period
    k = int(2 * round(3 * sigma) + 1)
    bank = []
    for b in range(n_bins):
        normal = b * np.pi / n_bins
        kern = cv2.getGaborKernel((k, k), sigma, normal, period, 1.0, 0, ktype=cv2.CV_32F)
        kern -= kern.mean()
        bank.append(kern / np.abs(kern).sum())
    return bank


def grow_ridges(direction: np.ndarray, period: float, rng: np.random.Generator,
                iters: int = 12, n_bins: int = 16) -> np.ndarray:
    """Iterative orientation-adaptive Gabor filtering of noise -> ridge field in [-1, 1]."""
    normal = np.mod(direction + np.pi / 2, np.pi)
    bins = np.round(normal / np.pi * n_bins).astype(np.int32) % n_bins
    bank = _gabor_bank(period, n_bins)
    img = rng.standard_normal(direction.shape).astype(np.float32)
    for _ in range(iters):
        out = np.zeros_like(img)
        for b, kern in enumerate(bank):
            sel = bins == b
            if sel.any():
                out[sel] = cv2.filter2D(img, cv2.CV_32F, kern, borderType=cv2.BORDER_REFLECT)[sel]
        img = np.tanh(3.0 * out / (out.std() + 1e-6))
    return img


@lru_cache(maxsize=64)
def master_print(identity: int, period: float = CANONICAL_PERIOD_PX) -> np.ndarray:
    """The identity's ridge pattern, float32 in [-1, 1] (ridges > 0), MASTER_SIZE square."""
    rng = np.random.default_rng(10_000 + identity)
    direction = _random_orientation(rng, MASTER_SIZE)
    p = period * rng.uniform(0.95, 1.05)
    ridges = grow_ridges(direction, p, rng)
    ridges.setflags(write=False)
    return ridges


def _smooth_field(shape, amp: float, scale: float, rng) -> np.ndarray:
    f = rng.standard_normal(shape).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), scale)
    return f / (np.abs(f).max() + 1e-6) * amp


def contact_impression(identity: int, sample: int = 0, size: int = 400) -> np.ndarray:
    """Ground-truth contact print (uint8, dark ridges on white), elastically distorted."""
    master = master_print(identity)
    rng = np.random.default_rng(identity * 1000 + sample)
    n = MASTER_SIZE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2
    # pressure: contact flattens the fingertip -> mild radial expansion from the centre
    r = np.hypot(xx - c, yy - c) / c
    expand = 1.0 - rng.uniform(0.03, 0.07) * np.clip(r, 0, 1.2)
    dx = _smooth_field((size, size), rng.uniform(3, 7), 40, rng)
    dy = _smooth_field((size, size), rng.uniform(3, 7), 40, rng)
    ox, oy = n / 2 + rng.uniform(-12, 12), n / 2 + rng.uniform(-12, 12)
    mx = ox + (xx - c) * expand + dx
    my = oy + (yy - c) * expand + dy
    ridge = cv2.remap(master, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # elliptical contact area with a ragged edge
    ell = ((xx - c) / (0.42 * size)) ** 2 + ((yy - c * 1.05) / (0.48 * size)) ** 2
    edge = 1.0 + _smooth_field((size, size), 0.08, 12, rng)
    inside = np.clip((edge - ell) * 8, 0, 1)
    # ink: dryness blotches, ridge saturation, sensor noise
    dry = np.clip(1.0 - np.maximum(_smooth_field((size, size), 1.2, 9, rng), 0), 0.15, 1)
    ink = np.clip(0.5 + 0.9 * ridge, 0, 1) * dry * inside
    img = 245 - 215 * ink + rng.normal(0, 6, (size, size))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.6)
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class ContactlessTruth:
    tip: tuple[float, float]
    axis: tuple[float, float]
    finger_width: float
    pprp: float                # nominal ridge period at the finger centre (px)
    mask: np.ndarray


def contactless_photo(identity: int, sample: int = 0, pprp: float = 11.0,
                      size: tuple[int, int] = (480, 640), skin: tuple[int, int, int] | None = None,
                      pose: bool = True) -> tuple[np.ndarray, ContactlessTruth]:
    """BGR photo of a fingertip (tip up) carrying the identity's ridges, plus ground truth."""
    master = master_print(identity)
    rng = np.random.default_rng(identity * 7919 + sample + 1)
    w, h = size
    scale = pprp / CANONICAL_PERIOD_PX                         # image px per master px
    R = 0.31 * w                                               # finger radius (px)
    cx, tip_y = w / 2, 0.08 * h
    cap_c = tip_y + R * 1.05                                   # centre of the tip cap
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx - cx
    # half-width per row: elliptical cap above cap_c, cylinder below
    t = np.clip((cap_c - yy) / (R * 1.05), -1, 1)
    half = np.where(yy < cap_c, R * np.sqrt(np.clip(1 - t**2, 0, 1)), R)
    inside = (np.abs(u) < half) & (yy >= tip_y)
    a = np.clip(u / R, -0.999, 0.999)
    s = R * np.arcsin(a)                                       # arc length on the cylinder
    # vertical foreshortening on the cap (the tip curls away from the camera)
    v = np.where(yy < cap_c, cap_c - (R * 1.05) * np.arcsin(t) * 0.9, yy)
    centre_x = MASTER_SIZE / 2 + rng.uniform(-10, 10)
    centre_y = MASTER_SIZE / 2 + rng.uniform(-10, 10)
    mx = centre_x + s / scale
    my = centre_y + (v - (cap_c + 0.35 * R)) / scale
    ridge = cv2.remap(master, mx.astype(np.float32), my.astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    nz = np.sqrt(np.clip(1 - a**2, 0, 1))                       # surface normal toward camera
    light = 0.35 + 0.65 * nz * (0.85 + 0.15 * np.cos(2.0 * a + rng.uniform(-0.5, 0.5)))
    spec = 0.25 * np.exp(-((a - rng.uniform(-0.3, 0.3)) ** 2) / 0.01) * nz
    contrast = rng.uniform(0.08, 0.14) * nz**1.5                # ridges fade at grazing angles
    lum = light * (1 - contrast * (0.5 + 0.5 * ridge)) + spec
    # distal interphalangeal crease ~1.3 widths below the tip
    crease_y = tip_y + 2.6 * R + rng.uniform(-8, 8)
    lum *= 1 - 0.35 * np.exp(-((yy - crease_y - 6 * np.sin(u / R * 2)) ** 2) / 18.0)
    if skin is None:
        skin = tuple(int(c) for c in rng.choice(
            [(120, 150, 210), (95, 125, 185), (70, 100, 150), (45, 70, 110)]))
    bgr = np.stack([lum * c for c in skin], axis=-1)
    bg = 40 + 50 * (yy / h) + _smooth_field((h, w), 25, 30, rng)
    bg = np.stack([bg * f for f in (1.0, 0.95, 0.9)], axis=-1)
    alpha = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 1.2)[..., None]
    img = alpha * bgr + (1 - alpha) * bg
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.7)
    img += rng.normal(0, 2.5, img.shape)
    tip, axis = (cx, tip_y), (0.0, 1.0)
    mask = (inside * 255).astype(np.uint8)
    if pose:  # in-plane rotation + slight perspective tilt
        ang = rng.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        tilt = rng.uniform(-0.04, 0.04) * w
        dst = np.float32([[tilt, 0], [w - tilt, 0], [w + tilt, h], [-tilt, h]])
        H = cv2.getPerspectiveTransform(src, dst) @ np.vstack([M, [0, 0, 1]])
        img = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)
        p = cv2.perspectiveTransform(np.float32([[tip, (cx, tip_y + 100)]]), H)[0]
        tip = (float(p[0, 0]), float(p[0, 1]))
        d = p[1] - p[0]
        axis = tuple((d / np.linalg.norm(d)).tolist())
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, ContactlessTruth(tip, axis, 2 * R, pprp, mask)


def synthetic_clip(identity: int = 0, n_frames: int = 24, scene_pprp: float = 10.0, downscale: float = 3.0,
                   blur_frames: float = 0.4, seed: int = 0) -> list[np.ndarray]:
    """A handheld 'video' of a fingertip: hand tremor (sub-pixel shifts + tiny rotations),
    lens PSF, area-integrating sensor, noise, and a fraction of motion-blurred frames.
    Each frame shows ~scene_pprp/downscale px per ridge."""
    scene, _ = contactless_photo(identity, sample=1, pprp=scene_pprp, pose=False)
    rng = np.random.default_rng(seed)
    h, w = scene.shape[:2]
    frames = []
    for _ in range(n_frames):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-0.6, 0.6), 1.0)
        M[:, 2] += rng.uniform(-4, 4, 2)
        f = cv2.warpAffine(scene, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rng.random() < blur_frames:                   # a shaky frame
            L = rng.uniform(6, 14)
            k = np.zeros((int(L) | 1, int(L) | 1), np.float32)
            k[k.shape[0] // 2, :] = 1
            R = cv2.getRotationMatrix2D((k.shape[1] / 2 - 0.5, k.shape[0] / 2 - 0.5), rng.uniform(0, 180), 1)
            k = cv2.warpAffine(k, R, k.shape[::-1])
            f = cv2.filter2D(f, -1, k / max(k.sum(), 1e-6))
        f = cv2.GaussianBlur(f, (0, 0), 0.5 * downscale)
        f = cv2.resize(f, (int(w / downscale), int(h / downscale)), interpolation=cv2.INTER_AREA)
        frames.append(np.clip(f + rng.normal(0, 2.5, f.shape), 0, 255).astype(np.uint8))
    return frames
