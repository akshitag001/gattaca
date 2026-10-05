"""Render the landing-page pipeline-reveal frames from REAL pipeline output.

Every frame is genuine OpenCV output from gattaca's own stages, run on a *synthetic*
hand (gattaca.synth). No real person's photo or biometrics are used anywhere in the
public demo - that is the point of the project, and it keeps the marketing page inside
the same consent rule as the tool (CLAUDE.md section 0).

    python scripts/make_landing_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gattaca import synth  # noqa: E402
from gattaca.analyze import analyze_photo  # noqa: E402
from gattaca.debug import to_u8  # noqa: E402
from gattaca.pipeline.detect import detect_fingertips  # noqa: E402
from gattaca.pipeline.extract import binarize, thin  # noqa: E402
from gattaca.types import ENDING  # noqa: E402

OUT = ROOT / "service" / "web" / "assets"
W, H = 900, 1200          # portrait frames for the sticky panel


def fit(img: np.ndarray, w: int = W, h: int = H, bg=(10, 13, 18)) -> np.ndarray:
    img = img if img.ndim == 3 else cv2.cvtColor(to_u8(img), cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), bg, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, default=None,
                    help="a REAL photo of your own hand (see docs/image_brief.md). "
                         "Omitted: a synthetic hand is used.")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.photo:
        photo = cv2.imdecode(np.fromfile(str(a.photo), np.uint8), cv2.IMREAD_COLOR)
        if photo is None:
            sys.exit(f"cannot read {a.photo}")
        h, w = photo.shape[:2]
        if max(h, w) > 2400:
            s_ = 2400 / max(h, w)
            photo = cv2.resize(photo, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)
        print(f"source: {a.photo} (real photo)")
    else:
        photo, _ = synth.contactless_photo(3, sample=1, pprp=10)
        print("source: synthetic hand (pass --photo for a real one)")
    frames: dict[str, np.ndarray] = {}

    frames["01_photo"] = photo

    dets = detect_fingertips(photo)
    vis = photo.copy()
    h0, w0 = vis.shape[:2]
    accent = (61, 163, 232)   # BGR for #e8a33d - warm amber reticle (matches the UI)
    for d in dets:
        tip, axis = np.array(d.tip), np.array(d.axis)
        perp = np.array([axis[1], -axis[0]])
        half, length = 0.62 * d.width, 1.45 * d.width
        start = tip - axis * 0.22 * d.width
        corners = [start - perp * half, start + perp * half,
                   start + axis * length + perp * half, start + axis * length - perp * half]
        pts = [np.clip(c, [4, 4], [w0 - 5, h0 - 5]).astype(int) for c in corners]
        arm = int(0.26 * d.width)
        for i, p in enumerate(pts):                      # L-shaped corner brackets
            for nb in (pts[(i + 1) % 4], pts[(i - 1) % 4]):
                v = nb - p
                n = np.linalg.norm(v)
                if n > 1:
                    q = (p + v / n * arm).astype(int)
                    cv2.line(vis, tuple(p), tuple(q), accent, 3, cv2.LINE_AA)
    cv2.putText(vis, "FINGERTIP ROI LOCKED", (18, h0 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, accent, 2, cv2.LINE_AA)
    frames["02_detect"] = vis

    found = analyze_photo(photo, max_fingers=1)
    if not found:
        sys.exit("no fingertip detected in that photo - try a tighter, sharper, better-lit shot")
    fa = found[0]
    an = fa.analysis
    frames["03_roi"] = cv2.cvtColor(to_u8(an.canonical), cv2.COLOR_GRAY2BGR)
    frames["04_enhance"] = cv2.cvtColor(to_u8(an.enh.enhanced), cv2.COLOR_GRAY2BGR)

    period = float(np.median(an.enh.period[an.enh.mask > 0])) if (an.enh.mask > 0).any() else 9.0
    skel = thin(binarize(an.enh, period))
    mv = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    mv[skel > 0] = (140, 140, 140)
    for m in an.minutiae:
        c = (90, 90, 255) if m["kind"] == ENDING else (120, 230, 120)
        p = (int(m["x"]), int(m["y"]))
        cv2.circle(mv, p, 6, c, 2, cv2.LINE_AA)
        q = (int(m["x"] + 14 * np.cos(m["angle"])), int(m["y"] + 14 * np.sin(m["angle"])))
        cv2.line(mv, p, q, c, 2, cv2.LINE_AA)
    frames["05_minutiae"] = mv

    base = cv2.cvtColor(to_u8(an.canonical), cv2.COLOR_GRAY2BGR).astype(np.float32)
    cm = an.quality.clarity_map
    cm = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 24) * (an.canonical_mask > 0)
    heat = cv2.applyColorMap(np.clip(cm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO).astype(np.float32)
    a = (0.72 * np.sqrt(np.clip(cm, 0, 1)))[..., None]
    frames["06_heat"] = to_u8(base * (1 - a) + heat * a)

    # hero: heat map composited back onto the FULL photo (fills a 3:4 frame, no letterbox)
    from gattaca.evidence import _to_source
    hero = photo.copy().astype(np.float32)
    hm = _to_source(cv2.GaussianBlur(an.quality.clarity_map.astype(np.float32), (0, 0), 24)
                    * (an.canonical_mask > 0), fa, photo.shape[:2])
    hcol = cv2.applyColorMap(np.clip(hm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO).astype(np.float32)
    ha = (0.68 * np.sqrt(np.clip(hm, 0, 1)))[..., None]
    frames["00_hero"] = to_u8(hero * (1 - ha) + hcol * ha)

    for name, img in frames.items():
        path = OUT / f"{name}.jpg"
        cv2.imwrite(str(path), fit(img), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {path.relative_to(ROOT)}  {img.shape[1]}x{img.shape[0]}")
    print(f"{len(frames)} frames -> {OUT.relative_to(ROOT)}  (source above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
