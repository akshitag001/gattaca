"""User-facing ridge-evidence map: a heat-map of local ridge clarity blended over the
user's own photo. It shows WHERE and HOW MUCH ridge detail is exposed.

Guardrails (docs/responsible_use.md, CLAUDE.md section 0): the enhanced ridge pattern
itself is deliberately NOT drawn. A crisp binarised ridge overlay registered to the
photo would amount to a clean print, which section 0 forbids, so the map encodes
exposure only as clarity. The output is downscaled to at most MAX_SIDE px and
watermarked with the disclaimer. No binarised, skeleton or minutiae-coordinate output
exists on this path.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca import DISCLAIMER

MAX_SIDE = 640
BAND_COLORS = {"HIGH": (40, 40, 220), "MEDIUM": (0, 170, 240), "LOW": (80, 170, 60)}


def _to_source(layer: np.ndarray, fa, shape) -> np.ndarray:
    roi = fa.roi
    an = fa.analysis
    small = cv2.resize(layer, (roi.image.shape[1], roi.image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.warpPerspective(small, roi.to_source, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _wrap_text(text: str, width: int, scale: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def render_zoom_callout(img: np.ndarray, fingers: list, scores: list, max_width: int = 900) -> np.ndarray:
    """Magnifier callout: the user's own photo with the fingertip region optically zoomed.

    The circle shows the SAME PIXELS the person already has, scaled up, with the exposure
    heat blended over them. It is an optical zoom of their own upload - not an enhanced,
    binarised or matcher-ready print (that remains out of scope, see module docstring).
    Its job is to make "this much is already visible" immediately obvious.
    """
    if not fingers:
        return render_evidence(img, fingers, scores)
    base = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h0, w0 = base.shape[:2]
    fa, sc = fingers[0], scores[0]
    an = fa.analysis
    cm = an.quality.clarity_map if an.quality.clarity_map is not None else np.zeros_like(an.canonical, np.float32)
    cm = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 3 * 9.0)
    skin = cv2.GaussianBlur((an.canonical_mask > 0).astype(np.float32), (0, 0), 4)
    heat_full = _to_source(cm * skin, fa, (h0, w0))

    det = fa.detection
    tip, axis = np.array(det.tip, np.float64), np.array(det.axis, np.float64)
    centre = tip + axis * 0.45 * det.width
    side = max(int(0.78 * det.width), 24)
    x0 = int(np.clip(centre[0] - side / 2, 0, max(w0 - side, 0)))
    y0 = int(np.clip(centre[1] - side / 2, 0, max(h0 - side, 0)))
    side = min(side, w0 - x0, h0 - y0)
    crop = base[y0:y0 + side, x0:x0 + side].astype(np.float32)
    hcrop = heat_full[y0:y0 + side, x0:x0 + side]

    # photo on the left, magnifier in its own panel on the right (no overlap)
    s = min(max_width * 0.58 / w0, 1.4)
    cw, ch = int(w0 * s), int(h0 * s)
    photo = cv2.resize(base, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32)
    R = int(min(ch * 0.42, max_width * 0.2))
    gap, pad = 26, 20
    W = cw + gap + 2 * R + pad
    canvas = np.full((ch, W, 3), 16, np.float32)
    canvas[:, :cw] = photo
    cx, cy = cw + gap + R, ch // 2

    mag = cv2.resize(crop, (2 * R, 2 * R), interpolation=cv2.INTER_CUBIC)
    magh = cv2.resize(hcrop, (2 * R, 2 * R), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap(np.clip(magh * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO).astype(np.float32)
    a = (0.3 * np.sqrt(np.clip(magh, 0, 1)))[..., None]        # light, so the skin itself stays readable
    mag = mag * (1 - a) + col * a

    lens = np.zeros((2 * R, 2 * R), np.uint8)
    cv2.circle(lens, (R, R), R - 2, 255, cv2.FILLED, cv2.LINE_AA)
    ys, xs = cy - R, cx - R
    region = canvas[ys:ys + 2 * R, xs:xs + 2 * R]
    m3 = (cv2.GaussianBlur(lens, (0, 0), 1.2).astype(np.float32) / 255.0)[..., None]
    canvas[ys:ys + 2 * R, xs:xs + 2 * R] = region * (1 - m3) + mag * m3

    out = np.clip(canvas, 0, 255).astype(np.uint8)
    accent = BAND_COLORS[sc.band]
    bx0, by0 = int(x0 * s), int(y0 * s)
    bx1, by1 = int((x0 + side) * s), int((y0 + side) * s)
    cv2.rectangle(out, (bx0, by0), (bx1, by1), accent, 2, cv2.LINE_AA)
    for corner in ((bx1, by0), (bx1, by1)):          # connectors run rightwards to the lens
        v = np.array([cx - corner[0], cy - corner[1]], np.float64)
        n = np.linalg.norm(v) + 1e-6
        end = (np.array(corner) + v / n * (n - R)).astype(int)
        cv2.line(out, corner, tuple(end), accent, 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), R - 2, accent, 3, cv2.LINE_AA)

    label = f"{sc.band}  ~{an.quality.pprp:.1f} px per ridge"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    lx, ly = int(np.clip(cx - tw / 2, 8, out.shape[1] - tw - 14)), int(np.clip(cy + R + 32, th + 10, ch - 10))
    cv2.rectangle(out, (lx - 8, ly - th - 8), (lx + tw + 8, ly + 8), accent, cv2.FILLED)
    cv2.putText(out, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    lines = _wrap_text(DISCLAIMER, out.shape[1] - 12, 0.4)
    bar = np.full((14 * len(lines) + 8, out.shape[1], 3), 24, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (6, 16 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([out, bar])


def render_evidence(img: np.ndarray, fingers: list, scores: list) -> np.ndarray:
    base = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = base.shape[:2]
    heat = np.zeros((h, w), np.float32)
    for fa in fingers:
        an = fa.analysis
        cm = an.quality.clarity_map if an.quality.clarity_map is not None else np.zeros_like(an.canonical, np.float32)
        # smooth well beyond the ridge period so the overlay cannot carry ridge geometry
        cm = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 3 * 9.0)
        skin = cv2.GaussianBlur((an.canonical_mask > 0).astype(np.float32), (0, 0), 4)
        heat = np.maximum(heat, _to_source(cm * skin, fa, (h, w)))
    heat_u8 = np.clip(heat * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO).astype(np.float32)
    alpha = (0.6 * np.sqrt(np.clip(heat, 0, 1)))[..., None]
    out = base.astype(np.float32) * (1 - alpha) + colored * alpha
    s = MAX_SIDE / max(h, w)
    if s < 1:
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        s = 1.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    for fa, sc in zip(fingers, scores):
        tip = (np.array(fa.detection.tip) * s).astype(int)
        label = f"{sc.band}  {fa.analysis.quality.minutiae_count} features"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = int(np.clip(tip[0] - tw // 2, 2, out.shape[1] - tw - 2)), int(max(th + 6, tip[1] - 10))
        cv2.rectangle(out, (x - 3, y - th - 4), (x + tw + 3, y + 4), BAND_COLORS[sc.band], cv2.FILLED)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    lines = _wrap_text(DISCLAIMER, out.shape[1] - 12, 0.4)
    bar = np.full((14 * len(lines) + 8, out.shape[1], 3), 24, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (6, 16 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([out, bar])


def encode_jpeg(img: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
