"""Multi-frame, multi-finger consolidation report: how much MORE does a video expose
than a single photo?

This answers the core research question from CLAUDE.md section 4 - the single-vs-fused
recoverability delta - across every finger visible in a clip. For each finger it tracks
the fingertip across frames, selects and fuses the best ones (the handheld multi-frame
super-resolution principle), and compares the recoverability of the best single frame
against the fused result.

Scope (CLAUDE.md section 0): the input is a video of the user's OWN hand or a consented
research subject. This is a MEASUREMENT of exposure, not a targeting tool: it does not
pick a person out of surveillance footage, run face/person recognition, identify anyone,
or emit an extractable template. It reports risk bands and the fusion delta, so people
understand that posting a clip is riskier than posting a photo - and can protect
accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from gattaca.agent import tools
from gattaca.analyze import FingerAssessment, analyze_photo
from gattaca.debug import to_u8
from gattaca.evidence import BAND_COLORS
from gattaca.pipeline.detect import detect_fingertips
from gattaca.pipeline.enhance import EnhanceParams
from gattaca.scoring import Recoverability, score_finger
from gattaca.video.fuse import fuse
from gattaca.video.select import score_frame, select_best


@dataclass
class FingerConsolidation:
    label: str
    single: Recoverability
    fused: Recoverability
    single_fa: FingerAssessment
    fused_fa: FingerAssessment
    frames_used: int
    best_frame: int

    @property
    def delta_p(self) -> float:
        return self.fused.probability - self.single.probability

    @property
    def minutiae_gain(self) -> int:
        return self.fused_fa.analysis.quality.minutiae_count - self.single_fa.analysis.quality.minutiae_count


@dataclass
class ConsolidationReport:
    fingers: list[FingerConsolidation]
    chart: np.ndarray
    n_frames: int

    def summary(self) -> dict:
        if not self.fingers:
            return {"fingers": 0, "note": "no finger could be tracked through the clip"}
        worst = max(self.fingers, key=lambda f: f.fused.probability)
        return {
            "fingers": len(self.fingers), "frames": self.n_frames,
            "worst_finger": {"label": worst.label, "single_band": worst.single.band,
                             "fused_band": worst.fused.band, "single_p": round(worst.single.probability, 3),
                             "fused_p": round(worst.fused.probability, 3)},
            "max_delta_p": round(max(f.delta_p for f in self.fingers), 3),
            "fingers_promoted": sum(1 for f in self.fingers
                                    if _rank(f.fused.band) > _rank(f.single.band)),
            "per_finger": [{"label": f.label, "single": f.single.band, "fused": f.fused.band,
                            "delta_p": round(f.delta_p, 3), "minutiae_gain": f.minutiae_gain,
                            "frames_used": f.frames_used} for f in self.fingers],
        }


def _rank(band: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}[band]


def _heat_panel(fa: FingerAssessment, w: int, h: int, title: str, band: str) -> np.ndarray:
    an = fa.analysis
    base = cv2.cvtColor(to_u8(an.canonical), cv2.COLOR_GRAY2BGR).astype(np.float32)
    cm = an.quality.clarity_map
    if cm is not None:
        cm = cv2.GaussianBlur(cm.astype(np.float32), (0, 0), 27) * (an.canonical_mask > 0)
        heat = cv2.applyColorMap(np.clip(cm * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO).astype(np.float32)
        a = (0.6 * np.sqrt(np.clip(cm, 0, 1)))[..., None]
        base = base * (1 - a) + heat * a
    img = to_u8(base)
    ih, iw = img.shape[:2]
    s = min((w - 12) / iw, (h - 56) / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    panel = np.full((h, w, 3), 24, np.uint8)
    y0, x0 = 34 + (h - 56 - r.shape[0]) // 2, 6 + (w - 12 - r.shape[1]) // 2
    panel[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (8, h - 26), (8 + 78, h - 6), BAND_COLORS[band], cv2.FILLED)
    cv2.putText(panel, band, (14, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_report(fingers: list[FingerConsolidation]) -> np.ndarray:
    if not fingers:
        img = np.full((120, 640, 3), 24, np.uint8)
        cv2.putText(img, "No finger could be tracked through the clip.", (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return img
    pw, ph = 320, 300
    rows = []
    for f in fingers[:4]:
        left = _heat_panel(f.single_fa, pw, ph, f"{f.label}: best single frame", f.single.band)
        right = _heat_panel(f.fused_fa, pw, ph, f"{f.label}: fused ({f.frames_used} frames)", f.fused.band)
        mid = np.full((ph, 150, 3), 18, np.uint8)
        arrow = "+{:.2f}".format(f.delta_p) if f.delta_p >= 0 else "{:.2f}".format(f.delta_p)
        cv2.arrowedLine(mid, (18, ph // 2), (132, ph // 2), (120, 180, 255), 2, tipLength=0.25)
        cv2.putText(mid, "P " + arrow, (20, ph // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(mid, f"{f.minutiae_gain:+d} min", (22, ph // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        rows.append(np.hstack([left, mid, right]))
    grid = np.vstack(rows)
    footer = np.full((54, grid.shape[1], 3), 16, np.uint8)
    promoted = sum(1 for f in fingers if _rank(f.fused.band) > _rank(f.single.band))
    cv2.putText(footer, f"Multi-frame fusion vs best single frame.  {promoted}/{len(fingers)} finger(s) moved "
                "to a higher risk band.", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(footer, "GATTACA exposure measurement - a video reveals more than a photo. Not an identification tool.",
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([grid, footer])


def _assess_crop(crop: np.ndarray, params: EnhanceParams, pprp_override: float | None = None) -> FingerAssessment | None:
    fa = analyze_photo(crop, params=params, max_fingers=1)
    if not fa:
        return None
    fa = fa[0]
    if pprp_override is not None:
        import dataclasses
        fa.analysis.quality = dataclasses.replace(fa.analysis.quality, pprp=pprp_override)
    return fa


def consolidate_video(frames: list[np.ndarray], *, k_fuse: int = 6, max_fingers: int = 4,
                      params: EnhanceParams = EnhanceParams()) -> ConsolidationReport:
    """Per-finger single-vs-fused recoverability across a clip of one's own/consented hand."""
    if not frames:
        return ConsolidationReport([], render_report([]), 0)
    ref = int(np.argmax([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f,
                                        cv2.CV_32F).var() for f in frames]))
    dets = detect_fingertips(frames[ref])[:max_fingers]
    out: list[FingerConsolidation] = []
    for i, det in enumerate(dets):
        box = tools.finger_box(det, frames[ref].shape)
        _, (boxes, motion) = tools.tool_track(frames, box, ref)
        scores = [score_frame(frames[j], boxes[j], j, float(motion[j])) for j in range(len(frames))
                  if boxes[j] is not None]
        if not scores:
            continue
        sel = select_best(scores, k=k_fuse)
        if len(sel) < 2:
            continue
        best = max((s for s in scores if s.index in sel), key=lambda s: s.score)
        size = (int(boxes[best.index][2]), int(boxes[best.index][3]))
        def crop_at(j):
            bx, by, bw, bh = boxes[j]
            return cv2.getRectSubPix(frames[j], size, (bx + bw / 2, by + bh / 2))
        single_fa = _assess_crop(crop_at(best.index), params)
        if single_fa is None:
            continue
        capture_pprp = single_fa.analysis.pprp
        crops = [crop_at(j) for j in sel]
        order = sorted(range(len(sel)), key=lambda t: 0 if sel[t] == best.index else 1)
        fused, info = fuse([crops[t] for t in order], 0, scale=2)
        fused_fa = _assess_crop(fused, params, pprp_override=capture_pprp)
        if fused_fa is None:
            continue
        label = det.source.split(":")[-1] if ":" in det.source else f"finger {i + 1}"
        out.append(FingerConsolidation(label, score_finger(single_fa), score_finger(fused_fa),
                                       single_fa, fused_fa, info["frames_used"], best.index))
    out.sort(key=lambda f: -f.fused.probability)
    return ConsolidationReport(out, render_report(out), len(frames))
