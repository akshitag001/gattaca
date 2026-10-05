"""End-to-end analysis of a photo (contactless) or a ground-truth contact print.

photo -> detect -> oriented ROI -> rectify (+ cylinder unwrap) -> measure PPRP ->
resample to the canonical period -> enhance -> extract minutiae -> quality.

PPRP is measured twice:
* **spectral**: dominant ridge frequency of the ROI. Only works if ridges are visible.
* **geometric**: finger width / typical ridge count across a fingertip. It shows
  the effective resolution even when blur hides the ridges, which separates
  "too few pixels" from "enough pixels but blurred".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gattaca import CANONICAL_PERIOD_PX
from gattaca.debug import DebugSink, dump
from gattaca.pipeline import detect as det_mod
from gattaca.pipeline.enhance import EnhanceParams, enhance, to_gray
from gattaca.pipeline.extract import extract_minutiae
from gattaca.pipeline.quality import assess, measure_pprp
from gattaca.pipeline.rectify import rectify
from gattaca.pipeline.roi import oriented_patch
from gattaca.types import EnhanceResult, FingerROI, FingertipDetection, QualityReport

# An adult fingertip is ~14-20 mm wide at the distal phalanx: ~36 ridge periods of 0.45 mm.
RIDGES_ACROSS_FINGER = 36.0
MAX_UPSCALE = 4.0


@dataclass
class PrintAnalysis:
    pprp: float                    # PPRP used for scale normalisation
    pprp_spectral: float           # nan if no ridge peak
    pprp_geometric: float          # nan for contact prints
    spectral_snr: float
    scale: float                   # canonical / pprp
    enh: EnhanceResult
    minutiae: np.ndarray           # canonical-scale coordinates
    quality: QualityReport
    canonical: np.ndarray          # canonical-scale grayscale ROI
    canonical_mask: np.ndarray


@dataclass
class FingerAssessment:
    detection: FingertipDetection
    roi: FingerROI
    analysis: PrintAnalysis
    label: str = ""
    extra: dict = field(default_factory=dict)


def choose_pprp(spectral: float, geometric: float) -> float:
    """Trust the spectral estimate when it agrees with geometry; otherwise geometry."""
    if np.isfinite(spectral) and (not np.isfinite(geometric) or 0.5 < spectral / geometric < 2.0):
        return spectral
    if np.isfinite(geometric):
        return geometric
    return CANONICAL_PERIOD_PX


def analyze_print(gray: np.ndarray, mask: np.ndarray, *, pprp_geometric: float = float("nan"),
                  params: EnhanceParams = EnhanceParams(), pprp_override: float | None = None,
                  debug: DebugSink | None = None) -> PrintAnalysis:
    spectral, snr = measure_pprp(gray, mask)
    pprp = pprp_override if pprp_override else choose_pprp(spectral, pprp_geometric)
    scale = float(np.clip(CANONICAL_PERIOD_PX / max(pprp, 1e-3), 0.2, MAX_UPSCALE))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    canon = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)
    cmask = cv2.resize(mask, (canon.shape[1], canon.shape[0]), interpolation=cv2.INTER_NEAREST)
    dump(debug, "canonical_roi", canon)
    enh = enhance(canon, cmask, params=params, debug=debug)
    minutiae = extract_minutiae(enh, debug=debug)
    q = assess(enh, minutiae, pprp, snr, debug=debug)
    return PrintAnalysis(pprp, spectral, pprp_geometric, snr, scale, enh, minutiae, q, canon, cmask)


def contact_mask(gray: np.ndarray) -> np.ndarray:
    """Foreground of an inked/live-scan contact print (dark ridges on light background)."""
    var = cv2.GaussianBlur((gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 4)) ** 2,
                           (0, 0), 8)
    _, m = cv2.threshold(cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        m = np.where(lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax()), 255, 0).astype(np.uint8)
    return m


def analyze_contact(img: np.ndarray, debug: DebugSink | None = None) -> PrintAnalysis:
    gray = to_gray(img)
    return analyze_print(gray, contact_mask(gray), debug=debug)


def analyze_finger(img: np.ndarray, det: FingertipDetection, *, params: EnhanceParams = EnhanceParams(),
                   unwrap: bool = True, debug: DebugSink | None = None) -> FingerAssessment:
    patch, mpatch, M = oriented_patch(img, det)
    dump(debug, "roi_patch", patch)
    roi = rectify(patch, mpatch, M, det.width, unwrap=unwrap, debug=debug)
    # rectified width is measured in source px, so geometric PPRP refers to the photo
    geometric = det.width / RIDGES_ACROSS_FINGER
    an = analyze_print(roi.image, roi.mask, pprp_geometric=geometric, params=params, debug=debug)
    return FingerAssessment(det, roi, an, label=det.source)


def analyze_photo(img: np.ndarray, *, backend: str = "auto", max_fingers: int = 5,
                  params: EnhanceParams = EnhanceParams(), debug: DebugSink | None = None) -> list[FingerAssessment]:
    """Assess every detected fingertip; most exposed first."""
    dets = det_mod.detect_fingertips(img, backend=backend, debug=debug)[:max_fingers]
    out = []
    for i, d in enumerate(dets):
        sub = None if debug is None else DebugSink(debug.dir, prefix=f"f{i}_")
        try:
            out.append(analyze_finger(img, d, params=params, debug=sub))
        except (cv2.error, ValueError, IndexError):
            continue  # a degenerate candidate must not sink the whole photo
    return sorted(out, key=lambda f: -f.analysis.quality.nfiq_like)
