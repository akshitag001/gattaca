"""Per-image recoverability measurements: PPRP, spectral SNR, orientation certainty,
local ridge clarity, minutiae statistics, and an NFIQ-style aggregate.

Local ridge clarity is the local normalised correlation between the observed
(normalised) image and its Gabor-enhanced version. It is close to 1 where the
image really contains a clean oriented ridge sinusoid, and near 0 where the Gabor
filter is only "hallucinating" ridges out of noise or blur. This is the heat map
shown to users.
"""

from __future__ import annotations

import cv2
import numpy as np

from gattaca.debug import DebugSink, dump
from gattaca.pipeline.ridgefreq import dominant_period
from gattaca.types import EnhanceResult, QualityReport

REF_AREA = 200 * 200          # canonical-scale px: roughly a full contact-print core region
REF_MINUTIAE = 30             # Lin & Kumar report ~29 per print


def measure_pprp(gray: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Pixels-per-ridge-period of the input image and its spectral SNR."""
    period, snr, _ = dominant_period(gray, mask)
    return period, snr


def _lcorr(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    ab = cv2.GaussianBlur(a * b, (0, 0), s)
    return ab / np.sqrt(cv2.GaussianBlur(a * a, (0, 0), s) * cv2.GaussianBlur(b * b, (0, 0), s) + 1e-6)


def periodicity_map(enh: EnhanceResult, s: float) -> np.ndarray:
    """(r(T) - r(T/2)) / 2: local autocorrelation along the ridge normal at one period
    vs half a period. A ridge train gives ~+1 at T and ~-1 at T/2. A single edge, a
    specular highlight bar, a silhouette or noise gives ~0."""
    n = enh.normalized
    h, w = n.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, si = np.cos(enh.orientation), np.sin(enh.orientation)
    r = []
    for frac in (0.5, 1.0):
        d = enh.period * frac
        acc = np.zeros_like(n)
        for sign in (1, -1):
            sh = cv2.remap(n, xx + sign * d * c, yy + sign * d * si, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
            acc += _lcorr(n, sh, s)
        r.append(acc / 2)
    return np.clip((r[1] - r[0]) / 2, 0, 1)


def clarity_map(enh: EnhanceResult, period: float) -> np.ndarray:
    """Local ridge clarity in [0, 1] = Gabor model fit x periodicity x coherence gate."""
    s = max(period, 2.0)
    fit = np.clip(_lcorr(enh.normalized, enh.enhanced, s), 0, 1)
    periodic = np.clip(periodicity_map(enh, s) / 0.6, 0, 1)
    # isotropic band-limited noise can fit its own Gabor output locally, but it is
    # never orientation-coherent over several periods
    gate = np.clip((enh.coherence - 0.2) / 0.4, 0, 1)
    return (fit * periodic * gate * (enh.mask > 0)).astype(np.float32)


def assess(enh: EnhanceResult, minutiae: np.ndarray, pprp: float, spectral_snr: float,
           debug: DebugSink | None = None) -> QualityReport:
    m = enh.mask > 0
    area = int(m.sum())
    period = float(np.median(enh.period[m])) if area else 9.0
    cmap = clarity_map(enh, period)
    dump(debug, "quality_clarity", cmap)
    clarity = float(cmap[m].mean()) if area else 0.0
    coherence = float(enh.coherence[m].mean()) if area else 0.0
    mq = float(minutiae["quality"].mean()) if len(minutiae) else 0.0
    # NFIQ-style aggregate: geometric blend of clarity, certainty, area and minutiae support
    nfiq = 100.0 * (clarity ** 0.4) * (coherence ** 0.3) \
        * min(1.0, area / REF_AREA) ** 0.15 * min(1.0, len(minutiae) / REF_MINUTIAE) ** 0.15
    if not np.isfinite(pprp):
        nfiq = 0.0
    return QualityReport(pprp=float(pprp) if np.isfinite(pprp) else 0.0,
                         spectral_snr=float(spectral_snr), coherence=coherence, clarity=clarity,
                         area_px=area, minutiae_count=int(len(minutiae)), minutiae_quality=mq,
                         nfiq_like=float(nfiq), clarity_map=cmap)
