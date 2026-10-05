"""Calibrated recoverability: features -> P(identifiable) -> HIGH / MEDIUM / LOW.

The probability comes from a logistic model fitted by `study/calibrate.py` on the
measurement study. The target is "rank-1 identification against the gallery
succeeded", using only features available at inference time (no ground truth).
The calibration file also carries bootstrap replicates of the coefficients.
**Confidence** is the fraction of replicates that put the image in the same band
as the point estimate.

Distances are never the model input. They are derived for the user message from
the measured PPRP and an explicit camera model (see docs/recoverability_metric.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from gattaca import RIDGE_PERIOD_MM
from gattaca.types import QualityReport

CALIB_DIR = Path(__file__).resolve().parent / "calibration"
DEFAULT_CALIB = CALIB_DIR / "default.json"

# (label, horizontal px, 35mm-equivalent focal length)
CAMERAS = [
    ("12 MP main camera (26 mm)", 4032, 26.0),
    ("48 MP main camera, full-res (26 mm)", 8064, 26.0),
    ("3x telephoto (77 mm, 12 MP)", 4032, 77.0),
    ("5x telephoto (120 mm, 12 MP)", 4032, 120.0),
]
SOCIAL_WIDTH = 1080   # a typical platform resize


def pprp_at_distance(distance_mm: float, px_wide: int, focal_eq_mm: float) -> float:
    """PPRP of a fingertip `distance_mm` from a camera (36 mm-wide equivalent sensor)."""
    field_mm = distance_mm * 36.0 / focal_eq_mm
    return RIDGE_PERIOD_MM * px_wide / field_mm


def distance_for_pprp(pprp: float, px_wide: int, focal_eq_mm: float) -> float:
    return RIDGE_PERIOD_MM * px_wide * focal_eq_mm / (36.0 * max(pprp, 1e-6))


@dataclass
class Calibration:
    source: str
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    boot: np.ndarray                     # (B, n_features + 1) bootstrap [coef..., intercept]
    high: float = 0.7
    low: float = 0.3
    curve: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIB) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(d["source"], d["features"], np.array(d["mean"]), np.array(d["std"]),
                   np.array(d["coef"]), float(d["intercept"]), np.array(d.get("boot", [])),
                   d["bands"]["high"], d["bands"]["low"], d.get("curve", []), d.get("thresholds", {}))

    @classmethod
    def prior(cls) -> "Calibration":
        """Hand-set fallback used only until a study has produced default.json."""
        feats = ["log2_pprp", "clarity", "coherence", "log_minutiae", "spectral_snr"]
        return cls("uncalibrated-prior", feats, np.array([2.0, 0.4, 0.5, 3.0, 4.0]),
                   np.array([1.0, 0.2, 0.2, 1.0, 2.0]), np.array([2.0, 1.5, 0.5, 1.0, 0.5]), -0.5,
                   np.zeros((0, 6)))

    def _logit(self, x: np.ndarray, coef: np.ndarray, b: float) -> float:
        z = (x - self.mean) / self.std
        return float(1 / (1 + np.exp(-(z @ coef + b))))

    def band(self, p: float) -> str:
        return "HIGH" if p >= self.high else "LOW" if p < self.low else "MEDIUM"

    def predict(self, feats: dict[str, float]) -> tuple[float, float]:
        x = np.array([feats[f] for f in self.features], np.float64)
        p = self._logit(x, self.coef, self.intercept)
        if len(self.boot):
            ps = [self._logit(x, r[:-1], r[-1]) for r in self.boot]
            conf = float(np.mean([self.band(q) == self.band(p) for q in ps]))
        else:  # no replicates: distance from the nearest band edge
            edge = min(abs(p - self.high), abs(p - self.low))
            conf = float(np.clip(0.5 + edge, 0.5, 0.95))
        return p, conf


@lru_cache(maxsize=4)
def load_calibration(path: str | None = None) -> Calibration:
    p = Path(path) if path else DEFAULT_CALIB
    return Calibration.load(p) if p.exists() else Calibration.prior()


@dataclass
class Recoverability:
    band: str
    probability: float
    confidence: float
    pprp: float
    limiting_factor: str
    explanation: str
    distances: list[dict]
    calibration_source: str
    metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


def limiting_factor(q: QualityReport, pprp_spectral: float, pprp_geometric: float,
                    calib: Calibration, roi_mean: float | None = None) -> str:
    p50 = calib.thresholds.get("pprp_50", 4.0)
    res = pprp_geometric if np.isfinite(pprp_geometric) else q.pprp
    if res < p50:
        return "resolution"
    if roi_mean is not None and (roi_mean < 45 or roi_mean > 225):
        return "lighting"
    if not np.isfinite(pprp_spectral) or q.clarity < 0.35:
        return "focus/blur"
    if q.area_px < 120 * 120:
        return "coverage"
    return "none"


_ADVICE = {
    "resolution": "There are too few pixels per ridge for the pattern to be resolved.",
    "focus/blur": "There are enough pixels, but blur or motion has washed out the ridges.",
    "lighting": "Exposure or contrast is hiding the ridges.",
    "coverage": "Only a small part of the fingertip is visible.",
    "none": "The fingertips are sharp, well lit and large in the frame.",
}


def explain(band: str, factor: str, q: QualityReport, distances: list[dict]) -> str:
    n = q.minutiae_count
    main = distances[0]
    head = {
        "HIGH": f"Fingerprint ridges are clearly recoverable from this photo: we found about {n} "
                f"ridge features at {q.pprp:.1f} px per ridge. Crop, blur or downscale your fingertips before posting.",
        "MEDIUM": f"Partial ridge detail is exposed (~{n} ridge features at {q.pprp:.1f} px per ridge). "
                  "A motivated adversary with several frames might recover more. Consider blurring your fingertips.",
        "LOW": f"Little usable ridge detail is exposed ({q.pprp:.1f} px per ridge). "
               "Low risk from this image alone.",
    }[band]
    tail = (f" {_ADVICE[factor]} For reference, this detail level matches a hand about "
            f"{main['distance_cm']:.0f} cm from a {main['camera']}.")
    return head + tail


def score(q: QualityReport, *, pprp_spectral: float = float("nan"), pprp_geometric: float = float("nan"),
          roi_mean: float | None = None, calib: Calibration | None = None) -> Recoverability:
    calib = calib or load_calibration()
    feats = q.features()
    p, conf = calib.predict(feats)
    if q.pprp <= 0 or q.minutiae_count == 0:
        p, conf = 0.0, max(conf, 0.9)
    band = calib.band(p)
    factor = limiting_factor(q, pprp_spectral, pprp_geometric, calib, roi_mean)
    distances = [{"camera": name, "distance_cm": distance_for_pprp(max(q.pprp, 0.5), px, f) / 10}
                 for name, px, f in CAMERAS]
    metrics = {"pprp": q.pprp, "pprp_spectral": pprp_spectral, "pprp_geometric": pprp_geometric,
               "clarity": q.clarity, "coherence": q.coherence, "minutiae_count": q.minutiae_count,
               "nfiq_like": q.nfiq_like, "area_px": q.area_px}
    metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
    return Recoverability(band, p, conf, q.pprp, factor, explain(band, factor, q, distances),
                          distances, calib.source, metrics)


def score_finger(fa) -> Recoverability:
    """Score a `gattaca.analyze.FingerAssessment`."""
    an = fa.analysis
    m = an.canonical_mask > 0
    roi_mean = float(an.canonical[m].mean()) if m.any() else None
    return score(an.quality, pprp_spectral=an.pprp_spectral, pprp_geometric=an.pprp_geometric,
                 roi_mean=roi_mean)
