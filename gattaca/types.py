"""Shared data types for the pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Minutiae are a structured array so stages can vectorise over them.
# angle: radians in [0, 2*pi), image coordinates (x right, y down).
MINUTIA_DTYPE = np.dtype(
    [("x", "f4"), ("y", "f4"), ("angle", "f4"), ("kind", "u1"), ("quality", "f4")]
)
ENDING = 1
BIFURCATION = 3


def empty_minutiae() -> np.ndarray:
    return np.zeros(0, dtype=MINUTIA_DTYPE)


@dataclass
class FingertipDetection:
    """A fingertip located in a full image."""

    tip: tuple[float, float]          # (x, y) of the fingertip extreme
    axis: tuple[float, float]         # unit vector pointing from the tip towards the palm
    width: float                      # finger width in px near the distal phalanx
    score: float                      # detector confidence in [0, 1]
    mask: np.ndarray | None = None    # uint8 mask of this finger (full image), if known
    source: str = "classical"


@dataclass
class FingerROI:
    """An upright, rectified crop of the distal phalanx (tip at the top)."""

    image: np.ndarray                 # grayscale uint8
    mask: np.ndarray                  # uint8 {0,255}
    to_source: np.ndarray             # 3x3 homography ROI px -> source image px
    width_px: float                   # finger width in source px
    color: np.ndarray | None = None   # BGR crop, for visualisation only


@dataclass
class EnhanceResult:
    enhanced: np.ndarray              # float32, ridges > 0, valleys < 0, 0 outside mask
    orientation: np.ndarray           # float32 ridge-normal angle in [0, pi)
    coherence: np.ndarray             # float32 [0, 1]
    period: np.ndarray                # float32 local ridge period (px)
    mask: np.ndarray                  # uint8 {0,255} usable ridge area
    normalized: np.ndarray            # float32 pre-Gabor normalised image


@dataclass
class QualityReport:
    pprp: float                       # pixels per ridge period in the *input* image
    spectral_snr: float               # ridge-band peak / background spectral energy
    coherence: float                  # mean orientation certainty in the ridge area
    clarity: float                    # mean local ridge clarity [0, 1]
    area_px: int                      # usable ridge area (canonical-scale px)
    minutiae_count: int
    minutiae_quality: float
    nfiq_like: float                  # 0-100 aggregate, NFIQ-style
    clarity_map: np.ndarray | None = field(default=None, repr=False)

    def features(self) -> dict[str, float]:
        """Inference-time features used by the calibrated recoverability model."""
        return {
            "log2_pprp": float(np.log2(max(self.pprp, 0.5))),
            "clarity": float(self.clarity),
            "coherence": float(self.coherence),
            "log_minutiae": float(np.log1p(self.minutiae_count)),
            "spectral_snr": float(np.log1p(self.spectral_snr)),
        }
