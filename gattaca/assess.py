"""Top-level assessment of one image: every fingertip scored, the most exposed one
reported, and the evidence map rendered. `summary()` is JSON-safe and contains
no coordinates, minutiae or ridge images, so it is what leaves the process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gattaca import DISCLAIMER
from gattaca.analyze import FingerAssessment, analyze_photo
from gattaca.evidence import render_evidence
from gattaca.pipeline.enhance import EnhanceParams
from gattaca.scoring import Recoverability, score_finger


@dataclass
class ImageAssessment:
    fingers: list[FingerAssessment]
    scores: list[Recoverability]
    evidence: np.ndarray | None

    @property
    def overall(self) -> Recoverability | None:
        return max(self.scores, key=lambda s: s.probability) if self.scores else None

    def summary(self) -> dict:
        o = self.overall
        return {
            "disclaimer": DISCLAIMER,
            "fingers_found": len(self.fingers),
            "band": o.band if o else "LOW",
            "probability": round(o.probability, 3) if o else 0.0,
            "confidence": round(o.confidence, 3) if o else 0.5,
            "explanation": o.explanation if o else (
                "No fingertip was found in this image, so no ridge detail can be assessed. "
                "If your fingertips are visible, try a brighter, plainer background."),
            "limiting_factor": o.limiting_factor if o else "no-finger",
            "calibration_source": o.calibration_source if o else None,
            "distances": o.distances if o else [],
            "per_finger": [{"label": f.label, "band": s.band, "probability": round(s.probability, 3),
                            "confidence": round(s.confidence, 3), "metrics": s.metrics}
                           for f, s in zip(self.fingers, self.scores)],
        }


def assess_image(img: np.ndarray, *, backend: str = "auto", params: EnhanceParams = EnhanceParams(),
                 render: bool = True, max_fingers: int = 5) -> ImageAssessment:
    fingers = analyze_photo(img, backend=backend, params=params, max_fingers=max_fingers)
    scores = [score_finger(f) for f in fingers]
    order = np.argsort([-s.probability for s in scores])
    fingers = [fingers[i] for i in order]
    scores = [scores[i] for i in order]
    ev = render_evidence(img, fingers, scores) if render else None
    return ImageAssessment(fingers, scores, ev)
