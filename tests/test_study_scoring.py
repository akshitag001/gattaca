import json

import numpy as np
import pytest

from datasets.loader import DatasetNotAvailable, load_dataset
from gattaca.scoring import Calibration, distance_for_pprp, pprp_at_distance, score
from gattaca.types import QualityReport
from study import degrade


def test_ladders_are_physical(photo0):
    img, truth = photo0
    native = 9.5
    assert degrade.apply(img, degrade.Condition("pprp", 12), native) is None   # never upsample
    small = degrade.apply(img, degrade.Condition("pprp", 4.75), native, seed=1)
    assert abs(small.shape[1] / img.shape[1] - 0.5) < 0.01
    j = degrade.apply(img, degrade.Condition("jpeg", 30), native)
    assert j.shape == img.shape
    dark = degrade.apply(img, degrade.Condition("exposure", -3), native)
    assert dark.mean() < img.mean() * 0.6
    a = degrade.apply(img, degrade.Condition("gauss", 0.3), native, seed=5)
    b = degrade.apply(img, degrade.Condition("gauss", 0.3), native, seed=5)
    assert np.array_equal(a, b)                                                   # deterministic


def test_camera_model_round_trip():
    p = pprp_at_distance(500, 4032, 26)
    assert abs(distance_for_pprp(p, 4032, 26) - 500) < 1e-6
    assert pprp_at_distance(500, 4032, 77) > 2.5 * p      # telephoto resolves more


def _q(pprp, clarity, n):
    return QualityReport(pprp=pprp, spectral_snr=50, coherence=0.8, clarity=clarity, area_px=90000,
                         minutiae_count=n, minutiae_quality=0.6, nfiq_like=70)


def test_score_monotone_and_banded():
    cal = Calibration.prior()
    lo = score(_q(2.0, 0.02, 4), calib=cal)
    hi = score(_q(9.0, 0.8, 60), calib=cal)
    assert lo.probability < hi.probability
    assert hi.band == "HIGH" and lo.band == "LOW"
    assert "cm" in hi.explanation
    assert 0.0 <= lo.confidence <= 1.0


def test_shipped_calibration_loads():
    from gattaca.scoring import DEFAULT_CALIB
    if not DEFAULT_CALIB.exists():
        pytest.skip("no calibration produced yet")
    cal = Calibration.load(DEFAULT_CALIB)
    d = json.loads(DEFAULT_CALIB.read_text())
    assert len(cal.coef) == len(cal.features) and "curve" in d


def test_loader_synthetic_and_missing_licensed():
    pairs = list(load_dataset("synthetic", subjects=2))
    assert len(pairs) == 2 and pairs[0].contact_gt.ndim == 2 and pairs[0].contactless_img.ndim == 3
    assert pairs[0].key != pairs[1].key
    with pytest.raises(DatasetNotAvailable):
        load_dataset("hkpolyu")
