import numpy as np

from gattaca import synth
from gattaca.analyze import analyze_contact, analyze_photo
from gattaca.pipeline.deform import TPS, DeformationCorrectionModel, rtps_fit
from gattaca.pipeline.match import match
from gattaca.types import MINUTIA_DTYPE


def _transform(m, ang, t):
    out = m.copy()
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    xy = np.c_[m["x"], m["y"]] @ R.T + t
    out["x"], out["y"] = xy[:, 0], xy[:, 1]
    out["angle"] = np.mod(m["angle"] + ang, 2 * np.pi)
    return out


def test_tps_interpolates_and_recovers_warp():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 300, (30, 2))
    warp = lambda p: p + np.c_[5 * np.sin(p[:, 1] / 60), 4 * np.cos(p[:, 0] / 50)]  # noqa: E731
    tps = TPS.fit(src, warp(src), lam=0.0)
    assert np.abs(tps(src) - warp(src)).max() < 1e-6
    test = rng.uniform(40, 260, (50, 2))
    assert np.abs(tps(test) - warp(test)).mean() < 1.0


def test_rtps_rejects_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 300, (40, 2))
    dst = src + np.array([3.0, -2.0])
    dst[:5] += rng.uniform(30, 60, (5, 2))          # 5 wrong correspondences
    _, inl = rtps_fit(src, dst)
    assert not inl[:5].any() and inl[5:].mean() > 0.9


def test_match_is_invariant_to_rigid_motion(contact0):
    m = analyze_contact(contact0).minutiae
    moved = _transform(m, np.deg2rad(20), np.array([30.0, -15.0]))
    res = match(moved, m)
    assert res.n_matched >= 0.9 * len(m)


def test_genuine_beats_impostors():
    gallery = [analyze_contact(synth.contact_impression(i)) for i in range(5)]
    for i in range(3):
        img, _ = synth.contactless_photo(i, sample=1, pprp=10)
        probe = analyze_photo(img, backend="classical")[0].analysis.minutiae
        scores = [match(probe, g.minutiae).score for g in gallery]
        assert int(np.argmax(scores)) == i
        assert scores[i] > 2 * sorted(scores)[-2]


def test_dcm_learns_and_applies_systematic_displacement():
    mask = np.zeros((400, 400), np.uint8)
    mask[50:350, 80:320] = 255
    dcm = DeformationCorrectionModel()
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform([90, 60], [310, 340], (40, 2))
        dcm.add_pair(p, p + np.array([4.0, -3.0]), mask)
    dcm = dcm.finalize()
    q = rng.uniform([120, 100], [280, 300], (10, 2))
    corrected = dcm.apply(q, mask)
    assert np.abs((corrected - q).mean(axis=0) - [4.0, -3.0]).max() < 0.8


def test_degenerate_inputs():
    empty = np.zeros(0, MINUTIA_DTYPE)
    assert match(empty, empty).score == 0.0
