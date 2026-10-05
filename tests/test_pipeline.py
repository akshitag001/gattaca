"""Stage-level tests on synthetic prints with known ground truth."""

import cv2
import numpy as np
import pytest

from gattaca import synth
from gattaca.analyze import analyze_contact, analyze_photo
from gattaca.pipeline.detect import detect_fingertips
from gattaca.pipeline.enhance import enhance, orientation_field
from gattaca.pipeline.extract import crossing_number, extract_minutiae
from gattaca.pipeline.quality import clarity_map, measure_pprp
from gattaca.pipeline.rectify import cylinder_unwrap
from gattaca.types import BIFURCATION, ENDING


@pytest.mark.parametrize("pprp", [10, 6, 4, 3])
def test_pprp_estimate_tracks_resolution(photo0, pprp):
    img, truth = photo0
    s = pprp / truth.pprp
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(truth.mask, g.shape[::-1], interpolation=cv2.INTER_NEAREST)
    est, snr = measure_pprp(g, m)
    # cylindrical foreshortening makes the photo-wide period ~10% shorter than at the centre
    assert 0.75 * pprp < est < 1.05 * pprp
    assert snr > 4


def test_orientation_matches_synthetic_field():
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    normal_true = np.deg2rad(30)
    img = (128 + 80 * np.cos(2 * np.pi / 9 * (xx * np.cos(normal_true) + yy * np.sin(normal_true)))).astype(np.uint8)
    norm = img.astype(np.float32) - img.mean()
    normal, coh = orientation_field(norm, np.full((n, n), 255, np.uint8), 7)
    c = slice(64, 192)
    err = np.angle(np.exp(2j * (normal[c, c] - normal_true))) / 2
    assert np.abs(err).mean() < np.deg2rad(3)
    assert coh[c, c].mean() > 0.9


def test_crossing_number_endings_and_bifurcations():
    sk = np.zeros((20, 20), np.uint8)
    sk[10, 2:15] = 1           # a line: two endings
    sk[3:10, 8] = 1            # a branch joining it: one bifurcation + one more ending
    cn = crossing_number(sk)
    assert (cn == 1).sum() == 3
    assert (cn == 3).sum() >= 1


def test_minutiae_extracted_from_contact_print(contact0):
    an = analyze_contact(contact0)
    m = an.minutiae
    assert 15 <= len(m) <= 120
    assert set(np.unique(m["kind"])) <= {ENDING, BIFURCATION}
    assert np.all((m["angle"] >= 0) & (m["angle"] < 2 * np.pi))
    assert an.quality.clarity > 0.6


def test_clarity_rejects_noise_edges_and_highlights(contact0):
    rng = np.random.default_rng(0)
    full = np.full((300, 300), 255, np.uint8)
    noise = np.clip(128 + 160 * cv2.GaussianBlur(rng.normal(0, 1, (300, 300)), (0, 0), 3), 0, 255).astype(np.uint8)
    bar = np.full((300, 300), 120, np.uint8)
    bar[:, 140:162] = 200
    ridges = contact0[50:350, 50:350]
    vals = {}
    for name, img in {"noise": noise, "bar": cv2.GaussianBlur(bar, (0, 0), 3), "ridges": ridges}.items():
        e = enhance(img, full)
        vals[name] = float(clarity_map(e, 9.0).mean())
    assert vals["ridges"] > 0.6
    assert vals["noise"] < 0.1 and vals["bar"] < 0.1


def test_detect_single_finger(photo0):
    img, truth = photo0
    dets = detect_fingertips(img, backend="classical")
    assert dets
    d = dets[0]
    assert np.hypot(d.tip[0] - truth.tip[0], d.tip[1] - truth.tip[1]) < 0.25 * truth.finger_width
    assert np.dot(d.axis, truth.axis) > 0.9
    assert 0.7 * truth.finger_width < d.width < 1.3 * truth.finger_width


def test_cylinder_unwrap_stretches_edges():
    h, w = 50, 100
    stripes = (np.arange(w) % 6 < 3).astype(np.uint8)[None].repeat(h, 0) * 255
    out, m, _, mx = cylinder_unwrap(stripes, np.full((h, w), 255, np.uint8))
    step = np.diff(mx[0])
    assert step[len(step) // 2] > step[2]  # centre samples are spaced wider than edge samples
    assert out.shape[1] > w


def test_full_photo_pipeline_extracts_ridges(photo0):
    img, _ = photo0
    fingers = analyze_photo(img, backend="classical")
    assert fingers
    an = fingers[0].analysis
    assert 8 < an.pprp < 12
    assert an.quality.minutiae_count > 25
    assert an.quality.clarity > 0.3


def test_debug_dumps_every_stage(tmp_path, photo0):
    from gattaca.debug import DebugSink
    analyze_photo(photo0[0], backend="classical", debug=DebugSink(tmp_path))
    names = " ".join(p.name for p in tmp_path.iterdir())
    for stage in ("detect", "rectify", "enhance_gabor", "extract_skeleton", "quality_clarity"):
        assert stage in names
