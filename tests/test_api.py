import glob
import os
import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gattaca import DISCLAIMER, synth
from gattaca.video.decode import write_video
from service.api import app

client = TestClient(app)


def _jpeg(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["disclaimer"] == DISCLAIMER


def test_index_has_disclaimer_and_consent():
    html = client.get("/").text
    assert DISCLAIMER in html and "my own hand" in html


def test_consent_required(photo0):
    r = client.post("/api/score", files={"file": ("h.jpg", _jpeg(photo0[0]), "image/jpeg")})
    assert r.status_code == 400


def test_score_image_returns_assessment_without_templates(photo0):
    r = client.post("/api/score", files={"file": ("h.jpg", _jpeg(photo0[0]), "image/jpeg")}, data={"consent": "true"})
    assert r.status_code == 200
    j = r.json()
    assert j["band"] in ("HIGH", "MEDIUM", "LOW") and j["disclaimer"] == DISCLAIMER
    assert j["evidence"].startswith("data:image/jpeg;base64,")
    text = r.text
    for forbidden in ('"minutiae":', '"x":', '"y":', "template", "skeleton"):
        assert forbidden not in text


def test_video_upload_leaves_no_temp_files(tmp_path):
    frames = synth.synthetic_clip(2, n_frames=10, downscale=2.0, seed=2)
    path = tmp_path / "clip.mp4"
    write_video(str(path), frames, fps=10)
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "gattaca_*")))
    r = client.post("/api/score", files={"file": ("clip.mp4", path.read_bytes(), "video/mp4")},
                    data={"consent": "true"})
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "gattaca_*")))
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "video" and r.json()["agent"]["trace"]
    assert after == before


def test_protect_lowers_risk():
    img, _ = synth.contactless_photo(1, sample=2, pprp=10)
    r = client.post("/api/protect", files={"file": ("h.jpg", _jpeg(img), "image/jpeg")}, data={"consent": "true"})
    assert r.status_code == 200
    j = r.json()
    assert j["after"]["probability"] < j["before"]["probability"]


def test_minutiae_endpoint_is_off_by_default(photo0, monkeypatch):
    monkeypatch.delenv("GATTACA_RESEARCH", raising=False)
    r = client.post("/api/minutiae", files={"file": ("h.jpg", _jpeg(photo0[0]), "image/jpeg")},
                    data={"consent": "true"})
    assert r.status_code == 404 and "research" in r.text.lower()
    assert client.get("/health").json()["research_mode"] is False


def test_minutiae_endpoint_in_research_mode(photo0, monkeypatch):
    monkeypatch.setenv("GATTACA_RESEARCH", "1")
    assert client.get("/health").json()["research_mode"] is True
    r = client.post("/api/minutiae", files={"file": ("h.jpg", _jpeg(photo0[0]), "image/jpeg")},
                    data={"consent": "true"})
    assert r.status_code == 200
    j = r.json()
    assert j["minutiae_count"] > 10
    assert j["endings"] + j["bifurcations"] == j["minutiae_count"]
    assert j["chart"].startswith("data:image/jpeg;base64,")
    # the chart is returned, but never the coordinates
    for forbidden in ('"x_px"', '"y_px"', '"angle_deg"', '"table"'):
        assert forbidden not in r.text


def test_minutiae_requires_consent(photo0, monkeypatch):
    monkeypatch.setenv("GATTACA_RESEARCH", "1")
    r = client.post("/api/minutiae", files={"file": ("h.jpg", _jpeg(photo0[0]), "image/jpeg")})
    assert r.status_code == 400


def test_rejects_garbage():
    r = client.post("/api/score", files={"file": ("x.txt", b"hello", "text/plain")}, data={"consent": "true"})
    assert r.status_code == 415
