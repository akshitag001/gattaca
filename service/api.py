"""GATTACA web service: POST a photo or short clip of YOUR OWN hand -> exposure score.

    uvicorn service.api:app --host 0.0.0.0 --port 8000

Data handling (docs/responsible_use.md):
* explicit consent flag required, otherwise HTTP 400;
* images decoded in memory; videos go through a temp file deleted in `finally:`;
* responses contain scores, counts and an overlay image only, never coordinates or templates;
* logs contain request metadata only (kind, size, band, latency).
"""

from __future__ import annotations

import base64
import logging
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gattaca import DISCLAIMER, __version__
from gattaca.agent import AgentConfig, run_agent
from gattaca.assess import assess_image
from gattaca.evidence import encode_jpeg, render_evidence, render_zoom_callout
from gattaca.protect import protect_photo
from gattaca.scoring import load_calibration
from gattaca.video.decode import read_video_bytes

log = logging.getLogger("gattaca.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

WEB = Path(__file__).resolve().parent / "web"
MAX_BYTES = 25 * 1024 * 1024
MAX_SIDE = 2400           # larger photos are downscaled for latency (PPRP is re-measured)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


def research_mode() -> bool:
    """Local forensic-research mode (GATTACA_RESEARCH=1), OFF by default.

    Only when it is on does /api/minutiae exist. A deployed public endpoint therefore
    never emits the minutiae chart, while a researcher running the service locally on
    their own prints can. Even then the response carries the rendered educational chart
    and a count, never minutiae coordinates: those stay in the local CLI
    (`python -m gattaca minutiae`). See docs/responsible_use.md.
    """
    return os.getenv("GATTACA_RESEARCH", "").strip() == "1"

app = FastAPI(title="GATTACA", version=__version__,
              description="Fingerprint-exposure risk for photos of your own hand. " + DISCLAIMER)


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _b64jpeg(img: np.ndarray | None) -> str | None:
    return None if img is None else "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(img)).decode()


def _limit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


async def _read(file: UploadFile, consent: bool) -> bytes:
    if not consent:
        raise HTTPException(400, "Consent is required: confirm this is a photo or video of your own hand.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (25 MB max).")
    if not data:
        raise HTTPException(400, "Empty upload.")
    return data


def _decode_image(data: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else _limit(img)


def _summary_from_agent(res, kind: str) -> dict:
    trace = [{k: v for k, v in e.items() if k != "t"} for e in res.trace]
    if res.best is None:
        return {"kind": kind, "band": "LOW", "probability": 0.0, "confidence": 0.5, "fingers_found": 0,
                "explanation": "No fingertip was found, so no ridge detail can be assessed. If your fingertips "
                               "are visible, try better light and a plainer background.",
                "limiting_factor": "no-finger", "agent": {"status": res.status, "trace": trace},
                "evidence": None}
    b = res.best
    ev = render_evidence(res.evidence_input, [res.best_finger], [b])
    zoom = render_zoom_callout(res.evidence_input, [res.best_finger], [b])
    return {"kind": kind, "band": b.band, "probability": round(b.probability, 3),
            "confidence": round(b.confidence, 3), "explanation": b.explanation,
            "limiting_factor": b.limiting_factor, "calibration_source": b.calibration_source,
            "distances": b.distances, "metrics": b.metrics, "best_source": res.best_source,
            "agent": {"status": res.status, "trace": trace}, "evidence": _b64jpeg(ev),
            "zoom": _b64jpeg(zoom)}


@app.get("/health")
def health():
    cal = load_calibration()
    return {"status": "ok", "version": __version__, "opencv": cv2.__version__,
            "calibration": cal.source, "research_mode": research_mode(), "disclaimer": DISCLAIMER}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


# landing-page imagery: generated from the pipeline on a SYNTHETIC hand
# (scripts/make_landing_assets.py), so the public page shows no real person's data
if (WEB / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")


@app.post("/api/score")
async def score(file: UploadFile = File(...), consent: bool = Form(False)):
    t0 = time.perf_counter()
    data = await _read(file, consent)
    kind = "image"
    try:
        img = _decode_image(data)
        if img is not None:
            res = run_agent(img, AgentConfig(max_steps=6))
        else:
            if Path(file.filename or "").suffix.lower() not in VIDEO_EXT and not (file.content_type or "").startswith("video"):
                raise HTTPException(415, "Unsupported file: upload a JPEG/PNG photo or a short MP4/MOV clip.")
            kind = "video"
            video = read_video_bytes(data, suffix=Path(file.filename or "x.mp4").suffix or ".mp4",
                                     max_frames=90, max_side=1920)
            if not video.frames:
                raise HTTPException(422, "Could not decode any frames from the video.")
            frames = video.frames

            def source(start: int, n: int):
                return frames[start:start + n]

            from gattaca.agent.loop import RecoverabilityAgent
            res = RecoverabilityAgent(source, AgentConfig(frame_budget=len(frames), batch=30)).run()
        out = _summary_from_agent(res, kind)
    finally:
        del data  # nothing about the upload outlives the request
    out["disclaimer"] = DISCLAIMER
    out["retention"] = "Your upload was processed in memory and has been deleted."
    log.info("score kind=%s band=%s ms=%d", kind, out["band"], (time.perf_counter() - t0) * 1000)
    return JSONResponse(_clean(out))


@app.post("/api/minutiae")
async def minutiae(file: UploadFile = File(...), consent: bool = Form(False)):
    """Forensic minutiae chart for a frame. Local research mode only (see research_mode)."""
    if not research_mode():
        raise HTTPException(404, "Minutiae extraction is disabled. It is a local research feature: "
                                 "run the service with GATTACA_RESEARCH=1, or use "
                                 "`python -m gattaca minutiae <image>`.")
    t0 = time.perf_counter()
    data = await _read(file, consent)
    try:
        img = _decode_image(data)
        if img is None:
            raise HTTPException(415, "Minutiae extraction works on photos (JPEG/PNG).")
        from gattaca.forensic import ISO_19794_2_NOTE, minutiae_report
        try:
            rep = minutiae_report(img, photo=True, title="live frame")
        except ValueError as e:
            raise HTTPException(422, str(e))
    finally:
        del data
    endings = sum(1 for r in rep.table if r["type"] == "ridge-ending")
    out = {"source": rep.source, "minutiae_count": len(rep.table), "endings": endings,
           "bifurcations": len(rep.table) - endings,
           "pprp": rep.analysis.quality.pprp, "clarity": rep.analysis.quality.clarity,
           "mean_quality": rep.analysis.quality.minutiae_quality,
           "chart": _b64jpeg(rep.chart), "iso_notes": ISO_19794_2_NOTE,
           "disclaimer": DISCLAIMER,
           "note": "Educational chart only. Minutiae coordinates are not returned over the API; "
                   "use `python -m gattaca minutiae` locally for the table.",
           "retention": "Your frame was processed in memory and has been deleted."}
    log.info("minutiae n=%d ms=%d", len(rep.table), (time.perf_counter() - t0) * 1000)
    return JSONResponse(_clean(out))


@app.post("/api/protect")
async def protect(file: UploadFile = File(...), consent: bool = Form(False)):
    t0 = time.perf_counter()
    data = await _read(file, consent)
    try:
        img = _decode_image(data)
        if img is None:
            raise HTTPException(415, "Protection works on photos (JPEG/PNG).")
        before = assess_image(img, render=False)
        prot, _ = protect_photo(img, [f.detection for f in before.fingers] or None)
        after = assess_image(prot, render=False)
    finally:
        del data
    b, a = before.summary(), after.summary()
    out = {"before": {k: b[k] for k in ("band", "probability", "fingers_found")},
           "after": {k: a[k] for k in ("band", "probability", "fingers_found")},
           "protected_image": _b64jpeg(prot), "disclaimer": DISCLAIMER,
           "retention": "Your upload was processed in memory and has been deleted. Download the protected copy now."}
    log.info("protect before=%s after=%s ms=%d", b["band"], a["band"], (time.perf_counter() - t0) * 1000)
    return JSONResponse(_clean(out))
