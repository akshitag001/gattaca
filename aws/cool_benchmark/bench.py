"""COOL-on-Graviton vs stock-OpenCV-on-x86 benchmark for the GATTACA workload.

Run the SAME script on both machines:

    # Graviton4 (COOL AMI)
    source /opt/cool/venvs/python_3.12/bin/activate && python aws/cool_benchmark/bench.py --label cool-c8g
    # x86 baseline (stock wheel)
    /opt/stock/bin/python aws/cool_benchmark/bench.py --label stock-c7i
    # then
    python aws/cool_benchmark/compare.py results/cool-c8g.json results/stock-c7i.json

What it records:
* provenance: cv2 version and file path (proves which build ran), the KleidiCV/NEON
  flags from cv2.getBuildInformation(), the python prefix (COOL venv), CPU model,
  instance type (IMDSv2) and core count;
* micro: every OpenCV kernel the pipeline uses, at pipeline-representative sizes,
  single-threaded and with OpenCV's default threading;
* macro: end-to-end photo assessment latency (p50/p95), multi-process throughput
  (images/s, all cores), and the video agent on a synthetic clip.

Inputs are synthetic and seeded, so runs are reproducible and contain no biometric data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _imds(path: str) -> str | None:
    try:
        tok = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        t = urllib.request.urlopen(tok, timeout=0.5).read().decode()
        req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{path}",
                                     headers={"X-aws-ec2-metadata-token": t})
        return urllib.request.urlopen(req, timeout=0.5).read().decode()
    except Exception:
        return None


def provenance() -> dict:
    info = cv2.getBuildInformation()
    flags = [ln.strip() for ln in info.splitlines()
             if any(k in ln.lower() for k in ("kleidicv", "neon", "sve", "cpu/hw features", "baseline:",
                                              "dispatched", "parallel framework", "lapack"))]
    cpu = platform.processor()
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.lower().startswith(("model name", "cpu part")):
                cpu = ln.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"opencv": cv2.__version__, "cv2_path": cv2.__file__, "python_prefix": sys.prefix,
            "python": platform.python_version(), "machine": platform.machine(), "cpu": cpu,
            "cores": os.cpu_count(), "instance_type": _imds("instance-type"), "build_flags": flags,
            "kleidicv": "kleidicv" in info.lower(), "threads_default": cv2.getNumThreads()}


def timeit(fn, reps: int = 20, warm: int = 3) -> dict:
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(t), "p95_ms": float(np.percentile(t, 95)), "reps": reps}


def micro(threads: int) -> dict:
    from gattaca import synth
    cv2.setNumThreads(threads)
    rng = np.random.default_rng(0)
    photo, _ = synth.contactless_photo(0, sample=1, pprp=10)
    big = cv2.resize(photo, (4032, 3024))                        # 12 MP phone frame
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (360, 440))
    f32 = roi.astype(np.float32)
    gabor = cv2.getGaborKernel((25, 25), 4.5, 0.7, 9.0, 0.6, 0, ktype=cv2.CV_32F)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    binary = ((cv2.GaussianBlur(f32, (0, 0), 2) - f32) > 0).astype(np.uint8) * 255
    mx, my = np.meshgrid(np.linspace(0, 359, 400, dtype=np.float32), np.arange(440, dtype=np.float32))
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [359, 0], [359, 439], [0, 439]]),
                                    np.float32([[5, 3], [350, 0], [359, 430], [0, 439]]))
    shifted = cv2.warpAffine(gray, np.float32([[1, 0, 1.3], [0, 1, -0.7]]), gray.shape[::-1])
    pts = cv2.goodFeaturesToTrack(gray, 80, 0.01, 5)
    jpg = cv2.imencode(".jpg", big)[1]
    ops = {
        "imdecode_12MP_jpeg": lambda: cv2.imdecode(jpg, cv2.IMREAD_COLOR),
        "resize_area_12MP_to_2400": lambda: cv2.resize(big, (2400, 1800), interpolation=cv2.INTER_AREA),
        "cvtColor_YCrCb_12MP": lambda: cv2.cvtColor(big, cv2.COLOR_BGR2YCrCb),
        "gaussian_blur_12MP_s3": lambda: cv2.GaussianBlur(big, (0, 0), 3),
        "clahe_roi": lambda: clahe.apply(roi),
        "bilateral_roi": lambda: cv2.bilateralFilter(roi, 5, 20, 3),
        "nlmeans_roi": lambda: cv2.fastNlMeansDenoising(roi, None, 7, 5, 15),
        "sobel_x_y_roi": lambda: (cv2.Sobel(f32, cv2.CV_32F, 1, 0), cv2.Sobel(f32, cv2.CV_32F, 0, 1)),
        "gabor_filter2D_x48_roi": lambda: [cv2.filter2D(f32, cv2.CV_32F, gabor) for _ in range(48)],
        "thinning_roi": lambda: cv2.ximgproc.thinning(binary),
        "remap_unwrap_roi": lambda: cv2.remap(roi, mx, my, cv2.INTER_CUBIC),
        "warpPerspective_roi": lambda: cv2.warpPerspective(roi, H, (360, 440), flags=cv2.INTER_CUBIC),
        "findTransformECC_affine": lambda: cv2.findTransformECC(gray.astype(np.float32), shifted.astype(np.float32),
                                                                np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                                                                (3, 100, 1e-6), None, 5),
        "calcOpticalFlowPyrLK_80pts": lambda: cv2.calcOpticalFlowPyrLK(gray, shifted, pts, None),
        "imencode_jpeg_640": lambda: cv2.imencode(".jpg", photo),
    }
    from gattaca.pipeline.detect import DnnHandLandmarker
    if DnnHandLandmarker.available():
        net = cv2.dnn.readNetFromONNX(str(ROOT / "models" / "palm_detection_mediapipe_2023feb.onnx"))
        blob = rng.random((1, 192, 192, 3), dtype=np.float32)

        def palm():
            net.setInput(blob)
            net.forward(net.getUnconnectedOutLayersNames())
        ops["dnn_palm_detector_192"] = palm
    return {k: timeit(v, reps=10 if "x48" in k or "nlmeans" in k else 20) for k, v in ops.items()}


def _one(i: int) -> float:
    from gattaca import synth
    from gattaca.assess import assess_image
    cv2.setNumThreads(1)
    img, _ = synth.contactless_photo(i % 8, sample=1, pprp=10)
    t0 = time.perf_counter()
    assess_image(img, backend="classical")
    return time.perf_counter() - t0


def macro(n_latency: int, n_throughput: int, workers: int) -> dict:
    from gattaca import synth
    from gattaca.agent import AgentConfig, run_agent
    for i in range(8):           # warm the synthetic cache so generation is not timed
        synth.master_print(i)
    lat = [_one(i) * 1000 for i in range(n_latency)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_one, range(n_throughput)))
    wall = time.perf_counter() - t0
    clip = synth.synthetic_clip(2, n_frames=24, downscale=2.0, seed=1)
    t1 = time.perf_counter()
    res = run_agent(clip, AgentConfig(batch=12))
    agent_s = time.perf_counter() - t1
    return {"photo_latency_ms": {"p50": statistics.median(lat), "p95": float(np.percentile(lat, 95)), "n": n_latency},
            "photo_throughput": {"images_per_s": n_throughput / wall, "workers": workers, "n": n_throughput},
            "agent_clip": {"seconds": agent_s, "steps": len(res.trace), "status": res.status}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="e.g. cool-c8g.xlarge or stock-c7i.xlarge")
    ap.add_argument("--latency-n", type=int, default=20)
    ap.add_argument("--throughput-n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=ROOT / "aws" / "cool_benchmark" / "results")
    a = ap.parse_args(argv)
    out = {"label": a.label, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "provenance": provenance()}
    print(json.dumps(out["provenance"], indent=1))
    out["micro_1thread"] = micro(1)
    out["micro_default_threads"] = micro(out["provenance"]["threads_default"])
    out["macro"] = macro(a.latency_n, a.throughput_n, a.workers)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.label}.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["macro"], indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
