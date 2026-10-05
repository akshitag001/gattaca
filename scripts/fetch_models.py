"""Download the OpenCV-zoo MediaPipe hand models (Apache-2.0) used by the DNN detector.

    python scripts/fetch_models.py

These are public model weights, not data about people. Without them, GATTACA falls
back to the classical skin-mask fingertip detector.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "models"
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "palm_detection_mediapipe_2023feb.onnx": f"{BASE}/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx",
    "handpose_estimation_mediapipe_2023feb.onnx":
        f"{BASE}/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx",
}


def main() -> int:
    DEST.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        path = DEST / name
        if path.exists() and path.stat().st_size > 1_000_000:
            print(f"ok   {name}")
            continue
        print(f"get  {name}")
        urllib.request.urlretrieve(url, path)
        if path.stat().st_size < 1_000_000 or path.read_bytes()[:4] == b"vers":  # git-lfs pointer, not weights
            path.unlink()
            print(f"failed: {name} (got an LFS pointer?)", file=sys.stderr)
            return 1
        print(f"     sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
