<div align="center">
  <h1>🧬 GATTACA</h1>
  <p><strong>A biometric-privacy research & awareness tool for the OpenCV AI Competition 2026.</strong></p>
  
  <p>
    <a href="https://github.com/akshitag001/gattaca/actions"><img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=github" alt="Build Status"></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-5.0-red?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
    <a href="https://aws.amazon.com/graviton/"><img src="https://img.shields.io/badge/AWS-Graviton4-orange?style=for-the-badge&logo=amazonaws" alt="AWS"></a>
    <a href="https://github.com/akshitag001/gattaca/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge" alt="License"></a>
  </p>

  <p>
    <em>How much of your fingerprint does that photo give away?</em>
  </p>
</div>

---

> ⚠️ **IMPORTANT**: GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems.

## 🌟 What is GATTACA?

**GATTACA** measures how much fingerprint ridge detail is recoverable from photos or short clips of your hand under real capture conditions. Upload your media, and GATTACA will return:

1. 📊 **Recoverability Score**: `HIGH` / `MEDIUM` / `LOW`, including a calibrated probability and bootstrap confidence.
2. 🗺️ **Ridge-Evidence Heat Map**: A visual overlay on your photo exposing where ridge detail is accessible.
3. 💬 **Plain-Language Explanation**: Explains limiting factors (resolution, blur, lighting, coverage) and computes an equivalent camera distance.
4. 🛡️ **Protected Copy (Optional)**: A variant of your image with ridge detail suppressed, re-scored to demonstrate the privacy gain.

### 🔬 Research Contribution
GATTACA delivers a measured, reproducible **recoverability curve**—providing the identification rate against paired ground truth as a function of *pixels per ridge period* (PPRP), blur, motion, compression, exposure, and noise. It replaces unsourced internet claims ("100% at 1.5 m") with rigorous methodology.

![Recoverability vs PPRP](study/plots/synthetic/recoverability_vs_pprp.png)
*Current figures validate the instrument using synthetic paired prints. Final curve to be re-run on HKPolyU/RidgeBase.*

---

## 🚀 Quickstart

### 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/akshitag001/gattaca.git
cd gattaca

# Set up virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Fetch models for cv2.dnn
python scripts/fetch_models.py
```

### 🎮 Running the Tools

```bash
# 1. Run the test suite (~20s for 36 tests)
python -m pytest -q

# 2. Synthetic demo (no local data required)
python -m gattaca demo

# 3. Assess a local image
python -m gattaca assess my_hand.jpg --evidence ev.jpg --debug debug/

# 4. Agentic video processing
python -m gattaca agent my_hand_clip.mp4 --trace trace.json

# 5. Protect a photo (suppress ridges)
python -m gattaca protect my_hand.jpg protected.jpg

# 6. Forensic minutiae report (generates charts + CSV + ISO notes)
python -m gattaca minutiae print.png --out report/ --notes

# 7. Multi-frame consolidation
python -m gattaca consolidate clip.mp4 --out report/
```

### 🌐 Web API & UI

```bash
# Standard Web UI (Upload, crop, live camera)
uvicorn service.api:app --port 8000

# Research Mode (Includes live minutiae chart)
GATTACA_RESEARCH=1 uvicorn service.api:app --port 8000
```
Navigate to `http://localhost:8000` to interact with the service.

---

## 📈 Key Results (Synthetic Validation)

| Finding | Value |
|---------|-------|
| 🎯 **Rank-1 ID (≥ 95%)** down to | **3.9 px/ridge** (50% at 2.9) |
| 🗜️ **JPEG q75 Re-encode Cost** | **~1 px/ridge** (95% needs 5.5) |
| 🌫️ **Defocus / Motion limit** | σ ≥ 0.3 periods / ≥ 1 period |
| 📊 **Band Reliability (5-fold)** | HIGH: 96% · MEDIUM: 46% · LOW: 8% |
| 🔄 **Multi-frame Fusion vs Single** | Matched minutiae ~2×; Can move LOW→MEDIUM |
| 📷 **12 MP Equivalent Distance** | ≤ ~39 cm (5× tele: ~180 cm) for 80% ID |

---

## ⚙️ How It Works (Under the Hood)

GATTACA relies heavily on **OpenCV 5.0** to perform the computational heavy lifting.

| Stage | Module | OpenCV 5 Capabilities Utilized |
|---|---|---|
| **Hand/Fingertip Detection** | `detect.py` | `cv2.dnn` (MediaPipe ONNX), `convexHull`, `distanceTransform` |
| **Finger ROI** | `roi.py` | YCrCb + Otsu, erode/dilate, largest contour, `warpAffine` |
| **Perspective Rectify** | `rectify.py` | `fitLine`, `getPerspectiveTransform`, `warpPerspective`, `remap` |
| **PPRP Measurement** | `ridgefreq.py` | block FFT (whitened polar spectrum) |
| **Ridge Enhancement** | `enhance.py` | `createCLAHE`, `bilateralFilter`, `Sobel` tensor, `getGaborKernel` + `filter2D` bank |
| **Minutiae Extraction** | `extract.py` | `ximgproc.thinning`, `connectedComponentsWithStats`, crossing number |
| **Quality Assessment** | `quality.py` | local correlation, `remap` autocorrelation, coherence |
| **Deformation & Matching** | `deform.py`, `match.py` | `estimateAffinePartial2D`, `createThinPlateSplineShapeTransformer` |
| **Video Fusion** | `video/` | `VideoCapture`, `calcOpticalFlowPyrLK`, `findTransformECC`, `ORB`, kernel-regression |
| **Agentic Loop** | `agent/` | perceive → decide (R1–R8) → act, full trace generation |

---

## ☁️ AWS Architecture

The project's AWS infrastructure (`aws/infra`) synthesizes to 50 CloudFormation resources via AWS CDK. The stack includes:
- **API Gateway → ALB → Auto Scaling Group**
- Instances: **Graviton4 c8g** running the **COOL AMI** (Cloud Optimized OpenCV v3.1).
- Storage: **S3** for uploads (1-day expiry backstop) and datasets.
- Benchmarking: The `aws/cool_benchmark` compares COOL/Graviton vs stock OpenCV/x86 reporting speed-ups, latency, and throughput per 1,000 photos.

*For more details, see [docs/architecture.md](docs/architecture.md).*

---

## 📚 Documentation & Research

- **[Research Notes](docs/research_notes.md)**: Literature review and study background.
- **[Recoverability Metric](docs/recoverability_metric.md)**: How we measure exposure risk.
- **[Architecture](docs/architecture.md)**: Diagrams for AWS, COOL, and Agent workflow.
- **[Responsible Use](docs/responsible_use.md)**: Our commitment to ethics and safe data handling.
- **[Datasets](datasets/README.md)**: Instructions for requesting access to evaluation data.

---

<div align="center">
  <p>Built for the <strong>OpenCV AI Competition 2026</strong></p>
  <p>Made with ❤️ by high-paid engineers (and AI assistants).</p>
</div>
