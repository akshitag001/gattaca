# GATTACA — Technical Report

**OpenCV AI Competition 2026, powered by AWS**
Repository: https://github.com/akshitag001/gattaca

> GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems.

---

## 1. Problem

Viral posts and news articles claim that an ordinary photo — a peace sign, a selfie holding a phone — can be used to lift a usable fingerprint "from 1.5 metres away." None of the widely shared claims cite a methodology. The most-shared Korean example (SCMP-sourced, covered by multiple outlets) illustrates the claim with a zoomed fingerprint image that is itself captioned **"AI 생성 이미지" (AI-generated image)** — the "proof" is fabricated, not extracted.

The public conversation has no real number to replace the folklore with. GATTACA produces one: given a photo or short clip of someone's own hand, how much fingerprint ridge detail is actually recoverable, under what conditions, and why.

## 2. Users and intended use

- **Individuals** deciding whether to post a photo of their hand, who want a concrete answer instead of a rumor.
- **Researchers and students** (forensic science, biometrics) who want to see the classical minutiae-extraction chain run end to end, with an explanation of the ISO/IEC 19794-2 record format, without a tool that hands them an exploitable extraction pipeline.
- **Judges and the OpenCV/AWS community**, as a demonstration that a substantive OpenCV 5 pipeline plus a disciplined responsible-use boundary are not in tension — the measurement is real; the "attack tool" version of it was deliberately not built.

GATTACA only accepts: (a) the user's own hand, via direct upload with an explicit consent flag, or (b) licensed research datasets (HKPolyU, RidgeBase, IIITD ISPFD) under their own terms. There is no URL ingestion, scraping, or social-media integration anywhere in the codebase.

## 3. What it does

1. **Recoverability score** — `HIGH` / `MEDIUM` / `LOW`, a calibrated probability, and a bootstrap confidence.
2. **A heat map** over the user's own photo showing *where* ridge detail is exposed (never a clean, matcher-ready ridge image — see §8).
3. **A plain-language explanation** naming the limiting factor (resolution, blur, lighting, coverage) and an equivalent camera distance for a stated camera model.
4. **A one-click "protect this photo"** action that suppresses the ridge band and re-scores the result, demonstrating the risk drop.
5. **A live camera mode** and a **video mode**, the latter driven by an agentic perceive-decide-act loop (§6).

## 4. The research contribution: the recoverability curve

Public claims parameterize risk by distance in meters, which is meaningless without focal length, sensor width, and any later resizing. GATTACA parameterizes by **pixels per ridge period (PPRP)** — the number of pixels spanning one ridge period (≈0.45 mm) at the fingertip — which is what physically determines whether ridge information survives the Nyquist limit. PPRP is measured two independent ways (spectral, from a whitened block-FFT; geometric, from finger width), which lets the system distinguish "too few pixels" from "enough pixels but blurred." Distance only re-enters the output as a derived, camera-specific, upper-bound estimate for the user-facing message (`docs/recoverability_metric.md §5`).

### Study design

A controlled degradation study: paired contactless/contact fingerprint samples are degraded along six independent axes — resolution (the PPRP ladder), Gaussian defocus, motion blur, JPEG quality, under-exposure, and sensor noise — then run through the full pipeline and matched against ground truth (`n² / (N_P · N_Q)`, minutiae correspondence after rigid alignment + TPS deformation correction). The measured quantities per condition: minutiae recovered, mean minutiae quality, ridge clarity, rank-1 identification against a gallery, and verification EER. Full methodology: `docs/recoverability_metric.md`.

### Headline numbers (current data: synthetic paired prints — see §9)

60 synthetic fingers × 47 degradation conditions = 2,734 probe-conditions against a 60-finger gallery (chance rank-1 = 1.7%).

| Finding | Value |
|---|---|
| Rank-1 identification ≥ 95% down to | **3.9 px/ridge** (50% at 2.9) |
| Cost of one JPEG q75 re-encode | **~1 px/ridge** (95% then needs 5.5 px/ridge) |
| What a 12 MP phone resolves at 1.5 m | **0.87 px/ridge** — below Nyquist, physically unrecoverable |
| Defocus / motion that destroys recovery at full resolution | σ ≥ 0.3 ridge periods / ≥ 1 ridge period of motion |
| Held-out band reliability (5-fold CV, grouped by finger) | HIGH → 96% identified · MEDIUM → 46% · LOW → 8% |
| Multi-frame video fusion vs. best single frame | ~2× matched minutiae; a finger can move LOW → MEDIUM |

**Direct test of the "1.5 m" claim:** a 12 MP main camera at 1.5 m yields 0.87 px/ridge — below the physical resolution limit. Only a 5× optical telephoto, on the *original, uncompressed* file, reaches enough resolution at that distance (4.0 px/ridge); a single social-media JPEG re-encode erases that margin. The viral claim does not hold for an ordinary phone photo.

## 5. OpenCV 5 implementation

OpenCV 5.0.0 performs the substantive image/video analysis throughout — not a wrapper around a learned model. **110 distinct `cv2` functions across ~500 call sites**, verified by static analysis of the committed code (full breakdown: `docs/architecture.md`).

| Stage | Module | Key OpenCV 5 operations |
|---|---|---|
| Hand / fingertip detection | `gattaca/pipeline/detect.py` | `cv2.dnn.readNetFromONNX` + `NMSBoxes` (MediaPipe hand-landmark ONNX), with `convexHull` / `convexityDefects` / `distanceTransform` as a classical fallback |
| ROI extraction | `pipeline/roi.py` | YCrCb + Otsu `threshold`, morphological erode/dilate, `findContours`, `warpAffine` |
| Rectification | `pipeline/rectify.py` | `fitLine` edge fitting, `getPerspectiveTransform` / `warpPerspective`, cylindrical unwrap via `remap` |
| Resolution measurement | `pipeline/ridgefreq.py` | whitened polar-spectrum block FFT — the PPRP estimator |
| Ridge enhancement | `pipeline/enhance.py` | `createCLAHE`, `bilateralFilter` / `fastNlMeansDenoising`, Sobel structure-tensor orientation field, an oriented `getGaborKernel` bank applied with `filter2D` |
| Minutiae extraction | `pipeline/extract.py` | binarize → `ximgproc.thinning` (Zhang-Suen) → crossing-number minutiae → `connectedComponentsWithStats` pruning |
| Deformation correction + matching (study only) | `pipeline/deform.py`, `pipeline/match.py` | `createThinPlateSplineShapeTransformer` (TPS), `estimateAffinePartial2D` |
| Video: tracking | `video/track.py` | `calcOpticalFlowPyrLK` with forward-backward consistency check |
| Video: alignment | `video/align.py` | `findTransformECC` sub-pixel alignment, `ORB_create` + `estimateAffinePartial2D` fallback |
| Video: fusion | `video/fuse.py` | kernel-regression super-resolution splatting with robustness weighting (handheld multi-frame SR principle) |

Every stage is a pure, independently testable function; `--debug` dumps the intermediate image at every step for inspection.

## 6. The agentic loop (Agentic Vision)

`gattaca/agent/loop.py` implements a perceive → decide → act loop whose goal is the **worst-case** exposure estimate — a motivated adversary would not stop at the first blurry frame, so neither does the agent.

```
PERCEIVE  frame quality (sharpness, ridge-band energy, exposure, motion) +
          current assessment (band, P, PPRP, clarity, limiting factor)
DECIDE    an ordered rule policy (R1–R8); each decision logs the rule, the
          measurement that triggered it, and the resulting change in estimate
ACT       pull_frames | reselect_frame | retune_enhance | switch_finger |
          fuse_frames | stop   (one real OpenCV/video-pipeline call per action)
```

**OpenCV output demonstrably changes the next action.** Verbatim trace, synthetic handheld clip, shipped calibration:

```
[R1] pull_frames     because no frames decoded yet
    -> LOW    P=0.154 clarity=0.109 pprp=3.83
[R4] fuse_frames     because band LOW with 6 usable frames; fusion can add sub-pixel detail
    -> MEDIUM P=0.561 clarity=0.238          (delta +0.407)
[R5] retune_enhance  because clarity 0.238 at adequate PPRP 3.83
    -> LOW    P=0.031                         (kept best: 0.561)
[R8] stop            because no remaining action is expected to raise the estimate
```

The LOW measurement on the first frame caused the fusion call; the fused result's low clarity caused the enhancement retune. On an easy clip the same agent converges in one step (`[R1] → HIGH → [R2] stop`), so it never spends compute it does not need.

**Human control:** a frame budget, a step budget, an action allow-list, an `approve(action, context)` gate consulted before the expensive fusion step, and an `on_step(entry)` hook that can abort mid-run. `tests/test_video_agent.py` covers: early convergence on an easy clip, fusion triggered on a borderline clip with the triggering measurement asserted, a declined-approval run, and a restricted allow-list run.

**Task-level evaluation:** fusion improves matched minutiae over the best single frame on ≥2 of 3 synthetic test clips at ~3.1 px/ridge (roughly 2×); the agent reaches `stop` in every tested scenario (no infinite loops); human-control hooks are exercised, not just declared.

## 7. AWS deployment

**Architecture (designed, full scale-out path — `aws/infra`, CDK):** API Gateway → Application Load Balancer → an Auto Scaling group of Graviton4 (`c8g`) instances running the **Cloud-Optimized OpenCV Library (COOL) v3.1 AMI**, with S3 for datasets, study results, and a transient upload bucket (1-day lifecycle expiry as a backstop — the service itself never persists an upload; see §8). The CDK stack synthesizes cleanly to 50 CloudFormation resources. `aws/cool_benchmark` is a harness that runs an identical seeded workload on COOL/Graviton and on a stock-OpenCV x86 baseline, recording per-kernel speed-ups, latency, throughput, cost, and **provenance** (`cv2.__file__`, the KleidiCV build flag, `platform.machine()`, EC2 instance type via IMDS) as evidence of which build served the request.

**What is actually running at submission time:** given the build-phase timeline, the deployed instance is a single Graviton (`t4g.small`, `arm64`) EC2 instance running the same FastAPI service directly (no ALB/ASG — the cost- and time-minimal path), reachable at:

> **`http://<INSTANCE_PUBLIC_IP>:8000`** — filled in once the account's identity verification clears; see the submission form for the live value.

This still satisfies "a meaningful component on AWS" and "a working web endpoint": the full OpenCV 5 pipeline (detection, enhancement, Gabor filtering, extraction, scoring) executes on that Graviton instance for every request. The fuller ALB/ASG/COOL architecture remains in the repository (`aws/infra`, `aws/DEPLOY.md`) as the designed production path, and is the basis for a Best-Use-of-COOL submission if time allows a redeploy onto the COOL AMI before judging closes.

**x86 ↔ Arm coexistence:** every pinned dependency (`opencv-contrib-python-headless`, numpy, scipy) ships `manylinux` wheels for both `x86_64` and `aarch64`, so the identical codebase and `Dockerfile` run unmodified on either architecture. COOL adds Arm-specific kernels beneath the same `cv2` API — no GATTACA code is architecture-specific.

## 8. Responsible use (full policy: `docs/responsible_use.md`)

GATTACA's threat model is a motivated adversary with a single public photo or clip; the system reports the *upper bound* of what such an adversary could recover, so the advice errs toward caution. It deliberately does **not**:

| Not built | Enforced by |
|---|---|
| Ingest other people's photos (scraping, URL fetch, social-media integration) | Direct upload + required consent flag only; no HTTP client exists in `gattaca/` |
| Export a fingerprint template (ISO/IEC 19794-2, ANSI-378, WSQ, vendor formats) | API returns minutiae **counts** and quality only; coordinates never leave the process, including from the research-mode minutiae endpoint |
| Expose minutiae extraction on the public deployment | `/api/minutiae` returns HTTP 404 unless `GATTACA_RESEARCH=1` is set on the host |
| Produce a print-ready or binarized ridge image | The evidence map is a heat map blended over the original photo, capped and watermarked; binarized/skeleton images exist only in local `--debug` dumps |
| Tune anything toward real-matcher/sensor acceptance, or build presentation-attack artifacts | No such code path exists anywhere in the repository |

Uploads are decoded in memory; video is written to a temp file deleted in a `finally:` block (`tests/test_api.py` asserts no file survives a request). No raw biometric data is logged. Every surface — the API JSON, the web page, the evidence image watermark — carries the one-line disclaimer at the top of this report.

## 9. Evaluation evidence, including failure cases and limitations

**What has been measured:**
- The 2,734-condition degradation study above, with 95% Wilson confidence intervals per point and 5-fold cross-validation grouped by finger (not by image, to avoid leakage) for band reliability.
- The `cv2.dnn` hand-landmark detector was tested against three real (non-synthetic) photographs — a studio portrait V-sign, a casual phone-camera V-sign, and an open-palm shot — and found 5/5 fingers in each. Both V-sign photos scored LOW, correctly, because the raised fingertips carried only 1.1–2.3 px/ridge at arm's length: below the resolution floor, not a detector failure.
- 43 automated tests cover the pipeline, matching/deformation correction, the study/calibration code, the video/agent path, and the API (including the no-persisted-upload guarantee).

**Known failure modes and limitations, stated plainly:**
- **The headline curve above is calibrated on synthetic paired prints, not real skin.** This validates the measurement *instrument* (every stage, the matcher, the calibration pipeline) but is not yet the publishable result. Requests for the licensed real-world datasets (HKPolyU, RidgeBase, IIITD ISPFD — `datasets/README.md`) are outstanding; the identical code and commands re-run unchanged once access is granted (`python -m study.run_study --dataset hkpolyu`).
- **The classical matcher is a lower bound, not an upper bound, on adversary capability.** A learned cross-domain matcher (Ridgeformer) reports 2.83% EER on HKPolyU vs. 14.33% for the RTPS+DCM approach used here for measurement; a real adversary using a learned matcher would recover more at the same PPRP than this report's numbers suggest.
- **The agent's worst-case estimate is a maximum over several noisy strategies** (best single frame, fusion, re-tuned enhancement), which is optimistically biased by construction — intentional for conservative privacy advice, but the agent's reported probability is not the same calibrated quantity as a single-image score.
- **Deformation correction (TPS/DCM) showed no measurable benefit on synthetic data** — the systematic distortion in generated prints sits inside the matcher's tolerance window. Lin & Kumar's original result is on real contact-print pressure distortion, which the synthetic data does not model; this needs re-testing once real paired data is available. Reported rather than hidden.
- **Geometric PPRP assumes an adult-typical fingertip width**; children's or unusually narrow/wide fingers shift the estimate by roughly ±20%.
- **Skin-tone and camera-pipeline diversity** in the study is currently limited by the synthetic generator; a per-subgroup error breakdown is deferred to the real-dataset re-run.
- **Off-axis (rolled > ~45°) or partially occluded fingertips** are not modeled in the current degradation ladders.

## 10. Build, deploy, and test instructions

```bash
git clone https://github.com/akshitag001/gattaca.git && cd gattaca
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # all dependencies pinned
python scripts/fetch_models.py                           # optional: ONNX hand-landmark model

python -m pytest -q                                      # 43 tests, ~30 s
uvicorn service.api:app --port 8000                       # local web UI + API
```

AWS deployment: `aws/DEPLOY.md` (minimal single-instance path) and `aws/infra` + `aws/DEPLOY.md` §Phase 3 (full ALB/ASG/COOL CDK path).

## 11. References

Lin & Kumar, *Matching Contactless and Contact-based Fingerprint Images*, IEEE TIP 2018 (RTPS + DCM, reimplemented in `pipeline/deform.py`) · Pandey, Jawade & Setlur, *Ridgeformer*, ICIP 2025 (learned cross-domain matcher, cited as the upper-bound baseline) · Li et al., *FingerSafe*, arXiv:2208.10688 (attack-simulation preprocessing informing `pipeline/roi.py`) · Wronski et al., *Handheld Multi-Frame Super-Resolution*, SIGGRAPH 2019 (the principle behind `video/fuse.py`) · Stein, Bouatou & Busch, BIOSIG 2013 (video fingerphoto consolidation) · Chowdhury & Imtiaz, *J. Cybersecurity & Privacy* 2022 (field survey) · Kumbo et al., *MUST J. Res. & Dev.* 2025 (motivating prior work, explicitly lacking the resolution methodology this project supplies). Full annotated notes: `docs/research_notes.md`.
