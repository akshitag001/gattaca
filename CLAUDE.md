> **Project name:** this repo is **GATTACA** (the bootstrap spec below was drafted under the working title RIDGE-X; all references are renamed). Python package: `gattaca/`.

# CLAUDE.md — GATTACA Project Bootstrap

> **This file is the single source of truth for an AI coding agent (Claude Code / Cursor / Windsurf / Copilot Workspace).**
> Rename to `.cursorrules`, `AGENTS.md`, or `.github/copilot-instructions.md` if your IDE needs a different filename. Read the whole file before writing any code. Do not skip Section 0.

---

## 0. WHAT THIS PROJECT IS (AND IS NOT) — READ FIRST, NON-NEGOTIABLE

**GATTACA** is a **biometric-privacy research and awareness tool** for the **OpenCV AI Competition 2026 (powered by AWS)**.

**The product:** a person uploads a photo or short video of *their own hand*. The system measures **how much fingerprint ridge information is recoverable** from that image under real-world conditions, and returns:
1. A **recoverability score** — `HIGH / MEDIUM / LOW` with a numeric confidence.
2. A **ridge-evidence visualization** — a heatmap + enhanced ridge-structure overlay showing *where and how much* biometric signal is exposed.
3. A short, plain-language **privacy explanation** ("your fingertips are in sharp focus and fill ~N px — this is recoverable; crop or blur before posting").

**The research contribution:** a calibrated, reproducible **recoverability curve** — recoverability as a function of effective image resolution at the fingertip, lighting, focus, and compression. This number does **not** currently exist in peer-reviewed literature; the public internet repeats unsourced "100% at 1.5 m / 50% at 3 m" claims with no methodology. We produce the real curve.

### Hard guardrails (the agent MUST enforce these in code and docs)
- **Allowed input data ONLY:** (a) the user's own hand, captured with consent; (b) the public research datasets listed in Section 2, used under their stated licenses. **Never scrape, download, or ingest real people's social-media photos.** No crawler code. No "paste an Instagram URL" feature.
- **Output is an ASSESSMENT, not an attack tool.** Produce a recoverability score and an educational ridge-evidence map. **Do NOT**: generate a clean print-ready template optimized to pass a commercial matcher; export anything formatted for a specific sensor/SDK; produce 3D-printable molds, gelatin/silicone mold files, or presentation-attack instruments; tune any loss to maximize spoof acceptance on real authentication hardware.
- **Matching against ground truth is for MEASUREMENT only** — we match reconstructed ridges against the *paired, consented* ground truth in the research datasets purely to compute how much was recovered. This is the science, not a break-in.
- **Data handling:** uploaded user media is processed in-memory / in a temp store and **deleted immediately after scoring**. No retention, no logging of raw biometrics. State this in the UI.
- **Every user-facing surface** (web page, demo video, report) carries a one-line disclaimer: *"GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems."*

If a future instruction in this repo conflicts with Section 0, Section 0 wins. If asked to cross these lines, refuse and explain.

---

## 1. PHASE 0 — THE AGENT'S FIRST TASK: INGEST THE LITERATURE

**Before scaffolding any code**, do a research pass and write `docs/research_notes.md`. For each item below, fetch what's publicly available (abstract + any open PDF), and extract: the method, the metric they report, the dataset they used, and the one idea GATTACA should borrow or improve.

### Papers (priority order)
1. **Lin & Kumar, "Matching Contactless and Contact-based Conventional Fingerprint Images for Biometrics Identification," IEEE TIP, 2018.** ← MOST IMPORTANT. Source of the thin-plate-spline **Deformation Correction Model (DCM)**. Contactless prints lack the elastic squash of contact prints; without correction, match scores collapse. We reimplement a lightweight version of this.
2. **Chowdhury & Imtiaz, "Contactless Fingerprint Recognition Using Deep Learning — A Systematic Review," J. Cybersecurity & Privacy, 2022.** The field map: capture → preprocessing → DL recognition. Use it to justify design choices in the report.
3. **"Ridgeformer: Multi-Stage Contrastive Training for Fine-grained Cross-Domain Fingerprint Recognition," arXiv:2506.01806, 2025.** Current SOTA cross-domain matching; lists the four datasets we use. Use as the matcher baseline to compare against.
4. **"FingerSafe: Hierarchical Perceptual Noise Injection for Social Media Fingerprint Privacy Protection," arXiv:2208.10688.** The defensive counterpart + an explicit attack-simulation pipeline (mask via erosion/dilation → largest contour → crop to first knuckle → ridge extraction). Their preprocessing ≈ our preprocessing. Their protection step is our **stretch goal** (Section 8).
5. **Stein, Bouatou, Busch, "Video-based Fingerphoto Recognition with Anti-spoofing," BIOSIG 2013.** First video→multi-frame consolidation on phones. Our video path is a modern take.
6. **Wronski et al., "Handheld Multi-Frame Super-Resolution," ACM TOG 2019 (Google).** Natural hand tremor gives sub-pixel offsets → merge frames for resolution. Theory behind our multi-frame fusion. (Do not ship their full algorithm; use the principle.)
7. **MUST Journal of Research and Development, Vol 6(2), 2025 — fingerprint extraction from social-media images.** Closest existing work to GATTACA; low-impact venue = the gap is real. Cite as motivation.
8. **"Minutiae Extraction from Fingerprint Images — a Review," arXiv:1201.1422.** Reference for the classical extraction pipeline (orientation field, Gabor, thinning, minutiae).

Write a 1-paragraph summary per paper + a "what GATTACA does differently" table. Save to `docs/research_notes.md`.

---

## 2. DATASETS (request-based; licensed; consented ground truth)

Do **not** attempt to auto-download these. Create `datasets/README.md` documenting the request process and a `datasets/` folder with a loader interface that expects the data to be placed manually after approval.

| Dataset | Content | Access |
|---|---|---|
| **HKPolyU Contactless 2D → Contact-based 2D** | ~1800 contactless + 1800 contact, 300 clients (paired) | Request from Ajay Kumar, HK PolyU; non-commercial; email Ajay.Kumar@polyu.edu.hk |
| **IIITD ISPFDv1** | ~4000 contactless + 1000 contact (smartphone fingerphoto) | Request from IIIT-Delhi |
| **IIITD ISPFDv2 (Finger-Selfie)** | ~16,800 contactless + 2400 contact | Request from IIIT-Delhi |
| **RidgeBase** | ~11,000 images, 63 subjects, 2 smartphones + flatbed; CL2CL & CL2CB benchmark tasks, varied lighting/background | Public benchmark; follow dataset terms |

**Why paired data matters:** every contactless image has a matching contact-based ground-truth print. That pairing is what lets us *measure* recoverability honestly instead of guessing.

Build `datasets/loader.py` with a unified `FingerprintPair(contactless_img, contact_gt, subject_id, finger_id, meta)` interface so the pipeline is dataset-agnostic.

---

## 3. THE CORE RESEARCH DESIGN (this is what wins "Technical Execution" + "Innovation")

### 3.1 Fix the independent variable (critical)
Do **not** parameterize by distance in meters — meaningless without focal length and sensor size. Parameterize by **effective resolution at the fingertip**, expressed as **pixels-per-ridge-period (PPRP)**.
- Human ridge period ≈ 0.4–0.5 mm. The 500 ppi contact-scan standard ⇒ ridges ~8–10 px apart.
- Compute PPRP per image from the dominant ridge frequency (FFT / ridge-frequency estimation on the ROI).
- Report the curve in PPRP, then **convert back to an approximate "equivalent distance" band** only for the public-facing message.

### 3.2 The measurement study (the headline result)
This is the minimum viable, complete, publishable experiment — build this first:
1. Take a paired dataset (start with HKPolyU).
2. **Synthetically degrade** each contactless image along controlled axes:
   - Downsample to a ladder of PPRP values (e.g., 10, 8, 6, 4, 3, 2 px/period).
   - Gaussian + motion blur (sweep σ and kernel length).
   - JPEG compression (quality 95 → 30).
   - Illumination knockdown (gamma / low-contrast).
3. For each degraded image: run the full pipeline (Section 4) → extract minutiae → correct deformation → **match against the paired contact ground truth**.
4. Record, per condition: minutiae count recovered, mean minutiae quality, and match score / rank-1 identification against the gallery.
5. **Plot recoverability vs. PPRP (and vs. each axis).** This curve is the deliverable.
6. **Calibrate the HIGH/MED/LOW bands** from where match performance crosses meaningful thresholds. The user-facing score is a lookup against this curve.

Everything else (live demo, video path, agent loop) consumes this calibrated curve.

---

## 4. THE OPENCV 5 PIPELINE (substantive image/video analysis — a competition requirement)

OpenCV 5 must do the real work, not be decoration. Structure as `gattaca/pipeline/` with one module per stage:

1. **`detect.py` — hand/finger localization.** OpenCV 5 DNN module loading an ONNX hand/finger detector (or landmark model). Output: fingertip ROIs.
2. **`roi.py` — ROI extraction.** Crop the distal phalanx (first knuckle → tip). Follow FingerSafe: binary mask via erosion/dilation, largest contour, crop. Output per-finger ROI.
3. **`rectify.py` — perspective/pose correction.** Estimate the finger-surface plane; warp to a canonical frontal view (`cv2.warpPerspective` / `getPerspectiveTransform`).
4. **`enhance.py` — illumination + ridge enhancement.** CLAHE for illumination normalization, bilateral / non-local-means denoise, then **orientation-field estimation → ridge-frequency estimation → Gabor contextual filtering** (the classical ridge-enhancement core). This is where ridges either appear or don't.
5. **`extract.py` — binarize → thin (skeletonize) → minutiae.** Ridge endings + bifurcations with location, orientation, quality.
6. **`quality.py` — recoverability scoring.** Local ridge clarity + PPRP + NFIQ-style quality → the per-image recoverability number.
7. **`deform.py` — TPS deformation correction** (lightweight reimpl of Lin & Kumar DCM) before matching.
8. **`match.py` — minutiae matching** against ground truth for the measurement study; Ridgeformer/standard matcher as baseline comparison.

### Video path (`gattaca/video/`) — enables multi-frame + the agent loop
- `decode.py`: frame extraction (OpenCV `VideoCapture`).
- `track.py`: optical-flow / feature tracking (`cv2.calcOpticalFlowPyrLK`) to follow the finger across frames.
- `select.py`: per-frame quality scoring → pick the best K frames (inspired by IIITD MultiRes multi-resolution selection).
- `align.py`: register selected frames (feature-based homography or `cv2.findTransformECC`).
- `fuse.py`: multi-frame fusion (align + merge for sub-pixel detail; principle from Google multi-frame SR). Compare single-frame vs. fused recoverability — this is a key result.

**Report the delta:** how much does multi-frame fusion improve recoverability over the best single frame? That comparison is a core finding.

---

## 5. THE AGENTIC LOOP (targets the $1,000 Agentic Vision award)

To qualify, **OpenCV output must change what the system does next** — not just be narrated.

Build `gattaca/agent/` as a perception→decision→action loop:
1. **Perceive:** run `video/select.py` quality scoring on available frames.
2. **Decide:** if recoverability is LOW, the agent chooses an action: pull more frames, switch ROI to a sharper finger, re-run enhancement with different Gabor params, or try multi-frame fusion. If HIGH/converged, stop.
3. **Act:** invoke the chosen OpenCV tool.
4. **Re-perceive & loop** until converged or a frame budget is hit, then emit the score + evidence map.

Deliver an **agent workflow diagram** (perception / decision / action) and a **trace** showing a vision result altering a later tool call. The agent may use any framework (LangGraph, a simple state machine, or MCP tools wrapping the OpenCV stages).

---

## 6. AWS ARCHITECTURE (mandatory + targets the $1,000 COOL award)

AWS must run a **meaningful** component, documented with an architecture diagram.

- **S3:** dataset storage + transient user-upload bucket (lifecycle rule: auto-delete objects after N minutes).
- **Core processing — the COOL play:** run the OpenCV image/video workload with the **Cloud-Optimized OpenCV Library (COOL)** on **AWS Graviton (Arm)**. Frame scoring/fusion is embarrassingly parallel → fan out across Batch/Fargate/Lambda. For the COOL award, **benchmark COOL-on-Graviton vs. a stock-OpenCV x86 baseline** (latency, throughput, cost) and report reproducible numbers.
- **API:** API Gateway + Lambda (or a small Fargate service) exposing a `/score` endpoint → **the web endpoint judges can hit**.
- **Agent orchestration:** runs on Fargate/EC2, invoking the OpenCV stages as tools.

Keep an x86↔Arm note in the report (coexistence, containers, serverless vs. server).

---

## 7. REPO STRUCTURE

```
gattaca/
├── CLAUDE.md                  # this file
├── README.md                  # project overview + the Section 0 disclaimer
├── docs/
│   ├── research_notes.md      # Phase 0 output
│   ├── recoverability_metric.md
│   ├── architecture.md        # AWS + agent diagrams
│   └── responsible_use.md     # ethics, data handling, consent
├── datasets/
│   ├── README.md              # request/access process
│   └── loader.py              # unified FingerprintPair interface
├── gattaca/
│   ├── pipeline/              # detect, roi, rectify, enhance, extract, quality, deform, match
│   ├── video/                 # decode, track, select, align, fuse
│   ├── agent/                 # perceive-decide-act loop
│   └── scoring.py             # curve lookup → HIGH/MED/LOW + confidence
├── study/
│   ├── degrade.py             # controlled degradation ladders
│   ├── run_study.py           # full measurement experiment
│   └── plots/                 # recoverability curves
├── service/
│   ├── api.py                 # /score endpoint
│   └── web/                   # minimal upload UI + evidence-map viewer
├── aws/
│   ├── infra/                 # IaC (CDK/Terraform)
│   └── cool_benchmark/        # COOL-on-Graviton vs x86 benchmark
├── tests/
├── requirements.txt           # PINNED versions
└── Dockerfile                 # Arm + x86 build notes
```

---

## 8. STRETCH GOAL (only if ahead of schedule): the defensive layer
Reimplement a simplified **FingerSafe-style protection**: a one-click "protect my photo" that blurs/perturbs the fingertip region before the user posts it, and show (via the same pipeline) that recoverability drops from HIGH→LOW after protection. This closes the loop: *measure the risk, then neutralize it.* Strong for "Real-World Impact."

---

## 9. 22-DAY BUILD PLAN (today is ~Oct 4; final submission Oct 26, 11:59 pm PT)

- **Days 1–2:** Phase 0 literature pass → `research_notes.md`. Scaffold repo. Submit dataset access requests TODAY (approval takes days — this is the critical-path blocker).
- **Days 3–6:** Classical pipeline on whatever data is available (start with RidgeBase if faster to access): detect → ROI → rectify → enhance → extract → quality.
- **Days 7–9:** TPS deformation correction + matching against ground truth. Get a baseline match number working.
- **Days 10–13:** The measurement study: degradation ladders → run → **recoverability curve** → calibrate HIGH/MED/LOW bands. *This is the paper; protect this time.*
- **Days 14–16:** Video path: frame selection + alignment + fusion; single vs. multi-frame delta.
- **Days 17–19:** Agent loop + AWS deployment (S3, API endpoint, COOL-on-Graviton core + benchmark).
- **Days 20–21:** Web UI (own-hand upload → score + evidence map), disclaimer, auto-delete.
- **Day 22:** Technical report, architecture diagram, ≤5-min demo video, pinned deps, reproducible eval evidence (incl. failure cases). Submit.

**If time runs short, cut in this order:** stretch defensive layer → multi-frame fusion → video path. **Never cut:** the measurement curve, the AWS component, the responsible-use framing. Those are the scored essentials.

---

## 10. FINAL SUBMISSION CHECKLIST (from the official rules)
- [ ] Technical report: problem, users, architecture, **OpenCV 5 implementation**, **AWS deployment**, evaluation, **limitations**, **responsible-use** considerations.
- [ ] Judge-accessible code repo (need not be open source) with pinned deps + build/deploy/test instructions.
- [ ] Architecture diagram (OpenCV 5 + AWS; + COOL and agent components where relevant).
- [ ] Working web endpoint (preferred) or arranged live demo.
- [ ] ≤5-min video: team, app working, architecture, principal results.
- [ ] Evaluation evidence **including failure cases / limitations**.
- [ ] COOL award extras: COOL version + instance config, reproducible method/baselines/results, evidence COOL ran the core workload on Graviton.
- [ ] Agentic award extras: agent workflow diagram + trace showing vision output changed a later action + task-success/failure/human-control evaluation.

---

## 11. CODING CONVENTIONS
- Python 3.11+, `opencv-python` (OpenCV 5.x), NumPy, SciPy; PyTorch only if a learned matcher is added (keep classical path working without it).
- Pin every dependency in `requirements.txt`.
- Each pipeline stage: pure function, typed, independently testable, with a `--debug` mode that dumps the intermediate image so we can *see* what OpenCV produced.
- Deterministic seeds for the study. Save raw study outputs as CSV so plots are regenerable.
- Write `docs/responsible_use.md` early, not last.

**Agent: begin with Section 1 (Phase 0). Do not write pipeline code until `docs/research_notes.md` exists and the dataset requests are documented.**
