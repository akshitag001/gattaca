# GATTACA — Phase 0 Research Notes

> GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems.

Literature pass done 2026-10-04. For each paper: **method**, **metric**, **dataset**, and **the one idea GATTACA borrows or improves**. Where the open PDF was read directly, numbers are quoted from its tables; where only the abstract was reachable, that is stated.

---

## 1. Lin & Kumar — Matching Contactless and Contact-based Conventional Fingerprint Images for Biometrics Identification (IEEE TIP 27(4):2008–2021, 2018) — *full PDF read*

**Method.** Pipeline: enhancement + **scale correction** → minutiae extraction → finger-impression type/intensity estimation → **deformation correction** of the *contact* print → minutiae (+ minutiae-related ridge) matching. The core contribution is the **Robust Thin-Plate Spline (RTPS)**: ordinary TPS interpolates control points exactly but mislocalises everything between them; RTPS adds a localisation-error term over extra marked points (`E_error`) to the TPS bending energy and iterates (per-point weights ω ∈ [0, 0.3]) until `E_error` stops decreasing for three iterations. A **generalised Deformation Correction Model (DCM)** is trained offline from contact↔contactless pairs: the finger is divided into blocks around its centre (a cylindrical-finger assumption), each block gets a mean displacement, and an unseen contact print is corrected by adding its block's displacement to every minutia. Matching score **S = n² / (N_P · N_Q)** (n = matched pairs).

**Metric / numbers (PolyU CL→CB, Tables I–II).** Cross-matching EER: NBIS 36.30% → MCC 38.90% → minutiae+TPS+DCM 18.12% → **minutiae+ridge+RTPS+DCM 14.33%**. Rank-1: NBIS 16.61% → **RTPS+DCM 66.67%**. On the 2D/3D benchmark: EER 40.34% (NBIS) → 19.81%. Average minutiae per print ≈ 29.

**Dataset.** New PolyU Contactless 2D → Contact-based 2D database: 1800 contactless (1400×900) + 1800 contact (URU4000, 356×328, 500 dpi), 300 fingers × 6 samples.

**What GATTACA borrows.** (a) Scale correction before matching — we do it *automatically* from the measured ridge period (PPRP), so every probe is resampled to a canonical 500-ppi-equivalent period. (b) A lightweight **RTPS-lite** (robust, iteratively re-weighted TPS with the paper's "stop after three non-improving iterations" rule) and a block-wise **DCM** learned from training pairs. (c) Their score `n²/(N_P·N_Q)` so our numbers are comparable. **Improvement:** they report one operating point at full resolution; we sweep resolution, blur, compression and lighting to get the *curve*.

## 2. Chowdhury & Imtiaz — Contactless Fingerprint Recognition Using Deep Learning: A Systematic Review (J. Cybersecurity & Privacy 2(3):714–730, 2022) — *abstract only (publisher PDF blocked)*

**Method.** PRISMA-style systematic review; **8 articles** met inclusion criteria. Organises the field into (1) capture devices/sensors, (2) classical preprocessing (segmentation, enhancement, scale/orientation normalisation), (3) deep-learning recognition.

**Metric.** Narrative comparison; DL methods report higher accuracy than classical/ML counterparts, but with gaps for real-world deployment (generalisation, robustness to capture conditions).

**Dataset.** Surveys the public sets (PolyU, IIITD fingerphoto sets, etc.).

**What GATTACA borrows.** The capture → preprocess → recognise decomposition justifies our stage-per-module design, and its "robustness to capture conditions" gap is precisely the axis we measure. We keep a **classical, interpretable** pipeline as the measurement instrument (every stage can dump its intermediate image) and treat DL matchers as an upper-bound baseline.

## 3. Pandey, Jawade, Setlur — Ridgeformer: Multi-Stage Contrastive Training for Fine-grained Cross-Domain Fingerprint Recognition (arXiv:2506.01806, ICIP 2025) — *full PDF read*

**Method.** ViT backbone. Stage 1: global embedding (patch tokens → self-attention → GAP → MLP) trained contrastively. Stage 2: cross-attention between contactless and contact token sets for fine-grained alignment, cosine similarity on pooled output.

**Metric / numbers (Tables 1–2).** HKPolyU CL→CB: EER **2.83%**, TAR@FAR=1% 89.34, Rank-1 **87.40%** (vs Verifinger EER 19.31%, Rank-1 80.73%; RTPS+DCM EER 14.33%, Rank-1 66.67%). RidgeBase Task 1 CL→CB: EER 5.25%, Rank-1 69.90% (Verifinger 72.50%); CL→CL: EER 7.60%, Rank-1 100%.

**Dataset.** HKPolyU, RidgeBase, ISPFDv1, ISPFDv2 (fingers segmented with SAM+CLIP for ISPFD).

**What GATTACA borrows.** It is the **upper-bound matcher**: our classical matcher gives a *lower bound* on what an adversary recovers; a learned matcher shifts the curve left (recoverable at lower PPRP). Reporting both brackets the true risk. Note also: Verifinger beats Ridgeformer on RidgeBase CL→CB Rank-1 — cross-domain matching is not solved, which supports measuring rather than assuming.

## 4. Li et al. — FingerSafe: Hierarchical Perceptual Noise Injection for Social Media Fingerprint Privacy Protection (arXiv:2208.10688) — *HTML read*

**Method.** Adversarial protective noise in two tiers: perturb the **orientation field** (high-level ridge semantics shared across models → transferable) while regularising the response of a **Lateral Geniculate Nucleus** model to suppress visible low-level contrast changes. Includes an explicit attack-simulation pipeline: *binary mask via erosion/dilation → largest contour → crop to the first knuckle → ridge extraction (MHS / HG / Frangi) → matcher.*

**Metric.** Protection rate = drop in identification accuracy / verification TPR: up to **94.12%** (digital) and **68.75%** (after real Twitter/Facebook upload).

**Dataset.** HKPolyU (336 subjects, colourised); 1,000 self-collected iPhone 12 Pro images (100 identities).

**What GATTACA borrows.** Their segmentation recipe is our `roi.py`. Their protection idea motivates our **stretch "protect my photo"** (we use a simpler, non-adversarial ridge-band suppression and verify with the same pipeline that recoverability drops). Their 68.75% real-world number shows platform recompression matters — hence our JPEG axis.

## 5. Stein, Bouatou, Busch — Video-based Fingerphoto Recognition with Anti-spoofing Techniques with Smartphone Cameras (BIOSIG 2013) — *abstract/secondary sources*

**Method.** Android prototype: finger detection, segmentation and multi-frame consolidation from smartphone video, plus presentation-attack detection.

**Metric.** Reported EER ≈ **1.2%** for the recognition component (≈3.0% cited in some secondary sources for a different configuration).

**Dataset.** Self-collected smartphone video.

**What GATTACA borrows.** Video gives many chances to catch a sharp frame. Our `video/select.py` scores every frame and keeps the best K; the agent decides when more frames are worth pulling. (We do not implement anti-spoofing — out of scope and not our threat model.)

## 6. Wronski et al. — Handheld Multi-Frame Super-Resolution (ACM TOG / SIGGRAPH 2019) — *abstract read*

**Method.** Natural hand tremor gives sub-pixel offsets between burst frames. Align the burst, then merge with **anisotropic kernel regression** directly from raw data, with a **robustness model** that down-weights pixels inconsistent with the reference (local motion/occlusion).

**Metric.** Resolution/SNR gains; ~100 ms per 12-MP frame on phones (powers Super-Res Zoom).

**What GATTACA borrows (principle only).** `video/fuse.py`: ECC alignment to sub-pixel accuracy → **splat each frame's pixels onto a 2× grid at their true sub-pixel positions** (isotropic Gaussian kernel regression) → robustness weighting against the reference. We report the **fused-vs-best-single-frame recoverability delta** — the honest answer to "can several mediocre frames reveal what one frame hides?".

## 7. Kumbo et al. — The Potential Risks of Using Social Media: How Sharing Pictures with a Particular Pose Could Expose Biometric Data (MUST J. Res. & Dev. 6(2):270–282, 2025, doi:10.62277/mjrd2025v6i20016) — *full PDF read*

**Method.** Exploratory/qualitative: content analysis of public Instagram/Facebook photos plus interviews; describes contrast enhancement, noise reduction and ridge extraction conceptually.

**Metric.** None quantitative. Their own limitations section: *"the study does not provide a detailed examination of how resolution impacts fingerprint extraction accuracy, which could be an area for future research"*, and forensic-tool evaluation "was not part of this study".

**What GATTACA does differently.** That stated gap **is our research contribution**: a measured, ground-truth-matched recoverability curve vs resolution, focus, compression and lighting. We also do the opposite of their data sourcing: **no social-media scraping**; only consented research datasets and the user's own hand.

## 8. Bansal, Sehgal, Bedi — Minutiae Extraction from Fingerprint Images: a Review (IJCSI 8(5), 2011; arXiv:1201.1422) — *abstract read*

**Method.** Taxonomy of minutiae extraction: on **binarised + thinned** images (crossing number, morphology) vs **directly on grey-level** images (ridge following); preceded by enhancement (normalisation, orientation field, frequency, Gabor filtering) and followed by false-minutiae removal.

**What GATTACA borrows.** The classical chain in `enhance.py` / `extract.py`: normalise → CLAHE → orientation field (gradient structure tensor) → ridge frequency → oriented Gabor bank → binarise → Zhang-Suen thinning (`cv2.ximgproc.thinning`) → crossing number → false-minutiae pruning (border, short-distance pairs).

---

## What GATTACA does differently

| Prior work | What they established | Gap | GATTACA |
|---|---|---|---|
| Lin & Kumar 2018 (RTPS+DCM) | CL→CB matching needs deformation + scale correction; EER 14.33% at full res | One operating point; manual scale handling | Automatic scale from measured ridge period; RTPS-lite + DCM; **sweep the degradation axes** |
| Ridgeformer 2025 | Learned cross-domain matcher, EER 2.83% (PolyU) | Evaluated on clean data only | Used as the **upper-bound** matcher to bracket risk |
| Chowdhury & Imtiaz 2022 | Field map; DL > classical; robustness gap | No quantitative robustness study | Robustness *is* the study |
| FingerSafe 2022 | Attack-sim pipeline; adversarial protection | Measures protection, not baseline exposure vs. capture conditions | Same preprocessing; measures exposure first; simple protection as stretch |
| Stein et al. 2013 | Video fingerphoto consolidation | Pre-deep-learning, cooperative capture | Agentic frame selection + fusion with a measured delta |
| Wronski et al. 2019 | Tremor → sub-pixel SR | Not applied to biometric exposure | Fused-vs-single recoverability delta |
| Kumbo et al. 2025 | Qualitative social-media fingerprint risk | No resolution analysis, no ground truth, scraped data | **Quantitative PPRP curve with paired, consented ground truth** |
| Internet folklore ("100% at 1.5 m") | — | No methodology; distance is meaningless without optics | Independent variable is **pixels-per-ridge-period**; distance only as a derived, camera-specific band |

## Design decisions these notes fix

1. **Independent variable = PPRP**, measured per image from the dominant ridge frequency (block FFT). Distance appears only in the user-facing message, via an explicit camera model (see `docs/recoverability_metric.md`).
2. **Scale normalisation from PPRP** before extraction (Lin & Kumar's scale-correction step, automated).
3. **Matcher**: rigid alignment hypotheses from local minutia descriptors → greedy pairing → RTPS-lite refinement → `n²/(N_P·N_Q)`; Ridgeformer as optional upper bound.
4. **Video**: ECC sub-pixel alignment + kernel-regression splatting with robustness weighting.
5. **Ethics**: research datasets under licence + the user's own hand only; outputs are scores and overlay maps, never extractable templates.

## Sources
- Lin & Kumar 2018: https://www4.comp.polyu.edu.hk/~csajaykr/myhome/papers/TIP2018.pdf
- Chowdhury & Imtiaz 2022: https://par.nsf.gov/biblio/10395541 (doi:10.3390/jcp2030036)
- Ridgeformer: https://arxiv.org/abs/2506.01806
- FingerSafe: https://arxiv.org/abs/2208.10688
- Stein et al. 2013: https://dl.gi.de/handle/20.500.12116/17696
- Wronski et al. 2019: https://arxiv.org/abs/1905.03277
- Kumbo et al. 2025: https://mjrd.must.ac.tz/index.php/mjrd/article/download/122/165/1092
- Bansal et al. 2011: https://arxiv.org/abs/1201.1422
- RidgeBase: https://arxiv.org/abs/2307.05563
- COOL on AWS Marketplace (Graviton4, v3.1, OpenCV 5.0): https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs
