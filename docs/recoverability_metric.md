# The recoverability metric

> GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems.

## 1. Independent variable: pixels per ridge period (PPRP)

"Distance" means nothing without focal length, sensor width, resolution and later
resizing. What determines whether ridges survive is how many **pixels span one ridge
period** at the fingertip. An adult ridge period is ~0.45 mm, and the 500 ppi contact
standard gives ~9 px per period.

GATTACA measures PPRP in two independent ways on every fingertip ROI:

| Estimate | How | Works when |
|---|---|---|
| **Spectral** | Hann-windowed block FFTs; per block, the polar power spectrum is whitened (×f²); the ridge frequency is the radius of the strongest angular peak; parabolic sub-bin refinement; SNR-weighted median over blocks (`gattaca/pipeline/ridgefreq.py`) | ridges are visible at all |
| **Geometric** | finger width (px) / 36 ridge periods across a typical fingertip (`gattaca/analyze.py`) | always, including heavy blur, so "too few pixels" can be separated from "enough pixels but blurred" |

Validation on synthetic photos with known PPRP: the spectral estimate tracks the truth from
10 down to 2.5 px/period, reading ~9% low because of cylindrical foreshortening off-centre. At
2 px/period it saturates at the Nyquist floor (~2.2), the physically correct answer:
ridges above Nyquist alias. The pipeline then **resamples every ROI to 9 px/period** before
enhancement, an automatic version of Lin & Kumar's scale-correction step.

## 2. Dependent variable: was the finger recovered?

For each degraded probe, the full pipeline extracts minutiae, which are matched against
**every** gallery ground-truth contact print (genuine + impostors) using the score
`n² / (N_P · N_Q)`. We report:

* **rank-1 identification**: the genuine finger scores highest in the gallery (headline);
* **verification EER** from genuine vs impostor score distributions per condition;
* minutiae recovered, mean minutiae quality, ridge clarity, all per condition, in `study/results/*.csv`.

## 3. Ridge clarity (the heat map)

`clarity = Gabor-fit × periodicity × coherence-gate`, per pixel:

* **Gabor fit**: local normalised correlation between the image and its oriented-Gabor reconstruction;
* **periodicity**: `(r(T) − r(T/2)) / 2` from the local autocorrelation along the ridge
  normal (+1 for a ridge train; ~0 for a single edge, a specular highlight bar, a silhouette or noise);
* **coherence gate**: structure-tensor coherence over a window of several periods.

Each factor exists because a failure was observed without it. Band-limited noise fits its own
Gabor output (clarity 0.68 without the coherence gate, 0.01 with it). A specular highlight stripe passed both
(0.51 on a ridge-free "protected" finger; 0.00 with periodicity). Clean ridges score ~0.9.

## 4. From measurement to HIGH / MEDIUM / LOW

`study/calibrate.py` fits an L2-regularised logistic model of P(rank-1 success) on five
inference-time features: log₂ PPRP, clarity, coherence, log minutiae count, log spectral SNR.
**Confidence** is the fraction of 200 finger-level bootstrap refits that place the
image in the same band as the point estimate. Bands: HIGH P ≥ 0.7, LOW P < 0.3.

### Current calibration: synthetic (instrument validation, NOT the headline result)

60 synthetic fingers × 47 conditions = 2,734 probe-conditions against a 60-finger gallery
(chance rank-1 = 1.7%). Matcher: rigid + DCM + RTPS-lite.

| PPRP | 10 | 8 | 6 | 5 | 4 | 3.5 | 3 | 2.5 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| rank-1 | 1.00 | 1.00 | 1.00 | 0.98 | 0.97 | 0.88 | 0.58 | 0.17 | 0.07 |
| 95% CI low | 0.82 | 0.94 | 0.94 | 0.91 | 0.89 | 0.78 | 0.46 | 0.09 | 0.03 |
| EER | 0.05 | 0.05 | 0.07 | 0.07 | 0.08 | 0.10 | 0.17 | 0.36 | 0.45 |

* Crossings: **50% at 2.9 px/ridge, 80% at 3.4, 95% at 3.9**.
* **A JPEG q75 re-encode costs ~1 px/ridge**: crossings move to 3.75 / 4.5 / 5.5.
* Defocus σ ≥ 0.3 ridge periods or motion ≥ 1 period destroys recovery even at full resolution.
  Sensor noise up to σ = 8 and JPEG down to q30 are tolerated. Under-exposure to −5 stops is
  tolerated *on synthetic data*, which lacks real shadow noise, so expect real data to be harsher.
* Held-out quality (5-fold, grouped by finger): accuracy 0.914, Brier 0.065. Observed rank-1
  inside each predicted band: **HIGH 96%, MEDIUM 46%, LOW 8%**.
* Deformation correction (DCM, RTPS) makes **no measurable difference on synthetic data**:
  their systematic distortion sits inside the 14 px match tolerance. Real contact prints, with
  pressure and rolling distortion, are where Lin & Kumar report the gain. Re-test there.

Figures: `study/plots/synthetic/` (each PNG has a CSV table beside it).

## 5. Converting PPRP to an "equivalent distance" (public message only)

`PPRP = 0.45 mm × image_width_px × f_35mm / (36 mm × distance)`, assuming a 36 mm-wide-equivalent
sensor, the fingertip in focus and facing the camera, and lens sharpness no worse than the pixel
grid. Real lenses and phone ISPs are softer, so treat these as **upper bounds** on distance.

| Camera (fingertip in focus, facing camera) | 95% rank-1 | 80% | 50% |
|---|---|---|---|
| 12 MP main camera, 26 mm | 34 cm | 39 cm | 45 cm |
| 48 MP main camera, full-res | 67 cm | 78 cm | 90 cm |
| 3× telephoto, 77 mm, 12 MP | 100 cm | 115 cm | 134 cm |
| 5× telephoto, 120 mm, 12 MP | 155 cm | 180 cm | 209 cm |
| 12 MP main, **posted at 1080 px with JPEG q75** | 6 cm | 8 cm | 9 cm |

**"100% at 1.5 m / 50% at 3 m":** at 1.5 m a 12 MP main camera gives 0.87 px/ridge, below
Nyquist and unrecoverable. Only a 5× telephoto on the original file (4.0 px/ridge at
1.5 m) gets there, and at 3 m even that drops to 2.0 px/ridge. The folklore numbers
are only plausible for zoomed, uncompressed originals. Platform resizing is a strong
(accidental) protection. *These conclusions come from the synthetic calibration and must be
re-confirmed on HKPolyU / RidgeBase before they are published.*

## 6. Known limitations

* **Synthetic calibration.** Real skin, sweat pores, creases, cameras' ISP sharpening and denoising
  will move the curve. This is the first thing to re-run when datasets arrive (`python -m study.run_study --dataset hkpolyu`).
* **Classical matcher = lower bound** on adversary capability. A learned matcher (Ridgeformer
  reports 2.83% EER on HKPolyU CL→CB vs 14.33% for RTPS+DCM) would shift the curve toward lower PPRP.
* **Worst-case agent bias.** The agent takes the maximum over several strategies (best frame, fusion,
  retuned enhancement). Max-of-noisy-estimates is optimistically biased. That is intentionally conservative for
  privacy advice, but the agent's P is not the same calibrated quantity as a single-image P.
* **Geometric PPRP** assumes a typical adult fingertip width (36 periods). Children and very wide or narrow
  fingers shift it by roughly ±20%.
* Off-axis fingertips (rolled > ~45°) and partial occlusion are not modelled in the ladders.
