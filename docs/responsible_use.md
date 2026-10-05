# Responsible use, data handling and consent

> **GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems.**

This document is part of the code. Each commitment below lists the place in the codebase that enforces it.

## 1. What GATTACA is for

People post photos of their hands (peace signs, rings, holding a coffee). Popular articles claim fingerprints can be lifted "from 1.5 m away" without any methodology. GATTACA gives an evidence-based answer for **your own photo**: how much ridge detail it exposes, where on the image, and what to change (crop, blur, lower resolution) before posting. The research output is a measured recoverability curve, so the public conversation can use numbers instead of folklore.

## 2. Threat model and what we deliberately do not build

We model a motivated adversary with a single public photo or short clip, who uses image enhancement and multi-frame fusion. We estimate the **upper end** of what they could recover, so the advice errs on the side of caution.

GATTACA does **not**, and must not be extended to:

| Not built | Enforced by |
|---|---|
| Ingest other people's photos: no URL fetch, scraping, crawler or social-media integration | `service/api.py` takes only a direct file upload plus an "this is my own hand" consent flag, and rejects requests without it. No HTTP client exists in `gattaca/`. |
| Export fingerprint templates (ISO/IEC 19794-2, ANSI-378, WSQ, vendor SDK formats) | The API returns minutiae **counts** and quality statistics only. Coordinates and angles never leave the process, including from `/api/minutiae`. |
| Expose minutiae extraction on a deployed public endpoint | `/api/minutiae` returns 404 unless `GATTACA_RESEARCH=1` is set on the host, so a public deployment cannot extract for strangers. Even in research mode it returns the rendered educational chart and counts, never coordinates. |
| Produce print-ready or binarised ridge images | The evidence map is a **heat-map blended over the original photo**, capped at 640 px and watermarked with the disclaimer (`gattaca/evidence.py`). The binarised and skeleton images exist only in local `--debug` dumps of research-dataset images. |
| Molds, 3D-printable fingers, gelatin/silicone/latex recipes, presentation-attack instruments | Out of scope. No code path exists. |
| Optimise anything to be accepted by commercial matchers or real sensors | The matcher is used only to **measure** recovery against paired ground truth (`study/`). No loss, search or tuning targets spoof acceptance. |

Any change that crosses these lines will be refused (see CLAUDE.md §0).

## 3. Data handling

- **Uploads:** decoded in memory with `cv2.imdecode`. Video needs a file for `cv2.VideoCapture`, so it goes to a `tempfile` that is deleted in a `finally:` block once scoring finishes (`service/api.py`, `gattaca/video/decode.py`). A test asserts that no file survives (`tests/test_api.py`).
- **No retention, no logging of raw biometrics:** the service logs only request metadata (size, duration, band). It never logs image bytes, ROIs, minutiae or evidence maps.
- **AWS:** if a deployment ever writes uploads to S3, the bucket has an expiry lifecycle rule (1 day, the minimum S3 lifecycle granularity) as a backstop, plus blocked public access and SSE. The service itself deletes immediately (`aws/infra`).
- **Research datasets:** used under their licences and never redistributed, committed or uploaded to public storage. The S3 dataset bucket is private and encrypted.
- **Results:** study CSVs contain only aggregate metrics per image (counts, scores, PPRP). They hold no images and no minutiae coordinates.

## 3a. Forensic research mode (education)

`python -m gattaca minutiae` and, with `GATTACA_RESEARCH=1`, `/api/minutiae` render the classical
AFIS chain (enhancement → thinning → crossing-number minutiae) and explain the ISO/IEC 19794-2
record fields. This is intended for forensic-science coursework and research on **lawfully held
prints**: one's own inked/livescan prints, lab exemplars, or licensed research datasets.

It remains out of scope, here and anywhere downstream, to tune any stage toward acceptance by a
real matcher or sensor, or to produce presentation-attack instruments (molds, 3D prints, gelatin
or silicone artefacts). Those are not part of minutiae education and are the actual dual-use risk;
measuring and visualising minutiae is not.

## 4. Consent

- Web UI and API: the user must tick or send `consent=true`, confirming "this is a photo of my own hand, and I consent to it being processed for this assessment". Without it the request is rejected (HTTP 400).
- Datasets: their subjects consented under the original collection protocols, and we use the data under the dataset licences.

## 5. Honest limitations (also in the report)

- The score estimates **exposure risk**, not a guarantee. A LOW score does not mean "safe to post", only that our measured pipeline, and the calibrated curve, recovered little.
- Until the licensed datasets arrive, calibration comes from **synthetic** prints. The UI labels this ("calibration source").
- A classical matcher gives a lower bound on adversary capability. Learned matchers (e.g. Ridgeformer) are stronger, so we bracket the risk and say so.
- Skin tone, lighting and camera pipelines vary. The study's illumination axis covers part of this, and the per-subgroup error analysis waits on real data.
- The "equivalent distance" in the message comes from a stated camera model (focal length, sensor width, megapixels). It changes with the phone and the zoom.

## 6. Disclaimer placement

The one-line disclaimer appears on: the web page header and footer, every evidence-map image (watermark), every API JSON response (`disclaimer` field), README, the technical report, and the demo video's title card.
