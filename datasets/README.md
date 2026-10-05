# Datasets — access, licensing, layout

> GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store biometrics or defeat security systems.

**Nothing in this repo downloads these datasets.** Each is licensed for non-commercial research, released only on request. Place the data here by hand once approved. Never commit dataset images: `datasets/*/` is git-ignored.

Every dataset is **paired**: each contactless photo has a contact-based ground-truth print of the same finger, captured with consent. That pairing is what makes the study a *measurement*: we match what our pipeline recovers against a known answer.

| Key | Dataset | Content | Access | Status |
|---|---|---|---|---|
| `hkpolyu` | HK PolyU Contactless 2D → Contact-based 2D v1.0 | 1800 contactless + 1800 contact, 300 fingers × 6 | Request from Prof. Ajay Kumar (Ajay.Kumar@polyu.edu.hk); page: https://www4.comp.polyu.edu.hk/~csajaykr/fingerprint.htm | ☐ requested ☐ approved |
| `ispfd1` | IIITD SmartPhone Fingerphoto DB v1 | ~4000 fingerphotos + live-scan, 64 subjects × 2 fingers | Signed licence → databases@iab-rubric.org (https://iab-rubric.org) | ☐ requested ☐ approved |
| `ispfd2` | IIITD SmartPhone Finger-selfie DB v2 | ~16,800 finger-selfies + 2432 live-scan, 304 fingers | Signed licence (head of institution/registrar must sign) → databases@iab-rubric.org, subject `License agreement for IIITD SmartPhone Finger-selfie Database v2 (ISPFDv2)` | ☐ requested ☐ approved |
| `ridgebase` | RidgeBase (UB CUBS) | 15k+ CL/CB pairs, 88 people, 2 phones + flatbed; CL2CL & CL2CB tasks | Licence form at https://www.buffalo.edu/cubs/research/datasets/ridgebase-benchmark-dataset.html | ☐ requested ☐ approved |
| `synthetic` | Built-in synthetic pairs (`gattaca.synth`) | Unlimited, procedurally generated | None, ships with the code | ✅ always available |

**Critical path:** send all four requests on day 1. The IIITD licences need an institutional signature, so start that today. Until approval, use `synthetic` to develop and validate the pipeline. Synthetic results validate the *instrument*. They are **not** the headline curve.

## Request templates

Each requires an institutional email address and a supervisor's name. Fill the brackets and send it yourself.

<details><summary>HK PolyU (Prof. Ajay Kumar)</summary>

```
Subject: Request for access — PolyU Contactless 2D to Contact-based 2D Fingerprint Database v1.0

Dear Prof. Kumar,

I am [name], a [student/researcher] at [institution], supervised by [supervisor].
We request access to the PolyU Contactless 2D to Contact-based 2D Fingerprint
Images Database v1.0 for non-commercial academic research.

Purpose: a biometric-privacy *awareness* study (OpenCV AI Competition 2026) that
measures how much fingerprint ridge information remains recoverable from contactless
photos as resolution, blur, compression and lighting degrade. The output is a
recoverability curve and a privacy-risk score for users' own photos. We do not
build spoofing or authentication-bypass tools.

We will cite Lin & Kumar, IEEE TIP 2018, keep the data on access-controlled
storage, and not redistribute it. I can sign any licence agreement you require.

Kind regards,
[name, title, institution, institutional email]
```
</details>

<details><summary>IIITD ISPFD v1 / v2 (IAB lab)</summary>

Download the licence agreement from iab-rubric.org, have it **signed by the head of institution or registrar**, and email it to `databases@iab-rubric.org` with the exact subject line the site specifies (one email per database). Body: same purpose paragraph as above.
</details>

<details><summary>RidgeBase (University at Buffalo CUBS)</summary>

Complete the licence/request form on the dataset page. Purpose: same paragraph as above. Cite Jawade et al., *RidgeBase*, IJCB 2022 / arXiv:2307.05563.
</details>

## Expected layout after approval

The loader is **manifest-driven**, so it works whatever the archive's internal layout is:

```
datasets/
  hkpolyu/
    raw/                 # unpack the archive here, untouched
    manifest.csv         # generated: contactless,contact,subject_id,finger_id,split
  ispfd1/ ... ispfd2/ ... ridgebase/   (same pattern)
```

Generate a manifest with `scripts/make_manifest.py`. It matches file names with a regex per dataset. **The default regexes are best guesses: check them against the real archive before the first run** (`--dry-run` prints the first 10 pairs):

```bash
python scripts/make_manifest.py hkpolyu --root datasets/hkpolyu/raw --dry-run
python scripts/make_manifest.py hkpolyu --root datasets/hkpolyu/raw \
    --cl-regex "<regex with (?P<subject>) and (?P<finger>)>" --cb-regex "..."
```

`manifest.csv` columns: `contactless` and `contact` (paths relative to the dataset dir), `subject_id`, `finger_id`, `split` (`train` or `test`; DCM training uses `train` only), plus any extra columns, which are kept in `meta`.

## Python interface

```python
from datasets.loader import load_dataset
for pair in load_dataset("hkpolyu", split="test"):     # or "synthetic"
    pair.contactless_img, pair.contact_gt, pair.subject_id, pair.finger_id, pair.meta
```

Images load lazily, as grayscale `uint8` or BGR, and stay in memory only for the duration of an experiment.
