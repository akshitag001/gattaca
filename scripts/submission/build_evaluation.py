"""Regenerate the failure-case evidence shipped in the submission's evaluation/ folder.

    python scripts/submission/build_evaluation.py <out_dir>

Every number is recomputed from the current code; nothing is copied from notes.

1. Clarity false positives. For each test input, the clarity metric is computed
   three ways, adding one factor at a time:
     fit only          local correlation between the image and its Gabor reconstruction
     fit x coherence   + structure-tensor coherence gate
     final             + periodicity term (the shipped metric)
   Inputs: clean ridges, band-limited noise, a specular highlight bar, a step edge.
2. Frame selection. Spectral anisotropy (the first frame-ranking signal) vs ridge-band
   energy (the shipped one), on sharp and motion-blurred frames of the same clip.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gattaca import synth  # noqa: E402
from gattaca.debug import to_u8  # noqa: E402
from gattaca.pipeline.enhance import EnhanceParams, enhance, gabor_bank_filter  # noqa: E402
from gattaca.pipeline.quality import _lcorr, periodicity_map  # noqa: E402
from gattaca.pipeline.ridgefreq import dominant_period  # noqa: E402
from gattaca.video.select import ridge_band_fraction  # noqa: E402

N = 300


def inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    ridges = synth.contact_impression(0)[50:350, 50:350]
    noise = np.clip(128 + 160 * cv2.GaussianBlur(rng.normal(0, 1, (N, N)), (0, 0), 3), 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    skin = 120 + 25 * (xx / N)                                   # smooth skin shading, no ridges
    bar = 95 * np.exp(-((xx - 150) ** 2) / (2 * 7.0 ** 2))         # specular highlight stripe
    spec = np.clip(skin + bar + rng.normal(0, 1.5, (N, N)), 0, 255).astype(np.uint8)
    step = np.full((N, N), 80, np.uint8)
    step[:, 150:] = 180
    step = cv2.GaussianBlur(step, (0, 0), 1.5)
    return {"clean ridges": ridges, "band-limited noise": noise, "specular highlight": spec, "step edge": step}


def clarity_variants(img: np.ndarray) -> dict[str, np.ndarray]:
    full = np.full(img.shape, 255, np.uint8)
    e = enhance(img, full)
    period = 9.0
    # Gabor response over the whole frame (the shipped path zeroes it outside the ridge mask,
    # which already hides most of the noise case; here we show the metric itself)
    gab = gabor_bank_filter(e.normalized, e.orientation, e.period, full, EnhanceParams())
    fit = np.clip(_lcorr(e.normalized, gab, period), 0, 1)
    gate = np.clip((e.coherence - 0.2) / 0.4, 0, 1)
    e_full = type(e)(enhanced=gab, orientation=e.orientation, coherence=e.coherence,
                     period=e.period, mask=full, normalized=e.normalized)
    periodic = np.clip(periodicity_map(e_full, period) / 0.6, 0, 1)
    return {"fit only": fit, "fit x coherence": fit * gate, "final (shipped)": fit * gate * periodic}


def tile(img: np.ndarray, title: str, value: float | None, w=300, h=300, ridges: bool = False) -> np.ndarray:
    g = to_u8(img)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if g.ndim == 2 else g
    if value is not None:
        bgr = cv2.applyColorMap(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    out = np.full((h + 64, w, 3), 250, np.uint8)
    out[64:] = bgr
    cv2.putText(out, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    if value is not None:
        correct = (value > 0.1) == ridges          # high is right only for real ridges
        col = (40, 130, 40) if correct else (40, 40, 200)
        cv2.putText(out, f"mean clarity {value:.2f}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "evaluation")
    out.mkdir(parents=True, exist_ok=True)

    rows, grid = [], []
    for name, img in inputs().items():
        v = clarity_variants(img)
        means = {k: float(m.mean()) for k, m in v.items()}
        rows.append({"input": name, **{k: round(x, 3) for k, x in means.items()}})
        grid.append(np.hstack([tile(img, name, None)] + [tile(m, k, means[k], ridges=(name == "clean ridges"))
                                                          for k, m in v.items()]))
        print(f"{name:20s} " + "  ".join(f"{k}={x:.3f}" for k, x in means.items()))
    fig = np.vstack(grid)
    banner = np.full((70, fig.shape[1], 3), 250, np.uint8)
    cv2.putText(banner, "Ridge clarity: each factor added after a measured false positive "
                "(green = correct for that input, red = false positive)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(banner, "Only 'clean ridges' should score high. Expected: noise caught by the coherence gate, "
                "highlight/edge caught by periodicity.", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (90, 90, 90), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "failure_cases_clarity.png"), np.vstack([banner, fig]))
    with (out / "failure_cases_clarity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # frame selection: old signal (spectral anisotropy SNR) vs shipped signal (ridge-band energy)
    frames = synth.synthetic_clip(2, n_frames=12, downscale=2.0, blur_frames=0.5, seed=4)
    h, w = frames[0].shape[:2]
    box = (int(w * 0.2), int(h * 0.05), int(w * 0.6), int(h * 0.5))
    fr = []
    for i, f in enumerate(frames):
        x, y, bw, bh = box
        g = cv2.cvtColor(f[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var() / (g.astype(np.float32).var() + 1))
        period, snr, _ = dominant_period(g)
        fr.append({"frame": i, "sharpness": round(sharp, 3), "anisotropy_snr_old": round(float(snr), 1),
                   "ridge_band_new": round(ridge_band_fraction(g, period), 3)})
    fr.sort(key=lambda r: r["sharpness"])
    blur = fr[:2]
    sharp = fr[-3:]
    with (out / "failure_cases_frame_selection.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(fr[0]))
        wr.writeheader()
        wr.writerows(fr)
    print("blurriest frames:", blur)
    print("sharpest frames :", sharp)
    print("old picks top-3 by anisotropy:", sorted(fr, key=lambda r: -r["anisotropy_snr_old"])[:3])
    print("new picks top-3 by ridge band:", sorted(fr, key=lambda r: -r["ridge_band_new"])[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
