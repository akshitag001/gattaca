"""Fit the recoverability model from study CSVs and write a calibration JSON.

Target: rank-1 identification succeeded (genuine finger ranked first in the gallery)
with the chosen matcher variant. Features: inference-time only (no ground truth).
Model: L2-regularised logistic regression on standardised features. Uncertainty:
a **subject-level cluster bootstrap** (resample probes, refit) whose replicates
gattaca.scoring uses to report band confidence.

Also writes the headline curve: rank-1 rate vs nominal PPRP with Wilson 95% CIs,
plus EER per PPRP level, and the PPRP where rank-1 crosses 50% / 80%.

    python -m study.calibrate --csv study/results/synthetic.csv --install
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["log2_pprp", "clarity", "coherence", "log_minutiae", "spectral_snr"]
DEFAULT_OUT = ROOT / "gattaca" / "calibration" / "default.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    """Returns [coef..., intercept] for standardised X."""
    n, d = X.shape

    def loss(w):
        z = X @ w[:d] + w[d]
        nll = np.sum(np.logaddexp(0, z) - y * z)
        return nll + 0.5 * l2 * np.sum(w[:d] ** 2)

    def grad(w):
        z = X @ w[:d] + w[d]
        r = 1 / (1 + np.exp(-z)) - y
        return np.r_[X.T @ r + l2 * w[:d], r.sum()]

    return minimize(loss, np.zeros(d + 1), jac=grad, method="L-BFGS-B").x


def eer(gen: np.ndarray, imp: np.ndarray) -> float:
    if len(gen) == 0 or len(imp) == 0:
        return float("nan")
    th = np.unique(np.r_[gen, imp])
    frr = np.array([(gen < t).mean() for t in th])
    far = np.array([(imp >= t).mean() for t in th])
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def crossing(levels: np.ndarray, rate: np.ndarray, target: float) -> float | None:
    """Lowest PPRP at which the (monotonised) rank-1 curve reaches `target`."""
    order = np.argsort(levels)
    lv, r = levels[order], np.maximum.accumulate(rate[order])
    if r[-1] < target:
        return None
    if r[0] >= target:
        return float(lv[0])
    j = int(np.argmax(r >= target))
    return float(np.interp(target, [r[j - 1], r[j]], [lv[j - 1], lv[j]]))


def headline_curve(df: pd.DataFrame, imp: pd.DataFrame | None, variant: str) -> list[dict]:
    rows = []
    pp = df[df.axis == "pprp"]
    clean = df[df.axis == "none"]
    for lv, g in sorted(pp.groupby("level"), key=lambda t: -t[0]) + [("native", clean)]:
        k, n = int((g[f"rank_{variant}"] == 1).sum()), len(g)
        lo, hi = wilson(k, n)
        e = float("nan")
        if imp is not None:
            ax, level = ("none", 0.0) if lv == "native" else ("pprp", lv)
            sel = imp[(imp.axis == ax) & (imp.level == level) & (imp.variant == variant)]
            e = eer(g[f"genuine_{variant}"].to_numpy(), sel.score.to_numpy())
        rows.append({"pprp": float(g.native_pprp.median()) if lv == "native" else float(lv),
                     "label": "native" if lv == "native" else f"{lv:g}", "n": n, "rank1": k / n if n else None,
                     "lo": lo, "hi": hi, "eer": e, "minutiae": float(g.get("minutiae", pd.Series([0])).mean()),
                     "genuine_score": float(g[f"genuine_{variant}"].median())})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--variant", default=None, help="matcher variant (default: best available)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--install", action="store_true", help=f"also write {DEFAULT_OUT}")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv)
    imp_path = a.csv.with_name(a.csv.stem + "_impostors.csv")
    imp = pd.read_csv(imp_path) if imp_path.exists() else None
    meta_path = a.csv.with_name(a.csv.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    variants = [c[5:] for c in df.columns if c.startswith("rank_")]
    variant = a.variant or ("dcm_rtps" if "dcm_rtps" in variants else variants[-1])
    d = df[df.detected.astype(bool)].dropna(subset=FEATURES)
    X = d[FEATURES].to_numpy(float)
    y = (d[f"rank_{variant}"] == 1).to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = fit_logistic(Z, y)
    rng = np.random.default_rng(a.seed)
    probes = d.probe.unique()
    by_probe = {p: np.flatnonzero(d.probe.to_numpy() == p) for p in probes}
    boot = []
    for _ in range(a.boot):
        idx = np.concatenate([by_probe[p] for p in rng.choice(probes, len(probes))])
        if 0 < y[idx].mean() < 1:
            boot.append(fit_logistic(Z[idx], y[idx]))
    p_hat = 1 / (1 + np.exp(-(Z @ w[:-1] + w[-1])))
    acc = float(((p_hat >= 0.5) == y).mean())
    brier = float(np.mean((p_hat - y) ** 2))
    # held-out estimate: 5-fold cross-validation grouped by FINGER (no finger in train and test)
    keys = d.key.to_numpy()
    folds = np.array_split(rng.permutation(np.unique(keys)), 5)
    p_cv = np.zeros_like(y)
    for f in folds:
        te = np.isin(keys, f)
        wf = fit_logistic(Z[~te], y[~te])
        p_cv[te] = 1 / (1 + np.exp(-(Z[te] @ wf[:-1] + wf[-1])))
    cv_acc = float(((p_cv >= 0.5) == y).mean())
    cv_brier = float(np.mean((p_cv - y) ** 2))
    band_cv = np.where(p_cv >= 0.7, "HIGH", np.where(p_cv < 0.3, "LOW", "MEDIUM"))
    band_rates = {b: {"n": int((band_cv == b).sum()),
                      "observed_rank1": float(y[band_cv == b].mean()) if (band_cv == b).any() else None}
                  for b in ("HIGH", "MEDIUM", "LOW")}
    curve = headline_curve(df, imp, variant)
    lv = np.array([c["pprp"] for c in curve if c["label"] != "native"])
    rr = np.array([c["rank1"] for c in curve if c["label"] != "native"])
    thresholds = {"pprp_50": crossing(lv, rr, 0.5), "pprp_80": crossing(lv, rr, 0.8),
                  "pprp_95": crossing(lv, rr, 0.95)}
    out = {
        "source": f"{meta.get('dataset', a.csv.stem)}-study" + (" (synthetic: validates the instrument, not the "
                                                                 "headline curve)" if "synthetic" in a.csv.stem else ""),
        "dataset": meta.get("dataset", a.csv.stem), "variant": variant, "target": "rank-1 identification",
        "n_rows": int(len(d)), "n_probes": int(len(probes)), "gallery": meta.get("gallery"),
        "features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(), "coef": w[:-1].tolist(),
        "intercept": float(w[-1]), "boot": [b.tolist() for b in boot],
        "bands": {"high": 0.7, "low": 0.3}, "fit": {"accuracy": acc, "brier": brier, "base_rate": float(y.mean()),
                "cv5_by_finger": {"accuracy": cv_acc, "brier": cv_brier, "bands": band_rates}},
        "thresholds": thresholds, "curve": curve,
    }
    calib_dir = ROOT / "study" / "results"
    dst = calib_dir / f"calibration_{a.csv.stem}.json"
    dst.write_text(json.dumps(out, indent=1))
    if a.install:
        DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_OUT.write_text(json.dumps(out, indent=1))
    print(f"variant={variant}  rows={len(d)}  base rate={y.mean():.2f}  train acc={acc:.3f}  brier={brier:.3f}")
    print(f"held-out (5-fold by finger): acc={cv_acc:.3f}  brier={cv_brier:.3f}  bands={band_rates}")
    print("coef (standardised):", dict(zip(FEATURES, np.round(w[:-1], 2))), "intercept", round(w[-1], 2))
    print("thresholds:", thresholds)
    for c in curve:
        print(f"  PPRP {c['label']:>6s}  n={c['n']:3d}  rank-1 {c['rank1']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}]"
              f"  EER {c['eer']:.3f}  minutiae {c['minutiae']:.1f}")
    print(f"wrote {dst}" + (f" and {DEFAULT_OUT}" if a.install else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
