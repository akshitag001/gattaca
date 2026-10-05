"""Regenerate every study figure from the raw CSVs (and write the plotted numbers as
CSV tables next to each PNG, so each figure has a table view).

    python -m study.plots --csv study/results/synthetic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from study.calibrate import eer, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# reference palette (light surface); slots 1-3 validate all-pairs for CVD
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
VARIANT_LABEL = {"rigid": "rigid", "dcm": "+ DCM", "dcm_rtps": "+ DCM + RTPS", "rtps": "+ RTPS"}
AXES = {
    "pprp": ("Pixels per ridge period", False),
    "gauss": ("Defocus blur σ (ridge periods)", False),
    "motion": ("Motion blur length (ridge periods)", False),
    "jpeg": ("JPEG quality", True),
    "exposure": ("Under-exposure (stops)", True),
    "noise": ("Sensor noise σ (grey levels)", False),
    "pprp_jpeg": ("Pixels per ridge period, then JPEG q75", True),
}


def _style(ax, xlabel, ylabel="Rank-1 identification"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)


def _rate_table(df, axis, variant):
    g = df[df.axis == axis].groupby("level")
    rows = []
    for lv, x in g:
        k, n = int((x[f"rank_{variant}"] == 1).sum()), len(x)
        lo, hi = wilson(k, n)
        rows.append({"level": lv, "n": n, "rank1": k / n, "lo": lo, "hi": hi,
                     "minutiae": x.get("minutiae", pd.Series(dtype=float)).mean(),
                     "clarity": x.get("clarity", pd.Series(dtype=float)).mean()})
    return pd.DataFrame(rows).sort_values("level")


def _line(ax, t, color, label=None, band=True):
    if band:
        ax.fill_between(t.level, t.lo, t.hi, color=color, alpha=0.12, lw=0)
    ax.plot(t.level, t.rank1, color=color, lw=2, solid_capstyle="round", label=label)
    ax.scatter(t.level, t.rank1, s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)


def _caption(fig, df):
    src = str(df.dataset.dropna().iloc[0]) if "dataset" in df and df.dataset.notna().any() else "?"
    n = df.key.nunique()
    note = " (synthetic prints validate the instrument; they are not the headline result)" if src == "synthetic" else ""
    fig.text(0.01, 0.005, f"Data: {src}, {n} fingers, 95% Wilson CIs{note}", color=INK2, fontsize=8)


def headline(df, imp, variant, out: Path):
    t = _rate_table(df, "pprp", variant)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
    _line(a1, t, SERIES[0])
    a1.set_xscale("log", base=2)
    a1.set_xticks(t.level)
    a1.set_xticklabels([f"{v:g}" for v in t.level])
    a1.invert_xaxis()
    a1.set_ylim(-0.03, 1.05)
    _style(a1, "Pixels per ridge period (log scale; resolution falls →)")
    for thr in (0.5, 0.8):
        a1.axhline(thr, color=INK2, lw=0.8, ls=(0, (3, 3)))
        a1.text(t.level.max(), thr + 0.015, f"{int(thr * 100)}%", color=INK2, fontsize=8)
    a1.set_title("Fingerprint recoverability vs resolution at the fingertip", color=INK, fontsize=11, loc="left")
    e = []
    for lv in t.level:
        sel = imp[(imp.axis == "pprp") & (imp.level == lv) & (imp.variant == variant)] if imp is not None else None
        gen = df[(df.axis == "pprp") & (df.level == lv)][f"genuine_{variant}"].to_numpy()
        e.append(eer(gen, sel.score.to_numpy()) if sel is not None else np.nan)
    t["eer"] = e
    a2.plot(t.level, t.eer, color=SERIES[0], lw=2)
    a2.scatter(t.level, t.eer, s=36, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    a2.set_xscale("log", base=2)
    a2.set_xticks(t.level)
    a2.set_xticklabels([f"{v:g}" for v in t.level])
    a2.invert_xaxis()
    a2.set_ylim(-0.02, 0.55)
    _style(a2, "Pixels per ridge period", "Equal error rate (verification)")
    a2.set_title("Verification EER", color=INK, fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _caption(fig, df)
    fig.savefig(out / "recoverability_vs_pprp.png", dpi=160, facecolor=SURFACE)
    t.to_csv(out / "recoverability_vs_pprp.csv", index=False)
    plt.close(fig)


def per_axis(df, variant, out: Path):
    axes = [a for a in AXES if a != "pprp" and (df.axis == a).any()]
    n = len(axes)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, grid = plt.subplots(rows, cols, figsize=(11, 3.3 * rows), facecolor=SURFACE, squeeze=False)
    clean = df[df.axis == "none"]
    base = (clean[f"rank_{variant}"] == 1).mean() if len(clean) else np.nan
    tables = []
    for i, axis in enumerate(axes):
        ax = grid[i // cols][i % cols]
        t = _rate_table(df, axis, variant)
        _line(ax, t, SERIES[0])
        ax.axhline(base, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.set_ylim(-0.03, 1.05)
        if AXES[axis][1]:
            ax.invert_xaxis()
        _style(ax, AXES[axis][0], "Rank-1" if i % cols == 0 else "")
        tables.append(t.assign(axis=axis))
    for j in range(n, rows * cols):
        grid[j // cols][j % cols].axis("off")
    fig.suptitle("Recoverability under each degradation (dashed: undegraded)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _caption(fig, df)
    fig.savefig(out / "recoverability_by_axis.png", dpi=160, facecolor=SURFACE)
    pd.concat(tables).to_csv(out / "recoverability_by_axis.csv", index=False)
    plt.close(fig)


def variants_plot(df, out: Path):
    order = ["rigid", "dcm", "rtps", "dcm_rtps"]
    variants = sorted((c[5:] for c in df.columns if c.startswith("rank_")), key=order.index)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    rows = []
    for k, v in enumerate(variants[:3]):
        t = _rate_table(df, "pprp", v)
        _line(ax, t, SERIES[k], VARIANT_LABEL.get(v, v), band=False)   # bands overlap: see CSV for CIs
        rows.append(t.assign(variant=v))
    ax.set_xscale("log", base=2)
    ax.set_xticks(t.level)
    ax.set_xticklabels([f"{x:g}" for x in t.level])
    ax.invert_xaxis()
    ax.set_ylim(-0.03, 1.05)
    _style(ax, "Pixels per ridge period")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title("Matcher variants: effect of deformation correction", color=INK, fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _caption(fig, df)
    fig.savefig(out / "matcher_variants.png", dpi=160, facecolor=SURFACE)
    pd.concat(rows).to_csv(out / "matcher_variants.csv", index=False)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--variant", default=None)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv)
    ip = a.csv.with_name(a.csv.stem + "_impostors.csv")
    imp = pd.read_csv(ip) if ip.exists() else None
    variants = [c[5:] for c in df.columns if c.startswith("rank_")]
    variant = a.variant or ("dcm_rtps" if "dcm_rtps" in variants else variants[-1])
    out = ROOT / "study" / "plots" / a.csv.stem
    out.mkdir(parents=True, exist_ok=True)
    headline(df, imp, variant, out)
    per_axis(df, variant, out)
    variants_plot(df, out)
    print(f"figures + tables in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
