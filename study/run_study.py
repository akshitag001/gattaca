"""The measurement study: recoverability vs PPRP, blur, motion, compression, exposure, noise.

For each paired probe and each degradation condition:
  degrade -> full pipeline (detect, ROI, rectify, PPRP-normalise, enhance, extract)
  -> match against EVERY gallery ground truth (genuine + impostors) under three
     matcher variants: rigid | +DCM | +DCM+RTPS
  -> record minutiae count/quality, clarity, PPRP, genuine score, impostor scores and
     the rank of the genuine finger.

Raw outputs are CSVs (one row per probe x condition, plus impostor scores), so plots
and calibration can be regenerated without re-running.

    python -m study.run_study --dataset synthetic --subjects 40 --workers 8
    python -m study.run_study --dataset hkpolyu --axes pprp,gauss,jpeg --workers 32
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.loader import load_dataset  # noqa: E402
from gattaca.analyze import analyze_contact, analyze_photo  # noqa: E402
from gattaca.pipeline.deform import DeformationCorrectionModel  # noqa: E402
from gattaca.pipeline.match import match  # noqa: E402
from study import degrade  # noqa: E402

RESULTS = ROOT / "study" / "results"
FEATURES = ["log2_pprp", "clarity", "coherence", "log_minutiae", "spectral_snr"]

_G: dict = {}  # per-worker gallery/DCM


def _init(gallery, dcm, variants, backend):
    cv2.setNumThreads(1)  # parallelism comes from processes
    _G.update(gallery=gallery, dcm=dcm, variants=variants, backend=backend)


def _variants(dcm_ok: bool) -> list[str]:
    return ["rigid", "dcm", "dcm_rtps"] if dcm_ok else ["rigid", "rtps"]


def _match_all(minutiae, variant):
    dcm = _G["dcm"] if variant.startswith("dcm") else None
    rt = variant.endswith("rtps")
    return [match(minutiae, g["minutiae"], q_mask=g["mask"], dcm=dcm, rtps=rt) for g in _G["gallery"]]


def run_probe(task):
    probe_id, img, key, meta, conds = task
    keys = [g["key"] for g in _G["gallery"]]
    gi = keys.index(key)
    clean = analyze_photo(img, backend=_G["backend"], max_fingers=1)
    native = clean[0].analysis.pprp if clean else float("nan")
    rows, imps = [], []
    for cond in conds:
        if not np.isfinite(native):
            break
        seed = zlib.crc32(f"{probe_id}|{cond.label}".encode())
        deg = degrade.apply(img, cond, native, seed=seed)
        if deg is None:
            continue
        t0 = time.perf_counter()
        fingers = analyze_photo(deg, backend=_G["backend"], max_fingers=1)
        t_an = time.perf_counter() - t0
        row = {"probe": probe_id, "key": key, "dataset": meta.get("dataset"), "axis": cond.axis,
               "level": cond.level, "native_pprp": native,
               "nominal_pprp": cond.level if cond.axis.startswith("pprp") else native,
               "detected": bool(fingers), "t_analyze": t_an}
        if not fingers:
            for v in _G["variants"]:
                row.update({f"rank_{v}": len(keys), f"genuine_{v}": 0.0, f"nmatch_{v}": 0, f"maximp_{v}": 0.0})
            rows.append(row)
            continue
        an = fingers[0].analysis
        q = an.quality
        row.update({"pprp_est": an.pprp, "pprp_spectral": an.pprp_spectral, "pprp_geometric": an.pprp_geometric,
                    "minutiae": q.minutiae_count, "minutiae_quality": q.minutiae_quality, "clarity": q.clarity,
                    "coherence": q.coherence, "spectral_snr_raw": q.spectral_snr, "area_px": q.area_px,
                    "nfiq_like": q.nfiq_like, **q.features()})
        t0 = time.perf_counter()
        for v in _G["variants"]:
            res = _match_all(an.minutiae, v)
            sc = np.array([r.score for r in res])
            gen = sc[gi]
            imp = np.delete(sc, gi)
            row.update({f"rank_{v}": int(1 + (imp >= gen).sum()) if gen > 0 else len(keys),
                        f"genuine_{v}": float(gen), f"nmatch_{v}": res[gi].n_matched,
                        f"maximp_{v}": float(imp.max()) if len(imp) else 0.0})
            imps += [(probe_id, cond.axis, cond.level, v, float(s)) for s in imp]
        row["t_match"] = time.perf_counter() - t0
        rows.append(row)
    return rows, imps


def build_gallery(pairs) -> list[dict]:
    seen, gallery = set(), []
    for p in pairs:
        if p.key in seen:
            continue
        seen.add(p.key)
        an = analyze_contact(p.contact_gt)
        gallery.append({"key": p.key, "minutiae": an.minutiae, "mask": an.enh.mask})
    return gallery


def train_dcm(train_pairs, backend: str) -> DeformationCorrectionModel | None:
    dcm = DeformationCorrectionModel()
    n = 0
    for p in train_pairs:
        fingers = analyze_photo(p.contactless_img, backend=backend, max_fingers=1)
        if not fingers:
            continue
        gt = analyze_contact(p.contact_gt)
        res = match(fingers[0].analysis.minutiae, gt.minutiae)
        if res.n_matched < 8:
            continue
        qxy = np.c_[gt.minutiae["x"], gt.minutiae["y"]]
        dcm.add_pair(res.aligned_p[res.pairs[:, 0]], qxy[res.pairs[:, 1]], gt.enh.mask)
        n += 1
    print(f"DCM trained on {n} pairs")
    return dcm.finalize() if n >= 5 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--subjects", type=int, default=30, help="synthetic only")
    ap.add_argument("--axes", default=",".join(degrade.LADDERS))
    ap.add_argument("--limit", type=int, default=0, help="max probes (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--backend", default="classical", help="detector backend for probes")
    ap.add_argument("--no-dcm", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    name = a.name or a.dataset
    kw = {"subjects": a.subjects} if a.dataset == "synthetic" else {}
    test = list(load_dataset(a.dataset, split="test", **kw))
    if a.limit:
        test = test[: a.limit]
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dcm = None
    if not a.no_dcm:
        train_kw = {"subjects": max(10, a.subjects // 2)} if a.dataset == "synthetic" else {}
        try:
            dcm = train_dcm(list(load_dataset(a.dataset, split="train", **train_kw)), a.backend)
        except Exception as e:  # no train split available
            print(f"DCM training skipped: {e}")
        if dcm is not None:
            dcm.save(RESULTS / f"dcm_{name}.json")
    gallery = build_gallery(test)
    variants = _variants(dcm is not None)
    conds = degrade.conditions([x for x in a.axes.split(",") if x])
    print(f"{len(test)} probes x {len(conds)} conditions, gallery {len(gallery)}, variants {variants}, "
          f"{a.workers} workers")
    tasks = [(f"{name}:{i}", p.contactless_img, p.key, p.meta, conds) for i, p in enumerate(test)]
    rows, imps = [], []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(gallery, dcm, variants, a.backend)) as ex:
        for i, (r, im) in enumerate(ex.map(run_probe, tasks)):
            rows += r
            imps += im
            print(f"  probe {i + 1}/{len(tasks)} done ({time.time() - t0:.0f}s)", flush=True)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("probe", "key", "axis", "level"), k))
    out = RESULTS / f"{name}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with (RESULTS / f"{name}_impostors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["probe", "axis", "level", "variant", "score"])
        w.writerows(imps)
    meta = {"dataset": a.dataset, "probes": len(test), "gallery": len(gallery), "conditions": len(conds),
            "variants": variants, "wall_s": round(time.time() - t0, 1), "workers": a.workers,
            "opencv": cv2.__version__, "machine": platform.machine(), "python": platform.python_version()}
    (RESULTS / f"{name}_meta.json").write_text(__import__("json").dumps(meta, indent=2))
    print(f"wrote {out} ({len(rows)} rows) in {meta['wall_s']} s")
    _print_summary(rows, variants)
    return 0


def _print_summary(rows, variants):
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["axis"], r["level"])].append(r)
    print(f"{'condition':18s} n  " + "  ".join(f"r1_{v:9s}" for v in variants) + "  minutiae")
    for (ax, lv), rs in sorted(agg.items(), key=lambda kv: (kv[0][0] != "none", kv[0][0], -kv[0][1])):
        r1 = [np.mean([x[f"rank_{v}"] == 1 for x in rs]) for v in variants]
        mn = np.mean([x.get("minutiae", 0) for x in rs])
        print(f"{ax + '=' + format(lv, 'g'):18s} {len(rs):<3d}" + "".join(f"  {v:11.2f}" for v in r1) + f"  {mn:8.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
