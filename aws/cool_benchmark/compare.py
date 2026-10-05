"""Compare two bench.py results: speed-up per OpenCV kernel, end-to-end, and cost.

    python aws/cool_benchmark/compare.py results/cool-c8g.json results/stock-c7i.json \
        --price-a 0.1595 --price-b 0.1785 --sw-a 0.02

Prices are $/hour. Defaults are us-east-1 on-demand list prices for c8g.xlarge /
c7i.xlarge at the time of writing. VERIFY on the AWS pricing page and pass the
COOL AMI software fee (--sw-a) from the Marketplace listing for your instance size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="candidate (COOL on Graviton)")
    ap.add_argument("b", type=Path, help="baseline (stock x86)")
    ap.add_argument("--price-a", type=float, default=0.1595)
    ap.add_argument("--price-b", type=float, default=0.1785)
    ap.add_argument("--sw-a", type=float, default=0.0, help="COOL AMI software $/h")
    ap.add_argument("--out", type=Path, default=None)
    x = ap.parse_args(argv)
    A, B = json.loads(x.a.read_text()), json.loads(x.b.read_text())
    lines = [f"# COOL benchmark: {A['label']} vs {B['label']}", "",
             "| | " + A["label"] + " | " + B["label"] + " |", "|---|---|---|"]
    for k in ("opencv", "cv2_path", "python_prefix", "machine", "cpu", "cores", "instance_type", "kleidicv"):
        lines.append(f"| {k} | `{A['provenance'].get(k)}` | `{B['provenance'].get(k)}` |")
    for mode in ("micro_1thread", "micro_default_threads"):
        lines += ["", f"## OpenCV kernels ({mode.replace('micro_', '').replace('_', ' ')}), median ms", "",
                  f"| kernel | {A['label']} | {B['label']} | speed-up |", "|---|---:|---:|---:|"]
        sp = []
        for k, va in A[mode].items():
            vb = B[mode].get(k)
            if not vb:
                continue
            s = vb["median_ms"] / va["median_ms"]
            sp.append(s)
            lines.append(f"| {k} | {va['median_ms']:.2f} | {vb['median_ms']:.2f} | {s:.2f}x |")
        if sp:
            geo = 1.0
            for s in sp:
                geo *= s
            lines.append(f"| **geometric mean** | | | **{geo ** (1 / len(sp)):.2f}x** |")
    ma, mb = A["macro"], B["macro"]
    ta, tb = ma["photo_throughput"]["images_per_s"], mb["photo_throughput"]["images_per_s"]
    ca = (x.price_a + x.sw_a) / 3600 / ta * 1000
    cb = x.price_b / 3600 / tb * 1000
    lines += ["", "## End to end (GATTACA photo assessment)", "",
              f"| metric | {A['label']} | {B['label']} | ratio |", "|---|---:|---:|---:|",
              f"| latency p50 (ms, 1 thread) | {ma['photo_latency_ms']['p50']:.0f} | {mb['photo_latency_ms']['p50']:.0f}"
              f" | {mb['photo_latency_ms']['p50'] / ma['photo_latency_ms']['p50']:.2f}x |",
              f"| latency p95 (ms) | {ma['photo_latency_ms']['p95']:.0f} | {mb['photo_latency_ms']['p95']:.0f} | |",
              f"| throughput (images/s, all cores) | {ta:.2f} | {tb:.2f} | {ta / tb:.2f}x |",
              f"| agent on 24-frame clip (s) | {ma['agent_clip']['seconds']:.2f} | {mb['agent_clip']['seconds']:.2f} | "
              f"{mb['agent_clip']['seconds'] / ma['agent_clip']['seconds']:.2f}x |",
              f"| **cost per 1,000 photos ($)** | **{ca:.4f}** | **{cb:.4f}** | **{cb / ca:.2f}x cheaper** |",
              "", f"Prices: A ${x.price_a}/h + ${x.sw_a}/h software, B ${x.price_b}/h (verify before publishing)."]
    md = "\n".join(lines)
    print(md)
    (x.out or x.a.parent / f"compare_{A['label']}_vs_{B['label']}.md").write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
