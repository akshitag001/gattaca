"""Build datasets/<key>/manifest.csv by pairing contactless and contact files by regex.

The default patterns are BEST GUESSES at each archive's naming scheme. Always run
with --dry-run first and pass --cl-regex/--cb-regex if the pairs look wrong. Each
regex must define named groups `subject` and `finger` (and may define `sample`).

    python scripts/make_manifest.py hkpolyu --root datasets/hkpolyu/raw --dry-run
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

IMG = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# (contactless regex, contact regex) on the path relative to --root, using forward slashes
DEFAULTS = {
    "hkpolyu": (r"(?i)contactless.*?/p?(?P<subject>\d+)[_/-](?P<sample>\d+)\.\w+$",
                r"(?i)contact[-_ ]?based.*?/(?P<subject>\d+)[_/-](?P<sample>\d+)\.\w+$"),
    "ispfd1": (r"(?i)(?P<subject>\d+)[_-](?P<finger>[a-z0-9]+)[_-].*?(?:phone|photo|selfie|camera).*\.\w+$",
               r"(?i)(?P<subject>\d+)[_-](?P<finger>[a-z0-9]+)[_-].*?(?:live|scan|sensor).*\.\w+$"),
    "ispfd2": (r"(?i)(?P<subject>\d+)[_-](?P<finger>[a-z0-9]+)[_-].*?(?:selfie|phone).*\.\w+$",
               r"(?i)(?P<subject>\d+)[_-](?P<finger>[a-z0-9]+)[_-].*?(?:live|scan).*\.\w+$"),
    "ridgebase": (r"(?i)contactless.*?(?P<subject>\d+).*?(?P<finger>index|middle|ring|little|\d)\D.*\.\w+$",
                  r"(?i)contact[-_]?based.*?(?P<subject>\d+).*?(?P<finger>index|middle|ring|little|\d)\D.*\.\w+$"),
}


def scan(root: Path, rx: re.Pattern) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG:
            continue
        rel = p.relative_to(root).as_posix()
        m = rx.search(rel)
        if m:
            g = m.groupdict()
            out[(g["subject"].lstrip("0") or "0", (g.get("finger") or "0").lower())].append(rel)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DEFAULTS))
    ap.add_argument("--root", required=True, type=Path, help="unpacked archive directory")
    ap.add_argument("--cl-regex")
    ap.add_argument("--cb-regex")
    ap.add_argument("--train-frac", type=float, default=0.3, help="fraction of fingers for DCM training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cl_rx, cb_rx = DEFAULTS[a.dataset]
    cl = scan(a.root, re.compile(a.cl_regex or cl_rx))
    cb = scan(a.root, re.compile(a.cb_regex or cb_rx))
    keys = sorted(set(cl) & set(cb))
    print(f"contactless fingers: {len(cl)}  contact fingers: {len(cb)}  paired: {len(keys)}")
    if not keys:
        print("No pairs: inspect the archive layout and pass --cl-regex/--cb-regex.", file=sys.stderr)
        return 1
    rng = random.Random(a.seed)
    train = set(rng.sample(keys, int(len(keys) * a.train_frac)))
    out_dir = a.root.parent
    rows = []
    for k in keys:
        gt = sorted(cb[k])[0]   # first contact impression is the gallery entry
        for p in sorted(cl[k]):
            rows.append({"contactless": (a.root / p).relative_to(out_dir).as_posix(),
                         "contact": (a.root / gt).relative_to(out_dir).as_posix(),
                         "subject_id": k[0], "finger_id": k[1], "split": "train" if k in train else "test"})
    for r in rows[:10]:
        print(r)
    if a.dry_run:
        print(f"(dry run) would write {len(rows)} rows to {out_dir / 'manifest.csv'}")
        return 0
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
