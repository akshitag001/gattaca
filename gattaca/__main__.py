"""Command line.

    python -m gattaca assess hand.jpg [--debug out/] [--evidence ev.jpg] [--backend auto|dnn|classical]
    python -m gattaca agent clip.mp4 [--trace trace.json]
    python -m gattaca protect hand.jpg protected.jpg
    python -m gattaca demo [--out demo/]          # synthetic photo + clip, no data needed

`--debug DIR` dumps every intermediate image (mask, detection, rectification, CLAHE,
orientation field, period map, Gabor output, binary, skeleton, minutiae, clarity) so
you can see what each OpenCV stage produced. Use it on research-dataset images or
your own hand only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from gattaca import DISCLAIMER


def _read(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read image {path}")
    return img


def _print_trace(trace: list[dict]) -> None:
    for e in trace:
        if e["kind"] == "decision":
            print(f"  [{e['rule']}] {e['action']:<15s} because {e['reason']}")
        else:
            a = e["observation"].get("assessment", {})
            if a.get("band"):
                print(f"      -> {a['band']} P={a['probability']} clarity={a.get('clarity')} pprp={a.get('pprp')}"
                      f"  (best so far {e['best_probability']}, delta {e['delta']:+})")


def cmd_assess(a) -> int:
    from gattaca.assess import assess_image
    from gattaca.debug import DebugSink
    from gattaca.analyze import analyze_photo
    img = _read(a.image)
    if a.debug:
        analyze_photo(img, backend=a.backend, debug=DebugSink(a.debug))
        print(f"debug images in {a.debug}")
    res = assess_image(img, backend=a.backend)
    s = res.summary()
    print(json.dumps({k: s[k] for k in ("band", "probability", "confidence", "limiting_factor", "explanation",
                                        "fingers_found", "calibration_source")}, indent=1))
    if a.evidence and res.evidence is not None:
        cv2.imwrite(a.evidence, res.evidence)
        print(f"evidence map -> {a.evidence}")
    return 0


def cmd_agent(a) -> int:
    from gattaca.agent import AgentConfig, run_agent
    from gattaca.video.decode import read_video
    p = Path(a.source)
    frames = [_read(str(p))] if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") else read_video(str(p)).frames
    res = run_agent(frames, AgentConfig())
    print(f"status={res.status}  best={res.best.band if res.best else None} "
          f"P={res.best.probability:.3f} from {res.best_source}" if res.best else f"status={res.status}: no finger")
    _print_trace(res.trace)
    if a.trace:
        Path(a.trace).write_text(json.dumps(res.trace, indent=1, default=float))
    return 0


def cmd_protect(a) -> int:
    from gattaca.assess import assess_image
    from gattaca.protect import protect_photo
    img = _read(a.image)
    before = assess_image(img, render=False)
    out, _ = protect_photo(img, [f.detection for f in before.fingers] or None)
    after = assess_image(out, render=False)
    cv2.imwrite(a.output, out)
    print(f"before {before.summary()['band']} P={before.summary()['probability']}  ->  "
          f"after {after.summary()['band']} P={after.summary()['probability']}   saved {a.output}")
    return 0


def cmd_minutiae(a) -> int:
    from gattaca.forensic import ISO_19794_2_NOTE, minutiae_report, write_report
    img = _read(a.image)
    rep = minutiae_report(img, photo=a.photo, title=Path(a.image).name)
    info = write_report(rep, a.out, stem=Path(a.image).stem)
    print(f"source: {rep.source}   minutiae: {info['minutiae']}")
    print(f"{'idx':>3} {'x':>7} {'y':>7} {'angle':>6}  {'type':<13} {'qual'}")
    for r in rep.table[:a.limit] if a.limit else rep.table:
        print(f"{r['index']:>3} {r['x_px']:>7} {r['y_px']:>7} {r['angle_deg']:>6}  "
              f"{r['type']:<13} {r['quality_0_100']}")
    if a.notes:
        print("\n" + ISO_19794_2_NOTE)
    print(f"\nchart -> {info['chart']}\ncsv   -> {info['csv']}\nnotes -> {info['notes']}")
    return 0


def cmd_consolidate(a) -> int:
    import cv2
    from gattaca.consolidate import consolidate_video
    from gattaca.video.decode import read_video
    p = Path(a.video)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        sys.exit("consolidate needs a video; use `assess` for a single image")
    frames = read_video(str(p), max_frames=a.max_frames).frames
    if not frames:
        sys.exit("could not decode any frames")
    rep = consolidate_video(frames, k_fuse=a.k)
    s = rep.summary()
    print(f"frames: {rep.n_frames}   fingers tracked: {s['fingers']}")
    for f in rep.fingers:
        print(f"  {f.label:<10} single {f.single.band:<6} P={f.single.probability:.3f}  ->  "
              f"fused {f.fused.band:<6} P={f.fused.probability:.3f}   "
              f"(delta {f.delta_p:+.3f}, +{f.minutiae_gain} minutiae, {f.frames_used} frames)")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    chart = str(Path(a.out) / f"{p.stem}_consolidation.png")
    cv2.imwrite(chart, rep.chart)
    print(f"chart -> {chart}")
    return 0


def cmd_demo(a) -> int:
    from gattaca import synth
    from gattaca.agent import AgentConfig, run_agent
    from gattaca.assess import assess_image
    from gattaca.video.decode import write_video
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img, _ = synth.contactless_photo(3, sample=1, pprp=10)
    cv2.imwrite(str(out / "synthetic_finger.jpg"), img)
    res = assess_image(img)
    cv2.imwrite(str(out / "evidence.jpg"), res.evidence)
    print("photo:", res.summary()["band"], res.summary()["probability"])
    clip = synth.synthetic_clip(2, n_frames=24, downscale=2.6, seed=1)    # borderline: the agent must work
    write_video(str(out / "synthetic_clip.mp4"), clip, fps=12)
    r = run_agent(clip, AgentConfig(batch=12))
    print(f"clip: status={r.status} best={r.best.band} P={r.best.probability:.3f} from {r.best_source}")
    _print_trace(r.trace)
    (out / "agent_trace.json").write_text(json.dumps(r.trace, indent=1, default=float))
    print(f"outputs in {out}")
    return 0


def main(argv=None) -> int:
    print(DISCLAIMER, file=sys.stderr)
    ap = argparse.ArgumentParser(prog="gattaca", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assess")
    s.add_argument("image")
    s.add_argument("--debug")
    s.add_argument("--evidence")
    s.add_argument("--backend", default="auto")
    s.set_defaults(fn=cmd_assess)
    s = sub.add_parser("agent")
    s.add_argument("source")
    s.add_argument("--trace")
    s.set_defaults(fn=cmd_agent)
    s = sub.add_parser("protect")
    s.add_argument("image")
    s.add_argument("output")
    s.set_defaults(fn=cmd_protect)
    s = sub.add_parser("minutiae", help="forensic minutiae report (educational; lawfully held prints)")
    s.add_argument("image")
    s.add_argument("--out", default="minutiae_report")
    s.add_argument("--photo", action="store_true", help="input is a contactless fingertip photo, not a print")
    s.add_argument("--limit", type=int, default=0, help="max table rows to print (0 = all)")
    s.add_argument("--notes", action="store_true", help="also print the ISO/IEC 19794-2 explainer")
    s.set_defaults(fn=cmd_minutiae)
    s = sub.add_parser("consolidate", help="multi-frame, multi-finger exposure report from a video (own/consented hand)")
    s.add_argument("video")
    s.add_argument("--out", default="consolidation_report")
    s.add_argument("--k", type=int, default=6, help="frames to fuse per finger")
    s.add_argument("--max-frames", type=int, default=90)
    s.set_defaults(fn=cmd_consolidate)
    s = sub.add_parser("demo")
    s.add_argument("--out", default="demo")
    s.set_defaults(fn=cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
