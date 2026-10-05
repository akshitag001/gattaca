import numpy as np

from gattaca import synth
from gattaca.agent import AgentConfig, run_agent
from gattaca.analyze import analyze_contact, analyze_photo
from gattaca.pipeline.match import match
from gattaca.video.align import align
from gattaca.video.fuse import fuse
from gattaca.video.select import score_frame, select_best


def test_ecc_recovers_subpixel_shift():
    frames = synth.synthetic_clip(1, n_frames=2, downscale=2.0, blur_frames=0, seed=3)
    import cv2
    shifted = cv2.warpAffine(frames[0], np.float32([[1, 0, 1.4], [0, 1, -0.6]]), frames[0].shape[1::-1],
                             borderMode=cv2.BORDER_REFLECT)
    W, cc, method = align(frames[0], shifted)
    assert method == "ecc" and cc > 0.9
    # ECC convention: W maps reference coords -> frame coords, so it equals the applied shift
    assert abs(W[0, 2] - 1.4) < 0.15 and abs(W[1, 2] - (-0.6)) < 0.15


def test_frame_selection_prefers_sharp_frames():
    frames = synth.synthetic_clip(2, n_frames=12, downscale=2.0, blur_frames=0.5, seed=4)
    h, w = frames[0].shape[:2]
    box = (w * 0.2, h * 0.05, w * 0.6, h * 0.5)
    scores = [score_frame(f, box, i) for i, f in enumerate(frames)]
    best = select_best(scores, k=3)
    by_sharp = [s.index for s in sorted(scores, key=lambda s: s.sharpness)]
    blurriest = set(by_sharp[:3])
    assert not blurriest & set(best)            # motion-blurred frames are never chosen
    assert set(best) <= set(by_sharp[len(by_sharp) // 2:])


def test_fusion_beats_best_single_frame():
    gained = 0
    for ident in range(3):
        frames = synth.synthetic_clip(ident, n_frames=8, downscale=3.2, blur_frames=0, seed=ident)
        gt = analyze_contact(synth.contact_impression(ident))
        fused, info = fuse(frames, 0, scale=2)
        assert info["frames_used"] >= 6
        n = []
        for im in (frames[0], fused):
            f = analyze_photo(im, backend="classical", max_fingers=1)
            n.append(match(f[0].analysis.minutiae, gt.minutiae).n_matched if f else 0)
        gained += n[1] > n[0]
    assert gained >= 2


def test_agent_stops_early_when_risk_is_already_high():
    frames = synth.synthetic_clip(2, n_frames=24, downscale=2.0, seed=1)   # sharp frames available
    res = run_agent(frames, AgentConfig(batch=12))
    decisions = [e for e in res.trace if e["kind"] == "decision"]
    assert [d["rule"] for d in decisions] == ["R1", "R2"] and res.status == "high-risk"


def test_agent_trace_shows_vision_driving_actions():
    frames = synth.synthetic_clip(2, n_frames=24, downscale=2.6, seed=1)   # borderline resolution
    res = run_agent(frames, AgentConfig(batch=12))
    decisions = [e for e in res.trace if e["kind"] == "decision"]
    assert decisions[0]["action"] == "pull_frames"
    assert decisions[-1]["action"] == "stop"
    actions = [d["action"] for d in decisions]
    assert "fuse_frames" in actions
    # the fusion decision was triggered by the measured band of the first assessment
    fuse = next(d for d in decisions if d["action"] == "fuse_frames")
    assert fuse["perception"]["band"] in ("LOW", "MEDIUM") and fuse["rule"] == "R4"
    # every non-initial decision cites a measurement from the previous observation
    for d in decisions[1:]:
        assert d["perception"] and d["reason"]
    assert res.best is not None


def test_agent_respects_human_control():
    frames = synth.synthetic_clip(2, n_frames=12, downscale=3.0, seed=1)
    asked = []
    res = run_agent(frames, AgentConfig(batch=12, approve=lambda a, ctx: asked.append(a) or False))
    assert "fuse_frames" in asked
    assert not any(e.get("action") == "fuse_frames" and e["kind"] == "result" for e in res.trace)
    res2 = run_agent(frames, AgentConfig(batch=12, allowed=frozenset({"pull_frames", "stop"})))
    assert res2.status in ("blocked", "converged")


def test_consolidation_report_runs_and_compares():
    from gattaca.consolidate import consolidate_video
    frames = synth.synthetic_clip(4, n_frames=16, downscale=2.6, seed=2)
    rep = consolidate_video(frames, k_fuse=6)
    assert rep.n_frames == 16 and len(rep.fingers) >= 1
    f = rep.fingers[0]
    assert f.single.band in ("LOW", "MEDIUM", "HIGH") and f.fused.band in ("LOW", "MEDIUM", "HIGH")
    assert f.frames_used >= 2
    s = rep.summary()
    assert set(s["per_finger"][0]) == {"label", "single", "fused", "delta_p", "minutiae_gain", "frames_used"}
    assert rep.chart.ndim == 3
    # the report is a measurement, not an identification: no identity/template fields
    import json
    assert not any(k in json.dumps(s).lower() for k in ("identity", "name", "template", "subject"))
