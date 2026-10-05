"""The agentic loop: OpenCV measurements decide what the system does next.

Goal: estimate the **worst-case** exposure of a clip or photo, i.e. what a motivated
adversary using the best frames, better enhancement and multi-frame fusion could
recover. A privacy assessor must not stop at the first mediocre frame. So the agent
keeps choosing the most promising OpenCV action until the estimate converges, the
risk is already HIGH, or the budget runs out.

PERCEIVE  frame quality scores (sharpness, ridge SNR, exposure, motion) and the
          current assessment (band, P, limiting factor, clarity, PPRP)
DECIDE    an ordered rule policy (POLICY below); each decision cites the rule and
          the measurement that triggered it
ACT       invoke one tool: pull_frames | reselect_frame | retune_enhance |
          switch_finger | fuse_frames | stop

Human control: frame/step budgets, an allow-list of actions, an `approve` hook
consulted before expensive actions, and an `on_step` hook that can abort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gattaca.agent import tools
from gattaca.pipeline.enhance import EnhanceParams

POLICY = [
    ("R1", "no finger visible in the frames assessed so far, frames remain", "pull_frames"),
    ("R2", "estimate is HIGH with confidence >= 0.7", "stop"),
    ("R3", "limiting factor is focus/blur or motion, and sharper unassessed frames exist", "reselect_frame"),
    ("R4", "not HIGH, >=3 aligned-quality frames selected, not fused yet", "fuse_frames"),
    ("R5", "ridge clarity low although resolution is adequate, presets remain", "retune_enhance"),
    ("R6", "another detected finger has not been assessed", "switch_finger"),
    ("R7", "unread frames remain in the source and the pool is small", "pull_frames"),
    ("R8", "no further action is expected to change the estimate", "stop"),
]


@dataclass
class AgentConfig:
    frame_budget: int = 90            # max frames decoded
    batch: int = 30                   # frames per pull
    max_steps: int = 12
    k_fuse: int = 6
    allowed: frozenset = frozenset({"pull_frames", "reselect_frame", "retune_enhance",
                                    "switch_finger", "fuse_frames", "stop"})
    approve: Callable[[str, dict], bool] | None = None      # human gate for expensive actions
    on_step: Callable[[dict], bool] | None = None           # return False to abort


@dataclass
class AgentResult:
    best: object | None                       # Recoverability
    best_source: str
    trace: list[dict]
    evidence_input: np.ndarray | None = None  # the image the best estimate came from
    best_finger: object | None = None
    status: str = "converged"
    timings: dict = field(default_factory=dict)


class RecoverabilityAgent:
    def __init__(self, frame_source: Callable[[int, int], list[np.ndarray]], cfg: AgentConfig = AgentConfig()):
        """`frame_source(start, n)` returns up to n frames starting at index `start`."""
        self.src = frame_source
        self.cfg = cfg
        self.frames: list[np.ndarray] = []
        self.exhausted = False
        self.trace: list[dict] = []
        self.boxes: list = []
        self.motion = np.zeros(0)
        self.scores: list = []
        self.assessed_frames: set[int] = set()
        self.finger = 0
        self.n_fingers = 0
        self.tried_fingers: set[int] = set()
        self.preset = -1
        self.fused = False
        self.best = None
        self.best_source = ""
        self.best_img = None
        self.best_fa = None
        self.last = None              # last observation from an assessment
        self.history: list[float] = []
        self.ref_det = None

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, **kw) -> dict:
        e = {"step": len(self.trace), "t": round(time.perf_counter() - self.t0, 3), "kind": kind, **kw}
        self.trace.append(e)
        return e

    def _consider(self, obs, rec, fa, img, source):
        self.last = obs
        if rec is not None:
            self.history.append(rec.probability)
            if self.best is None or rec.probability > self.best.probability:
                self.best, self.best_source, self.best_img, self.best_fa = rec, source, img, fa

    def _pull(self) -> dict:
        new = self.src(len(self.frames), min(self.cfg.batch, self.cfg.frame_budget - len(self.frames)))
        if not new:
            self.exhausted = True
        self.frames += new
        if len(self.frames) >= self.cfg.frame_budget:
            self.exhausted = True
        return {"pulled": len(new), "pool": len(self.frames), "exhausted": self.exhausted}

    def _perceive_frames(self) -> dict:
        """(Re)detect on the sharpest-looking frame, track, score every frame."""
        if self.ref_det is None:
            lap = [float(np.var(np.diff(f.astype(np.float32), axis=1))) for f in self.frames]
            ref = int(np.argmax(lap))
            obs, dets = tools.tool_detect(self.frames[ref])
            self.n_fingers = len(dets)
            if not dets:
                return {"detected": 0, "ref_frame": ref}
            self.ref_det, self.ref_idx = dets[self.finger % len(dets)], ref
            self.dets = dets
        box = tools.finger_box(self.ref_det, self.frames[0].shape)
        tobs, (self.boxes, self.motion) = tools.tool_track(self.frames, box, self.ref_idx)
        sobs, self.scores = tools.tool_score_frames(self.frames, self.boxes, self.motion, range(len(self.frames)))
        return {"detected": self.n_fingers, "ref_frame": self.ref_idx, **tobs, **sobs}

    def _assess_frame(self, idx: int, params: EnhanceParams) -> dict:
        obs, rec, fa = tools.tool_assess_frame(self.frames[idx], params, self.finger)
        self.assessed_frames.add(idx)
        self._consider(obs, rec, fa, self.frames[idx], f"frame {idx}")
        return obs

    def _params(self) -> EnhanceParams:
        return tools.RETUNE_PRESETS[self.preset] if self.preset >= 0 else EnhanceParams()

    # ------------------------------------------------------------------ policy
    def decide(self) -> tuple[str, str, str]:
        last = self.last or {}
        ranked = sorted(self.scores, key=lambda s: -s.score)
        unassessed = [s for s in ranked if s.index not in self.assessed_frames]
        if not last.get("band") and not self.exhausted:
            why = "no frames decoded yet" if not self.frames else f"no finger found in {len(self.frames)} frames"
            return "R1", why, "pull_frames"
        if last.get("band") == "HIGH" and last.get("confidence", 0) >= 0.7:
            return "R2", f"band HIGH, confidence {last['confidence']}", "stop"
        if last.get("limiting_factor") in ("focus/blur",) and unassessed:
            cur = max((s.sharpness for s in self.scores if s.index in self.assessed_frames), default=0)
            if unassessed[0].sharpness > 1.1 * cur:
                return "R3", (f"limiting factor '{last['limiting_factor']}'; frame {unassessed[0].index} "
                              f"is sharper ({unassessed[0].sharpness:.2f} vs {cur:.2f})"), "reselect_frame"
        good = [s for s in ranked if np.isfinite(s.score)][: self.cfg.k_fuse]
        if last.get("band") != "HIGH" and len(good) >= 3 and not self.fused and last.get("band"):
            return "R4", f"band {last.get('band')} with {len(good)} usable frames; fusion can add sub-pixel detail", \
                "fuse_frames"
        if last.get("band") and last.get("clarity", 1) < 0.3 and last.get("limiting_factor") != "resolution" \
                and self.preset + 1 < len(tools.RETUNE_PRESETS):
            return "R5", f"clarity {last.get('clarity')} at adequate PPRP {last.get('pprp')}", "retune_enhance"
        untried = [i for i in range(self.n_fingers) if i not in self.tried_fingers]
        if last.get("band") != "HIGH" and untried:
            return "R6", f"{len(untried)} other finger(s) detected; current band {last.get('band')}", "switch_finger"
        if not self.exhausted and len(self.frames) < self.cfg.k_fuse and self.fused is False:
            return "R7", f"pool has only {len(self.frames)} frames", "pull_frames"
        return "R8", "no remaining action is expected to raise the estimate", "stop"

    # ------------------------------------------------------------------ act
    def act(self, action: str) -> dict:
        if action == "pull_frames":
            obs = self._pull()
            self.ref_det = None
            pobs = self._perceive_frames() if self.frames else {"detected": 0}
            obs.update(pobs)
            if pobs.get("detected"):
                best = max(self.scores, key=lambda s: s.score)
                obs["assessment"] = self._assess_frame(best.index, self._params())
                self.tried_fingers.add(self.finger)
            return obs
        if action == "reselect_frame":
            nxt = next(s for s in sorted(self.scores, key=lambda s: -s.sharpness)
                       if s.index not in self.assessed_frames)
            return {"frame": nxt.index, "assessment": self._assess_frame(nxt.index, self._params())}
        if action == "retune_enhance":
            self.preset += 1
            p = self._params()
            if self.best_source.startswith("frame"):
                idx = int(self.best_source.split()[1])
            else:
                idx = max(self.scores, key=lambda s: s.score).index
            return {"preset": self.preset, "params": {"denoise": p.denoise, "clahe_clip": p.clahe_clip,
                                                      "gabor_sigma": p.gabor_sigma},
                    "assessment": self._assess_frame(idx, p)}
        if action == "switch_finger":
            self.finger = next(i for i in range(self.n_fingers) if i not in self.tried_fingers)
            self.tried_fingers.add(self.finger)
            self.ref_det = None
            self.fused = False
            pobs = self._perceive_frames()
            best = max(self.scores, key=lambda s: s.score)
            return {"finger": self.finger, **pobs, "assessment": self._assess_frame(best.index, self._params())}
        if action == "fuse_frames":
            sel = tools.select_best(self.scores, k=self.cfg.k_fuse)
            capture_pprp = float((self.last or {}).get("pprp") or 3.0)   # PPRP measured on a single frame
            obs, rec, fa, fused = tools.tool_fuse(self.frames, self.boxes, sel, capture_pprp, self._params())
            self.fused = True
            self._consider(obs, rec, fa, fused, f"fusion of frames {sel}")
            return {"frames": sel, "assessment": obs}
        raise ValueError(action)

    # ------------------------------------------------------------------ run
    def run(self) -> AgentResult:
        self.t0 = time.perf_counter()
        status = "converged"
        for _ in range(self.cfg.max_steps):
            rule, why, action = self.decide()
            if action not in self.cfg.allowed:
                self._log("decision", rule=rule, reason=why, action=action, note="not allowed; stopping")
                status = "blocked"
                break
            if action == "fuse_frames" and self.cfg.approve and not self.cfg.approve(action, {"reason": why}):
                self._log("decision", rule=rule, reason=why, action=action, note="human declined")
                self.fused = True  # do not ask again
                continue
            entry = self._log("decision", rule=rule, reason=why, action=action,
                              perception=dict(self.last or {}))
            if action == "stop":
                status = "high-risk" if rule == "R2" else "converged"
                break
            before = self.best.probability if self.best else 0.0
            result = self.act(action)
            after = self.best.probability if self.best else 0.0
            self._log("result", action=action, observation=result, best_probability=round(after, 3),
                      delta=round(after - before, 3))
            if self.cfg.on_step and self.cfg.on_step(entry) is False:
                status = "aborted"
                break
        else:
            status = "budget"
        return AgentResult(self.best, self.best_source, self.trace, self.best_img, self.best_fa, status)


def run_agent(frames_or_image, cfg: AgentConfig = AgentConfig()) -> AgentResult:
    """Run on a list of frames or a single image (treated as a 1-frame source)."""
    pool = [frames_or_image] if isinstance(frames_or_image, np.ndarray) else list(frames_or_image)

    def source(start: int, n: int):
        return pool[start:start + n]

    return RecoverabilityAgent(source, cfg).run()
