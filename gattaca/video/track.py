"""Follow a fingertip box across frames with pyramidal Lucas-Kanade optical flow."""

from __future__ import annotations

import cv2
import numpy as np


def _features(gray: np.ndarray, box) -> np.ndarray | None:
    x, y, w, h = [int(v) for v in box]
    m = np.zeros_like(gray)
    m[max(y, 0):y + h, max(x, 0):x + w] = 255
    return cv2.goodFeaturesToTrack(gray, maxCorners=80, qualityLevel=0.01, minDistance=5, mask=m)


def track_box(frames: list[np.ndarray], box, start: int = 0):
    """Boxes (x, y, w, h) per frame and inter-frame motion (px) of the tracked region.
    Frames before `start` are tracked backwards."""
    n = len(frames)
    boxes: list[tuple | None] = [None] * n
    motion = np.zeros(n)
    boxes[start] = tuple(float(v) for v in box)
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f for f in frames]
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    for direction in (1, -1):
        cur = start
        pts = _features(gray[cur], boxes[cur])
        while 0 <= cur + direction < n:
            nxt = cur + direction
            if pts is None or len(pts) < 6:
                pts = _features(gray[cur], boxes[cur])
                if pts is None or len(pts) < 6:
                    boxes[nxt] = boxes[cur]
                    cur = nxt
                    continue
            new, st, _ = cv2.calcOpticalFlowPyrLK(gray[cur], gray[nxt], pts, None, **lk)
            # forward-backward check rejects drifting points
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[nxt], gray[cur], new, None, **lk)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 1.0)
            if ok.sum() >= 4:
                d = np.median((new - pts).reshape(-1, 2)[ok], axis=0)
                x, y, w, h = boxes[cur]
                boxes[nxt] = (x + d[0], y + d[1], w, h)
                motion[nxt] = float(np.linalg.norm(d))
                pts = new[ok].reshape(-1, 1, 2)
            else:
                boxes[nxt] = boxes[cur]
                pts = None
            cur = nxt
    return boxes, motion
