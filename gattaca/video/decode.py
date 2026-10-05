"""Frame extraction with cv2.VideoCapture.

Uploaded bytes must touch disk because VideoCapture needs a path. The temp file is
deleted in `finally:` whether decoding succeeds or not (docs/responsible_use.md)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Video:
    frames: list[np.ndarray]
    fps: float
    total: int          # frames in the source (may exceed len(frames))


def _resize(f: np.ndarray, max_side: int) -> np.ndarray:
    h, w = f.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else f


def read_video(path: str, max_frames: int = 90, stride: int = 1, max_side: int = 1920) -> Video:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames, i = [], 0
    try:
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(_resize(f, max_side))
            i += 1
    finally:
        cap.release()
    return Video(frames, float(fps), max(total, i))


def read_video_bytes(data: bytes, suffix: str = ".mp4", **kw) -> Video:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="gattaca_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return read_video(path, **kw)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def write_video(path: str, frames: list[np.ndarray], fps: float = 30.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for f in frames:
            vw.write(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    finally:
        vw.release()
