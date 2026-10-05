"""Unified, dataset-agnostic access to paired contactless / contact fingerprints.

    from datasets.loader import load_dataset
    for pair in load_dataset("hkpolyu", split="test"):
        pair.contactless_img, pair.contact_gt, pair.subject_id, pair.finger_id, pair.meta

Licensed datasets are read from `datasets/<key>/manifest.csv` (written by
scripts/make_manifest.py once the data has been placed by hand). `synthetic`
generates pairs on the fly. Images load lazily, one pair at a time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LICENSED = {
    "hkpolyu": "HK PolyU Contactless 2D to Contact-based 2D v1.0",
    "ispfd1": "IIITD SmartPhone Fingerphoto Database v1",
    "ispfd2": "IIITD SmartPhone Finger-selfie Database v2",
    "ridgebase": "RidgeBase (UB CUBS)",
}


@dataclass
class FingerprintPair:
    """One contactless probe with its paired contact ground truth."""

    _load_cl: Callable[[], np.ndarray] = field(repr=False)
    _load_gt: Callable[[], np.ndarray] = field(repr=False)
    subject_id: str = ""
    finger_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def contactless_img(self) -> np.ndarray:
        return self._load_cl()

    @property
    def contact_gt(self) -> np.ndarray:
        return self._load_gt()

    @property
    def key(self) -> str:
        """Identity of the *finger* (gallery key)."""
        return f"{self.subject_id}/{self.finger_id}"


class DatasetNotAvailable(RuntimeError):
    pass


def _imread(path: Path, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    # np.fromfile + imdecode handles non-ASCII Windows paths, which cv2.imread does not
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if img is None:
        raise IOError(f"cannot read image {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


class ManifestDataset:
    def __init__(self, key: str, root: Path | None = None, split: str | None = None):
        self.key = key
        self.root = root or ROOT / key
        self.manifest = self.root / "manifest.csv"
        if not self.manifest.exists():
            raise DatasetNotAvailable(
                f"{LICENSED.get(key, key)} is not installed: expected {self.manifest}.\n"
                f"Request access (datasets/README.md), unpack to {self.root / 'raw'}, then run\n"
                f"  python scripts/make_manifest.py {key} --root {self.root / 'raw'}")
        with self.manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if split is None or r.get("split", "test") == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[FingerprintPair]:
        for r in self.rows:
            cl, gt = self.root / r["contactless"], self.root / r["contact"]
            meta = {k: v for k, v in r.items() if k not in ("contactless", "contact", "subject_id", "finger_id")}
            meta["dataset"] = self.key
            yield FingerprintPair(lambda p=cl: _imread(p), lambda p=gt: _imread(p, cv2.IMREAD_GRAYSCALE),
                                  r["subject_id"], r["finger_id"], meta)


class SyntheticDataset:
    """Procedural pairs from gattaca.synth. Subjects [0, n) for test, [n, 2n) for train."""

    def __init__(self, subjects: int = 30, samples: int = 1, pprp: float = 10.0, split: str = "test"):
        self.subjects, self.samples, self.pprp = subjects, samples, pprp
        self.offset = 0 if split == "test" else 100_000

    def __len__(self) -> int:
        return self.subjects * self.samples

    def __iter__(self) -> Iterator[FingerprintPair]:
        from gattaca import synth
        for s in range(self.subjects):
            ident = self.offset + s
            for k in range(self.samples):
                yield FingerprintPair(
                    lambda i=ident, k=k: synth.contactless_photo(i, sample=k + 1, pprp=self.pprp)[0],
                    lambda i=ident: synth.contact_impression(i, sample=0),
                    subject_id=f"syn{ident:05d}", finger_id="0",
                    meta={"dataset": "synthetic", "nominal_pprp": self.pprp, "sample": k})


def load_dataset(name: str, split: str = "test", **kw):
    if name == "synthetic":
        return SyntheticDataset(split=split, **kw)
    if name in LICENSED:
        return ManifestDataset(name, split=split, root=kw.get("root"))
    raise KeyError(f"unknown dataset {name!r}; choose from synthetic, {', '.join(LICENSED)}")


def available() -> dict[str, bool]:
    return {"synthetic": True, **{k: (ROOT / k / "manifest.csv").exists() for k in LICENSED}}
