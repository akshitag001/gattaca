"""Forensic minutiae report — an EDUCATIONAL tool for the classical AFIS pipeline.

Intended for forensic-science coursework and research on lawfully held fingerprint
images: your own inked/livescan prints, lab exemplars, or standard research datasets
(FVC, NIST SD, the paired contactless sets in datasets/README.md). It visualises the
standard minutiae-extraction chain and the fields of an ISO/IEC 19794-2 minutiae
record, so you can see what a template actually contains.

Scope (CLAUDE.md section 0): this reports the minutiae a classical pipeline finds and
explains the template format. It does NOT tune anything to pass or spoof a real matcher
or sensor, and it does NOT produce presentation-attack instruments (molds, 3D prints).
Those are not part of learning minutiae extraction and are out of scope here.

    python -m gattaca minutiae print.png --out report/          # inked / livescan print
    python -m gattaca minutiae fingertip.jpg --photo --out report/   # contactless fingertip photo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gattaca.analyze import PrintAnalysis, analyze_contact, analyze_photo
from gattaca.debug import to_u8
from gattaca.pipeline.enhance import EnhanceParams
from gattaca.pipeline.extract import binarize, thin
from gattaca.types import BIFURCATION, ENDING

TYPE_NAME = {ENDING: "ridge-ending", BIFURCATION: "bifurcation"}
EDU_BANNER = "GATTACA forensic minutiae report - educational use on lawfully held prints"


@dataclass
class MinutiaeReport:
    analysis: PrintAnalysis
    source: str                 # "contact" | "contactless"
    chart: np.ndarray           # BGR multi-panel figure
    table: list[dict]           # one row per minutia


def _panel(img: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    """Letterbox `img` into a titled BGR panel of the given (w, h)."""
    w, h = size
    g = to_u8(img)
    bgr = g if g.ndim == 3 else cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    ih, iw = bgr.shape[:2]
    s = min((w - 16) / iw, (h - 40) / ih)
    r = cv2.resize(bgr, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 24, np.uint8)
    y0, x0 = 34 + (h - 40 - r.shape[0]) // 2, 8 + (w - 16 - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def minutiae_overlay(an: PrintAnalysis) -> np.ndarray:
    """Skeleton with minutiae marked: endings red, bifurcations green, with orientation."""
    binary = binarize(an.enh, float(np.median(an.enh.period[an.enh.mask > 0])) if (an.enh.mask > 0).any() else 9.0)
    skel = thin(binary)
    vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    vis[(skel > 0)] = (120, 120, 120)
    for m in an.minutiae:
        c = (60, 60, 230) if m["kind"] == ENDING else (60, 200, 60)
        p = (int(m["x"]), int(m["y"]))
        cv2.circle(vis, p, 5, c, 1, cv2.LINE_AA)
        q = (int(m["x"] + 12 * np.cos(m["angle"])), int(m["y"] + 12 * np.sin(m["angle"])))
        cv2.line(vis, p, q, c, 1, cv2.LINE_AA)
    return vis


def build_table(an: PrintAnalysis) -> list[dict]:
    rows = []
    for i, m in enumerate(np.sort(an.minutiae, order=["y", "x"]), 1):
        rows.append({"index": i, "x_px": round(float(m["x"]), 1), "y_px": round(float(m["y"]), 1),
                     "angle_deg": round(float(np.degrees(m["angle"])) % 360, 1),
                     "type": TYPE_NAME[int(m["kind"])], "quality_0_100": int(round(float(m["quality"]) * 100))})
    return rows


def _skeleton(an: PrintAnalysis) -> np.ndarray:
    period = float(np.median(an.enh.period[an.enh.mask > 0])) if (an.enh.mask > 0).any() else 9.0
    return thin(binarize(an.enh, period))


def render_chart(an: PrintAnalysis, source: str, title: str) -> np.ndarray:
    pw, ph = 420, 360
    panels = [
        _panel(an.canonical, "1. Input (scaled to ~500 ppi)", (pw, ph)),
        _panel(an.enh.enhanced, "2. Ridge enhancement (Gabor)", (pw, ph)),
        _panel(_skeleton(an), "3. Thinned ridge skeleton", (pw, ph)),
        _panel(minutiae_overlay(an), "4. Minutiae (red=ending, green=bifurcation)", (pw, ph)),
    ]
    top = np.hstack([panels[0], panels[1]])
    bot = np.hstack([panels[2], panels[3]])
    grid = np.vstack([top, bot])
    endings = int((an.minutiae["kind"] == ENDING).sum())
    bifs = int((an.minutiae["kind"] == BIFURCATION).sum())
    q = an.quality
    footer = np.full((70, grid.shape[1], 3), 18, np.uint8)
    line1 = (f"{len(an.minutiae)} minutiae  ({endings} endings, {bifs} bifurcations)   "
             f"PPRP {q.pprp:.1f} px/ridge   mean quality {q.minutiae_quality:.2f}   clarity {q.clarity:.2f}")
    cv2.putText(footer, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(footer, EDU_BANNER + f"   |   {title}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([grid, footer])


def minutiae_report(img: np.ndarray, *, photo: bool = False, title: str = "",
                    params: EnhanceParams = EnhanceParams()) -> MinutiaeReport:
    """Run the classical pipeline and assemble a forensic minutiae report."""
    if photo:
        fingers = analyze_photo(img, params=params, max_fingers=1)
        if not fingers:
            raise ValueError("no fingertip found in the photo; try a tighter crop or a clearer image")
        an = fingers[0].analysis
        source = "contactless"
    else:
        an = analyze_contact(img)
        source = "contact"
    chart = render_chart(an, source, title or source)
    return MinutiaeReport(an, source, chart, build_table(an))


ISO_19794_2_NOTE = """ISO/IEC 19794-2 - Finger Minutiae Record (how the table maps to the standard)

A minutiae template is a compact record, not a ridge image. The core per-minutia fields:

  Field                Standard definition                         This report's column
  -------------------  ------------------------------------------  ---------------------
  Minutia type         ridge ending / bifurcation (2 bits)         type
  X / Y position       location on a 500-ppi grid (14 bits each)   x_px / y_px
  Angle                ridge direction, 0-255 (~1.4 deg steps)     angle_deg (0-360)
  Quality              0-100 confidence                            quality_0_100

A full record also carries a header (format id, capture device, image size, ppi),
and optional extended data (ridge-count between minutiae, core/delta singular points).
Related standards: ANSI/NIST-ITL 1-2011 (type-9 minutiae exchange), WSQ (500-ppi
grayscale image compression used by the FBI IAFIS/NGI).

Educational notes:
  * Position and angle are what a matcher aligns; ridge counts add robustness.
  * The 500-ppi grid is why GATTACA rescales every input to ~9 px per ridge first.
  * Endings and bifurcations are duals: dilating/eroding a print swaps many of them,
    which is why polarity-robust angle estimation matters.
"""


def write_report(report: MinutiaeReport, out_dir: str | Path, stem: str = "minutiae") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart_path = out / f"{stem}_chart.png"
    csv_path = out / f"{stem}.csv"
    note_path = out / "ISO19794-2_notes.txt"
    cv2.imwrite(str(chart_path), report.chart)
    cols = ["index", "x_px", "y_px", "angle_deg", "type", "quality_0_100"]
    lines = ["# " + EDU_BANNER, "# source: " + report.source, ",".join(cols)]
    lines += [",".join(str(r[c]) for c in cols) for r in report.table]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    note_path.write_text(ISO_19794_2_NOTE, encoding="utf-8")
    return {"chart": str(chart_path), "csv": str(csv_path), "notes": str(note_path),
            "minutiae": len(report.table)}
