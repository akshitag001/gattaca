"""Forensic minutiae report (educational AFIS pipeline on fingerprint images)."""

import numpy as np

from gattaca import synth
from gattaca.forensic import ISO_19794_2_NOTE, minutiae_report, write_report


def test_report_on_contact_print(contact0):
    rep = minutiae_report(contact0)
    assert rep.source == "contact"
    assert 15 <= len(rep.table) <= 120
    r = rep.table[0]
    assert set(r) == {"index", "x_px", "y_px", "angle_deg", "type", "quality_0_100"}
    assert r["type"] in ("ridge-ending", "bifurcation")
    assert 0 <= r["angle_deg"] < 360 and 0 <= r["quality_0_100"] <= 100
    # chart is a 2x2 grid plus footer
    assert rep.chart.ndim == 3 and rep.chart.shape[0] > rep.chart.shape[1] // 2


def test_report_on_contactless_photo(photo0):
    rep = minutiae_report(photo0[0], photo=True)
    assert rep.source == "contactless" and len(rep.table) > 20


def test_write_report_files(contact0, tmp_path):
    info = write_report(minutiae_report(contact0), tmp_path, stem="p")
    csv = (tmp_path / "p.csv").read_text()
    assert csv.startswith("# GATTACA forensic minutiae report")
    assert "x_px,y_px,angle_deg,type,quality_0_100" in csv
    assert (tmp_path / "p_chart.png").exists()
    assert "ISO/IEC 19794-2" in (tmp_path / "ISO19794-2_notes.txt").read_text()
    assert info["minutiae"] == len(csv.strip().splitlines()) - 3   # 2 comment lines + header
