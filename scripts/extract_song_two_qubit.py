"""Recover the four hardware means and error bars in Song et al., Science Advances Fig. 3G."""
import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from statistics import mean

import pdfplumber


JOURNAL_URL = "https://rcastoragev2.blob.core.windows.net/579b2353fdfbedf6bf0f02e0777ca90c/aaw5686.PMC6731091.pdf"
JOURNAL_SHA = "8c2c522803fdba4fa13bc2c27203217633a7b1f69eb60845679f4612ac0b79a3"
ARXIV_URL = "https://arxiv.org/src/1812.10903v1"
ARXIV_SHA = "ebd54f0b6f2de5661bb2e7bb16e08d1ef9cf44da2a20196a92367116f5eb3f38"
FIGURE_SHA = "dbcf769ef6ffa591a4bc6b20d1c235ae860d32a2b482e89d4390ef58380cb20f"
REPORT = Path(__file__).resolve().parents[1] / "reports/pec_song/two_qubit_targets.json"
PHASES = (0.25, 0.5, 0.75, 1.0)


def checked(payload, expected):
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Unexpected source checksum")
    return payload


def ideal(phase):
    return (1 + math.cos(math.pi * phase)) / 2


def center_y(obj):
    return (obj["top"] + obj["bottom"]) / 2


def stem_values(stems, y_zero, y_half, x_ticks):
    """Each marker is where the two signed error-bar stems of one color meet."""
    unit = 2 * (y_zero - y_half)
    if len(stems) != 8 or unit <= 0:
        raise ValueError("Expected two stems for each of the four phases")
    result = []
    for x in x_ticks:
        pair = sorted((s for s in stems if abs((s["x0"] + s["x1"]) / 2 - x) < 0.2),
                      key=lambda s: s["top"])
        if len(pair) != 2 or abs(pair[0]["bottom"] - pair[1]["top"]) > 0.0011:
            raise ValueError("Disconnected error bar or incorrect horizontal phase position")
        junction = (pair[0]["bottom"] + pair[1]["top"]) / 2
        result.append(((y_zero - junction) / unit, (pair[1]["bottom"] - pair[0]["top"]) / (2 * unit)))
    return result


def journal_points(page):
    # Journal production outlined its strokes as filled rectangles. These two
    # horizontal grid lines are at measured X expectation +0.5 and 0.
    grid = [r for r in page.rects if r["non_stroking_color"] == (0.87451,) * 3
            and 420 < r["x0"] < 493 and 30 < r["width"] < 35 and 0.23 < r["height"] < 0.25]
    y_half, y_zero = sorted({round(center_y(r), 5) for r in grid})
    phase_ticks = [r for r in page.rects if r["non_stroking_color"] == (0.76471,) * 3
                   and 420 < r["x0"] < 493 and abs(center_y(r) - y_zero) < 0.001]
    xs = sorted((r["x0"] + r["x1"]) / 2 for r in phase_ticks)
    if len(xs) != 3 or abs((xs[1] - xs[0]) - (xs[2] - xs[1])) > 0.01:
        raise ValueError("Expected regularly spaced phase grid intersections")
    xs.append(xs[2] + (xs[2] - xs[0]) / 2)
    result = {}
    for key, color in [("raw", (0.22353, 0.32549, 0.63922)), ("pec", (0.92549, 0.12549, 0.14118))]:
        stems = [r for r in page.rects if r["non_stroking_color"] == color and 421 < r["x0"] < 528
                 and 318 < r["top"] < 410 and 0.85 < r["width"] < 0.87]
        result[key] = stem_values(stems, y_zero, y_half, xs)
    return result


def preprint_points(page):
    grid = [l for l in page.lines if l["stroking_color"] == 0.149 and l["x0"] > 345
            and l["width"] > 140 and 300 < l["top"] < 360]
    y_half, y_zero = sorted(l["top"] for l in grid)
    x0, xmax = grid[0]["x0"], grid[0]["x1"]
    xs = [x0 + phase * (xmax - x0) for phase in PHASES]
    result = {}
    for key, color in [("raw", (0.224, 0.325, 0.639)), ("pec", (0.925, 0.125, 0.141))]:
        stems = [l for l in page.lines if l["stroking_color"] == color and 380 < l["x0"] < 492
                 and l["top"] > 280 and l["width"] == 0]
        result[key] = stem_values(stems, y_zero, y_half, xs)
    curve, = [c for c in page.curves if c["stroking_color"] == 0 and 347 < c["x0"] < 349
              and c["width"] > 140 and len(c["pts"]) == 33]
    unit = 2 * (y_zero - y_half)
    curve_error = max(abs((y_zero - y) / unit - ideal((x - x0) / (xmax - x0))) for x, y in curve["pts"])
    if curve_error > 0.002:
        raise ValueError("Digitized ideal curve disagrees with the controlled-phase circuit")
    return result, curve_error


def extract(journal_path, arxiv_path):
    with pdfplumber.open(io.BytesIO(checked(journal_path.read_bytes(), JOURNAL_SHA))) as doc:
        measured = journal_points(doc.pages[2])
    with tarfile.open(fileobj=io.BytesIO(checked(arxiv_path.read_bytes(), ARXIV_SHA))) as archive:
        payload = checked(archive.extractfile("figure3.pdf").read(), FIGURE_SHA)
    with pdfplumber.open(io.BytesIO(payload)) as doc:
        other, curve_error = preprint_points(doc.pages[0])
    rows = []
    for i, phase in enumerate(PHASES):
        row = dict(phase_over_pi=phase, ideal=ideal(phase))
        for method in ("raw", "pec"):
            value, sd = measured[method][i]
            row[method], row[method + "_sd"] = round(value, 8), round(sd, 8)
        rows.append(row)
    max_mean_diff = max(abs(measured[key][i][0] - other[key][i][0]) for i in range(4) for key in measured)
    max_sd_diff = max(abs(measured[key][i][1] - other[key][i][1]) for i in range(4) for key in measured)
    if max(max_mean_diff, max_sd_diff) >= 0.0001:
        raise ValueError("Conflicting journal and preprint estimates")
    anchor = rows[1]
    printed = {"raw": 0.1690, "pec": 0.0102}
    computed = {key: round(abs(anchor[key] - anchor["ideal"]), 8) for key in printed}
    if any(abs(printed[key] - computed[key]) > 0.0001 for key in printed):
        raise ValueError("Conflict with printed pi/2 absolute errors")
    return {
        "paper_doi": "10.1126/sciadv.aaw5686", "journal_figure": "3G", "journal_pdf_page": 3,
        "journal_pdf_url": JOURNAL_URL, "journal_pdf_sha256": JOURNAL_SHA,
        "validation_source_url": ARXIV_URL, "source_tar_sha256": ARXIV_SHA, "figure3_pdf_sha256": FIGURE_SHA,
        "observable": "Pauli X on Q1 following a controlled phase on |++>. The ideal is (1 + cos(phi))/2.",
        "data_basis": "The center of each journal vector error bar gives the aggregate hardware mean. "
                      "The 0 and +0.5 horizontal grid lines calibrate its vertical scale. "
                      "The phase is an identification key and the ideal value is reserved for scoring.",
        "error_bars": "The vertical half-range is the paper's SD across 100 batch averages of 10,000 instances. "
                      "Each plotted mean aggregates 1,000,000 instances; the bars are not its standard error.",
        "digitization_stress_step": 0.0001,
        "digitization_interpretation": "An analyst-chosen perturbation of each raw input, in expectation units. "
                                       "It covers the observed cross-version mean differences and printed-anchor "
                                       "discrepancies; it is separate from the finite-shot SD.",
        "max_journal_preprint_mean_difference": round(max_mean_diff, 8),
        "max_journal_preprint_sd_difference": round(max_sd_diff, 8),
        "max_black_curve_difference_at_33_vertices": round(curve_error, 8),
        "pi_over_two_anchor": {"printed_absolute_errors": printed, "extracted_absolute_errors": computed},
        "mean_absolute_error_over_four_phases": {
            key: round(mean(abs(row[key] - row["ideal"]) for row in rows), 8) for key in printed},
        "targets": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal_pdf", type=Path)
    parser.add_argument("arxiv_source_tar", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = extract(args.journal_pdf, args.arxiv_source_tar)
    if args.check:
        if json.loads(REPORT.read_text()) != result:
            raise ValueError("Saved target table differs from checked figures")
    else:
        REPORT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Verified four journal hardware means per method, their SDs, the pi/2 text anchors and the ideal curve.")
