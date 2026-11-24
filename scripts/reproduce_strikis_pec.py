"""Recover the displayed hardware estimates in Strikis et al., PRX Quantum Fig. 10."""
import argparse
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from statistics import mean

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/pec_strikis/figure_scores.json"
JOURNAL_URL = "https://harvest.aps.org/v2/journals/articles/10.1103/PRXQuantum.2.040330/fulltext"
JOURNAL_SHA = "72c58d8e8034cad3e043d0fa20d99432551eeec8b237c0c2da48811dc5c3ae38"
SOURCE_URL = "https://arxiv.org/src/2005.07601v2"
SOURCE_SHA = "0f1c68f8f419c535e487631a3415dfbf5c0ecd72c341d5ce007ef299ee5f0ca9"
FIGURE_SHA = "c1e1425a86c1c23f17dbe00ca32ce735edd0980788f580de5cdc10e9b2ed2b64"
DEVICES = ("ibmq_5_yorktown", "ibmq_ourense", "ibmq_santiago")
METHODS = ("raw", "mitigated")


def checked(payload, expected):
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Source checksum differs from the pinned publication")
    return payload


def points(page):
    # Each unbroken gray ideal curve contains 101 vertices and spans 0 to 2π.
    curves = sorted((c for c in page.curves if len(c["pts"]) == 101 and c["stroke"]),
                    key=lambda c: c["x0"])
    if len(curves) != len(DEVICES):
        raise ValueError("Expected three hardware panels")
    labels = page.extract_words()
    raw = [c for c in page.curves if c["fill"] and c["non_stroking_color"] == (0., 0., 1.)]
    corrected = [r for r in page.rects if r["fill"] and isinstance(r["non_stroking_color"], tuple)
                 and tuple(round(v, 2) for v in r["non_stroking_color"]) == (1., 0.50, 0.)]
    rows = []
    max_ideal_error = 0
    for device, curve in zip(DEVICES, curves):
        left, width = curve["x0"], curve["width"]
        if not any(w["text"].lower() == device and left < (w["x0"] + w["x1"]) / 2 < left + width
                   and 0 < curve["top"] - w["top"] < width / 5 for w in labels):
            raise ValueError(f"Could not confirm the label for panel {device}")
        # The longer left-side ticks mark +1, +0.5, 0, -0.5 and -1.
        ticks = sorted(l["top"] for l in page.lines if abs(l["x0"] - left) < width / 1000
                       and 0.010 < l["width"] / width < 0.013 and l["height"] < 0.001
                       and curve["top"] - 1 < l["top"] < curve["bottom"] + 1)
        if len(ticks) != 5:
            raise ValueError("Missing labeled vertical-axis ticks")
        unit = (ticks[-1] - ticks[0]) / 2
        zero = (ticks[-1] + ticks[0]) / 2
        if max(abs((zero - y) / unit - (1 - i / 2)) for i, y in enumerate(ticks)) > 0.0001:
            raise ValueError("Nonuniform vertical axis")
        for x, y in curve["pts"]:
            ideal = math.cos(2 * math.pi * (x - left) / width)
            max_ideal_error = max(max_ideal_error, abs((zero - y) / unit - ideal))

        for m in range(10):
            x = left + width * m / 10
            row = dict(device=device, phase_index=m, theta_over_pi=m / 5,
                       ideal=round(math.cos(math.pi * m / 5), 9))
            for key, markers in zip(METHODS, (raw, corrected)):
                # The zero- and π-phase markers are included, and legend symbols
                # fall outside the ten target x coordinates.
                matched = [s for s in markers if abs((s["x0"] + s["x1"]) / 2 - x) < width / 1000
                           and ticks[0] - unit / 10 < s["top"] < ticks[-1] + unit / 10]
                if len(matched) != 1:
                    raise ValueError(f"Expected one {key} marker for {device} at phase {m}")
                marker = matched[0]
                center = (marker["top"] + marker["bottom"]) / 2
                row[key] = round((zero - center) / unit, 8)
            rows.append(row)
    if max_ideal_error > 0.0002:
        raise ValueError("Ideal trace disagrees with the analytic cosine")
    return rows, max_ideal_error


def measure(rows):
    return dict(
        device_phase_settings=len(rows),
        **{method + "_mae": round(mean(abs(r[method] - r["ideal"]) for r in rows), 6)
           for method in METHODS},
        settings_with_smaller_absolute_error=sum(abs(r["mitigated"] - r["ideal"]) <
                                                 abs(r["raw"] - r["ideal"]) for r in rows),
    )


def extract(journal_path, archive_path):
    with pdfplumber.open(io.BytesIO(checked(journal_path.read_bytes(), JOURNAL_SHA))) as doc:
        journal, journal_curve_error = points(doc.pages[11])
    with tarfile.open(fileobj=io.BytesIO(checked(archive_path.read_bytes(), SOURCE_SHA))) as archive:
        inventory = sorted(m.name for m in archive.getmembers() if m.isfile())
        figure = checked(archive.extractfile("IBMq2.pdf").read(), FIGURE_SHA)
    with pdfplumber.open(io.BytesIO(figure)) as doc:
        preprint, preprint_curve_error = points(doc.pages[0])
    agreement = max(abs(j[method] - a[method]) for j, a in zip(journal, preprint)
                    for method in METHODS)
    if agreement > 0.0001:
        raise ValueError("The journal and preprint marker estimates disagree")
    return {
        "citation": "Strikis et al., Learning-Based Quantum Error Mitigation, PRX Quantum 2, 040330 (2021), Figure 10.",
        "doi": "https://doi.org/10.1103/PRXQuantum.2.040330",
        "journal_pdf": {"url": JOURNAL_URL, "sha256": JOURNAL_SHA, "page": 12},
        "preprint_source": {"url": SOURCE_URL, "sha256": SOURCE_SHA, "files": inventory,
                            "figure": "IBMq2.pdf", "figure_sha256": FIGURE_SHA},
        "observable": "Upper-qubit Z expectation for the two-qubit DQCp circuit at θ = 2πm/10. The ideal equals cos(θ).",
        "derivation": "Marker centers from the journal's blue circles (raw) and orange squares (mitigated), "
                      "scaled by the labeled y-axis ticks. Both series are plotted final estimates. Values "
                      "are approximate. MAE is our equal-weight descriptive summary.",
        "verification": {"max_marker_difference_journal_vs_preprint": round(agreement, 8),
                         "max_ideal_curve_error_journal": round(journal_curve_error, 8),
                         "max_ideal_curve_error_preprint": round(preprint_curve_error, 8)},
        "protocol": {"devices": list(DEVICES), "training_clifford_circuits_per_device": 24,
                     "test_phase_settings_per_device": 10, "pauli_insertions_per_phase": ["I", "X", "Y", "Z"],
                     "shots_per_measured_circuit": 8192,
                     "data_available_in_figure": "I-insertion test estimates and learned quasiprobability outputs.",
                     "still_required": "Identified Clifford calibration means and the X/Y/Z insertion test "
                                       "measurements. Figure coordinates do not supply those inputs."},
        "scores_by_device": {d: measure([row for row in journal if row["device"] == d]) for d in DEVICES},
        "across_devices": measure(journal),
        "device_phase_estimates": journal,
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("journal_pdf", type=Path)
    cli.add_argument("arxiv_source", type=Path)
    cli.add_argument("--check", action="store_true")
    args = cli.parse_args()
    result = extract(args.journal_pdf, args.arxiv_source)
    if args.check:
        if result != json.loads(REPORT.read_text()):
            raise ValueError("Committed figures or summary differ from source")
    else:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("Verified all 60 hardware markers against two published versions of Figure 10.")
