"""Verify the Algiers hardware references in the authors' source workbook and notebook."""
import argparse
import hashlib
import io
import json
import posixpath
import re
from pathlib import Path
from statistics import mean
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "faf3e44ac686d62b2184a983d89e655084759a26"
SOURCE = "https://raw.githubusercontent.com/qiskit-community/ml-qem/" + REVISION + "/docs/paper_figures/ML-QEM%20Source%20data.xlsx"
DIGEST = "b42607140bfc732c73957a81f35a0bd4c0a21f78b9713a4b9207a107181eb71f"
GIT_BLOB = "d2dafa3c421b8b3e4c8d6c1e9c18919a1cda715e"


def read_depth_series(cells, header_row, expected_label):
    header = f"A{header_row}"
    label = f"A{header_row + 11}"
    if cells[header] != "step" or cells[label].split(",")[0] != f"Name: {expected_label}":
        raise ValueError(f"Unexpected Figure 4 labels for {expected_label}")
    values = []
    for depth in range(10):
        step, printed = cells[f"A{header_row + depth + 1}"].split()
        if int(step) != depth:
            raise ValueError("Unexpected figure depth order")
        values.append(float(printed))
    return {"header": header, "range": f"A{header_row + 1}:A{header_row + 10}",
            "series_label": cells[label], "series_label_cell": label,
            "mean_circuit_l2_per_depth_6dp": values, "equal_depth_mean_of_printed_values": mean(values)}


def notebook_reference():
    manifest = json.loads((ROOT / "data/upstream/manifest.json").read_text())
    if manifest["commit"] != REVISION:
        raise ValueError("Notebook and workbook revisions differ")
    entry, = [f for f in manifest["files"] if f["upstream_path"].endswith("/demo2_ising_4q_hardware_plot.ipynb")]
    path = ROOT / entry["local_path"]
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(entry["url"], timeout=60) as response:
            raw = response.read()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Unexpected upstream notebook")
    scores = {}
    names = {"noisy": "raw", "mitigated": "rf"}
    for cell in json.loads(raw)["cells"]:
        for output in cell.get("outputs", []):
            lines = "".join(output.get("text", [])).splitlines()
            for line in lines:
                match = re.fullmatch(r"RMSE_(noisy|mitigated): ([0-9.]+)", line)
                if match:
                    key = names[match[1]]
                    if key in scores:
                        raise ValueError(f"Multiple notebook pooled scores for {key}")
                    scores[key] = float(match[2])
    if set(scores) != set(names.values()):
        raise ValueError("Pooled notebook metrics missing")
    return {"source": entry["url"], "sha256": entry["sha256"], "printed_pooled_rmse": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, help="Optional local copy of the pinned upstream workbook")
    parser.add_argument("--check", action="store_true", help="Verify the committed report without rewriting it")
    args = parser.parse_args()
    if args.workbook:
        raw = args.workbook.read_bytes()
    else:
        with urlopen(SOURCE, timeout=60) as response:
            raw = response.read()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if hashlib.sha256(raw).hexdigest() != DIGEST or blob != GIT_BLOB:
        raise ValueError("Unexpected original workbook")
    with ZipFile(io.BytesIO(raw)) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        strings = ["".join(cell.itertext()) for cell in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        def sheet_cells(name):
            sheet = workbook.find(f".//{{*}}sheet[@name='{name}']")
            relation = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rels.find(f"{{*}}Relationship[@Id='{relation}']").attrib["Target"]
            xml = ET.fromstring(archive.read(posixpath.join("xl", target)))
            return {c.attrib["r"]: strings[int(c.findtext("{*}v"))] if c.get("t") == "s" else c.findtext("{*}v")
                    for c in xml.findall(".//{*}c")}
        cells = sheet_cells("Figure 4")
        figure2 = sheet_cells("Figure 2")
    series = {key: read_depth_series(cells, row, label) for key, row, label in (
        ("raw", 1, "L2_noisy"), ("rf", 13, "L2_rfr_list"), ("zne", 25, "L2_zne"))}
    notebook = notebook_reference()
    comparison = json.loads((ROOT / "reports/rf_reproduction.json").read_text())
    if comparison["reference_figure4_mean_circuit_l2_per_depth_6dp"] != series["rf"]["mean_circuit_l2_per_depth_6dp"]:
        raise ValueError("Workbook disagrees with RF reference used for reproduction")
    for key in ("raw", "zne"):
        reconstructed = [round(v, 6) for v in comparison[key]["mean_circuit_l2_per_depth"]]
        if reconstructed != series[key]["mean_circuit_l2_per_depth_6dp"]:
            raise ValueError(f"Reconstructed {key} scores disagree with the workbook")
    if notebook["printed_pooled_rmse"] != {"raw": comparison["raw"]["pooled_rmse"],
                                          "rf": comparison["reference_notebook_pooled_rmse"]}:
        raise ValueError("Pooled scores disagree with the saved notebook output")
    simulation_scores = {figure2[f"A{i}"]: float(figure2[f"B{i}"]) for i in range(1, 7)}
    if set(simulation_scores) != {"Unmitigated", "ZNE", "OLS", "RF", "MLP", "GNN"}:
        raise ValueError("Unexpected Figure 2 simulation reference labels")
    report = {"upstream_revision": REVISION, "source": SOURCE, "sha256": DIGEST, "git_blob_sha1": blob,
              "bytes": len(raw), "sheet": "Figure 4", "depths": list(range(10)), **series["rf"],
              "matches_committed_rf_reference": True, "series": series, "upstream_notebook": notebook,
              "separate_simulation_benchmark": {
                  "sheet": "Figure 2", "range": "A1:B6", "dataset": "Four-qubit random circuits, FakeLima simulation",
                  "training_circuits": 5000, "test_circuits": 2000, "mean_circuit_l2": simulation_scores,
                  "protocol": "https://arxiv.org/html/2309.17368v2#S2.SS1.SSS1"}}
    path = ROOT / "reports/figure4_reference_check.json"
    if args.check:
        if json.loads(path.read_text()) != report:
            raise ValueError("Committed report disagrees with upstream source artifacts")
    else:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print({name: record["equal_depth_mean_of_printed_values"] for name, record in series.items()})


if __name__ == "__main__":
    main()
