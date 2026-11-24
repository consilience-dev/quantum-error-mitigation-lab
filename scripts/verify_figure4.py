"""Verify the reported RF reference against the pinned workbook, using the standard library."""
import argparse
import hashlib
import io
import json
import posixpath
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "faf3e44ac686d62b2184a983d89e655084759a26"
SOURCE = "https://raw.githubusercontent.com/qiskit-community/ml-qem/" + REVISION + "/docs/paper_figures/ML-QEM%20Source%20data.xlsx"
DIGEST = "b42607140bfc732c73957a81f35a0bd4c0a21f78b9713a4b9207a107181eb71f"
GIT_BLOB = "d2dafa3c421b8b3e4c8d6c1e9c18919a1cda715e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, help="Optional local copy of the pinned upstream workbook")
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
        sheet = ET.fromstring(archive.read("xl/workbook.xml")).find(".//{*}sheet[@name='Figure 4']")
        relation = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = rels.find(f"{{*}}Relationship[@Id='{relation}']").attrib["Target"]
        strings = ["".join(cell.itertext()) for cell in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        xml = ET.fromstring(archive.read(posixpath.join("xl", target)))
        cells = {c.attrib["r"]: strings[int(c.findtext("{*}v"))] for c in xml.findall(".//{*}c[@t='s']")}
    if cells["A13"] != "step" or "L2_rfr_list" not in cells["A24"]:
        raise ValueError("Unexpected Figure 4 labels")
    values = []
    for depth in range(10):
        step, printed = cells[f"A{depth+14}"].split()
        if int(step) != depth:
            raise ValueError("Unexpected figure depth order")
        values.append(float(printed))
    comparison = json.loads((ROOT / "reports/rf_reproduction.json").read_text())
    if comparison["reference_figure4_mean_circuit_l2_per_depth_6dp"] != values:
        raise ValueError("Workbook disagrees with RF reference used for reproduction")
    report = {"upstream_revision": REVISION, "source": SOURCE, "sha256": DIGEST, "git_blob_sha1": blob,
              "bytes": len(raw), "sheet": "Figure 4", "header": "A13", "range": "A14:A23",
              "series_label": cells["A24"], "series_label_cell": "A24",
              "depths": list(range(10)), "mean_circuit_l2_per_depth_6dp": values,
              "equal_depth_mean_of_printed_values": sum(values)/len(values),
              "matches_committed_rf_reference": True}
    (ROOT / "reports/figure4_reference_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["equal_depth_mean_of_printed_values"])


if __name__ == "__main__":
    main()
