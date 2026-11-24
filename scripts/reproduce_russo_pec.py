"""Recreate the hardware PEC points of Russo et al., IEEE TQE Figures 3–4."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import tarfile
import tempfile
import zlib
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pec_russo"
REVISION = "e0457555978bf3125b3338e48f93dbfc50efd9fd"
SOURCE = {
    "data.tar.gz": (
        f"https://codeload.github.com/unitaryfoundation/research/tar.gz/{REVISION}",
        "431d33029d1689f1dc26c751fd88010dbc0c80dd29cca299ab3b7c214e820387",
    ),
    "figures.tar": (
        "https://arxiv.org/src/2210.07194v2",
        "02355c96586fe071e709fad2f6fbaf568d7465157a394c4a93dc8b546e1262c1",
    ),
}
DEVICES = {"ibmq": "IBMQ Lima", "ionq": "IonQ Harmony", "rigetti": "Rigetti Aspen-M2"}
# Matplotlib hardware colors and physical axis limits in the pinned source PDFs.
COLORS = {(31, 119, 180): "ibmq", (44, 160, 44): "ionq", (255, 127, 14): "rigetti"}
FIELDS = ("noisy_values", "pec_values", "true_values", "cnot_counts", "oneq_counts")
ROW = ("figure", "device", "qubits", "family", "depth", "instance", "source_row",
       "one_qubit_gate_count", "cnot_count", "ideal", "raw", "pec")
SCORE = ("figure", "device", "qubits", "family", "depth", "circuits", "raw_rmse",
         "pec_rmse", "improvement_factor", "published_figure_factor")


def ensure_archive(folder, name, download):
    url, sha = SOURCE[name]
    path = folder / name
    if not path.exists() and download:
        folder.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError(f"Unexpected downloaded archive: {name}")
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as f:
            f.write(content)
        Path(f.name).replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f"Archive checksum mismatch: {name}")
    return path


def page_stream(pdf):
    # These immutable one-page Matplotlib figures have one Flate-compressed
    # content stream. Read only that PDF object, whose coordinate pairs are
    # the points actually plotted. Source bounds and marker counts are checked.
    ref = re.search(rb"/Contents\s+(\d+) 0 R", pdf).group(1)
    content = re.search(rb"(?:\n|^)" + ref + rb" 0 obj\s*<<([^>]*)>>\s*stream\r?\n"
                        rb"(.*?)\r?\nendstream", pdf, flags=re.S)
    if b"/FlateDecode" not in content.group(1):
        raise ValueError("Unexpected figure encoding")
    return zlib.decompress(content.group(2)).decode("ascii", errors="strict")


def plotted_points(source):
    points = {}
    with tarfile.open(source) as archive:
        for width, figure in ((3, 3), (5, 4)):
            page = page_stream(archive.extractfile(f"images/v2/n{width}.pdf").read())
            # Right column = PEC. Solid /A3 lines = hardware. Dashed lines
            # are simulated devices. The upper row is RB, the lower mirror.
            blocks = re.finditer(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re W n /A3 gs "
                                 r"(.*?)\n\nS\n", page, flags=re.S)
            for block in blocks:
                x0, y0, w, h = map(float, block.groups()[:4])
                if x0 < 800:
                    continue
                path = block[5]
                rgb = re.search(r"([\d.]+)\s+([\d.]+)\s+([\d.]+) RG", path)
                color = tuple(round(255 * float(c)) for c in rgb.groups())
                device = COLORS[color]
                family = "rb" if y0 > 200 else "mirror"
                ymin, ymax = (-1, 10) if width == 3 else (0, 8)
                for px, py in re.findall(r"^([\d.]+) ([\d.]+) [ml]$", path, flags=re.M):
                    x = 0.6 + 8.8 * (float(px) - x0) / w
                    if abs(x - round(x)) > 1e-6:
                        raise ValueError("Figure marker is not at an integer depth")
                    key = (figure, device, width, family, round(x))
                    if key in points:
                        raise ValueError(f"Duplicate reference marker {key}")
                    points[key] = ymin + (ymax - ymin) * (float(py) - y0) / h
    if len(points) != 25:
        raise ValueError(f"Unexpected number of hardware PEC figure markers: {len(points)}")
    return points


def hardware_tables(source):
    base = f"research-{REVISION}/qem-on-hardware/data/hardware/pec/"
    by_run = defaultdict(dict)
    with tarfile.open(source) as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(base) or not member.name.endswith(".txt"):
                continue
            family, device, group, filename = member.name[len(base):].split("/")
            parts = group.split("_")
            if parts[:3] != [device, "pec", family] or parts[-2:] != ["10000", "4"]:
                raise ValueError(f"Unexpected experiment name: {group}")
            field = next(f for f in (*FIELDS, "noise_scaled_expectation_values") if filename.startswith(f + "_"))
            stamp = filename[len(field)+1:-4]
            by_run[(device, int(parts[3]), family, stamp)][field] = archive.extractfile(member).read()
    groups, duplicates = {}, []
    for (device, width, family, stamp), fields in sorted(by_run.items()):
        if set(fields) != {*FIELDS, "noise_scaled_expectation_values"} or fields["noise_scaled_expectation_values"].strip():
            raise ValueError("Unexpected hardware PEC measurements or score files")
        group = (device, width, family)
        if group in groups:
            first_stamp, original = groups[group]
            if fields != original:
                raise ValueError("Second nonidentical export requires an explicit source mapping")
            duplicates.append(dict(device=DEVICES[device], family=family, qubits=width,
                                   retained=first_stamp, byte_identical_export=stamp))
        else:
            groups[group] = (stamp, fields)
    tables = {}
    for group, (_, raw) in groups.items():
        fields = {k: [list(map(float, row.split())) for row in raw[k].decode().splitlines()] for k in FIELDS}
        n = len(fields["true_values"])
        if any(len(matrix) != n or any(len(row) != 4 for row in matrix) for matrix in fields.values()):
            raise ValueError("Unequal matrices or missing circuit instances")
        if any(value != 1 for row in fields["true_values"] for value in row):
            raise ValueError("Ideal projector no longer constant")
        if any(not math.isfinite(v) for matrix in fields.values() for row in matrix for v in row):
            raise ValueError("Missing circuit values")
        tables[group] = fields
    return tables, duplicates


def encode(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def score(rows):
    def rmse(field):
        return math.sqrt(math.fsum((r[field] - r["ideal"])**2 for r in rows) / len(rows))
    raw, pec = rmse("raw"), rmse("pec")
    return raw, pec, raw / pec


def reproduce(data_archive, figure_archive):
    points = plotted_points(figure_archive)
    tables, duplicates = hardware_tables(data_archive)
    conditions = defaultdict(list)
    for (figure, device, width, family, depth), factor in sorted(points.items()):
        conditions[(device, width, family)].append((figure, depth, factor))
    if set(conditions) != set(tables):
        raise ValueError("Archive and source figure panels differ")
    rows, scores, checked, groups = [], [], [], []
    for (device, width, family), entries in sorted(conditions.items()):
        matrices = tables[device, width, family]
        # Each plotted series starts at the first archive row. Verify this
        # alignment against each marker, particularly in the long IBM runs.
        groups.append(dict(device=DEVICES[device], qubits=width, family=family,
                           archived_depth_rows=len(matrices["true_values"]), plotted_depths=[e[1] for e in entries]))
        for i, (figure, depth, published) in enumerate(entries):
            common = dict(figure=figure, device=DEVICES[device], qubits=width, family=family, depth=depth)
            cell = []
            for j in range(4):
                obs = {k: matrices[k][i][j] for k in FIELDS}
                cell.append(dict(**common, instance=j+1, source_row=i+1,
                                 one_qubit_gate_count=int(obs["oneq_counts"]), cnot_count=int(obs["cnot_counts"]),
                                 ideal=obs["true_values"], raw=obs["noisy_values"], pec=obs["pec_values"]))
            raw, pec, factor = score(cell)
            delta = abs(published - factor)
            if delta > 2e-7:
                raise ValueError(f"Paper and archive disagree for {common}: {factor} vs {published}")
            rows.extend(cell)
            checked.append(delta)
            scores.append(dict(**common, circuits=4, raw_rmse=f"{raw:.12g}", pec_rmse=f"{pec:.12g}",
                               improvement_factor=f"{factor:.12g}", published_figure_factor=f"{published:.12g}"))
    audit = dict(paper="https://doi.org/10.1109/TQE.2023.3305232", repository_revision=REVISION,
                 sources={k: dict(url=url, sha256=sha) for k, (url,sha) in SOURCE.items()},
                 experiment="One trial per circuit, four circuits at each plotted depth, one projector each. "
                            "Equation 6 is the square root of the ratio of summed squared errors, "
                            "with equal shot counts in both arms.", raw_shots_per_circuit=10000,
                 pec_sampled_variants_per_circuit=100, shots_per_pec_variant=100,
                 source_figures=[3, 4], archive_nonduplicate_circuit_observations=sum(4*e["archived_depth_rows"] for e in groups),
                 plotted_circuit_observations=len(rows), plotted_points=len(scores),
                 factors_above_one=sum(float(s["improvement_factor"])>1 for s in scores),
                 max_absolute_source_figure_difference=round(max(checked),10), tolerance=2e-7,
                 native_supervised_training_rows=0,
                 ideal_target_values=sorted({r["ideal"] for r in rows}), groups=groups, duplicate_exports=duplicates)
    return {"published_circuit_rows.csv": encode(rows, ROW), "depth_scores.csv": encode(scores, SCORE),
            "audit.json": json.dumps(audit, indent=2, allow_nan=False) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Directory containing the pinned data.tar.gz and figures.tar")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check committed outputs instead of writing")
    args = parser.parse_args()
    data = ensure_archive(args.cache, "data.tar.gz", args.download)
    figures = ensure_archive(args.cache, "figures.tar", args.download)
    files = reproduce(data, figures)
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        if args.check:
            if (OUT / name).read_text() != expected:
                raise ValueError(f"Committed result differs: {name}")
        else:
            (OUT / name).write_text(expected)
    print("Verified all 25 published hardware PEC factors against 100 paired circuit observations.")


if __name__ == "__main__":
    main()
