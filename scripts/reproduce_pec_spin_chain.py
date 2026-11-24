"""Recompute Figures 2–4 of Köylüoğlu et al. from their hardware probability archive."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


URL = "https://zenodo.org/api/records/17291855/files/analysis.zip/content"
MD5 = "2e2347270d203632f015557808c25807"
SHA256 = "a850fd80ea06e8e240d03fe751706f20f318ba086a8c1ed46f4bf5d81bb1d3da"
REPORT = Path(__file__).resolve().parents[1] / "reports/pec_spin_chain/scores.json"


def checked_archive(path, download):
    if download and not path.exists():
        with urlopen(URL, timeout=120) as response:
            payload = response.read()
    else:
        payload = path.read_bytes()
    if (len(payload) != 19_967_731 or hashlib.md5(payload).hexdigest() != MD5
            or hashlib.sha256(payload).hexdigest() != SHA256):
        raise ValueError("Archive differs from Zenodo record 17291855")
    if download and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return ZipFile(io.BytesIO(payload))


def array(archive, member, shape):
    with archive.open("analysis/" + member) as stream:
        if member.endswith(".npy"):
            data = np.load(io.BytesIO(stream.read()), allow_pickle=False)
        else:
            data = np.loadtxt(stream, delimiter=",")
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"Unexpected data in {member}")
    return data


def fit_charge(x, entropies, factor):
    """Return the scaled slope and its standard error from unweighted least squares."""
    coeff, covariance = np.polyfit(x, entropies, 1, cov=True)
    return float(factor * coeff[0]), float(factor * np.sqrt(covariance[0, 0]))


def charge_rows(archive, notebook, estimates):
    # Only read the saved output of the authors' notebook. The scores are fitted independently.
    cells = json.loads(archive.read("analysis/" + notebook))["cells"]
    expected = {}
    for cell in cells:
        for output in cell.get("outputs", []):
            for line in output.get("text", []):
                match = re.fullmatch(r"n=([\d.]+), experiment, c=([\d.e+-]+)±([\d.e+-]+)\s*", line)
                if match:
                    n, charge, error = map(float, match.groups())
                    expected[n] = (charge, error)
    if expected.keys() != estimates.keys():
        raise ValueError(f"Different moments from {notebook}")
    rows = []
    for n, (charge, error) in estimates.items():
        np.testing.assert_allclose((charge, error), expected[n], atol=2e-7, rtol=0)
        rows.append({"n": n, "central_charge": round(charge, 8), "standard_error": round(error, 8),
                     "notebook_central_charge": expected[n][0], "notebook_standard_error": expected[n][1]})
    return rows


def open_chain(archive):
    distributions = {name: array(archive, f"fig2/N_10_TIF_open_{name}.csv", (1024,))
                     for name in ("exact", "raw", "mitigated")}
    for p in distributions.values():
        np.testing.assert_allclose(p.sum(), 1, rtol=0, atol=1e-10)
    reference = distributions["exact"]
    if np.min(reference) < 0 or np.min(distributions["raw"]) < -1e-12:
        raise ValueError("Invalid exact or raw probabilities")
    amplitudes = {}
    fidelities = {}
    for name in ("raw", "mitigated"):
        # Figure 2's archived notebook clips all negative values to zero before normalization.
        p = np.maximum(distributions[name], 0.)
        amplitudes[name] = np.sqrt(p / p.sum())
        fidelities[name] = float(np.dot(amplitudes[name], np.sqrt(reference)) ** 2)
    if round(fidelities["mitigated"], 2) != 0.96:
        raise ValueError("Figure 2C reconstructed fidelity differs from the printed 0.96")

    # Experimental regression uses the first five x values and the reversed entropy curve.
    # Qiskit traces out high-index qubits. The singular values of the bipartite amplitude
    # matrix are the Schmidt coefficients, so their squares give the retained density spectrum.
    lengths = np.arange(1, 6)
    x = np.log((20 / np.pi) * np.sin(np.pi * lengths / 10))
    spectra = []
    for length in lengths:
        matrix = amplitudes["mitigated"].reshape(2**int(length), -1)
        weights = np.linalg.svd(matrix, compute_uv=False) ** 2
        spectra.append(weights[weights > 0])
    estimates = {}
    for n in range(1, 11):
        if n == 1:
            entropies = [-np.dot(w, np.log(w)) for w in spectra]
        else:
            entropies = [np.log(np.sum(w**n)) / (1 - n) for w in spectra]
        estimates[n] = fit_charge(x, entropies, 12 * n / (n + 1))
    return {"reconstructed_fidelity": {k: round(v, 8) for k, v in fidelities.items()},
            "negative_mitigated_entries_before_clipping": int((distributions["mitigated"] < 0).sum()),
            "central_charge": charge_rows(archive, "fig2/Fit_with_error_propagation_TFIM_L=10_OBC.ipynb", estimates)}


def classical_entropy(probabilities, n):
    return np.log(np.sum(probabilities**n, axis=-1)) / (1 - n)


def periodic_chain(archive, folder, model, basis, filename, moments):
    boot = array(archive, f"{folder}/{filename}", (20, 4096))
    occupancy = np.array([bin(i).count("1") for i in range(4096)])
    allowed = occupancy == 6 if basis == "z" else occupancy % 2 == 0
    cutoff = np.sqrt(np.mean(np.min(boot[:, allowed], axis=1) ** 2))
    probabilities = np.where(allowed & (boot >= cutoff), boot, 0.)
    normalizers = probabilities.sum(axis=1, keepdims=True)
    if np.any(normalizers <= 0):
        raise ValueError("Empty symmetry-filtered distribution")
    probabilities /= normalizers
    lengths = (2, 4, 6, 8, 10)
    x = np.log((12 / np.pi) * np.sin(np.pi * np.array(lengths) / 12))
    estimates = {}
    for n in moments:
        # The authors take both marginals from leading blocks of size l and L-l, respectively.
        # This choice matters when the experimental state lacks exact translation invariance.
        full_entropy = classical_entropy(probabilities, n)
        differences = []
        for length in lengths:
            left = probabilities.reshape(20, 2**length, -1).sum(axis=2)
            complement = probabilities.reshape(20, 2**(12-length), -1).sum(axis=2)
            differences.append(classical_entropy(left, n) + classical_entropy(complement, n) - full_entropy)
        fits = [fit_charge(x, curve, 4 * (n - 1) / n) for curve in np.array(differences).T]
        centres, fit_errors = np.transpose(fits)
        # The deposit combines between-bootstrap variance with mean squared fit uncertainty.
        estimates[n] = (np.mean(centres), np.sqrt(np.var(centres, ddof=1) + np.mean(fit_errors**2)))
    notebook = f"{folder}/Fit_with_error_propagation_{model}_L=12_PBC_{basis}.ipynb"
    return {"bootstrap_replicates": len(boot), "cutoff": round(float(cutoff), 8),
            "central_charge": charge_rows(archive, notebook, estimates)}


def reproduce(archive):
    return {
        "paper_doi": "10.1038/s41467-025-66775-9", "release_doi": "10.5281/zenodo.17291855",
        "archive_md5": MD5, "archive_sha256": SHA256,
        "figure2_TFI_open_z": open_chain(archive),
        "figure3_TFI_periodic_x": periodic_chain(archive, "fig3", "TFIM", "x",
                                                "TIF_12_periodic_x_bootstrapped_bitstrings.npy", (2, 4, 6)),
        "figure4_XXZ_periodic_x": periodic_chain(archive, "fig4", "XXZ", "x",
                                                "XXZ_12_periodic_x_bootstrapped_bitstrings.csv", (4, 4.5, 6)),
        "figure4_XXZ_periodic_z": periodic_chain(archive, "fig4", "XXZ", "z",
                                                "XXZ_12_periodic_z_bootstrapped_bitstrings.csv", (4, 4.5, 6)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Local analysis.zip path")
    parser.add_argument("--download", action="store_true", help="Fetch the verified release if absent")
    parser.add_argument("--check", action="store_true", help="Verify committed scores")
    args = parser.parse_args()
    with checked_archive(args.archive, args.download) as source:
        report = reproduce(source)
    if args.check:
        if report != json.loads(REPORT.read_text()):
            raise ValueError("Saved report differs from recomputed scores")
    else:
        REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Verified Figure 2C fidelity and 19 central-charge estimates and uncertainties against the source.")
