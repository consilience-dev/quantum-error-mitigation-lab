# Qermit CDR with February 2022 software

The original depolarising simulator CDR records were acquired February 3–6, 2022, following the January 11–15 raw and ZNE runs. The February Qermit source implements each correction through its own 20 near-Clifford calibration circuits and one noisy target query. We ran that source directly on the released circuit and observable IDs. Its generator and fitter files are verified by SHA-256. The authors' April example supplies the seven-qubit noise model and shot settings. [Original archive](https://zenodo.org/records/6472281) · [Acquisition-period Qermit](https://github.com/Quantinuum/Qermit/tree/aa25c21a9a17aa5b46d03c4152d2d61387356d21) · [Original figure definitions](https://quantum-journal.org/papers/q-2023-07-13-1059/)

The original environment lock, individual calibration circuits and random seeds were unavailable. The tested pytket/pytket-qiskit dependency series changed from 0.17/0.20 to 0.19/0.22 during CDR acquisition. Our replay uses pytket 0.19.2 and pytket-qiskit 0.22.1 from the latter series. These patch versions were released on February 18 and February 9, respectively. Qiskit 0.34.2 and Aer 0.10.3 were released February 9. [Dependency transition](https://github.com/Quantinuum/Qermit/commit/aa25c21a9a17aa5b46d03c4152d2d61387356d21) · [pytket release](https://pypi.org/project/pytket/0.19.2/) · [Adapter release](https://pypi.org/project/pytket-qiskit/0.22.1/) · [Aer release](https://pypi.org/project/qiskit-aer/0.10.3/)

## Pilot and full cohort

Before choosing whether to extend the historical comparison, we fixed one complete 4-qubit, depth-4 cell from each circuit family. The pilot used the original per-shot bitstring return from Aer, and Qermit computed the measured expectations from its parity counts. The published CDR mean absolute errors were 0.04495 on Random and 0.01595 on Pauli-gadget. The February replay yielded 0.04415 and 0.01098 on the same ten targets per family. The modern replay yielded 0.11464 and 0.01291. [Pilot data and audit](pilot_width4_depth4/native_graph_audit.json)

With the same final generation seed and target, switching the Qermit/pytket generator changed the noiseless auxiliary expectations by 0.23063 mean absolute difference on Random and below 2e-16 on Pauli-gadget. This comparison covers the 200 auxiliary values in each pilot cell, paired in their generation order. [Individual controlled results](same_seed_width4_depth4.json)

We reran both pilot cells with pytket 0.19.0 and pytket-qiskit 0.22.0, released before acquisition on January 31 and February 1. Holding the Qermit source, Aer version and seeds fixed, all 20 targets had identical generated surrogate hashes, ideal/channel means, finite-shot measurements and CDR estimates across the two compiler/adapter pairs. This controls those two patch changes on these cells. [Patch replay and audit](compiler_patch_pilot/native_graph_audit.json) · [Earlier pytket release](https://pypi.org/project/pytket/0.19.0/) · [Earlier adapter release](https://pypi.org/project/pytket-qiskit/0.22.0/)

The follow-up used all 680 non-mirrored targets from the Random and Pauli-gadget panels, including the 20 used for the pilot. Both arms use the native Aer sampling engine. The full cohort retains per-circuit count histograms, and the pilot retains all returned bitstrings. All finite-shot auxiliary measurements, target queries and raw estimates agreed exactly on the 20 overlapping targets. On all 340 Random targets the February-source CDR MAE was 0.05348, compared with 0.06001 in the archive and 0.11670 with the modern implementation. On 340 Pauli-gadget targets these values were 0.01546, 0.01218 and 0.01218. [Complete source and calibration audit](full_unmirrored/native_graph_audit.json)

The original task graph runs its diversity check and up to ten attempts using exact simulator values. It then obtains two independent 23,809-shot estimates for each surrogate, one noisy and one noiseless. A further 23,809-shot noisy query brings the noisy CDR budget to 499,989 shots. The standalone raw denominator receives 100,000 independent shots with the original level-1 compiler. CDR uses the original level-0 compiler. This implementation observes the fitted data, warning flags and final correction. It additionally computes noisy channel means for the actual compiled circuits for diagnostics. The published ZNE and CDR columns belong to the archived acquisitions. [Original graph and compilation](https://github.com/Quantinuum/Qermit/blob/aa25c21a9a17aa5b46d03c4152d2d61387356d21/qermit/clifford_noise_characterisation/ccl.py)

## Reproduction

The replay used Python 3.9.6, NumPy 1.26.4, Qiskit 0.34.2/Terra 0.19.2/Aer 0.10.3, pytket 0.19.2 and pytket-qiskit 0.22.1. Use a Python 3.9 environment and install Qermit from the February commit. From the repository root run the following.

```sh
git clone https://github.com/Quantinuum/Qermit.git /tmp/qermit-feb-source
git -C /tmp/qermit-feb-source checkout --detach aa25c21a9a17aa5b46d03c4152d2d61387356d21
python -m pip install numpy==1.26.4 pytket==0.19.2 pytket-qiskit==0.22.1 qiskit==0.34.2 /tmp/qermit-feb-source
python scripts/audit_qermit.py /tmp/qermit-data --download --check
python scripts/run_qermit_acquisition_cdr.py /tmp/qermit-data --workers 5 --work /tmp/qermit-cdr-cache --out reports/qermit_cdr_simulation/acquisition_era/full_unmirrored
python scripts/verify_qermit_acquisition.py /tmp/qermit-data reports/qermit_cdr_simulation/acquisition_era/full_unmirrored --pilot reports/qermit_cdr_simulation/acquisition_era/pilot_width4_depth4
```

For the pilot, pass `--widths 4 --depths 4 --shot-memory` and a separate output directory. For the compiler patch check, install pytket 0.19.0 and pytket-qiskit 0.22.0 in a separate environment with the other versions fixed. Run the same command with widths and depths set to 4 and the default histogram mode. For the controlled generator comparison, run `scripts/audit_qermit_surrogate_versions.py` with the archive directory and a JSON destination supplied through `--out`. The audit of saved results uses only the current project environment and the verified data archive.

The legacy tweedledum 1.1.0 wheel installed on our Apple Silicon host contained an Intel-only extension. We resolved this by building release 1.1.1 from its official source in the temporary Python 3.9 environment. Our build used setuptools 58.0.4, wheel 0.38.4, scikit-build 0.17.6, CMake 3.31.10 and Ninja 1.13.2. Installing these tools first, then using `python -m pip install --no-build-isolation --no-binary=tweedledum tweedledum==1.1.1` builds a native extension. [Source release](https://pypi.org/project/tweedledum/1.1.1/)
