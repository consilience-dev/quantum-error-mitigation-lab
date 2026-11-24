# Qermit depolarising simulator, target-specific calibration

This comparison uses all 680 non-mirrored targets in Qermit’s simulated random and Pauli-gadget panels. Each family has 34 width/depth cells with ten target circuits per cell. The local CDR fit and each frozen checkpoint receive the same twenty noisy/ideal surrogate pairs and the same noisy query for a given target. A new fit is made for every target. The published circuit/observable IDs, ideal values, and published raw, ZNE and CDR estimates are carried alongside the newly generated results. [Qermit paper, Figures 11 and 12](https://quantum-journal.org/papers/q-2023-07-13-1059/) · [Archive](https://doi.org/10.5281/zenodo.6472281)

## Circuit generation and shot allocation

We used the released example’s depolarizing model on seven fully connected qubits, with error probabilities 0.001 on the specified single-qubit gates and 0.01 on CX. For each target, the Qermit generator aims to retain ten non-Clifford rotations and exchanges up to three pairs of rotations per surrogate, subject to the source circuit’s available gates. The source’s diversity check retries sets whose exact surrogate expectations all lie within 0.01 of the first. After ten attempts the last set is retained. Distinct surrogate circuit counts can be smaller than twenty, particularly for shallow circuits. Repeated surrogates receive independent shot samples. Qermit’s proximity warning fires if more than 15 of the 20 noisy training values differ from the noisy query by more than 0.1. All twenty rows enter each fit, including flagged cases. [Original CDR generation](https://github.com/Quantinuum/Qermit/blob/b89b20cc236bac25b6fe3e18dcd5234f3a212879/qermit/clifford_noise_characterisation/ccl.py) · [Correction and warnings](https://github.com/Quantinuum/Qermit/blob/b89b20cc236bac25b6fe3e18dcd5234f3a212879/qermit/clifford_noise_characterisation/cdr_post.py) · [Support and warnings](context_diagnostics.json)

Each target uses 23,809 noisy shots on each of the 20 surrogates and on the query (499,989 noisy shots total). The noiseless simulator also supplies 23,809 simulated shots per surrogate. The separate raw estimate receives 100,000 noisy shots. Native relative-error cell summaries divide the absolute mitigated error by the absolute error of this separate raw estimate. The study also publishes ZNE with five allocations of 100,000 shots each. The archived ZNE estimate used a distinct noisy realization. [Resource settings](protocol.json)

The generator verifies the exact all-Z observable for every original target against the archive before scoring. Each finite-shot estimate is obtained by sampling the binary parity distribution of that observable from its statevector or Aer noisy density-matrix expectation. This gives the same finite-shot distribution as sampling complete bitstrings and retaining their parity. Source code, seeds, raw channel means and the twenty sampled labels are available for replay. [All inputs and fits](contexts.npz)

These are fresh simulator measurements. The 2022 archive stores its final method estimates and omits the actual near-Clifford contexts and random seeds. The run uses modern pytket compilation and the current Qermit rebase implementation. Legacy PauliExpBox circuit JSON uses the current Tree synthesis default when its synthesis field is absent. Logical target expectations were verified, but synthesis choices can change the sequence of noisy gates. The archive’s hardware measurements remain distinct from these simulator data.

## Reproduce the contexts

Use Python 3.12 and the versions in the protocol. In a clean environment, install the scientific dependencies and run the following from the project root. The optional work directory caches completed targets; the number of workers changes execution time. The final check needs only NumPy, the committed arrays and the checksum-verified source archive.

```sh
python -m pip install qermit==0.9.3 pytket==2.18.5 pytket-qiskit==0.77.0 qiskit==2.6.2 qiskit-aer==0.18.4 numpy==2.5.5
python scripts/audit_qermit.py /tmp/qermit --download --check
python scripts/run_qermit_cdr_simulator.py /tmp/qermit --workers 6 --work /tmp/qermit-cdr-work
python scripts/run_qermit_cdr_simulator.py /tmp/qermit --check
```
