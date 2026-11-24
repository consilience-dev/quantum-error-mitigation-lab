# Provenance and recovery status

## Local evidence

On 2026-11-24, `/Users/parm/Documents/ChatGPT/quantum-error-mitigation-lab` contained no files, `.git`, or remote. Searches of local Documents, Desktop, Downloads, and user paths did not locate a QEM project package. No predecessor commits are claimed.

Read-only conversation evidence: “Quantum Error Mitigation Data Sets”, conversation `6b049964-db78-83ea-91d2-70a8ee895a7c`. It reports IBM ML-QEM 4-qubit hardware data, 169 features, four targets, 180 test circuits, and calibration budgets 18/90. Those local data-processing and result claims remain unverified. “Identify Experiment Dataset” also records the absence of the experiment repository in its workspace. Conversation text is not a substitute for saved arrays, split IDs, or executable code.

## Independently retrieved upstream evidence

- Research repository: https://github.com/qiskit-community/ml-qem
- Pinned research commit: `faf3e44ac686d62b2184a983d89e655084759a26`
- Protocol notebook: `docs/demos/demo2_ising_4q_hardware_plot.ipynb`
- Hardware results: `docs/tutorials/data/ising_init_from_qasm_hardware/shuffled/results.pk`
- Shuffling map: `docs/tutorials/data/ising_init_from_qasm_hardware/shuffled/index_order.json`
- Feature encoder: `docs/tutorials/mlp.py`
- Archival reference: https://zenodo.org/records/13769804 (DOI 10.5281/zenodo.13769804). Its 659.7 MB archive is a software snapshot; we did not download or verify that archive against the GitHub files.
- Paper: https://arxiv.org/abs/2309.17368

The downloaded notebook unshuffles measurements using `index_order`, combines 200+100 circuits at each step 0–9, then selects the first 50 per step for training and last 250 for testing. Thus it explicitly specifies 500 training and 2,500 testing rows. It encodes four observables and fits four 100-tree random forests. The notebook alone does not establish canonical circuit disjointness or validate the former local 169-feature extraction.

Upstream files retain their upstream license; ML-QEM identifies Apache-2.0. Record original paths, URLs, byte counts, and SHA-256 in the retrieval manifest. Do not modify upstream files in place. No secrets, user conversation exports, or model credentials are committed.

## Executed measurement checks

`scripts/audit_measurements.py` verified all four retrieved file checksums, the 3,000-element shuffle permutation, expected array dimensions, finite measurements, and ideal/noisy observable bounds (allowing floating-point roundoff). It reconstructed the notebook's row ordering and 50/250 split per depth. The committed JSON report records baseline errors, NumPy version, and the measurement checksum. ZNE estimates outside [-1, 1] are counted and retained. These checks do not establish canonical circuit uniqueness, recover the old local test split, or run a learned model.

## Circuit, encoder and figure checks

Subsequent code validates 60 hardware batches and 20 logical source files by immutable Git blob IDs. It checks circuit operations against the source order and audits duplicates under both compiled and logical identities. At depths 1–9, all 2,700 rows survive deduplication. Three identities account for all 300 rows at depth 0. The original train/test split shares those three identities. [Source manifests](../data/rf_logical_sources.json) · [Split audit](../data/clean_split.json)

We independently ran the unmodified upstream encoder functions from the checksum-verified source on all recovered circuits, using the original measurement arrays. All 507,000 float32 feature entries and 12,000 target entries match byte-for-byte. The Figure 4 workbook also agrees with the ten stored reference values at the precision published. [Encoder check](../reports/original_encoder_check.json) · [Workbook check](../reports/figure4_reference_check.json)
