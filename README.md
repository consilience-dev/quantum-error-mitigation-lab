# Quantum error mitigation lab

Research question: Can pretrained tabular foundation models improve quantum error mitigation accuracy?

This is a new continuation repository created on 2026-11-24. The selected workspace was empty, with no Git history or remote. The previous project package, feature matrix, split IDs, trained models, and experiment outputs were not recovered. Earlier Nori results are conversation reports, not verified results in this repository. Fresh model evaluations use the recovered upstream protocol and explicitly identified splits.

The reported 18 and 90 calibration circuits were deliberately small few-shot budgets: 2 and 10 circuits at each of nine nonzero Trotter depths, after excluding depth-0 duplicates. Each circuit supplies four target observables; those are not four independent calibration circuits. Those budgets omit the published 500-circuit training setting and should be expanded.

See [provenance](docs/provenance.md) and [evaluation protocol](docs/evaluation.md). The upstream measurement files are pinned and checksummed in `data/upstream/manifest.json`; they are not the missing local experiment package.

## Follow-up split planning

Use Python 3.9+; split planning and its tests have no external dependencies.

```
python3 -m unittest discover -s tests -v
python3 scripts/plan_splits.py verified_circuit_manifest.json --output splits.json
```

The input is a JSON list of circuit records with `id`, `circuit_hash`, integer `depth`, and `partition` (`calibration` or `test`). Hashes must represent canonical circuit content, not row IDs or target values. Recover or independently build this manifest before planning real runs. All four observables stay together. The tool rejects duplicate IDs, canonical train/test overlap, and repeated calibration circuits. It preserves the supplied test set and fails if any requested budget is unavailable; it never fills shortages by sampling with replacement.

The proposed nonzero-depth curve is 18, 45, 90, 180, 270, 450, and a conditional 500 circuits, with five split seeds. The published 500-row track includes three repeated logical circuits at depth 0. The [frozen clean split](data/clean_split.json) contains all 35 calibration draws and 900 disjoint test circuits. See the [random-forest reproduction](reports/rf_reproduction.json) and [clean learning curve](reports/rf_clean.json).

## Refit the hardware random forest

In a Python 3.9 environment, install `requirements-reproduction.txt`. Run these commands from the repository root.

```
python scripts/fetch_upstream.py
python scripts/fetch_rf_batches.py
python scripts/reproduce_rf.py
python scripts/freeze_clean_split.py
python scripts/run_clean_rf.py
```

The published split has 500 training and 2,500 test rows. Five fresh forest fits obtained pooled RMSE of 0.07453–0.07461. The notebook saved 0.07471. On the fresh nonzero-depth split the mean circuit error for the full forest at 500 was 0.10996 across five calibration draws. The clean split's [protocol](docs/evaluation.md) and the [model comparison](docs/model-comparison.md) describe the different uses of the two splits.

## Reproduce the upstream numeric audit

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python scripts/fetch_upstream.py
.venv/bin/python scripts/audit_measurements.py
```

The fetch script verifies every file against the committed checksum before use. Large upstream files are downloaded locally and excluded from Git. The audit uses the published row split, not the missing former local test split. It reconstructs 500 training and 2,500 test rows from 3,000 measured circuits, each with four targets. All measurement values are finite. It does not load the circuit objects or establish canonical circuit independence.

The recorded [numeric audit](reports/upstream_measurement_audit.json) gives unmitigated pooled RMSE **0.117135** and ZNE pooled RMSE **0.104507** on the upstream test rows. ZNE's 63 out-of-range observable estimates are retained without clipping. These are baseline calculations, not trained-model results or a replication of earlier Nori runs.
