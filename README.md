# Quantum error mitigation lab

Research question: Can pretrained tabular foundation models improve quantum error mitigation accuracy?

This is a new continuation repository created on 2026-11-24. The selected workspace was empty, with no Git history or remote. The previous project package, feature matrix, split IDs, trained models, and experiment outputs were not recovered. Earlier Nori results are conversation reports, not verified results in this repository. No new model evaluations have been run.

The reported 18 and 90 calibration circuits were deliberately small few-shot budgets: 2 and 10 circuits at each of nine nonzero Trotter depths, after excluding depth-0 duplicates. Each circuit supplies four target observables; those are not four independent calibration circuits. Those budgets omit the published 500-circuit training setting and should be expanded.

See [provenance](docs/provenance.md) and [evaluation protocol](docs/evaluation.md). The upstream measurement files are pinned and checksummed in `data/upstream/manifest.json`; they are not the missing local experiment package.

## Follow-up split planning

Use Python 3.9+; split planning and its tests have no external dependencies.

```
python3 -m unittest discover -s tests -v
python3 scripts/plan_splits.py verified_circuit_manifest.json --output splits.json
```

The input is a JSON list of circuit records with `id`, `circuit_hash`, integer `depth`, and `partition` (`calibration` or `test`). Hashes must represent canonical circuit content, not row IDs or target values. Recover or independently build this manifest before planning real runs. All four observables stay together. The tool rejects duplicate IDs, canonical train/test overlap, and repeated calibration circuits. It preserves the supplied test set and fails if any requested budget is unavailable; it never fills shortages by sampling with replacement.

The proposed nonzero-depth curve is 18, 45, 90, 180, 270, 450, and a conditional 500 circuits, with five split seeds. A separate published-protocol 500-circuit track includes depth 0 and needs a duplicate audit. Neither track is a completed evaluation.

## Reproduce the upstream numeric audit

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python scripts/fetch_upstream.py
.venv/bin/python scripts/audit_measurements.py
```

The fetch script verifies every file against the committed checksum before use. Large upstream files are downloaded locally and excluded from Git. The audit uses the published row split, not the missing former local test split. It reconstructs 500 training and 2,500 test rows from 3,000 measured circuits, each with four targets. All measurement values are finite. It does not load the circuit objects or establish canonical circuit independence.

The recorded [numeric audit](reports/upstream_measurement_audit.json) gives unmitigated pooled RMSE **0.117135** and ZNE pooled RMSE **0.104507** on the upstream test rows. ZNE's 63 out-of-range observable estimates are retained without clipping. These are baseline calculations, not trained-model results or a replication of earlier Nori runs.
