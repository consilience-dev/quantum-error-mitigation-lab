# Quantum error mitigation with pretrained tabular models

Can pretrained regressors improve noisy quantum observable estimates when every method gets the same calibration circuits? This repository reproduces the four-qubit IBM Algiers experiment from ML-QEM, compares estimators on the authors' inputs, and checks whether hardware measurements contribute beyond knowing which circuit was run.

**[Read the results](reports/RESULTS.md)** · [per-run scores](reports/scores.csv) · [paired comparisons](reports/clean_paired_summary.json) · [model coverage](docs/model-coverage.md)

## Two experiments

- **Published reference.** The released hardware notebook uses 500 calibration rows and 2,500 test rows from ten Trotter depths. Each circuit has 169 input features and four ideal target observables. Five new random-forest fits reproduce the published mean circuit error of approximately 0.1000. Running the authors' encoder on the source circuits produced identical bytes for all 3,000 feature and target rows. [Notebook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/demos/demo2_ising_4q_hardware_plot.ipynb) · [Reproduction](reports/rf_reproduction.json) · [Encoder audit](reports/original_encoder_check.json)
- **Unique held-out circuits.** The 300 depth-zero rows represent three distinct circuits. The earlier 18- and 90-circuit budgets intentionally used two or ten circuits at each of nine nonzero depths. The expanded study evaluates 18, 45, 90, 180, 270, 450 and 500. Five nested calibration draws share a 900-circuit holdout. At 500, each depth receives 55 or 56 unique calibration circuits. [Split and identity audit](data/clean_split.json) · [Evaluation protocol](docs/evaluation.md)

The scorecard includes per-observable affine, four-input affine and a linear ridge model on the full 169 inputs. The [few-shot comparison](reports/RESULTS.md#how-far-do-the-few-shot-budgets-go) shows how their ranking changes with the label budget. The original local package and its 180 test IDs could not be recovered. This repository reconstructs the experiments from pinned upstream measurements. All reported scores come from new runs. [Provenance](docs/provenance.md)

## Reproduce the data and baseline

Use Python 3.9 for the upstream Qiskit circuit format. Run from the repository root.

```sh
python -m pip install -r requirements-reproduction.txt
python scripts/fetch_upstream.py
python scripts/fetch_rf_batches.py
python scripts/reproduce_rf.py
python scripts/freeze_clean_split.py
python scripts/run_clean_rf.py
python scripts/extract_measurement_basis.py
python scripts/run_affine_baseline.py
python scripts/run_linear_comparators.py
python scripts/run_kernel_baseline.py
python scripts/run_physics_control.py
```

To independently check the saved features against the authors' encoder, install `torch==2.5.1` into the reproduction environment and run `python scripts/verify_original_encoder.py`. The older numeric-only audit can also be reproduced with `requirements-audit.txt` and `scripts/audit_measurements.py`. Large upstream files and pretrained weights are cached locally. Source paths, revisions and checksums are recorded in the repository.

See the [model settings and execution commands](docs/running-models.md) and [comparisons and external validation design](docs/model-comparison.md). Run the split-planning tests with `python -m unittest discover -s tests -v`.
