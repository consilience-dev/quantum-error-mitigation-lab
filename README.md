# Quantum error mitigation with pretrained tabular models

Can pretrained regressors improve noisy quantum observable estimates when every method gets the same calibration circuits? This repository reproduces the four-qubit IBM Algiers experiment from ML-QEM, compares estimators on the authors' inputs, and checks whether hardware measurements contribute beyond knowing which circuit was run.

**[Read the results](reports/RESULTS.md)** · [per-run scores](reports/scores.csv) · [paired comparisons](reports/clean_paired_summary.json) · [model coverage](docs/model-coverage.md) · [QEM method coverage and next benchmarks](docs/mitigation-benchmarks.md)

## Hardware and simulator benchmarks

- **Published reference.** The released hardware notebook uses 500 calibration rows and 2,500 test rows from ten Trotter depths. Each circuit has 169 input features and four ideal target observables. Five new random-forest fits reproduce the published mean circuit error of approximately 0.1000. Running the authors' encoder on the source circuits produced identical bytes for all 3,000 feature and target rows. [Notebook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/demos/demo2_ising_4q_hardware_plot.ipynb) · [Reproduction](reports/rf_reproduction.json) · [Encoder audit](reports/original_encoder_check.json)
- **Unique held-out circuits.** The 300 depth-zero rows represent three distinct circuits. The earlier 18- and 90-circuit budgets intentionally used two or ten circuits at each of nine nonzero depths. The expanded study evaluates 18, 45, 90, 180, 270, 450 and 500. Five nested calibration draws share a 900-circuit holdout. At 500, each depth receives 55 or 56 unique calibration circuits. [Split and identity audit](data/clean_split.json) · [Evaluation protocol](docs/evaluation.md)
- **Published simulator comparisons.** On the authors’ four-qubit FakeLima OLS tutorial, the released OLS model reproduced their 0.07411 validation RMSE. Frozen TabPFN v2 scored 0.01676 on the same 1,500 rows. On 4,500 held-out Figure 3 rows at training depths, TabPFN v2 achieved 0.01186 MAE and the authors’ forest achieved 0.01551 MAE. At unseen depths 15–29, measurement-only TabPFN beat both affine controls but trailed the authors’ circuit-aware forest. [Depth transfer](reports/RESULTS.md#do-measurement-corrections-transfer-to-deeper-circuits) · [Figure 3 comparison](reports/RESULTS.md#comparison-with-the-published-figure-3-predictions) · [Tutorial comparison](reports/RESULTS.md#comparison-with-the-authors-simulated-ols-tutorial)

The scorecard includes per-observable affine, four-input affine and a linear ridge model on the full 169 inputs. The [few-shot comparison](reports/RESULTS.md#how-far-do-the-few-shot-budgets-go) shows how their ranking changes with the label budget. The original local package and its 180 test IDs could not be recovered. This repository reconstructs the experiments from pinned upstream measurements. The reports distinguish new runs from scores printed by the authors and scores recovered from archived predictions. [Provenance](docs/provenance.md)

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
