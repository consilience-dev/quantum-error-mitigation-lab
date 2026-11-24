# Reproducing the model evaluations

Build the numeric input archive and frozen split using the [root instructions](../README.md). Model environments use Python 3.12. Each result JSON records the checkpoint, package versions, settings and split. The saved prediction files identify the original unshuffled hardware row. The device used in a reported run can affect floating-point results.

## Nori 1.5 and TabICLv2

Install `synthefy-nori==0.22.0`, `tabicl[finetune]==2.2.0` and `torch==2.13.0` in a dedicated environment. Download the `checkpoint_url` in [Nori](../reports/nori_published.json) or [TabICL](../reports/tabicl_published.json) into a local weights directory using its upstream filename. The runner checks its SHA-256 before loading it. For the published runs use the prediction chunk size recorded in each result JSON. The examples use Apple GPU inference.

```sh
python scripts/run_nori_tabicl.py nori --weights weights --device mps --chunk 250
python scripts/run_nori_tabicl.py tabicl --weights weights --device mps --chunk 250
python scripts/run_nori_tabicl.py nori --weights weights --device mps --clean-seed 0 --chunk 450
python scripts/run_nori_tabicl.py tabicl --weights weights --device mps --clean-seed 0 --chunk 450 --finetune
```

Use seeds 0 through 4 for the frozen clean runs. Controls take `--inputs descriptors_only` or `--inputs measurements_only`. Nori uses four ensemble members and TabICLv2 uses eight. The clean Nori configuration fits its high-dimensional feature selection on the training inputs. TabICLv2 fine-tuning logs the selected checkpoint's parameter change separately for each observable and allows selection of the starting checkpoint using calibration-only validation.

## TabPFN and TabDPT

Use `torch==2.15.0`, `tabpfn==6.4.1` and `tabdpt==1.3.2` for TabPFN v2 and TabDPT 1.3. For the supplemental TabPFN releases, use a separate environment with the versions listed in the corresponding JSON. Retrieve the pinned checkpoint URL in the result JSON into the local weights directory. The runner verifies the file hash.

```sh
python scripts/run_tabpfn_tabdpt.py tabpfn_v2 weights --device mps
python scripts/run_tabpfn_tabdpt.py tabpfn_v2 weights --device mps --clean-seed 0 --clean-budget 18
python scripts/run_tabpfn_tabdpt.py tabpfn_v2 weights --device mps --clean-seed 0 --fine-tune
python scripts/run_tabpfn_tabdpt.py tabdpt_v1_3 weights --device mps --clean-seed 0
```

Each run uses one ensemble member. Use `--inputs descriptors`, `noisy`, or `shuffle_train_noisy` for frozen calibration controls. For the permutation control, first run `scripts/extract_measurement_basis.py` in the Qiskit environment. The four noisy observables move together within training depth and basis. Test measurements stay paired with their original circuits.

## Additional models and conventional baselines

The [registry snapshot](../configs/model_inventory.json) pins TabTune source revision `5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e` (0.2.2). Use this version with `torch==2.15.0`. The runners download or verify the pinned weights. Registry discovery and model execution are recorded separately. [Checkpoint manifest](../data/extra_model_sources.json)

```sh
python scripts/run_extra_frozen.py causilo --batch-size 500
python scripts/run_extra_frozen.py causilo --split clean --seed 0 --batch-size 300
python scripts/run_extra_frozen.py tababax --batch-size 500
python scripts/run_registry_regression.py KumoTabular --device cpu
python scripts/run_registry_regression.py ILTM --device cpu
```

See `query_batch_size` in the saved result for the exact chunking used. Install `catboost==1.2.10` to run `scripts/run_boosted_baseline.py`. Kernel ridge uses five calibration folds grouped by circuit identity. The Ising interpolation control uses a separate input representation with the problem's known coupling, measurement basis and depth.

## Rebuild the scorecard

After running the comparisons, use Python 3.12 with NumPy to run `python scripts/summarize_paired.py`. It aligns saved predictions by circuit ID, rejects missing or duplicate test rows, verifies the two headline metrics against each report, and writes the per-observable and per-depth scorecard. The paired summary compares common tested circuits and common calibration draws. It includes both acquisition-batch and depth-stratified nominal bootstrap intervals. The published split shares some circuits with the second experiment, so their agreement is a sensitivity check on the same hardware campaign.

A score-only verification needs no checkpoints, circuit pickles or model environment. Install `requirements-audit.txt`, run `python scripts/fetch_upstream.py` and then `python scripts/summarize_paired.py --check`. The verifier reconstructs the test targets and hardware batches directly from the pinned numeric release. To check the source workbook, run `python scripts/verify_figure4.py`. The exploratory hardware-residual comparison uses `scripts/run_physics_residual.py` in the Qiskit environment. To render the learning curve, run `scripts/plot_learning_curve.py` in an environment with NumPy and Matplotlib.

The iLTM engine imports XGBoost. On macOS it requires a compatible OpenMP runtime. For the recorded run we used the isolated environment's existing PyTorch OpenMP library by including its `torch/lib` directory in `DYLD_LIBRARY_PATH`. The independent refit and linear sanity check can be reproduced with `python scripts/run_registry_regression.py ILTM --device cpu --audit-saved-iltm`. Kumo small used CPU because its preprocessing requires float64 on this platform. See [iLTM diagnostics](../reports/iltm_prediction_audit.json).

The TabPFN initialization control uses `--clean-seed 0 --initialization-seed 0` (repeat with 1 and 2) on the v2 runner and package version 6.4.1. Its audit records the parameter digest and the checkpoint-derived criterion state retained for this comparison.

For the linear controls run `python scripts/run_affine_baseline.py` and `python scripts/run_linear_comparators.py` after preparing the Algiers feature archive. The second script evaluates four-input multivariate OLS and a calibration-cross-validated linear ridge model on all 169 features. For the frozen Nori few-shot runs use `--clean-seed S --clean-budget B --chunk 450` with S from 0 through 4 and B equal to 18 or 90. Use `--inputs measurements_only` to give Nori the same four noisy observables as the multivariate affine comparator. All model settings otherwise match the clean 500-circuit runs.

## Reproduce the FakeLima simulation comparisons

The OLS tutorial uses the immutable sources listed in [the h12 manifest](../data/simulation_h12_sources.json). In the Python 3.9 circuit environment, install the reproduction requirements and `torch==2.5.1`, then run the first two commands below. Run the model in the TabPFN v2 environment described above with its pinned checkpoint in the weights directory. The final score check only needs the audit requirements and committed prediction files.

```sh
python scripts/simulation_fetch.py
python scripts/simulation_h12.py
python scripts/simulation_tabpfn.py weights --device mps
python scripts/simulation_verify.py
```

The [Figure 3 reproduction](../reports/simulation/fig3/README.md#reproduction) reuses the tutorial’s training rows and adds the archived test circuits and six reference predictions.

The four-input depth-transfer comparison uses the same training rows, the archived Figure 3 measurements, and the same TabPFN v2 checkpoint. In the Python 3.9 circuit environment with pandas 1.5.3, export the archived test vectors. In the TabPFN environment, fit all three estimators. The score check works in the audit environment.

```sh
python scripts/simulation_depth_transfer.py prepare
python scripts/simulation_depth_transfer.py fit weights --device mps
python scripts/simulation_depth_transfer.py verify
```

## Reproduce Qermit’s published comparisons

Install Pillow and run `python scripts/reproduce_qermit.py /tmp/qermit --download --check`. This verifies the pinned Zenodo files and the arXiv v3 source before checking all hardware and simulator figure cells and the printed hardware tables. The [Qermit protocol](mitigation-benchmarks.md#qermit-benchmark-reproduction) identifies the calibration records needed for a matched pretrained hardware fit.

### Fit frozen regressors with Qermit’s per-target calibration sets

For the fresh depolarising simulation, install the package versions recorded in `reports/qermit_cdr_simulation/full_unmirrored/protocol.json` in an isolated environment. Use the original Qermit target circuit files, measurement operators and notebook. The simulator uses the release’s gate-local noise model and shot counts. The compiler and random draws are new. Each estimator fits the same 20 noisy/ideal auxiliary measurements from the target at hand. The target ideal is only used to evaluate predictions.

```sh
python scripts/audit_qermit.py /tmp/qermit --download --check
python scripts/run_qermit_cdr_simulator.py /tmp/qermit --workers 6 --work /tmp/qermit-cdr-work
python scripts/run_qermit_cdr_simulator.py /tmp/qermit --check
python scripts/qermit_context_models.py tabpfn_v2 --weights weights --device cpu
python scripts/qermit_context_models.py nori --weights weights --device cpu
python scripts/qermit_context_models.py tabicl --weights weights --device cpu
```

Use the existing TabPFN environment for the TabPFN command and the Nori/TabICL environment for the other two. The default input covers all 680 unmirrored targets, with 340 circuits per family, in the modern depolarizing simulation. For another acquisition, pass its `contexts.npz` through `--contexts` to each fit. For a constant training target, TabPFN v2 returns its native constant prediction and reports zero checkpoint ensemble members. In the NumPy audit environment, verify the saved predictions and native group scores with the same input path.

```sh
python scripts/qermit_context_models.py MODEL --contexts PATH/contexts.npz --check
python scripts/summarize_qermit_panel.py --folder PATH --check
```

For the acquisition-era run, follow the [February software instructions](../reports/qermit_cdr_simulation/acquisition_era/README.md). Its generator runs the historical Qermit correction graph and saves the 20 actual calibration pairs passed to its affine fitter. Supply `reports/qermit_cdr_simulation/acquisition_era/full_unmirrored/contexts.npz` to each of the three frozen model commands.

To reproduce all 940 mirrored targets under the modern compiler, run `python scripts/run_qermit_cdr_simulator.py /tmp/qermit --mirrors-only --workers 8 --work /tmp/qermit-mirror-work`. Run each model with `--contexts reports/qermit_cdr_simulation/full_mirrored/contexts.npz`. The summary checker verifies every published circuit ID and includes the constant +1 prediction dictated by the mirror construction.
