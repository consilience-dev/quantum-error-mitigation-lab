# Generalization to deeper simulated circuits

Frozen TabPFN v2 used the four noisy observables to reduce the mean absolute error (MAE) at Trotter depths 15–29 from 0.125889 unmitigated to **0.100010**. It had lower MAE than either affine calibration and zero-noise extrapolation (ZNE) at all 15 deeper depths. The archived circuit-aware random forest had a lower overall MAE of **0.093013** and a lower MAE than four-input TabPFN at 13 of those 15 depths. This comparison uses the original ML-QEM no-readout-error FakeLima simulation. [Upstream experiment](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/tutorials/h17_compare_over_steps.ipynb)

| Estimator | MAE, depths 0–14 | Mean circuit L2, 0–14 | MAE, depths 15–29 | Mean circuit L2, 15–29 |
|---|---:|---:|---:|---:|
| Archived unmitigated | 0.060945 | 0.150454 | 0.125889 | 0.300644 |
| Archived ZNE | 0.041865 | 0.104831 | 0.113853 | 0.274634 |
| Archived full-feature ordinary least squares | 0.046249 | 0.115497 | 0.120944 | 0.292724 |
| Archived full-feature random forest | 0.015510 | 0.037336 | 0.093013 | 0.222042 |
| Diagonal affine | 0.054484 | 0.134728 | 0.120780 | 0.288966 |
| Four-input affine | 0.047202 | 0.117829 | 0.116706 | 0.281253 |
| Four-input frozen TabPFN v2 | 0.018126 | 0.043363 | 0.100010 | 0.238595 |

Each new estimator used the same 4,500 training circuits at depths 0–14. Diagonal affine fits one slope and intercept per observable using its own noisy measurement. Four-input affine uses all four noisy measurements and an intercept for each output. Each frozen TabPFN scalar regressor used all 4,500 training examples as context, a single ensemble member, seed zero, float32 inference and batches of 100 query rows. Each model fit was reused for both test partitions. Depth, circuit descriptors and ideal test labels were excluded from the predictors. The four-input configuration was fixed before fitting and scoring, and predictions were scored without clipping. This is a retrospective comparison. Previous 58-feature within-depth results and archived comparator scores were already available.

Each test partition has 4,500 archived rows, 300 per depth. MAE averages the four absolute observable errors per circuit and then averages over circuits. Mean circuit L2 averages the Euclidean norm of each four-component error, the endpoint plotted in Figure 3. The ideal targets are 10,000-shot ideal-simulator estimates. All 90 plotted raw, ZNE and forest L2 values across 30 depths reproduce the original workbook to six decimals. The forest and full-feature linear regression use circuit descriptors in addition to measurements, and ZNE uses additional noise-scaled measurements. [Source workbook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/paper_figures/ML-QEM%20Source%20data.xlsx) · [Per-depth scores and estimator details](scores.json)

The 300 depth-zero test rows share compiled circuit identities with training. The remaining within-depth test rows passed the original-circuit identity audit. The pinned release includes per-row predictions and targets for depths 15–29 but lacks the corresponding validation circuit objects. The deeper comparison therefore establishes transfer to previously unseen **depth labels** in this simulator. Compiled circuit identity overlap at these deeper depths remains unverified. The released archive also limits the deeper TabPFN test to measured-observable inputs. The [58-feature comparison](../fig3/README.md) covers depths 0–14.

## Reproduction

Use the h12 reproduction environment to export the pinned Figure 3 archive. Use the TabPFN v2 environment for inference with the previously downloaded, hash-verified checkpoint in `weights/`. The final command needs only NumPy and the committed artifacts and can run offline. To also verify the published plot values, pass `--workbook PATH` to the final command. [Environment and training-data instructions](../../../docs/running-models.md) · [Upstream source hashes](../../../data/simulation_fig3_sources.json) · [Targets and archived estimates](archive.csv.gz) · [New row predictions](predictions.csv.gz)

```sh
python scripts/simulation_depth_transfer.py prepare
python scripts/simulation_depth_transfer.py fit weights --device mps
python scripts/simulation_depth_transfer.py verify
```
