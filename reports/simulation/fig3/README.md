# ML-QEM Figure 3 simulation comparison

On 4,500 held-out FakeLima Ising circuits at Trotter steps 0–14, frozen TabPFN v2 achieved mean absolute error (MAE) **0.011863**. The strongest archived comparator, the random forest, scored **0.015510**. TabPFN used the same training population of 4,500 labeled circuits and the authors’ 58 tabular features. Each of four scalar regressors used a single frozen ensemble member, a fixed seed of zero and the full context. The reference columns come from the original paper’s saved predictions. [ML-QEM evaluation notebook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/tutorials/h17_compare_over_steps.ipynb) · [QEMScore reanalysis and endpoint](https://github.com/yzhao062/QEMScore/blob/4b95b3e91b44bbfa75d70279ac5d68f05d148ce2/reanalysis/mlqem/README.md)

| Method | Mean MAE, all 4,500 | Mean L2, all 4,500 | Mean MAE, 4,200 unseen |
|---|---:|---:|---:|
| Frozen TabPFN v2 | 0.011863 | 0.028171 | 0.012114 |
| Archived random forest | 0.015510 | 0.037336 | 0.016031 |
| Archived MLP | 0.022274 | 0.052709 | 0.023061 |
| Archived ZNE | 0.041865 | 0.104831 | 0.043861 |
| Archived all-feature OLS | 0.046249 | 0.115497 | 0.045677 |
| Archived GNN | 0.051041 | 0.126836 | 0.051403 |
| Unmitigated | 0.060945 | 0.150454 | 0.064487 |

The original Figure 3 plots the mean Euclidean distance from the archived ideal four-observable vector at each depth. The all-row L2 column averages these depth means with equal weights of 300 circuits each. The MAE endpoint follows QEMScore, averaging the four absolute observable errors within a circuit and then averaging over circuits. All six archived MAEs reproduce the reanalysis at its six-decimal precision. All per-record L2 errors match the original stored L2 series. The 45 displayed workbook values for raw, ZNE and RF in the top-left panel agree to six decimals. [Scoring implementation](https://github.com/yzhao062/QEMScore/blob/4b95b3e91b44bbfa75d70279ac5d68f05d148ce2/reanalysis/mlqem/published_errors.py) · [Figure 3 source workbook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/paper_figures/ML-QEM%20Source%20data.xlsx)

Each test row matches its archived paper row on all four noisy values and all four ideal values in the order preserved by the source notebook within its depth file. Source files, content digests and original row identifiers are retained. Labels are the released ideal-simulator estimates from 10,000 shots. No readout error is present in this simulated setting. The comparison covers test circuits at training depths 0–14. Figure 3 also reports extrapolation to depths 15–29.

The 300 depth-zero test rows share six compiled circuit identities with training. The remaining 4,200 test rows have distinct compiled circuit fingerprints absent from training. TabPFN has lower MAE than RF at all 14 remaining depths. On these unseen identities the paired MAE difference, TabPFN minus RF, is −0.003917 (95% bootstrap interval −0.004165 to −0.003673). The bootstrap resamples whole paired circuits within each depth and conditions on the training set, checkpoint and simulated noise setting. [Paired result](paired_rf.json) · [Archived reference predictions](reference.csv.gz) · [Frozen predictions](tabpfn_v2.csv.gz)

## Reproduction

Use the repository’s h12 environment and prepare its verified training tensors first. The archived figure dataframe requires pandas 1.5.3 in that Python 3.9 environment. Use the model environment and the pinned weight recorded in the TabPFN report for the frozen run. The final verification uses only NumPy and the committed prediction and report files. Supplying the pinned original workbook with `--workbook PATH` also checks the plotted values. [h12 instructions](../../../docs/running-models.md) · [Source manifest](../../../data/simulation_fig3_sources.json) · [Encoder and feature digests](reference.json) · [Checkpoint and settings](tabpfn_v2.json)

```sh
python scripts/simulation_fig3_fetch.py
python scripts/simulation_fig3_reference.py
python scripts/simulation_fig3_tabpfn.py weights --device mps
python scripts/simulation_fig3_verify.py --check
```
