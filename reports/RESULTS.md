# What the benchmark comparisons establish

On ML-QEM’s published Algiers hardware split of 500 training and 2,500 test rows, frozen Nori 1.5 scored 0.05318 mean circuit error, compared with 0.10004 for the authors’ random forest and 0.18682 for ZNE. A control using the known Ising parameters and calibration labels performed better still. The four-qubit FakeLima simulation also favored full-feature pretrained models at familiar depths. The separate few-shot study examines how those rankings change as labels become scarce.

Qermit asks how to correct a single target using 20 related calibration runs. On 680 unmirrored depolarizing simulator targets, affine CDR improved the median error ratio over raw in all 68 circuit groups in our acquisition-era replay. The three frozen regressors did so in 46–52 groups. Each frequently beat CDR on the median for Pauli gadgets, while CDR had lower mean absolute error in both families. [Qermit comparison](#qermit-with-20-calibration-runs-per-target)

## Hardware probabilistic error cancellation

Song et al. measured four controlled-phase circuits on a superconducting processor. At π/2 we recovered the published reduction in absolute error from 0.1690 to 0.0102 within 0.0001. Across all four phases, the mean absolute error was **0.0181 for hardware PEC**, 0.1341 for raw hardware, 0.1046 for affine, 0.1103 for Nori 1.5, 0.1831 for TabICLv2 and 0.0952 for TabPFN v2. Each regressor received the four published readout calibration states and the raw target expectation. The PEC arm also used measurements of the entangling gate. The released figures support a same-circuit comparison with this difference in available information. They cover one processor and four related targets. TabPFN’s score ranged from 0.0747 to 0.1832 under small digitization changes, so even its ranking against raw depends on the input reconstruction. [Paper](https://doi.org/10.1126/sciadv.aaw5686) · [Per-circuit scores, one-qubit control and sensitivity](pec_song/README.md)

Two independent hardware checks give additional evidence for the published PEC scores. From Russo et al.’s archived circuit outcomes we reproduced all 25 plotted improvement factors, including the six groups where PEC increased error at equal test-shot cost. Their ideal success probabilities all equal one. For Köylüoğlu et al.’s ten-qubit spin chain, the authors’ processed distributions give ground-state fidelity 0.88129 before mitigation and 0.96147 after PEC and readout mitigation, consistent with their published 0.96. The spin-chain release lacks the hardware calibration records for a comparable model fit. [Russo reproduction](pec_russo/README.md) · [Spin-chain reproduction](pec_spin_chain/README.md)

For the learned PEC protocol in Strikis et al., we recovered all 60 raw and corrected estimates shown in Figure 10 from two versions of the figure. Across the same ten phase settings on three IBM devices, the mean absolute error fell from 0.0626 to 0.0162. No public copy of the 24 labeled Clifford measurements and all 40 Pauli-insertion test measurements per device was located. The observed endpoints support this figure check. A refit and a same-input pretrained comparison require the original measurements. [Figure 10 and data availability](pec_strikis/README.md)

## Reproducing the published experiment

ML-QEM supplies real noisy measurements, ideal simulation targets, ZNE outputs, the circuits themselves, and a model reference. The authors' hardware notebook selects 50 training and 250 test rows at each depth 0–9. We used the same 169 inputs, ideal targets, row IDs and four independent scalar predictions per circuit. Running the upstream encoder on all 3,000 rows gave byte-identical features and targets. Five fresh fits of the four 100-tree forests gave mean circuit error **0.09977–0.10032**, compared with **0.10004** in the Figure 4 workbook. Pooled RMSE was **0.07453–0.07461**, close to the notebook's **0.07471**. Its model seed was unspecified. This reproduces the four-qubit hardware reference. Other architectures in the paper were studied in other settings. [Notebook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/demos/demo2_ising_4q_hardware_plot.ipynb) · [Forest refits](rf_reproduction.json) · [Encoder check](original_encoder_check.json) · [Workbook check](figure4_reference_check.json)

**Author-reported anchors.** Equal-weighting the ten published depth means yields **0.2097070** for unmitigated hardware, **0.1868189** for ZNE and **0.1000394** for RF. Our frozen Nori 1.5 score on those same 2,500 test rows is **0.0531771**. That is a 46.84% reduction relative to the published RF aggregate. The workbook provides per-depth aggregates. Our separate forest refits supply paired circuit predictions. Figure 4 has no affine entry. Scores for OLS, MLP and GNN in the paper's simulated experiments use different test circuits. [Upstream Figure 4 data](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/paper_figures/ML-QEM%20Source%20data.xlsx) · [Source verification](figure4_reference_check.json)

## Original and fresh 500-circuit comparisons

The primary error is the mean Euclidean distance between each predicted four-observable vector and its ideal vector. Lower is better. Pooled RMSE weights observable entries equally and can rank configurations differently. The original split has 500 training rows representing 453 distinct circuits and 2,500 test rows. It shares three depth-zero circuit identities across partitions. The fresh split uses 500 **unique** calibration circuits and 900 other unique circuits, balanced over depths 1–9. Every model within a draw receives the same circuit IDs. All draws use the same 900 test circuits. [Split manifest](../data/clean_split.json)

| Method | Our mean error on authors’ split | Our RMSE on authors’ split | Fresh mean error | Fresh calibration draws |
|---|---:|---:|---:|---:|
| Unmitigated hardware | 0.2097 | 0.1171 | 0.2167 | fixed |
| ZNE | 0.1868 | 0.1045 | 0.1905 | fixed |
| Per-observable affine | 0.1702 | 0.0994 | 0.1643 | 5 |
| Four-input affine | 0.1437 | 0.0872 | 0.1409 | 5 |
| Full-feature linear ridge | 0.1430 | 0.0879 | 0.1554 | 5 |
| Random forest | 0.1000 | 0.0746 | 0.1100 | 5 |
| CatBoost | 0.1043 | 0.0714 | 0.1106 | 5 |
| Kernel ridge | 0.0956 | 0.0660 | 0.0921 | 5 |
| Nori 1.5 (4 members) | 0.0532 | 0.0456 | 0.0552 | 5 |
| Causilo | 0.0556 | 0.0486 | 0.0554 | 5 |
| KumoTabular small | 0.0550 | 0.0483 | 0.0545 | 1 |
| TabAbax | 0.0582 | 0.0490 | 0.0561 | 1 |
| TabICLv2 (8 members) | 0.0616 | 0.0514 | 0.0626 | 5 |
| TabPFN v2 | 0.0671 | 0.0545 | 0.0669 | 5 |
| TabPFN v2.5 | 0.0661 | 0.0536 | — | 0 |
| TabPFN v2.6 | 0.0741 | 0.0572 | — | 0 |
| TabDPT 1.3 | 0.1332 | 0.0881 | 0.1354 | 5 |
| iLTM r128bn (query-batch normalization) | 0.6492 | 0.3855 | 0.6214 | 5 |

The **per-observable affine** model fits one slope and one intercept per observable. The **four-input affine** predicts each output from all four noisy measurements and an intercept. The **full-feature linear ridge** model uses the same 169 inputs as the full pretrained runs and selects its regularization strength within grouped calibration folds. These three entries are local fits. A separate exploratory residual correction in the Ising section uses the exact coupling and serves a different purpose. [Affine coefficients](affine.json) · [Linear models and validation](linear_comparators.json)

A preliminary one-member TabICLv2 timing run gave published mean error 0.06025. Its original row predictions were overwritten. A separately archived repeat matched its two original scores and all ten depth means exactly. The table uses the eight-member default with its complete prediction file. [Timing pilot](tabicl_single_member_timing_pilot.json) · [Repeat and checksum](tabicl_published_single_member_pilot_rerun.json)

The published forest entry averages its five refits. Other published model entries represent one specified configuration. Fresh entries average the completed draws shown. Dashes indicate an unscored configuration. Hardware and ZNE use fixed acquired values for each test circuit. No test-based model selection or post-hoc clipping was applied by the benchmark runner. The models have different inference defaults and compute costs. The additional frozen checkpoints use one ensemble member where supported. Nori uses the released inference configuration on the published split and training-only high-dimensional projection on the fresh split. The compact iLTM engine uses unlabeled query-batch statistics when scaling outputs. It also failed a noiseless linear-regression sanity check (RMSE 0.7000 versus 0.4445 for a constant prediction). Its score is a diagnostic sensitivity result for this adapter and checkpoint. [iLTM audit](iltm_prediction_audit.json) These measurements characterize the tested configurations. [Coverage and deferrals](../docs/model-coverage.md) · [Settings and reproduction commands](../docs/running-models.md)

Across five draws, the paired difference in mean error against the matched forest was **-0.0545 [-0.0627, -0.0477]** for Causilo, **-0.0430 [-0.0499, -0.0375]** for TabPFN v2 and **+0.0254 [+0.0179, +0.0318]** for TabDPT. Brackets give nominal 95% intervals resampling the 60 acquisition batches, conditional on these five draws. For example, Causilo's difference against the calibration-tuned kernel-ridge baseline was **-0.0366 [-0.0399, -0.0336]**. The individual intervals are exploratory and have no multiple-comparison correction. Full per-observable RMSE, per-depth error, paired estimates against raw/ZNE/affine/forest/kernel ridge and depth-stratified sensitivity intervals are available. [Scorecard](scores.csv) · [Paired comparisons](clean_paired_summary.json)

An additional simultaneous 95% sensitivity analysis covers eight contrasts. It includes Nori 1.5, TabICLv2 and TabPFN v2 against the forest and kernel ridge, and both tuned models against their own frozen checkpoints. All six frozen-model intervals lie below zero. The TabPFN tuning interval includes zero, and the TabICLv2 tuning interval lies above zero. It uses the same acquisition-batch resamples and five shared calibration draws. Other checkpoint-screen comparisons remain exploratory.

## How far do the few-shot budgets go?

The pilot’s 18 and 90 calibration circuits gave two or ten examples at each of the nine nonzero depths, after excluding depth-zero duplicates. That pilot used 180 test circuits and omitted the paper’s 500-row calibration budget. Its numeric affine result and exact test IDs could not be recovered. The table below reports new experiments on the 900-circuit holdout. Within each of five draws, the larger budgets contain the smaller budgets’ circuits. At 500, five depths receive 56 circuits and four receive 55. [Pilot provenance](../docs/provenance.md) · [Frozen circuit IDs](../data/clean_split.json)

| Unique calibration circuits | Per-observable affine | Four-input affine | Nori, four inputs | Forest, full inputs | TabPFN v2, full inputs | Nori, full inputs |
|---:|---:|---:|---:|---:|---:|---:|
| 18 | 0.1876 | 0.1994 | 0.2348 | 0.2771 | 0.2064 | 0.2152 |
| 45 | 0.1692 | 0.1582 | — | 0.1830 | 0.1480 | — |
| 90 | 0.1653 | 0.1459 | 0.1247 | 0.1507 | 0.1230 | 0.1134 |
| 180 | 0.1650 | 0.1433 | — | 0.1314 | 0.0982 | — |
| 270 | 0.1647 | 0.1412 | — | 0.1218 | 0.0848 | — |
| 450 | 0.1643 | 0.1409 | — | 0.1120 | 0.0685 | — |
| 500 | 0.1643 | 0.1409 | 0.0989 | 0.1100 | 0.0669 | 0.0552 |

Each cell is mean circuit L2 across the five calibration draws. Dashes mark budgets at which Nori was not run. At 18 circuits, the per-observable affine correction had the lowest mean error in the table. It beat both full-input pretrained models in four of the five draws. At 90 and 500, both full-input models beat it on every draw. Using all 169 inputs, linear ridge scored 0.3994, 0.2474 and 0.1554 at 18, 90 and 500 respectively.

Giving Nori and four-input affine **the same four noisy measurements** isolates a comparison of the two estimators. Nori minus affine was **+0.0354** at 18 circuits (higher error in four draws), **−0.0211** at 90 and **−0.0420** at 500 (lower error in every draw at both budgets). Nominal 95% intervals from paired resampling of the 60 acquisition batches were [0.0304, 0.0401], [−0.0240, −0.0181] and [−0.0481, −0.0368]. These exploratory intervals condition on the five observed calibration draws and one hardware campaign. [Per-draw scores](scores.csv) · [Paired differences and intervals](clean_paired_summary.json)

![Mean circuit error by calibration budget, with separate panels for full inputs and measurement-only affine and Nori models](learning_curve.png)

Markers show evaluated budgets. Shading spans the five calibration draws. Horizontal lines show raw hardware and ZNE. Calibration counts track ideal circuit labels. A total hardware-cost comparison would also require shot counts and the extra noise-scaled circuits used by ZNE.

## What did the noisy measurements add?

Each row below is a matched input ablation on fresh calibration draw 0. The descriptor column uses the 165 gate counts and rotation-bin features. The measurement column uses the four noisy hardware observables. The full model receives both.

| Frozen checkpoint or baseline | Descriptors alone | Measurements alone | Both |
|---|---:|---:|---:|
| Random forest | 0.2398 | 0.1160 | 0.1113 |
| Causilo | 0.0930 | 0.0961 | 0.0535 |
| Nori 1.5 | 0.0898 | 0.0999 | 0.0565 |
| TabICLv2 | 0.1278 | 0.0969 | 0.0600 |
| TabPFN v2 | 0.1288 | 0.0999 | 0.0668 |
| iLTM | 0.6017 | 0.6329 | 0.6242 |

For TabPFN v2, jointly permuting the four calibration measurements within Trotter depth and Pauli basis raised mean error from 0.06684 to 0.13987. All 900 test circuits kept their real measurements. Both the omission and permutation controls show that the pairing of measurements with circuit descriptors contributed predictive information for this model. [Permutation run](tabpfn_v2_clean_s0_shuffle_train_noisy.json)

**The representation changes the interpretation.** Each ideal vector in this family depends on a known Ising coupling, depth and measurement basis. Interpolating in the exact coupling within each depth and basis, using the same 500 calibration labels and **zero measured features**, scored **0.02161** on the published rows and **0.01447** across the five fresh draws. This post-hoc control has access to exact problem parameters beyond the paper's binned representation. It shows how much of this benchmark can be solved by predicting ideal circuit behavior. Adding a calibration-fitted affine measurement-residual correction to that interpolator raised error to **0.01602**. Calibration residuals were calculated by leaving each calibration circuit out in turn. The change was **+0.00154** [0.00140, 0.00168] across the observed batches. [Interpolation and endpoint treatment](physics_control.json) · [Residual control](physics_residual.json)

## Does fine-tuning help?

For each observable we compared in-context inference from a frozen checkpoint with gradient updates starting from that same checkpoint. The 500-label budget includes 450 optimization and 50 validation circuits, followed by final inference with all 500 as context. The test circuits never supplied optimization or validation labels.

- **TabPFN v2.** The fixed three-epoch adaptation changed the decoder weights for all observables. On the published split mean error moved from 0.06706 to 0.0663. Across five matched fresh draws it moved from **0.06693 to 0.06687**. The paired change was -0.0001 [-0.0005, +0.0003]. This short schedule did not establish an adaptation benefit on the fresh split.
- **TabICLv2.** Across five matched fresh draws, mean error moved from **0.0626 to 0.0632**. The paired change was +0.0006 [+0.0004, +0.0008]. Calibration-only validation retained updated weights for 12 of 20 target fits. Tuning increased test error on four of the five draws. [Fine-tuning history](tabicl_clean_500_seed0_tuned.json)

**Does the checkpoint matter?** We also ran the official TabPFN v2 architecture with newly initialized neural weights on the same 500 calibration and 900 test circuits as frozen draw 0. All 900 predictions were constant within each observable for each newly initialized network. Three initialization seeds gave mean errors of 0.4054, 0.4075, 0.4072, versus 0.0668 with the pretrained weights. The official constructor initialized all 11,081,864 neural parameters and sets the residual outputs to zero at initialization. The architecture, target-bin distribution, inference preprocessing and calibration labels were retained. This isolates the contribution of the stored neural weights for this frozen configuration. A network trained from scratch under a matched optimization budget is a separate comparison. [Initialization audit](tabpfn_v2_init0_clean_s0.json)

## Comparison with the authors’ simulated OLS tutorial

The ML-QEM `h12_ols` notebook uses 4,500 training and 1,500 validation rows from a four-qubit Ising simulation on FakeLima with readout noise disabled. Its OLS model uses all 58 encoded inputs. We evaluated a frozen, single-member TabPFN v2 with those same inputs, training labels and validation circuits. The notebook prints pooled RMSE over all four observables. Depth files were loaded in numeric order for our replay and refit because the notebook uses an unspecified filesystem order. [Authors’ notebook](https://github.com/qiskit-community/ml-qem/blob/faf3e44ac686d62b2184a983d89e655084759a26/docs/tutorials/h12_ols.ipynb) · [Pinned source files](../data/simulation_h12_sources.json)

| Model | Notebook RMSE | Our RMSE, 1,500 validation rows | Our RMSE, 1,400 circuits absent from training |
|---|---:|---:|---:|
| Unmitigated | 0.097725 | 0.097725 | 0.101085 |
| Authors’ released OLS model | 0.074108 | 0.074104 | 0.073694 |
| Fresh OLS refit | — | 0.074052 | 0.073605 |
| Frozen TabPFN v2 | — | 0.016764 | 0.017149 |

Six compiled circuit identities shared with training account for all 100 depth-zero validation rows. Removing them leaves 1,400 unique validation circuits. TabPFN’s improvement over OLS persists in that subset. These are the notebook’s validation rows, and the experiment uses the same tractable Ising family as the hardware test. [Targets and OLS scores](simulation/reference.json) · [TabPFN checkpoint and scores](simulation/tabpfn_v2.json)

## Comparison with the published Figure 3 predictions

The paper also releases test predictions for FakeLima Ising circuits. We matched the four noisy and four ideal values for **every row** in the Figure 3 no-readout-error condition at depths 0–14. We used the same 4,500 training rows as the OLS tutorial, and scored the 4,500 `val_extra` test rows. Frozen Nori 1.5, TabICLv2 and TabPFN v2 received the 58 tabular features and all 4,500 training rows, with one regressor per observable. The published model predictions below were rescored from the archive. The forest, MLP, OLS and GNN retain their original model-specific representations. QEMScore separately reports a mean absolute error (MAE) on this subset. We reproduced all six of its reference values to six decimal places. [ML-QEM, Figure 3](https://arxiv.org/html/2309.17368v2#S2.F3) · [QEMScore reanalysis](https://github.com/yzhao062/QEMScore/blob/4b95b3e91b44bbfa75d70279ac5d68f05d148ce2/reanalysis/mlqem/README.md)

| Method | Mean circuit L2, 4,500 rows | MAE, 4,500 rows | MAE, 4,200 unique circuits absent from training |
|---|---:|---:|---:|
| Unmitigated | 0.150454 | 0.060945 | 0.064487 |
| ZNE | 0.104831 | 0.041865 | 0.043861 |
| OLS | 0.115497 | 0.046249 | 0.045677 |
| Random forest | 0.037336 | 0.015510 | 0.016031 |
| MLP | 0.052709 | 0.022274 | 0.023061 |
| GNN | 0.126836 | 0.051041 | 0.051403 |
| Frozen Nori 1.5 | 0.027161 | 0.011466 | 0.011727 |
| Frozen TabICLv2 | 0.027231 | 0.011467 | 0.011727 |
| Frozen TabPFN v2 | 0.028171 | 0.011863 | 0.012114 |

Nori and TabICLv2 gave similar scores, each reducing mean circuit L2 by about 27% compared with the archived forest. Both used their complete training contexts. [Nori run](simulation/fig3/nori.json) · [TabICLv2 run](simulation/fig3/tabicl.json)

TabPFN reduced MAE by 23.5% relative to the archived forest on the 4,500 test rows. On the 4,200 unique nonzero-depth circuits, the paired MAE difference was −0.00392 (nominal 95% interval −0.00417 to −0.00367, resampling circuits within depth). TabPFN had lower MAE at each of depths 1–14 and the forest had lower MAE at depth 0. This interval conditions on the training set and simulated noise setting. The same six training identities cover all 300 depth-zero test rows. The 4,200 remaining circuits are unique and absent from both training and the OLS tutorial validation set under compiled circuit identity. This covers interpolation over held-out circuits at familiar depths, one simulated noise condition and one fixed training set. The archive also has predictions for depths 15–29, but its corresponding `val_extra` circuit files for that condition end at depth 14 in the pinned release. A full-feature comparison at those extrapolation depths requires the missing circuit descriptors. [Matched sources and row predictions](simulation/fig3/reference.json) · [Checkpoint and per-depth results](simulation/fig3/tabpfn_v2.json) · [Paired uncertainty](simulation/fig3/paired_rf.json)

## Do measurement corrections transfer to deeper circuits?

The Figure 3 archive retains the four noisy observables, ideal targets and predictions for 4,500 circuits at depths 15–29. We fit each new estimator **once** on the four measurements from all 4,500 training rows at depths 0–14, then used the fitted model on both test populations. At each depth, 300 rows were scored. All rows use the same four observable targets. The archived RF and OLS had access to additional circuit features.

| Method | MAE at depths 0–14 | MAE at depths 15–29 | Mean circuit L2 at depths 15–29 |
|---|---:|---:|---:|
| Unmitigated | 0.060945 | 0.125889 | 0.300644 |
| ZNE | 0.041865 | 0.113853 | 0.274634 |
| Diagonal affine on measurements | 0.054484 | 0.120780 | 0.288966 |
| Four-input affine | 0.047202 | 0.116706 | 0.281253 |
| Frozen TabPFN v2, same four inputs | 0.018126 | 0.100010 | 0.238595 |
| Authors’ full-feature OLS | 0.046249 | 0.120944 | 0.292724 |
| Authors’ circuit-aware RF | 0.015510 | 0.093013 | 0.222042 |

Four-input TabPFN had lower MAE than both affine corrections and ZNE at all 15 deeper depths. The authors’ forest had lower MAE than four-input TabPFN at 13 of those depths and lower overall error. At depths 0–14, supplying all 58 features to TabPFN reduced MAE from 0.018126 to 0.011863. The corresponding full-feature TabPFN score and compiled-circuit identity audit at depths 15–29 remain unavailable because the circuit objects are missing. These are results for extrapolation in depth on the same simulator and circuit family. [Scores, coefficients, checkpoint and source hashes](simulation/depth_transfer/scores.json) · [Row-level predictions](simulation/depth_transfer/predictions.csv.gz)

## Qermit with 20 calibration runs per target

Qermit fits one affine CDR correction to 20 near-Clifford calibration runs for each target. Its depolarizing simulator has 680 unmirrored targets. Each of the 68 published circuit-family, width and depth groups contains ten targets. For each target, divide absolute prediction error by the absolute error of its separate raw measurement. Report the median and worst ratio within each group. The counts below give each group one vote. Each frozen regressor used precisely the 20 noisy/ideal pairs and the noisy query passed to the affine CDR fit. [Paper](https://quantum-journal.org/papers/q-2023-07-13-1059/) · [Released results](qermit_reference_cells.csv)

### Acquisition-era software

Using the February 2022 Qermit source and February dependency releases, affine CDR improved the median relative error over raw in all 34 Pauli-gadget and all 34 random groups. Archived CDR did so in 34 and 33. CDR mean absolute error in our replay was 0.01546 on Pauli gadgets and 0.05348 on random circuits, compared with 0.01218 and 0.06001 in the archive. We used fresh random seeds. The exact installed versions and calibration measurements from the original acquisition are unavailable. [Source, settings and direct graph validation](qermit_cdr_simulation/acquisition_era/README.md) · [Complete circuit audit](qermit_cdr_simulation/acquisition_era/full_unmirrored/native_graph_audit.json)

| Estimator on the same calibration inputs | Lower median ratio than raw, out of 68 | Lower worst ratio than raw, out of 68 | Pauli-gadget groups with lower median ratio than CDR, out of 34 | Random groups with lower median ratio than CDR, out of 34 | Mean absolute error across 680 targets |
|---|---:|---:|---:|---:|---:|
| Raw | — | — | 0 | 0 | 0.09871 |
| Affine CDR | 68 | 51 | — | — | 0.03447 |
| Frozen Nori 1.5 | 52 | 29 | 27 | 6 | 0.09444 |
| Frozen TabICLv2 | 46 | 24 | 26 | 8 | 0.12320 |
| Frozen TabPFN v2 | 47 | 27 | 27 | 6 | 0.09857 |

The three frozen models frequently lowered the median relative error on Pauli gadgets. Their larger errors on some targets raised mean absolute error above affine CDR for every model in both families. For Nori, mean absolute error minus CDR was +0.00603 on Pauli gadgets and +0.11390 on random circuits. Exploratory paired bootstrap intervals adjusted across six comparisons were [0.00222, 0.01050] and [0.09214, 0.13640], respectively. All six intervals lie above zero, conditional on these circuit groups and this simulated acquisition. [Native group scores](qermit_cdr_simulation/acquisition_era/full_unmirrored/summary.json) · [Paired analysis and all six intervals](qermit_cdr_simulation/acquisition_era/full_unmirrored/paired_mae_intervals.json)

### Modern software sensitivity

We also replayed the same unmirrored target IDs with the current generator and compiler. All estimators in this table share the same fresh calibration draws, shot allocations and queries from that run. [Settings and paired inputs](qermit_cdr_simulation/full_unmirrored/README.md) · [All group scores](qermit_cdr_simulation/full_unmirrored/summary.json)

| Fresh estimator | Lower median error ratio than raw, out of 68 | Lower worst error ratio than raw, out of 68 | Pauli-gadget groups with lower median ratio than CDR, out of 34 | Random groups with lower median ratio than CDR, out of 34 | Mean absolute error across 680 targets |
|---|---:|---:|---:|---:|---:|
| Raw | — | — | 0 | 15 | 0.09643 |
| Affine CDR | 53 | 31 | — | — | 0.06444 |
| Frozen Nori 1.5 | 44 | 25 | 27 | 10 | 0.10454 |
| Frozen TabICLv2 | 38 | 22 | 26 | 8 | 0.13351 |
| Frozen TabPFN v2 | 39 | 22 | 27 | 7 | 0.11575 |

All three frozen models improved the median relative error over CDR for most Pauli-gadget groups. Large errors on individual targets reversed that ranking on mean absolute error within the same family. CDR had lower mean absolute error than every frozen model in both families. This is one calibration draw and shot realization per target. [Per-family median, worst and pooled results](qermit_cdr_simulation/full_unmirrored/summary.json)

The archived simulation supplies an independent reference on these circuit IDs. The authors’ ZNE improved the median over raw in all 34 groups of each family. Archived CDR did so in 34 Pauli-gadget and 33 random groups. Fresh CDR reproduced the Pauli-gadget count but reached only 19 random groups. On random circuits, mean absolute error was 0.06001 for archived CDR and 0.11670 in the replay. On Pauli gadgets both round to 0.01218. Only the fresh estimators share identical calibration measurements. The historical hardware archive supplies final method outputs without the corresponding per-target CDR inputs. [Published scores](qermit_reference_cells.csv) · [Calibration diagnostics](qermit_cdr_simulation/full_unmirrored/context_diagnostics.json)

Within the fresh random-circuit replay, 132 of 340 target queries fell within their 20 auxiliaries’ noise-affected expectation range before shot sampling. Replacing every sampled value in the CDR fit and query with its simulator expectation reduced MAE from 0.11670 to 0.11275. Finite-shot sampling alone therefore cannot account for the gap from the archived 0.06001. The historical compilation and the authors’ auxiliary circuits and seeds are unknown. [Shot-free and sampled diagnostics](qermit_cdr_simulation/full_unmirrored/summary.json)

### Mirrored simulator control

The other 940 targets in the published depolarizing simulator form 94 mirrored circuit groups. The circuit construction fixes the ideal observable at +1. Predicting +1 for every target gives mean absolute error about 2 × 10⁻¹⁵ against the stored numerical labels. In the modern replay, mean absolute error was 0.52275 for raw, 0.34748 for CDR, 0.46890 for Nori, 0.51468 for TabICL and 0.45478 for TabPFN. The unmirrored circuits above test performance when the ideal output varies. [Mirror scores and constant control](qermit_cdr_simulation/full_mirrored/summary.json)

## Limits and next independent test

The hardware results cover one four-qubit Ising family measured in one campaign on one processor. The FakeLima simulations add measurements under simulated noise for that same circuit family. The original and fresh split overlap across experiments. Repeated calibration draws quantify sensitivity to label choice within the same device and circuit family. Checkpoint provenance, including possible overlap of undisclosed real-table pretraining corpora with released benchmarks, matters for interpreting the model comparison.

The next confirmatory experiment should freeze a model, conventional baseline, parameter-only control, calibration budget and endpoint before examining independent test labels. A concrete candidate is the Placidi et al. random-circuit release with separate Algiers training and Hanoi test archives and full output distributions. Derive fixed single-qubit Z expectations with a verified bit order, audit logical circuit overlap, and evaluate measurement-only, circuit-only and combined inputs. The Qermit experimental archive supplies a further circuit-family/device check. QEM-Bench also provides existing OLS, forest and QEMFormer scores for 50-qubit Kyiv hardware. The filtered rows and splits needed for a direct evaluation were unavailable in its official repository. These independent hardware evaluations remain prospective. [QEM-Bench reference scores](../docs/qembench-reference.md) · [External protocol](../docs/model-comparison.md) · [Placidi et al. archive](https://zenodo.org/records/18541433) · [Qermit archive](https://doi.org/10.5281/zenodo.6472281)
