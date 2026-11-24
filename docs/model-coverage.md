# Model coverage on 24 November 2026

The discovery set starts with all 24 named models or versions in the pinned [TabTune registry](../configs/model_inventory.json) and adds regressors from other publishers. We screened for pretrained checkpoints that predict continuous targets. A release, model size or inference preset can share its architecture or weights with another entry. Counting these as independent model families would overstate breadth.

| Execution status | Models and qualifications |
|---|---|
| Frozen comparisons on the published split | Nori 1.5, TabICLv2, TabPFN v2, v2.5 and v2.6, TabDPT 1.3, Causilo, TabAbax, KumoTabular small and iLTM r128bn. The compact iLTM configuration failed an independent linear-task sanity check and is diagnostic only. TabAbax is derived from TabICLv2. For v2.6 we used the independently pinned Prior Labs checkpoint. |
| Downstream weight adaptation | TabPFN v2 and TabICLv2. Each implementation compares against its own frozen checkpoint at the same calibration budget. The local Nori 1.5 interface used here provides in-context regression. |
| Weights obtained, evaluation deferred | Mitra and Mitra v2, TabPFN v3 and v3.5. Priority went to repeating calibration draws, input controls and paired fine-tuning. |
| Other candidate families | LimiX, LimiX v2, TabFM, TabLDM, EXAONE-Tabular, ContextTab, TabPTM and TabSTAR. The EXAONE registry lists a recommended feature limit of 100 for the 169-feature input. ContextTab lists gated weights. TabSTAR requires downstream adaptation. These need compatible, documented evaluation paths. |
| Additional sizes and services | Nori 6M, 30M and 100M, other Kumo sizes, TabPFN v3.5 Fast and other historical checkpoints were outside this execution pass. TabH2O is a hosted or licensed candidate. |
| No matching pretrained regression path in this registry | The original TabICL v1 and the OrionBix, OrionMSP and OrionMSP v1.5 entries have classification paths. XRFM is trained from scratch. |

The scorecard includes every completed arm, including results worse than unmitigated hardware. Some models have results only on the published rows. See the actual draw counts alongside the [results](../reports/RESULTS.md). This is an exploratory study of the named checkpoints. Testing every historical size would add correlated comparisons while leaving the limitation of a single Ising experiment on one device intact.
