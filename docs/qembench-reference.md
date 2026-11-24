# External reference scores for QEM-Bench

Bao et al. (ICML 2025) report 22 datasets spanning structured and random circuits, simulated noise, and real IBM devices. Their first real-device task in Table 1, **Kyiv Pre**, uses 50-qubit Trotter circuits after filtering extreme noise outliers. The authors report 400 training, 78 validation and 113 test examples. Table 4 reports scalar expectation-value MAE and RMSE on this task. [Proceedings](https://proceedings.mlr.press/v267/bao25c.html) · [Paper, Tables 1 and 4](https://raw.githubusercontent.com/mlresearch/v267/main/assets/bao25c/bao25c.pdf)

| Published method on Kyiv Pre | MAE | RMSE |
|---|---:|---:|
| Noisy measurement | 0.057 | 0.069 |
| ZNE | 0.047 | 0.076 |
| OLS | 0.037 | 0.045 |
| Random forest | 0.030 | 0.037 |
| GTranQEM | 0.019 | 0.027 |
| QEMFormer | 0.018 | 0.026 |

These are author-reported scores for selected methods. The ML-QEM four-qubit Algiers results use different circuits, labels and partitions. A numerical comparison against Kyiv Pre requires predictions on its actual test rows, matching its outlier filter and calibration allowance. The Algiers circuit-vector L2 also aggregates the four observables differently from Kyiv's scalar RMSE. Our per-observable affine calibration and the OLS entry here each require an explicit feature specification before comparing their implementations.

On November 24, 2026 the repository named by the authors contained one 75-byte README at commit `1840abee179acd1ede268c69eae45d26d2e7c44d`. The public releases page was empty. Two public dataset requests had no replies supplying artifacts. We could verify the published numbers against the proceedings PDF (SHA-256 `5682825f9aeda0ee3449069b73616c99f4c05d93364e33b44881b97df47b6ba1`). The measured inputs, ideal targets, retained-circuit identities and train/validation/test assignments needed to evaluate pretrained models were unavailable through those sources. The Kyiv and Brisbane results supply external reference targets. A matched pretrained score on either device remains open. [Pinned repository](https://github.com/btyll/QEM-Bench-ICML/tree/1840abee179acd1ede268c69eae45d26d2e7c44d) · [Releases](https://github.com/btyll/QEM-Bench-ICML/releases) · [First dataset request](https://github.com/btyll/QEM-Bench-ICML/issues/1) · [Second dataset request](https://github.com/btyll/QEM-Bench-ICML/issues/2)
