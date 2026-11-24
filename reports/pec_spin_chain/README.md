# Hardware PEC reproduction on critical spin chains

Köylüoğlu et al. measured central charge on IBM quantum processors. Their protocol combines sparse Pauli–Lindblad probabilistic error cancellation (PEC), readout mitigation and physics-based post-processing. For the ten-spin open Ising chain, the paper reports 10,000 sampled mitigation circuits with 200 shots each. The resulting probability vector represents one target state in one measurement basis. [Peer-reviewed paper, Nature Communications 17, 305 (2026)](https://www.nature.com/articles/s41467-025-66775-9) · [Data and notebooks](https://doi.org/10.5281/zenodo.17291855)

## Recomputed scores

The ten-spin reconstructed fidelity in Figure 2C is **0.96146628**, which reproduces the paper’s **0.96**. Applying the same overlap calculation to the raw vector gives **0.88128638**. This calculation uses the squared overlap between the square roots of the measured and exact ground-state probabilities. Following the authors’ notebook, we set 12 negative mitigated probabilities to zero and renormalized the vector. The fidelity has the paper’s positive-amplitude, coherent pure-state interpretation. The measurements do not establish the hardware state's coherences.

| Paper panel and observable | Published estimate | Recomputed estimate |
|---|---:|---:|
| Fig. 2D, open Ising, Z basis, central charge at moment 1 | 0.59 ± 0.11 | 0.59298 ± 0.10746 |
| Fig. 2E, open Ising, Z basis, moment 3 | 0.50 ± 0.13 in the main text | 0.50000 ± 0.13171 |
| Fig. 3E, periodic Ising, X basis, moment 4 | 0.39 ± 0.08 | 0.39426 ± 0.08408 |
| Fig. 4B, periodic XXZ, X basis, moment 4 | 0.86 ± 0.05 | 0.86187 ± 0.05209 |
| Fig. 4D, periodic XXZ, Z basis, moment 4 | 0.96 ± 0.07 | 0.96401 ± 0.06554 |

The Figure 2E caption gives 0.49 for moment 3. The article’s main text gives 0.50, and the independent fit and archived notebook yield 0.4999953. We independently fit all 19 experimental central-charge estimates plotted in these panels and checked their uncertainties against the saved notebook outputs within 2 × 10⁻⁷. The ten-spin uncertainty is a regression standard error. The twelve-spin estimates combine the dispersion across 20 archived bootstrap distributions with the within-fit uncertainty. The notebook applies a symmetry-sector filter, a threshold based on negative quasiprobabilities, and normalization to each bootstrap instance. [Complete numerical results](scores.json) · [Paper, Figures 2–4 and Methods](https://www.nature.com/articles/s41467-025-66775-9)

## Data available for a pretrained comparison

The archive provides raw, mitigated and exact outcome distributions for the four- and ten-spin open Ising examples. For twelve spins it provides post-PEC bootstrap distributions for the periodic Ising X-basis and XXZ X- and Z-basis charge analyses. The optimization trajectories in the ten-spin directory come from classically simulated VQE. The benchmark learns hardware noise, then samples its inverse on each target. The archive omits the layer-characterization measurements and the signed, sampled circuit-level outcomes that precede these probability vectors. Recomputing the published scores starts from the authors’ processed hardware data.

A matched supervised comparison requires access to the calibration measurements and known references available independently of the test state. Using the exact probabilities of selected test bitstrings as training labels would expose the target state and change the paper’s resources. Multiple Rényi moments and bootstrap distributions reuse the same measurements. We therefore report no pretrained circuit-level score for this release. [Archive description and file](https://zenodo.org/records/17291855)

## Reproduction

From the repository root, run the following command with NumPy installed. The script checks the 19,967,731-byte archive against the published MD5 and pinned SHA-256 before opening its numeric files. It reads the saved notebook values for verification and executes no notebook code.

```sh
python scripts/reproduce_pec_spin_chain.py /tmp/pec-spin-chain/analysis.zip --download --check
```
