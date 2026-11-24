# Hardware probabilistic error cancellation on three processors

Russo et al. measured the probability of the ideal output bit string on randomized benchmarking (RB) and mirror circuits. We reproduced all **25 hardware PEC improvement factors** in Figures 3 and 4 from the authors’ circuit-level estimates. PEC reduced root-mean-square error in **19 of 25** groups at equal shot cost. The largest absolute difference between our calculation and the authors’ vector figure coordinates was 3.23 × 10⁻⁸. The paper appeared in the peer-reviewed *IEEE Transactions on Quantum Engineering* in 2023. [Published paper](https://doi.org/10.1109/TQE.2023.3305232) · [Authors’ figure source](https://arxiv.org/src/2210.07194v2) · [Authors’ archived data and notebook](https://github.com/unitaryfoundation/research/tree/e0457555978bf3125b3338e48f93dbfc50efd9fd/qem-on-hardware)

| Processor | Qubits and circuit families with published hardware PEC results | Groups with lower error after PEC |
|---|---|---:|
| IBMQ Lima | 3 and 5, RB and mirror | 15/16 |
| IonQ Harmony | 3, RB and mirror | 4/6 |
| Rigetti Aspen-M2 | 3, RB | 0/3 |

Each group contains four circuit instances at one depth, with one trial per instance. Each raw estimate uses 10,000 shots. Each PEC estimate uses 100 sampled circuit variants with 100 shots apiece. The authors modeled two-qubit gate noise using provider-reported error rates. Equation 6 reduces here to raw RMSE divided by PEC RMSE. A factor greater than one favors PEC. We retained reported estimates outside the physical probability range, as the paper's signed estimator permits. On IonQ Harmony's three-qubit RB circuits, for example, the factors at depths 1, 5 and 9 were 0.03036, 3.05823 and 2.32564. [Scores by depth](depth_scores.csv) · [100 paired circuit observations](published_circuit_rows.csv) · [Source repository license](https://github.com/unitaryfoundation/research/blob/e0457555978bf3125b3338e48f93dbfc50efd9fd/LICENSE)

The archive has 180 circuit observations across distinct exports. A second export of IonQ mirror results repeats all six numeric files byte-for-byte. The plotted points select 100 observations from the distinct exports. We verified the alignment of each depth, including the three-qubit IBM mirror series at the figure's x coordinates 1 through 5. [Release and duplicate audit](audit.json)

## What a pretrained model could learn here

Every scored observable is the projector onto the target circuit's deterministic ideal output, so all 100 ideal probabilities are exactly one. The constant prediction one achieves zero RMSE, making the denominator of Equation 6 zero. The protocol supplies no supervised training circuits. Each hardware PEC export contains raw and corrected estimates, ideal values, gate counts and an empty noise-scale sidecar. It omits the individual signed variant outcomes needed to evaluate a new estimator on precisely the same PEC samples. These circuits are a useful control for a known answer. A supervised model trained on their labels would not establish that the model can correct unknown observables.

## Verification

Run from the repository root with Python 3.9 or newer. The command downloads both checksum-pinned archives into the supplied cache, scores the 100 paired circuit observations, extracts the 25 hardware marker positions directly from the authors’ figure PDFs, and checks the committed outputs.

```sh
python scripts/reproduce_russo_pec.py /tmp/pec-russo --download --check
```
