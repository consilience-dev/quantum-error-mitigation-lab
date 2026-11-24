# Frozen regressors on Song et al.’s hardware PEC circuits

Song et al. used gate-set tomography to calibrate quasiprobabilistic error cancellation on a superconducting processor. Their one-qubit benchmark corrects readout error. Their two-qubit benchmark also corrects a controlled-phase gate. We use the original hardware calibration and test-circuit records from the figures at their released precision. [Science Advances paper](https://doi.org/10.1126/sciadv.aaw5686) · [Authors’ figure source](https://arxiv.org/abs/1812.10903)

## Four two-qubit circuits with different ideal answers

At phase π/2 the authors report an absolute error of 0.1690 before correction and 0.0102 with PEC. Reading the centers of the published error bars gives 0.16894 and 0.01023. The table reports absolute error in the X expectation of qubit 1 for every phase in Figure 3G. The MAE is our equal-weight average of those four errors. Smaller is better. [Reconstructed hardware results and independent figure check](two_qubit_targets.json)

| Method | π/4 | π/2 | 3π/4 | π | MAE |
|---|---:|---:|---:|---:|---:|
| Uncorrected hardware | 0.0718 | 0.1689 | 0.1880 | 0.1077 | 0.1341 |
| Hardware PEC | 0.0006 | 0.0102 | 0.0406 | 0.0210 | **0.0181** |
| Affine readout correction | 0.0231 | 0.1189 | 0.1751 | 0.1014 | 0.1046 |
| Frozen Nori 1.5 | 0.0948 | 0.1937 | 0.0858 | 0.0670 | 0.1103 |
| Frozen TabICLv2 | 0.1305 | 0.2014 | 0.0903 | 0.3102 | 0.1831 |
| Frozen TabPFN v2 | 0.0834 | 0.1403 | 0.1465 | 0.0106 | 0.0952 |

The last two raw expectations fall below the smallest X calibration measurement (−0.0190). All four phases come from one gate family on one processor. The published error bars describe variability across 100 hardware acquisitions at each phase. The release provides their means and standard deviations. It permits scoring a regressor once on each pooled raw mean. Individual block measurements would be needed to estimate how often a method wins across acquisitions or to score a nonlinear regressor on each block separately.

We checked sensitivity to recovering numbers from a figure. We used two renderings of the Q1 calibration and shifted all four raw means by −0.0001, zero or +0.0001 for each rendering. Across those six variants the MAE ranges were 0.10457–0.10468 for affine, 0.09431–0.11597 for Nori, 0.18308–0.18324 for TabICL and 0.07467–0.18319 for TabPFN. Hardware PEC had the lowest aggregate error throughout. Even TabPFN’s improvement over raw depends on which rendering and input precision are used. These perturbations probe digitization sensitivity only. [Affine](affine.json) · [Nori](nori.json) · [TabICL](tabicl.json) · [TabPFN](tabpfn_v2.json)

## One-qubit readout control

Both qubits have ideal Z expectation zero for the test preparation. The authors give a signed mean and standard deviation across 100 acquisitions. For the models the archive permits a range for the signed mean, using the published raw histogram and the reported raw mean rounded to three decimal places. The endpoints below round outward. The bounds cover the corrected mean across blocks, conditional on the digitized calibration. [Published supplement, Figure S3](https://pmc-oa-opendata.s3.amazonaws.com/PMC6731091.1/aaw5686_SM.pdf) · [Recovered histograms](single_qubit_inputs.json)

| Method | Q1 | Q2 |
|---|---:|---:|
| Uncorrected hardware, mean ± SD | 0.027 ± 0.016 | 0.034 ± 0.019 |
| Hardware PEC, mean ± SD | 0.003 ± 0.022 | 0.007 ± 0.024 |
| Affine, possible mean | [0.0081, 0.0093] | [0.0078, 0.0091] |
| Frozen Nori 1.5, possible mean | [−0.0431, 0.0202] | [−0.0319, 0.0006] |
| Frozen TabICLv2, possible mean | [0.0137, 0.0493] | [0.0176, 0.0357] |
| Frozen TabPFN v2, possible mean | [−0.0007, 0.0015] | [−0.0050, 0.0295] |

## Comparison protocol

1. Use the paper’s four state preparations per qubit. For a measured axis, use its four hardware expectation values from the Gram matrix as the single input and its four known ideal values from the paper’s state-preparation matrix as the label. Fit a separate readout correction for each qubit and axis. This estimator consumes the row for the measured axis. The authors’ PEC can use cross-axis measurements.
2. Run the three frozen checkpoints already used in the Qermit comparison, with the same checkpoint revisions, ensemble requests and seed. Each predict call receives a noisy expectation. Fit only on the four calibrations. Include a straight-line least-squares correction on those same four points. Apply no clipping, hyperparameter search, test-angle features or test-label calibration.
3. For the one-qubit experiment, retain the authors’ 100 blocks of 3,000 shots on each qubit. The released histogram groups the blocks into bins. Enumerate the allowed shot counts within each occupied raw bin, apply each regressor and bound the mean of the 100 corrected estimates using the bin minima, maxima and multiplicities. These are bounds from the published histogram, conditional on the digitized calibration, and do not represent confidence intervals. The test preparation has the same nominal state as the fourth calibration. This tests readout correction on repeated acquisitions of that state.
4. For the four two-qubit controlled-phase settings, use Q1’s X calibration to predict from the reported raw mean X at each phase. Report the paper’s PEC and raw absolute error for the same four circuits and the per-setting absolute error of each readout correction. Each setting represents one million hardware shots per arm. The model acts on the published pooled mean. The paper’s full PEC also characterizes and cancels two-qubit gate error, and the release shows only an example of those gate calibrations. This comparison tests the information in the common readout calibration. A learned replacement with all the gate-characterization inputs remains open.

The four calibration states are the labeled examples for each readout fit. The hardware costs of estimating the full calibration matrices and of characterizing each gate are additional to the stated test shots. The raw and PEC arms were acquired separately. No simulator calibration or additional supervised circuits enter the models.

## Reproduction

Download the [source archive](https://arxiv.org/src/1812.10903v1), [publisher supplement](https://pmc-oa-opendata.s3.amazonaws.com/PMC6731091.1/aaw5686_SM.pdf) and [open-access journal PDF](https://rcastoragev2.blob.core.windows.net/579b2353fdfbedf6bf0f02e0777ca90c/aaw5686.PMC6731091.pdf). Song et al. published the original article and figures under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). We transcribed the calibration bars and histogram bins and reconstructed the target means and standard deviations. The extraction scripts verify the hashes of the source documents. We checked the figures with NumPy 2.3.5, pdfplumber 0.11.9 and pypdf 6.21.4. The supplement and preprint figure S3 have identical drawing instructions. Two independent versions of Figure 3G give means within 0.000016. [Publication verification](publication_check.json)

```sh
python scripts/reproduce_song_calibrations.py song-source.tar.gz --published-supplement aaw5686_SM.pdf --check
python scripts/extract_song_two_qubit.py song-journal.pdf song-source.tar.gz --check
```

Run from the repository root. Use the existing Nori/TabICL and TabPFN environments for their respective commands, with the checkpoint files in `weights`. Each model JSON lists the package versions and the exact checkpoint URL and checksum. Each learned model makes one prediction at a time. The runner repeats the fit and verifies identical predictions when it visits the queries in reverse order. [Model environments](../../docs/running-models.md)

```sh
python scripts/song_readout_models.py affine
python scripts/song_readout_models.py nori --weights weights
python scripts/song_readout_models.py tabicl --weights weights
python scripts/song_readout_models.py tabpfn_v2 --weights weights
```

To verify the committed predictions, bounds and all six sensitivity scores using NumPy alone, append `--check` to each model command. Check mode uses the saved prediction files and source fingerprints.
