# Learning-based PEC on three IBM processors

Strikis et al. measured the upper-qubit Z expectation of a two-qubit circuit at ten rotation angles on Yorktown, Ourense and Santiago. Figure 10 in *PRX Quantum* shows the ideal cosine, the uncorrected hardware result and the authors’ learned probabilistic error cancellation (PEC) result. The paper calls this a deterministic quantum computation with pure states (DQCp) circuit. [Paper, section X.A and Figures 9–10](https://doi.org/10.1103/PRXQuantum.2.040330) · [Published PDF, page 12](https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040330#page=12)

## What the figure shows

We recovered the centers of all 60 published markers from the journal PDF and repeated the extraction from the authors’ figure source. Each device has ten raw and ten corrected expectations. The greatest difference between versions was 0.000026. Both versions display the same experiment. The table gives our equal-weight mean absolute error (MAE) relative to the ideal cosine. These are descriptive summaries of the figure. Refitting the authors’ estimator requires the underlying measurements.

| Processor | Raw MAE | Learned PEC MAE | Phase settings with lower error |
|---|---:|---:|---:|
| IBM Yorktown | 0.1023 | 0.0153 | 8/10 |
| IBM Ourense | 0.0462 | 0.0219 | 7/10 |
| IBM Santiago | 0.0393 | 0.0113 | 9/10 |
| All three processors | 0.0626 | 0.0162 | 24/30 |

The [figure values and source fingerprints](figure_scores.json) retain all ten phases, the axis checks and the analytic reference. To repeat the extraction, download the [APS journal PDF](https://harvest.aps.org/v2/journals/articles/10.1103/PRXQuantum.2.040330/fulltext) and [arXiv source archive](https://arxiv.org/src/2005.07601v2). Run from the repository root with pdfplumber 0.11.9.

```sh
python scripts/reproduce_strikis_pec.py strikis-journal.pdf strikis-source.tar.gz --check
```

## Which measurements are available?

The public figure provides an independent check on the endpoints of a future replay. The arXiv source contains the TeX and 19 figure PDFs. Oxford’s article deposit provides the journal PDF. We found no underlying calibration and test tables in those deposits or in searches by title, author and DOI in DataCite, Zenodo and GitHub. An unindexed or privately held copy could still exist. [Authors’ source archive](https://arxiv.org/src/2005.07601v2) · [Oxford deposit](https://ora.ox.ac.uk/objects/uuid%3Ae50706cc-c1fc-4568-b0b0-7ba110a22fe2)

Strikis’ doctoral thesis covers the related numerical simulations in Chapter 3. We checked the full thesis and its embedded-file inventory for a separate release of the hardware records. The thesis contains no attached data files. [Oxford thesis, Chapter 3](https://ora.ox.ac.uk/objects/uuid:3332385b-4b31-4ef9-aaa2-4baa99aadc56/files/dks65hd09f)

| Per device | Paper’s acquisition | Values recoverable from Figure 10 |
|---|---|---|
| Calibration | 24 upper-qubit Z measurements, one for each single-qubit Clifford with identity insertion | None of the 24 calibration observations |
| Test | Ten phases × four insertions (I, X, Y, Z), with 8,192 shots for each of the 40 circuits | Ten uncorrected I-insertion estimates at figure precision |
| Published output | At each phase, a weighted sum of the four hardware measurements plus a fitted offset | Ten corrected estimates at figure precision |

Even with the fitted weights, the corrected output at one phase gives only one equation for the three missing X, Y and Z responses. The plot therefore cannot supply the four inputs for a fair model comparison. A data request can be fulfilled with the original counts or per-circuit expectation values, the device, the preparation or test identifiers, and a convention identifying the measured qubit and each Clifford. An existing public link or original file format is sufficient. The corresponding author listed by APS is Ying Li. [Journal author record](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040330)

## Comparison when the records become available

1. Work one device at a time. Preserve the deposited files and their checksums and record the permitted redistribution terms. For counts, recover the upper-qubit expectation as the difference between the counts of zero and one divided by the shot count. The paper reports 8,192 shots for every executed circuit.
2. Build the four features for each of the 24 calibration rows by looking up the measured Clifford right-products with I, X, Y and Z, ignoring global phase. The paper reuses its 24 measured responses to form this table. From the Figure 9 circuit, the ideal label for a Clifford C is the X expectation in C|+⟩. Over the complete group the labels comprise four +1, four −1 and sixteen zeros.
3. Fit the published least-squares rule with four signed coefficients and an intercept on these 24 calibration rows. Keep the four separately measured inputs at each test phase. Check the uncorrected and corrected estimates against the plots, record the matrix rank and conditioning, and resolve discrepancies before interpreting model rankings.
4. Give frozen Nori 1.5, TabICLv2 and TabPFN v2 the identical 24 labeled rows and identical four test features, separately per device. Use the checkpoint revisions from the existing hardware PEC comparison. Reserve the phase and its known cosine for scoring. The ten angles are 2πm/10 for m = 0 through 9. Report all ten estimates and absolute errors per device. Show the eight non-Clifford angles separately to reveal the influence of the two test rotations that also occur among the calibration gates. Each device repeats the same ten angles. The three devices do not supply 30 distinct circuit settings. [Frozen checkpoint settings](../pec_song/README.md#comparison-protocol)
