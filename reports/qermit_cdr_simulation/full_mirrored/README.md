# Mirrored depolarising simulator control

Qermit's 940 mirrored circuit/observable pairs form 94 width/depth cells. Each cell contains ten targets. These circuits are constructed by reversing a circuit, so their ideal full-register Z parity is +1 up to numerical precision. An estimator that always returns +1 is a useful reference for interpreting every learned or mitigated result in these panels. The independent benchmark evidence on circuits with varying ideal expectations appears in the two unmirrored families. [Qermit benchmark](https://quantum-journal.org/papers/q-2023-07-13-1059/) · [Dataset](https://zenodo.org/records/6472281)

This is a fresh simulation using the modern Qermit/pytket implementation, the original target circuits and the authors' depolarising noise model. There are 470 Random and 470 Pauli-gadget mirror targets. For each target, CDR and each frozen regressor receive its own same 20 near-Clifford noisy/ideal pairs and the same noisy target query. Each noisy surrogate and query receives 23,809 shots, while the standalone raw estimator receives 100,000 independent noisy shots. The ideal labels use 23,809 noiseless simulator shots per surrogate. Independent archive outputs accompany the new simulation. [Modern simulation protocol](../full_unmirrored/README.md)

The maximum absolute deviation of the 940 archived ideal values from +1 was 1.98 × 10⁻¹⁴. The fresh Random raw/CDR mean absolute errors were 0.56697 and 0.49776. The archived Random raw/CDR/ZNE errors were 0.56690, 0.37891 and 0.32902. On Pauli-gadget mirrors the fresh raw/CDR errors were 0.47853 and 0.19719. Archived raw/CDR/ZNE errors were 0.48938, 0.19811 and 0.24102. The proximity warning applied to 307 Random and 127 Pauli-gadget local fits. Sixty Pauli-gadget contexts were still low-diversity after ten generation attempts. Every target was retained. [Context diagnostics](context_diagnostics.json) · [Simulator settings and checksums](protocol.json)

To repeat, use the pinned modern NumPy, Qermit and compiler versions documented for the unmirrored simulations. From the repository root run the following commands.

```sh
python scripts/audit_qermit.py /tmp/qermit-data --download --check
python scripts/run_qermit_cdr_simulator.py /tmp/qermit-data --mirrors-only --workers 8 --work /tmp/qermit-mirror-work
python scripts/run_qermit_cdr_simulator.py /tmp/qermit-data --mirrors-only --check
```
