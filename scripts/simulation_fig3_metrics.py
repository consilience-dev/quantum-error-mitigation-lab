"""Metrics for matched ML-QEM Figure 3 rows and archived four-observable predictions."""
import numpy as np


def metrics(prediction, target, depth, novel):
    delta = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    if delta.shape != (4500, 4) or not np.isfinite(delta).all():
        raise ValueError('Expected finite errors on 4,500 four-observable test vectors')
    circuit_l2 = np.linalg.norm(delta, axis=1)
    circuit_mae = np.abs(delta).mean(axis=1)
    def scores(mask):
        return dict(mean_circuit_mae=float(circuit_mae[mask].mean()),
                    mean_circuit_l2=float(circuit_l2[mask].mean()),
                    pooled_rmse=float(np.mean(delta[mask] ** 2)**0.5))
    return dict(official=scores(np.ones(len(delta), dtype=bool)), without_train_identity=scores(novel),
                by_depth={int(d):scores(depth == d) for d in np.unique(depth)})
