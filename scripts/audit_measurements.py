"""Audit pinned upstream numeric measurements, without loading circuit pickles."""
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class NumericUnpickler(pickle.Unpickler):
    # Only the NumPy globals present in this pinned numeric artifact.
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'),
            ('numpy.core.multiarray', 'scalar'),
            ('numpy', 'ndarray'), ('numpy', 'dtype'),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unexpected pickle global: {module}.{name}')
        return super().find_class(module, name)


def main():
    provenance = json.loads((ROOT / 'data/upstream/manifest.json').read_text())
    for record in provenance['files']:
        raw = (ROOT / record['local_path']).read_bytes()
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f"Checksum mismatch: {record['local_path']}")
    raw = (ROOT / 'data/upstream/results.pk').read_bytes()
    results = NumericUnpickler(io.BytesIO(raw)).load()
    order = json.loads((ROOT / 'data/upstream/index_order.json').read_text())
    if sorted(order) != list(range(3000)):
        raise ValueError('Shuffling map must be a 3000-row permutation')
    arrays = {}
    shapes = {}
    for key in ('ideal', 'noisy', 'zne_mitigated'):
        source = np.asarray(results[key], dtype=float)
        shapes[key] = list(source.shape)
        expected = (3000, 4) if key == 'ideal' else (60, 200)
        if source.shape != expected or not np.isfinite(source).all():
            raise ValueError(f'Invalid {key} shape or nonfinite measurements')
        source = source.reshape(3000, 4)
        unshuffled = np.empty_like(source)
        unshuffled[order] = source
        arrays[key] = unshuffled
    if np.abs(arrays['ideal']).max() > 1 + 1e-10 or np.abs(arrays['noisy']).max() > 1 + 1e-10:
        raise ValueError('Ideal/noisy observables outside physical bounds')

    depth_reports = []
    test = {key: [] for key in arrays}
    for depth in range(10):
        ids = list(range(200 * depth, 200 * (depth + 1))) + list(range(2000 + 100 * depth, 2000 + 100 * (depth + 1)))
        train_ids, test_ids = ids[:50], ids[50:]
        if set(train_ids) & set(test_ids):
            raise ValueError('Row index overlap')
        for key in arrays:
            test[key].extend(arrays[key][test_ids])
        depth_reports.append({'depth': depth, 'rows': len(ids), 'training_rows': len(train_ids), 'test_rows': len(test_ids)})
    test = {key: np.asarray(values) for key, values in test.items()}
    baselines = {}
    for key in ('noisy', 'zne_mitigated'):
        error = test[key] - test['ideal']
        baselines[key] = {
            'pooled_rmse': float(np.sqrt(np.mean(error ** 2))),
            'mean_circuit_l2': float(np.linalg.norm(error, axis=1).mean()),
        }
    report = {
        'status': 'numeric_audit_only_no_learned_model_evaluation',
        'upstream_commit': provenance['commit'],
        'results_sha256': hashlib.sha256(raw).hexdigest(),
        'numpy_version': np.__version__,
        'original_shapes': shapes,
        'shuffle_map_is_permutation': True,
        'all_measurements_finite': True,
        'observables_per_circuit': 4,
        'published_row_protocol': depth_reports,
        'published_row_protocol_test_baselines': baselines,
        'zne_values_outside_unit_interval': int(np.sum(np.abs(arrays['zne_mitigated']) > 1)),
        'canonical_circuit_identity_audited': False,
        'prior_local_180_test_split_recovered': False,
    }
    destination = ROOT / 'reports/upstream_measurement_audit.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
