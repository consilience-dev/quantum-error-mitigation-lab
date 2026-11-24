"""Test whether known Ising circuit parameters alone predict the ideal observables."""
import csv
import gzip
import json
import pickle

import numpy as np

from freeze_clean_split import CACHE, ROOT, SOURCES, fetch_one


def main():
    archive = np.load(CACHE / 'rf_features.npz')
    plan = json.loads((ROOT / 'data/clean_split.json').read_bytes())
    revision = json.loads((CACHE / 'manifest.json').read_bytes())['commit']
    coupling, pauli, depth = [], [], []
    for i, entry in enumerate(json.loads(SOURCES.read_bytes())):
        phase, level = ('train' if i < 10 else 'val'), i % 10
        assert entry['path'].endswith(f'/{phase}/step_{level:02d}.pk')
        rows = pickle.loads(fetch_one(entry, revision).read_bytes())
        for row in rows:
            coupling.append(float(row['J']))
            pauli.append(str(row['basis']).upper())
            depth.append(level)
    j, basis, depth = np.asarray(coupling), np.asarray(pauli), np.asarray(depth)
    np.testing.assert_array_equal(depth, archive['depth'])
    assert ''.join(pauli) == json.loads((ROOT / 'data/measurement_basis.json').read_bytes())['basis_by_unshuffled_row_id']
    draws = [('published', None, archive['train_ids'], archive['test_ids'])]
    for r in plan['runs']:
        if r['budget'] == 500:
            draws.append(('clean', r['seed'], np.asarray(r['calibration_ids'],int), np.asarray(plan['test_ids'],int)))
    out = ROOT / 'reports'
    runs = []
    with gzip.open(out / 'physics_control_predictions.csv.gz', 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'seed', 'row_id'] + [f'pred_{q}' for q in range(4)])
        for split, seed, train, test in draws:
            p = np.full((len(test),4), np.nan)
            outside = 0
            for d in np.unique(depth[test]):
                for b in np.unique(basis[test]):
                    tr = train[(depth[train] == d) & (basis[train] == b)]
                    te = (depth[test] == d) & (basis[test] == b)
                    tr = tr[np.argsort(j[tr])]
                    if len(tr) == 0:
                        raise ValueError('Missing calibration stratum')
                    coords, labels = j[tr], archive['y'][tr]
                    outside += np.count_nonzero((j[test[te]] < coords.min()) | (j[test[te]] > coords.max()))
                    for q in range(4):
                        p[te, q] = np.interp(j[test[te]], coords, labels[:,q])
            assert np.isfinite(p).all()
            writer.writerows((split, seed, int(i), *v) for i, v in zip(test, p))
            delta = p - archive['ideal'][test]
            runs.append(dict(split=split, seed=seed, endpoint_clamped_rows=int(outside),
                             mean_circuit_l2=float(np.linalg.norm(delta,axis=1).mean()),
                             pooled_rmse=float(np.sqrt(np.mean(delta**2)))))
    report = dict(method='Linear interpolation in J within each (Trotter depth, Pauli basis), constant endpoints',
                  inputs=['Trotter depth', 'Pauli measurement basis', 'Ising coupling J'],
                  calibration_rows=500, upstream_revision=revision, runs=runs)
    (out/'physics_control.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
