"""Verify the h12 scorecard from committed targets and row-level predictions without model dependencies."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'reports/simulation'


def read_csv(name, digest):
    path = OUTPUT/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'Unexpected SHA-256 in {name}')
    with gzip.open(path,'rt') as f:
        rows = list(csv.DictReader(f))
    return rows


def values(rows, prefix):
    a = np.array([[float(r[f'{prefix}_{q}']) for q in range(4)] for r in rows])
    if a.shape != (1500,4) or not np.isfinite(a).all():
        raise ValueError('Expected 1500 finite four-observable rows')
    return a


def check(pred, ideal, recorded):
    err = pred - ideal
    np.testing.assert_allclose([np.sqrt(np.mean(err**2)), np.linalg.norm(err,axis=1).mean()],
                               [recorded['pooled_rmse'],recorded['mean_circuit_l2']], rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.mean(err**2,axis=0)**.5, recorded['per_observable_rmse'], rtol=0, atol=1e-10)


def main():
    reference_file = OUTPUT/'reference.json'
    ref = json.loads(reference_file.read_bytes())
    model = json.loads((OUTPUT/'tabpfn_v2.json').read_bytes())
    if model['reference_artifact_sha256'] != hashlib.sha256(reference_file.read_bytes()).hexdigest():
        raise ValueError('TabPFN references a different evaluation')
    if ref['source_manifest_sha256'] != hashlib.sha256((ROOT/'data/simulation_h12_sources.json').read_bytes()).hexdigest():
        raise ValueError('Referenced simulation source manifest differs')
    rows = read_csv('reference.csv.gz',ref['predictions_sha256'])
    preds = read_csv('tabpfn_v2.csv.gz', model['predictions_sha256'])
    expected = [f'val/step_{d}/{i}' for d in range(15) for i in range(100)]
    if [r['row_id'] for r in rows] != expected or [r['row_id'] for r in preds] != expected:
        raise ValueError('Row IDs are missing, repeated or out of order')
    novel = np.array([r['seen_in_train']=='0' for r in rows])
    np.testing.assert_array_equal(novel,np.array([i//100 > 0 for i in range(1500)]))
    novel_hashes = {r['compiled_hash'] for r,keep in zip(rows,novel) if keep}
    if len(novel_hashes) != 1400 or int(novel.sum()) != ref['rows_without_training_identity'] or int(novel.sum()) != model['rows_without_training_identity']:
        raise ValueError('Unexpected unique held-out circuit population')
    target = values(rows,'ideal')
    for name, data in ref['metrics'].items():
        p = values(rows,name)
        check(p,target,data)
        check(p[novel],target[novel],ref['subset_without_training_identity'][name])
    p = values(preds,'pred')
    check(p,target,model['official_split'])
    check(p[novel],target[novel],model['subset_without_training_identity'])
    published = ref['original_notebook_rmse']
    if abs(ref['metrics']['raw']['pooled_rmse'] - published['RMSE_noisy']) > 1e-6:
        raise ValueError('Raw measurements do not match the source notebook')
    if abs(ref['metrics']['ols_saved']['pooled_rmse'] - published['RMSE_mitigated']) > 1e-4:
        raise ValueError('OLS replay differs from the source notebook')
    print('Verified h12 raw, refit OLS, released OLS and TabPFN on 1,500 author-split rows and 1,400 unique rows absent from training.')


if __name__ == '__main__':
    main()
