"""Evaluate multivariate measurement OLS and full-feature linear ridge on frozen ML-QEM splits."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PENALTIES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.)
MEASUREMENT_MODEL = 'Multivariate affine'
FULL_MODEL = 'Full-feature linear ridge'


def predict_ridge(x_train, y_train, x_test, penalty):
    scale = StandardScaler().fit(x_train)
    model = Ridge(alpha=penalty * len(y_train), solver='cholesky')
    model.fit(scale.transform(x_train), y_train)
    return model.predict(scale.transform(x_test))


def grouped_ridge(x, y, depth, group, train, test):
    _, counts = np.unique(depth[train], return_counts=True)
    folds = min(5, int(counts.min()))
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=0)
    errors = np.zeros(len(PENALTIES))
    held_out = []
    for a, b in splitter.split(train, depth[train], group[train]):
        fit, val = train[a], train[b]
        if np.intersect1d(group[fit], group[val]).size:
            raise ValueError('Repeated circuit in a ridge training and validation fold')
        held_out.extend(val.tolist())
        for i, penalty in enumerate(PENALTIES):
            p = predict_ridge(x[fit], y[fit], x[val], penalty)
            errors[i] += np.linalg.norm(p - y[val], axis=1).sum()
    if sorted(held_out) != sorted(train.tolist()):
        raise ValueError('A calibration row has missing or repeated fold predictions')
    penalty = PENALTIES[np.argmin(errors)]
    return predict_ridge(x[train], y[train], x[test], penalty), {
        'penalty': penalty, 'folds': folds, 'calibration_cv_mean_circuit_l2': float(errors.min() / len(train)),
    }


def main():
    cache_file = ROOT / 'data/upstream/rf_features.npz'
    split_file = ROOT / 'data/clean_split.json'
    data = np.load(cache_file, allow_pickle=False)
    plan = json.loads(split_file.read_bytes())
    draws = [('published', None, data['train_ids'], data['test_ids'])]
    for row in plan['runs']:
        draws.append(('clean', row['seed'], np.asarray(row['calibration_ids'],int),
                      np.asarray(plan['test_ids'],int)))
    x = data['X'].astype(float)
    y = data['y'].astype(float)
    ideal, depth, groups = data['ideal'], data['depth'], data['content_hash']
    # Original per-observable affine uses the released full-precision measurements
    # and ideal values. Preserve its inputs and labels for the nested 4x4 comparison.
    design = np.column_stack([data['noisy'], np.ones(len(ideal))])
    results = []
    out = ROOT / 'reports'
    table = out / 'linear_comparators.csv.gz'
    with table.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['split','seed','budget','model','row_id'] + [f'pred_{i}' for i in range(4)])
            for split, seed, train, test in draws:
                if np.intersect1d(train, test).size or len(set(train)) != len(train):
                    raise ValueError('Unexpected duplicate or overlap in row IDs')
                coef, _, rank, _ = np.linalg.lstsq(design[train], ideal[train], rcond=None)
                if rank != design.shape[1]:
                    raise ValueError('Rank-deficient multivariate measurement design')
                ridge, settings = grouped_ridge(x, y, depth, groups, train, test)
                for name, p, meta in [
                    (MEASUREMENT_MODEL, design[test] @ coef, {'weights_and_intercepts': coef.tolist()}),
                    (FULL_MODEL, ridge, settings),
                ]:
                    if not np.isfinite(p).all():
                        raise ValueError('Nonfinite test predictions')
                    err = p - ideal[test]
                    writer.writerows((split, seed, len(train), name, int(i), *v) for i,v in zip(test,p))
                    results.append(dict(model=name, split=split, seed=seed, budget=len(train), n_test=len(test),
                                        mean_circuit_l2=float(np.linalg.norm(err, axis=1).mean()),
                                        pooled_rmse=float(np.sqrt(np.mean(err**2))), **meta))
    report = {
        'measurement_model': 'Unrestricted multioutput least squares on all four raw noisy observables plus an intercept. Matrix rows are the four measurements then the intercept; columns are the four outputs.',
        'full_model': 'Ridge regression on the 169 float32 upstream features and four float32 training targets, cast to float64. Center/scale on inner training rows only. Intercept unpenalized. Select penalty by grouped depth-stratified out-of-fold mean circuit L2, then refit on all calibration rows. No output clipping.',
        'penalty_grid': PENALTIES, 'objective': 'sum squared error plus n_train * penalty * squared coefficient norm',
        'folds': 'Up to five folds with random_state=0, limited by the minimum depth count. Groups use compiled circuit fingerprints.',
        'protocol': 'Exploratory locally fitted linear controls on frozen published and fresh splits.',
        'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
        'clean_split_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
        'feature_cache_sha256': hashlib.sha256(cache_file.read_bytes()).hexdigest(),
        'predictions_sha256': hashlib.sha256(table.read_bytes()).hexdigest(), 'runs': results,
    }
    (out/'linear_comparators.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
