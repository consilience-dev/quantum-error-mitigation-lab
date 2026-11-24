"""Tune an RBF kernel ridge regressor by grouped calibration-only cross-validation."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
GRID = [(width, alpha) for width in (0.25, 1., 4.) for alpha in (0.0001, 0.001, 0.01, 0.1)]


def distances(x, train, query):
    # Normalize the two feature groups to their median training-pair distance.
    d_train = np.zeros((len(train), len(train)))
    d_query = np.zeros((len(query), len(train)))
    for block in (x[:, :-4], x[:, -4:]):
        train_block = block[train]
        pair = cdist(train_block, train_block, 'sqeuclidean')
        scale = np.median(pair[np.triu_indices(len(train), k=1)])
        if scale > 0:
            d_train += pair / scale
            d_query += cdist(block[query], train_block, 'sqeuclidean') / scale
    return d_train, d_query


def predict(dist, y, width, alpha):
    d_train, d_query = dist
    kernel = np.exp(-width * d_train)
    kernel.flat[::len(kernel) + 1] += alpha
    center = y.mean(axis=0)
    coef = solve(kernel, y - center, assume_a='pos')
    return np.exp(-width * d_query) @ coef + center


def main():
    cache = np.load(ROOT / 'data/upstream/rf_features.npz')
    plan = json.loads((ROOT / 'data/clean_split.json').read_text())
    draws = [('published', None, cache['train_ids'], cache['test_ids'])]
    for r in plan['runs']:
        if r['budget'] == 500:
            draws.append(('clean', r['seed'], np.asarray(r['calibration_ids'], int), np.asarray(plan['test_ids'], int)))
    x, y = cache['X'].astype(float), cache['y'].astype(float)
    results = []
    with gzip.open(ROOT / 'reports/kernel_predictions.csv.gz', 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['split', 'seed', 'row_id'] + [f'pred_{i}' for i in range(4)])
        for split, seed, train, test in draws:
            inner = StratifiedGroupKFold(5, shuffle=True, random_state=0)
            scores = np.zeros(len(GRID))
            for a, b in inner.split(train, cache['depth'][train], groups=cache['content_hash'][train]):
                d = distances(x, train[a], train[b])
                for i, (width, alpha) in enumerate(GRID):
                    p = predict(d, y[train[a]], width, alpha)
                    scores[i] += np.linalg.norm(p - y[train[b]], axis=1).sum()
            width, alpha = GRID[np.argmin(scores)]
            p = predict(distances(x, train, test), y[train], width, alpha)
            writer.writerows((split, seed, int(i), *v) for i, v in zip(test, p))
            delta = p - cache['ideal'][test]
            results.append(dict(split=split, seed=seed, width=width, alpha=alpha,
                                calibration_cv_l2=float(scores.min()/len(train)),
                                mean_circuit_l2=float(np.linalg.norm(delta, axis=1).mean()),
                                pooled_rmse=float(np.sqrt(np.mean(delta**2)))))
    report = dict(method='RBF kernel ridge, five depth-stratified calibration folds grouped by compiled circuit',
                  grid_width_alpha=GRID, calibration_rows=500, runs=results)
    (ROOT / 'reports/kernel.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
