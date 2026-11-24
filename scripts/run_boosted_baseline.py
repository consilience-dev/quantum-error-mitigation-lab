"""Fixed-setting boosted-tree comparator for the two 500-row hardware protocols."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import catboost
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    split = ROOT / 'data/clean_split.json'
    data = np.load(ROOT / 'data/upstream/rf_features.npz')
    plan = json.loads(split.read_text())
    draws = [('published', None, data['train_ids'], data['test_ids'])]
    for run in plan['runs']:
        if run['budget'] == 500:
            draws.append(('clean', run['seed'], np.array(run['calibration_ids'],int), np.array(plan['test_ids'],int)))
    # Fixed conventional baseline, with no test-set tuning or early stopping.
    params = dict(iterations=300, depth=6, learning_rate=0.05, loss_function='RMSE',
                  random_seed=0, thread_count=2, verbose=False, allow_writing_files=False)
    out = ROOT / 'reports'
    runs = []
    with gzip.open(out/'boosted_predictions.csv.gz','wt',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split','seed','row_id', 'pred_0','pred_1','pred_2','pred_3'])
        for name,seed,train,test in draws:
            predictions = np.column_stack([
                catboost.CatBoostRegressor(**params)
                .fit(data['X'][train],data['y'][train,q])
                .predict(data['X'][test]) for q in range(4)])
            delta = predictions - data['ideal'][test]
            writer.writerows((name,seed,int(i),*p) for i,p in zip(test,predictions))
            runs.append(dict(split=name, seed=seed, mean_circuit_l2=float(np.linalg.norm(delta,axis=1).mean()),
                             pooled_rmse=float(np.sqrt(np.mean(delta**2)))))
    result = dict(catboost=catboost.__version__, parameters=params, runs=runs,
                  clean_split_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
    (out/'boosted.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
