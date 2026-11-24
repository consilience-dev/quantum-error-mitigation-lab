"""Fit a per-observable noisy-to-ideal affine calibration on identical circuit IDs."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT/'data/upstream/rf_features.npz')
plan = json.loads((ROOT/'data/clean_split.json').read_text())
rows = [('published', 0, data['train_ids'], data['test_ids'])]
for r in plan['runs']:
    rows.append(('clean', r['seed'], np.asarray(r['calibration_ids'],int), np.asarray(plan['test_ids'],int)))
noisy, ideal, depths = data['noisy'], data['ideal'], data['depth']
results = []
for split, seed, train, test in rows:
    coefs = []
    for q in range(4):
        design = np.column_stack((noisy[train,q], np.ones(len(train))))
        coefs.append(np.linalg.lstsq(design,ideal[train,q],rcond=None)[0])
    scale, shift = np.array(coefs).T
    err = noisy[test] * scale + shift - ideal[test]
    results.append(dict(split=split,seed=seed,budget=len(train),n_test=len(test),
                        scales=scale.tolist(),offsets=shift.tolist(),
                        pooled_rmse=float(np.sqrt(np.mean(err**2))),
                        mean_circuit_l2=float(np.linalg.norm(err,axis=1).mean())))
out = ROOT/'reports/affine.json'
out.write_text(json.dumps({'method':'Independent least-squares slope and offset per measured observable; raw output without clipping','runs':results},indent=2)+'\n')
