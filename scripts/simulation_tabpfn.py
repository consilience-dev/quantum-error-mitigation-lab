"""Evaluate frozen TabPFN v2 on the complete FakeLima h12 tutorial after checking its reference."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import torch
from tabpfn import TabPFNRegressor

from run_tabpfn_tabdpt import CHECKPOINTS
from simulation_fetch import ROOT, CACHE

OUTPUT = ROOT / 'reports/simulation'


def score(y, p):
    err = p.astype(float) - y.astype(float)
    return dict(pooled_rmse=float(np.mean(err**2)**.5), mean_circuit_l2=float(np.linalg.norm(err,axis=1).mean()),
                per_observable_rmse=np.sqrt(np.mean(err**2,axis=0)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weight_dir', type=Path)
    parser.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    args = parser.parse_args()
    ref = json.loads((OUTPUT/'reference.json').read_bytes())
    published = ref['original_notebook_rmse']
    metrics = ref['metrics']
    if abs(published['RMSE_noisy'] - metrics['raw']['pooled_rmse']) > 1e-6:
        raise ValueError('Upstream raw reference has not been reproduced')
    if min(abs(published['RMSE_mitigated']-metrics[k]['pooled_rmse']) for k in ['ols_refit','ols_saved']) > 1e-4:
        raise ValueError('Upstream OLS reference has not been reproduced')
    repo, revision, name, expected = CHECKPOINTS['tabpfn_v2']
    weights = args.weight_dir / name
    with weights.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest() == expected
    with np.load(CACHE/'h12_features.npz', allow_pickle=False) as z:
        for key, digest in ref['feature_digests'].items():
            if hashlib.sha256(z[key].tobytes()).hexdigest() != digest:
                raise ValueError(f'Feature data changed: {key}')
        x, y = z['train_X'], z['train_y']
        test, target, ids = z['val_X'], z['val_y'], z['val_id']
        novel = ~np.isin(z['val_hash'],z['train_hash'])
    assert x.shape == (4500,58) and test.shape == (1500,58)
    assert y.shape == (4500,4) and target.shape == (1500,4)
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    predictions = np.empty((len(test), 4))
    start = time.monotonic()
    for q in range(4):
        model = TabPFNRegressor(model_path=weights, device=args.device, random_state=0, n_estimators=1,
                                fit_mode='fit_with_cache', inference_precision=torch.float32)
        model.fit(x, y[:,q])
        predictions[:,q] = np.concatenate([model.predict(test[i:i+100]) for i in range(0,len(test),100)])
        del model
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
        print(f'Completed observable {q+1}/4 after {time.monotonic()-start:.1f}s', flush=True)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite prediction')
    output = OUTPUT/'tabpfn_v2.csv.gz'
    with gzip.open(output, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id']+[f'pred_{i}' for i in range(4)])
        writer.writerows((i,*v) for i,v in zip(ids,predictions))
    report = dict(model='TabPFN v2, one frozen ensemble member', seed=0, device=args.device,
                  checkpoint_url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}', checkpoint_sha256=expected,
                  reference_artifact_sha256=hashlib.sha256((OUTPUT/'reference.json').read_bytes()).hexdigest(),
                  predictions_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  train_rows=len(x), test_rows=len(test), inputs=x.shape[1], one_scalar_regressor_per_observable=True,
                  query_chunk_rows=100, seconds=time.monotonic()-start, fit_mode='fit_with_cache',
                  versions={name:version(name) for name in ['tabpfn','torch','numpy','scikit-learn']},
                  official_split=score(target,predictions), rows_without_training_identity=int(novel.sum()),
                  subset_without_training_identity=score(target[novel],predictions[novel]))
    (OUTPUT/'tabpfn_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['official_split'],flush=True)


if __name__=='__main__':
    main()
