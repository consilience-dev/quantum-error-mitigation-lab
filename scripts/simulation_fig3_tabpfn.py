"""Run frozen TabPFN v2 on aligned ML-QEM Figure 3 no-readout steps 0 through 14."""
import argparse
import csv
import gc
import gzip
import hashlib
import io
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
from simulation_fetch import CACHE, ROOT
from simulation_fig3_metrics import metrics

OUT = ROOT / 'reports/simulation/fig3'


def verify_tensors(bundle, digests):
    for key, expected in digests.items():
        if hashlib.sha256(bundle[key].tobytes()).hexdigest() != expected:
            raise ValueError(f'Input digest mismatch for {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weight_dir', type=Path)
    parser.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    args = parser.parse_args()
    ref = json.loads((OUT/'reference.json').read_bytes())
    published = ref['qemscore_mean_circuit_mae_6dp']
    if len(published) != 6:
        raise ValueError('Expected six verified reference series')
    for source, expected in published.items():
        if abs(ref['scores'][source]['official']['mean_circuit_mae'] - expected) > 5.01e-7:
            raise ValueError(f'Published reference mismatch for {source}')
    repo, revision, filename, checkpoint_sha = CHECKPOINTS['tabpfn_v2']
    weights = args.weight_dir / filename
    with weights.open('rb') as source:
        if hashlib.file_digest(source,'sha256').hexdigest() != checkpoint_sha:
            raise ValueError('Checkpoint digest mismatch')
    with np.load(CACHE/'h12_features.npz',allow_pickle=False) as train:
        verify_tensors(train,ref['h12_training_digests'])
        x, y = train['train_X'], train['train_y']
    with np.load(CACHE/'fig3_features.npz',allow_pickle=False) as test:
        verify_tensors(test,ref['test_digests'])
        test_x, target = test['X'], test['y']
        ids, paper_ids = test['row_id'], test['paper_row_id']
        depth, novel = test['depth'], test['without_training_identity']
    if x.shape != (4500,58) or test_x.shape != x.shape or y.shape != (4500,4) or target.shape != y.shape:
        raise ValueError('Unexpected input shape')
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    started = time.monotonic()
    predictions = np.empty_like(target,dtype=np.float64)
    for obs in range(4):
        model = TabPFNRegressor(model_path=weights, device=args.device, random_state=0, n_estimators=1,
                                fit_mode='fit_with_cache', inference_precision=torch.float32)
        model.fit(x,y[:,obs])
        for i in range(0,len(test_x),100):
            predictions[i:i+100,obs] = model.predict(test_x[i:i+100])
        del model
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
        print(f'Completed observable {obs+1}/4 after {time.monotonic()-started:.1f}s',flush=True)
    scores = metrics(predictions,target,depth,novel)
    path = OUT/'tabpfn_v2.csv.gz'
    with path.open('wb') as file, gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz, io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id','paper_row_id',*[f'pred_{j}' for j in range(4)]])
        writer.writerows([i,j,*p] for i,j,p in zip(ids,paper_ids,predictions))
    report = dict(model='Frozen TabPFN v2, one ensemble member per observable',device=args.device,seed=0,
                  checkpoint_url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',checkpoint_sha256=checkpoint_sha,
                  reference_artifact_sha256=hashlib.sha256((OUT/'reference.json').read_bytes()).hexdigest(),
                  predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_rows=len(x),test_rows=len(test_x),
                  n_test_without_training_identity=int(novel.sum()),inputs=x.shape[1],fit_mode='fit_with_cache',
                  query_chunk_rows=100,inference_precision='torch.float32',one_scalar_regressor_per_observable=True,
                  versions={s:version(s) for s in ('tabpfn','torch','numpy','scikit-learn')},
                  seconds=time.monotonic()-started,scores=scores)
    (OUT/'tabpfn_v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(scores['official'],scores['without_train_identity'],flush=True)


if __name__ == '__main__':
    main()
