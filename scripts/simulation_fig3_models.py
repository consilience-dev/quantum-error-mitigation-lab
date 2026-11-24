"""Run frozen Nori 1.5 or TabICLv2 on the authors' full Figure 3 within-depth split."""
import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from frozen_regressors import checkpoint, check_full_context, regressor
from simulation_fetch import CACHE, ROOT
from simulation_fig3_metrics import metrics

OUT = ROOT/'reports/simulation/fig3'


def verified_arrays(path, digests):
    with np.load(path, allow_pickle=False) as data:
        arrays = {k:data[k] for k in digests}
    for key, arr in arrays.items():
        if hashlib.sha256(arr.tobytes()).hexdigest() != digests[key]:
            raise ValueError(f'Unexpected Figure 3 array: {key}')
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['nori','tabicl'])
    parser.add_argument('weights', type=Path)
    parser.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    parser.add_argument('--chunk', type=int, default=250)
    parser.add_argument('--check-prefix', type=int, help='Also predict this many initial queries separately for the first observable')
    args = parser.parse_args()
    if args.check_prefix is not None and not 0 < args.check_prefix < min(4500,args.chunk):
        parser.error('Use fewer prefix rows than the regular inference chunk')
    if args.chunk <= 0:
        parser.error('Query batch size must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    weights, info = checkpoint(args.model,args.weights,args.device)
    manifest = (OUT/'reference.json').read_bytes()
    ref = json.loads(manifest)
    train = verified_arrays(CACHE/'h12_features.npz', ref['h12_training_digests'])
    test = verified_arrays(CACHE/'fig3_features.npz', ref['test_digests'])
    X, y = train['train_X'], train['train_y']
    query = test['X']
    if X.shape != (4500,58) or query.shape != X.shape or y.shape != (4500,4):
        raise ValueError('Expected the original 4500/4500 split with 58 input features')
    started = time.monotonic()
    pred = np.empty((len(query),4))
    sensitivity = None
    for q in range(4):
        fitted = regressor(args.model,weights,args.device)
        fitted.fit(X,y[:,q])
        for i in range(0,len(query),args.chunk):
            pred[i:i+args.chunk,q] = fitted.predict(query[i:i+args.chunk])
            effective = check_full_context(args.model,fitted,len(X))
            if effective != info['requested_ensemble_members']:
                raise ValueError('Unexpected effective Figure 3 ensemble size')
            print(f'{args.model}, observable {q+1}/4, {min(i+args.chunk,len(query))}/{len(query)}, {time.monotonic()-started:.0f}s',flush=True)
        if q == 0 and args.check_prefix:
            baseline = fitted.predict(query[:args.check_prefix])
            check_full_context(args.model,fitted,len(X))
            delta = np.abs(baseline - pred[:args.check_prefix,q])
            sensitivity = dict(observable=0, prefix_rows=args.check_prefix,
                               maximum_absolute_prediction_difference=float(delta.max()),
                               mean_absolute_prediction_difference=float(delta.mean()))
        del fitted
        gc.collect()
        if args.device == 'mps':
            torch.mps.empty_cache()
    scores = metrics(pred,test['y'],test['depth'],test['without_training_identity'])
    output = OUT/f'{args.model}.csv.gz'
    with output.open('wb') as file, gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_id','paper_row_id',*[f'pred_{i}' for i in range(4)]])
            writer.writerows((row,paper,*p) for row,paper,p in zip(test['row_id'],test['paper_row_id'],pred))
    report = dict(**info,train_rows=len(X),test_rows=len(query),inputs=X.shape[1],
                  n_test_without_training_identity=int(test['without_training_identity'].sum()),
                  full_context_verified_each_query_batch=True, ensemble_members=effective, one_scalar_regressor_per_observable=True,
                  preprocessing=('Released Nori configuration with training-only high-dimensional projection.'
                                 if args.model == 'nori' else 'Released TabICLv2 regression preprocessing with cached context.'),
                  reference_artifact_sha256=hashlib.sha256(manifest).hexdigest(), query_chunk_rows=args.chunk,
                  predictions_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), seconds=time.monotonic()-started,scores=scores)
    if sensitivity is not None:
        report['query_batch_sensitivity'] = sensitivity
    (OUT/f'{args.model}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(scores['official'],flush=True)


if __name__ == '__main__':
    main()
