"""Align ML-QEM Figure 3 archived model outputs and no-readout test inputs for steps 0–14."""
import csv
import gzip
import hashlib
import io
import json
import pickle
from importlib.metadata import version
from typing import Any

import numpy as np
import torch
from qiskit.providers import BackendV1
from qiskit.providers.fake_provider import FakeLima

from reproduce_rf import circuit_features
from simulation_fetch import CACHE, ROOT, verified
from simulation_fig3_fetch import MANIFEST
from simulation_fig3_metrics import metrics
from simulation_h12 import SUBDIR, functions

OUT = ROOT / 'reports/simulation/fig3'
COLUMNS = ('noisy', 'zne', 'ols_full', 'rfr_list', 'mlp', 'gnn')
PUBLISHED_MAE = dict(noisy=.060945, zne=.041865, ols_full=.046249, rfr_list=.015510, mlp=.022274, gnn=.051041)


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    sources = json.loads(MANIFEST.read_bytes())
    h12_manifest = json.loads((ROOT/'data/simulation_h12_sources.json').read_bytes())
    if sources['revision'] != h12_manifest['revision']:
        raise ValueError('Different upstream revisions for training and test')
    code = [entry for entry in h12_manifest['files'] if entry['path'].endswith(('utils.py','mlp.py'))]
    for source in sources['files'] + code:
        if not verified(CACHE/source['path'], source):
            raise ValueError(f"Missing or changed source: {source['path']}")
    h12 = json.loads((ROOT/'reports/simulation/reference.json').read_bytes())
    train = np.load(CACHE/'h12_features.npz', allow_pickle=False)
    for key in ('train_X','train_y','train_hash','train_id'):
        if digest(train[key]) != h12['feature_digests'][key]:
            raise ValueError(f"Unverified training inputs: {key}")
    frame = pickle.loads((CACHE/'docs/paper_figures/no_readout_over_depths.pk').read_bytes())['df']
    if len(frame) != 9000 or not frame.index.is_unique:
        raise ValueError('Unexpected Figure 3 archive')
    scope = dict(np=np, torch=torch, BackendV1=BackendV1, Any=Any)
    functions(CACHE/'blackwater/data/utils.py',['gate_to_index','get_backend_properties_v1'],scope)
    functions(CACHE/'docs/tutorials/mlp.py',['count_gates_by_rotation_angle','recursive_dict_loop','encode_data'],scope)
    props = scope['get_backend_properties_v1'](FakeLima())
    edges = np.arange(-2*np.pi, 2*np.pi+0.1*np.pi, 0.1*np.pi)
    rows, ids, hashes, xs, targets = [], [], [], [], []
    for depth in range(15):
        rec = pickle.loads((CACHE/SUBDIR/'val_extra'/f'step_{depth}.pk').read_bytes())
        block = frame[frame['step'] == depth]
        if len(rec) != len(block) or len(rec) != 300:
            raise ValueError(f'Unexpected population for depth {depth}')
        circuits = [r['circuit'] for r in rec]
        ideal = [r['ideal_exp_value'] for r in rec]
        noisy = [r['noisy_exp_values'][0] for r in rec]
        # Upstream traversed files in filesystem order but preserved each file's row order.
        # Check all eight data values for every row before associating any prediction.
        for column, values in [('ideal',ideal), ('noisy',noisy)]:
            archived = np.stack([np.ravel(v) for v in block[column]])
            np.testing.assert_array_equal(np.asarray(values), archived)
        x, _ = scope['encode_data'](circuits, props, ideal, noisy, num_qubits=4)
        xs.append(x.numpy())
        targets.extend(ideal)
        rows.extend(block.index.tolist())
        ids.extend(f'val_extra/step_{depth}/{i}' for i in range(len(rec)))
        hashes.extend(circuit_features(c, edges)[1] for c in circuits)
    if len(set(rows)) != 4500 or set(rows) != set(frame.index[frame['step'] < 15]):
        raise ValueError('Unmatched paper prediction rows')
    frame = frame.loc[rows]
    target = np.asarray(targets)
    X, hashes, ids = np.concatenate(xs), np.asarray(hashes), np.asarray(ids)
    depth = frame['step'].to_numpy()
    if X.shape != (4500,58) or not np.isfinite(X).all():
        raise ValueError('Unexpected test matrix')
    novel = ~np.isin(hashes, train['train_hash'])
    predictions = {k:np.stack([np.ravel(v) for v in frame[k]]) for k in COLUMNS}
    scores = {k:metrics(p, target, depth, novel) for k,p in predictions.items()}
    for name in COLUMNS:
        np.testing.assert_allclose(scores[name]['official']['mean_circuit_mae'], PUBLISHED_MAE[name], rtol=0, atol=5.01e-7)
        np.testing.assert_allclose(np.linalg.norm(predictions[name]-target,axis=1), frame['L2_'+name], rtol=0, atol=1e-7)
    OUT.mkdir(parents=True,exist_ok=True)
    bundle=dict(X=X, y=target, row_id=ids, depth=depth, compiled_hash=hashes,
                paper_row_id=np.asarray(rows), without_training_identity=novel)
    np.savez_compressed(CACHE/'fig3_features.npz', **bundle)
    path = OUT/'reference.csv.gz'
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz, io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:
        columns = {'ideal':target, **predictions}
        writer = csv.writer(f)
        writer.writerow(['row_id','paper_row_id','compiled_hash','seen_in_train'] +
                        [f'{name}_{q}' for name in columns for q in range(4)])
        for i, ident in enumerate(ids):
            writer.writerow([ident,rows[i],hashes[i],int(not novel[i]),*[v for a in columns.values() for v in a[i]]])
    report = dict(source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  upstream_revision=sources['revision'], qemscore_reference_revision=sources['qemscore_reference_revision'],
                  ground_truth='Archived ideal simulation measurement at 10,000 shots per circuit, as scored by the authors',
                  dataset='FakeLima simulation, no readout error, published Figure 3, steps 0 to 14',
                  training='Identical train rows and input hashes to the h12 tutorial, in depth/source-file order',
                  row_alignment='Every archived raw and ideal four-vector matches the original circuit record exactly in per-depth file order. Each paper row is used once.',
                  scope='Within-depth Figure 3 subset; steps 15 to 29 are outside this comparison.',
                  n_train=len(train['train_X']), n_test=len(X), unique_test_hashes=len(set(hashes)),
                  overlap_hashes=len(set(hashes) & set(train['train_hash'])), n_test_without_training_identity=int(novel.sum()),
                  train_overlap_rows_by_depth={int(d):int((~novel & (depth == d)).sum()) for d in np.unique(depth)},
                  h12_training_digests={k:h12['feature_digests'][k] for k in ('train_X','train_y','train_hash','train_id')},
                  test_digests={k:digest(v) for k,v in bundle.items()}, predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  versions={p:version(p) for p in ('qiskit-terra','numpy','torch','pandas')},
                  qemscore_mean_circuit_mae_6dp=PUBLISHED_MAE, scores=scores)
    (OUT/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v['official'] for k,v in scores.items()}, flush=True)


if __name__ == '__main__':
    main()
