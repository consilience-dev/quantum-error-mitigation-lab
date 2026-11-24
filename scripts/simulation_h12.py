"""Rebuild the FakeLima simulation tutorial's exact 4500/1500 circuit splits and OLS scores."""
import ast
import csv
import gzip
import hashlib
import io
import json
import pickle
import re
from importlib.metadata import version
from typing import Any

import numpy as np
import torch
from qiskit.providers import BackendV1
from qiskit.providers.fake_provider import FakeLima
from sklearn.linear_model import LinearRegression

from reproduce_rf import circuit_features
from simulation_fetch import ROOT, CACHE, MANIFEST, verified

OUTPUT = ROOT / 'reports/simulation'
SUBDIR = 'docs/tutorials/data/ising_init_from_qasm_no_readout'


def functions(path, names, scope):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in tree.body} != set(names):
        raise ValueError(f'Unexpected function set in {path}')
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def metrics(pred, target):
    diff = pred.astype(float) - target.astype(float)
    return dict(pooled_rmse=float(np.mean(diff ** 2) ** .5),
                per_observable_rmse=np.sqrt(np.mean(diff ** 2, axis=0)).tolist(),
                mean_circuit_l2=float(np.linalg.norm(diff, axis=1).mean()))


def targets_from_notebook():
    nb = json.loads((CACHE / 'docs/tutorials/h12_ols.ipynb').read_bytes())
    outputs = [''.join(out.get('text', '')) for cell in nb['cells'] for out in cell.get('outputs', [])]
    lines, = [t for t in outputs if 'RMSE_noisy_0:' in t and 'RMSE_mitigated:' in t]
    return {name: float(v) for name, v in re.findall(r'^(RMSE_[\w]+): ([.0-9]+)$', lines, re.M)}


def main():
    manifest = json.loads(MANIFEST.read_bytes())
    for src in manifest['files']:
        if not verified(CACHE / src['path'], src):
            raise ValueError(f"Missing or invalid {src['path']}; run simulation_fetch.py first")
    env = {'np':np, 'torch':torch, 'BackendV1':BackendV1, 'Any':Any}
    functions(CACHE/'blackwater/data/utils.py', ['gate_to_index','get_backend_properties_v1'], env)
    functions(CACHE/'docs/tutorials/mlp.py', ['count_gates_by_rotation_angle','recursive_dict_loop','encode_data'], env)
    props = env['get_backend_properties_v1'](FakeLima())
    refs = targets_from_notebook()
    bundle = {}
    group_sets = {}
    edges = np.arange(-2*np.pi, 2*np.pi+0.1*np.pi, 0.1*np.pi)
    for part, per_depth in [('train',300), ('val',100)]:
        xs, ys, ids, hashes = [], [], [], []
        for depth in range(15):
            path = CACHE / SUBDIR / part / f'step_{depth}.pk'
            records = pickle.loads(path.read_bytes())
            if len(records) != per_depth:
                raise ValueError(f'Unexpected row count in {path}')
            circuits = [r['circuit'] for r in records]
            noisy = [r['noisy_exp_values'][0] for r in records]
            y = [r['ideal_exp_value'] for r in records]
            x, targets = env['encode_data'](circuits, props, y, noisy, num_qubits=4)
            xs.append(x.numpy()); ys.append(targets.numpy())
            ids.extend(f'{part}/step_{depth}/{i}' for i in range(len(y)))
            hashes.extend(circuit_features(c, edges)[1] for c in circuits)
        bundle[part+'_X'] = np.concatenate(xs)
        bundle[part+'_y'] = np.concatenate(ys)
        bundle[part+'_id'] = np.asarray(ids)
        bundle[part+'_hash'] = np.asarray(hashes)
        group_sets[part] = set(hashes)
        print('Encoded', part, bundle[part+'_X'].shape, flush=True)
        if bundle[part+'_X'].shape != (15 * per_depth, 58):
            raise ValueError('Unexpected encoder dimensions')
    np.savez_compressed(CACHE / 'h12_features.npz', **bundle)
    x, y = bundle['train_X'], bundle['train_y']
    test, target = bundle['val_X'], bundle['val_y']
    estimator = LinearRegression().fit(x, y)
    saved = pickle.loads((CACHE/'docs/tutorials/model/ising_init_from_qasm_no_readout/ols_full.pk').read_bytes())
    predictions = {'raw': test[:, -4:], 'ols_refit': estimator.predict(test), 'ols_saved': saved.predict(test)}
    novel = ~np.isin(bundle['val_hash'], bundle['train_hash'])
    report = dict(upstream_revision=manifest['revision'], source_manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  tutorial='docs/tutorials/h12_ols.ipynb', dataset='FakeLima simulation, Ising, no readout noise',
                  train_rows=len(x), test_rows=len(test), feature_count=x.shape[1], target_count=y.shape[1],
                  row_order='Depths 0 through 14 in numerical order, within-file order preserved, exactly the upstream train and val populations. Notebook os.listdir ordering was unspecified.',
                  n_unique_train=len(group_sets['train']), n_unique_test=len(group_sets['val']),
                  shared_compiled_circuit_hashes=len(group_sets['train'] & group_sets['val']),
                  versions={name: version(name) for name in ['numpy','scipy','scikit-learn','qiskit-terra','torch']},
                  original_notebook_rmse=refs, rows_without_training_identity=int(novel.sum()),
                  subset_without_training_identity={k:metrics(p[novel],target[novel]) for k,p in predictions.items()},
                  feature_digests={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in bundle.items()},
                  metrics={k:metrics(p, target) for k,p in predictions.items()})
    OUTPUT.mkdir(exist_ok=True)
    output = OUTPUT/'reference.csv.gz'
    with output.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as gz, io.TextIOWrapper(gz,encoding='utf-8',newline='') as f:
        table = csv.writer(f)
        cols = {'ideal':target, **predictions}
        table.writerow(['row_id','compiled_hash','seen_in_train'] + [f'{name}_{j}' for name in cols for j in range(4)])
        for i, ident in enumerate(bundle['val_id']):
            table.writerow([ident, bundle['val_hash'][i], int(not novel[i]), *[float(v) for values in cols.values() for v in values[i]]])
    report['predictions_sha256'] = hashlib.sha256((OUTPUT/'reference.csv.gz').read_bytes()).hexdigest()
    (OUTPUT/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['metrics']), flush=True)


if __name__=='__main__':
    main()
