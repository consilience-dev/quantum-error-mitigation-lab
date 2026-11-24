"""Use only measured observables to evaluate frozen FakeLima fits at new depths."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import pickle
import time
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from simulation_fetch import CACHE, ROOT, verified

OUT = ROOT / 'reports/simulation/depth_transfer'
FIG_MANIFEST = ROOT / 'data/simulation_fig3_sources.json'
H12_REPORT = ROOT / 'reports/simulation/reference.json'
REFERENCE = OUT / 'archive.csv.gz'
PREDICTIONS = OUT / 'predictions.csv.gz'
SCORES = OUT / 'scores.json'
ARCHIVE_METHODS = ('noisy', 'zne', 'ols_full', 'rfr_list')
NEW_METHODS = ('affine_diagonal', 'affine_four_inputs', 'tabpfn_v2_four_inputs')
KEYS = ('paper_row_id', 'depth')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(path, ids, depths, matrices):
    columns = [f'{name}_{q}' for name in matrices for q in range(4)]
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8', newline='') as stream:
            table = csv.writer(stream)
            table.writerow([*KEYS, *columns])
            table.writerows([int(i), int(d), *[float(v) for vector in vectors for v in vector]]
                            for i, d, *vectors in zip(ids, depths, *matrices.values()))


def read_table(path, names):
    with gzip.open(path, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    ids = np.asarray([int(r['paper_row_id']) for r in rows])
    depths = np.asarray([int(r['depth']) for r in rows])
    np.testing.assert_array_equal(np.sort(ids), np.arange(9000))
    np.testing.assert_array_equal(depths, np.repeat(np.arange(30), 300))
    matrices = {name: np.array([[float(r[f'{name}_{q}']) for q in range(4)] for r in rows])
                for name in names}
    if any(not np.isfinite(m).all() for m in matrices.values()):
        raise ValueError('Nonfinite simulation value')
    return ids, depths, matrices


def prepare():
    manifest = json.loads(FIG_MANIFEST.read_bytes())
    source, = (s for s in manifest['files'] if s['path'] == 'docs/paper_figures/no_readout_over_depths.pk')
    archive = CACHE / source['path']
    if not verified(archive, source):
        raise ValueError('Unexpected Figure 3 archive')
    frame = pickle.loads(archive.read_bytes())['df'].sort_values('step', kind='stable')
    ids, depths = frame.index.to_numpy(), frame['step'].to_numpy()
    np.testing.assert_array_equal(np.sort(ids), np.arange(9000))
    np.testing.assert_array_equal(depths, np.repeat(np.arange(30), 300))
    matrices = {name: np.stack([np.ravel(v) for v in frame[name]]) for name in ('ideal', *ARCHIVE_METHODS)}
    if any(matrix.shape != (9000, 4) for matrix in matrices.values()):
        raise ValueError('Expected 9,000 vectors with four observables per archive series')
    for name in ARCHIVE_METHODS:
        np.testing.assert_allclose(np.linalg.norm(matrices[name] - matrices['ideal'], axis=1),
                                   frame['L2_' + name], rtol=0, atol=1e-7)
    OUT.mkdir(parents=True, exist_ok=True)
    write_table(REFERENCE, ids, depths, matrices)


def scores(predictions, target, depths):
    def score(p, mask):
        err = p[mask] - target[mask]
        return {'mean_circuit_mae': float(np.abs(err).mean()),
                'mean_circuit_l2': float(np.linalg.norm(err, axis=1).mean())}
    groups = {'within_depth_0_14': depths < 15, 'unseen_depth_15_29': depths >= 15}
    groups.update({f'depth_{d}':depths == d for d in range(30)})
    return {group: {name: score(p, mask) for name, p in predictions.items()} for group, mask in groups.items()}


def train_arrays():
    report = json.loads(H12_REPORT.read_bytes())
    with np.load(CACHE / 'h12_features.npz', allow_pickle=False) as z:
        for key in ('train_X', 'train_y', 'train_id'):
            if hashlib.sha256(z[key].tobytes()).hexdigest() != report['feature_digests'][key]:
                raise ValueError(f'Training tensor digest mismatch for {key}')
        expected = [f'train/step_{d}/{i}' for d in range(15) for i in range(300)]
        np.testing.assert_array_equal(z['train_id'], expected)
        return np.ascontiguousarray(z['train_X'][:, -4:]), z['train_y']


def fit(args):
    os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from tabpfn import TabPFNRegressor
    from run_tabpfn_tabdpt import CHECKPOINTS

    repo, rev, filename, expected_sha = CHECKPOINTS['tabpfn_v2']
    weights = args.weight_dir / filename
    if sha(weights) != expected_sha:
        raise ValueError('Wrong TabPFN weights')
    ids, depths, archived = read_table(REFERENCE, ('ideal', *ARCHIVE_METHODS))
    x, y = train_arrays()
    query = archived['noisy'].astype(np.float32)
    start = time.monotonic()
    coefs = []
    ones = np.ones(len(x))
    for q in range(4):
        coefs.append(np.linalg.lstsq(np.column_stack([x[:,q], ones]), y[:,q], rcond=None)[0])
    diagonal = np.asarray(coefs)
    full, _, rank, _ = np.linalg.lstsq(np.column_stack([x, ones]), y, rcond=None)
    if rank != 5:
        raise ValueError('Rank-deficient four-input affine design')
    predictions = {NEW_METHODS[0]: query * diagonal[:,0] + diagonal[:,1],
                   NEW_METHODS[1]: np.column_stack([query, np.ones(len(query))]) @ full}
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    outputs = np.empty((len(query), 4))
    for q in range(4):
        model = TabPFNRegressor(model_path=weights, device=args.device, random_state=0, n_estimators=1,
                                fit_mode='fit_with_cache', inference_precision=torch.float32)
        model.fit(x, y[:,q])
        for i in range(0, len(query), 100):
            outputs[i:i+100,q] = model.predict(query[i:i+100])
        print(f'Finished observable {q+1}/4 at {time.monotonic()-start:.1f}s', flush=True)
    predictions[NEW_METHODS[2]] = outputs
    if any(not np.isfinite(m).all() for m in predictions.values()):
        raise ValueError('Nonfinite predictions')
    write_table(PREDICTIONS, ids, depths, predictions)
    sources = json.loads(FIG_MANIFEST.read_bytes())
    archive_source, = (s for s in sources['files'] if s['path'].endswith('no_readout_over_depths.pk'))
    reference = {name: archived[name] for name in ARCHIVE_METHODS}
    report = {'dataset':'ML-QEM Figure 3, four-qubit FakeLima Ising simulation, no readout error.',
              'experiment':'Fit once on 4,500 h12 training rows at depths 0 through 14. Predict both 4,500-row test subsets with the same fit. Four measured observables at both training and inference. No model selection or clipping.',
              'target':'Archived simulated ideal expectation estimates with 10,000 shots.',
              'row_order':'Numeric depth, within-depth original archive order; paper_row_id is the archive DataFrame index.',
              'comparators':'noisy and ZNE are archived estimates. ols_full and rfr_list are archived circuit-aware predictions. New estimators receive only the four noisy observables.',
              'identity_audit':'At depths 0 through 14, 300 test rows at depth 0 share compiled identities with training. The pinned source does not supply val_extra circuit objects for depths 15 through 29. Compiled-identity overlap is unverified there.',
              'train_rows': len(x), 'within_rows': int(np.sum(depths < 15)), 'unseen_depth_rows': int(np.sum(depths >= 15)),
              'upstream_revision':sources['revision'], 'archived_prediction_source':archive_source,
              'figure_source_manifest_sha256':sha(FIG_MANIFEST), 'h12_reference_sha256':sha(H12_REPORT),
              'train_X4_sha256':hashlib.sha256(x.tobytes()).hexdigest(), 'train_y_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
              'checkpoint_url':f'https://huggingface.co/{repo}/resolve/{rev}/{filename}', 'checkpoint_sha256':expected_sha,
              'tabpfn':{'seed':0, 'device':args.device, 'n_estimators':1, 'fit_mode':'fit_with_cache',
                        'precision':'torch.float32', 'query_chunk_rows':100, 'calibration_rows_per_target':len(x)},
              'diagonal_scale_intercept_by_observable':diagonal.tolist(), 'multivariate_coefficients_intercept_last':full.tolist(),
              'versions':{name:version(name) for name in ('tabpfn','torch','numpy','scikit-learn')},
              'archive_csv_sha256':sha(REFERENCE), 'new_predictions_sha256':sha(PREDICTIONS),
              'seconds':time.monotonic()-start, 'scores':scores({**reference, **predictions},archived['ideal'],depths)}
    SCORES.write_text(json.dumps(report,indent=2) + '\n')
    verify()


def verify_workbook(path, computed):
    from verify_figure4 import DIGEST as workbook_sha256

    if sha(path) != workbook_sha256:
        raise ValueError('Unexpected ML-QEM source workbook')
    with ZipFile(path) as workbook:
        shared = ElementTree.fromstring(workbook.read('xl/sharedStrings.xml'))
        strings = [''.join(cell.itertext()) for cell in shared]
        sheet = ElementTree.fromstring(workbook.read('xl/worksheets/sheet2.xml'))
        cells = {c.get('r'): strings[int(c.findtext('{*}v'))]
                 for c in sheet.findall('.//{*}c') if c.get('t') == 's'}
    if cells['A1'] != 'Top row; left column':
        raise ValueError('Expected Figure 3, no readout noise')
    for name, start in [('noisy', 3), ('zne', 35), ('rfr_list', 67)]:
        if cells[f'A{start + 30}'].split(',')[0] != 'Name: L2_' + name:
            raise ValueError(f'Unexpected Figure 3 series for {name}')
        for depth in range(30):
            step, printed_l2 = cells[f'A{start + depth}'].split()
            expected = computed[f'depth_{depth}'][name]['mean_circuit_l2']
            if int(step) != depth or abs(float(printed_l2) - expected) > 5.01e-7:
                raise ValueError(f'Workbook disagrees with {name} at depth {depth}')
    print('Verified all 90 published raw, ZNE and RF depths against the Figure 3 workbook.')


def verify(workbook=None):
    r = json.loads(SCORES.read_bytes())
    for path, key in [(REFERENCE,'archive_csv_sha256'), (PREDICTIONS,'new_predictions_sha256'),
                      (FIG_MANIFEST,'figure_source_manifest_sha256'), (H12_REPORT,'h12_reference_sha256')]:
        if sha(path) != r[key]:
            raise ValueError(f'Digest mismatch for {path.name}')
    ids, depths, archived = read_table(REFERENCE, ('ideal', *ARCHIVE_METHODS))
    pred_ids, pred_depths, preds = read_table(PREDICTIONS, NEW_METHODS)
    np.testing.assert_array_equal(ids, pred_ids)
    np.testing.assert_array_equal(depths, pred_depths)
    query = archived['noisy'].astype(np.float32)
    coefs = np.asarray(r['diagonal_scale_intercept_by_observable'])
    np.testing.assert_allclose(query * coefs[:,0] + coefs[:,1], preds[NEW_METHODS[0]], rtol=0, atol=1e-12)
    design = np.column_stack([query, np.ones(len(query))])
    affine = np.einsum('ij,jk->ik', design, r['multivariate_coefficients_intercept_last'])
    np.testing.assert_allclose(affine, preds[NEW_METHODS[1]], rtol=0, atol=1e-12)
    target = archived.pop('ideal')
    computed = scores({**archived, **preds}, target, depths)
    for group, group_scores in computed.items():
        for model, entry in group_scores.items():
            for metric, actual in entry.items():
                np.testing.assert_allclose(actual, r['scores'][group][model][metric], rtol=0, atol=1e-12)
    fig3 = json.loads((OUT.parent/'fig3/reference.json').read_bytes())
    for name in ARCHIVE_METHODS:
        for metric, value in computed['within_depth_0_14'][name].items():
            np.testing.assert_allclose(value, fig3['scores'][name]['official'][metric], rtol=0, atol=1e-12)
    if workbook:
        verify_workbook(workbook, computed)
    print('Verified 9,000 paper row IDs and seven paired sets of predictions at depths 0–29.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest='mode', required=True)
    tasks.add_parser('prepare', help='Export the verified archive in the h12 reproduction environment')
    task = tasks.add_parser('fit', help='Fit and predict in the pinned TabPFN v2 environment')
    task.add_argument('weight_dir', type=Path)
    task.add_argument('--device', default='cpu', choices=['cpu','mps','cuda'])
    check = tasks.add_parser('verify', help='Rescore the committed tables using only NumPy')
    check.add_argument('--workbook', type=Path, help='Check the original Figure 3 plotted L2 series at all depths')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'fit':
        fit(args)
    else:
        verify(args.workbook)
