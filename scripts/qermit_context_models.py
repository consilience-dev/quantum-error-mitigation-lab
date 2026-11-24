"""Fit each frozen regressor separately to the same 20 near-Clifford pairs as simulated CDR."""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

DEFAULT = Path(__file__).resolve().parents[1] / 'reports/qermit_cdr_simulation/full_unmirrored/contexts.npz'
MODELS = ('nori', 'tabicl', 'tabpfn_v2')


def read_contexts(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data['experiment_ids'].astype(str)
        x, y = data['context_noisy'], data['context_ideal']
        query = data['query_noisy']
        ideal, raw, cdr = data['target_ideal'], data['raw_noisy'], data['cdr_prediction']
        cells = list(zip(data['families'].astype(str), data['mirrored'].tolist(),
                         data['widths'].tolist(), data['depths'].tolist()))
    if x.shape != (len(ids), 20) or x.shape != y.shape or len(set(ids)) != len(ids):
        raise ValueError('Each distinct target must have exactly 20 calibration pairs')
    if any(a.shape != (len(ids),) for a in (query, ideal, raw, cdr)) or len(cells) != len(ids):
        raise ValueError('Invalid target population')
    if any(not np.isfinite(a).all() for a in (x, y, query, ideal, raw, cdr)):
        raise ValueError('Nonfinite target or calibration value')
    return ids, x, y, query, ideal, raw, cdr, cells


def cell_scores(cells, predictions, raw, ideal):
    error = np.abs(predictions - ideal)
    raw_error = np.abs(raw - ideal)
    groups = {}
    for i, cell in enumerate(cells):
        groups.setdefault(cell, []).append(i)
    rows = []
    for (family, mirrored, width, depth), idx in sorted(groups.items()):
        if len(idx) != 10:
            raise ValueError('A published Qermit simulator cell requires all ten target circuits')
        denom = raw_error[idx]
        delta = error[idx]
        # Retain every target. Flag a cell with an undefined ratio explicitly.
        ratios = delta / denom if np.all(denom > 0) else None
        rows.append(dict(family=family, mirrored=mirrored, width=width, depth=depth,
                         targets=len(idx), zero_raw_error=int((denom == 0).sum()),
                         median_relative_error=float(np.median(ratios)) if ratios is not None else None,
                         worst_relative_error=float(np.max(ratios)) if ratios is not None else None,
                         mean_absolute_error=float(np.mean(delta))))
    return rows


def fit_one(example, *, name, path, device):
    import torch
    from frozen_regressors import regressor, check_full_context

    ident, train_x, train_y, test_x = example
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    fitted = regressor(name, path, device)
    fitted.fit(train_x.reshape(20, 1), train_y)
    pred = np.asarray(fitted.predict(np.asarray(test_x).reshape(1, 1))).reshape(-1)
    if pred.size != 1 or not np.isfinite(pred[0]):
        raise ValueError(f'Invalid prediction for {ident}')
    # TabPFN's fit detects a constant label vector before building its
    # inference engine. Preserve and verify its ordinary public prediction.
    if name == 'tabpfn_v2' and fitted.is_constant_target_:
        if np.any(train_y != train_y[0]) or pred[0] != train_y[0]:
            raise ValueError(f'Unexpected TabPFN constant prediction for {ident}')
        return float(pred[0]), 0
    return float(pred[0]), check_full_context(name, fitted, 20)


def run(args):
    from frozen_regressors import checkpoint

    ids, x, y, query, ideal, raw, cdr, cells = read_contexts(args.contexts)
    weights, info = checkpoint(args.model, args.weights, args.device)
    examples = zip(ids, x, y, query)
    fit = partial(fit_one, name=args.model, path=weights, device=args.device)
    started = time.monotonic()
    # A worker receives only this target's calibration pairs and noisy query.
    # map preserves the checked source order regardless of worker completion order.
    if args.workers == 1:
        results = list(map(fit, examples))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(fit, examples, chunksize=5))
    predictions, members = zip(*results)
    output = args.contexts.parent / f'{args.model}.csv'
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('experiment_id', 'prediction', 'ensemble_members'))
        writer.writerows(zip(ids, predictions, members))
    scores = dict(**info, scope='Fresh Qermit depolarising simulation', targets=len(ids),
                  training_pairs_per_target=20, input='One noisy expectation; a new fit for each target.',
                  effective_ensemble_sizes=sorted(set(members)), workers=args.workers,
                  seconds=time.monotonic()-started,
                  contexts_sha256=digest(args.contexts), predictions_sha256=digest(output),
                  cells=cell_scores(cells, np.asarray(predictions), raw, ideal),
                  cdr_cells=cell_scores(cells, cdr, raw, ideal))
    output.with_suffix('.json').write_text(json.dumps(scores, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path, model):
    ids, _, train_y, _, ideal, raw, cdr, cells = read_contexts(path)
    pred_path = path.parent / f'{model}.csv'
    report = json.loads(pred_path.with_suffix('.json').read_bytes())
    if (report['model'] != model or report['contexts_sha256'] != digest(path)
            or report['predictions_sha256'] != digest(pred_path)
            or report['targets'] != len(ids) or report['training_pairs_per_target'] != 20):
        raise ValueError('Prediction provenance differs')
    with pred_path.open() as stream:
        rows = list(csv.DictReader(stream))
    if [r['experiment_id'] for r in rows] != list(ids):
        raise ValueError('Predictions and contexts have different target IDs')
    a = np.asarray([float(r['prediction']) for r in rows])
    members = np.array([int(r['ensemble_members']) for r in rows])
    sizes = sorted(set(members.tolist()))
    shortcut = (np.ptp(train_y, axis=1) == 0) & (model == 'tabpfn_v2')
    if (sizes != report['effective_ensemble_sizes'] or np.any(members < 0)
            or np.any(members > report['requested_ensemble_members'])
            or not np.array_equal(members == 0, shortcut)
            or not np.array_equal(a[shortcut], train_y[shortcut, 0])):
        raise ValueError('Invalid ensemble sizes')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite predictions')
    for vals, reference in ((a, report['cells']), (cdr, report['cdr_cells'])):
        expected = cell_scores(cells, vals, raw, ideal)
        if len(expected) != len(reference):
            raise ValueError('Missing published cells')
        for new, old in zip(expected, reference):
            for k, v in new.items():
                if isinstance(v, float):
                    np.testing.assert_allclose(v, old[k], rtol=0, atol=1e-12)
                elif v != old[k]:
                    raise ValueError(f'Cell mismatch for {k}')
    print(f'Verified {model} on {len(ids)} target-specific fits across {len(report["cells"])} cells.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--contexts', type=Path, default=DEFAULT)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    parser.add_argument('--workers', type=int, default=1, help='Independent CPU worker processes')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.workers > 1 and args.device != 'cpu'):
        parser.error('Multiple workers require CPU inference')
    if not args.check:
        if not args.weights:
            parser.error('Fitting requires --weights')
        run(args)
    verify(args.contexts, args.model)
