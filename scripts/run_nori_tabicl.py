"""Evaluate Nori 1.5 and TabICLv2 on the 4-qubit ML-QEM hardware splits."""
import argparse
import csv
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    'nori': ('Synthefy/Nori-1-5', '79afc5a65b2775d0c62977211a63620ce87846c7', 'nori.pt',
             'b1313b45225066f74a78e99d5cd56dbd0015bdbeadf948252cf82d2a962766a2'),
    'tabicl': ('jingang/TabICL', '4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
               'tabicl-regressor-v2-20260212.ckpt', '0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=CHECKPOINTS)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--features', type=Path, default=ROOT / 'data/upstream/rf_features.npz')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--chunk', type=int, default=250)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports')
    parser.add_argument('--clean-seed', type=int, choices=range(5))
    parser.add_argument('--clean-budget', type=int, choices=[18, 45, 90, 180, 270, 450, 500], default=500)
    parser.add_argument('--finetune', action='store_true', help='Three epochs of TabICLv2 adaptation on clean calibration data')
    parser.add_argument('--single-member-pilot', action='store_true',
                        help='Repeat the published-split TabICLv2 timing pilot in separate output files')
    parser.add_argument('--inputs', choices=['full', 'descriptors_only', 'measurements_only'], default='full')
    args = parser.parse_args()
    if args.clean_budget != 500 and (args.clean_seed is None or args.finetune):
        parser.error('Smaller calibration budgets require a frozen clean run')
    if args.single_member_pilot and (args.model != 'tabicl' or args.clean_seed is not None or args.finetune):
        parser.error('The timing pilot uses frozen TabICLv2 on the published split')
    if args.inputs != 'full' and args.clean_seed is None:
        parser.error('Omission controls require the clean split')
    if args.finetune and (args.model != 'tabicl' or args.clean_seed is None):
        parser.error('fine-tuning requires TabICLv2 and a clean calibration seed')
    if args.chunk < 1:
        parser.error('chunk must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    repo, revision, filename, weight_hash = CHECKPOINTS[args.model]
    path = args.weights / filename
    if digest(path) != weight_hash:
        raise ValueError('Unexpected checkpoint digest')
    with np.load(args.features, allow_pickle=False) as data:
        X, y, ideal = data['X'], data['y'], data['ideal']
        train, test, depth = data['train_ids'], data['test_ids'], data['depth']
    if (X.shape != (3000, 169) or y.shape != (3000, 4) or len(train) != 500 or len(test) != 2500
            or not np.array_equal(np.sort(np.concatenate([train, test])), np.arange(3000))):
        raise ValueError('Expected the published 500/2500 split')
    protocol = 'published'
    split_hash = None
    if args.clean_seed is not None:
        split_path = ROOT / 'data/clean_split.json'
        split_hash = digest(split_path)
        split = json.loads(split_path.read_bytes())
        run, = (r for r in split['runs'] if r['seed'] == args.clean_seed and r['budget'] == args.clean_budget)
        train = np.asarray(run['calibration_ids'], dtype=int)
        test = np.asarray(split['test_ids'], dtype=int)
        if (len(train) != args.clean_budget or len(test) != 900 or np.intersect1d(train, test).size
                or np.unique(train).size != args.clean_budget or np.unique(test).size != 900 or 0 in depth[np.r_[train, test]]):
            raise ValueError('Expected a disjoint nonzero-depth split at the requested budget')
        protocol = f'clean_{args.clean_budget}_seed{args.clean_seed}'
    prefix = f'{args.model}_{protocol}' + ('_tuned' if args.finetune else '')
    tabicl_members = 1 if args.single_member_pilot else 8
    if args.single_member_pilot:
        prefix += '_single_member_pilot_rerun'
    if args.inputs != 'full':
        prefix += f'_{args.inputs}'
        X = X[:, :-4] if args.inputs == 'descriptors_only' else X[:, -4:]
    tuning = []
    def build():
        if args.model == 'nori':
            from synthefy_nori import NoriRegressor
            members = None
            if args.clean_seed is not None:
                from importlib.resources import files
                source = files('synthefy_nori.configs') / 'model_v3_inference.json'
                members = json.loads(source.read_text())
                for member in members:
                    if 'HighDimFeatureSelector' in member:
                        member['HighDimFeatureSelector']['fit_on_test'] = False
            return NoriRegressor(inference_config=members, model='nori-1-5', model_path=str(path), device=args.device, categorical_columns=None,
                                 memory_policy={'allow_subsample': False})
        if args.finetune:
            from tabicl import FinetunedTabICLRegressor
            from tabicl._finetune.logging import NullLogger

            class History(NullLogger):
                def __init__(self):
                    self.epochs = []

                def log_epoch(self, metrics, step):
                    self.epochs.append({'optimizer_steps': step, **metrics})

            class TracedRegressor(FinetunedTabICLRegressor):
                def _make_experiment_logger(self):
                    self.history = History()
                    return self.history

            return TracedRegressor(
                model_path=str(path), device=args.device, allow_auto_download=False,
                epochs=3, learning_rate=1e-5, max_data_size=96, min_delta=0.0,
                n_estimators_finetune=1, n_estimators_validation=1, n_estimators_inference=8,
                validation_split_ratio=0.1, early_stopping=True, patience=3, random_state=0,
                verbose=True, extra_regressor_kwargs={'batch_size': 1, 'n_jobs': 1, 'kv_cache': True},
            )
        from tabicl import TabICLRegressor
        return TabICLRegressor(model_path=str(path), allow_auto_download=False, device=args.device,
                               n_estimators=tabicl_members, batch_size=1, n_jobs=1, kv_cache=True, random_state=0)
    started = time.monotonic()
    predictions = np.empty((len(test), 4), dtype=float)
    for q in range(4):
        model = build()
        fit_start = time.monotonic()
        model.fit(X[train], y[train, q])
        fit_elapsed = time.monotonic() - fit_start
        if args.finetune:
            original = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
            changes = [torch.linalg.vector_norm(p.detach().cpu() - original[name]).item()
                       for name, p in model.model_.named_parameters()]
            tuning.append({'target': q, 'selected_parameter_change_l2': float(np.linalg.norm(changes)),
                           'selected_validation_mse': float(-model._best_metric_),
                           'trainable_parameters': sum(p.numel() for p in model.model_.parameters() if p.requires_grad),
                           'epochs': model.history.epochs, 'fit_seconds': fit_elapsed})
            del original
        for start in range(0, len(test), args.chunk):
            rows = test[start:start + args.chunk]
            predictions[start:start + len(rows), q] = model.predict(X[rows])
            print(f'{args.model} observable {q+1}/4: {start+len(rows)}/{len(test)} ({time.monotonic()-started:.0f}s)', flush=True)
        del model
        if args.device == 'mps':
            torch.mps.empty_cache()
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite predictions')
    err = predictions - ideal[test]
    output = args.output
    output.mkdir(exist_ok=True, parents=True)
    pred_file = output / f'{prefix}_predictions.csv'
    with pred_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'depth', 'pred_0', 'pred_1', 'pred_2', 'pred_3'])
        writer.writerows((int(i), int(depth[i]), *p) for i, p in zip(test, predictions))
    report = {'protocol': protocol, 'clean_split_sha256': split_hash, 'model': args.model, 'frozen': not args.finetune,
              'input_arm': args.inputs, 'context_rows_per_target': len(train), 'test_rows': len(test), 'features': X.shape[1],
              'target_fits': 4, 'features_npz_sha256': digest(args.features), 'predictions_sha256': digest(pred_file),
              'checkpoint_url': f'https://huggingface.co/{repo}/resolve/{revision}/{filename}', 'checkpoint_sha256': weight_hash,
              'device': args.device, 'platform': platform.platform(), 'python': platform.python_version(),
              'versions': {k:version(k) for k in ['torch','synthefy-nori','tabicl','numpy','scikit-learn','scipy']},
              'seed': 0, 'prediction_chunk_rows': args.chunk, 'ensemble_members': 4 if args.model == 'nori' else tabicl_members,
              'settings': ('numeric columns, checkpoint preprocessing; high-dimensional projection fit on train only'
                           if args.clean_seed is not None else 'numeric columns, checkpoint preprocessing') if args.model == 'nori' else 'eight default ensemble members, cached context',
              'pooled_rmse': float(np.sqrt(np.mean(err**2))), 'mean_circuit_l2':float(np.mean(np.linalg.norm(err,axis=1))),
              'mean_circuit_l2_by_depth': {int(d):float(np.linalg.norm(err[depth[test]==d],axis=1).mean()) for d in np.unique(depth[test])},
              'runtime_seconds': time.monotonic()-started}
    if args.single_member_pilot:
        report['settings'] = 'one ensemble member, cached context'
        source = ROOT / 'reports/tabicl_single_member_timing_pilot.json'
        original = json.loads(source.read_bytes())
        report['pilot_rerun'] = {
            'original_score_log': str(source.relative_to(ROOT)), 'original_score_log_sha256': digest(source),
            'provenance': 'Predictions newly generated for this repeat. Original pilot aggregates were retained.',
            'repeat_minus_original': {key: report[key] - original[key] for key in ('mean_circuit_l2', 'pooled_rmse')},
            'max_absolute_depth_mean_l2_difference': max(abs(mean - original['mean_circuit_l2_by_depth'][str(d)])
                                                         for d, mean in report['mean_circuit_l2_by_depth'].items()),
        }
    if args.finetune:
        report['fine_tuning'] = {'epochs_max': 3, 'fit_rows': 450, 'validation_rows': 50,
                                  'final_context_rows': 500, 'lr': 1e-5, 'episode_cap': 96,
                                  'selection': 'validation MSE including frozen starting weights', 'targets': tuning}
    (output / f'{prefix}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
