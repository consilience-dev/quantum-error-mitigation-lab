"""Score selected TabTune regression adapters on the published or clean 500-row hardware split."""
import argparse
import gc
import hashlib
import importlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.model-cache'
os.environ.setdefault('SKB_DATA_DIRECTORY', str(CACHE / 'skrub'))
os.environ.setdefault('HF_HOME', str(CACHE / 'hf'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import numpy as np
import torch

# One estimator per scalar regression target for this resource-bounded comparison.
# The explicit filenames avoid silent fallback to another checkpoint.
MODELS = {
    'KumoTabular': ('kumo', 'KumoTabularRegressorWrapper', 'small/regressor.pt',
                    'model_path', dict(size='small', num_estimators=1, query_batch_size=64, max_context_size=500)),
    'ILTM': ('iltm', 'ILTMRegressorWrapper', 'r128bn.pth', 'checkpoint_dir',
             dict(checkpoint='r128bn', n_ensemble=1, random_state=0)),

}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


PINNED_WEIGHTS = {
    'small/regressor.pt': 'fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423',
    'r128bn.pth': '9e24c80ee0de16b658344eb8f61ff308205619c219b6de6e5fbd79c6914731a5',
}


def checkpoint(row, filename):
    repo, revision = row['repository'], row['repo_revision']
    base = 'https://huggingface.co'
    sha = PINNED_WEIGHTS[filename]
    local = CACHE / 'registry' / revision / filename
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        tmp = local.with_suffix('.part')
        with urlopen(f'{base}/{repo}/resolve/{revision}/{filename}', timeout=90) as src, tmp.open('wb') as out:
            for chunk in iter(lambda: src.read(4 * 1024**2), b''):
                out.write(chunk)
        if digest(tmp) != sha:
            raise ValueError('Downloaded weight checksum mismatch')
        tmp.replace(local)
    elif digest(local) != sha:
        raise ValueError('Local weight checksum mismatch')
    return local, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=MODELS)
    parser.add_argument('--device', default='cpu', choices=['mps','cpu','cuda'])
    parser.add_argument('--clean-seed', type=int, choices=range(5))
    parser.add_argument('--inputs', default='full', choices=['full', 'descriptors_only', 'measurements_only'])
    parser.add_argument('--audit-saved-iltm', action='store_true', help='Check the saved published predictions with a target-zero refit and permutation check')
    args = parser.parse_args()
    if args.audit_saved_iltm and (args.model != 'ILTM' or args.clean_seed is not None or args.inputs != 'full'):
        parser.error('The saved iLTM audit uses the full-feature published split')
    name = args.model
    rows = json.loads((ROOT/'configs/model_inventory.json').read_text())['candidates']
    row = next(r for r in rows if r['name'] == name)
    module, cls, filename, path_arg, settings = MODELS[name]
    adapter = getattr(importlib.import_module(f'tabtune.models.regression.{module}.regressor'), cls)
    path, sha = checkpoint(row, filename)
    params = {**settings, path_arg: str(path.parent if path_arg == 'checkpoint_dir' else path),
              'device': args.device}
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    bundle = np.load(ROOT/'data/upstream/rf_features.npz', allow_pickle=False)
    X, y = bundle['X'], bundle['y']
    tr, te = bundle['train_ids'], bundle['test_ids']
    if X.shape != (3000,169) or len(tr) != 500 or len(te) != 2500 or set(tr) & set(te):
        raise ValueError('Unexpected inputs for the published evaluation')
    split_name, manifest_sha = 'published 500/2500', None
    tag = name
    if args.clean_seed is not None:
        raw = (ROOT / 'data/clean_split.json').read_bytes()
        manifest_sha = hashlib.sha256(raw).hexdigest()
        plan = json.loads(raw)
        run = next(r for r in plan['runs'] if r['budget'] == 500 and r['seed'] == args.clean_seed)
        tr, te = np.asarray(run['calibration_ids'], dtype=int), np.asarray(plan['test_ids'], dtype=int)
        if (np.unique(tr).size != len(tr) or len(tr) != 500 or np.unique(te).size != len(te)
                or len(te) != 900 or np.intersect1d(tr, te).size):
            raise ValueError('Unexpected clean calibration or test membership')
        split_name = 'clean 500/900'
        tag = f'{name}_clean_s{args.clean_seed}'
    if args.inputs != 'full':
        X = X[:, :-4] if args.inputs == 'descriptors_only' else X[:, -4:]
        tag += '_' + args.inputs
    if args.audit_saved_iltm:
        saved = np.loadtxt(ROOT/'reports/registry/ILTM_predictions.csv.gz', delimiter=',', skiprows=1)
        np.testing.assert_array_equal(saved[:, 0], te)
        scores = saved[:, 1:]
        if scores.shape != (2500, 4) or not np.isfinite(scores).all():
            raise ValueError('Invalid saved iLTM scores')
        model = adapter(**params).fit(X[tr], y[tr, 0])
        engine = model.estimator_
        if Path(engine.model_path).resolve() != path.resolve() or engine.finetuning:
            raise ValueError('Unexpected iLTM backbone or update setting')
        reproduced = model.predict(X[te])
        order = np.random.default_rng(9911).permutation(len(te))
        permuted = model.predict(X[te[order]])[np.argsort(order)]
        np.testing.assert_allclose(reproduced, scores[:, 0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(permuted, reproduced, rtol=0, atol=1e-6)
        report = {
            'applies_to': 'All ILTM published and clean prediction reports for r128bn, including both seed-0 ablations.',
            'interpretation': 'Transductive sensitivity arm. With normalize_predictions_ enabled, the upstream engine standardizes each observable across the complete unlabeled query batch, clips in calibration-standardized units, and then restores calibration mean and standard deviation. Each fit uses the whole fixed test split in one predict call. Predictions can differ if query batch composition changes.',
            'checkpoint': filename, 'weight_sha256': sha, 'state_dict_entries': len(model.model_.state_dict()),
            'calibration_rows': len(tr), 'internal_batch_size': engine.batch_size,
            'projection_dimension': engine.n_dims, 'pca_sampling': engine.pca_sampling,
            'weight_finetuning': engine.finetuning, 'normalize_query_predictions': engine.normalize_predictions_,
            'clip_to_calibration_range': engine.clip_predictions_, 'test_rows': len(te),
            'per_target_range': np.stack([scores.min(axis=0), scores.max(axis=0)], axis=1).tolist(),
            'per_target_correlation': [float(np.corrcoef(scores[:, q], y[te,q])[0,1]) for q in range(4)],
            'n_observables_outside_unit_bounds': int((abs(scores)>1).sum()),
            'max_abs_error_refit_vs_saved_target_zero': float(abs(reproduced-scores[:,0]).max()),
            'max_abs_error_restored_query_order': float(abs(reproduced-permuted).max()),
            'saved_mean_circuit_l2': float(np.linalg.norm(scores-y[te],axis=1).mean()),
            'constant_training_target_mean_l2': float(np.linalg.norm(y[tr].mean(axis=0)-y[te],axis=1).mean()),
        }
        # Known-answer regression diagnostic independent of all hardware labels and scores.
        rng = np.random.default_rng(17341)
        fixture_X = rng.uniform(-1, 1, size=(750, 4)).astype(np.float32)
        fixture_y = .7*fixture_X[:, 0] - .3*fixture_X[:, 1]
        torch.manual_seed(0)
        np.random.seed(0)
        fixture_model = adapter(**params).fit(fixture_X[:500], fixture_y[:500])
        fixture_pred = fixture_model.predict(fixture_X[500:])
        fixture_rmse = float(np.sqrt(np.mean((fixture_pred - fixture_y[500:])**2)))
        constant_rmse = float(np.sqrt(np.mean((fixture_y[:500].mean()-fixture_y[500:])**2)))
        report['known_answer_check'] = {
            'design': 'seed=17341, 750 four-feature iid Uniform(-1,1) rows; y=0.7*x0-0.3*x1; first 500 calibration, last 250 test',
            'pearson_r':float(np.corrcoef(fixture_pred, fixture_y[500:])[0,1]),
            'model_rmse':fixture_rmse, 'constant_training_mean_rmse':constant_rmse,
            'beats_constant_on_noiseless_linear_regression': fixture_rmse < constant_rmse,
            'scope':'The frozen r128bn single-member wrapper also fails on this independent simple task. Hardware scores are diagnostic and cannot support conclusions about the general iLTM method.',
        }
        (ROOT/'reports/iltm_prediction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        return
    start = time.monotonic()
    pred = np.zeros((len(te),4), dtype=float)
    for j in range(4):
        model = adapter(**params)
        model.fit(X[tr], y[tr,j])
        pred[:,j] = np.asarray(model.predict(X[te])).reshape(len(te))
        if not np.isfinite(pred[:,j]).all():
            raise ValueError('Nonfinite prediction')
        del model
        gc.collect()
        print(f'{name} finished target {j+1}/4', flush=True)
    err = pred - y[te]
    metrics = {'model':name,'checkpoint':filename,'revision':row['repo_revision'],'weights_sha256':sha,
               'tabtune':version('tabtune'),'torch':torch.__version__,'device':args.device,'settings':settings,
               'split':split_name,'calibration_seed':args.clean_seed,'inputs':args.inputs,'input_columns':X.shape[1],
               'calibration_rows':len(tr),'test_rows':len(te),'clean_split_manifest_sha256':manifest_sha,
               'weights_updated':False,'feature_matrix_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
               'split_sha256':hashlib.sha256(np.r_[tr,te].astype('<i8').tobytes()).hexdigest(),
               'seed':0,'seconds':time.monotonic()-start,'mean_circuit_l2':float(np.linalg.norm(err,axis=1).mean()),
               'pooled_rmse':float(np.sqrt(np.mean(err**2)))}
    out=ROOT/'reports/registry'
    out.mkdir(exist_ok=True)
    np.savetxt(out/f'{tag}_predictions.csv.gz',np.column_stack((te,pred)),delimiter=',',
               fmt=['%d']+['%.12g']*4,header='row_id,pred_0,pred_1,pred_2,pred_3',comments='')
    (out/f'{tag}.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics),flush=True)


if __name__ == '__main__':
    main()
