"""Frozen single-target regressors with pinned checkpoints and explicit full context."""
import hashlib
import json
import os
from importlib.metadata import version
from importlib.resources import files

os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch

from run_nori_tabicl import CHECKPOINTS as GENERAL
from run_tabpfn_tabdpt import CHECKPOINTS as PFN

CHECKPOINTS = {**GENERAL, 'tabpfn_v2': PFN['tabpfn_v2']}
MEMBERS = {'nori': 4, 'tabicl': 8, 'tabpfn_v2': 1}
PACKAGE = {'nori': 'synthefy-nori', 'tabicl': 'tabicl', 'tabpfn_v2': 'tabpfn'}


def checkpoint(name, weight_dir, device):
    repo, revision, filename, expected = CHECKPOINTS[name]
    path = weight_dir/filename
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != expected:
            raise ValueError(f'Unexpected checkpoint for {name}')
    info = dict(model=name, seed=0, frozen=True, device=device, requested_ensemble_members=MEMBERS[name],
                checkpoint_url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',
                checkpoint_sha256=expected, versions={p:version(p) for p in (PACKAGE[name], 'torch', 'numpy', 'scikit-learn')})
    return path, info


def regressor(name, path, device):
    if name == 'tabicl':
        from tabicl import TabICLRegressor
        return TabICLRegressor(model_path=str(path), allow_auto_download=False, device=device,
                               n_estimators=8, batch_size=1, n_jobs=1, kv_cache=True, random_state=0)
    if name == 'nori':
        from synthefy_nori import NoriRegressor
        members = json.loads((files('synthefy_nori.configs')/'model_v3_inference.json').read_text())
        if len(members) != MEMBERS[name]:
            raise ValueError('Expected four Nori members')
        for member in members:
            if 'HighDimFeatureSelector' in member:
                member['HighDimFeatureSelector']['fit_on_test'] = False
        return NoriRegressor(model='nori-1-5', model_path=str(path), device=device,
                             inference_config=members, categorical_columns=None,
                             memory_policy={'allow_subsample': False}, large_context_policy=None)
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(model_path=path, device=device, random_state=0, n_estimators=1,
                           fit_mode='fit_with_cache', inference_precision=torch.float32)


def check_full_context(name, fitted, size):
    """Check observed context rows after prediction and return the effective ensemble size."""
    if name == 'tabicl':
        caches = [c.train_shape for c in fitted.model_kv_cache_.values()]
        actual = sum(s[0] for s in caches)
        if any(shape[1] != size for shape in caches) or not 0 < actual <= MEMBERS[name]:
            raise ValueError(f'Unexpected TabICL context cache: {caches}')
        return actual
    if name == 'nori':
        report = fitted.memory_report_
        if len(fitted.X_train_) != size or not isinstance(report, dict):
            raise ValueError('Missing Nori training context audit')
        if report['dropped_context_rows'] != 0 or fitted.large_context_report_ is not None:
            raise ValueError('Nori context was reduced')
    if name == 'tabpfn_v2':
        members = fitted.executor_.ensemble_members
        if len(members) != 1 or len(members[0].X_train) != size:
            raise ValueError('TabPFN preprocessing changed the context size')
    return MEMBERS[name]
