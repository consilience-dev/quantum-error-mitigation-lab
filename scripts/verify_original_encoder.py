"""Compare every cached feature and target with the original ML-QEM PyTorch encoder."""
import ast
import hashlib
import io
import json
import pickle

import numpy as np
import torch

from audit_measurements import NumericUnpickler
from fetch_rf_batches import CACHE, ROOT, SOURCES, check_blob
from fetch_upstream import verify


def main():
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    for entry in manifest['files']:
        verify((ROOT / entry['local_path']).read_bytes(), entry)
    # Load exactly the two upstream functions needed for these data, without
    # requiring the unrelated MLP training dependencies imported at module level.
    path = CACHE / 'mlp.py'
    tree = ast.parse(path.read_bytes(), filename=str(path))
    names = {'count_gates_by_rotation_angle', 'encode_data_v2_ecr'}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in tree.body} == names
    scope = dict(np=np, torch=torch)
    exec(compile(tree, str(path), 'exec'), scope)
    arrays = np.load(CACHE / 'rf_features.npz')
    order = np.array(json.loads((CACHE / 'index_order.json').read_text()))
    results = NumericUnpickler(io.BytesIO((CACHE / 'results.pk').read_bytes())).load()
    ideal, noisy = (np.array(results[k]).reshape(-1, 4) for k in ('ideal', 'noisy'))
    verified = []
    for b, entry in enumerate(json.loads(SOURCES.read_text())):
        payload = (CACHE / entry['name']).read_bytes()
        check_blob(payload, entry)
        batch = pickle.loads(payload)
        region = slice(50*b, 50*(b+1))
        ids = order[region]
        np.testing.assert_array_equal(ids, batch['index_order_batch'])
        x, y = scope['encode_data_v2_ecr'](batch['circuit_batch'], ideal[region].tolist(),
                                           noisy[region].tolist(), 4, two_q_gate='cx')
        for key, values in [('X', x.numpy()), ('y', y.numpy())]:
            expected = arrays[key][ids]
            if expected.dtype != values.dtype or expected.tobytes() != values.tobytes():
                raise ValueError(f'{key} byte mismatch in {entry["name"]}')
        verified.extend(ids.tolist())
    assert sorted(verified) == list(range(3000))
    report = dict(upstream_revision=manifest['commit'], encoder_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  torch=torch.__version__, circuit_rows=len(verified), feature_columns=arrays['X'].shape[1],
                  targets=4, agreement='Identical float32 bytes for every feature and target',
                  features_sha256=hashlib.sha256(arrays['X'].tobytes()).hexdigest(),
                  targets_sha256=hashlib.sha256(arrays['y'].tobytes()).hexdigest())
    (ROOT / 'reports/original_encoder_check.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
