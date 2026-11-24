"""Verify historical-graph data without installing the 2022 quantum software."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit_qermit import ROOT, index, read_results, table, verify_inputs

SHOTS = 500_000 // 21
ARCHIVE_SOURCE = 'aa25c21a9a17aa5b46d03c4152d2d61387356d21'
DEFAULT = ROOT / 'reports/qermit_cdr_simulation/acquisition_era'
MODERN = ROOT / 'reports/qermit_cdr_simulation/full_unmirrored'


def load(folder):
    metadata = json.loads((folder / 'protocol.json').read_text())
    file = folder / 'contexts.npz'
    if hashlib.sha256(file.read_bytes()).hexdigest() != metadata['contexts_sha256']:
        raise ValueError('Context checksum mismatch')
    with np.load(file, allow_pickle=False) as a:
        return metadata, dict(a)


def verify(cache, folder, pilot):
    protocol, data = load(folder)
    if protocol['source_digests'] != verify_inputs(cache, False):
        raise ValueError('Original archive checksums differ')
    if (protocol['acquisition_code'] != ARCHIVE_SOURCE or protocol['n_context'] != 20
            or protocol['shots_per_context_or_cdr_target'] != SHOTS or protocol['shots_per_fresh_raw'] != 100_000
            or protocol['ideal_simulated_shots_per_context'] != SHOTS
            or protocol['noisy_shots_per_cdr_estimate'] != 21 * SHOTS):
        raise ValueError('Unexpected source or resource allocation')
    source = index(table(cache, 'experiments.csv'), 'experiment ID')
    circuits = index(table(cache, 'circuits.csv'), 'circuit ID')
    archive = read_results(cache, source)
    ids = []
    for key, e in sorted(source.items()):
        c = circuits[e['circuit ID']]
        if c['mirrored'] == 'False' and int(c['width']) in protocol['widths'] and int(c['depth']) in protocol['depths']:
            ids.append(key)
    np.testing.assert_array_equal(data['experiment_ids'], ids)
    if len(ids) != protocol['n_targets'] or data['context_noisy'].shape != (len(ids), 20):
        raise ValueError('Missing historical calibration data')
    for i, key in enumerate(ids):
        e = source[key]
        c = circuits[e['circuit ID']]
        if (data['circuit_ids'][i] != e['circuit ID'] or data['mirrored'][i]
                or data['families'][i] != c['type'] or data['depths'][i] != int(c['depth'])
                or data['widths'][i] != int(c['width'])):
            raise ValueError('Archived circuit identity or cell mismatch')
        record = archive['depolarising simulator', key]
        np.testing.assert_array_equal(
            [data[k][i] for k in ('target_ideal', 'archived_raw', 'archived_cdr', 'archived_zne')],
            [float(e['ideal expectation']), record['NOISY'], record['CDR'], record['ZNE']])
    cells = Counter(zip(data['families'], data['widths'], data['depths']))
    if set(cells.values()) != {10}:
        raise ValueError('Incomplete native simulator cells')
    for k in ('context_noisy', 'context_ideal', 'surrogate_hashes', 'context_channel', 'context_exact'):
        if data[k].shape != (len(ids), 20):
            raise ValueError(f'Incorrect shape for {k}')
    for k in ('context_noisy', 'context_ideal', 'query_noisy', 'raw_noisy',
              'context_exact', 'context_channel', 'query_channel', 'raw_channel'):
        if not np.isfinite(data[k]).all() or np.max(np.abs(data[k])) > 1 + 1e-9:
            raise ValueError(f'Invalid Pauli expectation in {k}')
    for k in ('context_noisy', 'context_ideal', 'query_noisy', 'raw_noisy'):
        n = 100_000 if k == 'raw_noisy' else SHOTS
        n_even = (data[k] + 1) * n / 2
        np.testing.assert_allclose(n_even, np.rint(n_even), rtol=0, atol=1e-8)
    x = data['context_noisy']
    y = data['context_ideal']
    x0, y0 = x.mean(axis=1), y.mean(axis=1)
    centered = x - x0[:, None]
    slopes = (centered * (y - y0[:, None])).sum(axis=1) / (centered ** 2).sum(axis=1)
    predicted = y0 + slopes * (data['query_noisy'] - x0)
    np.testing.assert_allclose(predicted, data['cdr_prediction'], rtol=0, atol=1e-9)
    warnings = (np.abs(x - data['query_noisy'][:, None]) > 0.1).sum(axis=1) > 15
    np.testing.assert_array_equal(warnings, data['quality_warning'])
    low = (np.abs(data['context_exact'] - data['context_exact'][:, :1]) <= 0.01).all(axis=1)
    np.testing.assert_array_equal(low, data['low_ideal_spread'])
    if (np.any((data['attempts'] < 1) | (data['attempts'] > 10)) or np.any(data['attempts'][low] != 10)
            or protocol['n_low_ideal_spread'] != int(low.sum())
            or protocol['n_quality_warnings'] != int(warnings.sum())):
        raise ValueError('Historical warning or retry count differs')

    sample_error = {}
    for k, mean in [('context_noisy', 'context_channel'), ('context_ideal', 'context_exact'),
                    ('query_noisy', 'query_channel'), ('raw_noisy', 'raw_channel')]:
        sample_error[k] = float(np.max(np.abs(data[k] - data[mean])))
    modern_protocol, modern = load(MODERN)
    modern_ids = {key: i for i, key in enumerate(modern['experiment_ids'])}
    families = {}
    for name in sorted(set(data['families'])):
        idx = np.flatnonzero(data['families'] == name)
        current = [modern_ids[data['experiment_ids'][i]] for i in idx]
        target = data['target_ideal'][idx]
        mae = {key: float(np.mean(abs(data[key][idx] - target))) for key in
               ('archived_raw', 'archived_cdr', 'archived_zne', 'raw_noisy', 'cdr_prediction')}
        mae['modern_cdr'] = float(np.mean(abs(modern['cdr_prediction'][current] - target)))
        families[str(name)] = {'target_count': len(idx), 'mae': mae}
    report = dict(contexts_sha256=protocol['contexts_sha256'], modern_sha256=modern_protocol['contexts_sha256'],
                  n_targets=len(ids), n_cells=len(cells), families=families,
                  largest_absolute_shot_error=sample_error)
    if pilot is not None:
        pilot_protocol, earlier = load(pilot)
        if pilot_protocol['resource_mode'] != 'bitstring memory' or protocol['resource_mode'] != 'count histograms':
            raise ValueError('Pilot comparison requires the two distinct native Aer return formats')
        positions = {v: i for i, v in enumerate(ids)}
        matched = [positions[v] for v in earlier['experiment_ids']]
        if earlier.keys() != data.keys():
            raise ValueError('Native result schemas differ')
        for k in data:
            # Reject any changed measured counts. Numeric channel means can vary
            # in their final floating-point bits across operating systems.
            if k in ('context_exact', 'context_channel', 'query_channel', 'raw_channel', 'cdr_prediction'):
                np.testing.assert_allclose(data[k][matched], earlier[k], rtol=0, atol=1e-10)
            else:
                np.testing.assert_array_equal(data[k][matched], earlier[k])
        report['bitstring_pilot'] = dict(source_sha256=pilot_protocol['contexts_sha256'],
                                        targets=len(matched), exact_measured_count_agreement=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = verify(args.cache, args.folder, args.pilot)
    path = args.folder / 'native_graph_audit.json'
    if args.write:
        path.write_text(json.dumps(result, indent=2) + '\n')
    elif path.exists():
        saved = json.loads(path.read_text())
        if result != saved:
            # Allow typical final-bit reductions across supported NumPy builds.
            for family, stats in result['families'].items():
                reference = saved['families'][family]['mae']
                if reference.keys() != stats['mae'].keys():
                    raise ValueError('Missing method in saved MAE')
                for k, v in stats['mae'].items():
                    np.testing.assert_allclose(v, reference[k], rtol=0, atol=1e-12)
                saved['families'][family]['mae'] = stats['mae']
            if result != saved:
                raise ValueError('Historical graph audit differs')
    print(f"Verified {result['n_targets']} targets and {result['n_cells']} complete volume cells.")
