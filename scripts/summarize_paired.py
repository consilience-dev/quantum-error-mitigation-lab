"""Rebuild the scorecard from circuit predictions and compare matched draws."""
import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_measurements import read_measurements

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def read_json(path):
    return json.loads(path.read_bytes())


def prediction_groups(path, group_keys=()):
    opener = gzip.open if path.suffix == '.gz' else open
    groups = defaultdict(dict)
    with opener(path, 'rt') as stream:
        reader = csv.DictReader(stream)
        row_id = next(k for k in ('row_id', 'circuit_id', 'unshuffled_row_id') if k in reader.fieldnames)
        cols = sorted(k for k in reader.fieldnames if re.fullmatch(r'(pred|prediction|predicted)_[0-3]', k))
        if len(cols) != 4:
            raise ValueError(f'Expected four observables in {path}')
        for row in reader:
            group = groups[tuple(row[k] for k in group_keys)]
            idx = int(row[row_id])
            if idx in group:
                raise ValueError(f'Repeated circuit {idx} in {path}')
            group[idx] = [float(row[c]) for c in cols]
    return groups


def align(rows, ids):
    if rows.keys() != set(ids):
        raise ValueError('Unexpected circuit IDs')
    values = np.asarray([rows[i] for i in ids])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite circuit prediction')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the committed scores without writing files')
    args = parser.parse_args()
    _, _, order, data, _ = read_measurements()
    data['zne'] = data['zne_mitigated']
    depth = np.r_[np.repeat(np.arange(10),200), np.repeat(np.arange(10),100)]
    data['depth'] = depth
    data['test_ids'] = np.concatenate([np.flatnonzero(depth == d)[50:] for d in range(10)])
    data['hardware_batch'] = np.argsort(order) // 50
    plan = read_json(ROOT / 'data/clean_split.json')
    ids = {'published': data['test_ids'], 'clean': np.asarray(plan['test_ids'], int)}
    manifest_sha = hashlib.sha256((ROOT/'data/clean_split.json').read_bytes()).hexdigest()
    pool_runs = {(r['seed'], r['budget']): np.asarray(r['calibration_ids'],int) for r in plan['runs']}
    budgets = [18,45,90,180,270,450,500]
    assert len(plan['runs']) == len(pool_runs) == 35 and len(set(ids['clean'])) == len(ids['clean']) == 900
    assert np.array_equal(np.bincount(depth[ids['clean']],minlength=10), [0] + [100]*9)
    for seed in range(5):
        previous = set()
        for budget in budgets:
            train = pool_runs[seed,budget]
            counts = np.bincount(depth[train],minlength=10)
            assert len(set(train)) == len(train) == budget and previous <= set(train)
            assert train.min() >= 0 and train.max() < 3000
            assert not (set(train) & set(ids['clean'])) and counts[0] == 0 and np.ptp(counts[1:]) <= 1
            previous = set(train)
    records = []
    losses = defaultdict(dict)

    def add(model, split, seed, budget, arm, pred, reference=None):
        indices = ids[split]
        delta = pred - data['ideal'][indices]
        l2 = np.linalg.norm(delta, axis=1)
        row = dict(model=model, split=split, seed=seed, budget=budget, inputs=arm,
                   mean_circuit_l2=float(l2.mean()), pooled_rmse=float(np.sqrt(np.mean(delta**2))))
        if reference is not None:
            for key in ('mean_circuit_l2', 'pooled_rmse'):
                np.testing.assert_allclose(row[key], reference[key], rtol=0, atol=1e-7)
        for q in range(4):
            row[f'rmse_observable_{q}'] = np.sqrt(np.mean(delta[:,q]**2))
        depth = data['depth'][indices]
        for d in np.unique(depth):
            row[f'mean_l2_depth_{d}'] = l2[depth == d].mean()
        records.append(row)
        if split == 'clean' and budget == 500:
            key = model if arm == 'full' else model + '/' + arm
            if seed in losses[key]:
                raise ValueError(f'Duplicate result for {key}, seed {seed}')
            losses[key][seed] = l2

    for split, indices in ids.items():
        for name, array in [('Hardware', 'noisy'), ('ZNE', 'zne')]:
            add(name, split, None, 0, 'measurement', data[array][indices])

    for r in read_json(REPORTS / 'affine.json')['runs']:
        split = r['split']
        p = data['noisy'][ids[split]] * r['scales'] + r['offsets']
        add('Affine', split, r['seed'], r['budget'], 'full', p, r)

    for name, key in [('CatBoost', 'boosted'), ('Kernel ridge', 'kernel'), ('Ising interpolation', 'physics_control')]:
        tables = prediction_groups(REPORTS / (key + '_predictions.csv.gz'), ('split','seed'))
        for r in read_json(REPORTS / (key + '.json'))['runs']:
            split, seed = r['split'], r['seed']
            label = (split, str(seed) if seed is not None else '')
            add(name, split, seed, 500, 'full', align(tables.pop(label), ids[split]), r)
        assert not tables

    tables = prediction_groups(REPORTS / 'rf_predictions.csv', ('seed',))
    for r in read_json(REPORTS / 'rf_reproduction.json')['rf_refits']:
        add('Random forest', 'published', r['seed'], 500, 'full',
            align(tables.pop((str(r['seed']),)), ids['published']), r)
    assert not tables
    tables = prediction_groups(REPORTS / 'rf_clean_predictions.csv.gz', ('budget','calibration_seed','inputs'))
    for r in read_json(REPORTS / 'rf_clean.json')['runs']:
        budget, seed, arm = r['budget'], r['calibration_seed'], r['inputs']
        p = align(tables.pop((str(budget),str(seed),arm)), ids['clean'])
        add('Random forest', 'clean', seed, budget, arm, p, r)
    assert not tables

    tables = prediction_groups(REPORTS / 'physics_residual_predictions.csv.gz', ('seed','arm'))
    for r in read_json(REPORTS / 'physics_residual.json')['seeds']:
        seed = r['seed']
        for arm, mean in r['mean_circuit_l2'].items():
            p = align(tables.pop((str(seed),arm)), ids['clean'])
            actual = np.linalg.norm(p - data['ideal'][ids['clean']], axis=1)
            np.testing.assert_allclose(actual.mean(), mean, atol=1e-9, rtol=0)
            if arm == 'parameter_interpolation':
                np.testing.assert_allclose(actual, losses['Ising interpolation'][seed], atol=1e-12)
            else:
                add('Ising interpolation', 'clean', seed, 500, arm, p)
    assert not tables

    combined = {'boosted', 'kernel', 'physics_control', 'physics_residual', 'rf', 'rf_clean'}
    labels = {'nori':'Nori 1.5', 'tabicl':'TabICLv2', 'causilo':'Causilo', 'tababax':'TabAbax',
              'tabdpt_v1_3':'TabDPT 1.3', 'tabpfn_v2':'TabPFN v2', 'tabpfn_v2_5':'TabPFN v2.5',
              'tabpfn_v2_6':'TabPFN v2.6', 'ILTM':'iLTM'}
    for path in sorted(REPORTS.rglob('*_predictions.csv*')):
        stem = path.name.split('_predictions.csv')[0]
        if stem in combined:
            continue
        report = read_json(path.with_name(stem + '.json'))
        base = report['model'].split('_clean')[0].removesuffix('_tuned')
        model = labels.get(base, base)
        if model == 'TabICLv2' and report.get('ensemble_members') == 1:
            model += ' single-member timing repeat'
        if 'initialization_control' in report:
            model = f"TabPFN v2 random initialization {report['initialization_control']['initialization_seed']}"
        for key in ('clean_split_sha256','clean_manifest_sha256','clean_split_manifest_sha256'):
            if report.get(key) and report[key] != manifest_sha:
                raise ValueError(f'Unexpected clean split digest in {path}')
        if report.get('weights_updated') or report.get('frozen') is False:
            model += ' tuned'
        split = 'clean' if '_clean_' in stem else 'published'
        match = re.search(r'_(?:s|seed)(\d+)(?:_|$)', stem)
        seed = int(match[1]) if split == 'clean' else None
        inputs = report.get('inputs', report.get('input_arm', 'full'))
        arm = {'all':'full','descriptors':'descriptors_only','noisy':'measurements_only'}.get(inputs, inputs)
        budget = int(report.get('calibration_rows', report.get('context_rows_per_target', report.get('n_train', 500))))
        p = align(prediction_groups(path)[()], ids[split])
        add(model, split, seed, budget, arm, p, report)

    fields = list(dict.fromkeys(k for row in records for k in row))
    destination = REPORTS / 'scores.csv'
    if args.check:
        with destination.open(newline='') as source:
            saved = list(csv.DictReader(source))
        assert len(saved) == len(records)
        for row, expected in zip(records, saved):
            for key, value in row.items():
                if key in ('model', 'inputs', 'split', 'budget', 'seed'):
                    assert expected[key] == ('' if value is None else str(value)), (key, value, expected[key])
                else:
                    np.testing.assert_allclose(float(expected[key]), value, rtol=0, atol=1e-9)
    else:
        with destination.open('w', newline='') as out:
            writer = csv.DictWriter(out, fields)
            writer.writeheader()
            writer.writerows(records)

    # Shared test circuits and shared acquisition batches across calibration draws.
    # Average the paired seed differences circuit-by-circuit before resampling.
    clean = ids['clean']
    _, batch = np.unique(data['hardware_batch'][clean], return_inverse=True)
    count = np.bincount(batch)
    rng = np.random.default_rng(20261124)
    draws = rng.integers(len(count), size=(10000, len(count)))
    depth = data['depth'][clean]
    stratified = [rng.integers(0, np.sum(depth == d), size=(10000, np.sum(depth == d)), dtype=np.int16)
                  for d in np.unique(depth)]
    paired = []
    # Joint acquisition-batch resamples preserve dependence across test contrasts.
    # This family is fixed here independently of ranking or observed sign.
    joint_family = {(m, c) for m in ('Nori 1.5','TabICLv2','TabPFN v2')
                    for c in ('Random forest','Kernel ridge')}
    joint_family |= {('TabPFN v2 tuned','TabPFN v2'), ('TabICLv2 tuned','TabICLv2')}
    joint = {}
    for candidate, by_seed in losses.items():
        comparators = ['Random forest', 'Kernel ridge', 'Affine', 'Hardware', 'ZNE']
        if '/' in candidate:
            comparators = [candidate.split('/')[0]]
        if candidate.startswith('TabPFN v2 random initialization '):
            comparators = ['TabPFN v2']
        if candidate.endswith(' tuned'):
            comparators += [candidate.removesuffix(' tuned')]
        for cmp in comparators:
            if cmp == candidate:
                continue
            if cmp in ('Hardware', 'ZNE'):
                ref = data['noisy' if cmp == 'Hardware' else 'zne'][clean] - data['ideal'][clean]
                reference = {s: np.linalg.norm(ref, axis=1) for s in by_seed}
            else:
                reference = losses[cmp]
            seeds = sorted(by_seed.keys() & reference.keys())
            per_seed = [by_seed[s] - reference[s] for s in seeds]
            difference = np.mean(per_seed, axis=0)
            totals = np.bincount(batch, weights=difference)
            boot = totals[draws].sum(axis=1) / count[draws].sum(axis=1)
            if (candidate,cmp) in joint_family and len(seeds) == 5:
                joint[(candidate,cmp)] = (difference.mean(), boot)
            within = sum(difference[depth == d][r].sum(axis=1) for d,r in zip(np.unique(depth),stratified)) / len(clean)
            paired.append(dict(candidate=candidate, comparator=cmp, calibration_seeds=seeds,
                               delta=float(difference.mean()), delta_each_seed=[float(x.mean()) for x in per_seed],
                               acquisition_batch_interval=np.quantile(boot,[.025,.975]).tolist(),
                               depth_stratified_interval=np.quantile(within,[.025,.975]).tolist()))
    result = dict(test_circuits=len(clean), acquisition_batches=len(count), bootstrap_draws=len(draws),
                  budget=500, interpretation='Candidate minus comparator mean circuit L2. Nominal exploratory 95% intervals conditional on the calibration draws. Test circuits and acquisition batches are shared across draws.',
                  comparisons=paired)
    if joint.keys() == joint_family:
        theta = np.array([x[0] for x in joint.values()])
        replicates = np.stack([x[1] for x in joint.values()], axis=1)
        se = replicates.std(axis=0, ddof=1)
        if (se <= 0).any():
            raise ValueError('Degenerate bootstrap contrast')
        critical = np.quantile(np.max(np.abs((replicates-theta)/se), axis=1), .95)
        result['simultaneous_95pct'] = {
            'scope': 'Exploratory joint eight-contrast family over five calibration draws. Maximum absolute centered batch-bootstrap deviation, standardized by each bootstrap SD. Conditional on this acquisition campaign.',
            'comparisons': [dict(candidate=a, comparator=b, delta=float(v), interval=[v-critical*s,v+critical*s])
                            for (a,b),v,s in zip(joint,theta,se)]}
    destination = REPORTS / 'clean_paired_summary.json'
    if args.check:
        prior = read_json(destination)
        assert len(prior['comparisons']) == len(paired)
        for row, expected in zip(paired, prior['comparisons']):
            for field, value in row.items():
                if field in ('candidate','comparator','calibration_seeds'):
                    assert value == expected[field]
                else:
                    np.testing.assert_allclose(value, expected[field], atol=1e-9, rtol=0)
        if 'simultaneous_95pct' in prior or 'simultaneous_95pct' in result:
            expected = prior['simultaneous_95pct']['comparisons']
            values = result['simultaneous_95pct']['comparisons']
            assert len(values) == len(expected) == 8
            for a,b in zip(values,expected):
                assert (a['candidate'],a['comparator']) == (b['candidate'],b['comparator'])
                np.testing.assert_allclose(a['interval'], b['interval'], rtol=0, atol=1e-9)
        print(f'Verified {len(records)} runs and {len(paired)} paired comparisons against the upstream measurements.')
    else:
        destination.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
