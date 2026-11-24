"""Independently check one complete Qermit depolarising panel, stratified by mirror status."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from qermit_context_models import MODELS, verify as verify_model


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports/qermit_cdr_simulation/full_unmirrored'
REFERENCE = ROOT / 'reports/qermit_reference_cells.csv'
# From the unmirrored and mirrored records of Zenodo 6472281, joined by experiment ID to
# circuits.csv, experiments.csv and the three depolarising simulator result files.
# Hash the sorted list of [experiment, circuit, family, mirrored, width, depth,
# ideal, raw, CDR, ZNE] as compact JSON, encoding each float with float.hex().
SOURCE_ROWS_SHA = {
    False: '2b8782ee6579b2b10982f666624e7f98a5a3a94e0871044522387398216b534d',
    True: '38c95fe59548cb0f344e7f17b798eebd1f3b93743730c619f8b7e8399415ab4d',
}
FAMILY_SIZE = {False: 340, True: 470}
SOURCE_CELLS_SHA = 'd51dd3963b0db559715b5e01f0bfe01edcefb541fb728cf699bf8baf8cffc86f'
FAMILIES = ('Pauli-gadget', 'Random')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source(z, protocol):
    reference = json.loads((ROOT / 'reports/qermit_reference.json').read_bytes())
    actual = {name: entry['sha256'] for name, entry in protocol['source_digests'].items()}
    if actual != reference['release_files_sha256'] or digest(REFERENCE) != SOURCE_CELLS_SHA:
        raise ValueError('Qermit source checksums changed')
    mirrors = set(z['mirrored'].tolist())
    if len(mirrors) != 1:
        raise ValueError('Each report must contain exactly one mirror condition')
    mirrored = mirrors.pop()
    size = FAMILY_SIZE[mirrored]
    ids = z['experiment_ids'].tolist()
    if (len(ids) != 2 * size or len(set(ids)) != len(ids) or len(set(z['circuit_ids'])) != len(ids)
            or Counter(z['families']) != {f: size for f in FAMILIES}):
        raise ValueError('Expected the complete release population for this mirror condition')
    if mirrored:
        np.testing.assert_allclose(z['target_ideal'], 1, rtol=0, atol=1e-12)
    rows = []
    for i, ident in enumerate(ids):
        rows.append([ident, str(z['circuit_ids'][i]), str(z['families'][i]), bool(z['mirrored'][i]),
                     int(z['widths'][i]), int(z['depths'][i]),
                     *[float(z[k][i]).hex() for k in ('target_ideal','archived_raw','archived_cdr','archived_zne')]])
    text = json.dumps(sorted(rows), ensure_ascii=True, separators=(',', ':'))
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_ROWS_SHA[mirrored]:
        raise ValueError('Per-circuit source identities, observables or archived values differ from Zenodo')
    return mirrored


def channel_diagnostic(z, indices):
    """Remove noisy and noiseless shot sampling while holding the generated contexts fixed."""
    threshold = 1e-12
    by_cell = defaultdict(list)
    inside = 0
    for i in indices:
        x = z['context_channel'][i].tolist()
        y = z['context_exact'][i].tolist()
        query = float(z['query_channel'][i])
        inside += min(x) <= query <= max(x)
        cell = int(z['widths'][i]), int(z['depths'][i])
        if max(x) - min(x) <= threshold:
            by_cell[cell].append(None)
            continue
        x_mean, y_mean = mean(x), mean(y)
        slope = (math.fsum((v-x_mean)*(label-y_mean) for v,label in zip(x,y))
                 / math.fsum((v-x_mean)**2 for v in x))
        estimate = y_mean + slope * (query - x_mean)
        # The archived CDR comes from a different realization on the same target.
        ideal = float(z['target_ideal'][i])
        predictions = (estimate, z['cdr_prediction'][i], z['archived_cdr'][i])
        by_cell[cell].append(tuple(abs(float(p) - ideal) for p in predictions))
    identified = [loss for losses in by_cell.values() for loss in losses if loss is not None]
    complete = [losses for losses in by_cell.values() if None not in losses]

    def paired_maes(losses):
        fields = ('channel_mean_cdr', 'sampled_cdr', 'archived_cdr')
        return {field: mean(row[j] for row in losses) if losses else None
                for j,field in enumerate(fields)}

    return dict(query_within_auxiliary_channel_range=inside,
                range_rule='Inclusive minimum and maximum of the 20 auxiliary channel means, before shot sampling.',
                fit='Affine fit of exact ideal labels to noisy channel means, evaluated at the target channel mean.',
                excluded_channel_range_at_most=threshold,
                rank_degenerate_targets=len(indices)-len(identified),
                defined_targets=len(identified), paired_mae_defined_targets=paired_maes(identified),
                complete_volume_cells=len(complete),
                incomplete_cells=[dict(width=w, depth=d) for (w,d),losses in sorted(by_cell.items()) if None in losses],
                paired_mae_complete_volume_cells=paired_maes([row for losses in complete for row in losses]))


def check_contexts(z, protocol):
    if protocol['shots_per_context_or_cdr_target'] != 500000 // 21 or protocol['shots_per_fresh_raw'] != 100000:
        raise ValueError('The simulator shot allocation differs from the published notebook')
    for k in ('context_noisy', 'context_ideal', 'context_channel', 'context_exact', 'surrogate_hashes'):
        if z[k].shape != (len(z['experiment_ids']), 20):
            raise ValueError(f'Expected twenty observations per target in {k}')
    for k in ('context_channel', 'context_exact', 'query_channel', 'raw_channel'):
        if not np.isfinite(z[k]).all() or np.any(np.abs(z[k]) > 1 + 1e-9):
            raise ValueError(f'Invalid pre-sampling Pauli expectation values in {k}')
    for k in ('context_noisy', 'context_ideal', 'query_noisy', 'raw_noisy'):
        values = z[k]
        if not np.isfinite(values).all() or np.any(np.abs(values) > 1 + 1e-12):
            raise ValueError(f'Invalid Pauli expectation values in {k}')
        shots = protocol['shots_per_fresh_raw' if k == 'raw_noisy' else 'shots_per_context_or_cdr_target']
        plus_count = (values + 1) * shots / 2
        np.testing.assert_allclose(plus_count, plus_count.round(), atol=1e-7, rtol=0)

    low = np.max(np.abs(z['context_exact'] - z['context_exact'][:, :1]), axis=1) <= 0.01
    warn = (np.abs(z['context_noisy'] - z['query_noisy'][:, None]) > 0.1).sum(axis=1) > 15
    np.testing.assert_array_equal(low, z['low_ideal_spread'])
    np.testing.assert_array_equal(warn, z['quality_warning'])
    if np.any(z['attempts'][low] != 10) or np.any((z['attempts'] < 1) | (z['attempts'] > 10)):
        raise ValueError('The recorded regeneration limit was not respected')
    by_family = {}
    for family in FAMILIES:
        ix = np.flatnonzero(z['families'] == family)
        counts = [len(set(z['surrogate_hashes'][i])) for i in ix]
        matches_target = np.abs(z['context_exact'][ix] - z['target_ideal'][ix,None]) <= 1e-10
        by_family[family] = dict(targets=len(ix), observations_per_target=20,
                                histogram_distinct_surrogate_hashes_per_target={
                                    str(n):v for n,v in sorted(Counter(counts).items())},
                                low_ideal_spread_at_retry_limit=int(low[ix].sum()),
                                contexts_with_quality_warning=int(warn[ix].sum()),
                                any_surrogate_ideal_equal_target_ideal=int(matches_target.any(axis=1).sum()),
                                all_20_surrogate_ideals_equal_target_ideal=int(matches_target.all(axis=1).sum()),
                                channel_mean_diagnostic=channel_diagnostic(z, ix))
    return by_family


def statistics(pred, ideal, denominator):
    errors = [abs(float(p) - float(y)) for p,y in zip(pred, ideal)]
    if any(d == 0 for d in denominator):
        return dict(mean_absolute_error=mean(errors), median_relative_error=None, worst_relative_error=None)
    ratios = [e / float(d) for e,d in zip(errors, denominator)]
    return dict(mean_absolute_error=mean(errors), median_relative_error=median(ratios), worst_relative_error=max(ratios))


def predictions_and_reports(z, folder):
    results = {'raw': z['raw_noisy'], 'CDR': z['cdr_prediction']}
    reports = {}
    for model in MODELS:
        verify_model(folder/'contexts.npz', model)
        with (folder/f'{model}.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        by_id = {r['experiment_id']: float(r['prediction']) for r in rows}
        if set(by_id) != set(z['experiment_ids']) or len(rows) != len(by_id):
            raise ValueError(f'Misaligned {model} prediction IDs')
        results[model] = np.array([by_id[i] for i in z['experiment_ids']])
        report = json.loads((folder/f'{model}.json').read_text())
        reports[model] = {(r['family'],r['width'],r['depth']): r for r in report['cells']}
    return results, reports


def compare_scores(computed, reported, tolerance):
    for field, value in computed.items():
        if value is None:
            if reported[field] is not None:
                raise ValueError(f'Undefined statistic was reported for {field}')
        elif not math.isclose(value, float(reported[field]), abs_tol=tolerance, rel_tol=tolerance):
            raise ValueError(f'Different {field}: {value} and {reported[field]}')


def compare_cells(z, predictions, model_reports, mirrored):
    groups = defaultdict(list)
    for i,(f,w,d) in enumerate(zip(z['families'],z['widths'],z['depths'])):
        groups[str(f),int(w),int(d)].append(i)
    if len(groups) != 2 * FAMILY_SIZE[mirrored] // 10 or any(len(indices) != 10 for indices in groups.values()):
        raise ValueError('Expected all published family / width / depth cells of ten circuits')
    with REFERENCE.open() as f:
        ref = {(r['family'],int(r['width']),int(r['depth']),r['method']):r for r in csv.DictReader(f)
               if r['backend'] == 'depolarising simulator' and r['mirrored'] == str(mirrored)}
    if len(ref) != 3 * len(groups):
        raise ValueError('Archived reference has different cell coverage')
    cells = []
    for group, indices in sorted(groups.items()):
        ix = np.array(indices)
        ideal = z['target_ideal'][ix]
        raw_error = np.abs(z['raw_noisy'][ix] - ideal)
        archived_raw = np.abs(z['archived_raw'][ix] - ideal)
        for method, source in [('NOISY','archived_raw'), ('CDR','archived_cdr'), ('ZNE','archived_zne')]:
            actual = statistics(z[source][ix], ideal, archived_raw)
            compare_scores(actual, ref[(*group, method)], 5e-11)
        scores = {}
        for name, p in predictions.items():
            scores[name] = statistics(p[ix], ideal, raw_error)
            if name in model_reports:
                compare_scores(scores[name], model_reports[name][group], 1e-12)
        cells.append(dict(family=group[0], width=group[1], depth=group[2], n=len(ix),
                          zero_raw_errors=int((raw_error == 0).sum()), methods=scores))
    return cells


def lower_count(rows, method, baseline, field):
    pairs = [(r['methods'][method][field],r['methods'][baseline][field]) for r in rows]
    defined = [(a,b) for a,b in pairs if a is not None and b is not None]
    return {'lower': sum(a < b for a,b in defined), 'equal': sum(a == b for a,b in defined),
            'higher': sum(a > b for a,b in defined), 'undefined':len(pairs)-len(defined)}


def summarize(cells, z, predictions):
    summary = {}
    for name in (*FAMILIES, 'all'):
        rows = [r for r in cells if name == 'all' or r['family'] == name]
        ix = np.arange(len(z['families'])) if name == 'all' else np.flatnonzero(z['families'] == name)
        methods = {}
        for model, p in predictions.items():
            counts = {}
            for baseline in ['raw', 'CDR']:
                counts[baseline] = {field: lower_count(rows, model, baseline, field)
                                    for field in ['median_relative_error', 'worst_relative_error']}
            errors = [abs(float(estimate) - float(ideal)) for estimate, ideal
                      in zip(p[ix], z['target_ideal'][ix])]
            methods[model] = {'secondary_pooled_mean_absolute_error': mean(errors),
                              'cell_comparison_counts': counts}
        summary[name] = dict(targets=len(ix), volume_cells=len(rows), methods=methods)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=FOLDER)
    parser.add_argument('--check', action='store_true', help='Verify the committed summary')
    args = parser.parse_args()
    folder = args.folder
    p = json.loads((folder/'protocol.json').read_bytes())
    if digest(folder/'contexts.npz') != p['contexts_sha256']:
        raise ValueError('Contexts checksum differs from protocol')
    with np.load(folder/'contexts.npz', allow_pickle=False) as data:
        z = {k:data[k] for k in data.files}
    mirrored = check_source(z, p)
    targets = len(z['experiment_ids'])
    diagnostics = check_contexts(z, p)
    predictions, models = predictions_and_reports(z, folder)
    if mirrored:
        # Mirroring the target circuit fixes its exact all-Z expectation to +1.
        # The archive preserves small floating-point deviations from that value.
        predictions['always_plus_one'] = np.ones(targets)
    # Independent OLS identity for the full-rank one-feature affine regression.
    x, y = z['context_noisy'], z['context_ideal']
    x_mean, y_mean = x.mean(axis=1), y.mean(axis=1)
    x_centered, y_centered = x - x_mean[:, None], y - y_mean[:, None]
    sum_squares = (x_centered ** 2).sum(axis=1)
    if np.any(np.ptp(x, axis=1) == 0):
        raise ValueError('A context lacks variation in its measured predictor')
    slope = (x_centered * y_centered).sum(axis=1) / sum_squares
    recomputed = y_mean + slope * (z['query_noisy'] - x_mean)
    np.testing.assert_allclose(recomputed, predictions['CDR'], rtol=1e-9, atol=1e-9)
    cells = compare_cells(z, predictions, models, mirrored)
    scope = 'mirrored' if mirrored else 'non-mirrored'
    report = dict(scope=f'Fresh depolarising simulation on all {targets} {scope} Qermit targets, one fixed '
                        'acquisition per target. The archived raw, ZNE and CDR estimates come from a separate acquisition.',
                  mirrored=mirrored,
                  estimand='Within each family, width and depth cell, compute each target’s absolute error divided '
                           'by the fresh raw absolute error, then take the median and maximum. Count strict '
                           'improvements over the baseline cell statistic. Count undefined cells separately.',
                  source_rows_sha256=SOURCE_ROWS_SHA[mirrored], source_reference_cells_sha256=SOURCE_CELLS_SHA,
                  contexts_sha256=p['contexts_sha256'], linked_input_sha256={
                       k:digest(folder/k) for k in ['protocol.json',*[f'{m}.{ext}' for m in MODELS for ext in ('json','csv')]]},
                  source_check=f'Checked all {targets} experiment IDs, circuit IDs, families, mirror flags, widths, depths, '
                               'ideal targets and raw/CDR/ZNE archival estimates against the independent release digest. '
                               f'All {3*len(cells)} archive cell–method rows reproduce the source-linked published reference.',
                  context_diagnostics=diagnostics, summary=summarize(cells, z, predictions), cells=cells)
    if mirrored:
        report['mirror_control'] = ('The circuit construction fixes every ideal answer to +1. The always-+1 '
                                    'predictor exposes this property; residuals reflect literal released float targets.')
    output = folder/'summary.json'
    if args.check:
        expected = json.loads(output.read_text())
        if json.dumps(report, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError('Qermit full-panel summary differs from the committed result')
    else:
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Verified {targets} {scope} targets, {len(cells)} complete volume cells, '
          f'{3*len(cells)} archived reference rows, and all three frozen models.')


if __name__ == '__main__':
    main()
