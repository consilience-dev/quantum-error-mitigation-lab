"""Fit and audit frozen readout corrections on Song et al.'s published hardware calibrations."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np


FOLDER = Path(__file__).resolve().parents[1] / 'reports/pec_song'
SINGLE = FOLDER / 'single_qubit_inputs.json'
DOUBLE = FOLDER / 'two_qubit_targets.json'
PUBLICATION = FOLDER / 'publication_check.json'
METHODS = ('affine', 'nori', 'tabicl', 'tabpfn_v2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    single = json.loads(SINGLE.read_text())
    targets = json.loads(DOUBLE.read_text())['targets']
    if [row['phase_over_pi'] for row in targets] != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError('Expected the four published controlled-phase circuits in order')
    ideal = np.asarray(single['ideal_gram'], dtype=float)
    np.testing.assert_array_equal(ideal, [[1, 1, 1, 1], [0, 0, 1, 0],
                                         [0, 0, 0, -1], [1, -1, 0, 0]])
    for qubit, axis, axis_idx in [('Q1', 'Z', 3), ('Q2', 'Z', 3), ('Q1', 'X', 1)]:
        records = single['qubits'][qubit]
        train = np.asarray(records['gram'], dtype=float)[axis_idx].reshape(4, 1)
        if axis == 'Z':
            bins = records['raw_bins']
            if sum(b['blocks'] for b in bins) != 100:
                raise ValueError('The raw histogram must cover all 100 blocks')
            grid = sorted({k for b in bins if b['blocks'] > 0
                           for k in range(b['raw_plus_count_min'], b['raw_plus_count_max'] + 1)})
            if not grid or min(grid) < 0 or max(grid) > 3000:
                raise ValueError('Invalid 3,000-shot expectation grid')
            queries = np.array([2 * k / 3000 - 1 for k in grid])
            yield qubit, axis, train, ideal[axis_idx], grid, queries, records
        else:
            queries = np.array([r['raw'] for r in targets])
            yield qubit, axis, train, ideal[axis_idx], [r['phase_over_pi'] for r in targets], queries, targets


def prediction_file(model):
    return FOLDER / f'{model}_predictions.csv'


def predict_points(model, path, x, y, query):
    if model == 'affine':
        return np.polyval(np.polyfit(x[:, 0], y, 1), query), 1
    import torch
    from frozen_regressors import regressor, check_full_context

    def reset_seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    reset_seed()
    fitted = regressor(model, path, 'cpu')
    fitted.fit(x, y)
    values = []
    for v in query:
        reset_seed()
        values.append(float(np.asarray(fitted.predict(np.array([[v]]))).item()))
    return np.array(values), check_full_context(model, fitted, len(y))


def calculate(model, weights):
    path = None
    if model == 'affine':
        info = {'model': model, 'estimator': 'OLS with slope and intercept'}
    else:
        import torch
        from frozen_regressors import checkpoint

        torch.set_num_threads(2)
        path, info = checkpoint(model, weights, 'cpu')

    data = []
    sensitivity = []
    for qubit, axis, x, y, keys, queries, _ in cases():
        # Each possible block is a separate query. A fresh fit with the reversed
        # sequence independently verifies that the histogram has a fixed map.
        pred, members = predict_points(model, path, x, y, queries)
        reverse, repeat_members = predict_points(model, path, x, y, queries[::-1])
        np.testing.assert_array_equal(pred, reverse[::-1])
        if members != repeat_members:
            raise ValueError('A repeat fit changed the ensemble size')
        if not np.isfinite(pred).all():
            raise ValueError(f'Nonfinite {model} prediction for {qubit} {axis}')
        data.extend(zip([qubit]*len(keys), [axis]*len(keys), keys, queries, pred, [members]*len(keys)))
        if axis == 'X':
            alternative = np.asarray(json.loads(PUBLICATION.read_text())['q1_gram_from_main_figure'])[1, :, None]
            delta = json.loads(DOUBLE.read_text())['digitization_stress_step']
            for source, calibration in [('supplement', x), ('main', alternative)]:
                for shift in [-delta, 0., delta]:
                    altered, _ = predict_points(model, path, calibration, y, queries + shift)
                    sensitivity.extend(zip([source]*len(keys), [shift]*len(keys), keys, altered))
    with prediction_file(model).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('qubit', 'axis', 'plus_count_or_phase_over_pi', 'raw', 'prediction', 'ensemble_members'))
        writer.writerows(data)
    with (FOLDER / f'{model}_digitization.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('calibration_figure', 'input_shift', 'phase_over_pi', 'prediction'))
        writer.writerows(sensitivity)
    return info


def histogram_bounds(predictions, records):
    """Bound any allowed allocation of the 100 blocks, including the printed raw mean."""
    bins, slopes = [], [0.0]
    for b in records['raw_bins']:
        if b['blocks'] == 0:
            continue
        keys = range(b['raw_plus_count_min'], b['raw_plus_count_max'] + 1)
        x = 2 * np.asarray(keys) / 3000 - 1
        y = np.array([predictions[k] for k in keys])
        bins.append((b['blocks'], x, y))
        # Extremal residuals change their active grid point at these slopes.
        i, j = np.triu_indices(len(x), k=1)
        slopes.extend((y[j] - y[i]) / (x[j] - x[i]))
    a = np.unique(slopes)
    low, high = np.zeros(len(a)), np.zeros(len(a))
    for multiplicity, x, y in bins:
        residual = y - a[:, None] * x
        low += multiplicity * residual.min(axis=1)
        high += multiplicity * residual.max(axis=1)
    raw_mean = records['reported']['raw']['mean']
    lower = low / 100 + a * raw_mean - abs(a) * 0.0005
    upper = high / 100 + a * raw_mean + abs(a) * 0.0005
    zero = a.tolist().index(0.0)
    if lower.max() > upper.min() + 1e-12:
        raise ValueError('Published mean and histogram are inconsistent')
    return {'histogram_only': [round(float(low[zero]/100), 12), round(float(high[zero]/100), 12)],
            'including_rounded_raw_mean': [round(float(lower.max()), 12), round(float(upper.min()), 12)]}


def scores(model, info):
    with prediction_file(model).open() as f:
        rows = list(csv.DictReader(f))
    result = {}
    for qubit, axis, train_x, train_y, keys, query, records in cases():
        group = [r for r in rows if (r['qubit'], r['axis']) == (qubit, axis)]
        if [float(r['plus_count_or_phase_over_pi']) for r in group] != keys:
            raise ValueError('Prediction rows must match every published circuit or admissible shot count')
        if not all(int(r['ensemble_members']) >= 1 for r in group):
            raise ValueError('Empty fitted ensemble')
        np.testing.assert_array_equal([float(r['raw']) for r in group], query)
        pred = np.array([float(r['prediction']) for r in group])
        if not np.isfinite(pred).all():
            raise ValueError('Invalid predictions')
        if model == 'affine':
            np.testing.assert_allclose(pred, np.polyval(np.polyfit(train_x[:, 0], train_y, 1), query), atol=1e-12)
        if axis == 'Z':
            result[qubit] = {'mean_prediction_bounds': histogram_bounds(dict(zip(keys, pred)), records),
                             'blocks': 100, 'shots_per_block': 3000}
        else:
            errors = np.abs(pred - [r['ideal'] for r in records])
            result['two_qubit'] = {'settings': [
                dict(phase_over_pi=k, prediction=round(float(v), 12), absolute_error=round(float(e), 12))
                for k, v, e in zip(keys, pred, errors)], 'four_setting_mae': round(float(errors.mean()), 12)}
            stress = FOLDER / f'{model}_digitization.csv'
            with stress.open() as f:
                variants = list(csv.DictReader(f))
            step = json.loads(DOUBLE.read_text())['digitization_stress_step']
            maes = []
            for label in ('supplement', 'main'):
                for shift in (-step, 0., step):
                    group = [r for r in variants if r['calibration_figure'] == label and float(r['input_shift']) == shift]
                    if [float(r['phase_over_pi']) for r in group] != keys:
                        raise ValueError('Invalid digitization sensitivity settings')
                    estimates = np.array([float(r['prediction']) for r in group])
                    if not np.isfinite(estimates).all():
                        raise ValueError('Nonfinite sensitivity prediction')
                    if label == 'supplement' and shift == 0:
                        np.testing.assert_array_equal(estimates, pred)
                    maes.append(float(np.abs(estimates - [r['ideal'] for r in records]).mean()))
            if len(variants) != 6 * len(keys):
                raise ValueError('Unexpected sensitivity prediction rows')
            result['two_qubit']['six_digitization_variants_mae_range'] = [round(min(maes), 12), round(max(maes), 12)]
    expected_size = sum(len(c[4]) for c in cases())
    if len(rows) != expected_size:
        raise ValueError('Unexpected extra prediction rows')
    return {'configuration': info, 'calibration_states_per_fit': 4,
            'one_qubit_input_sha256': digest(SINGLE), 'two_qubit_input_sha256': digest(DOUBLE),
            'publication_check_sha256': digest(PUBLICATION),
            'digitization_predictions_sha256': digest(FOLDER / f'{model}_digitization.csv'),
            'predictions_sha256': digest(prediction_file(model)), 'scores': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=METHODS)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = FOLDER / f'{args.model}.json'
    if args.check:
        saved = json.loads(report.read_text())
        if saved['configuration']['model'] != args.model or saved != scores(args.model, saved['configuration']):
            raise ValueError('Saved score or source fingerprint changed')
    else:
        if args.model != 'affine' and args.weights is None:
            parser.error('The frozen estimators require --weights')
        result = scores(args.model, calculate(args.model, args.weights))
        report.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Checked {args.model} on two one-qubit hardware histograms and four two-qubit circuit means.')


if __name__ == '__main__':
    main()
