"""Estimate six paired MAE contrasts on the acquisition-period unmirrored Qermit panel."""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_qermit_panel import (
    FAMILIES, FOLDER, MODELS, acquisition_description, check_source, digest, predictions_and_reports,
)


PANEL = FOLDER.parent / 'acquisition_era/full_unmirrored'
REPLICATES = 50_000
SEED = 20261124
FAMILYWISE_ALPHA = 0.05


def bootstrap_family(indices, z, predictions, rng, quantiles):
    groups = {}
    for i in indices:
        key = int(z['widths'][i]), int(z['depths'][i])
        groups.setdefault(key, []).append(i)
    if len(groups) != 34 or any(len(members) != 10 for members in groups.values()):
        raise ValueError('Expected 34 fixed volume groups, each containing ten targets')
    targets = np.array([sorted(groups[key], key=lambda i: z['experiment_ids'][i]) for key in sorted(groups)])
    ideal = z['target_ideal'][targets]
    cdr_error = np.abs(predictions['CDR'][targets] - ideal)
    differences = [np.abs(predictions[model][targets] - ideal) - cdr_error for model in MODELS]
    bootstrap = np.empty((len(MODELS), REPLICATES))
    group_rows = np.arange(len(groups))[None, :, None]
    # Each sample draws ten indices in each group, with replacement. The same
    # draw is reused for all contrasts in the family. Batching bounds memory.
    for start in range(0, REPLICATES, 250):
        stop = min(start + 250, REPLICATES)
        selected = rng.integers(0, 10, size=(stop-start, len(groups), 10), dtype=np.int64)
        for j, delta in enumerate(differences):
            bootstrap[j, start:stop] = delta[group_rows, selected].mean(axis=(1, 2))
    results = {}
    for j, model in enumerate(MODELS):
        bounds = np.quantile(bootstrap[j], quantiles, method='linear')
        results[model] = dict(mean_absolute_error_difference=round(float(np.mean(differences[j])), 12),
                              adjusted_percentile_interval=[round(float(x), 12) for x in bounds])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=PANEL)
    parser.add_argument('--check', action='store_true', help='Check the stored intervals')
    args = parser.parse_args()
    folder = args.folder
    protocol = json.loads((folder/'protocol.json').read_bytes())
    if 'acquisition_code' not in protocol or digest(folder/'contexts.npz') != protocol['contexts_sha256']:
        raise ValueError('Expected the source-linked February acquisition graph')
    with np.load(folder/'contexts.npz', allow_pickle=False) as data:
        z = dict(data)
    if check_source(z, protocol):
        raise ValueError('This analysis is defined for the non-mirrored, varying-target-ideal panel')
    acquisition_description(z, protocol)
    predictions, _ = predictions_and_reports(z, folder)
    six = len(MODELS) * len(FAMILIES)
    if six != 6:
        raise ValueError('The prespecified family contains six contrasts')
    tail = FAMILYWISE_ALPHA / (2 * six)
    quantiles = (tail, 1-tail)
    rng = np.random.Generator(np.random.PCG64(SEED))
    families = {}
    for family in FAMILIES:
        indices = np.flatnonzero(z['families'] == family)
        families[family] = bootstrap_family(indices, z, predictions, rng, quantiles)
    report = dict(
        estimand='Secondary mean paired target absolute-error difference, model minus affine CDR. '
                 'A negative value favors the frozen model.',
        scope='Exploratory uncertainty conditional on this one acquisition and the 34 fixed width/depth '
              'groups of each family. The intervals describe within-group circuit resampling. Additional '
              'devices, software versions, calibration and shot draws would introduce other variation.',
        sampling='Within each fixed group, sample ten of its ten circuits with replacement. Keep all '
                 'three checkpoint losses and the CDR loss paired on each draw. Average over all 340 '
                 'drawn targets per family. Families are analysed separately.',
        interval='Bonferroni-adjusted percentile bootstrap intervals for six comparisons, with '
                 'approximate simultaneous 95% coverage conditional on circuit resampling.',
        resamples=REPLICATES, generator='NumPy PCG64', seed=SEED, quantiles=quantiles,
        contrast_family_size=six, group_order='family, width, depth, experiment ID',
        contexts_sha256=protocol['contexts_sha256'], protocol_sha256=digest(folder/'protocol.json'),
        predictions_sha256={m: digest(folder/f'{m}.csv') for m in MODELS}, results=families)
    output = folder/'paired_mae_intervals.json'
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.check:
        if rendered != output.read_text():
            raise ValueError('Recomputed six-contrast bootstrap differs from stored report')
    else:
        output.write_text(rendered)
    print('Verified six paired MAE contrasts on 680 unmirrored February-graph targets.')


if __name__ == '__main__':
    main()
