"""Reproduce Qermit Figures 7, 8 and 11–14 and check Tables 3 and 4 against the authors' source."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import tarfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from urllib.request import urlopen

from PIL import Image, ImageChops

from audit_qermit import HARDWARE, METHODS, ROOT, SIMULATORS, index, read_results, table, verify_inputs


SOURCE_URL = 'https://arxiv.org/src/2204.09725v3'
SOURCE_SHA256 = '33edc49e1d9efb7f10e871e3ad30cfa2395f310de5044f0c67f2e61a973ceb82'
OUTPUT = ROOT / 'reports'
FAMILIES = ('Pauli-gadget', 'Random')


def cell_scores(folder):
    experiments = index(table(folder, 'experiments.csv'), 'experiment ID')
    circuits = index(table(folder, 'circuits.csv'), 'circuit ID')
    results = read_results(folder, experiments)
    grouped = defaultdict(list)
    smallest_raw_error = {}
    for (backend, experiment), estimates in results.items():
        target = float(experiments[experiment]['ideal expectation'])
        circuit = circuits[experiments[experiment]['circuit ID']]
        raw_error = abs(estimates['NOISY'] - target)
        if raw_error == 0:
            raise ValueError(f'Undefined paper ratio for {backend}, {experiment}')
        smallest_raw_error[backend] = min(smallest_raw_error.get(backend, raw_error), raw_error)
        for method, prediction in estimates.items():
            absolute_error = abs(prediction - target)
            group = (backend, circuit['type'], circuit['mirrored'] == 'True', method,
                     int(circuit['width']), int(circuit['depth']))
            grouped[group].append((absolute_error, absolute_error / raw_error))

    scores = {}
    for group, losses in sorted(grouped.items()):
        expected = 5 if group[0] in HARDWARE else 10
        if len(losses) != expected:
            raise ValueError(f'Unexpected cell size for {group}: {len(losses)}')
        ratios = [relative for _, relative in losses]
        scores[group] = dict(n=len(losses), mean_absolute_error=mean(x for x, _ in losses),
                             median_relative_error=median(ratios), worst_relative_error=max(ratios))
    return scores, smallest_raw_error


def check_printed_tables(source, scores):
    tex = source.extractfile('errormitg.tex').read().decode()
    tables = re.findall(r'Width & Depth & none.*?\\bottomrule', tex, flags=re.S)
    if len(tables) != 2:
        raise ValueError('Expected the two absolute-error tables')
    tested = {'cells': 0, 'averages': 0}
    mismatches = []
    # The primary source table columns are Lagos then Casablanca, mirrored first.
    headers = list(itertools.product(('ibm_lagos', 'ibmq_casablanca'), (True, False), METHODS))
    for table_number, (family, body) in enumerate(zip(FAMILIES, tables), start=3):
        for line in body.splitlines():
            cols = [c.strip() for c in line.split(r'\\')[0].split('&')]
            if len(cols) != 14 or not (cols[0].isdigit() or cols[0] == 'Average'):
                continue
            average_row = cols[0] == 'Average'
            for printed, (backend, mirrored, method) in zip(cols[2:], headers):
                if not printed:
                    continue
                prefix = (backend, family, mirrored, method)
                if average_row:
                    # Every cell within each published table column has equal size.
                    value = mean(row['mean_absolute_error'] for key, row in scores.items() if key[:4] == prefix)
                else:
                    key = (*prefix, int(cols[0]), int(cols[1]))
                    value = scores[key]['mean_absolute_error']
                tested['averages' if average_row else 'cells'] += 1
                # Several raw means are truncated rather than rounded to three decimals.
                if abs(value - float(printed)) > 0.00100000001:
                    mismatches.append(dict(table=table_number, row=cols[:2], backend=backend,
                                           family=family, mirrored=mirrored, method=method,
                                           printed=float(printed), recomputed=value))
    if tested != {'cells': 192, 'averages': 24}:
        raise ValueError(f'Unexpected table dimensions: {tested}')
    return dict(checked=tested, absolute_tolerance=0.001, outside_tolerance=mismatches)


def colored_spans(image):
    # The axes and background are grayscale. Only the colored median/worst squares remain.
    r, g, b = image.split()
    light = ImageChops.lighter(ImageChops.lighter(r, g), b)
    dark = ImageChops.darker(ImageChops.darker(r, g), b)
    mask = ImageChops.subtract(light, dark).point(lambda x: 255 if x > 40 else 0)

    def runs(projection):
        spans = []
        for visible, block in itertools.groupby(enumerate(projection), key=lambda item: bool(item[1])):
            if visible:
                positions = [i for i, _ in block]
                spans.append((positions[0], positions[-1]))
        return spans

    x, y = mask.getprojection()
    return runs(x), list(reversed(runs(y)))


def figure_checks(source, scores):
    def image(name):
        with Image.open(io.BytesIO(source.extractfile(f'imag/{name}.png').read())) as im:
            return im.convert('RGB')

    bar = image('colour_bar')
    if bar.size != (363, 47):
        raise ValueError('Unexpected source colorbar geometry')
    # Interior pixels between the labelled 0 and 1 ticks, original unscaled PNG.
    palette = [bar.getpixel((x, 15)) for x in range(15, 349)]

    @lru_cache(maxsize=None)
    def palette_value(rgb):
        distances = [max(abs(a - b) for a, b in zip(rgb, entry)) for entry in palette]
        least = min(distances)
        if least > 5:
            raise ValueError(f'Color absent from published scale: {rgb}')
        return mean(i for i, d in enumerate(distances) if d == least) / (len(palette) - 1)

    results = []
    for prefix in sorted({k[:4] for k in scores if k[3] != 'NOISY'}):
        backend, family, mirrored, method = prefix
        panel = 'deep' if family == 'Pauli-gadget' else 'square'
        name = f"{panel}_{backend.replace(' ', '_')}_{mirrored}_{method}"
        png = image(name)
        grid = {key[-2:]: row for key, row in scores.items() if key[:4] == prefix}
        x, y = colored_spans(png)
        depths = sorted({d for _, d in grid})
        widths = sorted({w for w, _ in grid})
        if len(x) != len(depths) or len(y) != len(widths):
            raise ValueError(f'Unexpected published grid axes for {name}')
        largest = 0.
        for (w, d), row in grid.items():
            xa, xb = x[depths.index(d)]
            ya, yb = y[widths.index(w)]
            for statistic, xp in [('median_relative_error', xa + (xb - xa) // 8),
                                  ('worst_relative_error', (xa + xb) // 2)]:
                observed = palette_value(png.getpixel((xp, (ya + yb) // 2)))
                difference = abs(min(1., row[statistic]) - observed)
                if difference > 0.012:
                    raise ValueError(f'Figure disagrees for {name} at width {w}, depth {d}, {statistic}')
                largest = max(largest, difference)
        results.append(dict(source_panel=name, cells=len(grid), color_samples=2 * len(grid),
                            max_colorbar_difference=round(largest, 8)))
    return dict(figure_numbers=[7, 8, 11, 12, 13, 14], panels=results,
                colored_values_checked=sum(p['color_samples'] for p in results),
                range=[0, 1], absolute_colorbar_tolerance=0.012,
                interpretation='Native publication PNGs encode median on the outer square and worst case at center. '
                               'Ratios at or above one saturate in red. The CSV preserves uncapped values. '
                               'Color lookup checks the displayed precision and cannot recover precise ratios above one.')


def hardware_summary(scores):
    batches = []
    for backend, mirrored, method in itertools.product(HARDWARE, (False, True), ('CDR', 'ZNE')):
        rows = [r for (b, _, mirror, m, _, _), r in scores.items() if (b, mirror, m) == (backend, mirrored, method)]
        batches.append(dict(backend=backend, mirrored=mirrored, method=method, cells=len(rows),
                            median_below_one=sum(r['median_relative_error'] < 1. for r in rows),
                            median_equal_one=sum(r['median_relative_error'] == 1. for r in rows),
                            worst_below_one=sum(r['worst_relative_error'] < 1. for r in rows)))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Qermit cache (uses the seven pinned availability-audit files)')
    parser.add_argument('--source', type=Path, help='arXiv v3 TeX and original panel archive')
    parser.add_argument('--download', action='store_true', help='Fetch missing source files')
    parser.add_argument('--check', action='store_true', help='Check both committed outputs without modifying them')
    args = parser.parse_args()
    files = verify_inputs(args.folder, args.download)
    source_path = args.source or args.folder / 'qermit-arxiv-v3.tar.gz'
    if not source_path.exists() and args.download:
        with urlopen(SOURCE_URL, timeout=90) as stream:
            blob = stream.read()
        if hashlib.sha256(blob).hexdigest() != SOURCE_SHA256:
            raise ValueError('Unexpected arXiv source archive')
        source_path.write_bytes(blob)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Source file must match the exact published arXiv v3 archive')
    scores, smallest_raw = cell_scores(args.folder)
    with tarfile.open(source_path) as source:
        tables = check_printed_tables(source, scores)
        figures = figure_checks(source, scores)

    table_out = io.StringIO(newline='')
    writer = csv.writer(table_out, lineterminator='\n')
    writer.writerow(['backend', 'family', 'mirrored', 'method', 'width', 'depth', 'n',
                     'mean_absolute_error', 'median_relative_error', 'worst_relative_error'])
    for key, row in scores.items():
        writer.writerow([*key, row['n'], *[f'{row[k]:.12g}' for k in
                          ['mean_absolute_error', 'median_relative_error', 'worst_relative_error']]])
    csv_payload = table_out.getvalue()
    report = dict(
        source_data='https://doi.org/10.5281/zenodo.6472281', release_files_sha256={k: e['sha256'] for k,e in files.items()},
        paper='https://doi.org/10.22331/q-2023-07-13-1059', source_figure_and_table_archive=SOURCE_URL,
        source_sha256=SOURCE_SHA256, baseline_methods=list(METHODS), device_backends=list(HARDWARE),
        simulated_backends=list(SIMULATORS), estimand='Absolute error against archived ideal, divided by absolute '
            'error of the raw estimate for the same experiment ID and backend. Aggregate the ratios over circuits '
            'within circuit family, mirror status, width and depth, using median and maximum.',
        circuits_per_cell={'hardware': 5, 'simulators': 10}, hardware_target_ids=160,
        training_protocol='CDR constructs 20 near-Clifford training circuits separately for each target. '
            'Figures 7 and 8 specify no global supervised calibration subset. Re-score the archived final outputs.',
        hardware_shots_per_target=dict(raw=32000, zne_five_noise_scales=160000,
            cdr_nominal=160000, cdr_per_program=7619, cdr_total_from_notebook_integer_division=159999),
        shot_source='Section 4.6 and the Zenodo example_experiment.ipynb give the protocol and allocation. '
            'These are planned executions. The release provides aggregate results and does not expose actual '
            'job-level usage or each near-Clifford measurement.',

        hardware_cell_method_rows=sum(key[0] in HARDWARE for key in scores), total_cell_method_rows=len(scores),
        smallest_raw_absolute_error=smallest_raw, undefined_relative_errors=0,
        zero_policy='Exact zero raw error makes the paper ratio undefined. Fail if any occurs in this pinned archive. '
            'Keep every finite ratio and every released mitigation result, including worsened estimates.',
        hardware_cell_improvements=hardware_summary(scores), figures=figures, absolute_error_tables=tables,
        cells_file_sha256=hashlib.sha256(csv_payload.encode()).hexdigest(),
    )
    csv_path = OUTPUT / 'qermit_reference_cells.csv'
    json_path = OUTPUT / 'qermit_reference.json'
    if args.check:
        if report != json.loads(json_path.read_bytes()) or csv_payload != csv_path.read_text():
            raise ValueError('Recomputed Qermit results differ from the committed reference')
    else:
        csv_path.write_text(csv_payload)
        json_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {figures['colored_values_checked']} plotted values across {len(figures['panels'])} panels; "
          f"tabulated {len(scores)} cell–method rows. Table discrepancies: {len(tables['outside_tolerance'])}.")


if __name__ == '__main__':
    main()
