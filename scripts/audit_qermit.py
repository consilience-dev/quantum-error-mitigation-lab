"""Audit the primary Qermit archive and identify eligible sim-to-hardware circuits."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import mean
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RECORD = 'https://zenodo.org/api/records/6472281'
FILES = {
    'circuits.csv': 'bdb5df44879a8e27b6361705729dfbb3',
    'circuits.zip': '633aab94b407c93bb4674bf24b5b727f',
    'device_experiments.csv': '0bb58916894410280ce32c53a3e9d513',
    'experiments.csv': 'a34aca2f21d196ffca1e0aa4392e2a81',
    'observables.zip': '982459a252630dff94527214bebdf44a',
    'results.csv': '427fc66a8caa305c9c1ef4e9ed6cbcf8',
    'results.zip': 'b3c392d25bae0597f0e7aed489799480',
}
METHODS = ('NOISY', 'ZNE', 'CDR')
HARDWARE = ('ibmq_casablanca', 'ibm_lagos')
SIMULATORS = ('depolarising simulator', 'ibm_lagos emulator')


def verify_inputs(folder, download):
    files = {}
    for name, md5 in FILES.items():
        path = folder / name
        url = f'{RECORD}/files/{name}/content'
        if path.exists():
            blob = path.read_bytes()
        elif download:
            with urlopen(url, timeout=75) as response:
                blob = response.read()
        else:
            raise FileNotFoundError(f'Missing {path}. Add --download to fetch the verified inputs.')
        if hashlib.md5(blob).hexdigest() != md5:
            raise ValueError(f'Release checksum mismatch for {name}')
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        files[name] = {'url': url, 'md5': md5, 'sha256': hashlib.sha256(blob).hexdigest()}
    return files


def table(folder, name):
    with (folder / name).open() as f:
        return list(csv.DictReader(f))


def index(rows, key):
    indexed = {r[key]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return indexed


def fingerprint(circuit_zip, ident):
    # pytket assigns a UUID to each box instance even for identical operations.
    def remove_box_uuid(obj):
        if 'box' in obj:
            obj['box'].pop('id', None)
        return obj

    circuit = json.loads(circuit_zip.read(f'circuits/{ident}.json'), object_hook=remove_box_uuid)
    canonical = json.dumps(circuit, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(canonical).hexdigest()


def read_results(folder, experiments):
    rows = table(folder, 'results.csv')
    index(rows, 'result ID')
    estimates = defaultdict(dict)
    with ExitStack() as stack:
        results = stack.enter_context(ZipFile(folder / 'results.zip'))
        obs = stack.enter_context(ZipFile(folder / 'observables.zip'))
        for row in rows:
            key = row['experiment ID']
            observable_id = experiments[key]['observable ID']
            observable, = json.loads(obs.read(f'observables/{observable_id}.json'))
            result, = json.loads(results.read(f"results/{row['result ID']}.json"))
            method = row['Mitex']
            expected = observable['string']
            actual = result['string']
            if method == 'ZNE':
                # The ZNE writer used backend nodes after routing. The experiment
                # ID links it to the logical target. Check the Pauli factors too.
                matches = (all(q[0] == 'node' for q, _ in actual) and
                           sorted(p for _, p in actual) == sorted(p for _, p in expected))
            else:
                matches = actual == expected
            real, imag = result['coefficient']
            block = estimates[(row['backend name'], key)]
            if (method not in METHODS or method in block or not matches or observable['coefficient'] != [1, 0]
                    or imag != 0 or not math.isfinite(real)):
                raise ValueError(f"Unexpected result {row['result ID']}")
            block[method] = real
    if any(set(block) != set(METHODS) for block in estimates.values()):
        raise ValueError('A circuit/observable/backend is missing a method output')
    return estimates


def audit(folder, files):
    circuits = index(table(folder, 'circuits.csv'), 'circuit ID')
    experiments = index(table(folder, 'experiments.csv'), 'experiment ID')
    by_circuit = index(list(experiments.values()), 'circuit ID')
    if set(circuits) != set(by_circuit):
        raise ValueError('The experiment and circuit populations differ')
    device_rows = index(table(folder, 'device_experiments.csv'), 'experiment ID')
    if any(experiments[k] != r for k, r in device_rows.items()):
        raise ValueError('Hardware experiment specifications differ from the master list')
    estimates = read_results(folder, experiments)
    nonmirrored = {k for k, r in experiments.items() if circuits[r['circuit ID']]['mirrored'] == 'False'}
    devices = {}
    for backend in (*HARDWARE, *SIMULATORS):
        ids = {k for b, k in estimates if b == backend}
        if backend in HARDWARE and ids != set(device_rows):
            raise ValueError(f'Hardware result IDs differ from device_experiments.csv on {backend}')
        groups = {}
        for name, keys in [('unmirrored', ids & nonmirrored), ('mirrored', ids - nonmirrored)]:
            ordered = sorted(keys)
            targets = [float(experiments[k]['ideal expectation']) for k in ordered]
            groups[name] = {
                'complete_circuit_observable_ids': len(ordered),
                'circuit_families': dict(Counter(circuits[experiments[k]['circuit ID']]['type'] for k in ordered)),
                'ideal_range': [min(targets), max(targets)],
                'ideal_values_within_1e-12_of_one': sum(abs(v - 1) < 1e-12 for v in targets),
            }
            if backend in HARDWARE:
                groups[name]['archived_mean_absolute_error'] = {
                    method: round(mean(abs(estimates[backend, k][method] - target)
                                       for k, target in zip(ordered, targets)), 9) for method in METHODS
                }
        devices[backend] = groups

    test = set(device_rows) & nonmirrored
    eligible = nonmirrored - test
    for simulator in SIMULATORS:
        ids = {k for b, k in estimates if b == simulator} & nonmirrored
        if ids != nonmirrored:
            raise ValueError('Unexpected nonmirrored simulator population')
    with ZipFile(folder / 'circuits.zip') as source:
        hashes = {k: fingerprint(source, experiments[k]['circuit ID']) for k in nonmirrored}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('Repeated unmirrored circuit structure; group these before splitting')

    def population(keys):
        ids = sorted(experiments[k]['circuit ID'] for k in keys)
        return {'circuit_ids': len(ids), 'distinct_structural_hashes': len({hashes[k] for k in keys}),
                'circuit_families': dict(Counter(circuits[i]['type'] for i in ids)),
                'sorted_circuit_ids_sha256': hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()}

    return {
        'release': 'https://doi.org/10.5281/zenodo.6472281',
        'paper': 'https://quantum-journal.org/papers/q-2023-07-13-1059/',
        'files': files,
        'scope': 'Availability and reaggregation of authors’ completed runs. No pretrained model was evaluated.',
        'pairing': 'Join the experiment ID, circuit ID and observable ID, then the released method result ID by backend. ZNE uses routed node-labelled Pauli terms; its association to a logical observable relies on the authors’ experiment ID.',
        'archived_methods': list(METHODS), 'complete_backend_experiment_pairs': len(estimates),
        'total_method_results': len(estimates) * len(METHODS), 'unique_device_experiment_ids': len(device_rows),
        'devices': devices,
        'proposed_sim_to_hardware': {
            'common_hardware_test_population': population(test),
            'available_calibration_population_in_either_simulator': population(eligible),
            'overlapping_structural_hashes': len({hashes[k] for k in test} & {hashes[k] for k in eligible}),
            'scope': 'Unmirrored circuits only. Both physical devices have identical test identities. The two simulators share the same calibration IDs. Hardware-only calibration and simulation-trained transfer use different resources.',
            'fingerprint': 'SHA-256 of the sorted-key circuit JSON after omitting per-box UUIDs. This checks identical circuit structure. Functional equivalence is untested.',
        },
        'paper_protocol': {
            'per_target_hardware_shots': {'NOISY': 32000, 'ZNE': 160000, 'CDR': 160000},
            'zne': 'Exponential fit at folding factors 1, 3, 5, 7 and 9.',
            'cdr': '20 near-Clifford auxiliary circuits and the unaltered target circuit.',
            'native_metric': 'Relative error of mitigation, absolute mitigated error divided by absolute raw error. The paper displays median and worst-case values within width/depth cells for each circuit class.',
            'interpretation': 'The pooled MAEs in this audit are new arithmetic summaries of the published scalar outputs. Shot budgets are taken from the paper. Counts, PEC signed runs and auxiliary target-level samples are absent from these audited result tables.',
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Directory of Qermit archive inputs, outside the tracked repository')
    parser.add_argument('--download', action='store_true', help='Fetch missing input files from the pinned Zenodo record')
    parser.add_argument('--check', action='store_true', help='Verify the committed availability report without writing')
    args = parser.parse_args()
    report = audit(args.folder, verify_inputs(args.folder, args.download))
    output = ROOT / 'reports/qermit_availability.json'
    if args.check:
        if report != json.loads(output.read_text()):
            raise ValueError('Qermit availability report does not match')
    else:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f"Verified {report['total_method_results']:,} Qermit outputs across both hardware devices and two simulators.")


if __name__ == '__main__':
    main()
