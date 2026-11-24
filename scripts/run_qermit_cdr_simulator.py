"""Regenerate target-specific Qermit CDR contexts on complete benchmark volume cells."""
import argparse
import hashlib
import inspect
from importlib.metadata import version
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np

from audit_qermit import ROOT, index, read_results, table, verify_inputs

SHOTS = 500_000 // 21
ROOT_SEED = 20261124
OUT = ROOT / 'reports/qermit_cdr_simulation/full_unmirrored'


def upgrade_box(obj):
    # The 2022 PauliExpBox JSON predates the configurable synthesis order.
    # The modern Tree default preserves the unitary; verify every ideal target.
    if obj.get('type') == 'PauliExpBox' and 'paulis' in obj:
        obj.setdefault('cx_config', 'Tree')
    return obj


def expectation(circuit):
    """All released observables are full-width Z products. Parity is endian invariant."""
    state = circuit.get_statevector()
    diagonal = np.asarray([(-1) ** i.bit_count() for i in range(len(state))])
    return float(np.dot(np.abs(state) ** 2, diagonal))


@lru_cache(maxsize=1)
def make_backend():
    from pytket.extensions.qiskit import AerBackend
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel()
    single, double = depolarizing_error(0.001, 1), depolarizing_error(0.01, 2)
    for i in range(7):
        model.add_quantum_error(single, ['h', 'rx', 'rz', 'u'], [i])
        for j in range(7):
            if i != j:
                model.add_quantum_error(double, 'cx', [i, j])
    return AerBackend(noise_model=model, simulation_method='density_matrix'), AerSimulator(
        method='density_matrix', noise_model=model, max_parallel_threads=1)


def measurement_circuit(circuit, backend, level):
    from pytket.extensions.qiskit import tk_to_qiskit
    from qiskit.quantum_info import Pauli

    # Measure the selected logical qubits before compilation, so placement and
    # implicit permutations cannot change which output bits define the Pauli.
    compiled = backend.get_compiled_circuit(circuit.copy().measure_all(), optimisation_level=level)
    qc = tk_to_qiskit(compiled)
    qubits = [qc.find_bit(op.qubits[0]).index for op in qc.data if op.operation.name == 'measure']
    if len(set(qubits)) != circuit.n_qubits:
        raise ValueError('Missing terminal measurements')
    qc.remove_final_measurements()
    qc.save_expectation_value(Pauli('Z' * len(qubits)), qubits, label='parity')
    return qc


def sample(expectations, shots, rng):
    vals = np.asarray(expectations)
    if not np.isfinite(vals).all() or (np.abs(vals) > 1 + 1e-9).any():
        raise ValueError('Invalid Pauli expectation')
    probability = np.clip((1 + vals) / 2, 0, 1)
    return 2 * rng.binomial(shots, probability) / shots - 1


def simulate(circuit, experiment_id, backend, simulator):
    from qermit.clifford_noise_characterisation.ccl import gen_state_circuits

    # Stable per-target streams keep circuit selection and shot draws independent
    # of scheduling, selected cells, archive traversal or other model predictions.
    entropy = int(hashlib.sha256(experiment_id.encode()).hexdigest()[:16], 16)
    generation, sampling = np.random.SeedSequence([ROOT_SEED, entropy]).spawn(2)
    generation_rng = np.random.default_rng(generation)
    rng = np.random.default_rng(sampling)
    for attempt in range(1, 11):
        seed = int(generation_rng.integers(0, 2**32, dtype=np.uint32))
        states = gen_state_circuits(circuit.copy(), 10, 3, 20, seed=seed)
        ideal = np.array([expectation(state) for state in states])
        low_spread = np.all(np.abs(ideal - ideal[0]) <= 0.01)
        if not low_spread:
            break

    # Historical default MitEx uses optimisation level 0 for CDR context
    # and query. The released example uses level 1 for its separate raw arm.
    execution = [measurement_circuit(s, backend, 0) for s in [*states, circuit]]
    execution.append(measurement_circuit(circuit, backend, 1))
    result = simulator.run(execution, shots=1).result()
    if not result.success:
        raise RuntimeError(result.status)
    means = np.array([float(result.data(i)['parity']) for i in range(len(execution))])
    # A Pauli observable is binary. Drawing parity from its exact channel mean
    # gives the same finite-shot marginal as sampling the full bitstrings.
    noisy = sample(means[:20], SHOTS, rng)
    labels = sample(ideal, SHOTS, rng)
    query = float(sample(means[20], SHOTS, rng))
    raw = float(sample(means[21], 100_000, rng))
    weights = np.polyfit(noisy, labels, deg=1)
    pred = float(np.polyval(weights, query))
    hashes = [hashlib.sha256(json.dumps(s.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
              for s in states]
    return dict(context_noisy=noisy, context_ideal=labels, query_noisy=query, raw_noisy=raw,
                context_exact=ideal, context_channel=means[:20], query_channel=means[20],
                raw_channel=means[21], cdr_prediction=pred, surrogate_hashes=hashes,
                attempts=attempt, low_ideal_spread=bool(low_spread), generation_seed=seed,
                quality_warning=np.count_nonzero(np.abs(noisy-query) > 0.1) > 15)


def run_target(job, work):
    from pytket import Circuit

    fields, circuit_json = job
    path = work / (fields['experiment_ids'] + '.npz') if work else None
    if path and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            row = dict(saved)
        if any(row[k] != v for k, v in fields.items()):
            raise ValueError('Unexpected target in resume cache')
        return row
    circuit = Circuit.from_dict(json.loads(circuit_json, object_hook=upgrade_box))
    np.testing.assert_allclose(expectation(circuit), fields['target_ideal'], rtol=0, atol=1e-8)
    values = simulate(circuit, fields['experiment_ids'], *make_backend())
    row = dict(**fields, **values)
    if path:
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f, **row)
        temporary.replace(path)
    return row


def generate(args):
    verified = verify_inputs(args.cache, False)
    notebook = args.cache / 'example_experiment.ipynb'
    if not notebook.exists():
        from urllib.request import urlopen
        with urlopen('https://zenodo.org/api/records/6472281/files/example_experiment.ipynb/content', timeout=60) as response:
            blob = response.read()
        if hashlib.md5(blob).hexdigest() != '0ccf1e27f936ba3a60a192105d5fb258':
            raise ValueError('Unexpected release notebook')
        notebook.write_bytes(blob)
    if hashlib.md5(notebook.read_bytes()).hexdigest() != '0ccf1e27f936ba3a60a192105d5fb258':
        raise ValueError('Unexpected release notebook')
    experiments = index(table(args.cache, 'experiments.csv'), 'experiment ID')
    circuits = index(table(args.cache, 'circuits.csv'), 'circuit ID')
    archived = read_results(args.cache, experiments)
    jobs = []
    started = time.monotonic()
    with ZipFile(args.cache / 'circuits.zip') as circ, ZipFile(args.cache / 'observables.zip') as obs:
        for ident, e in sorted(experiments.items()):
            c = circuits[e['circuit ID']]
            width, depth = int(c['width']), int(c['depth'])
            mirror = c['mirrored'] == 'True'
            if width not in args.widths or depth not in args.depths or (mirror and not (args.include_mirrors or args.mirrors_only)) or (args.mirrors_only and not mirror):
                continue
            measure, = json.loads(obs.read(f"observables/{e['observable ID']}.json"))
            expected = [[['q', [q]], 'Z'] for q in range(width)]
            if measure['coefficient'] != [1, 0] or measure['string'] != expected:
                raise ValueError('Unsupported observable')
            values = archived['depolarising simulator', ident]
            fields = dict(experiment_ids=ident, circuit_ids=e['circuit ID'], families=c['type'],
                          widths=width, depths=depth, mirrored=mirror, target_ideal=float(e['ideal expectation']),
                          archived_raw=values['NOISY'], archived_cdr=values['CDR'], archived_zne=values['ZNE'])
            jobs.append((fields, circ.read(f"circuits/{e['circuit ID']}.json")))
    versions = {p:version(p) for p in ('qermit','pytket','pytket-qiskit','qiskit','qiskit-aer','numpy')}
    work = args.work
    if work:
        defining_code = '\n'.join(inspect.getsource(f) for f in
                                  (upgrade_box, expectation, make_backend, measurement_circuit,
                                   sample, simulate, run_target))
        signature = json.dumps([ROOT_SEED, SHOTS, defining_code, versions, verified], sort_keys=True).encode()
        work = work / hashlib.sha256(signature).hexdigest()[:16]
        work.mkdir(parents=True, exist_ok=True)
    rows = []
    # Spread different circuit sizes and families over all workers. Only the
    # target ID determines its random stream; final ordering is restored below.
    order = np.random.default_rng(0).permutation(len(jobs))
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        futures = [workers.submit(run_target, jobs[i], work) for i in order]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['experiment_ids']}: {row['widths']}q d{row['depths']}, {row['families']}", flush=True)
    rows.sort(key=lambda row: str(row['experiment_ids']))
    if not rows:
        raise ValueError('Empty selection')
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'contexts.npz'
    np.savez_compressed(destination, **{k:np.array([r[k] for r in rows]) for k in rows[0]})
    metadata = dict(
        dataset='New depolarising simulation on original Qermit targets. No hardware was run.',
        source='https://doi.org/10.5281/zenodo.6472281',
        historical_code='https://github.com/Quantinuum/Qermit/tree/aa25c21a9a17aa5b46d03c4152d2d61387356d21/qermit/clifford_noise_characterisation',
        released_example='https://zenodo.org/records/6472281/files/example_experiment.ipynb',
        chronology='Deposited CDR timestamps span 2022-02-03 through 2022-02-06 for the depolarising simulator. The February source is contemporaneous. The later example notebook documents the setup, and the exact acquisition environment is unavailable.',
        generator_code='https://github.com/Quantinuum/Qermit/blob/fba795b895d109b844583d0749334c4951e10ddf/qermit/clifford_noise_characterisation/ccl.py',
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(), source_digests=verified,
        input='One noisy all-Z estimate per target-local surrogate; one independently sampled label from its noiseless simulator.',
        n_context=20, nonclifford_limit=10, n_pairs=3, max_attempts=10, tolerance=0.01,
        distance_tolerance=0.1, calibration_fraction=0.75, selection='Retry when all ideal context means are within 0.01 of first; warning-only at final attempt or poor noisy-context support. Retain all 20 contexts; no fallback or clipping.',
        sampling='One independent binomial draw per measured Pauli expectation from its exact all-Z distribution under Qiskit Aer density-matrix noise or ideal statevector.',
        noise='7-qubit fully connected, local depolarization e1=.001 for H/Rx/Rz/U, e2=.01 for CX, no readout noise.',
        shots_per_context_or_cdr_target=SHOTS, shots_per_fresh_raw=100_000,
        noisy_shots_per_cdr_estimate=SHOTS*21, ideal_simulated_shots_per_context=SHOTS,
        seed=ROOT_SEED, versions=versions,
        differences='Seeded fresh draws, modern pytket rebase/compiler/backend and Tree synthesis for legacy PauliExpBox JSON, and equivalent parity sampling. The original individual surrogate circuits, calibration pairs and random seeds are unavailable. Archived outputs use separate realizations.',
        widths=args.widths, depths=args.depths, include_mirrors=args.include_mirrors or args.mirrors_only, mirrors_only=args.mirrors_only,
        n_targets=len(rows), n_quality_warnings=int(sum(r['quality_warning'] for r in rows)),
        n_low_ideal_spread=int(sum(r['low_ideal_spread'] for r in rows)),
        surrogate_count_distribution=dict(sorted(Counter(len(set(r['surrogate_hashes'])) for r in rows).items())),
        seconds=round(time.monotonic()-started, 3),
        contexts_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (args.out / 'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')


def context_diagnostics(path):
    with np.load(path, allow_pickle=False) as data:
        family_results = {}
        for family in sorted(set(data['families'])):
            select = data['families'] == family
            y = data['target_ideal'][select]
            unique = [len(set(row)) for row in data['surrogate_hashes'][select]]
            errors = {name:float(np.abs(data[name][select] - y).mean())
                      for name in ('archived_raw','raw_noisy','archived_cdr','cdr_prediction','archived_zne')}
            family_results[str(family)] = {
                'n_targets': int(select.sum()),
                'mean_absolute_error': errors,
                'with_quality_warning': int(data['quality_warning'][select].sum()),
                'with_low_spread_at_final_attempt': int(data['low_ideal_spread'][select].sum()),
                'distinct_surrogate_hash_counts': dict(sorted(Counter(unique).items())),
            }
        return {
            'source_npz_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'interpretation': 'Each target has twenty separately sampled noisy and ideal calibration estimates. Distinct hashes count different serialized gate sequences. Repeated surrogate circuits are retained with independent shots. Hardware measurements are absent from these simulation summaries.',
            'raw_comparison': 'The archive and fresh simulation are different measurement realizations. Only new estimators trained on these same per-target pairs can be paired directly with the fresh CDR fit at the same query and shot budget.',
            'families': family_results,
        }


def verify(cache, out):
    """Validate target identities, conditional finite-shot draws and the local linear fits."""
    metadata = json.loads((out / 'protocol.json').read_text())
    recorded_numpy = metadata['versions']['numpy']
    if np.__version__ != recorded_numpy:
        raise RuntimeError(
            f"Exact replay of the recorded binomial draws requires NumPy {recorded_numpy}. "
            f"This environment has {np.__version__}.")
    path = out / 'contexts.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['contexts_sha256']:
        raise ValueError('Context checksum mismatch')
    if verify_inputs(cache, False) != metadata['source_digests']:
        raise ValueError('Source manifest mismatch')
    experiments = index(table(cache, 'experiments.csv'), 'experiment ID')
    circuits = index(table(cache, 'circuits.csv'), 'circuit ID')
    archived = read_results(cache, experiments)
    with np.load(path, allow_pickle=False) as data:
        selected = []
        for ident, e in sorted(experiments.items()):
            c = circuits[e['circuit ID']]
            if (int(c['width']) in metadata['widths'] and int(c['depth']) in metadata['depths']
                    and (metadata['include_mirrors'] or c['mirrored'] == 'False')
                    and (not metadata.get('mirrors_only', False) or c['mirrored'] == 'True')):
                selected.append(ident)
        np.testing.assert_array_equal(data['experiment_ids'], selected)
        if data['context_noisy'].shape != (len(selected), 20) or len(selected) != metadata['n_targets']:
            raise ValueError('Unexpected number of calibration rows or target circuits')
        cells = Counter(zip(data['families'], data['widths'], data['depths'], data['mirrored']))
        if set(cells.values()) != {10}:
            raise ValueError('Incomplete native simulator volume cell')
        for i, ident in enumerate(selected):
            e = experiments[ident]
            c = circuits[e['circuit ID']]
            if (data['circuit_ids'][i] != e['circuit ID'] or data['families'][i] != c['type']
                    or data['widths'][i] != int(c['width']) or data['depths'][i] != int(c['depth'])
                    or data['mirrored'][i] != (c['mirrored'] == 'True')):
                raise ValueError('Circuit identity or cell label mismatch')
            source = archived['depolarising simulator', ident]
            np.testing.assert_array_equal(
                [data[key][i] for key in ('target_ideal','archived_raw','archived_cdr','archived_zne')],
                [float(e['ideal expectation']),source['NOISY'],source['CDR'],source['ZNE']])
            entropy = int(hashlib.sha256(ident.encode()).hexdigest()[:16],16)
            gen, shot = np.random.SeedSequence([ROOT_SEED,entropy]).spawn(2)
            rng = np.random.default_rng(shot)
            attempts = int(data['attempts'][i])
            if not 1 <= attempts <= 10:
                raise ValueError('Invalid number of generation attempts')
            last_seed = np.random.default_rng(gen).integers(0,2**32,size=attempts,dtype=np.uint32)[-1]
            np.testing.assert_equal(int(last_seed), int(data['generation_seed'][i]))
            for measured, mean, n in [('context_noisy','context_channel',SHOTS),('context_ideal','context_exact',SHOTS),
                                      ('query_noisy','query_channel',SHOTS),('raw_noisy','raw_channel',100_000)]:
                np.testing.assert_array_equal(sample(data[mean][i], n, rng), data[measured][i])
            x, y, query = data['context_noisy'][i], data['context_ideal'][i], data['query_noisy'][i]
            cdr = np.polyval(np.polyfit(x,y,1), query)
            np.testing.assert_allclose(cdr, data['cdr_prediction'][i], atol=1e-9, rtol=0)
            np.testing.assert_equal(np.count_nonzero(np.abs(x - query) > 0.1) > 15, data['quality_warning'][i])
            exact = data['context_exact'][i]
            low = np.all(np.abs(exact - exact[0]) <= 0.01)
            if low != data['low_ideal_spread'][i] or (low and attempts < 10):
                raise ValueError('Incorrect state-circuit retry decision')
    print(f'Verified {len(selected)} target-specific 20-pair contexts in {len(cells)} complete Qermit simulator cells.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache',type=Path)
    parser.add_argument('--widths',type=int,nargs='+',default=list(range(2,8)))
    parser.add_argument('--depths', type=int, nargs='+', help='All native depths for the selected mirror condition by default')
    parser.add_argument('--workers', type=int, choices=range(1,9), default=1)
    parser.add_argument('--work', type=Path, help='Optional external per-target cache for resuming simulation')
    views = parser.add_mutually_exclusive_group()
    views.add_argument('--include-mirrors', action='store_true')
    views.add_argument('--mirrors-only', action='store_true')
    parser.add_argument('--out', type=Path, help='Defaults to a separate directory for each mirror condition')
    parser.add_argument('--check', action='store_true', help='Verify saved contexts using only NumPy and pinned archive files')
    args = parser.parse_args()
    mirrored_depths = [2, 4, 6, 8, 16, 32, 64, 128]
    if args.depths is None:
        args.depths = mirrored_depths if args.mirrors_only else list(range(2, 8))
        if args.include_mirrors:
            args.depths = sorted(set(args.depths + mirrored_depths))
    if args.out is None:
        args.out = OUT
        if args.mirrors_only:
            args.out = OUT.parent / 'full_mirrored'
        elif args.include_mirrors:
            args.out = OUT.parent / 'all_targets'
    if args.check:
        verify(args.cache, args.out)
        recorded = args.out / 'context_diagnostics.json'
        if recorded.exists():
            expected = json.loads(recorded.read_text())
            actual = json.loads(json.dumps(context_diagnostics(args.out / 'contexts.npz')))
            for family, stats in actual['families'].items():
                errors = stats.pop('mean_absolute_error')
                saved = expected['families'][family].pop('mean_absolute_error')
                if errors.keys() != saved.keys():
                    raise ValueError('Missing comparator in diagnostics')
                for name, value in errors.items():
                    np.testing.assert_allclose(value, saved[name], rtol=0, atol=1e-12)
            if expected != actual:
                raise ValueError('Context diagnostics mismatch')
    else:
        generate(args)
        verify(args.cache, args.out)
        diagnostics = context_diagnostics(args.out / 'contexts.npz')
        (args.out / 'context_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
