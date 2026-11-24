"""Replay per-target CDR with an acquisition-period Qermit/pytket/Aer stack."""
import argparse
import hashlib
import inspect
import json
import random
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from audit_qermit import index, read_results, table, verify_inputs

SHOTS = 500_000 // 21
SEED = 20261124
SOURCE = 'aa25c21a9a17aa5b46d03c4152d2d61387356d21'
GENERATOR_DIGEST = 'd89e243d42805a84562ed82792be04d09a99c8a588293fd70ec3e520474110e5'
FITTER_DIGEST = '4c633304a83d58024263dd0e4ebdca6af7e069b0770b96acd6f1db363237d860'


def stack():
    from qermit.clifford_noise_characterisation import ccl, cdr_post

    for module, expected in [(ccl, GENERATOR_DIGEST), (cdr_post, FITTER_DIGEST)]:
        if hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() != expected:
            raise ValueError('Install the recorded acquisition-period Qermit source')
    packages = {name: version(name) for name in ('pytket', 'pytket-qiskit', 'qiskit', 'qiskit-aer', 'numpy')}
    compiler_pair = packages['pytket'], packages['pytket-qiskit']
    if compiler_pair not in {('0.19.2', '0.22.1'), ('0.19.0', '0.22.0')}:
        raise ValueError('Use a documented February compiler/adapter pair')
    return packages


def depolarising_model():
    from qiskit.providers.aer import noise

    model = noise.NoiseModel()
    single = noise.depolarizing_error(0.001, 1)
    double = noise.depolarizing_error(0.01, 2)
    for i in range(7):
        model.add_quantum_error(single, ['h', 'rx', 'rz', 'u'], [i])
        for j in range(i):
            model.add_quantum_error(double, 'cx', [i, j])
            model.add_quantum_error(double, 'cx', [j, i])
    return model


def channel_means(circuits, noise_model):
    """The CDR graph has already compiled and placed these measured circuits."""
    from pytket.extensions.qiskit import tk_to_qiskit
    from qiskit.providers.aer import AerSimulator
    from qiskit.quantum_info import Pauli

    batch = []
    for circuit in circuits:
        qc = tk_to_qiskit(circuit)
        measured = [qc.find_bit(bits[0]).index for op, bits, _ in qc.data if op.name == 'measure']
        if len(set(measured)) != len(circuit.bits):
            raise ValueError('Missing measured qubits in acquisition-era circuit')
        qc.remove_final_measurements()
        qc.save_expectation_value(Pauli('Z' * len(measured)), measured, 'parity')
        batch.append(qc)
    density = AerSimulator(method='density_matrix', max_parallel_threads=1, noise_model=noise_model)
    result = density.run(batch, shots=1).result()
    if not result.success:
        raise RuntimeError(result.status)
    return np.array([float(result.data(i)['parity']) for i in range(len(batch))])


def capture_target(job, shot_memory):
    from pytket import Circuit
    from pytket.extensions.qiskit import AerBackend
    from pytket.utils import QubitPauliOperator
    from qermit import AnsatzCircuit, MitEx, ObservableExperiment, ObservableTracker, SymbolsDict
    from qermit.clifford_noise_characterisation import ccl, gen_CDR_MitEx
    from qermit.clifford_noise_characterisation.cdr_post import _PolyCDRCorrect
    from qermit.taskgraph.mitex import gen_compiled_MitRes

    fields, serialized, observable = job
    ident = fields['experiment_ids']
    entropy = int(hashlib.sha256(ident.encode()).hexdigest()[:16], 16)
    generation, device_seed, ideal_seed, raw_seed = np.random.SeedSequence([SEED, entropy]).spawn(4)
    initial = int(np.random.default_rng(generation).integers(0, 2**32, dtype=np.uint32))
    random.seed(initial)
    np.random.seed(initial)
    means, values = {}, {}

    class SeededAer(AerBackend):
        def __init__(self, seed, arm, noise=None):
            super().__init__(noise)
            self._arm = arm
            self._stream = np.random.default_rng(seed)
            self._memory = shot_memory
            self._backend.set_options(max_parallel_threads=1)

        def process_circuits(self, circuits, **kwargs):
            if self._noise_model is not None:
                means[self._arm, len(circuits)] = channel_means(circuits, self._noise_model)
            kwargs['seed'] = int(self._stream.integers(1, 2**31))
            return super().process_circuits(circuits, **kwargs)

    class RecordingFit(_PolyCDRCorrect):
        def calibrate(self, x, y):
            values['context_noisy'] = np.array(x, dtype=float)
            values['context_ideal'] = np.array(y, dtype=float)
            return super().calibrate(x, y)

        def correct(self, query):
            values['query_noisy'] = float(query)
            return super().correct(query)

    original_gen = ccl.gen_state_circuits
    original_ideal = ccl.get_operator_expectation_value
    attempts = 0
    exact = []

    def record_states(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        exact.clear()
        states = original_gen(*args, **kwargs)
        values['surrogate_hashes'] = [
            hashlib.sha256(json.dumps(s.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for s in states]
        return states

    def record_ideal(*args, **kwargs):
        value = original_ideal(*args, **kwargs)
        exact.append(float(np.real(complex(value))))
        return value

    circuit = Circuit.from_dict(json.loads(serialized))
    parity = np.array([(-1) ** bin(i).count('1') for i in range(2 ** circuit.n_qubits)])
    np.testing.assert_allclose(np.dot(abs(circuit.get_statevector()) ** 2, parity), fields['target_ideal'],
                               rtol=0, atol=1e-8)

    def wire(shots):
        qpo = QubitPauliOperator.from_list(json.loads(observable))
        ansatz = AnsatzCircuit(circuit.copy(), shots, SymbolsDict())
        return [ObservableExperiment(ansatz, ObservableTracker(qpo))]

    device = SeededAer(device_seed, 'cdr', depolarising_model())
    ideal = SeededAer(ideal_seed, 'ideal')
    with warnings.catch_warnings(record=True), patch.object(ccl, 'gen_state_circuits', record_states), \
            patch.object(ccl, 'get_operator_expectation_value', record_ideal):
        mitigation = gen_CDR_MitEx(device, ideal, 10, 3, 20, calibration_fraction=0.75, model=RecordingFit(1))
        result, = mitigation.run(wire(SHOTS))
    values['cdr_prediction'] = float(next(iter(result._dict.values())))
    raw = SeededAer(raw_seed, 'raw', depolarising_model())
    raw_result, = MitEx(raw, mitres=gen_compiled_MitRes(raw, 1)).run(wire(100_000))
    values['raw_noisy'] = float(next(iter(raw_result._dict.values())))
    x, y, q = values['context_noisy'], values['context_ideal'], values['query_noisy']
    if len(exact) != 20 or x.shape != (20,) or y.shape != (20,):
        raise ValueError('Incomplete historical Qermit calibration set')
    np.testing.assert_allclose(np.polyval(np.polyfit(x, y, 1), q), values['cdr_prediction'], atol=1e-10)
    return dict(**fields, **values, context_channel=means['cdr', 20], query_channel=means['cdr', 1][0],
                raw_channel=means['raw', 1][0], context_exact=exact, attempts=attempts, initial_seed=initial,
                quality_warning=np.count_nonzero(abs(x - q) > 0.1) > 15,
                low_ideal_spread=np.all(np.abs(np.array(exact) - exact[0]) <= 0.01))


def run(job, cache, shot_memory):
    path = cache / (job[0]['experiment_ids'] + '.npz')
    if not path.exists():
        row = capture_target(job, shot_memory)
        with path.with_suffix('.tmp').open('wb') as tmp:
            np.savez_compressed(tmp, **row)
        path.with_suffix('.tmp').replace(path)
    with np.load(path, allow_pickle=False) as data:
        row = dict(data)
    if any(row[k] != v for k, v in job[0].items()):
        raise ValueError('Unexpected cached experiment')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--widths', nargs='+', type=int, default=list(range(2, 8)))
    parser.add_argument('--depths', nargs='+', type=int, default=list(range(2, 8)))
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 9))
    parser.add_argument('--shot-memory', action='store_true',
                        help='Return every bitstring to pytket before its parity count aggregation.')
    args = parser.parse_args()
    packages = stack()
    digests = verify_inputs(args.archive, False)
    code = Path(__file__).read_bytes()
    signature = hashlib.sha256(code + json.dumps([packages, digests, args.shot_memory],
                                                  sort_keys=True).encode()).hexdigest()
    cache = args.work / signature[:16]
    cache.mkdir(exist_ok=True, parents=True)
    expts = index(table(args.archive, 'experiments.csv'), 'experiment ID')
    circuits = index(table(args.archive, 'circuits.csv'), 'circuit ID')
    archived = read_results(args.archive, expts)
    jobs = []
    with ZipFile(args.archive / 'circuits.zip') as c, ZipFile(args.archive / 'observables.zip') as z:
        for ident, experiment in sorted(expts.items()):
            circuit_id = experiment['circuit ID']
            circuit = circuits[circuit_id]
            width, depth = int(circuit['width']), int(circuit['depth'])
            if circuit['mirrored'] != 'False' or width not in args.widths or depth not in args.depths:
                continue
            obs = z.read('observables/' + experiment['observable ID'] + '.json')
            expected = [{'coefficient': [1, 0], 'string': [[['q', [i]], 'Z'] for i in range(width)]}]
            if json.loads(obs) != expected:
                raise ValueError('Unsupported observable')
            reference = archived['depolarising simulator', ident]
            info = dict(experiment_ids=ident, circuit_ids=circuit_id, families=circuit['type'],
                        widths=width, depths=depth, mirrored=False, target_ideal=float(experiment['ideal expectation']),
                        archived_raw=reference['NOISY'], archived_cdr=reference['CDR'], archived_zne=reference['ZNE'])
            jobs.append((info, c.read('circuits/' + circuit_id + '.json'), obs))
    cells = Counter((j[0]['families'], j[0]['widths'], j[0]['depths']) for j in jobs)
    if set(cells.values()) != {10}:
        raise ValueError('Incomplete benchmark volume cells')
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, jobs[i], cache, args.shot_memory) for i in np.random.default_rng(0).permutation(len(jobs))]
        try:
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"{len(rows)}/{len(jobs)} {row['experiment_ids']} {row['families']}", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    rows.sort(key=lambda v: str(v['experiment_ids']))
    args.out.mkdir(exist_ok=True, parents=True)
    destination = args.out / 'contexts.npz'
    np.savez_compressed(destination, **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
    metadata = dict(dataset='Fresh per-target CDR on the original unmirrored depolarising targets.',
                    acquisition_code=SOURCE, generator_sha256=GENERATOR_DIGEST, fitter_sha256=FITTER_DIGEST,
                    versions=packages, seed=SEED, shots_per_context_or_cdr_target=SHOTS,
                    shots_per_fresh_raw=100_000, noisy_shots_per_cdr_estimate=21 * SHOTS,
                    ideal_simulated_shots_per_context=SHOTS, n_context=20, n_targets=len(rows),
                    widths=args.widths, depths=args.depths, include_mirrors=False, mirrors_only=False,
                    selection='Native Qermit state generation, exact diversity test and up to ten attempts, warning-only proximity test, all 20 retained.',
                    experiment='Native gen_CDR_MitEx. Actual Aer shot counts used by all fits; exact channel means calculated independently on the same compiled circuits.',
                    resource_mode='bitstring memory' if args.shot_memory else 'count histograms',
                    differences='Public acquisition-era implementation and a tested dependency range. Original installed software, random seeds and auxiliary measurements are unavailable.',
                    n_quality_warnings=int(sum(r['quality_warning'] for r in rows)),
                    n_low_ideal_spread=int(sum(r['low_ideal_spread'] for r in rows)),
                    source_digests=digests, seconds=round(time.monotonic() - started, 3),
                    contexts_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (args.out / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
