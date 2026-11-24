"""Replay recorded generation seeds with the February 2022 Qermit/pytket stack."""
import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pytket import Circuit
from qermit.clifford_noise_characterisation.ccl import gen_state_circuits

from audit_qermit import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modern', type=Path,
                        default=ROOT / 'reports/qermit_cdr_simulation/full_unmirrored')
    args = parser.parse_args()
    source = Path(inspect.getfile(gen_state_circuits))
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_digest != 'd89e243d42805a84562ed82792be04d09a99c8a588293fd70ec3e520474110e5':
        raise ValueError('Run with the acquisition-period Qermit CDR source')
    if version('pytket') != '0.19.2':
        raise ValueError('This controlled comparison uses pytket 0.19.2')
    modern = args.modern / 'contexts.npz'
    protocol = json.loads((args.modern / 'protocol.json').read_text())
    if hashlib.sha256(modern.read_bytes()).hexdigest() != protocol['contexts_sha256']:
        raise ValueError('Modern context checksum mismatch')
    archive = args.cache / 'circuits.zip'
    if hashlib.md5(archive.read_bytes()).hexdigest() != '633aab94b407c93bb4674bf24b5b727f':
        raise ValueError('Circuit archive checksum mismatch')

    records = []
    with np.load(modern, allow_pickle=False) as saved, ZipFile(archive) as circuits:
        data = dict(saved)
        selected = np.flatnonzero((data['widths'] == args.width) &
                                  (data['depths'] == args.depth) & ~data['mirrored'])
        parity = np.array([(-1) ** bin(i).count('1') for i in range(2 ** args.width)])
        for i in selected:
            seed = int(data['generation_seed'][i])
            circuit = Circuit.from_dict(json.loads(circuits.read('circuits/' + data['circuit_ids'][i] + '.json')))
            states = gen_state_circuits(circuit, 10, 3, 20, seed=seed)
            ideal = np.array([np.dot(abs(state.get_statevector()) ** 2, parity) for state in states])
            records.append(dict(experiment_id=str(data['experiment_ids'][i]), family=str(data['families'][i]),
                                generation_seed=seed, legacy_ideal=ideal.tolist(),
                                mean_absolute_difference=float(np.mean(abs(ideal - data['context_exact'][i])))))
    families = {}
    for name in sorted({r['family'] for r in records}):
        rows = [r for r in records if r['family'] == name]
        if len(rows) != 10:
            raise ValueError('Incomplete native cell')
        families[name] = {'n_targets': len(rows),
                          'mean_absolute_difference': float(np.mean([r['mean_absolute_difference'] for r in rows]))}
    if len(families) != 2:
        raise ValueError('Both native circuit families are required')
    result = dict(acquisition_code='aa25c21a9a17aa5b46d03c4152d2d61387356d21',
                  generator_sha256=source_digest,
                  pytket=version('pytket'), numpy=version('numpy'),
                  modern_context_sha256=protocol['contexts_sha256'], width=args.width, depth=args.depth,
                  comparison='Same released target and final integer generation seed used by the modern run. Compare the 20 exact, noiseless auxiliary expectations in original generator output order, before shot sampling or fitting.',
                  families=families, records=records)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(families, indent=2))


if __name__ == '__main__':
    main()
