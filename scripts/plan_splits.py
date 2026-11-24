"""Plan nested, disjoint calibration budgets from verified circuit identities."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def plan(records, budgets, seeds, depths=tuple(range(1, 10))):
    if not records or not budgets or not seeds:
        raise ValueError('Records, budgets, and seeds must be nonempty')
    if any(type(b) is not int or b < len(depths) for b in budgets):
        raise ValueError('Budgets must be integers covering every depth')
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be nonnegative integers')
    ids = set()
    test = []
    pool = {d: [] for d in depths}
    calibration_hashes, test_hashes = set(), set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Each record must be an object')
        identity, fingerprint = row.get('id'), row.get('circuit_hash')
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError('Circuit IDs must be nonempty unique strings')
        ids.add(identity)
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError('Missing canonical circuit hash')
        d = row.get('depth')
        if type(d) is not int or d not in depths:
            raise ValueError('Clean track requires integer nonzero depths 1–9')
        partition = row.get('partition')
        if partition == 'test':
            if fingerprint in test_hashes:
                raise ValueError('Repeated canonical circuit in test set')
            test.append(identity)
            test_hashes.add(fingerprint)
        elif partition == 'calibration':
            if fingerprint in calibration_hashes:
                raise ValueError('Repeated canonical calibration circuit')
            calibration_hashes.add(fingerprint)
            pool[d].append(identity)
        else:
            raise ValueError('Partition must be calibration or test')
    if calibration_hashes & test_hashes:
        raise ValueError('Canonical circuit overlap between calibration and test')
    if not test or {r['depth'] for r in records if r['partition'] == 'test'} != set(depths):
        raise ValueError('Test set must contain every evaluation depth')

    runs = []
    for seed in sorted(set(seeds)):
        rng = random.Random(seed)
        ordered = {d: sorted(pool[d]) for d in depths}
        for d in depths:
            rng.shuffle(ordered[d])
        # A fixed rotating order for the remainder preserves nested prefixes.
        remainder_order = list(depths[seed % len(depths):] + depths[:seed % len(depths)])
        for budget in sorted(set(budgets)):
            base, extra = divmod(budget, len(depths))
            counts = {d: base + (d in remainder_order[:extra]) for d in depths}
            shortages = {d: (counts[d], len(pool[d])) for d in depths if counts[d] > len(pool[d])}
            if shortages:
                raise ValueError(f'Budget {budget}, seed {seed}: requested/available by depth {shortages}')
            runs.append({
                'seed': seed, 'budget': budget,
                'calibration_ids': [i for d in depths for i in ordered[d][:counts[d]]],
                'calibration_per_depth': counts,
            })
    return {'protocol': 'clean_nonzero_v1', 'test_ids': sorted(test), 'runs': runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/followup.json').read_text())['clean_curve']
    raw = args.manifest.read_bytes()
    try:
        result = plan(json.loads(raw), args.budgets or config['budgets'], args.seeds or config['seeds'])
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    result['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    result['status'] = 'split_plan_only_not_evaluated'
    # Refuse to overwrite the input manifest or an existing plan.
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
