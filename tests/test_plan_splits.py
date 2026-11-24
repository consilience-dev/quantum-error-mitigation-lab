import unittest
from scripts.plan_splits import plan


def fixture(n=60):
    return [
        {'id': f'{partition}-{d}-{i}', 'circuit_hash': f'hash-{partition}-{d}-{i}',
         'depth': d, 'partition': partition}
        for d in range(1, 10)
        for partition, count in [('calibration', n), ('test', 20)]
        for i in range(count)
    ]


class SplitsTest(unittest.TestCase):
    def test_nested_balanced_full_500_and_fixed_test(self):
        records = fixture()
        result = plan(records, [18, 45, 90, 180, 270, 450, 500], range(5))
        self.assertEqual(len(result['test_ids']), 180)
        for seed in range(5):
            previous = set()
            for run in [r for r in result['runs'] if r['seed'] == seed]:
                selected = set(run['calibration_ids'])
                self.assertEqual(len(selected), run['budget'])
                self.assertTrue(previous <= selected)
                self.assertFalse(selected & set(result['test_ids']))
                counts = list(run['calibration_per_depth'].values())
                self.assertLessEqual(max(counts) - min(counts), 1)
                previous = selected
        self.assertEqual(result, plan(list(reversed(records)), [18, 45, 90, 180, 270, 450, 500], range(5)))

    def test_original_450_pool_cannot_provide_500(self):
        self.assertEqual(len(plan(fixture(50), [450], [0])['runs'][0]['calibration_ids']), 450)
        with self.assertRaisesRegex(ValueError, 'requested/available'):
            plan(fixture(50), [500], [0])

    def test_different_ids_same_circuit_cannot_leak(self):
        rows = fixture()
        test_row = next(r for r in rows if r['partition'] == 'test')
        test_row['circuit_hash'] = rows[0]['circuit_hash']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            plan(rows, [18], [0])

    def test_repeated_calibration_cannot_inflate_budget(self):
        rows = fixture()
        rows[1]['circuit_hash'] = rows[0]['circuit_hash']
        with self.assertRaisesRegex(ValueError, 'Repeated canonical'):
            plan(rows, [18], [0])

    def test_depth_zero_and_duplicate_ids_rejected(self):
        rows = fixture()
        rows[0]['depth'] = 0
        with self.assertRaises(ValueError):
            plan(rows, [18], [0])
        rows = fixture()
        rows[-1]['id'] = rows[0]['id']
        with self.assertRaises(ValueError):
            plan(rows, [18], [0])


if __name__ == '__main__':
    unittest.main()
