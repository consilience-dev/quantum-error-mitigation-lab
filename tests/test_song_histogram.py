import unittest

from scripts.song_readout_models import histogram_bounds


class SongHistogramTest(unittest.TestCase):
    def test_rounded_mean_bounds_cover_all_discrete_allocations(self):
        # Each of two bins contains 50 blocks. Only two shot counts are
        # possible per bin, so the complete set of allocations can be enumerated.
        bins = [{'blocks': 50, 'raw_plus_count_min': lo, 'raw_plus_count_max': lo + 1}
                for lo in (1497, 1503)]
        records = {'raw_bins': bins, 'reported': {'raw': {'mean': 0.0}}}
        for sign in (1, -1):
            predictions = {k: sign * y for k, y in [(1497, 0.1), (1498, 0.6),
                                                    (1503, -0.2), (1504, 0.8)]}
            possible = []
            for i in range(51):
                for j in range(51):
                    counts = {1497: 50 - i, 1498: i, 1503: 50 - j, 1504: j}
                    # Use integer counts to include points exactly on the rounding boundary.
                    total_plus = sum(k * n for k, n in counts.items())
                    if abs(2 * total_plus - 300000) <= 150:
                        possible.append(sum(predictions[k] * n for k, n in counts.items()) / 100)
            bounds = histogram_bounds(predictions, records)
            low, high = bounds['including_rounded_raw_mean']
            self.assertAlmostEqual(low, min(possible))
            self.assertAlmostEqual(high, max(possible))
            self.assertLess(high - low, bounds['histogram_only'][1] - bounds['histogram_only'][0])
