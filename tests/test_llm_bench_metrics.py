import math
import unittest

from llm_bench.metrics import accept, summarize


class MetricsTests(unittest.TestCase):
    def test_acceptance_boundaries_are_fixed(self):
        self.assertTrue(accept(.02, .20))
        self.assertFalse(accept(.020001, .10))
        self.assertFalse(accept(.01, .200001))
        self.assertFalse(accept(math.nan, .10))
        self.assertFalse(accept(.01, math.inf))

    def test_summary_preserves_raw_order_and_interpolates_percentile(self):
        result = summarize([4, 1, 3, 2])
        self.assertEqual(result['samples_ms'], [4, 1, 3, 2])
        self.assertEqual(result['median_ms'], 2.5)
        self.assertAlmostEqual(result['p95_ms'], 3.85)
        for bad in ([], [0], [-1], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                summarize(bad)


if __name__ == '__main__':
    unittest.main()
