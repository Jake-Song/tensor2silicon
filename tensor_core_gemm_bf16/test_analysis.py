"""CPU-only checks for result provenance and profiler unit conversion."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze import summarize
from profile_summary import read


def records():
    rows = []
    for name, ms in [('cublas', 2.0), ('native', 1.0)]:
        for phase in ['before', 'after']:
            rows.append(dict(kind='check', variant=name, m=128, n=128, k=128,
                             seed=42, group=8, phase=phase, **{'pass': True}))
        for round_id in range(5):
            rows.append(dict(kind='timing', variant=name, size=128, seed=42, group=8,
                             stage=100 if name == 'cublas' else 6, ms=ms,
                             operation='D=A@B', timing='cuda_graph', round=round_id))
    return rows


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'results.jsonl'

    def tearDown(self):
        self.directory.cleanup()

    def write(self, rows, path=None):
        target = path or self.path
        target.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return target

    def test_paired_baseline_and_sample_count(self):
        result = summarize([self.write(records())])
        native = next(r for r in result if r['variant'] == 'native')
        self.assertEqual(native['samples'], 5)
        self.assertEqual(native['cublas_ratio'], 2.0)

    def test_failed_check_blocks_performance(self):
        rows = records()
        rows[0]['pass'] = False
        with self.assertRaisesRegex(ValueError, 'Correctness failure'):
            summarize([self.write(rows)])

    def test_other_file_cannot_supply_missing_post_check(self):
        valid = self.write(records())
        incomplete = [r for r in records() if not (r['kind'] == 'check' and r['variant'] == 'native' and r['phase'] == 'after')]
        second = self.write(incomplete, Path(self.directory.name) / 'incomplete.jsonl')
        with self.assertRaisesRegex(ValueError, 'Missing pre/post'):
            summarize([valid, second])

    def test_mixed_graph_event_samples_rejected(self):
        rows = records()
        next(r for r in rows if r['kind'] == 'timing')['timing'] = 'event'
        with self.assertRaisesRegex(ValueError, 'Mixed seed or timing'):
            summarize([self.write(rows)])

    def test_profiler_units_and_missing_counter(self):
        path = Path(self.directory.name) / 'profile.csv'
        path.write_text('"ID","Kernel Name","gpu__time_duration.sum"\n"","","us"\n"0","kernel","1250"\n')
        result = read(path)[0]
        self.assertEqual(result['profile_ms'], 1.25)
        self.assertIsNone(result['shared_ldsm_bank_conflicts'])


if __name__ == '__main__':
    unittest.main()
