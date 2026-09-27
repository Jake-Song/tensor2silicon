"""CPU tests for the fixed evaluator, plus opt-in real accelerator checks.

RUN_LLM_GPU_TESTS=1 or RUN_LLM_TPU_TESTS=1 enables tiny baseline checks.
Full-size hardware acceptance additionally runs bench.py for both phases.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from autoresearch import llm_common as bench
from llm_roofline import spec

ROOT = Path(__file__).resolve().parents[1]


class ScoringTests(unittest.TestCase):
    def test_confirmation_and_strict_threshold(self):
        self.assertTrue(bench.improvement_confirmed(10, 9.8, 9.85))
        for first, second in ((9.9, 9.8), (9.8, 9.9), (9.8, 10), (0, 9), (9, float('nan'))):
            self.assertFalse(bench.improvement_confirmed(10, first, second))


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.expected = (np.ones((2, 1, 4), np.float32),
                         [(np.ones((2, 2, 3, 4), np.float32), np.ones((2, 2, 3, 4), np.float32))])

    def check(self, value):
        return bench.validate_output(value, self.expected, np.asarray, np.dtype('float32'))

    def test_all_outputs_required(self):
        self.assertEqual(self.check(self.expected), 0)
        for output in (self.expected[0], (self.expected[0], []), (self.expected[0], [(self.expected[1][0][0],)])):
            with self.assertRaises(AssertionError):
                self.check(output)

    def test_wrong_values_shapes_dtype_and_nonfinite(self):
        for logits in (np.zeros((2, 1, 4), np.float32), np.ones((2, 1, 3), np.float32),
                       np.ones((2, 1, 4), np.float64), np.full((2, 1, 4), np.nan, np.float32)):
            with self.assertRaises(AssertionError):
                self.check((logits, self.expected[1]))

    def test_wrong_cache_is_rejected_even_when_logits_match(self):
        wrong = (self.expected[0], [(np.zeros((2, 2, 3, 4), np.float32), self.expected[1][0][1])])
        with self.assertRaises(AssertionError):
            self.check(wrong)

    def test_prefix_requires_exact_preservation(self):
        class Backend:
            dtype = np.dtype('float32')
            numpy = staticmethod(np.asarray)
            sync = staticmethod(lambda x: x)
            clone_cache = staticmethod(lambda c: [(k.copy(), v.copy()) for k, v in c])
        def reference(p, t, c, cache):
            return np.ones((2, 1, 4), np.float32), cache
        def wrong(p, t, c, cache):
            cache[0][0][:, :, 0] += 0.001  # Inside allclose tolerance but prefix writes are forbidden.
            return reference(p, t, c, cache)
        with self.assertRaisesRegex(AssertionError, 'modified prefix'):
            bench.check_case(Backend(), wrong, reference, (None, None, None, self.expected[1]))


class TimingTests(unittest.TestCase):
    def test_complete_outputs_synchronized_and_returned_cache_reused(self):
        synchronized = []
        class Backend:
            def sync(self, value):
                synchronized.append(value)
                return value
        seen = []
        def forward(p, tokens, consts, cache):
            seen.append(cache)
            return 'logits', cache + 1
        with patch.object(bench, 'WARMUP', 2), patch.object(bench, 'ROUNDS', 3), patch.object(bench, 'CALLS', 4):
            result = bench.time_forward(Backend(), forward, ('params', 'tokens', 'consts', 0))
        self.assertEqual(seen, list(range(15)))
        self.assertEqual(result[3][-1], 15)
        self.assertEqual(result[4], ('logits', 15))
        self.assertEqual(len(result[1]), 3)
        self.assertTrue(all(out[0] == 'logits' for out in synchronized[1:]))
        self.assertEqual(synchronized[0], ('params', 'tokens', 'consts', 0))


class ArtifactTests(unittest.TestCase):
    def test_models_preserve_forward_interface(self):
        for kind in ('gpu', 'tpu'):
            source = ROOT / f'autoresearch/llm_{kind}/model.py'
            tree = ast.parse(source.read_text())
            forward = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            self.assertEqual([arg.arg for arg in forward.args.args], ['params', 'tokens', 'consts', 'cache'])

    def test_launchers_parse_and_have_no_saved_outputs(self):
        for kind, target in [('gpu', 'a100'), ('tpu', 'v5e')]:
            n = json.loads((ROOT / f'notebooks/autoresearch_llm_{kind}_{target}.ipynb').read_text())
            ids = []
            for cell in n['cells']:
                ids.append(cell['id'])
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
                    self.assertEqual(cell['outputs'], [])
            self.assertEqual(len(ids), len(set(ids)))

    def test_help_without_accelerator_imports(self):
        for kind in ('gpu', 'tpu'):
            p = subprocess.run([sys.executable, 'bench.py', '--help'], cwd=ROOT / f'autoresearch/llm_{kind}',
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn('--phase', p.stdout)

    def test_backend_failure_has_machine_readable_status(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            output = Path(td) / 'result.json'
            with patch.object(bench, 'JaxBackend', side_effect=RuntimeError('runtime unavailable')):
                with patch.object(sys, 'argv', ['bench.py', '--phase', 'decode', '--json', str(output)]):
                    with patch('traceback.print_exc'), patch('builtins.print'):
                        status = bench.main('tpu', Path('unused.py'))
            self.assertEqual(status, 1)
            result = json.loads(output.read_text())
            self.assertEqual(result['status'], 'error')
            self.assertFalse(result['scored'])
            self.assertIn('runtime unavailable', result['error'])


class HardwareTests(unittest.TestCase):
    def run_track(self, kind):
        for phase in ('prefill', 'decode'):
            p = subprocess.run([sys.executable, 'bench.py', '--phase', phase, '--smoke'],
                               cwd=ROOT / f'autoresearch/llm_{kind}', capture_output=True, text=True, timeout=900)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            result = json.loads(next(line.removeprefix('result_json: ') for line in p.stdout.splitlines()
                                     if line.startswith('result_json: ')))
            self.assertTrue(result['correct'])
            self.assertFalse(result['scored'])

    @unittest.skipUnless(os.environ.get('RUN_LLM_TPU_TESTS') == '1', 'requires opt-in JAX runtime')
    def test_jax_rejects_mask_and_rope_errors(self):
        backend = bench.JaxBackend(smoke=True)
        preset = spec.PRESETS['tiny']
        phase = preset.prefill
        reference = backend.prepare(backend.ref.forward, phase)
        for defect in ('mask', 'rope'):
            def broken(p, t, c, cache, defect=defect):
                changed = dict(c)
                if defect == 'mask':
                    changed['mask'] = backend.np.zeros_like(c['mask'])
                else:
                    changed['cos'] = backend.np.zeros_like(c['cos'])
                    changed['sin'] = backend.np.zeros_like(c['sin'])
                return backend.ref.forward(p, t, changed, cache)
            with self.subTest(defect=defect), self.assertRaises(AssertionError):
                bench.check_case(backend, backend.prepare(broken, phase), reference,
                                 backend.case(preset.model, phase, 53))

    @unittest.skipUnless(os.environ.get('RUN_LLM_GPU_TESTS') == '1', 'requires opt-in CUDA/Triton')
    def test_gpu_baseline(self):
        self.run_track('gpu')

    @unittest.skipUnless(os.environ.get('RUN_LLM_TPU_TESTS') == '1', 'requires opt-in JAX runtime')
    def test_tpu_baseline(self):
        self.run_track('tpu')


if __name__ == '__main__':
    unittest.main()
