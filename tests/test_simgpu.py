import copy
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

import toy


def linear(x, w, b):
    return toy.relu(x @ w + b)


@unittest.skipUnless(importlib.util.find_spec("simgpu"), "optional simgpu checkout not installed")
class SimGPUBackendTests(unittest.TestCase):
    def setUp(self):
        from simgpu import GPUConfig
        self.config = GPUConfig(block_size=7, warp_size=4, num_sms=2)
        self.rng = np.random.default_rng(17)

    def array(self, shape):
        return self.rng.standard_normal(shape, dtype=np.float32)

    def check(self, fn, args, expected=None):
        graph = toy.trace(fn, *args)
        original = str(graph)
        expected = fn(*args) if expected is None else expected
        expected = expected if isinstance(expected, tuple) else (expected,)
        executions = []
        for fuse in (None, False, True):
            lowered = graph if fuse is None else toy.fuse(graph, fuse_matmul=fuse)
            compiled = toy.compile_simgpu(lowered, config=self.config)
            execution = compiled.run(*args, record=True)
            self.assertEqual(len(execution.outputs), len(expected))
            for actual, reference in zip(execution.outputs, expected):
                np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4, equal_nan=True)
                self.assertEqual(actual.dtype, np.float32)
            for kernel in compiled.kernels:
                self.assertEqual(len(kernel.source_map), len(kernel.program))
                self.assertTrue(all(n.startswith("n") for n in kernel.source_map))
            executions.append(execution)
        self.assertEqual(str(graph), original)
        self.assertEqual(self.config.num_blocks, 1)
        return executions

    def test_examples_fusion_and_partial_warps(self):
        x, w, b = self.array((3, 5)), self.array((5, 5)), self.array((5,))
        self.check(lambda x, w: x @ w, (x, w))
        runs = self.check(linear, (x, w, b))
        self.assertEqual([len(r.kernels) for r in runs], [4, 2, 1])
        self.assertLess(runs[-1].stats.global_loads, runs[0].stats.global_loads)
        self.assertLess(runs[-1].stats.global_stores, runs[0].stats.global_stores)
        self.assertTrue(any(k.stats.divergent_branches for k in runs[-1].kernels))

    def test_shared_intermediates_and_multiple_outputs(self):
        def fn(x, w, b):
            h = x @ w
            y = toy.relu(h + b) + h
            return y, h, y, x
        self.check(fn, (self.array((2, 3)), self.array((3, 5)), self.array((5,))))

    def test_elementwise_broadcast_and_scalars(self):
        self.check(lambda x, b: toy.relu(b + x), (self.array((2, 3, 5)), self.array((5,))))
        self.check(lambda x, b: x + b, (self.array((3,)), np.array(1.25, dtype=np.float32)))
        self.check(lambda x: toy.relu(x), (np.array(-2.5, dtype=np.float32),))
        graph = toy.Graph()
        x = graph.input("x", (2, 3))
        graph.output(graph.broadcast_in_dim(x, (2, 4, 3), (0, 2)))
        data = self.array((2, 3))
        np.testing.assert_array_equal(toy.compile_simgpu(graph)(data)[0], np.broadcast_to(data[:, None, :], (2, 4, 3)))

    def test_repeated_fused_operand_and_two_reductions(self):
        def fn(x, w):
            a = x @ w
            b = x @ w
            return toy.relu(a + a + b)
        self.check(fn, (self.array((2, 3)), self.array((3, 4))))

    def test_empty_dimensions_and_identity(self):
        self.check(lambda x: toy.relu(x), (np.empty((0, 3), dtype=np.float32),))
        self.check(lambda x, w: x @ w, (np.empty((2, 0), dtype=np.float32), np.empty((0, 3), dtype=np.float32)))
        self.check(lambda x: (x, x), (self.array((3,)),))
        self.check(lambda x: (), (self.array((3,)),))

    def test_nonfinite_and_negative_zero(self):
        x = np.array([-np.inf, np.inf, np.nan, -0.0, -1.0], dtype=np.float32)
        self.check(lambda x: toy.relu(x), (x,))
        y = toy.compile_simgpu(toy.trace(lambda x: toy.relu(x), x))(x)[0]
        self.assertFalse(np.signbit(y[3]))

    def test_repeated_calls_and_shape_validation(self):
        fn = toy.jit(linear, backend="simgpu", sim_config=self.config)
        args = (self.array((3, 4)), self.array((4, 2)), self.array((2,)))
        compiled = fn.compile(*args)
        self.assertIs(fn.compile(*args), compiled)
        changed = tuple(a * 2 for a in args)
        np.testing.assert_allclose(fn(*changed), linear(*changed), rtol=1e-4, atol=1e-4)
        self.assertEqual(len(fn.cache), 1)
        fn(self.array((2, 4)), args[1], args[2])
        self.assertEqual(len(fn.cache), 2)
        with self.assertRaisesRegex(ValueError, "inputs"):
            compiled(args[0])
        with self.assertRaisesRegex(ValueError, "shape"):
            compiled(args[0][:2], *args[1:])
        # Float64 and strided input arrays are converted to contiguous float32.
        converted = (args[0].astype(np.float64)[:, ::-1], args[1], args[2])
        np.testing.assert_allclose(compiled(*converted)[0], linear(*converted), rtol=1e-4, atol=1e-4)

    def test_compile_snapshot_and_deterministic_source(self):
        graph = toy.trace(lambda x: toy.relu(x), (5,))
        first = toy.compile_simgpu(graph)
        self.assertEqual(first.source, toy.compile_simgpu(graph).source)
        graph.nodes[0].op = "unsupported"
        np.testing.assert_array_equal(first(np.arange(-2, 3))[0], [0, 0, 0, 1, 2])

    def test_errors_and_resource_limits(self):
        from simgpu import GPUConfig, SimError
        graph = toy.trace(linear, (3, 4), (4, 2), (2,))
        tiled = toy.tile_matmuls(copy.deepcopy(graph), (2, 2, 2))
        with self.assertRaisesRegex(toy.SimGPUCompileError, "tile"):
            toy.compile_simgpu(tiled)
        bad = toy.trace(lambda x: toy.relu(x), (2,))
        bad.nodes[0].op = "unknown"
        with self.assertRaisesRegex(toy.SimGPUCompileError, "unsupported"):
            toy.compile_simgpu(bad)
        bad.nodes[0].op = "relu"
        bad.inputs[0].dtype = "f64"
        with self.assertRaisesRegex(toy.SimGPUCompileError, "f32"):
            toy.compile_simgpu(bad)
        with self.assertRaisesRegex(toy.SimGPUCompileError, "register exhaustion"):
            toy.compile_simgpu(toy.trace(lambda x: toy.relu(x), (1,) * 16))
        args = (self.array((3, 4)), self.array((4, 2)), self.array((2,)))
        with self.assertRaisesRegex(SimError, "max_cycles"):
            toy.compile_simgpu(graph, config=GPUConfig(max_cycles=1))(*args)
        with self.assertRaisesRegex(SimError, "samples"):
            toy.compile_simgpu(graph).run(*args, record=True, max_samples=1)


class BackendSelectionTests(unittest.TestCase):
    def test_invalid_backend_and_cpu_options(self):
        for opts in ({"backend": "invalid"}, {"backend": "simgpu", "tile": "auto"},
                     {"backend": "simgpu", "tile": (2, 2, 2)}, {"sim_config": object()}):
            with self.assertRaises(ValueError):
                toy.jit(lambda x: x, **opts)

    def test_optional_dependency_error(self):
        with patch.dict("sys.modules", {"simgpu": None}):
            with self.assertRaisesRegex(ImportError, "with-editable"):
                toy.compile_simgpu(toy.trace(lambda x: x, (3,)))


if __name__ == "__main__":
    unittest.main()
