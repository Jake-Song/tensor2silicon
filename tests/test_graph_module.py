"""Run with: python -m unittest discover -s tests -v."""

import contextlib
import io
import shutil
import unittest

import numpy as np

import toy


def linear(x, w, b):
    return toy.relu(x @ w + b)


class GraphModuleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.standard_normal((3, 4), dtype=np.float32)
        self.w = rng.standard_normal((4, 2), dtype=np.float32)
        self.b = rng.standard_normal((2,), dtype=np.float32)
        self.args = (self.x, self.w, self.b)

    def test_linear_from_shapes_and_examples(self):
        for specs in (self.args, tuple(a.shape for a in self.args)):
            with self.subTest(specs=tuple(type(s).__name__ for s in specs)):
                gm = toy.symbolic_trace(linear, *specs)
                self.assertIsInstance(gm, toy.GraphModule)
                np.testing.assert_allclose(gm(*self.args), linear(*self.args))
                np.testing.assert_allclose(gm(*self.args), toy.run(gm.graph, *self.args)[0])
                self.assertEqual([n.op for n in gm.graph.nodes],
                                 ["matmul", "broadcast_in_dim", "add", "relu"])
                self.assertEqual(gm.graph.outputs[0].shape, (3, 2))

    def test_source_is_standalone_executable_and_deterministic(self):
        gm = toy.symbolic_trace(linear, *self.args)
        namespace = {}
        exec(compile(gm.code, "<test_generated>", "exec"), namespace)
        np.testing.assert_array_equal(namespace["forward"](*self.args), gm(*self.args))
        self.assertIn(" @ ", gm.code)
        self.assertIn("np.maximum", gm.code)
        self.assertEqual(gm.code, toy.symbolic_trace(linear, *self.args).code)

    def test_shared_intermediate_and_repeated_outputs(self):
        def f(x):
            h = toy.relu(x)
            return h + h, h, h

        gm = toy.symbolic_trace(f, self.x)
        self.assertEqual([n.op for n in gm.graph.nodes], ["relu", "add"])
        actual = gm(self.x)
        self.assertIsInstance(actual, tuple)
        for result, expected in zip(actual, f(self.x)):
            np.testing.assert_array_equal(result, expected)
        self.assertIs(actual[1], actual[2])

    def test_return_structures(self):
        for f in (lambda x: x, lambda x: (x,), lambda x: (x, x), lambda x: ()):
            with self.subTest(function=f):
                gm = toy.symbolic_trace(f, self.x)
                actual, expected = gm(self.x), f(self.x)
                self.assertEqual(type(actual), type(expected))
                if isinstance(expected, tuple):
                    self.assertEqual(len(actual), len(expected))
                    for a, e in zip(actual, expected):
                        np.testing.assert_array_equal(a, e)
                else:
                    np.testing.assert_array_equal(actual, expected)
        self.assertEqual(toy.symbolic_trace(lambda: ())(), ())

    def test_original_function_runs_only_during_capture(self):
        calls = []

        def f(x):
            calls.append(True)
            return toy.relu(x)

        gm = toy.symbolic_trace(f, self.x)
        self.assertEqual(len(calls), 1)
        gm(self.x)
        np.testing.assert_array_equal(gm(-self.x), np.maximum(-self.x, 0))
        self.assertEqual(len(calls), 1)

    def test_broadcast_rank_promotion_on_either_side(self):
        for f in (lambda x, b: x + b, lambda x, b: b + x):
            x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            b = np.arange(4, dtype=np.float32)
            gm = toy.symbolic_trace(f, x, b)
            np.testing.assert_array_equal(gm(x, b), f(x, b))
        scalar = np.array(2, dtype=np.float32)
        gm = toy.symbolic_trace(lambda x, b: x + b, self.x, scalar)
        np.testing.assert_array_equal(gm(self.x, scalar), self.x + scalar)

    def test_explicit_broadcast_dimensions(self):
        graph = toy.Graph()
        x = graph.input("x", (3,))
        graph.output(graph.broadcast_in_dim(x, (2, 3, 4), (1,)))
        gm = toy.GraphModule(graph)
        x = np.arange(3, dtype=np.float32)
        np.testing.assert_array_equal(gm(x), np.broadcast_to(x[None, :, None], (2, 3, 4)))

    def test_input_conversion_and_validation(self):
        gm = toy.symbolic_trace(lambda x: toy.relu(x), (2,))
        result = gm([-1, 2])
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, [0, 2])
        with self.assertRaisesRegex(ValueError, "expected shape"):
            gm(np.zeros((3,), dtype=np.float32))
        for args in ((), ([1, 2], [3, 4])):
            with self.assertRaises(TypeError):
                gm(*args)

    def test_input_names_cannot_collide_with_generated_names(self):
        def f(np, forward, v0, arg0):
            return toy.relu(np + forward + v0 + arg0)

        gm = toy.symbolic_trace(f, *((2,),) * 4)
        np.testing.assert_array_equal(gm(*([1, 2],) * 4), [4, 8])
        graph = toy.Graph()
        x = graph.input("not-a-python-identifier'\n", (2,))
        graph.output(x)
        np.testing.assert_array_equal(toy.GraphModule(graph)([1, 2]), [1, 2])

    def test_table_includes_dependencies_and_metadata(self):
        gm = toy.symbolic_trace(linear, *self.args)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = gm.graph.print_tabular()
        self.assertIsNone(result)
        table = stream.getvalue()
        for text in ("op", "name", "inputs", "shape", "dtype", "attrs", "input",
                     "matmul", "broadcast_in_dim", "relu", "output", "('x', 'w')",
                     "('%0', '%1')", "(3, 2)", "f32", "'dims': (1,)"):
            self.assertIn(text, table)

    def test_capture_failures(self):
        with self.assertRaisesRegex(TypeError, "specs"):
            toy.symbolic_trace(linear, self.x)
        with self.assertRaises(toy.ShapeError):
            toy.symbolic_trace(linear, (3, 4), (5, 2), (2,))
        with self.assertRaisesRegex(TypeError, "constants"):
            toy.symbolic_trace(lambda x: x + 1, self.x)
        with self.assertRaises(TypeError):
            toy.symbolic_trace(lambda x: x * x, self.x)
        with self.assertRaisesRegex(toy.TraceError, "cannot branch"):
            toy.symbolic_trace(lambda x: x if x else toy.relu(x), self.x)
        with self.assertRaisesRegex(toy.TraceError, "cannot iterate"):
            toy.symbolic_trace(lambda x: tuple(x), self.x)
        for f in (lambda x: 1, lambda x: [x], lambda x: ((x,),)):
            with self.assertRaisesRegex(toy.TraceError, "non-Tensor"):
                toy.symbolic_trace(f, self.x)

    def test_foreign_trace_outputs_are_rejected(self):
        graph = toy.Graph()
        foreign = toy.Tensor(graph, graph.input("foreign", self.x.shape))
        for capture in (toy.trace, toy.symbolic_trace):
            with self.assertRaisesRegex(toy.TraceError, "different trace"):
                capture(lambda x: foreign, self.x)
            with self.assertRaisesRegex(toy.TraceError, "different traces"):
                capture(lambda x: x + foreign, self.x)

    def test_codegen_rejects_unsupported_graphs(self):
        gm = toy.symbolic_trace(linear, *self.args)
        with self.assertRaisesRegex(NotImplementedError, "fusion"):
            toy.GraphModule(toy.fuse(gm.graph))
        graph = toy.Graph()
        graph.output(graph.input("x", (2,), "f64"))
        with self.assertRaisesRegex(TypeError, "f32"):
            toy.GraphModule(graph)

    @unittest.skipUnless(shutil.which("gcc"), "C compiler not available")
    def test_existing_trace_passes_and_c_compilation(self):
        gm = toy.symbolic_trace(linear, *self.args)
        original_ir = str(gm.graph)
        self.assertEqual(original_ir, str(toy.trace(linear, *self.args)))
        for graph in (gm.graph, toy.fuse(gm.graph), toy.tile_matmuls(toy.fuse(gm.graph), (2, 2, 2))):
            actual = toy.compile_graph(graph)(*self.args)[0]
            np.testing.assert_allclose(actual, gm(*self.args), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(toy.jit(linear)(*self.args), gm(*self.args), rtol=1e-5, atol=1e-6)
        self.assertEqual(str(gm.graph), original_ir)


if __name__ == "__main__":
    unittest.main()
