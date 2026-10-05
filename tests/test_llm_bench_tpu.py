"""Correctness and audit contracts; device tests require a real TPU explicitly."""
import os
import unittest

import numpy as np

from llm_bench_tpu.metrics import errors, summarize
from llm_bench_tpu.runner import audit_hlo
from llm_bench_tpu.config import gemm_tile
from llm_bench_tpu.profile_summary import exclusive_times


class ValidationTests(unittest.TestCase):
    def test_profile_aggregation_does_not_double_count_nested_or_overlapped_ops(self):
        events = [{"name": "native_gemm", "ts": 0, "dur": 10},
                  {"name": "copy", "ts": 2, "dur": 6, "args": {"hlo_category": "copy"}},
                  {"name": "copy", "ts": 9, "dur": 3, "args": {"hlo_category": "copy"}}]
        self.assertEqual(exclusive_times(events), {"GEMM": 10, "XLA/copy": 2})

    def test_tiles_clamp_small_shapes_and_reject_empty_or_oversized_grids(self):
        self.assertEqual(gemm_tile(1, 256, 128, {"default": [512, 1024, 2048]}), (1, 256, 128))
        self.assertEqual(gemm_tile(8, 5632, 2048), (8, 512, 512))
        for tile in ([8, 0, 128], [8, 1024, 2048], [8, 5632, 2048]):
            with self.assertRaises(ValueError):
                gemm_tile(8, 5632, 2048, {"default": tile})

    def test_error_gates_do_not_hide_outliers_or_nonfinite_outputs(self):
        expected = np.ones(10000, dtype=np.float32)
        actual = expected.copy()
        actual[0] = 1.25
        self.assertLess(errors(actual, expected)["nrmse"], .02)
        self.assertFalse(errors(actual, expected)["passed"])
        for value in (np.nan, np.inf):
            actual[0] = value
            with self.assertRaises(AssertionError):
                errors(actual, expected)
        with self.assertRaises(AssertionError):
            errors(expected[:2], expected)

    def test_latency_samples_preserve_order_and_reject_invalid_measurements(self):
        summary = summarize([4, 1, 3, 2])
        self.assertEqual(summary["samples_ms"], [4, 1, 3, 2])
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)
        for bad in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                summarize(bad)

    def test_native_audit_rejects_unwrapped_compute_and_unknown_calls(self):
        pallas = "stablehlo.custom_call @tpu_custom_call(%0)"
        self.assertTrue(audit_hlo(pallas + " stablehlo.reshape", "native", 1)["passed"])
        for op in ("dot_general", "gather", "reduce", "exponential", "add", "transpose"):
            with self.assertRaises(AssertionError):
                audit_hlo(pallas + " stablehlo." + op, "native", 1)
        with self.assertRaises(AssertionError):
            audit_hlo(pallas + " stablehlo.custom_call @vendor_gemm(%0)", "native", 1)
        with self.assertRaises(AssertionError):
            audit_hlo("stablehlo.reshape", "native", 1)
        with self.assertRaises(AssertionError):
            audit_hlo(pallas, "jax", 1)
        with self.assertRaises(AssertionError):
            audit_hlo(pallas, "hybrid", 2)


@unittest.skipUnless(os.environ.get("RUN_TPU_PALLAS_TESTS") == "1", "set RUN_TPU_PALLAS_TESTS=1 on TPU")
class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        if jax.devices()[0].platform != "tpu":
            raise AssertionError("hardware tests require a TPU")

    def test_causal_attention_cannot_read_future_tokens(self):
        import jax
        import jax.numpy as jnp
        from functools import partial
        from llm_bench_tpu.kernels import attention
        keys = jax.random.split(jax.random.key(19), 3)
        q = jax.random.normal(keys[0], (1, 2, 256, 128), jnp.bfloat16) * .1
        k = jax.random.normal(keys[1], (1, 1, 256, 128), jnp.bfloat16)
        v = jax.random.normal(keys[2], (1, 1, 256, 128), jnp.bfloat16)
        for block_q in (128, 256, 512):
            fn = jax.jit(partial(attention, block_q=block_q))
            a = fn(q, k, v)
            b = fn(q, k, v.at[:, :, 128:, :].set(100))
            np.testing.assert_array_equal(np.asarray(a[:, :, :128, :]), np.asarray(b[:, :, :128, :]))
            self.assertFalse(np.array_equal(np.asarray(a[:, :, 128:, :]), np.asarray(b[:, :, 128:, :])))

    def test_grouped_decode_attention_includes_all_heads_and_supports_gqa(self):
        import jax
        import jax.numpy as jnp
        from functools import partial
        from llm_bench_tpu.kernels import attention
        q = jax.random.normal(jax.random.key(4), (2, 4, 1, 128), jnp.bfloat16) * .1
        for kv_heads in (2, 4):
            k = jax.random.normal(jax.random.key(5), (2, kv_heads, 256, 128), jnp.bfloat16)
            v = jax.random.normal(jax.random.key(6), k.shape, jnp.bfloat16)
            expected = jax.jit(attention)(q, k, v)
            got = jax.jit(partial(attention, decode_heads=4))(q, k, v)
            self.assertTrue(errors(np.asarray(got, np.float32), np.asarray(expected, np.float32))["passed"])

    def test_native_cache_update_and_embedding(self):
        import jax
        import jax.numpy as jnp
        from llm_bench_tpu import native
        old = jax.random.normal(jax.random.key(7), (2, 2, 512, 128), jnp.bfloat16)
        new = jnp.full((2, 2, 1, 128), 7, jnp.bfloat16)
        got = jax.jit(native.kv_write)(old, new)
        np.testing.assert_array_equal(np.asarray(got[:, :, :-1, :]), np.asarray(old[:, :, :-1, :]))
        np.testing.assert_array_equal(np.asarray(got[:, :, -1:, :]), np.asarray(new))
        table = jax.random.normal(jax.random.key(8), (256, 256), jnp.bfloat16)
        tokens = jnp.asarray([[0, 255, 3, 3], [200, 17, 1, 0]], jnp.int32)
        embedded = jax.jit(native.embed)(tokens, table)
        np.testing.assert_array_equal(np.asarray(embedded), np.asarray(table)[np.asarray(tokens)])

    def test_grouped_kv_pair_preserves_inputs_and_repeated_calls(self):
        import jax
        import jax.numpy as jnp
        from functools import partial
        from llm_bench_tpu.native import kv_write_grouped
        old = jax.random.normal(jax.random.key(3), (2, 4, 512, 128), jnp.bfloat16)
        saved = np.asarray(old).copy()
        new = jax.random.normal(jax.random.key(10), (2, 4, 1, 128), jnp.bfloat16)
        fn = jax.jit(partial(kv_write_grouped, block_s=2048, group_heads=4))
        reread = jax.jit(lambda a: a.astype(jnp.float32))
        for _ in range(2):
            k, v = fn((old, old), (new, -new))
            np.testing.assert_array_equal(np.asarray(k[:, :, :-1]), saved[:, :, :-1])
            np.testing.assert_array_equal(np.asarray(v[:, :, :-1]), saved[:, :, :-1])
            np.testing.assert_array_equal(np.asarray(k[:, :, -1:]), np.asarray(new))
            np.testing.assert_array_equal(np.asarray(v[:, :, -1:]), -np.asarray(new))
            np.testing.assert_array_equal(np.asarray(reread(old)), saved.astype(np.float32))

    def test_flat_cache_tail_updates_distinct_heads_and_preserves_input(self):
        import jax
        import jax.numpy as jnp
        from functools import partial
        from llm_bench_tpu.native import kv_write_flat
        old = jax.random.normal(jax.random.key(41), (2, 4, 512, 128), jnp.bfloat16)
        new = jax.random.normal(jax.random.key(42), (2, 4, 1, 128), jnp.bfloat16)
        saved = np.asarray(old).copy()
        fn = jax.jit(partial(kv_write_flat, group_heads=4))
        reread = jax.jit(lambda a: a.astype(jnp.float32))
        for _ in range(2):
            got = fn(old, new)
            np.testing.assert_array_equal(np.asarray(got[:, :, :-1]), saved[:, :, :-1])
            np.testing.assert_array_equal(np.asarray(got[:, :, -1:]), np.asarray(new))
            np.testing.assert_array_equal(np.asarray(reread(old)), saved.astype(np.float32))

    def test_gemm_k_reduction_and_vocab_tiles(self):
        import jax
        import jax.numpy as jnp
        from llm_bench_tpu.native import linear
        for m, k, n in ((1, 2048, 256), (8, 5632, 256), (512, 1024, 768)):
            a = jax.random.normal(jax.random.key(1), (m, k), jnp.bfloat16)
            b = jax.random.normal(jax.random.key(2), (k, n), jnp.bfloat16) * .02
            got = jax.jit(linear)(a, b)
            with jax.default_matmul_precision("highest"):
                ref = a.astype(jnp.float32) @ b.astype(jnp.float32)
            self.assertTrue(errors(np.asarray(got, np.float32), np.asarray(ref))["passed"])

    def test_projection_fusion_keeps_bf16_rounding_and_head_order(self):
        import jax
        import jax.numpy as jnp
        from functools import partial
        from llm_bench_tpu.native import linear, linear_heads, split_heads
        for b, t in ((8, 1), (2, 128)):
            x = jax.random.normal(jax.random.key(12), (b, t, 256), jnp.bfloat16)
            w = jax.random.normal(jax.random.key(13), (256, 256), jnp.bfloat16) * .02
            angles = jax.random.normal(jax.random.key(14), (t, 64), jnp.bfloat16)
            for scale in (None, 1.0, 128**-.5):
                options = {} if scale is None else {"cos": jnp.cos(angles), "sin": jnp.sin(angles), "scale": scale}
                ref = jax.jit(lambda a, w: split_heads(linear(a, w), 128, **options))(x, w)
                got = jax.jit(partial(linear_heads, head_dim=128, **options))(x, w)
                self.assertEqual(got.shape, (b, 2, t, 128))
                self.assertTrue(errors(np.asarray(got, np.float32), np.asarray(ref, np.float32))["passed"])


if __name__ == "__main__":
    unittest.main()
