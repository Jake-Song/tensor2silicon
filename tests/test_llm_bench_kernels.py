"""Opt-in numerical and cache-boundary tests for the native Triton kernels.

Run with RUN_LLM_BENCH_GPU_TESTS=1 on a CUDA runtime with PyTorch and Triton.
Collection and ordinary CPU test runs do not import either accelerator package.
"""

import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_LLM_BENCH_GPU_TESTS") == "1", "requires opt-in CUDA/Triton")
class NativeTritonKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch.nn.functional as functional
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from llm_bench.triton_attention import attention
        from llm_bench.triton_matmul import linear

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        cls.torch = torch
        cls.functional = functional
        cls.sdpa_kernel = staticmethod(sdpa_kernel)
        cls.math_backend = SDPBackend.MATH
        cls.attention = staticmethod(attention)
        cls.linear = staticmethod(linear)

    def setUp(self):
        self.torch.manual_seed(421)

    def tensor(self, shape):
        return self.torch.randn(shape, device="cuda", dtype=self.torch.bfloat16)

    def assert_relative_error(self, actual, expected, limit=0.006):
        torch = self.torch
        actual, expected = actual.float(), expected.float()
        self.assertTrue(bool(torch.isfinite(actual).all().item()))
        scale = expected.square().mean().sqrt().clamp_min(1e-6)
        nrmse = ((actual - expected).square().mean().sqrt() / scale).item()
        self.assertLessEqual(nrmse, limit, f"normalized RMS error {nrmse:.6g}")
        normalized_max = ((actual - expected).abs().max() / scale).item()
        self.assertLessEqual(normalized_max, 0.05, f"normalized maximum error {normalized_max:.6g}")

    def reference_attention(self, q, k, v, position=None):
        torch = self.torch
        allowed = None
        if position is not None:
            allowed = torch.arange(k.shape[2], device=k.device) <= position
        with self.sdpa_kernel(self.math_backend):
            out = self.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=allowed,
                is_causal=position is None, enable_gqa=q.shape[1] != k.shape[1],
            )
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def test_linear_ragged_dimensions_and_strided_weights(self):
        torch = self.torch
        # Exercise both small-M and large-M kernels, and masked K/N tails.
        for shape, n in [((2, 3, 57), 73), ((2, 17, 63), 127), ((1, 8, 257), 129)]:
            with self.subTest(shape=shape, n=n), torch.inference_mode():
                x = self.tensor(shape)
                w = self.tensor((n, shape[-1])).transpose(0, 1)
                actual = self.linear(x, w)
                expected = x.float() @ w.float()
                self.assertEqual(actual.shape, (*shape[:-1], n))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assert_relative_error(actual, expected)

    def test_causal_prefill_gqa_and_strided_inputs(self):
        torch = self.torch
        with torch.inference_mode():
            # Non-power-of-two sequence length exercises the final causal tile.
            q = self.tensor((2, 19, 4, 16)).transpose(1, 2)
            k = self.tensor((2, 19, 2, 16)).transpose(1, 2)
            v = self.tensor((2, 19, 2, 16)).transpose(1, 2)
            actual = self.attention(q, k, v)
            self.assertEqual(actual.shape, (2, 19, 64))
            self.assertTrue(actual.is_contiguous())
            self.assert_relative_error(actual, self.reference_attention(q, k, v))

    def test_decode_gqa_nondivisible_capacity_and_position_boundaries(self):
        torch = self.torch
        with torch.inference_mode():
            q = self.tensor((2, 4, 1, 16))
            # A view into a larger allocation checks cache batch/head strides.
            k = self.tensor((2, 2, 269, 16))[:, :, :257]
            v = self.tensor((2, 2, 269, 16))[:, :, :257]
            for slot in (0, 128, 256):
                position = torch.tensor(slot, device="cuda", dtype=torch.int64)
                reference = self.reference_attention(q, k, v, position)
                for split in (None, 1, 9, 32):
                    with self.subTest(position=slot, split_kv=split):
                        actual = self.attention(q, k, v, position, split_kv=split)
                        self.assert_relative_error(actual, reference)
                        if slot == 0:
                            # At 32 splits, 31 partitions are empty; the one
                            # nonempty partition must return exactly V[0].
                            expected = v[:, :, 0].repeat_interleave(2, dim=1).reshape(2, 1, 64)
                            self.assertTrue(torch.equal(actual, expected))

    def test_decode_future_slots_are_not_read(self):
        torch = self.torch
        with torch.inference_mode():
            q = self.tensor((1, 4, 1, 16))
            k, v = self.tensor((1, 2, 257, 16)), self.tensor((1, 2, 257, 16))
            position = torch.tensor(17, device="cuda", dtype=torch.int64)
            poisoned_k, poisoned_v = k.clone(), v.clone()
            poisoned_k[:, :, 18:] = float("nan")
            poisoned_v[:, :, 18:] = float("nan")
            reference = self.reference_attention(q, k, v, position)
            for split in (None, 1, 9, 32):
                with self.subTest(split_kv=split):
                    expected = self.attention(q, k, v, position, split_kv=split)
                    actual = self.attention(q, poisoned_k, poisoned_v, position, split_kv=split)
                    self.assert_relative_error(actual, reference)
                    self.assertTrue(torch.equal(actual, expected))

    def test_single_token_prefill(self):
        torch = self.torch
        with torch.inference_mode():
            q = self.tensor((2, 4, 1, 16))
            k, v = self.tensor((2, 2, 1, 16)), self.tensor((2, 2, 1, 16))
            expected = v.repeat_interleave(2, dim=1).transpose(1, 2).reshape(2, 1, 64)
            for split in (None, 1, 9, 32):
                with self.subTest(split_kv=split):
                    actual = self.attention(q, k, v, split_kv=split)
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
