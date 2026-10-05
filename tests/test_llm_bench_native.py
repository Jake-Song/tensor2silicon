"""Opt-in CUDA correctness checks for the complete native Triton decoder.

Run with ``RUN_LLM_BENCH_GPU_TESTS=1 python -m unittest discover -s tests
-p 'test_llm_bench_native.py'``. Discovery requires neither torch nor Triton.
The reference is the repository's original independent FP32 model, initialized
from the exact same BF16-rounded weights as the Triton runner.
"""

import importlib.util
import os
import unittest


RUN_GPU = os.environ.get("RUN_LLM_BENCH_GPU_TESTS") == "1"
HAVE_DEPS = all(importlib.util.find_spec(name) is not None for name in ("torch", "triton"))


@unittest.skipUnless(RUN_GPU and HAVE_DEPS, "set RUN_LLM_BENCH_GPU_TESTS=1 with torch and Triton installed")
class NativeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("a CUDA GPU with BF16 support is required")
        from llm_bench import triton_ops
        from llm_bench.native import NativeRunner
        from llm_roofline import spec, torch_llm

        cls.torch, cls.ops, cls.spec, cls.original = torch, triton_ops, spec, torch_llm
        cls.cfg = spec.PRESETS["tiny"].model
        cls.params = torch_llm.random_params(cls.cfg, torch.bfloat16, torch.device("cuda"), seed=41)
        cls.fp32_params = torch_llm.to_torch(cls.params, torch.float32, torch.device("cuda"))
        cls.runner = NativeRunner(cls.cfg, cls.params, capacity=24)

    def setUp(self):
        torch = self.torch
        generator = torch.Generator(device="cuda").manual_seed(17)
        # Noncontiguous tokens exercise storage offsets and both tensor strides.
        full = torch.randint(0, self.cfg.vocab, (2, 32), generator=generator, device="cuda")
        self.tokens = full[:, ::2]

    def reference(self, tokens):
        phase = self.spec.PhaseConfig("prefill", tokens.shape[0], tokens.shape[1], tokens.shape[1])
        consts = self.original.make_consts(self.cfg, phase, self.torch.float32, self.torch.device("cuda"))
        return self.original.forward(self.fp32_params, tokens, consts)

    def assert_bf16_close(self, actual, expected):
        torch = self.torch
        a, e = actual.float(), expected.float()
        self.assertTrue(torch.isfinite(a).all().item())
        error = a - e
        normalized_rmse = (error.square().mean() / e.square().mean().clamp_min(1e-12)).sqrt().item()
        self.assertLess(normalized_rmse, 0.03)
        self.assertLess(error.abs().max().item(), 0.15)

    def cache_with_prefix(self, prefix, fill=77.0):
        torch = self.torch
        shape = (2, self.cfg.n_kv_heads, self.runner.capacity, self.cfg.head_dim)
        cache = [(torch.full(shape, fill, dtype=torch.bfloat16, device="cuda"),
                  torch.full(shape, fill, dtype=torch.bfloat16, device="cuda"))
                 for _ in range(self.cfg.n_layers)]
        self.runner.copy_prefill_cache(prefix, cache)
        return cache

    def test_full_prefill_matches_original_fp32_with_strided_tokens(self):
        torch = self.torch
        with torch.inference_mode():
            actual, actual_kv = self.runner.prefill(self.tokens)
            expected, expected_kv = self.reference(self.tokens)
            self.assert_bf16_close(actual, expected)
            for (ak, av), (ek, ev) in zip(actual_kv, expected_kv):
                self.assert_bf16_close(ak, ek)
                self.assert_bf16_close(av, ev)
            embedded = self.ops.embedding(self.tokens, self.params["embed"])
            torch.testing.assert_close(embedded, self.params["embed"][self.tokens], rtol=0, atol=0)

    def test_decode_dynamic_position_preserves_prefix_and_future_sentinels(self):
        torch = self.torch
        with torch.inference_mode():
            _, prefix = self.runner.prefill(self.tokens[:, :12])
            cache = self.cache_with_prefix(prefix)
            before = [(k.clone(), v.clone()) for k, v in cache]
            position = torch.tensor(12, dtype=torch.int64, device="cuda")
            actual, actual_cache = self.runner.decode(self.tokens[:, 12:13], cache, position)
            expected, expected_kv = self.reference(self.tokens[:, :13])
            self.assert_bf16_close(actual[:, 0], expected[:, -1])
            for (ak, av), (bk, bv), (ek, ev) in zip(actual_cache, before, expected_kv):
                for after, prior, reference in ((ak, bk, ek), (av, bv, ev)):
                    torch.testing.assert_close(after[:, :, :12], prior[:, :, :12], rtol=0, atol=0)
                    torch.testing.assert_close(after[:, :, 13:], prior[:, :, 13:], rtol=0, atol=0)
                    self.assert_bf16_close(after[:, :, :13], reference)
            self.runner.advance(position)
            self.assertEqual(position.item(), 13)
            actual, _ = self.runner.decode(self.tokens[:, 13:14], cache, position)
            expected, _ = self.reference(self.tokens[:, :14])
            self.assert_bf16_close(actual[:, 0], expected[:, -1])

    def test_causal_prefix_matches_shorter_prefill(self):
        torch = self.torch
        with torch.inference_mode():
            full, _ = self.runner.prefill(self.tokens)
            short, _ = self.runner.prefill(self.tokens[:, :8])
            self.assert_bf16_close(full[:, :8], short)

    def test_argmax_ties_last_position_and_strided_logits(self):
        torch = self.torch
        with torch.inference_mode():
            backing = torch.full((2, 4, 256), -5, dtype=torch.bfloat16, device="cuda")
            logits = backing[:, ::2, ::2]
            logits[:, 0, 0] = 100  # The earlier timestep must not influence selection.
            logits[0, -1, 2] = logits[0, -1, 7] = 10
            logits[1, -1, 3] = logits[1, -1, 8] = 2
            expected = torch.tensor([[2], [3]], dtype=torch.int64, device="cuda")
            actual = self.runner.select_token(logits)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            target = torch.empty_like(actual)
            self.ops.copy_token(actual, target)
            torch.testing.assert_close(target, expected, rtol=0, atol=0)

    def test_non_power_of_two_layernorm_and_bf16_residual_boundary(self):
        torch = self.torch
        import torch.nn.functional as F

        with torch.inference_mode():
            x = torch.randn((2, 3, 70), device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            weight = torch.randn((70,), device="cuda", dtype=torch.bfloat16)
            bias = torch.randn_like(weight)
            summed, actual = self.ops.add_layernorm(x, residual, weight, bias)
            expected_sum = x + residual
            torch.testing.assert_close(summed, expected_sum, rtol=0, atol=0)
            expected = F.layer_norm(expected_sum, (70,), weight, bias, eps=1e-5)
            torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)

    def test_cuda_graph_observes_changing_position_tokens_and_cache(self):
        torch = self.torch
        with torch.inference_mode():
            _, prefix = self.runner.prefill(self.tokens[:, :4])
            cache = self.cache_with_prefix(prefix)
            token = self.tokens[:, 4:5].contiguous().clone()
            position = torch.tensor(4, dtype=torch.int64, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.runner.decode(token, cache, position)
                self.runner.advance(position)
            torch.cuda.current_stream().wait_stream(stream)
            position.fill_(4)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits, _ = self.runner.decode(token, cache, position)
                self.runner.advance(position)
            position.fill_(4)
            graph.replay()
            first = logits.clone()
            expected, _ = self.reference(self.tokens[:, :5])
            self.assert_bf16_close(first[:, 0], expected[:, -1])
            self.assertEqual(position.item(), 5)
            token.copy_(self.tokens[:, 5:6])
            graph.replay()
            expected, _ = self.reference(self.tokens[:, :6])
            self.assert_bf16_close(logits[:, 0], expected[:, -1])
            self.assertEqual(position.item(), 6)
            # Reusing the same graph must read the current cache, not a saved result.
            cache[0][1][:, :, 0].add_(20)
            token.copy_(self.tokens[:, 4:5])
            position.fill_(4)
            graph.replay()
            self.assertFalse(torch.allclose(logits, first, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
