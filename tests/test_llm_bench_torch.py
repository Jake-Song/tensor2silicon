"""Small, independent correctness checks for the optimized model graph."""

import importlib.util
import unittest


HAVE_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAVE_TORCH, "torch is required for model correctness tests")
class TorchRunnerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from llm_bench.torch_backend import TorchRunner
        from llm_roofline import spec, torch_llm

        self.torch, self.spec, self.original = torch, spec, torch_llm
        self.runner_cls = TorchRunner
        self.cfg = spec.PRESETS["tiny"].model
        self.params = torch_llm.random_params(self.cfg, torch.float32, torch.device("cpu"), seed=41)
        self.runner = TorchRunner(self.cfg, self.params, capacity=24, compile_model=False)
        self.tokens = torch.randint(0, self.cfg.vocab, (2, 16), generator=torch.Generator().manual_seed(17))

    def reference(self, tokens):
        phase = self.spec.PhaseConfig("prefill", tokens.shape[0], tokens.shape[1], tokens.shape[1])
        consts = self.original.make_consts(self.cfg, phase, self.torch.float32, self.torch.device("cpu"))
        return self.original.forward(self.params, tokens, consts)

    def assert_close(self, actual, expected):
        self.torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)

    def test_prefill_matches_existing_model_and_gqa(self):
        with self.torch.inference_mode():
            actual, actual_kv = self.runner.prefill(self.tokens)
            expected, expected_kv = self.reference(self.tokens)
        self.assert_close(actual, expected)
        for (ak, av), (ek, ev) in zip(actual_kv, expected_kv):
            self.assert_close(ak, ek)
            self.assert_close(av, ev)
        self.assertEqual(actual.shape, (2, 16, self.cfg.vocab))
        self.assertEqual(actual_kv[0][0].shape[2], 16)

    def test_decode_matches_prefill_and_preserves_cache_prefix(self):
        torch = self.torch
        with torch.inference_mode():
            _, prefix = self.runner.prefill(self.tokens[:, :12])
            cache = []
            for k, v in prefix:
                shape = (2, self.cfg.n_kv_heads, self.runner.capacity, self.cfg.head_dim)
                kc, vc = torch.randn(shape) * 100, torch.randn(shape) * 100
                kc[:, :, :12], vc[:, :, :12] = k, v
                cache.append((kc, vc))
            before = [(k.clone(), v.clone()) for k, v in cache]
            position = torch.tensor(12, dtype=torch.int64)
            actual, output_cache = self.runner.decode(self.tokens[:, 12:13], cache, position)
            expected, expected_kv = self.reference(self.tokens[:, :13])
            self.assert_close(actual[:, 0], expected[:, -1])
            for (ak, av), (bk, bv), (ek, ev) in zip(output_cache, before, expected_kv):
                self.assertTrue(torch.equal(ak[:, :, :12], bk[:, :, :12]))
                self.assertTrue(torch.equal(av[:, :, :12], bv[:, :, :12]))
                self.assertTrue(torch.equal(ak[:, :, 13:], bk[:, :, 13:]))
                self.assertTrue(torch.equal(av[:, :, 13:], bv[:, :, 13:]))
                self.assert_close(ak[:, :, :13], ek)
                self.assert_close(av[:, :, :13], ev)
            self.runner.advance(position)
            self.assertEqual(position.item(), 13)
            actual, _ = self.runner.decode(self.tokens[:, 13:14], cache, position)
            expected, _ = self.reference(self.tokens[:, :14])
            self.assert_close(actual[:, 0], expected[:, -1])

    def test_causal_prefix_does_not_see_future_tokens(self):
        with self.torch.inference_mode():
            full, _ = self.runner.prefill(self.tokens)
            prefix, _ = self.runner.prefill(self.tokens[:, :8])
        self.assert_close(full[:, :8], prefix)

    def test_changed_inputs_and_weights_are_computed(self):
        torch = self.torch
        with torch.inference_mode():
            original, _ = self.runner.prefill(self.tokens)
            changed, _ = self.runner.prefill((self.tokens + 1) % self.cfg.vocab)
            params = self.original.random_params(self.cfg, torch.float32, torch.device("cpu"), seed=42)
            runner = self.runner_cls(self.cfg, params, capacity=24, compile_model=False)
            reweighted, _ = runner.prefill(self.tokens)
            selected = self.runner.select_token(original)
        self.assertFalse(torch.equal(original, changed))
        self.assertFalse(torch.equal(original, reweighted))
        self.assertTrue(torch.equal(selected, original[:, -1].argmax(-1, keepdim=True)))
        self.assertEqual(selected.dtype, torch.int64)

    def test_copy_prefill_cache_preserves_unused_capacity(self):
        torch = self.torch
        with torch.inference_mode():
            _, source = self.runner.prefill(self.tokens[:, :8])
            shape = (2, self.cfg.n_kv_heads, self.runner.capacity, self.cfg.head_dim)
            destination = [(torch.full(shape, 17.0), torch.full(shape, -11.0)) for _ in source]
            pointers = [(k.data_ptr(), v.data_ptr()) for k, v in destination]
            self.runner.copy_prefill_cache(source, destination)
        for (sk, sv), (dk, dv), (kp, vp) in zip(source, destination, pointers):
            self.assertTrue(torch.equal(sk, dk[:, :, :8]))
            self.assertTrue(torch.equal(sv, dv[:, :, :8]))
            self.assertTrue(bool((dk[:, :, 8:] == 17.0).all()))
            self.assertTrue(bool((dv[:, :, 8:] == -11.0).all()))
            self.assertEqual((dk.data_ptr(), dv.data_ptr()), (kp, vp))

    def test_native_torch_rejects_custom_compute_policies(self):
        """Policy overrides must never silently change the PyTorch baseline."""
        for policy in ("decode-triton", "m1-triton"):
            with self.subTest(matmul_backend=policy), self.assertRaisesRegex(ValueError, "hybrid"):
                self.runner_cls(
                    self.cfg, self.params, capacity=24, compile_model=False, matmul_backend=policy
                )
        for policy in ("triton", "decode-triton"):
            with self.subTest(attention_backend=policy), self.assertRaisesRegex(ValueError, "hybrid"):
                self.runner_cls(
                    self.cfg, self.params, capacity=24, compile_model=False, attention_backend=policy
                )

    def test_full_context_decode_matches_original_model(self):
        """Omitting the mask is valid when decode fills the final cache slot."""
        torch = self.torch
        runner = self.runner_cls(
            self.cfg, self.params, capacity=16, compile_model=False, decode_full_context=True
        )
        with torch.inference_mode():
            _, prefix = runner.prefill(self.tokens[:, :15])
            shape = (2, self.cfg.n_kv_heads, 16, self.cfg.head_dim)
            cache = [(torch.zeros(shape), torch.zeros(shape)) for _ in prefix]
            runner.copy_prefill_cache(prefix, cache)
            actual, _ = runner.decode(self.tokens[:, 15:16], cache, torch.tensor(15, dtype=torch.int64))
            expected, _ = self.reference(self.tokens)
        self.assert_close(actual[:, 0], expected[:, -1])

    def test_decode_traces_fullgraph_without_position_recompile(self):
        """Changing a tensor position must retain one graph and attend new KV."""
        torch = self.torch
        traced_graphs = []

        def backend(graph, example_inputs):
            traced_graphs.append(graph)
            return graph.forward

        decode = torch.compile(self.runner._decode, backend=backend, fullgraph=True, dynamic=False)
        with torch.inference_mode():
            _, prefix = self.runner.prefill(self.tokens[:, :12])
            shape = (2, self.cfg.n_kv_heads, self.runner.capacity, self.cfg.head_dim)
            cache = [(torch.zeros(shape), torch.zeros(shape)) for _ in prefix]
            self.runner.copy_prefill_cache(prefix, cache)
            position = torch.tensor(12, dtype=torch.int64)
            token_buffer = self.tokens[:, 12:13].contiguous().clone()
            for step in (12, 13):
                position.fill_(step)
                token_buffer.copy_(self.tokens[:, step:step+1])
                actual, _ = decode(token_buffer, cache, position)
                expected, _ = self.reference(self.tokens[:, :step+1])
                self.assert_close(actual[:, 0], expected[:, -1])
        self.assertEqual(len(traced_graphs), 1)


if __name__ == "__main__":
    unittest.main()
