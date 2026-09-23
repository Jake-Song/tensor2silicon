import importlib.util
import unittest

import numpy as np

from llm_roofline import spec

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_JAX = importlib.util.find_spec("jax") is not None


class SpecTests(unittest.TestCase):
    def test_nine_matmuls_per_layer(self):
        for phase in (spec.PRESETS["colab"].prefill, spec.PRESETS["colab"].decode):
            idx = [op.mm_index for op in spec.op_specs(spec.ModelConfig(), phase) if op.mm_index]
            self.assertEqual(idx, list(range(1, 10)))

    def test_matmul_flops_match_2_tokens_params(self):
        cfg, ph = spec.ModelConfig(), spec.PRESETS["colab"].prefill
        D, N, K, H, F = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff
        tokens = ph.batch * ph.q_len
        per_layer = {op.name: op.flops for op in spec.op_specs(cfg, ph) if op.mm_index}
        weight_mm = sum(f for n, f in per_layer.items() if n.endswith("_proj"))
        self.assertEqual(weight_mm, 2 * tokens * (D * (N + 2 * K) * H + N * H * D + 3 * D * F))
        self.assertEqual(per_layer["qk"] + per_layer["pv"], 4 * ph.batch * N * ph.q_len * ph.kv_len * H)

    def test_decode_weight_matmul_intensity_is_about_batch(self):
        ph = spec.PRESETS["colab"].decode
        for op in spec.op_specs(spec.ModelConfig(), ph):
            if op.name.endswith("_proj"):
                self.assertAlmostEqual(op.flops / op.bytes, ph.batch, delta=0.2)

    def test_classify(self):
        op = spec.op_specs(spec.ModelConfig(), spec.PRESETS["colab"].prefill)[2]   # q_proj
        row = spec.classify(op, op.flops / 100e12, peak_flops=200e12, peak_bw=1e12)
        self.assertEqual((row["bound"], row["measured"]), ("compute", "compute"))
        self.assertAlmostEqual(row["pct_roof"], 0.5)


def _logits(forward, params, tokens, consts, cache=None):
    return np.asarray(forward(params, tokens, consts, cache)[0], dtype=np.float32)


@unittest.skipUnless(HAVE_TORCH and HAVE_JAX, "torch and jax not installed (uv run --with torch --with jax ...)")
class ParityTests(unittest.TestCase):
    """Same NumPy weights through both frameworks, fp32 on CPU."""

    def setUp(self):
        import jax
        import torch
        from llm_roofline import jax_llm, torch_llm
        jax.config.update("jax_default_matmul_precision", "highest")
        self.torch, self.J, self.T = torch, jax_llm, torch_llm
        self.preset = spec.PRESETS["tiny"]
        self.cfg = self.preset.model
        self.np_params = spec.init_params(self.cfg, seed=0)
        self.cpu = torch.device("cpu")

    def run_both(self, phase, tokens, np_cache=None):
        import jax.numpy as jnp
        torch, J, T = self.torch, self.J, self.T
        tp = T.to_torch(self.np_params, torch.float32, self.cpu)
        tc = T.make_consts(self.cfg, phase, torch.float32, self.cpu)
        tcache = None if np_cache is None else [(torch.tensor(k), torch.tensor(v)) for k, v in np_cache]
        with torch.inference_mode():
            t_out = _logits(T.forward, tp, torch.tensor(tokens), tc, tcache)
        jcache = None if np_cache is None else [(jnp.asarray(k), jnp.asarray(v)) for k, v in np_cache]
        j_out = _logits(J.forward, J.to_jax(self.np_params, jnp.float32), jnp.asarray(tokens),
                        J.make_consts(self.cfg, phase, jnp.float32), jcache)
        return t_out, j_out

    def test_prefill_torch_matches_jax(self):
        ph = self.preset.prefill
        tokens = np.random.default_rng(1).integers(0, self.cfg.vocab, (ph.batch, ph.q_len))
        t_out, j_out = self.run_both(ph, tokens)
        np.testing.assert_allclose(t_out, j_out, rtol=1e-4, atol=1e-4)

    def test_decode_step_matches_prefill_last_position(self):
        """Prefill S-1 tokens, put their K/V in the cache, decode token S-1: its logits must equal
        the last row of a full S-token prefill. Checks the KV cache, RoPE positions and masking."""
        torch, T = self.torch, self.T
        dec = self.preset.decode
        S, B = dec.kv_len, dec.batch
        tokens = np.random.default_rng(2).integers(0, self.cfg.vocab, (B, S))
        full = self.run_both(spec.PhaseConfig("prefill", B, S, S), tokens)

        short = spec.PhaseConfig("prefill", B, S - 1, S - 1)
        tp = T.to_torch(self.np_params, torch.float32, self.cpu)
        with torch.inference_mode():
            _, kv = T.forward(tp, torch.tensor(tokens[:, :-1]), T.make_consts(self.cfg, short, torch.float32, self.cpu))
        pad = [(0, 0), (0, 0), (0, 1), (0, 0)]
        cache = [(np.pad(k.numpy(), pad), np.pad(v.numpy(), pad)) for k, v in kv]
        t_out, j_out = self.run_both(dec, tokens[:, -1:], cache)
        np.testing.assert_allclose(t_out[:, 0], full[0][:, -1], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(j_out[:, 0], full[1][:, -1], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
