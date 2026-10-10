"""All-compute Triton or CUDA C++/cuBLAS implementation of the repository's decoder-only model.

PyTorch supplies tensor storage, metadata/views and the caller's CUDA Graph
capture machinery. Model arithmetic, KV writes and token selection are native
kernels. Weight concatenation and RoPE table transfer happen only at setup.
"""

from __future__ import annotations

import numpy as np
import torch

from llm_roofline.spec import rope_tables



class NativeRunner:
    """Native decoder with explicit compact-prefill and static-decode caches."""

    def __init__(self, cfg, params, capacity: int, implementation="triton"):
        if implementation not in ("triton", "cuda"):
            raise ValueError("unknown native implementation")
        if not cfg.n_layers:
            raise ValueError("native runner requires at least one layer")
        if cfg.d_model != cfg.n_heads * cfg.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if cfg.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if params["embed"].dtype != torch.bfloat16:
            raise ValueError("NativeRunner benchmarks BF16 parameters")
        if implementation == "cuda":
            from . import cuda_ops
            cuda_ops.extension()
            self.ops, self.linear, self.attention = cuda_ops, cuda_ops.linear, cuda_ops.attention
        else:
            from . import triton_ops, triton_matmul, triton_attention
            self.ops, self.linear, self.attention = triton_ops, triton_matmul.linear, triton_attention.attention
        self.cfg = cfg
        self.params = params
        self.capacity = capacity
        self.device = params["embed"].device
        self.dtype = params["embed"].dtype
        self.layers = []
        for p in params["layers"]:
            layer = dict(p)
            layer["wqkv"] = torch.cat((p["wq"], p["wk"], p["wv"]), dim=1).contiguous()
            layer["w_gate_up"] = torch.cat((p["w_gate"], p["w_up"]), dim=1).contiguous()
            self.layers.append(layer)
        cos, sin = rope_tables(cfg.head_dim, np.arange(capacity))
        # Shared FP32 constants also used by the PyTorch and FP32 reference paths.
        self.cos = torch.tensor(cos, device=self.device, dtype=torch.float32)
        self.sin = torch.tensor(sin, device=self.device, dtype=torch.float32)

    def _finish_layer(self, x, projected, layer_index):
        p = self.layers[layer_index]
        x, h = self.ops.add_layernorm(x, projected, p["ln2_w"], p["ln2_b"])
        gated = self.ops.swiglu(self.linear(h, p["w_gate_up"]))
        down = self.linear(gated, p["w_down"])
        if layer_index + 1 < len(self.layers):
            next_p = self.layers[layer_index + 1]
            return self.ops.add_layernorm(x, down, next_p["ln1_w"], next_p["ln1_b"])
        return self.ops.add_layernorm(x, down, self.params["lnf_w"], self.params["lnf_b"])

    def prefill(self, tokens):
        """Return all ``[B,T,vocab]`` logits and compact per-layer ``[B,K,T,H]`` KV."""
        if tokens.shape[1] > self.capacity:
            raise ValueError("prefill length exceeds RoPE/cache capacity")
        x = self.ops.embedding(tokens, self.params["embed"])
        first = self.layers[0]
        h = self.ops.layernorm(x, first["ln1_w"], first["ln1_b"])
        cache = []
        for i, p in enumerate(self.layers):
            qkv = self.linear(h, p["wqkv"])
            q, k, v = self.ops.rope_qkv(qkv, self.cos, self.sin, self.cfg.n_heads, self.cfg.n_kv_heads)
            cache.append((k, v))
            projected = self.linear(self.attention(q, k, v), p["wo"])
            x, h = self._finish_layer(x, projected, i)
        return self.linear(h, self.params["lm_head"]), cache

    def decode(self, tokens, cache, position):
        """Write one token at ``position`` and attend only through that cache slot.

        The caller owns ``position`` (CUDA int64 scalar) and its increment. The
        cache must be initialized through ``position-1``; its unused capacity
        can contain arbitrary values because attention masks future slots.
        """
        if tokens.shape[1] != 1:
            raise ValueError("decode accepts one token per sequence")
        x = self.ops.embedding(tokens, self.params["embed"])
        first = self.layers[0]
        h = self.ops.layernorm(x, first["ln1_w"], first["ln1_b"])
        for i, p in enumerate(self.layers):
            qkv = self.linear(h, p["wqkv"])
            k, v = cache[i]
            q = self.ops.rope_qkv_decode(
                qkv, self.cos, self.sin, self.cfg.n_heads, self.cfg.n_kv_heads, position, k, v,
            )
            projected = self.linear(self.attention(q, k, v, position=position), p["wo"])
            x, h = self._finish_layer(x, projected, i)
        return self.linear(h, self.params["lm_head"]), cache

    def copy_prefill_cache(self, src_cache, dst_cache):
        """Copy valid prefill prefixes using only Triton device work."""
        for (sk, sv), (dk, dv) in zip(src_cache, dst_cache, strict=True):
            self.ops.copy_prefix(sk, dk)
            self.ops.copy_prefix(sv, dv)

    def select_token(self, logits):
        return self.ops.select_token(logits)

    def advance(self, position):
        self.ops.advance(position)

    def write_token(self, logits, token_buffer, output_tokens, index):
        self.ops.select_token_into(logits, token_buffer)
        self.ops.record_token(token_buffer, output_tokens, index)

    def copy_token(self, src, dst):
        self.ops.copy_token(src, dst)
