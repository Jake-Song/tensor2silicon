"""All-compute Triton implementation of the repository's decoder-only model.

PyTorch supplies tensor storage, metadata/views and the caller's CUDA Graph
capture machinery. Model arithmetic, KV writes and token selection are Triton
kernels. Weight concatenation and RoPE table transfer happen only at setup.
"""

from __future__ import annotations

import numpy as np
import torch

from llm_roofline.spec import rope_tables

from . import triton_ops as ops
from .triton_attention import attention
from .triton_matmul import linear


class NativeRunner:
    """Triton decoder with explicit compact-prefill and static-decode caches."""

    def __init__(self, cfg, params, capacity: int):
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
        x, h = ops.add_layernorm(x, projected, p["ln2_w"], p["ln2_b"])
        gated = ops.swiglu(linear(h, p["w_gate_up"]))
        down = linear(gated, p["w_down"])
        if layer_index + 1 < len(self.layers):
            next_p = self.layers[layer_index + 1]
            return ops.add_layernorm(x, down, next_p["ln1_w"], next_p["ln1_b"])
        return ops.add_layernorm(x, down, self.params["lnf_w"], self.params["lnf_b"])

    def prefill(self, tokens):
        """Return all ``[B,T,vocab]`` logits and compact per-layer ``[B,K,T,H]`` KV."""
        if tokens.shape[1] > self.capacity:
            raise ValueError("prefill length exceeds RoPE/cache capacity")
        x = ops.embedding(tokens, self.params["embed"])
        first = self.layers[0]
        h = ops.layernorm(x, first["ln1_w"], first["ln1_b"])
        cache = []
        for i, p in enumerate(self.layers):
            qkv = linear(h, p["wqkv"])
            q, k, v = ops.rope_qkv(qkv, self.cos, self.sin, self.cfg.n_heads, self.cfg.n_kv_heads)
            cache.append((k, v))
            projected = linear(attention(q, k, v), p["wo"])
            x, h = self._finish_layer(x, projected, i)
        return linear(h, self.params["lm_head"]), cache

    def decode(self, tokens, cache, position):
        """Write one token at ``position`` and attend only through that cache slot.

        The caller owns ``position`` (CUDA int64 scalar) and its increment. The
        cache must be initialized through ``position-1``; its unused capacity
        can contain arbitrary values because attention masks future slots.
        """
        if tokens.shape[1] != 1:
            raise ValueError("decode accepts one token per sequence")
        x = ops.embedding(tokens, self.params["embed"])
        first = self.layers[0]
        h = ops.layernorm(x, first["ln1_w"], first["ln1_b"])
        for i, p in enumerate(self.layers):
            qkv = linear(h, p["wqkv"])
            k, v = cache[i]
            q = ops.rope_qkv_decode(
                qkv, self.cos, self.sin, self.cfg.n_heads, self.cfg.n_kv_heads, position, k, v,
            )
            projected = linear(attention(q, k, v, position=position), p["wo"])
            x, h = self._finish_layer(x, projected, i)
        return linear(h, self.params["lm_head"]), cache

    def copy_prefill_cache(self, src_cache, dst_cache):
        """Copy valid prefill prefixes using only Triton device work."""
        for (sk, sv), (dk, dv) in zip(src_cache, dst_cache, strict=True):
            ops.copy_prefix(sk, dk)
            ops.copy_prefix(sv, dv)

    @staticmethod
    def select_token(logits):
        return ops.select_token(logits)

    @staticmethod
    def advance(position):
        ops.advance(position)

    @staticmethod
    def write_token(logits, token_buffer, output_tokens, index):
        ops.select_token_into(logits, token_buffer)
        ops.record_token(token_buffer, output_tokens, index)
