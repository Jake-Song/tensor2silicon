"""Compiled PyTorch and compiled PyTorch plus custom Triton model runners.

Parameter packing and RoPE table creation happen in ``__init__``, before timing.
Both runners use BF16 tensor storage with FP32 normalization/RoPE/SwiGLU
intermediates. The benchmark's independent FP32 reference is the original
``llm_roofline.torch_llm.forward`` using the same quantized weights promoted to
FP32; its execution is not part of BF16 timing.

Prefill returns compact K/V tensors. Decode consumes caller-owned fixed-capacity
caches and updates the GPU-resident ``position`` slot in place.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from llm_roofline import spec


class TorchRunner:
    """The same decoder graph with either PyTorch or custom small operations.

    ``params`` follows :func:`llm_roofline.spec.init_params`. Its tensors must
    already reside on the target device and share a floating point dtype.
    ``position`` is a one-element int64 tensor on that device; decode requires
    exactly one new token per batch and a valid slot in ``[0, capacity)``.
    Call all entry points under ``torch.inference_mode()``.
    """

    def __init__(
        self, cfg, params, capacity, hybrid=False, compile_model=True,
        attention_backend="sdpa", matmul_backend="torch", decode_full_context=False,
        flex_kernel_options=None, custom_ops="triton",
    ):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if cfg.n_heads % cfg.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if cfg.head_dim % 2:
            raise ValueError("split-half RoPE requires an even head_dim")
        if len(params["layers"]) != cfg.n_layers:
            raise ValueError("parameter layer count does not match cfg")
        if attention_backend not in ("sdpa", "triton", "decode-triton", "flex-decode"):
            raise ValueError("attention_backend must be 'sdpa', 'triton', 'decode-triton' or 'flex-decode'")
        if attention_backend in ("triton", "decode-triton") and not hybrid:
            raise ValueError("custom Triton attention is only available to the hybrid runner")
        if matmul_backend not in ("torch", "decode-triton", "m1-triton"):
            raise ValueError("matmul_backend must be 'torch', 'decode-triton' or 'm1-triton'")
        if matmul_backend != "torch" and not hybrid:
            raise ValueError("custom Triton matmul is only available to the hybrid runner")
        self.cfg = cfg
        self.capacity = int(capacity)
        self.hybrid = bool(hybrid)
        self.attention_backend = attention_backend
        self.matmul_backend = matmul_backend
        self.flex_kernel_options = None if flex_kernel_options is None else dict(flex_kernel_options)
        # Explicit caller promise: every cache slot is valid in fixed-context
        # decode. Dynamic generation must leave this False to mask future slots.
        self.decode_full_context = bool(decode_full_context)
        self.compiled = bool(compile_model)
        self.dtype = params["embed"].dtype
        self.device = params["embed"].device
        self.params = {
            "embed": params["embed"],
            "lnf_w": params["lnf_w"],
            "lnf_b": params["lnf_b"],
            "lm_head": params["lm_head"],
            "layers": [
                {
                    "qkv": torch.cat((p["wq"], p["wk"], p["wv"]), dim=-1).contiguous(),
                    "gate_up": torch.cat((p["w_gate"], p["w_up"]), dim=-1).contiguous(),
                    **{k: p[k] for k in ("ln1_w", "ln1_b", "ln2_w", "ln2_b", "wo", "w_down")},
                }
                for p in params["layers"]
            ],
        }
        cos, sin = spec.rope_tables(cfg.head_dim, np.arange(self.capacity))
        self.cos = torch.as_tensor(cos, dtype=torch.float32, device=self.device)
        self.sin = torch.as_tensor(sin, dtype=torch.float32, device=self.device)
        self.cache_positions = torch.arange(self.capacity, dtype=torch.int64, device=self.device)
        if self.hybrid:
            if custom_ops == "cuda":
                from . import cuda_ops
                cuda_ops.extension()
                self.ops = cuda_ops
            else:
                from . import triton_ops
                self.ops = triton_ops
        else:
            self.ops = None
        if attention_backend in ("triton", "decode-triton"):
            from .triton_attention import attention

            self.triton_attention = attention
        else:
            self.triton_attention = None
        if attention_backend == "flex-decode":
            from torch.nn.attention.flex_attention import flex_attention

            self.flex_attention = flex_attention
        else:
            self.flex_attention = None
        if matmul_backend != "torch":
            from .triton_matmul import linear

            self.triton_linear = linear
        else:
            self.triton_linear = None
        compile_options = dict(fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)
        self.prefill = torch.compile(self._prefill, **compile_options) if compile_model else self._prefill
        self.decode = torch.compile(self._decode, **compile_options) if compile_model else self._decode
        self.select_token = (
            torch.compile(self._select_token, **compile_options) if compile_model else self._select_token
        )
        self.advance = torch.compile(self._advance, **compile_options) if compile_model else self._advance

    def _layernorm(self, x, w, b):
        if self.hybrid:
            return self.ops.layernorm(x, w, b, eps=1e-5)
        return F.layer_norm(x, (self.cfg.d_model,), w, b, eps=1e-5)

    def _add_layernorm(self, x, residual, w, b):
        if self.hybrid:
            return self.ops.add_layernorm(x, residual, w, b, eps=1e-5)
        summed = x + residual
        return summed, F.layer_norm(summed, (self.cfg.d_model,), w, b, eps=1e-5)

    def _rope_qkv(self, packed, position=None):
        cfg = self.cfg
        if self.hybrid:
            return self.ops.rope_qkv(packed, self.cos, self.sin, cfg.n_heads, cfg.n_kv_heads, position)
        batch, length, _ = packed.shape
        nq, nk, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q, k, v = torch.split(packed, (nq * hd, nk * hd, nk * hd), dim=-1)
        if position is None:
            cos, sin = self.cos[:length], self.sin[:length]
        else:
            indices = position.reshape(1)
            cos = torch.index_select(self.cos, 0, indices)
            sin = torch.index_select(self.sin, 0, indices)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]

        def rotate(x, heads):
            x = x.reshape(batch, length, heads, hd).float()
            a, b = x[..., : hd // 2], x[..., hd // 2 :]
            y = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
            return y.to(packed.dtype).transpose(1, 2).contiguous()

        q = rotate(q, nq)
        k = rotate(k, nk)
        v = v.reshape(batch, length, nk, hd).transpose(1, 2).contiguous()
        # Q is deliberately unscaled. SDPA applies exactly one 1/sqrt(head_dim).
        return q, k, v

    def _swiglu(self, packed):
        if self.hybrid:
            return self.ops.swiglu(packed)
        gate, up = packed.chunk(2, dim=-1)
        return (F.silu(gate.float()) * up.float()).to(packed.dtype)

    def _linear(self, x, weight, is_decode):
        use_triton = self.matmul_backend == "decode-triton" or (
            self.matmul_backend == "m1-triton" and x.numel() // x.shape[-1] == 1
        )
        if use_triton and is_decode:
            return self.triton_linear(x, weight)
        return x @ weight

    def _flex_decode(self, q, k, v, position):
        # PyTorch's score_mod API takes scalar (score, batch, head, q_idx,
        # kv_idx) tensors. Closing over this device tensor keeps cache position
        # dynamic without building a dense mask or authoring a Triton kernel.
        current_position = position.reshape(())

        def score_mod(score, batch, head, q_idx, kv_idx):
            return torch.where(kv_idx <= current_position, score, float("-inf"))

        return self.flex_attention(
            q, k, v, score_mod=score_mod, enable_gqa=self.cfg.n_heads != self.cfg.n_kv_heads,
            kernel_options=self.flex_kernel_options,
        )

    def _forward(self, tokens, cache=None, position=None):
        cfg, params = self.cfg, self.params
        x = self.ops.embedding(tokens, params["embed"]) if self.hybrid else F.embedding(tokens, params["embed"])
        if cfg.n_layers:
            first = params["layers"][0]
            h = self._layernorm(x, first["ln1_w"], first["ln1_b"])
        else:
            h = self._layernorm(x, params["lnf_w"], params["lnf_b"])
        out_cache = []
        # A tensor comparison preserves a single compiled decode graph as the
        # position changes. Future cache slots cannot influence attention.
        allowed = (
            None if cache is None or self.decode_full_context
            else (self.cache_positions <= position.reshape(())).view(1, -1)
        )
        is_decode = cache is not None
        for i, p in enumerate(params["layers"]):
            packed = self._linear(h, p["qkv"], is_decode)
            if cache is not None and self.hybrid:
                k, v = cache[i]
                q = self.ops.rope_qkv_decode(
                    packed, self.cos, self.sin, cfg.n_heads, cfg.n_kv_heads, position, k, v
                )
            else:
                q, k, v = self._rope_qkv(packed, position)
            if cache is not None and not self.hybrid:
                kc, vc = cache[i]
                indices = position.reshape(1)
                kc.index_copy_(2, indices, k)
                vc.index_copy_(2, indices, v)
                k, v = kc, vc
            out_cache.append((k, v))
            if self.attention_backend == "triton" or (
                self.attention_backend == "decode-triton" and cache is not None
            ):
                # The custom attention output is already contiguous [B,T,N*H].
                attention = self.triton_attention(q, k, v, position)
            else:
                if self.attention_backend == "flex-decode" and is_decode and not self.decode_full_context:
                    attention = self._flex_decode(q, k, v, position)
                else:
                    attention = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=allowed,
                        dropout_p=0.0,
                        is_causal=cache is None,
                        enable_gqa=cfg.n_heads != cfg.n_kv_heads,
                    )
                attention = attention.transpose(1, 2).reshape(
                    tokens.shape[0], tokens.shape[1], cfg.n_heads * cfg.head_dim
                )
            x, h = self._add_layernorm(
                x, self._linear(attention, p["wo"], is_decode), p["ln2_w"], p["ln2_b"]
            )
            residual = self._linear(
                self._swiglu(self._linear(h, p["gate_up"], is_decode)), p["w_down"], is_decode
            )
            if i + 1 < cfg.n_layers:
                nxt = params["layers"][i + 1]
                x, h = self._add_layernorm(x, residual, nxt["ln1_w"], nxt["ln1_b"])
            else:
                x, h = self._add_layernorm(x, residual, params["lnf_w"], params["lnf_b"])
        return self._linear(h, params["lm_head"], is_decode), out_cache

    def _prefill(self, tokens):
        return self._forward(tokens)

    def copy_prefill_cache(self, src, dst):
        """Copy compact prefill state into a caller-owned capacity cache.

        This helper is intentionally not separately compiled. The generation
        harness includes its copy kernels in the prefill CUDA graph/timing.
        """
        for (sk, sv), (dk, dv) in zip(src, dst, strict=True):
            if self.hybrid:
                self.ops.copy_prefix(sk, dk)
                self.ops.copy_prefix(sv, dv)
            else:
                dk[:, :, : sk.shape[2], :].copy_(sk)
                dv[:, :, : sv.shape[2], :].copy_(sv)

    def _decode(self, tokens, cache, position):
        return self._forward(tokens, cache, position)

    def _select_token(self, logits):
        if self.hybrid:
            return self.ops.select_token(logits)
        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    def _advance(self, position):
        if self.hybrid:
            self.ops.advance(position)
        else:
            position.add_(1)

    def copy_token(self, src, dst):
        if self.hybrid:
            self.ops.copy_token(src, dst)
        else:
            dst.copy_(src)
