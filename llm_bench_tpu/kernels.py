"""Fused, full-context attention with explicit BF16 score/probability boundaries.

Q has already received RoPE and the head_dim**-0.5 scale. Each program
computes a query tile for one batch/head, keeping scores/probabilities in
VMEM. This is full-context tiled attention, not online FlashAttention.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _attention_kernel(q_ref, k_ref, v_ref, out_ref, *, q_len, kv_len):
    scores = jax.lax.dot_general(
        q_ref[...], k_ref[...], (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32, precision=jax.lax.Precision.DEFAULT,
    ).astype(q_ref.dtype).astype(jnp.float32)
    rows = pl.program_id(2) * q_ref.shape[0] + jnp.arange(q_ref.shape[0])
    cols = jnp.arange(kv_len)
    scores = jnp.where(cols[None, :] <= rows[:, None] + kv_len - q_len,
                       scores, -jnp.inf)
    numer = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
    probs = (numer / jnp.sum(numer, axis=-1, keepdims=True)).astype(q_ref.dtype)
    out_ref[...] = jnp.matmul(
        probs, v_ref[...], preferred_element_type=jnp.float32, precision=jax.lax.Precision.DEFAULT,
    ).astype(out_ref.dtype)


def attention(q, k, v, *, block_q=128, decode_heads=1, interpret=False):
    """Causal prefill or fixed-context decode; supports grouped query heads.

Inputs/outputs: [batch, heads, sequence, head_dim]. Blocks span the entire
KV context, so this educational kernel targets the benchmark's 2048 slots.
It is not intended as a general long-context attention implementation.
"""
    batch, heads, q_len, head_dim = q.shape
    kv_heads, kv_len = k.shape[1:3]
    if k.shape != v.shape or k.shape[0] != batch or k.shape[-1] != head_dim:
        raise ValueError("K/V shape must match Q's batch and head dimension")
    if heads % kv_heads or q_len > kv_len:
        raise ValueError("query heads must be a multiple of KV heads; Q length <= KV length")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, V must have the same dtype")
    bq = min(block_q, q_len)
    if bq <= 0 or q_len % bq:
        raise ValueError("block_q must be positive and divide the query length")
    groups = heads // kv_heads
    if q_len == 1 and decode_heads > 1:
        return _decode_grouped(q, k, v, decode_heads, interpret=interpret)
    return pl.pallas_call(
        partial(_attention_kernel, q_len=q_len, kv_len=kv_len),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(batch, heads, q_len // bq),
        in_specs=[
            pl.BlockSpec((None, None, bq, head_dim), lambda b, h, i: (b, h, i, 0)),
            pl.BlockSpec((None, None, kv_len, head_dim), lambda b, h, i: (b, h // groups, 0, 0)),
            pl.BlockSpec((None, None, kv_len, head_dim), lambda b, h, i: (b, h // groups, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, bq, head_dim), lambda b, h, i: (b, h, i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
        interpret=interpret,
        name="same_model_pallas_attention",
    )(q, k, v)


def _decode_grouped(q, k, v, heads_per_block, *, interpret=False):
    """Load contiguous head groups together; preserve each head's BF16 softmax."""
    batch, heads, _, hdim = q.shape
    kv_heads, context = k.shape[1:3]
    groups = heads // kv_heads
    gh = min(heads_per_block, heads)
    if gh <= 0 or heads % gh or gh % groups:
        raise ValueError("decode head block must divide Q heads and contain whole KV groups")
    gkv = gh // groups

    def kernel(q_ref, k_ref, v_ref, out_ref):
        for h in range(gh):
            scores = jax.lax.dot_general(
                q_ref[h, :, :], k_ref[h // groups, :, :], (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32, precision=jax.lax.Precision.DEFAULT,
            ).astype(q.dtype).astype(jnp.float32)
            numer = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            probs = (numer / jnp.sum(numer, axis=-1, keepdims=True)).astype(q.dtype)
            out_ref[h, :, :] = jnp.matmul(probs, v_ref[h // groups, :, :],
                preferred_element_type=jnp.float32, precision=jax.lax.Precision.DEFAULT).astype(q.dtype)

    qs = pl.BlockSpec((None, gh, 1, hdim), lambda b, g: (b, g, 0, 0))
    ks = pl.BlockSpec((None, gkv, context, hdim), lambda b, g: (b, g, 0, 0))
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(batch, heads // gh), in_specs=[qs, ks, ks], out_specs=qs,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interpret, name="same_model_pallas_attention_grouped")(q, k, v)
