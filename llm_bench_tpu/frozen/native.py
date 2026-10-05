"""All decoder compute in explicit Pallas TPU kernels.

JAX supplies arrays, compilation/dispatch, and metadata-only reshapes. No
JAX matmul, normalization, gather, activation or attention computes outside
the Pallas calls. Weights retain the reference's row-major shapes.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernels import attention


def _parallel(n):
    return pltpu.CompilerParams(dimension_semantics=("parallel",) * n)


def embed(tokens, table, *, interpret=False):
    m, d = math.prod(tokens.shape), table.shape[1]

    def kernel(tokens_ref, table_ref, out_ref):
        out_ref[...] = table_ref[...]

    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((m, 1, d), table.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(m,),
            in_specs=[pl.BlockSpec((None, 1, d), lambda i, ids: (ids[i], 0, 0))],
            out_specs=pl.BlockSpec((None, 1, d), lambda i, ids: (i, 0, 0)),
        ),
        compiler_params=_parallel(1), interpret=interpret, name="native_embed",
    )(tokens.reshape(m), table.reshape(table.shape[0], 1, d))
    return out.reshape(*tokens.shape, d)


def layernorm(x, w, b, *, interpret=False):
    m, d = math.prod(x.shape[:-1]), x.shape[-1]
    bm = min(128, m)

    def kernel(x_ref, w_ref, b_ref, out_ref):
        xf = x_ref[...].astype(jnp.float32)
        mean = jnp.sum(xf, axis=1, keepdims=True) / d
        centered = xf - mean
        var = jnp.sum(centered * centered, axis=1, keepdims=True) / d
        y = centered * jax.lax.rsqrt(var + 1e-5)
        out_ref[...] = (y * w_ref[...].astype(jnp.float32)[None, :]
                        + b_ref[...].astype(jnp.float32)[None, :]).astype(x.dtype)

    bs = pl.BlockSpec((bm, d), lambda i: (i, 0))
    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((m, d), x.dtype), grid=(m // bm,),
        in_specs=[bs, pl.BlockSpec((d,), lambda i: (0,)), pl.BlockSpec((d,), lambda i: (0,))],
        out_specs=bs, compiler_params=_parallel(1), interpret=interpret, name="native_layernorm",
    )(x.reshape(m, d), w, b)
    return out.reshape(x.shape)


def _matmul_kernel(x_ref, w_ref, out_ref, acc_ref, *, steps):
    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    acc_ref[...] += jnp.matmul(x_ref[...], w_ref[...], preferred_element_type=jnp.float32,
                              precision=jax.lax.Precision.DEFAULT)

    @pl.when(pl.program_id(2) == steps - 1)
    def finish():
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def linear(x, w, *, interpret=False):
    m, k, n = math.prod(x.shape[:-1]), w.shape[0], w.shape[1]
    bm = min(512, m)
    bk = min(1024 if k % 1024 == 0 else 512, k)
    bn = min(512 if n % 512 == 0 else 256, n)
    if x.shape[-1] != k or m % bm or k % bk or n % bn:
        raise ValueError(f"unsupported GEMM shape: {(m, k, n)}")
    out = pl.pallas_call(
        partial(_matmul_kernel, steps=k // bk),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype), grid=(m // bm, n // bn, k // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, r: (i, r)),
                  pl.BlockSpec((bk, bn), lambda i, j, r: (r, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, r: (i, j)),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        interpret=interpret, name="native_gemm",
    )(x.reshape(m, k), w)
    return out.reshape(*x.shape[:-1], n)


def _rope_kernel(x_ref, cos_ref, sin_ref, out_ref, *, scale):
    h = x_ref.shape[-1]
    x = x_ref[...].astype(jnp.float32)
    c = (cos_ref[...].astype(jnp.float32) * scale).astype(x_ref.dtype).astype(jnp.float32)
    s = (sin_ref[...].astype(jnp.float32) * scale).astype(x_ref.dtype).astype(jnp.float32)
    left, right = x[:, :h // 2], x[:, h // 2:]
    out_ref[...] = jnp.concatenate((left * c - right * s, right * c + left * s), axis=1).astype(out_ref.dtype)


def split_heads(x, head_dim, *, cos=None, sin=None, scale=1.0, interpret=False):
    b, t, d = x.shape
    heads, bt = d // head_dim, min(128, t)
    grid = (b, heads, t // bt)
    ins = pl.BlockSpec((None, bt, None, None, head_dim), lambda b, h, i: (b, i, h, 0, 0))
    outs = pl.BlockSpec((None, None, bt, head_dim), lambda b, h, i: (b, h, i, 0))
    args = [x.reshape(b, t, heads, 1, head_dim)]
    specs = [ins]
    if cos is not None:
        kernel = partial(_rope_kernel, scale=scale)
        args.extend([cos, sin])
        cs = pl.BlockSpec((bt, head_dim // 2), lambda b, h, i: (i, 0))
        specs.extend([cs, cs])
    else:
        def kernel(x_ref, out_ref):
            out_ref[...] = x_ref[...]
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((b, heads, t, head_dim), x.dtype),
        grid=grid, in_specs=specs, out_specs=outs, compiler_params=_parallel(3),
        interpret=interpret, name="native_rope" if cos is not None else "native_split_heads",
    )(*args)


def merge_heads(x, *, interpret=False):
    b, heads, t, h = x.shape
    bt = min(128, t)

    def kernel(x_ref, out_ref):
        out_ref[...] = x_ref[...]

    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((b, t, heads, 1, h), x.dtype), grid=(b, heads, t // bt),
        in_specs=[pl.BlockSpec((None, None, bt, h), lambda b, h, i: (b, h, i, 0))],
        out_specs=pl.BlockSpec((None, bt, None, None, h), lambda b, h, i: (b, i, h, 0, 0)),
        compiler_params=_parallel(3), interpret=interpret, name="native_merge_heads",
    )(x)
    return out.reshape(b, t, heads * h)


def kv_write(cache, new, *, interpret=False):
    b, heads, s, h = cache.shape
    bs = min(256, s)

    def kernel(old_ref, new_ref, out_ref):
        idx = pl.program_id(2) * bs + jnp.arange(bs)
        out_ref[...] = jnp.where(idx[:, None] == s - 1, new_ref[...], old_ref[...])

    cs = pl.BlockSpec((None, None, bs, h), lambda b, h, i: (b, h, i, 0))
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype), grid=(b, heads, s // bs),
        in_specs=[cs, pl.BlockSpec((None, None, 1, h), lambda b, h, i: (b, h, 0, 0))],
        out_specs=cs, compiler_params=_parallel(3), interpret=interpret, name="native_kv_write",
    )(cache, new)


def elementwise(x, y, *, swiglu=False, interpret=False):
    m, d = math.prod(x.shape[:-1]), x.shape[-1]
    bm = min(128, m)
    bn = min(512, d)
    if d % bn or m % bm:
        raise ValueError("unsupported elementwise shape")

    def kernel(x_ref, y_ref, out_ref):
        a, b = x_ref[...].astype(jnp.float32), y_ref[...].astype(jnp.float32)
        if swiglu:
            silu = (a / (1 + jnp.exp(-a))).astype(x_ref.dtype).astype(jnp.float32)
            out_ref[...] = (silu * b).astype(out_ref.dtype)
        else:
            out_ref[...] = (a + b).astype(out_ref.dtype)

    bs = pl.BlockSpec((bm, bn), lambda i, j: (i, j))
    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((m, d), x.dtype), grid=(m // bm, d // bn),
        in_specs=[bs, bs], out_specs=bs, compiler_params=_parallel(2), interpret=interpret,
        name="native_swiglu" if swiglu else "native_residual",
    )(x.reshape(m, d), y.reshape(m, d))
    return out.reshape(x.shape)


def forward(params, tokens, consts, cache=None, *, block_q=128, interpret=False):
    kw = {"interpret": interpret}
    hdim = 2 * consts["cos"].shape[-1]
    x = embed(tokens, params["embed"], **kw)
    kv = []
    for i, p in enumerate(params["layers"]):
        h = layernorm(x, p["ln1_w"], p["ln1_b"], **kw)
        q = split_heads(linear(h, p["wq"], **kw), hdim, cos=consts["cos"], sin=consts["sin"], scale=hdim**-0.5, **kw)
        k = split_heads(linear(h, p["wk"], **kw), hdim, cos=consts["cos"], sin=consts["sin"], **kw)
        v = split_heads(linear(h, p["wv"], **kw), hdim, **kw)
        if cache is not None:
            k, v = kv_write(cache[i][0], k, **kw), kv_write(cache[i][1], v, **kw)
        kv.append((k, v))
        o = attention(q, k, v, block_q=block_q, **kw)
        x = elementwise(x, linear(merge_heads(o, **kw), p["wo"], **kw), **kw)
        h = layernorm(x, p["ln2_w"], p["ln2_b"], **kw)
        act = elementwise(linear(h, p["w_gate"], **kw), linear(h, p["w_up"], **kw), swiglu=True, **kw)
        x = elementwise(x, linear(act, p["w_down"], **kw), **kw)
    return linear(layernorm(x, params["lnf_w"], params["lnf_b"], **kw), params["lm_head"], **kw), kv
