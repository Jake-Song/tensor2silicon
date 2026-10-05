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
from .config import gemm_tile, phase_options


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


def linear(x, w, *, tiles=None, interpret=False):
    m, k, n = math.prod(x.shape[:-1]), w.shape[0], w.shape[1]
    bm, bk, bn = gemm_tile(m, k, n, tiles)
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


def linear_heads(x, w, head_dim, *, cos=None, sin=None, scale=1.0, tiles=None, interpret=False):
    """Projection and head layout/RoPE in one kernel, retaining BF16 boundaries."""
    b, t, k = x.shape
    n = w.shape[1]
    bm, bk, bn = gemm_tile(b * t, k, n, tiles)
    decode = t == 1
    if not decode:
        bm = min(bm, t)
    if n % head_dim or bn % head_dim or (b if decode else t) % bm:
        raise ValueError("projection tile must divide head and token dimensions")
    nh = bn // head_dim
    reduction_axis = 2 if decode else 3

    def kernel(*refs):
        x_ref, w_ref = refs[:2]
        out_ref, acc_ref = refs[-2:]
        @pl.when(pl.program_id(reduction_axis) == 0)
        def init():
            acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        acc_ref[...] += jnp.matmul(x_ref[...], w_ref[...],
            preferred_element_type=jnp.float32, precision=jax.lax.Precision.DEFAULT)
        @pl.when(pl.program_id(reduction_axis) == k // bk - 1)
        def finish():
            # Match the separate linear output rounding before applying RoPE.
            y = acc_ref[...].astype(x.dtype).reshape(bm, nh, head_dim)
            if cos is not None:
                yf = y.astype(jnp.float32)
                c = (refs[2][...].astype(jnp.float32) * scale).astype(x.dtype).astype(jnp.float32)
                s = (refs[3][...].astype(jnp.float32) * scale).astype(x.dtype).astype(jnp.float32)
                c, s = c[:, None, :], s[:, None, :]
                left, right = yf[:, :, :head_dim // 2], yf[:, :, head_dim // 2:]
                y = jnp.concatenate((left * c - right * s, right * c + left * s), axis=2).astype(x.dtype)
            out_ref[...] = y.reshape(bm, nh, 1, head_dim) if decode else jnp.transpose(y, (1, 0, 2))

    if decode:
        grid = (b // bm, n // bn, k // bk)
        specs = [pl.BlockSpec((bm, bk), lambda i, j, r: (i, r)),
                 pl.BlockSpec((bk, bn), lambda i, j, r: (r, j))]
        outs = pl.BlockSpec((bm, nh, 1, head_dim), lambda i, j, r: (i, j, 0, 0))
        cs = pl.BlockSpec((1, head_dim // 2), lambda i, j, r: (0, 0))
        args = [x.reshape(b, k), w]
    else:
        grid = (b, t // bm, n // bn, k // bk)
        specs = [pl.BlockSpec((None, bm, bk), lambda b, i, j, r: (b, i, r)),
                 pl.BlockSpec((bk, bn), lambda b, i, j, r: (r, j))]
        outs = pl.BlockSpec((None, nh, bm, head_dim), lambda b, i, j, r: (b, j, i, 0))
        cs = pl.BlockSpec((bm, head_dim // 2), lambda b, i, j, r: (i, 0))
        args = [x, w]
    if cos is not None:
        specs += [cs, cs]
        args += [cos, sin]
    return pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((b, n // head_dim, t, head_dim), x.dtype),
        grid=grid, in_specs=specs, out_specs=outs,
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * (len(grid) - 1) + ("arbitrary",)),
        interpret=interpret, name="native_gemm_heads")(*args)


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


def kv_write(cache, new, *, block_s=256, group_heads=1, interpret=False):
    b, heads, s, h = cache.shape
    bs = min(block_s, s)
    if bs <= 0 or s % bs:
        raise ValueError("KV block must be positive and divide the context")
    if group_heads > 1:
        return kv_write_grouped((cache,), (new,), block_s=bs, group_heads=group_heads,
                                interpret=interpret)[0]

    def kernel(old_ref, new_ref, out_ref):
        idx = pl.program_id(2) * bs + jnp.arange(bs)
        out_ref[...] = jnp.where(idx[:, None] == s - 1, new_ref[...], old_ref[...])

    cs = pl.BlockSpec((None, None, bs, h), lambda b, h, i: (b, h, i, 0))
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype), grid=(b, heads, s // bs),
        in_specs=[cs, pl.BlockSpec((None, None, 1, h), lambda b, h, i: (b, h, 0, 0))],
        out_specs=cs, compiler_params=_parallel(3), interpret=interpret, name="native_kv_write",
    )(cache, new)


def kv_write_grouped(caches, news, *, block_s=2048, group_heads=4, interpret=False):
    """Contiguous head groups amortize tiny pipeline iterations; no input aliasing."""
    b, heads, s, h = caches[0].shape
    bs, gh = min(block_s, s), min(group_heads, b * heads)
    if bs <= 0 or gh <= 0 or s % bs or (b * heads) % gh:
        raise ValueError("KV blocks must divide the context and flattened head count")
    count = len(caches)
    if count != len(news) or count not in (1, 2):
        raise ValueError("expected one cache or a K/V pair")

    def kernel(*refs):
        idx = pl.program_id(1) * bs + jnp.arange(bs)
        for i in range(count):
            refs[2 * count + i][...] = jnp.where(
                idx[None, :, None] == s - 1, refs[count + i][...], refs[i][...])

    cs = pl.BlockSpec((gh, bs, h), lambda g, i: (g, i, 0))
    ns = pl.BlockSpec((gh, 1, h), lambda g, i: (g, 0, 0))
    outputs = pl.pallas_call(
        kernel, out_shape=tuple(jax.ShapeDtypeStruct((b * heads, s, h), c.dtype) for c in caches),
        grid=(b * heads // gh, s // bs), in_specs=[cs] * count + [ns] * count,
        out_specs=tuple([cs] * count), compiler_params=_parallel(2), interpret=interpret,
        name="native_kv_pair" if count == 2 else "native_kv_grouped",
    )(*(c.reshape(b * heads, s, h) for c in caches), *(n.reshape(b * heads, 1, h) for n in news))
    return tuple(o.reshape(b, heads, s, h) for o in outputs)


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


def kv_write_flat(cache, new, *, group_heads=4, interpret=False):
    """Copy flat contiguous rows, then select only each head's aligned tail."""
    b, heads, s, h = cache.shape
    gh = min(group_heads, b * heads)
    if gh <= 0 or (b * heads) % gh or s < 8 or s % 8:
        raise ValueError("flat KV copy requires whole head groups and 8-row contexts")
    rows = gh * s

    def kernel(old_ref, new_ref, out_ref):
        out_ref[...] = old_ref[...]
        for head in range(gh):
            start = (head + 1) * s - 8
            out_ref[start:start + 8, :] = jnp.where(
                jax.lax.broadcasted_iota(jnp.int32, (8, h), 0) == 7,
                new_ref[head, :, :], old_ref[start:start + 8, :])

    cs = pl.BlockSpec((rows, h), lambda i: (i, 0))
    result = pl.pallas_call(kernel,
        out_shape=jax.ShapeDtypeStruct((b * heads * s, h), cache.dtype),
        grid=(b * heads // gh,),
        in_specs=[cs, pl.BlockSpec((gh, 1, h), lambda i: (i, 0, 0))], out_specs=cs,
        compiler_params=_parallel(1), interpret=interpret, name="native_kv_flat",
    )(cache.reshape(b * heads * s, h), new.reshape(b * heads, 1, h))
    return result.reshape(cache.shape)


def forward(params, tokens, consts, cache=None, *, block_q=128, tuning=None, interpret=False):
    kw = {"interpret": interpret}
    options = phase_options(tuning, "native", cache)
    block_q = options.get("block_q", block_q)
    gemm = partial(linear, tiles=options.get("gemm"), **kw)
    hdim = 2 * consts["cos"].shape[-1]
    x = embed(tokens, params["embed"], **kw)
    kv = []
    for i, p in enumerate(params["layers"]):
        h = layernorm(x, p["ln1_w"], p["ln1_b"], **kw)
        if options.get("fuse_projection", False):
            projection = partial(linear_heads, head_dim=hdim, tiles=options.get("gemm"), **kw)
            q = projection(h, p["wq"], cos=consts["cos"], sin=consts["sin"], scale=hdim**-0.5)
            k = projection(h, p["wk"], cos=consts["cos"], sin=consts["sin"])
            v = projection(h, p["wv"])
        else:
            q = split_heads(gemm(h, p["wq"]), hdim, cos=consts["cos"], sin=consts["sin"], scale=hdim**-0.5, **kw)
            k = split_heads(gemm(h, p["wk"]), hdim, cos=consts["cos"], sin=consts["sin"], **kw)
            v = split_heads(gemm(h, p["wv"]), hdim, **kw)
        if cache is not None:
            ck = dict(block_s=options.get("kv_block_s", 256),
                      group_heads=options.get("kv_group_heads", 1), **kw)
            if options.get("kv_flat_tail", False):
                if options.get("kv_pair", False) or min(ck["block_s"], cache[i][0].shape[2]) != cache[i][0].shape[2]:
                    raise ValueError("flat KV copy requires full-context blocks and separate K/V outputs")
                k, v = (kv_write_flat(cache[i][0], k, group_heads=ck["group_heads"], **kw),
                        kv_write_flat(cache[i][1], v, group_heads=ck["group_heads"], **kw))
            elif options.get("kv_pair", False):
                k, v = kv_write_grouped(cache[i], (k, v), **ck)
            else:
                k, v = kv_write(cache[i][0], k, **ck), kv_write(cache[i][1], v, **ck)
        kv.append((k, v))
        o = attention(q, k, v, block_q=block_q, decode_heads=options.get("decode_heads", 1), **kw)
        x = elementwise(x, gemm(merge_heads(o, **kw), p["wo"]), **kw)
        h = layernorm(x, p["ln2_w"], p["ln2_b"], **kw)
        act = elementwise(gemm(h, p["w_gate"]), gemm(h, p["w_up"]), swiglu=True, **kw)
        x = elementwise(x, gemm(act, p["w_down"]), **kw)
    return gemm(layernorm(x, params["lnf_w"], params["lnf_b"], **kw), params["lm_head"]), kv
