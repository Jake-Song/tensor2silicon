"""Basic decoder-only LLM in JAX, timed op by op against the roofline (TPU, GPU or CPU).

    python -m llm_roofline.jax_llm --preset colab --phase both --out results_tpu.json

Same model and op list as torch_llm.py. Each op is jitted on its own, so XLA compiles it
as a standalone kernel (fusion only inside the op); the jitted whole model shows what
fusion across ops buys.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from . import spec

DTYPES = {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}


# ---------------------------------------------------------------------------
# Ops (one function per row of spec.op_specs)
# ---------------------------------------------------------------------------

def embed(tokens, table):
    return jnp.take(table, tokens, axis=0)


def layernorm(x, w, b):
    x32 = x.astype(jnp.float32)
    mu = x32.mean(-1, keepdims=True)
    var = jnp.square(x32 - mu).mean(-1, keepdims=True)
    y = (x32 - mu) * jax.lax.rsqrt(var + 1e-5)
    return (y * w.astype(jnp.float32) + b.astype(jnp.float32)).astype(x.dtype)


def linear(x, w):
    return x @ w


def rope_split_heads(q, k, v, cos, sin):
    """RoPE on q and k (q also takes the 1/sqrt(H) score scale), then [B,T,heads*H] -> [B,heads,T,H]."""
    B, T, _ = q.shape
    H = 2 * cos.shape[-1]
    scale = H ** -0.5

    def rot(x, c, s):
        x = x.reshape(B, T, -1, H)
        x1, x2 = x[..., : H // 2], x[..., H // 2:]
        c, s = c[None, :, None, :], s[None, :, None, :]
        return jnp.concatenate((x1 * c - x2 * s, x2 * c + x1 * s), axis=-1).transpose(0, 2, 1, 3)

    return rot(q, cos * scale, sin * scale), rot(k, cos, sin), v.reshape(B, T, -1, H).transpose(0, 2, 1, 3)


def kv_write(cache, new):
    """Decode: store the new token's K or V in the last cache slot (in place when the cache is donated)."""
    return jax.lax.dynamic_update_slice_in_dim(cache, new, cache.shape[2] - 1, axis=2)


def qk(q, k):
    """[B,N,T,H] x [B,K,S,H] -> scores [B,N,T,S]. GQA: the N/K query heads of a group share one K head."""
    B, N, T, H = q.shape
    K = k.shape[1]
    return (q.reshape(B, K, (N // K) * T, H) @ jnp.swapaxes(k, -1, -2)).reshape(B, N, T, -1)


def softmax_masked(s, mask):
    return jax.nn.softmax(jnp.where(mask, -jnp.inf, s.astype(jnp.float32)), axis=-1).astype(s.dtype)


def pv(p, v):
    B, N, T, S = p.shape
    K = v.shape[1]
    return (p.reshape(B, K, (N // K) * T, S) @ v).reshape(B, N, T, -1)


def merge_heads(o):
    B, N, T, H = o.shape
    return o.transpose(0, 2, 1, 3).reshape(B, T, N * H)


def add(x, y):
    return x + y


def swiglu(g, u):
    return jax.nn.silu(g) * u


OPS = {f.__name__: f for f in (embed, layernorm, linear, rope_split_heads, kv_write, qk, softmax_masked, pv,
                               merge_heads, add, swiglu)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def forward(params, tokens, consts, cache=None):
    """Prefill when cache is None; otherwise one decode step that fills the last cache slot.
    Returns (logits, [(k, v) per layer])."""
    x = embed(tokens, params["embed"])
    kv = []
    for i, p in enumerate(params["layers"]):
        h = layernorm(x, p["ln1_w"], p["ln1_b"])
        q, k, v = rope_split_heads(linear(h, p["wq"]), linear(h, p["wk"]), linear(h, p["wv"]), consts["cos"], consts["sin"])
        if cache is not None:
            k, v = kv_write(cache[i][0], k), kv_write(cache[i][1], v)
        kv.append((k, v))
        o = pv(softmax_masked(qk(q, k), consts["mask"]), v)
        x = add(x, linear(merge_heads(o), p["wo"]))
        h = layernorm(x, p["ln2_w"], p["ln2_b"])
        x = add(x, linear(swiglu(linear(h, p["w_gate"]), linear(h, p["w_up"])), p["w_down"]))
    return linear(layernorm(x, params["lnf_w"], params["lnf_b"]), params["lm_head"]), kv


def to_jax(tree, dtype):
    return jax.tree.map(lambda a: jnp.asarray(a, dtype) if np.issubdtype(np.asarray(a).dtype, np.floating)
                        else jnp.asarray(a), tree)


def random_params(cfg, dtype, seed=0):
    keys = iter(jax.random.split(jax.random.key(seed), 16 * cfg.n_layers + 8))

    def normal(shape, mean, std):
        return (jax.random.normal(next(keys), shape, jnp.float32) * std + mean).astype(dtype)

    return spec.init_params(cfg, normal=normal)


def make_consts(cfg, phase, dtype):
    cos, sin = spec.rope_tables(cfg.head_dim, spec.phase_positions(phase))
    return {"cos": jnp.asarray(cos, dtype), "sin": jnp.asarray(sin, dtype),
            "mask": jnp.asarray(spec.causal_mask(phase.q_len, phase.kv_len))}


def make_cache(cfg, phase, dtype, seed=1):
    shape = (phase.batch, cfg.n_kv_heads, phase.kv_len, cfg.head_dim)
    keys = jax.random.split(jax.random.key(seed), 2 * cfg.n_layers)
    return [(jax.random.normal(keys[2 * i], shape, dtype), jax.random.normal(keys[2 * i + 1], shape, dtype))
            for i in range(cfg.n_layers)]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timeit(step, state, target_s=0.02, rounds=7):
    """Median seconds per call of `state = step(state)`; blocks only at the end of each round,
    so dispatch overlaps device work."""
    state = jax.block_until_ready(step(state))
    t0 = time.perf_counter()
    state = jax.block_until_ready(step(state))
    reps = max(1, math.ceil(target_s / max(time.perf_counter() - t0, 1e-7)))
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            state = step(state)
        jax.block_until_ready(state)
        times.append((time.perf_counter() - t0) / reps)
    return statistics.median(times)


SHARE_BYTES = 64 * 2**20   # args at least this big are not copied per iteration: no on-chip memory can hold them


def make_args(op, cfg, dtype, key, k):
    """Returns (values, rotated). Rotated args are stacked k times along a new leading axis
    (k distinct copies to cycle through); big args and the in-place carry are passed once."""
    vals, rotated = [], []
    for i, (a, sub) in enumerate(zip(op.args, jax.random.split(key, len(op.args)))):
        itemsize = 4 if a.dtype == "int" else 1 if a.dtype == "bool" else cfg.dtype_bytes
        rot = math.prod(a.shape) * itemsize < SHARE_BYTES and not (op.inplace and i == 0)
        shape = (k, *a.shape) if rot else a.shape
        if a.dtype == "int":
            v = jax.random.randint(sub, shape, 0, cfg.vocab)
        elif a.dtype == "bool":
            v = jnp.broadcast_to(jnp.asarray(spec.causal_mask(*a.shape)), shape)
        else:
            std = 1 / math.sqrt(a.shape[0]) if a.role == "weight" and len(a.shape) == 2 else 1.0
            v = (jax.random.normal(sub, shape, jnp.float32) * std).astype(dtype)
        vals.append(v)
        rotated.append(rot)
    return vals, rotated


def scanned(fn, inplace, rotated):
    """jit(lax.scan) applying fn once per stacked copy of the rotated args: device time without
    per-call dispatch, and nothing to hoist since every iteration reads different data. For an
    in-place op (KV cache write) the first arg is the loop carry, so it is updated in place."""
    def call(shared, x, first=()):
        s, r = iter(shared), iter(x)
        return fn(*first, *[next(r) if rot else next(s) for rot in rotated[len(first):]])

    if inplace:
        def run(carry, shared, xs):
            return jax.lax.scan(lambda c, x: (call(shared, x, (c,)), None), carry, xs)[0]
        return jax.jit(run, donate_argnums=0)

    def run(shared, xs):
        return jax.lax.scan(lambda c, x: (c, call(shared, x)), None, xs)[1]
    return jax.jit(run)


def cost_analysis(fn, args, inplace):
    try:
        ca = jax.jit(fn, donate_argnums=0 if inplace else ()).lower(*args).compile().cost_analysis()
        ca = ca[0] if isinstance(ca, (list, tuple)) else ca
        return float(ca.get("flops", math.nan)), float(ca.get("bytes accessed", math.nan))
    except Exception:
        return math.nan, math.nan


def measure_ceilings(dtype, n, bw_bytes):
    """Matmul FLOP/s, and HBM bandwidth as the better of a copy (read + write) and a read-only sum."""
    a = jax.random.normal(jax.random.key(0), (n, n), dtype)
    b = jax.random.normal(jax.random.key(1), (n, n), dtype)
    mm = jax.jit(lambda a, b: a @ b)
    t_mm = timeit(lambda _: mm(a, b), None, rounds=5)
    x = jnp.zeros(bw_bytes // jnp.dtype(dtype).itemsize, dtype)
    inc = jax.jit(lambda x: x + 1)
    total = jax.jit(lambda x: jnp.sum(x, dtype=jnp.float32))
    bw_copy = 2 * x.nbytes / timeit(lambda _: inc(x), None, rounds=5)
    bw_read = x.nbytes / timeit(lambda _: total(x), None, rounds=5)
    return 2 * n**3 / t_mm, max(bw_copy, bw_read)


def bench_ops(cfg, phase, dtype, peaks, rotate_bytes):
    rows = []
    key = jax.random.key(0)
    for i, op in enumerate(spec.op_specs(cfg, phase)):
        fn = OPS[op.fn]
        sub = jax.random.fold_in(key, i)
        probe, rotated = make_args(op, cfg, dtype, sub, 1)
        # size k by the op's whole traffic: the stacked outputs (e.g. QK^T scores) can dwarf the inputs
        k = spec.rotation_count(max(op.bytes, sum(v.nbytes for v, r in zip(probe, rotated) if r)), rotate_bytes)
        del probe
        try:
            vals, rotated = make_args(op, cfg, dtype, sub, k)
            shared = [v for v, r in zip(vals, rotated) if not r]
            xs = [v for v, r in zip(vals, rotated) if r]
            run = scanned(fn, op.inplace, rotated)
            if op.inplace:
                t = timeit(lambda c: run(c, shared[1:], xs), shared[0]) / k
            else:
                t = timeit(lambda _: run(shared, xs), None) / k
        except Exception as e:  # e.g. out of memory on a small device: report and keep going
            print(f"[{op.name} skipped: {type(e).__name__}: {str(e).splitlines()[0][:160]}]")
            continue
        row = spec.classify(op, t, *peaks)
        single = [v[0] if r else v for v, r in zip(vals, rotated)]
        row["xla_flops"], row["xla_bytes"] = cost_analysis(fn, single, op.inplace)
        rows.append(row)
        del vals, shared, xs, single
    return rows


def bench_attention(cfg, phase, dtype):
    """Unfused QK^T -> mask+softmax -> PV (what forward() runs) against a fused kernel:
    Pallas flash attention on TPU when it applies, else jax.nn.dot_product_attention."""
    B, T, S, N, K, H = phase.batch, phase.q_len, phase.kv_len, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    ks = jax.random.split(jax.random.key(2), 3)
    q = (jax.random.normal(ks[0], (B, N, T, H), jnp.float32) * H ** -0.5).astype(dtype)
    k = jax.random.normal(ks[1], (B, K, S, H), dtype)
    v = jax.random.normal(ks[2], (B, K, S, H), dtype)
    mask = jnp.asarray(spec.causal_mask(T, S))
    unfused = jax.jit(lambda q, k, v: pv(softmax_masked(qk(q, k), mask), v))

    fused, impl = None, None
    if jax.devices()[0].platform == "tpu" and N == K and T >= 128:
        try:
            from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
            fused = jax.jit(lambda q, k, v: flash_attention(q, k, v, causal=T > 1, sm_scale=1.0))
            fused(q, k, v).block_until_ready()
            impl = "pallas flash_attention (TPU)"
        except Exception as e:
            print(f"[pallas flash_attention unavailable: {type(e).__name__}: {str(e).splitlines()[0][:200]}]")
            fused = None
    if fused is None:
        def sdpa(q, k, v):
            o = jax.nn.dot_product_attention(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
                                             is_causal=T > 1, scale=1.0)
            return o.transpose(0, 2, 1, 3)
        fused, impl = jax.jit(sdpa), "jax.nn.dot_product_attention"

    err = float(jnp.abs(unfused(q, k, v).astype(jnp.float32) - fused(q, k, v).astype(jnp.float32)).max())
    return {"unfused_s": timeit(lambda _: unfused(q, k, v), None), "fused_s": timeit(lambda _: fused(q, k, v), None),
            "fused_impl": impl, "max_abs_err": err}


def bench_model(cfg, phase, dtype):
    params = random_params(cfg, dtype)
    consts = make_consts(cfg, phase, dtype)
    tokens = jax.random.randint(jax.random.key(3), (phase.batch, phase.q_len), 0, cfg.vocab)
    if phase.q_len > 1:
        f = jax.jit(lambda p, t: forward(p, t, consts)[0])
        t0 = time.perf_counter()
        f(params, tokens).block_until_ready()
        compile_s = time.perf_counter() - t0
        t = timeit(lambda _: f(params, tokens), None, rounds=5)
    else:   # decode: donate the cache so the slot write is in place, and feed the new cache back in
        f = jax.jit(lambda p, t, c: forward(p, t, consts, c), donate_argnums=2)
        cache = make_cache(cfg, phase, dtype)
        t0 = time.perf_counter()
        cache = jax.block_until_ready(f(params, tokens, cache))[1]
        compile_s = time.perf_counter() - t0
        t = timeit(lambda c: f(params, tokens, c)[1], cache, rounds=5)
    return {"jit_s": t, "compile_time_s": compile_s}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="colab", choices=spec.PRESETS)
    ap.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    ap.add_argument("--roofline", default="measured", choices=["measured", "spec"],
                    help="classify against measured ceilings or spec-sheet peaks")
    ap.add_argument("--rotate-mb", type=float, default=512, help="distinct input bytes cycled per op")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", message=".*donated buffers were not usable.*")

    dtype = DTYPES[args.dtype]
    preset = spec.PRESETS[args.preset]
    cfg = dataclasses.replace(preset.model, dtype_bytes=jnp.dtype(dtype).itemsize)
    dev = jax.devices()[0]
    name = dev.device_kind
    accel = dev.platform != "cpu"
    print(f"jax {jax.__version__} | device {name} ({dev.platform}) | dtype {args.dtype} | preset {args.preset}")

    measured = measure_ceilings(dtype, 8192 if accel else 1024, 2**30 if accel else 2**26)
    sheet = spec.spec_peaks(name)
    print(f"measured ceilings: {measured[0] / 1e12:.1f} TFLOP/s matmul, {measured[1] / 1e9:.0f} GB/s memory (best of copy, read)"
          + (f"  (spec sheet: {sheet[0] / 1e12:.0f} TFLOP/s, {sheet[1] / 1e9:.0f} GB/s)" if sheet else ""))
    peaks = sheet if args.roofline == "spec" and sheet else measured

    phases = {}
    for phase in ([preset.prefill, preset.decode] if args.phase == "both" else [getattr(preset, args.phase)]):
        rows = bench_ops(cfg, phase, dtype, peaks, args.rotate_mb * 2**20)
        spec.print_table(f"{phase.name}: B={phase.batch} T={phase.q_len} S={phase.kv_len}", rows, *peaks)
        attn = bench_attention(cfg, phase, dtype)
        print(f"attention: unfused {attn['unfused_s'] * 1e6:.1f} us vs fused {attn['fused_impl']} "
              f"{attn['fused_s'] * 1e6:.1f} us ({attn['unfused_s'] / attn['fused_s']:.2f}x), max |diff| {attn['max_abs_err']:.3g}")
        model = bench_model(cfg, phase, dtype)
        summary = spec.summarize(rows)
        print(f"whole model: jax.jit {model['jit_s'] * 1e3:.3f} ms (compile {model['compile_time_s']:.1f} s) "
              f"| sum of per-op jits {summary['sum_of_ops_s'] * 1e3:.3f} ms")
        phases[phase.name] = {"phase": dataclasses.asdict(phase), "rows": rows, "attention": attn,
                              "model": model, "summary": summary}

    if args.out:
        meta = {"framework": "jax", "version": jax.__version__, "device": name, "platform": dev.platform,
                "dtype": args.dtype, "measured_peaks": measured, "spec_peaks": sheet, "roofline": args.roofline,
                "peaks_used": peaks, **spec.preset_meta(args.preset)}
        spec.save_json(args.out, meta, phases)


if __name__ == "__main__":
    main()
