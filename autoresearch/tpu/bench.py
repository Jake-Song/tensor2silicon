"""Fixed harness for the TPU autoresearch loop. DO NOT EDIT (the agent edits kernel.py only).

Checks kernel.relu_linear against an fp32 reference, requires a Pallas (Mosaic) kernel
instead of an XLA dot, times it against XLA, and prints a grep-able summary block.
Usage: python bench.py
"""
import re
import statistics
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

SHAPES = [(4096, 4096, 4096), (1024, 2048, 512)]    # (M, K, N); the first one is timed
ATOL = RTOL = 2e-2                                  # bf16 output tolerance
FORBIDDEN = re.compile(r"stablehlo\.(dot_general|dot|convolution)\b")
PEAK_TFLOPS = {"TPU v6": 918.0, "TPU v5 lite": 197.0, "TPU v5e": 197.0}   # dense bf16


def make_inputs(M, K, N, seed):
    kx, kw, kb = jax.random.split(jax.random.key(seed), 3)
    x = jax.random.normal(kx, (M, K), jnp.float32).astype(jnp.bfloat16)
    w = (jax.random.normal(kw, (K, N), jnp.float32) / np.sqrt(K)).astype(jnp.bfloat16)   # keeps |y| ~ 1
    b = jax.random.normal(kb, (N,), jnp.float32).astype(jnp.bfloat16)
    return x, w, b


@jax.jit
def reference(x, w, b):
    f32 = jnp.float32
    y = jnp.dot(x.astype(f32), w.astype(f32), precision=jax.lax.Precision.HIGHEST)
    return jax.nn.relu(y + b.astype(f32))


@jax.jit
def xla_baseline(x, w, b):
    return jax.nn.relu(x @ w + b)


def check(fn, x, w, b):
    """Returns max_abs_err, or raises AssertionError with the reason."""
    y = fn(x, w, b)
    assert y.shape == (x.shape[0], w.shape[1]), f"wrong shape {y.shape}"
    assert y.dtype == jnp.bfloat16, f"wrong dtype {y.dtype}"
    y = np.asarray(y.astype(jnp.float32))
    assert np.isfinite(y).all(), "NaN/Inf in output"
    ref = np.asarray(reference(x, w, b))
    err = float(np.abs(y - ref).max())
    assert np.allclose(y, ref, atol=ATOL, rtol=RTOL), f"mismatch vs reference (max_abs_err={err:.4g})"
    return err


def bench(fn, *args, n=50, rounds=7):
    """Median ms per call. Dispatches n calls back to back so host overhead overlaps device work."""
    fn(*args).block_until_ready()                      # compile
    for _ in range(3):
        fn(*args).block_until_ready()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n):
            y = fn(*args)
        y.block_until_ready()
        ts.append((time.perf_counter() - t0) / n)
    return statistics.median(ts) * 1e3


def run():
    from kernel import relu_linear

    r = {"status": "error"}
    fn = jax.jit(relu_linear)

    # 1. correctness on every shape
    errs = []
    for i, (M, K, N) in enumerate(SHAPES):
        try:
            errs.append(check(fn, *make_inputs(M, K, N, seed=i)))
        except AssertionError as e:
            print(f"[check] shape M={M} K={K} N={N}: {e}")
            r.update(status="incorrect", correct=False)
            return r
    r.update(correct=True, max_abs_err=max(errs))

    # 2. guard: must lower to a Mosaic custom call, with no XLA dot/convolution around it
    M, K, N = SHAPES[0]
    x, w, b = make_inputs(M, K, N, seed=100)
    hlo = fn.lower(x, w, b).as_text()
    n_custom = hlo.count("tpu_custom_call")
    bad = sorted(set(FORBIDDEN.findall(hlo)))
    print(f"[guard] tpu_custom_call x{n_custom}, forbidden ops: {bad}")
    if n_custom == 0 or bad:
        print("[guard] the matmul must run inside a Pallas TPU kernel (no XLA dot, no interpret mode)")
        r.update(status="forbidden")
        return r

    # 3. timing
    t = bench(fn, x, w, b)
    tb = bench(xla_baseline, x, w, b)

    # 4. re-check after timing with fresh data (defeats output caching)
    try:
        check(fn, *make_inputs(M, K, N, seed=200))
    except AssertionError as e:
        print(f"[check] after timing (fresh data): {e}")
        r.update(status="incorrect", correct=False)
        return r

    tflops = 2 * M * N * K / (t * 1e-3) / 1e12
    kind = jax.devices()[0].device_kind
    peak = next((v for k, v in PEAK_TFLOPS.items() if kind.startswith(k)), float("nan"))
    r.update(status="ok", time_ms=t, tflops=tflops, pct_peak=100 * tflops / peak, baseline_ms=tb, speedup=tb / t)
    return r


def main():
    d = jax.devices()[0]
    print(f"device: {d.device_kind} ({d.platform}) | jax {jax.__version__}")
    try:
        r = run()
    except Exception:
        traceback.print_exc()
        r = {"status": "error"}
    nan = float("nan")
    print("---")
    print(f"status:      {r['status']}")
    print(f"correct:     {r.get('correct', False)}")
    print(f"max_abs_err: {r.get('max_abs_err', nan):.4g}")
    print(f"time_ms:     {r.get('time_ms', nan):.4f}")
    print(f"tflops:      {r.get('tflops', nan):.1f}")
    print(f"pct_peak:    {r.get('pct_peak', nan):.1f}")
    print(f"baseline_ms: {r.get('baseline_ms', nan):.4f}")
    print(f"speedup:     {r.get('speedup', nan):.3f}")
    sys.exit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
