"""Benchmark ReLU(XW + b) schedules against NumPy.  Run with: uv run -m toy.bench

Columns are wall time per call and achieved GFLOP/s (2*M*N*K flops for the
matmul).  Each schedule is the same traced graph lowered with a different
``tile`` setting; only the generated loop structure changes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import toy
from toy import jit
from toy.passes import TILE_CANDIDATES


def relu_linear(x, w, b):
    return toy.relu(x @ w + b)


def bench(fn, args, repeats: int = 5) -> float:
    fn(*args)  # warm-up (and, for jit, compile)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    for n in args.sizes:
        M = N = K = n
        x = rng.standard_normal((M, K), dtype=np.float32)
        w = rng.standard_normal((K, N), dtype=np.float32)
        b = rng.standard_normal((N,), dtype=np.float32)
        flops = 2.0 * M * N * K
        ref = np.maximum(x @ w + b, 0)

        schedules: list[tuple[str, object]] = [
            ("numpy (BLAS sgemm)", relu_linear),
            ("naive i,j,k (fused)", jit(relu_linear, tile=None)),
            (f"reorder i,k,j only, tile=({M},{N},{K})", jit(relu_linear, tile=(M, N, K))),
        ]
        schedules += [(f"tile={t}", jit(relu_linear, tile=t)) for t in TILE_CANDIDATES]
        auto = jit(relu_linear, tile="auto")
        schedules.append(("auto (best of candidates)", auto))

        print(f"\n== M=N=K={n}  ({flops / 1e9:.2f} GFLOP per call) ==")
        print(f"{'schedule':<42} {'ms':>9} {'GFLOP/s':>9}")
        for name, fn in schedules:
            t = bench(fn, (x, w, b), args.repeats)
            out = fn(x, w, b)
            ok = np.allclose(out, ref, rtol=1e-3, atol=1e-3)
            print(f"{name:<42} {t * 1e3:>9.2f} {flops / t / 1e9:>9.1f}{'' if ok else '  MISMATCH'}")
        key = next(iter(auto.tuning))
        best = min(auto.tuning[key], key=auto.tuning[key].get)
        print(f"auto picked tile={best}")


if __name__ == "__main__":
    main()
