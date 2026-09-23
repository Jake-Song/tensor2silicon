"""Fixed harness for the GPU autoresearch loop. DO NOT EDIT (the agent edits kernel.py only).

Checks kernel.relu_linear against an fp32 reference, rejects vendor GEMMs, times it
against cuBLAS, and prints a grep-able summary block. Usage: python bench.py
"""
import math
import re
import statistics
import sys
import traceback

import torch
import triton

SHAPES = [(4096, 4096, 4096), (1024, 2048, 512)]    # (M, K, N); the first one is timed
ATOL = RTOL = 2e-2                                  # bf16 output tolerance
FORBIDDEN = re.compile(r"gemm|cutlass|cublas|xmma|cudnn|sm80_|sm90_", re.IGNORECASE)
TIMING_ALLOCS = 5                                   # median over this many input allocations
PEAK_TFLOPS = {"A100": 312.0, "H100": 989.0, "L4": 121.0}   # dense bf16


def make_inputs(M, K, N, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(K, N, device="cuda", generator=g) / math.sqrt(K)).to(torch.bfloat16)   # keeps |y| ~ 1
    b = torch.randn(N, device="cuda", generator=g).to(torch.bfloat16)
    return x, w, b


def reference(x, w, b):
    return torch.relu(x.float() @ w.float() + b.float())


def check(fn, x, w, b):
    """Returns max_abs_err, or raises AssertionError with the reason."""
    y = fn(x, w, b)
    torch.cuda.synchronize()
    assert y.shape == (x.shape[0], w.shape[1]), f"wrong shape {tuple(y.shape)}"
    assert y.dtype == torch.bfloat16, f"wrong dtype {y.dtype}"
    y = y.float()
    assert torch.isfinite(y).all(), "NaN/Inf in output"
    ref = reference(x, w, b)
    err = (y - ref).abs().max().item()
    assert torch.allclose(y, ref, atol=ATOL, rtol=RTOL), f"mismatch vs reference (max_abs_err={err:.4g})"
    return err


def time_over_allocs(fn, M, K, N, n=TIMING_ALLOCS):
    """Median over n fresh input allocations. A kernel's speed can depend on where its
    buffers land in memory, so one allocation per process gives a lucky or unlucky draw."""
    ts, hold = [], []
    for i in range(n):
        hold.append(torch.empty((i + 1) * 3 * 2**20, dtype=torch.uint8, device="cuda"))   # shift the next allocation
        x, w, b = make_inputs(M, K, N, seed=1000 + i)
        ts.append(triton.testing.do_bench(lambda: fn(x, w, b), warmup=50, rep=300, return_mode="median"))
    return statistics.median(ts), ts


def cuda_kernel_names(fn, *args):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return sorted({e.key for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA})


def run():
    from kernel import relu_linear

    r = {"status": "error"}
    # 1. correctness on every shape
    errs = []
    for i, (M, K, N) in enumerate(SHAPES):
        try:
            errs.append(check(relu_linear, *make_inputs(M, K, N, seed=i)))
        except AssertionError as e:
            print(f"[check] shape M={M} K={K} N={N}: {e}")
            r.update(status="incorrect", correct=False)
            return r
    r.update(correct=True, max_abs_err=max(errs))

    # 2. guard: the result must come from our own kernel, not a vendor GEMM
    M, K, N = SHAPES[0]
    x, w, b = make_inputs(M, K, N, seed=100)
    names = cuda_kernel_names(relu_linear, x, w, b)
    print("[guard] CUDA kernels launched:", names)
    bad = [n for n in names if FORBIDDEN.search(n)]
    if bad:
        print("[guard] vendor GEMM kernels are not allowed:", bad)
        r.update(status="forbidden")
        return r

    # 3. timing (median ms, L2 flushed between reps by do_bench)
    t, ts = time_over_allocs(relu_linear, M, K, N)
    tb, tbs = time_over_allocs(lambda x, w, b: torch.relu(torch.addmm(b, x, w)), M, K, N)
    print("[timing] candidate per allocation:", " ".join(f"{v:.4f}" for v in ts))
    print("[timing] baseline  per allocation:", " ".join(f"{v:.4f}" for v in tbs))

    # 4. re-check after timing with new values in the SAME buffers (defeats output caching)
    x2, w2, b2 = make_inputs(M, K, N, seed=200)
    x.copy_(x2); w.copy_(w2); b.copy_(b2)
    try:
        check(relu_linear, x, w, b)
    except AssertionError as e:
        print(f"[check] after timing (fresh data, same buffers): {e}")
        r.update(status="incorrect", correct=False)
        return r

    tflops = 2 * M * N * K / (t * 1e-3) / 1e12
    name = torch.cuda.get_device_name()
    peak = next((v for k, v in PEAK_TFLOPS.items() if k in name), float("nan"))
    r.update(status="ok", time_ms=t, tflops=tflops, pct_peak=100 * tflops / peak, baseline_ms=tb, speedup=tb / t)
    return r


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name} | SMs {p.multi_processor_count} | torch {torch.__version__} | triton {triton.__version__}")
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
