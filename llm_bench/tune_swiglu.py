"""Compare complete native linear+SwiGLU against the optional fused epilogue.

    python -m llm_bench.tune_swiglu --out results/swiglu-fusion.json

The default is M=2048, K=2048, F=5632. Timing includes all computation and output
stores for each path. No active model implementation is changed by this command.
"""

from __future__ import annotations

import argparse
import gc
import re
import statistics
import time
import traceback
from pathlib import Path

from .tune import _event_samples


def _pairwise_check(actual, baseline, label):
    import torch
    from .bench import validate_tensor

    check = validate_tensor(actual, baseline, label)
    check["bitwise_equal_fraction"] = (actual == baseline).float().mean().item()
    check["allclose_atol"] = 0.003
    check["allclose_rtol"] = 0.015
    check["allclose_passed"] = bool(torch.allclose(actual.float(), baseline.float(), atol=0.003, rtol=0.015))
    if not check["allclose_passed"]:
        raise AssertionError(f"{label}: failed declared BF16-path closeness atol=.003 rtol=.015: {check}")
    return check


def run_experiment(args):
    import torch
    import torch.nn.functional as functional
    from triton.runtime import driver
    from .bench import Captured, environment, validate_tensor
    from .triton_matmul import linear, tuning_results
    from .triton_ops import swiglu
    from .triton_swiglu import SWIGLU_CONFIGS, launch_swiglu, swiglu_linear

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    m, k, f = args.m, args.k, args.f
    shared_limit = int(driver.active.utils.get_device_properties(0)["max_shared_mem"])
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randn((1, m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    w = (torch.randn((k, 2 * f), device="cuda", dtype=torch.float32, generator=generator) / k**0.5).to(torch.bfloat16)
    reference_packed = x.float() @ w.float()
    reference_gate, reference_up = reference_packed.chunk(2, dim=-1)
    fp32_reference = functional.silu(reference_gate) * reference_up
    rounded_gate, rounded_up = reference_packed.to(torch.bfloat16).float().chunk(2, dim=-1)
    rounded_reference = (functional.silu(rounded_gate) * rounded_up).to(torch.bfloat16)
    del reference_packed, reference_gate, reference_up, rounded_gate, rounded_up
    baseline_output = swiglu(linear(x, w))
    validate_tensor(baseline_output, fp32_reference, "baseline/fp32")
    graphs, candidates = {}, {}
    for name in ["linear_plus_swiglu", *SWIGLU_CONFIGS]:
        started = time.perf_counter()
        try:
            fn = (lambda: swiglu(linear(x, w))) if name == "linear_plus_swiglu" else (
                lambda config=name: swiglu_linear(x, w, config))
            if name == "linear_plus_swiglu":
                actual = fn()
                resources = None
            else:
                actual, handle = launch_swiglu(x, w, name)
                shared = int(handle.metadata.shared)
                if shared > shared_limit:
                    raise RuntimeError(f"compiled shared memory {shared} exceeds {shared_limit}")
                resources = {
                    "shared_memory_bytes": shared,
                    "registers_per_thread": getattr(handle, "n_regs", None),
                    "local_spills": getattr(handle, "n_spills", None),
                    "mma_sync_instructions": len(re.findall(r"\bmma\.sync\b", handle.asm["ptx"])),
                    "bf16_conversion_instructions": [
                        line.strip() for line in handle.asm["ptx"].splitlines()
                        if re.search(r"cvt\.[^;]*bf16", line)
                    ],
                }
            torch.cuda.synchronize()
            accuracy = validate_tensor(actual, fp32_reference, f"{name}/fp32")
            precision = _pairwise_check(actual, baseline_output, f"{name}/existing_bf16")
            rounded = _pairwise_check(actual, rounded_reference, f"{name}/rounded_fp32_reference")
            graph = Captured(fn)
            for _ in range(args.warmup):
                graph()
            torch.cuda.synchronize()
            _pairwise_check(graph.output, baseline_output, f"captured/{name}")
            graphs[name] = graph
            candidates[name] = {
                "status": "ok", "fp32_accuracy": accuracy,
                "existing_bf16_closeness": precision, "rounded_reference_closeness": rounded,
                "resources": resources, "setup_compile_capture_s": time.perf_counter() - started,
                "raw_samples_ms": [], "round_medians_ms": [],
            }
        except Exception as exc:
            candidates[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()}
            print(f"candidate {name} failed: {type(exc).__name__}: {str(exc)[:220]}", flush=True)
            if name == "linear_plus_swiglu":
                raise
            torch.cuda.synchronize()
    flush = (torch.empty(args.cache_flush_mib * 2**20, device="cuda", dtype=torch.uint8)
             if args.cache_mode == "cold" else None)
    active = list(graphs)
    orders = []
    for round_index in range(args.rounds):
        offset = round_index % len(active)
        order = active[offset:] + active[:offset]
        orders.append(order)
        for name in order:
            graph = graphs[name]
            for _ in range(args.warmup):
                graph()
            torch.cuda.synchronize()
            samples = _event_samples(graph, args.calls, flush)
            if any(sample <= 0 for sample in samples):
                raise RuntimeError("CUDA timing returned a nonpositive observation")
            candidates[name]["raw_samples_ms"].append(samples)
            candidates[name]["round_medians_ms"].append(statistics.median(samples))
    for name in active:
        candidates[name]["median_ms"] = statistics.median(candidates[name]["round_medians_ms"])
        _pairwise_check(graphs[name].output, baseline_output, f"timed/{name}")
    fused = [name for name in active if name != "linear_plus_swiglu"]
    best = min(fused, key=lambda name: candidates[name]["median_ms"]) if fused else None
    base_ms = candidates["linear_plus_swiglu"]["median_ms"]
    best_ms = candidates[best]["median_ms"] if best is not None else None
    result = {
        "status": "ok" if fused else "failed",
        "kind": "optional_fused_gate_up_swiglu_experiment",
        "environment": environment(), "shape": dict(M=m, K=k, F=f),
        "configurations": SWIGLU_CONFIGS, "candidates": candidates,
        "round_candidate_orders": orders, "baseline_tile_tuning": tuning_results(),
        "baseline_median_ms": base_ms, "best_fused_config": best,
        "best_fused_median_ms": best_ms,
        "best_fused_speedup": base_ms / best_ms if best_ms is not None else None,
        "recommended_backend": "fused" if best_ms is not None and best_ms < base_ms * 0.99 else "linear_plus_swiglu",
        "recommendation_gate": ">1% improvement; whole-model correctness/timing confirmation still required",
        "adopted_by_default_runner": False,
        "saved_intermediate_io_bytes_per_call": 4 * m * (2 * f),
        "saved_intermediate_io_scope": "logical BF16 intermediate write plus read; not measured DRAM traffic",
    }
    graphs.clear()
    del graph, actual, baseline_output, fp32_reference, rounded_reference, x, w
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--f", type=int, default=5632)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cache-mode", choices=("cold", "hot"), default="cold")
    parser.add_argument("--cache-flush-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if min(args.m, args.k, args.f, args.rounds, args.calls, args.warmup, args.cache_flush_mib) < 1:
        parser.error("dimensions, rounds, calls, warmup and cache-flush-mib must be positive")
    from .bench import write_json

    method = {
        "timer": "CUDA events around full linear+SwiGLU or fused graph replay",
        "rounds": args.rounds, "calls_per_round": args.calls, "warmup_replays": args.warmup,
        "cache_mode": args.cache_mode,
        "cache_flush_mib": args.cache_flush_mib if args.cache_mode == "cold" else 0,
        "cache_flush_in_timing": False, "same_quantized_inputs": True,
        "projection_rounding": "FP32 dot accumulation -> BF16 -> FP32 SiLU/multiply -> BF16 output",
        "fp32_reference_tf32": False, "seed": args.seed,
    }
    result = {"status": "running", "method": method}
    write_json(args.out, result)
    started = time.perf_counter()
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("fused SwiGLU experiment requires CUDA")
        with torch.inference_mode():
            result = run_experiment(args)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
        traceback.print_exc()
    result["method"] = method
    result["elapsed_s"] = time.perf_counter() - started
    write_json(args.out, result)
    print(f"wrote {args.out}; status={result['status']}", flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
