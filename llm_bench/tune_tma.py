"""Isolated SM120 TMA-prefill experiment, compared with current native linear.

    python -m llm_bench.tune_tma --out results/tma-prefill.json

The default runs the five actual M=2048 model projection shapes. It never
changes NativeRunner or the active matmul implementation. Every successful
candidate must pass independent FP32 checks and contain TMA instructions in
compiled PTX. Unsupported configurations are saved as failures, not timings.
"""

from __future__ import annotations

import argparse
import gc
import re
import statistics
import time
import traceback
from pathlib import Path

from .tune import DEFAULT_PROJECTIONS, _event_samples, shape_key


def parse_shape(value):
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,K,N") from exc
    if len(shape) != 3 or min(shape) <= 0 or shape[1] % 8 or shape[2] % 8:
        raise argparse.ArgumentTypeError("positive M,K,N required; K and N must be divisible by 8")
    return shape


def inspect_kernel(handle, config, shared_limit):
    from .triton_tma import estimated_shared_bytes

    ptx = handle.asm["ptx"]
    instructions = sorted(set(line.strip() for line in ptx.splitlines() if "cp.async.bulk.tensor" in line))
    shared = int(handle.metadata.shared)
    if not instructions:
        raise RuntimeError("descriptor kernel contains no cp.async.bulk.tensor PTX; TMA was not demonstrated")
    if shared > shared_limit:
        raise RuntimeError(f"compiled shared memory {shared} exceeds device limit {shared_limit}")
    return {
        "shared_memory_bytes": shared,
        "estimated_shared_memory_bytes": estimated_shared_bytes(config),
        "registers_per_thread": getattr(handle, "n_regs", None),
        "local_spills": getattr(handle, "n_spills", None),
        "tma_ptx_instructions": instructions,
        "mma_sync_instructions": len(re.findall(r"\bmma\.sync\b", ptx)),
        "wgmma_instructions": len(re.findall(r"\bwgmma\.", ptx)),
        "tcgen05_instructions": len(re.findall(r"\btcgen05\.", ptx)),
    }


def tune_shape(shape, args, flush, shared_limit):
    import torch
    from .bench import Captured, validate_tensor
    from .triton_matmul import linear, tuning_results
    from .triton_tma import TMA_CONFIGS, launch_tma, linear_tma

    m, k, n = shape
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randn((1, m, k), device="cuda", dtype=torch.bfloat16, generator=generator)
    w = (torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator) / k**0.5).to(torch.bfloat16)
    expected = x.float() @ w.float()
    graphs, candidates = {}, {}
    for name in ["linear", *TMA_CONFIGS]:
        started = time.perf_counter()
        try:
            fn = (lambda: linear(x, w)) if name == "linear" else (
                lambda config=name: linear_tma(x, w, config))
            if name == "linear":
                first = fn()
                resources = None
            else:
                first, handle = launch_tma(x, w, name)
                resources = inspect_kernel(handle, name, shared_limit)
            torch.cuda.synchronize()
            accuracy = validate_tensor(first, expected, f"{name}[{shape_key(shape)}]")
            graph = Captured(fn)
            for _ in range(args.warmup):
                graph()
            torch.cuda.synchronize()
            validate_tensor(graph.output, expected, f"captured/{name}[{shape_key(shape)}]")
            graphs[name] = graph
            candidates[name] = {
                "status": "ok", "accuracy": accuracy, "resources": resources,
                "setup_compile_autotune_capture_s": time.perf_counter() - started,
                "raw_samples_ms": [], "round_medians_ms": [],
            }
        except Exception as exc:
            candidates[name] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"candidate {name} failed: {type(exc).__name__}: {str(exc)[:240]}", flush=True)
            if name == "linear":
                raise RuntimeError("existing native linear baseline failed") from exc
            # A device error poisons the CUDA context and must stop the run.
            # Pure compile/configuration errors leave synchronization usable.
            torch.cuda.synchronize()
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
        candidates[name]["tflops"] = 2 * m * n * k / (candidates[name]["median_ms"] * 1e9)
        validate_tensor(graphs[name].output, expected, f"timed/{name}[{shape_key(shape)}]")
    tma_names = [name for name in active if name != "linear"]
    best = min(tma_names, key=lambda name: candidates[name]["median_ms"]) if tma_names else None
    baseline = candidates["linear"]["median_ms"]
    best_time = candidates[best]["median_ms"] if best is not None else None
    result = {
        "shape": dict(M=m, K=k, N=n),
        "candidates": candidates,
        "round_candidate_orders": orders,
        "baseline_median_ms": baseline,
        "baseline_tile_tuning": tuning_results(),
        "best_tma_config": best,
        "best_tma_median_ms": best_time,
        "best_tma_speedup": baseline / best_time if best_time is not None else None,
        "recommended_backend": "tma" if best_time is not None and best_time < baseline * 0.99 else "linear",
        "recommendation_gate": "TMA must beat baseline by >1%; whole-model confirmation still required",
    }
    graphs.clear()
    del graph, first, expected, x, w
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", dest="shapes", action="append", type=parse_shape,
                        help="repeat M,K,N; defaults to five actual M=2048 projections")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cache-mode", choices=("cold", "hot"), default="cold")
    parser.add_argument("--cache-flush-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if min(args.rounds, args.calls, args.warmup, args.cache_flush_mib) < 1:
        parser.error("rounds, calls, warmup and cache-flush-mib must be positive")
    shapes = args.shapes or [(2048, k, n) for k, n in DEFAULT_PROJECTIONS]
    from .bench import write_json

    result = {
        "status": "running", "kind": "optional_tma_prefill_experiment",
        "adopted_by_default_runner": False,
        "method": {
            "timer": "CUDA events around full projection graph replay",
            "rounds": args.rounds, "calls_per_round": args.calls, "warmup_replays": args.warmup,
            "cache_mode": args.cache_mode,
            "cache_flush_mib": args.cache_flush_mib if args.cache_mode == "cold" else 0,
            "cache_flush_in_timing": False, "reference_dtype": "float32", "tf32": False,
            "storage_dtype": "bfloat16", "weight_layout": "row-major K,N; no transpose packing",
            "seed": args.seed,
        },
        "shapes": {},
    }
    write_json(args.out, result)
    started = time.perf_counter()
    try:
        import torch
        from triton.runtime import driver
        from .bench import environment
        from .triton_tma import TMA_CONFIGS

        if not torch.cuda.is_available():
            raise RuntimeError("TMA tuning requires a CUDA GPU")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        result["environment"] = environment()
        result["configurations"] = TMA_CONFIGS
        shared_limit = int(driver.active.utils.get_device_properties(0)["max_shared_mem"])
        result["shared_memory_limit_bytes"] = shared_limit
        with torch.inference_mode():
            flush = (torch.empty(args.cache_flush_mib * 2**20, device="cuda", dtype=torch.uint8)
                     if args.cache_mode == "cold" else None)
            for shape in shapes:
                key = shape_key(shape)
                print(f"TMA experiment {key}", flush=True)
                measured = tune_shape(shape, args, flush, shared_limit)
                result["shapes"][key] = measured
                result["elapsed_s"] = time.perf_counter() - started
                write_json(args.out, result)
                print(f"{key}: best={measured['best_tma_config']} speedup={measured['best_tma_speedup']}", flush=True)
        successful = [row for row in result["shapes"].values() if row["best_tma_config"] is not None]
        result["status"] = "ok" if len(successful) == len(shapes) else "partial" if successful else "unsupported"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    result["elapsed_s"] = time.perf_counter() - started
    write_json(args.out, result)
    print(f"wrote {args.out}; status={result['status']}", flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
