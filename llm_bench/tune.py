"""Measure complete native decode projections and select split-K factors.

    python -m llm_bench.tune --out results/split-k-tuning.json

Each candidate captures the entire ``linear`` operation, including the FP32
split reduction when present. Compilation and autotuning finish before timing.
The default clears L2 before each measured call because full-model decode streams
weights that exceed L2 capacity; cache clearing itself is outside the timer.
Run this process by itself on the GPU, then pass the saved JSON to the main
benchmark with ``--tuning-config results/split-k-tuning.json``.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import traceback
from pathlib import Path


DEFAULT_PROJECTIONS = ((2048, 6144), (2048, 2048), (2048, 11264), (5632, 2048), (2048, 32000))
SPLIT_CANDIDATES = (1, 2, 4, 8, 16)


def shape_key(shape):
    return ",".join(map(str, shape))


def parse_shape(value):
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,K,N") from exc
    if len(shape) != 3 or any(dim <= 0 for dim in shape) or shape[0] > 16:
        raise argparse.ArgumentTypeError("shape must be M,K,N with 1 <= M <= 16 and positive K,N")
    return shape


def _event_samples(graph, calls, flush):
    """Measure graph replay only; cold-cache preparation stays before start."""
    import torch

    pairs = []
    for _ in range(calls):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


def tune_shape(shape, args, flush):
    import torch
    from .bench import Captured, validate_tensor
    from .triton_matmul import linear

    m, k, n = shape
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randn((m, 1, k), device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    w = (torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator) / k**0.5).to(torch.bfloat16)
    # The reference uses exactly the BF16-quantized input values promoted to FP32.
    expected = x.float() @ w.float()
    graphs, candidates = {}, {}
    for split in args.splits:
        started = time.perf_counter()
        fn = lambda split=split: linear(x, w, split_k=split)
        first = fn()  # JIT and tile autotuning are outside the timing graph.
        torch.cuda.synchronize()
        accuracy = validate_tensor(first, expected, f"linear[{shape_key(shape)}]/split{split}")
        graph = Captured(fn)
        for _ in range(args.warmup):
            graph()
        torch.cuda.synchronize()
        validate_tensor(graph.output, expected, f"captured[{shape_key(shape)}]/split{split}")
        graphs[split] = graph
        candidates[str(split)] = {
            "split_k": split,
            "accuracy": accuracy,
            "compile_autotune_capture_warmup_s": time.perf_counter() - started,
            "round_medians_ms": [],
            "raw_samples_ms": [],
        }
    round_orders = []
    for round_index in range(args.rounds):
        offset = round_index % len(args.splits)
        order = args.splits[offset:] + args.splits[:offset]
        round_orders.append(order)
        for split in order:
            graph = graphs[split]
            for _ in range(args.warmup):
                graph()
            torch.cuda.synchronize()
            samples = _event_samples(graph, args.calls, flush)
            if any(sample <= 0 for sample in samples):
                raise RuntimeError("CUDA event timing returned a nonpositive observation")
            candidates[str(split)]["raw_samples_ms"].append(samples)
            candidates[str(split)]["round_medians_ms"].append(statistics.median(samples))
    for split, candidate in candidates.items():
        candidate["median_ms"] = statistics.median(candidate["round_medians_ms"])
        validate_tensor(graphs[int(split)].output, expected, f"timed[{shape_key(shape)}]/split{split}")
    chosen = min(args.splits, key=lambda split: (candidates[str(split)]["median_ms"], split))
    result = {
        "shape": {"M": m, "K": k, "N": n},
        "selected_split_k": chosen,
        "selected_median_ms": candidates[str(chosen)]["median_ms"],
        "selection": "lowest median of round medians; exact ties choose fewer splits",
        "round_candidate_orders": round_orders,
        "candidates": candidates,
    }
    # Releasing all graphs between shapes bounds retained graph-pool memory.
    graphs.clear()
    del graph, first, expected, x, w
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shape", dest="shapes", action="append", type=parse_shape,
                        help="repeat M,K,N; defaults to all five model projections at M=1 and M=8")
    parser.add_argument("--splits", type=int, nargs="+", choices=SPLIT_CANDIDATES,
                        default=list(SPLIT_CANDIDATES))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--cache-mode", choices=("cold", "hot"), default="cold")
    parser.add_argument("--cache-flush-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.calls < 1 or args.warmup < 1 or args.cache_flush_mib < 1:
        parser.error("rounds, calls, warmup and cache-flush-mib must be positive")
    args.splits = list(dict.fromkeys(args.splits))
    shapes = args.shapes or [(m, k, n) for m in (1, 8) for k, n in DEFAULT_PROJECTIONS]

    # Accelerator imports follow argument parsing, so --help works on CPU hosts.
    import torch
    from .bench import environment, write_json
    from .triton_matmul import get_split_k_overrides, set_split_k_overrides, tuning_results

    if not torch.cuda.is_available():
        raise RuntimeError("split-K tuning requires a CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    set_split_k_overrides({})
    result = {
        "status": "running",
        "kind": "native_triton_split_k_tuning",
        "environment": environment(),
        "method": {
            "timer": "CUDA events around full linear CUDA graph replay",
            "rounds": args.rounds,
            "calls_per_round": args.calls,
            "warmup_replays": args.warmup,
            "cache_mode": args.cache_mode,
            "cache_flush_mib": args.cache_flush_mib if args.cache_mode == "cold" else 0,
            "cache_flush_in_timing": False,
            "same_quantized_inputs_for_all_candidates": True,
            "reference_dtype": "float32",
            "tf32": False,
            "seed": args.seed,
            "splits": args.splits,
            "gpu_execution": "sequential; no other benchmark may share this GPU",
        },
        "split_k_overrides": get_split_k_overrides(),
        "shapes": {},
    }
    write_json(args.out, result)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            flush = (torch.empty(args.cache_flush_mib * 2**20, device="cuda", dtype=torch.uint8)
                     if args.cache_mode == "cold" else None)
            for shape in shapes:
                print(f"tuning {shape_key(shape)}", flush=True)
                measured = tune_shape(shape, args, flush)
                key = shape_key(shape)
                result["shapes"][key] = measured
                result["split_k_overrides"][key] = measured["selected_split_k"]
                result["tile_tuning"] = tuning_results()
                result["elapsed_s"] = time.perf_counter() - started
                write_json(args.out, result)
                print(f"selected {key}: split={measured['selected_split_k']} "
                      f"{measured['selected_median_ms'] * 1000:.3f} us", flush=True)
        result["status"] = "ok"
        result["load_instruction"] = f"python -m llm_bench ... --tuning-config {args.out}"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    result["elapsed_s"] = time.perf_counter() - started
    write_json(args.out, result)
    print(f"wrote {args.out}; status={result['status']}", flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
