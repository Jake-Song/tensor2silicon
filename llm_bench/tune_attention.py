"""Select native decode attention split-KV factors on the benchmark GPU.

    python -m llm_bench.tune_attention --out results/attention-tuning.json

Measures complete captured attention, including the partial-output reduction.
Compilation/autotuning/warmup and optional L2 preparation are outside timing.
One choice per (B,NQ,NK,S,H) minimizes mean latency across that shape's measured
positions. The B=1 default checks both ends of the generation context range.
Run sequentially with other GPU jobs; this command never allocates a runtime.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import traceback
from pathlib import Path

from .tune import _event_samples, shape_key


SPLIT_CANDIDATES = (1, 2, 4, 8, 9, 16, 32)
DEFAULT_CASES = (
    (1, 16, 16, 2176, 128, 2048),
    (1, 16, 16, 2176, 128, 2174),
    (8, 16, 16, 2048, 128, 2047),
)


def parse_case(value):
    try:
        case = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be B,NQ,NK,S,H,POSITION") from exc
    if (len(case) != 6 or any(dim <= 0 for dim in case[:5])
            or case[1] % case[2] or case[4] > 256 or not 0 <= case[5] < case[3]):
        raise argparse.ArgumentTypeError(
            "case requires positive B,NQ,NK,S,H, NQ % NK = 0, H <= 256, and 0 <= POSITION < S")
    return case


def _reference(q, k, v, slot):
    """Independent unfused FP32 QK -> softmax -> PV from BF16 inputs."""
    import torch

    groups = q.shape[1] // k.shape[1]
    qf = q.float()
    kf = k.float().repeat_interleave(groups, dim=1) if groups != 1 else k.float()
    vf = v.float().repeat_interleave(groups, dim=1) if groups != 1 else v.float()
    scores = (qf @ kf.transpose(-1, -2)) * q.shape[-1]**-0.5
    forbidden = torch.arange(k.shape[2], device=q.device) > slot
    probability = torch.softmax(scores.masked_fill(forbidden, float("-inf")), dim=-1)
    out = probability @ vf
    return out.transpose(1, 2).reshape(q.shape[0], 1, -1)


def tune_case(case, args, flush):
    import torch
    from .bench import Captured, validate_tensor
    from .triton_attention import attention

    b, nq, nk, s, h, slot = case
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q = torch.randn((b, nq, 1, h), device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn((b, nk, s, h), device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn((b, nk, s, h), device="cuda", dtype=torch.bfloat16, generator=generator)
    position = torch.tensor(slot, device="cuda", dtype=torch.int64)
    expected = _reference(q, k, v, slot)
    graphs, candidates = {}, {}
    for split in args.splits:
        started = time.perf_counter()
        fn = lambda split=split: attention(q, k, v, position, split_kv=split)
        first = fn()
        torch.cuda.synchronize()
        accuracy = validate_tensor(first, expected, f"attention[{shape_key(case)}]/split{split}")
        graph = Captured(fn)
        for _ in range(args.warmup):
            graph()
        torch.cuda.synchronize()
        validate_tensor(graph.output, expected, f"captured[{shape_key(case)}]/split{split}")
        graphs[split] = graph
        candidates[str(split)] = {
            "split_kv": split,
            "accuracy": accuracy,
            "compile_autotune_capture_warmup_s": time.perf_counter() - started,
            "round_medians_ms": [],
            "raw_samples_ms": [],
        }
    orders = []
    for round_index in range(args.rounds):
        offset = round_index % len(args.splits)
        order = args.splits[offset:] + args.splits[:offset]
        orders.append(order)
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
        validate_tensor(graphs[int(split)].output, expected, f"timed[{shape_key(case)}]/split{split}")
    best = min(args.splits, key=lambda split: (candidates[str(split)]["median_ms"], split))
    result = {
        "shape": dict(B=b, NQ=nq, NK=nk, S=s, H=h),
        "position": slot,
        "shape_key": shape_key(case[:5]),
        "best_split_at_position": best,
        "best_median_ms": candidates[str(best)]["median_ms"],
        "round_candidate_orders": orders,
        "candidates": candidates,
    }
    graphs.clear()
    del graph, first, expected, q, k, v, position
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate_choices(cases, splits):
    grouped = {}
    for case in cases.values():
        grouped.setdefault(case["shape_key"], []).append(case)
    overrides, summaries = {}, {}
    for key, rows in grouped.items():
        scores = {str(split): statistics.mean(row["candidates"][str(split)]["median_ms"] for row in rows)
                  for split in splits}
        best = min(splits, key=lambda split: (scores[str(split)], split))
        overrides[key] = best
        summaries[key] = {
            "positions": [row["position"] for row in rows],
            "mean_position_median_ms_by_split": scores,
            "selected_split_kv": best,
            "selection": "lowest arithmetic mean of position medians; exact ties choose fewer splits",
        }
    return overrides, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", dest="cases", action="append", type=parse_case,
                        help="repeat B,NQ,NK,S,H,POSITION; defaults to actual B=1/B=8 decode shapes")
    parser.add_argument("--splits", nargs="+", type=int, choices=SPLIT_CANDIDATES,
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
    cases = list(dict.fromkeys(args.cases or DEFAULT_CASES))

    import torch
    from .bench import environment, write_json
    from .triton_attention import set_attention_split_overrides, tuning_results

    if not torch.cuda.is_available():
        raise RuntimeError("attention tuning requires a CUDA GPU")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    set_attention_split_overrides({})
    result = {
        "status": "running",
        "kind": "native_triton_attention_split_kv_tuning",
        "environment": environment(),
        "method": {
            "timer": "CUDA events around full attention CUDA graph replay, including split reduction",
            "rounds": args.rounds,
            "calls_per_round": args.calls,
            "warmup_replays": args.warmup,
            "cache_mode": args.cache_mode,
            "cache_flush_mib": args.cache_flush_mib if args.cache_mode == "cold" else 0,
            "cache_flush_in_timing": False,
            "same_quantized_inputs_for_all_candidates": True,
            "reference": "independent unfused FP32 QK, softmax, PV",
            "tf32": False,
            "seed": args.seed,
            "splits": args.splits,
            "gpu_execution": "sequential; no other benchmark may share this GPU",
        },
        "attention_split_overrides": {},
        "cases": {},
    }
    write_json(args.out, result)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            flush = (torch.empty(args.cache_flush_mib * 2**20, device="cuda", dtype=torch.uint8)
                     if args.cache_mode == "cold" else None)
            for case in cases:
                key = shape_key(case)
                print(f"tuning attention {key}", flush=True)
                measured = tune_case(case, args, flush)
                result["cases"][key] = measured
                result["tile_tuning"] = tuning_results()
                result["elapsed_s"] = time.perf_counter() - started
                write_json(args.out, result)
                print(f"best at position {case[-1]}: split={measured['best_split_at_position']} "
                      f"{measured['best_median_ms'] * 1000:.3f} us", flush=True)
        result["attention_split_overrides"], result["shape_selection"] = aggregate_choices(
            result["cases"], args.splits)
        result["status"] = "ok"
        result["load_instruction"] = "Load attention_split_overrides before compilation and CUDA graph capture."
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
