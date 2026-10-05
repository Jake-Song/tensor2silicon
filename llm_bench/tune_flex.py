"""Tune PyTorch FlexAttention decode on the benchmark's B1 static KV layout.

Run in the already allocated GPU runtime:
    python -m llm_bench.tune_flex --out results/flex-tuning.json

This is a component experiment. A selected option still requires whole-model
correctness and timing before adoption. ``SPLIT_KV`` is a version-sensitive
compiler option: installed-source evidence and candidate failures are saved.
Public option documentation:
https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html
"""

from __future__ import annotations

import argparse
import gc
import inspect
from pathlib import Path
import time
import traceback

from .bench import Captured, environment, validate_tensor, write_json
from .metrics import summarize


def installed_option_evidence(torch):
    """Record installed implementation lines rather than assume private keys."""
    root = Path(torch.__file__).resolve().parent
    evidence = []
    for path in sorted((root / "_inductor").rglob("*flex*.py")):
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if "SPLIT_KV" in line and len(evidence) < 30:
                evidence.append(dict(path=str(path.relative_to(root)), line=number, text=line.strip()))
    return evidence


def make_attention(options):
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    def attention(q, k, v, position):
        current_position = position.reshape(())

        def score_mod(score, batch, head, q_idx, kv_idx):
            return torch.where(kv_idx <= current_position, score, float("-inf"))

        return flex_attention(q, k, v, score_mod=score_mod, kernel_options=options)

    return torch.compile(attention, fullgraph=True, mode="max-autotune-no-cudagraphs", dynamic=False)


def reference(q, k, v, position):
    """Explicit FP32 score/softmax/value product, independent of FlexAttention."""
    import torch

    score = (q.float() @ k.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
    allowed = torch.arange(k.shape[-2], device=q.device) <= position.reshape(())
    probability = torch.softmax(score.masked_fill(~allowed, -float("inf")), dim=-1)
    return probability @ v.float()


def measure_warm(fn, repeats, rounds):
    """One captured group amortizes host launch cost over repeated attention."""
    import torch

    graph = Captured(lambda: [fn() for _ in range(repeats)])
    for _ in range(5):
        graph()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / repeats)
    return dict(**summarize(times), sample_unit=f"mean of {repeats} captured attention calls")


def measure_after_eviction(fn, flush_buffer, samples):
    """Eviction runs before each start event, outside the measured interval.

    This approximates a cold KV cache; touching a large buffer does not prove
    that every relevant cache line was evicted on every GPU.
    """
    import torch

    graph = Captured(fn)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(samples)]
    for start, end in events:
        flush_buffer.add_(1)
        start.record()
        graph()
        end.record()
    torch.cuda.synchronize()
    return dict(**summarize([start.elapsed_time(end) for start, end in events]),
                sample_unit="one captured attention call after untimed cache eviction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--warm-repeats", type=int, default=64)
    parser.add_argument("--warm-rounds", type=int, default=20)
    parser.add_argument("--flush-mib", type=int, default=256)
    args = parser.parse_args()
    if min(args.samples, args.warm_repeats, args.warm_rounds, args.flush_mib) < 1:
        parser.error("sample counts and flush size must be positive")

    import torch
    from torch.nn.attention.flex_attention import flex_attention

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("a CUDA GPU with BF16 support is required")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    candidates = [{}, {"ROWS_GUARANTEED_SAFE": True}] + [
        {"SPLIT_KV": split, "ROWS_GUARANTEED_SAFE": True} for split in (1, 2, 4, 8, 16, 22, 32)
    ]
    result = dict(
        environment=environment(), arguments=vars(args),
        shape=dict(batch=1, heads=16, query_length=1, capacity=2176, head_dim=128),
        dtype="bfloat16", timed_position=2048, validation_positions=[0, 2047, 2048, 2175],
        flex_signature=str(inspect.signature(flex_attention)),
        installed_split_kv_evidence=installed_option_evidence(torch),
        method="compiled fullgraph; CUDA Graph; warm groups and untimed 256 MiB eviction by default",
        candidate_order=candidates, results=[], complete=False,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    write_json(args.out, result)
    with torch.inference_mode():
        generator = torch.Generator(device="cuda").manual_seed(8471)
        q = torch.randn((1, 16, 1, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, 16, 2176, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(k.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        position = torch.tensor(2048, device="cuda", dtype=torch.int64)
        flush_buffer = torch.zeros(args.flush_mib * 2**20 // 4, device="cuda", dtype=torch.int32)
        result["layout"] = dict(q_stride=list(q.stride()), k_stride=list(k.stride()),
                                v_stride=list(v.stride()), kv_bytes=(k.numel() + v.numel()) * k.element_size())
        for options in candidates:
            print(f"FLEX_TUNE_BEGIN {options}", flush=True)
            # Isolate the option guards between candidates rather than exceed
            # Dynamo's per-code-object recompilation limit during this sweep.
            torch._dynamo.reset()
            compiled = None
            row = dict(options=options)
            try:
                compiled = make_attention(options)
                fn = lambda: compiled(q, k, v, position)
                position.fill_(2048)
                started = time.perf_counter()
                output = fn()
                torch.cuda.synchronize()
                row["compile_first_call_s"] = time.perf_counter() - started
                checks = []
                for pos in result["validation_positions"]:
                    position.fill_(pos)
                    checks.append(validate_tensor(fn(), reference(q, k, v, position), f"position={pos}"))
                row["correctness"] = checks
                position.fill_(2048)
                row["warm"] = measure_warm(fn, args.warm_repeats, args.warm_rounds)
                row["after_eviction"] = measure_after_eviction(fn, flush_buffer, args.samples)
                row["status"] = "ok"
                print(f"FLEX_TUNE_OK {options} {row['after_eviction']['median_ms']:.6f} ms", flush=True)
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}",
                           traceback=traceback.format_exc())
                print(f"FLEX_TUNE_ERROR {options}: {error}", flush=True)
            result["results"].append(row)
            successful = [item for item in result["results"] if item["status"] == "ok"]
            result["selected_options"] = (
                min(successful, key=lambda item: item["after_eviction"]["median_ms"])["options"]
                if successful else None
            )
            write_json(args.out, result)
            del compiled
            gc.collect()
        result["complete"] = True
        result["successful_candidates"] = sum(row["status"] == "ok" for row in result["results"])
        write_json(args.out, result)
    print("FLEX_TUNE_COMPLETE", flush=True)
    return 0 if result["successful_candidates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
