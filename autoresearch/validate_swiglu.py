"""Hardware validation for the native CUDA SwiGLU tracks.

Run from the repository root: python autoresearch/validate_swiglu.py --track a100
"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def synchronized_latency(torch, fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(20):
            out = fn()
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / 20)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


def graph_latency(torch, fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Many calls per replay amortize graph launch overhead for small decode ops.
        for _ in range(100):
            out = fn()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 1000)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("a100", "g4"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", track=args.track)
    try:
        import torch
        import triton
        from autoresearch.llm_common import load_candidate
        from llm_roofline.torch_llm import swiglu as reference

        torch.set_grad_enabled(False)
        name = torch.cuda.get_device_name()
        capability = torch.cuda.get_device_capability()
        if args.track == "a100":
            assert "A100" in name and capability == (8, 0), (name, capability)
        else:
            assert "RTX PRO 6000" in name and "Blackwell" in name and capability == (12, 0), (name, capability)
        track_dir = ROOT / "autoresearch" / "gpu" / ("cuda" if args.track == "a100" else "cuda_g4")
        result["hardware"] = dict(name=name, capability=capability,
                                  memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                                  torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                                  nvcc=subprocess.check_output(["nvcc", "--version"], text=True))
        result["source_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [track_dir / n for n in ("model.py", "swiglu.cu", "bindings.cpp", "bench.py")]
            + [ROOT / "autoresearch/llm_common.py", ROOT / "llm_roofline/torch_llm.py", ROOT / "llm_roofline/spec.py"]
        }
        load_candidate(track_dir / "model.py")
        candidate = sys.modules["llm_candidate"].swiglu
        checks, timings = [], []
        torch.manual_seed(17)
        # Empty/tail/noncontiguous shapes and both actual workloads.
        for shape in ((0,), (1,), (257,), (2, 3, 160), (8, 1, 5632), (1, 2048, 5632)):
            g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            u = torch.randn_like(g)
            start = time.perf_counter()
            got = candidate(g, u)
            torch.cuda.synchronize()
            first_s = time.perf_counter() - start
            expected = reference(g, u)
            torch.testing.assert_close(got, expected, atol=0.02, rtol=0.02)
            checks.append(dict(shape=shape, max_abs_err=(got.float() - expected.float()).abs().max().item()
                               if got.numel() else 0, first_call_s=first_s))
            if len(shape) == 3 and shape[-1] == 5632:
                row = dict(shape=shape)
                for label, fn in (("eager_pytorch", lambda: reference(g, u)),
                                  ("native_cuda", lambda: candidate(g, u))):
                    row[label] = dict(synchronized=synchronized_latency(torch, fn),
                                      cuda_graph=graph_latency(torch, fn))
                timings.append(row)
        g = torch.linspace(-100, 100, 2050, device="cuda", dtype=torch.bfloat16)[::2]
        u = torch.ones_like(g)
        torch.testing.assert_close(candidate(g, u), reference(g, u), atol=0.02, rtol=0.02)
        checks.append(dict(case="noncontiguous_extreme_finite", passed=True))
        # Respect the caller's current stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            g = torch.randn((257,), device="cuda", dtype=torch.bfloat16)
            u = torch.randn_like(g)
            got, expected = candidate(g, u), reference(g, u)
        stream.synchronize()
        torch.testing.assert_close(got, expected, atol=0.02, rtol=0.02)
        checks.append(dict(case="nondefault_stream", passed=True))
        result.update(operator_checks=checks, operator_timings=timings)
        write_json(args.out / "operator.json", {**result, "status": "ok", "scope": "operator_only"})
        # Independent processes include the full-model evaluator's changed-input
        # replay gates, two-seed tiny prefill/decode, GQA, all logits and KV leaves.
        trials = []
        for phase in ("prefill", "decode"):
            for repeat in (1, 2):
                name = f"{phase}-{repeat}"
                output = args.out / f"{name}.json"
                with (args.out / f"{name}.log").open("w") as log:
                    process = subprocess.run([sys.executable, str(track_dir / "bench.py"), "--phase", phase,
                                              "--json", str(output)], stdout=log, stderr=subprocess.STDOUT,
                                             timeout=900, cwd=ROOT)
                trial = json.loads(output.read_text()) if output.exists() else dict(status="error", error="missing JSON")
                trials.append(dict(run=name, returncode=process.returncode, result=trial))
                result["full_model_trials"] = trials
                write_json(args.out / "progress.json", result)
                if process.returncode or trial.get("status") != "ok" or not trial.get("correct") or not trial.get("scored"):
                    raise RuntimeError(f"{name} failed; see saved log and JSON")
        result["status"] = "ok"
    except Exception as exc:
        result.update(status="error", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        write_json(args.out / "summary.json", result)
    print(json.dumps(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
