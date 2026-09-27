"""Fixed full-model evaluator shared by the GPU and TPU tracks.

Only each track's model.py is an experiment surface. No accelerator imports occur
at module import time, so scoring and output validation can be tested on CPU.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np

from llm_roofline import spec

ATOL = RTOL = 0.02
WARMUP, ROUNDS, CALLS = 5, 7, 10


def improvement_confirmed(incumbent_ms, first_ms, confirmation_ms):
    """Both independent runs must beat the incumbent by strictly more than 1%."""
    values = (incumbent_ms, first_ms, confirmation_ms)
    return (all(math.isfinite(v) and v > 0 for v in values)
            and max(first_ms, confirmation_ms) < incumbent_ms * 0.99)


def validate_output(actual, expected, to_numpy, dtype):
    """Check all logits and cache leaves, not just the last token or an argmax."""
    max_error = 0.0

    def visit(a, e, label):
        nonlocal max_error
        if isinstance(e, (tuple, list)):
            if not isinstance(a, (tuple, list)) or len(a) != len(e):
                raise AssertionError(f"{label}: wrong output structure")
            for i, (x, y) in enumerate(zip(a, e)):
                visit(x, y, f"{label}/{i}")
            return
        if getattr(a, "shape", None) != e.shape or getattr(a, "dtype", None) != dtype:
            raise AssertionError(f"{label}: wrong shape or dtype")
        x, y = to_numpy(a), to_numpy(e)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise AssertionError(f"{label}: nonfinite output")
        err = float(np.max(np.abs(x - y)))
        max_error = max(max_error, err)
        if not np.allclose(x, y, atol=ATOL, rtol=RTOL):
            raise AssertionError(f"{label}: max_abs_err={err:.6g}")

    visit(actual, expected, "output")
    return max_error


class TorchBackend:
    def __init__(self, smoke):
        import torch
        import triton
        from llm_roofline import torch_llm
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required, including for the Triton smoke test")
        self.t, self.ref = torch, torch_llm
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.name = torch.cuda.get_device_name()
        if not smoke and "A100" not in self.name:
            raise RuntimeError(f"Scored GPU runs require A100; found {self.name}")
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.versions = {"torch": torch.__version__, "triton": triton.__version__}

    def prepare(self, fn, phase):
        return fn

    def case(self, cfg, phase, seed):
        params = self.ref.random_params(cfg, self.dtype, self.device, seed=seed)
        consts = self.ref.make_consts(cfg, phase, self.dtype, self.device)
        g = self.t.Generator(device=self.device).manual_seed(seed + 100)
        tokens = self.t.randint(cfg.vocab, (phase.batch, phase.q_len), generator=g, device=self.device)
        cache = None
        if phase.name == "decode":
            shape = (phase.batch, cfg.n_kv_heads, phase.kv_len, cfg.head_dim)
            cache = [(self.t.randn(shape, generator=g, device=self.device, dtype=self.dtype),
                      self.t.randn(shape, generator=g, device=self.device, dtype=self.dtype))
                     for _ in range(cfg.n_layers)]
        return params, tokens, consts, cache

    def clone_cache(self, cache):
        return None if cache is None else [(k.clone(), v.clone()) for k, v in cache]

    def sync(self, output):
        self.t.cuda.synchronize()
        return output

    def numpy(self, x):
        return x.detach().float().cpu().numpy()

    def freshen(self, case, cfg, phase, seed):
        fresh = self.case(cfg, phase, seed)
        def copy(a, b):
            if isinstance(a, dict):
                for key in a:
                    copy(a[key], b[key])
            elif isinstance(a, (tuple, list)):
                for x, y in zip(a, b):
                    copy(x, y)
            elif a is not None:
                a.copy_(b)
        copy(case, fresh)
        return case

    def cleanup(self):
        gc.collect()
        self.t.cuda.empty_cache()


class JaxBackend:
    def __init__(self, smoke):
        import jax
        import jax.numpy as jnp
        from llm_roofline import jax_llm
        self.j, self.ref, self.dtype = jax, jax_llm, jnp.bfloat16
        device = jax.devices()[0]
        self.name = device.device_kind
        if not smoke and (device.platform != "tpu" or not any(s in self.name.lower() for s in ("v5 lite", "v5e", "v6 lite", "v6e"))):
            raise RuntimeError(f"Scored TPU runs require v5e or v6e; found {device}")
        self.versions = {"jax": jax.__version__}
        from importlib.metadata import version, PackageNotFoundError
        for package in ("jaxlib", "libtpu"):
            try:
                self.versions[package] = version(package)
            except PackageNotFoundError:
                pass
        self.np = jnp

    def prepare(self, fn, phase):
        # Full (logits, cache) output escapes the compiled boundary in both phases.
        return self.j.jit(fn, donate_argnums=(3,) if phase.name == "decode" else ())

    def case(self, cfg, phase, seed):
        params = self.ref.random_params(cfg, self.dtype, seed=seed)
        consts = self.ref.make_consts(cfg, phase, self.dtype)
        tokens = self.j.random.randint(self.j.random.key(seed + 100),
                                       (phase.batch, phase.q_len), 0, cfg.vocab)
        cache = self.ref.make_cache(cfg, phase, self.dtype, seed=seed + 200) if phase.name == "decode" else None
        return params, tokens, consts, cache

    def clone_cache(self, cache):
        return None if cache is None else self.j.tree.map(lambda x: x.copy(), cache)

    def sync(self, output):
        return self.j.block_until_ready(output)

    def numpy(self, x):
        return np.asarray(x, dtype=np.float32)

    def freshen(self, case, cfg, phase, seed):
        # JAX arrays are immutable; new values must be passed through fresh buffers.
        return self.case(cfg, phase, seed)

    def cleanup(self):
        gc.collect()
        self.j.clear_caches()


def check_case(backend, candidate, reference, case):
    p, t, c, cache = case
    # Independent cache copy: the reference cannot overwrite or donate the candidate input.
    expected = backend.sync(reference(p, t, c, backend.clone_cache(cache)))
    actual = backend.sync(candidate(p, t, c, cache))
    error = validate_output(actual, expected, backend.numpy, backend.dtype)
    if cache is not None:
        for layer, (got, wanted) in enumerate(zip(actual[1], expected[1])):
            for a, e in zip(got, wanted):
                if not np.array_equal(backend.numpy(a[:, :, :-1]), backend.numpy(e[:, :, :-1])):
                    raise AssertionError(f"cache layer {layer}: modified prefix")
    return error, (p, t, c, actual[1] if cache is not None else None)


def time_forward(backend, fn, case):
    p, tokens, consts, cache = case

    def step():
        nonlocal cache
        out = backend.sync(fn(p, tokens, consts, cache))
        if cache is not None:
            cache = out[1]
        return out

    backend.sync(case)  # Finish device input creation before starting the clock.
    start = time.perf_counter()
    output = step()
    first_call_s = time.perf_counter() - start
    for _ in range(WARMUP):
        output = step()
    samples = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        for _ in range(CALLS):
            output = step()
        samples.append((time.perf_counter() - start) * 1000 / CALLS)
    return statistics.median(samples), samples, first_call_s, (p, tokens, consts, cache), output


def load_candidate(path):
    module_spec = importlib.util.spec_from_file_location("llm_candidate", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module  # Triton/compilers resolve the function module.
    module_spec.loader.exec_module(module)
    return module.forward


def evaluate(backend, forward, phase_name, smoke):
    errors = []
    # Both phases, independent seeds, nontrivial positions, and GQA are correctness gates.
    tiny = spec.PRESETS["tiny"]
    for phase in (tiny.prefill, tiny.decode):
        candidate = backend.prepare(forward, phase)
        reference = backend.prepare(backend.ref.forward, phase)
        for seed in (11, 29):
            error, state = check_case(backend, candidate, reference, backend.case(tiny.model, phase, seed))
            errors.append(error)
            del state
        del candidate, reference
    backend.cleanup()
    preset = tiny if smoke else spec.PRESETS["colab"]
    phase = getattr(preset, phase_name)
    # Baseline and candidate use identical values, but are allocated and timed separately.
    reference = backend.prepare(backend.ref.forward, phase)
    baseline_ms, baseline_samples, baseline_compile, state, baseline_output = time_forward(
        backend, reference, backend.case(preset.model, phase, 41))
    del state
    backend.cleanup()
    candidate = backend.prepare(forward, phase)
    case = backend.case(preset.model, phase, 41)
    # Compile/first execution is measured before correctness warms this specialization.
    time_ms, samples, compile_s, case, timed_output = time_forward(backend, candidate, case)
    errors.append(validate_output(timed_output, baseline_output, backend.numpy, backend.dtype))
    del baseline_output, timed_output
    error, case = check_case(backend, candidate, reference, case)
    errors.append(error)
    case = backend.freshen(case, preset.model, phase, 73)
    error, case = check_case(backend, candidate, reference, case)
    errors.append(error)
    return dict(status="ok", correct=True, scored=not smoke, phase=phase_name,
                preset="tiny" if smoke else "colab", device=backend.name, versions=backend.versions,
                time_ms=time_ms, baseline_ms=baseline_ms, speedup=baseline_ms / time_ms,
                max_abs_err=max(errors), compile_and_first_call_s=compile_s,
                baseline_compile_and_first_call_s=baseline_compile,
                samples_ms=samples, baseline_samples_ms=baseline_samples)


def main(kind, model_path):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny correctness/timing check; never a scored result")
    parser.add_argument("--json", type=Path, help="write a machine-readable result, including failures")
    args = parser.parse_args()
    result = dict(status="error", correct=False, scored=False, phase=args.phase)
    try:
        backend = TorchBackend(args.smoke) if kind == "gpu" else JaxBackend(args.smoke)
        result = evaluate(backend, load_candidate(model_path), args.phase, args.smoke)
    except AssertionError as e:
        result.update(status="incorrect", error=str(e))
        traceback.print_exc()
    except Exception as e:
        result.update(error=str(e))
        traceback.print_exc()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print("---")
    for key in ("status", "correct", "scored", "phase", "device", "time_ms", "baseline_ms", "speedup",
                "max_abs_err", "compile_and_first_call_s", "error"):
        if key in result:
            print(f"{key}: {result[key]}")
    print("result_json: " + json.dumps(result))
    return 0 if result["status"] == "ok" else 1
