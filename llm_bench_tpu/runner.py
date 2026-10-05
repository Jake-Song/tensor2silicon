"""Three implementations, identical arrays, complete-output checks and timings."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
import traceback

from .metrics import NRMSE_LIMIT, NORMALIZED_MAX_LIMIT, RMS_FLOOR, errors, summarize


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = [p for p in (root / "llm_bench_tpu").glob("*.py") if p.name != "build_notebook.py"]
    files += list((root / "llm_bench_tpu" / "frozen").glob("*.py"))
    files += [root / "llm_roofline" / name for name in ("__init__.py", "jax_llm.py", "spec.py")]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def audit_hlo(text, backend, layers, attention_calls_per_layer=1):
    backend = backend.removeprefix("original_").removeprefix("incumbent_")
    ops = Counter(re.findall(r"\bstablehlo\.([a-z_]+)\b", text))
    targets = Counter(re.findall(r"stablehlo\.custom_call\s+@([^\s(]+)", text))
    pallas_calls = sum(n for k, n in targets.items() if "tpu_custom_call" in k)
    if backend == "jax" and pallas_calls:
        raise AssertionError("baseline unexpectedly contains Pallas calls")
    if backend == "hybrid" and pallas_calls != layers * attention_calls_per_layer:
        raise AssertionError(f"attention manifest mismatch, found {pallas_calls}")
    if backend == "native":
        unexpected = set(ops) - {"custom_call", "reshape", "constant"}
        if not pallas_calls or unexpected:
            raise AssertionError(f"native path contains compute outside Pallas: {unexpected}, calls={pallas_calls}")
        if any("tpu_custom_call" not in k for k in targets):
            raise AssertionError(f"native path has an unknown external call: {targets}")
    return {"passed": True, "stablehlo_ops": dict(ops), "custom_call_targets": dict(targets),
            "pallas_calls": pallas_calls, "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest()}


def validate(actual, expected, *, cache, dtype):
    import numpy as np
    if len(actual[1]) != len(expected[1]):
        raise AssertionError("wrong number of KV layers")
    rows = {}
    def check(name, a, e):
        if a.dtype != dtype:
            raise AssertionError(f"{name}: dtype {a.dtype}, expected {dtype}")
        rows[name] = errors(np.asarray(a, np.float32), np.asarray(e, np.float32))
    check("logits", actual[0], expected[0])
    for i, (got, ref) in enumerate(zip(actual[1], expected[1])):
        if len(got) != 2:
            raise AssertionError("each layer must return K and V")
        for j, kind in enumerate(("k", "v")):
            check(f"layer_{i}.{kind}", got[j], ref[j])
            if cache is not None:
                # Do not hide a corrupt new token inside a large correct prefix.
                check(f"layer_{i}.{kind}.new_token", got[j][:, :, -1:, :], ref[j][:, :, -1:, :])
                if not np.array_equal(np.asarray(got[j][:, :, :-1, :]), np.asarray(cache[i][j][:, :, :-1, :])):
                    raise AssertionError(f"layer_{i}.{kind}: cache prefix changed")
    return {"passed": all(r["passed"] for r in rows.values()), "arrays": rows,
            "max_nrmse": max(r["nrmse"] for r in rows.values()),
            "max_normalized_error": max(r["normalized_max"] for r in rows.values()),
            "cache_prefix_exact": cache is not None}


def run(args, result, save):
    import jax
    import jax.numpy as jnp
    from llm_roofline import jax_llm as base, spec
    from . import model, native
    from .config import load_config
    from .frozen import native as original_native, model as original_hybrid

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if not all(d.platform == "tpu" and "v6" in d.device_kind.lower() for d in devices):
        raise RuntimeError(f"this scored run requires a TPU v6e: {devices}")
    if len(devices) != 1:
        raise RuntimeError("this comparison requires a single TPU v6e device")
    result["runtime"] = {"python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu", "numpy")},
        "devices": [{"kind": d.device_kind, "platform": d.platform, "memory": d.memory_stats()} for d in devices],
        "default_matmul_precision": "highest (FP32 reference); timed inputs/outputs are BF16"}
    preset = spec.PRESETS["colab"]
    if args.smoke:
        preset = spec.Preset(
            spec.ModelConfig(vocab=256, d_model=256, n_heads=2, n_kv_heads=1, head_dim=128, d_ff=512, n_layers=2),
            spec.PhaseConfig("prefill", batch=2, q_len=128, kv_len=128),
            spec.PhaseConfig("decode", batch=2, q_len=1, kv_len=128))
    cfg = preset.model
    result["model"] = asdict(cfg)
    result["scored"] = not args.smoke
    print("RUNTIME", json.dumps(result["runtime"]), flush=True)
    dtype = jnp.bfloat16
    params = jax.block_until_ready(base.random_params(cfg, dtype, seed=args.seed))
    result["parameter_count"] = sum(p.size for p in jax.tree.leaves(params))
    result["weight_bytes"] = sum(p.nbytes for p in jax.tree.leaves(params))
    params32 = jax.block_until_ready(jax.tree.map(lambda x: x.astype(jnp.float32), params))
    tuning = load_config(args.tuning_config)
    result["tuning_config"] = tuning
    available = {"jax": base.forward,
        "hybrid": partial(model.forward, block_q=args.block_q, tuning=tuning),
        "native": partial(native.forward, block_q=args.block_q, tuning=tuning),
        "original_native": partial(original_native.forward, block_q=args.block_q),
        "original_hybrid": partial(original_hybrid.forward, block_q=args.block_q)}
    if args.incumbent_config:
        incumbent = load_config(args.incumbent_config)
        result["incumbent_config"] = incumbent
        available["incumbent_native"] = partial(native.forward, block_q=args.block_q, tuning=incumbent)
        available["incumbent_hybrid"] = partial(model.forward, block_q=args.block_q, tuning=incumbent)
    functions = {name: available[name] for name in args.backends.split(",")}
    order = list(functions)
    if args.reverse:
        order.reverse()
    result["compile_order"] = order
    phases = [preset.prefill, preset.decode] if args.phase == "both" else [getattr(preset, args.phase)]
    for phase in phases:
        print("PHASE", phase.name, flush=True)
        consts = base.make_consts(cfg, phase, dtype)
        tokens = jax.random.randint(jax.random.key(args.seed + 20), (phase.batch, phase.q_len), 0, cfg.vocab)
        cache = base.make_cache(cfg, phase, dtype, seed=args.seed + 1) if phase.name == "decode" else None
        inputs = jax.block_until_ready((params, tokens, consts, cache))
        consts32 = jax.tree.map(lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x, consts)
        cache32 = jax.tree.map(lambda x: x.astype(jnp.float32), cache)
        ref_fn = jax.jit(base.forward)
        t0 = time.perf_counter()
        expected = jax.block_until_ready(ref_fn(params32, tokens, consts32, cache32))
        row = {"config": asdict(phase), "fp32_reference_compile_and_first_call_s": time.perf_counter() - t0,
               "backends": {}}
        result["phases"][phase.name] = row
        compiled = {}
        for name in order:
            print("COMPILE", phase.name, name, flush=True)
            start = time.perf_counter()
            lowered = jax.jit(functions[name]).lower(*inputs)
            ir = str(lowered.compiler_ir("stablehlo"))
            evidence = audit_hlo(ir, name, cfg.n_layers)
            if args.hlo_dir:
                folder = Path(args.hlo_dir)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"{phase.name}-{name}.stablehlo.txt").write_text(ir)
            lowering_s = time.perf_counter() - start
            start = time.perf_counter()
            fn = lowered.compile()
            compile_s = time.perf_counter() - start
            start = time.perf_counter()
            actual = jax.block_until_ready(fn(*inputs))
            first_ms = (time.perf_counter() - start) * 1000
            accuracy = validate(actual, expected, cache=cache, dtype=dtype)
            row["backends"][name] = {"hlo_audit": evidence, "lowering_s": lowering_s,
                "compile_s": compile_s, "first_call_ms": first_ms, "accuracy_fp32": accuracy}
            print("ACCURACY", phase.name, name, accuracy["passed"], accuracy["max_nrmse"], accuracy["max_normalized_error"], flush=True)
            save()
            if not accuracy["passed"]:
                bad = {k:v for k,v in accuracy["arrays"].items() if not v["passed"]}
                raise AssertionError(f"{phase.name}/{name} failed fixed FP32 gates: {bad}")
            compiled[name] = fn
            del actual, lowered, ir
        del expected, ref_fn, consts32, cache32
        gc.collect()
        for _ in range(args.warmup):
            for name in order:
                jax.block_until_ready(compiled[name](*inputs))
        samples = {name: [] for name in order}
        sample_orders = []
        for i in range(args.samples):
            names = order[i % len(order):] + order[:i % len(order)]
            if (i // len(order)) % 2:
                names = names[::-1]
            sample_orders.append(names)
            for name in names:
                start = time.perf_counter_ns()
                output = compiled[name](*inputs)
                jax.block_until_ready(output)  # logits AND all KV outputs are live
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
                del output
        row["sample_orders"] = sample_orders
        for name in order:
            stats = summarize(samples[name])
            stats["tokens_per_s"] = phase.batch * phase.q_len * 1000 / stats["median_ms"]
            row["backends"][name]["latency"] = stats
            print("TIME", phase.name, name, stats["median_ms"], flush=True)
        if "jax" in row["backends"]:
            baseline_ms = row["backends"]["jax"]["latency"]["median_ms"]
            for name in order:
                row["backends"][name]["speedup_vs_jax"] = baseline_ms / row["backends"][name]["latency"]["median_ms"]
        # Recheck outputs after timing so corruption is not hidden by warmup.
        checked = jax.block_until_ready(jax.jit(base.forward)(*inputs))
        for name in order:
            actual = jax.block_until_ready(compiled[name](*inputs))
            row["backends"][name]["post_timing_vs_jax"] = validate(actual, checked, cache=cache, dtype=dtype)
            if not row["backends"][name]["post_timing_vs_jax"]["passed"]:
                raise AssertionError(f"{phase.name}/{name}: post-timing JAX comparison failed")
            del actual
        row["memory_after_phase"] = devices[0].memory_stats()
        if args.profile_dir:
            from .profiling import capture
            for name in order:
                print("PROFILE", phase.name, name, flush=True)
                row["backends"][name]["profile"] = capture(
                    compiled[name], inputs, Path(args.profile_dir) / f"{phase.name}-{name}",
                    args.profile_steps, f"{phase.name}/{name}")
                save()
        save()
        del compiled, inputs, checked, cache, consts, tokens, fn
        jax.clear_caches()
        gc.collect()
    result["complete"] = True
    result["status"] = "ok"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print("TPU_THREE_WAY_COMPLETE", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tpu-comparison.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--block-q", type=int, default=128)
    parser.add_argument("--hlo-dir")
    parser.add_argument("--tuning-config")
    parser.add_argument("--incumbent-config")
    parser.add_argument("--backends", default="jax,hybrid,native")
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-steps", type=int, default=20)
    args = parser.parse_args(argv)
    if args.samples <= 0 or args.warmup < 0 or args.block_q <= 0 or args.profile_steps <= 0:
        parser.error("samples/block-q must be positive; warmup must be nonnegative")
    result = {"status": "running", "complete": False, "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "source_sha256": source_hashes(), "phases": {},
        "accuracy_contract": {"reference": "original full JAX model; identical BF16 weights/cache/tables promoted to FP32",
            "nrmse_limit": NRMSE_LIMIT, "normalized_max_limit": NORMALIZED_MAX_LIMIT, "rms_floor": RMS_FLOOR,
            "outputs": "all logits, all KV, separately the new decode token, exact unchanged cache prefix"},
        "timing_contract": {"clock": "synchronized host wall-clock, one complete request per sample",
            "inputs": "same resident arrays for every backend; full logits and all KV returned; no donation",
            "excluded": "input setup, host transfers, compilation, warmup, FP32 validation",
            "workload": "prefill and one fixed-context decode step; not autoregressive generation",
            "order": "rotate and reverse backend order across samples; raw observations retained"}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    source_file = out.with_suffix(".sources.json")
    source_file.write_text(json.dumps({p: (source_root / p).read_text() for p in result["source_sha256"]}) + "\n")
    result["source_snapshot"] = str(source_file)
    def save():
        out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save()
    try:
        run(args, result, save)
    except Exception as exc:
        result.update(status="error", complete=False, error=f"{type(exc).__name__}: {exc}")
        save()
        traceback.print_exc()
        return 1
    return 0
