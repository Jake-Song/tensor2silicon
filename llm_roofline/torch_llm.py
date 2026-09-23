"""Basic decoder-only LLM in PyTorch, timed op by op against the roofline.

    python -m llm_roofline.torch_llm --preset colab --phase both --out results_gpu.json

The model's forward() is built from the op functions below, and the per-op benchmark
times those same functions on inputs of the shapes listed by spec.op_specs().
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time

import torch
import torch.nn.functional as F

from . import spec

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ---------------------------------------------------------------------------
# Ops (one function per row of spec.op_specs)
# ---------------------------------------------------------------------------

def embed(tokens, table):
    return F.embedding(tokens, table)


def layernorm(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps=1e-5)


def linear(x, w):
    return x @ w


def rope_split_heads(q, k, v, cos, sin):
    """RoPE on q and k (q also takes the 1/sqrt(H) score scale), then [B,T,heads*H] -> [B,heads,T,H]."""
    B, T, _ = q.shape
    H = 2 * cos.shape[-1]
    scale = H ** -0.5

    def rot(x, c, s):
        x = x.view(B, T, -1, H)
        x1, x2 = x[..., : H // 2], x[..., H // 2:]
        c, s = c[None, :, None, :], s[None, :, None, :]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).transpose(1, 2).contiguous()

    return rot(q, cos * scale, sin * scale), rot(k, cos, sin), v.view(B, T, -1, H).transpose(1, 2).contiguous()


def kv_write(cache, new):
    """Decode: store the new token's K or V in the last cache slot, in place."""
    cache[:, :, -1:].copy_(new)
    return cache


def qk(q, k):
    """[B,N,T,H] x [B,K,S,H] -> scores [B,N,T,S]. GQA: the N/K query heads of a group share one K head."""
    B, N, T, H = q.shape
    K = k.shape[1]
    return (q.view(B, K, (N // K) * T, H) @ k.transpose(-1, -2)).view(B, N, T, -1)


def softmax_masked(s, mask):
    return torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)


def pv(p, v):
    B, N, T, S = p.shape
    K = v.shape[1]
    return (p.view(B, K, (N // K) * T, S) @ v).view(B, N, T, -1)


def merge_heads(o):
    B, N, T, H = o.shape
    return o.transpose(1, 2).reshape(B, T, N * H)


def add(x, y):
    return x + y


def swiglu(g, u):
    return F.silu(g) * u


OPS = {f.__name__: f for f in (embed, layernorm, linear, rope_split_heads, kv_write, qk, softmax_masked, pv,
                               merge_heads, add, swiglu)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def forward(params, tokens, consts, cache=None):
    """Prefill when cache is None; otherwise one decode step that fills the last cache slot.
    Returns (logits, [(k, v) per layer])."""
    x = embed(tokens, params["embed"])
    kv = []
    for i, p in enumerate(params["layers"]):
        h = layernorm(x, p["ln1_w"], p["ln1_b"])
        q, k, v = rope_split_heads(linear(h, p["wq"]), linear(h, p["wk"]), linear(h, p["wv"]), consts["cos"], consts["sin"])
        if cache is not None:
            k, v = kv_write(cache[i][0], k), kv_write(cache[i][1], v)
        kv.append((k, v))
        o = pv(softmax_masked(qk(q, k), consts["mask"]), v)
        x = add(x, linear(merge_heads(o), p["wo"]))
        h = layernorm(x, p["ln2_w"], p["ln2_b"])
        x = add(x, linear(swiglu(linear(h, p["w_gate"]), linear(h, p["w_up"])), p["w_down"]))
    return linear(layernorm(x, params["lnf_w"], params["lnf_b"]), params["lm_head"]), kv


def to_torch(tree, dtype, device):
    if isinstance(tree, dict):
        return {k: to_torch(v, dtype, device) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(to_torch(v, dtype, device) for v in tree)
    t = torch.as_tensor(tree, device=device)
    return t.to(dtype) if t.is_floating_point() else t


def random_params(cfg, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)

    def normal(shape, mean, std):
        return (torch.randn(shape, device=device, generator=g) * std + mean).to(dtype)

    return spec.init_params(cfg, normal=normal)


def make_consts(cfg, phase, dtype, device):
    cos, sin = spec.rope_tables(cfg.head_dim, spec.phase_positions(phase))
    return {"cos": torch.tensor(cos, dtype=dtype, device=device), "sin": torch.tensor(sin, dtype=dtype, device=device),
            "mask": torch.tensor(spec.causal_mask(phase.q_len, phase.kv_len), device=device)}


def make_cache(cfg, phase, dtype, device):
    shape = (phase.batch, cfg.n_kv_heads, phase.kv_len, cfg.head_dim)
    return [(torch.randn(shape, dtype=dtype, device=device), torch.randn(shape, dtype=dtype, device=device))
            for _ in range(cfg.n_layers)]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timeit(fn, arg_sets, device, target_s=0.02, rounds=7, graph=False):
    """Median seconds per call. Each round calls fn back to back on every arg set (rotating inputs so
    the L2 cache can't hold them), repeated until the round lasts ~target_s.
    graph=True captures one pass into a CUDA graph, which removes CPU launch overhead so the number
    is kernel time only."""
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    def one_pass():
        for a in arg_sets:
            fn(*a)

    if graph and cuda:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            one_pass()                      # warm up (cuBLAS handles, workspaces) outside capture
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            one_pass()
        one_pass = g.replay

    def run(reps):
        for _ in range(reps):
            one_pass()

    run(1)
    sync()
    t0 = time.perf_counter()
    run(1)
    sync()
    reps = max(1, math.ceil(target_s / max(time.perf_counter() - t0, 1e-7)))
    times = []
    for _ in range(rounds):
        if cuda:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            run(reps)
            e1.record()
            e1.synchronize()
            t = e0.elapsed_time(e1) / 1e3
        else:
            t0 = time.perf_counter()
            run(reps)
            t = time.perf_counter() - t0
        times.append(t / (reps * len(arg_sets)))
    return statistics.median(times)


def make_args(op, cfg, phase, dtype, device, g):
    out = []
    for a in op.args:
        if a.dtype == "int":
            out.append(torch.randint(0, cfg.vocab, a.shape, device=device, generator=g))
        elif a.dtype == "bool":
            out.append(torch.tensor(spec.causal_mask(*a.shape), device=device))
        else:
            std = 1 / math.sqrt(a.shape[0]) if a.role == "weight" and len(a.shape) == 2 else 1.0
            out.append((torch.randn(a.shape, device=device, generator=g) * std).to(dtype))
    return out


def measure_ceilings(dtype, device, n, bw_bytes):
    """Matmul FLOP/s, and HBM bandwidth as the better of a copy (read + write) and a read-only sum."""
    a = torch.randn(n, n, device=device).to(dtype)
    b = torch.randn(n, n, device=device).to(dtype)
    t_mm = timeit(lambda: a @ b, [()], device, rounds=5)
    x = torch.zeros(bw_bytes // a.element_size(), dtype=dtype, device=device)
    bw_copy = 2 * x.nbytes / timeit(lambda: x + 1, [()], device, rounds=5)
    bw_read = x.nbytes / timeit(lambda: x.sum(dtype=torch.float32), [()], device, rounds=5)
    return 2 * n**3 / t_mm, max(bw_copy, bw_read)


def bench_ops(cfg, phase, dtype, device, peaks, rotate_bytes):
    g = torch.Generator(device=device).manual_seed(0)
    rows = []
    for op in spec.op_specs(cfg, phase):
        first = make_args(op, cfg, phase, dtype, device, g)
        k = spec.rotation_count(max(op.bytes, sum(t.numel() * t.element_size() for t in first)), rotate_bytes)
        try:
            sets = [first] + [make_args(op, cfg, phase, dtype, device, g) for _ in range(k - 1)]
            t = timeit(OPS[op.fn], sets, device, graph=True)
        except Exception as e:  # e.g. out of memory on a small device: report and keep going
            print(f"[{op.name} skipped: {type(e).__name__}: {str(e).splitlines()[0][:160]}]")
            continue
        finally:
            sets = first = None
        rows.append(spec.classify(op, t, *peaks))
    return rows


def bench_attention(cfg, phase, dtype, device):
    """Unfused QK^T -> mask+softmax -> PV (what forward() runs) against fused SDPA."""
    B, T, S, N, K, H = phase.batch, phase.q_len, phase.kv_len, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q = torch.randn(B, N, T, H, device=device).to(dtype) * H ** -0.5
    k = torch.randn(B, K, S, H, device=device).to(dtype)
    v = torch.randn(B, K, S, H, device=device).to(dtype)
    mask = torch.tensor(spec.causal_mask(T, S), device=device)

    def unfused():
        return pv(softmax_masked(qk(q, k), mask), v)

    def fused():
        return F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, scale=1.0, enable_gqa=N != K)

    err = (unfused().float() - fused().float()).abs().max().item()
    return {"unfused_s": timeit(unfused, [()], device, graph=True), "fused_s": timeit(fused, [()], device, graph=True),
            "fused_impl": "F.scaled_dot_product_attention", "max_abs_err": err}


def bench_model(cfg, phase, dtype, device, compile_model, profile):
    params = random_params(cfg, dtype, device)
    consts = make_consts(cfg, phase, dtype, device)
    tokens = torch.randint(0, cfg.vocab, (phase.batch, phase.q_len), device=device)
    cache = make_cache(cfg, phase, dtype, device) if phase.q_len == 1 else None
    run = lambda: forward(params, tokens, consts, cache)  # noqa: E731
    res = {"eager_s": timeit(run, [()], device, rounds=5)}
    if device.type == "cuda":   # same kernels, launched from a CUDA graph: the gap is CPU launch overhead
        res["eager_graph_s"] = timeit(run, [()], device, rounds=5, graph=True)
    if compile_model:
        cf = torch.compile(forward)
        try:
            t0 = time.perf_counter()
            cf(params, tokens, consts, cache)
            res["compile_time_s"] = time.perf_counter() - t0
            res["compiled_s"] = timeit(lambda: cf(params, tokens, consts, cache), [()], device, rounds=5)
        except Exception as e:  # keep the per-op results even if Inductor fails on this setup
            print(f"[torch.compile failed: {type(e).__name__}: {e}]")
    if profile and device.type == "cuda":
        from torch.profiler import ProfilerActivity
        for label, fn in [("eager", forward)] + ([("compiled", cf)] if "compiled_s" in res else []):
            with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(params, tokens, consts, cache)
                torch.cuda.synchronize()
            print(f"\n--- {phase.name}: top CUDA kernels, {label} forward ---")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15, max_name_column_width=60))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="colab", choices=spec.PRESETS)
    ap.add_argument("--phase", default="both", choices=["prefill", "decode", "both"])
    ap.add_argument("--dtype", default="bf16", choices=DTYPES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--roofline", default="measured", choices=["measured", "spec"],
                    help="classify against measured ceilings or spec-sheet peaks")
    ap.add_argument("--rotate-mb", type=float, default=512, help="distinct input bytes cycled per op (defeats L2)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    device, dtype = torch.device(args.device), DTYPES[args.dtype]
    preset = spec.PRESETS[args.preset]
    cfg = dataclasses.replace(preset.model, dtype_bytes=torch.finfo(dtype).bits // 8)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    accel = device.type == "cuda"
    print(f"torch {torch.__version__} | device {name} | dtype {args.dtype} | preset {args.preset}")

    with torch.inference_mode():
        measured = measure_ceilings(dtype, device, 8192 if accel else 1024, 2**30 if accel else 2**26)
        sheet = spec.spec_peaks(name)
        print(f"measured ceilings: {measured[0] / 1e12:.1f} TFLOP/s matmul, {measured[1] / 1e9:.0f} GB/s memory (best of copy, read)"
              + (f"  (spec sheet: {sheet[0] / 1e12:.0f} TFLOP/s, {sheet[1] / 1e9:.0f} GB/s)" if sheet else ""))
        peaks = sheet if args.roofline == "spec" and sheet else measured

        phases = {}
        for phase in ([preset.prefill, preset.decode] if args.phase == "both" else [getattr(preset, args.phase)]):
            rows = bench_ops(cfg, phase, dtype, device, peaks, args.rotate_mb * 2**20)
            label = f"{phase.name}: B={phase.batch} T={phase.q_len} S={phase.kv_len}"
            spec.print_table(label, rows, *peaks)
            attn = bench_attention(cfg, phase, dtype, device)
            print(f"attention: unfused {attn['unfused_s'] * 1e6:.1f} us vs fused SDPA {attn['fused_s'] * 1e6:.1f} us "
                  f"({attn['unfused_s'] / attn['fused_s']:.2f}x), max |diff| {attn['max_abs_err']:.3g}")
            model = bench_model(cfg, phase, dtype, device, not args.no_compile, not args.no_profile)
            summary = spec.summarize(rows)
            print(f"whole model: eager {model['eager_s'] * 1e3:.3f} ms"
                  + (f" | eager+CUDA graph {model['eager_graph_s'] * 1e3:.3f} ms" if "eager_graph_s" in model else "")
                  + (f" | torch.compile {model['compiled_s'] * 1e3:.3f} ms" if "compiled_s" in model else "")
                  + f" | sum of ops {summary['sum_of_ops_s'] * 1e3:.3f} ms")
            phases[phase.name] = {"phase": dataclasses.asdict(phase), "rows": rows, "attention": attn,
                                  "model": model, "summary": summary}

    if args.out:
        meta = {"framework": "pytorch", "version": torch.__version__, "device": name, "dtype": args.dtype,
                "measured_peaks": measured, "spec_peaks": sheet, "roofline": args.roofline, "peaks_used": peaks,
                **spec.preset_meta(args.preset)}
        spec.save_json(args.out, meta, phases)


if __name__ == "__main__":
    main()
