"""Framework-free description of the benchmark: model shapes, per-op FLOPs/bytes, roofline math.

Both torch_llm.py and jax_llm.py build their model from the op list returned by
``op_specs`` and report through ``classify`` / ``print_table``, so the two frameworks
are measured against exactly the same analytic counts.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 32000
    d_model: int = 2048       # D
    n_heads: int = 16         # N (query heads)
    n_kv_heads: int = 16      # K (key/value heads, K < N is GQA)
    head_dim: int = 128       # H
    d_ff: int = 5632          # F (gated MLP hidden size, ~8D/3)
    n_layers: int = 8         # L
    dtype_bytes: int = 2      # bf16 / fp16


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    batch: int                # B
    q_len: int                # T: tokens processed this step
    kv_len: int               # S: tokens attended to (T for prefill, cache length for decode)


@dataclass(frozen=True)
class Preset:
    model: ModelConfig
    prefill: PhaseConfig
    decode: PhaseConfig


PRESETS = {
    # ~0.5B-parameter LLaMA-shaped model: fits A100-40GB and a 16 GB TPU v5e with room to spare.
    "colab": Preset(ModelConfig(),
                    PhaseConfig("prefill", batch=1, q_len=2048, kv_len=2048),
                    PhaseConfig("decode", batch=8, q_len=1, kv_len=2048)),
    # Fits an 8 GB consumer GPU.
    "small": Preset(ModelConfig(vocab=32000, d_model=1024, n_heads=8, n_kv_heads=8, head_dim=128, d_ff=2816, n_layers=4),
                    PhaseConfig("prefill", batch=1, q_len=1024, kv_len=1024),
                    PhaseConfig("decode", batch=8, q_len=1, kv_len=1024)),
    # CPU smoke tests and the torch/JAX parity test.
    "tiny": Preset(ModelConfig(vocab=128, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=160, n_layers=2),
                   PhaseConfig("prefill", batch=2, q_len=16, kv_len=16),
                   PhaseConfig("decode", batch=2, q_len=1, kv_len=16)),
}


# ---------------------------------------------------------------------------
# Op list
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arg:
    shape: tuple[int, ...]
    dtype: str = "float"      # "float" (model dtype) | "bool" | "int"
    role: str = "act"         # "act" | "weight" | "mask" | "tokens"


@dataclass(frozen=True)
class OpSpec:
    name: str                 # row label, e.g. "q_proj"
    kind: str                 # "matmul" | "norm" | "softmax" | "elementwise"
    fn: str                   # op function name in torch_llm / jax_llm
    args: tuple[Arg, ...]
    flops: float              # per call
    bytes: float              # ideal HBM traffic per call: each input read once, output written once
    count: int                # calls per forward pass
    desc: str = ""            # human-readable shape
    mm_index: int | None = None   # 1..9 for the nine per-layer matmuls
    inplace: bool = False     # first arg is updated in place (KV cache write)


def causal_mask(q_len: int, kv_len: int) -> np.ndarray:
    """True where attention is NOT allowed. Query i sits at absolute position i + (S - T)."""
    i = np.arange(q_len)[:, None] + (kv_len - q_len)
    j = np.arange(kv_len)[None, :]
    return j > i


def rope_tables(head_dim: int, positions: np.ndarray, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    inv = base ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    ang = positions.astype(np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def phase_positions(phase: PhaseConfig) -> np.ndarray:
    """Absolute positions of the T new tokens. Decode fills the last cache slot (S - 1)."""
    return np.arange(phase.kv_len - phase.q_len, phase.kv_len)


def op_specs(cfg: ModelConfig, ph: PhaseConfig) -> list[OpSpec]:
    """Every op of one forward pass, in execution order. Per-layer ops have count = L."""
    B, T, S = ph.batch, ph.q_len, ph.kv_len
    D, N, K, H, F, V, L = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff, cfg.vocab, cfg.n_layers
    e = cfg.dtype_bytes
    M = B * T
    ops: list[OpSpec] = []
    mm = iter(range(1, 10))

    def linear(name, din, dout, count=L, idx=True):
        ops.append(OpSpec(name, "matmul", "linear", (Arg((B, T, din)), Arg((din, dout), role="weight")),
                          flops=2 * M * din * dout, bytes=e * (M * din + din * dout + M * dout), count=count,
                          desc=f"[{M}x{din}]@[{din}x{dout}]", mm_index=next(mm) if idx else None))

    def layernorm(name, count=L):
        ops.append(OpSpec(name, "norm", "layernorm", (Arg((B, T, D)), Arg((D,), role="weight"), Arg((D,), role="weight")),
                          flops=8 * M * D, bytes=e * (2 * M * D + 2 * D), count=count, desc=f"[{M}x{D}]"))

    def residual(name):
        ops.append(OpSpec(name, "elementwise", "add", (Arg((B, T, D)), Arg((B, T, D))),
                          flops=M * D, bytes=e * 3 * M * D, count=L, desc=f"[{M}x{D}]"))

    ops.append(OpSpec("embed", "elementwise", "embed", (Arg((B, T), "int", "tokens"), Arg((V, D), role="weight")),
                      flops=0, bytes=e * 2 * M * D + 4 * M, count=1, desc=f"gather {M} rows of [{V}x{D}]"))
    layernorm("ln1")
    linear("q_proj", D, N * H)
    linear("k_proj", D, K * H)
    linear("v_proj", D, K * H)
    ops.append(OpSpec("rope+split_heads", "elementwise", "rope_split_heads",
                      (Arg((B, T, N * H)), Arg((B, T, K * H)), Arg((B, T, K * H)),
                       Arg((T, H // 2), role="rope"), Arg((T, H // 2), role="rope")),
                      flops=3 * M * (N + K) * H + M * N * H,
                      bytes=e * (2 * M * (N + 2 * K) * H + 2 * T * (H // 2)), count=L,
                      desc=f"q,k,v [{M}x{(N + 2 * K) * H}]"))
    if T == 1:
        ops.append(OpSpec("kv_cache_write", "elementwise", "kv_write",
                          (Arg((B, K, S, H), role="cache"), Arg((B, K, 1, H))),
                          flops=0, bytes=e * 2 * B * K * H, count=2 * L, inplace=True,
                          desc=f"[{B}x{K}x1x{H}] -> slot {S - 1}"))
    ops.append(OpSpec("qk", "matmul", "qk", (Arg((B, N, T, H)), Arg((B, K, S, H))),
                      flops=2 * B * N * T * S * H, bytes=e * (B * N * T * H + B * K * S * H + B * N * T * S), count=L,
                      desc=f"{B * N}x[{T}x{H}]@[{H}x{S}]", mm_index=next(mm)))
    ops.append(OpSpec("mask+softmax", "softmax", "softmax_masked", (Arg((B, N, T, S)), Arg((T, S), "bool", "mask")),
                      flops=5 * B * N * T * S, bytes=e * 2 * B * N * T * S + T * S, count=L,
                      desc=f"{B * N}x[{T}x{S}]"))
    ops.append(OpSpec("pv", "matmul", "pv", (Arg((B, N, T, S)), Arg((B, K, S, H))),
                      flops=2 * B * N * T * S * H, bytes=e * (B * N * T * S + B * K * S * H + B * N * T * H), count=L,
                      desc=f"{B * N}x[{T}x{S}]@[{S}x{H}]", mm_index=next(mm)))
    if T > 1:   # for T == 1 the head merge is a free reshape (no kernel)
        ops.append(OpSpec("merge_heads", "elementwise", "merge_heads", (Arg((B, N, T, H)),),
                          flops=0, bytes=e * 2 * M * N * H, count=L, desc=f"[{B}x{N}x{T}x{H}] -> [{M}x{N * H}]"))
    linear("o_proj", N * H, D)
    residual("residual1")
    layernorm("ln2")
    linear("gate_proj", D, F)
    linear("up_proj", D, F)
    ops.append(OpSpec("swiglu", "elementwise", "swiglu", (Arg((B, T, F)), Arg((B, T, F))),
                      flops=5 * M * F, bytes=e * 3 * M * F, count=L, desc=f"[{M}x{F}]"))
    linear("down_proj", F, D)
    residual("residual2")
    layernorm("ln_final", count=1)
    linear("lm_head", D, V, count=1, idx=False)
    return ops


# ---------------------------------------------------------------------------
# Hardware peaks (dense 16-bit matmul TFLOP/s, HBM GB/s) per JAX/torch device
# ---------------------------------------------------------------------------

# First substring match wins. TPU v2/v3 expose one JAX device per TensorCore (half a chip).
PEAKS: list[tuple[str, float, float]] = [
    ("A100-SXM4-80GB", 312.0, 2039.0),
    ("A100 80GB", 312.0, 1935.0),
    ("A100", 312.0, 1555.0),
    ("H100", 989.0, 3350.0),
    ("L4", 121.0, 300.0),
    ("T4", 65.0, 320.0),
    ("RTX 2080 SUPER", 44.6, 496.0),     # GeForce: fp16 tensor core with fp32 accumulate
    ("TPU v6", 918.0, 1640.0),
    ("TPU v5 lite", 197.0, 819.0),
    ("TPU v5e", 197.0, 819.0),
    ("TPU v5", 459.0, 2765.0),
    ("TPU v4", 275.0, 1200.0),
    ("TPU v3", 61.5, 450.0),
    ("TPU v2", 22.5, 300.0),
]


def spec_peaks(device_name: str) -> tuple[float, float] | None:
    """(FLOP/s, bytes/s) from the spec sheet, or None for an unknown device."""
    for key, tflops, gbps in PEAKS:
        if key.lower() in device_name.lower():
            return tflops * 1e12, gbps * 1e9
    return None


# ---------------------------------------------------------------------------
# Roofline classification
# ---------------------------------------------------------------------------

LATENCY_FLOOR_S = 5e-6    # below this, launch/dispatch latency dominates


def classify(op: OpSpec, t_s: float, peak_flops: float, peak_bw: float) -> dict:
    """One result row. `bound` is the theory (AI vs ridge); `measured` is what the timing shows."""
    ai = op.flops / op.bytes if op.bytes else math.inf
    ridge = peak_flops / peak_bw
    t_roof = max(op.flops / peak_flops, op.bytes / peak_bw)
    pct_flops = op.flops / t_s / peak_flops
    pct_bw = op.bytes / t_s / peak_bw
    pct_roof = t_roof / t_s
    if t_roof < LATENCY_FLOOR_S and pct_roof < 0.25:
        measured = "latency"
    else:
        measured = "compute" if pct_flops >= pct_bw else "memory"
    return {
        "name": op.name, "kind": op.kind, "mm_index": op.mm_index, "count": op.count, "desc": op.desc,
        "flops": op.flops, "bytes": op.bytes, "ai": ai, "ridge": ridge,
        "bound": "compute" if ai >= ridge else "memory", "measured": measured,
        "t_s": t_s, "t_roof_s": t_roof, "tflops": op.flops / t_s / 1e12, "gbps": op.bytes / t_s / 1e9,
        "pct_flops": pct_flops, "pct_bw": pct_bw, "pct_roof": pct_roof,
    }


def summarize(rows: list[dict]) -> dict:
    """Time per forward pass, split by op kind (each row weighted by its count)."""
    total = sum(r["t_s"] * r["count"] for r in rows)
    by_kind: dict[str, float] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0.0) + r["t_s"] * r["count"]
    flops = sum(r["flops"] * r["count"] for r in rows)
    return {"sum_of_ops_s": total, "by_kind_s": by_kind, "flops": flops,
            "bytes": sum(r["bytes"] * r["count"] for r in rows)}


def _fmt(x: float, unit: str = "") -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= scale:
            return f"{x / scale:.3g}{suffix}{unit}"
    return f"{x:.3g}{unit}"


def print_table(title: str, rows: list[dict], peak_flops: float, peak_bw: float) -> None:
    print(f"\n=== {title}  (roofline: {peak_flops / 1e12:.3g} TFLOP/s, {peak_bw / 1e9:.0f} GB/s, "
          f"ridge AI = {peak_flops / peak_bw:.0f} FLOP/B) ===")
    extra = "xla_flops" in rows[0]
    hdr = (f"{'#':>2} {'op':<17} {'x':>3} {'shape':<30} {'FLOP':>7} {'bytes':>7} {'AI':>6} {'theory':>7} "
           f"{'us':>9} {'TFLOP/s':>8} {'GB/s':>7} {'%roof':>6} {'measured':>8}")
    if extra:
        hdr += f" {'xlaFLOP':>8} {'xlaB':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['mm_index'] or '':>2} {r['name']:<17} {r['count']:>3} {r['desc'][:30]:<30} {_fmt(r['flops']):>7} "
                f"{_fmt(r['bytes']):>7} {r['ai']:>6.1f} {r['bound']:>7} {r['t_s'] * 1e6:>9.1f} {r['tflops']:>8.1f} "
                f"{r['gbps']:>7.0f} {r['pct_roof'] * 100:>5.0f}% {r['measured']:>8}")
        if extra:
            line += f" {_fmt(r['xla_flops']):>8} {_fmt(r['xla_bytes']):>7}"
        print(line)
    s = summarize(rows)
    shares = "  ".join(f"{k} {v / s['sum_of_ops_s'] * 100:.0f}%" for k, v in sorted(s["by_kind_s"].items(), key=lambda kv: -kv[1]))
    print(f"sum of ops x count = {s['sum_of_ops_s'] * 1e3:.3f} ms per forward  |  time share: {shares}")


def rotation_count(arg_bytes: int, target_bytes: float, lo: int = 2, hi: int = 64) -> int:
    """How many distinct input copies to cycle through so on-chip caches can't hold them."""
    return int(min(hi, max(lo, math.ceil(target_bytes / max(arg_bytes, 1)))))


def save_json(path: str, meta: dict, phases: dict) -> None:
    with open(path, "w") as f:
        json.dump({"meta": meta, "phases": phases}, f, indent=1, default=float)
    print(f"\nwrote {path}")


def preset_meta(name: str) -> dict:
    p = PRESETS[name]
    return {"preset": name, "model": asdict(p.model), "prefill": asdict(p.prefill), "decode": asdict(p.decode)}


def init_params(cfg: ModelConfig, seed: int = 0, normal=None) -> dict:
    """Weights as a nested dict. `normal(shape, mean, std)` builds each array; the default makes
    deterministic float32 NumPy arrays (shared by both frameworks in the parity test)."""
    if normal is None:
        rng = np.random.default_rng(seed)

        def normal(shape, mean, std):
            return (mean + std * rng.standard_normal(shape)).astype(np.float32)

    D, N, K, H, F, V = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff, cfg.vocab

    def w(din, dout):
        return normal((din, dout), 0.0, 1 / math.sqrt(din))

    layers = []
    for _ in range(cfg.n_layers):
        layers.append({"ln1_w": normal((D,), 1.0, 0.1), "ln1_b": normal((D,), 0.0, 0.1),
                       "wq": w(D, N * H), "wk": w(D, K * H), "wv": w(D, K * H), "wo": w(N * H, D),
                       "ln2_w": normal((D,), 1.0, 0.1), "ln2_b": normal((D,), 0.0, 0.1),
                       "w_gate": w(D, F), "w_up": w(D, F), "w_down": w(F, D)})
    return {"embed": normal((V, D), 0.0, 1.0), "layers": layers,
            "lnf_w": normal((D,), 1.0, 0.1), "lnf_b": normal((D,), 0.0, 0.1), "lm_head": w(D, V)}
