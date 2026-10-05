"""Explicit, phase-specific tuning. An absent config preserves the original path."""
import json
from pathlib import Path


def load_config(filename):
    if filename is None:
        return {}
    value = json.loads(Path(filename).read_text())
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("tuning config requires version: 1")
    if set(value) - {"version", "native", "hybrid", "description"}:
        raise ValueError("unknown tuning config field")
    for backend in ("native", "hybrid"):
        for phase, options in value.get(backend, {}).items():
            if phase not in ("decode", "prefill") or not isinstance(options, dict):
                raise ValueError("expected prefill/decode option dictionaries")
            allowed = {"block_q", "decode_heads"}
            if backend == "native":
                allowed |= {"gemm", "kv_block_s", "kv_group_heads", "kv_pair", "kv_flat_tail", "fuse_projection"}
            if set(options) - allowed:
                raise ValueError(f"unknown {backend} options: {set(options) - allowed}")
            for key, item in options.items():
                if key in {"kv_pair", "kv_flat_tail", "fuse_projection"}:
                    if type(item) is not bool:
                        raise ValueError(f"{key} must be a boolean")
                elif key != "gemm" and (type(item) is not int or item <= 0):
                    raise ValueError(f"{key} must be a positive integer")
    return value


def phase_options(config, backend, cache):
    return (config or {}).get(backend, {}).get("decode" if cache is not None else "prefill", {})


def gemm_tile(m, k, n, overrides=None):
    defaults = (min(512, m), min(1024 if k % 1024 == 0 else 512, k),
                min(512 if n % 512 == 0 else 256, n))
    settings = overrides or {}
    selected = settings.get(f"{m}x{k}x{n}", settings.get("default", defaults))
    if len(selected) != 3 or any(type(x) is not int or x <= 0 for x in selected):
        raise ValueError("GEMM tile must contain positive integer M, K, N sizes")
    tile = tuple(min(x, dim) for x, dim in zip(selected, (m, k, n)))
    if any(dim % x for dim, x in zip((m, k, n), tile)):
        raise ValueError(f"GEMM tile {tile} must divide shape {(m, k, n)}")
    bm, bk, bn = tile
    # Conservative double-buffered inputs/outputs plus FP32 accumulator.
    estimate = 4 * (bm * bk + bk * bn + bm * bn) + 4 * bm * bn
    if estimate > 24 * 1024**2:
        raise ValueError(f"GEMM tile {tile} exceeds the 24 MiB planning budget")
    return tile
