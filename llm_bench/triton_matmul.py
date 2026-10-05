"""BF16 linear projections computed entirely by Triton.

Large token matrices use grouped tensor-core GEMM, or opt-in measured TMA
configurations for explicitly selected shapes. Small token matrices use
split-K tensor-core GEMM plus a deterministic FP32 reduction. Autotuning only
changes tiling, staging and warp count, never the numerical contract.

The grouped launch ordering follows the Triton matrix multiplication tutorial:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

_SPLIT_K_OVERRIDES: dict[tuple[int, int, int], int] = {}
_SPLIT_K_VALUES = (1, 2, 4, 8, 16)
_TMA_OVERRIDES: dict[tuple[int, int, int], str] = {}


def set_split_k_overrides(mapping: dict) -> None:
    """Replace process-local split-K choices before warmup or graph capture.

    Keys are ``(M,K,N)`` tuples or JSON-friendly ``"M,K,N"`` strings. Values
    must be one of 1, 2, 4, 8, 16. An empty mapping restores the default policy.
    Already-captured CUDA graphs retain their original choices.
    """
    parsed = {}
    for key, value in mapping.items():
        if isinstance(key, str):
            try:
                key = tuple(int(part.strip()) for part in key.split(","))
            except ValueError as exc:
                raise ValueError(f"invalid split-K shape key: {key!r}") from exc
        if (not isinstance(key, tuple) or len(key) != 3
                or any(type(dim) is not int or dim <= 0 for dim in key) or key[0] > 16):
            raise ValueError(f"split-K key must be (M,K,N) with 1 <= M <= 16: {key!r}")
        if type(value) is not int or value not in _SPLIT_K_VALUES:
            raise ValueError(f"split-K must be one of {_SPLIT_K_VALUES}: {value!r}")
        parsed[key] = value
    _SPLIT_K_OVERRIDES.clear()
    _SPLIT_K_OVERRIDES.update(parsed)


def get_split_k_overrides() -> dict[str, int]:
    """Return a JSON-serializable copy of the active per-shape choices."""
    return {",".join(map(str, key)): value for key, value in sorted(_SPLIT_K_OVERRIDES.items())}


def set_tma_overrides(mapping: dict) -> None:
    """Replace measured large-M TMA choices before warmup or graph capture.

    Keys are ``(M,K,N)`` tuples or JSON-friendly ``"M,K,N"`` strings; values
    name configurations in ``triton_tma.TMA_CONFIGS``. An empty map keeps the
    ordinary GEMM path and does not import descriptor/TMA support. Invalid
    entries leave the previous map unchanged. No device work occurs here.
    """
    configs = {}
    if mapping:
        from .triton_tma import TMA_CONFIGS

        configs = TMA_CONFIGS
    parsed = {}
    for key, value in mapping.items():
        if isinstance(key, str):
            try:
                key = tuple(int(part.strip()) for part in key.split(","))
            except ValueError as exc:
                raise ValueError(f"invalid TMA shape key: {key!r}") from exc
        if (not isinstance(key, tuple) or len(key) != 3
                or any(type(dim) is not int or dim <= 0 for dim in key)
                or key[0] <= 16 or key[1] % 8 or key[2] % 8):
            raise ValueError(f"TMA key must be (M,K,N) with M > 16 and positive K,N multiples of 8: {key!r}")
        if not isinstance(value, str) or value not in configs:
            raise ValueError(f"unknown TMA configuration: {value!r}")
        parsed[key] = value
    _TMA_OVERRIDES.clear()
    _TMA_OVERRIDES.update(parsed)


def get_tma_overrides() -> dict[str, str]:
    """Return a JSON-serializable copy of active, opt-in TMA choices."""
    return {",".join(map(str, key)): value for key, value in sorted(_TMA_OVERRIDES.items())}


@triton.jit
def _gemm_tile(X, W, Y, pid, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               W_K: tl.constexpr, W_N: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    group = pid // (8 * tiles_n)
    first_m = group * 8
    group_m = tl.minimum(tiles_m - first_m, 8)
    tile_m = first_m + (pid % (8 * tiles_n)) % group_m
    tile_n = (pid % (8 * tiles_n)) // group_m
    rows = tile_m * BM + tl.arange(0, BM)
    cols = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(W + k[:, None] * W_K + cols[None, :] * W_N,
                    (k[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32, "PERSISTENT": False}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 32, "PERSISTENT": False}, num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "PERSISTENT": False}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "PERSISTENT": False}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "PERSISTENT": False}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "PERSISTENT": False}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 256, "BK": 32, "PERSISTENT": False}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "PERSISTENT": False}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 32, "PERSISTENT": False}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "PERSISTENT": False}, num_warps=4, num_stages=2),
        triton.Config({"BM": 256, "BN": 128, "BK": 32, "PERSISTENT": False}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "PERSISTENT": True}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "PERSISTENT": True}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K", "W_K", "W_N", "SMS"],
)
@triton.jit
def _gemm(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          W_K: tl.constexpr, W_N: tl.constexpr, SMS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, PERSISTENT: tl.constexpr):
    pid = tl.program_id(0)
    if PERSISTENT:
        tiles = tl.cdiv(M, BM) * tl.cdiv(N, BN)
        for tile in range(pid, tiles, SMS):
            _gemm_tile(X, W, Y, tile, M, N, K, W_K, W_N, BM, BN, BK)
    else:
        _gemm_tile(X, W, Y, pid, M, N, K, W_K, W_N, BM, BN, BK)


@triton.autotune(
    configs=[
        triton.Config({"BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BN": 64, "BK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BN": 64, "BK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BN": 128, "BK": 64}, num_warps=4, num_stages=3),
        triton.Config({"BN": 128, "BK": 128}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K", "SPLIT_K", "W_K", "W_N"],
)
@triton.jit
def _small_gemm(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                W_K: tl.constexpr, W_N: tl.constexpr,
                SPLIT_K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile_n = tl.program_id(0)
    part = tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = tile_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((16, BN), tl.float32)
    # Interleaved K blocks avoid unbalanced work for non-divisible dimensions.
    for block in range(tl.cdiv(K, BK * SPLIT_K)):
        k = (block * SPLIT_K + part) * BK + kk
        a = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        b = tl.load(W + k[:, None] * W_K + cols[None, :] * W_N,
                    (k[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + part * M * N + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _sum_parts(P, Y, SIZE: tl.constexpr, SPLIT_K: tl.constexpr,
               BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    part = tl.arange(0, SPLIT_K)
    partial = tl.load(P + part[:, None] * SIZE + i[None, :],
                      i[None, :] < SIZE, other=0)
    tl.store(Y + i, tl.sum(partial, axis=0), i < SIZE)


@lru_cache(maxsize=8)
def _multiprocessors(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def linear(x: torch.Tensor, w: torch.Tensor, split_k: int | None = None) -> torch.Tensor:
    """Return ``x @ w`` with BF16 output and FP32 dot accumulation.

    ``x`` must be contiguous with shape ``[..., K]``; ``w[K,N]`` may have
    arbitrary strides. Only allocation and metadata inspection use PyTorch.
    Autotuning must be warmed before CUDA graph capture.
    Explicit ``split_k`` overrides the process-local map for M <= 16.
    Explicitly mapped large-M shapes use descriptor/TMA GEMM without fallback.
    """
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError("Triton linear expects BF16 activations and weights")
    if not x.is_cuda or not w.is_cuda or x.device != w.device:
        raise ValueError("Triton linear inputs must share a CUDA device")
    if not x.is_contiguous() or x.ndim < 2 or w.ndim != 2 or x.shape[-1] != w.shape[0]:
        raise ValueError("linear expects contiguous [...,K] and compatible [K,N]")
    k, n = w.shape
    m = x.numel() // k
    if split_k is not None and (type(split_k) is not int or split_k not in _SPLIT_K_VALUES or m > 16):
        raise ValueError("explicit split_k requires M <= 16 and a split in (1,2,4,8,16)")
    if m > 16:
        tma_config = _TMA_OVERRIDES.get((m, k, n))
        if tma_config is not None:
            from .triton_tma import linear_tma

            return linear_tma(x, w, tma_config)
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    if m <= 16:
        default_split = 1 if k < 256 else (8 if n <= 4096 else 4 if n <= 16384 else 2)
        split = split_k if split_k is not None else _SPLIT_K_OVERRIDES.get((m, k, n), default_split)
        if split == 1:
            scratch = out
        else:
            scratch = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        _small_gemm[lambda meta: (triton.cdiv(n, meta["BN"]), split)](
            x, w, scratch, m, n, k, *w.stride(), SPLIT_K=split)
        if split != 1:
            _sum_parts[(triton.cdiv(m * n, 256),)](
                scratch, out, m * n, split, 256, num_warps=4)
    else:
        sms = _multiprocessors(x.device)

        def grid(meta):
            tiles = triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"])
            return (min(sms, tiles) if meta["PERSISTENT"] else tiles,)

        _gemm[grid](x, w, out, m, n, k, *w.stride(), sms)
    return out


def tuning_results() -> dict:
    """Report selected Triton configurations after warmup/autotuning."""
    return {
        "split_k_overrides": get_split_k_overrides(),
        "tma_overrides": get_tma_overrides(),
        "gemm": {str(key): str(config) for key, config in _gemm.cache.items()},
        "small_gemm": {str(key): str(config) for key, config in _small_gemm.cache.items()},
    }
