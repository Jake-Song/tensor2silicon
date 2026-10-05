"""Optional packed gate/up GEMM with a SwiGLU epilogue.

Each CTA loads one activation tile and accumulates separate gate/up dots. Both
accumulators are explicitly rounded to BF16 before FP32 SiLU and multiplication,
matching the existing linear -> SwiGLU precision boundaries. No [M,2F] tensor
is written to global memory. Baseline runners do not import this experiment.

The epilogue-fusion pattern follows Triton's matrix-multiplication tutorial:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


SWIGLU_CONFIGS = {
    "m64n64k32_w4s3": dict(BM=64, BN=64, BK=32, num_warps=4, num_stages=3),
    "m64n128k32_w4s3": dict(BM=64, BN=128, BK=32, num_warps=4, num_stages=3),
    "m128n64k32_w4s3": dict(BM=128, BN=64, BK=32, num_warps=4, num_stages=3),
    "m128n128k32_w8s2": dict(BM=128, BN=128, BK=32, num_warps=8, num_stages=2),
}
DEFAULT_CONFIG = "m64n64k32_w4s3"


@triton.jit
def _gemm_swiglu(X, Packed, Out, M: tl.constexpr, K: tl.constexpr, F: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.cdiv(M, BM)
    columns = tl.cdiv(F, BN)
    group = tile // (8 * columns)
    first_row = group * 8
    group_rows = tl.minimum(rows - first_row, 8)
    row = first_row + (tile % (8 * columns)) % group_rows
    column = (tile % (8 * columns)) // group_rows
    mi = row * BM + tl.arange(0, BM)
    fi = column * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate_acc = tl.zeros((BM, BN), tl.float32)
    up_acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        ki = block * BK + kk
        activation = tl.load(X + mi[:, None] * K + ki[None, :],
                             (mi[:, None] < M) & (ki[None, :] < K), other=0)
        offset = ki[:, None] * (2 * F) + fi[None, :]
        valid = (ki[:, None] < K) & (fi[None, :] < F)
        gate_weight = tl.load(Packed + offset, valid, other=0)
        up_weight = tl.load(Packed + offset + F, valid, other=0)
        gate_acc = tl.dot(activation, gate_weight, gate_acc)
        up_acc = tl.dot(activation, up_weight, up_acc)
    # Do not remove these casts: the unfused baseline materializes BF16
    # gate/up projection outputs before evaluating FP32 SwiGLU.
    gate = gate_acc.to(tl.bfloat16).to(tl.float32)
    up = up_acc.to(tl.bfloat16).to(tl.float32)
    result = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(Out + mi[:, None] * F + fi[None, :], result,
             (mi[:, None] < M) & (fi[None, :] < F))


def launch_swiglu(x: torch.Tensor, packed_w: torch.Tensor, config: str = DEFAULT_CONFIG):
    """Return ``(SwiGLU(x @ packed_w), compiled_kernel)`` without fallback."""
    if config not in SWIGLU_CONFIGS:
        raise ValueError(f"unknown fused SwiGLU configuration {config!r}")
    if x.dtype != torch.bfloat16 or packed_w.dtype != x.dtype:
        raise TypeError("fused SwiGLU GEMM requires BF16 inputs")
    if not x.is_cuda or packed_w.device != x.device:
        raise ValueError("fused SwiGLU GEMM inputs must share a CUDA device")
    if (x.ndim < 2 or packed_w.ndim != 2 or x.shape[-1] != packed_w.shape[0]
            or packed_w.shape[1] % 2 or not x.is_contiguous() or not packed_w.is_contiguous()):
        raise ValueError("expected contiguous [...,K] and packed [K,2F] tensors")
    k, packed_n = packed_w.shape
    if k <= 0 or packed_n <= 0 or x.numel() == 0:
        raise ValueError("fused SwiGLU GEMM dimensions must be positive")
    m, f = x.numel() // k, packed_n // 2
    out = torch.empty((*x.shape[:-1], f), device=x.device, dtype=x.dtype)
    settings = SWIGLU_CONFIGS[config]
    grid = (triton.cdiv(m, settings["BM"]) * triton.cdiv(f, settings["BN"]),)
    handle = _gemm_swiglu[grid](x, packed_w, out, m, k, f, **settings)
    return out, handle


def swiglu_linear(x: torch.Tensor, packed_w: torch.Tensor, config: str = DEFAULT_CONFIG) -> torch.Tensor:
    """BF16 output with explicit BF16 gate/up and FP32 activation arithmetic."""
    return launch_swiglu(x, packed_w, config)[0]
