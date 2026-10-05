"""Optional BF16 descriptor/TMA GEMM experiment; never used by default runners.

Uses row-major activations and weights without a weight transpose. Tensor
descriptor loads and stores lower to TMA when supported by the compiler/GPU.
The companion tuner records PTX evidence and treats unsupported candidates as
failures, rather than falling back to an ordinary memory-load kernel.

API and alignment references (official Triton v3.6 and NVIDIA documentation):
https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/09-persistent-matmul.py
https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/tools/tensor_descriptor.py
https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-tuning-guide/index.html
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Even a conservative estimate including a full output staging tile remains
# below the SM120 99 KiB per-block limit. Real compiled usage is recorded too.
TMA_CONFIGS = {
    "m64n128k64_w4s2": dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=2),
    "m128n128k32_w4s3": dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3),
    "m128n128k64_w4s2": dict(BM=128, BN=128, BK=64, num_warps=4, num_stages=2),
    "m128n128k64_w8s2": dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=2),
}
DEFAULT_CONFIG = "m64n128k64_w4s2"


@triton.jit
def _tma_gemm(A, W, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    rows = tl.cdiv(M, BM)
    columns = tl.cdiv(N, BN)
    group = tile // (8 * columns)
    first_row = group * 8
    group_rows = tl.minimum(rows - first_row, 8)
    row = first_row + (tile % (8 * columns)) % group_rows
    column = (tile % (8 * columns)) // group_rows
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for block in range(tl.cdiv(K, BK)):
        # W is [K,N] with contiguous N. No hidden parameter-derived copy is
        # made: this tests TMA loads for the model's existing weight layout.
        a = A.load([row * BM, block * BK])
        w = W.load([block * BK, column * BN])
        accumulator = tl.dot(a, w, accumulator)
    Out.store([row * BM, column * BN], accumulator.to(tl.bfloat16))


@lru_cache(maxsize=8)
def _device_properties(device):
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        raise RuntimeError(f"TMA requires a supporting NVIDIA GPU; found capability {capability}")
    return torch.cuda.get_device_properties(device)


def launch_tma(x: torch.Tensor, w: torch.Tensor, config: str = DEFAULT_CONFIG):
    """Return ``(output, compiled_kernel)`` for an explicitly named candidate.

    No fallback is provided. Compile/runtime failures propagate to the tuner.
    ``x[...,K]`` and ``w[K,N]`` must be contiguous BF16 CUDA tensors, and both
    leading row strides must satisfy tensor-descriptor 16-byte alignment.
    """
    if config not in TMA_CONFIGS:
        raise ValueError(f"unknown TMA configuration {config!r}")
    if x.dtype != torch.bfloat16 or w.dtype != x.dtype:
        raise TypeError("TMA GEMM requires BF16 activations and weights")
    if not x.is_cuda or w.device != x.device:
        raise ValueError("TMA GEMM inputs must share a CUDA device")
    if x.ndim < 2 or w.ndim != 2 or x.shape[-1] != w.shape[0]:
        raise ValueError("TMA GEMM expects compatible [...,K] and [K,N] shapes")
    if not x.is_contiguous() or not w.is_contiguous():
        raise ValueError("TMA experiment requires contiguous row-major inputs")
    k, n = w.shape
    m = x.numel() // k
    if min(m, n, k) <= 0 or k % 8 or n % 8:
        raise ValueError("BF16 TMA row strides K and N must be positive multiples of 8 elements")
    _device_properties(x.device)
    settings = TMA_CONFIGS[config]
    bm, bn, bk = settings["BM"], settings["BN"], settings["BK"]
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    a_desc = TensorDescriptor.from_tensor(x.view(m, k), [bm, bk])
    w_desc = TensorDescriptor.from_tensor(w, [bk, bn])
    out_desc = TensorDescriptor.from_tensor(out.view(m, n), [bm, bn])
    handle = _tma_gemm[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a_desc, w_desc, out_desc, m, n, k, **settings)
    return out, handle


def linear_tma(x: torch.Tensor, w: torch.Tensor, config: str = DEFAULT_CONFIG) -> torch.Tensor:
    """Optional TMA projection with the same output contract as native linear."""
    return launch_tma(x, w, config)[0]


def estimated_shared_bytes(config: str) -> int:
    settings = TMA_CONFIGS[config]
    bm, bn, bk = settings["BM"], settings["BN"], settings["BK"]
    return settings["num_stages"] * (bm * bk + bk * bn) * 2 + bm * bn * 2
