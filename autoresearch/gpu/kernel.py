"""Y = ReLU(XW + b) as one Triton kernel. The ONLY file the agent edits.

Baseline: textbook tiled matmul with bias + ReLU fused into the epilogue.
Deliberately untuned (small tiles, no autotune, no grouped ordering).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def relu_linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # fp32 accumulator
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)                                  # bf16 x bf16 -> fp32 on Tensor Cores
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # epilogue: bias + ReLU while the tile is still in registers
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    y = tl.maximum(acc + b[None, :].to(tl.float32), 0.0)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def relu_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    relu_linear_kernel[grid](
        x, w, b, y, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return y
