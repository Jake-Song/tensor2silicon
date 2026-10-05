"""Compile-compatible Triton primitives for the repository's BF16 decoder.

The public operators deliberately expose allocations and metadata to
``torch.compile`` via ``triton_op``. All device computation in these operators
is performed by Triton. RoPE and SwiGLU use FP32 intermediates and round only
their outputs; residual additions round to the model dtype before LayerNorm.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _embedding(Tokens, Table, Out, T: tl.constexpr, SB: tl.constexpr, ST: tl.constexpr,
               D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token = tl.load(Tokens + (row // T) * SB + (row % T) * ST).to(tl.int64)
    value = tl.load(Table + token * D + col, col < D, other=0)
    tl.store(Out + row * D + col, value, col < D)


@triton_op("llm_bench::embedding", mutates_args={})
def embedding(tokens: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Gather a dense embedding table for possibly strided ``tokens[B,T]``."""
    d = table.shape[1]
    out = torch.empty((*tokens.shape, d), device=table.device, dtype=table.dtype)
    wrap_triton(_embedding)[(tokens.numel(),)](
        tokens, table, out, tokens.shape[1], *tokens.stride(),
        d, triton.next_power_of_2(d), num_warps=4,
    )
    return out


@triton.jit
def _layernorm(X, W, B, Out, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    centered = tl.where(col < D, x - mean, 0)
    variance = tl.sum(centered * centered, 0) / D
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    b = tl.load(B + col, col < D, other=0).to(tl.float32)
    value = centered * tl.rsqrt(variance + EPS) * w + b
    tl.store(Out + row * D + col, value, col < D)


@triton_op("llm_bench::layernorm", mutates_args={})
def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Affine LayerNorm, using biased FP32 variance and repository epsilon."""
    d = x.shape[-1]
    out = torch.empty_like(x)
    wrap_triton(_layernorm)[(x.numel() // d,)](
        x, w, b, out, d, eps, triton.next_power_of_2(d), num_warps=4,
    )
    return out


@triton.jit
def _add(X, Y, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    y = tl.load(Y + i, i < N, other=0).to(tl.float32)
    tl.store(Out + i, x + y, i < N)


@triton_op("llm_bench::add", mutates_args={})
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    wrap_triton(_add)[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), 1024)
    return out


@triton.jit
def _add_layernorm(
    X, Residual, W, B, Sum, Norm,
    D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offsets = row * D + col
    x = tl.load(X + offsets, col < D, other=0).to(tl.float32)
    residual = tl.load(Residual + offsets, col < D, other=0).to(tl.float32)
    # Preserve the BF16 residual boundary of separate add then LayerNorm.
    summed = (x + residual).to(Sum.dtype.element_ty)
    tl.store(Sum + offsets, summed, col < D)
    xf = summed.to(tl.float32)
    mean = tl.sum(xf, 0) / D
    centered = tl.where(col < D, xf - mean, 0)
    variance = tl.sum(centered * centered, 0) / D
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    b = tl.load(B + col, col < D, other=0).to(tl.float32)
    value = centered * tl.rsqrt(variance + EPS) * w + b
    tl.store(Norm + offsets, value, col < D)


@triton_op("llm_bench::add_layernorm", mutates_args={})
def add_layernorm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return model-dtype residual sum and affine-normalized sum in one launch."""
    d = x.shape[-1]
    summed = torch.empty_like(x)
    norm = torch.empty_like(x)
    wrap_triton(_add_layernorm)[(x.numel() // d,)](
        x, residual, w, b, summed, norm, d, eps, triton.next_power_of_2(d), num_warps=4,
    )
    return summed, norm


@triton.jit
def _swiglu(Packed, Out, N: tl.constexpr, F: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // F, i % F
    gate = tl.load(Packed + row * (2 * F) + col, i < N, other=0).to(tl.float32)
    up = tl.load(Packed + row * (2 * F) + F + col, i < N, other=0).to(tl.float32)
    value = (gate / (1.0 + tl.exp(-gate))) * up
    tl.store(Out + i, value, i < N)


@triton_op("llm_bench::swiglu", mutates_args={})
def swiglu(packed: torch.Tensor) -> torch.Tensor:
    """SwiGLU from contiguous ``[gate, up]`` projections on the last axis."""
    f = packed.shape[-1] // 2
    out = torch.empty((*packed.shape[:-1], f), device=packed.device, dtype=packed.dtype)
    wrap_triton(_swiglu)[(triton.cdiv(out.numel(), 1024),)](packed, out, out.numel(), f, 1024)
    return out


@triton.jit
def _rope_qkv(
    Packed, Cos, Sin, Position, Q, K, V,
    T: tl.constexpr, N: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
    CAPACITY: tl.constexpr, DECODE: tl.constexpr, BLOCK: tl.constexpr,
):
    # One CTA handles adjacent time/feature elements of one output head.
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t, d = flat // H, flat % H
    bh = tl.program_id(1)
    b, head = bh // (N + NK), bh % (N + NK)
    width: tl.constexpr = (N + 2 * NK) * H
    half: tl.constexpr = H // 2
    valid = flat < T * H
    pos = tl.load(Position).to(tl.int32) if DECODE else 0
    c = tl.load(Cos + (pos + t) * half + d % half, valid, other=0).to(tl.float32)
    s = tl.load(Sin + (pos + t) * half + d % half, valid, other=0).to(tl.float32)
    offset = (b * T + t) * width + head * H + d
    x = tl.load(Packed + offset, valid, other=0).to(tl.float32)
    partner_offset = (b * T + t) * width + head * H + tl.where(d < half, d + half, d - half)
    partner = tl.load(Packed + partner_offset, valid, other=0).to(tl.float32)
    rotated = tl.where(d < half, x * c - partner * s, x * c + partner * s)
    if head < N:
        tl.store(Q + ((b * N + head) * T + t) * H + d, rotated, valid)
    else:
        kh = head - N
        dest_t = pos + t if DECODE else t
        dest_s: tl.constexpr = CAPACITY if DECODE else T
        dest = ((b * NK + kh) * dest_s + dest_t) * H + d
        cache_valid = valid & (dest_t < dest_s) & (dest_t >= 0)
        tl.store(K + dest, rotated, cache_valid)
        voffset = (b * T + t) * width + (N + NK + kh) * H + d
        v = tl.load(Packed + voffset, valid, other=0)
        tl.store(V + dest, v, cache_valid)


@triton_op("llm_bench::rope_qkv", mutates_args={})
def rope_qkv(
    packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    n_heads: int, n_kv_heads: int, position: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split packed projections, rotate Q/K, and transpose to ``[B,heads,T,H]``.

    Q is unscaled: the attention operator applies the score scale exactly once.
    ``position=None`` starts at zero. A scalar GPU position selects the decode
    RoPE row without a CPU synchronization.
    """
    b, t, _ = packed.shape
    h = cos.shape[-1] * 2
    q = torch.empty((b, n_heads, t, h), device=packed.device, dtype=packed.dtype)
    k = torch.empty((b, n_kv_heads, t, h), device=packed.device, dtype=packed.dtype)
    v = torch.empty_like(k)
    # The same kernel's decode mode writes absolute cache slots, so offset-only
    # rotation into compact outputs has its own small wrapper kernel below.
    if position is None:
        wrap_triton(_rope_qkv)[(triton.cdiv(t * h, 512), b * (n_heads + n_kv_heads))](
            packed, cos, sin, packed, q, k, v, t, n_heads, n_kv_heads, h, t, False, 512,
            num_warps=4,
        )
    else:
        wrap_triton(_rope_qkv_compact_position)[(triton.cdiv(t * h, 512), b * (n_heads + n_kv_heads))](
            packed, cos, sin, position, q, k, v, t, n_heads, n_kv_heads, h, 512, num_warps=4,
        )
    return q, k, v


@triton.jit
def _rope_qkv_compact_position(
    Packed, Cos, Sin, Position, Q, K, V,
    T: tl.constexpr, N: tl.constexpr, NK: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    t, d = flat // H, flat % H
    bh = tl.program_id(1)
    b, head = bh // (N + NK), bh % (N + NK)
    half: tl.constexpr = H // 2
    width: tl.constexpr = (N + 2 * NK) * H
    valid = flat < T * H
    pos = tl.load(Position).to(tl.int32)
    c = tl.load(Cos + (pos + t) * half + d % half, valid, other=0).to(tl.float32)
    s = tl.load(Sin + (pos + t) * half + d % half, valid, other=0).to(tl.float32)
    offset = (b * T + t) * width + head * H + d
    x = tl.load(Packed + offset, valid, other=0).to(tl.float32)
    partner = tl.load(
        Packed + (b * T + t) * width + head * H + tl.where(d < half, d + half, d - half),
        valid, other=0,
    ).to(tl.float32)
    rotated = tl.where(d < half, x * c - partner * s, x * c + partner * s)
    if head < N:
        tl.store(Q + ((b * N + head) * T + t) * H + d, rotated, valid)
    else:
        kh = head - N
        dest = ((b * NK + kh) * T + t) * H + d
        tl.store(K + dest, rotated, valid)
        value = tl.load(Packed + (b * T + t) * width + (N + NK + kh) * H + d, valid, other=0)
        tl.store(V + dest, value, valid)


@triton_op("llm_bench::rope_qkv_decode", mutates_args={"kcache", "vcache"})
def rope_qkv_decode(
    packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    n_heads: int, n_kv_heads: int, position: torch.Tensor,
    kcache: torch.Tensor, vcache: torch.Tensor,
) -> torch.Tensor:
    """Fuse RoPE, head transpose and writes to one static KV cache slot."""
    b, t, _ = packed.shape
    h = cos.shape[-1] * 2
    q = torch.empty((b, n_heads, t, h), device=packed.device, dtype=packed.dtype)
    wrap_triton(_rope_qkv)[(triton.cdiv(t * h, 512), b * (n_heads + n_kv_heads))](
        packed, cos, sin, position, q, kcache, vcache,
        t, n_heads, n_kv_heads, h, kcache.shape[2], True, 512, num_warps=4,
    )
    return q


@triton.jit
def _kv_write(
    KC, VC, K, V, Position, S: tl.constexpr, H: tl.constexpr, NK: tl.constexpr,
    TOTAL: tl.constexpr, BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Position).to(tl.int32)
    row, d = i // H, i % H
    target = row * S * H + pos * H + d
    valid = (i < TOTAL) & (pos >= 0) & (pos < S)
    k = tl.load(K + i, i < TOTAL, other=0)
    v = tl.load(V + i, i < TOTAL, other=0)
    tl.store(KC + target, k, valid)
    tl.store(VC + target, v, valid)


@triton_op("llm_bench::kv_write", mutates_args={"kcache", "vcache"})
def kv_write(
    kcache: torch.Tensor, vcache: torch.Tensor, knew: torch.Tensor, vnew: torch.Tensor,
    position: torch.Tensor,
) -> None:
    """Write one compact ``[B,K,1,H]`` token at a GPU-selected cache position."""
    wrap_triton(_kv_write)[(triton.cdiv(knew.numel(), 512),)](
        kcache, vcache, knew, vnew, position, kcache.shape[2], kcache.shape[3],
        kcache.shape[1], knew.numel(), 512,
    )


@triton.jit
def _argmax(
    Logits, Out, T: tl.constexpr, V: tl.constexpr,
    STRIDE_B: tl.constexpr, STRIDE_T: tl.constexpr, STRIDE_V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    values = tl.load(Logits + b * STRIDE_B + (T - 1) * STRIDE_T + i * STRIDE_V,
                     i < V, other=-float("inf")).to(tl.float32)
    token = tl.argmax(values, 0, tie_break_left=True)
    tl.store(Out + b, token.to(tl.int64))


@triton_op("llm_bench::select_token", mutates_args={})
def select_token(logits: torch.Tensor) -> torch.Tensor:
    """Greedy token from the last time position; equal maxima choose lowest id."""
    b, t, v = logits.shape
    out = torch.empty((b, 1), device=logits.device, dtype=torch.int64)
    wrap_triton(_argmax)[(b,)](
        logits, out, t, v, *logits.stride(), triton.next_power_of_2(v),
        num_warps=8 if v >= 8192 else 4,
    )
    return out


@triton_op("llm_bench::select_token_into", mutates_args={"out"})
def select_token_into(logits: torch.Tensor, out: torch.Tensor) -> None:
    b, t, v = logits.shape
    wrap_triton(_argmax)[(b,)](
        logits, out, t, v, *logits.stride(), triton.next_power_of_2(v),
        num_warps=8 if v >= 8192 else 4,
    )


@triton.jit
def _advance(Position):
    value = tl.load(Position)
    tl.store(Position, value + 1)


@triton_op("llm_bench::advance", mutates_args={"position"})
def advance(position: torch.Tensor) -> None:
    """Advance a scalar device-side position, safe for CUDA Graph replay."""
    wrap_triton(_advance)[(1,)](position, num_warps=1)


@triton.jit
def _copy_prefix(
    Src, Dst, TOTAL: tl.constexpr, K: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    SB: tl.constexpr, SK: tl.constexpr, ST: tl.constexpr, SH: tl.constexpr,
    DB: tl.constexpr, DK: tl.constexpr, DT: tl.constexpr, DH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % H
    t = (i // H) % T
    head = (i // (H * T)) % K
    b = i // (H * T * K)
    value = tl.load(Src + b * SB + head * SK + t * ST + d * SH, i < TOTAL, other=0)
    tl.store(Dst + b * DB + head * DK + t * DT + d * DH, value, i < TOTAL)


@triton_op("llm_bench::copy_prefix", mutates_args={"dst"})
def copy_prefix(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy compact prefill KV into its capacity cache, leaving future slots alone."""
    wrap_triton(_copy_prefix)[(triton.cdiv(src.numel(), 1024),)](
        src, dst, src.numel(), src.shape[1], src.shape[2], src.shape[3],
        *src.stride(), *dst.stride(), 1024,
    )


@triton.jit
def _copy_token(Src, Dst, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Src + i, i < TOTAL, other=0)
    tl.store(Dst + i, value, i < TOTAL)


@triton_op("llm_bench::copy_token", mutates_args={"dst"})
def copy_token(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy contiguous token buffers through Triton, without an ATen kernel."""
    wrap_triton(_copy_token)[(triton.cdiv(src.numel(), 256),)](src, dst, src.numel(), 256, num_warps=4)


@triton.jit
def _record_token(Token, History, INDEX: tl.constexpr, WIDTH: tl.constexpr, B: tl.constexpr,
                  BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    value = tl.load(Token + b, b < B, other=0)
    tl.store(History + b * WIDTH + INDEX, value, b < B)


@triton_op("llm_bench::record_token", mutates_args={"history"})
def record_token(token: torch.Tensor, history: torch.Tensor, index: int) -> None:
    wrap_triton(_record_token)[(1,)](
        token, history, index, history.shape[1], token.shape[0],
        triton.next_power_of_2(token.shape[0]), num_warps=1,
    )


# A descriptive alias for callers that think in terms of reduction operators.
argmax = select_token
