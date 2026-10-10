"""Lazy CUDA C++ operators, shared by compiled hybrid and native runners.

Only allocation/metadata happens in Python. Mutation contracts are explicit so
Inductor may functionalize these operators without hiding cache updates.
"""
from pathlib import Path
from typing import Optional
import functools
import torch


@functools.lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).with_name('csrc')
    return load(name='llm_bench_cuda_v1', sources=[str(root / 'bindings.cpp'), str(root / 'kernels.cu')],
                extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--fmad=false'],
                extra_ldflags=['-lcublas'], verbose=False)


def empty(shape, x, dtype=None):
    return torch.empty(shape, device=x.device, dtype=dtype or x.dtype)


@torch.library.custom_op('llm_bench_cuda::embedding', mutates_args=())
def embedding(tokens: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    out = empty((*tokens.shape, table.shape[1]), table)
    extension().embedding(tokens, table, out)
    return out


@embedding.register_fake
def _(tokens, table):
    return empty((*tokens.shape, table.shape[1]), table)


@torch.library.custom_op('llm_bench_cuda::layernorm', mutates_args=())
def layernorm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    out = torch.empty_like(x)
    extension().norm(x, x, w, b, out, out, eps, False)
    return out


@layernorm.register_fake
def _(x, w, b, eps=1e-5):
    return torch.empty_like(x)


@torch.library.custom_op('llm_bench_cuda::add_layernorm', mutates_args=())
def add_layernorm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                  eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
    summed, out = torch.empty_like(x), torch.empty_like(x)
    extension().norm(x, residual, w, b, summed, out, eps, True)
    return summed, out


@add_layernorm.register_fake
def _(x, residual, w, b, eps=1e-5):
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op('llm_bench_cuda::swiglu', mutates_args=())
def swiglu(packed: torch.Tensor) -> torch.Tensor:
    out = empty((*packed.shape[:-1], packed.shape[-1] // 2), packed)
    extension().swiglu(packed, out)
    return out


@swiglu.register_fake
def _(packed):
    return empty((*packed.shape[:-1], packed.shape[-1] // 2), packed)


def rope_outputs(packed, cos, nq, nk):
    b, t, _ = packed.shape
    h = cos.shape[-1] * 2
    return empty((b, nq, t, h), packed), empty((b, nk, t, h), packed), empty((b, nk, t, h), packed)


@torch.library.custom_op('llm_bench_cuda::rope_qkv', mutates_args=())
def rope_qkv(packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_heads: int,
             n_kv_heads: int, position: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = rope_outputs(packed, cos, n_heads, n_kv_heads)
    extension().rope(packed, cos, sin, position, q, k, v, False)
    return q, k, v


@rope_qkv.register_fake
def _(packed, cos, sin, n_heads, n_kv_heads, position=None):
    return rope_outputs(packed, cos, n_heads, n_kv_heads)


@torch.library.custom_op('llm_bench_cuda::rope_qkv_decode', mutates_args={'kcache', 'vcache'})
def rope_qkv_decode(packed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_heads: int,
                    n_kv_heads: int, position: torch.Tensor, kcache: torch.Tensor,
                    vcache: torch.Tensor) -> torch.Tensor:
    q = empty((packed.shape[0], n_heads, packed.shape[1], cos.shape[-1] * 2), packed)
    extension().rope(packed, cos, sin, position, q, kcache, vcache, True)
    return q


@rope_qkv_decode.register_fake
def _(packed, cos, sin, n_heads, n_kv_heads, position, kcache, vcache):
    return empty((packed.shape[0], n_heads, packed.shape[1], cos.shape[-1] * 2), packed)


@torch.library.custom_op('llm_bench_cuda::select_token', mutates_args=())
def select_token(logits: torch.Tensor) -> torch.Tensor:
    out = empty((logits.shape[0], 1), logits, torch.int64)
    extension().argmax(logits, out)
    return out


@select_token.register_fake
def _(logits):
    return empty((logits.shape[0], 1), logits, torch.int64)


@torch.library.custom_op('llm_bench_cuda::advance', mutates_args={'position'})
def advance(position: torch.Tensor) -> None:
    extension().advance(position)


@torch.library.custom_op('llm_bench_cuda::copy_prefix', mutates_args={'dst'})
def copy_prefix(src: torch.Tensor, dst: torch.Tensor) -> None:
    extension().copy_prefix(src, dst)


@torch.library.custom_op('llm_bench_cuda::copy_token', mutates_args={'dst'})
def copy_token(src: torch.Tensor, dst: torch.Tensor) -> None:
    extension().copy_token(src, dst)


def linear(x, weight):
    out = empty((*x.shape[:-1], weight.shape[1]), x)
    extension().linear(x, weight, out)
    return out


def attention(q, k, v, position=None):
    b, nq, t, h = q.shape
    out = empty((b, t, nq * h), q)
    if position is None:
        # Tensor Core QK/PV, FP32 scores/softmax, BF16 probabilities. These
        # intermediates belong to native CUDA and are never ATen arithmetic.
        scores = empty((b, nq, t, k.shape[2]), q, torch.float32)
        probs = empty(scores.shape, q)
        heads = empty(q.shape, q)
        extension().prefill_attention(q, k, v, scores, probs, heads, out)
    else:
        parts = 32
        scratch = empty((b, nq, parts, h + 2), q, torch.float32)
        extension().decode_attention(q, k, v, position, scratch, out, parts)
    return out


def build_metadata():
    return dict(cublas_version=extension().cublas_version(), cflags=["-O3"],
                nvcc_flags=["-O3", "--fmad=false"], library="cuBLAS",
                prefill_attention="cuBLAS QK + CUDA causal softmax + cuBLAS PV",
                decode_attention="CUDA FP32 online softmax, 32 KV splits",
                probability_storage="bfloat16", compute_storage="bfloat16", accumulation="float32")


@torch.library.custom_op('llm_bench_cuda::select_token_into', mutates_args={'out'})
def select_token_into(logits: torch.Tensor, out: torch.Tensor) -> None:
    extension().argmax(logits, out)


@torch.library.custom_op('llm_bench_cuda::record_token', mutates_args={'history'})
def record_token(token: torch.Tensor, history: torch.Tensor, index: int) -> None:
    if not 0 <= index < history.shape[1]:
        raise ValueError('token history index out of bounds')
    extension().record_token(token, history, index)
