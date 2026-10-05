"""Triton attention with online softmax and split-KV decode.

Inputs use [batch, heads, sequence, head_dim] and may be strided. Queries are
UNscaled; these kernels apply 1/sqrt(head_dim) exactly once. Outputs are merged
contiguous [batch, query_length, query_heads * head_dim] BF16 tensors.

Algorithm reference: Triton's fused-attention tutorial (online softmax),
https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_ATTENTION_SPLIT_VALUES = (1, 2, 4, 8, 9, 16, 32)
_ATTENTION_SPLIT_OVERRIDES: dict[tuple[int, int, int, int, int], int] = {}


def set_attention_split_overrides(mapping: dict) -> None:
    """Replace decode split choices before compilation, warmup or graph capture.

    Keys are ``(B,NQ,NK,S,H)`` tuples or comma-separated JSON strings. Choices
    are validated atomically. Empty maps restore the original split heuristic;
    compiled/captured callers must be recreated after changing the map.
    """
    parsed = {}
    for key, value in mapping.items():
        if isinstance(key, str):
            try:
                key = tuple(int(part.strip()) for part in key.split(","))
            except ValueError as exc:
                raise ValueError(f"invalid attention split shape key: {key!r}") from exc
        if (not isinstance(key, tuple) or len(key) != 5
                or any(type(dim) is not int or dim <= 0 for dim in key)
                or key[1] % key[2] or key[4] > 256):
            raise ValueError(f"attention split key must be positive (B,NQ,NK,S,H), NQ % NK = 0, H <= 256: {key!r}")
        if type(value) is not int or value not in _ATTENTION_SPLIT_VALUES:
            raise ValueError(f"attention split must be one of {_ATTENTION_SPLIT_VALUES}: {value!r}")
        parsed[key] = value
    _ATTENTION_SPLIT_OVERRIDES.clear()
    _ATTENTION_SPLIT_OVERRIDES.update(parsed)


def get_attention_split_overrides() -> dict[str, int]:
    """Return JSON-serializable per-shape decode split choices."""
    return {",".join(map(str, key)): value for key, value in sorted(_ATTENTION_SPLIT_OVERRIDES.items())}


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 64}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128}, num_warps=8, num_stages=3),
    ],
    key=["T", "S", "NQ", "NK", "H"],
)
@triton.jit
def _prefill(Q, K, V, O, T: tl.constexpr, S: tl.constexpr,
             NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
             QB: tl.constexpr, QN: tl.constexpr, QT: tl.constexpr, QH: tl.constexpr,
             KB: tl.constexpr, KN: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
             VB: tl.constexpr, VN: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, DH: tl.constexpr):
    qi = tl.program_id(0) * BM + tl.arange(0, BM)
    head_batch = tl.program_id(1)
    batch = head_batch // NQ
    head = head_batch % NQ
    kvhead = head // (NQ // NK)
    dim = tl.arange(0, DH)
    q = tl.load(Q + batch * QB + head * QN + qi[:, None] * QT + dim[None, :] * QH,
                (qi[:, None] < T) & (dim[None, :] < H), other=0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DH), tl.float32)
    # Causal queries can skip every KV block beyond the last row in this tile.
    stop = tl.minimum((tl.program_id(0) + 1) * BM, S)
    for start in range(0, stop, BN):
        si = start + tl.arange(0, BN)
        k = tl.load(K + batch * KB + kvhead * KN + dim[:, None] * KH + si[None, :] * KS,
                    (dim[:, None] < H) & (si[None, :] < S), other=0)
        scores = tl.dot(q, k) * (H ** -0.5)
        scores = tl.where((si[None, :] <= qi[:, None]) & (si[None, :] < S),
                          scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        new_l = l * alpha + tl.sum(p, axis=1)
        acc *= alpha[:, None]
        v = tl.load(V + batch * VB + kvhead * VN + si[:, None] * VS + dim[None, :] * VH,
                    (si[:, None] < S) & (dim[None, :] < H), other=0)
        acc = tl.dot(p.to(q.dtype), v, acc)
        l = new_l
        m = new_m
    result = acc / l[:, None]
    offsets = ((batch * T + qi[:, None]) * NQ + head) * H + dim[None, :]
    tl.store(O + offsets, result, (qi[:, None] < T) & (dim[None, :] < H))


@triton.autotune(
    configs=[
        triton.Config({"BN": 32}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=4, num_stages=2),
        triton.Config({"BN": 128}, num_warps=4, num_stages=2),
        triton.Config({"BN": 64}, num_warps=8, num_stages=2),
    ],
    key=["BATCH", "S", "NQ", "NK", "H", "SPLITS"],
)
@triton.jit
def _decode_parts(Q, K, V, Position, Partial, Stats,
                  BATCH: tl.constexpr, S: tl.constexpr, NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
                  QB: tl.constexpr, QN: tl.constexpr, QH: tl.constexpr,
                  KB: tl.constexpr, KN: tl.constexpr, KS: tl.constexpr, KH: tl.constexpr,
                  VB: tl.constexpr, VN: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
                  HAS_POSITION: tl.constexpr, SPLITS: tl.constexpr,
                  CHUNK: tl.constexpr, BN: tl.constexpr, DH: tl.constexpr):
    head_batch = tl.program_id(0)
    part = tl.program_id(1)
    batch = head_batch // NQ
    head = head_batch % NQ
    kvhead = head // (NQ // NK)
    dim = tl.arange(0, DH)
    q = tl.load(Q + batch * QB + head * QN + dim * QH, dim < H, other=0).to(tl.float32)
    position = S - 1
    if HAS_POSITION:
        position = tl.load(Position)
    m = tl.full((), -float("inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((DH,), tl.float32)
    start = part * CHUNK
    stop = tl.minimum(start + CHUNK, tl.minimum(S, position + 1))
    for block in range(start, stop, BN):
        si = block + tl.arange(0, BN)
        valid = (si < S) & (si <= position) & (si < start + CHUNK)
        k = tl.load(K + batch * KB + kvhead * KN + si[:, None] * KS + dim[None, :] * KH,
                    valid[:, None] & (dim[None, :] < H), other=0).to(tl.float32)
        score = tl.sum(k * q[None, :], axis=1) * (H ** -0.5)
        score = tl.where(valid, score, -float("inf"))
        new_m = tl.maximum(m, tl.max(score, axis=0))
        alpha = tl.exp(m - new_m)
        p = tl.exp(score - new_m)
        l = l * alpha + tl.sum(p, axis=0)
        v = tl.load(V + batch * VB + kvhead * VN + si[:, None] * VS + dim[None, :] * VH,
                    valid[:, None] & (dim[None, :] < H), other=0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = new_m
    tl.store(Partial + (head_batch * SPLITS + part) * H + dim, acc, dim < H)
    tl.store(Stats + (head_batch * SPLITS + part) * 2, m)
    tl.store(Stats + (head_batch * SPLITS + part) * 2 + 1, l)


@triton.jit
def _decode_reduce(Partial, Stats, O, NQ: tl.constexpr, H: tl.constexpr,
                   SPLITS: tl.constexpr, DS: tl.constexpr, DH: tl.constexpr):
    head_batch = tl.program_id(0)
    part = tl.arange(0, DS)
    dim = tl.arange(0, DH)
    m = tl.load(Stats + (head_batch * SPLITS + part) * 2, part < SPLITS, other=-float("inf"))
    l = tl.load(Stats + (head_batch * SPLITS + part) * 2 + 1, part < SPLITS, other=0)
    largest = tl.max(m, axis=0)
    # Empty splits have m=-inf and l=0. Avoid evaluating -inf - -inf into NaNs.
    weight = tl.where(l > 0, tl.exp(m - largest), 0.0)
    denom = tl.sum(l * weight, axis=0)
    partial = tl.load(Partial + (head_batch * SPLITS + part[:, None]) * H + dim[None, :],
                      (part[:, None] < SPLITS) & (dim[None, :] < H), other=0)
    result = tl.sum(partial * weight[:, None], axis=0) / denom
    tl.store(O + head_batch * H + dim, result, dim < H)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              position: torch.Tensor | None = None, split_kv: int | None = None) -> torch.Tensor:
    """Causal prefill or one-token decode over a fixed-capacity KV cache.

    Prefill requires T=S and ``position=None``. Decode T=1 attends only through
    ``position`` (a CUDA int64 scalar), or through S-1 when omitted. GQA maps each
    consecutive group of query heads to one KV head. No host scalar read occurs.
    Shape-specific autotuning must complete before CUDA graph capture.
    Explicit ``split_kv`` overrides the process-local map for one-token decode.
    """
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Triton attention expects BF16 q, k, v")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("attention expects [B,heads,sequence,H] tensors")
    b, nq, t, h = q.shape
    bk, nk, s, hk = k.shape
    if b != bk or h != hk or nq % nk or h > 256:
        raise ValueError("incompatible attention batch/head dimensions")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("attention inputs must share a CUDA device")
    if position is not None:
        if t != 1 or position.numel() != 1 or position.device != q.device or position.dtype != torch.int64:
            raise ValueError("position must be a CUDA int64 scalar for one-token decode")
    elif t != 1 and t != s:
        raise ValueError("prefill requires equal query and KV lengths")
    if split_kv is not None and (type(split_kv) is not int or split_kv not in _ATTENTION_SPLIT_VALUES or t != 1):
        raise ValueError(f"explicit split_kv requires T=1 and a split in {_ATTENTION_SPLIT_VALUES}")
    out = torch.empty((b, t, nq * h), dtype=q.dtype, device=q.device)
    dh = max(16, triton.next_power_of_2(h))
    if t == 1:
        default_splits = min(16, max(1, triton.cdiv(s, 256)))
        splits = (split_kv if split_kv is not None
                  else _ATTENTION_SPLIT_OVERRIDES.get((b, nq, nk, s, h), default_splits))
        chunk = triton.cdiv(s, splits)
        partial = torch.empty((b * nq, splits, h), dtype=torch.float32, device=q.device)
        stats = torch.empty((b * nq, splits, 2), dtype=torch.float32, device=q.device)
        _decode_parts[(b * nq, splits)](
            q, k, v, q if position is None else position, partial, stats,
            b, s, nq, nk, h, q.stride(0), q.stride(1), q.stride(3),
            *k.stride(), *v.stride(), position is not None,
            splits, chunk, DH=dh)
        _decode_reduce[(b * nq,)](
            partial, stats, out, nq, h, splits, triton.next_power_of_2(splits), dh,
            num_warps=4)
    else:
        _prefill[lambda meta: (triton.cdiv(t, meta["BM"]), b * nq)](
            q, k, v, out, t, s, nq, nk, h, *q.stride(), *k.stride(), *v.stride(),
            DH=dh)
    return out


def tuning_results() -> dict:
    """Return selected attention tile configurations after warmup."""
    return {
        "attention_split_overrides": get_attention_split_overrides(),
        "prefill": {str(key): str(config) for key, config in _prefill.cache.items()},
        "decode": {str(key): str(config) for key, config in _decode_parts.cache.items()},
    }
