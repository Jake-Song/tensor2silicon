"""Editable full-model candidate. Preserve the forward inputs and all outputs."""
import torch
import triton
import triton.language as tl
from llm_roofline.torch_llm import (embed, layernorm, linear, rope_split_heads, kv_write, qk, softmax_masked, pv, merge_heads, add)


@triton.jit
def _swiglu(g, u, out, size: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(g + i, i < size, other=0).to(tl.float32)
    y = tl.load(u + i, i < size, other=0).to(tl.float32)
    # Match the reference bf16 rounding between SiLU and the multiply.
    activated = (x * tl.sigmoid(x)).to(out.dtype.element_ty).to(tl.float32)
    tl.store(out + i, activated * y, i < size)


def swiglu(g, u):
    g, u = g.contiguous(), u.contiguous()
    out = torch.empty_like(g)
    _swiglu[(triton.cdiv(g.numel(), 256),)](g, u, out, g.numel(), BLOCK=256)
    return out


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
