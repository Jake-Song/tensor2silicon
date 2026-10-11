"""A100 candidate: native CUDA SwiGLU, ordinary PyTorch for other ops."""
from functools import lru_cache
from pathlib import Path
from torch.utils.cpp_extension import load
from llm_roofline.torch_llm import (
    embed, layernorm, linear, rope_split_heads, kv_write, qk,
    softmax_masked, pv, merge_heads, add,
)


@lru_cache(maxsize=1)
def _extension():
    root = Path(__file__).resolve().parent
    return load(
        name="autoresearch_llm_cuda_a100",
        sources=[str(root / "bindings.cpp"), str(root / "swiglu.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--fmad=false",
                           "-gencode=arch=compute_80,code=sm_80",
                           "-gencode=arch=compute_80,code=compute_80"],
        verbose=False,
    )


def swiglu(g, u):
    return _extension().swiglu_cuda(g.contiguous(), u.contiguous())


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
