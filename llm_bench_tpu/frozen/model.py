"""The repository decoder, replacing only QK -> causal softmax -> PV."""
from llm_roofline import jax_llm as base

from .kernels import attention


def forward(params, tokens, consts, cache=None, *, block_q=128, interpret=False):
    x = base.embed(tokens, params["embed"])
    kv = []
    for i, p in enumerate(params["layers"]):
        h = base.layernorm(x, p["ln1_w"], p["ln1_b"])
        q, k, v = base.rope_split_heads(
            base.linear(h, p["wq"]), base.linear(h, p["wk"]), base.linear(h, p["wv"]),
            consts["cos"], consts["sin"],
        )
        if cache is not None:
            k, v = base.kv_write(cache[i][0], k), base.kv_write(cache[i][1], v)
        kv.append((k, v))
        o = attention(q, k, v, block_q=block_q, interpret=interpret)
        x = base.add(x, base.linear(base.merge_heads(o), p["wo"]))
        h = base.layernorm(x, p["ln2_w"], p["ln2_b"])
        x = base.add(x, base.linear(
            base.swiglu(base.linear(h, p["w_gate"]), base.linear(h, p["w_up"])), p["w_down"],
        ))
    logits = base.linear(base.layernorm(x, params["lnf_w"], params["lnf_b"]), params["lm_head"])
    return logits, kv
