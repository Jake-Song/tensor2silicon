"""Editable full-model candidate. Preserve the forward inputs and all outputs."""
from llm_roofline.jax_llm import (embed, layernorm, linear, rope_split_heads, kv_write, qk, softmax_masked, pv, merge_heads, add, swiglu)


def forward(params, tokens, consts, cache=None):
    """Prefill when cache is None; otherwise one decode step that fills the last cache slot.
    Returns (logits, [(k, v) per layer])."""
    x = embed(tokens, params["embed"])
    kv = []
    for i, p in enumerate(params["layers"]):
        h = layernorm(x, p["ln1_w"], p["ln1_b"])
        qkv = linear(h, jnp.concatenate((p["wq"],p["wk"],p["wv"]),axis=1))
        nq,nk=p["wq"].shape[1],p["wk"].shape[1]
        q, k, v = rope_split_heads(qkv[...,:nq],qkv[...,nq:nq+nk],qkv[...,nq+nk:],consts["cos"],consts["sin"])
        if cache is not None:
            k, v = kv_write(cache[i][0], k), kv_write(cache[i][1], v)
        kv.append((k, v))
        o = pv(softmax_masked(qk(q, k), consts["mask"]), v)
        x = add(x, linear(merge_heads(o), p["wo"]))
        h = layernorm(x, p["ln2_w"], p["ln2_b"])
        x = add(x, linear(swiglu(linear(h, p["w_gate"]), linear(h, p["w_up"])), p["w_down"]))
    return linear(layernorm(x, params["lnf_w"], params["lnf_b"]), params["lm_head"]), kv


import jax.numpy as jnp
