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


@triton.jit
def _rope_kernel(X,C,S,Y,T:tl.constexpr,N:tl.constexpr,H:tl.constexpr,SIZE:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    d=i%H
    t=(i//H)%T
    n=(i//(H*T))%N
    b=i//(H*T*N)
    j=d%(H//2)
    offset=((b*T+t)*N+n)*H+j
    x1=tl.load(X+offset,i<SIZE,other=0).to(tl.float32)
    x2=tl.load(X+offset+H//2,i<SIZE,other=0).to(tl.float32)
    c=tl.load(C+t*(H//2)+j,i<SIZE,other=0).to(tl.float32)
    s=tl.load(S+t*(H//2)+j,i<SIZE,other=0).to(tl.float32)
    c=(c*SCALE).to(Y.dtype.element_ty).to(tl.float32)
    s=(s*SCALE).to(Y.dtype.element_ty).to(tl.float32)
    ac=(x1*c).to(Y.dtype.element_ty).to(tl.float32)
    bs=(x2*s).to(Y.dtype.element_ty).to(tl.float32)
    bc=(x2*c).to(Y.dtype.element_ty).to(tl.float32)
    ass=(x1*s).to(Y.dtype.element_ty).to(tl.float32)
    y=tl.where(d<H//2,ac-bs,bc+ass)
    tl.store(Y+i,y,i<SIZE)

def rope_split_heads(q,k,v,cos,sin):
    B,T,_=q.shape
    H=2*cos.shape[-1]
    N,K=q.shape[-1]//H,k.shape[-1]//H
    qo=torch.empty((B,N,T,H),device=q.device,dtype=q.dtype)
    ko=torch.empty((B,K,T,H),device=k.device,dtype=k.dtype)
    _rope_kernel[(triton.cdiv(q.numel(),128),)](q,cos,sin,qo,T,N,H,q.numel(),H**-0.5,128,num_warps=4,enable_fp_fusion=False)
    _rope_kernel[(triton.cdiv(k.numel(),128),)](k,cos,sin,ko,T,K,H,k.numel(),1.0,128,num_warps=4,enable_fp_fusion=False)
    return qo,ko,v.view(B,T,K,H).transpose(1,2).contiguous()

# Replay the same kernels: no compiler reassociation or precision changes.
_eager_forward = forward
_graphs = {}

def _leaves(value):
    if isinstance(value,dict):
        for key in sorted(value):
            yield from _leaves(value[key])
    elif isinstance(value,(tuple,list)):
        for part in value:yield from _leaves(part)
    elif value is not None:yield value

def forward(params,tokens,consts,cache=None):
    args=(params,tokens,consts,cache)
    leaves=tuple(_leaves(args))
    key=tuple((x.data_ptr(),tuple(x.shape),tuple(x.stride()),x.dtype) for x in leaves)
    if key not in _graphs:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                outputs=_eager_forward(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            outputs=_eager_forward(*args)
        # Strong input references prevent allocator pointer reuse from aliasing a cache key.
        _graphs[key]=(graph,outputs,args)
    graph,outputs,_=_graphs[key]
    graph.replay()  # Recompute all logits and cache writes on EVERY invocation.
    return outputs
