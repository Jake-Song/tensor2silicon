# How FlashAttention Works

**FlashAttention speeds up attention by keeping intermediate calculations in fast on-chip memory and avoiding a huge attention matrix in GPU main memory.** It computes exact attention, subject to floating-point rounding. [Original paper](https://arxiv.org/abs/2205.14135)

Attention computes:

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

A straightforward implementation stores the scores $QK^\top$ and softmax probabilities as $N\times N$ matrices. For 8,192 tokens, just one such matrix has about 67 million entries—128 MiB at FP16, per head.

FlashAttention processes **small tiles**:

1. Load blocks of $Q,K,V$ into on-chip memory.
2. Compute a tile of attention scores.
3. Incorporate it into a running softmax and weighted sum of $V$.
4. Discard the score tile and continue.

The trick is **online softmax**. For each query, maintain a running maximum $m$, exponential sum $\ell$, and weighted-value sum $u$. Given new scores $s_j$, update:

$$
\begin{aligned}
m' &= \max(m,\max_j s_j)\\
\alpha &= e^{m-m'}\\
\ell' &= \alpha\ell+\sum_j e^{s_j-m'}\\
u' &= \alpha u+\sum_j e^{s_j-m'}v_j
\end{aligned}
$$

The final output is $u/\ell$. Rescaling previous sums by $\alpha$ makes every tile share the same normalization.

During training, the backward pass recomputes score tiles instead of storing the full matrix. **Dense attention still requires quadratic computation**, but its memory footprint becomes linear in sequence length for fixed head dimensions, with much less traffic between HBM and the SMs. [Algorithm and memory analysis](https://arxiv.org/html/2205.14135v2#S3)

## Differences Between FlashAttention Versions

**Each version tackles the next GPU bottleneck:** FA1 reduces HBM traffic, FA2 improves work distribution, FA3 overlaps operations on Hopper, and FA4 addresses Blackwell's softmax and shared-memory bottlenecks.

All retain the tiled attention approach described above.

| Version | Main problem addressed | Main change | Hardware focus |
|---|---|---|---|
| **FA1** | Huge attention matrices moving through HBM | Tiling, online softmax, recomputation | GPU memory hierarchy |
| **FA2** | Idle SMs and excessive communication between warps | Better division of work, fewer non-matmul operations | Especially A100 |
| **FA3** | Memory transfers, matmul, and softmax insufficiently overlapped | Asynchronous pipelines and specialized warps | Hopper, especially H100 |
| **FA4** | Tensor Cores outpace exponential units and shared memory | New pipelines, software exponentials, tensor-memory reuse | Blackwell, especially B200 |

Sources: [FA1](https://arxiv.org/abs/2205.14135), [FA2](https://arxiv.org/abs/2307.08691), [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451).

### FlashAttention-2: Distribute the Work Better

Avoiding HBM traffic does not automatically keep every SM busy. FA2 splits a single attention head's computation across more thread blocks, improving occupancy when batch size or head count is small. Inside each block, it changes how warps divide the work to reduce shared-memory communication. It also reduces softmax-related bookkeeping.

The paper reports approximately **2× faster attention than FA1**. [FA2 paper](https://arxiv.org/abs/2307.08691)

### FlashAttention-3: Execute Different Stages Simultaneously

Hopper provides asynchronous Tensor Core operations and a Tensor Memory Accelerator (TMA) for data transfers. FA3 assigns warps different responsibilities and pipelines tiles so that memory movement, matrix multiplication, and softmax overlap.

Conceptually, different hardware units can work on different tiles at the same time:

```text
Memory transfers:      load upcoming tile
Tensor Cores:          multiply current tile
Other compute units:   softmax another tile
```

FA3 also adds FP8 techniques with quantization-error mitigation. Its paper reports **1.5–2× faster FP16 attention than FA2 on H100**. [FA3 paper](https://arxiv.org/abs/2407.08608)

### FlashAttention-4: Keep Blackwell's Faster Tensor Cores Supplied

On Blackwell, matrix-multiply throughput grows much faster than exponential throughput or shared-memory bandwidth. Softmax can therefore hold up the forward pass, while shared-memory traffic limits the backward pass.

FA4 addresses these with:

- **Software exponentials:** distribute exponential calculations between dedicated hardware and polynomial evaluation on ordinary arithmetic units.
- **Less rescaling:** conditionally perform online-softmax rescaling to reduce bookkeeping.
- **Tensor memory (TMEM):** retain intermediates in Blackwell's dedicated Tensor Core storage, reducing shared-memory traffic.
- **Cooperating CTA pairs and deeper pipelines:** process larger tiles and overlap more work.

The authors report up to **1,605 TFLOP/s in BF16 on B200**. That is a different GPU and benchmark setup from FA3's H100 results, so these numbers do not establish a direct generation-to-generation speedup. [FA4 authors' explanation](https://www.together.ai/blog/flashattention-4)

These versions still perform dense attention with **quadratic arithmetic work**. Their gains come from executing that work more efficiently; FP8 and approximate exponential implementations also introduce numerical differences.
