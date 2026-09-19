Inference 단계에서 Roofline 모델 기준으로 보면 최적화 방향은 꽤 명확하게 나눌 수 있습니다.

핵심은:

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Traffic}}
$$

이고,

* **Memory Bound**: 데이터를 가져오는 시간이 병목 → **메모리 이동량을 줄이거나 재사용률을 높이는 최적화**
* **Compute Bound**: 연산 자체가 병목 → **연산량을 줄이거나 GPU/NPU 연산 효율을 높이는 최적화**

라고 보면 됩니다.

## 한눈에 정리

| 구간                  | 병목                 | 핵심 목표                    | 대표 최적화                                                                 |
| ------------------- | ------------------ | ------------------------ | ---------------------------------------------------------------------- |
| Memory Bound        | HBM/DRAM bandwidth | Memory traffic ↓         | KV cache, quantization, kernel fusion, batching, cache reuse           |
| Compute Bound       | Tensor Core / ALU  | FLOPs ↓ 또는 utilization ↑ | FlashAttention, GEMM 최적화, speculative decoding, sparsity, quantization |
| Communication Bound | GPU↔GPU / network  | communication ↓          | TP/PP 최적화, collective 최적화, KV placement                                |

Inference에서는 특히 **Prefill과 Decode가 서로 다른 영역에 들어가는 경우가 많습니다.**

---

# 1. Memory Bound일 때

Memory bound라는 것은 대략

$$
\frac{\text{FLOPs}}{\text{Bytes}}
<
\frac{\text{Peak FLOPs}}{\text{Memory Bandwidth}}
$$

인 상태입니다.

즉 GPU는 계산할 능력이 남아 있는데 **데이터가 늦게 들어와서 기다리는 상황**입니다.

LLM inference에서는 특히 **Decode 단계**에서 매우 자주 발생합니다.

예를 들어 token 하나를 생성할 때마다 weight를 HBM에서 계속 읽어야 하기 때문입니다.

### ① Weight memory traffic 줄이기

가장 직접적인 방법입니다.

FP16 대신:

$$
FP16 \rightarrow FP8 \rightarrow INT8 \rightarrow INT4
$$

처럼 weight를 줄입니다.

예를 들어 70B 모델이면 대략:

* FP16 → 140 GB
* INT8 → 70 GB
* INT4 → 35 GB

따라서 token 하나 생성할 때 HBM에서 읽어야 하는 데이터가 크게 줄어듭니다.

그래서 **quantization은 단순히 모델을 메모리에 넣기 위한 기술만이 아니라 memory bandwidth optimization**이기도 합니다.

---

### ② KV Cache 줄이기

Decode가 길어질수록 KV cache 읽기가 커집니다.

주요 방법:

* KV cache quantization
* GQA
* MQA
* Sliding Window Attention
* KV cache eviction
* KV cache compression

예를 들어 MHA에서는 여러 attention head가 각자의 K,V를 가지지만,

GQA/MQA에서는 여러 Q head가 KV를 공유합니다.

따라서:

$$
KV\ traffic \downarrow
$$

가 됩니다.

---

### ③ Cache hit / Prefix caching

이전 대화에서 이야기했던 부분입니다.

예를 들어 system prompt가:

```text
You are a coding assistant...
repository information...
tool descriptions...
```

처럼 반복된다면 이를 매번 Prefill하지 않고 기존 KV를 재사용할 수 있습니다.

그러면

$$
Memory\ traffic + Compute
$$

를 동시에 줄일 수 있습니다.

특히 coding agent처럼 긴 prefix가 반복되는 workload에서 효과가 큽니다.

---

### ④ Kernel Fusion

예를 들어 기존에는

```text
MatMul
 ↓
Bias
 ↓
Activation
 ↓
Normalization
```

각 단계마다

```text
HBM → GPU
GPU → HBM
```

이 발생할 수 있습니다.

이를 fused kernel로 만들면:

```text
HBM
 ↓
MatMul → Bias → Activation → Norm
 ↓
HBM
```

으로 바뀝니다.

즉:

$$
Memory\ Round\ Trip \downarrow
$$

대표적으로:

* RMSNorm fusion
* SwiGLU fusion
* Attention fusion
* RoPE fusion

등이 있습니다.

---

### ⑤ Continuous batching / Larger batch

Decode에서는 batch가 작으면 weight를 읽어놓고 계산을 조금밖에 하지 못합니다.

예를 들어 batch=1이면:

```text
weight 100MB 읽기
→ token 1개 계산
```

하지만 batch=32라면:

```text
weight 100MB 읽기
→ token 32개 계산
```

이 됩니다.

즉 같은 memory traffic으로 더 많은 FLOPs를 수행하므로

$$
AI = \frac{FLOPs}{Bytes}
$$

가 증가합니다.

Roofline에서 보면 **오른쪽으로 이동**하는 것입니다.

그래서 continuous batching이 inference throughput에 매우 중요합니다.

---

### ⑥ Memory layout / data movement 최적화

예:

* contiguous memory
* coalesced memory access
* paged KV cache
* PagedAttention
* Tensor layout 변경
* cache locality 개선

목표는 모두 비슷합니다.

$$
Effective\ Bandwidth \uparrow
$$

입니다.

vLLM의 PagedAttention 역시 이런 관점에서 이해할 수 있습니다.

---

# 2. Compute Bound일 때

Compute bound는

$$
AI > Ridge\ Point
$$

인 영역입니다.

데이터 공급은 충분한데,

> Tensor Core가 계산하느라 바쁜 상태

입니다.

Prefill에서 sequence가 길거나 batch가 클 때 자주 나타납니다.

이때는 메모리 bandwidth를 더 높여도 성능 증가가 거의 없습니다.

---

## ① FLOPs 자체를 줄인다

가장 근본적인 방법입니다.

예를 들어 attention:

$$
O(N^2d)
$$

연산량을 줄이는 방법:

* Sliding Window Attention
* Sparse Attention
* Local Attention
* Linear Attention
* Token pruning

등이 있습니다.

모델 자체를 줄일 수도 있습니다.

* smaller model
* MoE
* layer dropping
* speculative decoding

---

## ② 더 낮은 precision 사용

Quantization은 Memory-bound에서만 의미 있는 게 아닙니다.

예를 들어 GPU가 지원한다면:

```text
FP32
 ↓
FP16/BF16
 ↓
FP8
 ↓
INT8
```

로 내려가면서 Tensor Core throughput이 증가할 수 있습니다.

예:

$$
FP16: 100\ TFLOPS
$$

$$
FP8: 200\ TFLOPS
$$

라면 Roofline의 ceiling 자체가 올라갑니다.

즉 compute-bound 영역에서 성능이 올라갑니다.

---

# 3. GEMM 최적화

Transformer 연산 대부분은 결국:

$$
C = AB
$$

GEMM입니다.

Compute-bound에서 중요한 것은 Tensor Core utilization입니다.

예를 들어:

```text
bad tile size
poor occupancy
small GEMM
```

이면 theoretical FLOPs를 제대로 사용하지 못합니다.

최적화:

* tile size
* warp scheduling
* tensor core utilization
* persistent kernel
* CUDA Graph
* CUTLASS kernel tuning
* fused GEMM

등을 사용할 수 있습니다.

---

# 4. FlashAttention

FlashAttention은 조금 특별합니다.

Attention의 일반적인 구현은:

$$
QK^T
$$

중간 matrix를 HBM에 저장했다가 다시 읽습니다.

FlashAttention은 tiling으로 이 intermediate 결과를 SRAM에 유지합니다.

그래서 원래는 주로

$$
Memory\ IO \downarrow
$$

최적화입니다.

하지만 GPU utilization도 높이기 때문에 compute efficiency 역시 좋아질 수 있습니다.

즉 **Roofline에서 AI를 높이는 대표적인 알고리즘**으로 생각하면 좋습니다.

---

# 5. Speculative Decoding

Speculative decoding은 compute-bound / latency optimization 관점에서 재미있는 방법입니다.

기존:

```text
Large Model
token 1
token 2
token 3
token 4
```

Speculative:

```text
Small Model
↓
token 1,2,3,4 후보

Large Model
↓
한 번에 verification
```

즉 expensive large model forward pass 횟수를 줄입니다.

결과적으로

$$
Large\ Model\ Compute \downarrow
$$

시킬 수 있습니다.

특히 decode latency를 줄이는 데 많이 사용됩니다.

---

# 6. Parallelism 최적화

큰 모델에서는 Roofline을 조금 확장해야 합니다.

실제 inference에서는:

$$
Performance =
\min(
Compute,
Memory,
Communication
)
$$

이기 때문입니다.

예를 들어 Tensor Parallelism:

```text
GPU0 ─┐
GPU1 ─┼─ AllReduce
GPU2 ─┤
GPU3 ─┘
```

에서는 NVLink/PCIe communication이 bottleneck이 될 수 있습니다.

이 경우:

* Tensor Parallel degree 조절
* Pipeline Parallel
* Expert Parallel
* AllReduce fusion
* communication-computation overlap

등이 중요합니다.

이전 질문에서 말한 것처럼 아주 단순한 모델에서는 communication을 memory bandwidth와 비슷한 데이터 이동 비용으로 묶어 생각할 수도 있지만, **실제 시스템 분석에서는 HBM bandwidth와 GPU 간 communication bandwidth를 따로 보는 편이 좋습니다.**

---

# Prefill vs Decode로 다시 보면

LLM inference에서는 이 구분이 특히 중요합니다.

### Prefill

긴 prompt:

$$
X_{1:N}
$$

을 한꺼번에 처리합니다.

Matrix multiplication이 크기 때문에 AI가 높아집니다.

대체로:

$$
\boxed{Compute\ Bound}
$$

쪽으로 가기 쉽습니다.

따라서:

* FlashAttention
* Tensor Core 활용
* FP8
* GEMM 최적화
* kernel fusion
* prompt caching

이 중요합니다.

---

### Decode

매번 token 하나:

$$
x_t \rightarrow x_{t+1}
$$

를 생성합니다.

GEMM이 작아지고 weight를 계속 읽어야 합니다.

그래서:

$$
\boxed{Memory\ Bound}
$$

쪽으로 가기 쉽습니다.

중요한 최적화:

* Weight quantization
* KV cache optimization
* Continuous batching
* GQA/MQA
* PagedAttention
* Prefix caching

입니다.

---

# Roofline에서 움직이는 방향으로 생각하면 더 간단함

Roofline:

```text
Performance
  ↑
  │                     ───────────── Compute Ceiling
  │                  /
  │               /
  │            /
  │         /
  │      /
  │   /
  └────────────────────────────→ Arithmetic Intensity
       Memory Bound      Compute Bound
                    ↑
                  Ridge
```

### Memory Bound라면

목표는 주로:

$$
AI=\frac{FLOPs}{Bytes}\uparrow
$$

즉 **오른쪽으로 이동**입니다.

방법:

> Memory traffic ↓
> Data reuse ↑
> Batch ↑
> Cache reuse ↑
> Quantization

---

### Compute Bound라면

이미 오른쪽에 있기 때문에 더 오른쪽으로 가는 것이 크게 도움이 되지 않습니다.

목표는:

$$
Peak\ Compute \uparrow
$$

또는

$$
Required\ FLOPs \downarrow
$$

입니다.

방법:

> Tensor Core utilization ↑
> Lower precision
> Better GEMM kernel
> FLOPs ↓
> Speculative decoding
> Sparse computation

---

## 실전적으로 외우면

| 문제                      | 질문                            | 최적화 방향                                      |
| ----------------------- | ----------------------------- | ------------------------------------------- |
| **Memory Bound**        | "같은 데이터를 너무 많이 읽고 있나?"        | Quantization, cache, fusion, batching       |
| **Compute Bound**       | "계산을 너무 많이 하나 / 연산기를 제대로 쓰나?" | FP8, Tensor Core, GEMM, sparsity            |
| **Communication Bound** | "GPU끼리 너무 많이 통신하나?"           | TP degree, overlap, collective optimization |

그리고 LLM inference에서는 아주 거칠게 말하면:

$$
\boxed{
Prefill \rightarrow Compute\ optimization
}
$$

$$
\boxed{
Decode \rightarrow Memory\ optimization
}
$$

이라는 관점부터 시작하면 대부분의 inference optimization 기술을 꽤 잘 분류할 수 있습니다.
