# 추론 최적화: Compute Bound와 Memory Bound

> **한 줄 답.** Memory-bound에서는 데이터 이동량을 줄이고 재사용률을 높이며, compute-bound에서는 연산량을 줄이고 연산기 활용률을 높인다.

Roofline 모델은 추론 단계의 병목에 따라 최적화 방향을 구분한다.

Arithmetic intensity는 다음과 같이 정의된다.

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Traffic}}
$$

이에 따라 병목은 다음 두 종류로 구분된다.

- **Memory Bound**: 데이터를 가져오는 시간이 병목 → **메모리 이동량을 줄이거나 재사용률을 높이는 최적화**
- **Compute Bound**: 연산 자체가 병목 → **연산량을 줄이거나 GPU/NPU 연산 효율을 높이는 최적화**

## 한눈에 정리

| 구간                  | 병목                 | 핵심 목표                    | 대표 최적화                                                                 |
| ------------------- | ------------------ | ------------------------ | ---------------------------------------------------------------------- |
| Memory Bound        | HBM/DRAM bandwidth | Memory traffic ↓         | KV cache, quantization, kernel fusion, batching, cache reuse           |
| Compute Bound       | Tensor Core / ALU  | FLOPs ↓ 또는 utilization ↑ | FlashAttention, GEMM 최적화, speculative decoding, sparsity, quantization |
| Communication Bound | GPU↔GPU / network  | communication ↓          | TP/PP 최적화, collective 최적화, KV placement                                |

Inference에서는 특히 **Prefill과 Decode가 서로 다른 영역에 들어가는 경우가 많다.**

---

## 1. Memory Bound 최적화

Memory-bound 영역은 대략 다음 조건에 해당한다.

$$
\frac{\text{FLOPs}}{\text{Bytes}}
<
\frac{\text{Peak FLOPs}}{\text{Memory Bandwidth}}
$$

즉 GPU는 계산할 능력이 남아 있는데 **데이터가 늦게 들어와서 기다리는 상황**이다.

LLM inference에서는 특히 **Decode 단계**에서 매우 자주 발생한다.

예를 들어 token 하나를 생성할 때마다 weight를 HBM에서 계속 읽어야 하기 때문이다.

### ① Weight memory traffic 줄이기

Weight quantization은 가중치의 데이터 이동량을 직접 줄인다. 예를 들어 다음과 같이 저장 정밀도를 낮춘다.

$$
FP16 \rightarrow FP8 \rightarrow INT8 \rightarrow INT4
$$

예를 들어 70B 모델이면 대략:

- FP16 → 140 GB
- INT8 → 70 GB
- INT4 → 35 GB

따라서 token 하나 생성할 때 HBM에서 읽어야 하는 데이터가 크게 줄어든다.

**Quantization은 모델의 메모리 사용량을 줄이는 동시에 메모리 대역폭 부담을 낮춘다.**

---

### ② KV Cache 줄이기

Decode가 길어질수록 KV cache 읽기가 커진다.

주요 방법은 다음과 같다.

- KV cache quantization
- GQA
- MQA
- Sliding Window Attention
- KV cache eviction
- KV cache compression

예를 들어 MHA에서는 여러 attention head가 각자의 K,V를 가지지만, GQA/MQA에서는 여러 Q head가 KV를 공유한다.

따라서:

$$
KV\ traffic \downarrow
$$

가 된다.

---

### ③ Cache hit / Prefix caching

예를 들어 system prompt가:

```text
You are a coding assistant...
repository information...
tool descriptions...
```

처럼 반복된다면 이를 매번 Prefill하지 않고 기존 KV를 재사용할 수 있다.

그러면

$$
Memory\ traffic + Compute
$$

를 동시에 줄일 수 있다.

특히 coding agent처럼 긴 prefix가 반복되는 workload에서 효과가 크다.

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

이 발생할 수 있다.

이를 fused kernel로 만들면:

```text
HBM
 ↓
MatMul → Bias → Activation → Norm
 ↓
HBM
```

으로 바뀐다.

즉:

$$
Memory\ Round\ Trip \downarrow
$$

대표적으로:

- RMSNorm fusion
- SwiGLU fusion
- Attention fusion
- RoPE fusion

---

### ⑤ Continuous batching / Larger batch

Decode에서는 batch가 작으면 weight를 읽어놓고 계산을 조금밖에 하지 못한다.

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

이 된다.

즉 같은 memory traffic으로 더 많은 FLOPs를 수행하므로

$$
AI = \frac{FLOPs}{Bytes}
$$

가 증가한다.

이는 Roofline에서 **오른쪽으로 이동**하는 것에 해당한다.

그래서 continuous batching이 inference throughput에 매우 중요하다.

---

### ⑥ Memory layout / data movement 최적화

예:

- contiguous memory
- coalesced memory access
- paged KV cache
- PagedAttention
- Tensor layout 변경
- cache locality 개선

이 기법들의 공통 목표는 유효 메모리 대역폭을 높이는 것이다.

$$
Effective\ Bandwidth \uparrow
$$

vLLM의 PagedAttention 역시 이런 관점에서 이해할 수 있다.

---

## 2. Compute Bound 최적화

Compute-bound 영역의 조건은 다음과 같다.

$$
AI > Ridge\ Point
$$

데이터 공급은 충분하지만 Tensor Core의 연산 처리량이 병목인 상태다.

Prefill에서 sequence가 길거나 batch가 클 때 자주 나타난다.

이때는 메모리 bandwidth를 더 높여도 성능 증가가 거의 없다.

---

### ① FLOPs 자체를 줄인다

필요한 연산량 자체를 줄이는 접근이다.

예를 들어 attention:

$$
O(N^2d)
$$

연산량을 줄이는 대표적인 방법은 다음과 같다.

- Sliding Window Attention
- Sparse Attention
- Local Attention
- Linear Attention
- Token pruning

모델 자체를 줄일 수도 있다.

- smaller model
- MoE
- layer dropping
- speculative decoding

---

### ② 더 낮은 precision 사용

Quantization은 compute-bound 영역에서도 유효하다.

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

로 내려가면서 Tensor Core throughput이 증가할 수 있다.

예:

$$
FP16: 100\ TFLOPS
$$

$$
FP8: 200\ TFLOPS
$$

라면 Roofline의 ceiling 자체가 올라간다.

즉 compute-bound 영역에서 성능이 올라간다.

---

## 3. GEMM 최적화

Transformer 연산 대부분은 결국:

$$
C = AB
$$

GEMM이다.

Compute-bound에서 중요한 것은 Tensor Core utilization이다.

예를 들어:

```text
bad tile size
poor occupancy
small GEMM
```

이면 theoretical FLOPs를 제대로 사용하지 못한다.

최적화:

- tile size
- warp scheduling
- tensor core utilization
- persistent kernel
- CUDA Graph
- CUTLASS kernel tuning
- fused GEMM

등을 사용할 수 있다.

---

## 4. FlashAttention

FlashAttention은 메모리 I/O를 줄이는 attention 알고리즘이다.

Attention의 일반적인 구현은:

$$
QK^T
$$

중간 matrix를 HBM에 저장했다가 다시 읽는다.

FlashAttention은 tiling으로 이 intermediate 결과를 SRAM에 유지한다.

그래서 원래는 주로

$$
Memory\ IO \downarrow
$$

최적화이다.

하지만 GPU utilization도 높이기 때문에 compute efficiency 역시 좋아질 수 있다.

즉 **Roofline에서 AI를 높이는 대표적인 알고리즘**이다.

---

## 5. Speculative Decoding

Speculative decoding은 후보 토큰을 묶어 검증하여 decode 지연 시간을 줄이는 기법이다.

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

즉 expensive large model forward pass 횟수를 줄인다.

결과적으로

$$
Large\ Model\ Compute \downarrow
$$

시킬 수 있다.

특히 decode latency를 줄이는 데 많이 사용된다.

---

## 6. Parallelism 최적화

큰 모델에서는 Roofline을 조금 확장해야 한다.

실제 inference에서는:

$$
Performance =
\min(
Compute,
Memory,
Communication
)
$$

이기 때문이다.

예를 들어 Tensor Parallelism:

```text
GPU0 ─┐
GPU1 ─┼─ AllReduce
GPU2 ─┤
GPU3 ─┘
```

에서는 NVLink/PCIe communication이 bottleneck이 될 수 있다.

이 경우:

- Tensor Parallel degree 조절
- Pipeline Parallel
- Expert Parallel
- AllReduce fusion
- communication-computation overlap

등이 중요하다.

단순화한 모델에서는 communication을 memory bandwidth와 비슷한 데이터 이동 비용으로 묶어 생각할 수도 있지만, **실제 시스템 분석에서는 HBM bandwidth와 GPU 간 communication bandwidth를 구분한다.**

---

## 7. Prefill과 Decode의 최적화

LLM inference에서는 이 구분이 특히 중요하다.

### Prefill

긴 prompt:

$$
X_{1:N}
$$

을 한꺼번에 처리한다.

Matrix multiplication이 크기 때문에 AI가 높아진다.

대체로:

$$
\boxed{Compute\ Bound}
$$

영역에 해당하기 쉽다.

따라서:

- FlashAttention
- Tensor Core 활용
- FP8
- GEMM 최적화
- kernel fusion
- prompt caching

이 중요하다.

---

### Decode

매번 token 하나:

$$
x_t \rightarrow x_{t+1}
$$

를 생성한다.

GEMM이 작아지고 weight를 계속 읽어야 한다.

그래서:

$$
\boxed{Memory\ Bound}
$$

영역에 해당하기 쉽다.

주요 최적화는 다음과 같다.

- Weight quantization
- KV cache optimization
- Continuous batching
- GQA/MQA
- PagedAttention
- Prefix caching

---

## 8. Roofline에서의 성능 변화

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

즉 **오른쪽으로 이동**이다.

대표적인 방법은 다음과 같다.

- Memory traffic ↓
- Data reuse ↑
- Batch ↑
- Cache reuse ↑
- Quantization

---

### Compute Bound라면

이미 오른쪽에 있기 때문에 더 오른쪽으로 가는 것이 크게 도움이 되지 않는다.

목표는:

$$
Peak\ Compute \uparrow
$$

또는

$$
Required\ FLOPs \downarrow
$$

대표적인 방법은 다음과 같다.

- Tensor Core utilization ↑
- Lower precision
- Better GEMM kernel
- FLOPs ↓
- Speculative decoding
- Sparse computation

---

## 9. 요약: 병목별 최적화 방향

| 문제                      | 점검 항목                            | 최적화 방향                                      |
| ----------------------- | ----------------------------- | ------------------------------------------- |
| **Memory Bound**        | 동일 데이터의 반복 읽기        | Quantization, cache, fusion, batching       |
| **Compute Bound**       | 연산량과 연산기 활용률 | FP8, Tensor Core, GEMM, sparsity            |
| **Communication Bound** | GPU 간 통신량           | TP degree, overlap, collective optimization |

LLM 추론의 최적화 방향은 대략 다음과 같이 요약된다.

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
