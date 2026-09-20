# 컴파일러 최적화: 데이터 이동과 연산 배치

> **한 줄 답.** 컴파일러는 데이터 이동, 메모리 배치, 연산 스케줄링을 최적화하여 주어진 하드웨어의 실제 성능을 Roofline ceiling에 가깝게 끌어올린다.

Roofline 모델에서 컴파일러 최적화는 데이터 이동량과 연산기 활용률을 개선하는 역할을 한다.

컴파일러의 핵심 역할은 다음과 같다.

$$
\boxed{\text{같은 모델을 하드웨어에서 더 적게 움직이고, 더 잘 재사용하고, 더 효율적으로 계산하게 만드는 것}}
$$

즉 컴파일러는 하드웨어의 **HBM bandwidth나 Tensor Core 개수 자체를 늘리지는 못하지만**, 주어진 하드웨어에서 실제 성능을 Roofline ceiling에 더 가깝게 끌어올린다.

---

## 전체 그림

| 병목                      | 컴파일러가 최적화하는 문제      | 대표 방법                                                 |
| ----------------------- | ------------------- | ----------------------------------------------------- |
| Memory Bound            | 불필요한 memory traffic | Fusion, tiling, layout, buffer reuse                  |
| Compute Bound           | 연산기 utilization 부족  | Kernel selection, vectorization, tensorization        |
| Communication Bound     | GPU/NPU 간 통신 과다     | communication fusion, overlap, partitioning           |
| Memory Capacity         | tensor/KV가 너무 큼     | buffer planning, recomputation, quantization lowering |
| Launch/Control overhead | kernel이 너무 잘게 나뉨    | graph fusion, static scheduling                       |

---

## 1. Memory Bound 최적화: Operator Fusion

메모리 이동량 감소는 컴파일러의 핵심 최적화 영역 중 하나다.

Roofline에서

$$
AI=\frac{FLOPs}{Bytes}
$$

이므로 컴파일러는 주로 **Bytes를 줄인다.**

Fusion 적용 전:

```text
MatMul
 ↓
HBM write
 ↓
Bias
 ↓
HBM write
 ↓
Activation
 ↓
HBM write
```

Fusion 적용 후:

```text
MatMul → Bias → Activation
```

중간 결과를 register/SRAM에 유지한다.

따라서

$$
Memory\ traffic \downarrow
$$

대표적인 fusion 대상은 다음과 같다.

- MatMul + Bias
- MatMul + Activation
- RMSNorm + residual
- QKV projection
- RoPE
- Attention subgraph
- SwiGLU

---

## 2. Tiling

큰 tensor를 그대로 처리하면 SRAM에 들어가지 않는다.

예를 들어

$$
C=A\times B
$$

를 전체 matrix 단위로 계산하는 대신:

```text
A tile
B tile
  ↓
SRAM
  ↓
MAC array
```

처럼 조각내서 계산한다.

컴파일러는 타일 크기와 실행 순서를 결정한다.

예:

$$
M \times N \times K
$$

에 대해

```text
tile_m = 128
tile_n = 128
tile_k = 32
```

같은 tile size를 선택한다.

좋은 tiling은:

$$
Data\ reuse \uparrow
$$

$$
HBM\ access \downarrow
$$

를 만든다.

타일링은 특히 NPU 컴파일러에서 중요하다.

---

## 3. Memory Layout 최적화

같은 tensor라도 memory layout에 따라 성능이 크게 달라진다.

예:

```text
NCHW
NHWC
blocked layout
```

또는:

```text
[row][column]

vs

[tile][tile]
```

컴파일러가 하드웨어에 맞는 layout으로 바꿀 수 있다.

예를 들어 Tensor Core가 특정 형태를 선호한다면:

$$
Tensor \rightarrow TensorCore-friendly\ layout
$$

로 변환한다.

주요 목표는 다음과 같다.

- contiguous access
- coalesced access
- bank conflict 감소
- vector load 가능

---

## 4. Buffer Allocation / Memory Planning

컴파일러는 tensor를 어디에 둘지도 정할 수 있다.

예:

```text
HBM
L2
SRAM
Register
```

예를 들어:

```text
Tensor A → SRAM
Tensor B → SRAM
Intermediate → Register
Output → HBM
```

처럼 memory hierarchy에 배치한다.

SRAM 용량이 제한되므로, 유지할 데이터와 내보낼 데이터를 결정해야 한다.

이는 NPU 컴파일러의 주요 스케줄링 문제다.

---

## 5. Buffer Reuse

예를 들어:

```text
Tensor A
 ↓
사용 완료

Tensor B
```

라면 A가 쓰던 memory를 B가 재사용할 수 있다.

즉:

```text
buffer 1 → Tensor A
        → Tensor B
        → Tensor C
```

처럼 재사용한다.

그 결과 최대 메모리 사용량이 감소한다.

$$
Peak\ Memory\ Usage \downarrow
$$

LLM에서는 activation이나 intermediate buffer 관리에 중요하다.

---

## 6. Prefetch / Double Buffering

Prefetch와 double buffering의 실행 방식은 다음과 같다.

```text
Compute tile N

동시에

Load tile N+1
```

이 실행 방식도 컴파일러 스케줄링의 대상이다.

컴파일러가 DMA와 compute를 배치해서:

```text
Load A
Compute A + Load B
Compute B + Load C
Compute C
```

처럼 schedule할 수 있다.

즉:

$$
Memory\ latency
$$

를

$$
Compute
$$

뒤에 숨긴다.

이 기법은 NPU 컴파일러에서 특히 중요하다.

---

## 7. Compute Bound 최적화: Kernel Selection

Compute-bound 최적화에서는 FLOPs와 함께 연산기 활용률을 확인해야 한다.

같은 연산도 여러 kernel이 있을 수 있다.

예:

```text
MatMul
```

에 대해:

```text
kernel A → small batch
kernel B → large batch
kernel C → FP8
kernel D → INT8
```

컴파일러가 shape과 hardware를 보고 가장 적절한 kernel을 고른다.

---

## 8. Vectorization

예를 들어 CPU/NPU가 한 번에 8개의 값을 계산할 수 있다면:

기존:

```text
a0*b0
a1*b1
a2*b2
...
```

컴파일러가:

```text
[a0..a7] × [b0..b7]
```

로 바꾼다.

즉 SIMD/vector instruction을 사용한다.

$$
Compute\ utilization \uparrow
$$

---

## 9. Tensorization

AI accelerator에서는 vectorization보다 더 큰 단위가 있다.

예를 들어 Tensor Core가:

$$
16\times16\times16
$$

matrix multiply를 한 instruction으로 처리한다면, 컴파일러는 일반적인 loop:

```text
for i
 for j
  for k
```

를 찾아서

```text
TensorCore MMA
```

instruction으로 바꾼다.

이를 **tensorization**이라 한다.

TVM 같은 compiler에서 매우 중요한 개념이다.

---

## 10. Loop Transformation

컴파일러가 loop 순서를 바꾸는 것도 중요하다.

예:

```text
for i
 for j
  for k
```

를

```text
for i_tile
 for k_tile
  for j_tile
```

처럼 변경한다.

대표적인 기법은 다음과 같다.

- loop tiling
- loop interchange
- loop unrolling
- vectorization
- parallelization

목표는 캐시 지역성과 연산기 활용률을 높이는 것이다.

$$
Cache\ locality \uparrow
$$

$$
Compute\ utilization \uparrow
$$

---

## 11. Quantization lowering

Quantization은 모델 알고리즘 영역처럼 보이지만 compiler 역할도 크다.

예를 들어 모델이:

```text
FP16 MatMul
```

이라도 compiler가:

```text
INT8 MatMul
+
scale
+
dequant
```

으로 lower할 수 있다.

또는:

```text
INT4 weights
→ packed representation
→ hardware INT4 instruction
```

으로 변환한다.

즉 compiler가:

$$
Model\ representation
\rightarrow
Hardware\ instruction
$$

을 연결한다.

---

## 12. Sparsity 활용

모델 weight에 sparsity가 있더라도 hardware가 자동으로 빠르게 계산하는 것은 아니다.

Compiler가:

```text
dense matmul
```

을

```text
sparse matmul
```

로 바꾸거나, hardware sparse instruction을 사용하도록 내려야 한다.

예:

$$
2:4\ structured\ sparsity
$$

를 지원하는 accelerator라면 compiler가 이를 검출하고 sparse Tensor Core instruction을 사용한다.

---

## 13. Kernel Fusion vs Graph Fusion

Graph fusion과 kernel fusion은 적용 수준이 다르다.

### Graph-level

```text
MatMul
 ↓
Bias
 ↓
ReLU
```

를 하나의 graph node로 묶는다.

### Kernel-level

Kernel-level fusion은 실제로 하나의 커널을 생성한다.

```text
fused_matmul_bias_relu()
```

즉 compiler pipeline은 대략:

```text
Graph IR
   ↓
Fusion
   ↓
Tensor IR
   ↓
Tiling
   ↓
Scheduling
   ↓
Kernel IR
   ↓
Machine code
```

---

## 14. Communication Bound 최적화

Multi-GPU/NPU에서는 compiler가:

```text
MatMul
 ↓
AllReduce
 ↓
MatMul
```

을 그대로 실행하는 대신:

```text
MatMul chunk 1
    ↓
AllReduce chunk 1

동시에

MatMul chunk 2
```

처럼 communication과 compute를 overlap할 수 있다.

즉:

$$
T_{total}
\neq
T_{compute}+T_{communication}
$$

이며, 이상적으로는

$$
T_{total}
\approx
\max(T_{compute},T_{communication})
$$

에 가깝게 만든다.

---

## 15. 병렬화 분할 (Parallelism Partitioning)

모델을 여러 device에 나눌 때:

```text
Tensor Parallel
Pipeline Parallel
Expert Parallel
```

을 어떻게 배치할지도 compiler/runtime가 담당할 수 있다.

예:

```text
Layer 0-10 → GPU0
Layer 11-20 → GPU1
```

또는:

```text
MatMul shard → GPU0/GPU1/GPU2/GPU3
```

이 과정은:

$$
Compute + Memory + Communication
$$

세 가지를 동시에 최적화해야 하는 문제이다.

---

## 16. 컴파일러 최적화의 한계

컴파일러는:

$$
HBM\ bandwidth
$$

자체를 늘리지 못한다.

또한:

$$
TensorCore\ count
$$

도 늘릴 수 없다.

예를 들어 hardware가:

```text
1 TB/s HBM
100 TFLOPS
```

라면 compiler가 이를

```text
2 TB/s
200 TFLOPS
```

로 만들 수는 없다.

대신 실제 성능이:

```text
20 TFLOPS
```

밖에 안 나오고 있었다면 compiler가:

```text
70~90 TFLOPS
```

에 가깝게 만드는 역할을 한다.

즉:

$$
\boxed{
Compiler = hardware utilization optimizer
}
$$

로 요약할 수 있다.

---

## 17. Roofline 관점의 최적화 목표

### Memory-bound workload

컴파일러 목표:

$$
Bytes \downarrow
$$

대표적인 방법은 다음과 같다.

- Fusion
- Tiling
- Buffer reuse
- Layout optimization
- Prefetch
- Double buffering

결과:

$$
Arithmetic\ Intensity \uparrow
$$

즉 Roofline에서 **오른쪽으로 이동**한다.

---

### Compute-bound workload

컴파일러 목표:

$$
Actual\ FLOPS \rightarrow Peak\ FLOPS
$$

대표적인 방법은 다음과 같다.

- Tensorization
- Vectorization
- Kernel selection
- Loop optimization
- Instruction scheduling

즉 Roofline의 위쪽 ceiling에 더 가까이 간다.

---

### Communication-bound workload

컴파일러 목표:

$$
Communication \downarrow
$$

또는:

$$
Communication \parallel Compute
$$

대표적인 방법은 다음과 같다.

- Partitioning
- Collective fusion
- Communication overlap
- Placement

---

## 18. NPU 컴파일러의 최적화 파이프라인

```text
               AI Model
                  │
                  ▼
              Graph IR
                  │
       ┌──────────┴───────────┐
       │                      │
      Fusion              Quantization
       │                      │
       └──────────┬───────────┘
                  ▼
              Tensor IR
                  │
       ┌──────────┼─────────────┐
       │          │             │
     Tiling     Layout       Buffering
       │          │             │
       └──────────┼─────────────┘
                  ▼
              Schedule
                  │
         ┌────────┴─────────┐
         │                  │
     Tensorization      Prefetch/DMA
         │                  │
         └────────┬─────────┘
                  ▼
             Machine Code
                  │
                  ▼
           NPU / GPU Hardware
```

컴파일러의 역할은 다음과 같이 요약된다.

$$
\boxed{
\text{Compiler는 데이터 이동, 연산 배치, 메모리 배치, instruction mapping을 최적화한다.}
}
$$

**Tiling → Buffer allocation → Operator fusion → Tensorization → DMA/compute scheduling**은 NPU 컴파일러의 주요 학습 순서다. 이 다섯 영역에서 하드웨어 구조와 컴파일러 최적화가 직접 연결된다.
