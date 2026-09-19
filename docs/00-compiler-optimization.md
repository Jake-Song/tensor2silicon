좋아. 여기까지의 Roofline 관점에 **컴파일러 층**을 끼워 넣으면 가장 깔끔하게 정리할 수 있습니다.

컴파일러가 하는 핵심 역할은 결국

$$
\boxed{\text{같은 모델을 하드웨어에서 더 적게 움직이고, 더 잘 재사용하고, 더 효율적으로 계산하게 만드는 것}}
$$

입니다.

즉 컴파일러는 하드웨어의 **HBM bandwidth나 Tensor Core 개수 자체를 늘리지는 못하지만**, 주어진 하드웨어에서 실제 성능을 Roofline ceiling에 더 가깝게 끌어올립니다.

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

# 1. Memory Bound에서 컴파일러가 가장 많이 하는 일

이 부분이 컴파일러의 핵심 영역 중 하나입니다.

Roofline에서

$$
AI=\frac{FLOPs}{Bytes}
$$

이므로 컴파일러는 주로 **Bytes를 줄이려 합니다.**

## ① Operator Fusion

원래:

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

컴파일러가 fusion하면:

```text
MatMul → Bias → Activation
```

중간 결과를 register/SRAM에 유지합니다.

따라서

$$
Memory\ traffic \downarrow
$$

대표적인 fusion:

* MatMul + Bias
* MatMul + Activation
* RMSNorm + residual
* QKV projection
* RoPE
* Attention subgraph
* SwiGLU

입니다.

---

# 2. Tiling

큰 tensor를 그대로 처리하면 SRAM에 들어가지 않습니다.

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

처럼 조각내서 계산합니다.

이걸 컴파일러가 결정합니다.

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

같은 tile size를 선택합니다.

좋은 tiling은:

$$
Data\ reuse \uparrow
$$

$$
HBM\ access \downarrow
$$

를 만듭니다.

특히 NPU 컴파일러에서는 굉장히 중요합니다.

---

# 3. Memory Layout 최적화

같은 tensor라도 memory layout에 따라 성능이 크게 달라집니다.

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

컴파일러가 하드웨어에 맞는 layout으로 바꿀 수 있습니다.

예를 들어 Tensor Core가 특정 형태를 선호한다면:

$$
Tensor \rightarrow TensorCore-friendly\ layout
$$

로 변환합니다.

목표는:

* contiguous access
* coalesced access
* bank conflict 감소
* vector load 가능

입니다.

---

# 4. Buffer Allocation / Memory Planning

컴파일러는 tensor를 어디에 둘지도 정할 수 있습니다.

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

처럼 memory hierarchy에 배치합니다.

중요한 문제는:

> SRAM은 작기 때문에 무엇을 남겨두고 무엇을 내보낼 것인가?

입니다.

이게 NPU compiler에서 굉장히 중요한 scheduling 문제입니다.

---

# 5. Buffer Reuse

예를 들어:

```text
Tensor A
 ↓
사용 완료

Tensor B
```

라면 A가 쓰던 memory를 B가 재사용할 수 있습니다.

즉:

```text
buffer 1 → Tensor A
        → Tensor B
        → Tensor C
```

처럼 합니다.

결과적으로:

$$
Peak\ Memory\ Usage \downarrow
$$

합니다.

LLM에서는 activation이나 intermediate buffer 관리에 중요합니다.

---

# 6. Prefetch / Double Buffering

앞에서 이야기한:

```text
Compute tile N

동시에

Load tile N+1
```

도 compiler scheduling 문제입니다.

컴파일러가 DMA와 compute를 배치해서:

```text
Load A
Compute A + Load B
Compute B + Load C
Compute C
```

처럼 schedule할 수 있습니다.

즉:

$$
Memory\ latency
$$

를

$$
Compute
$$

뒤에 숨깁니다.

이건 NPU compiler에서 특히 중요합니다.

---

# 7. Compute Bound에서의 Compiler 역할

Compute Bound에서는 FLOPs 자체보다도

> "연산기를 제대로 사용하고 있는가?"

가 중요합니다.

---

## ① Kernel Selection

같은 연산도 여러 kernel이 있을 수 있습니다.

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

컴파일러가 shape과 hardware를 보고 가장 적절한 kernel을 고릅니다.

---

# 8. Vectorization

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

로 바꿉니다.

즉 SIMD/vector instruction을 사용합니다.

$$
Compute\ utilization \uparrow
$$

---

# 9. Tensorization

AI accelerator에서는 vectorization보다 더 큰 단위가 있습니다.

예를 들어 Tensor Core가:

$$
16\times16\times16
$$

matrix multiply를 한 instruction으로 처리한다면,

컴파일러는 일반적인 loop:

```text
for i
 for j
  for k
```

를 찾아서

```text
TensorCore MMA
```

instruction으로 바꿉니다.

이를 보통:

> tensorization

이라고 부릅니다.

TVM 같은 compiler에서 매우 중요한 개념입니다.

---

# 10. Loop Transformation

컴파일러가 loop 순서를 바꾸는 것도 중요합니다.

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

처럼 변경합니다.

대표적인 기법:

* loop tiling
* loop interchange
* loop unrolling
* vectorization
* parallelization

입니다.

목적은 결국:

$$
Cache\ locality \uparrow
$$

$$
Compute\ utilization \uparrow
$$

입니다.

---

# 11. Quantization lowering

Quantization은 모델 알고리즘 영역처럼 보이지만 compiler 역할도 큽니다.

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

으로 lower할 수 있습니다.

또는:

```text
INT4 weights
→ packed representation
→ hardware INT4 instruction
```

으로 변환합니다.

즉 compiler가:

$$
Model\ representation
\rightarrow
Hardware\ instruction
$$

을 연결합니다.

---

# 12. Sparsity 활용

모델 weight에 sparsity가 있더라도 hardware가 자동으로 빠르게 계산하는 것은 아닙니다.

Compiler가:

```text
dense matmul
```

을

```text
sparse matmul
```

로 바꾸거나,

hardware sparse instruction을 사용하도록 내려야 합니다.

예:

$$
2:4\ structured\ sparsity
$$

를 지원하는 accelerator라면

compiler가 이를 검출하고 sparse Tensor Core instruction을 사용합니다.

---

# 13. Kernel Fusion vs Graph Fusion

둘은 약간 다릅니다.

### Graph-level

```text
MatMul
 ↓
Bias
 ↓
ReLU
```

를 하나의 graph node로 묶음.

### Kernel-level

실제로 하나의 kernel로 생성:

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

입니다.

---

# 14. Communication Bound도 Compiler가 다룰 수 있음

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

처럼 communication과 compute를 overlap할 수 있습니다.

즉:

$$
T_{total}
\neq
T_{compute}+T_{communication}
$$

가 아니라 이상적으로

$$
T_{total}
\approx
\max(T_{compute},T_{communication})
$$

에 가깝게 만듭니다.

---

# 15. Parallelism partition도 compiler 문제

모델을 여러 device에 나눌 때:

```text
Tensor Parallel
Pipeline Parallel
Expert Parallel
```

을 어떻게 배치할지도 compiler/runtime가 담당할 수 있습니다.

예:

```text
Layer 0-10 → GPU0
Layer 11-20 → GPU1
```

또는:

```text
MatMul shard → GPU0/GPU1/GPU2/GPU3
```

입니다.

이 과정은:

$$
Compute + Memory + Communication
$$

세 가지를 동시에 최적화해야 하는 문제입니다.

---

# 16. Compiler가 못 하는 것

이것도 중요합니다.

컴파일러는:

$$
HBM\ bandwidth
$$

자체를 늘리지 못합니다.

또한:

$$
TensorCore\ count
$$

도 늘릴 수 없습니다.

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

로 만들 수는 없습니다.

대신 실제 성능이:

```text
20 TFLOPS
```

밖에 안 나오고 있었다면 compiler가:

```text
70~90 TFLOPS
```

에 가깝게 만드는 역할을 합니다.

즉:

$$
\boxed{
Compiler = hardware utilization optimizer
}
$$

라고 이해하면 꽤 정확합니다.

---

# Roofline과 연결하면 가장 중요하게 봐야 할 것

### Memory-bound workload

컴파일러 목표:

$$
Bytes \downarrow
$$

방법:

> Fusion
> Tiling
> Buffer reuse
> Layout optimization
> Prefetch
> Double buffering

결과:

$$
Arithmetic\ Intensity \uparrow
$$

즉 Roofline에서 **오른쪽으로 이동**합니다.

---

### Compute-bound workload

컴파일러 목표:

$$
Actual\ FLOPS \rightarrow Peak\ FLOPS
$$

방법:

> Tensorization
> Vectorization
> Kernel selection
> Loop optimization
> Instruction scheduling

즉 Roofline의 위쪽 ceiling에 더 가까이 갑니다.

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

방법:

> Partitioning
> Collective fusion
> Communication overlap
> Placement

입니다.

---

## 그래서 NPU Compiler 공부할 때는 이렇게 연결하면 좋습니다

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

그리고 이걸 한 문장으로 압축하면:

$$
\boxed{
\text{Compiler는 데이터 이동, 연산 배치, 메모리 배치, instruction mapping을 최적화한다.}
}
$$

특히 **NPU compiler를 공부한다면 Tiling → Buffer allocation → Operator fusion → Tensorization → DMA/compute scheduling** 이 다섯 가지를 우선순위 높게 보는 것이 좋습니다. 이 부분이 하드웨어 구조와 컴파일러가 실제로 만나는 지점입니다.
