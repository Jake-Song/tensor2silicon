# 하드웨어 최적화: 메모리 대역폭과 연산 처리량

> **한 줄 답.** 하드웨어 최적화는 메모리 대역폭과 최대 연산 처리량을 높이고, 데이터 재사용과 통신을 지원하는 구조를 개선한다.

하드웨어 최적화는 Roofline의 **ceiling 자체를 높이거나 실제 하드웨어가 ceiling에 더 가깝게 동작하도록 만드는 작업**이다.

예를 들어 Roofline에서:

$$
P = \min(AI \times BW_{mem},\ P_{compute})
$$

하드웨어 관점의 핵심 최적화 방향은 다음과 같다.

- Memory Bound라면 **\(BW_{mem}\) 증가**
- Compute Bound라면 **\(P_{compute}\) 증가**
- Multi-GPU라면 **interconnect bandwidth 증가**

## 추론 병목별 소프트웨어와 하드웨어 최적화

| 병목                  | 소프트웨어 / 알고리즘                                 | 하드웨어                                                  |
| ------------------- | -------------------------------------------- | ----------------------------------------------------- |
| Memory Bound        | Quantization, KV cache 최적화, batching, fusion | HBM bandwidth ↑, cache/SRAM ↑, memory channel ↑       |
| Compute Bound       | GEMM 최적화, FlashAttention, sparsity, FP8      | Tensor Core/MAC ↑, clock ↑, low-precision accelerator |
| Communication Bound | TP/PP 최적화, overlap                           | NVLink/PCIe/CXL/Network bandwidth ↑                   |
| Memory Capacity     | KV compression, quantization                 | HBM/DRAM capacity ↑                                   |
| Latency             | Kernel fusion, speculative decoding          | clock, on-chip memory, dedicated units                |

---

## 1. Memory Bound에서의 하드웨어 최적화

Memory Bound라면:

$$
Performance \approx AI \times Memory\ Bandwidth
$$

이므로 가장 직접적인 방법은 **memory bandwidth 자체를 키우는 것**이다.

### HBM bandwidth 증가

예를 들어:

$$
1.5\ TB/s \rightarrow 3\ TB/s
$$

가 되면 memory-bound workload에서는 이론적으로 성능 ceiling이 거의 2배 올라간다.

하드웨어 설계 관점의 주요 방법은 다음과 같다.

- 더 빠른 HBM 세대 사용
- HBM stack 수 증가
- Memory bus width 증가
- Memory channel 증가
- Memory controller 개선

HBM 대역폭은 LLM decode에서 특히 중요하다.

GPU/NPU 추론 성능을 평가할 때는 FLOPS와 함께 **HBM bandwidth / model size**도 고려해야 한다.

---

## 2. On-chip SRAM / Cache 확대

HBM보다 SRAM이나 cache는 훨씬 빠르다.

개념적으로:

```text
Register
↓
SRAM / L1
↓
L2 Cache
↓
HBM
↓
CPU DRAM
```

이 계층을 아래로 내려갈수록 일반적으로 다음 경향이 나타난다.

- capacity ↑
- latency ↑
- bandwidth ↓

따라서 하드웨어에서 SRAM/L2를 키우면:

$$
HBM\ access \downarrow
$$

시킬 수 있다.

특히:

- weight tile
- activation
- attention intermediate
- KV cache 일부

를 on-chip에 오래 유지할 수 있다.

온칩 메모리 용량은 FlashAttention 같은 알고리즘과도 직접 연결된다.

FlashAttention은 SRAM에서 데이터를 재사용하여 HBM 접근을 줄인다. 따라서 **SRAM capacity가 큰 하드웨어일수록 데이터 재사용에 유리**하다.

---

## 3. Memory hierarchy 자체를 개선

단순히 cache를 크게 하는 것 외에도:

- prefetcher 개선
- cache replacement policy
- memory controller scheduling
- bank conflict 감소
- interleaving
- burst access

같은 하드웨어 최적화가 있다.

예를 들어 GPU가 앞으로 필요한 weight block을 예측해:

```text
compute block N
          +
load block N+1
```

을 동시에 수행하면 memory latency를 숨길 수 있다.

즉:

$$
Memory\ latency \rightarrow overlapped
$$

된다.

---

## 4. Compute Bound에서의 하드웨어 최적화

Compute Bound에서는:

$$
Performance \approx Peak\ Compute
$$

이므로 연산기 자체를 강화해야 한다.

대표적인 방법은 다음과 같다.

- Tensor Core 증가
- MAC unit 증가
- SIMD width 증가
- higher clock
- larger systolic array

예:

```text
GPU A
100 TFLOPS

GPU B
200 TFLOPS
```

같은 AI에서 compute bound라면 GPU B가 거의 2배 유리할 수 있다.

---

## 5. Low Precision 전용 하드웨어

낮은 정밀도 연산 지원은 AI 가속기의 주요 설계 요소다.

예를 들어:

```text
FP32 MAC
FP16 MAC
BF16 MAC
FP8 MAC
INT8 MAC
INT4 MAC
```

을 별도로 지원할 수 있다.

낮은 precision일수록 같은 silicon area에서 더 많은 연산기를 넣을 수 있다.

예를 들어 단순화하면:

$$
FP16\ MAC = 1
$$

이라면

$$
INT8\ MAC \approx 2
$$

$$
INT4\ MAC \approx 4
$$

처럼 병렬성을 늘릴 수 있다.

그래서 NVIDIA Tensor Core나 TPU/NPU의 systolic array는 low precision AI 연산에 최적화되어 있다.

이 경우 quantization은:

$$
Memory\ traffic \downarrow
$$

뿐 아니라

$$
Compute\ throughput \uparrow
$$

까지 동시에 얻을 수 있다.

---

## 6. Matrix multiplication 전용 하드웨어

Transformer에서 가장 큰 연산은 결국:

$$
Y = XW
$$

이므로 hardware도 GEMM에 맞춰진다.

GPU:

```text
Tensor Core
```

TPU/NPU:

```text
Systolic Array
```

같은 구조이다.

예를 들어:

```text
      Weight →
      ↓
Input → MAC → MAC → MAC
        ↓     ↓     ↓
       MAC → MAC → MAC
```

처럼 데이터가 PE(Processing Element) 사이에서 직접 흐르게 만들면 HBM이나 register 접근을 줄일 수 있다.

즉:

$$
Data\ reuse \uparrow
$$

하면서

$$
Compute\ throughput \uparrow
$$

가 동시에 가능하다.

---

## 7. 데이터 이동 최적화 하드웨어

AI accelerator에서는 실제로 계산보다 데이터 이동 에너지가 더 큰 경우가 많다.

따라서 하드웨어 설계에서는 연산기 수뿐 아니라 데이터 이동량을 줄이는 구조가 중요하다.

대표적인 접근은 다음과 같다.

- Near-memory compute
- Processing-in-Memory(PIM)
- SRAM compute
- Compute-near-memory

예를 들어 기존에는:

```text
HBM
 ↓
Tensor Core
 ↓
HBM
```

이라면 PIM은:

```text
HBM
 ├ Compute
 ├ Compute
 └ Compute
```

처럼 memory 근처에서 일부 연산을 수행한다.

이 구조는 LLM 추론처럼 memory-bound인 워크로드의 데이터 이동을 줄이는 데 유리하다.

---

## 8. Compute와 Memory overlap

하드웨어가 동시에:

```text
Compute A
+
Load B
```

를 수행할 수 있다면 memory latency를 숨길 수 있다.

이를 위해:

- DMA engine
- asynchronous memory copy
- multiple execution queues
- double buffering

등을 사용한다.

예를 들어:

```text
Buffer A → Compute

동시에

HBM → Buffer B
```

그리고:

```text
Buffer B → Compute

동시에

HBM → Buffer A
```

를 반복한다.

이를 **double buffering**이라 한다.

AI accelerator 설계에서 매우 자주 쓰인다.

---

## 9. Communication Bound 하드웨어 최적화

큰 모델에서는 GPU 하나로 inference가 안 되기 때문에:

```text
GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3
```

통신이 생긴다.

이때 Roofline을 확장하면:

$$
Performance =
\min(
Compute,
Memory,
Communication
)
$$

가 된다.

하드웨어에서는:

- PCIe bandwidth 증가
- NVLink
- NVSwitch
- InfiniBand
- Ethernet accelerator
- CXL

같은 interconnect 개선이 중요하다.

예를 들어 Tensor Parallelism에서는 layer마다:

$$
AllReduce
$$

가 발생할 수 있기 때문에 GPU의 FLOPS가 아무리 높아도 network가 느리면 GPU들이 기다린다.

---

## 10. GPU 간 memory 공유

또 다른 방향은 GPU마다 데이터를 복사하지 않는 것이다.

예:

```text
GPU0 memory
GPU1 memory
GPU2 memory
GPU3 memory
```

각각 따로 관리하는 대신:

```text
Shared / Unified Memory Pool
```

처럼 사용할 수 있다.

대표적인 접근은 다음과 같다.

- Unified Memory
- CXL memory
- shared HBM
- coherent accelerator memory

다만 latency와 bandwidth trade-off가 있다.

---

## 11. Prefill과 Decode의 하드웨어 최적화

### Prefill

대체로:

$$
Compute\ Bound
$$

쪽이다.

하드웨어 최적화 우선순위는 다음과 같다.

1. Tensor Core / MAC throughput 증가
2. FP8/BF16 전용 unit
3. systolic array
4. large SRAM
5. high compute utilization

---

### Decode

대체로:

$$
Memory\ Bound
$$

하드웨어 최적화 우선순위는 다음과 같다.

1. HBM bandwidth
2. L2 / SRAM capacity
3. memory controller
4. KV cache 전용 memory 구조
5. near-memory / PIM

---

## 12. Roofline 관점의 하드웨어 최적화

하드웨어 최적화는 크게 세 방향으로 구분된다.

```text
                  Compute ceiling ↑
Performance       ─────────────────────
  ↑                    /
  │                  /
  │                /
  │              /
  │            /
  │          /
  └────────────────────────→ AI
        ↑
   Memory slope 증가
```

### ① Memory bandwidth 증가

Roofline의 왼쪽 기울기를 높인다.

$$
Slope = Memory\ Bandwidth
$$

→ HBM, SRAM, cache, memory controller

### ② Peak Compute 증가

Roofline의 위쪽 ceiling을 높인다.

$$
Ceiling = Peak\ FLOPS
$$

→ Tensor Core, MAC, systolic array, FP8/INT8

### ③ Arithmetic Intensity를 높이기 쉬운 architecture 제공

AI는 워크로드 특성이지만, 다음 하드웨어 요소는 데이터 재사용을 지원한다.

- 큰 SRAM
- 좋은 cache
- data reuse
- systolic array

이 요소들은 **effective memory traffic을 줄여 실제 AI를 높이는 데 도움**을 준다.

LLM 추론 하드웨어는 다음 네 가지 지표를 함께 평가해야 한다.

$$
\boxed{
Compute\ Throughput,\ 
HBM\ Bandwidth,\ 
On-chip\ Memory,\ 
Interconnect
}
$$

특히 **Decode에서는 FLOPS보다 HBM bandwidth가 더 중요한 경우가 많고, Prefill에서는 Tensor Core throughput이 상대적으로 더 중요하다**고 정리할 수 있다.
