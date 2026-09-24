# 3주차: 커널의 시간은 어디에서 소비되는가

> **한 줄 답.** 커널의 계산량과 읽고 쓰는 데이터량을 각각 시간으로 바꾸면, 어떤 자원이 실행시간을 제한하는지 추정할 수 있다.

2주차에는 모델에서 실리콘까지의 흐름을 살펴봤다. 이번에는 커널 하나를 골라 **계산하는 시간과 데이터를 옮기는 시간**을 나누어 본다. 중심 질문은 “같은 연산인데 왜 어떤 커널은 빠르고 어떤 커널은 느린가?”다.

이번 주에는 하나의 가속기 안에서 일어나는 메모리 이동에 집중한다. 입력은 이미 가속기의 메모리에 있다고 가정하며, GPU 간 통신이나 CPU에서 GPU로 보내는 시간은 계산하지 않는다. Arithmetic Intensity(산술 강도), Roofline 그래프, Tiling(타일링)은 이 계산을 바탕으로 **4주차**에 다룬다.

## 1. 학습 목표와 읽는 순서

이번 주를 마치면 다음 세 가지를 설명할 수 있어야 한다.

- 해야 할 계산량과 하드웨어의 연산 성능, 메모리 용량과 대역폭을 구분한다.
- 간단한 커널의 이상적인 실행시간을 가정과 단위까지 붙여 추정한다.
- 연산 시간과 메모리 시간을 비교해 어떤 조건에서 Memory-bound인지 설명한다.

| 순서 | 교재 | 중심 질문 |
|---|---|---|
| ① | [FLOPs와 FLOPs/s](./01-flops-and-throughput.md) | 계산의 양과 초당 처리량은 어떻게 다른가? |
| ② | [메모리 용량과 대역폭](./02-capacity-and-bandwidth.md) | 16 GB와 1 TB/s는 무엇을 뜻하는가? |
| ③ | [HBM·DRAM·SRAM과 메모리 계층](./03-memory-hierarchy.md) | 왜 모든 데이터를 연산 장치 가까이에 둘 수 없는가? |
| ④ | [Compute Time과 Memory Time](./04-compute-and-memory-time.md) | 두 시간을 왜 더하지 않고 큰 값으로 추정하는가? |
| ⑤ | [Compute-bound와 Memory-bound](./05-compute-bound-and-memory-bound.md) | 어떤 자원을 늘려야 빨라지는가? |
| ⑥ | [Vector Add 손계산과 풀이](./06-vector-add.md) | 읽기·덧셈·쓰기를 실제 숫자와 코드로 연결할 수 있는가? |

이전 주차 → [2주차: 모델 → 연산](../week2/01-model-to-ops.md)

## 2. 공통으로 읽을 자료

| 자료 | 이번 주에 읽을 범위 | 읽으며 확인할 질문 |
|---|---|---|
| [Scaling Book 1장: All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | 첫 절 **Where Does the Time Go?**의 시간 공식과 Compute-bound / Communication-bound 설명까지 | 계산량과 이동량을 어떻게 시간으로 바꾸는가? |
| [MLC: GPU Acceleration](https://mlc.ai/chapter_gpu_acceleration/part1.html) | **GPU Architecture**의 구조 그림과 첫 Vector Add 코드 | `C[i] = A[i] + B[i]`에서 무엇을 읽고, 계산하고, 쓰는가? |
| [Scaling Book 12장: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/#memory) | **Memory**의 HBM, L2, L1/Shared Memory, Register 역할 | 데이터가 연산 장치에 도착하기까지 어떤 저장 공간을 이용하는가? |

MLC의 뒤쪽 루프 분할·스케줄 변환은 이번 주 필수 범위가 아니며 코드를 실행할 필요도 없다. 메모리 계층은 하드웨어별 용량 수치를 외우기보다 위치와 역할을 따라 읽는다.

**링크 확인 상태:** MLC 원문은 문서 작성 시 사용한 웹 도구에서 접근되지 않아 본문과 절 위치를 확인하지 못했다. 위 주소와 아래 Memory Spaces 주소는 읽기 안내로 남기며, 확인된 자료로 표시하지 않는다. 접근이 되지 않으면 GPU 구조는 Scaling Book 12장, 읽기·계산·쓰기는 Triton 튜토리얼을 함께 본다.

## 3. 주제별 보조 자료

| 자료 | 읽을 범위 | 어떤 질문에 도움이 되는가? |
|---|---|---|
| [NVIDIA: GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) | **GPU Architecture Fundamentals**, **Understanding Performance** | 메모리 용량과 대역폭을 어떻게 구분하고 두 시간 중 병목을 어떻게 찾는가? |
| [MLC: Memory Spaces](https://mlc.ai/chapter_gpu_acceleration/part1.html#memory-spaces) | 첫 설명과 표, 링크 확인 상태는 위와 같음 | GPU의 저장 공간은 누가 사용하고 어떻게 관리하는가? |
| [Scaling Book 2장: How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) | **What Is a TPU?**의 HBM·VMEM 설명, 선택 읽기 | GPU의 메모리 계층과 TPU의 HBM·VMEM을 어떻게 비교할 수 있는가? |
| [NVIDIA: Memory-Limited Layers](https://docs.nvidia.com/deeplearning/performance/dl-performance-memory-limited/index.html) | **Memory-Limited Layers**, **Activations** | 계산량이 적은 원소별 연산도 왜 오래 걸릴 수 있는가? |
| [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | `add_kernel`의 `tl.load`, 덧셈, `tl.store` | 손으로 센 입력 읽기와 출력 쓰기가 코드의 어느 부분인가? |
| [Micron: Introduction to Memory (PDF)](https://www.micron.com/content/dam/micron/educatorhub/intro-to-memory/micron-intro-to-memory-presentation.pdf) | SRAM·DRAM의 속도·집적도 비교표, 선택 읽기 | 왜 빠른 SRAM으로 모든 메모리를 대체하지 않는가? |

## 4. 함께 사용할 시간 모델

$$
t_{\mathrm{compute}} = \frac{\text{총 FLOPs}}{\text{연산 성능 (FLOPs/s)}}
$$

$$
t_{\mathrm{memory}} = \frac{\text{읽고 쓴 총 Byte}}{\text{메모리 대역폭 (Byte/s)}}
$$

$$
t_{\mathrm{estimated}} \approx \max(t_{\mathrm{compute}}, t_{\mathrm{memory}})
$$

연산과 데이터 이동을 충분히 겹치고 주어진 성능을 활용한다고 가정한다. 최대 성능을 넣은 값은 **최소 실행시간의 기준**이며 실제 측정값과 같다는 뜻이 아니다. 이번 주의 Memory-bound는 메모리 **대역폭**이 시간을 제한하는 경우를 뜻한다. 자세한 가정과 한계는 [④ 시간 모델](./04-compute-and-memory-time.md)에서 설명한다.

## 5. 모임 전에 계산해 올 문제

길이 `N = 100,000,000`인 FP32 벡터 A와 B를 더해 새 벡터 C에 저장한다. FP32 원소 하나는 4 Byte다. 연습용 가상 하드웨어의 FP32 덧셈 처리 성능은 `20 × 10¹² FLOPs/s`, HBM 대역폭은 `1 × 10¹² Byte/s`다. **실제 제품의 사양이나 측정값이 아니다.**

입력은 HBM에서 한 번씩 읽고 결과는 한 번 쓴다. 캐시 재사용과 추가 데이터 이동은 없으며 A·B·C를 저장할 공간은 충분하다. 이 교재에서 `1 GB = 10⁹ Byte`, `1 TB = 10¹² Byte`다.

풀이를 보기 전에 다음을 계산한다.

1. A와 B를 읽는 데이터 크기와 C를 쓰는 데이터 크기
2. 필요한 덧셈 횟수와 총 FLOPs
3. 연산 시간과 메모리 시간
4. 두 시간 중 더 큰 값과, 그 값으로 설명할 수 있는 병목
5. 연산 성능·대역폭·용량을 각각 늘렸을 때의 변화

계산을 마친 뒤 [⑥ 단계별 풀이](./06-vector-add.md)와 비교한다.

## 6. 모임에서 함께 확인할 질문

- 계산량이 작아도 느릴 수 있는 이유를 입력 읽기와 결과 쓰기로 설명할 수 있는가?
- 연산 성능만 두 배로 높이면 Vector Add가 두 배 빨라지는가? 대역폭만 두 배라면 어떠한가?
- “메모리에 들어간다”와 “메모리에서 빨리 옮긴다”를 서로 다른 계산으로 보여줄 수 있는가?
- 예상 실행시간을 말할 때 입력 위치, 이동 횟수, 성능 활용과 중첩 가정을 함께 말할 수 있는가?

추가로 좋은 자료를 찾았다면 링크만 공유하지 않고, **어떤 질문에 도움이 되는지와 읽을 범위**를 함께 적는다.
