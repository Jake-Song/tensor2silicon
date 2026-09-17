# ⑤ 커널 → 칩 내부: 연산은 어디서 이루어지고 데이터는 어디에서 이동해올까?

> **한 줄 답.** 행렬곱은 전용 행렬 유닛(TPU의 **MXU**, GPU의 **Tensor Core**)에서, 나머지 elementwise 연산은 벡터 유닛(TPU의 **VPU**, GPU의 **CUDA core**)에서 이루어진다. 데이터는 **HBM → 온칩 메모리(VMEM / SMEM·L2) → 레지스터 → 연산 유닛** 순으로 올라오고, 결과는 거꾸로 내려간다. 각 단계의 대역폭은 한 단계 내려갈 때마다 수 배~수십 배씩 떨어지므로, 커널 성능은 대개 "얼마나 계산하느냐"가 아니라 "HBM을 몇 번 오가느냐"로 정해진다.

```
        느림·큼                                             빠름·작음
   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐
   │   HBM    │ ─▶ │  온칩 메모리  │ ─▶ │ 레지스터  │ ─▶ │  연산 유닛    │
   │ 수십 GB   │ ◀─ │ VMEM / SMEM  │ ◀─ │  vreg    │ ◀─ │ MXU / TC     │
   │ ~TB/s    │    │ ~MB, ~수십TB/s│    │          │    │ VPU / CUDA   │
   └──────────┘    └──────────────┘    └──────────┘    └──────────────┘
        ▲
        │ PCIe (호스트) / ICI·NVLink (다른 칩)
```

이전 단계 ← [④ 실행 요청 → 커널 실행](./04-dispatch-to-kernel.md)

---

## 1. TPU: 컴파일러가 모든 이동을 지시하는 칩

Scaling Book 2장에 따르면 TPU는 **TensorCore**(계산 유닛) 하나와 그에 붙은 빠른 메모리 스택(HBM)으로 이루어진다. TensorCore 안에는 세 종류의 유닛이 있다.

### 1-1. 연산 유닛

| 유닛 | 하는 일 | 규모 |
|---|---|---|
| **MXU** (Matrix Multiply Unit) | 행렬곱. 128×128 **systolic array**(v6e는 256×256). `bf16[8,128] @ bf16[128,128] → f32[8,128]`을 8 사이클마다 하나씩 처리한다. | 1.5 GHz 기준 MXU 하나당 약 5×10¹³ bf16 FLOPs/s. TensorCore당 MXU 2~4개. v5e 칩 전체 약 2×10¹⁴ FLOPs/s |
| **VPU** (Vector Processing Unit) | elementwise 연산(ReLU, 덧셈, 곱셈)과 reduction. | v5p 기준 약 7×10¹² FLOPs/s. MXU의 약 1/30 |
| **스칼라 유닛** | 제어 흐름, 주소 계산, VLIW 명령 디스패치. | |

MXU가 FLOPs의 대부분이다. [①](./01-model-to-ops.md)에서 "Transformer = 큰 matmul 7개"라고 한 것과 짝이 맞는다. 행렬곱이 아닌 연산은 VPU로 가고, VPU는 MXU보다 30배쯤 느리므로 elementwise 연산이 많아지면 MXU가 논다.

### 1-2. Systolic array의 동작

128×128 격자의 각 칸이 곱셈-누산기(MAC)다. 가중치가 격자 위에서 아래로, 활성화가 옆에서 흘러 들어가며 각 칸이 자기 자리의 곱을 누적한다. 처음 가중치와 활성화를 채우는 동안 파이프라인 버블이 있지만, 그 뒤로는 새 데이터를 **멈춤 없이 계속** 밀어 넣을 수 있다. 그래서 MXU는 shape이 타일(8×128, 128×128)의 배수일 때 가장 효율적이고, [③](./03-ir-to-executable.md)의 컴파일러가 레이아웃과 패딩을 정하는 이유가 된다.

### 1-3. 메모리 계층

| 메모리 | 위치 | 크기 (v5e) | 대역폭 |
|---|---|---|---|
| **HBM** | 칩 옆 | 16 GB (v5p 96 GB) | ~8.2×10¹¹ B/s (v5p ~2.8×10¹²) |
| **VMEM** (Vector Memory) | TensorCore 안 스크래치패드 | 128 MiB | HBM의 약 22배 |
| **vreg** (벡터 레지스터) | VPU/MXU 바로 옆 | 작음 | 최고 |
| **IMEM** | 명령 메모리 | | LLO 바이너리가 여기에 올라감 |

TPU에는 **캐시가 없다**. VMEM은 하드웨어가 알아서 채우는 캐시가 아니라 **컴파일러가 DMA로 명시적으로 채우는 스크래치패드**다. [③](./03-ir-to-executable.md)에서 "TPU 백엔드는 HBM ↔ VMEM 이동을 컴파일 시점에 전부 스케줄링한다"고 한 것이 이 구조 때문이다.

### 1-4. 데이터 흐름

```
HBM ──DMA──▶ VMEM ──▶ vreg ──▶ MXU / VPU ──▶ vreg ──▶ VMEM ──DMA──▶ HBM
```

한 fusion이 실행되면 컴파일러가 짠 순서대로 DMA 엔진이 다음 타일을 VMEM으로 끌어오는 동안 MXU가 현재 타일을 계산하고, 결과는 전체 행렬이 완성되기를 기다리지 않고 파이프라인으로 HBM에 되돌아간다. 계산과 이동을 겹치는 것이 소프트웨어 파이프라이닝이고, 이것을 LLO가 만든다.

### 1-5. 칩 밖

- **PCIe**: 호스트 CPU와 연결. 한 트레이에 칩 4개, 칩당 약 1.6×10¹⁰ B/s. 느리므로 호스트↔디바이스 전송은 최소화해야 한다([④](./04-dispatch-to-kernel.md)의 `np.asarray`가 비싼 이유).
- **ICI** (Inter-Chip Interconnect): 칩끼리 2D/3D 토러스로 직접 연결. v5p 기준 양방향 약 9×10¹⁰ B/s. [③](./03-ir-to-executable.md)의 SPMD 파티셔너가 넣은 all-reduce가 이 위를 지난다.
- 칩 하나에 TensorCore 두 개가 메모리를 공유하는 **megacore** 구성이 일반적이다.

### 1-6. 세대별 수치

| 모델 | HBM 용량 | HBM 대역폭 (B/s) | bf16 FLOPs/s |
|---|---|---|---|
| TPU v5e | 16 GB | 8.2×10¹¹ | 1.97×10¹⁴ |
| TPU v5p | 96 GB | 2.8×10¹² | 4.59×10¹⁴ |
| TPU v6e | 32 GB | 1.6×10¹² | 9.20×10¹⁴ |

## 2. GPU: 수백 개의 작은 코어와 캐시 계층

Scaling Book 12장은 GPU를 TPU와 대비해서 설명한다.

### 2-1. SM (Streaming Multiprocessor)

GPU는 **SM**이라는 작은 코어를 수백 개 갖는다. H100은 132개, B200은 148개다. SM 하나는 4개의 동일한 **subpartition**(사분면)으로 나뉘고, 각 subpartition에는:

| 구성 요소 | 내용 |
|---|---|
| **Warp scheduler** | 이 사분면의 실행 유닛을 제어. 사이클마다 warp 하나의 명령 하나를 낸다. |
| **CUDA core** | fp32 코어 32개(+ 소수의 int32, fp64). 같은 사이클에 같은 명령을 실행하는 SIMD/SIMT 벡터 유닛. |
| **Tensor Core** | 행렬곱 전용 유닛. H100 기준 사이클당 약 1024 bf16 FLOPs. |
| **레지스터** | 32비트 레지스터 16k개. SM 전체로 256 kB. |

**Tensor Core가 FLOPs의 대부분**이다. H100은 Tensor Core로 990 bf16 TFLOP/s, CUDA core로 66 TFLOP/s다. TPU의 MXU vs VPU 비율(약 30:1)보다 더 극단적이다(약 15:1).

**SIMT 모델**: CUDA 프로그래밍 모델에서 각 스레드는 자기 명령 포인터를 갖는 것처럼 프로그래밍하지만, 실제로는 32개 스레드가 **warp** 하나로 묶여 같은 명령을 실행한다. 스레드마다 분기가 갈리면 일부 코어가 마스킹되어 논다. SM은 멀티스레드 CPU처럼 warp를 여러 개(SM당 최대 64개) 동시에 붙잡고 있다가, 어떤 warp가 메모리를 기다리면 다른 warp를 실행해 지연을 숨긴다.

### 2-2. 메모리 계층

| 메모리 | 위치 | 크기 | 대역폭 | 누가 관리 |
|---|---|---|---|---|
| **레지스터** | subpartition | 256 kB / SM | 최고 | 컴파일러 |
| **SMEM / L1** | SM 안 | 256 kB / SM (H100 합계 약 33 MB) | 매우 높음 | 프로그래머(shared memory) 또는 하드웨어(L1 캐시) |
| **L2 캐시** | 모든 SM이 공유 | H100 약 50 MB, B200 126 MB | 약 5.5 TB/s | 하드웨어 |
| **HBM** | 칩 옆 | H100 80 GB, B200 192 GB | H100 3.35 TB/s, B200 9 TB/s | |

SMEM은 "프로그래머가 제어하는 shared memory로도, 하드웨어가 쓰는 L1 캐시로도" 쓸 수 있다. 그리고 그 위에 **L2 캐시**가 하나 더 있다. TPU와의 결정적 차이는 이 캐시 계층이다. 접근 패턴을 컴파일러가 전부 정적으로 알지 못해도 캐시가 어느 정도 흡수해 준다.

### 2-3. 세대별 수치

| GPU | HBM 용량 | HBM 대역폭 | L2 | SMEM/SM | bf16 Tensor Core FLOPs/s |
|---|---|---|---|---|---|
| H100 | 80 GB | 3.35 TB/s | 50 MB | 256 kB | 990 TFLOP/s |
| B200 | 192 GB | 9 TB/s | 126 MB | 256 kB | 2.3 PFLOP/s |

## 3. TPU vs GPU 대응

| 역할 | TPU | GPU | 차이 |
|---|---|---|---|
| 행렬곱 유닛 | MXU (128×128 systolic array, TensorCore당 2~4개) | Tensor Core (SM당 4개 × 132 SM) | 소수의 큰 유닛 vs 다수의 작은 유닛 |
| 벡터 유닛 | VPU 하나 (8×128 유닛 4개, 총 4096 ALU) | CUDA core (132×4 = 528개 SIMD 유닛 × 32-wide, 총 약 16k ALU) | GPU가 벡터 ALU 수가 4배 많음 |
| 온칩 메모리 | VMEM 128 MiB (스크래치패드) | SMEM 256 kB/SM (합계 33 MB) + L2 50 MB | TPU가 프로그래머 제어 메모리가 훨씬 큼 |
| 메모리 이동 결정 | 컴파일러가 정적으로 DMA 스케줄 | 캐시 + warp 스케줄러가 동적으로 흡수 | |
| 지연 숨기기 | 소프트웨어 파이프라이닝 (컴파일 시점) | warp 교체 (런타임) | |
| 프로그래밍 단위 | 프로그램 하나가 코어 전체를 점유 | 그리드 × 블록 × warp × 스레드 | |
| 칩 간 연결 | ICI (2D/3D 토러스) | NVLink / NVSwitch | |

Scaling Book은 이렇게 요약한다. "GPU는 작은 SM 수백 개를 갖는다. … TPU는 훨씬 많은 빠른 캐시 메모리를 갖는다. TPU는 GPU의 SMEM보다 훨씬 큰 VMEM을 갖고, 이 메모리에 가중치와 활성화를 올려 두면 극도로 빠르게 로드해 쓸 수 있다."

## 4. 어디서 병목이 생기는가: arithmetic intensity

[①](./01-model-to-ops.md)에서 행렬곱은 연산 O(N³), 데이터 O(N²)라고 했다. 실제로 compute-bound가 되려면 **연산 유닛이 데이터를 기다리지 않을 만큼** 바이트당 FLOPs가 높아야 한다.

```
arithmetic intensity = FLOPs / 이동한 바이트
임계값 = (칩 FLOPs/s) / (메모리 대역폭 B/s)
```

| 상황 | 임계 배치 크기 (Scaling Book) |
|---|---|
| 가중치가 **HBM**에 있을 때 | 약 240 이상이어야 compute-bound |
| 가중치가 **VMEM**에 들어갈 때 | 약 10~20이면 충분 |

VMEM 대역폭이 HBM의 22배이므로, 가중치가 VMEM에 들어가면 훨씬 작은 배치에서도 MXU를 포화시킬 수 있다. 문서는 "알고리즘의 입출력이 모두 VMEM에 들어가면 통신 병목에 걸릴 가능성이 훨씬 낮다"고 적는다. 이것이 [③](./03-ir-to-executable.md)의 퓨전이 성능에 결정적인 이유다. 퓨전은 중간 결과를 HBM에 내렸다 올리는 대신 VMEM/레지스터에 머물게 한다.

## 5. ④의 Triton 커널을 칩 위에 놓아 보기

[④](./04-dispatch-to-kernel.md)의 `add_kernel`을 GPU 위에 올리면:

| 커널 코드 | 칩에서 벌어지는 일 |
|---|---|
| `add_kernel[grid](...)`, grid = ⌈n / BLOCK_SIZE⌉ | 프로그램 인스턴스(CTA) 그리드 크기만큼 생성. 각 CTA가 SM 하나에 배정된다. SM 132개보다 CTA가 많으면 순서대로 채워진다. |
| `pid = tl.program_id(0)` | 이 CTA의 인덱스. |
| `tl.load(x_ptr + offsets, mask)` | HBM → (L2 → L1) → 레지스터. `BLOCK_SIZE=1024`개 원소를 CTA의 warp들이 나눠 읽는다(coalesced access). |
| `output = x + y` | CUDA core에서 레지스터끼리 덧셈. Tensor Core는 쓰이지 않는다. |
| `tl.store(...)` | 레지스터 → HBM. |
| `BLOCK_SIZE` | CTA당 레지스터/SMEM 사용량과 직결. 너무 크면 SM에 동시에 올릴 수 있는 CTA가 줄어 지연 숨기기가 나빠진다. |

벡터 덧셈은 원소당 1 FLOP에 12바이트(읽기 8, 쓰기 4)를 이동하므로 arithmetic intensity가 약 0.08이다. H100의 임계값(990e12 / 3.35e12 ≈ 300)에 한참 못 미치는, 전형적인 **memory-bound** 커널이다. 이런 커널을 빠르게 하는 방법은 연산을 줄이는 것이 아니라 [③](./03-ir-to-executable.md)의 퓨전으로 **HBM 왕복 자체를 없애는 것**뿐이다.

같은 커널을 TPU에서 Pallas로 쓰면 `pid`에 해당하는 grid 인덱스마다 컴파일러가 HBM → VMEM DMA를 미리 스케줄링하고, 덧셈은 VPU에서 8×128 타일 단위로 일어난다.

## 6. 전체 되돌아보기

| 단계 | 질문 | 답 |
|---|---|---|
| [①](./01-model-to-ops.md) | 모델을 펼치면 무엇이 나오나 | 큰 matmul 7개 + elementwise, 6ND FLOPs |
| [②](./02-ops-to-graph-ir.md) | Python이 어떻게 그래프가 되나 | 추적(JAX) 또는 바이트코드 해석(Dynamo) → SSA IR |
| [③](./03-ir-to-executable.md) | 컴파일러가 무엇을 정하나 | 퓨전, 레이아웃, 버퍼, 스케줄 → 타겟 코드 |
| [④](./04-dispatch-to-kernel.md) | 함수 반환 = 계산 끝인가 | 아니다. 큐에 넣고 반환, 값을 읽을 때 블록 |
| ⑤ | 연산과 데이터는 어디에 | MXU/Tensor Core와 VPU/CUDA core, HBM → 온칩 → 레지스터 |

①의 FLOPs 계산과 ⑤의 대역폭 수치를 합치면 어떤 연산이 memory-bound인지 알 수 있고, 그것이 ③의 컴파일러가 퓨전을 어디에 넣어야 하는지를 결정하며, ④의 비동기 실행이 그 커널들의 런치 비용을 숨긴다.

---

## 참고 자료

- Scaling Book 2장 *How to Think About TPUs*: **What Is a TPU?** 절 — https://jax-ml.github.io/scaling-book/tpus/
- Scaling Book 12장 *How to Think About GPUs*: **What Is a GPU?**, **Memory** 절 — https://jax-ml.github.io/scaling-book/gpus/
