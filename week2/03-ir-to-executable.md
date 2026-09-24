# ③ IR → 실행 코드: 컴파일러는 무엇을 결정하고 하드웨어별로 무엇이 달라질까?

> **한 줄 답.** 컴파일러는 그래프를 받아 **어떤 연산을 하나의 커널로 묶을지(퓨전), 텐서를 메모리에 어떻게 놓을지(레이아웃), 버퍼를 언제 할당·재사용할지, 어떤 순서로 실행할지**를 결정하고, 마지막에 타겟별 기계 코드를 만든다. 앞부분(타겟 독립 최적화)은 하드웨어와 무관하고, 뒷부분(타겟 의존 최적화·코드 생성)은 백엔드마다 완전히 다르다.

```
StableHLO (②의 출력)
   │
   ▼
┌─────────────────────────────────────────────┐
│ 1. 타겟 독립 최적화 (Target-independent)     │  StableHLO → HLO 변환
│    CSE · 상수 접기 · 퓨전 · 버퍼 분석         │  어떤 백엔드든 동일
└─────────────────────────────────────────────┘
   │  최적화된 HLO
   ▼
┌─────────────────────────────────────────────┐
│ 2. 타겟 의존 최적화 (Target-dependent)       │  백엔드 지식으로 HLO 재작성
│    GPU: 스트림 분할, cuBLAS/cuDNN 패턴 매칭   │
│    TPU: MXU 타일 레이아웃, VMEM 스케줄링      │
└─────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────┐
│ 3. 코드 생성 (Code generation)               │
│    CPU / GPU: LLVM IR → x86 / PTX(NVPTX)     │
│    TPU: LLO → TPU 바이너리 (비공개 백엔드)    │
└─────────────────────────────────────────────┘
   │
   ▼
실행 파일 (executable) ─▶ ④ 런타임이 디스패치
```

이전 단계 ← [② 연산 → 그래프·IR](./02-ops-to-graph-ir.md) · 다음 단계 → [④ 실행 요청 → 커널 실행](./04-dispatch-to-kernel.md)

---

## 1. XLA가 이루려는 것 (Objectives)

OpenXLA 문서는 XLA의 목표를 네 가지로 적는다.

| 목표 | 방법 |
|---|---|
| **실행 속도** | 서브그래프 전체를 컴파일해 짧은 연산들의 실행 오버헤드를 없애고, 파이프라인된 연산을 퓨전해 메모리 왕복을 줄이고, 알려진 텐서 shape에 특수화해 더 공격적인 상수 전파를 한다. |
| **메모리 사용** | 메모리 사용을 분석·스케줄링해 중간 버퍼를 제거한다. |
| **커스텀 op 의존 감소** | 자동 퓨전된 저수준 op의 성능을 손으로 짠 커스텀 op 수준으로 끌어올려, 커스텀 op의 필요를 줄인다. |
| **이식성** | 새 하드웨어용 백엔드를 쉽게 만들 수 있게 해서, 모델 코드 수정 없이 여러 하드웨어에서 돌게 한다. |

첫 두 항목은 [①](./01-model-to-ops.md)에서 본 "elementwise 연산은 memory-bound"라는 문제에 대한 직접적인 답이다. exp, add, mul을 따로 실행하면 각각 HBM을 왕복하지만, 하나로 묶으면 한 번만 읽고 쓴다.

## 2. 어떻게 동작하는가 (How it works)

XLA의 입력은 StableHLO로 정의된 그래프다. 컴파일은 세 단계로 진행된다.

### 2-1. 타겟 독립 최적화

StableHLO 그래프에 백엔드와 무관한 패스를 돌린다.

- 공통 부분식 제거(CSE), 대수적 단순화, 상수 접기, 죽은 코드 제거
- **타겟 독립 퓨전**: 연속된 elementwise 연산을 하나의 fusion 노드로 묶는다.
- **버퍼 분석**: 각 중간 텐서에 런타임 메모리를 어떻게 배정할지 분석한다.

이 단계에서 StableHLO 방언은 XLA 내부의 **HLO** 방언으로 변환된다. Scaling Book 9장은 HLO를 "matmul, pointwise 연산, convolution 같은 핵심 선형대수 연산을 LLVM 스타일 그래프로 표현한 것"이라 설명하며 이런 예를 든다.

```
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3,
                                    f32[128,256]{1,0} %Arg_0.1),
              lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

읽는 법:

- `f32[16,128]`: 결과 shape과 dtype.
- `{1,0}`: **레이아웃**(minor-to-major 순서). `{1,0}`은 마지막 축이 메모리에서 가장 빠르게 변하는 row-major다. 이 표기가 있다는 것 자체가 "레이아웃은 컴파일러가 정하는 것"임을 보여준다.
- `lhs_contracting_dims={0}, rhs_contracting_dims={1}`: [①](./01-model-to-ops.md)의 contracting 축. jaxpr의 `dimension_numbers`가 그대로 내려온 것이다.

### 2-2. 타겟 의존 최적화

HLO를 백엔드로 보내면, 백엔드가 **자기 하드웨어에 대한 지식**으로 HLO 수준 최적화를 다시 한다.

- **GPU 백엔드**: GPU 프로그래밍 모델에 특히 유리한 방식으로 연산을 퓨전하고, 계산을 여러 **스트림**으로 분할해 병렬 실행하며, 특정 연산 패턴을 최적화된 라이브러리 호출(cuBLAS, cuDNN)로 **패턴 매칭**한다.
- **TPU 백엔드**: MXU가 요구하는 타일 레이아웃(f32는 8×128, bf16은 16×128)에 맞춰 텐서 배치를 정하고 타일 배수가 아닌 shape은 패딩한다. TPU에는 캐시가 없으므로 HBM ↔ VMEM 데이터 이동을 **컴파일 시점에 전부** 스케줄링한다. SPMD 파티셔너가 샤딩 어노테이션을 보고 프로그램을 칩별로 나누고 all-reduce / all-gather를 삽입하는 것도 이 수준이다.

### 2-3. 코드 생성

- **CPU / GPU**: HLO를 **LLVM IR**로 내리고 LLVM이 최적화·네이티브 코드 생성을 맡는다. NVIDIA GPU는 LLVM NVPTX 백엔드로 PTX를 만든다.
- **TPU**: HLO를 **LLO**(Low-Level Optimizer) IR로 내린다. Scaling Book에 따르면 LLO는 "TPU를 직접 프로그래밍"하는 수준으로, 메모리 복사 스케줄링, systolic array 동작, 메모리 공간 간 DMA를 지시한다. LLO는 기계 코드로 컴파일되어 TPU의 IMEM(명령 메모리)에 로드된다. 각 fusion은 스칼라·VPU·MXU·DMA 명령이 한 번들로 묶인 VLIW 명령으로 변환되고, 소프트웨어 파이프라이닝으로 DMA와 계산을 겹친다.

이 모듈 구조 덕분에 새 하드웨어를 위한 백엔드를 추가할 때 1단계는 그대로 두고 2·3단계만 구현하면 된다. 이것이 "이식성" 목표의 실체다.

## 3. 컴파일러가 결정하는 것

프레임워크(②)가 넘긴 그래프에는 "무엇을 계산하는가"만 있다. 컴파일러가 채워 넣는 "어떻게"는 다음이다.

| 결정 | 내용 | 영향 |
|---|---|---|
| **퓨전 경계** | 어떤 연산들을 하나의 커널(fusion)로 묶을지. 보통 matmul 하나 + 그 앞뒤 elementwise 연산. | HBM 왕복 횟수, 커널 수 |
| **레이아웃** | 각 텐서의 축 순서·타일링·패딩. | MXU/Tensor Core 활용률, 전치 비용 |
| **버퍼 할당** | 중간 텐서에 메모리를 언제 잡고 언제 해제·재사용할지. 살아있는 범위(liveness) 분석. | 피크 메모리 |
| **스케줄** | 의존성이 허용하는 범위에서 연산 실행 순서. 통신과 계산 겹치기. | 지연 시간, 메모리 |
| **집합 통신** | 샤딩된 프로그램에 all-reduce / all-gather / reduce-scatter를 어디에 넣을지. | 다중 칩 확장성 |
| **라이브러리 vs 생성 코드** | matmul을 cuBLAS 같은 외부 커널로 보낼지 직접 생성할지. | 성능, 퓨전 가능 범위 |
| **재계산** | 메모리를 아끼려고 일부 중간값을 다시 계산할지(rematerialization). | 메모리 ↔ FLOPs 트레이드오프 |

## 4. 하드웨어별로 무엇이 달라지는가

| | TPU | GPU | CPU |
|---|---|---|---|
| 실행 단위 | 소수의 큰 TensorCore, VLIW 번들 | 수백 개 SM, 커널 런치 단위 | 코어 + SIMD |
| 행렬 유닛 | MXU (systolic array), 타일 8×128 / 16×128 | Tensor Core, 타일 16×16 등 | AVX/AMX |
| 온칩 메모리 | VMEM (컴파일러가 명시적으로 DMA) | SMEM/L1 + L2 캐시 (하드웨어 캐시 + 프로그래머 제어) | 캐시 계층 (하드웨어) |
| 메모리 이동 결정 | **컴파일 시점에 정적** 스케줄 | 캐시가 동적으로 흡수, 커널 안에서는 프로그래머/컴파일러 | 캐시가 동적으로 흡수 |
| 코드 생성 | LLO → TPU 바이너리 (비공개) | LLVM NVPTX → PTX → SASS | LLVM → x86/ARM |
| 병렬화 표현 | 프로그램 하나가 칩 전체를 정적으로 점유 | 스트림, 그리드/블록 | OpenMP 스레드 |
| 외부 라이브러리 | 없음 (전부 컴파일러 생성) | cuBLAS/cuDNN 패턴 매칭 | oneDNN/MKL |

TPU는 "컴파일러가 전부 결정"하는 쪽 끝이고, GPU는 하드웨어 캐시·스케줄러가 일부를 런타임에 흡수하는 쪽이다. 이 차이가 [⑤ 칩 내부](./05-kernel-to-chip.md)의 메모리 계층 차이에서 나온다.

## 5. PyTorch 쪽 대응: TorchInductor

Inductor는 ATen 그래프를 받아 같은 결정을 다른 방식으로 한다.

| XLA | Inductor |
|---|---|
| HLO | Inductor IR (텐서를 "인덱스 → 값" 함수로 표현하는 define-by-run IR) |
| 타겟 독립 퓨전 | 스케줄러가 pointwise끼리, pointwise + reduction을 퓨전 |
| 라이브러리 패턴 매칭 | matmul은 기본 cuBLAS, `max-autotune`이면 Triton 템플릿과 비교, epilogue fusion |
| LLVM 코드 생성 | GPU: **Triton** 커널 소스 생성 → Triton 컴파일러가 LLVM → PTX. CPU: C++ + OpenMP 생성 |
| 실행 파일 하나 | 커널들 + 순서대로 런치하는 래퍼 코드 |

XLA는 "그래프 전체 → 실행 파일 하나"이고, Inductor는 "그래프 → 커널 여러 개 + 호출 순서를 담은 래퍼"라는 형태 차이가 있다. 이 차이가 [④](./04-dispatch-to-kernel.md)에서 실행 요청 모양의 차이로 나타난다. 상세는 [pytorch-compile.md](./pytorch-compile.md) 3절 참고.

## 6. 확인 명령

| 보고 싶은 것 | JAX / XLA | PyTorch / Inductor |
|---|---|---|
| 최적화 전 IR | `jax.jit(f).lower(*args).as_text()` | `TORCH_LOGS="aot_graphs"` |
| 최적화 후 (퓨전·레이아웃 반영) | `jax.jit(f).lower(*args).compile().as_text()` | `TORCH_LOGS="fusion"` |
| 생성된 커널 코드 | 프로파일러 (LLO는 비공개) | `TORCH_LOGS="output_code"` |
| 컴파일러 덤프 | `XLA_FLAGS=--xla_dump_to=/tmp/dump` | `TORCH_TRACE` + `tlparse` |

## 7. 다음 단계로

컴파일 결과는 TPU 바이너리, PTX 커널 묶음, 혹은 C++ 공유 라이브러리다. 아직 아무것도 실행되지 않았다. 이것을 디바이스에 올리고, 입력 버퍼를 넘기고, 큐에 넣는 것이 [④ 실행 요청 → 커널 실행](./04-dispatch-to-kernel.md)이다.

---

## 참고 자료

- OpenXLA *XLA architecture*: **Objectives**, **How it works** 절 — https://openxla.org/xla/architecture
- Scaling Book 9장 첫 절의 HLO / LLO 설명 — https://jax-ml.github.io/scaling-book/profiling/
