# 7주차: Tiled MatMul과 Triton 컴파일 파이프라인

**프로그램 하나가 2차원 출력 타일을 맡아 K축으로 누적한다. L2 그룹핑은 타일 간 재사용을 높이고, MLIR 기반 다단계 IR은 블록 연산을 GPU 기계어로 구체화한다.**

[4주차 Tiling·Roofline](../week4/03-tiling.md) → [6주차 1차원 블록·Fusion](../week6/README.md)을 2차원 행렬곱 커널과 컴파일 파이프라인으로 확장한다. 여기서 익힌 블록 수준 DSL과 다단계 IR 구조는 8주차 가속기 컴파일러 분석의 밑바탕이 된다.

실습: [노트북](../notebooks/week7_triton_matmul.ipynb) · [Colab에서 열기](https://colab.research.google.com/github/Jake-Song/tensor2silicon/blob/main/notebooks/week7_triton_matmul.ipynb)

---

## ① 2차원 포인터 블록과 K축 누적

`C = A @ B` (`A: [M, K]`, `B: [K, N]`, `C: [M, N]`)에서 프로그램 인스턴스 하나는 출력 타일 `C[m:m+BM, n:n+BN]` 하나를 온칩 누산기(`accumulator`)에 계산한 뒤 마지막에 한 번 HBM에 쓴다.

### 브로드캐스팅을 이용한 2차원 포인터 블록 생성과 K축 전진

1차원 오프셋 벡터 두 개에 `[:, None]`(열 차원 추가, `[BM, 1]`)과 `[None, :]`(행 차원 추가, `[1, BK]`)을 적용해 더하면, 브로드캐스팅을 통해 `[BM, BK]`와 `[BK, BN]` 크기의 **2차원 포인터 블록**이 만들어진다.

```python
offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
offs_k = tl.arange(0, BLOCK_SIZE_K)

# [BM, 1] + [1, BK] -> [BM, BK] 포인터 블록
a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
# [BK, 1] + [1, BN] -> [BK, BN] 포인터 블록
b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
```

K 루프를 한 번 돌 때마다 A 포인터 블록은 열 방향(`stride_ak`)으로, B 포인터 블록은 행 방향(`stride_bk`)으로 `BLOCK_SIZE_K`만큼 전진한다.

```text
          A [M, K]                              B [K, N]
   ┌──────┬──────┬──────┐                ┌──────┬──────┬──────┐
   │      │      │      │                │      │b_ptrs│      │ ┐ BK
pid_m     │      │      │                │      │[BK,BN]      │ ┘
┌─ ├──────┼──────┼──────┤                ├──────┼──────┼──────┤
│BM│a_ptrs│  ──► │ +BK  │                │      │  │   │      │
└─ ├──────┼──────┼──────┤                │      │  ▼   │      │
   │[BM,BK]  k=0   k=1  │                │      │ +BK  │      │
   └──────┴──────┴──────┘                └──────┴──────┴──────┘
              │                                     │
              └──────────► tl.dot(a, b) ◄───────────┘
                                │
                                ▼
                    accumulator [BM, BN] (FP32)
                     ├── + a_0 @ b_0  (k = 0)
                     ├── + a_1 @ b_1  (k = 1)
                     └── ...
                                │
                     [선택: leaky_relu(acc)]  ← 온칩 레지스터에서 즉시 결합 (Epilogue Fusion)
                                │
                     c = acc.to(tl.float16)
                                │
                                ▼
                     C [pid_m, pid_n] (FP16, HBM 1회 쓰기)
```

### 4주차 타일 수식과 코드의 대응

```python
accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk

if ACTIVATION == "leaky_relu":
    accumulator = leaky_relu(accumulator)
c = accumulator.to(tl.float16)
```

- **왜 입력은 FP16인데 `accumulator`는 `tl.float32`인가?**
  FP16은 유효숫자(가수부 10비트)와 표현 범위가 좁아, 수천 개의 곱셈 결과를 계속 더하면 반올림 오차가 누적되거나 오버플로우(최댓값 65,504)가 발생하기 쉽다. GPU Tensor Core 역시 FP16 입력 두 개를 곱한 뒤 **FP32 누산기 레지스터에 더하는 혼합 정밀도(mixed-precision) 명령(`mma.sync...f32.f16.f16.f32`)**을 기본으로 지원하므로, 속도 손실 없이 정밀도를 지킬 수 있다.
- **왜 `leaky_relu`를 끼워 넣으면 추가 HBM 왕복 없이 결합(Epilogue Fusion)되는가?**
  K 루프가 끝난 직후 `accumulator`의 값들은 아직 온칩 **레지스터**에 머물러 있다. `tl.store`로 HBM에 내보내기 전에 활성화 함수를 적용하고 FP16으로 변환하면, 행렬곱 결과를 HBM에 썼다가 활성화 커널이 다시 읽고 쓰는 `2 × M × N × 2 Byte`의 추가 HBM 왕복이 완전히 사라진다.

근거: [Tutorial 03: Motivations · Pointer Arithmetic · Final Result](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) · [tl.dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)

핵심 용어: **2D pointer broadcasting · mixed-precision accumulation (`tl.dot`) · epilogue fusion**

---

## ② 프로그램 실행 순서와 L2 캐시(Grouped Ordering)

4주차 손계산에서는 “서로 다른 출력 타일 사이의 캐시 재사용은 없다”고 가정했다. 하지만 실제 GPU에는 모든 SM이 공유하는 **L2 캐시**가 있다. 여러 SM이 동시에 실행하는 프로그램 인스턴스들이 **같은 A 행 블록이나 같은 B 열 블록**을 요청하면 L2 캐시에서 재사용(Hit)할 수 있다.

### 9×9 블록 그리드에서 첫 9개 출력 블록(`pid = 0 ~ 8`) 비교

`A`와 `B`가 각각 9×9 블록으로 나뉘어 있고(`num_pid_m = 9`, `num_pid_n = 9`, `K / BK = 9`), 한 행 블록이나 한 열 블록을 끝까지 계산하려면 K축을 따라 **9개의 입력 타일**이 필요하다.

```text
[Row-major 순서: pid 0~8이 0행 전체 담당]        [Grouped 순서 (GROUP_SIZE_M=3): 3×3 정사각 구역 담당]
      B 열 블록: 0 1 2 3 4 5 6 7 8 (9줄)               B 열 블록: 0 1 2 (3줄)
               ┌─────────────────┐                             ┌─────┬───────────┐
A 행 블록 0 -> │ 0 1 2 3 4 5 6 7 8│             A 행 블록 0 -> │ 0 3 6│ . . . . . │
  (1줄)        ├─────────────────┤               (3줄)    1 -> │ 1 4 7│ . . . . . │
          1 -> │ . . . . . . . . │                        2 -> │ 2 5 8│ . . . . . │
               │ . . . . . . . . │                             ├─────┴───────────┤
               └─────────────────┘                             │ . . . . . . . . │
                                                               └─────────────────┘
필요 입력 타일:                                 필요 입력 타일:
A 1줄(9개) + B 9줄(81개) = 90개                 A 3줄(27개) + B 3줄(27개) = 54개 (40% 감소!)
```

- **행 우선 순서(Row-major)**:
  `pid = 0 ~ 8`이 `C[0, 0] ~ C[0, 8]`을 맡는다.
  - 필요한 A 입력: 0번째 행 블록 **1줄** × K축 9개 = **9개 타일**
  - 필요한 B 입력: 0~8번째 열 블록 **9줄** × K축 9개 = **81개 타일**
  - 합계: `9 + 81 = 90개` 입력 타일을 L2로 불러와야 한다.
- **그룹 순서(`GROUP_SIZE_M = 3`)**:
  몇 개 행(`GROUP_SIZE_M = 3`)씩 묶어 열 방향으로 진행하는 Swizzling 인덱싱을 사용한다.

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n          # 3 * 9 = 27
group_id = pid // num_pid_in_group                   # pid 0~8 -> 0
first_pid_m = group_id * GROUP_SIZE_M                # 0
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 3
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

| `pid` | Row-major `(pid_m, pid_n)` | Grouped `(pid_m, pid_n)` (`GROUP_SIZE_M=3`) |
|---:|:---:|:---:|
| 0 | `(0, 0)` | `(0, 0)` |
| 1 | `(0, 1)` | `(1, 0)` |
| 2 | `(0, 2)` | `(2, 0)` |
| 3 | `(0, 3)` | `(0, 1)` |
| 4 | `(0, 4)` | `(1, 1)` |
| 5 | `(0, 5)` | `(2, 1)` |
| 6 | `(0, 6)` | `(0, 2)` |
| 7 | `(0, 7)` | `(1, 2)` |
| 8 | `(0, 8)` | `(2, 2)` |

  - 필요한 A 입력: 0~2번째 행 블록 **3줄** × K축 9개 = **27개 타일** (3개 열이 공유)
  - 필요한 B 입력: 0~2번째 열 블록 **3줄** × K축 9개 = **27개 타일** (3개 행이 공유)
  - 합계: `27 + 27 = 54개` 입력 타일만 있으면 9개 출력 블록을 모두 계산할 수 있다(**90개 → 54개, 40% 감소**).

타일 크기(`BM, BN, BK`)와 총 연산량은 전혀 바뀌지 않았지만, **비슷한 시각에 실행되는 프로그램들이 좁은 정사각 구역의 A·B를 함께 참조**하므로 L2 캐시 적중률이 높아지고 실제 HBM 트래픽이 줄어든다.

근거: [Tutorial 03: L2 Cache Optimizations](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations)

핵심 용어: **L2 cache hit rate · grouped ordering (swizzling) · `GROUP_SIZE_M`**

---

## ③ 메타파라미터와 자동 튜닝(`@triton.autotune`)

최적의 타일 크기와 파이프라인 설정은 **행렬 크기(`M, N, K`)**와 **GPU 아키텍처(SM 수, Shared Memory 용량, 레지스터 파일 크기)**에 따라 달라진다.

- `M, N`이 매우 작으면 `128×256`처럼 큰 타일은 출력 그리드(`grid`)가 너무 작아져 GPU의 모든 SM을 채우지 못한다(Wave quantization / 낮은 병렬성).
- 반대로 `M, N, K`가 충분히 크면 큰 타일이 HBM 산술 강도를 높이지만, Shared Memory나 레지스터를 많이 써서 SM당 상주 워프 수(Occupancy)가 줄어들거나 하드웨어 한계(`OutOfResources`)를 넘을 수 있다.

따라서 하나로 고정하지 않고 `triton.Config` 후보 목록을 준 뒤 `@triton.autotune`으로 실행 환경에서 직접 측정해 고른다.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # ...
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
```

- **언제 후보들을 다시 벤치마크하는가?**
  `key=['M', 'N', 'K']`에 지정한 인자의 **실제 런타임 값**과 **입력 텐서들의 자료형(dtype)** 조합이 바뀔 때마다 모든 후보 `Config`를 컴파일·실행해 가장 빠른 설정을 캐시에 저장한다. 같은 `(M, N, K, dtype)`으로 다시 호출하면 벤치마크 없이 캐시된 최적 `Config`를 즉시 사용한다.
- **`num_warps`와 `num_stages`의 역할과 자원 압박**:
  - `num_warps`: 프로그램 인스턴스(CTA) 하나를 몇 개의 워프(1 warp = 32 threads)가 나눠 계산할지 정한다. 워프 수를 늘리면 `[BM, BN]` 타일을 더 잘게 나눠 스레드당 레지스터 부담을 줄이고 명령 발행률을 높일 수 있지만, 워프 간 동기화 비용이 들고 블록당 스레드 수가 늘어난다.
  - `num_stages`: K 루프에서 **소프트웨어 파이프라이닝(Software Pipelining)** 단계 수다. `num_stages=3`이면 현재 $k$번째 타일을 Tensor Core로 계산하는 동안 $k+1, k+2$번째 타일을 HBM에서 Shared Memory로 비동기 프리페치(`cp.async`)해 메모리 지연시간을 연산 뒤에 숨긴다(Overlap). 단, Shared Memory에 **`num_stages` 배수의 다중 버퍼(multi-buffer)**를 잡아야 하므로 온칩 메모리 사용량이 비례해서 커진다.

근거: [triton.autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) · [Modern GPU Programming: Overlap & Occupancy](https://gau-nernst.github.io/matmul-cpu-gpu/)

핵심 용어: **`@triton.autotune` · software pipelining (`num_stages`) · resource pressure (`num_warps`)**

---

## ④ Triton 컴파일 파이프라인과 IR 계층

`@triton.jit` 함수는 왜 Python에서 곧바로 PTX/기계어로 번역하지 않고 여러 중간 표현(IR)을 거칠까?
한 단계에서 타일 데이터 흐름 분석, 워프/스레드 분할, Shared Memory 배치 및 뱅크 충돌 회피(swizzling), 비동기 복사 파이프라이닝, 레지스터 할당을 동시에 처리하면 컴파일러 구조가 지나치게 복잡해지고 다른 GPU 백엔드(AMD ROCm 등)로 확장하기도 어렵기 때문이다. 현대 Triton은 **MLIR(Multi-Level IR)** 위에서 추상화 수준을 단계별로 낮춘다(Lowering).

```text
[Python @triton.jit 함수]
       │  Python AST 파싱 & 블록 의미론 검사
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. TTIR (Triton IR, tt dialect) — 하드웨어 독립적인 타일/블록 수준 IR   │
│  • 스레드·워프·Shared Memory 개념 없이 2D 텐서 블록 연산만 표현         │
│  • tt.splat, tt.broadcast, tt.addptr, tt.load, scf.for, tt.dot, tt.store│
│  • 타일 수준 데이터 흐름 분석, 브로드캐스팅 정규화, 공통 부분식 제거     │
└─────────────────────────────────────────────────────────────────────────┘
       │  target GPU 정보 + num_warps, num_stages 반영
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. TTGIR (TritonGPU IR, ttg dialect) — GPU 특화 레이아웃 & 파이프라인 IR│
│  • 텐서마다 레이아웃 인코딩 부여:                                       │
│    - #ttg.blocked (글로벌 load/store 병합을 위한 스레드 분산 배치)      │
│    - #ttg.shared / #ttg.nvmma_shared (Shared Memory 배치 및 swizzling)  │
│    - #ttg.dot_op / #ttg.nvidia_mma (Tensor Core 입력/누산 레지스터 배치)│
│  • Coalescing, ttg.local_alloc(SMEM 할당), scf.for 소프트웨어 파이프라이닝│
│    (ttg.async_copy_global_to_local / cp.async 삽입 및 대기 동기화)      │
└─────────────────────────────────────────────────────────────────────────┘
       │  레지스터 조각(fragment)·포인터 연산·워프 내 스레드 명령으로 분해
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. LLVM IR (LLIR) — 스레드 관점의 저수준 중간 표현 (NVVM intrinsics 포함)│
└─────────────────────────────────────────────────────────────────────────┘
       │  LLVM NVPTX 백엔드
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. PTX (Parallel Thread Execution) — NVIDIA 가상 ISA 어셈블리           │
│  • cp.async.cg.shared.global, ldmatrix, mma.sync.aligned.m16n8k16, ...  │
└─────────────────────────────────────────────────────────────────────────┘
       │  ptxas 어셈블러 (실제 레지스터 수·SMEM 확정)
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. CUBIN (SASS) — 특정 GPU 아키텍처(sm_75, sm_80 등)의 실제 실행 기계어 │
└─────────────────────────────────────────────────────────────────────────┘
```

### TTIR vs TTGIR 핵심 차이

| 비교 항목 | TTIR (`tt` dialect) | TTGIR (`ttg` dialect) |
|---|---|---|
| **관점** | 프로그램 인스턴스 1개가 다루는 **논리적 블록/타일** | **`num_warps`개 워프(스레드)**와 **온칩 메모리 계층**에 나뉜 물리적 배치 |
| **텐서 타입 예시** | `tensor<128x64xf16>` | `tensor<128x64xf16, #ttg.blocked<...>>` 또는 `#ttg.dot_op<...>` |
| **메모리 계층** | HBM 포인터 load/store만 존재 (Shared Memory 미표현) | `ttg.local_alloc`으로 Shared Memory 명시 할당, `ttg.convert_layout`으로 레이아웃 변환 |
| **루프 파이프라이닝** | 단순 순차 `scf.for` 루프 | `num_stages`에 맞춰 비동기 복사(`async_copy_global_to_local`)와 연산을 중첩한 파이프라인 루프 |
| **행렬곱(`tl.dot`)** | 추상 타일 곱 `tt.dot` | GPU 아키텍처에 맞는 Tensor Core 레이아웃(`#ttg.nvidia_mma`) 기반 `tt.dot` / `ttng.warp_group_dot` |

근거: [Triton MAPL 2019 논문 §4.2, §5](https://www.PhilippeTillet.com/assets/publications/triton.pdf) · [Triton MLIR Dialects](https://github.com/triton-lang/triton/tree/main/include/triton/Dialect)

핵심 용어: **MLIR lowering · TTIR (`tt` dialect) · TTGIR (`ttg` layout encodings)**

---

## ⑤ 컴파일 산출물 확인과 벤치마크

### 컴파일된 커널에서 IR·PTX·메타데이터 확인하기

커널을 한 번 실행(또는 `.warmup(...)` 호출)하면 `CompiledKernel` 객체가 반환된다.

```python
compiled = matmul_kernel[grid](a, b, c, M, N, K, ...)

# 1. 온칩 자원 및 컴파일 메타데이터 확인
print(compiled.metadata.shared)      # 블록(CTA)당 정적 할당된 Shared Memory (Byte)
print(compiled.metadata.num_warps)   # 인스턴스당 워프 수
print(compiled.metadata.num_stages)  # 파이프라인 단계 수
print(compiled.n_regs)               # 스레드당 사용 레지스터 수
print(compiled.n_spills)             # 레지스터 스필(spill) 바이트 수

# 2. 단계별 컴파일 산출물(문자열) 확인
print(compiled.asm["ttir"][:1000])   # Triton IR
print(compiled.asm["ttgir"][:1000])  # TritonGPU IR
print(compiled.asm["llir"][:1000])   # LLVM IR
print(compiled.asm["ptx"][:1000])    # PTX 어셈블리 (len(compiled.asm["cubin"])으로 바이너리 크기 확인)
```

환경변수를 사용하면 코드 수정 없이도 확인 가능하다.
- `TRITON_PRINT_AUTOTUNING=1`: 오토튜닝이 끝났을 때 어떤 `Config`가 선택되었고 각 후보가 몇 ms 걸렸는지 터미널에 출력한다.
- `TRITON_KERNEL_DUMP=1` + `TRITON_DUMP_DIR=/tmp/triton_dump`: 컴파일 과정에서 생성된 모든 `.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin` 파일을 지정 폴더에 저장한다.

### 왜 6주차 Softmax는 GB/s를 재고, 7주차 MatMul은 TFLOPs/s를 재는가?

4주차 **Roofline 모델**에서 커널이 달성하고자 하는 상한선이 다르기 때문이다.

| 비교 항목 | 6주차 Fused Softmax | 7주차 Tiled MatMul (`C = A @ B`) |
|---|---|---|
| **산술 강도 ($I = F/Q$)** | 낮음 (원소당 몇 번의 연산만 수행, $O(1)$) | 높음 (타일 재사용으로 $I \approx \frac{1}{1/BM + 1/BN}$, 수십~수백 FLOPs/Byte) |
| **Roofline 위치** | **대역폭 병목(Memory-bound)** — 기울어진 선 아래 | **연산 병목(Compute-bound)** — 수평 연산 상한선 아래 (충분히 큰 행렬·타일 기준) |
| **핵심 성능 지표** | 하드웨어 HBM 대역폭(`BW`)에 얼마나 근접했는가? → **실효 대역폭 (GB/s)** | 하드웨어 Tensor Core 최대 성능(`P`)에 얼마나 근접했는가? → **실효 연산 처리량 (TFLOPs/s)** |
| **계산식** | $\dfrac{2 \times M \times N \times \text{ sizeof(dtype)}}{t_{\text{ms}} \times 10^{-3}} \times 10^{-9}\ \text{GB/s}$ | $\dfrac{2 \times M \times N \times K \times 10^{-12}}{t_{\text{ms}} \times 10^{-3}}\ \text{TFLOPs/s}$ |

행렬곱에서 출력 원소 $C_{i,j}$ 하나를 구하려면 길이 $K$인 두 벡터의 내적(곱셈 $K$번 + 덧셈 $K$번 = $2K$ FLOPs)이 필요하므로, 전체 행렬곱의 총 연산량은 항상 $F = 2MNK\ \text{FLOPs}$다. 이를 실행 시간 $t\ (\text{s}) = t_{\text{ms}} \times 10^{-3}$으로 나누고 $10^{-12}$를 곱하면 초당 테라플롭스(**TFLOPs/s**)가 된다.

근거: [Tutorial 03: Benchmark](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#benchmark) · [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

핵심 용어: **`.asm` / `.metadata` · Roofline ridge point · TFLOPs/s (`2MNK / t`)**

---

## 함께 해볼 손계산 풀이: `M = N = K = 4096` FP16 MatMul과 3개 후보 비교

### 1. 문제 조건과 공통 값

- **행렬 크기**: $M = N = K = 4096$
- **자료형**: 입력 $A, B$와 출력 $C$는 FP16 (**2 Byte**), 온칩 누산기(`accumulator`)는 FP32 (**4 Byte**)
- **가상 하드웨어**:
  - 적용 가능한 최대 연산 성능 $P = 80 \times 10^{12}\ \text{FLOPs/s}$ (**80 TFLOPs/s**)
  - HBM 대역폭 $BW = 1 \times 10^{12}\ \text{Byte/s}$ (**1 TB/s**)
  - Roofline 분기점(Ridge Point): $I_{\text{ridge}} = \dfrac{P}{BW} = \mathbf{80\ \text{FLOPs/Byte}}$
- **가정**: 입력은 이미 HBM에 있고, 서로 다른 출력 타일 사이의 L2 캐시 재사용은 없다고 가정한다(타일 내부 재사용만 반영).
- **공통 전체 연산량 $F$와 연산 시간 하한 $t_{\text{compute}}$**:
  $$F = 2MNK = 2 \times 4096^3 = \mathbf{137,438,953,472\ \text{FLOPs}}\ (\approx 137.439\ \text{GFLOPs})$$
  $$t_{\text{compute}} = \frac{F}{P} = \frac{137,438,953,472}{80 \times 10^{12}} \approx \mathbf{1.717987\ \text{ms}}$$
- **공통 출력 쓰기 이동량 $Q_{\text{out}}$**:
  $$Q_{\text{out}} = 2 \times M \times N = 2 \times 4096^2 = \mathbf{33,554,432\ \text{Byte}}\ (\approx 33.554\ \text{MB})$$

---

### 2. 세 후보의 그리드 크기·K 루프·HBM 이동량·산술 강도·이상적 시간 비교

출력 타일 크기가 $(BM, BN)$이면:
- 프로그램 인스턴스 수 `grid` $= \dfrac{M}{BM} \times \dfrac{N}{BN}$
- 각 출력 타일은 $A$의 $[BM, K]$ 행 조각($2 \times BM \times K$ Byte)과 $B$의 $[K, BN]$ 열 조각($2 \times K \times BN$ Byte)을 읽는다.
- 모든 출력 타일에 대해 합산한 HBM 입력 읽기 Byte는 다음과 같다:
  $$Q_{\text{in}} = \left(\frac{M}{BM}\frac{N}{BN}\right) \times 2K(BM + BN) = 2MNK\left(\frac{1}{BN} + \frac{1}{BM}\right)$$
  - $A$ 전체 읽기($2MK$ Byte)를 열 타일 수 $N/BN$번 반복: $Q_A = \dfrac{2MNK}{BN}$
  - $B$ 전체 읽기($2KN$ Byte)를 행 타일 수 $M/BM$번 반복: $Q_B = \dfrac{2MNK}{BM}$

| 항목 | 후보 1 (`64×32×32`, `stages=5, warps=2`) | 후보 2 (`128×128×32`, `stages=4, warps=4`) | 후보 3 (`128×256×64`, `stages=3, warps=8`) |
|---|---:|---:|---:|
| `BM, BN, BK` | `64, 32, 32` | `128, 128, 32` | `128, 256, 64` |
| 실행 그리드 `grid = (M/BM * N/BN,)` | `(64 × 128,) = (8,192,)` | `(32 × 32,) = (1,024,)` | `(32 × 16,) = (512,)` |
| 타일당 K 루프 반복 횟수 `K / BK` | `4096 / 32 = 128회` | `4096 / 32 = 128회` | `4096 / 64 = 64회` |
| HBM $A$ 읽기 ($2MNK / BN$) | `4,294,967,296 Byte` | `1,073,741,824 Byte` | `536,870,912 Byte` |
| HBM $B$ 읽기 ($2MNK / BM$) | `2,147,483,648 Byte` | `1,073,741,824 Byte` | `1,073,741,824 Byte` |
| **HBM 입력 읽기 합계 $Q_{\text{in}}$** | **6,442,450,944 Byte** (6.442 GB) | **2,147,483,648 Byte** (2.147 GB) | **1,610,612,736 Byte** (1.611 GB) |
| HBM 출력 쓰기 $Q_{\text{out}}$ ($2MN$) | `33,554,432 Byte` (0.034 GB) | `33,554,432 Byte` (0.034 GB) | `33,554,432 Byte` (0.034 GB) |
| **총 HBM 이동량 $Q = Q_{\text{in}} + Q_{\text{out}}$** | **6,476,005,376 Byte** (6.476 GB) | **2,181,038,080 Byte** (2.181 GB) | **1,644,167,168 Byte** (1.644 GB) |
| **산술 강도 $I = F / Q$** (출력 쓰기 제외 근사) | **21.22 FLOPs/Byte** (21.33) | **63.02 FLOPs/Byte** (64.00) | **83.59 FLOPs/Byte** (85.33) |
| 메모리 시간 $t_{\text{mem}} = Q / BW$ | `6.476005 ms` | `2.181038 ms` | `1.644167 ms` |
| 연산 시간 $t_{\text{compute}} = F / P$ | `1.717987 ms` | `1.717987 ms` | `1.717987 ms` |
| **이상적 실행시간 $t_{\text{ideal}} = \max(t_{\text{comp}}, t_{\text{mem}})$** | **6.476005 ms** | **2.181038 ms** | **1.717987 ms** |
| **병목 구간 ($I_{\text{ridge}} = 80$ 기준)** | **대역폭 병목 (Memory-bound)** | **대역폭 병목 (Memory-bound)** | **연산 병목 (Compute-bound)** |

> **해설**: `BLOCK_SIZE_K`(`BK`)는 한 번의 K 루프에서 읽는 폭을 바꿀 뿐, 루프 횟수(`K / BK`)와 상쇄되므로 **HBM 총 이동량 $Q$에는 영향을 주지 않는다.** $Q$와 산술 강도 $I$를 결정하는 것은 출력 타일의 가로·세로 크기인 **`BM`과 `BN`**이다. 후보 3(`128×256`)에 이르러서야 산술 강도가 `83.59 FLOPs/Byte`로 분기점(`80`)을 넘어 **연산 병목(Compute-bound)** 구간에 진입한다.

---

### 3. 온칩 버퍼 크기(Shared Memory · 레지스터 누산기)와 64 KiB 자원 한계 판정

- **1단계(1 stage) A·B 입력 타일 크기**:
  $$\text{Buffer}_{\text{1-stage}} = 2\ \text{Byte} \times (BM \times BK + BK \times BN)$$
- **`num_stages` 파이프라인 입력 버퍼 합계(Shared Memory)**:
  $$\text{Buffer}_{\text{pipeline}} = \text{ num\_stages} \times \text{Buffer}_{\text{1-stage}}\quad (1\ \text{KiB} = 1024\ \text{Byte})$$
- **FP32 누산기(`accumulator`) 크기(레지스터 파일 분산 보관)**:
  $$\text{Acc}_{\text{FP32}} = 4\ \text{Byte} \times BM \times BN$$

| 온칩 자원 항목 | 후보 1 (`64×32×32`, `stages=5, warps=2`) | 후보 2 (`128×128×32`, `stages=4, warps=4`) | 후보 3 (`128×256×64`, `stages=3, warps=8`) |
|---|---:|---:|---:|
| 1단계 $A$ 타일 ($2 \times BM \times BK$) | `4,096 Byte` (4 KiB) | `8,192 Byte` (8 KiB) | `16,384 Byte` (16 KiB) |
| 1단계 $B$ 타일 ($2 \times BK \times BN$) | `2,048 Byte` (2 KiB) | `8,192 Byte` (8 KiB) | `32,768 Byte` (32 KiB) |
| **1단계 입력 합계 ($A + B$)** | **6,144 Byte (6.0 KiB)** | **16,384 Byte (16.0 KiB)** | **49,152 Byte (48.0 KiB)** |
| 파이프라인 단계 수 (`num_stages`) | `5` | `4` | `3` |
| **파이프라인 입력 버퍼 합계 (SMEM)** | **30,720 Byte (30.0 KiB)** | **65,536 Byte (64.0 KiB)** | **147,456 Byte (144.0 KiB)** |
| **FP32 누산기 크기 (레지스터 합계)** | **8,192 Byte (8.0 KiB)** | **65,536 Byte (64.0 KiB)** | **131,072 Byte (128.0 KiB)** |
| 스레드당 FP32 누산기 레지스터 수 (`Acc / (warps×32×4B)`) | `64개 / thread` (총 64스레드) | `128개 / thread` (총 128스레드) | `128개 / thread` (총 256스레드) |
| **블록당 최대 SMEM 64 KiB 한계 판정** | **통과** (30 KiB ≤ 64 KiB) | **경계 통과** (64 KiB ≤ 64 KiB) | **자원 한계 초과 (`OutOfResources`: 144 KiB > 64 KiB)** |

> **해설**:
> - **후보 3(`128×256×64, num_stages=3`)**은 산술 강도가 가장 높아 연산 병목 구간에 들어가지만, Shared Memory 파이프라인 버퍼만 **144 KiB**가 필요하다. 따라서 블록당 최대 Shared Memory가 **64 KiB**인 GPU(예: Turing 아키텍처 **Tesla T4, sm_75**)에서는 컴파일 또는 실행 시 `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 147456, Hardware limit: 65536`가 발생해 사용할 수 없다(A100 `sm_80`처럼 SM당 최대 164 KiB를 지원하는 GPU에서만 실행 가능).
> - 또한 후보 3의 FP32 누산기는 `128 KiB`(32,768개 float32)에 달한다. 만약 `num_warps=4`(128스레드)로 두면 스레드당 누산기에만 256개 레지스터가 필요해 하드웨어 스레드당 최대 레지스터 한계(255개)를 넘기 때문에, **`num_warps=8`(256스레드)로 늘려 스레드당 누산기 레지스터를 128개로 분산**시킨 것이다.

---

### 4. `matmul_kernel` 핵심 구문의 컴파일 단계별 구체화 표

| Python 커널 구문 | AST → **TTIR (`tt` dialect)** | **TTGIR (`ttg` dialect)** | **LLVM IR (LLIR)** | **PTX (NVIDIA 가상 어셈블리)** |
|---|---|---|---|---|
| `tl.program_id(axis=0)` | `tt.get_program_id x : i32` | 동일 (`tt.get_program_id x`), CTA 단위 블록 좌표 계산에 사용 | `@llvm.nvvm.read.ptx.sreg.ctaid.x()` 호출 | `mov.u32 %r1, %ctaid.x;` (특수 레지스터 읽기) |
| 2차원 `tl.load(a_ptrs, mask=..., other=0.0)` | `tt.splat`, `tt.broadcast`, `tt.addptr`로 만든 `tensor<BMxBKx!tt.ptr<f16>>`에 대해 `tt.load` 수행 | `#ttg.blocked` 레이아웃 결정(스레드별 연속 주소 병합), `ttg.local_alloc`으로 SMEM 버퍼 할당 및 `ttg.async_copy_global_to_local`로 변환 | 주소 오프셋 계산 및 `@llvm.nvvm.cp.async...` 또는 조건부 글로벌 load intrinsic 생성 | `cp.async.cg.shared.global [smem], [gmem], 16;` (또는 `@p ld.global.b32`) |
| `for k in range(0, tl.cdiv(K, BK)):` | 단순 순차 루프 `scf.for %k = %c0 to %K step %BK iter_args(%acc = ...)` | `num_stages`에 맞춰 프롤로그(선행 프리페치) + 본 루프(`async_copy` → `async_commit_group` → `async_wait` → `dot`) + 에필로그로 **소프트웨어 파이프라이닝** 변환 | 루프 분기 블록(`br`, `phi`)과 파이프라인 스테이지별 SMEM 링버퍼 인덱스(`rem`, `xor`) 연산 | `cp.async.commit_group;` / `cp.async.wait_group N;` 및 루프 레이블 분기(`@p bra LOOP;`) |
| `accumulator = tl.dot(a, b, accumulator)` | `tt.dot %a, %b, %acc : tensor<BMxBKxf16> * tensor<BKxBNxf16> -> tensor<BMxBNxf32>` | SMEM(`#ttg.shared` / `#ttg.nvmma_shared`)에서 레지스터(`#ttg.dot_op`)로 `ttg.local_load` 후, `#ttg.nvidia_mma` 레이아웃의 `tt.dot` 수행 | Shared Memory에서 행렬 조각을 읽는 `ldmatrix`와 Tensor Core `mma.sync` intrinsic 연속 호출 | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` + `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| `tl.store(c_ptrs, c, mask=c_mask)` | `arith.truncf`로 FP16 변환 후 `tt.store %c_ptrs, %c, %c_mask` | `#ttg.nvidia_mma` 레지스터 배치에서 글로벌 병합 쓰기에 유리한 `#ttg.blocked`(또는 SMEM 경유) 레이아웃으로 `ttg.convert_layout` 후 `tt.store` | FP32→FP16 패킹 및 조건부 글로벌 메모리 저장 명령 생성 | `cvt.rn.f16x2.f32` 후 `@p st.global.v2.b32 [ptr], {r1, r2};` |

---

## 함께 기록할 것

| 항목 | 기록 |
|---|---|
| 맡은 질문 / 핵심 용어 3개 | |
| 2D 포인터 블록 · L2 그룹핑 · 컴파일 파이프라인 그림 | 위 도식 중 맡은 주제에 직접 주석 달기 |
| 후보 1·2·3 손계산 비교표 및 64 KiB 한계 판정 | |
| GPU / Compute Capability / Python / PyTorch / Triton 버전 | 미실행이면 ‘손계산 / CPU 해석’ |
| 정확도 확인 (`512×512`, `500×600×300` `allclose`) | |
| 선택된 Autotune Config / `4096³` ms · TFLOPs/s / `.metadata` · `.asm` 확인 | |
| 읽은 링크·절 / 함께 이야기할 질문 | |

### 모임에서 함께 이야기할 질문

1. **타일 크기와 K축 폭(`BLOCK_SIZE_K`)의 역할 차이**: 왜 `BLOCK_SIZE_K`를 32에서 64로 키워도 HBM 총 이동량 $Q$와 산술 강도 $I$는 바뀌지 않을까? 그렇다면 `BLOCK_SIZE_K`는 무엇을 위해 조정하는 걸까?
2. **L2 그룹핑과 하드웨어 스케줄링**: `GROUP_SIZE_M`을 너무 크게(예: `num_pid_m` 전체) 잡으면 다시 열 우선(column-major) 순서와 같아져서 L2 캐시 적중률이 떨어지는 이유는 무엇일까?
3. **8주차 연결(Legato를 향하여)**: Triton에서는 작성자가 `BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_stages`, `num_warps`를 직접 후보로 적어주어야 했다. 새로운 가속기나 다중 메모리 계층을 대상으로 할 때, 알고리즘(무엇을 계산할지)과 스케줄/매핑(어떻게 타일링·배치·파이프라이닝할지)을 어떻게 분리하거나 자동화해야 할까?

공식 문서 확인: 2026-10-29. 이 문서의 손계산 시간은 가상 하드웨어(`P = 80 TFLOPs/s`, `BW = 1 TB/s`) 기준이며 GPU 실측값이 아니다.
