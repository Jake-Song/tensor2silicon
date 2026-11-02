# A100 Tensor Core GEMM (TF32) 단계별 구현

[SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)는 FP32 SIMT 코어만으로 GEMM을 최적화합니다.
이 디렉터리는 같은 FP32 입출력(`C = α·A·B + β·C`, row-major)을 유지한 채 A100의 **TF32 Tensor Core**로 같은 계산을 수행합니다.

| A100 (SXM, dense) | 피크 |
|---|---|
| FP32 SIMT (SGEMM_CUDA가 쓰는 경로) | 19.5 TFLOPS |
| TF32 Tensor Core (이 코드) | 156 TFLOPS |

TF32는 FP32와 지수 범위(8비트)가 같고 mantissa만 10비트로 줄인 형식입니다.
입력을 TF32로 반올림한 뒤 곱하고, 누적은 FP32로 합니다.

## 단계별 커널

각 단계는 이전 단계에 개념 하나만 추가합니다.

| # | 파일 | 추가되는 개념 | SGEMM_CUDA 대응 |
|---|---|---|---|
| 1 | [01_wmma_naive](kernels/01_wmma_naive.cuh) | `nvcuda::wmma` 16×16×8 fragment, global에서 직접 load | kernel 1 |
| 2 | [02_wmma_smem_warptile](kernels/02_wmma_smem_warptile.cuh) | SMEM block tile 128×128×16, warp tile 64×32, padding | kernel 5, 10 |
| 3 | [03_mma_sync_tf32](kernels/03_mma_sync_tf32.cuh) | PTX `mma.sync.m16n8k8`, lane별 fragment 레이아웃을 직접 계산 | — |
| 4 | [04_cp_async_pipeline](kernels/04_cp_async_pipeline.cuh) | `cp.async` 3-stage SMEM 파이프라인 (Ampere 전용) | kernel 11, 12 |
| 5 | [05_swizzle_tuned](kernels/05_swizzle_tuned.cuh) | XOR swizzle, BK=32, 128×256 타일, SMEM 경유 coalesced epilogue | — |
| 6 | [06_vectorized_frag](kernels/06_vectorized_frag.cuh) | k·n 순서 재배치로 fragment를 `LDS.128`로, 정수 덧셈 1개로 TF32 반올림, 주소 계산을 루프 밖으로 | kernel 6 (vectorize) |
| 7 | [07_frag_double_buffer](kernels/07_frag_double_buffer.cuh) | fragment 더블 버퍼링(warp 수준 파이프라인), barrier 앞당기기, block rasterization | — |
| 8 | [08_ldmatrix_distributed](kernels/08_ldmatrix_distributed.cuh) | 7 + A LDMATRIX / natural-K 배치, 분산 `cp.async` | — |
| 9 | [09_prefetch_epilogue](kernels/09_prefetch_epilogue.cuh) | 8 + C prefetch, 두 개의 제한된 레지스터 버퍼로 epilogue 중첩 | — |
| 10 | [10_raked_b_copy](kernels/10_raked_b_copy.cuh) | 9 + B의 N방향 복사, natural-N / serpentine MMA, MMA 뒤 복사 발행 | — |
| 0 | cuBLAS `COMPUTE_32F` | SIMT FP32 기준선 | kernel 0 |
| 100 | cuBLAS `COMPUTE_32F_FAST_TF32` | Tensor Core 목표 상한 | — |
| 101 | [cuBLASLt TF32](kernels/99_library_gemm.cuh) | row-major descriptor, heuristic 알고리즘 선택, workspace 재사용 | — |
| 102 | [CUTLASS TF32](kernels/99_library_gemm.cuh) | 공개 TensorOp 구성, 3-stage pipeline (선택 빌드) | — |

PTX 래퍼(`cvt.rna.tf32`, `mma.sync`, `cp.async`)와 fragment 레이아웃 표는 [common.cuh](kernels/common.cuh)에 있습니다.

### 7번 이후 단계 분리

기존 7번 파일에 누적되어 있던 최적화를 7~10의 독립적인 커널 본문으로 분리했다.
8번은 분산 복사와 A LDMATRIX, 9번은 C prefetch epilogue, 10번은 B 복사 배치 개선을 추가한다.
각 파일은 다른 단계의 커널을 호출하는 래퍼가 아니라 해당 단계까지의 전체 계산을 담는다.
라이브러리 구현은 [99_library_gemm.cuh](kernels/99_library_gemm.cuh)에 두며 실행 번호 100~102는 유지한다.

```bash
make
./build/tc_gemm 7 8192 4096
./build/tc_gemm 8 8192 4096
./build/tc_gemm 9 8192 4096
./build/tc_gemm 10 8192 4096
```

단계 비교에서는 `TC7_*` 환경변수를 해제한다. 7~10은 같은 shape에 같은 타일 정책과 group=8을
사용하고 복사·fragment·epilogue 구현을 달리한다. 큰 정방행렬은 128×256×32, 3-stage,
warp tile 64×64다. 작은 shape는 64×64 또는 64×128 타일을 사용한다.
기존 초기 7번의 크기별 타일 정책까지 재현하는 것은 아니므로 과거 표와 새 측정을 구분한다.

**기존 튜닝 호환성:** 설정 조합 전체와 이전 자동 정책은
[detail/frag_pipe_tuning.cuh](kernels/detail/frag_pipe_tuning.cuh)에 보존했다.
네이티브 튜너와 전체 설정 검사는 `run_configured_frag_pipe`로 이 구현을 사용한다.
CLI에서 7번에 `TC7_*` 환경변수를 지정해도 이전 설정 조합 경로를 사용한다.
8~10은 고정된 학습 단계이므로 `TC7_*`를 사용하지 않는다.

최신 결과 표는 분리 전 자동 정책의 측정이며, 큰 타일의 최종 경로는 현재 10번에 대응한다.
단계별 과거 측정과 튜닝 과정은 [WORKLOG.md](WORKLOG.md)에 보관한다.

### 핵심 포인트

- **1 → 2**: kernel 1은 Tensor Core를 쓰지만 데이터 재사용이 없어 메모리 대역폭에 묶입니다. SMEM 타일링으로 global load 한 번당 계산량(산술 강도)을 높입니다.
- **2 → 3**: WMMA는 fragment 내부 배치를 숨깁니다. `mma.sync`를 직접 쓰면 lane `g = lane/4`, `t = lane%4`가 어떤 원소를 들고 있는지 알 수 있습니다. 그래서 padding(`LDA % 8 == 4`, `LDB % 32 == 8`)으로 fragment load의 bank conflict를 정확히 없앨 수 있습니다.
- **3 → 4**: `cp.async`는 global → SMEM 복사가 레지스터를 거치지 않으므로, 다음 타일 복사와 현재 타일의 mma가 겹칩니다. `wait_group<STAGES-2>`로 필요한 타일만 기다립니다.
- **4 → 5**: padding 대신 16바이트 chunk 단위 XOR swizzle을 씁니다. A는 `chunk ^ (row % 8)`, B는 `chunk ^ ((row % 4) * 2)`입니다. 큰 타일과 BK=32로 SMEM load당 mma 수를 늘리고, epilogue는 SMEM을 거쳐 float4로 C를 씁니다.
- **5 → 6**: mma는 k에 대한 합이라 A와 B에 같은 k 순열을 쓰면 결과가 같습니다. 출력 열의 순서도 epilogue에서 되돌리면 바꿀 수 있습니다. 이 두 가지로 lane이 필요한 값을 연속으로 만들어, 스칼라 LDS 대신 `LDS.128`로 읽습니다. `cvt.rna.tf32`는 sm_80에서 정수 명령 3~4개로 풀리므로 `bits + 0x1000` 하나로 바꿉니다(mma가 하위 13비트를 무시). 또 모든 SMEM 주소를 "스레드별 기준값 + 상수"로 만들어, 레지스터가 부족해도 루프 안에서 swizzle을 다시 계산하지 않게 합니다.
- **6 → 7**: CUTLASS `MmaMultistage`와 같은 구조입니다. k8 step i의 mma 전에 step i+1의 fragment LDS를 먼저 내고, 타일 경계의 wait·barrier를 마지막 step 직전으로 옮깁니다. fragment 레지스터가 2배가 되므로 k 순열을 step 단위로 바꿔 A를 conflict 없는 `LDS.64`로 읽습니다.

모든 커널은 M·N·K가 타일 크기의 배수라고 가정합니다. SGEMM_CUDA도 같은 가정을 씁니다. 조건이 맞지 않으면 실행 전에 오류를 출력합니다.

## 빌드와 실행 (A100, sm_80)

CUDA 11.0 이상의 `nvcc`와 cuBLAS가 필요합니다. sm_80 미만 GPU에서는 실행을 거부합니다.

```bash
cd tensor_core_gemm
make                       # build/tc_gemm
./build/tc_gemm 7          # kernel 7, 128 ~ 8192 정사각 행렬
./build/tc_gemm 4 4096     # 최대 크기 지정
./build/tc_gemm 7 4096 4096  # 한 크기만 (ncu용)
make bench                 # 0, 100, 1~10 전부 실행 → results/*.txt + GFLOPS 표
make regs                  # 커널별 레지스터·spill
make profile K=5 SIZE=4096 # Nsight Compute (ncu) 보고서
```

튜닝용 환경변수: `TC6_CFG`, `TC7_CFG`로 타일 구성을 고정할 수 있습니다(번호는 [runner.cu](runner.cu) 주석 참고). `TC7_GROUP=0`은 rasterization을 끕니다(기본 8). `TC6_ROUND=0`은 반올림 없이 truncate하는 실험 옵션입니다.

Colab에서는 [tensor_core_gemm_a100.ipynb](../notebooks/tensor_core_gemm_a100.ipynb)를 A100 런타임에서 실행합니다.
빌드 후 모든 커널을 cuBLAS FP32·TF32와 비교합니다. 정확도, TFLOPs, `% of cuBLAS TF32` 표와 그래프, SASS 명령 수를 출력하고 결과를 JSON으로 저장합니다.

로컬에서 [Colab CLI](https://pypi.org/project/google-colab-cli/)로 직접 실행할 수도 있습니다. 단계별 커널을 실행하는 예시입니다.

```bash
tar czf /tmp/tcgemm.tgz --exclude=build --exclude=results tensor_core_gemm
colab new -s tcgemm-a100 --gpu A100
colab upload -s tcgemm-a100 /tmp/tcgemm.tgz /content/tcgemm.tgz
echo 'import subprocess; print(subprocess.run("cd /content && tar xzf tcgemm.tgz && make -C tensor_core_gemm && cd tensor_core_gemm && for k in 1 2 3 4 5 6 7 8 9 10 0 100; do ./build/tc_gemm $k; done", shell=True, capture_output=True, text=True).stdout)' \
  | colab exec -s tcgemm-a100 --timeout 1800
colab stop -s tcgemm-a100
```

## 정확도 검증

크기마다 같은 초기 C에서 cuBLAS FP32 결과와 비교합니다.
TF32는 입력 반올림 오차가 약 2⁻¹¹이어서 SGEMM_CUDA의 원소별 절대 오차 `0.01` 기준은 큰 K에서 통과하지 못합니다.
대신 `max|diff| / max|ref| < 1e-3`을 통과 기준으로 사용하고 실제 값을 출력합니다.
`./build/tc_gemm 100`(cuBLAS TF32)의 상대 오차가 TF32 오차의 기준값입니다.

## 최신 검증 결과

2026-10-16의 최종 v13 기본 경로를 Colab A100-SXM4-40GB에서 측정했습니다.
SM 클럭 1140 MHz, 메모리 클럭 1215 MHz를 유지하고 동일 입력의 실행 순서를 섞어
5회 측정한 중앙값입니다. 연산은 `C = 0.5AB + 3C`, 입력·출력은 FP32,
곱셈은 TF32, 누산은 FP32입니다. cuBLAS TF32는 같은 실행의 비교 기준입니다.

| M=N=K | 자체 자동 경로 (TFLOPS) | cuBLAS TF32 기준 (TFLOPS) | cuBLAS 대비 |
|---:|---:|---:|---:|
| 128 | 0.62 | 0.49 | 125.5% |
| 256 | 3.80 | 2.62 | 145.1% |
| 512 | 20.87 | 19.24 | 108.5% |
| 1024 | 49.45 | 52.18 | 94.8% |
| 2048 | 74.36 | 77.55 | 95.9% |
| 4096 | 107.40 | 108.55 | 98.9% |
| 8192 | 116.57 | 117.89 | 98.9% |

8192의 Tensor elapsed 사용률은 자체 경로 **92.34%**, cuBLAS **93.43%**였습니다.
직사각형·짧은 K·alpha/beta·행 그룹 경계를 포함한 엄격 검사 **13,224건**을 통과했습니다.
최대 정규화 오차는 원래 FP32 입력 기준 **3.13e-4**(허용치 1e-3),
독립적으로 TF32 양자화한 입력 기준 **1.10e-6**(허용치 1e-5)입니다.

이 표는 단계 분리 전 자동 정책의 측정입니다. 큰 타일의 cfg0/order6/copy4/epi4 경로는
현재 10번에 대응하며, 작은 크기는 당시의 cfg6/4·order3/copy0/epi4 정책입니다.
분리한 7~10번의 새 실행 결과는 아직 측정하지 않았습니다. 기본 클럭에서의 결과와도 구분해야 합니다.

[최종 측정값·검증 요약](a100_2026-10-16-rake-summary.json) ·
[실험 과정과 기각한 후보](WORKLOG.md)

## cuBLASLt와 CUTLASS 구현

동일한 row-major FP32 입출력 `C = αAB + βC`를 라이브러리 두 가지로 구현했습니다.
소스는 [99_library_gemm.cuh](kernels/99_library_gemm.cuh)입니다.

| 번호 | 구현 | 주요 설정 |
|---|---|---|
| 101 | cuBLASLt | `CUBLAS_COMPUTE_32F_FAST_TF32`, row-major descriptor, heuristic 후보 1개, workspace 상한 32 MiB |
| 102 | CUTLASS (선택 빌드) | SM80 TensorOp, block 128×128×32, warp 64×32×32, MMA 16×8×8, 3-stage pipeline |

CUTLASS는 FP32 저장 형식에서 TF32로 변환하여 곱하고 FP32로 누적합니다.
`OpMultiplyAdd`를 명시하여 여러 TF32 곱으로 정확도를 높이는 TF32x3 경로와 구분합니다.
cuBLASLt의 FAST_TF32는 라이브러리의 연산 허용 모드이며 내부 타일·명령 구성은 라이브러리가 선택합니다.
두 구현의 결과가 비트 단위로 같다고 가정하지 않습니다.

### 빌드와 실행

cuBLASLt는 기본 빌드에 포함됩니다. CUDA Toolkit의 cuBLASLt가 필요합니다.

```bash
cd tensor_core_gemm
make
./build/tc_gemm 101 4096 128
```

CUTLASS는 공식 v3.5.1 헤더로 컴파일을 검증했습니다. 외부 의존성은 저장소에 포함하지 않습니다.
별도 빌드 디렉터리를 사용하므로 기본 실행기와 섞이지 않습니다.

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git /tmp/tc-cutlass
make libraries CUTLASS_DIR=/tmp/tc-cutlass
./build-libraries/tc_gemm 101 4096 128
./build-libraries/tc_gemm 102 4096 128

# 직사각형 행렬, 여러 alpha/beta, shape 변경 후 계획 재선택 검증
make BUILD_DIR=build-libraries \
  NVCCFLAGS='-O3 -std=c++17 -arch=sm_80 -lineinfo -DTC_ENABLE_CUTLASS -I/tmp/tc-cutlass/include' \
  check-libraries
```

cuBLASLt는 첫 호출에서 descriptor·알고리즘·workspace를 준비하고 같은 shape에서는 재사용합니다.
기존 벤치마크는 정확성 검사와 warmup 후 CUDA event로 반복 구간을 측정하므로 첫 준비 비용은 제외합니다.
알고리즘 전수 튜닝이나 CUDA Graph 측정은 아니며, 반복 호출 중 CPU dispatch 간격이 측정에 반영될 수 있습니다.
이 어댑터는 단일 GPU·기본 stream·단일 host thread 실행용입니다.

정확성 검사는 동일한 초기 C를 사용한 cuBLAS FP32 결과와 비교하여
`max|out-ref| / max(max|ref|, 1e-30) < 1e-3`을 확인합니다.
추가 검사 프로그램은 고정 seed의 입력과 직사각형 shape를 사용하고 최대 절대 오차도 출력합니다.

참고: [CUTLASS Ampere TF32 예제](https://github.com/NVIDIA/cutlass/tree/v3.5.1/examples/14_ampere_tf32_tensorop_gemm),
[cuBLASLt API](https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api).
