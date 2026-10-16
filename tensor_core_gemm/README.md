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
| 0 | cuBLAS `COMPUTE_32F` | SIMT FP32 기준선 | kernel 0 |
| 1 | [01_wmma_naive](kernels/01_wmma_naive.cuh) | `nvcuda::wmma` 16×16×8 fragment, global에서 직접 load | kernel 1 |
| 2 | [02_wmma_smem_warptile](kernels/02_wmma_smem_warptile.cuh) | SMEM block tile 128×128×16, warp tile 64×32, padding | kernel 5, 10 |
| 3 | [03_mma_sync_tf32](kernels/03_mma_sync_tf32.cuh) | PTX `mma.sync.m16n8k8`, lane별 fragment 레이아웃을 직접 계산 | — |
| 4 | [04_cp_async_pipeline](kernels/04_cp_async_pipeline.cuh) | `cp.async` 3-stage SMEM 파이프라인 (Ampere 전용) | kernel 11, 12 |
| 5 | [05_swizzle_tuned](kernels/05_swizzle_tuned.cuh) | XOR swizzle, BK=32, 128×256 타일, SMEM 경유 coalesced epilogue | — |
| 6 | [06_vectorized_frag](kernels/06_vectorized_frag.cuh) | k·n 순서 재배치로 fragment를 `LDS.128`로, 정수 덧셈 1개로 TF32 반올림, 주소 계산을 루프 밖으로 | kernel 6 (vectorize) |
| 7 | [07_frag_double_buffer](kernels/07_frag_double_buffer.cuh) | fragment 더블 버퍼링(warp 수준 파이프라인), barrier 앞당기기, block rasterization | — |
| 100 | cuBLAS `COMPUTE_32F_FAST_TF32` | Tensor Core 목표 상한 | — |

PTX 래퍼(`cvt.rna.tf32`, `mma.sync`, `cp.async`)와 fragment 레이아웃 표는 [common.cuh](kernels/common.cuh)에 있습니다.

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
make bench                 # 0, 100, 1~7 전부 실행 → results/*.txt + GFLOPS 표
make regs                  # 커널별 레지스터·spill
make profile K=5 SIZE=4096 # Nsight Compute (ncu) 보고서
```

튜닝용 환경변수: `TC6_CFG`, `TC7_CFG`로 타일 구성을 고정할 수 있습니다(번호는 [runner.cu](runner.cu) 주석 참고). `TC7_GROUP=0`은 rasterization을 끕니다(기본 8). `TC6_ROUND=0`은 반올림 없이 truncate하는 실험 옵션입니다.

Colab에서는 [tensor_core_gemm_a100.ipynb](../notebooks/tensor_core_gemm_a100.ipynb)를 A100 런타임에서 실행합니다.
빌드 후 모든 커널을 cuBLAS FP32·TF32와 비교합니다. 정확도, TFLOPs, `% of cuBLAS TF32` 표와 그래프, SASS 명령 수를 출력하고 결과를 JSON으로 저장합니다.

로컬에서 [Colab CLI](https://pypi.org/project/google-colab-cli/)로 직접 실행할 수도 있습니다. 아래는 결과표를 측정한 방식입니다.

```bash
tar czf /tmp/tcgemm.tgz --exclude=build --exclude=results tensor_core_gemm
colab new -s tcgemm-a100 --gpu A100
colab upload -s tcgemm-a100 /tmp/tcgemm.tgz /content/tcgemm.tgz
echo 'import subprocess; print(subprocess.run("cd /content && tar xzf tcgemm.tgz && make -C tensor_core_gemm && cd tensor_core_gemm && for k in 0 100 1 2 3 4 5 6 7; do ./build/tc_gemm $k; done", shell=True, capture_output=True, text=True).stdout)' \
  | colab exec -s tcgemm-a100 --timeout 1800
colab stop -s tcgemm-a100
```

## 정확도 검증

크기마다 같은 초기 C에서 cuBLAS FP32 결과와 비교합니다.
TF32는 입력 반올림 오차가 약 2⁻¹¹이어서 SGEMM_CUDA의 원소별 절대 오차 `0.01` 기준은 큰 K에서 통과하지 못합니다.
대신 `max|diff| / max|ref| < 1e-3`을 통과 기준으로 사용하고 실제 값을 출력합니다.
`./build/tc_gemm 100`(cuBLAS TF32)의 상대 오차가 TF32 오차의 기준값입니다.

## 결과

Colab A100-SXM4-40GB(CUDA 13.0.88)에서 2026-10-15에 측정했습니다. 단위는 TFLOPs/s이고, 50회 반복의 평균입니다.
같은 바이너리를 두 번 실행했으며, 두 결과의 차이는 4096의 kernel 2·3·5에서 2~4%였고 나머지는 1% 이내였습니다. 아래 표는 첫 번째 실행 값입니다.
원시 로그는 [a100_2026-10-15-opt.log](a100_2026-10-15-opt.log)(kernel 0~7)와 [a100_2026-10-15.log](a100_2026-10-15.log)(최초 kernel 0~5 측정)에 있습니다.

| kernel | 1024 | 2048 | 4096 | 8192 | cuBLAS TF32 대비 (8192) | cuBLAS FP32 대비 (8192) |
|---|---:|---:|---:|---:|---:|---:|
| 0 cuBLAS FP32 | 12.9 | 13.7 | 18.1 | 19.2 | 14% | 1.00× |
| 1 WMMA naive | 9.0 | 9.3 | 12.4 | 12.6 | 9% | 0.66× |
| 2 WMMA + SMEM | 11.6 | 15.4 | 26.2 | 31.1 | 23% | 1.62× |
| 3 mma.sync | 14.2 | 18.4 | 36.7 | 38.8 | 29% | 2.02× |
| 4 cp.async | 23.2 | 39.5 | 52.8 | 70.7 | 53% | 3.68× |
| 5 swizzle | 27.7 | 35.0 | 59.3 | 82.1 | 61% | 4.28× |
| 6 vectorized frag | 36.9 | 51.5 | 71.0 | 92.9 | 69% | 4.84× |
| 7 frag double buffer | **40.9** | **58.3** | **84.6** | **117.9** | **88%** | **6.14×** |
| 100 cuBLAS TF32 | 50.0 | 74.3 | 104.0 | 134.6 | 100% | 7.01× |

kernel 7은 cuBLAS TF32 대비 1024에서 82%, 2048에서 78%, 4096에서 81%, 8192에서 88%입니다.

**정확도**: 모든 커널이 모든 크기에서 PASS했습니다(2회 실행, 112/112). cuBLAS FP32 대비 상대 오차는 3.6e-4~4.3e-4로, cuBLAS TF32(3.7e-4)와 같은 수준입니다.

### 결과 해석 (kernel 1 → 5)

- **Tensor Core로 바꾸기만 해서는 빨라지지 않습니다.** kernel 1(12.6)은 Tensor Core를 쓰지만 데이터 재사용이 없어 cuBLAS FP32 SIMT(19.2)보다 느립니다. SMEM 타일링(kernel 2)을 넣은 뒤에야 SIMT 피크(19.5)를 넘습니다.
- **`cp.async`의 효과가 큽니다(+82%, 38.8 → 70.7).** FP32 SIMT에서는 double buffering(SGEMM_CUDA kernel 11)의 효과가 작았습니다. Tensor Core에서는 연산이 빨라 load 지연이 그대로 드러나므로 파이프라인의 효과가 큽니다.

## ncu 병목 분석과 kernel 6·7

가설, 측정값, 기각된 시도까지 포함한 전체 과정은 [WORKLOG.md](WORKLOG.md)에 있습니다.

아래 수치는 4096에서 측정한 ncu 값입니다. ncu가 클럭을 고정하므로 시간의 절댓값은 벤치마크와 다릅니다.

| | Tensor 파이프 사용률 | 실행 명령 | 레지스터 | 주요 stall (issue당) |
|---|---:|---:|---:|---|
| kernel 5 (128×256) | 51.9% | 441M | 222 | wait 1.28, math 1.29 |
| kernel 6 (256×128) | 62.2% | 199M | 227 | wait 2.68, math 2.14, mio 0.77, barrier 0.61 |
| kernel 7 (256×128) | 73.9% | 200M | 251 | wait 2.18, math 2.15, mio 0.66, barrier 0.08 |
| cuBLAS TF32 | 91.4% | 281M | 230 | math 1.67, wait 0.99 |

cuBLAS가 고른 커널은 `cutlass_80_tensorop_s1688gemm_128x256_32x3_nn`입니다. 타일 128×256, BK=32, 3-stage, 8 warp로, kernel 5와 **같은 타일 구성**입니다. 그래서 차이는 타일 크기가 아니라 warp 안의 명령 흐름에 있습니다.

1. **kernel 5: 명령이 너무 많음.** HMMA는 이론상 6,710만 개인데 실행 명령은 4억 4,100만 개였습니다. SASS를 보니 `cvt.rna.tf32.f32`가 FSETP·IMAD·LOP3 여러 개로 풀려 있었고, fragment마다 스칼라 LDS를 썼습니다. 이 두 가지가 HMMA 1개당 6.45개의 명령을 만들었습니다. SM당 warp가 8개뿐이라 명령 발행이 Tensor 파이프를 따라가지 못합니다.
2. **kernel 6 첫 버전: 명령을 줄였는데 속도가 같음.** LDS.128과 IADD 반올림으로 명령이 23% 줄었지만 8192 성능은 81.1로 kernel 5(82.0)와 같았습니다. 반올림을 아예 빼도(truncate) 67.8로 오히려 느려졌습니다. 루프 SASS를 보니 IMAD·LEA·LOP3·SHF가 HMMA 1개당 2개씩 남아 있었습니다. 레지스터가 239개로 꽉 차서, 컴파일러가 swizzle 주소를 매 타일 다시 계산하고 있었습니다.
3. **kernel 6: 주소를 "기준값 + 상수"로.** 루프 명령이 HMMA 128개당 632개에서 370개로 줄었습니다. 8192에서 99.4(cfg 1), Tensor 파이프 62%가 됐습니다. 실행 명령 수는 cuBLAS보다 적지만 파이프 사용률은 여전히 낮았습니다.
4. **PC sampling.** 남은 손실은 fragment LDS 결과 대기(MIO 9%, IADD의 short scoreboard 5%), 타일 경계 barrier 7%, epilogue의 C 로드 5%였습니다. 원인은 fragment load가 같은 step의 mma 바로 앞에서 일어나 지연이 그대로 드러난다는 점이었습니다.
5. **kernel 7: fragment 더블 버퍼링.** 다음 step의 fragment를 먼저 요청하고 barrier를 앞당겼습니다. barrier stall이 0.61에서 0.08로 줄고 Tensor 파이프가 74%가 됐습니다. 8192에서 114.5 → rasterization(group 8)으로 117.9입니다.

### 남은 차이 (8192에서 12%, 그 외 크기에서 18~22%)

- `wait` stall이 cuBLAS의 2배 이상입니다. PC sampling에서는 대부분 HMMA 명령 자체에 걸렸습니다. 원인은 HMMA 순서나 레지스터 재사용 패턴 차이로 추정하지만 확인하지 않았습니다.
- epilogue에서 C를 읽는 동안 Tensor Core가 쉽니다(PC sampling에서 long scoreboard 약 5%). SM당 block이 1개라 다른 block과 겹치지 않습니다.
- 작은 크기(1024~2048)에서는 block 수가 적어 SM을 다 채우지 못합니다. split-K나 stream-K로 해결할 수 있습니다.
- 4096 이하에서 kernel 7은 128×128 3-stage 구성(SM당 block 1개)을 씁니다. 크기별 최적 구성 탐색은 cfg 4개 중에서만 했습니다.

## 작성 시 확인한 것 (GPU 없이)

- 모든 커널을 로컬 CUDA 12.4와 Colab CUDA 13.0에서 `-arch=sm_80`로 경고·spill 없이 빌드했습니다.
- kernel 3~7의 SMEM 인덱스, swizzle, k·n 순열, `m16n8k8` fragment 매핑을 Python으로 시뮬레이션했습니다. 블록 하나의 결과가 행렬곱과 일치했고, 모든 fragment load(LDS.32/64/128)에서 bank conflict가 없었습니다.

### 추가 튜닝 실험 (2026-10-16, 진행 중)

kernel 7에서 `TC7_ORDER=0`은 기존 행 우선, `1`은 행 serpentine,
`2`는 열 serpentine HMMA 순회다. `3`은 LDMATRIX A + 행 우선,
`4`는 LDMATRIX A + 행 serpentine이다. 자동 설정은 작은 정사각 행렬에서 `0`,
4096~8192에서 `3`을 선택한다.
`TC7_CFG=4/5`는 64×128 / 128×64 타일(32×32 warp, 72 KiB SMEM),
`6`은 64×64 타일(32×32 warp, 48 KiB SMEM)이다.

GPU에서 아래 명령으로 정확도를 통과한 측정만 비교할 수 있다.
실행 순서를 섞어 여러 번 측정하고 중앙값과 cuBLAS TF32 대비 비율을 저장한다.

```bash
cd tensor_core_gemm
make
python3 tune.py --rounds 3
# 후보만 재측정
python3 tune.py --sizes 1024 2048 --configs 2 4 5 6 --orders 0 1 2 --rounds 3
```

원시 로그와 `summary.json`은 `results/tuning/`에 저장된다.


전체 조합은 네이티브 튜너로 더 빠르게 비교할 수 있다. 입력 생성, 할당,
FP32 reference를 크기당 한 번만 수행하고 같은 입력으로 모든 조합을 검증한다.
`legacy`는 최적화 전의 크기별 설정이며, `default`는 현재 기본 설정이다.

```bash
make build/tc_tune
mkdir -p results
./build/tc_tune 8192 128 3 > results/native.jsonl  # max, min, rounds
```

`TC7_COPY=1`은 다음 타일의 global→SMEM 복사 명령을 4개 MMA 단계에 분산한다.
복사 그룹은 마지막 단계에서 한 번 commit하며, stage 전환 전에 필요한 그룹을 기다린다.
정사각 행렬의 자동 복사 설정은 128~512와 4096~8192에서 `1`, 1024~2048에서 `0`이다.
256×128 타일을 선택하면 큰 행렬에서 단계별 포인터를 유지하는 `TC7_COPY=2`를 쓴다.
네이티브 튜너는 세 복사 방식을 모두 비교한다.


현재 자동 타일은 정사각 행렬 기준 128~1024에서 cfg 6, 2048에서 cfg 4,
4096~8192에서 cfg 0이다. 특정 조합만 네이티브 튜너에서 재측정하려면
`TC_TUNE_VARIANTS`에 이름을 쉼표로 구분해 지정한다. cuBLAS, legacy, default는
항상 함께 측정한다. 입력 생성 RNG와 실행 순서 RNG는 분리되어 있다.

```bash
TC_TUNE_VARIANTS=cfg1-order0-copy0,cfg1-order0-copy1 ./build/tc_tune 8192 4096 5
```

### 1140 MHz에서 확인한 v3 기본 경로 (2026-10-16)

A100-SXM4-40GB에서 SM 1140 MHz, 메모리 1215 MHz로 비교했다.
기존 구성과 v3 자동 설정, cuBLAS TF32를 같은 입력으로 섞어 5회 실행한 중앙값이다.
이 표의 낮은 클럭 때문에 위의 2026-10-15 절대 TFLOPS와 직접 비교하면 안 된다.

| 크기 | 기존 구성 TFLOPS | 새 기본 TFLOPS | cuBLAS TF32 TFLOPS | cuBLAS 대비 |
|---|---:|---:|---:|---:|
| 128 | 0.293 | 0.507 | 0.515 | 98.5% |
| 256 | 1.722 | 3.180 | 2.686 | 118.4% |
| 512 | 8.954 | 18.109 | 19.105 | 94.8% |
| 1024 | 42.679 | 46.712 | 52.113 | 89.6% |
| 2048 | 60.634 | 69.088 | 77.181 | 89.5% |
| 4096 | 87.917 | 95.885 | 108.227 | 88.6% |
| 8192 | 96.243 | 105.368 | 117.328 | 89.8% |

빌드 소스 manifest 일치를 확인했고, 네이티브 정확도 검증 105/105 및 기존 CLI의
kernel 7·100에서 각각 7/7 PASS했다. 원시 측정값은
[a100_2026-10-16-default-summary.json](a100_2026-10-16-default-summary.json)에 있다.
8192 Tensor 파이프의 elapsed 사용률은 기존 76.6% → 분산 복사 83.9%였고,
cuBLAS는 93.4%였다. 아직 cuBLAS와 같은 활용률에 도달한 상태는 아니다.


이후 단계별 포인터 경로(copy2)는 4096에서 97.37, 8192에서 107.15 TFLOPS로
개선됐다(1140 MHz, 5회 중앙값). 같은 실행의 cuBLAS는 각각 108.23, 117.37이다.
8192 Tensor elapsed 사용률은 copy1 83.28% → copy2 85.21%였고 register spill은 없었다.
추가 C prefetch 실험의 진행 상태와 코드별 측정 근거는 [WORKLOG.md](WORKLOG.md)에 기록한다.


LDMATRIX 경로(cfg0/order3/copy1)는 5회 재확인에서 4096 **99.69**, 8192
**110.05 TFLOPS**였다. 같은 실행의 cuBLAS는 108.24 / 117.46 TFLOPS였으며,
8192 Tensor elapsed 사용률은 87.42%였다(기존 copy2 85.22%, cuBLAS 93.38%).
이 조합을 큰 정사각 행렬의 자동 설정에 반영했다.

`TC7_EPI=4/8`은 order3에서 C 값을 4/8개 float4씩 먼저 읽는 epilogue 실험이다.
자동 경로는 1024 이하에서 cfg6/order3/copy0/epi4, 2048에서
cfg4/order3/copy0/epi4, 4096 이상 큰 정사각 행렬에서 cfg0/order6/copy4/epi4를 선택한다.
작은 행렬 설정은 같은 입력·1140 MHz·무작위 실행 순서의 5회 측정으로 확인했다.
이 자동 정책은 새 빌드에서 전체 정확도 검사와 크기별 5회 측정으로 재확인했다.
비교하려면 `TC7_CFG`, `TC7_ORDER`, `TC7_COPY`, `TC7_EPI`를 함께 고정한다.


C 버퍼 수를 제한한 epilogue4 경로는 독립 5회 재확인에서 4096 **104.16**,
8192 **112.98 TFLOPS**였다(같은 실행의 cuBLAS 108.25 / 117.41, 약 **96.2%**).
8192 Tensor elapsed 사용률은 87.39% → **89.99%**, C 읽기 long scoreboard/issue는
0.22 → 0.06으로 줄었다. cuBLAS 사용률 93.38%까지는 아직 차이가 남아 있다.
측정값은 [epilogue summary](a100_2026-10-16-epilogue-summary.json)에 보존했다.

B 복사 배치를 변경한 최신 후보는 스레드가 K 행을 나눠 맡고 N 방향의 연속
구간을 복사한다. 같은 글로벌 기준 주소와 상수 오프셋을 재사용하며,
order6의 자연 N 배치·행 serpentine MMA와 함께 사용한다.
`TC7_COPY=4`가 이 경로이며 `TC7_ORDER=6`, `TC7_EPI=4`, `TC7_ROUND=0`을 요구한다.

교체된 A100에서 1140 MHz·동일 입력·무작위 순서의 5회 중앙값은 다음과 같다.

| 크기 | 이전 경로 TFLOPS | B 복사 변경 TFLOPS | cuBLAS TFLOPS | cuBLAS 대비 |
|---:|---:|---:|---:|---:|
| 4096 | 104.48 | 107.40 | 108.55 | 98.9% |
| 8192 | 113.42 | 116.57 | 117.89 | 98.9% |

8192 Tensor elapsed 사용률은 이전 경로 **90.11% → 기본 copy4 경로 92.34%**,
cuBLAS는 **93.43%**다. 아직 cuBLAS와 같은 활용률에 도달하지 않았다.
기본 경로를 새로 빌드해 128~8192의 5회 측정과 기존 CLI 실행을 모두 확인했다.
[측정 요약](a100_2026-10-16-rake-summary.json)에 반복 측정값을 보존했다.

`make check`는 직사각형·짧은 K·alpha/beta 조합과 행 그룹 경계를 검사한다.
원래 FP32 입력의 cuBLAS 결과에는 상대오차 1e-3, CUDA TF32 변환 명령으로
별도 변환한 입력의 FP32 GEMM 결과에는 1e-5를 적용한다.
현재 기본 경로와 모든 지원 설정은 **13,224건**을 통과했으며,
최대 오차는 각각 **3.13e-4 / 1.10e-6**였다.
