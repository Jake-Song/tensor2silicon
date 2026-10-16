# Worklog: TF32 Tensor Core GEMM 최적화 (A100)

2026-10-15 · Colab A100-SXM4-40GB · CUDA 13.0.88 · Nsight Compute 2025.3.1

kernel 5 **82.1 → kernel 7 117.9 TFLOPs/s** (8192³): cuBLAS TF32 대비 61% → 88%.
가설마다 ncu·SASS로 원인을 좁히고, 고친 뒤 다시 측정했다. 효과가 없거나 역효과였던 시도도 그대로 남긴다.

## 측정 조건

- 하네스: `./build/tc_gemm <kernel> [max] [min]`. 크기마다 cuBLAS FP32와 비교해 `max|diff|/max|ref| < 1e-3`이면 PASS다. 이후 warmup 1회, 50회 평균을 `cudaEvent`로 잰다.
- 표의 값은 TFLOPs/s(2·N³/시간)다. 별도 표시가 없으면 단일 실행 값이다.
- ncu는 클럭을 고정한다(SM 약 1.09 GHz). 그래서 ncu의 시간은 벤치마크와 절댓값이 다르고, 커널 사이의 비교에만 쓴다.
- 로컬(WSL)에는 GPU가 없다. CUDA 12.4 `nvcc`와 `cuobjdump`로 빌드, 레지스터, SASS를 확인했다. 실행은 `colab` CLI로 A100 세션에서 했다.
- 새 인덱싱(swizzle, 순열, fragment 매핑)은 GPU에 올리기 전에 Python(numpy)으로 블록 하나를 시뮬레이션했다. 결과 일치와 bank conflict 0을 확인한 뒤 올렸다.

## 0. 시작점: kernel 0~5 기준 측정

| kernel | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| 4 cp.async | 23.2 | 39.5 | 52.9 | 69.8 |
| 5 swizzle | 27.8 | 35.1 | 59.5 | 81.6 |
| 100 cuBLAS TF32 | 50.1 | 74.6 | 104.2 | 132.6 |

kernel 5는 cuBLAS TF32의 62%였다. 이 단계에서 원인 후보로 적은 것은 스칼라 LDS, `cvt.rna`, SM당 block 1개, rasterization 없음이었다. 모두 추정이었고 측정하지 않았다.

## 1. ncu: kernel 5는 메모리가 아니라 지연에 묶여 있다

4096, kernel 5(128×256, 3-stage)의 결과다.

| 지표 | 값 |
|---|---:|
| Compute (SM) Throughput | 49.1% |
| Memory Throughput / DRAM | 29.4% / 14.9% |
| L2 hit rate | 78.4% |
| Issued IPC (SM) | 1.71 |
| scheduler당 active / eligible warp | 2.00 / 0.58 |
| 이론·실제 occupancy | 12.5% / 12.5% (레지스터 222개, SMEM 144KB → SM당 1 block) |
| 실행 명령 | 441.3M |

메모리 대역폭은 여유가 있었다. ncu 규칙도 "latency issue"로 판정했다.

4096³의 HMMA(m16n8k8)는 4096³/1024 = 67.1M개다. 그러므로 실행 명령의 85%가 HMMA가 아니었다. kernel 4(128×128, SM당 2 block)도 Compute 50%, 실행 명령 619M으로 같은 양상이었다.

cuBLAS 커널은 처음 kernel 이름 정규식(`tensorop|s1688|tf32|sm80`)이 맞지 않아 수집하지 못했다. 이후 `-c 3`으로 모든 커널을 받아 이름을 확인했다(3절).

## 2. SASS: `cvt.rna.tf32`와 스칼라 LDS가 명령을 불린다

로컬 `cuobjdump`로 kernel 5(128×256)의 메인 루프를 셌다.

| | 루프 명령 | HMMA | HMMA당 | 주요 opcode |
|---|---:|---:|---:|---|
| kernel 5 | 825 | 128 | 6.45 | IMAD 237, LDS 140, FSETP 128, LEA 54 |

`FSETP` 128개는 HMMA 피연산자 수와 같다. sm_80에는 `cvt.rna.tf32.f32` 하드웨어 명령이 없어서, 원소마다 FSETP·IMAD·LOP3 등 여러 개의 정수 명령으로 에뮬레이션된다. 여기에 fragment 원소마다 스칼라 `LDS`가 하나씩 붙는다.

**가설 A**: 반올림과 fragment load의 명령 수를 줄이면 명령 발행이 Tensor 파이프를 따라갈 것이다.

## 3. kernel 6 v1: LDS.128 + 정수 반올림 → 효과 없음

### 변경

- **반올림:** `bits + 0x1000` 하나로 바꿨다. mma.tf32는 하위 13비트를 무시하므로 round-half-away와 같다. inf 입력만 NaN이 된다.
- **k 순열:** mma는 k에 대한 합이므로 A와 B에 같은 순열을 쓰면 결과가 같다. 16개 k 묶음에서 lane t의 step s가 k = 4t+2s, 4t+2s+1을 쓰게 했다. 이렇게 하면 A 한 행에서 lane이 필요한 값 4개가 연속이 되어 `LDS.128` 하나로 읽힌다.
- **n 순열:** fragment n의 열 j를 실제 열 j·NFRAG+n에 대응시켰다. 이렇게 하면 B 한 행에서 lane g가 필요한 값이 연속이 되어 `LDS.128`로 읽힌다. epilogue에서 같은 대응으로 되돌린다.
- **swizzle:** A는 `chunk ^ ((row&1)<<2)`, B는 `chunk ^ f((row>>2)&3)`로 정했다. quarter-warp의 LDS.128이 32 bank에 퍼지도록 고른 값이다. 시뮬레이션에서 결과가 일치했고 conflict는 0이었다.

### 결과

루프 명령은 632개로, HMMA당 6.45에서 4.94로 줄었다. FSETP는 0, LDS는 140개에서 44개가 됐다.

| 구성 (`TC6_CFG`) | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| kernel 5 | 6.3 | 27.7 | 35.0 | 59.3 | 82.0 |
| 0: 128×256 3-stage | 3.8 | 16.9 | 34.6 | 58.9 | 81.1 |
| 1: 256×128 3-stage | 3.8 | 17.2 | 34.9 | 59.5 | 82.4 |
| 2: 128×128 3-stage | 6.5 | 29.2 | 40.4 | 51.8 | 70.6 |
| 3: 128×128 2-stage (SM당 2 block) | 6.4 | 28.7 | 39.8 | 53.8 | 69.8 |
| 4: 128×128 4-stage | 6.4 | 28.6 | 39.7 | 50.6 | 69.4 |
| cuBLAS TF32 | 18.2 | 50.1 | 74.3 | 104.1 | 135.0 |

**판정: 기각.** 명령을 23% 줄였는데도 같은 타일 구성에서 속도가 그대로였다. 따라서 병목은 명령 발행량이 아니다.

## 4. ncu stall 비교: cuBLAS도 같은 타일을 쓴다

4096, issue당 stall이다(`smsp__average_warps_issue_stalled_*_per_issue_active`).

| | Tensor 파이프 | 명령 | 레지스터 | wait | math | barrier | short_sb | long_sb | mio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kernel 5 (128×256) | 51.9% | 441M | 222 | 1.28 | 1.29 | 0.18 | 0.07 | 0.20 | — |
| kernel 6 v1 cfg 0 | 51.6% | 345M | 239 | 1.82 | 1.54 | 0.27 | 0.26 | 0.25 | 0.20 |
| kernel 6 v1 cfg 3 | 44.3% | 436M | 135 | 1.80 | 1.03 | 0.36 | 0.51 | 0.20 | 0.06 |
| cuBLAS TF32 | **91.4%** | 281M | 230 | 0.99 | 1.67 | 0.06 | — | 0.10 | — |

cuBLAS TF32는 `cutlass_80_tensorop_s1688gemm_128x256_32x3_nn` 커널을 쓴다. 이는 128×256 타일, BK=32, 3-stage, 8 warp(grid 512)로 kernel 5·6 cfg 0과 **같은 구성**이다. 레지스터 수(230)와 occupancy도 같다. 그러므로 차이는 타일 선택이 아니라 warp 안의 명령 흐름에서 난다. 가장 큰 차이는 `wait`(고정 지연 의존성 대기)였다.

참고로 cuBLAS FP32 검증 단계에서 호출된 `ampere_sgemm_128x64_nn`은 Tensor 파이프 0%, 명령 2.49G였다.

## 5. 가설 B: 반올림 IADD가 의존성 사슬을 만든다 → 기각

LDS → IADD(반올림) → HMMA 사슬이 `wait`를 만든다고 보고, 반올림을 빼는 실험 옵션 `TC6_ROUND=0`(truncate)을 넣었다.

| | 1024 | 2048 | 4096 | 8192 | 최대 rel err |
|---|---:|---:|---:|---:|---:|
| cfg 0, 반올림 | 17.0 | 34.6 | 58.9 | 81.1 | 3.7e-4 |
| cfg 0, truncate | 14.3 | 29.2 | 49.0 | 67.8 | 6.3e-4 |
| cfg 3, truncate | 24.0 | 33.2 | 41.8 | 57.5 | 6.5e-4 |

**판정: 기각.** 반올림을 빼자 오히려 느려졌다. ncu에서 Tensor 파이프는 42.9%, wait는 2.29였다. 정확도도 나빠졌다(오차 약 2배). 느려진 원인은 확인하지 않았다. 옵션은 실험용으로만 남겼다.

## 6. 가설 C: 레지스터 부족으로 주소를 매 타일 다시 계산한다 → 채택

kernel 6 v1의 루프 SASS를 다시 보니, HMMA 128개 사이에 IMAD 97, LEA 55, LOP3 49, SHF 25가 있었다. HMMA당 약 2개의 정수 주소 계산이다. 레지스터는 239/255였다.

swizzle 주소는 타일과 무관한 값이다. 그런데도 레지스터가 부족해 컴파일러가 루프 밖에 보관하지 못하고 다시 계산(rematerialization)한 것으로 판단했다.

### 변경

모든 SMEM 주소를 "스레드별 기준값 + 컴파일 시간 상수"로 만들었다.
- **cp.async:** 반복마다 행 이동이 swizzle 주기의 배수가 되도록 해서 기준값 하나로 처리했다. global 포인터는 타일마다 `+= BK` 증가시킨다.
- **fragment:** A는 k 묶음별 기준 2개, B는 q별 기준 1~2개로 줄였다. m, h, kk, s 오프셋은 모두 상수다.
- **stage 인덱스:** `tile % STAGES` 대신 증가하고 되감는 카운터를 쓴다.

### 결과

루프 명령은 370개(HMMA당 2.89)가 됐고, 레지스터는 231개로 줄었다.

| 구성 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| 0: 128×256 3-stage | 4.4 | 20.1 | 41.3 | 71.0 | 93.0 |
| 1: 256×128 3-stage | 4.5 | 20.3 | 41.5 | **71.6** | **99.4** |
| 2: 128×128 3-stage | 7.8 | 35.8 | 50.1 | 65.2 | 84.4 |
| 3: 128×128 2-stage | **8.0** | **36.8** | **51.5** | 66.9 | 90.8 |
| 4: 128×128 4-stage | 7.6 | 35.1 | 49.1 | 63.8 | 81.8 |
| cuBLAS TF32 | 18.2 | 50.1 | 74.3 | 104.0 | 135.1 |

**판정: 채택.** 8192에서 82 → 99.4(cuBLAS의 74%)가 됐다.

ncu(cfg 1)는 다음과 같았다.
- Tensor 파이프 62.2%, 실행 명령 199.2M으로 cuBLAS(281M)보다 적다.
- stall은 wait 2.68, math 2.14, mio 0.77, barrier 0.61, long_sb 0.49, short_sb 0.41이었다.

명령 수는 더 이상 문제가 아니다. 남은 것은 지연이다.

## 7. PC sampling: 남은 지연은 어디에 있나

kernel 6 cfg 1, 4096, `SourceCounters`를 opcode별로 합산했다.

| 이유 | 비중 | | opcode | 비중 | 주 원인 |
|---|---:|---|---|---:|---|
| wait | 31.2% | | HMMA | 52.4% | math 23.8, wait 21.5 |
| math_pipe_throttle | 24.7% | | IADD3 | 16.1% | wait 6.0, short_sb 5.3 |
| selected | 11.5% | | LDS | 9.8% | mio 6.9 |
| mio_throttle | 9.3% | | BRA | 7.0% | barrier 6.8 |
| barrier | 6.9% | | FMUL | 5.3% | long_sb 5.2 (epilogue의 C 로드) |
| long_scoreboard | 6.7% | | IMAD | 5.1% | |
| short_scoreboard | 5.4% | | LDGSTS | 2.4% | mio 1.7 |

HMMA에 걸린 대기(math, wait)는 Tensor 파이프가 일하고 있다는 뜻이다. 손실로 볼 수 있는 것은 네 가지다.
- LDS 큐와 LDS 결과 대기: mio 6.9 + short_sb 5.3
- 타일 경계 barrier: 6.8
- epilogue: 5.2

원인은 fragment를 같은 step의 mma 바로 앞에서 읽는다는 점이다. SM당 warp가 8개뿐이라 다른 warp가 이 지연을 가려 주지 못한다.

**가설 D**: CUTLASS `MmaMultistage`처럼 다음 step의 fragment를 미리 읽고 barrier를 앞당기면, 지연이 HMMA 뒤에 숨는다.

## 8. kernel 7: fragment 더블 버퍼링 → 채택

### 변경

- **fragment 레지스터 2벌:** k8 step i의 mma를 내기 전에 step i+1의 LDS를 먼저 낸다.
- **barrier 앞당기기:** 타일 경계의 `cp.async.wait_group` + `__syncthreads`를 마지막 step의 mma 앞으로 옮기고, 다음 타일 step 0의 fragment를 먼저 읽는다. 다음 stage로 쓰는 버퍼의 안전성(직전 barrier 이후 아무도 읽지 않음)과 `wait_group<STAGES-2>` 개수를 확인했다.
- **k 순열 변경:** 레지스터가 2배가 되므로 A를 step 단위(16 레지스터)로 읽어야 한다. k 순열을 step 단위로 바꿔 lane t가 k = 8s+2t, 8s+2t+1을 쓰게 하고 A를 `LDS.64`로 읽었다.
- **swizzle 변경:** half-warp LDS.64가 32 bank에 퍼지도록 A를 `chunk ^ ((row&3)<<1)`로, B는 swizzle 기준을 `t = (row>>1)&3`로 바꿨다. 세 구성 모두 시뮬레이션에서 결과가 일치했고 conflict는 0이었다.
- 레지스터는 256×128에서 251, 128×256에서 255였고 spill은 0이었다.

### 결과

| 구성 (`TC7_CFG`) | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|
| kernel 6 cfg 1 | 4.5 | 20.3 | 41.5 | 71.4 | 99.5 |
| 0: 128×256 3-stage | 5.0 | 23.3 | 48.1 | 84.0 | 100.7 |
| 1: 256×128 3-stage | 5.0 | 23.3 | 48.1 | **84.2** | **114.5** |
| 2: 128×128 3-stage | **8.6** | **40.9** | **58.0** | 76.6 | 96.2 |
| 3: 128×128 4-stage | 8.6 | 40.8 | 57.4 | 75.7 | 86.7 |
| cuBLAS TF32 | 18.2 | 50.1 | 74.3 | 104.3 | 134.0 |

ncu(cfg 1, 4096)를 kernel 6 cfg 1과 비교한 결과다.
- Tensor 파이프: 62.2% → **73.9%**
- barrier stall: 0.61 → **0.08**
- wait: 2.68 → 2.18, mio: 0.77 → 0.66, short_sb: 0.41 → 0.27
- 실행 명령: 199.6M (거의 같음)

**판정: 채택.** 모든 크기에서 개선됐다. 기본 구성은 M·N ≥ 4096²이고 M이 256의 배수이면 cfg 1, 그 외에는 cfg 2로 정했다.

## 9. 가설 E: L2 재사용 (block rasterization) → 8192에서만 채택

8192에서 128×256(100.7)과 256×128(114.5)이 크게 달랐다. 같은 계산량에서 grid 모양만 다르므로 L2 재사용 차이로 봤다. 그래서 연속 실행되는 block이 GROUP개 행 블록 안에서 열 방향으로 지그재그로 돌게 했다(`TC7_GROUP`).

| 구성 | group 0 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| cfg 0, 8192 | 100.6 | 113.8 | 115.5 | 115.7 |
| cfg 1, 8192 | 115.4 | **118.0** | **118.0** | 115.4 |
| cfg 1, 4096 | 84.7 | 84.7 | 84.5 | 84.4 |
| cfg 2, 8192 | 95.5 | — | 105.1 | — |

**판정: 채택(기본 group 8).** 8192에서 +2%이고, 4096 이하에서는 차이가 없다.

## 10. 최종 측정 (2회 실행)

| kernel | 1024 | 2048 | 4096 | 8192 | cuBLAS TF32 대비 (8192) |
|---|---:|---:|---:|---:|---:|
| 5 swizzle | 27.7 / 27.7 | 35.0 / 35.0 | 59.3 / 60.9 | 82.1 / 82.0 | 61% |
| 6 vectorized frag | 36.9 / 36.8 | 51.5 / 51.5 | 71.0 / 71.1 | 92.9 / 92.9 | 69% |
| 7 frag double buffer | 40.9 / 41.1 | 58.3 / 58.3 | 84.6 / 84.5 | 117.9 / 117.9 | **88%** |
| 100 cuBLAS TF32 | 50.0 / 50.1 | 74.3 / 74.3 | 104.0 / 104.0 | 134.6 / 133.3 | 100% |

kernel 0~7의 모든 크기에서 정확도 검증이 PASS했다(112/112). 최대 상대 오차는 4.3e-4다.

kernel 6의 기본 구성은 8192에서 cfg 0(92.9)을 쓴다. cfg 1(99.4)이 더 빠르지만, 코드와 측정값을 일치시키려고 그대로 두었다.

원시 로그는 [a100_2026-10-15-opt.log](a100_2026-10-15-opt.log)에 있다.

## 남은 차이와 다음 후보

- **`wait` stall:** cuBLAS(0.99)의 2배 이상이고, PC sampling상 대부분 HMMA 자체에 걸린다. HMMA 순서(CUTLASS는 serpentine 순서)나 operand reuse 패턴 차이로 추정하지만 확인하지 않았다. 다음 실험 후보는 m·n 순회 순서를 바꾸는 것이다.
- **epilogue:** C 로드 동안 Tensor Core가 쉰다(FMUL long_sb 약 5%). C를 메인 루프 마지막 타일 즈음에 미리 읽는 방법이 있다.
- **작은 크기(1024~2048):** 78~82%에 머문다. block 수가 SM 수에 비해 적다. split-K 또는 stream-K가 후보다.
- **truncate 반올림이 느려진 이유:** 5절의 원인은 확인하지 않았다.

## 교훈

- ncu의 "latency bound" 판정만으로는 무엇을 고칠지 알 수 없었다. **같은 타일 구성의 cuBLAS 커널을 프로파일해 비교**하자 타일 크기가 원인이 아니라는 것이 분명해졌다.
- 명령 수를 줄여도(3절) 속도가 그대로일 수 있다. 다음 단계는 SASS 루프 본문을 직접 세어 **어떤 명령이 남았는지** 보는 것이었다. 레지스터가 꽉 찬 커널에서는 컴파일러가 주소를 다시 계산하므로, 주소를 "기준값 + 상수"로 설계해야 한다.
- 가설이 틀린 실험(3절, 5절)도 원인을 좁히는 데 쓰였다. 5절에서 반올림을 뺐을 때 느려진 결과는 "명령 수 = 병목"이라는 가정을 다시 버리게 했다.
- 레지스터를 많이 쓰는 큰 타일 커널(SM당 8 warp)에서는 다른 warp가 지연을 가려 주지 못한다. **한 warp 안에서 load와 mma를 겹치는 소프트웨어 파이프라이닝**이 가장 큰 개선(8192에서 cuBLAS 대비 74% → 85%)을 냈다.
- 새 인덱싱을 numpy로 먼저 시뮬레이션해서, GPU에서는 정확도 실패가 한 번도 없었다.

## 운영 메모

- `colab exec`가 실행 전에 "Connection was lost"로 한 번 실패했다. 세션은 살아 있었고, 다시 실행하자 정상 동작했다.
- cuBLAS 커널은 `ncu -k regex:.* -c 3`으로 받았다. 검증 단계의 cuBLAS FP32(`ampere_sgemm_*`) 호출도 함께 잡힌다.
- 세션은 측정이 끝날 때마다 `colab stop`으로 종료했다.

## 2026-10-16: cuBLAS 수준 활용률을 위한 추가 탐색 (진행 중)

A100-SXM4-40GB Colab 세션 `tcgemm-opt`에서 CUDA 13.0으로 빌드했다.
`TC7_ORDER=0/1/2`는 기존 행 우선 / 행 serpentine / 열 serpentine HMMA 순서다.
각 accumulator의 K 누적 순서는 유지한다. `TC7_CFG=4/5`는 각각
64×128 / 128×64 block tile과 32×32 warp tile이며, 3-stage SMEM은 72 KiB다.
작은 타일은 block 수와 SM당 상주 block 수를 늘리는 실험이다.

`tune.py`는 환경의 TC7 설정을 제거한 뒤 cuBLAS TF32, 기존 기본값,
6개 구성 × 3개 순서를 무작위 순서로 실행한다. 매 실행에서 FP32 기준 정확도
PASS 및 정상 종료를 요구하며, 크기별 raw log와 반복별 측정값을 JSON에 남긴다.
탐색 결과만으로 기본값을 바꾸지 않고 별도 반복 측정으로 확인한다.

### 실행 핸들과 현재 상태

- 설치된 CLI: `/home/jake/.local/bin/colab` (uv tool 환경). 프로젝트 `.venv/bin`에는 없다.
- `colab sessions`는 인증 성공, 기존 활성 세션 없음. `colab new -s tcgemm-opt --gpu A100` 성공.
- A100-SXM4-40GB, driver 580.82.07, CUDA 13.0에서 첫 버전 빌드 성공.
- 첫 탐색은 **계속 실행 중**: 로컬 unified exec session **34653**.
  `colab exec -s tcgemm-opt --timeout 1800 < /tmp/tcgemm-tune-run.py`.
  원격 `/content/tensor_core_gemm/tune.py --rounds 2 --output /content/tcgemm-tuning`.
  첫 버전은 cfg 0~5, order 0~2만 포함한다. `summary.json`은 크기가 끝날 때 갱신된다.
  `colab ls`에서 8192 첫 round의 로그가 증가하는 것을 확인했다. 관측이 조용하다고 재시작하지 않는다.
- 완료된 중간 결과(두 측정의 중앙값, TFLOPS):
  512 default 10.921 / cfg5-order1 16.353 / cuBLAS 23.286;
  1024 default 49.196 / cfg2-order0 52.707 / cuBLAS 63.773;
  2048 default 73.025 / cfg4-order0 84.830 / cuBLAS 84.422;
  4096 default 84.534 / cfg0-order2 88.339 / cuBLAS 104.042.
  8192는 아직 완료되지 않았다. 이 값은 최종 검증 결과가 아니다.
- 로컬의 후속 변경은 아직 업로드·빌드하지 않았다: cfg6 (64×64),
  `TC7_COPY=1` (global cp.async를 k8 step 4개에 분산하고 마지막 step에서 commit),
  cuBLAS 호출 오류 검사, tune.py의 후보 선택 옵션.
- 다음: 실행 핸들 재확인 → 전체 원시 로그/summary 보관 → 후속 소스 업로드 후 빌드 →
  유망 후보 반복 확인 및 ncu Tensor 파이프 사용률 비교 → 검증된 기본값 반영.
  현재 세션은 후속 측정에 사용하므로 종료하지 않았다.


### 후속 네이티브 튜너 (대기열 등록)

- 첫 탐색 핸들 34653을 다시 poll하여 실행 중임을 확인했다. 8192 첫 측정
  cfg1-order2는 120.563 TFLOPS, rel 3.504e-4 PASS였다.
- 새 `tune.cu`는 입력/FP32 reference/할당을 크기당 한 번 준비하고 모든 조합을
  같은 입력에서 비교한다. 측정 전 C를 재설정하고, 매 조합/round마다 정확도를 검증한다.
  기존 `main.cu`의 측정 경로는 보존했다.
- 설정 API `configure_frag_pipe`로 한 프로세스에서 조합을 바꿀 수 있게 했다.
  SMEM opt-in도 order/copy별 커널에 한 번씩 설정한다.
- v2 archive `/content/tcgemm-v2.tgz` 업로드 완료.
  후속 `colab exec`의 로컬 핸들은 **28078**이며, 첫 측정과 같은 Jupyter kernel의
  대기열에 있다. `/tmp/tcgemm-native-run.py`는 별도 디렉터리
  `/content/tcgemm-v2/tensor_core_gemm`에서 빌드하고 max8192/min128/3round 전체탐색을 한다.
  결과 `/content/tcgemm-native.jsonl`, source SHA manifest, build log, 최초탐색로그를
  `/content/tcgemm-opt-results.tgz`에 보존한다. 아직 실행 결과를 확인하지 않았다.
- native 실행 뒤 기존/최상 후보/cuBLAS의 ncu 비교를 대기열에 추가했다.
  `/tmp/tcgemm-profile.py`, 결과 `/content/tcgemm-profiles.tgz`.


## 2026-10-16 최신 확인 상태 (앞선 실행 중/대기열 메모를 대체)

초기 탐색, v2 native 전체 탐색, 1410 MHz 설정 탐색, 1140 MHz 후보 확인은 모두 종료했다.
대기열에 미리 등록한 native 실행은 Jupyter 접속 대기에서 실패했고, 원격 ps로 프로세스가
없는 것을 확인한 뒤 다시 실행했다. 단순 관측 시간 초과만으로 재시작하지 않았다.

- native 전체: 909/909 정확도 PASS. 1410 MHz 설정 탐색도 909/909 PASS.
- 1410 MHz 설정에서는 전력 제한으로 cuBLAS의 실제 SM 클럭이 1275~1365 MHz로 내려갔다.
  이 조건의 약 94% 성능 비율을 동등 클럭의 결과로 사용하지 않는다.
- 1140 MHz 후보 확인: 435/435 PASS, baseline/default/cuBLAS의 105개 SM 클럭 기록 모두
  1140 MHz, 메모리 1215 MHz였다. 5회 중앙값은 `a100_2026-10-16-1140-summary.json`.
- v3는 작은 타일과 분산 cp.async를 크기별 기본값으로 반영했다. cfg6/4/1을
  각각 정사각 128~1024 / 2048 / 4096~8192에 선택한다. copy는 128~512와
  4096~8192에서 분산, 1024~2048에서는 burst다. order는 0이다.
- v3 source를 A100에서 다시 빌드했다. 네이티브 검증 105/105 PASS, 기존 CLI
  `tc_gemm 7` 및 `tc_gemm 100` 각각 7/7 PASS. 빌드한 CUDA/헤더/Makefile의
  SHA-256이 현재 소스와 일치함을 로컬에서 확인했다.
- v3 1140 MHz 결과: `a100_2026-10-16-default-summary.json`.
  4096 기존구성 87.917 → 자동설정 95.885 TFLOPS, cuBLAS 108.227 (88.6%).
  8192 기존구성 96.243 → 자동설정 105.368 TFLOPS, cuBLAS 117.328 (89.8%).
  아직 cuBLAS 수준의 활용률에 도달했다고 판정하지 않는다.

### Nsight 직접 비교

최초 `regex:cutlass` 필터는 cuBLAS의 base kernel 이름이 `Kernel2`여서 놓쳤다.
반환코드 0이어도 `No kernels were profiled`였으므로 해당 파일은 비교 증거에서 제외한다.
이후 모든 커널 중 첫 2개(검증 FP32 + 대상 TF32)를 잡아 실제 TF32 지표를 확인했다.

| 크기 | 기존 구성 Tensor elapsed 사용률 | 분산 복사 사용률 | cuBLAS TF32 사용률 |
|---|---:|---:|---:|
| 4096 | 70.05% | 76.47% | 86.07% |
| 8192 | 76.59% | 83.90% | 93.38% |

지표는 모두 `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`다.
사용률 지표와 stall은 ncu 보고서에서 확인한 값이며, throughput 비율을 사용률로
대체하지 않았다. 기존/custom ncu 기본 클럭은 약 1095 MHz였고, cuBLAS 재측정은
1140 MHz 고정 + `--clock-control none`이다. 클럭 조건은 다르지만 사용률 개선과
남은 차이의 방향은 1140 MHz의 동등 조건 throughput 측정과 일치한다.
8192에서 wait/issue는 기존 2.17 → 분산 1.75이며 cuBLAS는 0.99다.
분산 복사로 MIO/short scoreboard는 줄었지만, fragment LDS/반올림과 MMA의 배치가
다음 분석 대상이다. SASS 확인 스크립트는 results 디렉터리에 보존했다.

### 결과 보관 위치

- 로컬 `tensor_core_gemm/results/2026-10-16/`: 초기 raw log, native JSONL,
  실제 클럭 로그, 소스 manifest/archive, ncu CSV, 검증 스크립트.
- v3 `validation/`: build.log, final.jsonl, clocks.log, main-7.log, main-100.log,
  4096/8192-cublas-all.csv, source-sha256.json.
- Colab `tcgemm-opt`는 활성이다. 원격 작업 디렉터리는
  `/content/tcgemm-v3/tensor_core_gemm`, 완료 아카이브 `/content/tcgemm-v3-validation.tgz`.
- 마지막 SASS 추출 실행의 로컬 handle은 21931. 원격 결과는
  `/content/tcgemm-v3-best.sass`. 완료 여부를 먼저 확인한다.


## v4: 단계별 global 포인터 유지 실험 → 큰 타일에서 채택

SASS는 이미 LDS/IADD와 HMMA를 interleave하고 있었다. "fragment 처리가 전부
MMA 앞에 몰려 있다"는 가설은 기각했다. v3 loop는 HMMA 128개에 약 393개 명령,
IMAD 53개였다. 다음으로 복사 step마다 global 포인터를 다음 행 묶음으로 이동시키고
마지막 step에서 K tile 시작 주소로 복귀하는 `TC7_COPY=2`를 실험했다.

- 모든 7개 tile × 3개 MMA 순서에서 새 경로를 검증: 128의 불가능한 큰 tile 제외,
  baseline 포함 162/162 PASS.
- 1140 MHz, 같은 입력·무작위 순서·5회 중앙값:
  4096 copy1 95.339 → copy2 97.366 TFLOPS (cuBLAS 108.231),
  8192 copy1 104.802 → copy2 107.145 TFLOPS (cuBLAS 117.373).
- ncu도 동일 1140 MHz: 8192 Tensor elapsed 사용률 83.28% → 85.21%.
  두 경로 모두 252 registers, spill 0이었다.
- **가설 수정:** dynamic 명령 수는 두 경로 모두 1,722,351,616으로 같았다.
  mainloop도 393개 명령으로 같았고 IMAD 하나가 ULDC 하나로 바뀐 수준이었다.
  따라서 이득을 명령 수 감소로 설명하지 않는다. 포인터 데이터흐름/스케줄링의
  변화로 약 2% 개선된 것으로 보며, 정확한 stall 원인은 아직 확정하지 않았다.
- 큰 자동 타일(256×128)의 copy2를 선택하도록 host 기본값을 바꿨다.
  다른 크기의 기존 최적 선택은 유지한다.
- 결과/코드 manifest/SASS는 `results/2026-10-16/v4/`, remote archive는
  `/content/tcgemm-v4-results.tgz`에 보존했다.

## v5: C epilogue L2 prefetch 실험 (실행 중)

PTX 공식 문서를 Context7 `/websites/nvidia_cuda`에서 확인했다.
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
`prefetch.global.L2 [address]`는 주소를 포함한 cache line을 L2에 가져온다.
결과 값을 레지스터에 보관하지 않고, epilogue는 기존처럼 C를 정상 load한다.

- 큰 tile에만 별도 template 경로를 만들었다. 기본 prefetch는 아직 off다.
- C tile의 32-byte 간격 주소에 hint를 내며, K tile 종료 4/16/32개 앞에서
  한 번씩 실행하는 세 설정을 비교한다. 작은 K에서는 첫 tile로 clamp한다.
- 명령은 `/content/tcgemm-v5`에서 실행 중. 로컬 unified handle **42648**.
  보존된 실행 스크립트 `results/2026-10-16/test_v5.py`.
- 먼저 전체 크기 정확도(새 hint + baseline), 다음 4096/8192에서 5회 비교한다.
  이득이 있으면 ncu로 baseline과 최고 hint를 비교하고 결과를
  `/content/tcgemm-v5-results.tgz`에 보존한다.
- 소스는 현재 v5 실험 상태다. 결과를 확인한 뒤 이득이 없으면 prefetch 코드를
  제거하고 v4의 검증된 기본 경로를 남긴다. 아직 목표 완료로 판정하지 않는다.


## v5 prefetch 기각 및 cuBLAS SASS 확인

C prefetch는 정확도 39/39 및 성능 단계의 모든 비교에서 PASS했으나 느렸다.
1140 MHz 5회 중앙값: 8192 baseline copy2 107.101 TFLOPS, prefetch4 105.782,
prefetch16 104.844, prefetch32 104.379. 4096에서도 모두 느렸다.
252 registers, spill 0이므로 spill 때문으로 설명하지 않는다. prefetch 경로/knob은
현재 코드에서 제거했다. v5 원시 로그와 manifest는 `results/2026-10-16/v5/`에 있다.

`compare_sass.py`로 ncu SourceCounters report를 만들고 `--page source --print-source sass`
CSV를 내보냈다. base 이름 대신 demangled `regex:.*s1688gemm.*`를 사용해 TF32 커널을
정확히 선택했다. 공식 CLI 문서는 Context7에서 확인했다.
- `/content/tcgemm-sass-comparison.tgz` 회수 완료.
- 로컬 `results/2026-10-16/sass-comparison/`: custom/cublas ncu-rep, SASS CSV,
  PC sampling, profile log.
- cuBLAS는 A fragment를 `LDSM.16.M88.4`로 읽는다. 이 결과 R152가 이후 HMMA의
  A operand인 것도 확인했다. 현재 custom은 A에 LDS.64 2개를 쓴다.
- cuBLAS의 HMMA는 행 serpentine 순서이며, 같은 A로 첫 행 B를 증가 순서로 쓰고
  다음 A에서 B를 역순으로 쓴다. 이전의 단순 순서 실험은 fragment layout도 달랐다.

## v6: LDMATRIX A + 자연 K 배치 실험 (실행 중)

공식 PTX `ldmatrix` 문서를 Context7 `/websites/nvidia_cuda`에서 확인했다.
A의 16×8 FP32 bit pattern을 8×8 b16 matrix 4개로 읽으면 반환 word가
TF32 a0=(g,t), a1=(g+8,t), a2=(g,t+4), a3=(g+8,t+4)에 맞는다.
이를 위해 다음을 함께 바꿨다. 반올림/누적 정밀도와 tolerance는 그대로다.

- A SMEM chunk XOR: row&7, 각 8행의 16-byte row chunk를 다른 bank로 배치.
- LDMATRIX 주소: rows lane&15, columns 8*step + 4*(lane>>4).
- B도 K를 t / t+4로 읽고, SMEM swizzle 기준을 row&3으로 바꿔 같은 K 순열을 유지.
- `TC7_ORDER=3`: LDMATRIX + row-major MMA, `4`: LDMATRIX + row serpentine.
  기본 order는 아직 0이며, 새 경로는 실험 옵션이다.
- 복사 0/1/2, tile 0~6, 새 순서 3/4 전체의 정확도를 먼저 검사한다.
  다음 4096/8192에서 cfg0/1 × order3/4 × copy1/2를 비교하고, 좋아지면
  최고 후보를 5회 재확인 및 ncu 비교한다.
- Colab 실행: local unified handle **88070**, 세션 `tcgemm-opt`.
  원격 `/content/tcgemm-v6`, 결과 `/content/tcgemm-v6-results.tgz`.
  재현 스크립트 `results/2026-10-16/test_v6.py`.
- 현재 소스는 v6 실험 상태다. 결과를 확인한 뒤 채택/제거한다. 목표는 계속 active.


## v6 결과와 v7 epilogue 파이프라인 (현재 최신 상태)

v6는 완료됐다. 새 LDMATRIX 두 순서 × 복사 3종 × 모든 tile/크기의
정확도는 baseline 포함 303/303 PASS. 독립 5회 재측정에서:
- 4096: 기존 최고 cfg1-order0-copy2 97.360 → cfg0-order3-copy1 99.693 TFLOPS;
  cuBLAS 108.237.
- 8192: 107.254 → 110.050 TFLOPS; cuBLAS 117.463 (약 93.7%).
- ncu 동일 1140 MHz: Tensor elapsed 85.22% → 87.42%.
  LDMATRIX 후보 registers 254, long scoreboard/issue 0.22.
- 결과는 로컬 `results/2026-10-16/v6/`, 원격 `/content/tcgemm-v6-results.tgz`.

PC sampling과 cuBLAS SASS에서 다음 epilogue 차이를 확인했다:
- custom은 C 한 float4를 LDG한 후 곧바로 FMUL/FFMA/STG하고 반복한다.
- cuBLAS는 `LDG.E.LTC128B.128` 4개를 먼저 요청하고 barrier/STS 후 계산한다.
- long scoreboard sample은 custom 25,445 vs cuBLAS 5,681이고 대부분 FMUL에 걸렸다.
  cache hint만 추가한 v5는 실패했으므로 실제 C 값을 여러 개 레지스터에 먼저 읽는
  메모리 병렬성을 실험한다.

v7는 `TC7_EPI=4/8` (order3에만)을 추가했다. 최초 batch를 mainloop 직후,
SMEM epilogue의 두 barrier/STS 전에 읽고, 다음 batch의 global C 읽기와 현재 batch의
shared load/FMA/store를 겹친다. mainloop를 가로질러 C 값이 hoist되지 않게 compiler
memory barrier를 두었다. CUDA 동기화/수식/오차 기준은 바꾸지 않았다.
기본 epi는 아직 1이며, 최고 LDMATRIX 옵션도 아직 자동 기본값에 넣지 않았다.

- A100 실행 local unified handle **25866** (Colab `tcgemm-opt`).
- 원격 `/content/tcgemm-v7/tensor_core_gemm`; 결과 `/content/tcgemm-v7-results.tgz`.
- 재현 스크립트 `results/2026-10-16/test_v7.py`.
- 먼저 모든 tile/copy의 EPI4/8 정확도, 다음 큰 두 tile의 성능, 좋아지면
  최고 후보 5회 재검증과 ncu 비교를 수행한다. 실행 중임을 handle poll로 확인했다.
- `/tmp` 파일은 turn 전환 때 사라질 수 있다. 보존된 스크립트와 원격 archive를 쓴다.
- 목표는 여전히 active. cuBLAS의 8192 Tensor elapsed 93.38%와 아직 차이가 있다.


## v7 結果 / v8 latest runtime state

v7 finished: 303/303 broad correctness PASS. No spills in epi4/8.
- 4096: cfg0/order3/copy1/epi1 99.705, cfg1/order3/copy2/epi8 101.439 TFLOPS;
  cuBLAS 108.220 (3-round screen).
- 8192: cfg0/order3/copy1/epi1 110.019, cfg1/order3/copy2/epi8 109.982;
  cfg0/order3/copy1/epi4 only 107.265. cuBLAS 117.410.
- Hence epilogue batch loading helps some layouts/sizes, but does not universally improve.
  The script's 8192 gate skipped confirmation; 4096/small promising candidates still need 5-round confirmation.
- Records in `results/2026-10-16/v7/`, remote `/content/tcgemm-v7-results.tgz`.

v8 is currently executing, local unified handle **86735**, Colab `tcgemm-opt`.
It bounds C buffers using a non-unrolled outer loop processing two batches at a time,
with fixed buffer indices 0/1. This avoids retaining C values for all unrolled batches.
The default large square path now selects verified cfg0/order3/copy1/epi1 automatically.
`FragPipeConfig.order=-1` means automatic; explicit legacy/config candidates remain fixed.
Source/archive `/content/tcgemm-v8.tgz`; run dir `/content/tcgemm-v8/tensor_core_gemm`;
results `/content/tcgemm-v8-results.tgz`; script `results/2026-10-16/test_v8.py`.
Build passed. Native default correctness has passed through 8192; broad candidates still running
at the latest handle poll. Default 4096 99.621 / 8192 109.907 TFLOPS were printed.
Do not restart while a tool/process is live or merely because observation timed out.
Current worktree is this v8 experiment; automatic epi remains 1 pending results.

### New evidence for a possible next layout experiment (not implemented)

cuBLAS SASS differs further: B uses scalar LDS, and the epilogue uses STS.64,
whereas custom permutes N to vectorize B LDS.128 and writes STS.128.
SourceCounters show custom STS.128 wavefronts 131072 vs ideal 65536 (2x),
cuBLAS STS.64 wavefronts 32768 == ideal 32768. These are actual bank conflict measurements.
The PC samples also show custom C-load FMUL long-scoreboard samples 25445 vs cuBLAS 5681.
Do not treat instruction count alone as a bottleneck; cuBLAS executes more instructions overall.

If v8 does not close the gap, a coherent experiment is natural N + scalar B (matching cuBLAS):
- Keep LDMATRIX A/natural K. B fragment logical columns become warpCol*WN + 8*n + g,
  K rows 8*step + t + 4*r2. B SMEM chunk XOR becomes (row&3)<<1, including NFRAG8.
  This makes scalar B LDS bank-free: column low 3 bits=g, bank high 2 bits XOR t.
- Precompute B fragment addresses outside the K loop; each scalar value gets the same +0x1000 rounding.
- C accumulator maps to column warpCol*WN + 8*n + 2*t. Store each pair e0/e1 as float2,
  use LDC=BN+8, then existing coalesced epilogue reads the same natural row-major C.
  STS.64 half-warp banks become 8*g+2*t, covering all 32 banks without conflicts.
- Possible new TC7_ORDER=5 (natural N, row-major) / 6 (natural N, row serpentine).
  Existing 0..4 retain old layout. No such paths are currently implemented.
- If adding order 5/6, update mask offsets to 21/42 for epi4/8 (7 orders * 3 copies),
  using uint64_t to avoid bit collisions (max bit 62). Current 15/30 only covers 5 orders.
- Validate all new copy/layout/epi combinations at 128/256/512 first; then screen
  cfg0/1, new orders, copies1/2, epis1/4/8 at 4096/8192 once; confirm winners 5 rounds.

The goal remains active: strongest verified Tensor elapsed is 87.42%, cuBLAS 93.38%.
Latest committed-style summary artifact: `a100_2026-10-16-ldmatrix-summary.json` (v6).
All `/tmp` helper files may disappear across turns; project results scripts persist.


## v8 recent poll and next-step evidence

Handle 86735 is still live; v8 broad correctness finished **303/303 PASS**.
Three-round screening (1140 MHz) is complete:
- 4096: default/cfg0-order3-copy1 epi1 99.67 TFLOPS;
  cfg0-order3-copy1-epi4 **104.13**; cuBLAS 108.23.
- 8192: default 109.96, cfg0-order3-copy1-epi1 109.98;
  cfg0-order3-copy1-epi4 **112.95**; cuBLAS 117.43.
- The script is now doing 5-round confirmation, then ncu for baseline/epi4.
  Do not mark these screening values as final. Poll 86735 first or inspect the same remote job.

Further confirmed cuBLAS layout evidence:
- cuBLAS B loads are scalar `LDS` (default 32-bit), not LDS.128; its A is LDSM.16.M88.4.
- cuBLAS C stores are STS.64 with zero excessive wavefronts; custom STS.128 has 2x ideal wavefronts.
- A coherent follow-up is *natural N*, not just another traversal order:
  B source col=warpCol*WN+8*n+g, K=t+4*r2, with chunk XOR `(row&3)<<1`.
  C store col=warpCol*WN+8*n+2*t, float2 store for e0/e1, **LDC=BN+8**.
  This gives bank-free STS.64 (half warp bank index 8*g+2*t), and existing float4 C read stays coalesced.
  Current custom N permutation (g*NFRAG+n) is why vector C stores have conflicts.
- A possible new order 5/6 = natural N row-major/row-serpentine, with LDMATRIX A retained.
  TRAVERSAL must map 5→0 and 6→1, not simply order-3. Current code has only orders 0..4.
- For 7 orders × 3 copies, epi mask offsets must change from 15/30 to **21/42**,
  max uint64 bit 62. EPI4/8 would need to accept orders 3/5/6.
- Efficient validation for a new N layout: all copies/epis/tiles at max512 min128 first
  (cheap), then screen large candidates once, confirm winners 5 rounds. No new layout implemented yet.

Before finalizing all defaults, confirm small winners too (v7/v8 one-round exploration):
- <=1024: cfg6/order3/copy0/epi4; 2048 cfg4/order3/copy0/epi4.
  Small stages show strong gains but have not yet received independent 5-round confirmation.
- Current auto small settings are still older cfg6/4 order0. Large auto is cfg0/order3/copy1,
  epi1 until the v8 confirmation completes. Current worktree is v8.
- Goal still active. Tensor utilization 87.42% was verified for v6; do not infer new v8
  utilization from GFLOPS before reading its ncu CSV. cuBLAS target is 93.38% at 8192.


## v8 completed (latest authoritative state)

Handle 86735 returned exit 0; no benchmark is currently running.
Remote `/content/tcgemm-v8-results.tgz` was downloaded/extracted to local `results/2026-10-16/v8/`.
- Broad correctness: 303/303 PASS; confirmation: 50/50 PASS.
- 5-round median: 4096 baseline99.702 → epi4 **104.165**, cuBLAS108.249 TFLOPS.
- 8192 baseline110.017 → epi4 **112.978**, cuBLAS117.409 TFLOPS (~96.2%).
- ncu same 1140 MHz: Tensor elapsed baseline87.39% → epi4 **89.99%**;
  long scoreboard/issue0.22→0.06, registers254→253. cuBLAS target93.38% remains higher.
- Tracked-style summary file `a100_2026-10-16-epilogue-summary.json`; raw samples/counters/build
  in local v8 results. Clock settings were restored by the script.
- After collecting evidence, local runner auto policy was changed to epi4 for verified
  cfg0/order3/copy1 at area>=4096², with other settings epi1. This one host policy edit
  happened *after* the measured v8 archive. Rebuild/test the automatic path before final completion.
  GPU kernel code is the same tested epi4 specialization.
- Current source is v8 + this host policy adoption; small auto settings remain old order0.
  Small LDMATRIX/epi4 candidates still need 5-round confirmation before adopting them.
- Colab tcgemm-opt remains active for subsequent work, but no live measurement job remains.
  Next safe work is the natural-N/scalar-B/bank-free STS.64 experiment documented above,
  or small-case confirmation; do not restart finished v8 as if it were lost.
- Goal is active, not complete: 89.99% vs cuBLAS93.38% Tensor elapsed is a real remaining gap.

## v9–v12 and A100 recovery (supersedes earlier live-job notes)

- v9 natural-N/scalar-B/STS.64 layouts passed 351 small checks but did not beat v8.
- v10 direct TF32 conversion and v11 guarded bias conversion were slower. v11 passed
  522 small checks; guarded candidates reached only about 54–82 TFLOPS at large sizes.
- Post-v11 integration checks passed 7,772 rectangular/short-K/alpha-beta cases.
  Five-round small-matrix winners were cfg6/order3/copy0/epi4 through 1024 and
  cfg4/order3/copy0/epi4 at 2048: 0.597, 3.804, 20.503, 49.183, 74.132 TFLOPS.
  These small settings have not yet been adopted by the automatic policy.
- Raster groups 0/2/4/8/16/32/64 gave no meaningful large-matrix improvement.
- v12 deferred conversion and copies scheduled after MMA passed 522 small checks.
  Screening found no winner over the existing cfg0/order3/copy1/epi4:
  4096 baseline104.135 vs cuBLAS108.223; 8192 baseline112.863 vs cuBLAS117.325 TFLOPS.
  Handle 90599 finished exit0; archive was downloaded and extracted under
  results/2026-10-16/{v12,source-v12.tgz,v12-results.tgz}.
- The subsequent strict-check launch reported the old tcgemm-opt session lost
  (404/401), and `colab sessions` confirmed it absent. A fresh A100 session with
  the same name was created successfully. qwen35-val was left untouched.
- Current live work: handle **72783**, rebuilding v12 and running **check_v12.py**
  on the new A100. Re-poll this handle before launching any new Colab execution.
  Checker independently quantizes A/B with CUDA TF32 conversion and compares
  custom results with FP32 GEMM on those inputs at 1e-5, in addition to 1e-3
  versus original FP32 inputs and exact alpha0/beta1 identity. The stronger test
  has NOT passed yet; do not confuse it with the older 7,772-case pass.
- Prepared next diagnostic: results/2026-10-16/warp_probe.py. It generates an
  isolated small build from source-v12, tests 16-warp geometries (WM64/WN32 and
  WM32/WN64 for both 128x256 and 256x128) and epilogue batch2/4/8. Production
  source remains unchanged. Run only after current checker is terminal.
- Goal remains active. Best measured Tensor elapsed is still89.99%, cuBLAS93.38%.
- Host automatic policy now adopts the confirmed small configurations listed above;
  README describes them. This edit happened after source-v12 was archived, and the
  currently running strict checker still tests the archived v12 automatic policy.
  Explicit small specializations were already measured/checked. Fresh automatic-path
  build verification of the host policy change remains required.

## Strict v12 check completed; warp probe launched

- Handle72783 completed exit0. **All13,108 checks PASS**, including the independent
  TF32-quantized input reference at1e-5 and original FP32 reference at1e-3.
  Archive `results/2026-10-16/v12-strict.tgz` is local and extracted.
- An interrupted attempt to launch the warp probe did not start execution;
  `colab status` confirmed IDLE with the prior check as last execution.
- A subsequent explicit launch is now active: unified exec handle **10008**,
  `warp_probe.py`, session tcgemm-opt. Poll this handle before any new execution.
  It first archives prior compiled binaries at `/content/tcgemm-v12-build.tgz`.
  Download that archive when available to avoid future full recompilation.

## Warp probe completed; direct epilogue probe active

- Handle10008 finished exit0. Source and raw output archived locally as
  `warp-probe-results.tgz` and extracted under `warp-probe/`.
- No 16-warp geometry beat the existing 8-warp baseline. Representative cfg7
  order3/copy1/epi4 required128 registers/thread with60B spill stores /44B loads.
  Merely increasing warp count worsens register pressure.
- Epi2 also lost: 8192 five-round112.862 vs existing epi4 **113.456 TFLOPS**,
  paired cuBLAS **117.968**. 4096 baseline104.493 vs cuBLAS108.566.
  These measurements are from the replacement physical A100; compare paired
  ratios rather than mixing absolute rates with the former runtime.
- Verified v12 compiled binaries now preserved in local `v12-build.tgz`.
- Next experiment actually launched: handle **31681**, `direct_probe.py`.
  Tests natural-N float2 output directly from accumulator registers, avoiding
  shared-memory epilogue transpose/barriers. It retains alpha/beta and input math.
  Production kernel is unchanged. Re-poll this handle before new execution.

## Direct epilogue result; public CUTLASS diagnostic

- Handle31681 completed exit0; direct-probe-results.tgz downloaded/extracted.
  Direct natural-N float2 output passed accuracy but lost to the shared epilogue:
  five-round8192 best111.616 vs custom baseline113.450 and cuBLAS117.974 TFLOPS.
  4096 best102.823 vs baseline104.503 and cuBLAS108.567. No production change adopted.
- Current execution handle **75585**, `cutlass_probe.py`, on tcgemm-opt.
  It sparsely clones official NVIDIA CUTLASS headers, records exact commit, builds
  eight float-input TF32 `OpMultiplyAdd` (not 3xTF32 FastF32) configurations,
  and compares them with our existing baseline/cuBLAS at1140 MHz,5 rounds.
  This is an independent diagnostic, not replacing the custom kernel with a library.
- Primary source confirms float/float OpMultiplyAdd uses TF32 internally:
  NVIDIA/cutlass include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h,
  DefaultMmaTensorOp specialization around lines130–169.

## Public CUTLASS evidence and instruction-path experiment

- Handle75585 completed. CUTLASS cfg1 (128x256x32,warp64x64,3 stages,identity8)
  reached108.046/117.438 TFLOPS at4096/8192, paired cuBLAS108.571/117.942.
  Custom default104.518/113.423. All accuracy records PASS. Exact CUTLASS commit,
  sources, binaries and clocks saved under local `cutlass-probe/` and archive.
- Handle4352 profiled custom/CUTLASS/cuBLAS at1140 and completed exit0.
  Local `cutlass-profiles/` includes ncu reports and SASS source counters.
  Tensor elapsed90.10/93.00/93.40%; registers253/228/230 respectively.
  Custom short-scoreboard/issue .03 versus CUTLASS .00, Cu .01;
  custom long-scoreboard/issue .07 versus both .04. Custom executes fewer
  instructions (1.691B vs CUTLASS2.362B/Cu2.225B), so instruction count alone
  is not the optimization target. IADD3 has2,498 custom short-SB PC samples.
- Handle23047 accumulator-storage/visitation probe completed exit0. Neither
  column-major accumulator storage nor new column-serpentine LDMATRIX paths won.
  8192 baseline113.458, same column-storage113.451, cuBLAS117.937 TFLOPS.
  Archive `accumulator-probe-results.tgz` downloaded/extracted. No kernel adopted.
- Current execution: handle **30825**, `mad_probe.py`, tcgemm-opt.
  Tests integer-MAD rounding with runtime multiplier1 (to prevent IADD folding),
  and predicated MAD, versus unchanged integer-add rounding. It saves SASS for
  modes0/4/5 before benchmarking. Input/output/TF32 arithmetic are unchanged.

## MAD completed; B-copy mapping diagnostic active

- Handle30825 completed exit0; MAD instruction paths did not beat baseline.
  Five-round8192 baseline113.440 / default113.456, cuBLAS117.943 TFLOPS.
  Archive mad-probe-results.tgz downloaded/extracted. No production change adopted.
- SASS exposes a concrete copy-layout difference: custom B copies use distinct
  64-bit global row addresses; CUTLASS reuses one base with constant0x80 column
  increments. The warp-raked map spans all32 K rows with threads, then N columns
  with copy iterations, reducing live address state without changing SMEM layout.
- Current live handle **32076**, `rake_probe.py`, tcgemm-opt. Broad small checks
  already passed; large performance screening is running. Scratch kernel adds
  RAKE_B template flag. Scratch config.epilogue8 selects RAKE_B with actual
  epilogue batch4; this diagnostic encoding must NOT be shipped as public API.
  Copy modes1/3, orders3/5/6, rounding0/1/2/3 tested for RAKE_B.
- Prepared strict follow-up `check_rake.py` (run only after32076 terminal).
  Tests only the actually instantiated diagnostic configurations on rectangular,
  K32/64/96/160, alpha/beta and raster-tail cases against both independent refs.
  Updated/uploaded check.cu also reports worst relative errors. Older13,108-case
  pass remains valid; the new maximum-error print is not yet executed.

## B-copy mapping wins repeated benchmark; strict validation active

- Handle32076 finished exit0. All small screening and large correctness records PASS.
  Five-round medians for cfg0/order6/copy3/epi4/round0/RAKE_B:
  4096 **107.3956** vs old baseline104.2999 / default104.5176 / cuBLAS108.5666 TFLOPS;
  8192 **116.5987** vs old baseline113.4190 / default113.4183 / cuBLAS117.9223.
  Thus98.9% of paired cuBLAS throughput at both sizes; utilization not yet measured.
  Archive `rake-probe-results.tgz` downloaded/extracted locally.
- Important: ptxas still reports252 registers and no spills for the natural-N
  RAKE_B candidates. Address reuse is proven by layout/source, but a reduction
  in total allocated registers is NOT proven and must not be claimed.
- Current live handle **89725**, `check_rake.py`, independently checks all supported
  RAKE_B/baseline combinations at1e-5 quantized TF32 /1e-3 FP32. Poll first.
  After it passes, profile cfg0/order6/copy3/RAKE_B at1140 and compare with paired
  custom baseline/CUTLASS/cuBLAS reports. Still no production adoption of RAKE_B.
- User explicitly asked whether accuracy is checked. Answered: every performance
  trial checks FP32 before timing; strengthened full checks gate adoption. Preserve
  that requirement and report measured worst errors once new checker completes.

## New candidate accuracy confirmed

- Handle89725 completed exit0. **All1,200 B-copy diagnostic checks PASS**:
  rectangular tiles, K32/64/96/160, four alpha/beta pairs, full/partial raster groups,
  all actually instantiated RAKE_B configurations and baseline controls.
- Worst FP32-reference normalized error **3.12642872e-4** (limit1e-3).
  Worst independently quantized TF32-reference error **1.10369408e-6** (limit1e-5).
  Exact alpha0/beta1 identity also checked. Archive rake-check.tgz is local/extracted.
- Current live job **95484**, `profile_rake.py`, tcgemm-opt. Profiles baseline,
  cfg0/order6/copy3/RAKE_B/epi4/round0 and cuBLAS, clocks1140, with SASS sampling.
  Poll this handle before launching more execution. Result path when done:
  `/content/tcgemm-rake-profiles.tgz`. Production RAKE_B adoption and actual current
  automatic policy build verification still pending. Goal remains active.

## Production copy4 adoption and v13 verification

- Handle95484 completed; rake-profiles.tgz downloaded/extracted.
  Same1140 MHz Tensor elapsed: baseline90.11%, RAKE_B **92.32%**, cuBLAS93.40%.
  RAKE_B252 registers/no spills; long-SB/issue.07, short-SB/issue.07.
  Target remains unmet by1.08 percentage points.
- Production source now exposes **COPY_MODE4**: B-raked map plus post-MMA copy
  groups. Host restricts it to order6/epi4/round0, and auto selects it for large
  cfg0 tiles. This is the same algorithm as diagnostic copy3+RAKE_B=true;
  the diagnostic epi8 flag encoding was NOT adopted. Tuner/checker includecopy4.
  Small auto settings cfg6/4 order3/copy0/epi4 are also adopted.
- Current-source archive source-v13.tgz uploaded and extracted. Async CPU build
  workerPID15754 completed returncode0 in351.21s. Build-done.json downloaded.
- Epi batch2/8, explicit FMA contraction order, and scaling accumulators before
  SMEM stores all lost to existing epi4. Archive epilogue-rake-results.tgz local.
- C prefetch buffers3/4/8 also lost; largest settings were much slower. Raw
  cbuffer-probe-results.tgz downloaded/extracted. No changes adopted.
- Active unified exec handle **16231**, `validate_v13.py`: actual production
  code passed **13,224** strict checks; max relative FP32 **3.12791562e-4**,
  quantized TF32 **1.10369408e-6**. Default benchmarks128..8192 are running,
  followed by original CLI and ncu of actual default and cuBLAS. Poll this handle.
- Prepared next diagnostic raster_probe.py (NOT started). Official CUTLASS
  GemmIdentityThreadblockSwizzle groups N tiles, whereas current custom raster
  groups M tiles. Test negative diagnostic group values for N grouping and
  positive M groups only after16231 terminal. Production group API unchanged.
- Latest README/summary describe candidate measurements; after actual v13 completes,
  update README validation status and summary to current-source evidence.

## v13 production validation completed; further gap analysis

- Handle16231 finished exit0. v13-results.tgz and v13-build.tgz are local.
  Actual auto default five-round107.401/116.569 TFLOPS at4096/8192 versus
  cuBLAS108.551/117.892; explicit copy4 matches default within noise.
  Original CLI7 at every size128..8192 passes accuracy. Clocks all sampled1140.
  Actual default ncu name confirms specialization<...,6,4,4,0>; Tensor elapsed
  **92.34%** vs paired cuBLAS**93.43%**. All13,224 strict checks pass as above.
- README and a100_2026-10-16-rake-summary.json updated to actual production
  measurements, including all sizes and repeated samples. tune.py acceptscopy4
  and skips unsupported order/copy and128-size tile combinations (Python-only edit).
- Handle56261 raster probe finished exit0. No meaningful N-grouping improvement:
  8192 N-group4 116.632 vs M-group8 116.593; pairedCu117.898 TFLOPS.
  Keep production group8. Archive raster-probe-results.tgz downloaded/extracted.
- Current next execution is `load_round_probe.py` (record returned handle below).
  New modes4/5/6 round the NEXT loaded A+B/A-only/B-only fragment after its
  loads, before returning loadNext. This differs from v12 rounding CURRENT
  fragments just before MMA. Shared wait PC sample evidence: IADD3 R160 immediately
  using LDSM result has4,680 short-SB samples in rake profile, the largest site.
  New modes remain diagnostic, not public production configuration.

## Post-v13 diagnostics and register allocation experiment

- Load/round handle67069 finished exit0. New next-fragment rounding modes4/5/6
  did not win: best116.600 TFLOPS vs default116.596 at8192, cuBLAS117.915.
  Archive load-round-probe-results.tgz local/extracted. No production change.
- A-two-ahead handle42293 finished exit0. Accuracy passed but performance fell
  to80.025/86.184 TFLOPS (4096/8192). It used242 registers with **zero spills**;
  do not attribute this regression to spilling. Archive ahead-probe-results.tgz
  downloaded/extracted. No live GPU job remained afterward; colab status confirmedIDLE.
- Current live execution **16917**, regcap_probe.py, tcgemm-opt.
  It separately compiles the SAME production B-raked math with limits224/228/
  232/240/248/252/255, removing launch_bounds in limited modules. Control0 retains
  original launch_bounds; original production default is also measured in-process.
  Each module has its own kernel name and wrapper; production code unchanged.
- Build evidence: control252 regs/no spills; limit224→224 regs/28B stores+loads;
  228→228/12B; **232→232/no spills**;240→240/no spills;248→247/no spills;
  252→249/no spills;255→252/no spills. Correctness/performance measurement follows.
- Goal still active: measured production92.34% vs cuBLAS93.43%; do not mark complete.

## Paused state and final benchmark summary

- User paused optimization and requested a summary. This entry supersedes the
  active-job/active-goal status above. Colab tcgemm-opt is IDLE; no new experiment
  is authorized by the summary request.
- Register-cap handle16917 completed exit0. Five-round confirmation, TFLOPS:
  4096: cap240 107.864967873, production107.398830457, cuBLAS108.559598956;
  8192: cap240 117.161640172, production116.625553742, cuBLAS117.916355366.
  Cap240 uses240 registers with zero spills and reaches about99.36% of paired
  cuBLAS throughput. Basic FP32 accuracy checks passed at256/512/4096/8192.
  It has NOT received the full independent TF32 validation or an ncu Tensor
  utilization profile and is NOT adopted in production. Raw confirmation is in
  results/2026-10-16/regcap-probe/confirmation.jsonl.
- The finer register sweep regcap_fine.py did not start: its remote result
  directory and archive do not exist. It has no benchmark result.
- Production remains v13 COPY_MODE4/order6/epi4 for large tiles, within kernel7
  rather than a separate numbered kernel. Its validated auto performance is
  107.401/116.569 TFLOPS at4096/8192, paired cuBLAS108.551/117.892.
  All13,224 strict checks passed. Tensor elapsed remains92.34% vs93.43% cuBLAS;
  utilization parity has not been achieved, and the goal remains paused.
