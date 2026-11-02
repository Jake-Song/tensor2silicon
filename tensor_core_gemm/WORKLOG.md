# 작업 기록: TF32 Tensor Core GEMM 최적화 (A100)

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
- `TC7_ORDER=3`: LDMATRIX + 행 우선 MMA, `4`: LDMATRIX + 행 지그재그 MMA.
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


## v7 결과 / v8의 최신 실행 상태

v7이 완료됐다. 전체 정확도 검사 303/303 PASS. epi4/8에서 spill은 없었다.
- 4096: cfg0/order3/copy1/epi1 99.705, cfg1/order3/copy2/epi8 101.439 TFLOPS;
  cuBLAS 108.220 (3회 예비 측정).
- 8192: cfg0/order3/copy1/epi1 110.019, cfg1/order3/copy2/epi8 109.982;
  cfg0/order3/copy1/epi4는 107.265에 그쳤다. cuBLAS 117.410.
- epilogue 일괄 읽기는 일부 배치·크기에는 도움이 되지만 항상 개선되지는 않는다.
  스크립트의 8192 조건을 통과하지 못해 재확인은 건너뛰었다. 4096과 작은 크기의
  유망 후보는 여전히 5회 재확인이 필요하다.
- 기록: `results/2026-10-16/v7/`, 원격 `/content/tcgemm-v7-results.tgz`.

v8이 현재 실행 중이다. 로컬 통합 실행 핸들은 **86735**, Colab 세션은 `tcgemm-opt`다.
외부 루프를 펼치지 않고 한 번에 두 배치를 처리하며 버퍼 인덱스를 0/1로 고정해
C 버퍼 크기를 제한한다. 펼쳐진 모든 배치의 C 값을 동시에 보관하지 않도록 한 것이다.
큰 정방행렬의 기본 경로는 검증된 cfg0/order3/copy1/epi1을 자동 선택한다.
`FragPipeConfig.order=-1`은 자동 선택을 뜻하며, 명시한 기존 구성과 후보 설정은 고정된다.
소스 압축파일 `/content/tcgemm-v8.tgz`, 실행 디렉터리 `/content/tcgemm-v8/tensor_core_gemm`,
결과 `/content/tcgemm-v8-results.tgz`, 스크립트 `results/2026-10-16/test_v8.py`.
빌드는 통과했다. 네이티브 기본 경로는 8192까지 정확도 검사를 통과했으며,
마지막 핸들 조회 시 전체 후보 검사는 계속 실행 중이었다.
기본 경로에서 4096 99.621 / 8192 109.907 TFLOPS가 출력됐다.
도구·프로세스가 실행 중이거나 단순히 관측 시간이 초과됐다는 이유로 재시작하지 않는다.
현재 작업 트리는 v8 실험 상태다. 결과가 나오기 전까지 자동 epi는 1을 유지한다.

### 다음 배치 실험을 위한 새 근거 (아직 구현하지 않음)

cuBLAS SASS에는 추가 차이가 있다. B는 스칼라 LDS, epilogue는 STS.64를 쓴다.
반면 자체 구현은 N 순열로 B의 LDS.128을 벡터화하고 STS.128로 쓴다.
SourceCounters에서 자체 STS.128의 wavefront 수는 이상적인 65536 대비 131072로 2배다.
cuBLAS STS.64는 실제 32768 == 이상적인 32768이다. 실제 측정된 bank conflict 근거다.
PC 표본에서도 C 읽기 뒤 FMUL의 long-scoreboard 표본은 자체 구현 25445, cuBLAS 5681이다.
cuBLAS의 전체 실행 명령은 더 많으므로 명령 수만으로 병목을 판단하지 않는다.

v8이 격차를 충분히 줄이지 못하면 자연 N 배치 + 스칼라 B 읽기를 함께 실험한다.
- LDMATRIX A와 자연 K 배치는 유지한다. B fragment의 논리 열은 warpCol*WN + 8*n + g,
  K행은 8*step + t + 4*r2로 정한다. B SMEM chunk XOR은 NFRAG8에서도 (row&3)<<1이다.
  스칼라 B LDS의 열 하위 3비트는 g, bank 상위 2비트는 t와 XOR되어 충돌을 피한다.
- B fragment 주소를 K 루프 밖에서 미리 계산하고, 각 스칼라에 같은 +0x1000 반올림을 적용한다.
- C 누산기의 열은 warpCol*WN + 8*n + 2*t다. e0/e1 쌍을 float2로 저장하고
  LDC=BN+8을 사용한다. 기존 병합 접근 epilogue는 자연 행 우선 C를 그대로 읽는다.
  STS.64의 half-warp bank는 8*g+2*t가 되어 32개 bank를 충돌 없이 덮는다.
- 새 TC7_ORDER 후보는 5(자연 N, 행 우선) / 6(자연 N, 행 지그재그)다.
  기존 0..4의 배치는 유지한다. 현재 이 경로들은 구현되지 않았다.
- order 5/6을 추가하면 epi4/8의 마스크 오프셋을 21/42로 바꿔야 한다(7개 순서 * 3개 복사).
  uint64_t로 비트 충돌을 피한다(최대 비트 62). 현재 15/30은 순서 5개만 지원한다.
- 새 복사·배치·epi 조합을 128/256/512에서 먼저 모두 검증한다. 이후 4096/8192에서
  cfg0/1, 새 순서, copies1/2, epis1/4/8을 한 번 탐색하고 우승 후보를 5회 재확인한다.

목표는 계속 진행 중이다. 검증된 최고 Tensor elapsed 사용률은 87.42%, cuBLAS는 93.38%다.
최신 요약 산출물은 `a100_2026-10-16-ldmatrix-summary.json`(v6)이다.
`/tmp` 보조 파일은 턴 사이에 사라질 수 있지만 프로젝트 결과 스크립트는 보존된다.


## v8 최근 조회 결과와 다음 단계의 근거

핸들 86735는 계속 실행 중이며, v8 전체 정확도 검사는 **303/303 PASS**로 끝났다.
1140 MHz에서 3회 예비 측정도 완료됐다.
- 4096: 기본 경로/cfg0-order3-copy1 epi1 99.67 TFLOPS;
  cfg0-order3-copy1-epi4 **104.13**; cuBLAS 108.23.
- 8192: 기본 경로 109.96, cfg0-order3-copy1-epi1 109.98;
  cfg0-order3-copy1-epi4 **112.95**; cuBLAS 117.43.
- 스크립트는 현재 5회 재확인 중이며, 이후 기준 경로/epi4의 ncu를 수집한다.
  예비 측정값을 최종값으로 표시하지 않는다. 먼저 86735를 조회하거나 같은 원격 작업을 확인한다.

추가로 확인한 cuBLAS 배치 근거:
- cuBLAS의 B 읽기는 LDS.128이 아닌 스칼라 `LDS`(기본 32비트)이며, A는 LDSM.16.M88.4다.
- cuBLAS C 쓰기는 STS.64이며 추가 wavefront가 없다. 자체 STS.128은 이상적인 수의 2배다.
- 다음 실험은 순회 순서만 바꾸는 것이 아니라 *자연 N 배치*를 함께 적용하는 것이다.
  B 원본 col=warpCol*WN+8*n+g, K=t+4*r2, chunk XOR은 `(row&3)<<1`이다.
  C 쓰기는 col=warpCol*WN+8*n+2*t, e0/e1을 float2로 저장하며 **LDC=BN+8**이다.
  STS.64의 half-warp bank 인덱스 8*g+2*t는 충돌이 없고, 기존 float4 C 읽기도 병합된다.
  현재 자체 N 순열(g*NFRAG+n) 때문에 벡터 C 쓰기에서 충돌이 생긴다.
- 새 order 5/6은 자연 N의 행 우선/행 지그재그 순서이며 LDMATRIX A를 유지한다.
  TRAVERSAL은 단순 order-3이 아니라 5→0, 6→1로 대응해야 한다. 현재 코드는 0..4만 지원한다.
- 순서 7개 × 복사 3개를 지원하려면 epi 마스크 오프셋 15/30을 **21/42**로 바꿔야 한다.
  uint64의 최대 비트는 62다. EPI4/8은 order 3/5/6을 허용해야 한다.
- 새 N 배치는 모든 복사·epi·타일을 max512 min128에서 먼저 저비용으로 검증한다.
  이후 큰 후보를 한 번 탐색하고 우승 후보를 5회 재확인한다. 새 배치는 아직 구현하지 않았다.

기본값을 확정하기 전에 작은 크기의 우승 후보도 재확인한다(v7/v8의 1회 탐색).
- <=1024: cfg6/order3/copy0/epi4; 2048: cfg4/order3/copy0/epi4.
  작은 크기는 개선 폭이 크지만 아직 독립적인 5회 재확인을 거치지 않았다.
- 작은 크기의 현재 자동 설정은 여전히 예전 cfg6/4 order0다. 큰 크기는 cfg0/order3/copy1이며,
  v8 재확인이 끝날 때까지 epi1을 유지한다. 현재 작업 트리는 v8이다.
- 목표는 계속 진행 중이다. 87.42%는 v6에서 검증한 사용률이다. ncu CSV를 읽기 전에는
  GFLOPS만으로 v8의 새 사용률을 추정하지 않는다. 8192의 cuBLAS 목표값은 93.38%다.


## v8 완료 (이전 상태 기록을 대체하는 최신 결과)

핸들 86735가 종료 코드 0을 반환했다. 현재 실행 중인 벤치마크는 없다.
원격 `/content/tcgemm-v8-results.tgz`를 로컬 `results/2026-10-16/v8/`로 다운로드·해제했다.
- 전체 정확도 303/303 PASS; 재확인 50/50 PASS.
- 5회 중앙값: 4096 기준 경로 99.702 → epi4 **104.165**, cuBLAS 108.249 TFLOPS.
- 8192 기준 경로 110.017 → epi4 **112.978**, cuBLAS 117.409 TFLOPS(약 96.2%).
- 동일 1140 MHz의 ncu: Tensor elapsed 사용률 87.39% → epi4 **89.99%**;
  long scoreboard/issue 0.22→0.06, 레지스터 254→253. cuBLAS 목표 93.38%에는 못 미친다.
- 요약 파일은 `a100_2026-10-16-epilogue-summary.json`이며 원시 표본·카운터·빌드는
  로컬 v8 결과에 있다. 스크립트가 클럭 설정을 복원했다.
- 근거 수집 후 로컬 실행기의 자동 정책을 area>=4096²의 검증된 cfg0/order3/copy1에서
  epi4로 바꾸고 나머지는 epi1로 유지했다. 이 호스트 정책 변경은 측정된 v8 압축파일을
  만든 *이후*에 수행했다. 완료 전 자동 경로를 다시 빌드·검사해야 한다.
  GPU 커널 코드는 검사한 epi4 특수화와 같다.
- 현재 소스는 v8에 이 호스트 정책을 반영한 상태다. 작은 크기의 자동 설정은 여전히 order0다.
  작은 크기의 LDMATRIX/epi4 후보는 채택 전 5회 재확인이 필요하다.
- Colab tcgemm-opt는 후속 작업을 위해 활성 상태지만 진행 중인 측정 작업은 없다.
  다음 작업은 위의 자연 N/스칼라 B/bank 충돌 없는 STS.64 실험 또는 작은 크기 재확인이다.
  완료된 v8을 유실된 작업처럼 다시 시작하지 않는다.
- 목표는 아직 완료되지 않았다. Tensor elapsed 사용률 89.99%와 cuBLAS 93.38%의 차이가 남아 있다.

## v9–v12와 A100 복구 (이전 실행 중 메모를 대체)

- v9 자연 N/스칼라 B/STS.64 배치는 작은 크기 검사 351건을 통과했으나 v8보다 느렸다.
- v10 직접 TF32 변환과 v11 특수값을 보호하는 바이어스 변환은 더 느렸다.
  v11은 작은 검사 522건을 통과했으며, 보호 처리를 넣은 후보는 큰 크기에서 약 54–82 TFLOPS였다.
- v11 이후 통합 검사는 직사각형/짧은 K/alpha-beta 조합 7,772건을 통과했다.
  작은 행렬의 5회 측정 우승 후보는 1024까지 cfg6/order3/copy0/epi4,
  2048에서는 cfg4/order3/copy0/epi4였다. 성능은 0.597, 3.804, 20.503, 49.183, 74.132 TFLOPS다.
  이 작은 크기 설정들은 아직 자동 정책에 반영하지 않았다.
- raster 그룹 0/2/4/8/16/32/64는 큰 행렬에서 유의미한 개선을 보이지 않았다.
- v12 변환 지연과 MMA 뒤 복사 배치는 작은 검사 522건을 통과했다.
  예비 측정에서 기존 cfg0/order3/copy1/epi4를 이기는 후보는 없었다.
  4096 기준 104.135 vs cuBLAS 108.223; 8192 기준 112.863 vs cuBLAS 117.325 TFLOPS.
  핸들 90599는 종료 코드 0으로 끝났으며 압축파일을 다운로드해
  results/2026-10-16/{v12,source-v12.tgz,v12-results.tgz}에 해제했다.
- 이후 엄격 검사 실행 시 기존 tcgemm-opt 세션 유실(404/401)이 보고됐고,
  `colab sessions`에서도 세션이 없음을 확인했다. 같은 이름으로 새 A100 세션을 만들었다.
  qwen35-val은 건드리지 않았다.
- 현재 실행 중인 작업은 핸들 **72783**이다. 새 A100에서 v12를 다시 빌드하고
  **check_v12.py**를 실행한다. 새 Colab 실행 전에 이 핸들을 먼저 조회한다.
  검사기는 CUDA TF32 변환으로 A/B를 독립적으로 양자화하고 그 입력의 FP32 GEMM 결과와
  1e-5로 비교한다. 원래 FP32 입력 기준 1e-3 검사와 alpha0/beta1의 정확한 항등성도 확인한다.
  강화된 검사는 **아직 통과하지 않았다**. 이전 7,772건 통과와 혼동하지 않는다.
- 다음 진단 스크립트 results/2026-10-16/warp_probe.py를 준비했다.
  source-v12에서 분리한 작은 빌드를 만들고 128x256, 256x128 각각의 16-warp 배치
  (WM64/WN32, WM32/WN64)와 epilogue batch2/4/8을 검사한다.
  실제 기본 소스는 바꾸지 않는다. 현재 검사기가 종료된 뒤 실행한다.
- 목표는 계속 진행 중이다. 최고 측정 Tensor elapsed 사용률은 여전히 89.99%, cuBLAS는 93.38%다.
- 호스트 자동 정책에는 위에서 재확인한 작은 크기 구성을 반영했고 README에도 설명했다.
  이 수정은 source-v12 압축파일을 만든 이후이며, 현재 실행 중인 엄격 검사는 압축파일의
  v12 자동 정책을 검사한다. 명시적인 작은 크기 특수화는 이미 측정·검사했다.
  호스트 정책 변경을 포함한 새 자동 경로는 다시 빌드해 검증해야 한다.

## 엄격한 v12 검사 완료; warp 실험 시작

- 핸들 72783이 종료 코드 0으로 완료됐다. **13,108건 모두 PASS**했다.
  독립적인 TF32 양자화 입력 기준 1e-5와 원래 FP32 기준 1e-3 검사를 포함한다.
  `results/2026-10-16/v12-strict.tgz`를 로컬에 보존하고 해제했다.
- warp 실험을 시작하려던 중단된 요청은 실제 실행으로 이어지지 않았다.
  `colab status`에서 IDLE이며 마지막 실행은 이전 검사임을 확인했다.
- 이후 명시적으로 시작한 작업이 현재 실행 중이다. 통합 실행 핸들은 **10008**,
  스크립트는 `warp_probe.py`, 세션은 tcgemm-opt다. 새 실행 전 이 핸들을 조회한다.
  먼저 이전 컴파일 바이너리를 `/content/tcgemm-v12-build.tgz`로 보관한다.
  준비되면 다운로드해 이후 전체 재컴파일을 피한다.

## warp 실험 완료; 직접 epilogue 실험 실행 중

- 핸들 10008이 종료 코드 0으로 끝났다. 소스와 원시 출력은
  `warp-probe-results.tgz`로 로컬에 보존하고 `warp-probe/`에 해제했다.
- 16-warp 구성 중 기존 8-warp 기준을 이긴 것은 없었다. 대표적인 cfg7
  order3/copy1/epi4는 스레드당 레지스터 128개와 spill 쓰기 60B / 읽기 44B가 필요했다.
  warp 수를 늘리기만 하면 레지스터 부담이 커진다.
- Epi2도 느렸다. 8192의 5회 측정은 112.862, 기존 epi4는 **113.456 TFLOPS**,
  같은 실행의 cuBLAS는 **117.968**이었다. 4096 기준 104.493 vs cuBLAS 108.566.
  교체된 물리 A100에서 측정했으므로 이전 런타임의 절대값과 섞지 말고 같은 실행의 비율로 비교한다.
- 검증된 v12 컴파일 바이너리를 로컬 `v12-build.tgz`에 보존했다.
- 다음 실험을 실제 시작했다. 핸들 **31681**, 스크립트 `direct_probe.py`다.
  누산 레지스터에서 자연 N의 float2 출력을 직접 써 shared memory epilogue의 전치·barrier를
  피하는 실험이다. alpha/beta와 입력 계산은 유지하며 기본 커널은 바꾸지 않는다.
  새 실행 전 이 핸들을 다시 조회한다.

## 직접 epilogue 결과; 공개 CUTLASS 진단

- 핸들 31681이 종료 코드 0으로 끝났다. direct-probe-results.tgz를 다운로드·해제했다.
  자연 N float2 직접 출력은 정확도를 통과했으나 shared epilogue보다 느렸다.
  8192의 5회 측정 최고값은 111.616, 자체 기준 113.450, cuBLAS 117.974 TFLOPS다.
  4096 최고값은 102.823, 기준 104.503, cuBLAS 108.567이다. 기본 경로에 반영하지 않았다.
- 현재 실행 핸들은 **75585**, 스크립트는 `cutlass_probe.py`, 세션은 tcgemm-opt다.
  공식 NVIDIA CUTLASS 헤더만 선택적으로 복제하고 정확한 커밋을 기록한다.
  float 입력 TF32 `OpMultiplyAdd` 구성 8개를 빌드한다(3xTF32 FastF32 경로가 아님).
  1140 MHz에서 기존 자체 기준/cuBLAS와 5회 비교한다.
  독립적인 진단이며 자체 커널을 라이브러리로 교체하는 작업은 아니다.
- 원본 소스에서 float/float OpMultiplyAdd가 내부적으로 TF32를 사용함을 확인했다.
  NVIDIA/cutlass include/cutlass/gemm/warp/default_mma_tensor_op_sm80.h의
  DefaultMmaTensorOp 특수화 약 130–169줄이다.

## 공개 CUTLASS 근거와 명령 경로 실험

- 핸들 75585가 완료됐다. CUTLASS cfg1(128x256x32, warp64x64, 3 stages, identity8)은
  4096/8192에서 108.046/117.438 TFLOPS, 같은 실행의 cuBLAS는 108.571/117.942였다.
  자체 기본값은 104.518/113.423이었다. 정확도 기록은 모두 PASS다. 정확한 CUTLASS 커밋,
  소스·바이너리·클럭을 로컬 `cutlass-probe/`와 압축파일에 보존했다.
- 핸들 4352는 1140에서 자체/CUTLASS/cuBLAS를 프로파일하고 종료 코드 0으로 끝났다.
  로컬 `cutlass-profiles/`에 ncu 보고서와 SASS 소스 카운터가 있다.
  Tensor elapsed 사용률은 각각 90.10/93.00/93.40%, 레지스터는 253/228/230개다.
  자체 short-scoreboard/issue는 .03, CUTLASS는 .00, cuBLAS는 .01이다.
  자체 long-scoreboard/issue는 .07, 나머지는 모두 .04다. 자체 실행 명령은
  1.691B로 CUTLASS 2.362B/cuBLAS 2.225B보다 적다. 명령 수만을 최적화 목표로 삼지 않는다.
  자체 IADD3에는 short-SB PC 표본 2,498개가 있다.
- 핸들 23047의 누산기 저장·순회 실험이 종료 코드 0으로 끝났다.
  열 우선 누산기 저장과 새 열 지그재그 LDMATRIX 경로 모두 개선되지 않았다.
  8192 기준 113.458, 열 우선 저장 113.451, cuBLAS 117.937 TFLOPS다.
  `accumulator-probe-results.tgz`를 다운로드·해제했다. 새 커널은 채택하지 않았다.
- 현재 실행은 핸들 **30825**, `mad_probe.py`, tcgemm-opt다.
  런타임 승수 1을 써 IADD로 접히지 않게 한 정수 MAD 반올림과 조건부 MAD를,
  기존 정수 덧셈 반올림과 비교한다. 측정 전 modes0/4/5의 SASS를 보존한다.
  입출력과 TF32 연산은 바꾸지 않는다.

## MAD 완료; B 복사 배치 진단 실행 중

- 핸들 30825가 종료 코드 0으로 끝났다. MAD 명령 경로는 기준보다 빠르지 않았다.
  8192의 5회 측정은 기준 113.440 / 기본 113.456, cuBLAS 117.943 TFLOPS다.
  mad-probe-results.tgz를 다운로드·해제했다. 기본 경로에는 반영하지 않았다.
- SASS에서 복사 배치의 구체적인 차이를 찾았다. 자체 B 복사는 서로 다른 64비트 global
  행 주소를 쓰지만, CUTLASS는 기준 주소 하나에 상수 0x80 열 오프셋을 더해 재사용한다.
  warp-raked 배치는 스레드들이 K행 32개를 나누고 복사 반복에서는 N열을 이동한다.
  SMEM 배치를 바꾸지 않으면서 살아 있는 주소 상태를 줄이는 구조다.
- 현재 핸들 **32076**, `rake_probe.py`, tcgemm-opt가 실행 중이다.
  작은 크기의 전체 검사는 이미 통과했고 큰 크기의 예비 성능 측정 중이다.
  실험 커널에 RAKE_B 템플릿 플래그를 추가했다. 임시 config.epilogue8은 RAKE_B를 선택하지만
  실제 epilogue 배치는 4다. 이 진단용 인코딩을 공개 API로 내보내면 안 된다.
  RAKE_B에 대해 복사 modes1/3, orders3/5/6, rounding0/1/2/3을 검사한다.
- 후속 엄격 검사 `check_rake.py`를 준비했다(32076 종료 후 실행).
  실제 인스턴스화한 진단 구성만 대상으로 직사각형, K32/64/96/160, alpha/beta,
  raster 마지막 그룹을 두 독립 기준과 비교한다. 수정·업로드한 check.cu는 최대 상대 오차도
  출력한다. 기존 13,108건 통과 기록은 유효하지만 새 최대 오차 출력은 아직 실행하지 않았다.

## B 복사 배치의 반복 측정 개선 확인; 엄격 검사 실행 중

- 핸들 32076이 종료 코드 0으로 끝났다. 작은 예비 검사와 큰 정확도 기록 모두 PASS다.
  cfg0/order6/copy3/epi4/round0/RAKE_B의 5회 중앙값:
  4096 **107.3956** vs 이전 기준 104.2999 / 기본 104.5176 / cuBLAS 108.5666 TFLOPS;
  8192 **116.5987** vs 이전 기준 113.4190 / 기본 113.4183 / cuBLAS 117.9223.
  두 크기 모두 같은 실행의 cuBLAS 처리량 대비 98.9%다. 사용률은 아직 측정하지 않았다.
  `rake-probe-results.tgz`를 로컬로 다운로드·해제했다.
- 주의: 자연 N RAKE_B 후보는 ptxas 기준 여전히 레지스터 252개, spill 없음이다.
  주소 재사용은 배치·소스로 확인했지만 **전체 할당 레지스터 수 감소는 확인되지 않았다.**
  레지스터 수가 줄었다고 주장하면 안 된다.
- 현재 핸들 **89725**, `check_rake.py`가 지원하는 모든 RAKE_B/기준 조합을 독립적으로
  양자화 TF32 1e-5 / FP32 1e-3 기준으로 검사한다. 먼저 이 핸들을 조회한다.
  통과하면 1140에서 cfg0/order6/copy3/RAKE_B를 프로파일하고 같은 조건의
  자체 기준/CUTLASS/cuBLAS 보고서와 비교한다. RAKE_B는 아직 기본 경로로 채택하지 않았다.
- 사용자가 정확도 검사 여부를 명시적으로 물었다. 각 성능 실험은 시간 측정 전에 FP32
  정확도를 검사하며 강화된 전체 검사 통과 후에 채택한다고 답했다. 이 요구를 유지하고
  새 검사기가 완료되면 측정된 최대 오차를 보고한다.

## 새 후보의 정확도 확인

- 핸들 89725가 종료 코드 0으로 끝났다. **B 복사 진단 검사 1,200건 모두 PASS**다.
  직사각형 타일, K32/64/96/160, alpha/beta 네 조합, 완전한/불완전한 raster 그룹,
  실제 인스턴스화한 모든 RAKE_B 구성과 기준 경로를 포함한다.
- FP32 기준 최대 정규화 오차는 **3.12642872e-4**(허용치 1e-3)다.
  독립적으로 양자화한 TF32 기준 최대 오차는 **1.10369408e-6**(허용치 1e-5)다.
  alpha0/beta1의 정확한 항등성도 검사했다. rake-check.tgz를 로컬에 보존·해제했다.
- 현재 작업 **95484**, `profile_rake.py`, tcgemm-opt가 실행 중이다.
  1140 클럭에서 기준 경로, cfg0/order6/copy3/RAKE_B/epi4/round0, cuBLAS를
  SASS 표본과 함께 프로파일한다. 추가 실행 전 이 핸들을 조회한다.
  완료 후 결과 경로는 `/content/tcgemm-rake-profiles.tgz`다. RAKE_B 채택과 실제 현재
  자동 정책의 빌드 검증은 아직 남아 있다. 목표는 계속 진행 중이다.

## 기본 copy4 채택과 v13 검증

- 핸들 95484가 완료됐다. rake-profiles.tgz를 다운로드·해제했다.
  동일 1140 MHz의 Tensor elapsed 사용률은 기준 90.11%, RAKE_B **92.32%**, cuBLAS 93.40%다.
  RAKE_B는 레지스터 252개/spill 없음이며 long-SB/issue .07, short-SB/issue .07이다.
  목표까지 1.08%포인트가 남아 있다.
- 기본 소스에 **COPY_MODE4**를 추가했다. B-raked 배치와 MMA 뒤 복사 그룹 발행을 합친 경로다.
  호스트는 order6/epi4/round0만 허용하며 큰 cfg0 타일에서 자동 선택한다.
  진단용 copy3+RAKE_B=true와 같은 알고리즘이고, 진단용 epi8 인코딩은 채택하지 않았다.
  튜너·검사기에 copy4를 추가했다. 작은 자동 설정 cfg6/4 order3/copy0/epi4도 반영했다.
- 현재 소스 압축파일 source-v13.tgz를 업로드·해제했다. 비동기 CPU 빌드 작업
  workerPID15754는 351.21s 후 returncode0으로 완료됐다. Build-done.json을 다운로드했다.
- Epi batch2/8, 명시적인 FMA 결합 순서, SMEM 저장 전 누산값에 배율 적용은
  모두 기존 epi4보다 느렸다. epilogue-rake-results.tgz를 로컬에 보존했다.
- C prefetch 버퍼 3/4/8도 더 느렸고 가장 큰 설정은 크게 느려졌다.
  원시 cbuffer-probe-results.tgz를 다운로드·해제했다. 변경은 채택하지 않았다.
- 현재 통합 실행 핸들 **16231**, `validate_v13.py`: 실제 기본 코드가 엄격 검사
  **13,224건**을 통과했다. 최대 상대 오차는 FP32 **3.12791562e-4**,
  양자화 TF32 **1.10369408e-6**이다. 기본 벤치마크 128..8192를 실행 중이며,
  이후 기존 CLI와 실제 기본 경로/cuBLAS의 ncu를 실행한다. 먼저 이 핸들을 조회한다.
- 다음 진단 raster_probe.py를 준비했다(아직 시작하지 않음). 공식 CUTLASS의
  GemmIdentityThreadblockSwizzle은 N 타일을 묶고 현재 자체 raster는 M 타일을 묶는다.
  16231 종료 후 음수 진단 group 값으로 N 그룹핑, 양수로 M 그룹핑을 검사한다.
  기본 group API는 바꾸지 않는다.
- 현재 README/요약은 후보 측정값을 설명한다. 실제 v13 실행이 끝나면 현재 소스의
  근거에 맞춰 README 검증 상태와 요약을 갱신한다.

## v13 기본 경로 검증 완료; 남은 차이 분석

- 핸들 16231이 종료 코드 0으로 끝났다. v13-results.tgz와 v13-build.tgz를 로컬에 보존했다.
  실제 자동 기본값의 5회 측정은 4096/8192에서 107.401/116.569 TFLOPS,
  cuBLAS는 108.551/117.892였다. 명시적인 copy4는 기본값과 측정 변동 범위 안에서 일치한다.
  기존 CLI7은 모든 크기 128..8192에서 정확도를 통과했다. 클럭 표본은 모두 1140이었다.
  실제 기본값의 ncu 커널 이름으로 특수화 <...,6,4,4,0>을 확인했다. Tensor elapsed는
  **92.34%**, 같은 실행의 cuBLAS는 **93.43%**였다. 위의 엄격 검사 13,224건 모두 통과했다.
- README와 a100_2026-10-16-rake-summary.json을 모든 크기·반복 표본을 포함한
  실제 기본 경로의 측정값으로 갱신했다. tune.py는 copy4를 허용하고 지원하지 않는
  order/copy 및 크기 128의 타일 조합은 건너뛴다(Python만 수정).
- 핸들 56261의 raster 실험이 종료 코드 0으로 끝났다. N 그룹핑의 유의미한 개선은 없었다.
  8192 N-group4 116.632 vs M-group8 116.593; 같은 실행의 cuBLAS 117.898 TFLOPS.
  기본 group8을 유지한다. raster-probe-results.tgz를 다운로드·해제했다.
- 다음 실행은 `load_round_probe.py`다(반환된 핸들은 아래에 기록).
  새 modes4/5/6은 다음 A+B/A만/B만의 fragment를 읽은 뒤 loadNext에서 반환하기 전에
  반올림한다. 현재 fragment를 MMA 직전에 반올림하는 v12와 다르다.
  shared 대기의 PC 표본 근거: rake 프로파일에서 LDSM 결과를 바로 쓰는 IADD3 R160에
  short-SB 표본 4,680개가 있어 가장 큰 지점이었다.
  새 모드는 진단용이며 공개 기본 설정이 아니다.

## v13 이후 진단과 레지스터 할당 실험

- 읽기/반올림 핸들 67069가 종료 코드 0으로 끝났다. 다음 fragment 반올림 modes4/5/6은
  개선되지 않았다. 8192 최고 116.600 TFLOPS vs 기본 116.596, cuBLAS 117.915.
  load-round-probe-results.tgz를 로컬에 보존·해제했다. 기본 경로는 바꾸지 않았다.
- A를 두 단계 앞서 읽는 핸들 42293이 종료 코드 0으로 끝났다.
  정확도는 통과했지만 4096/8192 성능은 80.025/86.184 TFLOPS로 떨어졌다.
  레지스터 242개, **spill 0**이므로 이 저하를 spilling 탓으로 설명하면 안 된다.
  ahead-probe-results.tgz를 다운로드·해제했다. 이후 실행 중인 GPU 작업은 없으며
  colab status에서 IDLE을 확인했다.
- 현재 실행 **16917**, regcap_probe.py, tcgemm-opt.
  같은 기본 B-raked 계산을 레지스터 상한 224/228/232/240/248/252/255로 각각 컴파일한다.
  제한 모듈에서는 launch_bounds를 제거하고, 대조군 0은 원래 launch_bounds를 유지한다.
  원래 기본 경로도 같은 프로세스에서 측정한다. 모듈마다 별도 커널 이름과 래퍼를 쓰며
  기본 소스는 바꾸지 않는다.
- 빌드 근거: 대조군 252개/spill 없음; 상한 224→224개/쓰기+읽기 28B;
  228→228/12B; **232→232/spill 없음**; 240→240/spill 없음; 248→247/spill 없음;
  252→249/spill 없음; 255→252/spill 없음. 이후 정확도·성능을 측정한다.
- 목표는 계속 진행 중이다. 기본 사용률 92.34% vs cuBLAS 93.43%이므로 완료 처리하지 않는다.

## 일시 중지 상태와 최종 벤치마크 요약

- 사용자가 최적화를 중지하고 요약을 요청했다. 이 기록이 위의 실행 중 작업·진행 중 목표
  상태를 대체한다. Colab tcgemm-opt는 IDLE이며 요약 요청은 새 실험 실행을 허용하지 않는다.
- 레지스터 상한 핸들 16917이 종료 코드 0으로 완료됐다. 5회 재확인 결과(TFLOPS):
  4096: cap240 107.864967873, 기본 107.398830457, cuBLAS 108.559598956;
  8192: cap240 117.161640172, 기본 116.625553742, cuBLAS 117.916355366.
  Cap240은 레지스터 240개, spill 0으로 같은 실행의 cuBLAS 처리량 대비 약 99.36%에 도달했다.
  기본 FP32 정확도 검사는 256/512/4096/8192에서 통과했다.
  전체 독립 TF32 검사와 ncu Tensor 사용률 프로파일은 **아직 수행하지 않았고 기본 경로에도
  채택하지 않았다.** 원시 재확인 기록은 results/2026-10-16/regcap-probe/confirmation.jsonl이다.
- 더 세밀한 레지스터 탐색 regcap_fine.py는 시작되지 않았다.
  원격 결과 디렉터리와 압축파일이 없고 벤치마크 결과도 없다.
- 기본 경로는 큰 타일에서 v13 COPY_MODE4/order6/epi4다. 당시에는 별도 번호의 커널이 아닌
  kernel7 안에 있었다. 검증된 자동 성능은 4096/8192에서 107.401/116.569 TFLOPS,
  같은 실행의 cuBLAS는 108.551/117.892다. 엄격 검사 13,224건 모두 통과했다.
  Tensor elapsed는 92.34% vs cuBLAS 93.43%다. 사용률 동등성은 달성하지 못했고 목표는 중지 상태다.

## 2026-11-01: cuBLASLt·CUTLASS 검증과 측정 기록

검증 범위: 로컬 CUDA 12.0 / `sm_80` 기본·CUTLASS 포함 빌드와 검사 프로그램 컴파일 통과.
2026-11-01 Colab A100-SXM4-40GB (39.49 GiB, SM80, 108 SM), CUDA 13.0.88에서도
동일 소스를 빌드하고 GPU 수치 검사·성능 측정을 완료했습니다.

### 라이브러리 A100 측정 결과

단위는 TFLOPS이며, `M=N=K`, `α=0.5`, `β=3`, warmup 1회 후 50회 평균입니다.
이번 실행의 세 구현을 같은 표에서 비교합니다. 10월의 자체 커널 실험과는 별도 측정입니다.

| 구현 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|
| cuBLAS TF32 (100) | 50.34 | 74.60 | 104.27 | 134.16 |
| cuBLASLt TF32 (101) | 50.15 | 74.58 | 104.24 | 134.51 |
| CUTLASS TF32 (102) | 44.79 | 65.34 | 84.90 | 104.04 |

직사각형 4개(shape 복귀 포함) × alpha/beta 4조합 × 구현 2개의 **32개 검사 전부 통과**했습니다.
추가로 세 구현 각각 128~8192의 7개 정방행렬 검사도 모두 통과했습니다.
정방행렬 검사에서 FP32 기준 최대 정규화 오차는 cuBLASLt `3.858e-4`, CUTLASS `4.370e-4`로,
허용치 `1e-3` 이내였습니다. 최대 절대 오차는 각각 `0.5435`, `0.5696`입니다.

이번 설정에서 cuBLASLt와 cuBLAS는 비슷했고, CUTLASS 기본 설정은 8192에서 cuBLASLt의 약 77%였습니다.
CUTLASS 타일·stage·스케줄 튜닝은 수행하지 않았습니다.
각 벤치마크 프로세스의 입력은 기존 생성기의 같은 분포를 사용하지만 seed가 달라 동일한 값은 아닙니다.
작은 크기의 시간에는 host dispatch 간격도 반영되며, CUDA Graph 기반 순수 커널 시간은 아닙니다.

[원본 로그·장치 정보·실행 스크립트](benchmarks/library_a100_20261101/),
[전체 크기별 수치](benchmarks/library_a100_20261101/summary.csv)를 보관했습니다.
A100 세션 종료 후 서버에 활성 세션이 없음을 확인했습니다.


## 2026-11-02: 더블 버퍼링 이후 최적화를 학습 단계로 분리

- `07_frag_double_buffer.cuh`는 고정된 더블 버퍼링 기준 커널이다.
  `08_ldmatrix_distributed.cuh`, `09_prefetch_epilogue.cuh`, `10_raked_b_copy.cuh`에
  단계별 최적화를 누적한 전체 커널 본문을 추가했다(CLI 번호 8/9/10).
  고정 단계들은 같은 타일 정책을 공유해 단계 비교 시 타일까지 함께 바뀌지 않게 했다.
- 기존 설정 조합 구현은 include 경로를 제외하고 바이트 단위로 동일하게
  `kernels/detail/frag_pipe_tuning.cuh`에 보존했다. 네이티브 튜너와 설정 검사는
  `run_configured_frag_pipe`를 호출한다. 명시적인 TC7_* CLI 실험은 번호 7로 계속 지원한다.
  TC7_* 환경변수를 지정하지 않으면 번호 7은 고정 학습 단계로 동작한다.
- 라이브러리 구현을 내용 변경 없이 `08_library_gemm.cuh`에서 `99_library_gemm.cuh`로 옮겼다.
  번호 100/101/102는 유지했다. include, README, 벤치마크 순서, 노트북 단계 이름을 갱신했다.
  과거 소스 manifest는 해당 측정 당시의 경로를 유지한다.
- 기존 직사각형/짧은 K/alpha-beta 검사에 고정 단계 검사도 추가했다.
  FP32 및 독립적으로 TF32 양자화한 입력의 기준 결과와 비교한다.
- 벤치마크·검사기·네이티브 튜너의 로컬 CUDA sm_80 빌드·링크를 통과했다.
  최종 고정 단계 컴파일(4개 단계 x 3개 타일 형상)은 경고와 register spill 없이 통과했다.
  원래 튜닝·라이브러리 내용 보존, 노트북/Python/셸 문법, README 링크와 diff 공백을 확인했다.
- 로컬 GPU 접근이 차단되어 새 단계의 런타임 정확도 검사와 성능 측정은 실행하지 못했다.
  기존 성능 표는 과거 측정 기록으로 유지한다.
