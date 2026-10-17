# G4 BF16 matmul 최적화 기록

## 실행 조건

- 시작: 2026-10-17 05:39:31 UTC. 총 예산 2시간, 07:29:31 UTC에 신규 실험 중단.
- Colab 세션: `bf16-g4-1017`, 실제 장비 RTX PRO 6000 Blackwell Server Edition.
- SM120, 188 SM, 101,974,081,536 bytes VRAM, 블록당 opt-in SMEM 101,376 bytes.
- CUDA 13.0.88, PyTorch 2.11.0+cu130, Nsight Compute 2025.3.1.

## 계산 계약 변경

사용자 요청으로 초기 `alpha*AB + beta*C`에서 **`D=A@B`만** 수행하도록 변경했다.
A/B는 BF16, D와 Tensor Core 내부 누산은 FP32다. C 입력, alpha/beta 인자,
C 로드 및 epilogue 산술을 직접 작성한 커널에서 제거했다. cuBLASLt도 alpha=1,
beta=0으로 비교한다. 변경 전 결과는 최종 matmul 성능표에 사용하지 않는다.

## 초기 구현과 레이아웃

- SIMT → global WMMA → SMEM WMMA → 직접 MMA/LDMATRIX → cp.async → fragment
  더블 버퍼링을 각각 실행 가능한 단계로 구현했다.
- 초기 A SMEM 행은 BK=32에서도 64개 BF16을 예약했다. BF16 32열에는 행 parity가
  이미 bank의 상/하반을 선택하므로, `((row >> 1) & 3) * 8`로 XOR하면
  행당 32개만으로 8개 행의 16-byte 접근을 서로 다른 bank group에 배치할 수 있다.
- BK=64는 `(row & 7) * 8`을 사용한다. B는 N 방향이 64 이상이므로 같은
  8행 XOR를 사용한다. 주소와 실제 결과를 독립적으로 검증한 뒤 채택한다.
- 첫 레지스트리의 타일 이름이 warp 모양을 포함하지 않아 두 설정이 겹쳤다.
  이름에 warp 크기를 추가했다. 해당 초기 탐색 결과는 최종 비교에서 제외한다.

## 1차 순수 matmul 측정

- 압축 A 레이아웃은 CPU에서 144개 layout/rasterization 검사를 통과했다.
- 변경 전 alpha/beta GEMM 결과를 제외하고 다시 측정했다. 기존 event 배치 측정에서
  4096 WMMA global 약 52.6, 기본 cp.async 약 335.6, 타일 튜닝 최고 약 353.6
  TFLOP/s였다. 같은 실행 cuBLASLt는 약 391.5였다. 이 값은 탐색용이며 최종
  CUDA Graph 측정과 직접 섞지 않는다.
- Nsight Compute: `t128x128k32w64x64s2`의 Tensor elapsed는 4096에서 77.91%,
  8192에서 87.64%; 같은 프로파일링의 cuBLAS는 84.68% / 95.21%였다.
  사용 레지스터 213, spill 0. occupancy를 높이는 것만으로 해결된다고 볼 근거는 없다.
- cuBLAS가 선택한 커널은 4096 `cutlass_80_tensorop_s16816gemm_bf16_256x128_32x3_nn_align8`,
  8192 `...128x256_32x3_nn_align8`였다. 이름과 하드웨어 카운터는 프로파일링
  결과이며, 라이브러리 내부 구현에 관한 추정과 구분한다.
- 타일 경계 barrier를 마지막 MMA 앞으로 옮기고 다음 tile fragment를 미리 읽는
  `x*` 후보는 정확도를 통과했지만 기존 최고보다 느렸다. 예: 4096 기존 약 0.389 ms,
  `x128_s3` 약 0.402 ms. 따라서 이 기법을 기본 설정으로 채택하지 않았다.
- B의 인접 N fragment 두 개를 `ldmatrix.x4.trans` 한 번으로 읽는 `p*` 후보를 추가했다.
- 이 시점 전체 GPU 검증은 6,952건 PASS, 최대 정규화 오차 5.94e-7,
  최대 상대 L2 오차 3.24e-7였다. 74개 설정의 비정렬 shape 거부도 확인했다.

## 측정 방식 보완

작은 행렬에서는 CPU의 다음 launch가 늦으면 CUDA Event 구간에도 GPU idle이 들어간다.
최종 기본 측정은 각 후보와 cuBLAS 모두 동일하게 여러 matmul을 CUDA Graph로 capture하고,
graph 실행 구간을 CUDA Event로 잰 뒤 연산 수로 나눈다. capture/instantiate/warmup은 제외한다.
`BF16_TIMING=event`로 기존 직접 launch 측정을 별도 재현할 수 있다. 각 JSON row에
`timing`을 기록하며 두 방식을 섞어서 비교하지 않는다.

## 큰 타일 탐색과 주소 계산

- 32×32 / 32×64 / 64×32 타일을 추가해 작은 행렬의 block 수를 늘렸다.
  단순히 128×128 설정을 모든 크기에 적용하면 작은 크기에서 SM을 충분히 쓰지 못했다.
- B fragment 두 개의 통합 로드, 복사 포인터의 단계별 갱신, MMA의 M/N 우선 순회,
  serpentine·2×2 순회를 각각 후보로 유지했다. 큰 타일의 copy 발행을 MMA 앞/뒤로
  나누어 비교했다. 이 시점 등록된 후보는 264개다.
- 탐색 중 8192의 row group=32는 group=8보다 좋아졌으나 64는 다시 느려졌다.
  L2 재사용 설명은 가설이며, 실제 HBM 트래픽이나 L2 hit 변화까지 이 결과만으로
  확정하지 않는다.
- Nsight 실행 명령 수는 8192에서 초기 `z128x256s3c4o2xtrue` 약 1.076B,
  해당 실행의 cuBLAS 약 0.758B였다. SASS/PC sampling에 정수 주소 계산과
  predicated-off UIADD3 명령이 많아, fragment SMEM 주소 사전 계산과 modulo 없는
  ring index를 추가 후보로 삼았다. 명령 수만으로 시간 차이 전체를 설명하지 않는다.

## 탈락한 주소 후보와 수정

`h128x256_addr/both`의 첫 구현은 72개 GPU 검사에서 탈락해 성능 측정을 중단했다.
작은 N 타일을 추가하면서 B swizzle의 mask를 `BN/8-1`로 확장한 탓에 BN=256의
swizzle 주기가 32행이 되었지만, 새 주소 사전 계산은 8행 주기를 가정하고 있었다.
기존 load/copy 양쪽은 같은 식을 써 정상이었으나 새 사전 계산만 달랐다.

B swizzle을 BN=32에서는 `(row>>1)&3`, BN>=64에서는 `row&7`로 명시했다.
주소 사전 계산 식의 동치 검사를 CPU 검증에 추가해 총 180건으로 확장했다.
Inline PTX shared load의 숨은 메모리 의존성도 `memory` clobber로 명시했다.
두 수정을 혼동하지 않으며, swizzle 주기 불일치가 확인된 정확도 원인이다.
수정 이후에는 전체 GPU 검증과 성능 탐색을 다시 수행한다.

## 실제 클럭 확인

2430 MHz lock 요청은 성공했지만 telemetry에는 실제 2340~2422 MHz가 관측됐다.
NVML 누적 event counter에는 SW power capping과 일부 SW thermal slowdown이 있었다.
이는 설정값이 실제 주파수를 보장하지 않음을 보여 준다. 최종 제어 비교는 목표를
2100 MHz로 낮추고 실제 관측 범위를 함께 저장한다. 메모리 클럭은 변경하지 않았다.
실험 마지막에는 코어 클럭 자동 제어를 복원한다.

## 최종 검증과 채택

수정한 전체 264개 후보에서 GPU 비교 17,736건, CPU FP64 비교 24건, 부적합 shape
거부 264건을 통과했다. CPU 레이아웃 180건, 분석 단위 검사 5건도 통과했다.
최종 설정은 selected.json에 저장했다. 1% 미만의 차이는 더 단순한 Q 경로를 채택했다.

두 seed 각각 5회씩 128~8192의 단계별·선택 커널·cuBLAS를 재측정했다. 2100 MHz
목표 데이터와 자동 클럭 데이터를 별도 디렉터리에 저장했다. 256·2048에서는 두 seed
모두 cuBLAS보다 빨랐으나 다른 크기는 cuBLAS가 앞섰다. 최종 결과의 전체 출력
검사는 모두 통과했다. 최종 Nsight 5개 보고서와 SASS를 보관했다. 14개 계산 소스
해시는 다운로드 이후 로컬 파일과 일치함을 확인했다.

## 저장·종료 확인

다운로드한 release SHA256은 원격 파일과 일치했다. CPU 모드로 노트북 7개 코드 셀을
실행해 표·그래프 2개를 생성했다. 06:29 UTC에 GPU 클럭 reset이 성공했고 전용
`bf16-g4-1017` 세션을 종료했다. 기존 별도 A100 세션은 유지했다. 종료 후 잔여
크레딧은 229.91 CU이며, 계정의 다른 세션 사용량을 포함하므로 초기 잔액과의 차이를
이 실험 단독 비용으로 해석하지 않는다.
