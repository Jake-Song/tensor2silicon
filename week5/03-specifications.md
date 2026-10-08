# ③ 성능표의 숫자는 어떤 조건의 값일까?

[5주차 목차](./README.md) · 이전 → [② 하드웨어](./02-hardware.md) · 다음 → [④ 소프트웨어](./04-software.md)

최대 연산 성능에는 최소한 **연산 종류, 입력·누산 자료형, 희소성, 집계 단위**가 필요하다. 아래는 2026-10-08에 확인한 제품별 공시값이다. 이번 연산의 측정 결과가 아니다.

## 1. 공시 연산 성능

| 대상·집계 범위 | 공시값 | 읽어야 할 조건 | 아직 연결할 수 없는 값 |
|---|---|---|---|
| H100 SXM GPU 1개, BF16 Tensor Core | **1,979 TFLOP/s** | 제품 표의 별표는 **sparsity 적용** | dense `XW`의 실제 처리율 |
| H100 SXM GPU 1개, INT8 Tensor Core | **3,958 TOPS** | 역시 sparsity 적용. BF16과 자료형이 다름 | BF16 커널의 실행시간 |
| TPU v5e 칩 1개, BF16 | **197 TFLOP/s** | 칩당 peak. 사양 표에 별도 sparse 배수 각주 없음 | 특정 shape의 실제 MXU 이용률 |
| TPU v5e 칩 1개, INT8 | **393 TOPS** | 정수 연산 peak. BF16 행과 분리 | 양자화·변환을 포함한 추론 시간 |
| Core Ultra 9 288V의 NPU | **48 TOPS, INT8** | NPU 항목. `Sparsity Support: Yes`는 별도 항목 | 사양 페이지에서 48의 dense/sparse 조건·누산은 이번 조사에서 미확인 |
| Core Ultra 9 288V 전체 | **120 TOPS, INT8** | NPU만의 값이 아닌 프로세서 전체 항목 | 한 그래프가 이 처리율을 달성한다는 보장 |

출처: [NVIDIA H100 Product Specifications의 **SXM 열과 각주**][h100], [Google TPU v5e System architecture][v5e], [Intel 288V Essentials / NPU Specifications][intel].

H100의 구조화된 희소성은 조건에 맞는 Tensor Core 연산의 처리율을 두 배로 높이는 기능이다. 따라서 공시된 sparse BF16 수치를 절반으로 환산하면 dense 대응값은 **약 990 TFLOP/s**다. 이는 반올림된 공시값 `1,979 / 2 ≈ 989.5`에서 얻은 계산이며, `ReLU(XW+b)` 실측값이 아니다. 임의의 0이 많은 행렬이 이 희소성 경로를 자동으로 사용하는 것도 아니다. [NVIDIA Hopper Architecture의 sparsity 설명][hopper]

INT8의 정수 곱셈·덧셈과 BF16의 부동소수점 연산은 오차·범위·실행 경로가 다르다. 같은 `tera = 10¹²` 접두사가 있어도 TOPS/TFLOP/s 숫자만으로 순위를 매기지 않는다. MAC/FMA 1회를 곱셈+덧셈의 **2 operations**로 세었는지도 확인한다. 이 과제의 MatMul 계산은 `2MKN` 관례를 사용한다.

## 2. 입력·계산·누산·출력은 각각 적는다

| 예시 경로 | 입력·가중치 | 곱셈·누산 | bias·출력 | 판정 |
|---|---|---|---|---|
| 공통 커널 비교 계약 | BF16 | BF16 곱셈 / FP32 누산 | FP32 | 이 계약을 실제로 지킨 구현끼리 비교 |
| H100 Tensor Core 구현 후보 | BF16 | BF16 / FP32를 선택하고 생성 코드로 확인 | FP32를 유지하도록 구현 | 사양의 BF16 peak만으로 구현을 확인할 수 없음 |
| v5e MXU 구현 후보 | BF16 | BF16 / FP32 경로. JAX의 반환 dtype·정밀도 설정도 확인 | FP32를 유지하도록 구현 | 출력 dtype만으로 내부 알고리즘을 단정하지 않음 |
| OpenVINO NPU 비양자화 경로 | 모델에 FP32/F16 표현 가능 | 문서는 HW 계산 FP16 명시. 실제 누산은 이번 조사에서 미확인 | 그래프·변환·실행 결과 확인 | FP32 모델 입력을 FP32 하드웨어 계산으로 해석하지 않음 |
| INT8 양자화 경로 | scale·zero-point·양자화 축도 필요 | 정수 곱셈, 누산·재양자화 조건 별도 확인 | 복원·clipping 여부 기록 | 기존 BF16 계약과 다른 실험 |

연산 지원의 근거는 [Triton MatMul의 FP32 accumulator 예][triton], [Scaling Book의 TPU MXU 설명][tpus], [JAX dot의 precision / preferred_element_type][jax-dot], [OpenVINO NPU의 Supported Inference Data Types][npu]다. 표의 “계약”과 “구현 후보”는 이번 과제를 위한 설계이며 실행 검증을 마친 결과가 아니다.

## 3. 메모리와 연결 대역폭

| 항목 | H100 SXM 1개 | TPU v5e 1칩 | 288V NPU 1개 |
|---|---|---|---|
| 큰 메모리 | 80 GB HBM3 | 16 GB HBM, 공식 표기 유지 | 프로세서 사양 최대 32 GB LPDDR5X. NPU 전용 용량이 아님 |
| 큰 메모리 대역폭 | 3.35 TB/s | **800 GiB/s**, 확인한 공식 문서 표기 | NPU가 지속적으로 쓸 수 있는 유효 대역폭은 아직 미측정 |
| 온칩 저장 공간 | H100 L2 50 MB, Shared Memory·레지스터 | VMEM·벡터 레지스터. v5e의 상세 용량은 이번 조사에서 미확인 | Scratchpad 등. 해당 SKU의 상세 용량은 이번 조사에서 미확인 |
| 가속기 간 연결 | NVLink 900 GB/s, 양방향 합계 | ICI 400 GB/s/칩, 양방향 합계 | 이번 단일 NPU 과제에서 해당 없음 |
| 호스트 연결 | PCIe Gen5 128 GB/s, 양방향 합계 | 호스트 연결의 해당 구성별 대역폭은 이번 조사에서 미확인 | SoC 내부 연결·공유 메모리. GPU의 PCIe 전송 모델과 구분 |

출처: [H100 제품 표][h100], [Hopper Tuning Guide의 L2 / NVLink][tuning], [Hopper Architecture의 PCIe 설명][hopper], [v5e 표][v5e], [Intel 메모리 사양][intel], [NPU 4 구조 자료][npu4]. 상세 메모리 구조는 [②](./02-hardware.md)를 참고한다.

- `1 TB/s = 10¹² Byte/s`, `1 GiB/s = 2³⁰ Byte/s`다. 위 v5e 값은 계산하면 약 **858.99 GB/s**다. Scaling Book의 `8.2×10¹¹ Byte/s` 표기와 다르므로 출처·단위를 섞어 평균내거나 같은 값으로 쓰지 않는다. 실제 제품 사양 기반 계산을 추가한다면 어느 값을 채택했는지 명시한다. [Scaling Book TPU specs][tpus]
- 양방향 합계는 한 방향 전송률이 아니다. H100의 PCIe 128 GB/s는 방향당 64 GB/s이며 실제 복사는 프로토콜·호스트·메모리의 제약을 받는다. NVLink나 ICI 수치를 `HBM Q/BW`의 BW에 넣지 않는다.
- 시스템 메모리의 용량·최대 속도는 NPU에 예약된 용량·보장된 대역폭이 아니다. 캐시 크기 역시 모든 커널이 독점할 수 있는 작업 버퍼 크기를 뜻하지 않는다.

## 4. 전력 사양을 읽는 법

H100 SXM의 최대 TDP는 설정 가능한 **최대 700 W**다. 288V의 **Processor Base Power 30 W / Maximum Turbo Power 37 W**는 프로세서 전력 항목이며 NPU 단독의 측정 전력이 아니다. v5e의 칩 전력은 이번에 확인한 사양 표에서 확인하지 못했다. [H100][h100], [Intel 288V][intel]

따라서 `TFLOP/s ÷ TDP`로 세 장치의 실제 추론 효율을 결론 내리지 않는다. 같은 품질·부하·측정 경계에서 구한 J/request 또는 J/sample이 필요하다. 계산 방법은 [⑤](./05-measurement.md)에 있다.

[h100]: https://www.nvidia.com/en-us/data-center/h100/
[v5e]: https://docs.cloud.google.com/tpu/docs/v5e
[intel]: https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html
[hopper]: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
[tuning]: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
[triton]: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
[tpus]: https://jax-ml.github.io/scaling-book/tpus/
[jax-dot]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot.html
[npu]: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
[npu4]: https://cdrdv2-public.intel.com/824436/2024_Intel_Tech%20Tour%20TW_Lunar%20Lake%20AI%20Hardware%20Accelerators.pdf
