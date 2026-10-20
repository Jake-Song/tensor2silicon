# ③ 성능표의 숫자는 어떤 조건의 값일까?

[5주차 목차](./README.md) · 이전 → [② 하드웨어](./02-hardware.md) · 다음 → [④ 소프트웨어](./04-software.md)

## 1. 공시 연산 성능

| 대상·집계 범위 | 표기값 | 읽어야 할 조건 | 아직 연결할 수 없는 값 |
|---|---|---|---|
| H100 SXM GPU 1개, BF16 Tensor Core | **약 990 TFLOP/s (추산)** | sparsity 적용 공시값 1,979를 절반으로 환산 | dense `XW`의 실제 처리율 |
| H100 SXM GPU 1개, INT8 Tensor Core | **약 1,979 TOPS (추산)** | sparsity 적용 공시값 3,958을 절반으로 환산. BF16과 자료형이 다름 | BF16 커널의 실행시간 |
| TPU v5e 칩 1개, BF16 | **197 TFLOP/s** | 칩당 peak. 사양 표에 별도 sparse 배수 각주 없음 | 특정 shape의 실제 MXU 이용률 |
| TPU v5e 칩 1개, INT8 | **393 TOPS** | 정수 연산 peak. BF16 행과 분리 | 양자화·변환을 포함한 추론 시간 |
| Core Ultra 9 288V의 NPU | **48 TOPS, INT8** | NPU 항목. `Sparsity Support: Yes`는 별도 항목 | 사양 페이지에서 48의 dense/sparse 조건·누산은 이번 조사에서 미확인 |
| Core Ultra 9 288V 전체 | **120 TOPS, INT8** | NPU만의 값이 아닌 프로세서 전체 항목 | 한 그래프가 이 처리율을 달성한다는 보장 |

출처: [NVIDIA H100 Product Specifications의 **SXM 열과 각주**][h100], [Google TPU v5e System architecture][v5e], [Intel 288V Essentials / NPU Specifications][intel].

> **추산값 안내:** H100의 표 값은 구조화된 sparsity 적용 공시 peak를 절반으로 나눈 dense 대응 추정치다. 실측 성능이나 보장값은 아니며, 조건에 맞는 희소성 형식과 실행 경로가 필요하다. 임의의 0이 많은 행렬에 그대로 적용되는 값이 아니다. [NVIDIA Hopper Architecture의 sparsity 설명][hopper]

## 2. 입력·계산·누산·출력은 각각 적는다

| 예시 경로 | 입력·가중치 | 곱셈·누산 | bias·출력 | 
|---|---|---|---|
| 공통 커널 비교 계약 | BF16 | BF16 곱셈 / FP32 누산 | FP32 | 이 계약을 실제로 지킨 구현끼리 비교 |
| H100 Tensor Core 구현 후보 | BF16 | BF16 / FP32를 선택하고 생성 코드로 확인 | FP32를 유지하도록 구현 | 사양의 BF16 peak만으로 구현을 확인할 수 없음 |
| v5e MXU 구현 후보 | BF16 | BF16 / FP32 경로. JAX의 반환 dtype·정밀도 설정도 확인 | FP32를 유지하도록 구현 | 출력 dtype만으로 내부 알고리즘을 단정하지 않음 |
| OpenVINO NPU 비양자화 경로 | 모델에 FP32/F16 표현 가능 | 문서는 HW 계산 FP16 명시. 실제 누산은 이번 조사에서 미확인 | 그래프·변환·실행 결과 확인 | FP32 모델 입력을 FP32 하드웨어 계산으로 해석하지 않음 |

## 3. 메모리와 연결 대역폭

| 항목 | H100 SXM 1개 | TPU v5e 1칩 | 288V NPU 1개 |
|---|---|---|---|
| 큰 메모리 | 80 GB HBM3 | 16 GB HBM, 공식 표기 유지 | 프로세서 사양 최대 32 GB LPDDR5X. NPU 전용 용량이 아님 |
| 큰 메모리 대역폭 | 3.35 TB/s | **800 GiB/s**, 확인한 공식 문서 표기 | NPU가 지속적으로 쓸 수 있는 유효 대역폭은 아직 미측정 |
| 온칩 저장 공간 | H100 L2 50 MB, Shared Memory·레지스터 | VMEM·벡터 레지스터. v5e의 상세 용량은 이번 조사에서 미확인 | Scratchpad 등. 해당 SKU의 상세 용량은 이번 조사에서 미확인 |
| 가속기 간 연결 | NVLink 900 GB/s, 양방향 합계 | ICI 400 GB/s/칩, 양방향 합계 | 이번 단일 NPU 과제에서 해당 없음 |
| 호스트 연결 | PCIe Gen5 128 GB/s, 양방향 합계 | 호스트 연결의 해당 구성별 대역폭은 이번 조사에서 미확인 | SoC 내부 연결·공유 메모리. GPU의 PCIe 전송 모델과 구분 |

출처: [H100 제품 표][h100], [Hopper Tuning Guide의 L2 / NVLink][tuning], [Hopper Architecture의 PCIe 설명][hopper], [v5e 표][v5e], [Intel 메모리 사양][intel], [NPU 4 구조 자료][npu4]. 상세 메모리 구조는 [②](./02-hardware.md)를 참고한다.

- `1 TB/s = 10¹² Byte/s`, `1 GiB/s = 2³⁰ Byte/s`다. 위 v5e 값은 계산하면 약 **858.99 GB/s**다. Scaling Book은 `8.2×10¹¹ Byte/s` 표기

## 4. 전력 사양을 읽는 법

H100 SXM의 최대 TDP는 설정 가능한 **최대 700 W**다. 288V의 **Processor Base Power 30 W / Maximum Turbo Power 37 W**는 프로세서 전력 항목이며 NPU 단독의 측정 전력이 아니다. v5e의 칩 전력은 이번에 확인한 사양 표에서 확인하지 못했다. [H100][h100], [Intel 288V][intel]

같은 품질·부하·측정 경계에서 J/request 또는 J/sample 구할 수 있다. 계산 방법은 [⑤](./05-measurement.md)에 있다.

> **참고:** TDP(Thermal Design Power)는 열·전력 설계를 위한 제품 사양이지, 작업 중 실제 소비전력이나 에너지 측정값은 아니다. 제품마다 전력 항목의 정의와 범위도 다를 수 있다.

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
