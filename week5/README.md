# 5주차: GPU·TPU·NPU를 어떤 조건으로 비교할까?

같은 AI 작업에 쓸 가속기를 비교하려면 **작업·품질·측정 범위·장치 수**부터 맞춘다. 제품의 최대 연산 성능은 이 조건을 확인한 뒤 읽는다.

[4주차](../week4/README.md)에서는 Roofline과 Tiling으로 연산 성능·메모리 대역폭·데이터 재사용을 연결했다. 이번에는 그 관점으로 실제 제품의 구조, 소프트웨어 지원 조건, 성능표를 읽는다. 사례는 **NVIDIA H100 SXM, Google TPU v5e, Intel Core Ultra 9 288V의 NPU**다. 이 세 제품의 차이를 GPU·TPU·NPU 전체의 차이로 일반화하지 않는다.

이 자료는 **2026-10-08에 확인한 문헌과 직접 계산한 예제**다. 가속기에서 새로 실행한 벤치마크는 없다. 실제 설치 버전과 컴파일 성공 여부는 미확인이고, 지연시간·에너지는 아직 미측정이다. 문서의 버전과 실제 실행 환경의 버전을 구분한다.

## 1. 공통으로 읽을 자료

| 자료 | 읽을 범위 | 중심 질문 |
|---|---|---|
| [Scaling Book 12장: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) | **GPUs vs. TPUs at the chip level**부터. 필요하면 **What Is a GPU? / Memory** | 행렬곱·원소별 연산·실행 제어·데이터 재사용은 어디서 이루어지는가? |
| [OpenVINO 2026: NPU Device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) | 도입부의 컴파일 예, **Supported Inference Data Types / Dynamic Shapes / Limitations** | 입력 자료형·shape·컴파일러·드라이버가 지원 여부에 어떤 영향을 주는가? |
| [MLPerf Inference: Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/) | **Scenarios & Metrics / Divisions / Submission Information** | 모델·품질·요청 방식·장치 수·소프트웨어를 왜 함께 기록하는가? |

Scaling Book의 과거 가격은 현재 가격으로 사용하지 않는다. 구조 설명에 등장하는 TPU v5p·v6e 등의 수치를 v5e에 옮겨 적지 않는다. OpenVINO의 `/2026/`, JAX의 `/latest/`, MLPerf의 현재 규칙 페이지도 갱신되므로 열람일과 실제 사용한 릴리스를 남긴다.

## 2. 주제별 학습 문서

| 순서 | 문서 | 읽고 설명할 내용 |
|---|---|---|
| ① | [비교 단위와 사용 목적](./01-comparison-scope.md) | 칩·호스트·시스템, 학습·추론, 커널 비교·품질 비교 |
| ② | [하드웨어 구조와 데이터 이동](./02-hardware.md) | 세 제품의 연산 장치·큰 메모리·온칩 저장 공간·호스트 연결 그림 |
| ③ | [FLOPs/s·TOPS와 제품 사양 읽기](./03-specifications.md) | 자료형·누산·희소성·집계 단위·메모리와 연결 대역폭 |
| ④ | [모델에서 장치 실행까지](./04-software.md) | CUDA·Triton, JAX·XLA·Pallas, OpenVINO NPU 경로와 지원 조건 |
| ⑤ | [지연시간·처리량·품질·에너지](./05-measurement.md) | 컴파일·전송·준비 실행·동기화, MLPerf 결과를 읽는 순서 |
| ⑥ | [공통 연산의 비교 조건표와 손계산](./06-comparison-exercise.md) | 배치 1·128 조건표, 가상 장치의 Roofline 계산 |

④의 지원 조건과 ⑥의 표는 함께 읽는다. 모델 입력이 FP32라는 사실만으로 내부 곱셈과 누산까지 FP32라고 판정할 수는 없다.

## 3. 각자 정리할 내용

한 제품을 맡아 [제출용 기록 양식](./comparison-template.md)을 채운다.

1. 제조사·제품·세대·할당된 장치 수와 실제 연산에 사용한 수를 적는다.
2. 운영체제·프레임워크·컴파일러·런타임·드라이버의 정확한 버전을 남긴다. 실행하지 않았다면 그 사실을 적는다.
3. 연산 장치·메모리·실행 경로를 작은 그림으로 연결한다.
4. 비교표의 숫자에 자료형·희소성·배치·측정 범위와 출처를 붙인다.
5. 확인하지 못한 조건과 그 때문에 아직 내릴 수 없는 결론을 적는다.

| 표기 | 사용 기준 |
|---|---|
| 문헌 확인 | 출처와 해당 절에서 직접 확인한 사실 |
| 계산 | 입력값·가정·공식으로 구한 값. 실측과 구분 |
| 미공개 | 출처에서 비공개라고 명시한 값. 검색에서 찾지 못했다는 뜻으로 쓰지 않음 |
| 이번 조사에서 미확인 | 확인한 문헌만으로 값을 확정할 수 없음 |
| 아직 미측정 | 장치에서 실행·측정하지 않음 |

## 4. 공통 과제

추론 연산 `Y = ReLU(XW + b)`를 사용한다.

- `X: [M, 1024]`, `W: [1024, 1024]`, `b: [1024]`, `Y: [M, 1024]`
- `M = 128`과 `M = 1`을 별도로 비교한다. 역전파와 가중치 갱신은 없다.
- **커널 비교:** 입력·계산·누산·출력 자료형까지 맞춘다. 지원을 확인하지 못한 장치는 동등한 조건의 결과로 넣지 않는다.
- **모델·시스템 비교:** 같은 입력과 품질 목표를 맞추고 각 장치의 실행 경로를 허용한다. 내부 정밀도와 장치별 분담은 공개한다.

[⑥의 조건표와 풀이](./06-comparison-exercise.md)를 채우면 문헌 조사만으로도 참여할 수 있다. 가상 장치 A·B 계산은 다음 명령으로 재현한다. Python 표준 라이브러리만 사용하며 실제 가속기 성능을 측정하지 않는다.

```bash
python3 week5/calculate_comparison.py
```

## 5. 선택 실습: 무료 원격 환경과 CPU 참여

[Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)는 무료 GPU·TPU 자원과 함께, 할당·제품 종류·사용 한도가 보장되지 않는다고 설명한다. **H100 SXM이나 TPU v5e를 무료로 배정받는다는 뜻은 아니다.** 실제 배정된 제품명으로 비교표를 바꾼다.

| 참여 방식 | 시작할 곳 | 이번 주에 남길 내용 |
|---|---|---|
| Colab GPU | [기존 PyTorch GPU 노트북](../notebooks/pytorch_gpu_relu_linear.ipynb) · [Colab에서 열기](https://colab.research.google.com/github/Jake-Song/tensor2silicon/blob/main/notebooks/pytorch_gpu_relu_linear.ipynb) | 실제 GPU·메모리·장치 수·버전, 추론 설정, 두 shape의 실행 경로 |
| Colab TPU | [기존 JAX TPU 노트북](../notebooks/jax_tpu_relu_linear.ipynb) · [Colab에서 열기](https://colab.research.google.com/github/Jake-Song/tensor2silicon/blob/main/notebooks/jax_tpu_relu_linear.ipynb) | 실제 TPU 세대·할당 수·사용 수, JAX/jaxlib/libtpu 버전, 컴파일·완료 대기 |
| 로컬 Intel NPU | [OpenVINO NPU 문서](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html) | 보유 장치의 정확한 SKU, NPU 인식·컴파일·출력 검증 결과 |
| CPU·문헌·시뮬레이션 | [⑥ 손계산](./06-comparison-exercise.md), 위 계산 스크립트, [기존 CPU 그래프 노트북](../notebooks/xla_fx_graphviz.ipynb) | FLOPs·Byte·산술 강도·가정. CPU나 시뮬레이터 시간은 해당 환경의 결과로 표시 |

기존 GPU·TPU 노트북은 컴파일 경로를 탐색하는 자료다. 현재 shape·자료형·역전파 셀을 그대로 실행해 이번 주의 벤치마크로 사용하지 않는다. GPU에서는 추론만 남기고, TPU에서는 Pallas의 `interpret` 대체 실행 여부를 확인한다. 해석 모드 결과를 네이티브 Pallas 커널 성능으로 기록하지 않는다.

Core Ultra 9 288V NPU를 지정해 쓸 수 있는 무료 원격 서비스는 이번 조사에서 확인하지 못했다. NPU 장비가 없다면 문헌 조사와 계산으로 참여하면 된다.

## 6. 모임에서 확인할 질문

- H100의 sparse BF16 수치, v5e의 BF16 수치, 288V NPU의 INT8 TOPS를 한 줄로 정렬하면 무엇이 빠지는가?
- GPU의 Tensor Core와 TPU의 TensorCore를 같은 크기의 구성 단위로 세어도 되는가?
- 배치 128의 시간이 배치 1보다 길어도 samples/s는 높아질 수 있는가?
- 출력 dtype, 실제 누산, 출력 직전 반올림을 각각 확인했는가?
- 장치 내부 시간이 빨라도 전체 요청 시간이 느려지는 경로는 무엇인가?
- 같은 가중치를 반복 측정할 때 캐시 상태가 Roofline의 이동량 가정을 바꾸는가?
- 어떤 추가 측정이 있어야 제품의 우열을 말할 수 있는가?
