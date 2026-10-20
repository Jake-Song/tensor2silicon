# ⑥ 같은 연산의 비교 조건표와 손계산

[5주차 목차](./README.md) · 이전 → [⑤ 측정](./05-measurement.md) · [제출용 양식](./comparison-template.md)

$$
Y=\operatorname{ReLU}(XW+b),\quad
X\in\mathbb{R}^{M\times1024},\quad
W\in\mathbb{R}^{1024\times1024},\quad
b\in\mathbb{R}^{1024},\quad M\in\{1,128\}
$$

b는 모든 행에 broadcast한다. 추론만 수행하며 학습·gradient·optimizer는 없다. 아래의 **설계 조건**, **문헌 확인**, **계산**, **아직 미측정**을 구분한다.

## 1. 먼저 비교 계약을 둘로 나눈다

| 조건 | A. 같은 연산 조건의 커널 비교 | B. 같은 품질 목표의 모델·시스템 비교 |
|---|---|---|
| 식·shape | 위 식, M=1과 128을 별도로 실행 | 동일 |
| 원본 데이터 | 모든 장치에 동일한 값·layout | 동일한 FP32 원본·동일한 평가 입력 |
| 입력·가중치 | X·W를 **한 번 BF16으로 반올림한 동일한 값** | FP32 원본에서 각 경로로 변환. 변환 방법 기록 |
| 계산·누산 | BF16 곱셈, FP32 누산 | 각 장치의 지원 경로 허용. dtype·양자화·누산 공개 |
| bias·ReLU·출력 | FP32. MatMul 뒤 저정밀도 저장을 끼우지 않음 | FP32로 출력 회수·비교. 내부 변환과 반올림 공개 |

## 2. 제품별 비교 조건표

제품 사양 자체는 [③](./03-specifications.md)에 모았다. 아래의 “후보”는 실행 계획이며 측정 결과가 아니다.

| 항목 | H100 SXM / Hopper / GPU 1개 | TPU v5e / 칩 1개 | 
|---|---|---|
| 소프트웨어 경로 | 프레임워크·라이브러리 또는 CUDA/Triton | JAX → XLA; 선택적으로 Pallas TPU | OpenVINO 2026 → NPU plugin/compiler → driver |
| 읽은 문서·설치 버전 | CUDA 온라인판·Triton main. 실제 버전 미확인 | JAX/Pallas latest. 실제 jax/jaxlib/libtpu 미확인 | 2026 문서. 실제 build·compiler type/version·driver 미확인 |
| A 계약의 구현 후보 | BF16 입력·곱셈, FP32 누산·bias·Y. 생성 코드·오차 확인 필요 | BF16 입력·MXU, FP32 dot 출력·bias·Y. lowering·오차 확인 필요 | BF16/FP32 계약의 지원을 이번 조사에서 확인 못함. A 비교 결과 없음 |
| B 계약의 구현 후보 | BF16 또는 FP16 경로를 하나 선택하고 명시 | BF16 경로와 출력 변환을 명시 | 비양자화 FP16 HW 경로부터 조사. 실제 누산 미확인. INT8은 별도 실험 |
| 데이터 위치 | W·b와 상주 측정의 X·Y는 GPU 메모리 | W·b와 상주 측정의 X·Y는 해당 TPU HBM | 모델·버퍼는 시스템 메모리, 실행 중 온칩 버퍼 사용. 실제 할당·복사 확인 |


제품별 근거: [NVIDIA 제품 사양](https://www.nvidia.com/en-us/data-center/h100/), [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [Triton MatMul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html), [v5e 사양](https://docs.cloud.google.com/tpu/docs/v5e), [JAX JIT](https://docs.jax.dev/en/latest/jit-compilation.html), [Pallas MatMul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html) 버전·지원 해석은 [④](./04-software.md)를 따른다.
