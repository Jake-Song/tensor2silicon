# ④ 같은 모델에서 장치 실행까지

[5주차 목차](./README.md) · 이전 → [③ 사양](./03-specifications.md) · 다음 → [⑤ 측정](./05-measurement.md)

## 1. 세 실행 경로

| 대상 | 모델 수준의 경로 | 직접 커널을 다룰 때 | 
|---|---|---|
| NVIDIA H100 SXM | 프레임워크 그래프 → GPU 라이브러리 또는 생성된 커널 → CUDA 실행 | CUDA에서는 thread/block·메모리·동기화, Triton에서는 블록 단위 계산·타일·입출력 경계를 표현 | 프레임워크의 실제 실행 로그, 생성 코드, CUDA 대상 아키텍처, 라이브러리·Triton의 연산과 dtype 조건 |
| Google TPU v5e | JAX 함수 → 추적·jaxpr → XLA 컴파일 → TPU 실행 | Pallas로 블록·메모리 접근·커널 계산을 작성하고 TPU backend로 컴파일 | JAX/XLA lowering 오류, shape·dtype, Pallas TPU 지원 범위, jax/jaxlib/libtpu 조합 |
| Intel 288V NPU | 모델 변환·읽기 → OpenVINO 그래프 → NPU plugin·compiler → 드라이버·NPU | 경로는 모델 컴파일 API. CUDA/Triton 커널을 NPU에 그대로 넘기는 경로가 아님 | 장치 인식, `query_model`, NPU 대상 컴파일 로그, 실행·출력 검증 |

근거: [CUDA Programming Model][cuda], [Triton MatMul][triton], [JAX JIT][jax-jit], [Pallas TPU MatMul][pallas], [OpenVINO NPU][npu], [OpenVINO Core API][ov-core].

커널을 직접 작성하면 타일 재사용과 fusion을 제어할 여지가 생기지만, 프레임워크의 기존 최적화보다 항상 빠른 것은 아니다. `matmul → add → relu` 그래프가 있다고 해서 하나의 fused 커널이 올바르게 생성되었다고 볼 수 없다. 실제로는 실행 파일이나 프로파일에서 확인해야 한다.

## 2. NVIDIA: host와 device의 역할

호스트는 메모리를 준비하고 GPU 작업을 제출한다. CUDA 커널은 thread block들로 실행된다. 라이브러리에 MatMul을 맡길 수도 있고 Triton으로 행렬곱과 bias·ReLU를 함께 계산할 수도 있다. [CUDA][cuda], [Triton][triton]

이번 strict 커널 조건에서는 X·W를 BF16, 누산·bias·ReLU·Y를 FP32로 유지한다. MatMul 결과를 BF16으로 먼저 저장한 뒤 FP32 bias를 더하면 **중간 반올림이 추가된다.** 

GPU 커널 제출은 비동기일 수 있다. 호스트 함수가 돌아온 시점만 재면 실행 완료 시간이 아니다. 장치 event 또는 명시적인 완료 대기가 필요하다. [CUDA: Asynchronous Execution][cuda-async]

> **참고:** FP32 결과를 BF16으로 변환해 저장할 때 BF16으로 표현 가능한 값으로 반올림되며, 이후 FP32로 변환해도 버려진 정밀도는 복원되지 않는다.

## 3. TPU: JIT 컴파일과 정밀도를 구분한다

`jax.jit`의 첫 실행에는 추적·컴파일이 포함될 수 있고, 같은 조건의 다음 호출은 컴파일된 코드를 재사용한다. [JAX JIT][jax-jit]

JAX의 dot 연산은 `precision`과 `preferred_element_type`을 구분한다. 후자는 출력 dtype을 지정하며, 알고리즘을 별도로 지정하지 않은 경우 누산 dtype에 대한 **컴파일러 힌트**이기도 하다. BF16 입력과 FP32 출력 설정, 컴파일 산출물, 오차를 함께 확인한다. [JAX dot API][jax-dot]

Pallas는 이보다 낮은 수준에서 블록과 커널을 표현한다. TPU 타일의 배수 제약·padding·경계 처리를 확인해야 하므로, 배치 128용 커널을 배치 1에 그대로 적용하지 않는다. [Pallas TPU MatMul][pallas]

[cuda]: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
[cuda-async]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
[triton]: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
[jax-jit]: https://docs.jax.dev/en/latest/jit-compilation.html
[jax-dot]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot.html
[pallas]: https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
[npu]: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html

