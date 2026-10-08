# ④ 같은 모델에서 장치 실행까지

[5주차 목차](./README.md) · 이전 → [③ 사양](./03-specifications.md) · 다음 → [⑤ 측정](./05-measurement.md)

모델을 프레임워크에 넘기는 것과 직접 커널을 작성하는 것은 서로 다른 수준의 작업이다. 같은 수학식도 컴파일러·라이브러리·입력 조건에 따라 여러 커널, 자료형 변환, 서로 다른 장치 실행으로 나뉠 수 있다.

## 1. 세 실행 경로

| 대상 | 모델 수준의 경로 | 직접 커널을 다룰 때 | 지원되지 않을 때 확인할 곳 |
|---|---|---|---|
| NVIDIA H100 SXM | 프레임워크 그래프 → GPU 라이브러리 또는 생성된 커널 → CUDA 실행 | CUDA에서는 thread/block·메모리·동기화, Triton에서는 블록 단위 계산·타일·입출력 경계를 표현 | 프레임워크의 실제 실행 로그, 생성 코드, CUDA 대상 아키텍처, 라이브러리·Triton의 연산과 dtype 조건 |
| Google TPU v5e | JAX 함수 → 추적·jaxpr → XLA 컴파일 → TPU 실행 | Pallas로 블록·메모리 접근·커널 계산을 작성하고 TPU backend로 컴파일 | JAX/XLA lowering 오류, shape·dtype, Pallas TPU 지원 범위, jax/jaxlib/libtpu 조합 |
| Intel 288V NPU | 모델 변환·읽기 → OpenVINO 그래프 → NPU plugin·compiler → 드라이버·NPU | 이 과제의 경로는 모델 컴파일 API. CUDA/Triton 커널을 NPU에 그대로 넘기는 경로가 아님 | 장치 인식, `query_model`, NPU 대상 컴파일 로그, 실행·출력 검증 |

근거: [CUDA Programming Model][cuda], [Triton MatMul][triton], [JAX JIT][jax-jit], [Pallas TPU MatMul][pallas], [OpenVINO NPU][npu], [OpenVINO Core API][ov-core].

커널을 직접 작성하면 타일 재사용과 fusion을 제어할 여지가 생기지만, 프레임워크의 기존 최적화보다 항상 빠른 것은 아니다. `matmul → add → relu` 그래프가 있다고 해서 하나의 fused 커널이 생성되었다고 가정하지 않는다. 실행 파일이나 프로파일에서 확인한다.

## 2. NVIDIA: host와 device의 역할

호스트는 메모리를 준비하고 GPU 작업을 제출한다. CUDA 커널은 thread block들로 실행된다. 라이브러리에 MatMul을 맡길 수도 있고 Triton으로 행렬곱과 bias·ReLU를 함께 계산할 수도 있다. Triton의 공식 MatMul 예제는 FP32 accumulator를 사용하지만 입력·출력 dtype을 포함한 나머지 조건도 읽어야 한다. [CUDA][cuda], [Triton][triton]

이번 strict 커널 조건에서는 X·W를 BF16, 누산·bias·ReLU·Y를 FP32로 유지한다. MatMul 결과를 BF16으로 먼저 저장한 뒤 FP32 bias를 더하면 **중간 반올림이 추가되어 다른 계약**이다. 직접 작성한 커널도 출력 저장 전 dtype 변환을 확인한다.

GPU 커널 제출은 비동기일 수 있다. 호스트 함수가 돌아온 시점만 재면 실행 완료 시간이 아니다. 장치 event 또는 명시적인 완료 대기가 필요하다. [CUDA: Asynchronous Execution][cuda-async]

## 3. TPU: JIT 컴파일과 정밀도를 구분한다

`jax.jit`의 첫 실행에는 추적·컴파일이 포함될 수 있고, 같은 조건의 다음 호출은 컴파일된 코드를 재사용한다. shape·dtype 등이 바뀌면 별도 컴파일이 필요할 수 있으므로 배치 1과 128을 각각 준비한다. `block_until_ready()`로 완료를 기다리는 시간에는 호스트 dispatch도 포함될 수 있다. 이를 장치 profiler의 커널 시간과 동일시하지 않는다. [JAX JIT][jax-jit]

JAX의 dot 연산은 `precision`과 `preferred_element_type`을 구분한다. 후자는 출력 dtype을 지정하며, 알고리즘을 별도로 지정하지 않은 경우 누산 dtype에 대한 **컴파일러 힌트**이기도 하다. FP32 출력을 요청했다는 사실만으로 모든 곱셈·누산 조건을 입증하지 않는다. BF16 입력과 FP32 출력 설정, 컴파일 산출물, 오차를 함께 확인한다. [JAX dot API][jax-dot]

Pallas는 이보다 낮은 수준에서 블록과 커널을 표현한다. TPU 타일의 배수 제약·padding·경계 처리를 확인해야 하므로, 배치 128용 커널을 배치 1에 그대로 적용하지 않는다. [Pallas TPU MatMul][pallas]

## 4. Intel NPU: 입력 dtype만으로 지원을 판정하지 않는다

OpenVINO 2026 NPU 문서는 F32/F16 표현과 양자화 U8 경로를 열거하고, 비양자화 HW 계산 정밀도를 FP16으로 설명한다. 양자화 모델은 INT8 또는 혼합 FP16–INT8일 수 있다. 이 설명에서 **BF16 곱셈·FP32 누산 지원을 추론할 수는 없다**. [Supported Inference Data Types][npu]

NPU 문서의 현재 2026판에서 동적 shape는 **preview**다. 조건은 다음과 같다. [Dynamic Shapes / Limitations][npu]

- NPU40XX, 즉 Lunar Lake 세대 이후이며, 각 동적 차원에 상한이 있어야 한다.
- Windows NPU driver **32.0.100.4621 이상** 또는 Linux NPU driver **1.30 이상**이 필요하다.
- **Compiler-In-Plugin** 경로만 지원한다. 문서의 검증 범위는 vision 모델 중심이다.

이 조건을 만족한다고 이번 MatMul 그래프의 동적 shape 실행까지 보장되지는 않는다. 공통 실습에서는 M=1과 M=128의 **별도 정적 그래프**를 사용한다. 2026.0에는 plugin 내 컴파일러가 preview로 들어왔고, 2026.1부터 우선 선택되므로 **패치 버전과 실제 compiler type도 기록**한다. [NPU 도입부 / compiler_type][npu]

지원 확인 절차는 아래와 같다. [Core API의 available_devices / query_model / compile_model][ov-core]

1. 장치 목록에 NPU가 보이는지 확인한다.
2. 고정 shape의 MatMul·bias broadcast·ReLU 모델을 만들고 `core.query_model(model, "NPU")`의 지원 연산을 확인한다.
3. `core.compile_model(model, "NPU")`로 대상 장치를 명시한다. 질의 결과만으로 성공을 확정하지 않는다.
4. 실제 입력으로 실행하고 출력·오차·실행 장치를 확인한다. 지원 실패나 컴파일 오류도 결과로 남긴다.

`AUTO`로 성공한 실행을 NPU 실행의 증거로 삼지 않는다. CPU·GPU 혼합 실행을 선택했다면 어느 텐서 연산이 어디서 수행됐는지와 전송 비용을 기록하고, 시스템 비교로 분류한다. 호스트 CPU가 작업을 제출하는 것과 모델 연산이 CPU로 넘어가는 것은 다르다.

## 5. 문서 버전과 실행 버전 기록

| 대상 | 이번에 읽은 문서의 범위 | 실행 결과에 반드시 추가할 버전 |
|---|---|---|
| H100 | CUDA Programming Guide 온라인판, Hopper Tuning Guide **13.4**, Triton **main**, 2026-10-08 열람 | OS, NVIDIA driver, CUDA runtime/toolkit, 프레임워크, GPU 라이브러리, Triton. 전력 제한·MIG도 기록 |
| TPU v5e | JAX/Pallas **latest**, v5e 문서, 2026-10-08 열람 | JAX, jaxlib, libtpu, TPU VM image/runtime, compiler flags, 실제 device kind·사용 수 |
| 288V NPU | OpenVINO **2026** 온라인판, 2026-10-08 열람 | OpenVINO 전체 버전·build, NPU plugin/compiler 버전·type, NPU driver, OS·기기 모델·전력 모드 |

**세 장치의 실제 설치 버전은 모두 이번 조사에서 미확인이다.** 위 표는 설치 완료나 호환성 검증을 주장하지 않는다. `latest`, `main`, `2026`이라는 문서 경로만 적은 실행 보고서는 재현에 부족하다. 실행했다면 정확한 버전 출력과 코드를 함께 남긴다.

[cuda]: https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html
[cuda-async]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
[triton]: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
[jax-jit]: https://docs.jax.dev/en/latest/jit-compilation.html
[jax-dot]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot.html
[pallas]: https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
[npu]: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
[ov-core]: https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.Core.html
