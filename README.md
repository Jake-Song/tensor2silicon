# tensor2silicon

텐서 연산이 Python 코드에서 출발해 실리콘 위에서 실행되기까지의 흐름을 정리한 학습 노트.

## 모델 → 칩 5단계

| # | 문서 | 질문 |
|---|---|---|
| ① | [모델 → 연산](docs/01-model-to-ops.md) | Transformer를 펼치면 어떤 연산과 텐서가 나올까? |
| ② | [연산 → 그래프·IR](docs/02-ops-to-graph-ir.md) | Python 코드가 어떻게 컴파일러가 다룰 표현으로 바뀔까? |
| ③ | [IR → 실행 코드](docs/03-ir-to-executable.md) | 컴파일러는 무엇을 결정하고 하드웨어별로 무엇이 달라질까? |
| ④ | [실행 요청 → 커널 실행](docs/04-dispatch-to-kernel.md) | Python 함수가 반환되면 가속기 계산도 끝난 걸까? 커널은 어떻게 실행할까? |
| ⑤ | [커널 → 칩 내부](docs/05-kernel-to-chip.md) | 연산은 어디서 이루어지고 데이터는 어디에서 이동해올까? |

## 예제로 따라가기

- [Y = ReLU(XW + b)가 코드에서 커널까지 내려가는 길](docs/example-relu-linear.md) — JAX와 PyTorch 양쪽의 실제 IR 출력을 단계별로 대조
- [실제 GPU: PyTorch 2 on NVIDIA A100](docs/example-pytorch-gpu-a100.md) — Colab A100에서 Dynamo → AOT → Inductor → Triton IR → PTX(sm_80) → SASS까지 실물 출력, f32/TF32/bf16 Tensor Core 비교, T4와의 차이
- [실제 TPU: JAX on TPU v5e](docs/example-jax-tpu-v5e.md) — Colab v5e에서 jaxpr → StableHLO → 타일 레이아웃·VMEM·DMA가 드러난 HLO까지 실물 출력

## Colab 노트북

- [`notebooks/pytorch_gpu_relu_linear.ipynb`](notebooks/pytorch_gpu_relu_linear.ipynb) — GPU 런타임. Dynamo/AOT/Inductor 로그, Triton 캐시의 TTIR·TTGIR·PTX·SASS, 비동기 디스패치, 직접 쓴 Triton 커널
- [`notebooks/jax_tpu_relu_linear.ipynb`](notebooks/jax_tpu_relu_linear.ipynb) — TPU 런타임. jaxpr, StableHLO, 타일 레이아웃·VMEM·DMA가 보이는 HLO, cost/memory analysis, 2048² 타일링, Pallas 커널
- 터미널에서: `colab new -s gpu --gpu A100 && colab exec -s gpu -f notebooks/pytorch_gpu_relu_linear.ipynb --timeout 900 && colab stop -s gpu` (TPU는 `--tpu v5e1`)

## 스택별 상세 파이프라인

- [JAX 코드가 TPU 실행 코드로 변환되는 흐름](docs/jax-to-tpu.md)
- [PyTorch 코드가 torch.compile로 컴파일되는 흐름](docs/pytorch-compile.md)

## 원문 자료

- Scaling Book: [2장 TPU](https://jax-ml.github.io/scaling-book/tpus/), [4장 Transformer Math](https://jax-ml.github.io/scaling-book/transformers/), [9장 Profiling](https://jax-ml.github.io/scaling-book/profiling/), [12장 GPU](https://jax-ml.github.io/scaling-book/gpus/)
- [PyTorch torch.compiler 개요](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [OpenXLA: XLA architecture](https://openxla.org/xla/architecture)
- [JAX: Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
