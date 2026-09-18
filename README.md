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

## 직접 만들어보기: toy 컴파일러

같은 식 `Y = ReLU(XW + b)`를 처리하는 아주 작은 컴파일러를 [`toy/`](toy/)에 만들었다. 위 5단계 중 ②~③을 손으로 구현한 것이다.

```python
@toy.jit
def f(x, w, b):
    return toy.relu(x @ w + b)

y = f(x, w, b)   # 첫 호출: 추적 → 퓨전 → C 컴파일. 같은 shape이면 캐시 히트

@toy.jit(tile=(64, 256, 32))   # matmul 스케줄: 블록 크기. tile="auto"면 후보를 실측해 선택
def g(x, w, b):
    return toy.relu(x @ w + b)
```

| 파일 | 역할 | 대응되는 실제 단계 |
|---|---|---|
| [`toy/trace.py`](toy/trace.py) | `Tensor` 연산자 오버로딩으로 함수를 추적해 그래프 기록, `jit`은 shape별로 컴파일 결과 캐시 | `jax.jit` 추적 / TorchDynamo |
| [`toy/ir.py`](toy/ir.py) | 그래프 IR + shape 추론 (`matmul`, `add`, `relu`, `broadcast_in_dim`) | jaxpr / FX 그래프 |
| [`toy/interp.py`](toy/interp.py) | NumPy 레퍼런스 인터프리터. 모든 코드 생성 결과의 정답 기준 | eager 실행 |
| [`toy/passes.py`](toy/passes.py) | 퓨전 패스(소비자가 하나뿐인 원소별 연산과 matmul을 소비자 루프 안으로 흡수), `tile_matmuls` 스케줄 지정 | XLA fusion / Inductor 커널 스케줄링 |
| [`toy/codegen_c.py`](toy/codegen_c.py) | 노드마다 C 루프를 생성 → `gcc`로 빌드 → `ctypes`로 로드. `fusion` 노드는 루프 하나에 스칼라 문장으로 펼침. `tile`이 있는 matmul은 블록(ii,jj,kk) + i,k,j 순서로, 에필로그는 블록 reduction이 끝난 뒤 적용 | Inductor C++ / XLA CPU |
| [`toy/bench.py`](toy/bench.py) | 같은 그래프를 스케줄만 바꿔 NumPy(BLAS)와 비교. `uv run -m toy.bench` | Inductor max-autotune / XLA cost model |

```bash
uv run -m toy   # 추적된 IR → 퓨전 IR → 생성된 C → 검증 → shape별 재추적 → 그래프 브레이크 예시
```

i7-10700K(AVX2) 1코어에서 M=N=K=1024 기준(`uv run -m toy.bench`):

| 스케줄 | ms | GFLOP/s |
|---|---|---|
| NumPy sgemm(멀티스레드 BLAS) | 8.6 | 250 |
| naive i,j,k | 1327 | 1.6 |
| i,k,j 순서만 변경 | 132 | 16 |
| tile=(64,256,32) | 85 | 25 |

루프 순서만 바꿔도 10배, 블록화로 1.5배 더. 남은 10배는 레지스터 블로킹·FMA·멀티스레드 몫이다. `-O2`로는 타일 커널이 3.8 GFLOP/s에 그치고 `-O3 -march=native`여야 j 루프가 벡터화된다.

다음 단계 후보: 레지스터 블로킹(마이크로커널), Triton 타깃, autodiff, 상수·size-1 브로드캐스트 지원.

## 스택별 상세 파이프라인

- [JAX 코드가 TPU 실행 코드로 변환되는 흐름](docs/jax-to-tpu.md)
- [PyTorch 코드가 torch.compile로 컴파일되는 흐름](docs/pytorch-compile.md)

## 원문 자료

- Scaling Book: [2장 TPU](https://jax-ml.github.io/scaling-book/tpus/), [4장 Transformer Math](https://jax-ml.github.io/scaling-book/transformers/), [9장 Profiling](https://jax-ml.github.io/scaling-book/profiling/), [12장 GPU](https://jax-ml.github.io/scaling-book/gpus/)
- [PyTorch torch.compiler 개요](https://docs.pytorch.org/docs/stable/torch.compiler.html)
- [OpenXLA: XLA architecture](https://openxla.org/xla/architecture)
- [JAX: Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [Triton: Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
