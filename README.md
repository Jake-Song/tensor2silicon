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

## 추론 성능: 무엇이 병목인가

- [⑥ 추론과 Roofline](docs/06-inference-roofline.md) — prefill은 compute-bound, decode는 memory-bound. 각 구간에서 실제로 먹히는 최적화와, 잘못 짝지으면 손해인 조합

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
| [`toy/graph_module.py`](toy/graph_module.py) | 추출한 그래프를 `GraphModule`로 감싸고 실행 가능한 NumPy Python 코드를 생성 | FX `symbolic_trace` / `GraphModule` |
| [`toy/ir.py`](toy/ir.py) | 그래프 IR + shape 추론 (`matmul`, `add`, `relu`, `broadcast_in_dim`) | jaxpr / FX 그래프 |
| [`toy/interp.py`](toy/interp.py) | NumPy 레퍼런스 인터프리터. 모든 코드 생성 결과의 정답 기준 | eager 실행 |
| [`toy/passes.py`](toy/passes.py) | 퓨전 패스(소비자가 하나뿐인 원소별 연산과 matmul을 소비자 루프 안으로 흡수), `tile_matmuls` 스케줄 지정 | XLA fusion / Inductor 커널 스케줄링 |
| [`toy/codegen_c.py`](toy/codegen_c.py) | 노드마다 C 루프를 생성 → `gcc`로 빌드 → `ctypes`로 로드. `fusion` 노드는 루프 하나에 스칼라 문장으로 펼침. `tile`이 있는 matmul은 블록(ii,jj,kk) + i,k,j 순서로, 에필로그는 블록 reduction이 끝난 뒤 적용 | Inductor C++ / XLA CPU |
| [`toy/bench.py`](toy/bench.py) | 같은 그래프를 스케줄만 바꿔 NumPy(BLAS)와 비교. `uv run -m toy.bench` | Inductor max-autotune / XLA cost model |

```bash
uv run -m toy   # 추적된 IR → 퓨전 IR → 생성된 C → 검증 → shape별 재추적 → 그래프 브레이크 예시
```

### FX 스타일로 그래프 추출하기

```python
import numpy as np
import toy

def forward(x, w, b):
    return toy.relu(x @ w + b)

# shape 튜플 대신 실제 배열을 넣어도 된다. 추적 중 텐서 계산은 하지 않는다.
gm = toy.symbolic_trace(forward, (16, 8), (8, 4), (4,))
print(gm.graph)            # shape과 dtype이 붙은 기존 IR
gm.graph.print_tabular()   # 입력 → 연산 → 출력, 노드 의존 관계
print(gm.code)             # 독립 실행 가능한 NumPy forward 소스

rng = np.random.default_rng(0)
x = rng.standard_normal((16, 8), dtype=np.float32)
w = rng.standard_normal((8, 4), dtype=np.float32)
b = rng.standard_normal((4,), dtype=np.float32)
y = gm(x, w, b)            # 생성된 Python을 실행. 원래 forward는 다시 호출하지 않는다.
np.testing.assert_allclose(y, forward(x, w, b))

# 추출한 그래프를 기존 최적화·C 컴파일 파이프라인에 그대로 연결한다.
compiled = toy.compile_graph(toy.fuse(gm.graph))
np.testing.assert_allclose(compiled(x, w, b)[0], y, rtol=1e-5, atol=1e-6)
```

PyTorch FX에서 착안한 작은 인터페이스이며, PyTorch 설치는 필요 없다. 기존
`toy.trace`처럼 입력 shape이 필요하고 `matmul`, `add`, `relu`, 명시적 broadcast만
지원한다. 입력은 위치 인자로 전달하며 실행 시 float32로 변환하고 shape을 검사한다.
shape이 바뀌면 다시 추출해야 한다. 반환값은 텐서 또는 평평한 튜플이며, 원래 함수의
단일 텐서·한 원소 튜플·빈 튜플 구분을 유지한다.

Python 함수는 추적할 때 한 번 실행되므로 `print` 등의 부수 효과도 그때 발생한다.
텐서 값에 의존하는 분기, 텐서 반복, 상수 피연산자, 중첩 반환 구조는 지원하지 않는다.
`gm.graph`는 검사와 컴파일 패스 입력용이며, 직접 수정해도 이미 생성된 `gm.code`와
실행 함수는 바뀌지 않는다. 생성된 Python은 퓨전 전 그래프만 지원한다.

```bash
uv run python -m unittest discover -s tests -v
```

### sim-gpu: 그래프에서 GPU 실행까지

CPU C backend에 더해, 같은 float32 그래프를 `sim-gpu`의 작은 SIMT ISA로
컴파일하고 실제 시뮬레이터에서 실행할 수 있다. float32 ISA가 추가된 simulator
checkout이 필요하며 C backend만 사용할 때는 설치하지 않아도 된다.

두 프로젝트 디렉터리가 나란히 있는 구성에서는 tensor2silicon에서 실행한다:

```sh
uv run --with-editable ../sim-gpu python -m toy.web --port 8010
```

브라우저에서 `http://127.0.0.1:8010`을 열고 실행 대상을 **sim-gpu · SIMT**로
선택한다. 예제/shape/fusion을 바꾼 뒤 **컴파일 & 실행**을 누르면 다음을 확인한다:

- Python 모델, 추적 그래프, fusion 이후 그래프, 생성된 NumPy/C 코드
- NumPy reference와 simulator 출력 비교 및 최대 절대 오차
- 커널별 assembly, word 단위 메모리 배치, launch 구성, simulated cycles
- graph node → kernel 선택, assembly 명령어 → warp timeline 강조, active-lane mask

Linear + ReLU는 fusion 없음/epilogue/full에서 각각 4/2/1개 커널이 된다.
브로드캐스트도 fusion이 없으면 별도 커널이며, fusion 안에서는 인덱스 변환으로
처리된다. 다른 연산이 함께 사용하는 중간값은 글로벌 메모리 버퍼로 남는다.

Python에서 직접 사용하기:

```python
import numpy as np
import toy
from simgpu import GPUConfig

@toy.jit(backend="simgpu")
def forward(x, w, b):
    return toy.relu(x @ w + b)

rng = np.random.default_rng(0)
x = rng.standard_normal((3, 5), dtype=np.float32)
w = rng.standard_normal((5, 4), dtype=np.float32)
b = rng.standard_normal((4,), dtype=np.float32)
y = forward(x, w, b)                         # 같은 shape/dtype은 컴파일 캐시 재사용
np.testing.assert_allclose(y, np.maximum(x @ w + b, 0), rtol=1e-4, atol=1e-4)

gm = toy.symbolic_trace(forward.fn, x, w, b)
compiled = toy.compile_simgpu(toy.fuse(gm.graph), config=GPUConfig(block_size=8))
execution = compiled.run(x, w, b, record=True)
print(execution.stats)                        # 순차 launch들의 합계
print(compiled.kernels[0].source)             # 개별 커널의 실행 가능한 assembly
print(compiled.layout)                        # 버퍼 주소는 byte가 아니라 word 단위
report = execution.kernels[0].report          # JSON 호환 timeline/통계/명령어 표
```

`compiled(*args)`는 C compiled object처럼 배열 리스트를 반환한다.
`.run(..., record=True)`는 `outputs`, `kernels`, `stats`를 가진 `SimulationRun`을
반환한다. 각 `KernelRun`은 `kernel`, `stats`, `report`를 포함한다. 기록을 끄면
`report`는 `None`이다. `kernel.source_map[pc]`는 웹 그래프와 동일한 node ID이며,
fusion 내부 연산까지 구분한다. `compiled.source`는 검사하기 위한 전체 커널
소스의 연결본이다. 커널마다 labels/grid가 별개이므로 실행하려면 개별
`kernel.source`와 `kernel.config`를 사용한다.

`GPUConfig`의 block/warp/SM/메모리 지연 설정을 사용하며, grid와 최소 글로벌
메모리 용량은 그래프에서 계산한다. JIT에는 `sim_config=GPUConfig(...)`로 전달한다.
매 실행은 새 메모리에서 시작하고, 그 실행 안의 커널들만 중간 버퍼를 공유한다.

지원 범위와 제한:

- `matmul`(2-D), `add`, `relu`, 명시적 broadcast, fusion, 다중 출력, float32.
  입력은 C-contiguous float32로 변환하고 shape을 검사한다.
- 출력 원소 하나당 스레드 하나; matmul의 K reduction은 스레드 안의 순차 루프다.
  `fmul` 뒤 `fadd`를 실행하며 fused multiply-add는 사용하지 않는다.
- CPU `tile`과 `tile="auto"`는 지원하지 않는다. 16개 레지스터를 초과하는
  복잡한 fusion은 명시적인 compile error이며, 메모리 spilling은 구현하지 않았다.
- 브라우저는 M/K/N ≤ 16, kernel당 50,000 cycles, 요청당 100,000 samples,
  worker 실행 30초로 제한한다. Python API는 GPUConfig의 cycle 한도를 사용하고,
  기록 시 기본 `max_samples=100_000`을 적용한다.
- 통계는 교육용 simulator 측정값이다. 실제 GPU 시간, Tensor Core, shared-memory
  tiling 또는 launch overhead를 모델링하지 않는다. 각 커널의 timeline은 cycle 0부터 시작한다.

검증:

```sh
uv run --with-editable ../sim-gpu python -m unittest discover -s tests -v
# sim-gpu 디렉터리에서는 uv run pytest
```

simulator 없이 실행하는 기존 테스트는 simulator 전용 사례만 건너뛴다.

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
