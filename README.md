# tensor2silicon

Python 텐서 연산이 그래프·IR·실행 코드로 바뀌는 과정을 배우는 학습 프로젝트입니다.
로컬 웹에서 컴파일 과정을 실험하고 GPU/TPU 구조를 탐색할 수 있습니다.

## 빠른 시작

Python 3.13 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.
C 코드 실행에는 GCC가 필요하며, NumPy는 `uv run`이 자동으로 설치합니다.

프로젝트 디렉터리에서 실행하세요.

```bash
uv run python -m toy.web
```

브라우저에서 **http://127.0.0.1:8000**을 엽니다.
포트가 사용 중이면 `--port 8001`을 붙이고, 종료하려면 `Ctrl+C`를 누릅니다.
GCC가 없어도 그래프와 Python 결과는 볼 수 있습니다.

## 웹 사용법

| 화면 | 사용 방법 |
|---|---|
| 컴파일러 실험실 | 예제와 M/K/N, fusion, tiling을 선택하고 **실행하기**를 누릅니다. Python → 그래프 → 최적화 → 코드 생성 → 검증 순서로 확인합니다. |
| GPU / TPU 구조 | 상단 탭에서 GPU·TPU 구조도를 열고 블록을 눌러 내부 구조와 설명을 확인합니다. |
| GPU 실행 | 아래 sim-gpu 설정 후 **컴파일 & 실행**을 눌러 assembly, warp timeline, NumPy와의 결과 비교를 확인합니다. |

그래프 노드를 선택하면 shape·dtype·입력 관계를 볼 수 있고, 생성된 NumPy/C 코드는 복사할 수 있습니다.

### sim-gpu 연결 (선택)

float32 ISA를 지원하는 `sim-gpu` 저장소가 이 프로젝트와 나란히 있어야 합니다.

```bash
uv run --with-editable ../sim-gpu python -m toy.web --port 8010
```

**http://127.0.0.1:8010/simulator**에서 실행 대상을 **sim-gpu · SIMT**로 선택합니다.
예제·shape·fusion을 바꾸며 커널과 실행 흐름을 비교하세요.
시뮬레이터 통계는 교육용이며 실제 GPU 성능을 나타내지 않습니다.

## 터미널과 Python에서 사용하기

```bash
uv run -m toy        # 추적 → 최적화 → C 생성·실행·검증 데모
uv run -m toy.bench  # CPU 루프 순서·타일링 성능 비교
```

`toy.jit`으로 함수를 C로 컴파일하거나, `symbolic_trace`로 그래프와 NumPy 코드를 확인합니다.
아래 코드는 프로젝트 환경에서 `uv run python 파일명.py`로 실행할 수 있습니다.

```python
import numpy as np
import toy

@toy.jit
def forward(x, w, b):
    return toy.relu(x @ w + b)

x = np.ones((16, 8), dtype=np.float32)
w = np.ones((8, 4), dtype=np.float32)
b = np.zeros((4,), dtype=np.float32)
print(forward(x, w, b))

gm = toy.symbolic_trace(forward.fn, x, w, b)
print(gm.graph)
print(gm.code)
```

`@toy.jit(tile=(64, 256, 32))`로 CPU 타일 크기를 지정하거나,
sim-gpu 설치 후 `@toy.jit(backend="simgpu")`로 시뮬레이터를 사용할 수 있습니다.
지원 연산은 `matmul`, `add`, `relu`, `broadcast_in_dim`이며 텐서 값에 의존하는 분기는 지원하지 않습니다.

## CPU에서 XLA / PyTorch 그래프 비교

[XLA HLO / PyTorch FX 시각화 노트북](notebooks/xla_fx_graphviz.ipynb)에서 작은 `ReLU(XW + b)` 예제를 실행하고 실제 그래프를 Graphviz로 그립니다.
GPU·TPU 없이 실행할 수 있으며, 패키지 설치 안내와 실행 결과, DOT·SVG 저장 코드가 포함되어 있습니다.
순전파 비교에 이어 PyTorch AOTAutograd의 조인트 그래프와 분리된 순전파·역전파 FX 그래프, JAX의 손실·gradient 통합 HLO를 그리고 미분 결과를 검증합니다.

## Colab에서 실제 GPU / TPU 실행

노트북을 Colab에서 열고 해당 가속기 런타임을 선택한 뒤 셀을 순서대로 실행하세요.

- [PyTorch GPU 노트북](notebooks/pytorch_gpu_relu_linear.ipynb): Dynamo → Inductor → Triton → PTX·SASS 확인
- [JAX TPU 노트북](notebooks/jax_tpu_relu_linear.ipynb): jaxpr → StableHLO → HLO와 Pallas 커널 확인

## 학습 문서

순서대로 읽기:
[① 모델 → 연산](docs/01-model-to-ops.md) →
[② 그래프·IR](docs/02-ops-to-graph-ir.md) →
[③ 실행 코드](docs/03-ir-to-executable.md) →
[④ 커널 실행](docs/04-dispatch-to-kernel.md) →
[⑤ 칩 내부](docs/05-kernel-to-chip.md)

- [ReLU(XW + b) 단계별 예제](docs/example-relu-linear.md)
- [PyTorch / A100 실행 예제](docs/example-pytorch-gpu-a100.md) · [JAX / TPU v5e 실행 예제](docs/example-jax-tpu-v5e.md)
- [PyTorch 컴파일 파이프라인](docs/pytorch-compile.md) · [JAX → TPU 파이프라인](docs/jax-to-tpu.md)

## 테스트

```bash
uv run python -m unittest discover -s tests -v

# sim-gpu 연동 테스트 포함
uv run --with-editable ../sim-gpu python -m unittest discover -s tests -v
```

sim-gpu가 없으면 시뮬레이터 전용 테스트는 건너뜁니다.
