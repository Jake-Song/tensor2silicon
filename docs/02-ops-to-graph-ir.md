# ② 연산 → 그래프·IR: Python 코드가 어떻게 컴파일러가 다룰 표현으로 바뀔까?

> **한 줄 답.** 컴파일러는 Python을 읽지 않는다. 프레임워크가 Python 함수를 **한 번 "추적"하거나 "해석"** 해서, 텐서 연산만 남긴 **SSA 형태의 그래프(IR)** 를 뽑아내고, 컴파일러는 그 그래프만 본다. JAX는 함수를 *Tracer로 실행*해서 뽑고, PyTorch 2는 *바이트코드를 해석*해서 뽑는다.

```
              JAX                                    PyTorch 2
   ─────────────────────────────          ────────────────────────────────
   Python 함수                             Python 함수
      │ jax.jit: Tracer로 실행                 │ torch.compile: Dynamo가
      │ (shape/dtype만 있는 추상값)             │ 바이트코드를 심볼릭 해석 (PEP 523)
      ▼                                       ▼
   jaxpr  ◀── grad / vmap 변환             FX 그래프 (torch.* op) + guard
      │ lowering 규칙                          │ AOT Autograd: joint fwd/bwd,
      ▼                                       │ ATen/Prim으로 분해, functionalize
   StableHLO (MLIR)                           ▼
      │                                    ATen FX 그래프 (forward / backward)
      ▼                                       ▼
   ── XLA로 넘김 ──                        ── Inductor로 넘김 ──
```

이전 단계 ← [① 모델 → 연산](./01-model-to-ops.md) · 다음 단계 → [③ IR → 실행 코드](./03-ir-to-executable.md)

---

## 1. 왜 그래프가 필요한가

[①](./01-model-to-ops.md)에서 본 대로 Transformer는 큰 matmul 몇 개와 그 사이의 elementwise 연산이다. Python이 이것을 한 줄씩 eager로 실행하면:

- 연산마다 커널 하나가 따로 런치되고, 중간 결과가 매번 HBM을 왕복한다.
- 컴파일러가 "이 exp와 이 mul을 합쳐도 된다", "이 버퍼는 재사용해도 된다"를 판단할 **전체 그림**이 없다.

그래서 프레임워크는 실행 전에 함수 전체를 **데이터플로 그래프**로 바꾼다. 그래프의 노드는 primitive 연산, 간선은 텐서(shape, dtype 포함)이고, 순수 함수형 SSA라서 최적화가 안전하다.

## 2. JAX: 함수를 실제로 한 번 실행해서 기록한다 (Tracing)

Scaling Book 9장은 TPU 소프트웨어 스택을 이렇게 요약한다. `jax.jit()`을 호출하면 JAX가 함수를 **추적(trace)** 하여 **StableHLO** 를 내보낸다. StableHLO는 "ML 계산을 위한 플랫폼 독립 IR"이고, 그다음부터는 XLA 컴파일러의 몫이다.

### 동작

1. `jax.jit(f)(x, w)`를 호출하면 실제 값 대신 **Tracer**(shape, dtype만 가진 추상 값)를 인자로 넣어 함수를 한 번 실행한다.
2. `x @ w`, `jnp.tanh` 같은 연산은 계산되지 않고 **primitive**(`dot_general`, `tanh`)로 기록된다.
3. 기록된 결과가 **jaxpr**이다. `grad`, `vmap`, `pmap` 같은 함수 변환은 모두 jaxpr 수준에서 이루어진다.
4. jaxpr의 각 primitive는 lowering 규칙에 따라 **StableHLO** (MLIR 방언)로 바뀐다.

```python
import jax, jax.numpy as jnp

def f(x, w):
    return jnp.tanh(x @ w)

print(jax.make_jaxpr(f)(jnp.ones((4, 8)), jnp.ones((8, 2))))
```

```
{ lambda ; a:f32[4,8] b:f32[8,2]. let
    c:f32[4,2] = dot_general[dimension_numbers=(([1], [0]), ([], []))] a b
    d:f32[4,2] = tanh c
  in (d,) }
```

`dot_general`의 `dimension_numbers`가 [①](./01-model-to-ops.md)에서 말한 contracting 축(`[1], [0]`)과 batching 축(`[], []`)이다. ①의 FLOPs 규칙이 IR에 그대로 적혀 있는 셈이다.

```python
print(jax.jit(f).lower(x, w).as_text())
```

```mlir
func.func public @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x2xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
  %1 = stablehlo.tanh %0
  return %1
}
```

### 추적 방식의 제약

함수를 "한 번 실행"해서 기록하므로:

- Python `if`/`for`는 추적 시점의 값으로 한 번만 펼쳐진다. 텐서 값에 따라 달라지는 분기는 `lax.cond`, `lax.scan`, `lax.while_loop`로 써야 한다.
- `print` 같은 부수 효과는 추적 시 한 번만 일어난다.
- 그래프는 입력 shape/dtype에 특수화된다. shape이 바뀌면 처음부터 재추적·재컴파일한다.

엄격하지만, 그 대가로 **그래프가 항상 하나의 완전한 프로그램**이라는 것이 보장된다.

## 3. PyTorch 2: 바이트코드를 해석해서 그래프를 뽑는다 (torch.compile)

PyTorch의 `torch.compile`은 "PyTorch 코드를 JIT 컴파일해 최적화된 커널로 만드는" 함수이고, 세 기술이 이어진 파이프라인이다.

| 컴포넌트 | 푸는 문제 | 출력 |
|---|---|---|
| **TorchDynamo** | Python에서 그래프를 *안전하게* 꺼내기 | FX 그래프 (torch.* 수준) + guard |
| **AOT Autograd** | backward를 *미리* 그래프로 만들기 | forward / backward ATen 그래프 |
| **TorchInductor** | 그래프를 빠른 커널로 만들기 ([③](./03-ir-to-executable.md)) | Triton (GPU) / C++·OpenMP (CPU) |

여기에 **PrimTorch** 가 있다. PyTorch의 2000개가 넘는 연산자를 약 250개의 primitive op(Prim IR)와 약 750개의 Core ATen op로 정규화해서, 백엔드가 구현해야 할 연산 집합을 줄이는 작업이다. AOT Autograd의 분해(decomposition)가 이 정규화된 집합을 목표로 한다.

### TorchDynamo

JAX와 가장 다른 지점이다. Dynamo는 함수를 실행하지 않고, **CPython의 프레임 평가 API(PEP 523)** 를 후킹해 함수의 **바이트코드**를 한 명령씩 심볼릭하게 해석한다.

- 텐서 연산을 만나면 **FX 그래프**에 노드로 기록한다.
- 텐서와 무관한 Python 로직(리스트, 딕셔너리, 속성 접근)은 그래프에 넣지 않고 Dynamo가 직접 처리한다.
- 추적 중에 가정한 조건(입력 dtype, shape, stride, `requires_grad`, 전역 변수 값 등)을 **guard**로 남긴다. 다음 호출 때 guard가 통과하면 캐시된 코드를, 실패하면 재컴파일한다.
- 처리할 수 없는 코드(`print`, 텐서 값 의존 분기, 미지원 라이브러리)를 만나면 **graph break**를 낸다. 앞부분은 컴파일, 그 지점은 eager, 뒷부분은 다시 컴파일한다. JAX라면 에러가 날 코드가 그냥 동작하는 이유다.
- 원래 바이트코드를 "컴파일된 그래프를 호출하는 새 바이트코드"로 교체해 코드 객체에 캐시한다.

```python
def forward(self, L_x_: "f32[16, 8]"):
    l_x_ = L_x_
    h: "f32[16, 4]" = self._modules['fc'](l_x_)
    gelu: "f32[16, 4]" = torch.nn.functional.gelu(h)
    mul: "f32[16, 4]" = gelu * 2
    return (mul,)
```

op가 아직 `torch.nn.functional.gelu` 같은 **사용자 수준**이라는 점에 주목한다.

### AOT Autograd

Dynamo의 그래프는 forward만 있고 op도 고수준이다. AOT(Ahead-Of-Time) Autograd는:

1. **FakeTensor**(메모리 없이 메타데이터만)로 forward를 실행하며 autograd 엔진을 추적해 forward + backward **조인트 그래프**를 만든다. eager에서는 backward가 실행 시점에 tape를 따라 결정되지만, 여기서는 컴파일 시점에 확정된다.
2. 복합 op를 **ATen 코어 op**로 **분해**(`gelu` → `mul, erf, add, mul`)하고, in-place 연산을 함수형으로 바꾼다(**functionalization**).
3. 조인트 그래프를 forward / backward로 **파티셔닝**하면서, 어떤 중간값을 저장하고 어떤 것을 재계산할지(min-cut) 정한다.

```python
# forward (ATen 수준, gelu가 분해됨)
def forward(self, primals_1, primals_2, primals_3):
    t = torch.ops.aten.t.default(primals_1)
    addmm = torch.ops.aten.addmm.default(primals_2, primals_3, t)
    mul = torch.ops.aten.mul.Tensor(addmm, 0.5)
    mul_1 = torch.ops.aten.mul.Tensor(addmm, 0.7071067811865476)
    erf = torch.ops.aten.erf.default(mul_1)
    add = torch.ops.aten.add.Tensor(erf, 1)
    mul_2 = torch.ops.aten.mul.Tensor(mul, add)
    mul_3 = torch.ops.aten.mul.Tensor(mul_2, 2)
    return (mul_3, primals_3, addmm)   # addmm은 backward용으로 저장
```

이 시점의 그래프가 jaxpr/StableHLO와 같은 역할, 즉 **백엔드 컴파일러가 받는 입력**이다.

## 4. 같은 연산, 두 스택의 IR 계층

| 수준 | JAX | PyTorch 2 | 비고 |
|---|---|---|---|
| 사용자 코드 | `jnp.tanh(x @ w)` | `torch.tanh(x @ w)` | 사람이 쓰는 API |
| 프레임워크 그래프 | jaxpr (`dot_general`, `tanh`) | FX 그래프 (torch.* op) | 미분·배치 변환이 일어나는 수준 |
| 정규화된 primitive | jaxpr primitive (수백 개) | ATen / Prim IR (PrimTorch) | 백엔드가 알아야 할 op 집합 |
| 백엔드 입력 IR | StableHLO (MLIR) | ATen FX 그래프 | 여기서 프레임워크 역할 끝 |
| 백엔드 내부 IR | HLO → LLO ([③](./03-ir-to-executable.md)) | Inductor IR → Triton/C++ ([③](./03-ir-to-executable.md)) | |

두 스택 모두 "고수준 op → 소수의 primitive → 백엔드 IR"의 계단을 내려간다. 차이는 **그래프를 꺼내는 방법**이다.

| | JAX (Tracing) | PyTorch 2 (Dynamo) |
|---|---|---|
| 방법 | Tracer로 함수 실행 | 바이트코드 심볼릭 해석 |
| 제어 흐름 | 구조적 primitive 필수 (`lax.cond`) | Python 제어 흐름 대부분 허용, 안 되면 graph break |
| 미지원 코드 | 에러 | graph break 후 eager fallback |
| 그래프 단위 | 함수 전체가 하나의 프로그램 | graph break마다 조각남 |
| 미분 | jaxpr 변환 (`grad`) | AOT Autograd 조인트 그래프 + 파티셔너 |
| 재컴파일 조건 | 입력 shape/dtype 캐시 키 불일치 | guard 실패 |
| 동적 shape | 기본적으로 재컴파일 | 두 번째 shape부터 심볼(`s0`)로 승격, `dynamic=True` |

## 5. 확인 명령

| 보고 싶은 것 | JAX | PyTorch 2 |
|---|---|---|
| 프레임워크 그래프 | `jax.make_jaxpr(f)(*args)` | `TORCH_LOGS="graph_code"` |
| 백엔드로 넘어가는 IR | `jax.jit(f).lower(*args).as_text()` | `TORCH_LOGS="aot_graphs"` |
| 재컴파일 / 가드 | shape 바꿔 호출해 보기 | `TORCH_LOGS="guards,recompiles"` |
| graph break 위치와 이유 | (해당 없음) | `torch._dynamo.explain(f)(*args)` |

## 6. 다음 단계로

여기까지가 **프레임워크의 일**이다. JAX는 StableHLO를, PyTorch는 ATen 그래프를 만들고 손을 뗀다. 그래프를 받아 퓨전 경계·메모리 레이아웃·실행 순서를 정하고 실제 기계 코드를 만드는 것은 [③ IR → 실행 코드](./03-ir-to-executable.md)의 XLA / Inductor다.

각 스택의 단계별 상세는 [jax-to-tpu.md](./jax-to-tpu.md), [pytorch-compile.md](./pytorch-compile.md) 참고.

---

## 참고 자료

- PyTorch *torch.compiler* 개요: 도입부, TorchDynamo / AOTAutograd / TorchInductor / PrimTorch 설명 — https://docs.pytorch.org/docs/stable/torch.compiler.html
- Scaling Book 9장 *How to Profile TPU Code*: **A Thousand-Foot View of the TPU Software Stack** 절 — https://jax-ml.github.io/scaling-book/profiling/
