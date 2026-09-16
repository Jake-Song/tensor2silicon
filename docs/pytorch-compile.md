# PyTorch 코드가 torch.compile로 컴파일되는 흐름

`torch.compile`은 세 개의 독립된 컴포넌트가 이어진 파이프라인이다. 각 컴포넌트는 서로 다른 문제를 푼다.

```
Python 함수 (eager 코드)
   │  TorchDynamo: 바이트코드를 분석해 FX 그래프 포착
   ▼
FX 그래프 (torch.* 수준 op)  +  가드(guard)
   │  AOT Autograd: forward + backward 조인트 그래프 생성, ATen 수준으로 분해
   ▼
forward ATen 그래프  /  backward ATen 그래프
   │  TorchInductor: 퓨전·스케줄링 후 Triton(GPU) / C++(CPU) 코드 생성
   ▼
컴파일된 커널  ──  수정된 파이썬 바이트코드가 대신 호출
```

- **Dynamo**는 "파이썬에서 그래프를 어떻게 안전하게 꺼내느냐"를 해결한다.
- **AOT Autograd**는 "backward를 어떻게 미리 컴파일하느냐"를 해결한다.
- **Inductor**는 "그래프를 어떻게 빠른 커널로 만드느냐"를 해결한다.

이 문서에서는 다음 코드를 예시로 따라간다.

```python
import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)
    def forward(self, x):
        h = self.fc(x)
        return torch.nn.functional.gelu(h) * 2

model = torch.compile(MLP())
y = model(torch.randn(16, 8, requires_grad=True))
y.sum().backward()
```

---

## 1. TorchDynamo: 파이썬 바이트코드에서 그래프 포착

JAX와 가장 다른 지점이다. JAX는 함수를 Tracer로 실행해 추적하지만, Dynamo는 **CPython의 프레임 평가 API(PEP 523)** 를 후킹해 함수의 **바이트코드 자체**를 심볼릭하게 해석한다.

### 동작 순서

1. 함수가 호출되면 CPython이 프레임을 실행하기 직전에 Dynamo가 끼어든다.
2. Dynamo는 바이트코드를 한 명령씩 해석하며 파이썬 스택을 시뮬레이션한다. 텐서는 `TensorVariable`, 상수는 `ConstantVariable` 같은 심볼릭 변수로 추적된다.
3. 텐서 연산(`self.fc(x)`, `gelu`, `* 2`)을 만나면 **FX 그래프**에 노드로 기록한다. 이 그래프의 op는 아직 `torch.nn.functional.gelu` 같은 사용자 수준 연산이다.
4. 텐서와 무관한 파이썬 로직(리스트 조작, 딕셔너리 접근, 클래스 속성 읽기 등)은 그래프에 넣지 않고 Dynamo 내부에서 직접 처리한다.
5. 해석이 끝나면 원래 바이트코드를 **컴파일된 그래프를 호출하는 새 바이트코드**로 교체하고, 코드 객체에 캐시한다.

### 가드(Guard)

Dynamo가 추적 중에 가정한 조건들이다. 입력 텐서의 dtype, shape, stride, `requires_grad`, 참조한 전역 변수 값, 모듈 파라미터 정체성 등이 포함된다. 다음 호출 때 가드 검사가 통과하면 캐시된 코드를 바로 실행하고, 실패하면 재컴파일한다.

```bash
TORCH_LOGS="guards" python train.py
```

```
TENSOR_MATCH: check_tensor(L['x'], Tensor, ..., torch.float32, device=None,
              requires_grad=False, size=[3, 3], stride=[3, 1])
```

### 그래프 브레이크(Graph Break)

Dynamo가 처리할 수 없는 파이썬 코드를 만나면 그래프를 거기서 자른다. 대표적인 원인:

- `print` 같은 부수 효과가 있는 내장 함수
- 텐서 값에 의존하는 분기 (`if b.sum() < 0`)
- 지원되지 않는 서드파티 라이브러리 호출

JAX는 이런 코드에서 에러를 내지만, Dynamo는 앞 부분을 컴파일하고 중간 부분을 eager로 실행한 뒤 뒷 부분을 다시 컴파일한다. 그래서 `torch.compile`은 대부분의 기존 코드에 그냥 씌워도 동작한다.

브레이크 위치와 이유는 `torch._dynamo.explain`으로 확인한다.

```python
import torch._dynamo as dynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b

print(dynamo.explain(toy_example)(torch.randn(10), torch.randn(10)))
# Graph Count: 3
# Graph Break Count: 2
# Break Reason 1: builtin: print
# Break Reason 2: generic_jump TensorVariable()
```

### 동적 shape

같은 함수가 다른 shape으로 두 번째 호출되면 Dynamo는 해당 차원을 심볼(`s0`)로 승격해 재컴파일한다. 이후로는 가드가 정확한 크기 대신 심볼 관계(`s0 > 1` 등)만 검사한다. `torch.compile(dynamic=True)`로 처음부터 동적으로 만들 수도 있다.

### Dynamo가 만든 그래프 확인

```bash
TORCH_LOGS="graph_code" python train.py
```

```python
def forward(self, L_x_: "f32[16, 8]"):
    l_x_ = L_x_
    h: "f32[16, 4]" = self._modules['fc'](l_x_)
    gelu: "f32[16, 4]" = torch.nn.functional.gelu(h)
    mul: "f32[16, 4]" = gelu * 2
    return (mul,)
```

---

## 2. AOT Autograd: backward를 미리 만들고 ATen으로 내리기

Dynamo의 그래프는 forward만 담고 있고 op도 고수준이다. AOT(Ahead-Of-Time) Autograd는 이 그래프를 받아 세 가지 일을 한다.

### (a) 조인트 그래프 생성

FakeTensor(메모리 없이 메타데이터만 있는 텐서)로 forward를 실행하면서 autograd 엔진을 추적해, forward와 backward를 하나로 합친 **조인트(joint) 그래프**를 만든다. eager 모드에서는 backward가 실행 시점에 tape를 따라 동적으로 결정되지만, 여기서는 컴파일 시점에 그래프로 확정된다.

### (b) 분해(Decomposition)와 함수화(Functionalization)

- `torch.nn.functional.gelu` 같은 복합 op를 **ATen 코어 op**(`aten.mm`, `aten.erf`, `aten.mul` 등 수백 개의 primitive)로 분해한다. Inductor가 다뤄야 할 op 종류가 줄어든다.
- `x.add_(1)` 같은 in-place 연산을 `x = x.add(1)` 형태로 바꾼다. 그래프가 순수 함수형이 되어야 뒤 단계 최적화가 안전하다.

### (c) 파티셔닝

조인트 그래프를 forward 그래프와 backward 그래프로 자른다. 이때 **어떤 중간값을 forward에서 저장하고 어떤 것을 backward에서 재계산할지** 결정한다. 이 결정을 하는 min-cut 파티셔너가 activation 메모리와 재계산 비용의 균형을 잡는다. 사용자가 `torch.utils.checkpoint`를 쓰지 않아도 컴파일러가 선택적으로 재계산을 넣는 이유가 여기 있다.

### 결과 확인

```bash
TORCH_LOGS="aot_graphs" python train.py
```

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

# backward
def forward(self, primals_3, addmm, tangents_1):
    ...  # gelu 미분 + mm 미분
```

이 두 그래프는 `torch.autograd.Function`으로 감싸져 등록되므로, 사용자가 `y.sum().backward()`를 호출하면 eager autograd 엔진이 컴파일된 backward 그래프를 하나의 노드로 실행한다.

---

## 3. TorchInductor: 커널 코드 생성

Inductor는 ATen 그래프를 받아 실제 GPU/CPU 코드를 만드는 백엔드다. `torch.compile(backend="inductor")`가 기본값이다.

### (a) 로워링(Lowering)

각 ATen op를 Inductor의 내부 IR로 바꾼다. 이 IR은 텐서를 "인덱스를 받아 값을 반환하는 함수"로 표현하는 **define-by-run 방식**이다. 예를 들어 `aten.mul(a, 2)`는 `lambda i: a[i] * 2`처럼 표현된다. 이 표현 덕분에 연속된 원소별 연산을 합치는 것이 함수 합성처럼 자연스럽게 된다.

### (b) 스케줄링과 퓨전

노드 간 의존성을 보고 어떤 것을 하나의 커널로 합칠지 정한다.

- 원소별(pointwise) 연산끼리는 거의 무조건 합친다. 위 예시에서 gelu의 `mul, mul, erf, add, mul`과 마지막 `* 2`는 하나의 Triton 커널이 된다.
- pointwise 연산은 뒤따르는 reduction(`sum`, `softmax` 등)에도 합쳐질 수 있다.
- 행렬 곱은 기본적으로 cuBLAS/cuDNN 외부 커널을 호출하고, `mode="max-autotune"`을 주면 Triton 템플릿 mm을 자동 튜닝해 비교한다. Epilogue fusion으로 mm 뒤의 bias 덧셈이나 activation을 mm 커널 안에 넣기도 한다.

### (c) 코드 생성

- **GPU**: **Triton** 언어로 커널을 쓴다. Triton은 파이썬 문법의 GPU 커널 DSL이고, Triton 컴파일러가 이를 LLVM IR을 거쳐 PTX/SASS로 내린다. 블록 크기 같은 파라미터는 Inductor가 휴리스틱 또는 자동 튜닝으로 정한다.
- **CPU**: C++ 코드와 OpenMP, 벡터화 intrinsic을 생성하고 시스템 컴파일러로 빌드한다.
- 생성된 커널들을 순서대로 호출하고 메모리를 할당하는 **래퍼 코드**(파이썬 또는 C++)도 함께 만든다.

### 생성된 코드 확인

```bash
TORCH_LOGS="output_code" python train.py
```

gelu 부분은 대략 이런 모양이다.

```python
@triton.jit
def triton_poi_fused_add_erf_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (xindex), xmask)
    tmp1 = tmp0 * 0.5
    tmp2 = tmp0 * 0.7071067811865476
    tmp3 = libdevice.erf(tmp2)
    tmp4 = tmp3 + 1.0
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 * 2.0
    tl.store(out_ptr0 + (xindex), tmp6, xmask)
```

eager 모드였다면 gelu 하나에 커널 5~6개가 따로 실행되고 그때마다 HBM을 왕복했을 것이다. 이것이 `torch.compile` 성능 이득의 대부분이다.

---

## 4. 실행과 캐시

이후 같은 함수를 호출하면:

1. Dynamo가 심어 둔 가드 검사 코드가 먼저 실행된다.
2. 통과하면 컴파일된 forward가 호출되고, autograd 그래프에 컴파일된 backward가 노드로 연결된다.
3. 실패하면 재컴파일한다. 기본 재컴파일 한도(`torch._dynamo.config.cache_size_limit`)를 넘으면 그 함수는 eager로 되돌아간다.

컴파일 결과는 로컬 디스크 캐시(Inductor FX 그래프 캐시, Triton 커널 캐시)에도 저장되므로, 프로세스를 다시 띄워도 두 번째부터는 빠르다.

---

## JAX 파이프라인과의 대응

| 역할 | JAX | PyTorch 2 |
|---|---|---|
| 그래프 포착 | Tracer로 함수 실행 (엄격, 제어 흐름은 구조적 primitive 필수) | Dynamo가 바이트코드 해석 (비엄격, 그래프 브레이크로 우회) |
| 중간 표현 | jaxpr → StableHLO | FX 그래프 (torch op) → FX 그래프 (ATen op) |
| 미분 | jaxpr 변환 (`grad`) | AOT Autograd 조인트 그래프 + 파티셔너 |
| 재컴파일 조건 | shape/dtype 캐시 키 | 가드 실패 |
| 백엔드 컴파일러 | XLA (fusion, layout, SPMD) | Inductor (fusion, Triton/C++ codegen) |
| 커널 생성 | LLO → TPU 바이너리 | Triton → PTX, 또는 C++ |
| 커스텀 커널 | Pallas | Triton 직접 작성, 또는 `torch.library` 커스텀 op |

JAX 쪽 흐름은 [jax-to-tpu.md](./jax-to-tpu.md) 참고.

---

## 단계별 확인 명령 요약

| 보고 싶은 것 | 방법 |
|---|---|
| 그래프 브레이크 위치와 이유 | `torch._dynamo.explain(fn)(*args)` |
| Dynamo가 포착한 FX 그래프 | `TORCH_LOGS="graph_code"` |
| 가드 목록 | `TORCH_LOGS="guards"` |
| 재컴파일 이유 | `TORCH_LOGS="recompiles"` |
| AOT Autograd 결과 (ATen forward/backward) | `TORCH_LOGS="aot_graphs"` |
| Inductor가 생성한 Triton/C++ 코드 | `TORCH_LOGS="output_code"` |
| 퓨전 결정 | `TORCH_LOGS="fusion"` |
| 전체 흐름을 한 번에 | `TORCH_LOGS="+dynamo,aot,inductor"` 또는 `TORCH_TRACE` + `tlparse` |
