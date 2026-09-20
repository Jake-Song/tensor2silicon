# JAX 코드가 TPU(Tensor Processing Unit) 실행 코드로 변환되는 흐름

전체 파이프라인은 크게 다음 순서로 진행된다.

```
Python 함수
   │  jax.jit의 JIT(Just-In-Time) 추적 (Tracer)
   ▼
jaxpr  ──── grad / vmap / pmap 변환은 이 수준에서 수행
   │  lowering 규칙
   ▼
StableHLO (MLIR: Multi-Level Intermediate Representation)  ──── JAX의 역할은 여기까지
   │
   ▼
HLO(High Level Optimizer)  ──── XLA(Accelerated Linear Algebra) 최적화 패스 (SPMD(Single Program, Multiple Data), 레이아웃, 퓨전, 메모리 스케줄링)
   │
   ▼
LLO(Low Level Optimizer) → TPU 바이너리  ──── XLA TPU 백엔드 (비공개)
   │
   ▼
PjRt(Portable JIT Runtime) 런타임 실행
```

용어: JIT는 호출 시점에 필요한 입력 형태에 맞춰 코드를 컴파일하는 방식이다. MLIR은 여러 컴파일러 표현을 담는 중간 표현 기반이다. HLO와 LLO는 각각 고수준·저수준 최적화 단계이며, XLA는 JAX 계산을 하드웨어용 코드로 컴파일하는 컴파일러다. PjRt는 컴파일한 프로그램을 장치에서 실행하는 런타임 API(Application Programming Interface)다.

---

## 1. Python 추적 (Tracing)

`jax.jit`로 감싼 함수를 호출하면 JAX는 실제 값 대신 **Tracer** 객체를 인자로 넣어 함수를 한 번 실행한다. Tracer는 shape와 dtype만 갖는 추상 값이다.

용어: Tracer는 실제 배열 대신 계산의 형태를 기록하는 객체다. shape는 배열 각 축의 크기이고, dtype은 원소의 자료형이다.

함수 안에서 호출되는 `jnp.dot`, `jnp.exp` 같은 연산은 Python 수준에서 계산되지 않고, 각 연산이 **primitive**(`dot_general`, `exp` 등)로 기록된다.

용어: `jnp`는 `jax.numpy` 모듈의 관례적인 별칭이다. primitive는 JAX가 추적할 수 있는 기본 연산이다.

주의할 점:

- Python의 `if`, `for` 같은 제어 흐름은 추적 시점에 한 번만 펼쳐진다.
- 값에 의존하는 분기나 반복은 Python 제어문 그대로 쓰면 안 되고, `lax.cond`, `lax.scan`, `lax.while_loop` 같은 구조적 primitive로 표현해야 한다.
- `print` 같은 부수 효과는 추적 시 한 번만 실행된다.

용어: 구조적 primitive는 분기나 반복 자체를 계산 그래프에 넣는 연산이다. 부수 효과는 반환값 외에 화면 출력처럼 프로그램 상태에 영향을 주는 동작이다.

예를 들어, 다음 코드는 JAX에서 잘 동작하지 않는다.

```python
import jax
import jax.numpy as jnp

@jax.jit
def bad(x):
    if x > 0:
        return x + 1
    else:
        return x - 1
```

`x`는 추적 시점의 abstract value이기 때문에 Python의 `if x > 0` 조건을 실제로 평가할 수 없다. 대신 다음처럼 바꿔야 한다.

용어: abstract value는 실제 원소값 없이 shape와 dtype만 가진 추상 입력이다.

```python
import jax
import jax.numpy as jnp

@jax.jit
def good(x):
    return jax.lax.cond(
        x > 0,
        lambda t: t + 1,
        lambda t: t - 1,
        operand=x,
    )
```

이처럼 값에 따라 동작이 바뀌는 로직은 Python의 분기 대신 JAX의 구조적 primitive로 표현해야 한다. 비슷하게, 값에 의존하는 루프는 `lax.scan`이나 `lax.while_loop`로 바꿔야 한다.

용어: `lax.cond`는 조건값에 따라 두 계산 경로 중 하나를 장치에서 선택하는 primitive다. `lax.scan`과 `lax.while_loop`는 반복을 계산 그래프에 기록하는 primitive다.

## 2. jaxpr 생성

추적 결과는 **jaxpr**이라는 JAX 고유의 중간 표현(IR, Intermediate Representation)으로 저장된다. 순수 함수형이고 SSA(Static Single Assignment) 형태다.

용어: jaxpr는 JAX 연산과 데이터 의존 관계를 기록하는 내부 표현이다. 순수 함수형은 같은 입력에 같은 출력을 내고 외부 상태를 바꾸지 않는 계산 방식이다. SSA는 각 중간값을 한 번만 정의하는 IR 형식이다.

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

용어: `f32`는 32비트 부동소수점(float32) 자료형이다. `dot_general`은 행렬 곱과 일반화된 텐서 수축을 표현하는 primitive다.

`grad`, `vmap(Vectorized Map)`, `pmap(Parallel Map)` 같은 함수 변환은 모두 이 jaxpr 수준에서 이루어진다.

- 자동 미분(`grad`)은 jaxpr를 변환해 새로운 jaxpr를 만든다.
- `vmap`은 각 primitive에 배치 규칙(batching rule)을 적용한다.
- `pmap`은 여러 디바이스에 걸쳐 같은 연산을 병렬로 수행할 수 있도록 shard/replicate 정보를 jaxpr에 반영한다.

용어: `grad`는 gradient의 약자로 미분값을 계산하는 변환이다. 배치 규칙(batching rule)은 단일 입력용 연산을 배치 축 전체에 적용하는 방법이다. shard는 분산 실행을 위해 원본 배열을 나눈 조각이고, replicate는 같은 값을 여러 장치에 복제하는 방식이다.

## vmap과 pmap: 배치 변환과 멀티 디바이스 병렬

JAX에서 `vmap`과 `pmap`는 모두 "함수를 여러 입력에 적용"하는 개념이지만, 목적과 실행 위치가 다르다.

### 1) `vmap`: 배치 차원에 대한 벡터화

`vmap(Vectorized Map)`은 함수가 한 예제(example)만 처리하도록 작성하고, 그 함수를 배치 전체에 자동으로 적용한다.

용어: 배치(batch)는 여러 예제를 하나의 배열에 모은 축이다.

```python
import jax
import jax.numpy as jnp

def f(x):
    return x * 2

xs = jnp.array([1, 2, 3])
y = jax.vmap(f)(xs)
# -> [2, 4, 6]
```

핵심은 다음과 같다.

- `f`는 단일 값에 대해 동작한다.
- `vmap`이 `f`의 배치 규칙을 만들어 배치 전체를 한 계산으로 확장한다.
- Python 루프를 직접 쓰지 않고도 배치 연산을 구현할 수 있다.
- 보통 단일 디바이스 안에서, 배치 축을 따라 연산을 확장한다.

즉, `vmap`은 "한 번에 배치 전체를 처리하는 규칙을 자동으로 만드는 것"이라 보면 된다. 모델 배치 처리나 시퀀스의 각 timestep에 같은 연산을 적용할 때 자주 쓴다.

용어: timestep은 시퀀스에서 한 단계의 위치다. `vmap`은 보통 단일 장치 안에서 배치 축을 따라 계산을 확장한다.

### 2) `pmap`: 디바이스 간 병렬화

`pmap(Parallel Map)`은 여러 TPU(Tensor Processing Unit)/GPU(Graphics Processing Unit) 코어 또는 디바이스에 데이터를 쪼개서 분산시켜 같은 함수가 각 shard를 병렬로 처리하게 한다.

용어: TPU는 행렬 연산 가속기이고, GPU는 그래픽과 범용 병렬 계산을 수행하는 가속기다.

```python
import jax
import jax.numpy as jnp

def f(x):
    return x * 2

xs = jnp.arange(8)
y = jax.pmap(f)(xs)
```

이 경우 JAX는 배열을 여러 디바이스로 나누고, 각 디바이스에서 `f`를 각각 독립적으로 실행한다. 결과는 다시 합쳐진다.

핵심 차이점은 다음과 같다.

- `vmap`: 같은 디바이스 안에서 배치 축을 벡터화한다.
- `pmap`: 여러 디바이스에 걸쳐 병렬로 계산한다.

### 3) 왜 JAX 컴파일 흐름에서 중요한가

`vmap`과 `pmap`는 모두 jaxpr 수준에서 변환되는 연산이다.

- `vmap`은 primitive에 batching rule을 적용해 batch 축을 확장한다.
- `pmap`는 sharding/parallelism 정보를 붙여서 XLA가 SPMD(Single Program, Multiple Data) 최적화를 할 수 있게 만든다.

즉, JAX는 고수준의 함수 변환을 jaxpr로 바꾸고, 그 다음 XLA가 실제 TPU 코어에 맞게 파티셔닝, 통신, fusion, 메모리 스케줄링을 수행한다.

용어: sharding은 배열이나 계산을 여러 장치에 나누는 방식이다. parallelism은 여러 작업을 동시에 수행하는 성질이다. SPMD는 같은 프로그램을 서로 다른 데이터 조각에 실행하는 분산 병렬 방식이다. fusion은 여러 연산을 하나의 실행 단위로 합치는 최적화다.

### 4) 한 줄 요약

- `vmap`: "배치 차원에 대해 같은 계산을 반복 적용"
- `pmap`: "여러 디바이스에 걸쳐 같은 계산을 병렬 실행"

이 둘을 잘 구분하면, JAX에서 데이터 병렬 학습, TPU sharding, 대규모 batch 연산을 이해하는 데 훨씬 수월해진다.

용어: 데이터 병렬은 각 장치가 서로 다른 데이터 조각에서 같은 모델 계산을 수행한 뒤 결과를 합치는 방식이다.

## 3. jaxpr → StableHLO(MLIR)

jaxpr의 각 primitive는 **lowering 규칙**을 통해 MLIR(Multi-Level Intermediate Representation)의 StableHLO 방언으로 변환된다. 예전에는 XLA의 HLO 프로토콜 버퍼를 직접 생성했지만, 현재는 MLIR 기반 StableHLO를 거친다.

용어: lowering은 높은 수준의 표현을 더 낮은 수준의 표현으로 바꾸는 과정이다. 방언(dialect)은 MLIR 안에서 특정 연산 집합을 정의하는 확장 단위다. StableHLO는 컴파일러와 프레임워크 사이에서 연산을 교환하기 위한 표준화된 HLO 연산 집합이다.

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

이 시점부터 JAX의 역할은 끝나고, 나머지는 백엔드 컴파일러가 맡는다. 샤딩 정보(`jax.sharding`, `shard_map`)도 이 단계에서 어노테이션 형태로 함께 실려 간다.

용어: 백엔드는 특정 하드웨어용 코드를 생성하는 컴파일러 부분이다. 어노테이션은 계산의 의미를 바꾸지 않고 배치·분할 같은 추가 정보를 담는 표시다.

## 4. XLA(Accelerated Linear Algebra) 컴파일(TPU 백엔드)

StableHLO는 XLA의 HLO로 변환되고, 여러 단계의 최적화 패스를 거친다.

| 패스 | 내용 |
|---|---|
| 타겟 독립 최적화 | 대수적 단순화, 상수 접기, 죽은 코드 제거, 공통 부분식 제거 |
| SPMD 파티셔너 | 샤딩 어노테이션을 보고 프로그램을 칩별 조각으로 분할하고, 집합 통신을 삽입 |
| 레이아웃 할당 | MXU(Matrix Multiply Unit)가 요구하는 타일 레이아웃(`f32`는 8×128, `bf16`은 16×128)에 맞게 텐서 메모리 배치를 결정. 타일 배수가 아닌 shape은 패딩됨 |
| 퓨전(Fusion) | 원소별 연산들을 하나의 커널로 합쳐 HBM(High Bandwidth Memory) 왕복을 줄임. 위 예시의 `dot_general`과 `tanh`는 하나의 fusion으로 합쳐질 가능성이 높음 |
| 메모리 할당 및 스케줄링 | HBM과 온칩 VMEM(Vector Memory) 사이 데이터 이동을 정적으로 계획. TPU는 캐시가 없어 컴파일러가 모든 메모리 이동을 미리 결정 |

용어: all-reduce는 각 장치의 값을 결합하고 결과를 모두에게 주는 집합 통신이다. all-gather는 각 장치의 조각을 모아 전체 배열을 만드는 집합 통신이다. MXU는 TPU의 행렬 곱 연산기다. `bf16`은 bfloat16 자료형이다. HBM은 장치에 연결된 고대역폭 메모리이고, VMEM은 TPU 코어 가까이에 있는 벡터 메모리다. fusion은 여러 연산을 하나의 커널로 합쳐 중간 결과의 메모리 이동을 줄이는 최적화다.

최적화된 HLO는 다음으로 확인할 수 있다.

```python
print(jax.jit(f).lower(x, w).compile().as_text())
```

## 5. LLO 및 TPU 바이너리 생성

최적화된 HLO는 TPU 전용 **LLO(Low Level Optimizer)** 단계로 내려간다. 이 부분은 XLA의 비공개 TPU 백엔드다.

용어: TPU 바이너리는 TPU 런타임이 장치에 로드해 실행할 수 있는 컴파일 결과물이다. 백엔드가 비공개라는 것은 이 저수준 코드 생성 구현을 사용자가 직접 볼 수 없다는 뜻이다.

- 각 fusion을 TPU의 VLIW(Very Long Instruction Word) 명령으로 변환한다. 스칼라 유닛, VPU(Vector Processing Unit), MXU, DMA(Direct Memory Access) 엔진에 대한 명령이 한 묶음에 들어간다.
- 소프트웨어 파이프라이닝과 명령어 스케줄링으로 DMA와 계산을 겹친다.
- 결과물은 TPU 실행 파일이며, 런타임이 이를 각 TPU 코어에 로드한다.

용어: VLIW는 서로 독립적인 여러 연산 명령을 하나의 긴 명령어로 묶는 형식이다. VPU는 벡터 원소를 병렬 처리하는 연산기다. DMA는 CPU 개입 없이 메모리와 장치 사이 데이터를 옮기는 기능이다. 소프트웨어 파이프라이닝은 데이터 이동과 계산을 겹치도록 명령을 배열하는 기법이다.

## 6. 실행(PjRt 런타임)

JAX는 **PjRt(Portable JIT Runtime)** API(Application Programming Interface)를 통해 컴파일된 실행 파일을 다룬다.

용어: 런타임은 컴파일 결과를 장치에 배치하고 실행을 관리하는 프로그램 구성 요소다.

- 입력 배열은 디바이스 버퍼로 전송되고, 실행 파일이 비동기로 디스패치된다.
- JAX 배열은 결과가 완성되기 전에 반환되며, 값을 실제로 읽을 때(`block_until_ready`, 출력 등) 대기한다.
- 컴파일 결과는 입력 shape/dtype 조합을 키로 캐시된다. 같은 shape으로 다시 호출하면 1~5단계를 건너뛰고 바로 실행된다.
- shape이 바뀌면 처음부터 재컴파일이 일어난다.

용어: 장치 버퍼는 TPU 메모리에 있는 배열 저장 공간이다. 비동기 디스패치는 Python 코드가 장치 계산의 완료를 기다리지 않고 계속 진행하는 실행 방식이다. 캐시는 이전 컴파일 결과를 다시 쓰기 위해 보관하는 저장소이며, 재컴파일은 새 shape나 dtype에 맞는 실행 파일을 다시 만드는 과정이다.

## 요약 표

| 단계 | 표현 | 담당 | 확인 방법 |
|---|---|---|---|
| 추적 | Tracer | JAX | `jax.make_jaxpr` |
| 중간 표현 | jaxpr | JAX | `jax.make_jaxpr` |
| 로워링 | StableHLO(MLIR) | JAX | `.lower().as_text()` |
| 최적화 | HLO(High Level Optimizer) | XLA | `.compile().as_text()` |
| 코드 생성 | LLO(Low Level Optimizer) → TPU 바이너리 | XLA TPU 백엔드 (비공개) | 프로파일러 |
| 실행 | 디바이스 버퍼 | PjRt | `block_until_ready` |

## 참고: Pallas

XLA가 만들어 주는 커널이 만족스럽지 않을 때는 `jax.experimental.pallas`로 TPU 커널을 직접 쓸 수 있다.

용어: Pallas는 JAX에서 GPU와 TPU용 커널을 직접 작성하는 실험적 API다. 커널은 장치에서 반복 실행되는 작은 계산 프로그램이다.

- jaxpr가 StableHLO 대신 **Mosaic** 방언(MLIR)으로 로워링된다.
- 4단계의 대부분을 우회한다.
- VMEM 블록 크기와 DMA 파이프라인을 사용자가 직접 지정한다.

용어: Mosaic은 TPU 커널 작성을 위한 JAX/MLIR 기반 프로그래밍 모델이다. DMA 파이프라인은 다음 데이터 조각을 옮기면서 현재 조각을 계산하도록 구성한 실행 순서다.
