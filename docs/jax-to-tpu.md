# JAX 코드가 TPU 실행 코드로 변환되는 흐름

전체 파이프라인은 크게 다음 순서로 진행된다.

```
Python 함수
   │  jax.jit 추적 (Tracer)
   ▼
jaxpr  ──── grad / vmap / pmap 변환은 이 수준에서 수행
   │  lowering 규칙
   ▼
StableHLO (MLIR)  ──── JAX의 역할은 여기까지
   │
   ▼
HLO  ──── XLA 최적화 패스 (SPMD, 레이아웃, 퓨전, 메모리 스케줄링)
   │
   ▼
LLO → TPU 바이너리  ──── XLA TPU 백엔드 (비공개)
   │
   ▼
PjRt 런타임 실행
```

---

## 1. Python 추적 (Tracing)

`jax.jit`로 감싼 함수를 호출하면 JAX는 실제 값 대신 **Tracer** 객체를 인자로 넣어 함수를 한 번 실행한다. Tracer는 shape와 dtype만 갖는 추상 값이다.

함수 안에서 호출되는 `jnp.dot`, `jnp.exp` 같은 연산은 Python 수준에서 계산되지 않고, 각 연산이 **primitive**(`dot_general`, `exp` 등)로 기록된다.

주의할 점:

- Python의 `if`, `for` 같은 제어 흐름은 추적 시점에 한 번만 펼쳐진다.
- 값에 의존하는 분기나 반복은 `lax.cond`, `lax.scan`, `lax.while_loop` 같은 구조적 primitive로 써야 한다.
- `print` 같은 부수 효과는 추적 시 한 번만 실행된다.

## 2. jaxpr 생성

추적 결과는 **jaxpr**이라는 JAX 고유의 중간 표현(IR)으로 저장된다. 순수 함수형이고 SSA(Static Single Assignment) 형태다.

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

`grad`, `vmap`, `pmap` 같은 함수 변환은 모두 이 jaxpr 수준에서 이루어진다.

- 자동 미분(`grad`)은 jaxpr를 변환해 새로운 jaxpr를 만든다.
- `vmap`은 각 primitive에 배치 규칙(batching rule)을 적용한다.

## 3. jaxpr → StableHLO (MLIR)

jaxpr의 각 primitive는 **lowering 규칙**을 통해 MLIR의 StableHLO 방언으로 변환된다. 예전에는 XLA의 HLO 프로토콜 버퍼를 직접 생성했지만, 현재는 MLIR 기반 StableHLO를 거친다.

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

## 4. XLA 컴파일 (TPU 백엔드)

StableHLO는 XLA의 HLO로 변환되고, 여러 단계의 최적화 패스를 거친다.

| 패스 | 내용 |
|---|---|
| 타겟 독립 최적화 | 대수적 단순화, 상수 접기, 죽은 코드 제거, 공통 부분식 제거 |
| SPMD 파티셔너 | 샤딩 어노테이션을 보고 프로그램을 칩별 조각으로 분할하고, all-reduce / all-gather 등 집합 통신 삽입 |
| 레이아웃 할당 | MXU가 요구하는 타일 레이아웃(f32는 8×128, bf16은 16×128)에 맞게 텐서 메모리 배치 결정. 타일 배수가 아닌 shape은 패딩됨 |
| 퓨전(Fusion) | 원소별 연산들을 하나의 커널로 합쳐 HBM 왕복을 줄임. 위 예시의 `dot_general`과 `tanh`는 하나의 fusion으로 합쳐질 가능성이 높음 |
| 메모리 할당 및 스케줄링 | HBM과 온칩 VMEM 사이 데이터 이동을 정적으로 계획. TPU는 캐시가 없어 컴파일러가 모든 메모리 이동을 미리 결정 |

최적화된 HLO는 다음으로 확인할 수 있다.

```python
print(jax.jit(f).lower(x, w).compile().as_text())
```

## 5. LLO 및 TPU 바이너리 생성

최적화된 HLO는 TPU 전용 **LLO(Low Level Optimizer)** 단계로 내려간다. 이 부분은 XLA의 비공개 TPU 백엔드다.

- 각 fusion을 TPU의 VLIW 명령어로 변환한다. 스칼라 유닛, 벡터 유닛(VPU), MXU, DMA 엔진에 대한 명령이 한 번들로 묶인다.
- 소프트웨어 파이프라이닝과 명령어 스케줄링으로 DMA와 계산을 겹친다.
- 결과물은 TPU 실행 파일이며, 런타임이 이를 각 TPU 코어에 로드한다.

## 6. 실행 (PjRt 런타임)

JAX는 **PjRt** API를 통해 컴파일된 실행 파일을 다룬다.

- 입력 배열은 디바이스 버퍼로 전송되고, 실행 파일이 비동기로 디스패치된다.
- JAX 배열은 결과가 완성되기 전에 반환되며, 값을 실제로 읽을 때(`block_until_ready`, 출력 등) 대기한다.
- 컴파일 결과는 입력 shape/dtype 조합을 키로 캐시된다. 같은 shape으로 다시 호출하면 1~5단계를 건너뛰고 바로 실행된다.
- shape이 바뀌면 처음부터 재컴파일이 일어난다.

## 요약 표

| 단계 | 표현 | 담당 | 확인 방법 |
|---|---|---|---|
| 추적 | Tracer | JAX | `jax.make_jaxpr` |
| 중간 표현 | jaxpr | JAX | `jax.make_jaxpr` |
| 로워링 | StableHLO (MLIR) | JAX | `.lower().as_text()` |
| 최적화 | HLO | XLA | `.compile().as_text()` |
| 코드 생성 | LLO → TPU 바이너리 | XLA TPU 백엔드 (비공개) | 프로파일러 |
| 실행 | 디바이스 버퍼 | PjRt | `block_until_ready` |

## 참고: Pallas

XLA가 만들어 주는 커널이 만족스럽지 않을 때는 `jax.experimental.pallas`로 TPU 커널을 직접 쓸 수 있다.

- jaxpr가 StableHLO 대신 **Mosaic** 방언(MLIR)으로 로워링된다.
- 4단계의 대부분을 우회한다.
- VMEM 블록 크기와 DMA 파이프라인을 사용자가 직접 지정한다.
