# ④ 실행 요청 → 커널 실행: Python 함수가 반환되면 가속기 계산도 끝난 걸까?

> **한 줄 답.** **아니다.** Python 함수는 "이 작업을 실행하라"는 요청을 디바이스 큐에 넣고 **즉시** 반환한다. 반환된 배열은 아직 값이 없는 **future**이고, 값을 실제로 읽으려는 순간에야 호스트가 디바이스를 기다린다. 커널은 "프로그램 인스턴스 여러 개가 각자 데이터 한 블록씩을 맡는" SPMD 그리드로 런치된다.

```
호스트 (Python)                                디바이스 (TPU / GPU)
──────────────────────                          ──────────────────────
y = f(x)         ─▶ 큐에 [f 실행] 넣고 즉시 반환      ┐
z = g(y)         ─▶ 큐에 [g 실행] 넣고 즉시 반환      │ 큐    f 실행 중 …
w = h(z)         ─▶ 큐에 [h 실행] 넣고 즉시 반환      │      g 대기
print(w)         ─▶ 여기서 블록 ─────────────────────┘      h 대기
                    (f, g, h가 모두 끝나야 값을 받음)
```

이전 단계 ← [③ IR → 실행 코드](./03-ir-to-executable.md) · 다음 단계 → [⑤ 커널 → 칩 내부](./05-kernel-to-chip.md)

---

## 1. 비동기 디스패치 (JAX Asynchronous dispatch)

JAX는 Python 오버헤드를 줄이려고 **비동기 디스패치**를 쓴다. `jnp.dot(x, x)`를 호출하면 JAX는 계산이 끝나기를 기다리지 않고 곧바로 `jax.Array`를 반환한다. 이 배열은 **아직 만들어지지 않은 값에 대한 future**다.

이렇게 하면 Python이 디바이스보다 **앞서 달릴 수** 있다. 호스트가 다음 연산을 디스패치하는 동안 디바이스는 이전 연산을 실행하고, 결과적으로 호스트의 Python 오버헤드가 임계 경로(critical path)에서 빠진다.

### 언제 블록되는가

JAX가 실제로 기다리는 시점은 **값을 호스트에서 봐야 할 때**뿐이다.

| 동작 | 블록 여부 |
|---|---|
| `y = jnp.dot(x, x)` | 안 함 (future 반환) |
| `y.shape`, `y.dtype` | 안 함 (메타데이터는 이미 안다) |
| `print(y)`, `repr(y)`, `str(y)` | 블록 |
| `np.asarray(y)`, `float(y)`, `y.tolist()` | 블록 + 호스트로 전송 |
| `y.block_until_ready()` | 블록 (전송 없음) |
| 다음 JAX 연산에 `y`를 넣기 | 안 함 (디바이스 큐에서 순서대로 실행) |

### 벤치마크 함정

JAX 문서의 예제다.

```python
x = jnp.ones((1000, 1000))
%time jnp.dot(x, x)                       # ~269 µs  ← 디스패치 시간만 잰 것
%time np.asarray(jnp.dot(x, x))           # ~8.09 ms ← 계산 + 호스트 전송
%time jnp.dot(x, x).block_until_ready()   # ~4.92 ms ← 계산만
```

첫 줄의 269 µs는 1000×1000 행렬곱 시간이 아니라 **작업을 큐에 넣는 데 걸린 시간**이다. 문서는 "비동기 디스패치가 우리를 속이고 있다. 실행 시간이 아니라 디스패치 시간만 재고 있다"고 적는다. 마이크로벤치마크는 `block_until_ready()`로 끝을 맞춰야 하고, `np.asarray`는 호스트 전송 비용까지 포함하므로 더 느리게 나온다.

### 이것이 가능한 조건

비동기 디스패치가 안전한 이유는 [②](./02-ops-to-graph-ir.md)와 [③](./03-ir-to-executable.md)에서 **출력 shape/dtype이 컴파일 시점에 확정**되기 때문이다. 결과 값은 몰라도 결과의 모양은 알기 때문에 future 객체를 미리 만들 수 있다. 값에 의존하는 분기(`if y.sum() > 0`)가 그래프를 깨는 이유도 같다. 그 지점에서 호스트가 값을 봐야 하므로 블록이 강제된다.

## 2. 커널 하나는 어떻게 실행되는가: Triton 벡터 덧셈

[③](./03-ir-to-executable.md)의 Inductor가 만드는 것이 바로 Triton 커널이다. Triton 튜토리얼의 벡터 덧셈은 "커널 코드"와 "호출 코드"가 어떻게 나뉘는지 보여주는 가장 작은 예다.

### 커널 쪽: 각 프로그램 인스턴스가 한 블록을 처리한다

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

| 요소 | 의미 |
|---|---|
| `@triton.jit` | 이 Python 함수를 Triton 컴파일러가 GPU 커널로 컴파일한다. Python 인터프리터가 실행하는 함수가 아니다. |
| `x_ptr`, `y_ptr`, `output_ptr` | 텐서가 아니라 **디바이스 메모리(HBM)의 첫 원소 포인터**다. |
| `BLOCK_SIZE: tl.constexpr` | 컴파일 시점 상수. 값이 바뀌면 다른 커널이 컴파일된다. shape 값으로 쓸 수 있는 이유다. |
| `tl.program_id(axis=0)` | "나는 몇 번째 프로그램 인스턴스인가". 같은 커널이 그리드 크기만큼 동시에 돌고, 이 값만 다르다. |
| `block_start + tl.arange(0, BLOCK_SIZE)` | 이 인스턴스가 맡은 원소들의 인덱스 벡터. 길이 256, 블록 64이면 인스턴스 0~3이 `[0:64]`, `[64:128]`, `[128:192]`, `[192:256]`을 맡는다. |
| `mask = offsets < n_elements` | 벡터 길이가 블록 크기의 배수가 아닐 때 마지막 인스턴스가 범위 밖을 읽고 쓰지 않도록 막는다. |
| `tl.load(ptr + offsets, mask=)` | HBM에서 블록 하나를 레지스터로 읽는다. 마스크가 거짓인 자리는 읽지 않는다. |
| `output = x + y` | 블록 단위 벡터 연산. 스레드를 직접 다루지 않고 Triton 컴파일러가 스레드에 나눠 준다. |
| `tl.store(ptr + offsets, output, mask=)` | 결과를 HBM에 쓴다. |

핵심은 **커널은 "전체 벡터"를 모른다**는 점이다. 자기 `pid`에 해당하는 블록 하나만 처리하고, 전체를 덮는 일은 호출 쪽이 그리드 크기로 보장한다.

### 호출 쪽: 출력 버퍼를 잡고, 그리드를 정하고, 런치하고, 기다리지 않는다

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```

| 요소 | 의미 |
|---|---|
| `torch.empty_like(x)` | **출력 버퍼는 호출자가 미리 잡는다.** 커널은 새 메모리를 할당하지 않고 포인터에 쓸 뿐이다. |
| `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)` | **SPMD 런치 그리드** = 동시에 실행할 커널 인스턴스 수. `cdiv`는 올림 나눗셈이라 마지막 조각까지 덮는다. `meta`로 받는 이유는 `BLOCK_SIZE`가 autotune으로 바뀔 수 있어서다. |
| `add_kernel[grid](...)` | `[grid]`로 인덱싱하면 호출 가능한 GPU 커널이 된다. 텐서 인자는 자동으로 첫 원소 포인터로 바뀐다. |
| `BLOCK_SIZE=1024` | constexpr 메타파라미터는 키워드 인자로 넘긴다. |
| `return output` | **커널은 아직 돌고 있다.** 원문 주석: "`torch.cuda.synchronize()`가 호출되지 않았으므로 이 시점에 커널은 여전히 비동기로 실행 중이다." 반환되는 것은 완성된 값이 아니라 핸들이다. |

즉 Triton 호출 코드에서도 JAX와 정확히 같은 일이 벌어진다. Python 함수 `add`는 커널을 CUDA 스트림에 넣고 반환한다. 이후 `output`을 CPU로 가져오거나(`.cpu()`, `.item()`), `torch.cuda.synchronize()`를 부를 때 호스트가 기다린다.

## 3. 호스트–디바이스 실행 모델 대응

| | JAX / XLA (TPU, GPU) | PyTorch / Inductor (GPU) |
|---|---|---|
| 실행 단위 | 컴파일된 실행 파일 하나 (PjRt executable) | Inductor 래퍼가 Triton/cuBLAS 커널을 **순서대로 런치** |
| 큐 | PjRt 런타임의 디바이스 큐 | CUDA 스트림 |
| 요청 반환 | 즉시, `jax.Array` future | 즉시, `torch.Tensor` 핸들 |
| 출력 버퍼 | 런타임이 실행 파일 출력용으로 할당 | 래퍼 코드가 `empty_strided` 등으로 미리 할당 |
| 명시적 동기화 | `x.block_until_ready()` | `torch.cuda.synchronize()`, `stream.synchronize()` |
| 암묵적 동기화 | `print`, `np.asarray`, `float()` | `.cpu()`, `.item()`, `.numpy()`, `print` |
| 여러 요청의 순서 | 같은 디바이스 큐에서 FIFO | 같은 스트림 안에서 FIFO |
| 병렬성 표현 | 실행 파일 내부에서 컴파일러가 결정 | 그리드 크기 × 블록 크기 (SPMD) |

두 스택 모두 **"호스트는 요청만 쌓고, 디바이스가 순서대로 비운다"** 는 생산자–소비자 모델이다. 차이는 요청의 굵기다. XLA는 그래프 전체가 요청 하나이고, Inductor는 커널마다 요청 하나다(CUDA graph를 쓰면 이 차이가 줄어든다).

## 4. 왜 이렇게 설계했는가

- **런치 오버헤드 숨기기.** 커널 하나 런치에 수 µs, Python 한 줄에 수십 µs가 든다. 디바이스가 이전 작업을 실행하는 동안 호스트가 다음 요청을 준비하면 이 비용이 겹쳐 사라진다.
- **호스트를 임계 경로에서 빼기.** [①](./01-model-to-ops.md)의 대형 matmul은 ms 단위다. 호스트가 매번 기다리면 디바이스가 그 사이 논다.
- **파이프라인 유지.** 학습 루프에서 step N의 backward가 도는 동안 step N+1의 데이터 로딩과 디스패치가 진행된다.

대가는 **디버깅과 측정이 직관과 어긋난다**는 것이다. 에러가 나중에 터지고, `%time`이 거짓말을 한다. 그래서 `block_until_ready()`와 `torch.cuda.synchronize()`를 알아야 한다.

## 5. 다음 단계로

큐에서 꺼내진 커널 인스턴스는 칩의 어떤 유닛에서 실행되고, `tl.load`가 읽는 데이터는 어디서 오는가. [⑤ 커널 → 칩 내부](./05-kernel-to-chip.md)에서 다룬다.

---

## 참고 자료

- JAX *Asynchronous dispatch* — https://docs.jax.dev/en/latest/async_dispatch.html
- Triton *Vector Addition* 튜토리얼: 커널(`add_kernel`)과 호출 코드(`add`) — https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
