# 6주차: Python에서 Triton 커널로

**프로그램 하나가 원소 블록을 처리한다. Fusion은 중간 결과의 HBM 왕복을 줄인다.**

[3주차 메모리 이동](../week3/06-vector-add.md) → [4주차 Roofline](../week4/02-roofline.md) → [5주차 측정 경계](../week5/05-measurement.md)를 실제 커널과 연결한다. `tl.dot`·Tiled MatMul·컴파일 IR은 7주차 범위다.

실습: [노트북](../notebooks/week6_triton.ipynb) · [Colab에서 열기](https://colab.research.google.com/github/Jake-Song/tensor2silicon/blob/main/notebooks/week6_triton.ipynb)

## ① 스레드 대신 원소 블록으로 생각하기

| CUDA의 전형적인 원소별 코드 | 이번 주 Triton 코드 |
|---|---|
| `blockIdx`, `threadIdx`로 스레드별 원소 계산 | `tl.program_id(0)`로 프로그램별 구간 계산 |
| 스레드마다 스칼라 연산 작성 | `tl.arange`로 만든 원소 묶음에 연산 작성 |

`grid = (triton.cdiv(N, BLOCK_SIZE),)`는 **프로그램 인스턴스 수**다. `BLOCK_SIZE`는 원소 수이며 스레드 수가 아니다. 컴파일러가 블록의 연산을 스레드·레지스터에 배치하고 coalescing을 최적화한다. 작성자는 연속 접근이 가능한 인덱스와 stride, 블록 크기 등을 정한다. [근거: Introduction의 Motivations·Challenges](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)

핵심 용어: **program instance · grid · coalescing**

## ② 포인터와 마스크

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

텐서를 커널에 넘기면, 커널은 **데이터가 어디서 시작하고, 원소 하나의 자료형이 무엇인지**를 받는다. 예를 들어 FP32 텐서 `X = [10, 20, 30, 40]`을 넘겼다면, 커널의 `x_ptr`은 첫 원소 `10`이 저장된 메모리 주소다.

```python
x_ptr      # 첫 번째 원소의 주소
x_ptr + 1  # 두 번째 원소의 주소
x_ptr + 2  # 세 번째 원소의 주소
```

**`+1`은 다음 원소로 이동한다는 뜻**이다. FP32 원소 하나는 4바이트이므로 실제 주소는 4바이트만큼 이동한다.

여러 위치도 한꺼번에 표현할 수 있다.

```python
offsets = tl.arange(0, 4)  # [0, 1, 2, 3]
ptrs = x_ptr + offsets    # [첫째 주소, 둘째 주소, 셋째 주소, 넷째 주소]
x = tl.load(ptrs)         # [10, 20, 30, 40]
```

여기서 **정수 블록**은 여러 정수를 묶은 것, **포인터 블록**은 여러 메모리 주소를 묶은 것이다. 주소를 만드는 것과 실제 값을 읽는 것은 별개이며, `tl.load`가 값을 읽는다.

이 포인터만으로는 텐서가 몇 행·몇 열인지(`shape`), 다음 행이나 열까지 몇 원소를 이동해야 하는지(`stride`)를 알 수 없다. 따라서 **커널 계산에 필요한 크기와 간격은 별도 인자로 넘긴다.** [근거: triton.jit](https://triton-lang.org/main/python-api/generated/triton.jit.html)

예를 들어 2행 3열 텐서에서 각 행을 읽는 커널은 다음처럼 작성할 수 있다.

```python
@triton.jit
def read_rows(x_ptr, n_cols, stride_row, stride_col,
              BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    ptrs = x_ptr + row * stride_row + cols * stride_col
    values = tl.load(ptrs, mask=cols < n_cols, other=0.0)
    # 여기서 values로 필요한 계산을 수행한다.


X = torch.tensor([[10, 20, 30],
                  [40, 50, 60]], device="cuda", dtype=torch.float32)

read_rows[(X.shape[0],)](
    X,
    X.shape[1],   # n_cols=3: 한 행에 있는 원소 수
    X.stride(0), # stride_row=3: 다음 행까지 3원소 이동
    X.stride(1), # stride_col=1: 다음 열까지 1원소 이동
    BLOCK_SIZE=4,
)
```

`X.shape[0]`은 프로그램을 2개 실행하는 데 사용하고, 열 개수와 두 stride는 커널 인자로 전달한다. 두 번째 행(`row=1`)에서는 `x_ptr + 3 + [0, 1, 2, 3]`이라는 주소 묶음을 만든다. 마스크가 마지막 위치의 읽기를 막으므로 `values`는 `[40, 50, 60, 0]`이 된다. stride의 단위는 바이트가 아닌 **원소 수**다.

이 예제는 인자 전달과 주소 계산을 보여주기 위한 것이다. 결과를 저장하지 않으므로 실제 컴파일에서는 사용되지 않는 load가 제거될 수 있다.

**손계산 정답: `N=10`, `BLOCK_SIZE=4`, `grid=(3,)`**

```text
원소       [0 1 2 3] [4 5 6 7] [8 9 · ·]
담당 pid       0         1         2
                                  ↑ 범위 밖은 읽기·쓰기 생략
```

| pid | offsets | mask (`offsets < 10`) |
|---|---|---|
| 0 | 0, 1, 2, 3 | T, T, T, T |
| 1 | 4, 5, 6, 7 | T, T, T, T |
| 2 | 8, 9, 10, 11 | T, T, F, F |

`mask=False`인 load는 메모리를 읽지 않고 `other`를 반환하며, store는 쓰지 않는다. `other`를 생략하면 해당 값은 미정이다. `BLOCK_SIZE: tl.constexpr`는 `tl.arange(0, BLOCK_SIZE)`의 블록 shape을 컴파일할 때 확정한다. 이 예제에서는 2의 거듭제곱을 쓴다. [Vector Add: Compute Kernel](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#compute-kernel) · [arange](https://triton-lang.org/main/python-api/generated/triton.language.arange.html) · [load](https://triton-lang.org/main/python-api/generated/triton.language.load.html) · [store](https://triton-lang.org/main/python-api/generated/triton.language.store.html)

핵심 용어: **pointer · offsets · mask**

## ③ Fusion: 같은 계산, 적은 HBM 이동

Vector Add는 이미 입력 둘을 읽고 출력 하나를 쓰는 단일 연산이다. FP32 기준 `Q=12N Byte`여서 Triton으로 바꾼 것만으로 큰 개선을 기대하기 어렵다. Softmax는 중간 텐서 왕복을 없앨 여지가 크다.

Softmax는 한 행의 값들을 **합이 1인 확률 분포**로 변환한다. 한 행에 원소가 $N$개 있을 때, $i$번째 출력은 다음과 같다.

$$
y_i = \frac{\exp(x_i)}{\sum_{j=0}^{N-1}\exp(x_j)}
$$

실제 계산에서는 지수 함수의 값이 너무 커지는 것을 막기 위해 행의 최댓값 $m$을 먼저 뺀다. 모든 원소에서 같은 값을 빼므로 Softmax 결과는 같다.

$$
\begin{aligned}
m &= \max_{0 \le j < N} x_j \\
z_i &= x_i - m \\
e_i &= \exp(z_i) \\
d &= \sum_{j=0}^{N-1} e_j \\
y_i &= \frac{e_i}{d}
\end{aligned}
$$

$x_i$는 입력 원소, $m$은 행의 최댓값, $z_i$는 최댓값을 뺀 값, $e_i$는 그 값의 지수, $d$는 행의 지수 값 합이다. 아래 그림의 `max → subtract → exp → sum → divide`가 이 계산 순서에 해당한다.

```text
단계별: HBM X → max → HBM m → subtract → HBM z
                          HBM z → exp → HBM e
                          HBM e → sum → HBM d
                       HBM e,d → divide → HBM Y

결합:   HBM X → [온칩: max → subtract → exp → sum → divide] → HBM Y
```

**가정:** `M=4096`, `N=1024`, FP32, 입력은 장치에 있음. 커널 간 캐시 적중·spill 없음. 길이 M 벡터는 broadcast할 때 한 번 읽는 것으로 센다.

| 단계 | HBM 읽기 원소 | HBM 쓰기 원소 | 총 이동 Byte |
|---|---:|---:|---:|
| 행별 max | MN | M | 16,793,600 |
| `x - max` | MN + M | MN | 33,570,816 |
| exp | MN | MN | 33,554,432 |
| 행별 sum | MN | M | 16,793,600 |
| `exp / sum` | MN + M | MN | 33,570,816 |
| **단계별 합계** | **5MN + 2M** | **3MN + 2M** | **134,283,264** |
| **결합 커널** | **MN** | **MN** | **33,554,432** |

공식 튜토리얼의 논리적 이동량 모델이다. `x.max(dim=1)`의 인덱스 출력·리덕션 임시 버퍼·추가 트랜잭션은 생략한다. 실제 PyTorch 커널 수와 HBM 트래픽은 구현에 따라 달라진다. [Fused Softmax: Motivations](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#motivations)

| 계산 항목 | 단계별 | 결합 |
|---|---:|---:|
| Q, 길이 M 벡터 생략 | `8MN × 4` = 134,217,728 Byte | `2MN × 4` = 33,554,432 Byte |
| Q, 길이 M 벡터 포함 | 134,283,264 Byte | 33,554,432 Byte |
| `t_mem = Q / 10¹²` | **0.134283264 ms** | **0.033554432 ms** |

첫 번째 행의 “길이 M 벡터 생략”은 다음 네 가지 메모리 이동을 제외한다.

| 단계 | 생략한 이동 | 원소 수 |
|---|---|---:|
| 행별 max | 최댓값 벡터 `m` 쓰기 | M |
| `x - max` | `m` 읽기 | M |
| 행별 sum | 합 벡터 `d` 쓰기 | M |
| `exp / sum` | `d` 읽기 | M |
| **합계** | | **4M** |

각 값은 해당 행의 `N`개 원소에 공통으로 사용한다(broadcast). 이 모델에서는 행마다 한 번 읽는다고 가정한다. FP32 기준 생략한 이동량은 `4M × 4 = 65,536 Byte`이며, 두 번째 행은 이 값을 포함한다.

- 메모리 시간 비율: `4 + 2/N = 4.001953125`배, 근사 **4배**. 실측 속도 향상을 보장하지 않는다.
- 같은 F를 세면 `I=F/Q`는 약 4배 증가한다. 가상 `P=20×10¹² FLOPs/s`, `BW=10¹² Byte/s`의 교점은 **20 FLOPs/Byte**다.
- 전체 이상적 시간은 `max(F/P, Q/BW)`다. exp·max를 FP32 FMA와 동일하게 세어 성능을 예측하지 않는다. 위 표는 **메모리 시간**이다.

연산마다 사용하는 실행 유닛과 처리량, 필요한 명령 수가 다르기 때문이다.

| 연산 | 계산 방식과 비용 |
|---|---|
| FP32 FMA | `a × b + c`를 한 명령으로 계산하며, 관례적으로 2 FLOPs로 센다. |
| `exp` | NVIDIA GPU에서는 보통 SFU(특수 함수 유닛)의 지수 근사 명령과 추가 연산으로 구현되며, FMA와 처리량이 다르다. 정확한 명령 구성은 컴파일 결과에 따라 달라진다. |
| 행별 `max` | 여러 원소를 비교해 하나의 최댓값으로 모으는 리덕션이다. 비교 연산 외에 스레드 사이의 데이터 교환도 필요할 수 있다. |

따라서 **`exp`나 행별 `max`를 단순히 몇 FLOPs로 세고 GPU의 최대 FP32 TFLOPS로 나누면 실행 시간을 제대로 예측할 수 없다.** `F/P`로 계산 시간을 추정하려면 연산 구성에 맞는 처리량 모델이 필요하다. [근거: CUDA Programming Guide — Arithmetic Instructions](https://docs.nvidia.com/cuda/archive/11.5.2/cuda-c-programming-guide/index.html#arithmetic-instructions)

핵심 용어: **kernel fusion · HBM 이동량 Q · 산술 강도 I**

## ④ 행 리덕션: 왜 `other=-inf`인가?

앞에서는 Softmax를 하나의 커널로 합치면 HBM 이동량이 줄어드는 이유를 살펴봤다. 이제 그 커널 안에서 **행별 최댓값과 합을 어떻게 계산하는지** 살펴본다. 여러 값을 하나로 모으는 연산을 **리덕션(reduction)**이라고 하며, Softmax의 행별 `max`와 `sum`이 여기에 해당한다.

```text
한 행의 N개 값
    → max 리덕션 → 최댓값 1개
    → 최댓값을 broadcast해서 각 원소에서 빼기 → N개 값에 exp 적용
    → sum 리덕션 → 합 1개
    → 합을 broadcast해서 각 원소를 나누기 → 출력 N개
```

**리덕션은 여러 값을 하나로 모으고, broadcast는 그 값을 여러 원소 계산에 사용한다.** 여기서 broadcast는 값을 행이나 열에 맞게 펼쳐 사용하는 논리적인 개념이며, 펼친 크기의 배열을 실제로 복사해 만들 필요는 없다.

아래 코드는 프로그램 하나가 행 하나를 처리하며, `tl.max`와 `tl.sum`으로 그 행을 리덕션한다. 행 길이를 블록 크기에 맞추면서 생기는 패딩도 리덕션에 참여하므로, **패딩을 어떤 값으로 채우는지**가 중요하다.

```python
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, input_row_stride,
                   output_row_stride, N, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)  # 프로그램 하나가 행 하나 담당
    cols = tl.arange(0, BLOCK_SIZE)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    row = tl.load(row_start_ptr + cols, mask=cols < N,
                  other=-float("inf"))
    e = tl.exp(row - tl.max(row, axis=0))
    y = e / tl.sum(e, axis=0)
    output_row_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_ptr + cols, y, mask=cols < N)


M, N = 1823, 781
x = torch.randn((M, N), device="cuda", dtype=torch.float32)
y = torch.empty_like(x)

BLOCK_SIZE = triton.next_power_of_2(N)  # 781개 원소 → 1024칸
grid = (M,)                           # 행마다 하나씩, 프로그램 1823개
softmax_kernel[grid](
    x, y,
    x.stride(0), y.stride(0),
    N,
    BLOCK_SIZE=BLOCK_SIZE,
    num_warps=4,
)

torch.testing.assert_close(y, torch.softmax(x, dim=1),
                           rtol=1e-5, atol=1e-8)
```

`BLOCK_SIZE=triton.next_power_of_2(N)`로 행을 담는 최소 2의 거듭제곱을 택한다. `N=781`이면 1024칸 중 243칸이 패딩이다. 열 stride는 1, 행 stride는 **원소 단위**다.

이 코드의 `tl.arange(0, BLOCK_SIZE)`는 원소 수가 2의 거듭제곱이어야 한다. 이 제약은 **warp를 포함한 GPU의 이진 구조에 맞춰 데이터 배치와 코드 생성을 단순하고 효율적으로 하는 설계**와 연결된다. Triton 개발진의 Linear Layouts 논문 §4.1은 warp의 32·64개 스레드처럼 GPU의 여러 구성 요소가 2의 거듭제곱이라는 점에서 출발해, 원소를 레지스터·스레드·warp에 배치하는 관계를 인덱스의 비트로 표현한다.

예를 들어 NVIDIA GPU에서 1024개 원소를 4개 warp에 균등하게 배분하는 단순한 배치는 다음과 같이 표현할 수 있다.

```text
1024개 = 스레드당 8개 × warp당 32스레드 × 4 warp
       =     2³      ×       2⁵       ×   2²

10비트 원소 인덱스 = [warp 2비트 | lane 5비트 | 스레드 내 원소 3비트]
```

원소 번호 `0~1023`은 10비트로 표현할 수 있다. 위 예시는 그 비트를 나눠 **어느 warp의 어느 스레드가, 자신이 맡은 원소 중 어느 것을 처리하는지** 나타낸 것이다.

| 구분 | 범위 | 의미 |
|---|---|---|
| warp 번호 | 0~3 | 4개 warp 중 어느 warp인가? |
| lane 번호 | 0~31 | 그 warp 안의 어느 스레드인가? |
| 스레드 내 원소 번호 | 0~7 | 그 스레드가 맡은 8개 원소 중 어느 것인가? |


예를 들어 원소 번호 299를 이 배치에 따라 나누면 다음과 같다.

```text
299 = 01 | 00101 | 011  (2진수)
       1      5      3
     warp   lane   스레드 내 원소
```

즉, **warp 1의 lane 5가 맡은 원소 8개 중 3번 원소**에 해당한다. 전체 텐서에서의 위치는 이 세 번호와 레이아웃을 함께 사용해 결정한다. 위 예시는 스레드 하나에 연속 원소 8개를 배정한 경우다. 반면 `[스레드 내 원소 3비트 | warp 2비트 | lane 5비트]`처럼 lane 비트를 아래쪽에 두면, 같은 내부 원소 번호에 대해 이웃 lane이 연속 주소를 담당한다.

이처럼 크기를 맞추면 배치와 재배치를 비트 연산으로 표현하기 쉽고, 컴파일러가 다양한 레이아웃을 일관되게 처리할 수 있다. 위 배치는 설명용 예이며, 실제 비트 배정은 컴파일러가 선택한 레이아웃에 따라 달라진다. 이는 현재 레이아웃 설계의 근거이며, 최초 Triton의 크기 제약이 오직 이 이유로 생겼다는 뜻은 아니다. [Linear Layouts §4.1](https://arxiv.org/html/2505.23819v1#S4.SS1) · [구현의 설계 주석](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h)

패딩 자체는 불필요한 연산을 늘릴 수 있으므로, 이 제약 안에서 행 전체를 담는 **가장 작은** 2의 거듭제곱을 고른다. 여기서 `BLOCK_SIZE`는 스레드 수가 아닌 원소 수이므로 1024를 넘을 수 있다. 예를 들어 `BLOCK_SIZE=4096`, `num_warps=4`라면 128개 스레드가 4096개 원소를 나눠 처리한다. 다만 실제 사용 가능한 크기는 레지스터·공유 메모리 등의 자원 한도에도 영향을 받는다.

**781개 원소가 모두 -2인 예**

| 패딩 | `tl.max` | 지수 합계 | 유효 출력 한 칸 | 유효 출력 합 |
|---|---:|---:|---:|---:|
| `-inf` | -2 | 781 | `1/781 ≈ 0.00128041` | 1 |
| 0 | 0 | `781×exp(-2)+243 ≈ 348.69686` | `≈ 0.00038812` | `≈ 0.30312` |

패딩을 **`-inf`로 채우면 최댓값에 영향을 주지 않고, `exp`를 거친 뒤에는 0이 되어 합에도 영향을 주지 않는다.** 반면 0으로 채우면 입력이 모두 음수일 때 최댓값이 바뀔 수 있고, `exp(0 - 최댓값)`도 양수이므로 분모에 불필요한 값이 더해진다.

1024 FP32 원소의 최소 입력 저장량은 **4096 Byte = 4 KiB**다. 실제로는 패딩·중간값·리덕션에도 공간이 필요하다. 행이 커지면 레지스터 사용 증가·occupancy 감소·spill 또는 자원 한도 초과가 생겨 분할 리덕션 등 다른 방식이 필요하다.

공식 현재 예제는 프로그램이 여러 행을 순회하는 persistent 방식이다. 실습은 읽기 쉽게 **한 프로그램=한 행**, `grid=(M,)`으로 단순화했다. [Fused Softmax: Compute Kernel](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#compute-kernel)

핵심 용어: **row stride · reduction · padding**

## ⑤ 정확도와 벤치마크

Python 함수가 반환되어도 비동기 GPU 작업은 진행 중일 수 있다. 측정할 때와 같은 입력 크기·자료형·커널 설정으로 먼저 실행해 컴파일을 끝낸다. GPU 작업이 완료되기를 기다린 뒤, 준비된 커널의 실행 시간을 `do_bench`로 측정한다.

| 확인 | 조건 |
|---|---|
| Vector Add | `N=10`, `BLOCK_SIZE=4`, 마지막 블록 검사 |
| Softmax 정확도 | FP32 `[1823,781]`, `BLOCK_SIZE=1024`, `torch.allclose(rtol=1e-5, atol=1e-8)` |
| 패딩 회귀 확인 | 음수만 있는 행, 출력 행의 합 ≈ 1 |
| 속도 | FP32 `[4096,1024]`, naive·torch.softmax·Triton, 선택적으로 torch.compile |
| 측정 | `do_bench(fn, warmup=25, rep=100, return_mode="median")` |

> `torch.allclose`는 직접 만든 Triton 커널의 결과가 기준 구현인 PyTorch와 허용 오차 안에서 일치하는지 확인하기 위해 사용한다. 부동소수점 계산은 연산 순서나 `exp`의 근사 방식에 따라 작은 차이가 생길 수 있으므로, `==`로 완전히 같은지 검사하는 대신 허용 오차를 적용한다. 속도를 비교하기 전에 계산 결과를 검증하는 단계이며, 검사한 입력에 대한 확인이지 모든 입력에서의 정확성을 증명하는 것은 아니다. 위 커널 실행 예제의 `torch.testing.assert_close`도 허용 오차로 결과를 비교하며, 일치하지 않으면 예외와 함께 차이를 보고한다.

>`rtol`은 기준값의 크기에 비례해 허용하는 **상대 오차**, `atol`은 기준값의 크기와 관계없이 허용하는 **절대 오차**다. `torch.allclose(actual, expected, rtol=1e-5, atol=1e-8)`는 모든 원소가 `|actual - expected| ≤ atol + rtol × |expected|`를 만족하는지 검사한다. 예를 들어 기준값이 `0.1`이면 허용 오차는 `1e-8 + 1e-5 × 0.1 = 1.01e-6`이다. 기준값이 0에 가까울 때는 `atol`이 작은 수치 차이를 허용하는 역할을 한다. 허용 오차 안에서 일치한다는 뜻이며, 비트 단위로 동일하다는 뜻은 아니다.

>`triton.testing.do_bench`는 Triton에서 제공하는 벤치마크 함수다. 전달한 함수를 반복 실행하며 GPU 실행 시간을 측정한다. Triton 커널뿐 아니라 PyTorch 연산도 측정할 수 있고, 위 설정에서는 실행 시간의 중앙값을 밀리초(ms)로 반환한다.

`warmup`, `rep`는 **반복 횟수가 아니라 ms 단위 시간 예산**이며, 이를 바탕으로 반복 수를 정한다. 입력 생성·장치 전송·정확도 검사는 측정 밖이다. 할당을 포함한 연산 함수를 호출하며, event 사이의 커널 실행·제출 간격을 측정한다. [do_bench API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) · [구현](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py)

```text
실효 대역폭 [GB/s] = (2 × M × N × 4 Byte) / (t_ms × 10⁻³) / 10⁹
                   = 33.554432 / t_ms       # 이번 입력
```

분자는 모든 구현에 공통인 **필수 입력 읽기+최종 출력 쓰기**다. naive의 실제 이동량을 뜻하지 않는다. 실효 대역폭과 하드웨어 HBM 처리량은 구분한다. `torch.softmax`도 최적화된 구현이고 `torch.compile`은 fusion을 할 수 있으므로 naive보다 차이가 작아질 수 있다. [PyTorch compiler 개요](https://docs.pytorch.org/docs/stable/torch.compiler.html)

GPU가 없으면 노트북 첫 설정에서 `INTERPRET=True`로 바꾸고 작은 입력만 확인한다. import 전에 `TRITON_INTERPRET=1`을 설정해야 한다. NumPy 기반 순차 해석이며 bfloat16·간접 포인터 접근에 제약이 있다. 시간 비교에는 쓰지 않는다. [Debugging: Using the Interpreter·Limitations](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html#using-the-interpreter)

핵심 용어: **동기화 · warmup · 실효 대역폭**

## 함께 기록할 것

| 항목 | 기록 |
|---|---|
| 맡은 질문 / 핵심 용어 3개 | |
| 인덱싱 또는 HBM↔온칩 그림 | 위 그림에 직접 주석 달기 |
| GPU / Python / PyTorch / Triton / CUDA 버전 | 미실행이면 ‘손계산’ |
| shape / dtype / BLOCK_SIZE / 측정 범위 | |
| 구현별 ms / 실효 GB/s / allclose 결과 | |
| 읽은 링크·절 / 함께 이야기할 질문 | |

**최댓값을 다르게 잡더라도, 분모에 실제 원소만 포함하면 수학적으로 Softmax 결과는 같다.** 예를 들어 실제 입력이 `[-2, -2]`인데 패딩 0 때문에 최댓값이 0으로 잡혔다고 하자.

```text
실제 최댓값 -2를 빼는 경우:
exp([-2, -2] - (-2)) = [1, 1]
출력 = [1/2, 1/2]

0을 빼되 패딩을 분모에서 제외하는 경우:
exp([-2, -2] - 0) = [exp(-2), exp(-2)]
출력 = [1/2, 1/2]
```

모든 원소에서 같은 상수 `c`를 빼면 지수 값에 공통 인자 `exp(-c)`가 생기고, 분자와 분모에서 약분된다. 이를 **Softmax의 상수 이동 불변성**이라고 한다.

따라서 앞의 패딩 예제에서 결과가 틀어지는 직접적인 이유는 **최댓값이 바뀌어서가 아니라, 패딩의 지수 값까지 분모에 더해지기 때문**이다. 다만 실제 부동소수점 계산에서는 어떤 상수를 빼느냐가 수치 안정성에 영향을 주므로, 실제 최댓값을 사용하는 것이 좋다.

공식 문서 확인: 2026-10-23. 이 문서의 시간은 가상 하드웨어 손계산이며 GPU 실측값이 아니다.
