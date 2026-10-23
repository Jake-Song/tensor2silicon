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

`torch.Tensor`는 커널에 **자료형이 있는 데이터 포인터**로 전달된다. shape·stride는 필요한 값을 별도 인자로 넘긴다. 포인터에 `1`을 더하면 다음 **원소**로 이동한다(FP32는 4 Byte). `offsets`는 정수 블록, `x_ptr + offsets`는 포인터 블록이다. [근거: triton.jit](https://triton-lang.org/main/python-api/generated/triton.jit.html)

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

- 메모리 시간 비율: `4 + 2/N = 4.001953125`배, 근사 **4배**. 실측 속도 향상을 보장하지 않는다.
- 같은 F를 세면 `I=F/Q`는 약 4배 증가한다. 가상 `P=20×10¹² FLOPs/s`, `BW=10¹² Byte/s`의 교점은 **20 FLOPs/Byte**다.
- 전체 이상적 시간은 `max(F/P, Q/BW)`다. exp·max를 FP32 FMA와 동일하게 세어 성능을 예측하지 않는다. 위 표는 **메모리 시간**이다.

핵심 용어: **kernel fusion · HBM 이동량 Q · 산술 강도 I**

## ④ 행 리덕션: 왜 `other=-inf`인가?

```python
row_idx = tl.program_id(0)  # 학습용: 프로그램 하나가 행 하나 담당
cols = tl.arange(0, BLOCK_SIZE)
row_start_ptr = input_ptr + row_idx * input_row_stride
row = tl.load(row_start_ptr + cols, cols < N, other=-float("inf"))
e = tl.exp(row - tl.max(row, axis=0))
y = e / tl.sum(e, axis=0)
```

`BLOCK_SIZE=triton.next_power_of_2(N)`로 행을 담는 최소 2의 거듭제곱을 택한다. `N=781`이면 1024칸 중 243칸이 패딩이다. 열 stride는 1, 행 stride는 **원소 단위**다.

**781개 원소가 모두 -2인 예**

| 패딩 | `tl.max` | 지수 합계 | 유효 출력 한 칸 | 유효 출력 합 |
|---|---:|---:|---:|---:|
| `-inf` | -2 | 781 | `1/781 ≈ 0.00128041` | 1 |
| 0 | 0 | `781×exp(-2)+243 ≈ 348.69686` | `≈ 0.00038812` | `≈ 0.30312` |

`-inf`는 최댓값을 바꾸지 않고, 유한한 최댓값을 빼서 exp하면 0이 된다. **store에만 마스크를 씌워도 잘못된 분모는 고쳐지지 않는다.** 여기서는 유한한 입력을 다룬다.

1024 FP32 원소의 최소 입력 저장량은 **4096 Byte = 4 KiB**다. 실제로는 패딩·중간값·리덕션에도 공간이 필요하다. 행이 커지면 레지스터 사용 증가·occupancy 감소·spill 또는 자원 한도 초과가 생겨 분할 리덕션 등 다른 방식이 필요하다.

공식 현재 예제는 프로그램이 여러 행을 순회하는 persistent 방식이다. 실습은 읽기 쉽게 **한 프로그램=한 행**, `grid=(M,)`으로 단순화했다. [Fused Softmax: Compute Kernel](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#compute-kernel)

핵심 용어: **row stride · reduction · padding**

## ⑤ 정확도와 벤치마크

Python 함수가 반환되어도 비동기 GPU 작업은 진행 중일 수 있다. 실습은 같은 shape으로 미리 호출해 컴파일을 끝내고, 동기화한 뒤 `do_bench`로 GPU event 구간을 잰다.

| 확인 | 조건 |
|---|---|
| Vector Add | `N=10`, `BLOCK_SIZE=4`, 마지막 블록 검사 |
| Softmax 정확도 | FP32 `[1823,781]`, `BLOCK_SIZE=1024`, `torch.allclose(rtol=1e-5, atol=1e-8)` |
| 패딩 회귀 확인 | 음수만 있는 행, 출력 행의 합 ≈ 1 |
| 속도 | FP32 `[4096,1024]`, naive·torch.softmax·Triton, 선택적으로 torch.compile |
| 측정 | `do_bench(fn, warmup=25, rep=100, return_mode="median")` |

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

토론: **패딩이 최댓값만 바꾸고 분모에는 들어가지 않는다면 결과도 달라질까?** Softmax의 상수 이동 불변성과 연결해보자.

공식 문서 확인: 2026-10-23. 이 문서의 시간은 가상 하드웨어 손계산이며 GPU 실측값이 아니다.
