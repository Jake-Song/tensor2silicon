# 4주차: Roofline과 Tiling

> **한 줄 답.** 같은 계산을 하더라도 HBM에서 데이터를 덜 가져오면 산술 강도가 높아져 성능 상한이 올라간다. 다만 연산 성능의 한계와 온칩 저장 공간의 제약이 있다.

지난주에는 연산 시간과 메모리 시간을 따로 계산했다. 이번에는 두 시간을 **Roofline 그래프**로 연결하고, **Tiling으로 입력을 재사용하는 방법**을 살펴본다. 중심 질문은 “같은 계산을 하면서 데이터를 덜 읽어오면, 커널은 얼마나 빨라질 수 있을까?”다.

그림과 손계산으로 원리를 익히는 주차다. 전체 GPU 커널 구현은 뒤의 Triton·Tiled MatMul 주차에서 이어간다. 입력은 이미 장치에 있고, 하나의 GPU 안에서 HBM을 읽고 쓰는 이동만 센다. 다중 장치 통신은 이번 범위에서 제외한다.

## 1. 학습 목표와 읽는 순서

| 순서 | 문서 | 중심 질문 |
|---|---|---|
| ① | [Arithmetic Intensity](./01-arithmetic-intensity.md) | 읽고 쓴 1 Byte로 얼마나 계산하는가? |
| ② | [Roofline](./02-roofline.md) | 기울어진 선과 수평선은 각각 무엇을 제한하는가? |
| ③ | [Tiling과 4×4 그림](./03-tiling.md) | 입력 한 번을 읽어 여러 출력을 어떻게 계산하는가? |
| ④ | [타일 크기와 저장 공간·Occupancy](./04-tile-size-and-occupancy.md) | 더 큰 타일의 이점과 비용은 무엇인가? |
| ⑤ | [4096×4096 FP32 MatMul 손계산](./05-matmul-exercise.md) | 재사용 없음과 32·64·128 타일은 얼마나 다른가? |

이전 주차 → [3주차 안내](../week3/README.md) · [3주차 MatMul 풀이](../week3/07-matmul.md)

## 2. 공통으로 읽어올 자료

| 자료 | 읽을 절과 범위 | 확인할 내용 |
|---|---|---|
| [Scaling Book 1장: All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | **Where Does the Time Go?**의 두 시간 공식 복습, Arithmetic Intensity 정의부터 **Visualizing rooflines**까지 | 가로축·세로축, 메모리 상한·연산 상한, 두 선의 교점 |
| [MLC: GPU Acceleration — Local Blocking](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#local-blocking) | **Local Blocking**의 그림과 첫 설명 | 출력 일부를 계산할 때 입력을 여러 번 사용하는 방법 |
| [MLC: Shared Memory Blocking](https://book.mlc.ai/chapter_gpu_acceleration/part1.html#shared-memory-blocking) | 해당 절의 그림과 첫 설명 | 같은 블록의 스레드들이 입력 조각을 공유하는 방법 |

TVM 스케줄 코드 전체를 이해하거나 실행할 필요는 없다. MLC는 본문을 확인한 `book.mlc.ai` 주소를 사용한다.

## 3. 주제별 보조 자료

| 자료 | 읽을 범위 | 연결할 질문 |
|---|---|---|
| [NVIDIA Nsight Compute: Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts) | **Overview** 그림, 축과 Ridge Point 설명 | 상한선과 실제 측정점은 어떻게 다른가? |
| [CUDA Best Practices: Calculating Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy) | 레지스터·Shared Memory가 상주 블록 수를 제한하는 설명 | 데이터를 더 보관하면 동시에 실행할 작업은 어떻게 되는가? |
| [Triton: Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#compute-kernel) | **Compute Kernel**의 첫 블록 알고리즘 | 출력 타일을 고정하고 K 방향으로 입력 조각을 바꾸는 과정 |

Triton 자료의 구현 예제는 FP16 행렬곱이다. 이번 주는 **블록 알고리즘**을 읽고, 계산표에는 아래 FP32 조건을 사용한다. 뒤의 L2 캐시 최적화·autotuning은 필수 범위가 아니다.

## 4. 함께 사용할 기호와 모델

| 기호 | 뜻 | 단위 |
|---|---|---|
| F | 총 부동소수점 연산량, FMA 한 번을 2 FLOPs로 셈 | FLOPs |
| Q | HBM에서 읽고 쓴 총 이동량 | Byte |
| I = F/Q | HBM 기준 산술 강도 | FLOPs/Byte |
| P | 해당 자료형·연산에 적용 가능한 최대 연산 성능 | FLOPs/s |
| BW | HBM 대역폭, 행렬 B와 구분 | Byte/s |
| T | 정사각 출력 타일의 한 변, 이번 문제는 K 조각 폭도 T | 원소 개수 |

$$
R_{\mathrm{roof}}=\min(P, BW\times I),\qquad
I_* = \frac{P}{BW}
$$

$$
t_{\mathrm{ideal}} = \frac{F}{R_{\mathrm{roof}}}
=\max\left(\frac{F}{P},\frac{Q}{BW}\right)
$$

그래프의 세로축은 **초당 처리율**이다. 선은 성능 상한이며, 이상적 시간은 주어진 모델의 실행시간 하한이다. 실제 커널이 그 성능을 달성한다는 뜻은 아니다.

## 5. 함께 계산할 문제

먼저 [4×4 행렬곱을 2×2 출력 타일로 나누는 그림](./03-tiling.md)을 그린다. 이어서 A·B·C가 모두 **4096×4096 FP32**인 `C = A @ B`를 계산한다.

- 원소와 중간 합은 FP32, 원소당 **4 Byte**다.
- 가상 하드웨어는 이 FP32 연산에 적용 가능한 `P = 20 × 10¹² FLOPs/s`, `BW = 1 × 10¹² Byte/s`다.
- 재사용 없음: 곱셈할 때마다 A·B 원소를 HBM에서 다시 읽는다.
- 타일 안에서 재사용: 출력 타일과 K 방향 입력 조각의 폭을 모두 `T = 32, 64, 128`로 바꾼다.
- 입력은 이미 장치에 있고 중간 합은 온칩에 유지한다. 기존 C는 읽지 않고, 최종 C만 한 번 쓴다.
- 서로 다른 출력 타일 사이의 캐시 재사용은 없다. 추가 이동·레지스터 spill은 없다고 가정한다.
- 버퍼는 A·B 입력 조각을 한 벌만 보관하는 **단일 버퍼** 기준이다.

각 경우의 FLOPs, HBM 이동량, 산술 강도, Roofline 상한, 두 시간과 이상적 실행시간, 입력 버퍼와 중간 합 크기를 구한다. [풀이와 비교표](./05-matmul-exercise.md)에서 답을 확인한다. 실제 제품의 측정값을 예측하기 위한 조건이 아니라 재사용 효과를 구분하기 위한 단순화다.

![HBM 산술 강도에 따른 성능 상한: 교점은 20 FLOPs/Byte이고 128 타일만 연산 상한에 도달한다. 점은 측정값이 아닌 모델 상한이다.](./assets/roofline.png)

## 6. 각자 정리해 올 내용

1. 맡은 질문의 설명과 핵심 용어를 `C = A @ B` 예제로 연결한다.
2. 축·단위·분기점이 표시된 Roofline 그림 또는 입력 재사용이 보이는 타일 그림을 그린다.
3. 연산량·이동량·산술 강도·예상 시간을 계산표로 만든다.
4. 자료형, 세어 본 메모리 구간, 캐시·재사용·중간 합 보관에 관한 가정을 적는다.
5. 참고 링크와 읽은 절, 함께 이야기하고 싶은 질문을 남긴다.

## 7. 모임에서 서로 확인할 질문

- Roofline의 두 상한과 교점의 단위를 설명할 수 있는가?
- Tiling 전후에 같은 FLOPs와 달라진 HBM Byte를 각각 셀 수 있는가?
- Vector Add를 조각내기만 해도 MatMul처럼 산술 강도가 높아질까?
- 64에서 128 타일로 키울 때 이동량이 거의 절반이 되어도 시간이 절반이 되지 않는 이유는 무엇인가?
- `128×128 출력 타일`이 `16,384개 스레드`를 뜻하지 않는 이유는 무엇인가?
- 큰 타일의 재사용 이점과 Shared Memory·레지스터·동시 실행량의 비용을 함께 설명할 수 있는가?

그림은 이 문제의 수치로 직접 작성했다. [그림 생성 스크립트](./make_figures.py)는 학습 자료 재생성용이며 GPU 커널이나 벤치마크가 아니다.
