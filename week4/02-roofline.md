# ② Roofline: 어느 자원이 성능을 제한할까?

> **한 줄 답.** 데이터 공급으로 가능한 처리율 `BW×I`와 연산 장치의 최대 처리율 P 중 작은 값이 성능 상한이다.

이전 ← [① Arithmetic Intensity](./01-arithmetic-intensity.md) · [4주차 목차](./README.md) · 다음 → [③ Tiling](./03-tiling.md)

## 1. 두 시간을 하나의 성능 그래프로 연결하기

지난주의 두 시간은 다음과 같다.

$$
t_{\mathrm{compute}}=\frac{F}{P},\qquad
t_{\mathrm{memory}}=\frac{Q}{BW}
$$

최대 성능을 쓰고 연산·이동이 충분히 겹친다고 두면 실행시간의 이상적 하한은 두 값 중 큰 값이다. 처리율은 같은 연산량 F를 시간으로 나눈 값이므로 다음처럼 바뀐다.

$$
R_{\mathrm{roof}}
=\frac{F}{\max(F/P,Q/BW)}
=\min\left(P,\frac{F}{Q/BW}\right)
=\min(P,BW\times I)
$$

시간에서는 **max**, 처리율에서는 **min**이다. 같은 모델을 다른 방향에서 읽는 것이다. 두 시간과 산술 강도의 연결은 [Scaling Book: Where Does the Time Go? · Visualizing rooflines](https://jax-ml.github.io/scaling-book/roofline/)을 참고한다.

## 2. 축과 두 선 읽기

![가상 FP32 하드웨어 Roofline. 가로축은 HBM 산술 강도, 세로축은 TFLOPs/s이며 양쪽 모두 로그 축이다.](./assets/roofline.png)

| 그래프 요소 | 뜻 | 단위·읽는 법 |
|---|---|---|
| 가로축 I | HBM 이동 1 Byte당 계산량 | FLOPs/Byte |
| 세로축 R | 초당 부동소수점 연산 처리율 | 그림은 TFLOPs/s, 1 TFLOPs/s = 10¹² FLOPs/s |
| 기울어진 선 BW×I | HBM이 공급할 수 있는 데이터로 가능한 처리율 | Byte/s × FLOPs/Byte = FLOPs/s |
| 수평선 P | 연산 장치가 낼 수 있는 최대 처리율 | I가 커져도 주어진 P는 일정 |
| 꺾이는 점 I* | 두 상한이 같아지는 산술 강도 | P/BW, FLOPs/Byte |

HBM이 초당 BW Byte를 옮기고 Byte당 I FLOPs를 수행한다면 초당 최대 `BW×I FLOPs`를 공급할 수 있다. I가 두 배가 되면 이 상한도 두 배가 되어 기울어진 선이 된다. 반면 P는 연산 장치가 처리할 수 있는 고정된 한계이므로 수평선이다.

그림은 **로그–로그 축**이다. 이때 `log R = log BW + log I`이므로 메모리 선의 기울기는 1이고, BW가 커지면 선이 위로 이동한다. 선형 축에서 `R = BW×I`의 기울기는 BW다. 로그 축의 기울기 숫자를 대역폭 자체로 읽지 않는다.

[Nsight Compute: Roofline Charts — Overview](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)의 그림에서도 축·메모리 경계·연산 경계·Ridge Point를 찾아보자. 실제 프로파일러의 측정점과 상한선을 구분하는 것이 중요하다.

## 3. 가상 하드웨어의 분기점

$$
I_* = \frac{20\times10^{12}\ \mathrm{FLOPs/s}}
{1\times10^{12}\ \mathrm{Byte/s}}
=20\ \mathrm{FLOPs/Byte}
$$

- `I < 20`: 이 모델에서는 HBM 대역폭 상한이 더 낮다.
- `I = 20`: 두 자원의 이상적 시간이 같다.
- `I > 20`: 이 모델에서는 연산 성능 상한이 더 낮다.

그래프의 네 점은 [4096×4096 문제](./05-matmul-exercise.md)의 **계산된 상한**이다. 재사용 없음·32 타일·64 타일은 각각 약 0.250·7.969·15.876 TFLOPs/s다. 128 타일의 I는 약 31.508이지만 처리율 상한은 20 TFLOPs/s에서 멈춘다. 점을 그래프 위에 놓은 것은 실제 측정에서 상한을 달성했다는 뜻이 아니다.

## 4. 그래프에서 시간을 구하기

같은 행렬곱의 `F = 137,438,953,472 FLOPs`를 사용하면 다음과 같다.

| 경우 | 성능 상한 (TFLOPs/s) | Roofline 예상 시간 (ms) |
|---|---:|---:|
| 재사용 없음 | 0.249969 | 549.822923 |
| T = 32 | 7.968872 | 17.246978 |
| T = 64 | 15.875969 | 8.657043 |
| T = 128 | 20.000000 | 6.871948 |

세로축 값이 높을수록 **같은 F**를 처리하는 시간은 짧다. 서로 다른 연산량의 커널은 처리율만으로 실행시간을 비교할 수 없다.

## 5. 상한과 실제 성능은 다르다

이 모델은 메모리 지연, 동기화, 명령어 부가 비용, 온칩 대역폭, 자원 사용에 따른 동시 실행량 등을 별도 계산하지 않는다. 실제 커널은 상한 아래에 놓일 수 있으며, 상한과의 차이를 모두 HBM 병목 때문이라고 단정할 수 없다. 최대 P와 BW의 단위뿐 아니라 해당 자료형·연산 경로와 메모리 구간도 맞춰야 한다.

**함께 이야기할 질문:** I가 10인 커널에서 P만 두 배로 높이면 Roofline 상한이 변할까? 같은 커널의 BW를 두 배로 높이면 어떻게 될까? (현재 조건에서 각각 10 TFLOPs/s 유지, 20 TFLOPs/s로 증가.)
