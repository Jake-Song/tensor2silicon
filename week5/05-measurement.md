# ⑤ 실제로 빠르다는 뜻: 무엇의 시간을 재었을까?

[5주차 목차](./README.md) · 이전 → [④ 소프트웨어](./04-software.md) · 다음 → [⑥ 조건표·계산](./06-comparison-exercise.md)

## 1. 시간의 시작과 끝

```mermaid
flowchart LR
    C["초기화 · 모델 읽기 · 컴파일"] --> W["준비 실행"]
    W --> Q["요청 도착 · 대기"]
    Q --> P["입력 준비 · 전송/변환"]
    P --> D["장치에서 그래프 실행"]
    D --> R["완료 대기 · 출력 회수"]
    R --> O["후처리 · 응답"]
```

직렬 경로를 보여주는 그림이다. 비동기 서비스에서는 전송·계산·다른 요청이 겹칠 수 있다.

GPU event로 얻은 시간과 JAX `block_until_ready()`를 포함한 호스트 시간을 바로 비교하지 않는다. NPU에 동등한 장치 타이머가 없다면 세 장치 모두 **동기화된 호출 시간**이나 **전체 요청 시간**처럼 공통으로 구현 가능한 범위로 맞춘다. 

## 2. 배치 지연시간과 처리량

이 과제에서 X의 **한 행을 sample 하나**, M행의 호출을 **batch 하나**로 정의한다.

즉, `X`의 shape이 `[M, K]`라면 각 행은 특성 K개를 가진 입력 하나다. `XW+b`는 각 행에 같은 가중치 `W`와 bias `b`를 적용해 대응하는 출력 행을 만든다. 따라서 한 번의 호출에서 M개 입력을 함께 처리하며 batch size는 M이다. `M=1`이면 1-sample batch, `M=128`이면 128-sample batch다.

$$
\text{batch latency}=t_M\ [\mathrm{ms/batch}],\qquad
\text{samples/s}=\frac{R\times M}{T\ [\mathrm{s}]}
$$

- R: 측정 구간에 완료한 **호출 수**
- M: 배치  


| 배치 | 지연시간 기록 | 처리량 기록 | 해석할 때 주의할 점 |
|---|---|---|---|
| M=1 | ms / 1-sample batch, p50·p95 등 | samples/s, 요청 하나=한 행이면 requests/s도 가능 | 작은 작업은 제출·동기화 비용이 지배할 수 있음 |
| M=128 | ms / 128-sample batch, p50·p95 등 | samples/s와 batches/s를 구분 | `t₁₂₈ / 128`은 분담된 시간이며 한 요청이 기다린 시간이 아님 |

## 4 토큰 당 전력 에너지

$$
E=\int_{t_0}^{t_1}P(t)\,dt\ [\mathrm{J}],\qquad
E_{\mathrm{sample}}=\frac{E}{\text{완료한 sample 수}}
$$

전력은 단위 시간당 에너지이므로 `P(t)=dE/dt`다. 아주 짧은 시간 `dt` 동안 소비한 에너지는 `dE=P(t)dt`이고, 이를 측정 시작 `t₀`부터 끝 `t₁`까지 누적하면 `E=∫[t₀,t₁]P(t)dt`가 된다. 여기서 J(줄, joule)는 에너지의 SI 단위다. 

실제 계측에서는 시간 간격 `Δt`마다 전력을 읽어 `E≈ΣᵢPᵢΔt`로 근사한다. 얻은 에너지를 같은 측정 구간에 완료한 sample 수로 나누면 sample 하나당 에너지(J/sample)가 된다. 

예를 들어 평균 100 W로 2초 동안 실행해 sample 200개를 완료했다면 `E=100×2=200 J`, 즉 `E_sample=200/200=1 J/sample`이다.
