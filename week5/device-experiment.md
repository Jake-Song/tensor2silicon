# week5-06 Colab 장치 실험 — 2026-10-19

이 문서는 첫 실험 기록이다. 이후 [GPU·TPU를 모두 JAX 0.11.1로 통일한 실측](./jax-device-experiment.md)을 추가했다.

`Y = ReLU(XW+b)`, `K=N=1024`, `M=1,128`을 실제 장치에서 실행했다. TPU만 JAX/XLA, GPU는 PyTorch/CUDA를 사용한다. 최초 H100·v5e·Intel NPU 문헌 비교와 구분해 실제 할당 제품을 기록한다.

## 공통 조건

- [⑥의 A 계약](./06-comparison-exercise.md): dense BF16 X·W, FP32 누산·bias·출력. 추론만 실행.
- NumPy에서 seed 0–4의 FP32 원본을 한 번 생성하고 round-to-nearest-even으로 BF16 값을 고정했다. M=1은 M=128의 첫 행이며 W·b가 같다. BF16 값을 CPU FP64로 확장해 계산한 뒤 FP32로 반올림한 결과가 기준이다.
- 모든 장치에 동일한 `inputs.npz`를 업로드했다. SHA256: `9de23ebe967f1e1360cc7d9f4d6f7137b7e4ce012c738bc333a8a69d9bb95590`. [배열별 해시](./results/2026-10-19-colab/inputs-manifest.json).
- X·W·b는 장치에 상주하고 런타임 인수로 전달한다. 각 seed·shape에서 50회 준비 실행 후 1,000회 순차 측정한다. shape별 총 5,000개 원시 호출 시간을 합쳐 p50/p95를 계산한다.
- 시간 범위는 호스트 호출 시작부터 완료 대기까지다. TPU는 `block_until_ready()`, GPU는 `torch.cuda.synchronize()`를 사용한다. 전송·출력 회수·검증·컴파일은 정상 상태 측정 밖이다. **순수 장치 커널 시간이나 호스트 X→호스트 Y 요청 시간이 아니다.**
- 처리량은 실제 완료한 `5,000 × M`을 다섯 측정 구간의 총 경과 시간으로 나눈 값이다. 측정 루프 비용도 포함하며 동시성은 1이다.
- 같은 W를 반복하므로 캐시가 따뜻할 수 있다. cold HBM 이동량을 가정한 손계산과 직접 비교하지 않는다.
- 기준은 원소별 `abs(error) <= 1e-4 + 1e-4*abs(reference)`. NaN/Inf는 실패다. 별도로 zero X와 양수·음수·0·0 부근 bias의 broadcast/ReLU 경계도 두 shape에서 확인했다. 회수한 출력은 로컬에서 다시 검증했다.

## 실행 경로

| Colab 요청 | 실제 장치 | 실행 버전·경로 |
|---|---|---|
| v5e1 | TPU v5 lite, JAX 장치 1개 | JAX/jaxlib 0.7.2, libtpu 0.0.21.1 |
| v6e1 | 미측정 | 재할당 두 차례 모두 Colab HTTP 502 실패 |
| A100 | NVIDIA A100-SXM4-40GB, MIG disabled | PyTorch 2.11.0+cu130, CUDA 13.0, driver 580.82.07 |
| G4 | NVIDIA RTX PRO 6000 Blackwell Server Edition | PyTorch 2.11.0+cu130, CUDA 13.0, driver 580.82.07 |

GPU는 `torch.mm(x,w,out_dtype=torch.float32).add_(b).relu_()`의 eager 경로다. BF16 reduced precision reduction을 끄고 FP32 중간 결과를 유지했다. 전체 식을 측정하며 fused 커널이라고 부르지 않는다.

TPU는 `lax.dot_general(..., preferred_element_type=float32, precision=HIGHEST)`와 FP32 bias·ReLU를 JIT 컴파일했다. StableHLO 및 최적화 HLO를 저장했다. v5e1의 최적화 HLO는 BF16 피연산자에서 FP32 convolution/dot 결과를 만들고 FP32 bias·maximum을 output fusion에 포함한다. 장치가 하나인지 실행 시 검사했다. HLO는 컴파일러 수준의 정밀도 근거이며 개별 하드웨어 명령을 프로파일링한 기록은 아니다.

## 실측

아래 숫자는 [원시 결과를 집계한 summary.json](./results/2026-10-19-colab/summary.json)에서 나온다.

| 대상 | M | p50 / p95 (ms/batch) | samples/s · 동시성 1 | 최대 절대오차 / 판정 |
|---|---:|---:|---:|---|
| v5e1 | 1 | 0.130180 / 0.150801 | 7,463 | 1.19e-07 / 통과 |
| v5e1 | 128 | 0.138865 / 0.161801 | 920,632 | 2.38e-07 / 통과 |
| v6e1 | 1·128 | 미측정 | 미측정 | 재할당 HTTP 502 실패 |
| A100 | 1 | 0.055047 / 0.070979 | 17,170 | 1.19e-07 / 통과 |
| A100 | 128 | 0.056509 / 0.072322 | 2,136,115 | 1.07e-06 / 통과 |
| G4 | 1 | 0.019260 / 0.020000 | 52,401 | 1.19e-07 / 통과 |
| G4 | 128 | 0.024420 / 0.024540 | 5,287,952 | 2.03e-06 / 통과 |

이 실행에서는 G4의 동기화된 호출 시간이 가장 짧았다. 프레임워크·호스트 CPU·런타임·fusion 경로가 다르며 작은 연산에서는 호출 비용의 영향이 크다. 이 표로 칩의 연산 peak나 순수 커널 성능 순위를 정하지 않는다. seed별 분포와 준비+첫 호출 시간은 각 `result.json`에 남겼다. 준비 시간에는 lowering·컴파일 캐시 조회·첫 호출 등이 섞이며 순수 컴파일 시간으로 해석하지 않는다. 에너지·비용·최대 동시 처리량은 측정하지 않았다.

## 원시 기록과 재현

- [입력 생성](./make_experiment_inputs.py), [Colab 실행 payload](./run_device_experiment.py), [출력 재검증·집계](./summarize_device_experiment.py), [소스 해시](./results/2026-10-19-colab/source-manifest.json).
- [v5e1 원시 시간·메타데이터](./results/2026-10-19-colab/v5e1/results/result.json), [A100](./results/2026-10-19-colab/A100/results/result.json), [G4](./results/2026-10-19-colab/G4/results/result.json). 같은 폴더의 `outputs.npz`는 실제 출력, TPU의 `m*-stablehlo.txt`·`m*-optimized.txt`는 컴파일 기록이다.
- 다운로드한 장치별 `.tar.gz`와 공통 `inputs.npz`도 `results/2026-10-19-colab/`에 보관했다.

로컬에서 `python week5/make_experiment_inputs.py`로 입력을 생성한다. Colab CLI가 인증된 환경에서 장치별 고유 session/config를 사용한다. 예시는 v5e1이다.

```bash
colab --config /tmp/week5-06-v5e1.json new -s week5-06-v5e1 --tpu v5e1
colab --config /tmp/week5-06-v5e1.json upload -s week5-06-v5e1 week5/results/2026-10-19-colab/inputs.npz /content/week5-inputs.npz
colab --config /tmp/week5-06-v5e1.json exec -s week5-06-v5e1 --timeout 600 --env WEEK5_BACKEND=jax --env WEEK5_TARGET=v5e1 -f week5/run_device_experiment.py
colab --config /tmp/week5-06-v5e1.json download -s week5-06-v5e1 /content/week5-results.tar.gz week5/results/2026-10-19-colab/v5e1.tar.gz
colab --config /tmp/week5-06-v5e1.json stop -s week5-06-v5e1
python week5/summarize_device_experiment.py
```

v6e1은 `--tpu v6e1`, GPU는 `--gpu A100` 또는 `--gpu G4`와 `WEEK5_BACKEND=torch`로 바꾼다. 대상명·config·세션명·다운로드 파일명도 함께 바꾼다. 실험 전용 세션은 결과 회수 후 종료하며, 기존 `int8-a100` 세션은 이번 실험에서 사용하지 않는다.

### v6e1 할당 재시도 기록

최초 네 장치의 할당 요청을 병렬로 시작했다. v5e1·A100·G4는 성공했지만 v6e1은 CLI 0.7.4에서 세 차례 `ReadTimeout (read timeout=120)`으로 실패했다. 서버의 assignments 목록에 v6e1이 없음을 확인했다. 세 번째 요청은 다른 실험 전용 장치를 종료한 뒤 실행했다. 따라서 현재까지 네 장치가 모두 동시에 실행됐다고 기록하지 않는다.

PyPI의 최신 배포도 0.7.4여서 공식 저장소의 `006ff96ac2d0e4dcfb702467a76493d687929be8`을 설치했다. 실제 버전은 `0.7.5.dev9+g006ff96ac`이다. 이 버전의 `COLAB_CLI_REQUEST_TIMEOUT` 지원을 소스에서 확인하고 다음과 같이 600초 응답 제한으로 재시도했다. 중복 할당을 피하기 위해 자동 재시도는 끄고 결과를 확인한다.

```bash
uv tool install --force git+https://github.com/googlecolab/google-colab-cli.git
env COLAB_CLI_REQUEST_TIMEOUT=600 COLAB_CLI_ASSIGN_RETRIES=0 colab --config /tmp/week5-06-v6e1.json new -s week5-06-v6e1 --tpu v6e1
```

600초 요청은 사용자 추가 대기 제한 뒤 종료 확인 과정에서 할당 성공이 확인됐다. 그때의 종료 지시에 따라 해당 세션을 종료했다. 이후 사용자가 “할당되었으면 실험해”로 지시를 변경했을 때는 이미 종료가 완료된 상태였다. v6e1을 같은 600초 설정으로 두 차례 재요청했지만 둘 다 할당 POST에서 HTTP `502 Bad Gateway`로 실패했다. 마지막 서버 세션 목록에 v6e1이 없음을 확인했다. v6e1은 출력·지연시간 기록이 없으며 다른 장치의 수치를 대신 쓰지 않는다. 완료된 세 장치는 모두 출력 회수·로컬 재검증 후 종료했다.
