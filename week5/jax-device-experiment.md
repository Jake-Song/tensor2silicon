# week5-06 JAX 통일 실험 — 2026-10-19

GPU와 TPU를 모두 **JAX/jaxlib 0.11.1**로 맞춰 `Y=ReLU(XW+b)`를 다시 측정했다. [첫 실험](./device-experiment.md)은 GPU PyTorch eager·TPU JAX 0.7.2였으므로 별도 결과로 보존한다.

## 조건과 검증

- `X:[M,1024]`, `W:[1024,1024]`, `b:[1024]`, `M=1,128`이며 추론만 수행한다.
- 첫 실험의 [동일한 입력 파일](./results/2026-10-19-colab/inputs.npz)과 [FP64 기준 출력·배열 해시](./results/2026-10-19-colab/inputs-manifest.json)를 재사용했다. archive SHA256은 `9de23ebe967f1e1360cc7d9f4d6f7137b7e4ce012c738bc333a8a69d9bb95590`이다.
- 입력 X·W는 BF16으로 미리 반올림한 값, bias와 출력은 FP32다. `lax.dot_general(..., preferred_element_type=float32, precision=HIGHEST)` 뒤 FP32 bias와 ReLU를 적용한다. `jax.jit`을 두 shape에 각각 컴파일하며 인수는 장치에 상주한다.
- 각 seed 0–4마다 50회 준비 실행 후 1,000회 순차 측정한다. shape별 5,000개 원시 시간을 합쳐 p50/p95를 계산한다. 동시성은 1이다.
- 모든 장치에서 `perf_counter_ns()`로 호출 시작부터 `block_until_ready()` 완료까지 측정한다. 컴파일·입출력 전송·정확도 검사는 측정 밖이다. 같은 가중치를 반복 사용하고 캐시를 비우지 않는다.
- 기준은 원소별 `abs(error) <= 1e-4 + 1e-4*abs(reference)`다. NaN/Inf를 실패 처리한다. zero X와 양수·음수·0·0 부근 bias 경계 검사도 두 shape에서 수행했다. 회수한 실제 출력은 로컬에서 다시 검증했다.
- 요청한 가속기 backend가 `gpu`/`tpu`인지, JAX 장치가 정확히 1개인지 검사한다. CPU 대체 실행을 허용하지 않는다.

## 실제 환경

| 대상 | 실제 장치 | 프레임워크·런타임 |
|---|---|---|
| v5e1 | TPU v5 lite, 1개 | JAX/jaxlib 0.11.1, libtpu 0.0.46.3 |
| v6e1 | 미측정 | 할당 두 차례 모두 HTTP 502 실패 |
| A100 | NVIDIA A100-SXM4-40GB | JAX/jaxlib·cuda13 plugin·PJRT 0.11.1, driver 580.82.07 |
| G4 | NVIDIA RTX PRO 6000 Blackwell Server Edition | JAX/jaxlib·cuda13 plugin·PJRT 0.11.1, driver 580.82.07 |

GPU 기본 설치 버전은 0.11.1이었다. v5e1 기본 설치 0.7.2를 `pip install --upgrade 'jax[tpu]==0.11.1'`로 업데이트하고 커널을 재시작한 뒤 측정했다. Python은 GPU 3.13.15, TPU 3.13.16이며 NumPy는 모두 2.1.3이다. 자세한 패키지 목록은 각 `result.json`의 `meta.runtime_packages`를 따른다.

## 실측

| 대상 | M | p50 / p95 (ms/batch) | samples/s · 동시성 1 | 최대 절대오차 / 판정 |
|---|---:|---:|---:|---|
| v5e1 | 1 | 0.162505 / 0.196320 | 6,041 | 1.19e-07 / 통과 |
| v5e1 | 128 | 0.151620 / 0.172681 | 838,639 | 2.38e-07 / 통과 |
| v6e1 | 1·128 | 미측정 | 미측정 | 할당 HTTP 502 실패 |
| A100 | 1 | 0.098612 / 0.148289 | 9,125 | 1.19e-07 / 통과 |
| A100 | 128 | 0.109679 / 0.147513 | 1,075,356 | 4.77e-07 / 통과 |
| G4 | 1 | 0.026230 / 0.028320 | 37,328 | 1.19e-07 / 통과 |
| G4 | 128 | 0.043605 / 0.045100 | 2,922,496 | 3.58e-07 / 통과 |

처리량은 실제 완료한 `5,000×M`을 다섯 측정 구간의 경과 시간 합으로 나눈 값이며 측정 루프 비용도 포함한다. [집계 JSON](./results/2026-10-19-colab-jax/summary.json)에 평균과 표본 수도 기록했다.

이번 실행의 G4 호출 시간이 가장 짧았다. **이 값은 순수 장치 커널 시간이 아니라 호스트 호출과 동기화를 포함한 지연시간**이다. 프레임워크 버전과 소스는 같아도 호스트 CPU·런타임·장치별 lowering은 다르다. 첫 실험보다 시간이 늘어난 사실만으로 JAX 자체의 성능 차이를 단정하지 않는다. seed별 시간 분포와 p95 차이는 원시 기록에 그대로 남긴다.

## 컴파일 기록

모든 실행 장치의 StableHLO·최적화 HLO를 저장했다. GPU M=128은 BF16 입력에서 FP32 결과를 만드는 Triton nested GEMM fusion과 FP32 reduce/bias/ReLU fusion 경로다. A100은 K를 4개, G4는 8개로 나눈 중간 결과가 최적화 HLO에 나타난다. 같은 JAX 코드라도 컴파일러가 선택한 실행 경로가 다르며 전체 식이 단일 커널이라고 가정하지 않는다. 하드웨어 명령별 프로파일링은 수행하지 않았다.

## 원시 결과와 재현

[시간 기록 분석 노트북](../notebooks/week5_benchmark_timings.ipynb)을 열면 GPU 없이 로컬에서 두 실험의 시간 원시 기록을 표·그래프로 볼 수 있다. 입력·출력 재검증이나 장치 코드는 실행하지 않는다.

- [실행 payload](./run_jax_experiment.py), [출력 재검증·집계](./summarize_jax_experiment.py), [소스 해시](./results/2026-10-19-colab-jax/source-manifest.json).
- [v5e1 원시 기록](./results/2026-10-19-colab-jax/v5e1/results/result.json), [A100](./results/2026-10-19-colab-jax/A100/results/result.json), [G4](./results/2026-10-19-colab-jax/G4/results/result.json). 각 폴더의 `outputs.npz`와 `m*-stablehlo.txt`·`m*-optimized.txt`도 보관했다.
- 압축된 원본 회수 파일은 `results/2026-10-19-colab-jax/*.tar.gz`다. 공통 입력은 첫 실험 폴더에서 읽으므로 복제하지 않았다.

Colab CLI 인증 후 장치별 session/config를 사용한다. GPU에는 `jax[cuda13]==0.11.1`, TPU에는 `jax[tpu]==0.11.1`을 사용하고 기존 JAX가 로드된 경우 커널을 재시작한다. GPU 예시는 다음과 같다.

```bash
env COLAB_CLI_REQUEST_TIMEOUT=600 COLAB_CLI_ASSIGN_RETRIES=0 colab --config /tmp/week5-jax-A100.json new -s week5-jax-A100 --gpu A100
colab --config /tmp/week5-jax-A100.json upload -s week5-jax-A100 week5/results/2026-10-19-colab/inputs.npz /content/week5-inputs.npz
colab --config /tmp/week5-jax-A100.json exec -s week5-jax-A100 --timeout 600 --env WEEK5_TARGET=A100 -f week5/run_jax_experiment.py
colab --config /tmp/week5-jax-A100.json download -s week5-jax-A100 /content/week5-jax-results.tar.gz week5/results/2026-10-19-colab-jax/A100.tar.gz
python week5/summarize_jax_experiment.py
colab --config /tmp/week5-jax-A100.json stop -s week5-jax-A100
```

TPU에서는 `--tpu v5e1`/`--tpu v6e1`, G4에서는 `--gpu G4`를 쓰고 config·세션·target·저장 파일명도 바꾼다. 실험 전용 세션은 회수·검증 후 종료하며 기존 `int8-val` 세션에는 변경을 가하지 않는다.

## 할당·연결 기록

네 장치를 병렬 요청했다. v6e1은 업데이트된 Colab CLI `0.7.5.dev9+g006ff96ac`에서 `COLAB_CLI_REQUEST_TIMEOUT=600`, `COLAB_CLI_ASSIGN_RETRIES=0`으로 최초 요청과 한 번의 재시도를 실행했지만 둘 다 할당 POST가 HTTP `502 Bad Gateway`로 실패했다. v6e1은 실행·출력·지연시간 기록이 없다. A100은 최초 연결이 끊겨 재접속·커널 재시작 후 실험을 완료했다. 세 장치의 최종 결과에는 모두 실제 JAX 가속기와 0.11.1 버전이 기록돼 있다. [할당 상태 기록](./results/2026-10-19-colab-jax/allocation-status.json).
