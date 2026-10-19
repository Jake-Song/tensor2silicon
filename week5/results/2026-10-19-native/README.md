# CUDA·Pallas 비교 실측

`ReLU(XW+b)`, `X:[M,1024]`, `W:[1024,1024]`. 기존 합성 입력을 그대로 사용했다.
BF16 입력·FP32 누산·FP32 출력, `atol=rtol=1e-4`, seed 5개, seed당 준비 실행 50회·측정 1,000회다.
각 장치에서 JAX 기준선을 다시 측정하고 seed별 실행 순서를 번갈아 바꿨다.

| 실제 장치 | M | JAX p50 (ms) | CUDA/Pallas p50 (ms) | p50 속도비 |
|---|---:|---:|---:|---:|
| A100-SXM4-40GB | 1 | 0.143696 | 0.019145 | 7.51× |
| A100-SXM4-40GB | 128 | 0.106976 | 0.046898 | 2.28× |
| TPU v5 lite | 1 | 0.139980 | 0.136445 | 1.03× |
| TPU v5 lite | 128 | 0.145925 | 0.146231 | 1.00× |

호스트 호출부터 장치 완료까지의 시간이다. CUDA의 출력 버퍼 재사용과 호출 경로 차이도 포함하며 CUDA Graph는 사용하지 않았다. 순수 커널 가속률로 해석하지 않는다. TPU에서는 뚜렷한 개선을 확인하지 못했다.

- A100: CUDA GEMV는 16 warps/block, WMMA GEMM은 8 warps/block을 선택했다. bias·ReLU를 커널에 합쳤다. CUDA 13.0, PyTorch 2.11.0+cu130은 메모리 할당·동기화에 사용했다.
- TPU: Pallas의 출력 열 타일은 M=1에서 256, M=128에서 1024다. BF16 MXU 입력·FP32 누산을 사용하며, 최적화 HLO에서 `tpu_custom_call`을 확인했다. `interpret=False`, libtpu 0.0.46.3이다.
- 공통 JAX/jaxlib 버전은 0.11.1이다. 컴파일·후보 선택은 최종 타이밍에서 제외했다.
- 출력 40개를 회수해 로컬 FP64 기반 기준값과 재검증했다. 경계·입력 변경 검사 16개도 통과했다.
- 전력은 측정하지 않았다. 노트북의 A100 samples/J는 400 W 상한 가정이며 TPU는 N/A다.
- G4 할당은 `Service Unavailable`로 실패했고, 사용자 선택에 따라 재시도하지 않았다. 두 실험용 세션은 종료했으며 기존 `int8-triton-1019` 세션은 유지했다.

원시 시간·선택 후보·실행 환경은 장치별 `results/result.json`, 실제 출력은 `outputs.npz`에 있다. TPU의 `tuning-failure.json`은 수정 전 컴파일 오류 기록이며 최종 결과는 `result.json`이다. 코드 해시는 각 실행의 `meta.sources`와 `source-manifest.json`에 있다.

로컬 재검증·집계:

```bash
.venv/bin/python week5/summarize_native_experiment.py
```

원격 재현 시 [실행 코드](../../run_native_experiment.py), [CUDA](../../native_relu.cu), [Pallas](../../pallas_relu.py)를 `/content/week5-native-src/`에 두고 기존 `2026-10-19-colab/inputs.npz`를 `/content/week5-inputs.npz`에 업로드한다. GPU는 `jax[cuda13]==0.11.1`과 nvcc, TPU는 `jax[tpu]==0.11.1`을 준비하고 `WEEK5_TARGET=A100` 또는 `v5e1`로 실행 코드를 실행한다. `/content/week5-native-results.tar.gz`를 회수한 뒤 전용 세션을 종료한다.

이전 JAX 기준 비교는 이 폴더의 `summary.json`과 `comparison.png`에 보존했다. [분석 노트북](../../../notebooks/week5_benchmark_timings.ipynb)은 A100의 PyTorch 기준 비교와 TPU의 JAX 기준 비교를 표시한다.
