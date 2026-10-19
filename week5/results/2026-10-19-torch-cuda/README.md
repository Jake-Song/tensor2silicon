# A100 기준선 정정: native PyTorch eager vs CUDA

기준선은 `torch.mm(X, W, out_dtype=torch.float32).add_(b).relu_()`다.
PyTorch 2.11.0+cu130, CUDA 13.0, 실제 장치는 NVIDIA A100-SXM4-40GB다.
JAX를 로드하지 않은 새 세션에서 PyTorch와 기존 CUDA 커널을 함께 측정했다.
앞선 JAX 비교는 `../2026-10-19-native/`에 그대로 보존했다.

| M | PyTorch p50 (ms) | CUDA p50 (ms) | p50 속도비 |
|---|---:|---:|---:|
| 1 | 0.054518 | 0.019189 | 2.84× |
| 128 | 0.055284 | 0.046608 | 1.19× |

기존과 같은 합성 입력, BF16 입력·FP32 누산 및 출력, `atol=rtol=1e-4`를 사용했다.
BF16 reduced-precision reduction과 TF32를 껐다. CUDA 설정은 기존 선택값인 M=1의 16 warps와 M=128의 8 warps로 고정했다.
seed 5개마다 50회 준비 실행 후 1,000회 측정했으며, PyTorch/CUDA 순서를 seed별로 바꿨다.
호스트 호출부터 장치 동기화 완료까지 측정하며 컴파일·전송은 제외했다. CUDA는 출력 버퍼를 재사용하고 PyTorch는 일반 eager 호출의 출력 할당을 포함한다. CUDA Graph·torch.compile은 사용하지 않았다.

출력 배열 20개를 회수해 로컬 기준값과 재검증했고, 원격 경계·입력 변경 검사 8개도 통과했다. 전력은 측정하지 않았으며 samples/J는 기록된 400 W 상한 가정이다.

[노트북](../../../notebooks/week5_benchmark_timings.ipynb)에는 이 A100 결과와 기존 TPU JAX/Pallas 결과를 함께 표시했다. 이전 A100 JAX 대비 최적화 비교 셀 4개는 제거하고, PyTorch 기준 비교를 남겼다.

```bash
.venv/bin/python week5/summarize_torch_cuda_experiment.py
```

원격 재현: [실행 코드](../../run_torch_cuda_experiment.py)와 [CUDA 소스](../../native_relu.cu)를 `/content/week5-torch-cuda-src/`에 두고, 기존 합성 `2026-10-19-colab/inputs.npz`를 `/content/week5-inputs.npz`에 업로드한다. PyTorch와 nvcc가 설치된 A100에서 `WEEK5_TARGET=A100`으로 실행하고 `/content/week5-torch-cuda-results.tar.gz`를 회수한다. 실험용 세션은 작업 후 종료한다.
