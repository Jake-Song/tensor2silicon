# Native CUDA SwiGLU: A100·G4 검증 (2026-10-11)

SwiGLU만 native CUDA로 교체하고, 나머지 모델 연산은 eager PyTorch로 실행했다. 전체 모델에 torch.compile/CUDA Graph는 적용하지 않았다. 별도 최적화 탐색 없이 시작 커널의 정확성과 성능을 검증했다.

## 환경과 정확성

| GPU | 실제 장치 | 메모리 | 빌드 | 전체 모델 정확성 |
|---|---|---:|---|---|
| A100 | NVIDIA A100-SXM4-40GB | 39.49 GiB | SM80 | 4/4 통과, max abs error 0 |
| G4 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 94.97 GiB | SM120 | 4/4 통과, max abs error 0 |

두 장치 모두 PyTorch 2.11.0+cu130, CUDA toolkit 13.0.88, Triton 3.6.0을 사용했다. Triton은 공유 검증기의 버전 기록용이며 SwiGLU 계산에는 사용하지 않는다. 빌드 플래그는 -O3, --fmad=false이며 fast-math는 사용하지 않았다.

SwiGLU 단독 검증은 빈 입력, 원소 1개, 257개(tail), tiny 모델, 실제 prefill/decode 크기, 비연속 입력, -100~100의 유한 입력, 기본 외 CUDA stream을 포함한다. 전체 모델은 두 seed의 tiny prefill/decode와 GQA, 전체 크기 선택 단계, 변경된 가중치·토큰·cache, 모든 logits와 KV cache 및 이전 cache prefix를 검사했다. 허용오차는 atol=rtol=0.02이며, 전체 모델 출력은 두 GPU 모두 기준선과 일치했다.

## 전체 모델 성능

각 단계는 독립 Python 프로세스에서 두 번 실행했다. 아래 값은 각 실행의 7개 round 중앙값이다. 각 round는 10회 호출하며 호출마다 GPU 완료를 기다린다. 컴파일·입력 준비·5회 warmup은 제외한다.

| GPU | 단계 | 실행 | eager PyTorch (ms) | CUDA SwiGLU 모델 (ms) | baseline / candidate |
|---|---|---:|---:|---:|---:|
| A100 | prefill | 1 | 22.6183 | 22.6194 | 0.9999x |
| A100 | prefill | 2 | 22.6199 | 22.6202 | 1.0000x |
| A100 | decode | 1 | 6.6501 | 6.5833 | 1.0102x |
| A100 | decode | 2 | 6.7923 | 6.7010 | 1.0136x |
| G4 | prefill | 1 | 13.6001 | 13.6234 | 0.9983x |
| G4 | prefill | 2 | 13.5455 | 13.5297 | 1.0012x |
| G4 | decode | 1 | 2.6322 | 2.6152 | 1.0065x |
| G4 | decode | 2 | 2.6286 | 2.6106 | 1.0069x |

Prefill: B=1,T=S=2048. Decode: B=8,T=1,S=2048, 고정 길이 cache의 마지막 slot을 갱신하는 한 단계다. 모델은 D=2048,F=5632,8 layers,16 query/KV heads,vocab=32000,BF16이다. 이는 토큰화·sampling·여러 토큰 생성 시간이나 기존 roofline 측정값과 직접 비교할 수 없다.

## SwiGLU 단독 성능

동일한 BF16 입력을 반복 사용했다. 동기 호출은 Python/바인딩·할당·launch·GPU 완료 대기를 포함하며 7 round × 20회 호출이다. CUDA Graph는 100개 연산을 담은 graph를 10회 replay하고 CUDA event로 측정해 연산당 시간을 구했다(7 round 중앙값). 이는 launch를 줄인 반복 입력 측정이며 전체 모델의 실행 방식과 다르다. cache가 따뜻한 상태의 시간이므로 HBM 대역폭이나 전체 모델 개선으로 환산하지 않는다.

| GPU | 크기 | 측정 | eager PyTorch (µs) | native CUDA (µs) | 속도비 |
|---|---|---|---:|---:|---:|
| A100 | 8×1×5632 | synchronized | 32.070 | 22.015 | 1.457x |
| A100 | 8×1×5632 | cuda_graph | 4.467 | 2.272 | 1.966x |
| A100 | 1×2048×5632 | synchronized | 110.836 | 102.857 | 1.078x |
| A100 | 1×2048×5632 | cuda_graph | 89.667 | 86.890 | 1.032x |
| G4 | 8×1×5632 | synchronized | 13.660 | 10.137 | 1.347x |
| G4 | 8×1×5632 | cuda_graph | 2.541 | 1.250 | 2.034x |
| G4 | 1×2048×5632 | synchronized | 29.636 | 35.612 | 0.832x |
| G4 | 1×2048×5632 | cuda_graph | 16.854 | 26.209 | 0.643x |

## 해석과 재현

G4 prefill 크기의 단독 CUDA 커널은 PyTorch보다 느렸다: CUDA Graph에서 16.854 → 26.209 µs(약 55.5% 증가). 현재 scalar 커널을 SM120으로 빌드한 것만으로 큰 입력에서 성능 개선이 보장되지는 않는다.

작은 decode 크기에서 SwiGLU를 한 커널로 합치면 단독 실행 비용을 줄일 수 있다. 전체 모델에는 GEMM, attention, 다른 연산과 host launch 비용이 남아 있으므로 이 커널의 단독 개선비를 전체 모델에 적용할 수 없다. 전체 모델 차이가 작거나 반복 실행에서 방향이 달라지는 경우에는 확정적인 최적화 성과로 해석하지 않는다. G4 버전은 같은 계산을 SM120에 맞춰 빌드한 버전이며, 별도 튜닝 탐색을 수행한 결과는 아니다.

원시 결과와 소스 SHA-256: [A100 summary.json](a100/summary.json), [G4 summary.json](g4/summary.json). 각 폴더에 단계별 두 실행의 JSON/log와 operator.json을 저장했다.

```bash
python3 autoresearch/validate_swiglu.py --track a100 --out /tmp/swiglu-a100
python3 autoresearch/validate_swiglu.py --track g4 --out /tmp/swiglu-g4
```

각 명령은 해당 GPU 런타임 안에서 실행한다. CUDA PyTorch, nvcc, C++ 컴파일러, Ninja, Triton, NumPy가 필요하다. Colab CLI로 A100과 G4를 순서대로 할당하고 검증 후 종료했다. 최종 서버 세션 상태는 [cleanup.json](cleanup.json)에 기록한다.

소스 폴더는 검증 후 [gpu/cuda](../../gpu/cuda/README.md)와 [gpu/cuda_g4](../../gpu/cuda_g4/README.md)로 이동했다. 원시 JSON/log의 경로·해시는 측정 당시 기록을 보존하며, 현재 경로와 이동 전후 해시는 [relocation.json](relocation.json)에 기록했다. 커널·모델 계산 코드는 그대로이고 bench.py는 저장소 루트 탐색 깊이만 변경했다.
