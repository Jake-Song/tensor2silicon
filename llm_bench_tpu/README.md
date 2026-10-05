# TPU v6e: JAX · JAX+Pallas · native Pallas

저장소의 동일한 decoder를 세 구현으로 실행합니다. Colab에서 **TPU v6e**를 선택하고
[실행 노트북](../notebooks/llm_backend_comparison_tpu_v6e.ipynb)을 순서대로 실행하세요.
노트북에는 실행에 필요한 소스가 내장되어 있습니다.

실측 자료: [프로파일링·최적화 보고서](results/2026-10-05-v6e-optimized/report.md),
[최적화 전 비교](results/2026-10-05-v6e/report.md).

| 구현 | 계산 범위 |
|---|---|
| JAX | 기존 `llm_roofline.jax_llm.forward` 전체를 JIT/XLA 컴파일 |
| JAX+Pallas | 같은 JAX 모델에서 QK → causal softmax → PV만 직접 작성한 Pallas 커널로 교체 |
| native Pallas | embedding, LayerNorm, 모든 GEMM, RoPE, head 배치, attention, residual, SwiGLU, KV 갱신을 직접 작성한 Pallas로 실행 |

native도 JAX의 배열·컴파일·실행 기능을 사용합니다. 모델 계산은 모두 Pallas 안에 있으며,
바깥의 StableHLO에는 `custom_call`, `reshape`, `constant`만 허용합니다.
JAX 경로는 Pallas 호출 0개, 혼합 경로는 층마다 attention 호출 1개인지 검사합니다.

```bash
pip install 'jax[tpu]==0.11.2' 'libtpu==0.0.48'
python -m llm_bench_tpu --smoke --samples 3 \
  --tuning-config llm_bench_tpu/configs/optimized-v6e.json --out /tmp/tpu-smoke.json
RUN_TPU_PALLAS_TESTS=1 python -m unittest discover -s tests -p 'test_llm_bench_tpu.py' -v
python -m llm_bench_tpu --seed 123 --hlo-dir /tmp/hlo-123 --out /tmp/tpu-run1.json \
  --tuning-config llm_bench_tpu/configs/optimized-v6e.json --backends jax,hybrid,native,original_native
python -m llm_bench_tpu --seed 321 --reverse --hlo-dir /tmp/hlo-321 --out /tmp/tpu-run2.json \
  --tuning-config llm_bench_tpu/configs/optimized-v6e.json --backends jax,hybrid,native,original_native
```

TPU를 초기화하는 프로세스는 한 번에 하나만 실행합니다. `--smoke`는 작은 모델의 정확도 검사이며
전체 모델 성능 점수로 사용하지 않습니다. 명령은 저장소 루트에서 실행합니다.
`--tuning-config`를 생략하면 원본 설정을 재현합니다. `frozen/`에는 수정 전 native와
attention을 함께 보존했으며, `original_native`로 현재 구현과 같은 실행에서 비교할 수 있습니다.

## 프로파일링과 제한된 탐색

```bash
python -m llm_bench_tpu --tuning-config llm_bench_tpu/configs/optimized-v6e.json \
  --profile-dir /tmp/tpu-profile-new --profile-steps 20 --out /tmp/tpu-profile-run.json
python -m llm_bench_tpu.profile_summary /tmp/tpu-profile-new --out /tmp/tpu-profile-summary.json
```

프로파일은 점수용 측정 후 별도로 수집합니다. 매번 새 디렉터리를 사용합니다.
XPlane·Perfetto 파일과 실제 TPU 장치 이벤트 20개를 검증하고, 중첩 구간을 중복 집계하지 않습니다.
장치 실행 시간과 호스트 호출·완료 대기를 포함한 전체 지연은 서로 다른 지표입니다.
XPlane을 XProf에서 열려면 `pip install xprof` 후 `xprof --logdir /tmp/tpu-profile-new`를 사용합니다.
프로파일러 설치 시 측정에 사용하는 JAX·jaxlib·libtpu 버전을 유지합니다.

`python -m llm_bench_tpu.tune --help`에서 재탐색 옵션을 확인할 수 있습니다.
후보 JSON은 `configs/candidates-*.json`, 선택 결과는 `selected.json`, 모든 결정은
`candidates.jsonl`에 저장합니다. 전체 모델이 seed 123·321 **각각 1% 초과 개선**되고
기존 정확도 검사를 모두 통과해야 채택합니다. 후보별 최대 300초, 전체 최대 30개이며,
`--deadline`은 새 후보를 시작할 수 있는 마지막 Unix 시각입니다.
Decode와 prefill 설정은 별도로 선택합니다. 부정확하거나 느린 후보도 로그에 남깁니다.

## 모델과 측정 조건

- 8층, D=2048, FFN=5632, query/KV heads=16, head_dim=128, vocabulary=32000.
- Prefill: B=1, T=S=2048, 모든 prompt 위치의 logits를 계산합니다.
- Decode: B=8, T=1, S=2048, 마지막 cache 슬롯을 갱신하는 한 번의 forward입니다.
- 각 실행에서 세 구현이 정확히 같은 BF16 가중치·입력·KV 배열을 공유합니다. 무작위 가중치를 사용합니다.
- BF16 가중치·activation·KV·logits, GEMM 누산과 정규화·softmax 내부 계산은 FP32입니다.
- Pallas attention은 설정된 query 블록별로 KV context 전체를 읽습니다. score와 probability의
  BF16 변환을 명시하며, 이 중간 배열은 VMEM 안에 유지합니다. 온라인 softmax/FlashAttention 구현은 아닙니다.
- native GEMM은 FP32 VMEM 누산기와 K축 pipeline을 사용합니다. dtype을 낮추거나 가중치를 압축하지 않습니다.
- 타일, KV/head 그룹, fusion 설정은 `configs/optimized-v6e.json`에 명시합니다.
  GEMM 설정의 배열 순서는 `[M, K, N]`이며, shape별 key는 `MxKxN`입니다.

전체 모델을 컴파일한 뒤 5회 warmup하고, 세 구현의 측정 순서를 순환·역전하면서 각각 60회 측정합니다.
각 표본은 logits와 모든 KV 출력의 완료를 기다리는 wall-clock 한 번입니다.
컴파일·입력 생성·host 전송·검증 시간은 제외하고, 컴파일 시간은 별도 기록합니다.
cache donation은 세 경로 모두 사용하지 않습니다. 고정 cache 복사 비용이 포함되므로,
이 결과를 연속 토큰 생성이나 cache를 제자리 갱신하는 서비스의 지연 시간으로 해석하지 않습니다.

## 정확도와 실행 증거

동일 BF16 가중치·KV·위치 테이블을 FP32로 승격한 **원본 JAX 모델**을 검증 기준으로 사용합니다.
FP32 기준 계산은 matmul `highest` precision을 사용하고 성능표에는 포함하지 않습니다.
허용 오차는 실행 전에 고정합니다.

- 모든 logits, 모든 층의 K/V: `RMSE / RMS(reference) <= 0.02`, `max_abs / RMS(reference) <= 0.20`.
- RMS 분모 하한 `1e-6`; 잘못된 shape/dtype·비유한 값은 실패합니다.
- Decode 새 토큰 K/V를 별도로 검사하고 기존 cache prefix가 비트 단위로 유지되는지 확인합니다.
- 측정 후 세 구현을 다시 원본 BF16 JAX와 비교합니다. 반올림 차이는 허용 오차로 검증합니다.
- 결과 JSON에 실제 장치·메모리·패키지 버전·설정·소스 SHA-256·StableHLO 연산 목록·원시 시간 표본을 저장합니다.
  같은 이름의 `.sources.json`에는 실행 소스 snapshot을 보존합니다.

CPU에서 가능한 검증:

```bash
python -m unittest discover -s tests -p 'test_llm_bench_tpu.py' -v
```

실행 방식의 배경은 [JAX Pallas 소개](https://docs.jax.dev/en/latest/pallas/index.html),
[TPU 블록·메모리 제약](https://docs.jax.dev/en/latest/pallas/tpu/details.html),
[Pallas GEMM 가이드](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)를 참고하세요.
