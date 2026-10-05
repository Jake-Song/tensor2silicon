# BF16 모델 전체 추론 비교

현재 프로젝트의 8층 decoder를 **컴파일 PyTorch**, **컴파일 PyTorch + 직접 작성한 Triton**,
**전체 계산을 직접 작성한 Triton**으로 실행한다. 기존 `llm_roofline` 모델과 과거 실험은 수정하지 않는다.

GPU 환경에서 저장소 루트를 작업 디렉터리로 사용한다.

```bash
# 정확도와 실행 경로 확인: 작은 모델이며 최종 성능 점수가 아니다.
python -m llm_bench --preset tiny --quick --profile --out /tmp/llm-tiny.json

# 실제 모델: prefill, 고정 context decode, 128토큰 연속 생성
python -m llm_bench --backend all --profile --out /tmp/llm-full.json

# 개별 구현/단계만 실행
python -m llm_bench --backend triton --workload decode --out /tmp/llm-decode.json

# 공식 PyTorch FlexAttention을 연속 생성 decode에 쓰는 기준선 후보
python -m llm_bench --backend torch --torch-attention flex-decode --profile --out /tmp/llm-flex-candidate.json

# CPU 검사와 실제 GPU 경계조건 검사
RUN_LLM_BENCH_GPU_TESTS=1 python -m unittest discover -s tests -p 'test_llm_bench*.py' -v
```

PyTorch와 Triton은 GPU 런타임에 설치된 패키지를 사용한다. 실행 결과에는 실제 GPU,
메모리, CUDA/PyTorch/Triton 버전과 Python 소스 SHA-256을 기록한다.
`--help`는 GPU 패키지가 없는 환경에서도 실행할 수 있다.

## 비교 조건

| 항목 | 조건 |
|---|---|
| 모델 | 8층, D=2048, FFN=5632, 16 query/KV heads, head_dim=128, vocabulary=32000 |
| Prefill | B=1, T=S=2048, 모든 prompt 위치의 logits |
| Decode | B=8, T=1, S=2048, 마지막 cache 슬롯을 갱신하는 전체 forward |
| 연속 생성 | B=1, prompt 2048, greedy 128토큰; prefill에서 1개 + decode 127회 |
| 자료형 | 가중치·저장 activation·KV·logits BF16, GEMM 누산과 일부 내부 계산 FP32 |
| 컴파일 | 두 PyTorch 경로는 Inductor `fullgraph=True`, `max-autotune-no-cudagraphs` |
| Graph | 세 구현 모두 외부 CUDA Graph; graph 없는 시간도 별도 측정 |

PyTorch native에도 Inductor가 생성한 Triton은 포함될 수 있다.
비교 경계는 **직접 작성한 Triton 커널의 범위**다.
Native Triton의 PyTorch 사용은 메모리 할당, 가중치 준비, 메타데이터와 CUDA Graph 관리로 제한한다.
profiler의 CUDA 계산 커널이 직접 작성한 커널 목록에 속하는지도 확인한다.
Chrome trace의 leaf kernel/memcpy/memset만 집계해 compiled-region annotation과 자식 커널을 중복 계산하지 않는다.

고정 decode에서는 모든 KV 슬롯이 유효하므로 PyTorch SDPA에 별도의 mask를 전달하지 않는다.
연속 생성은 위치가 증가하므로 미래 KV 슬롯을 제외해야 한다. `--torch-attention flex-decode`는
이 작업에 공식 PyTorch `flex_attention`과 GPU 위치 tensor를 참조하는 `score_mod`를 사용한다.
짧은 query의 decoding 전용 컴파일 경로와 위치 closure는
[PyTorch 공식 설명](https://pytorch.org/blog/flexattention-for-inference/)을 따른다.
Prefill과 고정 decode는 SDPA를 유지한다. 이 옵션 역시 사용자 작성 Triton이 없는 PyTorch 기준선이다.

무작위 가중치 모델의 실행 성능을 측정한다. 토큰화·네트워크·입력 전송·컴파일·튜닝은
정상 추론 시간에서 제외한다. 생성 시간에는 prefill, cache 복사, argmax, 위치 갱신과
127회의 실제 누적 decode가 포함된다. EOS 조기 종료는 없다.
Prefill과 생성의 첫 prefill은 모든 prompt 위치의 logits를 계산한다.
마지막 prompt 위치에만 LM head를 적용하는 production 최적화는 이번 비교에 포함하지 않는다.

## 정확도와 시간 해석

동일한 BF16 가중치를 FP32로 승격해 **기존 `llm_roofline.torch_llm.forward`**를 reference로 사용한다.
이 FP32 검사는 추론 성능표에 포함하지 않는다. BF16 커널 안의 FP32 누산은 실제 BF16 추론에 포함한다.
원본 eager BF16의 중간 반올림을 그대로 재현하는 대신 동일 수식의 수치 오차를 검증한다.

- logits와 모든 유효 KV: `RMSE / RMS(reference) <= 0.02`, `max_abs_error / RMS(reference) <= 0.20`.
- 분모 하한은 `1e-6`; 비유한 값·잘못된 shape/dtype·cache prefix 변경은 실패다.
- 연속 생성의 수치 검사는 양쪽에 같은 reference 토큰을 공급해 127회 decode를 진행하고 여섯 checkpoint에서 logits와 유효 KV를 검증한다. 캡처한 Graph도 두 번 연속 검증한다. 자유 생성 토큰의 차이를 수치 오류로 혼동하지 않는다.
- CUDA event 표본은 100회 forward를 묶은 평균이다. 해당 p95는 개별 요청의 p95가 아니다.
- 동기화 wall-clock은 호출마다 장치 완료를 기다린 개별 요청 시간이다.
- 생성의 CUDA/wall-clock 표본은 128토큰 생성 한 회 전체다.
- 첫 호출 시간에는 기존 런타임의 컴파일 캐시가 재사용될 수 있다. 완전한 cold-start 점수로 해석하지 않는다.
- 메모리는 BF16 원본/packed 가중치와 FP32 검증 가중치를 포함하는 프로세스 전체 peak다.

## 튜닝과 재측정

```bash
# 같은 GPU에서 순서대로 실행한다.
python -m llm_bench.tune --out /tmp/split-k-tuning.json
python -m llm_bench.tune_attention --out /tmp/attention-tuning.json
python -m llm_bench.tune_flex --out /tmp/flex-tuning.json
# 선택: 지원 GPU에서 TMA prefill을 독립적으로 측정한다.
python -m llm_bench.tune_tma --out /tmp/tma-prefill.json
# 선택: gate/up GEMM과 SwiGLU 결합 후보를 측정한다. 자동 적용하지 않는다.
python -m llm_bench.tune_swiglu --out /tmp/swiglu-fusion.json

# 측정 근거까지 보존해 하나의 config로 합친다.
python - <<'PY'
import json
from pathlib import Path
root = Path('/tmp')
files = ['split-k-tuning.json', 'attention-tuning.json', 'flex-tuning.json']
gemm, attention, flex = [json.loads((root / name).read_text()) for name in files]
assert gemm['status'] == attention['status'] == 'ok' and flex['complete']
assert flex['selected_options'] is not None
gemm.update(attention_split_overrides=attention['attention_split_overrides'], attention_tuning=attention,
            flex_kernel_options=flex['selected_options'], flex_tuning=flex, source_files=files)
if (root / 'tma-prefill.json').exists():
    tma = json.loads((root / 'tma-prefill.json').read_text())
    gemm.update(tma_tuning=tma, tma_overrides={shape: row['best_tma_config']
        for shape, row in tma.get('shapes', {}).items() if row.get('recommended_backend') == 'tma'})
    gemm['source_files'].append('tma-prefill.json')
if (root / 'swiglu-fusion.json').exists():
    gemm.update(swiglu_tuning=json.loads((root / 'swiglu-fusion.json').read_text()), swiglu_adopted=False)
    gemm['source_files'].append('swiglu-fusion.json')
(root / 'tuning-final.json').write_text(json.dumps(gemm, indent=2))
PY

python -m llm_bench --tuning-config /tmp/tuning-final.json --torch-attention flex-decode --hybrid-attention triton --hybrid-matmul m1-triton --profile --seed 123 --out /tmp/final-run1.json
python -m llm_bench --tuning-config /tmp/tuning-final.json --torch-attention flex-decode --hybrid-attention triton --hybrid-matmul m1-triton --profile --seed 321 --reverse --out /tmp/final-run2.json
```

`--tuning-config`는 GEMM split-K, native attention split-KV, PyTorch FlexAttention의 `kernel_options`,
선택 projection shape의 `tma_overrides`를 함께 적용한다. 합쳐진 config는 GEMM·attention·Flex·TMA의
후보별 정확도·시간·선택 근거도 보존한다.
구성요소 튜닝 시간은 전체 모델 시간과 구분하며, 최종 선택은 전체 추론의 정확도와 시간으로 다시 확인한다.
Cache eviction은 측정 구간 밖에서 수행하는 근사 조건이다. Flex compiler 옵션은 버전에 의존하므로
설치된 source의 지원 근거와 실패한 후보를 결과 JSON에 남긴다.

TMA 후보는 row-major 가중치를 별도로 packing하지 않는다. 실제 PTX의 TMA load/store와 `mma.sync`,
shared-memory 사용량을 확인하고, 구성요소에서 기존 GEMM보다 1% 넘게 빠른 shape만 전체 모델 후보로 전달한다.
기본 실행 경로는 config에 명시한 shape만 TMA로 전환한다. 이 작은 GEMM 배속을 전체 추론 배속으로 대신하지 않는다.

Gate/up GEMM+SwiGLU fusion은 기존 BF16 projection 반올림을 보존하는 독립 후보다.
이번 실측에서 네 후보 모두 정확성을 통과했지만 최고 배속 1.00707×는 1% 초과 채택 기준에 못 미쳐 적용하지 않았다.
중간 tensor I/O 약 92 MB 절감은 논리적 byte 수이며 DRAM traffic 실측값이 아니다.
`swiglu_tuning`은 근거 보존용이며 runner의 실행을 바꾸지 않는다.

| 선택 | 동작 |
|---|---|
| `--torch-attention sdpa` | 순수 PyTorch의 기본 SDPA 경로 |
| `--torch-attention flex-decode` | 순수 PyTorch의 연속 생성 decode에 공식 FlexAttention 사용 |
| `--hybrid-attention triton` | 혼합 구현의 prefill/decode attention을 직접 작성한 Triton으로 교체 |
| `--hybrid-attention decode-triton` | 혼합 구현의 decode attention만 직접 작성한 Triton으로 교체 |
| `--hybrid-matmul decode-triton` | 혼합 구현의 모든 decode projection과 LM head를 Triton GEMM으로 교체 |
| `--hybrid-matmul m1-triton` | 혼합 구현의 M=1 decode GEMM만 교체; B=8 decode와 prefill은 Inductor 사용 |

최종 선택과 seed·튜닝 config를 포함한 정확한 재현 명령은 결과 보고서에 기록한다.
이전 후보 파일의 masked-SDPA 시간은 진단 자료이며, 개선된 최종 PyTorch 기준선의 시간과 구분한다.
`--no-compile`과 `--quick` 결과는 최종 컴파일 성능으로 사용하지 않는다.
지원 여부와 시간 제한 때문에 모든 하드웨어 최적화를 망라했다고 주장하지 않는다.
실제 적용·측정 후 탈락·미측정 항목은 각 실행의 분석 보고서에 남긴다.

실측 자료: [2026-10-04 G4 결과](results/2026-10-04-g4/report.md).
