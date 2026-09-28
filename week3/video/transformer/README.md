# Transformer 연산 지도 — 파라미터와 FLOPs

저장소의 decoder-only LLM을 한 층씩 펼쳐 보는 한국어 10분 영상입니다.
어두운 2D 도식, SunHi 여성 음성, 한국어 자막, 9개 챕터로 구성합니다.

- [완성 영상](./transformer-ops.ko.mp4)
- [첫 60초 미리보기](./transformer-ops-preview.ko.mp4)
- [읽을 대본](./script.ko.md)
- [한국어 자막](./transformer.ko.srt)
- [계산 데이터](./facts.json) · [검증 결과](./validation.json)

## 설명 기준

`llm_roofline/spec.py`의 `colab` 설정: D=2048, F=5632, h=16, d=128,
L=8, V=32000. MHA, LayerNorm, RoPE, SwiGLU를 쓰고 선형층에는 bias가 없습니다.
Embedding과 LM head는 가중치를 공유하지 않습니다. 실제 LLaMA와 동일한 모델은 아닙니다.

한 층의 논리적 matmul은 Q/K/V, QKᵀ, PV, O, Gate/Up/Down의 9개입니다.
헤드별 연산을 batched matmul 하나로 셉니다. 전체 forward는 9×8+1=73개이며
실제 GPU 커널 실행 횟수와 다릅니다. 파라미터는 층당 51,388,416개,
전체 542,183,424개입니다.

곱셈과 덧셈 각각 1 FLOP, matmul의 대표값은 2mkn입니다. 순전파만 다루고,
prefill은 모든 위치의 logits를 계산하는 현재 구현 기준입니다. dense attention의
전체 점수표를 계산한 후 causal mask를 적용하므로 FLOPs를 절반으로 줄이지 않습니다.
LayerNorm, Softmax, SwiGLU, RoPE 등의 FLOPs는 저장소의 근사 집계 방식입니다.
0 FLOPs인 lookup·copy·재배치도 메모리 접근 비용은 있습니다. FLOPs는 실행시간이 아닙니다.

| 실행 | B | T | S | matmul FLOPs | 기타 연산 추정 FLOPs |
|---|---:|---:|---:|---:|---:|
| Prefill | 1 | 2048 | 2048 | 2,226,940,542,976 | 4,018,143,232 |
| Decode 1회 | 1 | 1 | 2048 | 1,087,373,312 | 1,961,984 |

`decode`는 비교를 위해 기존 colab preset의 batch=8을 **1로 변경**합니다.
KV cache는 파라미터가 아닌 activation입니다. 학습·역전파, GQA의 상세 유도,
cross-attention, sampling은 이번 영상의 계산 범위 밖입니다.

## 재생성

프로젝트 루트에서 실행합니다. Pillow·NumPy, FFmpeg/FFprobe(libass 포함),
UnDotum·DejaVuSans 폰트가 필요합니다. 음성 합성에는 인터넷과 edge-tts가 필요합니다.

```bash
uv run --with edge-tts python week3/video/transformer/build.py audio
.venv/bin/python week3/video/transformer/build.py preview
.venv/bin/python week3/video/transformer/build.py render
.venv/bin/python week3/video/transformer/build.py assemble
.venv/bin/python week3/video/transformer/build.py check
```

`render --scene 10`으로 한 장면만 렌더링할 수 있습니다. `sync`는 기존 음성으로
타임라인·자막·대본·챕터를 다시 계산합니다. `lesson.py`가 대본 원본이며,
`facts.py`는 저장소 연산 명세에서 숫자를 계산하고 파라미터 shape의 합과 대조합니다.
`lesson.json`, `facts.json`, `timeline.json`, 대본과 SRT는 `sync`에서 생성합니다.

음성은 단어 경계 정보를 저장해 자막과 화면 등장 시점에 사용합니다.
최대 1.15배까지만 속도를 조정하고, 더 길면 대본을 줄이도록 제작을 중단합니다.
새 음성·프레임을 생성하며 기존 영상의 장면을 입력으로 사용하지 않습니다.
음성은 대본·목소리 해시, 장면은 코드·타임라인·계산 데이터·폰트 해시로 캐시합니다.
코드나 대본 변경 후 `audio → render → assemble → check` 순으로 재실행하세요.

MP4는 1920×1080, 30fps, H.264/AAC, 600초입니다. 음성은 48kHz,
음량 목표 -16 LUFS이며 배경음악은 없습니다. 자막은 영상에 표시하고 SRT도 제공합니다.
MP4·음성·클립·미리보기 이미지·로그는 로컬 산출물로 Git에서 제외합니다.

## 근거

- [Attention Is All You Need](https://arxiv.org/html/1706.03762v7)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer](https://arxiv.org/abs/2104.09864)
- 모델의 정확한 연산 순서: `llm_roofline/torch_llm.py`의 `forward`
- 계산량 기준: `llm_roofline/spec.py`의 `op_specs` / 파라미터: `init_params`
