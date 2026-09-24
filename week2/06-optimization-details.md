# ⑥ 추론과 Roofline: 무엇이 병목이고 어디를 고쳐야 하나?

> **한 줄 답.** 추론은 성격이 정반대인 두 단계로 나뉜다. **Prefill**(프롬프트 처리)은 토큰이 많아 산술 강도가 높은 **compute-bound**이고, **Decode**(토큰 1개씩 생성)는 가중치를 통째로 읽어 곱셈 한 줄만 하는 **memory-bound**다. 그래서 최적화도 갈린다. memory-bound 구간에서는 **옮기는 바이트를 줄이거나 한 번 읽은 데이터로 더 많은 일을 하게** 만들고(배치·양자화·KV 캐시 축소·퓨전·speculative decoding), compute-bound 구간에서는 **피크 FLOPs를 높이거나 낭비되는 FLOPs를 없앤다**(저정밀 Tensor Core, 타일 정렬, causal 마스킹, prefix 캐시, 오토튜닝).

```
성능 (FLOPs/s)
   ▲
   │                    ┌──────────────────────  피크 FLOPs/s (compute 천장)
   │                   /│
   │                  / │      ← compute-bound 영역
   │                 /  │         "얼마나 빨리 계산하나"
   │   기울기 =     /   │
   │   HBM 대역폭  /    │
   │             /      │
   │  memory-   /       │
   │  bound    /        │
   │  영역    /         │
   └─────────┴──────────┴─────────────────────▶ 산술 강도 (FLOPs/byte)
          ridge point (임계값)
          = 피크 FLOPs/s ÷ HBM 대역폭

     ●decode                       ●prefill
      (I ≈ 배치 크기)                (I ≈ 수백~수천)
```

이전 단계 ← [⑤ 커널 → 칩 내부](./05-kernel-to-chip.md) · 표기법은 [①](./01-model-to-ops.md)의 B, T, S, D, F, N, K, H, L, V를 그대로 쓴다.

---

## 1. Roofline 모델 한 장 요약

[⑤-4](./05-kernel-to-chip.md)에서 정의한 산술 강도를 그대로 쓴다.

```
산술 강도 I = 연산 FLOPs / HBM에서 옮긴 바이트
달성 가능 성능 = min( 피크 FLOPs/s ,  I × HBM 대역폭 )
ridge point  = 피크 FLOPs/s / HBM 대역폭
```

`I < ridge point`면 연산 유닛이 데이터를 기다린다(**memory-bound**). `I > ridge point`면 메모리는 여유가 있고 연산 유닛이 한계다(**compute-bound**).

| 칩 | 피크 (bf16) | HBM 대역폭 | **ridge point** |
|---|---|---|---|
| A100 80GB | 3.12×10¹⁴ | 2.0×10¹² | 약 160 |
| H100 | 9.90×10¹⁴ | 3.35×10¹² | 약 **295** |
| B200 | 2.3×10¹⁵ | 9.0×10¹² | 약 256 |
| TPU v5e | 1.97×10¹⁴ | 8.2×10¹¹ | 약 240 |
| TPU v5p | 4.59×10¹⁴ | 2.8×10¹² | 약 164 |
| TPU v6e | 9.20×10¹⁴ | 1.6×10¹² | 약 575 |

대략 **200~300 FLOPs/byte**가 요즘 가속기의 경계선이다. 이 숫자 하나만 들고 있으면 커널 하나를 보고 어느 쪽인지 바로 판정할 수 있다.

> **주의.** ridge point는 정밀도에 따라 움직인다. H100의 FP8 피크는 1.98×10¹⁵라 ridge point가 약 590으로 **두 배 올라간다**. 양자화는 memory-bound 구간을 구해주지만, compute-bound가 되기 위한 문턱은 오히려 높인다. 2절의 결론이 여기서 갈린다.

## 2. 추론의 두 단계는 roofline의 양 끝에 있다

### 2-1. Prefill: 토큰 T개를 한 번에

프롬프트 T개 토큰을 한 번에 넣으므로 모든 matmul이 `[B·T, D] × [D, F]` 꼴이다. 가중치를 한 번 읽어 B·T개 행에 재사용한다.

```
FLOPs  ≈ 2 × (B·T) × P          (P = 파라미터 수)
바이트 ≈ 2 × P                   (bf16 가중치 1회 로드)
I      ≈ B·T                     ← 토큰 수에 비례
```

T가 512만 되어도 I ≈ 512 > 295이므로 **compute-bound**다. 프롬프트가 길수록 더 깊이 compute-bound가 된다. 여기서 쓸 수 있는 지표가 **MFU**(Model FLOPs Utilization) = 실제 달성 FLOPs/s ÷ 피크 FLOPs/s다.

### 2-2. Decode: 토큰 1개씩

KV 캐시 덕분에 한 스텝에 새 토큰 1개만 처리한다. 행렬곱이 **행렬-벡터곱**으로 쪼그라든다.

```
FLOPs  ≈ 2 × B × P              (배치 내 시퀀스마다 1토큰)
바이트 ≈ 2 × P                   (가중치는 배치 전체가 공유)
I      ≈ B                       ← 배치 크기와 같다
```

**decode의 산술 강도는 배치 크기 그 자체다.** B=1이면 I=1, ridge point의 1/300이다. 파라미터당 2 FLOPs를 하려고 2바이트를 읽는, roofline 그래프 맨 왼쪽 바닥의 점이다.

구체적으로 Llama-3 70B(bf16, 140 GB)를 H100 한 장(3.35 TB/s)에서 돌리면:

```
토큰당 최소 시간 = 140e9 B / 3.35e12 B/s ≈ 41.8 ms  →  약 24 tok/s
```

이 값은 **GPU FLOPs와 무관하다.** 계산을 아무리 빠르게 해도 가중치를 읽는 시간이 하한이다. 그래서 decode에서는 MFU 대신 **MBU**(Model Bandwidth Utilization) = 초당 읽은 모델 바이트 ÷ 피크 대역폭을 본다. MBU가 0.8 근처면 하드웨어를 다 쓴 것이고, 더 빠르게 하려면 **읽는 바이트 자체를 줄이거나 한 번 읽어 더 많이 써먹는 수밖에 없다.**

### 2-3. Attention은 배치를 늘려도 memory-bound다

가중치와 달리 **KV 캐시는 시퀀스마다 다르다.** 배치로 공유되지 않으므로 배치를 키워도 attention 부분의 강도는 그대로다.

```
FLOPs(QKᵀ + PV) ≈ 4 · S · N · H
바이트(K,V 읽기) ≈ 2 · S · K · H · 2 (bf16)
I ≈ N / K        ← 시퀀스 길이·배치와 무관, GQA 비율로만 결정
```

MHA(N=K)면 **I ≈ 1**, GQA 8:1이면 I ≈ 8. 어느 쪽이든 ridge point에 한참 못 미친다. 그래서 컨텍스트가 길어질수록 decode 시간의 지배항이 "가중치 읽기"에서 "**KV 캐시 읽기**"로 넘어가고, 최적화의 초점도 3-3으로 옮겨간다.

### 2-4. 정리

| | Prefill | Decode |
|---|---|---|
| 연산 모양 | 행렬-행렬 | 행렬-벡터 (배치하면 얇은 행렬) |
| 산술 강도 | ≈ B·T (수백~수천) | ≈ B (가중치), ≈ N/K (attention) |
| 병목 | **compute-bound** | **memory-bound** |
| 지표 | MFU, TTFT(첫 토큰까지 시간) | MBU, TPOT(토큰당 시간) |
| 지배 자원 | Tensor Core / MXU | HBM 대역폭 |
| 늘어나는 축 | 프롬프트 길이 | 생성 길이 × 모델 크기 |

한 요청 안에서 두 단계가 순서대로 일어나므로, **같은 서버가 두 종류의 병목을 번갈아 겪는다.** 이것이 추론 최적화가 학습 최적화보다 까다로운 이유다.

---

## 3. Memory-bound 구간(주로 decode) 최적화

목표는 셋 중 하나다. **① 바이트를 줄인다 ② 한 번 읽은 바이트로 FLOPs를 더 한다(강도↑) ③ 아예 HBM을 안 간다.**

### 3-1. 배치 키우기 — 강도를 직접 올리는 유일한 수단

`I ≈ B`이므로 배치가 곧 강도다. B를 1 → 64로 올리면 토큰당 지연은 거의 그대로인데 처리량은 64배가 된다(가중치 읽기 시간을 64개 시퀀스가 나눠 갖는다).

| 기법 | 내용 |
|---|---|
| **Continuous batching** (in-flight batching) | 요청이 끝나는 대로 빈 슬롯에 새 요청을 밀어 넣는다. 정적 배치는 가장 긴 시퀀스가 끝날 때까지 슬롯이 놀아 유효 배치가 줄어든다. |
| **Chunked prefill / piggyback** | 긴 prefill을 조각내 decode 스텝 사이에 끼워 넣는다. compute-bound 일감과 memory-bound 일감을 한 배치에 섞으면 두 자원을 동시에 쓴다. |
| **Prefill/decode 분리 (disaggregation)** | 반대 전략. 성격이 다른 두 단계를 다른 GPU 풀에 나눠, 각각을 자기 병목에 맞게 튜닝한다. |

배치를 못 키우는 진짜 이유는 대개 **KV 캐시 메모리**다 → 3-3으로 이어진다.

### 3-2. 양자화 — 옮길 바이트를 줄인다

decode 시간이 `가중치 바이트 / 대역폭`이므로, **bf16 → int8이면 그대로 2배, int4면 4배 빨라진다.** 계산 정밀도를 안 낮추고 저장만 줄이는 **weight-only 양자화**(GPTQ, AWQ 등: HBM에서 int4로 읽어 레지스터에서 bf16으로 풀어 계산)가 decode에 특히 잘 맞는다. 어차피 연산 유닛은 놀고 있으므로 역양자화 FLOPs는 공짜에 가깝다.

| | 가중치 바이트 | decode 이득 | 주의 |
|---|---|---|---|
| bf16/fp16 | 2P | 기준 | |
| fp8 | P | 약 2배 | H100/B200은 Tensor Core도 fp8 지원 → prefill도 이득 |
| int8 | P | 약 2배 | 캘리브레이션 필요 |
| int4 (weight-only) | P/2 | 약 4배 | 큰 배치에서는 역양자화 비용이 드러나고, 품질 저하 위험 |

> 2-1의 주의를 다시 보자. 양자화로 ridge point가 올라가므로, **배치가 커져 compute-bound로 넘어간 뒤에는 int4의 이득이 사라진다.** 작은 배치·저지연 서비스에서 제일 크게 먹힌다.

### 3-3. KV 캐시 줄이기 — 길어질수록 지배적

KV 캐시 크기는 `2 · B · S · L · K · H · (바이트)`다. 이것을 줄이면 **읽는 바이트가 줄어들고**(3-1의 attention 병목 완화) **배치를 키울 여유 메모리가 생긴다**(3-1의 강도 상승). 두 경로로 동시에 이득이다.

| 기법 | 무엇을 줄이나 |
|---|---|
| **GQA / MQA** | K를 N보다 작게. 캐시가 N/K배 줄고 attention 강도가 N/K배 오른다. |
| **MLA** (latent attention) | KV를 저차원 잠재 벡터로 압축해 저장, 읽은 뒤 복원. |
| **KV 캐시 양자화** (fp8/int8/int4) | 원소당 바이트. 보통 K보다 V가 양자화에 관대하다. |
| **Sliding window / 로컬 어텐션** | S를 윈도우 크기로 고정. 긴 컨텍스트에서 O(S)를 O(W)로. |
| **PagedAttention** | 크기가 아니라 **단편화**를 줄인다. 캐시를 페이지 블록으로 관리해 미리 잡아둔 낭비를 없애고, 남은 메모리를 배치에 쓴다. |
| **Prefix / prompt 캐싱** | 공통 시스템 프롬프트의 KV를 요청 간 공유. prefill FLOPs도 같이 없앤다(4-5). |
| **캐시 오프로드·축출** | 잘 안 쓰는 블록을 CPU/디스크로. PCIe가 느리므로(⑤-1-5) 신중히. |

### 3-4. 퓨전 — HBM 왕복 자체를 없앤다

[③](./03-ir-to-executable.md)의 퓨전이 추론에서도 그대로 핵심이다. 원소별 연산은 강도가 0.1 수준이라([⑤-5](./05-kernel-to-chip.md)의 벡터 덧셈 0.08) **혼자 두면 100% 대역폭 낭비**다.

- **elementwise 퓨전**: LayerNorm/RMSNorm, residual add, SwiGLU 게이팅, RoPE를 앞뒤 matmul 에필로그·프롤로그로 흡수. 중간 텐서가 HBM에 내려가지 않는다.
- **FlashAttention**: S×S 어텐션 행렬을 HBM에 쓰지 않고 타일 단위로 온칩(SMEM/VMEM)에서 online softmax로 처리. 메모리를 O(S²)에서 O(S)로 줄인다. decode용 **FlashDecoding**은 KV를 시퀀스 축으로 쪼개 병렬로 읽어, 배치가 작아 SM이 노는 상황을 메운다.
- **가중치 온칩 상주**: 작은 모델이나 층 일부는 VMEM/SMEM에 올려둔다. VMEM 대역폭은 HBM의 약 22배라 [⑤-4](./05-kernel-to-chip.md)에서 본 대로 임계 배치가 240 → 10~20으로 떨어진다.

### 3-5. Speculative decoding — 강도를 배수로 올리는 알고리즘적 트릭

작은 draft 모델(또는 n-gram, Medusa 헤드, EAGLE)이 토큰 k개를 먼저 제안하고, 큰 모델이 **한 번의 forward로 k개를 동시에 검증**한다.

```
기존:  가중치 1회 로드 → 토큰 1개      I ≈ B
투기:  가중치 1회 로드 → 토큰 k개 검증  I ≈ B·k
```

decode를 **행렬-벡터에서 얇은 행렬-행렬로 바꾸는** 기법이다. 놀고 있던 연산 유닛을 쓰는 것이므로 memory-bound일 때만 이득이고, 배치가 커서 이미 compute-bound라면 오히려 손해다. 수용률(acceptance rate)이 낮으면 헛FLOPs가 된다.

### 3-6. 런치·오버헤드 제거

강도가 낮은 커널은 실행 시간 자체가 짧아 **커널 런치와 Python 오버헤드가 그대로 드러난다**([④](./04-dispatch-to-kernel.md)의 비동기 디스패치가 못 숨길 만큼). decode 스텝 하나에 수백 개 커널이 나가면 런치 비용만으로 수 ms가 샌다.

- **CUDA Graph / XLA 정적 스케줄**: 스텝 전체를 그래프로 캡처해 한 번에 재생.
- **Persistent kernel / megakernel**: 층 여러 개를 커널 하나로 묶어 런치와 동기화를 없앤다.
- **CPU 오버헤드 제거**: 샘플링·시퀀스 관리 로직을 GPU에 올리거나 다음 스텝과 겹친다.

### 3-7. 병렬화 방식 선택

- **텐서 병렬(TP)**: 가중치를 쪼개므로 GPU당 읽을 바이트가 1/N → decode 지연이 줄어든다. 대신 층마다 all-reduce가 붙어, TP가 커지면 통신이 새 병목이 된다.
- **파이프라인 병렬(PP)**: 통신은 적지만 한 스텝의 지연은 안 줄어든다. 처리량 위주일 때.
- **MoE**: 전체 파라미터 중 활성 전문가만 읽으므로 읽을 바이트가 준다. 단, 배치 내 토큰이 서로 다른 전문가로 흩어지면 전문가별 배치가 작아져 다시 강도가 떨어지고, all-to-all 통신이 추가된다.

### 3-8. 접근 패턴

- **Coalesced access / 레이아웃**: 같은 바이트 수라도 흩어져 읽으면 실효 대역폭이 떨어진다. KV 캐시 레이아웃(헤드-major vs 토큰-major)이 대표적.
- **정렬·패딩**: 128B 트랜잭션 경계에 맞춘다.
- **비동기 복사 / DMA 프리페치**: 다음 타일을 읽는 동안 현재 타일을 계산([⑤-1-4](./05-kernel-to-chip.md)의 소프트웨어 파이프라이닝).

---

## 4. Compute-bound 구간(주로 prefill·큰 배치) 최적화

목표는 셋이다. **① 천장(피크 FLOPs/s)을 올린다 ② 낭비되는 FLOPs를 없앤다 ③ 필요한 FLOPs 자체를 줄인다.**

### 4-1. 정밀도를 낮춰 천장을 올린다

Tensor Core/MXU의 피크는 정밀도에 반비례해 커진다. **여기서는 양자화가 대역폭이 아니라 FLOPs/s를 위한 것**이다(3-2와 목적이 다르다).

| 정밀도 | H100 피크 | 배수 |
|---|---|---|
| fp32 | 67 TFLOP/s | 1× |
| tf32 | 495 TFLOP/s | 약 7× |
| bf16/fp16 | 990 TFLOP/s | 약 15× |
| fp8 | 1979 TFLOP/s | 약 30× |

[`example-pytorch-gpu-a100.md`](./example-pytorch-gpu-a100.md)의 f32/TF32/bf16 비교가 이 표의 실측판이다. 활성값까지 낮은 정밀도로 계산해야(W8A8, fp8 GEMM) 천장이 실제로 올라간다. weight-only 양자화는 계산이 여전히 bf16이라 **compute-bound 구간에서는 도움이 안 된다.**

### 4-2. 연산 유닛을 놀리지 않기 — 타일·정렬

MXU는 128×128, Tensor Core는 warp-level 타일 단위로 움직인다([⑤-1-2](./05-kernel-to-chip.md)). shape이 타일 배수가 아니면 마지막 타일이 패딩으로 채워져 그만큼 버려진다.

- **차원을 128(TPU)·64~128(GPU)의 배수로**: 어휘 크기, 은닉 차원, 헤드 수를 맞춘다. 4095 → 4096 하나로 수십 %가 바뀌기도 한다.
- **Wave quantization**: 타일 개수가 SM 수(H100 132개)의 배수가 아니면 마지막 wave에서 대부분의 SM이 논다. 배치·타일 크기를 SM 수에 맞춘다.
- **Split-K**: M, N이 작고 K만 큰 얇은 GEMM에서 K축을 쪼개 SM을 채운다.
- **레이아웃 변환 비용**: 전치·리포맷 커널이 붙으면 그 자체는 memory-bound다. [③](./03-ir-to-executable.md)의 레이아웃 배정이 중요한 이유.

### 4-3. 커널 스케줄 튜닝

`toy/`의 실험([README](../README.md)의 벤치 표)이 축소판이다. 루프 순서만 바꿔 10배, 블록화로 1.5배 더 나왔다. 실제 스택에서는:

- **오토튜닝**: Inductor `max-autotune`, Triton `autotune`(BLOCK_M/N/K, num_warps, num_stages), CUTLASS/cuBLASLt 알고리즘 선택, XLA cost model.
- **소프트웨어 파이프라이닝 단계 수**: 로드와 MMA를 겹친다. 단계가 많을수록 SMEM을 더 쓰고 occupancy가 준다 — 트레이드오프.
- **레지스터 블로킹 / 마이크로커널**: 레지스터에서의 재사용을 극대화.
- **Occupancy 조절**: 레지스터·SMEM 사용량이 SM당 동시 CTA 수를 정한다. compute-bound 커널은 오히려 occupancy를 낮추고 CTA당 자원을 키우는 쪽이 유리할 때가 많다.
- **Warp specialization / TMA** (Hopper 이상): 데이터 이동 전담 warp와 계산 전담 warp를 나눈다.

### 4-4. 낭비되는 FLOPs 제거

- **Causal 마스킹**: self-attention의 절반은 어차피 마스킹된다. FlashAttention의 causal 모드는 그 블록을 **아예 계산하지 않는다**(순진한 구현은 계산 후 버린다) → attention FLOPs 2배 절약.
- **패딩 토큰 제거**(varlen / unpadded 배치): 길이가 제각각인 요청을 패딩으로 맞추면 패딩만큼 FLOPs가 버려진다. 시퀀스를 이어 붙이고 offset으로 경계를 관리한다.
- **Chunked prefill의 청크 크기**: 너무 작게 쪼개면 다시 강도가 떨어져 compute-bound의 이점을 잃는다.

### 4-5. 필요한 FLOPs 자체를 줄인다

roofline 상의 점을 움직이는 게 아니라 **일감을 지운다.**

| 기법 | 내용 |
|---|---|
| **Prefix/prompt 캐싱** | 공통 프롬프트의 prefill을 재사용. 중복 요청이 많은 서비스에서 TTFT가 가장 크게 떨어진다. |
| **구조적 희소성 (2:4)** | Ampere 이상에서 2:4 희소 Tensor Core로 최대 2배. 재학습·정확도 검증 필요. |
| **가지치기 / 저랭크 분해 / 증류** | 파라미터 수 P를 줄여 prefill FLOPs와 decode 바이트를 동시에 줄인다. |
| **긴 컨텍스트용 어텐션 근사** | 희소 어텐션, 선형 어텐션 등으로 O(S²)를 완화. |
| **조기 종료 / 층 스킵** | 쉬운 토큰에 층을 덜 태운다. |

### 4-6. 통신을 계산에 겹치기

멀티 GPU prefill에서는 TP all-reduce가 계산 사이에 직렬로 끼어 연산 유닛을 세운다. **연산-통신 오버랩**(부분 결과가 나오는 대로 통신 시작, async TP, ring 기반 분해)이 이 구멍을 메운다. [③](./03-ir-to-executable.md)의 SPMD 파티셔너가 이 스케줄을 짠다.

---

## 5. 한눈에 보는 대응표

roofline 위에서 각 기법이 무엇을 바꾸는지로 묶으면 이렇게 된다.

| 무엇을 바꾸나 | 기법 | 언제 효과 |
|---|---|---|
| **점을 오른쪽으로** (강도↑) | 배치 키우기, continuous batching, speculative decoding, GQA, 퓨전, chunked prefill | memory-bound |
| **기울기를 세우기** (실효 대역폭↑) | coalescing, 레이아웃, KV 캐시 양자화, weight-only 양자화, 온칩 상주, 프리페치 | memory-bound |
| **천장을 올리기** (피크 FLOPs↑) | bf16/fp8/int8 **계산**, Tensor Core·MXU 경로 보장, 타일 정렬, wave quantization, 오토튜닝 | compute-bound |
| **일감을 지우기** (FLOPs·바이트↓) | prefix 캐싱, causal 스킵, 패딩 제거, 희소성, 가지치기·증류, MoE | 양쪽 |
| **구멍 메우기** (유휴 제거) | CUDA Graph, persistent kernel, 통신 오버랩, prefill/decode 혼합 배치 | 양쪽 |

반대로 **잘못 짝지으면 손해**인 조합도 분명하다.

| 하지 말 것 | 이유 |
|---|---|
| compute-bound 구간에서 weight-only int4 | 계산은 여전히 bf16인데 역양자화 오버헤드만 추가된다. |
| 이미 큰 배치에서 speculative decoding | 연산 유닛에 여유가 없어 헛FLOPs가 그대로 지연이 된다. |
| memory-bound 구간에서 GEMM 오토튜닝 | 병목이 대역폭인데 계산 스케줄을 다듬는다. |
| decode 지연을 줄이려고 배치만 키우기 | 처리량은 오르지만 토큰당 지연(TPOT)은 나빠진다. |
| KV 캐시를 안 줄인 채 배치만 키우기 | 메모리가 먼저 터지거나 attention 읽기가 새 병목이 된다. |

## 6. 내 커널은 어느 쪽인가 — 판정 절차

1. **손으로 추정한다.** 커널의 FLOPs와 옮긴 바이트를 세서 I를 구하고, 표 1의 ridge point와 비교한다. [①-1](./01-model-to-ops.md)의 세는 법이면 충분하다.
2. **상한 시간을 계산한다.** `max(FLOPs/피크, 바이트/대역폭)`가 이론 하한이다. 실측이 그 2배 안쪽이면 그 자원이 진짜 병목이고, 훨씬 느리면 병목은 제3의 것(런치 오버헤드, 동기화, 통신)이다.
3. **프로파일러로 확인한다.**
   - GPU: Nsight Compute의 roofline 차트, `dram__throughput.avg.pct_of_peak_sustained_elapsed`(대역폭 포화도)와 `sm__pipe_tensor_cycles_active`(Tensor Core 가동률)를 같이 본다. PyTorch Profiler / `torch.profiler`의 타임라인으로 커널 간 빈틈도 확인.
   - TPU: `jax` cost analysis와 memory analysis([`example-jax-tpu-v5e.md`](./example-jax-tpu-v5e.md)에서 `optimal_seconds`로 확인한 그것), XProf/TensorBoard의 step 분석.
4. **한 축만 흔들어 본다.** 배치만 2배로 했을 때 스텝 시간이 거의 그대로면 memory-bound가 확실하다(강도만 오른 것). 반대로 시간이 비례해 늘면 이미 compute-bound다. 정밀도를 bf16 → fp8로 바꿔 시간이 절반이 되면 대역폭 병목, 그대로면 다른 병목.
5. **MFU / MBU를 기록한다.** prefill은 MFU, decode는 MBU. 둘 다 낮으면 병목은 roofline 안이 아니라 **오버헤드**에 있다(3-6).

## 7. 정리

| | Memory-bound (decode) | Compute-bound (prefill) |
|---|---|---|
| 판정 | I < ridge point (≈200~300) | I > ridge point |
| 핵심 질문 | HBM을 몇 바이트 오가나 | 연산 유닛을 몇 % 채우나 |
| 1순위 | 배치 키우기 + KV 캐시 축소 | 저정밀 계산 + 타일 정렬 |
| 2순위 | 양자화(weight-only), 퓨전 | 오토튜닝, causal·패딩 낭비 제거 |
| 3순위 | speculative decoding, CUDA Graph | prefix 캐싱, 통신 오버랩 |
| 지표 | MBU, TPOT | MFU, TTFT |

[①](./01-model-to-ops.md)에서 센 FLOPs와 [⑤](./05-kernel-to-chip.md)에서 본 대역폭을 나누면 이 문서의 모든 판단이 나온다. 그리고 그 판단이 [③](./03-ir-to-executable.md)의 컴파일러에게 퓨전을 어디에 넣을지, [④](./04-dispatch-to-kernel.md)의 런타임에게 무엇을 겹칠지를 알려준다. **Roofline은 답이 아니라, 어느 질문을 할지 정해주는 도구다.**

---

## 참고 자료

- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (CACM 2009) — 원논문
- Scaling Book [4장 Transformer Math](https://jax-ml.github.io/scaling-book/transformers/), [7장 Inference](https://jax-ml.github.io/scaling-book/inference/), [9장 Profiling](https://jax-ml.github.io/scaling-book/profiling/)
- [FlashAttention](https://arxiv.org/abs/2205.14135) / [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [NVIDIA: Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/) — 타일·wave quantization
