# ① 모델 → 연산: Transformer를 펼치면 어떤 연산과 텐서가 나올까?

> **한 줄 답.** Transformer 한 층은 결국 **7개의 큰 행렬곱(matmul)** 과 그 사이를 잇는 소량의 원소별(elementwise) 연산으로 펼쳐진다. 학습 FLOPs는 거의 전부 이 행렬곱에서 나오고, 그 합은 `6 × 토큰 수 × 파라미터 수`로 근사된다.

```
토큰 [B, T]
   │ 임베딩 (lookup, FLOPs 무시)
   ▼
x [B, T, D] ─┬─ Q = x·W_Q [D, N, H]  ┐
             ├─ K = x·W_K [D, K, H]  ├─ attention 프로젝션 (matmul 4개)
             ├─ V = x·W_V [D, K, H]  │
             │                       │
             │  S = Q·Kᵀ [B, N, T, S]  ─ dot-product attention (matmul 2개, 파라미터 없음)
             │  P = softmax(S)          ─ elementwise + reduction
             │  A = P·V [B, T, N, H]
             │
             └─ out = A·W_O [N, H, D] ┘
   │ residual add, layernorm  (elementwise, FLOPs 무시)
   ▼
x [B, T, D] ─┬─ h1 = x·W_in1 [D, F] ┐
             ├─ h2 = x·W_in2 [D, F] ├─ gated MLP (matmul 3개)
             │  g  = σ(h1) ⊙ h2      │  elementwise
             └─ y  = g·W_out [F, D]  ┘
   │ residual add, layernorm
   ▼
… L층 반복 … ─▶ logits = x·W_unembed [D, V]
```

다음 단계 → [② 연산 → 그래프·IR](./02-ops-to-graph-ir.md)

---

## 1. FLOPs 세는 법 (Counting Dots)

모든 계산은 결국 "곱하고 더하기"이므로, 내적 하나부터 센다.

| 연산 | FLOPs | 읽어야 할 데이터 |
|---|---|---|
| 내적 `x·y` (길이 P) | 2P | 2P |
| 행렬-벡터 `A[N,P]·x[P]` | 2NP | NP + P |
| 행렬-행렬 `A[N,P]·B[P,M]` | 2NPM | NP + PM |

곱셈 1번 + 덧셈 1번이므로 원소당 2 FLOPs다.

**일반 규칙.** 행렬곱(einsum)의 축을 세 종류로 나눈다.

- **contracting 차원**: 두 입력에 모두 있고 출력에는 없는 축(위에서 P). 이 축을 따라 합산한다.
- **batching 차원**: 두 입력과 출력에 모두 있는 축(예: 배치 B, 헤드 N).
- 나머지 차원: 한쪽 입력과 출력에만 있는 축(N, M).

FLOPs는 "**모든 축 크기의 곱의 2배**, 단 batch와 contracting 축은 한 번만 센다"이다.

**왜 중요한가.** `A[N,N]·B[N,N]`은 연산이 O(N³), 데이터 이동이 O(N²)다. 행렬이 커질수록 연산이 데이터 이동보다 빨리 늘어나므로, 충분히 큰 행렬곱은 **compute-bound**가 된다. 반대로 원소별 연산은 연산과 데이터가 모두 O(N)이라 항상 memory-bound다. 이 구분이 [⑤ 칩 내부](./05-kernel-to-chip.md)의 arithmetic intensity 논의로 이어진다.

## 2. Forward vs Backward

학습은 forward만 하지 않는다. `C = A·B` (A는 [N,P], B는 [P,M]) 하나에 대해:

| 단계 | 계산 | FLOPs |
|---|---|---|
| forward | `C = A·B` | 2NPM |
| backward (∂L/∂B) | `Aᵀ · (∂L/∂C)` | 2NPM |
| backward (∂L/∂A) | `(∂L/∂C) · Bᵀ` | 2NPM |
| **학습 합계** | | **6NPM** |

즉 **학습 FLOPs = forward의 3배**다. 뒤에 나오는 "6ND"의 6이 여기서 온다. 아래 표의 "학습 FLOPs" 열은 모두 이 3배가 반영된 값이다.

## 3. 표기법

| 기호 | 의미 |
|---|---|
| B | 배치 크기 |
| T, S | 시퀀스 길이 (query, key/value). self-attention이면 T = S |
| D | 모델 차원 (d_model) |
| F | MLP 은닉 차원 (보통 4D 근처) |
| N | query 헤드 수 |
| K | key/value 헤드 수 (GQA에서는 K < N, MHA에서는 K = N) |
| H | 헤드당 차원 (보통 D = N·H) |
| L | 층 수 |
| V | 어휘 크기 |

## 4. Transformer 한 층의 연산과 텐서 (Transformer Accounting)

### 4-1. Gated MLP

| 연산 | 입력 텐서 | 가중치 | 출력 텐서 | 학습 FLOPs | 파라미터 |
|---|---|---|---|---|---|
| `x · W_in1` | [B,T,D] | [D,F] | [B,T,F] | 6BTDF | DF |
| `x · W_in2` | [B,T,D] | [D,F] | [B,T,F] | 6BTDF | DF |
| `σ(h1) ⊙ h2` | [B,T,F] ×2 | — | [B,T,F] | O(BTF) | — |
| `g · W_out` | [B,T,F] | [F,D] | [B,T,D] | 6BTDF | DF |
| **합계** | | | | **≈ 18BTDF** | **3DF** |

gating을 쓰지 않는 모델은 `W_in` 하나만 있어서 파라미터가 2DF, FLOPs가 12BTDF다.

### 4-2. Attention 프로젝션 (Q, K, V, O)

| 연산 | 입력 텐서 | 가중치 | 출력 텐서 | 학습 FLOPs | 파라미터 |
|---|---|---|---|---|---|
| Q | [B,T,D] | [D,N,H] | [B,T,N,H] | 6BTDNH | DNH |
| K | [B,S,D] | [D,K,H] | [B,S,K,H] | 6BTDKH | DKH |
| V | [B,S,D] | [D,K,H] | [B,S,K,H] | 6BTDKH | DKH |
| O | [B,T,N,H] | [N,H,D] | [B,T,D] | 6BTDNH | DNH |
| **합계** | | | | **12BTD(N+K)H** | **2D(N+K)H** |

### 4-3. Dot-product attention (T = S 가정)

| 연산 | 입력 텐서 | 출력 텐서 | 학습 FLOPs | 파라미터 |
|---|---|---|---|---|
| `Q · Kᵀ` (B, N을 batching) | [B,T,N,H], [B,S,K,H] | [B,N,T,S] | 6BT²NH | 0 |
| softmax | [B,N,T,S] | [B,N,T,S] | O(BT²N) | 0 |
| `P · V` | [B,N,T,S], [B,S,K,H] | [B,T,N,H] | 6BT²NH | 0 |
| **합계** | | | **≈ 12BT²NH** | **0** |

이 두 matmul은 **가중치가 없다**. 활성화끼리의 곱이라서 파라미터 수에는 잡히지 않지만, T가 커지면 FLOPs와 메모리에서 지배적이 된다.

### 4-4. 나머지

| 구성 요소 | 파라미터 | 비고 |
|---|---|---|
| layernorm (층당 2개) | 2D | FLOPs O(BTD), 무시 가능 |
| 임베딩 / unembedding | DV (공유 시 1개) | unembedding matmul 학습 FLOPs 6BTDV |

## 5. "6ND" 법칙 유도

층당 matmul FLOPs(attention dot-product 제외)를 모으면:

```
(18BTDF + 12BTD(N+K)H) · L
  = 6 · BT · (3DF + 2D(N+K)H) · L
  = 6 · (토큰 수) · (층당 파라미터 수) · L
```

`3DF + 2D(N+K)H`가 정확히 MLP + attention 프로젝션의 파라미터 수이므로, **학습 FLOPs ≈ 6 × 토큰 수 × 파라미터 수**가 된다. 추론(forward만)이면 2 × 토큰 수 × 파라미터 수다.

F = 4D, H = D/N, K = N인 전형적인 구조에서 층당 파라미터는 다음과 같다.

| 구성 | 파라미터 | F = 4D일 때 | F = 8D/3 (LLaMA 계열)일 때 |
|---|---|---|---|
| gated MLP | 3DF | 12D² | 8D² |
| attention 프로젝션 | 4DNH = 4D² | 4D² | 4D² |
| layernorm | 2D | 무시 | 무시 |
| **층당 합계** | | **≈ 16D²** | **≈ 12D²** |

핵심은 **파라미터 수가 D²에 비례하고, attention 프로젝션은 MLP의 1/3~1/2 수준**이라는 점이다.

## 6. Attention이 MLP를 이기는 조건

attention dot-product FLOPs(12BT²NH)와 나머지 matmul FLOPs를 비교하면, F = 4D, D = NH, K = N일 때:

```
attention FLOPs / matmul FLOPs = T / (8D)
```

따라서 **T > 8D일 때만 attention이 지배**한다.

| 모델 | D | 손익분기 T |
|---|---|---|
| D ≈ 8k인 대형 모델 | 8192 | ≈ 64K 토큰 |
| Gemma-27B | 4608 | ≈ 37K 토큰 |

일반적인 4K~8K 컨텍스트 학습에서는 MLP와 프로젝션이 FLOPs를 지배한다. 즉 "Transformer 최적화 = 큰 matmul 최적화"가 대부분 맞는 말이다.

## 7. 추론에서 추가로 생기는 텐서: KV cache

생성(decode) 시 이전 토큰의 K, V를 다시 계산하지 않고 저장해 둔다.

```
KV cache shape = [2, S, L, K, H]      (2는 K와 V)
바이트 수      = 2 · S · L · K · H · (dtype 크기)
```

예: S = 8k, L = 64, D = 8192, K = N = 64 (H = 128), int8이면 약 8 GiB다. K를 N보다 작게 잡는 GQA(grouped-query attention)가 KV cache를 K/N 배로 줄이므로 추론 메모리에서 결정적이다.

## 8. 요약: 컴파일러가 받게 될 것

Transformer 한 층에서 컴파일러가 보게 되는 것은 결국 다음이다.

| 종류 | 개수 (층당) | 특성 |
|---|---|---|
| 가중치 matmul (Q, K, V, O, in1, in2, out) | 7 | 큰 contracting 차원, compute-bound 후보 |
| 활성화 matmul (QKᵀ, PV) | 2 | 파라미터 없음, T²에 비례 |
| elementwise / reduction (softmax, gating, layernorm, residual) | 다수 | memory-bound, 퓨전 대상 |

이 연산들은 Python 코드에서 `x @ w`, `jnp.einsum`, `F.linear`, `F.scaled_dot_product_attention`으로 쓰이지만, 컴파일러는 그것을 `dot_general` / `aten.mm` 같은 primitive의 그래프로 받는다. 그 변환 과정이 [② 연산 → 그래프·IR](./02-ops-to-graph-ir.md)이다.

---

## 참고 자료

- Scaling Book 4장 *Transformer Math*: **Counting Dots**, **Transformer Accounting** 절 — https://jax-ml.github.io/scaling-book/transformers/
