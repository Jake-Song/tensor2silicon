# autoresearch: 에이전트가 스스로 커널을 최적화하는 실험

[Karpathy의 autoresearch](https://github.com/karpathy/autoresearch)를 커널 최적화에 옮긴 것이다.
Claude Code가 가속기 옆(Colab 터미널)에서 **커널 파일 하나만** 고치고, 고정된 벤치마크로 채점해 빨라진 변경만 남긴다.

```
┌─ 가설 ("타일을 128×128로 키우면 Tensor Core 활용이 오른다")
│  kernel.py 수정 → git commit
│  timeout 300 python bench.py > run.log      정확도 검사 · 반칙 검사 · 시간 측정
│  1% 이상 빨라짐 & status: ok → 유지 (브랜치 전진)
│  아니면 git reset --hard HEAD~1
└─ results.tsv 한 줄 + report.md 기록 (실패도) ── 반복 (기본 30회)
```

## 과제: Y = ReLU(XW + b)

저장소 전체에서 쓰는 식을 bf16 행렬곱 + bias + ReLU를 한 커널로 합친 형태로 푼다.
x `[M,K]`, w `[K,N]`, b `[N]`은 bf16, 누적은 fp32, 출력 y `[M,N]`은 bf16이다.
시간은 M = N = K = 4096에서 재고, (1024, 2048, 512)에서도 맞아야 한다.

| | GPU | TPU |
|---|---|---|
| 디렉터리 | [`gpu/`](gpu) | [`tpu/`](tpu) |
| Colab 런타임 | A100 | v6e-1 |
| 커널 언어 | Triton | Pallas (Mosaic) |
| 시작 커널 | 64×64×32 타일, `num_stages=2`, autotune 없음 | 128³ 블록, fp32 VMEM 누산기 |
| 비교 기준 (`speedup`) | cuBLAS `addmm` + ReLU | XLA `relu(x @ w + b)` |
| 반칙 검사 | profiler로 본 CUDA 커널 이름에 `gemm`/`cutlass`/`cublas` 등이 있으면 실패 | StableHLO에 `tpu_custom_call`이 없거나 `dot_general`/`convolution`이 있으면 실패 |
| `pct_peak` 기준 | 312 TFLOP/s | 918 TFLOP/s |

각 디렉터리의 파일:

| 파일 | 역할 | 누가 고치나 |
|---|---|---|
| `program.md` | 에이전트 지침: 준비, 규칙, 루프, 기록 형식, 하드웨어 메모 | 사람 |
| `bench.py` | 고정 채점기: 정확도(2개 shape), 반칙 검사, 시간 측정, 실행 후 새 데이터로 재검사. GPU는 입력을 5번 새로 할당해 각각 잰 중앙값을 쓴다. 할당 하나만 재면 실행마다 최대 20% 흔들렸다 | 아무도 |
| `kernel.py` | `relu_linear(x, w, b)` | **에이전트만** |
| `.claude/settings.json` | 허용: bench 실행·git·`kernel.py`/`results.tsv`/`report.md` 편집. 금지: `bench.py`/`program.md` 편집 | 사람 |
| `results.tsv`, `run.log` | 실행 중 생성 (gitignore) | 에이전트 |
| `report.md` | 모든 시도의 가설·결과·교훈. 실행 중에는 untracked, 끝나면 브랜치에 커밋 | 에이전트 |

## 실행

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/tensor2silicon/blob/main/notebooks/autoresearch_gpu_a100.ipynb) [A100 launcher](../notebooks/autoresearch_gpu_a100.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jake-Song/tensor2silicon/blob/main/notebooks/autoresearch_tpu_v6e.ipynb) [TPU v6e launcher](../notebooks/autoresearch_tpu_v6e.ipynb)

노트북이 저장소 clone, 베이스라인 측정, Claude Code 설치, 진행 그래프, Drive 저장을 맡는다. 에이전트는 Colab 터미널에서 시작한다.

```bash
cd /content/tensor2silicon/autoresearch/gpu     # 또는 tpu
export PATH="$HOME/.local/bin:$PATH"
claude
```

```
Read program.md and kick off a new experiment run. Tag: sep23. Stop after 30 experiments.
```

TPU 주의사항:

- 2026-09 기준 Colab 기본 이미지(jax 0.7.2 + libtpu 0.0.21.1)에서는 Pallas가 `Unsupported version: expected <= 7 but got 8`로 실패한다. 노트북 0-1 셀이 이를 감지해 `jax[tpu]`를 올린다. 확인한 조합은 jax 0.11.2 + libtpu 0.0.48이다.
- TPU는 한 번에 한 프로세스만 잡을 수 있다. 노트북 커널에서 jax를 import하거나, 에이전트가 도는 동안 노트북에서 `bench.py`를 돌리지 않는다.

브라우저 탭 없이 돌리려면 로컬에서 Colab CLI로 VM을 만들고 tmux 콘솔에서 같은 명령을 실행한다.

```bash
colab new --gpu A100        # 또는 --tpu v6e1
colab console               # 원격 tmux 셸: git clone 후 위 명령 실행
```

## 결과 읽기

`results.tsv`의 `status` 열:

- `keep`: 가장 빠른 기록보다 1% 넘게 빨라서 커밋을 남겼다.
- `discard`: 정상 실행됐지만 빠르지 않아서 되돌렸다.
- `incorrect`: 결과가 틀렸다.
- `forbidden`: 벤더 GEMM이나 XLA dot을 썼다.
- `crash`: 컴파일 또는 실행에 실패했다.

`report.md`는 사람이 읽는 실험 보고서다.

- **Summary**: 시작 → 최고 기록, 상태별 실험 수, 핵심 교훈, 남은 아이디어.
- **Kept**: 유지된 개선과 직전 최고 대비 향상률.
- **Failed trials**: 느려졌거나(`discard`) 틀렸거나 실패한 시도와 그 이유.
- **Experiment log**: 실험마다 가설 → 결과 → 교훈.

되돌린 커밋은 브랜치에서 사라지므로 실패한 시도는 이 문서에만 남는다. 로그 기록은 `git reset` 뒤에 하므로 reset이 기록을 지우지 않는다.

브랜치 `autoresearch/<tag>-gpu`의 커밋 로그가 성공한 최적화의 순서다. `git log --oneline`과 `git diff <baseline>..HEAD -- kernel.py`로 에이전트가 무엇을 바꿨는지 본다.

## 새 과제 추가

`gpu/`나 `tpu/`를 복사한 뒤 `bench.py`의 `SHAPES`/`reference`/벤더 기준선, `kernel.py`의 시작 커널, `program.md`의 과제 설명을 바꾼다. 요약 블록 형식은 그대로 둔다.
