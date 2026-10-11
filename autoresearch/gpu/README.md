# GPU autoresearch

| 디렉터리 | 과제 | 실행 파일 |
|---|---|---|
| [triton/](triton/program.md) | A100 Triton `ReLU(XW+b)` 커널 최적화 | `triton/bench.py` |
| [cuda/](cuda/README.md) | A100 native CUDA SwiGLU를 사용하는 LLM 전체 모델 | `cuda/bench.py` |
| [cuda_g4/](cuda_g4/README.md) | G4 SM120 native CUDA SwiGLU를 사용하는 LLM 전체 모델 | `cuda_g4/bench.py` |

각 디렉터리에서 `program.md`를 읽고 해당 `bench.py`를 실행한다.
전체 모델 Triton 실험은 [../llm_gpu/](../llm_gpu/program.md)에서 관리한다.
CUDA 정확성·성능 측정 기록은 [A100·G4 검증 보고서](../results/2026-10-11-swiglu/README.md)에 있다.
