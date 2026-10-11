# A100 LLM CUDA 커널 autoresearch

기존 `llm_gpu`와 동일한 전체 모델·BF16·고정 검증기를 사용한다.
시작 후보는 **SwiGLU만 CUDA C++로 교체**하고 GEMM·attention 등은 PyTorch를 사용한다.
전체 native CUDA 구현은 아니며, 시작 후보에는 torch.compile/CUDA Graph를 적용하지 않는다.
SwiGLU 계산은 [swiglu.cu](swiglu.cu)의 native CUDA 커널에서 수행한다.
[bindings.cpp](bindings.cpp)는 텐서 검사·출력 할당·CUDA 호출만 담당하고,
[model.py](model.py)는 확장 로딩과 전체 모델 연결을 담당한다.

```bash
cd autoresearch/gpu/cuda
python3 bench.py --phase decode --smoke
mkdir -p runs
timeout 900 python3 bench.py --phase decode --json runs/decode-baseline.json > runs/decode-baseline.log 2>&1
timeout 900 python3 bench.py --phase prefill --json runs/prefill-baseline.json > runs/prefill-baseline.log 2>&1
```

CUDA PyTorch, nvcc, C++ 컴파일러, Ninja, Triton(공유 검증기 버전 기록용)이 필요하다.
최초 호출에서 sm_80 + compute_80 PTX로 빌드하고 확장을 캐시한다.
SiLU 결과를 BF16으로 반올림한 후 곱해 원본 중간 정밀도를 보존한다.

점수 측정은 A100에서만 허용한다. Prefill은 B=1,T=S=2048, decode는 B=8,T=1,S=2048이다.
전체 logits·모든 KV cache·GQA·변경된 입력을 검증하며, 기준선은 eager PyTorch다.
컴파일과 warmup은 정상 latency에서 제외한다. 실험 절차는 [program.md](program.md)를 따른다.
실제 GPU에서 컴파일·정확성·성능 검증을 완료했다.
전체 모델 prefill/decode를 각각 두 번 검증했고, 모든 logits·KV cache가 기준선과 일치했다.
측정값과 원시 로그는 [2026-10-11 A100·G4 결과](../../results/2026-10-11-swiglu/README.md)에 있다.
