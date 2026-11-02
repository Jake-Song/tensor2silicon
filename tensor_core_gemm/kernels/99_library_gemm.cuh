#pragma once

#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

inline void ltCheck(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLASLt failed: status %d\n", int(status));
    std::exit(EXIT_FAILURE);
  }
}

// One cached plan for the benchmark's current device and shape. The first call
// selects an algorithm and allocates workspace; subsequent calls only dispatch.
// This default-stream benchmark adapter is not intended for concurrent host use.
struct LtGemmPlan {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulAlgo_t algo{};
  void *workspace = nullptr;
  static constexpr size_t workspaceBytes = 32ull << 20;
  int m = 0, n = 0, k = 0, device = -1;

  LtGemmPlan() { ltCheck(cublasLtCreate(&handle)); }
  void clear() {
    if (op) ltCheck(cublasLtMatmulDescDestroy(op));
    if (a) ltCheck(cublasLtMatrixLayoutDestroy(a));
    if (b) ltCheck(cublasLtMatrixLayoutDestroy(b));
    if (c) ltCheck(cublasLtMatrixLayoutDestroy(c));
    op = nullptr; a = b = c = nullptr;
  }
  ~LtGemmPlan() {
    clear();
    if (workspace) cudaFree(workspace);
    cublasLtDestroy(handle);
  }
  void prepare(int M, int N, int K) {
    int current;
    cudaCheck(cudaGetDevice(&current));
    if (m == M && n == N && k == K && device == current) return;
    if (device != -1 && device != current) {
      fprintf(stderr, "cuBLASLt adapter requires a fixed CUDA device\n");
      std::exit(EXIT_FAILURE);
    }
    clear();
    if (!workspace) cudaCheck(cudaMalloc(&workspace, workspaceBytes));
    ltCheck(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    ltCheck(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, M, K, K));
    ltCheck(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, K, N, N));
    ltCheck(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, M, N, N));
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    for (auto layout : {a, b, c})
      ltCheck(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                             &order, sizeof(order)));
    cublasLtMatmulPreference_t preference;
    ltCheck(cublasLtMatmulPreferenceCreate(&preference));
    const size_t limit = workspaceBytes;
    ltCheck(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)));
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    ltCheck(cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c,
                                          preference, 1, &result, &count));
    ltCheck(cublasLtMatmulPreferenceDestroy(preference));
    if (count == 0) {
      fprintf(stderr, "No cuBLASLt algorithm for %dx%dx%d\n", M, N, K);
      std::exit(EXIT_FAILURE);
    }
    ltCheck(result.state);
    algo = result.algo;
    m = M; n = N; k = K; device = current;
  }
};

inline void runCublasLt(int M, int N, int K, float alpha, float *A,
                        float *B, float beta, float *C) {
  static LtGemmPlan plan;
  plan.prepare(M, N, K);
  ltCheck(cublasLtMatmul(plan.handle, plan.op, &alpha, A, plan.a, B, plan.b,
      &beta, C, plan.c, C, plan.c, &plan.algo,
      plan.workspace, plan.workspaceBytes, nullptr));
}

#ifdef TC_ENABLE_CUTLASS
#include <cutlass/gemm/device/gemm.h>

// FP32 storage -> single-TF32 products, FP32 accumulation. Explicit OpMultiplyAdd
// avoids the multi-pass TF32x3 accuracy mode. CUTLASS handles conversion, cp.async,
// shared-memory layout, and epilogue; all matrices remain row-major.
using CutlassTf32Gemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3, 4, 4, false, cutlass::arch::OpMultiplyAdd>;

inline void cutlassCheck(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS failed: %s\n", cutlassGetStatusString(status));
    std::exit(EXIT_FAILURE);
  }
}

inline void runCutlass(int M, int N, int K, float alpha, float *A,
                       float *B, float beta, float *C) {
  CutlassTf32Gemm gemm;
  CutlassTf32Gemm::Arguments args{
      {M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {alpha, beta}};
  cutlassCheck(gemm.can_implement(args));
  // No split-K: no workspace is required.
  cutlassCheck(gemm(args));
}
#endif
