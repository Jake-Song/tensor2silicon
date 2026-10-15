#include "runner.cuh"

#include "kernels/01_wmma_naive.cuh"
#include "kernels/02_wmma_smem_warptile.cuh"
#include "kernels/03_mma_sync_tf32.cuh"
#include "kernels/04_cp_async_pipeline.cuh"
#include "kernels/05_swizzle_tuned.cuh"
#include "kernels/06_vectorized_frag.cuh"
#include "kernels/07_frag_double_buffer.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sys/time.h>

void cudaCheckImpl(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

void printDeviceInfo() {
  int deviceId;
  cudaCheck(cudaGetDevice(&deviceId));
  cudaDeviceProp props{};
  cudaCheck(cudaGetDeviceProperties(&props, deviceId));
  printf("Device %d: %s (sm_%d%d, %d SMs, smem/block opt-in %zu KB)\n",
         deviceId, props.name, props.major, props.minor,
         props.multiProcessorCount, props.sharedMemPerBlockOptin / 1024);
  if (props.major * 10 + props.minor < 80) {
    printf("This code requires sm_80+ (TF32 mma.sync, cp.async).\n");
    exit(EXIT_FAILURE);
  }
}

// SGEMM_CUDA와 같은 입력 분포: ±{0..4}.{00..04}
void randomize_matrix(float *mat, long n) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (long i = 0; i < n; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

// TF32는 mantissa가 10비트라 FP32 기준과 원소별 절대 오차 0.01로는 비교할 수 없다.
// 출력 최댓값으로 정규화한 최대 오차를 사용한다.
VerifyResult verify_matrix(const float *ref, const float *out, long n,
                           double tol) {
  VerifyResult r{0.0, 0.0, 0.0, true};
  for (long i = 0; i < n; i++) {
    const double diff = std::fabs((double)ref[i] - (double)out[i]);
    if (std::isnan(out[i]) || std::isinf(out[i]))
      r.ok = false;
    r.maxAbsDiff = std::fmax(r.maxAbsDiff, diff);
    r.maxAbsRef = std::fmax(r.maxAbsRef, std::fabs((double)ref[i]));
  }
  r.relErr = r.maxAbsDiff / std::fmax(r.maxAbsRef, 1e-30);
  r.ok = r.ok && r.relErr < tol;
  return r;
}

static void require_divisible(const char *name, int M, int N, int K, int TM,
                              int TN, int TK) {
  if (M % TM != 0 || N % TN != 0 || K % TK != 0) {
    fprintf(stderr,
            "%s requires M%%%d == 0, N%%%d == 0, K%%%d == 0 (got %d, %d, %d)\n",
            name, TM, TN, TK, M, N, K);
    exit(EXIT_FAILURE);
  }
}

// dynamic SMEM이 48KB를 넘는 커널은 한 번만 opt-in 속성을 설정한다.
template <typename Kernel>
static void set_max_dynamic_smem(Kernel kernel, int bytes) {
  cudaCheck(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

// row-major C = A @ B 를 column-major cuBLAS로 계산하려면 C^T = B^T @ A^T.
static void runCublas(cublasHandle_t handle, cublasComputeType_t compute,
                      int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C) {
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
               CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runWmmaNaive(int M, int N, int K, float alpha, float *A, float *B,
                  float beta, float *C) {
  constexpr int WARPS_M = 2, WARPS_N = 2;
  require_divisible("kernel 1", M, N, K, 16 * WARPS_M, 16 * WARPS_N, 8);
  dim3 gridDim(N / (16 * WARPS_N), M / (16 * WARPS_M));
  dim3 blockDim(WARPS_M * WARPS_N * WARPSIZE);
  tcgemmWmmaNaive<WARPS_M, WARPS_N>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runWmmaSmem(int M, int N, int K, float alpha, float *A, float *B,
                 float beta, float *C) {
  constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 32;
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARPSIZE;
  require_divisible("kernel 2", M, N, K, BM, BN, BK);
  dim3 gridDim(N / BN, M / BM);
  tcgemmWmmaSmem<BM, BN, BK, WM, WN, NUM_THREADS>
      <<<gridDim, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

void runMmaSync(int M, int N, int K, float alpha, float *A, float *B,
                float beta, float *C) {
  constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 32;
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARPSIZE;
  require_divisible("kernel 3", M, N, K, BM, BN, BK);
  dim3 gridDim(N / BN, M / BM);
  tcgemmMmaSync<BM, BN, BK, WM, WN, NUM_THREADS>
      <<<gridDim, NUM_THREADS>>>(M, N, K, alpha, A, B, beta, C);
}

void runCpAsync(int M, int N, int K, float alpha, float *A, float *B,
                float beta, float *C) {
  constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 32, STAGES = 3;
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARPSIZE;
  constexpr int SMEM =
      STAGES * (BM * (BK + 4) + BK * (BN + 8)) * (int)sizeof(float);
  require_divisible("kernel 4", M, N, K, BM, BN, BK);
  auto kernel = tcgemmCpAsync<BM, BN, BK, WM, WN, STAGES, NUM_THREADS>;
  static bool configured = false;
  if (!configured) {
    set_max_dynamic_smem(kernel, SMEM);
    configured = true;
  }
  dim3 gridDim(N / BN, M / BM);
  kernel<<<gridDim, NUM_THREADS, SMEM>>>(M, N, K, alpha, A, B, beta, C);
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
static void launchSwizzle(int M, int N, int K, float alpha, float *A, float *B,
                          float beta, float *C) {
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARPSIZE;
  constexpr int SMEM = STAGES * (BM * BK + BK * BN) * (int)sizeof(float);
  require_divisible("kernel 5", M, N, K, BM, BN, BK);
  auto kernel = tcgemmSwizzle<BM, BN, BK, WM, WN, STAGES, NUM_THREADS>;
  static bool configured = false;
  if (!configured) {
    set_max_dynamic_smem(kernel, SMEM);
    configured = true;
  }
  dim3 gridDim(N / BN, M / BM);
  kernel<<<gridDim, NUM_THREADS, SMEM>>>(M, N, K, alpha, A, B, beta, C);
}

void runSwizzle(int M, int N, int K, float alpha, float *A, float *B,
                float beta, float *C) {
  // 큰 행렬: 128x256 타일(144KB SMEM, SM당 block 1개)로 SMEM 재사용을 최대화한다.
  // 작은 행렬: block 수가 SM 108개보다 너무 적어지지 않게 128x128 타일을 쓴다.
  if ((long)M * N >= 2048L * 2048 && N % 256 == 0)
    launchSwizzle<128, 256, 32, 64, 64, 3>(M, N, K, alpha, A, B, beta, C);
  else
    launchSwizzle<128, 128, 32, 64, 32, 3>(M, N, K, alpha, A, B, beta, C);
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
static void launchVecFrag(int M, int N, int K, float alpha, float *A, float *B,
                          float beta, float *C) {
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARPSIZE;
  constexpr int SMEM =
      tc6SmemFloats<BM, BN, BK, STAGES>() * (int)sizeof(float);
  require_divisible("kernel 6", M, N, K, BM, BN, BK);
  // TC6_ROUND=0: TF32 반올림 없이 truncate (반올림 비용 측정용 실험 옵션)
  static const bool round = !(getenv("TC6_ROUND") && atoi(getenv("TC6_ROUND")) == 0);
  auto kernel = round ? tcgemmVecFrag<BM, BN, BK, WM, WN, STAGES, NUM_THREADS, true>
                      : tcgemmVecFrag<BM, BN, BK, WM, WN, STAGES, NUM_THREADS, false>;
  static bool configured = false;
  if (!configured) {
    set_max_dynamic_smem(kernel, SMEM);
    configured = true;
  }
  dim3 gridDim(N / BN, M / BM);
  kernel<<<gridDim, NUM_THREADS, SMEM>>>(M, N, K, alpha, A, B, beta, C);
}

// TC6_CFG 환경변수로 타일 구성을 강제할 수 있다 (튜닝용).
//   0: 128x256 3-stage (144KB, SM당 1 block)   1: 256x128 3-stage (144KB)
//   2: 128x128 3-stage (96KB, SM당 1 block)    3: 128x128 2-stage (66KB, SM당 2 block)
//   4: 128x128 4-stage (128KB)
void runVecFrag(int M, int N, int K, float alpha, float *A, float *B,
                float beta, float *C) {
  static const int forced = getenv("TC6_CFG") ? atoi(getenv("TC6_CFG")) : -1;
  int cfg = forced;
  if (cfg < 0)
    cfg = ((long)M * N >= 4096L * 4096 && N % 256 == 0) ? 0 : 3;
  switch (cfg) {
  case 0:
    launchVecFrag<128, 256, 32, 64, 64, 3>(M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    launchVecFrag<256, 128, 32, 64, 64, 3>(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    launchVecFrag<128, 128, 32, 64, 32, 3>(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    launchVecFrag<128, 128, 32, 64, 32, 2>(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    launchVecFrag<128, 128, 32, 64, 32, 4>(M, N, K, alpha, A, B, beta, C);
    break;
  }
}

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
static void launchFragPipe(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * WARPSIZE;
  constexpr int SMEM =
      tc6SmemFloats<BM, BN, BK, STAGES>() * (int)sizeof(float);
  require_divisible("kernel 7", M, N, K, BM, BN, BK);
  auto kernel = tcgemmFragPipe<BM, BN, BK, WM, WN, STAGES, NUM_THREADS>;
  static bool configured = false;
  if (!configured) {
    set_max_dynamic_smem(kernel, SMEM);
    configured = true;
  }
  // TC7_GROUP: rasterization 그룹 크기 (기본 8, 0이면 끔). A100 8192에서 +2%.
  static const int group = getenv("TC7_GROUP") ? atoi(getenv("TC7_GROUP")) : 8;
  dim3 gridDim(N / BN, M / BM);
  kernel<<<gridDim, NUM_THREADS, SMEM>>>(M, N, K, alpha, A, B, beta, C, group);
}

// TC7_CFG: 0: 128x256 3-stage  1: 256x128 3-stage  2: 128x128 3-stage  3: 128x128 4-stage
void runFragPipe(int M, int N, int K, float alpha, float *A, float *B,
                 float beta, float *C) {
  static const int forced = getenv("TC7_CFG") ? atoi(getenv("TC7_CFG")) : -1;
  int cfg = forced;
  if (cfg < 0)
    cfg = ((long)M * N >= 4096L * 4096 && M % 256 == 0) ? 1 : 2;
  switch (cfg) {
  case 0:
    launchFragPipe<128, 256, 32, 64, 64, 3>(M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    launchFragPipe<256, 128, 32, 64, 64, 3>(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    launchFragPipe<128, 128, 32, 64, 32, 3>(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    launchFragPipe<128, 128, 32, 64, 32, 4>(M, N, K, alpha, A, B, beta, C);
    break;
  }
}

bool is_valid_kernel(int kernel_num) {
  return (kernel_num >= 0 && kernel_num <= 7) || kernel_num == 100;
}

const char *kernel_name(int kernel_num) {
  switch (kernel_num) {
  case 0:
    return "cuBLAS FP32 (SIMT baseline)";
  case 1:
    return "WMMA naive";
  case 2:
    return "WMMA + SMEM warptiling";
  case 3:
    return "mma.sync m16n8k8 TF32";
  case 4:
    return "cp.async 3-stage pipeline";
  case 5:
    return "XOR swizzle + tuned tiles";
  case 6:
    return "vectorized fragments + fast TF32 round";
  case 7:
    return "fragment double buffering (warp-level pipeline)";
  case 100:
    return "cuBLAS TF32 (Tensor Core)";
  default:
    return "unknown";
  }
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublas(handle, CUBLAS_COMPUTE_32F, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    runWmmaNaive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    runWmmaSmem(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    runMmaSync(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runCpAsync(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSwizzle(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runVecFrag(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runFragPipe(M, N, K, alpha, A, B, beta, C);
    break;
  case 100:
    runCublas(handle, CUBLAS_COMPUTE_32F_FAST_TF32, M, N, K, alpha, A, B, beta,
              C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
