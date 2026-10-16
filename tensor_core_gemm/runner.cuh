#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define cudaCheck(err) (cudaCheckImpl(err, __FILE__, __LINE__))

void cudaCheckImpl(cudaError_t error, const char *file, int line);
void printDeviceInfo();

void randomize_matrix(float *mat, long n);

struct VerifyResult {
  double maxAbsDiff;
  double maxAbsRef;
  double relErr; // maxAbsDiff / maxAbsRef
  bool ok;
};
VerifyResult verify_matrix(const float *ref, const float *out, long n,
                           double tol);

bool is_valid_kernel(int kernel_num);
const char *kernel_name(int kernel_num);
void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle);

// Explicit configuration lets the native tuner reuse one set of input buffers.
struct FragPipeConfig {
  int tile = -1; // automatic
  int order = -1; // automatic
  int copy = -1; // automatic; 0: burst, 1: distributed, 2: step pointers,
                 // 3: groups after MMA, 4: raked B + groups after MMA (order6/epi4/round0)
  int group = 8;
  int epilogue = -1; // automatic, or vectors per C load batch: 1, 4, 8
  int rounding = -1; // automatic, 0: fast bits, 1: guarded, 2: guarded deferred, 3: fast deferred
};
void configure_frag_pipe(FragPipeConfig config);
