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
