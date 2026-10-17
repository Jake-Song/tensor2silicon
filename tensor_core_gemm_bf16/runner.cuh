#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdio>

inline void ck(cudaError_t x) { if(x != cudaSuccess) throw std::runtime_error(cudaGetErrorString(x)); }
inline void cb(cublasStatus_t x) { if(x != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error " + std::to_string(int(x))); }
struct Problem {
  int m,n,k;
  const __nv_bfloat16 *a,*b;
  float *d;
};
using Launch = void (*)(const Problem&, cudaStream_t, int);
struct Variant {
  std::string name; int stage,bm,bn,bk,smem,threads;
  Launch launch;
};
const std::vector<Variant>& variants();
void run(const Variant&, const Problem&, cudaStream_t = nullptr, int group = 0);
void reference(cublasHandle_t, const Problem&, const float *af, const float *bf);
class Library {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t al{},bl{},cl{},dl{};
  cublasLtMatmulAlgo_t algorithm{};
  void *workspace{};
  size_t workspace_size = 32UL << 20;
public:
  explicit Library(const Problem&);
  ~Library();
  Library(const Library&) = delete;
  Library& operator=(const Library&) = delete;
  void launch(const Problem&, cudaStream_t = nullptr) const;
};
