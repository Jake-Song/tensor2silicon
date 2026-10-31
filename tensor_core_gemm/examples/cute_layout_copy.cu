// One warp doubles an 8x8 row-major tile using CuTe thread partitioning.
// nvcc -O3 -std=c++17 -arch=sm_80 -I/tmp/tc-cutlass/include \
//   cute_layout_copy.cu -o /tmp/cute_layout_copy
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

// Launch contract: exactly one block of 32 threads; A and B contain 64 floats.
__global__ void double_tile(const float *A, float *B) {
  using namespace cute;
  auto matrix_layout = make_layout(make_shape(Int<8>{}, Int<8>{}),
                                   make_stride(Int<8>{}, Int<1>{}));
  auto a = make_tensor(make_gmem_ptr(A), matrix_layout);
  auto b = make_tensor(make_gmem_ptr(B), matrix_layout);

  // Arrange 32 threads as 4 rows x 8 columns, with columns varying fastest.
  auto thread_layout = make_layout(make_shape(Int<4>{}, Int<8>{}),
                                   make_stride(Int<8>{}, Int<1>{}));
  auto ta = local_partition(a, thread_layout, int(threadIdx.x));
  auto tb = local_partition(b, thread_layout, int(threadIdx.x));
  // Each thread sees only its two assigned elements.
  for (int i = 0; i < size(ta); ++i)
    tb(i) = 2.0f * ta(i);
}

int main() {
  float input[64], output[64];
  for (int i = 0; i < 64; ++i) input[i] = float(i);
  float *a, *b;
  check(cudaMalloc(&a, sizeof(input)));
  check(cudaMalloc(&b, sizeof(output)));
  check(cudaMemcpy(a, input, sizeof(input), cudaMemcpyHostToDevice));
  double_tile<<<1, 32>>>(a, b);
  check(cudaGetLastError());
  check(cudaMemcpy(output, b, sizeof(output), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) {
    if (output[i] != 2.0f * input[i]) {
      std::fprintf(stderr, "FAIL index=%d expected=%g actual=%g\n",
                   i, 2.0f * input[i], output[i]);
      return 1;
    }
  }
  check(cudaFree(a));
  check(cudaFree(b));
  std::puts("PASS: all 64 elements doubled");
}
