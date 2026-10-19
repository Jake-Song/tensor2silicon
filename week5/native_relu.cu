// BF16 inputs, FP32 accumulation/output; no intermediate BF16 rounding.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;

template<int WARPS>
__global__ void gemv(const __nv_bfloat16* x, const __nv_bfloat16* w,
                     const float* b, float* y) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int n = blockIdx.x * 32 + lane;
  float acc = 0.f;
  for (int k = warp; k < 1024; k += WARPS)
    acc = fmaf(__bfloat162float(x[k]), __bfloat162float(w[k*1024+n]), acc);
  __shared__ float partial[WARPS][32];
  partial[warp][lane] = acc;
  __syncthreads();
  if (warp == 0) {
    float sum = 0.f;
    for (int r = 0; r < WARPS; ++r) sum += partial[r][lane];
    y[n] = fmaxf(sum + b[n], 0.f);
  }
}

template<int WARPS>
__global__ void tensorcore(const __nv_bfloat16* x, const __nv_bfloat16* w,
                           const float* b, float* y) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row = blockIdx.y * 16, col = (blockIdx.x * WARPS + warp) * 16;
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> bb;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc;
  wmma::fill_fragment(acc, 0.f);
  for (int k = 0; k < 1024; k += 16) {
    wmma::load_matrix_sync(a, x + row*1024 + k, 1024);
    wmma::load_matrix_sync(bb, w + k*1024 + col, 1024);
    wmma::mma_sync(acc, a, bb, acc);
  }
  __shared__ __align__(32) float tile[WARPS][256];
  wmma::store_matrix_sync(tile[warp], acc, 16, wmma::mem_row_major);
  __syncwarp();
  for (int i = lane; i < 256; i += 32) {
    int r = i / 16, c = i % 16;
    y[(row+r)*1024+col+c] = fmaxf(tile[warp][i] + b[col+c], 0.f);
  }
}

// Explicit stream and reusable output allocation; each call performs fresh work.
extern "C" int launch(const void* x, const void* w, const void* b, void* y,
                       int m, int warps, void* stream) {
  auto xx = static_cast<const __nv_bfloat16*>(x);
  auto ww = static_cast<const __nv_bfloat16*>(w);
  auto bb = static_cast<const float*>(b);
  auto yy = static_cast<float*>(y);
  auto ss = static_cast<cudaStream_t>(stream);
  if (m == 1) {
    if (warps == 4) gemv<4><<<32,128,0,ss>>>(xx,ww,bb,yy);
    else if (warps == 8) gemv<8><<<32,256,0,ss>>>(xx,ww,bb,yy);
    else gemv<16><<<32,512,0,ss>>>(xx,ww,bb,yy);
  } else if (m == 128) {
    if (warps == 4) tensorcore<4><<<dim3(16,8),128,0,ss>>>(xx,ww,bb,yy);
    else tensorcore<8><<<dim3(8,8),256,0,ss>>>(xx,ww,bb,yy);
  } else return int(cudaErrorInvalidValue);
  return int(cudaGetLastError());
}
