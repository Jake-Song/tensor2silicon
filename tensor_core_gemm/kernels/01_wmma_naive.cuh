#pragma once

#include "common.cuh"
#include <mma.h>

using namespace nvcuda;

// Kernel 1: WMMA API 입문.
// warp 하나가 C의 16x16 타일 하나를 계산하고, A/B fragment를 global memory에서
// 바로 읽는다. 데이터 재사용이 없으므로 메모리 대역폭에 묶이지만,
// Tensor Core 명령 한 번이 16x16x8 MAC을 처리한다는 점을 확인하는 단계다.
template <const int WARPS_M, const int WARPS_N>
__global__ void __launch_bounds__(WARPS_M *WARPS_N *WARPSIZE)
    tcgemmWmmaNaive(int M, int N, int K, float alpha, const float *A,
                    const float *B, float beta, float *C) {
  const int warpIdx = threadIdx.x / WARPSIZE;
  const int tileRow = (blockIdx.y * WARPS_M + warpIdx / WARPS_N) * 16;
  const int tileCol = (blockIdx.x * WARPS_N + warpIdx % WARPS_N) * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                 wmma::row_major>
      aFrag;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                 wmma::row_major>
      bFrag;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc, cFrag;
  wmma::fill_fragment(acc, 0.0f);

  for (int k = 0; k < K; k += 8) {
    wmma::load_matrix_sync(aFrag, A + tileRow * K + k, K);
    wmma::load_matrix_sync(bFrag, B + k * N + tileCol, N);
    // fragment에 들어온 FP32 값을 TF32로 반올림해야 한다.
    for (int i = 0; i < aFrag.num_elements; ++i)
      aFrag.x[i] = wmma::__float_to_tf32(aFrag.x[i]);
    for (int i = 0; i < bFrag.num_elements; ++i)
      bFrag.x[i] = wmma::__float_to_tf32(bFrag.x[i]);
    wmma::mma_sync(acc, aFrag, bFrag, acc);
  }

  // acc와 cFrag는 같은 accumulator 타입이므로 원소별 연산이 가능하다.
  float *cTile = C + tileRow * N + tileCol;
  wmma::load_matrix_sync(cFrag, cTile, N, wmma::mem_row_major);
  for (int i = 0; i < cFrag.num_elements; ++i)
    cFrag.x[i] = alpha * acc.x[i] + beta * cFrag.x[i];
  wmma::store_matrix_sync(cTile, cFrag, N, wmma::mem_row_major);
}
