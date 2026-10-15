#pragma once

#include "common.cuh"
#include <mma.h>

using namespace nvcuda;

// Kernel 2: SMEM block tiling + warp tiling (SGEMM_CUDA kernel 10의 구조).
// block이 BM x BN 타일을, 각 warp가 WM x WN 타일(16x16 fragment 여러 개)을 맡는다.
// global -> SMEM은 float4로 읽고, 저장할 때 TF32로 미리 반올림한다.
// 행 끝에 4 float padding을 둬 WMMA load의 bank conflict를 줄인다.
// (WMMA는 포인터 32바이트 정렬과 ldm이 4의 배수일 것을 요구한다.)
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmWmmaSmem(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C) {
  constexpr int LDA = BK + 4;
  constexpr int LDB = BN + 4;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 16;
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(BK % 8 == 0);

  __shared__ __align__(128) float As[BM * LDA];
  __shared__ __align__(128) float Bs[BK * LDB];

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                 wmma::row_major>
      aFrag[MFRAG];
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                 wmma::row_major>
      bFrag[NFRAG];
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[MFRAG][NFRAG];
  for (int m = 0; m < MFRAG; ++m)
    for (int n = 0; n < NFRAG; ++n)
      wmma::fill_fragment(acc[m][n], 0.0f);

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int i = threadIdx.x; i < BM * BK / 4; i += NUM_THREADS) {
      const int r = i / (BK / 4);
      const int c = (i % (BK / 4)) * 4;
      float4 v = reinterpret_cast<const float4 *>(&A[r * K + c])[0];
      v.x = wmma::__float_to_tf32(v.x);
      v.y = wmma::__float_to_tf32(v.y);
      v.z = wmma::__float_to_tf32(v.z);
      v.w = wmma::__float_to_tf32(v.w);
      reinterpret_cast<float4 *>(&As[r * LDA + c])[0] = v;
    }
    for (int i = threadIdx.x; i < BK * BN / 4; i += NUM_THREADS) {
      const int r = i / (BN / 4);
      const int c = (i % (BN / 4)) * 4;
      float4 v = reinterpret_cast<const float4 *>(&B[r * N + c])[0];
      v.x = wmma::__float_to_tf32(v.x);
      v.y = wmma::__float_to_tf32(v.y);
      v.z = wmma::__float_to_tf32(v.z);
      v.w = wmma::__float_to_tf32(v.w);
      reinterpret_cast<float4 *>(&Bs[r * LDB + c])[0] = v;
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk += 8) {
      // SMEM 값은 이미 TF32로 반올림되어 있으므로 load 후 변환을 생략한다.
      for (int m = 0; m < MFRAG; ++m)
        wmma::load_matrix_sync(aFrag[m],
                               &As[(warpRow * WM + m * 16) * LDA + kk], LDA);
      for (int n = 0; n < NFRAG; ++n)
        wmma::load_matrix_sync(bFrag[n],
                               &Bs[kk * LDB + warpCol * WN + n * 16], LDB);
      for (int m = 0; m < MFRAG; ++m)
        for (int n = 0; n < NFRAG; ++n)
          wmma::mma_sync(acc[m][n], aFrag[m], bFrag[n], acc[m][n]);
    }

    A += BK;
    B += BK * N;
    __syncthreads();
  }

  wmma::fragment<wmma::accumulator, 16, 16, 8, float> cFrag;
  for (int m = 0; m < MFRAG; ++m) {
    for (int n = 0; n < NFRAG; ++n) {
      float *cTile = C + (warpRow * WM + m * 16) * N + warpCol * WN + n * 16;
      wmma::load_matrix_sync(cFrag, cTile, N, wmma::mem_row_major);
      for (int i = 0; i < cFrag.num_elements; ++i)
        cFrag.x[i] = alpha * acc[m][n].x[i] + beta * cFrag.x[i];
      wmma::store_matrix_sync(cTile, cFrag, N, wmma::mem_row_major);
    }
  }
}
