#pragma once

#include "common.cuh"

// Kernel 3: WMMA 대신 PTX mma.sync.m16n8k8을 직접 호출한다.
// 각 lane이 fragment의 어느 원소를 갖는지 직접 계산해 SMEM에서 읽는다
// (레이아웃은 common.cuh 참고). 타일 구조는 kernel 2와 같다.
//
// Bank conflict를 피하는 padding:
//   A fragment 읽기 As[(r0+g)*LDA + kk+t]: LDA % 8 == 4(예: 20)이면 g*LDA+t가 32개 bank에 퍼진다.
//   B fragment 읽기 Bs[(kk+t)*LDB + c0+g]: LDB % 32 == 8이면 t*8+g가 32개 bank에 퍼진다.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmMmaSync(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  constexpr int LDA = BK + 4;
  constexpr int LDB = BN + 8;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 8;
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(BK % 8 == 0 && LDA % 8 == 4 && LDB % 32 == 8);

  __shared__ __align__(16) uint32_t As[BM * LDA];
  __shared__ __align__(16) uint32_t Bs[BK * LDB];

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);
  const int lane = threadIdx.x % WARPSIZE;
  const int g = lane / 4;
  const int t = lane % 4;

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  float acc[MFRAG][NFRAG][4] = {};
  uint32_t aFrag[MFRAG][4];
  uint32_t bFrag[NFRAG][2];

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int i = threadIdx.x; i < BM * BK / 4; i += NUM_THREADS) {
      const int r = i / (BK / 4);
      const int c = (i % (BK / 4)) * 4;
      const float4 v = reinterpret_cast<const float4 *>(&A[r * K + c])[0];
      reinterpret_cast<uint4 *>(&As[r * LDA + c])[0] =
          make_uint4(f32_to_tf32(v.x), f32_to_tf32(v.y), f32_to_tf32(v.z),
                     f32_to_tf32(v.w));
    }
    for (int i = threadIdx.x; i < BK * BN / 4; i += NUM_THREADS) {
      const int r = i / (BN / 4);
      const int c = (i % (BN / 4)) * 4;
      const float4 v = reinterpret_cast<const float4 *>(&B[r * N + c])[0];
      reinterpret_cast<uint4 *>(&Bs[r * LDB + c])[0] =
          make_uint4(f32_to_tf32(v.x), f32_to_tf32(v.y), f32_to_tf32(v.z),
                     f32_to_tf32(v.w));
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk += 8) {
      for (int m = 0; m < MFRAG; ++m) {
        const uint32_t *a = &As[(warpRow * WM + m * 16 + g) * LDA + kk + t];
        aFrag[m][0] = a[0];
        aFrag[m][1] = a[8 * LDA];
        aFrag[m][2] = a[4];
        aFrag[m][3] = a[8 * LDA + 4];
      }
      for (int n = 0; n < NFRAG; ++n) {
        const uint32_t *b = &Bs[(kk + t) * LDB + warpCol * WN + n * 8 + g];
        bFrag[n][0] = b[0];
        bFrag[n][1] = b[4 * LDB];
      }
      for (int m = 0; m < MFRAG; ++m)
        for (int n = 0; n < NFRAG; ++n)
          mma_m16n8k8_tf32(acc[m][n], aFrag[m], bFrag[n]);
    }

    A += BK;
    B += BK * N;
    __syncthreads();
  }

  // d0,d1은 (g, 2t..2t+1), d2,d3은 (g+8, 2t..2t+1): 연속된 2개를 float2로 쓴다.
  for (int m = 0; m < MFRAG; ++m) {
    for (int n = 0; n < NFRAG; ++n) {
      const int row = warpRow * WM + m * 16 + g;
      const int col = warpCol * WN + n * 8 + 2 * t;
      for (int half = 0; half < 2; ++half) {
        float2 *p = reinterpret_cast<float2 *>(&C[(row + half * 8) * N + col]);
        float2 c = p[0];
        c.x = alpha * acc[m][n][half * 2 + 0] + beta * c.x;
        c.y = alpha * acc[m][n][half * 2 + 1] + beta * c.y;
        p[0] = c;
      }
    }
  }
}
