#pragma once

#include "common.cuh"

// Kernel 5: XOR swizzle + 큰 타일 + SMEM 경유 epilogue.
//
// 1) padding 대신 XOR swizzle을 쓴다. 16바이트(float4) chunk 단위로 열 위치를 섞어
//    cp.async 쓰기와 fragment 읽기가 모두 bank conflict 없이 동작하며 SMEM 낭비도 없다.
//    A 타일 [BM][32]: 한 행 = 8 chunk. 물리 chunk = chunk ^ (row % 8).
//      A fragment 읽기는 행 g=0..7이 서로 다른 chunk에 놓이므로 32개 bank를 모두 쓴다.
//    B 타일 [32][BN]: 물리 chunk = chunk ^ ((row % 4) * 2).
//      B fragment 읽기는 행 t=0..3, 열 g=0..7(chunk 2개)이 서로 다른 bank 쌍에 놓인다.
// 2) BK=32, warp tile 64x64(또는 64x32)로 SMEM load당 mma 수를 늘린다.
// 3) epilogue에서 accumulator를 SMEM에 모은 뒤 float4로 coalesced하게 C를 읽고 쓴다.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int STAGES, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmSwizzle(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = BK * BN;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 8;
  constexpr int LDC = BN + 8; // epilogue용 SMEM 행 길이
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(BK == 32, "A swizzle은 한 행이 정확히 8 chunk라고 가정한다");
  static_assert(BN % 32 == 0);
  static_assert(BM * LDC <= STAGES * (A_STAGE + B_STAGE),
                "epilogue 타일이 파이프라인 SMEM에 들어가야 한다");

  extern __shared__ __align__(128) float smem[];
  float *As = smem;                    // [STAGES][BM][BK] swizzled
  float *Bs = smem + STAGES * A_STAGE; // [STAGES][BK][BN] swizzled

  // 논리 좌표 (row, col) -> swizzle된 SMEM 오프셋
  auto aIdx = [](int row, int col) {
    return row * BK + ((((col >> 2) ^ (row & 7))) << 2) + (col & 3);
  };
  auto bIdx = [](int row, int col) {
    return row * BN + ((((col >> 2) ^ ((row & 3) << 1))) << 2) + (col & 3);
  };

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);
  const int lane = threadIdx.x % WARPSIZE;
  const int g = lane / 4;
  const int t = lane % 4;

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  auto loadTile = [&](int stage, int tileIdx) {
    const float *aSrc = A + tileIdx * BK;
    const float *bSrc = B + tileIdx * BK * N;
    float *aDst = As + stage * A_STAGE;
    float *bDst = Bs + stage * B_STAGE;
#pragma unroll
    for (int i = threadIdx.x; i < BM * BK / 4; i += NUM_THREADS) {
      const int r = i / (BK / 4);
      const int c = (i % (BK / 4)) * 4;
      cp_async_16(&aDst[aIdx(r, c)], &aSrc[r * K + c]);
    }
#pragma unroll
    for (int i = threadIdx.x; i < BK * BN / 4; i += NUM_THREADS) {
      const int r = i / (BN / 4);
      const int c = (i % (BN / 4)) * 4;
      cp_async_16(&bDst[bIdx(r, c)], &bSrc[r * N + c]);
    }
  };

  float acc[MFRAG][NFRAG][4] = {};
  uint32_t aFrag[MFRAG][4];
  uint32_t bFrag[NFRAG][2];

  const int numTiles = K / BK;

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < numTiles)
      loadTile(s, s);
    cp_async_commit();
  }

  for (int tile = 0; tile < numTiles; ++tile) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    const int next = tile + STAGES - 1;
    if (next < numTiles)
      loadTile(next % STAGES, next);
    cp_async_commit();

    const float *as = As + (tile % STAGES) * A_STAGE;
    const float *bs = Bs + (tile % STAGES) * B_STAGE;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
#pragma unroll
      for (int m = 0; m < MFRAG; ++m) {
        const int r = warpRow * WM + m * 16 + g;
        aFrag[m][0] = f32_to_tf32(as[aIdx(r, kk + t)]);
        aFrag[m][1] = f32_to_tf32(as[aIdx(r + 8, kk + t)]);
        aFrag[m][2] = f32_to_tf32(as[aIdx(r, kk + t + 4)]);
        aFrag[m][3] = f32_to_tf32(as[aIdx(r + 8, kk + t + 4)]);
      }
#pragma unroll
      for (int n = 0; n < NFRAG; ++n) {
        const int c = warpCol * WN + n * 8 + g;
        bFrag[n][0] = f32_to_tf32(bs[bIdx(kk + t, c)]);
        bFrag[n][1] = f32_to_tf32(bs[bIdx(kk + t + 4, c)]);
      }
#pragma unroll
      for (int m = 0; m < MFRAG; ++m)
#pragma unroll
        for (int n = 0; n < NFRAG; ++n)
          mma_m16n8k8_tf32(acc[m][n], aFrag[m], bFrag[n]);
    }
  }

  // 파이프라인 SMEM을 epilogue 버퍼로 재사용한다.
  cp_async_wait<0>();
  __syncthreads();
  float *Cs = smem; // [BM][LDC]
#pragma unroll
  for (int m = 0; m < MFRAG; ++m) {
#pragma unroll
    for (int n = 0; n < NFRAG; ++n) {
      const int row = warpRow * WM + m * 16 + g;
      const int col = warpCol * WN + n * 8 + 2 * t;
      reinterpret_cast<float2 *>(&Cs[row * LDC + col])[0] =
          make_float2(acc[m][n][0], acc[m][n][1]);
      reinterpret_cast<float2 *>(&Cs[(row + 8) * LDC + col])[0] =
          make_float2(acc[m][n][2], acc[m][n][3]);
    }
  }
  __syncthreads();

  // 연속된 thread가 C의 연속된 float4를 담당한다 (coalesced).
  for (int i = threadIdx.x; i < BM * BN / 4; i += NUM_THREADS) {
    const int r = i / (BN / 4);
    const int c = (i % (BN / 4)) * 4;
    const float4 ab = reinterpret_cast<const float4 *>(&Cs[r * LDC + c])[0];
    float4 *p = reinterpret_cast<float4 *>(&C[r * N + c]);
    float4 v = p[0];
    v.x = alpha * ab.x + beta * v.x;
    v.y = alpha * ab.y + beta * v.y;
    v.z = alpha * ab.z + beta * v.z;
    v.w = alpha * ab.w + beta * v.w;
    p[0] = v;
  }
}
