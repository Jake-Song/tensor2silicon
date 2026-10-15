#pragma once

#include "common.cuh"

// Kernel 6: fragment load 벡터화 + 싼 TF32 반올림.
//
// ncu로 kernel 5를 보면 Tensor 파이프는 절반만 쓰이고, 실행 명령 중 HMMA는 15%뿐이다.
// 나머지는 원소마다 하는 스칼라 LDS와 cvt.rna.tf32(sm_80에서 정수 명령 3~4개로 풀림)다.
//
// 1) 반올림: mma.tf32는 피연산자의 하위 13비트를 무시하므로 비트에 0x1000(반 ulp)만 더하면
//    round-half-away-from-zero가 된다. IADD 1개. (입력이 inf면 NaN이 되는 차이만 있다.)
// 2) k 순서 재배치: mma는 k에 대한 합이므로 A와 B에 같은 순열을 쓰면 결과가 같다.
//    16개 k 묶음에서 lane t의 step s(0/1)가 쓰는 k를 (4t+2s, 4t+2s+1)로 두면
//    A의 한 행에서 lane t가 필요한 값이 연속 4개(4t..4t+3)가 되어 LDS.128 하나로 읽힌다.
// 3) n 순서 재배치: fragment n의 열 j를 실제 열 j*NFRAG + n에 대응시키면
//    한 k 행에서 lane g가 필요한 B 값이 연속 NFRAG개(g*NFRAG..)가 되어 LDS.128로 읽힌다.
//    epilogue에서 같은 대응으로 되돌린다.
// 4) swizzle은 위 LDS.128 접근(quarter-warp 8 lane)이 서로 다른 bank를 쓰도록 다시 정했다.
//    A [BM][32]: 물리 chunk = chunk ^ ((row & 1) << 2)
//    B [32][BN]: 물리 chunk = chunk ^ f(t),  t = (row >> 2) & 3
//                NFRAG=8: f = (t & 1) | ((t & 2) << 1),  NFRAG=4: f = t << 1

template <bool ROUND> __device__ __forceinline__ uint32_t tf32_round_fast(float x) {
  // ROUND=false는 하위 비트를 그대로 넘겨 mma가 버리게 한다 (round toward zero, 비교 실험용).
  return ROUND ? __float_as_uint(x) + 0x1000u : __float_as_uint(x);
}

template <const int BM, const int BN, const int BK, const int STAGES>
constexpr int tc6SmemFloats() {
  constexpr int pipe = STAGES * (BM * BK + BK * BN);
  constexpr int epi = BM * (BN + 4);
  return pipe > epi ? pipe : epi;
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int STAGES, const int NUM_THREADS, const bool ROUND = true>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmVecFrag(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = BK * BN;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 8;
  constexpr int LDC = BN + 4;
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(BK == 32, "A 한 행 = 8 chunk, k 묶음 16개 x 2");
  static_assert(NFRAG == 4 || NFRAG == 8, "B swizzle은 NFRAG 4 또는 8을 가정한다");

  extern __shared__ __align__(128) float smem[];
  float *As = smem;
  float *Bs = smem + STAGES * A_STAGE;

  // B swizzle: t = (row >> 2) & 3에 따라 chunk 하위 3비트를 섞는다.
  auto swzB = [](int t) {
    return NFRAG == 8 ? ((t & 1) | ((t & 2) << 1)) : (t << 1);
  };

  const int tid = threadIdx.x;
  const int warpIdx = tid / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);
  const int lane = tid % WARPSIZE;
  const int g = lane / 4;
  const int t = lane % 4;

  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // ---- 루프 밖에서 한 번만 계산하는 주소 ----
  // 레지스터가 꽉 찬 커널에서 swizzle 주소를 매 타일 다시 계산하지 않도록
  // 모든 SMEM 주소를 "스레드별 기준값 + 컴파일 시간 상수"로 만든다.
  //
  // cp.async: A는 반복마다 A_ROWS행씩(짝수) 내려가므로 swizzle(행 홀짝)이 변하지 않는다.
  constexpr int A_CH = BK / 4, A_ROWS = NUM_THREADS / A_CH, A_ITERS = BM / A_ROWS;
  constexpr int B_CH = BN / 4, B_ROWS = NUM_THREADS / B_CH, B_ITERS = BK / B_ROWS;
  static_assert(A_ROWS % 2 == 0 && B_ROWS % 4 == 0);
  const int aRow0 = tid / A_CH, aCh = tid % A_CH;
  const int bRow0 = tid / B_CH, bCh = tid % B_CH;
  const int aDst0 = aRow0 * BK + ((aCh ^ ((aRow0 & 1) << 2)) << 2);
  const float *aG = A + (blockIdx.y * BM + aRow0) * K + aCh * 4;
  const float *bG = B + bRow0 * N + blockIdx.x * BN + bCh * 4;

  // fragment load: A는 k 묶음(kk=0/16)별 2개, B는 q(NFRAG/4)별 기준 주소.
  // 나머지(m, h, kk, s, r2)는 모두 상수 오프셋이다.
  int aFragOff[2], bFragOff[NFRAG / 4];
#pragma unroll
  for (int j = 0; j < 2; ++j)
    aFragOff[j] = (warpRow * WM + g) * BK + (((4 * j + t) ^ ((g & 1) << 2)) << 2);
#pragma unroll
  for (int q = 0; q < NFRAG / 4; ++q)
    bFragOff[q] = 4 * t * BN +
                  ((((warpCol * WN + g * NFRAG) / 4 + q) ^ swzB(t)) << 2);

  auto loadTile = [&](int stage) {
    float *aD = As + stage * A_STAGE + aDst0;
#pragma unroll
    for (int j = 0; j < A_ITERS; ++j)
      cp_async_16(aD + j * A_ROWS * BK, aG + (long)j * A_ROWS * K);
    float *bD = Bs + stage * B_STAGE;
#pragma unroll
    for (int j = 0; j < B_ITERS; ++j) {
      const int row = bRow0 + j * B_ROWS;
      cp_async_16(bD + row * BN + ((bCh ^ swzB((row >> 2) & 3)) << 2),
                  bG + (long)j * B_ROWS * N);
    }
    aG += BK;
    bG += BK * N;
  };

  float acc[MFRAG][NFRAG][4] = {};

  const int numTiles = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < numTiles)
      loadTile(s);
    cp_async_commit();
  }

  int readStage = 0, writeStage = STAGES - 1;
  for (int tile = 0; tile < numTiles; ++tile) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    if (tile + STAGES - 1 < numTiles)
      loadTile(writeStage);
    cp_async_commit();
    writeStage = writeStage == STAGES - 1 ? 0 : writeStage + 1;

    const float *as = As + readStage * A_STAGE;
    const float *bs = Bs + readStage * B_STAGE;
    readStage = readStage == STAGES - 1 ? 0 : readStage + 1;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      // A: 행 g, g+8에서 k = kk+4t..kk+4t+3 (두 step 분량)을 LDS.128로
      uint32_t aReg[MFRAG][2][4];
#pragma unroll
      for (int m = 0; m < MFRAG; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float4 v = reinterpret_cast<const float4 *>(
              as + aFragOff[kk / 16] + (m * 16 + h * 8) * BK)[0];
          aReg[m][h][0] = tf32_round_fast<ROUND>(v.x);
          aReg[m][h][1] = tf32_round_fast<ROUND>(v.y);
          aReg[m][h][2] = tf32_round_fast<ROUND>(v.z);
          aReg[m][h][3] = tf32_round_fast<ROUND>(v.w);
        }
      }
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        // B: k 행 kk+4t+2s(+1), 열 warpCol*WN + g*NFRAG .. +NFRAG-1
        uint32_t bFrag[NFRAG][2];
#pragma unroll
        for (int r2 = 0; r2 < 2; ++r2) {
#pragma unroll
          for (int q = 0; q < NFRAG / 4; ++q) {
            const float4 v = reinterpret_cast<const float4 *>(
                bs + bFragOff[q] + (kk + 2 * s + r2) * BN)[0];
            bFrag[4 * q + 0][r2] = tf32_round_fast<ROUND>(v.x);
            bFrag[4 * q + 1][r2] = tf32_round_fast<ROUND>(v.y);
            bFrag[4 * q + 2][r2] = tf32_round_fast<ROUND>(v.z);
            bFrag[4 * q + 3][r2] = tf32_round_fast<ROUND>(v.w);
          }
        }
#pragma unroll
        for (int m = 0; m < MFRAG; ++m) {
          // a0=(g, k0) a1=(g+8, k0) a2=(g, k1) a3=(g+8, k1),  k0=4t+2s, k1=k0+1
          const uint32_t aFrag[4] = {aReg[m][0][2 * s], aReg[m][1][2 * s],
                                     aReg[m][0][2 * s + 1],
                                     aReg[m][1][2 * s + 1]};
#pragma unroll
          for (int n = 0; n < NFRAG; ++n)
            mma_m16n8k8_tf32(acc[m][n], aFrag, bFrag[n]);
        }
      }
    }
  }

  cp_async_wait<0>();
  __syncthreads();

  // fragment n의 열 j -> 실제 열 j*NFRAG + n.
  // lane의 d0(열 2t)는 n=0..NFRAG-1에 걸쳐 연속이므로 float4로 SMEM에 쓴다.
  float *Cs = smem; // [BM][LDC]
#pragma unroll
  for (int m = 0; m < MFRAG; ++m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = warpRow * WM + m * 16 + g + h * 8;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float *dst = &Cs[row * LDC + warpCol * WN + (2 * t + e) * NFRAG];
#pragma unroll
        for (int q = 0; q < NFRAG / 4; ++q)
          reinterpret_cast<float4 *>(dst)[q] =
              make_float4(acc[m][4 * q + 0][2 * h + e],
                          acc[m][4 * q + 1][2 * h + e],
                          acc[m][4 * q + 2][2 * h + e],
                          acc[m][4 * q + 3][2 * h + e]);
      }
    }
  }
  __syncthreads();

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
