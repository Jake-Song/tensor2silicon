#pragma once

#include "common.cuh"

// Kernel 4: cp.async 다단계 파이프라인 (SGEMM_CUDA kernel 11/12의 Ampere 버전).
// global -> SMEM 복사가 레지스터를 거치지 않으므로 다음 타일의 복사와 현재 타일의
// mma가 겹친다. STAGES개의 SMEM 버퍼를 ring buffer로 쓴다.
// cp.async는 값을 그대로 복사하므로 TF32 변환을 fragment load 시점으로 옮긴다.
// SMEM이 48KB를 넘으므로 dynamic shared memory를 사용한다.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int STAGES, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmCpAsync(int M, int N, int K, float alpha, const float *A,
                  const float *B, float beta, float *C) {
  constexpr int LDA = BK + 4;
  constexpr int LDB = BN + 8;
  constexpr int A_STAGE = BM * LDA;
  constexpr int B_STAGE = BK * LDB;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 8;
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(BK % 8 == 0 && LDA % 8 == 4 && LDB % 32 == 8);
  static_assert(STAGES >= 2);

  extern __shared__ __align__(128) float smem[];
  float *As = smem;                    // [STAGES][BM][LDA]
  float *Bs = smem + STAGES * A_STAGE; // [STAGES][BK][LDB]

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);
  const int lane = threadIdx.x % WARPSIZE;
  const int g = lane / 4;
  const int t = lane % 4;

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // k 타일 tileIdx를 stage 버퍼에 비동기로 복사한다.
  auto loadTile = [&](int stage, int tileIdx) {
    const float *aSrc = A + tileIdx * BK;
    const float *bSrc = B + tileIdx * BK * N;
    float *aDst = As + stage * A_STAGE;
    float *bDst = Bs + stage * B_STAGE;
    for (int i = threadIdx.x; i < BM * BK / 4; i += NUM_THREADS) {
      const int r = i / (BK / 4);
      const int c = (i % (BK / 4)) * 4;
      cp_async_16(&aDst[r * LDA + c], &aSrc[r * K + c]);
    }
    for (int i = threadIdx.x; i < BK * BN / 4; i += NUM_THREADS) {
      const int r = i / (BN / 4);
      const int c = (i % (BN / 4)) * 4;
      cp_async_16(&bDst[r * LDB + c], &bSrc[r * N + c]);
    }
  };

  float acc[MFRAG][NFRAG][4] = {};
  uint32_t aFrag[MFRAG][4];
  uint32_t bFrag[NFRAG][2];

  const int numTiles = K / BK;

  // prologue: STAGES-1개 타일을 미리 요청한다.
  // 타일이 없어도 commit해 그룹 수를 일정하게 유지한다 (wait_group 계산 단순화).
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < numTiles)
      loadTile(s, s);
    cp_async_commit();
  }

  for (int tile = 0; tile < numTiles; ++tile) {
    // tile 번째 그룹이 도착할 때까지 기다린다 (이후 STAGES-2개는 진행 중이어도 됨).
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    // 위 __syncthreads() 덕분에 모든 warp가 tile-1 버퍼 사용을 끝냈으므로 덮어써도 안전하다.
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
        const float *a = &as[(warpRow * WM + m * 16 + g) * LDA + kk + t];
        aFrag[m][0] = f32_to_tf32(a[0]);
        aFrag[m][1] = f32_to_tf32(a[8 * LDA]);
        aFrag[m][2] = f32_to_tf32(a[4]);
        aFrag[m][3] = f32_to_tf32(a[8 * LDA + 4]);
      }
#pragma unroll
      for (int n = 0; n < NFRAG; ++n) {
        const float *b = &bs[(kk + t) * LDB + warpCol * WN + n * 8 + g];
        bFrag[n][0] = f32_to_tf32(b[0]);
        bFrag[n][1] = f32_to_tf32(b[4 * LDB]);
      }
#pragma unroll
      for (int m = 0; m < MFRAG; ++m)
#pragma unroll
        for (int n = 0; n < NFRAG; ++n)
          mma_m16n8k8_tf32(acc[m][n], aFrag[m], bFrag[n]);
    }
  }
  cp_async_wait<0>();

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
