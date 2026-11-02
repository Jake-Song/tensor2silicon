#pragma once

#include "common.cuh"
#include "06_vectorized_frag.cuh"

// 단계 10: 9번 + natural-N/serpentine MMA + 같은 K행에서 N방향으로 B 복사. 복사는 MMA 뒤에 발행.
// 비교를 위해 7~10은 같은 shape에서 같은 타일 정책을 사용한다.
// 설정 조합 전체를 지원하는 기존 구현은 detail/frag_pipe_tuning.cuh에 보존한다.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int STAGES, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmRakedBCopy(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C, int group) {
  constexpr bool RAKE_B = true;
  constexpr bool CU_COPY = true;
  constexpr bool LDM_A = true;
  constexpr bool NATURAL_N = true;
  constexpr int EPILOGUE_VECTORS = 4;
  static_assert(BM * BN % (4 * NUM_THREADS * EPILOGUE_VECTORS) == 0);
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = BK * BN;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 8;
  constexpr int LDC = BN + (NATURAL_N ? 8 : 4);
  constexpr int KSTEPS = BK / 8;
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(BK == 32, "A 한 행 = 8 chunk");
  static_assert(NFRAG == 4 || NFRAG == 8);
  static_assert(STAGES >= 3, "다음 타일 로드와 현재 타일 읽기가 겹치므로 3 stage 이상");

  extern __shared__ __align__(128) float smem[];
  float *As = smem;
  float *Bs = smem + STAGES * A_STAGE;

  auto swzB = [](int t) {
    return !NATURAL_N && NFRAG == 8 ? ((t & 1) | ((t & 2) << 1)) : (t << 1);
  };

  const int tid = threadIdx.x;
  const int warpIdx = tid / WARPSIZE;
  const int warpRow = warpIdx / (BN / WN);
  const int warpCol = warpIdx % (BN / WN);
  const int lane = tid % WARPSIZE;
  const int g = lane / 4;
  const int t = lane % 4;

  int bx = blockIdx.x, by = blockIdx.y;
  if (group > 0) {
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int perGroup = group * gridDim.x;
    const int first = (lin / perGroup) * group;
    const int rows = min((int)gridDim.y - first, group);
    by = first + (lin % perGroup) % rows;
    bx = (lin % perGroup) / rows;
  }
  C += by * BM * N + bx * BN;

  // ---- cp.async 주소 (kernel 6과 같은 방식, swizzle만 다름) ----
  constexpr int A_CH = BK / 4, A_ROWS = NUM_THREADS / A_CH, A_ITERS = BM / A_ROWS;
  // Raked B copies keep a thread on one K row and advance across adjacent
  // N chunks. Global addresses then share one base plus constant offsets.
  constexpr int B_CH = RAKE_B ? NUM_THREADS / BK : BN / 4;
  constexpr int B_ROWS = RAKE_B ? BK : NUM_THREADS / B_CH;
  constexpr int B_ITERS = RAKE_B ? BN / (4 * B_CH) : BK / B_ROWS;
  static_assert(!RAKE_B || (NUM_THREADS % BK == 0 && BN % (4 * B_CH) == 0));
  static_assert(A_ROWS % 4 == 0 && B_ROWS % 2 == 0);
  static_assert(!LDM_A || (A_ROWS % 8 == 0 && B_ROWS % 4 == 0));
  const int aRow0 = tid / A_CH, aCh = tid % A_CH;
  const int bRow0 = tid / B_CH, bCh = tid % B_CH;
  const int aDst0 = aRow0 * BK + ((aCh ^ (LDM_A ? (aRow0 & 7) : ((aRow0 & 3) << 1))) << 2);
  const float *aG = A + (by * BM + aRow0) * K + aCh * 4;
  const float *bG = B + bRow0 * N + bx * BN + bCh * 4;

  constexpr int A_GROUP = (A_ITERS + KSTEPS - 1) / KSTEPS;
  constexpr int B_GROUP = (B_ITERS + KSTEPS - 1) / KSTEPS;

  auto loadTile = [&](int stage, int part = -1) {
    float *aD = As + stage * A_STAGE + aDst0;
#pragma unroll
    for (int j = 0; j < A_ITERS; ++j)
      if (part < 0 || (CU_COPY ? j / A_GROUP : j % KSTEPS) == part)
        cp_async_16(aD + j * A_ROWS * BK, aG + (long)j * A_ROWS * K);
    float *bD = Bs + stage * B_STAGE;
#pragma unroll
    for (int j = 0; j < B_ITERS; ++j) {
      const int row = bRow0 + (RAKE_B ? 0 : j * B_ROWS);
      const int chunk = bCh + (RAKE_B ? j * B_CH : 0);
      if (part < 0 || (CU_COPY ? j / B_GROUP : j % KSTEPS) == part)
        cp_async_16(bD + row * BN + ((chunk ^ swzB(LDM_A ? (row & 3) : ((row >> 1) & 3))) << 2),
                    bG + (RAKE_B ? (long)j * B_CH * 4 :
                      (long)j * B_ROWS * N));
    }
    if (part < 0 || part == KSTEPS - 1) {
      aG += BK;
      bG += (long)BK * N;
    }
  };

  // ---- fragment 주소 ----
  // B: 행 = 8*step + 2t + r2 이므로 swizzle t = (row>>1)&3 = t로 고정.
  int bNaturalOff[NFRAG];
#pragma unroll
  for (int n = 0; n < NFRAG; ++n)
    bNaturalOff[n] = t * BN + ((warpCol * WN + 8 * n + g) ^ (t << 3));

  uint32_t aFrag[2][MFRAG][4];
  uint32_t bFrag[2][NFRAG][2];

  auto loadFrag = [&](int buf, const float *as, const float *bs, int step) {

      // Four 8x8 matrices of raw 16-bit halves hold one 16x8 FP32 tile.
      // Lane groups 0..7/8..15/16..23/24..31 supply its four row-address sets.
      // The returned words match TF32 a0=(g,t), a1=(g+8,t),
      // a2=(g,t+4), a3=(g+8,t+4); B uses the same natural K order.
      const int row = warpRow * WM + (lane & 15);
      const int chunk = ((2 * step + (lane >> 4)) ^ (lane & 7));
      const float *aBase = as + row * BK + chunk * 4;
#pragma unroll
      for (int m = 0; m < MFRAG; ++m) {
        const uint32_t addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(aBase + m * 16 * BK));
        uint32_t r0, r1, r2, r3;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                     "{%0, %1, %2, %3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
        aFrag[buf][m][0] = tf32_round_fast<true>(__uint_as_float(r0));
        aFrag[buf][m][1] = tf32_round_fast<true>(__uint_as_float(r1));
        aFrag[buf][m][2] = tf32_round_fast<true>(__uint_as_float(r2));
        aFrag[buf][m][3] = tf32_round_fast<true>(__uint_as_float(r3));
      }

#pragma unroll
      for (int n = 0; n < NFRAG; ++n) {
#pragma unroll
        for (int r2 = 0; r2 < 2; ++r2) {
          const float v = bs[bNaturalOff[n] + (8 * step + 4 * r2) * BN];
          bFrag[buf][n][r2] = tf32_round_fast<true>(v);
        }
      }

  };

  float acc[MFRAG][NFRAG][4] = {};

  const int numTiles = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < numTiles)
      loadTile(s);
    cp_async_commit();
  }
  cp_async_wait<STAGES - 2>();
  __syncthreads();

  int readStage = 0, writeStage = STAGES - 1;
  const float *as = As;
  const float *bs = Bs;
  loadFrag(0, as, bs, 0);

  for (int tile = 0; tile < numTiles; ++tile) {
#pragma unroll
    for (int step = 0; step < KSTEPS; ++step) {
      // 복사 그룹은 아래 MMA 뒤에서 분산 발행한다.

      // 다음 step(또는 다음 타일 step 0)의 fragment를 현재 mma보다 먼저 요청한다.
      if (step < KSTEPS - 1) {
        loadFrag((step + 1) & 1, as, bs, step + 1);
      } else if (tile + 1 < numTiles) {
        as = As + readStage * A_STAGE;
        bs = Bs + readStage * B_STAGE;
        loadFrag(0, as, bs, 0);
      }
#pragma unroll
      for (int m = 0; m < MFRAG; ++m) {
#pragma unroll
        for (int inner = 0; inner < NFRAG; ++inner) {
          // 행 경계에서도 같은 B fragment를 재사용하도록 열을 왕복한다.
          const int n = (m & 1) ? NFRAG - 1 - inner : inner;
          mma_m16n8k8_tf32(acc[m][n], aFrag[step & 1][m], bFrag[step & 1][n]);
        }
      }

        if (step < KSTEPS - 1) {
          if (tile + STAGES - 1 < numTiles) loadTile(writeStage, step);
          if (step == KSTEPS - 2) {
            if (tile + STAGES - 1 < numTiles) loadTile(writeStage, KSTEPS - 1);
            cp_async_commit();
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            writeStage = writeStage == STAGES - 1 ? 0 : writeStage + 1;
            readStage = readStage == STAGES - 1 ? 0 : readStage + 1;
          }
        }

    }
  }

  // Load actual C values before the SMEM epilogue barriers. Subsequent
  // batches overlap global loads with current shared loads and stores.
  float4 cValues[2][EPILOGUE_VECTORS];
  auto cVector = [&](int index) {
    const int row = index / (BN / 4), col = (index % (BN / 4)) * 4;
    return reinterpret_cast<float4 *>(C + (long)row * N + col);
  };
  auto loadC = [&](int buf, int batch) {
#pragma unroll
    for (int j = 0; j < EPILOGUE_VECTORS; ++j)
      cValues[buf][j] = cVector(tid + (batch * EPILOGUE_VECTORS + j) * NUM_THREADS)[0];
  };

    // Keep these loads after the MMA loop in compiler IR; holding them
    // across the main loop would increase its register requirement.
    asm volatile("" ::: "memory");
    loadC(0, 0);

  cp_async_wait<0>();
  __syncthreads();

  // natural-N fragment를 원래 출력 열 순서로 shared memory에 배치한다.
  float *Cs = smem;

#pragma unroll
    for (int m = 0; m < MFRAG; ++m)
#pragma unroll
      for (int n = 0; n < NFRAG; ++n)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = warpRow * WM + 16 * m + g + 8 * h;
          const int col = warpCol * WN + 8 * n + 2 * t;
          reinterpret_cast<float2 *>(Cs + row * LDC + col)[0] =
              make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
        }

  __syncthreads();

    constexpr int BATCHES = BM * BN / (4 * NUM_THREADS * EPILOGUE_VECTORS);
    auto storeC = [&](int buf, int batch) {
#pragma unroll
      for (int j = 0; j < EPILOGUE_VECTORS; ++j) {
        const int i = tid + (batch * EPILOGUE_VECTORS + j) * NUM_THREADS;
        const int r = i / (BN / 4), c = (i % (BN / 4)) * 4;
        const float4 ab = reinterpret_cast<const float4 *>(Cs + r * LDC + c)[0];
        float4 v = cValues[buf][j];
        v.x = alpha * ab.x + beta * v.x;
        v.y = alpha * ab.y + beta * v.y;
        v.z = alpha * ab.z + beta * v.z;
        v.w = alpha * ab.w + beta * v.w;
        cVector(i)[0] = v;
      }
    };
    // Keep two fixed register buffers. A non-unrolled outer loop bounds
    // live C values, while fixed buffer indices permit scalar replacement.
#pragma unroll 1
    for (int batch = 0; batch < BATCHES; batch += 2) {
      if (batch + 1 < BATCHES) loadC(1, batch + 1);
      storeC(0, batch);
      if (batch + 2 < BATCHES) loadC(0, batch + 2);
      if (batch + 1 < BATCHES) storeC(1, batch + 1);
    }

}
