#pragma once

#include "common.cuh"
#include "06_vectorized_frag.cuh"

template <bool GUARDED>
__device__ __forceinline__ uint32_t tc7_round_bits(uint32_t bits) {
  if constexpr (GUARDED) {
    // Match the cuBLAS SASS strategy: preserve Inf/NaN, otherwise add half
    // a TF32 ulp. MMA ignores the low bits, so no explicit truncation is needed.
    uint32_t result = bits;
    const float value = __uint_as_float(bits);
    asm volatile("{ .reg .pred special; .reg .f32 magnitude; "
                 "abs.f32 magnitude, %1; "
                 "setp.geu.f32 special, magnitude, 0f7f800000; "
                 "@!special mad.lo.u32 %0, %0, 1, 4096; }"
                 : "+r"(result) : "f"(value));
    return result;
  } else return bits + 0x1000u;
}

template <int MODE>
__device__ __forceinline__ uint32_t tc7_fragment_bits(uint32_t bits) {
  if constexpr (MODE >= 2) return bits;
  else return tc7_round_bits<MODE == 1>(bits);
}

template <bool GUARDED>
__device__ __forceinline__ void tc7_round_inplace(uint32_t &bits) {
  if constexpr (GUARDED) {
    const float value = __uint_as_float(bits);
    asm volatile("{ .reg .pred special; .reg .f32 magnitude; "
                 "abs.f32 magnitude, %1; "
                 "setp.geu.f32 special, magnitude, 0f7f800000; "
                 "@!special mad.lo.u32 %0, %0, 1, 4096; }"
                 : "+r"(bits) : "f"(value));
  } else {
    asm volatile("add.u32 %0, %0, 4096;" : "+r"(bits));
  }
}

// Kernel 7: warp 수준 소프트웨어 파이프라인 (CUTLASS MmaMultistage와 같은 구조).
//
// kernel 6의 PC sampling에서 남은 손실은 fragment LDS 결과 대기(short scoreboard, MIO)와
// 타일 경계 barrier였다. SM당 warp가 8개뿐이라 다른 warp가 이 지연을 가려 주지 못한다.
//
// 1) fragment 더블 버퍼링: k8 step i의 mma를 내기 전에 step i+1의 fragment LDS를 먼저 낸다.
//    LDS 지연이 이전 step의 HMMA 뒤에 숨는다.
// 2) 타일 경계: 마지막 step 직전에 cp.async wait + __syncthreads를 하고,
//    마지막 step의 mma보다 먼저 다음 타일 step 0의 fragment를 읽는다.
// 3) threadblock rasterization: 연속 실행되는 block들이 GROUP개 행 블록 안에서 열 방향으로
//    지그재그로 돌게 해 A 행 패널과 B 열 패널을 L2에서 재사용한다 (group <= 0이면 끔).
// 4) step 단위로 fragment를 들고 있으려면(레지스터 2배) A를 LDS.64로 읽어야 한다.
//    k 순열을 k8 step 단위로 바꿔 lane t가 step의 k = 8s+2t, 8s+2t+1을 쓰게 하면
//    A의 두 값이 인접해 LDS.64 하나로 읽힌다.
//    A swizzle: chunk ^ ((row & 3) << 1)   (half-warp 16 lane의 LDS.64가 32 bank에 퍼짐)
//    B swizzle: chunk ^ f(t), t = (row >> 1) & 3  (quarter-warp의 LDS.128이 32 bank에 퍼짐)
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int STAGES, const int NUM_THREADS, const int MMA_ORDER = 0,
          const int COPY_MODE = 0, const int EPILOGUE_VECTORS = 1, const int ROUND_MODE = 0>
__global__ void __launch_bounds__(NUM_THREADS)
    tcgemmFragPipe(int M, int N, int K, float alpha, const float *A,
                   const float *B, float beta, float *C, int group) {
  constexpr bool DEFER_ROUNDING = ROUND_MODE >= 2;
  constexpr bool RAKE_B = COPY_MODE == 4;
  constexpr bool CU_COPY = COPY_MODE == 3 || RAKE_B;
  static_assert(ROUND_MODE >= 0 && ROUND_MODE <= 3);
  constexpr bool LDM_A = MMA_ORDER >= 3;
  constexpr bool NATURAL_N = MMA_ORDER >= 5;
  constexpr int TRAVERSAL = NATURAL_N ? MMA_ORDER - 5 : LDM_A ? MMA_ORDER - 3 : MMA_ORDER;
  constexpr bool DISTRIBUTE_COPY = COPY_MODE != 0;
  constexpr bool STEP_POINTERS = COPY_MODE == 2;
  static_assert(COPY_MODE >= 0 && COPY_MODE <= 4);
  static_assert(EPILOGUE_VECTORS == 1 || EPILOGUE_VECTORS == 4 || EPILOGUE_VECTORS == 8);
  static_assert(BM * BN % (4 * NUM_THREADS * EPILOGUE_VECTORS) == 0);
  constexpr int A_STAGE = BM * BK;
  constexpr int B_STAGE = BK * BN;
  constexpr int MFRAG = WM / 16;
  constexpr int NFRAG = WN / 8;
  constexpr int LDC = BN + (NATURAL_N ? 8 : 4);
  constexpr int KSTEPS = BK / 8;
  static_assert((BM / WM) * (BN / WN) * WARPSIZE == NUM_THREADS);
  static_assert(MMA_ORDER >= 0 && MMA_ORDER <= 6);
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
        cp_async_16(aD + j * A_ROWS * BK, aG + (long)(j - (STEP_POINTERS && part >= 0 ? part : 0)) * A_ROWS * K);
    float *bD = Bs + stage * B_STAGE;
#pragma unroll
    for (int j = 0; j < B_ITERS; ++j) {
      const int row = bRow0 + (RAKE_B ? 0 : j * B_ROWS);
      const int chunk = bCh + (RAKE_B ? j * B_CH : 0);
      if (part < 0 || (CU_COPY ? j / B_GROUP : j % KSTEPS) == part)
        cp_async_16(bD + row * BN + ((chunk ^ swzB(LDM_A ? (row & 3) : ((row >> 1) & 3))) << 2),
                    bG + (RAKE_B ? (long)j * B_CH * 4 :
                      (long)(j - (STEP_POINTERS && part >= 0 ? part : 0)) * B_ROWS * N));
    }
    if (STEP_POINTERS && part >= 0) {
      // Carry source pointers between copy steps. j-part is a compile-time
      // 0 or KSTEPS, avoiding a fresh row-stride product for each step.
      aG += part < KSTEPS - 1 ? (long)A_ROWS * K
                              : BK - (long)(KSTEPS - 1) * A_ROWS * K;
      bG += part < KSTEPS - 1 ? (long)B_ROWS * N
                              : (long)(BK - (KSTEPS - 1) * B_ROWS) * N;
    } else if (part < 0 || part == KSTEPS - 1) {
      aG += BK;
      bG += (long)BK * N;
    }
  };

  // ---- fragment 주소 ----
  // A: 행 = warpRow*WM + g (+ m*16 + h*8), row&3 = g&3로 고정.
  //    step의 논리 chunk = 2*step + t/2 -> 물리 chunk = 2*(step ^ (g&3)) + t/2.
  const int aRowOff = (warpRow * WM + g) * BK + (t >> 1) * 4 + (t & 1) * 2;
  const int gSw = g & 3;
  // B: 행 = 8*step + 2t + r2 이므로 swizzle t = (row>>1)&3 = t로 고정.
  int bFragOff[NFRAG / 4];
#pragma unroll
  for (int q = 0; q < NFRAG / 4; ++q)
    bFragOff[q] = (LDM_A ? t : 2 * t) * BN +
                  ((((warpCol * WN + g * NFRAG) / 4 + q) ^ swzB(t)) << 2);

  int bNaturalOff[NFRAG];
#pragma unroll
  for (int n = 0; n < NFRAG; ++n)
    bNaturalOff[n] = t * BN + ((warpCol * WN + 8 * n + g) ^ (t << 3));

  uint32_t aFrag[2][MFRAG][4];
  uint32_t bFrag[2][NFRAG][2];

  auto loadFrag = [&](int buf, const float *as, const float *bs, int step) {
    if constexpr (LDM_A) {
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
        aFrag[buf][m][0] = tc7_fragment_bits<ROUND_MODE>(r0);
        aFrag[buf][m][1] = tc7_fragment_bits<ROUND_MODE>(r1);
        aFrag[buf][m][2] = tc7_fragment_bits<ROUND_MODE>(r2);
        aFrag[buf][m][3] = tc7_fragment_bits<ROUND_MODE>(r3);
      }
    } else {
    const float *aBase = as + aRowOff + ((step ^ gSw) << 3);
#pragma unroll
    for (int m = 0; m < MFRAG; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 v =
            reinterpret_cast<const float2 *>(aBase + (m * 16 + h * 8) * BK)[0];
        aFrag[buf][m][h + 0] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v.x)); // a0 / a1 (k0)
        aFrag[buf][m][h + 2] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v.y)); // a2 / a3 (k1)
      }
    }
    }
    if constexpr (NATURAL_N) {
#pragma unroll
      for (int n = 0; n < NFRAG; ++n) {
#pragma unroll
        for (int r2 = 0; r2 < 2; ++r2) {
          const float v = bs[bNaturalOff[n] + (8 * step + 4 * r2) * BN];
          bFrag[buf][n][r2] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v));
        }
      }
    } else {
#pragma unroll
    for (int r2 = 0; r2 < 2; ++r2) {
#pragma unroll
      for (int q = 0; q < NFRAG / 4; ++q) {
        const float4 v = reinterpret_cast<const float4 *>(
            bs + bFragOff[q] + (8 * step + (LDM_A ? 4 * r2 : r2)) * BN)[0];
        bFrag[buf][4 * q + 0][r2] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v.x));
        bFrag[buf][4 * q + 1][r2] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v.y));
        bFrag[buf][4 * q + 2][r2] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v.z));
        bFrag[buf][4 * q + 3][r2] = tc7_fragment_bits<ROUND_MODE>(__float_as_uint(v.w));
      }
    }    }

  };

  auto roundFrag = [&](int buf) {
#pragma unroll
    for (int m = 0; m < MFRAG; ++m)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        tc7_round_inplace<ROUND_MODE == 2>(aFrag[buf][m][r]);
#pragma unroll
    for (int n = 0; n < NFRAG; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r)
        tc7_round_inplace<ROUND_MODE == 2>(bFrag[buf][n][r]);
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
  if constexpr (DEFER_ROUNDING) roundFrag(0);

  for (int tile = 0; tile < numTiles; ++tile) {
#pragma unroll
    for (int step = 0; step < KSTEPS; ++step) {
      // Spread global copies across HMMA steps instead of issuing a burst.
      // All pieces belong to one async group, committed before the stage wait.
      if (!CU_COPY && (DISTRIBUTE_COPY || step == 0)) {
        if (tile + STAGES - 1 < numTiles)
          loadTile(writeStage, DISTRIBUTE_COPY ? step : -1);
        if (!DISTRIBUTE_COPY || step == KSTEPS - 1) {
          cp_async_commit();
          writeStage = writeStage == STAGES - 1 ? 0 : writeStage + 1;
        }
      }
      if (!CU_COPY && step == KSTEPS - 1) {
        // 다음 타일 데이터가 도착했는지 확인하고 stage를 넘긴다.
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        readStage = readStage == STAGES - 1 ? 0 : readStage + 1;
      }
      // 다음 step(또는 다음 타일 step 0)의 fragment를 현재 mma보다 먼저 요청한다.
      if (step < KSTEPS - 1) {
        loadFrag((step + 1) & 1, as, bs, step + 1);
      } else if (tile + 1 < numTiles) {
        as = As + readStage * A_STAGE;
        bs = Bs + readStage * B_STAGE;
        loadFrag(0, as, bs, 0);
      }
      if constexpr (DEFER_ROUNDING)
        if (step > 0) roundFrag(step & 1);
#pragma unroll
      for (int outer = 0; outer < (TRAVERSAL == 2 ? NFRAG : MFRAG); ++outer) {
#pragma unroll
        for (int inner = 0; inner < (TRAVERSAL == 2 ? MFRAG : NFRAG); ++inner) {
          // Compile-time permutations preserve each accumulator's K order.
          // Serpentine traversal reuses an operand at row/column boundaries.
          const int m = TRAVERSAL == 2 ? ((outer & 1) ? MFRAG - 1 - inner : inner) : outer;
          const int n = TRAVERSAL == 2 ? outer :
              ((TRAVERSAL == 1 && (outer & 1)) ? NFRAG - 1 - inner : inner);
          mma_m16n8k8_tf32(acc[m][n], aFrag[step & 1][m], bFrag[step & 1][n]);
        }
      }
      if constexpr (CU_COPY) {
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
      if constexpr (DEFER_ROUNDING)
        if (step == KSTEPS - 1 && tile + 1 < numTiles) roundFrag(0);

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
  if constexpr (EPILOGUE_VECTORS > 1) {
    // Keep these loads after the MMA loop in compiler IR; holding them
    // across the main loop would increase its register requirement.
    asm volatile("" ::: "memory");
    loadC(0, 0);
  }

  cp_async_wait<0>();
  __syncthreads();

  // epilogue: kernel 6과 같다 (fragment n의 열 j -> 실제 열 j*NFRAG + n).
  float *Cs = smem;
  if constexpr (NATURAL_N) {
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
  } else {
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
  }
  __syncthreads();

  if constexpr (EPILOGUE_VECTORS == 1) {
  for (int i = tid; i < BM * BN / 4; i += NUM_THREADS) {
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
  }  } else {
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

}
