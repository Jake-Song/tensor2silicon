#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

constexpr int WARPSIZE = 32;

// FP32 -> TF32 (round to nearest, ties away from zero).
// 결과는 하위 13비트가 0인 32비트 패턴이며 mma 피연산자로 그대로 넘긴다.
__device__ __forceinline__ uint32_t f32_to_tf32(float x) {
  uint32_t r;
  asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
  return r;
}

// D = A(16x8, row) * B(8x8, col) + D, warp 단위 Tensor Core 명령.
// fragment 레이아웃 (g = lane / 4, t = lane % 4):
//   A: a0=(g, t)   a1=(g+8, t)   a2=(g, t+4)   a3=(g+8, t+4)
//   B: b0=(t, g)   b1=(t+4, g)
//   D: d0=(g, 2t)  d1=(g, 2t+1)  d2=(g+8, 2t)  d3=(g+8, 2t+1)
__device__ __forceinline__ void mma_m16n8k8_tf32(float d[4], const uint32_t a[4],
                                                 const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Ampere 비동기 복사: global -> shared 16바이트, 레지스터를 거치지 않는다.
__device__ __forceinline__ void cp_async_16(void *smem, const void *gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// 가장 최근 N개 그룹을 제외한 모든 cp.async 그룹이 끝날 때까지 기다린다.
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
