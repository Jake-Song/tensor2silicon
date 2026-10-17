#pragma once
#include "../runner.cuh"

__device__ __forceinline__ void mma_bf16(float d[4], const unsigned a[4], const unsigned b[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
__device__ __forceinline__ unsigned shared_addr(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldm_a(unsigned r[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(shared_addr(p)) : "memory");
}
__device__ __forceinline__ void ldm_b(unsigned r[2], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
    : "=r"(r[0]),"=r"(r[1]) : "r"(shared_addr(p)) : "memory");
}
__device__ __forceinline__ void ldm_b4(unsigned r[4], const void *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(shared_addr(p)) : "memory");
}
__device__ __forceinline__ void ldm_a_addr(unsigned r[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldm_b_addr(unsigned r[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
    : "=r"(r[0]),"=r"(r[1]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldm_b4_addr(unsigned r[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void cp16(void *d, const void *s) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(shared_addr(d)),"l"(s) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template<int N> __device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}
