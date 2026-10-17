#pragma once
#include "common.cuh"
#include <mma.h>

__global__ void bf16Simt(Problem p) {
  const int row=blockIdx.y*16+threadIdx.y, col=blockIdx.x*16+threadIdx.x;
  float acc=0;
  for(int k=0;k<p.k;++k) acc=fmaf(__bfloat162float(p.a[row*p.k+k]),__bfloat162float(p.b[k*p.n+col]),acc);
  p.d[row*p.n+col]=acc;
}

__global__ void bf16WmmaNaive(Problem p) {
  using namespace nvcuda;
  const int warp=threadIdx.x/32, row=(blockIdx.y*2+warp/2)*16, col=(blockIdx.x*2+warp%2)*16;
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> acc;
  wmma::fill_fragment(acc,0.f);
  for(int k=0;k<p.k;k+=16) {
    wmma::load_matrix_sync(a,p.a+row*p.k+k,p.k);
    wmma::load_matrix_sync(b,p.b+k*p.n+col,p.n);
    wmma::mma_sync(acc,a,b,acc);
  }
  wmma::store_matrix_sync(p.d+row*p.n+col,acc,p.n,wmma::mem_row_major);
}

__global__ void bf16WmmaSmem(Problem p) {
  using namespace nvcuda;
  constexpr int BM=64,BN=64,BK=32;
  __shared__ __align__(32) __nv_bfloat16 as[BM*BK],bs[BK*BN];
  const int warp=threadIdx.x/32, wr=warp/2*32,wc=warp%2*32;
  const int row=blockIdx.y*BM,col=blockIdx.x*BN;
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a[2];
  wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b[2];
  wmma::fragment<wmma::accumulator,16,16,16,float> acc[2][2];
  for(int m=0;m<2;++m) for(int n=0;n<2;++n) wmma::fill_fragment(acc[m][n],0.f);
  for(int k=0;k<p.k;k+=BK) {
    for(int i=threadIdx.x;i<BM*BK/8;i+=128) {
      int r=i/(BK/8),q=i%(BK/8)*8;
      reinterpret_cast<uint4*>(as+r*BK+q)[0]=reinterpret_cast<const uint4*>(p.a+(row+r)*p.k+k+q)[0];
    }
    for(int i=threadIdx.x;i<BK*BN/8;i+=128) {
      int r=i/(BN/8),q=i%(BN/8)*8;
      reinterpret_cast<uint4*>(bs+r*BN+q)[0]=reinterpret_cast<const uint4*>(p.b+(k+r)*p.n+col+q)[0];
    }
    __syncthreads();
    for(int kk=0;kk<BK;kk+=16) {
      for(int m=0;m<2;++m) wmma::load_matrix_sync(a[m],as+(wr+m*16)*BK+kk,BK);
      for(int n=0;n<2;++n) wmma::load_matrix_sync(b[n],bs+kk*BN+wc+n*16,BN);
      for(int m=0;m<2;++m) for(int n=0;n<2;++n) wmma::mma_sync(acc[m][n],a[m],b[n],acc[m][n]);
    }
    __syncthreads();
  }
  for(int m=0;m<2;++m) for(int n=0;n<2;++n) {
    const int off=(row+wr+m*16)*p.n+col+wc+n*16;
    wmma::store_matrix_sync(p.d+off,acc[m][n],p.n,wmma::mem_row_major);
  }
}
