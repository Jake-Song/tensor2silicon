#pragma once
#include "common.cuh"

// BF16 layouts are independent from the A100 TF32 code. For BK=32, row parity
// already selects half of the banks; the other row bits swizzle four chunks.
template<int BM,int BN,int BK,int WM,int WN,int STAGES,bool ASYNC,bool DOUBLE,int EPI=0,int COPY=0,int ORDER=0,bool CROSS=false,bool PACKB=false,bool POINTERS=false,bool ADDR=false,bool RING=false>
__global__ __launch_bounds__((BM/WM)*(BN/WN)*32)
void bf16Pipeline(Problem p, int group) {
  constexpr int THREADS=(BM/WM)*(BN/WN)*32, LA=BK;
  constexpr int AS=BM*LA,BS=BK*BN,MF=WM/16,NF=WN/8,KS=BK/16;
  extern __shared__ __align__(128) __nv_bfloat16 sm[];
  auto as=sm; auto bs=sm+STAGES*AS;
  int bx=blockIdx.x,by=blockIdx.y;
  if(group>0) {
    int linear=by*gridDim.x+bx, per=group*gridDim.x, first=linear/per*group;
    int rows=min(int(gridDim.y)-first,group);
    by=first+(linear%per)%rows; bx=(linear%per)/rows;
  }
  const int tid=threadIdx.x, lane=tid%32, warp=tid/32;
  const int wr=warp/(BN/WN)*WM,wc=warp%(BN/WN)*WN;
  const auto ag=p.a+by*BM*p.k;
  const auto bg=p.b+bx*BN;
  auto dg=p.d+by*BM*p.n+bx*BN;
  auto ax=[] __device__ (int r,int c) { return r*LA+(c^(((BK==32?r>>1:r)&(BK/8-1))*8)); };
  auto bxoff=[] __device__ (int r,int c) { return r*BN+(c^(((BN==32?r>>1:r)&(BN==32?3:7))*8)); };
  constexpr bool RAKE=COPY==2 || COPY==3 || COPY==5;
  constexpr bool DISTRIBUTE=COPY==1 || COPY==2 || COPY==4 || COPY==5;
  constexpr bool AFTER=COPY==4 || COPY==5;
  const auto *ap=ag+(tid/(BK/8))*p.k+(tid%(BK/8))*8;
  constexpr int BCH=RAKE?THREADS/BK:BN/8;
  const auto *bp=bg+(tid/BCH)*p.n+(tid%BCH)*8;
  auto copy_tile=[&](int stage,int tile,int part=-1) {
    auto ad=as+stage*AS; auto bd=bs+stage*BS;
#pragma unroll
    for(int j=0;j<(BM*BK/8+THREADS-1)/THREADS;++j) {
      int i=tid+j*THREADS;
      if(i<BM*BK/8 && (part<0 || j%KS==part)) {
        int r=i/(BK/8),c=i%(BK/8)*8;
        auto src=POINTERS?ap+j*(THREADS/(BK/8))*p.k:ag+r*p.k+tile*BK+c;
        auto dst=ad+ax(r,c);
        if constexpr(ASYNC) cp16(dst,src);
        else reinterpret_cast<uint4*>(dst)[0]=reinterpret_cast<const uint4*>(src)[0];
      }
    }
#pragma unroll
    for(int j=0;j<(BK*BN/8+THREADS-1)/THREADS;++j) {
      int i=tid+j*THREADS;
      if(i<BK*BN/8 && (part<0 || j%KS==part)) {
        int r,c;
        if constexpr(RAKE) {
          constexpr int CH=THREADS/BK;
          r=tid/CH; c=(tid%CH+j*CH)*8;
        } else { r=i/(BN/8); c=i%(BN/8)*8; }
        auto src=POINTERS?(RAKE?bp+j*BCH*8:bp+j*(THREADS/(BN/8))*p.n):bg+(tile*BK+r)*p.n+c;
        auto dst=bd+bxoff(r,c);
        if constexpr(ASYNC) cp16(dst,src);
        else reinterpret_cast<uint4*>(dst)[0]=reinterpret_cast<const uint4*>(src)[0];
      }
    }
    if constexpr(POINTERS) if(part<0 || part==KS-1) { ap+=BK;bp+=BK*p.n; }
  };
  float acc[MF][NF][4]={};
  unsigned af[DOUBLE?2:1][MF][4],bf[DOUBLE?2:1][NF][2];
  unsigned aa[MF],ba[PACKB?NF/2:NF];
  if constexpr(ADDR) {
#pragma unroll
    for(int m=0;m<MF;++m) {
      unsigned addr=shared_addr(as+ax(wr+(lane&15)+m*16,(lane/16)*8));
      asm volatile("mov.u32 %0, %1;" : "=r"(aa[m]) : "r"(addr));
    }
#pragma unroll
    for(int n=0;n<(PACKB?NF/2:NF);++n) {
      unsigned addr=shared_addr(bs+bxoff(lane&15,wc+(PACKB?n*16+(lane/16)*8:n*8)));
      asm volatile("mov.u32 %0, %1;" : "=r"(ba[n]) : "r"(addr));
    }
  }
  auto load=[&](int buf,int stage,int step) {
    auto a=as+stage*AS; auto b=bs+stage*BS;
    int ar=wr+(lane&15),ac=step*16+(lane/16)*8;
#pragma unroll
    for(int m=0;m<MF;++m) {
      if constexpr(ADDR)ldm_a_addr(af[buf][m],(aa[m]+stage*AS*2)^(step*32));
      else ldm_a(af[buf][m],a+ax(ar+m*16,ac));
    }
    int br=step*16+(lane&15);
    if constexpr(PACKB) {
#pragma unroll
      for(int n=0;n<NF;n+=2) {
        unsigned r[4];
        if constexpr(ADDR)ldm_b4_addr(r,ba[n/2]+stage*BS*2+step*16*BN*2);
        else ldm_b4(r,b+bxoff(br,wc+n*8+(lane/16)*8));
        bf[buf][n][0]=r[0];bf[buf][n][1]=r[1];
        bf[buf][n+1][0]=r[2];bf[buf][n+1][1]=r[3];
      }
    } else {
#pragma unroll
      for(int n=0;n<NF;++n) {
        if constexpr(ADDR)ldm_b_addr(bf[buf][n],ba[n]+stage*BS*2+step*16*BN*2);
        else ldm_b(bf[buf][n],b+bxoff(br,wc+n*8));
      }
    }
  };
  auto compute=[&](int buf) {
    if constexpr(ORDER<2) {
#pragma unroll
    for(int m=0;m<MF;++m)
#pragma unroll
      for(int n0=0;n0<NF;++n0) {
        int n=ORDER && (m&1)?NF-1-n0:n0;
        mma_bf16(acc[m][n],af[buf][m],bf[buf][n]);
      }
    } else if constexpr(ORDER<4) {
#pragma unroll
      for(int n=0;n<NF;++n)
#pragma unroll
        for(int m0=0;m0<MF;++m0) {
          int m=ORDER==3 && (n&1)?MF-1-m0:m0;
          mma_bf16(acc[m][n],af[buf][m],bf[buf][n]);
        }
    } else {
      static_assert(MF%2==0 && NF%2==0);
#pragma unroll
      for(int m=0;m<MF;m+=2)
#pragma unroll
        for(int n=0;n<NF;n+=2)
#pragma unroll
          for(int j=0;j<2;++j)
#pragma unroll
            for(int i=0;i<2;++i)mma_bf16(acc[m+i][n+j],af[buf][m+i],bf[buf][n+j]);
    }
  };
  const int tiles=p.k/BK;
  if constexpr(ASYNC) {
#pragma unroll
    for(int s=0;s<STAGES-1;++s) { if(s<tiles) copy_tile(s,s); commit(); }
  }
  if constexpr(CROSS) {
    static_assert(ASYNC && DOUBLE);
    wait<STAGES-2>(); __syncthreads(); load(0,0,0);
  }
  int read_stage=0,write_stage=STAGES-1;
  for(int tile=0;tile<tiles;++tile) {
    int stage=RING?read_stage:tile%STAGES,next=tile+STAGES-1;
    int dest=RING?write_stage:next%STAGES;
    if constexpr(!CROSS) {
      if constexpr(ASYNC) wait<STAGES-2>();
      else copy_tile(0,tile);
      __syncthreads();
    }
    if constexpr(ASYNC && !DISTRIBUTE) { if(next<tiles) copy_tile(dest,next); commit(); }
    if constexpr(DOUBLE && !CROSS) load(0,stage,0);
#pragma unroll
    for(int step=0;step<KS;++step) {
      if constexpr(ASYNC && DISTRIBUTE) {
        if(!AFTER || (CROSS && step==KS-1))if(next<tiles)copy_tile(dest,next,step);
      }
      if constexpr(DOUBLE) {
        if(step+1<KS) load((step+1)%2,stage,step+1);
        if constexpr(CROSS) {
          if(step==KS-1) {
            if constexpr(DISTRIBUTE) commit();
            if(tile+1<tiles) {
              // All shared reads of this tile have reached registers. Synchronize
              // before the last MMA, and prefetch the next tile's first fragment.
              wait<STAGES-2>(); __syncthreads();
              load(0,RING?(read_stage+1==STAGES?0:read_stage+1):(tile+1)%STAGES,0);
            }
          }
        }
        compute(step%2);
      } else { load(0,stage,step); compute(0); }
      if constexpr(ASYNC && DISTRIBUTE && AFTER) {
        if(!(CROSS && step==KS-1))if(next<tiles)copy_tile(dest,next,step);
      }
    }
    if constexpr(ASYNC && DISTRIBUTE && !CROSS) commit();
    if constexpr(!ASYNC) __syncthreads();
    if constexpr(RING) {
      read_stage=read_stage+1==STAGES?0:read_stage+1;
      write_stage=write_stage+1==STAGES?0:write_stage+1;
    }
  }
  if constexpr(ASYNC) wait<0>();
  if constexpr(EPI==0) {
#pragma unroll
    for(int m=0;m<MF;++m)
#pragma unroll
      for(int n=0;n<NF;++n) {
        int row=wr+m*16+lane/4,col=wc+n*8+(lane%4)*2;
#pragma unroll
        for(int h=0;h<2;++h) {
          int off=(row+h*8)*p.n+col;
          reinterpret_cast<float2*>(dg+off)[0]=make_float2(acc[m][n][2*h],acc[m][n][2*h+1]);
        }
      }
  } else {
    __syncthreads();
    // Reuse the input staging allocation for a bank-padded FP32 output tile.
    constexpr int LD=BN+4;
    float *out=reinterpret_cast<float*>(sm);
#pragma unroll
    for(int m=0;m<MF;++m)
#pragma unroll
      for(int n=0;n<NF;++n) {
        int row=wr+m*16+lane/4,col=wc+n*8+(lane%4)*2;
        reinterpret_cast<float2*>(out+row*LD+col)[0]=make_float2(acc[m][n][0],acc[m][n][1]);
        reinterpret_cast<float2*>(out+(row+8)*LD+col)[0]=make_float2(acc[m][n][2],acc[m][n][3]);
      }
    __syncthreads();
#pragma unroll
    for(int base=0;base<BM*BN/4;base+=THREADS*EPI) {
#pragma unroll
      for(int j=0;j<EPI;++j) {
        int i=base+tid+j*THREADS;
        if(i<BM*BN/4) {
          int row=i/(BN/4),col=i%(BN/4)*4;
          float4 v=reinterpret_cast<const float4*>(out+row*LD+col)[0];
          reinterpret_cast<float4*>(dg+row*p.n+col)[0]=v;
        }
      }
    }
  }
}
