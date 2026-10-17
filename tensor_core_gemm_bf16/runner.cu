#include "runner.cuh"
#include "kernels/basics.cuh"
#include "kernels/pipeline.cuh"
#include <algorithm>
#include <limits>

static void simt(const Problem& p,cudaStream_t s,int) { bf16Simt<<<dim3(p.n/16,p.m/16),dim3(16,16),0,s>>>(p); }
static void wmma(const Problem& p,cudaStream_t s,int) { bf16WmmaNaive<<<dim3(p.n/32,p.m/32),128,0,s>>>(p); }
static void wmma_smem(const Problem& p,cudaStream_t s,int) { bf16WmmaSmem<<<dim3(p.n/64,p.m/64),128,0,s>>>(p); }

template<int BM,int BN,int BK,int WM,int WN,int S,bool A,bool D,int E=0,int C=0,int O=0>
constexpr int shared_bytes() {
  constexpr int input=S*(BM*BK+BK*BN)*2;
  return E?std::max(input,BM*(BN+4)*4):input;
}
template<int BM,int BN,int BK,int WM,int WN,int S,bool A,bool D,int E=0,int C=0,int O=0,bool X=false,bool P=false,bool PTR=false,bool ADDR=false,bool RING=false>
static void pipeline(const Problem& p,cudaStream_t stream,int group) {
  constexpr int bytes=shared_bytes<BM,BN,BK,WM,WN,S,A,D,E,C,O>();
  auto fn=bf16Pipeline<BM,BN,BK,WM,WN,S,A,D,E,C,O,X,P,PTR,ADDR,RING>;
  static bool configured=false;
  if(!configured) { ck(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes)); configured=true; }
  fn<<<dim3(p.n/BN,p.m/BM),(BM/WM)*(BN/WN)*32,bytes,stream>>>(p,group);
}
const std::vector<Variant>& variants() {
  static const std::vector<Variant> all=[] {
    std::vector<Variant> v{
      {"s0_simt",0,16,16,16,0,256,simt},
      {"s1_wmma_global",1,32,32,16,0,128,wmma},
      {"s2_wmma_smem",2,64,64,32,8192,128,wmma_smem}};
#define ADD(NAME,ST,BM,BN,BK,WM,WN,S,A,D,E,C,O) \
    v.push_back({NAME,ST,BM,BN,BK,shared_bytes<BM,BN,BK,WM,WN,S,A,D,E,C,O>(),(BM/WM)*(BN/WN)*32,pipeline<BM,BN,BK,WM,WN,S,A,D,E,C,O>});
    ADD("s3_mma_ldmatrix",3,128,128,32,64,32,1,false,false,0,0,0)
    ADD("s4_async",4,128,128,32,64,32,3,true,false,0,0,0)
    ADD("s5_frag_double",5,128,128,32,64,32,3,true,true,0,0,0)
#define TILE(BM,BN,BK,WM,WN,S) \
    ADD("t" #BM "x" #BN "k" #BK "w" #WM "x" #WN "s" #S,6,BM,BN,BK,WM,WN,S,true,true,0,0,0)
    TILE(64,64,32,32,32,2) TILE(64,64,32,32,32,3) TILE(64,64,32,32,32,4)
    TILE(32,32,32,32,32,2) TILE(32,32,32,32,32,3) TILE(32,32,32,32,32,4)
    TILE(32,32,64,32,32,2) TILE(32,32,64,32,32,3) TILE(32,32,64,32,32,4)
    TILE(32,64,32,32,32,2) TILE(32,64,32,32,32,3) TILE(32,64,32,32,32,4)
    TILE(32,64,64,32,32,2) TILE(32,64,64,32,32,3) TILE(32,64,64,32,32,4)
    TILE(64,32,32,32,32,2) TILE(64,32,32,32,32,3) TILE(64,32,32,32,32,4)
    TILE(64,32,64,32,32,2) TILE(64,32,64,32,32,3) TILE(64,32,64,32,32,4)
    TILE(64,128,32,32,64,2) TILE(64,128,32,32,64,3) TILE(64,128,32,32,64,4)
    TILE(128,64,32,64,32,2) TILE(128,64,32,64,32,3) TILE(128,64,32,64,32,4)
    TILE(128,128,32,64,32,2) TILE(128,128,32,64,32,4)
    TILE(128,128,32,64,64,2) TILE(128,128,32,64,64,3)
    TILE(128,256,32,64,64,2) TILE(128,256,32,64,64,3)
    TILE(256,128,32,64,64,2) TILE(256,128,32,64,64,3)
    TILE(64,64,64,32,32,2) TILE(64,64,64,32,32,3) TILE(64,64,64,32,32,4)
    TILE(64,128,64,32,64,2) TILE(64,128,64,32,64,3)
    TILE(128,64,64,64,32,2) TILE(128,64,64,64,32,3)
    TILE(128,128,64,64,32,2) TILE(128,128,64,64,32,3)
    TILE(128,256,64,64,64,2) TILE(256,128,64,64,64,2)
#define FINAL(NAME,BM,BN,BK,WM,WN,S,E,C,O) ADD(NAME,6,BM,BN,BK,WM,WN,S,true,true,E,C,O)
    FINAL("f64_epi1",64,64,32,32,32,3,1,0,0)
    FINAL("f64_epi4",64,64,32,32,32,3,4,0,0)
    FINAL("f64_dist",64,64,32,32,32,3,4,1,0)
    FINAL("f64_rake",64,64,32,32,32,3,4,2,1)
    FINAL("f128_epi1",128,128,32,64,32,3,1,0,0)
    FINAL("f128_epi4",128,128,32,64,32,3,4,0,0)
    FINAL("f128_dist",128,128,32,64,32,3,4,1,0)
    FINAL("f128_rake",128,128,32,64,32,3,4,2,1)
    FINAL("f128_rake64",128,128,64,64,32,2,4,2,1)
    FINAL("f128_w64",128,128,32,64,64,3,4,2,1)
    FINAL("f128x256_dist",128,256,32,64,64,3,0,1,1)
    FINAL("f128x256_rake",128,256,32,64,64,3,0,2,1)
    FINAL("f256x128_dist",256,128,32,64,64,3,0,1,1)
    FINAL("f256x128_rake",256,128,32,64,64,3,0,2,1)
#define CROSS(NAME,BM,BN,BK,WM,WN,S,E,C,O) \
    v.push_back({NAME,6,BM,BN,BK,shared_bytes<BM,BN,BK,WM,WN,S,true,true,E,C,O>(),(BM/WM)*(BN/WN)*32,pipeline<BM,BN,BK,WM,WN,S,true,true,E,C,O,true>});
    CROSS("x128_s2",128,128,32,64,64,2,0,0,0)
    CROSS("x128_s3",128,128,32,64,64,3,0,0,0)
    CROSS("x128_s4",128,128,32,64,64,4,0,0,0)
    CROSS("x128_dist",128,128,32,64,64,3,0,1,0)
    CROSS("x128_rake",128,128,32,64,64,3,0,2,1)
    CROSS("x128_k64",128,128,64,64,64,2,0,0,0)
    CROSS("x128_w32",128,128,32,64,32,3,0,0,0)
    CROSS("x128_epi",128,128,32,64,64,3,4,1,1)
    CROSS("x128x256_s3",128,256,32,64,64,3,0,0,0)
    CROSS("x128x256_dist",128,256,32,64,64,3,0,1,1)
    CROSS("x256x128_dist",256,128,32,64,64,3,0,1,1)
    CROSS("x64_s3",64,64,32,32,32,3,0,0,0)
    CROSS("x64x128_s3",64,128,32,32,64,3,0,0,0)
    CROSS("x128x64_s3",128,64,32,64,32,3,0,0,0)
#undef CROSS
#define PACK(NAME,BM,BN,BK,WM,WN,S,E,C,O,X) \
    v.push_back({NAME,6,BM,BN,BK,shared_bytes<BM,BN,BK,WM,WN,S,true,true,E,C,O>(),(BM/WM)*(BN/WN)*32,pipeline<BM,BN,BK,WM,WN,S,true,true,E,C,O,X,true>});
    PACK("p128_s2",128,128,32,64,64,2,0,0,0,false)
    PACK("p128_s3",128,128,32,64,64,3,0,0,0,false)
    PACK("p128_cross2",128,128,32,64,64,2,0,0,0,true)
    PACK("p128_cross3",128,128,32,64,64,3,0,0,0,true)
    PACK("p128_dist",128,128,32,64,64,3,0,1,1,true)
    PACK("p128_rake",128,128,32,64,64,3,0,2,1,true)
    PACK("p128_k64",128,128,64,64,64,2,0,0,0,true)
    PACK("p128x256",128,256,32,64,64,3,0,1,1,true)
    PACK("p256x128",256,128,32,64,64,3,0,1,1,true)
    PACK("p64_s3",64,64,32,32,32,3,0,0,0,true)
    PACK("p64x128_s3",64,128,32,32,64,3,0,0,0,true)
    PACK("p128x64_s3",128,64,32,64,32,3,0,0,0,true)
#undef PACK
#define PTR(NAME,BM,BN,BK,WM,WN,S,E,C,O,X) \
    v.push_back({NAME,6,BM,BN,BK,shared_bytes<BM,BN,BK,WM,WN,S,true,true,E,C,O>(),(BM/WM)*(BN/WN)*32,pipeline<BM,BN,BK,WM,WN,S,true,true,E,C,O,X,true,true>});
    PTR("q128_s2",128,128,32,64,64,2,0,0,0,false)
    PTR("q128_s3",128,128,32,64,64,3,0,0,0,false)
    PTR("q128_cross3",128,128,32,64,64,3,0,0,0,true)
    PTR("q128_burst_rake",128,128,32,64,64,3,0,3,0,false)
    PTR("q128_dist_rake",128,128,32,64,64,3,0,2,0,false)
    PTR("q128x256_burst",128,256,32,64,64,3,0,3,0,true)
    PTR("q128x256_dist",128,256,32,64,64,3,0,2,1,true)
    PTR("q256x128_dist",256,128,32,64,64,3,0,2,1,true)
    PTR("q128_k64",128,128,64,64,64,2,0,0,0,false)
    PTR("q128x64_s2",128,64,32,64,32,2,0,0,0,false)
    PTR("q128x64_s3",128,64,32,64,32,3,0,0,0,true)
#undef PTR
#define ZONE(BM,BN,S,C,O,X) \
    v.push_back({"z" #BM "x" #BN "s" #S "c" #C "o" #O "x" #X,6,BM,BN,32,shared_bytes<BM,BN,32,64,64,S,true,true,0,C,O>(),(BM/64)*(BN/64)*32,pipeline<BM,BN,32,64,64,S,true,true,0,C,O,X,true,true>});
#define ZC(BM,BN,C) \
    ZONE(BM,BN,3,C,0,false) ZONE(BM,BN,3,C,0,true) \
    ZONE(BM,BN,3,C,1,false) ZONE(BM,BN,3,C,1,true) \
    ZONE(BM,BN,3,C,2,false) ZONE(BM,BN,3,C,2,true) \
    ZONE(BM,BN,3,C,3,false) ZONE(BM,BN,3,C,3,true) \
    ZONE(BM,BN,3,C,4,false) ZONE(BM,BN,3,C,4,true)
    ZC(128,256,0) ZC(128,256,1) ZC(128,256,2)
    ZC(128,256,3) ZC(128,256,4) ZC(128,256,5)
    ZC(256,128,0) ZC(256,128,1) ZC(256,128,2)
    ZC(256,128,3) ZC(256,128,4) ZC(256,128,5)
#define ZST(BM,BN,S) \
    ZONE(BM,BN,S,0,0,false) ZONE(BM,BN,S,0,0,true) \
    ZONE(BM,BN,S,0,1,false) ZONE(BM,BN,S,0,1,true) \
    ZONE(BM,BN,S,1,0,false) ZONE(BM,BN,S,1,0,true) \
    ZONE(BM,BN,S,2,1,false) ZONE(BM,BN,S,2,1,true)
    ZST(128,256,2) ZST(128,256,4) ZST(256,128,2) ZST(256,128,4)
#undef ZST
#undef ZC
#undef ZONE
#define HOIST(NAME,BM,BN,S,C,O,X,A,R) \
    v.push_back({NAME,6,BM,BN,32,shared_bytes<BM,BN,32,64,64,S,true,true,0,C,O>(),(BM/64)*(BN/64)*32,pipeline<BM,BN,32,64,64,S,true,true,0,C,O,X,true,true,A,R>});
    HOIST("h128_addr",128,128,2,0,0,false,true,false)
    HOIST("h128_ring",128,128,3,0,0,true,false,true)
    HOIST("h128_both",128,128,3,0,0,true,true,true)
    HOIST("h128x256_addr",128,256,3,4,2,true,true,false)
    HOIST("h128x256_ring",128,256,3,4,2,true,false,true)
    HOIST("h128x256_both",128,256,3,4,2,true,true,true)
    HOIST("h256x128_addr",256,128,3,3,3,true,true,false)
    HOIST("h256x128_ring",256,128,3,3,3,true,false,true)
    HOIST("h256x128_both",256,128,3,3,3,true,true,true)
#undef HOIST
#undef FINAL
#undef TILE
#undef ADD
    return v;
  }();
  return all;
}

void run(const Variant& v,const Problem& p,cudaStream_t stream,int group) {
  if(p.m<=0||p.n<=0||p.k<=0||p.m%v.bm||p.n%v.bn||p.k%v.bk||group<0)
    throw std::invalid_argument("shape incompatible with " + v.name);
  v.launch(p,stream,group); ck(cudaGetLastError());
}
void reference(cublasHandle_t handle,const Problem& p,const float *af,const float *bf) {
  cb(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  const float alpha=1.f,beta=0.f;
  cb(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,p.n,p.m,p.k,&alpha,
      bf,CUDA_R_32F,p.n,af,CUDA_R_32F,p.k,&beta,p.d,CUDA_R_32F,p.n,
      CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
}
Library::Library(const Problem& p) {
  cb(cublasLtCreate(&handle));
  cb(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  // Row-major D^T=B^T A^T: no transpose or conversion inside the timed region.
  cb(cublasLtMatrixLayoutCreate(&al,CUDA_R_16BF,p.n,p.k,p.n));
  cb(cublasLtMatrixLayoutCreate(&bl,CUDA_R_16BF,p.k,p.m,p.k));
  cb(cublasLtMatrixLayoutCreate(&cl,CUDA_R_32F,p.n,p.m,p.n));
  cb(cublasLtMatrixLayoutCreate(&dl,CUDA_R_32F,p.n,p.m,p.n));
  ck(cudaMalloc(&workspace,workspace_size));
  cublasLtMatmulPreference_t pref;
  cb(cublasLtMatmulPreferenceCreate(&pref));
  cb(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_size,sizeof(workspace_size)));
  cublasLtMatmulHeuristicResult_t choices[32]; int count=0;
  cb(cublasLtMatmulAlgoGetHeuristic(handle,desc,al,bl,cl,dl,pref,32,choices,&count));
  cb(cublasLtMatmulPreferenceDestroy(pref));
  if(!count) throw std::runtime_error("No cuBLASLt BF16/FP32 algorithm");
  cudaEvent_t start,end; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
  cudaStream_t stream;ck(cudaStreamCreate(&stream));
  const bool graph_mode=!(getenv("BF16_TIMING") && std::string(getenv("BF16_TIMING"))=="event");
  float best=std::numeric_limits<float>::infinity();
  cublasLtMatmulAlgo_t best_algorithm=choices[0].algo;
  for(int i=0;i<count;++i) {
    if(choices[i].state!=CUBLAS_STATUS_SUCCESS) continue;
    algorithm=choices[i].algo;
    for(int j=0;j<3;++j)launch(p,stream);
    float samples[3];
    cudaGraphExec_t exec=nullptr;
    if(graph_mode) {
      cudaGraph_t graph;ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
      for(int j=0;j<10;++j)launch(p,stream);
      ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
      ck(cudaGraphDestroy(graph));ck(cudaGraphLaunch(exec,stream));
    }
    for(int sample=0;sample<3;++sample) {
      ck(cudaEventRecord(start,stream));
      if(graph_mode)ck(cudaGraphLaunch(exec,stream));
      else for(int j=0;j<10;++j)launch(p,stream);
      ck(cudaEventRecord(end,stream));ck(cudaEventSynchronize(end));
      ck(cudaEventElapsedTime(&samples[sample],start,end));
    }
    std::sort(samples,samples+3);float ms=samples[1];
    if(exec)ck(cudaGraphExecDestroy(exec));
    if(ms<best) {best=ms;best_algorithm=choices[i].algo;}
  }
  algorithm=best_algorithm;
  int id=-1;size_t written=0;
  cb(cublasLtMatmulAlgoConfigGetAttribute(&algorithm,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written));
  printf("{\"kind\":\"library\",\"m\":%d,\"n\":%d,\"k\":%d,\"algorithm_id\":%d,\"heuristic_candidates\":%d,\"workspace_bytes\":%zu,\"selection_timing\":\"%s\"}\n",p.m,p.n,p.k,id,count,workspace_size,graph_mode?"cuda_graph":"event");
  ck(cudaStreamDestroy(stream));
  ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
}
void Library::launch(const Problem& p,cudaStream_t stream) const {
  const float alpha=1.f,beta=0.f;
  cb(cublasLtMatmul(handle,desc,&alpha,p.b,al,p.a,bl,&beta,p.d,cl,p.d,dl,&algorithm,workspace,workspace_size,stream));
}
Library::~Library() {
  cudaFree(workspace);
  cublasLtMatrixLayoutDestroy(al);cublasLtMatrixLayoutDestroy(bl);
  cublasLtMatrixLayoutDestroy(cl);cublasLtMatrixLayoutDestroy(dl);
  cublasLtMatmulDescDestroy(desc);cublasLtDestroy(handle);
}
