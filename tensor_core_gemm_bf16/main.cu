#include "runner.cuh"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

struct Buffers {
  int m,n,k,pattern;
  std::vector<__nv_bfloat16> a,b;
  std::vector<float> af,bf,ref,out;
  __nv_bfloat16 *da{},*db{};
  float *daf{},*dbf{},*dr{},*dd{};
  Buffers(int M,int N,int K,unsigned seed,int pattern=0):m(M),n(N),k(K),pattern(pattern),
      a(size_t(m)*k),b(size_t(k)*n),af(a.size()),bf(b.size()),ref(size_t(m)*n),out(ref.size()) {
    std::mt19937 gen(seed); std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(size_t i=0;i<a.size();++i) {
      float v=pattern==1?0.f:pattern==2?((i%2)?-1.f:1.f)*float(1+(i%7))/8:pattern==3?-std::abs(dist(gen)):dist(gen);
      a[i]=__float2bfloat16_rn(v); af[i]=__bfloat162float(a[i]);
    }
    for(size_t i=0;i<b.size();++i) {
      float v=pattern==1?0.f:pattern==2?((i/n)%2?-1.f:1.f)*float(1+(i%5))/8:pattern==3?-std::abs(dist(gen)):dist(gen);
      b[i]=__float2bfloat16_rn(v); bf[i]=__bfloat162float(b[i]);
    }
    ck(cudaMalloc(&da,a.size()*2));ck(cudaMalloc(&db,b.size()*2));
    ck(cudaMalloc(&daf,af.size()*4));ck(cudaMalloc(&dbf,bf.size()*4));
    ck(cudaMalloc(&dr,ref.size()*4));ck(cudaMalloc(&dd,ref.size()*4));
    ck(cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(daf,af.data(),af.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dbf,bf.data(),bf.size()*4,cudaMemcpyHostToDevice));
  }
  Problem problem() { return {m,n,k,da,db,dd}; }
  void make_reference(cublasHandle_t handle) {
    auto p=problem();p.d=dr; reference(handle,p,daf,dbf);
    ck(cudaMemcpy(ref.data(),dr,ref.size()*4,cudaMemcpyDeviceToHost));
  }
  ~Buffers() {cudaFree(da);cudaFree(db);cudaFree(daf);cudaFree(dbf);cudaFree(dr);cudaFree(dd);}
};
struct Error { double max_rel,l2_rel,max_abs; bool pass; };
static Error verify(Buffers& x) {
  ck(cudaDeviceSynchronize());
  ck(cudaMemcpy(x.out.data(),x.dd,x.out.size()*4,cudaMemcpyDeviceToHost));
  double ma=0,mr=0,ss=0,rr=0;bool finite=true;
  for(size_t i=0;i<x.out.size();++i) {
    double d=double(x.out[i])-x.ref[i]; finite=finite&&std::isfinite(x.out[i]);
    ma=std::max(ma,std::abs(d));mr=std::max(mr,std::abs(double(x.ref[i])));ss+=d*d;rr+=double(x.ref[i])*x.ref[i];
  }
  if(!finite)return {1e300,1e300,1e300,false};
  double a=ma/std::max(mr,1e-30),b=std::sqrt(ss/std::max(rr,1e-60));
  return {a,b,ma,finite && a<=1e-4 && b<=1e-4};
}
static void print_error(const std::string& name,const Buffers& x,const Error& e,int seed,int group,const char *phase="validation") {
  printf("{\"kind\":\"check\",\"variant\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"seed\":%d,\"pattern\":%d,\"group\":%d,\"phase\":\"%s\",\"max_rel\":%.9g,\"l2_rel\":%.9g,\"max_abs\":%.9g,\"pass\":%s}\n",
    name.c_str(),x.m,x.n,x.k,seed,x.pattern,group,phase,e.max_rel,e.l2_rel,e.max_abs,e.pass?"true":"false");
  fflush(stdout);
}
static bool selected(const std::string& name) {
  const char *env=getenv("BF16_VARIANTS");
  return !env || (","+std::string(env)+",").find(","+name+",")!=std::string::npos;
}
static bool supports(const Variant& v,int m,int n,int k,int limit) {
  return m%v.bm==0 && n%v.bn==0 && k%v.bk==0 && v.smem<=limit;
}
static float elapsed(const std::function<void()>& fn,int count,cudaStream_t stream=nullptr) {
  cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));
  ck(cudaEventRecord(a,stream));for(int i=0;i<count;++i) fn();
  ck(cudaEventRecord(b,stream));ck(cudaEventSynchronize(b));
  float ms;ck(cudaEventElapsedTime(&ms,a,b));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
  return ms/count;
}
static int checks(int limit,cublasHandle_t handle) {
  int total=0,failed=0,rejected=0,cpu_checks=0;double worst_max=0,worst_l2=0;
  // Rectangles, K shorter than the pipeline, K exactly one or several ring wraps,
  // and nine-row-block rasterizations exercise every tile's production layout.
  const int shapes[][3]={{64,64,16},{128,128,32},{128,256,64},{256,128,96},{256,256,160},{576,256,64},{1152,512,64},{2304,384,64},{128,128,256}};
  for(int seed:{17,20261017}) for(const auto& shape:shapes) for(int pattern:{0,1,2,3}) {
    Buffers x(shape[0],shape[1],shape[2],seed,pattern);
    {
      auto p=x.problem();x.make_reference(handle);
      if(x.m<=128 && x.n<=128) {
        double max_diff=0,max_ref=0;
        for(int i=0;i<x.m;++i)for(int j=0;j<x.n;++j) {
          double sum=0;for(int k=0;k<x.k;++k)sum+=double(x.af[i*x.k+k])*x.bf[k*x.n+j];
          max_diff=std::max(max_diff,std::abs(sum-x.ref[i*x.n+j]));max_ref=std::max(max_ref,std::abs(sum));
        }
        double error=max_diff/std::max(max_ref,1e-30);++cpu_checks;
        if(error>1e-5)++failed;
        printf("{\"kind\":\"cpu_reference_check\",\"m\":%d,\"n\":%d,\"k\":%d,\"seed\":%d,\"pattern\":%d,\"max_rel\":%.9g,\"pass\":%s}\n",x.m,x.n,x.k,seed,pattern,error,error<=1e-5?"true":"false");
      }
      Library lib(p);ck(cudaMemset(x.dd,0xff,x.out.size()*4));lib.launch(p);auto le=verify(x);print_error("cublas",x,le,seed,0);
      ++total;if(!le.pass)++failed;
      for(const auto& v:variants()) {
        if(!selected(v.name) || !supports(v,x.m,x.n,x.k,limit)) continue;
        for(int group: {0,8}) {
          if(v.stage<3 && group)continue;
          // Poison output: incomplete writes must fail even for zero-input tests.
          ck(cudaMemset(x.dd,0xff,x.out.size()*4));
          run(v,p,nullptr,group);auto e=verify(x);print_error(v.name,x,e,seed,group);
          worst_max=std::max(worst_max,e.max_rel);worst_l2=std::max(worst_l2,e.l2_rel);
          ++total;if(!e.pass)++failed;
        }
      }
    }
  }
  // Public launch contract rejects unsupported shapes rather than silently truncating.
  Buffers x(128,128,32,7);
  for(const auto& v:variants()) if(selected(v.name) && v.smem<=limit) {
    auto p=x.problem();p.n=127;
    try {run(v,p);fprintf(stderr,"missing shape rejection %s\n",v.name.c_str());++failed;}
    catch(const std::invalid_argument&) {++rejected;}
  }
  printf("{\"kind\":\"check_summary\",\"checks\":%d,\"cpu_reference_checks\":%d,\"failed\":%d,\"shape_rejections\":%d,\"worst_max_rel\":%.9g,\"worst_l2_rel\":%.9g}\n",total,cpu_checks,failed,rejected,worst_max,worst_l2);
  return failed?1:0;
}
static int benchmark(int size,int rounds,unsigned seed,int limit,cublasHandle_t handle,bool tuning) {
  std::mt19937 order(seed^0x596193u);
  int group=getenv("BF16_GROUP")?std::stoi(getenv("BF16_GROUP")):8;
  std::vector<int> sizes=size?std::vector<int>{size}:std::vector<int>{128,256,512,1024,2048,4096,8192};
  for(int n:sizes) {
    Buffers x(n,n,n,seed);auto p=x.problem();x.make_reference(handle);Library lib(p);
    cudaStream_t stream;ck(cudaStreamCreate(&stream));
    const bool graph_mode=!(getenv("BF16_TIMING") && std::string(getenv("BF16_TIMING"))=="event");
    struct Trial {std::string name;int stage;std::function<void()> fn;int reps;cudaGraphExec_t graph=nullptr;};
    std::vector<Trial> trials{{"cublas",100,[&]{lib.launch(p,stream);},0}};
    for(const auto& v:variants()) {
      if(!selected(v.name)||!supports(v,n,n,n,limit)||(v.stage==0 && n>2048)) continue;
      if(!tuning && v.stage==6)continue;
      const Variant *ptr=&v;
      trials.push_back({v.name,v.stage,[&,ptr]{run(*ptr,p,stream,group);},0});
    }
    for(auto& t:trials) {
      ck(cudaMemset(x.dd,0xff,x.out.size()*4));t.fn();auto e=verify(x);
      print_error(t.name,x,e,seed,group,"before");if(!e.pass)return 1;
      for(int i=0;i<5;++i)t.fn();
      float ms=elapsed(t.fn,3,stream);
      t.reps=std::max(3,std::min(200,int(25.f/std::max(ms,.001f))));
      if(graph_mode) {
        cudaGraph_t graph;
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        for(int i=0;i<t.reps;++i)t.fn();
        ck(cudaStreamEndCapture(stream,&graph));
        ck(cudaGraphInstantiate(&t.graph,graph,nullptr,nullptr,0));ck(cudaGraphDestroy(graph));
        ck(cudaGraphLaunch(t.graph,stream));ck(cudaStreamSynchronize(stream));
      }
    }
    for(int round=0;round<rounds;++round) {
      std::shuffle(trials.begin(),trials.end(),order);
      for(auto& t:trials) {
        float ms=graph_mode?elapsed([&]{ck(cudaGraphLaunch(t.graph,stream));},1,stream)/t.reps:elapsed(t.fn,t.reps,stream);
        printf("{\"kind\":\"timing\",\"size\":%d,\"variant\":\"%s\",\"stage\":%d,\"round\":%d,\"seed\":%u,\"group\":%d,\"repeats\":%d,\"ms\":%.9g,\"tflops\":%.9g,\"operation\":\"D=A@B\",\"timing\":\"%s\"}\n",
          n,t.name.c_str(),t.stage,round,seed,group,t.reps,ms,2.*n*n*n/(ms*1e9),graph_mode?"cuda_graph":"event");
        fflush(stdout);
      }
    }
    // Post-timing checks cover every measured candidate, including cuBLAS.
    for(auto& t:trials) {t.fn();auto e=verify(x);print_error(t.name,x,e,seed,group,"after");if(!e.pass)return 1;}
    for(auto& t:trials)if(t.graph)ck(cudaGraphExecDestroy(t.graph));
    ck(cudaStreamDestroy(stream));
  }
  return 0;
}
int main(int argc,char **argv) try {
  setvbuf(stdout,nullptr,_IOLBF,0);
  ck(cudaSetDevice(0));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=12 || prop.minor!=0) throw std::runtime_error("G4 track requires sm_120");
  printf("{\"kind\":\"device\",\"name\":\"%s\",\"sm\":120,\"sms\":%d,\"smem_limit\":%zu,\"memory_bytes\":%zu}\n",prop.name,prop.multiProcessorCount,prop.sharedMemPerBlockOptin,prop.totalGlobalMem);
  const std::string mode=argc>1?argv[1]:"bench";
  if(mode=="list") {
    for(const auto& v:variants()) printf("{\"kind\":\"variant\",\"name\":\"%s\",\"stage\":%d,\"bm\":%d,\"bn\":%d,\"bk\":%d,\"smem\":%d,\"threads\":%d,\"supported\":%s}\n",v.name.c_str(),v.stage,v.bm,v.bn,v.bk,v.smem,v.threads,v.smem<=prop.sharedMemPerBlockOptin?"true":"false");
    return 0;
  }
  cublasHandle_t handle;cb(cublasCreate(&handle));int result=0;
  if(mode=="check") result=checks(prop.sharedMemPerBlockOptin,handle);
  else if(mode=="bench"||mode=="tune") {
    int size=argc>2?std::stoi(argv[2]):(mode=="tune"?4096:0);
    int rounds=argc>3?std::stoi(argv[3]):5;
    unsigned seed=argc>4?std::stoul(argv[4]):20261017;
    if((size && (size<128||size>8192||(size&(size-1)))) || rounds<1)throw std::invalid_argument("size must be power of two 128..8192 or 0; rounds >=1");
    result=benchmark(size,rounds,seed,prop.sharedMemPerBlockOptin,handle,mode=="tune");
  } else if(mode=="profile") {
    if(argc<4)throw std::invalid_argument("profile VARIANT SIZE [GROUP]");
    int n=std::stoi(argv[3]),group=argc>4?std::stoi(argv[4]):8;
    Buffers x(n,n,n,20261017);auto p=x.problem();
    if(std::string(argv[2])=="cublas") {
      Library lib(p);for(int i=0;i<5;++i)lib.launch(p);ck(cudaDeviceSynchronize());
      ck(cudaProfilerStart());lib.launch(p);ck(cudaDeviceSynchronize());ck(cudaProfilerStop());
    }
    else {
      bool found=false;
      for(const auto& v:variants())if(v.name==argv[2]) {
        for(int i=0;i<5;++i)run(v,p,nullptr,group);ck(cudaDeviceSynchronize());
        ck(cudaProfilerStart());run(v,p,nullptr,group);ck(cudaDeviceSynchronize());ck(cudaProfilerStop());found=true;
      }
      if(!found)throw std::invalid_argument("unknown variant");
      ck(cudaDeviceSynchronize());
    }
  } else throw std::invalid_argument("usage: bf16_gemm {list|check|bench [SIZE|0 ROUNDS SEED]|tune [SIZE|0 ROUNDS SEED]|profile VARIANT SIZE [GROUP]}");
  cb(cublasDestroy(handle));return result;
} catch(const std::exception& e) {fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
