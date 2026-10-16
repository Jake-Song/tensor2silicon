// GPU integration checks: rectangular tiles, short pipelines, and alpha/beta semantics.
#include "runner.cuh"
#include "kernels/common.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Use the CUDA TF32 conversion instruction as an independent input-precision
// reference; the kernels under test use bit transforms and different timing.
__global__ void quantize_reference(const float *in, float *out, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __uint_as_float(f32_to_tf32(in[i]));
}

static int checked = 0;
static double worst_fp32 = 0.0, worst_tf32 = 0.0;
static void check_shape(int M, int N, int K, int tile, cublasHandle_t handle,
                        std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a(size_t(M)*K), b(size_t(K)*N), c(size_t(M)*N), ref(c.size()), quantRef(c.size()), out(c.size());
  for (auto *v : {&a, &b, &c}) for (float &x : *v) x = dist(rng);
  float *da, *db, *dc, *dr, *daq, *dbq;
  cudaCheck(cudaMalloc(&da,a.size()*sizeof(float)));
  cudaCheck(cudaMalloc(&db,b.size()*sizeof(float)));
  cudaCheck(cudaMalloc(&dc,c.size()*sizeof(float)));
  cudaCheck(cudaMalloc(&dr,c.size()*sizeof(float)));
  cudaCheck(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
  cudaCheck(cudaMalloc(&daq,a.size()*sizeof(float)));
  cudaCheck(cudaMalloc(&dbq,b.size()*sizeof(float)));
  quantize_reference<<<(a.size()+255)/256,256>>>(da,daq,a.size());
  quantize_reference<<<(b.size()+255)/256,256>>>(db,dbq,b.size());
  cudaCheck(cudaGetLastError());
  std::vector<FragPipeConfig> configs{{}};
  for (int order=0; order<=6; ++order)
    for (int copy=0; copy<=4; ++copy)
      for (int epi : {1,4,8}) {
        const bool extended = order==3 || order==5 || order==6;
        if (!extended && epi!=1) continue;
        for (int rounding=0; rounding<=(extended ? (epi==4 ? 3 : 1) : 0); ++rounding) {
          if (copy==4 && (order!=6 || epi!=4 || rounding!=0)) continue;
          configs.push_back({tile,order,copy,8,epi,rounding});
        }
      }
  const float params[][2]={{0.0f,1.0f},{1.0f,0.0f},{0.5f,3.0f},{-0.75f,-0.25f}};
  for (const auto &param : params) {
    const float alpha=param[0], beta=param[1];
    cudaCheck(cudaMemcpy(dr,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    run_kernel(0,M,N,K,alpha,da,db,beta,dr,handle);
    cudaCheck(cudaMemcpy(ref.data(),dr,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dr,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
    run_kernel(0,M,N,K,alpha,daq,dbq,beta,dr,handle);
    cudaCheck(cudaMemcpy(quantRef.data(),dr,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for (const auto &config : configs) {
      configure_frag_pipe(config);
      cudaCheck(cudaMemcpy(dc,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      run_kernel(7,M,N,K,alpha,da,db,beta,dc,handle);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(out.data(),dc,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
      const auto result=verify_matrix(ref.data(),out.data(),out.size(),1e-3);
      const auto quantResult=verify_matrix(quantRef.data(),out.data(),out.size(),1e-5);
      worst_fp32=std::max(worst_fp32,result.relErr);
      worst_tf32=std::max(worst_tf32,quantResult.relErr);
      const bool identity = alpha!=0 || beta!=1 || std::equal(out.begin(),out.end(),c.begin());
      if (!result.ok || !quantResult.ok || !identity) {
        fprintf(stderr,"FAIL M=%d N=%d K=%d tile=%d order=%d copy=%d epi=%d round=%d alpha=%g beta=%g rel=%g quant_rel=%g identity=%d\n",
                M,N,K,config.tile,config.order,config.copy,config.epilogue,config.rounding,
                alpha,beta,result.relErr,quantResult.relErr,(int)identity);
        exit(EXIT_FAILURE);
      }
      ++checked;
    }
  }
  cudaCheck(cudaFree(daq));cudaCheck(cudaFree(dbq));
  cudaCheck(cudaFree(da));cudaCheck(cudaFree(db));cudaCheck(cudaFree(dc));cudaCheck(cudaFree(dr));
  printf("PASS M=%d N=%d K=%d tile=%d (%d checks so far)\n",M,N,K,tile,checked);
  fflush(stdout);
}

int main() {
  cudaCheck(cudaSetDevice(getenv("DEVICE") ? atoi(getenv("DEVICE")) : 0));
  printDeviceInfo();
  cublasHandle_t handle;
  if (cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
  std::mt19937 rng(20261017);
  const int shapes[][2]={{128,256},{256,128},{128,128},{128,128},{64,128},{128,64},{64,64}};
  for (int tile=0;tile<7;++tile)
    for (int K : {32,64,96,160})
      check_shape(3*shapes[tile][0],3*shapes[tile][1],K,tile,handle,rng);
  // Rasterization with one complete row group and a partial final group.
  check_shape(9*128,3*256,96,0,handle,rng);
  cublasDestroy(handle);
  printf("All %d rectangular / short-K / alpha-beta checks PASS\n",checked);
  printf("Worst relative error: FP32 reference %.9g; quantized TF32 reference %.9g\n",
         worst_fp32,worst_tf32);
  return EXIT_SUCCESS;
}
