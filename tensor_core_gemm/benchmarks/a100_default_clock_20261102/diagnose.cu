// Default-clock paired benchmark. Link with the repository's unchanged runner.o.
#include "runner.cuh"
#include "kernels/common.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

__global__ void quantize_inputs(const float *in, float *out, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __uint_as_float(f32_to_tf32(in[i]));
}
static long long now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
static void initialize(std::vector<float> &v, std::mt19937 &rng) {
  for (float &x : v) {
    unsigned bits = rng();
    x = float(bits % 5) + 0.01f * float((bits >> 8) % 5);
    if (bits & 0x80000000u) x = -x;
  }
}
int main() {
  cudaCheck(cudaSetDevice(0));
  cudaDeviceProp props{};
  cudaCheck(cudaGetDeviceProperties(&props,0));
  if (props.major != 8 || props.minor != 0) return 2;
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaEvent_t begin,end;
  cudaCheck(cudaEventCreate(&begin)); cudaCheck(cudaEventCreate(&end));
  std::mt19937 rng(20261016), schedule(7);
  constexpr float alpha=0.5f, beta=3.0f;
  constexpr int repeats=1, rounds=1;
  for (int size : {4096}) {
    size_t count=size_t(size)*size, bytes=count*sizeof(float);
    std::vector<float> a(count),b(count),c(count),ref(count),quant_ref(count),out(count);
    initialize(a,rng); initialize(b,rng); initialize(c,rng);
    float *da,*db,*dc,*dr,*daq,*dbq;
    cudaCheck(cudaMalloc(&da,bytes)); cudaCheck(cudaMalloc(&db,bytes));
    cudaCheck(cudaMalloc(&dc,bytes)); cudaCheck(cudaMalloc(&dr,bytes));
    cudaCheck(cudaMalloc(&daq,bytes)); cudaCheck(cudaMalloc(&dbq,bytes));
    cudaCheck(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dr,c.data(),bytes,cudaMemcpyHostToDevice));
    run_kernel(0,size,size,size,alpha,da,db,beta,dr,handle);
    cudaCheck(cudaMemcpy(ref.data(),dr,bytes,cudaMemcpyDeviceToHost));
    quantize_inputs<<<(count+255)/256,256>>>(da,daq,count);
    quantize_inputs<<<(count+255)/256,256>>>(db,dbq,count);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(dr,c.data(),bytes,cudaMemcpyHostToDevice));
    run_kernel(0,size,size,size,alpha,daq,dbq,beta,dr,handle);
    cudaCheck(cudaMemcpy(quant_ref.data(),dr,bytes,cudaMemcpyDeviceToHost));
    for (int round=0;round<rounds;++round) {
      std::vector<int> kernels{7,8,9,10,100};
      std::shuffle(kernels.begin(),kernels.end(),schedule);
      for (int kernel : kernels) {
        cudaCheck(cudaMemcpy(dc,c.data(),bytes,cudaMemcpyHostToDevice));
        auto run=[&](){run_kernel(kernel,size,size,size,alpha,da,db,beta,dc,handle);};
        run(); cudaCheck(cudaGetLastError()); cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out.data(),dc,bytes,cudaMemcpyDeviceToHost));
        auto fp32=verify_matrix(ref.data(),out.data(),count,1e-3);
        auto tf32=verify_matrix(quant_ref.data(),out.data(),count,1e-5);
        if (!fp32.ok) {
          fprintf(stderr,"FAIL size=%d round=%d kernel=%d fp32=%g tf32=%g\n",
                  size,round,kernel,fp32.relErr,tf32.relErr);
          return 4;
        }
        if (kernel==10) {
          cudaCheck(cudaMemcpy(dc,c.data(),bytes,cudaMemcpyHostToDevice));
          configure_frag_pipe({});
          run_configured_frag_pipe(size,size,size,alpha,da,db,beta,dc);
          cudaCheck(cudaDeviceSynchronize());
          std::vector<float> legacy(count);
          cudaCheck(cudaMemcpy(legacy.data(),dc,bytes,cudaMemcpyDeviceToHost));
          auto split=verify_matrix(legacy.data(),out.data(),count,1e-12);
          auto old_tf=verify_matrix(quant_ref.data(),legacy.data(),count,1e-5);
          size_t worst=0;
          for(size_t i=1;i<count;++i)
            if(std::abs(double(out[i])-quant_ref[i])>std::abs(double(out[worst])-quant_ref[worst])) worst=i;
          cudaCheck(cudaMemcpy(a.data(),daq,bytes,cudaMemcpyDeviceToHost));
          cudaCheck(cudaMemcpy(b.data(),dbq,bytes,cudaMemcpyDeviceToHost));
          int row=worst/size,col=worst%size;
          double sum=0;
          for(int k=0;k<size;++k) sum+=double(a[size_t(row)*size+k])*double(b[size_t(k)*size+col]);
          double exact=double(alpha)*sum+double(beta)*c[worst];
          printf("DIAGNOSTIC split_vs_original=%g original_vs_quantized=%g row=%d col=%d split=%.12g original=%.12g fp32_quantized=%.12g fp64_quantized=%.12g split_abs_error=%.12g reference_abs_error=%.12g\n",
                 split.relErr,old_tf.relErr,row,col,double(out[worst]),double(legacy[worst]),double(quant_ref[worst]),exact,
                 std::abs(double(out[worst])-exact),std::abs(double(quant_ref[worst])-exact));
        }
        for(int w=0;w<5;++w) run();
        cudaCheck(cudaMemcpy(dc,c.data(),bytes,cudaMemcpyHostToDevice));
        long long start=now_us();
        cudaCheck(cudaEventRecord(begin));
        for(int r=0;r<repeats;++r) run();
        cudaCheck(cudaEventRecord(end)); cudaCheck(cudaEventSynchronize(end));
        long long stop=now_us();
        cudaCheck(cudaGetLastError());
        float ms; cudaCheck(cudaEventElapsedTime(&ms,begin,end));
        printf("{\"size\":%d,\"round\":%d,\"kernel\":%d,\"ms\":%.9f,\"tflops\":%.9f,"
               "\"fp32_error\":%.12g,\"quantized_tf32_error\":%.12g,\"pass\":true,"
               "\"start_us\":%lld,\"stop_us\":%lld}\n",
               size,round,kernel,ms/repeats,2.0*size*size*size/(ms/repeats)*1e-9,
               fp32.relErr,tf32.relErr,start,stop);
        fflush(stdout);
      }
    }
    for(float *p:{da,db,dc,dr,daq,dbq}) cudaCheck(cudaFree(p));
  }
  cudaCheck(cudaEventDestroy(begin)); cudaCheck(cudaEventDestroy(end));
  cublasDestroy(handle);
}
