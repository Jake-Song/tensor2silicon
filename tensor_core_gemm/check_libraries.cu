// Integration checks: rectangular strides, alpha/beta, and plan shape changes.
#include "runner.cuh"
#include <cstdio>
#include <random>
#include <vector>

int main() {
  cudaCheck(cudaSetDevice(0));
  printDeviceInfo();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return 1;
  std::mt19937 rng(20261101);
  std::uniform_real_distribution<float> dist(-1, 1);
  int checks = 0;
  const int shapes[][3] = {{128,128,32}, {128,256,64},
                            {256,128,96}, {128,128,32}};
  for (const auto &shape : shapes) {
    const int m=shape[0], n=shape[1], k=shape[2];
    std::vector<float> a(m*k), b(k*n), c(m*n), ref(m*n), out(m*n);
    for (auto v : {&a, &b, &c}) for (auto &x : *v) x=dist(rng);
    float *da, *db, *dc, *dr;
    cudaCheck(cudaMalloc(&da,a.size()*sizeof(float)));
    cudaCheck(cudaMalloc(&db,b.size()*sizeof(float)));
    cudaCheck(cudaMalloc(&dc,c.size()*sizeof(float)));
    cudaCheck(cudaMalloc(&dr,c.size()*sizeof(float)));
    cudaCheck(cudaMemcpy(da,a.data(),a.size()*sizeof(float),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice));
    const float params[][2] = {{1,0}, {0,1}, {0.5f,3}, {-0.75f,-0.25f}};
    for (const auto &param : params) {
      cudaCheck(cudaMemcpy(dr,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
      run_kernel(0,m,n,k,param[0],da,db,param[1],dr,handle);
      cudaCheck(cudaMemcpy(ref.data(),dr,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
      for (int id : {101,102}) {
        if (!is_valid_kernel(id)) continue;
        cudaCheck(cudaMemcpy(dc,c.data(),c.size()*sizeof(float),cudaMemcpyHostToDevice));
        run_kernel(id,m,n,k,param[0],da,db,param[1],dc,handle);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(out.data(),dc,c.size()*sizeof(float),cudaMemcpyDeviceToHost));
        auto r=verify_matrix(ref.data(),out.data(),out.size(),1e-3);
        printf("%s id=%d M=%d N=%d K=%d alpha=%g beta=%g max_abs=%g rel=%g\n",
               r.ok?"PASS":"FAIL",id,m,n,k,param[0],param[1],r.maxAbsDiff,r.relErr);
        if (!r.ok) return 1;
        ++checks;
      }
    }
    cudaCheck(cudaFree(da)); cudaCheck(cudaFree(db));
    cudaCheck(cudaFree(dc)); cudaCheck(cudaFree(dr));
  }
  cublasDestroy(handle);
  printf("%d library checks passed\n",checks);
}
