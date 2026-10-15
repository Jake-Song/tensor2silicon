// A100 TF32 Tensor Core GEMM 벤치마크.
// 구조는 SGEMM_CUDA/sgemm.cu와 같다: 크기별로 cuBLAS FP32 결과와 비교한 뒤 50회 반복 측정한다.
#include "runner.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <kernel> [max_size] [min_size]\n"
                    "  0: cuBLAS FP32, 1-7: Tensor Core kernels, 100: cuBLAS TF32\n",
            argv[0]);
    return EXIT_FAILURE;
  }
  const int kernel_num = std::stoi(argv[1]);
  if (!is_valid_kernel(kernel_num)) {
    fprintf(stderr, "Invalid kernel number %d (0-7 or 100)\n", kernel_num);
    return EXIT_FAILURE;
  }
  const long max_size = argc >= 3 ? std::stol(argv[2]) : 8192;
  const long min_size = argc >= 4 ? std::stol(argv[3]) : 128; // ncu로 한 크기만 볼 때

  int deviceIdx = 0;
  if (getenv("DEVICE") != nullptr)
    deviceIdx = atoi(getenv("DEVICE"));
  cudaCheck(cudaSetDevice(deviceIdx));
  printDeviceInfo();
  printf("Running kernel %d: %s\n", kernel_num, kernel_name(kernel_num));

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "Create cublas handle error.\n");
    return EXIT_FAILURE;
  }

  std::vector<long> sizes;
  for (long s = 128; s <= max_size; s *= 2)
    if (s >= min_size)
      sizes.push_back(s);

  const float alpha = 0.5f, beta = 3.0f; // C = alpha * A @ B + beta * C
  const double tol = 1e-3;               // TF32 상대 오차 허용치
  const int repeat_times = 50;
  const size_t bytes = sizeof(float) * max_size * max_size;

  std::vector<float> A(max_size * max_size), B(max_size * max_size),
      C(max_size * max_size), C_out(max_size * max_size),
      C_ref(max_size * max_size);
  randomize_matrix(A.data(), A.size());
  randomize_matrix(B.data(), B.size());
  randomize_matrix(C.data(), C.size());

  float *dA, *dB, *dC, *dC_ref;
  cudaCheck(cudaMalloc(&dA, bytes));
  cudaCheck(cudaMalloc(&dB, bytes));
  cudaCheck(cudaMalloc(&dC, bytes));
  cudaCheck(cudaMalloc(&dC_ref, bytes));
  cudaCheck(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  bool all_ok = true;
  for (long size : sizes) {
    const int m = size, n = size, k = size;
    const size_t mn_bytes = sizeof(float) * m * n;
    printf("dimensions(m=n=k) %d, alpha: %.1f, beta: %.1f\n", m, alpha, beta);

    // 같은 초기 C에서 기준(cuBLAS FP32)과 대상 커널을 한 번씩 실행해 비교한다.
    cudaCheck(cudaMemcpy(dC, C.data(), mn_bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C.data(), mn_bytes, cudaMemcpyHostToDevice));
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaMemcpy(C_out.data(), dC, mn_bytes, cudaMemcpyDeviceToHost));
      cudaCheck(
          cudaMemcpy(C_ref.data(), dC_ref, mn_bytes, cudaMemcpyDeviceToHost));
      const VerifyResult v =
          verify_matrix(C_ref.data(), C_out.data(), (long)m * n, tol);
      printf("  vs cuBLAS FP32: max|diff| %.4g, max|ref| %.4g, rel %.3e -> %s\n",
             v.maxAbsDiff, v.maxAbsRef, v.relErr, v.ok ? "PASS" : "FAIL");
      if (!v.ok) {
        all_ok = false;
        printf("Failed to pass the correctness verification (tol %.0e).\n",
               tol);
        break;
      }
    }

    // warmup 1회 후 측정. beta != 0이라 반복할수록 C 값이 커지지만 시간에는 영향이 없다.
    run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++)
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    cudaEventRecord(end);
    cudaCheck(cudaEventSynchronize(end));
    cudaCheck(cudaGetLastError());

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, beg, end);
    const double sec = elapsed_ms / 1000.0 / repeat_times;
    const double flops = 2.0 * m * n * k;
    printf("Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. "
           "size: (%d).\n",
           sec, flops * 1e-9 / sec, m);
    fflush(stdout);
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC_ref);
  cublasDestroy(handle);
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
