// Reuse inputs and allocations across experiments; keep the legacy benchmark intact.
#include "runner.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Variant {
  std::string name;
  int kernel;
  FragPipeConfig config;
};

static void initialize(std::vector<float> &v, std::mt19937 &rng) {
  for (float &x : v) {
    const unsigned bits = rng();
    x = float(bits % 5) + 0.01f * float((bits >> 8) % 5);
    if (bits & 0x80000000u) x = -x;
  }
}

int main(int argc, char **argv) {
  const int maxSize = argc > 1 ? std::stoi(argv[1]) : 8192;
  const int minSize = argc > 2 ? std::stoi(argv[2]) : 128;
  const int rounds = argc > 3 ? std::stoi(argv[3]) : 3;
  if (minSize < 128 || maxSize < minSize || rounds < 1 ||
      (maxSize & (maxSize - 1)) || (minSize & (minSize - 1))) {
    fprintf(stderr, "Usage: %s [max_size power of 2] [min_size power of 2 >=128] [rounds >0]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (getenv("NVIDIA_TF32_OVERRIDE") && std::string(getenv("NVIDIA_TF32_OVERRIDE")) == "0") {
    fprintf(stderr, "TF32 is disabled by NVIDIA_TF32_OVERRIDE\n");
    return EXIT_FAILURE;
  }
  cudaCheck(cudaSetDevice(getenv("DEVICE") ? atoi(getenv("DEVICE")) : 0));
  printDeviceInfo();
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
  cudaEvent_t begin, end;
  cudaCheck(cudaEventCreate(&begin));
  cudaCheck(cudaEventCreate(&end));
  std::mt19937 rng(20261016), schedule(7);
  constexpr float alpha = 0.5f, beta = 3.0f;
  constexpr int repeats = 50;
  for (int size = minSize; size <= maxSize; size *= 2) {
    const size_t count = size_t(size) * size, bytes = count * sizeof(float);
    std::vector<float> a(count), b(count), c(count), ref(count), out(count);
    initialize(a, rng); initialize(b, rng); initialize(c, rng);
    float *da, *db, *dc, *dr;
    cudaCheck(cudaMalloc(&da, bytes)); cudaCheck(cudaMalloc(&db, bytes));
    cudaCheck(cudaMalloc(&dc, bytes)); cudaCheck(cudaMalloc(&dr, bytes));
    cudaCheck(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dr, c.data(), bytes, cudaMemcpyHostToDevice));
    run_kernel(0, size, size, size, alpha, da, db, beta, dr, handle);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(ref.data(), dr, bytes, cudaMemcpyDeviceToHost));
    std::vector<Variant> variants{{"cublas", 100, {}}, {"default", 7, {}},
        {"legacy", 7, {size >= 4096 ? 1 : 2, 0, 0, 8, 1, 0}}};
    for (int cfg = 0; cfg <= 6; ++cfg)
      for (int order = 0; order <= 6; ++order)
        for (int copy = 0; copy <= 3; ++copy) {
          if (size == 128 && (cfg == 0 || cfg == 1)) continue;
          variants.push_back({"cfg" + std::to_string(cfg) + "-order" +
              std::to_string(order) + "-copy" + std::to_string(copy),
              7, {cfg, order, copy, 8, 1, 0}});
        }
    for (int cfg = 0; cfg <= 6; ++cfg) {
      if (size == 128 && (cfg == 0 || cfg == 1)) continue;
      for (int order : {3, 5, 6})
        for (int copy = 0; copy <= 3; ++copy)
          for (int epi : {4, 8})
            variants.push_back({"cfg" + std::to_string(cfg) + "-order" + std::to_string(order) + "-copy" +
                std::to_string(copy) + "-epi" + std::to_string(epi),
                7, {cfg, order, copy, 8, epi, 0}});
    }
    for (int cfg = 0; cfg <= 6; ++cfg) {
      if (size == 128 && (cfg == 0 || cfg == 1)) continue;
      for (int order : {3, 5, 6})
        for (int copy = 0; copy <= 3; ++copy)
          for (int epi : {1, 4, 8})
            variants.push_back({"cfg" + std::to_string(cfg) + "-order" + std::to_string(order) + "-copy" +
                std::to_string(copy) + (epi == 1 ? "" : "-epi" + std::to_string(epi)) + "-round1",
                7, {cfg, order, copy, 8, epi, 1}});
    }
    for (int cfg = 0; cfg <= 6; ++cfg) {
      if (size == 128 && (cfg == 0 || cfg == 1)) continue;
      for (int order : {3,5,6})
        for (int copy = 0; copy <= 3; ++copy)
          for (int rounding : {2,3})
            variants.push_back({"cfg" + std::to_string(cfg) + "-order" + std::to_string(order) + "-copy" +
                std::to_string(copy) + "-epi4-round" + std::to_string(rounding),
                7, {cfg,order,copy,8,4,rounding}});
    }
    for (int cfg = 0; cfg <= 6; ++cfg) {
      if (size == 128 && (cfg == 0 || cfg == 1)) continue;
      variants.push_back({"cfg" + std::to_string(cfg) + "-order6-copy4-epi4",
                          7, {cfg,6,4,8,4,0}});
    }
    if (const char *selected = getenv("TC_TUNE_VARIANTS")) {
      const std::string names = std::string(",") + selected + ",";
      variants.erase(std::remove_if(variants.begin(), variants.end(), [&](const Variant &v) {
        return v.name.rfind("cfg", 0) == 0 && names.find("," + v.name + ",") == std::string::npos;
      }), variants.end());
    }
    if (const char *groups = getenv("TC_TUNE_GROUPS")) {
      const auto selected = variants;
      std::stringstream input(groups);
      std::string token;
      while (std::getline(input, token, ',')) {
        const int group = std::stoi(token);
        if (group < 0) return EXIT_FAILURE;
        for (auto v : selected) {
          if (v.name.rfind("cfg", 0) != 0) continue;
          v.name += "-group" + std::to_string(group);
          v.config.group = group;
          variants.push_back(v);
        }
      }
    }
    for (int round = 0; round < rounds; ++round) {
      std::shuffle(variants.begin(), variants.end(), schedule);
      for (const auto &v : variants) {
        configure_frag_pipe(v.config);
        cudaCheck(cudaMemcpy(dc, c.data(), bytes, cudaMemcpyHostToDevice));
        run_kernel(v.kernel, size, size, size, alpha, da, db, beta, dc, handle);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(out.data(), dc, bytes, cudaMemcpyDeviceToHost));
        const auto check = verify_matrix(ref.data(), out.data(), count, 1e-3);
        if (!check.ok) {
          fprintf(stderr, "FAIL size=%d variant=%s rel=%.6g\n", size, v.name.c_str(), check.relErr);
          return EXIT_FAILURE;
        }
        // Warmup, then reset C so beta=3 cannot accumulate across trials.
        for (int i = 0; i < 5; ++i)
          run_kernel(v.kernel, size, size, size, alpha, da, db, beta, dc, handle);
        cudaCheck(cudaMemcpy(dc, c.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaEventRecord(begin));
        for (int i = 0; i < repeats; ++i)
          run_kernel(v.kernel, size, size, size, alpha, da, db, beta, dc, handle);
        cudaCheck(cudaEventRecord(end));
        cudaCheck(cudaEventSynchronize(end));
        cudaCheck(cudaGetLastError());
        float ms;
        cudaCheck(cudaEventElapsedTime(&ms, begin, end));
        const double gflops = 2.0 * size * size * size / (ms / repeats) * 1e-6;
        printf("{\"size\":%d,\"round\":%d,\"variant\":\"%s\",\"gflops\":%.6f,"
               "\"rel_error\":%.9g,\"pass\":true}\n", size, round, v.name.c_str(), gflops, check.relErr);
        fflush(stdout);
      }
    }
    cudaCheck(cudaFree(da)); cudaCheck(cudaFree(db));
    cudaCheck(cudaFree(dc)); cudaCheck(cudaFree(dr));
  }
  cudaCheck(cudaEventDestroy(begin)); cudaCheck(cudaEventDestroy(end));
  cublasDestroy(handle);
  return EXIT_SUCCESS;
}
