#!/bin/bash
set -euo pipefail
cd /content/tc-library-run
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git cutlass
git -C cutlass rev-parse HEAD > cutlass-sha.txt
cd tensor_core_gemm
make libraries CUTLASS_DIR=/content/tc-library-run/cutlass
make BUILD_DIR=build-libraries NVCCFLAGS='-O3 -std=c++17 -arch=sm_80 -lineinfo -DTC_ENABLE_CUTLASS -I/content/tc-library-run/cutlass/include' build-libraries/tc_check_libraries
./build-libraries/tc_check_libraries | tee ../correctness.log
for id in 100 101 102; do
  ./build-libraries/tc_gemm "$id" 8192 128 | tee "../benchmark_${id}.log"
done
