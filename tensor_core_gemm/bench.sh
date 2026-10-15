#!/usr/bin/env bash
# 모든 커널을 실행해 results/<kernel>_output.txt에 저장하고 크기별 GFLOPS 표를 출력한다.
set -euo pipefail
cd "$(dirname "$0")"

MAX_SIZE=${MAX_SIZE:-8192}
KERNELS=(0 100 1 2 3 4 5 6 7)
mkdir -p results

for k in "${KERNELS[@]}"; do
  echo ""
  ./build/tc_gemm "$k" "$MAX_SIZE" | tee "results/${k}_output.txt" || echo "kernel $k failed"
  sleep 2
done

echo ""
printf "%-8s" "kernel"
for s in 128 256 512 1024 2048 4096 8192; do
  [ "$s" -le "$MAX_SIZE" ] && printf "%10s" "$s"
done
echo "   (GFLOPS)"
for k in "${KERNELS[@]}"; do
  printf "%-8s" "$k"
  grep -oP 'performance: \(\s*\K[0-9.]+' "results/${k}_output.txt" \
    | xargs printf "%10s"
  echo ""
done
