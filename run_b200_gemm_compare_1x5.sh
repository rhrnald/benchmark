#!/usr/bin/env bash
set -euo pipefail

BIN_DIR=${BIN_DIR:-/root/gemm_compare}
OUT_DIR=${OUT_DIR:-/root/gemm_compare/results}
mkdir -p "$OUT_DIR"

run_custom() {
  local dist=$1 size=$2 pass=$3
  local init=random
  [[ $dist == signed8 ]] && init=random-signed8
  echo "BEGIN method=custom dist=$dist size=$size pass=$pass"
  nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader,nounits
  "$BIN_DIR/gemm256_tma_tcgen05_persistent" \
    --sizes "$size" --warmup 1 --iters 5 --input-init "$init" \
    --persistent-ctas 148 \
    --csv "$OUT_DIR/custom_${dist}_${size}_p${pass}.csv"
  echo "END method=custom dist=$dist size=$size pass=$pass"
}

run_cublas() {
  local dist=$1 size=$2 pass=$3
  echo "BEGIN method=cublas dist=$dist size=$size pass=$pass"
  nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader,nounits
  "$BIN_DIR/cublas_gemm_bench" \
    --m "$size" --n "$size" --k "$size" \
    --warmup 1 --repeat 5 --mode bf16fp32 --input-dist "$dist"
  echo "END method=cublas dist=$dist size=$size pass=$pass"
}

run_cutlass() {
  local dist=$1 size=$2 pass=$3
  local binary=cutlass_bf16_best_clc_bench
  local extra=()
  if [[ $size == 8192 ]]; then
    binary=cutlass_bf16_streamk_bench
    extra=(--cluster_m=2 --cluster_n=1 \
           --decomposition=Heuristic --reduction=Deterministic)
  fi
  echo "BEGIN method=cutlass dist=$dist size=$size pass=$pass"
  nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader,nounits
  "$BIN_DIR/$binary" \
    --m="$size" --n="$size" --k="$size" \
    --warmup=1 --iterations=5 --input-dist="$dist" "${extra[@]}"
  echo "END method=cutlass dist=$dist size=$size pass=$pass"
}

run_case() {
  local method=$1 dist=$2 size=$3 pass=$4
  "run_${method}" "$dist" "$size" "$pass"
}

echo "PROTOCOL warmup=1 timed=5 one_case_per_process=true"
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total \
  --format=csv,noheader,nounits

# Rotate method, size, and distribution order to reduce ordering bias.  Every
# run_case invocation launches a fresh benchmark process.
for size in 8192 16384 32768; do
  for dist in unit signed8; do
    for method in custom cublas cutlass; do
      run_case "$method" "$dist" "$size" 1
    done
  done
done

for size in 32768 16384 8192; do
  for dist in signed8 unit; do
    for method in cutlass custom cublas; do
      run_case "$method" "$dist" "$size" 2
    done
  done
done

for size in 16384 32768 8192; do
  for dist in unit signed8; do
    for method in cublas cutlass custom; do
      run_case "$method" "$dist" "$size" 3
    done
  done
done

nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm \
  --format=csv,noheader,nounits
