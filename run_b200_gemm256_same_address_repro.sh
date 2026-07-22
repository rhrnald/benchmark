#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm256_same_address_repro}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=1 -DGEMM_REPEAT_TUNING=1
  -DGEMM_DENSE_L2_TUNING=0 -DGEMM_REPEAT_B_BROADCAST=0
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
)

"$nvcc_bin" "${common[@]}" gemm256_tma_tcgen05_bench.cu -lcuda \
  -o "$out_dir/bin/normal"

"$nvcc_bin" "${common[@]}" \
  -DGEMM_PERSISTENT_CTA=1 \
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16 \
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16 \
  -DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18 \
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1 \
  gemm256_tma_tcgen05_bench.cu -lcuda -o "$out_dir/bin/persistent"

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/*.cu > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

for variant in normal persistent; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 1 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

orders=(
  "normal persistent"
  "persistent normal"
  "normal persistent"
)
: > "$out_dir/raw.log"
for pass_idx in 1 2 3; do
  read -r -a pass_variants <<< "${orders[$((pass_idx - 1))]}"
  for size in 8192 16384 32768; do
    for variant in "${pass_variants[@]}"; do
      echo "pass=$pass_idx size=$size variant=$variant" | tee -a "$out_dir/raw.log"
      "$out_dir/bin/$variant" --sizes "$size" --warmup 1 --iters 5 \
        --input-init random --persistent-ctas 148 \
        --csv "$out_dir/csv/${variant}_${size}_p${pass_idx}.csv" \
        2>&1 | tee -a "$out_dir/raw.log"
    done
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
