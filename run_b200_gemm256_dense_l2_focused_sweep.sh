#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm256_dense_l2_focused_sweep}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_B_BROADCAST=0
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_PERSISTENT_CTA=1
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1
)

build() {
  local name=$1
  shift
  "$nvcc_bin" "${common[@]}" "$@" gemm256_tma_tcgen05_bench.cu \
    -lcuda -o "$out_dir/bin/$name"
}

# Paired control from the matched dense baseline: repeat-tuned phase and 8K
# promotion, 16x16 at 8K, and 8x18 at 32K.
build d0_control \
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=0 \
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16 \
  -DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18

# Dense tuning fixes TMA/MMA phase at 0/0 and disables A/B L2 promotion.
# The three candidates retain the known-good 16x16 shape at 16K and compare
# only balanced 144-tile cache waves at 8K and 32K.
build d1_8x18 \
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 \
  -DGEMM_PERSISTENT_8K_MACRO_M=8 -DGEMM_PERSISTENT_8K_MACRO_N=18 \
  -DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18
build d2_9x16 \
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 \
  -DGEMM_PERSISTENT_8K_MACRO_M=9 -DGEMM_PERSISTENT_8K_MACRO_N=16 \
  -DGEMM_PERSISTENT_32K_MACRO_M=9 -DGEMM_PERSISTENT_32K_MACRO_N=16
build d3_12x12 \
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 \
  -DGEMM_PERSISTENT_8K_MACRO_M=12 -DGEMM_PERSISTENT_8K_MACRO_N=12 \
  -DGEMM_PERSISTENT_32K_MACRO_M=12 -DGEMM_PERSISTENT_32K_MACRO_N=12

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/*.cu > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

variants=(d0_control d1_8x18 d2_9x16 d3_12x12)
for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 1 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

orders=(
  "d0_control d1_8x18 d2_9x16 d3_12x12"
  "d3_12x12 d2_9x16 d1_8x18 d0_control"
  "d1_8x18 d3_12x12 d0_control d2_9x16"
)
: > "$out_dir/raw.log"
for pass_idx in 1 2 3; do
  read -r -a pass_variants <<< "${orders[$((pass_idx - 1))]}"
  for size in 8192 16384 32768; do
    for variant in "${pass_variants[@]}"; do
      # All dense candidates are identical at 16K; run only the paired
      # control and d1 to avoid spending GPU time on duplicate binaries.
      if [[ $size == 16384 && $variant != d0_control && $variant != d1_8x18 ]]; then
        continue
      fi
      echo "pass=$pass_idx size=$size variant=$variant" | tee -a "$out_dir/raw.log"
      "$out_dir/bin/$variant" --sizes "$size" --warmup 1 --iters 5 \
        --input-init random --persistent-ctas 148 \
        --csv "$out_dir/csv/${variant}_${size}_p${pass_idx}.csv" \
        2>&1 | tee -a "$out_dir/raw.log"
    done
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
