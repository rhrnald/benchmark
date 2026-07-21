#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm128x256_broadcast_ablation}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=1 -DGEMM_REPEAT_TUNING=1
  -DGEMM_PERSISTENT_CTA=1
  -DGEMM_CTA_M=128 -DGEMM_STAGE_K=128
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16
  -DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1
)

build() {
  local name=$1
  shift
  "$nvcc_bin" "${common[@]}" "$@" gemm256_tma_tcgen05_bench.cu \
    -lcuda -o "$out_dir/bin/$name"
}

# B0: current integrated baseline: distinct left/right B panels, two stages.
build b0_distinct_s2 -DGEMM_STAGES=2 -DGEMM_REPEAT_B_BROADCAST=0
# B1: isolate B-panel reuse while retaining two stages.
build b1_broadcast_s2 -DGEMM_STAGES=2 -DGEMM_REPEAT_B_BROADCAST=1
# B2: use the reclaimed 32 KiB/stage to restore the historical depth of three.
build b2_broadcast_s3 -DGEMM_STAGES=3 -DGEMM_REPEAT_B_BROADCAST=1

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/*.cu > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

variants=(b0_distinct_s2 b1_broadcast_s2 b2_broadcast_s3)
for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 1 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

orders=(
  "b0_distinct_s2 b1_broadcast_s2 b2_broadcast_s3"
  "b2_broadcast_s3 b1_broadcast_s2 b0_distinct_s2"
  "b1_broadcast_s2 b0_distinct_s2 b2_broadcast_s3"
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
