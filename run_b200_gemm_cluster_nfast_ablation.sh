#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm_cluster_nfast_ablation}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_A_INPUT=0 -DGEMM_REPEAT_B_INPUT=0
  -DGEMM_REPEAT_B_BROADCAST=0 -DGEMM_REPEAT_TUNING=1
  -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_TMA_MULTICAST_B=1 -DGEMM_TMA_MULTICAST_B_CLUSTER_SIZE=2
  -DGEMM_PERSISTENT_CTA=1 -DGEMM_PERSISTENT_STATIC_SCHEDULER=1
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16
  -DGEMM_PERSISTENT_32K_MACRO_M=16 -DGEMM_PERSISTENT_32K_MACRO_N=16
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1
)

compile_variant() {
  local name=$1
  shift
  "$nvcc_bin" "${common[@]}" "$@" \
    gemm256_tma_tcgen05_bench.cu -lcuda -o "$out_dir/bin/$name"
}

# Baseline orders CTAs M-fast.  N-fast variants retain adjacent-M cluster
# ranks, but order cluster IDs across N to make A panels temporally adjacent.
compile_variant baseline_mfast_16x16 \
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
compile_variant nfast_32x8 -DGEMM_PERSISTENT_CLUSTER_N_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_M=32 -DGEMM_PERSISTENT_MACRO_N=8
compile_variant nfast_16x16 -DGEMM_PERSISTENT_CLUSTER_N_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
compile_variant nfast_8x32 -DGEMM_PERSISTENT_CLUSTER_N_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_M=8 -DGEMM_PERSISTENT_MACRO_N=32
compile_variant nfast_4x64 -DGEMM_PERSISTENT_CLUSTER_N_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_M=4 -DGEMM_PERSISTENT_MACRO_N=64
compile_variant nfast_wave16x9 -DGEMM_PERSISTENT_CLUSTER_N_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16 \
  -DGEMM_PERSISTENT_WAVE_M=16 -DGEMM_PERSISTENT_WAVE_N=9

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
cp ../run_b200_gemm_cluster_nfast_ablation.sh "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

variants=(
  baseline_mfast_16x16 nfast_32x8 nfast_16x16
  nfast_8x32 nfast_4x64 nfast_wave16x9
)

for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 2 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

: > "$out_dir/raw.log"
count=${#variants[@]}
for pass_idx in 1 2 3; do
  order=()
  if [[ $pass_idx == 1 ]]; then
    order=("${variants[@]}")
  elif [[ $pass_idx == 2 ]]; then
    for ((i=count-1; i>=0; --i)); do order+=("${variants[$i]}"); done
  else
    for ((i=0; i<count; ++i)); do
      idx=$(((i + 3) % count))
      order+=("${variants[$idx]}")
    done
  fi
  for variant in "${order[@]}"; do
    persistent_ctas=148
    if [[ $variant == nfast_wave16x9 ]]; then persistent_ctas=144; fi
    echo "pass=$pass_idx size=16384 variant=$variant ctas=$persistent_ctas" \
      | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$variant" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas "$persistent_ctas" \
      --csv "$out_dir/csv/${variant}_p${pass_idx}.csv" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
