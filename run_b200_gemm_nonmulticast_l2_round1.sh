#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm_nonmulticast_l2_round1}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
definition_commit=${DEFINITION_COMMIT:-unknown}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_A_INPUT=0 -DGEMM_REPEAT_B_INPUT=0
  -DGEMM_REPEAT_B_BROADCAST=0 -DGEMM_REPEAT_TUNING=1
  -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_TMA_MULTICAST_A=0 -DGEMM_TMA_MULTICAST_B=0
  -DGEMM_PERSISTENT_CTA=1 -DGEMM_PERSISTENT_STATIC_SCHEDULER=0
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16
  -DGEMM_PERSISTENT_32K_MACRO_M=16 -DGEMM_PERSISTENT_32K_MACRO_N=16
)

compile_variant() {
  local name=$1
  shift
  "$nvcc_bin" "${common[@]}" "$@" \
    gemm256_tma_tcgen05_bench.cu -lcuda -o "$out_dir/bin/$name"
}

# Current-source paper ablation for the two scheduling levels.
compile_variant order_mn -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1
compile_variant order_mm -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=0
compile_variant order_nn -DGEMM_PERSISTENT_LOCAL_M_FAST=0 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1
compile_variant order_nm -DGEMM_PERSISTENT_LOCAL_M_FAST=0 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=0

# Preserve local M-fast B cohorts, changing only macro-row boundaries.
compile_variant snake_1 -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1 -DGEMM_PERSISTENT_SNAKE_MODE=1
compile_variant snake_2 -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1 -DGEMM_PERSISTENT_SNAKE_MODE=2

# One dynamic allocation owns S consecutive N outputs with fixed M.
compile_variant strip_2 -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1 -DGEMM_PERSISTENT_N_STRIP=2
compile_variant strip_4 -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1 -DGEMM_PERSISTENT_N_STRIP=4

python3 test_persistent_l2_mapping.py > "$out_dir/mapping_test.log"
cp gemm256_tma_tcgen05_bench.cu test_persistent_l2_mapping.py \
  "$out_dir/source/"
cp NON_MULTICAST_L2_EXPERIMENT_PLAN.md "$out_dir/source/"
cp ../run_b200_gemm_nonmulticast_l2_round1.sh "$out_dir/source/"
printf '%s\n' "$definition_commit" > "$out_dir/definition_commit.txt"
"$nvcc_bin" --version > "$out_dir/nvcc_version.txt"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

variants=(
  order_mn order_mm order_nn order_nm
  snake_1 snake_2 strip_2 strip_4
)

for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 148 \
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
      idx=$(((i + 4) % count))
      order+=("${variants[$idx]}")
    done
  fi
  for variant in "${order[@]}"; do
    echo "pass=$pass_idx size=16384 variant=$variant ctas=148" \
      | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$variant" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas 148 \
      --csv "$out_dir/csv/${variant}_p${pass_idx}.csv" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
