#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm256_multicast_16k_scheduler_sweep}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_B_BROADCAST=0
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_PERSISTENT_CTA=1 -DGEMM_TMA_MULTICAST_B=1
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16
  -DGEMM_PERSISTENT_32K_MACRO_M=16 -DGEMM_PERSISTENT_32K_MACRO_N=16
)

shapes=(8x8 8x16 8x32 16x8 16x16 16x32 32x8 32x16 32x32)
variants=()
for scheduler in dynamic static; do
  static_flag=0
  [[ $scheduler == static ]] && static_flag=1
  for shape in "${shapes[@]}"; do
    macro_m=${shape%x*}
    macro_n=${shape#*x}
    name="${scheduler}_${shape}"
    variants+=("$name")
    "$nvcc_bin" "${common[@]}" \
      -DGEMM_PERSISTENT_STATIC_SCHEDULER="$static_flag" \
      -DGEMM_PERSISTENT_MACRO_M="$macro_m" \
      -DGEMM_PERSISTENT_MACRO_N="$macro_n" \
      gemm256_tma_tcgen05_bench.cu -lcuda -o "$out_dir/bin/$name"
  done
done

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
cp ../run_b200_gemm256_multicast_16k_scheduler_sweep.sh "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 2 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

# One process per case. Pass 2 reverses order; pass 3 rotates by seven places
# so scheduler and macro-shape order cannot consistently benefit from clocks.
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
      idx=$(((i + 7) % count))
      order+=("${variants[$idx]}")
    done
  fi
  for variant in "${order[@]}"; do
    echo "pass=$pass_idx size=16384 variant=$variant" | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$variant" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas 148 \
      --csv "$out_dir/csv/${variant}_p${pass_idx}.csv" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
