#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm_multicast_pipeline_epilogue_ablation}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_B_BROADCAST=0
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_PERSISTENT_CTA=1 -DGEMM_PERSISTENT_STATIC_SCHEDULER=1
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
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

# 256x256 multicast operand, transaction width, and stage-depth ablations.
compile_variant b_split_k64s3 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 \
  -DGEMM_STAGES=3 -DGEMM_TMA_MULTICAST_B=1
compile_variant a_split_k64s3 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 \
  -DGEMM_STAGES=3 -DGEMM_TMA_MULTICAST_A=1
compile_variant b_wide_k64s3 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 \
  -DGEMM_STAGES=3 -DGEMM_TMA_MULTICAST_B=1 -DGEMM_WIDE_B_TMA=1
compile_variant b_split_k32s4 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=32 \
  -DGEMM_STAGES=4 -DGEMM_TMA_MULTICAST_B=1
compile_variant b_split_k32s5 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=32 \
  -DGEMM_STAGES=5 -DGEMM_TMA_MULTICAST_B=1
compile_variant a_split_k32s4 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=32 \
  -DGEMM_STAGES=4 -DGEMM_TMA_MULTICAST_A=1
compile_variant a_split_k32s5 -DGEMM_CTA_M=256 -DGEMM_STAGE_K=32 \
  -DGEMM_STAGES=5 -DGEMM_TMA_MULTICAST_A=1

# Dense-address 128x256 controls for the existing persistent TMEM ping-pong
# epilogue. E0 is the serialized SMEM/TMA store; E2 overlaps a dedicated
# four-warp direct-store warpgroup with the next output tile.
compile_variant epilogue_e0_m128 -DGEMM_CTA_M=128 -DGEMM_STAGE_K=64 \
  -DGEMM_STAGES=3 -DGEMM_EPILOGUE_MODE=0
compile_variant epilogue_e2_m128 -DGEMM_CTA_M=128 -DGEMM_STAGE_K=64 \
  -DGEMM_STAGES=3 -DGEMM_EPILOGUE_MODE=2 -DGEMM_EPILOGUE_WARPS=4

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
cp ../run_b200_gemm_multicast_pipeline_epilogue_ablation.sh "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

variants=(
  b_split_k64s3 a_split_k64s3 b_wide_k64s3
  b_split_k32s4 b_split_k32s5 a_split_k32s4 a_split_k32s5
  epilogue_e0_m128 epilogue_e2_m128
)

for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 2 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

# One process per case, W1/I5. Reverse and rotate the order to prevent a fixed
# clock/temperature ordering from favoring one variant.
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
    echo "pass=$pass_idx size=16384 variant=$variant" | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$variant" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas 148 \
      --csv "$out_dir/csv/${variant}_p${pass_idx}.csv" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
