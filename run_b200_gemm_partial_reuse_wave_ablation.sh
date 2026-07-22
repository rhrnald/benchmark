#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm_partial_reuse_wave_ablation}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_B_BROADCAST=0
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_TMA_MULTICAST_B=1
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

# Cache-ceiling diagnostic: hold the pipeline and scheduler fixed and vary
# only whether A and/or B select the same source tile for every output tile.
compile_variant c2_dense -DGEMM_REPEAT_A_INPUT=0 -DGEMM_REPEAT_B_INPUT=0
compile_variant c2_repeat_a -DGEMM_REPEAT_A_INPUT=1 -DGEMM_REPEAT_B_INPUT=0
compile_variant c2_repeat_b -DGEMM_REPEAT_A_INPUT=0 -DGEMM_REPEAT_B_INPUT=1
compile_variant c2_repeat_ab -DGEMM_REPEAT_A_INPUT=1 -DGEMM_REPEAT_B_INPUT=1

# Explicit 144-task waves test whether aligning scheduler locality with one
# resident B200 wave is better than the existing clipped 16x16 macroblocks.
compile_variant c2_wave16x9 -DGEMM_PERSISTENT_WAVE_M=16 \
  -DGEMM_PERSISTENT_WAVE_N=9
compile_variant c2_wave12x12 -DGEMM_PERSISTENT_WAVE_M=12 \
  -DGEMM_PERSISTENT_WAVE_N=12
compile_variant c4_wave16x9 -DGEMM_TMA_MULTICAST_B_CLUSTER_SIZE=4 \
  -DGEMM_PERSISTENT_WAVE_M=16 -DGEMM_PERSISTENT_WAVE_N=9
compile_variant c4_wave12x12 -DGEMM_TMA_MULTICAST_B_CLUSTER_SIZE=4 \
  -DGEMM_PERSISTENT_WAVE_M=12 -DGEMM_PERSISTENT_WAVE_N=12

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
cp ../run_b200_gemm_partial_reuse_wave_ablation.sh "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

variants=(
  c2_dense_148 c2_repeat_a c2_repeat_b c2_repeat_ab
  c2_dense_144 c2_wave16x9 c2_wave12x12
  c4_wave16x9 c4_wave12x12
)

binary_for() {
  case "$1" in
    c2_dense_148|c2_dense_144) echo c2_dense ;;
    *) echo "$1" ;;
  esac
}

ctas_for() {
  case "$1" in
    c2_dense_148|c2_repeat_a|c2_repeat_b|c2_repeat_ab) echo 148 ;;
    *) echo 144 ;;
  esac
}

for variant in "${variants[@]}"; do
  binary=$(binary_for "$variant")
  validation_ctas=2
  validation_size=512
  if [[ $variant == c4_* ]]; then
    validation_ctas=4
    # Four multicast ranks must take the same valid/padded branch.  M=1024
    # gives four 256-row tiles; M=512 would leave half the cluster out.
    validation_size=1024
  fi
  "$out_dir/bin/$binary" --validate --validate-size "$validation_size" \
    --validate-pattern pattern --persistent-ctas "$validation_ctas" \
    > "$out_dir/validate_${variant}.log" 2>&1
done

# One process per case, W1/I5.  Reverse and rotate order across three passes.
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
    binary=$(binary_for "$variant")
    persistent_ctas=$(ctas_for "$variant")
    echo "pass=$pass_idx size=16384 variant=$variant ctas=$persistent_ctas" \
      | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$binary" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas "$persistent_ctas" \
      --csv "$out_dir/csv/${variant}_p${pass_idx}.csv" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
