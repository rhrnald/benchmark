#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm256_tma_multicast_b}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_B_BROADCAST=0
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_PERSISTENT_CTA=1
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
  -DGEMM_PERSISTENT_8K_MACRO_M=12 -DGEMM_PERSISTENT_8K_MACRO_N=12
  -DGEMM_PERSISTENT_32K_MACRO_M=12 -DGEMM_PERSISTENT_32K_MACRO_N=12
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1
)

build() {
  local name=$1
  shift
  "$nvcc_bin" "${common[@]}" "$@" gemm256_tma_tcgen05_bench.cu \
    -lcuda -o "$out_dir/bin/$name"
}

# The binaries differ only in cluster launch, paired task allocation, and the
# B-stage transfer/synchronization path.  Both keep 148 CTA workers, K64/S3,
# the selected dense tile schedule, random BF16 [0,1), and full FP32 C stores.
build control_1cta -DGEMM_TMA_MULTICAST_B=0
build multicast_b_2cta -DGEMM_TMA_MULTICAST_B=1

cp gemm256_tma_tcgen05_bench.cu "$out_dir/source/"
cp Makefile "$out_dir/source/"
cp ../run_b200_gemm256_tma_multicast_b.sh "$out_dir/source/"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

"$out_dir/bin/control_1cta" --validate --validate-size 512 \
  --validate-pattern pattern --persistent-ctas 2 \
  > "$out_dir/validate_control_1cta.log" 2>&1
"$out_dir/bin/multicast_b_2cta" --validate --validate-size 512 \
  --validate-pattern pattern --persistent-ctas 2 \
  > "$out_dir/validate_multicast_b_2cta.log" 2>&1

# One process per case, warmup 1 and five timed launches.  Reverse/rotate the
# order across three passes to expose temperature/order effects.
orders=(
  "control_1cta multicast_b_2cta"
  "multicast_b_2cta control_1cta"
  "control_1cta multicast_b_2cta"
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
