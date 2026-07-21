#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm128x256_epilogue_ablation}
mkdir -p "$out_dir/bin" "$out_dir/csv"
cd "$src_dir"

nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=1 -DGEMM_REPEAT_TUNING=1
  -DGEMM_PERSISTENT_CTA=1
  -DGEMM_CTA_M=128 -DGEMM_STAGE_K=128 -DGEMM_STAGES=2
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

# E0: historical-style serialized SMEM -> TMA epilogue.
build e0_tma_serial -DGEMM_EPILOGUE_MODE=0
# E1: direct-store control without overlap.
build e1_direct_serial -DGEMM_EPILOGUE_MODE=1
# E2: the previous tile drains through a dedicated epilogue warpgroup while
# the next tile computes in the other half of TMEM.  Four warps are required:
# warpgroup-local warp IDs 0..3 exclusively access TMEM lanes 0..127.
build e2_overlap_4w -DGEMM_EPILOGUE_MODE=2 -DGEMM_EPILOGUE_WARPS=4

nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"
sha256sum "$out_dir"/bin/* > "$out_dir/binary_sha256.txt"

variants=(e0_tma_serial e1_direct_serial e2_overlap_4w)
for variant in "${variants[@]}"; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 1 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

# One size per process.  Rotate order to distribute clock/temperature drift.
orders=(
  "e0_tma_serial e1_direct_serial e2_overlap_4w"
  "e2_overlap_4w e1_direct_serial e0_tma_serial"
  "e1_direct_serial e0_tma_serial e2_overlap_4w"
)
: > "$out_dir/raw.log"
for pass_idx in 1 2 3; do
  read -r -a pass_variants <<< "${orders[$((pass_idx - 1))]}"
  for size in 8192 16384 32768; do
    for variant in "${pass_variants[@]}"; do
      csv="$out_dir/csv/${variant}_${size}_p${pass_idx}.csv"
      echo "pass=$pass_idx size=$size variant=$variant" | tee -a "$out_dir/raw.log"
      "$out_dir/bin/$variant" --sizes "$size" --warmup 1 --iters 5 \
        --input-init random --persistent-ctas 148 --csv "$csv" \
        2>&1 | tee -a "$out_dir/raw.log"
    done
  done
done
nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
