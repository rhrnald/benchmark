#!/usr/bin/env bash
set -euo pipefail

# Run inside the B200 instance after the archived source tree has been copied
# to source_root with sibling 5-1.gemm and 5.GEMM directories.
source_root=${1:-/workspace/historical_1797}
out_dir=${2:-/workspace/historical_1797_remeasure_1x5}
src_dir="$source_root/5-1.gemm"
helper_dir="$source_root/5.GEMM"
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$out_dir/csv" "$out_dir/source"
cd "$src_dir"

"$nvcc_bin" -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a \
  gemm128x256_repeated_tile.cu -lcuda \
  -o "$out_dir/gemm128x256_repeated_tile"

cp gemm128x256_repeated_tile.cu "$out_dir/source/"
cp "$helper_dir/gemm256_tma_tcgen05_bench.cu" "$out_dir/source/"
sha256sum "$out_dir/gemm128x256_repeated_tile" "$out_dir"/source/*.cu \
  > "$out_dir/sha256.txt"
nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"

"$out_dir/gemm128x256_repeated_tile" \
  --blocks 1 --steps 8 --warmup 1 --iters 1 --validate \
  --seed 20260718 --csv "$out_dir/csv/validate.csv" \
  > "$out_dir/validate.log" 2>&1

: > "$out_dir/raw.log"
for pass_idx in 1 2 3; do
  echo "pass=$pass_idx" | tee -a "$out_dir/raw.log"
  "$out_dir/gemm128x256_repeated_tile" \
    --blocks 592 --steps 8192 --warmup 1 --iters 5 \
    --seed 20260718 --csv "$out_dir/csv/pass_${pass_idx}.csv" \
    2>&1 | tee -a "$out_dir/raw.log"
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
