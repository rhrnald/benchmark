#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_signed8_best_compare}
cutlass_8k_bin=${CUTLASS_8K_BIN:-/root/gemm_compare/cutlass_bf16_streamk_bench}
cutlass_clc_bin=${CUTLASS_CLC_BIN:-/root/gemm_compare/cutlass_bf16_best_clc_bench}

nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
ours_src="$repo_dir/5.GEMM/gemm256_tma_tcgen05_bench.cu"
cublas_src="$repo_dir/6.cuBLAS/cublas_gemm_bench.cu"
ours_bin="$out_dir/bin/ours_best"
cublas_bin="$out_dir/bin/cublas_gemm_bench"

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/comparison_raw.log") 2>&1

for path in "$ours_src" "$cublas_src" "$cutlass_8k_bin" "$cutlass_clc_bin"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

"$nvcc_bin" -O3 -std=c++17 \
  -gencode arch=compute_100a,code=sm_100a \
  -DGEMM_REPEAT_TUNING=1 -DGEMM_DENSE_L2_TUNING=1 \
  -DGEMM_PERSISTENT_CTA=1 \
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16 \
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16 \
  -DGEMM_PERSISTENT_32K_MACRO_M=8 -DGEMM_PERSISTENT_32K_MACRO_N=18 \
  -DGEMM_PERSISTENT_LOCAL_M_FAST=1 -DGEMM_PERSISTENT_MACRO_N_FAST=1 \
  "$ours_src" -lcuda -o "$ours_bin"
"$nvcc_bin" -O3 -std=c++17 \
  "$cublas_src" -lcublas -o "$cublas_bin"

cp -- "$ours_src" "$cublas_src" "$0" "$out_dir/source/"

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_ours() {
  local size=$1 pass=$2
  "$ours_bin" \
    --sizes "$size" --warmup 1 --iters 5 \
    --input-init random-signed8 --persistent-ctas 148 \
    --csv "$out_dir/csv/ours_${size}_p${pass}.csv"
}

run_cublas() {
  local size=$1
  "$cublas_bin" \
    --device 0 --m "$size" --n "$size" --k "$size" \
    --warmup 1 --repeat 5 --mode bf16fp32 --input-dist signed8
}

run_cutlass() {
  local size=$1
  if [[ "$size" == 8192 ]]; then
    "$cutlass_8k_bin" \
      --m="$size" --n="$size" --k="$size" \
      --warmup=1 --iterations=5 --input-dist=signed8 \
      --cluster_m=2 --cluster_n=1 \
      --decomposition=Heuristic --reduction=Deterministic
  else
    "$cutlass_clc_bin" \
      --m="$size" --n="$size" --k="$size" \
      --warmup=1 --iterations=5 --input-dist=signed8
  fi
}

run_case() {
  local method=$1 size=$2 pass=$3
  echo "BEGIN method=$method input=signed8 size=$size pass=$pass"
  snapshot_gpu "$out_dir/logs/gpu_${method}_${size}_p${pass}_before.csv"
  "run_$method" "$size" "$pass" \
    2>&1 | tee "$out_dir/logs/${method}_${size}_p${pass}.log"
  echo "END method=$method input=signed8 size=$size pass=$pass"
}

echo "PROTOCOL one_case_per_process=true warmup=1 timed=5 processes=3 input=BF16_uniform_[-8,8)"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
snapshot_gpu "$out_dir/logs/gpu_start.csv"
{
  sha256sum "$ours_src" "$cublas_src" "$ours_bin" "$cublas_bin"
  sha256sum "$cutlass_8k_bin" "$cutlass_clc_bin"
} | tee "$out_dir/SHA256SUMS"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"

"$ours_bin" --validate --validate-size 512 \
  --validate-pattern pattern --persistent-ctas 148 \
  2>&1 | tee "$out_dir/logs/ours_validate_pattern.log"
"$ours_bin" --validate --validate-size 512 \
  --validate-pattern ones --persistent-ctas 148 \
  2>&1 | tee "$out_dir/logs/ours_validate_ones.log"

printf 'position\tpass\tsize\tmethod\n' >"$out_dir/sequence.tsv"

position=0
for size in 8192 16384 32768; do
  for method in ours cublas cutlass; do
    position=$((position + 1))
    printf '%d\t1\t%d\t%s\n' "$position" "$size" "$method" >>"$out_dir/sequence.tsv"
    run_case "$method" "$size" 1
  done
done

position=0
for size in 32768 16384 8192; do
  for method in cutlass ours cublas; do
    position=$((position + 1))
    printf '%d\t2\t%d\t%s\n' "$position" "$size" "$method" >>"$out_dir/sequence.tsv"
    run_case "$method" "$size" 2
  done
done

position=0
for size in 16384 32768 8192; do
  for method in cublas cutlass ours; do
    position=$((position + 1))
    printf '%d\t3\t%d\t%s\n' "$position" "$size" "$method" >>"$out_dir/sequence.tsv"
    run_case "$method" "$size" 3
  done
done

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$repo_dir/5.GEMM/summarize_gemm_signed8_best_compare.py" "$out_dir"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
