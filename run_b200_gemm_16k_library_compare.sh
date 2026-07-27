#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_16k_library_compare}
cutlass_bin=${CUTLASS_BIN:-/root/gemm_compare/cutlass_bf16_best_clc_bench}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

ours_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
cublas_src="$repo_dir/6.cuBLAS/cublas_gemm_bench.cu"
summarizer="$repo_dir/5.GEMM/summarize_gemm_16k_library_compare.py"
ours_bin="$out_dir/bin/ours"
cublas_bin="$out_dir/bin/cublas"

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/raw.log") 2>&1

for path in "$ours_src" "$cublas_src" "$summarizer" "$cutlass_bin"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

"$nvcc_bin" -O3 -std=c++17 --resource-usage \
  -gencode arch=compute_100a,code=sm_100a \
  "$ours_src" -lcuda -o "$ours_bin" \
  2>&1 | tee "$out_dir/logs/build_ours.log"
"$nvcc_bin" -O3 -std=c++17 \
  "$cublas_src" -lcublas -o "$cublas_bin" \
  2>&1 | tee "$out_dir/logs/build_cublas.log"

cp -- "$ours_src" "$cublas_src" "$summarizer" "$0" "$out_dir/source/"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
{
  ldd "$cublas_bin"
  cublas_so=$(ldd "$cublas_bin" | awk '$1 ~ /^libcublas.so/ {print $3; exit}')
  cublas_lt_so=$(ldd "$cublas_bin" | awk '$1 ~ /^libcublasLt.so/ {print $3; exit}')
  readlink -f "$cublas_so"
  readlink -f "$cublas_lt_so"
  sha256sum "$(readlink -f "$cublas_so")" "$(readlink -f "$cublas_lt_so")"
} >"$out_dir/logs/cublas_runtime.txt"
{
  sha256sum "$ours_src" "$cublas_src" "$ours_bin" "$cublas_bin" "$cutlass_bin"
} | tee "$out_dir/SHA256SUMS"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"

"$ours_bin" --validate --validate-size 512 --validate-pattern pattern \
  2>&1 | tee "$out_dir/logs/validate_ours_pattern.log"
"$ours_bin" --validate --validate-size 512 --validate-pattern ones \
  2>&1 | tee "$out_dir/logs/validate_ours_ones.log"

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_ours() {
  local input_name=$1 pass=$2
  local init=random
  [[ "$input_name" == signed8 ]] && init=random-signed8
  "$ours_bin" --warmup 1 --iters 5 --input-init "$init" \
    --csv "$out_dir/csv/${input_name}_ours_p${pass}.csv"
}

run_cublas() {
  local input_name=$1
  "$cublas_bin" --device 0 --m 16384 --n 16384 --k 16384 \
    --warmup 1 --repeat 5 --mode bf16fp32 --input-dist "$input_name"
}

run_cutlass() {
  local input_name=$1
  "$cutlass_bin" --m=16384 --n=16384 --k=16384 \
    --warmup=1 --iterations=5 --input-dist="$input_name"
}

run_case() {
  local input_name=$1 method=$2 pass=$3
  echo "BEGIN input=$input_name method=$method pass=$pass"
  snapshot_gpu "$out_dir/logs/gpu_${input_name}_${method}_p${pass}_before.csv"
  "run_$method" "$input_name" "$pass" \
    2>&1 | tee "$out_dir/logs/${input_name}_${method}_p${pass}.log"
  echo "END input=$input_name method=$method pass=$pass"
}

orders=(
  "ours cublas cutlass"
  "cutlass cublas ours"
  "cublas ours cutlass"
  "ours cutlass cublas"
  "cutlass ours cublas"
  "cublas cutlass ours"
)

echo "PROTOCOL size=16384 one_case_per_process=true warmup=1 timed=5 processes=6"
printf 'position\tpass\tinput\tmethod\n' >"$out_dir/sequence.tsv"
for pass in 1 2 3 4 5 6; do
  read -r -a method_order <<<"${orders[$((pass - 1))]}"
  if ((pass % 2 == 1)); then
    input_order=(unit signed8)
  else
    input_order=(signed8 unit)
  fi
  position=0
  for input_name in "${input_order[@]}"; do
    for method in "${method_order[@]}"; do
      position=$((position + 1))
      printf '%d\t%d\t%s\t%s\n' \
        "$position" "$pass" "$input_name" "$method" >>"$out_dir/sequence.tsv"
      run_case "$input_name" "$method" "$pass"
    done
  done
done

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$summarizer" "$out_dir"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
