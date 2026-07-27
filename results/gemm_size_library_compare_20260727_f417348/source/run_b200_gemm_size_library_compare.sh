#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_size_library_compare}
cutlass_8k_bin=${CUTLASS_8K_BIN:-/workspace/gemm_compare_bins/cutlass_bf16_streamk_bench}
cutlass_clc_bin=${CUTLASS_CLC_BIN:-/workspace/gemm_compare_bins/cutlass_bf16_best_clc_bench}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_size_port.py"
cublas_src="$repo_dir/6.cuBLAS/cublas_gemm_bench.cu"
summarizer="$repo_dir/5.GEMM/summarize_gemm_size_library_compare.py"
cublas_bin="$out_dir/bin/cublas"

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/raw.log") 2>&1

for path in "$base_src" "$generator" "$cublas_src" "$summarizer" \
  "$cutlass_8k_bin" "$cutlass_clc_bin"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

for size in 8192 16384 32768; do
  source="$out_dir/source/ours_${size}.cu"
  python3 "$generator" --base "$base_src" --output "$source" \
    --size "$size" --macro-m 8 --macro-n 16 --scheduler static
  "$nvcc_bin" -O3 -std=c++17 --resource-usage \
    -gencode arch=compute_100a,code=sm_100a \
    "$source" -lcuda -o "$out_dir/bin/ours_${size}" \
    2>&1 | tee "$out_dir/logs/build_ours_${size}.log"
done
"$nvcc_bin" -O3 -std=c++17 "$cublas_src" -lcublas -o "$cublas_bin" \
  2>&1 | tee "$out_dir/logs/build_cublas.log"

cp -- "$base_src" "$generator" "$cublas_src" "$summarizer" "$0" \
  "$out_dir/source/"
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
  sha256sum "$base_src" "$cublas_src" "$out_dir"/source/ours_*.cu
  sha256sum "$out_dir"/bin/* "$cutlass_8k_bin" "$cutlass_clc_bin"
} | tee "$out_dir/SHA256SUMS"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"

for size in 8192 16384 32768; do
  for pattern in pattern ones; do
    "$out_dir/bin/ours_${size}" \
      --validate --validate-size 512 --validate-pattern "$pattern" \
      2>&1 | tee "$out_dir/logs/validate_${size}_${pattern}.log"
  done
done

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_ours() {
  local size=$1 input_name=$2 pass=$3
  local init=random
  [[ "$input_name" == signed8 ]] && init=random-signed8
  "$out_dir/bin/ours_${size}" --warmup 1 --iters 5 --input-init "$init" \
    --csv "$out_dir/csv/${size}_${input_name}_ours_p${pass}.csv"
}

run_cublas() {
  local size=$1 input_name=$2
  "$cublas_bin" --device 0 --m "$size" --n "$size" --k "$size" \
    --warmup 1 --repeat 5 --mode bf16fp32 --input-dist "$input_name"
}

run_cutlass() {
  local size=$1 input_name=$2
  if [[ "$size" == 8192 ]]; then
    "$cutlass_8k_bin" --m="$size" --n="$size" --k="$size" \
      --warmup=1 --iterations=5 --input-dist="$input_name" \
      --cluster_m=2 --cluster_n=1 \
      --decomposition=Heuristic --reduction=Deterministic
  else
    "$cutlass_clc_bin" --m="$size" --n="$size" --k="$size" \
      --warmup=1 --iterations=5 --input-dist="$input_name"
  fi
}

run_case() {
  local size=$1 input_name=$2 method=$3 pass=$4
  echo "BEGIN size=$size input=$input_name method=$method pass=$pass"
  snapshot_gpu \
    "$out_dir/logs/gpu_${size}_${input_name}_${method}_p${pass}_before.csv"
  "run_$method" "$size" "$input_name" "$pass" \
    2>&1 | tee "$out_dir/logs/${size}_${input_name}_${method}_p${pass}.log"
  echo "END size=$size input=$input_name method=$method pass=$pass"
}

orders=("ours cublas cutlass" "cublas cutlass ours" "cutlass ours cublas")
echo "PROTOCOL sizes=8192,16384,32768 one_case_per_process=true warmup=1 timed=5 processes=3"
printf 'position\tpass\tsize\tinput\tmethod\n' >"$out_dir/sequence.tsv"
for pass in 1 2 3; do
  read -r -a method_order <<<"${orders[$((pass - 1))]}"
  if [[ "$pass" == 2 ]]; then
    size_order=(32768 16384 8192)
    input_order=(signed8 unit)
  elif [[ "$pass" == 3 ]]; then
    size_order=(16384 8192 32768)
    input_order=(unit signed8)
  else
    size_order=(8192 16384 32768)
    input_order=(unit signed8)
  fi
  position=0
  for size in "${size_order[@]}"; do
    for input_name in "${input_order[@]}"; do
      for method in "${method_order[@]}"; do
        position=$((position + 1))
        printf '%d\t%d\t%d\t%s\t%s\n' \
          "$position" "$pass" "$size" "$input_name" "$method" \
          >>"$out_dir/sequence.tsv"
        run_case "$size" "$input_name" "$method" "$pass"
      done
    done
  done
done

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$summarizer" "$out_dir"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
