#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_signed8_compare}
cutlass_8k_bin=${CUTLASS_8K_BIN:-/workspace/gemm_compare_bins/cutlass_bf16_streamk_bench}
cutlass_clc_bin=${CUTLASS_CLC_BIN:-/workspace/gemm_compare_bins/cutlass_bf16_best_clc_bench}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}

base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_size_port.py"
cublas_src="$repo_dir/6.cuBLAS/cublas_gemm_bench.cu"
cublas_bin="$out_dir/bin/cublas_gemm_bench"

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/comparison_raw.log") 2>&1

for path in "$base_src" "$generator" "$cublas_src" \
  "$cutlass_8k_bin" "$cutlass_clc_bin"; do
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
done

build_ours() {
  local size=$1 macro_m=$2 macro_n=$3
  local name="ours_${macro_m}x${macro_n}_${size}"
  local source="$out_dir/source/${name}.cu"
  python3 "$generator" --base "$base_src" --output "$source" \
    --size "$size" --macro-m "$macro_m" --macro-n "$macro_n"
  "$nvcc_bin" -O3 -std=c++17 \
    -gencode arch=compute_100a,code=sm_100a \
    "$source" -lcuda -o "$out_dir/bin/$name"
}

for size in 8192 32768; do
  build_ours "$size" 16 16
  build_ours "$size" 12 12
  build_ours "$size" 8 18
done
build_ours 16384 16 16

"$nvcc_bin" -O3 -std=c++17 "$cublas_src" -lcublas -o "$cublas_bin"
cp -- "$base_src" "$generator" "$cublas_src" "$0" "$out_dir/source/"

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_ours_16x16() {
  local size=$1 pass=$2
  "$out_dir/bin/ours_16x16_${size}" \
    --warmup 1 --iters 5 --input-init random-signed8 \
    --csv "$out_dir/csv/ours_16x16_${size}_p${pass}.csv"
}

run_ours_12x12() {
  local size=$1 pass=$2
  "$out_dir/bin/ours_12x12_${size}" \
    --warmup 1 --iters 5 --input-init random-signed8 \
    --csv "$out_dir/csv/ours_12x12_${size}_p${pass}.csv"
}

run_ours_8x18() {
  local size=$1 pass=$2
  "$out_dir/bin/ours_8x18_${size}" \
    --warmup 1 --iters 5 --input-init random-signed8 \
    --csv "$out_dir/csv/ours_8x18_${size}_p${pass}.csv"
}

run_cublas() {
  local size=$1
  "$cublas_bin" --device 0 --m "$size" --n "$size" --k "$size" \
    --warmup 1 --repeat 5 --mode bf16fp32 --input-dist signed8
}

run_cutlass() {
  local size=$1
  if [[ "$size" == 8192 ]]; then
    "$cutlass_8k_bin" --m="$size" --n="$size" --k="$size" \
      --warmup=1 --iterations=5 --input-dist=signed8 \
      --cluster_m=2 --cluster_n=1 \
      --decomposition=Heuristic --reduction=Deterministic
  else
    "$cutlass_clc_bin" --m="$size" --n="$size" --k="$size" \
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

echo "PROTOCOL recent_nsplit=true one_case_per_process=true warmup=1 timed=5 processes=3 input=BF16_uniform_[-8,8)"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
snapshot_gpu "$out_dir/logs/gpu_start.csv"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"
{
  sha256sum "$base_src" "$cublas_src" "$out_dir"/bin/*
  sha256sum "$cutlass_8k_bin" "$cutlass_clc_bin"
} | tee "$out_dir/SHA256SUMS"

for size in 8192 32768; do
  for macro in 16x16 12x12 8x18; do
    "$out_dir/bin/ours_${macro}_${size}" \
      --validate --validate-size 512 --validate-pattern pattern \
      2>&1 | tee "$out_dir/logs/validate_${macro}_${size}_pattern.log"
    "$out_dir/bin/ours_${macro}_${size}" \
      --validate --validate-size 512 --validate-pattern ones \
      2>&1 | tee "$out_dir/logs/validate_${macro}_${size}_ones.log"
  done
done
for pattern in pattern ones; do
  "$out_dir/bin/ours_16x16_16384" \
    --validate --validate-size 512 --validate-pattern "$pattern" \
    2>&1 | tee "$out_dir/logs/validate_16x16_16384_${pattern}.log"
done

methods_8k=(ours_16x16 ours_12x12 ours_8x18 cublas cutlass)
methods_16k=(ours_16x16 cublas cutlass)
methods_32k=(ours_16x16 ours_12x12 ours_8x18 cublas cutlass)
printf 'position\tpass\tsize\tmethod\n' >"$out_dir/sequence.tsv"

run_group() {
  local pass=$1 size=$2
  shift 2
  local position=0 method
  for method in "$@"; do
    position=$((position + 1))
    printf '%d\t%d\t%d\t%s\n' "$position" "$pass" "$size" "$method" \
      >>"$out_dir/sequence.tsv"
    run_case "$method" "$size" "$pass"
  done
}

run_group 1 8192 "${methods_8k[@]}"
run_group 1 16384 "${methods_16k[@]}"
run_group 1 32768 "${methods_32k[@]}"
run_group 2 32768 cutlass cublas ours_8x18 ours_12x12 ours_16x16
run_group 2 16384 cutlass cublas ours_16x16
run_group 2 8192 cutlass cublas ours_8x18 ours_12x12 ours_16x16
run_group 3 16384 cublas ours_16x16 cutlass
run_group 3 32768 ours_12x12 cutlass ours_16x16 cublas ours_8x18
run_group 3 8192 ours_12x12 cutlass ours_16x16 cublas ours_8x18

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$repo_dir/5.GEMM/summarize_gemm_nsplit_signed8_compare.py" "$out_dir"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
