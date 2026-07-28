#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_tma_only}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
ncu_bin=${NCU:-/usr/local/cuda/bin/ncu}
base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_tma_only.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_tma_only.py"

mkdir -p "$out_dir"/{bin,csv,logs,profiles,source}
exec > >(tee "$out_dir/raw.log") 2>&1

python3 "$generator" --base "$base_src" \
  --output "$out_dir/source/tma_only.cu"
cp -- "$base_src" "$generator" "$summarizer" "$0" "$out_dir/source/"

"$nvcc_bin" -O3 -std=c++17 --resource-usage \
  -gencode arch=compute_100a,code=sm_100a \
  "$base_src" -lcuda -o "$out_dir/bin/gemm" \
  2>&1 | tee "$out_dir/logs/build_gemm.log"
"$nvcc_bin" -O3 -std=c++17 --resource-usage \
  -gencode arch=compute_100a,code=sm_100a \
  "$out_dir/source/tma_only.cu" -lcuda -o "$out_dir/bin/tma_only" \
  2>&1 | tee "$out_dir/logs/build_tma_only.log"

"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
sha256sum "$base_src" "$out_dir/source/tma_only.cu" "$out_dir"/bin/* \
  | tee "$out_dir/SHA256SUMS"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"

"$out_dir/bin/gemm" --validate --validate-size 512 --validate-pattern pattern \
  2>&1 | tee "$out_dir/logs/validate_gemm_pattern.log"
"$out_dir/bin/gemm" --validate --validate-size 512 --validate-pattern ones \
  2>&1 | tee "$out_dir/logs/validate_gemm_ones.log"
"$out_dir/bin/tma_only" --warmup 0 --iters 1 --input-init random \
  --csv "$out_dir/csv/tma_only_smoke.csv" \
  2>&1 | tee "$out_dir/logs/tma_only_smoke.log"

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_case() {
  local input_name=$1 method=$2 pass=$3
  echo "BEGIN input=$input_name method=$method pass=$pass"
  snapshot_gpu "$out_dir/logs/gpu_${input_name}_${method}_p${pass}_before.csv"
  "$out_dir/bin/$method" --warmup 1 --iters 5 --input-init "$input_name" \
    --csv "$out_dir/csv/${input_name}_${method}_p${pass}.csv" \
    2>&1 | tee "$out_dir/logs/${input_name}_${method}_p${pass}.log"
  echo "END input=$input_name method=$method pass=$pass"
}

echo "PROTOCOL size=16384 methods=gemm,tma_only warmup=1 timed=5 processes=4"
printf 'position\tpass\tinput\tmethod\n' >"$out_dir/sequence.tsv"
for pass in 1 2 3 4; do
  if ((pass == 1 || pass == 4)); then
    methods=(gemm tma_only)
    inputs=(random random-signed8)
  else
    methods=(tma_only gemm)
    inputs=(random-signed8 random)
  fi
  position=0
  for input_name in "${inputs[@]}"; do
    for method in "${methods[@]}"; do
      position=$((position + 1))
      printf '%d\t%d\t%s\t%s\n' \
        "$position" "$pass" "$input_name" "$method" >>"$out_dir/sequence.tsv"
      run_case "$input_name" "$method" "$pass"
    done
  done
done

python3 "$summarizer" "$out_dir"

if [[ -x "$ncu_bin" ]]; then
  "$ncu_bin" --version >"$out_dir/logs/ncu_version.txt"
  for method in gemm tma_only; do
    kernel_regex="regex:.*gemm256_bf16_16k_kernel.*"
    [[ "$method" == tma_only ]] && kernel_regex="regex:.*tma_load_only_kernel.*"
    "$ncu_bin" --target-processes all \
      --kernel-name-base demangled --kernel-name "$kernel_regex" \
      --launch-count 1 --section MemoryWorkloadAnalysis \
      --export "$out_dir/profiles/${method}" --force-overwrite \
      "$out_dir/bin/$method" --warmup 0 --iters 1 --input-init random \
      --csv "$out_dir/csv/profile_${method}.csv" \
      >"$out_dir/logs/ncu_${method}.log" 2>&1 || true
    if [[ -f "$out_dir/profiles/${method}.ncu-rep" ]]; then
      "$ncu_bin" --import "$out_dir/profiles/${method}.ncu-rep" \
        --page raw --csv >"$out_dir/profiles/${method}_raw.csv" 2>&1 || true
    fi
  done
fi

snapshot_gpu "$out_dir/logs/gpu_end.csv"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
