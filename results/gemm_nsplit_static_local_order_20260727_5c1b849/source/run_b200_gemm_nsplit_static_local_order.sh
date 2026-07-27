#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_static_local_order}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_static_local_order.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_static_local_order.py"
orders=(mfast nfast)
inputs=(random random-signed8)

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/raw.log") 2>&1

for order in "${orders[@]}"; do
  source="$out_dir/source/static_8x16_${order}.cu"
  python3 "$generator" --base "$base_src" --output "$source" --order "$order"
  "$nvcc_bin" -O3 -std=c++17 --resource-usage \
    -gencode arch=compute_100a,code=sm_100a \
    "$source" -lcuda -o "$out_dir/bin/static_8x16_${order}" \
    2>&1 | tee "$out_dir/logs/build_${order}.log"
done

cp -- "$base_src" "$generator" "$summarizer" "$0" "$out_dir/source/"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"
sha256sum "$base_src" "$out_dir"/bin/* | tee "$out_dir/SHA256SUMS"

for order in "${orders[@]}"; do
  for pattern in pattern ones; do
    "$out_dir/bin/static_8x16_${order}" \
      --validate --validate-size 512 --validate-pattern "$pattern" \
      2>&1 | tee "$out_dir/logs/validate_${order}_${pattern}.log"
  done
done

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_case() {
  local pass=$1 input_name=$2 order=$3
  snapshot_gpu "$out_dir/logs/gpu_${input_name}_${order}_p${pass}_before.csv"
  "$out_dir/bin/static_8x16_${order}" \
    --warmup 1 --iters 5 --input-init "$input_name" \
    --csv "$out_dir/csv/${input_name}_${order}_p${pass}.csv" \
    2>&1 | tee "$out_dir/logs/${input_name}_${order}_p${pass}.log"
}

printf 'position\tpass\tinput\torder\n' >"$out_dir/sequence.tsv"
for input_name in "${inputs[@]}"; do
  for pass in 1 2 3 4; do
    if ((pass == 1 || pass == 4)); then
      run_orders=(mfast nfast)
    else
      run_orders=(nfast mfast)
    fi
    position=0
    for order in "${run_orders[@]}"; do
      position=$((position + 1))
      printf '%d\t%d\t%s\t%s\n' \
        "$position" "$pass" "$input_name" "$order" >>"$out_dir/sequence.tsv"
      run_case "$pass" "$input_name" "$order"
    done
  done
done

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$summarizer" "$out_dir"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
