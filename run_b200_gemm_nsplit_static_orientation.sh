#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_static_orientation}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_size_port.py"
variants=(8x16 16x8)

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/raw.log") 2>&1

for variant in "${variants[@]}"; do
  IFS=x read -r macro_m macro_n <<<"$variant"
  source="$out_dir/source/static_${variant}.cu"
  python3 "$generator" --base "$base_src" --output "$source" \
    --size 16384 --macro-m "$macro_m" --macro-n "$macro_n" \
    --scheduler static
  "$nvcc_bin" -O3 -std=c++17 --resource-usage \
    -gencode arch=compute_100a,code=sm_100a \
    "$source" -lcuda -o "$out_dir/bin/static_${variant}" \
    2>&1 | tee "$out_dir/logs/build_static_${variant}.log"
done

cp -- "$base_src" "$generator" "$0" \
  "$repo_dir/5.GEMM/summarize_gemm_nsplit_static_orientation.py" \
  "$out_dir/source/"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"
sha256sum "$base_src" "$out_dir"/bin/* | tee "$out_dir/SHA256SUMS"

for variant in "${variants[@]}"; do
  for pattern in pattern ones; do
    "$out_dir/bin/static_${variant}" \
      --validate --validate-size 512 --validate-pattern "$pattern" \
      2>&1 | tee "$out_dir/logs/validate_static_${variant}_${pattern}.log"
  done
done

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_case() {
  local pass=$1 variant=$2
  snapshot_gpu "$out_dir/logs/gpu_static_${variant}_p${pass}_before.csv"
  "$out_dir/bin/static_${variant}" \
    --warmup 1 --iters 5 --input-init random-signed8 \
    --csv "$out_dir/csv/static_${variant}_p${pass}.csv" \
    2>&1 | tee "$out_dir/logs/static_${variant}_p${pass}.log"
}

printf 'position\tpass\tvariant\n' >"$out_dir/sequence.tsv"
for pass in 1 2 3 4; do
  if ((pass == 1 || pass == 4)); then
    order=(8x16 16x8)
  else
    order=(16x8 8x16)
  fi
  position=0
  for variant in "${order[@]}"; do
    position=$((position + 1))
    printf '%d\t%d\t%s\n' "$position" "$pass" "$variant" \
      >>"$out_dir/sequence.tsv"
    run_case "$pass" "$variant"
  done
done

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$repo_dir/5.GEMM/summarize_gemm_nsplit_static_orientation.py" \
  "$out_dir"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
