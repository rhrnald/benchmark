#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_scheduler_sweep}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_size_port.py"

sizes=(8192 16384 32768)
shapes=(4x16 8x16 4x32 8x18 12x12 16x16)
schedulers=(dynamic static)
variants=()
for scheduler in "${schedulers[@]}"; do
  for shape in "${shapes[@]}"; do
    variants+=("${scheduler}_${shape}")
  done
done

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/raw.log") 2>&1

for size in "${sizes[@]}"; do
  for scheduler in "${schedulers[@]}"; do
    for shape in "${shapes[@]}"; do
      IFS=x read -r macro_m macro_n <<<"$shape"
      name="${scheduler}_${shape}_${size}"
      source="$out_dir/source/${name}.cu"
      python3 "$generator" --base "$base_src" --output "$source" \
        --size "$size" --macro-m "$macro_m" --macro-n "$macro_n" \
        --scheduler "$scheduler"
      "$nvcc_bin" -O3 -std=c++17 \
        -gencode arch=compute_100a,code=sm_100a \
        "$source" -lcuda -o "$out_dir/bin/$name"
    done
  done
done

cp -- "$base_src" "$generator" "$0" "$out_dir/source/"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"
sha256sum "$base_src" "$out_dir"/bin/* | tee "$out_dir/SHA256SUMS"

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

for size in "${sizes[@]}"; do
  for variant in "${variants[@]}"; do
    "$out_dir/bin/${variant}_${size}" \
      --validate --validate-size 512 --validate-pattern pattern \
      2>&1 | tee "$out_dir/logs/validate_${variant}_${size}_pattern.log"
  done
done
for variant in "${variants[@]}"; do
  "$out_dir/bin/${variant}_16384" \
    --validate --validate-size 512 --validate-pattern ones \
    2>&1 | tee "$out_dir/logs/validate_${variant}_16384_ones.log"
done

run_case() {
  local pass=$1 size=$2 variant=$3
  echo "BEGIN pass=$pass size=$size variant=$variant"
  snapshot_gpu "$out_dir/logs/gpu_${variant}_${size}_p${pass}_before.csv"
  "$out_dir/bin/${variant}_${size}" \
    --warmup 1 --iters 5 --input-init random-signed8 \
    --csv "$out_dir/csv/${variant}_${size}_p${pass}.csv" \
    2>&1 | tee "$out_dir/logs/${variant}_${size}_p${pass}.log"
  echo "END pass=$pass size=$size variant=$variant"
}

printf 'position\tpass\tsize\tvariant\n' >"$out_dir/sequence.tsv"
run_order() {
  local pass=$1 size=$2
  shift 2
  local position=0 variant
  for variant in "$@"; do
    position=$((position + 1))
    printf '%d\t%d\t%d\t%s\n' "$position" "$pass" "$size" "$variant" \
      >>"$out_dir/sequence.tsv"
    run_case "$pass" "$size" "$variant"
  done
}

run_order 1 8192 "${variants[@]}"
run_order 1 16384 "${variants[@]}"
run_order 1 32768 "${variants[@]}"

reverse_variants=()
for ((i=${#variants[@]} - 1; i>=0; --i)); do
  reverse_variants+=("${variants[i]}")
done
run_order 2 32768 "${reverse_variants[@]}"
run_order 2 16384 "${reverse_variants[@]}"
run_order 2 8192 "${reverse_variants[@]}"

rotated_variants=("${variants[@]:4}" "${variants[@]:0:4}")
run_order 3 16384 "${rotated_variants[@]}"
run_order 3 32768 "${rotated_variants[@]}"
run_order 3 8192 "${rotated_variants[@]}"

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$repo_dir/5.GEMM/summarize_gemm_nsplit_scheduler_sweep.py" "$out_dir"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
