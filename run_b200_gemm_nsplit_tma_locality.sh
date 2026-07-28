#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_tma_locality}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_tma_only.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_tma_locality.py"

mkdir -p "$out_dir"/{bin,csv,logs,source}
exec > >(tee "$out_dir/raw.log") 2>&1

for mode in dense same_a same_b same; do
  generator_mode=${mode//_/-}
  python3 "$generator" --base "$base_src" \
    --output "$out_dir/source/${mode}.cu" --address-mode "$generator_mode"
  "$nvcc_bin" -O3 -std=c++17 --resource-usage \
    -gencode arch=compute_100a,code=sm_100a \
    "$out_dir/source/${mode}.cu" -lcuda -o "$out_dir/bin/${mode}" \
    2>&1 | tee "$out_dir/logs/build_${mode}.log"
done
cp -- "$base_src" "$generator" "$summarizer" "$0" "$out_dir/source/"

"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
sha256sum "$base_src" "$out_dir"/source/{dense,same_a,same_b,same}.cu \
  "$out_dir"/bin/* | tee "$out_dir/SHA256SUMS"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"

for mode in dense same_a same_b same; do
  "$out_dir/bin/$mode" --warmup 0 --iters 1 --input-init random \
    --csv "$out_dir/csv/smoke_${mode}.csv" \
    2>&1 | tee "$out_dir/logs/smoke_${mode}.log"
done

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_case() {
  local input_name=$1 mode=$2 pass=$3
  echo "BEGIN input=$input_name address=$mode pass=$pass"
  snapshot_gpu "$out_dir/logs/gpu_${input_name}_${mode}_p${pass}_before.csv"
  "$out_dir/bin/$mode" --warmup 1 --iters 5 --input-init "$input_name" \
    --csv "$out_dir/csv/${input_name}_${mode}_p${pass}.csv" \
    2>&1 | tee "$out_dir/logs/${input_name}_${mode}_p${pass}.log"
  echo "END input=$input_name address=$mode pass=$pass"
}

orders=(
  "dense same_a same_b same"
  "same same_b same_a dense"
  "same_a dense same same_b"
  "same_b same dense same_a"
)
echo "PROTOCOL size=16384 payload=64GiB warmup=1 timed=5 processes=4"
printf 'position\tpass\tinput\taddress_mode\n' >"$out_dir/sequence.tsv"
for pass in 1 2 3 4; do
  read -r -a mode_order <<<"${orders[$((pass - 1))]}"
  if ((pass == 1 || pass == 4)); then
    input_order=(random random-signed8)
  else
    input_order=(random-signed8 random)
  fi
  position=0
  for input_name in "${input_order[@]}"; do
    for mode in "${mode_order[@]}"; do
      position=$((position + 1))
      printf '%d\t%d\t%s\t%s\n' \
        "$position" "$pass" "$input_name" "$mode" >>"$out_dir/sequence.tsv"
      run_case "$input_name" "$mode" "$pass"
    done
  done
done

python3 "$summarizer" "$out_dir"
snapshot_gpu "$out_dir/logs/gpu_end.csv"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
