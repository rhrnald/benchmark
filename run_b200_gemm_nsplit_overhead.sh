#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_overhead}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_overhead.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_overhead.py"
variants=(baseline no_memset sink_keep sink_trim fixed suspend fixed_sink all)
inputs=(random random-signed8)

mkdir -p "$out_dir"/{bin,csv,logs,sass,source}
exec > >(tee "$out_dir/raw.log") 2>&1

generate_variant() {
  local variant=$1
  local args=()
  case "$variant" in
    baseline) ;;
    no_memset) args+=(--sink no-memset) ;;
    sink_keep) args+=(--sink keep-sync) ;;
    sink_trim) args+=(--sink trim) ;;
    fixed) args+=(--fixed-16k) ;;
    suspend) args+=(--suspend-producer) ;;
    fixed_sink) args+=(--sink trim --fixed-16k) ;;
    all) args+=(--sink trim --fixed-16k --suspend-producer) ;;
    *) echo "unknown variant: $variant" >&2; exit 1 ;;
  esac
  python3 "$generator" --base "$base_src" \
    --output "$out_dir/source/${variant}.cu" --label "$variant" "${args[@]}"
}

for variant in "${variants[@]}"; do
  generate_variant "$variant"
  "$nvcc_bin" -O3 -std=c++17 --resource-usage \
    -gencode arch=compute_100a,code=sm_100a \
    "$out_dir/source/${variant}.cu" -lcuda -o "$out_dir/bin/${variant}" \
    2>&1 | tee "$out_dir/logs/build_${variant}.log"
  /usr/local/cuda/bin/cuobjdump -sass "$out_dir/bin/${variant}" \
    >"$out_dir/sass/${variant}.sass"
done

cp -- "$base_src" "$generator" "$summarizer" "$0" "$out_dir/source/"
"$nvcc_bin" --version | tee "$out_dir/logs/nvcc_version.txt"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_before.txt"
git -C "$repo_dir" rev-parse HEAD | tee "$out_dir/definition_commit.txt"
sha256sum "$base_src" "$out_dir"/bin/* | tee "$out_dir/SHA256SUMS"

for variant in "${variants[@]}"; do
  for pattern in pattern ones; do
    "$out_dir/bin/${variant}" \
      --validate --validate-size 512 --validate-pattern "$pattern" \
      2>&1 | tee "$out_dir/logs/validate_${variant}_${pattern}.log"
  done
done

snapshot_gpu() {
  local output=$1
  nvidia-smi \
    --query-gpu=timestamp,name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader >"$output"
}

run_case() {
  local pass=$1 input_name=$2 variant=$3
  snapshot_gpu "$out_dir/logs/gpu_${input_name}_${variant}_p${pass}_before.csv"
  "$out_dir/bin/${variant}" \
    --warmup 1 --iters 5 --input-init "$input_name" \
    --csv "$out_dir/csv/${input_name}_${variant}_p${pass}.csv" \
    2>&1 | tee "$out_dir/logs/${input_name}_${variant}_p${pass}.log"
}

order_p1=(baseline no_memset sink_keep sink_trim fixed suspend fixed_sink all)
order_p2=(all fixed_sink suspend fixed sink_trim sink_keep no_memset baseline)
order_p3=(fixed suspend fixed_sink all baseline no_memset sink_keep sink_trim)
order_p4=(sink_trim sink_keep no_memset baseline all fixed_sink suspend fixed)

printf 'position\tpass\tinput\tvariant\n' >"$out_dir/sequence.tsv"
for pass in 1 2 3 4; do
  order_name="order_p${pass}[@]"
  run_order=("${!order_name}")
  if ((pass == 1 || pass == 4)); then
    input_order=(random random-signed8)
  else
    input_order=(random-signed8 random)
  fi
  for input_name in "${input_order[@]}"; do
    position=0
    for variant in "${run_order[@]}"; do
      position=$((position + 1))
      printf '%d\t%d\t%s\t%s\n' \
        "$position" "$pass" "$input_name" "$variant" >>"$out_dir/sequence.tsv"
      run_case "$pass" "$input_name" "$variant"
    done
  done
done

snapshot_gpu "$out_dir/logs/gpu_end.csv"
python3 "$summarizer" "$out_dir"
nvidia-smi -q >"$out_dir/logs/nvidia_smi_q_after.txt"
tar -C "$(dirname "$out_dir")" -czf "${out_dir}.tar.gz" \
  "$(basename "$out_dir")"
sha256sum "${out_dir}.tar.gz" >"${out_dir}.tar.gz.sha256"
