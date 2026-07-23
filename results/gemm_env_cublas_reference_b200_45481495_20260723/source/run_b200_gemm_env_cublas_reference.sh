#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out_dir=${OUT_DIR:-"$repo_dir/results/gemm_env_cublas_reference"}
p0_bin=${P0_BIN:-"$repo_dir/results/gemm_phase_resweep_b200_45465499/bin/p0"}
cublas_reference_bin=${CUBLAS_REFERENCE_BIN:-"$repo_dir/6.cuBLAS/cublas_gemm_bench"}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
definition_commit=${DEFINITION_COMMIT:-unknown}

size=16384
warmup=1
iters=5
expected_p0_sha=166044d6690b52dc448befefb79baabb1c9cb56950b16a0536454249d623ff72
expected_cublas_sha=1f47799b1ffd8d815f457aa908be4ded8c8a3539d6a11fcc8053410612f5148b

mkdir -p -- "$out_dir/bin" "$out_dir/csv" "$out_dir/logs" "$out_dir/source"
printf '%s\n' "$definition_commit" >"$out_dir/definition_commit.txt"

if [[ ! -x "$p0_bin" ]]; then
  echo "missing executable p0 binary: $p0_bin" >&2
  exit 2
fi

actual_p0_sha=$(sha256sum "$p0_bin" | awk '{print $1}')
if [[ "$actual_p0_sha" != "$expected_p0_sha" ]]; then
  echo "unexpected p0 SHA-256: $actual_p0_sha" >&2
  exit 2
fi

if [[ ! -x "$cublas_reference_bin" ]]; then
  echo "missing executable cuBLAS reference binary: $cublas_reference_bin" >&2
  exit 2
fi

actual_cublas_sha=$(sha256sum "$cublas_reference_bin" | awk '{print $1}')
if [[ "$actual_cublas_sha" != "$expected_cublas_sha" ]]; then
  echo "unexpected cuBLAS reference SHA-256: $actual_cublas_sha" >&2
  exit 2
fi

current_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
cublas_src="$repo_dir/6.cuBLAS/cublas_gemm_bench.cu"
current_bin="$out_dir/bin/current_e7a"
cublas_bin="$out_dir/bin/cublas_gemm_bench"

cp -- "$current_src" "$out_dir/source/gemm256_bf16_16k.cu"
cp -- "$cublas_src" "$out_dir/source/cublas_gemm_bench.cu"
cp -- "$0" "$out_dir/source/run_b200_gemm_env_cublas_reference.sh"

"$nvcc_bin" -O3 -std=c++17 \
  -gencode arch=compute_100a,code=sm_100a \
  "$current_src" -lcuda -o "$current_bin"
cp -- "$cublas_reference_bin" "$cublas_bin"

{
  date --iso-8601=seconds
  uname -a
  "$nvcc_bin" --version
  nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,memory.total \
    --format=csv,noheader,nounits
  ldd "$cublas_bin"
  cublas_so=$(ldd "$cublas_bin" | awk '$1 ~ /^libcublas.so/ {print $3; exit}')
  cublas_lt_so=$(ldd "$cublas_bin" | awk '$1 ~ /^libcublasLt.so/ {print $3; exit}')
  readlink -f "$cublas_so"
  readlink -f "$cublas_lt_so"
  sha256sum "$(readlink -f "$cublas_so")" "$(readlink -f "$cublas_lt_so")"
} >"$out_dir/logs/environment.txt" 2>&1
nvidia-smi -q >"$out_dir/logs/nvidia_smi_before.txt"

{
  sha256sum "$current_src" "$cublas_src" "$p0_bin"
  sha256sum "$current_bin" "$cublas_bin" "$cublas_reference_bin"
} >"$out_dir/logs/sha256.txt"

"$current_bin" --validate --validate-size 512 --validate-pattern pattern \
  >"$out_dir/logs/validate_current.log" 2>&1
"$p0_bin" --validate --validate-size 512 --validate-pattern pattern \
  --persistent-ctas 1 >"$out_dir/logs/validate_p0.log" 2>&1

printf 'pass\tdistribution\tposition\tmethod\n' >"$out_dir/sequence.tsv"

run_case() {
  local method=$1
  local distribution=$2
  local pass=$3
  local position=$4
  local custom_input=random
  local cublas_input=unit

  if [[ "$distribution" == signed8 ]]; then
    custom_input=random-signed8
    cublas_input=signed8
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "$pass" "$distribution" "$position" "$method" >>"$out_dir/sequence.tsv"
  nvidia-smi \
    --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    >"$out_dir/logs/nvidia_${distribution}_${method}_p${pass}_before.txt"

  case "$method" in
    p0)
      "$p0_bin" --sizes "$size" --warmup "$warmup" --iters "$iters" \
        --input-init "$custom_input" --persistent-ctas 148 \
        --csv "$out_dir/csv/p0_${distribution}_p${pass}.csv" \
        >"$out_dir/logs/p0_${distribution}_p${pass}.log" 2>&1
      ;;
    current)
      "$current_bin" --warmup "$warmup" --iters "$iters" \
        --input-init "$custom_input" \
        --csv "$out_dir/csv/current_${distribution}_p${pass}.csv" \
        >"$out_dir/logs/current_${distribution}_p${pass}.log" 2>&1
      ;;
    cublas)
      "$cublas_bin" --device 0 --m "$size" --n "$size" --k "$size" \
        --warmup "$warmup" --repeat "$iters" --mode bf16fp32 \
        --input-dist "$cublas_input" \
        >"$out_dir/logs/cublas_${distribution}_p${pass}.log" 2>&1
      ;;
    *)
      echo "unknown method: $method" >&2
      exit 2
      ;;
  esac
}

run_group() {
  local pass=$1
  local distribution=$2
  shift 2
  local position=0
  local method
  for method in "$@"; do
    position=$((position + 1))
    run_case "$method" "$distribution" "$pass" "$position"
  done
}

run_group 1 unit p0 current cublas
run_group 1 signed8 cublas p0 current
run_group 2 signed8 p0 current cublas
run_group 2 unit current cublas p0
run_group 3 unit cublas p0 current
run_group 3 signed8 current cublas p0

nvidia-smi -q >"$out_dir/logs/nvidia_smi_after.txt"

echo "completed: $out_dir"
