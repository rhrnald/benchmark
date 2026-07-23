#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out_dir=${OUT_DIR:-"$repo_dir/results/gemm_env_cublas_size_extension"}
definition_commit=${DEFINITION_COMMIT:-unknown}
warmup=1
iters=5

expected_p0_sha=166044d6690b52dc448befb79baabb1c9cb56950b16a0536454249d623ff72
expected_current_sha=43919d6a43f088d5637df1bd352fae30c919df0a698a2d37a86db0d0967b30c3
expected_cublas_sha=1f47799b1ffd8d815f457aa908be4ded8c8a3539d6a11fcc8053410612f5148b

p0_bin="$out_dir/bin/p0"
current_bin="$out_dir/bin/current_8k_m16n16"
cublas_bin="$out_dir/bin/cublas_gemm_bench"

if [[ ! -d "$out_dir/logs" || ! -d "$out_dir/csv" ||
      ! -d "$out_dir/source" ]]; then
  echo "missing initial size-extension result directory: $out_dir" >&2
  exit 2
fi

check_sha() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ ! -x "$path" || "$actual" != "$expected" ]]; then
    echo "missing or unexpected binary $path: $actual" >&2
    exit 2
  fi
}

check_sha "$p0_bin" "$expected_p0_sha"
check_sha "$current_bin" "$expected_current_sha"
check_sha "$cublas_bin" "$expected_cublas_sha"

for pass in 4 5 6; do
  for method in p0 current_m16n16 cublas; do
    if [[ -e "$out_dir/logs/${method}_8192_signed8_p${pass}.log" ]]; then
      echo "refusing to overwrite pass $pass method $method" >&2
      exit 2
    fi
  done
done

printf '%s\n' "$definition_commit" \
  >"$out_dir/confirmation_definition_commit.txt"
cp -- "$0" \
  "$out_dir/source/run_b200_gemm_8k_signed_confirmation.sh"
printf 'pass\tsize\tdistribution\tposition\tmethod\n' \
  >"$out_dir/sequence_8k_signed_confirmation.tsv"

run_case() {
  local method=$1
  local pass=$2
  local position=$3

  printf '%s\t8192\tsigned8\t%s\t%s\n' \
    "$pass" "$position" "$method" \
    >>"$out_dir/sequence_8k_signed_confirmation.tsv"
  nvidia-smi \
    --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    >"$out_dir/logs/nvidia_8192_signed8_${method}_p${pass}_before.txt"

  case "$method" in
    p0)
      "$p0_bin" \
        --sizes 8192 --warmup "$warmup" --iters "$iters" \
        --input-init random-signed8 --persistent-ctas 148 \
        --csv "$out_dir/csv/p0_8192_signed8_p${pass}.csv" \
        >"$out_dir/logs/p0_8192_signed8_p${pass}.log" 2>&1
      ;;
    current_m16n16)
      "$current_bin" \
        --warmup "$warmup" --iters "$iters" \
        --input-init random-signed8 \
        --csv "$out_dir/csv/current_m16n16_8192_signed8_p${pass}.csv" \
        >"$out_dir/logs/current_m16n16_8192_signed8_p${pass}.log" 2>&1
      ;;
    cublas)
      "$cublas_bin" \
        --device 0 --m 8192 --n 8192 --k 8192 \
        --warmup "$warmup" --repeat "$iters" --mode bf16fp32 \
        --input-dist signed8 \
        >"$out_dir/logs/cublas_8192_signed8_p${pass}.log" 2>&1
      ;;
    *)
      echo "unknown method: $method" >&2
      exit 2
      ;;
  esac
}

run_group() {
  local pass=$1
  shift
  local position=0
  local method
  for method in "$@"; do
    position=$((position + 1))
    run_case "$method" "$pass" "$position"
  done
}

# A second complete Latin rotation gives six process samples per method when
# combined with passes 1--3 from the initial size-extension run.
run_group 4 p0 current_m16n16 cublas
run_group 5 current_m16n16 cublas p0
run_group 6 cublas p0 current_m16n16

nvidia-smi -q >"$out_dir/logs/nvidia_smi_after_8k_signed_confirmation.txt"
{
  sha256sum "$out_dir/source/run_b200_gemm_8k_signed_confirmation.sh"
  sha256sum "$p0_bin" "$current_bin" "$cublas_bin"
} >"$out_dir/logs/sha256_8k_signed_confirmation.txt"

echo "completed: $out_dir"
