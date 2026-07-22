#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
binary=${1:-"$script_dir/gemm256_bf16_16k"}
output_dir=${2:-"$script_dir/results"}
input=${3:-random}

case "$input" in
  random|random-signed8) ;;
  *)
    echo "input must be random or random-signed8" >&2
    exit 2
    ;;
esac

mkdir -p -- "$output_dir"
stamp=$(date +%Y%m%d_%H%M%S)
name="clean_16384_${input}_${stamp}"

"$binary" \
  --warmup 1 \
  --iters 5 \
  --input-init "$input" \
  --csv "$output_dir/$name.csv" \
  2>&1 | tee "$output_dir/$name.log"
