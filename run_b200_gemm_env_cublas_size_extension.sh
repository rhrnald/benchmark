#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
out_dir=${OUT_DIR:-"$repo_dir/results/gemm_env_cublas_size_extension"}
p0_bin=${P0_BIN:-"$repo_dir/results/gemm_phase_resweep_b200_45465499/bin/p0"}
cublas_reference_bin=${CUBLAS_REFERENCE_BIN:-"$repo_dir/6.cuBLAS/cublas_gemm_bench"}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
cuobjdump_bin=${CUOBJDUMP:-/usr/local/cuda-12.9/bin/cuobjdump}
definition_commit=${DEFINITION_COMMIT:-unknown}

warmup=1
iters=5
expected_base_sha=37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a
expected_p0_sha=166044d6690b52dc448befefb79baabb1c9cb56950b16a0536454249d623ff72
expected_cublas_sha=1f47799b1ffd8d815f457aa908be4ded8c8a3539d6a11fcc8053410612f5148b

if [[ -e "$out_dir" ]]; then
  echo "refusing to mix artifacts in existing OUT_DIR: $out_dir" >&2
  exit 2
fi
mkdir -p -- "$out_dir/bin" "$out_dir/csv" "$out_dir/logs" "$out_dir/source"
printf '%s\n' "$definition_commit" >"$out_dir/definition_commit.txt"

base_src="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
cublas_src="$repo_dir/6.cuBLAS/cublas_gemm_bench.cu"
base_sha=$(sha256sum "$base_src" | awk '{print $1}')
p0_sha=$(sha256sum "$p0_bin" | awk '{print $1}')
cublas_sha=$(sha256sum "$cublas_reference_bin" | awk '{print $1}')

if [[ "$base_sha" != "$expected_base_sha" ]]; then
  echo "unexpected E7a base source SHA-256: $base_sha" >&2
  exit 2
fi
if [[ ! -x "$p0_bin" || "$p0_sha" != "$expected_p0_sha" ]]; then
  echo "missing or unexpected p0 binary: $p0_sha" >&2
  exit 2
fi
if [[ ! -x "$cublas_reference_bin" ||
      "$cublas_sha" != "$expected_cublas_sha" ]]; then
  echo "missing or unexpected cuBLAS reference binary: $cublas_sha" >&2
  exit 2
fi

for literal in \
  "static constexpr int kBenchmarkSize = 16384;" \
  "static constexpr int kPersistentMacroM = 16;" \
  "static constexpr int kPersistentMacroN = 16;"; do
  if [[ $(grep -Fxc "$literal" "$base_src") != 1 ]]; then
    echo "base source does not contain exactly one expected literal: $literal" >&2
    exit 2
  fi
done
if [[ $(grep -Foc "scheduler=dynamic_16x16_mfast" "$base_src") != 1 ]]; then
  echo "base source does not contain exactly one expected scheduler banner" >&2
  exit 2
fi

generate_variant() {
  local size=$1
  local macro_m=$2
  local macro_n=$3
  local tag=$4
  local dst="$out_dir/source/gemm256_bf16_${tag}.cu"

  sed \
    -e "s/Clean 16K dense/Clean ${size} dense/" \
    -e "s/static constexpr int kBenchmarkSize = 16384;/static constexpr int kBenchmarkSize = ${size};/" \
    -e "s/static constexpr int kPersistentMacroM = 16;/static constexpr int kPersistentMacroM = ${macro_m};/" \
    -e "s/static constexpr int kPersistentMacroN = 16;/static constexpr int kPersistentMacroN = ${macro_n};/" \
    -e "s/gemm256_bf16_16k/gemm256_bf16_${tag}/g" \
    -e "s/scheduler=dynamic_16x16_mfast/scheduler=dynamic_${macro_m}x${macro_n}_mfast/" \
    "$base_src" >"$dst"

  grep -Fqx "static constexpr int kBenchmarkSize = ${size};" "$dst"
  grep -Fqx "static constexpr int kPersistentMacroM = ${macro_m};" "$dst"
  grep -Fqx "static constexpr int kPersistentMacroN = ${macro_n};" "$dst"
  grep -Fq "scheduler=dynamic_${macro_m}x${macro_n}_mfast" "$dst"
  diff -u "$base_src" "$dst" >"$out_dir/logs/diff_${tag}.patch" || true
}

generate_variant 8192 16 16 8k_m16n16
generate_variant 32768 16 16 32k_m16n16
generate_variant 32768 8 18 32k_m8n18
cp -- "$cublas_src" "$out_dir/source/cublas_gemm_bench.cu"
cp -- "$0" "$out_dir/source/run_b200_gemm_env_cublas_size_extension.sh"

build_variant() {
  local tag=$1
  "$nvcc_bin" -O3 -std=c++17 \
    -gencode arch=compute_100a,code=sm_100a \
    "$out_dir/source/gemm256_bf16_${tag}.cu" -lcuda \
    -o "$out_dir/bin/current_${tag}" \
    >"$out_dir/logs/build_${tag}.log" 2>&1
  "$cuobjdump_bin" --dump-resource-usage "$out_dir/bin/current_${tag}" \
    >"$out_dir/logs/resources_${tag}.txt" 2>&1
}

build_variant 8k_m16n16
build_variant 32k_m16n16
build_variant 32k_m8n18
cp -- "$p0_bin" "$out_dir/bin/p0"
cp -- "$cublas_reference_bin" "$out_dir/bin/cublas_gemm_bench"

{
  date --iso-8601=seconds
  uname -a
  "$nvcc_bin" --version
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,memory.total,memory.free \
    --format=csv,noheader,nounits
  ldd "$out_dir/bin/cublas_gemm_bench"
  cublas_so=$(ldd "$out_dir/bin/cublas_gemm_bench" |
    awk '$1 ~ /^libcublas.so/ {print $3; exit}')
  cublas_lt_so=$(ldd "$out_dir/bin/cublas_gemm_bench" |
    awk '$1 ~ /^libcublasLt.so/ {print $3; exit}')
  readlink -f "$cublas_so"
  readlink -f "$cublas_lt_so"
  sha256sum "$(readlink -f "$cublas_so")" "$(readlink -f "$cublas_lt_so")"
} >"$out_dir/logs/environment.txt" 2>&1
nvidia-smi -q >"$out_dir/logs/nvidia_smi_before.txt"

{
  sha256sum "$base_src" "$cublas_src" "$p0_bin" "$cublas_reference_bin"
  sha256sum "$out_dir/source/"*.cu "$out_dir/source/"*.sh
  sha256sum "$out_dir/bin/"*
} >"$out_dir/logs/sha256.txt"

validate_variant() {
  local tag=$1
  local pattern=$2
  "$out_dir/bin/current_${tag}" \
    --validate --validate-size 512 --validate-pattern "$pattern" \
    >"$out_dir/logs/validate_${tag}_${pattern}.log" 2>&1
}

validate_variant 8k_m16n16 pattern
validate_variant 8k_m16n16 ones
validate_variant 32k_m16n16 pattern
validate_variant 32k_m8n18 pattern
validate_variant 32k_m8n18 ones
"$out_dir/bin/p0" \
  --validate --validate-size 512 --validate-pattern pattern \
  --persistent-ctas 1 >"$out_dir/logs/validate_p0_pattern.log" 2>&1

printf 'pass\tsize\tdistribution\tposition\tmethod\n' >"$out_dir/sequence.tsv"

run_case() {
  local method=$1
  local size=$2
  local distribution=$3
  local pass=$4
  local position=$5
  local custom_input=random
  local cublas_input=unit

  if [[ "$distribution" == signed8 ]]; then
    custom_input=random-signed8
    cublas_input=signed8
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$pass" "$size" "$distribution" "$position" "$method" \
    >>"$out_dir/sequence.tsv"
  nvidia-smi \
    --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv,noheader,nounits \
    >"$out_dir/logs/nvidia_${size}_${distribution}_${method}_p${pass}_before.txt"

  case "$method" in
    p0)
      "$out_dir/bin/p0" \
        --sizes "$size" --warmup "$warmup" --iters "$iters" \
        --input-init "$custom_input" --persistent-ctas 148 \
        --csv "$out_dir/csv/p0_${size}_${distribution}_p${pass}.csv" \
        >"$out_dir/logs/p0_${size}_${distribution}_p${pass}.log" 2>&1
      ;;
    current_m16n16)
      local current_tag=8k_m16n16
      if [[ "$size" == 32768 ]]; then
        current_tag=32k_m16n16
      fi
      "$out_dir/bin/current_${current_tag}" \
        --warmup "$warmup" --iters "$iters" --input-init "$custom_input" \
        --csv "$out_dir/csv/current_m16n16_${size}_${distribution}_p${pass}.csv" \
        >"$out_dir/logs/current_m16n16_${size}_${distribution}_p${pass}.log" 2>&1
      ;;
    current_m8n18)
      if [[ "$size" != 32768 ]]; then
        echo "current_m8n18 is defined only for size 32768" >&2
        exit 2
      fi
      "$out_dir/bin/current_32k_m8n18" \
        --warmup "$warmup" --iters "$iters" --input-init "$custom_input" \
        --csv "$out_dir/csv/current_m8n18_${size}_${distribution}_p${pass}.csv" \
        >"$out_dir/logs/current_m8n18_${size}_${distribution}_p${pass}.log" 2>&1
      ;;
    cublas)
      "$out_dir/bin/cublas_gemm_bench" \
        --device 0 --m "$size" --n "$size" --k "$size" \
        --warmup "$warmup" --repeat "$iters" --mode bf16fp32 \
        --input-dist "$cublas_input" \
        >"$out_dir/logs/cublas_${size}_${distribution}_p${pass}.log" 2>&1
      ;;
    *)
      echo "unknown method: $method" >&2
      exit 2
      ;;
  esac
}

run_group() {
  local pass=$1
  local size=$2
  local distribution=$3
  shift 3
  local position=0
  local method
  for method in "$@"; do
    position=$((position + 1))
    run_case "$method" "$size" "$distribution" "$pass" "$position"
  done
}

# Three-method Latin rotation at 8K.
run_group 1 8192 unit p0 current_m16n16 cublas
run_group 1 8192 signed8 p0 current_m16n16 cublas
run_group 2 8192 signed8 current_m16n16 cublas p0
run_group 2 8192 unit current_m16n16 cublas p0
run_group 3 8192 unit cublas p0 current_m16n16
run_group 3 8192 signed8 cublas p0 current_m16n16

# Four-method Latin rotation at 32K.  The direct 16x16 size port and the
# historical size-tuned 8x18 scheduler are deliberately separate rows.
run_group 1 32768 signed8 p0 current_m8n18 current_m16n16 cublas
run_group 1 32768 unit p0 current_m8n18 current_m16n16 cublas
run_group 2 32768 unit current_m8n18 current_m16n16 cublas p0
run_group 2 32768 signed8 current_m8n18 current_m16n16 cublas p0
run_group 3 32768 signed8 current_m16n16 cublas p0 current_m8n18
run_group 3 32768 unit current_m16n16 cublas p0 current_m8n18
run_group 4 32768 unit cublas p0 current_m8n18 current_m16n16
run_group 4 32768 signed8 cublas p0 current_m8n18 current_m16n16

nvidia-smi -q >"$out_dir/logs/nvidia_smi_after.txt"
echo "completed: $out_dir"
