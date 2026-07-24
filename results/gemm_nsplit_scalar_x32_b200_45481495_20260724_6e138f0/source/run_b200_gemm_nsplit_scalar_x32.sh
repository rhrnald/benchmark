#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_scalar_x32_b200}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
if [[ ! -x "$nvcc_bin" ]]; then
  nvcc_bin=/usr/local/cuda/bin/nvcc
fi
cuobjdump_bin="$(dirname "$nvcc_bin")/cuobjdump"
test -x "$nvcc_bin"
test -x "$cuobjdump_bin"
runner_path=$(realpath "$0")

exact_source="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
transpose_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose.py"
x32_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose_scalar_x32.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_scalar_x32.py"

expected_exact_sha256=cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca
expected_x64_sha256=a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc
expected_x32_sha256=be19af8e1c885ef1865795f573a42b72b249e7261ccb15cf56d24aa35b557006

test "$(sha256sum "$exact_source" | awk '{print $1}')" = \
  "$expected_exact_sha256"
for required in \
  "$transpose_generator" \
  "$x32_generator" \
  "$summarizer"; do
  test -f "$required"
done
test ! -e "$out_dir"

mkdir -p "$out_dir"/{bin,csv,logs,source}
if git -C "$repo_dir" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$repo_dir" rev-parse HEAD >"$out_dir/definition_commit.txt"
else
  printf '%s\n' "${GEMM_DEFINITION_COMMIT:-unknown}" \
    >"$out_dir/definition_commit.txt"
fi

cp "$exact_source" "$out_dir/source/nsplit_exact.cu"
cp \
  "$transpose_generator" \
  "$x32_generator" \
  "$summarizer" \
  "$runner_path" \
  "$out_dir/source/"

python3 "$transpose_generator" \
  --source "$exact_source" \
  --output "$out_dir/source/nsplit_transpose_scalar_x64.cu" \
  >"$out_dir/logs/generate_scalar_x64.log"
test "$(
  sha256sum "$out_dir/source/nsplit_transpose_scalar_x64.cu" |
    awk '{print $1}'
)" = "$expected_x64_sha256"

python3 "$x32_generator" \
  --input "$out_dir/source/nsplit_transpose_scalar_x64.cu" \
  --output "$out_dir/source/nsplit_transpose_scalar_x32.cu" \
  >"$out_dir/logs/generate_scalar_x32.log"
test "$(
  sha256sum "$out_dir/source/nsplit_transpose_scalar_x32.cu" |
    awk '{print $1}'
)" = "$expected_x32_sha256"

variants=(
  nsplit_exact
  nsplit_transpose_scalar_x64
  nsplit_transpose_scalar_x32
)
common_flags=(
  -std=c++17 -O3
  -gencode arch=compute_100a,code=sm_100a
  -lineinfo -Xptxas=-v
)

for variant in "${variants[@]}"; do
  "$nvcc_bin" "${common_flags[@]}" "$out_dir/source/${variant}.cu" \
    -o "$out_dir/bin/$variant" -lcuda \
    >"$out_dir/logs/build_${variant}.log" 2>&1
  "$cuobjdump_bin" -res-usage "$out_dir/bin/$variant" \
    >"$out_dir/logs/resource_${variant}.txt"
  "$cuobjdump_bin" -sass "$out_dir/bin/$variant" \
    >"$out_dir/logs/sass_${variant}.txt"
  awk '
    /Function : .*gemm256_bf16_16k_kernel/ { take = 1; next }
    take && /Function : / { exit }
    take
  ' "$out_dir/logs/sass_${variant}.txt" |
    sed -nE \
      's@^[[:space:]]*/\*[0-9a-f]+\*/[[:space:]]+(.*)[[:space:]]+;[[:space:]]*/\*.*@\1;@p' \
      >"$out_dir/logs/kernel_ops_${variant}.txt"
  test -s "$out_dir/logs/kernel_ops_${variant}.txt"
  awk '
    /Function .*gemm256_bf16_16k_kernel/ { take = 1; next }
    take { print; found = 1; exit }
    END { if (!found) exit 1 }
  ' "$out_dir/logs/resource_${variant}.txt" \
    >"$out_dir/logs/kernel_resource_${variant}.txt"
  test -s "$out_dir/logs/kernel_resource_${variant}.txt"
  grep -Eq 'REG:[0-9]+ STACK:0 .*LOCAL:0' \
    "$out_dir/logs/kernel_resource_${variant}.txt"
  test "$(
    grep -Ec '[1-9][0-9]* bytes spill (stores|loads)' \
      "$out_dir/logs/build_${variant}.log" || true
  )" -eq 0
done

count_op() {
  local variant=$1
  local pattern=$2
  grep -Ec "$pattern" "$out_dir/logs/kernel_ops_${variant}.txt" || true
}

test "$(wc -l <"$out_dir/logs/kernel_ops_nsplit_exact.txt")" -eq 1913
test "$(
  wc -l <"$out_dir/logs/kernel_ops_nsplit_transpose_scalar_x64.txt"
)" -eq 2229
test "$(
  wc -l <"$out_dir/logs/kernel_ops_nsplit_transpose_scalar_x32.txt"
)" -eq 2011

test "$(count_op nsplit_exact '^UTCHMMA ')" -eq 8
for variant in \
  nsplit_transpose_scalar_x64 \
  nsplit_transpose_scalar_x32; do
  test "$(count_op "$variant" '^UTCHMMA ')" -eq 4
done
for variant in "${variants[@]}"; do
  test "$(count_op "$variant" '^UTCBAR')" -eq 1
  test "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" -eq 21
  test "$(count_op "$variant" 'UTMASTG.4D')" -eq 4
  test "$(count_op "$variant" 'SYNCS.PHASECHK')" -eq 48
done

test "$(count_op nsplit_exact 'LDTM.x64')" -eq 8
test "$(count_op nsplit_exact 'LDTM.x(16|32)')" -eq 0
test "$(count_op nsplit_transpose_scalar_x64 'LDTM.x64')" -eq 8
test "$(count_op nsplit_transpose_scalar_x64 'LDTM.x(16|32)')" -eq 0
test "$(count_op nsplit_transpose_scalar_x32 'LDTM.x32')" -eq 4
test "$(count_op nsplit_transpose_scalar_x32 'LDTM.x(16|64)')" -eq 0

test "$(count_op nsplit_transpose_scalar_x64 '^ST.E ')" -eq 512
test "$(count_op nsplit_transpose_scalar_x32 '^ST.E ')" -eq 128
test "$(count_op nsplit_transpose_scalar_x64 '^SHFL')" -eq 0
test "$(count_op nsplit_transpose_scalar_x32 '^SHFL')" -eq 0

get_reg() {
  local variant=$1
  sed -nE 's/.*REG:([0-9]+).*/\1/p' \
    "$out_dir/logs/kernel_resource_${variant}.txt"
}

test "$(get_reg nsplit_exact)" -eq 174
test "$(get_reg nsplit_transpose_scalar_x64)" -eq 174
test "$(get_reg nsplit_transpose_scalar_x32)" -eq 91

{
  printf '%s\n' \
    'variant	ops	reg	utchmma	utcbar	tma_load	tma_store	tmem_x16	tmem_x32	tmem_x64	shfl	scalar_st32'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(wc -l <"$out_dir/logs/kernel_ops_${variant}.txt")" \
      "$(get_reg "$variant")" \
      "$(count_op "$variant" '^UTCHMMA ')" \
      "$(count_op "$variant" '^UTCBAR')" \
      "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" \
      "$(count_op "$variant" 'UTMASTG.4D')" \
      "$(count_op "$variant" 'LDTM.x16')" \
      "$(count_op "$variant" 'LDTM.x32')" \
      "$(count_op "$variant" 'LDTM.x64')" \
      "$(count_op "$variant" '^SHFL')" \
      "$(count_op "$variant" '^ST.E ')"
  done
} >"$out_dir/codegen_counts.tsv"

{
  printf 'variant\tsource_sha256\tbinary_sha256\tsass_sha256\tkernel_ops_sha256\tkernel_resource_sha256\n'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(sha256sum "$out_dir/source/${variant}.cu" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/bin/$variant" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/sass_${variant}.txt" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/kernel_ops_${variant}.txt" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/kernel_resource_${variant}.txt" | awk '{print $1}')"
  done
} >"$out_dir/codegen_hashes.tsv"

if [[ "${GEMM_CODEGEN_ONLY:-0}" == 1 ]]; then
  (
    cd "$out_dir"
    find . -type f ! -name SHA256SUMS -print0 | sort -z |
      xargs -0 sha256sum >SHA256SUMS
  )
  echo "completed codegen-only result_dir=$out_dir"
  exit 0
fi

{
  date -Is
  uname -a
  "$nvcc_bin" --version
  nvidia-smi -L
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,temperature.gpu,power.limit,clocks.max.sm,memory.total \
    --format=csv,noheader
} >"$out_dir/logs/environment_before.txt"

timeout -k 10s 180s "$out_dir/bin/nsplit_exact" \
  --validate --validate-size 512 --validate-pattern pattern \
  >"$out_dir/logs/validate_nsplit_exact_pattern512.log" 2>&1
timeout -k 10s 180s "$out_dir/bin/nsplit_exact" \
  --validate --validate-size 512 --validate-pattern ones \
  >"$out_dir/logs/validate_nsplit_exact_ones512.log" 2>&1
for variant in \
  nsplit_transpose_scalar_x64 \
  nsplit_transpose_scalar_x32; do
  for specification in "256 pattern" "512 pattern" "512 ones"; do
    read -r size pattern <<<"$specification"
    timeout -k 10s 180s "$out_dir/bin/$variant" \
      --validate --validate-size "$size" --validate-pattern "$pattern" \
      >"$out_dir/logs/validate_${variant}_${pattern}${size}.log" 2>&1
  done
done

exact=nsplit_exact
x64=nsplit_transpose_scalar_x64
x32=nsplit_transpose_scalar_x32
random_orders=(
  "$exact $x64 $x32"
  "$exact $x32 $x64"
  "$x64 $exact $x32"
  "$x64 $x32 $exact"
  "$x32 $exact $x64"
  "$x32 $x64 $exact"
)
signed_orders=(
  "$x32 $x64 $exact"
  "$x32 $exact $x64"
  "$x64 $x32 $exact"
  "$x64 $exact $x32"
  "$exact $x32 $x64"
  "$exact $x64 $x32"
)

sequence_path="$out_dir/sequence.tsv"
printf 'input\tpass\torder\n' >"$sequence_path"
for input_name in random random-signed8; do
  if [[ "$input_name" == random ]]; then
    orders=("${random_orders[@]}")
  else
    orders=("${signed_orders[@]}")
  fi
  for pass_offset in "${!orders[@]}"; do
    printf '%s\t%s\t%s\n' \
      "$input_name" "$((pass_offset + 1))" "${orders[$pass_offset]}" \
      >>"$sequence_path"
  done
done

run_case() {
  local input_name=$1
  local pass_idx=$2
  local variant=$3
  local csv_path="$out_dir/csv/${variant}_${input_name}_p${pass_idx}.csv"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader \
    >"$out_dir/logs/nvidia_${variant}_${input_name}_p${pass_idx}_before.txt"
  echo "run input=$input_name pass=$pass_idx variant=$variant" \
    | tee -a "$out_dir/logs/perf.log"
  timeout -k 30s 300s "$out_dir/bin/$variant" \
    --warmup 1 --iters 5 --input-init "$input_name" --csv "$csv_path" \
    2>&1 | tee -a "$out_dir/logs/perf.log"
  nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm \
    --format=csv,noheader \
    >"$out_dir/logs/nvidia_${variant}_${input_name}_p${pass_idx}_after.txt"
}

while IFS=$'\t' read -r input_name pass_idx order; do
  for variant in $order; do
    run_case "$input_name" "$pass_idx" "$variant"
  done
done < <(tail -n +2 "$sequence_path")

python3 "$summarizer" "$out_dir" | tee "$out_dir/logs/summarize.log"

{
  date -Is
  nvidia-smi \
    --query-gpu=name,uuid,driver_version,temperature.gpu,power.limit,clocks.sm,memory.total \
    --format=csv,noheader
} >"$out_dir/logs/environment_after.txt"

(
  cd "$out_dir"
  find . -type f ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum >SHA256SUMS
)

echo "completed result_dir=$out_dir"
