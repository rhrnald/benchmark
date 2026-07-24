#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_epilogue_b200}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
if [[ ! -x "$nvcc_bin" ]]; then
  nvcc_bin=/usr/local/cuda/bin/nvcc
fi
cuobjdump_bin="$(dirname "$nvcc_bin")/cuobjdump"
test -x "$cuobjdump_bin"
runner_path=$(realpath "$0")

exact_source="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
transpose_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose.py"
nostore_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_no_cstore.py"
vec2_x32_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose_vec2_x32.py"
vec2_cf1_x32_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose_vec2_cf1_x32.py"
vec2_cf_x32_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose_vec2_cf_x32.py"
vec2_cf_x64_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose_vec2_cf.py"
vec4_cf_x64_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_transpose_vec4_cf.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_epilogue.py"

expected_exact_sha256=cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca
expected_scalar_sha256=a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc
expected_nostore_sha256=d12f93596f50f92c93fd40b9c790c4c5069c5f3b1e469633f81e9ed6589add98
expected_vec2_x32_sha256=329199e0dd8c655d75d94aec61542aacb9e8dddee21e7ff6e8cae0ec05a6130c
expected_vec2_cf1_x32_sha256=8ae3a4a3dcdb2fa81f4baedbcae2c7c2faf6c78ec0bc57d8d4db16cc57d14b0f
expected_vec2_cf_x32_sha256=5dc5629f7abf64e1e59e90cfbe0f18ddbdd5e6ec64d33911ebbe1f5fc5125229
expected_vec2_cf_x64_sha256=3ba921094fe3393a1329462f2de05974cfcb781db7e8f58314b04b11bd28ce6c
expected_vec4_cf_x64_sha256=b128e00ef200d065b8eb117f33f4ff1033ee68e3cccc39e6b66f21bc277489cc

test "$(sha256sum "$exact_source" | awk '{print $1}')" = \
  "$expected_exact_sha256"
for required in \
  "$transpose_generator" \
  "$nostore_generator" \
  "$vec2_x32_generator" \
  "$vec2_cf1_x32_generator" \
  "$vec2_cf_x32_generator" \
  "$vec2_cf_x64_generator" \
  "$vec4_cf_x64_generator" \
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
  "$nostore_generator" \
  "$vec2_x32_generator" \
  "$vec2_cf1_x32_generator" \
  "$vec2_cf_x32_generator" \
  "$vec2_cf_x64_generator" \
  "$vec4_cf_x64_generator" \
  "$summarizer" \
  "$runner_path" \
  "$out_dir/source/"

python3 "$transpose_generator" --source "$exact_source" \
  --output "$out_dir/source/nsplit_transpose_scalar.cu" \
  >"$out_dir/logs/generate_transpose_scalar.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_scalar.cu" | awk '{print $1}')" = \
  "$expected_scalar_sha256"

python3 "$nostore_generator" \
  --source "$out_dir/source/nsplit_transpose_scalar.cu" \
  --variant transpose \
  --output "$out_dir/source/nsplit_transpose_nostore.cu" \
  >"$out_dir/logs/generate_transpose_nostore.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_nostore.cu" | awk '{print $1}')" = \
  "$expected_nostore_sha256"

python3 "$vec2_x32_generator" \
  --input "$out_dir/source/nsplit_transpose_scalar.cu" \
  --output "$out_dir/source/nsplit_transpose_vec2_x32.cu" \
  >"$out_dir/logs/generate_vec2_x32.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_vec2_x32.cu" | awk '{print $1}')" = \
  "$expected_vec2_x32_sha256"

python3 "$vec2_cf1_x32_generator" \
  --input "$out_dir/source/nsplit_transpose_scalar.cu" \
  --output "$out_dir/source/nsplit_transpose_vec2_cf1_x32.cu" \
  >"$out_dir/logs/generate_vec2_cf1_x32.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_vec2_cf1_x32.cu" | awk '{print $1}')" = \
  "$expected_vec2_cf1_x32_sha256"

python3 "$vec2_cf_x32_generator" \
  --input "$out_dir/source/nsplit_transpose_scalar.cu" \
  --output "$out_dir/source/nsplit_transpose_vec2_cf_x32.cu" \
  >"$out_dir/logs/generate_vec2_cf_x32.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_vec2_cf_x32.cu" | awk '{print $1}')" = \
  "$expected_vec2_cf_x32_sha256"

python3 "$vec2_cf_x64_generator" \
  --input "$out_dir/source/nsplit_transpose_scalar.cu" \
  --output "$out_dir/source/nsplit_transpose_vec2_cf_x64.cu" \
  >"$out_dir/logs/generate_vec2_cf_x64.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_vec2_cf_x64.cu" | awk '{print $1}')" = \
  "$expected_vec2_cf_x64_sha256"

python3 "$vec4_cf_x64_generator" \
  --input "$out_dir/source/nsplit_transpose_scalar.cu" \
  --output "$out_dir/source/nsplit_transpose_vec4_cf_x64.cu" \
  >"$out_dir/logs/generate_vec4_cf_x64.log"
test "$(sha256sum "$out_dir/source/nsplit_transpose_vec4_cf_x64.cu" | awk '{print $1}')" = \
  "$expected_vec4_cf_x64_sha256"

variants=(
  nsplit_exact
  nsplit_transpose_nostore
  nsplit_transpose_scalar
  nsplit_transpose_vec2_x32
  nsplit_transpose_vec2_cf1_x32
  nsplit_transpose_vec2_cf_x32
  nsplit_transpose_vec2_cf_x64
  nsplit_transpose_vec4_cf_x64
)
epilogue_variants=(
  nsplit_transpose_scalar
  nsplit_transpose_vec2_x32
  nsplit_transpose_vec2_cf1_x32
  nsplit_transpose_vec2_cf_x32
  nsplit_transpose_vec2_cf_x64
  nsplit_transpose_vec4_cf_x64
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
  test "$(grep -Ec '[1-9][0-9]* bytes spill (stores|loads)' \
    "$out_dir/logs/build_${variant}.log" || true)" -eq 0
done

count_op() {
  local variant=$1
  local pattern=$2
  grep -Ec "$pattern" "$out_dir/logs/kernel_ops_${variant}.txt" || true
}

test "$(count_op nsplit_exact '^UTCHMMA ')" -eq 8
for variant in "${variants[@]:1}"; do
  test "$(count_op "$variant" '^UTCHMMA ')" -eq 4
done
for variant in "${variants[@]}"; do
  test "$(count_op "$variant" '^UTCBAR')" -eq 1
  test "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" -eq 21
  test "$(count_op "$variant" 'SYNCS.PHASECHK')" -eq 48
done
test "$(count_op nsplit_transpose_nostore 'UTMASTG.4D')" -eq 0
test "$(count_op nsplit_transpose_nostore 'LDTM.x(32|64)')" -eq 0
for variant in nsplit_exact "${epilogue_variants[@]}"; do
  test "$(count_op "$variant" 'UTMASTG.4D')" -eq 4
done

test "$(count_op nsplit_exact 'LDTM.x64')" -eq 8
test "$(count_op nsplit_transpose_scalar 'LDTM.x64')" -eq 8
for variant in \
  nsplit_transpose_vec2_x32 \
  nsplit_transpose_vec2_cf1_x32 \
  nsplit_transpose_vec2_cf_x32; do
  test "$(count_op "$variant" 'LDTM.x32')" -eq 4
done
for variant in \
  nsplit_transpose_vec2_cf_x64 \
  nsplit_transpose_vec4_cf_x64; do
  test "$(count_op "$variant" 'LDTM.x64')" -eq 4
done

test "$(count_op nsplit_transpose_scalar '^ST.E ')" -eq 512
test "$(count_op nsplit_transpose_vec2_x32 '^ST.E.64')" -eq 64
test "$(count_op nsplit_transpose_vec2_cf1_x32 '^ST.E.64')" -eq 64
test "$(count_op nsplit_transpose_vec2_cf_x32 '^ST.E.64')" -eq 64
test "$(count_op nsplit_transpose_vec2_cf_x64 '^STS.64')" -eq 128
test "$(count_op nsplit_transpose_vec4_cf_x64 '^STS.128')" -eq 64
test "$(count_op nsplit_transpose_vec2_x32 '^SHFL')" -eq 64
test "$(count_op nsplit_transpose_vec2_cf1_x32 '^SHFL')" -eq 64
test "$(count_op nsplit_transpose_vec2_cf_x32 '^SHFL')" -eq 128
test "$(count_op nsplit_transpose_vec2_cf_x64 '^SHFL')" -eq 256
test "$(count_op nsplit_transpose_vec4_cf_x64 '^SHFL')" -eq 256

{
  printf '%s\n' \
    'variant	ops	reg	utchmma	utcbar	tma_load	tma_store	tmem_x32	tmem_x64	shfl	st32	st64	st128'
  for variant in "${variants[@]}"; do
    reg=$(sed -nE 's/.*REG:([0-9]+).*/\1/p' \
      "$out_dir/logs/kernel_resource_${variant}.txt")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(wc -l < "$out_dir/logs/kernel_ops_${variant}.txt")" \
      "$reg" \
      "$(count_op "$variant" '^UTCHMMA ')" \
      "$(count_op "$variant" '^UTCBAR')" \
      "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" \
      "$(count_op "$variant" 'UTMASTG.4D')" \
      "$(count_op "$variant" 'LDTM.x32')" \
      "$(count_op "$variant" 'LDTM.x64')" \
      "$(count_op "$variant" '^SHFL')" \
      "$(count_op "$variant" '^(ST.E|STS) ')" \
      "$(count_op "$variant" '^(ST.E|STS).64')" \
      "$(count_op "$variant" '^(ST.E|STS).128')"
  done
} >"$out_dir/codegen_counts.tsv"

{
  printf 'variant\tsource_sha256\tbinary_sha256\tsass_sha256\tkernel_ops_sha256\n'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(sha256sum "$out_dir/source/${variant}.cu" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/bin/$variant" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/sass_${variant}.txt" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/kernel_ops_${variant}.txt" | awk '{print $1}')"
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
for variant in "${epilogue_variants[@]}"; do
  for specification in "256 pattern" "512 pattern" "512 ones"; do
    read -r size pattern <<<"$specification"
    timeout -k 10s 180s "$out_dir/bin/$variant" \
      --validate --validate-size "$size" --validate-pattern "$pattern" \
      >"$out_dir/logs/validate_${variant}_${pattern}${size}.log" 2>&1
  done
done

sequence_path="$out_dir/sequence.tsv"
printf 'input\tpass\torder\n' >"$sequence_path"
random_order=("${variants[@]}")
signed_order=(
  nsplit_transpose_vec4_cf_x64
  nsplit_transpose_vec2_cf_x64
  nsplit_transpose_vec2_cf_x32
  nsplit_transpose_vec2_cf1_x32
  nsplit_transpose_vec2_x32
  nsplit_transpose_scalar
  nsplit_transpose_nostore
  nsplit_exact
)
williams_indices=(0 1 7 2 6 3 5 4)
for input_name in random random-signed8; do
  if [[ "$input_name" == random ]]; then
    base_order=("${random_order[@]}")
  else
    base_order=("${signed_order[@]}")
  fi
  for pass_idx in $(seq 0 7); do
    order=
    for position in $(seq 0 7); do
      index=$(((pass_idx + williams_indices[position]) % 8))
      order+="${base_order[$index]} "
    done
    printf '%s\t%s\t%s\n' \
      "$input_name" "$((pass_idx + 1))" "${order% }" >>"$sequence_path"
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
