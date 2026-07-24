#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_cross_tile_prefetch_b200}
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
prefetch_generator="$repo_dir/5.GEMM/generate_gemm_nsplit_cross_tile_prefetch.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_cross_tile_prefetch.py"

expected_exact_sha256=cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca
expected_scalar_x64_sha256=a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc
expected_nonoverlap_sha256=980a9cb9a3ad82a0a8fb511a8d045436db5bccbaf93d23778725e2c5108a0658
expected_overlap_sha256=91646153f07629cb1503de6221938e6d138f05b04012d7d580433f73c51536df

test "$(sha256sum "$exact_source" | awk '{print $1}')" = \
  "$expected_exact_sha256"
for required in \
  "$transpose_generator" \
  "$prefetch_generator" \
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
  "$prefetch_generator" \
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
)" = "$expected_scalar_x64_sha256"

for mode in nonoverlap overlap; do
  variant="nsplit_prefetch_${mode}"
  python3 "$prefetch_generator" \
    --input "$out_dir/source/nsplit_transpose_scalar_x64.cu" \
    --output "$out_dir/source/${variant}.cu" \
    --mode "$mode" \
    >"$out_dir/logs/generate_${variant}.log"
done

test "$(
  sha256sum "$out_dir/source/nsplit_prefetch_nonoverlap.cu" |
    awk '{print $1}'
)" = "$expected_nonoverlap_sha256"
test "$(
  sha256sum "$out_dir/source/nsplit_prefetch_overlap.cu" |
    awk '{print $1}'
)" = "$expected_overlap_sha256"
test "$(
  sha256sum "$out_dir/source/nsplit_prefetch_nonoverlap.cu" |
    awk '{print $1}'
)" != "$(
  sha256sum "$out_dir/source/nsplit_prefetch_overlap.cu" |
    awk '{print $1}'
)"

variants=(
  nsplit_transpose_scalar_x64
  nsplit_prefetch_nonoverlap
  nsplit_prefetch_overlap
)
candidate_variants=(
  nsplit_prefetch_nonoverlap
  nsplit_prefetch_overlap
)
common_flags=(
  -std=c++17 -O3
  -gencode arch=compute_100a,code=sm_100a
  -lineinfo -Xptxas=-v
)

make_opcode_multiset() {
  local input_path=$1
  local output_path=$2
  # kernel_ops has already removed instruction addresses and encoding words.
  # Drop predicates and operands (whose register allocation and branch targets
  # may move), retain the complete opcode plus modifiers, and preserve every
  # occurrence while removing only execution order.
  awk '{
    opcode = $1 ~ /^@/ ? $2 : $1
    sub(/;$/, "", opcode)
    print opcode
  }' "$input_path" | LC_ALL=C sort >"$output_path"
}

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
  make_opcode_multiset \
    "$out_dir/logs/kernel_ops_${variant}.txt" \
    "$out_dir/logs/kernel_opcode_multiset_${variant}.txt"
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

get_resource_field() {
  local variant=$1
  local field=$2
  sed -nE "s/.*${field}:([0-9]+).*/\\1/p" \
    "$out_dir/logs/kernel_resource_${variant}.txt"
}

get_reg() {
  get_resource_field "$1" REG
}

get_static_shared() {
  get_resource_field "$1" SHARED
}

baseline=nsplit_transpose_scalar_x64
nonoverlap=nsplit_prefetch_nonoverlap
overlap=nsplit_prefetch_overlap

# The audited scalar-x64 control is byte-for-byte fixed. These are static SASS
# counts, not estimates of the dynamic loop trip counts.
test "$(wc -l <"$out_dir/logs/kernel_ops_${baseline}.txt")" -eq 2229
test "$(get_reg "$baseline")" -eq 174
test "$(get_static_shared "$baseline")" -eq 1184
test "$(count_op "$baseline" '^UTCHMMA ')" -eq 4
test "$(count_op "$baseline" '^UTCBAR')" -eq 1
test "$(count_op "$baseline" 'UTMALDG\.(2D|4D)')" -eq 21
test "$(count_op "$baseline" 'UTMASTG.4D')" -eq 4
test "$(count_op "$baseline" 'SYNCS.PHASECHK')" -eq 48
test "$(count_op "$baseline" 'LDTM.x64')" -eq 8
test "$(count_op "$baseline" 'LDTM.x(16|32)')" -eq 0
test "$(count_op "$baseline" '^ST.E ')" -eq 512
test "$(count_op "$baseline" '^SHFL')" -eq 0

# The two hash-pinned candidates execute the same work and may differ only in
# which of two common call sites performs the next-tile K0 issue. These are
# audited static SASS counts; they are not dynamic loop transaction counts.
for variant in "${candidate_variants[@]}"; do
  test "$(wc -l <"$out_dir/logs/kernel_ops_${variant}.txt")" -eq 2832
  test "$(get_reg "$variant")" -eq 194
  test "$(get_static_shared "$variant")" -eq 1184
  test "$(count_op "$variant" '^UTCHMMA ')" -eq 4
  test "$(count_op "$variant" '^UTCBAR')" -eq 1
  test "$(count_op "$variant" 'UTMALDG.2D')" -eq 8
  test "$(count_op "$variant" 'UTMALDG.4D')" -eq 16
  test "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" -eq 24
  test "$(count_op "$variant" 'UTMASTG.4D')" -eq 4
  test "$(count_op "$variant" 'SYNCS.PHASECHK')" -eq 54
  test "$(count_op "$variant" 'LDTM.x64')" -eq 8
  test "$(count_op "$variant" 'LDTM.x(16|32)')" -eq 0
  test "$(count_op "$variant" '^ST.E ')" -eq 512
  test "$(count_op "$variant" '^SHFL')" -eq 0
  test "$(count_op "$variant" 'CALL.REL.NOINC')" -eq 5
done

test "$(wc -l <"$out_dir/logs/kernel_ops_${nonoverlap}.txt")" -eq \
  "$(wc -l <"$out_dir/logs/kernel_ops_${overlap}.txt")"
test "$(get_reg "$nonoverlap")" -eq "$(get_reg "$overlap")"
test "$(get_static_shared "$nonoverlap")" -eq \
  "$(get_static_shared "$overlap")"
if ! cmp -s \
  "$out_dir/logs/kernel_resource_${nonoverlap}.txt" \
  "$out_dir/logs/kernel_resource_${overlap}.txt"; then
  diff -u \
    "$out_dir/logs/kernel_resource_${nonoverlap}.txt" \
    "$out_dir/logs/kernel_resource_${overlap}.txt" \
    >"$out_dir/logs/prefetch_resource_diff.txt" || true
  echo "B/C kernel resources differ" >&2
  exit 1
fi
if ! cmp -s \
  "$out_dir/logs/kernel_opcode_multiset_${nonoverlap}.txt" \
  "$out_dir/logs/kernel_opcode_multiset_${overlap}.txt"; then
  diff -u \
    "$out_dir/logs/kernel_opcode_multiset_${nonoverlap}.txt" \
    "$out_dir/logs/kernel_opcode_multiset_${overlap}.txt" \
    >"$out_dir/logs/prefetch_opcode_multiset_diff.txt" || true
  echo "B/C opcode(+modifier) multisets differ" >&2
  exit 1
fi
test "$(count_op "$nonoverlap" 'UTMALDG\.(2D|4D)')" -eq \
  "$(count_op "$overlap" 'UTMALDG\.(2D|4D)')"
test "$(count_op "$nonoverlap" 'UTMALDG.2D')" -eq \
  "$(count_op "$overlap" 'UTMALDG.2D')"
test "$(count_op "$nonoverlap" 'UTMALDG.4D')" -eq \
  "$(count_op "$overlap" 'UTMALDG.4D')"
if cmp -s \
  "$out_dir/logs/kernel_ops_${nonoverlap}.txt" \
  "$out_dir/logs/kernel_ops_${overlap}.txt"; then
  echo "B/C ordered kernel SASS is identical; issue placement was not proven" \
    >&2
  exit 1
fi
diff -u \
  "$out_dir/logs/kernel_ops_${nonoverlap}.txt" \
  "$out_dir/logs/kernel_ops_${overlap}.txt" \
  >"$out_dir/logs/prefetch_ordered_kernel_ops_diff.txt" || true

{
  printf '%s\n' \
    'B/C kernel resources: identical' \
    'B/C opcode(+modifier) multiset: identical' \
    'B/C ordered kernel SASS: different as required by issue placement' \
    'B/C static input-TMA instruction counts: identical'
} >"$out_dir/logs/prefetch_codegen_equivalence.txt"

{
  printf '%s\n' \
    'variant	ops	reg	static_shared	utchmma	utcbar	tma_load_2d	tma_load_4d	tma_load_total	tma_store	phasecheck	tmem_x16	tmem_x32	tmem_x64	shfl	scalar_st32	call_noinc'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(wc -l <"$out_dir/logs/kernel_ops_${variant}.txt")" \
      "$(get_reg "$variant")" \
      "$(get_static_shared "$variant")" \
      "$(count_op "$variant" '^UTCHMMA ')" \
      "$(count_op "$variant" '^UTCBAR')" \
      "$(count_op "$variant" 'UTMALDG.2D')" \
      "$(count_op "$variant" 'UTMALDG.4D')" \
      "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" \
      "$(count_op "$variant" 'UTMASTG.4D')" \
      "$(count_op "$variant" 'SYNCS.PHASECHK')" \
      "$(count_op "$variant" 'LDTM.x16')" \
      "$(count_op "$variant" 'LDTM.x32')" \
      "$(count_op "$variant" 'LDTM.x64')" \
      "$(count_op "$variant" '^SHFL')" \
      "$(count_op "$variant" '^ST.E ')" \
      "$(count_op "$variant" 'CALL.REL.NOINC')"
  done
} >"$out_dir/codegen_counts.tsv"

{
  printf 'variant\tsource_sha256\tbinary_sha256\tsass_sha256\tkernel_ops_sha256\tkernel_opcode_multiset_sha256\tkernel_resource_sha256\n'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(sha256sum "$out_dir/source/${variant}.cu" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/bin/$variant" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/sass_${variant}.txt" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/kernel_ops_${variant}.txt" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/logs/kernel_opcode_multiset_${variant}.txt" | awk '{print $1}')" \
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

validate_case() {
  local variant=$1
  local size=$2
  local pattern=$3
  local log_path="$out_dir/logs/validate_${variant}_${pattern}${size}.log"
  timeout -k 10s 180s "$out_dir/bin/$variant" \
    --validate --validate-size "$size" --validate-pattern "$pattern" \
    >"$log_path" 2>&1
  grep -Eq 'status=ok .*bad=0$' "$log_path"
}

for variant in "${variants[@]}"; do
  validate_case "$variant" 256 pattern
  validate_case "$variant" 512 pattern
  validate_case "$variant" 512 ones
done

random_orders=(
  "$baseline $nonoverlap $overlap"
  "$baseline $overlap $nonoverlap"
  "$nonoverlap $baseline $overlap"
  "$nonoverlap $overlap $baseline"
  "$overlap $baseline $nonoverlap"
  "$overlap $nonoverlap $baseline"
)
signed_orders=(
  "$overlap $nonoverlap $baseline"
  "$overlap $baseline $nonoverlap"
  "$nonoverlap $overlap $baseline"
  "$nonoverlap $baseline $overlap"
  "$baseline $overlap $nonoverlap"
  "$baseline $nonoverlap $overlap"
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
