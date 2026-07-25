#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_a_locality_b200}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
if [[ ! -x "$nvcc_bin" ]]; then
  nvcc_bin=/usr/local/cuda/bin/nvcc
fi
cuobjdump_bin="$(dirname "$nvcc_bin")/cuobjdump"
test -x "$nvcc_bin"
test -x "$cuobjdump_bin"
test ! -e "$out_dir"

canonical="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_a_locality.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_a_locality.py"
expected_source_sha256=cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca

test "$(sha256sum "$canonical" | awk '{print $1}')" = "$expected_source_sha256"
test -f "$generator"
test -f "$summarizer"

mkdir -p "$out_dir"/{bin,csv,logs,source}
git -C "$repo_dir" rev-parse HEAD >"$out_dir/definition_commit.txt"
cp "$canonical" "$out_dir/source/baseline_source.cu"
cp "$generator" "$summarizer" "$0" "$out_dir/source/"

variants=(baseline nfast_16x16 nfast_8x32 nfast_4x64)
for variant in "${variants[@]}"; do
  python3 "$generator" --source "$canonical" --variant "$variant" \
    --output "$out_dir/source/${variant}.cu" \
    >"$out_dir/logs/generate_${variant}.log"
done

common_flags=(-std=c++17 -O3 -gencode arch=compute_100a,code=sm_100a -lineinfo -Xptxas=-v)

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
  awk '
    /Function .*gemm256_bf16_16k_kernel/ { take = 1; next }
    take { print; found = 1; exit }
    END { if (!found) exit 1 }
  ' "$out_dir/logs/resource_${variant}.txt" \
    >"$out_dir/logs/kernel_resource_${variant}.txt"
  grep -Eq 'REG:[0-9]+ STACK:0 .*LOCAL:0' \
    "$out_dir/logs/kernel_resource_${variant}.txt"
  test "$(
    grep -Ec '[1-9][0-9]* bytes spill (stores|loads)' \
      "$out_dir/logs/build_${variant}.log" || true
  )" -eq 0
done

{
  printf 'variant\tops\treg\tstatic_shared\tutchmma\ttma_load_total\ttma_store\tphasecheck\n'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(wc -l <"$out_dir/logs/kernel_ops_${variant}.txt")" \
      "$(get_resource_field "$variant" REG)" \
      "$(get_resource_field "$variant" SHARED)" \
      "$(count_op "$variant" '^UTCHMMA ')" \
      "$(count_op "$variant" 'UTMALDG\.(2D|4D)')" \
      "$(count_op "$variant" 'UTMASTG.4D')" \
      "$(count_op "$variant" 'SYNCS.PHASECHK')"
  done
} >"$out_dir/codegen_counts.tsv"

{
  printf 'variant\tsource_sha256\tbinary_sha256\tkernel_ops_sha256\tkernel_resource_sha256\n'
  for variant in "${variants[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$variant" \
      "$(sha256sum "$out_dir/source/${variant}.cu" | awk '{print $1}')" \
      "$(sha256sum "$out_dir/bin/$variant" | awk '{print $1}')" \
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

for variant in "${variants[@]}"; do
  for pattern in pattern ones; do
    timeout -k 10s 180s "$out_dir/bin/$variant" \
      --validate --validate-size 512 --validate-pattern "$pattern" \
      >"$out_dir/logs/validate_${variant}_${pattern}.log" 2>&1
    grep -Eq 'status=ok .*bad=0$' "$out_dir/logs/validate_${variant}_${pattern}.log"
  done
done

cat >"$out_dir/sequence.tsv" <<'EOF'
input	pass	order
random	1	baseline nfast_16x16 nfast_8x32 nfast_4x64
random	2	nfast_4x64 nfast_8x32 nfast_16x16 baseline
random	3	nfast_8x32 baseline nfast_4x64 nfast_16x16
random	4	nfast_16x16 nfast_4x64 baseline nfast_8x32
random-signed8	1	nfast_4x64 nfast_8x32 nfast_16x16 baseline
random-signed8	2	baseline nfast_16x16 nfast_8x32 nfast_4x64
random-signed8	3	nfast_16x16 nfast_4x64 baseline nfast_8x32
random-signed8	4	nfast_8x32 baseline nfast_4x64 nfast_16x16
EOF

run_order() {
  local input_name=$1
  local pass_idx=$2
  shift 2
  local variant
  for variant in "$@"; do
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
  done
}

while IFS=$'\t' read -r input_name pass_idx order; do
  run_order "$input_name" "$pass_idx" $order
done < <(tail -n +2 "$out_dir/sequence.tsv")

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
