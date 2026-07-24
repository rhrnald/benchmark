#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_ws_b200}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
if [[ ! -x "$nvcc_bin" ]]; then
  nvcc_bin=/usr/local/cuda/bin/nvcc
fi
cuda_bin_dir=$(dirname "$nvcc_bin")
cuobjdump_bin="$cuda_bin_dir/cuobjdump"
test -x "$cuobjdump_bin"
runner_path=$(realpath "$0")

nsplit_source="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
e7a_source="$repo_dir/results/gemm_e7a_phase_redesign_b200_45481495_20260724_99246ad0/source/baseline.cu"
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_ws.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_ws.py"
expected_nsplit_sha256=cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca
expected_e7a_sha256=37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a

test "$(sha256sum "$nsplit_source" | awk '{print $1}')" = \
  "$expected_nsplit_sha256"
test "$(sha256sum "$e7a_source" | awk '{print $1}')" = \
  "$expected_e7a_sha256"
test -f "$generator"
test -f "$summarizer"
test ! -e "$out_dir"

mkdir -p "$out_dir"/{bin,csv,logs,source}
if git -C "$repo_dir" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$repo_dir" rev-parse HEAD >"$out_dir/definition_commit.txt"
else
  printf '%s\n' "${GEMM_DEFINITION_COMMIT:-unknown}" \
    >"$out_dir/definition_commit.txt"
fi
cp "$e7a_source" "$out_dir/source/e7a_exact.cu"
cp "$nsplit_source" "$out_dir/source/nsplit_exact.cu"
cp "$generator" "$summarizer" "$runner_path" "$out_dir/source/"
python3 "$generator" --source "$nsplit_source" \
  --variant static-control \
  --output "$out_dir/source/nsplit_static_control.cu" \
  >"$out_dir/logs/generate_nsplit_static_control.log"
python3 "$generator" --source "$nsplit_source" \
  --variant ws-b01 \
  --output "$out_dir/source/nsplit_ws_b01.cu" \
  >"$out_dir/logs/generate_nsplit_ws_b01.log"

variants=(e7a_exact nsplit_exact nsplit_static_control nsplit_ws_b01)
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
done

ordinary_static_count=$(
  grep -c '^UTCHMMA ' \
    "$out_dir/logs/kernel_ops_nsplit_static_control.txt" || true
)
ws_static_count=$(
  grep -c '^UTCHMMA.WS' \
    "$out_dir/logs/kernel_ops_nsplit_static_control.txt" || true
)
ordinary_ws_count=$(
  grep -c '^UTCHMMA ' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_count=$(
  grep -c '^UTCHMMA.WS' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_keep_count=$(
  grep -c '^UTCHMMA.WS.*B_KEEP' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_reuse_count=$(
  grep -c '^UTCHMMA.WS.*B_REUSE' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_buffer1_count=$(
  grep -c '^UTCHMMA.WS.*BUFFER1' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_keep_buffer1_count=$(
  grep -c '^UTCHMMA.WS.*B_KEEP.BUFFER1' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_reuse_buffer1_count=$(
  grep -c '^UTCHMMA.WS.*B_REUSE.BUFFER1' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
ws_other_buffer_count=$(
  grep -Ec '^UTCHMMA.WS.*BUFFER[23]' \
    "$out_dir/logs/kernel_ops_nsplit_ws_b01.txt" || true
)
test "$ordinary_static_count" -eq 16
test "$ws_static_count" -eq 0
test "$ordinary_ws_count" -eq 0
test "$ws_count" -eq 16
test "$ws_keep_count" -eq 8
test "$ws_reuse_count" -eq 8
test "$ws_buffer1_count" -eq 8
test "$ws_keep_buffer1_count" -eq 4
test "$ws_reuse_buffer1_count" -eq 4
test "$ws_other_buffer_count" -eq 0
{
  printf '%s\n' \
    'variant	ordinary_utchmma	utchmma_ws	b_keep	b_reuse	buffer1	other_buffer'
  printf 'nsplit_static_control\t%s\t%s\t0\t0\t0\t0\n' \
    "$ordinary_static_count" "$ws_static_count"
  printf 'nsplit_ws_b01\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ordinary_ws_count" "$ws_count" "$ws_keep_count" "$ws_reuse_count" \
    "$ws_buffer1_count" "$ws_other_buffer_count"
} >"$out_dir/opcode_counts.tsv"
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

{
  date -Is
  uname -a
  "$nvcc_bin" --version
  nvidia-smi -L
  nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,power.limit,clocks.max.sm,memory.total \
    --format=csv,noheader
} >"$out_dir/logs/environment_before.txt"

for variant in "${variants[@]}"; do
  for pattern in pattern ones; do
    timeout -k 10s 180s "$out_dir/bin/$variant" \
      --validate --validate-size 512 --validate-pattern "$pattern" \
      >"$out_dir/logs/validate_${variant}_${pattern}.log" 2>&1
  done
done

cat >"$out_dir/sequence.tsv" <<'EOF'
input	pass	order
random	1	e7a_exact nsplit_exact nsplit_static_control nsplit_ws_b01
random	2	nsplit_ws_b01 e7a_exact nsplit_exact nsplit_static_control
random	3	nsplit_static_control nsplit_ws_b01 e7a_exact nsplit_exact
random	4	nsplit_exact nsplit_static_control nsplit_ws_b01 e7a_exact
random-signed8	1	nsplit_ws_b01 nsplit_static_control nsplit_exact e7a_exact
random-signed8	2	e7a_exact nsplit_ws_b01 nsplit_static_control nsplit_exact
random-signed8	3	nsplit_exact e7a_exact nsplit_ws_b01 nsplit_static_control
random-signed8	4	nsplit_static_control nsplit_exact e7a_exact nsplit_ws_b01
EOF

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
done < <(tail -n +2 "$out_dir/sequence.tsv")

python3 "$summarizer" "$out_dir" | tee "$out_dir/logs/summarize.log"

{
  date -Is
  nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,power.limit,clocks.sm,memory.total \
    --format=csv,noheader
} >"$out_dir/logs/environment_after.txt"

(
  cd "$out_dir"
  find . -type f ! -name SHA256SUMS -print0 | sort -z |
    xargs -0 sha256sum >SHA256SUMS
)

echo "completed result_dir=$out_dir"
