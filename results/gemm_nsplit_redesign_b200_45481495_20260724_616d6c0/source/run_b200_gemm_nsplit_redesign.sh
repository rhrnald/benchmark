#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_nsplit_redesign_b200}
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
generator="$repo_dir/5.GEMM/generate_gemm_nsplit_redesign.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_nsplit_redesign.py"
expected_nsplit_sha256=cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca
expected_e7a_sha256=37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a

test -f "$nsplit_source"
test -f "$e7a_source"
test -f "$generator"
test -f "$summarizer"
test "$(sha256sum "$nsplit_source" | awk '{print $1}')" = \
  "$expected_nsplit_sha256"
test "$(sha256sum "$e7a_source" | awk '{print $1}')" = \
  "$expected_e7a_sha256"
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
  --variant nsplit_consumer_u1 \
  --output "$out_dir/source/nsplit_consumer_u1.cu" \
  >"$out_dir/logs/generate_nsplit_consumer_u1.log"
python3 "$generator" --source "$nsplit_source" \
  --variant nsplit_u1_suspend_prod \
  --output "$out_dir/source/nsplit_u1_suspend_prod.cu" \
  >"$out_dir/logs/generate_nsplit_u1_suspend_prod.log"

variants=(
  e7a_exact
  nsplit_exact
  nsplit_consumer_u1
  nsplit_u1_suspend_prod
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
done

if ! cmp -s "$out_dir/logs/kernel_ops_nsplit_exact.txt" \
  "$out_dir/logs/kernel_ops_nsplit_consumer_u1.txt"; then
  echo "consumer_u1 main-kernel SASS differs; timed protocol must be revised" \
    >&2
  exit 1
fi
printf '%s\n' \
  "nsplit_exact and nsplit_consumer_u1 normalized main-kernel SASS: identical" \
  >"$out_dir/logs/consumer_u1_codegen_equivalence.txt"

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
random	1	e7a_exact nsplit_exact nsplit_u1_suspend_prod
random	2	nsplit_u1_suspend_prod e7a_exact nsplit_exact
random	3	nsplit_exact nsplit_u1_suspend_prod e7a_exact
random-signed8	1	nsplit_u1_suspend_prod nsplit_exact e7a_exact
random-signed8	2	e7a_exact nsplit_u1_suspend_prod nsplit_exact
random-signed8	3	nsplit_exact e7a_exact nsplit_u1_suspend_prod
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
}

run_case random 1 e7a_exact
run_case random 1 nsplit_exact
run_case random 1 nsplit_u1_suspend_prod
run_case random 2 nsplit_u1_suspend_prod
run_case random 2 e7a_exact
run_case random 2 nsplit_exact
run_case random 3 nsplit_exact
run_case random 3 nsplit_u1_suspend_prod
run_case random 3 e7a_exact

run_case random-signed8 1 nsplit_u1_suspend_prod
run_case random-signed8 1 nsplit_exact
run_case random-signed8 1 e7a_exact
run_case random-signed8 2 e7a_exact
run_case random-signed8 2 nsplit_u1_suspend_prod
run_case random-signed8 2 nsplit_exact
run_case random-signed8 3 nsplit_exact
run_case random-signed8 3 e7a_exact
run_case random-signed8 3 nsplit_u1_suspend_prod

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
