#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-/workspace/benchmark}
out_dir=${2:-/workspace/gemm_e7a_phase_redesign_b200}
nvcc_bin=${NVCC:-/usr/local/cuda-12.9/bin/nvcc}
if [[ ! -x "$nvcc_bin" ]]; then
  nvcc_bin=/usr/local/cuda/bin/nvcc
fi
cuda_bin_dir=$(dirname "$nvcc_bin")
cuobjdump_bin="$cuda_bin_dir/cuobjdump"
test -x "$cuobjdump_bin"
runner_path=$(realpath "$0")

canonical="$repo_dir/5.GEMM/baseline/gemm256_bf16_16k.cu"
generator="$repo_dir/5.GEMM/generate_gemm_e7a_phase_redesign.py"
summarizer="$repo_dir/5.GEMM/summarize_gemm_e7a_phase_redesign.py"
expected_source_sha256=37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a

test -f "$canonical"
test -f "$generator"
test -f "$summarizer"
actual_source_sha256=$(sha256sum "$canonical" | awk '{print $1}')
test "$actual_source_sha256" = "$expected_source_sha256"
test ! -e "$out_dir"

mkdir -p "$out_dir"/{bin,csv,logs,source}
if git -C "$repo_dir" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$repo_dir" rev-parse HEAD >"$out_dir/definition_commit.txt"
else
  printf '%s\n' "${GEMM_DEFINITION_COMMIT:-unknown}" \
    >"$out_dir/definition_commit.txt"
fi
cp "$canonical" "$out_dir/source/baseline.cu"
cp "$generator" "$summarizer" "$runner_path" "$out_dir/source/"

variants=(baseline cta4 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross)
candidates=(cta4 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross)

for variant in "${candidates[@]}"; do
  python3 "$generator" --source "$canonical" --variant "$variant" \
    --output "$out_dir/source/${variant}.cu" \
    >"$out_dir/logs/generate_${variant}.log"
done

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
done

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
random	1	baseline cta4 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross
random	2	b1_cross b1_gap64 b1_gap32 b1_gap0 cta8 cta4 baseline
random	3	cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross baseline cta4
random-signed8	1	b1_cross b1_gap64 b1_gap32 b1_gap0 cta8 cta4 baseline
random-signed8	2	baseline cta4 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross
random-signed8	3	b1_gap32 b1_gap0 cta8 cta4 baseline b1_cross b1_gap64
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
  done
}

run_order random 1 baseline cta4 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross
run_order random 2 b1_cross b1_gap64 b1_gap32 b1_gap0 cta8 cta4 baseline
run_order random 3 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross baseline cta4
run_order random-signed8 1 b1_cross b1_gap64 b1_gap32 b1_gap0 cta8 cta4 baseline
run_order random-signed8 2 baseline cta4 cta8 b1_gap0 b1_gap32 b1_gap64 b1_cross
run_order random-signed8 3 b1_gap32 b1_gap0 cta8 cta4 baseline b1_cross b1_gap64

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
