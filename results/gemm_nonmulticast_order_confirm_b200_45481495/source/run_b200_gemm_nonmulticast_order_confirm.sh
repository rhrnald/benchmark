#!/usr/bin/env bash
set -euo pipefail

src_dir=${1:-/workspace/benchmark/5.GEMM}
out_dir=${2:-/workspace/gemm_nonmulticast_order_confirm}
nvcc_bin=${NVCC:-/usr/local/cuda/bin/nvcc}
definition_commit=${DEFINITION_COMMIT:?set DEFINITION_COMMIT to the checked-out definition commit}
kernel_base_commit=f669b6f
expected_kernel_sha256=ae8799c19d79db0bf9ce0f07043dc660fec210fcd94817fa974f48cf41bdabef

if [[ -e "$out_dir" ]]; then
  echo "refusing to reuse existing output directory: $out_dir" >&2
  exit 2
fi
mkdir -p "$out_dir/bin" "$out_dir/csv" "$out_dir/source"
exec > >(tee "$out_dir/driver.log") 2>&1
cd "$src_dir"

read -r actual_kernel_sha256 _ < <(sha256sum gemm256_tma_tcgen05_bench.cu)
if [[ "$actual_kernel_sha256" != "$expected_kernel_sha256" ]]; then
  echo "kernel source does not match $kernel_base_commit: $actual_kernel_sha256" >&2
  exit 3
fi

common=(
  -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a
  -DGEMM_REPEAT_INPUT=0 -DGEMM_REPEAT_A_INPUT=0 -DGEMM_REPEAT_B_INPUT=0
  -DGEMM_REPEAT_B_BROADCAST=0 -DGEMM_REPEAT_TUNING=1
  -DGEMM_DENSE_L2_TUNING=1
  -DGEMM_CTA_M=256 -DGEMM_STAGE_K=64 -DGEMM_STAGES=3
  -DGEMM_TMA_MULTICAST_A=0 -DGEMM_TMA_MULTICAST_B=0
  -DGEMM_PERSISTENT_CTA=1 -DGEMM_PERSISTENT_STATIC_SCHEDULER=0
  -DGEMM_PERSISTENT_MACRO_M=16 -DGEMM_PERSISTENT_MACRO_N=16
  -DGEMM_PERSISTENT_8K_MACRO_M=16 -DGEMM_PERSISTENT_8K_MACRO_N=16
  -DGEMM_PERSISTENT_32K_MACRO_M=16 -DGEMM_PERSISTENT_32K_MACRO_N=16
)

: > "$out_dir/compile_commands.txt"
compile_variant() {
  local name=$1
  shift
  local command=(
    "$nvcc_bin" "${common[@]}" "$@"
    gemm256_tma_tcgen05_bench.cu -lcuda -o "$out_dir/bin/$name"
  )
  printf '%q ' "${command[@]}" >> "$out_dir/compile_commands.txt"
  printf '\n' >> "$out_dir/compile_commands.txt"
  "${command[@]}"
}

# A: current default. Consecutive local tasks vary M; consecutive macros vary N.
compile_variant order_mn -DGEMM_PERSISTENT_LOCAL_M_FAST=1 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=1
# B: round-one candidate. Consecutive local tasks vary N; consecutive macros vary M.
compile_variant order_nm -DGEMM_PERSISTENT_LOCAL_M_FAST=0 \
  -DGEMM_PERSISTENT_MACRO_N_FAST=0

python3 test_persistent_l2_mapping.py > "$out_dir/mapping_test.log"
cp gemm256_tma_tcgen05_bench.cu test_persistent_l2_mapping.py \
  NON_MULTICAST_L2_EXPERIMENT_PLAN.md "$out_dir/source/"
cp ../run_b200_gemm_nonmulticast_order_confirm.sh "$out_dir/source/"
printf '%s\n' "$definition_commit" > "$out_dir/definition_commit.txt"
printf '%s\n' "$kernel_base_commit" > "$out_dir/kernel_base_commit.txt"
printf '%s\n' "$expected_kernel_sha256" \
  > "$out_dir/kernel_expected_sha256.txt"
"$nvcc_bin" --version > "$out_dir/nvcc_version.txt"

for variant in order_mn order_nm; do
  "$out_dir/bin/$variant" --validate --validate-size 512 \
    --validate-pattern pattern --persistent-ctas 148 \
    > "$out_dir/validate_${variant}.log" 2>&1
done

nvidia-smi -q > "$out_dir/nvidia_smi_before.txt"
printf 'pair\tposition\tvariant\tcsv\n' > "$out_dir/sequence.tsv"
printf 'pair,position,timestamp,uuid,temp_c,power_w,sm_clock_mhz,gpu_util_pct\n' \
  > "$out_dir/pair_telemetry.csv"
: > "$out_dir/raw.log"

# Each candidate runs first and second three times. Adjacent pairs contain no
# telemetry calls or sleeps so pair-wise drift is minimized.
pair_orders=(
  "order_mn order_nm"
  "order_nm order_mn"
  "order_nm order_mn"
  "order_mn order_nm"
  "order_mn order_nm"
  "order_nm order_mn"
)

for pair_index in "${!pair_orders[@]}"; do
  pair=$((pair_index + 1))
  read -r first second <<< "${pair_orders[$pair_index]}"
  telemetry=$(nvidia-smi -i 0 \
    --query-gpu=timestamp,uuid,temperature.gpu,power.draw,clocks.sm,utilization.gpu \
    --format=csv,noheader,nounits)
  printf '%d,before,%s\n' "$pair" "$telemetry" \
    >> "$out_dir/pair_telemetry.csv"
  position=0
  for variant in "$first" "$second"; do
    position=$((position + 1))
    csv_name="${variant}_pair${pair}_pos${position}.csv"
    printf '%d\t%d\t%s\t%s\n' "$pair" "$position" "$variant" "$csv_name" \
      >> "$out_dir/sequence.tsv"
    echo "pair=$pair position=$position variant=$variant size=16384 ctas=148" \
      | tee -a "$out_dir/raw.log"
    "$out_dir/bin/$variant" --sizes 16384 --warmup 1 --iters 5 \
      --input-init random --persistent-ctas 148 \
      --csv "$out_dir/csv/$csv_name" \
      2>&1 | tee -a "$out_dir/raw.log"
  done
done

nvidia-smi -q > "$out_dir/nvidia_smi_after.txt"
sha256sum "$out_dir"/bin/* "$out_dir"/source/* \
  "$out_dir"/csv/* "$out_dir"/validate_*.log \
  "$out_dir/compile_commands.txt" "$out_dir/definition_commit.txt" \
  "$out_dir/kernel_base_commit.txt" "$out_dir/kernel_expected_sha256.txt" \
  "$out_dir/mapping_test.log" "$out_dir/nvcc_version.txt" \
  "$out_dir/nvidia_smi_before.txt" "$out_dir/nvidia_smi_after.txt" \
  "$out_dir/pair_telemetry.csv" "$out_dir/raw.log" \
  "$out_dir/sequence.tsv" \
  > "$out_dir/sha256.txt"
